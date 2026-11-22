"""
Cross-encoder reranker using BAAI/bge-reranker-base.

Why cross-encoder over bi-encoder for reranking:
- Bi-encoder (what we use for retrieval): embeds query and chunk
  separately, compares with cosine. Fast, approximate.
- Cross-encoder (reranker): reads query + chunk TOGETHER, outputs
  a single relevance score. Slower but much more accurate.

We run this only on top-20 retrieved chunks (not all 865),
so the extra compute is acceptable (~200ms for 20 chunks).
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Model: BAAI/bge-reranker-base — small, fast, good quality
def _get_model_path() -> str:
    env_path = os.getenv("RERANKER_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    local_candidates = [
        Path("bge-reranker"),
        Path("C:/models/bge-reranker-base"),
    ]
    for candidate in local_candidates:
        if candidate.exists() and (
            (candidate / "model.safetensors").exists()
            or (candidate / "pytorch_model.bin").exists()
        ):
            return str(candidate)
    return "BAAI/bge-reranker-base"

RERANKER_MODEL = _get_model_path()

# Lazy-loaded singletons
_tokenizer = None
_model = None

def load_reranker():
    """Load reranker model and tokenizer (once per process)."""
    global _tokenizer, _model
    if _model is None:
        model_target = _get_model_path()
        logger.info(f"Loading reranker model: {model_target}")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tokenizer = AutoTokenizer.from_pretrained(model_target)
        _model = AutoModelForSequenceClassification.from_pretrained(
            model_target
        )
        _model.eval()
        logger.info("Reranker model loaded successfully.")
    return _tokenizer, _model

@dataclass
class RankedResult:
    """A result after reranking — adds rerank_score to hybrid result."""
    chunk_id: str
    arxiv_id: str
    title: str
    year: int
    section: str
    text: str
    token_count: int
    vector_score: float
    graph_score: float
    final_score: float
    rerank_score: float       # cross-encoder score (higher = more relevant)
    rerank_rank: int          # rank after reranking (1 = best)
    source: str
    matched_entities: list
    retrieval_path: str

def rerank(
    query: str,
    results: list,
    top_k: int = 8,
    batch_size: int = 16,
) -> list[RankedResult]:
    """
    Rerank a list of HybridResult or SearchResult objects using
    a cross-encoder model.

    Args:
        query: the original search query
        results: list of HybridResult objects from the retriever
        top_k: number of results to keep after reranking
        batch_size: how many pairs to score at once

    Returns:
        list of RankedResult sorted by rerank_score descending
    """
    if not results:
        return []
    
    start = time.time()
    try:
        import torch
        tokenizer, model = load_reranker()

        pairs = [(query, r.text) for r in results]
        all_scores = []

        # Score in batches
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(**encoded)
                # logits shape: (batch_size, 1) or (batch_size, 2)
                logits = outputs.logits
                if logits.shape[-1] == 1:
                    scores = logits.squeeze(-1).tolist()
                else:
                    # Binary classification — take positive class score
                    scores = logits[:, 1].tolist()
                if not isinstance(scores, list):
                    scores = [scores]
                all_scores.extend(scores)

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            f"Reranker: scored {len(pairs)} chunks in {elapsed_ms}ms, "
            f"keeping top {top_k}"
        )

        # Attach scores to results and sort
        scored = sorted(
            zip(results, all_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build RankedResult objects
        ranked = []
        for rank, (result, score) in enumerate(scored[:top_k], 1):
            ranked.append(RankedResult(
                chunk_id=result.chunk_id,
                arxiv_id=result.arxiv_id,
                title=result.title,
                year=result.year,
                section=result.section,
                text=result.text,
                token_count=result.token_count,
                vector_score=getattr(result, "vector_score", 0.0),
                graph_score=getattr(result, "graph_score", 0.0),
                final_score=getattr(result, "final_score",
                                    getattr(result, "similarity", 0.0)),
                rerank_score=round(float(score), 4),
                rerank_rank=rank,
                source=getattr(result, "source", "vector_only"),
                matched_entities=getattr(result, "matched_entities", []),
                retrieval_path=getattr(result, "retrieval_path",
                                       f"Vector sim={getattr(result, 'similarity', 0):.3f}"),
            ))
        return ranked

    except Exception as e:
        logger.warning(f"Cross-encoder reranking unavailable ({e}). Falling back to hybrid score ranking.")
        sorted_results = sorted(
            results,
            key=lambda x: getattr(x, "final_score", getattr(x, "similarity", 0.0)),
            reverse=True,
        )[:top_k]

        fallback_ranked = []
        for rank, r in enumerate(sorted_results, 1):
            f_score = round(float(getattr(r, "final_score", getattr(r, "similarity", 0.0))), 4)
            fallback_ranked.append(RankedResult(
                chunk_id=r.chunk_id,
                arxiv_id=r.arxiv_id,
                title=r.title,
                year=r.year,
                section=r.section,
                text=r.text,
                token_count=r.token_count,
                vector_score=round(float(getattr(r, "vector_score", getattr(r, "similarity", 0.0))), 4),
                graph_score=round(float(getattr(r, "graph_score", 0.0)), 4),
                final_score=f_score,
                rerank_score=f_score,
                rerank_rank=rank,
                source=getattr(r, "source", "vector_only"),
                matched_entities=getattr(r, "matched_entities", []),
                retrieval_path=getattr(r, "retrieval_path", f"Hybrid score={f_score:.3f}"),
            ))
        return fallback_ranked

def pretty_print_reranked(query: str, results: list[RankedResult]):
    """Print reranked results with before/after rank comparison."""
    print(f"\n{'='*70}")
    print(f"RERANKED RESULTS: {query[:60]}")
    print(f"{'='*70}")
    for r in results:
        source_icon = "🔵" if r.source == "graph_boosted" else "⚪"
        print(f"\n  [{r.rerank_rank}] {source_icon} "
              f"Rerank: {r.rerank_score:.4f} | "
              f"Hybrid: {r.final_score:.4f}")
        print(f"       Paper:   {r.title[:60]}")
        print(f"       Year:    {r.year} | Section: {r.section}")
        print(f"       Source:  {r.source}")
        print(f"       Text:    {r.text[:150]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test
    from retrieval.hybrid_retriever import HybridKAGRetriever
    retriever = HybridKAGRetriever()

    query = "how does shifted window attention work in Swin Transformer"
    results, trace = retriever.retrieve(query, top_k=20)

    print(f"Before reranking: {len(results)} results")
    ranked = rerank(query, results, top_k=8)
    print(f"After reranking: {len(ranked)} results")

    pretty_print_reranked(query, ranked)