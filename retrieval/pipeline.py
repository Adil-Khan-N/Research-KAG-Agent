"""
Full RAG pipeline: query → retrieve → rerank → generate → answer.

This is what the FastAPI /query endpoint (Day 10) will call.
Combines HybridKAGRetriever + Reranker + Generator into one call.
"""

import logging
import time
from dataclasses import dataclass

from retrieval.hybrid_retriever import HybridKAGRetriever, RetrievalTrace
from retrieval.reranker import rerank, RankedResult
from retrieval.generator import generate_answer, GeneratedAnswer
from retrieval.vector_only import vector_only_retrieve

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Full result from the end-to-end pipeline."""
    query: str
    answer: GeneratedAnswer
    ranked_chunks: list[RankedResult]
    trace: RetrievalTrace
    total_latency_ms: int
    pipeline_variant: str   # "hybrid" or "vector_only"


# Singleton retriever — loaded once
_retriever = None

def get_retriever() -> HybridKAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridKAGRetriever()
    return _retriever


def run_pipeline(
    query: str,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 8,
    use_graph: bool = True,
    pipeline_variant: str = "hybrid",
) -> PipelineResult:
    """
    Run the full pipeline end-to-end.

    Args:
        query: user question
        top_k_retrieve: how many chunks to retrieve (before reranking)
        top_k_rerank: how many chunks to keep after reranking
        use_graph: False = vector-only mode (for A/B comparison)
        pipeline_variant: label for query_logs

    Returns:
        PipelineResult with answer, citations, chunks, trace
    """
    start = time.time()
    retriever = get_retriever()

    # Stage 1 + 2: Hybrid retrieval (or vector-only)
    if use_graph:
        hybrid_results, trace = retriever.retrieve(
            query=query,
            top_k=top_k_retrieve,
            use_graph=True,
        )
    else:
        # Vector-only baseline
        vector_results = vector_only_retrieve(query, top_k=top_k_retrieve)
        hybrid_results = vector_results
        trace = RetrievalTrace(
            query=query,
            extracted_entities={},
            graph_papers_found=[],
            vector_results_count=len(vector_results),
            graph_boosted_count=0,
            final_results_count=len(vector_results),
            latency_ms=0,
            entity_count=0,
        )

    # Stage 3: Rerank
    ranked_chunks = rerank(
        query=query,
        results=hybrid_results,
        top_k=top_k_rerank,
    )

    # Stage 4: Generate
    answer = generate_answer(
        query=query,
        ranked_chunks=ranked_chunks,
        max_chunks=top_k_rerank,
    )

    total_ms = int((time.time() - start) * 1000)

    logger.info(
        f"Pipeline complete: {total_ms}ms total | "
        f"retrieve={trace.latency_ms}ms | "
        f"rerank=~200ms | "
        f"generate={answer.latency_ms}ms"
    )

    return PipelineResult(
        query=query,
        answer=answer,
        ranked_chunks=ranked_chunks,
        trace=trace,
        total_latency_ms=total_ms,
        pipeline_variant=pipeline_variant,
    )

def log_pipeline_result(result: PipelineResult, engine=None):
    """Write pipeline result to query_logs table."""
    if engine is None:
        from ingestion.db import engine

    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO query_logs
                (query, top_k, retrieved_chunk_ids,
                 answer, latency_ms, pipeline_variant)
            VALUES
                (:query, :top_k, :chunk_ids,
                 :answer, :latency_ms, :variant)
        """), {
            "query": result.query,
            "top_k": len(result.ranked_chunks),
            "chunk_ids": [r.chunk_id for r in result.ranked_chunks],
            "answer": result.answer.answer[:2000],
            "latency_ms": result.total_latency_ms,
            "variant": result.pipeline_variant,
        })
        conn.commit()