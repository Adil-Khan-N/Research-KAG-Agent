"""
Retrieval Explanation Layer.

For every chunk used in an answer, explains WHY it was retrieved:
- Vector similarity score
- Graph path that led to it (if graph-boosted)
- Reranker score
- Which entity matched in the graph

Makes retrieval transparent — "why this evidence" panel in UI.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkExplanation:
    """
    Full provenance explanation for one retrieved chunk.
    Displayed in the UI as "why this evidence" per citation.
    """
    chunk_id: str
    arxiv_id: str
    title: str
    year: int
    section: str
    text_preview: str

    # Retrieval scores
    vector_score: float          # raw cosine similarity
    rerank_score: float          # cross-encoder score
    graph_score: float           # graph traversal score
    final_score: float           # combined score used for ranking

    # Graph provenance
    source: str                  # "graph_boosted" or "vector_only"
    matched_entities: list[str]  # entities that triggered graph match
    graph_path: str              # human-readable graph path

    # Rank info
    vector_rank: int             # rank before reranking
    rerank_rank: int             # rank after reranking
    rank_change: int             # rerank_rank - vector_rank (negative = promoted)

    # Summary
    explanation: str             # one-sentence human-readable explanation


def build_explanation(
    chunk,
    vector_rank: int = 0,
) -> ChunkExplanation:
    """
    Build a full explanation for one retrieved chunk.

    Args:
        chunk: RankedResult object from the reranker
        vector_rank: position in the pre-rerank list

    Returns:
        ChunkExplanation with all provenance info
    """
    # Extract scores
    vector_score = getattr(chunk, "vector_score", 0.0)
    rerank_score = getattr(chunk, "rerank_score", 0.0)
    graph_score  = getattr(chunk, "graph_score",  0.0)
    final_score  = getattr(chunk, "final_score",  vector_score)
    source       = getattr(chunk, "source",        "vector_only")
    matched      = getattr(chunk, "matched_entities", [])
    rerank_rank  = getattr(chunk, "rerank_rank",   0)
    retrieval_path = getattr(chunk, "retrieval_path", "")

    rank_change = rerank_rank - vector_rank

    # Build graph path description
    if source == "graph_boosted" and matched:
        entities_str = ", ".join(matched[:3])
        graph_path = (
            f"Graph traversal via entities: [{entities_str}] "
            f"→ +{0.3:.1f} score boost"
        )
    else:
        graph_path = "No graph path — retrieved by vector similarity only"

    # Build human-readable explanation
    explanation = _build_explanation_text(
        source=source,
        vector_score=vector_score,
        rerank_score=rerank_score,
        matched=matched,
        rank_change=rank_change,
        section=getattr(chunk, "section", ""),
    )

    return ChunkExplanation(
        chunk_id       = chunk.chunk_id,
        arxiv_id       = chunk.arxiv_id,
        title          = chunk.title,
        year           = chunk.year,
        section        = getattr(chunk, "section", ""),
        text_preview   = chunk.text[:200],
        vector_score   = round(vector_score, 4),
        rerank_score   = round(rerank_score, 4),
        graph_score    = round(graph_score, 4),
        final_score    = round(final_score, 4),
        source         = source,
        matched_entities = matched,
        graph_path     = graph_path,
        vector_rank    = vector_rank,
        rerank_rank    = rerank_rank,
        rank_change    = rank_change,
        explanation    = explanation,
    )


def _build_explanation_text(
    source: str,
    vector_score: float,
    rerank_score: float,
    matched: list[str],
    rank_change: int,
    section: str,
) -> str:
    """Build a one-sentence explanation of why this chunk was retrieved."""
    parts = []

    if source == "graph_boosted":
        entities_str = (
            ", ".join(f'"{e}"' for e in matched[:2])
            if matched else "graph entities"
        )
        parts.append(
            f"Retrieved via graph traversal matching {entities_str}"
        )
        parts.append(f"(vector={vector_score:.3f} + graph boost=0.3)")
    else:
        parts.append(
            f"Retrieved by vector similarity (score={vector_score:.3f})"
        )

    if rerank_score > 0:
        parts.append(f"reranker confirmed relevance ({rerank_score:.2f})")

    if rank_change < -2:
        parts.append(
            f"promoted {abs(rank_change)} positions by reranker"
        )
    elif rank_change > 2:
        parts.append(
            f"demoted {rank_change} positions by reranker"
        )

    if section and section not in ("preamble", "unknown"):
        parts.append(f"from {section} section")

    return "; ".join(parts) + "."


def build_all_explanations(
    ranked_chunks: list,
    original_results: list = None,
) -> list[ChunkExplanation]:
    """
    Build explanations for all chunks in a result set.

    Args:
        ranked_chunks: list of RankedResult from reranker
        original_results: pre-rerank results for rank_change calculation

    Returns:
        list of ChunkExplanation in same order as ranked_chunks
    """
    # Build lookup for pre-rerank positions
    pre_rank_lookup = {}
    if original_results:
        for i, r in enumerate(original_results):
            chunk_id = getattr(r, "chunk_id", "")
            if chunk_id:
                pre_rank_lookup[chunk_id] = i + 1

    explanations = []
    for chunk in ranked_chunks:
        chunk_id = getattr(chunk, "chunk_id", "")
        vector_rank = pre_rank_lookup.get(chunk_id, 0)
        exp = build_explanation(chunk, vector_rank=vector_rank)
        explanations.append(exp)

    return explanations


def format_explanations_for_api(
    explanations: list[ChunkExplanation],
) -> list[dict]:
    """Convert explanations to JSON-serializable dicts for API response."""
    return [
        {
            "chunk_id":         e.chunk_id,
            "arxiv_id":         e.arxiv_id,
            "title":            e.title[:60],
            "year":             e.year,
            "section":          e.section,
            "text_preview":     e.text_preview,
            "scores": {
                "vector":   e.vector_score,
                "rerank":   e.rerank_score,
                "graph":    e.graph_score,
                "final":    e.final_score,
            },
            "retrieval": {
                "source":           e.source,
                "matched_entities": e.matched_entities,
                "graph_path":       e.graph_path,
                "vector_rank":      e.vector_rank,
                "rerank_rank":      e.rerank_rank,
                "rank_change":      e.rank_change,
            },
            "explanation": e.explanation,
        }
        for e in explanations
    ]


def print_explanations(explanations: list[ChunkExplanation]):
    """Print explanations in a readable format for debugging."""
    print(f"\n{'─'*70}")
    print(f"RETRIEVAL EXPLANATIONS ({len(explanations)} chunks)")
    print(f"{'─'*70}")

    for i, e in enumerate(explanations, 1):
        source_icon = "🔵" if e.source == "graph_boosted" else "⚪"
        rank_arrow = (
            f"↑{abs(e.rank_change)}" if e.rank_change < 0
            else f"↓{e.rank_change}" if e.rank_change > 0
            else "="
        )

        print(f"\n[{i}] {source_icon} {e.title[:55]} ({e.year})")
        print(f"     Section:  {e.section}")
        print(f"     Scores:   vec={e.vector_score:.3f} | "
              f"rerank={e.rerank_score:.3f} | "
              f"graph={e.graph_score:.1f} | "
              f"final={e.final_score:.3f}")
        print(f"     Rank:     #{e.vector_rank}→#{e.rerank_rank} ({rank_arrow})")
        print(f"     Why:      {e.explanation}")
        if e.source == "graph_boosted":
            print(f"     Graph:    {e.graph_path}")
        print(f"     Preview:  {e.text_preview[:120]}...")