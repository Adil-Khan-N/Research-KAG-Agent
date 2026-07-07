"""
A/B Test Framework for pipeline comparison.

Routes queries through both vector_only and hybrid pipelines,
logs both results side-by-side in query_logs with pipeline_variant column.

Day 19 dashboard reads query_logs and shows:
- Average faithfulness proxy by variant
- Latency distribution by variant  
- Side-by-side answer comparison for same queries
- Win rate: which variant produces better answers

This is the logging infrastructure — Day 19 builds the UI on top.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ABResult:
    """Result from one pipeline variant."""
    query:            str
    answer:           str
    confidence:       str
    latency_ms:       int
    citations_count:  int
    graph_boosted:    int
    pipeline_variant: str
    chunk_ids:        list[str]
    faithfulness:     Optional[float] = None


@dataclass
class ABComparison:
    """Side-by-side comparison of two pipeline results."""
    query:          str
    vector_result:  ABResult
    hybrid_result:  ABResult
    winner:         str   # "hybrid", "vector", "tie"
    winner_reason:  str


def run_single_variant(
    query: str,
    use_graph: bool,
    variant_name: str,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 8,
) -> ABResult:
    """
    Run one pipeline variant and return result.

    Args:
        query: user question
        use_graph: True = hybrid, False = vector-only
        variant_name: label for logging
        top_k_retrieve: chunks to retrieve
        top_k_rerank: chunks after reranking

    Returns:
        ABResult with answer and metrics
    """
    import re
    from retrieval.pipeline import run_pipeline

    start = time.time()

    try:
        result = run_pipeline(
            query=query,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            use_graph=use_graph,
            pipeline_variant=variant_name,
        )

        answer = result.answer.answer
        answer = re.sub(r'\nCITATIONS:.*', '', answer, flags=re.DOTALL)
        answer = re.sub(r'\nCONFIDENCE:.*', '', answer, flags=re.DOTALL)
        answer = answer.strip()

        elapsed = int((time.time() - start) * 1000)

        return ABResult(
            query=query,
            answer=answer,
            confidence=result.answer.confidence,
            latency_ms=elapsed,
            citations_count=len(result.answer.citations),
            graph_boosted=result.trace.graph_boosted_count,
            pipeline_variant=variant_name,
            chunk_ids=[r.chunk_id for r in result.ranked_chunks],
        )

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        logger.error(f"Variant {variant_name} failed: {e}")
        return ABResult(
            query=query,
            answer=f"Pipeline error: {e}",
            confidence="low",
            latency_ms=elapsed,
            citations_count=0,
            graph_boosted=0,
            pipeline_variant=variant_name,
            chunk_ids=[],
        )


def run_ab_comparison(
    query: str,
    engine=None,
    delay_between: float = 5.0,
) -> ABComparison:
    """
    Run both vector-only and hybrid pipelines on the same query.
    Logs both results to query_logs with pipeline_variant column.

    Args:
        query: user question
        engine: SQLAlchemy engine for logging
        delay_between: seconds between variant calls

    Returns:
        ABComparison with both results and winner
    """
    if engine is None:
        from ingestion.db import engine

    logger.info(f"A/B test: '{query[:50]}'")

    # Run vector-only first
    logger.info("  Running vector_only variant...")
    vector_result = run_single_variant(
        query=query,
        use_graph=False,
        variant_name="vector_only",
    )

    time.sleep(delay_between)

    # Run hybrid
    logger.info("  Running hybrid variant...")
    hybrid_result = run_single_variant(
        query=query,
        use_graph=True,
        variant_name="hybrid",
    )

    # Log both to query_logs
    _log_ab_results(query, vector_result, hybrid_result, engine)

    # Determine winner
    winner, reason = _determine_winner(vector_result, hybrid_result)

    return ABComparison(
        query=query,
        vector_result=vector_result,
        hybrid_result=hybrid_result,
        winner=winner,
        winner_reason=reason,
    )


def _log_ab_results(
    query: str,
    vector_result: ABResult,
    hybrid_result: ABResult,
    engine,
):
    """Log both A/B results to query_logs table."""
    from sqlalchemy import text

    for result in [vector_result, hybrid_result]:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO query_logs
                        (query, top_k, retrieved_chunk_ids,
                         answer, latency_ms, pipeline_variant)
                    VALUES
                        (:query, :top_k, :chunk_ids,
                         :answer, :latency_ms, :variant)
                """), {
                    "query":      query,
                    "top_k":      len(result.chunk_ids),
                    "chunk_ids":  result.chunk_ids,
                    "answer":     result.answer[:2000],
                    "latency_ms": result.latency_ms,
                    "variant":    result.pipeline_variant,
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log {result.pipeline_variant}: {e}")


def _determine_winner(
    vector: ABResult,
    hybrid: ABResult,
) -> tuple[str, str]:
    """
    Determine which variant is better based on observable metrics.

    Scoring:
    - Citations: more = better (more grounded)
    - Confidence: high > medium > low
    - Latency: faster = better (minor factor)
    - Graph boost: hybrid advantage indicator
    """
    vector_score = 0
    hybrid_score = 0
    reasons = []

    # Citations
    if hybrid.citations_count > vector.citations_count:
        hybrid_score += 2
        reasons.append(
            f"hybrid has more citations "
            f"({hybrid.citations_count} vs {vector.citations_count})"
        )
    elif vector.citations_count > hybrid.citations_count:
        vector_score += 2
        reasons.append(
            f"vector has more citations "
            f"({vector.citations_count} vs {hybrid.citations_count})"
        )

    # Confidence
    conf_rank = {"high": 3, "medium": 2, "low": 1}
    h_conf = conf_rank.get(hybrid.confidence, 1)
    v_conf = conf_rank.get(vector.confidence, 1)
    if h_conf > v_conf:
        hybrid_score += 1
        reasons.append(f"hybrid confidence higher ({hybrid.confidence})")
    elif v_conf > h_conf:
        vector_score += 1
        reasons.append(f"vector confidence higher ({vector.confidence})")

    # Graph boost (only hybrid can have this)
    if hybrid.graph_boosted > 0:
        hybrid_score += 1
        reasons.append(
            f"hybrid used graph ({hybrid.graph_boosted} boosted chunks)"
        )

    # Determine winner
    if hybrid_score > vector_score:
        winner = "hybrid"
    elif vector_score > hybrid_score:
        winner = "vector"
    else:
        winner = "tie"

    reason = "; ".join(reasons) if reasons else "equal performance"
    return winner, reason


def run_ab_batch(
    queries: list[str],
    engine=None,
    delay_between_queries: float = 15.0,
    delay_between_variants: float = 5.0,
) -> list[ABComparison]:
    """
    Run A/B comparison for a batch of queries.
    Saves all results to query_logs for Day 19 dashboard.

    Args:
        queries: list of questions to test
        engine: SQLAlchemy engine
        delay_between_queries: seconds between queries (rate limiting)
        delay_between_variants: seconds between vector/hybrid per query

    Returns:
        list of ABComparison
    """
    if engine is None:
        from ingestion.db import engine

    results = []
    for i, query in enumerate(queries):
        print(f"\n  [{i+1}/{len(queries)}] {query[:55]}...")
        comparison = run_ab_comparison(
            query=query,
            engine=engine,
            delay_between=delay_between_variants,
        )
        results.append(comparison)

        print(f"    Vector: {comparison.vector_result.latency_ms}ms, "
              f"conf={comparison.vector_result.confidence}, "
              f"cit={comparison.vector_result.citations_count}")
        print(f"    Hybrid: {comparison.hybrid_result.latency_ms}ms, "
              f"conf={comparison.hybrid_result.confidence}, "
              f"cit={comparison.hybrid_result.citations_count}, "
              f"boosted={comparison.hybrid_result.graph_boosted}")
        print(f"    Winner: {comparison.winner} — {comparison.winner_reason}")

        if i < len(queries) - 1:
            time.sleep(delay_between_queries)

    return results


def get_ab_summary(engine=None) -> dict:
    """
    Aggregate A/B results from query_logs table.
    Called by Day 19 dashboard.

    Returns dict with per-variant stats.
    """
    if engine is None:
        from ingestion.db import engine

    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            stats = conn.execute(text("""
                SELECT
                    pipeline_variant,
                    COUNT(*) AS query_count,
                    AVG(latency_ms)::int AS avg_latency_ms,
                    MIN(latency_ms) AS min_latency,
                    MAX(latency_ms) AS max_latency,
                    AVG(ARRAY_LENGTH(retrieved_chunk_ids, 1))::float
                        AS avg_chunks_retrieved
                FROM query_logs
                WHERE pipeline_variant IN ('hybrid', 'vector_only', 'cache_hit')
                GROUP BY pipeline_variant
                ORDER BY query_count DESC
            """)).fetchall()

            recent = conn.execute(text("""
                SELECT
                    pipeline_variant,
                    query,
                    answer,
                    latency_ms,
                    created_at
                FROM query_logs
                WHERE pipeline_variant IN ('hybrid', 'vector_only')
                ORDER BY created_at DESC
                LIMIT 20
            """)).fetchall()

            # Find paired queries (same query, both variants)
            paired = conn.execute(text("""
                SELECT
                    a.query,
                    a.latency_ms AS hybrid_ms,
                    b.latency_ms AS vector_ms,
                    a.answer     AS hybrid_answer,
                    b.answer     AS vector_answer
                FROM query_logs a
                JOIN query_logs b ON a.query = b.query
                WHERE a.pipeline_variant = 'hybrid'
                  AND b.pipeline_variant = 'vector_only'
                ORDER BY a.created_at DESC
                LIMIT 10
            """)).fetchall()

        return {
            "stats_by_variant": [
                {
                    "pipeline_variant":    row[0],
                    "query_count":         row[1],
                    "avg_latency_ms":      row[2],
                    "min_latency":         row[3],
                    "max_latency":         row[4],
                    "avg_chunks":          round(float(row[5] or 0), 1),
                }
                for row in stats
            ],
            "recent_queries": [
                {
                    "variant":    row[0],
                    "query":      row[1][:80],
                    "answer":     row[2][:150] if row[2] else "",
                    "latency_ms": row[3],
                    "created_at": str(row[4]),
                }
                for row in recent
            ],
            "paired_comparisons": [
                {
                    "query":          row[0][:60],
                    "hybrid_ms":      row[1],
                    "vector_ms":      row[2],
                    "hybrid_answer":  row[3][:150] if row[3] else "",
                    "vector_answer":  row[4][:150] if row[4] else "",
                }
                for row in paired
            ],
        }
    except Exception as e:
        logger.error(f"AB summary failed: {e}")
        return {"error": str(e)}