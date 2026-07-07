"""
A/B Dashboard — aggregates query_logs and computes per-variant metrics.

Reads from query_logs table (populated by Day 18 A/B framework).
Computes:
- Average latency per variant
- Confidence distribution
- Citation count distribution
- Paired query side-by-side comparison
- Win rate over time
"""

import logging
from datetime import datetime
from sqlalchemy import text

logger = logging.getLogger(__name__)


def get_variant_stats(engine=None) -> dict:
    """
    Compute per-variant statistics from query_logs.
    Returns dict suitable for Streamlit charts.
    """
    if engine is None:
        from ingestion.db import engine

    try:
        with engine.connect() as conn:

            # Per-variant aggregates
            stats = conn.execute(text("""
                SELECT
                    pipeline_variant,
                    COUNT(*)                    AS query_count,
                    AVG(latency_ms)::int        AS avg_latency,
                    MIN(latency_ms)             AS min_latency,
                    MAX(latency_ms)             AS max_latency,
                    PERCENTILE_CONT(0.5)
                        WITHIN GROUP (ORDER BY latency_ms)::int
                                                AS median_latency,
                    AVG(ARRAY_LENGTH(
                        retrieved_chunk_ids, 1
                    ))::float                   AS avg_chunks
                FROM query_logs
                WHERE pipeline_variant IN (
                    'hybrid', 'vector_only',
                    'cache_hit', 'decomposed'
                )
                GROUP BY pipeline_variant
                ORDER BY query_count DESC
            """)).fetchall()

            # Queries over time (last 50)
            timeline = conn.execute(text("""
                SELECT
                    pipeline_variant,
                    latency_ms,
                    created_at,
                    ARRAY_LENGTH(retrieved_chunk_ids, 1) AS chunks
                FROM query_logs
                WHERE pipeline_variant IN ('hybrid', 'vector_only')
                ORDER BY created_at DESC
                LIMIT 50
            """)).fetchall()

            # Paired comparisons (same query, both variants)
            paired = conn.execute(text("""
                SELECT
                    a.query,
                    a.latency_ms    AS hybrid_ms,
                    b.latency_ms    AS vector_ms,
                    a.answer        AS hybrid_answer,
                    b.answer        AS vector_answer,
                    ARRAY_LENGTH(a.retrieved_chunk_ids, 1)
                                    AS hybrid_chunks,
                    ARRAY_LENGTH(b.retrieved_chunk_ids, 1)
                                    AS vector_chunks,
                    a.created_at
                FROM query_logs a
                JOIN query_logs b
                    ON a.query = b.query
                   AND ABS(EXTRACT(EPOCH FROM (a.created_at - b.created_at)))
                       < 300
                WHERE a.pipeline_variant = 'hybrid'
                  AND b.pipeline_variant = 'vector_only'
                ORDER BY a.created_at DESC
                LIMIT 20
            """)).fetchall()

            # Cache hit rate
            cache_stats = conn.execute(text("""
                SELECT
                    COUNT(*) FILTER (
                        WHERE pipeline_variant = 'cache_hit'
                    )                           AS cache_hits,
                    COUNT(*) FILTER (
                        WHERE pipeline_variant != 'cache_hit'
                    )                           AS cache_misses,
                    COUNT(*)                    AS total
                FROM query_logs
            """)).fetchone()

        # Build response
        stats_by_variant = {}
        for row in stats:
            stats_by_variant[row[0]] = {
                "query_count":   row[1],
                "avg_latency":   row[2],
                "min_latency":   row[3],
                "max_latency":   row[4],
                "median_latency": row[5],
                "avg_chunks":    round(float(row[6] or 0), 1),
            }

        timeline_data = [
            {
                "variant":    row[0],
                "latency_ms": row[1],
                "created_at": str(row[2]),
                "chunks":     row[3] or 0,
            }
            for row in timeline
        ]

        paired_data = [
            {
                "query": row[0][:80],
                "hybrid_ms": row[1],
                "vector_ms": row[2],
                "hybrid_answer": (row[3] or "")[:200],
                "vector_answer": (row[4] or "")[:200],
                "hybrid_chunks": row[5] or 0,
                "vector_chunks": row[6] or 0,
                "hybrid_faster": row[1] < row[2],
            }
            for row in paired
        ] if paired else []

        cache_hit_rate = 0.0
        if cache_stats and cache_stats[2] > 0:
            cache_hit_rate = round(
                cache_stats[0] / cache_stats[2], 3
            )

        return {
            "stats_by_variant":  stats_by_variant,
            "timeline":          timeline_data,
            "paired":            paired_data,
            "cache_hit_rate":    cache_hit_rate,
            "total_queries":     cache_stats[2] if cache_stats else 0,
        }

    except Exception as e:
        logger.error(f"get_variant_stats failed: {e}")
        return {
            "stats_by_variant": {},
            "timeline": [],
            "paired": [],
            "cache_hit_rate": 0.0,
            "total_queries": 0,
            "error": str(e),
        }


def compute_win_rate(paired_data: list[dict]) -> dict:
    """
    Compute win rate from paired comparison data.
    Winner = more citations (proxy for grounded answer).
    """
    if not paired_data:
        return {
            "hybrid": 0,
            "vector": 0,
            "tie": 0,
            "total": 0,
            "hybrid_pct": 0.0,
            "vector_pct": 0.0,
        }

    hybrid_wins = 0
    vector_wins = 0
    ties = 0

    for pair in paired_data:
        h_chunks = pair.get("hybrid_chunks", 0) or 0
        v_chunks = pair.get("vector_chunks", 0) or 0

        if h_chunks > v_chunks:
            hybrid_wins += 1
        elif v_chunks > h_chunks:
            vector_wins += 1
        else:
            ties += 1

    total = len(paired_data)
    return {
        "hybrid":  hybrid_wins,
        "vector":  vector_wins,
        "tie":     ties,
        "total":   total,
        "hybrid_pct": round(hybrid_wins / total * 100, 1),
        "vector_pct": round(vector_wins / total * 100, 1),
    }