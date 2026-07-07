"""
Semantic Query Cache.

How it works:
1. Incoming query is embedded (384-dim vector)
2. Check Postgres cache table for similar past queries
3. If cosine similarity > threshold → return cached answer instantly
4. If miss → run full pipeline, store result in cache

Why semantic (not exact string match):
- "how does ViT work?" and "explain ViT architecture" are the same question
- Exact match would miss this, semantic cache catches it
- Threshold 0.92 is tight enough to avoid false positives

CV number to track:
- Cache hit rate over time
- Latency: cold ~340ms vs hot ~8ms
"""

import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Similarity threshold for cache hit
# 0.92 = very similar (same question rephrased)
# 0.85 = somewhat similar (same topic, different angle)
SIMILARITY_THRESHOLD = 0.92

# Max cache entries to keep
MAX_CACHE_SIZE = 500


@dataclass
class CacheResult:
    """Result from cache lookup."""
    hit: bool
    answer: str = ""
    similarity: float = 0.0
    original_query: str = ""
    latency_ms: int = 0
    cached_at: str = ""


def _ensure_cache_table(engine):
    """Create query_cache table if it doesn't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id              SERIAL PRIMARY KEY,
                query           TEXT NOT NULL,
                query_embedding VECTOR(384) NOT NULL,
                answer          TEXT NOT NULL,
                citations       JSONB,
                pipeline_variant VARCHAR(50),
                hit_count       INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT NOW(),
                last_hit_at     TIMESTAMP DEFAULT NOW()
            )
        """))

        # Index for fast vector search on cache
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_cache_embedding
            ON query_cache
            USING ivfflat (query_embedding vector_cosine_ops)
            WITH (lists = 10)
        """))
        conn.commit()
    logger.info("Query cache table ready")


def cache_lookup(
    query: str,
    engine=None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> CacheResult:
    """
    Look up a query in the semantic cache.

    Args:
        query: incoming user query
        engine: SQLAlchemy engine
        threshold: minimum cosine similarity for cache hit

    Returns:
        CacheResult with hit=True if found, hit=False if miss
    """
    if engine is None:
        from ingestion.db import engine

    from retrieval.search import embed_query

    start = time.time()

    try:
        query_embedding = embed_query(query)

        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT
                    id,
                    query,
                    answer,
                    citations,
                    1 - (query_embedding <=> CAST(:embedding AS vector))
                        AS similarity,
                    created_at
                FROM query_cache
                WHERE 1 - (query_embedding <=> CAST(:embedding AS vector))
                      >= :threshold
                ORDER BY query_embedding <=> CAST(:embedding AS vector)
                LIMIT 1
            """), {
                "embedding": str(query_embedding),
                "threshold": threshold,
            }).fetchone()

            if row:
                # Update hit count
                conn.execute(text("""
                    UPDATE query_cache
                    SET hit_count = hit_count + 1,
                        last_hit_at = NOW()
                    WHERE id = :id
                """), {"id": row[0]})
                conn.commit()

                elapsed = int((time.time() - start) * 1000)
                similarity = round(float(row[4]), 4)

                logger.info(
                    f"Cache HIT: similarity={similarity:.4f} "
                    f"in {elapsed}ms | original='{row[1][:40]}'"
                )

                return CacheResult(
                    hit=True,
                    answer=row[2],
                    similarity=similarity,
                    original_query=row[1],
                    latency_ms=elapsed,
                    cached_at=str(row[5]),
                )

        elapsed = int((time.time() - start) * 1000)
        logger.info(f"Cache MISS in {elapsed}ms for '{query[:40]}'")
        return CacheResult(hit=False, latency_ms=elapsed)

    except Exception as e:
        logger.error(f"Cache lookup failed: {e}")
        return CacheResult(hit=False, latency_ms=0)


def cache_store(
    query: str,
    answer: str,
    citations: list = None,
    pipeline_variant: str = "hybrid",
    engine=None,
) -> bool:
    """
    Store a query+answer in the cache.

    Args:
        query: the user's question
        answer: the generated answer
        citations: list of citation dicts
        pipeline_variant: "hybrid" or "vector_only"
        engine: SQLAlchemy engine

    Returns:
        True if stored successfully
    """
    if engine is None:
        from ingestion.db import engine

    from retrieval.search import embed_query

    try:
        query_embedding = embed_query(query)
        citations_json = json.dumps(citations or [])

        with engine.connect() as conn:
            # Check if query already cached (exact match)
            existing = conn.execute(text("""
                SELECT id FROM query_cache
                WHERE query = :query
                LIMIT 1
            """), {"query": query}).fetchone()

            if existing:
                # Update existing entry
                conn.execute(text("""
                    UPDATE query_cache
                    SET answer = :answer,
                        citations = CAST(:citations AS jsonb),
                        last_hit_at = NOW()
                    WHERE id = :id
                """), {
                    "answer": answer,
                    "citations": citations_json,
                    "id": existing[0],
                })
            else:
                # Insert new entry
                conn.execute(text("""
                    INSERT INTO query_cache
                        (query, query_embedding, answer,
                        citations, pipeline_variant)
                    VALUES
                        (:query, CAST(:embedding AS vector), :answer,
                        CAST(:citations AS jsonb), :variant)
                """), {
                    "query": query,
                    "embedding": str(query_embedding),
                    "answer": answer[:5000],
                    "citations": citations_json,
                    "variant": pipeline_variant,
                })

            conn.commit()

        logger.info(f"Cached query: '{query[:50]}'")
        return True

    except Exception as e:
        logger.error(f"Cache store failed: {e}")
        return False


def get_cache_stats(engine=None) -> dict:
    """Get cache statistics for monitoring."""
    if engine is None:
        from ingestion.db import engine

    try:
        with engine.connect() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM query_cache")
            ).fetchone()[0]

            total_hits = conn.execute(
                text("SELECT COALESCE(SUM(hit_count), 0) FROM query_cache")
            ).fetchone()[0]

            top_queries = conn.execute(text("""
                SELECT query, hit_count, last_hit_at
                FROM query_cache
                ORDER BY hit_count DESC
                LIMIT 5
            """)).fetchall()

        return {
            "total_cached": total,
            "total_hits": total_hits,
            "top_queries": [
                {
                    "query": row[0][:60],
                    "hits": row[1],
                    "last_hit": str(row[2]),
                }
                for row in top_queries
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def clear_cache(engine=None):
    """Clear all cache entries. Dev use only."""
    if engine is None:
        from ingestion.db import engine

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM query_cache"))
        conn.commit()
    logger.info("Cache cleared")