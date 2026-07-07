"""
Vector search over the chunks_adaptive table.
Same interface as retrieval/search.py but uses adaptive chunks.
"""

import logging
import time

from retrieval.search import SearchResult, embed_query
from sqlalchemy import text

logger = logging.getLogger(__name__)


def search_adaptive(
    query: str,
    top_k: int = 5,
    engine=None,
) -> list[SearchResult]:
    """
    Search adaptive chunks table.
    Same interface as search() but queries chunks_adaptive.
    """
    if engine is None:
        from ingestion.db import engine

    start = time.time()
    query_embedding = embed_query(query)

    sql = text("""
        SELECT
            c.chunk_id,
            c.arxiv_id,
            p.title,
            p.year,
            c.section,
            c.text,
            c.token_count,
            1 - (c.embedding <=> CAST(:embedding AS vector))
                AS similarity
        FROM chunks_adaptive c
        JOIN papers p ON c.arxiv_id = p.arxiv_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {
            "embedding": str(query_embedding),
            "top_k": top_k,
        }).fetchall()

    elapsed = int((time.time() - start) * 1000)
    logger.info(f"Adaptive search: {len(rows)} results in {elapsed}ms")

    return [
        SearchResult(
            chunk_id=row[0],
            arxiv_id=row[1],
            title=row[2],
            year=row[3],
            section=row[4],
            text=row[5],
            token_count=row[6],
            similarity=round(float(row[7]), 4),
        )
        for row in rows
    ]