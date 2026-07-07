"""
Vector search over the chunks table using pgvector cosine distance.
This is your first working retrieval slice — naive RAG, vector-only.
"""

import logging
import time
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from sqlalchemy import text

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer = None

def get_model()->SentenceTransformer:
    """Load the embedding model (singleton)"""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        
    return _model

@dataclass
class SearchResult:
    """One search result with chunk text + paper metadata."""
    chunk_id: str
    arxiv_id: str
    title: str
    year: int
    section: str
    text: str
    token_count: int
    similarity: float

def embed_query(query: str)->list[float]:
    """Embed a query string into a 384-dim vector."""
    model = get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()

def search(
        query: str,
        top_k: int = 5,
        engine = None,
        section_filter: str = None,
        year_filter: int = None,
)->list[SearchResult]:
    """
    Search for the top-k most relevant chunks for a query.

    Args:
        query: natural language question
        top_k: number of results to return
        engine: SQLAlchemy engine (uses default if None)
        section_filter: only return chunks from this section (e.g. "methods")
        year_filter: only return chunks from papers published this year or later

    Returns:
        list of SearchResult sorted by similarity descending
    """

    if engine is None:
        from ingestion.db import engine

    start = time.time()
    query_embedding = embed_query(query)

    filters = []
    params = {
        "embedding": str(query_embedding),
        "top_k": top_k,
    }

    if section_filter:
        filters.append("c.section ILIKE :section")
        params["section"] = f"%{section_filter}%"

    if year_filter:
        filters.append("p.year >= :year")
        params["year"] = year_filter

    where_clause = ""
    if filters:
        where_clause = " AND " + " AND ".join(filters)

    sql = text(f"""
        SELECT
            c.chunk_id,
            c.arxiv_id,
            p.title,
            p.year,
            c.section,
            c.text,
            c.token_count,
            1 - (c.embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM chunks c
        JOIN papers p ON c.arxiv_id = p.arxiv_id
        WHERE c.embedding IS NOT NULL
        {where_clause}
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"Search for '{query}' returned {len(rows)} results in {elapsed_ms}ms")


    results = [
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

    return results

def log_query(query: str, results: list[SearchResult], latency_ms: int, engine=None):
    """Write a query + its results to query_logs table."""
    if engine is None:
        from ingestion.db import engine

    chunk_ids = [r.chunk_id for r in results]
    
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO query_logs (query, top_k, retrieved_chunk_ids, latency_ms)
            VALUES (:query, :top_k, :chunk_ids, :latency_ms)
        """), {
            "query": query,
            "top_k": len(results),
            "chunk_ids": chunk_ids,
            "latency_ms": latency_ms,
        })
        conn.commit()

def pretty_print_results(query: str, results: list[SearchResult]):
    """Print search results in a readable format."""
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")
    print(f"Found {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r.similarity:.4f}")
        print(f"       Paper: {r.title[:65]}")
        print(f"       Year:  {r.year} | Section: {r.section}")
        print(f"       Text:  {r.text[:180]}...")
        print()
   

if __name__ == "__main__":
    from ingestion.db import engine

    # Test queries — these are your Day 5 checkpoint queries
    test_queries = [
        "how does attention work in vision transformers",
        "what is patch embedding in ViT",
        "shifted window attention in Swin Transformer",
        "masked autoencoder pretraining for images",
        "how to train vision transformers without large datasets",
    ]

    for query in test_queries:
        start = time.time()
        results = search(query, top_k=5, engine=engine)
        latency = int((time.time() - start) * 1000)

        pretty_print_results(query, results)
        log_query(query, results, latency, engine)

    print("\nAll test queries logged to query_logs table.")