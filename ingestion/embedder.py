"""
Embedding pipeline:
- Loads processed paper JSONs from data/processed/
- Embeds each chunk using sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- Inserts papers + chunks into Postgres
- Skips already-embedded papers (safe to re-run)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

def load_model()->SentenceTransformer:
    """Load the embedding model"""
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded successfully. Output dim: {model.get_sentence_embedding_dimension()}")
    return model

def get_all_preprocessed_papers()->list[dict]:
    """Load all JSON records from data/processed/"""
    papers = []
    paths = sorted(PROCESSED_DIR.glob("*.json"))
    for path in paths:
        with open(path) as f:
            papers.append(json.load(f))

    logger.info(f"Loaded {len(papers)} preprocessed papers from {PROCESSED_DIR}")
    return papers

def paper_already_ingested(conn, arxiv_id:str)->bool:
    """Check if a paper with this arxiv_id already exists in the DB"""
    result = conn.execute(text("SELECT 1 FROM papers WHERE arxiv_id = :arxiv_id"), {"arxiv_id": arxiv_id})
    return result.fetchone() is not None

def insert_paper(conn, paper: dict):
    """Insert one paper record into the papers table."""

    conn.execute(text("""
        INSERT INTO papers 
            (arxiv_id, title, authors, year, abstract, 
             categories, pdf_url, page_count, sections_found, total_chunks)
        VALUES 
            (:arxiv_id, :title, :authors, :year, :abstract,
             :categories, :pdf_url, :page_count, :sections_found, :total_chunks)
        ON CONFLICT (arxiv_id) DO NOTHING
    """), {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "authors": paper.get("authors", []),
        "year": paper.get("year"),
        "abstract": paper.get("abstract", ""),
        "categories": paper.get("categories", []),
        "pdf_url": paper.get("pdf_url", ""),
        "page_count": paper.get("page_count", 0),
        "sections_found": paper.get("sections_found", []),
        "total_chunks": paper.get("total_chunks", 0),
    })

def insert_chunks_batch(conn, chunks: list[dict], embeddings: np.ndarray):
    """Insert a batch of chunks with their embeddings."""

    for chunk, embedding in zip(chunks, embeddings):
        conn.execute(text("""
            INSERT INTO chunks
                (chunk_id, arxiv_id, chunk_index, section, text, token_count, embedding)
            VALUES
                (:chunk_id, :arxiv_id, :chunk_index, :section, :text, :token_count, :embedding)
            ON CONFLICT (chunk_id) DO NOTHING
        """), {
            "chunk_id": chunk["chunk_id"],
            "arxiv_id": chunk["arxiv_id"],
            "chunk_index": chunk["chunk_index"],
            "section": chunk.get("section", "unknown"),
            "text": chunk["text"],
            "token_count": chunk.get("token_count", 0),
            "embedding": embedding.tolist(),
        })

def embed_and_ingest_paper(
        paper: dict,
        model: SentenceTransformer,
        engine,
        skip_existing: bool = True
)->Optional[dict]:
    """
    Embed all chunks for one paper and insert into Postgres.
    Returns stats dict or None if skipped.
    """

    arxiv_id = paper["arxiv_id"]
    chunks = paper.get("chunks", [])

    if not chunks:
        logger.warning(f"{arxiv_id}: No chunks found, skipping embedding")
        return None
    
    with engine.connect() as conn:
        if skip_existing and paper_already_ingested(conn, arxiv_id):
            logger.info(f"{arxiv_id}: Paper already ingested, skipping")
            return None
        
        insert_paper(conn, paper)
        conn.commit()

        texts = [c["text"] for c in chunks]
        all_embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(batch_embeddings)

        all_embeddings = np.vstack(all_embeddings)
        insert_chunks_batch(conn, chunks, all_embeddings)   
        conn.commit()

    return {
        "arxiv_id": arxiv_id,
        "chunks_embedded": len(chunks),
        "embedding_dim": all_embeddings.shape[1],
    }

def run_embedding_pipeline():
    """
    Embed and ingest all processed papers into Postgres.
    Safe to re-run — skips already-ingested papers.
    """

    from ingestion.db import engine
    
    papers = get_all_preprocessed_papers()
    if not papers:
        print("No preprocessed papers found. Run the chunking pipeline first.")
        return
    
    model = load_model()

    results = []

    skipped = 0
    failed = 0
    start_time = time.time()

    for paper in tqdm(papers, desc="Embedding papers"):
        try:
            stats = embed_and_ingest_paper(paper, model, engine, skip_existing=True)
            if stats:
                results.append(stats)
            else:
                skipped += 1
        except Exception as e:
            logger.error(f"Error embedding paper {paper['arxiv_id']}: {e}")
            failed += 1

    elapsed_time = time.time() - start_time
    total_chunks = sum(r["chunks_embedded"] for r in results)
    
    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"  Papers embedded:  {len(results)}")
    print(f"  Papers skipped:   {skipped}")
    print(f"  Papers failed:    {failed}")
    print(f"  Total chunks:     {total_chunks}")
    print(f"  Time elapsed:     {elapsed_time:.1f}s")
    print(f"  Avg per paper:    {elapsed_time/max(len(results),1):.1f}s")
    print("=" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_embedding_pipeline()


