"""
Re-ingest all papers with adaptive chunking into parallel DB tables.

Creates:
- chunks_adaptive table (same schema as chunks but with adaptive chunks)
- Embeds adaptive chunks with same model
- Adds chunking_strategy column for comparison

This lets us query both strategies side-by-side without losing
the original fixed-size chunks.
"""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ingestion.db import engine
from ingestion.adaptive_chunker import adaptive_chunk_paper
from ingestion.embedder import load_model
from sqlalchemy import text
import numpy as np


def create_adaptive_chunks_table():
    """Create chunks_adaptive table — parallel to chunks."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chunks_adaptive (
                id              SERIAL PRIMARY KEY,
                chunk_id        VARCHAR(60) UNIQUE NOT NULL,
                arxiv_id        VARCHAR(20) NOT NULL
                                REFERENCES papers(arxiv_id)
                                ON DELETE CASCADE,
                chunk_index     INTEGER NOT NULL,
                section         TEXT,
                text            TEXT NOT NULL,
                token_count     INTEGER,
                target_tokens   INTEGER,
                chunking_strategy VARCHAR(20) DEFAULT 'adaptive',
                embedding       VECTOR(384),
                embedded_at     TIMESTAMP DEFAULT NOW()
            )
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_adaptive_arxiv_id
            ON chunks_adaptive(arxiv_id)
        """))

        conn.commit()
    print("✓ chunks_adaptive table created")


def create_adaptive_vector_index():
    """Create IVFFlat index on adaptive chunks."""
    with engine.connect() as conn:
        # Count rows first — IVFFlat needs data
        count = conn.execute(
            text("SELECT COUNT(*) FROM chunks_adaptive "
                 "WHERE embedding IS NOT NULL")
        ).fetchone()[0]

        if count < 10:
            print(f"  Only {count} embedded rows — skipping index")
            return

        lists = max(10, min(count // 100, 100))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS
            idx_adaptive_embedding_ivfflat
            ON chunks_adaptive
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists})
        """))
        conn.commit()
    print(f"✓ IVFFlat index created (lists={lists})")


def run_adaptive_ingestion():
    """
    Full adaptive ingestion pipeline:
    1. Create chunks_adaptive table
    2. Load all processed paper JSONs
    3. Chunk with adaptive strategy
    4. Embed and insert
    5. Create vector index
    6. Print comparison stats
    """
    print("="*70)
    print("ADAPTIVE CHUNKING INGESTION")
    print("="*70)

    # Step 1: Create table
    print("\nStep 1: Creating chunks_adaptive table...")
    create_adaptive_chunks_table()

    # Step 2: Load papers
    processed_dir = Path("data/processed")
    papers = []
    for path in sorted(processed_dir.glob("*.json")):
        with open(path) as f:
            papers.append(json.load(f))
    print(f"\nStep 2: Loaded {len(papers)} papers")

    # Step 3+4: Chunk + embed
    print("\nStep 3: Adaptive chunking + embedding...")
    model = load_model()

    total_chunks = 0
    total_fixed = 0
    skipped = 0
    failed = 0

    from ingestion.chunker import chunk_paper as fixed_chunk_paper

    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]

        # Check if already ingested
        with engine.connect() as conn:
            existing = conn.execute(text("""
                SELECT COUNT(*) FROM chunks_adaptive
                WHERE arxiv_id = :arxiv_id
            """), {"arxiv_id": arxiv_id}).fetchone()[0]

        if existing > 0:
            logger.info(f"  Skipping {arxiv_id} (already ingested)")
            skipped += 1
            continue

        try:
            # Adaptive chunks
            adaptive_chunks = adaptive_chunk_paper(paper)

            # Fixed chunks for comparison
            fixed_chunks = fixed_chunk_paper(paper)
            total_fixed += len(fixed_chunks)

            if not adaptive_chunks:
                logger.warning(f"  No chunks for {arxiv_id}")
                continue

            # Embed
            texts = [c["text"] for c in adaptive_chunks]
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            # Insert
            with engine.connect() as conn:
                for chunk, emb in zip(adaptive_chunks, embeddings):
                    conn.execute(text("""
                        INSERT INTO chunks_adaptive
                            (chunk_id, arxiv_id, chunk_index,
                             section, text, token_count,
                             target_tokens, chunking_strategy,
                             embedding)
                        VALUES
                            (:chunk_id, :arxiv_id, :chunk_index,
                             :section, :text, :token_count,
                             :target_tokens, :strategy,
                             CAST(:embedding AS vector))
                        ON CONFLICT (chunk_id) DO NOTHING
                    """), {
                        "chunk_id":     chunk["chunk_id"],
                        "arxiv_id":     arxiv_id,
                        "chunk_index":  chunk["chunk_index"],
                        "section":      chunk.get("section", ""),
                        "text":         chunk["text"],
                        "token_count":  chunk["token_count"],
                        "target_tokens": chunk.get("target_tokens", 300),
                        "strategy":     "adaptive",
                        "embedding":    emb.tolist(),
                    })
                conn.commit()

            total_chunks += len(adaptive_chunks)
            print(
                f"  [{i+1}/{len(papers)}] {arxiv_id}: "
                f"{len(adaptive_chunks)} adaptive chunks "
                f"(vs {len(fixed_chunks)} fixed)"
            )

        except Exception as e:
            logger.error(f"  Failed {arxiv_id}: {e}")
            failed += 1

    # Step 5: Vector index
    print("\nStep 4: Creating vector index...")
    create_adaptive_vector_index()

    # Step 6: Stats
    print("\n" + "="*70)
    print("ADAPTIVE INGESTION COMPLETE")
    print("="*70)
    print(f"  Papers processed: {len(papers) - skipped - failed}")
    print(f"  Papers skipped:   {skipped}")
    print(f"  Papers failed:    {failed}")
    print(f"  Adaptive chunks:  {total_chunks}")
    print(f"  Fixed chunks:     {total_fixed}")

    if total_fixed > 0:
        ratio = total_chunks / total_fixed
        print(f"  Chunk ratio:      {ratio:.2f}x "
              f"({'more' if ratio > 1 else 'fewer'} adaptive chunks)")

    # Section breakdown
    with engine.connect() as conn:
        section_stats = conn.execute(text("""
            SELECT section,
                   COUNT(*) as chunk_count,
                   AVG(token_count)::int as avg_tokens,
                   AVG(target_tokens)::int as target
            FROM chunks_adaptive
            GROUP BY section
            ORDER BY chunk_count DESC
            LIMIT 15
        """)).fetchall()

    print("\n  Adaptive chunk sizes by section:")
    print(f"  {'Section':<30} {'Count':>6} {'Avg':>6} {'Target':>8}")
    print(f"  {'-'*55}")
    for row in section_stats:
        print(
            f"  {str(row[0])[:30]:<30} "
            f"{row[1]:>6} "
            f"{row[2]:>6} "
            f"{row[3]:>8}"
        )

    print("="*70)


if __name__ == "__main__":
    run_adaptive_ingestion()