"""
PostgreSQL schema creation.
Run this once to set up all tables and indexes.
Tables: papers, chunks (with vector embedding), query_logs
"""

import os
import logging
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers table: one row per paper
CREATE TABLE IF NOT EXISTS papers (
    id              SERIAL PRIMARY KEY,
    arxiv_id        VARCHAR(20) UNIQUE NOT NULL,
    title           TEXT NOT NULL,
    authors         TEXT[],
    year            INTEGER,
    abstract        TEXT,
    categories      TEXT[],
    pdf_url         TEXT,
    page_count      INTEGER,
    sections_found  TEXT[],
    total_chunks    INTEGER,
    ingested_at     TIMESTAMP DEFAULT NOW()
);

-- Chunks table: one row per chunk, with embedding
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    chunk_id        VARCHAR(50) UNIQUE NOT NULL,
    arxiv_id        VARCHAR(20) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    section         TEXT,
    text            TEXT NOT NULL,
    token_count     INTEGER,
    embedding       VECTOR(384),
    embedded_at     TIMESTAMP DEFAULT NOW()
);

-- Query logs table: every search call gets logged
CREATE TABLE IF NOT EXISTS query_logs (
    id              SERIAL PRIMARY KEY,
    query           TEXT NOT NULL,
    top_k           INTEGER,
    retrieved_chunk_ids  TEXT[],
    answer          TEXT,
    latency_ms      INTEGER,
    pipeline_variant VARCHAR(50) DEFAULT 'hybrid',
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id 
    ON chunks(arxiv_id);

CREATE INDEX IF NOT EXISTS idx_chunks_section 
    ON chunks(section);

CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id 
    ON papers(arxiv_id);

CREATE INDEX IF NOT EXISTS idx_papers_year 
    ON papers(year);
"""

IVFFLAT_INDEX_SQL = """
-- IVFFlat index for approximate nearest neighbor vector search
-- lists = 100 is good for ~10k-50k vectors
-- Run AFTER embeddings are inserted, not before
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
    ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

def create_schema(engine):
    """Create all tables. Safe to run multiple times (IF NOT EXISTS)."""
    with engine.connect() as connection:
        connection.execute(text(SCHEMA_SQL))
        connection.commit()

    logger.info("Schema created successfully.")
    print("Tables created: papers, chunks, query_logs. Extensions: pgvector.")
    print("Basic Indexes created")

def create_vector_index(engine):
    """
    Create the IVFFlat vector index.
    MUST be called AFTER embeddings are inserted.
    IVFFlat needs data to cluster — empty table = useless index.
    """

    with engine.connect() as connection:
        connection.execute(text(IVFFLAT_INDEX_SQL))
        connection.commit()

    logger.info("IVFFlat vector index created successfully.")
    print("IVFFlat index on chunks(embedding) created. Ready for fast vector search.")

def drop_all(engine):
    """Dangerous: drop all tables. Use with caution."""
    with engine.connect() as connection:
        connection.execute(text("DROP TABLE IF EXISTS query_logs CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS papers CASCADE"))
        connection.commit()

    print("All tables dropped.")

def get_table_stats(engine):
    """Print row counts for each table."""
    with engine.connect() as connection:
        for table in ["papers", "chunks", "query_logs"]:
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.fetchone()[0]
            print(f"  {table}: {count} rows")

if __name__ == "__main__":
    from ingestion.db import engine

    print("setting up database schema")
    create_schema(engine)

    print("\nCurrent table stats:")
    get_table_stats(engine)