"""
FastAPI application entry point.
Run with: uvicorn api.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Hybrid KAG API...")

    # Pre-load the embedding model on startup
    # so the first query doesn't have 8s cold start
    try:
        from retrieval.search import get_model
        get_model()
        logger.info("Embedding model pre-loaded")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")

    # Pre-load graph vocab for NER
    try:
        from retrieval.query_ner import _get_graph_vocab
        _get_graph_vocab()
        logger.info("Graph vocabulary pre-loaded")
    except Exception as e:
        logger.warning(f"Could not pre-load graph vocab: {e}")

    logger.info("API ready")
    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Hybrid KAG Research Assistant",
    description="""
    A production-grade Hybrid Knowledge-Augmented Generation system
    for scientific paper research.

    ## Features
    - **Hybrid retrieval**: pgvector dense search + Neo4j graph traversal
    - **Cross-encoder reranking**: BAAI/bge-reranker-base
    - **Cited answer generation**: Gemini 1.5 Flash with inline citations
    - **Knowledge graph**: 400+ nodes, 466+ edges across 25 papers

    ## Pipeline
    `query → NER → graph traversal → vector search → rerank → generate`
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Streamlit frontend (Day 13)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(router, prefix="")

@app.get("/")
async def root():
    return {
        "message": "Hybrid KAG Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /query",
            "POST /ingest",
            "GET /graph-explore",
            "GET /papers",
            "GET /stats",
            "GET /health",
            "GET /query-logs",
        ]
    }