"""
Pure vector-only retriever — no graph.
Used for A/B comparison against HybridKAGRetriever.
Day 11 RAGAS evaluation compares these two directly.
"""

import time
import logging
from retrieval.search import search, SearchResult

logger = logging.getLogger(__name__)

def vector_only_retrieve(
    query: str,
    top_k: int = 20,
    engine=None,
) -> list[SearchResult]:
    """
    Pure vector search with no graph augmentation.
    Baseline for RAGAS comparison on Day 11.
    """
    start = time.time()
    results = search(query=query, top_k=top_k, engine=engine)
    elapsed = int((time.time() - start) * 1000)
    logger.info(f"VectorOnly: {len(results)} results in {elapsed}ms")
    return results

