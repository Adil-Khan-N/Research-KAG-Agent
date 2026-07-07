"""
PaperSearchAgent — runs each sub-query through the hybrid retriever.

Input:  sub_queries (list of queries from PlannerAgent)
Output: retrieved_papers, retrieved_chunks
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def paper_search_agent(state: dict) -> dict:
    """
    Run each sub-query through the hybrid retriever.
    Deduplicates papers across queries.
    Updates state with retrieved_papers and retrieved_chunks.
    """
    from retrieval.hybrid_retriever import HybridKAGRetriever
    from retrieval.reranker import rerank

    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {
            **state,
            "retrieved_papers": [],
            "retrieved_chunks": [],
            "total_papers_found": 0,
        }

    print(f"\n[PaperSearchAgent] Running {len(sub_queries)} queries...")
    retriever = HybridKAGRetriever()

    # Collect unique papers and their best chunks
    paper_chunks = defaultdict(list)  # arxiv_id → list of chunks
    paper_meta = {}                    # arxiv_id → metadata

    for i, query in enumerate(sub_queries):
        print(f"  [{i+1}/{len(sub_queries)}] '{query[:50]}'...")
        try:
            results, trace = retriever.retrieve(
                query, top_k=15, use_graph=True
            )
            ranked = rerank(query, results, top_k=5)

            for chunk in ranked:
                arxiv_id = chunk.arxiv_id
                paper_chunks[arxiv_id].append({
                    "chunk_id":      chunk.chunk_id,
                    "arxiv_id":      arxiv_id,
                    "text":          chunk.text,
                    "section":       chunk.section,
                    "rerank_score":  chunk.rerank_score,
                    "query":         query,
                })
                if arxiv_id not in paper_meta:
                    paper_meta[arxiv_id] = {
                        "arxiv_id": arxiv_id,
                        "title":    chunk.title,
                        "year":     chunk.year,
                    }

            print(f"     Found {len(ranked)} chunks from "
                  f"{len(set(c.arxiv_id for c in ranked))} papers")

        except Exception as e:
            logger.error(f"Query failed '{query[:40]}': {e}")
            state.setdefault("errors", []).append(
                f"PaperSearchAgent query '{query[:30]}': {e}"
            )

    # Build output lists
    retrieved_papers = list(paper_meta.values())

    # Sort papers by year
    retrieved_papers.sort(key=lambda p: p.get("year", 0))

    # Flatten chunks — best chunks per paper (top 3)
    retrieved_chunks = []
    for arxiv_id, chunks in paper_chunks.items():
        # Sort by rerank score, take top 3
        top_chunks = sorted(
            chunks,
            key=lambda c: c["rerank_score"],
            reverse=True
        )[:3]
        retrieved_chunks.extend(top_chunks)

    print(f"\n[PaperSearchAgent] Found {len(retrieved_papers)} unique papers")
    for p in retrieved_papers:
        n_chunks = len(paper_chunks[p["arxiv_id"]])
        print(f"  [{p['year']}] {p['title'][:55]} "
              f"({n_chunks} chunks)")

    return {
        **state,
        "retrieved_papers":  retrieved_papers,
        "retrieved_chunks":  retrieved_chunks,
        "total_papers_found": len(retrieved_papers),
        "errors": state.get("errors", []),
    }