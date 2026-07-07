"""
HybridKAGRetriever — the core of the entire project.

Combines:
- Graph traversal (Neo4j) for structural entity relationships
- Vector search (pgvector) for semantic similarity
- Score fusion with graph-guided boosting

Pipeline:
  query → NER → graph traversal → vector search (2 passes) → merge → return
"""

import logging
import time
from dataclasses import dataclass, field

from retrieval.search import search, SearchResult, embed_query
from retrieval.query_ner import extract_entities_from_query
from graph.graph_queries import expand_neighbors
from graph.neo4j_client import get_neo4j_client
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Score boost applied to chunks from graph-identified papers
GRAPH_BOOST = 0.3

# How many results to fetch in each vector search pass
VECTOR_FETCH_K = 20

# Final output size before reranking (Day 9 cuts this to 8)
OUTPUT_K = 20

@dataclass
class HybridResult:
    """
    One result from the hybrid retriever.
    Extends SearchResult with graph provenance information.
    """
    chunk_id: str
    arxiv_id: str
    title: str
    year: int
    section: str
    text: str
    token_count: int
    vector_score: float        # raw cosine similarity from pgvector
    graph_score: float         # graph traversal relevance score (0 if not graph-found)
    final_score: float         # vector_score + graph_boost (if graph-found)
    source: str                # "vector_only", "graph_boosted", "graph_only"
    matched_entities: list     # which graph entities led to this result
    retrieval_path: str        # human-readable explanation of why retrieved

@dataclass
class RetrievalTrace:
    """
    Full trace of a retrieval run — logged for debugging and Day 15's
    retrieval explanation feature.
    """
    query: str
    extracted_entities: dict
    graph_papers_found: list
    vector_results_count: int
    graph_boosted_count: int
    final_results_count: int
    latency_ms: int
    entity_count: int

class HybridKAGRetriever:
    """
    Hybrid Knowledge-Augmented Generation retriever.

    Usage:
        retriever = HybridKAGRetriever()
        results, trace = retriever.retrieve("how does ViT handle image patches?")
    """

    def __init__(self, engine=None, neo4j_client=None):
        from ingestion.db import engine as default_engine
        self.engine = engine or default_engine
        self.neo4j = neo4j_client or get_neo4j_client()
        logger.info("HybridKAGRetriever initialized")

    def retrieve(
        self,
        query: str,
        top_k: int = OUTPUT_K,
        use_graph: bool = True,
    ) -> tuple[list[HybridResult], RetrievalTrace]:
        """
        Main retrieval method.

        Args:
            query: natural language question
            top_k: number of results to return
            use_graph: set False to run vector-only (for A/B comparison)

        Returns:
            (results, trace) tuple
        """

        start =  time.time()
        logger.info(f"HybridKAGRetriever.retrieve: '{query[:60]}'")

        # ── Stage 1: Query NER ────────────────────────────────
        entities = extract_entities_from_query(query)
        entity_names = entities["all"]

        logger.info(
            f"  Stage 1 NER: {len(entity_names)} entities — "
            f"{entity_names[:5]}"
        )

        # ── Stage 2: Graph Traversal ──────────────────────────
        graph_papers = []
        graph_arxiv_ids = set()

        if use_graph and entity_names:
            graph_papers = expand_neighbors(
                entity_names=entity_names,
                hops=2,
                client=self.neo4j,
            )
            graph_arxiv_ids = {p["arxiv_id"] for p in graph_papers}
            logger.info(
                f"  Stage 2 Graph: {len(graph_papers)} papers from "
                f"{len(entity_names)} entities"
            )
        else:
            logger.info("  Stage 2 Graph: skipped (no entities or use_graph=False)")

        # ── Stage 3a: Standard vector search ─────────────────
        vector_results = search(
            query=query,
            top_k=VECTOR_FETCH_K,
            engine=self.engine,
        )
        logger.info(f"  Stage 3a Vector: {len(vector_results)} results")

        # ── Stage 3b: Graph-filtered vector search ────────────
        graph_vector_results = []
        if graph_arxiv_ids:
            graph_vector_results = self._search_within_papers(
                query=query,
                arxiv_ids=list(graph_arxiv_ids),
                top_k=VECTOR_FETCH_K,
            )
            logger.info(
                f"  Stage 3b Graph-filtered vector: "
                f"{len(graph_vector_results)} results"
            )
        
        # ── Stage 4: Merge + Score + Deduplicate ─────────────
        merged = self._merge_results(
            vector_results=vector_results,
            graph_vector_results=graph_vector_results,
            graph_arxiv_ids=graph_arxiv_ids,
            graph_papers=graph_papers,
            entity_names=entity_names,
        )

        # Sort by final_score descending, take top_k
        merged.sort(key=lambda x: x.final_score, reverse=True)
        final_results = merged[:top_k]

        elapsed_ms = int((time.time() - start) * 1000)

        graph_boosted = sum(1 for r in final_results if r.source == "graph_boosted")

        trace = RetrievalTrace(
            query=query,
            extracted_entities=entities,
            graph_papers_found=[p["arxiv_id"] for p in graph_papers],
            vector_results_count=len(vector_results),
            graph_boosted_count=graph_boosted,
            final_results_count=len(final_results),
            latency_ms=elapsed_ms,
            entity_count=len(entity_names),
        )

        logger.info(
            f"  Stage 4 Merge: {len(final_results)} final results "
            f"({graph_boosted} graph-boosted) in {elapsed_ms}ms"
        )

        return final_results, trace
    
    def _search_within_papers(
        self,
        query: str,
        arxiv_ids: list[str],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Vector search restricted to a specific set of papers.
        Used for graph-guided search pass.
        """

        if not arxiv_ids:
            return []

        query_embedding = embed_query(query)
        # Build placeholders for IN clause
        placeholders = ", ".join(f":id_{i}" for i in range(len(arxiv_ids)))
        params = {f"id_{i}": aid for i, aid in enumerate(arxiv_ids)}
        params["embedding"] = str(query_embedding)
        params["top_k"] = top_k

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
            WHERE c.arxiv_id IN ({placeholders})
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

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
    
    def _merge_results(
        self,
        vector_results: list[SearchResult],
        graph_vector_results: list[SearchResult],
        graph_arxiv_ids: set[str],
        graph_papers: list[dict],
        entity_names: list[str],
    ) -> list[HybridResult]:
        """
        Merge vector and graph-filtered results.

        Scoring:
        - Vector-only result: final_score = vector_score
        - Graph-boosted result: final_score = vector_score + GRAPH_BOOST
        - Deduplication: keep highest-scored version of each chunk
        """
        # Build graph paper score lookup
        graph_paper_scores = {
            p["arxiv_id"]: p.get("graph_score", 1.0)
            for p in graph_papers
        }

        # Build entity match lookup for retrieval path explanation
        graph_entity_matches = {
            p["arxiv_id"]: p.get("matched_entities", [])
            for p in graph_papers
        }

        seen_chunks: dict[str, HybridResult] = {}

        def add_result(sr: SearchResult, is_graph_result: bool):
            in_graph = sr.arxiv_id in graph_arxiv_ids
            g_score = graph_paper_scores.get(sr.arxiv_id, 0.0)

            if in_graph:
                boost = GRAPH_BOOST
                source = "graph_boosted"
                matched = graph_entity_matches.get(sr.arxiv_id, [])
                path = (
                    f"Vector sim={sr.similarity:.3f} + "
                    f"Graph boost={boost:.1f} "
                    f"(entities: {matched[:2]})"
                )
            else:
                boost = 0.0
                source = "vector_only"
                matched = []
                path = f"Vector sim={sr.similarity:.3f} (no graph match)"

            final = sr.similarity + boost

            hybrid = HybridResult(
                chunk_id=sr.chunk_id,
                arxiv_id=sr.arxiv_id,
                title=sr.title,
                year=sr.year,
                section=sr.section,
                text=sr.text,
                token_count=sr.token_count,
                vector_score=sr.similarity,
                graph_score=g_score,
                final_score=final,
                source=source,
                matched_entities=matched,
                retrieval_path=path,
            )

            # Keep highest-scored version if duplicate
            if sr.chunk_id not in seen_chunks:
                seen_chunks[sr.chunk_id] = hybrid
            else:
                if final > seen_chunks[sr.chunk_id].final_score:
                    seen_chunks[sr.chunk_id] = hybrid

        # Add all results
        for sr in vector_results:
            add_result(sr, is_graph_result=False)

        for sr in graph_vector_results:
            add_result(sr, is_graph_result=True)

        return list(seen_chunks.values())
    
def pretty_print_hybrid_results(
    query: str,
    results: list[HybridResult],
    trace: RetrievalTrace,
):
    """Print hybrid results with full provenance."""
    print(f"\n{'='*70}")
    print(f"HYBRID RETRIEVAL: {query}")
    print(f"{'='*70}")
    print(f"Entities found:   {trace.entity_count} → {trace.extracted_entities['all'][:5]}")
    print(f"Graph papers:     {len(trace.graph_papers_found)}")
    print(f"Vector results:   {trace.vector_results_count}")
    print(f"Graph-boosted:    {trace.graph_boosted_count}")
    print(f"Total results:    {trace.final_results_count}")
    print(f"Latency:          {trace.latency_ms}ms")
    print(f"\nTop {min(5, len(results))} results:")

    for i, r in enumerate(results[:5], 1):
        source_icon = "🔵" if r.source == "graph_boosted" else "⚪"
        print(f"\n  [{i}] {source_icon} Score: {r.final_score:.4f} "
            f"(vec={r.vector_score:.3f} + graph={r.final_score - r.vector_score:.1f})")
        print(f"       Paper:  {r.title[:60]}")
        print(f"       Year:   {r.year} | Section: {r.section}")
        print(f"       Source: {r.source}")
        print(f"       Why:    {r.retrieval_path}")
        print(f"       Text:   {r.text[:150]}...")
