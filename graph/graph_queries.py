"""
Parameterized Cypher query library for the KAG retriever.
Every function returns a list of dicts — easy to consume downstream.

Day 8 imports:
    from graph.graph_queries import (
        expand_neighbors,
        get_papers_by_method,
        get_papers_by_dataset,
        get_citation_chain,
        find_papers_sharing_concepts,
        find_contradicting_papers,
    )
"""


import logging
from graph.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Query 1: Expand neighbors around a set of entity names
# Most general query — called first in hybrid retrieval
# ─────────────────────────────────────────────────────────────


def expand_neighbors(
    entity_names: list[str],
    hops: int = 2,
    rel_types: list[str] = None,
    client: Neo4jClient = None,
) -> list[dict]:
    """
    Given a list of entity names (methods, datasets, concepts, paper titles),
    find all Papers connected within `hops` relationship steps.

    Args:
        entity_names: list of names to search for e.g. ["ViT", "ImageNet"]
        hops: traversal depth (1 or 2 recommended, 3+ can be slow)
        rel_types: relationship types to follow. None = all types.
        client: Neo4j client (uses singleton if None)

    Returns:
        list of dicts with paper arxiv_id, title, year, relevance_score
    """

    client = client or get_neo4j_client()
    if not entity_names:
        return []

    # Build relationship type filter
    if rel_types:
        rel_filter = "|".join(rel_types)
        rel_pattern = f"[:{rel_filter}*1..{hops}]"
    else:
        rel_pattern = f"[*1..{hops}]"

    results = {}

    for name in entity_names:
        name = name.strip()
        if not name:
            continue

        rows = client.run(f"""
            MATCH (entity)
            WHERE (
                entity.name CONTAINS $name OR
                entity.title CONTAINS $name
            )
            MATCH (entity)-{rel_pattern}-(p:Paper)
            RETURN DISTINCT
                p.arxiv_id AS arxiv_id,
                p.title AS title,
                p.year AS year,
                p.abstract AS abstract
            LIMIT 20
        """, {"name": name})

        for row in rows:
            arxiv_id = row["arxiv_id"]
            if arxiv_id not in results:
                results[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "title": row["title"],
                    "year": row["year"],
                    "abstract": row.get("abstract", ""),
                    "matched_entities": [name],
                    "graph_score": 1.0,
                }
            else:
                # Paper found via multiple entity matches — boost score
                results[arxiv_id]["matched_entities"].append(name)
                results[arxiv_id]["graph_score"] += 0.5

        sorted_results = sorted(
            results.values(),
            key=lambda x: x["graph_score"],
            reverse=True
        )

        logger.info(
            f"expand_neighbors({entity_names}, hops={hops}): "
            f"{len(sorted_results)} papers found"
        )
        return sorted_results
    
# ─────────────────────────────────────────────────────────────
# Query 2: Get papers by method name
# ─────────────────────────────────────────────────────────────

def get_papers_by_method(
        method_name: str, 
        client: Neo4jClient = None,
) -> list[dict]:
    """
    Find all papers that USE a specific method.

    Example: get_papers_by_method("Multi-Head Attention")
    Returns: [ViT, DeiT, Swin, ...]
    """

    client = client or get_neo4j_client()

    rows = client.run("""
        MATCH (p:Paper)-[:USES]->(m:Method)
        WHERE m.name CONTAINS $method_name
        RETURN
            p.arxiv_id AS arxiv_id,
            p.title AS title,
            p.year AS year,
            p.abstract AS abstract,
            m.name AS matched_method
        ORDER BY p.year DESC
    """, {"method_name": method_name})

    logger.info(f"get_papers_by_method('{method_name}'): {len(rows)} papers")
    return rows


# ─────────────────────────────────────────────────────────────
# Query 3: Get papers by dataset name
# ─────────────────────────────────────────────────────────────

def get_papers_by_dataset(
    dataset_name: str,
    client: Neo4jClient = None,
) -> list[dict]:
    """
    Find all papers evaluated on a specific dataset.

    Example: get_papers_by_dataset("ImageNet")
    Returns: [ViT, DeiT, Swin, MAE, ...]
    """
    client = client or get_neo4j_client()

    rows = client.run("""
        MATCH (p:Paper)-[:EVALUATES_ON]->(d:Dataset)
        WHERE d.name CONTAINS $dataset_name
        RETURN DISTINCT
            p.arxiv_id AS arxiv_id,
            p.title AS title,
            p.year AS year,
            p.abstract AS abstract,
            d.name AS matched_dataset
        ORDER BY p.year DESC, p.arxiv_id 
    """, {"dataset_name": dataset_name})

    logger.info(f"get_papers_by_dataset('{dataset_name}'): {len(rows)} papers")
    return rows


# ─────────────────────────────────────────────────────────────
# Query 4: Follow EXTENDS chain from a seed paper
# ─────────────────────────────────────────────────────────────

def get_citation_chain(
    arxiv_id: str,
    hops: int = 3,
    direction: str = "descendants",
    client: Neo4jClient = None,
) -> list[dict]:
    """
    Follow EXTENDS relationships from a seed paper.

    Args:
        arxiv_id: starting paper
        hops: how many EXTENDS hops to follow
        direction: "descendants" (papers extending this one) or
                   "ancestors" (papers this one extends)

    Example: get_citation_chain("2010.11929", direction="descendants")
    Returns: [DeiT, Swin, MAE, ...] — all papers in ViT's family tree
    """
    client = client or get_neo4j_client()

    if direction == "descendants":
        # Papers that extend this paper (children)
        query = f"""
            MATCH (seed:Paper {{arxiv_id: $arxiv_id}})
            MATCH (p:Paper)-[:EXTENDS*1..{hops}]->(seed)
            RETURN DISTINCT
                p.arxiv_id AS arxiv_id,
                p.title AS title,
                p.year AS year,
                p.abstract AS abstract
            ORDER BY p.year ASC
        """
    else:
        # Papers this paper extends (ancestors)
        query = f"""
            MATCH (seed:Paper {{arxiv_id: $arxiv_id}})
            MATCH (seed)-[:EXTENDS*1..{hops}]->(p:Paper)
            RETURN DISTINCT
                p.arxiv_id AS arxiv_id,
                p.title AS title,
                p.year AS year,
                p.abstract AS abstract
            ORDER BY p.year ASC
        """

    rows = client.run(query, {"arxiv_id": arxiv_id})
    logger.info(
        f"get_citation_chain('{arxiv_id}', {direction}, hops={hops}): "
        f"{len(rows)} papers"
    )
    return rows


# ─────────────────────────────────────────────────────────────
# Query 5: Find papers sharing concepts with a given paper
# ─────────────────────────────────────────────────────────────

def find_papers_sharing_concepts(
    arxiv_id: str,
    min_shared: int = 1,
    client: Neo4jClient = None,
) -> list[dict]:
    """
    Find papers that share concepts with a given paper,
    ranked by number of shared concepts.

    Example: find_papers_sharing_concepts("2010.11929")
    Returns: papers most conceptually similar to ViT
    """
    client = client or get_neo4j_client()

    rows = client.run("""
        MATCH (seed:Paper {arxiv_id: $arxiv_id})-[:DISCUSSES]->(c:Concept)
        MATCH (other:Paper)-[:DISCUSSES]->(c)
        WHERE other.arxiv_id <> $arxiv_id
        WITH other, collect(c.name) AS shared_concepts,
             count(c) AS shared_count
        WHERE shared_count >= $min_shared
        RETURN
            other.arxiv_id AS arxiv_id,
            other.title AS title,
            other.year AS year,
            shared_concepts,
            shared_count
        ORDER BY shared_count DESC
        LIMIT 10
    """, {"arxiv_id": arxiv_id, "min_shared": min_shared})

    logger.info(
        f"find_papers_sharing_concepts('{arxiv_id}'): "
        f"{len(rows)} related papers"
    )
    return rows


# ─────────────────────────────────────────────────────────────
# Query 6: Find papers that might contradict each other
# ─────────────────────────────────────────────────────────────

def find_contradicting_papers(
    arxiv_id: str,
    client: Neo4jClient = None,
) -> list[dict]:
    """
    Find papers with explicit CONTRADICTS edges from/to this paper.
    Falls back to papers sharing datasets but using very different methods
    (proxy for potential contradiction).
    """
    client = client or get_neo4j_client()

    # First: explicit CONTRADICTS edges
    explicit = client.run("""
        MATCH (p1:Paper {arxiv_id: $arxiv_id})-[:CONTRADICTS]-(p2:Paper)
        RETURN
            p2.arxiv_id AS arxiv_id,
            p2.title AS title,
            p2.year AS year,
            'explicit' AS contradiction_type
    """, {"arxiv_id": arxiv_id})

    if explicit:
        logger.info(f"find_contradicting_papers: {len(explicit)} explicit contradictions")
        return explicit

    # Fallback: papers on same dataset but different task approach
    implicit = client.run("""
        MATCH (seed:Paper {arxiv_id: $arxiv_id})-[:EVALUATES_ON]->(d:Dataset)
        MATCH (other:Paper)-[:EVALUATES_ON]->(d)
        WHERE other.arxiv_id <> $arxiv_id
        MATCH (seed)-[:USES]->(m1:Method)
        MATCH (other)-[:USES]->(m2:Method)
        WHERE m1.name <> m2.name
        WITH other, d, collect(DISTINCT m1.name) AS seed_methods,
             collect(DISTINCT m2.name) AS other_methods
        RETURN
            other.arxiv_id AS arxiv_id,
            other.title AS title,
            other.year AS year,
            d.name AS shared_dataset,
            seed_methods,
            other_methods,
            'implicit_dataset_overlap' AS contradiction_type
        LIMIT 5
    """, {"arxiv_id": arxiv_id})

    logger.info(
        f"find_contradicting_papers('{arxiv_id}'): "
        f"{len(implicit)} implicit candidates"
    )
    return implicit


# ─────────────────────────────────────────────────────────────
# Query 7: Get full paper neighborhood (used in /graph-explore API)
# ─────────────────────────────────────────────────────────────

def get_paper_neighborhood(
    arxiv_id: str,
    client: Neo4jClient = None,
) -> dict:
    """
    Get everything connected to a paper — methods, datasets,
    concepts, tasks, authors, related papers.
    Used by the /graph-explore API endpoint on Day 10.
    """
    client = client or get_neo4j_client()

    methods = client.run("""
        MATCH (p:Paper {arxiv_id: $id})-[:USES]->(m:Method)
        RETURN m.name AS name
    """, {"id": arxiv_id})

    datasets = client.run("""
        MATCH (p:Paper {arxiv_id: $id})-[:EVALUATES_ON]->(d:Dataset)
        RETURN d.name AS name
    """, {"id": arxiv_id})

    concepts = client.run("""
        MATCH (p:Paper {arxiv_id: $id})-[:DISCUSSES]->(c:Concept)
        RETURN c.name AS name
    """, {"id": arxiv_id})

    tasks = client.run("""
        MATCH (p:Paper {arxiv_id: $id})-[:ADDRESSES]->(t:Task)
        RETURN t.name AS name
    """, {"id": arxiv_id})

    extends = client.run("""
        MATCH (p:Paper {arxiv_id: $id})-[:EXTENDS]->(parent:Paper)
        RETURN parent.arxiv_id AS arxiv_id, parent.title AS title
    """, {"id": arxiv_id})

    extended_by = client.run("""
        MATCH (child:Paper)-[:EXTENDS]->(p:Paper {arxiv_id: $id})
        RETURN child.arxiv_id AS arxiv_id, child.title AS title
    """, {"id": arxiv_id})

    return {
        "arxiv_id": arxiv_id,
        "methods": [r["name"] for r in methods],
        "datasets": [r["name"] for r in datasets],
        "concepts": [r["name"] for r in concepts],
        "tasks": [r["name"] for r in tasks],
        "extends": extends,
        "extended_by": extended_by,
    }


# ─────────────────────────────────────────────────────────────
# Query 8: Extract entity names from a query string
# Used by the retriever to decide what to graph-search for
# ─────────────────────────────────────────────────────────────

def get_all_entity_names(client: Neo4jClient = None) -> dict:
    """
    Return all known entity names from the graph.
    Used to build a lookup for query-time NER matching.
    """
    client = client or get_neo4j_client()

    methods = client.run("MATCH (m:Method) RETURN m.name AS name")
    datasets = client.run("MATCH (d:Dataset) RETURN d.name AS name")
    concepts = client.run("MATCH (c:Concept) RETURN c.name AS name")
    tasks = client.run("MATCH (t:Task) RETURN t.name AS name")
    papers = client.run("MATCH (p:Paper) RETURN p.title AS name, p.arxiv_id AS arxiv_id")

    return {
        "methods": [r["name"] for r in methods],
        "datasets": [r["name"] for r in datasets],
        "concepts": [r["name"] for r in concepts],
        "tasks": [r["name"] for r in tasks],
        "papers": [{"title": r["name"], "arxiv_id": r["arxiv_id"]} for r in papers],
    }