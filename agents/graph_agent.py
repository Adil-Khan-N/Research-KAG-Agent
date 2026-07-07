"""
GraphAgent — finds relationships between papers using Neo4j.

Input:  retrieved_papers
Output: graph_relationships, timeline
"""

import logging

logger = logging.getLogger(__name__)


def graph_agent(state: dict) -> dict:
    """
    Use Neo4j to find relationships between retrieved papers.
    Builds timeline and relationship map.
    Updates state with graph_relationships and timeline.
    """
    from graph.neo4j_client import get_neo4j_client
    from graph.graph_queries import get_paper_neighborhood

    retrieved_papers = state.get("retrieved_papers", [])
    if not retrieved_papers:
        return {
            **state,
            "graph_relationships": [],
            "timeline": [],
        }

    print(f"\n[GraphAgent] Analyzing {len(retrieved_papers)} papers...")
    client = get_neo4j_client()

    arxiv_ids = [p["arxiv_id"] for p in retrieved_papers]
    id_to_title = {
        p["arxiv_id"]: p["title"] for p in retrieved_papers
    }

    # Find EXTENDS relationships among retrieved papers
    relationships = []
    try:
        result = client.run("""
            MATCH (p1:Paper)-[r:EXTENDS]->(p2:Paper)
            WHERE p1.arxiv_id IN $ids AND p2.arxiv_id IN $ids
            RETURN p1.arxiv_id AS from_id,
                   p1.title    AS from_title,
                   p2.arxiv_id AS to_id,
                   p2.title    AS to_title,
                   type(r)     AS rel_type
        """, {"ids": arxiv_ids})

        for row in result:
            relationships.append({
                "paper1":       row["from_title"][:50],
                "paper1_id":    row["from_id"],
                "paper2":       row["to_title"][:50],
                "paper2_id":    row["to_id"],
                "relationship": "EXTENDS",
                "description":  (
                    f"{row['from_title'][:40]} builds upon "
                    f"{row['to_title'][:40]}"
                ),
            })

        print(f"  EXTENDS relationships: {len(relationships)}")

    except Exception as e:
        logger.error(f"EXTENDS query failed: {e}")

    # Find shared methods
    try:
        result = client.run("""
            MATCH (p1:Paper)-[:USES]->(m:Method)<-[:USES]-(p2:Paper)
            WHERE p1.arxiv_id IN $ids
              AND p2.arxiv_id IN $ids
              AND p1.arxiv_id < p2.arxiv_id
            RETURN p1.arxiv_id AS id1,
                   p1.title    AS title1,
                   p2.arxiv_id AS id2,
                   p2.title    AS title2,
                   collect(m.name)[..3] AS methods
            LIMIT 10
        """, {"ids": arxiv_ids})

        for row in result:
            relationships.append({
                "paper1":       row["title1"][:50],
                "paper1_id":    row["id1"],
                "paper2":       row["title2"][:50],
                "paper2_id":    row["id2"],
                "relationship": "SHARES_METHOD",
                "description":  (
                    f"Both use: {', '.join(row['methods'])}"
                ),
            })

        print(f"  Shared method relationships: "
              f"{len([r for r in relationships if r['relationship'] == 'SHARES_METHOD'])}")

    except Exception as e:
        logger.error(f"Shared methods query failed: {e}")

    # Build timeline
    timeline = []
    for paper in sorted(retrieved_papers, key=lambda p: p.get("year", 0)):
        arxiv_id = paper["arxiv_id"]
        try:
            neighborhood = get_paper_neighborhood(arxiv_id, client=client)
            methods = neighborhood.get("methods", [])[:2]
            datasets = neighborhood.get("datasets", [])[:2]
        except Exception:
            methods = []
            datasets = []

        timeline.append({
            "year":             paper.get("year", ""),
            "arxiv_id":         arxiv_id,
            "title":            paper["title"],
            "methods":          methods,
            "datasets":         datasets,
            "key_contribution": (
                f"Uses {', '.join(methods[:2])}" if methods
                else "See paper for details"
            ),
        })

    print(f"\n[GraphAgent] Timeline ({len(timeline)} papers):")
    for t in timeline:
        print(f"  [{t['year']}] {t['title'][:50]}")
        if t["methods"]:
            print(f"           Methods: {t['methods'][:2]}")

    print(f"\n[GraphAgent] Relationships: {len(relationships)}")

    return {
        **state,
        "graph_relationships": relationships,
        "timeline":            timeline,
        "errors":              state.get("errors", []),
    }