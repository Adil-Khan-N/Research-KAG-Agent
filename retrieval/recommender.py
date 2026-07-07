"""
Graph-based paper recommendations.
Pure Cypher traversal — zero LLM calls.

Ranking factors (all from Neo4j graph):
1. Shared methods (USES relationships)
2. Shared datasets (EVALUATES_ON relationships)
3. Shared concepts (DISCUSSES relationships)
4. Citation proximity (EXTENDS chain distance)
5. Shared tasks (ADDRESSES relationships)

Each factor contributes a weighted score.
Papers with more shared graph edges rank higher.

CV talking point:
"Graph-based recommendation system using pure Cypher traversal,
zero LLM inference cost, sub-50ms latency"
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Weights for each relationship type
WEIGHTS = {
    "method":   2.0,  # shared methods = strong signal
    "dataset":  1.5,  # shared datasets = good signal
    "concept":  1.0,  # shared concepts = weaker signal
    "task":     1.0,  # shared tasks = weaker signal
    "extends":  3.0,  # direct citation = strongest signal
}


@dataclass
class Recommendation:
    """One paper recommendation with explanation."""
    arxiv_id:       str
    title:          str
    year:           int
    score:          float
    shared_methods:  list[str]
    shared_datasets: list[str]
    shared_concepts: list[str]
    shared_tasks:    list[str]
    extends:         bool        # directly extends seed paper
    extended_by:     bool        # seed extends this paper
    reasoning:       str         # human-readable explanation


def get_recommendations(
    arxiv_id: str,
    top_k: int = 5,
    client=None,
) -> list[Recommendation]:
    """
    Get paper recommendations for a seed paper.

    Algorithm:
    1. Find all papers sharing methods, datasets, concepts, tasks
    2. Find papers in EXTENDS chain
    3. Score by weighted shared edge count
    4. Return top_k sorted by score

    Args:
        arxiv_id: seed paper arXiv ID
        top_k: number of recommendations
        client: Neo4j client

    Returns:
        list of Recommendation sorted by score descending
    """
    if client is None:
        from graph.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

    scores: dict[str, dict] = {}

    def add_score(
        other_id: str,
        other_title: str,
        other_year: int,
        weight: float,
        rel_type: str,
        entity_name: str,
    ):
        if other_id == arxiv_id:
            return
        if other_id not in scores:
            scores[other_id] = {
                "arxiv_id":       other_id,
                "title":          other_title,
                "year":           other_year,
                "score":          0.0,
                "shared_methods":  [],
                "shared_datasets": [],
                "shared_concepts": [],
                "shared_tasks":    [],
                "extends":         False,
                "extended_by":     False,
            }
        scores[other_id]["score"] += weight
        key = f"shared_{rel_type}s"
        if key in scores[other_id]:
            if entity_name not in scores[other_id][key]:
                scores[other_id][key].append(entity_name)

    # ── Query 1: Shared methods ───────────────────────────────
    try:
        rows = client.run("""
            MATCH (seed:Paper {arxiv_id: $id})-[:USES]->(m:Method)
            MATCH (other:Paper)-[:USES]->(m)
            WHERE other.arxiv_id <> $id
            RETURN other.arxiv_id AS id,
                   other.title    AS title,
                   other.year     AS year,
                   m.name         AS method
        """, {"id": arxiv_id})

        for row in rows:
            add_score(
                row["id"], row["title"], row["year"] or 0,
                WEIGHTS["method"], "method", row["method"]
            )
        logger.info(f"Methods query: {len(rows)} rows")
    except Exception as e:
        logger.error(f"Methods query failed: {e}")

    # ── Query 2: Shared datasets ──────────────────────────────
    try:
        rows = client.run("""
            MATCH (seed:Paper {arxiv_id: $id})
                  -[:EVALUATES_ON]->(d:Dataset)
            MATCH (other:Paper)-[:EVALUATES_ON]->(d)
            WHERE other.arxiv_id <> $id
            RETURN other.arxiv_id AS id,
                   other.title    AS title,
                   other.year     AS year,
                   d.name         AS dataset
        """, {"id": arxiv_id})

        for row in rows:
            add_score(
                row["id"], row["title"], row["year"] or 0,
                WEIGHTS["dataset"], "dataset", row["dataset"]
            )
        logger.info(f"Datasets query: {len(rows)} rows")
    except Exception as e:
        logger.error(f"Datasets query failed: {e}")

    # ── Query 3: Shared concepts ──────────────────────────────
    try:
        rows = client.run("""
            MATCH (seed:Paper {arxiv_id: $id})
                  -[:DISCUSSES]->(c:Concept)
            MATCH (other:Paper)-[:DISCUSSES]->(c)
            WHERE other.arxiv_id <> $id
            RETURN other.arxiv_id AS id,
                   other.title    AS title,
                   other.year     AS year,
                   c.name         AS concept
            LIMIT 50
        """, {"id": arxiv_id})

        for row in rows:
            add_score(
                row["id"], row["title"], row["year"] or 0,
                WEIGHTS["concept"], "concept", row["concept"]
            )
        logger.info(f"Concepts query: {len(rows)} rows")
    except Exception as e:
        logger.error(f"Concepts query failed: {e}")

    # ── Query 4: Shared tasks ─────────────────────────────────
    try:
        rows = client.run("""
            MATCH (seed:Paper {arxiv_id: $id})
                  -[:ADDRESSES]->(t:Task)
            MATCH (other:Paper)-[:ADDRESSES]->(t)
            WHERE other.arxiv_id <> $id
            RETURN other.arxiv_id AS id,
                   other.title    AS title,
                   other.year     AS year,
                   t.name         AS task
        """, {"id": arxiv_id})

        for row in rows:
            add_score(
                row["id"], row["title"], row["year"] or 0,
                WEIGHTS["task"], "task", row["task"]
            )
        logger.info(f"Tasks query: {len(rows)} rows")
    except Exception as e:
        logger.error(f"Tasks query failed: {e}")

    # ── Query 5: EXTENDS relationships ────────────────────────
    try:
        # Papers this paper extends
        rows = client.run("""
            MATCH (seed:Paper {arxiv_id: $id})-[:EXTENDS]->(parent:Paper)
            RETURN parent.arxiv_id AS id,
                   parent.title    AS title,
                   parent.year     AS year
        """, {"id": arxiv_id})

        for row in rows:
            add_score(
                row["id"], row["title"], row["year"] or 0,
                WEIGHTS["extends"], "method", "direct extension"
            )
            if row["id"] in scores:
                scores[row["id"]]["extends"] = True

        # Papers that extend this paper
        rows = client.run("""
            MATCH (child:Paper)-[:EXTENDS]->(seed:Paper {arxiv_id: $id})
            RETURN child.arxiv_id AS id,
                   child.title    AS title,
                   child.year     AS year
        """, {"id": arxiv_id})

        for row in rows:
            add_score(
                row["id"], row["title"], row["year"] or 0,
                WEIGHTS["extends"], "method", "extends this paper"
            )
            if row["id"] in scores:
                scores[row["id"]]["extended_by"] = True

        logger.info(f"EXTENDS query done")
    except Exception as e:
        logger.error(f"EXTENDS query failed: {e}")

    # ── Build and sort recommendations ────────────────────────
    recommendations = []
    for data in scores.values():
        # Build reasoning string
        parts = []
        if data["extends"]:
            parts.append("Directly extends this paper")
        if data["extended_by"]:
            parts.append("This paper extends it")
        if data["shared_methods"]:
            parts.append(
                f"Shares method{'s' if len(data['shared_methods']) > 1 else ''}: "
                f"{', '.join(data['shared_methods'][:2])}"
            )
        if data["shared_datasets"]:
            parts.append(
                f"Shares dataset{'s' if len(data['shared_datasets']) > 1 else ''}: "
                f"{', '.join(data['shared_datasets'][:2])}"
            )
        if data["shared_concepts"]:
            parts.append(
                f"Shares concept{'s' if len(data['shared_concepts']) > 1 else ''}: "
                f"{', '.join(data['shared_concepts'][:2])}"
            )

        reasoning = "; ".join(parts) if parts else "Related paper"

        recommendations.append(Recommendation(
            arxiv_id        = data["arxiv_id"],
            title           = data["title"],
            year            = data["year"],
            score           = round(data["score"], 2),
            shared_methods  = data["shared_methods"][:5],
            shared_datasets = data["shared_datasets"][:5],
            shared_concepts = data["shared_concepts"][:5],
            shared_tasks    = data["shared_tasks"][:5],
            extends         = data["extends"],
            extended_by     = data["extended_by"],
            reasoning       = reasoning,
        ))

    # Sort by score descending, break ties by year (newer first)
    recommendations.sort(
        key=lambda r: (-r.score, -(r.year or 0))
    )

    logger.info(
        f"Recommendations for {arxiv_id}: "
        f"{len(recommendations)} candidates, "
        f"returning top {top_k}"
    )
    return recommendations[:top_k]


def get_recommendations_for_entity(
    entity_name: str,
    entity_type: str = "method",
    top_k: int = 5,
    client=None,
) -> list[dict]:
    """
    Find papers related to a specific entity name.
    Used for "papers that use ImageNet" or "papers using attention".
    """
    if client is None:
        from graph.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

    rel_map = {
        "method":  "USES",
        "dataset": "EVALUATES_ON",
        "concept": "DISCUSSES",
        "task":    "ADDRESSES",
    }
    rel = rel_map.get(entity_type, "USES")

    try:
        rows = client.run(f"""
            MATCH (e:{entity_type.capitalize()})
            WHERE e.name CONTAINS $name
            MATCH (p:Paper)-[:{rel}]->(e)
            RETURN p.arxiv_id AS arxiv_id,
                   p.title    AS title,
                   p.year     AS year,
                   e.name     AS entity
            ORDER BY p.year DESC
            LIMIT $top_k
        """, {"name": entity_name, "top_k": top_k})

        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Entity recommendation failed: {e}")
        return []