"""
Timeline Evolution View.

Given a seed concept or paper, queries the EXTENDS chain
in Neo4j and reconstructs a chronological timeline.

Returns:
  [
    {year: 2017, title: "Attention Is All You Need", contribution: "..."},
    {year: 2021, title: "ViT", contribution: "..."},
    {year: 2021, title: "DeiT", contribution: "..."},
    ...
  ]
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    year:         int
    arxiv_id:     str
    title:        str
    contribution: str
    methods:      list[str]
    datasets:     list[str]
    extends:      list[str]   # parent paper titles
    depth:        int          # hops from seed


def build_timeline(
    seed_arxiv_id: str = None,
    seed_concept:  str = None,
    max_depth:     int = 4,
    client=None,
) -> list[TimelineEntry]:
    """
    Build chronological timeline from a seed paper or concept.

    Strategy:
    1. If seed_arxiv_id: follow EXTENDS chain up and down
    2. If seed_concept: find papers discussing that concept,
       then follow their EXTENDS chains

    Args:
        seed_arxiv_id: arXiv ID to start from
        seed_concept: concept name to search for
        max_depth: max EXTENDS hops to follow
        client: Neo4j client

    Returns:
        list of TimelineEntry sorted by year
    """
    if client is None:
        from graph.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

    paper_ids = set()

    # ── Collect papers via seed ───────────────────────────────
    if seed_arxiv_id:
        # Start from this paper
        paper_ids.add(seed_arxiv_id)

        # Follow ancestors (papers this extends)
        try:
            rows = client.run(f"""
                MATCH (seed:Paper {{arxiv_id: $id}})
                MATCH (seed)-[:EXTENDS*1..{max_depth}]->(ancestor:Paper)
                RETURN DISTINCT ancestor.arxiv_id AS arxiv_id
            """, {"id": seed_arxiv_id})
            for row in rows:
                if row["arxiv_id"]:
                    paper_ids.add(row["arxiv_id"])
        except Exception as e:
            logger.error(f"Ancestor query failed: {e}")

        # Follow descendants (papers extending this)
        try:
            rows = client.run(f"""
                MATCH (seed:Paper {{arxiv_id: $id}})
                MATCH (descendant:Paper)
                    -[:EXTENDS*1..{max_depth}]->(seed)
                RETURN DISTINCT descendant.arxiv_id AS arxiv_id
            """, {"id": seed_arxiv_id})
            for row in rows:
                if row["arxiv_id"]:
                    paper_ids.add(row["arxiv_id"])
        except Exception as e:
            logger.error(f"Descendant query failed: {e}")

    elif seed_concept:
        # Find papers discussing this concept
        try:
            rows = client.run("""
                MATCH (c:Concept)
                WHERE c.name CONTAINS $concept
                MATCH (p:Paper)-[:DISCUSSES]->(c)
                RETURN DISTINCT p.arxiv_id AS arxiv_id
                LIMIT 10
            """, {"concept": seed_concept})
            for row in rows:
                if row["arxiv_id"]:
                    paper_ids.add(row["arxiv_id"])

            # Also find papers connected by EXTENDS
            for pid in list(paper_ids):
                rows = client.run(f"""
                    MATCH (seed:Paper {{arxiv_id: $id}})
                    OPTIONAL MATCH (seed)
                        -[:EXTENDS*1..2]->(ancestor:Paper)
                    OPTIONAL MATCH (descendant:Paper)
                        -[:EXTENDS*1..2]->(seed)
                    RETURN DISTINCT
                        ancestor.arxiv_id   AS ancestor_id,
                        descendant.arxiv_id AS descendant_id
                """, {"id": pid})
                for row in rows:
                    if row["ancestor_id"]:
                        paper_ids.add(row["ancestor_id"])
                    if row["descendant_id"]:
                        paper_ids.add(row["descendant_id"])

        except Exception as e:
            logger.error(f"Concept query failed: {e}")

    if not paper_ids:
        logger.warning("No papers found for timeline")
        return []

    # ── Fetch details for all collected papers ─────────────────
    entries = []
    paper_id_list = list(paper_ids)

    try:
        rows = client.run("""
            MATCH (p:Paper)
            WHERE p.arxiv_id IN $ids
            OPTIONAL MATCH (p)-[:USES]->(m:Method)
            OPTIONAL MATCH (p)-[:EVALUATES_ON]->(d:Dataset)
            OPTIONAL MATCH (p)-[:EXTENDS]->(parent:Paper)
            RETURN
                p.arxiv_id  AS arxiv_id,
                p.title     AS title,
                p.year      AS year,
                p.abstract  AS abstract,
                collect(DISTINCT m.name)[..3] AS methods,
                collect(DISTINCT d.name)[..3] AS datasets,
                collect(DISTINCT parent.title)[..2] AS extends
        """, {"ids": paper_id_list})

        for row in rows:
            year = row["year"] or 0
            title = row["title"] or row["arxiv_id"]
            abstract = row["abstract"] or ""

            # Generate short contribution from abstract
            contribution = _extract_contribution(abstract, title)

            entries.append(TimelineEntry(
                year         = year,
                arxiv_id     = row["arxiv_id"],
                title        = title,
                contribution = contribution,
                methods      = [m for m in (row["methods"] or []) if m],
                datasets     = [d for d in (row["datasets"] or []) if d],
                extends      = [e for e in (row["extends"] or []) if e],
                depth        = 0,
            ))

    except Exception as e:
        logger.error(f"Timeline details query failed: {e}")

    # Sort by year
    entries.sort(key=lambda e: (e.year, e.title))

    logger.info(
        f"Timeline built: {len(entries)} papers "
        f"({min(e.year for e in entries) if entries else 0}"
        f"-{max(e.year for e in entries) if entries else 0})"
    )
    return entries


def _extract_contribution(abstract: str, title: str) -> str:
    """Extract a one-sentence contribution from the abstract."""
    if not abstract:
        return title

    # Look for "we propose", "we introduce", "we present"
    import re
    patterns = [
        r'we\s+(?:propose|introduce|present|demonstrate)\s+[^.]+\.',
        r'this\s+(?:paper|work)\s+[^.]+\.',
        r'we\s+show\s+[^.]+\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, abstract.lower())
        if match:
            sentence = abstract[match.start():match.end()]
            return sentence[:150].strip()

    # Fallback: first sentence
    sentences = abstract.split(".")
    if sentences:
        return sentences[0][:150].strip()

    return title[:100]


def format_timeline_for_display(
    entries: list[TimelineEntry],
) -> list[dict]:
    """Convert timeline entries to JSON-serializable dicts."""
    return [
        {
            "year":         e.year,
            "arxiv_id":     e.arxiv_id,
            "title":        e.title,
            "contribution": e.contribution,
            "methods":      e.methods[:3],
            "datasets":     e.datasets[:3],
            "extends":      e.extends[:2],
        }
        for e in entries
    ]