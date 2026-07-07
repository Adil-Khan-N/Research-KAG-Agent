"""
Auto Knowledge Graph Updater.

When a new paper is ingested, this module:
1. Extracts entities from the paper
2. Diffs against existing Neo4j nodes/edges
3. MERGEs only NEW entities — no full rebuild
4. Logs what was added

This is explicitly incremental — not a full graph rebuild.

CV talking point:
"Implemented incremental knowledge graph updates with entity
diffing — new papers add only novel nodes and edges,
preserving existing graph structure"
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result of one auto-update run."""
    arxiv_id:          str
    title:             str
    methods_new:       list[str]  = field(default_factory=list)
    methods_existing:  list[str]  = field(default_factory=list)
    datasets_new:      list[str]  = field(default_factory=list)
    datasets_existing: list[str]  = field(default_factory=list)
    concepts_new:      list[str]  = field(default_factory=list)
    relationships_new: int = 0
    elapsed_ms:        int = 0


def get_existing_entities(client) -> dict:
    """
    Load all existing entity names from Neo4j.
    Used to diff against newly extracted entities.
    Returns {type: set_of_names}
    """
    existing = {
        "methods":  set(),
        "datasets": set(),
        "concepts": set(),
        "tasks":    set(),
        "papers":   set(),
    }

    try:
        rows = client.run(
            "MATCH (m:Method) RETURN m.name AS name"
        )
        existing["methods"] = {r["name"] for r in rows if r["name"]}

        rows = client.run(
            "MATCH (d:Dataset) RETURN d.name AS name"
        )
        existing["datasets"] = {r["name"] for r in rows if r["name"]}

        rows = client.run(
            "MATCH (c:Concept) RETURN c.name AS name"
        )
        existing["concepts"] = {r["name"] for r in rows if r["name"]}

        rows = client.run(
            "MATCH (t:Task) RETURN t.name AS name"
        )
        existing["tasks"] = {r["name"] for r in rows if r["name"]}

        rows = client.run(
            "MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id"
        )
        existing["papers"] = {r["arxiv_id"] for r in rows if r["arxiv_id"]}

    except Exception as e:
        logger.error(f"Failed to load existing entities: {e}")

    logger.info(
        f"Existing graph: {len(existing['methods'])} methods, "
        f"{len(existing['datasets'])} datasets, "
        f"{len(existing['concepts'])} concepts, "
        f"{len(existing['papers'])} papers"
    )
    return existing


def diff_entities(
    extracted: dict,
    existing: dict,
) -> dict:
    """
    Diff extracted entities against existing graph nodes.
    Returns {new: [...], existing: [...]} for each entity type.
    """
    diff = {}

    for entity_type in ["methods", "datasets", "concepts", "tasks"]:
        extracted_set = set(extracted.get(entity_type, []))
        existing_set  = existing.get(entity_type, set())

        new_entities      = extracted_set - existing_set
        already_existing  = extracted_set & existing_set

        diff[entity_type] = {
            "new":      list(new_entities),
            "existing": list(already_existing),
        }

    return diff


def merge_new_entities(
    client,
    arxiv_id: str,
    diff: dict,
    paper_data: dict,
) -> int:
    """
    MERGE only new entities into Neo4j.
    Returns count of new relationships created.
    """
    rels_created = 0

    # New methods
    for method_name in diff.get("methods", {}).get("new", []):
        if not method_name.strip():
            continue
        try:
            client.run_write("""
                MERGE (m:Method {name: $name})
                WITH m
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                MERGE (p)-[:USES]->(m)
            """, {"name": method_name, "arxiv_id": arxiv_id})
            rels_created += 1
        except Exception as e:
            logger.error(f"Failed to merge method {method_name}: {e}")

    # Existing methods — just add relationship if missing
    for method_name in diff.get("methods", {}).get("existing", []):
        if not method_name.strip():
            continue
        try:
            client.run_write("""
                MATCH (m:Method {name: $name})
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                MERGE (p)-[:USES]->(m)
            """, {"name": method_name, "arxiv_id": arxiv_id})
            rels_created += 1
        except Exception as e:
            logger.error(f"Failed to link method {method_name}: {e}")

    # New datasets
    for dataset_name in diff.get("datasets", {}).get("new", []):
        if not dataset_name.strip():
            continue
        try:
            client.run_write("""
                MERGE (d:Dataset {name: $name})
                WITH d
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                MERGE (p)-[:EVALUATES_ON]->(d)
            """, {"name": dataset_name, "arxiv_id": arxiv_id})
            rels_created += 1
        except Exception as e:
            logger.error(f"Failed to merge dataset {dataset_name}: {e}")

    # Existing datasets
    for dataset_name in diff.get("datasets", {}).get("existing", []):
        if not dataset_name.strip():
            continue
        try:
            client.run_write("""
                MATCH (d:Dataset {name: $name})
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                MERGE (p)-[:EVALUATES_ON]->(d)
            """, {"name": dataset_name, "arxiv_id": arxiv_id})
            rels_created += 1
        except Exception as e:
            logger.error(f"Failed to link dataset {dataset_name}: {e}")

    # New concepts
    for concept_name in diff.get("concepts", {}).get("new", []):
        if not concept_name.strip():
            continue
        try:
            client.run_write("""
                MERGE (c:Concept {name: $name})
                WITH c
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                MERGE (p)-[:DISCUSSES]->(c)
            """, {"name": concept_name, "arxiv_id": arxiv_id})
            rels_created += 1
        except Exception as e:
            logger.error(f"Failed to merge concept {concept_name}: {e}")

    return rels_created


def auto_update_graph(
    paper_data: dict,
    entities: dict = None,
    client=None,
) -> UpdateResult:
    """
    Incrementally update the knowledge graph for a new paper.

    Args:
        paper_data: dict with arxiv_id, title, sections, etc.
        entities: pre-extracted entities (or extract them here)
        client: Neo4j client

    Returns:
        UpdateResult showing what was added vs already existed
    """
    if client is None:
        from graph.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

    start = time.time()
    arxiv_id = paper_data["arxiv_id"]
    title    = paper_data.get("title", arxiv_id)

    logger.info(f"Auto-updating graph for: {title[:50]}")

    # Step 1: Extract entities if not provided
    if entities is None:
        from graph.entity_extractor import extract_entities_from_paper
        entities = extract_entities_from_paper(paper_data)
        logger.info(
            f"Extracted: {len(entities.get('methods', []))} methods, "
            f"{len(entities.get('datasets', []))} datasets"
        )

    # Step 2: Ensure Paper node exists
    from graph.graph_ingestion import ingest_paper_node
    ingest_paper_node(client, paper_data)

    # Step 3: Load existing graph state
    existing = get_existing_entities(client)

    # Step 4: Diff
    diff = diff_entities(entities, existing)

    logger.info(
        f"Diff results for {arxiv_id}:\n"
        f"  Methods:  {len(diff['methods']['new'])} new, "
        f"{len(diff['methods']['existing'])} existing\n"
        f"  Datasets: {len(diff['datasets']['new'])} new, "
        f"{len(diff['datasets']['existing'])} existing\n"
        f"  Concepts: {len(diff['concepts']['new'])} new"
    )

    # Step 5: Merge only new entities
    rels_created = merge_new_entities(
        client=client,
        arxiv_id=arxiv_id,
        diff=diff,
        paper_data=paper_data,
    )

    elapsed = int((time.time() - start) * 1000)

    result = UpdateResult(
        arxiv_id          = arxiv_id,
        title             = title,
        methods_new       = diff["methods"]["new"],
        methods_existing  = diff["methods"]["existing"],
        datasets_new      = diff["datasets"]["new"],
        datasets_existing = diff["datasets"]["existing"],
        concepts_new      = diff["concepts"]["new"],
        relationships_new = rels_created,
        elapsed_ms        = elapsed,
    )

    logger.info(
        f"Auto-update complete for {arxiv_id}: "
        f"{len(result.methods_new)} new methods, "
        f"{len(result.datasets_new)} new datasets, "
        f"{rels_created} relationships added "
        f"in {elapsed}ms"
    )

    return result


def get_graph_delta_log(client=None, limit: int = 20) -> list[dict]:
    """
    Get recent CONTRADICTS edges — these are auto-added by
    Day 15 contradiction detector. Used in timeline view.
    """
    if client is None:
        from graph.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

    try:
        rows = client.run("""
            MATCH (p1:Paper)-[r:CONTRADICTS]->(p2:Paper)
            RETURN p1.title  AS paper1,
                   p2.title  AS paper2,
                   r.topic   AS topic,
                   r.confidence AS confidence,
                   r.detected_at AS detected_at
            ORDER BY r.detected_at DESC
            LIMIT $limit
        """, {"limit": limit})

        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Graph delta log failed: {e}")
        return []