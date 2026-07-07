"""
Ingest extracted entities into Neo4j.
Creates nodes and relationships using MERGE (idempotent).
Safe to re-run — won't create duplicates.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def ingest_paper_node(client: Neo4jClient, paper: dict):
    """MERGE a Paper node with metadata."""
    client.run_write("""
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p.title = $title,
            p.year = $year,
            p.abstract = $abstract,
            p.authors = $authors,
            p.pdf_url = $pdf_url
    """, {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "year": paper.get("year", 0),
        "abstract": paper.get("abstract", "")[:500],
        "authors": paper.get("authors", [])[:5],
        "pdf_url": paper.get("pdf_url", ""),
    })


def ingest_entities_and_relationships(
    client: Neo4jClient,
    entities: dict,
    paper_meta: dict,
):
    """
    Given extracted entities for one paper, create all nodes
    and relationships in Neo4j.
    """
    arxiv_id = entities["arxiv_id"]

    # 1. Methods — Paper USES Method
    for method_name in entities.get("methods", []):
        if not method_name.strip():
            continue
        client.run_write("""
            MERGE (m:Method {name: $name})
            WITH m
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:USES]->(m)
        """, {"name": method_name.strip(), "arxiv_id": arxiv_id})

    # 2. Datasets — Paper EVALUATES_ON Dataset
    for dataset_name in entities.get("datasets", []):
        if not dataset_name.strip():
            continue
        client.run_write("""
            MERGE (d:Dataset {name: $name})
            WITH d
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:EVALUATES_ON]->(d)
        """, {"name": dataset_name.strip(), "arxiv_id": arxiv_id})

    # 3. Tasks — Paper ADDRESSES Task
    for task_name in entities.get("tasks", []):
        if not task_name.strip():
            continue
        client.run_write("""
            MERGE (t:Task {name: $name})
            WITH t
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:ADDRESSES]->(t)
        """, {"name": task_name.strip(), "arxiv_id": arxiv_id})

    # 4. Concepts — Paper DISCUSSES Concept
    for concept_name in entities.get("concepts", []):
        if not concept_name.strip():
            continue
        client.run_write("""
            MERGE (c:Concept {name: $name})
            WITH c
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:DISCUSSES]->(c)
        """, {"name": concept_name.strip(), "arxiv_id": arxiv_id})

    # 5. Authors — Paper AUTHORED_BY Author
    for author_name in paper_meta.get("authors", [])[:5]:
        if not author_name.strip():
            continue
        client.run_write("""
            MERGE (a:Author {name: $name})
            WITH a
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:AUTHORED_BY]->(a)
        """, {"name": author_name.strip(), "arxiv_id": arxiv_id})


def resolve_extends_relationships(
    client: Neo4jClient,
    all_entities: list[dict],
):
    """
    Resolve EXTENDS relationships between papers.
    The extractor returns names like "ViT", "BERT" — we need to
    match these to actual Paper nodes by title fuzzy matching.
    """
    logger.info("Resolving EXTENDS relationships...")

    # Build a lookup: short name → arxiv_id
    # e.g. "ViT" → "2010.11929", "Swin" → "2103.14030"
    title_lookup = {}
    for e in all_entities:
        title = e["title"].lower()
        arxiv_id = e["arxiv_id"]
        # Add full title
        title_lookup[title] = arxiv_id
        # Add common short names
        short_names = _extract_short_names(e["title"])
        for name in short_names:
            title_lookup[name.lower()] = arxiv_id

    extends_created = 0
    for entities in all_entities:
        arxiv_id = entities["arxiv_id"]
        for extends_name in entities.get("extends", []):
            extends_lower = extends_name.lower().strip()
            # Find the target paper
            target_id = None
            for lookup_name, lookup_id in title_lookup.items():
                if extends_lower in lookup_name or lookup_name in extends_lower:
                    if lookup_id != arxiv_id:  # don't self-reference
                        target_id = lookup_id
                        break

            if target_id:
                client.run_write("""
                    MATCH (p1:Paper {arxiv_id: $from_id})
                    MATCH (p2:Paper {arxiv_id: $to_id})
                    MERGE (p1)-[:EXTENDS]->(p2)
                """, {"from_id": arxiv_id, "to_id": target_id})
                extends_created += 1
                logger.info(f"  EXTENDS: {arxiv_id} → {target_id} (via '{extends_name}')")

    logger.info(f"Created {extends_created} EXTENDS relationships")
    return extends_created


def _extract_short_names(title: str) -> list[str]:
    """Extract common short names from a paper title."""
    import re
    names = []
    # Find acronyms in parentheses e.g. "Vision Transformer (ViT)"
    acronyms = re.findall(r'\(([A-Z][A-Za-z0-9\-]+)\)', title)
    names.extend(acronyms)
    # Find capitalized model names
    model_names = re.findall(r'\b([A-Z][a-z]+(?:ViT|BERT|GPT|Net|Former|Transformer))\b', title)
    names.extend(model_names)
    # First word if capitalized and short
    words = title.split()
    if words and len(words[0]) <= 8 and words[0][0].isupper():
        names.append(words[0])
    return names


def run_graph_ingestion(
    entities_path: str = "data/entities.json",
    metadata_path: str = "data/papers_metadata.json",
):
    """
    Full graph ingestion pipeline:
    1. Load extracted entities
    2. Create Paper nodes
    3. Create entity nodes + relationships
    4. Resolve EXTENDS relationships
    """
    client = Neo4jClient()

    # Load data
    with open(entities_path) as f:
        all_entities = json.load(f)

    with open(metadata_path) as f:
        all_metadata = json.load(f)

    meta_lookup = {m["arxiv_id"]: m for m in all_metadata}

    print(f"Ingesting {len(all_entities)} papers into Neo4j...")

    # Step 1: Create all Paper nodes first
    print("\nStep 1: Creating Paper nodes...")
    for meta in tqdm(all_metadata, desc="Paper nodes"):
        ingest_paper_node(client, meta)

    # Step 2: Create entity nodes + relationships
    print("\nStep 2: Creating entity nodes and relationships...")
    for entities in tqdm(all_entities, desc="Entities"):
        arxiv_id = entities["arxiv_id"]
        paper_meta = meta_lookup.get(arxiv_id, {})
        ingest_entities_and_relationships(client, entities, paper_meta)

    # Step 3: Resolve EXTENDS relationships
    print("\nStep 3: Resolving EXTENDS relationships...")
    extends_count = resolve_extends_relationships(client, all_entities)

    # Step 4: Print stats
    print("\n" + "="*60)
    print("GRAPH INGESTION COMPLETE")
    print("="*60)
    stats = client.get_stats()

    print("\nNode counts:")
    for row in stats["nodes"]:
        print(f"  {row['label']}: {row['count']}")

    print("\nRelationship counts:")
    for row in stats["relationships"]:
        print(f"  {row['type']}: {row['count']}")

    print(f"\n  EXTENDS relationships created: {extends_count}")
    print("="*60)

    client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_graph_ingestion()