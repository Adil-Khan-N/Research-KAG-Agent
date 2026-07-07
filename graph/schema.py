"""
Create Neo4j constraints and indexes.
Run once before ingesting any data.
"""

import logging
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT paper_arxiv_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT task_name IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
]

# Full-text search indexes for name lookup
INDEXES = [
    "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
    "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
]

def create_schema(client: Neo4jClient):
    """Create all constraints and indexes."""
    print("Creating Neo4j constraints...")

    for constraint in CONSTRAINTS:
        try:
            client.run_write(constraint)
            print(f"Created constraint: {constraint[17:60]}")

        except Exception as e:
            print(f"Already exists or failed to create constraint: {constraint[17:60]} - {e}")

    print("Creating Neo4j indexes...")
    for index in INDEXES:
        try:
            client.run_write(index)
            print(f"Created index: {index[13:50]}")
        except Exception as e:
            print(f"Already exists or failed to create index: {index[13:50]} - {e}")

    print("\n✓ Neo4j schema ready")

def clear_all_data(client: Neo4jClient):
    """Nuclear option — wipe all nodes and relationships. Dev only."""
    client.run_write("MATCH (n) DETACH DELETE n")
    print("✓ All Neo4j data cleared")

if __name__ == "__main__":
    from graph.neo4j_client import Neo4jClient
    client = Neo4jClient()
    client.test_connection()
    create_schema(client)
    client.close()
