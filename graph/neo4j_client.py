"""
Neo4j connection client.
Single place to get a Neo4j driver — import this everywhere.
"""

import os
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

class Neo4jClient:
    """Wrapper around Neo4j driver with helper methods."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        logger.info(f"Connected to Neo4j at {NEO4J_URI} as user {NEO4J_USER}")

    def close(self):
        self.driver.close()

    def run(self, query:str, parameters:dict=None):
        """Run a Cypher query with optional parameters."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return result.data()

    def run_write(self, query:str, parameters:dict = None):
        """Run a write Cypher query"""
        with self.driver.session() as session:
            session.run(query, parameters or {})

    def test_connection(self)->bool:
        """Verify Neo4j is reachable."""
        try:
            result = self.run("RETURN 'connected' AS status")
            print(f"Neo4j connected: {result[0]['status']}")
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            return False
        
    def get_stats(self) -> dict:
        """Get node and relationship counts."""
        nodes = self.run("MATCH (n) RETURN labels(n)[0] AS label, COUNT(n) AS count ORDER BY count DESC")
        rels = self.run("MATCH ()-[r]->() RETURN type(r) AS type, COUNT(r) AS count ORDER BY count DESC")
        return {"nodes": nodes, "relationships": rels}
    
    # Singleton instance
_client = None

def get_neo4j_client() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client


if __name__ == "__main__":
    client = Neo4jClient()
    client.test_connection()
    client.close()
