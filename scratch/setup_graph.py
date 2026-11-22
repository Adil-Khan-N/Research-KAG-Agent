"""Run once to create Neo4j constraints and indexes."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.neo4j_client import Neo4jClient
from graph.schema import create_schema

client = Neo4jClient()
client.test_connection()
create_schema(client)
client.close()