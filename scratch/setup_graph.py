"""Run once to create Neo4j constraints and indexes."""
from graph.neo4j_client import Neo4jClient
from graph.schema import create_schema

client = Neo4jClient()
client.test_connection()
create_schema(client)
client.close()