"""Ingest extracted entities into Neo4j."""
import logging
logging.basicConfig(level=logging.INFO)

from graph.graph_ingestion import run_graph_ingestion
run_graph_ingestion()