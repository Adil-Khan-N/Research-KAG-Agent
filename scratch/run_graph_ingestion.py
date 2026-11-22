"""Ingest extracted entities into Neo4j."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
logging.basicConfig(level=logging.INFO)

from graph.graph_ingestion import run_graph_ingestion
run_graph_ingestion()