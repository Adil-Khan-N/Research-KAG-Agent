"""Run this to embed all processed papers into Postgres."""
import logging
logging.basicConfig(level=logging.INFO)

from ingestion.embedder import run_embedding_pipeline
from ingestion.schema import create_vector_index, get_table_stats
from ingestion.db import engine

# Step 1: Embed and insert all papers
run_embedding_pipeline()

# Step 2: Create the IVFFlat index AFTER data is inserted
print("\nCreating IVFFlat vector index...")
create_vector_index(engine)

# Step 3: Show final counts
print("\nFinal table stats:")
get_table_stats(engine)