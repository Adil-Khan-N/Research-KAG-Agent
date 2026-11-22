import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.db import engine, test_connection
from ingestion.schema import create_schema, get_table_stats

print("Testing connection...")
test_connection()

print("\nCreating schema...")
create_schema(engine)

print("\nTable stats after creation:")
get_table_stats(engine)