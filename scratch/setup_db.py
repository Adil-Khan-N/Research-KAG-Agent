from ingestion.db import engine, test_connection
from ingestion.schema import create_schema, get_table_stats

print("Testing connection...")
test_connection()

print("\nCreating schema...")
create_schema(engine)

print("\nTable stats after creation:")
get_table_stats(engine)