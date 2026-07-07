"""
Audit the graph for bad EXTENDS relationships and fix them.
Run this before writing the query library.
"""

from graph.neo4j_client import Neo4jClient

client = Neo4jClient()

print("="*70)
print("GRAPH AUDIT — All EXTENDS Relationships")
print("="*70)

# Show all EXTENDS with years so we can spot wrong direction
result = client.run("""
    MATCH (p1:Paper)-[:EXTENDS]->(p2:Paper)
    RETURN p1.title AS child, p1.year AS child_year,
           p2.title AS parent, p2.year AS parent_year,
           p1.arxiv_id AS child_id, p2.arxiv_id AS parent_id
    ORDER BY p2.title
""")

print(f"\nFound {len(result)} EXTENDS relationships:\n")
bad_candidates = []

for i, row in enumerate(result, 1):
    child_year = row['child_year'] or 0
    parent_year = row['parent_year'] or 0

    is_suspicious = (
        child_year < parent_year or
        row['child_id'] == row['parent_id']
    )

    flag = "SUSPICIOUS" if is_suspicious else ""
    print(f"  [{i:02d}] {flag}")
    print(f"        {row['child'][:55]} ({child_year})")
    print(f"        → {row['parent'][:55]} ({parent_year})")


    if is_suspicious:
        bad_candidates.append({
            "child_id": row['child_id'],
            "parent_id": row['parent_id'],
            "child": row['child'],
            "parent": row['parent'],
        })

    print(f"\n{'='*70}")
    print(f"Suspicious relationships: {len(bad_candidates)}")
    print(f"{'='*70}")

    if bad_candidates:
        print("\nThese look wrong based on publication year:")
        for b in bad_candidates:
            print(f"  {b['child'][:50]}")
            print(f"  → {b['parent'][:50]}")
            print()

    answer = input("Delete all suspicious EXTENDS relationships? (y/n): ")
    if answer.lower() == 'y':
        for b in bad_candidates:
            client.run_write("""
                MATCH (p1:Paper {arxiv_id: $child_id})
                      -[r:EXTENDS]->
                      (p2:Paper {arxiv_id: $parent_id})
                DELETE r
            """, {"child_id": b["child_id"], "parent_id": b["parent_id"]})
            print(f"  Deleted: {b['child'][:40]} → {b['parent'][:40]}")
        print(f"\n✓ Deleted {len(bad_candidates)} bad relationships")
    else:
        print("Skipped deletion.")


# Also check for orphan non-Paper nodes
print("\n" + "="*70)
print("Checking for orphan entity nodes (no paper connected):")
print("="*70)

for label in ["Method", "Dataset", "Concept", "Task"]:
    result = client.run(f"""
        MATCH (n:{label})
        WHERE NOT (n)<-[]-(:Paper)
        RETURN n.name AS name
        LIMIT 10
    """)
    if result:
        print(f"\n  Orphan {label} nodes ({len(result)}):")
        for row in result:
            print(f"    - {row['name']}")
    else:
        print(f"\n  ✓ No orphan {label} nodes")

client.close()
print("\n✓ Audit complete")
