"""
Graph verification script — Day 6 checkpoint proof.
Runs 6 Cypher queries and prints results clearly.
"""
from graph.neo4j_client import Neo4jClient

client = Neo4jClient()

print("\n" + "="*60)
print("GRAPH VERIFICATION — Day 6")
print("="*60)

# Query 1: Node counts
print("\n[1] Node counts:")
result = client.run("""
    MATCH (n) 
    RETURN labels(n)[0] AS label, COUNT(n) AS count 
    ORDER BY count DESC
""")
for row in result:
    print(f"    {row['label']}: {row['count']}")

# Query 2: Relationship counts
print("\n[2] Relationship counts:")
result = client.run("""
    MATCH ()-[r]->() 
    RETURN type(r) AS type, COUNT(r) AS count 
    ORDER BY count DESC
""")
for row in result:
    print(f"    {row['type']}: {row['count']}")

# Query 3: Papers using the most methods
print("\n[3] Top 5 papers by method count:")
result = client.run("""
    MATCH (p:Paper)-[:USES]->(m:Method)
    RETURN p.title AS title, COUNT(m) AS method_count
    ORDER BY method_count DESC 
    LIMIT 5
""")
for row in result:
    print(f"    [{row['method_count']} methods] {row['title'][:55]}")

# Query 4: EXTENDS chains
print("\n[4] EXTENDS relationships (child → parent):")
result = client.run("""
    MATCH (p1:Paper)-[:EXTENDS]->(p2:Paper)
    RETURN p1.title AS child, p2.title AS parent
    ORDER BY p2.title
    LIMIT 15
""")
if result:
    for row in result:
        print(f"    {row['child'][:38]} → {row['parent'][:38]}")
else:
    print("    None found")

# Query 5: Most used datasets
print("\n[5] Most evaluated-on datasets:")
result = client.run("""
    MATCH (p:Paper)-[:EVALUATES_ON]->(d:Dataset)
    RETURN d.name AS dataset, COUNT(p) AS paper_count
    ORDER BY paper_count DESC 
    LIMIT 8
""")
for row in result:
    print(f"    {row['dataset']}: {row['paper_count']} papers")

# Query 6: Multi-hop — papers extending ViT and their methods
print("\n[6] Multi-hop: methods used by papers that EXTEND ViT:")
result = client.run("""
    MATCH (seed:Paper)
    WHERE seed.title CONTAINS 'Image is Worth'
    MATCH (p:Paper)-[:EXTENDS]->(seed)
    MATCH (p)-[:USES]->(m:Method)
    RETURN p.title AS paper, collect(m.name)[..4] AS methods
    LIMIT 8
""")
if result:
    for row in result:
        print(f"    {row['paper'][:45]}")
        print(f"      Methods: {row['methods']}")
else:
    print("    No results — try broader EXTENDS query below")
    # Fallback: show all EXTENDS chains 2 hops deep
    result2 = client.run("""
        MATCH (p1:Paper)-[:EXTENDS*1..2]->(p2:Paper)
        RETURN p1.title AS descendant, p2.title AS ancestor
        LIMIT 10
    """)
    for row in result2:
        print(f"    {row['descendant'][:40]} → {row['ancestor'][:40]}")

# Query 7: Papers sharing datasets (proves graph connectivity)
print("\n[7] Papers sharing the same dataset:")
result = client.run("""
    MATCH (p1:Paper)-[:EVALUATES_ON]->(d:Dataset)<-[:EVALUATES_ON]-(p2:Paper)
    WHERE p1.arxiv_id < p2.arxiv_id
    RETURN p1.title AS paper1, p2.title AS paper2, d.name AS dataset
    ORDER BY d.name
    LIMIT 8
""")
if result:
    for row in result:
        print(f"    [{row['dataset']}]")
        print(f"      {row['paper1'][:50]}")
        print(f"      {row['paper2'][:50]}")
else:
    print("    None found")

# Query 8: Most discussed concepts
print("\n[8] Top 8 concepts across corpus:")
result = client.run("""
    MATCH (p:Paper)-[:DISCUSSES]->(c:Concept)
    RETURN c.name AS concept, COUNT(p) AS paper_count
    ORDER BY paper_count DESC
    LIMIT 8
""")
for row in result:
    print(f"    {row['concept']}: {row['paper_count']} papers")

# Query 9: Isolated nodes check (graph health)
print("\n[9] Graph health — isolated Paper nodes (no relationships):")
result = client.run("""
    MATCH (p:Paper)
    WHERE NOT (p)-[]-()
    RETURN p.title AS title, p.arxiv_id AS arxiv_id
""")
if result:
    for row in result:
        print(f"    ISOLATED: {row['title'][:55]}")
else:
    print("    ✓ No isolated paper nodes — all papers connected")

# Query 10: 2-hop path example
print("\n[10] 2-hop path: concept → paper → dataset:")
result = client.run("""
    MATCH (c:Concept)<-[:DISCUSSES]-(p:Paper)-[:EVALUATES_ON]->(d:Dataset)
    RETURN c.name AS concept, p.title AS paper, d.name AS dataset
    ORDER BY c.name
    LIMIT 6
""")
for row in result:
    print(f"    {row['concept'][:25]} ← {row['paper'][:35]} → {row['dataset']}")

client.close()

print("\n" + "="*60)
print("✓ Verification complete")
print("="*60)