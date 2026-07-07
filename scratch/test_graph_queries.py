"""
Test all graph query functions with real queries against your corpus.
This is your Day 7 checkpoint proof.
Run: python scratch/test_graph_queries.py
"""
from graph.neo4j_client import get_neo4j_client

from graph.graph_queries import (
    expand_neighbors,
    get_papers_by_method,
    get_papers_by_dataset,
    get_citation_chain,
    find_papers_sharing_concepts,
    find_contradicting_papers,
    get_paper_neighborhood,
    get_all_entity_names,
)

print("="*70)
print("GRAPH QUERY LIBRARY — Day 7 Tests")
print("="*70)

# ── Test 1: expand_neighbors ──────────────────────────────────
print("\n[TEST 1] expand_neighbors(['Multi-Head Attention', 'ImageNet'], hops=2)")
results = expand_neighbors(
    entity_names=["Multi-Head Attention", "ImageNet"],
    hops=2
)
print(f"  Found {len(results)} papers:")
for r in results[:5]:
    print(f"    [{r['graph_score']:.1f}] {r['title'][:55]} "
          f"| via: {r['matched_entities']}")

# ── Test 2: get_papers_by_method ─────────────────────────────
print("\n[TEST 2] get_papers_by_method('Attention')")
results = get_papers_by_method("Attention")
print(f"  Found {len(results)} papers:")
for r in results[:5]:
    print(f"    {r['title'][:55]} ({r['year']})")
    print(f"    Matched method: {r['matched_method']}")

# ── Test 3: get_papers_by_dataset ────────────────────────────
print("\n[TEST 3] get_papers_by_dataset('ImageNet')")
results = get_papers_by_dataset("ImageNet")
print(f"  Found {len(results)} papers:")
for r in results[:5]:
    print(f"    {r['title'][:55]} ({r['year']})")

# ── Test 4a: get_citation_chain — descendants ─────────────────
print("\n[TEST 4a] get_citation_chain('2010.11929', direction='descendants')")
print("  (Papers that extend ViT)")
results = get_citation_chain(
    arxiv_id="2010.11929",
    hops=3,
    direction="descendants"
)
print(f"  Found {len(results)} descendants:")
for r in results:
    print(f"    {r['title'][:60]} ({r['year']})")

# ── Test 4b: get_citation_chain — ancestors ───────────────────
print("\n[TEST 4b] get_citation_chain('2010.11929', direction='ancestors')")
print("  (Papers that ViT extends)")
results = get_citation_chain(
    arxiv_id="2010.11929",
    hops=2,
    direction="ancestors"
)
print(f"  Found {len(results)} ancestors:")
for r in results:
    print(f"    {r['title'][:60]} ({r['year']})")

# ── Test 5: find_papers_sharing_concepts ─────────────────────
print("\n[TEST 5] find_papers_sharing_concepts('2010.11929')")
print("  (Papers most conceptually similar to ViT)")
results = find_papers_sharing_concepts("2010.11929", min_shared=1)
print(f"  Found {len(results)} related papers:")
for r in results[:5]:
    print(f"    {r['title'][:50]} | shared: {r['shared_concepts'][:3]}")

# ── Test 6: find_contradicting_papers ───────────────────────
print("\n[TEST 6] find_contradicting_papers('2010.11929')")
results = find_contradicting_papers("2010.11929")
print(f"  Found {len(results)} contradiction candidates:")
for r in results[:3]:
    print(f"    {r['title'][:50]} | type: {r['contradiction_type']}")

# ── Test 7: get_paper_neighborhood ───────────────────────────
print("\n[TEST 7] get_paper_neighborhood('2103.14030') — Swin Transformer")
neighborhood = get_paper_neighborhood("2103.14030")
print(f"  Methods:    {neighborhood['methods'][:4]}")
print(f"  Datasets:   {neighborhood['datasets'][:4]}")
print(f"  Concepts:   {neighborhood['concepts'][:4]}")
print(f"  Tasks:      {neighborhood['tasks'][:3]}")
print(f"  Extends:    {[p['title'][:35] for p in neighborhood['extends']]}")
print(f"  ExtendedBy: {[p['title'][:35] for p in neighborhood['extended_by']]}")

# ── Test 8: get_all_entity_names ─────────────────────────────
print("\n[TEST 8] get_all_entity_names() — full entity vocabulary")
vocab = get_all_entity_names()
print(f"  Methods ({len(vocab['methods'])}): {vocab['methods'][:5]}...")
print(f"  Datasets ({len(vocab['datasets'])}): {vocab['datasets'][:5]}...")
print(f"  Concepts ({len(vocab['concepts'])}): {vocab['concepts'][:5]}...")
print(f"  Tasks ({len(vocab['tasks'])}): {vocab['tasks'][:5]}...")
print(f"  Papers ({len(vocab['papers'])}): {[p['title'][:30] for p in vocab['papers'][:3]]}...")

# ── Multi-hop demo: the key differentiator ────────────────────
print("\n" + "="*70)
print("MULTI-HOP DEMO — Why the graph beats naive vector search")
print("="*70)
print("\nQuery: 'What dataset was used to evaluate methods from papers extending ViT?'")
print("This requires 3 hops: ViT → [EXTENDS] → Papers → [USES] → Methods")
print("                                        → [EVALUATES_ON] → Datasets\n")

from graph.neo4j_client import get_neo4j_client
client = get_neo4j_client()

result = client.run("""
    MATCH (vit:Paper {arxiv_id: '2010.11929'})
    MATCH (p:Paper)-[:EXTENDS]->(vit)
    MATCH (p)-[:USES]->(m:Method)
    MATCH (p)-[:EVALUATES_ON]->(d:Dataset)

    RETURN
        p.title AS paper,
        p.year AS year,
        collect(DISTINCT m.name)[..3] AS methods,
        collect(DISTINCT d.name) AS datasets

    ORDER BY year
""")

for row in result:
    print(f"  Paper: {row['paper'][:55]}")
    print(f"    Methods:  {row['methods']}")
    print(f"    Datasets: {row['datasets']}")
    print()

print("✓ A flat vector search cannot answer this in one shot.")
print("✓ The graph traversal connects all 3 hops in one Cypher query.")

print("\n" + "="*70)
print("✓ All graph query tests complete")
print("="*70)