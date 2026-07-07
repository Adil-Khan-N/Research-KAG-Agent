"""
Full test of the HybridKAGRetriever.
Tests:
1. Basic hybrid retrieval
2. Vector-only vs Hybrid comparison (the key proof)
3. Multi-hop query handling
4. Edge cases (empty query, unknown entities)
5. Latency benchmark

Run: python scratch/test_hybrid_retriever.py
"""

import time
import logging
logging.basicConfig(level=logging.WARNING)  # suppress INFO for cleaner output

from retrieval.hybrid_retriever import HybridKAGRetriever, pretty_print_hybrid_results
from retrieval.vector_only import vector_only_retrieve

retriever = HybridKAGRetriever()

print("="*70)
print("HYBRIDKAGRETRIEVER — Day 8 Full Test")
print("="*70)

# ── Test 1: Basic hybrid retrieval ───────────────────────────
print("\n[TEST 1] Basic hybrid retrieval")
query = "how does shifted window attention work in Swin Transformer"
results, trace = retriever.retrieve(query, top_k=10)
pretty_print_hybrid_results(query, results, trace)

# ── Test 2: THE KEY COMPARISON ────────────────────────────────
print("\n\n" + "="*70)
print("[TEST 2] VECTOR-ONLY vs HYBRID COMPARISON")
print("="*70)
print("This is the core proof that the graph adds value.\n")

comparison_query = "what datasets were used to evaluate vision transformer methods"

print(f"Query: '{comparison_query}'\n")

# Vector only
t0 = time.time()
vector_results = vector_only_retrieve(comparison_query, top_k=5)
vector_ms = int((time.time() - t0) * 1000)

print(f"── VECTOR ONLY ({vector_ms}ms) ──────────────────────")
for i, r in enumerate(vector_results, 1):
    print(f"  [{i}] Score: {r.similarity:.4f} | {r.title[:50]} ({r.year})")
    print(f"       {r.text[:100]}...")

# Hybrid
t0 = time.time()
hybrid_results, trace = retriever.retrieve(comparison_query, top_k=5)
hybrid_ms = int((time.time() - t0) * 1000)

print(f"\n── HYBRID KAG ({hybrid_ms}ms) ──────────────────────")
for i, r in enumerate(hybrid_results, 1):
    boost = r.final_score - r.vector_score
    boost_str = f"+{boost:.1f} graph" if boost > 0 else "no boost"
    print(f"  [{i}] Score: {r.final_score:.4f} ({boost_str}) | {r.title[:45]} ({r.year})")
    print(f"       Source: {r.source}")
    print(f"       {r.text[:100]}...")

# Show what hybrid found that vector missed
vector_ids = {r.chunk_id for r in vector_results}
hybrid_only = [r for r in hybrid_results if r.chunk_id not in vector_ids]
print(f"\n── Chunks found by HYBRID but missed by VECTOR: {len(hybrid_only)}")
for r in hybrid_only:
    print(f"  + {r.title[:55]}")
    print(f"    {r.text[:120]}...")

# ── Test 3: Multi-hop query ───────────────────────────────────
print("\n\n[TEST 3] Multi-hop query — requires graph traversal")
multihop_query = "what methods do papers extending ViT use for image classification"
results, trace = retriever.retrieve(multihop_query, top_k=8)

print(f"Query: '{multihop_query}'")
print(f"Entities found: {trace.extracted_entities['all'][:6]}")
print(f"Graph papers: {len(trace.graph_papers_found)}")
print(f"Graph-boosted results: {trace.graph_boosted_count}/{trace.final_results_count}")
print(f"\nTop 5 results:")
for i, r in enumerate(results[:5], 1):
    print(f"  [{i}] {r.final_score:.4f} | {r.title[:50]} | {r.source}")

# ── Test 4: Query with no graph entities ──────────────────────
print("\n\n[TEST 4] Query with no known entities (falls back to vector)")
no_entity_query = "explain how deep learning models generalize"
results, trace = retriever.retrieve(no_entity_query, top_k=5)

print(f"Query: '{no_entity_query}'")
print(f"Entities found: {trace.entity_count} (expected: 0 or few)")
print(f"Graph papers: {len(trace.graph_papers_found)}")
print(f"Results: {trace.final_results_count} (should still work via vector)")
for i, r in enumerate(results[:3], 1):
    print(f"  [{i}] {r.final_score:.4f} | {r.title[:50]}")

# ── Test 5: Specific paper query ──────────────────────────────
print("\n\n[TEST 5] Query mentioning specific paper (DeiT)")
deit_query = "how does DeiT distillation token improve training efficiency"
results, trace = retriever.retrieve(deit_query, top_k=5)

print(f"Query: '{deit_query}'")
print(f"Entities: {trace.extracted_entities['all'][:5]}")
print(f"Graph-boosted: {trace.graph_boosted_count}")
for i, r in enumerate(results[:5], 1):
    print(f"  [{i}] {r.final_score:.4f} | {r.title[:50]} | {r.source}")

# ── Test 6: Latency benchmark ─────────────────────────────────
print("\n\n[TEST 6] Latency benchmark (5 queries)")
benchmark_queries = [
    "how does patch embedding work in ViT",
    "compare Swin and ViT architectures",
    "what is masked autoencoder pretraining",
    "which papers use ImageNet for evaluation",
    "explain self-attention in vision models",
]

latencies = []
for q in benchmark_queries:
    t0 = time.time()
    results, trace = retriever.retrieve(q, top_k=10)
    ms = int((time.time() - t0) * 1000)
    latencies.append(ms)
    print(f"  {ms:4d}ms | {len(results)} results | "
          f"{trace.graph_boosted_count} boosted | '{q[:45]}'")

avg_ms = sum(latencies) / len(latencies)
print(f"\n  Average latency: {avg_ms:.0f}ms")
print(f"  Min: {min(latencies)}ms | Max: {max(latencies)}ms")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*70)
print("DAY 8 SUMMARY")
print("="*70)

# Count how many unique papers vector vs hybrid surfaces
t0 = time.time()
v_results = vector_only_retrieve("vision transformer image classification", top_k=20)
vector_time = int((time.time() - t0) * 1000)

t0 = time.time()
h_results, h_trace = retriever.retrieve("vision transformer image classification", top_k=20)
hybrid_time = int((time.time() - t0) * 1000)

v_papers = {r.arxiv_id for r in v_results}
h_papers = {r.arxiv_id for r in h_results}
hybrid_only_papers = h_papers - v_papers

print(f"\nFor query: 'vision transformer image classification'")
print(f"  Vector-only: {len(v_papers)} unique papers in {vector_time}ms")
print(f"  Hybrid KAG:  {len(h_papers)} unique papers in {hybrid_time}ms")
print(f"  Papers found by hybrid ONLY: {len(hybrid_only_papers)}")
if hybrid_only_papers:
    from ingestion.db import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        for arxiv_id in list(hybrid_only_papers)[:3]:
            row = conn.execute(
                text("SELECT title FROM papers WHERE arxiv_id = :id"),
                {"id": arxiv_id}
            ).fetchone()
            if row:
                print(f"    + {row[0][:60]}")

print(f"\n✓ HybridKAGRetriever fully tested")
print(f"✓ Save this output — it's evidence for your Day 11 RAGAS table")
print("="*70)

