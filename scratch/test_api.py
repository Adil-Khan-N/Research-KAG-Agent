"""
Full API test suite for Day 10.
Tests all endpoints using the requests library (no server needed for basic tests).
Also tests via actual HTTP calls when server is running.

Run FIRST (without server):  python scratch/test_api.py --no-server
Run AFTER starting server:   python scratch/test_api.py
"""

import sys
import json
import time
import requests

BASE_URL = "http://localhost:8000"
NO_SERVER = "--no-server" in sys.argv

print("="*70)
print("DAY 10 — API TEST SUITE")
print("="*70)

if NO_SERVER:
    print("\nRunning in --no-server mode (testing route logic directly)\n")

    # Test route logic directly without HTTP
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Test 1: Query endpoint logic
    print("[TEST 1] Query endpoint — direct call")
    from retrieval.pipeline import run_pipeline
    result = run_pipeline(
        "How does patch embedding work in ViT?",
        top_k_retrieve=20,
        top_k_rerank=8,
    )
    print(f"  Query: How does patch embedding work in ViT?")
    print(f"  Confidence: {result.answer.confidence}")
    print(f"  Citations: {len(result.answer.citations)}")
    print(f"  Latency: {result.total_latency_ms}ms")
    import re
    clean = re.sub(r'\nCITATIONS:.*', '', result.answer.answer, flags=re.DOTALL)
    print(f"  Answer: {clean[:250].strip()}...")

    # Test 2: Graph explore logic
    print("\n[TEST 2] Graph explore — direct call")
    from graph.graph_queries import get_paper_neighborhood
    neighborhood = get_paper_neighborhood("2010.11929")
    print(f"  Paper: ViT (2010.11929)")
    print(f"  Methods: {neighborhood['methods'][:3]}")
    print(f"  Datasets: {neighborhood['datasets']}")
    print(f"  Extends: {[p['title'][:30] for p in neighborhood['extends']]}")
    print(f"  Extended by: {len(neighborhood['extended_by'])} papers")

    # Test 3: Stats
    print("\n[TEST 3] Stats endpoint — direct call")
    from ingestion.db import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        papers = conn.execute(text("SELECT COUNT(*) FROM papers")).fetchone()[0]
        chunks = conn.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()[0]
        logs = conn.execute(text("SELECT COUNT(*) FROM query_logs")).fetchone()[0]
    print(f"  Papers: {papers}")
    print(f"  Chunks: {chunks}")
    print(f"  Query logs: {logs}")

    # Test 4: Health check logic
    print("\n[TEST 4] Health check — direct call")
    from ingestion.db import engine as pg_engine
    from graph.neo4j_client import get_neo4j_client
    neo4j = get_neo4j_client()
    print(f"  Postgres: ✓ ({papers} papers)")
    print(f"  Neo4j: {'✓' if neo4j.test_connection() else '✗'}")

    print("\n" + "="*70)
    print("Direct tests complete.")
    print("\nTo test via HTTP, start the server first:")
    print("  uvicorn api.main:app --reload --port 8000")
    print("Then run:")
    print("  python scratch/test_api.py")
    print("="*70)
    sys.exit(0)


# ── HTTP Tests (server must be running) ───────────────────────

def check_server():
    """Check if server is running."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


if not check_server():
    print("\nERROR: Server not running.")
    print("Start it with: uvicorn api.main:app --reload --port 8000")
    print("Then re-run this script.")
    sys.exit(1)

print(f"\nServer is running at {BASE_URL}\n")

# ── Test 1: Root endpoint ─────────────────────────────────────
print("[TEST 1] GET / — root endpoint")
r = requests.get(f"{BASE_URL}/")
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
data = r.json()
print(f"  Status: {r.status_code}")
print(f"  Message: {data['message']}")
print(f"  Endpoints: {data['endpoints']}")
print("  ✓ PASSED")

# ── Test 2: Health check ──────────────────────────────────────
print("\n[TEST 2] GET /health")
r = requests.get(f"{BASE_URL}/health")
assert r.status_code == 200
data = r.json()
print(f"  Status: {data['status']}")
print(f"  Postgres: {data['postgres']} ({data['paper_count']} papers, {data['chunk_count']} chunks)")
print(f"  Neo4j: {data['neo4j']}")
print(f"  Embedding model: {data['embedding_model']}")
assert data["postgres"], "Postgres should be healthy"
assert data["neo4j"], "Neo4j should be healthy"
print("  ✓ PASSED")

# ── Test 3: List papers ───────────────────────────────────────
print("\n[TEST 3] GET /papers")
r = requests.get(f"{BASE_URL}/papers")
assert r.status_code == 200
data = r.json()
print(f"  Total papers: {data['total']}")
for p in data["papers"][:3]:
    print(f"    [{p['year']}] {p['title'][:50]} ({p['total_chunks']} chunks)")
assert data["total"] >= 20, f"Expected 20+ papers, got {data['total']}"
print("  ✓ PASSED")

# ── Test 4: Stats endpoint ────────────────────────────────────
print("\n[TEST 4] GET /stats")
r = requests.get(f"{BASE_URL}/stats")
assert r.status_code == 200
data = r.json()
print(f"  Postgres — papers: {data['postgres']['papers']}, chunks: {data['postgres']['chunks']}")
print(f"  Neo4j nodes: {data['neo4j']['nodes']}")
print(f"  Neo4j relationships: {data['neo4j']['relationships']}")
print("  ✓ PASSED")

# ── Test 5: Query endpoint — basic ───────────────────────────
print("\n[TEST 5] POST /query — basic query")
payload = {
    "query": "What is the key innovation of Vision Transformer ViT?",
    "top_k_retrieve": 20,
    "top_k_rerank": 8,
    "use_graph": True,
}
t0 = time.time()
r = requests.post(f"{BASE_URL}/query", json=payload)
elapsed = int((time.time() - t0) * 1000)
assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
data = r.json()
print(f"  Status: {r.status_code} ({elapsed}ms round-trip)")
print(f"  Confidence: {data['confidence']}")
print(f"  Citations: {len(data['citations'])}")
print(f"  Graph papers found: {data['graph_papers_found']}")
print(f"  Graph boosted: {data['graph_boosted_count']}")
print(f"  Answer preview: {data['answer'][:200]}...")
if data["citations"]:
    print(f"  First citation: {data['citations'][0]['title'][:50]}")
print("  ✓ PASSED")

# ── Test 6: Query endpoint — vector only ─────────────────────
print("\n[TEST 6] POST /query — vector only (use_graph=False)")
payload = {
    "query": "How does self-attention work in transformers?",
    "use_graph": False,
    "pipeline_variant": "vector_only",
}
r = requests.post(f"{BASE_URL}/query", json=payload)
assert r.status_code == 200
data = r.json()
print(f"  Graph papers found: {data['graph_papers_found']} (expected: 0)")
print(f"  Graph boosted: {data['graph_boosted_count']} (expected: 0)")
print(f"  Confidence: {data['confidence']}")
print(f"  Pipeline variant: {data['pipeline_variant']}")
assert data["graph_papers_found"] == 0
print("  ✓ PASSED")

# ── Test 7: Query validation ──────────────────────────────────
print("\n[TEST 7] POST /query — input validation")

# Empty query
r = requests.post(f"{BASE_URL}/query", json={"query": ""})
print(f"  Empty query → {r.status_code} (expected 422)")
assert r.status_code == 422

# Too short
r = requests.post(f"{BASE_URL}/query", json={"query": "hi"})
print(f"  Too short → {r.status_code} (expected 422)")
assert r.status_code == 422

print("  ✓ PASSED")

# ── Test 8: Graph explore — paper ────────────────────────────
print("\n[TEST 8] GET /graph-explore — ViT paper")
r = requests.get(
    f"{BASE_URL}/graph-explore",
    params={"entity": "2010.11929", "entity_type": "paper"}
)
assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
data = r.json()
print(f"  Entity: {data['entity']}")
print(f"  Nodes: {len(data['nodes'])}")
print(f"  Edges: {len(data['edges'])}")
print(f"  Summary: {data['neighborhood_summary']}")
node_types = [n["type"] for n in data["nodes"]]
print(f"  Node types: {set(node_types)}")
assert len(data["nodes"]) > 0, "Should have nodes"
assert len(data["edges"]) > 0, "Should have edges"
print("  ✓ PASSED")

# ── Test 9: Graph explore — dataset ──────────────────────────
print("\n[TEST 9] GET /graph-explore — ImageNet dataset")
r = requests.get(
    f"{BASE_URL}/graph-explore",
    params={"entity": "ImageNet", "entity_type": "dataset"}
)
assert r.status_code == 200
data = r.json()
print(f"  Entity: {data['entity']} ({data['entity_type']})")
print(f"  Connected nodes: {len(data['nodes'])}")
print(f"  Summary: {data['neighborhood_summary']}")
print("  ✓ PASSED")

# ── Test 10: Query logs ───────────────────────────────────────
print("\n[TEST 10] GET /query-logs")
r = requests.get(f"{BASE_URL}/query-logs", params={"limit": 5})
assert r.status_code == 200
data = r.json()
print(f"  Total logs: {data['total']}")
for log in data["logs"][:3]:
    print(f"    [{log['id']}] {log['query'][:50]} "
          f"| {log['latency_ms']}ms | {log['pipeline_variant']}")
print("  ✓ PASSED")

# ── Test 11: 404 for unknown paper ───────────────────────────
print("\n[TEST 11] GET /graph-explore — unknown paper")
r = requests.get(
    f"{BASE_URL}/graph-explore",
    params={"entity": "9999.99999", "entity_type": "paper"}
)
print(f"  Unknown paper → {r.status_code} (expected 404)")
assert r.status_code == 404
print("  ✓ PASSED")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*70)
print("DAY 10 API TEST SUMMARY")
print("="*70)
print("  All 11 tests passed ✓")
print(f"\n  API docs available at: {BASE_URL}/docs")
print(f"  ReDoc available at:   {BASE_URL}/redoc")
print("="*70)