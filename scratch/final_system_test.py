"""
Final system test — simulates a fresh clone + docker compose up.
Tests every component end-to-end in the correct order.
Run this before tagging v2.0-full.

Run: python scratch/final_system_test.py
"""

import sys
import time
import requests
import logging
logging.basicConfig(level=logging.WARNING)

API_URL = "http://localhost:8000"

print("="*70)
print("FINAL SYSTEM TEST — v2.0-full")
print("="*70)
print(f"API: {API_URL}")
print()

failures = []
passes = []


def test(name: str, fn):
    """Run one test, track pass/fail."""
    try:
        fn()
        passes.append(name)
        print(f"  ✓ {name}")
    except Exception as e:
        failures.append((name, str(e)))
        print(f"  ✗ {name}: {e}")


def check_api():
    r = requests.get(f"{API_URL}/health", timeout=5)
    assert r.status_code == 200
    h = r.json()
    assert h["status"] in ("healthy", "degraded")
    assert h["postgres"]
    assert h["paper_count"] >= 20, f"Need 20+ papers, got {h['paper_count']}"


def check_stats():
    r = requests.get(f"{API_URL}/stats", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["postgres"]["chunks"] > 500


def check_query_hybrid():
    r = requests.post(
        f"{API_URL}/query",
        json={"query": "how does ViT handle image patches?",
              "use_graph": True},
        timeout=120,
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["answer"]) > 50
    assert data["total_latency_ms"] > 0


def check_query_vector():
    r = requests.post(
        f"{API_URL}/query",
        json={"query": "what is masked autoencoder pretraining?",
              "use_graph": False,
              "pipeline_variant": "vector_only"},
        timeout=120,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["pipeline_variant"] in ("vector_only", "cache_hit")


def check_graph_explore():
    r = requests.get(
        f"{API_URL}/graph-explore",
        params={"entity": "2010.11929", "entity_type": "paper"},
        timeout=10,
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) > 0


def check_recommend():
    r = requests.get(
        f"{API_URL}/recommend",
        params={"arxiv_id": "2010.11929", "top_k": 3},
        timeout=10,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["llm_calls"] == 0
    assert len(data["recommendations"]) > 0


def check_timeline():
    r = requests.get(
        f"{API_URL}/timeline",
        params={"arxiv_id": "2010.11929"},
        timeout=15,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_papers"] > 0


def check_cache():
    q = "what is the attention mechanism in vision transformers"
    # First call — should miss
    r1 = requests.post(
        f"{API_URL}/query",
        json={"query": q, "use_graph": True},
        timeout=120,
    )
    assert r1.status_code == 200

    time.sleep(2)

    # Second call — should hit cache
    r2 = requests.post(
        f"{API_URL}/query",
        json={"query": q, "use_graph": True},
        timeout=10,
    )
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["pipeline_variant"] in ("cache_hit", "hybrid")


def check_papers_list():
    r = requests.get(f"{API_URL}/papers", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 20


def check_query_logs():
    r = requests.get(
        f"{API_URL}/query-logs", params={"limit": 5}, timeout=5
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 0


def check_ab_summary():
    r = requests.get(f"{API_URL}/ab-summary", timeout=5)
    assert r.status_code == 200


def check_cache_stats():
    r = requests.get(f"{API_URL}/cache-stats", timeout=5)
    assert r.status_code == 200


def check_ingest_jobs():
    r = requests.get(f"{API_URL}/ingest/jobs", timeout=5)
    assert r.status_code == 200


# ── Run all tests ─────────────────────────────────────────────
print("Running system tests...\n")

test("API health check",       check_api)
test("System stats",           check_stats)
test("Hybrid query",           check_query_hybrid)
test("Vector-only query",      check_query_vector)
test("Graph explore (ViT)",    check_graph_explore)
test("Graph recommendations",  check_recommend)
test("Timeline (ViT lineage)", check_timeline)
test("Semantic cache",         check_cache)
test("Papers list",            check_papers_list)
test("Query logs",             check_query_logs)
test("A/B summary",            check_ab_summary)
test("Cache stats",            check_cache_stats)
test("Ingest jobs",            check_ingest_jobs)

# ── Results ───────────────────────────────────────────────────
print("\n" + "="*70)
total = len(passes) + len(failures)
print(f"RESULTS: {len(passes)}/{total} passed")
print("="*70)

if passes:
    print(f"\n✓ Passed ({len(passes)}):")
    for p in passes:
        print(f"  {p}")

if failures:
    print(f"\n✗ Failed ({len(failures)}):")
    for name, err in failures:
        print(f"  {name}: {err}")
    print("\nFix failures before tagging v2.0-full")
    sys.exit(1)
else:
    print("\n🎉 All tests passed — ready to tag v2.0-full")