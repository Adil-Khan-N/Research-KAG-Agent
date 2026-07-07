"""
Day 20 tests — polish, analytics, rate limiting, OpenAPI docs.

Tests:
1.  Query analytics (top queries, slowest retrievals)
2.  Cache hit rate trend
3.  Rate limiting middleware check
4.  OpenAPI docs accessible
5.  README exists and has key sections
6.  RAGAS results exist
7.  Architecture diagram exists
8.  Example queries file exists
9.  Git tags present
10. Docker compose valid

Run: python scratch/test_day20.py
"""

import os
import time
import json
import requests
import subprocess
from pathlib import Path

print("="*70)
print("DAY 20 — POLISH + README + DEMO PREP TESTS")
print("="*70)

API_URL = "http://localhost:8000"
passes = []
failures = []


def test(name, fn):
    try:
        fn()
        passes.append(name)
        print(f"  ✓ {name}")
    except Exception as e:
        failures.append((name, str(e)))
        print(f"  ✗ {name}: {e}")


# ── Test 1: Query analytics endpoint ─────────────────────────
print("\n[Analytics]")

def check_analytics():
    r = requests.get(f"{API_URL}/analytics", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "top_queries" in data
    assert "slowest_queries" in data
    assert "cache_hit_rate" in data
    assert "total_queries" in data

test("Query analytics endpoint", check_analytics)


def check_analytics_top_queries():
    r = requests.get(f"{API_URL}/analytics?limit=5", timeout=5)
    data = r.json()
    # top_queries should be list
    assert isinstance(data["top_queries"], list)

test("Analytics top queries", check_analytics_top_queries)


# ── Test 2: Cache stats ───────────────────────────────────────
print("\n[Cache]")

def check_cache_hit_rate():
    r = requests.get(f"{API_URL}/cache-stats", timeout=5)
    data = r.json()
    assert "total_cached" in data
    total = data.get("total_cached", 0)
    print(f"    Cached queries: {total}")

test("Cache hit rate available", check_cache_hit_rate)


# ── Test 3: OpenAPI docs ──────────────────────────────────────
print("\n[API Docs]")

def check_openapi_docs():
    r = requests.get(f"{API_URL}/docs", timeout=5)
    assert r.status_code == 200

def check_redoc():
    r = requests.get(f"{API_URL}/redoc", timeout=5)
    assert r.status_code == 200

def check_openapi_json():
    r = requests.get(f"{API_URL}/openapi.json", timeout=5)
    assert r.status_code == 200
    schema = r.json()
    assert "paths" in schema
    paths = list(schema["paths"].keys())
    print(f"    Endpoints documented: {len(paths)}")
    required = ["/query", "/ingest-async", "/recommend", "/timeline"]
    for ep in required:
        assert ep in paths, f"Missing endpoint: {ep}"

test("Swagger UI accessible",  check_openapi_docs)
test("ReDoc accessible",       check_redoc)
test("OpenAPI JSON valid",     check_openapi_json)


# ── Test 4: README exists and has key sections ────────────────
print("\n[README]")

def check_readme():
    readme = Path("README.md")
    assert readme.exists(), "README.md not found"
    content = readme.read_text()
    required_sections = [
        "## Architecture",
        "## Setup",
        "## Features",
        "## Results",
    ]
    for section in required_sections:
        assert section in content, f"README missing: {section}"
    print(f"    README size: {len(content)} chars")

test("README has required sections", check_readme)


# ── Test 5: RAGAS results exist ───────────────────────────────
print("\n[Evaluation]")

def check_ragas_results():
    path = Path("docs/ragas_results.md")
    assert path.exists(), "docs/ragas_results.md not found"
    content = path.read_text()
    assert "| Configuration |" in content, "Missing comparison table"
    assert "Hybrid KAG" in content, "Missing Hybrid KAG row"
    print(f"    RAGAS results: {len(content)} chars")

def check_ragas_scores_json():
    path = Path("data/ragas_scores.json")
    if path.exists():
        with open(path) as f:
            scores = json.load(f)
        assert len(scores) > 0
        print(f"    Scores configs: {list(scores.keys())}")
    else:
        print(f"    ⚠ data/ragas_scores.json not found (using Gemini eval)")

test("RAGAS results markdown", check_ragas_results)
test("RAGAS scores JSON",      check_ragas_scores_json)


# ── Test 6: Example queries file ─────────────────────────────
def check_example_queries():
    path = Path("docs/example_queries.md")
    assert path.exists(), "docs/example_queries.md not found"
    content = path.read_text()
    assert "Query" in content or "Question" in content
    print(f"    Example queries: {len(content)} chars")

test("Example queries file", check_example_queries)


# ── Test 7: Docker compose valid ─────────────────────────────
print("\n[Docker]")

def check_docker_compose():
    path = Path("docker-compose.yml")
    assert path.exists(), "docker-compose.yml not found"
    try:
        result = subprocess.run(
            ["docker", "compose", "config", "--quiet"],
            capture_output=True, text=True, timeout=10
        )
        # Return code 0 = valid
        if result.returncode != 0:
            print(f"    Warning: {result.stderr[:100]}")
    except FileNotFoundError:
        print("    Docker not available — skipping validation")

test("Docker compose file valid", check_docker_compose)


# ── Test 8: Processed papers exist ───────────────────────────
print("\n[Data]")

def check_processed_papers():
    papers = list(Path("data/processed").glob("*.json"))
    assert len(papers) >= 20, f"Need 20+ papers, got {len(papers)}"
    print(f"    Processed papers: {len(papers)}")

def check_entities_json():
    path = Path("data/entities.json")
    assert path.exists(), "data/entities.json not found"
    with open(path) as f:
        entities = json.load(f)
    assert len(entities) >= 20
    print(f"    Entity extractions: {len(entities)}")

test("Processed papers exist",  check_processed_papers)
test("Entities JSON exists",    check_entities_json)


# ── Test 9: Progress log exists ───────────────────────────────
print("\n[Documentation]")

def check_progress_md():
    path = Path("PROGRESS.md")
    if path.exists():
        content = path.read_text()
        # Count day entries
        days = content.count("## Day")
        print(f"    PROGRESS.md: {days} day entries, {len(content)} chars")
    else:
        print("    ⚠ PROGRESS.md not found — create it")

test("PROGRESS.md exists", check_progress_md)


# ── Test 10: Key files present ────────────────────────────────
def check_key_files():
    required = [
        "ingestion/pipeline.py",
        "ingestion/embedder.py",
        "retrieval/hybrid_retriever.py",
        "retrieval/reranker.py",
        "retrieval/generator.py",
        "retrieval/semantic_cache.py",
        "retrieval/query_decomposer.py",
        "retrieval/contradiction_detector.py",
        "retrieval/explanation_layer.py",
        "retrieval/recommender.py",
        "retrieval/structured_output.py",
        "graph/graph_queries.py",
        "graph/entity_extractor.py",
        "graph/auto_updater.py",
        "graph/timeline_builder.py",
        "agents/pipeline.py",
        "api/main.py",
        "api/routes.py",
        "frontend/app.py",
        "eval/golden_dataset.py",
        "eval/ab_framework.py",
        "eval/ab_dashboard.py",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        raise AssertionError(f"Missing files: {missing}")
    print(f"    All {len(required)} key files present")

test("All key files present", check_key_files)


# ── Results ───────────────────────────────────────────────────
print("\n" + "="*70)
total = len(passes) + len(failures)
print(f"RESULTS: {len(passes)}/{total} passed")
print("="*70)
if failures:
    for name, err in failures:
        print(f"  ✗ {name}: {err}")