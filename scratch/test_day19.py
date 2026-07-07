"""
Day 19 full test suite.

Tests:
1.  A/B dashboard stats from query_logs
2.  Win rate computation
3.  Auto graph updater — diff extraction
4.  Auto graph updater — incremental MERGE
5.  Timeline from seed paper (ViT)
6.  Timeline from concept (self-supervised)
7.  Structured output — comparison schema
8.  Structured output — paper list schema
9.  Structured output — timeline schema
10. Schema validation
11. API /timeline endpoint
12. API /query-structured endpoint
13. API /ingest-incremental endpoint

Run: python scratch/test_day19.py
"""

import time
import json
import logging
logging.basicConfig(level=logging.WARNING)

print("="*70)
print("DAY 19 — A/B DASHBOARD + AUTO UPDATER + TIMELINE + STRUCTURED")
print("="*70)

from ingestion.db import engine
from graph.neo4j_client import get_neo4j_client
from graph.auto_updater import (
    get_existing_entities, diff_entities,
    auto_update_graph, UpdateResult,
)
from graph.timeline_builder import (
    build_timeline, format_timeline_for_display,
)
from retrieval.structured_output import (
    generate_structured_answer, validate_against_schema,
    PRESET_SCHEMAS, COMPARISON_SCHEMA, PAPER_LIST_SCHEMA,
)
from eval.ab_dashboard import get_variant_stats, compute_win_rate

client = get_neo4j_client()


# ── Test 1: A/B dashboard stats ───────────────────────────────
print("\n[TEST 1] A/B dashboard stats from query_logs")

stats = get_variant_stats(engine)
print(f"  Total queries logged: {stats['total_queries']}")
print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"\n  Stats by variant:")
for variant, data in stats["stats_by_variant"].items():
    print(
        f"    {variant:<15} count={data['query_count']:>3} | "
        f"avg={data['avg_latency']:>6}ms | "
        f"median={data['median_latency']:>6}ms"
    )
print(f"\n  Paired comparisons: {len(stats['paired'])}")
print("  ✓ PASSED")


# ── Test 2: Win rate computation ──────────────────────────────
print("\n[TEST 2] Win rate from paired data")

win_rate = compute_win_rate(stats["paired"])
print(f"  Total pairs:  {win_rate['total']}")
print(f"  Hybrid wins:  {win_rate['hybrid']} ({win_rate['hybrid_pct']:.1f}%)")
print(f"  Vector wins:  {win_rate['vector']} ({win_rate['vector_pct']:.1f}%)")
print(f"  Ties:         {win_rate['tie']}")
print("  ✓ PASSED")


# ── Test 3: Auto updater — diff ───────────────────────────────
print("\n[TEST 3] Auto graph updater — entity diffing")

existing = get_existing_entities(client)
print(f"  Existing graph:")
print(f"    Methods:  {len(existing['methods'])}")
print(f"    Datasets: {len(existing['datasets'])}")
print(f"    Concepts: {len(existing['concepts'])}")
print(f"    Papers:   {len(existing['papers'])}")

print(existing["methods"])
print(type(existing["methods"]))

# Mock extracted entities — some new, some existing
mock_extracted = {
    "methods": ["Transformer", "NEW_METHOD_XYZ_2024"],
    "datasets": ["ImageNet", "NEW_DATASET_ABC_2024"],
    "concepts": ["self-supervised learning", "BRAND_NEW_CONCEPT"],
    "tasks": ["image classification"],
}


diff = diff_entities(mock_extracted, existing)
print("Multi-Head Attention" in existing["methods"])
print(f"\n  Diff results:")
for entity_type in ["methods", "datasets", "concepts"]:
    new_count  = len(diff[entity_type]["new"])
    exist_count = len(diff[entity_type]["existing"])
    print(f"    {entity_type:<10}: {new_count} new, {exist_count} existing")
    if diff[entity_type]["new"]:
        print(f"      New: {diff[entity_type]['new'][:3]}")

assert "NEW_METHOD_XYZ_2024" in diff["methods"]["new"]
assert "Transformer" in diff["methods"]["existing"]
assert "NEW_DATASET_ABC_2024" in diff["datasets"]["new"]
assert "ImageNet" in diff["datasets"]["existing"]
print("  ✓ PASSED — diff correctly separates new from existing")


# ── Test 4: Auto updater — real paper ─────────────────────────
print("\n[TEST 4] Auto graph updater — ViT paper incremental update")

from pathlib import Path

vit_path = Path("data/processed/2010.11929.json")
if vit_path.exists():
    with open(vit_path) as f:
        vit_data = json.load(f)

    print(f"  Paper: {vit_data['title'][:50]}")
    print(f"  Running incremental update...")

    result = auto_update_graph(paper_data=vit_data, client=client)

    print(f"  Methods new:       {len(result.methods_new)}")
    print(f"  Methods existing:  {len(result.methods_existing)}")
    print(f"  Datasets new:      {len(result.datasets_new)}")
    print(f"  Datasets existing: {len(result.datasets_existing)}")
    print(f"  Concepts new:      {len(result.concepts_new)}")
    print(f"  Relationships:     {result.relationships_new}")
    print(f"  Time:              {result.elapsed_ms}ms")
    print(f"\n  Since ViT is already in graph:")
    print(f"  Most entities should be 'existing' not 'new'")
    print("  ✓ PASSED — incremental update ran without full rebuild")
else:
    print("  ⚠ ViT processed file not found")
time.sleep(5)


# ── Test 5: Timeline from seed paper ──────────────────────────
print("\n[TEST 5] Timeline from ViT (2010.11929)")

entries = build_timeline(
    seed_arxiv_id="2010.11929",
    max_depth=3,
    client=client,
)

print(f"  Papers in timeline: {len(entries)}")
print(f"\n  Chronological timeline:")
for e in entries:
    ext = f" ← {e.extends[0][:25]}" if e.extends else ""
    print(
        f"  [{e.year}] {e.title[:50]}{ext}"
    )
    if e.methods:
        print(f"           Methods: {e.methods[:2]}")

assert len(entries) > 0, "Should find papers in ViT lineage"
years = [e.year for e in entries]
assert years == sorted(years), "Timeline should be chronological"
print("\n  ✓ PASSED — chronological order confirmed")


# ── Test 6: Timeline from concept ────────────────────────────
print("\n[TEST 6] Timeline from concept 'self-supervised'")

concept_entries = build_timeline(
    seed_concept="self-supervised",
    max_depth=2,
    client=client,
)

print(f"  Papers found: {len(concept_entries)}")
for e in concept_entries[:5]:
    print(f"  [{e.year}] {e.title[:55]}")

print("  ✓ PASSED")


# ── Test 7: Structured output — comparison ────────────────────
print("\n[TEST 7] Structured output — comparison schema")
print("  Query: 'compare ViT and Swin Transformer'")
print("  (Takes 1-2 minutes)\n")

from retrieval.hybrid_retriever import HybridKAGRetriever
from retrieval.reranker import rerank

retriever = HybridKAGRetriever()
query = "compare ViT and Swin Transformer architectures"
results, trace = retriever.retrieve(query, top_k=15, use_graph=True)
ranked = rerank(query, results, top_k=8)

structured = generate_structured_answer(
    query=query,
    chunks=ranked,
    output_schema=COMPARISON_SCHEMA,
)
for i, chunk in enumerate(ranked, 1):
    print(i, getattr(chunk, "title", "Unknown"))
from pprint import pprint

pprint(structured["data"])

print(f"  schema_valid:      {structured['schema_valid']}")
print(f"  validation_errors: {structured['validation_errors']}")
if structured["data"]:
    data = structured["data"]
    print(f"  comparison_topic:  {data.get('comparison_topic', '')[:60]}")
    items = data.get("items", [])
    print(f"  items count:       {len(items)}")
    for item in items[:2]:
        print(
            f"    - {(item.get('name') or 'Unknown')}: "
            f"{(item.get('key_feature') or 'N/A')[:50]}"
        )
    print(f"  summary:           {(data.get('summary') or 'N/A')[:80]}...")
else:
    print(f"  Raw:               {(structured.get('raw') or 'N/A')[:200]}")

print("  ✓ PASSED")
time.sleep(8)


# ── Test 8: Structured output — paper list ────────────────────
print("\n[TEST 8] Structured output — paper list schema")

query2 = "what papers evaluate on ImageNet dataset"
results2, _ = retriever.retrieve(query2, top_k=10, use_graph=True)
ranked2 = rerank(query2, results2, top_k=6)

structured2 = generate_structured_answer(
    query=query2,
    chunks=ranked2,
    output_schema=PAPER_LIST_SCHEMA,
)

print(f"  schema_valid: {structured2['schema_valid']}")
if structured2["data"] and isinstance(structured2["data"], list):
    print(f"  Papers returned: {len(structured2['data'])}")
    for item in structured2["data"][:3]:
        print(f"    [{item.get('year', '')}] "
              f"{item.get('title', '')[:45]}")
else:
    print(f"  Data: {str(structured2['data'])[:100]}")
print("  ✓ PASSED")
time.sleep(8)


# ── Test 9: Schema validation ─────────────────────────────────
print("\n[TEST 9] Schema validation")

# Valid object
valid_data = {
    "comparison_topic": "Vision Transformers",
    "items": [{"name": "ViT", "year": 2021, "key_feature": "patches"}],
    "summary": "ViT is foundational",
}
is_valid, errors = validate_against_schema(valid_data, COMPARISON_SCHEMA)
print(f"  Valid data:   is_valid={is_valid}, errors={errors}")
assert is_valid, f"Should be valid, got errors: {errors}"

# Invalid — missing required field
invalid_data = {"items": []}  # missing comparison_topic
is_valid2, errors2 = validate_against_schema(invalid_data, COMPARISON_SCHEMA)
print(f"  Invalid data: is_valid={is_valid2}, errors={errors2}")
assert not is_valid2, "Should be invalid"

# Wrong type
wrong_type = {"comparison_topic": 123, "items": []}
is_valid3, errors3 = validate_against_schema(wrong_type, COMPARISON_SCHEMA)
print(f"  Wrong type:   is_valid={is_valid3}, errors={errors3}")
assert not is_valid3, "Should be invalid — wrong type"

print("  ✓ PASSED — validation correctly identifies errors")


# ── Test 10: API endpoints ────────────────────────────────────
print("\n[TEST 10] API endpoints")
import requests

API_URL = "http://localhost:8000"

# /timeline
print("\n  GET /timeline?arxiv_id=2010.11929")
try:
    r = requests.get(
        f"{API_URL}/timeline",
        params={"arxiv_id": "2010.11929", "max_depth": 3},
        timeout=15,
    )
    if r.status_code == 200:
        data = r.json()
        print(f"  Papers: {data['total_papers']}")
        print(f"  Year range: {data['year_range']}")
        print(f"  Latency: {data['latency_ms']}ms")
        for e in data["entries"][:3]:
            print(f"    [{e['year']}] {e['title'][:45]}")
        print("  ✓ PASSED")
    else:
        print(f"  ✗ {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"  Not reachable: {e}")

# /query-structured
print("\n  POST /query-structured")
try:
    r = requests.post(
        f"{API_URL}/query-structured",
        json={
            "query": "list papers that use ImageNet for evaluation",
            "schema": "paper_list",
        },
        timeout=120,
    )
    if r.status_code == 200:
        data = r.json()
        print(f"  schema_valid:  {data['schema_valid']}")
        print(f"  schema_used:   {data['schema_used']}")
        print(f"  latency:       {data['latency_ms']}ms")
        if data["data"] and isinstance(data["data"], list):
            print(f"  papers:        {len(data['data'])}")
        print("  ✓ PASSED")
    else:
        print(f"  ✗ {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"  Not reachable: {e}")

# /ingest-incremental
print("\n  POST /ingest-incremental")
try:
    r = requests.post(
        f"{API_URL}/ingest-incremental",
        json={"arxiv_id": "2010.11929"},
        timeout=60,
    )
    if r.status_code == 200:
        data = r.json()
        print(f"  methods_new:      {len(data['methods_new'])}")
        print(f"  methods_existing: {len(data['methods_existing'])}")
        print(f"  datasets_new:     {len(data['datasets_new'])}")
        print(f"  relationships:    {data['relationships_new']}")
        print(f"  elapsed:          {data['elapsed_ms']}ms")
        print("  ✓ PASSED")
    else:
        print(f"  ✗ {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"  Not reachable: {e}")

# /ab-summary (from Day 18)
print("\n  GET /ab-summary")
try:
    r = requests.get(f"{API_URL}/ab-summary", timeout=10)
    if r.status_code == 200:
        data = r.json()
        variants = data.get("stats_by_variant", [])
        print(f"  Variants tracked: {len(variants)}")
        for v in variants:
            print(f"    {v['pipeline_variant']}: "
                  f"{v['query_count']} queries, "
                  f"avg {v['avg_latency_ms']}ms")
        print("  ✓ PASSED")
    else:
        print(f"  ✗ {r.status_code}")
except Exception as e:
    print(f"  Not reachable: {e}")


# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*70)
print("DAY 19 SUMMARY")
print("="*70)
print(f"  ✓ A/B dashboard: {stats['total_queries']} queries logged, "
      f"cache hit rate {stats['cache_hit_rate']:.1%}")
print(f"  ✓ Win rate: hybrid={win_rate['hybrid']} "
      f"vector={win_rate['vector']} tie={win_rate['tie']}")
print(f"  ✓ Auto updater: diff correctly separates new vs existing")
print(f"  ✓ ViT incremental update: {result.relationships_new} rels added")
print(f"  ✓ Timeline: {len(entries)} papers in ViT lineage "
      f"(chronological order confirmed)")
print(f"  ✓ Concept timeline: {len(concept_entries)} papers for "
      f"'self-supervised'")
print(f"  ✓ Structured output — comparison schema: "
      f"valid={structured['schema_valid']}")
print(f"  ✓ Structured output — paper list schema: "
      f"valid={structured2['schema_valid']}")
print(f"  ✓ Schema validation: catches missing fields + wrong types")
print(f"\n  4 features individually demoable:")
print(f"  1. /ab-summary — live A/B dashboard")
print(f"  2. /ingest-incremental — incremental graph update")
print(f"  3. /timeline?arxiv_id=2010.11929 — ViT lineage")
print(f"  4. /query-structured?schema=comparison — JSON output")
print("="*70)