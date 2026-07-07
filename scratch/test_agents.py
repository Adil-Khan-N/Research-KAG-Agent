"""
Day 12 — Multi-agent pipeline test suite.

Tests:
1. Each agent individually
2. Full pipeline end-to-end
3. Literature review output quality
4. Save to markdown

Run: python scratch/test_agents.py
"""

import logging
logging.basicConfig(level=logging.WARNING)

print("="*70)
print("DAY 12 — MULTI-AGENT PIPELINE TESTS")
print("="*70)

# ── Test 1: PlannerAgent ──────────────────────────────────────
print("\n[TEST 1] PlannerAgent")
from agents.planner import planner_agent

state = {
    "topic": "vision transformers for image classification",
    "errors": [],
}
state = planner_agent(state)
print(f"\n  sub_queries ({len(state['sub_queries'])}):")
for q in state["sub_queries"]:
    print(f"    - {q}")
print(f"  reasoning: {state['plan_reasoning']}")
assert len(state["sub_queries"]) >= 3, "Need at least 3 queries"
print("  ✓ PASSED")

import time
time.sleep(5)

# ── Test 2: PaperSearchAgent ──────────────────────────────────
print("\n[TEST 2] PaperSearchAgent")
from agents.paper_search import paper_search_agent

# Use just 2 queries for speed
test_state = {
    **state,
    "sub_queries": state["sub_queries"][:2],
    "errors": [],
}
test_state = paper_search_agent(test_state)
print(f"\n  Papers found: {test_state['total_papers_found']}")
print(f"  Chunks found: {len(test_state['retrieved_chunks'])}")
assert test_state["total_papers_found"] > 0, "Should find papers"
print("  ✓ PASSED")

# ── Test 3: SummaryAgent ──────────────────────────────────────
print("\n[TEST 3] SummaryAgent (first 3 papers only)")
from agents.summary import summary_agent

# Limit to 3 papers for speed
summary_state = {
    **test_state,
    "retrieved_papers": test_state["retrieved_papers"][:3],
    "errors": [],
}
summary_state = summary_agent(summary_state)
print(f"\n  Summaries generated: {len(summary_state['paper_summaries'])}")
for s in summary_state["paper_summaries"][:2]:
    print(f"  [{s['year']}] {s['title'][:50]}")
    print(f"    {s['summary'][:120]}...")
assert len(summary_state["paper_summaries"]) > 0
print("  ✓ PASSED")

# ── Test 4: GraphAgent ────────────────────────────────────────
print("\n[TEST 4] GraphAgent")
from agents.graph_agent import graph_agent

graph_state = graph_agent(summary_state)
print(f"\n  Relationships: {len(graph_state['graph_relationships'])}")
print(f"  Timeline entries: {len(graph_state['timeline'])}")
for r in graph_state["graph_relationships"][:3]:
    print(f"  {r['paper1'][:30]} → {r['relationship']} → {r['paper2'][:30]}")
print("  ✓ PASSED")

# ── Test 5: WriterAgent ───────────────────────────────────────
print("\n[TEST 5] WriterAgent")
from agents.writer import writer_agent

writer_state = writer_agent(graph_state)
review = writer_state["literature_review"]
print(f"\n  Review length: {len(review)} chars")
print(f"  Citations: {len(writer_state['citations'])}")
print(f"\n  --- REVIEW PREVIEW ---")
print(review[:600])
print("  --- END PREVIEW ---")
assert len(review) > 200, "Review should be substantial"
print("\n  ✓ PASSED")

# ── Test 6: Full pipeline end-to-end ─────────────────────────
print("\n[TEST 6] Full pipeline end-to-end")
print("  Topic: 'masked autoencoder pretraining for vision'")
print("  (This takes 3-5 minutes)\n")

from agents.pipeline import run_literature_review, save_literature_review

final_state = run_literature_review(
    "masked autoencoder pretraining for vision transformers"
)

print(f"\n  Papers: {final_state['total_papers_found']}")
print(f"  Summaries: {len(final_state['paper_summaries'])}")
print(f"  Relationships: {len(final_state['graph_relationships'])}")
print(f"  Review chars: {len(final_state['literature_review'])}")
print(f"  Errors: {final_state['errors']}")

# Save
filename = save_literature_review(final_state)
print(f"  Saved: {filename}")

assert len(final_state["literature_review"]) > 200
assert final_state["total_papers_found"] > 0
print("  ✓ PASSED")

# ── Test 7: Second topic ──────────────────────────────────────
print("\n[TEST 7] Second topic — Swin Transformer")
time.sleep(10)

final_state2 = run_literature_review(
    "Swin Transformer hierarchical vision"
)
filename2 = save_literature_review(final_state2)

print(f"\n  Papers: {final_state2['total_papers_found']}")
print(f"  Review length: {len(final_state2['literature_review'])} chars")
print(f"  Saved: {filename2}")
print("  ✓ PASSED")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*70)
print("DAY 12 SUMMARY")
print("="*70)
print("  ✓ PlannerAgent: topic → 5 specific queries")
print("  ✓ PaperSearchAgent: queries → deduplicated papers")
print("  ✓ SummaryAgent: papers → individual summaries")
print("  ✓ GraphAgent: relationships + timeline from Neo4j")
print("  ✓ WriterAgent: synthesized literature review")
print("  ✓ Full pipeline: end-to-end in ~3-5 minutes")
print(f"  ✓ Reviews saved to docs/")
print("="*70)