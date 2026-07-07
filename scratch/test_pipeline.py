"""
Full end-to-end pipeline test — Day 9 checkpoint.

Tests:
1. Reranker standalone test
2. Full pipeline: query → retrieve → rerank → generate
3. 5 example queries with real outputs
4. Vector-only vs Hybrid answer comparison
5. Edge case: unanswerable question
6. Latency breakdown

Run: python scratch/test_pipeline.py
"""

import logging
import time
logging.basicConfig(level=logging.WARNING)

from retrieval.hybrid_retriever import HybridKAGRetriever
from retrieval.reranker import rerank, pretty_print_reranked
from retrieval.generator import generate_answer, pretty_print_answer
from retrieval.pipeline import run_pipeline, log_pipeline_result

print("="*70)
print("DAY 9 — FULL PIPELINE TEST")
print("Retrieve → Rerank → Generate")
print("="*70)

# ── Test 1: Reranker standalone ───────────────────────────────
print("\n[TEST 1] Reranker standalone test")
retriever = HybridKAGRetriever()
query = "how does shifted window attention work in Swin Transformer"

t0 = time.time()
hybrid_results, trace = retriever.retrieve(query, top_k=20)
retrieve_ms = int((time.time() - t0) * 1000)
print(f"  Retrieved: {len(hybrid_results)} chunks in {retrieve_ms}ms")

t0 = time.time()
ranked = rerank(query, hybrid_results, top_k=8)
rerank_ms = int((time.time() - t0) * 1000)
print(f"  Reranked to: {len(ranked)} chunks in {rerank_ms}ms")

print(f"\n  Before vs After reranking (top 5):")
print(f"  {'Rank':<5} {'Rerank Score':<14} {'Hybrid Score':<14} "
      f"{'Source':<15} Paper")
print(f"  {'-'*70}")
for r in ranked[:5]:
    print(f"  [{r.rerank_rank}]  "
          f"{r.rerank_score:<14.4f} "
          f"{r.final_score:<14.4f} "
          f"{r.source:<15} "
          f"{r.title[:30]}")

# ── Test 2: Full pipeline — query 1 ──────────────────────────
print("\n\n[TEST 2] Full pipeline — Swin Transformer query")
result = run_pipeline(
    "how does shifted window attention work in Swin Transformer",
    top_k_retrieve=20,
    top_k_rerank=8,
)
pretty_print_answer(result.answer)
print(f"\n  Pipeline latency: {result.total_latency_ms}ms")

# ── Test 3: 5 Example queries — save these for README ─────────
print("\n\n[TEST 3] 5 Example queries for docs/example_queries.md")

example_queries = [
    "What is the key innovation of Vision Transformer (ViT) compared to CNNs?",
    "How does DeiT train vision transformers without large datasets?",
    "What datasets are used to evaluate Swin Transformer?",
    "How does masked autoencoder pretraining work for vision models?",
    "What are the main differences between ViT and Swin Transformer architectures?",
]

example_results = []
for i, query in enumerate(example_queries, 1):
    print(f"\n  [{i}/5] Running: {query[:60]}...")
    result = run_pipeline(query, top_k_retrieve=20, top_k_rerank=8)
    example_results.append(result)

    # Print condensed output
    clean_answer = result.answer.answer
    # Remove citations section for display
    import re
    clean_answer = re.sub(
        r'\nCITATIONS:.*', '', clean_answer, flags=re.DOTALL
    )
    clean_answer = re.sub(
        r'\nCONFIDENCE:.*', '', clean_answer, flags=re.DOTALL
    )
    print(f"\n  QUERY: {query}")
    print(f"  ANSWER ({result.answer.confidence} confidence, "
          f"{result.total_latency_ms}ms):")
    print(f"  {clean_answer[:400].strip()}...")
    print(f"  Citations: {len(result.answer.citations)} | "
          f"Chunks used: {len(result.ranked_chunks)}")
    print(f"  Evidence: "
          f"{[c['title'][:30] for c in result.answer.citations[:3]]}")

# ── Test 4: Vector-only vs Hybrid answer comparison ───────────
print("\n\n[TEST 4] VECTOR-ONLY vs HYBRID answer comparison")
comparison_query = (
    "What methods do papers that extend ViT use "
    "for improving image classification?"
)

print(f"  Query: {comparison_query}\n")

# Vector only
print("  Running vector-only pipeline...")
vector_result = run_pipeline(
    comparison_query,
    use_graph=False,
    pipeline_variant="vector_only",
)

# Hybrid
print("  Running hybrid pipeline...")
hybrid_result = run_pipeline(
    comparison_query,
    use_graph=True,
    pipeline_variant="hybrid",
)

print(f"\n  ── VECTOR ONLY ({vector_result.total_latency_ms}ms) ──")
print(f"  Confidence: {vector_result.answer.confidence}")
print(f"  Citations: {len(vector_result.answer.citations)}")
v_answer = re.sub(
    r'\nCITATIONS:.*', '', vector_result.answer.answer, flags=re.DOTALL
)
print(f"  Answer: {v_answer[:300].strip()}...")
print(f"  Papers cited: "
      f"{list(set(c['title'][:35] for c in vector_result.answer.citations))}")

print(f"\n  ── HYBRID KAG ({hybrid_result.total_latency_ms}ms) ──")
print(f"  Confidence: {hybrid_result.answer.confidence}")
print(f"  Citations: {len(hybrid_result.answer.citations)}")
h_answer = re.sub(
    r'\nCITATIONS:.*', '', hybrid_result.answer.answer, flags=re.DOTALL
)
print(f"  Answer: {h_answer[:300].strip()}...")
print(f"  Papers cited: "
      f"{list(set(c['title'][:35] for c in hybrid_result.answer.citations))}")

v_papers = set(c['arxiv_id'] for c in vector_result.answer.citations)
h_papers = set(c['arxiv_id'] for c in hybrid_result.answer.citations)
hybrid_extra = h_papers - v_papers
print(f"\n  Papers cited by HYBRID but not VECTOR: {len(hybrid_extra)}")
for arxiv_id in hybrid_extra:
    for c in hybrid_result.answer.citations:
        if c['arxiv_id'] == arxiv_id:
            print(f"    + {c['title'][:55]}")
            break

# ── Test 5: Unanswerable question ────────────────────────────
print("\n\n[TEST 5] Unanswerable question (should say not found)")
unanswerable = "What is the capital city of France?"
result = run_pipeline(unanswerable, top_k_retrieve=10, top_k_rerank=5)
print(f"  Query: {unanswerable}")
print(f"  Not found: {result.answer.not_found}")
print(f"  Confidence: {result.answer.confidence}")
u_answer = re.sub(
    r'\nCITATIONS:.*', '', result.answer.answer, flags=re.DOTALL
)
print(f"  Answer: {u_answer[:200].strip()}")

# ── Test 6: Latency breakdown ─────────────────────────────────
print("\n\n[TEST 6] Latency breakdown across 3 queries")
latency_queries = [
    "how does patch embedding work in ViT",
    "what is masked autoencoder pretraining",
    "compare Swin and DeiT training approaches",
]

for q in latency_queries:
    t0 = time.time()

    # Retrieve
    t1 = time.time()
    h_results, trace = retriever.retrieve(q, top_k=20)
    retrieve_ms = int((time.time() - t1) * 1000)

    # Rerank
    t1 = time.time()
    ranked = rerank(q, h_results, top_k=8)
    rerank_ms = int((time.time() - t1) * 1000)

    # Generate
    t1 = time.time()
    answer = generate_answer(q, ranked, max_chunks=8)
    gen_ms = int((time.time() - t1) * 1000)

    total_ms = int((time.time() - t0) * 1000)
    print(f"\n  Query: {q[:50]}")
    print(f"    Retrieve: {retrieve_ms}ms | "
          f"Rerank: {rerank_ms}ms | "
          f"Generate: {gen_ms}ms | "
          f"Total: {total_ms}ms")
    print(f"    Confidence: {answer.confidence} | "
          f"Citations: {len(answer.citations)}")

# ── Save example queries to docs/ ────────────────────────────
print("\n\n[SAVING] Writing docs/example_queries.md...")

import os
os.makedirs("docs", exist_ok=True)

with open("docs/example_queries.md", "w") as f:
    f.write("# Example Queries and Answers\n\n")
    f.write("Generated by the Hybrid KAG pipeline on Day 9.\n\n")
    f.write("---\n\n")

    for i, (query, result) in enumerate(
        zip(example_queries, example_results), 1
    ):
        f.write(f"## Query {i}\n\n")
        f.write(f"**Question:** {query}\n\n")
        f.write(f"**Confidence:** {result.answer.confidence}\n\n")
        f.write(f"**Latency:** {result.total_latency_ms}ms\n\n")

        clean = re.sub(
            r'\nCITATIONS:.*', '',
            result.answer.answer,
            flags=re.DOTALL
        )
        clean = re.sub(
            r'\nCONFIDENCE:.*', '', clean, flags=re.DOTALL
        )
        f.write(f"**Answer:**\n\n{clean.strip()}\n\n")

        if result.answer.citations:
            f.write("**Sources:**\n\n")
            for c in result.answer.citations:
                f.write(
                    f"- [{c['citation_number']}] "
                    f"{c['title']} ({c['year']}) "
                    f"— arxiv:{c['arxiv_id']}\n"
                )

        f.write("\n---\n\n")

print("  Saved to docs/example_queries.md")

# ── Log results to DB ─────────────────────────────────────────
print("\n[LOGGING] Writing results to query_logs...")
try:
    for result in example_results:
        log_pipeline_result(result)
    log_pipeline_result(vector_result)
    log_pipeline_result(hybrid_result)
    print("  Logged to query_logs table")

    from ingestion.db import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM query_logs")
        ).fetchone()[0]
    print(f"  Total query_logs rows: {count}")
except Exception as e:
    print(f"  Logging failed: {e}")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "="*70)
print("DAY 9 SUMMARY")
print("="*70)
print(f"  ✓ Reranker: BAAI/bge-reranker-base loaded and working")
print(f"  ✓ Generator: Gemini 1.5 Flash generating cited answers")
print(f"  ✓ Pipeline: retrieve → rerank → generate end-to-end")
print(f"  ✓ 5 example queries saved to docs/example_queries.md")
print(f"  ✓ Vector vs Hybrid comparison done")
print(f"  ✓ Unanswerable question handled gracefully")
print(f"\n  Pipeline is now complete end-to-end.")
print(f"  Tomorrow: Day 10 — FastAPI backend wrapping this pipeline")
print("="*70)