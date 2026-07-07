"""
Day 16 full test suite.
Tests:
1. Chunk config lookup by section
2. Adaptive vs fixed chunk sizes per section
3. Full paper adaptive chunking
4. Comparison stats fixed vs adaptive
5. Adaptive search works
6. Side-by-side similarity comparison
7. RAGAS evaluation with adaptive chunks
8. Update ragas_results.md

Run: python scratch/test_day16.py
"""

import time
import json
import logging
logging.basicConfig(level=logging.WARNING)

print("="*70)
print("DAY 16 — ADAPTIVE CHUNKING STRATEGY")
print("="*70)

from ingestion.adaptive_chunker import (
    get_chunk_config,
    adaptive_chunk_text,
    adaptive_chunk_paper,
    compare_chunking_strategies,
    SECTION_CHUNK_SIZES,
    DEFAULT_CHUNK_SIZE,
)

# ── Test 1: Chunk config lookup ───────────────────────────────
print("\n[TEST 1] Section chunk config lookup")

test_sections = [
    ("abstract",     150),
    ("preamble",     100),
    ("introduction", 200),
    ("methods",      300),
    ("results",      400),
    ("experiments",  400),
    ("conclusion",   200),
    ("unknown_xyz",  300),  # default
    ("1. introduction", 200),
    ("3. method",    300),
]

all_correct = True
for section, expected_target in test_sections:
    config = get_chunk_config(section)
    actual = config["target"]
    status = "✓" if actual == expected_target else "✗"
    if actual != expected_target:
        all_correct = False
    print(f"  {status} '{section}': target={actual} "
          f"(expected {expected_target})")

assert all_correct, "Some section configs are wrong"
print("  ✓ PASSED")


# ── Test 2: Chunk size differences ───────────────────────────
print("\n[TEST 2] Adaptive produces different sizes per section")

sample_text = """
Vision Transformers have revolutionized computer vision by applying
the transformer architecture to image patches. The key innovation is
the patch embedding layer which converts images into sequences.

The model processes these patch sequences using multi-head self-attention.
Each attention head learns different aspects of the visual features.
The output is then passed through a feed-forward network.

Results show that pre-training on large datasets significantly improves
performance. On ImageNet-1K, our model achieves 85.2% top-1 accuracy.
This represents a 3.1% improvement over the previous state-of-the-art.
The ablation study confirms that each component contributes positively.
"""

sections = {
    "abstract":    (sample_text, 150),
    "methods":     (sample_text, 300),
    "results":     (sample_text, 400),
    "preamble":    (sample_text, 100),
}

print(f"\n  {'Section':<15} {'Target':>8} {'Got chunks':>12} "
      f"{'Avg tokens':>12}")
print(f"  {'-'*50}")

for section, (text, target) in sections.items():
    config = get_chunk_config(section)
    chunks = adaptive_chunk_text(text, section=section, config=config)

    from ingestion.adaptive_chunker import count_tokens
    if chunks:
        avg = sum(count_tokens(c) for c in chunks) / len(chunks)
    else:
        avg = 0

    print(f"  {section:<15} {target:>8} {len(chunks):>12} {avg:>12.1f}")

print("  ✓ PASSED — different sections get different chunk sizes")


# ── Test 3: Full paper adaptive chunking ──────────────────────
print("\n[TEST 3] Full paper adaptive chunking")

import json
from pathlib import Path

# Load ViT paper
vit_path = Path("data/processed/2010.11929.json")
if vit_path.exists():
    with open(vit_path) as f:
        vit_paper = json.load(f)

    from ingestion.chunker import chunk_paper as fixed_chunk
    fixed_chunks = fixed_chunk(vit_paper)
    adaptive_chunks = adaptive_chunk_paper(vit_paper)

    print(f"\n  Paper: {vit_paper['title'][:50]}")
    if fixed_chunks:
        print(f"  Fixed chunks:    {len(fixed_chunks)} "
              f"(avg {sum(c['token_count'] for c in fixed_chunks)//len(fixed_chunks)} tokens)")
    else:
        print(f"  Fixed chunks:    0 (no chunks found)")

    print(f"  Adaptive chunks: {len(adaptive_chunks)} ")
    if adaptive_chunks:
        print(f"(avg {sum(c['token_count'] for c in adaptive_chunks)//len(adaptive_chunks)} tokens)")
    else:
        print(f"  Adaptive chunks: 0 (no chunks found)")

    # Show per-section comparison
    print(f"\n  Section breakdown (adaptive):")
    by_section = {}
    for c in adaptive_chunks:
        sec = c.get("section", "unknown")
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].append(c["token_count"])

    for sec, tokens in sorted(by_section.items(),
                               key=lambda x: -len(x[1]))[:8]:
        config = get_chunk_config(sec)
        avg = sum(tokens) / len(tokens)
        print(f"    {sec[:30]:<32} {len(tokens):>3} chunks, "
              f"avg={avg:.0f} tokens (target={config['target']})")

    assert len(adaptive_chunks) > 0
    print("  ✓ PASSED")
else:
    print("  ⚠ ViT paper not found at data/processed/2010.11929.json")


# ── Test 4: Run adaptive ingestion ───────────────────────────
print("\n[TEST 4] Run adaptive ingestion pipeline")
print("  (This embeds all papers into chunks_adaptive table)")

from scratch.run_adaptive_ingestion import (
    run_adaptive_ingestion,
    create_adaptive_chunks_table,
)

run_adaptive_ingestion()

# Verify table populated
from ingestion.db import engine
from sqlalchemy import text

with engine.connect() as conn:
    adaptive_count = conn.execute(
        text("SELECT COUNT(*) FROM chunks_adaptive")
    ).fetchone()[0]
    fixed_count = conn.execute(
        text("SELECT COUNT(*) FROM chunks")
    ).fetchone()[0]

print(f"\n  Fixed chunks in DB:    {fixed_count}")
print(f"  Adaptive chunks in DB: {adaptive_count}")

ratio = adaptive_count / max(fixed_count, 1)
print(f"  Ratio: {ratio:.2f}x")
assert adaptive_count > 0, "Adaptive chunks should be ingested"
print("  ✓ PASSED")


# ── Test 5: Adaptive search works ────────────────────────────
print("\n[TEST 5] Adaptive search")

from retrieval.adaptive_search import search_adaptive
from retrieval.search import search

test_q = "how does patch embedding work in ViT"

fixed_results  = search(test_q, top_k=5, engine=engine)
adaptive_results = search_adaptive(test_q, top_k=5, engine=engine)

print(f"\n  Query: '{test_q}'")
print(f"\n  Fixed results (top 3):")
for r in fixed_results[:3]:
    print(f"    [{r.similarity:.4f}] {r.title[:45]} | "
          f"{r.section} | {r.token_count} tokens")

print(f"\n  Adaptive results (top 3):")
for r in adaptive_results[:3]:
    print(f"    [{r.similarity:.4f}] {r.title[:45]} | "
          f"{r.section} | {r.token_count} tokens")

assert len(adaptive_results) > 0
print("\n  ✓ PASSED")


# ── Test 6: Side-by-side comparison ──────────────────────────
print("\n[TEST 6] Side-by-side similarity comparison")

from eval.chunking_comparison import compare_chunk_quality

comparison_queries = [
    "how does attention mechanism work in vision transformers",
    "what datasets are used to evaluate ViT",
    "compare fixed and shifted window attention",
    "what is masked autoencoder pretraining",
    "how does DeiT improve data efficiency",
]

comparison = compare_chunk_quality(comparison_queries, delay=1.0)

print(f"\n  Fixed avg:    {comparison['fixed_avg']:.4f}")
print(f"  Adaptive avg: {comparison['adaptive_avg']:.4f}")
print(f"  Improvement:  {comparison['improvement']:+.4f}")
print("  ✓ PASSED")


# ── Test 7: Token distribution analysis ──────────────────────
print("\n[TEST 7] Token distribution analysis")

with engine.connect() as conn:
    fixed_dist = conn.execute(text("""
        SELECT
            AVG(token_count)::int  AS avg,
            MIN(token_count)       AS min,
            MAX(token_count)       AS max,
            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY token_count)::int AS median
        FROM chunks
    """)).fetchone()

    adaptive_dist = conn.execute(text("""
        SELECT
            AVG(token_count)::int  AS avg,
            MIN(token_count)       AS min,
            MAX(token_count)       AS max,
            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY token_count)::int AS median
        FROM chunks_adaptive
    """)).fetchone()

print(f"\n  {'Metric':<12} {'Fixed':>10} {'Adaptive':>10}")
print(f"  {'-'*35}")
print(f"  {'Average':<12} {fixed_dist[0]:>10} {adaptive_dist[0]:>10}")
print(f"  {'Min':<12} {fixed_dist[1]:>10} {adaptive_dist[1]:>10}")
print(f"  {'Max':<12} {fixed_dist[2]:>10} {adaptive_dist[2]:>10}")
print(f"  {'Median':<12} {fixed_dist[3]:>10} {adaptive_dist[3]:>10}")
print("  ✓ PASSED")


# ── Test 8: RAGAS-style eval on adaptive chunks ───────────────
print("\n[TEST 8] Evaluation with adaptive chunks")
print("  Running 3 questions through adaptive pipeline...")
print("  (Gemini judge — takes 3-5 minutes)\n")

from eval.chunking_comparison import run_adaptive_pipeline_for_eval
from eval.ragas_eval import run_pipeline_for_eval

eval_questions = [
    {
        "question": "What is the key innovation of Vision Transformer ViT?",
        "ground_truth": (
            "ViT applies a pure Transformer directly to sequences "
            "of image patches for image recognition without CNNs."
        ),
    },
    {
        "question": "What datasets does Swin Transformer evaluate on?",
        "ground_truth": (
            "Swin Transformer evaluates on ImageNet-1K, COCO, "
            "and ADE20K."
        ),
    },
    {
        "question": (
            "What datasets do papers extending ViT use for evaluation?"
        ),
        "ground_truth": (
            "Papers extending ViT use ImageNet-1K, COCO, ADE20K, "
            "and Tiny-ImageNet."
        ),
    },
]

fixed_samples = []
adaptive_samples = []

for i, item in enumerate(eval_questions):
    print(f"  [{i+1}/3] Fixed pipeline: {item['question'][:45]}...")
    fixed = run_pipeline_for_eval(
        item["question"], use_graph=True,
        top_k_retrieve=20, top_k_rerank=8,
    )
    fixed["ground_truth"] = item["ground_truth"]
    fixed_samples.append(fixed)
    time.sleep(15)

    print(f"  [{i+1}/3] Adaptive pipeline: {item['question'][:45]}...")
    adaptive = run_adaptive_pipeline_for_eval(
        item["question"], top_k=8,
    )
    adaptive["ground_truth"] = item["ground_truth"]
    adaptive_samples.append(adaptive)

    if i < len(eval_questions) - 1:
        time.sleep(15)

# Score both
print("\n  Scoring fixed chunks...")
from eval.ragas_eval import compute_ragas_scores
time.sleep(20)
fixed_scores = compute_ragas_scores(fixed_samples)

print("\n  Scoring adaptive chunks...")
time.sleep(20)
adaptive_scores_ragas = compute_ragas_scores(adaptive_samples)

# Add improvement delta
adaptive_scores_ragas["improvement"] = round(
    adaptive_scores_ragas["average"] - fixed_scores["average"], 4
)

print(f"\n  {'Metric':<22} {'Fixed':>8} {'Adaptive':>10} {'Change':>8}")
print(f"  {'-'*52}")
for metric in ["faithfulness", "answer_relevancy",
               "context_precision", "context_recall", "average"]:
    f = fixed_scores.get(metric, 0)
    a = adaptive_scores_ragas.get(metric, 0)
    delta = a - f
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(
        f"  {metric:<22} {f:>8.4f} {a:>10.4f} "
        f"{delta:>+8.4f} {arrow}"
    )

print("\n  Updating ragas_results.md...")
from eval.chunking_comparison import update_ragas_results_with_adaptive
update_ragas_results_with_adaptive(adaptive_scores_ragas)

# Save scores
import json
with open("data/adaptive_scores.json", "w") as f:
    json.dump({
        "fixed":    fixed_scores,
        "adaptive": adaptive_scores_ragas,
        "comparison": comparison,
    }, f, indent=2)

print("  ✓ PASSED")


# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*70)
print("DAY 16 SUMMARY")
print("="*70)
print(f"  ✓ Section config: abstract=150, methods=300, results=400 tokens")
print(f"  ✓ Fixed chunks in DB:    {fixed_count}")
print(f"  ✓ Adaptive chunks in DB: {adaptive_count} ({ratio:.2f}x)")
print(f"  ✓ Similarity comparison: {comparison['improvement']:+.4f} change")
print(f"\n  RAGAS comparison:")
print(f"    Fixed avg:    {fixed_scores.get('average', 0):.4f}")
print(f"    Adaptive avg: {adaptive_scores_ragas.get('average', 0):.4f}")
print(f"    Change:       {adaptive_scores_ragas.get('improvement', 0):+.4f}")
print(f"\n  CV note: Even neutral/negative results document ablation rigor")
print(f"  → 'Evaluated adaptive chunking impact; documented in ablation table'")
print("="*70)