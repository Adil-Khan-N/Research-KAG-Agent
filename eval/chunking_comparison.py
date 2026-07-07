"""
Compare fixed vs adaptive chunking using the same eval metrics.
Adds a third row to the RAGAS results table.
"""

import json
import time
import re
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


def run_adaptive_pipeline_for_eval(
    question: str,
    top_k: int = 8,
) -> dict:
    """
    Run pipeline using adaptive chunks instead of fixed chunks.
    Uses HybridKAGRetriever but swaps the vector search to adaptive table.
    """
    from retrieval.adaptive_search import search_adaptive
    from retrieval.reranker import rerank
    from retrieval.generator import generate_answer
    from ingestion.db import engine

    try:
        # Use adaptive search
        results = search_adaptive(question, top_k=top_k * 2, engine=engine)

        ranked = rerank(question, results, top_k=top_k)
        contexts = [r.text for r in ranked]

        answer_obj = generate_answer(question, ranked, max_chunks=top_k)
        answer = answer_obj.answer
        answer = re.sub(r'\nCITATIONS:.*', '', answer, flags=re.DOTALL)
        answer = re.sub(r'\nCONFIDENCE:.*', '', answer, flags=re.DOTALL)
        answer = answer.strip()

        return {
            "question": question,
            "answer":   answer,
            "contexts": contexts if contexts else ["No context."],
        }

    except Exception as e:
        logger.error(f"Adaptive pipeline failed: {e}")
        return {
            "question": question,
            "answer":   "Error occurred.",
            "contexts": ["No context."],
        }


def compare_chunk_quality(
    test_queries: list[str],
    delay: float = 5.0,
) -> dict:
    """
    Compare fixed vs adaptive chunks on the same queries.
    Returns side-by-side stats.
    """
    from retrieval.search import search
    from retrieval.adaptive_search import search_adaptive
    from ingestion.db import engine

    print("\nComparing chunk quality on test queries:")
    print(f"{'Query':<50} {'Fixed avg':>10} {'Adaptive avg':>12}")
    print("-"*75)

    fixed_scores = []
    adaptive_scores = []

    for query in test_queries:
        fixed = search(query, top_k=5, engine=engine)
        adaptive = search_adaptive(query, top_k=5, engine=engine)

        fixed_avg = (
            sum(r.similarity for r in fixed) / len(fixed)
            if fixed else 0
        )
        adaptive_avg = (
            sum(r.similarity for r in adaptive) / len(adaptive)
            if adaptive else 0
        )

        fixed_scores.append(fixed_avg)
        adaptive_scores.append(adaptive_avg)

        better = "✓ adaptive" if adaptive_avg > fixed_avg else "= fixed"
        print(
            f"  {query[:48]:<50} "
            f"{fixed_avg:.4f}  "
            f"{adaptive_avg:.4f}  {better}"
        )

        time.sleep(delay)

    overall_fixed = sum(fixed_scores) / len(fixed_scores)
    overall_adaptive = sum(adaptive_scores) / len(adaptive_scores)
    improvement = overall_adaptive - overall_fixed

    print(f"\n  Overall avg similarity:")
    print(f"    Fixed:    {overall_fixed:.4f}")
    print(f"    Adaptive: {overall_adaptive:.4f}")
    print(
        f"    Change:   {improvement:+.4f} "
        f"({'improvement' if improvement > 0 else 'regression'})"
    )

    return {
        "fixed_avg":      round(overall_fixed, 4),
        "adaptive_avg":   round(overall_adaptive, 4),
        "improvement":    round(improvement, 4),
        "per_query": [
            {
                "query":    q,
                "fixed":    round(f, 4),
                "adaptive": round(a, 4),
            }
            for q, f, a in zip(
                test_queries, fixed_scores, adaptive_scores
            )
        ],
    }


def update_ragas_results_with_adaptive(
    adaptive_scores: dict,
    output_path: str = "docs/ragas_results.md",
):
    """
    Add adaptive chunking row to existing RAGAS results markdown.
    """
    existing_path = Path(output_path)
    if not existing_path.exists():
        print(f"  No existing results at {output_path}")
        return

    with open(output_path) as f:
        content = f.read()

    # Add adaptive row to table
    new_row = (
        f"| Hybrid KAG + Adaptive Chunking "
        f"| {adaptive_scores.get('faithfulness', 0):.4f} "
        f"| {adaptive_scores.get('answer_relevancy', 0):.4f} "
        f"| {adaptive_scores.get('context_precision', 0):.4f} "
        f"| {adaptive_scores.get('context_recall', 0):.4f} "
        f"| {adaptive_scores.get('average', 0):.4f} |\n"
    )

    # Insert before the interpretation section
    if "## Interpretation" in content:
        content = content.replace(
            "## Interpretation",
            new_row + "\n## Interpretation",
        )
    else:
        content += "\n" + new_row

    # Add chunking analysis section
    content += f"""
## Adaptive Chunking Analysis

### Section-Aware Chunk Sizes

| Section | Target Tokens | Rationale |
|---|---|---|
| Abstract | 150 | Dense, self-contained sentences |
| Preamble | 100 | Title/author noise, keep small |
| Introduction | 200 | Narrative context |
| Methods | 300 | Technical detail, standard size |
| Results | 400 | Needs surrounding context for numbers |
| Discussion | 250 | Synthesis prose |

### Impact

- Context Precision: {adaptive_scores.get('context_precision', 0):.4f} 
  (fixed: see above)
- Context Recall: {adaptive_scores.get('context_recall', 0):.4f}
- Adaptive chunking {"improved" if adaptive_scores.get('average', 0) > 0 else "changed"} 
  average score by {adaptive_scores.get('improvement', 0):+.4f}

*Note: Results include ablation with negative/neutral outcomes — 
real research documents what doesn't work as well as what does.*
"""

    with open(output_path, "w") as f:
        f.write(content)

    print(f"  Updated {output_path} with adaptive chunking row")