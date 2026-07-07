"""
Run RAGAS on pre-generated samples.
Run with: eval_env (python scratch/run_ragas_on_samples.py)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("RAGAS EVALUATION on pre-generated samples")
print("="*60)

# Verify versions
import ragas, numpy, google.generativeai as g
print(f"\nragas:               {ragas.__version__}")
print(f"numpy:               {numpy.__version__}")
print(f"google-generativeai: {g.__version__}")
print()

# Load samples
samples_path = Path("data/eval_samples.json")
if not samples_path.exists():
    print("ERROR: data/eval_samples.json not found")
    print("Run first: rag\\Scripts\\activate && python scratch/generate_eval_samples.py")
    exit(1)

with open(samples_path) as f:
    all_samples = json.load(f)

print(f"Loaded samples for configs: {list(all_samples.keys())}")


def evaluate_config(samples: list[dict], config_name: str) -> dict:
    """Run RAGAS on one config's samples."""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = os.getenv("GEMINI_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        convert_system_message_to_human=True,
        temperature=0,
    )
    emb = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = emb
    context_precision.llm = llm
    context_recall.llm = llm

    dataset = Dataset.from_dict({
        "question":      [s["question"] for s in samples],
        "answer":        [s["answer"] for s in samples],
        "contexts":      [
            s["contexts"] if s["contexts"] else ["No context."]
            for s in samples
        ],
        "ground_truths": [[s["ground_truth"]] for s in samples],
    })

    print(f"\n  Evaluating {config_name} ({len(samples)} samples)...")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            raise_exceptions=False,
        )

        scores = {
            "faithfulness":      round(float(result["faithfulness"]), 4),
            "answer_relevancy":  round(float(result["answer_relevancy"]), 4),
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall":    round(float(result["context_recall"]), 4),
        }
        scores["average"] = round(sum(scores.values()) / 4, 4)
        print(f"  Scores: {scores}")
        return scores

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "average": 0.0,
            "error": str(e),
        }


# Run evaluation for each config
all_results = {}
configs = ["naive_vector", "vector_rerank", "hybrid_kag"]

for i, config_name in enumerate(configs):
    if config_name not in all_samples:
        print(f"Skipping {config_name} — not in samples file")
        continue

    scores = evaluate_config(all_samples[config_name], config_name)
    all_results[config_name] = scores

    if i < len(configs) - 1:
        print(f"\n  Waiting 30s before next config...")
        time.sleep(30)

# Save scores
Path("data").mkdir(exist_ok=True)
with open("data/ragas_scores.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print table
print("\n" + "="*70)
print("FINAL RAGAS COMPARISON TABLE")
print("="*70)
print(f"\n{'Config':<25} {'Faith':>8} {'Rel':>8} "
      f"{'Prec':>8} {'Recall':>8} {'Avg':>8}")
print("-"*70)

display = {
    "naive_vector":  "Naive Vector RAG",
    "vector_rerank": "Vector + Reranker",
    "hybrid_kag":    "Hybrid KAG (Ours)",
}
for config_name in configs:
    if config_name in all_results:
        s = all_results[config_name]
        print(
            f"{display[config_name]:<25} "
            f"{s['faithfulness']:>8.4f} "
            f"{s['answer_relevancy']:>8.4f} "
            f"{s['context_precision']:>8.4f} "
            f"{s['context_recall']:>8.4f} "
            f"{s['average']:>8.4f}"
        )

# Save markdown
Path("docs").mkdir(exist_ok=True)
with open("docs/ragas_results.md", "w") as f:
    f.write("# RAGAS Evaluation Results\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
    f.write(f"**Corpus:** 25 Vision Transformer papers\n\n")
    f.write(f"**Questions:** {len(all_samples.get('hybrid_kag', []))} per config\n\n")
    f.write("---\n\n")
    f.write("## Metric Comparison\n\n")
    f.write("| Configuration | Faithfulness | Answer Relevancy | "
            "Context Precision | Context Recall | Average |\n")
    f.write("|---|---|---|---|---|---|\n")

    config_display = {
        "naive_vector":  "Naive Vector RAG",
        "vector_rerank": "Vector + Reranker",
        "hybrid_kag":    "**Hybrid KAG (Ours)**",
    }
    for config, name in config_display.items():
        if config not in all_results:
            continue
        s = all_results[config]
        f.write(
            f"| {name} "
            f"| {s['faithfulness']:.4f} "
            f"| {s['answer_relevancy']:.4f} "
            f"| {s['context_precision']:.4f} "
            f"| {s['context_recall']:.4f} "
            f"| {s['average']:.4f} |\n"
        )

    if "hybrid_kag" in all_results and "naive_vector" in all_results:
        h = all_results["hybrid_kag"]
        n = all_results["naive_vector"]
        f.write("\n## Interpretation\n\n")
        f.write("### Key Findings\n\n")
        f.write(
            f"- **Faithfulness**: Hybrid KAG {h['faithfulness']:.4f} "
            f"vs naive {n['faithfulness']:.4f} "
            f"({h['faithfulness']-n['faithfulness']:+.4f})\n"
        )
        f.write(
            f"- **Context Precision**: "
            f"{h['context_precision']-n['context_precision']:+.4f} "
            f"from graph-guided retrieval\n"
        )
        f.write(
            f"- **Context Recall**: "
            f"{h['context_recall']-n['context_recall']:+.4f} "
            f"from graph traversal\n"
        )

print(f"\n✓ Saved docs/ragas_results.md")
print(f"✓ Saved data/ragas_scores.json")
print("="*70)