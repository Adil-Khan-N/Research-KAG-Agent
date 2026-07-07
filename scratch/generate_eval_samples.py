"""
Generate evaluation samples using the main pipeline.
Saves to data/eval_samples.json for evaluation in eval_env.
Run with: rag env (python scratch/generate_eval_samples.py)
"""

import json
import time
import re
import logging
logging.basicConfig(level=logging.WARNING)

from eval.golden_dataset import GOLDEN_DATASET
from retrieval.hybrid_retriever import HybridKAGRetriever
from retrieval.reranker import rerank
from retrieval.generator import generate_answer
from retrieval.vector_only import vector_only_retrieve

print("="*60)
print("GENERATING EVAL SAMPLES")
print("="*60)

# Take 9 questions (3 per type) for quick eval
factual   = [q for q in GOLDEN_DATASET if q["type"] == "factual"][:3]
comparison= [q for q in GOLDEN_DATASET if q["type"] == "comparison"][:3]
multi_hop = [q for q in GOLDEN_DATASET if q["type"] == "multi_hop"][:3]
subset    = factual + comparison + multi_hop

print(f"Questions: {len(subset)} (3 factual, 3 comparison, 3 multi-hop)")

retriever = HybridKAGRetriever()

def run_config(questions, use_graph, top_k_retrieve, top_k_rerank):
    samples = []
    for i, item in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {item['question'][:50]}...")
        try:
            if use_graph:
                results, _ = retriever.retrieve(
                    item["question"], top_k=top_k_retrieve, use_graph=True
                )
            else:
                results = vector_only_retrieve(
                    item["question"], top_k=top_k_retrieve
                )

            ranked = rerank(item["question"], results, top_k=top_k_rerank)
            contexts = [r.text for r in ranked]

            answer_obj = generate_answer(
                item["question"], ranked, max_chunks=top_k_rerank
            )
            answer = answer_obj.answer
            answer = re.sub(r'\nCITATIONS:.*', '', answer, flags=re.DOTALL)
            answer = re.sub(r'\nCONFIDENCE:.*', '', answer, flags=re.DOTALL)
            answer = answer.strip()

            samples.append({
                "question":     item["question"],
                "answer":       answer,
                "contexts":     contexts if contexts else ["No context."],
                "ground_truth": item["ground_truth"],
                "type":         item["type"],
            })
            print(f"         OK — {len(contexts)} contexts")

        except Exception as e:
            print(f"         ERROR: {e}")
            samples.append({
                "question":     item["question"],
                "answer":       "Error generating answer.",
                "contexts":     ["No context."],
                "ground_truth": item["ground_truth"],
                "type":         item["type"],
            })

        if i < len(questions) - 1:
            time.sleep(15)

    return samples

all_samples = {}

configs = [
    ("naive_vector",  False, 20, 20),
    ("vector_rerank", False, 20,  8),
    ("hybrid_kag",    True,  20,  8),
]

for config_name, use_graph, top_k_r, top_k_rr in configs:
    print(f"\nConfig: {config_name}")
    samples = run_config(subset, use_graph, top_k_r, top_k_rr)
    all_samples[config_name] = samples
    print(f"  Generated {len(samples)} samples")
    if config_name != "hybrid_kag":
        print("  Waiting 30s...")
        time.sleep(30)

# Save
with open("data/eval_samples.json", "w") as f:
    json.dump(all_samples, f, indent=2)

print(f"\n✓ Saved to data/eval_samples.json")
print(f"  Total samples: {sum(len(v) for v in all_samples.values())}")
print("\nNow run RAGAS evaluation:")
print("  eval_env\\Scripts\\activate")
print("  python scratch/run_ragas_on_samples.py")