"""
RAGAS quick sanity check using ragas==0.4.3
Run: python scratch/test_ragas_quick.py
"""

import os
import time
import logging
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("RAGAS SANITY CHECK — ragas 0.4.3")
print("="*60)

# Verify versions
import ragas
print(f"\nragas version: {ragas.__version__}")

# Verify RAGAS 0.4.x imports work
try:
    from ragas import evaluate, EvaluationDataset
    # from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from ragas.metrics import ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    print("All ragas 0.4.x imports: OK")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)

# Verify LangChain Google GenAI
try:
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    print("langchain-google-genai imports: OK")
except ImportError as e:
    print(f"langchain-google-genai failed: {e}")
    exit(1)

print()

# Step 1: Generate 3 samples
from eval.ragas_eval import run_pipeline_for_eval

test_questions = [
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

print("Step 1: Generating 3 pipeline samples...")
samples = []
for i, item in enumerate(test_questions):
    print(f"  [{i+1}/3] {item['question'][:55]}...")
    result = run_pipeline_for_eval(
        item["question"],
        use_graph=True,
        top_k_retrieve=20,
        top_k_rerank=8,
    )
    result["ground_truth"] = item["ground_truth"]
    print(f"         Answer:   {result['answer'][:80]}...")
    print(f"         Contexts: {len(result['contexts'])} chunks")
    samples.append(result)
    if i < 2:
        print("         Waiting 15s (rate limit)...")
        time.sleep(15)

# Step 2: Run RAGAS 0.4.x evaluation
print("\nStep 2: Running RAGAS 0.4.x evaluation...")
print("        (2-5 minutes — Gemini as LLM judge)")
time.sleep(15)

from eval.ragas_eval import compute_ragas_scores
scores = compute_ragas_scores(samples)

# Step 3: Print results
print("\n" + "="*60)
print("RAGAS SCORES:")
print("="*60)

for metric, score in scores.items():
    if metric == "error":
        print(f"  ERROR: {score}")
        continue
    try:
        val = float(score)
        bar = "█" * int(val * 20)
        print(f"  {metric:<22} {val:.4f}  {bar}")
    except Exception:
        print(f"  {metric:<22} {score}")

print("\n" + "="*60)
avg = scores.get("average", 0)
try:
    if float(avg) > 0:
        print("✓ RAGAS 0.4.3 working correctly")
        print("\nRun full evaluation:")
        print("  python scratch/run_ragas_eval.py --quick  (9 questions)")
        print("  python scratch/run_ragas_eval.py          (15 questions)")
    else:
        print("✗ Scores are 0 — check error above")
except Exception:
    print("✗ Could not parse scores")
print("="*60)
# 
# """
# Quick evaluation sanity check — 3 questions, no RAGAS dependency.
# Run: python scratch/test_ragas_quick.py
# """

# import os
# import time
# import logging
# logging.basicConfig(level=logging.WARNING)

# from dotenv import load_dotenv
# load_dotenv()

# print("="*60)
# print("EVALUATION QUICK SANITY CHECK")
# print("="*60)

# import google.generativeai as genai_pkg
# print(f"\ngoogle-generativeai: {genai_pkg.__version__}")
# print("No RAGAS dependency — using Gemini judge directly\n")

# from eval.ragas_eval import run_pipeline_for_eval, score_sample

# test_questions = [
#     {
#         "question": "What is the key innovation of Vision Transformer ViT?",
#         "ground_truth": (
#             "ViT applies a pure Transformer directly to sequences "
#             "of image patches for image recognition without CNNs."
#         ),
#     },
#     {
#         "question": "What datasets does Swin Transformer evaluate on?",
#         "ground_truth": (
#             "Swin Transformer evaluates on ImageNet-1K, COCO, and ADE20K."
#         ),
#     },
#     {
#         "question": (
#             "What datasets do papers extending ViT use for evaluation?"
#         ),
#         "ground_truth": (
#             "Papers extending ViT use ImageNet-1K, COCO, ADE20K, "
#             "and Tiny-ImageNet."
#         ),
#     },
# ]

# print("Step 1: Running 3 questions through hybrid pipeline...")
# samples = []
# for i, item in enumerate(test_questions):
#     print(f"  [{i+1}/3] {item['question'][:55]}...")
#     result = run_pipeline_for_eval(
#         item["question"], use_graph=True,
#         top_k_retrieve=20, top_k_rerank=8,
#     )
#     result["ground_truth"] = item["ground_truth"]
#     print(f"         Answer: {result['answer'][:100]}...")
#     print(f"         Contexts: {len(result['contexts'])} chunks")
#     samples.append(result)
#     if i < 2:
#         print("         Waiting 15s...")
#         time.sleep(15)

# print("\nStep 2: Scoring first sample with Gemini judge...")
# sample = samples[0]
# scores = score_sample(sample, delay=5.0)

# print(f"\nScores for: '{sample['question'][:50]}'")
# for metric, score in scores.items():
#     bar = "█" * int(score * 20)
#     print(f"  {metric:<22} {score:.4f}  {bar}")

# print("\n" + "="*60)
# if any(s > 0 for s in scores.values()):
#     print("✓ Gemini judge working correctly")
#     print("\nRun full evaluation:")
#     print("  python scratch/run_ragas_eval.py --quick   (9 questions)")
#     print("  python scratch/run_ragas_eval.py           (15 questions)")
# else:
#     print("✗ All scores 0 — check GEMINI_API_KEY and rate limits")
# print("="*60)