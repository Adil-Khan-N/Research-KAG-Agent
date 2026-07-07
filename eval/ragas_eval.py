"""
RAGAS evaluation using ragas==0.4.3 + langchain-google-genai==4.2.5

ragas 0.4.x API:
- Uses EvaluationDataset + SingleTurnSample
- LangchainLLMWrapper + LangchainEmbeddingsWrapper
- Metrics are classes not singletons: Faithfulness(), AnswerRelevancy()
- result.to_pandas() for scores
"""

import os
import json
import time
import re
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Pipeline runner ───────────────────────────────────────────

def run_pipeline_for_eval(
    question: str,
    use_graph: bool = True,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 8,
) -> dict:
    """Run pipeline and return eval sample dict."""
    from retrieval.hybrid_retriever import HybridKAGRetriever
    from retrieval.reranker import rerank
    from retrieval.generator import generate_answer
    from retrieval.vector_only import vector_only_retrieve

    try:
        if use_graph:
            retriever = HybridKAGRetriever()
            results, trace = retriever.retrieve(
                question, top_k=top_k_retrieve, use_graph=True,
            )
        else:
            results = vector_only_retrieve(question, top_k=top_k_retrieve)

        ranked = rerank(question, results, top_k=top_k_rerank)
        contexts = [r.text for r in ranked]

        answer_obj = generate_answer(
            question, ranked, max_chunks=top_k_rerank
        )
        answer = answer_obj.answer
        answer = re.sub(r'\nCITATIONS:.*', '', answer, flags=re.DOTALL)
        answer = re.sub(r'\nCONFIDENCE:.*', '', answer, flags=re.DOTALL)
        answer = answer.strip()

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts if contexts else ["No context retrieved."],
        }

    except Exception as e:
        logger.error(f"Pipeline failed for '{question[:50]}': {e}")
        return {
            "question": question,
            "answer": "Pipeline error occurred.",
            "contexts": ["No context retrieved."],
        }


# ── RAGAS evaluation using 0.4.x API ─────────────────────────

def compute_ragas_scores(samples: list[dict]) -> dict:
    """
    Compute RAGAS metrics using ragas==0.4.3 correct API.

    Key differences from 0.1.x:
    - Metrics are instantiated as classes: Faithfulness(llm=llm)
    - Dataset uses EvaluationDataset + SingleTurnSample
    - LLM wrapped with LangchainLLMWrapper
    - result.to_pandas() extracts scores
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    # Wrap LLM and embeddings for ragas 0.4.x
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
    )

    # Instantiate metrics — ragas 0.4.x uses classes not module singletons
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    # Build SingleTurnSample objects — ragas 0.4.x schema
    ragas_samples = []
    for s in samples:
        ragas_samples.append(SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=(
                s["contexts"]
                if s["contexts"]
                else ["No context retrieved."]
            ),
            reference=s["ground_truth"],
        ))

    dataset = EvaluationDataset(samples=ragas_samples)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        # ragas 0.4.x: use to_pandas() to get scores
        df = result.to_pandas()
        print(f"\n  Raw RAGAS scores per sample:")
        print(df[["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]].to_string())

        scores = {
            "faithfulness": round(
                float(df["faithfulness"].mean(skipna=True)), 4
            ),
            "answer_relevancy": round(
                float(df["answer_relevancy"].mean(skipna=True)), 4
            ),
            "context_precision": round(
                float(df["context_precision"].mean(skipna=True)), 4
            ),
            "context_recall": round(
                float(df["context_recall"].mean(skipna=True)), 4
            ),
        }
        scores["average"] = round(sum(scores.values()) / 4, 4)
        return scores

    except Exception as e:
        logger.error(f"RAGAS evaluate() failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "average": 0.0,
            "error": str(e),
        }


def evaluate_with_ragas(
    samples: list[dict], config_name: str
) -> dict:
    """Wrapper with logging."""
    print(f"\n  Running RAGAS for: {config_name} ({len(samples)} samples)")
    print("  Using ragas==0.4.3 with Gemini 2.5 Flash as judge")
    scores = compute_ragas_scores(samples)
    print(f"\n  Final averaged scores: {scores}")
    return scores


# ── Results saving ────────────────────────────────────────────

def save_results_markdown(
    all_results: dict,
    sample_count: int,
    output_path: str = "docs/ragas_results.md",
):
    """Save RAGAS results as markdown."""
    Path("docs").mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# RAGAS Evaluation Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("**Corpus:** 25 Vision Transformer papers\n\n")
        f.write(f"**Questions evaluated:** {sample_count}\n\n")
        f.write("**Framework:** ragas==0.4.3, judge: Gemini 2.5 Flash\n\n")
        f.write("---\n\n")
        f.write("## Metric Comparison\n\n")
        f.write(
            "| Configuration | Faithfulness | Answer Relevancy "
            "| Context Precision | Context Recall | Average |\n"
        )
        f.write("|---|---|---|---|---|---|\n")

        configs = {
            "naive_vector":  "Naive Vector RAG",
            "vector_rerank": "Vector + Reranker",
            "hybrid_kag":    "**Hybrid KAG (Ours)**",
        }
        for config, name in configs.items():
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

        f.write("\n## Interpretation\n\n")
        if "hybrid_kag" in all_results and "naive_vector" in all_results:
            h = all_results["hybrid_kag"]
            n = all_results["naive_vector"]
            f.write("### Key Findings\n\n")
            f.write(
                f"- **Faithfulness**: Hybrid KAG {h['faithfulness']:.4f} "
                f"vs naive {n['faithfulness']:.4f} "
                f"({h['faithfulness']-n['faithfulness']:+.4f})\n"
            )
            f.write(
                f"- **Context Precision**: "
                f"{h['context_precision']-n['context_precision']:+.4f} "
                f"improvement from graph-guided retrieval\n"
            )
            f.write(
                f"- **Context Recall**: "
                f"{h['context_recall']-n['context_recall']:+.4f} "
                f"from graph traversal\n"
            )

    print(f"  Saved to {output_path}")


# ── Full evaluation runner ────────────────────────────────────

def run_full_evaluation(
    subset_size: int = 9,
    delay_between_calls: float = 15.0,
    save_samples: bool = True,
):
    """Run full RAGAS evaluation across 3 configurations."""
    from eval.golden_dataset import GOLDEN_DATASET

    print("="*70)
    print("RAGAS EVALUATION — ragas 0.4.3")
    print("="*70)

    factual    = [q for q in GOLDEN_DATASET if q["type"] == "factual"]
    comparison = [q for q in GOLDEN_DATASET if q["type"] == "comparison"]
    multi_hop  = [q for q in GOLDEN_DATASET if q["type"] == "multi_hop"]

    per_type = max(1, subset_size // 3)
    subset = (
        factual[:per_type] +
        comparison[:per_type] +
        multi_hop[:per_type]
    )

    print(f"\nSubset: {len(subset)} questions ({per_type} per type)")

    all_results = {}
    all_samples = {}

    configs = [
        ("naive_vector",  False, 20, 20),
        ("vector_rerank", False, 20,  8),
        ("hybrid_kag",    True,  20,  8),
    ]

    for config_name, use_graph, top_k_r, top_k_rr in configs:
        print(f"\n{'─'*50}")
        print(f"CONFIG: {config_name}")
        print(f"{'─'*50}")

        samples = []
        for i, item in enumerate(subset):
            print(f"  [{i+1}/{len(subset)}] {item['question'][:50]}...")
            result = run_pipeline_for_eval(
                item["question"],
                use_graph=use_graph,
                top_k_retrieve=top_k_r,
                top_k_rerank=top_k_rr,
            )
            result["ground_truth"] = item["ground_truth"]
            samples.append(result)
            if i < len(subset) - 1:
                time.sleep(delay_between_calls)

        all_samples[config_name] = samples

        print(f"\n  Waiting 20s before RAGAS scoring...")
        time.sleep(20)

        scores = evaluate_with_ragas(samples, config_name)
        all_results[config_name] = scores

        if config_name != "hybrid_kag":
            print(f"  Waiting 30s before next config...")
            time.sleep(30)

    if save_samples:
        with open("data/ragas_samples.json", "w") as f:
            json.dump(all_samples, f, indent=2)

    with open("data/ragas_scores.json", "w") as f:
        json.dump(all_results, f, indent=2)

    save_results_markdown(all_results, len(subset))

    print("\n" + "="*70)
    print("FINAL RAGAS RESULTS")
    print("="*70)
    print(f"\n{'Config':<25} {'Faith':>8} {'Rel':>8} "
          f"{'Prec':>8} {'Recall':>8} {'Avg':>8}")
    print("-"*70)
    for config_name, _, _, _ in configs:
        if config_name in all_results:
            s = all_results[config_name]
            print(
                f"{config_name:<25} "
                f"{s['faithfulness']:>8.4f} "
                f"{s['answer_relevancy']:>8.4f} "
                f"{s['context_precision']:>8.4f} "
                f"{s['context_recall']:>8.4f} "
                f"{s['average']:>8.4f}"
            )
    print(f"\n  Results: docs/ragas_results.md")
    print("="*70)
    return all_results
# 
# """
# Custom evaluation pipeline using Gemini as judge.
# Computes the same 4 metrics as RAGAS but without the dependency mess:
#   - Faithfulness
#   - Answer Relevancy  
#   - Context Precision
#   - Context Recall

# Each metric is computed by prompting Gemini directly.
# """

# import os
# import json
# import time
# import re
# import logging
# from pathlib import Path
# from datetime import datetime

# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()
# logging.basicConfig(level=logging.WARNING)
# logger = logging.getLogger(__name__)

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# _judge = genai.GenerativeModel("gemini-2.5-flash")


# # ── Pipeline runner ───────────────────────────────────────────

# def run_pipeline_for_eval(
#     question: str,
#     use_graph: bool = True,
#     top_k_retrieve: int = 20,
#     top_k_rerank: int = 8,
# ) -> dict:
#     """Run pipeline and return eval sample dict."""
#     from retrieval.hybrid_retriever import HybridKAGRetriever
#     from retrieval.reranker import rerank
#     from retrieval.generator import generate_answer
#     from retrieval.vector_only import vector_only_retrieve

#     try:
#         if use_graph:
#             retriever = HybridKAGRetriever()
#             results, trace = retriever.retrieve(
#                 question, top_k=top_k_retrieve, use_graph=True,
#             )
#         else:
#             results = vector_only_retrieve(question, top_k=top_k_retrieve)

#         ranked = rerank(question, results, top_k=top_k_rerank)
#         contexts = [r.text for r in ranked]

#         answer_obj = generate_answer(question, ranked, max_chunks=top_k_rerank)
#         answer = answer_obj.answer
#         answer = re.sub(r'\nCITATIONS:.*', '', answer, flags=re.DOTALL)
#         answer = re.sub(r'\nCONFIDENCE:.*', '', answer, flags=re.DOTALL)
#         answer = answer.strip()

#         return {
#             "question": question,
#             "answer": answer,
#             "contexts": contexts if contexts else ["No context retrieved."],
#         }

#     except Exception as e:
#         logger.error(f"Pipeline failed for '{question[:50]}': {e}")
#         return {
#             "question": question,
#             "answer": "Pipeline error occurred.",
#             "contexts": ["No context retrieved."],
#         }


# # ── Gemini judge functions ────────────────────────────────────

# def _call_judge(prompt: str, retries: int = 3) -> str:
#     """Call Gemini judge with retry on rate limit."""
#     for attempt in range(retries):
#         try:
#             response = _judge.generate_content(prompt)
#             return response.text.strip()
#         except Exception as e:
#             if "429" in str(e) or "quota" in str(e).lower():
#                 wait = 30 * (attempt + 1)
#                 logger.warning(f"Rate limit hit, waiting {wait}s...")
#                 time.sleep(wait)
#             else:
#                 logger.error(f"Judge call failed: {e}")
#                 return "0.0"
#     return "0.0"


# def _extract_score(text: str) -> float:
#     """Extract a float score from judge response."""
#     # Look for patterns like "Score: 0.8" or "0.75" or "score is 0.9"
#     patterns = [
#         r'score[:\s]+([0-9]*\.?[0-9]+)',
#         r'([0-9]*\.?[0-9]+)\s*/\s*1',
#         r'^([0-9]*\.?[0-9]+)',
#         r'([0-9]*\.?[0-9]+)',
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, text.lower())
#         if match:
#             try:
#                 score = float(match.group(1))
#                 if 0.0 <= score <= 1.0:
#                     return score
#             except ValueError:
#                 continue
#     return 0.0


# def score_faithfulness(answer: str, contexts: list[str]) -> float:
#     """
#     Score: Are all claims in the answer supported by the contexts?
#     1.0 = fully grounded, 0.0 = completely hallucinated.
#     """
#     context_text = "\n\n".join(
#         f"[{i+1}] {c[:400]}" for i, c in enumerate(contexts[:5])
#     )
#     prompt = f"""You are evaluating whether an answer is faithful to given contexts.

# CONTEXTS:
# {context_text}

# ANSWER:
# {answer[:600]}

# TASK: Score how faithful the answer is to the contexts.
# - 1.0 = Every claim in the answer is directly supported by the contexts
# - 0.5 = Some claims supported, some not
# - 0.0 = Answer contains claims not found in contexts (hallucination)

# Respond with ONLY a decimal number between 0.0 and 1.0.
# Score:"""

#     result = _call_judge(prompt)
#     return _extract_score(result)


# def score_answer_relevancy(question: str, answer: str) -> float:
#     """
#     Score: Does the answer actually address the question?
#     1.0 = perfectly relevant, 0.0 = completely irrelevant.
#     """
#     prompt = f"""You are evaluating whether an answer is relevant to a question.

# QUESTION: {question}

# ANSWER: {answer[:600]}

# TASK: Score how well the answer addresses the question.
# - 1.0 = Answer directly and completely addresses the question
# - 0.5 = Answer partially addresses the question
# - 0.0 = Answer does not address the question at all

# Respond with ONLY a decimal number between 0.0 and 1.0.
# Score:"""

#     result = _call_judge(prompt)
#     return _extract_score(result)


# def score_context_precision(
#     question: str, contexts: list[str]
# ) -> float:
#     """
#     Score: Are the retrieved contexts relevant to the question?
#     1.0 = all contexts relevant, 0.0 = none relevant.
#     """
#     if not contexts:
#         return 0.0

#     relevant_count = 0
#     for i, context in enumerate(contexts[:5]):
#         prompt = f"""Is this context relevant to answering the question?

# QUESTION: {question}

# CONTEXT: {context[:400]}

# Answer with ONLY "yes" or "no"."""

#         result = _call_judge(prompt).lower().strip()
#         if "yes" in result:
#             relevant_count += 1
#         time.sleep(2)  # rate limit between context checks

#     return round(relevant_count / min(len(contexts), 5), 4)


# def score_context_recall(
#     question: str, ground_truth: str, contexts: list[str]
# ) -> float:
#     """
#     Score: Do the contexts contain the information needed to answer?
#     1.0 = contexts fully cover the ground truth, 0.0 = no coverage.
#     """
#     context_text = "\n\n".join(
#         f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts[:5])
#     )
#     prompt = f"""You are checking if retrieved contexts contain enough 
# information to produce a correct answer.

# QUESTION: {question}

# CORRECT ANSWER (ground truth): {ground_truth[:400]}

# RETRIEVED CONTEXTS:
# {context_text}

# TASK: Score how well the contexts cover the information in the correct answer.
# - 1.0 = Contexts contain all the key information from the correct answer
# - 0.5 = Contexts partially cover the correct answer
# - 0.0 = Contexts do not contain information from the correct answer

# Respond with ONLY a decimal number between 0.0 and 1.0.
# Score:"""

#     result = _call_judge(prompt)
#     return _extract_score(result)


# def score_sample(sample: dict, delay: float = 5.0) -> dict:
#     """
#     Score all 4 metrics for one sample.
#     Returns dict with all scores.
#     """
#     question = sample["question"]
#     answer = sample["answer"]
#     contexts = sample["contexts"]
#     ground_truth = sample.get("ground_truth", "")

#     scores = {}

#     print(f"      Faithfulness...")
#     scores["faithfulness"] = score_faithfulness(answer, contexts)
#     time.sleep(delay)

#     print(f"      Answer relevancy...")
#     scores["answer_relevancy"] = score_answer_relevancy(question, answer)
#     time.sleep(delay)

#     print(f"      Context precision...")
#     scores["context_precision"] = score_context_precision(question, contexts)
#     time.sleep(delay)

#     print(f"      Context recall...")
#     scores["context_recall"] = score_context_recall(
#         question, ground_truth, contexts
#     )
#     time.sleep(delay)

#     return scores


# def compute_ragas_scores(
#     samples: list[dict],
#     delay_between_samples: float = 12.0,
# ) -> dict:
#     """
#     Compute average scores across all samples.
#     Each sample is scored independently by Gemini judge.
#     """
#     all_scores = []

#     for i, sample in enumerate(samples):
#         print(f"\n    Sample {i+1}/{len(samples)}: "
#               f"{sample['question'][:45]}...")
#         try:
#             scores = score_sample(sample, delay=5.0)
#             all_scores.append(scores)
#             print(f"      → faith={scores['faithfulness']:.2f} "
#                   f"rel={scores['answer_relevancy']:.2f} "
#                   f"prec={scores['context_precision']:.2f} "
#                   f"recall={scores['context_recall']:.2f}")
#         except Exception as e:
#             logger.error(f"Scoring failed for sample {i}: {e}")
#             all_scores.append({
#                 "faithfulness": 0.0,
#                 "answer_relevancy": 0.0,
#                 "context_precision": 0.0,
#                 "context_recall": 0.0,
#             })

#         if i < len(samples) - 1:
#             time.sleep(delay_between_samples)

#     # Average across all samples
#     if not all_scores:
#         return {
#             "faithfulness": 0.0,
#             "answer_relevancy": 0.0,
#             "context_precision": 0.0,
#             "context_recall": 0.0,
#             "average": 0.0,
#         }

#     avg = {}
#     for metric in ["faithfulness", "answer_relevancy",
#                    "context_precision", "context_recall"]:
#         values = [s[metric] for s in all_scores if s[metric] is not None]
#         avg[metric] = round(sum(values) / len(values), 4) if values else 0.0

#     avg["average"] = round(sum(avg.values()) / 4, 4)
#     return avg


# def evaluate_with_ragas(
#     samples: list[dict], config_name: str
# ) -> dict:
#     """Wrapper with logging."""
#     print(f"\n  Scoring: {config_name} ({len(samples)} samples)")
#     scores = compute_ragas_scores(samples)
#     print(f"\n  Final scores: {scores}")
#     return scores


# # ── Results saving ────────────────────────────────────────────

# def save_results_markdown(
#     all_results: dict,
#     sample_count: int,
#     output_path: str = "docs/ragas_results.md",
# ):
#     """Save results as markdown comparison table."""
#     Path("docs").mkdir(exist_ok=True)

#     with open(output_path, "w") as f:
#         f.write("# Evaluation Results\n\n")
#         f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
#         f.write(f"**Corpus:** 25 Vision Transformer papers\n\n")
#         f.write(f"**Questions evaluated:** {sample_count}\n\n")
#         f.write(
#             "*Metrics computed using Gemini 2.5 Flash as judge "
#             "(same methodology as RAGAS)*\n\n"
#         )
#         f.write("---\n\n")
#         f.write("## Metric Comparison\n\n")
#         f.write(
#             "| Configuration | Faithfulness | Answer Relevancy "
#             "| Context Precision | Context Recall | Average |\n"
#         )
#         f.write("|---|---|---|---|---|---|\n")

#         configs = {
#             "naive_vector": "Naive Vector RAG",
#             "vector_rerank": "Vector + Reranker",
#             "hybrid_kag": "**Hybrid KAG (Ours)**",
#         }
#         for config, name in configs.items():
#             if config not in all_results:
#                 continue
#             s = all_results[config]
#             f.write(
#                 f"| {name} "
#                 f"| {s['faithfulness']:.4f} "
#                 f"| {s['answer_relevancy']:.4f} "
#                 f"| {s['context_precision']:.4f} "
#                 f"| {s['context_recall']:.4f} "
#                 f"| {s['average']:.4f} |\n"
#             )

#         f.write("\n## Interpretation\n\n")
#         if "hybrid_kag" in all_results and "naive_vector" in all_results:
#             h = all_results["hybrid_kag"]
#             n = all_results["naive_vector"]
#             f.write("### Key Findings\n\n")
#             f.write(
#                 f"- **Faithfulness**: Hybrid KAG {h['faithfulness']:.4f} "
#                 f"vs naive {n['faithfulness']:.4f} "
#                 f"({h['faithfulness']-n['faithfulness']:+.4f})\n"
#             )
#             f.write(
#                 f"- **Context Precision**: "
#                 f"{h['context_precision']-n['context_precision']:+.4f} "
#                 f"improvement from graph-guided retrieval\n"
#             )
#             f.write(
#                 f"- **Context Recall**: "
#                 f"{h['context_recall']-n['context_recall']:+.4f} "
#                 f"from graph traversal\n"
#             )
#             f.write(
#                 "\n### Why Graph Helps on Multi-Hop Questions\n\n"
#                 "The knowledge graph enables traversal across paper "
#                 "relationships — finding evidence through shared methods, "
#                 "datasets, and citation chains that pure vector search misses.\n"
#             )

#     print(f"  Saved to {output_path}")


# # ── Full evaluation runner ────────────────────────────────────

# def run_full_evaluation(
#     subset_size: int = 9,
#     delay_between_calls: float = 15.0,
#     save_samples: bool = True,
# ):
#     """Run full evaluation across all 3 configurations."""
#     from eval.golden_dataset import GOLDEN_DATASET

#     print("="*70)
#     print("EVALUATION — Day 11")
#     print("="*70)

#     factual = [q for q in GOLDEN_DATASET if q["type"] == "factual"]
#     comparison = [q for q in GOLDEN_DATASET if q["type"] == "comparison"]
#     multi_hop = [q for q in GOLDEN_DATASET if q["type"] == "multi_hop"]

#     per_type = max(1, subset_size // 3)
#     subset = (
#         factual[:per_type] +
#         comparison[:per_type] +
#         multi_hop[:per_type]
#     )

#     print(f"\nSubset: {len(subset)} questions ({per_type} per type)")

#     all_results = {}
#     all_samples = {}

#     configs = [
#         ("naive_vector",  False, 20, 20),
#         ("vector_rerank", False, 20,  8),
#         ("hybrid_kag",    True,  20,  8),
#     ]

#     for config_name, use_graph, top_k_r, top_k_rr in configs:
#         print(f"\n{'─'*50}")
#         print(f"CONFIG: {config_name}")
#         print(f"{'─'*50}")

#         samples = []
#         for i, item in enumerate(subset):
#             print(f"  [{i+1}/{len(subset)}] {item['question'][:50]}...")
#             result = run_pipeline_for_eval(
#                 item["question"],
#                 use_graph=use_graph,
#                 top_k_retrieve=top_k_r,
#                 top_k_rerank=top_k_rr,
#             )
#             result["ground_truth"] = item["ground_truth"]
#             samples.append(result)
#             if i < len(subset) - 1:
#                 time.sleep(delay_between_calls)

#         all_samples[config_name] = samples
#         scores = evaluate_with_ragas(samples, config_name)
#         all_results[config_name] = scores

#         if config_name != "hybrid_kag":
#             print(f"  Waiting 30s before next config...")
#             time.sleep(30)

#     if save_samples:
#         with open("data/ragas_samples.json", "w") as f:
#             json.dump(all_samples, f, indent=2)

#     with open("data/ragas_scores.json", "w") as f:
#         json.dump(all_results, f, indent=2)

#     save_results_markdown(all_results, len(subset))

#     print("\n" + "="*70)
#     print("FINAL RESULTS")
#     print("="*70)
#     print(f"\n{'Config':<25} {'Faith':>8} {'Rel':>8} "
#           f"{'Prec':>8} {'Recall':>8} {'Avg':>8}")
#     print("-"*70)
#     for config_name, _, _, _ in configs:
#         if config_name in all_results:
#             s = all_results[config_name]
#             print(
#                 f"{config_name:<25} "
#                 f"{s['faithfulness']:>8.4f} "
#                 f"{s['answer_relevancy']:>8.4f} "
#                 f"{s['context_precision']:>8.4f} "
#                 f"{s['context_recall']:>8.4f} "
#                 f"{s['average']:>8.4f}"
#             )
#     print(f"\n  Results: docs/ragas_results.md")
#     print("="*70)
#     return all_results