"""
Query Decomposition.

Detects multi-part queries and splits them into focused sub-queries,
each run through the retriever independently, then synthesized.

Triggers on:
- "compare X and Y"
- "X versus Y"
- "differences between X and Y"
- "how does X differ from Y"
- "X and Y" (when both are known entities)
- "what are the pros and cons of X"

CV number to track:
- % of queries detected as multi-part
- Answer quality improvement on comparison questions
"""

import os
import re
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash")

# Patterns that signal a multi-part query
COMPARISON_PATTERNS = [
    r'\bcompare\b',
    r'\bversus\b',
    r'\bvs\.?\b',
    r'\bdifference[s]?\s+between\b',
    r'\bhow\s+does\s+\w+\s+differ\b',
    r'\bcontrast\b',
    r'\bpros\s+and\s+cons\b',
    r'\badvantages\s+and\s+disadvantages\b',
]

MULTI_PART_PATTERNS = [
    r'\band\s+also\b',
    r'\bas\s+well\s+as\b',
    r'\bin\s+addition\s+to\b',
    r'\bboth\s+\w+\s+and\b',
]

DECOMPOSE_PROMPT = """You are a query decomposition specialist.

Given a complex multi-part question, decompose it into 2-3 simpler,
focused sub-questions. Each sub-question should be independently 
answerable and together they should fully answer the original question.

Original question: {query}

Return ONLY a JSON object with no markdown:
{{
  "is_multi_part": true,
  "sub_queries": [
    "focused sub-question 1",
    "focused sub-question 2"
  ],
  "synthesis_instruction": "one sentence on how to combine the answers"
}}

If the question is simple and doesn't need decomposition:
{{
  "is_multi_part": false,
  "sub_queries": ["{query}"],
  "synthesis_instruction": "answer directly"
}}"""

SYNTHESIS_PROMPT = """You are synthesizing answers to sub-questions
into a single coherent answer to the original question.

Original question: {original_query}

Sub-answers:
{sub_answers}

Synthesis instruction: {synthesis_instruction}

Write a comprehensive answer that integrates all sub-answers.
Use inline citations like [1], [2] from the sub-answers.
Be specific and technical.

Answer:"""


def is_multi_part_query(query: str) -> bool:
    """
    Quick heuristic check for multi-part queries.
    No LLM call — pure regex for speed.
    """
    query_lower = query.lower()

    for pattern in COMPARISON_PATTERNS + MULTI_PART_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    # Check for "X and Y" where X and Y are likely named entities
    # (both capitalized words)
    cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    if len(cap_words) >= 2:
        # Check if "and" appears between them
        if " and " in query or " vs " in query or " versus " in query:
            return True

    return False


def decompose_query(query: str) -> dict:
    """
    Decompose a multi-part query into sub-queries using Gemini.

    Returns:
        {
            "is_multi_part": bool,
            "sub_queries": [str, ...],
            "synthesis_instruction": str,
            "method": "llm" | "heuristic" | "passthrough"
        }
    """
    # Fast heuristic check first
    if not is_multi_part_query(query):
        logger.info(f"Query not multi-part: '{query[:50]}'")
        return {
            "is_multi_part": False,
            "sub_queries": [query],
            "synthesis_instruction": "answer directly",
            "method": "passthrough",
        }

    # LLM decomposition
    try:
        prompt = DECOMPOSE_PROMPT.format(query=query)
        response = _model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown fences
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = re.sub(r',\s*([}\]])', r'\1', raw)  # trailing commas
        raw = raw.strip()

        data = __import__('json').loads(raw)
        data["method"] = "llm"

        logger.info(
            f"Decomposed '{query[:40]}' into "
            f"{len(data['sub_queries'])} sub-queries"
        )
        return data

    except Exception as e:
        logger.error(f"LLM decomposition failed: {e}")
        # Heuristic fallback
        return _heuristic_decompose(query)


def _heuristic_decompose(query: str) -> dict:
    """
    Fallback decomposition without LLM.
    Handles "compare X and Y" patterns directly.
    """
    query_lower = query.lower()

    # "compare X and Y" or "X vs Y"
    for pattern, sep in [
        (r'compare\s+(.+?)\s+and\s+(.+)', 'and'),
        (r'(.+?)\s+vs\.?\s+(.+)', 'vs'),
        (r'(.+?)\s+versus\s+(.+)', 'versus'),
        (r'differences?\s+between\s+(.+?)\s+and\s+(.+)', 'diff'),
    ]:
        match = re.search(pattern, query_lower)
        if match:
            part1 = match.group(1).strip()
            part2 = match.group(2).strip()
            return {
                "is_multi_part": True,
                "sub_queries": [
                    f"What is {part1} and how does it work?",
                    f"What is {part2} and how does it work?",
                ],
                "synthesis_instruction": (
                    f"Compare {part1} and {part2} by contrasting "
                    f"their approaches, strengths, and use cases"
                ),
                "method": "heuristic",
            }

    # Generic fallback — treat as single query
    return {
        "is_multi_part": False,
        "sub_queries": [query],
        "synthesis_instruction": "answer directly",
        "method": "passthrough",
    }


def synthesize_sub_answers(
    original_query: str,
    sub_queries: list[str],
    sub_answers: list[str],
    synthesis_instruction: str,
    sub_citations: list[list] = None,
) -> str:
    """
    Synthesize multiple sub-answers into one coherent answer.

    Args:
        original_query: the user's original question
        sub_queries: list of decomposed sub-questions
        sub_answers: list of answers to each sub-question
        synthesis_instruction: how to combine them
        sub_citations: citations from each sub-answer

    Returns:
        Synthesized answer string
    """
    # Format sub-answers
    sub_answers_text = ""
    for i, (sq, sa) in enumerate(zip(sub_queries, sub_answers), 1):
        sub_answers_text += (
            f"\n[Sub-question {i}]: {sq}\n"
            f"[Answer {i}]: {sa[:600]}\n"
        )

    try:
        prompt = SYNTHESIS_PROMPT.format(
            original_query=original_query,
            sub_answers=sub_answers_text,
            synthesis_instruction=synthesis_instruction,
        )
        response = _model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        # Fallback: concatenate
        parts = []
        for i, (sq, sa) in enumerate(zip(sub_queries, sub_answers), 1):
            parts.append(f"**{sq}**\n\n{sa}")
        return "\n\n---\n\n".join(parts)


def run_decomposed_pipeline(
    query: str,
    engine=None,
) -> dict:
    """
    Full pipeline with decomposition.

    If multi-part: decompose → run each sub-query → synthesize
    If single: run normally

    Returns:
        {
            "answer": str,
            "was_decomposed": bool,
            "sub_queries": list,
            "sub_answers": list,
            "all_citations": list,
            "decomposition_method": str,
            "latency_ms": int,
        }
    """
    from retrieval.pipeline import run_pipeline

    start = time.time()

    # Step 1: Detect and decompose
    decomp = decompose_query(query)

    if not decomp["is_multi_part"]:
        # Single query — run normally
        result = run_pipeline(query, use_graph=True)
        elapsed = int((time.time() - start) * 1000)

        import re as re_module
        clean = re_module.sub(
            r'\nCITATIONS:.*', '',
            result.answer.answer,
            flags=re_module.DOTALL
        )
        clean = re_module.sub(
            r'\nCONFIDENCE:.*', '', clean, flags=re_module.DOTALL
        )

        return {
            "answer": clean.strip(),
            "was_decomposed": False,
            "sub_queries": [query],
            "sub_answers": [clean.strip()],
            "all_citations": result.answer.citations,
            "decomposition_method": decomp["method"],
            "latency_ms": elapsed,
        }

    # Step 2: Run each sub-query
    sub_queries = decomp["sub_queries"]
    sub_answers = []
    all_citations = []

    logger.info(
        f"Decomposed into {len(sub_queries)} sub-queries: "
        f"{sub_queries}"
    )

    for i, sub_query in enumerate(sub_queries):
        logger.info(
            f"Running sub-query {i+1}/{len(sub_queries)}: "
            f"'{sub_query[:50]}'"
        )
        try:
            result = run_pipeline(sub_query, use_graph=True)

            import re as re_module
            clean = re_module.sub(
                r'\nCITATIONS:.*', '',
                result.answer.answer,
                flags=re_module.DOTALL
            )
            clean = re_module.sub(
                r'\nCONFIDENCE:.*', '', clean,
                flags=re_module.DOTALL
            )
            sub_answers.append(clean.strip())
            all_citations.extend(result.answer.citations)

            # Rate limit between sub-queries
            if i < len(sub_queries) - 1:
                time.sleep(5)

        except Exception as e:
            logger.error(f"Sub-query {i+1} failed: {e}")
            sub_answers.append(f"Could not retrieve answer: {e}")

    # Step 3: Synthesize
    synthesized = synthesize_sub_answers(
        original_query=query,
        sub_queries=sub_queries,
        sub_answers=sub_answers,
        synthesis_instruction=decomp["synthesis_instruction"],
    )

    elapsed = int((time.time() - start) * 1000)

    return {
        "answer": synthesized,
        "was_decomposed": True,
        "sub_queries": sub_queries,
        "sub_answers": sub_answers,
        "all_citations": all_citations,
        "decomposition_method": decomp["method"],
        "latency_ms": elapsed,
    }