"""
LLM answer generation using Gemini 2.5 Flash.

Takes reranked evidence chunks and generates a cited, grounded answer.
Key design decisions:
- Instruct the LLM to cite which chunk supports each claim
- Instruct it to say "not found in evidence" rather than hallucinate
- Return structured output: answer + citations + confidence
"""

import os
import logging
import time
import json
import re
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_client = None
_backend = None

def _get_gemini_client():
    """Lazily load Gemini client, supporting both google.genai and google.generativeai."""
    global _client, _backend
    if _client is not None:
        return _client, _backend

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in your .env file.")

    # Try modern google-genai SDK first
    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        _backend = "genai"
        logger.info("Initialized Gemini via google-genai SDK.")
        return _client, _backend
    except ImportError:
        pass

    # Try legacy google-generativeai SDK
    try:
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        _client = legacy_genai.GenerativeModel(model_name)
        _backend = "google.generativeai"
        logger.info("Initialized Gemini via google-generativeai SDK.")
        return _client, _backend
    except ImportError:
        raise ImportError(
            "Neither google-genai nor google-generativeai is installed. "
            "Please run: pip install google-genai"
        )

@dataclass
class GeneratedAnswer:
    """Structured output from the generator."""
    query: str
    answer: str
    citations: list[dict]      # [{chunk_id, arxiv_id, title, quote}]
    evidence_used: list[str]   # chunk_ids actually referenced
    confidence: str            # "high", "medium", "low"
    not_found: bool            # True if answer couldn't be found in evidence
    latency_ms: int
    model: str = "gemini-2.5-flash"

GENERATION_PROMPT = """You are a scientific research assistant helping answer questions about computer vision and deep learning papers.

You will be given a question and a set of numbered evidence chunks extracted from research papers.

YOUR TASK:
1. Answer the question using ONLY the provided evidence
2. For every factual claim, cite the evidence chunk number [1], [2], etc.
3. If the evidence does not contain enough information to answer, say exactly: "The evidence does not contain sufficient information to answer this question."
4. Do NOT use any knowledge outside the provided evidence
5. Be specific and technical — this is for researchers

EVIDENCE:
{evidence_text}

QUESTION: {query}

ANSWER FORMAT:
- Write a clear, detailed answer with inline citations like [1], [2]
- After the answer, list: CITATIONS: chunk_id → paper_title for each chunk you referenced
- End with CONFIDENCE: high/medium/low based on how well the evidence answers the question

Begin your answer now:"""
def safe_get(chunk, attr, default=None):
    """Works for both dict and object."""
    if isinstance(chunk, dict):
        return chunk.get(attr, default)
    return getattr(chunk, attr, default)

def format_evidence(chunks: list[Any]) -> Tuple[str, Dict[int, dict]]:
    """
    Format chunks into numbered evidence text for the prompt.

    Returns:
        (evidence_text, chunk_lookup) where chunk_lookup maps
        number → chunk metadata for citation building
    """

    lines = []
    chunk_lookup = {}

    for i, chunk in enumerate(chunks, start=1):
        # Safe attribute access
        chunk_id = safe_get(chunk, "chunk_id", f"chunk_{i}")
        title = safe_get(chunk, "title", "Unknown")
        year = safe_get(chunk, "year", "")
        section = safe_get(chunk, "section", "")
        text = safe_get(chunk, "text", "")
        arxiv_id = safe_get(chunk, "arxiv_id", "")

        lines.append(
            f"[{i}] Source: {title} ({year}) — {section}\n"
            f"     {text[:500]}"
        )

        chunk_lookup[i] = {
            "chunk_id": chunk_id,
            "arxiv_id": arxiv_id,
            "title": title,
            "year": year,
            "section": section,
        }

    evidence_text = "\n\n".join(lines)
    return evidence_text, chunk_lookup


def parse_citations_from_answer(
    answer_text: str,
    chunk_lookup: dict,
) -> tuple[str, list[dict], list[str]]:
    """
    Extract citation numbers from answer text and build citation list.

    Returns:
        (clean_answer, citations, evidence_used_ids)
    """
    # Find all citation numbers in the answer [1], [2], etc.
    cited_numbers = set(
        int(m) for m in re.findall(r'\[(\d+)\]', answer_text)
        if int(m) in chunk_lookup
    )

    citations = []
    evidence_used = []

    for num in sorted(cited_numbers):
        chunk_info = chunk_lookup[num]
        citations.append({
            "citation_number": num,
            "chunk_id": chunk_info["chunk_id"],
            "arxiv_id": chunk_info["arxiv_id"],
            "title": chunk_info["title"],
            "year": chunk_info["year"],
        })
        evidence_used.append(chunk_info["chunk_id"])

    return answer_text, citations, evidence_used

def determine_confidence(answer_text: str, citations: list) -> str:
    """Determine confidence level from answer content."""
    answer_lower = answer_text.lower()

    # Extract explicit confidence if model included it
    conf_match = re.search(
        r'confidence:\s*(high|medium|low)', answer_lower
    )
    if conf_match:
        return conf_match.group(1)

    # Infer from citations and content
    not_found_phrases = [
        "does not contain", "insufficient", "not found",
        "cannot answer", "no information"
    ]
    if any(p in answer_lower for p in not_found_phrases):
        return "low"
    if len(citations) >= 3:
        return "high"
    if len(citations) >= 1:
        return "medium"
    return "low"

def generate_answer(
    query: str,
    ranked_chunks: list,
    max_chunks: int = 8,
    calibration=None,
    symbolic_constraints: Optional[dict] = None,
) -> GeneratedAnswer:
    """
    Generate a cited answer from reranked evidence chunks,
    constrained by symbolic knowledge graph rules and calibrated uncertainty.

    Args:
        query: the user's question
        ranked_chunks: list of RankedResult objects (from reranker)
        max_chunks: max evidence chunks to include in prompt
        calibration: optional CalibrationResult for calibrated uncertainty & abstention
        symbolic_constraints: optional graph constraints dict

    Returns:
        GeneratedAnswer with answer text, citations, confidence
    """
    start = time.time()

    # Check for calibrated abstention ("Not Identified" / Refusal)
    if calibration is not None and getattr(calibration, "should_abstain", False):
        abstention_msg = getattr(calibration, "abstention_message", "Evidence insufficient to identify answer.")
        answer_text = (
            f"**[CALIBRATED REFUSAL — NOT IDENTIFIED]**\n\n"
            f"{abstention_msg}\n\n"
            f"*Uncertainty Breakdown:*\n"
            f"- Calibrated Confidence: {int(calibration.confidence_score * 100)}%\n"
            f"- Dense Vector Support: {int(calibration.dense_support * 100)}%\n"
            f"- Symbolic Graph Grounding: {int(calibration.symbolic_grounding * 100)}%\n"
            f"- Margin Separation: {int(calibration.margin_score * 100)}%"
        )
        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=[],
            evidence_used=[],
            confidence="low",
            not_found=True,
            latency_ms=int((time.time() - start) * 1000),
        )

    # Use top max_chunks
    evidence_chunks = ranked_chunks[:max_chunks]

    if not evidence_chunks:
        return GeneratedAnswer(
            query=query,
            answer="No evidence chunks available to answer this question in the verified corpus.",
            citations=[],
            evidence_used=[],
            confidence="low",
            not_found=True,
            latency_ms=0,
        )

    # Format evidence for prompt
    evidence_text, chunk_lookup = format_evidence(evidence_chunks)

    # Build prompt with optional symbolic knowledge graph constraints
    constraint_text = ""
    if symbolic_constraints:
        from retrieval.symbolic_verifier import get_symbolic_verifier
        constraint_text = get_symbolic_verifier().format_constraints_for_prompt(symbolic_constraints)

    prompt = GENERATION_PROMPT.format(
        evidence_text=evidence_text,
        query=query,
    )
    if constraint_text:
        prompt = prompt + "\n\n" + constraint_text

    try:
        delay = float(os.getenv("GEMINI_RATE_LIMIT_DELAY", "0"))
        if delay > 0:
            time.sleep(delay)

        client, backend = _get_gemini_client()
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if backend == "genai":
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            answer_text = response.text.strip()
        else:
            response = client.generate_content(prompt)
            answer_text = response.text.strip()

        # Parse citations
        answer_text, citations, evidence_used = parse_citations_from_answer(
            answer_text, chunk_lookup
        )

        # Check if not found
        not_found = any(phrase in answer_text.lower() for phrase in [
            "does not contain sufficient",
            "not found in evidence",
            "cannot answer",
        ])

        confidence = determine_confidence(answer_text, citations)
        elapsed_ms = int((time.time() - start) * 1000)

        logger.info(
            f"Generator: answer generated in {elapsed_ms}ms, "
            f"{len(citations)} citations, confidence={confidence}"
        )

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            evidence_used=evidence_used,
            confidence=confidence,
            not_found=not_found,
            latency_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        logger.error(f"Generation failed: {e}")
        return GeneratedAnswer(
            query=query,
            answer=f"Generation failed: {e}",
            citations=[],
            evidence_used=[],
            confidence="low",
            not_found=True,
            latency_ms=elapsed_ms,
        )

def pretty_print_answer(result: GeneratedAnswer):
    """Print a GeneratedAnswer in a readable format."""
    print(f"\n{'='*70}")
    print(f"QUERY: {result.query}")
    print(f"{'='*70}")
    print(f"\nANSWER ({result.confidence.upper()} confidence, "
          f"{result.latency_ms}ms):\n")
    # Strip the CITATIONS and CONFIDENCE suffix for clean display
    clean = re.sub(r'\nCITATIONS:.*', '', result.answer, flags=re.DOTALL)
    clean = re.sub(r'\nCONFIDENCE:.*', '', clean, flags=re.DOTALL)
    print(clean.strip())

    if result.citations:
        print(f"\nCITATIONS ({len(result.citations)}):")
        for c in result.citations:
            print(f"  [{c['citation_number']}] {c['title'][:55]} "
                  f"({c['year']}) — arxiv:{c['arxiv_id']}")

    print(f"\nNot found in evidence: {result.not_found}")
    print(f"Confidence: {result.confidence}")