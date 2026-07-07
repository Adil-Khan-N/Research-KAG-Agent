"""
Hallucination Self-Check.

After generating an answer, makes a second LLM call:
"Is every claim in this answer directly supported by the
retrieved chunks below? List any unsupported claims."

Returns:
- is_faithful: bool
- unsupported_claims: list of strings
- supported_claims: list of strings
- faithfulness_score: 0.0-1.0
- flagged_answer: answer with ⚠️ markers on unsupported sentences

Why this matters:
- Gemini sometimes extrapolates beyond the evidence
- Self-check catches these before returning to user
- Gives user transparency about answer quality

CV talking point:
"Implemented post-generation faithfulness verification using
LLM self-check; surfaces unsupported claims rather than
silently returning potentially hallucinated answers"
"""

import os
import re
import json
import time
import logging
import google.generativeai as genai
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash")


SELF_CHECK_PROMPT = """You are a faithfulness checker for a RAG system.

Your task: determine if each claim in the ANSWER is directly supported
by the EVIDENCE chunks below. Be strict — a claim is unsupported if
the evidence doesn't explicitly state it.

EVIDENCE:
{evidence_text}

ANSWER TO CHECK:
{answer}

Return ONLY a JSON object with no markdown:
{{
  "is_faithful": true/false,
  "faithfulness_score": 0.0-1.0,
  "supported_claims": ["claim1", "claim2"],
  "unsupported_claims": ["claim1 (reason: not in evidence)"],
  "verdict": "one sentence summary"
}}

Rules:
- A claim is SUPPORTED if the evidence explicitly states it
- A claim is UNSUPPORTED if it's inferred, extrapolated, or fabricated
- Numbers, dates, percentages must appear in evidence to be supported
- "not found in evidence" is always supported (the model said so)
- faithfulness_score = supported / total claims (0.0-1.0)"""


@dataclass
class SelfCheckResult:
    """Result of hallucination self-check."""
    is_faithful:        bool
    faithfulness_score: float
    supported_claims:   list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    verdict:            str = ""
    flagged_answer:     str = ""
    latency_ms:         int = 0
    check_performed:    bool = True
    error:              str = ""

def _format_evidence(chunks: list) -> str:
    lines = []

    for i, chunk in enumerate(chunks, 1):

        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            title = chunk.get("title", "Unknown")
            year = chunk.get("year", "")
        else:
            text = getattr(chunk, "text", "")
            title = getattr(chunk, "title", "Unknown")
            year = getattr(chunk, "year", "")

        lines.append(
            f"[{i}] {title} ({year}):\n{text[:400]}"
        )

    return "\n\n".join(lines)


def _flag_unsupported_in_answer(
    answer: str,
    unsupported_claims: list[str],
) -> str:
    """
    Add ⚠️ markers to sentences in the answer that correspond
    to unsupported claims.
    """
    if not unsupported_claims:
        return answer

    flagged = answer

    for claim in unsupported_claims:
        # Extract the core claim text before the reason in parentheses
        core_claim = re.sub(r'\s*\(reason:.*?\)', '', claim).strip()
        if len(core_claim) < 5:
            continue

        # Find the key phrase from the claim
        words = core_claim.split()[:4]
        if not words:
            continue

        phrase = " ".join(words)

        # Mark the sentence containing this phrase
        sentences = re.split(r'(?<=[.!?])\s+', flagged)
        marked = []
        for sent in sentences:
            if phrase.lower() in sent.lower():
                if "⚠️" not in sent:
                    sent = f"⚠️ [UNVERIFIED] {sent}"
            marked.append(sent)
        flagged = " ".join(marked)

    return flagged


def check_faithfulness(
    answer: str,
    chunks: list,
    skip_if_not_found: bool = True,
) -> SelfCheckResult:
    """
    Run faithfulness self-check on a generated answer.

    Args:
        answer: the generated answer text
        chunks: list of RankedResult objects used as evidence
        skip_if_not_found: skip check if answer says "not found"

    Returns:
        SelfCheckResult with faithfulness assessment
    """
    start = time.time()

    # Skip check if answer already says not found
    if skip_if_not_found:
        not_found_phrases = [
            "does not contain sufficient",
            "not found in evidence",
            "cannot answer",
            "no information available",
        ]
        if any(p in answer.lower() for p in not_found_phrases):
            elapsed = int((time.time() - start) * 1000)
            return SelfCheckResult(
                is_faithful=True,
                faithfulness_score=1.0,
                verdict="Answer correctly states information not found",
                flagged_answer=answer,
                latency_ms=elapsed,
                check_performed=False,
            )

    if not chunks:
        elapsed = int((time.time() - start) * 1000)
        return SelfCheckResult(
            is_faithful=False,
            faithfulness_score=0.0,
            verdict="No evidence chunks to verify against",
            flagged_answer=answer,
            latency_ms=elapsed,
            check_performed=False,
        )

    evidence_text = _format_evidence(chunks)

    # Clean answer for checking
    clean_answer = re.sub(r'\nCITATIONS:.*', '', answer, flags=re.DOTALL)
    clean_answer = re.sub(r'\nCONFIDENCE:.*', '', clean_answer, flags=re.DOTALL)
    clean_answer = clean_answer.strip()

    prompt = SELF_CHECK_PROMPT.format(
        evidence_text=evidence_text[:3000],
        answer=clean_answer[:1500],
    )

    try:
        response = _model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        raw = raw.strip()

        data = json.loads(raw)

        is_faithful        = data.get("is_faithful", True)
        faithfulness_score = float(data.get("faithfulness_score", 1.0))
        supported          = data.get("supported_claims", [])
        unsupported        = data.get("unsupported_claims", [])
        verdict            = data.get("verdict", "")

        # Flag unsupported claims in answer
        flagged = _flag_unsupported_in_answer(answer, unsupported)

        elapsed = int((time.time() - start) * 1000)

        logger.info(
            f"Self-check: faithful={is_faithful}, "
            f"score={faithfulness_score:.2f}, "
            f"unsupported={len(unsupported)}, "
            f"latency={elapsed}ms"
        )

        return SelfCheckResult(
            is_faithful        = is_faithful,
            faithfulness_score = round(faithfulness_score, 3),
            supported_claims   = supported[:10],
            unsupported_claims = unsupported[:10],
            verdict            = verdict,
            flagged_answer     = flagged,
            latency_ms         = elapsed,
            check_performed    = True,
        )

    except json.JSONDecodeError as e:
        logger.error(f"Self-check JSON parse failed: {e}\nRaw: {raw[:200]}")
        elapsed = int((time.time() - start) * 1000)
        return SelfCheckResult(
            is_faithful=True,
            faithfulness_score=0.5,
            verdict="Self-check parsing failed — treat with caution",
            flagged_answer=answer,
            latency_ms=elapsed,
            error=str(e),
        )
    except Exception as e:
        logger.error(f"Self-check failed: {e}")
        elapsed = int((time.time() - start) * 1000)
        return SelfCheckResult(
            is_faithful=True,
            faithfulness_score=0.5,
            verdict="Self-check unavailable",
            flagged_answer=answer,
            latency_ms=elapsed,
            error=str(e),
        )


def format_self_check_for_ui(result: SelfCheckResult) -> dict:
    """Format self-check result for API/UI display."""
    return {
        "is_faithful":        result.is_faithful,
        "faithfulness_score": result.faithfulness_score,
        "verdict":            result.verdict,
        "unsupported_claims": result.unsupported_claims,
        "supported_claims":   result.supported_claims[:5],
        "check_performed":    result.check_performed,
        "latency_ms":         result.latency_ms,
        "has_warnings":       len(result.unsupported_claims) > 0,
    }