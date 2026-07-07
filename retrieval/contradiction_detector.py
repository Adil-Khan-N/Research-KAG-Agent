"""
Contradiction Detector.

For retrieved chunks from different papers discussing the same entity
(Method or Dataset), checks if they make opposing claims.

Two modes:
1. LLM-based: accurate, costs API calls, used for final answer
2. Keyword-based: fast, zero cost, used for quick flagging

When contradiction found:
- Flag in the answer response
- Optionally write CONTRADICTS edge to Neo4j (feeds Day 19 auto-updater)

CV demo: query about "data requirements for transformers" flags
ViT ("needs large data") vs DeiT ("works with small data")
"""

import os
import re
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash")


# Keywords that signal opposing claims
CONTRADICTION_SIGNALS = [
    # Opposing performance claims
    ("outperforms", "underperforms"),
    ("better than", "worse than"),
    ("superior to", "inferior to"),
    ("improves over", "degrades from"),
    # Opposing requirement claims
    ("requires large", "requires small"),
    ("needs large", "needs small"),
    ("data-hungry", "data-efficient"),
    ("large dataset", "small dataset"),
    # Opposing architectural claims
    ("with convolutions", "without convolutions"),
    ("global attention", "local attention"),
    ("quadratic complexity", "linear complexity"),
    # Opposing conclusion claims
    ("we show that", "we demonstrate that"),
    ("contradicts", "confirms"),
    ("challenges", "supports"),
]

CONTRADICTION_PROMPT = """You are checking if two excerpts from different
research papers make opposing or contradictory claims.

Paper A: "{title_a}" ({year_a})
Excerpt A: {text_a}

Paper B: "{title_b}" ({year_b})
Excerpt B: {text_b}

Do these excerpts make opposing or contradictory claims about the
same concept, method, or finding?

Consider:
- Different conclusions about the same experiment
- Opposing claims about data requirements
- Contradicting performance comparisons
- Different assertions about architectural trade-offs

Respond with ONLY a JSON object:
{{
  "contradicts": true/false,
  "confidence": "high/medium/low",
  "claim_a": "what paper A claims (one sentence)",
  "claim_b": "what paper B claims (one sentence)",
  "topic": "what they disagree about"
}}"""


@dataclass
class ContradictionResult:
    """Result of contradiction check between two chunks."""
    contradicts: bool
    confidence: str          # "high", "medium", "low"
    claim_a: str
    claim_b: str
    topic: str
    paper_a: str
    paper_b: str
    arxiv_id_a: str
    arxiv_id_b: str
    method: str              # "llm" or "keyword"


def _keyword_contradiction_check(text_a: str, text_b: str) -> bool:
    """
    Fast keyword-based contradiction check.
    No API call — used for quick pre-filtering.
    Returns True if texts likely contradict based on keyword patterns.
    """
    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()

    for word_a, word_b in CONTRADICTION_SIGNALS:
        if word_a in text_a_lower and word_b in text_b_lower:
            return True
        if word_b in text_a_lower and word_a in text_b_lower:
            return True

    return False


def check_contradiction_llm(
    chunk_a: dict,
    chunk_b: dict,
) -> ContradictionResult:
    """
    LLM-based contradiction check between two chunks.

    Args:
        chunk_a: dict with keys: text, title, year, arxiv_id
        chunk_b: dict with keys: text, title, year, arxiv_id

    Returns:
        ContradictionResult
    """
    import json

    prompt = CONTRADICTION_PROMPT.format(
        title_a=chunk_a.get("title", "Unknown"),
        year_a=chunk_a.get("year", ""),
        text_a=chunk_a.get("text", "")[:400],
        title_b=chunk_b.get("title", "Unknown"),
        year_b=chunk_b.get("year", ""),
        text_b=chunk_b.get("text", "")[:400],
    )

    try:
        response = _model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        data = json.loads(raw)

        return ContradictionResult(
            contradicts=data.get("contradicts", False),
            confidence=data.get("confidence", "low"),
            claim_a=data.get("claim_a", ""),
            claim_b=data.get("claim_b", ""),
            topic=data.get("topic", ""),
            paper_a=chunk_a.get("title", ""),
            paper_b=chunk_b.get("title", ""),
            arxiv_id_a=chunk_a.get("arxiv_id", ""),
            arxiv_id_b=chunk_b.get("arxiv_id", ""),
            method="llm",
        )

    except Exception as e:
        logger.error(f"LLM contradiction check failed: {e}")
        return ContradictionResult(
            contradicts=False,
            confidence="low",
            claim_a="",
            claim_b="",
            topic="",
            paper_a=chunk_a.get("title", ""),
            paper_b=chunk_b.get("title", ""),
            arxiv_id_a=chunk_a.get("arxiv_id", ""),
            arxiv_id_b=chunk_b.get("arxiv_id", ""),
            method="llm_failed",
        )


def detect_contradictions(
    chunks: list,
    use_llm: bool = True,
    max_pairs: int = 5,
    delay: float = 3.0,
) -> list[ContradictionResult]:
    """
    Detect contradictions across a list of retrieved chunks.

    Strategy:
    1. Group chunks by arxiv_id (same paper = no contradiction)
    2. For each cross-paper pair, keyword check first
    3. If keyword check flags it, run LLM check
    4. Return all confirmed contradictions

    Args:
        chunks: list of RankedResult or dict objects
        use_llm: whether to use LLM for confirmation
        max_pairs: max pairs to check (prevents excessive API calls)
        delay: seconds between LLM calls

    Returns:
        list of ContradictionResult where contradicts=True
    """
    if len(chunks) < 2:
        return []

    # Normalize chunks to dicts
    normalized = []
    for c in chunks:
        if hasattr(c, "text"):
            # RankedResult object
            normalized.append({
                "text":     c.text,
                "title":    c.title,
                "year":     c.year,
                "arxiv_id": c.arxiv_id,
                "section":  getattr(c, "section", ""),
            })
        else:
            normalized.append(c)

    # Group by paper
    papers = {}
    for chunk in normalized:
        arxiv_id = chunk.get("arxiv_id", "")
        if arxiv_id not in papers:
            papers[arxiv_id] = []
        papers[arxiv_id].append(chunk)

    if len(papers) < 2:
        return []

    # Check cross-paper pairs
    paper_ids = list(papers.keys())
    contradictions = []
    pairs_checked = 0

    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            if pairs_checked >= max_pairs:
                break

            id_a = paper_ids[i]
            id_b = paper_ids[j]

            # Use best chunk from each paper
            chunk_a = papers[id_a][0]
            chunk_b = papers[id_b][0]

            # Fast keyword check first
            keyword_flag = _keyword_contradiction_check(
                chunk_a["text"], chunk_b["text"]
            )

            if keyword_flag or use_llm:
                if use_llm:
                    result = check_contradiction_llm(chunk_a, chunk_b)
                    pairs_checked += 1
                    if result.contradicts:
                        contradictions.append(result)
                        logger.info(
                            f"Contradiction found: "
                            f"{result.paper_a[:30]} vs "
                            f"{result.paper_b[:30]} "
                            f"on '{result.topic}'"
                        )
                    if pairs_checked < max_pairs:
                        time.sleep(delay)
                elif keyword_flag:
                    # Keyword-only result
                    contradictions.append(ContradictionResult(
                        contradicts=True,
                        confidence="low",
                        claim_a=chunk_a["text"][:100],
                        claim_b=chunk_b["text"][:100],
                        topic="keyword-detected",
                        paper_a=chunk_a.get("title", ""),
                        paper_b=chunk_b.get("title", ""),
                        arxiv_id_a=id_a,
                        arxiv_id_b=id_b,
                        method="keyword",
                    ))

    logger.info(
        f"Checked {pairs_checked} pairs, "
        f"found {len(contradictions)} contradictions"
    )
    return contradictions


def write_contradicts_to_neo4j(
    contradiction: ContradictionResult,
):
    """
    Write a CONTRADICTS relationship to Neo4j.
    Called when a high-confidence contradiction is found.
    Feeds Day 19's auto graph updater.
    """
    if not contradiction.contradicts:
        return
    if contradiction.confidence == "low":
        return

    try:
        from graph.neo4j_client import get_neo4j_client
        client = get_neo4j_client()
        client.run_write("""
            MATCH (p1:Paper {arxiv_id: $id_a})
            MATCH (p2:Paper {arxiv_id: $id_b})
            MERGE (p1)-[r:CONTRADICTS]->(p2)
            SET r.topic = $topic,
                r.confidence = $confidence,
                r.detected_at = datetime()
        """, {
            "id_a":       contradiction.arxiv_id_a,
            "id_b":       contradiction.arxiv_id_b,
            "topic":      contradiction.topic,
            "confidence": contradiction.confidence,
        })
        logger.info(
            f"Wrote CONTRADICTS edge: "
            f"{contradiction.arxiv_id_a} → {contradiction.arxiv_id_b}"
        )
    except Exception as e:
        logger.error(f"Failed to write CONTRADICTS to Neo4j: {e}")


def format_contradiction_for_display(
    contradictions: list[ContradictionResult],
) -> str:
    """Format contradictions for inclusion in the answer."""
    if not contradictions:
        return ""

    lines = ["\n\n⚠️ **Contradictions Detected in Evidence:**\n"]
    for c in contradictions:
        if c.contradicts:
            lines.append(
                f"- **{c.paper_a[:40]}** claims: {c.claim_a}\n"
                f"  **{c.paper_b[:40]}** claims: {c.claim_b}\n"
                f"  *Disagreement on: {c.topic}* "
                f"(confidence: {c.confidence})\n"
            )
    return "\n".join(lines)