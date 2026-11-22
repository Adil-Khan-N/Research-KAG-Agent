"""
Calibrated Uncertainty & Epistemic Abstention Engine for KAG.

Provides calibrated confidence scoring, uncertainty breakdown, and
selective refusal ("Not Identified") when queries lack verifiable evidence
in either the dense corpus or symbolic knowledge graph.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

REFUSAL_THRESHOLD = 0.32
HIGH_CONF_THRESHOLD = 0.72
MEDIUM_CONF_THRESHOLD = 0.48


@dataclass
class CalibrationResult:
    """Calibrated confidence and epistemic uncertainty output."""
    confidence_score: float         # 0.0 to 1.0 calibrated probability
    confidence_band: str           # "HIGH", "MEDIUM", "LOW", "REFUSAL"
    should_abstain: bool           # True if system refuses to hallucinate
    abstention_message: Optional[str]
    dense_support: float           # Semantic vector similarity support (0-1)
    symbolic_grounding: float      # Graph entity & path connectivity (0-1)
    margin_score: float            # Evidence margin (gap between top and distractor)
    contradiction_risk: float      # Penalty for detected empirical contradictions
    unidentified_entities: List[str]
    verified_entities: List[str]
    metrics_summary: Dict[str, Any] = field(default_factory=dict)


def compute_calibration(
    query: str,
    retrieved_chunks: list,
    extracted_entities: Optional[dict] = None,
    graph_papers: Optional[list] = None,
    contradictions: Optional[list] = None,
    graph_vocab: Optional[dict] = None,
) -> CalibrationResult:
    """
    Compute multi-factor calibrated confidence and determine whether
    the system has sufficient grounding or should trigger calibrated refusal.
    """
    extracted_entities = extracted_entities or {}
    graph_papers = graph_papers or []
    contradictions = contradictions or []

    # 1. Dense vector support
    if retrieved_chunks:
        # Collect top similarity / rerank scores
        raw_scores = []
        for c in retrieved_chunks[:5]:
            score = getattr(c, "rerank_score", None)
            if score is None:
                score = getattr(c, "vector_score", None)
            if score is None:
                score = getattr(c, "similarity", None)
            if score is not None:
                raw_scores.append(float(score))
        if raw_scores:
            # Normalize to 0-1
            avg_top = sum(raw_scores[:3]) / max(len(raw_scores[:3]), 1)
            dense_support = max(0.0, min(1.0, (avg_top - 0.2) / 0.7))
        else:
            dense_support = 0.2
    else:
        dense_support = 0.0

    # 2. Symbolic knowledge graph grounding
    all_query_entities = extracted_entities.get("all", [])
    known_methods = set(extracted_entities.get("methods", []))
    known_datasets = set(extracted_entities.get("datasets", []))
    known_papers = set(extracted_entities.get("paper_titles", []))
    total_found_entities = len(known_methods) + len(known_datasets) + len(known_papers)

    verified_entities = []
    unidentified_entities = []

    if all_query_entities:
        for ent in all_query_entities:
            if ent in known_methods or ent in known_datasets or ent in known_papers:
                verified_entities.append(ent)
            else:
                unidentified_entities.append(ent)
        grounding_ratio = len(verified_entities) / max(len(all_query_entities), 1)
        # Factor in graph paper connectivity
        graph_connectivity = min(1.0, len(graph_papers) / 4.0)
        symbolic_grounding = 0.6 * grounding_ratio + 0.4 * graph_connectivity
    else:
        # No named entities detected in query (abstract/broad query)
        symbolic_grounding = 0.45 if graph_papers else 0.30

    # 3. Evidence Margin Score (Top 1 vs rank 5 / distractor)
    if len(retrieved_chunks) >= 3:
        top_score = getattr(retrieved_chunks[0], "rerank_score", None) or getattr(retrieved_chunks[0], "final_score", 0.5)
        tail_score = getattr(retrieved_chunks[-1], "rerank_score", None) or getattr(retrieved_chunks[-1], "final_score", 0.3)
        margin = max(0.0, min(1.0, (float(top_score) - float(tail_score)) * 2.5))
        margin_score = margin
    else:
        margin_score = 0.35

    # 4. Contradiction risk penalty
    if contradictions:
        contradiction_risk = min(0.35, 0.15 * len(contradictions))
    else:
        contradiction_risk = 0.0

    # 5. Composite calibrated confidence formula
    calibrated = (
        0.38 * dense_support +
        0.35 * symbolic_grounding +
        0.27 * margin_score -
        contradiction_risk
    )
    calibrated_score = round(max(0.05, min(0.98, calibrated)), 3)

    # 6. Epistemic Abstention / Refusal logic
    should_abstain = False
    abstention_msg = None

    # Condition A: Extremely low overall confidence
    if calibrated_score < REFUSAL_THRESHOLD:
        should_abstain = True
        abstention_msg = (
            f"REFUSAL / NOT IDENTIFIED (Calibrated Confidence: {int(calibrated_score*100)}%). "
            f"The verified research corpus (24 Vision Transformer papers) does not contain sufficient "
            f"empirical evidence or knowledge graph relations to reliably answer this query without hallucination."
        )

    # Condition B: High-specificity query with completely unknown/unidentified domain entities
    elif all_query_entities and len(verified_entities) == 0 and dense_support < 0.42:
        should_abstain = True
        missing_str = ", ".join(f"'{e}'" for e in all_query_entities[:3])
        abstention_msg = (
            f"NOT IDENTIFIED: The specific entity mentions ({missing_str}) could not be grounded in "
            f"the research knowledge graph. The system abstains from stating unverified assertions."
        )

    # Categorize into confidence bands
    if should_abstain:
        confidence_band = "REFUSAL"
    elif calibrated_score >= HIGH_CONF_THRESHOLD:
        confidence_band = "HIGH"
    elif calibrated_score >= MEDIUM_CONF_THRESHOLD:
        confidence_band = "MEDIUM"
    else:
        confidence_band = "LOW"

    return CalibrationResult(
        confidence_score=calibrated_score,
        confidence_band=confidence_band,
        should_abstain=should_abstain,
        abstention_message=abstention_msg,
        dense_support=round(dense_support, 3),
        symbolic_grounding=round(symbolic_grounding, 3),
        margin_score=round(margin_score, 3),
        contradiction_risk=round(contradiction_risk, 3),
        unidentified_entities=unidentified_entities,
        verified_entities=verified_entities,
        metrics_summary={
            "dense_weight": 0.38,
            "symbolic_weight": 0.35,
            "margin_weight": 0.27,
            "refusal_threshold": REFUSAL_THRESHOLD,
        }
    )
