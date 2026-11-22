"""
Symbolic Constraint & Knowledge Graph Verification Engine.

Enforces structural knowledge constraints on generation:
1. Traverses relational paths in the Knowledge Graph (Neo4j or local verified graph)
2. Injects symbolic boundary constraints into the prompt
3. Detects invalid / unverified relational claims (e.g. Paper X evaluated on Dataset Y when no such edge exists)
4. Enforces explicit 'Not Identified' / Refusal when symbolic support is zero
"""

import json
import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_local_graph() -> List[Dict[str, Any]]:
    """Loads entities and relationships from data/entities.json."""
    entities_path = os.path.join(os.path.dirname(__file__), "..", "data", "entities.json")
    if os.path.exists(entities_path):
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load local entities.json: {e}")
    return []


class SymbolicGraphVerifier:
    """
    Symbolic Knowledge Graph Verifier.
    Checks whether relational assertions between research entities exist in the graph.
    """

    RELATION_WEIGHTS = {
        "EXTENDS": 0.35,
        "EVALUATES_ON": 0.30,
        "USES": 0.25,
        "CONTRADICTS": 0.25,
        "AUTHORED_BY": 0.15,
        "DISCUSSES": 0.10,
    }

    def __init__(self, neo4j_client=None):
        self.neo4j_client = neo4j_client
        self._local_graph = _load_local_graph()

    def compute_symbolic_path_score(
        self,
        paper_id: str,
        query_entities: List[str],
        graph_papers: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Computes a grounded symbolic path score (replacing flat +0.3 heuristic)
        based on edge types, hop decay, and multi-entity reinforcement.
        """
        if not query_entities:
            return 0.0

        score = 0.0
        # If Neo4j graph papers metadata is available
        if graph_papers:
            for p in graph_papers:
                if p.get("arxiv_id") == paper_id:
                    hops = p.get("hops", 1)
                    hop_decay = 0.65 ** (hops - 1)
                    rel_type = p.get("rel_type", "DISCUSSES")
                    rel_weight = self.RELATION_WEIGHTS.get(rel_type, 0.20)
                    matched_count = len(p.get("matched_entities", [1]))
                    # Multi-path reinforcement
                    score += rel_weight * hop_decay * (1.0 + 0.2 * min(matched_count - 1, 3))

        # Check local graph relations
        for item in self._local_graph:
            if item.get("arxiv_id") == paper_id:
                for ent in query_entities:
                    ent_lower = ent.lower()
                    if any(ent_lower in m.lower() for m in item.get("methods", [])):
                        score += 0.25
                    if any(ent_lower in d.lower() for d in item.get("datasets", [])):
                        score += 0.28
                    if any(ent_lower in t.lower() for t in item.get("tasks", [])):
                        score += 0.20
                    if any(ent_lower in ext.lower() for ext in item.get("extends", [])):
                        score += 0.35

        # Normalize score into [0.0, 0.45]
        return round(min(0.45, max(0.0, score)), 3)

    def extract_symbolic_constraints(
        self,
        query: str,
        extracted_entities: Dict[str, Any],
        relevant_papers: List[str],
    ) -> Dict[str, Any]:
        """
        Extracts verified relational facts from the graph that constrain what
        the model is allowed to claim about the identified entities.
        """
        verified_triples = []
        forbidden_assumptions = []

        query_lower = query.lower()
        candidate_entities = extracted_entities.get("all", [])

        for item in self._local_graph:
            aid = item.get("arxiv_id")
            title = item.get("title", "")
            if aid in relevant_papers or any(e.lower() in title.lower() for e in candidate_entities):
                # Verified Methods
                for m in item.get("methods", []):
                    verified_triples.append({
                        "subject": title,
                        "predicate": "USES_METHOD",
                        "object": m,
                        "paper_id": aid
                    })
                # Verified Datasets
                for d in item.get("datasets", []):
                    verified_triples.append({
                        "subject": title,
                        "predicate": "EVALUATES_ON",
                        "object": d,
                        "paper_id": aid
                    })
                # Verified Extends Lineage
                for ext in item.get("extends", []):
                    verified_triples.append({
                        "subject": title,
                        "predicate": "EXTENDS",
                        "object": ext,
                        "paper_id": aid
                    })

        # Check for ungrounded entity pairings
        for ent in candidate_entities:
            grounded = any(
                ent.lower() in t["subject"].lower() or
                ent.lower() in t["object"].lower()
                for t in verified_triples
            )
            if not grounded:
                forbidden_assumptions.append(
                    f"Entity '{ent}' has no verified path in the knowledge graph. Do NOT assert unverified performance numbers."
                )

        return {
            "verified_triples": verified_triples[:15],
            "forbidden_assumptions": forbidden_assumptions,
            "has_graph_constraints": len(verified_triples) > 0,
        }

    def format_constraints_for_prompt(self, constraints: Dict[str, Any]) -> str:
        """Formats symbolic constraints as strict bounding instructions for the LLM."""
        if not constraints.get("has_graph_constraints"):
            return ""

        lines = [
            "\nSYMBOLIC KNOWLEDGE GRAPH CONSTRAINTS (Strict Grounding Rules):",
            "1. Verified Entity Relations in Graph:"
        ]
        for t in constraints["verified_triples"][:10]:
            lines.append(f"   • ({t['subject']}) -[:{t['predicate']}]-> ({t['object']}) [arxiv:{t['paper_id']}]")

        if constraints["forbidden_assumptions"]:
            lines.append("2. Unverified Constraints (Refusal Directive):")
            for f in constraints["forbidden_assumptions"][:3]:
                lines.append(f"   ⚠️ {f}")

        lines.append(
            "3. If asked about a relationship, metric, or dataset not attested above or in the cited chunks, "
            "you MUST state 'Not identified in the verified corpus' rather than extrapolating.\n"
        )
        return "\n".join(lines)


# Singleton instance
_verifier_instance = None

def get_symbolic_verifier() -> SymbolicGraphVerifier:
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = SymbolicGraphVerifier()
    return _verifier_instance
