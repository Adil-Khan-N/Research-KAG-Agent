"""
Shared state passed between all agents in the pipeline.
LangGraph passes this state object from node to node.
"""

from typing import TypedDict, Optional


class LitReviewState(TypedDict):
    """
    State object shared across all agents.
    Each agent reads from and writes to this dict.
    """
    # Input
    topic: str                          # user's topic request

    # Planner output
    sub_queries: list[str]              # decomposed search queries
    plan_reasoning: str                 # why these queries were chosen

    # PaperSearch output
    retrieved_papers: list[dict]        # papers found via hybrid retriever
    retrieved_chunks: list[dict]        # raw chunks for each paper

    # Summary output
    paper_summaries: list[dict]         # {arxiv_id, title, year, summary, contribution}

    # Graph output
    graph_relationships: list[dict]     # {paper1, paper2, relationship, description}
    timeline: list[dict]                # [{year, arxiv_id, title, key_contribution}]

    # Writer output
    literature_review: str              # final markdown document
    citations: list[dict]               # papers cited in the review

    # Metadata
    total_papers_found: int
    errors: list[str]