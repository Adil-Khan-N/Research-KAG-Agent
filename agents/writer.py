"""
WriterAgent — synthesizes the final literature review.

Input:  paper_summaries, graph_relationships, timeline, topic
Output: literature_review (markdown document), citations
"""

import os
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash")

WRITER_PROMPT = """You are writing a structured literature review.

TOPIC: {topic}

PAPERS (chronological order):
{papers_text}

RELATIONSHIPS BETWEEN PAPERS:
{relationships_text}

Write a literature review with these exact sections:

## Introduction
Brief overview of the research area and why it matters (2-3 sentences).

## Chronological Evolution
Trace how the field developed over time. Reference specific papers 
using [Author et al., YEAR] format. Focus on how each paper built 
on previous work.

## Key Methods and Approaches
Group papers by their methodological approach. Discuss trade-offs 
between approaches.

## Benchmarks and Evaluation
What datasets and benchmarks are used? How do methods compare?

## Research Gaps and Future Directions
What problems remain unsolved based on these papers?

## References
List all papers cited.

IMPORTANT:
- Be specific and technical
- Cite papers by title when referencing them
- Show the progression of ideas across papers
- Keep total length to 600-800 words"""


def writer_agent(state: dict) -> dict:
    """
    Synthesize the final literature review from all agent outputs.
    Updates state with literature_review and citations.
    """
    topic = state.get("topic", "")
    paper_summaries = state.get("paper_summaries", [])
    graph_relationships = state.get("graph_relationships", [])
    timeline = state.get("timeline", [])

    if not paper_summaries:
        return {
            **state,
            "literature_review": "No papers found for this topic.",
            "citations": [],
        }

    print(f"\n[WriterAgent] Writing review for {len(paper_summaries)} papers...")

    # Format papers text
    papers_lines = []
    for i, p in enumerate(paper_summaries, 1):
        papers_lines.append(
            f"[{i}] {p['title']} ({p['year']})\n"
            f"    Summary: {p['summary'][:300]}"
        )
    papers_text = "\n\n".join(papers_lines)

    # Format relationships
    if graph_relationships:
        rel_lines = []
        for r in graph_relationships[:8]:
            rel_lines.append(
                f"- {r['paper1'][:40]} → {r['relationship']} → "
                f"{r['paper2'][:40]}: {r['description'][:80]}"
            )
        relationships_text = "\n".join(rel_lines)
    else:
        relationships_text = "No explicit relationships found."

    try:
        prompt = WRITER_PROMPT.format(
            topic=topic,
            papers_text=papers_text[:4000],
            relationships_text=relationships_text[:1000],
        )

        response = _model.generate_content(prompt)
        review = response.text.strip()

        # Build citations list
        citations = [
            {
                "number":   i,
                "arxiv_id": p["arxiv_id"],
                "title":    p["title"],
                "year":     p["year"],
            }
            for i, p in enumerate(paper_summaries, 1)
        ]

        print(f"\n[WriterAgent] Review generated "
              f"({len(review)} chars, {len(citations)} citations)")
        print(f"  Preview: {review[:200]}...")

        return {
            **state,
            "literature_review": review,
            "citations":         citations,
            "errors":            state.get("errors", []),
        }

    except Exception as e:
        logger.error(f"WriterAgent failed: {e}")
        # Fallback: assemble from summaries
        fallback = f"# Literature Review: {topic}\n\n"
        for p in paper_summaries:
            fallback += (
                f"## {p['title']} ({p['year']})\n\n"
                f"{p['summary']}\n\n"
            )
        return {
            **state,
            "literature_review": fallback,
            "citations":         [],
            "errors": state.get("errors", []) + [f"WriterAgent: {e}"],
        }