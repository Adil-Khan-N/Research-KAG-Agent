"""
SummaryAgent — summarizes each paper's contribution.

Input:  retrieved_papers, retrieved_chunks
Output: paper_summaries
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

SUMMARY_PROMPT = """You are summarizing a research paper for a literature review.

Paper: {title} ({year})

Relevant excerpts:
{chunks_text}

Write a 2-3 sentence summary covering:
1. What problem this paper solves
2. The key method or contribution
3. Main results or impact

Be specific and technical. Do not use vague phrases like "the paper presents".
Respond with ONLY the summary text, no labels or headers."""


def summary_agent(state: dict) -> dict:
    """
    Generate a concise summary for each retrieved paper.
    Updates state with paper_summaries.
    """
    retrieved_papers = state.get("retrieved_papers", [])
    retrieved_chunks = state.get("retrieved_chunks", [])

    if not retrieved_papers:
        return {**state, "paper_summaries": []}

    print(f"\n[SummaryAgent] Summarizing {len(retrieved_papers)} papers...")

    # Build chunk lookup by arxiv_id
    chunks_by_paper = {}
    for chunk in retrieved_chunks:
        arxiv_id = chunk.get("arxiv_id", "")
        if not arxiv_id:
            chunk_id = chunk.get("chunk_id", "")
            arxiv_id = chunk_id.split("_chunk_")[0] if "_chunk_" in chunk_id else ""
        if arxiv_id:
            if arxiv_id not in chunks_by_paper:
                chunks_by_paper[arxiv_id] = []
            chunks_by_paper[arxiv_id].append(chunk["text"])

    paper_summaries = []

    for i, paper in enumerate(retrieved_papers):
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        year = paper.get("year", "")

        # Get chunks for this paper
        chunks = chunks_by_paper.get(arxiv_id, [])
        if not chunks:
            paper_summaries.append({
                "arxiv_id":    arxiv_id,
                "title":       title,
                "year":        year,
                "summary":     "No content available for this paper.",
                "contribution": "",
            })
            continue

        chunks_text = "\n\n".join(
            f"[{j+1}] {c[:300]}"
            for j, c in enumerate(chunks[:3])
        )

        print(f"  [{i+1}/{len(retrieved_papers)}] {title[:55]}...")

        try:
            prompt = SUMMARY_PROMPT.format(
                title=title,
                year=year,
                chunks_text=chunks_text,
            )
            response = _model.generate_content(prompt)
            summary = response.text.strip()

            paper_summaries.append({
                "arxiv_id":    arxiv_id,
                "title":       title,
                "year":        year,
                "summary":     summary,
                "contribution": summary.split(".")[0] + ".",
            })

            print(f"     {summary[:100]}...")

        except Exception as e:
            logger.error(f"Summary failed for {arxiv_id}: {e}")
            paper_summaries.append({
                "arxiv_id":    arxiv_id,
                "title":       title,
                "year":        year,
                "summary":     f"Summary unavailable: {e}",
                "contribution": "",
            })

        # Rate limiting
        if i < len(retrieved_papers) - 1:
            time.sleep(3)

    print(f"\n[SummaryAgent] Generated {len(paper_summaries)} summaries")

    return {
        **state,
        "paper_summaries": paper_summaries,
        "errors": state.get("errors", []),
    }