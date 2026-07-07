"""
PlannerAgent — decomposes the user's topic into specific search queries.

Input:  topic (str)
Output: sub_queries (list of 4-6 search queries)
"""

import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash")

PLANNER_PROMPT = """You are a research planning agent.

Your task is to decompose a literature review topic into 5 specific 
search queries that will find the most relevant papers.

Each query should target a different aspect:
1. Core method/architecture papers
2. Foundational/background papers  
3. Evaluation and benchmarks
4. Extensions and improvements
5. Comparisons and analysis

Topic: {topic}

Return ONLY a JSON object with this exact structure, no other text:
{{
  "queries": [
    "query 1 here",
    "query 2 here", 
    "query 3 here",
    "query 4 here",
    "query 5 here"
  ],
  "reasoning": "one sentence explaining the decomposition strategy"
}}"""


def planner_agent(state: dict) -> dict:
    """
    Decompose the topic into specific search queries.
    Updates state with sub_queries and plan_reasoning.
    """
    topic = state["topic"]
    logger.info(f"PlannerAgent: planning for '{topic}'")
    print(f"\n[PlannerAgent] Topic: '{topic}'")

    try:
        import json
        import re

        prompt = PLANNER_PROMPT.format(topic=topic)
        response = _model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown fences
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()

        raw = raw.replace('\n', ' ').replace('\r', '')
        # Remove trailing commas before } or ]
        import re
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        data = json.loads(raw)
        queries = data.get("queries", [])
        reasoning = data.get("reasoning", "")

        print(f"[PlannerAgent] Generated {len(queries)} queries:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        print(f"[PlannerAgent] Reasoning: {reasoning}")

        return {
            **state,
            "sub_queries": queries,
            "plan_reasoning": reasoning,
            "errors": state.get("errors", []),
        }

    except Exception as e:
        logger.error(f"PlannerAgent failed: {e}")
        # Fallback: use the topic directly as a query
        fallback_queries = [
            topic,
            f"{topic} architecture",
            f"{topic} evaluation benchmarks",
            f"{topic} improvements extensions",
            f"{topic} comparison survey",
        ]
        return {
            **state,
            "sub_queries": fallback_queries,
            "plan_reasoning": f"Fallback due to error: {e}",
            "errors": state.get("errors", []) + [f"PlannerAgent: {e}"],
        }