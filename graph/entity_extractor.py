"""
LLM-based entity extraction from paper abstracts and text.
Uses Google Gemini API (free tier) to extract Methods, Datasets,
Tasks, Concepts, and relationships between papers.
"""

import os
import json
import time
import logging
import re
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")  # fast + free tier

PROCESSED_DIR = Path("data/processed")


ENTITY_EXTRACTION_PROMPT = """You are a scientific paper entity extractor specializing in computer vision and deep learning research.

Given a paper's title, abstract, and introduction excerpt, extract all named entities.

Return ONLY valid JSON with NO additional text, explanation, or markdown formatting.
Do NOT include ```json or ``` in your response. Just raw JSON.

The JSON must follow this exact structure:
{{
  "methods": ["name1", "name2"],
  "datasets": ["name1", "name2"],
  "tasks": ["name1", "name2"],
  "concepts": ["name1", "name2"],
  "extends": ["short_name1"],
  "contradicts": []
}}

Guidelines:
- methods: named techniques, architectures, modules (e.g. "Multi-Head Attention", "Patch Embedding", "Shifted Window Attention", "Masked Autoencoder")
- datasets: benchmark datasets used or mentioned (e.g. "ImageNet", "COCO", "ADE20K", "Kinetics")
- tasks: ML tasks addressed (e.g. "image classification", "object detection", "semantic segmentation")
- concepts: key ideas and themes (e.g. "self-supervised learning", "transfer learning", "positional encoding", "hierarchical features")
- extends: short names of papers/models this paper directly builds upon (e.g. "ViT", "BERT", "ResNet", "Swin")
- contradicts: names of papers whose findings this paper challenges (usually empty)

Be specific and accurate. Only include entities clearly mentioned in the text.
Do not include generic terms like "neural network" or "deep learning" as methods.

Paper Title: {title}

Abstract: {abstract}

Introduction (first 800 chars): {intro}
"""


def extract_entities_from_paper(paper: dict) -> dict:
    """
    Extract entities from one paper using Gemini API.
    Returns structured entity dict.
    """
    arxiv_id = paper["arxiv_id"]
    title = paper["title"]
    abstract = paper.get("abstract", "")

    # Get intro text from chunks
    intro_text = ""
    for chunk in paper.get("chunks", []):
        if "introduction" in chunk.get("section", "").lower():
            intro_text = chunk["text"][:800]
            break
    if not intro_text:
        chunks = paper.get("chunks", [])
        if chunks:
            intro_text = chunks[0]["text"][:800]

    prompt = ENTITY_EXTRACTION_PROMPT.format(
        title=title,
        abstract=abstract[:1000],
        intro=intro_text,
    )

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown fences if Gemini adds them anyway
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        entities = json.loads(raw)

        result = {
            "arxiv_id": arxiv_id,
            "title": title,
            "methods": entities.get("methods", []),
            "datasets": entities.get("datasets", []),
            "tasks": entities.get("tasks", []),
            "concepts": entities.get("concepts", []),
            "extends": entities.get("extends", []),
            "contradicts": entities.get("contradicts", []),
        }

        logger.info(
            f"{arxiv_id}: extracted "
            f"{len(result['methods'])} methods, "
            f"{len(result['datasets'])} datasets, "
            f"{len(result['tasks'])} tasks, "
            f"{len(result['concepts'])} concepts"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"{arxiv_id}: JSON parse error: {e}\nRaw response: {raw[:300]}")
        return _empty_entities(arxiv_id, title)
    except Exception as e:
        logger.error(f"{arxiv_id}: Extraction failed: {e}")
        return _empty_entities(arxiv_id, title)


def _empty_entities(arxiv_id: str, title: str) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "methods": [],
        "datasets": [],
        "tasks": [],
        "concepts": [],
        "extends": [],
        "contradicts": [],
    }


def extract_all_papers(
    papers: list[dict],
    delay: float = 1.0,
    save_path: str = "data/entities.json",
) -> list[dict]:
    """
    Extract entities from all papers with rate limiting.
    Saves progress after each paper so crashes don't lose work.
    Free tier limit: 15 requests/minute — delay=1.0 keeps you safe.
    """
    # Load existing progress
    existing = {}
    if Path(save_path).exists():
        with open(save_path) as f:
            existing_list = json.load(f)
            existing = {e["arxiv_id"]: e for e in existing_list}
        logger.info(f"Loaded {len(existing)} existing extractions, skipping those")

    results = dict(existing)

    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]

        if arxiv_id in results:
            logger.info(f"[{i+1}/{len(papers)}] Skipping {arxiv_id} (already done)")
            continue

        logger.info(f"[{i+1}/{len(papers)}] Extracting: {paper['title'][:60]}...")
        entities = extract_entities_from_paper(paper)
        results[arxiv_id] = entities

        # Save after every paper
        with open(save_path, "w") as f:
            json.dump(list(results.values()), f, indent=2)

        # Gemini free tier: 15 RPM — 1 second delay is safe
        time.sleep(delay)

    print(f"\n✓ Extracted entities for {len(results)} papers → {save_path}")
    return list(results.values())


def load_entities(path: str = "data/entities.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    papers = []
    for path in sorted(PROCESSED_DIR.glob("*.json")):
        with open(path) as f:
            papers.append(json.load(f))

    print(f"Extracting entities from {len(papers)} papers using Gemini...")
    results = extract_all_papers(papers)

    # Summary
    total_methods = sum(len(r["methods"]) for r in results)
    total_datasets = sum(len(r["datasets"]) for r in results)
    total_tasks = sum(len(r["tasks"]) for r in results)
    total_concepts = sum(len(r["concepts"]) for r in results)

    print(f"\nExtraction Summary:")
    print(f"  Papers processed:   {len(results)}")
    print(f"  Methods extracted:  {total_methods}")
    print(f"  Datasets extracted: {total_datasets}")
    print(f"  Tasks extracted:    {total_tasks}")
    print(f"  Concepts extracted: {total_concepts}")