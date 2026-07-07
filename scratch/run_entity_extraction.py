"""Extract entities from all processed papers using Gemini API."""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Verify API key is set
import os
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set in .env file")
    print("Get your free key at: https://aistudio.google.com")
    exit(1)

from graph.entity_extractor import extract_all_papers

# Load all processed papers
papers = []
for path in sorted(Path("data/processed").glob("*.json")):
    with open(path) as f:
        papers.append(json.load(f))

print(f"Found {len(papers)} papers.")
print(f"Using Gemini 1.5 Flash (free tier: 15 requests/minute)")
print(f"Estimated time: ~{len(papers) * 1.5 / 60:.1f} minutes\n")

results = extract_all_papers(papers, delay=1.0)
print(f"\nDone. Entities saved to data/entities.json")