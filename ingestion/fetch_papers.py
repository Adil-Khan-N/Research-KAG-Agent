"""
Fetch paper metadata from arXiv API and download PDFs.
"""

import arxiv
import os
import time
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VISION_TRANSFORMER_PAPERS = [
    "2010.11929",  # ViT
    "1706.03762",  # Attention Is All You Need
    "2012.12877",  # DeiT
    "2103.14030",  # Swin V1
    "2111.09883",  # Swin V2
    "2111.06377",  # MAE
    "2208.00173",  # BEiT V2
    "2106.08254",  # BEiT V1
    "2104.01136",  # PVT
    "2102.12122",  # TNT
    "2105.15082",  # CvT
    "2103.15808",  # CrossViT
    "2106.04560",  # LocalViT
    "2005.12872",  # DETR
    "2203.23743",  # DINO
    "2104.11227",  # Segmenter
    "2107.14795",  # CoAtNet
    "2103.10697",  # LeViT
    "2101.11605",  # T2T-ViT
    "2106.10270",  # How to train ViTs
    "2112.13492",  # MetaFormer
    "2110.02178",  # MobileViT
    "2108.01072",  # ViT-Adapter
    "2204.07118",  # DeiT III
    "2209.07399",  # Benchmarking ViTs
]

def fetch_metadata(arxiv_ids):
    """Fetch metadata for a list of arXiv IDs."""
    logger.info(f"Fetching metadata for {len(arxiv_ids)} papers...")

    client = arxiv.Client()
    search = arxiv.Search(id_list=arxiv_ids)

    papers = []
    for result in client.results(search):
        arxiv_id = result.entry_id.split("/")[-1]

        arxiv_id = arxiv_id.split("v")[0]  # Remove version number

        paper = {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "year": result.published.year,
            "abstract": result.summary.strip().replace("\n", " "),
            "categories": result.categories,
            "pdf_url": result.pdf_url,
            "pdf_path": str(RAW_DIR / f"{arxiv_id}.pdf"),
        }
        papers.append(paper)
        logger.info(f"Fetched metadata for paper: {arxiv_id} - {result.title}")

    logger.info(f"Fetched metadata for {len(papers)} papers.")
    return papers

def download_pdfs(papers: list[dict]) -> list[dict]:
    """Download PDFs for all papers using requests (arxiv 2.x compatible)."""
    import requests

    logger.info(f"Downloading PDFs for {len(papers)} papers...")

    for paper in tqdm(papers, desc="Download PDFs"):
        arxiv_id = paper["arxiv_id"]
        if not paper.get("pdf_path"):
            paper["pdf_path"] = str(RAW_DIR / f"{arxiv_id}.pdf")
        pdf_path = Path(paper["pdf_path"])

        if pdf_path.exists():
            logger.info(f"Already exists: {arxiv_id}")
            continue

        # Construct the PDF URL directly — no need to re-query arXiv API
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        try:
            response = requests.get(pdf_url, timeout=30, stream=True)
            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = pdf_path.stat().st_size / 1024
            logger.info(f"Downloaded: {arxiv_id} ({size_kb:.0f} KB)")

            # Be polite to arXiv servers
            time.sleep(3)

        except Exception as e:
            logger.error(f"Failed to download {arxiv_id}: {e}")
            paper["pdf_path"] = None

    return papers


def save_metadata(papers: list[dict], path: str = "data/papers_metadata.json"):
    """Save metadata to a JSON file."""
    with open(path, "w") as f:
        json.dump(papers, f, indent=4)
    logger.info(f"Saved metadata for {len(papers)} papers to {path}")

def load_metadata(path: str = "data/papers_metadata.json") -> list[dict]:
    """Load metadata from a JSON file."""
    if not os.path.exists(path):
        logger.warning(f"Metadata file {path} does not exist.")
        return []

    with open(path, "r") as f:
        papers = json.load(f)
    logger.info(f"Loaded metadata for {len(papers)} papers from {path}")
    return papers

if __name__ == "__main__":
    papers = fetch_metadata(VISION_TRANSFORMER_PAPERS)
    save_metadata(papers)
    papers = download_pdfs(papers)
    save_metadata(papers)  # Save again to update PDF paths
    print(f"\n Done. Metadata saved to data/papers_metadata.json and PDFs downloaded to data/raw/")