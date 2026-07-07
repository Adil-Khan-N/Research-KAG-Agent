"""
Full ingestion pipeline:
fetch metadata → download PDFs → extract text → chunk → save JSON
"""

import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

from ingestion.fetch_papers import (
    fetch_metadata,
    download_pdfs,
    save_metadata,
    load_metadata,
    VISION_TRANSFORMER_PAPERS,
)
from ingestion.extract_text import extract_text_from_pdf
from ingestion.chunker import chunk_paper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def process_paper(paper_meta: dict) -> dict | None:
    """Run extraction + chunking for one paper. Returns full paper record."""
    arxiv_id = paper_meta["arxiv_id"]
    pdf_path = paper_meta.get("pdf_path")
    
    if not pdf_path or not Path(pdf_path).exists():
        logger.warning(f"Skipping {arxiv_id}: PDF not found at {pdf_path}")
        return None
    
    try:
        # Step 1: Extract text + sections from PDF
        extraction = extract_text_from_pdf(pdf_path)
        
        # Step 2: Build combined paper dict for chunking
        paper_data = {
            **paper_meta,
            "full_text": extraction["full_text"],
            "sections": extraction["sections"],
            "page_count": extraction["page_count"],
        }
        
        # Step 3: Chunk
        chunks = chunk_paper(paper_data)
        
        # Step 4: Assemble final record
        record = {
            "arxiv_id": arxiv_id,
            "title": paper_meta["title"],
            "authors": paper_meta["authors"],
            "year": paper_meta["year"],
            "abstract": paper_meta["abstract"],
            "categories": paper_meta.get("categories", []),
            "pdf_url": paper_meta.get("pdf_url", ""),
            "page_count": extraction["page_count"],
            "sections_found": list(extraction["sections"].keys()),
            "total_chunks": len(chunks),
            "chunks": chunks,
        }
        
        logger.info(
            f"{arxiv_id}: {len(chunks)} chunks across "
            f"{len(extraction['sections'])} sections "
            f"({extraction['page_count']} pages)"
        )
        return record
    
    except Exception as e:
        logger.error(f"Failed to process {arxiv_id}: {e}")
        return None


def save_paper_record(record: dict):
    """Save one paper's full record to data/processed/<arxiv_id>.json"""
    out_path = PROCESSED_DIR / f"{record['arxiv_id']}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)


def load_paper_record(arxiv_id: str) -> dict | None:
    """Load a previously processed paper record."""
    path = PROCESSED_DIR / f"{arxiv_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def run_pipeline(
    arxiv_ids: list[str] = None,
    skip_existing: bool = True,
    skip_download: bool = False,
):
    """
    Run the full ingestion pipeline.
    
    Args:
        arxiv_ids: list of arXiv IDs (defaults to VISION_TRANSFORMER_PAPERS)
        skip_existing: skip papers already in data/processed/
        skip_download: skip PDF download (use if PDFs already downloaded)
    """
    arxiv_ids = arxiv_ids or VISION_TRANSFORMER_PAPERS
    
    # Step 1: Fetch metadata
    metadata_path = "data/papers_metadata.json"
    if Path(metadata_path).exists():
        logger.info("Loading existing metadata...")
        papers = load_metadata(metadata_path)
    else:
        logger.info("Fetching metadata from arXiv API...")
        papers = fetch_metadata(arxiv_ids)
        save_metadata(papers, metadata_path)
    
    # Step 2: Download PDFs
    if not skip_download:
        papers = download_pdfs(papers)
        save_metadata(papers, metadata_path)
    
    # Step 3: Process each paper
    results = []
    skipped = 0
    failed = 0
    
    for paper in tqdm(papers, desc="Processing papers"):
        arxiv_id = paper["arxiv_id"]
        
        # Skip if already processed
        if skip_existing and (PROCESSED_DIR / f"{arxiv_id}.json").exists():
            logger.info(f"Skipping {arxiv_id} (already processed)")
            skipped += 1
            existing = load_paper_record(arxiv_id)
            if existing:
                results.append(existing)
            continue
        
        record = process_paper(paper)
        if record:
            save_paper_record(record)
            results.append(record)
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"INGESTION COMPLETE")
    print(f"="*60)
    print(f"  Total papers:     {len(papers)}")
    print(f"  Processed:        {len(results)}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed:           {failed}")
    
    total_chunks = sum(r["total_chunks"] for r in results)
    avg_chunks = total_chunks / max(len(results), 1)
    print(f"  Total chunks:     {total_chunks}")
    print(f"  Avg chunks/paper: {avg_chunks:.1f}")
    print(f"  Output:           data/processed/")
    print("="*60)
    
    return results


if __name__ == "__main__":
    run_pipeline()