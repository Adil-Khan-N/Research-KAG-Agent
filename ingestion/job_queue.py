"""
Lightweight background job queue using Python threads.
Same pattern as Celery but requires no Redis or extra services.

For production: swap _ThreadJobQueue for CeleryJobQueue
by changing QUEUE_BACKEND in .env

Job lifecycle:
  queued → processing → complete | failed

CV talking point:
"Implemented non-blocking API design with background job queue,
returning job IDs immediately while processing runs async"
"""

import uuid
import time
import logging
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Any

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    COMPLETE   = "complete"
    FAILED     = "failed"


@dataclass
class Job:
    """Represents one background ingestion job."""
    job_id:      str
    status:      JobStatus
    pdf_path:    str
    filename:    str
    created_at:  str
    started_at:  str  = ""
    finished_at: str  = ""
    progress:    int  = 0      # 0-100
    message:     str  = "Queued"
    result:      dict = field(default_factory=dict)
    error:       str  = ""


class JobQueue:
    """
    Thread-based job queue.
    Stores jobs in memory (survives process lifetime).
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, pdf_path: str, filename: str) -> str:
        """Create a new job and return its ID."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = Job(
            job_id    = job_id,
            status    = JobStatus.QUEUED,
            pdf_path  = pdf_path,
            filename  = filename,
            created_at = datetime.now().isoformat(),
        )
        with self._lock:
            self._jobs[job_id] = job
        logger.info(f"Job created: {job_id} for {filename}")
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus = None,
        progress: int = None,
        message: str = None,
        result: dict = None,
        error: str = None,
    ):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if status:
                job.status = status
                if status == JobStatus.PROCESSING:
                    job.started_at = datetime.now().isoformat()
                elif status in (JobStatus.COMPLETE, JobStatus.FAILED):
                    job.finished_at = datetime.now().isoformat()
            if progress is not None:
                job.progress = progress
            if message:
                job.message = message
            if result:
                job.result = result
            if error:
                job.error = error

    def get_all_jobs(self) -> list[dict]:
        """Get all jobs as dicts for the /jobs endpoint."""
        with self._lock:
            return [
                {
                    "job_id":      j.job_id,
                    "status":      j.status,
                    "filename":    j.filename,
                    "progress":    j.progress,
                    "message":     j.message,
                    "created_at":  j.created_at,
                    "finished_at": j.finished_at,
                    "result":      j.result,
                    "error":       j.error[:100] if j.error else "",
                }
                for j in self._jobs.values()
            ]


# ── Global singleton queue ────────────────────────────────────
_queue = JobQueue()


def get_queue() -> JobQueue:
    return _queue


# ── Job runner ────────────────────────────────────────────────

def run_ingestion_job(job_id: str, pdf_path: str, filename: str):
    """
    Run full ingestion pipeline in a background thread.
    Updates job status at each stage.

    Stages (with progress %):
      0  → Queued
      10 → Text extraction
      30 → Chunking
      50 → Embedding + Postgres insert
      75 → Entity extraction (LLM)
      90 → Neo4j ingestion
      100 → Complete
    """
    queue = get_queue()

    try:
        queue.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            progress=5,
            message="Starting ingestion pipeline",
        )

        # ── Stage 1: Text extraction ──────────────────────────
        queue.update_job(
            job_id, progress=10,
            message="Extracting text from PDF",
        )

        from ingestion.extract_text import extract_text_from_pdf
        extraction = extract_text_from_pdf(pdf_path)

        arxiv_id = filename.replace(".pdf", "")
        paper_data = {
            "arxiv_id":       arxiv_id,
            "title":          arxiv_id,
            "authors":        [],
            "year":           datetime.now().year,
            "abstract":       extraction["sections"].get("abstract", "")[:500],
            "categories":     [],
            "pdf_url":        "",
            "page_count":     extraction["page_count"],
            "sections_found": list(extraction["sections"].keys()),
            "full_text":      extraction["full_text"],
            "sections":       extraction["sections"],
        }

        logger.info(
            f"Job {job_id}: extracted {extraction['page_count']} pages, "
            f"{len(extraction['sections'])} sections"
        )

        # ── Stage 2: Chunking ─────────────────────────────────
        queue.update_job(
            job_id, progress=30,
            message="Chunking text into segments",
        )

        from ingestion.chunker import chunk_paper
        chunks = chunk_paper(paper_data)
        paper_data["chunks"] = chunks
        paper_data["total_chunks"] = len(chunks)

        logger.info(f"Job {job_id}: {len(chunks)} chunks created")

        # ── Stage 3: Postgres insert + embedding ──────────────
        queue.update_job(
            job_id, progress=50,
            message=f"Embedding {len(chunks)} chunks",
        )

        from ingestion.db import engine
        from ingestion.embedder import (
            load_model, insert_paper, insert_chunks_batch,
        )
        from sqlalchemy import text
        import numpy as np

        model = load_model()

        # Insert paper
        with engine.connect() as conn:
            # Check if paper already exists
            existing = conn.execute(text("""
                SELECT 1 FROM papers WHERE arxiv_id = :arxiv_id
            """), {"arxiv_id": arxiv_id}).fetchone()

            if not existing:
                insert_paper(conn, paper_data)
                conn.commit()

        # Embed and insert chunks
        BATCH = 32
        texts = [c["text"] for c in chunks]
        all_embeddings = []

        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            embs = model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.append(embs)

            progress = 50 + int((i / len(texts)) * 25)
            queue.update_job(
                job_id, progress=progress,
                message=f"Embedding chunk {i+len(batch)}/{len(texts)}",
            )

        all_embeddings = np.vstack(all_embeddings)

        with engine.connect() as conn:
            insert_chunks_batch(conn, chunks, all_embeddings)
            conn.commit()

        logger.info(f"Job {job_id}: embeddings inserted")

        # ── Stage 4: Entity extraction ────────────────────────
        queue.update_job(
            job_id, progress=75,
            message="Extracting entities with LLM",
        )

        from graph.entity_extractor import extract_entities_from_paper
        entities = extract_entities_from_paper(paper_data)

        logger.info(
            f"Job {job_id}: extracted "
            f"{len(entities.get('methods', []))} methods, "
            f"{len(entities.get('datasets', []))} datasets"
        )

        # ── Stage 5: Neo4j ingestion ──────────────────────────
        queue.update_job(
            job_id, progress=90,
            message="Updating knowledge graph",
        )

        from graph.neo4j_client import get_neo4j_client
        from graph.graph_ingestion import (
            ingest_paper_node,
            ingest_entities_and_relationships,
        )

        neo4j = get_neo4j_client()
        ingest_paper_node(neo4j, paper_data)
        ingest_entities_and_relationships(neo4j, entities, paper_data)

        # ── Complete ──────────────────────────────────────────
        result = {
            "arxiv_id":     arxiv_id,
            "title":        paper_data["title"],
            "chunks":       len(chunks),
            "pages":        extraction["page_count"],
            "sections":     list(extraction["sections"].keys()),
            "methods":      len(entities.get("methods", [])),
            "datasets":     len(entities.get("datasets", [])),
            "concepts":     len(entities.get("concepts", [])),
        }

        queue.update_job(
            job_id,
            status   = JobStatus.COMPLETE,
            progress = 100,
            message  = f"Complete — {len(chunks)} chunks ingested",
            result   = result,
        )
        logger.info(f"Job {job_id}: COMPLETE")

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Job {job_id} FAILED: {e}\n{tb}")
        queue.update_job(
            job_id,
            status  = JobStatus.FAILED,
            message = f"Failed: {error_msg[:100]}",
            error   = tb[:500],
        )


def submit_job(pdf_path: str, filename: str) -> str:
    """
    Submit an ingestion job to run in background.
    Returns job_id immediately — non-blocking.
    """
    queue = get_queue()
    job_id = queue.create_job(pdf_path, filename)

    thread = threading.Thread(
        target=run_ingestion_job,
        args=(job_id, pdf_path, filename),
        daemon=True,
        name=f"ingest-{job_id}",
    )
    thread.start()

    logger.info(
        f"Submitted job {job_id} for {filename} "
        f"(thread: {thread.name})"
    )
    return job_id