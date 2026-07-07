"""
FastAPI route handlers for all endpoints.
"""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse

from api.models import (
    QueryRequest, QueryResponse, CitationModel, ChunkModel,
    IngestResponse,
    GraphExploreRequest, GraphExploreResponse, GraphNode, GraphEdge,
    HealthResponse,
)
from retrieval.pipeline import run_pipeline, log_pipeline_result
from graph.neo4j_client import get_neo4j_client
from graph.graph_queries import get_paper_neighborhood, expand_neighbors

logger = logging.getLogger(__name__)
router = APIRouter()

# ── POST /query ───────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Run the full hybrid KAG pipeline on a question.
    Checks semantic cache first — returns instantly on hit.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    from ingestion.db import engine as pg_engine
    from retrieval.semantic_cache import (
        cache_lookup, cache_store, _ensure_cache_table
    )
    from retrieval.query_decomposer import (
        is_multi_part_query, run_decomposed_pipeline
    )

    # Ensure cache table exists
    try:
        _ensure_cache_table(pg_engine)
    except Exception:
        pass

    # Step 1: Check semantic cache
    cache_result = cache_lookup(request.query, engine=pg_engine)

    if cache_result.hit:
        logger.info(
            f"Cache hit: similarity={cache_result.similarity:.4f}, "
            f"latency={cache_result.latency_ms}ms"
        )
        return QueryResponse(
            query=request.query,
            answer=cache_result.answer,
            confidence="high",
            not_found=False,
            citations=[],
            chunks_used=[],
            entity_count=0,
            graph_papers_found=0,
            graph_boosted_count=0,
            total_latency_ms=cache_result.latency_ms,
            pipeline_variant="cache_hit",
        )

    # Step 2: Check if multi-part query
    if is_multi_part_query(request.query) and request.use_graph:
        logger.info(f"Multi-part query detected: '{request.query[:50]}'")
        try:
            decomp_result = run_decomposed_pipeline(
                request.query, engine=pg_engine
            )

            # Cache the result
            cache_store(
                query=request.query,
                answer=decomp_result["answer"],
                citations=decomp_result["all_citations"],
                pipeline_variant="decomposed",
                engine=pg_engine,
            )

            return QueryResponse(
                query=request.query,
                answer=decomp_result["answer"],
                confidence="high",
                not_found=False,
                citations=[],
                chunks_used=[],
                entity_count=0,
                graph_papers_found=0,
                graph_boosted_count=0,
                total_latency_ms=decomp_result["latency_ms"],
                pipeline_variant="decomposed",
            )
        except Exception as e:
            logger.error(f"Decomposed pipeline failed: {e}, falling back")

    # Step 3: Standard pipeline
    try:
        result = run_pipeline(
            query=request.query,
            top_k_retrieve=request.top_k_retrieve,
            top_k_rerank=request.top_k_rerank,
            use_graph=request.use_graph,
            pipeline_variant=request.pipeline_variant,
        )

        # Log to query_logs
        try:
            log_pipeline_result(result)
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")

        # Store in cache
        try:
            cache_store(
                query=request.query,
                answer=result.answer.answer,
                citations=result.answer.citations,
                pipeline_variant=request.pipeline_variant,
                engine=pg_engine,
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

        # Build response
        citations = [CitationModel(**c) for c in result.answer.citations]
        chunks_used = [
            ChunkModel(
                chunk_id=r.chunk_id,
                arxiv_id=r.arxiv_id,
                title=r.title,
                year=r.year,
                section=r.section,
                text=r.text[:300],
                rerank_score=r.rerank_score,
                rerank_rank=r.rerank_rank,
                source=r.source,
                retrieval_path=r.retrieval_path,
            )
            for r in result.ranked_chunks
        ]

        return QueryResponse(
            query=result.query,
            answer=result.answer.answer,
            confidence=result.answer.confidence,
            not_found=result.answer.not_found,
            citations=citations,
            chunks_used=chunks_used,
            entity_count=result.trace.entity_count,
            graph_papers_found=len(result.trace.graph_papers_found),
            graph_boosted_count=result.trace.graph_boosted_count,
            total_latency_ms=result.total_latency_ms,
            pipeline_variant=result.pipeline_variant,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 

@router.post("/query-enhanced")
async def query_enhanced_endpoint(request: QueryRequest):
    """
    Enhanced query endpoint with contradiction detection
    and retrieval explanations for every chunk.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    from retrieval.contradiction_detector import (
        detect_contradictions,
        format_contradiction_for_display,
        write_contradicts_to_neo4j,
    )
    from retrieval.explanation_layer import (
        build_all_explanations,
        format_explanations_for_api,
    )
    from retrieval.hybrid_retriever import HybridKAGRetriever
    from retrieval.reranker import rerank
    from retrieval.generator import generate_answer
    import time

    start = time.time()

    try:
        # Stage 1: Retrieve
        retriever = HybridKAGRetriever()
        hybrid_results, trace = retriever.retrieve(
            request.query,
            top_k=request.top_k_retrieve,
            use_graph=request.use_graph,
        )

        # Stage 2: Rerank
        ranked_chunks = rerank(
            request.query,
            hybrid_results,
            top_k=request.top_k_rerank,
        )

        # Stage 3: Build explanations
        explanations = build_all_explanations(
            ranked_chunks=ranked_chunks,
            original_results=hybrid_results,
        )
        explanations_data = format_explanations_for_api(explanations)

        # Stage 4: Detect contradictions
        contradictions = detect_contradictions(
            chunks=ranked_chunks,
            use_llm=True,
            max_pairs=3,
            delay=3.0,
        )

        # Write high-confidence contradictions to Neo4j
        for c in contradictions:
            if c.confidence in ("high", "medium"):
                write_contradicts_to_neo4j(c)

        contradiction_text = format_contradiction_for_display(contradictions)

        # Stage 5: Generate answer
        answer_obj = generate_answer(
            request.query, ranked_chunks,
            max_chunks=request.top_k_rerank
        )

        # Append contradiction warning to answer if found
        full_answer = answer_obj.answer
        if contradiction_text:
            full_answer += contradiction_text

        # Log
        try:
            log_pipeline_result(
                run_pipeline(
                    request.query,
                    use_graph=request.use_graph,
                )
            )
        except Exception:
            pass

        elapsed = int((time.time() - start) * 1000)

        # Build citation models
        citations = [
            CitationModel(**c) for c in answer_obj.citations
        ]

        chunks_used = [
            ChunkModel(
                chunk_id=r.chunk_id,
                arxiv_id=r.arxiv_id,
                title=r.title,
                year=r.year,
                section=r.section,
                text=r.text[:300],
                rerank_score=r.rerank_score,
                rerank_rank=r.rerank_rank,
                source=r.source,
                retrieval_path=r.retrieval_path,
            )
            for r in ranked_chunks
        ]

        contradiction_models = [
            {
                "paper_a":    c.paper_a,
                "paper_b":    c.paper_b,
                "arxiv_id_a": c.arxiv_id_a,
                "arxiv_id_b": c.arxiv_id_b,
                "claim_a":    c.claim_a,
                "claim_b":    c.claim_b,
                "topic":      c.topic,
                "confidence": c.confidence,
                "method":     c.method,
            }
            for c in contradictions if c.contradicts
        ]

        explanation_models = [
            {
                "chunk_id":    e["chunk_id"],
                "title":       e["title"],
                "year":        e["year"],
                "section":     e["section"],
                "scores":      e["scores"],
                "retrieval":   e["retrieval"],
                "explanation": e["explanation"],
            }
            for e in explanations_data
        ]

        return {
            "query":               request.query,
            "answer":              full_answer,
            "confidence":          answer_obj.confidence,
            "not_found":           answer_obj.not_found,
            "citations":           [c.dict() for c in citations],
            "chunks_used":         [c.dict() for c in chunks_used],
            "explanations":        explanation_models,
            "contradictions":      contradiction_models,
            "contradiction_count": len(contradiction_models),
            "entity_count":        trace.entity_count,
            "graph_papers_found":  len(trace.graph_papers_found),
            "graph_boosted_count": trace.graph_boosted_count,
            "total_latency_ms":    elapsed,
            "pipeline_variant":    "enhanced",
        }

    except Exception as e:
        logger.error(f"Enhanced query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ── POST /ingest ──────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Ingest a new PDF paper into the system.

    Runs the full ingestion pipeline:
    PDF → text extraction → chunking → embedding → Neo4j entity extraction
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    start = time.time()

    # Save uploaded file to data/raw/
    upload_dir = Path("data/raw")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded PDF: {file_path}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {e}"
        )

    # Run ingestion pipeline
    try:
        from ingestion.extract_text import extract_text_from_pdf
        from ingestion.chunker import chunk_paper
        from ingestion.embedder import embed_and_ingest_paper, load_model
        from ingestion.embedder import insert_paper
        from graph.entity_extractor import extract_entities_from_paper
        from graph.graph_ingestion import (
            ingest_paper_node,
            ingest_entities_and_relationships,
        )
        from ingestion.db import engine
        from graph.neo4j_client import get_neo4j_client

        # Step 1: Extract text
        extraction = extract_text_from_pdf(str(file_path))

        # Build paper record
        arxiv_id = file.filename.replace(".pdf", "")
        paper_data = {
            "arxiv_id": arxiv_id,
            "title": arxiv_id,  # Will be updated if metadata found
            "authors": [],
            "year": 2024,
            "abstract": extraction["sections"].get("abstract", "")[:500],
            "categories": [],
            "pdf_url": "",
            "page_count": extraction["page_count"],
            "sections_found": list(extraction["sections"].keys()),
            "full_text": extraction["full_text"],
            "sections": extraction["sections"],
        }

        # Step 2: Chunk
        chunks = chunk_paper(paper_data)
        paper_data["chunks"] = chunks
        paper_data["total_chunks"] = len(chunks)

        # Step 3: Embed and insert into Postgres
        model = load_model()
        with engine.connect() as conn:
            insert_paper(conn, paper_data)
            conn.commit()

        stats = embed_and_ingest_paper(
            paper=paper_data,
            model=model,
            engine=engine,
            skip_existing=False,
        )

        # Step 4: Extract entities and insert into Neo4j
        neo4j = get_neo4j_client()
        entities = extract_entities_from_paper(paper_data)
        ingest_paper_node(neo4j, paper_data)
        ingest_entities_and_relationships(neo4j, entities, paper_data)

        elapsed_ms = int((time.time() - start) * 1000)

        return IngestResponse(
            arxiv_id=arxiv_id,
            title=paper_data["title"],
            chunks_created=len(chunks),
            sections_found=list(extraction["sections"].keys()),
            entities_extracted={
                "methods": len(entities.get("methods", [])),
                "datasets": len(entities.get("datasets", [])),
                "concepts": len(entities.get("concepts", [])),
                "tasks": len(entities.get("tasks", [])),
            },
            graph_nodes_created=sum([
                len(entities.get("methods", [])),
                len(entities.get("datasets", [])),
                len(entities.get("concepts", [])),
            ]),
            latency_ms=elapsed_ms,
            status="success",
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")



# ── GET /graph-explore ────────────────────────────────────────

@router.get("/graph-explore", response_model=GraphExploreResponse)
async def graph_explore_endpoint(
    entity: str,
    entity_type: str = "paper",
    hops: int = 1,
):
    """
    Explore the graph neighborhood of an entity.

    For papers: returns methods, datasets, concepts, EXTENDS chains.
    For methods/datasets: returns all papers connected to that entity.

    Example:
        GET /graph-explore?entity=2103.14030&entity_type=paper
        GET /graph-explore?entity=ImageNet&entity_type=dataset
    """
    if not entity.strip():
        raise HTTPException(status_code=400, detail="Entity cannot be empty")

    try:
        neo4j = get_neo4j_client()
        nodes = []
        edges = []

        if entity_type == "paper":
            # Get full paper neighborhood
            neighborhood = get_paper_neighborhood(entity, client=neo4j)

            # Add the central paper node
            paper_info = neo4j.run(
                "MATCH (p:Paper {arxiv_id: $id}) RETURN p",
                {"id": entity}
            )
            if not paper_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Paper not found: {entity}"
                )

            paper_props = dict(paper_info[0]["p"])
            nodes.append(GraphNode(
                id=entity,
                label=paper_props.get("title", entity)[:50],
                type="paper",
                properties=paper_props,
            ))

            # Add connected nodes and edges
            for method in neighborhood["methods"]:
                nodes.append(GraphNode(
                    id=f"method:{method}",
                    label=method,
                    type="method",
                    properties={"name": method},
                ))
                edges.append(GraphEdge(
                    source=entity,
                    target=f"method:{method}",
                    relationship="USES",
                ))

            for dataset in neighborhood["datasets"]:
                nodes.append(GraphNode(
                    id=f"dataset:{dataset}",
                    label=dataset,
                    type="dataset",
                    properties={"name": dataset},
                ))
                edges.append(GraphEdge(
                    source=entity,
                    target=f"dataset:{dataset}",
                    relationship="EVALUATES_ON",
                ))

            for concept in neighborhood["concepts"][:10]:
                nodes.append(GraphNode(
                    id=f"concept:{concept}",
                    label=concept,
                    type="concept",
                    properties={"name": concept},
                ))
                edges.append(GraphEdge(
                    source=entity,
                    target=f"concept:{concept}",
                    relationship="DISCUSSES",
                ))

            for parent in neighborhood["extends"]:
                nodes.append(GraphNode(
                    id=parent["arxiv_id"],
                    label=parent["title"][:50],
                    type="paper",
                    properties=parent,
                ))
                edges.append(GraphEdge(
                    source=entity,
                    target=parent["arxiv_id"],
                    relationship="EXTENDS",
                ))

            for child in neighborhood["extended_by"]:
                nodes.append(GraphNode(
                    id=child["arxiv_id"],
                    label=child["title"][:50],
                    type="paper",
                    properties=child,
                ))
                edges.append(GraphEdge(
                    source=child["arxiv_id"],
                    target=entity,
                    relationship="EXTENDS",
                ))

            summary = {
                "methods_count": len(neighborhood["methods"]),
                "datasets_count": len(neighborhood["datasets"]),
                "concepts_count": len(neighborhood["concepts"]),
                "tasks_count": len(neighborhood["tasks"]),
                "extends_count": len(neighborhood["extends"]),
                "extended_by_count": len(neighborhood["extended_by"]),
            }

        else:
            # For non-paper entities, find connected papers
            results = expand_neighbors(
                entity_names=[entity],
                hops=hops,
                client=neo4j,
            )

            nodes.append(GraphNode(
                id=f"{entity_type}:{entity}",
                label=entity,
                type=entity_type,
                properties={"name": entity},
            ))

            for paper in results[:15]:
                nodes.append(GraphNode(
                    id=paper["arxiv_id"],
                    label=paper["title"][:50],
                    type="paper",
                    properties=paper,
                ))
                edges.append(GraphEdge(
                    source=f"{entity_type}:{entity}",
                    target=paper["arxiv_id"],
                    relationship="CONNECTED_TO",
                ))

            summary = {"connected_papers": len(results)}

        return GraphExploreResponse(
            entity=entity,
            entity_type=entity_type,
            nodes=nodes,
            edges=edges,
            neighborhood_summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph explore failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /papers ───────────────────────────────────────────────

@router.get("/papers")
async def list_papers():
    """List all ingested papers with basic metadata."""
    try:
        from ingestion.db import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT arxiv_id, title, year, total_chunks,
                       array_length(authors, 1) as author_count
                FROM papers
                ORDER BY year DESC, title
            """)).fetchall()

        return {
            "total": len(rows),
            "papers": [
                {
                    "arxiv_id": r[0],
                    "title": r[1],
                    "year": r[2],
                    "total_chunks": r[3],
                    "author_count": r[4] or 0,
                }
                for r in rows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /stats ────────────────────────────────────────────────

@router.get("/stats")
async def get_stats():
    """Get system statistics — paper count, chunk count, graph stats."""
    try:
        from ingestion.db import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            paper_count = conn.execute(
                text("SELECT COUNT(*) FROM papers")
            ).fetchone()[0]
            chunk_count = conn.execute(
                text("SELECT COUNT(*) FROM chunks")
            ).fetchone()[0]
            query_count = conn.execute(
                text("SELECT COUNT(*) FROM query_logs")
            ).fetchone()[0]

        neo4j = get_neo4j_client()
        graph_stats = neo4j.get_stats()

        return {
            "postgres": {
                "papers": paper_count,
                "chunks": chunk_count,
                "queries_logged": query_count,
            },
            "neo4j": {
                "nodes": {r["label"]: r["count"] for r in graph_stats["nodes"]},
                "relationships": {
                    r["type"]: r["count"]
                    for r in graph_stats["relationships"]
                },
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache-stats")
async def cache_stats():
    """Get semantic cache statistics."""
    from ingestion.db import engine
    from retrieval.semantic_cache import get_cache_stats
    return get_cache_stats(engine)


@router.delete("/cache")
async def clear_cache():
    """Clear the semantic cache. Dev use only."""
    from ingestion.db import engine
    from retrieval.semantic_cache import clear_cache
    clear_cache(engine)
    return {"status": "cache cleared"}

# ── GET /health ───────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — verify all system components are reachable."""
    postgres_ok = False
    neo4j_ok = False
    model_ok = False
    paper_count = 0
    chunk_count = 0

    # Check Postgres
    try:
        from ingestion.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            paper_count = conn.execute(
                text("SELECT COUNT(*) FROM papers")
            ).fetchone()[0]
            chunk_count = conn.execute(
                text("SELECT COUNT(*) FROM chunks")
            ).fetchone()[0]
        postgres_ok = True
    except Exception as e:
        logger.warning(f"Postgres health check failed: {e}")

    # Check Neo4j
    try:
        neo4j = get_neo4j_client()
        neo4j_ok = neo4j.test_connection()
    except Exception as e:
        logger.warning(f"Neo4j health check failed: {e}")

    # Check embedding model
    try:
        from retrieval.search import get_model
        get_model()
        model_ok = True
    except Exception as e:
        logger.warning(f"Model health check failed: {e}")

    overall = "healthy" if all([postgres_ok, neo4j_ok]) else "degraded"

    return HealthResponse(
        status=overall,
        postgres=postgres_ok,
        neo4j=neo4j_ok,
        embedding_model=model_ok,
        paper_count=paper_count,
        chunk_count=chunk_count,
    )


# ── GET /query-logs ───────────────────────────────────────────

@router.get("/query-logs")
async def get_query_logs(limit: int = 20, offset: int = 0):
    """Get recent query logs for monitoring."""
    try:
        from ingestion.db import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT id, query, top_k, answer,
                       latency_ms, pipeline_variant, created_at
                FROM query_logs
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """), {"limit": limit, "offset": offset}).fetchall()

            total = conn.execute(
                text("SELECT COUNT(*) FROM query_logs")
            ).fetchone()[0]

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "logs": [
                {
                    "id": r[0],
                    "query": r[1][:100],
                    "top_k": r[2],
                    "answer_preview": (r[3] or "")[:150],
                    "latency_ms": r[4],
                    "pipeline_variant": r[5],
                    "created_at": str(r[6]),
                }
                for r in rows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── POST /ingest-async (non-blocking) ────────────────────────

@router.post("/ingest-async")
async def ingest_async_endpoint(
    file: UploadFile = File(...),
):
    """
    Non-blocking PDF ingestion.
    Returns job_id immediately — pipeline runs in background.

    Poll GET /ingest/status/{job_id} for progress.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files accepted"
        )

    import time
    start = time.time()

    # Save file
    from pathlib import Path
    upload_dir = Path("data/raw")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Submit background job — returns instantly
    from ingestion.job_queue import submit_job
    job_id = submit_job(
        pdf_path=str(file_path),
        filename=file.filename,
    )

    elapsed = int((time.time() - start) * 1000)

    return {
        "job_id":      job_id,
        "status":      "queued",
        "filename":    file.filename,
        "message":     "Ingestion queued. Poll /ingest/status/{job_id}",
        "poll_url":    f"/ingest/status/{job_id}",
        "queued_in_ms": elapsed,
    }


@router.get("/ingest/status/{job_id}")
async def ingest_status_endpoint(job_id: str):
    """
    Poll ingestion job status.

    Returns:
        status: queued | processing | complete | failed
        progress: 0-100
        message: current stage description
        result: final result (when complete)
    """
    from ingestion.job_queue import get_queue
    queue = get_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )

    return {
        "job_id":      job.job_id,
        "status":      job.status,
        "progress":    job.progress,
        "message":     job.message,
        "filename":    job.filename,
        "created_at":  job.created_at,
        "started_at":  job.started_at,
        "finished_at": job.finished_at,
        "result":      job.result,
        "error":       job.error[:200] if job.error else "",
    }


@router.get("/ingest/jobs")
async def list_jobs_endpoint():
    """List all ingestion jobs."""
    from ingestion.job_queue import get_queue
    return {
        "jobs": get_queue().get_all_jobs()
    }


# ── GET /recommend ────────────────────────────────────────────

@router.get("/recommend")
async def recommend_endpoint(
    arxiv_id: str,
    top_k: int = 5,
):
    """
    Get paper recommendations for a seed paper.
    Pure graph traversal — zero LLM calls, sub-50ms.

    Example: GET /recommend?arxiv_id=2010.11929&top_k=5
    """
    if not arxiv_id:
        raise HTTPException(
            status_code=400,
            detail="arxiv_id required"
        )

    import time
    start = time.time()

    try:
        from retrieval.recommender import get_recommendations
        from graph.neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        recs = get_recommendations(
            arxiv_id=arxiv_id,
            top_k=top_k,
            client=client,
        )

        elapsed = int((time.time() - start) * 1000)

        return {
            "seed_arxiv_id": arxiv_id,
            "top_k":         top_k,
            "latency_ms":    elapsed,
            "method":        "graph_traversal_only",
            "llm_calls":     0,
            "recommendations": [
                {
                    "arxiv_id":       r.arxiv_id,
                    "title":          r.title,
                    "year":           r.year,
                    "score":          r.score,
                    "shared_methods":  r.shared_methods,
                    "shared_datasets": r.shared_datasets,
                    "shared_concepts": r.shared_concepts,
                    "extends":         r.extends,
                    "extended_by":     r.extended_by,
                    "reasoning":       r.reasoning,
                }
                for r in recs
            ],
        }

    except Exception as e:
        logger.error(f"Recommend failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommend/entity")
async def recommend_by_entity_endpoint(
    entity: str,
    entity_type: str = "method",
    top_k: int = 5,
):
    """
    Find papers related to a specific entity.
    GET /recommend/entity?entity=ImageNet&entity_type=dataset
    GET /recommend/entity?entity=attention&entity_type=method
    """
    try:
        from retrieval.recommender import get_recommendations_for_entity
        from graph.neo4j_client import get_neo4j_client

        import time
        start = time.time()

        client = get_neo4j_client()
        papers = get_recommendations_for_entity(
            entity_name=entity,
            entity_type=entity_type,
            top_k=top_k,
            client=client,
        )

        elapsed = int((time.time() - start) * 1000)

        return {
            "entity":      entity,
            "entity_type": entity_type,
            "latency_ms":  elapsed,
            "papers":      papers,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ── POST /query-checked (with self-check) ────────────────────

@router.post("/query-checked")
async def query_checked_endpoint(request: QueryRequest):
    """
    Query with post-generation hallucination self-check.
    Makes one extra LLM call to verify faithfulness.
    Returns flagged answer with ⚠️ on unverified claims.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    from retrieval.hybrid_retriever import HybridKAGRetriever
    from retrieval.reranker import rerank
    from retrieval.generator import generate_answer
    from retrieval.hallucination_checker import (
        check_faithfulness, format_self_check_for_ui
    )
    import time

    start = time.time()

    try:
        # Stage 1: Retrieve
        retriever = HybridKAGRetriever()
        hybrid_results, trace = retriever.retrieve(
            request.query,
            top_k=request.top_k_retrieve,
            use_graph=request.use_graph,
        )

        # Stage 2: Rerank
        ranked = rerank(
            request.query, hybrid_results, top_k=request.top_k_rerank
        )

        # Stage 3: Generate
        answer_obj = generate_answer(
            request.query, ranked, max_chunks=request.top_k_rerank
        )

        # Stage 4: Self-check
        self_check = check_faithfulness(
            answer=answer_obj.answer,
            chunks=ranked,
        )

        elapsed = int((time.time() - start) * 1000)

        # Log to query_logs
        try:
            log_pipeline_result(
                run_pipeline(request.query, use_graph=request.use_graph)
            )
        except Exception:
            pass

        import re
        clean_answer = re.sub(
            r'\nCITATIONS:.*', '',
            answer_obj.answer, flags=re.DOTALL
        )
        clean_answer = re.sub(
            r'\nCONFIDENCE:.*', '', clean_answer, flags=re.DOTALL
        )

        return {
            "query":              request.query,
            "answer":             clean_answer.strip(),
            "flagged_answer":     self_check.flagged_answer,
            "confidence":         answer_obj.confidence,
            "self_check":         format_self_check_for_ui(self_check),
            "citations":          answer_obj.citations,
            "graph_boosted":      trace.graph_boosted_count,
            "total_latency_ms":   elapsed,
            "pipeline_variant":   "checked",
        }

    except Exception as e:
        logger.error(f"Checked query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /ab-summary ───────────────────────────────────────────

@router.get("/ab-summary")
async def ab_summary_endpoint():
    """
    Get A/B test summary from query_logs.
    Shows per-variant stats and paired comparisons.
    """
    from eval.ab_framework import get_ab_summary
    from ingestion.db import engine
    return get_ab_summary(engine)


# ── POST /ab-test ─────────────────────────────────────────────

@router.post("/ab-test")
async def ab_test_endpoint(body: dict):
    """
    Run A/B comparison for a single query.
    Logs both variants to query_logs.

    Body: {"query": "your question"}
    """
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    from eval.ab_framework import run_ab_comparison
    from ingestion.db import engine

    comparison = run_ab_comparison(query=query, engine=engine)

    return {
        "query":    comparison.query,
        "winner":   comparison.winner,
        "reason":   comparison.winner_reason,
        "vector": {
            "answer":      comparison.vector_result.answer[:500],
            "latency_ms":  comparison.vector_result.latency_ms,
            "confidence":  comparison.vector_result.confidence,
            "citations":   comparison.vector_result.citations_count,
        },
        "hybrid": {
            "answer":      comparison.hybrid_result.answer[:500],
            "latency_ms":  comparison.hybrid_result.latency_ms,
            "confidence":  comparison.hybrid_result.confidence,
            "citations":   comparison.hybrid_result.citations_count,
            "graph_boosted": comparison.hybrid_result.graph_boosted,
        },
    }

# ── GET /timeline ─────────────────────────────────────────────

@router.get("/timeline")
async def timeline_endpoint(
    arxiv_id: str = None,
    concept:  str = None,
    max_depth: int = 4,
):
    """
    Build chronological timeline from a seed paper or concept.

    GET /timeline?arxiv_id=2010.11929
    GET /timeline?concept=self-supervised+learning
    """
    if not arxiv_id and not concept:
        raise HTTPException(
            status_code=400,
            detail="Provide arxiv_id or concept"
        )

    from graph.timeline_builder import (
        build_timeline, format_timeline_for_display
    )
    from graph.neo4j_client import get_neo4j_client
    import time

    start = time.time()
    client = get_neo4j_client()

    entries = build_timeline(
        seed_arxiv_id=arxiv_id,
        seed_concept=concept,
        max_depth=max_depth,
        client=client,
    )

    elapsed = int((time.time() - start) * 1000)

    return {
        "seed_arxiv_id": arxiv_id,
        "seed_concept":  concept,
        "entries":       format_timeline_for_display(entries),
        "total_papers":  len(entries),
        "latency_ms":    elapsed,
        "year_range": {
            "start": min(e.year for e in entries) if entries else 0,
            "end":   max(e.year for e in entries) if entries else 0,
        },
    }


# ── POST /query-structured ────────────────────────────────────

@router.post("/query-structured")
async def query_structured_endpoint(body: dict):
    """
    Query with structured JSON output.
    Returns answer formatted to match provided schema.

    Body:
    {
      "query": "compare ViT and Swin Transformer",
      "schema": "comparison",          # preset name
      "output_schema": {...}           # or custom JSON schema
    }
    """
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    schema_name   = body.get("schema", "")
    output_schema = body.get("output_schema", {})

    from retrieval.structured_output import (
        generate_structured_answer, PRESET_SCHEMAS
    )
    from retrieval.hybrid_retriever import HybridKAGRetriever
    from retrieval.reranker import rerank
    import time

    # Resolve schema
    if schema_name and schema_name in PRESET_SCHEMAS:
        output_schema = PRESET_SCHEMAS[schema_name]
    elif not output_schema:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provide 'schema' (one of: "
                f"{list(PRESET_SCHEMAS.keys())}) "
                f"or 'output_schema'"
            )
        )

    start = time.time()

    retriever = HybridKAGRetriever()
    results, trace = retriever.retrieve(query, top_k=15, use_graph=True)
    ranked = rerank(query, results, top_k=8)

    structured = generate_structured_answer(
        query=query,
        chunks=ranked,
        output_schema=output_schema,
    )

    elapsed = int((time.time() - start) * 1000)

    return {
        "query":             query,
        "schema_used":       schema_name or "custom",
        "data":              structured["data"],
        "schema_valid":      structured["schema_valid"],
        "validation_errors": structured["validation_errors"],
        "latency_ms":        elapsed,
        "graph_boosted":     trace.graph_boosted_count,
    }


# ── POST /ingest-incremental ──────────────────────────────────

@router.post("/ingest-incremental")
async def ingest_incremental_endpoint(body: dict):
    """
    Incrementally update graph for an existing paper.
    Only adds NEW entities — no full rebuild.

    Body: {"arxiv_id": "2010.11929"}
    """
    arxiv_id = body.get("arxiv_id", "").strip()
    if not arxiv_id:
        raise HTTPException(status_code=400, detail="arxiv_id required")

    from graph.auto_updater import auto_update_graph
    from ingestion.db import engine
    from sqlalchemy import text
    import json as json_mod
    from pathlib import Path

    # Load paper data
    paper_path = Path(f"data/processed/{arxiv_id}.json")
    if not paper_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Paper not found: {arxiv_id}"
        )

    with open(paper_path) as f:
        paper_data = json_mod.load(f)

    result = auto_update_graph(paper_data=paper_data)

    return {
        "arxiv_id":          result.arxiv_id,
        "title":             result.title,
        "methods_new":       result.methods_new,
        "methods_existing":  result.methods_existing,
        "datasets_new":      result.datasets_new,
        "datasets_existing": result.datasets_existing,
        "concepts_new":      result.concepts_new,
        "relationships_new": result.relationships_new,
        "elapsed_ms":        result.elapsed_ms,
    }

@router.get("/analytics")
async def analytics_endpoint(limit: int = 10):
    """
    Query log analytics:
    - Top queries by frequency
    - Slowest retrievals
    - Cache hit rate trend
    - Variant distribution
    """
    from ingestion.db import engine
    from sqlalchemy import text

    try:
        with engine.connect() as conn:

            # Top queries
            top = conn.execute(text("""
                SELECT query, COUNT(*) AS count,
                       AVG(latency_ms)::int AS avg_ms
                FROM query_logs
                GROUP BY query
                ORDER BY count DESC
                LIMIT :limit
            """), {"limit": limit}).fetchall()

            # Slowest queries
            slowest = conn.execute(text("""
                SELECT query, latency_ms, pipeline_variant,
                       created_at
                FROM query_logs
                WHERE pipeline_variant != 'cache_hit'
                ORDER BY latency_ms DESC
                LIMIT :limit
            """), {"limit": limit}).fetchall()

            # Cache hit rate
            cache = conn.execute(text("""
                SELECT
                    COUNT(*) FILTER (
                        WHERE pipeline_variant = 'cache_hit'
                    ) AS hits,
                    COUNT(*) AS total
                FROM query_logs
            """)).fetchone()

            # Variant distribution
            variants = conn.execute(text("""
                SELECT pipeline_variant,
                       COUNT(*) AS count
                FROM query_logs
                GROUP BY pipeline_variant
                ORDER BY count DESC
            """)).fetchall()

            # Latency over time (last 20)
            latency_trend = conn.execute(text("""
                SELECT latency_ms, pipeline_variant, created_at
                FROM query_logs
                ORDER BY created_at DESC
                LIMIT 20
            """)).fetchall()

        hit_rate = 0.0
        if cache and cache[1] > 0:
            hit_rate = round(cache[0] / cache[1], 3)

        return {
            "total_queries":   cache[1] if cache else 0,
            "cache_hit_rate":  hit_rate,
            "top_queries": [
                {
                    "query":   row[0][:80],
                    "count":   row[1],
                    "avg_ms":  row[2],
                }
                for row in top
            ],
            "slowest_queries": [
                {
                    "query":   row[0][:80],
                    "ms":      row[1],
                    "variant": row[2],
                    "when":    str(row[3]),
                }
                for row in slowest
            ],
            "variant_distribution": [
                {"variant": row[0], "count": row[1]}
                for row in variants
            ],
            "latency_trend": [
                {
                    "ms":      row[0],
                    "variant": row[1],
                    "when":    str(row[2]),
                }
                for row in latency_trend
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))