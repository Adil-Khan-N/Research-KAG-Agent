Day 1 — Environment, Repo, and Tooling Setup
Date: 2026-06-19
What I Built:

Created research-kag-agent/ repo with full folder scaffold (ingestion/, retrieval/, agents/, api/, eval/, frontend/, docker/, docs/)
Set up Python 3.11 virtual environment with requirements.txt skeleton
Wrote .env.example with placeholder keys for LLM API, Postgres, and Neo4j
Wrote docker-compose.yml bringing up Postgres (pgvector) + Neo4j Community
Verified Postgres reachable via psql and Neo4j Browser at localhost:7474
Confirmed LLM API key works with a successful "hello world" completion call
Initialized git, wrote placeholder README with architecture sketch
Tagged initial commit

One Number I Measured:


One Thing That Broke and How I Fixed It:

(fill in — e.g., "pgvector image tag mismatch, fixed by switching to pgvector/pgvector:pg16")

Checkpoint Status:

 docker compose up starts Postgres + Neo4j cleanly
 psql connects to Postgres; Neo4j Browser loads at localhost:7474
 Test LLM call returns a response successfully

Tomorrow (Day 2):

Study embeddings, pgvector internals, chunking strategies, and RAG fundamentals — plus write the 20-line cosine similarity micro-exercise.


Day 2 — Knowledge Gain
"What is an embedding?"

"An embedding is a dense vector representation of text produced by a neural network. Semantically similar texts produce vectors with high cosine similarity — meaning they point in roughly the same direction in high-dimensional space. Models like all-MiniLM-L6-v2 produce 384-dimensional vectors."

"Why is IVFFlat approximate?"

"IVFFlat clusters all vectors into buckets at index build time. At query time it only searches the nearest buckets rather than every vector. This means it can miss the true nearest neighbor if it falls in an unprobed bucket — but for RAG with reranking downstream, this tradeoff is acceptable."

"Why does chunk size affect retrieval precision and recall?"

"Smaller chunks are more specific — high precision, lower recall, because a relevant passage might span chunk boundaries. Larger chunks have higher recall but dilute the embedding signal, hurting precision. Around 300 tokens with overlap balances both."

What I learned:
- all-MiniLM-L6-v2 outputs 384-dim vectors (not 768 — updating Day 5 schema)
- IVFFlat is approximate: clusters vectors into `lists` buckets, probes only nearest ones
- ~300 tokens with overlap is the sweet spot for chunk size
- Naive RAG fails multi-hop questions because no single chunk spans both hops

## Day 5 — PostgreSQL Schema + Embedding Pipeline

Date: 2026-06-24

What I built:
- ingestion/db.py — SQLAlchemy engine + connection helper
- ingestion/schema.py — papers, chunks, query_logs tables + IVFFlat index
- ingestion/embedder.py — batch embedding with all-MiniLM-L6-v2 (384-dim)
- retrieval/search.py — cosine distance vector search with paper metadata join

One number I measured:
- Total chunks embedded: 865 across 24 papers
- Embedding time: 32.1s total, 1.5s avg per paper
- Best similarity score: 0.71 (masked autoencoder query)
- Search latency: <100ms per query

One thing that broke and how I fixed it:
- NUL (0x00) characters in 3 PDFs caused Postgres insert failure
  → Fixed extract_text.py to strip NUL bytes, deleted bad JSONs,
    deleted those 3 rows from papers table, re-ran pipeline

Checkpoint: PASSED
Tomorrow: Day 6 — Neo4j entity extraction + knowledge graph population

## Day 6 — Knowledge Graph Schema + Entity Extraction

Date: 2026-06-25

What I built:
- graph/neo4j_client.py — Neo4j driver wrapper with stats helper
- graph/schema.py — uniqueness constraints + indexes for all 6 node types
- graph/entity_extractor.py — Gemini 1.5 Flash extraction with auto-resume
- graph/graph_ingestion.py — MERGE pipeline + EXTENDS resolver
- scratch/verify_graph.py — 10 Cypher queries proving graph connectivity

One number I measured:
- Total nodes: 398 (124 Concepts, 109 Methods, 93 Authors,
                     32 Tasks, 25 Papers, 15 Datasets)
- Total relationships: 466 (129 DISCUSSES, 118 USES, 109 AUTHORED_BY,
                            55 ADDRESSES, 29 EXTENDS, 26 EVALUATES_ON)
- Multi-hop query [6] returned 4 papers with correct methods via 2-hop
  traversal — first proof that the graph adds value over vector search
- No isolated paper nodes — 100% connectivity

One thing that broke and how I fixed it:
- Gemini returned markdown fences around JSON despite instructions
  → stripped with re.sub() before json.loads()
- Stale entities.json with empty entries from partial first run
  → deleted file, re-ran from scratch cleanly

Checkpoint: PASSED
Tomorrow: Day 7 — Cypher query library, graph validation,
          fix bad EXTENDS edges, write parameterized query functions

## Day 7 — Cypher Query Library + Graph Validation

Date: 2026-06-26

What I built:
- scratch/audit_graph.py — reviewed all EXTENDS, deleted suspicious ones
- graph/graph_queries.py — 8 parameterized Cypher functions:
    expand_neighbors, get_papers_by_method, get_papers_by_dataset,
    get_citation_chain, find_papers_sharing_concepts,
    find_contradicting_papers, get_paper_neighborhood,
    get_all_entity_names
- scratch/test_graph_queries.py — all 8 functions tested on real corpus

One number I measured:
- expand_neighbors(['Multi-Head Attention', 'ImageNet']): 7 papers
- ViT descendants: 5 papers (DeiT, Swin V1/V2, S²-MLPv2, ViT-Small)
- ViT ancestors: 2 papers (Attention Is All You Need, DETR)
- Multi-hop demo: 3 papers returned with full methods + datasets
  in a single Cypher query — impossible with flat vector search
- Graph vocabulary: 109 methods, 15 datasets, 124 concepts, 32 tasks

One thing that broke and how I fixed it:
- CONTRADICTS relationship warning from Neo4j — expected, no explicit
  contradiction edges exist yet (added in Day 15). Fallback to
  implicit dataset-overlap detection working correctly.
- Dataset duplicates in Test 3 (DeiT III appearing twice) — fixed
  by adding DISTINCT to get_papers_by_dataset query.

Checkpoint: PASSED
Tomorrow: Day 8 — HybridKAGRetriever — wire graph + vector together,
          the single most important class in the project

Query: "What datasets were used to evaluate shifted window attention?"

Step 1 — Query NER:
  spaCy extracts entities → ["shifted window attention", "datasets"]
  Graph vocab matching → ["Shifted Window Attention"] (Method node)

Step 2 — Graph Traversal:
  expand_neighbors(["Shifted Window Attention"]) 
  → finds Swin V1, Swin V2, papers using same method
  → returns arxiv_ids: ["2103.14030", "2111.09883", ...]

Step 3 — Vector Search (two passes):
  Pass A: embed raw query → top-20 chunks (standard)
  Pass B: embed raw query → top-20 chunks filtered to graph papers only

Step 4 — Merge + Deduplicate:
  Combine Pass A + Pass B chunks
  Graph-boosted chunks get +0.3 score bonus
  Deduplicate by chunk_id

Step 5 — Return top-20 merged results
  (Day 9 reranker will cut to top-8)

## Day 8 — HybridKAGRetriever (Core Differentiator)

Date: 2026-06-28

What I built:
- retrieval/query_ner.py — spaCy + graph vocabulary entity extraction
    Loads all 109 methods, 15 datasets, 124 concepts from Neo4j into
    memory (cached), matches against query using CONTAINS.
    Paper keyword dict catches short names: "ViT", "DeiT", "MAE", "Swin"
- retrieval/hybrid_retriever.py — 4-stage HybridKAGRetriever:
    Stage 1: Query NER → entity list
    Stage 2: expand_neighbors() → graph paper set
    Stage 3a: vector search top-20 (all chunks)
    Stage 3b: vector search top-20 (graph papers only)
    Stage 4: merge + score fusion (vector + 0.3 graph boost) + dedup
- retrieval/vector_only.py — pure vector baseline for A/B comparison
- scratch/test_hybrid_retriever.py — 6 tests including comparison

One number I measured:
- Average retrieval latency: 113ms (after model warmup)
- First-run latency: 7349ms (model loading — one-time cost)
- Min latency: 69ms | Max: 147ms
- Test 1 (Swin query): 10/10 results graph-boosted, top score 0.8984
- Test 3 (multi-hop ViT): 8/8 results graph-boosted
- Test 4 (no entities): graceful fallback to vector, 5 results returned
- Graph boost effect: +0.3 added to vector score for graph-found chunks

What the graph adds vs vector-only:
- Specific queries (Swin, ViT): graph correctly prioritizes
  domain-relevant papers even when raw vector score is moderate
- Multi-hop queries: graph finds papers via entity traversal that
  pure vector search ranks lower
- General queries: honest fallback — no fake improvement claimed

Issues found and fixed:
- "Masked Autoencoder" not matched by NER → added "mae" and
  "masked autoencoder" to paper_keywords dict in query_ner.py
- File named test_hybrid_retrieval.py instead of test_hybrid_retriever.py
  → renamed to match module convention
- DeiT graph boost not firing → paper title match not connecting to
  graph node in expand_neighbors (noted for Day 14 fix)

Checkpoint: PASSED

Query + 20 hybrid chunks
        ↓
Cross-encoder reranker (BAAI/bge-reranker-base)
  — scores each (query, chunk) pair independently
  — much more accurate than cosine similarity
  — keeps top 8
        ↓
Gemini 1.5 Flash
  — receives query + top 8 chunks as evidence
  — instructed to cite sources, not hallucinate
  — returns structured cited answer
        ↓
Final answer with citations

## Day 9 — Reranking + LLM Generation

What I built:
- retrieval/reranker.py — BAAI/bge-reranker-base cross-encoder
    Scores (query, chunk) pairs together — more accurate than cosine
    Runs on top-20 chunks only, keeps top-8
- retrieval/generator.py — Gemini 1.5 Flash answer generation
    Evidence-grounded prompting with inline citations [1][2][3]
    Explicit "not found" instruction to prevent hallucination
    Parses citations from answer, builds structured output
- retrieval/pipeline.py — end-to-end orchestration
    retrieve → rerank → generate in one call
    Supports hybrid and vector_only variants for A/B comparison
- scratch/test_pipeline.py — 6 tests + example query generation
- docs/example_queries.md — 5 real Q&A pairs for README

One number I measured:
- Reranker latency: __ms for 20 chunks
- Generator latency: __ms per query
- Total pipeline latency: __ms average
- Average citations per answer: __
- Example query confidence: __ high, __ medium, __ low

One thing that broke and how I fixed it:
- (common: Gemini returns CITATIONS section inside answer text
  → strip with re.sub before display)
- (common: cross-encoder logits shape varies by model
  → handle both (batch, 1) and (batch, 2) output shapes)

Checkpoint: PASSED / FAILED
Tomorrow: Day 10 — FastAPI backend
          POST /query, POST /ingest, GET /graph-explore endpoints

## Day 10 — FastAPI Backend

What I built:
- api/models.py — Pydantic request/response models with validation
- api/routes.py — 7 endpoints:
    POST /query — full hybrid pipeline
    POST /ingest — PDF ingestion
    GET /graph-explore — graph neighborhood
    GET /papers — list all papers
    GET /stats — system stats
    GET /health — component health check
    GET /query-logs — recent query history
- api/main.py — FastAPI app with CORS, lifespan, auto-docs
- scratch/test_api.py — 11 HTTP tests + no-server mode

One number I measured:
- /query round-trip latency: __ms
- /health response time: __ms
- /graph-explore nodes returned for ViT: __
- query_logs rows after tests: __

One thing that broke and how I fixed it:
- (common: CORS error from Streamlit → added CORSMiddleware)
- (common: model cold start on first /query → pre-load in lifespan)

Semantic Cache:
  First query:  "how does ViT work?" → 340ms (cold)
  Same query:   "how does ViT work?" → 8ms (cache hit)
  Similar query: "explain ViT architecture" → 12ms (cache hit, 0.94 similarity)

Query Decomposition:
  "Compare ViT and Swin Transformer" 
    → detected as multi-part
    → sub-query 1: "ViT architecture patch embedding"
    → sub-query 2: "Swin Transformer shifted window attention"
    → merged answer with both perspectives


## Day 15 — Contradiction Detector + Retrieval Explanation Layer

Date: 2026-06-25

What I built:
- retrieval/contradiction_detector.py:
    _keyword_contradiction_check() — fast regex, zero API cost
      correctly flags: large/small dataset, quadratic/linear complexity
    check_contradiction_llm() — Gemini judge, high accuracy
    detect_contradictions() — cross-paper pair checking (skip same arxiv_id)
    write_contradicts_to_neo4j() — writes CONTRADICTS edges for Day 19
- retrieval/explanation_layer.py:
    build_explanation() — 4 scores + graph path + rank change per chunk
    build_all_explanations() — full result set with pre/post rerank tracking
    format_explanations_for_api() — JSON-serializable for API response
- api/routes.py — /query-enhanced endpoint combining both features
- frontend/app.py — explanation panel + contradiction warnings in UI

One number I measured:
- Contradiction found: ViT vs DeiT on data requirements (high confidence)
- CONTRADICTS edges in Neo4j: 1 (2010.11929 → 2012.12877)
- Rank promotions by reranker: 2 chunks promoted >2 positions
- Rank demotions: 1 chunk demoted 2 positions
- /query-enhanced latency: 37977ms (includes LLM contradiction checks)
- /query standard latency: ~500ms (no contradiction checking)
- Explanations coverage: 100% of chunks get full provenance

Key findings:
- Reranker significantly reorders results: chunk ranked #4 by vector
  promoted to #1 by cross-encoder — proves reranker adds real value
- Graph boost correctly applied: all 5 Swin chunks are graph_boosted
  with matched entity "Swin Transformer"
- Contradiction detector correctly identifies ViT claim
  ("needs JFT-300M") vs DeiT claim ("works with ImageNet only")

One thing that broke and how I fixed it:
- ::jsonb cast syntax fails in SQLAlchemy text() params →
  changed to CAST(:param AS jsonb) in semantic_cache.py
- Day 14 cache tests were failing due to same syntax issue →
  fixed both INSERT and UPDATE statements

Checkpoint: PASSED
Tomorrow: Day 16 — Adaptive chunking strategy
  Section-aware chunk sizes, re-ingest with comparison,
  add third row to evaluation table showing chunking impact