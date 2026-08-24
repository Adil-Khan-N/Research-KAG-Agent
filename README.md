# Hybrid KAG Research Assistant

> A production-grade **Hybrid Knowledge-Augmented Generation** system
> over scientific papers — combining pgvector dense retrieval, Neo4j
> knowledge graph traversal, cross-encoder reranking, and multi-agent
> orchestration. Built as a CV-defining project over 20 days.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138-green)]()
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-blue)]()
[![pgvector](https://img.shields.io/badge/pgvector-0.4-orange)]()

---

## Architecture
Query → NER → Graph Traversal → Vector Search → Rerank → Generate → Answer
↓          ↓                ↓              ↓
spaCy +    Neo4j Cypher    pgvector        BAAI/
Graph vocab  (398 nodes,     cosine sim    bge-reranker
466 edges)     IVFFlat       + Gemini 1.5

**3-database design:**
- **PostgreSQL + pgvector**: 865 chunks, 384-dim embeddings, IVFFlat index
- **Neo4j**: 398 nodes (Paper/Method/Dataset/Concept/Task/Author),
  466 relationships (EXTENDS/USES/EVALUATES_ON/DISCUSSES/CONTRADICTS)
- **Semantic cache**: pgvector similarity cache, ~{N}x speedup

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo>
cd research-kag-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start databases
docker compose up postgres neo4j -d

# 3. Create .env (copy from .env.example)
cp .env.example .env  # add GEMINI_API_KEY

# 4. Ingest papers
python -m ingestion.pipeline
python scratch/run_embeddings.py
python scratch/run_graph_ingestion.py

# 5. Start API
uvicorn api.main:app --port 8000

# 6. Start Streamlit UI
streamlit run frontend/app.py --server.port 8501
```

---

## Features

### Core Pipeline (Days 4-13)

| Feature | Details |
|---|---|
| PDF Ingestion | arXiv API + PyMuPDF, section tagging, 300-token chunks |
| Embedding | all-MiniLM-L6-v2, 384-dim, IVFFlat index |
| Knowledge Graph | 24 papers, 114 methods, 24 datasets extracted by Gemini |
| Hybrid Retrieval | NER → graph traversal → dual vector search → score fusion |
| Cross-Encoder Rerank | BAAI/bge-reranker-base, top-20 → top-8 |
| Answer Generation | Gemini 1.5 Flash, cited, grounded, "not found" fallback |

### Tier 1 Features (Days 14-15)

| Feature | Details |
|---|---|
| Semantic Cache | pgvector similarity, threshold=0.92, Nx latency reduction |
| Query Decomposition | heuristic + LLM, detects "compare X and Y" patterns |
| Contradiction Detector | keyword + LLM, flags opposing claims, writes Neo4j edges |
| Retrieval Explanation | per-chunk: vector score + graph path + rank change |

### Tier 2 Features (Days 16-17)

| Feature | Details |
|---|---|
| Adaptive Chunking | section-aware sizes: abstract=150, methods=300, results=400 tokens |
| Async Ingestion | thread-based job queue, instant job_id, poll /ingest/status |
| Graph Recommendations | pure Cypher, 0 LLM calls, weighted by shared edges |

### Tier 3 Features (Days 18-19)

| Feature | Details |
|---|---|
| Hallucination Self-Check | second LLM pass, flags unsupported claims with ⚠️ |
| A/B Framework | logs vector vs hybrid side-by-side to query_logs |
| Auto Graph Updater | incremental MERGE, diffs new vs existing entities |
| Timeline View | EXTENDS chain traversal, chronological paper lineage |
| Structured Output | JSON schema validation, 4 preset schemas |

---

## Corpus

**Domain:** Vision Transformers (24 papers, 2017-2022)

Key papers: ViT, DeiT, Swin V1/V2, MAE, BEiT, DETR, ConViT,
MobileViT, PVT, T2T-ViT, LeViT, CvT, CoAtNet, Attention Is All You Need

---

### Adaptive Chunking Ablation

Evaluated section-aware chunk sizes vs fixed 300-token chunks.
Results documented in `docs/ragas_results.md`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Full hybrid pipeline |
| POST | `/query-enhanced` | + contradiction detection + explanations |
| POST | `/query-checked` | + hallucination self-check |
| POST | `/query-structured` | + JSON schema output |
| POST | `/ingest-async` | Non-blocking PDF ingestion |
| GET | `/ingest/status/{job_id}` | Poll ingestion progress |
| GET | `/recommend` | Graph-based recommendations |
| GET | `/timeline` | Paper lineage timeline |
| GET | `/graph-explore` | Graph neighborhood |
| GET | `/ab-summary` | A/B test results |
| GET | `/analytics` | Query log analytics |
| GET | `/cache-stats` | Cache performance |
| GET | `/stats` | System stats |
| GET | `/health` | Health check |

Full docs at `http://localhost:8000/docs`

---

## Example Queries

See `docs/example_queries.md` for real outputs.

**Factual:**
- "What is the key innovation of Vision Transformer ViT?"
- "What datasets does Swin Transformer evaluate on?"
- "How does MAE masking differ from BERT?"

**Comparison (triggers query decomposition):**
- "Compare ViT and Swin Transformer architectures"
- "How does DeiT differ from ViT in training approach?"

**Multi-hop (graph required):**
- "What datasets do papers extending ViT use?"
- "Which methods are shared between papers evaluating on ImageNet?"

**Structured:**
- POST /query-structured with schema=comparison
- POST /query-structured with schema=timeline

---

## Project Structure
research-kag-agent/
├── ingestion/          # PDF → chunks → embeddings → Postgres
├── retrieval/          # hybrid retriever, reranker, cache, decomposer
├── graph/              # Neo4j client, queries, entity extraction
├── agents/             # multi-agent literature review pipeline
├── api/                # FastAPI backend
├── frontend/           # Streamlit UI
├── eval/               # RAGAS evaluation, A/B framework
├── data/               # processed papers, embeddings, eval results
└── docs/               # architecture, results, examples

---

## Technical Details

**Retrieval pipeline latency (after warmup):**
- Hybrid retrieval: ~113ms
- Cache hit: ~8ms
- Full pipeline (retrieve+rerank+generate): ~10-15s

**Graph stats:**
- 398 nodes: 25 Paper, 109 Method, 15 Dataset, 124 Concept, 32 Task
- 466 relationships: EXTENDS(29), USES(118), EVALUATES_ON(26),
  DISCUSSES(129), ADDRESSES(55), AUTHORED_BY(109)

---

*Built: July 2026 | Domain: Vision Transformers | Papers: 24*
