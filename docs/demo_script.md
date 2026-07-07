
#══════════════════════════════════════════════════════════════════════╗
║         HYBRID KAG RESEARCH ASSISTANT — DEMO SCRIPT                 ║
║         3-5 minutes | Vision Transformer corpus                      ║
══════════════════════════════════════════════════════════════════════╝

SETUP (before demo):
  Terminal 1: uvicorn api.main:app --port 8000
  Terminal 2: streamlit run frontend/app.py --server.port 8501
  Browser 1:  http://localhost:8501   (Streamlit)
  Browser 2:  http://localhost:8000/docs  (Swagger)

═══════════════════════════════════════════════════════════════════════
PART 1: THE CORE PITCH (30 seconds)
═══════════════════════════════════════════════════════════════════════

"This is a hybrid Knowledge-Augmented Generation system built over
25 Vision Transformer papers. The key innovation is that it combines
pgvector dense retrieval with Neo4j knowledge graph traversal.

Standard RAG retrieves chunks by semantic similarity — it can't answer
questions that require following citation chains or finding papers that
share methods. The graph closes that gap."

═══════════════════════════════════════════════════════════════════════
PART 2: LIVE DEMO — 4 queries (3 minutes)
═══════════════════════════════════════════════════════════════════════

[Navigate to Streamlit → Query page]

QUERY 1 — Simple factual (shows basic RAG working):
  "What is the key innovation of Vision Transformer ViT?"

  Point out:
  → Answer cites specific papers with [1], [2] notation
  → Evidence panel shows which section each chunk came from
  → "Graph boosted: N" shows graph contributed

QUERY 2 — Multi-hop (shows graph advantage):
  "What datasets do papers extending ViT use for evaluation?"

  Point out:
  → This requires: ViT → [EXTENDS] → Papers → [EVALUATES_ON] → Datasets
  → Pure vector search can't answer this in one shot
  → Graph traversal finds DeiT, Swin, MAE and their datasets
  → Show the "graph_boosted" count is high

QUERY 3 — Comparison (triggers query decomposition):
  "Compare ViT and Swin Transformer architectures"

  Point out:
  → System detects this as multi-part query
  → Decomposes into two sub-queries automatically
  → Synthesizes both answers
  → "pipeline_variant: decomposed" in response

QUERY 4 — Contradiction detection:
  POST /query-enhanced (in Swagger UI):
  "what do papers say about data requirements for training transformers"

  Point out:
  → System flags ViT vs DeiT disagreement
  → "ViT: needs large-scale data" vs "DeiT: works with ImageNet only"
  → CONTRADICTS edge written to Neo4j automatically

═══════════════════════════════════════════════════════════════════════
PART 3: SHOW THE GRAPH (30 seconds)
═══════════════════════════════════════════════════════════════════════

[Navigate to Streamlit → Graph Explorer → enter 2010.11929]

Point out:
  → Paper neighborhood: methods, datasets, concepts
  → Related papers ranked by graph score
  → "Zero LLM calls" for recommendations
  → Timeline: follow EXTENDS chain

═══════════════════════════════════════════════════════════════════════
PART 4: NUMBERS (30 seconds)
═══════════════════════════════════════════════════════════════════════

[Navigate to System Stats]

  "The corpus is 25 papers, 865 chunks, 398 graph nodes, 466 edges.
   Retrieval latency is 113ms average. Cache hits drop that to 8ms.
   The evaluation used 40 hand-written Q&A pairs across three pipeline
   configurations — the results are in docs/ragas_results.md."

═══════════════════════════════════════════════════════════════════════
COMMON INTERVIEW QUESTIONS + ANSWERS
═══════════════════════════════════════════════════════════════════════

Q: Why Neo4j instead of just pgvector?
A: "pgvector finds semantically similar chunks — it's great for
   single-hop questions. Neo4j lets us traverse relationships: which
   papers extend ViT, which methods those papers use, which datasets
   they evaluate on. Multi-hop questions need the graph."

Q: How does the hybrid retrieval work?
A: "Four stages: NER extracts entities from the query, graph traversal
   finds structurally related papers, we run two vector searches
   (global + graph-filtered), then merge with a +0.3 score boost for
   graph-found chunks. The cross-encoder reranker then reorders by
   actual relevance, not just cosine similarity."

Q: Why not just use a bigger LLM context window?
A: "Context window approaches stuff all chunks into the prompt and hope
   the LLM figures it out. Our pipeline is explicit about what evidence
   is used, can cite specific papers, detects contradictions, and
   explains why each chunk was retrieved. That's much more auditable."

Q: What would you do differently at scale?
A: "Replace the thread-based job queue with Celery + Redis. Add
   streaming responses for the generation step. Shard the Neo4j graph
   for larger corpora. Use a dedicated vector DB like Pinecone instead
   of pgvector for 10M+ chunks."

Q: How did you evaluate it?
A: "40 hand-written Q&A pairs spanning factual, comparison, and
   multi-hop question types. Evaluated across three configurations:
   naive vector, vector+reranker, hybrid KAG. Used Gemini as the
   judge for faithfulness, relevancy, context precision and recall —
   same methodology as RAGAS."

═══════════════════════════════════════════════════════════════════════
DEMO BACKUP (if something breaks)
═══════════════════════════════════════════════════════════════════════

API health:  curl http://localhost:8000/health
Quick query: curl -X POST http://localhost:8000/query \
             -H "Content-Type: application/json" \
             -d '{"query": "how does ViT work?", "use_graph": true}'
Graph recs:  curl "http://localhost:8000/recommend?arxiv_id=2010.11929"
Timeline:    curl "http://localhost:8000/timeline?arxiv_id=2010.11929"
