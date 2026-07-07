"""
Pydantic models for API request and response validation.
FastAPI uses these for automatic docs + input validation.
"""

from pydantic import BaseModel, Field
from typing import Optional

# ── /query endpoint ───────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about the paper corpus",
        example="How does shifted window attention work in Swin Transformer?"
    )

    top_k_retrieve: int = Field(
        default = 20,
        ge = 5, 
        le = 50,
        description = "Number of chunks to retrieve before reranking"
    )

    top_k_rerank: int = Field(
        default=8,
        ge=3,
        le=20,
        description="Number of chunks to keep after reranking"
    )

    use_graph: bool = Field(
        default=True,
        description="Use graph-augmented retrieval (False = vector only)"
    )
    pipeline_variant: str = Field(
        default="hybrid",
        description="Pipeline variant label for logging"
    )

class CitationModel(BaseModel):
    citation_number: int
    chunk_id: str
    arxiv_id: str
    title: str
    year: int

class ChunkModel(BaseModel):
    chunk_id: str
    arxiv_id: str
    title: str
    year: int
    section: str
    text: str
    rerank_score: float
    rerank_rank: int
    source: str
    retrieval_path: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    not_found: bool
    citations: list[CitationModel]
    chunks_used: list[ChunkModel]
    entity_count: int
    graph_papers_found: int
    graph_boosted_count: int
    total_latency_ms: int
    pipeline_variant: str

# ── /ingest endpoint ──────────────────────────────────────────
class IngestResponse(BaseModel):
    arxiv_id: str
    title: str
    chunks_created: int
    sections_found: list[str]
    entities_extracted: dict
    graph_nodes_created: int
    latency_ms: int
    status: str

# ── /graph-explore endpoint ───────────────────────────────────

class GraphExploreRequest(BaseModel):
    entity: str = Field(
        ...,
        description="Entity name or arxiv_id to explore",
        example="2103.14030"
    )
    entity_type: str = Field(
        default="paper",
        description="Type: paper, method, dataset, concept, task"
    )
    hops: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Graph traversal depth"
    )

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str


class GraphExploreResponse(BaseModel):
    entity: str
    entity_type: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    neighborhood_summary: dict

# ── /health endpoint ──────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    postgres: bool
    neo4j: bool
    embedding_model: bool
    paper_count: int
    chunk_count: int

# ── /query response with explanations ────────────────────────

class ContradictionModel(BaseModel):
    paper_a: str
    paper_b: str
    arxiv_id_a: str
    arxiv_id_b: str
    claim_a: str
    claim_b: str
    topic: str
    confidence: str
    method: str


class ChunkExplanationModel(BaseModel):
    chunk_id: str
    title: str
    year: int
    section: str
    scores: dict
    retrieval: dict
    explanation: str


class EnhancedQueryResponse(BaseModel):
    """Extended query response with contradictions + explanations."""
    query: str
    answer: str
    confidence: str
    not_found: bool
    citations: list[CitationModel]
    chunks_used: list[ChunkModel]
    explanations: list[ChunkExplanationModel]
    contradictions: list[ContradictionModel]
    contradiction_count: int
    entity_count: int
    graph_papers_found: int
    graph_boosted_count: int
    total_latency_ms: int
    pipeline_variant: str