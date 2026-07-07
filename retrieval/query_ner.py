"""
Query-time Named Entity Recognition.
Extracts entity mentions from user queries to drive graph traversal.

Strategy:
- spaCy for fast extraction (queries happen on every request)
- Graph vocabulary matching to catch domain terms spaCy misses
- Returns entity names that exist in our Neo4j graph
"""

import re
import logging
from functools import lru_cache
import spacy
from graph.graph_queries import get_all_entity_names
from graph.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)

# Load spaCy once at module level
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

@lru_cache(maxsize=1)
def _get_graph_vocab()->dict:
    """
    Load all entity names from Neo4j into memory.
    Cached — only hits the DB once per process.
    Returns dict with methods, datasets, concepts, tasks, papers.
    """

    try:
        client = get_neo4j_client()
        vocab = get_all_entity_names(client)
        logger.info(
            f"Graph vocab loaded: "
            f"{len(vocab['methods'])} methods, "
            f"{len(vocab['datasets'])} datasets, "
            f"{len(vocab['concepts'])} concepts"
        )
        return vocab
    except Exception as e:
        logger.error(f"Failed to load graph vocab: {e}")
        return {"methods": [], "datasets": [], "concepts": [], "tasks": [], "papers": []}


def extract_entities_from_query(query: str) -> dict:
    """
    Extract named entities from a query string.

    Strategy:
    1. spaCy NER for standard entities (ORG, PRODUCT, WORK_OF_ART)
    2. Graph vocabulary matching — find known method/dataset names in query
    3. Capitalize first letters of multi-word matches for graph lookup

    Returns:
        {
            "all": [...],          # all entity strings found
            "methods": [...],      # matched Method nodes
            "datasets": [...],     # matched Dataset nodes
            "concepts": [...],     # matched Concept nodes
            "tasks": [...],        # matched Task nodes
            "paper_titles": [...], # matched Paper titles
        }
    """
    query_lower = query.lower()
    found = {
        "all": [],
        "methods": [],
        "datasets": [],
        "concepts": [],
        "tasks": [],
        "paper_titles": [],
    }

    # ── Step 1: spaCy extraction ──────────────────────────────
    if nlp:
        doc = nlp(query)
        spacy_entities = []
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART", "GPE", "PERSON"}:
                spacy_entities.append(ent.text)
        # Also grab noun chunks as potential entity mentions
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # multi-word only
                spacy_entities.append(chunk.text)
        found["all"].extend(spacy_entities)

    # ── Step 2: Graph vocabulary matching ─────────────────────
    vocab = _get_graph_vocab()

    # Match methods
    for method in vocab["methods"]:
        if method.lower() in query_lower:
            found["methods"].append(method)
            if method not in found["all"]:
                found["all"].append(method)

    # Match datasets
    for dataset in vocab["datasets"]:
        if dataset.lower() in query_lower:
            found["datasets"].append(dataset)
            if dataset not in found["all"]:
                found["all"].append(dataset)

    # Match concepts
    for concept in vocab["concepts"]:
        if concept.lower() in query_lower and len(concept) > 4:
            found["concepts"].append(concept)
            if concept not in found["all"]:
                found["all"].append(concept)

    # Match tasks
    for task in vocab["tasks"]:
        if task.lower() in query_lower:
            found["tasks"].append(task)
            if task not in found["all"]:
                found["all"].append(task)

    # Match paper short names / titles
    paper_keywords = {
        "vit": "An Image is Worth 16x16 Words",
        "swin": "Swin Transformer",
        "deit": "Training data-efficient image transformers",
        "mae": "Masked Autoencoders Are Scalable Vision Learners",
        "beit": "BEiT",
        "detr": "End-to-End Object Detection with Transformers",
        "dino": "Emerging Properties in Self-Supervised Vision Transformers",
        "bert": "BERT",
        "convit": "ConViT",
        "mobilevit": "MobileViT",
        "masked autoencoder": "Masked Autoencoders Are Scalable Vision Learners",
    }
    for keyword, title in paper_keywords.items():
        if keyword in query_lower:
            found["paper_titles"].append(title)
            if title not in found["all"]:
                found["all"].append(title)

    # ── Step 3: Deduplicate ───────────────────────────────────
    found["all"] = list(dict.fromkeys(found["all"]))  # preserve order
    found["methods"] = list(set(found["methods"]))
    found["datasets"] = list(set(found["datasets"]))
    found["concepts"] = list(set(found["concepts"]))
    found["tasks"] = list(set(found["tasks"]))
    found["paper_titles"] = list(set(found["paper_titles"]))

    logger.info(
        f"Query NER: '{query[:50]}' → "
        f"{len(found['methods'])} methods, "
        f"{len(found['datasets'])} datasets, "
        f"{len(found['concepts'])} concepts, "
        f"{len(found['paper_titles'])} papers"
    )
    return found

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "how does shifted window attention work in Swin Transformer",
        "what datasets were used to evaluate ViT and DeiT",
        "compare masked autoencoder pretraining with supervised methods",
        "which papers use patch embedding on ImageNet",
        "what is self-supervised learning in vision transformers",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        entities = extract_entities_from_query(q)
        print(f"  Methods:   {entities['methods']}")
        print(f"  Datasets:  {entities['datasets']}")
        print(f"  Concepts:  {entities['concepts']}")
        print(f"  Papers:    {entities['paper_titles']}")
        print(f"  All:       {entities['all']}")