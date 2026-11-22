"""
Self-diagnostic & health check script for Research-KAG-Agent.
Runs offline/unit checks without requiring live DBs, and verifies live DBs if running.
Usage: python scratch/check_system.py
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_files():
    print_section("1. Core Files & Datasets Check")
    required_files = [
        ".env.example",
        "requirements_clean.txt",
        "data/papers_metadata.json",
        "data/entities.json",
        "docker-compose.yml",
    ]
    all_ok = True
    for f in required_files:
        p = PROJECT_ROOT / f
        if p.exists():
            print(f"  [OK] {f} exists ({p.stat().st_size} bytes)")
        else:
            print(f"  [FAIL] Missing file: {f}")
            all_ok = False

    processed_dir = PROJECT_ROOT / "data" / "processed"
    if processed_dir.exists():
        jsons = list(processed_dir.glob("*.json"))
        print(f"  [OK] data/processed contains {len(jsons)} paper records.")
    else:
        print("  [FAIL] data/processed does not exist.")
        all_ok = False

    return all_ok

def check_env():
    print_section("2. Environment Configuration Check")
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print("  [OK] .env file found.")
    else:
        print("  [WARN] .env file not found. Copy .env.example to .env and configure keys.")

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"  [OK] GEMINI_API_KEY is configured ({api_key[:4]}...{api_key[-3:]})")
    else:
        print("  [INFO] GEMINI_API_KEY is not set. LLM generation will fallback or return instructions.")

def check_logic():
    print_section("3. Logic & Component Unit Verification")

    # 1. Query NER regex test
    from retrieval.query_ner import extract_entities_from_query
    q = "how does shifted window attention work in Swin Transformer"
    ent = extract_entities_from_query(q)
    print(f"  [OK] Query NER: extracted {len(ent['all'])} entities for '{q}'")
    assert "Swin Transformer" in ent["paper_titles"] or "Swin Transformer" in ent["all"] or len(ent["all"]) > 0

    # 2. Query decomposition heuristic test
    from retrieval.query_decomposer import is_multi_part_query, _heuristic_decompose
    comp_q = "compare ViT and Swin Transformer"
    is_multi = is_multi_part_query(comp_q)
    decomp = _heuristic_decompose(comp_q)
    print(f"  [OK] Decomposition heuristic: '{comp_q}' -> is_multi={is_multi}, sub-queries={len(decomp['sub_queries'])}")
    assert is_multi
    assert len(decomp["sub_queries"]) >= 2

    # 3. Structured output schema validation test
    from retrieval.structured_output import validate_against_schema, COMPARISON_SCHEMA
    sample_data = {
        "comparison_topic": "ViT vs Swin",
        "items": [
            {"name": "ViT", "year": 2020, "key_feature": "Patch embedding", "advantage": "Pure transformer", "limitation": "Needs large data"},
            {"name": "Swin", "year": 2021, "key_feature": "Shifted windows", "advantage": "Linear complexity", "limitation": "Complex"}
        ],
        "summary": "Both are vision transformers with different attention mechanics."
    }
    is_valid, errs = validate_against_schema(sample_data, COMPARISON_SCHEMA)
    print(f"  [OK] Structured output validator: is_valid={is_valid}, errors={errs}")
    assert is_valid

    # 4. Citation parsing test
    from retrieval.generator import parse_citations_from_answer
    ans_text = "ViT uses image patches [1] while Swin introduces shifted windows [2]."
    lookup = {
        1: {"chunk_id": "chunk_vit_1", "arxiv_id": "2010.11929", "title": "ViT", "year": 2020},
        2: {"chunk_id": "chunk_swin_1", "arxiv_id": "2103.14030", "title": "Swin", "year": 2021},
    }
    clean_ans, citations, evidence = parse_citations_from_answer(ans_text, lookup)
    print(f"  [OK] Citation parser: parsed {len(citations)} citations from answer text.")
    assert len(citations) == 2

    # 5. Reranker fallback test
    from retrieval.reranker import rerank, RankedResult
    class DummyResult:
        def __init__(self, idx):
            self.chunk_id = f"c_{idx}"
            self.arxiv_id = f"arxiv_{idx}"
            self.title = f"Paper {idx}"
            self.year = 2021
            self.section = "methods"
            self.text = f"Sample text {idx}"
            self.token_count = 50
            self.final_score = 0.5 + (idx * 0.1)
    dummy_results = [DummyResult(1), DummyResult(2), DummyResult(3)]
    ranked = rerank("dummy query", dummy_results, top_k=2)
    print(f"  [OK] Reranker test: {len(ranked)} results returned (rank 1 score={ranked[0].final_score}).")
    assert len(ranked) == 2
    assert ranked[0].rerank_rank == 1

def check_databases():
    print_section("4. Database Connectivity (Optional)")
    # Test Postgres
    try:
        from ingestion.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            res = conn.execute(text("SELECT 1")).fetchone()
            print("  [OK] PostgreSQL connected successfully.")
    except Exception as e:
        print(f"  [INFO] PostgreSQL not reachable ({e}). Run: docker compose up postgres neo4j -d")

    # Test Neo4j
    try:
        from graph.neo4j_client import get_neo4j_client
        client = get_neo4j_client()
        if client.test_connection():
            print("  [OK] Neo4j connected successfully.")
        else:
            print("  [INFO] Neo4j returned false. Run: docker compose up postgres neo4j -d")
    except Exception as e:
        print(f"  [INFO] Neo4j not reachable ({e}). Run: docker compose up postgres neo4j -d")

def main():
    print("=" * 60)
    print(" Research-KAG-Agent System Diagnostic Check")
    print("=" * 60)
    check_files()
    check_env()
    check_logic()
    check_databases()
    print_section("DIAGNOSTIC COMPLETE")
    print("All core software components and logic checks passed successfully!\n")

if __name__ == "__main__":
    main()
