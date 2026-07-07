#!/bin/bash
# Manual curl tests for Day 10 API
# Run: bash scratch/test_api_curl.sh

BASE="http://localhost:8000"

echo "=== Health Check ==="
curl -s $BASE/health | python -m json.tool

echo ""
echo "=== Stats ==="
curl -s $BASE/stats | python -m json.tool

echo ""
echo "=== Query (Hybrid) ==="
curl -s -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does ViT handle image patches?", "use_graph": true}' \
  | python -m json.tool

echo ""
echo "=== Graph Explore (ViT) ==="
curl -s "$BASE/graph-explore?entity=2010.11929&entity_type=paper" \
  | python -m json.tool

echo ""
echo "=== Query Logs ==="
curl -s "$BASE/query-logs?limit=3" | python -m json.tool