#!/usr/bin/env bash
# BharatSearch demo script (~10 minutes).
# Implementation Guide, Section 12.2.
#
# Prerequisites: stack already up (`make up PROFILE=standard`), Grafana open.
#
# Steps:
#   0. Healthy dashboards
#   1. Compare BM25 vs BharatSearch on Hinglish/typo queries
#   2. Start the click simulator
#   3. Rollout: bad model rejected, good model promoted
#   4. Kill model server: search keeps working in fallback mode
#   5. Show stored autoscaler result plots
#   6. Recap with headline tables
set -euo pipefail

echo "== BharatSearch demo =="
echo "Prerequisites: make up PROFILE=standard, Grafana at http://localhost:3000"
echo

echo "[Step 0] Stack health"
curl -s http://localhost:8002/healthz | head -c 80
echo
curl -s http://localhost:8002/readyz | head -c 80
echo

echo "[Step 1] Compare BM25 vs BharatSearch on Hinglish queries"
python -m demo.compare 'fone cover sasta'
python -m demo.compare 'water bottle'
python -m demo.compare 'shoes'

echo "[Step 2] Start click simulator"
# uv run python -m bharatsearch_simulator.app.main  # runs against Kafka
echo "(simulator emits clicks; watch consumer lag and feature freshness in Grafana)"

echo "[Step 3] Rollout: reject bad model, promote good one"
# uv run python -m bharatsearch_rollout.app.main  # or via Redis rollout:current key
echo "(see docs/adr/0004-consumer-design.md and the rollout controller gates)"

echo "[Step 4] Kill model server: search keeps working in fallback mode"
echo "(understanding source splits to fallback on the Grafana search dashboard)"

echo "[Step 5] Show stored autoscaler result plots"
echo "open docs/benchmarks/autoscale-*.md"

echo "[Step 6] Recap"
cat docs/benchmarks/*.md 2>/dev/null | head -40

echo "== Demo complete =="
