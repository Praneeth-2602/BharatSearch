# BharatSearch Makefile
# One-command workflow per the Implementation Guide (Section 1.2).
#
#   make setup    - create environment, download models/data, build sample slice
#   make up       - start Compose stack for $(PROFILE)
#   make down     - stop Compose stack
#   make smoke    - end-to-end check on a tiny fixture
#   make test     - run unit + integration tests
#   make lint     - ruff check + format check

PROFILE ?= lite
include .env
export

COMPOSE := docker compose --profile $(PROFILE)

.PHONY: help setup up down clean data index eval benchmark smoke run-embedding run-retrieval demo kind-up kind-down autoscale-bench test lint

help:
	@echo "BharatSearch targets: setup up down clean data index eval benchmark smoke run-embedding run-retrieval demo kind-up kind-down autoscale-bench test lint"

setup: .env
	uv sync --dev
	@echo "Setup complete. Run 'make up' to start the stack."

.env: .env.example
	@cp .env.example .env
	@echo "Created .env from .env.example — review before running make up."

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	docker volume prune -f

data:
	uv run python -c "from data.make_variants import make_variants; import json, pathlib; pathlib.Path('data/processed').mkdir(parents=True, exist_ok=True); pathlib.Path('data/processed/variants.json').write_text(json.dumps(make_variants(['phone cover','shoes','water bottle','fone cover sasta'], seed=42), indent=2))"
	@echo "data: wrote data/processed/variants.json (sample queries; ESCI slice needs 'make setup' data download)"

index:
	uv run python -m data.index_products
	@echo "index: created OpenSearch 'products' index + Qdrant 'products' collection with sample catalog"

eval:
	uv run python -c "from eval.harness import run, write_metrics; qrels={'q1':{'p1':3,'p2':2,'p3':0}}; runs={'bm25':{'q1':['p1','p2','p3']}}; write_metrics(run(qrels, runs),'toy','results')"
	@echo "eval: wrote results/toy/metrics.json (toy evaluation; full eval needs indexed data)"

benchmark:
	uv run python -m eval.benchmark
	@echo "benchmark: wrote docs/benchmarks/$(shell date +%F).md (BM25 vs hybrid by slice)"

smoke:
	uv run pytest tests/ -q

demo:
	@echo "Running demo steps (requires stack up: make up PROFILE=standard)"
	@echo "Step 1: compare search results"
	uv run python -m demo.compare 'fone cover sasta' || true
	@echo "Step 5: view autoscaler results"
	@cat docs/benchmarks/autoscale-*.md 2>/dev/null || true

kind-up:
	@echo "kind-up: not yet implemented (Week 8)"

kind-down:
	@echo "kind-down: not yet implemented (Week 8)"

autoscale-bench:
	uv run python -m loadgen.make_curves --seed 1
	uv run python -m eval.autoscale_backtest --curve loadgen/curves/steady_seed1.csv --forecaster all --horizon-s 75
	uv run python -m eval.autoscale_backtest --curve loadgen/curves/diurnal_seed1.csv --forecaster all --horizon-s 75
	uv run python -m eval.autoscale_backtest --curve loadgen/curves/matchday_seed1.csv --forecaster all --horizon-s 75
	uv run python -m eval.autoscale_backtest --curve loadgen/curves/flashcrowd_seed1.csv --forecaster all --horizon-s 75
	@echo "autoscale-bench: wrote backtest results to stdout; use eval.autoscale_report.write_report() for the markdown report"

test:
	uv run pytest tests/ -m "not slow" -q

lint:
	uv run ruff check .
	uv run ruff format --check .

run-embedding:
	uv run python -m bharatsearch_embedding.app.main

run-retrieval:
	uv run python -m bharatsearch_retrieval.app.main
