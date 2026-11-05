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

.PHONY: help setup up down clean data index eval smoke demo kind-up kind-down autoscale-bench test lint

help:
	@echo "BharatSearch targets: setup up down clean data index eval smoke demo kind-up kind-down autoscale-bench test lint"

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
	@echo "data: not yet implemented (Week 2)"

index:
	@echo "index: not yet implemented (Week 2-3)"

eval:
	@echo "eval: not yet implemented (Week 3)"

smoke:
	uv run pytest tests/ -q

demo:
	@echo "demo: not yet implemented (Week 10)"

kind-up:
	@echo "kind-up: not yet implemented (Week 8)"

kind-down:
	@echo "kind-down: not yet implemented (Week 8)"

autoscale-bench:
	@echo "autoscale-bench: not yet implemented (Week 8-9)"

test:
	uv run pytest tests/ -m "not slow" -q

lint:
	uv run ruff check .
	uv run ruff format --check .
