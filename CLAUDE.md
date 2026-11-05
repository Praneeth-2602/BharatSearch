# BharatSearch — Claude Code Project Conventions

## Commit conventions

- **Do not reorganize or squash the existing 3 commits.** Keep them as-is:
  - `c466477` Week 1 bootstrap: uv workspace, libs, stub-inference, compose, Makefile, tests
  - `792a13d` Week 2: data pipeline and search baseline
  - `372a752` Week 3: hybrid retrieval, search API, and benchmark
- **Going forward, do not use week-wise commit messages.** Group similar task files together
  and name commits descriptively by what they add/change (e.g. "Add embedding service with
  stub backend", "Add retrieval search API with BM25 + dense + RRF fusion").
- Do not append "Co-Authored-By: Claude Code" or any AI attribution line to commit messages.

## Repo-local git config

- `user.name` = `praneeth-2602`
- `user.email` = `praneeth.palugula@gmail.com`
- These are set repo-local (not global). Use `gh auth switch` to the `Praneeth-2602`
  personal account before pushing to `https://github.com/Praneeth-2602/BharatSearch`.

## Project structure

- `libs/contracts` — shared pydantic v2 models (SearchRequest/Response, EmbedRequest/Response, Understanding, EventEnvelope)
- `libs/common` — config, logging, tracing, metrics helpers
- `services/embedding` — FastAPI app, POST /embed, port 8001 (stub 384-dim embeddings)
- `services/retrieval` — FastAPI app, POST /v1/search (BM25 + dense + RRF), port 8002
- `eval/` — metrics, harness, benchmark script
- `tests/` — unit tests (pytest, no pytest-asyncio; use asyncio.run() for async tests)
- `data/` — data pipeline (make_variants, make_slice, review, stats)

## Development commands

- `uv run pytest tests/ -q` — run unit tests
- `uv run ruff check . && uv run ruff format --check .` — lint + format check
- `make lint` — ruff check + format check
- `make test` — pytest with `-m "not slow"`
- `make eval` — toy evaluation via eval.harness
- `make benchmark` — BM25 vs hybrid benchmark report (writes docs/benchmarks/<date>.md)
- `make run-embedding` — start embedding service in dev mode
- `make run-retrieval` — start retrieval service in dev mode

## Ruff config

- line-length = 100
- select = E, F, I, B, C4, PLC, PLE, PLR, PLW
- tests/* ignores PLR2004

## Pytest config

- testpaths = ["tests"]
- markers: unit, integration, e2e, slow
- No pytest-asyncio; use `asyncio.run()` for async code in tests
