# Resume Project Tracking — BharatSearch

Tracks measurable deliverables for the BharatSearch search/retrieval project. Resume bullets use measured values only and link to exact result tables (Implementation Guide §14 "Honesty" rule).

## Project state (2026-11-14)

- Branch: `master`, 1 commit ahead of `BharatSearch/master` (5f86fb1 "Add ESCI search baseline with real embedder support")
- Stack (docker-compose, `lite` profile, all healthy): OpenSearch 9200, Qdrant 6333, Redis 6379, Kafka 9092
- Services (started directly): embedding :8001 (sentence-transformers `intfloat/multilingual-e5-small`, 384-dim, Apple Silicon MPS), retrieval :8002 (`POST /v1/search` hybrid BM25+dense RRF)
- Catalog: 100k slice = 50,481 judged products + 49,519 distractors; disjoint bench (2,000) / dev (750) / test (750) queries; `results/gold_dev.json`, `results/gold_test.json` written
- OpenSearch: slice re-indexed (100,000 docs) ✓
- Qdrant: slice re-indexed with real e5 passage vectors (99,998 points, status green) ✓ (`RECREATE=1 SLICE=1 uv run python -m data.reindex_qdrant_real`)
- Benchmark: `uv run python -m eval.run_esci_benchmark` → bm25/dense/hybrid + `ci_95` + `judged@10` ✓
- Lint/format: clean (`uv run ruff check . && uv run ruff format --check .`) ✓

## Pass / fail criteria (definition of done)

| # | Criterion | Status | Evidence / result table |
|---|-----------|--------|-------------------------|
| 1 | 100k slice catalog built (judged + distractors), disjoint bench/dev/test queries | done | `results/gold_dev.json`, `results/gold_test.json` |
| 2 | OpenSearch slice re-indexed (100k docs) | done | `curl /products/_count` → 100000 |
| 3 | Qdrant slice re-indexed with real e5 passage vectors (MPS, batch 256, sorted by length, pipelined) | done | `data/reindex_qdrant_real.py`; 99,998 points, green |
| 4 | Embedding throughput profile captured (MPS verified, max len 128, sort by length, batch 256, producer/consumer pipeline) | done | `/tmp/reindex_slice.log` |
| 5 | BM25-only baseline with judged@10 | done | `results/esci_benchmark/esci/metrics.json` |
| 6 | Dense-only baseline with judged@10 | done | `results/esci_benchmark/esci/metrics.json` |
| 7 | Hybrid (RRF) baseline with judged@10 | done | `results/esci_benchmark/esci/metrics.json` |
| 8 | Bootstrap 95% CI per config/metric (1,000 resamples) | done | `results/esci_benchmark/esci/metrics.json` (`ci_95`) |
| 9 | ruff check + format clean | done | `uv run ruff check . && uv run ruff format --check .` |
| 10 | Unit tests pass (`make test`) | pending | `uv run pytest tests/ -q -m "not slow"` |
| 11 | Commit + push to `Praneeth-2602/BharatSearch` (descriptive message, no Co-Authored-By) | pending | `git push` |

## Measured results (ESCI benchmark, 2,000 clean English queries, k=10)

| Config | nDCG@10 | CI 95% (nDCG@10) | MRR@10 | Recall@10 | Judged@10 |
|--------|---------|------------------|--------|-----------|-----------|
| BM25 | 0.4543 | [0.4384, 0.4704] | 0.5907 | 0.3448 | 0.4424 |
| Dense | 0.4789 | [0.4636, 0.4960] | 0.6192 | 0.3586 | 0.4668 |
| Hybrid | 0.4911 | [0.4755, 0.5060] | 0.6323 | 0.3718 | 0.4746 |

Result tables: `results/esci_benchmark/esci/metrics.json`, `results/esci_benchmark/queries.json`

Sanity check: BM25 nDCG@10 = 0.4543 ≫ 0.048 (previous stub-embedding baseline). Hybrid > Dense > BM25, as expected.

## Next actions

1. Run `uv run pytest tests/ -q -m "not slow"` (criterion 10)
2. Add standards doc (pass/fail criteria above) to Implementation Guide §14 as definition of done
3. Commit + push (`gh auth switch` to `Praneeth-2602` first; descriptive message; no `Co-Authored-By`)

## Later (not blocking this release)

- Hinglish / code-mixed slices (Step 6 of the overhaul plan)
- `gold_test` single eval
- Kafka consumer framework, rollout gate, autoscaler
- docs/README/ADRs/demo polish; tag `v1.0`
