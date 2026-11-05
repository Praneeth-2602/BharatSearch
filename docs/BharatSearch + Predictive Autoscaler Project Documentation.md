# BharatSearch + Predictive Autoscaler: Project Documentation

Draft v1 · November 2026 · Resume-grade systems + AI project

## 1. Overview

**BharatSearch** is a Hinglish- and vernacular-aware commerce search engine with a real-time learning loop. **The Predictive Autoscaler** keeps it alive under bursty, match-day style traffic by forecasting load and pre-scaling instead of reacting late. The two are built as one system: BharatSearch is the workload, the autoscaler is the control plane that scales it.

**One-line pitch:** *A search system for how India actually types (Hinglish, typos, regional languages), with an event-driven feedback loop, safe model rollouts, and a forecast-driven autoscaler that survives IPL-style traffic spikes.*

**Delivery model:** local-first. The whole system runs on a developer machine with no cloud account, and the GitHub repo carries a README with one-command setup, a demo script, and reproducible benchmark scripts. A cloud deployment is optional and not part of the core deliverable.

### Goals

- Measurably improve search relevance on code-mixed and misspelled queries versus a plain BM25 baseline.
- Close the loop: clicks and impressions flow through Kafka, update online features, and retrain the ranker, gated by an automated regression check.
- Show that a predictive autoscaler reduces SLO violations (or cost) versus standard reactive autoscaling under realistic bursty load.
- Produce reproducible numbers, dashboards, and a write-up that stand up in interviews.

### Non-goals

- Real production traffic, real users, or an always-on public deployment. All traffic is simulated and everything runs locally.
- Training a foundation LLM. Models are pretrained and self-hosted.
- Beating commercial search. The goal is a rigorous, honest engineering artifact.

## 2. Problem statements and inspirations

Each component traces back to a problem a company has publicly written about. These come from public engineering write-ups and secondary summaries; read the originals before quoting any number in an interview.

| Problem | Company | What it inspires here |
| --- | --- | --- |
| Misspelled, vernacular, and long-tail queries hurt search | Zepto, Meesho, Swiggy | LLM + RAG query correction over catalog candidates |
| Consumer head-of-line blocking, poison pills, hidden rebalances | Uber | Resilient Kafka consumers with a dead-letter queue and lag monitoring |
| Safe migration of live systems | Abnormal AI | Shadow traffic, percentage rollouts, automated parity gates |
| ML-driven routing that must work in real time | Razorpay | Online features and fast model updates (learning loop) |
| Reactive autoscaling fails at extreme spikes | JioHotstar | Pre-scaling ladders and traffic forecasting |
| Unused or unowned Kafka topics | Netflix | Topic hygiene checks (optional extension) |

## 3. Scope and phasing

**Core (must ship):** baseline search, evaluation harness, query understanding, Kafka event loop, ranker training, rollout gate, load generator, reactive vs predictive autoscaling comparison, one-command local setup, demo script.

**Stretch (cut in this order if time runs short):**

1. Cold-start handling for anonymous users.
2. Automated ranker retraining (keep a static ranker instead).
3. Learned forecaster (keep rule-based pre-scale ladders instead).

**Protect at all costs:** the evaluation harness and the rollout gate. They are what make every claim credible.

## 4. System architecture

### 4.1 Request path

1. **API gateway (FastAPI)** receives a search request and assigns a session ID.
2. **Query understanding service**
   - Unicode NFC normalization, script and language detection, romanized-Indic detection.
   - Candidate retrieval from the catalog (brand, product, category names) by fuzzy and n-gram match.
   - A quantized 7-8B model served with vLLM receives the raw query plus candidates and returns structured JSON: corrected query, detected intent, optional filters, language tag.
   - Strict timeout and a fallback to the normalized raw query so the LLM never becomes a single point of failure.
3. **Retrieval:** OpenSearch BM25 plus Qdrant dense vectors, fused with reciprocal rank fusion (RRF).
4. **Ranking:** LightGBM learning-to-rank over retrieval scores, product features, and online session or popularity features from Redis.
5. **Response** returns ranked results and emits an impression event.

### 4.2 Event backbone

| Topic | Producer | Consumers |
| --- | --- | --- |
| `search.queries` | API gateway | Analytics, query-log store |
| `search.impressions` | API gateway | Feature updater, training-data builder |
| `search.clicks` | Simulator or client | Feature updater, training-data builder |
| `search.shadow_diffs` | Shadow evaluator | Rollout controller |
| `search.dlq` | All consumers | Replay tool, alerts |

Consumers use a resilient pattern: per-partition ordered processing with out-of-order commits where safe, bounded retries with delay, a dead-letter queue for poison messages, and lag and rebalance metrics exported to Prometheus.

### 4.3 Learning loop

1. Stream processor aggregates click-through features per product, query cluster, and session into **Redis** (online) and **Parquet** (offline).
2. A trainer job builds a training set from logged impressions and clicks and trains a candidate LightGBM ranker.
3. The candidate is registered in a simple model registry (MLflow or a versioned folder with a manifest).
4. The **rollout controller** promotes it through shadow, 1%, 10%, 50%, 100%, checking gates at each step (Section 7).

### 4.4 Control plane (autoscaler)

Prometheus collects request rate, p50/p95/p99 latency, error rate, Kafka lag, vLLM queue depth, and GPU or CPU utilization. The forecaster predicts load for the next few minutes. A controller converts the forecast into replica targets through a pre-scaling ladder and applies them through KEDA or the Kubernetes API (details in Section 6).

## 5. Search component specifications

### 5.1 Data

- **Catalog and relevance labels:** Amazon ESCI (Shopping Queries) or a public Flipkart/Amazon product dataset. ESCI provides query-product relevance judgments (Exact, Substitute, Complement, Irrelevant), which become graded labels.
- **Hinglish and vernacular queries (synthetic):** transform English queries while keeping their relevance judgments, using romanized Hindi transliteration, code-mixing, vowel dropping, phonetic misspellings, and keyboard-adjacent typos. Hold out a **gold set of 300-500 hand-checked queries** for final reporting.
- **Limitation (state it in the write-up):** synthetic queries approximate real user behavior but do not replace it.

### 5.2 Query understanding

- Prompt returns JSON only; validate with a schema and fall back on failure.
- Cache normalized-query to corrected-query results (most traffic repeats).
- Measure correction accuracy, latency, and the NDCG delta with and without the step.

### 5.3 Retrieval and ranking

- Dense model candidates: BGE-M3 or multilingual-e5. Pick one using recall@k on your gold set.
- Fusion: RRF with tuned k; compare against weighted score fusion.
- LightGBM LambdaRank with features such as BM25 score, dense score, cluster match (ESCI has no category or price data), historical CTR, session popularity, and query language.

### 5.4 Cold start (stretch)

For anonymous sessions, fall back to category and geo popularity priors and use mild exploration (for example a contextual bandit over a few ranking variants) while features accumulate. Measure first-session CTR against the logged-in-style ranker.

### 5.5 Tokenizer gate (inditok)

The custom Indic tokenizer is **not assumed** to be useful. It enters the system only where you control tokenization:

- As an input tokenizer for query-side models. It is not a BM25 analyzer unless the catalog gains Indic-language fields, because the ESCI catalog is English.
- In spell-correction candidate generation.
- In any small model trained from scratch.

**Go/no-go test:** on shared languages and the same test data (FLORES-200 devtest and IN22-Gen), the tokenizer must land within roughly 10-15% of Sarvam-1's fertility at a similar vocabulary size and beat the baseline tokenizers on the query-side experiment in the Implementation Guide (Section 5.5). Otherwise use an existing tokenizer in the product and keep inditok as a separate research artifact. Pretrained embedding models and LLMs keep their own tokenizers; swapping them requires retraining.

## 6. Predictive autoscaler specification

### 6.1 Why reactive scaling fails here

The LLM and embedding services have slow cold starts (model load plus warmup). A reactive scaler sees the spike after latency has already degraded, then adds capacity that takes minutes to become useful. Pre-scaling ahead of forecast load hides that startup time.

**Local mode:** the comparison runs against a stub inference service with a configurable cold-start delay, latency curve, and per-pod capacity, so it works on a laptop and in CI without a GPU. Validate the stub's parameters once against the real model and state them next to the results.

### 6.2 Load profiles (simulated)

| Profile | Shape | Purpose |
| --- | --- | --- |
| Steady | Constant RPS with noise | Sanity baseline |
| Diurnal | Smooth daily curve | Forecastable seasonality |
| Match-day | Ramp before start, spikes at key moments, sharp drop after | Main scenario |
| Flash crowd | Unannounced step increase | Forecaster failure mode; tests the reactive fallback |

Generate load with k6 (or Locust) driven by replayable curve files so every strategy sees identical traffic.

### 6.3 Strategies compared

1. **HPA on CPU** (standard baseline).
2. **HPA or KEDA on request rate or queue depth** (reactive, better metric).
3. **Rule-based pre-scale ladders:** scheduled replica floors around known events plus ladders tied to predicted RPS.
4. **Predictive scaler:** forecaster plus ladders, with automatic fallback to strategy 2 when forecast confidence is low or error is high.

Forecaster candidates, in order of complexity: seasonal-naive, EWMA with trend, gradient-boosted regression on lag features, then optionally a small sequence model. Do not skip the simple baselines; a complex model that only matches seasonal-naive is a finding worth reporting.

### 6.4 Safety rules

- Minimum replica floors per service and a maximum budget cap.
- Stepwise scale-down with cooldown (never drop capacity sharply right after a spike).
- Confidence-based fallback to reactive scaling.
- A manual override and a kill switch.

### 6.5 Metrics

| Metric | Definition |
| --- | --- |
| SLO violation minutes | Minutes where p99 exceeds the target or errors exceed the threshold |
| Pod-minutes | Sum of running pods over time (cost proxy; weight model-serving pods higher) |
| Over-provisioning ratio | Provisioned capacity divided by needed capacity |
| Time-to-capacity | Delay from load rise to adequate capacity |
| Scale events (thrash) | Number of scale-up and scale-down actions |

Report each strategy on each load profile in one table, with variance across at least 3 runs.

## 7. Evaluation plan

### 7.1 Search quality

Run the same queries through each configuration, reported on the gold set and the larger synthetic set:

| Configuration | What it shows |
| --- | --- |
| Raw BM25 (whitespace analyzer) | Baseline |
| BM25 + normalization | Value of cleanup alone |
| + LLM correction | Value of query understanding |
| + dense retrieval (hybrid, RRF) | Value of semantic matching |
| + LightGBM ranker | Value of learning-to-rank |
| + inditok-based query model (if it passes the gate) | Value of the custom tokenizer |

Metrics: NDCG@10, MRR, recall@50, correction accuracy, and per-language and per-query-type breakdowns (clean, typo, romanized, code-mixed).

### 7.2 System quality

- p50/p95/p99 latency per stage and end to end.
- Throughput at the SLO and cost per 1,000 queries.
- Consumer lag under load and DLQ rate.
- Rollout gate results over several candidate-model cycles.

### 7.3 Rollout gate rules

A candidate ranker advances a stage only if NDCG@10 is not worse than the incumbent beyond a set tolerance, p99 latency stays within budget, and error rate stays flat. Any failure triggers automatic rollback and an alert. Record at least one deliberately bad candidate to prove the gate works.

## 8. Local-first infrastructure and setup

- **Primary target:** the whole system runs on one machine. Core services run with Docker Compose; the autoscaler experiments run on a local kind (or k3d) Kubernetes cluster.
- **One-command workflow:** `make setup` (dependencies, models, sample data), `make up` and `make down`, `make eval`, `make demo`, `make autoscale-bench`.
- **Observability:** OpenTelemetry traces, Prometheus metrics, and pre-provisioned Grafana dashboards, bundled in the Compose stack.
- **CI:** lint, unit tests, a small end-to-end smoke test on a tiny data slice, and an eval job that fails the build on metric regressions. CI uses the stub model, so it needs no GPU.
- **Optional cloud:** Terraform modules live under `deploy/optional-cloud` for anyone who wants to deploy. They are not part of the core deliverable.

### Hardware profiles

The demo has to run on a typical laptop, so every heavy component has a lighter fallback chosen by one config setting. Model serving sits behind an OpenAI-compatible endpoint, which makes Ollama, llama.cpp, vLLM, and the stub interchangeable.

1. **Lite (no GPU):** the stub or a small quantized model on CPU, a reduced catalog slice, single-node Kafka, and a small CPU embedding model. Enough for the full pipeline and the autoscaler comparison.
2. **Standard (consumer GPU or Apple Silicon):** a 7-8B quantized model through Ollama or llama.cpp, and a larger catalog slice. This is the demo profile.
3. **Full (NVIDIA GPU):** vLLM, the full catalog, and the final latency and throughput benchmarks.

### Setup and demo requirements

- **README:** prerequisites (Docker, Python version, make, kind), the `make setup` flow, how to pick a profile, disk and RAM needs, and a troubleshooting section. Measure the minimum specs for each profile and document them instead of guessing.
- **Reproducibility:** pinned versions, lockfiles, and seeded data generation so the benchmark tables regenerate with `make eval`.
- **Demo script (about 10 minutes):** (1) raw BM25 against BharatSearch on typo and Hinglish queries, side by side; (2) the click simulator driving live feature updates in Grafana; (3) a deliberately bad ranker rejected by the rollout gate; (4) the match-day load on kind, with reactive HPA against the predictive scaler, showing replica counts and p99.
- **Demo fallback:** a recorded video and cached results in case the live run fails.
- **Fresh-machine test:** clone onto a clean machine or VM and follow only the README before calling the project done.

## 9. Repository layout

```
bharatsearch/
  README.md                quickstart, hardware profiles, troubleshooting
  Makefile                 setup, up, down, eval, demo, autoscale-bench
  docker-compose.yml       core stack (profiles: lite, standard, full)
  .env.example             profile and model-server settings
  docs/                    architecture, setup guide, ADRs, benchmark reports, demo script
  data/                    dataset download + synthetic query generation scripts
  services/
    api-gateway/           FastAPI entrypoint
    query-understanding/   normalization, candidates, LLM client (OpenAI-compatible)
    retrieval/             OpenSearch + Qdrant clients, RRF fusion
    ranker/                LightGBM inference + feature fetch
    stream-processor/      Kafka consumers, DLQ, feature updater
    trainer/               dataset builder, training, registry push
    rollout-controller/    shadow evaluator, gates, promotion/rollback
    stub-inference/        fake model server: configurable cold start and latency
  autoscaler/
    forecaster/            baselines + learned models
    controller/            ladders, KEDA/K8s actuation, safety rules
    loadgen/               k6/Locust scripts + load-curve files
  eval/                    search eval harness, autoscaler comparison
  deploy/
    kind/                  local cluster config, manifests, KEDA objects
    observability/         Prometheus, Grafana dashboards, OTel collector
    optional-cloud/        Terraform (not part of the core deliverable)
  tokenizer-bench/         inditok vs baselines harness
  scripts/                 setup helpers, fresh-machine check
```

## 10. Milestones (about 10 weeks)

| Week | Deliverable | Exit criterion |
| --- | --- | --- |
| 1 | Repo, dataset, synthetic Hinglish queries, tokenizer benchmark harness, Docker Compose + Makefile skeleton | Gold set exists; tokenizer go/no-go decided |
| 2-3 | Baseline search with BM25 and dense retrieval, eval harness | Reproducible NDCG@10 table for baselines |
| 4-5 | Query understanding with vLLM and RAG correction | Measured NDCG and accuracy delta, latency budget met |
| 6-7 | Kafka loop, click simulator, LightGBM ranker, shadow rollout and gate | Bad candidate auto-rejected; good candidate promoted |
| 8-9 | Load generator, four scaling strategies compared | Results table across all load profiles, 3+ runs each |
| 10 | Dashboards, write-up, demo recording, fresh-machine setup test, resume bullets | Everything reproducible from the README |

If you fall behind, follow the cut order in Section 3 rather than shrinking evaluation.

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Scope: two flagship projects at once | Treat them as one system; protect eval and gate; cut stretch items first |
| Laptop resource limits (RAM, GPU) | Lite, standard, and full profiles; stub or quantized model; documented minimum specs |
| Synthetic queries unrepresentative | Hand-check a gold set; state the limitation openly |
| LLM adds latency or fails | Timeout, cache, and fallback to normalized raw query |
| Forecaster no better than seasonal-naive | Report it honestly; that is still a valid, publishable finding |
| Custom tokenizer underperforms | Go/no-go gate; keep it as a separate research artifact |
| Results not reproducible | Pin versions, seed everything, store configs and load curves in the repo |

## 12. Resume bullet templates

Fill in measured numbers only after the final runs; never use targets.

- Built a Hinglish- and vernacular-aware product search system (OpenSearch + Qdrant + LightGBM LTR) improving NDCG@10 by **\[X\]%** over a BM25 baseline on a **\[N\]**-query code-mixed evaluation set.
- Implemented LLM + RAG query correction on a self-hosted quantized model (vLLM), raising correction accuracy from **\[A\]%** to **\[B\]%** at **\[L\] ms** p95 added latency.
- Designed a Kafka-based feedback loop with dead-letter queues and consumer-lag monitoring, plus shadow-traffic rollouts with automated NDCG and latency gates that rejected **\[K\]** regressing models.
- Built a forecast-driven pre-scaling controller for Kubernetes that cut SLO-violation minutes by **\[X\]%** (or pod-minutes by **\[Y\]%**) versus CPU-based HPA under a simulated match-day load.
- Packaged the full stack for one-command local setup with Docker Compose and kind; instrumented with OpenTelemetry, Prometheus, and Grafana.
