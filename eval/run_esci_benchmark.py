"""ESCI benchmark: retrieval service vs qrels, write metrics to results/.

Loads ESCI queries that have qrels judgments, filters to clean English
queries (BM25's working language), samples a benchmark set, calls the
retrieval service (hybrid BM25+dense RRF), OpenSearch directly
(BM25-only), and Qdrant directly with a real dense embedding (dense-only)
for each query, and evaluates with eval.harness.
Usage: uv run python -m eval.run_esci_benchmark
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
from pathlib import Path

import httpx
import pandas as pd
from bharatsearch_retrieval.opensearch import INDEX_NAME, bm25_query
from bharatsearch_retrieval.understanding import Language, understand

from eval.harness import bootstrap_ci, run, write_metrics

QRELS_PATH = Path("results/esci_qrels.json")
QUERIES_GLOB = "data/esci/queries/*.parquet"
OUT_DIR = Path("results/esci_benchmark")
OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "products"
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8001")
SERVICE_URL = "http://localhost:8002/v1/search"
SAMPLE_SIZE = 2000
RANDOM_SEED = 42
K = 10
CONCURRENCY = 8


def load_benchmark_queries(sample_size: int = SAMPLE_SIZE) -> list[dict]:
    """Return [{query_id, query}] for clean English ESCI queries that have qrels.

    Reads the queries parquet files, keeps queries present in qrels, filters to
    clean English (language == en per the rule-based understander), then samples
    deterministically. Sampling from all English-with-judgment queries keeps the
    benchmark size stable even as qrels grows.
    """
    qrels = json.loads(QRELS_PATH.read_text())
    qrel_ids = set(qrels.keys())

    frames = [pd.read_parquet(f) for f in sorted(glob.glob(QUERIES_GLOB))]
    queries = pd.concat(frames, ignore_index=True)
    queries = queries[queries["query_id"].astype(str).isin(qrel_ids)]
    queries = queries.drop_duplicates(subset="query_id")

    english = [
        {"query_id": str(r.query_id), "query": r.query}
        for r in queries.itertuples(index=False)
        if understand(r.query).language == Language.en
    ]
    print(f"[esci_benchmark] {len(english)} clean English queries with qrels")

    if sample_size and len(english) > sample_size:
        sampled = pd.DataFrame(english).sample(n=sample_size, random_state=RANDOM_SEED)
        english = [
            {"query_id": str(r.query_id), "query": r.query} for r in sampled.itertuples(index=False)
        ]
    return english


async def fetch_ranked(client: httpx.AsyncClient, qid: str, query: str, k: int = K) -> list[str]:
    """Call the retrieval service (hybrid BM25+dense); return ranked product ids."""
    resp = await client.post(
        SERVICE_URL,
        json={"query": query, "session_id": "esci-bench", "k": k},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [r["product_id"] for r in results]


async def fetch_bm25(client: httpx.AsyncClient, qid: str, query: str, k: int = K) -> list[str]:
    """Run BM25 against OpenSearch directly; return ranked product ids.

    No Qdrant, no RRF: the BM25-only baseline (Step 3).
    """
    resp = await client.post(
        f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
        json=bm25_query(query, k=k),
    )
    resp.raise_for_status()
    hits = resp.json()["hits"]["hits"]
    return [h["_source"]["product_id"] for h in hits]


async def fetch_dense(client: httpx.AsyncClient, qid: str, query: str, k: int = K) -> list[str]:
    """Dense-only: embed the query with the real e5 model, then search Qdrant.

    No BM25, no RRF: returns the top-k by dense cosine similarity (Step 4).
    """
    resp = await client.post(
        f"{EMBEDDING_URL}/embed",
        json={"texts": [query], "input_type": "query"},
    )
    resp.raise_for_status()
    vector = resp.json()["embeddings"][0]
    resp = await client.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        json={"vector": vector, "limit": k, "with_payload": True},
    )
    resp.raise_for_status()
    points = resp.json()["result"]
    return [p["payload"]["product_id"] for p in points]


async def main() -> None:
    queries = load_benchmark_queries()
    if not queries:
        raise SystemExit(
            "[esci_benchmark] no clean English benchmark queries; check qrels and indexing"
        )

    print(f"[esci_benchmark] {len(queries)} benchmark queries")

    qrels = json.loads(QRELS_PATH.read_text())
    bench_qrels = {q["query_id"]: qrels[q["query_id"]] for q in queries}

    runs: dict[str, dict[str, list[str]]] = {"bm25": {}, "dense": {}, "hybrid": {}}
    timeout = httpx.Timeout(30.0)
    limits = httpx.Limits(max_connections=CONCURRENCY)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(q: dict) -> None:
        async with sem:
            qid = q["query_id"]
            runs["bm25"][qid] = await fetch_bm25(client, qid, q["query"], k=K)
            runs["dense"][qid] = await fetch_dense(client, qid, q["query"], k=K)
            runs["hybrid"][qid] = await fetch_ranked(client, qid, q["query"], k=K)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(*(one(q) for q in queries))

    report = run(bench_qrels, runs, k=K)
    ci = bootstrap_ci(bench_qrels, runs, k=K)
    for name in report:
        report[name]["ci_95"] = {}
        for metric, (lo, hi) in ci[name].items():
            report[name]["ci_95"][metric] = [lo, hi] if lo is not None else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "queries.json").write_text(json.dumps(queries, indent=2))
    write_metrics(report, "esci", out_dir=str(OUT_DIR))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
