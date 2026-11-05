"""Retrieval API service.

Combines OpenSearch BM25 + Qdrant dense retrieval with RRF fusion.
Exposes POST /v1/search plus /healthz, /readyz, /metrics
(Implementation Guide, Section 3.1-3.4 / Section 0 rule 1).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

from bharatsearch_common.config import get
from bharatsearch_common.logging import setup_logging
from bharatsearch_common.tracing import setup_tracing
from bharatsearch_contracts import (
    EmbedRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

from bharatsearch_retrieval.events import (
    ClickRequest,
    ImpressionRequest,
    publish_click_event,
    publish_impression_event,
    publish_query_event,
)
from bharatsearch_retrieval.fusion import rrf
from bharatsearch_retrieval.opensearch import INDEX_NAME, bm25_query
from bharatsearch_retrieval.qdrant import COLLECTION_NAME
from bharatsearch_retrieval.understanding import understand

setup_logging(get("RETRIEVAL_LOG_LEVEL", "INFO"))
_provider = setup_tracing("retrieval")
_tracer = _provider.get_tracer("retrieval")

OPENSEARCH_URL = get("OPENSEARCH_URL", "http://localhost:9200")
QDRANT_URL = get("QDRANT_URL", "http://localhost:6333")
EMBEDDING_URL = get("EMBEDDING_URL", "http://localhost:8001")
RRF_K = int(get("RRF_K", "60"))
DEFAULT_K = int(get("DEFAULT_K", "20"))
PORT = int(get("RETRIEVAL_PORT", "8002"))

SEARCH_REQUESTS_TOTAL = Counter("retrieval_search_requests_total", "Total search requests")
SEARCH_LATENCY = Histogram("retrieval_search_latency_seconds", "Search latency in seconds")


class _State:
    http: httpx.AsyncClient | None = None
    ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _State.http = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    _State.ready = True
    yield
    await _State.http.aclose()


app = FastAPI(lifespan=lifespan, title="retrieval")


async def _bm25_search(query_text: str, k: int) -> list[tuple[str, float]]:
    """Run BM25 against OpenSearch; return [(product_id, score)]."""
    body = bm25_query(query_text, k=k)
    resp = await _State.http.post(f"{OPENSEARCH_URL}/{INDEX_NAME}/_search", json=body)  # type: ignore[union-attr]
    resp.raise_for_status()
    hits = resp.json()["hits"]["hits"]
    return [(h["_source"]["product_id"], float(h["_score"])) for h in hits]


async def _embed_query(query_text: str) -> list[float]:
    """Embed a single query via the embedding service."""
    resp = await _State.http.post(  # type: ignore[union-attr]
        f"{EMBEDDING_URL}/embed",
        json=EmbedRequest(texts=[query_text], input_type="query").model_dump(),
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


async def _dense_search(query_text: str, k: int) -> list[tuple[str, float]]:
    """Run dense search against Qdrant; return [(product_id, score)]."""
    vector = await _embed_query(query_text)
    resp = await _State.http.post(  # type: ignore[union-attr]
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        json={
            "vector": vector,
            "limit": k,
            "with_payload": True,
        },
    )
    resp.raise_for_status()
    points = resp.json()["result"]
    return [(p["payload"]["product_id"], float(p["score"])) for p in points]


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/readyz")
async def readyz() -> PlainTextResponse:
    if _State.ready:
        return PlainTextResponse("ready")
    raise HTTPException(status_code=503, detail="not ready")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain")


@app.post("/v1/search")
async def search(req: SearchRequest) -> SearchResponse:
    if not _State.ready:
        raise HTTPException(status_code=503, detail="not ready")
    k = req.k or DEFAULT_K

    start = time.time()
    timing: dict[str, float] = {}

    with _tracer.start_as_current_span("retrieval.search"):
        # BM25 stage
        with _tracer.start_as_current_span("retrieval.bm25"):
            t0 = time.time()
            bm25_results = await _bm25_search(req.query, k)
            timing["bm25_ms"] = (time.time() - t0) * 1000

        # Dense stage
        with _tracer.start_as_current_span("retrieval.dense"):
            t0 = time.time()
            dense_results = await _dense_search(req.query, k)
            timing["dense_ms"] = (time.time() - t0) * 1000

        # Fuse stage
        with _tracer.start_as_current_span("retrieval.fuse"):
            t0 = time.time()
            fused = rrf(
                [[pid for pid, _ in bm25_results], [pid for pid, _ in dense_results]],
                k=RRF_K,
                top=k,
            )
            timing["fuse_ms"] = (time.time() - t0) * 1000

    results = [
        SearchResult(product_id=pid, score=score, rank=i + 1)
        for i, (pid, score) in enumerate(fused)
    ]

    timing["total_ms"] = (time.time() - start) * 1000

    SEARCH_REQUESTS_TOTAL.inc()
    SEARCH_LATENCY.observe(time.time() - start)

    resp = SearchResponse(
        request_id=str(uuid.uuid4()),
        results=results,
        understanding=understand(req.query),
        timing_ms=timing,
        model_versions={
            "embedder": "stub",
            "ranker": "rrf",
            "llm": "none",
            "understanding": "rule-based",
        },
    )
    publish_query_event(req, resp)
    return resp


@app.post("/v1/impression")
async def impression(req: ImpressionRequest) -> PlainTextResponse:
    publish_impression_event(req.session_id, req.request_id, req.product_ids)
    return PlainTextResponse("ok")


@app.post("/v1/click")
async def click(req: ClickRequest) -> PlainTextResponse:
    publish_click_event(req.session_id, req.request_id, req.product_id, req.position)
    return PlainTextResponse("ok")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
