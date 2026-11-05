"""Click simulator service (Implementation Guide, Section 7.1).

Generates synthetic search sessions with impressions and clicks, publishing
events through the retrieval event pipeline (``bharatsearch_retrieval.events``).
Uses a position-biased click model: P(click | position i) ~ relevance / (i+1).

Endpoints:
  POST /simulate — run N synthetic sessions and publish events
  GET  /healthz, /readyz, /metrics
"""

from __future__ import annotations

import random
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from bharatsearch_common.config import get
from bharatsearch_common.logging import setup_logging
from bharatsearch_contracts import (
    Intent,
    Language,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Understanding,
)
from bharatsearch_retrieval.events import (
    publish_click_event,
    publish_impression_event,
    publish_query_event,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

setup_logging(get("SIMULATOR_LOG_LEVEL", "INFO"))

SIMULATOR_PORT = int(get("SIMULATOR_PORT", "8003"))
SESSIONS_TOTAL = Counter("simulator_sessions_total", "Total simulated sessions")
CLICKS_TOTAL = Counter("simulator_clicks_total", "Total simulated clicks")
SIMULATE_LATENCY = Histogram("simulator_simulate_latency_seconds", "Simulate latency in seconds")

# Default product catalog used by the simulator when none is provided.
_DEFAULT_PRODUCTS = [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "p10",
]

# Relevance scores per (query, product) for the click model. Higher = more
# likely to click. The simulator looks up (query, product) and falls back to
# a uniform relevance when no entry exists.
_DEFAULT_RELEVANCE: dict[tuple[str, str], float] = {
    ("phone cover", "p1"): 0.9,
    ("phone cover", "p2"): 0.7,
    ("phone cover", "p3"): 0.3,
    ("shoes", "p4"): 0.9,
    ("shoes", "p5"): 0.6,
    ("water bottle", "p6"): 0.8,
    ("water bottle", "p7"): 0.5,
    ("fone cover sasta", "p1"): 0.85,
    ("fone cover sasta", "p2"): 0.65,
}


class _State:
    ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _State.ready = True
    yield


app = FastAPI(lifespan=lifespan, title="simulator")


def _understanding(query: str) -> Understanding:
    return Understanding(
        corrected_query=query,
        language=Language.en,
        intent=Intent.product,
        brand=None,
        source="fallback",
    )


def _click_probability(query: str, product_id: str, position: int) -> float:
    """Position-biased click probability.

    P(click) = relevance(query, product) / (position + 1)
    """
    relevance = _DEFAULT_RELEVANCE.get((query, product_id), 0.3)
    return relevance / (position + 1)


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


@app.post("/simulate")
async def simulate(body: dict[str, Any]) -> dict[str, Any]:
    """Run N simulated sessions.

    Body:
      - sessions: int (default 10)
      - queries: list[str] (default built-in queries)
      - products: list[str] (default built-in products)
      - k: int (default 5) — results per session
      - seed: int (default 42) — RNG seed for determinism
    """
    if not _State.ready:
        raise HTTPException(status_code=503, detail="not ready")

    n_sessions = int(body.get("sessions", 10))
    queries = body.get("queries") or list(_DEFAULT_RELEVANCE.keys())
    queries = [q[0] if isinstance(q, tuple) else q for q in queries]
    products = body.get("products") or _DEFAULT_PRODUCTS
    k = int(body.get("k", 5))
    seed = int(body.get("seed", 42))

    rng = random.Random(seed)
    sessions_run = 0
    clicks_total = 0
    start = time.time()

    for _ in range(n_sessions):
        query = rng.choice(queries)
        session_id = f"sim-{int(time.time() * 1000)}-{sessions_run}"
        request_id = f"req-{sessions_run}"

        # Shuffle products to produce a result ranking.
        ranked = list(products)
        rng.shuffle(ranked)
        top_k = ranked[:k]

        # Publish impression.
        publish_impression_event(session_id, request_id, top_k)

        # Publish query event.
        resp = SearchResponse(
            request_id=request_id,
            results=[
                SearchResult(product_id=pid, score=1.0 / (i + 1), rank=i + 1)
                for i, pid in enumerate(top_k)
            ],
            understanding=_understanding(query),
            timing_ms={},
            model_versions={
                "embedder": "stub",
                "ranker": "rrf",
                "llm": "none",
                "understanding": "fallback",
            },
        )
        req = SearchRequest(query=query, session_id=session_id, user_id=None, k=k)
        publish_query_event(req, resp)

        # Simulate clicks with position bias.
        for position, product_id in enumerate(top_k):
            if rng.random() < _click_probability(query, product_id, position):
                publish_click_event(session_id, request_id, product_id, position=position)
                clicks_total += 1

        sessions_run += 1

    SESSIONS_TOTAL.inc(sessions_run)
    CLICKS_TOTAL.inc(clicks_total)
    SIMULATE_LATENCY.observe(time.time() - start)

    return {
        "sessions": sessions_run,
        "clicks": clicks_total,
        "seed": seed,
        "elapsed_ms": (time.time() - start) * 1000,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SIMULATOR_PORT)
