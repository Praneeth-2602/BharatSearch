"""Embedding service.

Deterministic/stub embedding by default (offline, CI-friendly); optional
sentence-transformers backend (multilingual-e5-small) via EMBEDDING_BACKEND.
Exposes POST /embed plus /healthz, /readyz, /metrics (Implementation Guide,
Section 3.2 / Section 0 rule 1).
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from contextlib import asynccontextmanager

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
from bharatsearch_contracts import EmbedRequest, EmbedResponse

setup_logging(get("EMBEDDING_LOG_LEVEL", "INFO"))
_provider = setup_tracing("embedding")
_tracer = _provider.get_tracer("embedding")

EMBEDDING_BACKEND = get("EMBEDDING_BACKEND", "stub")
EMBEDDING_MODEL = get("EMBEDDING_MODEL", "stub")
EMBEDDING_DIM = int(get("EMBEDDING_DIM", "384"))
EMBEDDING_CACHE_SIZE = int(get("EMBEDDING_CACHE_SIZE", "1000"))
EMBEDDING_BATCH_SIZE = int(get("EMBEDDING_BATCH_SIZE", "32"))
PORT = int(get("EMBEDDING_PORT", "8001"))

EMBED_REQUESTS_TOTAL = Counter("embedding_requests_total", "Total embed requests")
EMBED_CACHE_HITS = Counter("embedding_cache_hits_total", "Embedding cache hits")
EMBED_CACHE_MISSES = Counter("embedding_cache_misses_total", "Embedding cache misses")
EMBED_BATCH_SIZE_HIST = Histogram("embedding_batch_size", "Embedding request batch size")
EMBED_LATENCY = Histogram("embedding_latency_seconds", "Embedding latency in seconds")


class _LRUCache:
    """Tiny LRU cache for query embeddings."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self._cache: dict[str, list[float]] = {}
        self._order: list[str] = []

    def get(self, key: str) -> list[float] | None:
        if self.capacity <= 0 or key not in self._cache:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._cache[key]

    def put(self, key: str, value: list[float]) -> None:
        if self.capacity <= 0:
            return
        if key in self._cache:
            self._order.remove(key)
        self._cache[key] = value
        self._order.append(key)
        while len(self._order) > self.capacity:
            oldest = self._order.pop(0)
            del self._cache[oldest]


class _State:
    model: object = None
    cache: _LRUCache | None = None
    backend: str = ""
    ready: bool = False


def _cache_key(input_type: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{input_type}:{digest}"


def _prefix(input_type: str, text: str) -> str:
    """e5-style prefix: ` query: ` / ` passage: `."""
    if input_type == "passage":
        return f"passage: {text}"
    return f"query: {text}"


def _stable_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-random unit vector (stub backend)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:4], "little"))
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed a batch of texts (one model forward pass worth of work)."""
    if _State.backend == "sentence-transformers" and _State.model is not None:
        prefixed = [_prefix(input_type, t) for t in texts]
        return _State.model.encode(prefixed, normalize_embeddings=True).tolist()
    return [_stable_vector(_prefix(input_type, t), EMBEDDING_DIM) for t in texts]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _State.cache = _LRUCache(EMBEDDING_CACHE_SIZE)
    _State.backend = EMBEDDING_BACKEND
    if EMBEDDING_BACKEND == "sentence-transformers":
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers backend requires the sentence-transformers package"
            )
        _State.model = SentenceTransformer(EMBEDDING_MODEL)
    _State.ready = True
    yield


app = FastAPI(lifespan=lifespan, title="embedding")


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


@app.post("/embed")
async def embed(req: EmbedRequest) -> EmbedResponse:
    if not _State.ready:
        raise HTTPException(status_code=503, detail="not ready")
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    EMBED_REQUESTS_TOTAL.inc()
    start = time.time()

    embeddings: list[list[float] | None] = [None] * len(req.texts)
    uncached: list[tuple[int, str]] = []

    for i, text in enumerate(req.texts):
        key = _cache_key(req.input_type, text)
        cached = _State.cache.get(key) if _State.cache else None
        if cached is not None:
            EMBED_CACHE_HITS.inc()
            embeddings[i] = cached
        else:
            EMBED_CACHE_MISSES.inc()
            uncached.append((i, text))

    if uncached:
        batch_texts = [t for _, t in uncached]
        EMBED_BATCH_SIZE_HIST.observe(len(batch_texts))
        vectors = _embed_texts(batch_texts, req.input_type)
        for (idx, text), vec in zip(uncached, vectors, strict=True):
            embeddings[idx] = vec
            key = _cache_key(req.input_type, text)
            if _State.cache:
                _State.cache.put(key, vec)

    final: list[list[float]] = [e for e in embeddings if e is not None]  # type: ignore[assignment]
    EMBED_LATENCY.observe(time.time() - start)
    return EmbedResponse(
        embeddings=final,
        model=EMBEDDING_MODEL,
        usage={"tokens": sum(len(t) for t in req.texts), "batch_size": len(req.texts)},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
