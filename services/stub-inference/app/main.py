"""Stub inference server.

A minimal OpenAI-compatible model server standing in for the real LLM during
development, CI, and autoscaler experiments (Implementation Guide, Section 9.2).

Configurable via environment variables:
  SERVICE_MEAN_MS   mean service time (ms)
  SERVICE_SIGMA     log-normal shape of service time
  CONCURRENCY       requests processed in parallel per pod
  QUEUE_LIMIT       requests waiting before the pod rejects with 503
  COLD_START_S      delay before /readyz returns 200 (simulated model load)
  CPU_FRACTION      share of each service time spent burning CPU
  MODEL_NAME        model name echoed in responses
  PORT              listen port
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest

SERVICE_MEAN_MS = float(os.environ.get("SERVICE_MEAN_MS", "180"))
SERVICE_SIGMA = float(os.environ.get("SERVICE_SIGMA", "0.4"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
QUEUE_LIMIT = int(os.environ.get("QUEUE_LIMIT", "64"))
COLD_START_S = float(os.environ.get("COLD_START_S", "40"))
CPU_FRACTION = float(os.environ.get("CPU_FRACTION", "0.3"))
MODEL_NAME = os.environ.get("MODEL_NAME", "stub")
PORT = int(os.environ.get("PORT", "8000"))

REQUESTS_TOTAL = Counter("inference_requests_total", "Total inference requests")
IN_FLIGHT = Gauge("inference_in_flight", "Requests in flight")
QUEUE_DEPTH = Gauge("inference_queue_depth", "Requests waiting in queue")
LATENCY = Histogram("inference_latency_seconds", "Inference latency in seconds")
REJECTED_TOTAL = Counter("inference_rejected_total", "Rejected requests")


class _State:
    semaphore: asyncio.Semaphore | None = None
    ready = False
    in_flight = 0


def _burn_cpu(seconds: float) -> None:
    end = time.time() + seconds
    x = 0.0
    while time.time() < end:
        x += 1.0


async def _simulate_service() -> None:
    draw = random.lognormvariate(0.0, SERVICE_SIGMA) * (SERVICE_MEAN_MS / 1000.0)
    cpu_s = draw * CPU_FRACTION
    sleep_s = draw * (1.0 - CPU_FRACTION)
    if cpu_s > 0:
        await asyncio.to_thread(_burn_cpu, cpu_s)
    if sleep_s > 0:
        await asyncio.sleep(sleep_s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _State.semaphore = asyncio.Semaphore(CONCURRENCY)
    await asyncio.sleep(COLD_START_S)
    _State.ready = True
    yield


app = FastAPI(lifespan=lifespan, title="stub-inference")


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/readyz")
async def readyz() -> PlainTextResponse:
    if _State.ready:
        return PlainTextResponse("ready")
    raise HTTPException(status_code=503, detail="cold start in progress")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain")


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    if _State.semaphore is None:
        raise HTTPException(status_code=503, detail="not ready")
    if _State.in_flight >= QUEUE_LIMIT:
        REJECTED_TOTAL.inc()
        raise HTTPException(status_code=503, detail="queue full")
    _State.in_flight += 1
    IN_FLIGHT.set(_State.in_flight)
    QUEUE_DEPTH.set(_State.in_flight)
    REQUESTS_TOTAL.inc()
    start = time.time()
    try:
        async with _State.semaphore:
            await _simulate_service()
    finally:
        _State.in_flight -= 1
        IN_FLIGHT.set(_State.in_flight)
        QUEUE_DEPTH.set(_State.in_flight)
    LATENCY.observe(time.time() - start)
    messages = body.get("messages", [])
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break
    return {
        "id": "stub-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": user_text or "canned response"},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
