"""Autoscaler controller service (Implementation Guide, Section 9.7).

FastAPI app exposing the predictive autoscaler controller state:

  GET  /state        — current controller state and last decision
  POST /observe      — feed an observation, returns the scaling decision
  POST /enable / /disable — toggle the kill switch (ConfigMap key equivalent)
  GET  /healthz, /readyz, /metrics
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from bharatsearch_autoscaler import AutoscalerController, HoltLinearForecaster, Observation
from bharatsearch_common.config import get
from bharatsearch_common.logging import setup_logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, generate_latest

setup_logging(get("AUTOSCALER_LOG_LEVEL", "INFO"))

AUTOSCALER_PORT = int(get("AUTOSCALER_PORT", "8005"))

DECISIONS_TOTAL = Counter(
    "autoscaler_decisions_total", "Autoscaler scaling decisions", ["mode", "reason"]
)
CURRENT_REPLICAS = Gauge("autoscaler_current_replicas", "Current ready replicas")
FORECAST_RPS = Gauge("autoscaler_forecast_rps", "Forecasted request rate over horizon")


class _State:
    controller: AutoscalerController | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _State.controller = AutoscalerController(
        cap_per_pod=float(get("AUTOSCALER_CAP_PER_POD", "44.0")),
        target_util=float(get("AUTOSCALER_TARGET_UTIL", "0.7")),
        horizon_s=float(get("AUTOSCALER_HORIZON_S", "75.0")),
        min_replicas=int(get("AUTOSCALER_MIN_REPLICAS", "2")),
        max_replicas=int(get("AUTOSCALER_MAX_REPLICAS", "20")),
        max_step=int(get("AUTOSCALER_MAX_STEP", "4")),
        scale_down_intervals=int(get("AUTOSCALER_SCALE_DOWN_INTERVALS", "8")),
        forecaster=HoltLinearForecaster(),
    )
    yield


app = FastAPI(lifespan=lifespan, title="autoscaler")


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/readyz")
async def readyz() -> PlainTextResponse:
    if _State.controller is not None:
        return PlainTextResponse("ready")
    raise HTTPException(status_code=503, detail="not ready")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain")


@app.get("/state")
async def get_state() -> dict[str, Any]:
    if _State.controller is None:
        raise HTTPException(status_code=503, detail="not ready")
    c = _State.controller
    last = c.decisions[-1] if c.decisions else None
    return {
        "current_replicas": c.current,
        "enabled": c.enabled,
        "min_replicas": c.min_replicas,
        "max_replicas": c.max_replicas,
        "last_decision": (
            {
                "target_replicas": last.target_replicas,
                "reactive_need": last.reactive_need,
                "predictive_need": last.predictive_need,
                "forecast_rps": last.forecast_rps,
                "confident": last.confident,
                "mode": last.mode,
                "reason": last.reason,
            }
            if last is not None
            else None
        ),
        "num_decisions": len(c.decisions),
    }


@app.post("/observe")
async def observe(body: dict[str, Any]) -> dict[str, Any]:
    if _State.controller is None:
        raise HTTPException(status_code=503, detail="not ready")
    rps = float(body.get("rps", 0.0))
    p99_ms = body.get("p99_ms")
    error_ratio = body.get("error_ratio")
    queue_depth = body.get("queue_depth")
    ready_replicas = body.get("ready_replicas")
    ts = body.get("timestamp")
    obs = Observation(
        timestamp=float(ts) if ts is not None else time.time(),
        rps=rps,
        p99_ms=p99_ms,
        error_ratio=error_ratio,
        queue_depth=queue_depth,
        ready_replicas=ready_replicas,
    )
    decision = _State.controller.step(obs)
    CURRENT_REPLICAS.set(decision.target_replicas)
    FORECAST_RPS.set(decision.forecast_rps)
    DECISIONS_TOTAL.labels(mode=decision.mode, reason=decision.reason).inc()
    return {
        "target_replicas": decision.target_replicas,
        "reactive_need": decision.reactive_need,
        "predictive_need": decision.predictive_need,
        "forecast_rps": decision.forecast_rps,
        "confident": decision.confident,
        "mode": decision.mode,
        "reason": decision.reason,
        "current_replicas": _State.controller.current,
    }


@app.post("/enable")
async def enable() -> dict[str, Any]:
    if _State.controller is None:
        raise HTTPException(status_code=503, detail="not ready")
    _State.controller.enabled = True
    return {"enabled": True}


@app.post("/disable")
async def disable() -> dict[str, Any]:
    if _State.controller is None:
        raise HTTPException(status_code=503, detail="not ready")
    _State.controller.enabled = False
    return {"enabled": False}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=AUTOSCALER_PORT)
