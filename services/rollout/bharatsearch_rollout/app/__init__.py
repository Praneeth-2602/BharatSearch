"""Rollout controller service (Implementation Guide, Section 7.4).

State machine for controlled model rollout:

  shadow → canary → full

Each state gates the next based on metrics (CTR, NDCG, latency, error rate).
Traffic splitting is controlled per state:

  shadow:   0% traffic to new model, 100% to baseline, observes only
  canary:   N% traffic to new model, rest to baseline
  full:     100% traffic to new model

Endpoints:
  GET  /state        — current rollout state
  POST /transition   — attempt state transition (gate-checked)
  POST /traffic      — set canary traffic percentage
  GET  /healthz, /readyz, /metrics
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from bharatsearch_common.config import get
from bharatsearch_common.logging import setup_logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, generate_latest

setup_logging(get("ROLLOUT_LOG_LEVEL", "INFO"))

ROLLOUT_PORT = int(get("ROLLOUT_PORT", "8004"))

STATE_CHANGES_TOTAL = Counter(
    "rollout_state_changes_total", "Rollout state transitions", ["from_state", "to_state"]
)
GATE_EVALUATIONS_TOTAL = Counter(
    "rollout_gate_evaluations_total", "Gate evaluations passed", ["gate"]
)
CURRENT_STATE = Gauge("rollout_current_state", "Current rollout state (0=shadow,1=canary,2=full)")
CANARY_TRAFFIC = Gauge("rollout_canary_traffic_pct", "Canary traffic percentage")

MAX_ERROR_RATE = 0.05
MAX_TRAFFIC_PCT = 100
CTR_LIFT_MARGIN = 1.05


class RolloutState:
    SHADOW = "shadow"
    CANARY = "canary"
    FULL = "full"


_STATES = [RolloutState.SHADOW, RolloutState.CANARY, RolloutState.FULL]


class RolloutController:
    """State machine for controlled rollout with metric-based gates.

    Transitions require passing the gate for the target state. The gate
    function receives the current metrics dict and returns (passed, reason).
    """

    def __init__(self) -> None:
        self.state = RolloutState.SHADOW
        self.canary_traffic_pct = 10
        self._metrics: dict[str, float] = {}
        self._gates: dict[str, callable] = {
            RolloutState.CANARY: self._gate_canary,
            RolloutState.FULL: self._gate_full,
        }

    def _gate_canary(self, metrics: dict[str, float]) -> tuple[bool, str]:
        """Gate for shadow → canary: baseline must be healthy."""
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > MAX_ERROR_RATE:
            return False, f"error_rate {error_rate:.2%} > {MAX_ERROR_RATE:.0%}"
        return True, "ok"

    def _gate_full(self, metrics: dict[str, float]) -> tuple[bool, str]:
        """Gate for canary → full: canary CTR must beat baseline by margin."""
        canary_ctr = metrics.get("canary_ctr", 0.0)
        baseline_ctr = metrics.get("baseline_ctr", 0.0)
        if baseline_ctr > 0 and canary_ctr < baseline_ctr * CTR_LIFT_MARGIN:
            return False, (
                f"canary_ctr {canary_ctr:.3f} not "
                f"{int((CTR_LIFT_MARGIN - 1) * 100)}% above baseline {baseline_ctr:.3f}"
            )
        return True, "ok"

    def set_metrics(self, metrics: dict[str, float]) -> None:
        self._metrics = dict(metrics)

    def try_transition(self, target: str | None = None) -> dict[str, Any]:
        """Attempt to transition to ``target`` (or next state if None)."""
        if target is None:
            idx = _STATES.index(self.state)
            if idx + 1 >= len(_STATES):
                return {"state": self.state, "transitioned": False, "reason": "already at full"}
            target = _STATES[idx + 1]

        if target not in _STATES:
            raise HTTPException(status_code=400, detail=f"unknown state: {target}")

        if target == self.state:
            return {"state": self.state, "transitioned": False, "reason": "already in target state"}

        gate = self._gates.get(target)
        if gate is None:
            return {"state": self.state, "transitioned": False, "reason": f"no gate for {target}"}

        passed, reason = gate(self._metrics)
        if not passed:
            return {"state": self.state, "transitioned": False, "reason": reason}

        from_state = self.state
        self.state = target  # type: ignore[assignment]
        CURRENT_STATE.set(_STATES.index(self.state))
        STATE_CHANGES_TOTAL.labels(from_state=from_state, to_state=target).inc()
        GATE_EVALUATIONS_TOTAL.labels(gate=target).inc()
        return {"state": self.state, "transitioned": True, "reason": reason}

    def set_traffic(self, pct: int) -> dict[str, Any]:
        if self.state != RolloutState.CANARY:
            raise HTTPException(status_code=400, detail="traffic only settable in canary state")
        if not 0 <= pct <= MAX_TRAFFIC_PCT:
            raise HTTPException(status_code=400, detail=f"traffic must be 0-{MAX_TRAFFIC_PCT}")
        self.canary_traffic_pct = pct
        CANARY_TRAFFIC.set(pct)
        return {"state": self.state, "canary_traffic_pct": self.canary_traffic_pct}

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "canary_traffic_pct": self.canary_traffic_pct,
            "metrics": dict(self._metrics),
        }


class _State:
    controller: RolloutController | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _State.controller = RolloutController()
    yield


app = FastAPI(lifespan=lifespan, title="rollout")


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
    return _State.controller.status()


@app.post("/transition")
async def transition(body: dict[str, Any] | None = None) -> dict[str, Any]:
    if _State.controller is None:
        raise HTTPException(status_code=503, detail="not ready")
    target = (body or {}).get("target")
    return _State.controller.try_transition(target)


@app.post("/traffic")
async def set_traffic(body: dict[str, Any]) -> dict[str, Any]:
    if _State.controller is None:
        raise HTTPException(status_code=503, detail="not ready")
    pct = int(body.get("pct", 10))
    return _State.controller.set_traffic(pct)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=ROLLOUT_PORT)
