"""Unit tests for the rollout controller state machine."""

from __future__ import annotations

import pytest
from bharatsearch_rollout.app import RolloutController, _State, app
from fastapi.testclient import TestClient


def test_rollout_initial_state():
    rc = RolloutController()
    assert rc.state == "shadow"
    assert rc.canary_traffic_pct == 10


def test_rollout_transition_shadow_to_canary():
    rc = RolloutController()
    result = rc.try_transition("canary")
    assert result["transitioned"] is True
    assert result["state"] == "canary"
    assert rc.state == "canary"


def test_rollout_transition_canary_to_full():
    rc = RolloutController()
    rc.state = "canary"
    result = rc.try_transition("full")
    assert result["transitioned"] is True
    assert result["state"] == "full"


def test_rollout_gate_blocks_full_without_ctr_lift():
    """canary -> full requires canary_ctr >= 1.05 * baseline_ctr."""
    rc = RolloutController()
    rc.state = "canary"
    rc.set_metrics({"canary_ctr": 0.1, "baseline_ctr": 0.2})
    result = rc.try_transition("full")
    assert result["transitioned"] is False
    assert rc.state == "canary"


def test_rollout_gate_allows_full_with_ctr_lift():
    rc = RolloutController()
    rc.state = "canary"
    rc.set_metrics({"canary_ctr": 0.25, "baseline_ctr": 0.2})
    result = rc.try_transition("full")
    assert result["transitioned"] is True
    assert rc.state == "full"


def test_rollout_transition_idempotent():
    rc = RolloutController()
    result = rc.try_transition("shadow")
    assert result["transitioned"] is False
    assert result["reason"] == "already in target state"


def test_rollout_already_at_full():
    rc = RolloutController()
    rc.state = "full"
    result = rc.try_transition()
    assert result["transitioned"] is False
    assert result["reason"] == "already at full"


def test_rollout_set_traffic_only_in_canary():
    rc = RolloutController()
    with pytest.raises(Exception) as exc_info:
        rc.set_traffic(50)
    assert "traffic only settable" in str(exc_info.value)


def test_rollout_set_traffic_in_canary():
    rc = RolloutController()
    rc.state = "canary"
    result = rc.set_traffic(30)
    assert result["canary_traffic_pct"] == 30
    assert rc.canary_traffic_pct == 30


def test_rollout_status():
    rc = RolloutController()
    status = rc.status()
    assert status["state"] == "shadow"
    assert status["canary_traffic_pct"] == 10
    assert status["metrics"] == {}


def test_rollout_endpoints():
    _State.controller = RolloutController()

    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200

    resp = client.get("/state")
    assert resp.status_code == 200
    assert resp.json()["state"] == "shadow"

    resp = client.post("/transition", json={"target": "canary"})
    assert resp.status_code == 200
    assert resp.json()["transitioned"] is True

    resp = client.post("/traffic", json={"pct": 25})
    assert resp.status_code == 200
    assert resp.json()["canary_traffic_pct"] == 25

    resp = client.get("/metrics")
    assert resp.status_code == 200
