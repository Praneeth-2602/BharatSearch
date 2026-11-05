"""Unit tests for the autoscaler controller service."""

from __future__ import annotations

from bharatsearch_autoscaler import AutoscalerController
from bharatsearch_autoscaler_service.app import _State, app
from fastapi.testclient import TestClient


def test_autoscaler_healthz():
    _State.controller = AutoscalerController(cap_per_pod=44.0, target_util=0.7)
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.text == "ok"


def test_autoscaler_state():
    _State.controller = AutoscalerController(cap_per_pod=44.0, target_util=0.7)
    client = TestClient(app)
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "current_replicas" in data
    assert "min_replicas" in data
    assert "max_replicas" in data


def test_autoscaler_observe():
    _State.controller = AutoscalerController(cap_per_pod=44.0, target_util=0.7)
    client = TestClient(app)
    resp = client.post("/observe", json={"rps": 300.0})
    assert resp.status_code == 200
    data = resp.json()
    assert "target_replicas" in data
    assert "forecast_rps" in data
    assert "mode" in data
    assert data["mode"] in ("predictive", "reactive")


def test_autoscaler_metrics():
    _State.controller = AutoscalerController(cap_per_pod=44.0, target_util=0.7)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "autoscaler_decisions_total" in resp.text
