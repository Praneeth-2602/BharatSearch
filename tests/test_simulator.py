"""Unit tests for the click simulator (no Kafka required)."""

from __future__ import annotations

from bharatsearch_simulator.app import _click_probability, _State, app
from fastapi.testclient import TestClient


def test_click_probability_position_bias():
    """P(click) decreases with position for fixed relevance."""
    p0 = _click_probability("phone cover", "p1", 0)
    p1 = _click_probability("phone cover", "p1", 1)
    p2 = _click_probability("phone cover", "p1", 2)
    assert p0 > p1 > p2


def test_click_probability_relevance():
    """Higher relevance -> higher click probability at same position."""
    high = _click_probability("phone cover", "p1", 0)  # relevance 0.9
    low = _click_probability("phone cover", "p3", 0)  # relevance 0.3
    assert high > low

    # Unknown query/product -> default relevance 0.3
    default = _click_probability("unknown query", "unknown product", 0)
    assert abs(default - 0.3) < 1e-9


def test_simulate_endpoint():
    """The /simulate endpoint publishes events and returns stats."""
    _State.ready = True

    client = TestClient(app)
    resp = client.post(
        "/simulate",
        json={"sessions": 3, "queries": ["phone cover", "shoes"], "k": 3, "seed": 42},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sessions"] == 3
    assert data["clicks"] >= 0
    assert data["seed"] == 42
    assert "elapsed_ms" in data


def test_simulate_not_ready():
    """When not ready, /simulate returns 503."""
    _State.ready = False

    client = TestClient(app)
    resp = client.post("/simulate", json={"sessions": 1})
    assert resp.status_code == 503


def test_healthz_and_metrics():
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.text == "ok"

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "simulator_sessions_total" in resp.text
