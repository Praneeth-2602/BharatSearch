"""Unit tests for decayed feature counters (no Redis required)."""

from __future__ import annotations

import time

from bharatsearch_features import DecayedCounter, FeatureStore

_FLOAT_TOL = 1e-6


def _approx(a: float, b: float) -> bool:
    return abs(a - b) < _FLOAT_TOL


def test_decayed_counter_increment_and_get():
    c = DecayedCounter(decay_half_life_s=3600.0)
    assert _approx(c.get("k1"), 0.0)
    c.increment("k1", 5.0)
    assert _approx(c.get("k1"), 5.0)
    c.increment("k1", 3.0)
    assert _approx(c.get("k1"), 8.0)


def test_decayed_counter_independent_keys():
    c = DecayedCounter(decay_half_life_s=3600.0)
    c.increment("a", 1.0)
    c.increment("b", 2.0)
    assert _approx(c.get("a"), 1.0)
    assert _approx(c.get("b"), 2.0)
    assert _approx(c.get("missing"), 0.0)


def test_decayed_counter_set_resets():
    c = DecayedCounter(decay_half_life_s=3600.0)
    c.increment("k1", 5.0)
    c.set("k1", 42.0)
    assert _approx(c.get("k1"), 42.0)


def test_decayed_counter_keys():
    c = DecayedCounter(decay_half_life_s=3600.0)
    assert c.keys() == []
    c.increment("a", 1.0)
    c.increment("b", 2.0)
    assert sorted(c.keys()) == ["a", "b"]


def test_decayed_counter_decays_over_time():
    c = DecayedCounter(decay_half_life_s=1.0)
    c.increment("k1", 100.0)
    assert _approx(c.get("k1"), 100.0)
    time.sleep(2.1)
    val = c.get("k1")
    assert 20.0 < val < 30.0, f"expected ~25 after decay, got {val}"


def test_feature_store_ctr():
    fs = FeatureStore()
    fs.increment_query_clicks("phone cover", "p1", 3.0)
    fs.increment_query_impressions("phone cover", "p1", 10.0)
    assert _approx(fs.ctr("phone cover", "p1"), 0.3)

    # No impressions -> CTR 0
    assert _approx(fs.ctr("phone cover", "p9"), 0.0)


def test_feature_store_global_ctr():
    fs = FeatureStore()
    fs.increment_product_clicks("p1", 4.0)
    fs.increment_product_impressions("p1", 20.0)
    assert _approx(fs.global_ctr("p1"), 0.2)


def test_feature_store_product_counters():
    fs = FeatureStore()
    fs.increment_product_impressions("p1", 5.0)
    assert _approx(fs.product_impressions("p1"), 5.0)
    fs.increment_product_clicks("p1", 2.0)
    assert _approx(fs.product_clicks("p1"), 2.0)
