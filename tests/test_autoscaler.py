"""Unit tests for the predictive autoscaler forecasters and controller."""

from __future__ import annotations

import time

from bharatsearch_autoscaler import (
    AutoscalerController,
    HoltLinearForecaster,
    LightGBMQuantileForecaster,
    Observation,
    PersistenceForecaster,
    ScalingDecision,
)


def _obs(rps: float, offset: float = 0.0) -> Observation:
    return Observation(timestamp=time.time() + offset, rps=rps)


def test_persistence_forecaster_repeats_last_value():
    f = PersistenceForecaster()
    assert f.predict_upper(75.0) == 0.0
    f.update(_obs(100.0))
    assert f.predict_upper(75.0) == 100.0
    f.update(_obs(150.0))
    assert f.predict_upper(75.0) == 150.0
    assert f.confident() is True


def test_holt_linear_forecasts_trend():
    f = HoltLinearForecaster(alpha=0.5, beta=0.5)
    # Steadily increasing series.
    for i, rps in enumerate([100.0, 110.0, 120.0, 130.0, 140.0]):
        f.update(_obs(rps, offset=float(i)))
    pred = f.predict_upper(horizon_s=60.0)
    # With a positive trend, forecast should be above the last value.
    assert pred > 140.0
    assert f.confident() is True


def test_holt_linear_confident_after_min_history():
    f = HoltLinearForecaster(min_history=3)
    assert f.confident() is False
    f.update(_obs(100.0))
    f.update(_obs(100.0))
    assert f.confident() is False
    f.update(_obs(100.0))
    assert f.confident() is True


def test_lightgbm_quantile_falls_back_without_history():
    f = LightGBMQuantileForecaster(min_history=50)
    f.update(_obs(100.0))
    # Not enough history -> falls back to Holt linear.
    assert f.predict_upper(75.0) == 100.0
    assert f.confident() is False


def test_controller_scales_up_on_high_load():
    c = AutoscalerController(
        cap_per_pod=44.0,
        target_util=0.7,
        min_replicas=2,
        max_replicas=20,
        max_step=4,
        scale_down_intervals=8,
    )
    # 300 rps with cap_per_pod*util = 30.8 -> need ceil(300/30.8) = 10
    obs = _obs(300.0)
    decision = c.step(obs)
    assert decision.target_replicas > c.min_replicas
    assert decision.reactive_need == 10  # ceil(300 / 30.8)
    assert c.current > 2


def test_controller_holds_at_min_when_idle():
    c = AutoscalerController(
        cap_per_pod=44.0,
        target_util=0.7,
        min_replicas=2,
        max_replicas=20,
        max_step=4,
        scale_down_intervals=8,
    )
    c.current = 5
    decision = c.step(_obs(1.0))
    assert decision.target_replicas == 5  # holds, below scale-down threshold
    assert decision.mode == "reactive"


def test_controller_disabled_holds():
    c = AutoscalerController(
        cap_per_pod=44.0,
        target_util=0.7,
        min_replicas=2,
        max_replicas=20,
        max_step=4,
        scale_down_intervals=8,
    )
    c.enabled = False
    c.current = 7
    decision = c.step(_obs(500.0))
    assert decision.target_replicas == 7
    assert decision.mode == "disabled"


def test_controller_next_rung_snaps_to_ladder():
    c = AutoscalerController(
        cap_per_pod=44.0,
        target_util=0.7,
        min_replicas=2,
        max_replicas=20,
        max_step=4,
        scale_down_intervals=8,
    )
    assert c._next_rung(5) == 6
    assert c._next_rung(1) == 2
    assert c._next_rung(21) == 20
    assert c._prev_rung(10) == 8
    assert c._prev_rung(2) == 2


def test_controller_limits_scale_up_step():
    c = AutoscalerController(
        cap_per_pod=44.0,
        target_util=0.7,
        min_replicas=2,
        max_replicas=20,
        max_step=1,
        scale_down_intervals=8,
    )
    # Huge load would want many replicas, but max_step=1 limits the increase.
    decision = c.step(_obs(1000.0))
    assert decision.target_replicas <= c.current + 1


def test_scaling_decision_fields():
    d = ScalingDecision(
        target_replicas=6,
        reactive_need=5,
        predictive_need=6,
        forecast_rps=200.0,
        confident=True,
        mode="predictive",
        reason="scaled up",
    )
    assert d.target_replicas == 6
    assert d.confident is True


def test_lightgbm_unavailable_uses_holt():
    # Even when LightGBM is not installed, the forecaster constructs and falls back.
    f = LightGBMQuantileForecaster(min_history=3)
    for i in range(5):
        f.update(_obs(100.0 + i * 10.0, offset=float(i)))
    assert f.confident() is True
    pred = f.predict_upper(75.0)
    assert pred > 0.0
