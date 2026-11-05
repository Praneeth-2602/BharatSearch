"""Unit tests for the autoscaler forecaster backtest (Implementation Guide, Section 9.6)."""

from __future__ import annotations

from eval.autoscale_backtest import backtest


def _rising_curve() -> list[tuple[float, float]]:
    """A steadily increasing curve: 100, 110, 120, ..."""
    return [(float(t), 100.0 + t * 10.0) for t in range(0, 200, 10)]


def test_backtest_returns_metrics():
    curve = _rising_curve()
    result = backtest(curve, "holt", horizon_s=75.0)
    assert result["forecaster"] == "holt"
    assert result["n_points"] > 0
    assert result["pinball_loss"] is not None
    assert result["pinball_loss"] >= 0.0
    assert result["coverage"] is not None
    assert 0.0 <= result["coverage"] <= 1.0


def test_backtest_persistence_under_forecasts_ramps():
    curve = _rising_curve()
    pers = backtest(curve, "persistence", horizon_s=75.0)
    holt = backtest(curve, "holt", horizon_s=75.0)
    # Persistence under-forecasts every rising step; Holt should do better.
    assert pers["under_forecast_rate"] == 1.0
    assert holt["under_forecast_rate"] < pers["under_forecast_rate"]


def test_backtest_coverage_near_tau():
    curve = _rising_curve()
    result = backtest(curve, "holt", horizon_s=75.0)
    # Coverage at tau=0.9 should be in a reasonable range.
    assert result["coverage"] is not None
    assert result["coverage"] > 0.5


def test_backtest_short_curve():
    curve = [(0.0, 100.0), (10.0, 110.0)]
    result = backtest(curve, "holt", horizon_s=75.0)
    assert result["n_points"] == 2
    assert result["pinball_loss"] is None
