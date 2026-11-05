"""Forecaster backtest via rolling origin (Implementation Guide, Section 9.6).

Evaluates persistence, Holt linear, and LightGBM quantile forecasters on
held-out load curves using rolling-origin evaluation at horizon H. Reports
pinball loss at tau=0.9, quantile coverage, and under-forecast rate during
rising phases. Deterministic given a seed (Section 0 rule 4).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bharatsearch_autoscaler import (
    HoltLinearForecaster,
    LightGBMQuantileForecaster,
    Observation,
    PersistenceForecaster,
)

FORECASTERS = {
    "persistence": PersistenceForecaster,
    "holt": HoltLinearForecaster,
    "lightgbm": LightGBMQuantileForecaster,
}

TAU = 0.9  # upper quantile
STEP_S = 15  # control interval
MIN_HISTORY_HOLT = 3
MIN_CURVE_POINTS = 3


def _load_curve(path: str | Path) -> list[tuple[float, float]]:
    """Load a curve CSV into a list of (t_seconds, rps)."""
    points: list[tuple[float, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((float(row["t"]), float(row["rps"])))
    return points


def _pinball_loss(actual: float, prediction: float, tau: float = TAU) -> float:
    """Pinball (quantile) loss at level tau."""
    diff = actual - prediction
    return tau * diff if diff >= 0 else (tau - 1) * diff


def _rising(prev: float, curr: float) -> bool:
    return curr > prev


def backtest(
    curve: list[tuple[float, float]],
    forecaster_name: str = "holt",
    horizon_s: float = 75.0,
    min_history: int | None = None,
    step_s: float = STEP_S,
) -> dict:
    """Run rolling-origin backtest of one forecaster on one curve.

    Returns a dict with pinball loss, coverage, under-forecast rate, and
    per-point records for downstream reporting.
    """
    builder = FORECASTERS[forecaster_name]
    if forecaster_name == "lightgbm":
        f = builder(horizon_s=horizon_s, min_history=min_history or 50)
    elif forecaster_name == "holt":
        f = builder(min_history=min_history or MIN_HISTORY_HOLT)
    else:
        f = builder()

    n = len(curve)
    if n < MIN_CURVE_POINTS:
        return {
            "forecaster": forecaster_name,
            "n_points": n,
            "pinball_loss": None,
            "coverage": None,
            "under_forecast_rate": None,
            "records": [],
        }

    # Start evaluating once there is enough history for the forecaster.
    start_idx = (min_history or 3) if forecaster_name != "persistence" else 1
    records: list[dict] = []
    losses: list[float] = []
    covered: int = 0
    under_rising: int = 0
    rising_count: int = 0

    for i in range(start_idx, n - 1):
        history = curve[: i + 1]
        for t, rps in history:
            f.update(Observation(timestamp=t, rps=rps))

        origin_t, origin_rps = curve[i]
        target_i = min(i + 1, n - 1)
        actual_t, actual_rps = curve[target_i]

        # Advance the forecaster to the origin point (already done above via history).
        prediction = f.predict_upper(horizon_s)

        loss = _pinball_loss(actual_rps, prediction)
        losses.append(loss)
        covered += 1 if actual_rps <= prediction else 0

        if _rising(origin_rps, actual_rps):
            rising_count += 1
            if prediction < actual_rps:
                under_rising += 1

        records.append(
            {
                "t": actual_t,
                "origin_rps": origin_rps,
                "actual_rps": actual_rps,
                "predicted_rps": prediction,
                "pinball_loss": loss,
                "covered": actual_rps <= prediction,
                "under_forecast": prediction < actual_rps,
            }
        )

    n_records = len(records)
    return {
        "forecaster": forecaster_name,
        "n_points": n_records,
        "pinball_loss": sum(losses) / n_records if losses else None,
        "coverage": covered / n_records if n_records else None,
        "under_forecast_rate": (under_rising / rising_count) if rising_count else None,
        "records": records,
    }


def backtest_all(
    curve: list[tuple[float, float]],
    horizon_s: float = 75.0,
) -> dict[str, dict]:
    """Run backtest for every forecaster on one curve."""
    return {name: backtest(curve, name, horizon_s) for name in FORECASTERS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest autoscaler forecasters")
    parser.add_argument("--curve", required=True, help="Path to curve CSV")
    parser.add_argument(
        "--forecaster",
        choices=list(FORECASTERS) + ["all"],
        default="all",
    )
    parser.add_argument("--horizon-s", type=float, default=75.0)
    args = parser.parse_args(argv)

    curve = _load_curve(args.curve)
    if args.forecaster == "all":
        results = backtest_all(curve, args.horizon_s)
    else:
        results = {args.forecaster: backtest(curve, args.forecaster, args.horizon_s)}

    for name, r in results.items():
        print(
            f"{name}: pinball={r['pinball_loss']:.4f} "
            f"coverage={r['coverage']:.3f} "
            f"under_forecast={r['under_forecast_rate']} "
            f"(n={r['n_points']})"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
