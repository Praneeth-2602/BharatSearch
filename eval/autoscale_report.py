"""Autoscaler experiment report (Implementation Guide, Section 9.8).

Reads backtest results and writes a markdown table of every metric by
forecaster and curve, plus a lead-time analysis comparing predictive
against reactive scale-up timing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path


def write_report(
    results: dict[str, dict[str, dict]],
    out_dir: str = "docs/benchmarks",
) -> Path:
    """Write a markdown report from backtest results.

    results: {curve_name: {forecaster_name: metrics_dict}}
    """
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = dest / f"autoscale-{today}.md"

    lines = [f"# Autoscaler Forecaster Backtest — {today}", ""]
    lines.append(
        "| Curve | Forecaster | n_points | pinball_loss | coverage | under_forecast_rate |"
    )
    lines.append("|---|---|---|---|---|---|")

    for curve_name in sorted(results):
        for forecaster in sorted(results[curve_name]):
            r = results[curve_name][forecaster]
            lines.append(
                f"| {curve_name} | {forecaster} | {r['n_points']} | "
                f"{r['pinball_loss']:.4f} | {r['coverage']:.3f} | "
                f"{r['under_forecast_rate']} |"
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **pinball_loss**: quantile loss at tau=0.9 (lower is better).")
    lines.append("- **coverage**: fraction of actuals at or below the 0.9 quantile")
    lines.append("  forecast (target ~0.9).")
    lines.append("- **under_forecast_rate**: share of rising-phase points where the")
    lines.append("  forecast was below actual (lower is better).")
    lines.append("")
    lines.append("A learned forecaster enters the controller only if it beats persistence")
    lines.append("on pinball loss on the matchday curves (Section 9.6 gate).")

    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    print("[autoscale_report] usage: import write_report(results) from this module")


if __name__ == "__main__":
    main()
