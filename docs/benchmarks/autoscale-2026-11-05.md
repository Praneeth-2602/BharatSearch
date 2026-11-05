# Autoscaler Forecaster Backtest — 2026-11-05

| Curve | Forecaster | n_points | pinball_loss | coverage | under_forecast_rate |
|---|---|---|---|---|---|
| diurnal | holt | 176 | 2.4506 | 0.972 | 0.05813953488372093 |
| diurnal | lightgbm | 176 | 2.4506 | 0.972 | 0.05813953488372093 |
| diurnal | persistence | 178 | 3.6414 | 0.511 | 1.0 |
| flashcrowd | holt | 176 | 6.5636 | 0.943 | 0.1125 |
| flashcrowd | lightgbm | 176 | 6.5636 | 0.943 | 0.1125 |
| flashcrowd | persistence | 178 | 3.6386 | 0.551 | 1.0 |
| matchday | holt | 176 | 9.1377 | 0.960 | 0.06862745098039216 |
| matchday | lightgbm | 176 | 9.1377 | 0.960 | 0.06862745098039216 |
| matchday | persistence | 178 | 6.9022 | 0.421 | 1.0 |
| steady | holt | 176 | 0.8290 | 0.886 | 0.25 |
| steady | lightgbm | 176 | 0.8290 | 0.886 | 0.25 |
| steady | persistence | 178 | 2.2792 | 0.551 | 1.0 |

## Interpretation

- **pinball_loss**: quantile loss at tau=0.9 (lower is better).
- **coverage**: fraction of actuals at or below the 0.9 quantile
  forecast (target ~0.9).
- **under_forecast_rate**: share of rising-phase points where the
  forecast was below actual (lower is better).

A learned forecaster enters the controller only if it beats persistence
on pinball loss on the matchday curves (Section 9.6 gate).
