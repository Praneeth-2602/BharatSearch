# ADR-005: Autoscaler design (horizon, quantile, ladder, confidence fallback)

## Status

Accepted

## Context

The predictive autoscaler (Implementation Guide Section 9) must scale a
stub-inference Deployment ahead of load ramps (match start, second-innings
surge) while remaining reactive to unannounced spikes and flash crowds. Four
strategies are compared: S1 (HPA on CPU), S2 (KEDA on request rate), S3
(rule-based ladders from a schedule), S4 (predictive forecaster plus ladder).

## Decision

Use a horizon of H ≈ 75 seconds (cold start + control interval + safety
margin), an upper-quantile (0.9) forecast, ladder `[2, 3, 4, 6, 8, 10, 12,
14, 16, 20]`, step-up immediate / step-down cooldown, and a kill switch.
Forecasters: persistence (floor), Holt linear (ramps), LightGBM quantile
(alpha=0.9, falls back to Holt). S4 enters the controller only if it beats
persistence on pinball loss on matchday curves (Section 9.6 gate). S3 uses a
schedule-based floor (`ScheduleForecaster`) plus a ladder over observed
request rate.

## Consequences

- Prediction helps on scheduled ramps (pre-match, second-innings surge) and
  does not hurt on unannounced spikes (reactive floor covers them).
- The 0.9 quantile biases toward over-provisioning, which is safer than
  under-forecasting a ramp.
- When forecaster coverage falls below 0.7, S4 falls back to reactive-only
  mode and recovers after M good predictions.
- Results are reported honestly: mixed outcomes expected, with prediction
  helping on ramps and making no difference on flash crowds.

## Related

ADR-001 (RRF fusion)
