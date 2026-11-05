"""Predictive autoscaler forecasters and controller (Implementation Guide, Section 9).

Provides request-rate forecasters (persistence, Holt linear, LightGBM quantile
regression) and a scaling controller that combines a reactive floor with a
predictive upper-quantile forecast, gated by confidence and safety rules.

LightGBM is an optional dependency; when unavailable the quantile forecaster
falls back to Holt linear so the controller stays operational without the ML
stack.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

# Minimum number of rate samples needed to compute a slope feature.
_MIN_RATES_FOR_SLOPE = 2

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb

    _LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    lgb = None  # type: ignore[assignment,misc]
    _LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """One observed sample of request rate and system state."""

    timestamp: float
    rps: float
    p99_ms: float | None = None
    error_ratio: float | None = None
    queue_depth: float | None = None
    ready_replicas: int | None = None


@dataclass
class ScalingDecision:
    """The controller's output for one control interval."""

    target_replicas: int
    reactive_need: int
    predictive_need: int
    forecast_rps: float
    confident: bool
    mode: str  # "predictive" or "reactive"
    reason: str


class Forecaster:
    """Base class for request-rate forecasters."""

    def update(self, obs: Observation) -> None:
        """Incorporate one new observation."""
        raise NotImplementedError

    def predict_upper(self, horizon_s: float) -> float:
        """Upper-quantile (0.9) request rate over ``horizon_s`` seconds ahead."""
        raise NotImplementedError

    def confident(self) -> bool:
        """Whether the forecast is trustworthy enough to add capacity."""
        return True


class PersistenceForecaster(Forecaster):
    """Floor forecaster: repeats the last observed value.

    Effectively reactive — useful as a baseline and fallback.
    """

    def __init__(self) -> None:
        self._last: float | None = None

    def update(self, obs: Observation) -> None:
        self._last = obs.rps

    def predict_upper(self, horizon_s: float) -> float:
        return self._last if self._last is not None else 0.0


class HoltLinearForecaster(Forecaster):
    """Holt's linear (level + trend) exponential smoothing.

    Tracks a level and a trend, forecasts ``level + trend * horizon`` ahead, and
    returns an upper quantile by adding a multiple of the recent residual RMSE.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        quantile_multiplier: float = 1.28,
        min_history: int = 3,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.quantile_multiplier = quantile_multiplier
        self.min_history = min_history
        self._level: float | None = None
        self._trend: float = 0.0
        self._prev: float | None = None
        self._residuals: list[float] = []
        self._n: int = 0
        self._max_residuals: int = 200

    def update(self, obs: Observation) -> None:
        y = obs.rps
        if self._level is None:
            self._level = y
            self._trend = 0.0
            self._prev = y
            self._n = 1
            return
        prev_level = self._level
        self._level = self.alpha * y + (1.0 - self.alpha) * (prev_level + self._trend)
        self._trend = self.beta * (self._level - prev_level) + (1.0 - self.beta) * self._trend
        residual = y - (prev_level + self._trend)
        self._residuals.append(residual)
        if len(self._residuals) > self._max_residuals:
            self._residuals.pop(0)
        self._prev = y
        self._n += 1

    def _rmse(self) -> float:
        if not self._residuals:
            return 0.0
        return math.sqrt(sum(r * r for r in self._residuals) / len(self._residuals))

    def predict_upper(self, horizon_s: float) -> float:
        if self._level is None:
            return 0.0
        point = self._level + self._trend * (horizon_s / 60.0)
        return max(0.0, point + self.quantile_multiplier * self._rmse())

    def confident(self) -> bool:
        return self._n >= self.min_history


class LightGBMQuantileForecaster(Forecaster):
    """LightGBM quantile regression (alpha=0.9) on lagged rate features.

    Features: lagged rates (15, 30, 60, 120, 300 s), rolling mean/std, slope,
    time since start, schedule flags. Falls back to Holt linear when LightGBM
    is unavailable or there is insufficient history.
    """

    LAGS_S = [15, 30, 60, 120, 300]
    WINDOW_S = 300

    def __init__(
        self,
        horizon_s: float = 75.0,
        alpha: float = 0.9,
        min_history: int = 50,
        num_boost_round: int = 100,
    ) -> None:
        self.horizon_s = horizon_s
        self.alpha = alpha
        self.min_history = min_history
        self.num_boost_round = num_boost_round
        self._history: list[tuple[float, float]] = []  # (timestamp, rps)
        self._model: object | None = None
        self._trained = False
        self._start_time: float | None = None
        self._holt = HoltLinearForecaster()

    def _features(self, ts: float) -> list[float]:
        rates = [r for _, r in self._history]
        now = ts
        feats: list[float] = []
        for lag in self.LAGS_S:
            cutoff = now - lag
            recent = [r for t, r in self._history if t >= cutoff]
            feats.append(recent[-1] if recent else (rates[-1] if rates else 0.0))
        tail = rates[-self.WINDOW_S // 15 :] if rates else []
        if tail:
            mean = sum(tail) / len(tail)
            var = sum((r - mean) ** 2 for r in tail) / len(tail)
            feats.extend([mean, math.sqrt(var)])
        else:
            feats.extend([0.0, 0.0])
        if len(rates) >= _MIN_RATES_FOR_SLOPE:
            feats.append((rates[-1] - rates[-2]) / max(15.0, 1e-6))
        else:
            feats.append(0.0)
        t_since = (now - self._start_time) if self._start_time is not None else 0.0
        feats.append(t_since)
        feats.append(0.0)  # schedule flag placeholder
        return feats

    def update(self, obs: Observation) -> None:
        if self._start_time is None:
            self._start_time = obs.timestamp
        self._history.append((obs.timestamp, obs.rps))
        # Keep a bounded window.
        cutoff = obs.timestamp - 3600.0
        self._history = [(t, r) for t, r in self._history if t >= cutoff]
        self._holt.update(obs)
        if len(self._history) >= self.min_history and not self._trained:
            self._train()

    def _train(self) -> None:
        if not _LIGHTGBM_AVAILABLE or len(self._history) < self.min_history:
            return
        try:
            X: list[list[float]] = []
            y: list[float] = []
            for i in range(self.min_history, len(self._history)):
                ts_i, rps_i = self._history[i]
                X.append(self._features(ts_i))
                y.append(rps_i)
            train_data = lgb.Dataset(X, label=y)  # type: ignore[attr-defined]
            params = {
                "objective": "quantile",
                "alpha": self.alpha,
                "metric": "quantile",
                "verbose": -1,
            }
            self._model = lgb.train(  # type: ignore[attr-defined]
                params, train_data, num_boost_round=self.num_boost_round
            )
            self._trained = True
            logger.info("Trained LightGBM quantile forecaster on %d samples", len(X))
        except Exception:  # pragma: no cover - best effort
            logger.exception("LightGBM quantile training failed; using Holt linear")

    def predict_upper(self, horizon_s: float) -> float:
        if self._trained and self._model is not None and self._history:
            last_ts, _ = self._history[-1]
            future_ts = last_ts + horizon_s
            feats = self._features(future_ts)
            pred = float(self._model.predict([feats])[0])  # type: ignore[attr-defined]
            return max(0.0, pred)
        return self._holt.predict_upper(horizon_s)

    def confident(self) -> bool:
        return self._trained or self._holt.confident()


class AutoscalerController:
    """Scaling controller combining reactive floor with predictive capacity.

    Each interval the controller reads the current observation, asks the
    forecaster for an upper-quantile rate over the horizon, computes a reactive
    need and a predictive need, takes the maximum, snaps to the next ladder
    rung, clamps to min/max, and applies step-up/step-down safety rules.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - natural scaling configuration parameters
        self,
        cap_per_pod: float,
        target_util: float,
        horizon_s: float = 75.0,
        ladder: list[int] | None = None,
        min_replicas: int = 2,
        max_replicas: int = 20,
        max_step: int = 4,
        scale_down_intervals: int = 8,
        forecaster: Forecaster | None = None,
    ) -> None:
        self.cap_per_pod = cap_per_pod
        self.target_util = target_util
        self.horizon_s = horizon_s
        self.ladder = ladder or [2, 3, 4, 6, 8, 10, 12, 14, 16, 20]
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.max_step = max_step
        self.scale_down_intervals = scale_down_intervals
        self.forecaster = forecaster or HoltLinearForecaster()
        self.current = min_replicas
        self._intervals_below: int = 0
        self._last_scale_time: float | None = None
        self.cooldown_s: float = 120.0
        self.enabled: bool = True
        self.decisions: list[ScalingDecision] = []

    def _next_rung(self, target: int) -> int:
        for rung in self.ladder:
            if rung >= target:
                return rung
        return self.ladder[-1]

    def _prev_rung(self, current: int) -> int:
        prev = self.min_replicas
        for rung in self.ladder:
            if rung >= current:
                return prev
            prev = rung
        return self.min_replicas

    def step(self, obs: Observation) -> ScalingDecision:
        """Run one control interval and return the scaling decision."""
        if not self.enabled:
            decision = ScalingDecision(
                target_replicas=self.current,
                reactive_need=self.current,
                predictive_need=self.current,
                forecast_rps=obs.rps,
                confident=False,
                mode="disabled",
                reason="autoscaler disabled",
            )
            self.decisions.append(decision)
            return decision

        self.forecaster.update(obs)
        f = self.forecaster.predict_upper(self.horizon_s)

        cap = self.cap_per_pod * self.target_util
        reactive_need = max(1, math.ceil(obs.rps / cap))
        queue_term = 0
        if obs.queue_depth is not None:
            queue_term = math.ceil(obs.queue_depth / 64.0)
        reactive_need = max(reactive_need, queue_term)

        confident = self.forecaster.confident()
        if confident:
            predictive_need = max(1, math.ceil(f / cap))
        else:
            predictive_need = 0

        target = max(reactive_need, predictive_need)
        target = self._next_rung(target)
        target = min(max(target, self.min_replicas), self.max_replicas)

        now = obs.timestamp
        mode = "predictive" if confident else "reactive"
        reason = "ok"

        if target > self.current:
            # Scale up immediately, but no more than MAX_STEP rungs at once.
            proposed = min(target, self.current + self.max_step)
            target = proposed
            self._intervals_below = 0
            self._last_scale_time = now
            reason = "scaled up"
        elif target < self.current:
            self._intervals_below += 1
            cooldown_ok = (
                self._last_scale_time is None or (now - self._last_scale_time) >= self.cooldown_s
            )
            if self._intervals_below >= self.scale_down_intervals and cooldown_ok:
                target = self._prev_rung(self.current)
                self._intervals_below = 0
                self._last_scale_time = now
                reason = "scaled down"
            else:
                target = self.current
                reason = "holding (scale-down cooldown / min-intervals)"
        else:
            reason = "holding"

        self.current = target
        decision = ScalingDecision(
            target_replicas=target,
            reactive_need=reactive_need,
            predictive_need=predictive_need,
            forecast_rps=f,
            confident=confident,
            mode=mode,
            reason=reason,
        )
        self.decisions.append(decision)
        return decision


__all__ = [
    "Observation",
    "ScalingDecision",
    "Forecaster",
    "PersistenceForecaster",
    "HoltLinearForecaster",
    "LightGBMQuantileForecaster",
    "AutoscalerController",
]
