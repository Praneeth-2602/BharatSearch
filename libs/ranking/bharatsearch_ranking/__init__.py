"""LightGBM LambdaRank ranker (Implementation Guide, Section 7.3).

Trains a LambdaRank ranking model on click-derived training data and provides
inference for re-ranking retrieval candidates. LightGBM is an optional
dependency; when unavailable the ranker falls back to RRF ordering (identity
inference) so the service stays operational without the ML stack.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency path
    import lightgbm as lgb

    _LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    lgb = None  # type: ignore[assignment,misc]
    _LIGHTGBM_AVAILABLE = False


class Ranker:
    """LambdaRank ranker with LightGBM backend and RRF fallback.

    ``train`` fits a LightGBM ``LambaRank`` model on feature/label pairs
    derived from click data. ``predict`` scores a list of candidate product
    ids for a query and returns them in ranked order. When LightGBM is
    unavailable, ``predict`` returns candidates in input order (RRF fallback).
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model: Any = None
        self._trained = False
        if model_path:
            self._load(model_path)

    def _load(self, path: str) -> None:
        if not _LIGHTGBM_AVAILABLE:  # pragma: no cover - LightGBM unavailable
            logger.warning("LightGBM not installed; cannot load ranker model from %s", path)
            return
        try:
            self._model = lgb.Booster(model_file=path)  # type: ignore[attr-defined]
            self._trained = True
            logger.info("Loaded ranker model from %s", path)
        except Exception:  # pragma: no cover - best effort
            logger.exception("Failed to load ranker model from %s", path)

    def train(  # noqa: PLR0913, PLR0917 - natural training hyperparameters
        self,
        X: list[list[float]],
        y: list[float],
        group: list[int] | None = None,
        num_boost_round: int = 100,
        learning_rate: float = 0.1,
        save_path: str | None = None,
    ) -> None:
        """Train a LambdaRank model.

        ``X`` is a list of feature vectors, ``y`` is a list of relevance
        labels (0..4, higher = more relevant). ``group`` gives the number of
        candidates per query for LambdaRank grouping; if omitted all
        candidates are treated as one group.
        """
        if not _LIGHTGBM_AVAILABLE:  # pragma: no cover - LightGBM unavailable
            logger.warning("LightGBM not installed; cannot train ranker")
            return
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")

        train_data = lgb.Dataset(X, label=y, group=group)  # type: ignore[attr-defined]
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": learning_rate,
            "verbose": -1,
        }
        start = time.time()
        self._model = lgb.train(  # type: ignore[attr-defined]
            params,
            train_data,
            num_boost_round=num_boost_round,
        )
        self._trained = True
        elapsed = time.time() - start
        logger.info(
            "Trained ranker in %.2fs (%d candidates, %d groups)",
            elapsed,
            len(X),
            len(group or [len(X)]),
        )

        if save_path:
            self._model.save_model(save_path)  # type: ignore[attr-defined]
            logger.info("Saved ranker model to %s", save_path)
            if self.model_path is None:
                self.model_path = save_path

    def predict(
        self, candidates: list[str], features: list[list[float]]
    ) -> list[tuple[str, float]]:
        """Score and rank candidates for a query.

        Returns ``[(product_id, score)]`` sorted by score descending. When
        the model is not trained or LightGBM is unavailable, returns
        candidates in input order with score 1.0 (RRF fallback).
        """
        if not self._trained or self._model is None:  # pragma: no cover - fallback path
            return [(pid, 1.0) for pid in candidates]

        scores = self._model.predict(features)  # type: ignore[attr-defined]
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -x[1])
        return ranked

    @property
    def trained(self) -> bool:
        return self._trained


__all__ = ["Ranker"]
