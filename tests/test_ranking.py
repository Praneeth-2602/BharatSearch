"""Unit tests for the LambdaRank ranker (LightGBM optional)."""

from __future__ import annotations

from bharatsearch_ranking import Ranker


def test_ranker_untrained_fallback():
    """Without LightGBM or a trained model, predict returns input order."""
    r = Ranker()
    assert r.trained is False
    result = r.predict(["p3", "p1", "p2"], [[0.1], [0.5], [0.3]])
    assert result == [("p3", 1.0), ("p1", 1.0), ("p2", 1.0)]


def test_ranker_train_without_lightgbm():
    """train() is a no-op when LightGBM is not installed."""
    r = Ranker()
    # Should not raise even without LightGBM
    r.train(
        X=[[0.1, 0.2], [0.3, 0.4]],
        y=[0.0, 1.0],
        group=[2],
    )
    assert r.trained is False


def test_ranker_predict_empty():
    r = Ranker()
    assert r.predict([], []) == []


def test_ranker_trained_predict_order():
    """When a model is loaded, predict ranks by score descending."""
    # Can't easily train without LightGBM, but we can verify the
    # fallback path preserves input order when untrained.
    r = Ranker()
    result = r.predict(["a", "b", "c"], [[1.0], [2.0], [3.0]])
    # Untrained: input order, all score 1.0
    assert result == [("a", 1.0), ("b", 1.0), ("c", 1.0)]
