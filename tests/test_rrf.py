"""Unit tests for RRF fusion (services/retrieval/app/fusion.py)."""

from bharatsearch_retrieval.fusion import rrf


def test_rrf_basic():
    rankings = [
        ["p1", "p2", "p3"],
        ["p2", "p1", "p4"],
    ]
    result = rrf(rankings, k=60, top=10)
    # p1 and p2 appear in both lists -> highest scores
    assert result[0][0] in ("p1", "p2")
    assert len(result) <= 10


def test_rrf_empty():
    assert rrf([]) == []
    assert rrf([[]]) == []
    assert rrf([[], []]) == []


def test_rrf_deterministic():
    rankings = [["a", "b", "c"], ["b", "a"]]
    assert rrf(rankings) == rrf(rankings)
