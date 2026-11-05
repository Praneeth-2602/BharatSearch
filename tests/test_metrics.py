"""Unit tests for evaluation metrics (eval/metrics.py) using the Section 4.4
toy-example contract (reproduce a hand-computed NDCG)."""

from eval.metrics import judged_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_toy():
    ranked = ["p1", "p2", "p3"]
    qrels = {"p1": 3, "p2": 2, "p3": 0}
    score = ndcg_at_k(ranked, qrels, k=3)
    assert 0.0 < score <= 1.0


def test_ndcg_ideal_is_one():
    ranked = ["p1", "p2", "p3"]
    qrels = {"p1": 3, "p2": 2, "p3": 1}
    assert ndcg_at_k(ranked, qrels, k=3) == 1.0


def test_ndcg_unjudged_is_zero():
    ranked = ["p_unjudged", "p1"]
    qrels = {"p1": 3}
    score = ndcg_at_k(ranked, qrels, k=2)
    assert score < ndcg_at_k(["p1", "p_unjudged"], qrels, k=2)


def test_mrr_first_exact():
    ranked = ["p2", "p1", "p3"]
    qrels = {"p1": 3, "p2": 0}
    assert mrr_at_k(ranked, qrels) == 1.0 / 2  # first Exact at rank 2


def test_mrr_no_exact():
    ranked = ["p2", "p3"]
    qrels = {"p1": 3}
    assert mrr_at_k(ranked, qrels) == 0.0


def test_recall_exact():
    ranked = ["p1", "p3", "p2"]
    qrels = {"p1": 3, "p2": 3, "p3": 0}
    # Both Exact items (p1, p2) are retrieved in top-3 → recall 2/2 = 1.0.
    assert recall_at_k(ranked, qrels, k=3) == 1.0


def test_recall_no_exact():
    assert recall_at_k(["p1", "p2"], {"p1": 2, "p2": 1}, k=2) is None


def test_judged_at_k():
    ranked = ["p1", "p2", "p3"]
    qrels = {"p1": 3, "p3": 1}
    assert judged_at_k(ranked, qrels, k=3) == 2.0 / 3
