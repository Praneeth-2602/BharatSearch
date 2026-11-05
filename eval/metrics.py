"""Evaluation metrics (Implementation Guide, Section 4.2).

GAIN scale: Exact 3, Substitute 2, Complement 1, Irrelevant 0.
"""

from __future__ import annotations

import math

GAIN = {3: 1.0, 2: 0.1, 1: 0.01, 0: 0.0}

EXACT_GRADE = 3


def dcg(labels) -> float:
    return sum(GAIN[label] / math.log2(i + 2) for i, label in enumerate(labels))


def ndcg_at_k(ranked, qrels_q, k=10):
    """NDCG@k. Unjudged products count as 0; skip queries with no relevant item."""
    gains = [qrels_q.get(p, 0) for p in ranked[:k]]
    ideal = sorted(qrels_q.values(), reverse=True)[:k]
    d = dcg(ideal)
    return dcg(gains) / d if d > 0 else None


def mrr_at_k(ranked, qrels_q, k=10):
    """MRR over the first Exact item."""
    for i, p in enumerate(ranked[:k], start=1):
        if qrels_q.get(p, 0) == EXACT_GRADE:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked, qrels_q, k=50):
    """Recall@k over Exact items."""
    exact = [p for p, v in qrels_q.items() if v == EXACT_GRADE]
    if not exact:
        return None
    found = [p for p in ranked[:k] if qrels_q.get(p, 0) == EXACT_GRADE]
    return len(found) / len(exact)


def judged_at_k(ranked, qrels_q, k=10):
    """Share of top-k results that carry a relevance judgment."""
    if not k:
        return 0.0
    judged = [p for p in ranked[:k] if p in qrels_q]
    return len(judged) / k
