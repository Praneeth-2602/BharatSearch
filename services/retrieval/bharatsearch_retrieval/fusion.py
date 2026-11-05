"""Reciprocal rank fusion (Implementation Guide, Section 3.3).

Fuse top-k results from multiple retrievers. Tune `k` on gold_dev only.
"""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60, top: int = 100) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]
