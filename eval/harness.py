"""Evaluation harness: run configurations and write results/ (Implementation Guide, Section 4)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from eval.metrics import query_metrics


def run(qrels: dict[str, dict[str, int]], runs: dict[str, dict[str, list[str]]], k=10) -> dict:
    """Compute metrics for each configuration.

    qrels: {query_id: {product_id: label}}
    runs: {config_name: {query_id: [product_id, ...]}}
    """
    report = {}
    for name, run_ in runs.items():
        ndcgs, mrrs, recalls, judgeds = [], [], [], []
        for qid, ranked in run_.items():
            m = query_metrics(ranked, qrels.get(qid, {}), k=k)
            if m["ndcg@k"] is not None:
                ndcgs.append(m["ndcg@k"])
            mrrs.append(m["mrr@k"])
            if m["recall@k"] is not None:
                recalls.append(m["recall@k"])
            judgeds.append(m["judged@k"])

        report[name] = {
            "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else None,
            "mrr@k": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            "recall@k": sum(recalls) / len(recalls) if recalls else None,
            "judged@k": sum(judgeds) / len(judgeds) if judgeds else 0.0,
            "n_queries": len(run_),
        }
    return report


def bootstrap_ci(
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, list[str]]],
    k: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, tuple[float, float] | tuple[None, None]]]:
    """Bootstrap confidence intervals per config/metric.

    Resamples queries with replacement ``n_bootstrap`` times, computes the mean
    metric per resample, and returns the 2.5th/97.5th percentiles as
    {config: {metric: (lo, hi)}}. ``ndcg@k``/``recall@k`` average only over
    queries where they are applicable (non-None); ``mrr@k``/``judged@k`` over
    all queries, mirroring :func:`run`.
    """
    rng = random.Random(seed)
    query_ids = list(runs[next(iter(runs))].keys())
    n = len(query_ids)
    metrics = ["ndcg@k", "mrr@k", "recall@k", "judged@k"]

    # per-query values per config/metric (value or None)
    values: dict[str, dict[str, list[float | None]]] = {}
    for name, run_ in runs.items():
        values[name] = {}
        for metric in metrics:
            values[name][metric] = []
            for qid in query_ids:
                ranked = run_.get(qid, [])
                m = query_metrics(ranked, qrels.get(qid, {}), k=k)
                values[name][metric].append(m[metric])

    result: dict[str, dict[str, tuple]] = {}
    for name in runs:
        result[name] = {}
        for metric in metrics:
            arr = values[name][metric]
            boot_means: list[float] = []
            for _ in range(n_bootstrap):
                total = 0.0
                count = 0
                for _ in range(n):
                    v = arr[rng.randrange(n)]
                    if v is not None:
                        total += v
                        count += 1
                if count > 0:
                    boot_means.append(total / count)
            if boot_means:
                boot_means.sort()
                lo = boot_means[max(0, int(0.025 * len(boot_means)) - 1)]
                hi = boot_means[min(len(boot_means) - 1, int(0.975 * len(boot_means)))]
                result[name][metric] = (lo, hi)
            else:
                result[name][metric] = (None, None)
    return result


def write_metrics(report: dict, run_id: str, out_dir: str = "results") -> Path:
    dest = Path(out_dir) / run_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "metrics.json"
    path.write_text(json.dumps(report, indent=2))
    return path
