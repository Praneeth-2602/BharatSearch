"""Evaluation harness: run configurations and write results/ (Implementation Guide, Section 4)."""

from __future__ import annotations

import json
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


def write_metrics(report: dict, run_id: str, out_dir: str = "results") -> Path:
    dest = Path(out_dir) / run_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "metrics.json"
    path.write_text(json.dumps(report, indent=2))
    return path
