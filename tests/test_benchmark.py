"""Unit tests for eval/benchmark.py: runs on toy data and writes a markdown report."""

from __future__ import annotations

from eval.benchmark import _slice_metrics, write_report


def test_benchmark_toy_writes_report(tmp_path):
    qrels = {"q1": {"p1": 3, "p2": 2}}
    runs = {
        "bm25": {"q1": ["p1", "p2"]},
        "hybrid": {"q1": ["p2", "p1"]},
    }
    variants = {"q1": "clean_en"}
    report = _slice_metrics(qrels, runs, variants)

    assert "bm25" in report
    assert "hybrid" in report
    assert "overall" in report["bm25"]
    assert "clean_en" in report["bm25"]
    assert report["bm25"]["overall"]["n_queries"] == 1
    assert report["bm25"]["clean_en"]["n_queries"] == 1

    path = write_report(report, out_dir=str(tmp_path))
    assert path.exists()
    content = path.read_text()
    assert "# Search Benchmark" in content
    assert "bm25" in content
    assert "hybrid" in content
    assert "clean_en" in content
    assert "nDCG@10" in content
