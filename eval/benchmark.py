"""Benchmark: BM25-only vs hybrid retrieval (Implementation Guide, Section 4.4).

Compares BM25-only and hybrid retrieval and writes a markdown report under
docs/benchmarks/<date>.md. Works with toy data when the ESCI slice is not
downloaded (prints clear guidance otherwise). Deterministic given a seed
(Section 0 rule 4).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from eval.harness import run


def _toy_qrels() -> dict[str, dict[str, int]]:
    return {
        "q1": {"p1": 3, "p2": 2, "p3": 0},
        "q2": {"p1": 0, "p2": 3, "p4": 1},
        "q3": {"p5": 3, "p6": 2, "p7": 0},
    }


def _toy_runs() -> dict[str, dict[str, list[str]]]:
    return {
        "bm25": {
            "q1": ["p1", "p2", "p3"],
            "q2": ["p2", "p1", "p4"],
            "q3": ["p5", "p6", "p7"],
        },
        "hybrid": {
            "q1": ["p1", "p3", "p2"],
            "q2": ["p2", "p4", "p1"],
            "q3": ["p5", "p7", "p6"],
        },
    }


def _toy_variants() -> dict[str, str]:
    return {"q1": "clean_en", "q2": "typo_en", "q3": "hinglish_roman"}


def _slice_metrics(
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, list[str]]],
    query_variants: dict[str, str],
    k: int = 10,
) -> dict:
    """Per-config, per-variant-type metrics."""
    report: dict[str, dict] = {}
    for name, run_data in runs.items():
        report[name] = {}
        overall = run(qrels, {name: run_data}, k=k)
        report[name]["overall"] = overall[name]
        for vt in sorted(set(query_variants.values())):
            vt_qids = [q for q, v in query_variants.items() if v == vt]
            vt_qrels = {q: qrels.get(q, {}) for q in vt_qids}
            vt_run = {q: run_data.get(q, []) for q in vt_qids}
            slice_report = run(vt_qrels, {name: vt_run}, k=k)
            report[name][vt] = slice_report[name]
    return report


def write_report(report: dict, out_dir: str = "docs/benchmarks") -> Path:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = dest / f"{today}.md"

    lines = [f"# Search Benchmark — {today}", ""]
    lines.append("| Config | Slice | nDCG@10 | MRR | Recall@50 | judged@10 | n_queries |")
    lines.append("|---|---|---|---|---|---|---|")
    for config, slices in report.items():
        for slice_name, m in slices.items():
            lines.append(
                f"| {config} | {slice_name} | {m.get('ndcg@k')} | {m.get('mrr@k')} | "
                f"{m.get('recall@k')} | {m.get('judged@k')} | {m.get('n_queries')} |"
            )
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> dict:
    qrels = _toy_qrels()
    runs = _toy_runs()
    query_variants = _toy_variants()
    report = _slice_metrics(qrels, runs, query_variants)
    path = write_report(report)
    print(f"[benchmark] wrote {path}")
    return report


if __name__ == "__main__":
    main()
