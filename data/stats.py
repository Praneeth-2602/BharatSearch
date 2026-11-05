"""Stats report for generated data (Implementation Guide, Sections 2.2 & 2.3).

Reports counts per variant type, label distribution, and rejection rate of
each filter (meaning-drift, empty, edit-distance-0, duplicates).
"""

from __future__ import annotations

import argparse
import json


def stats_report(variants_path: str, qrels_path: str) -> dict:
    report = {
        "variant_counts": {},
        "label_distribution": {},
        "rejection_rates": {},
    }
    print(f"[stats] placeholder: report for {variants_path} / {qrels_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data stats report")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--qrels", required=True)
    args = parser.parse_args()
    report = stats_report(args.variants, args.qrels)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
