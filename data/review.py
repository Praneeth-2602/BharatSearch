"""Gold set review CLI (Implementation Guide, Section 2.3).

Review generated variants with keys for accept, fix (edit the text), and
reject. Splits the reviewed set into gold_dev (150) and gold_test (250).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def review(variants_path: str, out: str = "data/gold") -> None:
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[review] placeholder: review variants from {variants_path} -> {dest}")
    print("[review] implement interactive accept/fixed/reject keys, then split 150/250")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review gold set variants")
    parser.add_argument("--variants", required=True, help="Path to variants parquet/jsonl")
    parser.add_argument("--out", default="data/gold", help="Output directory")
    args = parser.parse_args()
    review(args.variants, args.out)


if __name__ == "__main__":
    main()
