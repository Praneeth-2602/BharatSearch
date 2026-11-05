"""ESCI dataset download (Implementation Guide, Section 2.1).

Downloads the Amazon Shopping Queries (ESCI) dataset. Data files are never
committed to the repo; this script downloads them on demand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Placeholder default — the real ESCI distribution URL must be verified against
# the Amazon Shopping Queries dataset page before first use (Section 0 rule 3:
# no hidden constants; verify flags/mapping against pinned version docs).
DEFAULT_URL = "https://example.com/amazon-shopping-queries-dataset.zip"


def download(url: str = DEFAULT_URL, out: str = "data/raw") -> Path:
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] placeholder: would download {url} -> {dest}")
    print("[download] edit DEFAULT_URL or pass --url with the real ESCI source")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ESCI dataset")
    parser.add_argument("--url", default=DEFAULT_URL, help="ESCI dataset URL")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    args = parser.parse_args()
    download(args.url, args.out)


if __name__ == "__main__":
    main()
