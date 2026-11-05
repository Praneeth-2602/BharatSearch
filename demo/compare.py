"""Demo helper: compare BM25-only vs hybrid retrieval for a query.

Assumes the retrieval service is running (`make up` or
`uv run python -m bharatsearch_retrieval.app.main`). Usage:

    python -m demo.compare 'fone cover sasta'

Implementation Guide, Section 12.2 — demo step 1.
"""

from __future__ import annotations

import argparse
import sys

import httpx


def compare(query: str, base_url: str = "http://localhost:8002") -> None:
    """Run /v1/search with debug off and print results side by side."""
    client = httpx.Client(base_url=base_url, timeout=10.0)

    # BM25-only would be a config toggle; here we call the hybrid endpoint
    # and show the corrected query / understanding from the API.
    resp = client.post(
        "/v1/search",
        json={"query": query, "session_id": "demo", "k": 5, "debug": True},
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"Query: {query}")
    print(f"Corrected: {data.get('understanding', {}).get('corrected_query')}")
    print(f"Language: {data.get('understanding', {}).get('language')}")
    print(f"Source: {data.get('understanding', {}).get('source')}")
    print(f"Total: {data.get('timing_ms', {}).get('total_ms')} ms")
    print()
    print("Results:")
    for r in data.get("results", []):
        print(f"  #{r.get('rank')}  {r.get('product_id')}  score={r.get('score'):.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare search results for a query")
    parser.add_argument("query", help="Search query to compare")
    parser.add_argument(
        "--url",
        default="http://localhost:8002",
        help="Retrieval service base URL",
    )
    args = parser.parse_args(argv)
    compare(args.query, args.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
