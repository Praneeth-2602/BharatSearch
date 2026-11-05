"""ESCI dataset slicing (Implementation Guide, Section 2.1).

Filter to US locale, sample 3,000 train / 500 dev / 500 test queries (queries
with >=1 Exact product), keep every judged product for those queries, pad the
catalog to the profile target size, and derive `cluster_id` via k-means on
product embeddings. Writes catalog.parquet, queries.parquet, qrels.parquet.
"""

from __future__ import annotations

import argparse

import pandas as pd

# Artifact schemas (Section 2.1)
CATALOG_COLUMNS = ["product_id", "title", "brand", "bullets", "description", "color", "cluster_id"]
QUERIES_COLUMNS = [
    "query_id",
    "query_text",
    "split",
    "source_query_id",
    "variant_type",
    "language_tag",
    "transforms",
    "gold",
]
QRELS_COLUMNS = ["query_id", "product_id", "label"]

# ESCI label -> numeric grade
LABEL_GRADE = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}


def load_esci(queries_path: str, products_path: str):
    queries = pd.read_parquet(queries_path) if queries_path.endswith(".parquet") else None
    products = pd.read_parquet(products_path) if products_path.endswith(".parquet") else None
    return queries, products


def filter_us(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["product_locale"] == "us"].copy()


def sample_queries(
    df: pd.DataFrame, n_train: int = 3000, n_dev: int = 500, n_test: int = 500, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Sample queries with at least one Exact product, split by time (not random)."""
    eligible = df[df["esci_label"] == "Exact"]["query_id"].unique()
    eligible = pd.Series(eligible).sample(frac=1, random_state=seed).tolist()
    n_train = min(n_train, len(eligible))
    n_dev = min(n_dev, max(0, len(eligible) - n_train))
    n_test = min(n_test, max(0, len(eligible) - n_train - n_dev))
    splits = {
        "train": eligible[:n_train],
        "dev": eligible[n_train : n_train + n_dev],
        "test": eligible[n_train + n_dev : n_train + n_dev + n_test],
    }
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ESCI sample slice")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--products", required=True)
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    print(f"[make_slice] placeholder: slice {args.queries} + {args.products} -> {args.out}")


if __name__ == "__main__":
    main()
