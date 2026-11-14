"""Index ESCI dataset products into OpenSearch + Qdrant, and build qrels for eval.

Builds the catalog from judged products (products with qrels judgments) plus a
configurable sample of distractor products, so the benchmark is evaluated against
products that actually have relevance judgments. Replaces the 100K random sample.
Qrels are rebuilt from the ESCI queries parquet carrying esci_label E/S/C/I,
mapped to grades 3/2/1/0.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import random
from pathlib import Path

import httpx
import pandas as pd

OPENSEARCH_URL = "http://localhost:9200"
QDRANT_URL = "http://localhost:6333"
INDEX_NAME = "products"
COLLECTION_NAME = "products"
VECTOR_SIZE = 384
RANDOM_SEED = 42
QDRANT_BATCH = 2000
DISTRACTOR_N = 50_000
QRELS_PATH = Path("results/esci_qrels.json")

PRODUCT_COLUMNS = [
    "product_id",
    "product_title",
    "product_brand",
    "product_bullet_point",
    "product_description",
    "product_locale",
]

GRADE_MAP = {"E": 3, "S": 2, "C": 1, "I": 0}


def _stable_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-random unit vector (same as stub embedding service)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:4], "little"))
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _point_id(product_id: str) -> int:
    """Deterministic unsigned 32-bit ID for Qdrant points."""
    digest = hashlib.sha256(product_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0xFFFFFFFF


def _load_judged_product_ids() -> set[str]:
    """Load judged product IDs from the current qrels file."""
    if not QRELS_PATH.exists():
        return set()
    qrels = json.loads(QRELS_PATH.read_text())
    judged: set[str] = set()
    for products in qrels.values():
        judged.update(products.keys())
    return judged


def load_products(
    judged_ids: set[str] | None = None, distractor_n: int = DISTRACTOR_N
) -> list[dict]:
    """Load ESCI US products: all judged products plus a sample of distractors.

    Reads product parquet files one at a time to keep memory bounded, filtering
    to judged products first and sampling distractors from the remainder.
    """
    if judged_ids is None:
        judged_ids = _load_judged_product_ids()

    product_files = [f"data/esci/products/train-0000{i}-of-00004.parquet" for i in range(4)] + [
        f"data/esci/products/test-0000{i}-of-00002.parquet" for i in range(2)
    ]

    judged_parts: list[pd.DataFrame] = []
    non_judged_parts: list[pd.DataFrame] = []

    for path in product_files:
        df = pd.read_parquet(path, columns=PRODUCT_COLUMNS)
        df = df[df["product_locale"] == "us"]
        df = df.drop_duplicates(subset="product_id")
        df = df.dropna(subset=["product_title"])
        df["_pid"] = df["product_id"].astype(str)

        judged_mask = df["_pid"].isin(judged_ids)
        judged_parts.append(df[judged_mask].copy())
        non_judged_parts.append(df[~judged_mask].copy())

    judged_df = pd.concat(judged_parts, ignore_index=True) if judged_parts else pd.DataFrame()
    non_judged_df = (
        pd.concat(non_judged_parts, ignore_index=True) if non_judged_parts else pd.DataFrame()
    )

    if distractor_n > 0 and len(non_judged_df) > 0:
        n_distractors = min(distractor_n, len(non_judged_df))
        distractor_df = non_judged_df.sample(n=n_distractors, random_state=RANDOM_SEED)
    else:
        distractor_df = non_judged_df.iloc[:0]

    catalog = pd.concat([judged_df, distractor_df], ignore_index=True)
    catalog = catalog.drop_duplicates(subset="product_id")
    catalog = catalog.drop(columns=["_pid", "product_locale"], errors="ignore")
    return catalog.to_dict("records")


def coverage_check(judged_ids: set[str], catalog_ids: set[str]) -> float:
    """Return the fraction of judged products present in the catalog."""
    if not judged_ids:
        return 0.0
    return len(judged_ids & catalog_ids) / len(judged_ids)


def index_opensearch(client: httpx.Client, products: list[dict]) -> None:
    """Create OpenSearch index and bulk-index products."""
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}},
            }
        },
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "fields": {"en": {"type": "text", "analyzer": "english"}},
                },
                "brand": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                "bullets": {"type": "text"},
                "description": {"type": "text"},
                "cluster_id": {"type": "integer"},
                "title_tok": {"type": "text", "analyzer": "whitespace"},
            }
        },
    }

    # Delete + recreate
    delete_resp = client.delete(f"{OPENSEARCH_URL}/{INDEX_NAME}")
    if delete_resp.status_code not in (200, 404):
        delete_resp.raise_for_status()
    resp = client.put(f"{OPENSEARCH_URL}/{INDEX_NAME}", json=index_body)
    resp.raise_for_status()

    # Bulk index
    lines = []
    for p in products:
        title = str(p.get("product_title") or "") if pd.notna(p.get("product_title")) else ""
        brand = str(p.get("product_brand") or "") if pd.notna(p.get("product_brand")) else ""
        bullets = (
            str(p.get("product_bullet_point") or "").split("\n")
            if pd.notna(p.get("product_bullet_point"))
            else []
        )
        description = (
            str(p.get("product_description") or "")
            if pd.notna(p.get("product_description"))
            else ""
        )
        doc = {
            "product_id": str(p["product_id"]),
            "title": title,
            "brand": brand,
            "bullets": bullets,
            "description": description,
            "cluster_id": hash(p["product_id"]) % 100,
        }
        lines.append(json.dumps({"index": {"_index": INDEX_NAME, "_id": p["product_id"]}}))
        lines.append(json.dumps(doc))

    # Bulk in chunks of 1000 docs (2000 lines)
    for i in range(0, len(lines), 2000):
        chunk = "\n".join(lines[i : i + 2000]) + "\n"
        resp = client.post(
            f"{OPENSEARCH_URL}/{INDEX_NAME}/_bulk",
            data=chunk,
            headers={"Content-Type": "application/x-ndjson"},
        )
        resp.raise_for_status()

    client.post(f"{OPENSEARCH_URL}/{INDEX_NAME}/_refresh")
    print(f"[index_esci] OpenSearch: indexed {len(products)} products")


def index_qdrant(client: httpx.Client, products: list[dict]) -> None:
    """Create Qdrant collection and upsert product points."""
    # Delete + recreate
    delete_resp = client.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    if delete_resp.status_code not in (200, 404):
        delete_resp.raise_for_status()
    resp = client.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        json={"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}},
    )
    resp.raise_for_status()

    # Upsert in batches
    batch_size = QDRANT_BATCH
    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        points = []
        for p in batch:
            text = f"{p.get('product_title', '')} {p.get('product_brand', '')}"
            vector = _stable_vector(text, VECTOR_SIZE)
            title = str(p.get("product_title") or "") if pd.notna(p.get("product_title")) else ""
            brand = str(p.get("product_brand") or "") if pd.notna(p.get("product_brand")) else ""
            points.append(
                {
                    "id": _point_id(p["product_id"]),
                    "vector": vector,
                    "payload": {
                        "product_id": str(p["product_id"]),
                        "cluster_id": hash(p["product_id"]) % 100,
                        "title": title,
                        "brand": brand,
                    },
                }
            )
        resp = client.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={"points": points},
        )
        resp.raise_for_status()
    print(f"[index_esci] Qdrant: indexed {len(products)} products")


def build_qrels() -> dict:
    """Build qrels from ESCI queries parquet (esci_label E/S/C/I -> grades 3/2/1/0).

    Replaces the previous approach (pos_product_ids from exact_queries files,
    all grade 3) with per-query-product judgments across all four ESCI grades.
    """
    qrels: dict[str, dict[str, int]] = {}
    for f in sorted(glob.glob("data/esci/queries/*.parquet")):
        df = pd.read_parquet(f)
        df = df[df["esci_label"].isin(["E", "S", "C", "I"])]
        df["grade"] = df["esci_label"].map(GRADE_MAP)
        df["query_id"] = df["query_id"].astype(str)
        df["product_id"] = df["product_id"].astype(str)
        # For duplicate (query_id, product_id) keep the highest grade.
        df = df.sort_values("grade", ascending=False).drop_duplicates(
            subset=["query_id", "product_id"], keep="first"
        )
        for qid, group in df.groupby("query_id"):
            qrels[qid] = dict(zip(group["product_id"], group["grade"], strict=False))

    out_path = Path("results/esci_qrels.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(qrels, indent=2))

    grade_counts = dict.fromkeys(range(4), 0)
    for products in qrels.values():
        for grade in products.values():
            grade_counts[grade] += 1
    print(f"[index_esci] Qrels: {len(qrels)} queries, grades {grade_counts} -> {out_path}")
    return qrels


def main() -> None:
    judged_ids = _load_judged_product_ids()
    products = load_products(judged_ids=judged_ids, distractor_n=DISTRACTOR_N)

    catalog_ids = {str(p["product_id"]) for p in products}
    pre_coverage = coverage_check(judged_ids, catalog_ids)
    print(
        f"[index_esci] Loaded {len(products)} products "
        f"({len(judged_ids)} judged + up to {DISTRACTOR_N} distractors)"
    )
    print(f"[index_esci] Pre-index coverage: {pre_coverage:.1%} of judged products in catalog")

    client = httpx.Client(timeout=120.0)
    try:
        index_opensearch(client, products)
        index_qdrant(client, products)
    finally:
        client.close()

    build_qrels()
    print("[index_esci] Done — run 'make run-retrieval' then curl /v1/search")


if __name__ == "__main__":
    main()
