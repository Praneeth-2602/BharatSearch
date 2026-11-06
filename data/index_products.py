"""Index sample products into OpenSearch and Qdrant for local development.

Creates the `products` index (OpenSearch BM25) and `products` collection
(Qdrant dense) with a small seed catalog so the retrieval service can serve
/search without needing the full ESCI dataset.
"""

from __future__ import annotations

import hashlib
import math
import random

import httpx
from bharatsearch_retrieval.opensearch import INDEX_BODY, INDEX_NAME
from bharatsearch_retrieval.qdrant import COLLECTION_NAME, DEFAULT_VECTOR_SIZE

OPENSEARCH_URL = "http://localhost:9200"
QDRANT_URL = "http://localhost:6333"

SAMPLE_PRODUCTS = [
    {
        "product_id": "p1",
        "title": "phone cover",
        "brand": "Generic",
        "bullets": ["Shockproof", "Slim fit"],
        "description": "Protective phone cover for smartphones",
        "cluster_id": 1,
    },
    {
        "product_id": "p2",
        "title": "shoes",
        "brand": "Generic",
        "bullets": ["Running", "Comfortable"],
        "description": "Running shoes for daily use",
        "cluster_id": 2,
    },
    {
        "product_id": "p3",
        "title": "water bottle",
        "brand": "Generic",
        "bullets": ["Leakproof", "500ml"],
        "description": "Reusable water bottle for gym and office",
        "cluster_id": 3,
    },
    {
        "product_id": "p4",
        "title": "fone cover sasta",
        "brand": "Generic",
        "bullets": ["Cheap", "Basic"],
        "description": "Budget phone cover",
        "cluster_id": 1,
    },
    {
        "product_id": "p5",
        "title": "smartphone case",
        "brand": "Generic",
        "bullets": ["Durable", "Slim"],
        "description": "Protective smartphone case",
        "cluster_id": 1,
    },
]


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


def create_opensearch_index(client: httpx.Client) -> None:
    # Delete existing index if present, then recreate
    delete_resp = client.delete(f"{OPENSEARCH_URL}/{INDEX_NAME}")
    if delete_resp.status_code not in (200, 404):
        delete_resp.raise_for_status()
    resp = client.put(f"{OPENSEARCH_URL}/{INDEX_NAME}", json=INDEX_BODY)
    resp.raise_for_status()
    for p in SAMPLE_PRODUCTS:
        resp = client.post(
            f"{OPENSEARCH_URL}/{INDEX_NAME}/_doc/{p['product_id']}",
            json=p,
        )
        resp.raise_for_status()
    client.post(f"{OPENSEARCH_URL}/{INDEX_NAME}/_refresh")
    print(
        f"[index_products] OpenSearch: created index '{INDEX_NAME}'"
        f" with {len(SAMPLE_PRODUCTS)} products"
    )


def create_qdrant_collection(client: httpx.Client) -> None:
    delete_resp = client.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    if delete_resp.status_code not in (200, 404):
        delete_resp.raise_for_status()
    resp = client.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        json={"vectors": {"size": DEFAULT_VECTOR_SIZE, "distance": "Cosine"}},
    )
    resp.raise_for_status()
    points = []
    for p in SAMPLE_PRODUCTS:
        vector = _stable_vector(p["title"], DEFAULT_VECTOR_SIZE)
        points.append(
            {
                "id": _point_id(p["product_id"]),
                "vector": vector,
                "payload": {
                    "product_id": p["product_id"],
                    "cluster_id": p["cluster_id"],
                },
            }
        )
    resp = client.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
        json={"points": points},
    )
    resp.raise_for_status()
    print(
        f"[index_products] Qdrant: created collection '{COLLECTION_NAME}' with {len(points)} points"
    )


def main() -> None:
    client = httpx.Client(timeout=10.0)
    try:
        create_opensearch_index(client)
        create_qdrant_collection(client)
        print("[index_products] Indexing complete — run 'make run-retrieval' then curl /v1/search")
    finally:
        client.close()


if __name__ == "__main__":
    main()
