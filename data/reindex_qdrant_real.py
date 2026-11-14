"""Re-index Qdrant with REAL passage vectors from a sentence-transformers model.

Replaces the stub (deterministic pseudo-random) vectors in Qdrant with real
``intfloat/multilingual-e5-small`` passage embeddings so dense search is
semantic rather than random. Reuses the catalog from ``data.index_esci.load_products``
(judged products + distractors), so the point IDs match what's already in
OpenSearch / Qdrant.

The collection is not recreated; vectors are upserted in place for the existing
point IDs (deterministic ``_point_id``). Encoding and upserting stream per chunk
to keep memory bounded, and the model runs on Apple Silicon (MPS) when available.

Usage:
    REAL_VECTORS=1 uv run python -m data.reindex_qdrant_real
"""

from __future__ import annotations

import os

import httpx
import torch
from sentence_transformers import SentenceTransformer

from data.index_esci import _point_id, load_products

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "products"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
ENCODE_BATCH = int(os.environ.get("REINDEX_ENCODE_BATCH", "1024"))
UPSERT_BATCH = int(os.environ.get("REINDEX_UPSERT_BATCH", "1024"))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


def passage_text(p: dict) -> str:
    """Build the passage text for a product (mirrors e5 indexing convention)."""
    title = str(p.get("product_title") or "")
    brand = str(p.get("product_brand") or "")
    return f"passage: {title} {brand}".strip()


def iter_chunks(seq: list, size: int) -> list[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def upsert_chunk(client: httpx.Client, products: list[dict], vectors: list[list[float]]) -> int:
    """Upsert (product_id -> vector) to Qdrant in batches; returns count upserted."""
    upserted = 0
    for prod_chunk, vec_chunk in zip(
        iter_chunks(products, UPSERT_BATCH), iter_chunks(vectors, UPSERT_BATCH), strict=False
    ):
        points = [
            {
                "id": _point_id(str(p["product_id"])),
                "vector": vec,
                "payload": {
                    "product_id": str(p["product_id"]),
                    "title": str(p.get("product_title") or ""),
                    "brand": str(p.get("product_brand") or ""),
                },
            }
            for p, vec in zip(prod_chunk, vec_chunk, strict=False)
        ]
        resp = client.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={"points": points},
        )
        resp.raise_for_status()
        upserted += len(points)
        print(f"[reindex_qdrant_real] upserted {upserted}", flush=True)
    return upserted


def main() -> None:
    products = load_products()
    print(
        f"[reindex_qdrant_real] {len(products)} products; "
        f"encoding {EMBEDDING_MODEL} passage vectors on {DEVICE}"
    )

    client = httpx.Client(timeout=120.0)
    try:
        upserted = 0
        for chunk_start in range(0, len(products), ENCODE_BATCH):
            chunk = products[chunk_start : chunk_start + ENCODE_BATCH]
            texts = [passage_text(p) for p in chunk]
            vectors = MODEL.encode(
                texts,
                batch_size=ENCODE_BATCH,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            upserted += upsert_chunk(client, chunk, vectors.tolist())
            print(f"[reindex_qdrant_real] encoded+upserted {upserted}/{len(products)}", flush=True)
    finally:
        client.close()
    print(f"[reindex_qdrant_real] done — upserted {upserted} real vectors to {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
