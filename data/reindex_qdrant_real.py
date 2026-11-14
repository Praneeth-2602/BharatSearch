"""Re-index Qdrant with REAL passage vectors from a sentence-transformers model.

Replaces the stub (deterministic pseudo-random) vectors in Qdrant with real
``intfloat/multilingual-e5-small`` passage embeddings so dense search is
semantic rather than random. Accepts an optional product list (slice catalog);
otherwise loads all products via ``data.index_esci.load_products``.

Encoding and upserting stream per chunk to keep memory bounded, the model runs
on Apple Silicon (MPS) when available, and embedding/upserting are pipelined
across two threads so neither waits on the other.

Usage:
    REAL_VECTORS=1 uv run python -m data.reindex_qdrant_real        # full catalog
    RECREATE=1 SLICE=1 uv run python -m data.reindex_qdrant_real    # slice catalog
"""

from __future__ import annotations

import os
import threading
from queue import Queue

import httpx
import torch
from sentence_transformers import SentenceTransformer

from data.index_esci import _point_id, build_slice_catalog, load_products, sample_slice_queries

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "products"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
ENCODE_BATCH = int(os.environ.get("REINDEX_ENCODE_BATCH", "256"))
UPSERT_BATCH = int(os.environ.get("REINDEX_UPSERT_BATCH", "256"))
MAX_PASSAGE_LEN = int(os.environ.get("REINDEX_MAX_PASSAGE_LEN", "128"))
RECREATE = os.environ.get("RECREATE", "0") == "1"
SLICE = os.environ.get("SLICE", "0") == "1"
SLICE_TARGET = int(os.environ.get("SLICE_TARGET", "100000"))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


def passage_text(p: dict) -> str:
    """Build the passage text for a product (title + brand + first bullet, truncated)."""
    title = str(p.get("product_title") or "")
    brand = str(p.get("product_brand") or "")
    bullets = p.get("product_bullet_point") or ""
    first_bullet = str(bullets).split("\n")[0] if bullets else ""
    text = f"passage: {title} {brand} {first_bullet}".strip()
    return text[:MAX_PASSAGE_LEN]


def iter_chunks(seq, size):
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
    return upserted


def encode_producer(products: list[dict], queue: Queue) -> None:
    """Encode passage vectors in chunks and enqueue them for upserting.

    Products are sorted by passage length so each batch has similar-length
    inputs (less padding waste).
    """
    ordered = sorted(products, key=lambda p: len(passage_text(p)))
    for i in range(0, len(ordered), ENCODE_BATCH):
        chunk = ordered[i : i + ENCODE_BATCH]
        texts = [passage_text(p) for p in chunk]
        vectors = MODEL.encode(
            texts,
            batch_size=ENCODE_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        queue.put((chunk, vectors.tolist()))
    queue.put(None)  # sentinel


def upsert_consumer(client: httpx.Client, queue: Queue, total: int) -> int:
    """Dequeue encoded chunks and upsert them to Qdrant."""
    upserted = 0
    while True:
        item = queue.get()
        if item is None:
            break
        chunk, vectors = item
        upserted += upsert_chunk(client, chunk, vectors)
        print(f"[reindex_qdrant_real] encoded+upserted {upserted}/{total}", flush=True)
    return upserted


def main() -> None:
    if SLICE:
        bench_qids, dev_qids, test_qids = sample_slice_queries()
        slice_qids = bench_qids | dev_qids | test_qids
        products = build_slice_catalog(slice_qids, SLICE_TARGET)
    else:
        products = load_products()
    total = len(products)
    print(
        f"[reindex_qdrant_real] {total} products; encoding "
        f"{EMBEDDING_MODEL} passage vectors on {DEVICE}"
    )

    if RECREATE:
        client = httpx.Client(timeout=120.0)
        try:
            delete_resp = client.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
            if delete_resp.status_code not in (200, 404):
                delete_resp.raise_for_status()
            resp = client.put(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
                json={"vectors": {"size": 384, "distance": "Cosine"}},
            )
            resp.raise_for_status()
            print(f"[reindex_qdrant_real] recreated collection {COLLECTION_NAME}")
        finally:
            client.close()

    client = httpx.Client(timeout=120.0)
    try:
        queue: Queue = Queue(maxsize=4)
        producer_thread = threading.Thread(
            target=encode_producer, args=(products, queue), daemon=True
        )
        producer_thread.start()
        upserted = upsert_consumer(client, queue, total)
        producer_thread.join()
    finally:
        client.close()
    print(f"[reindex_qdrant_real] done — upserted {upserted} real vectors to {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
