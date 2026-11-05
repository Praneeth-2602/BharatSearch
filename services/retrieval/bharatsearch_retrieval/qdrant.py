"""Qdrant collection management and dense search (Implementation Guide, Section 3.2).

One collection, cosine distance, vector size set by the embedding model
(384 for multilingual-e5-small, 1024 for BGE-M3; verify). Payload carries
product_id and cluster_id. e5 models expect `query: ` / `passage: ` prefixes;
BGE-M3 does not — the prefix rule lives in the model config, not call sites.
"""

from __future__ import annotations

COLLECTION_NAME = "products"
DEFAULT_VECTOR_SIZE = 384  # multilingual-e5-small; verify against pinned model


def collection_config(vector_size: int = DEFAULT_VECTOR_SIZE) -> dict:
    return {
        "vectors": {"size": vector_size, "distance": "Cosine"},
        "payload_schema": ["product_id", "cluster_id"],
    }
