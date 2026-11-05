"""OpenSearch index management and BM25 search (Implementation Guide, Section 3.1).

Index schema:
  product_id (keyword), title (text, en sub-field), brand (text + raw keyword),
  bullets (text), description (text), cluster_id (integer), title_tok (text,
  whitespace analyzer, reserved for sarvam1 pre-tokenized text).
"""

from __future__ import annotations

INDEX_NAME = "products"

INDEX_BODY = {
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

# Query fields / boosts (kept in one function so analyzer experiments swap fields)
SEARCH_FIELDS = ["title^3", "title.en^2", "brand^2", "bullets", "description^0.5"]


def bm25_query(query_text: str, k: int = 20) -> dict:
    return {
        "size": k,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": SEARCH_FIELDS,
            }
        },
    }
