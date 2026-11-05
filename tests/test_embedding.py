"""Unit tests for the embedding service (stub backend): determinism, LRU cache,
e5 prefixes, batching, never-empty output. No services or models required."""

from __future__ import annotations

from bharatsearch_embedding.app.main import (
    EMBEDDING_DIM,
    _cache_key,
    _embed_texts,
    _LRUCache,
    _prefix,
    _stable_vector,
    _State,
)


def test_stub_vector_dim_and_unit_norm():
    vec = _stable_vector("phone cover", EMBEDDING_DIM)
    assert len(vec) == EMBEDDING_DIM
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_stub_vector_deterministic():
    assert _stable_vector("query", EMBEDDING_DIM) == _stable_vector("query", EMBEDDING_DIM)


def test_stub_vector_different_inputs_differ():
    assert _stable_vector("a", EMBEDDING_DIM) != _stable_vector("b", EMBEDDING_DIM)


def test_e5_prefixes():
    assert _prefix("query", "phone cover") == "query: phone cover"
    assert _prefix("passage", "phone cover") == "passage: phone cover"
    assert _prefix("other", "phone cover") == "query: phone cover"


def test_cache_key_includes_input_type():
    assert _cache_key("query", "text") != _cache_key("passage", "text")


def test_lru_cache_put_get_eviction():
    cache = _LRUCache(capacity=2)
    cache.put("a", [1.0])
    cache.put("b", [2.0])
    assert cache.get("a") == [1.0]  # a: MRU
    assert cache.get("b") == [2.0]  # b: MRU, a: now LRU
    # Adding "c" evicts "a" (LRU), not "b" (MRU).
    cache.put("c", [3.0])
    assert cache.get("a") is None  # evicted
    assert cache.get("b") == [2.0]  # still present (MRU)
    assert cache.get("c") == [3.0]


def test_lru_cache_zero_capacity():
    cache = _LRUCache(capacity=0)
    assert cache.get("a") is None
    cache.put("a", [1.0])  # no-op
    assert cache.get("a") is None


def test_embed_texts_stub_never_empty():
    _State.backend = "stub"
    _State.model = None
    vecs = _embed_texts(["a", "b", "c"], "query")
    assert len(vecs) == 3
    assert all(len(v) == EMBEDDING_DIM for v in vecs)
    assert all(abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-6 for v in vecs)
