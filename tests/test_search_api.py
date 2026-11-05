"""Unit tests for /v1/search contract with mocked OpenSearch/Qdrant/embedding
backends. No services required."""

from __future__ import annotations

import pytest
from bharatsearch_contracts import SearchRequest
from bharatsearch_retrieval.app.main import _State, search


class _FakeResp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _make_http(bm25_hits, embed_vec, dense_points):
    async def post(url, json=None, **kwargs):
        url_s = str(url)
        if "/embed" in url_s:
            return _FakeResp({"embeddings": [embed_vec]})
        if "/_search" in url_s:
            return _FakeResp({"hits": {"hits": bm25_hits}})
        if "/points/search" in url_s:
            return _FakeResp({"result": dense_points})
        return _FakeResp({})

    class _MockHttp:
        pass

    http = _MockHttp()
    http.post = post  # instance attribute: not bound as a method
    return http


@pytest.mark.anyio
async def test_search_contract_and_timing():
    _State.ready = True
    _State.http = _make_http(
        bm25_hits=[{"_source": {"product_id": "p1"}, "_score": 0.9}],
        embed_vec=[0.1] * 384,
        dense_points=[{"payload": {"product_id": "p2"}, "score": 0.8}],
    )
    req = SearchRequest(query="phone cover", session_id="s1", user_id="u1", k=10)
    resp = await search(req)
    assert resp.request_id
    assert len(resp.results) > 0
    assert resp.understanding.source == "fallback"
    assert "bm25_ms" in resp.timing_ms
    assert "dense_ms" in resp.timing_ms
    assert "fuse_ms" in resp.timing_ms
    assert "total_ms" in resp.timing_ms
    assert resp.model_versions["embedder"] == "stub"
    assert resp.model_versions["ranker"] == "rrf"
    assert resp.model_versions["llm"] == "none"


@pytest.mark.anyio
async def test_search_not_ready():
    _State.ready = False
    req = SearchRequest(query="x", session_id="s1")
    with pytest.raises(Exception) as exc_info:
        await search(req)
    assert getattr(exc_info.value, "status_code", None) == 503
