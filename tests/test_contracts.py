"""Unit tests for cross-service contracts (libs/contracts)."""

from bharatsearch_contracts import (
    EventEnvelope,
    EventType,
    Intent,
    Language,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Understanding,
)


def test_search_request_defaults():
    req = SearchRequest(query="fone cover sasta", session_id="s1")
    assert req.query == "fone cover sasta"
    assert req.session_id == "s1"
    assert req.k == 20
    assert req.debug is False


def test_search_request_explicit_k():
    req = SearchRequest(query="phone cover", session_id="s1", k=10, debug=True)
    assert req.k == 10
    assert req.debug is True


def test_search_response_round_trip():
    understanding = Understanding(
        corrected_query="phone cover",
        language=Language.en,
        intent=Intent.product,
        brand=None,
        source="fallback",
    )
    resp = SearchResponse(
        request_id="r1",
        results=[SearchResult(product_id="p1", score=0.9, rank=1)],
        understanding=understanding,
        timing_ms={"qu": 5.0, "retrieval": 12.0},
        model_versions={"embedder": "e5-small"},
    )
    assert resp.request_id == "r1"
    assert resp.results[0].product_id == "p1"
    assert resp.understanding.source == "fallback"


def test_event_envelope_defaults():
    env = EventEnvelope(
        event_type=EventType.query,
        session_id="s1",
        request_id="r1",
        payload={"query": "fone cover"},
    )
    assert env.schema_version == 1
    assert str(env.event_id)
    assert env.event_type == EventType.query
