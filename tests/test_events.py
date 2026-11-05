"""Unit tests for the search event pipeline (no Kafka required)."""

from __future__ import annotations

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
from bharatsearch_retrieval.events import (
    ClickRequest,
    ImpressionRequest,
    NoOpEventProducer,
    get_producer,
    publish_click_event,
    publish_impression_event,
    publish_query_event,
    reset_producer,
    set_producer,
)


class _CaptureProducer:
    """Captures published envelopes for assertions."""

    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    def publish(self, envelope: EventEnvelope) -> None:
        self.events.append(envelope)


# --- Query event ------------------------------------------------------------


def test_publish_query_event():
    capture = _CaptureProducer()
    set_producer(capture)
    try:
        req = SearchRequest(query="phone cover", session_id="s1", user_id="u1", k=10)
        resp = SearchResponse(
            request_id="r1",
            results=[SearchResult(product_id="p1", score=0.9, rank=1)],
            understanding=Understanding(
                corrected_query="phone cover",
                language=Language.en,
                intent=Intent.product,
                brand=None,
                source="rule-based",
            ),
            timing_ms={"bm25_ms": 5.0},
            model_versions={},
        )
        publish_query_event(req, resp)
        assert len(capture.events) == 1
        env = capture.events[0]
        assert env.event_type == EventType.query
        assert env.session_id == "s1"
        assert env.request_id == "r1"
        assert env.payload["query"] == "phone cover"
        assert env.payload["corrected_query"] == "phone cover"
        assert env.payload["language"] == "en"
        assert env.payload["intent"] == "product"
        assert env.payload["brand"] is None
        assert env.payload["k"] == 10
        assert env.payload["n_results"] == 1
        assert env.event_id
        assert env.ts
    finally:
        reset_producer()


# --- Impression event -------------------------------------------------------


def test_publish_impression_event():
    capture = _CaptureProducer()
    set_producer(capture)
    try:
        publish_impression_event("s1", "r1", ["p1", "p2", "p3"])
        assert len(capture.events) == 1
        env = capture.events[0]
        assert env.event_type == EventType.impression
        assert env.session_id == "s1"
        assert env.request_id == "r1"
        assert env.payload["product_ids"] == ["p1", "p2", "p3"]
        assert env.payload["n_results"] == 3
        assert env.event_id
    finally:
        reset_producer()


# --- Click event ------------------------------------------------------------


def test_publish_click_event():
    capture = _CaptureProducer()
    set_producer(capture)
    try:
        publish_click_event("s1", "r1", "p2", position=2)
        assert len(capture.events) == 1
        env = capture.events[0]
        assert env.event_type == EventType.click
        assert env.session_id == "s1"
        assert env.request_id == "r1"
        assert env.payload["product_id"] == "p2"
        assert env.payload["position"] == 2
        assert env.event_id
    finally:
        reset_producer()


def test_publish_click_event_no_position():
    capture = _CaptureProducer()
    set_producer(capture)
    try:
        publish_click_event("s1", "r1", "p2")
        assert len(capture.events) == 1
        env = capture.events[0]
        assert env.payload["product_id"] == "p2"
        assert env.payload["position"] is None
    finally:
        reset_producer()


# --- Producer lifecycle -----------------------------------------------------


def test_noop_producer_does_not_raise():
    producer = NoOpEventProducer()
    env = EventEnvelope(
        event_type=EventType.query,
        session_id="s1",
        request_id="r1",
        payload={"query": "phone cover"},
    )
    producer.publish(env)  # should not raise


def test_set_get_reset_producer():
    capture = _CaptureProducer()
    set_producer(capture)
    assert get_producer() is capture
    reset_producer()
    # After reset, a new NoOp producer is created lazily
    producer = get_producer()
    assert isinstance(producer, NoOpEventProducer)
    assert producer is not capture
    reset_producer()


# --- Request models ---------------------------------------------------------


def test_impression_request():
    req = ImpressionRequest(session_id="s1", request_id="r1", product_ids=["p1", "p2"])
    assert req.session_id == "s1"
    assert req.request_id == "r1"
    assert req.product_ids == ["p1", "p2"]


def test_click_request():
    req = ClickRequest(session_id="s1", request_id="r1", product_id="p2", position=3)
    assert req.session_id == "s1"
    assert req.request_id == "r1"
    assert req.product_id == "p2"
    assert req.position == 3


def test_click_request_no_position():
    req = ClickRequest(session_id="s1", request_id="r1", product_id="p2")
    assert req.position is None
