"""Search event pipeline (Implementation Guide, Section 6).

Publishes search.query, search.impression, and search.click events using
the EventEnvelope contract. Defaults to a no-op producer for CI/dev; wire
a Kafka producer via ``set_producer()`` when a broker is available.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from bharatsearch_contracts import (
    EventEnvelope,
    EventType,
    SearchRequest,
    SearchResponse,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ImpressionRequest(BaseModel):
    """Request body for recording an impression event."""

    session_id: str
    request_id: str
    product_ids: list[str]


class ClickRequest(BaseModel):
    """Request body for recording a click event."""

    session_id: str
    request_id: str
    product_id: str
    position: int | None = None


class EventProducer(ABC):
    """Abstract event producer."""

    @abstractmethod
    def publish(self, envelope: EventEnvelope) -> None:
        """Publish an event envelope."""


class NoOpEventProducer(EventProducer):
    """No-op producer that logs events (default for CI/dev)."""

    def publish(self, envelope: EventEnvelope) -> None:
        logger.debug("event: type=%s id=%s", envelope.event_type, envelope.event_id)


_PRODUCER_BOX: list[EventProducer | None] = [None]


def set_producer(producer: EventProducer) -> None:
    """Set the global event producer (mainly for tests and Kafka wiring)."""
    _PRODUCER_BOX[0] = producer


def get_producer() -> EventProducer:
    """Return the global producer, lazily creating a NoOp producer."""
    if _PRODUCER_BOX[0] is None:
        _PRODUCER_BOX[0] = NoOpEventProducer()
    return _PRODUCER_BOX[0]


def reset_producer() -> None:
    """Reset the global producer (mainly for tests)."""
    _PRODUCER_BOX[0] = None


def _envelope(
    event_type: EventType,
    session_id: str,
    request_id: str,
    payload: dict[str, Any],
) -> EventEnvelope:
    return EventEnvelope(
        event_type=event_type,
        session_id=session_id,
        request_id=request_id,
        payload=payload,
    )


def publish_query_event(req: SearchRequest, resp: SearchResponse) -> None:
    """Publish a search.query event from a search request/response pair."""
    producer = get_producer()
    envelope = _envelope(
        EventType.query,
        req.session_id,
        resp.request_id,
        {
            "query": req.query,
            "corrected_query": resp.understanding.corrected_query,
            "language": resp.understanding.language,
            "intent": resp.understanding.intent,
            "brand": resp.understanding.brand,
            "k": req.k,
            "n_results": len(resp.results),
        },
    )
    producer.publish(envelope)


def publish_impression_event(session_id: str, request_id: str, product_ids: list[str]) -> None:
    """Publish a search.impression event."""
    producer = get_producer()
    envelope = _envelope(
        EventType.impression,
        session_id,
        request_id,
        {
            "product_ids": product_ids,
            "n_results": len(product_ids),
        },
    )
    producer.publish(envelope)


def publish_click_event(
    session_id: str,
    request_id: str,
    product_id: str,
    position: int | None = None,
) -> None:
    """Publish a search.click event."""
    producer = get_producer()
    envelope = _envelope(
        EventType.click,
        session_id,
        request_id,
        {
            "product_id": product_id,
            "position": position,
        },
    )
    producer.publish(envelope)
