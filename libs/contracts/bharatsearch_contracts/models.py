"""Shared request/response and event-envelope contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Language(str, Enum):
    en = "en"
    hi_latn = "hi-Latn"
    hi = "hi"
    mixed = "mixed"
    other = "other"


class Intent(str, Enum):
    product = "product"
    brand = "brand"
    attribute = "attribute"
    unknown = "unknown"


class Understanding(BaseModel):
    """Output of query understanding."""

    corrected_query: str
    language: Language
    intent: Intent
    brand: Optional[str] = None
    source: str  # "llm" | "cache" | "fallback"


class SearchResult(BaseModel):
    product_id: str
    score: float
    rank: int


class SearchRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    k: int = 20
    debug: bool = False


class SearchResponse(BaseModel):
    request_id: str
    results: list[SearchResult]
    understanding: Understanding
    timing_ms: dict[str, float]
    model_versions: dict[str, str]


class EventType(str, Enum):
    query = "search.queries"
    impression = "search.impressions"
    click = "search.clicks"
    shadow_diff = "search.shadow_diffs"
    dlq = "search.dlq"


class EventEnvelope(BaseModel):
    """Envelope carried on every Kafka event (Implementation Guide, Section 6.2)."""

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    schema_version: int = 1
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str
    request_id: str
    payload: dict


class EmbedRequest(BaseModel):
    """Request to embed one or more texts (Implementation Guide, Section 3.2)."""

    texts: list[str]
    input_type: str = "query"  # "query" | "passage"
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    """Response from the embedding service."""

    embeddings: list[list[float]]
    model: str
    usage: dict[str, int]
