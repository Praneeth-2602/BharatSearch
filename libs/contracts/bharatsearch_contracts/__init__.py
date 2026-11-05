"""Cross-service pydantic contracts.

Services import these models and never redefine them (Implementation Guide,
Section 0, rule 2).
"""

from bharatsearch_contracts.models import (
    EventEnvelope,
    EventType,
    Intent,
    Language,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Understanding,
)

__all__ = [
    "EventEnvelope",
    "EventType",
    "Intent",
    "Language",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "Understanding",
]
