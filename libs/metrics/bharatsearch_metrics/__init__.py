"""Prometheus metrics helpers."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter("bharatsearch_requests_total", "Total requests", ["service", "endpoint"])
REQUEST_LATENCY = Histogram(
    "bharatsearch_request_duration_seconds",
    "Request latency in seconds",
    ["service", "endpoint"],
)
