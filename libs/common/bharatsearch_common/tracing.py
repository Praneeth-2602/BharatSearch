"""OpenTelemetry tracing setup skeleton.

One trace per search request; context propagated over HTTP and Kafka headers.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider


def setup_tracing(service_name: str) -> trace.TracerProvider:
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    return provider
