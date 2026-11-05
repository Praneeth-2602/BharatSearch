"""Resilient Kafka consumer utilities (Implementation Guide, Section 6.3).

OffsetTracker tracks per-partition completion so commits only advance a
contiguous processed prefix, never skipping in-flight work. The Consumer
framework wraps OffsetTracker with a poll loop, key-hash partitioned worker
dispatch, backpressure, retry/DLQ handling, and rebalance safety.
"""

from __future__ import annotations

from bharatsearch_kafka_utils.consumer import (
    DLQ_TOPIC,
    RETRY_TOPIC_5S,
    RETRY_TOPIC_30S,
    Consumer,
    ConsumerConfig,
    MessageHandlingError,
    OffsetTracker,
)

__all__ = [
    "Consumer",
    "ConsumerConfig",
    "DLQ_TOPIC",
    "MessageHandlingError",
    "OffsetTracker",
    "RETRY_TOPIC_30S",
    "RETRY_TOPIC_5S",
]
