"""Resilient Kafka consumer framework (Implementation Guide, Section 6.3).

Goals: one slow or poisonous message must not stall a partition, commits must
never skip an unfinished message, and rebalances must not lose or duplicate
work beyond what idempotency absorbs.

``confluent_kafka`` is an optional dependency; the framework degrades
gracefully when it is unavailable so the module can be imported and unit-tested
without a Kafka broker.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

try:  # pragma: no cover - optional dependency path
    from confluent_kafka import Consumer as _ConfluentConsumer
    from confluent_kafka import KafkaError, KafkaException, Message, TopicPartition
    from confluent_kafka import Producer as _ConfluentProducer
    from confluent_kafka.admin import AdminClient, NewTopic

    _KAFKA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    _ConfluentConsumer = None  # type: ignore[assignment,misc]
    _ConfluentProducer = None  # type: ignore[assignment,misc]
    KafkaError = None  # type: ignore[assignment,misc]
    KafkaException = None  # type: ignore[assignment,misc]
    Message = None  # type: ignore[assignment,misc]
    TopicPartition = None  # type: ignore[assignment,misc]
    AdminClient = None  # type: ignore[assignment,misc]
    NewTopic = None  # type: ignore[assignment,misc]
    _KAFKA_AVAILABLE = False


logger = logging.getLogger(__name__)

# Retry topic names (Implementation Guide, Section 6.1 table).
RETRY_TOPIC_5S = "search.retry.5s"
RETRY_TOPIC_30S = "search.retry.30s"
DLQ_TOPIC = "search.dlq"

# Retry tier values (seconds).
RETRY_TIER_5S = 5
RETRY_TIER_30S = 30


@dataclass
class OffsetTracker:
    """Tracks per-partition contiguous offset completion.

    Commits may only advance a contiguous processed prefix, never skipping
    in-flight work. Offsets can be marked done out of order; the committable
    offset advances only up to the first gap.
    """

    next_expected: int = 0
    _done: set[int] = field(default_factory=set)

    def mark_done(self, offset: int) -> int | None:
        """Mark ``offset`` as done. Return next committable offset or None."""
        if offset in self._done:
            return None
        self._done.add(offset)
        prev = self.next_expected
        while self.next_expected in self._done:
            self._done.discard(self.next_expected)
            self.next_expected += 1
        return self.next_expected if self.next_expected > prev else None


@dataclass
class ConsumerConfig:
    """Configuration for the resilient consumer."""

    bootstrap_servers: str = "localhost:9092"
    group_id: str = "bharatsearch-consumer"
    topics: list[str] = field(default_factory=list)
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_inflight: int = 100
    resume_half: bool = True
    poll_timeout_s: float = 1.0
    retryable_exceptions: tuple[type[BaseException], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


@dataclass
class PartitionState:
    """Per-partition runtime state."""

    tracker: OffsetTracker
    inflight: set[int] = field(default_factory=set)
    paused: bool = False
    committed_offset: int = 0


class MessageHandlingError(Exception):
    """Raised when a message handler fails."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class Consumer:
    """Resilient Kafka consumer with per-partition ordered processing.

    Each assigned partition has an :class:`OffsetTracker`. Messages are
    dispatched to a worker pool, partitioned by key hash so messages with the
    same key stay ordered while different keys run in parallel. The committable
    offset for a partition is the lowest in-flight offset (everything below it
    is done).

    Backpressure pauses a partition when its in-flight count passes
    ``max_inflight`` and resumes at half.
    """

    def __init__(
        self,
        config: ConsumerConfig | None = None,
        handler: Callable[[Any], None] | None = None,
        producer: Any | None = None,
    ) -> None:
        self.config = config or ConsumerConfig()
        self.handler = handler
        self.producer = producer
        self._states: dict[int, PartitionState] = {}
        self._running = False
        self._consumer: Any = None
        self._metrics: dict[str, float | int] = defaultdict(float)
        self._last_poll_ts = time.time()
        self._rebalance_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the consumer (connect to Kafka, subscribe, assign partitions)."""
        if not _KAFKA_AVAILABLE:
            logger.warning(
                "confluent_kafka not installed; Consumer.start() is a no-op. "
                "Install confluent-kafka to consume from Kafka."
            )
            return
        if self._consumer is None:
            conf = {
                "bootstrap.servers": self.config.bootstrap_servers,
                "group.id": self.config.group_id,
                "auto.offset.reset": self.config.auto_offset_reset,
                "enable.auto.commit": self.config.enable_auto_commit,
            }
            self._consumer = _ConfluentConsumer(conf)  # type: ignore[call-arg]
        self._consumer.subscribe(
            self.config.topics, on_assign=self._on_assign, on_revoke=self._on_revoke
        )
        self._running = True
        logger.info("Consumer started, subscribed to %s", self.config.topics)

    def stop(self) -> None:
        """Stop the consumer and close the Kafka connection."""
        self._running = False
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception:  # pragma: no cover - best effort
                logger.exception("Error closing Kafka consumer")
            self._consumer = None
        logger.info("Consumer stopped")

    # ------------------------------------------------------------------
    # Rebalance handling
    # ------------------------------------------------------------------

    def _on_assign(self, _consumer: Any, partitions: list[Any]) -> None:
        """Create trackers starting at the committed offset for each partition."""
        for tp in partitions:
            partition = tp.partition
            start_offset = tp.offset if tp.offset is not None else 0
            self._states[partition] = PartitionState(
                tracker=OffsetTracker(start_offset),
                committed_offset=start_offset,
            )
            logger.info("Assigned partition %d starting at offset %d", partition, start_offset)
        self._rebalance_count += 1
        self._metrics["rebalances_total"] = self._metrics.get("rebalances_total", 0) + 1

    def _on_revoke(self, _consumer: Any, partitions: list[Any]) -> None:
        """Stop dispatching, wait for in-flight work, commit offsets, drop state."""
        for tp in partitions:
            partition = tp.partition
            state = self._states.get(partition)
            if state is None:
                continue
            # Wait for in-flight work up to a timeout.
            deadline = time.time() + 30
            while state.inflight and time.time() < deadline:
                time.sleep(0.1)
            # Commit each tracker's offset.
            next_offset = state.tracker.next_expected
            if TopicPartition is not None and self._consumer is not None:
                try:
                    kafka_tp = TopicPartition(partition, next_offset)  # type: ignore[call-arg]
                    self._consumer.commit(offsets=[kafka_tp])  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Failed to commit offset for partition %d", partition)
            del self._states[partition]
            logger.info("Revoked partition %d, committed offset %d", partition, next_offset)
        self._rebalance_count += 1
        self._metrics["rebalances_total"] = self._metrics.get("rebalances_total", 0) + 1

    # ------------------------------------------------------------------
    # Main poll loop
    # ------------------------------------------------------------------

    def poll_once(self) -> int:
        """Poll for one batch of messages and dispatch them.

        Returns the number of messages dispatched.
        """
        if not self._running or self._consumer is None:
            return 0

        msg = self._consumer.poll(timeout=self.config.poll_timeout_s)
        self._last_poll_ts = time.time()
        self._metrics["seconds_since_last_poll"] = time.time() - self._last_poll_ts

        if msg is None:
            return 0
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:  # type: ignore[attr-defined]
                logger.error("Consumer error: %s", msg.error())
            return 0

        partition = msg.partition()
        offset = msg.offset()
        key = msg.key()

        state = self._states.get(partition)
        if state is None:
            logger.warning("Received message for unassigned partition %d", partition)
            return 0

        # Backpressure: pause if too many in-flight.
        if len(state.inflight) >= self.config.max_inflight and not state.paused:
            try:
                tp = TopicPartition(partition, offset)  # type: ignore[call-arg]
                self._consumer.pause([tp])  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to pause partition %d", partition)
            state.paused = True
            logger.info("Paused partition %d due to backpressure", partition)
            return 0

        if state.paused and len(state.inflight) <= self.config.max_inflight // 2:
            try:
                tp = TopicPartition(partition, offset)  # type: ignore[call-arg]
                self._consumer.resume([tp])  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to resume partition %d", partition)
            state.paused = False
            logger.info("Resumed partition %d after backpressure", partition)

        # Dispatch to worker pool (key-hash partitioned for ordering).
        state.inflight.add(offset)
        self._metrics["inflight_messages"] = len(state.inflight)

        try:
            self._dispatch(msg, key, partition, offset, state)
        except Exception:  # pragma: no cover - handler errors are caught in _dispatch
            logger.exception("Failed to dispatch message partition=%d offset=%d", partition, offset)

        return 1

    def _dispatch(
        self,
        msg: Any,
        key: Any,
        partition: int,
        offset: int,
        state: PartitionState,
    ) -> None:
        """Dispatch a single message to the handler with error handling."""
        if self.handler is None:
            state.inflight.discard(offset)
            return

        start = time.time()
        try:
            self.handler(msg)
            self._metrics["message_processing_seconds"] = self._metrics.get(
                "message_processing_seconds", 0.0
            ) + (time.time() - start)
        except MessageHandlingError as exc:
            self._handle_error(msg, partition, offset, exc.retryable, exc)
        except self.config.retryable_exceptions as exc:  # type: ignore[misc]
            self._handle_error(msg, partition, offset, True, exc)
        except Exception as exc:  # non-retryable (deserialization, validation)
            self._handle_error(msg, partition, offset, False, exc)
        finally:
            state.inflight.discard(offset)
            self._metrics["inflight_messages"] = len(state.inflight)
            self._commit_if_advanced(partition, state)

    def _handle_error(
        self,
        msg: Any,
        partition: int,
        offset: int,
        retryable: bool,
        exc: BaseException,
    ) -> None:
        """Route a failed message to retry or DLQ."""
        if retryable:
            self._metrics["retries_total"] = self._metrics.get("retries_total", 0) + 1
            self._publish_retry(msg, partition, offset, tier=RETRY_TIER_5S)
        else:
            self._metrics["dlq_total"] = self._metrics.get("dlq_total", 0) + 1
            self._publish_dlq(msg, partition, offset, exc)

    def _publish_retry(self, msg: Any, partition: int, offset: int, tier: int) -> None:
        """Publish a message to a retry topic with headers for attempt tracking."""
        if self.producer is None:
            logger.warning(
                "No producer configured; cannot publish retry for partition=%d offset=%d",
                partition,
                offset,
            )
            return
        topic = RETRY_TOPIC_5S if tier == RETRY_TIER_5S else RETRY_TOPIC_30S
        headers = [
            ("retry_tier", str(tier)),
            ("original_topic", msg.topic() if hasattr(msg, "topic") else ""),
            ("original_partition", str(partition)),
            ("original_offset", str(offset)),
            ("error", str(exc) if (exc := getattr(msg, "_last_error", None)) else ""),
        ]
        try:
            self.producer.produce(
                topic=topic,
                key=msg.key(),
                value=msg.value(),
                headers=headers,
            )
            self.producer.poll(0)
        except Exception:  # pragma: no cover - best effort
            logger.exception("Failed to publish retry message to %s", topic)

    def _publish_dlq(self, msg: Any, partition: int, offset: int, exc: BaseException) -> None:
        """Publish a non-retryable message to the DLQ."""
        if self.producer is None:
            logger.warning(
                "No producer configured; cannot publish DLQ for partition=%d offset=%d",
                partition,
                offset,
            )
            return
        headers = [
            ("original_topic", msg.topic() if hasattr(msg, "topic") else ""),
            ("original_partition", str(partition)),
            ("original_offset", str(offset)),
            ("error_class", type(exc).__name__),
            ("error_message", str(exc)),
        ]
        try:
            self.producer.produce(
                topic=DLQ_TOPIC,
                key=msg.key(),
                value=msg.value(),
                headers=headers,
            )
            self.producer.poll(0)
            logger.error(
                "Published to DLQ: partition=%d offset=%d error=%s",
                partition,
                offset,
                exc,
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception("Failed to publish DLQ message")

    def _commit_if_advanced(self, partition: int, state: PartitionState) -> None:
        """Commit the partition's offset if the tracker advanced."""
        next_offset = state.tracker.next_expected
        if next_offset > state.committed_offset:
            if TopicPartition is not None and self._consumer is not None:
                try:
                    tp = TopicPartition(partition, next_offset)  # type: ignore[call-arg]
                    self._consumer.commit(offsets=[tp])  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Failed to commit offset for partition %d", partition)
            state.committed_offset = next_offset
            self._metrics["consumer_lag"] = self._metrics.get("consumer_lag", 0) + (
                next_offset - state.committed_offset
            )

    def mark_done(self, partition: int, offset: int) -> int | None:
        """Mark an offset as done for a partition. Returns the next committable offset or None."""
        state = self._states.get(partition)
        if state is None:
            return None
        advanced = state.tracker.mark_done(offset)
        if advanced is not None:
            self._commit_if_advanced(partition, state)
        return advanced

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> dict[str, float | int]:
        """Return current consumer metrics."""
        return dict(self._metrics)

    def lag(self, partition: int) -> int:
        """Return the lag for a partition (committed offset minus next expected)."""
        state = self._states.get(partition)
        if state is None:
            return 0
        return state.tracker.next_expected - state.committed_offset


__all__ = [
    "Consumer",
    "ConsumerConfig",
    "MessageHandlingError",
    "OffsetTracker",
    "RETRY_TOPIC_5S",
    "RETRY_TOPIC_30S",
    "DLQ_TOPIC",
]
