"""Unit tests for the Kafka consumer framework (no broker required)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bharatsearch_kafka_utils import (
    DLQ_TOPIC,
    RETRY_TOPIC_5S,
    RETRY_TOPIC_30S,
    Consumer,
    ConsumerConfig,
    MessageHandlingError,
    OffsetTracker,
)

# ---------------------------------------------------------------------------
# OffsetTracker re-export
# ---------------------------------------------------------------------------


def test_offset_tracker_reexported():
    assert OffsetTracker is not None


# ---------------------------------------------------------------------------
# Consumer lifecycle (no Kafka)
# ---------------------------------------------------------------------------


def test_consumer_start_without_kafka():
    consumer = Consumer(ConsumerConfig(topics=["search.queries"]))
    consumer.start()  # should not raise
    consumer.stop()  # should not raise


def test_consumer_config_defaults():
    config = ConsumerConfig()
    assert config.bootstrap_servers == "localhost:9092"
    assert config.group_id == "bharatsearch-consumer"
    assert config.topics == []
    assert config.enable_auto_commit is False
    assert config.max_inflight == 100


# ---------------------------------------------------------------------------
# MessageHandlingError
# ---------------------------------------------------------------------------


def test_message_handling_error_retryable():
    exc = MessageHandlingError("boom", retryable=True)
    assert exc.retryable is True


def test_message_handling_error_non_retryable():
    exc = MessageHandlingError("boom", retryable=False)
    assert exc.retryable is False


# ---------------------------------------------------------------------------
# Mark done / offset tracking
# ---------------------------------------------------------------------------


def test_mark_done_advances_partition():
    consumer = Consumer(ConsumerConfig())
    consumer._states[0] = SimpleNamespace(
        tracker=OffsetTracker(5),
        committed_offset=5,
        inflight=set(),
        paused=False,
    )
    # 6 done but 5 missing -> nothing
    assert consumer.mark_done(0, 6) is None
    # 5 done -> commit through 7
    assert consumer.mark_done(0, 5) == 7


def test_mark_done_unknown_partition():
    consumer = Consumer(ConsumerConfig())
    assert consumer.mark_done(99, 0) is None


# ---------------------------------------------------------------------------
# Error handling classification
# ---------------------------------------------------------------------------


def test_retryable_error_routes_to_5s_retry():
    consumer = Consumer(ConsumerConfig())
    mock_producer = MagicMock()
    consumer.producer = mock_producer

    msg = MagicMock()
    msg.key.return_value = b"k1"
    msg.value.return_value = b"v1"
    msg.topic.return_value = "search.queries"
    msg.partition.return_value = 0
    msg.offset.return_value = 5

    state = SimpleNamespace(
        tracker=OffsetTracker(0),
        committed_offset=0,
        inflight={5},
        paused=False,
    )
    consumer._states[0] = state

    consumer._handle_error(msg, 0, 5, True, ValueError("retryable"))
    mock_producer.produce.assert_called_once()
    call_args = mock_producer.produce.call_args
    assert call_args.kwargs["topic"] == RETRY_TOPIC_5S
    headers = dict(call_args.kwargs["headers"])
    assert headers["retry_tier"] == "5"
    assert headers["original_topic"] == "search.queries"
    assert headers["original_partition"] == "0"
    assert headers["original_offset"] == "5"


def test_non_retryable_error_routes_to_dlq():
    consumer = Consumer(ConsumerConfig())
    mock_producer = MagicMock()
    consumer.producer = mock_producer

    msg = MagicMock()
    msg.key.return_value = b"k1"
    msg.value.return_value = b"v1"
    msg.topic.return_value = "search.queries"
    msg.partition.return_value = 0
    msg.offset.return_value = 5

    state = SimpleNamespace(
        tracker=OffsetTracker(0),
        committed_offset=0,
        inflight={5},
        paused=False,
    )
    consumer._states[0] = state

    consumer._handle_error(msg, 0, 5, False, ValueError("poison pill"))
    mock_producer.produce.assert_called_once()
    call_args = mock_producer.produce.call_args
    assert call_args.kwargs["topic"] == DLQ_TOPIC
    headers = dict(call_args.kwargs["headers"])
    assert headers["error_class"] == "ValueError"
    assert headers["error_message"] == "poison pill"


def test_no_producer_does_not_raise():
    consumer = Consumer(ConsumerConfig())
    consumer.producer = None

    msg = MagicMock()
    # Should not raise even without a producer
    consumer._handle_error(msg, 0, 0, True, ValueError("x"))


# ---------------------------------------------------------------------------
# Rebalance callbacks
# ---------------------------------------------------------------------------


def test_on_assign_creates_trackers():
    consumer = Consumer(ConsumerConfig())
    mock_consumer = MagicMock()
    consumer._consumer = mock_consumer

    tp0 = SimpleNamespace(partition=0, offset=100)
    tp1 = SimpleNamespace(partition=1, offset=200)

    consumer._on_assign(mock_consumer, [tp0, tp1])

    assert 0 in consumer._states
    assert 1 in consumer._states
    assert consumer._states[0].tracker.next_expected == 100
    assert consumer._states[1].tracker.next_expected == 200
    assert consumer._metrics["rebalances_total"] == 1


def test_on_revoke_commits_and_clears():
    consumer = Consumer(ConsumerConfig())
    mock_consumer = MagicMock()
    consumer._consumer = mock_consumer

    state = SimpleNamespace(
        tracker=OffsetTracker(100),
        committed_offset=100,
        inflight=set(),
        paused=False,
    )
    consumer._states[0] = state

    tp0 = SimpleNamespace(partition=0, offset=200)
    with patch("bharatsearch_kafka_utils.consumer.TopicPartition", MagicMock()):
        consumer._on_revoke(mock_consumer, [tp0])

    assert 0 not in consumer._states
    mock_consumer.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_retry_topic_constants():
    assert RETRY_TOPIC_5S == "search.retry.5s"
    assert RETRY_TOPIC_30S == "search.retry.30s"
    assert DLQ_TOPIC == "search.dlq"


# Add a helper to test poll_once dispatch path without Kafka
def test_poll_once_dispatches_to_handler():
    consumer = Consumer(ConsumerConfig(topics=["search.queries"]))
    handled = []

    def handler(msg):
        handled.append(msg.value())

    consumer.handler = handler

    msg = MagicMock()
    msg.partition.return_value = 0
    msg.offset.return_value = 0
    msg.key.return_value = b"k1"
    msg.error.return_value = None
    msg.value.return_value = b"test message"
    msg.topic.return_value = "search.queries"

    # Manually set up state so the message is accepted
    state = SimpleNamespace(
        tracker=OffsetTracker(0),
        committed_offset=0,
        inflight=set(),
        paused=False,
    )
    consumer._states[0] = state

    # Call _dispatch directly to avoid Kafka dependency
    consumer._dispatch(msg, b"k1", 0, 0, state)
    assert handled == [b"test message"]
    assert 0 not in state.inflight  # removed after handling


def test_dispatch_handler_error_is_caught():
    consumer = Consumer(ConsumerConfig(topics=["search.queries"]))

    def failing_handler(msg):
        raise ValueError("poison pill")

    consumer.handler = failing_handler

    msg = MagicMock()
    msg.partition.return_value = 0
    msg.offset.return_value = 0
    msg.key.return_value = b"k1"
    msg.error.return_value = None
    msg.value.return_value = b"bad message"
    msg.topic.return_value = "search.queries"

    state = SimpleNamespace(
        tracker=OffsetTracker(0),
        committed_offset=0,
        inflight=set(),
        paused=False,
    )
    consumer._states[0] = state

    # Non-retryable exception -> DLQ path, no crash
    consumer._dispatch(msg, b"k1", 0, 0, state)
    assert 0 not in state.inflight


def test_message_handling_error_is_retryable():
    consumer = Consumer(ConsumerConfig(topics=["search.queries"]))

    def raising_handler(msg):
        raise MessageHandlingError("transient", retryable=True)

    consumer.handler = raising_handler

    msg = MagicMock()
    msg.partition.return_value = 0
    msg.offset.return_value = 0
    msg.key.return_value = b"k1"
    msg.error.return_value = None
    msg.value.return_value = b"transient failure"
    msg.topic.return_value = "search.queries"

    state = SimpleNamespace(
        tracker=OffsetTracker(0),
        committed_offset=0,
        inflight=set(),
        paused=False,
    )
    consumer._states[0] = state

    consumer._dispatch(msg, b"k1", 0, 0, state)
    assert 0 not in state.inflight


# Give Consumer a poll_once_impl for testing without Kafka
# (poll_once requires a real Kafka consumer, so we test _dispatch directly)
# This test just ensures Consumer can be instantiated and configured
def test_consumer_instantiation():
    config = ConsumerConfig(
        bootstrap_servers="kafka:9092",
        group_id="test-group",
        topics=["search.queries", "search.impressions"],
        max_inflight=50,
    )
    consumer = Consumer(config)
    assert consumer.config.bootstrap_servers == "kafka:9092"
    assert consumer.config.group_id == "test-group"
    assert consumer.config.topics == ["search.queries", "search.impressions"]
    assert consumer.config.max_inflight == 50
    assert consumer._running is False
    assert consumer._consumer is None
