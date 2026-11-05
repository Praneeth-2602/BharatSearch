# ADR-004: Kafka consumer design (partition-keyed workers with out-of-order commit)

## Status

Accepted

## Context

The Kafka consumer framework (Implementation Guide Section 6.3) must ensure
that one slow or poisonous message does not stall a partition, commits never
skip an unfinished message, and rebalances do not lose or duplicate work
beyond what idempotency absorbs.

## Decision

Use one poll loop per consumer with auto-commit off. Each assigned partition
has an `OffsetTracker`. Messages are dispatched to a worker pool, partitioned
by key hash so messages with the same key stay ordered while different keys
run in parallel. The committable offset for a partition is the lowest
in-flight offset (everything below it is done). On revoke: stop dispatching,
wait for in-flight work up to a timeout, commit each tracker's offset, drop
the state. On assign: create trackers starting at the committed offset.

## Consequences

- Ordering is preserved per key while different keys are processed in
  parallel.
- Poison/slow messages are isolated to their key's worker; the partition
  keeps moving.
- Retryable errors go to `search.retry.5s` then `search.retry.30s`; after the
  last tier, or immediately for non-retryable errors, messages go to
  `search.dlq`.
- Idempotency via Redis SET NX with TTL on `event_id` absorbs duplicates from
  rebalances and retries.

## Related

None
