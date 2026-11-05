"""Resilient Kafka consumer utilities (Implementation Guide, Section 6.3).

OffsetTracker tracks per-partition completion so commits only advance a
contiguous processed prefix, never skipping in-flight work.
"""

from __future__ import annotations


class OffsetTracker:
    def __init__(self, start: int) -> None:
        self.next_expected = start  # lowest offset not yet finished
        self.done: set[int] = set()

    def mark_done(self, offset: int) -> int | None:
        self.done.add(offset)
        advanced = False
        while self.next_expected in self.done:
            self.done.remove(self.next_expected)
            self.next_expected += 1
            advanced = True
        return self.next_expected if advanced else None
