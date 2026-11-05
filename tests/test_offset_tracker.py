"""Unit tests for the OffsetTracker consumer utility (libs/kafka_utils)."""

from bharatsearch_kafka_utils import OffsetTracker


def test_mark_done_advances_contiguous_prefix():
    t = OffsetTracker(5)
    # 6 done but 5 missing -> nothing committable
    assert t.mark_done(6) is None
    assert t.next_expected == 5
    # 5 done -> 5 and 6 are contiguous, commit through 7 (next uncommitted)
    assert t.mark_done(5) == 7
    assert t.next_expected == 7
    # 7 done -> commit through 8
    assert t.mark_done(7) == 8
    assert t.next_expected == 8
    # 9 done but 8 missing -> no advance
    assert t.mark_done(9) is None
    assert t.next_expected == 8


def test_duplicate_done_is_idempotent():
    t = OffsetTracker(0)
    assert t.mark_done(0) == 1
    assert t.mark_done(0) is None
    assert t.next_expected == 1


def test_out_of_order_then_fill():
    t = OffsetTracker(0)
    assert t.mark_done(2) is None
    assert t.mark_done(1) is None  # 0 still missing
    assert t.mark_done(0) == 3  # 0,1,2 contiguous -> commit through 3
    assert t.next_expected == 3
