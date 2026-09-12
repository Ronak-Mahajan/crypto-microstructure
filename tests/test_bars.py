"""Bar construction: the off-by-one fix, gap flags, and no fabricated bars.

The old harness applied an event to the accumulator and closed the bar
afterwards, so bar k carried the first event of bar k+1, and a websocket
outage produced one stale bar per missing second. Both are asserted against
here with hand-written event times.
"""
from __future__ import annotations

import numpy as np
import pytest

from ofi.bars import BarBuilder, concat_bars

S = 1_000_000_000          # one second in ns


def build(bar_s=1.0, max_gap_s=5.0) -> BarBuilder:
    return BarBuilder(int(bar_s * S), int(max_gap_s * S))


def state(mid: float) -> tuple:
    return (mid, 1.0, 0.0)


def test_rejects_nonsense_widths():
    with pytest.raises(ValueError):
        BarBuilder(0, S)
    with pytest.raises(ValueError):
        BarBuilder(S, -1)


def test_event_at_the_boundary_closes_the_previous_bar_first():
    """Events at t = 0.5 s (ofi +2) and t = 1.0 s (ofi +7).

    The 1.0 s event lands at bar_end for the [0, 1) bar, so that bar must
    close with ofi = 2 BEFORE the +7 is applied; +7 belongs to the [1, 2)
    bar. The old code produced ofi = 9 for the first bar.
    """
    b = build()
    b.on_event(S // 2, ofi=2.0, state=state(100.0))
    b.on_event(S, ofi=7.0, state=state(101.0))
    out = b.finish()
    assert list(out["t_end"]) == [S, 2 * S]
    assert list(out["ofi"]) == [2.0, 7.0]
    # the close mid of bar 1 is the state at its last event, not the next one
    assert out["mid"][0] == pytest.approx(100.0)
    assert out["mid"][1] == pytest.approx(101.0)
    assert list(out["n_events"]) == [1, 1]


def test_one_bar_holds_everything_strictly_inside_it():
    b = build()
    for t, e in ((100, 1.0), (S // 3, 2.0), (S - 1, 4.0)):
        b.on_event(t, ofi=e, state=state(100.0))
    out = b.finish()
    assert list(out["t_end"]) == [S]
    assert out["ofi"][0] == pytest.approx(7.0)
    assert out["n_events"][0] == 3


def test_short_quiet_stretch_yields_empty_but_real_bars():
    """3 s of silence with max_gap 5 s: the book genuinely did not move, so
    the intervening bars exist with n_events = 0, zero flow and the mid
    carried forward. They are NOT flagged as gaps."""
    b = build(max_gap_s=5.0)
    b.on_event(S // 2, ofi=1.0, state=state(100.0))
    b.on_event(3 * S + S // 2, ofi=5.0, state=state(102.0))
    out = b.finish()
    assert list(out["t_end"]) == [S, 2 * S, 3 * S, 4 * S]
    assert list(out["ofi"]) == [1.0, 0.0, 0.0, 5.0]
    assert list(out["n_events"]) == [1, 0, 0, 1]
    assert list(out["mid"]) == [100.0, 100.0, 100.0, 102.0]
    assert list(out["is_gap"]) == [0, 0, 0, 0]
    assert list(out["seg"]) == [0, 0, 0, 0]


def test_long_silence_produces_no_bars_at_all_for_the_dead_span():
    """60 s of silence with max_gap 5 s. The old harness emitted 60 stale
    bars; this one emits the bar that was open, then nothing until the feed
    comes back, and flags the resuming bar."""
    b = build(max_gap_s=5.0)
    b.on_event(S // 2, ofi=1.0, state=state(100.0))
    b.on_event(60 * S + S // 2, ofi=5.0, state=state(120.0))
    out = b.finish()
    assert list(out["t_end"]) == [S, 61 * S]
    assert list(out["ofi"]) == [1.0, 5.0]
    assert list(out["is_gap"]) == [0, 1]
    assert list(out["seg"]) == [0, 1]
    # nothing was fabricated: 2 bars of book time for 61 s of wall time
    assert len(out["t_end"]) == 2


def test_explicit_break_marker_splits_the_segment_without_a_time_gap():
    """A disconnect marker (or a resent snapshot) ends the run even when the
    next message arrives immediately."""
    b = build(max_gap_s=5.0)
    b.on_event(S // 2, ofi=1.0, state=state(100.0))
    b.on_break(S // 2 + 1000)
    b.on_event(S // 2 + 2000, ofi=3.0, state=state(100.0))
    out = b.finish()
    assert list(out["is_gap"]) == [0, 1]
    assert list(out["seg"]) == [0, 1]
    assert list(out["ofi"]) == [1.0, 3.0]


def test_first_bar_is_never_flagged_as_a_gap():
    b = build()
    b.on_event(5 * S + 10, state=state(100.0))
    out = b.finish()
    assert out["is_gap"][0] == 0
    assert out["seg"][0] == 0
    assert out["t_end"][0] == 6 * S


def test_a_price_never_crosses_a_gap():
    """A trade-only event after a long silence must not be priced pre-gap.

    The book is seen once at t=1 s (mid 100), then nothing for a minute,
    then a trade arrives with no book message of its own. The resuming bar
    belongs to a new segment, so pricing it at 100 would date its mid to
    before the gap and let a forward return start from a stale price. The
    honest mid is NaN until the book is observed inside the new segment.
    """
    b = build(max_gap_s=5.0)
    b.on_event(S, ofi=1.0, state=state(100.0))
    b.on_event(61 * S, tflow=2.0, is_book_event=False)
    out = b.finish()
    assert list(out["seg"]) == [0, 1]
    assert list(out["is_gap"]) == [0, 1]
    assert out["mid"][0] == pytest.approx(100.0)
    assert np.isnan(out["mid"][1]), "resuming bar was priced before the gap"
    assert out["tflow"][1] == pytest.approx(2.0)   # the flow itself is kept


def test_a_book_observation_inside_the_new_segment_prices_it_again():
    b = build(max_gap_s=5.0)
    b.on_event(S, ofi=1.0, state=state(100.0))
    b.on_event(61 * S, tflow=2.0, is_book_event=False)
    b.on_event(61 * S + S // 2, ofi=1.0, state=state(120.0))
    out = b.finish()
    assert list(out["seg"]) == [0, 1]
    assert out["mid"][1] == pytest.approx(120.0)


def test_trades_add_flow_without_counting_as_book_events():
    b = build()
    b.on_event(S // 4, ofi=2.0, state=state(100.0))
    b.on_event(S // 2, tflow=-0.75, is_book_event=False)
    out = b.finish()
    assert out["n_events"][0] == 1
    assert out["tflow"][0] == pytest.approx(-0.75)
    assert out["ofi"][0] == pytest.approx(2.0)


def test_empty_builder_returns_empty_typed_arrays():
    out = build().finish()
    assert len(out["t_end"]) == 0
    assert out["t_end"].dtype == np.int64
    assert out["ofi"].dtype == np.float64


def test_concat_rebases_segments_and_never_joins_two_days():
    a = build()
    a.on_event(S // 2, ofi=1.0, state=state(100.0))
    a.on_event(60 * S, ofi=1.0, state=state(100.0))     # forces seg 1
    aa = a.finish()
    b = build()
    b.on_event(S // 2, ofi=1.0, state=state(200.0))
    bb = b.finish()
    out = concat_bars([aa, bb])
    assert list(out["seg"]) == [0, 1, 2]
    # the first bar of the second day always opens a gap
    assert out["is_gap"][2] == 1
    assert len(out["t_end"]) == len(aa["t_end"]) + len(bb["t_end"])


def test_concat_of_nothing_is_empty():
    assert len(concat_bars([])["t_end"]) == 0
    assert len(concat_bars([build().finish()])["t_end"]) == 0
