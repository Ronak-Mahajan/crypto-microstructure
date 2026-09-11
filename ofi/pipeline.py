"""Messages -> bars for one Coinbase product-day (plus trade streams).

Clock: bars are stamped with the ARRIVAL time `t_ns`, not the venue's own
`time` field. That is a decision, not an oversight. Arrival time is what a
trader reacting to the feed actually has, it exists for every message and
both sources (the recorder stamps it on receipt, Tardis replays its own
arrival stamp), and it is the same clock on both venues, which the lead-lag
section needs. Venue time is preserved through ofi.io as `ev[2]` for anyone
who wants to measure exchange-side batching latency later; nothing in this
module reads it.

`rebuild(events, ...)` consumes the event stream from ofi.io and drives the
book, the per-change feature increments and the bar builder. It returns
the bars and a statistics dict that the report prints per day:

    n_msgs, n_snapshots, n_l2updates, n_changes, n_matches,
    n_disconnects, n_skipped_updates (diffs seen while the book was stale),
    n_crossed (messages after which bid >= ask), bars, segments, gap_bars,
    covered_s (bar-seconds of book time), first_t_ns, last_t_ns,
    snapshot_checks: [{t_ns, rate, mismatched, levels_book, levels_snapshot,
                       best_bid_match, best_ask_match, after_gap}]
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .bars import BarBuilder
from .book import Book
from .features import (best_level_ofi, micro_deviation_bps, multilevel_ofi,
                       signed_size_coinbase_match, spread_bps)


def _state(book: Book) -> tuple[float, float, float] | None:
    b, a = book.best_bid(), book.best_ask()
    if b is None or a is None:
        return None
    return (0.5 * (b[0] + a[0]), spread_bps(b[0], a[0]),
            micro_deviation_bps(b[0], b[1], a[0], a[1]))


def rebuild(events: Iterable[tuple], bar_s: float = 1.0, max_gap_s: float = 5.0,
            k_levels: int = 5, product: str | None = None) -> tuple[dict, dict]:
    bar_ns = int(round(bar_s * 1e9))
    max_gap_ns = int(round(max_gap_s * 1e9))
    book = Book()
    bars = BarBuilder(bar_ns, max_gap_ns)
    stats = {"n_msgs": 0, "n_snapshots": 0, "n_l2updates": 0, "n_changes": 0,
             "n_matches": 0, "n_disconnects": 0, "n_skipped_updates": 0,
             "n_crossed": 0, "snapshot_checks": [],
             "first_t_ns": None, "last_t_ns": None}
    prev_b = prev_a = None
    prev_top = None
    after_gap = False

    for ev in events:
        kind = ev[0]
        t_ns = ev[1]
        if stats["first_t_ns"] is None:
            stats["first_t_ns"] = t_ns
        stats["last_t_ns"] = t_ns
        if kind == "marker":
            if ev[2] == "disconnect":
                stats["n_disconnects"] += 1
                book.invalidate()
                bars.on_break(t_ns)
                after_gap = True
            continue
        m = ev[3]
        if product is not None and m.get("product_id") not in (None, product):
            continue
        stats["n_msgs"] += 1
        typ = m.get("type")

        if typ == "snapshot":
            # Compare the diff-rebuilt book with EVERY resent snapshot,
            # including the one that follows a disconnect: invalidate() only
            # stops diffs from being applied, it does not discard the levels.
            # A check tagged after_gap=True measures the gap as much as the
            # rebuild, so the report separates the two; a non-zero rate on a
            # check with after_gap=False is a genuine replay bug.
            chk = None
            if book.n_snapshots > 0 and (book.bids.keys or book.asks.keys):
                chk = book.compare(m.get("bids", []), m.get("asks", []))
                chk["t_ns"] = t_ns
                chk["after_gap"] = after_gap
                stats["snapshot_checks"].append(chk)
            book.apply_snapshot(m.get("bids", []), m.get("asks", []))
            stats["n_snapshots"] += 1
            # A resent snapshot ends the contiguous run only if something was
            # actually missed: either a disconnect was recorded, or the
            # snapshot disagrees with the diff-rebuilt book. A snapshot that
            # matches the rebuild exactly proves the replay was complete
            # across it, so there is no gap to flag.
            if stats["n_snapshots"] > 1 and (
                    after_gap or chk is None or chk["mismatched"] > 0):
                bars.on_break(t_ns)
            after_gap = False
            prev_b, prev_a = book.best_bid(), book.best_ask()
            prev_top = book.top(k_levels)
            st = _state(book)
            if st is not None:
                bars.on_event(t_ns, state=st, is_book_event=False)
            continue

        if typ == "l2update":
            if not book.valid:
                stats["n_skipped_updates"] += 1
                continue
            stats["n_l2updates"] += 1
            e_sum = 0.0
            ml_sum = 0.0
            n_ch = 0
            for side, p, s in m.get("changes", []):
                book.apply_change(side, float(p), float(s))
                n_ch += 1
                cur_b, cur_a = book.best_bid(), book.best_ask()
                cur_top = book.top(k_levels)
                if (prev_b is not None and prev_a is not None
                        and cur_b is not None and cur_a is not None):
                    e_sum += best_level_ofi(prev_b, cur_b, prev_a, cur_a)
                    ml_sum += multilevel_ofi(prev_top[0], cur_top[0],
                                             prev_top[1], cur_top[1], k_levels)
                prev_b, prev_a, prev_top = cur_b, cur_a, cur_top
            stats["n_changes"] += n_ch
            if book.crossed():
                stats["n_crossed"] += 1
            st = _state(book)
            if st is not None:
                bars.on_event(t_ns, ofi=e_sum, mlofi=ml_sum, state=st)
            continue

        if typ in ("match", "last_match"):
            sz = signed_size_coinbase_match(m)
            if sz is None:
                continue
            stats["n_matches"] += 1
            # trades do not move the book state; they add to tflow only and
            # do not count as book events. They still respect gap logic, and
            # are ignored while the book is stale (between a disconnect and
            # the next snapshot) so they cannot open a bar on old state.
            if book.valid and bars.last_t is not None:
                bars.on_event(t_ns, tflow=sz, is_book_event=False)
            continue

    out = bars.finish()
    n = len(out["t_end"])
    stats["bars"] = n
    stats["segments"] = int(out["seg"].max()) + 1 if n else 0
    stats["gap_bars"] = int(out["is_gap"].sum()) if n else 0
    stats["covered_s"] = n * bar_s
    stats["span_s"] = ((stats["last_t_ns"] - stats["first_t_ns"]) / 1e9
                       if stats["first_t_ns"] is not None else 0.0)
    clean = [c for c in stats["snapshot_checks"] if not c["after_gap"]]
    if stats["snapshot_checks"]:
        rates = np.array([c["rate"] for c in stats["snapshot_checks"]])
        stats["mismatch_rate_mean"] = float(rates.mean())
        stats["mismatch_rate_max"] = float(rates.max())
        stats["n_checks"] = int(len(rates))
    else:
        stats["mismatch_rate_mean"] = None
        stats["mismatch_rate_max"] = None
        stats["n_checks"] = 0
    if clean:
        r = np.array([c["rate"] for c in clean])
        stats["mismatch_rate_clean_mean"] = float(r.mean())
        stats["mismatch_rate_clean_max"] = float(r.max())
        stats["n_checks_clean"] = int(len(r))
    else:
        stats["mismatch_rate_clean_mean"] = None
        stats["mismatch_rate_clean_max"] = None
        stats["n_checks_clean"] = 0
    # Gapped checks get their own max. Reusing mismatch_rate_max for the
    # after-a-gap column would print a clean check's rate in the gap column
    # whenever a clean check is the worse of the two -- which is exactly the
    # case that must not be hidden, because a non-zero CLEAN rate is a
    # replay bug and a non-zero gapped rate is only a measure of the gap.
    gapped = [c for c in stats["snapshot_checks"] if c["after_gap"]]
    if gapped:
        r = np.array([c["rate"] for c in gapped])
        stats["mismatch_rate_gap_mean"] = float(r.mean())
        stats["mismatch_rate_gap_max"] = float(r.max())
        stats["n_checks_gap"] = int(len(r))
    else:
        stats["mismatch_rate_gap_mean"] = None
        stats["mismatch_rate_gap_max"] = None
        stats["n_checks_gap"] = 0
    return out, stats
