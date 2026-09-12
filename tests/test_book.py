"""Book rebuild: absolute-size l2update semantics, best levels, snapshot check.

Every expected value in this file is written out by hand from the message
sequence above it, so a regression in ofi/book.py cannot be papered over by
recomputing the expectation from the code under test.
"""
from __future__ import annotations

import random

import pytest

from ofi.book import Book, Side

# A hand-written snapshot, deliberately unsorted so load() has to sort it.
SNAP_BIDS = [["99.50", "3"], ["99.80", "1"], ["99.00", "10"], ["99.90", "2"]]
SNAP_ASKS = [["100.30", "4"], ["100.10", "5"], ["100.00", "1"]]


def fresh() -> Book:
    b = Book()
    b.apply_snapshot(SNAP_BIDS, SNAP_ASKS)
    return b


def test_snapshot_sorts_both_sides_and_drops_zero_levels():
    b = Book()
    b.apply_snapshot(SNAP_BIDS + [["98.00", "0"]], SNAP_ASKS)
    assert b.best_bid() == (99.90, 2.0)
    assert b.best_ask() == (100.00, 1.0)
    assert b.bids.top(4) == [(99.90, 2.0), (99.80, 1.0), (99.50, 3.0), (99.00, 10.0)]
    assert b.asks.top(3) == [(100.00, 1.0), (100.10, 5.0), (100.30, 4.0)]
    assert 98.00 not in b.bids.sizes        # size "0" in a snapshot is not a level
    assert len(b.bids) == 4 and len(b.asks) == 3
    assert b.mid() == pytest.approx(99.95)


def test_l2update_size_is_absolute_not_a_delta():
    b = fresh()
    b.apply_l2update([["buy", "99.90", "7"]])      # 2 -> 7, NOT 2 + 7
    assert b.best_bid() == (99.90, 7.0)
    b.apply_l2update([["buy", "99.90", "1.5"]])    # sizes can shrink
    assert b.best_bid() == (99.90, 1.5)


def test_delete_best_promotes_the_next_level():
    b = fresh()
    b.apply_l2update([["buy", "99.90", "0"]])
    assert b.best_bid() == (99.80, 1.0)
    b.apply_l2update([["sell", "100.00", "0"], ["sell", "100.10", "0"]])
    assert b.best_ask() == (100.30, 4.0)
    assert b.mid() == pytest.approx((99.80 + 100.30) / 2)


def test_delete_deep_level_leaves_best_untouched():
    """The old dict book rescanned every level on any deletion; this one must
    not even change the best when a deep level is removed."""
    b = fresh()
    before = b.best_bid()
    b.apply_l2update([["buy", "99.00", "0"]])
    assert b.best_bid() == before
    assert len(b.bids) == 3


def test_insert_inside_the_spread_becomes_the_best():
    b = fresh()
    b.apply_l2update([["buy", "99.95", "0.25"], ["sell", "99.99", "0.75"]])
    assert b.best_bid() == (99.95, 0.25)
    assert b.best_ask() == (99.99, 0.75)
    assert b.crossed() is False
    b.apply_l2update([["buy", "100.50", "1"]])
    assert b.crossed() is True             # bid 100.50 >= ask 99.99


def test_deleting_a_price_that_is_not_there_is_a_no_op():
    b = fresh()
    n_b, n_a = len(b.bids), len(b.asks)
    b.apply_l2update([["buy", "12.34", "0"], ["sell", "555.0", "0"]])
    assert (len(b.bids), len(b.asks)) == (n_b, n_a)


def test_empty_side_reports_no_best_and_no_mid():
    b = fresh()
    b.apply_l2update([["sell", p, "0"] for p in ("100.00", "100.10", "100.30")])
    assert b.best_ask() is None
    assert b.mid() is None
    assert b.complete() is False


def test_invalidate_keeps_levels_but_marks_the_book_stale():
    b = fresh()
    b.invalidate()
    assert b.valid is False
    assert b.best_bid() == (99.90, 2.0)     # levels survive so compare() works


def test_compare_against_an_identical_snapshot_is_clean():
    b = fresh()
    chk = b.compare(SNAP_BIDS, SNAP_ASKS)
    assert chk["mismatched"] == 0
    assert chk["rate"] == 0.0
    assert chk["levels_book"] == chk["levels_snapshot"] == 7
    assert chk["best_bid_match"] and chk["best_ask_match"]


def test_compare_counts_missing_extra_and_resized_levels():
    b = fresh()
    # rebuilt book: drop 99.00, resize 99.80, invent 101.00 on the ask side
    b.apply_l2update([["buy", "99.00", "0"], ["buy", "99.80", "6"],
                      ["sell", "101.00", "2"]])
    chk = b.compare(SNAP_BIDS, SNAP_ASKS)
    # union of prices = 4 bids (99.00 99.50 99.80 99.90) + 4 asks
    # (100.00 100.10 100.30 101.00) = 8; mismatched = 99.00 (missing),
    # 99.80 (size 6 vs 1), 101.00 (extra) = 3.
    assert chk["levels_snapshot"] == 7
    assert chk["levels_book"] == 7
    assert chk["mismatched"] == 3
    assert chk["rate"] == pytest.approx(3 / 8)
    assert chk["best_bid_match"] is True    # 99.90 still best on both
    assert chk["best_ask_match"] is True


def test_replay_of_random_diffs_matches_a_reference_dict_book():
    """Property check: the sorted book and a naive dict book must agree on
    every level and on the best after a long random diff sequence."""
    rng = random.Random(7)
    b = Book()
    ref_bid: dict[float, float] = {}
    ref_ask: dict[float, float] = {}
    b.apply_snapshot([], [])
    for _ in range(4000):
        side = rng.choice(("buy", "sell"))
        price = round(rng.uniform(90.0, 110.0), 2)
        size = 0.0 if rng.random() < 0.35 else round(rng.uniform(0.1, 5.0), 3)
        b.apply_change(side, price, size)
        ref = ref_bid if side == "buy" else ref_ask
        if size <= 0.0:
            ref.pop(price, None)
        else:
            ref[price] = size
    assert b.bids.as_dict() == ref_bid
    assert b.asks.as_dict() == ref_ask
    assert (b.best_bid()[0] if ref_bid else None) == (max(ref_bid) if ref_bid else None)
    assert (b.best_ask()[0] if ref_ask else None) == (min(ref_ask) if ref_ask else None)
    # the sorted key lists must stay sorted and in step with the size dicts
    assert b.bids.keys == sorted(b.bids.keys)
    assert b.asks.keys == sorted(b.asks.keys)
    assert len(b.bids.keys) == len(ref_bid)
    assert len(b.asks.keys) == len(ref_ask)


def test_side_top_and_depth():
    s = Side(True)
    s.load([[10.0, 1.0], [12.0, 2.0], [11.0, 3.0]])
    assert s.top(2) == [(12.0, 2.0), (11.0, 3.0)]
    assert s.depth(2) == pytest.approx(5.0)
    assert s.depth(99) == pytest.approx(6.0)
    assert s.level(2) == (10.0, 1.0)
    assert s.level(3) is None
