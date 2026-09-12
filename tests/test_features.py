"""CKS order-flow imbalance and the other per-event features.

The expectations are worked out by hand from Cont, Kukanov and Stoikov
(2014) eq. (10),

    e_n =  1{P^B_n >= P^B_{n-1}} q^B_n  -  1{P^B_n <= P^B_{n-1}} q^B_{n-1}
         - 1{P^A_n <= P^A_{n-1}} q^A_n  +  1{P^A_n >= P^A_{n-1}} q^A_{n-1}

with the arithmetic written out in each docstring so the number in the
assertion can be checked without running anything.
"""
from __future__ import annotations

import pytest

from ofi.features import (best_level_ofi, micro_deviation_bps, micro_price,
                          multilevel_ofi, signed_size_coinbase_match,
                          signed_size_hyperliquid_trade, spread_bps)


def test_bid_size_added_at_an_unchanged_price_is_a_pure_delta():
    """P^B equal both ways, so e = q^B_n - q^B_{n-1} = 5 - 2 = +3.
    The ask is untouched: -q^A_n + q^A_{n-1} = -1 + 1 = 0."""
    e = best_level_ofi((100.0, 2.0), (100.0, 5.0), (101.0, 1.0), (101.0, 1.0))
    assert e == pytest.approx(3.0)


def test_bid_size_removed_at_an_unchanged_price_is_negative():
    """e = 5 - 9 = -4."""
    e = best_level_ofi((100.0, 9.0), (100.0, 5.0), (101.0, 1.0), (101.0, 1.0))
    assert e == pytest.approx(-4.0)


def test_new_better_bid_counts_its_whole_size():
    """P^B up (100.0 -> 100.5): only 1{>=} fires, e = +q^B_n = +4.
    Ask unchanged contributes -1 + 1 = 0."""
    e = best_level_ofi((100.0, 9.0), (100.5, 4.0), (101.0, 1.0), (101.0, 1.0))
    assert e == pytest.approx(4.0)


def test_bid_stepping_down_removes_the_old_size():
    """P^B down (100.0 -> 99.5): only 1{<=} fires, e = -q^B_{n-1} = -9."""
    e = best_level_ofi((100.0, 9.0), (99.5, 4.0), (101.0, 1.0), (101.0, 1.0))
    assert e == pytest.approx(-9.0)


def test_new_better_ask_is_negative_flow():
    """P^A down (101.0 -> 100.5): only 1{<=} fires, e = -q^A_n = -3.
    Bid unchanged: +2 - 2 = 0."""
    e = best_level_ofi((100.0, 2.0), (100.0, 2.0), (101.0, 7.0), (100.5, 3.0))
    assert e == pytest.approx(-3.0)


def test_ask_lifted_upwards_is_positive_flow():
    """P^A up (101.0 -> 101.5): only 1{>=} fires, e = +q^A_{n-1} = +7."""
    e = best_level_ofi((100.0, 2.0), (100.0, 2.0), (101.0, 7.0), (101.5, 3.0))
    assert e == pytest.approx(7.0)


def test_both_sides_move_in_the_same_event():
    """Bid 100.0/2 -> 100.5/4 gives +4; ask 101.0/7 -> 100.5/3 gives -3.
    Total e = +1."""
    e = best_level_ofi((100.0, 2.0), (100.5, 4.0), (101.0, 7.0), (100.5, 3.0))
    assert e == pytest.approx(1.0)


def test_a_no_op_event_has_zero_flow():
    """Everything unchanged: (q - q) on each side = 0."""
    e = best_level_ofi((100.0, 2.0), (100.0, 2.0), (101.0, 7.0), (101.0, 7.0))
    assert e == pytest.approx(0.0)


def test_ofi_is_antisymmetric_under_swapping_the_sides():
    """Mirroring the book across the mid must flip the sign of e."""
    prev_b, cur_b = (100.0, 2.0), (100.5, 4.0)
    prev_a, cur_a = (101.0, 7.0), (101.0, 5.0)
    e = best_level_ofi(prev_b, cur_b, prev_a, cur_a)
    # mirror: price p -> -p, bid <-> ask
    mb, mcb = (-prev_a[0], prev_a[1]), (-cur_a[0], cur_a[1])
    ma, mca = (-prev_b[0], prev_b[1]), (-cur_b[0], cur_b[1])
    assert best_level_ofi(mb, mcb, ma, mca) == pytest.approx(-e)


def test_multilevel_ofi_sums_the_same_rule_over_k_levels():
    """k = 2. Level 1 bid 100.0/2 -> 100.0/5 = +3; level 2 bid 99.5/1 ->
    99.0/4 (price down) = -1. Level 1 ask 101.0/1 -> 101.0/1 = 0; level 2
    ask 101.5/2 -> 101.5/6 (price equal) = -6 + 2 = -4. Total = -2."""
    e = multilevel_ofi(
        [(100.0, 2.0), (99.5, 1.0)], [(100.0, 5.0), (99.0, 4.0)],
        [(101.0, 1.0), (101.5, 2.0)], [(101.0, 1.0), (101.5, 6.0)], 2)
    assert e == pytest.approx(-2.0)


def test_multilevel_ofi_with_k_1_equals_best_level_ofi():
    prev_b, cur_b = [(100.0, 2.0)], [(100.5, 4.0)]
    prev_a, cur_a = [(101.0, 7.0)], [(100.5, 3.0)]
    assert multilevel_ofi(prev_b, cur_b, prev_a, cur_a, 1) == \
        pytest.approx(best_level_ofi(prev_b[0], cur_b[0], prev_a[0], cur_a[0]))


def test_multilevel_ofi_ignores_levels_the_book_does_not_have():
    """The book is 1 deep on the bid side; asking for k = 3 must not raise
    and must only count the level that exists on both transitions."""
    e = multilevel_ofi([(100.0, 2.0)], [(100.0, 5.0)],
                       [(101.0, 1.0)], [(101.0, 1.0)], 3)
    assert e == pytest.approx(3.0)


def test_micro_price_leans_towards_the_thin_side():
    """(P^B q^A + P^A q^B) / (q^A + q^B) = (100*1 + 101*9)/10 = 100.9."""
    assert micro_price(100.0, 9.0, 101.0, 1.0) == pytest.approx(100.9)
    # balanced book -> exactly the mid
    assert micro_price(100.0, 4.0, 101.0, 4.0) == pytest.approx(100.5)
    # degenerate sizes fall back to the mid rather than dividing by zero
    assert micro_price(100.0, 0.0, 101.0, 0.0) == pytest.approx(100.5)


def test_micro_deviation_and_spread_in_bps():
    # mid 100.5, micro 100.9 -> (0.4 / 100.5) * 1e4 = 39.80 bps
    assert micro_deviation_bps(100.0, 9.0, 101.0, 1.0) == pytest.approx(39.8010, abs=1e-3)
    # spread 1.0 on a mid of 100.5 -> 99.5025 bps
    assert spread_bps(100.0, 101.0) == pytest.approx(99.5025, abs=1e-3)
    assert micro_deviation_bps(100.0, 4.0, 101.0, 4.0) == pytest.approx(0.0)


def test_coinbase_match_side_is_the_maker_side():
    """Coinbase `match.side` is the side of the RESTING order: side "sell"
    means a resting ask was hit, i.e. the taker bought."""
    assert signed_size_coinbase_match({"side": "sell", "size": "1.5"}) == 1.5
    assert signed_size_coinbase_match({"side": "buy", "size": "1.5"}) == -1.5
    assert signed_size_coinbase_match({"side": "sell", "size": "x"}) is None
    assert signed_size_coinbase_match({"size": "1"}) is None


def test_hyperliquid_trade_side_letters():
    assert signed_size_hyperliquid_trade({"side": "B", "sz": "0.25"}) == 0.25
    assert signed_size_hyperliquid_trade({"side": "A", "sz": "0.25"}) == -0.25
    assert signed_size_hyperliquid_trade({"side": "?", "sz": "1"}) is None
    assert signed_size_hyperliquid_trade({"side": "B"}) is None
