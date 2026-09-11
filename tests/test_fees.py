"""Fee schedule lookup and the edge-minus-cost arithmetic."""
from __future__ import annotations

import math

import pytest

from ofi import fees


def test_tier_lookup_is_monotone_and_hits_the_documented_endpoints():
    assert fees.coinbase_tier(0.0) == (60.0, 40.0)
    assert fees.coinbase_tier(9_999.0) == (60.0, 40.0)
    assert fees.coinbase_tier(10_000.0)[0] == 40.0        # boundary is inclusive
    assert fees.coinbase_tier(500_000_000.0) == (5.0, 0.0)
    takers = [fees.coinbase_tier(v)[0] for v in
              (0, 1e4, 5e4, 1e5, 1e6, 1.5e7, 7.5e7, 2.5e8, 4e8)]
    assert takers == sorted(takers, reverse=True)         # never goes up


def test_taking_pays_two_fees_and_the_whole_spread():
    """Mid-to-mid: cross the spread in and out = 2 * half-spread = spread,
    plus a taker fee on each leg."""
    assert fees.take_cost_bps(5.0, 2.0) == pytest.approx(12.0)
    assert fees.take_cost_bps(60.0, 0.0) == pytest.approx(120.0)


def test_passive_quoting_earns_the_spread_and_can_be_a_net_credit():
    """Two maker fees minus the spread earned. At Coinbase's top maker tier
    (0 bps) a 2 bps spread is a 2 bps credit before adverse selection."""
    assert fees.passive_cost_bps(0.0, 2.0) == pytest.approx(-2.0)
    assert fees.passive_cost_bps(40.0, 2.0) == pytest.approx(78.0)


def test_hurdle_rows_cover_both_venues_and_both_ways_of_trading():
    rows = fees.hurdle_rows(edge_bps=3.0, spread_bps=1.0)
    assert len(rows) == 6
    names = " ".join(r["scenario"] for r in rows)
    assert "Coinbase entry" in names and "Coinbase top" in names
    assert "Hyperliquid tier 0" in names and "Hyperliquid legacy" in names
    assert "passive" in names
    for r in rows:
        assert r["net_bps"] == pytest.approx(3.0 - r["cost_bps"])


def test_a_three_bps_edge_does_not_survive_taking_anywhere():
    """The headline arithmetic the study exists to state: a 3 bps edge is
    negative against every taker schedule once the spread is paid."""
    rows = fees.hurdle_rows(edge_bps=3.0, spread_bps=1.0)
    takes = [r for r in rows if r["scenario"].startswith("take")]
    assert len(takes) == 4
    assert all(r["net_bps"] < 0 for r in takes)
    # cheapest taker here is Hyperliquid legacy: 3 - (2*3.5 + 1) = -5 bps
    assert min(r["net_bps"] for r in takes) < -5.0
    assert max(r["net_bps"] for r in takes) == pytest.approx(-5.0)


def test_hurdle_rows_propagate_a_missing_edge_as_nan():
    rows = fees.hurdle_rows(edge_bps=float("nan"), spread_bps=1.0)
    assert all(math.isnan(r["net_bps"]) for r in rows)
    assert all(not math.isnan(r["cost_bps"]) for r in rows)


def test_the_schedule_carries_a_date_so_it_can_be_rechecked():
    assert len(fees.FEE_SCHEDULE_DATE) == 10
    assert fees.HYPERLIQUID_TAKER_BPS > 0
    assert fees.HYPERLIQUID_MAKER_BPS < fees.HYPERLIQUID_TAKER_BPS
