"""Fee schedules and the edge-minus-cost arithmetic.

Coinbase Exchange (the venue behind ws-feed.exchange.coinbase.com) prices
by trailing 30-day USD volume; the schedule below is the published
Exchange tier table (taker 60 bps at the entry tier down to 5 bps above
$400M), which is the 5-60 bps range the study plan quotes. The fee pages
refuse unauthenticated fetches, so these constants carry a date and must
be re-checked by the owner before a number built on them is published.

Hyperliquid perpetuals, tier 0 (docs fetched 2026-09-11): taker 0.045%
(4.5 bps), maker 0.015% (1.5 bps). The earlier schedule the plan quotes was
taker 0.035% (3.5 bps); both are kept so the table can show either.

Costs are stated against a MID-TO-MID move over the horizon:
    taking   : two taker fees plus the full spread (half on each leg)
    passive  : two maker fees minus the spread earned; this assumes both
               passive legs fill and ignores adverse selection, so it is an
               optimistic floor, labelled as such in the report.
"""
from __future__ import annotations

FEE_SCHEDULE_DATE = "2026-09-11"

# (30-day volume lower bound USD, taker bps, maker bps)
COINBASE_EXCHANGE_TIERS: list[tuple[float, float, float]] = [
    (0.0, 60.0, 40.0),
    (10_000.0, 40.0, 25.0),
    (50_000.0, 25.0, 15.0),
    (100_000.0, 20.0, 10.0),
    (1_000_000.0, 18.0, 8.0),
    (15_000_000.0, 16.0, 6.0),
    (75_000_000.0, 12.0, 3.0),
    (250_000_000.0, 8.0, 0.0),
    (400_000_000.0, 5.0, 0.0),
]

HYPERLIQUID_TAKER_BPS = 4.5
HYPERLIQUID_MAKER_BPS = 1.5
HYPERLIQUID_TAKER_BPS_LEGACY = 3.5


def coinbase_tier(volume_30d_usd: float) -> tuple[float, float]:
    """(taker_bps, maker_bps) for a trailing 30-day volume."""
    taker, maker = COINBASE_EXCHANGE_TIERS[0][1:]
    for lo, t, m in COINBASE_EXCHANGE_TIERS:
        if volume_30d_usd >= lo:
            taker, maker = t, m
    return taker, maker


def take_cost_bps(taker_bps: float, spread_bps: float) -> float:
    return 2.0 * taker_bps + spread_bps


def passive_cost_bps(maker_bps: float, spread_bps: float) -> float:
    return 2.0 * maker_bps - spread_bps


def hurdle_rows(edge_bps: float, spread_bps: float) -> list[dict]:
    """Edge minus cost under each venue/fee scenario, in bps."""
    cb_hi_t, cb_hi_m = coinbase_tier(0.0)
    cb_lo_t, cb_lo_m = coinbase_tier(400_000_000.0)
    scen = [
        ("take, Coinbase entry tier (%g bps)" % cb_hi_t,
         take_cost_bps(cb_hi_t, spread_bps)),
        ("take, Coinbase top tier (%g bps)" % cb_lo_t,
         take_cost_bps(cb_lo_t, spread_bps)),
        ("take, Hyperliquid tier 0 (%g bps)" % HYPERLIQUID_TAKER_BPS,
         take_cost_bps(HYPERLIQUID_TAKER_BPS, spread_bps)),
        ("take, Hyperliquid legacy (%g bps)" % HYPERLIQUID_TAKER_BPS_LEGACY,
         take_cost_bps(HYPERLIQUID_TAKER_BPS_LEGACY, spread_bps)),
        ("passive, Coinbase entry maker (%g bps), optimistic" % cb_hi_m,
         passive_cost_bps(cb_hi_m, spread_bps)),
        ("passive, Coinbase top maker (%g bps), optimistic" % cb_lo_m,
         passive_cost_bps(cb_lo_m, spread_bps)),
    ]
    return [{"scenario": name, "cost_bps": cost,
             "net_bps": (edge_bps - cost) if edge_bps == edge_bps else float("nan")}
            for name, cost in scen]
