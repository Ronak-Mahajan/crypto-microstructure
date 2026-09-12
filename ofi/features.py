"""Per-event features.

best_level_ofi  Cont, Kukanov and Stoikov (2014), "The price impact of order
                book events", eq. (10): with (P^B, q^B) and (P^A, q^A) the best
                bid/ask price and size before (n-1) and after (n) an event,

                    e_n =  1{P^B_n >= P^B_{n-1}} q^B_n
                         - 1{P^B_n <= P^B_{n-1}} q^B_{n-1}
                         - 1{P^A_n <= P^A_{n-1}} q^A_n
                         + 1{P^A_n >= P^A_{n-1}} q^A_{n-1}

                and OFI over a window is the sum of e_n over events in it.
                An event here is one l2update change applied to the book, not
                a 50 ms batch (record.py) or a Tardis message; the caller
                applies changes one at a time and sums the increments.

multilevel_ofi  the same formula applied level by level to the m-th best
                price/size on each side, summed over m = 1..k (Xu, Gould,
                Howison 2019 style "deep" OFI without the depth
                normalisation; in units of size like e_n).

micro_price     (P^B q^A + P^A q^B) / (q^A + q^B), the size-weighted mid.

signed trades   Coinbase `match`: `side` is the MAKER side, so side == "sell"
                means the taker bought (an up-tick) -> +size; "buy" -> -size.
                Hyperliquid `trades`: side "B" is a buy, "A" a sell.
"""
from __future__ import annotations

from typing import Sequence

Level = tuple[float, float]


def best_level_ofi(prev_b: Level, cur_b: Level,
                   prev_a: Level, cur_a: Level) -> float:
    """CKS eq. (10) increment e_n for one best-level transition."""
    pb0, qb0 = prev_b
    pb1, qb1 = cur_b
    pa0, qa0 = prev_a
    pa1, qa1 = cur_a
    e = 0.0
    if pb1 >= pb0:
        e += qb1
    if pb1 <= pb0:
        e -= qb0
    if pa1 <= pa0:
        e -= qa1
    if pa1 >= pa0:
        e += qa0
    return e


def multilevel_ofi(prev_bids: Sequence[Level], cur_bids: Sequence[Level],
                   prev_asks: Sequence[Level], cur_asks: Sequence[Level],
                   k: int) -> float:
    """Sum of the eq. (10) increment over the k best levels of each side.

    Levels missing on either side of the transition (book shallower than k)
    contribute nothing for that m.
    """
    e = 0.0
    kb = min(k, len(prev_bids), len(cur_bids))
    ka = min(k, len(prev_asks), len(cur_asks))
    for m in range(kb):
        pb0, qb0 = prev_bids[m]
        pb1, qb1 = cur_bids[m]
        if pb1 >= pb0:
            e += qb1
        if pb1 <= pb0:
            e -= qb0
    for m in range(ka):
        pa0, qa0 = prev_asks[m]
        pa1, qa1 = cur_asks[m]
        if pa1 <= pa0:
            e -= qa1
        if pa1 >= pa0:
            e += qa0
    return e


def micro_price(bid: float, qb: float, ask: float, qa: float) -> float:
    tot = qb + qa
    if tot <= 0.0:
        return 0.5 * (bid + ask)
    return (bid * qa + ask * qb) / tot


def micro_deviation_bps(bid: float, qb: float, ask: float, qa: float) -> float:
    """(micro-price - mid) / mid in basis points; positive = buy pressure."""
    mid = 0.5 * (bid + ask)
    if mid <= 0.0:
        return 0.0
    return (micro_price(bid, qb, ask, qa) - mid) / mid * 1e4


def spread_bps(bid: float, ask: float) -> float:
    mid = 0.5 * (bid + ask)
    return (ask - bid) / mid * 1e4 if mid > 0.0 else 0.0


def signed_size_coinbase_match(m: dict) -> float | None:
    """Signed taker volume of a Coinbase `match`/`last_match` message."""
    side = m.get("side")
    try:
        size = float(m.get("size"))
    except (TypeError, ValueError):
        return None
    if side == "sell":          # maker sold => taker bought => up-tick
        return size
    if side == "buy":
        return -size
    return None


def signed_size_hyperliquid_trade(t: dict) -> float | None:
    """Signed size of one element of a Hyperliquid `trades` data list."""
    side = t.get("side")
    try:
        size = float(t.get("sz"))
    except (TypeError, ValueError):
        return None
    if side == "B":
        return size
    if side == "A":
        return -size
    return None
