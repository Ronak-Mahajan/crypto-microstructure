"""Price-level order book with sorted best-level maintenance.

Each side keeps a dict price -> size and a bisect-maintained sorted list of
keys (bids stored negated so that index 0 is always the best level on both
sides). Lookup, best-level and top-k reads are O(log n) / O(k); insert and
delete are a bisect plus a C-level list memmove, which at the O(10^4)
levels of a full Coinbase book is far below one JSON parse per message.
The previous dict book rescanned every level on every deletion.

Coinbase level2 / level2_batch semantics (Coinbase Exchange websocket docs):
a snapshot lists [price, size] for the entire book; each l2update change
[side, price, size] carries the NEW ABSOLUTE size at that price, size "0"
meaning the level is gone. Sizes are never deltas.

Snapshot consistency: `compare(bids, asks)` diffs the rebuilt book against
a fresh snapshot and returns the mismatch rate, so a reconnect (the only
time Coinbase resends a snapshot) doubles as a check that the diff replay
was correct up to the moment the socket dropped.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from typing import Iterable, Sequence

Level = tuple[float, float]           # (price, size)


class Side:
    __slots__ = ("sign", "keys", "sizes")

    def __init__(self, is_bid: bool) -> None:
        self.sign = -1.0 if is_bid else 1.0
        self.keys: list[float] = []   # sorted; keys[0] is the best level
        self.sizes: dict[float, float] = {}

    def clear(self) -> None:
        self.keys.clear()
        self.sizes.clear()

    def load(self, levels: Iterable[Sequence]) -> None:
        """Replace the side with snapshot levels [[price, size], ...]."""
        self.clear()
        sizes = self.sizes
        for p, s in levels:
            price, size = float(p), float(s)
            if size > 0.0:
                sizes[price] = size
        sign = self.sign
        self.keys = sorted(sign * p for p in sizes)

    def set(self, price: float, size: float) -> None:
        """Absolute-size update; size <= 0 deletes the level."""
        sizes = self.sizes
        if size <= 0.0:
            if price in sizes:
                del sizes[price]
                key = self.sign * price
                i = bisect_left(self.keys, key)
                # key must be present; guard against float noise anyway
                if i < len(self.keys) and self.keys[i] == key:
                    del self.keys[i]
            return
        if price not in sizes:
            insort(self.keys, self.sign * price)
        sizes[price] = size

    def __len__(self) -> int:
        return len(self.keys)

    def best(self) -> Level | None:
        if not self.keys:
            return None
        price = self.sign * self.keys[0]
        return price, self.sizes[price]

    def level(self, i: int) -> Level | None:
        if i >= len(self.keys):
            return None
        price = self.sign * self.keys[i]
        return price, self.sizes[price]

    def top(self, k: int) -> list[Level]:
        sign, sizes = self.sign, self.sizes
        return [(sign * key, sizes[sign * key]) for key in self.keys[:k]]

    def depth(self, k: int) -> float:
        sign, sizes = self.sign, self.sizes
        return sum(sizes[sign * key] for key in self.keys[:k])

    def as_dict(self) -> dict[float, float]:
        return dict(self.sizes)


class Book:
    """Two-sided book with Coinbase snapshot / l2update application."""

    def __init__(self) -> None:
        self.bids = Side(True)
        self.asks = Side(False)
        self.valid = False            # False until a snapshot has been loaded
        self.n_snapshots = 0
        self.n_changes = 0

    # -- state ------------------------------------------------------------
    def invalidate(self) -> None:
        """After a disconnect: ignore diffs until the next snapshot."""
        self.valid = False

    def apply_snapshot(self, bids: Iterable[Sequence],
                       asks: Iterable[Sequence]) -> None:
        self.bids.load(bids)
        self.asks.load(asks)
        self.valid = True
        self.n_snapshots += 1

    def apply_change(self, side: str, price: float, size: float) -> None:
        (self.bids if side == "buy" else self.asks).set(price, size)
        self.n_changes += 1

    def apply_l2update(self, changes: Iterable[Sequence]) -> None:
        for side, p, s in changes:
            self.apply_change(side, float(p), float(s))

    # -- reads ------------------------------------------------------------
    def best_bid(self) -> Level | None:
        return self.bids.best()

    def best_ask(self) -> Level | None:
        return self.asks.best()

    def top(self, k: int) -> tuple[list[Level], list[Level]]:
        return self.bids.top(k), self.asks.top(k)

    def complete(self) -> bool:
        return self.valid and bool(self.bids.keys) and bool(self.asks.keys)

    def crossed(self) -> bool:
        b, a = self.bids.best(), self.asks.best()
        return b is not None and a is not None and b[0] >= a[0]

    def mid(self) -> float | None:
        b, a = self.bids.best(), self.asks.best()
        if b is None or a is None:
            return None
        return 0.5 * (b[0] + a[0])

    # -- consistency ------------------------------------------------------
    def compare(self, bids: Iterable[Sequence], asks: Iterable[Sequence],
                rel_tol: float = 1e-9) -> dict:
        """Diff the rebuilt book against a fresh snapshot.

        Returns {"levels_book", "levels_snapshot", "mismatched", "rate",
        "best_bid_match", "best_ask_match"} where mismatched counts prices
        present on only one side of the comparison or present on both with
        different sizes, and rate = mismatched / |union of prices|.
        """
        snap_b = {float(p): float(s) for p, s in bids if float(s) > 0.0}
        snap_a = {float(p): float(s) for p, s in asks if float(s) > 0.0}
        mism = 0
        union = 0
        for have, want in ((self.bids.sizes, snap_b), (self.asks.sizes, snap_a)):
            prices = set(have) | set(want)
            union += len(prices)
            for p in prices:
                h, w = have.get(p), want.get(p)
                if h is None or w is None:
                    mism += 1
                elif abs(h - w) > rel_tol * max(abs(h), abs(w), 1.0):
                    mism += 1
        bb, ba = self.bids.best(), self.asks.best()
        sb = max(snap_b) if snap_b else None
        sa = min(snap_a) if snap_a else None
        return {
            "levels_book": len(self.bids) + len(self.asks),
            "levels_snapshot": len(snap_b) + len(snap_a),
            "mismatched": mism,
            "rate": (mism / union) if union else 0.0,
            "best_bid_match": (bb[0] if bb else None) == sb,
            "best_ask_match": (ba[0] if ba else None) == sa,
        }
