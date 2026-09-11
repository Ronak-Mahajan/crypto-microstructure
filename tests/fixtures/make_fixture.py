"""Generate the committed synthetic fixture: `python tests/fixtures/make_fixture.py`.

THIS IS NOT MARKET DATA. It is a deterministic toy feed (seed 20260911) whose
only purpose is to exercise the whole pipeline end to end in CI and to give
`make smoke` a real, tiny results table to print. Every number the smoke run
produces is a property of the generator below, not of any exchange. Real
numbers wait on Tardis days pulled with `make fetch`.

What the generator plants, so the self-test tables can be read:

* A Coinbase-shaped BTC-USD feed for the synthetic UTC day 2001-01-01:
  one full `snapshot`, then `l2update` messages that carry ABSOLUTE sizes,
  interleaved with `match` messages.
* Price is moved by order flow the way Cont, Kukanov and Stoikov (2014)
  model it: a latent pressure accumulates the same e_n the analysis
  computes, and the book steps one tick whenever it crosses +/-`STEP_Q`.
  So the contemporaneous OFI-vs-mid-change R^2 in table (a) should come out
  HIGH by construction. That is the point: table (a) is the pipeline's
  sanity anchor, and on a feed where the mechanism is known it must fire.
* Nothing is planted in the FORWARD direction: the pressure that has not
  yet moved the price is public in the bar's own OFI, so table (b) is close
  to a null and the fee hurdle in (c) is comfortably negative. A self-test
  that predicted the future would mean the harness was leaking.
* One deliberate 90 s outage with a `disconnect` marker and a reconnect
  `snapshot`, so the gap flags, the segment split and the snapshot
  consistency check all appear in the report.
* A `snapshot` of the TRUE book every VERIFY_EVERY_S seconds. A real
  Coinbase feed only resends one on reconnect; this fixture resends them on
  purpose so the smoke run exercises the snapshot-vs-rebuild check and the
  report's "no gap" column has to come out 0.00%. If a change to book.py or
  features.py ever drops or double-counts a level, that column stops being
  zero and CI fails on tests/test_fixture.py.
* A Hyperliquid `trades` stream that LEADS the Coinbase matches by exactly
  LEAD_MS milliseconds, and an `l2Book` 20-level snapshot stream at 0.5 s,
  so table (d) has something to find at its stated resolution.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

FIX = Path(__file__).resolve().parent
ROOT = FIX.parents[1]
sys.path.insert(0, str(ROOT))

from ofi.io import write_lines            # noqa: E402

DAY = "2001-01-01"
SYMBOL = "BTC-USD"
COIN = "BTC"
SEED = 20260911

MID0_TICKS = 10_000       # best bid, in whole ticks (= $100.00)
DEPTH = 12                # levels per side in the snapshot
STEP_Q = 2.0             # units of OFI pressure that move the book one tick
LEAD_MS = 300             # Hyperliquid trade lead over the Coinbase match

SPAN_S = 1800             # seconds of synthetic book time before the outage
GAP_AT_S = 1200           # the outage starts here
GAP_LEN_S = 90
MEAN_DT_S = 2.2           # average seconds between l2update messages
MAX_DT_S = 4.0            # never longer than the 5 s max-gap, so no false gap
VERIFY_EVERY_S = 240      # re-emit the true book (fixture-only; see docstring)

S = 1_000_000_000
T0 = int(datetime(2001, 1, 1, tzinfo=timezone.utc).timestamp()) * S


def iso(t_ns: int) -> str:
    return (datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")


def px(ticks: int) -> str:
    """Prices are held as whole ticks and only ever formatted, never added
    as floats: 100.00 - k*0.01 in float lands on a .005 rounding boundary
    and two distinct levels can then print the same string."""
    return f"{ticks / 100.0:.2f}"


def sz(v: float) -> str:
    return f"{max(v, 0.0):.4f}"


class ToyBook:
    """Best bid/ask plus a static ladder behind them."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.pb = MID0_TICKS          # best bid, in whole ticks
        self.pa = MID0_TICKS + 1      # best ask, one tick above
        self.qb = round(rng.uniform(1.0, 6.0), 4)
        self.qa = round(rng.uniform(1.0, 6.0), 4)
        self.deep_b = [round(rng.uniform(1.0, 9.0), 4) for _ in range(DEPTH)]
        self.deep_a = [round(rng.uniform(1.0, 9.0), 4) for _ in range(DEPTH)]
        self.pressure = 0.0

    def snapshot(self) -> dict:
        bids = [[px(self.pb), sz(self.qb)]]
        asks = [[px(self.pa), sz(self.qa)]]
        for i, q in enumerate(self.deep_b, start=1):
            bids.append([px(self.pb - i), sz(q)])
        for i, q in enumerate(self.deep_a, start=1):
            asks.append([px(self.pa + i), sz(q)])
        return {"type": "snapshot", "product_id": SYMBOL, "bids": bids,
                "asks": asks}

    def step(self) -> list[list[str]]:
        """One l2update's worth of ABSOLUTE-size changes; updates pressure."""
        rng = self.rng
        changes: list[list[str]] = []
        e = 0.0
        if rng.random() < 0.5:                    # resize the best bid
            new = round(max(0.05, self.qb + rng.gauss(0.0, 1.2)), 4)
            e += new - self.qb
            self.qb = new
            changes.append(["buy", px(self.pb), sz(new)])
        if rng.random() < 0.5:                    # resize the best ask
            new = round(max(0.05, self.qa + rng.gauss(0.0, 1.2)), 4)
            e -= new - self.qa
            self.qa = new
            changes.append(["sell", px(self.pa), sz(new)])
        if not changes:                           # touch a deep level instead
            side = rng.choice(("buy", "sell"))
            i = rng.randrange(DEPTH)
            new = round(max(0.05, rng.uniform(0.5, 9.0)), 4)
            if side == "buy":
                self.deep_b[i] = new
                changes.append(["buy", px(self.pb - (i + 1)), sz(new)])
            else:
                self.deep_a[i] = new
                changes.append(["sell", px(self.pa + (i + 1)), sz(new)])

        self.pressure += e
        while abs(self.pressure) >= STEP_Q:
            up = self.pressure > 0
            self.pressure -= STEP_Q if up else -STEP_Q
            changes.extend(self._shift(up))
        return changes

    def _shift(self, up: bool) -> list[list[str]]:
        """Move the whole book one tick, as four absolute-size changes.

        Going up: the resting ask at P^A is lifted, a new bid appears one
        tick above P^B, the old best bid becomes the first deep bid, the
        deepest bid falls off the maintained window, and a new deepest ask
        is added. Every other level keeps both its price and its size, so
        no change has to be emitted for it.
        """
        rng = self.rng
        fresh = round(rng.uniform(1.0, 9.0), 4)
        tail = round(rng.uniform(1.0, 9.0), 4)
        out: list[list[str]] = []
        if up:
            out.append(["sell", px(self.pa), "0"])
            out.append(["buy", px(self.pb + 1), sz(fresh)])
            out.append(["buy", px(self.pb - DEPTH), "0"])
            out.append(["sell", px(self.pa + DEPTH + 1), sz(tail)])
            self.deep_b = [self.qb] + self.deep_b[:-1]
            self.qa = self.deep_a[0]            # pa + TICK is the new best ask
            self.deep_a = self.deep_a[1:] + [tail]
            self.qb = fresh
            self.pb, self.pa = self.pb + 1, self.pa + 1
        else:
            out.append(["buy", px(self.pb), "0"])
            out.append(["sell", px(self.pa - 1), sz(fresh)])
            out.append(["sell", px(self.pa + DEPTH), "0"])
            out.append(["buy", px(self.pb - DEPTH - 1), sz(tail)])
            self.deep_a = [self.qa] + self.deep_a[:-1]
            self.qb = self.deep_b[0]            # pb - TICK is the new best bid
            self.deep_b = self.deep_b[1:] + [tail]
            self.qa = fresh
            self.pb, self.pa = self.pb - 1, self.pa - 1
        return out

    def resync_snapshot(self, rng: random.Random) -> dict:
        """The book after an unobserved outage: same shape, moved and
        reshuffled, which is what a real reconnect snapshot looks like."""
        drift = rng.randint(-6, 6)
        self.pb += drift
        self.pa += drift
        self.qb = round(rng.uniform(1.0, 6.0), 4)
        self.qa = round(rng.uniform(1.0, 6.0), 4)
        self.deep_b = [round(rng.uniform(1.0, 9.0), 4) for _ in range(DEPTH)]
        self.deep_a = [round(rng.uniform(1.0, 9.0), 4) for _ in range(DEPTH)]
        self.pressure = 0.0
        return self.snapshot()


def line(t_ns: int, m: dict) -> str:
    return json.dumps({"t_ns": t_ns, "m": m}, separators=(",", ":"))


def marker(t_ns: int, kind: str) -> str:
    return json.dumps({"t_ns": t_ns, "marker": kind, "venue": "synthetic",
                       "reason": "fixture", "m": None}, separators=(",", ":"))


def generate() -> dict[str, list[str]]:
    rng = random.Random(SEED)
    book = ToyBook(rng)
    cb: list[str] = []
    hl_trades: list[str] = []
    hl_book: list[str] = []

    t = T0
    cb.append(marker(t, "connect"))
    cb.append(line(t, book.snapshot()))

    end = T0 + SPAN_S * S
    gap_start = T0 + GAP_AT_S * S
    did_gap = False
    next_hl_book = T0
    next_verify = T0 + VERIFY_EVERY_S * S
    trade_id = 1

    while t < end:
        dt = min(MAX_DT_S, max(0.3, rng.expovariate(1.0 / MEAN_DT_S)))
        t += int(dt * S)

        if t >= next_verify:
            cb.append(line(t, book.snapshot()))
            next_verify += VERIFY_EVERY_S * S

        if not did_gap and t >= gap_start:
            cb.append(marker(t, "disconnect"))
            t += GAP_LEN_S * S
            cb.append(marker(t, "connect"))
            cb.append(line(t, book.resync_snapshot(rng)))
            did_gap = True
            continue

        changes = book.step()
        m = {"type": "l2update", "product_id": SYMBOL,
             "changes": changes, "time": iso(t)}
        cb.append(line(t, m))

        # a trade roughly every fourth book event, printed on Hyperliquid
        # LEAD_MS earlier so table (d) has a known answer
        if rng.random() < 0.25:
            taker_buy = rng.random() < 0.5 + 0.2 * (1 if book.pressure > 0 else -1)
            size = round(rng.uniform(0.01, 0.6), 4)
            t_cb = t + int(rng.uniform(0.0, 0.05) * S)
            cb.append(line(t_cb, {
                "type": "match", "product_id": SYMBOL, "trade_id": trade_id,
                # Coinbase `side` is the MAKER side: a taker buy hits an ask
                "side": "sell" if taker_buy else "buy",
                "size": sz(size), "price": px(book.pa if taker_buy else book.pb),
                "time": iso(t_cb)}))
            t_hl = t_cb - LEAD_MS * 1_000_000
            hl_trades.append(line(t_hl, {
                "channel": "trades", "data": [{
                    "coin": COIN, "side": "B" if taker_buy else "A",
                    "px": px(book.pa if taker_buy else book.pb),
                    "sz": sz(size * 1.5), "time": t_hl // 1_000_000,
                    "tid": trade_id}]}))
            trade_id += 1

        while next_hl_book <= t:
            if not (did_gap and gap_start <= next_hl_book < gap_start + GAP_LEN_S * S):
                hl_book.append(line(next_hl_book, {
                    "channel": "l2Book",
                    "data": {"coin": COIN, "time": next_hl_book // 1_000_000,
                             "levels": [
                                 [{"px": px(book.pb - i),
                                   "sz": sz(book.qb + i), "n": 1}
                                  for i in range(3)],
                                 [{"px": px(book.pa + i),
                                   "sz": sz(book.qa + i), "n": 1}
                                  for i in range(3)]]}}))
            next_hl_book += S // 2

    return {f"coinbase/{SYMBOL}": cb,
            f"hyperliquid/{COIN}-trades": hl_trades,
            f"hyperliquid/{COIN}-l2Book": hl_book}


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import tardis_loader as tl

    out_root = FIX / "synthetic" / "data"
    manifest_path = FIX / "synthetic" / "manifest.json"
    ymd = DAY.replace("-", "")
    streams = generate()

    if out_root.exists():
        for p in sorted(out_root.rglob("*.jsonl.gz")):
            p.unlink()

    manifest = {"schema": 1, "files": {}, "days": {},
                "note": "SYNTHETIC self-test fixture written by "
                        "tests/fixtures/make_fixture.py. Not market data."}
    total = 0
    for rel, lines in streams.items():
        path = out_root / rel / f"{ymd}-00.jsonl.gz"
        n = write_lines(path, lines, compresslevel=9)
        total += n
        exchange, symbol = rel.split("/")
        key = path.relative_to(manifest_path.parent).as_posix()
        entry = tl.file_entry(path, "synthetic", exchange, symbol, DAY, n)
        manifest["files"][key] = entry
        manifest["days"][f"{exchange}/{symbol}/{DAY}"] = {
            "source": "synthetic", "lines": n, "files": 1,
            "generator": "tests/fixtures/make_fixture.py", "seed": SEED}
        print(f"{key}: {n} lines, {entry['bytes']} bytes")

    manifest["updated_at"] = "1970-01-01T00:00:00Z"     # keep the file stable
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    print(f"{total} lines total -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
