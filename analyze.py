"""Phase 1 harness: does order-flow imbalance predict short-horizon returns?

Rebuilds the Coinbase book from recorded snapshots and l2update diffs,
computes best-level order-flow imbalance (Cont, Kukanov and Stoikov 2014)
on one-second bars, and regresses forward mid returns on it with
time-ordered walk-forward splits. Every number this prints is out of
sample; in-sample fits are never reported.

This file is the HARNESS, shipped before the data: it runs end to end on
whatever the recorder has captured so far, prints fold-by-fold results with
sample sizes, and refuses to editorialize. Conclusions belong to the weeks
of data the recorder accumulates, not to the first night. A result on hours
of data is a smoke test of the pipeline, and the output says so.

    python analyze.py                       # all recorded BTC-USD
    python analyze.py --product ETH-USD --bar 1.0 --horizons 1,5,30
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "coinbase"


def iter_messages(product: str):
    """Yield (t_ns, msg) for one product from all recorded hours, in order.

    Files live one directory per product (data/coinbase/<product>/) and are
    ordered by their hour name; marker lines ("m": null) are skipped.
    """
    paths = glob.glob(str(DATA / product / "*.jsonl.gz"))
    for path in sorted(paths, key=lambda p: Path(p).name):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue      # torn final line from a hard stop
                    m = d.get("m")
                    if isinstance(m, dict) and m.get("product_id") == product:
                        yield d["t_ns"], m
        except EOFError:
            # The recorder is still writing this hour's file, so its last
            # gzip member has no end-of-stream marker yet. Everything read
            # before the tear is valid; analysis on live data expects this.
            continue


def bars_from_book(product: str, bar_s: float):
    """One-second (default) bars: (t_end_ns, mid, ofi_sum).

    The book is a dict price -> size, rebuilt from each snapshot and patched
    by every l2update. Best levels are tracked incrementally; OFI uses the
    best-level formulation: contributions from changes at (or through) the
    inside, signed by side and by whether the inside improved or retreated.
    """
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    best_b = best_a = None
    prev_b = prev_a = None            # (price, size) at last observation
    bar_end = None
    ofi = 0.0
    n_snapshots = 0
    bar_ns = int(bar_s * 1e9)

    for t_ns, m in iter_messages(product):
        typ = m.get("type")
        if typ == "snapshot":
            bids = {float(p): float(s) for p, s in m["bids"]}
            asks = {float(p): float(s) for p, s in m["asks"]}
            best_b = max(bids) if bids else None
            best_a = min(asks) if asks else None
            prev_b = (best_b, bids.get(best_b, 0.0)) if best_b else None
            prev_a = (best_a, asks.get(best_a, 0.0)) if best_a else None
            n_snapshots += 1
            continue
        if typ != "l2update" or best_b is None or best_a is None:
            continue

        for side, price_s, size_s in m.get("changes", []):
            price, size = float(price_s), float(size_s)
            book = bids if side == "buy" else asks
            if size == 0.0:
                book.pop(price, None)
            else:
                book[price] = size
            if side == "buy" and (price >= (best_b or -1) or size == 0.0):
                best_b = max(bids) if bids else None
            elif side == "sell" and (price <= (best_a or 1e18) or size == 0.0):
                best_a = min(asks) if asks else None

        if not bids or not asks:
            continue
        cur_b = (best_b, bids[best_b])
        cur_a = (best_a, asks[best_a])
        # Cont-Kukanov-Stoikov best-level OFI increments
        if prev_b is not None:
            if cur_b[0] > prev_b[0]:
                ofi += cur_b[1]
            elif cur_b[0] < prev_b[0]:
                ofi -= prev_b[1]
            else:
                ofi += cur_b[1] - prev_b[1]
        if prev_a is not None:
            if cur_a[0] < prev_a[0]:
                ofi -= cur_a[1]
            elif cur_a[0] > prev_a[0]:
                ofi += prev_a[1]
            else:
                ofi -= cur_a[1] - prev_a[1]
        prev_b, prev_a = cur_b, cur_a

        if bar_end is None:
            bar_end = (t_ns // bar_ns + 1) * bar_ns
        while t_ns >= bar_end:
            yield bar_end, (best_b + best_a) / 2.0, ofi
            ofi = 0.0
            bar_end += bar_ns

    print(f"  ({n_snapshots} snapshots seen; each resets the book)")


def walk_forward(xs, ys, n_folds: int):
    """Time-ordered folds: fit slope on fold i, score on fold i+1."""
    n = len(xs)
    fold = n // (n_folds + 1)
    results = []
    for i in range(n_folds):
        tr = slice(i * fold, (i + 1) * fold)
        te = slice((i + 1) * fold, (i + 2) * fold)
        xtr, ytr, xte, yte = xs[tr], ys[tr], xs[te], ys[te]
        mx = sum(xtr) / len(xtr)
        my = sum(ytr) / len(ytr)
        sxx = sum((x - mx) ** 2 for x in xtr)
        if sxx == 0:
            continue
        beta = sum((x - mx) * (y - my) for x, y in zip(xtr, ytr)) / sxx
        alpha = my - beta * mx
        pred = [alpha + beta * x for x in xte]
        ss_res = sum((p - y) ** 2 for p, y in zip(pred, yte))
        mte = sum(yte) / len(yte)
        ss_tot = sum((y - mte) ** 2 for y in yte)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results.append((len(xte), beta, r2))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--bar", type=float, default=1.0, help="bar size, seconds")
    p.add_argument("--horizons", default="1,5,30",
                   help="forward-return horizons in bars")
    p.add_argument("--folds", type=int, default=4)
    args = p.parse_args()

    print(f"rebuilding {args.product} book at {args.bar:g}s bars ...")
    bars = list(bars_from_book(args.product, args.bar))
    if len(bars) < 200:
        raise SystemExit(f"only {len(bars)} bars recorded so far; the "
                         f"pipeline needs a few hundred to demonstrate and "
                         f"weeks to conclude. Let the recorder run.")
    mids = [b[1] for b in bars]
    ofis = [b[2] for b in bars]
    hours = len(bars) * args.bar / 3600.0
    print(f"{len(bars):,} bars ({hours:.1f} hours of book time)")

    for h in [int(x) for x in args.horizons.split(",")]:
        xs = ofis[:-h]
        ys = [math.log(mids[i + h] / mids[i]) * 1e4          # bps
              for i in range(len(mids) - h)]
        folds = walk_forward(xs, ys, args.folds)
        print(f"\nhorizon {h} bar(s): forward return (bps) on OFI, "
              f"{len(folds)} walk-forward folds")
        for i, (n, beta, r2) in enumerate(folds):
            print(f"  fold {i + 1}: n={n:>6,}  beta={beta:+.3e}  "
                  f"oos R2={r2:+.4f}")
        pos = sum(1 for _, b, _ in folds if b > 0)
        print(f"  slope sign agreement: {pos}/{len(folds)} folds positive")

    print("\nNOTE: this run is a pipeline demonstration on whatever is "
          "recorded so far. Signal conclusions require weeks of data and "
          "the fee/impact analysis of Phase 2.")


if __name__ == "__main__":
    main()
