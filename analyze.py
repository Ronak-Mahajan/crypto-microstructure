"""Phase 1: does order-flow imbalance predict short-horizon returns, net of fees?

Command-line front end for the `ofi` package. Every table it writes is
regenerated from the data files listed in the manifest; nothing is typed in.

    python analyze.py results                       # every day in manifest.json
    python analyze.py results --manifest tests/fixtures/synthetic/manifest.json \
                              --out results-smoke --folds 3
    python analyze.py day --symbol BTC-USD --day 2025-08-01   # one day, no files written

Pipeline (see ofi/*.py):
  book       sorted price levels, Coinbase absolute-size l2update semantics,
             rebuilt book compared with every resent snapshot
  features   CKS best-level OFI per change (eq. 10), k-level OFI, micro-price,
             signed trade flow
  bars       1 s bars, bar closed BEFORE an event at t >= bar_end is applied,
             gap-flagged segments, no fabricated bars past max_gap
  inference  purged walk-forward (train ends h before test), Campbell-
             Thompson OOS R^2, Newey-West HAC with bandwidth >= h, moving-
             block bootstrap CIs, CKS contemporaneous R^2 at 1/10/60 s
  fees       Coinbase taker tiers, Hyperliquid taker, spread: edge minus cost
  leadlag    Hyperliquid vs Coinbase signed flow at 100 ms .. 5 s; books >= 1 s
  report     results/README.md in fixed order (a) CKS (b) forward (c) fees (d) lead-lag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ofi import report  # noqa: E402
from ofi.run import DEFAULT_PARAMS, run_results  # noqa: E402


def _params(args) -> dict:
    p = dict(DEFAULT_PARAMS)
    p.update({
        "bar_s": args.bar, "max_gap_s": args.max_gap, "k_levels": args.levels,
        "horizons_s": [float(x) if "." in x else int(x)
                       for x in args.horizons.split(",") if x],
        "n_folds": args.folds, "expanding": not args.rolling,
        "n_boot": args.n_boot, "seed": args.seed, "decile": args.decile,
    })
    return p


def _common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--manifest", default=str(ROOT / "manifest.json"))
    sp.add_argument("--data-root", default=None,
                    help="override the data directory the manifest keys resolve in")
    sp.add_argument("--bar", type=float, default=DEFAULT_PARAMS["bar_s"])
    sp.add_argument("--max-gap", type=float, default=DEFAULT_PARAMS["max_gap_s"],
                    help="seconds of silence after which bars stop (no stale bars)")
    sp.add_argument("--levels", type=int, default=DEFAULT_PARAMS["k_levels"])
    sp.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_PARAMS["horizons_s"]),
                    help="forward horizons in seconds")
    sp.add_argument("--folds", type=int, default=DEFAULT_PARAMS["n_folds"])
    sp.add_argument("--rolling", action="store_true",
                    help="rolling one-chunk training instead of expanding")
    sp.add_argument("--n-boot", type=int, default=DEFAULT_PARAMS["n_boot"])
    sp.add_argument("--seed", type=int, default=DEFAULT_PARAMS["seed"])
    sp.add_argument("--decile", type=float, default=DEFAULT_PARAMS["decile"])


def cmd_results(args) -> int:
    results = run_results(Path(args.manifest), args.data_root, _params(args),
                          repo_root=ROOT)
    md, js = report.write(results, Path(args.out))
    print(f"wrote {md} and {js}")
    if not results["days"]:
        print("no analysable day in the manifest; the report says so")
    return 0


def cmd_day(args) -> int:
    import json
    from ofi.run import days_from_manifest, analyse_day
    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    groups = days_from_manifest(manifest, Path(args.manifest).parent, ROOT,
                                Path(args.data_root) if args.data_root else None)
    key = ("coinbase", args.symbol, args.day)
    if key not in groups or not groups[key]["files"]:
        print(f"no readable files for {'/'.join(key)} in {args.manifest}")
        return 1
    p = _params(args)
    coin = args.symbol.split("-")[0]
    hl = {"trades": [q for _, q, _ in groups.get(("hyperliquid", f"{coin}-trades", args.day), {"files": []})["files"]],
          "l2Book": [q for _, q, _ in groups.get(("hyperliquid", f"{coin}-l2Book", args.day), {"files": []})["files"]]}
    res, _ = analyse_day(key, groups[key], hl, p)
    results = {"params": p, "manifest": args.manifest,
               "manifest_updated_at": manifest.get("updated_at"),
               "n_manifest_files": len(manifest.get("files", {})),
               "days": [res], "pooled": None, "skipped": []}
    sys.stdout.write(report.build_markdown(results))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("results", help="run every manifest day, write results/")
    _common(r)
    r.add_argument("--out", default=str(ROOT / "results"))
    r.set_defaults(fn=cmd_results)
    d = sub.add_parser("day", help="analyse one day and print its tables")
    _common(d)
    d.add_argument("--symbol", default="BTC-USD")
    d.add_argument("--day", required=True, help="YYYY-MM-DD")
    d.set_defaults(fn=cmd_day)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
