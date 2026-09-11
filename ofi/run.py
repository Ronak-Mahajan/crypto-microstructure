"""Run the whole measurement over the days listed in a manifest.

    results = run_results(manifest_path, data_root, params)
    report.write(results, out_dir)

Days are discovered from manifest["files"] (exchange/symbol/day of every
hashed file), so a self-recorded day added with `tardis_loader.py hash`
and a Tardis day added with `fetch` are treated identically. Only Coinbase
product days are analysed for OFI; a Hyperliquid <COIN>-trades / -l2Book
directory for the same UTC day, if hashed, feeds the lead-lag section.

Result dict (also dumped to results.json):
    params, days: [day_result, ...], pooled: {forward, hurdle, bars, ...}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import fees, inference, leadlag
from .bars import concat_bars
from .io import iter_files
from .pipeline import rebuild

FEATURES = ("ofi", "mlofi", "micro", "tflow")
FEATURE_LABEL = {"ofi": "best-level OFI", "mlofi": "multi-level OFI",
                 "micro": "micro-price dev.", "tflow": "signed trade flow"}
CKS_AGG_S = (1, 10, 60)
TRADE_LL_DT_S = (0.1, 0.5, 1.0, 5.0)
BOOK_LL_DT_S = (1.0, 5.0)

DEFAULT_PARAMS = {
    "bar_s": 1.0, "max_gap_s": 5.0, "k_levels": 5,
    "horizons_s": [1, 5, 10, 30, 60, 300], "n_folds": 4, "expanding": True,
    "n_boot": 200, "seed": 0, "decile": 0.1, "max_lag_s": 5.0,
    "book_max_lag_s": 30.0,
}


# ---------------------------------------------------------------------------
# manifest -> days
# ---------------------------------------------------------------------------

def resolve_path(key: str, manifest_dir: Path, repo_root: Path,
                 data_root: Path | None) -> Path | None:
    cands = [manifest_dir / key, repo_root / key]
    if data_root is not None:
        k = key
        if k.startswith("data/"):
            k = k[len("data/"):]
        cands.append(Path(data_root) / k)
    for c in cands:
        if c.exists():
            return c
    return None


def days_from_manifest(manifest: dict, manifest_dir: Path, repo_root: Path,
                       data_root: Path | None) -> dict:
    """{(exchange, symbol, day): {"files": [(key, path, entry)], ...}}"""
    groups: dict = defaultdict(lambda: {"files": [], "missing": []})
    for key, entry in sorted(manifest.get("files", {}).items()):
        ex, sym, day = entry.get("exchange"), entry.get("symbol"), entry.get("day")
        if not (ex and sym and day):
            continue
        p = resolve_path(key, manifest_dir, repo_root, data_root)
        g = groups[(ex, sym, day)]
        if p is None:
            g["missing"].append(key)
        else:
            g["files"].append((key, p, entry))
    for g in groups.values():
        g["files"].sort(key=lambda t: Path(t[0]).name)
    return dict(groups)


# ---------------------------------------------------------------------------
# per-day analysis
# ---------------------------------------------------------------------------

def _feature_matrix(bars: dict) -> dict[str, np.ndarray]:
    return {"ofi": bars["ofi"], "mlofi": bars["mlofi"],
            "micro": bars["micro"], "tflow": bars["tflow"]}


def forward_block(bars: dict, params: dict, with_ci: bool = True) -> dict:
    """Forward predictability for every feature x horizon on one bar set."""
    out = {}
    mid, seg = bars["mid"], bars["seg"]
    n = len(mid)
    feats = _feature_matrix(bars)
    for h_s in params["horizons_s"]:
        h = int(round(h_s / params["bar_s"]))
        if h < 1:
            continue
        y = inference.forward_returns(mid, seg, h)
        n_valid = int(np.isfinite(y).sum())
        for name, x in feats.items():
            key = (name, h_s)
            if n_valid < 50 or n < (params["n_folds"] + 1) * max(h + 2, 10):
                out[key] = {"h_s": h_s, "h_bars": h, "n_valid": n_valid,
                            "insufficient": True}
                continue
            wf = inference.walk_forward(x, y, h, params["n_folds"],
                                        params["expanding"])
            hac = inference.hac_regression(x, y, h)
            rec = {"h_s": h_s, "h_bars": h, "n_valid": n_valid,
                   "insufficient": wf["n_folds_used"] == 0,
                   "r2_oos": wf["r2_oos"], "n_test": wf["n_test"],
                   "n_folds_used": wf["n_folds_used"],
                   "sign_agreement": wf["sign_agreement"],
                   "fold_r2": [f["r2_oos"] for f in wf["folds"]],
                   "fold_beta": [f["beta"] for f in wf["folds"]],
                   "hac_beta": hac["beta"], "hac_se": hac["se"],
                   "hac_t": hac["t"], "hac_bandwidth": hac["bandwidth"],
                   "insample_r2": hac["r2"], "ci": (np.nan, np.nan),
                   "edge": None}
            if with_ci and not rec["insufficient"]:
                block = max(h, 30)
                rec["ci"] = inference.bootstrap_ci(
                    inference.oos_r2, [y, wf["yhat"], wf["ybar_tr"]],
                    block, params["n_boot"], params["seed"])
                edge = inference.decile_edge(y, wf["yhat"], params["decile"])
                if np.isfinite(edge["edge_bps"]):
                    edge["ci"] = inference.bootstrap_ci(
                        lambda yy, pp: inference.decile_edge_stat(
                            yy, pp, params["decile"]),
                        [y, wf["yhat"]], block, params["n_boot"],
                        params["seed"] + 1)
                else:
                    edge["ci"] = (np.nan, np.nan)
                rec["edge"] = edge
            out[key] = rec
    return out


def cks_block(bars: dict, params: dict) -> dict:
    out = {}
    for agg_s in CKS_AGG_S:
        agg = int(round(agg_s / params["bar_s"]))
        if agg < 1:
            continue
        out[agg_s] = inference.contemporaneous_r2(
            bars["mid"], bars["ofi"], bars["seg"], agg,
            window_bars=int(round(1800 / params["bar_s"])))
    return out


def leadlag_block(cb_events_fn, hl_groups: dict, params: dict) -> dict | None:
    """hl_groups: {"trades": [paths], "l2Book": [paths]} for the same day."""
    if not hl_groups.get("trades") and not hl_groups.get("l2Book"):
        return None
    out = {"trades": {}, "book": {}, "n_cb_trades": 0, "n_hl_trades": 0}
    cb_t, cb_v = leadlag.coinbase_trade_arrays(cb_events_fn())
    out["n_cb_trades"] = int(len(cb_t))
    if hl_groups.get("trades"):
        hl_t, hl_v = leadlag.hyperliquid_trade_arrays(
            iter_files(hl_groups["trades"]))
        out["n_hl_trades"] = int(len(hl_t))
        for dt in TRADE_LL_DT_S:
            r = leadlag.trade_flow_leadlag(cb_t, cb_v, hl_t, hl_v, dt,
                                           params["max_lag_s"])
            out["trades"][dt] = {k: r[k] for k in
                                 ("dt_s", "n", "peak_lag", "peak_lag_s",
                                  "peak_corr", "llr", "corr0")}
    if hl_groups.get("l2Book"):
        hl_t, hl_m = leadlag.hyperliquid_mid_arrays(
            iter_files(hl_groups["l2Book"]))
        cb_mt, cb_mm = _coinbase_mid_arrays(cb_events_fn())
        for dt in BOOK_LL_DT_S:
            r = leadlag.book_series(cb_mt, cb_mm, hl_t, hl_m, dt,
                                    params["book_max_lag_s"])
            out["book"][dt] = {k: r[k] for k in
                               ("dt_s", "n", "peak_lag", "peak_lag_s",
                                "peak_corr", "llr", "corr0")}
    return out


def _coinbase_mid_arrays(events) -> tuple[np.ndarray, np.ndarray]:
    """Mid after every book message, from a light rebuild (no features)."""
    from .book import Book
    book = Book()
    ts, ms = [], []
    for ev in events:
        if ev[0] == "marker":
            if ev[2] == "disconnect":
                book.invalidate()
            continue
        m = ev[3]
        typ = m.get("type")
        if typ == "snapshot":
            book.apply_snapshot(m.get("bids", []), m.get("asks", []))
        elif typ == "l2update" and book.valid:
            book.apply_l2update(m.get("changes", []))
        else:
            continue
        mid = book.mid()
        if mid is not None:
            ts.append(ev[1])
            ms.append(mid)
    return np.asarray(ts, dtype=np.int64), np.asarray(ms, dtype=float)


def analyse_day(key: tuple, group: dict, hl_groups: dict, params: dict,
                log=print) -> tuple[dict, dict]:
    ex, sym, day = key
    paths = [p for _, p, _ in group["files"]]
    sources = sorted({e.get("source", "?") for _, _, e in group["files"]})
    log(f"  {ex}/{sym}/{day}: {len(paths)} file(s), source {','.join(sources)}")
    bars, stats = rebuild(iter_files(paths), params["bar_s"],
                          params["max_gap_s"], params["k_levels"], product=sym)
    log(f"    {stats['n_msgs']:,} msgs, {stats['bars']:,} bars, "
        f"{stats['segments']} segment(s), {stats['n_snapshots']} snapshot(s)")
    res = {
        "key": f"{ex}/{sym}/{day}", "exchange": ex, "symbol": sym, "day": day,
        "sources": sources,
        "files": [{"path": k, "sha256": e.get("sha256", "")[:12],
                   "bytes": e.get("bytes")} for k, _, e in group["files"]],
        "missing": group["missing"],
        "stats": {k: v for k, v in stats.items() if k != "snapshot_checks"},
        "snapshot_checks": stats["snapshot_checks"][:200],
        "cks": cks_block(bars, params) if stats["bars"] else {},
        "forward": forward_block(bars, params) if stats["bars"] else {},
        "leadlag": leadlag_block(lambda: iter_files(paths), hl_groups, params),
        "mean_spread_bps": (float(np.nanmean(bars["spread"]))
                            if stats["bars"] else np.nan),
    }
    return res, bars


# ---------------------------------------------------------------------------
# whole run
# ---------------------------------------------------------------------------

def run_results(manifest_path: Path, data_root: Path | None, params: dict,
                repo_root: Path | None = None, log=print) -> dict:
    manifest_path = Path(manifest_path)
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    groups = days_from_manifest(manifest, manifest_path.parent, repo_root,
                                Path(data_root) if data_root else None)
    coin_days = {k: g for k, g in groups.items() if k[0] == "coinbase" and g["files"]}
    results = {"params": p, "manifest": str(manifest_path),
               "manifest_updated_at": manifest.get("updated_at"),
               "n_manifest_files": len(manifest.get("files", {})),
               "days": [], "pooled": None, "skipped": []}
    for k, g in groups.items():
        if k[0] == "coinbase" and not g["files"]:
            results["skipped"].append({"key": "/".join(k), "reason": "files missing",
                                       "missing": g["missing"]})
    if not coin_days:
        log("no Coinbase product-day with readable files in the manifest")
        return results

    day_bars = []
    for key in sorted(coin_days):
        ex, sym, day = key
        coin = sym.split("-")[0]
        hl = {"trades": [p for _, p, _ in groups.get(("hyperliquid", f"{coin}-trades", day), {"files": []})["files"]],
              "l2Book": [p for _, p, _ in groups.get(("hyperliquid", f"{coin}-l2Book", day), {"files": []})["files"]]}
        res, bars = analyse_day(key, coin_days[key], hl, p, log=log)
        results["days"].append(res)
        if len(bars["t_end"]):
            day_bars.append(bars)

    if day_bars:
        pooled_bars = concat_bars(day_bars)
        log(f"  pooled: {len(pooled_bars['t_end']):,} bars over {len(day_bars)} day(s)")
        fwd = forward_block(pooled_bars, p)
        results["pooled"] = {
            "n_days": len(day_bars), "bars": int(len(pooled_bars["t_end"])),
            "mean_spread_bps": float(np.nanmean(pooled_bars["spread"])),
            "forward": fwd,
            "hurdle": hurdle_block(fwd, float(np.nanmean(pooled_bars["spread"]))),
        }
    return results


def hurdle_block(fwd: dict, spread_bps: float) -> dict:
    out = {}
    for key, rec in fwd.items():
        edge = rec.get("edge")
        if not edge or not np.isfinite(edge.get("edge_bps", np.nan)):
            continue
        out[key] = {"edge_bps": edge["edge_bps"], "ci": edge["ci"],
                    "pred_bps": edge["pred_bps"], "n": edge["n"],
                    "n_pool": edge.get("n_pool"), "frac": edge.get("frac"),
                    "n_at_threshold": edge.get("n_at_threshold"),
                    "spread_bps": spread_bps,
                    "rows": fees.hurdle_rows(edge["edge_bps"], spread_bps)}
    return out


# ---------------------------------------------------------------------------
# json
# ---------------------------------------------------------------------------

def jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(str(x) for x in k)
                 if isinstance(k, tuple) else str(k)): jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
