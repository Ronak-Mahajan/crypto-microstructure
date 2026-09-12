"""Cross-venue lead-lag between Hyperliquid perps and Coinbase spot.

What the data supports (README, record.py): Hyperliquid `trades` carry
millisecond venue time and arrive continuously, so signed trade flow can be
compared at 100 ms to 5 s bins. Hyperliquid `l2Book` is a 20-level SNAPSHOT
pushed at most every ~0.5 s, so anything built from that book is only
meaningful at 1 s or coarser; `book_series` refuses finer bins.

Statistic: with x = Hyperliquid series and y = Coinbase series on a common
grid, corr_k = corr(x_t, y_{t+k}) for k in [-L, L]. A peak at k > 0 means
Hyperliquid leads Coinbase by k bins. The lead-lag ratio
LLR = sum_{k>0} corr_k^2 / sum_{k<0} corr_k^2 (Hayashi-Yoshida style, as in
Huth and Abergel 2014) is > 1 when x leads y.
"""
from __future__ import annotations

import numpy as np

from .features import signed_size_coinbase_match, signed_size_hyperliquid_trade


def bin_series(t_ns: np.ndarray, values: np.ndarray, t0_ns: int, dt_ns: int,
               n_bins: int) -> np.ndarray:
    """Sum `values` into bins [t0 + i dt, t0 + (i+1) dt)."""
    out = np.zeros(n_bins)
    if len(t_ns) == 0:
        return out
    idx = (np.asarray(t_ns, dtype=np.int64) - t0_ns) // dt_ns
    ok = (idx >= 0) & (idx < n_bins)
    np.add.at(out, idx[ok], np.asarray(values, dtype=float)[ok])
    return out


def last_value_series(t_ns: np.ndarray, values: np.ndarray, t0_ns: int,
                      dt_ns: int, n_bins: int) -> np.ndarray:
    """Last observed value per bin, carried forward; NaN before the first."""
    out = np.full(n_bins, np.nan)
    if len(t_ns) == 0:
        return out
    idx = (np.asarray(t_ns, dtype=np.int64) - t0_ns) // dt_ns
    ok = (idx >= 0) & (idx < n_bins)
    idx, vals = idx[ok], np.asarray(values, dtype=float)[ok]
    out[idx] = vals                    # later assignments win = last in bin
    # carry forward
    mask = np.isfinite(out)
    if not mask.any():
        return out
    pos = np.where(mask, np.arange(n_bins), 0)
    np.maximum.accumulate(pos, out=pos)
    first = int(np.argmax(mask))
    out = out[pos]
    out[:first] = np.nan
    return out


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """corr(x_t, y_{t+k}) for k in [-max_lag, max_lag]; NaN rows dropped."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(len(lags), np.nan)
    n = len(x)
    for i, k in enumerate(lags):
        if k >= 0:
            a, b = x[:n - k], y[k:]
        else:
            a, b = x[-k:], y[:n + k]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        a, b = a[ok], b[ok]
        sa, sb = a.std(), b.std()
        if sa <= 0 or sb <= 0:
            continue
        out[i] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return lags, out


def lead_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> dict:
    lags, c = cross_correlation(x, y, max_lag)
    out = {"lags": lags, "corr": c, "n": int(len(x)), "peak_lag": None,
           "peak_corr": np.nan, "llr": np.nan, "corr0": np.nan}
    if not np.isfinite(c).any():
        return out
    i = int(np.nanargmax(np.abs(c)))
    out["peak_lag"] = int(lags[i])
    out["peak_corr"] = float(c[i])
    out["corr0"] = float(c[lags == 0][0])
    pos = np.nansum(c[lags > 0] ** 2)
    neg = np.nansum(c[lags < 0] ** 2)
    out["llr"] = float(pos / neg) if neg > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# series from raw events
# ---------------------------------------------------------------------------

def coinbase_trade_arrays(events) -> tuple[np.ndarray, np.ndarray]:
    ts, vs = [], []
    for ev in events:
        if ev[0] != "msg":
            continue
        m = ev[3]
        if m.get("type") in ("match", "last_match"):
            s = signed_size_coinbase_match(m)
            if s is not None:
                ts.append(ev[1])
                vs.append(s)
    return np.asarray(ts, dtype=np.int64), np.asarray(vs, dtype=float)


def hyperliquid_trade_arrays(events) -> tuple[np.ndarray, np.ndarray]:
    ts, vs = [], []
    for ev in events:
        if ev[0] != "msg":
            continue
        m = ev[3]
        if m.get("channel") != "trades":
            continue
        for t in m.get("data") or []:
            if not isinstance(t, dict):
                continue
            s = signed_size_hyperliquid_trade(t)
            if s is not None:
                ts.append(ev[1])
                vs.append(s)
    return np.asarray(ts, dtype=np.int64), np.asarray(vs, dtype=float)


def hyperliquid_mid_arrays(events) -> tuple[np.ndarray, np.ndarray]:
    ts, vs = [], []
    for ev in events:
        if ev[0] != "msg":
            continue
        m = ev[3]
        if m.get("channel") != "l2Book":
            continue
        data = m.get("data") or {}
        levels = data.get("levels") or []
        if len(levels) < 2 or not levels[0] or not levels[1]:
            continue
        try:
            bid = float(levels[0][0]["px"])
            ask = float(levels[1][0]["px"])
        except (KeyError, TypeError, ValueError):
            continue
        ts.append(ev[1])
        vs.append(0.5 * (bid + ask))
    return np.asarray(ts, dtype=np.int64), np.asarray(vs, dtype=float)


def trade_flow_leadlag(cb_t, cb_v, hl_t, hl_v, dt_s: float,
                       max_lag_s: float) -> dict:
    """Signed-flow cross-correlation at bin width dt_s (100 ms .. 5 s)."""
    if len(cb_t) == 0 or len(hl_t) == 0:
        return {"dt_s": dt_s, "n": 0, "peak_lag": None, "peak_corr": np.nan,
                "llr": np.nan, "corr0": np.nan, "peak_lag_s": np.nan}
    dt_ns = int(round(dt_s * 1e9))
    t0 = int(min(cb_t.min(), hl_t.min()))
    t1 = int(max(cb_t.max(), hl_t.max()))
    n_bins = int((t1 - t0) // dt_ns) + 1
    x = bin_series(hl_t, hl_v, t0, dt_ns, n_bins)
    y = bin_series(cb_t, cb_v, t0, dt_ns, n_bins)
    L = max(1, int(round(max_lag_s / dt_s)))
    r = lead_lag(x, y, L)
    r["dt_s"] = dt_s
    r["peak_lag_s"] = (r["peak_lag"] * dt_s) if r["peak_lag"] is not None else np.nan
    return r


def book_series(cb_t, cb_mid, hl_t, hl_mid, dt_s: float, max_lag_s: float) -> dict:
    """Mid-return cross-correlation from the books; refuses dt_s < 1 s."""
    if dt_s < 1.0:
        raise ValueError("Hyperliquid l2Book is a 0.5 s snapshot feed; book-"
                         "based lead-lag is only defined at 1 s or coarser")
    if len(cb_t) == 0 or len(hl_t) == 0:
        return {"dt_s": dt_s, "n": 0, "peak_lag": None, "peak_corr": np.nan,
                "llr": np.nan, "corr0": np.nan, "peak_lag_s": np.nan}
    dt_ns = int(round(dt_s * 1e9))
    t0 = int(min(cb_t.min(), hl_t.min()))
    t1 = int(max(cb_t.max(), hl_t.max()))
    n_bins = int((t1 - t0) // dt_ns) + 1
    xm = last_value_series(hl_t, hl_mid, t0, dt_ns, n_bins)
    ym = last_value_series(cb_t, cb_mid, t0, dt_ns, n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.diff(np.log(xm)) * 1e4
        y = np.diff(np.log(ym)) * 1e4
    L = max(1, int(round(max_lag_s / dt_s)))
    r = lead_lag(x, y, L)
    r["dt_s"] = dt_s
    r["peak_lag_s"] = (r["peak_lag"] * dt_s) if r["peak_lag"] is not None else np.nan
    return r
