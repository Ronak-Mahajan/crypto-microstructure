"""Inference: purged walk-forward, HAC errors, block bootstrap, CKS R^2.

Everything here is numpy only.

Forward target      y_t = log(mid_{t+h} / mid_t) * 1e4 (bps), defined only
                    when bars t..t+h lie in one contiguous segment.
Purged folds        the bar range [0, n) is cut into n_folds + 1 equal
                    chunks; fold i tests on chunk i+1 and trains on
                    [0, (i+1)*fold - h) (expanding) or [i*fold, (i+1)*fold - h)
                    (rolling). The h-bar purge means no training target is
                    computed from a price inside the test chunk
                    (tests/test_inference.py proves this by construction).
OOS R^2             Campbell-Thompson: 1 - SS_res / sum (y_te - mean(y_tr))^2,
                    the benchmark is the TRAINING mean, never the test
                    fold's own mean; pooled across folds by summing both sums.
HAC                 Newey-West with a Bartlett kernel and bandwidth
                    max(h, 4 (n/100)^(2/9)); overlapping targets are
                    MA(h-1) by construction so the bandwidth must be >= h.
Block bootstrap     moving blocks of `block` consecutive rows, resampled
                    with replacement to the original length; percentile CI.
CKS replication     contemporaneous OLS of mid change on OFI summed over
                    `agg` bars, within segments; day-wide R^2 plus the mean
                    R^2 over 30-minute windows (the paper reports the latter
                    at 10 s aggregation, about 65% for large-tick names).
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# targets and folds
# ---------------------------------------------------------------------------

def forward_returns(mid: np.ndarray, seg: np.ndarray, h: int) -> np.ndarray:
    """bps log return from bar t close to bar t+h close, NaN across gaps."""
    n = len(mid)
    y = np.full(n, np.nan)
    if h <= 0 or n <= h:
        return y
    same = seg[h:] == seg[:-h]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(mid[h:] / mid[:-h]) * 1e4
    head = y[:-h]
    head[same] = r[same]
    return y


def purged_folds(n: int, n_folds: int, h: int,
                 expanding: bool = True) -> list[tuple[int, int, int, int]]:
    """[(train_lo, train_hi, test_lo, test_hi)] with train_hi = test_lo - h."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    fold = n // (n_folds + 1)
    out = []
    for i in range(n_folds):
        test_lo = (i + 1) * fold
        test_hi = (i + 2) * fold if i < n_folds - 1 else n
        train_hi = test_lo - h
        train_lo = 0 if expanding else i * fold
        if train_hi - train_lo < 2 or test_hi - test_lo < 1:
            continue
        out.append((train_lo, train_hi, test_lo, test_hi))
    return out


# ---------------------------------------------------------------------------
# OLS + HAC
# ---------------------------------------------------------------------------

def design(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(len(x)), x])


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def nw_bandwidth(n: int, h: int) -> int:
    return int(max(h, math.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0))))


def hac_cov(X: np.ndarray, resid: np.ndarray, bandwidth: int) -> np.ndarray:
    """Newey-West (Bartlett) covariance of the OLS coefficients."""
    n, p = X.shape
    Xu = X * resid[:, None]
    S = Xu.T @ Xu
    for lag in range(1, bandwidth + 1):
        if lag >= n:
            break
        w = 1.0 - lag / (bandwidth + 1.0)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    bread = np.linalg.pinv(X.T @ X)
    return bread @ S @ bread


def hac_regression(x: np.ndarray, y: np.ndarray, h: int,
                   bandwidth: int | None = None) -> dict:
    """Full-sample OLS of y on [1, x] with HAC t-stats (in-sample)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & (np.isfinite(x) if x.ndim == 1
                           else np.all(np.isfinite(x), axis=1))
    X = design(x[ok])
    yy = y[ok]
    n = len(yy)
    if n < X.shape[1] + 2:
        return {"n": n, "beta": np.nan, "se": np.nan, "t": np.nan,
                "bandwidth": None, "r2": np.nan}
    beta, resid = ols(X, yy)
    L = nw_bandwidth(n, h) if bandwidth is None else int(bandwidth)
    V = hac_cov(X, resid, L)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return {"n": n, "beta": float(beta[1]), "se": float(se[1]),
            "t": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
            "bandwidth": L, "r2": r2, "alpha": float(beta[0])}


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def walk_forward(x: np.ndarray, y: np.ndarray, h: int, n_folds: int,
                 expanding: bool = True) -> dict:
    """Purged walk-forward OLS; returns per-fold stats and pooled OOS R^2.

    `yhat` and `ybar_tr` are arrays aligned to the bars: NaN outside test
    rows. They are what the bootstrap and the fee hurdle consume.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    okx = np.isfinite(x) if x.ndim == 1 else np.all(np.isfinite(x), axis=1)
    ok = np.isfinite(y) & okx
    yhat = np.full(n, np.nan)
    ybar = np.full(n, np.nan)
    folds = []
    ss_res = ss_tot = 0.0
    n_test = 0
    for (tlo, thi, slo, shi) in purged_folds(n, n_folds, h, expanding):
        tr = np.zeros(n, dtype=bool)
        tr[tlo:thi] = True
        tr &= ok
        te = np.zeros(n, dtype=bool)
        te[slo:shi] = True
        te &= ok
        if tr.sum() < 3 or te.sum() < 1:
            continue
        X = design(x[tr])
        beta, _ = ols(X, y[tr])
        ytr_mean = float(y[tr].mean())
        pred = design(x[te]) @ beta
        yhat[te] = pred
        ybar[te] = ytr_mean
        res = float(((y[te] - pred) ** 2).sum())
        tot = float(((y[te] - ytr_mean) ** 2).sum())
        ss_res += res
        ss_tot += tot
        n_test += int(te.sum())
        folds.append({"n_train": int(tr.sum()), "n_test": int(te.sum()),
                      "beta": float(beta[1]) if X.shape[1] == 2 else
                      [float(b) for b in beta[1:]],
                      "r2_oos": (1.0 - res / tot) if tot > 0 else np.nan,
                      "train": (tlo, thi), "test": (slo, shi)})
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    slopes = [f["beta"] for f in folds if not isinstance(f["beta"], list)]
    return {"folds": folds, "r2_oos": r2, "n_test": n_test,
            "n_folds_used": len(folds), "yhat": yhat, "ybar_tr": ybar,
            "sign_agreement": (sum(1 for b in slopes if b > 0), len(slopes))}


def oos_r2(y: np.ndarray, yhat: np.ndarray, ybar_tr: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(ybar_tr)
    if ok.sum() == 0:
        return np.nan
    res = float(((y[ok] - yhat[ok]) ** 2).sum())
    tot = float(((y[ok] - ybar_tr[ok]) ** 2).sum())
    return 1.0 - res / tot if tot > 0 else np.nan


# ---------------------------------------------------------------------------
# block bootstrap
# ---------------------------------------------------------------------------

def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap index vector of length n."""
    block = max(1, min(int(block), n))
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
    return idx[:n]


def bootstrap_ci(stat, arrays: list[np.ndarray], block: int, n_boot: int,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of stat(*arrays) under a moving-block bootstrap.

    `arrays` are equal-length row-aligned arrays (rows with NaN in any of
    them are dropped first so the blocks are over usable rows only).
    """
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a)
    arrs = [np.asarray(a)[ok] for a in arrays]
    n = len(arrs[0])
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = block_indices(n, block, rng)
        vals[b] = stat(*[a[idx] for a in arrs])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# contemporaneous CKS replication
# ---------------------------------------------------------------------------

def _segment_runs(seg: np.ndarray) -> list[tuple[int, int]]:
    if len(seg) == 0:
        return []
    cuts = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(seg)]])
    return list(zip(starts.tolist(), ends.tolist()))


def aggregate_windows(mid: np.ndarray, ofi: np.ndarray, seg: np.ndarray,
                      agg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per window of `agg` bars within a segment: (dP, OFI, window_start)."""
    dps, ofis, starts = [], [], []
    for s, e in _segment_runs(seg):
        L = e - s
        n_win = (L - 1) // agg
        if n_win < 1:
            continue
        cs = np.concatenate([[0.0], np.cumsum(ofi[s:e])])
        for k in range(n_win):
            a = s + 1 + k * agg
            b = a + agg
            if not (np.isfinite(mid[b - 1]) and np.isfinite(mid[a - 1])):
                continue
            dps.append(mid[b - 1] - mid[a - 1])
            ofis.append(cs[b - s] - cs[a - s])
            starts.append(a)
    return (np.asarray(dps, dtype=float), np.asarray(ofis, dtype=float),
            np.asarray(starts, dtype=np.int64))


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    sxx = float(((x - x.mean()) ** 2).sum())
    syy = float(((y - y.mean()) ** 2).sum())
    if sxx <= 0 or syy <= 0:
        return np.nan
    sxy = float(((x - x.mean()) * (y - y.mean())).sum())
    return sxy * sxy / (sxx * syy)


def contemporaneous_r2(mid: np.ndarray, ofi: np.ndarray, seg: np.ndarray,
                       agg: int, window_bars: int = 1800,
                       min_obs: int = 10) -> dict:
    dp, x, starts = aggregate_windows(mid, ofi, seg, agg)
    out = {"agg": agg, "n": int(len(dp)), "r2": np.nan, "beta": np.nan,
           "r2_window_mean": np.nan, "n_windows": 0}
    if len(dp) >= 3:
        out["r2"] = _r2(x, dp)
        sxx = float(((x - x.mean()) ** 2).sum())
        out["beta"] = (float(((x - x.mean()) * (dp - dp.mean())).sum()) / sxx
                       if sxx > 0 else np.nan)
        w = starts // window_bars
        r2s = []
        for wid in np.unique(w):
            sel = w == wid
            if sel.sum() >= min_obs:
                r = _r2(x[sel], dp[sel])
                if np.isfinite(r):
                    r2s.append(r)
        out["r2_window_mean"] = float(np.mean(r2s)) if r2s else np.nan
        out["n_windows"] = len(r2s)
    return out


# ---------------------------------------------------------------------------
# decile edge (for the fee hurdle)
# ---------------------------------------------------------------------------

def decile_edge(y: np.ndarray, yhat: np.ndarray, q: float = 0.1) -> dict:
    """Realised signed move (bps) in the top |prediction| decile.

    Positions are taken in the direction of the prediction; edge is the
    mean of sign(yhat) * y over the top-q fraction of |yhat|.
    """
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 10:
        return {"n": int(ok.sum()), "edge_bps": np.nan, "pred_bps": np.nan}
    yy, pp = y[ok], yhat[ok]
    thr = np.quantile(np.abs(pp), 1.0 - q)
    sel = np.abs(pp) >= thr
    if sel.sum() == 0:
        return {"n": 0, "edge_bps": np.nan, "pred_bps": np.nan}
    s = np.sign(pp[sel])
    return {"n": int(sel.sum()), "edge_bps": float((s * yy[sel]).mean()),
            "pred_bps": float(np.abs(pp[sel]).mean()), "threshold": float(thr)}


def decile_edge_stat(y: np.ndarray, yhat: np.ndarray, q: float = 0.1) -> float:
    return decile_edge(y, yhat, q)["edge_bps"]
