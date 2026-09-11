"""Inference: purge, HAC, Campbell-Thompson OOS R^2, bootstrap, CKS R^2.

The purge test is constructive rather than a restatement of the slicing
code: prices inside the test fold are physically replaced and the fitted
training coefficients are required to be bit-identical.

The HAC test uses an AR(1) series whose long-run/short-run variance ratio
is known in closed form, so the Newey-West estimator is checked against a
number that does not come from this repo.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ofi import inference as inf

SEG0 = None


def seg_of(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


# ---------------------------------------------------------------------------
# forward targets
# ---------------------------------------------------------------------------

def test_forward_returns_are_log_bps_and_end_in_nan():
    mid = np.array([100.0, 101.0, 102.0, 103.0])
    y = inf.forward_returns(mid, seg_of(4), 1)
    assert y[0] == pytest.approx(math.log(101 / 100) * 1e4)
    assert y[2] == pytest.approx(math.log(103 / 102) * 1e4)
    assert np.isnan(y[3])


def test_forward_returns_never_cross_a_segment_boundary():
    """Bars 0-2 are segment 0 and bars 3-5 segment 1: a 2-bar forward return
    may not be formed from bar 2 (it would span the outage)."""
    mid = np.array([100.0, 101.0, 102.0, 200.0, 201.0, 202.0])
    seg = np.array([0, 0, 0, 1, 1, 1])
    y = inf.forward_returns(mid, seg, 2)
    assert y[0] == pytest.approx(math.log(102 / 100) * 1e4)
    assert np.isnan(y[1])        # would land in segment 1
    assert np.isnan(y[2])        # would land in segment 1
    assert y[3] == pytest.approx(math.log(202 / 200) * 1e4)
    assert np.isnan(y[4]) and np.isnan(y[5])


def test_forward_returns_degenerate_horizons():
    mid = np.arange(1.0, 6.0)
    assert np.all(np.isnan(inf.forward_returns(mid, seg_of(5), 0)))
    assert np.all(np.isnan(inf.forward_returns(mid, seg_of(5), 9)))


# ---------------------------------------------------------------------------
# purged folds
# ---------------------------------------------------------------------------

def test_purged_folds_end_training_h_bars_before_the_test_chunk():
    folds = inf.purged_folds(1000, 4, 30)
    assert len(folds) == 4
    for tlo, thi, slo, shi in folds:
        assert thi == slo - 30
        assert tlo < thi <= slo < shi
    assert folds[0][2] == 200 and folds[-1][3] == 1000     # last fold takes the tail


def test_rolling_folds_do_not_reach_back_to_the_start():
    folds = inf.purged_folds(1000, 4, 10, expanding=False)
    assert [f[0] for f in folds] == [0, 200, 400, 600]
    folds_x = inf.purged_folds(1000, 4, 10, expanding=True)
    assert [f[0] for f in folds_x] == [0, 0, 0, 0]


def test_purged_folds_drop_a_fold_the_purge_would_empty():
    """n = 100, 4 folds -> chunks of 20. With h = 25 the first fold would
    have to train on [0, -5), so it is dropped rather than silently trained
    on nothing; the later folds survive with a shortened training slice."""
    folds = inf.purged_folds(100, 4, 25)
    assert [f[2] for f in folds] == [40, 60, 80]     # the 20-test fold is gone
    assert folds[0][:2] == (0, 15)
    # a purge wider than the whole history leaves nothing at all
    assert inf.purged_folds(100, 4, 100) == []
    with pytest.raises(ValueError):
        inf.purged_folds(100, 0, 1)


def test_no_test_fold_price_can_reach_the_training_fit():
    """Constructive purge proof.

    Build a mid series, take purged folds, then REPLACE every price from the
    first test index onwards with a wildly different path. If any training
    target were computed from a price inside the test fold, the fitted slope
    of fold 0 would move. It must not move at all.
    """
    rng = np.random.default_rng(3)
    n, h, n_folds = 1200, 20, 4
    x = rng.normal(size=n)
    steps = 0.5 * x + rng.normal(scale=0.5, size=n)          # signal + noise
    mid = 100.0 * np.exp(np.cumsum(steps) / 1e4)
    seg = seg_of(n)

    folds = inf.purged_folds(n, n_folds, h)
    test_lo = folds[0][2]

    mid2 = mid.copy()
    mid2[test_lo:] = 100.0 * np.exp(np.cumsum(rng.normal(size=n - test_lo)))

    y1 = inf.forward_returns(mid, seg, h)
    y2 = inf.forward_returns(mid2, seg, h)
    # the two target series differ from test_lo - h onwards and nowhere before
    assert np.allclose(y1[:test_lo - h], y2[:test_lo - h], equal_nan=True)
    assert not np.allclose(y1[test_lo - h:test_lo], y2[test_lo - h:test_lo],
                           equal_nan=True)

    f1 = inf.walk_forward(x, y1, h, n_folds)["folds"][0]
    f2 = inf.walk_forward(x, y2, h, n_folds)["folds"][0]
    assert f1["n_train"] == f2["n_train"]
    assert f1["beta"] == f2["beta"]          # exact, not approx


def test_walk_forward_recovers_a_planted_slope_out_of_sample():
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.5, size=n)
    wf = inf.walk_forward(x, y, 1, 4)
    assert wf["n_folds_used"] == 4
    assert wf["r2_oos"] > 0.9
    assert all(f["beta"] == pytest.approx(2.0, abs=0.1) for f in wf["folds"])
    assert wf["sign_agreement"] == (4, 4)
    # predictions exist only on test rows
    assert np.isnan(wf["yhat"][: inf.purged_folds(n, 4, 1)[0][2]]).all()


def test_oos_r2_benchmarks_against_the_training_mean_not_the_test_mean():
    """Campbell-Thompson. y has mean 10 in the test rows but the training
    mean is 0, and a model predicting 0 must therefore score a NEGATIVE OOS
    R^2, not the 0 that a test-fold-mean benchmark would give."""
    y = np.full(100, 10.0)
    yhat = np.zeros(100)
    ybar_tr = np.zeros(100)
    assert inf.oos_r2(y, yhat, ybar_tr) == pytest.approx(0.0)   # same benchmark
    yhat2 = np.full(100, 5.0)
    # res = 100*25, tot = 100*100 -> 1 - 0.25 = 0.75
    assert inf.oos_r2(y, yhat2, ybar_tr) == pytest.approx(0.75)
    # against the test fold's own mean the totals would be 0 and the number
    # meaningless; the function never uses it.
    assert np.isnan(inf.oos_r2(y, yhat2, np.full(100, 10.0)))


# ---------------------------------------------------------------------------
# HAC
# ---------------------------------------------------------------------------

def test_nw_bandwidth_is_never_below_the_overlap():
    assert inf.nw_bandwidth(1000, 300) == 300
    assert inf.nw_bandwidth(1_000_000, 1) >= 4
    assert inf.nw_bandwidth(100, 1) == 4


def test_hac_recovers_the_known_ar1_variance_ratio():
    """For an AR(1) with parameter rho, gamma_l = gamma_0 rho^l, so the
    Bartlett long-run variance with bandwidth L is

        S / gamma_0 = 1 + 2 sum_{l=1..L} (1 - l/(L+1)) rho^l

    which for rho = 0.5, L = 50 is 2.9216 (the untruncated limit is
    (1+rho)/(1-rho) = 3). Regressing the series on a constant makes the HAC
    coefficient variance exactly S/n and the iid one gamma_0/n, so their
    ratio must hit that number.
    """
    rho, n, L = 0.5, 400_000, 50
    rng = np.random.default_rng(17)
    eps = rng.normal(size=n)
    u = np.empty(n)
    u[0] = eps[0] / math.sqrt(1 - rho ** 2)
    for i in range(1, n):
        u[i] = rho * u[i - 1] + eps[i]

    X = np.ones((n, 1))
    resid = u - u.mean()
    V_hac = inf.hac_cov(X, resid, L)[0, 0]
    V_iid = float((resid ** 2).sum()) / n ** 2      # bandwidth 0 equivalent
    expected = 1.0 + 2.0 * sum((1.0 - l / (L + 1.0)) * rho ** l
                               for l in range(1, L + 1))
    assert expected == pytest.approx(2.9216, abs=1e-3)
    assert V_hac / V_iid == pytest.approx(expected, rel=0.10)
    # and it is well below the untruncated 3.0 only by the Bartlett taper
    assert 2.5 < V_hac / V_iid < 3.4


def test_hac_bandwidth_zero_reduces_to_the_white_sandwich():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    X = inf.design(x)
    resid = rng.normal(size=n)
    Xu = X * resid[:, None]
    bread = np.linalg.pinv(X.T @ X)
    expected = bread @ (Xu.T @ Xu) @ bread
    assert np.allclose(inf.hac_cov(X, resid, 0), expected)


def test_hac_t_stat_shrinks_when_the_errors_are_persistent():
    """Same slope, same residual variance, but autocorrelated errors: the
    HAC t-stat must be materially smaller than the naive OLS one."""
    rng = np.random.default_rng(23)
    n, rho, h = 20_000, 0.9, 20
    eps = rng.normal(size=n)
    u = np.empty(n)
    u[0] = eps[0]
    for i in range(1, n):
        u[i] = rho * u[i - 1] + eps[i]
    x = rng.normal(size=n)
    y = 0.05 * x + u
    r = inf.hac_regression(x, y, h)
    assert r["bandwidth"] >= h
    naive_se = math.sqrt(float(((y - y.mean()) ** 2).sum()) / n /
                         float(((x - x.mean()) ** 2).sum()))
    assert r["se"] > naive_se          # persistence widens the interval
    assert r["beta"] == pytest.approx(0.05, abs=0.05)


def test_hac_regression_refuses_a_sample_it_cannot_fit():
    r = inf.hac_regression(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1)
    assert r["n"] == 2 and np.isnan(r["beta"])


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def test_block_indices_are_contiguous_runs_inside_the_sample():
    rng = np.random.default_rng(1)
    idx = inf.block_indices(100, 10, rng)
    assert len(idx) == 100
    assert idx.min() >= 0 and idx.max() < 100
    for s in range(0, 100, 10):
        blk = idx[s:s + 10]
        assert list(blk) == list(range(blk[0], blk[0] + len(blk)))


def test_bootstrap_ci_brackets_the_sample_statistic_at_the_right_width():
    """iid data, so the moving-block CI must reproduce the textbook
    +/- 1.96 sigma / sqrt(n) width around the SAMPLE mean (not the
    population mean, which a resampling interval knows nothing about)."""
    rng = np.random.default_rng(2)
    n = 2000
    a = rng.normal(loc=3.0, scale=1.0, size=n)
    lo, hi = inf.bootstrap_ci(lambda z: float(z.mean()), [a], block=20,
                              n_boot=400, seed=4)
    assert lo < a.mean() < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 * a.std() / math.sqrt(n), rel=0.25)


def test_bootstrap_ci_widens_when_the_data_are_autocorrelated():
    """A block bootstrap must charge for persistence: the same marginal
    variance with rho = 0.9 gives a much wider interval for the mean."""
    n, rho = 4000, 0.9
    rng = np.random.default_rng(6)
    eps = rng.normal(size=n)
    u = np.empty(n)
    u[0] = eps[0] / math.sqrt(1 - rho ** 2)
    for i in range(1, n):
        u[i] = rho * u[i - 1] + eps[i] * math.sqrt(1 - rho ** 2)
    iid = rng.normal(size=n)
    w_ar = np.subtract(*reversed(inf.bootstrap_ci(
        lambda z: float(z.mean()), [u], block=100, n_boot=300, seed=1)))
    w_iid = np.subtract(*reversed(inf.bootstrap_ci(
        lambda z: float(z.mean()), [iid], block=100, n_boot=300, seed=1)))
    assert w_ar > 2.0 * w_iid


def test_bootstrap_ci_on_too_little_data_is_nan():
    lo, hi = inf.bootstrap_ci(lambda z: float(z.mean()),
                              [np.array([np.nan, 1.0])], 5, 50)
    assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# contemporaneous CKS replication
# ---------------------------------------------------------------------------

def test_aggregate_windows_sums_flow_over_the_window_it_prices():
    mid = np.array([100.0, 101.0, 103.0, 106.0, 110.0])
    ofi = np.array([9.0, 1.0, 2.0, 3.0, 4.0])         # bar 0's flow is pre-window
    dp, x, starts = inf.aggregate_windows(mid, ofi, seg_of(5), 2)
    # windows of 2 bars starting at index 1: [1,2] and [3,4]
    assert list(dp) == [103.0 - 100.0, 110.0 - 103.0]
    assert list(x) == [1.0 + 2.0, 3.0 + 4.0]
    assert list(starts) == [1, 3]


def test_aggregate_windows_never_spans_a_segment():
    mid = np.array([100.0, 101.0, 500.0, 501.0])
    seg = np.array([0, 0, 1, 1])
    dp, x, _ = inf.aggregate_windows(mid, np.ones(4), seg, 1)
    assert list(dp) == [1.0, 1.0]         # within-segment steps only
    assert len(dp) == 2


def test_contemporaneous_r2_is_one_when_price_is_exactly_lambda_times_ofi():
    rng = np.random.default_rng(8)
    n = 3000
    ofi = rng.normal(size=n)
    mid = 100.0 + 0.01 * np.cumsum(ofi)
    r = inf.contemporaneous_r2(mid, ofi, seg_of(n), agg=10, window_bars=500)
    assert r["r2"] == pytest.approx(1.0, abs=1e-9)
    assert r["beta"] == pytest.approx(0.01, abs=1e-9)
    assert r["n"] == (n - 1) // 10
    assert r["r2_window_mean"] == pytest.approx(1.0, abs=1e-9)
    assert r["n_windows"] > 1


def test_contemporaneous_r2_is_small_when_ofi_is_unrelated_to_price():
    rng = np.random.default_rng(9)
    n = 3000
    ofi = rng.normal(size=n)
    mid = 100.0 + 0.01 * np.cumsum(rng.normal(size=n))
    r = inf.contemporaneous_r2(mid, ofi, seg_of(n), agg=10)
    assert r["r2"] < 0.05


def test_contemporaneous_r2_on_an_empty_series():
    r = inf.contemporaneous_r2(np.zeros(0), np.zeros(0),
                               np.zeros(0, dtype=np.int64), agg=10)
    assert r["n"] == 0 and np.isnan(r["r2"])


# ---------------------------------------------------------------------------
# decile edge
# ---------------------------------------------------------------------------

def test_decile_edge_takes_the_direction_of_the_prediction():
    y = np.array([10.0, -10.0, 0.1, -0.1] * 25)
    yhat = np.array([1.0, -1.0, 0.01, -0.01] * 25)
    e = inf.decile_edge(y, yhat, q=0.5)
    assert e["n"] == 50
    assert e["edge_bps"] == pytest.approx(10.0)
    assert e["pred_bps"] == pytest.approx(1.0)


def test_decile_edge_is_negative_when_the_signal_is_backwards():
    y = np.array([-10.0, 10.0] * 50)
    yhat = np.array([1.0, -1.0] * 50)
    assert inf.decile_edge(y, yhat, q=0.5)["edge_bps"] == pytest.approx(-10.0)


def test_decile_edge_on_too_few_points():
    assert np.isnan(inf.decile_edge(np.zeros(3), np.zeros(3))["edge_bps"])
