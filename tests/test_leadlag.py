"""Cross-venue lead-lag: binning, cross-correlation sign convention, limits.

Sign convention under test: corr_k = corr(x_t, y_{t+k}), so a peak at
k > 0 means x (Hyperliquid) LEADS y (Coinbase) by k bins.
"""
from __future__ import annotations

import numpy as np
import pytest

from ofi import leadlag as ll

S = 1_000_000_000


def test_bin_series_sums_into_half_open_bins():
    t = np.array([0, S // 2, S, 2 * S - 1, 5 * S], dtype=np.int64)
    v = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    out = ll.bin_series(t, v, 0, S, 4)
    assert list(out) == [3.0, 12.0, 0.0, 0.0]     # 5 s is past the last bin


def test_bin_series_ignores_events_before_the_start():
    t = np.array([-S, 0], dtype=np.int64)
    out = ll.bin_series(t, np.array([9.0, 1.0]), 0, S, 2)
    assert list(out) == [1.0, 0.0]


def test_last_value_series_carries_forward_and_is_nan_before_the_first():
    t = np.array([S, S + 10, 3 * S], dtype=np.int64)
    v = np.array([100.0, 101.0, 105.0])
    out = ll.last_value_series(t, v, 0, S, 5)
    assert np.isnan(out[0])
    assert out[1] == 101.0            # last value inside the bin wins
    assert out[2] == 101.0            # carried
    assert out[3] == 105.0
    assert out[4] == 105.0


def test_cross_correlation_of_a_series_with_itself_peaks_at_zero():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    lags, c = ll.cross_correlation(x, x, 5)
    assert list(lags) == list(range(-5, 6))
    assert c[lags == 0][0] == pytest.approx(1.0)
    assert np.nanmax(np.abs(c[lags != 0])) < 0.2


def test_a_lead_of_three_bins_is_found_at_lag_plus_three():
    """y_t = x_{t-3}: Coinbase repeats Hyperliquid three bins later, so
    corr(x_t, y_{t+3}) = 1 and the LLR must be well above 1."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    y = np.concatenate([np.zeros(3), x[:-3]])
    r = ll.lead_lag(x, y, 8)
    assert r["peak_lag"] == 3
    assert r["peak_corr"] == pytest.approx(1.0, abs=1e-6)
    assert r["llr"] > 5.0
    assert abs(r["corr0"]) < 0.1


def test_a_lag_is_found_at_a_negative_lag_and_flips_the_llr():
    rng = np.random.default_rng(3)
    y = rng.normal(size=5000)
    x = np.concatenate([np.zeros(2), y[:-2]])     # x repeats y: x follows
    r = ll.lead_lag(x, y, 8)
    assert r["peak_lag"] == -2
    assert r["llr"] < 1.0


def test_lead_lag_on_a_constant_series_is_all_nan_not_a_crash():
    r = ll.lead_lag(np.zeros(100), np.zeros(100), 3)
    assert r["peak_lag"] is None
    assert np.isnan(r["peak_corr"])


def test_trade_flow_leadlag_recovers_a_planted_200ms_lead():
    """Hyperliquid prints at t, Coinbase copies the same signed flow 200 ms
    later. At a 100 ms bin the peak must be at +2 bins = +0.2 s."""
    rng = np.random.default_rng(4)
    n = 4000
    step = S // 10                                 # 100 ms
    hl_t = (np.arange(n) * step).astype(np.int64)
    hl_v = rng.normal(size=n)
    cb_t = hl_t + 2 * step
    cb_v = hl_v.copy()
    r = ll.trade_flow_leadlag(cb_t, cb_v, hl_t, hl_v, dt_s=0.1, max_lag_s=1.0)
    assert r["peak_lag"] == 2
    assert r["peak_lag_s"] == pytest.approx(0.2)
    assert r["llr"] > 5.0


def test_trade_flow_leadlag_with_one_side_empty():
    r = ll.trade_flow_leadlag(np.zeros(0, dtype=np.int64), np.zeros(0),
                              np.array([0], dtype=np.int64), np.array([1.0]),
                              0.1, 1.0)
    assert r["n"] == 0 and r["peak_lag"] is None


def test_book_leadlag_refuses_sub_second_bins():
    """The Hyperliquid l2Book feed is a 20-level snapshot pushed at most
    every 0.5 s, so a 100 ms book lead-lag is not a measurable quantity and
    the function must say so instead of returning a number."""
    t = np.arange(10, dtype=np.int64) * S
    m = np.full(10, 100.0)
    with pytest.raises(ValueError, match="0.5 s snapshot"):
        ll.book_series(t, m, t, m, dt_s=0.1, max_lag_s=1.0)
    r = ll.book_series(t, m, t, m, dt_s=1.0, max_lag_s=3.0)
    assert r["dt_s"] == 1.0


def test_coinbase_and_hyperliquid_trade_extractors():
    events = [
        ("msg", 1, None, {"type": "match", "side": "sell", "size": "2"}),
        ("msg", 2, None, {"type": "last_match", "side": "buy", "size": "1"}),
        ("msg", 3, None, {"type": "l2update", "changes": []}),
        ("marker", 4, "disconnect", {}),
    ]
    t, v = ll.coinbase_trade_arrays(events)
    assert list(t) == [1, 2] and list(v) == [2.0, -1.0]

    hl = [("msg", 10, None, {"channel": "trades",
                             "data": [{"side": "B", "sz": "0.5"},
                                      {"side": "A", "sz": "0.25"}]}),
          ("msg", 11, None, {"channel": "l2Book", "data": {}})]
    t, v = ll.hyperliquid_trade_arrays(hl)
    assert list(t) == [10, 10] and list(v) == [0.5, -0.25]


def test_hyperliquid_mid_extractor_reads_the_20_level_snapshot():
    events = [("msg", 7, None, {"channel": "l2Book", "data": {"levels": [
        [{"px": "100.0", "sz": "1"}, {"px": "99.0", "sz": "2"}],
        [{"px": "102.0", "sz": "1"}]]}}),
        ("msg", 8, None, {"channel": "l2Book", "data": {"levels": [[], []]}})]
    t, m = ll.hyperliquid_mid_arrays(events)
    assert list(t) == [7] and list(m) == [101.0]
