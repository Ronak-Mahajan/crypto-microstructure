"""Whole-pipeline self-test over the committed synthetic fixture.

The fixture (tests/fixtures/make_fixture.py, seed 20260911) is NOT market
data. It is a toy feed in which order flow moves the price by an exactly
known mechanism and a Hyperliquid trade stream leads the Coinbase matches
by exactly 300 ms. That makes three things checkable end to end:

1. the diff replay is lossless   -> every no-gap snapshot check is 0.00%
2. the CKS anchor actually fires -> contemporaneous R^2 is high
3. nothing leaks from the future -> forward OOS R^2 is ~0 and the edge does
   not clear any fee schedule

If a change to book/features/bars/inference breaks any of those, this test
fails before the report can print a wrong table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ofi import inference as inf
from ofi import leadlag as ll
from ofi import report
from ofi.io import iter_files, iter_product
from ofi.pipeline import rebuild
from ofi.run import DEFAULT_PARAMS, run_results

FIX = Path(__file__).resolve().parent / "fixtures" / "synthetic"
DATA = FIX / "data"
MANIFEST = FIX / "manifest.json"
DAY = "2001-01-01"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="run `python tests/fixtures/make_fixture.py` to build the fixture")


@pytest.fixture(scope="module")
def bars_stats():
    return rebuild(iter_product(DATA, "coinbase", "BTC-USD"),
                   bar_s=1.0, max_gap_s=5.0, k_levels=5, product="BTC-USD")


def test_manifest_hashes_still_match_the_committed_bytes():
    import tardis_loader as tl
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(m["files"]) == 3
    for key, entry in m["files"].items():
        p = MANIFEST.parent / key
        assert p.exists(), key
        assert tl.sha256_of(p) == entry["sha256"], key
        assert entry["source"] == "synthetic"
        assert entry["day"] == DAY


def test_every_clean_snapshot_check_is_exactly_zero(bars_stats):
    """The generator re-emits the true book every 4 minutes. Replaying the
    diffs must reproduce it level for level, or the book engine is wrong."""
    _, st = bars_stats
    assert st["n_checks_clean"] >= 5
    assert st["mismatch_rate_clean_max"] == 0.0
    clean = [c for c in st["snapshot_checks"] if not c["after_gap"]]
    assert all(c["mismatched"] == 0 for c in clean)
    assert all(c["best_bid_match"] and c["best_ask_match"] for c in clean)


def test_the_planted_outage_shows_up_as_a_gap_not_as_stale_bars(bars_stats):
    bars, st = bars_stats
    assert st["n_disconnects"] == 1
    assert st["segments"] == 2
    assert int(bars["is_gap"].sum()) == 1
    # 90 s of outage inside a ~1800 s span: the old harness would have made
    # 90 stale bars, so book time must fall short of the span by about that
    assert st["span_s"] - st["covered_s"] > 60.0
    after_gap = [c for c in st["snapshot_checks"] if c["after_gap"]]
    assert len(after_gap) == 1
    assert after_gap[0]["rate"] > 0.0      # the book really did move unseen


def test_the_book_never_crosses_and_the_spread_is_one_tick(bars_stats):
    bars, st = bars_stats
    assert st["n_crossed"] == 0
    assert np.nanmean(bars["spread"]) == pytest.approx(1.0, abs=0.05)


def test_cks_anchor_recovers_the_planted_price_impact(bars_stats):
    """Price moves one tick per 2 units of accumulated OFI by construction,
    so the contemporaneous regression must explain most of the variance and
    must get sharper as the aggregation window grows."""
    bars, _ = bars_stats
    r2 = {}
    for agg in (1, 10, 60):
        r = inf.contemporaneous_r2(bars["mid"], bars["ofi"], bars["seg"], agg)
        r2[agg] = r["r2"]
        assert r["beta"] > 0
    assert r2[1] > 0.7
    assert r2[10] > 0.8
    assert r2[60] > 0.8
    assert r2[60] >= r2[1] - 1e-9


def test_forward_predictability_is_a_null_on_the_fixture(bars_stats):
    """Nothing is planted in the forward direction: the pressure that has
    not moved the price yet is already in the bar's own OFI. An OOS R^2
    materially above zero here would mean the harness is leaking."""
    bars, _ = bars_stats
    for h in (1, 10, 60):
        y = inf.forward_returns(bars["mid"], bars["seg"], h)
        wf = inf.walk_forward(bars["ofi"], y, h, 3)
        assert wf["n_folds_used"] == 3
        assert abs(wf["r2_oos"]) < 0.05, (h, wf["r2_oos"])


def test_forward_returns_never_bridge_the_outage(bars_stats):
    bars, _ = bars_stats
    y = inf.forward_returns(bars["mid"], bars["seg"], 30)
    seg = bars["seg"]
    finite = np.isfinite(y)
    assert finite.sum() > 100
    assert np.all(seg[np.flatnonzero(finite) + 30] == seg[finite])


def test_leadlag_finds_the_planted_300ms_hyperliquid_lead():
    cb_t, cb_v = ll.coinbase_trade_arrays(
        iter_product(DATA, "coinbase", "BTC-USD"))
    hl_t, hl_v = ll.hyperliquid_trade_arrays(
        iter_product(DATA, "hyperliquid", "BTC-trades"))
    assert len(cb_t) > 100 and len(hl_t) > 100
    r = ll.trade_flow_leadlag(cb_t, cb_v, hl_t, hl_v, dt_s=0.1, max_lag_s=1.0)
    assert r["peak_lag_s"] == pytest.approx(0.3, abs=0.1)
    assert r["llr"] > 2.0


def test_hyperliquid_book_stream_is_readable_at_its_stated_cadence():
    t, m = ll.hyperliquid_mid_arrays(iter_product(DATA, "hyperliquid", "BTC-l2Book"))
    assert len(t) > 1000
    dt = np.diff(t) / 1e9
    assert np.median(dt) == pytest.approx(0.5, abs=0.05)
    with pytest.raises(ValueError):
        ll.book_series(t, m, t, m, dt_s=0.1, max_lag_s=1.0)


def test_full_run_produces_a_report_whose_fee_hurdle_is_negative(tmp_path):
    """The end-to-end `make smoke` path, in-process."""
    params = dict(DEFAULT_PARAMS)
    params.update({"n_folds": 3, "n_boot": 30,
                   "horizons_s": [1, 10, 60]})
    res = run_results(MANIFEST, None, params, log=lambda *a, **k: None)
    assert len(res["days"]) == 1
    assert res["days"][0]["key"] == f"coinbase/BTC-USD/{DAY}"
    assert res["days"][0]["leadlag"]["n_hl_trades"] > 100
    pooled = res["pooled"]
    assert pooled["n_days"] == 1 and pooled["bars"] > 1000

    hurdle = pooled["hurdle"]
    assert hurdle, "no out-of-sample predictions reached the fee table"
    for rec in hurdle.values():
        takes = [r for r in rec["rows"] if r["scenario"].startswith("take")]
        assert takes and all(r["net_bps"] < 0 for r in takes)
        # the traded share is declared on every row, and it is a share
        assert rec["n_pool"] > 0
        assert rec["frac"] == pytest.approx(rec["n"] / rec["n_pool"])
        assert 0.0 < rec["frac"] <= 1.0
    # the fixture's flow features are zero most of the time, so at least one
    # row must land on a tied plateau -- that is the case the column exists
    # for, and it must not be silently reported as a clean decile
    assert any(r["frac"] > 0.15 for r in hurdle.values())

    md, js = report.write(res, tmp_path)
    text = md.read_text(encoding="utf-8")
    for heading in ("## Data", "## (a) CKS contemporaneous replication",
                    "## (b) Forward predictability", "## (c) Fee hurdle",
                    "## (d) Cross-venue lead-lag"):
        assert heading in text
    assert "synthetic" in text
    assert "| 0.00% |" in text or "0.00%" in text
    assert "share of OOS" in text
    assert "tied)" in text, "a tied plateau must be declared in the table"
    # every table in the real generated report must render: same cell count
    # in the header, the separator and every body row
    from test_report import assert_tables_well_formed
    assert_tables_well_formed(text, expect_at_least=5)
    loaded = json.loads(js.read_text(encoding="utf-8"))
    assert loaded["pooled"]["n_days"] == 1
    # nothing that lands in a committed artifact may carry an absolute path
    assert loaded["manifest"] == "tests/fixtures/synthetic/manifest.json"


def test_report_says_so_when_no_day_is_analysable(tmp_path):
    empty = tmp_path / "manifest.json"
    empty.write_text(json.dumps({"schema": 1, "files": {}, "days": {}}),
                     encoding="utf-8")
    res = run_results(empty, None, {}, log=lambda *a, **k: None)
    md, _ = report.write(res, tmp_path / "out")
    text = md.read_text(encoding="utf-8")
    assert "No day is analysable yet" in text
    assert res["days"] == []


def test_missing_files_are_listed_rather_than_silently_skipped(tmp_path):
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    m["files"]["data/coinbase/BTC-USD/29991231-23.jsonl.gz"] = {
        "sha256": "0" * 64, "bytes": 0, "source": "tardis",
        "exchange": "coinbase", "symbol": "BTC-USD", "day": "2999-12-31"}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    res = run_results(p, None, {"n_folds": 2, "n_boot": 10,
                                "horizons_s": [1]}, log=lambda *a, **k: None)
    assert any(s["key"] == "coinbase/BTC-USD/2999-12-31"
               for s in res["skipped"])


def test_iter_files_reads_the_marker_line():
    kinds = [e[0] for e in iter_files(
        [DATA / "coinbase" / "BTC-USD" / "20010101-00.jsonl.gz"])]
    assert kinds.count("marker") >= 3        # connect, disconnect, connect
    assert kinds.count("msg") > 1000
