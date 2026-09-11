"""report.py: provenance banner and the fixed section order.

The banner is the guard against the one mistake this repo cannot make: a
table generated from the synthetic fixture being read as a measurement of a
real venue. `source` is a column in the data table, which is easy to miss,
so the banner sits above every table and names the fixture.

The section order is asserted here rather than eyeballed because the whole
point of fixing it is that a bad result cannot be quietly demoted below a
good one.
"""
from __future__ import annotations

from ofi import report

PARAMS = {"bar_s": 1.0, "max_gap_s": 5.0, "k_levels": 5,
          "horizons_s": [1, 10], "n_folds": 3, "expanding": True,
          "n_boot": 10, "seed": 0, "decile": 0.1}


def _day(key: str, sources: list[str]) -> dict:
    return {
        "key": key, "exchange": "coinbase", "symbol": "BTC-USD",
        "day": key.split("/")[-1], "sources": sources,
        "files": [{"path": "x", "sha256": "abc", "bytes": 1}], "missing": [],
        "stats": {"n_msgs": 10, "n_snapshots": 1, "n_disconnects": 0,
                  "bars": 5, "segments": 1, "covered_s": 5.0, "span_s": 5.0,
                  "n_crossed": 0, "n_checks": 0, "n_checks_clean": 0,
                  "mismatch_rate_max": None, "mismatch_rate_clean_max": None},
        "snapshot_checks": [], "cks": {}, "forward": {}, "leadlag": None,
        "mean_spread_bps": 1.0,
    }


def _results(days: list[dict]) -> dict:
    return {"params": PARAMS, "manifest": "m.json", "manifest_updated_at": None,
            "n_manifest_files": len(days), "days": days, "pooled": None,
            "skipped": []}


def test_the_header_names_the_command_that_actually_ran():
    """`analyze.py day` prints a report too, and must not credit `make
    results` for numbers `make results` did not produce."""
    r = _results([_day("coinbase/BTC-USD/2001-01-01", ["synthetic"])])
    assert "by `make results` from" in report.build_markdown(r)
    r["command"] = "python analyze.py day --symbol BTC-USD --day 2001-01-01"
    md = report.build_markdown(r)
    assert "by `python analyze.py day --symbol BTC-USD --day 2001-01-01` from" in md
    assert "make results" not in md.splitlines()[2]


def test_all_synthetic_days_get_the_not_a_market_result_banner():
    md = report.build_markdown(_results([_day("coinbase/BTC-USD/2001-01-01",
                                              ["synthetic"])]))
    assert "SELF-TEST OUTPUT, NOT A MARKET RESULT" in md
    # and it is above the first table, not buried at the bottom
    assert md.index("SELF-TEST OUTPUT") < md.index("## Data")


def test_real_sources_get_no_banner():
    for src in (["tardis"], ["recorder"], ["tardis", "recorder"]):
        md = report.build_markdown(
            _results([_day("coinbase/BTC-USD/2025-08-01", src)]))
        assert "NOT A MARKET RESULT" not in md, src
        assert "WARNING" not in md, src


def test_mixing_fixture_and_real_days_is_flagged_as_mixed():
    md = report.build_markdown(_results([
        _day("coinbase/BTC-USD/2001-01-01", ["synthetic"]),
        _day("coinbase/BTC-USD/2025-08-01", ["tardis"]),
    ]))
    assert "mixes synthetic fixture days" in md
    assert "SELF-TEST OUTPUT" not in md


def test_empty_manifest_says_so_and_claims_nothing():
    md = report.build_markdown(_results([]))
    assert "No day is analysable yet" in md
    assert "## (a)" not in md and "## (b)" not in md


def test_sections_appear_in_the_fixed_a_b_c_d_order():
    md = report.build_markdown(_results([_day("coinbase/BTC-USD/2001-01-01",
                                              ["synthetic"])]))
    heads = ["## Data", "## (a) CKS", "## (b) Forward", "## (c) Fee hurdle",
             "## (d) Cross-venue"]
    pos = [md.index(h) for h in heads]
    assert pos == sorted(pos), list(zip(heads, pos))


def test_banner_helper_ignores_days_with_no_declared_source():
    assert report.provenance_banner(_results([_day("k", [])])) is None


def test_the_gap_column_prints_the_gap_max_not_the_overall_max():
    """Clean check 25% (a replay bug), gapped check 1% (just the gap).

    The data row must read `1: 25.00%` in the no-gap column and `1: 1.00%
    max` in the after-a-gap column. Printing the overall maximum in the gap
    column would report the replay bug as gap damage.
    """
    d = _day("coinbase/BTC-USD/2001-01-01", ["synthetic"])
    d["stats"].update({"n_checks": 2, "n_checks_clean": 1, "n_checks_gap": 1,
                       "mismatch_rate_clean_max": 0.25,
                       "mismatch_rate_gap_max": 0.01,
                       "mismatch_rate_max": 0.25})
    row = [ln for ln in report.build_markdown(_results([d])).splitlines()
           if ln.startswith("| coinbase/BTC-USD/")][0]
    assert "1: 25.00%" in row
    assert "1: 1.00% max" in row
