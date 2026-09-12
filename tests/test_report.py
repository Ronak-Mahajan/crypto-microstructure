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

import re

from ofi import report

PARAMS = {"bar_s": 1.0, "max_gap_s": 5.0, "k_levels": 5,
          "horizons_s": [1, 10], "n_folds": 3, "expanding": True,
          "n_boot": 10, "seed": 0, "decile": 0.1}

_SEP_RE = re.compile(r"\s*\|[\s:|-]+\|\s*\Z")


def md_cells(row: str) -> list[str]:
    r"""Split one GFM table row on UNESCAPED pipes (`\|` is a literal pipe)."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    out: list[str] = []
    cur: list[str] = []
    esc = False
    for ch in s:
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            cur.append(ch)
            esc = True
        elif ch == "|":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def md_tables(text: str) -> list[tuple[int, list[str], list[str], list[list[str]]]]:
    """Every markdown table as (line_no, header, separator, body rows)."""
    lines = text.splitlines()
    tables = []
    i = 0
    while i < len(lines):
        if (lines[i].strip().startswith("|") and i + 1 < len(lines)
                and _SEP_RE.fullmatch(lines[i + 1])):
            hdr, sep = md_cells(lines[i]), md_cells(lines[i + 1])
            j = i + 2
            body = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append(md_cells(lines[j]))
                j += 1
            tables.append((i + 1, hdr, sep, body))
            i = j
        else:
            i += 1
    return tables


def assert_tables_well_formed(text: str, expect_at_least: int = 1) -> None:
    """Header, separator and every body row must have the same cell count.

    A bare `|` inside a heading silently ENDS that cell, so a header can end
    up wider than its separator; GitHub then renders the table with the
    trailing columns unnamed and every heading after the stray pipe shifted
    left. It costs nothing to check and it is invisible until someone opens
    the page.
    """
    tables = md_tables(text)
    assert len(tables) >= expect_at_least, f"only {len(tables)} table(s) found"
    for line_no, hdr, sep, body in tables:
        assert len(hdr) == len(sep), (
            f"table at line {line_no}: {len(hdr)} header cells vs "
            f"{len(sep)} separator cells: {hdr}")
        for k, row in enumerate(body):
            assert len(row) == len(sep), (
                f"table at line {line_no}, body row {k}: {len(row)} cells "
                f"vs {len(sep)}: {row}")


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


def _pooled_with_hurdle() -> dict:
    """A pooled block shaped exactly like run.hurdle_block's output."""
    from ofi import fees
    rec = {"h_s": 1, "h_bars": 1, "n_valid": 100, "insufficient": False,
           "r2_oos": -0.001, "n_test": 100, "n_folds_used": 3,
           "sign_agreement": (3, 3), "fold_r2": [], "fold_beta": [],
           "hac_beta": 0.01, "hac_se": 0.005, "hac_t": 2.0,
           "hac_bandwidth": 7, "insample_r2": 0.01, "ci": (-0.01, 0.01),
           "edge": None}
    hurdle = {("ofi", 1): {"edge_bps": 0.04, "ci": (0.02, 0.06),
                           "pred_bps": 0.06, "n": 10, "n_pool": 100,
                           "frac": 0.1, "n_at_threshold": 4,
                           "spread_bps": 1.0, "spread_pooled_bps": 1.0,
                           "rows": fees.hurdle_rows(0.04, 1.0)}}
    return {"n_days": 1, "bars": 100, "mean_spread_bps": 1.0,
            "forward": {("ofi", 1): rec}, "hurdle": hurdle}


def test_the_cell_splitter_honours_escaped_pipes():
    """Otherwise the well-formedness guard below would pass vacuously."""
    assert md_cells(r"| a | b |") == [" a ", " b "]
    assert md_cells(r"| mean \|pred\| (bps) | x |") == \
        [r" mean \|pred\| (bps) ", " x "]


def test_every_table_in_a_report_is_well_formed():
    r = _results([_day("coinbase/BTC-USD/2001-01-01", ["synthetic"])])
    r["pooled"] = _pooled_with_hurdle()
    md = report.build_markdown(r)
    assert "## (c) Fee hurdle" in md
    # the fee-hurdle table is the one that broke: `mean |pred| (bps)` written
    # with bare pipes made its header two cells wider than its own separator
    assert r"mean \|pred\| (bps)" in md
    assert_tables_well_formed(md, expect_at_least=5)


def test_the_header_names_the_command_that_actually_ran():
    """`analyze.py day` and `make smoke` print reports too, and must not
    credit `make results` for files `make results` cannot produce."""
    r = _results([_day("coinbase/BTC-USD/2001-01-01", ["synthetic"])])
    assert "by `make results` from" in report.build_markdown(r)
    r["command"] = "python analyze.py day --symbol BTC-USD --day 2001-01-01"
    md = report.build_markdown(r)
    assert "by `python analyze.py day --symbol BTC-USD --day 2001-01-01` from" in md
    assert "make results" not in md.splitlines()[2]


def test_the_smoke_run_credits_make_smoke_not_make_results():
    """`make smoke` and `make results` both call `analyze.py results`, so the
    subcommand cannot name the target -- the manifest has to. Crediting
    `make results` at the top of results/self-test/README.md named a command
    that reads a different manifest and writes a different directory."""
    import analyze

    class A:
        manifest = "tests/fixtures/synthetic/manifest.json"
    assert analyze.reproducing_command(A) == "make smoke"
    A.manifest = "manifest.json"
    assert analyze.reproducing_command(A) == "make results"
    A.manifest = str(analyze.ROOT / "manifest.json")      # the argparse default
    assert analyze.reproducing_command(A) == "make results"
    A.manifest = "some/other/manifest.json"
    assert analyze.reproducing_command(A) == \
        "python analyze.py results --manifest some/other/manifest.json"


def test_a_generated_report_never_carries_an_absolute_path():
    """`--manifest` defaults to an absolute path; printing it verbatim wrote
    the operator's home directory into a committed artifact."""
    import analyze
    from ofi.run import rel_to_root
    assert rel_to_root(analyze.ROOT / "manifest.json", analyze.ROOT) == \
        "manifest.json"
    assert rel_to_root(analyze.ROOT / "tests" / "fixtures" / "synthetic"
                       / "manifest.json", analyze.ROOT) == \
        "tests/fixtures/synthetic/manifest.json"
    # a manifest outside the repo is left alone rather than mangled
    assert rel_to_root("/elsewhere/m.json", analyze.ROOT).endswith("m.json")


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
