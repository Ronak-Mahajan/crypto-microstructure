"""End-to-end: a hand-written Coinbase message sequence through rebuild().

Every bar value asserted here is derived by hand in the test docstring from
the messages above it, so this is a check on the composition of book +
features + bars, not a golden-file snapshot.
"""
from __future__ import annotations

import numpy as np
import pytest

from ofi.io import iter_file, write_lines
from ofi.pipeline import rebuild

S = 1_000_000_000


def msg(t_ns: int, m: dict) -> tuple:
    return ("msg", t_ns, None, m)


def marker(t_ns: int, kind: str) -> tuple:
    return ("marker", t_ns, kind, {"venue": "test"})


def snapshot(bids, asks) -> dict:
    return {"type": "snapshot", "product_id": "BTC-USD",
            "bids": bids, "asks": asks}


def l2(changes) -> dict:
    return {"type": "l2update", "product_id": "BTC-USD", "changes": changes}


def match(side: str, size: str) -> dict:
    return {"type": "match", "product_id": "BTC-USD", "side": side,
            "size": size, "price": "100.0"}


BASE_SNAP = snapshot([["99.0", "5"], ["98.0", "9"]], [["101.0", "4"], ["102.0", "7"]])


def test_clean_replay_has_no_snapshot_mismatch_and_the_hand_computed_ofi():
    """Snapshot: bid 99/5, ask 101/4. Then

      t=0.10 s  buy 99.0 -> 8      e = 8 - 5          = +3
      t=0.20 s  sell 101.0 -> 1    e = -1 + 4         = +3
      t=0.30 s  buy 99.5 -> 2      e = +2 (price up)  = +2
      t=0.40 s  buy 99.5 -> 0      e = -2 (price down back to 99.0/8) = -2
                                   (the new best 99.0/8 is not added: only
                                   1{<=} fires because the price fell)
    Bar [0, 1) therefore closes with OFI = 3 + 3 + 2 - 2 = +6, mid = 100.0.

    The same snapshot is then resent at t = 2 s with the book's own state,
    so the consistency check must be exactly 0% mismatched with no gap.
    """
    final_snap = snapshot([["99.0", "8"], ["98.0", "9"]],
                          [["101.0", "1"], ["102.0", "7"]])
    events = [
        msg(0, BASE_SNAP),
        msg(S // 10, l2([["buy", "99.0", "8"]])),
        msg(2 * S // 10, l2([["sell", "101.0", "1"]])),
        msg(3 * S // 10, l2([["buy", "99.5", "2"]])),
        msg(4 * S // 10, l2([["buy", "99.5", "0"]])),
        msg(2 * S, final_snap),
    ]
    bars, stats = rebuild(events, bar_s=1.0, max_gap_s=5.0, product="BTC-USD")
    assert stats["n_snapshots"] == 2
    assert stats["n_l2updates"] == 4
    assert stats["n_changes"] == 4
    assert stats["n_disconnects"] == 0
    assert stats["n_skipped_updates"] == 0
    assert stats["n_crossed"] == 0
    assert stats["n_checks"] == 1
    assert stats["n_checks_clean"] == 1
    assert stats["mismatch_rate_clean_max"] == 0.0
    assert bars["ofi"][0] == pytest.approx(6.0)
    assert bars["mid"][0] == pytest.approx(100.0)
    assert bars["n_events"][0] == 4


def test_a_dropped_change_shows_up_as_a_snapshot_mismatch():
    """Deliberately corrupt the replay by never sending one diff; the resent
    snapshot then disagrees with the rebuilt book on exactly one price, so
    the check reports 1 mismatch out of a 4-price union = 25%."""
    truth = snapshot([["99.0", "8"], ["98.0", "9"]],
                     [["101.0", "4"], ["102.0", "7"]])
    events = [msg(0, BASE_SNAP), msg(2 * S, truth)]   # the buy 99 -> 8 is lost
    _, stats = rebuild(events, product="BTC-USD")
    chk = stats["snapshot_checks"][0]
    assert chk["mismatched"] == 1
    assert chk["rate"] == pytest.approx(0.25)
    assert chk["after_gap"] is False
    assert stats["mismatch_rate_clean_max"] == pytest.approx(0.25)


def test_diffs_before_any_snapshot_are_counted_and_dropped():
    events = [msg(0, l2([["buy", "99.0", "1"]])), msg(S, BASE_SNAP)]
    bars, stats = rebuild(events, product="BTC-USD")
    assert stats["n_skipped_updates"] == 1
    assert stats["n_l2updates"] == 0
    assert stats["n_checks"] == 0          # nothing to compare the first one to


def test_disconnect_breaks_the_segment_and_stales_the_book():
    """A disconnect marker must (1) stop diffs being applied until the next
    snapshot, (2) end the bar run, (3) tag the reconnect snapshot check as
    after_gap so its mismatch is not read as a replay bug."""
    reconnect = snapshot([["95.0", "1"]], [["96.0", "1"]])
    events = [
        msg(0, BASE_SNAP),
        msg(S // 2, l2([["buy", "99.0", "8"]])),
        marker(S, "disconnect"),
        msg(2 * S, l2([["buy", "99.0", "50"]])),        # stale: must be dropped
        msg(3 * S, reconnect),
        msg(3 * S + S // 2, l2([["buy", "95.0", "3"]])),
    ]
    bars, stats = rebuild(events, bar_s=1.0, max_gap_s=60.0, product="BTC-USD")
    assert stats["n_disconnects"] == 1
    assert stats["n_skipped_updates"] == 1
    assert stats["n_checks"] == 1
    assert stats["snapshot_checks"][0]["after_gap"] is True
    assert stats["n_checks_clean"] == 0
    assert stats["segments"] >= 2
    assert bars["is_gap"].sum() >= 1
    # the reconnect really replaced the book rather than merging into it
    assert bars["mid"][-1] == pytest.approx(95.5)


def test_matches_add_signed_flow_without_being_book_events():
    events = [
        msg(0, BASE_SNAP),
        msg(S // 4, match("sell", "2")),    # maker sold -> taker bought -> +2
        msg(S // 2, match("buy", "0.5")),   # maker bought -> taker sold -> -0.5
        msg(3 * S // 4, l2([["buy", "99.0", "6"]])),
    ]
    bars, stats = rebuild(events, product="BTC-USD")
    assert stats["n_matches"] == 2
    assert bars["tflow"][0] == pytest.approx(1.5)
    assert bars["n_events"][0] == 1         # only the l2update counts


def test_long_silence_does_not_fabricate_bars():
    events = [
        msg(0, BASE_SNAP),
        msg(S // 2, l2([["buy", "99.0", "8"]])),
        msg(600 * S, l2([["buy", "99.0", "9"]])),
    ]
    bars, stats = rebuild(events, bar_s=1.0, max_gap_s=5.0, product="BTC-USD")
    assert stats["bars"] <= 3               # not 601
    assert stats["covered_s"] <= 3.0
    assert stats["span_s"] == pytest.approx(600.0)
    assert bars["is_gap"].sum() == 1


def test_other_products_are_filtered_out():
    other = {"type": "l2update", "product_id": "ETH-USD",
             "changes": [["buy", "1.0", "1"]]}
    events = [msg(0, BASE_SNAP), msg(S // 2, other),
              msg(3 * S // 4, l2([["buy", "99.0", "8"]]))]
    bars, stats = rebuild(events, product="BTC-USD")
    assert stats["n_msgs"] == 2
    assert bars["ofi"][0] == pytest.approx(3.0)


def test_crossed_book_is_counted_not_hidden():
    events = [msg(0, BASE_SNAP), msg(S // 2, l2([["buy", "105.0", "1"]]))]
    _, stats = rebuild(events, product="BTC-USD")
    assert stats["n_crossed"] == 1


def test_empty_stream_is_handled():
    bars, stats = rebuild([], product="BTC-USD")
    assert stats["bars"] == 0
    assert stats["segments"] == 0
    assert stats["mismatch_rate_mean"] is None
    assert len(bars["t_end"]) == 0


def test_round_trip_through_the_on_disk_format(tmp_path):
    """The same sequence written as gzip NDJSON and read back by ofi.io must
    produce identical bars, including the marker line."""
    import json
    events = [
        msg(0, BASE_SNAP),
        msg(S // 2, l2([["buy", "99.0", "8"]])),
        marker(S, "disconnect"),
        msg(2 * S, snapshot([["99.0", "8"]], [["101.0", "4"]])),
        msg(2 * S + S // 2, l2([["sell", "101.0", "2"]])),
    ]
    lines = []
    for ev in events:
        if ev[0] == "marker":
            lines.append(json.dumps({"t_ns": ev[1], "marker": ev[2],
                                     "venue": "test", "m": None}))
        else:
            lines.append(json.dumps({"t_ns": ev[1], "m": ev[3]}))
    p = tmp_path / "20010101-00.jsonl.gz"
    assert write_lines(p, lines) == len(lines)
    from_disk, stats_disk = rebuild(iter_file(p), max_gap_s=60.0, product="BTC-USD")
    in_mem, stats_mem = rebuild(events, max_gap_s=60.0, product="BTC-USD")
    for k in from_disk:
        assert np.array_equal(from_disk[k], in_mem[k], equal_nan=True)
    assert stats_disk["n_disconnects"] == stats_mem["n_disconnects"] == 1


def test_torn_and_malformed_lines_are_skipped(tmp_path):
    import json
    good = json.dumps({"t_ns": 0, "m": BASE_SNAP})
    p = tmp_path / "20010101-01.jsonl.gz"
    write_lines(p, [good, "{not json", json.dumps({"m": {}}),
                    json.dumps({"t_ns": "x", "m": {}}),
                    json.dumps({"t_ns": 1, "m": None}), ""])
    evs = list(iter_file(p))
    assert len(evs) == 1 and evs[0][0] == "msg"
