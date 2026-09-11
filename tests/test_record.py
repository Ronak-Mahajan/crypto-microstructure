"""Offline tests for record.py: line encoding, markers, the writer thread."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import record  # noqa: E402  (needs `websockets` importable; see requirements)


def test_iso_to_ns():
    assert record.iso_to_ns("2019-08-14T20:42:27.265Z") == 1565815347_265000000
    assert record.iso_to_ns("2014-11-07T08:19:28.464459Z") == \
        1415348368_464459000
    assert record.iso_to_ns("2014-11-07T08:19:28Z") == 1415348368_000000000
    assert record.iso_to_ns("2014-11-07T08:19:28") is None
    assert record.iso_to_ns("") is None
    assert record.iso_to_ns("nope-Z") is None


def test_venue_time():
    assert record.venue_time_ns({"type": "l2update",
                                 "time": "2019-08-14T20:42:27.265Z"}) == \
        1565815347_265000000
    assert record.venue_time_ns({"type": "snapshot", "bids": []}) is None
    assert record.venue_time_ns({"channel": "l2Book",
                                 "data": {"coin": "BTC", "time": 1700000000123}}) \
        == 1700000000123_000000
    assert record.venue_time_ns({"channel": "trades",
                                 "data": [{"time": 5, "px": "1"}]}) == 5_000000
    assert record.venue_time_ns({"channel": "pong"}) is None
    assert record.venue_time_ns({"channel": "trades", "data": []}) is None


def test_encode_line_verbatim_and_vt():
    raw = '{"type":"l2update","product_id":"BTC-USD","time":"2019-08-14T20:42:27.265Z","changes":[["buy","1","2"]]}'
    line, obj = record.encode_line(7, raw)
    assert line == '{"t_ns":7,"vt":1565815347265000000,"m":' + raw + '}\n'
    assert obj["product_id"] == "BTC-USD"
    d = json.loads(line)
    assert d["m"]["changes"] == [["buy", "1", "2"]]

    raw2 = '{"type":"snapshot","product_id":"ETH-USD","bids":[],"asks":[]}'
    line2, _ = record.encode_line(9, raw2)
    assert line2 == '{"t_ns":9,"m":' + raw2 + '}\n'


def test_encode_line_rejects_bad_frames():
    assert record.encode_line(1, "not json") == (None, None)
    assert record.encode_line(1, "[1,2,3]") == (None, None)
    assert record.encode_line(1, "") == (None, None)
    # a raw newline inside otherwise valid JSON is re-serialised compactly
    line, _ = record.encode_line(3, '{"a":\n1}')
    assert line == '{"t_ns":3,"m":{"a":1}}\n'
    assert "\n" not in line[:-1]


def test_marker_line_is_readable_and_skippable():
    line = record.marker_line(5, "disconnect", "coinbase", "ConnectionClosed")
    d = json.loads(line)
    assert d == {"t_ns": 5, "marker": "disconnect", "venue": "coinbase",
                 "reason": "ConnectionClosed", "m": None}
    assert not isinstance(d["m"], dict)          # what analyze.py checks


def test_streams_one_dir_per_product():
    streams = record.streams_for(["BTC", "ETH"])
    cb = streams[0]
    assert cb.venue == "coinbase" and cb.products == ["BTC-USD", "ETH-USD"]
    assert cb.route({"type": "l2update", "product_id": "ETH-USD"}) == "ETH-USD"
    assert cb.route({"type": "subscriptions"}) == "_control"
    assert cb.keepalive is None
    hl = [s for s in streams if s.venue == "hyperliquid"]
    assert [s.products[0] for s in hl] == ["BTC-l2Book", "BTC-trades",
                                           "ETH-l2Book", "ETH-trades"]
    assert all(s.keepalive == '{"method":"ping"}' for s in hl)
    assert hl[0].route({"channel": "l2Book"}) == "BTC-l2Book"
    assert record.MAX_FRAME_BYTES == 2 ** 26


def test_writer_thread_roundtrip(tmp_path):
    w = record.GzWriter(tmp_path, compresslevel=1)
    w.start()
    t0 = 1754006400_000000000                     # 2025-08-01T00:00:00Z
    t1 = t0 + 3600 * 10 ** 9
    w.put("coinbase", "BTC-USD", t0, record.marker_line(t0, "connect", "coinbase", None))
    for i in range(500):
        line, _ = record.encode_line(t0 + i, '{"type":"heartbeat","product_id":"BTC-USD"}')
        w.put("coinbase", "BTC-USD", t0 + i, line)
    w.put("coinbase", "ETH-USD", t0, '{"t_ns":%d,"m":{"x":1}}\n' % t0)
    w.put("coinbase", "BTC-USD", t1, '{"t_ns":%d,"m":{"x":2}}\n' % t1)
    w.flush()
    w.stop()
    assert not w.is_alive()
    assert w.written == 503 and w.dropped == 0

    btc = tmp_path / "coinbase" / "BTC-USD"
    assert sorted(p.name for p in btc.glob("*.jsonl.gz")) == \
        ["20250801-00.jsonl.gz", "20250801-01.jsonl.gz"]
    with gzip.open(btc / "20250801-00.jsonl.gz", "rt") as fh:
        rows = [json.loads(l) for l in fh]
    assert len(rows) == 501
    assert rows[0]["marker"] == "connect"
    assert rows[1]["m"]["type"] == "heartbeat"
    with gzip.open(tmp_path / "coinbase" / "ETH-USD" / "20250801-00.jsonl.gz", "rt") as fh:
        assert json.loads(fh.readline())["m"] == {"x": 1}
