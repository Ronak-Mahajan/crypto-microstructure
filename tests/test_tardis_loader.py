"""Offline tests for tardis_loader.py: parsing, writing, manifest, verify."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tardis_loader as tl  # noqa: E402


def test_ts_to_ns_forms():
    assert tl.tardis_ts_to_ns("2019-04-01T00:00:34.8345516Z") == \
        1554076834_834551600
    assert tl.tardis_ts_to_ns("2019-04-01T00:00:34Z") == 1554076834_000000000
    assert tl.tardis_ts_to_ns("2019-04-01T00:00:34.5Z") == 1554076834_500000000
    assert tl.tardis_ts_to_ns("2019-04-01T00:00:34.123456789Z") == \
        1554076834_123456789
    for bad in ("2019-04-01 00:00:34Z", "2019-04-01T00:00:34", "",
                "2019-04-01T00:00:34.12x3Z", "2019-04-01T00:00:34_5Z"):
        with pytest.raises(ValueError):
            tl.tardis_ts_to_ns(bad)


def test_split_line_and_type():
    t, msg = tl.split_line(
        '2025-08-01T00:00:00.0000000Z {"type":"l2update","product_id":"BTC-USD"}')
    assert t == 1754006400_000000000
    assert msg == '{"type":"l2update","product_id":"BTC-USD"}'
    assert tl.message_type(msg) == "l2update"
    assert tl.message_type('{"foo":1}') == "?"
    assert tl.split_line("garbage") is None
    assert tl.split_line("2025-08-01T00:00:00Z notjson") is None
    assert tl.split_line("") is None


def test_hour_name():
    assert tl.hour_name(1754006400_000000000) == "20250801-00"
    assert tl.hour_name(1754006400_000000000 + 3600 * 10 ** 9) == "20250801-01"


def _fake_day_lines(day: str):
    """A tiny synthetic day: snapshot at 00:00, updates across hours 0-2."""
    base = tl.tardis_ts_to_ns(f"{day}T00:00:00Z")
    out = []

    def add(offset_s, msg):
        t = base + int(offset_s * 1e9)
        out.append(f"{day}T{offset_s // 3600:02.0f}:{(offset_s % 3600) // 60:02.0f}"
                   f":{offset_s % 60:02.0f}.0000000Z {msg}")

    add(0.0, '{"type":"snapshot","product_id":"BTC-USD","bids":[["1","1"]],'
             '"asks":[["2","1"]]}')
    add(1.0, '{"type":"l2update","product_id":"BTC-USD","time":"x",'
             '"changes":[["buy","1","2"]]}')
    add(2.0, '{"type":"match","product_id":"BTC-USD","price":"1.5","size":"1"}')
    out.append("")                                    # Tardis disconnect marker
    add(3600.0, '{"type":"l2update","product_id":"BTC-USD","changes":[]}')
    add(7200.0, '{"type":"l2update","product_id":"BTC-USD","changes":[]}')
    return out


def test_fetch_day_writes_recorder_format(tmp_path):
    day = "2025-08-01"
    lines = _fake_day_lines(day)
    calls = []

    def fake_fetch(exchange, day_, offset, size, filters, api_key):
        calls.append((exchange, day_, offset, size, tuple(f["channel"] for f in filters)))
        # everything in the first slice; later slices empty (as a quiet minute is)
        return (lines if offset == 0 else []), 100

    stats = tl.fetch_day("coinbase", "BTC-USD", day, tl.DEFAULT_CHANNELS,
                         tmp_path, 10, 30, None, False, 1,
                         fetch=fake_fetch, log=lambda *a, **k: None)
    assert [c[2] for c in calls] == [0, 10, 20]
    assert calls[0][4] == ("snapshot", "l2update", "match")
    assert stats["lines"] == 5 and stats["bad_lines"] == 0
    assert stats["disconnects"] == 1
    assert stats["messages"] == {"l2update": 3, "match": 1, "snapshot": 1}
    assert stats["first_snapshot_t_ns"] == stats["first_t_ns"]
    assert stats["partial"] is True and stats["minutes"] == 30

    out_dir = tmp_path / "coinbase" / "BTC-USD"
    names = sorted(p.name for p in out_dir.glob("*.jsonl.gz"))
    assert names == ["20250801-00.jsonl.gz", "20250801-01.jsonl.gz",
                     "20250801-02.jsonl.gz"]
    assert not list(out_dir.glob(".tardis-*"))       # staging dir removed

    with gzip.open(out_dir / "20250801-00.jsonl.gz", "rt") as fh:
        rows = [json.loads(l) for l in fh]
    assert len(rows) == 4
    assert set(rows[0]) == {"t_ns", "m"}
    assert rows[0]["m"]["type"] == "snapshot"
    assert rows[1]["t_ns"] - rows[0]["t_ns"] == 10 ** 9
    assert rows[2]["m"]["price"] == "1.5"
    assert rows[3]["marker"] == "disconnect" and rows[3]["m"] is None
    # LF, not os.linesep: pulling the same Tardis day on two operating
    # systems must give the same bytes, or the manifest sha256 cannot say
    # which day a published number was computed from
    raw = gzip.open(out_dir / "20250801-00.jsonl.gz", "rb").read()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 4
    assert rows[3]["t_ns"] == rows[2]["t_ns"]           # stamped with last seen
    assert rows[3]["venue"] == "tardis:coinbase"

    # refuses to clobber without --force
    with pytest.raises(SystemExit):
        tl.fetch_day("coinbase", "BTC-USD", day, tl.DEFAULT_CHANNELS,
                     tmp_path, 10, 30, None, False, 1,
                     fetch=fake_fetch, log=lambda *a, **k: None)


def test_manifest_roundtrip_and_verify(tmp_path, monkeypatch):
    day = "2025-08-01"
    lines = _fake_day_lines(day)
    monkeypatch.setattr(tl, "ROOT", tmp_path)
    stats = tl.fetch_day("coinbase", "BTC-USD", day, tl.DEFAULT_CHANNELS,
                         tmp_path / "data", 10, 10, None, False, 1,
                         fetch=lambda *a: (lines, 1),
                         log=lambda *a, **k: None)
    mpath = tmp_path / "manifest.json"
    manifest = tl.load_manifest(mpath)
    tl.record_day(manifest, stats)
    tl.save_manifest(mpath, manifest)

    m = json.loads(mpath.read_text())
    assert m["schema"] == 1
    key = "data/coinbase/BTC-USD/20250801-00.jsonl.gz"
    assert key in m["files"]
    entry = m["files"][key]
    assert entry["source"] == "tardis" and entry["day"] == day
    assert entry["lines"] == 4 and entry["hour"] == 0
    assert len(entry["sha256"]) == 64
    assert entry["sha256"] == tl.sha256_of(tmp_path / key)
    dk = "coinbase/BTC-USD/2025-08-01"
    assert m["days"][dk]["files"] == sorted(m["days"][dk]["files"])
    assert m["days"][dk]["bytes"] == sum(m["files"][k]["bytes"]
                                         for k in m["days"][dk]["files"])

    class A:
        manifest = str(mpath)
        strict = False
        quiet = True
    assert tl.cmd_verify(A()) == 0

    # corrupt one file -> mismatch -> exit 1
    with open(tmp_path / key, "ab") as fh:
        fh.write(b"\x00")
    assert tl.cmd_verify(A()) == 1

    # missing file only fails under --strict
    (tmp_path / key).unlink()
    assert tl.cmd_verify(A()) == 0
    A.strict = True
    assert tl.cmd_verify(A()) == 1


def test_verify_resolves_keys_against_the_manifests_own_directory(tmp_path,
                                                                  monkeypatch):
    """A manifest that does not sit at the repo root must still verify.

    tests/fixtures/synthetic/manifest.json is exactly that case: its keys
    are `data/...` relative to itself. ofi.run.resolve_path already tried
    the manifest's directory first, so the analysis read the fixture fine,
    while `verify` resolved against the repo root only and called all three
    files MISSING -- the one command whose job is to tie a number to exact
    bytes, unable to check the only manifest that lists any. Every existing
    verify test put the manifest at the root, which is why nothing caught it.
    """
    monkeypatch.setattr(tl, "ROOT", tmp_path)
    sub = tmp_path / "fixtures" / "synthetic"
    d = sub / "data" / "coinbase" / "BTC-USD"
    d.mkdir(parents=True)
    p = d / "20250801-00.jsonl.gz"
    with gzip.open(p, "wt", newline="\n") as fh:
        fh.write('{"t_ns":1,"m":{"type":"snapshot"}}\n')
    key = "data/coinbase/BTC-USD/20250801-00.jsonl.gz"
    mpath = sub / "manifest.json"
    mpath.write_text(json.dumps({
        "schema": 1, "days": {},
        "files": {key: {"sha256": tl.sha256_of(p), "bytes": p.stat().st_size,
                        "source": "synthetic", "exchange": "coinbase",
                        "symbol": "BTC-USD", "day": "2025-08-01"}}}),
        encoding="utf-8")

    class A:
        manifest = str(mpath)
        strict = True                 # missing must be a failure here
        quiet = True
    assert tl.cmd_verify(A()) == 0, "the file is right there next to the manifest"
    assert tl.resolve_key(key, sub) == p
    # and a genuinely absent file is still reported, not silently resolved
    assert tl.resolve_key("data/coinbase/BTC-USD/29991231-23.jsonl.gz", sub) is None


def test_hash_adds_recorder_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "ROOT", tmp_path)
    d = tmp_path / "data" / "coinbase" / "ETH-USD"
    d.mkdir(parents=True)
    with gzip.open(d / "20250901-13.jsonl.gz", "wt") as fh:
        fh.write('{"t_ns":1,"m":{"type":"heartbeat"}}\n')
        fh.write('{"t_ns":2,"marker":"connect","venue":"coinbase","reason":null,"m":null}\n')

    class A:
        out = str(tmp_path / "data")
        manifest = str(tmp_path / "manifest.json")
        all = False
        no_lines = False
    assert tl.cmd_hash(A()) == 0
    m = json.loads((tmp_path / "manifest.json").read_text())
    e = m["files"]["data/coinbase/ETH-USD/20250901-13.jsonl.gz"]
    assert e["source"] == "recorder" and e["day"] == "2025-09-01"
    assert e["hour"] == 13 and e["lines"] == 2 and e["symbol"] == "ETH-USD"


def test_cli_refuses_non_free_day_without_key(monkeypatch):
    monkeypatch.delenv("TARDIS_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        tl.main(["fetch", "--day", "2025-08-02"])
    with pytest.raises(SystemExit):
        tl.main(["fetch", "--day", "not-a-day"])
