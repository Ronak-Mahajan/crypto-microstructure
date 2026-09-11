"""Pull Tardis.dev's free first-of-month raw Coinbase feed into the recorder's format.

Tardis.dev replays the raw websocket messages it recorded from each exchange
through one HTTP endpoint, and the first day of every month is served without
an API key (docs.tardis.dev, HTTP API reference, /data-feeds: "Without API key
historical data feeds for the first day of each month are available"). That
gives this project independent full days of Coinbase BTC-USD level-2 data on
day one, before the recorder in record.py has run for a week.

What this script does, and nothing more:

    GET https://api.tardis.dev/v1/data-feeds/coinbase
        ?from=YYYY-MM-01&offset=<minute>&sliceSize=10
        &filters=[{"channel":"snapshot","symbols":["BTC-USD"]},
                  {"channel":"l2update","symbols":["BTC-USD"]},
                  {"channel":"match","symbols":["BTC-USD"]}]

144 requests of ten minutes each cover one UTC day. Each response line is

    2025-08-01T00:00:34.8345516Z {"type":"l2update","product_id":"BTC-USD",...}

i.e. Tardis's local arrival timestamp (100 ns resolution) followed by the
exchange's native message. The loader writes every line as

    {"t_ns": <Tardis localTimestamp in ns>, "m": <verbatim message>}

into data/coinbase/BTC-USD/<YYYYMMDD-HH>.jsonl.gz, exactly the layout and
line format record.py produces, so analyze.py reads both without knowing
which source a day came from. The message's own "time" field stays inside
"m". Tardis re-subscribes at 00:00 UTC, so each day begins with a
"snapshot" message followed by incremental "l2update"s; further snapshots
appear only after Tardis's own reconnects, which the feed marks with an
empty line. The loader turns each empty line into a marker line of the same
shape record.py writes ({"t_ns", "marker": "disconnect", "venue":
"tardis:coinbase", "reason", "m": null}), stamped with the last arrival time
seen, so the gap is in the record and not only in a counter.

Sizes, honestly: Coinbase publishes level2 unbatched, and a BTC-USD day is
several million l2update lines. Expect on the order of 1-3 GB of gzipped
output per day and a similar download volume (the endpoint gzips its
responses). The loader streams slice by slice and never holds more than one
ten-minute slice in memory, but disk is yours to provide. `--minutes 10`
fetches a single slice as a pipeline smoke test.

Every file written is recorded in manifest.json with its sha256, byte and
line counts, source ("tardis"), exchange, symbol and day, plus per-day
message counts by channel and the arrival time of the first snapshot. The
analysis regenerates its tables from that manifest; `verify` recomputes the
hashes so a result can be tied to the exact bytes that produced it.

    python tardis_loader.py fetch --day 2025-08-01            # one free day
    python tardis_loader.py fetch --day 2025-08-01 --minutes 10  # smoke test
    python tardis_loader.py hash                              # add recorder files
    python tardis_loader.py verify                            # check hashes

Standard library only. With TARDIS_API_KEY set (or --api-key) any day works;
without it the loader refuses days other than the 1st rather than letting
the server return an authorization error 144 times.
"""
from __future__ import annotations

import argparse
import calendar
import gzip
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFEST = ROOT / "manifest.json"
API = "https://api.tardis.dev/v1/data-feeds"
DEFAULT_CHANNELS = ("snapshot", "l2update", "match")
MINUTES_PER_DAY = 1440
USER_AGENT = "crypto-microstructure/tardis_loader (stdlib urllib)"

_TYPE_RE = re.compile(r'"type"\s*:\s*"([A-Za-z0-9_]+)"')


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TS_CACHE: dict[str, int] = {}


def tardis_ts_to_ns(ts: str) -> int:
    """'2019-04-01T00:00:34.8345516Z' -> epoch ns.

    The second-resolution head is memoised because consecutive lines share
    it; strptime is the slow part and a day has millions of lines.
    """
    if len(ts) < 20 or ts[-1] != "Z" or ts[10] != "T":
        raise ValueError(f"bad Tardis timestamp {ts!r}")
    head = ts[:19]
    secs = _TS_CACHE.get(head)
    if secs is None:
        secs = calendar.timegm(time.strptime(head, "%Y-%m-%dT%H:%M:%S"))
        if len(_TS_CACHE) > 4096:
            _TS_CACHE.clear()
        _TS_CACHE[head] = secs
    rest = ts[19:-1]
    frac = 0
    if rest:
        if rest[0] != ".":
            raise ValueError(f"bad Tardis timestamp {ts!r}")
        digits = rest[1:]
        if not digits.isdigit():
            raise ValueError(f"bad Tardis timestamp {ts!r}")
        frac = int((digits + "000000000")[:9])
    return secs * 1_000_000_000 + frac


def split_line(line: str) -> tuple[int, str] | None:
    """One feed line -> (t_ns, message_json) or None if it is not one."""
    ts, sep, msg = line.partition(" ")
    if not sep:
        return None
    msg = msg.strip()
    if not (msg.startswith("{") and msg.endswith("}")):
        return None
    try:
        return tardis_ts_to_ns(ts), msg
    except ValueError:
        return None


def message_type(msg: str) -> str:
    m = _TYPE_RE.search(msg)
    return m.group(1) if m else "?"


def hour_name(t_ns: int) -> str:
    return time.strftime("%Y%m%d-%H", time.gmtime(t_ns // 1_000_000_000))


def marker_line(t_ns: int, marker: str, venue: str, reason: str | None) -> str:
    """Same shape as record.py's marker lines, so one reader handles both."""
    return json.dumps({"t_ns": t_ns, "marker": marker, "venue": venue,
                       "reason": reason, "m": None},
                      separators=(",", ":")) + "\n"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _decompress(body: bytes, encoding: str) -> bytes:
    enc = (encoding or "").lower()
    if enc in ("", "identity"):
        return body
    if enc == "gzip":
        return gzip.decompress(body)
    if enc == "zstd":
        try:                                     # Python 3.14+
            from compression import zstd         # type: ignore
            return zstd.decompress(body)
        except ImportError:
            pass
        try:                                     # pip install zstandard
            import zstandard                     # type: ignore
            return zstandard.ZstdDecompressor().decompressobj().decompress(body)
        except ImportError:
            raise RuntimeError("server sent zstd; use Python 3.14+ or "
                               "pip install zstandard")
    raise RuntimeError(f"unexpected Content-Encoding {encoding!r}")


def fetch_slice(exchange: str, day: str, offset: int, slice_size: int,
                filters: list[dict], api_key: str | None,
                timeout: float = 120.0, retries: int = 6) -> tuple[list[str], int]:
    """Fetch one slice; returns (lines, compressed_bytes). Retries 429/5xx."""
    qs = urllib.parse.urlencode({
        "from": day, "offset": offset, "sliceSize": slice_size,
        "filters": json.dumps(filters, separators=(",", ":")),
    })
    url = f"{API}/{exchange}?{qs}"
    headers = {"Accept-Encoding": "gzip", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                enc = resp.headers.get("Content-Encoding", "")
            text = _decompress(raw, enc).decode("utf-8", "replace")
            lines = text.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            return lines, len(raw)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SystemExit(
                    f"HTTP {exc.code} from Tardis for {day} offset {offset}: "
                    f"only the first day of each month is free without an "
                    f"API key (set TARDIS_API_KEY for other days).")
            if exc.code == 404:
                raise SystemExit(f"HTTP 404: no data for {exchange} {day} "
                                 f"offset {offset} with filters {filters}")
            if attempt >= retries:
                raise
            print(f"  HTTP {exc.code} at offset {offset}; retry in {delay:.0f}s",
                  file=sys.stderr, flush=True)
        except (urllib.error.URLError, http.client.HTTPException,
                TimeoutError, OSError, EOFError) as exc:
            if attempt >= retries:
                raise
            print(f"  {exc!r} at offset {offset}; retry in {delay:.0f}s",
                  file=sys.stderr, flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class HourFiles:
    """Hourly gz files inside one directory, appended as lines arrive."""

    def __init__(self, directory: Path, compresslevel: int) -> None:
        self.dir = directory
        self.level = compresslevel
        self._hour: str | None = None
        self._fh = None
        self.lines: dict[str, int] = {}

    def write(self, hour: str, text: str) -> None:
        if hour != self._hour:
            if self._fh is not None:
                self._fh.close()
            self._fh = gzip.open(self.dir / f"{hour}.jsonl.gz", "at",
                                 encoding="utf-8", compresslevel=self.level)
            self._hour = hour
        self._fh.write(text)
        self.lines[hour] = self.lines.get(hour, 0) + 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._hour = None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines_gz(path: Path) -> int:
    n = 0
    try:
        with gzip.open(path, "rb") as fh:
            for _ in fh:
                n += 1
    except EOFError:
        pass                     # torn tail from a live recorder; count what read
    return n


def rel_key(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema": 1, "files": {}, "days": {}}
    with open(path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    m.setdefault("schema", 1)
    m.setdefault("files", {})
    m.setdefault("days", {})
    return m


def save_manifest(path: Path, m: dict) -> None:
    m["updated_at"] = now_iso()
    m["files"] = dict(sorted(m["files"].items()))
    m["days"] = dict(sorted(m["days"].items()))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1, sort_keys=False)
        fh.write("\n")
    tmp.replace(path)


def file_entry(path: Path, source: str, exchange: str, symbol: str,
               day: str, lines: int | None) -> dict:
    st = path.stat()
    entry = {
        "sha256": sha256_of(path), "bytes": st.st_size,
        "source": source, "exchange": exchange, "symbol": symbol, "day": day,
        "hour": int(path.name[9:11]) if len(path.name) > 11 else None,
    }
    if lines is not None:
        entry["lines"] = lines
    return entry


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def fetch_day(exchange: str, symbol: str, day: str, channels: tuple,
              out_root: Path, slice_size: int, minutes: int,
              api_key: str | None, force: bool, compresslevel: int,
              fetch=fetch_slice, log=print) -> dict:
    """Download one day into out_root/<exchange>/<symbol>/ and return stats."""
    out_dir = out_root / exchange / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    ymd = day.replace("-", "")
    existing = sorted(out_dir.glob(f"{ymd}-*.jsonl.gz"))
    if existing and not force:
        raise SystemExit(f"{len(existing)} files for {day} already exist in "
                         f"{out_dir}; pass --force to overwrite them")

    stage = out_dir / f".tardis-{ymd}.part"
    stage.mkdir(exist_ok=True)
    for stale in stage.glob("*.jsonl.gz"):
        stale.unlink()                        # leftovers of an interrupted run

    filters = [{"channel": c, "symbols": [symbol]} for c in channels]
    writer = HourFiles(stage, compresslevel)
    counts: dict[str, int] = {}
    n_lines = n_bad = n_ooo = n_disc = 0
    first_ns = last_ns = first_snapshot_ns = None
    dl_bytes = 0
    t_start = time.time()
    minutes = max(1, min(minutes, MINUTES_PER_DAY))
    try:
        for offset in range(0, minutes, slice_size):
            size = min(slice_size, minutes - offset)
            lines, nbytes = fetch(exchange, day, offset, size, filters, api_key)
            dl_bytes += nbytes
            for line in lines:
                if not line.strip():
                    # Tardis marks its own reconnects with an empty line; keep
                    # it in the record as a marker, stamped with the last
                    # arrival time seen (its true time is not given).
                    n_disc += 1
                    if last_ns is not None:
                        writer.write(hour_name(last_ns), marker_line(
                            last_ns, "disconnect", f"tardis:{exchange}",
                            "empty line in feed"))
                    continue
                parsed = split_line(line)
                if parsed is None:
                    n_bad += 1
                    continue
                t_ns, msg = parsed
                typ = message_type(msg)
                counts[typ] = counts.get(typ, 0) + 1
                if typ == "snapshot" and first_snapshot_ns is None:
                    first_snapshot_ns = t_ns
                if first_ns is None:
                    first_ns = t_ns
                if last_ns is not None and t_ns < last_ns:
                    n_ooo += 1
                last_ns = t_ns
                writer.write(hour_name(t_ns), '{"t_ns":%d,"m":%s}\n' % (t_ns, msg))
                n_lines += 1
            log(f"  {day} {offset + size:4d}/{minutes} min  "
                f"{n_lines:,} lines  {dl_bytes / 1e6:,.0f} MB down  "
                f"{time.time() - t_start:,.0f}s", flush=True)
    finally:
        writer.close()

    files = []
    for f in sorted(stage.glob("*.jsonl.gz")):
        target = out_dir / f.name
        f.replace(target)
        files.append((target, writer.lines.get(f.name[:-len(".jsonl.gz")], 0)))
    try:
        stage.rmdir()
    except OSError:
        pass

    return {
        "source": "tardis", "exchange": exchange, "symbol": symbol, "day": day,
        "channels": list(channels), "api": API, "slice_size": slice_size,
        "minutes": minutes, "partial": minutes < MINUTES_PER_DAY,
        "fetched_at": now_iso(), "download_bytes": dl_bytes,
        "lines": n_lines, "bad_lines": n_bad, "out_of_order_lines": n_ooo,
        "disconnects": n_disc,
        "messages": dict(sorted(counts.items())),
        "first_t_ns": first_ns, "last_t_ns": last_ns,
        "first_snapshot_t_ns": first_snapshot_ns,
        "files": files,
    }


def record_day(manifest: dict, stats: dict) -> None:
    files = stats.pop("files")
    keys = []
    total = 0
    for path, lines in files:
        key = rel_key(path)
        entry = file_entry(path, "tardis", stats["exchange"], stats["symbol"],
                           stats["day"], lines)
        manifest["files"][key] = entry
        total += entry["bytes"]
        keys.append(key)
    stats["bytes"] = total
    stats["files"] = keys
    day_key = f"{stats['exchange']}/{stats['symbol']}/{stats['day']}"
    manifest["days"][day_key] = stats


def cmd_fetch(args) -> int:
    api_key = args.api_key or os.environ.get("TARDIS_API_KEY") or None
    days = [d.strip() for d in args.day.split(",") if d.strip()]
    for d in days:
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            raise SystemExit(f"--day must be YYYY-MM-DD, got {d!r}")
        if not api_key and not d.endswith("-01"):
            raise SystemExit(f"{d} is not the first of a month; without an "
                             f"API key Tardis serves only YYYY-MM-01 days")
    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    out_root = Path(args.out)
    manifest_path = Path(args.manifest)
    for d in days:
        print(f"fetching {args.exchange} {args.symbol} {d} "
              f"channels={','.join(channels)} -> {out_root}", flush=True)
        stats = fetch_day(args.exchange, args.symbol, d, channels, out_root,
                          args.slice_size, args.minutes, api_key, args.force,
                          args.compresslevel)
        manifest = load_manifest(manifest_path)
        record_day(manifest, stats)
        save_manifest(manifest_path, manifest)
        print(f"done {d}: {stats['lines']:,} lines, "
              f"{stats['bytes'] / 1e6:,.0f} MB on disk in {len(stats['files'])} "
              f"files, messages={stats['messages']}, "
              f"first snapshot at t_ns={stats['first_snapshot_t_ns']}, "
              f"bad={stats['bad_lines']} out_of_order={stats['out_of_order_lines']}",
              flush=True)
        if stats["first_snapshot_t_ns"] is None:
            print("WARNING: no snapshot message in this day; the book cannot "
                  "be initialised from it", flush=True)
    return 0


# ---------------------------------------------------------------------------
# hash / verify
# ---------------------------------------------------------------------------

def cmd_hash(args) -> int:
    """Add every data/<venue>/<product>/*.jsonl.gz not yet in the manifest."""
    out_root = Path(args.out)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    added = 0
    for path in sorted(out_root.glob("*/*/*.jsonl.gz")):
        key = rel_key(path)
        if key in manifest["files"] and not args.all:
            continue
        exchange, symbol = path.parent.parent.name, path.parent.name
        name = path.name
        day = f"{name[0:4]}-{name[4:6]}-{name[6:8]}" if len(name) >= 11 else "?"
        prev = manifest["files"].get(key, {})
        source = prev.get("source", "recorder")
        lines = None if args.no_lines else count_lines_gz(path)
        manifest["files"][key] = file_entry(path, source, exchange, symbol,
                                            day, lines)
        added += 1
        print(f"  {key}  {manifest['files'][key]['bytes']:,} B", flush=True)
    save_manifest(manifest_path, manifest)
    print(f"{added} file(s) hashed into {manifest_path}", flush=True)
    return 0


def cmd_verify(args) -> int:
    manifest = load_manifest(Path(args.manifest))
    ok = bad = missing = 0
    for key, entry in manifest["files"].items():
        path = ROOT / key if not Path(key).is_absolute() else Path(key)
        if not path.exists():
            missing += 1
            if not args.quiet:
                print(f"MISSING {key}", flush=True)
            continue
        digest = sha256_of(path)
        if digest == entry.get("sha256"):
            ok += 1
        else:
            bad += 1
            print(f"MISMATCH {key}: manifest {entry.get('sha256')} "
                  f"disk {digest}", flush=True)
    print(f"verify: {ok} ok, {bad} mismatched, {missing} missing "
          f"(of {len(manifest['files'])})", flush=True)
    if bad:
        return 1
    if missing and args.strict:
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download one or more days from Tardis")
    f.add_argument("--day", required=True,
                   help="UTC day YYYY-MM-DD (comma-separate several); free "
                        "without a key only when DD=01")
    f.add_argument("--exchange", default="coinbase")
    f.add_argument("--symbol", default="BTC-USD")
    f.add_argument("--channels", default=",".join(DEFAULT_CHANNELS),
                   help="Tardis channel names (default snapshot,l2update,match)")
    f.add_argument("--minutes", type=int, default=MINUTES_PER_DAY,
                   help="minutes from 00:00 UTC to fetch (default 1440; "
                        "10 makes a one-request smoke test)")
    f.add_argument("--slice-size", type=int, default=10, choices=range(1, 11),
                   metavar="1-10", help="minutes per HTTP request (default 10)")
    f.add_argument("--out", default=str(DATA), help="data root (default ./data)")
    f.add_argument("--manifest", default=str(MANIFEST))
    f.add_argument("--api-key", default=None,
                   help="Tardis API key (or env TARDIS_API_KEY); optional")
    f.add_argument("--compresslevel", type=int, default=6)
    f.add_argument("--force", action="store_true",
                   help="overwrite files for a day that already exists")
    f.set_defaults(func=cmd_fetch)

    h = sub.add_parser("hash", help="add recorder-written files to the manifest")
    h.add_argument("--out", default=str(DATA))
    h.add_argument("--manifest", default=str(MANIFEST))
    h.add_argument("--all", action="store_true", help="re-hash known files too")
    h.add_argument("--no-lines", action="store_true",
                   help="skip line counts (faster on large files)")
    h.set_defaults(func=cmd_hash)

    v = sub.add_parser("verify", help="recompute sha256 for every manifest file")
    v.add_argument("--manifest", default=str(MANIFEST))
    v.add_argument("--strict", action="store_true",
                   help="missing files fail too (default: only mismatches)")
    v.add_argument("--quiet", action="store_true")
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
