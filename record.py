"""Record raw L2 depth and trades from free crypto venue websockets.

Phase 0 of the microstructure study. The whole design bet is that RAW venue
messages are the right thing to store: snapshots, depth diffs, and trades
exactly as the exchange sent them, stamped with local arrival time in
nanoseconds. Order books can be rebuilt, features recomputed, and impact
estimated later under any definition; a normalized schema chosen today would
quietly bake in today's assumptions. Storage is the cheap resource, recorded
market states are the unrecoverable one.

Venues (all free, no API keys, no auth):

    coinbase     wss://ws-feed.exchange.coinbase.com
                 channels: level2_batch (full-depth snapshot + 50 ms batched
                 diffs), matches (trades), heartbeat
    hyperliquid  wss://api.hyperliquid.xyz/ws
                 subscriptions: l2Book (a 20-level-per-side SNAPSHOT pushed at
                 most every ~0.5 s; not full depth, not diffs) and trades

Binance is deliberately absent: binance.com websockets are geo-blocked from
US networks, and a recorder that dies by jurisdiction is not a recorder.

Output: data/<venue>/<product>/<YYYYMMDD-HH>.jsonl.gz, one directory per
product (coinbase/BTC-USD, coinbase/ETH-USD, hyperliquid/BTC-l2Book, ...),
one JSON object per line:

    {"t_ns": <local arrival, ns>, "vt": <venue time, ns, when present>,
     "m": <verbatim venue message>}

plus marker lines whenever a connection opens or drops, so gaps are in the
record itself rather than only on stdout:

    {"t_ns": ..., "marker": "connect"|"disconnect", "venue": "<stream>",
     "reason": null|"<exception repr>", "m": null}

("m": null keeps the line readable by any consumer that expects the "m" key;
analyze.py skips it.) Files rotate hourly by arrival time; a flush every
200 lines or 2 s bounds the loss from a hard kill.

Hardening relative to the first version (all cheap, all in this file):

  * max_size 64 MiB per frame: a full Coinbase BTC-USD snapshot is several
    MB and the old 8 MiB cap would close the socket with code 1009 and
    reconnect forever, recording nothing.
  * Backoff resets only after the first received message, so a venue that
    accepts the socket and immediately closes it backs off to 60 s instead
    of a 1 s reconnect storm.
  * gzip (compresslevel=1) and file I/O run in one worker thread fed by a
    queue; the event loop only stamps and enqueues, so a multi-MB snapshot
    on one stream no longer delays arrival stamps on the other four.
  * Frames are validated as JSON before being spliced into the line; a
    non-JSON frame or one containing a raw newline can no longer tear the
    NDJSON stream.
  * Hyperliquid gets an application-level {"method":"ping"} every 30 s
    (the server drops connections silent for 60 s; BTC/ETH streams never
    are, but illiquid coins can be).
  * The venue's own timestamp is parsed into "vt" (ns) when that is cheap:
    Coinbase "time" (ISO 8601) and Hyperliquid data.time (ms).

Run it on any always-on box:

    pip install websockets
    python record.py                 # BTC + ETH on both venues
    python record.py --symbols BTC   # fewer streams
"""
from __future__ import annotations

import argparse
import asyncio
import calendar
import gzip
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

DATA = Path(__file__).resolve().parent / "data"

MAX_FRAME_BYTES = 2 ** 26        # 64 MiB; a full Coinbase book snapshot is MBs
COMPRESSLEVEL = 1                # speed over ratio; the writer is off-loop anyway
FLUSH_EVERY_LINES = 200
FLUSH_EVERY_SECS = 2.0
HL_PING_SECS = 30.0
BACKOFF_MAX_SECS = 60.0
QUEUE_MAX = 500_000              # lines buffered before the recorder drops


# ---------------------------------------------------------------------------
# Line encoding
# ---------------------------------------------------------------------------

def iso_to_ns(s: str) -> int | None:
    """'2019-08-14T20:42:27.265Z' or '...27.028459Z' -> epoch ns, else None."""
    try:
        if not s or s[-1] != "Z":
            return None
        head, _, frac = s[:-1].partition(".")
        dt = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
        secs = calendar.timegm(dt.timetuple())
        frac_ns = int((frac + "000000000")[:9]) if frac else 0
        return secs * 1_000_000_000 + frac_ns
    except (ValueError, TypeError):
        return None


def venue_time_ns(obj: dict) -> int | None:
    """Best-effort venue timestamp from a parsed message; None if absent."""
    t = obj.get("time")
    if isinstance(t, str):                       # Coinbase l2update/match/heartbeat
        return iso_to_ns(t)
    data = obj.get("data")
    if isinstance(data, dict):                   # Hyperliquid l2Book
        t = data.get("time")
        if isinstance(t, int):
            return t * 1_000_000
    elif isinstance(data, list) and data:        # Hyperliquid trades (a list)
        first = data[0]
        if isinstance(first, dict):
            t = first.get("time")
            if isinstance(t, int):
                return t * 1_000_000
    return None


def encode_line(t_ns: int, raw: str) -> tuple[str | None, dict | None]:
    """Validate one websocket text frame and build the NDJSON line.

    Returns (line, parsed) or (None, None) when the frame is not a JSON
    object. The raw text is spliced in verbatim unless it contains a raw
    newline, in which case the compact re-serialisation is used instead.
    """
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    if "\n" in raw or "\r" in raw:
        raw = json.dumps(obj, separators=(",", ":"))
    vt = venue_time_ns(obj)
    if vt is None:
        return '{"t_ns":%d,"m":%s}\n' % (t_ns, raw), obj
    return '{"t_ns":%d,"vt":%d,"m":%s}\n' % (t_ns, vt, raw), obj


def marker_line(t_ns: int, marker: str, venue: str, reason: str | None) -> str:
    return json.dumps({"t_ns": t_ns, "marker": marker, "venue": venue,
                       "reason": reason, "m": None},
                      separators=(",", ":")) + "\n"


# ---------------------------------------------------------------------------
# Writer thread
# ---------------------------------------------------------------------------

class GzWriter(threading.Thread):
    """One thread owns every output file; producers only enqueue strings.

    Queue items: ("line", venue, product, t_ns, text) | ("flush",) | None.
    Files rotate hourly by the line's t_ns (UTC). A flush happens every
    FLUSH_EVERY_LINES lines per file or whenever the queue has been idle for
    FLUSH_EVERY_SECS, whichever first.
    """

    def __init__(self, root: Path, compresslevel: int = COMPRESSLEVEL) -> None:
        super().__init__(name="gz-writer", daemon=True)
        self.root = root
        self.compresslevel = compresslevel
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self.dropped = 0
        self.written = 0
        self._files: dict[tuple[str, str], list] = {}   # key -> [hour, fh, n]

    # -- producer side (event loop) ---------------------------------------
    def put(self, venue: str, product: str, t_ns: int, text: str) -> None:
        try:
            self.q.put_nowait(("line", venue, product, t_ns, text))
        except queue.Full:
            self.dropped += 1

    def flush(self) -> None:
        try:
            self.q.put_nowait(("flush",))
        except queue.Full:
            pass

    def stop(self, timeout: float = 30.0) -> None:
        try:
            self.q.put(None, timeout=timeout)
        except queue.Full:
            return
        self.join(timeout)

    # -- consumer side (this thread) --------------------------------------
    def run(self) -> None:
        while True:
            try:
                item = self.q.get(timeout=FLUSH_EVERY_SECS)
            except queue.Empty:
                self._flush_all()
                continue
            if item is None:
                self._close_all()
                return
            if item[0] == "flush":
                self._flush_all()
                continue
            _, venue, product, t_ns, text = item
            try:
                self._write(venue, product, t_ns, text)
            except OSError as exc:
                print(f"[{time.strftime('%H:%M:%S')}] writer: {exc!r}",
                      flush=True)
                self.dropped += 1

    def _write(self, venue: str, product: str, t_ns: int, text: str) -> None:
        hour = time.strftime("%Y%m%d-%H", time.gmtime(t_ns // 1_000_000_000))
        key = (venue, product)
        cur = self._files.get(key)
        if cur is None or cur[0] != hour:
            if cur is not None:
                cur[1].close()
            d = self.root / venue / product
            d.mkdir(parents=True, exist_ok=True)
            fh = gzip.open(d / f"{hour}.jsonl.gz", "at", encoding="utf-8",
                           compresslevel=self.compresslevel)
            cur = [hour, fh, 0]
            self._files[key] = cur
        cur[1].write(text)
        cur[2] += 1
        self.written += 1
        if cur[2] >= FLUSH_EVERY_LINES:
            cur[1].flush()
            cur[2] = 0

    def _flush_all(self) -> None:
        for cur in self._files.values():
            if cur[2]:
                cur[1].flush()
                cur[2] = 0

    def _close_all(self) -> None:
        for cur in self._files.values():
            cur[1].close()
        self._files.clear()


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

class Stream:
    """One websocket connection and how its frames map to product dirs."""

    def __init__(self, name: str, venue: str, url: str, subscribe: dict,
                 products: list[str], route, keepalive: str | None) -> None:
        self.name = name
        self.venue = venue
        self.url = url
        self.subscribe = subscribe
        self.products = products          # every product dir this stream feeds
        self.route = route                # parsed msg -> product dir name
        self.keepalive = keepalive        # text to send every HL_PING_SECS


def coinbase_route(obj: dict) -> str:
    pid = obj.get("product_id")
    return pid if isinstance(pid, str) and pid else "_control"


def streams_for(symbols: list[str]) -> list[Stream]:
    out = []
    cb_products = [f"{s}-USD" for s in symbols]
    out.append(Stream(
        "coinbase", "coinbase", "wss://ws-feed.exchange.coinbase.com",
        {"type": "subscribe", "product_ids": cb_products,
         "channels": ["level2_batch", "matches", "heartbeat"]},
        cb_products, coinbase_route, None,
    ))
    for s in symbols:
        for sub_type in ("l2Book", "trades"):
            product = f"{s}-{sub_type}"
            out.append(Stream(
                f"hyperliquid:{s}:{sub_type}", "hyperliquid",
                "wss://api.hyperliquid.xyz/ws",
                {"method": "subscribe",
                 "subscription": {"type": sub_type, "coin": s}},
                [product], (lambda obj, p=product: p),
                '{"method":"ping"}',
            ))
    return out


async def _keepalive(ws, text: str, every: float) -> None:
    """Application-level ping; ends quietly once the socket is gone."""
    try:
        while True:
            await asyncio.sleep(every)
            await ws.send(text)
    except asyncio.CancelledError:
        raise
    except Exception:
        return


def _log(name: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {msg}", flush=True)


async def run_stream(st: Stream, writer: GzWriter, stats: dict) -> None:
    backoff = 1.0
    while True:
        ping_task = None
        try:
            async with websockets.connect(st.url, ping_interval=20,
                                          max_size=MAX_FRAME_BYTES) as ws:
                await ws.send(json.dumps(st.subscribe))
                t0 = time.time_ns()
                for p in st.products:
                    writer.put(st.venue, p, t0,
                               marker_line(t0, "connect", st.name, None))
                _log(st.name, "connected")
                if st.keepalive:
                    ping_task = asyncio.create_task(
                        _keepalive(ws, st.keepalive, HL_PING_SECS))
                first = True
                async for raw in ws:
                    t_ns = time.time_ns()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    line, obj = encode_line(t_ns, raw)
                    if line is None:
                        stats["bad_frames"] = stats.get("bad_frames", 0) + 1
                        continue
                    if first:
                        backoff = 1.0      # only a real message proves health
                        first = False
                    writer.put(st.venue, st.route(obj), t_ns, line)
                    stats[st.name] = stats.get(st.name, 0) + 1
                reason = "closed by peer"
        except asyncio.CancelledError:
            t1 = time.time_ns()
            for p in st.products:
                writer.put(st.venue, p, t1,
                           marker_line(t1, "disconnect", st.name, "cancelled"))
            writer.flush()
            raise
        except Exception as exc:
            reason = repr(exc)
        finally:
            if ping_task is not None:
                ping_task.cancel()
        t1 = time.time_ns()
        for p in st.products:
            writer.put(st.venue, p, t1,
                       marker_line(t1, "disconnect", st.name, reason))
        writer.flush()
        _log(st.name, f"{reason}; reconnecting in {backoff:.0f}s")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, BACKOFF_MAX_SECS)


async def main_async(symbols: list[str], report_every: float,
                     writer: GzWriter) -> None:
    stats: dict = {}
    tasks = [asyncio.create_task(run_stream(st, writer, stats))
             for st in streams_for(symbols)]

    async def reporter():
        while True:
            await asyncio.sleep(report_every)
            total = sum(v for k, v in stats.items() if k != "bad_frames")
            parts = "  ".join(f"{k}={v}" for k, v in sorted(stats.items()))
            print(f"[{time.strftime('%H:%M:%S')}] {total:,} msgs  ({parts})  "
                  f"written={writer.written:,} queued={writer.q.qsize()} "
                  f"dropped={writer.dropped}", flush=True)

    tasks.append(asyncio.create_task(reporter()))
    await asyncio.gather(*tasks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--symbols", default="BTC,ETH",
                   help="comma-separated base symbols (default BTC,ETH)")
    p.add_argument("--report-every", type=float, default=60.0,
                   help="seconds between message-count log lines")
    p.add_argument("--out", default=str(DATA),
                   help="output root (default: ./data next to this file)")
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    root = Path(args.out)
    print(f"recording {symbols} from coinbase + hyperliquid -> {root}",
          flush=True)
    writer = GzWriter(root)
    writer.start()
    try:
        asyncio.run(main_async(symbols, args.report_every, writer))
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        writer.stop()


if __name__ == "__main__":
    main()
