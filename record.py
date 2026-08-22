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
                 channels: level2_batch (snapshot + 50ms-batched diffs),
                 matches (trades), heartbeats (sequence-gap detection)
    hyperliquid  wss://api.hyperliquid.xyz/ws
                 subscriptions: l2Book and trades per coin

Binance is deliberately absent: binance.com websockets are geo-blocked from
US networks, and a recorder that dies by jurisdiction is not a recorder.

Output: data/<venue>/<symbol>/<YYYYMMDD-HH>.jsonl.gz, one JSON object per
line: {"t_ns": <local arrival>, "m": <verbatim venue message>}. Files rotate
hourly; gzip members are appended per flush, which every standard gzip
reader handles. Roughly 150-400 MB/day for the default four streams,
depending on volatility.

Run it on any always-on box:

    pip install websockets
    python record.py                 # BTC + ETH on both venues
    python record.py --symbols BTC   # fewer streams

Each venue connection reconnects with exponential backoff and logs a line
per (re)connect, so gaps are visible in the record rather than silent.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import signal
import sys
import time
from pathlib import Path

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

DATA = Path(__file__).resolve().parent / "data"


class HourlyGzWriter:
    """Append-only gzip NDJSON with hourly rotation and periodic flush."""

    def __init__(self, venue: str, symbol: str) -> None:
        self.dir = DATA / venue / symbol.replace("/", "-")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._hour = ""
        self._fh = None
        self._n_since_flush = 0

    def write(self, msg: str) -> None:
        hour = time.strftime("%Y%m%d-%H", time.gmtime())
        if hour != self._hour:
            if self._fh:
                self._fh.close()
            self._fh = gzip.open(self.dir / f"{hour}.jsonl.gz", "at",
                                 encoding="utf-8")
            self._hour = hour
        self._fh.write('{"t_ns":%d,"m":%s}\n' % (time.time_ns(), msg))
        self._n_since_flush += 1
        if self._n_since_flush >= 200:
            self._fh.flush()
            self._n_since_flush = 0

    def close(self) -> None:
        if self._fh:
            self._fh.close()


async def run_stream(name: str, url: str, subscribe: dict,
                     writer: HourlyGzWriter, stats: dict) -> None:
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20,
                                          max_size=2 ** 23) as ws:
                await ws.send(json.dumps(subscribe))
                print(f"[{time.strftime('%H:%M:%S')}] {name}: connected",
                      flush=True)
                backoff = 1.0
                async for raw in ws:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    writer.write(raw)
                    stats[name] = stats.get(name, 0) + 1
        except asyncio.CancelledError:
            writer.close()
            raise
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {exc!r}; "
                  f"reconnecting in {backoff:.0f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


def streams_for(symbols: list[str]):
    out = []
    cb_products = [f"{s}-USD" for s in symbols]
    out.append((
        "coinbase",
        "wss://ws-feed.exchange.coinbase.com",
        {"type": "subscribe", "product_ids": cb_products,
         "channels": ["level2_batch", "matches", "heartbeat"]},
        HourlyGzWriter("coinbase", "+".join(cb_products)),
    ))
    for s in symbols:
        for sub_type in ("l2Book", "trades"):
            out.append((
                f"hyperliquid:{s}:{sub_type}",
                "wss://api.hyperliquid.xyz/ws",
                {"method": "subscribe",
                 "subscription": {"type": sub_type, "coin": s}},
                HourlyGzWriter("hyperliquid", f"{s}-{sub_type}"),
            ))
    return out


async def main_async(symbols: list[str], report_every: float) -> None:
    stats: dict = {}
    tasks = [asyncio.create_task(run_stream(n, u, sub, w, stats))
             for n, u, sub, w in streams_for(symbols)]

    async def reporter():
        while True:
            await asyncio.sleep(report_every)
            total = sum(stats.values())
            parts = "  ".join(f"{k}={v}" for k, v in sorted(stats.items()))
            print(f"[{time.strftime('%H:%M:%S')}] {total:,} msgs  ({parts})",
                  flush=True)

    tasks.append(asyncio.create_task(reporter()))
    await asyncio.gather(*tasks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--symbols", default="BTC,ETH",
                   help="comma-separated base symbols (default BTC,ETH)")
    p.add_argument("--report-every", type=float, default=60.0,
                   help="seconds between message-count log lines")
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"recording {symbols} from coinbase + hyperliquid -> {DATA}",
          flush=True)
    try:
        asyncio.run(main_async(symbols, args.report_every))
    except KeyboardInterrupt:
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
