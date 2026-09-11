# crypto-microstructure

Does order-flow imbalance predict short-horizon returns, and does anything
survive fees? A measurement project on free crypto level-2 data: full days
served by Tardis.dev's no-key feed, plus days this repo records itself.

Most microstructure studies at student level are impossible for one reason:
the data costs thousands. This project routes around that twice. Tardis.dev
replays the raw Coinbase websocket feed for the first day of every month
without an API key, which gives independent full days of BTC-USD level-2
depth on day one; and `record.py` captures Coinbase and Hyperliquid around
the clock for the days in between. Both land in one raw-message format, so
every analysis reads both without caring where a day came from.

## Status

- **Recorder** (`record.py`): hardened and tested offline (frame cap, writer
  thread, reconnect markers, per-product directories, Hyperliquid keepalive).
  No live capture is committed yet.
- **Tardis loader** (`tardis_loader.py`): endpoint verified against the
  Tardis HTTP API reference; the fetch, hash and verify paths are tested on
  synthetic slices. No day has been pulled yet, so `manifest.json` is empty.
- **Phase 1 harness** (`analyze.py`): rebuilds the Coinbase book, computes
  best-level OFI (Cont, Kukanov and Stoikov 2014), walk-forward OLS. No
  result has been produced on real data; the harness is being reworked
  (sorted book, per-event OFI, gap flags, purged folds, HAC errors) before
  the first number is published.

There is no headline number yet. When there is one it will sit here with
the command that regenerates it from the manifest.

## Data

### Tardis free days

```
python tardis_loader.py fetch --day 2025-08-01              # one free day
python tardis_loader.py fetch --day 2025-08-01 --minutes 10 # one-slice smoke test
python tardis_loader.py verify                              # recheck sha256s
```

The loader calls `GET https://api.tardis.dev/v1/data-feeds/coinbase` with
`from=<day>`, `offset=<minute>`, `sliceSize=10` and a `filters` list for the
`snapshot`, `l2update` and `match` channels on `BTC-USD`, 144 requests per
UTC day. Each response line is Tardis's local arrival timestamp followed by
the exchange-native message; the loader stores it as
`{"t_ns": <arrival ns>, "m": <verbatim message>}` in
`data/coinbase/BTC-USD/<YYYYMMDD-HH>.jsonl.gz`, the recorder's own layout.
Empty response lines, which Tardis uses to mark its own reconnects, become
marker lines in the same file so gaps stay in the record. Without an API key
only `YYYY-MM-01` days are served; the loader refuses other days up front.

Size, honestly: Coinbase publishes level-2 unbatched, so a BTC-USD day is
millions of `l2update` messages. Expect a multi-GB download per day and on
the order of 1-3 GB gzipped on disk; the loader streams ten minutes at a
time and never holds more than one slice in memory, but the disk is yours.
The true byte count of every pulled day is written to the manifest.

### Self-recorded days

```
pip install websockets
python record.py                 # BTC + ETH on both venues
python record.py --symbols BTC
```

Raw venue messages land in `data/<venue>/<product>/<YYYYMMDD-HH>.jsonl.gz`,
one directory per product, as `{"t_ns": <local arrival ns>, "vt": <venue
time ns when present>, "m": <verbatim message>}`. Every connect and
disconnect writes a marker line into the same stream. Coinbase
`level2_batch` is full depth (snapshot plus 50 ms batched diffs) with
`matches` and `heartbeat`. Hyperliquid `l2Book` is **not** full depth: it is
a 20-level-per-side snapshot pushed at most every ~0.5 s, plus a `trades`
stream; cross-venue comparisons on the book side are therefore limited to
1 s and coarser, while trades allow finer resolution. Compression runs in a
worker thread at level 1 so a multi-MB snapshot on one stream does not delay
arrival stamps on the others. Throughput per day is unmeasured until a live
run is committed. `python tardis_loader.py hash` adds recorded files to the
manifest.

### Manifest

`manifest.json` records, for every data file, its sha256, byte and line
counts, source (`tardis` or `recorder`), exchange, symbol, day and hour, and
per Tardis day the message counts by channel and the arrival time of the
first snapshot. Data files are gitignored; the manifest is the committed
statement of exactly which bytes any published number came from.

## The plan

**Phase 1, signal.** Rebuild books from the recorded diffs; compute
order-flow imbalance, multi-level OFI, micro-price and signed trade flow;
test whether they predict returns at horizons from 1 second to 5 minutes.
Purged walk-forward splits only, Newey-West standard errors because forward
returns overlap, block-bootstrap intervals, and a contemporaneous CKS
replication first as the pipeline sanity anchor. The edge is stated net of
Coinbase taker tiers, Hyperliquid taker and half-spread. The result gets
stated plainly whichever way it comes out: a well-measured negative is a
result.

**Phase 2, execution.** A minimal Kyle-lambda temporary-impact regression
from the recorded trades, then TWAP, VWAP, POV and Almgren-Chriss child-order
schedules against queue-position-aware fills.

**Phase 3, the loop.** Feed the measured impact and fill parameters back
into the [Avellaneda-Stoikov study](https://github.com/Ronak-Mahajan/hft-market-maker)
whose simulator assumed them.

The next stage adds a `Makefile` with `make test` (pytest), `make fetch
DAY=YYYY-MM-01` (Tardis pull plus manifest update), `make verify` (sha256
check) and `make results` (regenerate every table from the manifest), and
CI runs the tests plus a small-fixture smoke analysis.

## Ground rules

Carried over from [neural-options-lab](https://github.com/Ronak-Mahajan/neural-options-lab):
every headline number ships with the code that produced it, out-of-sample
means out-of-sample, and negative results get published at the same font
size as positive ones.

## Layout

```
record.py           raw L2 + trades recorder for Coinbase and Hyperliquid (websockets, no keys)
tardis_loader.py    Tardis.dev free-day loader, manifest hashing and verification (stdlib)
analyze.py          phase 1 harness: book rebuild, CKS best-level OFI, walk-forward OLS
manifest.json       sha256 / size / source per data file; the committed data record
tests/              offline pytest suites for the recorder and the loader
.github/workflows/  ci: compile, pytest, loader CLI smoke; failure logs land in .ci/
data/               captures and Tardis pulls (gitignored; grows without bound)
```
