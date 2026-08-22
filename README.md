# crypto-microstructure

Does order-flow imbalance predict short-horizon returns, and does anything
survive fees? A measurement project on free, self-recorded L2 crypto data.

Most microstructure studies at student level are impossible for one reason:
the data costs thousands. This project routes around that by recording its
own. Coinbase and Hyperliquid publish full L2 depth and trades over free
websockets; `record.py` captures the raw feeds around the clock, and every
analysis below runs on data this repo recorded itself. The dataset compounds
daily and cannot be bought, which is the point.

## Status

Phase 0 (recording): scaffolded, ready to run. Start it on an always-on
machine and the clock starts:

```
pip install websockets
python record.py
```

Raw venue messages (snapshots, depth diffs, trades) land in
`data/<venue>/<symbol>/<YYYYMMDD-HH>.jsonl.gz` with nanosecond arrival
stamps. Roughly 150 to 400 MB/day for BTC + ETH on both venues. Raw is the
deliberate choice: books can be rebuilt and features recomputed later under
any definition, while a normalized schema chosen today would bake in today's
assumptions.

## The plan

**Phase 1, signal (weeks 1-3 of data).** Rebuild books from the recorded
diffs; compute order-flow imbalance, micro-price, and depth features; test
whether they predict returns at horizons from 1 second to 5 minutes.
Walk-forward out-of-sample splits only, linear and gradient-boosted
baselines, confidence intervals on everything. The result gets stated
plainly whichever way it comes out: a well-measured negative is a result.

**Phase 2, execution (weeks 4-6).** Estimate temporary and permanent price
impact from the recorded books, then replay TWAP, VWAP, POV, and
Almgren-Chriss child-order schedules against queue-position-aware fills.
The deliverable is a transaction-cost comparison table on data the repo
recorded itself.

**Phase 3, the loop.** Feed the measured impact and fill parameters back
into the [Avellaneda-Stoikov study](https://github.com/Ronak-Mahajan/hft-market-maker)
this project's simulator assumed them for.

## Ground rules

Carried over from [neural-options-lab](https://github.com/Ronak-Mahajan/neural-options-lab):
every headline number ships with the code that produced it, out-of-sample
means out-of-sample, and negative results get published at the same font
size as positive ones.

## Layout

```
record.py    phase 0: raw L2 + trades recorder (websockets, no keys)
data/        recorded captures (gitignored; grows without bound)
```
