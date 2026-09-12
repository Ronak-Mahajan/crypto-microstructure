"""Order-flow-imbalance measurement pipeline for the recorded/Tardis data.

Modules
    io          read data/<venue>/<product>/<YYYYMMDD-HH>.jsonl.gz incl. markers
    book        sorted price-level book, Coinbase l2update semantics, snapshot check
    features    CKS best-level OFI (eq. 10), multi-level OFI, micro-price, trade sign
    bars        fixed-width bars with gap flags; closes a bar BEFORE applying t>=end
    pipeline    messages -> bars + book statistics for one day
    inference   purged walk-forward, HAC t-stats, block bootstrap, CKS R^2
    fees        Coinbase taker tiers, Hyperliquid taker, edge-minus-cost
    leadlag     cross-venue signed-flow cross-correlation
    report      results/README.md in the fixed table order
"""
__all__ = ["io", "book", "features", "bars", "pipeline", "inference",
           "fees", "leadlag", "report"]
