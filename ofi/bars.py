"""Fixed-width bars with explicit gap accounting.

Rules (each one is a test in tests/test_bars.py):

* An event with t >= bar_end closes every bar up to and including the one
  ending before t BEFORE the event is applied, so bar k never contains the
  first event of bar k+1 (the old harness applied first and closed after).
* A quiet stretch shorter than `max_gap` yields empty bars (n_events == 0,
  zero flow, mid carried) because the book genuinely did not move; a
  stretch longer than `max_gap`, a disconnect marker or a fresh snapshot
  yields NO bars for the silent span. The next bar starts a new segment
  (`seg` increments) and carries is_gap == 1. Nothing is ever fabricated
  for time the record does not cover.
* A price never crosses a break either: the carried book state is dropped
  with the segment, so a new segment's first bar is NaN-priced until a book
  observation is made inside it. Otherwise a trade arriving after a long
  silence would open a bar priced at the pre-gap mid.
* Forward returns are only ever formed within one segment (inference.py).

Bar fields (all numpy arrays of equal length after `finish()`):
    t_end     bar end, ns                     mid        mid at close
    spread    ask-bid at close, bps of mid    micro      micro-price - mid, bps
    ofi       sum of CKS e_n in the bar       mlofi      k-level OFI sum
    tflow     signed taker volume             n_events   book events in bar
    is_gap    1 if the previous bar is not contiguous with this one
    seg       contiguous-run id
"""
from __future__ import annotations

import numpy as np

FIELDS = ("t_end", "mid", "spread", "micro", "ofi", "mlofi", "tflow",
          "n_events", "is_gap", "seg")


class BarBuilder:
    def __init__(self, bar_ns: int, max_gap_ns: int) -> None:
        if bar_ns <= 0 or max_gap_ns < 0:
            raise ValueError("bar_ns must be > 0 and max_gap_ns >= 0")
        self.bar_ns = int(bar_ns)
        self.max_gap_ns = int(max_gap_ns)
        self.rows: list[tuple] = []
        self.bar_end: int | None = None
        self.last_t: int | None = None
        self.seg = -1
        self.pending_break = True     # first bar always opens a segment
        self.state = None             # (mid, spread_bps, micro_bps) at last event
        self._reset_accum()

    def _reset_accum(self) -> None:
        self.ofi = 0.0
        self.mlofi = 0.0
        self.tflow = 0.0
        self.n_events = 0

    # ------------------------------------------------------------------
    def on_break(self, t_ns: int | None = None) -> None:
        """Disconnect marker, fresh snapshot, or explicit gap: end the run.

        The open bar (if any) is closed with what it has; no bars are
        emitted for the span up to the next event, and that event opens a
        new segment.

        The carried book state is dropped too. Without that, a segment
        opened by an event that carries no state -- a trade arriving after
        a long silence, with no book message of its own -- would emit a bar
        stamped after the gap but priced before it, and a forward return
        measured from that bar would start from a mid that is as old as the
        gap. A new segment may only be priced by a book observation made
        inside it; until one arrives the mid is NaN, which every consumer
        already drops.
        """
        if self.bar_end is not None and self.last_t is not None:
            self._emit(self.bar_end)
        self.bar_end = None
        self.pending_break = True
        self.state = None
        self._reset_accum()
        if t_ns is not None:
            self.last_t = int(t_ns)

    def _emit(self, t_end: int) -> None:
        if self.pending_break:
            self.seg += 1
            is_gap = 1 if self.rows else 0
            self.pending_break = False
        else:
            is_gap = 0
        mid, spr, mic = self.state if self.state is not None else (np.nan,) * 3
        self.rows.append((t_end, mid, spr, mic, self.ofi, self.mlofi,
                          self.tflow, self.n_events, is_gap, self.seg))
        self._reset_accum()

    def close_through(self, t_ns: int) -> None:
        """Emit every bar whose end is <= t_ns (bars are [end-bar, end))."""
        while self.bar_end is not None and t_ns >= self.bar_end:
            self._emit(self.bar_end)
            self.bar_end += self.bar_ns

    def on_event(self, t_ns: int, ofi: float = 0.0, mlofi: float = 0.0,
                 tflow: float = 0.0, state: tuple | None = None,
                 is_book_event: bool = True) -> None:
        t_ns = int(t_ns)
        if self.last_t is not None and t_ns - self.last_t > self.max_gap_ns:
            self.on_break()
        if self.bar_end is None:
            # first event of a segment: open the bar containing t
            self.bar_end = (t_ns // self.bar_ns + 1) * self.bar_ns
        else:
            self.close_through(t_ns)        # BEFORE applying this event
        self.ofi += ofi
        self.mlofi += mlofi
        self.tflow += tflow
        if is_book_event:
            self.n_events += 1
        if state is not None:
            self.state = state
        self.last_t = t_ns

    def finish(self) -> dict[str, np.ndarray]:
        """Close the open bar and return the bars as numpy arrays."""
        if self.bar_end is not None and self.last_t is not None:
            self._emit(self.bar_end)
            self.bar_end = None
        return rows_to_arrays(self.rows)


def rows_to_arrays(rows: list[tuple]) -> dict[str, np.ndarray]:
    if not rows:
        return {"t_end": np.zeros(0, dtype=np.int64),
                "mid": np.zeros(0), "spread": np.zeros(0), "micro": np.zeros(0),
                "ofi": np.zeros(0), "mlofi": np.zeros(0), "tflow": np.zeros(0),
                "n_events": np.zeros(0, dtype=np.int64),
                "is_gap": np.zeros(0, dtype=np.int64),
                "seg": np.zeros(0, dtype=np.int64)}
    cols = list(zip(*rows))
    out = {}
    for name, col in zip(FIELDS, cols):
        if name in ("t_end", "n_events", "is_gap", "seg"):
            out[name] = np.asarray(col, dtype=np.int64)
        else:
            out[name] = np.asarray(col, dtype=np.float64)
    return out


def concat_bars(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate day bar sets; segment ids are re-based so days never join."""
    parts = [p for p in parts if len(p["t_end"])]
    if not parts:
        return rows_to_arrays([])
    out = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
    seg = np.zeros(len(out["t_end"]), dtype=np.int64)
    is_gap = out["is_gap"].copy()
    base = 0
    i = 0
    for j, p in enumerate(parts):
        n = len(p["t_end"])
        seg[i:i + n] = p["seg"] + base
        if j > 0:
            is_gap[i] = 1
        base = int(seg[i + n - 1]) + 1
        i += n
    out["seg"] = seg
    out["is_gap"] = is_gap
    return out
