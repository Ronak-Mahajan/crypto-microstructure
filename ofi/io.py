"""Readers for the recorder / Tardis on-disk layout.

Files: data/<venue>/<product>/<YYYYMMDD-HH>.jsonl.gz, one JSON object per line:

    {"t_ns": <arrival ns>, ["vt": <venue ns>], "m": {...venue message...}}
    {"t_ns": ..., "marker": "connect"|"disconnect", "venue": ..., "reason": ..., "m": null}

Both record.py and tardis_loader.py write exactly this. Lines are yielded as
events so the book rebuild can treat a disconnect marker as a gap boundary:

    ("msg",    t_ns, vt_or_None, m)
    ("marker", t_ns, marker_str, d)
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Iterable, Iterator

Event = tuple


def hour_files(product_dir: Path, day: str | None = None) -> list[Path]:
    """All hour files of one product dir, ordered by hour name (not path).

    `day` is YYYY-MM-DD; when given only that day's hours are returned.
    """
    prefix = day.replace("-", "") if day else None
    out = []
    for p in Path(product_dir).glob("*.jsonl.gz"):
        if prefix is not None and not p.name.startswith(prefix):
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.name)


def iter_file(path: Path) -> Iterator[Event]:
    """Yield events from one gzip NDJSON file; tolerate a torn tail."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line or line == "\n":
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue            # torn final line from a hard stop
                if not isinstance(d, dict) or "t_ns" not in d:
                    continue
                t_ns = d["t_ns"]
                if not isinstance(t_ns, int):
                    continue
                m = d.get("m")
                if m is None:
                    marker = d.get("marker")
                    if isinstance(marker, str):
                        yield ("marker", t_ns, marker, d)
                    continue
                if isinstance(m, dict):
                    yield ("msg", t_ns, d.get("vt"), m)
    except EOFError:
        # The recorder is still writing this file: the last gzip member has
        # no end-of-stream marker yet. Everything before the tear is valid.
        return


def iter_files(paths: Iterable[Path]) -> Iterator[Event]:
    for p in paths:
        yield from iter_file(Path(p))


def iter_product(root: Path, venue: str, product: str,
                 day: str | None = None) -> Iterator[Event]:
    """Events for one product (and optionally one day) under a data root."""
    return iter_files(hour_files(Path(root) / venue / product, day))


def write_lines(path: Path, lines: Iterable[str], compresslevel: int = 6) -> int:
    """Test/fixture helper: write NDJSON lines (each ending in newline).

    Byte-for-byte reproducible, which the committed fixture needs: its
    sha256 is in tests/fixtures/synthetic/manifest.json and is only a
    statement about the generator if re-running the generator reproduces it.
    Two things otherwise get in the way, and both are defaults:

    * `newline=None` (what `gzip.open(..., "wt")` passes to TextIOWrapper)
      translates every "\\n" to os.linesep on write, so the same generator
      emitted CRLF inside the gzip stream on Windows and LF on Linux.
    * GzipFile stamps the current time into the gzip header, so the same
      bytes hashed differently one second later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.GzipFile(path, "wb", compresslevel=compresslevel, mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as fh:
            for ln in lines:
                fh.write(ln if ln.endswith("\n") else ln + "\n")
                n += 1
    return n
