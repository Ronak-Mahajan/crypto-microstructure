"""Readers for the recorder / Tardis on-disk layout.

Files: data/<venue>/<product>/<YYYYMMDD-HH>.jsonl.gz, one JSON object per line:

    {"t_ns": <arrival ns>, ["vt": <venue ns>], "m": {...venue message...}}
    {"t_ns": ..., "marker": "connect"|"disconnect", "venue": ..., "reason": ..., "m": null}

Both record.py and tardis_loader.py write exactly this. Lines are yielded as
events so the book rebuild can treat a disconnect marker as a gap boundary:

    ("msg",    t_ns, vt_or_None, m)
    ("marker", t_ns, marker_str, d)

A line that is neither is dropped and counted by reason (DROP_REASONS) in
the optional `tally` dict, which also receives the sha256 of the gzip file
and of its decompressed content and the line count, all taken from the
bytes this reader actually consumed.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Iterator

Event = tuple

# Why a line was dropped. torn_tail is a final line with no newline that
# does not parse (a recorder stopped mid-write); bad_json is an unparseable
# line anywhere else.
DROP_REASONS = ("torn_tail", "bad_json", "not_object", "no_t_ns",
                "t_ns_not_int", "m_null_no_marker", "m_not_object", "blank")


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


class _HashingReader:
    """The raw gzip file, hashed and counted as gzip reads it."""

    def __init__(self, fh) -> None:
        self._fh = fh
        self.sha = hashlib.sha256()
        self.n = 0

    def read(self, size: int = -1) -> bytes:
        b = self._fh.read(size)
        self.sha.update(b)
        self.n += len(b)
        return b

    def seek(self, *a):                       # gzip only seeks to rewind
        raise io.UnsupportedOperation("hashing reader is forward-only")

    def seekable(self) -> bool:
        return False


class _ContentTap(io.RawIOBase):
    """The decompressed stream, optionally hashed, in large chunks.

    Hashing here rather than per line keeps the cost to one update per
    chunk. A gzip member with no end-of-stream marker (the recorder is
    still writing the file) ends the stream instead of raising, so the
    lines before the tear are read and the torn one reaches the caller.
    """

    def __init__(self, gz, hashed: bool) -> None:
        self._gz = gz
        self.sha = hashlib.sha256() if hashed else None
        self.n = 0
        self.torn = False

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        try:
            n = self._gz.readinto(b)
        except EOFError:
            self.torn = True
            return 0
        if n and self.sha is not None:
            self.sha.update(memoryview(b)[:n])
        self.n += n or 0
        return n


def new_tally() -> dict:
    return {"lines": 0, "content_bytes": 0, "content_sha256": None,
            "sha256": None, "bytes": 0, "torn": False, "complete": False,
            "dropped": {r: 0 for r in DROP_REASONS}}


def iter_file(path: Path, tally: dict | None = None) -> Iterator[Event]:
    """Yield events from one gzip NDJSON file; tolerate a torn tail.

    With `tally` (see new_tally) the file and its decompressed content are
    hashed as they are read, lines are counted, and every dropped line is
    counted by reason. `complete` is set only when the file was read to
    its end.
    """
    t = tally
    raw = open(path, "rb")
    src = _HashingReader(raw) if t is not None else raw
    drop = t["dropped"] if t is not None else None
    tap = None
    n_lines = 0
    loads = json.loads
    try:
        tap = _ContentTap(gzip.GzipFile(fileobj=src, mode="rb"), t is not None)
        for line in io.BufferedReader(tap, 1 << 16):
            n_lines += 1
            if line == b"\n":
                if drop is not None:
                    drop["blank"] += 1
                continue
            try:
                d = loads(line.decode("utf-8"))
            except ValueError:
                if drop is not None:
                    drop["torn_tail" if not line.endswith(b"\n")
                         else "bad_json"] += 1
                continue
            if not isinstance(d, dict):
                if drop is not None:
                    drop["not_object"] += 1
                continue
            if "t_ns" not in d:
                if drop is not None:
                    drop["no_t_ns"] += 1
                continue
            t_ns = d["t_ns"]
            if not isinstance(t_ns, int):
                if drop is not None:
                    drop["t_ns_not_int"] += 1
                continue
            m = d.get("m")
            if m is None:
                marker = d.get("marker")
                if isinstance(marker, str):
                    yield ("marker", t_ns, marker, d)
                elif drop is not None:
                    drop["m_null_no_marker"] += 1
                continue
            if isinstance(m, dict):
                yield ("msg", t_ns, d.get("vt"), m)
            elif drop is not None:
                drop["m_not_object"] += 1
        if t is not None:
            t["complete"] = True
    finally:
        raw.close()
        if t is not None and tap is not None:
            t["lines"] = n_lines
            t["torn"] = tap.torn
            t["content_bytes"] = tap.n
            t["content_sha256"] = tap.sha.hexdigest()
            t["sha256"] = src.sha.hexdigest()
            t["bytes"] = src.n


def iter_files(paths: Iterable[Path],
               tallies: list | None = None) -> Iterator[Event]:
    """Events of several files in order; with `tallies`, one tally per file
    is appended to it as that file is opened."""
    for p in paths:
        t = None
        if tallies is not None:
            t = new_tally()
            t["path"] = Path(p)
            tallies.append(t)
        yield from iter_file(Path(p), t)


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
