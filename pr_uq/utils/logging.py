"""Logging + result persistence utilities.

Two flavors:
  * `get_logger` — a stdlib logger that writes to console and a log file
    inside the run output directory.
  * `MetricsWriter` — appends per-image scores to a CSV.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path


def get_logger(name: str, output_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Construct a logger with a console handler and optional file handler.

    Calling `get_logger` twice with the same name and output_dir returns
    the same logger without duplicating handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class MetricsWriter:
    """Append-only CSV writer for per-image metrics.

    Use `write_row({...})` per image. The first call creates the file and
    writes a header row derived from the dict keys; later calls with new
    keys will fail loudly (no silent column drift).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def write_row(self, row: dict) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        if set(row.keys()) != set(self._fieldnames):
            missing = set(self._fieldnames) - set(row.keys())
            extra = set(row.keys()) - set(self._fieldnames)
            raise ValueError(
                f"Row schema mismatch for {self.path}: missing={missing}, extra={extra}"
            )
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)


def write_json(obj: dict, path: str | Path) -> None:
    """Pretty-print a dict to JSON. Tensors/Path values are str()'d."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        try:
            import torch
            if isinstance(o, torch.Tensor):
                return o.tolist()
        except ImportError:
            pass
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)
