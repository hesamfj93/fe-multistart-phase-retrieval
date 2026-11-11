#!/usr/bin/env python
"""Generate or verify the MNIST test-set protocol split files in splits/.

  python scripts/make_splits.py           # write splits/*.csv (refuses to overwrite)
  python scripts/make_splits.py --check   # verify files match deterministic regeneration

The committed CSV files are the source of truth for every protocol run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.data.splits import (
    DEV_INDICES, generate_split_lists, read_split_csv, split_path, write_split_csv,
    check_disjoint,
)


def mnist_test_labels(root: Path) -> dict[int, int]:
    """Read MNIST test labels from the offline IDX file (for the label column only)."""
    import gzip
    import numpy as np
    for name in ["t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"]:
        p = root / "MNIST" / "raw" / name
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rb") as f:
                data = f.read()
            labels = np.frombuffer(data, dtype=np.uint8, offset=8)
            return {i: int(v) for i, v in enumerate(labels)}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=str(REPO / "splits"))
    ap.add_argument("--data-root", default=str(REPO / "data"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    lists = generate_split_lists()
    sdir = Path(args.splits_dir)

    if args.check:
        ok = True
        for role, expected in lists.items():
            p = split_path(sdir, role)
            if not p.exists():
                print(f"[check] MISSING {p}")
                ok = False
                continue
            got = read_split_csv(p)
            same = got == expected
            ok &= same
            print(f"[check] {role:<8} {len(got):>4} rows  {'OK' if same else 'MISMATCH'}")
        on_disk = {r: read_split_csv(split_path(sdir, r)) for r in lists if split_path(sdir, r).exists()}
        try:
            check_disjoint(on_disk)
            print("[check] dev / heldout / smoke are pairwise disjoint")
        except ValueError as e:
            print(f"[check] OVERLAP: {e}")
            ok = False
        print("[check] ALL OK" if ok else "[check] FAILED")
        return 0 if ok else 1

    labels = mnist_test_labels(Path(args.data_root))
    for role, idx in lists.items():
        p = split_path(sdir, role)
        if p.exists() and not args.force:
            print(f"exists, not overwritten: {p}")
            continue
        write_split_csv(p, role, idx, labels)
        print(f"wrote {p} ({len(idx)} rows)")
    print(f"dev indices: {list(DEV_INDICES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
