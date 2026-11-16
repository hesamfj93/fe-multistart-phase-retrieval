"""Evaluation protocol splits for the MNIST test set.

Three disjoint index sets, all over the 10,000-image MNIST *test* split:

  dev      The 8 images used during development. The aggregation method and
           the FE-softmax temperature (T = 0.05) were selected on these images,
           so they must never appear in a held-out evaluation.
  heldout  Images used for the main (paper) evaluation. Drawn from a fixed
           permutation of the test set with the dev images removed. Ordered,
           so the first N rows are always the same N images. Extending from
           50 to 100 only appends rows 50..99.
  heldout3 Images of the third study (Poisson noise). Positions 200 to 399 of the same
           filtered permutation, written to their own file so that the held-out file,
           whose SHA-256 is recorded in the protocol locks of the first two studies,
           stays unchanged. Rank r of heldout3 is position 200 + r of the permutation.
  smoke    3 images used only for software verification. Taken from the end
           of the same filtered permutation, so they are disjoint from both
           dev and every held-out prefix up to 9,989 images.

The CSV files in `splits/` are the source of truth. `generate_split_lists`
reproduces them exactly (legacy NumPy RandomState, whose stream is stable
across NumPy versions); `scripts/make_splits.py --check` verifies this.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

MNIST_TEST_SIZE = 10_000

# The 8 development images (MNIST test-set indices). These are the images in
# results/mnist_k1_3000iter and results/mnist_k5_3000iter, analysed in
# AGGREGATION_FINALIZATION_REPORT.md.
DEV_INDICES: tuple[int, ...] = (429, 1729, 1917, 2890, 4919, 6044, 7532, 9399)

HELDOUT_PERMUTATION_SEED = 20261006   # fixed; documented in the split files
HELDOUT_LIST_LENGTH = 200             # rows written to the held-out CSV
SMOKE_COUNT = 3
HELDOUT3_START = 200                  # first permutation position of the third study
HELDOUT3_LENGTH = 200                 # rows written to the heldout3 CSV

SPLIT_FILES = {
    "dev": "mnist_test_dev_indices.csv",
    "heldout": "mnist_test_heldout_indices.csv",
    "heldout3": "mnist_test_heldout3_indices.csv",
    "smoke": "mnist_test_smoke_indices.csv",
}


def generate_split_lists(
    dev: tuple[int, ...] = DEV_INDICES,
    seed: int = HELDOUT_PERMUTATION_SEED,
    n_heldout: int = HELDOUT_LIST_LENGTH,
    n_smoke: int = SMOKE_COUNT,
    n_total: int = MNIST_TEST_SIZE,
) -> dict[str, list[int]]:
    """Deterministically build the three ordered index lists."""
    perm = np.random.RandomState(seed).permutation(n_total).tolist()
    dev_set = set(dev)
    pool = [i for i in perm if i not in dev_set]
    heldout = pool[:n_heldout]
    heldout3 = pool[HELDOUT3_START:HELDOUT3_START + HELDOUT3_LENGTH]
    smoke = pool[-n_smoke:]
    out = {"dev": list(dev), "heldout": heldout, "heldout3": heldout3, "smoke": smoke}
    check_disjoint(out)
    return out


def check_disjoint(lists: dict[str, list[int]]) -> None:
    """Raise if any two split lists share an index or a list has duplicates."""
    for name, xs in lists.items():
        if len(set(xs)) != len(xs):
            raise ValueError(f"split '{name}' contains duplicate indices")
    names = list(lists)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ov = set(lists[names[a]]) & set(lists[names[b]])
            if ov:
                raise ValueError(
                    f"splits '{names[a]}' and '{names[b]}' overlap in {len(ov)} "
                    f"indices: {sorted(ov)[:10]}")


def split_path(splits_dir: str | Path, role: str) -> Path:
    if role not in SPLIT_FILES:
        raise ValueError(f"unknown split role {role!r}; expected one of {list(SPLIT_FILES)}")
    return Path(splits_dir) / SPLIT_FILES[role]


def write_split_csv(path: Path, role: str, indices: list[int], labels: dict[int, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if role == "dev":
            f.write("# role=dev dataset=mnist split=test; the 8 development images used to "
                    "select the aggregation method and T=0.05; never used for held-out evaluation\n")
        elif role == "heldout3":
            f.write(f"# role=heldout3 dataset=mnist split=test permutation_seed={HELDOUT_PERMUTATION_SEED} "
                    f"dev_excluded={list(DEV_INDICES)}; rank r = position {HELDOUT3_START} + r of the "
                    f"held-out permutation; first N rows = the N-image selection\n")
        else:
            f.write(f"# role={role} dataset=mnist split=test permutation_seed={HELDOUT_PERMUTATION_SEED} "
                    f"dev_excluded={list(DEV_INDICES)}; first N rows = the N-image selection\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rank", "mnist_test_index", "label"])
        for r, i in enumerate(indices):
            w.writerow([r, i, labels.get(i, -1) if labels else -1])


def read_split_csv(path: str | Path) -> list[int]:
    """Return the ordered list of MNIST test indices in a split file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for r in csv.DictReader(lines):
        rows.append((int(r["rank"]), int(r["mnist_test_index"])))
    rows.sort()
    if [r for r, _ in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: ranks are not 0..N-1")
    return [i for _, i in rows]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol_indices(splits_dir: str | Path, role: str, n_images: int, start: int = 0) -> list[int]:
    """Indices for a protocol run: rows `start` to `start + n_images - 1` of the role's file
    (the first `n_images` rows when start = 0).

    Hard guarantees, enforced here:
      * n_images does not exceed the file length (no silent truncation);
      * a held-out or smoke selection contains no development index.
    """
    path = split_path(splits_dir, role)
    if not path.exists():
        raise FileNotFoundError(f"split file missing: {path} (run scripts/make_splits.py)")
    all_idx = read_split_csv(path)
    if start < 0 or start + n_images > len(all_idx):
        raise ValueError(
            f"requested rows {start} to {start + n_images - 1} of '{role}' but {path.name} has only "
            f"{len(all_idx)} rows")
    sel = all_idx[start:start + n_images]
    if role != "dev":
        dev_file = split_path(splits_dir, "dev")
        dev = set(read_split_csv(dev_file)) if dev_file.exists() else set(DEV_INDICES)
        dev |= set(DEV_INDICES)
        ov = sorted(set(sel) & dev)
        if ov:
            raise RuntimeError(f"PROTOCOL VIOLATION: '{role}' selection contains "
                               f"development indices {ov}")
    return sel
