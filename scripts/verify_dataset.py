#!/usr/bin/env python
"""Verify that an offline MNIST (or Fashion-MNIST) setup is correct.

Reads a small subset and prints diagnostics. Use this before running
real experiments to make sure your local files are in the right place.

Usage:
    python scripts/verify_dataset.py --dataset mnist --root ./data
    python scripts/verify_dataset.py --dataset mnist --root ./data --n 10 --save-thumbs

Exit codes:
    0 = OK   1 = files missing   2 = file parse error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from pr_uq.data.datasets import make_mnist_offline, _find_idx_pair, _read_idx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist"])
    p.add_argument("--root", default="./data")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n", type=int, default=5, help="Number of images to load as a check.")
    p.add_argument("--save-thumbs", action="store_true",
                   help="Save N PNG thumbnails to ./data/_verify_thumbs/ for visual inspection.")
    args = p.parse_args()

    # 1) Find the files
    print(f"[verify] dataset={args.dataset} split={args.split} root={args.root}")
    try:
        imgs_path, lbls_path = _find_idx_pair(args.root, args.split)
        print(f"[verify] OK images: {imgs_path}")
        print(f"[verify] OK labels: {lbls_path}")
    except FileNotFoundError as e:
        print(f"[verify] FAIL — files not found:\n{e}")
        return 1

    # 2) Parse the headers
    try:
        images = _read_idx(imgs_path)
        labels = _read_idx(lbls_path)
        print(f"[verify] OK parsed images shape={images.shape} dtype={images.dtype}")
        print(f"[verify] OK parsed labels shape={labels.shape} dtype={labels.dtype}")
    except Exception as e:
        print(f"[verify] FAIL — parse error: {e}")
        return 2

    # 3) Sanity checks on content
    n_expected = {"train": 60000, "test": 10000}.get(args.split, None)
    if n_expected is not None and images.shape[0] != n_expected:
        print(f"[verify] WARN — expected {n_expected} images for split={args.split}, "
              f"got {images.shape[0]}. Continuing anyway.")
    if images.shape[1:] != (28, 28):
        print(f"[verify] WARN — images are not 28x28: {images.shape[1:]}")
    if images.dtype != np.uint8 or images.min() < 0 or images.max() > 255:
        print(f"[verify] WARN — images not uint8 in [0, 255]: "
              f"dtype={images.dtype} min={images.min()} max={images.max()}")
    n_labels_unique = len(set(labels.tolist()[:1000]))
    print(f"[verify] OK first 1000 labels span {n_labels_unique} unique classes")

    # 4) Load through the high-level API
    samples = make_mnist_offline(n_images=args.n, size=32, root=args.root, split=args.split, seed=0)
    print(f"[verify] OK make_mnist_offline returned {len(samples)} samples")
    for s in samples:
        print(f"    {s.name}  label={s.label}  shape={tuple(s.image.shape)}  "
              f"range=[{s.image.min():.3f}, {s.image.max():.3f}]")

    # 5) Optionally write thumbnails
    if args.save_thumbs:
        try:
            from PIL import Image
        except ImportError:
            print("[verify] PIL not available; skipping thumbnails.")
        else:
            out = Path(args.root) / "_verify_thumbs"
            out.mkdir(parents=True, exist_ok=True)
            for s in samples:
                arr = (s.image.squeeze().numpy() * 255).clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(out / f"{s.name}.png")
            print(f"[verify] OK wrote thumbnails to {out}/")

    print("[verify] ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
