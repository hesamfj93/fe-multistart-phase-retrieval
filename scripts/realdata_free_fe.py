#!/usr/bin/env python
"""Post hoc, exploratory: a free FE on the measured CXIDB 28 pattern.

Not part of REALDATA_PLAN.md. Written after the planned runs, after the published two-cluster
structure was known to disagree with the lowest-FE RAAR start, and committed before its runs.

A set of measured pixels is withheld: their intensities are set to zero and they are treated as
unmeasured during reconstruction, like the pixels under the beamstop. Each start is then scored
by its FE on the remaining measured pixels (training FE) and on the withheld pixels (free FE),
in the spirit of the free R factor of crystallography. Centrosymmetric partners are withheld
together, so a nearly real object cannot carry a withheld pixel over from its mirror.

Two withheld sets, each run separately:
  ring   - every measured pixel within 3 pixels (Chebyshev distance) of an unmeasured pixel of
           the central beamstop region; tests whether solutions differ in how they continue into
           the low frequencies that the beamstop hides;
  random - 5% of the other measured pixels, drawn with seed 0; a general check of overfitting.

For each set: 50 RAAR starts with shrinkwrap and the settings of the planned run (5000 + 200
iterations, beta 0.87, seeds 0 to 49). Each start is classified as two clusters or streak by the
criterion of results/realdata/exploratory/README.md (at least 20% of the rows between the 0.5% and
99.5% quantiles of the vertical energy profile below 1% of the largest row energy).

    python scripts/realdata_free_fe.py --set ring --out results/realdata/exploratory/free_fe
    python scripts/realdata_free_fe.py --set random --out results/realdata/exploratory/free_fe
    python scripts/realdata_free_fe.py --summarize --out results/realdata/exploratory/free_fe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pr_uq.baselines.oversampled import run_raar_full  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402

FIELD, OVERSAMPLE, N_STARTS = 128, 2, 50
RAAR = dict(iters=5000, er_iters=200, beta=0.87, oversample=OVERSAMPLE, constraint="shrinkwrap")


def mirror(m: np.ndarray) -> np.ndarray:
    """Centrosymmetric partner with zero frequency at (128, 128): (i, j) -> (256 - i, 256 - j)."""
    return np.roll(np.roll(m[::-1, ::-1], 1, axis=0), 1, axis=1)


def free_set(mask: np.ndarray, which: str) -> np.ndarray:
    lab, _ = ndimage.label(~mask)
    beamstop = lab == lab[FIELD, FIELD]
    if which == "ring":
        near = ndimage.binary_dilation(beamstop, structure=np.ones((7, 7), bool))
        free = near & mask
    else:
        ring = ndimage.binary_dilation(beamstop, structure=np.ones((7, 7), bool)) & mask
        cand = np.argwhere(mask & ~ring)
        rng = np.random.default_rng(0)
        pick = cand[rng.choice(len(cand), size=int(round(0.05 * len(cand))), replace=False)]
        free = np.zeros_like(mask)
        free[pick[:, 0], pick[:, 1]] = True
    free = (free | mirror(free)) & mask
    return free


def gap_share(a2: np.ndarray) -> float:
    p = a2.sum(-1)
    lo, hi = np.searchsorted(np.cumsum(p) / p.sum(), [0.005, 0.995])
    seg = p[lo:hi + 1]
    return float((seg < 0.01 * seg.max()).mean())


def run(which: str, out: Path) -> None:
    torch.set_num_threads(1)
    z = np.load(REPO / "results/realdata/cxidb28_classical/measurement.npz")
    y, mask = z["y"].astype(np.float32), z["mask"].astype(bool)
    free = free_set(mask, which)
    train = mask & ~free
    yt = torch.from_numpy(np.where(train, y, 0)).view(1, 1, 256, 256)
    mt = torch.from_numpy(train).view(1, 1, 256, 256)
    yf = torch.from_numpy(y).view(1, 1, 256, 256)
    mf = torch.from_numpy(free).view(1, 1, 256, 256)
    d = out / which
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(N_STARTS):
        f = d / f"start_{k:02d}.npz"
        if not f.exists():
            t = time.perf_counter()
            r = run_raar_full(yt, (FIELD, FIELD), seed=k, device="cpu", mask=mt, **RAAR)
            x = r.recon.numpy()[0, 0]
            np.savez_compressed(f, real=x.real, imag=x.imag, runtime_s=time.perf_counter() - t)
        s = np.load(f)
        x = torch.from_numpy(s["real"] + 1j * s["imag"]).to(torch.complex64).view(1, 1, FIELD, FIELD)
        rows.append({"seed": k,
                     "fe_train": float(member_fourier_errors(x, yt, OVERSAMPLE, mask=mt)[0]),
                     "fe_free": float(member_fourier_errors(x, yf, OVERSAMPLE, mask=mf)[0]),
                     "two_clusters": gap_share(s["real"] ** 2 + s["imag"] ** 2) >= 0.2})
        print(which, rows[-1], flush=True)
    info = {"set": which, "n_free_pixels": int(free.sum()), "n_train_pixels": int(train.sum()),
            "free_share_of_measured_intensity": float(y[free].sum() / y[mask].sum()), "starts": rows}
    (d / "starts.json").write_text(json.dumps(info, indent=2))


def auroc(score, pos):
    s1, s0 = score[pos], score[~pos]
    if len(s1) == 0 or len(s0) == 0:
        return None
    return float(((s1[:, None] > s0[None, :]).sum() + 0.5 * (s1[:, None] == s0[None, :]).sum()) / (len(s1) * len(s0)))


def summarize(out: Path) -> None:
    lines = ["# Free FE on the measured pattern (post hoc, exploratory)", "",
             "Not part of REALDATA_PLAN.md; see the docstring of scripts/realdata_free_fe.py. 50 RAAR starts with shrinkwrap per withheld set.", "",
             "| withheld set | free pixels | share of measured intensity | two-cluster starts | FE train, two clusters | FE train, streak | FE free, two clusters | FE free, streak | lowest FE train | lowest FE free | AUROC of FE free for streak |",
             "|---|--:|--:|--:|---|---|---|---|---|---|--:|"]
    summ = {}
    for which in ("ring", "random"):
        f = out / which / "starts.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text())
        st = info["starts"]
        two = np.array([s["two_clusters"] for s in st])
        ft = np.array([s["fe_train"] for s in st])
        ff = np.array([s["fe_free"] for s in st])
        rng_ = lambda v: f"{v.min():.3f} to {v.max():.3f} (median {np.median(v):.3f})" if len(v) else "none"  # noqa: E731
        kind = lambda i: "two clusters" if two[i] else "streak"  # noqa: E731
        a = auroc(ff, ~two)
        summ[which] = {"n_free_pixels": info["n_free_pixels"], "free_share": info["free_share_of_measured_intensity"],
                       "n_two_clusters": int(two.sum()), "lowest_fe_train": kind(int(ft.argmin())),
                       "lowest_fe_free": kind(int(ff.argmin())), "auroc_free_fe_streak": a,
                       "fe_train_two": ft[two].tolist(), "fe_train_streak": ft[~two].tolist(),
                       "fe_free_two": ff[two].tolist(), "fe_free_streak": ff[~two].tolist()}
        lines.append(f"| {which} | {info['n_free_pixels']} | {info['free_share_of_measured_intensity']:.3f} | {int(two.sum())} of {len(st)} | "
                     f"{rng_(ft[two])} | {rng_(ft[~two])} | {rng_(ff[two])} | {rng_(ff[~two])} | {kind(int(ft.argmin()))} | "
                     f"{kind(int(ff.argmin()))} | {'n/a' if a is None else f'{a:.2f}'} |")
    lines += ["", "AUROC: probability that a streak start has a higher free FE than a two-cluster start (1 = free FE always ranks the two-cluster solutions first)."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=("ring", "random"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize(Path(a.out))
    else:
        run(a.set, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
