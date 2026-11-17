#!/usr/bin/env python
"""Preprocess the CXIDB entry 28 diffraction pattern (Marchesini et al., Phys. Rev. B 68, 140101, 2003).

Uses only the measured data, no reconstruction:
1. The stored values are magnitudes (square roots of intensity). Unmeasured pixels are stored as 0
   (beamstop and its wire, a 40-column strip at the left edge, a few dead pixels).
2. Center: the point c on a half-pixel grid that minimizes the relative centrosymmetry error
   sum |a(c+r) - a(c-r)|^2 / sum a(c+r)^2 over r in a 424 x 424 window, measured pixels only.
3. Crop C x C around c and bin B x B (sum of intensities), default C = 512, B = 2. The crop is placed
   so that c falls at the center of binned pixel (n/2, n/2), n = C/B, the zero frequency of the
   centered n x n FFT used by the solvers. A binned pixel is measured only if its B^2 detector pixels
   are measured.
4. Geometry (from the file): 105 mm detector distance, 24 um pixels, wavelength from the photon
   energy (about 2.1 nm). One pixel of the full 1024 x 1024 pattern is about 9 nm in the object plane.
   Cropping to C keeps the field of view and coarsens the object pixel by 1024 / C; binning by B
   halves the field of view for B = 2. With the defaults the object pixel is about 18 nm, the field
   of the 256 x 256 pattern about 4.6 um, and the object (two clusters of 50 nm gold balls, about
   2.1 x 0.6 um from the autocorrelation) fits in the central 128 x 128 field, the solver field with
   oversampling 2. A crop of 128 is not usable: the beamstop then covers 84% of the pixels within
   half the Nyquist radius.

Writes <out>/cxidb28_<n>.npz (magnitude, mask) and <out>/preprocess.json.

    python -I scripts/cxidb28_preprocess.py --cxi <path>/cxidb-28.cxi --out data/cxidb28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
WINDOW = 212  # half-size of the centrosymmetry window


def load(path: Path):
    import h5py
    with h5py.File(path, "r") as f:
        a = f["entry_1/instrument_1/detector_1/data"][()].astype(np.float64)
        geom = {}
        for key, ds in (("energy_J", "entry_1/instrument_1/source_1/energy"),
                        ("distance_m", "entry_1/instrument_1/detector_1/distance"),
                        ("pixel_m", "entry_1/instrument_1/detector_1/x_pixel_size")):
            try:
                geom[key] = float(f[ds][()])
            except KeyError:
                geom[key] = None
    geom["wavelength_m"] = H_PLANCK * C_LIGHT / geom["energy_J"] if geom.get("energy_J") else None
    return a, geom


def centrosymmetry_error(a: np.ndarray, m: np.ndarray, cy: float, cx: float) -> float:
    sy, sx = int(round(2 * cy)), int(round(2 * cx))
    c0y, c0x = int(cy), int(cx)
    ys = np.arange(c0y - WINDOW, c0y + WINDOW)
    xs = np.arange(c0x - WINDOW, c0x + WINDOW)
    p = a[np.ix_(ys, xs)]
    q = a[np.ix_(sy - ys, sx - xs)]
    w = m[np.ix_(ys, xs)] & m[np.ix_(sy - ys, sx - xs)]
    return float(np.sum(w * (p - q) ** 2) / np.sum(w * p ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cxi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--bin", type=int, default=2)
    a_ = ap.parse_args()
    CROP, BIN = a_.crop, a_.bin
    nb = CROP // BIN
    dc = nb // 2
    a, geom = load(Path(a_.cxi))
    n = a.shape[0]
    m = a > 0
    grid = np.arange(n // 2 - 12, n // 2 + 8.5, 0.5)
    errs = {(cy, cx): centrosymmetry_error(a, m, cy, cx) for cy in grid for cx in grid}
    (cy, cx), err = min(errs.items(), key=lambda kv: kv[1])
    # crop so that the center lies at the middle of binned pixel dc: crop index BIN*dc + (BIN-1)/2
    off = BIN * dc + (BIN - 1) / 2
    y0, x0 = int(round(cy - off)), int(round(cx - off))
    assert abs((cy - y0) - off) < 1e-9 and abs((cx - x0) - off) < 1e-9, "center not on the binned grid"
    assert y0 >= 0 and x0 >= 0 and y0 + CROP <= n and x0 + CROP <= n
    I = (a ** 2)[y0:y0 + CROP, x0:x0 + CROP]
    mk = m[y0:y0 + CROP, x0:x0 + CROP]
    Ib = I.reshape(CROP // BIN, BIN, CROP // BIN, BIN).sum(axis=(1, 3))
    mb = mk.reshape(CROP // BIN, BIN, CROP // BIN, BIN).all(axis=(1, 3))
    Ib[~mb] = 0.0
    amp = np.sqrt(Ib)
    # centrosymmetry of the binned pattern about (dc, dc): pixel (dc+i, dc+j) against (dc-i, dc-j)
    idx = np.arange(1, nb)
    P = amp[np.ix_(idx, idx)]
    Q = amp[np.ix_(2 * dc - idx, 2 * dc - idx)]
    W = mb[np.ix_(idx, idx)] & mb[np.ix_(2 * dc - idx, 2 * dc - idx)]
    yy, xx = np.indices(mb.shape)
    rr = np.hypot(yy - dc, xx - dc)
    err_b = float(np.sum(W * (P - Q) ** 2) / np.sum(W * P ** 2))
    pix_obj_full = (geom["wavelength_m"] * geom["distance_m"] / (n * geom["pixel_m"])
                    if all(v is not None for v in geom.values()) else None)
    out = Path(a_.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"cxidb28_{nb}.npz", magnitude=amp.astype(np.float32), mask=mb)
    info = {
        "source": "CXIDB entry 28, Marchesini et al., Phys. Rev. B 68, 140101 (2003), CC0",
        "geometry_from_file": geom,
        "object_pixel_full_pattern_nm": None if pix_obj_full is None else pix_obj_full * 1e9,
        "object_pixel_after_crop_nm": None if pix_obj_full is None else pix_obj_full * 1e9 * n / CROP,
        "unmeasured_pixels_full_pattern": int((~m).sum()),
        "center_yx": [cy, cx],
        "centrosymmetry_error_full": err,
        "crop_rows": [y0, y0 + CROP - 1], "crop_cols": [x0, x0 + CROP - 1],
        "crop": CROP, "bin": BIN, "binned_shape": [nb, nb], "dc_pixel": [dc, dc],
        "object_field_with_oversampling_2_px": nb // 2,
        "unmeasured_pixels_binned": int((~mb).sum()),
        "unmeasured_fraction_within_half_nyquist": float(np.mean(~mb[rr < nb / 4])),
        "centrosymmetry_error_binned": err_b,
        "sum_measured_intensity_binned": float(Ib.sum()),
    }
    (out / "preprocess.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
