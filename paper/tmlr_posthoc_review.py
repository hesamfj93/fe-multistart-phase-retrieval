#!/usr/bin/env python
"""Post hoc checks for the TMLR manuscript, from the stored per-image results only (no new runs).

1. N13 of Study 3 recomputed with the FE against the measurement (paper/tmlr_fe.py); the frozen
   report computed it with the FE against the noise-free magnitudes. Same bootstrap as the frozen
   script (10,000 resamples, its seed BOOT_SEED, runs in the same order).
2. FE-softmax against the mean PSNR of the five starts (the expected PSNR of one random start),
   in addition to start 0 (K = 1).
3. Does FE rank images better than simpler references?
   - the FE of start 0 against the PSNR of start 0;
   - a quantity computed from the measurement alone: the area of the autocorrelation
     |IFFT(y)| above 2% of its maximum (5% gives partial correlations within 0.01);
   - the partial Spearman correlation of FE with -PSNR given that area.
4. Study 3 at equal numbers of starts: FE-softmax of five untrained starts against the lowest-FE
   start among the first five classical starts (and among the first one).

    python paper/tmlr_posthoc_review.py
Writes paper/tmlr_posthoc_review.json and three row files in paper/tmlr_supp.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

from tmlr_fe import REPO, load_with_measured_fe

_spec = importlib.util.spec_from_file_location("s3", REPO / "scripts/study3_report.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

CAP = 60.0
RUNS = {
    "Study 1, real, ours": "results/stage2_mnist_heldout",
    "Study 2, complex, ours": "results/stage3_complex_heldout",
    "Study 3, real, moderate, ours": "results/study3/real_ours_moderate",
    "Study 3, real, strong, ours": "results/study3/real_ours_strong",
    "Study 3, complex, moderate, ours": "results/study3/complex_ours_moderate",
    "Study 3, complex, strong, ours": "results/study3/complex_ours_strong",
    "Study 3, real, moderate, Net-ADM": "results/study3/real_netadm_moderate",
    "Study 3, real, strong, Net-ADM": "results/study3/real_netadm_strong",
}
EQUAL_STARTS = [  # (label, untrained run, classical run, classical key)
    ("ours, real, moderate", "real_ours_moderate", "real_hio81_moderate", "hio_realpos"),
    ("ours, real, strong", "real_ours_strong", "real_hio81_strong", "hio_realpos"),
    ("ours, complex, moderate", "complex_ours_moderate", "complex_raar50_moderate", "raar_sw"),
    ("ours, complex, strong", "complex_ours_strong", "complex_raar50_strong", "raar_sw"),
    ("Net-ADM, real, moderate", "real_netadm_moderate", "real_hio81_moderate", "hio_realpos"),
    ("Net-ADM, real, strong", "real_netadm_strong", "real_hio81_strong", "hio_realpos"),
]


def cap(x):
    return np.minimum(np.asarray(x, dtype=float), CAP)


def paired(a, b):
    d = a - b
    n = len(d)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)],
            "first_better": int((d > 0).sum()), "second_better": int((d < 0).sum()),
            "p_wilcoxon": float(stats.wilcoxon(a, b).pvalue)}


def autocorr_area(path: str, frac: float) -> float:
    y = np.asarray(np.load(path)["measurement"], dtype=float).squeeze()
    a = np.abs(np.fft.ifft2(np.fft.ifftshift(y)))
    return float((a > frac * a.max()).sum())


def partial_spearman(x, y, z) -> float:
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)  # noqa: E731
    return float(stats.pearsonr(res(rx), res(ry))[0])


def main() -> None:
    out = {"runs": {}, "equal_starts": {}}
    rng = np.random.default_rng(s3.BOOT_SEED)  # used for the Study 3 runs in the order of the frozen report
    rows_ms, rows_rel = [], []
    for name, run in RUNS.items():
        ms = load_with_measured_fe(REPO / run)
        fs = cap([m["methods"]["fe_softmax"]["psnr"] for m in ms])
        k1 = cap([m["methods"]["k1"]["psnr"] for m in ms])
        mean_start = np.array([cap(m["member_psnr"]).mean() for m in ms])
        fe_fs = np.array([m["fe_softmax_measured"] for m in ms])
        fe_stored = np.array([m["methods"]["fe_softmax"]["fe"] for m in ms])
        fe_k1 = np.array([m["member_fe"][0] for m in ms])
        area2 = np.array([autocorr_area(m["measurement_path"], 0.02) for m in ms])
        area5 = np.array([autocorr_area(m["measurement_path"], 0.05) for m in ms])
        r = {"n": len(ms),
             "fesoftmax_minus_mean_start": paired(fs, mean_start),
             "k1_minus_mean_start": float((k1 - mean_start).mean()),
             "max_abs_stored_minus_measured_fe": float(np.abs(fe_stored - fe_fs).max()),
             "rho_fe_fesoftmax": float(stats.spearmanr(fe_fs, -fs)[0]),
             "rho_fe_start0": float(stats.spearmanr(fe_k1, -k1)[0]),
             "rho_area2": float(stats.spearmanr(area2, -fs)[0]),
             "partial_rho_fe_given_area2": partial_spearman(fe_fs, -fs, area2),
             "partial_rho_fe_given_area5": partial_spearman(fe_fs, -fs, area5)}
        if name.startswith("Study 3"):
            r["N13_measured_fe"] = s3.spearman_fe(fe_fs, fs, rng)
            r["N13_frozen"] = json.loads((REPO / "results/study3/report/summary.json").read_text())["fe_signal_N13"][Path(run).name]
        out["runs"][name] = r
        d = r["fesoftmax_minus_mean_start"]
        rows_ms.append(f"{name} & {r['n']} & {mean_start.mean():.2f} & {k1.mean():.2f} & {fs.mean():.2f} & "
                       f"${d['mean']:+.2f}$ & [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}] & {d['first_better']}/{d['second_better']} \\\\")
        n13 = (f"{r['N13_frozen']['rho']:.2f} & {r['N13_measured_fe']['rho']:.2f} [{r['N13_measured_fe']['ci95'][0]:.2f}, "
               f"{r['N13_measured_fe']['ci95'][1]:.2f}]") if "N13_frozen" in r else f"-- & {r['rho_fe_fesoftmax']:.2f}"
        rows_rel.append(f"{name} & {n13} & {r['rho_fe_start0']:.2f} & {r['rho_area2']:.2f} & {r['partial_rho_fe_given_area2']:.2f} \\\\")
        print(f"{name:34s} fs-mean {d['mean']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] | rho fs {r['rho_fe_fesoftmax']:.2f} "
              f"start0 {r['rho_fe_start0']:.2f} area {r['rho_area2']:.2f} partial {r['partial_rho_fe_given_area2']:.2f}"
              + (f" | N13 frozen {r['N13_frozen']['rho']:.2f} measured {r['N13_measured_fe']['rho']:.2f} p {r['N13_measured_fe']['p_one_sided']:.1e}" if "N13_frozen" in r else ""))

    rows_eq = []
    for label, ur, cr, key in EQUAL_STARTS:
        u, c = s3.load_run(REPO / "results/study3" / ur), s3.load_run(REPO / "results/study3" / cr)
        ids = sorted(set(u) & set(c))
        fs = cap([u[i]["methods"]["fe_softmax"]["psnr"] for i in ids])
        b5 = s3.best_of_first(c, ids, key, 5)[0]
        b1 = s3.best_of_first(c, ids, key, 1)[0]
        k1 = cap([u[i]["methods"]["k1"]["psnr"] for i in ids])
        d = paired(fs, b5)
        out["equal_starts"][label] = {"n": len(ids), "classical": key, "fesoftmax": float(fs.mean()), "k1": float(k1.mean()),
                                      "classical_best_of_5": float(b5.mean()), "classical_start_0": float(b1.mean()),
                                      "fesoftmax_minus_classical_best_of_5": d,
                                      "k1_minus_classical_start_0": paired(k1, b1)}
        cls = "HIO r+n" if key == "hio_realpos" else "RAAR sw"
        rows_eq.append(f"{label} & {cls} & {fs.mean():.2f} & {b5.mean():.2f} & ${d['mean']:+.2f}$ & [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}] & "
                       f"{d['first_better']}/{d['second_better']} & {k1.mean():.2f} & {b1.mean():.2f} \\\\")
        print(f"{label:26s} FE-softmax {fs.mean():.2f} vs {key} best of 5 {b5.mean():.2f}: {d['mean']:+.2f} "
              f"[{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] {d['first_better']}/{d['second_better']} | K=1 {k1.mean():.2f} vs start 0 {b1.mean():.2f}")

    (REPO / "paper/tmlr_posthoc_review.json").write_text(json.dumps(out, indent=2))
    sd = REPO / "paper/tmlr_supp"
    sd.mkdir(exist_ok=True)
    (sd / "tab_posthoc_meanstart.tex").write_text("\n".join(rows_ms) + "\n")
    (sd / "tab_posthoc_reliability.tex").write_text("\n".join(rows_rel) + "\n")
    (sd / "tab_posthoc_equalstarts.tex").write_text("\n".join(rows_eq) + "\n")


if __name__ == "__main__":
    main()
