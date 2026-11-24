#!/usr/bin/env python
"""Report of the real-nonnegative test of the third study (STUDY3_REALNN_PLAN.md).

Our solver with a real and nonnegative object (trainer.object_constraint = real_nonneg, results in
results/study3_realnn/real_oursnn_<level>) against the stored Study 3 runs on the same images and
measurements: our solver with the complex object model (results/study3/real_ours_<level>), HIO with
the real and nonnegative constraint, best of the first 81 starts by FE (results/study3/real_hio81_<level>),
and Net-ADM (results/study3/real_netadm_<level>, descriptive only).

PSNR as stored by the runners (aligned real image), capped at 60 dB; below 15 dB is a failure.
FE is the error against the noisy measurement (member_fe of the runner).

Statistics as in scripts/study3_report.py: paired mean with t-based 95% CI, two-sided Wilcoxon
signed-rank test, Holm adjustment over R1 to R6, a claim only if the Holm-adjusted p < 0.05 and the
CI excludes zero, direction given by the sign of the mean.

    python scripts/study3_realnn_report.py --study3 results/study3 --realnn results/study3_realnn --out results/study3_realnn/report

Check of the script before the runs: --selftest uses the stored complex-model run in place of the
real-nonnegative run. R1 and R2 are then zero on every image, R3 to R6 reproduce N7, N8, N1 and N2.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("study3_report", HERE / "study3_report.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

LEVELS = ("moderate", "strong")
N_IMAGES = 50
K_HIO = s3.K_EQ["real_ours"]  # 81, fixed in STUDY3_PLAN.md
N_REPEAT = 5


def stacks(run_dir: Path, i: int):
    """Stored members.npz of one image, or None."""
    f = list((run_dir / "per_image").glob(f"image_{i:05d}/members.npz"))
    return np.load(f[0]) if f else None


def same_measurements(da: Path, db: Path, ids) -> int | None:
    """Images whose stored measurement and ground truth are identical in both runs."""
    n = 0
    for i in ids:
        za, zb = stacks(da, i), stacks(db, i)
        if za is None or zb is None:
            return None
        keys = [k for k in za.files if k == "measurement" or k.startswith("gt")]
        n += int("measurement" in keys and all(k in zb.files and np.array_equal(za[k], zb[k]) for k in keys))
    return n


def object_check(run_dir: Path, ids) -> dict:
    """Largest |imaginary part| and most negative real part over all stored raw starts."""
    imag, neg = 0.0, 0.0
    for i in ids:
        z = stacks(run_dir, i)
        if z is None or "stack_raw__imag" not in z.files:
            return {"max_abs_imag": None, "min_real": None}
        imag = max(imag, float(np.abs(z["stack_raw__imag"]).max()))
        neg = min(neg, float(z["stack_raw__real"].min()))
    return {"max_abs_imag": imag, "min_real": neg}


def repeat_check(stored: dict, rerun: dict) -> dict:
    """Rerun of the frozen run on its first images: largest differences of the per-start values."""
    ids = sorted(set(stored) & set(rerun))
    if not ids:
        return {"images": 0}
    d_psnr = [np.abs(np.asarray(rerun[i]["member_psnr"], float) - np.asarray(stored[i]["member_psnr"], float)).max()
              for i in ids]
    d_fe = [np.abs(np.asarray(rerun[i]["member_fe"], float) - np.asarray(stored[i]["member_fe"], float)).max()
            for i in ids]
    same_sel = sum(int(rerun[i]["best_single_index"] == stored[i]["best_single_index"]) for i in ids)
    d_fs = [abs(rerun[i]["methods"]["fe_softmax"]["psnr"] - stored[i]["methods"]["fe_softmax"]["psnr"]) for i in ids]
    return {"images": len(ids), "max_abs_diff_member_psnr_db": float(max(d_psnr)),
            "max_abs_diff_member_fe": float(max(d_fe)), "same_selected_start": same_sel,
            "max_abs_diff_fe_softmax_psnr_db": float(max(d_fs))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study3", required=True)
    ap.add_argument("--realnn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true", help="use the stored complex-model run as the real-nonnegative run")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s3dir, nndir = Path(a.study3), Path(a.realnn)

    tests, desc, missing = [], {}, []
    for j, lv in enumerate(LEVELS):
        ours = s3.load_run(s3dir / f"real_ours_{lv}")
        hio = s3.load_run(s3dir / f"real_hio81_{lv}")
        netadm = s3.load_run(s3dir / f"real_netadm_{lv}")
        nn_dir = s3dir / f"real_ours_{lv}" if a.selftest else nndir / f"real_oursnn_{lv}"
        nn = s3.load_run(nn_dir)
        ids = sorted(set(ours) & set(hio) & set(nn) & set(netadm))
        if len(ids) != N_IMAGES:
            missing.append(f"{lv}: {len(ids)} of {N_IMAGES} images")
            continue
        agg = lambda run, key: s3.cap([run[i]["methods"][key]["psnr"] for i in ids])
        fs_nn, k1_nn, sel_nn = agg(nn, "fe_softmax"), agg(nn, "k1"), agg(nn, "best_single_by_fe")
        fs_ours, fs_net = agg(ours, "fe_softmax"), agg(netadm, "fe_softmax")
        hio_best = s3.best_of_first(hio, ids, "hio_realpos", K_HIO)[0]
        tests.append({"id": f"R{1 + j}", "level": lv, "comparison": "real-nonnegative FE-softmax - complex-model FE-softmax",
                      **s3.paired(fs_nn, fs_ours)})
        tests.append({"id": f"R{3 + j}", "level": lv, "comparison": f"real-nonnegative FE-softmax - HIO real+nonneg best of {K_HIO}",
                      **s3.paired(fs_nn, hio_best)})
        tests.append({"id": f"R{5 + j}", "level": lv, "comparison": "real-nonnegative FE-softmax - real-nonnegative K=1",
                      **s3.paired(fs_nn, k1_nn)})
        fe_nn = np.array([np.median(nn[i]["member_fe"]) for i in ids], dtype=float)
        fe_ours = np.array([np.median(ours[i]["member_fe"]) for i in ids], dtype=float)
        desc[lv] = {"real-nonnegative K=1": s3.describe(k1_nn), "real-nonnegative selection": s3.describe(sel_nn),
                    "real-nonnegative FE-softmax": s3.describe(fs_nn), "complex-model FE-softmax": s3.describe(fs_ours),
                    f"HIO best of {K_HIO}": s3.describe(hio_best), "Net-ADM FE-softmax": s3.describe(fs_net),
                    "real-nonnegative FE-softmax - Net-ADM FE-softmax": s3.paired(fs_nn, fs_net),
                    "median start FE real-nonnegative": float(np.median(fe_nn)),
                    "median start FE complex model": float(np.median(fe_ours)),
                    "s per image real-nonnegative": float(np.nanmean([nn[i].get("runtime_total_s", np.nan) for i in ids])),
                    "s per image complex model (Study 3)": float(np.nanmean([ours[i].get("runtime_total_s", np.nan) for i in ids])),
                    "images with identical measurement": same_measurements(s3dir / f"real_ours_{lv}", nn_dir, ids),
                    "object check": object_check(nn_dir, ids)}
    if missing:
        (out / "report.md").write_text("Incomplete runs:\n" + "\n".join(missing) + "\n", encoding="utf-8")
        print("incomplete:", missing)
        return 1
    tests.sort(key=lambda t: int(t["id"][1:]))
    p_holm = s3.holm(np.array([t["p_wilcoxon"] for t in tests]))
    for t, ph in zip(tests, p_holm):
        t["p_holm"] = float(ph)
        t["decision"] = s3.decide(t, ph)

    stored = s3.load_run(s3dir / "real_ours_moderate")
    rep_dir = s3dir / "real_ours_moderate" if a.selftest else nndir / "repeat_real_ours_moderate"
    rerun = s3.load_run(rep_dir) if rep_dir.exists() else {}
    repeat = repeat_check({i: stored[i] for i in sorted(stored)[:N_REPEAT]}, rerun)

    lines = ["# Study 3, real and nonnegative object for our solver: report"
             + (" (SELFTEST, the complex-model run stands in for the real-nonnegative run)" if a.selftest else ""), "",
             "PSNR in dB of the aligned real image, capped at 60 dB, 50 images per level. Holm adjustment over R1 to R6.", "",
             "| id | level | comparison | mean | 95% CI | median | first/second better | p | p (Holm) | decision |",
             "|---|---|---|--:|---|--:|--:|--:|--:|---|"]
    for t in tests:
        lines.append(f"| {t['id']} | {t['level']} | {t['comparison']} | {t['mean']:+.2f} | [{t['ci'][0]:.2f}, {t['ci'][1]:.2f}] | "
                     f"{t['median']:+.2f} | {t['first_better']}/{t['second_better']} | {t['p_wilcoxon']:.2g} | {t['p_holm']:.2g} | {t['decision']} |")
    lines += ["", "## Descriptive", "",
              "| level | method | mean | SD | median | failures |", "|---|---|--:|--:|--:|--:|"]
    for lv, d in desc.items():
        for k, v in d.items():
            if isinstance(v, dict) and "sd" in v:
                lines.append(f"| {lv} | {k} | {v['mean']:.2f} | {v['sd']:.2f} | {v['median']:.2f} | {v['failures']} |")
    lines += ["", "| level | real-nonneg FE-softmax - Net-ADM FE-softmax [95% CI] | median start FE real-nonneg / complex | "
              "s per image real-nonneg / complex (Study 3) | identical measurements | max abs imag / min real of the stored starts |",
              "|---|---|---|---|---|---|"]
    for lv, d in desc.items():
        p = d["real-nonnegative FE-softmax - Net-ADM FE-softmax"]
        oc = d["object check"]
        lines.append(f"| {lv} | {p['mean']:+.2f} [{p['ci'][0]:.2f}, {p['ci'][1]:.2f}] | "
                     f"{d['median start FE real-nonnegative']:.3f} / {d['median start FE complex model']:.3f} | "
                     f"{d['s per image real-nonnegative']:.0f} / {d['s per image complex model (Study 3)']:.0f} | "
                     f"{d['images with identical measurement'] if d['images with identical measurement'] is not None else 'n/a'} | "
                     f"{oc['max_abs_imag']} / {oc['min_real']} |")
    lines += ["", "## Repeat of the frozen run on ranks 0 to 4", "", "```", json.dumps(repeat, indent=2), "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"tests": tests, "descriptive": desc, "repeat": repeat,
                                                  "selftest": a.selftest}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
