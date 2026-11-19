#!/usr/bin/env python
"""Report of the compute-matched control of the third study (STUDY3_LONG_PLAN.md).

Compares five FE-guided starts of the stored Study 3 runs (results/study3/<run>) with one start
of five times the iterations (results/study3_long/<run>, method k1_long), on the same images and
measurements. PSNR as stored by the runner, capped at 60 dB; below 15 dB is a failure. FE is the
error against the noisy measurement (member_fe of the runner).

Statistics as in scripts/study3_report.py: paired mean with t-based 95% CI, two-sided Wilcoxon
signed-rank test, Holm adjustment over the twelve comparisons L1-L12, a claim only if the
Holm-adjusted p < 0.05 and the CI excludes zero, direction given by the sign of the mean.

    python scripts/study3_long_report.py --study3 results/study3 --long results/study3_long --out results/study3_long/report

Check of the script before the runs: --selftest uses the stored K=1 start in place of k1_long.
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

RUNS = ("real_ours_moderate", "real_ours_strong", "complex_ours_moderate", "complex_ours_strong",
        "real_netadm_moderate", "real_netadm_strong")
N_IMAGES = 50


def same_measurements(d5: Path, dl: Path, ids) -> int | None:
    """Images whose stored measurement and ground truth are identical in both runs (members.npz)."""
    n = 0
    for i in ids:
        f5 = list((d5 / "per_image").glob(f"image_{i:05d}/members.npz"))
        fl = list((dl / "per_image").glob(f"image_{i:05d}/members.npz"))
        if not f5 or not fl:
            return None
        z5, zl = np.load(f5[0]), np.load(fl[0])
        keys = [k for k in z5.files if k == "measurement" or k.startswith("gt")]
        n += int("measurement" in keys and all(k in zl.files and np.array_equal(z5[k], zl[k]) for k in keys))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study3", required=True)
    ap.add_argument("--long", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true", help="use the stored K=1 start as k1_long")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tests, desc, missing = [], {}, []
    for j, run in enumerate(RUNS):
        five = s3.load_run(Path(a.study3) / run)
        if a.selftest:
            long_ = {i: {"methods": {"k1_long": m["methods"]["k1"]}, "member_fe": m["member_fe"][:1],
                         "runtime_total_s": np.nan} for i, m in five.items()}
        else:
            long_ = s3.load_run(Path(a.long) / run)
        ids = sorted(set(five) & set(long_))
        if len(ids) != N_IMAGES:
            missing.append(f"{run}: {len(ids)} of {N_IMAGES} images")
            continue
        fs = s3.cap([five[i]["methods"]["fe_softmax"]["psnr"] for i in ids])
        sel = s3.cap([five[i]["methods"]["best_single_by_fe"]["psnr"] for i in ids])
        k1 = s3.cap([five[i]["methods"]["k1"]["psnr"] for i in ids])
        kl = s3.cap([long_[i]["methods"]["k1_long"]["psnr"] for i in ids])
        # FE against the measurement, as the runner computes it for every start (member_fe). The
        # "fe" entry under "methods" is computed against the noise-free magnitudes and is not used.
        fe_kl = np.array([long_[i]["member_fe"][0] for i in ids], dtype=float)
        fe_sel = np.array([min(five[i]["member_fe"]) for i in ids], dtype=float)
        rt_kl = float(np.nanmean([long_[i].get("runtime_total_s", np.nan) for i in ids]))
        rt_five = float(np.nanmean([five[i].get("runtime_total_s", np.nan) for i in ids]))
        long_lower_fe = fe_kl < fe_sel
        # in the self-test the stored run is compared with itself, which exercises the check
        n_same_meas = same_measurements(Path(a.study3) / run, Path(a.study3 if a.selftest else a.long) / run, ids)
        tests.append({"id": f"L{j + 1}", "run": run, "comparison": "FE-softmax - k1_long", **s3.paired(fs, kl)})
        tests.append({"id": f"L{j + 7}", "run": run, "comparison": "selection - k1_long", **s3.paired(sel, kl)})
        desc[run] = {"K=1": s3.describe(k1), "k1_long": s3.describe(kl), "selection": s3.describe(sel),
                     "FE-softmax": s3.describe(fs), "k1_long_minus_K1": s3.paired(kl, k1),
                     "median_fe_k1_long": float(np.median(fe_kl)), "median_fe_selection": float(np.median(fe_sel)),
                     "runtime_s_k1_long": rt_kl, "runtime_s_five_starts": rt_five,
                     "k1_long_lower_fe_than_selection": int(long_lower_fe.sum()),
                     "of_these_k1_long_higher_psnr": int((long_lower_fe & (kl > sel)).sum()),
                     "images_with_identical_measurement": n_same_meas}
    if missing:
        (out / "report.md").write_text("Incomplete runs:\n" + "\n".join(missing) + "\n", encoding="utf-8")
        print("incomplete:", missing)
        return 1
    tests.sort(key=lambda t: int(t["id"][1:]))
    p_holm = s3.holm(np.array([t["p_wilcoxon"] for t in tests]))
    for t, ph in zip(tests, p_holm):
        t["p_holm"] = float(ph)
        t["decision"] = s3.decide(t, ph)

    lines = ["# Study 3, compute-matched control: report" + (" (SELFTEST, k1_long replaced by K=1)" if a.selftest else ""), "",
             "PSNR in dB, capped at 60 dB, 50 images per run. Holm adjustment over L1 to L12.", "",
             "| id | run | comparison | mean | 95% CI | median | first/second better | p | p (Holm) | decision |",
             "|---|---|---|--:|---|--:|--:|--:|--:|---|"]
    for t in tests:
        lines.append(f"| {t['id']} | {t['run']} | {t['comparison']} | {t['mean']:+.2f} | [{t['ci'][0]:.2f}, {t['ci'][1]:.2f}] | "
                     f"{t['median']:+.2f} | {t['first_better']}/{t['second_better']} | {t['p_wilcoxon']:.2g} | {t['p_holm']:.2g} | {t['decision']} |")
    lines += ["", "## Descriptive", "",
              "| run | K=1 | k1_long | selection | FE-softmax | k1_long - K=1 [95% CI] | failures K=1 / k1_long / FE-softmax | median FE k1_long / selected start | FE of k1_long below selection (of these, higher PSNR) | s per image k1_long / five starts | identical measurements |",
              "|---|--:|--:|--:|--:|---|---|---|---|---|---|"]
    for run, d in desc.items():
        lines.append(f"| {run} | {d['K=1']['mean']:.2f} | {d['k1_long']['mean']:.2f} | {d['selection']['mean']:.2f} | "
                     f"{d['FE-softmax']['mean']:.2f} | {d['k1_long_minus_K1']['mean']:+.2f} "
                     f"[{d['k1_long_minus_K1']['ci'][0]:.2f}, {d['k1_long_minus_K1']['ci'][1]:.2f}] | "
                     f"{d['K=1']['failures']} / {d['k1_long']['failures']} / {d['FE-softmax']['failures']} | "
                     f"{d['median_fe_k1_long']:.3f} / {d['median_fe_selection']:.3f} | "
                     f"{d['k1_long_lower_fe_than_selection']} ({d['of_these_k1_long_higher_psnr']}) | "
                     f"{d['runtime_s_k1_long']:.0f} / {d['runtime_s_five_starts']:.0f} | "
                     f"{d['images_with_identical_measurement'] if d['images_with_identical_measurement'] is not None else 'n/a'} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"tests": tests, "descriptive": desc, "selftest": a.selftest}, indent=2),
                                      encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
