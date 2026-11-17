#!/usr/bin/env python
"""Report of the third study (STUDY3_PLAN.md): Poisson noise, two untrained solvers, classical
methods at equal wall time.

Inputs are the run folders of configs/study3 under --root. Untrained runs give K = 1, selection
(lowest-FE start), FE-softmax, uniform and median from five starts. Classical runs store the FE and
PSNR of every start, so "best of the first K starts" is the start with the lowest FE among seeds
0 ... K-1, which is what a run with K starts would have returned.

PSNR is the metric stored by the runners (real objects: PSNR of the aligned real image; complex
objects: complex PSNR), capped at 60 dB. Below 15 dB is a failure.

Held-out report:
    python scripts/study3_report.py --root results/study3 --out results/study3/report

Development check of the script (8 images, classical runs with 5 starts, so K = 5 everywhere):
    python scripts/study3_report.py --root results/study3_dev --out <tmp> --dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

CAP, FAIL = 60.0, 15.0
N_BOOT, BOOT_SEED = 10_000, 20261117
LEVELS = ("moderate", "strong")
AGGS = ("k1", "best_single_by_fe", "fe_softmax", "uniform", "median")
AGG_NAME = {"k1": "K=1", "best_single_by_fe": "selection", "fe_softmax": "FE-softmax", "uniform": "uniform",
            "median": "median"}
# equal-wall-time numbers of classical starts (results/study3_dev/timing.json, STUDY3_PLAN.md)
K_EQ = {"real_ours": 81, "real_netadm": 22, "complex_ours": 50}
HELDOUT = {"classical_real": "real_hio81_{lv}", "classical_complex": "complex_raar50_{lv}", "n": 50}
DEV = {"classical_real": "real_classical_{lv}", "classical_complex": "complex_classical_{lv}", "n": 8}
CLASSICAL_KEY = {"real": "hio_realpos", "complex": "raar_sw"}


def load_run(d: Path) -> dict:
    out = {}
    for mj in sorted((d / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") == "ok":
            out[int(m["mnist_test_index"])] = m
    return out


def cap(x) -> np.ndarray:
    return np.minimum(np.asarray(x, dtype=float), CAP)


def untrained_values(run: dict, ids: list[int]) -> dict:
    v = {a: cap([run[i]["methods"][a]["psnr"] for i in ids]) for a in AGGS}
    v["fe_fe_softmax"] = np.array([run[i]["methods"]["fe_softmax"].get("fe", np.nan) for i in ids], dtype=float)
    sp = np.array([cap(run[i]["member_psnr"]) for i in ids])
    fe = np.array([run[i]["member_fe"] for i in ids], dtype=float)
    v["best_start"], v["mean_start"] = sp.max(axis=1), sp.mean(axis=1)
    v["lowest_fe_is_best"] = int(sum(np.argmin(f) == np.argmax(p) for f, p in zip(fe, sp)))
    v["median_start_fe"] = float(np.median(fe))
    v["runtime_s"] = float(np.mean([run[i].get("runtime_total_s", np.nan) for i in ids]))
    return v


def best_of_first(run: dict, ids: list[int], key: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    ps, fs = [], []
    for i in ids:
        st = run[i]["starts"][key]
        fe = np.asarray(st["start_fe"][:k], dtype=float)
        if len(fe) < k:
            raise SystemExit(f"image {i}: {key} has {len(fe)} starts, {k} needed")
        j = int(np.argmin(fe))
        ps.append(st["start_psnr"][j])
        fs.append(fe[j])
    return cap(ps), np.array(fs)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    nz = d[np.abs(d) > 1e-9]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    return {"n": n, "mean": float(d.mean()), "ci": [float(d.mean() - t * se), float(d.mean() + t * se)],
            "median": float(np.median(d)), "first_better": int((d > 1e-9).sum()),
            "second_better": int((d < -1e-9).sum()), "p_wilcoxon": p}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    adj, run = np.empty(len(p)), 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[idx]))
        adj[idx] = run
    return adj


def decide(r: dict, p_holm: float) -> str:
    if p_holm >= 0.05:
        return "no claim"
    if r["mean"] > 0 and r["ci"][0] > 0:
        return "first better"
    if r["mean"] < 0 and r["ci"][1] < 0:
        return "second better"
    return "no claim"


def spearman_fe(fe: np.ndarray, psnr: np.ndarray, rng) -> dict:
    ok = np.isfinite(fe)
    fe, psnr = fe[ok], psnr[ok]
    rho, p_two = stats.spearmanr(fe, -psnr)
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    idx = rng.integers(0, len(fe), size=(N_BOOT, len(fe)))
    boot = [stats.spearmanr(fe[j], -psnr[j])[0] for j in idx]
    return {"n": int(len(fe)), "rho": float(rho), "p_one_sided": float(p_one),
            "ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]}


def describe(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "median": float(np.median(x)),
            "failures": int((x < FAIL).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", action="store_true", help="development check: 8 images, K = 5 for classical runs")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    names = DEV if a.dev else HELDOUT
    k_eq = {k: 5 for k in K_EQ} if a.dev else dict(K_EQ)
    rng = np.random.default_rng(BOOT_SEED)

    runs, missing = {}, []
    wanted = [f"{t}_{s}_{lv}" for t, s in (("real", "ours"), ("complex", "ours"), ("real", "netadm")) for lv in LEVELS]
    wanted += [names["classical_real"].format(lv=lv) for lv in LEVELS]
    wanted += [names["classical_complex"].format(lv=lv) for lv in LEVELS]
    for w in wanted:
        r = load_run(root / w)
        if len(r) != names["n"]:
            missing.append(f"{w}: {len(r)} of {names['n']} images")
        runs[w] = r
    noiseless = load_run(root / "real_netadm_none")
    if missing:
        print("Incomplete runs, no report:\n  " + "\n  ".join(missing))
        return 1

    # ---- confirmatory family ---------------------------------------------------------------
    tests = []
    def add(tid, label, run_a, va, run_b, vb):
        ids = sorted(set(run_a) & set(run_b))
        tests.append({"id": tid, "comparison": label, **paired(va(run_a, ids), vb(run_b, ids))})
    agg = lambda key: (lambda run, ids: cap([run[i]["methods"][key]["psnr"] for i in ids]))
    n_id = 0
    for t in ("real", "complex"):
        for lv in LEVELS:
            n_id += 1
            r = runs[f"{t}_ours_{lv}"]
            add(f"N{n_id}", f"our solver, {t}, {lv}: FE-softmax vs K=1", r, agg("fe_softmax"), r, agg("k1"))
    for lv in LEVELS:
        n_id += 1
        r = runs[f"real_netadm_{lv}"]
        add(f"N{n_id}", f"Net-ADM, real, {lv}: FE-softmax vs K=1", r, agg("fe_softmax"), r, agg("k1"))
    for t in ("real", "complex"):
        for lv in LEVELS:
            n_id += 1
            k = k_eq[f"{t}_ours"]
            cr = runs[names[f"classical_{t}"].format(lv=lv)]
            key = CLASSICAL_KEY[t]
            add(f"N{n_id}", f"{t}, {lv}: our FE-softmax vs {key} best of {k}", runs[f"{t}_ours_{lv}"],
                agg("fe_softmax"), cr, lambda run, ids, key=key, k=k: best_of_first(run, ids, key, k)[0])
    for lv in LEVELS:
        n_id += 1
        k = k_eq["real_netadm"]
        cr = runs[names["classical_real"].format(lv=lv)]
        add(f"N{n_id}", f"real, {lv}: Net-ADM FE-softmax vs hio_realpos best of {k}", runs[f"real_netadm_{lv}"],
            agg("fe_softmax"), cr, lambda run, ids, k=k: best_of_first(run, ids, "hio_realpos", k)[0])
    adj = holm(np.array([x["p_wilcoxon"] for x in tests]))
    for x, p in zip(tests, adj):
        x["p_holm"] = float(p)
        x["decision"] = decide(x, p)

    # ---- N13: FE as an image-level signal (outside the Holm family) ------------------------
    fe_signal = {}
    for t, s in (("real", "ours"), ("complex", "ours"), ("real", "netadm")):
        for lv in LEVELS:
            r = runs[f"{t}_{s}_{lv}"]
            ids = sorted(r)
            v = untrained_values(r, ids)
            fe_signal[f"{t}_{s}_{lv}"] = spearman_fe(v["fe_fe_softmax"], v["fe_softmax"], rng)

    # ---- descriptive ---------------------------------------------------------------------
    desc = {}
    for t, s in (("real", "ours"), ("complex", "ours"), ("real", "netadm")):
        for lv in LEVELS:
            r = runs[f"{t}_{s}_{lv}"]
            v = untrained_values(r, sorted(r))
            desc[f"{t}_{s}_{lv}"] = {**{AGG_NAME[k]: describe(v[k]) for k in AGGS},
                                     "best start (oracle)": describe(v["best_start"]),
                                     "mean start": describe(v["mean_start"]),
                                     "lowest FE is best start": v["lowest_fe_is_best"],
                                     "median start FE": v["median_start_fe"], "runtime_s": v["runtime_s"]}
    if noiseless:
        v = untrained_values(noiseless, sorted(noiseless))
        desc["real_netadm_none"] = {**{AGG_NAME[k]: describe(v[k]) for k in AGGS},
                                    "best start (oracle)": describe(v["best_start"]),
                                    "lowest FE is best start": v["lowest_fe_is_best"], "n": len(noiseless)}
    classical = {}
    for t in ("real", "complex"):
        ks = sorted({1, 5, 10, 20, k_eq[f"{t}_ours"]} | ({k_eq["real_netadm"]} if t == "real" else set()))
        for lv in LEVELS:
            r = runs[names[f"classical_{t}"].format(lv=lv)]
            ids = sorted(r)
            n_starts = min(len(r[i]["starts"][CLASSICAL_KEY[t]]["start_fe"]) for i in ids)
            classical[f"{t}_{CLASSICAL_KEY[t]}_{lv}"] = {
                f"best of {k}": describe(best_of_first(r, ids, CLASSICAL_KEY[t], k)[0]) for k in ks if k <= n_starts}

    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "STUDY3_PLAN.md", "dev_check": a.dev, "k_equal_wall_time": k_eq, "tests": tests,
               "fe_signal_N13": fe_signal, "descriptive": desc, "classical": classical}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    L = [f"# Third study{' (development check of the report script)' if a.dev else ''}", "",
         f"PSNR capped at {CAP:.0f} dB; failure below {FAIL:.0f} dB. Classical: best of the first K starts by FE.", "",
         "## Confirmatory family (Holm over N1 to N12)", "",
         "| id | comparison | n | mean diff (dB) | 95% CI | median | first / second better | Wilcoxon p | Holm p | decision |",
         "|---|---|--:|--:|---|--:|---|--:|--:|---|"]
    for x in tests:
        L.append(f"| {x['id']} | {x['comparison']} | {x['n']} | {x['mean']:+.2f} | [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] | "
                 f"{x['median']:+.2f} | {x['first_better']} / {x['second_better']} | {x['p_wilcoxon']:.2g} | "
                 f"{x['p_holm']:.2g} | {x['decision']} |")
    L += ["", "## N13: FE of the FE-softmax result against PSNR across images (one-sided, not in the Holm family)", "",
          "| run | n | Spearman(FE, -PSNR) | bootstrap 95% CI | one-sided p |", "|---|--:|--:|---|--:|"]
    for k, x in fe_signal.items():
        L.append(f"| {k} | {x['n']} | {x['rho']:.2f} | [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] | {x['p_one_sided']:.2g} |")
    L += ["", "## Untrained runs (descriptive)", "",
          "| run | K=1 | selection | FE-softmax | uniform | median | best start | failures K=1 / FE-softmax | lowest FE = best start | median start FE | s per image |",
          "|---|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|"]
    for k, x in desc.items():
        if k == "real_netadm_none":
            continue
        L.append(f"| {k} | {x['K=1']['mean']:.2f} | {x['selection']['mean']:.2f} | {x['FE-softmax']['mean']:.2f} | "
                 f"{x['uniform']['mean']:.2f} | {x['median']['mean']:.2f} | {x['best start (oracle)']['mean']:.2f} | "
                 f"{x['K=1']['failures']} / {x['FE-softmax']['failures']} | {x['lowest FE is best start']} | "
                 f"{x['median start FE']:.3f} | {x['runtime_s']:.0f} |")
    if "real_netadm_none" in desc:
        x = desc["real_netadm_none"]
        L += ["", f"Net-ADM without noise (n = {x['n']}, descriptive): K=1 {x['K=1']['mean']:.2f}, selection "
              f"{x['selection']['mean']:.2f}, FE-softmax {x['FE-softmax']['mean']:.2f}, best start "
              f"{x['best start (oracle)']['mean']:.2f} dB."]
    L += ["", "## Classical runs (descriptive): mean PSNR (failures)", ""]
    for k, x in classical.items():
        L.append(f"- {k}: " + "; ".join(f"{kk} {vv['mean']:.2f} ({vv['failures']})" for kk, vv in x.items()))
    L.append("")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
