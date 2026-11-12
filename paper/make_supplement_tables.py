"""Write the tables of supplement.tex from the stored results (no hand-typed numbers).

    python paper/make_supplement_tables.py --run-dir results/stage2_mnist_heldout \
        --stage3 results/stage3_complex_heldout/report

--stage3 reads summary.json (scripts/stage3_report.py) and posthoc_stage3.json (scripts/stage3_posthoc.py).
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

NAME = {"K=5 best-single-by-FE": "selection", "K=5 FE-softmax (T=0.05)": "FE-softmax", "K=5 uniform": "uniform",
        "K=5 median": "median", "K=1": "K=1", "K=1, 15000 it (compute-matched)": "\\texttt{k1\\_long}"}
HYP = {("FE-softmax", "\\texttt{k1\\_long}"): "H1a", ("selection", "\\texttt{k1\\_long}"): "H1b",
       ("FE-softmax", "HIO supp.\\ \\texttt{best5}"): "H2a", ("FE-softmax", "RAAR supp.\\ \\texttt{best5}"): "H2b"}


def base(b: str) -> str:
    if b in NAME:
        return NAME[b]
    alg = "HIO" if b.startswith("HIO") else "RAAR"
    con = "supp." if "support" in b else "r+n"
    st = "\\texttt{best5}" if "best of 5" in b else "\\texttt{s1}"
    return f"{alg} {con}\\ {st}"


def pfmt(p: float) -> str:
    if p < 1e-6:
        return "$<10^{-6}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--stage3")
    a = ap.parse_args()
    run = Path(a.run_dir); out = Path(__file__).resolve().parent / "supp"; out.mkdir(exist_ok=True)
    ph = json.loads((run / "report_n100/posthoc/posthoc_revision2.json").read_text(encoding="utf-8"))
    rows = []
    for i, r in enumerate(ph["holm_family"], 1):
        m, b = NAME.get(r["method"], r["method"]), base(r["baseline"])
        h = HYP.get((m, b), "")
        rows.append(f"{i} & {h} & {m} & {b} & ${r['mean']:+.2f}$ & [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] & "
                    f"{r['better']}/{r['worse']} & {pfmt(r['p_uncorrected'])} & {pfmt(r['p_holm_recomputed'])} \\\\")
    (out / "tab_holm.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    lab = {"k1": "K=1", "uniform": "uniform", "median": "median", "best_single_by_fe": "selection",
           "fe_softmax": "FE-softmax", "mad": "image-level MAD", "madpix": "pixel-wise MAD"}
    rows = []
    for k, v in ph["complex_metrics"].items():
        rows.append(f"{lab[k]} & {v['nrmse_complex']['mean']:.3f} & {v['nrmse_complex']['median']:.3f} & "
                    f"{v['nrmse_real']['mean']:.3f} & {100*v['imag_frac']['mean']:.2f} & {100*v['imag_frac']['median']:.2f} & "
                    f"{100*v['negreal_frac']['mean']:.2f} \\\\")
    (out / "tab_complex.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if "mad_paired_psnr" in ph:
        rows = []
        names = [("mad_vs_k1", "image-level MAD vs K=1"), ("mad_vs_uniform", "image-level MAD vs uniform"),
                 ("mad_vs_median", "image-level MAD vs median"), ("best_single_vs_mad", "selection vs image-level MAD"),
                 ("fe_softmax_vs_mad", "FE-softmax vs image-level MAD"),
                 ("madpix_vs_k1", "pixel-wise MAD vs K=1"), ("madpix_vs_uniform", "pixel-wise MAD vs uniform"),
                 ("madpix_vs_median", "pixel-wise MAD vs median"), ("best_single_vs_madpix", "selection vs pixel-wise MAD"),
                 ("fe_softmax_vs_madpix", "FE-softmax vs pixel-wise MAD")]
        for k, nm in names:
            v = ph["mad_paired_psnr"][k]
            rows.append(f"{nm} & ${v['mean']:+.2f}$ & [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] & {v['better']}/{v['worse']}/{v['equal']} & {pfmt(v['wilcoxon_p_uncorrected'])} \\\\")
        (out / "tab_mad.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
        s, q = ph["mad_summary"], ph["madpix_summary"]; kh = s["kept_histogram"]
        (out / "mad_summary.tex").write_text(
            f"\\newcommand{{\\madpsnr}}{{{s['psnr_mean']:.2f}}}\\newcommand{{\\madssim}}{{{s['ssim_mean']:.3f}}}"
            f"\\newcommand{{\\madfail}}{{{s['failures']}}}\\newcommand{{\\madrej}}{{{s['images_with_rejection']}}}"
            f"\\newcommand{{\\madkeptfour}}{{{kh['4']}}}\\newcommand{{\\madkeptthree}}{{{kh['3']}}}"
            f"\\newcommand{{\\madpixpsnr}}{{{q['psnr_mean']:.2f}}}\\newcommand{{\\madpixssim}}{{{q['ssim_mean']:.3f}}}"
            f"\\newcommand{{\\madpixfail}}{{{q['failures']}}}\\newcommand{{\\madpixrej}}{{{100*q['rejected_value_fraction_mean']:.0f}}}\n",
            encoding="utf-8")
    if a.stage3:
        stage3_tables(Path(a.stage3), out)
    print("written to", out)
    return 0


S3_NAME = {"k1": "K=1", "uniform": "uniform", "median": "median", "best_single_by_fe": "selection",
           "fe_softmax": "FE-softmax", "k1_long": "\\texttt{k1\\_long}"}


def s3_name(m: str) -> str:
    if m in S3_NAME:
        return S3_NAME[m]
    alg, con, st = m.split("_")
    con = {"supp": "supp.", "sw": "shrinkwrap", "realpos": "r+n"}[con]
    return f"{alg.upper()} {con}\\ \\texttt{{{st}}}"


def stage3_tables(rep: Path, out: Path) -> None:
    sm = json.loads((rep / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for d in sm["methods"]:
        rows.append(f"{s3_name(d['method'])} & ${d['psnr_mean']:.2f}\\pm{d['psnr_std']:.2f}$ & {d['psnr_median']:.2f} & "
                    f"[{d['psnr_ci'][0]:.2f}, {d['psnr_ci'][1]:.2f}] & {d['psnr_amp_mean']:.2f} & {d['phase_err_mean']:.3f} & "
                    f"{d['nrmse_c_mean']:.3f} & {d['ssim_amp_mean']:.3f} & {d['fe_mean']:.3f} & {d['failures']} & {d['exact']} & "
                    f"{d['runtime_s']:.1f} \\\\")
    (out / "tab_s3_methods.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    rows = []
    for h in sm["hypotheses"]:
        rows.append(f"{h['id']} & {s3_name(h['first'])} & {s3_name(h['second'])} & ${h['mean']:+.2f}$ & "
                    f"[{h['ci'][0]:.2f}, {h['ci'][1]:.2f}] & ${h['median']:+.2f}$ & {h['better']}/{h['worse']}/{h['equal']} & "
                    f"{pfmt(h['p_uncorrected'])} & {pfmt(h['p_holm'])} & {'yes' if h['supported'] else 'no'} \\\\")
    (out / "tab_s3_hyp.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    ph = json.loads((rep / "posthoc_stage3.json").read_text(encoding="utf-8"))
    rows = []
    for r in ph["fallback_raar_sw_then_fe_softmax"]:
        rows.append(f"{r['t']:g} & {r['raar_used']} & {r['mean']:.2f} & {r['median']:.2f} & {r['failures']} & {r['exact']} \\\\")
    for k, nm in (("raar_sw_best5", "RAAR shrinkwrap \\texttt{best5} alone"), ("fe_softmax", "FE-softmax alone")):
        r = ph["reference"][k]
        rows.append(f"\\multicolumn{{2}}{{@{{}}l}}{{{nm}}} & {r['mean']:.2f} & {r['median']:.2f} & {r['failures']} & {r['exact']} \\\\")
    (out / "tab_s3_fallback.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
