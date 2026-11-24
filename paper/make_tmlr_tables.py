#!/usr/bin/env python
"""LaTeX table rows for the appendix of the TMLR manuscript (Study 3 and the measured pattern).

Reads results/study3/report/summary.json and results/realdata*/; writes paper/tmlr_supp/*.tex.

    python paper/make_tmlr_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "tmlr_supp"


def p_fmt(p: float) -> str:
    if p < 1e-6:
        return "$<10^{-6}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def p_sci(p: float) -> str:
    """Exact p in scientific notation, for tests whose p is far below the display floor of p_fmt."""
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    s = json.loads((REPO / "results/study3/report/summary.json").read_text())
    rows = []
    for t in s["tests"]:
        comp = t["comparison"].replace("hio_realpos", "HIO r+n").replace("raar_sw", "RAAR sw").replace("_", "\\_")
        dec = {"first better": "first", "second better": "second"}.get(t["decision"], t["decision"])
        rows.append(f"{t['id']} & {comp} & ${t['mean']:+.2f}$ & [{t['ci'][0]:.2f}, {t['ci'][1]:.2f}] & ${t['median']:+.2f}$ & "
                    f"{t['first_better']}/{t['second_better']} & {p_fmt(t['p_wilcoxon'])} & {p_fmt(t['p_holm'])} & {dec} \\\\")
    (OUT / "tab_study3_tests.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for run, v in s["fe_signal_N13"].items():
        rows.append(f"{run.replace('_', ' ')} & {v['n']} & {v['rho']:.2f} & [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] & {p_sci(v['p_one_sided'])} \\\\")
    (OUT / "tab_study3_n13.tex").write_text("\n".join(rows) + "\n")
    rows = []
    order = ("K=1", "uniform", "median", "selection", "FE-softmax", "best start (oracle)")
    for run, d in s["descriptive"].items():
        cells = []
        for r in order:
            if r in d:
                cells.append(f"{d[r]['mean']:.2f} ({d[r]['failures']})")
            else:
                cells.append("--")
        rows.append(f"{run.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    (OUT / "tab_study3_desc.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for run, d in s["classical"].items():
        cells = [f"{k.replace('best of ', '')}: {v['mean']:.2f} ({v['failures']})" for k, v in d.items()]
        rows.append(f"{run.replace('_', ' ')} & " + "; ".join(cells) + " \\\\")
    (OUT / "tab_study3_classical.tex").write_text("\n".join(rows) + "\n")
    print("\n".join(sorted(p.name for p in OUT.glob("*.tex"))))
    print(json.dumps(list(s["descriptive"]["real_ours_moderate"].keys())))


if __name__ == "__main__":
    main()
