#!/usr/bin/env python
"""Appendix table of the compute-matched control (L1 to L12) from results/study3_long/report/summary.json."""
import json
from pathlib import Path

S = json.loads(Path("results/study3_long/report/summary.json").read_text())
NAME = {"real_ours_moderate": "ours, real, moderate", "real_ours_strong": "ours, real, strong",
        "complex_ours_moderate": "ours, complex, moderate", "complex_ours_strong": "ours, complex, strong",
        "real_netadm_moderate": "Net-ADM, real, moderate", "real_netadm_strong": "Net-ADM, real, strong"}
DEC = {"first better": "five starts", "no claim": "none", "second better": "\\texttt{k1\\_long}"}


def p(x):
    return f"{x:.2g}" if x >= 1e-3 else f"${x:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}$"


def n(x, sign=False):
    s = f"{x:+.2f}" if sign else f"{x:.2f}"
    return f"${s}$" if x < 0 else s


rows = []
for t in S["tests"]:
    d = S["descriptive"][t["run"]]
    first = "FE-softmax" if t["comparison"].startswith("FE-softmax") else "selection"
    extra = f"{d['K=1']['mean']:.2f} & {d['k1_long']['mean']:.2f} & {n(d['k1_long_minus_K1']['mean'], True)}" if first == "FE-softmax" else "& &"
    rows.append(f"{t['id']} & {NAME[t['run']]} & {first} & ${t['mean']:+.2f}$ & [{n(t['ci'][0])}, {n(t['ci'][1])}] & "
                f"{t['first_better']}/{t['second_better']} & {p(t['p_holm'])} & {DEC[t['decision']]} & {extra} \\\\")
Path("paper/tmlr_supp/tab_study3_long.tex").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
