#!/usr/bin/env python
"""Rows of the appendix tables for the post hoc analyses of the fifth review
(results/posthoc_review5/summary.json, from scripts/posthoc_review5.py).

    python paper/tmlr_review5_tables.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "tmlr_supp"
s = json.loads((REPO / "results/posthoc_review5/summary.json").read_text())
TEMPS = ("0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "uniform")
LABEL = {"S1 real, no noise": "1, ours, real, none", "S2 complex, no noise": "2, ours, complex, none",
         "S3 ours real, moderate": "3, ours, real, moderate", "S3 ours real, strong": "3, ours, real, strong",
         "S3 ours complex, moderate": "3, ours, complex, moderate", "S3 ours complex, strong": "3, ours, complex, strong",
         "S3 Net-ADM real, moderate": "3, Net-ADM, real, moderate", "S3 Net-ADM real, strong": "3, Net-ADM, real, strong"}


def ci(d):
    return f"${d['mean']:+.2f}$ [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"


t_rows, k_rows = [], []
for n, v in s.items():
    k1 = v["k_curve"]["1"]["selection"]
    t_rows.append(f"{LABEL[n]} & {k1:.2f} & " + " & ".join(f"{v['temperature'][T]['mean']:.2f}" for T in TEMPS) + r" \\")
    k_rows.append(f"{LABEL[n]} & " + " & ".join(f"{v['k_curve'][str(k)]['fe_softmax']:.2f}" for k in range(1, 6))
                  + f" & {ci(v['window_fe_softmax_minus_K1'])} & {ci(v['inside_object_fe_softmax_minus_K1'])}" + r" \\")
(OUT / "tab_posthoc_temperature.tex").write_text("\n".join(t_rows) + "\n", encoding="utf-8")
(OUT / "tab_posthoc_kcurve.tex").write_text("\n".join(k_rows) + "\n", encoding="utf-8")
print("\n".join(t_rows + k_rows))
