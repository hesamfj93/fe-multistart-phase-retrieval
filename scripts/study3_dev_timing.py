#!/usr/bin/env python
"""Numbers of classical starts at equal wall time (STUDY3_DEV_PLAN.md, Section 4).

Reads the development runs of results/study3_dev made in one laptop-GPU session:
- untrained methods: mean runtime_total_s per image of {real,complex}_ours_{level} and
  real_netadm_{level} (5 starts each, all aggregates included);
- classical methods: mean runtime of one start, from the best-of-5 runtime of
  {real,complex}_classical_{level}.
K_classical = round(T_untrained / t_start), with both means pooled over the two Poisson levels.

    python scripts/study3_dev_timing.py --root results/study3_dev --out results/study3_dev/timing.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEVELS = ("moderate", "strong")
CLASSICAL = {"real": ("hio_realpos", "raar_realpos", "hio_sw", "raar_sw"), "complex": ("hio_sw", "raar_sw")}
UNTRAINED = {"real": ("ours", "netadm"), "complex": ("ours",)}


def per_image(run: Path) -> list[dict]:
    out = []
    for mj in sorted((run / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") == "ok":
            out.append(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="results/study3_dev")
    ap.add_argument("--out", default="results/study3_dev/timing.json")
    a = ap.parse_args()
    root = Path(a.root)
    res = {"untrained_s_per_image": {}, "classical_s_per_start": {}, "devices": {}, "k_equal_wall_time": {}}
    for t, solvers in UNTRAINED.items():
        for s in solvers:
            ts, dev = [], set()
            for lv in LEVELS:
                for m in per_image(root / f"{t}_{s}_{lv}"):
                    ts.append(float(m["runtime_total_s"]))
                    dev.add(m.get("device_used"))
            res["untrained_s_per_image"][f"{t}_{s}"] = float(np.mean(ts))
            res["devices"][f"{t}_{s}"] = sorted(map(str, dev))
    for t, methods in CLASSICAL.items():
        for meth in methods:
            ts, dev = [], set()
            for lv in LEVELS:
                for m in per_image(root / f"{t}_classical_{lv}"):
                    ts.append(float(m["methods"][f"{meth}_best5"]["runtime_s"]) / 5)
                    dev.add(m.get("device_used"))
            res["classical_s_per_start"][f"{t}_{meth}"] = float(np.mean(ts))
            res["devices"][f"{t}_classical"] = sorted(map(str, dev))
            for s in UNTRAINED[t]:
                k = int(round(res["untrained_s_per_image"][f"{t}_{s}"] / res["classical_s_per_start"][f"{t}_{meth}"]))
                res["k_equal_wall_time"][f"{t}_{s}_vs_{meth}"] = max(k, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
