"""Component ablation: configurations differ from the base only where declared; decision rule."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
BASE = {"real": "stage2_mnist_dev8.yaml", "complex": "stage3_complex_dev.yaml"}
MODS = {"noproj": {"use_physics_projection": False},
        "noreg": {"lambda_tv_A": 0.0, "lambda_tv_phi": 0.0, "lambda_support": 0.0},
        "noclip": {"grad_clip": 0.0},
        "alloff": {"use_physics_projection": False, "lambda_tv_A": 0.0, "lambda_tv_phi": 0.0,
                   "lambda_support": 0.0, "grad_clip": 0.0}}


@pytest.mark.parametrize("otype", ["real", "complex"])
@pytest.mark.parametrize("variant", list(MODS))
def test_config_differs_from_base_only_where_declared(otype, variant):
    base = yaml.safe_load((REPO / "configs" / BASE[otype]).read_text(encoding="utf-8"))
    cfg = yaml.safe_load((REPO / "configs" / "component_ablation" / f"{otype}_{variant}.yaml").read_text(encoding="utf-8"))
    assert cfg["dataset"]["protocol_split"] == "dev" and cfg["dataset"]["n_images"] == 8
    assert cfg["experiment"]["output_dir"] == f"results/component_ablation/{otype}_{variant}"
    expected = copy.deepcopy(base)
    expected["trainer"].update(MODS[variant])
    for c in (cfg, expected):
        c.pop("experiment")
    assert cfg == expected


def _fake(root: Path, name: str, fe: float, psnr: float):
    for i in range(8):
        d = root / name / "per_image" / f"image_{i:05d}"
        d.mkdir(parents=True)
        ps = [psnr - 2, psnr - 1, psnr, psnr + 1, psnr + 2]
        m = {"status": "ok", "member_fe": [fe] * 5, "member_psnr": ps,
             "methods": {"k1": {"psnr": ps[0]}, "best_single_by_fe": {"psnr": ps[4]}, "fe_softmax": {"psnr": ps[3], "fe": fe}}}
        (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")


def test_decision_rule(tmp_path):
    spec = importlib.util.spec_from_file_location("car", REPO / "scripts" / "component_ablation_report.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    froot, root = tmp_path / "f", tmp_path / "c"
    for t in ("real", "complex"):
        _fake(froot, f"{t}_frozen", 0.20, 17.0)
        _fake(root, f"{t}_noproj", 0.19, 17.2)
        _fake(root, f"{t}_noclip", 0.21, 16.9)
        _fake(root, f"{t}_alloff", 0.09, 19.0)
    _fake(root, "real_noreg", 0.08, 18.5)
    _fake(root, "complex_noreg", 0.15, 16.0)            # 1 dB worse on complex: fails tolerance
    s = {t: {v: R.SAR.summarize(R.SAR.load_run(R.run_dir(root, froot, t, v))) for v in R.VARIANTS} for t in R.TYPES}
    d = R.decide(s)
    assert d["variants"]["noreg"]["materially_better"]["real"] and not d["variants"]["noreg"]["qualifies"]
    assert d["variants"]["alloff"]["qualifies"] and not d["variants"]["noproj"]["qualifies"]
    assert d["selected"] == "alloff"
