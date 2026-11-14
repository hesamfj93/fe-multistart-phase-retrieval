"""Spectral-weight ablation: trainer option, configurations and decision rule."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from pr_uq.losses.spectral_weight import SpectralAnnealer
from pr_uq.optim.trainer import TrainerConfig, spectral_weight_for_step

REPO = Path(__file__).resolve().parent.parent
BASE = {"real": "stage2_mnist_dev8.yaml", "complex": "stage3_complex_dev.yaml"}
MODS = {"frozen": {}, "uniform": {"use_spectral_annealing": False},
        "widening": {"sigma_start_frac": 0.05, "sigma_end_frac": 0.5},
        "dcfinal": {"unweighted_final_frac": 0.2}}


def _annealer(iters=3000):
    return SpectralAnnealer(64, 64, 32.0, 3.2, total_steps=iters)


def test_default_keeps_the_frozen_weight_at_every_step():
    cfg = TrainerConfig(iterations=3000)
    assert cfg.unweighted_final_frac == 0.0
    a, ref = _annealer(), _annealer()
    for step in (0, 1, 1799, 1800, 2399, 2400, 2999):
        assert torch.equal(spectral_weight_for_step(a, step, cfg), ref.weight_at(step))


def test_unweighted_final_fraction_switches_the_weight_off():
    cfg = TrainerConfig(iterations=3000, unweighted_final_frac=0.2)
    a = _annealer()
    assert spectral_weight_for_step(a, 2399, cfg) is not None
    assert all(spectral_weight_for_step(a, s, cfg) is None for s in (2400, 2700, 2999))
    assert spectral_weight_for_step(None, 10, cfg) is None


@pytest.mark.parametrize("otype", ["real", "complex"])
@pytest.mark.parametrize("variant", list(MODS))
def test_ablation_config_differs_from_base_only_where_declared(otype, variant):
    base = yaml.safe_load((REPO / "configs" / BASE[otype]).read_text(encoding="utf-8"))
    cfg = yaml.safe_load((REPO / "configs" / "spectral_ablation" / f"{otype}_{variant}.yaml").read_text(encoding="utf-8"))
    assert cfg["dataset"]["protocol_split"] == "dev" and cfg["dataset"]["n_images"] == 8
    assert cfg["experiment"]["output_dir"] == f"results/spectral_ablation/{otype}_{variant}"
    expected = copy.deepcopy(base)
    expected["trainer"].update(MODS[variant])
    for c in (cfg, expected):
        c.pop("experiment")
    assert cfg == expected


def _fake_run(root: Path, name: str, fe: float, psnr: float):
    for i in range(8):
        d = root / name / "per_image" / f"image_{i:05d}"
        d.mkdir(parents=True)
        ps = [psnr - 2, psnr - 1, psnr, psnr + 1, psnr + 2]
        m = {"status": "ok", "member_fe": [fe] * 5, "member_psnr": ps,
             "methods": {"k1": {"psnr": ps[0]}, "best_single_by_fe": {"psnr": ps[4]},
                         "fe_softmax": {"psnr": ps[3], "fe": fe}}}
        (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")


def _report():
    spec = importlib.util.spec_from_file_location("sar", REPO / "scripts" / "spectral_ablation_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decision_rule(tmp_path):
    R = _report()
    base = {"frozen": (0.25, 16.0), "uniform": (0.24, 16.2), "dcfinal": (0.10, 17.5)}
    for t in R.TYPES:
        for v, (fe, ps) in base.items():
            _fake_run(tmp_path, f"{t}_{v}", fe, ps)
    _fake_run(tmp_path, "real_widening", 0.05, 19.0)        # materially better on real
    _fake_run(tmp_path, "complex_widening", 0.20, 15.2)     # 0.8 dB worse on complex: fails tolerance
    s = {t: {v: R.summarize(R.load_run(tmp_path / f"{t}_{v}")) for v in R.VARIANTS} for t in R.TYPES}
    d = R.decide(s)
    assert d["variants"]["widening"]["materially_better"]["real"] is True
    assert d["variants"]["widening"]["qualifies"] is False
    assert d["variants"]["uniform"]["qualifies"] is False
    assert d["variants"]["dcfinal"]["qualifies"] is True    # FE 0.10 <= 0.125 and +1.5 dB on both
    assert d["outcome"] == "B" and d["selected"] == "dcfinal"
