"""The real-nonnegative test of the third study (STUDY3_REALNN_PLAN.md): its configurations differ from
those of Study 3 only in the keys the plan names, and the trainer option gives real, nonnegative starts."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from pr_uq.models.ensemble import EnsembleMember, MemberSpec
from pr_uq.optim.trainer import TrainerConfig, train_member

REPO = Path(__file__).resolve().parent.parent


def _strip(c, extra=()):
    c = copy.deepcopy(c)
    for k in ("name", "output_dir"):
        c["experiment"].pop(k, None)
    for sec, key in extra:
        c[sec].pop(key, None)
    return c


@pytest.mark.parametrize("level", ("moderate", "strong"))
def test_realnn_config_matches_study3(level):
    base = yaml.safe_load((REPO / "configs/study3" / f"real_ours_{level}.yaml").read_text())
    new = yaml.safe_load((REPO / "configs/study3_realnn" / f"real_oursnn_{level}.yaml").read_text())
    assert new["trainer"]["object_constraint"] == "real_nonneg"
    assert "object_constraint" not in base["trainer"]
    assert new["experiment"]["output_dir"] == f"results/study3_realnn/real_oursnn_{level}"
    assert _strip(base) == _strip(new, extra=[("trainer", "object_constraint")])


def test_repeat_config_matches_study3():
    base = yaml.safe_load((REPO / "configs/study3/real_ours_moderate.yaml").read_text())
    rep = yaml.safe_load((REPO / "configs/study3_realnn/repeat_real_ours_moderate.yaml").read_text())
    assert rep["dataset"]["n_images"] == 5 and rep["dataset"]["rank_start"] == base["dataset"]["rank_start"]
    assert rep["experiment"]["output_dir"] == "results/study3_realnn/repeat_real_ours_moderate"
    assert _strip(base, extra=[("dataset", "n_images")]) == _strip(rep, extra=[("dataset", "n_images")])


def _problem():
    torch.manual_seed(0)
    x = torch.zeros(1, 1, 32, 32)
    x[..., 10:20, 12:22] = torch.rand(1, 1, 10, 10)
    return (torch.fft.fft2(torch.nn.functional.pad(x, (0, 32, 0, 32))).abs() ** 2).float()


def test_real_nonneg_starts_are_real_and_nonnegative():
    y = _problem()
    cfg = TrainerConfig(iterations=30, log_every=10, support_update_every=10, object_constraint="real_nonneg")
    r, _ = train_member(EnsembleMember(MemberSpec(out_h=32, out_w=32, seed=1)), y, cfg)
    assert torch.all(r.imag == 0)
    assert torch.all(r.real >= 0) and torch.all(r.real <= 1)


def test_default_object_model_is_complex():
    assert TrainerConfig().object_constraint == "complex"
    y = _problem()
    cfg = TrainerConfig(iterations=30, log_every=10, support_update_every=10)
    r, _ = train_member(EnsembleMember(MemberSpec(out_h=32, out_w=32, seed=1)), y, cfg)
    assert torch.any(r.imag != 0)


def test_unknown_object_model_fails():
    with pytest.raises(ValueError):
        train_member(EnsembleMember(MemberSpec(out_h=32, out_w=32, seed=1)), _problem(),
                     TrainerConfig(iterations=1, object_constraint="real"))
