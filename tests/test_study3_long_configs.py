"""The configurations of the compute-matched control (STUDY3_LONG_PLAN.md) differ from those of
Study 3 only in the keys the plan names."""
import copy
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
RUNS = ("real_ours_moderate", "real_ours_strong", "complex_ours_moderate", "complex_ours_strong",
        "real_netadm_moderate", "real_netadm_strong")


@pytest.mark.parametrize("run", RUNS)
def test_long_config_matches_study3(run):
    base = yaml.safe_load((REPO / "configs/study3" / f"{run}.yaml").read_text())
    long_ = yaml.safe_load((REPO / "configs/study3_long" / f"{run}.yaml").read_text())
    assert long_["ensemble"]["K"] == 1
    assert long_["ensemble"]["base_seed"] == base["ensemble"]["base_seed"] == 0
    assert long_["experiment"]["single_method_name"] == "k1_long"
    assert long_["experiment"]["output_dir"] == f"results/study3_long/{run}"
    if "netadm" in run:
        assert long_["solver"]["netadm"]["outer_iters"] == 5 * base["solver"]["netadm"]["outer_iters"]
        assert long_["solver"]["netadm"]["lr_decay_every"] == 5 * base["solver"]["netadm"]["lr_decay_every"]
    else:
        assert long_["trainer"]["iterations"] == 5 * base["trainer"]["iterations"]
    a, b = copy.deepcopy(base), copy.deepcopy(long_)
    for c in (a, b):
        for k in ("name", "output_dir", "single_method_name"):
            c["experiment"].pop(k, None)
        c["ensemble"].pop("K")
        c["trainer"].pop("iterations")
        if "solver" in c:
            c["solver"]["netadm"].pop("outer_iters")
            c["solver"]["netadm"].pop("lr_decay_every")
    assert a == b
