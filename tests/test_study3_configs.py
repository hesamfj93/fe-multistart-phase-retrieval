"""The held-out configurations of the third study equal the development configurations except for
the fields STUDY3_PLAN.md allows to differ."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
DEV, HELD = REPO / "configs" / "study3_dev", REPO / "configs" / "study3"
UNTRAINED = ["real_ours_moderate", "real_ours_strong", "complex_ours_moderate", "complex_ours_strong",
             "real_netadm_none", "real_netadm_moderate", "real_netadm_strong"]
CLASSICAL = {"real_hio81_{lv}": ("real_classical_{lv}", ["hio"], ["real_nonneg"], 81),
             "complex_raar50_{lv}": ("complex_classical_{lv}", ["raar"], ["shrinkwrap"], 50)}


def load(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def strip(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    for k in ("name", "output_dir"):
        c["experiment"].pop(k, None)
    for k in ("protocol_split", "n_images", "rank_start"):
        c["dataset"].pop(k, None)
    return c


@pytest.mark.parametrize("name", UNTRAINED)
def test_untrained_heldout_equals_dev(name):
    h, d = load(HELD / f"{name}.yaml"), load(DEV / f"{name}.yaml")
    assert strip(h) == strip(d)
    assert h["dataset"]["protocol_split"] == "heldout3" and h["dataset"]["n_images"] == 50
    assert h["dataset"]["rank_start"] == (0 if name.startswith("real") else 100)
    assert h["experiment"]["output_dir"] == f"results/study3/{name}"


@pytest.mark.parametrize("lv", ["moderate", "strong"])
@pytest.mark.parametrize("pattern", list(CLASSICAL))
def test_classical_heldout_equals_dev(pattern, lv):
    dev_name, alg, con, starts = CLASSICAL[pattern]
    h, d = load(HELD / f"{pattern.format(lv=lv)}.yaml"), load(DEV / f"{dev_name.format(lv=lv)}.yaml")
    hb, db = h.pop("baselines"), d.pop("baselines")
    assert strip(h) == strip(d)
    assert hb["algorithms"] == alg and hb["constraints"] == con and hb["starts"] == starts
    for k in ("algorithms", "constraints", "starts"):
        hb.pop(k), db.pop(k)
    assert hb == db
    assert h["dataset"]["n_images"] == 50 and h["dataset"]["protocol_split"] == "heldout3"


def test_noise_levels_are_the_chosen_ones():
    chosen = json.loads((REPO / "results" / "study3_dev" / "noise_levels.json").read_text())["chosen_photon_budget"]
    for p in HELD.glob("*.yaml"):
        m = load(p)["measurement"]
        if "moderate" in p.stem:
            assert m["noise_model"] == "poisson" and m["noise_level"] == chosen["moderate"]
        elif "strong" in p.stem:
            assert m["noise_model"] == "poisson" and m["noise_level"] == chosen["strong"]
        else:
            assert m["noise_model"] == "none"


def test_equal_wall_time_numbers_match_the_timing_file():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s3r", REPO / "scripts" / "study3_report.py")
    s3r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s3r)
    k = json.loads((REPO / "results" / "study3_dev" / "timing.json").read_text())["k_equal_wall_time"]
    assert s3r.K_EQ == {"real_ours": k["real_ours_vs_hio_realpos"], "real_netadm": k["real_netadm_vs_hio_realpos"],
                        "complex_ours": k["complex_ours_vs_raar_sw"]}
