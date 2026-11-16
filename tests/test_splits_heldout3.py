"""The third-study split: continuation of the held-out permutation, disjoint from all others."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pr_uq.data.splits import (DEV_INDICES, HELDOUT3_LENGTH, HELDOUT3_START, HELDOUT_PERMUTATION_SEED,
                               MNIST_TEST_SIZE, generate_split_lists, load_protocol_indices, read_split_csv,
                               split_path)

REPO = Path(__file__).resolve().parent.parent


def test_heldout3_continues_the_heldout_permutation():
    perm = np.random.RandomState(HELDOUT_PERMUTATION_SEED).permutation(MNIST_TEST_SIZE).tolist()
    pool = [i for i in perm if i not in set(DEV_INDICES)]
    lists = generate_split_lists()
    assert lists["heldout"] == pool[:200]
    assert lists["heldout3"] == pool[HELDOUT3_START:HELDOUT3_START + HELDOUT3_LENGTH]


def test_heldout3_file_matches_and_is_disjoint():
    lists = generate_split_lists()
    on_disk = {r: read_split_csv(split_path(REPO / "splits", r)) for r in ("dev", "heldout", "heldout3", "smoke")}
    assert on_disk["heldout3"] == lists["heldout3"] and len(on_disk["heldout3"]) == 200
    for other in ("dev", "heldout", "smoke"):
        assert not set(on_disk["heldout3"]) & set(on_disk[other])


def test_heldout3_protocol_selection_rows():
    sel = load_protocol_indices(REPO / "splits", "heldout3", 100, start=100)
    assert sel == read_split_csv(split_path(REPO / "splits", "heldout3"))[100:200]
