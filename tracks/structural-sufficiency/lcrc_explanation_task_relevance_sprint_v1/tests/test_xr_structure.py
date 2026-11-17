"""Fast structural tests for the explanation-relevance sprint.

These tests read the committed frozen assets only: no GPU, no model checkpoint,
no full graph cache.  The GPU/identity gates live in ``code/xr_gates.py``.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "code"))

import xr_common as C  # noqa: E402
import xr_assets as A  # noqa: E402
import xr_discover as D  # noqa: E402


def test_pair_support_matches_frozen_catalogue():
    pairs = A.load_pairs()
    support = A.pair_support(pairs)
    assert len(pairs) == 823
    assert support["split_pair_counts"] == {"train-test": 27, "train-train": 791, "train-valid": 5}
    assert support["n_components"] == 231
    assert support["strata"]["train_train"]["n_pairs"] == 791
    assert support["strata"]["train_train"]["n_components"] == 228
    assert support["strata"]["at_least_one_nontrain"]["n_pairs"] == 32
    assert support["strata"]["at_least_one_nontrain"]["n_components"] == 11
    assert support["strata"]["both_nontrain"]["support"] == "NO_SUPPORT"
    assert support["split_pair_strata"]["train-test"]["n_pairs"] == 27
    assert support["split_pair_strata"]["train-valid"]["n_pairs"] == 5


def test_edit_regions_are_internally_consistent():
    endpoints = C.read_jsonl_gz(C.PAIR_ENDPOINTS)
    assert len(endpoints) == 1646
    for ep in endpoints[:50]:
        atoms = set(int(v) for v in ep["edit_atoms"])
        bonds = set(tuple(sorted(int(x) for x in b)) for b in ep["edit_bonds"])
        assert atoms == {u for b in bonds for u in b}
        assert len(atoms) == int(ep["edit_atom_count"])
        assert len(bonds) == int(ep["edit_bond_count"])
        assert int(ep["attach_atom"]) in atoms
        assert set(tuple(sorted(int(x) for x in b)) for b in ep["attach_bonds"]) <= bonds
        assert set(tuple(sorted(int(x) for x in b)) for b in ep["changed_core_bonds"]) <= bonds


def test_oracle_pair_contribution_closure():
    pairs = A.load_pairs()
    oracle = A.load_oracle()
    for pair in pairs[:50]:
        delta = A.contribution_delta(oracle, pair)
        assert delta["closed"], (pair["pair_id"], delta["closure_error"], delta["closure_tolerance"])
        assert delta["graph_closure_ok"]
        va = [int(v) for v in pair["var_atoms_a"]]
        vb = [int(v) for v in pair["var_atoms_b"]]
        # delta_a omits B-only atoms; delta_b omits A-only atoms, so neither
        # endpoint vector alone equals the pair union.  The two identities below
        # pin down the omitted variable-fragment mass.
        assert sum(delta["delta_a"]) == pytest.approx(delta["wc_delta"] - sum(delta["delta_b"][j] for j in vb))
        assert sum(delta["delta_b"]) == pytest.approx(delta["wc_delta"] - sum(delta["delta_a"][i] for i in va))


def test_mass_capture_basic():
    delta = [1.0, -2.0, 0.5, -0.5]
    assert D.mass_capture(delta, [0]) == pytest.approx(1.0 / 4.0)
    assert D.mass_capture(delta, [0, 1]) == pytest.approx(3.0 / 4.0)
    assert D.mass_capture([0.0, 0.0], [0]) is None
    assert D.mass_capture(delta, []) == 0.0


def test_component_bootstrap_resamples_components_not_pairs():
    values = {"x": np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])}
    components = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    out = C.component_bootstrap(values, components, lambda v: {"x": float(np.mean(v["x"]))}, n_boot=100, seed=20261117)
    assert out["n_pairs"] == 6
    assert out["n_components"] == 3
    assert out["ci95"]["x"][0] <= out["point"]["x"] <= out["ci95"]["x"][1]
    assert len(out["boot"]["x"]) == 100


def test_discovery_score_and_rank_rule():
    mods = C.source_modules()
    AS = mods["as_search"]
    grad = np.asarray([1.0, 2.0, -3.0], dtype=np.float64)
    assert AS.score_discovery([0, 1], grad) == pytest.approx(3.0 / math.sqrt(2.0))
    assert AS.score_discovery([], grad) == 0.0
    assert AS.rank_key(2.0, [4, 5]) < AS.rank_key(2.0, [4, 5, 6])
    assert AS.rank_key(2.0, [4, 5]) < AS.rank_key(1.0, [4, 5])


def test_discovery_api_is_label_blind():
    sig = inspect.signature(D.discover_from_gradient)
    assert list(sig.parameters) == ["mol", "grad_raw", "score_name"]
    src = inspect.getsource(D.discover_from_gradient)
    for forbidden in ("y_a", "y_b", "label", "oracle", "edit_atoms", "pair_id"):
        assert forbidden not in src
