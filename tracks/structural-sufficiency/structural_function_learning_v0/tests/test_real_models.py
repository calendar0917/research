"""Slow integration checks against the artefacts produced by the round.

They are skipped when the corresponding output does not exist (fresh checkout), and
can be deselected with ``-m 'not slow'``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import sfl_common as K  # noqa: E402

pytestmark = pytest.mark.slow


def _need(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"missing artefact {p}")


def test_add_tree_boosters_are_singleton():
    import json

    sys.path.insert(0, str(ROOT / "code"))
    from phaseB_add_tree import json_tree_stats

    import xgboost as xgb

    for arm in ("H-ADD-TREE", "Z-ADD-TREE"):
        path = K.MODELS / f"{arm}.json"
        _need(path)
        booster = xgb.Booster()
        booster.load_model(str(path))
        _, stats = json_tree_stats(booster)
        assert stats["n_trees_violating_singleton"] == 0
        assert stats["max_distinct_features_per_tree"] <= 1
        run = K.read_json(K.RUNS / f"{arm}.json")
        assert run["replay_check"]["residual_max_abs_dev_from_mean"] < 1e-4
        assert run["tree_violations"] == {"multi_feature_trees": 0,
                                          "size_and_structure_same_tree": 0,
                                          "two_structures_same_tree": 0}


def test_add_nn_f0_and_closure():
    import torch

    from phaseC_add_nn import AdditiveNN

    for arm in ("H-NN-ADD", "Z-NN-ADD"):
        path = K.RUNS / f"{arm}_best.pt"
        if not path.exists():
            path = K.MODELS / f"{arm}_best.pt"
        _need(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        p = sd["a"].shape[0]
        model = AdditiveNN(np.zeros((p, 4), dtype=np.float32), np.ones(p, dtype=bool))
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            f0 = model.feature_values(torch.zeros(4, p))
            assert float(f0.abs().max()) == 0.0
            rng = np.random.default_rng(0)
            x = torch.from_numpy(rng.normal(size=(16, p)).astype(np.float32))
            closure = model.bias + model.feature_values(x).sum(dim=1)
            assert torch.allclose(model(x), closure, atol=1e-6)
        checks = ckpt["additive_checks"]
        assert checks["f_j_at_zero_max_abs"] == 0.0
        assert checks["forward_minus_feature_sum_max_abs"] < 1e-5
        assert checks["dense_forward_vs_numpy_max_abs"] < 1e-4


def test_hiv_mlp_initialisation_hashes_match():
    _need(K.ANALYSIS / "phaseC_H-NN-JOINT.json",
          K.ANALYSIS / "phaseC_H-NN-JOINT-BAL.json")
    a = K.read_json(K.ANALYSIS / "phaseC_H-NN-JOINT.json")
    b = K.read_json(K.ANALYSIS / "phaseC_H-NN-JOINT-BAL.json")
    assert a["init_hash"] == b["init_hash"]
    assert a["parameters"] == b["parameters"]


def test_metrics_csv_matches_stored_predictions():
    import pandas as pd

    _need(K.ANALYSIS / "metrics.csv", K.PREDICTIONS / "H-ADD-TREE.npz")
    table = pd.read_csv(K.ANALYSIS / "metrics.csv")
    d = K.HIVData()
    z = np.load(K.PREDICTIONS / "H-ADD-TREE.npz")
    m = K.hiv_metrics(z["logit"], d.y, d.rows)
    for split in K.SPLITS:
        row = table[(table.model == "H-ADD-TREE") & (table.split == split)
                    & (table.selected_setting == "rounds=500")]
        assert len(row) == 1
        assert abs(float(row.roc_auc.iloc[0]) - m[split]["roc_auc"]) < 1e-12
        assert abs(float(row.average_precision.iloc[0])
                   - m[split]["average_precision"]) < 1e-12


def test_zinc_input_identity_against_historical_predictions():
    _need(K.SGA_RUNS / "phaseC" / "ZN1_pred.npz",
          K.SGA_RUNS / "phaseC" / "ZX1_pred.npz")
    d = K.ZincData()
    for arm in ("ZN1", "ZX1"):
        z = np.load(K.SGA_RUNS / "phaseC" / f"{arm}_pred.npz")
        assert np.array_equal(z["y"], d.y)
        for split in K.SPLITS:
            assert np.array_equal(z[f"{split}_rows"], d.rows[split])


def test_hiv_input_identity_against_historical_predictions():
    _need(K.ISRD_PRED / "COUNT-ALL.npz")
    d = K.HIVData()
    z = np.load(K.ISRD_PRED / "COUNT-ALL.npz")
    assert np.array_equal(z["y"], d.y)
    assert np.array_equal(z["graph_id"], np.arange(d.n))
    assert np.array_equal(z["split_code"], d.split_code())
    s = K.read_json(K.ANALYSIS / "phase0_contributions.json")
    assert s["reconstruction_max_abs_error_zero_baseline"] < 1e-8
    assert s["reconstruction_max_abs_error_mean_centred"] < 1e-8
    assert s["reconstruction_max_abs_error_old_buggy"] > 1.0


def test_phaseE_tree_and_nn_closures():
    _need(K.FUNCTIONS / "hiv_functions.json", K.FUNCTIONS / "zinc_functions.json")
    h = K.read_json(K.FUNCTIONS / "hiv_functions.json")
    assert h["closure"]["tree_extraction_max_abs_error"] < 1e-4
    assert h["closure"]["nn_forward_vs_sum_features_max_abs_error"] < 1e-4
    assert h["closure"]["nn_f_j_at_zero_max_abs_sampled"] < 1e-6
    for e in h["selected_structures"]:
        assert e["f_tree"][0] == 0.0 and e["f_nn"][0] == 0.0
    z = K.read_json(K.FUNCTIONS / "zinc_functions.json")
    assert z["closure"]["tree_extraction_max_abs_error"] < 1e-3
