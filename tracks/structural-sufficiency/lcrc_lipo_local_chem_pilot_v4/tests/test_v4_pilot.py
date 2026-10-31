"""Fast, write-free checks for the v4 local-chemistry pilot.

The full audit suite lives in ``v4_audit.py`` and is run through
``run_v4.py cpu-gates``.  These tests are the cheap CI layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import v4_atomchem as A
import v4_common as C
import v4_evaluate as EV
import v4_models as M
import v4_registry as R


@pytest.fixture(scope="module")
def fx():
    return A.build_fold_extras(C.FOLD)


@pytest.fixture(scope="module")
def view():
    cache = C.load_fit_cache()
    folds = C.F.load_folds()
    y = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    return C.D.build_fold_view(C.FOLD, cache=cache["cache"], y_fit=y, folds=folds)


def test_config_matches_frozen_constants():
    cfg = C.load_config()
    assert tuple(cfg["atomchem"]["names"]) == C.ATOMCHEM_NAMES
    assert cfg["atomchem"]["atom_dim_base"] == 51
    assert cfg["atomchem"]["atom_dim_augmented"] == 55
    assert cfg["decision"]["materiality_rmse"] == 0.03
    assert cfg["budget"]["gpu_hard_cap_hours"] == 0.20
    assert cfg["comparator"]["key"] == C.HISTORICAL["key"]


def test_parameter_delta_arithmetic():
    assert C.N_ATOM_EXTRA == 4
    assert C.EXPECTED_PARAMETER_DELTA == 4 * C.ATOM_HIDDEN == 512


def test_augmented_spec_is_frozen():
    spec = A.augmented_spec()
    base = C.load_frozen_spec()
    assert spec.hash() == "cc893a7f0dbaee43bc101705cca4d0d0b39dbb09d4463dd0e39b82ecdcee11c8"
    assert spec.atom_feat_dim == 55
    assert base.atom_feat_dim == 51
    assert spec.bond_feat_dim == base.bond_feat_dim
    assert spec.n_scalars == base.n_scalars
    assert spec.global_dim == base.global_dim


def test_extras_shapes_and_train_only_standardisation(fx):
    assert fx.train_std.shape == (42198, 4)
    assert fx.dev_std.shape == (21033, 4)
    assert fx.raw_fit.shape[0] == fx.train_std.shape[0] + fx.dev_std.shape[0]
    assert np.isfinite(fx.train_std).all() and np.isfinite(fx.dev_std).all()
    assert len(fx.constant_indices) == 0
    std = fx.audit["standardisation"]
    assert std["fit_scope"] == "fold1_train_atoms_only"
    assert std["dev_used_for_fit"] is False
    assert abs(std["train_mean_after_standardisation_max_abs"]) < 1e-9
    assert abs(std["train_std_after_standardisation_max_abs_deviation"]) < 1e-9


def test_broadcast_matches_per_molecule_mean(fx):
    for split, ptr, true_arr, bcast in (
        ("train", fx.train_mol_ptr, fx.train_std, fx.train_broadcast_std),
        ("dev", fx.dev_mol_ptr, fx.dev_std, fx.dev_broadcast_std),
    ):
        for m in range(len(ptr) - 1):
            i0, i1 = int(ptr[m]), int(ptr[m + 1])
            assert np.all(bcast[i0:i1] == bcast[i0]), split
            assert np.allclose(bcast[i0], true_arr[i0:i1].mean(axis=0), rtol=0.0, atol=1e-6), split


def test_broadcast_contrast_is_not_degenerate(fx):
    dev1 = A.atom_arrays_from_fold_extras(fx, "atomchem4")
    dev2 = A.atom_arrays_from_fold_extras(fx, "broadcastchem4")
    assert float(np.abs(dev1["dev"] - dev2["dev"]).max()) > 1e-3


def _batch(view, fx, variant, n=8):
    extras = A.atom_arrays_from_fold_extras(fx, variant)
    return extras, M.collate_with_atom_extra(view.dev_cache, list(range(n)), extras)


def test_zero_extra_forward_equivalence(view, fx):
    extras, batch = _batch(view, fx, "atomchem4")
    model, meta = M.build_atomchem_model("A_AtomChem4", extras["spec"], view.train_norm, C.SEED, "cpu", dropout=0.0)
    base_batch = C.D.collate(view.dev_cache, list(range(8)))
    zero = dict(batch)
    zero["atom_feat"] = torch.cat(
        [batch["atom_feat"][:, :51], torch.zeros(batch["atom_feat"].shape[0], 4)], dim=1
    )
    model.eval()
    with torch.no_grad():
        y0 = model(zero).cpu().numpy()
        y1 = meta["shared_with_original"](base_batch).cpu().numpy()
    assert np.allclose(y0, y1, rtol=0.0, atol=1e-6)
    assert meta["parameter_delta"] == 512
    assert meta["migration"]["new_columns_initialised_to_zero"]


def test_new_columns_receive_gradient(view, fx):
    extras, batch = _batch(view, fx, "atomchem4")
    model, _ = M.build_atomchem_model("A_AtomChem4", extras["spec"], view.train_norm, C.SEED, "cpu", dropout=0.0)
    model.train()
    loss = (model(batch) ** 2).mean()
    loss.backward()
    grad = model.atom_encoder.lins[0].weight.grad
    assert grad is not None
    new = grad[:, 51:]
    assert torch.isfinite(new).all()
    assert int(torch.count_nonzero(new)) == 512


def test_registry_is_exactly_p1_plus_conditional_p2():
    reg = R.planned_registry()
    assert reg["n_runs"] == 2
    assert reg["n_unconditional"] == 1
    assert reg["n_conditional"] == 1
    assert R.unconditional_keys() == ["F1_A_AtomChem4_R3_s0"]
    assert R.conditional_key() == "F1_A_BroadcastChem4_R3_s0"
    assert reg["runs"][1]["conditional"] is True


def test_tail_gate_flags_broad_gain_as_not_tail_dependent():
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    hist = y + rng.normal(scale=0.90, size=400)
    new = y + rng.normal(scale=0.60, size=400)
    out = EV.tail_analysis(y, hist, new)
    assert out["delta_full"] > 0
    assert out["delta_drop_top_tail"] > 0
    assert out["n_dropped"] == 8
    assert out["retained_fraction"] >= 0.5
    assert out["not_tail_dependent"]


def test_tail_gate_flags_single_outlier_rescue_as_tail_dependent():
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    hist = y + rng.normal(scale=0.05, size=400)
    hist[0] = y[0] + 30.0  # one catastrophic historical failure
    new = y + rng.normal(scale=0.60, size=400)
    out = EV.tail_analysis(y, hist, new)
    assert out["delta_full"] > 0
    assert not out["not_tail_dependent"]


def test_p1_gate_requires_materiality_mae_and_tail():
    y = np.linspace(-3.0, 5.0, 300)
    rng = np.random.default_rng(2)
    hist = y + rng.normal(scale=0.80, size=300)
    base = {
        "key": "h",
        "y": y,
        "pred": hist,
        "train_rmse": 0.0,
        "train_mae": 0.0,
        "dev_rmse": EV.rmse(y, hist),
        "dev_mae": EV.mae(y, hist),
        "gap": 0.0,
    }
    def _cmp(scale):
        new = y + rng.normal(scale=scale, size=300)
        other = {
            "key": "n",
            "y": y,
            "pred": new,
            "train_rmse": 0.0,
            "train_mae": 0.0,
            "dev_rmse": EV.rmse(y, new),
            "dev_mae": EV.mae(y, new),
            "gap": 0.0,
        }
        return EV.compare(base, other, label="synthetic")

    strong = EV.p1_gate(_cmp(0.35))
    assert strong["met"] and strong["status"] == "P1_GATE_PASSED"

    weak = EV.p1_gate(_cmp(0.79))
    assert not weak["met"] and weak["status"] == "NO_CLEAR_LOCAL_CHEM_GAIN"


def test_verdict_requires_broadcast_gap_for_alignment_claim():
    gate_pass = {"delta_rmse": 0.06, "delta_mae": 0.02, "tail": {"not_tail_dependent": True}}
    gate_fail = {"delta_rmse": 0.01, "delta_mae": 0.02, "tail": {"not_tail_dependent": True}}
    aligned = EV.decide(gate_pass, {"delta_rmse": 0.01, "delta_mae": 0.0})
    assert aligned["verdict"] == "LOCAL_CHEM_ALIGNMENT_PROMISING"
    info = EV.decide(gate_pass, {"delta_rmse": 0.055, "delta_mae": 0.0})
    assert info["verdict"] == "CHEMICAL_FEATURE_INFORMATION_PROMISING"
    stopped = EV.decide(gate_fail, None)
    assert stopped["verdict"] == "NO_CLEAR_LOCAL_CHEM_GAIN"


def test_evaluate_deltas_use_historical_minus_new():
    y = np.array([0.0, 1.0, 2.0])
    hist = np.array([0.2, 1.2, 2.2])
    new = np.array([0.1, 1.1, 2.1])
    a = {"key": "h", "y": y, "pred": hist, "train_rmse": 0.0, "train_mae": 0.0, "dev_rmse": EV.rmse(y, hist), "dev_mae": EV.mae(y, hist), "gap": 0.0}
    b = {"key": "n", "y": y, "pred": new, "train_rmse": 0.0, "train_mae": 0.0, "dev_rmse": EV.rmse(y, new), "dev_mae": EV.mae(y, new), "gap": 0.0}
    cmp = EV.compare(a, b, label="synthetic")
    assert cmp["delta_rmse"] == pytest.approx(0.1, abs=1e-12)
    assert cmp["delta_mae"] == pytest.approx(0.1, abs=1e-12)
    assert not cmp["bootstrap_rmse"]["ci_includes_zero"]


def test_historical_constants_and_files_are_pinned():
    assert C.HISTORICAL["dev_rmse"] == 0.7403427006133139
    assert C.HISTORICAL["dev_mae"] == 0.546694119695902
    for rel, sha in C.V2_HISTORICAL_FILES.items():
        assert C.sha256_file(C.REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2" / rel) == sha


def test_v1_v2_v3_frozen():
    assert C.assert_v1_unchanged()["ok"]
    assert C.assert_v2_unchanged()["ok"]
    assert C.assert_v3_unchanged(use_manifest=True)["ok"]
