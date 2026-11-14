"""Focused tests for ``lcrc-lipo-context-small-s0-v1`` (CPU, no GPU, no network).

These cover the frozen contract that the formal runs depend on: the committed
data identity, the shared initial state / parameter budget, the epoch-0
equivalence, the arm-specific source reach and the shared epoch plan.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import lp_common as C
import lp_data as D
import lp_model as M
import lp_preflight as P
import lp_source as S

import dissect_model as DM


@pytest.fixture(scope="module")
def view() -> D.View:
    return D.build_view()


def test_frozen_recipe_matches_upstream_r3() -> None:
    import dissect_common as DC

    assert DC.RECIPE["epochs"] == C.C1_RECIPE["epochs"] == 360
    assert DC.RECIPE["peak_lr"] == C.C1_RECIPE["peak_lr"]
    assert DC.RECIPE["min_lr"] == C.C1_RECIPE["min_lr"]
    assert DC.RECIPE["weight_decay"] == C.C1_RECIPE["weight_decay"]
    assert DC.RECIPE["batch_size"] == C.C1_RECIPE["batch_size"]
    assert DC.RECIPE["shuffle_offset"] == C.SHUFFLE_OFFSET == 917_000


def test_committed_artifacts_and_labels() -> None:
    manifest = C.read_json(C.DATA_MANIFEST)
    for name, entry in manifest["artifacts"].items():
        path = C.REPO / entry["path"]
        assert path.is_file(), name
        assert C.sha256_file(path) == entry["sha256"], name
    labels = S.load_labels()
    assert labels["y"].shape == (C.N_MOLECULES,)
    split = S.load_split()
    assert split["sizes"] == dict(C.SPLIT_SIZES)
    union = np.concatenate([split["rows"][k] for k in C.SPLITS])
    assert np.array_equal(np.sort(union), np.arange(C.N_MOLECULES))


def test_shared_initial_state_and_parameter_budget(view: D.View) -> None:
    full = DM.build_full_state(view.spec, view.train_norm, int(C.TRAIN_SEED))
    c1 = DM.remap_state("C1", full)
    states = {}
    for arm in ("B", *C.CONTEXT_ARMS):
        model = M.build_arm(view.spec, view.train_norm, arm)
        model.base.load_state_dict(c1, strict=True)
        assert M.parameter_count(model) == C.EXPECTED_PARAMS[arm], arm
        states[arm] = D.state_hash(model.base.state_dict())
    # the frozen C1 base is identical in every arm
    reference = D.state_hash(dict(c1))
    assert states["B"] == reference
    assert len(set(states.values())) == 1


def test_zero_residual_initialisation_equivalence(view: D.View) -> None:
    batch = D.to_device(D.PREP_DATA.collate(view.caches["train"], [0, 1, 2]), torch.device("cpu"))
    c1 = DM.remap_state("C1", DM.build_full_state(view.spec, view.train_norm, 0))
    base = DM.DissectLCRC(view.spec, view.train_norm, "C1")
    base.load_state_dict(c1, strict=True)
    base.eval()
    with torch.no_grad():
        reference = base(batch)
    for arm in C.CONTEXT_ARMS:
        model = M.build_arm(view.spec, view.train_norm, arm)
        model.base.load_state_dict(c1, strict=True)
        model.init_new_modules()
        model.eval()
        with torch.no_grad():
            got = model(batch)
        assert float((got - reference).abs().max()) <= C.ZERO_DRIFT_TOL, arm
        assert sum(p.numel() for _, p in model.new_parameters()) == C.NEW_PARAMS


def test_source_reach_on_chain(view: D.View) -> None:
    payload = P.section_b(view)
    assert payload["passed"], payload
    assert payload["S"]["reads_B"] is False
    assert payload["R1"]["reads_A"] is False and payload["R1"]["reads_B"] is True
    assert payload["D2"]["reads_A"] is True


def test_zero_bond_molecule_is_finite(view: D.View) -> None:
    zero = {
        "src": torch.zeros(0, dtype=torch.long),
        "dst": torch.zeros(0, dtype=torch.long),
        "bond_feat": torch.zeros((0, C.BOND_FEAT_DIM)),
        "deg": torch.zeros(1),
        "n_atoms": 1,
    }
    h0 = torch.ones(1, C.WIDTH)
    for arm in C.CONTEXT_ARMS:
        model = P._arm_model(view, arm, randomise=True)
        layers = model.message_layers(h0, zero, model.spec["source"])
        assert torch.isfinite(layers["h2"]).all(), arm
        # with the pre-registered zero initialisation the isolated atom is a no-op
        fresh = P._arm_model(view, arm)
        fresh_layers = fresh.message_layers(h0, zero, fresh.spec["source"])
        assert float((fresh_layers["h2"] - h0).abs().max()) <= C.ZERO_DRIFT_TOL, arm


def test_shared_epoch_plan_is_identical_across_arms(view: D.View) -> None:
    n = int(view.caches["train"]["n_molecules"])
    first = D.epoch_plan(n, C.EPOCHS["B"], int(C.TRAIN_SEED))
    second = D.epoch_plan(n, C.EPOCHS["B"], int(C.TRAIN_SEED))
    assert first["sha256"] == second["sha256"]
    plan = first["plan"]
    assert plan.shape == (C.EPOCHS["B"], n)
    for epoch in range(2):
        assert sorted(plan[epoch].tolist()) == list(range(n))


def test_target_normalisation_is_train_only(view: D.View) -> None:
    labels = S.load_labels()
    split = S.load_split()
    train = labels["y"][split["rows"]["train"]]
    assert view.target_norm["mean"] == pytest.approx(float(train.mean()))
    assert view.target_norm["scale"] == pytest.approx(float(train.std()))
    assert view.target_norm["valid_or_test_used"] is False
    assert view.train_norm["valid_or_test_used"] is False


def test_dmpnn_feature_archive_alignment() -> None:
    with np.load(C.DMPNN_FEATURES, allow_pickle=False) as handle:
        assert int(handle["n_molecules"]) == C.N_MOLECULES
        assert int(handle["atom_fdim"]) == 72 and int(handle["bond_fdim"]) == 14
        ptr = np.asarray(handle["mol_atom_ptr"])
        assert ptr.shape == (C.N_MOLECULES + 1,)
        assert int(ptr[-1]) == int(handle["n_atoms"])


def test_analysis_metrics_are_consistent() -> None:
    a = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([1.5, 1.0, 4.0])
    import lp_analysis as A

    assert A.rmse(a, y) == pytest.approx(math.sqrt(((a - y) ** 2).mean()))
    assert A.mae(a, y) == pytest.approx(np.abs(a - y).mean())
    weights = np.asarray([1.0, 2.0, 1.0])
    assert A.rmse(a, y, weights) == pytest.approx(math.sqrt(float(np.sum(weights * (a - y) ** 2) / weights.sum())))
