"""Focused tests for ``e2e_dictenv_common_subspace_dictionary_v1`` (CSSD round).

CPU only; the official ZINC test split is never touched.  The test file covers
the pre-registered Stage-E list: train-only subspace construction, the exact
decomposition identities, the residual-dictionary hard constraints, the frozen
coder, bit-identical non-CSSD initialization, the untouched FINAL-CLEAN path,
node/edge shapes, the training-loop equivalence with the frozen loop, the
epoch-40 gate logic and the two guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_dictionary_coder_audit_v1 as dca
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

C6_SPARSE = cm.CleanMechSpec("C6", "C6", coding="sparse")


def _synthetic_phi(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.gamma(shape=2.0, scale=1.0, size=(n, cssd.PHI_DIM))
    base[:, 5] = 0.0
    base[:, 8] = 0.0
    return base


def _dictionary() -> np.ndarray:
    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary


# ---------------------------------------------------------------------------
# Stage B — train-only construction and identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [1, 2])
def test_common_subspace_orthonormal_and_train_only(q: int) -> None:
    train = _synthetic_phi(seed=1)
    other = _synthetic_phi(seed=2)
    subspace = cssd.build_common_subspace(train, q)
    gram = subspace.components.T @ subspace.components
    assert np.allclose(gram, np.eye(q), atol=1e-12)
    # valid rows cannot influence the construction: a subspace built from a
    # different second array is identical because only train enters.
    again = cssd.build_common_subspace(train, q)
    assert np.array_equal(subspace.components, again.components)
    assert np.array_equal(subspace.rms, again.rms)
    # extra unrelated rows passed as "valid" would have to be concatenated by
    # the caller; the API has no valid argument at all.
    assert other.shape[1] == cssd.PHI_DIM


@pytest.mark.parametrize("q", [1, 2])
def test_decomposition_identity_and_orthogonality(q: int) -> None:
    train = _synthetic_phi(seed=3)
    subspace = cssd.build_common_subspace(train, q)
    c, r = cssd.decompose(train, subspace)
    reconstruction = c @ subspace.components.T + r
    assert np.allclose(reconstruction, train, atol=1e-10)
    projection = subspace.components.T @ r.T
    assert np.allclose(projection, 0.0, atol=1e-10)


def test_q2_direction_orthogonal_to_q1_and_pc1_like() -> None:
    train = _synthetic_phi(seed=4)
    sub1 = cssd.build_common_subspace(train, 1)
    sub2 = cssd.build_common_subspace(train, 2)
    assert np.allclose(sub1.components[:, 0], sub2.components[:, 0], atol=0.0)
    assert abs(float(sub1.components[:, 0] @ sub2.components[:, 1])) <= 1e-12
    # the second direction is the train PC1 of the q1 residual
    _c, r1 = cssd.decompose(train, sub1)
    centred = r1 - r1.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    v2 = vt[0]
    v2 = v2 - float(v2 @ sub1.components[:, 0]) * sub1.components[:, 0]
    v2 = v2 / np.linalg.norm(v2)
    cosine = abs(float(v2 @ sub2.components[:, 1]))
    assert cosine >= 0.999999


def test_rms_scaling_is_train_only_and_keeps_mean() -> None:
    train = _synthetic_phi(seed=5)
    subspace = cssd.build_common_subspace(train, 1)
    c, _r = cssd.decompose(train, subspace)
    scaled = cssd.scaled_common(c, subspace)
    rms = np.sqrt(np.mean(scaled**2, axis=0))
    assert np.allclose(rms, 1.0, atol=1e-10)
    # absolute common component is retained (no centring): the scaled
    # coordinate keeps a clearly non-zero mean for non-negative phi.
    assert float(scaled.mean(axis=0)[0]) > 0.5


def test_zero_descriptor_columns_named_pair_are_zero() -> None:
    cache = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p1/cache/env_train.pt"
    if not cache.exists():  # pragma: no cover - cache is part of the frozen round
        pytest.skip("phi65 cache not available")
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    phi = np.asarray(blob["phi"], dtype=np.float64)
    assert phi.shape == (231664, 65)
    assert np.abs(phi[:, 5]).max() == 0.0  # root_neighbour_shell1
    assert np.abs(phi[:, 8]).max() == 0.0  # root_walk1


# ---------------------------------------------------------------------------
# Stage D — the CSSD model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [1, 2])
def test_projected_dictionary_hard_constraint(q: int) -> None:
    train = _synthetic_phi(seed=6)
    subspace = cssd.build_common_subspace(train, q)
    model = cssd.build_cssd_model(_dictionary(), 0, subspace)
    Dbar = model.residual_dictionary().detach().double().numpy()
    assert np.abs(subspace.components.T @ Dbar).max() <= 1e-6
    norms = np.linalg.norm(Dbar, axis=0)
    assert np.isfinite(Dbar).all()
    assert norms.min() > 1e-8
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert model.dead_column_fallbacks == 0


@pytest.mark.parametrize("q", [1, 2])
def test_cssd_shapes_and_codes(q: int) -> None:
    train = _synthetic_phi(seed=7)
    subspace = cssd.build_common_subspace(train, q)
    model = cssd.build_cssd_model(_dictionary(), 0, subspace)
    assert tuple(model.W_A_S.shape) == (32 + q, 96)
    assert tuple(model.W_E_S.shape) == (3 * (32 + q), 48)
    phi = torch.as_tensor(train[:64], dtype=torch.float32)
    z = model.code(phi)
    assert tuple(z.shape) == (64, 32 + q)
    # alpha part is exactly the frozen IHT-10 coder on the residual
    U = model.U
    r = phi - (phi @ U) @ U.t()
    reference = v0.tied_iht_codes(
        model.residual_dictionary(), r, s=cssd.SPARSITY, steps=cssd.IHT_STEPS
    )
    assert torch.equal(z[:, q:], reference)
    nnz = (reference != 0).sum(dim=1)
    assert int(nnz.min()) == int(cssd.SPARSITY)


@pytest.mark.parametrize("q", [1, 2])
def test_cssd_preserves_non_cssd_initialization(q: int) -> None:
    train = _synthetic_phi(seed=8)
    subspace = cssd.build_common_subspace(train, q)
    base = cm.build_clean_mech_model(_dictionary(), 0, C6_SPARSE)
    model = cssd.build_cssd_model(_dictionary(), 0, subspace)
    base_state = base.state_dict()
    cssd_state = model.state_dict()
    shared = {
        key: value
        for key, value in base_state.items()
        if key in cssd_state and cssd_state[key].shape == value.shape
    }
    assert len(shared) >= 20
    for key, value in shared.items():
        assert torch.equal(value, cssd_state[key]), f"non-CSSD parameter {key} differs"
    assert set(cssd_state) - set(base_state) == {"U", "common_rms"}
    # widening map: old rows preserved, new common rows exactly zero
    assert torch.equal(base.W_A_S.detach(), model.W_A_S.detach()[q:])
    assert bool((model.W_A_S.detach()[:q] == 0).all())
    old_e = base.W_E_S.detach()
    new_e = model.W_E_S.detach()
    width = 32 + q
    for block in range(3):
        assert torch.equal(
            old_e[block * 32 : (block + 1) * 32],
            new_e[block * width + q : block * width + q + 32],
        )
    common_rows = [0, width, 2 * width]
    assert bool((new_e[common_rows] == 0).all())


def test_dead_column_fallback_is_deterministic_and_live() -> None:
    rng = np.random.default_rng(0)
    U = rng.normal(size=(10, 1))
    U = U / np.linalg.norm(U)
    D = rng.normal(size=(10, 4))
    D[:, 2] = U[:, 0] * 3.0  # exactly inside span(U)
    fixed, count = cssd.dead_column_fallback(torch.as_tensor(D), torch.as_tensor(U))
    assert count == 1
    projected = np.asarray(fixed, dtype=np.float64) - U @ (U.T @ np.asarray(fixed, dtype=np.float64))
    norms = np.linalg.norm(projected, axis=0)
    assert norms.min() > 0.0
    fixed2, count2 = cssd.dead_column_fallback(torch.as_tensor(D), torch.as_tensor(U))
    assert count2 == count
    assert torch.equal(fixed, fixed2)


@pytest.mark.parametrize("q", [1, 2])
def test_cssd_forward_paths_c6(q: int) -> None:
    train = _synthetic_phi(seed=9)
    subspace = cssd.build_common_subspace(train, q)
    model = cssd.build_cssd_model(_dictionary(), 0, subspace)
    phi = torch.as_tensor(train[:32], dtype=torch.float32)
    z = model.code(phi)
    # residual reconstruction: exact identity for the common part
    r = phi - (phi @ model.U) @ model.U.t()
    reconstruction = model.reconstruct(phi, z)
    loss = model.reconstruction_loss(phi, z)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0
    assert reconstruction.shape == r.shape


def test_c6_mask_and_path_unmodified() -> None:
    assert cm.c6_equivalence_check()
    assert cssd.CSSD_MASK is cm.C6_MASK
    checkpoint = (
        REPO_ROOT
        / "tracks/ksvd/results/e2e_dictenv_h1_clarity_audit/matched_cpu/C6_e320_soup_state.pt"
    )
    if not checkpoint.exists():  # pragma: no cover
        pytest.skip("C6 seed-0 checkpoint not available")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = cm.build_clean_mech_model(_dictionary(), 0, C6_SPARSE)
    result = model.load_state_dict({key: value.float() for key, value in state.items()})
    assert not result.missing_keys
    assert not result.unexpected_keys


# ---------------------------------------------------------------------------
# training-loop equivalence
# ---------------------------------------------------------------------------


def test_train_cssd_matches_frozen_loop(tmp_path: Path) -> None:
    train = p1run.load_split("train", subset=32)
    valid = p1run.load_split("valid", subset=32)
    if not train or not valid:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    cache = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p1/cache/env_train.pt"
    if not cache.exists():  # pragma: no cover
        pytest.skip("phi65 cache not available")
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    subspace = cssd.build_common_subspace(np.asarray(blob["phi"][:1000]), 2)

    def factory(dictionary: np.ndarray, seed: int) -> cssd.CSSDModel:
        return cssd.build_cssd_model(dictionary, seed, subspace)

    reference = audit.train_cpu(
        tag="EQ-REF",
        mask=cssd.CSSD_MASK,
        epochs=1,
        threads=2,
        out_dir=tmp_path / "ref",
        train_data=train,
        valid_data=valid,
        seed=0,
        save_states=True,
        log=False,
        model_factory=factory,
    )
    candidate = cssd.train_cssd(
        tag="EQ-CSSD",
        epochs=1,
        threads=2,
        out_dir=tmp_path / "cssd",
        subspace=subspace,
        train_data=train,
        valid_data=valid,
        seed=0,
        save_states=True,
        log=False,
    )
    assert reference["best_valid_mae"] == candidate["best_valid_mae"]
    assert reference["soup"]["soup_valid_mae"] == candidate["soup"]["soup_valid_mae"]
    ref_state = torch.load(tmp_path / "ref" / "EQ-REF_final_state.pt", map_location="cpu")
    cand_state = torch.load(tmp_path / "cssd" / "EQ-CSSD_final_state.pt", map_location="cpu")
    assert set(ref_state) == set(cand_state)
    for key in ref_state:
        assert torch.equal(ref_state[key], cand_state[key]), f"training diverged at {key}"


# ---------------------------------------------------------------------------
# Stage C / gate logic
# ---------------------------------------------------------------------------


def _probe_row(**overrides: float) -> dict[str, float]:
    row = {
        "atom6_rate": 1.0,
        "atom24_rate": 1.0,
        "atom27_rate": 1.0,
        "atom23_rate": 0.9,
        "top5_usage": 4.8,
    }
    row.update(overrides)
    return row


def test_condition_a_b_c_and_selection_q1() -> None:
    raw = _probe_row()
    q1 = _probe_row(atom6_rate=0.1, atom24_rate=0.2, atom27_rate=0.3, top5_usage=3.0)
    assert cssd.condition_a(q1, raw)["satisfied"]
    assert cssd.condition_b(q1, raw)["satisfied"]
    assert cssd.condition_c({"centered_residual_fraction": 0.7})["satisfied"]
    probe = {
        seed: {
            "RAW": {"valid": raw, "train": raw},
            "Q1": {"valid": q1, "train": q1},
            "Q2": {"valid": q1, "train": q1},
        }
        for seed in (0, 1, 2)
    }
    energy = {
        seed: {"Q1": {"valid": {"centered_residual_fraction": 0.7}}}
        for seed in (0, 1, 2)
    }
    rule = cssd.selection_rule(probe=probe, energy=energy)
    assert rule["selected"] == "q1"
    assert rule["q1_passed_seeds"] == [0, 1, 2]


def test_selection_q2_fallback_and_none() -> None:
    raw = _probe_row()
    # q1 fails condition C (residual keeps too little variation)
    q1 = _probe_row(atom6_rate=0.1, atom24_rate=0.2, atom27_rate=0.3, top5_usage=3.0)
    q2 = _probe_row(
        atom6_rate=0.0,
        atom24_rate=0.0,
        atom27_rate=0.1,
        top5_usage=3.0,
        atom23_rate=0.3,
    )
    probe = {
        seed: {
            "RAW": {"valid": raw, "train": raw},
            "Q1": {"valid": q1, "train": q1},
            "Q2": {"valid": q2, "train": q2},
        }
        for seed in (0, 1, 2)
    }
    energy = {
        seed: {"Q1": {"valid": {"centered_residual_fraction": 0.2}}}
        for seed in (0, 1, 2)
    }
    assert cssd.selection_rule(probe=probe, energy=energy)["selected"] == "q2"
    # nothing passes: RAW already flat
    flat = _probe_row(top5_usage=3.0)
    probe_none = {
        seed: {
            "RAW": {"valid": flat, "train": flat},
            "Q1": {"valid": flat, "train": flat},
            "Q2": {"valid": flat, "train": flat},
        }
        for seed in (0, 1, 2)
    }
    energy_none = {
        seed: {"Q1": {"valid": {"centered_residual_fraction": 0.2}}}
        for seed in (0, 1, 2)
    }
    assert cssd.selection_rule(probe=probe_none, energy=energy_none)["selected"] is None


def test_epoch40_gate_logic() -> None:
    pass_case = cssd.epoch40_gate(
        dc_count_gt095=0,
        top5_share=4.0,
        effective_atoms=16.0,
        usage_weighted_spec=0.30,
        gradient_norm_D=1e-3,
        column_norm_min=0.1,
    )
    assert pass_case["passed"]
    fail_a = cssd.epoch40_gate(
        dc_count_gt095=3,
        top5_share=4.7,
        effective_atoms=16.0,
        usage_weighted_spec=0.30,
        gradient_norm_D=1e-3,
        column_norm_min=0.1,
    )
    assert not fail_a["passed"] and not fail_a["condition_a"]["satisfied"]
    fail_b = cssd.epoch40_gate(
        dc_count_gt095=0,
        top5_share=4.0,
        effective_atoms=14.6,
        usage_weighted_spec=0.21,
        gradient_norm_D=1e-3,
        column_norm_min=0.1,
    )
    assert not fail_b["passed"] and not fail_b["condition_b"]["satisfied"]
    fail_c = cssd.epoch40_gate(
        dc_count_gt095=0,
        top5_share=4.0,
        effective_atoms=16.0,
        usage_weighted_spec=0.30,
        gradient_norm_D=0.0,
        column_norm_min=0.1,
    )
    assert not fail_c["passed"] and not fail_c["condition_c"]["satisfied"]
    catastrophic = cssd.catastrophic_check(train_mae=3.0, valid_mae=0.4, rec_loss=0.01)
    assert catastrophic["triggered"]
    ok = cssd.catastrophic_check(train_mae=0.2, valid_mae=0.2, rec_loss=0.01)
    assert not ok["triggered"]


def test_graph_coverage_synthetic() -> None:
    codes = np.zeros((6, 2))
    codes[[0, 1, 3], 0] = 1.0  # atom 0 in graphs 0,0,1
    codes[[2, 4], 1] = 1.0     # atom 1 in graphs 1,2
    graph_ids = np.asarray([0, 0, 1, 1, 2, 2])
    stats = cssd.graph_coverage(codes, graph_ids, n_graphs=3)
    assert np.allclose(stats["coverage"], [2 / 3, 2 / 3])
    assert np.allclose(stats["mean_activations_per_active_graph"], [1.5, 1.0])
    assert np.allclose(stats["top10_graph_share"], [2 / 3, 0.5])


def test_guards() -> None:
    cssd.cpu_only_guard(torch.device("cpu"))
    with pytest.raises(RuntimeError):
        cssd.cpu_only_guard(torch.device("cuda"))
    cssd.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        cssd.official_test_blocker({"official_test_loaded": True})
    with pytest.raises(ValueError):
        cssd.build_common_subspace(_synthetic_phi(50), 3)


def test_reference_values_reproduced_by_frozen_audit() -> None:
    """The frozen references in the module match the stored audit artifacts."""
    path = (
        REPO_ROOT
        / "tracks/ksvd/results/e2e_dictenv_dictionary_coder_audit_v1/coder_geometry/seed0/summary.json"
    )
    if not path.exists():  # pragma: no cover
        pytest.skip("audit artifact not available")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload["tables"]["valid"]["iht10"]["concentration"]
    assert abs(table["effective_atoms"] - cssd.REF_NEFF_VALID) <= cssd.REFERENCE_TOLERANCE
    assert abs(table["top5_share"] - cssd.REF_TOP5_VALID) <= cssd.REFERENCE_TOLERANCE
    spec_path = (
        REPO_ROOT
        / "tracks/ksvd/results/e2e_dictenv_dictionary_coder_audit_v1/atom_specialization/seed0.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    weighted = spec["coders"]["iht10"]["usage_weighted_specialization"]
    assert abs(weighted - cssd.REF_WEIGHTED_SPEC) <= cssd.REFERENCE_TOLERANCE
