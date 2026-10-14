"""Fast correctness tests for cscl-rw-local-vs-multiscale-v1.

No ZINC dataset, no checkpoints, no GPU: tiny synthetic molecules only.
Covers the R1 kernel (sym(P)), the K/Ks per-bond equality math, symmetrisation
row-weight reallocation, error-audit arithmetic, the frozen decision rule,
strict init/hash parity with the historical R123 model, relabel invariance,
gradient liveness, checkpoint replay and the no-official-data scan.
"""

from __future__ import annotations


import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bond_random_walk_organization_v1 as bwo  # noqa: E402
import bond_rw_local_vs_multiscale_v1 as lm  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bond_rw_local_vs_multiscale_v1.yaml"
DRIVER = REPO_ROOT / "tracks/structure-contribution/code/run_bond_rw_local_vs_multiscale_v1.py"
MODULE = REPO_ROOT / "tracks/structure-contribution/code/bond_rw_local_vs_multiscale_v1.py"
NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _mol(idx: int, atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(idx, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


def _chain(n: int, idx: int = 0):
    return _mol(idx, range(n), [(i, i + 1, 1) for i in range(n - 1)])


def _ring(n: int, idx: int = 0):
    bonds = [(i, (i + 1) % n, 1) for i in range(n)]
    return _mol(idx, range(n), bonds)


def _model_pred(model, obj) -> float:
    src, dst, w = lm.k1_triplets_from_objects(obj)
    batch = bwo.move_batch(bwo.batch_rw_objects([obj], [(src, dst, w)]), torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        return float(model(batch))


def _relabel(o, perm):
    inv = np.argsort(perm)
    return sro.MolObjects(
        mol_index=o.mol_index,
        smiles=o.smiles,
        n_atoms=o.n_atoms,
        atom_feat=o.atom_feat[perm],
        bond_index=inv[o.bond_index],
        bond_type=o.bond_type,
        bond_feat=o.bond_feat,
        junc_index=o.junc_index,
        junc_bond=o.junc_bond,
        junc_feat=o.junc_feat,
        global_feat=o.global_feat,
    )


# ---------------------------------------------------------------------------
# config / frozen vocabulary
# ---------------------------------------------------------------------------


def test_config_matches_module_constants():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol"] == lm.PROTOCOL_ID
    assert cfg["status"] == "frozen"
    assert cfg["phase1_model"]["arm_R1"]["relation"].startswith("K1 = sym(P)")
    assert cfg["phase1_model"]["arm_R1"]["expected_parameters"] == lm.EXPECTED_PARAMETERS
    assert float(cfg["training"]["gpu_cap_hours"]) == 0.5
    assert cfg["scoring"]["contrasts"]["primary"].startswith("R123_minus_R1")
    decision = cfg["scoring"]["decision"]
    assert decision["primary_ci_upper_negative"] == "MULTISCALE_EXTRA_GAIN"
    assert decision["primary_ci_lower_positive"] == "ONE_HOP_BETTER"
    assert decision["primary_ci_crosses_zero"] == "MULTISCALE_INDISTINGUISHABLE"
    assert decision["stop_after_round"] is True
    forbidden = cfg["phase1_model"]["forbidden"]
    for token in ("GNN", "attention", "Junction"):
        assert token in forbidden


def test_r123_provenance_constants_frozen_in_config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    arm = cfg["phase1_model"]["arm_R1"]
    assert arm["expected_init_sha256"] == lm.R123_INIT_SHA256
    assert arm["ssl_encoder_used_sha256"] == lm.R123_ENCODER_USED_SHA256
    assert arm["reference_arm_R123"]["relation"].startswith("Ks = sym((P + P^2 + P^3)/3)")


# ---------------------------------------------------------------------------
# R1 kernel
# ---------------------------------------------------------------------------


def test_k1_is_symmetrised_lazy_transition():
    for obj in (_chain(4), _ring(6), _mol(0, [0, 1], [(0, 1, 1)])):
        A = bwo.line_graph_adjacency(obj.bond_index)
        P = bwo.lazy_transition(A)
        K1 = lm.k1_matrix(obj)
        assert np.allclose(K1, bwo.symmetrize(P), atol=1e-12)
        assert np.allclose(K1, K1.T, atol=1e-12)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_k1_differs_from_r123_and_uniform():
    obj = _ring(8)
    K1 = lm.k1_matrix(obj)
    K123 = bwo.symmetrize(bwo.relation_kernel(bwo.kernel_powers(bwo.line_graph_adjacency(obj.bond_index), bwo.MAX_K)))
    assert np.abs(K1 - K123).max() > 1e-3
    n = K1.shape[0]
    assert np.abs(K1 - np.full((n, n), 1.0 / n)).max() > 1e-3


def test_k1_triplets_reconstruct_dense_double_sum():
    rng = np.random.default_rng(7)
    for obj in (_ring(7), _chain(6)):
        n = int(obj.bond_index.shape[0])
        K1 = lm.k1_matrix(obj)
        B = rng.normal(size=(n, 4))
        phi = lm._symmetric_phi(B)
        dense = np.einsum("ef,efk->ek", K1, phi)
        src, dst, w = lm.k1_triplets_from_objects(obj)
        w_eff = np.where(src == dst, w * 0.5, w)
        trip = np.zeros_like(dense)
        np.add.at(trip, src, w_eff[:, None] * phi[src, dst])
        np.add.at(trip, dst, w_eff[:, None] * phi[src, dst])
        assert np.allclose(trip, dense, atol=1e-12)


# ---------------------------------------------------------------------------
# Phase-0 mathematics
# ---------------------------------------------------------------------------


def test_k_vs_ks_double_sum_equal_but_per_bond_differs():
    rng = np.random.default_rng(0)
    n = 9
    K = rng.normal(size=(n, n))
    Ks = bwo.symmetrize(K)
    B = rng.normal(size=(n, 4))
    phi = lm._symmetric_phi(B)
    r_K = np.einsum("ef,efk->ek", K, phi)
    r_Ks = np.einsum("ef,efk->ek", Ks, phi)
    r_KT = np.einsum("fe,fek->ek", K, phi)
    # full-graph double sum is invariant under symmetrisation
    assert np.abs(r_K.sum(axis=0) - r_Ks.sum(axis=0)).max() <= 1e-12
    # but the per-bond aggregate r_e is not
    assert np.abs(r_K - r_Ks).max() > 1e-6
    # exact identity r'_e = (r_e(K) + (K^T phi)_e) / 2
    assert np.abs(r_Ks - 0.5 * (r_K + r_KT)).max() <= 1e-12
    # rows still change after symmetrisation (the model reads row sums)
    rowsum_K = K.sum(axis=1)
    rowsum_Ks = Ks.sum(axis=1)
    assert np.allclose(rowsum_Ks, 0.5 * (rowsum_K + K.sum(axis=0)), atol=1e-12)
    assert np.abs(rowsum_Ks - 1.0).max() > 1e-6


def test_math_audit_molecule_on_random_and_molecule_graphs():
    rng = np.random.default_rng(3)
    for obj in (_ring(9), _chain(8)):
        rec = lm.math_audit_molecule(obj, rng)
        assert rec["double_sum_equality_max_abs_P"] <= 1e-12
        assert rec["double_sum_equality_max_abs_K123"] <= 1e-12
        assert rec["per_bond_identity_residual_P"] <= 1e-12
        assert rec["per_bond_identity_residual_K123"] <= 1e-12
        assert rec["matrix_max_abs_P_minus_symP"] >= 0.0
    # a chain has a non-regular line graph: symmetrisation changes per-bond messages
    # and moves row mass (a regular ring gives symmetric P, so diff is 0 there)
    rec_chain = lm.math_audit_molecule(_chain(8), rng)
    assert rec_chain["per_bond_diff_max_abs_P"] > 0.0
    assert rec_chain["matrix_max_abs_P_minus_symP"] > 0.0
    assert rec_chain["mean_abs_rowsum_symP_minus_1"] > 0.0
    # paired control: non-symmetric deterministic P for which K != Ks
    A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    P = bwo.lazy_transition(A)
    assert np.abs(P - P.T).max() > 0.0
    Ks = bwo.symmetrize(P)
    assert np.abs(P.sum(axis=1) - 1.0).max() <= 1e-12
    assert np.abs(Ks.sum(axis=1) - 1.0).max() > 0.0


def test_summarize_math_audit_flags():
    rng = np.random.default_rng(4)
    per_mol = [lm.math_audit_molecule(_chain(k), rng) for k in (7, 8, 9)]
    summary = lm.summarize_math_audit(per_mol)
    assert summary["conclusions"]["double_sum_equality_exact"]
    assert summary["conclusions"]["double_sum_equality_exact_K123"]
    assert summary["conclusions"]["per_bond_equality_holds"] is False
    assert summary["conclusions"]["per_bond_identity_verified"]
    for key in ("rowsum_symK_cv", "corr_degree_rowsum_symK", "mean_abs_rowsum_symK_minus_1"):
        assert np.isfinite(summary["aggregate"][key])


def test_synthetic_rng_audit_detects_global_torch_rng(monkeypatch):
    def _stub(seed: int = 0):
        assert seed == 0
        v = float(torch.rand(1).item()) + 0.5
        return {
            "test_mae_real_K_trained": v,
            "test_mae_real_K_model_evaluated_uniform": v + 1.0,
            "test_mae_uniform_K_trained_uniform": v + 2.0,
            "test_mae_linear_summary": v + 3.0,
        }

    monkeypatch.setattr(lm.bwo, "synthetic_relation_task", _stub)
    out = lm.synthetic_rng_audit(torch_seeds=(0, 0, 12345))
    assert out["seed0_repeat_identical"] is True
    assert out["different_seed_differs"] is True
    assert out["runs"][0]["torch_manual_seed"] == 0


# ---------------------------------------------------------------------------
# error audit arithmetic
# ---------------------------------------------------------------------------


def test_arm_error_stats_hand_computed():
    y = np.zeros(10)
    pred = np.zeros(10)
    pred[9] = 3.0
    st = lm.arm_error_stats(y, pred)
    assert st["mae"] == pytest.approx(0.3)
    assert st["rmse"] == pytest.approx(np.sqrt(9.0 / 10.0))
    assert st["median_ae"] == pytest.approx(0.0)
    assert st["p90_ae"] == pytest.approx(0.3)
    assert st["p95_ae"] == pytest.approx(1.65)
    assert st["p99_ae"] == pytest.approx(2.73)
    assert st["max_ae"] == pytest.approx(3.0)
    assert st["worst_rows"][0]["row"] == 9
    assert st["worst1_sse_share"] == pytest.approx(1.0)
    assert st["top10_sse_share"] == pytest.approx(1.0)


def test_contrast_audit_extreme_contributions():
    y = np.zeros(20)
    new = np.zeros(20)
    new[19] = 3.0
    ref = np.ones(20)
    out = lm.contrast_audit(y, new, ref, bootstrap_n=200, bootstrap_seed=0)
    assert out["delta"] == pytest.approx((2.0 - 19.0) / 20.0)
    assert out["mae_new"] == pytest.approx(3.0 / 20.0)
    assert out["mae_ref"] == pytest.approx(1.0)
    assert out["bootstrap"]["ci_lo"] <= out["bootstrap"]["ci_hi"]
    # removing the worst |delta| row (row 19) removes both the +3 and the -1
    assert out["delta_wo_single_worst_delta"] == pytest.approx(-1.0)
    assert out["delta_wo_top10_delta"] == pytest.approx(-1.0)
    assert out["delta_wo_single_worst_ae"] == pytest.approx(-1.0)
    assert out["top10_removed_delta_arm_ae"] == pytest.approx(-1.0)
    assert out["top5_worst_delta_rows"][0]["row"] == 19
    assert out["top5_worst_delta_rows"][0]["delta"] == pytest.approx(2.0)
    assert len(out["top5_best_delta_rows"]) == 5


def test_error_audit_and_per_molecule_rows():
    y = np.zeros(6)
    preds = {"A": np.zeros(6), "B": np.ones(6)}
    audit = lm.error_audit(y, preds, [("B_minus_A", "B", "A")], bootstrap_n=100, bootstrap_seed=0)
    assert set(audit["arm_stats"]) == {"A", "B"}
    c = audit["contrasts"]["B_minus_A"]
    assert c["delta"] == pytest.approx(1.0)
    assert c["mae_new"] == pytest.approx(1.0)
    rows = lm.per_molecule_rows(y, preds)
    assert len(rows) == 6
    assert set(rows[3]) == {"row", "y", "A_pred", "A_ae", "B_pred", "B_ae"}


def test_primary_decision_rule_and_next_step():
    assert lm.primary_decision_from_ci(-0.02, -0.001) == "MULTISCALE_EXTRA_GAIN"
    assert lm.primary_decision_from_ci(0.001, 0.02) == "ONE_HOP_BETTER"
    assert lm.primary_decision_from_ci(-0.02, 0.02) == "MULTISCALE_INDISTINGUISHABLE"
    assert lm.primary_decision_from_ci(-0.02, 0.0) == "MULTISCALE_INDISTINGUISHABLE"
    assert "worth studying" in lm.next_step_for("MULTISCALE_EXTRA_GAIN")
    assert "sufficient" in lm.next_step_for("ONE_HOP_BETTER")
    assert "sufficient" in lm.next_step_for("MULTISCALE_INDISTINGUISHABLE")


# ---------------------------------------------------------------------------
# model-level correctness
# ---------------------------------------------------------------------------


def test_strict_init_match_with_historical_r123():
    model = bwo.build_model(0, NORM, "cpu")
    assert bwo.parameter_count(model) == lm.EXPECTED_PARAMETERS
    assert bwo.state_dict_hash(model) == lm.R123_INIT_SHA256
    other = bwo.build_model(1, NORM, "cpu")
    assert bwo.state_dict_hash(other) != lm.R123_INIT_SHA256
    assert bwo.forbidden_module_scan(model) == []


def test_model_atom_relabel_invariance_for_r1():
    obj = _ring(9, idx=5)
    model = bwo.build_model(0, NORM, "cpu")
    rng = np.random.default_rng(6)
    p1 = _model_pred(model, obj)
    p2 = _model_pred(model, _relabel(obj, rng.permutation(obj.n_atoms)))
    assert abs(p1 - p2) <= 1e-6


def test_bond_permutation_equivariance_of_r1_kernel():
    obj = _ring(8)
    K1 = lm.k1_matrix(obj)
    rng = np.random.default_rng(8)
    perm = rng.permutation(K1.shape[0])
    conjugated = K1[np.ix_(perm, perm)]
    assert np.allclose(conjugated, conjugated.T, atol=1e-12)
    assert np.allclose(np.sort(K1.sum(axis=1)), np.sort(conjugated.sum(axis=1)), atol=1e-12)


def test_model_gradients_live_for_r1():
    obj = _ring(10)
    src, dst, w = lm.k1_triplets_from_objects(obj)
    batch = bwo.move_batch(bwo.batch_rw_objects([obj], [(src, dst, w)]), torch.device("cpu"))
    model = bwo.build_model(0, NORM, "cpu")
    model.train()
    loss = model(batch).view(-1)[0]
    loss.backward()
    pair_grads = [float(p.grad.norm()) for p in model.pair_mlp.parameters() if p.grad is not None]
    assert pair_grads and all(g > 0.0 for g in pair_grads)
    encoder_grads = [float(p.grad.norm()) for p in model.bond_encoder.parameters() if p.grad is not None]
    assert encoder_grads and all(g >= 0.0 for g in encoder_grads)


def test_checkpoint_save_load_replay_identical():
    obj = _ring(9)
    src, dst, w = lm.k1_triplets_from_objects(obj)
    batch = bwo.move_batch(bwo.batch_rw_objects([obj], [(src, dst, w)]), torch.device("cpu"))
    model = bwo.build_model(0, NORM, "cpu")
    model.eval()
    with torch.no_grad():
        p1 = model(batch)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fresh = bwo.build_model(0, NORM, "cpu")
    fresh.load_state_dict(state)
    fresh.eval()
    with torch.no_grad():
        p2 = fresh(batch)
    assert float((p1 - p2).abs().max()) <= 1e-7


def test_r1_training_smoke_is_deterministic():
    objs = [_chain(5, 0), _ring(6, 1), _chain(4, 2)]
    triplets = [lm.k1_triplets_from_objects(o) for o in objs]
    y = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    kw = dict(
        arm="R1", seed=0, train_objs=objs, train_triplets=triplets, y_fit=y,
        norm=NORM, device=torch.device("cpu"), ssl_state=None, smoke_epochs=1,
        log_fn=lambda _m: None,
    )
    m_a, info_a = bwo.train_supervised_rw(**kw)
    m_b, info_b = bwo.train_supervised_rw(**kw)
    assert bwo.state_dict_hash(m_a) == bwo.state_dict_hash(m_b)
    assert info_a["batch_order"]["full_hash"] == info_b["batch_order"]["full_hash"]
    assert info_a["parameter_audit"]["actual_parameters"] == lm.EXPECTED_PARAMETERS


# ---------------------------------------------------------------------------
# discipline scan
# ---------------------------------------------------------------------------


def test_no_official_or_locked_data_path():
    module_src = MODULE.read_text(encoding="utf-8")
    driver_src = DRIVER.read_text(encoding="utf-8")
    for token in ("valid_rows", "test_rows", "encoded_valid", "encoded_test", "official_valid_rows"):
        assert token not in module_src
        assert token not in driver_src
    assert "official valid/test never loaded" in driver_src
    assert driver_src.count("load_y_rows(") == 2  # Fit (train) + Monitor (single scoring read)
    assert "monitor_read_D.lock" in driver_src
    assert "stop_after_round" in driver_src
    assert 'choices=("R1",)' in driver_src


def test_frozen_config_declares_single_model_and_cap():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["training"]["extra_seeds"] == 0
    assert cfg["training"]["extra_models"] == 0
    assert cfg["scoring"]["read_monitor_once"] is True
    assert "R1_minus_U" in cfg["scoring"]["contrasts"]["secondary"]
    assert cfg["split"]["locked_read"] is False
