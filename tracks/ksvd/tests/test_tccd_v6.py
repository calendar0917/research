"""Targeted correctness tests for the TCCD-v6 POST gain decomposition."""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v5 as V5
from tracks.ksvd.code import tccd_v6 as V


# ---------------------------------------------------------------------------
# synthetic algebraically consistent cache
# ---------------------------------------------------------------------------
def consistent_cache(n_graphs: int = 3, seed: int = 0):
    """Cache whose base is built exactly from C and R, so A/D must reconstruct."""
    rng = np.random.default_rng(seed)
    C_list, i0_list, i1_list, rel_list, base_rows = [], [], [], [], []
    iu0, iu1 = T.sym_indices(V.K_PROTO)
    ns = []
    for g in range(n_graphs):
        n = 6 + g
        ns.append(n)
        logits = rng.standard_normal((n, V.K_PROTO)).astype(np.float32)
        C = torch.softmax(torch.as_tensor(logits), dim=-1).numpy().astype(np.float32)
        C_list.append(C)
        ops = []
        for r in range(V.N_REL):
            R = rng.standard_normal((n, n)).astype(np.float32)
            R = 0.5 * (R + R.T)
            np.fill_diagonal(R, float(V.RELATION_DIAGONAL_PER_OCCURRENCE[r]))
            ops.append(R)
        a0, a1 = np.triu_indices(n, 1)
        i0_list.append(a0.astype(np.int64))
        i1_list.append(a1.astype(np.int64))
        rel_list.append(np.ascontiguousarray(np.stack(ops, axis=0)[:, a0, a1].T, dtype=np.float32))
        bag = C.sum(axis=0)
        rel_blocks = []
        for R in ops:
            M = C.T @ R @ C
            rel_blocks.append(M[iu0, iu1])
        base_rows.append(np.concatenate([bag, *rel_blocks]).astype(np.float32))
    n_patches = np.asarray([len(c) for c in C_list])
    n_pairs = np.asarray([len(a) for a in i0_list])
    cache = V.LabelFreeCache(
        base=np.stack(base_rows, axis=0),
        C=np.concatenate(C_list),
        c_offsets=np.concatenate([[0], np.cumsum(n_patches)]).astype(np.int64),
        pair_i0=np.concatenate(i0_list),
        pair_i1=np.concatenate(i1_list),
        pair_rel=np.concatenate(rel_list),
        pair_offsets=np.concatenate([[0], np.cumsum(n_pairs)]).astype(np.int64),
        graph_index=np.arange(n_graphs, dtype=np.int64),
        meta={"protocol": V.PROTOCOL_VERSION, "official_test_loaded": False},
    )
    y = np.asarray([float(g) for g in range(n_graphs)], dtype=np.float32)
    return cache, y


# ---------------------------------------------------------------------------
# geometry / parameters / masks
# ---------------------------------------------------------------------------
def test_descriptor_geometry_and_parameters():
    assert V.PAIR_WIDTH == 3 * V.K_PROTO + V.N_REL == 197
    assert V.FROZEN_BASE_DIM == T.h_dim(V.K_PROTO, V.N_REL) == 10464
    assert V.BLOCK_DIMS == {"A": 64, "B": 64, "C_prod": 64, "D": 5}
    assert V.RECON_MASK.sum() == 64 + 5
    assert V.NOVEL_MASK.sum() == 128
    assert np.array_equal(V.RECON_MASK + V.NOVEL_MASK, np.ones(197, dtype=np.float32))
    assert V.parameter_accounting() == {"branch": 13712, "head": 10481, "total": 24193}


def test_masks_are_target_free_and_frozen():
    # The frozen partition uses only the algebra audit outcome, never y.
    assert V.recon_mask_from_relations((True,) * 5).sum() == 69
    assert V.novel_mask_from_relations((True,) * 5).sum() == 128
    partial = V.recon_mask_from_relations((True, False, False, False, True))
    assert partial.sum() == 64 + 2
    assert V.novel_mask_from_relations((True, False, False, False, True)).sum() == 64 + 64 + 3


def test_arm_parameter_and_init_matching():
    models = {arm: V.ArmFactory.build(arm, seed=0) for arm in V.ARMS}
    counts = {arm: V.arm_parameter_count(m) for arm, m in models.items()}
    assert len({tuple(c.values()) for c in counts.values()}) == 1
    assert all(c == V.parameter_accounting() for c in counts.values())
    checksums = {arm: V5._state_checksum(m) for arm, m in models.items()}
    assert len(set(checksums.values())) == 1
    assert checksums["full_nl"] == V.FULL_INIT_CHECKSUM


def test_data_free_gate0_passes():
    checks = V.gate0_data_free_checks("cpu")
    for key in (
        "descriptor_dims_ok",
        "mask_partition_exact",
        "parameter_equality",
        "parameter_matches_registered",
        "init_equality",
        "init_bit_identical",
        "init_matches_v5_post",
        "shapes_ok",
        "full_equals_recon_plus_novel",
        "v5_descriptor_equivalence",
        "batching_invariant",
        "pair_order_invariant",
        "relabel_invariant",
        "relabel_prediction_invariant",
        "empty_pair_branch_zero",
        "empty_pair_pbar_zero",
        "data_free_all_pass",
    ):
        assert checks[key] is True, key
    assert checks["official_test_loaded"] is False


# ---------------------------------------------------------------------------
# label-free algebra audit
# ---------------------------------------------------------------------------
def test_audit_reconstructs_A_and_D_from_base_on_consistent_cache():
    cache, _ = consistent_cache(3)
    audit = V.label_free_audit(cache)
    assert audit["A"]["pass"] is True
    assert audit["A"]["max_abs_diff"] < 1e-6
    assert audit["D"]["status"] == "BASE_RECONSTRUCTIBLE"
    assert all(audit["D"]["reconstructible_mask"])
    assert audit["recon_mask_nonzero"] == 69
    assert audit["novel_mask_nonzero"] == 128
    assert audit["frozen_mask_matches"] is True
    assert audit["target_not_read"] is True
    assert audit["official_test_loaded"] is False


def test_audit_fails_when_base_is_random_and_has_no_y_access():
    cache, _ = V.tiny_cache()
    audit = V.label_free_audit(cache)
    # random base cannot mechanically reproduce A
    assert audit["A"]["pass"] is False
    assert audit["all_pass"] is False
    assert "y" not in audit


def test_graph_pair_means_match_v5_tensor_path():
    cache, y = consistent_cache(3)
    idx = list(range(cache.n_graphs))
    ours = np.concatenate(
        [V.make_arm_batch(cache, y, [i], "cpu").pbar.detach().cpu().numpy() for i in idx], axis=0
    )
    ref = V._pair_means_via_v5(cache, y, idx, "cpu")
    assert float(np.abs(ours - ref).max()) <= 1e-6
    blocks = V.graph_pair_means(cache)
    full = V.full_descriptor_from_blocks(blocks)
    # graph_pair_means averages in a different summation order, so allow float32 slack
    assert float(np.abs(full - ours).max()) <= 1e-5


# ---------------------------------------------------------------------------
# masks change only their own coordinates
# ---------------------------------------------------------------------------
def test_input_masks_gate_only_their_own_coordinates():
    torch.manual_seed(0)
    B = 4
    base = torch.zeros(B, V.FROZEN_BASE_DIM)
    pbar = torch.randn(B, V.PAIR_WIDTH)
    has_pairs = torch.ones(B, dtype=torch.bool)
    batch = V.ArmBatch(base=base, pbar=pbar, has_pairs=has_pairs, y=torch.zeros(B))
    shifted_novel = pbar.clone()
    shifted_novel[:, V.NOVEL_MASK.astype(bool)] += 0.5
    batch_novel = V.ArmBatch(base=base, pbar=shifted_novel, has_pairs=has_pairs, y=torch.zeros(B))
    shifted_recon = pbar.clone()
    shifted_recon[:, V.RECON_MASK.astype(bool)] += 0.5
    batch_recon = V.ArmBatch(base=base, pbar=shifted_recon, has_pairs=has_pairs, y=torch.zeros(B))

    recon_model = V.ArmFactory.build("recon_nl", seed=0).eval()
    novel_model = V.ArmFactory.build("novel_nl", seed=0).eval()
    full_model = V.ArmFactory.build("full_nl", seed=0).eval()
    with torch.no_grad():
        r0, r1 = recon_model(batch), recon_model(batch_novel)
        n0, n1 = novel_model(batch), novel_model(batch_recon)
        f0, f1 = full_model(batch), full_model(batch_novel)
    assert float((r0 - r1).abs().max()) == 0.0
    assert float((n0 - n1).abs().max()) == 0.0
    assert float((f0 - f1).abs().max()) > 0.0


# ---------------------------------------------------------------------------
# training smoke and decision rules
# ---------------------------------------------------------------------------
def test_train_arm_smoke_and_masked_perturbation_invariance():
    cache, y = consistent_cache(4, seed=1)
    train = np.arange(3)
    dev = np.arange(3, 4)
    res = V.train_arm(cache, y, train, dev, "cpu", "recon_nl", seed=0, max_epochs=2, patience=2, batch=2)
    assert res.soup_valid is not None and np.isfinite(res.soup_valid)
    assert len(res.soup_members) >= 1
    assert res.init_checksum == V.FULL_INIT_CHECKSUM


def test_decomposition_decision_cases():
    base = V.BASE_SOUP_MAE
    full = V.FULL_SOUP_MAE

    # R1: recon sufficient, nonlinearity immaterial
    d = V.decomposition_decision(
        recon_nl_soup=full + 0.003, recon_linfact_soup=full + 0.004, novel_nl_soup=full + 0.030
    )
    assert d["case"] == "R1" and d["g_recon"] >= 0.015

    # R2: recon sufficient, nonlinearity material
    d = V.decomposition_decision(
        recon_nl_soup=full + 0.003, recon_linfact_soup=full + 0.015, novel_nl_soup=full + 0.030
    )
    assert d["case"] == "R2"

    # ambiguity requires exactly one paired seed
    d = V.decomposition_decision(
        recon_nl_soup=full + 0.003, recon_linfact_soup=full + 0.010, novel_nl_soup=full + 0.030
    )
    assert d["case"] == "R_AMBIGUOUS_NEEDS_SEED1" and d["needs_seed1_recon"] is True
    # paired mean >= 0.0075 -> R2
    d = V.decomposition_decision(
        recon_nl_soup=full + 0.003,
        recon_linfact_soup=full + 0.010,
        novel_nl_soup=full + 0.030,
        recon_linfact_gap_seed1=0.009,
    )
    assert d["case"] == "R2"
    # paired mean < 0.0075 -> R1
    d = V.decomposition_decision(
        recon_nl_soup=full + 0.003,
        recon_linfact_soup=full + 0.010,
        novel_nl_soup=full + 0.030,
        recon_linfact_gap_seed1=0.005,
    )
    assert d["case"] == "R1"

    # M: only novel moments sufficient
    d = V.decomposition_decision(
        recon_nl_soup=full + 0.030, recon_linfact_soup=full + 0.031, novel_nl_soup=full + 0.003
    )
    assert d["case"] == "M" and d["case_M_novel_sufficient"] is True

    # J: both partial, neither sufficient
    d = V.decomposition_decision(
        recon_nl_soup=base - 0.008, recon_linfact_soup=base - 0.007, novel_nl_soup=base - 0.008
    )
    assert d["case"] == "J" and d["case_J_joint"] is True

    # N: neither explains FULL
    d = V.decomposition_decision(
        recon_nl_soup=base - 0.002, recon_linfact_soup=base - 0.001, novel_nl_soup=base - 0.002
    )
    assert d["case"] == "N" and d["case_N_strong_synergy"] is True

    # R has precedence over M when both hold
    d = V.decomposition_decision(
        recon_nl_soup=full + 0.003, recon_linfact_soup=full + 0.004, novel_nl_soup=full + 0.003
    )
    assert d["case"] == "R1" and d["case_M_novel_sufficient"] is True
