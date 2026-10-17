"""Fast correctness tests for cscl-p3-bond-relation-bridge-v1.

Synthetic molecules only: no ZINC dataset, no checkpoints, no GPU.
Covers the R/U kernel definitions (zero diagonal, exact row sums, exact pair
reconstruction, degeneracy), pair batching/offset alignment, the zero-init
residual (bit-identical P3 output), parameter/init matching, gradient liveness,
and the frozen decision vocabulary.
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

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import p3_bond_relation_bridge_v1 as pb  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/p3_bond_relation_bridge_v1.yaml"
NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def make_mol(idx: int, atom_types, bonds) -> sro.MolObjects:
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    edge_index, edge_attr = [], []
    for u, v, bt in bonds:
        edge_index.extend([(u, v), (v, u)])
        edge_attr.extend([bt, bt])
    return sro.build_mol_objects(idx, x, np.asarray(edge_index, dtype=np.int64).T, np.asarray(edge_attr, dtype=np.int64))


def chain(n: int) -> sro.MolObjects:
    return make_mol(0, range(n), [(i, i + 1, 1) for i in range(n - 1)])


def ring(n: int) -> sro.MolObjects:
    return make_mol(0, range(n), [(i, (i + 1) % n, 1) for i in range(n)])


def single_bond() -> sro.MolObjects:
    return make_mol(0, [0, 1], [(0, 1, 1)])


def branched() -> sro.MolObjects:
    return make_mol(0, range(6), [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1), (1, 5, 1)])


def batch_of(objs) -> dict[str, torch.Tensor]:
    return sro.batch_objects(list(objs))


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def test_line_adjacency_is_the_physical_bond_line_graph():
    mol = chain(5)  # 4 bonds: b0 b1 b2 b3, path line graph
    adj = pb.line_adjacency(mol)
    assert adj.shape == (4, 4)
    assert np.allclose(np.diag(adj), 0.0)
    expected = np.zeros((4, 4), dtype=np.float32)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        expected[a, b] = expected[b, a] = 1.0
    assert np.array_equal(adj, expected)


def test_R_is_row_normalised_uniform_over_real_neighbours():
    mol = chain(5)
    kernel = pb.kernel_matrices(mol)["R"]
    assert np.allclose(np.diag(kernel), 0.0)
    # terminal bonds have one neighbour, internal bonds two
    assert kernel[0, 1] == pytest.approx(1.0)
    assert kernel[1, 0] == pytest.approx(0.5) and kernel[1, 2] == pytest.approx(0.5)
    assert np.allclose(kernel.sum(axis=1), 1.0)


def test_U_is_uniform_over_all_other_bonds():
    mol = chain(6)  # 5 bonds
    kernel = pb.kernel_matrices(mol)["U"]
    assert np.allclose(np.diag(kernel), 0.0)
    off = kernel[~np.eye(5, dtype=bool)]
    assert np.allclose(off, 1.0 / 4.0)
    assert np.allclose(kernel.sum(axis=1), 1.0)


@pytest.mark.parametrize("mol", [single_bond(), chain(4), chain(9), ring(5), ring(8), branched()])
def test_row_sums_and_diagonal_match_between_R_and_U(mol):
    mats = pb.kernel_matrices(mol)
    n = mats["R"].shape[0]
    assert np.allclose(np.diag(mats["R"]), 0.0)
    assert np.allclose(np.diag(mats["U"]), 0.0)
    expected = np.zeros(n, dtype=np.float32) if n < 2 else np.ones(n, dtype=np.float32)
    assert np.allclose(mats["R"].sum(axis=1), expected, atol=1e-6)
    assert np.allclose(mats["U"].sum(axis=1), expected, atol=1e-6)
    assert mats["R"].sum() == pytest.approx(mats["U"].sum(), abs=1e-4)


def test_single_bond_has_no_pairs_and_no_relation_signal():
    mol = single_bond()
    for arm in ("R", "U"):
        pairs = pb.pair_lists(mol, arm)
        assert pairs["src"].shape[0] == 0
        assert np.allclose(pb.kernel_matrices(mol)[arm], 0.0)


def test_three_bond_ring_is_a_degenerate_R_equals_U_case():
    mol = ring(3)  # cyclopropane: line graph is complete -> R == U
    mats = pb.kernel_matrices(mol)
    assert np.array_equal(mats["R"], mats["U"])
    assert pb.audit_molecule(mol)["checks"]["R_equals_U"] is True


def test_ring_R_differs_from_U_above_three_bonds():
    mol = ring(6)
    mats = pb.kernel_matrices(mol)
    assert not np.array_equal(mats["R"], mats["U"])
    assert np.allclose(pb.kernel_matrices(mol)["R"].sum(axis=1), 1.0)


@pytest.mark.parametrize("arm", ["R", "U"])
@pytest.mark.parametrize("mol", [single_bond(), chain(7), ring(7), branched()])
def test_pair_lists_reconstruct_the_dense_kernel_exactly(arm, mol):
    kernel = pb.kernel_matrices(mol)[arm]
    rebuilt = pb.kernel_from_pairs(mol, arm)
    assert np.array_equal(rebuilt, kernel)
    pairs = pb.pair_lists(mol, arm)
    assert int(pairs["src"].shape[0]) == int(pairs["dst"].shape[0])
    assert np.all(pairs["src"] < pairs["dst"])  # strictly upper triangle
    assert np.allclose(kernel[pairs["src"], pairs["dst"]], pairs["w_src"])
    assert np.allclose(kernel[pairs["dst"], pairs["src"]], pairs["w_dst"])


def test_kernel_is_invariant_to_atom_renumbering():
    mol = branched()
    perm = np.array([3, 0, 5, 1, 4, 2], dtype=np.int64)
    inv = np.argsort(perm)
    relabelled = sro.MolObjects(
        mol_index=mol.mol_index,
        smiles=mol.smiles,
        n_atoms=mol.n_atoms,
        atom_feat=mol.atom_feat[perm],
        bond_index=inv[mol.bond_index],
        bond_type=mol.bond_type,
        bond_feat=mol.bond_feat,
        junc_index=mol.junc_index,
        junc_bond=mol.junc_bond,
        junc_feat=mol.junc_feat,
        global_feat=mol.global_feat,
    )
    for arm in ("R", "U"):
        assert np.array_equal(pb.kernel_matrices(mol)[arm], pb.kernel_matrices(relabelled)[arm])


# ---------------------------------------------------------------------------
# pair batching / index alignment
# ---------------------------------------------------------------------------


def test_pair_set_batch_offsets_follow_batch_concatenation_order():
    objs = [chain(4), single_bond(), ring(6), chain(3)]
    pset = pb.build_pair_set(objs, "R")
    index = [2, 0, 3]
    batch = pset.batch(index, "cpu")
    batch_obj = batch_of([objs[i] for i in index])
    n_bonds_total = int(batch_obj["bond_index"].shape[0])
    assert int(batch["src"].max()) < n_bonds_total
    assert int(batch["dst"].max()) < n_bonds_total
    assert torch.all(batch["src"] < batch["dst"])
    # per-molecule row sums of the aggregated weights must reconstruct to 1
    for arm in ("R",):
        sums = torch.zeros(n_bonds_total)
        sums.index_add_(0, batch["src"], batch["w_src"])
        sums.index_add_(0, batch["dst"], batch["w_dst"])
        offset = 0
        for i in index:
            n_b = int(np.asarray(objs[i].bond_index).reshape(-1, 2).shape[0])
            if n_b >= 2:
                assert torch.allclose(sums[offset : offset + n_b], torch.ones(n_b), atol=1e-6)
            offset += n_b


def test_empty_batch_of_pairs_is_usable():
    pset = pb.build_pair_set([single_bond(), single_bond()], "U")
    batch = pset.batch([0, 1], "cpu")
    assert int(batch["src"].numel()) == 0
    assert int(batch["w_src"].numel()) == 0


# ---------------------------------------------------------------------------
# model: zero-init residual, matched parameters, liveness
# ---------------------------------------------------------------------------


def test_bridge_parameter_count_and_matched_initialisation():
    p3 = bp.build_bjoc_null(0, NORM, "cpu")
    r = pb.build_bridge(0, NORM, "cpu")
    u = pb.build_bridge(0, NORM, "cpu")
    assert pb.model_parameters(p3) == pb.BASE_PARAMETERS == 413_505
    assert pb.model_parameters(r) == pb.model_parameters(u) == pb.BRIDGE_PARAMETERS
    assert pb.BRIDGE_PARAMETERS - pb.BASE_PARAMETERS == r.new_parameters
    assert pb.bridge_state_hash(r) == pb.bridge_state_hash(u)
    assert torch.count_nonzero(r.rel_scale).item() == 0
    assert pb.bridge_state_hash(r) == pb.bridge_state_hash(pb.build_bridge(0, NORM, "cpu"))
    assert pb.bridge_state_hash(r) != pb.bridge_state_hash(pb.build_bridge(1, NORM, "cpu"))


def test_zero_init_residual_is_bit_identical_to_p3():
    objs = [chain(6), ring(7), branched()]
    p3 = bp.build_bjoc_null(3, NORM, "cpu").eval()
    bridge = pb.build_bridge(3, NORM, "cpu").eval()
    batch = batch_of(objs)
    pairs = pb.build_pair_set(objs, "R").batch([0, 1, 2], "cpu")
    with torch.no_grad():
        base = p3(batch)
        with_rel = bridge(batch, pairs)
    assert float((with_rel - base).abs().max()) == 0.0
    assert torch.equal(with_rel, base)


def test_nonzero_scale_makes_the_relation_change_the_prediction():
    objs = [chain(6), ring(7), branched()]
    p3 = bp.build_bjoc_null(3, NORM, "cpu").eval()
    bridge = pb.build_bridge(3, NORM, "cpu").eval()
    batch = batch_of(objs)
    pairs_r = pb.build_pair_set(objs, "R").batch([0, 1, 2], "cpu")
    pairs_u = pb.build_pair_set(objs, "U").batch([0, 1, 2], "cpu")
    with torch.no_grad():
        base = p3(batch)
        bridge.rel_scale.fill_(0.1)
        pred_r = bridge(batch, pairs_r)
        pred_u = bridge(batch, pairs_u)
    assert float((pred_r - base).abs().max()) > 0.0
    assert float((pred_r - pred_u).abs().max()) > 0.0


def test_relation_parameters_receive_gradients_after_the_first_step():
    objs = [chain(6), ring(7), branched(), chain(5)]
    model = pb.build_bridge(5, NORM, "cpu")
    pairs = pb.build_pair_set(objs, "R")
    batch = batch_of(objs)
    rel = pairs.batch([0, 1, 2, 3], "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y = torch.tensor([0.1, -0.2, 0.3, 0.4])
    first = {}
    for step in range(2):
        loss = torch.nn.functional.l1_loss(model(batch, rel).view(-1), y)
        opt.zero_grad()
        loss.backward()
        if step == 0:
            first = {
                "pair_mlp": pb._module_grad_norm(model.pair_mlp),
                "rel_scale": float(model.rel_scale.grad.norm()),
            }
        opt.step()
    second_pair = pb._module_grad_norm(model.pair_mlp)
    assert first["rel_scale"] > 0.0
    assert first["pair_mlp"] == 0.0  # exact zero: gamma multiplies the pair branch
    assert second_pair > 0.0
    assert torch.count_nonzero(model.rel_scale).item() > 0


# ---------------------------------------------------------------------------
# kernels consume no labels and no junction content
# ---------------------------------------------------------------------------


def test_pair_lists_and_model_batch_carry_no_labels():
    objs = [chain(4), ring(5)]
    batch = batch_of(objs)
    assert "y" not in batch
    model = pb.build_bridge(0, NORM, "cpu")
    before = repr(sorted(model.base.state_dict().keys()))
    _ = pb.build_pair_set(objs, "U")
    assert repr(sorted(model.base.state_dict().keys())) == before
    for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
        assert name in batch


# ---------------------------------------------------------------------------
# frozen constants / decision vocabulary
# ---------------------------------------------------------------------------


def test_frozen_config_matches_module_constants():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol"] == pb.PROTOCOL_ID
    assert cfg["split_tag"] == pb.SPLIT_TAG
    assert tuple(cfg["seeds"]) == pb.SEEDS
    assert tuple(cfg["arms"]) == pb.ARMS
    assert cfg["pair_hidden"] == pb.PAIR_HIDDEN
    assert cfg["pair_init_offset"] == pb.PAIR_INIT_OFFSET
    assert cfg["base_parameters"] == pb.BASE_PARAMETERS
    assert cfg["bridge_parameters"] == pb.BRIDGE_PARAMETERS
    assert cfg["meaningful_delta"] == pb.MEANINGFUL_DELTA
    assert cfg["tail_threshold"] == pb.TAIL_THRESHOLD
    assert cfg["top_k_removed"] == pb.TOP_K_REMOVED
    assert cfg["bootstrap_n"] == pb.BOOTSTRAP_N
    assert cfg["bootstrap_seed"] == pb.BOOTSTRAP_SEED
    assert cfg["updates"] == int(bp.RECIPES["R3"]["epochs"]) * 63
    assert tuple(cfg["excluded_rows"]) == pb.EXCLUDED_ROWS


@pytest.mark.parametrize(
    "delta,per_seed,ci,ordinary,expected",
    [
        (-0.006, {"0": -0.005, "1": -0.007}, (-0.009, -0.003), -0.005, "GAIN"),
        (-0.006, {"0": -0.005, "1": 0.001}, (-0.009, -0.001), -0.005, "NEUTRAL"),  # seed sign
        (-0.006, {"0": -0.005, "1": -0.007}, (-0.009, 0.001), -0.005, "NEUTRAL"),  # CI
        (-0.006, {"0": -0.005, "1": -0.007}, (-0.009, -0.003), 0.002, "NEUTRAL"),  # ordinary worse
        (-0.003, {"0": -0.002, "1": -0.004}, (-0.006, -0.001), -0.003, "NEUTRAL"),  # below scale
        (0.006, {"0": 0.005, "1": 0.007}, (0.003, 0.009), 0.005, "LOSS"),
        (0.006, {"0": 0.005, "1": 0.007}, (-0.001, 0.009), 0.005, "NEUTRAL"),
    ],
)
def test_contrast_label_rule(delta, per_seed, ci, ordinary, expected):
    assert pb.contrast_label(delta, per_seed, ci[0], ci[1], ordinary) == expected


def test_replication_requires_negative_mean_and_both_seeds_negative():
    assert pb.replication_status({"0": -0.01, "1": -0.02})["replicated"] is True
    assert pb.replication_status({"0": -0.01, "1": 0.02})["replicated"] is False
    assert pb.replication_status({"0": 0.01, "1": 0.02})["replicated"] is False


@pytest.mark.parametrize(
    "labels,reps,expected",
    [
        (("GAIN", "GAIN", "GAIN"), (True, True, True), "CASE_A_RELATION_ADDS_VALUE"),
        (("NEUTRAL", "GAIN", "NEUTRAL"), (False, True, False), "CASE_B_RELATION_USED_NO_HEADLINE_GAIN"),
        (("GAIN", "NEUTRAL", "GAIN"), (True, False, True), "CASE_C_GENERIC_PAIR_CAPACITY"),
        (("GAIN", "NEUTRAL", "NEUTRAL"), (True, False, False), "CASE_E_R_GAIN_NOT_ATTRIBUTED"),
        (("GAIN", "NEUTRAL", "NEUTRAL"), (False, False, False), "CASE_E_R_GAIN_NOT_ATTRIBUTED"),
        (("NEUTRAL", "NEUTRAL", "GAIN"), (False, False, False), "CASE_D_NO_GAIN_ON_P3"),
        (("LOSS", "LOSS", "NEUTRAL"), (False, False, False), "CASE_D_NO_GAIN_ON_P3"),
    ],
)
def test_bridge_case_rule(labels, reps, expected):
    case = pb.bridge_case(
        label_r_p3=labels[0],
        label_r_u=labels[1],
        label_u_p3=labels[2],
        replicated_r_p3=reps[0],
        replicated_r_u=reps[1],
        replicated_u_p3=reps[2],
    )
    assert case["case"] == expected
    assert case["reasons"]


def test_pair_set_state_roundtrip():
    objs = [chain(5), single_bond(), ring(6)]
    pset = pb.build_pair_set(objs, "U")
    restored = pb.PairSet.from_state(pset.to_state())
    assert restored.n_pairs == pset.n_pairs
    a = pset.batch([0, 2], "cpu")
    b = restored.batch([0, 2], "cpu")
    for key in a:
        assert torch.equal(a[key], b[key])
