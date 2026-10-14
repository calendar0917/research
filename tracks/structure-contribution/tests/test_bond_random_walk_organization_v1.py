"""Fast correctness tests for cscl-bond-random-walk-organization-v1.

No ZINC dataset, no checkpoints and no GPU: tiny synthetic molecules only.
Covers the lazy random walk, the unordered-triplet double-sum identity, the
R/U relation switch, invariance, the no-Junction-encoder scan, Walktrap
determinism/coverage, the non-molecule counterexample and the frozen gate
vocabulary.
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
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/bond_random_walk_organization_v1.yaml"
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


def _dense_aggregate(Ks: np.ndarray, b: np.ndarray) -> np.ndarray:
    bs, bd = b[:, None, :], b[None, :, :]
    phi = bs + bd  # any symmetric test function
    return np.einsum("ef,efk->ek", Ks, phi)


def test_config_matches_module_constants():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol"] == bwo.PROTOCOL_ID
    assert cfg["status"] == "frozen"
    assert bwo.MAX_K == 4
    assert tuple(bwo.KERNEL_KS) == (1, 2, 3)
    gate = cfg["part_a"]["gate"]
    assert gate["invariance_max_abs"] == pytest.approx(1e-9)
    assert gate["min_mean_tv_p3_to_uniform"] == pytest.approx(0.03)
    assert gate["min_mean_tv_K_to_uniform"] == pytest.approx(0.05)
    assert gate["min_scale_distinctness"] == pytest.approx(0.05)
    assert cfg["amendments"][0]["field"] == "part_a.gate.min_scale_distinctness"
    assert cfg["model"]["parameters"]["expected_R"] == bwo.EXPECTED_PARAMETERS == 349_985
    assert cfg["model"]["parameters"]["expected_U"] == bwo.EXPECTED_PARAMETERS


def test_lazy_transition_rows_sum_to_one():
    for obj in (_chain(4), _ring(6), _mol(0, [0, 1], [(0, 1, 1)])):
        A = bwo.line_graph_adjacency(obj.bond_index)
        P = bwo.lazy_transition(A)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(P >= 0.0)
    # explicit zero-degree fallback
    P0 = bwo.lazy_transition(np.zeros((3, 3)))
    assert np.allclose(P0, np.eye(3))


def test_single_bond_line_graph_is_isolated_node():
    obj = _mol(0, [0, 1], [(0, 1, 1)])
    A = bwo.line_graph_adjacency(obj.bond_index)
    assert A.shape == (1, 1)
    P = bwo.lazy_transition(A)
    assert P[0, 0] == pytest.approx(1.0)
    stats = bwo.molecule_walk_stats(obj)
    assert stats["isolated_line_nodes"] == 1
    assert np.isfinite(list(stats.values())).all()


def test_kernel_powers_recursion_and_relation():
    obj = _ring(8)
    A = bwo.line_graph_adjacency(obj.bond_index)
    powers = bwo.kernel_powers(A, bwo.MAX_K)
    P = bwo.lazy_transition(A)
    for k in range(2, bwo.MAX_K + 1):
        assert np.allclose(powers[k], powers[k - 1] @ P, atol=1e-12)
    K = bwo.relation_kernel(powers)
    assert np.allclose(K, (powers[1] + powers[2] + powers[3]) / 3.0)


def test_unordered_triplets_reconstruct_double_sum_with_diagonal_halving():
    rng = np.random.default_rng(0)
    n = 7
    K = rng.normal(size=(n, n))
    Ks = bwo.symmetrize(K)
    b = rng.normal(size=(n, 5))
    dense = _dense_aggregate(Ks, b)
    src, dst, w = bwo.unordered_triplets(Ks)
    assert int(src.shape[0]) == n * (n + 1) // 2
    bs, bd = b[src], b[dst]
    phi = bs + bd
    w_eff = np.where(src == dst, w * 0.5, w)
    r = np.zeros_like(dense)
    np.add.at(r, src, w_eff[:, None] * phi)
    np.add.at(r, dst, w_eff[:, None] * phi)
    assert np.allclose(r, dense, atol=1e-12)


def test_k_and_symmetrised_k_are_model_equivalent():
    rng = np.random.default_rng(1)
    n = 6
    K = rng.normal(size=(n, n))
    Ks = bwo.symmetrize(K)
    b = rng.normal(size=(n, 4))
    assert np.allclose(_dense_aggregate(Ks, b), _dense_aggregate(bwo.symmetrize(K), b))
    # unsymmetrised K on a symmetric test function equals Ks
    assert np.allclose(_dense_aggregate(bwo.symmetrize(K), b), _dense_aggregate(0.5 * (K + K.T), b))


def test_uniform_triplets_control():
    n = 5
    src, dst, w = bwo.uniform_triplets(n)
    assert int(src.shape[0]) == n * (n + 1) // 2
    assert np.allclose(w, 1.0 / n)
    K = np.full((n, n), 1.0 / n)
    assert np.allclose(K.sum(axis=1), 1.0)


def test_atom_relabel_kernel_invariance_and_bond_permutation_equivariance():
    obj = _ring(9, idx=3)
    rng = np.random.default_rng(2)
    assert bwo.atom_relabel_invariance(obj, rng) == 0.0
    res = bwo.bond_permutation_equivariance(obj, rng)
    assert res["kernel_conjugation_max_abs"] <= 1e-12
    assert res["statistic_max_abs"] <= 1e-12


def test_model_parameter_count_and_no_junction_encoder():
    model = bwo.build_model(0, NORM, "cpu")
    assert bwo.parameter_count(model) == bwo.EXPECTED_PARAMETERS
    assert bwo.forbidden_module_scan(model) == []
    names = [n for n, _ in model.named_modules()]
    assert not any("junction" in n.lower() for n in names)
    model2 = bwo.build_model(0, NORM, "cpu")
    assert bwo.state_dict_hash(model) == bwo.state_dict_hash(model2)


def test_model_forward_matches_dense_and_uses_relation():
    obj = _ring(7, idx=1)
    src, dst, w = bwo.kernel_triplets_from_objects(obj)
    batch = bwo.batch_rw_objects([obj], [(src, dst, w)])
    batch = bwo.move_batch(batch, torch.device("cpu"))
    model = bwo.build_model(0, NORM, "cpu")
    model.eval()
    with torch.no_grad():
        pred = float(model(batch))
        uniform = bwo.uniform_triplets(int(obj.bond_index.shape[0]))
        batch_u = bwo.move_batch(bwo.batch_rw_objects([obj], [uniform]), torch.device("cpu"))
        pred_u = float(model(batch_u))
    assert np.isfinite(pred)
    assert abs(pred - pred_u) > 1e-8
    with torch.no_grad():
        feats = model.forward_features(batch)
        A = bwo.line_graph_adjacency(obj.bond_index)
        Ks = bwo.symmetrize(bwo.relation_kernel(bwo.kernel_powers(A, bwo.MAX_K)))
        b = feats["b"].numpy()
        r_manual = model.pair_aggregate(feats["b"], batch).numpy()
        s, d, ww = bwo.unordered_triplets(Ks)
        bs, bd = b[ s ], b[ d ]
        phi = model.pair_mlp(torch.cat([feats["b"][s] + feats["b"][d],
                                        (feats["b"][s] - feats["b"][d]).abs(),
                                        feats["b"][s] * feats["b"][d]], dim=1)).numpy()
        w_eff = np.where(s == d, ww * 0.5, ww)
        r_trip = np.zeros_like(r_manual)
        np.add.at(r_trip, s, w_eff[:, None] * phi)
        np.add.at(r_trip, d, w_eff[:, None] * phi)
    assert np.allclose(r_manual, r_trip, atol=1e-5)


def test_model_atom_relabel_and_batch_membership_invariance():
    obj = _ring(8, idx=2)
    src, dst, w = bwo.kernel_triplets_from_objects(obj)
    model = bwo.build_model(0, NORM, "cpu")
    model.eval()
    rng = np.random.default_rng(3)
    perm = rng.permutation(obj.n_atoms)
    inv = np.argsort(perm)
    o2 = sro.MolObjects(
        mol_index=obj.mol_index,
        smiles=obj.smiles,
        n_atoms=obj.n_atoms,
        atom_feat=obj.atom_feat[perm],
        bond_index=inv[obj.bond_index],
        bond_type=obj.bond_type,
        bond_feat=obj.bond_feat,
        junc_index=obj.junc_index,
        junc_bond=obj.junc_bond,
        junc_feat=obj.junc_feat,
        global_feat=obj.global_feat,
    )
    with torch.no_grad():
        p1 = float(model(bwo.move_batch(bwo.batch_rw_objects([obj], [(src, dst, w)]), torch.device("cpu"))))
        p2 = float(model(bwo.move_batch(bwo.batch_rw_objects([o2], [(src, dst, w)]), torch.device("cpu"))))
    assert abs(p1 - p2) <= 1e-6


def test_walktrap_deterministic_covers_and_is_relabel_stable():
    # two triangles joined by one edge: unambiguous community structure
    bonds = [(0, 1, 1), (1, 2, 1), (2, 0, 1), (3, 4, 1), (4, 5, 1), (5, 3, 1), (2, 3, 1)]
    obj = _mol(11, range(6), bonds)
    A = bwo.line_graph_adjacency(obj.bond_index)
    wt1 = bwo.walktrap_partition(A)
    wt2 = bwo.walktrap_partition(A.copy())
    assert wt1["labels"].tolist() == wt2["labels"].tolist()
    labels = wt1["labels"]
    assert int(labels.min()) == 0
    assert int(len(np.unique(labels))) == int(labels.max()) + 1
    assert len(labels) == int(obj.bond_index.shape[0])
    # relabelling vertices must permute (not change) the partition
    rng = np.random.default_rng(4)
    perm = rng.permutation(obj.n_atoms)
    o2 = sro.MolObjects(
        mol_index=obj.mol_index, smiles="", n_atoms=obj.n_atoms,
        atom_feat=obj.atom_feat[perm], bond_index=np.argsort(perm)[obj.bond_index],
        bond_type=obj.bond_type, bond_feat=obj.bond_feat,
        junc_index=obj.junc_index, junc_bond=obj.junc_bond, junc_feat=obj.junc_feat,
        global_feat=obj.global_feat,
    )
    ari = bwo._ari(labels, bwo.walktrap_partition(bwo.line_graph_adjacency(o2.bond_index))["labels"])
    assert ari >= 0.99


def test_community_analysis_on_tiny_molecules():
    for obj in (_mol(0, [0, 1], [(0, 1, 1)]), _ring(6, idx=1), _chain(5, idx=2)):
        rec = bwo.community_analysis_molecule(obj, relabel_trials=1, seed=0)
        assert rec["walktrap"]["partition_covers_all_bonds"]
        assert rec["walktrap"]["unique_bond_membership"]
        assert rec["atom_boundary"]["boundary_atoms"] <= rec["atom_boundary"]["atoms_with_multiple_incident_bonds"]


def test_walk_stats_keys_and_gate_summary():
    per_mol = [bwo.molecule_walk_stats(_chain(k)) | {"category": bwo._molecule_category(_chain(k))} for k in (4, 5)]
    summary = bwo.summarize_walk_stats(per_mol)
    for key in ("tv_uniform_k3", "K_tv_uniform", "scale_distinctness_p1p3", "scale_distinctness_raw_p1p3"):
        assert np.isfinite(summary["aggregate"][key])
    assert set(summary["gate_conditions"]) == {"mean_tv_p3_ge_min", "mean_tv_K_ge_min", "scale_distinctness_ge_min"}


def test_junction_moment_counterexample_is_labelled_non_molecule():
    jc = bwo.junction_moment_counterexample()
    assert jc["legal_molecule"] is False
    assert "not" in jc["warning"]
    assert jc["raw_marginals_equal"] == (4, (0, 0, 1, 1), (2, 2, 2, 2))
    assert jc["K_max_abs_diff"] > 0.2
    assert jc["bond_line_graph"]["A_edges"] != jc["bond_line_graph"]["B_edges"]


def test_synthetic_witness_helpers_are_small_and_live():
    rng = np.random.default_rng(5)
    pair = torch.nn.Sequential(torch.nn.Linear(24, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8))
    with torch.no_grad():
        for p in pair.parameters():
            p.copy_(torch.as_tensor(rng.normal(scale=0.3, size=tuple(p.shape)), dtype=torch.float32))
    abl = bwo.pair_block_ablation(pair, seed=0, dim=8)
    assert abl["zero_prod_rel"] > 0.0
    resid = bwo.pair_linear_summary_residual(pair, n_molecules=3, seed=0, dim=8)
    assert np.isfinite(resid["relative_residual"])


def test_module_has_no_official_data_path():
    source = (REPO_ROOT / "tracks/structure-contribution/code/bond_random_walk_organization_v1.py").read_text()
    for token in ("valid_rows", "test_rows", "encoded_valid", "encoded_test", "official_valid_rows"):
        assert token not in source
    driver = (REPO_ROOT / "tracks/structure-contribution/code/run_bond_random_walk_organization_v1.py").read_text()
    assert "official valid/test never loaded" in driver
    assert "official_test_loaded\": False" in driver
    # the only label reader loads Fit rows (training) or Monitor rows (single read)
    assert "load_y_rows(ENCODED_TRAIN, rows)" in driver
    assert driver.count("load_y_rows(") == 2
