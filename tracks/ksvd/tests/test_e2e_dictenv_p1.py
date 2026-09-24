"""Focused CPU tests for E2E-DictEnv-P1 (no ZINC, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1


def _synthetic_data(n: int, edges: list[tuple[int, int]], *, seed: int = 0):
    graph = from_edges(n, edges)
    edge_types = {graph.edge_key(a, b): (index % p1.BOND_CATEGORIES) for index, (a, b) in enumerate(edges)}
    incidence = v0.env_incidence(graph, edge_types)
    rng = np.random.default_rng(seed)
    data = Data()
    data.num_nodes = n
    data.dict_phi = torch.as_tensor(rng.standard_normal((n, p1.PHI_DIM)).astype(np.float32))
    data.dict_atom = torch.as_tensor(rng.integers(0, p1.ATOM_CATEGORIES, size=n), dtype=torch.long)
    data.env_occ_node = incidence["occ_node"]
    data.env_occ_root = incidence["occ_root"]
    data.env_occ_shell = incidence["occ_shell"]
    data.env_bond_root = incidence["bond_root"]
    data.env_bond_shellpair = incidence["bond_shellpair"]
    data.env_bond_type = incidence["bond_type"]
    data.env_bond_u = incidence["bond_u"]
    data.env_bond_v = incidence["bond_v"]
    anchor = p1.build_anchor_raw(
        data.dict_atom,
        data.env_occ_node,
        data.env_occ_root,
        data.env_bond_root,
        data.env_bond_type,
        n,
    )
    data.anchor = anchor
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    data.pair_index = torch.as_tensor(np.asarray(pairs, dtype=np.int64).T)
    data.pair_relation = torch.randn(len(pairs), p1.RELATION_WIDTH_RAW)
    data.pair_bucket = torch.as_tensor(rng.integers(0, p1.DISTANCE_BUCKETS, size=len(pairs)), dtype=torch.long)
    data.global_context = torch.randn(1, p1.GLOBAL_WIDTH)
    data.topology_features = torch.randn(1, p1.TOPOLOGY_HIDDEN + 9)
    data.y = torch.tensor([0.5])
    return data


def _synthetic_batch(n_molecules: int = 3):
    molecules = [_synthetic_data(4, [(0, 1), (1, 2), (2, 3)], seed=index) for index in range(n_molecules)]
    return p1.env_collate(molecules)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def test_parameter_accounting_matches_preregistration():
    local = p1.local_parameter_count()
    backend = p1.backend_parameter_count()
    total = p1.total_parameter_count()
    assert local == {
        "dictionary": 2080,
        "node_binding": 5760,
        "edge_binding": 4800,
        "decoder": 70122,
        "subtotal": 82762,
    }
    assert backend["subtotal"] == 15103
    assert total["whole_model"] == 97865
    assert total["within_budget"] is True
    assert p1.ENV_MLP_IN == 638
    assert p1.ANCHOR_DIM == 62
    assert p1.RELATION_WIDTH == 15


def test_model_parameter_counts_and_init_matching():
    sparse = p1.build_model(p1.SPARSE_ARM, seed=0)
    dense = p1.build_model(p1.DENSE_ARM, seed=0, reference_state=sparse.state_dict())
    expected = p1.total_parameter_count()["whole_model"]
    assert sum(parameter.numel() for parameter in sparse.parameters()) == expected
    assert sum(parameter.numel() for parameter in dense.parameters()) == expected
    sparse_state, dense_state = sparse.state_dict(), dense.state_dict()
    assert list(sparse_state.keys()) == list(dense_state.keys())
    for key in sparse_state:
        assert torch.equal(sparse_state[key], dense_state[key]), key
    assert torch.equal(sparse_state["D"], dense_state["D"])


# ---------------------------------------------------------------------------
# forward / purity
# ---------------------------------------------------------------------------


def test_forward_shapes_and_sparsity():
    batch = _synthetic_batch(3)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    prediction, aux = model(batch, return_aux=True)
    assert prediction.shape == (3,)
    assert aux["E"].shape == (12, p1.ENV_DIM)
    assert aux["coord"].shape == (12, p1.K_ATOMS)
    assert batch.anchor.shape == (12, p1.ANCHOR_DIM)
    l0 = (aux["coord"].abs() > 0).sum(dim=1)
    assert int(l0.max()) <= p1.SPARSITY


def test_coord_zero_removes_phi_dependence():
    batch = _synthetic_batch(2)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    mutated = batch.clone()
    mutated.dict_phi = torch.randn_like(mutated.dict_phi)
    with torch.no_grad():
        _p0, aux_a = model(batch, coord_zero=True, return_aux=True)
        _p1, aux_b = model(mutated, coord_zero=True, return_aux=True)
        _p2, aux_c = model(mutated, coord_zero=False, return_aux=True)
    assert torch.equal(aux_a["E"], aux_b["E"])
    assert not torch.equal(aux_b["E"], aux_c["E"])


def test_edge_role_symmetry():
    batch = _synthetic_batch(2)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        e_ab = model.edge_environment(coord, batch, batch.env_bond_u, batch.env_bond_v)
        e_ba = model.edge_environment(coord, batch, batch.env_bond_v, batch.env_bond_u)
    assert torch.allclose(e_ab, e_ba, atol=1e-6)


def test_shuffle_preserves_multisets_and_breaks_binding():
    data = _synthetic_data(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)], seed=3)
    occ_node, occ_root, occ_shell = data.env_occ_node, data.env_occ_root, data.env_occ_shell
    shuffled = p1.shuffled_occ_node_for_molecule(occ_node, occ_root, occ_shell, seed=101)
    # alpha-node multiset preserved per (root, shell)
    assert sorted(shuffled.tolist()) == sorted(occ_node.tolist())
    # the mapping is a permutation within each group
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(len(occ_node)):
        groups.setdefault((int(occ_root[index]), int(occ_shell[index])), []).append(index)
    for positions in groups.values():
        assert sorted(shuffled[positions].tolist()) == sorted(occ_node[positions].tolist())

    bond_root, bond_sp = data.env_bond_root, data.env_bond_shellpair
    u, v = data.env_bond_u, data.env_bond_v
    su, sv = p1.shuffled_bond_endpoints_for_molecule(bond_root, bond_sp, u, v, seed=202)
    bond_groups: dict[tuple[int, int], list[int]] = {}
    for index in range(len(bond_root)):
        bond_groups.setdefault((int(bond_root[index]), int(bond_sp[index])), []).append(index)
    for positions in bond_groups.values():
        assert sorted(zip(su[positions].tolist(), sv[positions].tolist())) == sorted(
            zip(u[positions].tolist(), v[positions].tolist())
        )
    # bond types unchanged by definition; combined role shuffle changes E
    batch_clean = p1.env_collate([data])
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    with torch.no_grad():
        _p, aux_clean = model(batch_clean, return_aux=True)
        _p2, aux_edge = model(batch_clean, bond_u=su, bond_v=sv, return_aux=True)
    assert not torch.equal(aux_clean["E"], aux_edge["E"])


def test_relation_chemistry_blocker():
    indices = list(p1.P1_RELATION_INDICES)
    assert indices == list(range(14)) + [18]
    relation = torch.randn(7, p1.RELATION_WIDTH_RAW)
    mutated = relation.clone()
    mutated[:, 14:18] = torch.randn_like(mutated[:, 14:18])  # path bond mean
    mutated[:, 19:23] = torch.randn_like(mutated[:, 19:23])  # adjacent bond type
    assert torch.equal(relation[:, indices], mutated[:, indices])


def test_environment_freeze_under_pair_relation_mutation():
    batch = _synthetic_batch(2)
    model = p1.build_model(p1.SPARSE_ARM, seed=0).eval()
    mutated = batch.clone()
    mutated.pair_relation = torch.randn_like(mutated.pair_relation) * 3.0
    with torch.no_grad():
        _p0, aux_a = model(batch, return_aux=True)
        _p1, aux_b = model(mutated, return_aux=True)
    assert torch.equal(aux_a["E"], aux_b["E"])


def test_task_gradient_reaches_dictionary():
    batch = _synthetic_batch(4)
    model = p1.build_model(p1.SPARSE_ARM, seed=0)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))
    loss.backward()
    assert model.D.grad is not None
    assert float(model.D.grad.norm()) > 0.0
    assert float(model.W_A_S.grad.norm()) > 0.0
    assert float(model.W_E_S.grad.norm()) > 0.0


def test_anchor_semantics():
    # anchor depends only on root identity, unconditioned mass and size
    data = _synthetic_data(5, [(0, 1), (1, 2), (2, 3), (3, 4)], seed=7)
    raw = data.anchor
    n = data.num_nodes
    q = torch.nn.functional.one_hot(data.dict_atom, num_classes=p1.ATOM_CATEGORIES).float()
    assert torch.equal(raw[:, p1.ANCHOR_ROOT], q)
    atom_mass = raw[:, p1.ANCHOR_ATOM_MASS]
    assert torch.allclose(atom_mass.sum(dim=1), raw[:, p1.ANCHOR_SIZE.start].exp() - 1.0, atol=1e-4)
    bond_mass = raw[:, p1.ANCHOR_BOND_MASS]
    assert torch.allclose(bond_mass.sum(dim=1), raw[:, p1.ANCHOR_SIZE.start + 1].exp() - 1.0, atol=1e-4)
    # no shell conditioning: relabelling the shell index leaves the anchor unchanged
    assert raw.shape[1] == p1.ANCHOR_DIM


def test_module_source_is_clean():
    import ast

    tree = ast.parse(Path(p1.__file__).read_text(encoding="utf-8"))
    forbidden = {"patch_cont", "atom_shell", "bond_shell"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.Name):
            seen.add(node.id)
    assert not (seen & forbidden)
