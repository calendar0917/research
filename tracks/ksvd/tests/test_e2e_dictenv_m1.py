"""Focused CPU tests for E2E-DictEnv-M1 (no OGB download, no GPU)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_m1 as m1


def _random_dictionary(seed: int = 0, k: int = 32) -> np.ndarray:
    rng = np.random.RandomState(seed)
    D = rng.randn(m1.PHI_DIM, k).astype(np.float32)
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-6)


def _synthetic_payloads():
    from tracks.ksvd.experiments.luyin16 import molhiv_e2e_dictenv_m1 as run
    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

    payloads = []
    for edges in ([(0, 1), (1, 2), (2, 3), (3, 4)], [(0, 1), (1, 2), (2, 0)]):
        n = 1 + max(max(a, b) for a, b in edges)
        graph = from_edges(n, edges)
        atom = np.zeros((n, 9), dtype=np.int64)
        edge = {tuple(sorted((a, b))): np.asarray([a % 5, b % 6, 0], dtype=np.int64) for a, b in edges}
        payload, _clip = run._per_graph_payload(
            graph, atom, edge, 0.0, topology_row=ztopo.compute_features(graph)
        )
        payloads.append(payload)
    return payloads


def test_parameter_accounting_matches_actual():
    total = m1.total_parameter_count()
    model = m1.build_model(_random_dictionary(), seed=0)
    actual = int(sum(p.numel() for p in model.parameters()))
    assert actual == int(total["whole_model"])
    assert m1.PARAM_BUDGET_MIN <= actual <= m1.PARAM_BUDGET_MAX


def test_anchor_layout_and_semantics():
    q = torch.randn(4, m1.ATOM_CHEM_DIM)
    b = torch.randn(3, m1.BOND_CHEM_DIM)
    occ_node = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    occ_root = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    bond_root = torch.tensor([0, 1, 2], dtype=torch.long)
    anchor = m1.build_anchor_raw(q, occ_root, q, occ_node, bond_root, b, 4)
    assert anchor.shape == (4, m1.ANCHOR_DIM)
    assert torch.allclose(anchor[:, m1.ANCHOR_ROOT], q)
    assert torch.allclose(anchor[0, m1.ANCHOR_ATOM_MASS], q[0] + q[1])
    # build_anchor_raw must not accept a shell argument (no shell-conditioning)
    import inspect

    assert "shell" not in inspect.signature(m1.build_anchor_raw).parameters


def test_pair_relation_is_topology_only_and_clips():
    graph = from_edges(8, [(i, i + 1) for i in range(7)])
    nodes = [set([i, i + 1]) for i in range(7)] + [set([7])]
    boundaries = [set() for _ in range(8)]
    pair_index, relation, stats = m1.pure_topology_pair_relation(graph, nodes, boundaries)
    assert relation.shape[1] == m1.RELATION_WIDTH == 15
    assert pair_index.shape[0] == 2
    # node 0 and node 7 are distance 7 > 5 -> clipped into the top bucket
    pair = (0, 7)
    pairs = [tuple(pair_index[:, i].tolist()) for i in range(pair_index.shape[1])]
    index = pairs.index(pair) if pair in pairs else pairs.index((pair[1], pair[0]))
    assert int(np.argmax(relation[index, : m1.DISTANCE_BUCKETS])) == m1.DISTANCE_BUCKETS - 1
    assert stats["clipped_distance_gt_5"] >= 1


def test_forward_and_collate_and_gradient():
    payloads = _synthetic_payloads()
    batch = m1.env_collate(payloads)
    model = m1.build_model(
        _random_dictionary(), seed=0, anchor_mean=torch.zeros(m1.ANCHOR_DIM), anchor_scale=torch.ones(m1.ANCHOR_DIM)
    )
    out = model(batch)
    assert out.shape == (len(payloads),)
    assert torch.isfinite(out).all()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch.y)
    loss.backward()
    assert float(model.D.grad.norm()) > 0.0


def test_exact_sparsity():
    model = m1.build_model(_random_dictionary(), seed=0).eval()
    phi = torch.randn(64, m1.PHI_DIM)
    with torch.no_grad():
        alpha = model.code(phi)
    l0 = (alpha.abs() > 0).sum(dim=1)
    assert int(l0.max()) <= m1.SPARSITY
    assert float((l0 == m1.SPARSITY).float().mean()) >= 0.99


def test_batched_anchor_matches_per_molecule_anchor():
    """The anchor-cache path must equal the per-molecule local-index computation."""
    payloads = _synthetic_payloads()
    model = m1.build_model(_random_dictionary(), seed=0)
    torch.manual_seed(0)
    with torch.no_grad():
        per_molecule = []
        for payload in payloads:
            q = model.atom_chem(payload.dict_atom)
            b = model.bond_chem(payload.env_bond_fields)
            per_molecule.append(
                m1.build_anchor_raw(
                    q,
                    payload.env_occ_root,
                    q,
                    payload.env_occ_node,
                    payload.env_bond_root,
                    b,
                    int(q.shape[0]),
                )
            )
        batch = m1.env_collate(payloads)
        q = model.atom_chem(batch.dict_atom)
        b = model.bond_chem(batch.env_bond_fields)
        batched = m1.build_anchor_raw(
            q,
            batch.env_occ_root,
            q,
            batch.env_occ_node,
            batch.env_bond_root,
            b,
            int(q.shape[0]),
        )
    assert batched.shape[0] == sum(part.shape[0] for part in per_molecule)
    assert torch.allclose(batched, torch.cat(per_molecule, dim=0), atol=1e-6)


def test_no_forbidden_local_descriptors_in_source():
    forbidden = {"patch_cont", "atom_shell", "bond_shell"}
    for module in (m1,):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert not (names & forbidden)
