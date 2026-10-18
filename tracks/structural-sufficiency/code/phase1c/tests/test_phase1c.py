"""Fast CPU tests for the Phase 1c readout/correspondence controls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[3]
for p in (HERE, HERE.parent / "phase1b"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssrc_data import MolGraph, build_cache, gather_batch  # noqa: E402
from ssrc_models import (  # noqa: E402
    MARGINAL_SIZE_KEYS,
    assert_marginal_size_batch,
    assert_size_only_batch,
    build_model,
    parameter_count,
)
from ssrc_models_p1c import P1CConfig, register_phase1c_models  # noqa: E402
from ssrc_train import _model_batch  # noqa: E402

register_phase1c_models()


def mol_chain(n: int, d: int = 4, cycle: bool = False) -> MolGraph:
    af = np.zeros((n, d), dtype=np.float32)
    af[np.arange(n), np.arange(n) % d] = 1.0
    if cycle:
        bu = np.arange(n, dtype=np.int64)
        bv = (np.arange(n) + 1) % n
    else:
        bu = np.arange(n - 1, dtype=np.int64)
        bv = np.arange(1, n, dtype=np.int64)
    bf = np.zeros((len(bu), 2), dtype=np.float32)
    bf[:, 0] = 1.0
    return MolGraph(af, bu, bv, bf)


def disjoint_union(mols: list[MolGraph]) -> MolGraph:
    atom_blocks = []
    bu, bv, bf = [], [], []
    offset = 0
    for mol in mols:
        atom_blocks.append(mol.atom_feat)
        bu.append(mol.bond_u + offset)
        bv.append(mol.bond_v + offset)
        bf.append(mol.bond_feat)
        offset += mol.n_atoms
    return MolGraph(
        np.concatenate(atom_blocks, axis=0),
        np.concatenate(bu).astype(np.int64),
        np.concatenate(bv).astype(np.int64),
        np.concatenate(bf, axis=0),
    )


def permute_mol(mol: MolGraph, perm: np.ndarray) -> MolGraph:
    n = mol.n_atoms
    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[perm] = np.arange(n)
    af = mol.atom_feat[perm].copy()
    bonds = []
    for k in range(mol.n_bonds):
        u = int(old_to_new[int(mol.bond_u[k])])
        v = int(old_to_new[int(mol.bond_v[k])])
        bonds.append((min(u, v), max(u, v), k))
    bonds.sort()
    bu = np.asarray([b[0] for b in bonds], dtype=np.int64)
    bv = np.asarray([b[1] for b in bonds], dtype=np.int64)
    bf = mol.bond_feat[[b[2] for b in bonds]].copy()
    return MolGraph(af, bu, bv, bf)


def make_batch(name: str, cache, rows, y=None):
    marginal = name == "R1_marginal_size"
    b = gather_batch(cache, rows, y, include_size=True, include_edge_mol=marginal)
    return _model_batch(name, b)


def mol_typed_cycle(n: int) -> MolGraph:
    """Cycle with two bond types, so edge-attribute permutations are visible."""
    af = np.zeros((n, 4), dtype=np.float32)
    af[np.arange(n), np.arange(n) % 4] = 1.0
    bu = np.arange(n, dtype=np.int64)
    bv = (np.arange(n) + 1) % n
    bf = np.zeros((n, 2), dtype=np.float32)
    bf[::2, 0] = 1.0
    bf[1::2, 1] = 1.0
    return MolGraph(af, bu, bv, bf)


def test_parameter_matching() -> None:
    cfg = P1CConfig()
    torch.manual_seed(0)
    s = build_model("S_cap_size", 21, 4, cfg)
    torch.manual_seed(0)
    r = build_model("R1_size", 21, 4, cfg)
    torch.manual_seed(0)
    m = build_model("R1_marginal_size", 21, 4, cfg)
    ns, nr, nm = parameter_count(s), parameter_count(r), parameter_count(m)
    assert nr == nm
    assert abs(ns - nr) / nr < 0.005
    # The matching is not a dead-parameter trick: the extra S_cap_size capacity
    # is in real graph FFNs, and R1_size spends its budget on the relation and
    # edge readout.  Exact counts are frozen in the protocol.
    assert ns == 115_883
    assert nr == 115_977


def test_r1_size_and_marginal_are_parameter_identical() -> None:
    cfg = P1CConfig()
    torch.manual_seed(0)
    r = build_model("R1_size", 4, 2, cfg)
    torch.manual_seed(0)
    m = build_model("R1_marginal_size", 4, 2, cfg)
    rs = r.state_dict()
    ms = m.state_dict()
    assert list(rs) == list(ms)
    assert all(torch.equal(rs[k], ms[k]) for k in rs)
    assert parameter_count(r) == parameter_count(m)


def test_shared_trunk_initialisation() -> None:
    cfg = P1CConfig()
    trunks = {}
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(0)
        model = build_model(name, 4, 2, cfg)
        trunks[name] = {k: v.clone() for k, v in model.trunk.state_dict().items()}
    for name in ["R1_size", "R1_marginal_size"]:
        assert all(torch.equal(trunks["S_cap_size"][k], trunks[name][k]) for k in trunks["S_cap_size"])


def test_scap_size_ignores_root_edges_and_guard() -> None:
    cfg = P1CConfig()
    cache = build_cache([mol_chain(7), mol_chain(6, cycle=True)])
    raw = gather_batch(cache, [0, 1], torch.tensor([0.0, 1.0]), include_size=True)
    b = _model_batch("S_cap_size", raw)
    assert_size_only_batch(b)
    try:
        assert_size_only_batch({**b, "root_edge_index": torch.zeros((2, 0), dtype=torch.long)})
        raised = False
    except AssertionError:
        raised = True
    assert raised
    torch.manual_seed(0)
    model = build_model("S_cap_size", 4, 2, cfg)
    model.eval()
    b_clean = _model_batch("S_cap_size", b)
    b_mut = dict(b)
    b_mut["root_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
    b_mut["root_edge_attr"] = torch.zeros((0, 2))
    b_mut = _model_batch("S_cap_size", b_mut)
    with torch.no_grad():
        assert torch.equal(model(b_clean), model(b_mut))


def test_marginal_has_no_endpoint_keys() -> None:
    assert "root_edge_index" not in MARGINAL_SIZE_KEYS
    assert "root_batch" not in MARGINAL_SIZE_KEYS
    assert "root_edge_mol" in MARGINAL_SIZE_KEYS
    assert "root_edge_unique" in MARGINAL_SIZE_KEYS
    cache = build_cache([mol_chain(7)])
    b = gather_batch(cache, [0], torch.tensor([0.0]), include_size=True, include_edge_mol=True)
    b = {k: v for k, v in b.items() if k in MARGINAL_SIZE_KEYS}
    assert_marginal_size_batch(b)
    try:
        assert_marginal_size_batch({**b, "root_edge_index": torch.zeros((2, 0), dtype=torch.long)})
        raised = False
    except AssertionError:
        raised = True
    assert raised


def _edge_mol(name: str, b):
    if name == "R1_marginal_size":
        return b["root_edge_mol"]
    return b["root_batch"][b["root_edge_index"][0]]


def _permute_physical_bond_attr(b, mol_ids: torch.Tensor, seed: int = 7):
    """Permute bond types across physical bonds, keeping both directions equal."""
    rng = np.random.default_rng(seed)
    attr = b["root_edge_attr"].clone()
    mol_np = mol_ids.numpy()
    if "root_edge_unique" in b:
        unique_mask = b["root_edge_unique"]
    else:
        unique_mask = b["root_edge_index"][0] < b["root_edge_index"][1]
    unique = np.flatnonzero(unique_mask.numpy())
    for mol in np.unique(mol_np):
        idx = unique[mol_np[unique] == mol]
        perm = rng.permutation(idx.size)
        for target, source in zip(idx, idx[perm]):
            value = b["root_edge_attr"][source]
            attr[target] = value
            attr[target ^ 1] = value  # the reverse directed record of the same physical bond
    return attr


def test_endpoint_correspondence_ablation() -> None:
    """R1_size is sensitive to which bond type sits on which endpoints;
    R1_marginal_size is not, because it never sees endpoints."""
    cfg = P1CConfig()
    cache = build_cache([mol_typed_cycle(8), mol_typed_cycle(7)])
    y = torch.tensor([0.0, 1.0])
    deltas = {}
    for name in ["R1_size", "R1_marginal_size"]:
        b = make_batch(name, cache, [0, 1], y)
        torch.manual_seed(3)
        model = build_model(name, 4, 2, cfg)
        model.eval()
        b_perm = dict(b)
        b_perm["root_edge_attr"] = _permute_physical_bond_attr(b, _edge_mol(name, b))
        with torch.no_grad():
            deltas[name] = float((model(b) - model(b_perm)).abs().max())
    assert deltas["R1_size"] > 1e-5
    assert deltas["R1_marginal_size"] <= 1e-6


def test_permutation_invariance() -> None:
    cfg = P1CConfig()
    mol = mol_chain(7, cycle=True)
    perm = np.array([3, 0, 6, 1, 5, 2, 4], dtype=np.int64)
    cache = build_cache([mol])
    cache_p = build_cache([permute_mol(mol, perm)])
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(1)
        model = build_model(name, 4, 2, cfg)
        model.eval()
        b = make_batch(name, cache, [0], torch.tensor([0.0]))
        bp = make_batch(name, cache_p, [0], torch.tensor([0.0]))
        with torch.no_grad():
            assert abs(float(model(b).item() - model(bp).item())) <= 1e-5


def test_batch_composition_invariance() -> None:
    cfg = P1CConfig()
    cache = build_cache([mol_chain(7), mol_chain(6, cycle=True), mol_chain(8)])
    y = torch.tensor([0.0, 1.0, -1.0])
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(2)
        model = build_model(name, 4, 2, cfg)
        model.eval()
        b1 = make_batch(name, cache, [0], torch.tensor([0.0]))
        b3 = make_batch(name, cache, [0, 1, 2], y)
        with torch.no_grad():
            assert abs(float(model(b1)[0].item() - model(b3)[0].item())) <= 1e-5


def test_size_readout_separates_duplicate_graph() -> None:
    """Mean-only S0 cannot see token multiplicity; size-aware arms can."""
    cfg = P1CConfig()
    g = mol_chain(8, cycle=True)
    g2 = disjoint_union([g, g])
    cache = build_cache([g, g2])
    y = torch.tensor([0.0, 0.0])
    # S0 has no size features; its mean over 2G equals its mean over G.
    torch.manual_seed(4)
    s0 = build_model("S0", 4, 2, cfg)
    s0.eval()
    b0 = _model_batch("S0", gather_batch(cache, [0, 1], y))
    with torch.no_grad():
        assert abs(float(s0(b0)[0].item() - s0(b0)[1].item())) <= 1e-6
    deltas = {}
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(4)
        model = build_model(name, 4, 2, cfg)
        model.eval()
        b = make_batch(name, cache, [0, 1], y)
        with torch.no_grad():
            deltas[name] = abs(float(model(b)[0].item() - model(b)[1].item()))
        assert deltas[name] > 1e-5
    assert all(v > 1e-5 for v in deltas.values())


def test_size_features_and_edge_keys() -> None:
    cache = build_cache([mol_chain(7, cycle=True), mol_chain(5)])
    b = gather_batch(cache, [0, 1], include_size=True, include_edge_mol=True)
    assert b["mol_size_feat"].shape == (2, 3)
    # rings = n_bonds - n_atoms + 1 for a connected graph
    assert torch.allclose(b["mol_size_feat"][0], torch.tensor([7 / 10, 7 / 10, 1.0]))
    assert b["root_edge_mol"].shape[0] == b["root_edge_attr"].shape[0]
    assert b["root_edge_unique"].dtype == torch.bool
    assert int(b["root_edge_unique"].sum()) == int(b["root_edge_attr"].shape[0] // 2)


def test_forward_and_gradients_live() -> None:
    cfg = P1CConfig()
    cache = build_cache([mol_chain(7), mol_chain(6, cycle=True), mol_chain(8)])
    y = torch.tensor([0.0, 1.0, -1.0])
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(5)
        model = build_model(name, 4, 2, cfg)
        b = make_batch(name, cache, [0, 1, 2], y)
        out = model(b)
        assert out.shape == (3,)
        assert torch.isfinite(out).all()
        out.sum().backward()
        if name == "S_cap_size":
            assert model.graph_ffns[0].net[0].weight.grad is not None
            assert float(model.graph_ffns[0].net[0].weight.grad.abs().sum()) > 0
        else:
            assert float(model.rel.net[0].weight.grad.abs().sum()) > 0
            assert float(model.edge_readout.net[0].weight.grad.abs().sum()) > 0
