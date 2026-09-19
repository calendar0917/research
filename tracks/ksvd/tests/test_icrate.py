"""Targeted tests for I-CRATE-v0 (pre-registration §7, tests A-J).

CPU-only, synthetic-data only (no ``data/``, no checkpoints, no official split).
They assert the algebraic properties the pre-registration requires and do not
train.
"""

from __future__ import annotations

import torch

from tracks.ksvd.experiments.luyin16 import icrate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _toy() -> icrate.Molecule:
    return icrate.molecule_from_arrays(
        [0, 1, 2, 3, 4, 0],
        [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 3), (4, 5, 2)],
        y=1.0,
    )


def _model() -> icrate.ICrateV0:
    torch.manual_seed(0)
    return icrate.ICrateV0(seed=0)


# ---------------------------------------------------------------------------
# A. undirected bond deduplication
# ---------------------------------------------------------------------------


def test_undirected_bond_deduplication() -> None:
    # atom0 -b1- atom1 -b2- atom2, stored as two directed entries per bond.
    atom_types = [0, 1, 2]
    edge_index = [[0, 1, 1, 2], [1, 0, 2, 1]]
    edge_attr = [1, 1, 2, 2]
    mol = icrate.molecule_from_edges(atom_types, edge_index, edge_attr)
    assert mol.n == 3
    assert mol.m == 2  # one token per undirected bond
    assert mol.src.tolist() == [0, 1]
    assert mol.dst.tolist() == [1, 2]
    assert mol.bond.tolist() == [0, 1]  # {1,2} re-indexed


def test_inconsistent_directed_bond_attributes_abort() -> None:
    edge_index = [[0, 1], [1, 0]]
    edge_attr = [1, 2]  # two copies disagree
    try:
        icrate.molecule_from_edges([0, 1], edge_index, edge_attr)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on inconsistent bond attributes")


# ---------------------------------------------------------------------------
# B. incidence mask correctness
# ---------------------------------------------------------------------------


def test_incidence_mask_chain() -> None:
    # atom0 -- bond0 -- atom1 -- bond1 -- atom2
    mol = icrate.molecule_from_arrays([0, 1, 2], [(0, 1, 1), (1, 2, 2)])
    keep = icrate.incidence_keep(mol)
    n = 3
    b0, b1 = n + 0, n + 1
    # atom0 sees only bond0
    assert keep[0, b0] and not keep[0, b1]
    # atom1 sees bond0 and bond1
    assert keep[1, b0] and keep[1, b1]
    # atom2 sees only bond1
    assert keep[2, b1] and not keep[2, b0]
    # bond0 sees atom0/atom1 only
    assert keep[b0, 0] and keep[b0, 1] and not keep[b0, 2]
    # bond1 sees atom1/atom2 only
    assert keep[b1, 1] and keep[b1, 2] and not keep[b1, 0]
    # no atom<->atom, no bond<->bond, no self
    assert not keep[:n, :n].any()
    assert not keep[n:, n:].any()
    assert not torch.diagonal(keep).any()
    assert torch.equal(keep, keep.t())


# ---------------------------------------------------------------------------
# C. permutation equivariance of every structural layer
# ---------------------------------------------------------------------------


def test_permutation_equivariance_layers() -> None:
    model = _model().eval()
    base = _toy()
    with torch.no_grad():
        _, info = model.forward_reference(base)

    # atom permutation: new atom i is old node_atom_perm[i]
    atom_perm = [3, 0, 5, 1, 4, 2]
    inverse = [0] * len(atom_perm)
    for i, p in enumerate(atom_perm):
        inverse[p] = i
    atoms_old = base.atom.tolist()
    new_atoms = [atoms_old[atom_perm[i]] for i in range(len(atom_perm))]
    new_bonds = []
    for e in range(base.m):
        u, v = int(base.src[e]), int(base.dst[e])
        new_bonds.append((inverse[u], inverse[v], int(base.bond[e]) + 1))
    mol_node = icrate.molecule_raw(new_atoms, new_bonds, y=base.y)
    with torch.no_grad():
        _, info_node = model.forward_reference(mol_node)
    for layer, (z_base, z_perm) in enumerate(zip(info.z_layers, info_node.z_layers)):
        expected = z_base[atom_perm]
        assert float((z_perm[: base.n] - expected).abs().max()) < 1e-5, layer

    # bond permutation: new bond j is old bond bond_perm[j]
    bond_perm = [4, 0, 3, 1, 2]
    perm_bonds = []
    for j in range(base.m):
        e = bond_perm[j]
        perm_bonds.append((int(base.src[e]), int(base.dst[e]), int(base.bond[e]) + 1))
    mol_bond = icrate.molecule_raw(base.atom.tolist(), perm_bonds, y=base.y)
    with torch.no_grad():
        _, info_bond = model.forward_reference(mol_bond)
    for layer, (z_base, z_perm) in enumerate(zip(info.z_layers, info_bond.z_layers)):
        expected = z_base[base.n :][bond_perm]
        assert float((z_perm[base.n :] - expected).abs().max()) < 1e-5, layer


# ---------------------------------------------------------------------------
# D. final invariance
# ---------------------------------------------------------------------------


def test_final_invariance() -> None:
    model = _model().eval()
    base = _toy()
    atom_perm = [2, 5, 0, 4, 1, 3]
    bond_perm = [1, 4, 0, 3, 2]
    perm = icrate.molecule_raw(
        [base.atom.tolist()[atom_perm[i]] for i in range(base.n)],
        sorted(
            (
                min(atom_perm.index(int(base.src[e])), atom_perm.index(int(base.dst[e]))),
                max(atom_perm.index(int(base.src[e])), atom_perm.index(int(base.dst[e]))),
                int(base.bond[e]) + 1,
            )
            for e in bond_perm
        ),
        y=base.y,
    )
    with torch.no_grad():
        pred_a, info_a = model.forward_reference(base)
        pred_b, info_b = model.forward_reference(perm)
    assert float((info_a.alpha - info_b.alpha).abs().max()) < 1e-5
    assert float((pred_a - pred_b).abs().max()) < 1e-5


# ---------------------------------------------------------------------------
# E. incidence sensitivity
# ---------------------------------------------------------------------------


def test_incidence_assignment_changes_representation() -> None:
    model = _model().eval()
    # Same atom multiset, same bond multiset, different bond<->endpoint pairs.
    a = icrate.molecule_from_arrays([0, 0, 1, 1], [(0, 1, 1), (2, 3, 2)])
    b = icrate.molecule_from_arrays([0, 0, 1, 1], [(0, 2, 1), (1, 3, 2)])
    with torch.no_grad():
        _, info_a = model.forward_reference(a)
        _, info_b = model.forward_reference(b)
    assert float((info_a.z_layers[-1] - info_b.z_layers[-1]).abs().max()) > 1e-4
    assert float((info_a.alpha - info_b.alpha).abs().max()) > 1e-6


# ---------------------------------------------------------------------------
# F. no-self structural dependency
# ---------------------------------------------------------------------------


def test_no_self_and_neighbor_dependency() -> None:
    model = _model().eval()
    mol = icrate.molecule_from_arrays([0, 1, 2], [(0, 1, 1)])
    keep = icrate.incidence_keep(mol)
    assert not torch.diagonal(keep).any()

    # An isolated atom's MSSA update is exactly zero despite its own feature.
    iso = icrate.molecule_from_arrays([0, 1, 2], [(0, 1, 1)])
    with torch.no_grad():
        _, info = model.forward_reference(iso)
    # atom2 is isolated -> its row has no legal keys -> zero update.
    assert float(info.delta_mssa[0][2].abs().max()) == 0.0

    # Changing an incident neighbour changes the bond token's MSSA update.
    mol2 = icrate.molecule_from_arrays([0, 2, 2], [(0, 1, 1)])
    with torch.no_grad():
        _, info2 = model.forward_reference(mol2)
    bond_token = mol.n  # bond0 index
    assert float((info.delta_mssa[0][bond_token] - info2.delta_mssa[0][bond_token]).abs().max()) > 1e-6


# ---------------------------------------------------------------------------
# G. ODL objective descent
# ---------------------------------------------------------------------------


def _odl_objective(a: torch.Tensor, x: torch.Tensor, d: torch.Tensor, lam: float) -> float:
    recon = a @ d.t()
    return float(0.5 * ((recon - x) ** 2).sum() + lam * a.abs().sum())


def test_token_ista_objective_non_increasing() -> None:
    torch.manual_seed(1)
    x = torch.randn(10, icrate.D_MODEL)
    d = icrate.normalize_columns(torch.randn(icrate.D_MODEL, icrate.M_DICT))
    spec = torch.linalg.matrix_norm(d, ord=2).detach()
    eta = 0.9 / (spec * spec + icrate.EPS)
    a = torch.zeros(x.shape[0], icrate.M_DICT)
    values = [_odl_objective(a, x, d, icrate.LAMBDA_TOK)]
    for _ in range(icrate.R_TOK):
        err = a @ d.t() - x
        a = torch.relu(a - eta * (err @ d) - eta * icrate.LAMBDA_TOK)
        values.append(_odl_objective(a, x, d, icrate.LAMBDA_TOK))
    assert all(values[i + 1] <= values[i] + 1e-9 for i in range(len(values) - 1))


def test_global_ista_objective_non_increasing() -> None:
    torch.manual_seed(2)
    g = torch.randn(icrate.D_MODEL)
    d = icrate.normalize_columns(torch.randn(icrate.D_MODEL, icrate.M_DICT))
    spec = torch.linalg.matrix_norm(d, ord=2).detach()
    eta = 0.9 / (spec * spec + icrate.EPS)
    alpha = torch.zeros(icrate.M_DICT)
    values = [0.5 * float(((alpha @ d.t() - g) ** 2).sum()) + icrate.LAMBDA_G * float(alpha.abs().sum())]
    for _ in range(icrate.R_G):
        err = alpha @ d.t() - g
        alpha = torch.relu(alpha - eta * (err @ d) - eta * icrate.LAMBDA_G)
        values.append(0.5 * float(((alpha @ d.t() - g) ** 2).sum()) + icrate.LAMBDA_G * float(alpha.abs().sum()))
    assert all(values[i + 1] <= values[i] + 1e-9 for i in range(len(values) - 1))


def test_model_token_odl_objective_non_increasing() -> None:
    model = _model().eval()
    batch = icrate.collate([_toy()])
    with torch.no_grad():
        _, info = model(batch)
    # recompute the per-layer ISTA objective across its steps from zero
    for li, layer in enumerate(model.layers):
        x = layer.ln_odl(info.z_half[li])
        d = icrate.normalize_columns(layer.d_a)
        spec = torch.linalg.matrix_norm(d, ord=2).detach()
        eta = 0.9 / (spec * spec + icrate.EPS)
        a = torch.zeros(x.shape[0], icrate.M_DICT)
        values = [_odl_objective(a, x, d, icrate.LAMBDA_TOK)]
        for _ in range(icrate.R_TOK):
            err = a @ d.t() - x
            a = torch.relu(a - eta * (err @ d) - eta * icrate.LAMBDA_TOK)
            values.append(_odl_objective(a, x, d, icrate.LAMBDA_TOK))
        assert all(values[i + 1] <= values[i] + 1e-9 for i in range(len(values) - 1)), li


# ---------------------------------------------------------------------------
# H. exact sparse coefficients
# ---------------------------------------------------------------------------


def test_exact_zeros() -> None:
    model = _model().eval()
    batch = icrate.collate([_toy()])
    with torch.no_grad():
        _, info = model(batch)
    for coeffs in info.coeffs:
        assert float((coeffs == 0).float().sum()) > 0
        assert bool((coeffs >= 0).all())
    assert float((info.alpha == 0).float().sum()) > 0


# ---------------------------------------------------------------------------
# I. batch / reference equivalence
# ---------------------------------------------------------------------------


def test_batch_reference_equivalence() -> None:
    model = _model().eval()
    mols = [
        _toy(),
        icrate.molecule_from_arrays([5, 5, 6, 7], [(0, 1, 1), (1, 2, 3), (2, 3, 2)], y=2.0),
    ]
    batch = icrate.collate(mols)
    nmax = batch.atom_idx.shape[1]
    with torch.no_grad():
        pred_batch, info_batch = model(batch)
        for g, mol in enumerate(mols):
            pred_ref, info_ref = model.forward_reference(mol)
            # batch layout is [Nmax atom slots ; Mmax bond slots]; reference is
            # [n atoms ; m bonds] -- map bond token e to slot nmax + e.
            idx = list(range(mol.n)) + [nmax + e for e in range(mol.m)]
            assert float((pred_batch[g] - pred_ref).abs().max()) < 1e-5
            assert float((info_batch.z_layers[-1][g, idx] - info_ref.z_layers[-1]).abs().max()) < 1e-5
            assert float((info_batch.g0[g] - info_ref.g0).abs().max()) < 1e-5
            assert float((info_batch.alpha[g] - info_ref.alpha).abs().max()) < 1e-5


# ---------------------------------------------------------------------------
# J. gradient flow
# ---------------------------------------------------------------------------


def test_gradient_reaches_all_blocks() -> None:
    torch.manual_seed(0)
    model = icrate.ICrateV0(seed=0)
    batch = icrate.collate([_toy()])
    pred, _ = model(batch)
    loss = torch.nn.functional.l1_loss(pred, batch.y)
    loss.backward()
    blocks = {
        "E_V": model.e_v.weight,
        "E_E": model.e_e.weight,
        "D_G": model.d_g,
        "U_G": model.u_g,
        "head": model.head[0].weight,
    }
    for li, layer in enumerate(model.layers):
        blocks[f"U^{li}"] = layer.u
        blocks[f"D_a^{li}"] = layer.d_a
        blocks[f"D_s^{li}"] = layer.d_s
    for name, parameter in blocks.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.norm()) > 0.0, name


# ---------------------------------------------------------------------------
# parameter budget
# ---------------------------------------------------------------------------


def test_parameter_budget() -> None:
    model = _model()
    total = sum(p.numel() for p in model.parameters())
    assert 45000 < total < 80000, total


# ---------------------------------------------------------------------------
# Control B — FFN replacement (preregistration §11)
# ---------------------------------------------------------------------------


def test_ffn_control_runs_and_gradients_flow() -> None:
    torch.manual_seed(0)
    model = icrate.ICrateFFNControl(seed=0)
    batch = icrate.collate([_toy()])
    pred, info = model(batch)
    assert pred.shape == (1,)
    assert info.alpha.shape == (1, icrate.M_DICT)
    loss = torch.nn.functional.l1_loss(pred, batch.y)
    loss.backward()
    blocks = {"E_V": model.e_v.weight, "E_E": model.e_e.weight, "U_G": model.u_g, "g_proj": model.g_proj.weight}
    for li, layer in enumerate(model.layers):
        blocks[f"U^{li}"] = layer.u
        blocks[f"w1^{li}"] = layer.w1.weight
        blocks[f"w2^{li}"] = layer.w2.weight
    for name, parameter in blocks.items():
        assert parameter.grad is not None, name
        assert float(parameter.grad.norm()) > 0.0, name
    total = model.parameter_breakdown()["total"]
    assert 50000 < total < 80000, total


def test_ffn_control_matches_reference_in_batch() -> None:
    torch.manual_seed(0)
    model = icrate.ICrateFFNControl(seed=0).eval()
    mols = [_toy(), icrate.molecule_from_arrays([5, 5, 6, 7], [(0, 1, 1), (1, 2, 3), (2, 3, 2)], y=2.0)]
    batch = icrate.collate(mols)
    nmax = batch.atom_idx.shape[1]
    with torch.no_grad():
        pred_b, info_b = model(batch)
        for g, mol in enumerate(mols):
            pred_r, info_r = model.forward_reference(mol)
            idx = list(range(mol.n)) + [nmax + e for e in range(mol.m)]
            assert float((pred_b[g] - pred_r).abs().max()) < 1e-5
            assert float((info_b.z_layers[-1][g, idx] - info_r.z_layers[-1]).abs().max()) < 1e-5
