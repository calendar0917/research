"""Data-free correctness tests for the Nonlinearity Placement Audit.

These lock the invariants the audit depends on and need no ZINC data,
checkpoints or results artifacts:

* scattering responses are permutation-equivariant and the KME readout is
  permutation-invariant;
* the tiny learned incidence lift is permutation-equivariant and its fixed-KME
  readout is permutation-invariant;
* incidence information (which endpoint carries which attribute) actually
  changes both responses and the final ``Phi``;
* every response / filter / KME / probe value stays finite.
"""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code.run_aiom_representation_audit import Mol, permute_mol
from tracks.ksvd.code.run_nonlinearity_placement_audit import (
    D_MAX_RFF,
    DistributionContext,
    H_DIM,
    TinyIncidenceLift,
    build_population,
    extract_scattering,
    make_lift_data,
    batch_tensors,
    phi_from_objects,
    sample_rff,
    scattering_blocks,
)


def _codebook(mol):
    atoms = sorted({int(c) for c in mol.node_types.tolist()})
    bonds = sorted({int(c) for c in mol.bond_types.tolist()})
    return atoms, bonds, {c: i for i, c in enumerate(atoms)}, {c: i for i, c in enumerate(bonds)}


def _mol(n, edges, node_types, bond_types):
    edges = [(min(a, b), max(a, b)) for a, b in edges]
    return Mol(n, edges, np.asarray(bond_types, dtype=np.int64),
               np.asarray(node_types, dtype=np.int64))


def _cycle_mol():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)]
    nt = [0, 1, 0, 1, 0, 0, 1]
    bt = [1 + (i % 2) for i in range(len(edges))]
    return _mol(7, edges, nt, bt)


def _small_context(d: int, seed: int = 0) -> DistributionContext:
    s_V = np.ones(d)
    s_E = np.ones(d)
    sigmas = {"V": 1.0, "E": 1.0, "I": 1.0}
    dims = {"V": d, "E": d, "I": 2 * d}
    rffs = {}
    for i, k in enumerate(("V", "E", "I")):
        omega, b = sample_rff(dims[k], sigmas[k], D_MAX_RFF, seed + i)
        rffs[k] = {"omega": omega, "b": b, "sigma": sigmas[k], "d": dims[k], "seed": seed + i}
    return DistributionContext(d=d, s_V=s_V, s_E=s_E, sigmas=sigmas, rffs=rffs, D=64)


def _phi_from_context(resp, ctx):
    pops = {k: build_population(resp, ctx.d, ctx.s_V, ctx.s_E, k) for k in ("V", "E", "I")}
    from tracks.ksvd.code.run_doi_representation_audit import (
        assemble,
        kernel_mean_embeddings,
    )
    mus, sums = {}, {}
    for k in ("V", "E", "I"):
        mu, s = kernel_mean_embeddings(pops[k], ctx.rffs[k]["omega"], ctx.rffs[k]["b"],
                                       D_MAX_RFF)
        mus[k], sums[k] = mu, s
    n = np.array([resp[0][0].shape[0]])
    m = np.array([resp[0][1].shape[0]])
    return assemble(mus["V"], mus["E"], mus["I"], sums["V"], sums["E"], sums["I"],
                    n, m, ctx.D)["full"]


def _model(C_V, C_E, seed=0):
    torch.manual_seed(seed)
    return TinyIncidenceLift(C_V, C_E, H_DIM, rounds=2).double()


def _torch_phi(model, mols, C_V, C_E, ni, bi, d):
    dtype = next(model.parameters()).dtype
    data = make_lift_data(mols, C_V, C_E, ni, bi)
    b = batch_tensors(data, list(range(len(mols))), "cpu")
    b = {k: (v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
         for k, v in b.items()}
    rffs = {}
    for i, k in enumerate(("V", "E", "I")):
        dim = H_DIM if k != "I" else 2 * H_DIM
        omega, bb = sample_rff(dim, 1.0, D_MAX_RFF, 100 + i)
        rffs[k] = {"omega": torch.as_tensor(omega, dtype=dtype),
                   "b": torch.as_tensor(bb, dtype=dtype)}
    with torch.no_grad():
        h, e, _ = model(b["node_oh"], b["bond_oh"], b["bond_u"], b["bond_v"],
                        b["inc_atom"], b["inc_bond"])
        phi = phi_from_objects(b, h, e, rffs, d)
    return h.numpy(), e.numpy(), phi.numpy()


# ---------------------------------------------------------------------------
# A. Scattering permutation equivariance
# ---------------------------------------------------------------------------
def test_scattering_permutation_equivariance():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    C_V, C_E = len(atoms), len(bonds)
    B = scattering_blocks(mol, C_V, C_E, ni, bi)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pa = rng.permutation(mol.n)
        pb = rng.permutation(mol.m)
        pm = permute_mol(mol, pa, pb)
        Bp = scattering_blocks(pm, C_V, C_E, ni, bi)
        assert Bp.shape == B.shape
        for v in range(mol.n):
            assert np.max(np.abs(Bp[pa[v]] - B[v])) < 1e-10
        for i in range(mol.m):
            # permute_mol sets bonds_new[i] = bonds_old[pb[i]]
            assert np.max(np.abs(Bp[mol.n + i] - B[mol.n + pb[i]])) < 1e-10


# ---------------------------------------------------------------------------
# B. Scattering final invariance after KME
# ---------------------------------------------------------------------------
def test_scattering_final_invariance():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    C_V, C_E = len(atoms), len(bonds)
    d = scattering_blocks(mol, C_V, C_E, ni, bi).shape[1]
    ctx = _small_context(d, seed=3)
    base = _phi_from_context(extract_scattering([mol], C_V, C_E, ni, bi), ctx)
    rng = np.random.default_rng(1)
    for _ in range(5):
        pm = permute_mol(mol, rng.permutation(mol.n), rng.permutation(mol.m))
        other = _phi_from_context(extract_scattering([pm], C_V, C_E, ni, bi), ctx)
        assert np.max(np.abs(other - base)) < 1e-10


# ---------------------------------------------------------------------------
# C. Learned lift permutation equivariance
# ---------------------------------------------------------------------------
def test_learned_lift_permutation_equivariance():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    C_V, C_E = len(atoms), len(bonds)
    model = _model(C_V, C_E, seed=0)
    h, e, _ = _torch_phi(model, [mol], C_V, C_E, ni, bi, 32)
    rng = np.random.default_rng(2)
    for _ in range(5):
        pa = rng.permutation(mol.n)
        pb = rng.permutation(mol.m)
        pm = permute_mol(mol, pa, pb)
        hp, ep, _ = _torch_phi(model, [pm], C_V, C_E, ni, bi, 32)
        for v in range(mol.n):
            assert np.max(np.abs(hp[pa[v]] - h[v])) < 1e-10
        for i in range(mol.m):
            # permute_mol sets bonds_new[i] = bonds_old[pb[i]]
            assert np.max(np.abs(ep[i] - e[pb[i]])) < 1e-10


# ---------------------------------------------------------------------------
# D. Learned final invariance
# ---------------------------------------------------------------------------
def test_learned_final_invariance():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    C_V, C_E = len(atoms), len(bonds)
    model = _model(C_V, C_E, seed=0)
    _, _, base = _torch_phi(model, [mol], C_V, C_E, ni, bi, 32)
    rng = np.random.default_rng(3)
    for _ in range(5):
        pm = permute_mol(mol, rng.permutation(mol.n), rng.permutation(mol.m))
        _, _, other = _torch_phi(model, [pm], C_V, C_E, ni, bi, 32)
        assert np.max(np.abs(other - base)) < 1e-10


# ---------------------------------------------------------------------------
# E. Incidence sensitivity
# ---------------------------------------------------------------------------
def _incidence_pair_graphs():
    # same atom multiset {0,0,1,1}, same bond multiset {0,1}, same counts,
    # different endpoint assignment.
    g1 = _mol(4, [(0, 1), (2, 3)], [0, 0, 1, 1], [0, 1])
    g2 = _mol(4, [(0, 2), (1, 3)], [0, 0, 1, 1], [0, 1])
    return g1, g2


def test_incidence_sensitivity_scattering_and_phi():
    g1, g2 = _incidence_pair_graphs()
    ni = {0: 0, 1: 1}
    bi = {0: 0, 1: 1}
    C_V, C_E = 2, 2
    B1 = scattering_blocks(g1, C_V, C_E, ni, bi)
    B2 = scattering_blocks(g2, C_V, C_E, ni, bi)
    assert B1.shape == B2.shape
    assert not np.allclose(B1, B2)
    d = B1.shape[1]
    ctx = _small_context(d, seed=5)
    f1 = _phi_from_context(extract_scattering([g1], C_V, C_E, ni, bi), ctx)
    f2 = _phi_from_context(extract_scattering([g2], C_V, C_E, ni, bi), ctx)
    assert not np.allclose(f1, f2)


def test_incidence_sensitivity_learned():
    g1, g2 = _incidence_pair_graphs()
    ni = {0: 0, 1: 1}
    bi = {0: 0, 1: 1}
    C_V, C_E = 2, 2
    model = _model(C_V, C_E, seed=0)
    h1, e1, phi1 = _torch_phi(model, [g1], C_V, C_E, ni, bi, 32)
    h2, e2, phi2 = _torch_phi(model, [g2], C_V, C_E, ni, bi, 32)
    assert not np.allclose(h1, h2) or not np.allclose(e1, e2)
    assert not np.allclose(phi1, phi2)


# ---------------------------------------------------------------------------
# F. No NaN / finite
# ---------------------------------------------------------------------------
def test_all_finite():
    sizes = [(2, 1), (3, 2), (5, 4), (7, 6)]
    for n, m in sizes:
        edges = [(i, i + 1) for i in range(m)]
        edges += [(0, 2)] if m >= 3 else []
        nt = [i % 2 for i in range(n)]
        bt = [i % 2 for i in range(len(edges))]
        mol = _mol(n, edges, nt, bt)
        atoms, bonds, ni, bi = _codebook(mol)
        C_V, C_E = len(atoms), len(bonds)
        B = scattering_blocks(mol, C_V, C_E, ni, bi)
        assert np.all(np.isfinite(B))
        resp = extract_scattering([mol], C_V, C_E, ni, bi)
        ctx = _small_context(B.shape[1], seed=7)
        phi = _phi_from_context(resp, ctx)
        assert np.all(np.isfinite(phi))
        model = _model(C_V, C_E, seed=0)
        h, e, phit = _torch_phi(model, [mol], C_V, C_E, ni, bi, 32)
        assert np.all(np.isfinite(h)) and np.all(np.isfinite(e))
        assert np.all(np.isfinite(phit))


def test_structural_param_budget():
    model = _model(21, 3, seed=0)
    n = model.structural_params()
    assert 2000 <= n <= 4000, n
    # input tables + two rounds of (4 x 16x16 + 2 x 16)
    expected = 21 * 16 + 3 * 16 + 2 * (4 * 16 * 16 + 2 * 16)
    assert n == expected


def test_zero_message_control_is_constructible():
    model = _model(21, 3, seed=0)
    with torch.no_grad():
        for r in range(model.rounds):
            model.W_VE[r].zero_()
            model.W_EV[r].zero_()
    assert float(model.W_VE[0].norm()) == 0.0
    assert float(model.W_EV[1].norm()) == 0.0
