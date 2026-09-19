"""Data-free correctness tests for the TIGD-v0 graph-space dictionary audit.

These lock the solver invariants the audit depends on and need no ZINC data,
checkpoints or results artifacts:

* the single-graph CPU float64 numpy reference and the batched torch solver
  agree on codes, energies and transports;
* log-domain Sinkhorn transport marginals are correct;
* the alternating inner solver is (essentially) non-increasing in ``E_G``;
* ``alpha_G`` is invariant under real atom/bond relabelling;
* ``alpha_G`` and ``E_G`` are invariant under a global latent-slot permutation
  applied to *all* dictionary atoms at once;
* two equal-size graphs with identical atom/bond feature multisets but
  different incidence produce different structure energy / transport / code.
"""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code.run_aiom_representation_audit import Mol, permute_mol
from tracks.ksvd.code.run_tigd_graph_dictionary_audit import (
    TIGDConfig,
    atoms_np_from_dict,
    collate,
    infer_batch,
    infer_reference_np,
    sinkhorn_np,
    mixture_np,
    components_np,
)

C_V = 21
C_E = 3
NODE_INDEX = {c: c for c in range(C_V)}
BOND_INDEX = {1: 0, 2: 1, 3: 2}


def _mol(n, edges, node_types, bond_types):
    edges = [(min(a, b), max(a, b)) for a, b in edges]
    return Mol(n, edges, np.asarray(bond_types, dtype=np.int64),
               np.asarray(node_types, dtype=np.int64))


def _cfg(**kw):
    base = dict(K=5, qV=6, qE=6, T_alt=6, sinkhorn_iters=200)
    base.update(kw)
    return TIGDConfig(**base)


def _random_atoms(rng, cfg):
    theta_V = rng.normal(size=(cfg.K, cfg.qV, C_V))
    theta_E = rng.normal(size=(cfg.K, cfg.qE, C_E))
    xi = 0.5 * rng.normal(size=(cfg.K, cfg.qV, cfg.qE))
    return atoms_np_from_dict({"theta_V": theta_V, "theta_E": theta_E, "xi": xi})


def _cycle_mol():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)]
    nt = [0, 1, 0, 1, 0, 0, 1]
    bt = [1 + (i % 2) for i in range(len(edges))]
    return _mol(7, edges, nt, bt)


# ---------------------------------------------------------------------------
# A. Sinkhorn marginal correctness
# ---------------------------------------------------------------------------
def test_sinkhorn_marginals():
    rng = np.random.default_rng(0)
    n, m = 6, 5
    C = rng.random((n, m)) * 0.5
    r = np.full(n, 1.0 / n)
    c = np.full(m, 1.0 / m)
    P = sinkhorn_np(C, r, c, eps=0.05, iters=200)
    assert np.all(P >= 0)
    assert np.abs(P.sum(1) - r).max() < 1e-8
    assert np.abs(P.sum(0) - c).max() < 1e-8


def test_batched_sinkhorn_matches_and_marginals():
    rng = np.random.default_rng(1)
    G, N, M = 3, 6, 5
    C = torch.as_tensor(rng.random((G, N, M)) * 0.5, dtype=torch.float64)
    av = torch.ones(G, N, dtype=torch.float64)
    av[2, 5] = 0.0  # one padded row
    n = av.sum(1)
    r = av / n[:, None]
    c = torch.full((G, M), 1.0 / M, dtype=torch.float64)
    from tracks.ksvd.code.run_tigd_graph_dictionary_audit import sinkhorn_batch

    P, err = sinkhorn_batch(C, r, c, eps=0.05, iters=200)
    assert err < 1e-8
    assert torch.all(P[2, 5] == 0)
    # compare valid block to the single-graph numpy reference
    P0 = sinkhorn_np(C[0].numpy(), r[0].numpy(), c[0].numpy(), 0.05, 200)
    assert np.abs(P[0].numpy() - P0).max() < 1e-8


# ---------------------------------------------------------------------------
# B. Energy descent of the alternating inner solver
# ---------------------------------------------------------------------------
def test_reference_energy_descent():
    rng = np.random.default_rng(2)
    cfg = _cfg()
    atoms = _random_atoms(rng, cfg)
    mol = _cycle_mol()
    A_V, A_E, R = atoms["A_V"], atoms["A_E"], atoms["R"]
    B, X_V, X_E = None, None, None
    from tracks.ksvd.code.run_tigd_graph_dictionary_audit import mol_arrays

    B, X_V, X_E = mol_arrays(mol, NODE_INDEX, BOND_INDEX)
    n, m = mol.n, mol.m
    alpha = np.full(cfg.K, 1.0 / cfg.K)
    P_V = np.full((n, cfg.qV), 1.0 / (cfg.qV * n))
    P_E = np.full((m, cfg.qE), 1.0 / (cfg.qE * m))
    r_V = np.full(n, 1.0 / n)
    c_V = np.full(cfg.qV, 1.0 / cfg.qV)
    r_E = np.full(m, 1.0 / m)
    c_E = np.full(cfg.qE, 1.0 / cfg.qE)
    totals = []
    for _ in range(cfg.T_alt):
        AV, AE, RB = mixture_np(A_V, A_E, R, alpha)
        Usq = (1.0 - RB) ** 2
        Wsq = RB ** 2
        cVc = np.sum((X_V[:, None, :] - AV[None, :, :]) ** 2, axis=2)
        from tracks.ksvd.code.run_tigd_graph_dictionary_audit import sinkhorn_np as sk
        C_V = cVc + (B @ (P_E @ Usq.T) + (1 - B) @ (P_E @ Wsq.T))
        P_V = sk(C_V, r_V, c_V, cfg.eps, cfg.sinkhorn_iters)
        cEc = np.sum((X_E[:, None, :] - AE[None, :, :]) ** 2, axis=2)
        C_E = cEc + (B.T @ (P_V @ Usq) + (1 - B).T @ (P_V @ Wsq))
        P_E = sk(C_E, r_E, c_E, cfg.eps, cfg.sinkhorn_iters)
        for _step in range(cfg.alpha_steps):
            _, _, _, E0, grad = components_np(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E)
            from tracks.ksvd.code.run_tigd_graph_dictionary_audit import proj_simplex_np
            eta, accepted = 1.0, False
            for _bt in range(cfg.alpha_backtrack):
                u = proj_simplex_np(alpha - eta * grad)
                _, _, _, En, _ = components_np(u, A_V, A_E, R, P_V, P_E, B, X_V, X_E)
                if En <= E0 + cfg.alpha_tol:
                    alpha, accepted = u, True
                    break
                eta *= 0.5
            assert accepted
        totals.append(components_np(alpha, A_V, A_E, R, P_V, P_E, B, X_V, X_E)[3])
    # alpha steps are guaranteed non-increasing; the transport steps are
    # entropic-OT (approximate LP), allow a small tolerance for those.
    totals = np.asarray(totals)
    assert np.all(np.diff(totals) <= 5e-2 + 1e-9), totals


# ---------------------------------------------------------------------------
# C. Reference <-> batched agreement + permutation invariance
# ---------------------------------------------------------------------------
def test_reference_matches_batched_and_permutation_invariance():
    rng = np.random.default_rng(3)
    cfg = _cfg()
    atoms = _random_atoms(rng, cfg)
    mol = _cycle_mol()
    ref = infer_reference_np(mol, atoms, NODE_INDEX, BOND_INDEX, cfg)
    A_V = torch.as_tensor(atoms["A_V"], dtype=torch.float64)
    A_E = torch.as_tensor(atoms["A_E"], dtype=torch.float64)
    R = torch.as_tensor(atoms["R"], dtype=torch.float64)
    batch = collate([mol], NODE_INDEX, BOND_INDEX, dtype=torch.float64)
    out = infer_batch(batch["B"], batch["X_V"], batch["X_E"], batch["n"], batch["m"],
                      batch["atom_valid"], batch["bond_valid"], A_V, A_E, R, cfg)
    assert np.abs(out["alpha"][0].numpy() - ref["alpha"]).max() < 1e-6
    assert abs(float(out["total"][0]) - ref["total"]) < 1e-6

    pa = rng.permutation(mol.n)
    pb = rng.permutation(mol.m)
    pmol = permute_mol(mol, pa, pb)
    ref2 = infer_reference_np(pmol, atoms, NODE_INDEX, BOND_INDEX, cfg)
    assert np.abs(ref2["alpha"] - ref["alpha"]).max() < 1e-6
    assert abs(ref2["total"] - ref["total"]) < 1e-6


# ---------------------------------------------------------------------------
# D. Global latent-slot permutation invariance
# ---------------------------------------------------------------------------
def test_global_slot_permutation_invariance():
    rng = np.random.default_rng(4)
    cfg = _cfg()
    atoms = _random_atoms(rng, cfg)
    mol = _cycle_mol()
    ref = infer_reference_np(mol, atoms, NODE_INDEX, BOND_INDEX, cfg)
    pv = rng.permutation(cfg.qV)
    pe = rng.permutation(cfg.qE)
    A_V2 = atoms["A_V"][:, pv, :]
    A_E2 = atoms["A_E"][:, pe, :]
    R2 = atoms["R"][:, pv][:, :, pe]
    ref2 = infer_reference_np(mol, {"A_V": A_V2, "A_E": A_E2, "R": R2},
                              NODE_INDEX, BOND_INDEX, cfg)
    assert np.abs(ref2["alpha"] - ref["alpha"]).max() < 1e-6
    assert abs(ref2["total"] - ref["total"]) < 1e-6


# ---------------------------------------------------------------------------
# E. Topology sensitivity
# ---------------------------------------------------------------------------
def test_topology_sensitivity():
    rng = np.random.default_rng(5)
    cfg = _cfg()
    atoms = _random_atoms(rng, cfg)
    nt = [0, 0, 1, 2]
    bt = [1, 2, 3]
    g1 = _mol(4, [(0, 1), (1, 2), (2, 3)], nt, bt)      # a path
    g2 = _mol(4, [(0, 1), (1, 2), (0, 3)], nt, bt)      # a different wiring
    r1 = infer_reference_np(g1, atoms, NODE_INDEX, BOND_INDEX, cfg)
    r2 = infer_reference_np(g2, atoms, NODE_INDEX, BOND_INDEX, cfg)
    # same size, same feature multisets
    assert g1.n == g2.n and g1.m == g2.m
    assert sorted(g1.node_types.tolist()) == sorted(g2.node_types.tolist())
    assert sorted(g1.bond_types.tolist()) == sorted(g2.bond_types.tolist())
    # incidence differs -> structure energy / transport / code must not be identical
    changed = (abs(r1["L_B"] - r2["L_B"]) > 1e-6
               or np.abs(r1["alpha"] - r2["alpha"]).max() > 1e-6
               or np.abs(r1["P_V"] - r2["P_V"]).max() > 1e-6)
    assert changed


# ---------------------------------------------------------------------------
# F. Dictionary init honours the frozen simplex / endpoint constraints
# ---------------------------------------------------------------------------
def test_dictionary_shape_and_incidence_constraints():
    rng = np.random.default_rng(6)
    cfg = _cfg()
    theta_V = rng.normal(size=(cfg.K, cfg.qV, C_V))
    theta_E = rng.normal(size=(cfg.K, cfg.qE, C_E))
    xi = 0.5 * rng.normal(size=(cfg.K, cfg.qV, cfg.qE))
    atoms = atoms_np_from_dict({"theta_V": theta_V, "theta_E": theta_E, "xi": xi})
    assert atoms["A_V"].shape == (cfg.K, cfg.qV, C_V)
    assert atoms["A_E"].shape == (cfg.K, cfg.qE, C_E)
    assert np.abs(atoms["A_V"].sum(-1) - 1.0).max() < 1e-9
    assert np.abs(atoms["A_E"].sum(-1) - 1.0).max() < 1e-9
    assert np.abs(atoms["R"].sum(1) - 2.0).max() < 1e-9
    assert (atoms["R"] >= 0).all()
