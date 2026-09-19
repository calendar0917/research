"""Data-free tests for the DOI representation audit.

These lock the correctness invariants the audit depends on and need no ZINC
data, checkpoints or results artifacts:

* the incidence-pair enumeration uses the real endpoints (2 per bond);
* zero-variance response coordinates are scaled safely (no NaN/Inf);
* the median heuristic reports a degenerate population and a safe fallback;
* RFF mean embeddings approximate the exact Gaussian kernel mean inner product;
* ``Phi_DOI`` (and its object-only / incidence-only views) is exactly
  permutation invariant under the real pipeline;
* the RFF-dimension rule selects the smallest candidate meeting the thresholds.
"""

from __future__ import annotations

import numpy as np

from tracks.ksvd.code.run_aiom_representation_audit import Mol, permute_mol
from tracks.ksvd.code.run_doi_representation_audit import (
    D_MAX_RFF,
    assemble,
    build_population,
    compute_embeddings,
    extract_responses,
    fit_rms_scales,
    gaussian_kernel_mean,
    median_heuristic,
    sample_rff,
    select_dimension,
)


def _codebook(mol):
    atoms = sorted({int(c) for c in mol.node_types.tolist()})
    bonds = sorted({int(c) for c in mol.bond_types.tolist()})
    return atoms, bonds, {c: i for i, c in enumerate(atoms)}, {c: i for i, c in enumerate(bonds)}


def _cycle_mol():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)]
    nt = [0, 1, 0, 1, 0, 0, 1]
    bt = [1 + (i % 2) for i in range(len(edges))]
    return Mol(7, [(min(a, b), max(a, b)) for a, b in edges],
               np.asarray(bt, dtype=np.int64), np.asarray(nt, dtype=np.int64))


def _synthetic_resp(rng, n_graphs, n_atoms=5, m=3, dmax=12):
    out = []
    for _ in range(n_graphs):
        RV = rng.normal(size=(n_atoms, dmax))
        RE = rng.normal(size=(m, dmax))
        pv = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        pe = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        out.append((RV, RE, pv, pe))
    return out


def test_incidence_pairs_use_real_endpoints():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    C = len(atoms) + len(bonds)
    (R_V, R_E, pv, pe), = extract_responses([mol], len(atoms), len(bonds), ni, bi)
    assert R_V.shape == (mol.n, (4 + 1) * C)
    assert R_E.shape == (mol.m, (4 + 1) * C)
    assert pv.shape == (2 * mol.m,) and pe.shape == (2 * mol.m,)
    for k in range(2 * mol.m):
        a, b = mol.bonds[int(pe[k])]
        assert int(pv[k]) in (a, b)
    # each bond appears exactly twice
    assert sorted(pe.tolist()) == sorted([e for e in range(mol.m) for _ in range(2)])


def test_moment_zero_is_one_hot_atom_type():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    C = len(atoms) + len(bonds)
    (R_V, R_E, _, _), = extract_responses([mol], len(atoms), len(bonds), ni, bi)
    for v in range(mol.n):
        row = R_V[v, :C]
        assert row.sum() == 1.0
        assert row[ni[int(mol.node_types[v])]] == 1.0


def test_zero_variance_scale_is_safe():
    rng = np.random.default_rng(0)
    resp = _synthetic_resp(rng, 4, dmax=24)
    # force one coordinate to be constant zero -> RMS 0
    resp = [(RV.copy(), RE.copy(), pv, pe) for RV, RE, pv, pe in resp]
    for RV, RE, _, _ in resp:
        RV[:, 3] = 0.0
        RE[:, 3] = 0.0
    s_V, _, _ = fit_rms_scales(resp, T=4, C=4, which="V")
    s_E, _, _ = fit_rms_scales(resp, T=4, C=4, which="E")
    assert s_V[3] == 0.0 and s_E[3] == 0.0
    pop = build_population(resp, 4, 4, s_V, s_E, "V")
    assert np.all(np.isfinite(pop.objs))
    popI = build_population(resp, 4, 4, s_V, s_E, "I")
    assert np.all(np.isfinite(popI.objs))


def test_population_offsets_and_counts():
    rng = np.random.default_rng(1)
    resp = _synthetic_resp(rng, 5)
    s_V = np.ones(4)
    s_E = np.ones(4)
    pop = build_population(resp, T=0, C=4, s_V=s_V, s_E=s_E, kind="V")
    assert pop.counts.tolist() == [5, 5, 5, 5, 5]
    assert pop.offsets.tolist() == [0, 5, 10, 15, 20]
    popI = build_population(resp, T=0, C=4, s_V=s_V, s_E=s_E, kind="I")
    assert popI.counts.tolist() == [6, 6, 6, 6, 6]
    assert popI.objs.shape[1] == 8  # 2 * (T+1)*C


def test_median_heuristic_degenerate_and_regular():
    objs = np.zeros((10, 5))
    st = median_heuristic(objs, np.random.default_rng(0), max_pairs=200)
    assert st["degenerate"] is True and st["median"] == 0.0
    assert st["sigma"] > 0
    rng = np.random.default_rng(0)
    regular = rng.normal(size=(200, 5))
    st2 = median_heuristic(regular, np.random.default_rng(0), max_pairs=5000)
    assert st2["degenerate"] is False and st2["sigma"] > 0


def test_rff_mean_embedding_approximates_kernel_mean():
    rng = np.random.default_rng(3)
    resp = _synthetic_resp(rng, 2, n_atoms=6, m=3, dmax=8)
    s_V = np.ones(8)
    s_E = np.ones(8)
    sigma = 2.0
    d = 4  # (T+1)*C = (0+1)*4
    rffs = {
        "V": {"omega": sample_rff(d, sigma, D_MAX_RFF, 1)[0],
              "b": sample_rff(d, sigma, D_MAX_RFF, 1)[1]},
        "E": {"omega": sample_rff(d, sigma, D_MAX_RFF, 2)[0],
              "b": sample_rff(d, sigma, D_MAX_RFF, 2)[1]},
        "I": {"omega": sample_rff(2 * d, sigma, D_MAX_RFF, 3)[0],
              "b": sample_rff(2 * d, sigma, D_MAX_RFF, 3)[1]},
    }
    emb = compute_embeddings(resp, T=0, C=4, s_V=s_V, s_E=s_E, rffs=rffs)
    popV = build_population(resp, 0, 4, s_V, s_E, "V")
    Kg = popV.objs[:popV.counts[0]]
    Kh = popV.objs[popV.offsets[1]:popV.offsets[1] + popV.counts[1]]
    exact = gaussian_kernel_mean(Kg, Kh, sigma)
    approx = float(emb["muV"][0] @ emb["muV"][1])
    assert abs(exact - approx) < 0.05


def test_phi_is_permutation_invariant_through_pipeline():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    C = len(atoms) + len(bonds)
    resp = extract_responses([mol], len(atoms), len(bonds), ni, bi)
    s_V, _, _ = fit_rms_scales(resp, T=2, C=C, which="V")
    s_E, _, _ = fit_rms_scales(resp, T=2, C=C, which="E")
    d = 3 * C  # (T+1)*C with T=2
    rffs = {
        "V": {"omega": sample_rff(d, 1.0, D_MAX_RFF, 11)[0],
              "b": sample_rff(d, 1.0, D_MAX_RFF, 11)[1]},
        "E": {"omega": sample_rff(d, 1.0, D_MAX_RFF, 12)[0],
              "b": sample_rff(d, 1.0, D_MAX_RFF, 12)[1]},
        "I": {"omega": sample_rff(2 * d, 1.0, D_MAX_RFF, 13)[0],
              "b": sample_rff(2 * d, 1.0, D_MAX_RFF, 13)[1]},
    }
    emb = compute_embeddings(resp, T=2, C=C, s_V=s_V, s_E=s_E, rffs=rffs)
    base = assemble(emb["muV"], emb["muE"], emb["muI"], emb["sumV"], emb["sumE"],
                    emb["sumI"], emb["n"], emb["m"], D=64)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pa = rng.permutation(mol.n)
        pb = rng.permutation(mol.m)
        pm = permute_mol(mol, pa, pb)
        resp2 = extract_responses([pm], len(atoms), len(bonds), ni, bi)
        emb2 = compute_embeddings(resp2, T=2, C=C, s_V=s_V, s_E=s_E, rffs=rffs)
        other = assemble(emb2["muV"], emb2["muE"], emb2["muI"], emb2["sumV"],
                         emb2["sumE"], emb2["sumI"], emb2["n"], emb2["m"], D=64)
        for key in ("full", "obj", "inc"):
            assert np.max(np.abs(other[key] - base[key])) < 1e-10


def test_assemble_shapes_and_mass():
    rng = np.random.default_rng(4)
    G, D = 7, 32
    muV = rng.normal(size=(G, D_MAX_RFF))
    muE = rng.normal(size=(G, D_MAX_RFF))
    muI = rng.normal(size=(G, D_MAX_RFF))
    sV, sE, sI = (rng.normal(size=(G, D_MAX_RFF)) for _ in range(3))
    n = np.arange(1, G + 1)
    m = np.arange(2, G + 2)
    out = assemble(muV, muE, muI, sV, sE, sI, n, m, D)
    assert out["full"].shape == (G, 3 * D + 2)
    assert out["obj"].shape == (G, 2 * D + 2)
    assert out["inc"].shape == (G, D + 2)
    assert out["sum"].shape == (G, 3 * D)
    assert np.allclose(out["full"][:, -2], np.log1p(n))
    assert np.allclose(out["full"][:, -1], np.log1p(m))


def test_select_dimension_rule():
    G = 20
    pairs = np.array([[i, (i + 1) % G] for i in range(G)])
    cal_emb = {f"mu{k}": np.zeros((G, D_MAX_RFF)) for k in ("V", "E", "I")}
    exact = {k: np.zeros(len(pairs)) for k in ("V", "E", "I")}
    chosen, per_kernel, ok = select_dimension(cal_emb, np.arange(G), pairs, exact,
                                              [512, 1024, 2048])
    assert chosen == 512 and ok is True
    cal_emb_bad = {f"mu{k}": np.ones((G, D_MAX_RFF)) for k in ("V", "E", "I")}
    chosen2, _, ok2 = select_dimension(cal_emb_bad, np.arange(G), pairs, exact,
                                       [512, 1024, 2048])
    assert chosen2 == 2048 and ok2 is False
