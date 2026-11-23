"""Pre-training mathematical and engineering checks for RICE-v0.

Groups A-F of ``PROTOCOL.md`` section 7.  Every check writes a machine-readable
record to ``analysis/checks.json`` and the process exits non-zero if any check
fails.  No formal training may start before all checks pass.

A  Mobius inversion identity against an independent brute-force subset sum
B  connectivity: disconnected root-anchored supports and the J = 0 limit
C  real structure sensitivity (topology, attribute/topology binding, triangle
   closure, 4-clique bond slots)
D  invariances (atom renumbering, bond direction, bond storage order, batch
   composition / molecule order)
E  gradients, existence of order-3 / order-4 interactions, 30-step decrease
F  compute cost, word-count distribution, vectorisation evidence
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import rice_common as C
import rice_data as D
import rice_energy as E
import rice_model as M
import rice_structure as S
import rice_tensor as T

ATOM_DIMS = [119, 5, 12, 12, 10, 6, 6, 2, 2]
BOND_DIMS = [5, 6, 2]
RECORDS: list[dict[str, Any]] = []


def record(check_id: str, group: str, ok: bool, **detail: Any) -> None:
    RECORDS.append({"id": check_id, "group": group, "status": "PASS" if ok else "FAIL",
                    **detail})
    print(f"[{ 'PASS' if ok else 'FAIL' }] {check_id} {group} {detail}", flush=True)


# ---------------------------------------------------------------------------
# synthetic molecules
# ---------------------------------------------------------------------------

def make_mol(n_atoms: int, edges: Sequence[tuple[int, int]],
             atom_fields: Sequence[Sequence[int]] | None = None,
             bond_fields: Sequence[Sequence[int]] | None = None, row: int = 0) -> dict[str, Any]:
    if atom_fields is None:
        atom_fields = [[6, 0, 0, 0, 0, 0, 0, 0, 0]] * n_atoms
    if bond_fields is None:
        bond_fields = [[0, 0, 0]] * len(edges)
    return {
        "row": int(row), "n_atoms": int(n_atoms), "n_bonds": len(edges),
        "atom_feat": np.asarray(atom_fields, np.int64).reshape(n_atoms, 9),
        "bond_feat": np.asarray(bond_fields, np.int64).reshape(len(edges), 3),
        "bond_u": np.asarray([e[0] for e in edges], np.int64),
        "bond_v": np.asarray([e[1] for e in edges], np.int64),
        "graph_scalars": np.zeros(8, np.float64),
    }


def _mixed_radix(values: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    out = np.zeros(values.shape[0], np.int64)
    for i, dim in enumerate(dims):
        out = out * int(dim) + values[:, i].astype(np.int64)
    return out


def field_potentials(mol: Mapping[str, Any], channels: int = 4
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic non-degenerate potentials *derived from the OGB fields*.

    Same atom fields => same unary potential, same bond fields => same pairwise
    potential, independent of atom/bond index, so "attribute bound to a different
    position" is a real change of the energy function.
    """
    a = _mixed_radix(np.asarray(mol["atom_feat"], np.int64), ATOM_DIMS) % 97
    b = _mixed_radix(np.asarray(mol["bond_feat"], np.int64), BOND_DIMS) % 89
    chan = np.arange(channels)[None, :, None]
    state = np.arange(2)[None, None, :]
    u = 0.45 * np.sin(1.0 + 0.7 * a[:, None, None] + 1.3 * chan + 2.1 * state)
    term = np.arange(3)[None, None, :]
    j = 0.55 * np.cos(0.9 + 1.1 * b[:, None, None] + 0.8 * chan + 1.7 * term)
    return u.astype(np.float64), j.astype(np.float64)


def vectorised_phi_kappa(mol: Mapping[str, Any], u: np.ndarray, j: np.ndarray,
                         dtype: torch.dtype = torch.float64) -> dict[str, Any]:
    st = S.build_molecule(mol)
    batch = T.collate([mol["row"]], {mol["row"]: st}, None, "cpu")
    n_atoms = int(mol["n_atoms"])
    ut = torch.as_tensor(u, dtype=dtype)
    jt = torch.as_tensor(j, dtype=dtype)
    channels = int(u.shape[1])
    phi = {k: E.compute_phi(ut, jt, batch[f"atoms_{k}"], batch[f"bonds_{k}"],
                            batch[f"bslot_{k}"]) for k in (2, 3, 4)}
    kappa = E.mobius(ut.reshape(n_atoms, channels * 2), phi,
                     {k: batch[f"atoms_{k}"] for k in (2, 3, 4)},
                     {k: batch[f"sub2_{k}"] for k in (3, 4)}, batch["sub3_4"])
    return {"struct": st, "batch": batch, "phi": phi, "kappa": kappa,
            "u_flat": ut.reshape(n_atoms, channels * 2)}


def support_rows(entry: Mapping[str, Any], size: int) -> list[tuple[tuple[int, ...], Any, Any]]:
    batch = entry["batch"]
    atoms = batch[f"atoms_{size}"]
    out = []
    for i in range(int(atoms.shape[0])):
        slots = tuple(int(x) for x in atoms[i] if int(x) >= 0)
        out.append((slots, entry["phi"][size][i].numpy(), entry["kappa"][size][i].numpy()))
    return out


# ---------------------------------------------------------------------------
# A -- Mobius inversion identity
# ---------------------------------------------------------------------------

def all_small_graphs(maxn: int = 4) -> list[tuple[int, list[tuple[int, int]]]]:
    graphs = []
    for n in range(1, maxn + 1):
        pairs = list(itertools.combinations(range(n), 2))
        for mask in range(1 << len(pairs)):
            graphs.append((n, [p for i, p in enumerate(pairs) if (mask >> i) & 1]))
    return graphs


def check_inversion(n_graphs: int) -> None:
    rng = np.random.default_rng(20261124)
    graphs = all_small_graphs(4)
    worst_vec = 0.0
    worst_identity = 0.0
    worst_identity_f32 = 0.0
    n_support = 0
    n_identity = 0
    for n, edges in graphs:
        if n == 1:
            continue
        mol = make_mol(n, edges)
        channels = 3
        u = rng.normal(size=(n, channels, 2))
        j = rng.normal(size=(len(edges), channels, 3))
        entry = vectorised_phi_kappa(mol, u, j)
        bond_list = [(int(a), int(b), i) for i, (a, b) in enumerate(edges)]
        entry32 = vectorised_phi_kappa(mol, u, j, dtype=torch.float32)
        for root in range(n):
            radius = _radius_atoms(n, edges, root)
            ref = E.brute_force_kappa(u, j, n, bond_list, root, radius)
            for size in (2, 3, 4):
                for slots, phi_vec, kappa_vec in support_rows(entry, size):
                    if slots[0] != root:
                        continue
                    key = frozenset(slots)
                    phi_ref = ref["phi"][key]
                    kappa_ref = ref["kappa"][key]
                    worst_vec = max(worst_vec,
                                    float(np.abs(phi_vec.reshape(channels, 2) - phi_ref).max()),
                                    float(np.abs(kappa_vec.reshape(channels, 2) - kappa_ref).max()))
                    total = np.zeros_like(phi_ref)
                    for sub_size in range(1, len(key) + 1):
                        for combo in itertools.combinations(sorted(key - {root}), sub_size - 1):
                            total = total + ref["kappa"][frozenset((root,) + combo)]
                    worst_identity = max(worst_identity, float(np.abs(phi_ref - total).max()))
                    n_identity += 1
                    n_support += 1
                arr = entry32["batch"][f"atoms_{size}"]
                kap32 = entry32["kappa"][size]
                kap64 = entry["kappa"][size]
                if int(kap32.shape[0]) == int(kap64.shape[0]):
                    worst_identity_f32 = max(
                        worst_identity_f32,
                        float((kap32.double() - kap64.double()).abs().max())
                        if int(kap32.numel()) else 0.0)
    record("A1", "A", worst_identity < 1e-5,
           description="Phi(S,v) == sum_{T subset S, v in T} kappa(T,v) on every "
                       "graph with <= 4 atoms, independent brute-force subset sum",
           graphs=int(n_graphs), supports=int(n_support), identities=int(n_identity),
           max_error=worst_identity)
    record("A2", "A", worst_vec < 1e-10,
           description="vectorised Phi/kappa == brute-force Phi/kappa (float64)",
           max_error=worst_vec)
    record("A3", "A", worst_identity_f32 < 1e-4,
           description="float32 production path stays within float32 noise of the "
                       "float64 reference (reported, informational)",
           max_error=worst_identity_f32)


def _radius_atoms(n: int, edges: Sequence[tuple[int, int]], root: int) -> list[int]:
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist = {root: 0}
    frontier = [root]
    while frontier:
        nxt = []
        for x in frontier:
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    nxt.append(y)
        frontier = nxt
    return sorted(k for k, v in dist.items() if v <= 2)


# ---------------------------------------------------------------------------
# B -- connectivity
# ---------------------------------------------------------------------------

def check_connectivity() -> None:
    worst_disconnected = 0.0
    n_disconnected = 0
    rng = np.random.default_rng(7)
    for n, edges in all_small_graphs(4):
        if n < 2:
            continue
        mol = make_mol(n, edges)
        channels = 4
        u = rng.normal(size=(n, channels, 2))
        j = rng.normal(size=(len(edges), channels, 3))
        bond_list = [(int(a), int(b), i) for i, (a, b) in enumerate(edges)]
        for root in range(n):
            radius = _radius_atoms(n, edges, root)
            ref = E.brute_force_kappa(u, j, n, bond_list, root, radius)
            for key, kappa in ref["kappa"].items():
                if len(key) < 2:
                    continue
                if not E.reference_is_connected(sorted(key), bond_list):
                    n_disconnected += 1
                    worst_disconnected = max(worst_disconnected, float(np.abs(kappa).max()))
    record("B1", "B", worst_disconnected < 1e-10,
           description="kappa(S,v) == 0 for every disconnected root-anchored S with |S|>1",
           cases=int(n_disconnected), max_abs=worst_disconnected)

    # J == 0 -> every |S| >= 2 kappa is exactly zero (all roots disagree => interaction-free)
    worst_j0 = 0.0
    n_cases = 0
    for n, edges in all_small_graphs(4):
        if n < 2:
            continue
        mol = make_mol(n, edges)
        channels = 4
        u = rng.normal(size=(n, channels, 2))
        j = np.zeros((len(edges), channels, 3))
        entry = vectorised_phi_kappa(mol, u, j)
        for size in (2, 3, 4):
            for _slots, _phi, kappa in support_rows(entry, size):
                worst_j0 = max(worst_j0, float(np.abs(kappa).max()))
                n_cases += 1
    record("B2", "B", worst_j0 < 1e-10,
           description="with every bond potential set to zero, every |S|>=2 kappa is zero",
           supports=int(n_cases), max_abs=worst_j0)


# ---------------------------------------------------------------------------
# C -- real structure sensitivity
# ---------------------------------------------------------------------------

def _single_support(mol: Mapping[str, Any], u: np.ndarray, j: np.ndarray,
                    size: int, root: int = 0) -> tuple[int, np.ndarray, np.ndarray]:
    entry = vectorised_phi_kappa(mol, u, j)
    rows = [r for r in support_rows(entry, size) if len(r[0]) == size and r[0][0] == int(root)]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one size-{size} support rooted at {root}, "
                           f"got {len(rows)}")
    return rows[0][0], rows[0][1], rows[0][2]


def check_structure_sensitivity() -> None:
    c_atom = [6, 0, 0, 0, 0, 0, 0, 0, 0]
    o_atom = [8, 0, 0, 0, 0, 0, 0, 0, 0]
    single = [1, 0, 1]

    # C1 identical atom and bond attribute multisets, different real connectivity:
    # 4-cycle vs triangle-with-a-tail (both {C,C,C,C} and {C-C single x 4}); root on
    # a vertex that sees every atom, so the whole molecule is a single 4-word
    cyc = make_mol(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    tail = make_mol(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
    u_a, j_a = field_potentials(cyc)
    u_b, j_b = field_potentials(tail)
    _, phi_a, kap_a = _single_support(cyc, u_a, j_a, 4, root=0)
    _, phi_b, kap_b = _single_support(tail, u_b, j_b, 4, root=2)
    record("C1", "C",
           float(np.abs(phi_a - phi_b).max()) > 1e-3
           and float(np.abs(kap_a - kap_b).max()) > 1e-3,
           description="same {C,C,C,C} atom multiset and same {C-C single x 4} bond "
                       "multiset, 4-cycle vs triangle+tail: Phi and kappa both separate",
           phi_gap=float(np.abs(phi_a - phi_b).max()),
           kappa_gap=float(np.abs(kap_a - kap_b).max()))

    # C2 identical atom and bond attribute multisets, different attribute/topology
    # binding: C-O-O rooted at C vs O-C-O rooted at C
    g1 = make_mol(3, [(0, 1), (1, 2)], [c_atom, o_atom, o_atom], [single, single])
    g2 = make_mol(3, [(0, 1), (1, 2)], [o_atom, c_atom, o_atom], [single, single])
    u1, j1 = field_potentials(g1)
    u2, j2 = field_potentials(g2)
    _, phi_a, kap_a = _single_support(g1, u1, j1, 3, root=0)
    _, phi_b, kap_b = _single_support(g2, u2, j2, 3, root=1)
    record("C2", "C",
           float(np.abs(phi_a - phi_b).max()) > 1e-3
           and float(np.abs(kap_a - kap_b).max()) > 1e-3,
           description="same {C,O,O} atom multiset and same {C-O single x 2} bond "
                       "multiset, attribute/topology binding swapped (root C in C-O-O "
                       "with a chain tail vs O-C-O with two leaf O): Phi and kappa both "
                       "separate",
           phi_gap=float(np.abs(phi_a - phi_b).max()),
           kappa_gap=float(np.abs(kap_a - kap_b).max()))

    # C3 triangle closure bond must be read
    tri = make_mol(3, [(0, 1), (1, 2), (0, 2)])
    chain = make_mol(3, [(0, 1), (1, 2)])
    st_tri = S.build_molecule(tri)
    n_internal = int((st_tri.bonds[3][0] >= 0).sum())
    u_tri, j_tri = field_potentials(tri)
    u_ch, j_ch = field_potentials(chain)
    _, phi_tri, kap_tri = _single_support(tri, u_tri, j_tri, 3, root=0)
    _, phi_ch, kap_ch = _single_support(chain, u_ch, j_ch, 3, root=0)
    record("C3", "C", n_internal == 3 and float(np.abs(kap_tri - kap_ch).max()) > 1e-3,
           description="3-cycle stores all three internal bonds (closing bond read) and "
                       "its kappa differs from the open 3-chain",
           internal_bonds=int(n_internal),
           kappa_gap=float(np.abs(kap_tri - kap_ch).max()))

    # C4 4-clique uses all six bond slots and stays finite
    clique = make_mol(4, list(itertools.combinations(range(4), 2)))
    st_clique = S.build_molecule(clique)
    n_internal = int((st_clique.bonds[4][0] >= 0).sum())
    u_cl, j_cl = field_potentials(clique)
    entry = vectorised_phi_kappa(clique, u_cl, j_cl)
    finite = bool(np.isfinite(entry["kappa"][4].numpy()).all())
    record("C4", "C", n_internal == 6 and finite,
           description="4-clique support uses all six internal bond slots and is finite",
           internal_bonds=int(n_internal), finite=finite)

    # C5 same atoms/bonds, path vs star topology: Phi separates them; the order-4
    # kappa of a middle-rooted 4-path is identically zero (two conditionally
    # independent branches given the root), and the implementation reproduces that
    path = make_mol(4, [(0, 1), (1, 2), (2, 3)])
    star = make_mol(4, [(0, 1), (0, 2), (0, 3)])
    u_p, j_p = field_potentials(path)
    u_s, j_s = field_potentials(star)
    _, phi_a, kap_a = _single_support(path, u_p, j_p, 4, root=1)
    _, phi_b, kap_b = _single_support(star, u_s, j_s, 4, root=0)
    record("C5", "C",
           float(np.abs(phi_a - phi_b).max()) > 1e-3
           and float(np.abs(kap_a).max()) < 1e-10
           and float(np.abs(kap_b).max()) < 1e-10,
           description="same {C,C,C,C}+{C-C x 3} multiset, 4-path (middle root) vs 4-star: "
                       "Phi separates them, and the order-4 kappa is identically zero in "
                       "both (theory: conditional independence of the root branches)",
           phi_gap=float(np.abs(phi_a - phi_b).max()),
           kappa_path=float(np.abs(kap_a).max()), kappa_star=float(np.abs(kap_b).max()))


# ---------------------------------------------------------------------------
# D -- invariances
# ---------------------------------------------------------------------------

def _struct_from_mol(mol: Mapping[str, Any]) -> S.MoleculeStruct:
    return S.build_molecule(mol)


def _kappa_by_certificate(st: S.MoleculeStruct, entry: Mapping[str, Any], size: int,
                          atom_dims: Sequence[int], bond_dims: Sequence[int]
                          ) -> tuple[np.ndarray, np.ndarray]:
    sig = S.canonical_signature(st, size, atom_dims, bond_dims)
    kappa = entry["kappa"][size].detach().numpy()
    if int(sig.shape[0]) == 0:
        return sig, kappa
    order = np.lexsort(sig.T[::-1])
    return sig[order], kappa[order]


def _predict(mol: Mapping[str, Any], model: M.RiceModel, arm: str,
             extra: Mapping[int, Mapping[str, Any]] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    row = int(mol["row"])
    structs = {row: S.build_molecule(mol)}
    if extra:
        for r, m in extra.items():
            structs[int(r)] = S.build_molecule(m)
    rows = sorted(structs)
    batch = T.collate(rows, structs, None, "cpu")
    out = model(batch, arm=arm, return_aux=True)
    idx = rows.index(row)
    return out["yhat_std"][idx:idx + 1].detach().numpy(), out


def check_invariances(bundle: D.DataBundle) -> None:
    cfg = C.load_config()
    model, _info = M.build_model(bundle.atom_dims, bundle.bond_dims, 0,
                                 D.normalisers(bundle), config=cfg)
    with torch.no_grad():
        torch.nn.init.normal_(model.atom_mlp[-2].weight, 0.0, 0.3)
        torch.nn.init.normal_(model.bond_mlp[-2].weight, 0.0, 0.3)
    model.eval()
    rng = np.random.default_rng(11)
    rows = [int(bundle.fit_rows[0]), int(bundle.fit_rows[1]), int(bundle.eval_rows[0])]
    mols = {r: D.molecule_arrays(bundle, r) for r in rows}
    base_pred: dict[int, np.ndarray] = {}
    base_kappa: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for r, mol in mols.items():
        entry = vectorised_phi_kappa(mol, *_potentials_for(model, mol), dtype=torch.float32)
        st = entry["struct"]
        base_pred[r], _ = _predict(mol, model, "RICE_CUMULANT")
        base_kappa[r] = {size: _kappa_by_certificate(st, entry, size, bundle.atom_dims,
                                                    bundle.bond_dims) for size in (2, 3, 4)}

    worst_renum = 0.0
    worst_renum_kappa = 0.0
    worst_flip = 0.0
    worst_flip_kappa = 0.0
    worst_order = 0.0
    sig_mismatch = 0
    for r, mol in mols.items():
        perm = rng.permutation(int(mol["n_atoms"]))
        renum = _relabel(mol, perm)
        entry = vectorised_phi_kappa(renum, *_potentials_for(model, renum), dtype=torch.float32)
        pred, _ = _predict(renum, model, "RICE_CUMULANT")
        worst_renum = max(worst_renum, float(np.abs(pred - base_pred[r]).max()))
        for size in (2, 3, 4):
            sig, kappa = _kappa_by_certificate(entry["struct"], entry, size,
                                               bundle.atom_dims, bundle.bond_dims)
            sig0, kappa0 = base_kappa[r][size]
            if sig.shape != sig0.shape or not np.array_equal(sig, sig0):
                sig_mismatch += 1
            elif kappa.shape == kappa0.shape:
                worst_renum_kappa = max(worst_renum_kappa,
                                        float(np.abs(kappa - kappa0).max()) if kappa.size else 0.0)

        flipped = dict(mol)
        flipped["bond_u"] = np.asarray(mol["bond_v"], np.int64)
        flipped["bond_v"] = np.asarray(mol["bond_u"], np.int64)
        entry = vectorised_phi_kappa(flipped, *_potentials_for(model, flipped), dtype=torch.float32)
        pred, _ = _predict(flipped, model, "RICE_CUMULANT")
        worst_flip = max(worst_flip, float(np.abs(pred - base_pred[r]).max()))
        for size in (2, 3, 4):
            sig, kappa = _kappa_by_certificate(entry["struct"], entry, size,
                                               bundle.atom_dims, bundle.bond_dims)
            sig0, kappa0 = base_kappa[r][size]
            if sig.shape == sig0.shape and kappa.shape == kappa0.shape and kappa.size:
                worst_flip_kappa = max(worst_flip_kappa, float(np.abs(kappa - kappa0).max()))

        order = rng.permutation(int(mol["n_bonds"]))
        shuffled = dict(mol)
        shuffled["bond_u"] = np.asarray(mol["bond_u"], np.int64)[order]
        shuffled["bond_v"] = np.asarray(mol["bond_v"], np.int64)[order]
        shuffled["bond_feat"] = np.asarray(mol["bond_feat"], np.int64)[order]
        pred, _ = _predict(shuffled, model, "RICE_CUMULANT")
        worst_order = max(worst_order, float(np.abs(pred - base_pred[r]).max()))

    record("D1", "D", worst_renum < 1e-5 and worst_renum_kappa < 1e-4 and sig_mismatch == 0,
           description="atom renumbering: identical prediction, identical canonical word "
                       "certificates and identical per-certificate kappa",
           pred_max_diff=worst_renum, kappa_max_diff=worst_renum_kappa,
           certificate_mismatches=int(sig_mismatch))
    record("D2", "D", worst_flip < 1e-6 and worst_flip_kappa < 1e-4,
           description="bond direction swap: identical prediction and kappa",
           pred_max_diff=worst_flip, kappa_max_diff=worst_flip_kappa)
    record("D3", "D", worst_order < 1e-5,
           description="bond storage order: identical prediction",
           pred_max_diff=worst_order)

    # D4 batch composition and molecule order
    worst_batch = 0.0
    others = [int(x) for x in bundle.fit_rows[5:12]]
    extra = {r: D.molecule_arrays(bundle, r) for r in others}
    for r, mol in mols.items():
        alone, _ = _predict(mol, model, "RICE_CUMULANT")
        with_extra, _ = _predict(mol, model, "RICE_CUMULANT", extra=extra)
        worst_batch = max(worst_batch, float(np.abs(alone - with_extra).max()))
    record("D4", "D", worst_batch < 1e-5,
           description="batch composition and molecule position: identical prediction",
           pred_max_diff=worst_batch)


def _relabel(mol: Mapping[str, Any], perm: np.ndarray) -> dict[str, Any]:
    """Apply atom relabelling ``perm`` (old -> new index)."""
    inverse = np.argsort(perm)
    return {
        "row": int(mol["row"]), "n_atoms": int(mol["n_atoms"]),
        "n_bonds": int(mol["n_bonds"]),
        "atom_feat": np.asarray(mol["atom_feat"], np.int64)[inverse],
        "bond_feat": np.asarray(mol["bond_feat"], np.int64),
        "bond_u": perm[np.asarray(mol["bond_u"], np.int64)],
        "bond_v": perm[np.asarray(mol["bond_v"], np.int64)],
        "graph_scalars": np.asarray(mol["graph_scalars"], np.float64),
    }


def _potentials_for(model: M.RiceModel, mol: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    st = S.build_molecule(mol)
    batch = T.collate([int(mol["row"])], {int(mol["row"]): st}, None, "cpu")
    with torch.no_grad():
        u, j = model.potentials(batch)
    return u.numpy(), j.numpy()


# ---------------------------------------------------------------------------
# E -- gradients and higher-order activity
# ---------------------------------------------------------------------------

def check_gradients(bundle: D.DataBundle) -> None:
    cfg = C.load_config()
    C.seed_everything(0)
    model, info = M.build_model(bundle.atom_dims, bundle.bond_dims, 0,
                                D.normalisers(bundle), config=cfg)
    model.train()
    rows = np.asarray(bundle.fit_rows[:64], np.int64)
    structs = S.build_all(bundle, rows)
    batch = T.collate(rows.tolist(), structs, bundle.raw, "cpu")
    y_std = bundle.labels_std()
    out = model(batch, arm="RICE_CUMULANT")
    target = torch.as_tensor(y_std[rows], dtype=torch.float32)
    loss = torch.mean((out["yhat_std"] - target) ** 2)
    loss.backward()
    grads = {name: float(torch.sqrt(sum((p.grad ** 2).sum() for p in module.parameters()
                                       if p.grad is not None)))
             for name, module in (("atom_mlp", model.atom_mlp), ("bond_mlp", model.bond_mlp),
                                  ("atom_emb", model.atom_emb), ("bond_emb", model.bond_emb),
                                  ("sentence_mlp", model.sentence_mlp),
                                  ("readout_mlp", model.readout_mlp))}
    finite = all(math.isfinite(v) for v in grads.values())
    nonzero = all(v > 0.0 for v in grads.values())
    record("E1", "E", finite and nonzero,
           description="first real batch: unary and pairwise potential gradients are "
                       "finite and non-zero, as are embedding/sentence/readout gradients",
           grad_norms=grads, loss=float(loss.detach()))

    # order-3 and order-4 irreducible terms really exist on non-degenerate test graphs
    tri = make_mol(3, [(0, 1), (1, 2), (0, 2)])
    u_t, j_t = field_potentials(tri)
    tri_entry = vectorised_phi_kappa(tri, u_t, j_t)
    cyc4 = make_mol(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    u_c, j_c = field_potentials(cyc4)
    cyc_entry = vectorised_phi_kappa(cyc4, u_c, j_c)
    chain4 = make_mol(4, [(0, 1), (1, 2), (2, 3)])
    u_ch, j_ch = field_potentials(chain4)
    chain_entry = vectorised_phi_kappa(chain4, u_ch, j_ch)
    kappa3 = float(np.abs(tri_entry["kappa"][3].numpy()).max())
    kappa4 = float(np.abs(cyc_entry["kappa"][4].numpy()).max())
    # chain rooted at an endpoint: the only root whose radius-2 set covers all four
    # atoms is 1 or 2; the endpoint-rooted 4-chain is not a valid word at all
    endpoint_ok = all(int(r) in (1, 2) for r in
                      chain_entry["batch"]["atoms_4"][:, 0].tolist())
    record("E2", "E", kappa3 > 1e-3 and kappa4 > 1e-3 and endpoint_ok,
           description="field-derived potentials: order-3 (3-cycle) and order-4 (4-cycle) "
                       "kappa are non-zero, and the radius-2 constraint excludes the "
                       "endpoint-rooted 4-chain word",
           kappa3_max=kappa3, kappa4_max=kappa4,
           chain4_root_atoms=chain_entry["batch"]["atoms_4"][:, 0].tolist())

    # 30 real training steps must reduce the standardised MSE
    model, _ = M.build_model(bundle.atom_dims, bundle.bond_dims, 0,
                             D.normalisers(bundle), config=cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["recipe"]["peak_lr"]),
                            weight_decay=float(cfg["recipe"]["weight_decay"]))
    rng = np.random.default_rng(0)
    order = rng.permutation(int(bundle.fit_rows.size))[:30 * 64]
    rows_all = np.asarray(bundle.fit_rows, np.int64)[order]
    structs_all = S.build_all(bundle, np.unique(rows_all))
    losses = []
    for step in range(30):
        part = rows_all[step * 64:(step + 1) * 64]
        batch = T.collate(part.tolist(), structs_all, bundle.raw, "cpu")
        out = model(batch, arm="RICE_CUMULANT")
        target = torch.as_tensor(y_std[part], dtype=torch.float32)
        loss = torch.mean((out["yhat_std"] - target) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))
    record("E3", "E", losses[-1] < losses[0]
           and float(np.mean(losses[-5:])) < float(np.mean(losses[:5]))
           and all(math.isfinite(x) for x in losses),
           description="30 optimiser steps on D0 Fit reduce the standardised MSE with no "
                       "NaN/Inf",
           first_loss=losses[0], last_loss=losses[-1],
           mean_first5=float(np.mean(losses[:5])), mean_last5=float(np.mean(losses[-5:])))


# ---------------------------------------------------------------------------
# F -- compute cost and vectorisation evidence
# ---------------------------------------------------------------------------

def check_cost(bundle: D.DataBundle) -> None:
    rows = np.union1d(bundle.fit_rows, bundle.eval_rows)
    t0 = time.time()
    structs = S.build_all(bundle, rows)
    prep = time.time() - t0
    stats = S.structure_stats(structs)
    record("F1", "F", stat_ok(stats),
           description="D0 root-conditioned word census: sizes 2/3/4 exist, and every "
                       "root is covered",
           **stats, prep_seconds=prep)

    cfg = C.load_config()
    model, info = M.build_model(bundle.atom_dims, bundle.bond_dims, 0,
                                D.normalisers(bundle), config=cfg)
    model.train()
    calls: dict[str, int] = {}

    def hook(name: str):
        def _f(_module, _inputs, _output):
            calls[name] = calls.get(name, 0) + 1
        return _f

    for name, module in (("atom_mlp", model.atom_mlp), ("bond_mlp", model.bond_mlp),
                         ("sentence_mlp", model.sentence_mlp),
                         ("readout_mlp", model.readout_mlp)):
        module.register_forward_hook(hook(name))
    for norm in model.order_norm:
        norm.register_forward_hook(hook("order_norm"))
    part = np.asarray(bundle.fit_rows[:64], np.int64)
    sub = {int(r): structs[int(r)] for r in part}
    batch = T.collate(part.tolist(), sub, bundle.raw, "cpu")
    t0 = time.time()
    out = model(batch, arm="RICE_CUMULANT")
    y_std = bundle.labels_std()
    target = torch.as_tensor(y_std[part], dtype=torch.float32)
    loss = torch.mean((out["yhat_std"] - target) ** 2)
    loss.backward()
    step_seconds = time.time() - t0
    n_words = int(batch["n_words"])
    vectorised = (calls.get("atom_mlp", 0) == 1 and calls.get("bond_mlp", 0) == 1
                  and calls.get("sentence_mlp", 0) == 1 and calls.get("readout_mlp", 0) == 1
                  and calls.get("order_norm", 0) == 4)
    record("F2", "F", vectorised,
           description="module forward calls per 64-molecule batch do not scale with the "
                       "word count: one atom MLP, one bond MLP, four order norms, one "
                       "sentence MLP and one readout for the whole batch",
           module_calls=calls, words_in_batch=n_words, fwd_bwd_seconds=step_seconds)
    ep = time.time()
    t0 = time.time()
    n_steps = int(math.ceil(int(bundle.fit_rows.size) / 64))
    one = 0.0
    for i in range(3):
        part = np.asarray(bundle.fit_rows[i * 64:(i + 1) * 64], np.int64)
        sub = {int(r): structs[int(r)] for r in part}
        batch = T.collate(part.tolist(), sub, bundle.raw, "cpu")
        t1 = time.time()
        out = model(batch, arm="RICE_CUMULANT")
        target = torch.as_tensor(y_std[part], dtype=torch.float32)
        loss = torch.mean((out["yhat_std"] - target) ** 2)
        loss.backward()
        one += time.time() - t1
    per_step = one / 3.0
    record("F3", "F", True,
           description="measured single-process CPU cost of one fit step (informational; "
                       "the formal runs are GPU)",
           steps_per_epoch=int(n_steps), cpu_step_seconds=per_step,
           cpu_epoch_seconds_estimate=per_step * n_steps,
           parameters=int(info["parameters"]["total"]),
           sentence_in_dim=int(info["sentence_in_dim"]),
           readout_in_dim=int(info["readout_in_dim"]),
           prep_seconds=prep, timer_seconds=time.time() - ep)


def stat_ok(stats: Mapping[str, Any]) -> bool:
    counts = stats["words_by_size"]
    return (int(counts["2"]) > 0 and int(counts["3"]) > 0 and int(counts["4"]) > 0
            and int(stats["roots_with_size"]["3"]) > 0
            and int(stats["roots_with_size"]["4"]) > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "checks.json"))
    args = parser.parse_args()
    C.assert_frozen_parent()
    graphs = all_small_graphs(4)
    check_inversion(len(graphs))
    check_connectivity()
    check_structure_sensitivity()
    bundle = D.load_bundle()
    check_invariances(bundle)
    check_gradients(bundle)
    check_cost(bundle)
    failed = [r for r in RECORDS if r["status"] != "PASS"]
    summary = {
        "round": "rooted_interaction_cumulant_encoder_pilot_v0",
        "config_sha256": C.sha256_json(C.load_config()),
        "code_sha256": C.code_sha(["rice_common.py", "rice_data.py", "rice_structure.py",
                                   "rice_energy.py", "rice_tensor.py", "rice_model.py",
                                   "rice_train.py", "rice_checks.py"]),
        "git_commit": C.git_commit(), "git_dirty": C.git_dirty(),
        "generated_utc": C.now_utc(),
        "n_checks": len(RECORDS), "n_pass": len(RECORDS) - len(failed),
        "n_fail": len(failed), "all_pass": not failed,
        "checks": RECORDS,
    }
    C.write_json(args.out, summary)
    print(f"\n{summary['n_pass']}/{summary['n_checks']} checks PASS -> {args.out}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()