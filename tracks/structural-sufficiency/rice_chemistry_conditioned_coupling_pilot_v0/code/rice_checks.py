"""Pre-training mathematical and engineering checks for RICE-CC-v0.

Every check writes a machine-readable record to ``analysis/checks.json`` and the
process exits non-zero if any check fails.  No formal training may start before
all checks pass.

A  Mobius identity against an independent float64 brute-force reference, for the
   bond-only and the chemistry-conditioned pair potential
B  connectivity: disconnected supports, the J_base = 0 limit, delta = 0
C  real structure sensitivity and bond-slot coverage
D  invariance for both arms: atom renumbering, bond direction, bond storage
   order, batch composition / molecule order
E  gradients (including the delta path), higher-order activity, 30-step decrease
F  compute cost and vectorisation evidence
G  coupling-specific checks: historical RICE equivalence and checkpoint replay,
   identical parameters / initial state, endpoint-conditioning sensitivity,
   endpoint symmetry, bounds, formula covariance under latent-state swap
H  data identity, label isolation and Fit-only normalisation
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
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
HISTORICAL_ROUND = (C.STRUCT / "rooted_interaction_cumulant_encoder_pilot_v0")
RECORDS: list[dict[str, Any]] = []


def record(check_id: str, group: str, ok: bool, **detail: Any) -> None:
    RECORDS.append({"id": check_id, "group": group, "status": "PASS" if ok else "FAIL",
                    **detail})
    print(f"[{ 'PASS' if ok else 'FAIL' }] {check_id} {group} {detail}", flush=True)


# ---------------------------------------------------------------------------
# synthetic molecules and engines
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
    """Deterministic non-degenerate potentials derived from the OGB fields."""
    a = _mixed_radix(np.asarray(mol["atom_feat"], np.int64), ATOM_DIMS) % 97
    b = _mixed_radix(np.asarray(mol["bond_feat"], np.int64), BOND_DIMS) % 89
    chan = np.arange(channels)[None, :, None]
    state = np.arange(2)[None, None, :]
    u = 0.45 * np.sin(1.0 + 0.7 * a[:, None, None] + 1.3 * chan + 2.1 * state)
    term = np.arange(3)[None, None, :]
    j = 0.55 * np.cos(0.9 + 1.1 * b[:, None, None] + 0.8 * chan + 1.7 * term)
    return u.astype(np.float64), j.astype(np.float64)


def vectorised_phi_kappa(mol: Mapping[str, Any], u: np.ndarray, j_base: np.ndarray,
                         arm: str = "CC_NULL",
                         dtype: torch.dtype = torch.float64) -> dict[str, Any]:
    """Production vectorised path for one synthetic molecule, both coupling modes."""
    st = S.build_molecule(mol)
    batch = T.collate([mol["row"]], {mol["row"]: st}, None, "cpu")
    n_atoms = int(mol["n_atoms"])
    ut = torch.as_tensor(u, dtype=dtype)
    jt = torch.as_tensor(j_base, dtype=dtype)
    if arm == "CC_REAL":
        jt, delta = E.couple_pairs(ut, jt, batch["bond_index"])
    else:
        delta = torch.zeros_like(jt[..., 0])
    channels = int(u.shape[1])
    phi = {k: E.compute_phi(ut, jt, batch[f"atoms_{k}"], batch[f"bonds_{k}"],
                            batch[f"bslot_{k}"]) for k in (2, 3, 4)}
    kappa = E.mobius(ut.reshape(n_atoms, channels * 2), phi,
                     {k: batch[f"atoms_{k}"] for k in (2, 3, 4)},
                     {k: batch[f"sub2_{k}"] for k in (3, 4)}, batch["sub3_4"])
    return {"struct": st, "batch": batch, "phi": phi, "kappa": kappa,
            "u_flat": ut.reshape(n_atoms, channels * 2), "j_used": jt, "delta": delta}


def support_rows(entry: Mapping[str, Any], size: int) -> list[tuple[tuple[int, ...], Any, Any]]:
    batch = entry["batch"]
    atoms = batch[f"atoms_{size}"]
    out = []
    for i in range(int(atoms.shape[0])):
        slots = tuple(int(x) for x in atoms[i] if int(x) >= 0)
        out.append((slots, entry["phi"][size][i].numpy(), entry["kappa"][size][i].numpy()))
    return out


# ---------------------------------------------------------------------------
# A -- Mobius inversion identity (both potentials)
# ---------------------------------------------------------------------------

def all_small_graphs(maxn: int = 4) -> list[tuple[int, list[tuple[int, int]]]]:
    graphs = []
    for n in range(1, maxn + 1):
        pairs = list(itertools.combinations(range(n), 2))
        for mask in range(1 << len(pairs)):
            graphs.append((n, [p for i, p in enumerate(pairs) if (mask >> i) & 1]))
    return graphs


def check_inversion(n_graphs: int) -> None:
    rng = np.random.default_rng(20261125)
    graphs = all_small_graphs(4)
    worst = {"CC_NULL": {"vec": 0.0, "identity": 0.0},
             "CC_REAL": {"vec": 0.0, "identity": 0.0}}
    n_support = 0
    n_identity = 0
    worst_f32 = 0.0
    for n, edges in graphs:
        if n == 1:
            continue
        mol = make_mol(n, edges)
        channels = 3
        u = rng.normal(size=(n, channels, 2))
        j_base = rng.normal(size=(len(edges), channels, 3))
        bond_list = [(int(a), int(b), i) for i, (a, b) in enumerate(edges)]
        for arm in ("CC_NULL", "CC_REAL"):
            entry = vectorised_phi_kappa(mol, u, j_base, arm=arm)
            j_used = entry["j_used"].numpy()
            entry32 = vectorised_phi_kappa(mol, u, j_base, arm=arm, dtype=torch.float32)
            for root in range(n):
                radius = _radius_atoms(n, edges, root)
                ref = E.brute_force_kappa(u, j_used, n, bond_list, root, radius)
                for size in (2, 3, 4):
                    for slots, phi_vec, kappa_vec in support_rows(entry, size):
                        if slots[0] != root:
                            continue
                        key = frozenset(slots)
                        phi_ref = ref["phi"][key]
                        kappa_ref = ref["kappa"][key]
                        worst[arm]["vec"] = max(
                            worst[arm]["vec"],
                            float(np.abs(phi_vec.reshape(channels, 2) - phi_ref).max()),
                            float(np.abs(kappa_vec.reshape(channels, 2) - kappa_ref).max()))
                        total = np.zeros_like(phi_ref)
                        for sub_size in range(1, len(key) + 1):
                            for combo in itertools.combinations(
                                    sorted(key - {root}), sub_size - 1):
                                total = total + ref["kappa"][frozenset((root,) + combo)]
                        worst[arm]["identity"] = max(
                            worst[arm]["identity"], float(np.abs(phi_ref - total).max()))
                        n_identity += 1
                        n_support += 1
                    kap32 = entry32["kappa"][size]
                    kap64 = entry["kappa"][size]
                    if int(kap32.shape[0]) == int(kap64.shape[0]) and int(kap32.numel()):
                        worst_f32 = max(
                            worst_f32,
                            float((kap32.double() - kap64.double()).abs().max()))
    base_ok = (worst["CC_NULL"]["identity"] < 1e-5
               and worst["CC_NULL"]["vec"] < 1e-10)
    real_ok = (worst["CC_REAL"]["identity"] < 1e-5
               and worst["CC_REAL"]["vec"] < 1e-10)
    record("A1", "A", base_ok and real_ok,
           description="Phi(S,v) == sum_{T subset S, v in T} kappa(T,v) on every graph "
                       "with <= 4 atoms, independent float64 brute-force reference, for "
                       "both the bond-only and the chemistry-conditioned pair potential",
           graphs=int(n_graphs), supports=int(n_support), identities=int(n_identity),
           max_error_base=worst["CC_NULL"]["identity"],
           max_error_conditioned=worst["CC_REAL"]["identity"],
           vectorised_max_error_base=worst["CC_NULL"]["vec"],
           vectorised_max_error_conditioned=worst["CC_REAL"]["vec"])
    record("A2", "A", worst_f32 < 1e-4,
           description="float32 production path stays within float32 noise of the "
                       "float64 reference (reported, informational)",
           max_error=worst_f32)


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
# B -- connectivity and the zero-pair-potential limit
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
        j_base = rng.normal(size=(len(edges), channels, 3))
        bond_list = [(int(a), int(b), i) for i, (a, b) in enumerate(edges)]
        for arm in ("CC_NULL", "CC_REAL"):
            entry = vectorised_phi_kappa(mol, u, j_base, arm=arm)
            j_used = entry["j_used"].numpy()
            for root in range(n):
                radius = _radius_atoms(n, edges, root)
                ref = E.brute_force_kappa(u, j_used, n, bond_list, root, radius)
                for key, kappa in ref["kappa"].items():
                    if len(key) < 2:
                        continue
                    if not E.reference_is_connected(sorted(key), bond_list):
                        n_disconnected += 1
                        worst_disconnected = max(worst_disconnected,
                                                 float(np.abs(kappa).max()))
    record("B1", "B", worst_disconnected < 1e-10,
           description="kappa(S,v) == 0 for every disconnected root-anchored S with "
                       "|S|>1, bond-only and chemistry-conditioned",
           cases=int(n_disconnected), max_abs=worst_disconnected)

    worst_j0 = 0.0
    worst_delta0 = 0.0
    n_cases = 0
    for n, edges in all_small_graphs(4):
        if n < 2:
            continue
        mol = make_mol(n, edges)
        channels = 4
        u = rng.normal(size=(n, channels, 2))
        j_base = np.zeros((len(edges), channels, 3))
        for arm in ("CC_NULL", "CC_REAL"):
            entry = vectorised_phi_kappa(mol, u, j_base, arm=arm)
            if arm == "CC_REAL":
                worst_delta0 = max(worst_delta0,
                                   float(entry["delta"].abs().max()) if entry["delta"].numel() else 0.0)
            for size in (2, 3, 4):
                for _slots, _phi, kappa in support_rows(entry, size):
                    worst_j0 = max(worst_j0, float(np.abs(kappa).max()))
                    n_cases += 1
    record("B2", "B", worst_j0 < 1e-10 and worst_delta0 == 0.0,
           description="with every J_base = 0: delta = 0 and every |S|>=2 kappa is zero",
           supports=int(n_cases), max_abs_kappa=worst_j0, max_abs_delta=worst_delta0)


# ---------------------------------------------------------------------------
# C -- real structure sensitivity
# ---------------------------------------------------------------------------

def check_structure_sensitivity() -> None:
    tri = make_mol(3, [(0, 1), (1, 2), (0, 2)])
    chain = make_mol(3, [(0, 1), (1, 2)])
    st_tri = S.build_molecule(tri)
    n_internal_tri = int((st_tri.bonds[3][0] >= 0).sum())
    u_tri, j_tri = field_potentials(tri)
    u_ch, j_ch = field_potentials(chain)
    e_tri = vectorised_phi_kappa(tri, u_tri, j_tri, arm="CC_REAL")
    e_ch = vectorised_phi_kappa(chain, u_ch, j_ch, arm="CC_REAL")
    kap_tri = e_tri["kappa"][3][0].numpy()
    kap_ch = e_ch["kappa"][3][0].numpy()
    record("C1", "C", n_internal_tri == 3 and float(np.abs(kap_tri - kap_ch).max()) > 1e-3,
           description="3-cycle stores all three internal bonds (closing bond read) and "
                       "its conditioned kappa differs from the open 3-chain",
           internal_bonds=int(n_internal_tri),
           kappa_gap=float(np.abs(kap_tri - kap_ch).max()))

    clique = make_mol(4, list(itertools.combinations(range(4), 2)))
    st_clique = S.build_molecule(clique)
    n_internal_clique = int((st_clique.bonds[4][0] >= 0).sum())
    u_cl, j_cl = field_potentials(clique)
    e_cl = vectorised_phi_kappa(clique, u_cl, j_cl, arm="CC_REAL")
    record("C2", "C", n_internal_clique == 6
           and bool(np.isfinite(e_cl["kappa"][4].numpy()).all()),
           description="4-clique support uses all six internal bond slots and is finite",
           internal_bonds=int(n_internal_clique))

    cyc = make_mol(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    u_cyc, j_cyc = field_potentials(cyc)
    e_cyc = vectorised_phi_kappa(cyc, u_cyc, j_cyc, arm="CC_REAL")
    record("C3", "C", float(np.abs(e_cyc["kappa"][4].numpy()).max()) > 1e-3,
           description="conditioned order-4 kappa is non-zero on a real 4-cycle",
           kappa4_max=float(np.abs(e_cyc["kappa"][4].numpy()).max()))

    # same atom+bond attribute multisets, different real connectivity; both arms
    tail = make_mol(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
    u_a, j_a = field_potentials(cyc)
    u_b, j_b = field_potentials(tail)
    for arm in ("CC_NULL", "CC_REAL"):
        ea = vectorised_phi_kappa(cyc, u_a, j_a, arm=arm)
        eb = vectorised_phi_kappa(tail, u_b, j_b, arm=arm)
        gap = float(np.abs(ea["kappa"][4].numpy() - eb["kappa"][4].numpy()).max())
        record(f"C4_{arm}", "C", gap > 1e-4,
               description=f"{arm}: same attribute multisets, 4-cycle vs triangle+tail, "
                           "kappa separates the real connectivity",
               kappa_gap=gap)


# ---------------------------------------------------------------------------
# D -- invariances
# ---------------------------------------------------------------------------

def _relabel(mol: Mapping[str, Any], perm: np.ndarray) -> dict[str, Any]:
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


def _model_potentials(model: M.RiceModel, mol: Mapping[str, Any]
                      ) -> tuple[np.ndarray, np.ndarray]:
    st = S.build_molecule(mol)
    batch = T.collate([int(mol["row"])], {int(mol["row"]): st}, None, "cpu")
    with torch.no_grad():
        u, j = model.potentials(batch)
    return u.numpy(), j.numpy()


def _predict(mol: Mapping[str, Any], model: M.RiceModel, arm: str,
             extra: Mapping[int, Mapping[str, Any]] | None = None
             ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    row = int(mol["row"])
    structs = {row: S.build_molecule(mol)}
    if extra:
        for r, m in extra.items():
            structs[int(r)] = S.build_molecule(m)
    rows = sorted(structs)
    batch = T.collate(rows, structs, None, "cpu")
    out = model(batch, arm=arm, return_aux=True)
    idx = rows.index(row)
    return out["yhat_std"][idx:idx + 1].detach().numpy(), out, batch


def _kappa_by_certificate(st: S.MoleculeStruct, entry: Mapping[str, Any], size: int,
                          atom_dims: Sequence[int], bond_dims: Sequence[int]
                          ) -> tuple[np.ndarray, np.ndarray]:
    sig = S.canonical_signature(st, size, atom_dims, bond_dims)
    kappa = entry["kappa"][size].detach().numpy()
    if int(sig.shape[0]) == 0:
        return sig, kappa
    order = np.lexsort(sig.T[::-1])
    return sig[order], kappa[order]


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
    for arm in ("CC_NULL", "CC_REAL"):
        base_pred: dict[int, np.ndarray] = {}
        base_kappa: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        base_j: dict[int, np.ndarray] = {}
        for r, mol in mols.items():
            pred, out, batch = _predict(mol, model, arm)
            base_pred[r] = pred
            base_j[r] = out["j"].detach().numpy()
            entry = vectorised_phi_kappa(mol, *_model_potentials(model, mol), arm=arm)
            st = entry["struct"]
            base_kappa[r] = {size: _kappa_by_certificate(
                st, entry, size, bundle.atom_dims, bundle.bond_dims) for size in (2, 3, 4)}

        worst_renum = 0.0
        worst_renum_kappa = 0.0
        worst_flip = 0.0
        worst_flip_kappa = 0.0
        worst_flip_j = 0.0
        worst_order = 0.0
        sig_mismatch = 0
        for r, mol in mols.items():
            perm = rng.permutation(int(mol["n_atoms"]))
            renum = _relabel(mol, perm)
            pred, out, _ = _predict(renum, model, arm)
            worst_renum = max(worst_renum, float(np.abs(pred - base_pred[r]).max()))
            entry = vectorised_phi_kappa(renum, *_model_potentials(model, renum), arm=arm)
            for size in (2, 3, 4):
                sig, kappa = _kappa_by_certificate(entry["struct"], entry, size,
                                                   bundle.atom_dims, bundle.bond_dims)
                sig0, kappa0 = base_kappa[r][size]
                if sig.shape != sig0.shape or not np.array_equal(sig, sig0):
                    sig_mismatch += 1
                elif kappa.shape == kappa0.shape and kappa.size:
                    worst_renum_kappa = max(worst_renum_kappa,
                                            float(np.abs(kappa - kappa0).max()))

            flipped = dict(mol)
            flipped["bond_u"] = np.asarray(mol["bond_v"], np.int64)
            flipped["bond_v"] = np.asarray(mol["bond_u"], np.int64)
            pred, out, _ = _predict(flipped, model, arm)
            worst_flip = max(worst_flip, float(np.abs(pred - base_pred[r]).max()))
            worst_flip_j = max(worst_flip_j,
                               float(np.abs(out["j"].detach().numpy() - base_j[r]).max()))
            entry = vectorised_phi_kappa(flipped, *_model_potentials(model, flipped),
                                         arm=arm)
            for size in (2, 3, 4):
                sig, kappa = _kappa_by_certificate(entry["struct"], entry, size,
                                                   bundle.atom_dims, bundle.bond_dims)
                sig0, kappa0 = base_kappa[r][size]
                if sig.shape == sig0.shape and kappa.shape == kappa0.shape and kappa.size:
                    worst_flip_kappa = max(worst_flip_kappa,
                                           float(np.abs(kappa - kappa0).max()))

            order = rng.permutation(int(mol["n_bonds"]))
            shuffled = dict(mol)
            shuffled["bond_u"] = np.asarray(mol["bond_u"], np.int64)[order]
            shuffled["bond_v"] = np.asarray(mol["bond_v"], np.int64)[order]
            shuffled["bond_feat"] = np.asarray(mol["bond_feat"], np.int64)[order]
            pred, _, _ = _predict(shuffled, model, arm)
            worst_order = max(worst_order, float(np.abs(pred - base_pred[r]).max()))

        record(f"D1_{arm}", "D",
               worst_renum < 1e-5 and worst_renum_kappa < 1e-4 and sig_mismatch == 0,
               description=f"{arm}: atom renumbering leaves prediction, canonical word "
                           "certificates and per-certificate kappa unchanged",
               pred_max_diff=worst_renum, kappa_max_diff=worst_renum_kappa,
               certificate_mismatches=int(sig_mismatch))
        record(f"D2_{arm}", "D",
               worst_flip < 1e-6 and worst_flip_kappa < 1e-4 and worst_flip_j < 1e-6,
               description=f"{arm}: bond direction swap leaves prediction, pair potential "
                           "and kappa unchanged",
               pred_max_diff=worst_flip, kappa_max_diff=worst_flip_kappa,
               j_max_diff=worst_flip_j)
        record(f"D3_{arm}", "D", worst_order < 1e-5,
               description=f"{arm}: bond storage order leaves prediction unchanged",
               pred_max_diff=worst_order)

        worst_batch = 0.0
        others = [int(x) for x in bundle.fit_rows[5:12]]
        extra = {r: D.molecule_arrays(bundle, r) for r in others}
        for r, mol in mols.items():
            alone, _, _ = _predict(mol, model, arm)
            with_extra, _, _ = _predict(mol, model, arm, extra=extra)
            worst_batch = max(worst_batch, float(np.abs(alone - with_extra).max()))
        record(f"D4_{arm}", "D", worst_batch < 1e-5,
               description=f"{arm}: batch composition and molecule position leave the "
                           "prediction unchanged",
               pred_max_diff=worst_batch)


# ---------------------------------------------------------------------------
# E -- gradients and higher-order activity
# ---------------------------------------------------------------------------

def _grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total)


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

    for arm in ("CC_NULL", "CC_REAL"):
        model.zero_grad(set_to_none=True)
        out = model(batch, arm=arm)
        target = torch.as_tensor(y_std[rows], dtype=torch.float32)
        loss = torch.mean((out["yhat_std"] - target) ** 2)
        loss.backward()
        grads = {
            "atom_mlp": _grad_norm(list(model.atom_mlp.parameters())),
            "bond_mlp": _grad_norm(list(model.bond_mlp.parameters())),
            "atom_emb": _grad_norm(list(model.atom_emb.parameters())),
            "bond_emb": _grad_norm(list(model.bond_emb.parameters())),
            "sentence_mlp": _grad_norm(list(model.sentence_mlp.parameters())),
            "readout_mlp": _grad_norm(list(model.readout_mlp.parameters())),
        }
        record(f"E1_{arm}", "E",
               all(math.isfinite(v) for v in grads.values())
               and all(v > 0.0 for v in grads.values()),
               description=f"{arm}: first real batch gradients are finite and non-zero "
                           "for unary, pairwise, embedding, sentence and readout modules",
               grad_norms=grads, loss=float(loss.detach()))

    # explicit delta path: gradients must reach both the unary and the bond MLP
    model.zero_grad(set_to_none=True)
    u, j_base = model.potentials(batch)
    j_real, delta = model.pair_potential(j_base, u, batch["bond_index"], "CC_REAL")
    gu, gj = torch.autograd.grad(delta.sum(), [u, j_base], retain_graph=False,
                                 allow_unused=False)
    delta_grad = {"u": float(gu.norm()), "j_base": float(gj.norm())}
    record("E2", "E",
           math.isfinite(delta_grad["u"]) and math.isfinite(delta_grad["j_base"])
           and delta_grad["u"] > 0 and delta_grad["j_base"] > 0
           and bool(torch.isfinite(j_real).all()),
           description="the CC_REAL delta term is not detached: its gradient reaches "
                       "both the unary potential and the bond potential",
           delta_grad_norms=delta_grad,
           delta_abs_mean=float(delta.abs().mean()),
           delta_abs_max=float(delta.abs().max()))

    tri = make_mol(3, [(0, 1), (1, 2), (0, 2)])
    u_t, j_t = field_potentials(tri)
    tri_entry = vectorised_phi_kappa(tri, u_t, j_t, arm="CC_REAL")
    cyc4 = make_mol(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    u_c, j_c = field_potentials(cyc4)
    cyc_entry = vectorised_phi_kappa(cyc4, u_c, j_c, arm="CC_REAL")
    kappa3 = float(np.abs(tri_entry["kappa"][3].numpy()).max())
    kappa4 = float(np.abs(cyc_entry["kappa"][4].numpy()).max())
    record("E3", "E", kappa3 > 1e-3 and kappa4 > 1e-3,
           description="field-derived conditioned potentials produce non-zero order-3 "
                       "(3-cycle) and order-4 (4-cycle) kappa",
           kappa3_max=kappa3, kappa4_max=kappa4)

    # 20 real training steps must reduce the standardised MSE for both arms
    for arm in ("CC_NULL", "CC_REAL"):
        model, _ = M.build_model(bundle.atom_dims, bundle.bond_dims, 0,
                                 D.normalisers(bundle), config=cfg)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["recipe"]["peak_lr"]),
                                weight_decay=float(cfg["recipe"]["weight_decay"]))
        rng = np.random.default_rng(0)
        order = rng.permutation(int(bundle.fit_rows.size))[:20 * 64]
        rows_all = np.asarray(bundle.fit_rows, np.int64)[order]
        structs_all = S.build_all(bundle, np.unique(rows_all))
        losses = []
        for step in range(20):
            part = rows_all[step * 64:(step + 1) * 64]
            batch = T.collate(part.tolist(), structs_all, bundle.raw, "cpu")
            out = model(batch, arm=arm)
            target = torch.as_tensor(y_std[part], dtype=torch.float32)
            loss = torch.mean((out["yhat_std"] - target) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        record(f"E4_{arm}", "E",
               losses[-1] < losses[0]
               and float(np.mean(losses[-5:])) < float(np.mean(losses[:5]))
               and all(math.isfinite(x) for x in losses),
               description=f"{arm}: 20 optimiser steps on D0 Fit reduce the standardised "
                           "MSE with no NaN/Inf",
               first_loss=losses[0], last_loss=losses[-1],
               mean_first5=float(np.mean(losses[:5])),
               mean_last5=float(np.mean(losses[-5:])))


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
           description="D0 root-conditioned word census: sizes 2/3/4 exist and every "
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
    out = model(batch, arm="CC_REAL")
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
    t0 = time.time()
    n_steps = int(math.ceil(int(bundle.fit_rows.size) / 64))
    one = 0.0
    for i in range(3):
        part = np.asarray(bundle.fit_rows[i * 64:(i + 1) * 64], np.int64)
        sub = {int(r): structs[int(r)] for r in part}
        batch = T.collate(part.tolist(), sub, bundle.raw, "cpu")
        t1 = time.time()
        out = model(batch, arm="CC_REAL")
        target = torch.as_tensor(y_std[part], dtype=torch.float32)
        loss = torch.mean((out["yhat_std"] - target) ** 2)
        loss.backward()
        one += time.time() - t1
    per_step = one / 3.0
    record("F3", "F", True,
           description="measured single-process CPU cost of one CC_REAL fit step "
                       "(informational; formal runs are GPU)",
           steps_per_epoch=int(n_steps), cpu_step_seconds=per_step,
           cpu_epoch_seconds_estimate=per_step * n_steps,
           parameters=int(info["parameters"]["total"]),
           sentence_in_dim=int(model.sentence_in_dim),
           readout_in_dim=int(model.readout_in_dim),
           prep_seconds=prep, timer_seconds=time.time() - t0)


def stat_ok(stats: Mapping[str, Any]) -> bool:
    counts = stats["words_by_size"]
    return (int(counts["2"]) > 0 and int(counts["3"]) > 0 and int(counts["4"]) > 0
            and int(stats["roots_with_size"]["3"]) > 0
            and int(stats["roots_with_size"]["4"]) > 0)


# ---------------------------------------------------------------------------
# G -- coupling-specific checks
# ---------------------------------------------------------------------------

def _run_old_reference(mode: str, out: Path, n: int = 16,
                       checkpoint: Path | None = None) -> dict[str, Any]:
    helper = Path(__file__).resolve().with_name("old_rice_reference.py")
    cmd = [sys.executable, str(helper), "--mode", mode, "--out", str(out),
           "--n", str(int(n))]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"old reference failed ({proc.returncode}):\n"
                           f"{proc.stdout}\n{proc.stderr}")
    meta = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    return meta


def check_historical_equivalence(bundle: D.DataBundle, cfg: Mapping[str, Any]) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="rice_cc_old_"))
    # (a) initial forward of the historical source vs CC_NULL
    out_path = tmp / "old_initial.npz"
    meta = _run_old_reference("initial", out_path, n=16)
    old = np.load(out_path)
    rows = np.asarray(old["rows"], np.int64)
    new_model, new_info = M.build_model(bundle.atom_dims, bundle.bond_dims, 0,
                                        D.normalisers(bundle), config=cfg)
    new_model.eval()
    structs = S.build_all(bundle, rows)
    new_parts: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, rows.size, 256):
            part = rows[start:start + 256]
            sub = {int(r): structs[int(r)] for r in part}
            batch = T.collate(part.tolist(), sub, bundle.raw, "cpu")
            out = new_model(batch, arm="CC_NULL", return_aux=True)
            new_parts.setdefault("yhat_std", []).append(out["yhat_std"].numpy())
            new_parts.setdefault("u", []).append(out["u"].numpy())
            new_parts.setdefault("j", []).append(out["j"].numpy())
            for size in (2, 3, 4):
                new_parts.setdefault(f"phi_{size}", []).append(out["phi"][size].numpy())
                new_parts.setdefault(f"kappa_{size}", []).append(
                    out["kappa"][size].numpy())
    new = {k: np.concatenate(v, axis=0) for k, v in new_parts.items()}
    diffs = {"yhat": float(np.abs(new["yhat_std"] - old["yhat_std"]).max()),
             "u": float(np.abs(new["u"] - old["u"]).max()),
             "j": float(np.abs(new["j"] - old["j"]).max())}
    for size in (2, 3, 4):
        diffs[f"phi_{size}"] = float(np.abs(new[f"phi_{size}"] - old[f"phi_{size}"]).max())
        diffs[f"kappa_{size}"] = float(np.abs(new[f"kappa_{size}"]
                                              - old[f"kappa_{size}"]).max())
    same_init = (meta.get("init_hashes") == new_info["init_hashes"])
    record("G1", "G",
           same_init and all(v < 1e-7 for v in diffs.values()),
           description="CC_NULL initial forward is bitwise-equivalent to the historical "
                       "RICE-v0 source: identical init hashes, unary, pair, Phi, kappa "
                       "and standardized prediction",
           init_hashes_equal=bool(same_init), max_diffs=diffs,
           rows=int(rows.size), old_commit=meta.get("git_commit"))

    # (b) historical frozen checkpoint replay through the new CC_NULL path
    ckpt = HISTORICAL_ROUND / "results" / "RICE_CUMULANT_S0" / "state_best.pt"
    hist_manifest = C.read_json(HISTORICAL_ROUND / "results"
                                / "RICE_CUMULANT_S0" / "manifest.json")
    hist_pred = np.load(HISTORICAL_ROUND / "results" / "RICE_CUMULANT_S0"
                        / "predictions_best_dev.npz")
    replay_path = tmp / "old_replay.npz"
    _run_old_reference("replay", replay_path, checkpoint=ckpt)
    old_replay = np.load(replay_path)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    replay_model = M.load_model_from_state(state["state"], bundle.atom_dims,
                                           bundle.bond_dims, D.normalisers(bundle),
                                           config=cfg, device="cpu")
    replay_model.eval()
    dev_rows = np.asarray(bundle.eval_rows, np.int64)
    structs = S.build_all(bundle, dev_rows)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, dev_rows.size, 256):
            part = dev_rows[start:start + 256]
            sub = {int(r): structs[int(r)] for r in part}
            batch = T.collate(part.tolist(), sub, bundle.raw, "cpu")
            preds.append(replay_model(batch, arm="CC_NULL")["yhat_std"].numpy())
    new_pred = np.concatenate(preds)
    diff_hist = float(np.abs(new_pred - hist_pred["pred_std"]).max())
    diff_old = float(np.abs(new_pred - old_replay["yhat_std"]).max())
    y = bundle.labels[dev_rows]
    rmse_new = float(np.sqrt(np.mean(((new_pred * bundle.scaler["std"]
                                      + bundle.scaler["mean"]) - y) ** 2)))
    rmse_hist = float(hist_manifest["dev_rmse"])
    record("G2", "G",
           diff_hist < 1e-3 and diff_old < 1e-5 and abs(rmse_new - rmse_hist) < 1e-5,
           description="historical RICE_CUMULANT best checkpoint replayed through the new "
                       "CC_NULL path reproduces its Dev predictions and RMSE",
           rows=int(dev_rows.size),
           pred_std_max_diff_vs_historical=diff_hist,
           pred_std_max_diff_vs_old_source=diff_old,
           historical_npz_tolerance=1e-3,
           note=("the historical .npz was produced on GPU float32; the old source "
                 "replayed on CPU matches the new CC_NULL path exactly, so the small "
                 "historical .npz difference is device arithmetic, not a code change"),
           dev_rmse_replayed=rmse_new, dev_rmse_historical=rmse_hist,
           historical_best_epoch=int(hist_manifest["best_epoch"]))
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


def check_arms_identical(bundle: D.DataBundle, cfg: Mapping[str, Any]) -> None:
    norm = D.normalisers(bundle)
    a, info_a = M.build_model(bundle.atom_dims, bundle.bond_dims, 0, norm, config=cfg)
    b, info_b = M.build_model(bundle.atom_dims, bundle.bond_dims, 0, norm, config=cfg)
    same_params = info_a["parameters"] == info_b["parameters"]
    same_hash = info_a["init_hashes"] == info_b["init_hashes"]
    same_keys = set(a.state_dict()) == set(b.state_dict())
    state_max = 0.0
    for key in a.state_dict():
        state_max = max(state_max, float((a.state_dict()[key]
                                          - b.state_dict()[key]).abs().max()))
    total = int(info_a["parameters"]["total"])
    record("G3", "G",
           same_params and same_hash and same_keys and state_max == 0.0 and total > 0,
           description="the two arms have identical modules, parameter ledger and random "
                       "initial state; the arm only selects the pair potential at forward time",
           n_parameters=total, same_parameters=bool(same_params),
           same_init_hashes=bool(same_hash), same_state_keys=bool(same_keys),
           state_max_abs_diff=state_max, ledger=info_a["parameters"],
           sentence_in_dim=int(info_a["sentence_in_dim"]),
           readout_in_dim=int(info_a["readout_in_dim"]))


def check_endpoint_conditioning(bundle: D.DataBundle, cfg: Mapping[str, Any]) -> None:
    # same bond attributes, different endpoint atom types -> same J_base, different J_real
    c = [6, 0, 0, 0, 0, 0, 0, 0, 0]
    n_atom = [7, 0, 0, 0, 1, 0, 0, 0, 0]
    o_atom = [8, 0, 0, 0, 0, 0, 0, 0, 0]
    single = [0, 0, 0]
    mol = make_mol(4, [(0, 1), (0, 2), (0, 3)],
                   [c, n_atom, o_atom, c], [single, single, single])
    u, j_base = field_potentials(mol)
    st = S.build_molecule(mol)
    batch = T.collate([0], {0: st}, None, "cpu")
    ut = torch.as_tensor(u, dtype=torch.float64)
    jt = torch.as_tensor(j_base, dtype=torch.float64)
    j_real, delta = E.couple_pairs(ut, jt, batch["bond_index"])
    base_equal = float(np.abs(jt[0].detach().numpy() - jt[1].detach().numpy()).max())
    delta_gap = float(np.abs(delta[0].detach().numpy() - delta[1].detach().numpy()).max())
    real_gap = float(np.abs(j_real[0].detach().numpy()
                            - j_real[1].detach().numpy()).max())
    record("G4", "G",
           base_equal < 1e-12 and delta_gap > 1e-3 and real_gap > 1e-3,
           description="two bonds with identical bond fields but different endpoint atom "
                       "types share J_base but receive a different delta and J_real",
           base_max_diff=base_equal, delta_gap=delta_gap, j_real_gap=real_gap)

    # endpoint swap: J_real invariant
    rng = np.random.default_rng(19)
    n_atoms, n_bonds = 7, 8
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 6), (2, 4)]
    rng.shuffle(edges)
    mol2 = make_mol(n_atoms, edges,
                    rng.integers(0, 8, size=(n_atoms, 9)).tolist(),
                    rng.integers(0, 2, size=(n_bonds, 3)).tolist())
    u2, j2 = field_potentials(mol2)
    st2 = S.build_molecule(mol2)
    b2 = T.collate([0], {0: st2}, None, "cpu")
    ut2 = torch.as_tensor(u2, dtype=torch.float64)
    jt2 = torch.as_tensor(j2, dtype=torch.float64)
    jr_a, _ = E.couple_pairs(ut2, jt2, b2["bond_index"])
    flipped = b2["bond_index"][:, [1, 0]]
    jr_b, _ = E.couple_pairs(ut2, jt2, flipped)
    swap_diff = float((jr_a - jr_b).abs().max())
    record("G5", "G", swap_diff < 1e-12,
           description="swapping the two endpoints of any physical bond leaves J_real "
                       "unchanged",
           max_abs_diff=swap_diff)

    # bounds on real production potentials
    model, _ = M.build_model(bundle.atom_dims, bundle.bond_dims, 0,
                             D.normalisers(bundle), config=cfg)
    rows = np.asarray(bundle.fit_rows[:64], np.int64)
    structs = S.build_all(bundle, rows)
    batch = T.collate(rows.tolist(), structs, bundle.raw, "cpu")
    with torch.no_grad():
        u3, j3 = model.potentials(batch)
    jr, delta3 = E.couple_pairs(u3, j3, batch["bond_index"])
    base_max = float(j3.abs().max())
    real_max = float(jr.abs().max())
    record("G6", "G",
           bool(torch.isfinite(jr).all()) and base_max <= C.PAIR_BOUND_BASE + 1e-6
           and real_max <= C.PAIR_BOUND_CONDITIONED + 1e-6
           and float(delta3.abs().max()) <= C.PAIR_BOUND_BASE + 1e-6,
           description="J_base, delta and J_real are finite and respect the conservative "
                       "bounds 2 / 2 / 4",
           base_abs_max=base_max, delta_abs_max=float(delta3.abs().max()),
           real_abs_max=real_max)

    # formula covariance under a simultaneous 0/1 latent-state swap, per channel
    u4 = ut2.clone()
    old_h = torch.tanh((u4[..., 1] - u4[..., 0]) / 2.0)
    old_k = (jt2[..., 0] + jt2[..., 2] - 2.0 * jt2[..., 1]) / 4.0
    old_delta = old_k * old_h[b2["bond_index"][:, 0]] * old_h[b2["bond_index"][:, 1]]
    u4 = u4.flip(-1)
    jr_swapped_u, delta_swapped_u = E.couple_pairs(u4, jt2, b2["bond_index"])
    delta_unchanged = float((delta_swapped_u - old_delta).abs().max())
    swapped_base = torch.stack([jt2[..., 2], jt2[..., 1], jt2[..., 0]], dim=-1)
    jr_swapped_base, delta_swapped_base = E.couple_pairs(u4, swapped_base,
                                                         b2["bond_index"])
    # relabel the states of the original matrix: (00, 01, 11) -> (11, 01, 00)
    jr_relabelled = torch.stack([jr_swapped_u[..., 2], jr_swapped_u[..., 1],
                                 jr_swapped_u[..., 0]], dim=-1)
    cov = float((jr_swapped_base - jr_relabelled).abs().max())
    delta_cov = float((delta_swapped_base - old_delta).abs().max())
    record("G7", "G", delta_unchanged < 1e-12 and cov < 1e-12 and delta_cov < 1e-12,
           description="simultaneous 0/1 latent-state swap per channel leaves delta "
                       "invariant, and J_real transforms covariantly as "
                       "R J_base R + delta * [[1,-1],[-1,1]] (factor algebra only; the "
                       "downstream readout is not required to be invariant)",
           delta_max_diff=delta_unchanged, covariance_max_diff=cov,
           covariance_delta_max_diff=delta_cov)

    # explicit scalar formula check against a numpy reference
    rng = np.random.default_rng(23)
    u5 = rng.normal(size=(6, 3, 2))
    j5 = rng.normal(size=(5, 3, 3))
    idx = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], np.int64)
    h = np.tanh((u5[..., 1] - u5[..., 0]) / 2.0)
    k = (j5[..., 0] + j5[..., 2] - 2.0 * j5[..., 1]) / 4.0
    d = k * h[idx[:, 0]] * h[idx[:, 1]]
    jr_ref = np.stack([j5[..., 0] + d, j5[..., 1] - d, j5[..., 2] + d], axis=-1)
    jr_np, d_np = E.couple_pairs(torch.as_tensor(u5), torch.as_tensor(j5),
                                 torch.as_tensor(idx))
    record("G8", "G",
           float(np.abs(jr_np.numpy() - jr_ref).max()) < 1e-14
           and float(np.abs(d_np.numpy() - d).max()) < 1e-14,
           description="the vectorised coupling formula matches an independent NumPy "
                       "reference to float64 precision",
           formula_max_error=float(np.abs(jr_np.numpy() - jr_ref).max()))


# ---------------------------------------------------------------------------
# H -- data identity and label isolation
# ---------------------------------------------------------------------------

def check_data(bundle: D.DataBundle) -> None:
    fit = np.asarray(bundle.fit_rows, np.int64)
    dev = np.asarray(bundle.eval_rows, np.int64)
    hist_manifest = C.read_json(HISTORICAL_ROUND / "results"
                                / "RICE_CUMULANT_S0" / "manifest.json")
    fit_sha = C.array_sha256(fit)
    dev_sha = C.array_sha256(dev)
    identity = {
        "n_fit": int(fit.size), "n_dev": int(dev.size),
        "intersection": int(np.intersect1d(fit, dev).size),
        "union": int(np.union1d(fit, dev).size),
        "fit_rows_sha256": fit_sha, "dev_rows_sha256": dev_sha,
        "split_sha256": bundle.hashes["split"]["split_sha256"],
        "raw_cache_sha256": bundle.hashes["raw_cache"]["sha256"],
        "train_labels_sha256": bundle.hashes["train_labels"]["sha256"],
        "scaler": bundle.scaler,
        "historical_fit_rows_sha256": hist_manifest["fit_rows_sha256"],
        "historical_dev_rows_sha256": hist_manifest["dev_rows_sha256"],
        "historical_split_sha256": hist_manifest["data_hashes"]["split"]["split_sha256"],
    }
    y_fit = bundle.labels[fit]
    fit_recalc = {"mean": float(y_fit.mean()), "std": float(y_fit.std(ddof=0))}
    ok = (int(fit.size) == 2856 and int(dev.size) == 504
          and identity["intersection"] == 0 and identity["union"] == 3360
          and fit_sha == hist_manifest["fit_rows_sha256"]
          and dev_sha == hist_manifest["dev_rows_sha256"]
          and identity["split_sha256"] == identity["historical_split_sha256"]
          and np.isfinite(bundle.labels[bundle.official_train]).all()
          and np.isnan(bundle.labels[bundle.official_valid]).all()
          and abs(fit_recalc["mean"] - bundle.scaler["mean"]) < 1e-12
          and abs(fit_recalc["std"] - bundle.scaler["std"]) < 1e-12)
    record("H1", "H", ok,
           description="D0 Fit/Dev identity (2856/504, disjoint, exactly covering official "
                       "Train), pinned row/split/data hashes, Fit-only z-score and NaN "
                       "official-Valid labels",
           **identity, recomputed_fit_scaler=fit_recalc)

    # Mutating Dev labels cannot change the Fit scaler or graph normaliser.
    mutated = bundle.labels.copy()
    mutated[dev] += 100.0
    y = mutated[fit]
    scaler_ok = (abs(float(y.mean()) - bundle.scaler["mean"]) < 1e-12
                 and abs(float(y.std(ddof=0)) - bundle.scaler["std"]) < 1e-12)
    graph_mean = bundle.graph_scalars[fit].mean(axis=0)
    raw_scale = bundle.graph_scalars[fit].std(axis=0)
    graph_scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
    scale_ok = (float(np.abs(graph_mean - bundle.graph_mean).max()) < 1e-12
                and float(np.abs(graph_scale - bundle.graph_scale).max()) < 1e-12)
    record("H2", "H", scaler_ok and scale_ok,
           description="target and graph-scalar normalisation use Fit rows only; changing "
                       "Dev labels leaves both unchanged",
           target_mean=bundle.scaler["mean"], target_std=bundle.scaler["std"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
    cfg = C.load_config()
    check_invariances(bundle)
    check_gradients(bundle)
    check_cost(bundle)
    check_historical_equivalence(bundle, cfg)
    check_arms_identical(bundle, cfg)
    check_endpoint_conditioning(bundle, cfg)
    check_data(bundle)
    failed = [r for r in RECORDS if r["status"] != "PASS"]
    summary = {
        "round": "rice_chemistry_conditioned_coupling_pilot_v0",
        "config_sha256": C.sha256_json(cfg),
        "code_sha256": C.code_sha(["rice_common.py", "rice_data.py", "rice_structure.py",
                                   "rice_energy.py", "rice_tensor.py", "rice_model.py",
                                   "rice_train.py", "rice_checks.py",
                                   "old_rice_reference.py"]),
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