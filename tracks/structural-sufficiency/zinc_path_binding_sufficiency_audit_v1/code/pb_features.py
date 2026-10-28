"""Frozen-encoder feature extraction and the T_real / T_null / D statistics.

Encodes the official train/valid caches with a frozen LCRC-A checkpoint and
returns, per atom, every quantity the audit needs:

* ``a_v`` (96), ``b_e`` (96), ``h_v`` (64) and ``X_v`` (529, the exact input of
  ``LocalContent.mlp``);
* ``S_v`` = ``X_v[:, 96:]`` (433) = existing S1/S2 statistics + local scalars;
* ``T_real``, ``T_null``, ``D`` and the within-root permutation control, for the
  learned content and for the train-only type-conditional content control.

Nothing here touches a label; see ``pb_common.load_cache``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

import pb_common as P
import pb_paths as PP
import lc_data as LD


# ---------------------------------------------------------------------------
# frozen encoder pass
# ---------------------------------------------------------------------------

def local_parts(model, batch: Mapping[str, torch.Tensor], a, b, scal):
    """Replicate ``LocalContent.forward`` while exposing every component."""
    lc = model.local
    n_atoms = a.shape[0]
    width = a.shape[1]
    bond_index = batch["bond_index"]
    u, v = bond_index[:, 0], bond_index[:, 1]
    degree = a.new_zeros(n_atoms)
    degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
    degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
    b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
    b_mean = b_sum / degree.clamp(min=1.0).unsqueeze(1)
    nbr = a.new_zeros((n_atoms, lc.rank)).index_add_(
        0, batch["nbr_atom"], lc.nbr_emb(batch["nbr_code"]))
    path_joint = lc.joint_proj(lc.path_u(batch["path_cat"]) * lc.path_v(batch["path_end63"]))
    path_emb = torch.cat([lc.path_emb(batch["path_cat"]), lc.end_emb(batch["path_end63"]),
                          path_joint, lc.close_emb(batch["path_closure"])], dim=1)
    path_agg = a.new_zeros((n_atoms, path_emb.shape[1])).index_add_(
        0, batch["path_atom"], path_emb)
    gate = lc.path_gate(batch["path_cat"])
    content = a[batch["path_end"]] * gate
    content_agg = a.new_zeros((n_atoms, width)).index_add_(0, batch["path_atom"], content)
    x = torch.cat([a, b_sum, b_mean, nbr, path_agg, content_agg, scal], dim=1)
    return {"b_sum": b_sum, "b_mean": b_mean, "nbr": nbr, "path_agg": path_agg,
            "content_agg": content_agg, "x": x, "degree": degree}


@torch.no_grad()
def encode(model, cache: Mapping[str, Any], *, batch_size: int = 256,
           check_exact: bool = True) -> dict[str, np.ndarray]:
    """Encode every atom of a split with one frozen LCRC-A checkpoint."""
    n_mols = int(cache["n_molecules"])
    n_atoms = int(np.asarray(cache["atom_ptr"])[-1])
    n_bonds = int(np.asarray(cache["bond_ptr"])[-1])
    a_parts, b_parts, h_parts, x_parts = [], [], [], []
    max_h_diff = 0.0
    for start in range(0, n_mols, batch_size):
        rows = np.arange(start, min(start + batch_size, n_mols), dtype=np.int64)
        batch = LD.collate(cache, rows)
        a = model.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bf = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        b = model.bond_encoder(bf)
        scal = (batch["atom_scalar"] - model.scalar_mean) / model.scalar_scale
        parts = local_parts(model, batch, a, b, scal)
        h = model.local(batch, a, b, scal)
        if check_exact:
            ref = model.local.norm(model.local.mlp(parts["x"]))
            max_h_diff = max(max_h_diff, float((ref - h).abs().max()))
        a_parts.append(a.numpy())
        b_parts.append(b.numpy())
        h_parts.append(h.numpy())
        x_parts.append(parts["x"].numpy())
    a_all = np.concatenate(a_parts).astype(np.float32)
    b_all = np.concatenate(b_parts).astype(np.float32)
    h_all = np.concatenate(h_parts).astype(np.float32)
    x_all = np.concatenate(x_parts).astype(np.float32)
    if a_all.shape != (n_atoms, P.N_ATOM_OUT):
        raise RuntimeError(f"atom encode shape {a_all.shape} != {(n_atoms, P.N_ATOM_OUT)}")
    if b_all.shape[0] != n_bonds:
        raise RuntimeError(f"bond encode shape {b_all.shape[0]} != {n_bonds}")
    if x_all.shape[1] != P.X_DIM:
        raise RuntimeError(f"X shape {x_all.shape} != (*, {P.X_DIM})")
    if check_exact and max_h_diff > 1e-5:
        raise RuntimeError(f"X_v does not reproduce h_v exactly: max diff {max_h_diff}")
    return {"a": a_all, "b": b_all, "h": h_all, "x": x_all,
            "s": x_all[:, P.N_ATOM_OUT:].copy(),
            "h_vs_x_max_abs_diff": float(max_h_diff),
            "n_atoms": int(n_atoms), "n_bonds": int(n_bonds)}


# ---------------------------------------------------------------------------
# train-only type-conditional content control
# ---------------------------------------------------------------------------

def bond_type_key(cache: Mapping[str, Any]) -> np.ndarray:
    """Unordered (type(u), type(v), bond_type) code for every bond."""
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    ends = PP.bond_endpoints(cache)
    tu, tv = atom_type[ends[:, 0]], atom_type[ends[:, 1]]
    lo, hi = np.minimum(tu, tv), np.maximum(tu, tv)
    return (lo * P.L.N_ATOM_TYPES + hi) * P.L.N_BOND_TYPES + (bond_type - 1)


def bond_molecule_index(cache: Mapping[str, Any]) -> np.ndarray:
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    return np.repeat(np.arange(len(bp) - 1), np.diff(bp))


def atom_molecule_index(cache: Mapping[str, Any]) -> np.ndarray:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    return np.repeat(np.arange(len(ap) - 1), np.diff(ap))


def type_content_tables(a: np.ndarray, b: np.ndarray, cache: Mapping[str, Any],
                        train_rows: np.ndarray) -> dict:
    """Train-only means of ``a`` per atom type and of ``b`` per bond type triple."""
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    key = bond_type_key(cache)
    atom_mol = atom_molecule_index(cache)
    bond_mol = bond_molecule_index(cache)
    atom_mask = np.isin(atom_mol, train_rows)
    bond_mask = np.isin(bond_mol, train_rows)
    d = a.shape[1]
    n_key = P.L.N_ATOM_TYPES * P.L.N_ATOM_TYPES * P.L.N_BOND_TYPES

    atom_sum = np.zeros((P.L.N_ATOM_TYPES, d), dtype=np.float64)
    np.add.at(atom_sum, atom_type[atom_mask], a[atom_mask].astype(np.float64))
    atom_cnt = np.bincount(atom_type[atom_mask], minlength=P.L.N_ATOM_TYPES)
    atom_mean = atom_sum / np.maximum(atom_cnt, 1)[:, None]

    bond_sum = np.zeros((n_key, d), dtype=np.float64)
    np.add.at(bond_sum, key[bond_mask], b[bond_mask].astype(np.float64))
    bond_cnt = np.bincount(key[bond_mask], minlength=n_key)
    global_mean = b[bond_mask].astype(np.float64).mean(0)
    bond_mean = bond_sum / np.maximum(bond_cnt, 1)[:, None]
    missing = bond_cnt == 0
    bond_mean[missing] = global_mean
    return {
        "atom_mean": atom_mean,
        "atom_cnt": atom_cnt,
        "bond_mean": bond_mean,
        "bond_cnt": bond_cnt,
        "global_bond_mean": global_mean,
        "n_missing_bond_keys": int(missing.sum()),
        "n_bond_keys": int(n_key),
        "bond_key": key,
        "atom_type": atom_type,
    }


def type_content_arrays(tables: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    a_type = tables["atom_mean"][tables["atom_type"]]
    b_type = tables["bond_mean"][tables["bond_key"]]
    return a_type, b_type


# ---------------------------------------------------------------------------
# projections and the path statistics
# ---------------------------------------------------------------------------

def path_factors(a: np.ndarray, b: np.ndarray, paths: Mapping[str, np.ndarray],
                 proj: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Q_p = W_Q[a_root; b_e1]``, ``K_p = W_K[a_mid; b_e2]``, ``Z_p = W_V a_end``."""
    wq, wk, wv = proj["W_Q"], proj["W_K"], proj["W_V"]
    wqa, wqb = wq[:, :P.N_ATOM_OUT], wq[:, P.N_ATOM_OUT:]
    wka, wkb = wk[:, :P.N_ATOM_OUT], wk[:, P.N_ATOM_OUT:]
    root, mid, end, e1, e2 = paths["root"], paths["mid"], paths["end"], paths["e1"], paths["e2"]
    q = a[root].astype(np.float64) @ wqa.T + b[e1].astype(np.float64) @ wqb.T
    k = a[mid].astype(np.float64) @ wka.T + b[e2].astype(np.float64) @ wkb.T
    z = a[end].astype(np.float64) @ wv.T
    return q, k, z


def _within_root_permuted(values: np.ndarray, root: np.ndarray, rng) -> np.ndarray:
    """Re-order ``values`` inside each root (same root sequence, random pairing)."""
    order = np.lexsort((rng.random(len(root)), root))
    return values[order]


def t_stats(paths: Mapping[str, np.ndarray], q: np.ndarray, k: np.ndarray,
            z: np.ndarray, n_atoms: int, *, rng=None) -> dict[str, np.ndarray]:
    """T_real, T_null = sqrt(n) q̄⊙k̄⊙z̄, D = T_real − T_null (+ permutation null)."""
    root = paths["root"]
    n = np.bincount(root, minlength=int(n_atoms)).astype(np.float64)
    inv_sqrt = np.where(n > 0, 1.0 / np.sqrt(np.maximum(n, 1.0)), 0.0)
    one_over_n = np.where(n > 0, 1.0 / np.maximum(n, 1.0), 0.0)
    t_real = P.seg_sum(q * k * z, root, int(n_atoms)) * inv_sqrt[:, None]
    qm = P.seg_sum(q, root, int(n_atoms)) * one_over_n[:, None]
    km = P.seg_sum(k, root, int(n_atoms)) * one_over_n[:, None]
    zm = P.seg_sum(z, root, int(n_atoms)) * one_over_n[:, None]
    t_null = np.sqrt(n)[:, None] * (qm * km * zm)
    out = {"T_real": t_real, "T_null": t_null, "D": t_real - t_null, "n_paths": n}
    if rng is not None:
        qp = _within_root_permuted(q, root, rng)
        kp = _within_root_permuted(k, root, rng)
        zp = _within_root_permuted(z, root, rng)
        t_perm = P.seg_sum(qp * kp * zp, root, int(n_atoms)) * inv_sqrt[:, None]
        out["T_perm"] = t_perm
        out["D_perm"] = t_real - t_perm
    return out


def t_swap_within_cell(paths: Mapping[str, np.ndarray], cache: Mapping[str, Any],
                       q: np.ndarray, k: np.ndarray, z: np.ndarray, n_atoms: int,
                       *, seed: int) -> dict:
    """Swap ``(K,Z)`` between paths of the **same** ``(cat189, end63, closure)`` cell.

    ``content_agg = sum_p gate[cat_p] ⊗ a_{u_p}`` only sees the *sum* of the
    endpoint content inside a cell, so this swap leaves every component of ``X_v``
    (and therefore ``h_v``) bit-identical while changing the triple product.
    The gap ``U = T_real - T_swap`` is therefore the within-cell pairing mass that
    LCRC-A's local input cannot represent, whatever readout is put on top.
    """
    root = paths["root"]
    cat = np.asarray(cache["path_cat"], dtype=np.int64)
    end63 = np.asarray(cache["path_end63"], dtype=np.int64)
    closure = np.asarray(cache["path_closure"], dtype=np.int64)
    cell = ((root.astype(np.int64) * 189 + cat) * 63 + end63) * 2 + closure
    rng = np.random.default_rng(int(seed))
    order = np.lexsort((rng.random(len(root)), cell))
    kp, zp = k[order], z[order]
    n = np.bincount(root, minlength=int(n_atoms)).astype(np.float64)
    inv_sqrt = np.where(n > 0, 1.0 / np.sqrt(np.maximum(n, 1.0)), 0.0)
    t_swap = P.seg_sum(q * kp * zp, root, int(n_atoms)) * inv_sqrt[:, None]
    # cell multiplicity statistics (multiplicity >= 2 is where the swap can act)
    uniq, counts = np.unique(cell, return_counts=True)
    per_root_multi = np.zeros(int(n_atoms), dtype=np.int64)
    root_of_cell = uniq // (189 * 63 * 2)
    multi_cells = counts >= 2
    np.add.at(per_root_multi, root_of_cell[multi_cells], counts[multi_cells])
    return {"T_swap": t_swap, "cell_multiplicity": per_root_multi,
            "n_cells": int(len(uniq)), "n_multi_cells": int(multi_cells.sum()),
            "max_cell_multiplicity": int(counts.max()) if len(counts) else 0}


def flat_t_reference(a: np.ndarray, b: np.ndarray, paths: Mapping[str, np.ndarray],
                     proj: Mapping[str, np.ndarray], n_atoms: int) -> np.ndarray:
    """Independent, per-path implementation of the flat sum (test reference)."""
    wq, wk, wv = proj["W_Q"], proj["W_K"], proj["W_V"]
    root, mid, end = paths["root"], paths["mid"], paths["end"]
    e1, e2 = paths["e1"], paths["e2"]
    n = np.bincount(root, minlength=int(n_atoms)).astype(np.float64)
    out = np.zeros((int(n_atoms), wq.shape[0]), dtype=np.float64)
    for p in range(len(root)):
        q = wq @ np.concatenate([a[root[p]], b[e1[p]]]).astype(np.float64)
        k = wk @ np.concatenate([a[mid[p]], b[e2[p]]]).astype(np.float64)
        z = wv @ a[end[p]].astype(np.float64)
        out[root[p]] += q * k * z
    denom = np.where(n > 0, np.sqrt(np.maximum(n, 1.0)), 1.0)
    return out / denom[:, None]


def factorized_t_reference(a: np.ndarray, b: np.ndarray, paths: Mapping[str, np.ndarray],
                           proj: Mapping[str, np.ndarray], n_atoms: int) -> np.ndarray:
    """Branch-first reference: ``sum_w Q(v,w) ⊙ (sum_u K(w,u) ⊙ Z(u))``.

    Written with an explicit outer loop over ``root`` and an inner loop over the
    root's first hops, i.e. it never materialises a per-path triple product and
    never sums over ``u`` for a fixed (v,w) in a single flattened expression.
    """
    wq, wk, wv = proj["W_Q"], proj["W_K"], proj["W_V"]
    root, mid, end = paths["root"], paths["mid"], paths["end"]
    e1, e2 = paths["e1"], paths["e2"]
    n = np.bincount(root, minlength=int(n_atoms)).astype(np.float64)
    branches: dict[tuple[int, int], list[np.ndarray]] = {}
    for p in range(len(root)):
        key = (int(root[p]), int(mid[p]))
        q = wq @ np.concatenate([a[root[p]], b[e1[p]]]).astype(np.float64)
        k = wk @ np.concatenate([a[mid[p]], b[e2[p]]]).astype(np.float64)
        z = wv @ a[end[p]].astype(np.float64)
        if key not in branches:
            branches[key] = [q, np.zeros(wk.shape[0], dtype=np.float64)]
        branches[key][1] = branches[key][1] + k * z
    out = np.zeros((int(n_atoms), wq.shape[0]), dtype=np.float64)
    for (v, _w), (q, acc) in branches.items():
        out[v] = out[v] + q * acc
    denom = np.where(n > 0, np.sqrt(np.maximum(n, 1.0)), 1.0)
    return out / denom[:, None]


__all__ = [
    "local_parts", "encode", "bond_type_key", "bond_molecule_index",
    "atom_molecule_index", "type_content_tables", "type_content_arrays",
    "path_factors", "t_stats", "t_swap_within_cell", "flat_t_reference", "factorized_t_reference",
]
