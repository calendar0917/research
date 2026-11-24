"""Batched tensor assembly for SES-v0.

Every struct is an independent attributed graph.  Collation concatenates its
arrays with integer offsets only; there are no cross-graph reductions here.
The batch key set is exactly the one consumed by the frozen ``GwicModel`` with
``use_words=False``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

import ses_structure as S


def _cat(arrays: Sequence[np.ndarray], dtype=np.int64, width: int | None = None) -> np.ndarray:
    if arrays:
        return np.concatenate(arrays).astype(dtype, copy=False)
    return np.zeros((0, width), dtype=dtype) if width else np.zeros(0, dtype=dtype)


def collate(structs: Sequence[S.Struct], device: Any = "cpu") -> dict[str, Any]:
    if not structs:
        raise ValueError("empty batch")
    atom_feat_parts = []
    bond_feat_parts = []
    bond_u_parts = []
    bond_v_parts = []
    atom_scalar_parts = []
    atom_mol_parts = []
    bond_mol_parts = []
    graph_parts = []
    dist1_parts = []
    dist1_seg = []
    dist2_parts = []
    dist2_seg = []
    path_cat_parts = []
    path_end_type_parts = []
    path_closure_parts = []
    path_end_atom_parts = []
    path_seg = []
    r2_bond_parts = []
    r2_bucket_parts = []
    r2_seg = []
    r2_bucket_seg = []
    mol_atom_ptr = [0]

    atom_off = 0
    bond_off = 0
    for m, st in enumerate(structs):
        n = st.n_atoms
        nb = st.n_bonds
        atom_feat_parts.append(st.atom_feat)
        bond_feat_parts.append(st.bond_feat)
        bond_u_parts.append(st.bond_u + atom_off)
        bond_v_parts.append(st.bond_v + atom_off)
        atom_scalar_parts.append(st.atom_scalars.astype(np.float32))
        atom_mol_parts.append(np.full(n, m, np.int64))
        bond_mol_parts.append(np.full(nb, m, np.int64))
        graph_parts.append(st.graph_scalars.astype(np.float32))

        for v in range(n):
            lo, hi = int(st.dist1_ptr[v]), int(st.dist1_ptr[v + 1])
            dist1_parts.append(st.dist1_atoms[lo:hi] + atom_off)
            lo, hi = int(st.dist2_ptr[v]), int(st.dist2_ptr[v + 1])
            dist2_parts.append(st.dist2_atoms[lo:hi] + atom_off)
        dist1_seg.append(np.repeat(np.arange(atom_off, atom_off + n, dtype=np.int64),
                                   np.diff(st.dist1_ptr)))
        dist2_seg.append(np.repeat(np.arange(atom_off, atom_off + n, dtype=np.int64),
                                   np.diff(st.dist2_ptr)))

        path_cat_parts.append(st.path_cat)
        path_end_type_parts.append(st.path_end_type)
        path_closure_parts.append(st.path_closure)
        path_end_atom_parts.append(st.path_end_atom + atom_off)
        path_seg.append(np.repeat(np.arange(atom_off, atom_off + n, dtype=np.int64),
                                  np.diff(st.path_ptr)))

        r2_bond_parts.append(st.r2_bond + bond_off)
        r2_bucket_parts.append(st.r2_bond_bucket)
        r2_seg.append(np.repeat(np.arange(atom_off, atom_off + n, dtype=np.int64),
                                np.diff(st.r2_bond_ptr)))
        r2_bucket_seg.append(
            np.repeat(np.arange(atom_off, atom_off + n, dtype=np.int64) * 3,
                      np.diff(st.r2_bond_ptr)) + st.r2_bond_bucket)

        atom_off += n
        bond_off += nb
        mol_atom_ptr.append(atom_off)

    atom_feat = _cat(atom_feat_parts, width=9)
    bond_feat = _cat(bond_feat_parts, width=3)
    bond_u = _cat(bond_u_parts)
    bond_v = _cat(bond_v_parts)
    atom_scalars = _cat(atom_scalar_parts, dtype=np.float32, width=5)
    atom_mol = _cat(atom_mol_parts)
    bond_mol = _cat(bond_mol_parts)
    graph_scalars = np.stack(graph_parts, axis=0).astype(np.float32)
    dist1_atoms = _cat(dist1_parts)
    dist2_atoms = _cat(dist2_parts)
    path_cat = _cat(path_cat_parts)
    path_end_type = _cat(path_end_type_parts)
    path_closure = _cat(path_closure_parts)
    path_end_atom = _cat(path_end_atom_parts)
    r2_bond = _cat(r2_bond_parts)
    r2_bucket = _cat(r2_bucket_parts)
    bond_index = np.stack([bond_u, bond_v], axis=1) if bond_u.size else np.zeros((0, 2), np.int64)

    out: dict[str, Any] = {
        "atom_feat": torch.as_tensor(atom_feat, dtype=torch.long),
        "bond_feat": torch.as_tensor(bond_feat, dtype=torch.long),
        "bond_index": torch.as_tensor(bond_index, dtype=torch.long),
        "atom_scalars": torch.as_tensor(atom_scalars, dtype=torch.float32),
        "atom_mol": torch.as_tensor(atom_mol, dtype=torch.long),
        "bond_mol": torch.as_tensor(bond_mol, dtype=torch.long),
        "graph_scalars": torch.as_tensor(graph_scalars, dtype=torch.float32),
        "dist1_atoms": torch.as_tensor(dist1_atoms, dtype=torch.long),
        "dist2_atoms": torch.as_tensor(dist2_atoms, dtype=torch.long),
        "dist1_seg": torch.as_tensor(_cat(dist1_seg), dtype=torch.long),
        "dist2_seg": torch.as_tensor(_cat(dist2_seg), dtype=torch.long),
        "path_cat": torch.as_tensor(path_cat, dtype=torch.long),
        "path_end_type": torch.as_tensor(path_end_type, dtype=torch.long),
        "path_closure": torch.as_tensor(path_closure, dtype=torch.long),
        "path_end_atom": torch.as_tensor(path_end_atom, dtype=torch.long),
        "path_seg": torch.as_tensor(_cat(path_seg), dtype=torch.long),
        "r2_bond": torch.as_tensor(r2_bond, dtype=torch.long),
        "r2_bucket": torch.as_tensor(r2_bucket, dtype=torch.long),
        "r2_seg": torch.as_tensor(_cat(r2_seg), dtype=torch.long),
        "r2_bucket_seg": torch.as_tensor(_cat(r2_bucket_seg), dtype=torch.long),
        "n_molecules": int(len(structs)),
        "n_atoms": int(atom_feat.shape[0]),
        "n_bonds": int(bond_feat.shape[0]),
        "mol_atom_ptr": torch.as_tensor(np.asarray(mol_atom_ptr), dtype=torch.long),
    }
    if device != "cpu":
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out
