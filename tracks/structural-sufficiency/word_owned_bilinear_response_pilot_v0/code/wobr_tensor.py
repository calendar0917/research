"""Batched tensor assembly for WOBR-v0.

All graph work (R2 sentences, candidate supports, induced bonds, boundary
counts, rooted 2-path vocabulary, deterministic scalars) is a pure CPU function
of one molecule and is precomputed once.  Collation concatenates molecule-local
arrays with integer offsets only; it never uses atom ids as features.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import wobr_structure as S


def _cat(arrays: Sequence[np.ndarray], dtype=np.int64, empty_width: int | None = None) -> np.ndarray:
    if arrays:
        return np.concatenate(arrays).astype(dtype, copy=False)
    if empty_width is None:
        return np.zeros(0, dtype=dtype)
    return np.zeros((0, empty_width), dtype=dtype)


def collate(rows: Sequence[int], structs: Mapping[int, S.MoleculeStruct],
            raw: Mapping[str, Any], device: Any = "cpu") -> dict[str, Any]:
    rows = [int(r) for r in rows]
    if not rows:
        raise ValueError("empty batch")
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_start = node_ptr[np.asarray(rows, np.int64)]
    bond_start = bond_ptr[np.asarray(rows, np.int64)]

    atom_feat_parts: list[np.ndarray] = []
    bond_feat_parts: list[np.ndarray] = []
    bond_u_parts: list[np.ndarray] = []
    bond_v_parts: list[np.ndarray] = []
    atom_scalar_parts: list[np.ndarray] = []
    atom_mol_parts: list[np.ndarray] = []
    bond_mol_parts: list[np.ndarray] = []
    graph_parts: list[np.ndarray] = []
    dist1_parts: list[np.ndarray] = []
    dist1_seg: list[np.ndarray] = []
    dist2_parts: list[np.ndarray] = []
    dist2_seg: list[np.ndarray] = []
    path_cat_parts: list[np.ndarray] = []
    path_end_type_parts: list[np.ndarray] = []
    path_closure_parts: list[np.ndarray] = []
    path_end_atom_parts: list[np.ndarray] = []
    path_seg: list[np.ndarray] = []
    r2_bond_parts: list[np.ndarray] = []
    r2_bucket_parts: list[np.ndarray] = []
    r2_seg: list[np.ndarray] = []
    r2_bucket_seg: list[np.ndarray] = []
    word_root_parts: list[np.ndarray] = []
    word_support_parts: list[np.ndarray] = []
    word_internal_parts: list[np.ndarray] = []
    word_scalar_parts: list[np.ndarray] = []
    word_size_parts: list[np.ndarray] = []
    word_internal_count_parts: list[np.ndarray] = []
    word_boundary_count_parts: list[np.ndarray] = []
    root_word_ptr_parts: list[np.ndarray] = []
    mol_atom_ptr = [0]
    mol_bond_ptr = [0]

    atom_off = 0
    bond_off = 0
    word_off = 0
    for m, row in enumerate(rows):
        st = structs[row]
        n = st.n_atoms
        m_bonds = st.n_bonds
        a0 = int(atom_start[m])
        b0 = int(bond_start[m])
        atom_feat_parts.append(np.asarray(raw["atom_feat"])[a0:a0 + n].astype(np.int64))
        bond_feat_parts.append(np.asarray(raw["bond_feat"])[b0:b0 + m_bonds].astype(np.int64))
        bond_u_parts.append(st.bond_u + atom_off)
        bond_v_parts.append(st.bond_v + atom_off)
        atom_scalar_parts.append(st.atom_scalars.astype(np.float32))
        atom_mol_parts.append(np.full(n, m, np.int64))
        bond_mol_parts.append(np.full(m_bonds, m, np.int64))
        graph_parts.append(st.graph_scalars.astype(np.float32))

        for v in range(n):
            lo, hi = int(st.dist1_ptr[v]), int(st.dist1_ptr[v + 1])
            dist1_parts.append(st.dist1_atoms[lo:hi] + atom_off)
            lo, hi = int(st.dist2_ptr[v]), int(st.dist2_ptr[v + 1])
            dist2_parts.append(st.dist2_atoms[lo:hi] + atom_off)
        g1 = np.repeat(np.arange(atom_off, atom_off + n, dtype=np.int64), np.diff(st.dist1_ptr))
        g2 = np.repeat(np.arange(atom_off, atom_off + n, dtype=np.int64), np.diff(st.dist2_ptr))
        dist1_seg.append(g1)
        dist2_seg.append(g2)

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
        bucket_seg = np.repeat(np.arange(atom_off, atom_off + n, dtype=np.int64) * 3,
                               np.diff(st.r2_bond_ptr)) + st.r2_bond_bucket
        r2_bucket_seg.append(bucket_seg)

        word_root_parts.append(st.word_root + atom_off)
        support = st.word_support.copy()
        valid = support >= 0
        support[valid] = support[valid] + atom_off
        word_support_parts.append(support)
        internal = st.word_internal.copy()
        valid_i = internal >= 0
        internal[valid_i] = internal[valid_i] + bond_off
        word_internal_parts.append(internal)
        word_scalar_parts.append(st.word_scalars.astype(np.float32))
        word_size_parts.append(st.word_size.astype(np.int8))
        word_internal_count_parts.append(st.word_internal_count.astype(np.int8))
        word_boundary_count_parts.append(st.word_boundary_count.astype(np.int16))
        root_word_ptr_parts.append(st.word_root_ptr[:-1] + word_off)

        atom_off += n
        bond_off += m_bonds
        word_off += st.n_words
        mol_atom_ptr.append(atom_off)
        mol_bond_ptr.append(bond_off)

    atom_feat = _cat(atom_feat_parts, empty_width=9)
    bond_feat = _cat(bond_feat_parts, empty_width=3)
    bond_u = _cat(bond_u_parts)
    bond_v = _cat(bond_v_parts)
    atom_scalars = _cat(atom_scalar_parts, dtype=np.float32, empty_width=5)
    atom_mol = _cat(atom_mol_parts)
    bond_mol = _cat(bond_mol_parts)
    graph_scalars = np.stack(graph_parts, axis=0).astype(np.float32) if graph_parts else np.zeros((0, 8), np.float32)
    dist1_atoms = _cat(dist1_parts)
    dist2_atoms = _cat(dist2_parts)
    dist1_seg = _cat(dist1_seg)
    dist2_seg = _cat(dist2_seg)
    path_cat = _cat(path_cat_parts)
    path_end_type = _cat(path_end_type_parts)
    path_closure = _cat(path_closure_parts)
    path_end_atom = _cat(path_end_atom_parts)
    path_seg = _cat(path_seg)
    r2_bond = _cat(r2_bond_parts)
    r2_bucket = _cat(r2_bucket_parts)
    r2_seg = _cat(r2_seg)
    r2_bucket_seg = _cat(r2_bucket_seg)
    word_root = _cat(word_root_parts)
    word_support = _cat(word_support_parts, empty_width=4)
    word_internal = _cat(word_internal_parts, empty_width=6)
    word_scalars = _cat(word_scalar_parts, dtype=np.float32, empty_width=25)
    word_size = _cat(word_size_parts, dtype=np.int8)
    word_internal_count = _cat(word_internal_count_parts, dtype=np.int8)
    word_boundary_count = _cat(word_boundary_count_parts, dtype=np.int16)
    root_word_ptr = _cat(root_word_ptr_parts)
    root_word_ptr = np.concatenate([root_word_ptr, np.asarray([word_off], np.int64)])

    # CSR atom -> incident physical bond ids, used only by contribution analysis.
    deg = np.zeros(atom_off, np.int64)
    if bond_u.size:
        deg += np.bincount(bond_u, minlength=atom_off)
        deg += np.bincount(bond_v, minlength=atom_off)
    atom_bond_ptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
    atom_bond_ids = np.zeros(int(atom_bond_ptr[-1]), np.int64)
    if bond_u.size:
        pos = atom_bond_ptr[:-1].copy()
        for e in range(bond_u.size):
            u, v = int(bond_u[e]), int(bond_v[e])
            atom_bond_ids[pos[u]] = e
            pos[u] += 1
            atom_bond_ids[pos[v]] = e
            pos[v] += 1

    bond_index = np.stack([bond_u, bond_v], axis=1) if bond_u.size else np.zeros((0, 2), np.int64)
    word_mol = atom_mol[word_root] if word_root.size else np.zeros(0, np.int64)
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
        "dist1_seg": torch.as_tensor(dist1_seg, dtype=torch.long),
        "dist2_seg": torch.as_tensor(dist2_seg, dtype=torch.long),
        "path_cat": torch.as_tensor(path_cat, dtype=torch.long),
        "path_end_type": torch.as_tensor(path_end_type, dtype=torch.long),
        "path_closure": torch.as_tensor(path_closure, dtype=torch.long),
        "path_end_atom": torch.as_tensor(path_end_atom, dtype=torch.long),
        "path_seg": torch.as_tensor(path_seg, dtype=torch.long),
        "r2_bond": torch.as_tensor(r2_bond, dtype=torch.long),
        "r2_bucket": torch.as_tensor(r2_bucket, dtype=torch.long),
        "r2_seg": torch.as_tensor(r2_seg, dtype=torch.long),
        "r2_bucket_seg": torch.as_tensor(r2_bucket_seg, dtype=torch.long),
        "word_root": torch.as_tensor(word_root, dtype=torch.long),
        "word_support": torch.as_tensor(word_support, dtype=torch.long),
        "word_internal": torch.as_tensor(word_internal, dtype=torch.long),
        "word_scalars": torch.as_tensor(word_scalars, dtype=torch.float32),
        "word_size": torch.as_tensor(word_size, dtype=torch.long),
        "word_internal_count": torch.as_tensor(word_internal_count, dtype=torch.long),
        "word_boundary_count": torch.as_tensor(word_boundary_count, dtype=torch.long),
        "word_mol": torch.as_tensor(word_mol, dtype=torch.long),
        "root_word_ptr": torch.as_tensor(root_word_ptr, dtype=torch.long),
        "atom_bond_ptr": torch.as_tensor(atom_bond_ptr, dtype=torch.long),
        "atom_bond_ids": torch.as_tensor(atom_bond_ids, dtype=torch.long),
        "n_molecules": int(len(rows)),
        "n_atoms": int(atom_feat.shape[0]),
        "n_bonds": int(bond_feat.shape[0]),
        "n_words": int(word_root.shape[0]),
        "rows": np.asarray(rows, np.int64),
        "mol_atom_ptr": torch.as_tensor(np.asarray(mol_atom_ptr), dtype=torch.long),
        "mol_bond_ptr": torch.as_tensor(np.asarray(mol_bond_ptr), dtype=torch.long),
    }
    if device != "cpu":
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out