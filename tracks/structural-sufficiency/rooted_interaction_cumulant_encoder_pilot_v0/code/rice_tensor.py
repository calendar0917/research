"""Batched tensor assembly for RICE-v0.

Every graph-theoretic quantity (radius-2 contexts, connected root-conditioned
supports, internal induced bonds, root-containing proper subsets) is a pure CPU
function of one molecule and is precomputed once per process.  Collation only
concatenates molecule-local arrays and adds integer offsets; atom ids are never
features.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import rice_structure as S

SIZES = (2, 3, 4)


def _cat(arrays: Sequence[np.ndarray], dtype=np.int64,
         empty_width: int | None = None) -> np.ndarray:
    if arrays:
        return np.concatenate(arrays).astype(dtype, copy=False)
    if empty_width is None:
        return np.zeros(0, dtype=dtype)
    return np.zeros((0, empty_width), dtype=dtype)


def collate(rows: Sequence[int], structs: Mapping[int, S.MoleculeStruct],
            raw: Mapping[str, Any] | None = None,
            device: Any = "cpu") -> dict[str, Any]:
    rows = [int(r) for r in rows]
    if not rows:
        raise ValueError("empty batch")
    atom_feat_parts: list[np.ndarray] = []
    bond_feat_parts: list[np.ndarray] = []
    bond_u_parts: list[np.ndarray] = []
    bond_v_parts: list[np.ndarray] = []
    graph_parts: list[np.ndarray] = []
    atom_mol_parts: list[np.ndarray] = []
    support = {size: [] for size in SIZES}
    bond_ids = {size: [] for size in SIZES}
    slots = {size: [] for size in SIZES}
    sub2_3: list[np.ndarray] = []
    sub2_4: list[np.ndarray] = []
    sub3_4: list[np.ndarray] = []
    mol_atom_ptr = [0]

    counts = {size: [int(structs[r].atoms[size].shape[0]) for r in rows] for size in SIZES}
    off_atom = 0
    off_bond = 0
    off = {size: 0 for size in SIZES}
    for m, row in enumerate(rows):
        st = structs[row]
        atom_feat_parts.append(st.atom_feat)
        bond_feat_parts.append(st.bond_feat)
        bond_u_parts.append(st.bond_u + off_atom)
        bond_v_parts.append(st.bond_v + off_atom)
        graph_parts.append(np.asarray(st.graph_scalars, np.float32))
        atom_mol_parts.append(np.full(st.n_atoms, m, np.int64))
        for size in SIZES:
            a = st.atoms[size].astype(np.int64)
            ok = a >= 0
            a = np.where(ok, a + off_atom, -1)
            support[size].append(a)
            b = st.bonds[size].astype(np.int64)
            bond_ids[size].append(np.where(b >= 0, b + off_bond, -1))
            slots[size].append(st.bslot[size].astype(np.int64))
        s2 = st.sub2[3].astype(np.int64)
        sub2_3.append(np.where(s2 >= 0, s2 + off[2], -1))
        s2 = st.sub2[4].astype(np.int64)
        sub2_4.append(np.where(s2 >= 0, s2 + off[2], -1))
        s3 = st.sub3[4].astype(np.int64)
        sub3_4.append(np.where(s3 >= 0, s3 + off[3], -1))
        off_atom += st.n_atoms
        off_bond += st.n_bonds
        for size in SIZES:
            off[size] += counts[size][m]
        mol_atom_ptr.append(off_atom)

    atom_feat = _cat(atom_feat_parts, empty_width=9)
    bond_feat = _cat(bond_feat_parts, empty_width=3)
    bond_u = _cat(bond_u_parts)
    bond_v = _cat(bond_v_parts)
    bond_index = (np.stack([bond_u, bond_v], axis=1) if bond_u.size
                  else np.zeros((0, 2), np.int64))
    out: dict[str, Any] = {
        "atom_feat": torch.as_tensor(atom_feat, dtype=torch.long),
        "bond_feat": torch.as_tensor(bond_feat, dtype=torch.long),
        "bond_index": torch.as_tensor(bond_index, dtype=torch.long),
        "graph_scalars": torch.as_tensor(
            np.stack(graph_parts, axis=0) if graph_parts else np.zeros((0, 8), np.float32),
            dtype=torch.float32),
        "atom_mol": torch.as_tensor(_cat(atom_mol_parts), dtype=torch.long),
        "mol_atom_ptr": torch.as_tensor(np.asarray(mol_atom_ptr, np.int64),
                                        dtype=torch.long),
        "n_molecules": int(len(rows)),
        "n_atoms": int(atom_feat.shape[0]),
        "n_bonds": int(bond_feat.shape[0]),
        "rows": np.asarray(rows, np.int64),
    }
    for size in SIZES:
        out[f"atoms_{size}"] = torch.as_tensor(
            _cat(support[size], empty_width=4), dtype=torch.long)
        out[f"bonds_{size}"] = torch.as_tensor(
            _cat(bond_ids[size], empty_width=6), dtype=torch.long)
        out[f"bslot_{size}"] = torch.as_tensor(
            _cat(slots[size], empty_width=6).reshape(-1, 6, 2)
            if slots[size] else np.zeros((0, 6, 2), np.int64), dtype=torch.long)
        out[f"n_words_{size}"] = int(out[f"atoms_{size}"].shape[0])
    out["sub2_3"] = torch.as_tensor(_cat(sub2_3, empty_width=2).reshape(-1, 2)
                                    if sub2_3 else np.zeros((0, 2), np.int64),
                                    dtype=torch.long)
    out["sub2_4"] = torch.as_tensor(_cat(sub2_4, empty_width=3).reshape(-1, 3)
                                    if sub2_4 else np.zeros((0, 3), np.int64),
                                    dtype=torch.long)
    out["sub3_4"] = torch.as_tensor(_cat(sub3_4, empty_width=3).reshape(-1, 3)
                                    if sub3_4 else np.zeros((0, 3), np.int64),
                                    dtype=torch.long)
    out["n_words"] = int(sum(out[f"n_words_{s}"] for s in SIZES))
    if device != "cpu":
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out