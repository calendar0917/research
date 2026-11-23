"""Rooted supports + the exact free-response type registry.

Two things live here, both pure structure functions:

1. ``build_structures`` builds the RICE-v0 rooted radius-2 support objects for
   any rows (the code is imported, not reimplemented).
2. ``build_registry`` assigns every singleton atom and every rooted connected
   support of size 2..4 to a **global type id** using RICE-v0's exact canonical
   certificate of the full 9/3 attributed root-marked induced subgraph.  The
   canonical certificate is decoded back into the four atom slots / six bond
   slots the free-response encoder consumes.

Nothing here reads a label.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import sr_common as C
import sr_data as D

import rice_structure as RS


def build_structures(bundle: Any, rows: Sequence[int]) -> dict[int, RS.MoleculeStruct]:
    return {int(r): RS.build_molecule(D.mol_arrays(bundle, int(r))) for r in rows}


def structure_stats(structs: Mapping[int, RS.MoleculeStruct]) -> dict[str, Any]:
    return RS.structure_stats(structs)


@dataclass
class TypeRegistry:
    atom_fields: np.ndarray      # [T,4,9] int16, 0 where absent
    bond_fields: np.ndarray      # [T,6,3] int16, 0 where absent
    atom_mask: np.ndarray        # [T,4] float32
    edge_mask: np.ndarray        # [T,6] float32
    k: np.ndarray                # [T] int8
    n_by_size: dict[int, int]
    offsets: dict[int, int]
    device: Any = None
    _torch: Any = None

    def n_types(self) -> int:
        return int(self.k.size)

    def torch_tensors(self, device: Any) -> dict[str, Any]:
        import torch

        if self._torch is None or self.device != device:
            self._torch = {
                "atom_fields": torch.as_tensor(self.atom_fields.astype(np.int64), device=device),
                "bond_fields": torch.as_tensor(self.bond_fields.astype(np.int64), device=device),
                "atom_mask": torch.as_tensor(self.atom_mask, device=device),
                "edge_mask": torch.as_tensor(self.edge_mask, device=device),
                "k": torch.as_tensor(self.k.astype(np.int64), device=device),
            }
            self.device = device
        return self._torch


def _unmix(values: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Mixed-radix decode, vectorised over rows.  ``values`` [n, len(dims)]."""
    n = int(values.shape[0])
    out = np.zeros((n, len(dims)), np.int16)
    work = np.asarray(values, np.int64).copy()
    for i in range(len(dims) - 1, -1, -1):
        out[:, i] = np.mod(work, int(dims[i]))
        work //= int(dims[i])
    return out


def _signatures(structs: Sequence[RS.MoleculeStruct], size: int,
                atom_dims: Sequence[int], bond_dims: Sequence[int]) -> list[np.ndarray]:
    if size == 1:
        out = []
        for st in structs:
            label = RS._mixed_radix(st.atom_feat, atom_dims)  # noqa: SLF001 (exact reuse)
            sig = np.full((st.n_atoms, 10), -1, np.int64)
            sig[:, 0] = label
            out.append(sig)
        return out
    return [RS.canonical_signature(st, size, atom_dims, bond_dims) for st in structs]


def build_registry(structs: Sequence[RS.MoleculeStruct],
                   atom_dims: Sequence[int], bond_dims: Sequence[int]) -> TypeRegistry:
    structs = list(structs)
    n_types_total = 0
    offsets: dict[int, int] = {}
    n_by_size: dict[int, int] = {}
    per_size_ids: dict[int, list[np.ndarray]] = {}
    uniq_by_size: dict[int, np.ndarray] = {}
    for size in (1, 2, 3, 4):
        sigs = _signatures(structs, size, atom_dims, bond_dims)
        cat = np.concatenate(sigs, axis=0) if sigs else np.zeros((0, 10), np.int64)
        if cat.shape[0]:
            uniq, inv = np.unique(cat, axis=0, return_inverse=True)
        else:  # pragma: no cover - only for empty inputs
            uniq = np.zeros((0, 10), np.int64)
            inv = np.zeros(0, np.int64)
        offsets[size] = n_types_total
        n_by_size[size] = int(uniq.shape[0])
        uniq_by_size[size] = uniq
        ids: list[np.ndarray] = []
        start = 0
        for st in structs:
            n = st.n_atoms if size == 1 else int(st.atoms[size].shape[0])
            ids.append((inv[start:start + n] + n_types_total).astype(np.int32))
            start += n
        per_size_ids[size] = ids
        n_types_total += int(uniq.shape[0])

    atom_fields = np.zeros((n_types_total, 4, 9), np.int16)
    bond_fields = np.zeros((n_types_total, 6, 3), np.int16)
    atom_mask = np.zeros((n_types_total, 4), np.float32)
    edge_mask = np.zeros((n_types_total, 6), np.float32)
    kinds = np.zeros(n_types_total, np.int8)
    for size in (1, 2, 3, 4):
        uniq = uniq_by_size[size]
        lo = offsets[size]
        hi = lo + int(uniq.shape[0])
        if hi == lo:
            continue
        kinds[lo:hi] = size
        node_codes = uniq[:, :4]
        node_ok = node_codes >= 0
        atom_mask[lo:hi] = node_ok.astype(np.float32)
        atom_fields[lo:hi][node_ok] = _unmix(node_codes[node_ok], atom_dims)
        bond_codes = uniq[:, 4:10]
        bond_ok = bond_codes >= 0
        edge_mask[lo:hi] = bond_ok.astype(np.float32)
        bond_fields[lo:hi][bond_ok] = _unmix(bond_codes[bond_ok], bond_dims)

    for st, ids1 in zip(structs, per_size_ids[1]):
        st.f_atom_type = ids1
    for size in (2, 3, 4):
        for st, ids in zip(structs, per_size_ids[size]):
            st.f_type = getattr(st, "f_type", {})
            st.f_type[size] = ids

    return TypeRegistry(atom_fields=atom_fields, bond_fields=bond_fields,
                        atom_mask=atom_mask, edge_mask=edge_mask, k=kinds,
                        n_by_size=n_by_size, offsets=offsets)


def registry_summary(reg: TypeRegistry) -> dict[str, Any]:
    return {
        "n_types_total": reg.n_types(),
        "n_types_by_size": {str(k): int(v) for k, v in reg.n_by_size.items()},
        "n_edges_total": int(reg.edge_mask.sum()),
    }