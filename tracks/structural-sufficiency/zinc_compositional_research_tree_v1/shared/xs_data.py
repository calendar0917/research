"""Base batch construction and segmentation helpers.

Only label-blind tensors are produced here.  The label vector is stored once,
outside the batch dictionary, and is touched only by the training loss / the
evaluator.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


def segments(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather the flat element ranges of ``rows`` under ``ptr``.

    Returns ``(flat_index, owner_slot, per_element_offset)`` where

    * ``flat_index`` indexes the split-level flat array,
    * ``owner_slot`` is the position of the molecule inside ``rows``,
    * ``per_element_offset`` is the batch-local start offset of that molecule's
      segment (subtract it from the split index to get a local index).
    """
    rows = np.asarray(rows, dtype=np.int64)
    lo = np.asarray(ptr, dtype=np.int64)[rows]
    hi = np.asarray(ptr, dtype=np.int64)[rows + 1]
    cnt = hi - lo
    total = int(cnt.sum())
    owner = np.repeat(np.arange(len(rows), dtype=np.int64), cnt)
    start = np.cumsum(cnt, dtype=np.int64) - cnt
    pos = np.arange(total, dtype=np.int64) - np.repeat(start, cnt)
    return lo[owner] + pos, owner, start


def _t(x: np.ndarray, device=None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.to(device) if device is not None else t


class BaseData:
    """Frozen per-split raw feature tables (label-blind)."""

    def __init__(self, cache: Mapping[str, Any], *, device: str | torch.device = "cpu"):
        self.n = int(len(cache["atom_ptr"]) - 1)
        self.atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
        self.bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
        self.n_atoms = np.asarray(cache["n_atoms_graph"], dtype=np.int64)
        self.n_bonds = np.asarray(cache["n_bonds_graph"], dtype=np.int64)
        self.atom_feat = torch.as_tensor(np.asarray(cache["atom_feat"], dtype=np.float32))
        self.bond_feat = torch.as_tensor(np.asarray(cache["bond_feat"], dtype=np.float32))
        self.bond_index = torch.as_tensor(np.asarray(cache["bond_index"], dtype=np.int64))
        self.global_feat = torch.as_tensor(np.asarray(cache["global_feat"], dtype=np.float32))
        self.labels = np.asarray(cache["labels"], dtype=np.float64)
        self.device = torch.device(device)
        if self.device.type != "cpu":
            self.to(self.device)

    def to(self, device) -> "BaseData":
        self.device = torch.device(device)
        for name in ("atom_feat", "bond_feat", "bond_index", "global_feat"):
            setattr(self, name, getattr(self, name).to(self.device))
        return self

    # -- batch ---------------------------------------------------------------
    def collate(self, rows: Sequence[int], *, norm: Mapping[str, Any]) -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        atom_idx, atom_owner, atom_off = segments(self.atom_ptr, rows)
        bond_idx, bond_owner, bond_off = segments(self.bond_ptr, rows)
        na = self.n_atoms[rows]
        nb = self.n_bonds[rows]
        bond_index = self.bond_index.index_select(
            0, _t(bond_idx, self.device)).clone()
        # per-molecule batch-local atom offset (bond_owner is the molecule slot)
        bond_index += _t(atom_off[bond_owner], self.device).unsqueeze(1)
        g = self.global_feat.index_select(0, _t(rows, self.device))
        mean = torch.as_tensor(norm["global_mean"], dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(norm["global_scale"], dtype=torch.float32, device=self.device)
        g = (g - mean) / scale
        return {
            "atom_feat": self.atom_feat.index_select(0, _t(atom_idx, self.device)),
            "bond_feat": self.bond_feat.index_select(0, _t(bond_idx, self.device)),
            "bond_index": bond_index,
            "global_feat": g,
            "atom_batch": _t(atom_owner, self.device),
            "bond_batch": _t(bond_owner, self.device),
            "n_graphs": int(len(rows)),
            "n_atoms": _t(na, self.device),
            "n_bonds": _t(nb, self.device),
            "atom_off": atom_off,
            "bond_off": bond_off,
            "atom_ptr": self.atom_ptr,
            "bond_ptr": self.bond_ptr,
            "rows": rows,
        }

    def targets(self, rows: Sequence[int]) -> np.ndarray:
        return self.labels[np.asarray(rows, dtype=np.int64)]
