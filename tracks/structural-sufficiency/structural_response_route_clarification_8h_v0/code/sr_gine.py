"""The strong-prediction reference: the repository GINE ``G0`` (5x128, Rf recipe).

The trunk and dense head are imported from ``gnn_route_clarification_8h_v1``
(read-only) rather than re-implemented.  Only the input packing is new: this
module builds the official OGB categorical tensors directly from the pinned
``sq-pilot`` cache, which carries exactly the fields G0 consumes
(9 atom categorical columns, degree, min-ring, 3 bond categorical columns).

``G0`` is the strong *application* reference.  It is **not** a same-parameter,
same-input operator control for FGCR; cross-family comparisons are application
comparisons only.
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import sr_common as C

if str(C.GR_CODE) not in sys.path:
    sys.path.insert(0, str(C.GR_CODE))

import gr_models as GM  # noqa: E402


def pack(bundle, rows: Sequence[int]) -> dict[str, torch.Tensor]:
    rows = [int(r) for r in rows]
    atom_parts, deg_parts, ring_parts, eidx_parts, ecat_parts = [], [], [], [], []
    counts: list[int] = []
    node_off = 0
    for row in rows:
        a0, a1 = int(bundle.node_ptr[row]), int(bundle.node_ptr[row + 1])
        b0, b1 = int(bundle.bond_ptr[row]), int(bundle.bond_ptr[row + 1])
        n = a1 - a0
        counts.append(n)
        atom_parts.append(bundle.atom_feat[a0:a1])
        deg_parts.append(bundle.degree[a0:a1])
        ring_parts.append(bundle.min_ring[a0:a1])
        bu = bundle.bond_u[b0:b1]
        bv = bundle.bond_v[b0:b1]
        bf = bundle.bond_feat[b0:b1]
        eidx_parts.append(np.concatenate(
            [np.stack([bu + node_off, bv + node_off]),
             np.stack([bv + node_off, bu + node_off])], axis=1))
        ecat_parts.append(np.concatenate([bf, bf], axis=0))
        node_off += n
    x_cat = np.concatenate(atom_parts, axis=0)
    deg = np.concatenate(deg_parts)
    ring = np.concatenate(ring_parts)
    edge_index = np.concatenate(eidx_parts, axis=1) if eidx_parts else np.zeros((2, 0), np.int64)
    edge_cat = np.concatenate(ecat_parts, axis=0) if ecat_parts else np.zeros((0, 3), np.int64)
    atom_batch = np.repeat(np.arange(len(counts), dtype=np.int64), np.asarray(counts, dtype=np.int64))
    return {
        "x_cat": torch.as_tensor(x_cat, dtype=torch.long),
        "deg": torch.as_tensor(deg, dtype=torch.long),
        "ring": torch.as_tensor(ring, dtype=torch.long),
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
        "edge_cat": torch.as_tensor(edge_cat, dtype=torch.long),
        "gine_batch": torch.as_tensor(atom_batch, dtype=torch.long),
        "gine_n_atoms": torch.as_tensor(np.asarray(counts, dtype=np.int64), dtype=torch.long),
    }


class GineG0(nn.Module):
    def __init__(self, n_categories: Mapping[str, int]) -> None:
        super().__init__()
        self.trunk = GM.GineTrunk("G0", None, {}, n_categories)
        self.head = GM.DenseHead("G0")

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        h = self.trunk(batch)
        parts = self.head.forward_parts(h, batch["gine_batch"], batch["gine_n_atoms"])
        return parts["pred_std"]

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def make_init_state(n_categories: Mapping[str, int], seed: int = 0) -> dict[str, torch.Tensor]:
    """One seeded from-scratch initial state, shared by every G0 arm."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        model = GineG0(n_categories)
        return {k: v.detach().clone() for k, v in model.state_dict().items()}
    finally:
        torch.random.set_rng_state(state)


def build_model(n_categories: Mapping[str, int],
                init_state: Mapping[str, torch.Tensor]) -> GineG0:
    model = GineG0(n_categories)
    model.load_state_dict({k: v.clone() for k, v in init_state.items()}, strict=True)
    return model


def forward_rows(model: GineG0, batch: Mapping[str, torch.Tensor],
                 rows: Sequence[int]) -> torch.Tensor:
    return model(batch)


@torch.no_grad()
def predict_std(model: GineG0, bundle, rows: np.ndarray, device: Any,
                batch_size: int = 256) -> np.ndarray:
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros(rows.size, dtype=np.float64)
    for start in range(0, rows.size, int(batch_size)):
        part = rows[start:start + int(batch_size)]
        batch = {k: v.to(device) for k, v in pack(bundle, part.tolist()).items()}
        out[start:start + part.size] = model(batch).double().cpu().numpy()
    return out


def base_loss(model: GineG0, bundle, rows: np.ndarray, device: Any) -> torch.Tensor:
    rows = np.asarray(rows, dtype=np.int64)
    batch = {k: v.to(device) for k, v in pack(bundle, rows.tolist()).items()}
    pred = model(batch)
    target = torch.as_tensor(bundle.labels_std[rows], dtype=torch.float32, device=device)
    return torch.mean((pred - target) ** 2)
