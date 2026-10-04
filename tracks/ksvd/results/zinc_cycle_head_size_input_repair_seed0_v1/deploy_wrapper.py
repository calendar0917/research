"""Deployable wrapper for zinc-cycle-head-size-input-repair-seed0-v1 (QNE).

Inference contract
------------------
    P_raw = frozen_h(x) + q_QNE(T25, size2)
    b_P   = one fit-only median (stored in the bundle)
    P_cal = P_raw + b_P

``size2`` is computed **online from the actual graph**: ``N = actual
num_nodes`` (isolated nodes included) and ``E = number of unique undirected
pairs {min(u,v), max(u,v)}``, standardised by the fit-8000 scaler stored in the
bundle.  It is never looked up from a stable-id-keyed feature table, and no
``y``/``c``/``k``/group/molecule label enters the forward path.

``T25`` must be the existing Full ``topology25`` model input (same 8000-fit
prep as the frozen ``O`` branch); this wrapper does not rebuild it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOPOLOGY_IN = 25
HIDDEN = (64, 32)


def unique_undirected_pairs(edge_u: Sequence[int], edge_v: Sequence[int]) -> list[tuple[int, int]]:
    pairs = set()
    for a, b in zip(edge_u, edge_v):
        a, b = int(a), int(b)
        pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def size2_from_graph(num_nodes: int, edge_u: Sequence[int], edge_v: Sequence[int],
                     scaler: Mapping[str, float]) -> np.ndarray:
    """The only place N/E are produced: actual graph, no labels, no id lookup."""
    n = int(num_nodes)
    e = len(unique_undirected_pairs(edge_u, edge_v))
    return np.array(
        [
            (n - float(scaler["n_mean"])) / float(scaler["n_std"]),
            (e - float(scaler["e_mean"])) / float(scaler["e_std"]),
        ],
        dtype=np.float32,
    )


class SizeConditionedHead(nn.Module):
    """Same module as the training runner: 25->64->32->1 plus gate * W_S(2->64)."""

    def __init__(self, gate: float = 1.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(TOPOLOGY_IN, HIDDEN[0])
        self.fc2 = nn.Linear(HIDDEN[0], HIDDEN[1])
        self.fc3 = nn.Linear(HIDDEN[1], 1)
        self.w_s = nn.Linear(2, HIDDEN[0], bias=False)
        self.gate = float(gate)

    def forward(self, t25: torch.Tensor, size2: torch.Tensor) -> torch.Tensor:
        z = self.fc1(t25) + self.gate * F.linear(size2, self.w_s.weight)
        z = F.silu(z)
        z = F.silu(self.fc2(z))
        return self.fc3(z).view(-1)


class SizeInputRepairPredictor(nn.Module):
    """Deployable QNE inference (no labels, no id-keyed features)."""

    def __init__(self, full: nn.Module, head: nn.Module, scaler: Mapping[str, float], b_P: float) -> None:
        super().__init__()
        self.full = full
        self.head = head
        self.scaler = dict(scaler)
        self.b_P = float(b_P)
        self.full.eval()
        self.head.eval()
        for parameter in self.full.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch: Any, t25: torch.Tensor, num_nodes: int,
                edge_u: Sequence[int], edge_v: Sequence[int],
                mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.full(batch, mask=mask).view(-1)
        size2 = torch.as_tensor(size2_from_graph(num_nodes, edge_u, edge_v, self.scaler),
                                dtype=torch.float32).view(1, 2)
        q = self.head(t25, size2).view(-1)
        return h + q + self.b_P


def load_predictor(bundle_path: str, build_full_model, prep_blob: Mapping[str, Any],
                   map_location: str = "cpu") -> SizeInputRepairPredictor:
    """Load the stored QNE bundle and the frozen O branch.

    ``build_full_model(blob, seed)`` is the repository Full constructor
    (``zinc_full_cycle_target_decomposition_v1.build_full_model``).
    """
    bundle = torch.load(bundle_path, map_location=map_location, weights_only=False)
    full = build_full_model(prep_blob, int(bundle["seed"]))
    o_path = Path(bundle["o_soup_state_path"])
    if not o_path.is_absolute():
        # bundle stores paths relative to tracks/ksvd/; accept cwd-relative too
        candidates = [Path.cwd() / o_path, Path(__file__).resolve().parents[2] / o_path]
        o_path = next((p for p in candidates if p.exists()), candidates[-1])
    full.load_state_dict(torch.load(str(o_path), map_location=map_location))
    head = SizeConditionedHead(float(bundle.get("gate", 1.0)))
    head.load_state_dict(bundle["head_state"])
    return SizeInputRepairPredictor(full, head, bundle["size_scaler"], float(bundle["b_P"]))
