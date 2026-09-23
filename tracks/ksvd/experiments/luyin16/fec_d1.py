"""FEC-D1 — Localized Sparse Structural Dictionary Binding: core primitives.

Pre-registration: ``tracks/ksvd/notes/fec_d1_preregistration.md`` (frozen).
Prior-artifact audit: ``tracks/ksvd/notes/fec_d1_prior_artifact_audit.md``.

Single question
---------------
Take the SDB-v0-validated frozen sparse pure-topology dictionary
``D_SDB in R^{65x32}`` (``s = 8``) and its node-centric code ``alpha_v``.
Localize ``alpha_v`` into every rooted chemical environment: for root ``i``
and shell ``k`` form the *within-shell centred* binding of ``alpha_v`` against
primitive atom chemistry ``q_v``, and add the resulting refinement to the
frozen FEC-S1 local environment ``e_i^shared in R^24``.  Does that give a
strong frozen FEC-S1 a *dictionary-specific* predictive gain?

Frozen path (see the pre-registration for the full specification)::

    phi_v^65 --D_SDB--> alpha_v^32
    C_i^D = [ C_i0 ; C_i1 ; C_i2 ] in R^2688
    C_i^D --W1(2688->8)--> u_i --W2(8->24)--> delta_e_i
    e_i^new = e_i^shared + delta_e_i      (frozen FEC-S1 downstream)

This module contains only definitions (no data loading, no training) so every
algebraic property demanded by the pre-registration is unit-testable without
touching ZINC.  The runner is ``zinc_fec_d1.py``; focused CPU tests are
``tracks/ksvd/tests/test_fec_d1.py``.

``official_test_loaded = false`` always.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb

# ---------------------------------------------------------------------------
# frozen configuration (pre-registered; no sweep)
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "fec_d1"

N_SHELLS = 3
PATCH_RADIUS = 2
K_ATOMS = int(sdb.K_ATOMS)  # 32
SPARSITY = int(sdb.SPARSITY)  # 8
ATOM_CATEGORIES = int(sdb.ATOM_CATEGORIES)  # 28
PHI_DIM = int(sdb.PHI_DIM)  # 65

STAT_DIM = N_SHELLS * K_ATOMS * ATOM_CATEGORIES  # 2688
BRANCH_RANK = 8
BRANCH_OUT = 24
BRANCH_SEED = 0

DICT_ARM = "dict"
PCA_ARM = "pca"
ARMS = (DICT_ARM, PCA_ARM)

SCALER_FLOOR = float(r2.SCALER_FLOOR)
SCALER_EPS = float(r2.SCALER_EPS)

#: frozen decision thresholds (pre-registration §15)
GATE_A_GAIN = 0.003
GATE_B_DICT_SPECIFIC = 0.002
GATE_C_ASSIGNMENT = 0.010

VERDICTS = {
    "supported": "FEC_D1_LOCALIZED_DICTIONARY_BINDING_SUPPORTED",
    "not_specific": "FEC_D1_LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC",
    "not_assignment": "FEC_D1_DICTIONARY_GAIN_NOT_ASSIGNMENT_MEDIATED",
    "no_gain": "FEC_D1_NO_MATERIAL_LOCAL_DICTIONARY_GAIN",
}


# ---------------------------------------------------------------------------
# rooted shells (root order = node order; pure topology)
# ---------------------------------------------------------------------------


def ego_shells(graph: Any, center: int, radius: int = PATCH_RADIUS) -> dict[int, list[int]]:
    """BFS distance-bucketed nodes inside the radius-``radius`` patch of ``center``."""
    center = int(center)
    distances: dict[int, int] = {center: 0}
    queue: deque[int] = deque([center])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbor in sorted(graph.neighbors(node)):
            neighbor = int(neighbor)
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    shells: dict[int, list[int]] = {}
    for node, distance in distances.items():
        shells.setdefault(int(distance), []).append(int(node))
    for key in shells:
        shells[key].sort()
    return shells


# ---------------------------------------------------------------------------
# localized within-shell centred binding
# ---------------------------------------------------------------------------


def per_root_binding(
    phi: np.ndarray,
    atom_idx: np.ndarray,
    graph: Any,
    coord: np.ndarray,
    *,
    shuffle_seed: int | None = None,
) -> np.ndarray:
    """``[n_roots, N_SHELLS * K * C]`` within-shell centred binding of one molecule.

    ``coord`` is the per-node structural coordinate (``alpha`` or ``z``) with
    shape ``[n, K]`` and row order equal to ``graph.nodes`` (0..n-1).
    ``shuffle_seed`` permutes the ``coord_v <-> q_v`` correspondence within every
    root and shell independently, keeping both multisets.
    """
    phi = np.asarray(phi, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64)
    atom_idx = np.asarray(atom_idx, dtype=np.int64).reshape(-1)
    n = int(phi.shape[0])
    if coord.shape[0] != n or atom_idx.shape[0] != n:
        raise RuntimeError("phi / coord / atom_idx node-count mismatch")
    k_dim = int(coord.shape[1])
    q = r2.one_hot_q(atom_idx, ATOM_CATEGORIES).astype(np.float64)
    out = np.zeros((n, N_SHELLS * k_dim * ATOM_CATEGORIES), dtype=np.float64)
    rng = None if shuffle_seed is None else np.random.default_rng(int(shuffle_seed))
    for root in range(n):
        shells = ego_shells(graph, root)
        for shell in range(N_SHELLS):
            index = shells.get(shell, [])
            if len(index) < 2:
                continue  # |S_ik| < 2  =>  C_ik = 0
            a = coord[index]
            qq = q[index]
            if rng is not None:
                qq = qq[rng.permutation(len(index))]
            a_centred = a - a.mean(axis=0, keepdims=True)
            q_centred = qq - qq.mean(axis=0, keepdims=True)
            block = a_centred.T @ q_centred
            lo = shell * k_dim * ATOM_CATEGORIES
            out[root, lo : lo + k_dim * ATOM_CATEGORIES] = block.reshape(-1)
    return out


# ---------------------------------------------------------------------------
# train-only RMS scaling (reuse SDB primitive verbatim)
# ---------------------------------------------------------------------------


def fit_binding_scaler(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-only per-coordinate RMS scaler + mask for a ``[N, STAT_DIM]`` array."""
    return sdb.fit_rms_scaler(np.asarray(stats, dtype=np.float64))


def apply_binding_scaler(stats: np.ndarray, scale: np.ndarray, mask: np.ndarray) -> np.ndarray:
    stats = np.asarray(stats, dtype=np.float64)
    return sdb.apply_rms_scaler(stats, scale, mask)


def scaler_report(scale: np.ndarray, mask: np.ndarray, stats: np.ndarray) -> dict[str, Any]:
    stats = np.asarray(stats, dtype=np.float64)
    raw_rms = np.sqrt((stats ** 2).mean(axis=0))
    effective = mask > 0.0
    return {
        "stat_dim": int(stats.shape[1]),
        "n_patches": int(stats.shape[0]),
        "effective_coordinates": int(effective.sum()),
        "zero_coordinates": int((~effective).sum()),
        "effective_fraction": float(effective.mean()),
        "raw_rms_min_effective": float(raw_rms[effective].min()) if effective.any() else 0.0,
        "raw_rms_max_effective": float(raw_rms[effective].max()) if effective.any() else 0.0,
        "raw_rms_max_all": float(raw_rms.max()),
        "floor": float(SCALER_FLOOR),
    }


# ---------------------------------------------------------------------------
# rank-8 branch adapter (the only trainable parameters)
# ---------------------------------------------------------------------------


class BindingLocalEnvAdapter(nn.Module):
    """FEC-S1 local-environment adapter plus a rank-8 localized-binding refinement.

    ``L(x) = base(x) + (C~ @ W1^T) @ W2^T`` with ``base`` the frozen FEC-S1
    ``LocalEnvironmentAdapter``.  ``W2`` is initialized to exact zero, so
    ``L == base`` at step 0.  The per-patch statistic must be installed with
    :meth:`set_stat` immediately before each forward; it has one row per patch
    in the *same patch order* as ``x``.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        stat_dim: int = STAT_DIM,
        rank: int = BRANCH_RANK,
        out_width: int = BRANCH_OUT,
        seed: int = BRANCH_SEED,
    ) -> None:
        super().__init__()
        self.base = base
        self.stat_dim = int(stat_dim)
        self.rank = int(rank)
        self.out_width = int(out_width)
        generator = torch.Generator().manual_seed(int(seed))
        weight1 = torch.empty(int(rank), int(stat_dim), dtype=torch.float32)
        nn.init.kaiming_uniform_(weight1, a=math.sqrt(5.0), generator=generator)
        self.W1 = nn.Parameter(weight1)
        self.W2 = nn.Parameter(torch.zeros(int(out_width), int(rank), dtype=torch.float32))
        self._stat: torch.Tensor | None = None

    def set_stat(self, stat: torch.Tensor | None) -> "BindingLocalEnvAdapter":
        self._stat = stat
        return self

    def clear_stat(self) -> "BindingLocalEnvAdapter":
        self._stat = None
        return self

    def branch_delta(self, stat: torch.Tensor | None = None) -> torch.Tensor:
        value = self._stat if stat is None else stat
        if value is None:
            raise RuntimeError("BindingLocalEnvAdapter.stat is not set")
        return (value @ self.W1.t()) @ self.W2.t()

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        output = self.base(descriptor)
        if self._stat is not None:
            output = output + self.branch_delta(self._stat)
        return output

    def branch_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.W1, self.W2


# ---------------------------------------------------------------------------
# parameter accounting / freezing
# ---------------------------------------------------------------------------


def branch_parameter_count(stat_dim: int = STAT_DIM, rank: int = BRANCH_RANK, out_width: int = BRANCH_OUT) -> int:
    return int(rank) * int(stat_dim) + int(out_width) * int(rank)


def freeze_base_train_branch(model: nn.Module) -> dict[str, Any]:
    """Freeze every parameter except the two branch matrices."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    branch = model.local_env_adapter
    if not isinstance(branch, BindingLocalEnvAdapter):
        raise RuntimeError("model.local_env_adapter is not a BindingLocalEnvAdapter")
    branch.W1.requires_grad_(True)
    branch.W2.requires_grad_(True)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [
        name for name in trainable if name not in {"local_env_adapter.W1", "local_env_adapter.W2"}
    ]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected}")
    return {
        "trainable_names": trainable,
        "trainable_numel": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_numel": int(sum(p.numel() for p in model.parameters())),
        "frozen_numel": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)),
    }


# ---------------------------------------------------------------------------
# branch evaluation helpers
# ---------------------------------------------------------------------------


def branch_delta_norm(model: nn.Module) -> float:
    branch = model.local_env_adapter
    assert isinstance(branch, BindingLocalEnvAdapter)
    if branch._stat is None:
        return float("nan")
    with torch.no_grad():
        return float(branch.branch_delta().norm())


__all__ = [
    "PROTOCOL_VERSION",
    "N_SHELLS",
    "PATCH_RADIUS",
    "K_ATOMS",
    "SPARSITY",
    "ATOM_CATEGORIES",
    "PHI_DIM",
    "STAT_DIM",
    "BRANCH_RANK",
    "BRANCH_OUT",
    "BRANCH_SEED",
    "DICT_ARM",
    "PCA_ARM",
    "ARMS",
    "GATE_A_GAIN",
    "GATE_B_DICT_SPECIFIC",
    "GATE_C_ASSIGNMENT",
    "VERDICTS",
    "ego_shells",
    "per_root_binding",
    "fit_binding_scaler",
    "apply_binding_scaler",
    "scaler_report",
    "BindingLocalEnvAdapter",
    "branch_parameter_count",
    "freeze_base_train_branch",
    "branch_delta_norm",
]
