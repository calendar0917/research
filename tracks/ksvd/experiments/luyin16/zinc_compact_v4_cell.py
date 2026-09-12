"""Compact-v4-cell: explicit persistent cycle-cell branch (minimal falsification).

This stage implements the **single** representation-family hypothesis authorised
by the Same-Budget Structural Computation Gap Analysis
(``notes/structural_computation_gap_analysis.md``,
``results/structural_computation_gap_analysis/top1_hypothesis.json``,
``.../minimal_falsification_plan.json``):

    Promote cycle structure from a graph-level descriptor to an explicit,
    persistent, identity-bearing structural object (a cycle *cell*) that
    participates in a small fixed number of learned incidence-aware rounds
    before the graph representation is pooled.

What is added
-------------
* ``compact-v4`` base pathway, unchanged: patch -> pair -> centre aggregation
  -> centre update -> unary/pair/global summaries -> global topology hinge
  branch -> ``R_original in R^302``.
* A new cell branch:
  - cycle cells = bounded *induced* (chordless) cycles of length 3..8,
    enumerated once per molecule (target-independent graph structure);
  - every cell gets a persistent learned state ``c_k`` (48D);
  - exactly **2** pre-registered incidence-aware rounds (patch<->cell);
  - the final cell states are pooled into ``S_cell = [mean; std; log1p(count)]
    in R^97``;
  - ``R_new = [R_original ; S_cell] in R^399`` and the existing head consumes
    the wider vector.  No new head depth / attention / FM / readout family.

What is *not* here
------------------
No CIN replication, no generic message passing, no per-cycle-length MLPs, no
target-derived feature, no official-test access, no depth sweep (T=2 fixed),
no hyper-parameter search.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_cell <stage>

Stages: ``protocol_lock cycle_lock extract param_audit compute_audit integrity
baseline stage1_true_seed0 stage1_decision
stage2_broken_seed0 stage2_decision
stage3_true_seed1 stage3a_decision
stage3_broken_seed1 stage3b_decision final diagnostics all``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import pickle
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch, Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.structural_context import _find_cycles
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    TYPED_TOKENIZER_V1_HISTORICAL,
    resolve_typed_tokenizer_version,
    typed_tokenizer_fingerprint,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _data_to_graph,
    _load_zinc,
)

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_cell"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
CACHE_DIR = RESULTS_DIR / "cache"
FIGURE_DIR = RESULTS_DIR / "figures"

CANONICAL_V4_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
CANONICAL_SUFFICIENCY_DIR = TRACK_ROOT / "results/compact_v4_training_sufficiency"
V4_SEED0_STATE = CANONICAL_SUFFICIENCY_DIR / "states/Pstar_A2_long_seed0_selection_state.pt"
V4_SEED1_STATE = CANONICAL_SUFFICIENCY_DIR / "states/Pstar_A2_long_seed1_selection_state.pt"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2

# --- locked architecture constants (see architecture_lock.json) --------------
CELL_DIM = 48
MAX_CYCLE_LEN = 8  # bounded induced (chordless) cycle enumeration; NOT swept
CELL_ROUNDS = 2  # exactly two pre-registered incidence-aware rounds
R_ORIGINAL = 302
CELL_SUMMARY_WIDTH = 2 * CELL_DIM + 1  # 97
R_NEW = R_ORIGINAL + CELL_SUMMARY_WIDTH  # 399

# --- frozen optimized training protocol (inherited, not re-tuned) ------------
OPTIMIZED_PROTOCOL: dict[str, Any] = {
    "optimizer": "Adam",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "max_epochs": 240,
    "patience": 40,
    "scheduler": "none",
    "gradient_clip_norm": 5.0,
    "loss": "L1 / mean absolute error",
    "checkpoint_selection": "best official-valid MAE",
    "single_stage": True,
}
TORCH_THREADS = 4

# --- reused optimized-v4 references (do NOT retrain) -------------------------
V4_SEED0_VALID = 0.14642022556537995
V4_SEED1_VALID = 0.1493322635096847
V4_SEED0_EPOCH = 169
V4_SEED1_EPOCH = 104
V4_PARAMS = 99613
V4_TYPED_VOCAB_WITH_OOV = 6785
V4_PARENT_VOCAB_WITH_OOV = 32

# --- pre-registered decision gates -------------------------------------------
ARCH_GATE = 0.004  # Delta_arch0 = v4_valid - true_valid
ARCH_GATE_SEED1 = 0.003
ARCH_MEAN_GATE = 0.004
INC_GATE = 0.0025  # Delta_inc0 = broken_valid - true_valid
INC_MEAN_GATE = 0.0025
BULK_GATE = 0.002  # candidate MAE - v4 MAE on target-independent bulk
PARAM_CEILING = 120000

CACHE_SCHEMA_VERSION = "compact_v4_cell_records_v2"
PROTOCOL_VERSION = "compact_v4_cell_minimal_falsification_v1"

# Break-incidence RNG: fixed deterministic base seed for the manifest.
BROKEN_BASE_SEED = 20260920


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _git_commit() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    if head.exists():
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        return text
    return "unknown"


def _environment_fingerprint() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "torch_global_seed": torch.initial_seed(),
        "device": "cpu",
    }


def _configure_determinism() -> None:
    torch.set_num_threads(int(TORCH_THREADS))
    torch.use_deterministic_algorithms(False)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


# ---------------------------------------------------------------------------
# cycle construction (target-independent, deterministic, permutation invariant)
# ---------------------------------------------------------------------------


def induced_cycles(graph: Any, max_len: int = MAX_CYCLE_LEN) -> list[tuple[int, ...]]:
    """Enumerate chordless (induced) simple cycles of length 3..``max_len``.

    Reuses the repository's tested exact min-node DFS (``_find_cycles``, the
    same enumerator behind the v4 topology channel) and then *filters to
    induced cycles*: a simple cycle is chordless iff every vertex on it has
    exactly two neighbours inside the cycle's vertex set.

    Returns a deterministically ordered list of sorted vertex tuples.  Node
    IDs are an enumeration device only; the returned *set* of vertex sets is
    invariant under relabelling, and the sort makes the list canonical.
    """
    max_len = int(max_len)
    if max_len < 3:
        raise ValueError("max_len must be >= 3")
    simple = _find_cycles(graph, max_len)
    out: list[tuple[int, ...]] = []
    for cycle in simple:
        vertices = {int(v) for v in cycle}
        induced = True
        for u in vertices:
            inside = sum(1 for v in graph.neighbors(u) if int(v) in vertices)
            if inside != 2:
                induced = False
                break
        if induced:
            out.append(tuple(sorted(vertices)))
    out.sort()
    return out


def cycles_for_data(data: Any, max_len: int = MAX_CYCLE_LEN) -> list[tuple[int, ...]]:
    graph, _node_types, _edge_types = _data_to_graph(data)
    return induced_cycles(graph, max_len=max_len)


def incidence_edges(
    cycles: Sequence[Sequence[int]], n_patches: int
) -> list[tuple[int, int]]:
    """Bipartite incidence edges ``(patch_index, cell_index)``.

    Patch index == centre atom index (the patch front-end builds exactly one
    radius-2 ego-net per atom, in node order), so atom membership in a cycle is
    exactly patch membership in a cell.  This is ``I_PC``: which lower
    (patch-centred) object belongs to which cycle cell.
    """
    edges: list[tuple[int, int]] = []
    for cell_index, cycle in enumerate(cycles):
        for atom in cycle:
            atom = int(atom)
            if 0 <= atom < int(n_patches):
                edges.append((atom, cell_index))
    edges.sort()
    return edges


def rewire_incidence(
    edges: Sequence[tuple[int, int]],
    n_patches: int,
    n_cells: int,
    seed: int,
    n_attempts: int | None = None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Deterministic within-molecule degree-preserving bipartite edge swaps.

    Preserves: per-cell incidence count (cell size), per-patch incidence degree
    (how many cells a patch belongs to), cell count and total incidence count.
    Changes only *which* patch is incident to *which* cell.
    """
    current = set((int(p), int(c)) for p, c in edges)
    E = len(current)
    info: dict[str, Any] = {
        "n_edges": int(E),
        "n_cells": int(n_cells),
        "swaps": 0,
        "unbreakable": True,
    }
    if E < 4 or n_cells < 2:
        return sorted(current), info
    rng = random.Random(int(seed))
    attempts = int(n_attempts) if n_attempts is not None else max(50, 10 * E)
    edge_list = sorted(current)
    swaps = 0
    for _ in range(attempts):
        i = rng.randrange(E)
        j = rng.randrange(E)
        if i == j:
            continue
        p1, c1 = edge_list[i]
        p2, c2 = edge_list[j]
        if p1 == p2 or c1 == c2:
            continue
        if (p1, c2) in current or (p2, c1) in current:
            continue
        current.discard((p1, c1))
        current.discard((p2, c2))
        current.add((p1, c2))
        current.add((p2, c1))
        edge_list[i] = (p1, c2)
        edge_list[j] = (p2, c1)
        swaps += 1
    info["swaps"] = int(swaps)
    info["unbreakable"] = swaps == 0
    return sorted(current), info


def _validate_incidence_marginals(
    true_edges: Sequence[tuple[int, int]],
    broken_edges: Sequence[tuple[int, int]],
    n_patches: int,
    n_cells: int,
) -> dict[str, Any]:
    def deg(edges, key_index, n):
        degs = [0] * int(n)
        for edge in edges:
            degs[int(edge[key_index])] += 1
        return degs

    tp = deg(true_edges, 0, n_patches)
    bp = deg(broken_edges, 0, n_patches)
    tc = deg(true_edges, 1, n_cells)
    bc = deg(broken_edges, 1, n_cells)
    return {
        "cell_count_identical": True,
        "patch_degrees_preserved": sorted(tp) == sorted(bp),
        "cell_degrees_preserved": sorted(tc) == sorted(bc),
        "total_edges_preserved": len(true_edges) == len(broken_edges),
        "cell_size_multiset_preserved": sorted(tc) == sorted(bc),
        "edges_different": set(true_edges) != set(broken_edges),
    }


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class PatchPathCellModel(zpp.PatchPathModel):
    """compact-v4 plus an explicit persistent cycle-cell branch.

    The base ``compact-v4`` pathway is inherited unchanged.  ``super().encode``
    returns the original 302D graph representation ``R_original``; the mature
    patch states ``h'_i`` (the ones that feed the unary readout) are captured
    with forward hooks.  The cell branch keeps its own branch-local patch
    states so that ``R_original`` is never modified.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        cell_dim: int = CELL_DIM,
        cell_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(typed_vocabulary_size, parent_vocabulary_size, **kwargs)
        self.cell_dim = int(cell_dim)
        self.cell_enabled = bool(cell_enabled)
        if self.cell_dim != int(self.patch_hidden):
            raise ValueError(
                "cell_dim must equal patch_hidden so the incidence rounds act on "
                "the existing lower-order state width"
            )
        dropout = float(kwargs.get("dropout", 0.05))
        self.cell_dropout = dropout
        # --- cell initialization: mean boundary patch state -> cell state ---
        self.cell_init = nn.Sequential(
            nn.Linear(self.patch_hidden, self.cell_dim),
            nn.LayerNorm(self.cell_dim),
            nn.ReLU(),
        )
        # --- shared rank-aware incidence operators (shared across 2 rounds) --
        self.cell_update = nn.Linear(self.cell_dim, self.cell_dim)
        self.cell_norm = nn.LayerNorm(self.cell_dim)
        self.patch_update = nn.Linear(self.cell_dim, self.patch_hidden)
        # zero-init the residual outputs so the branch starts as a no-op for
        # the patch states (numerically stable, no spurious bias injection).
        nn.init.zeros_(self.cell_update.weight)
        nn.init.zeros_(self.cell_update.bias)
        nn.init.zeros_(self.patch_update.weight)
        nn.init.zeros_(self.patch_update.bias)
        # --- final cell summary + widened head (302 -> 399) ------------------
        self.cell_summary_width = 2 * self.cell_dim + 1
        widened = int(self.unified_graph_width) + int(self.cell_summary_width)
        self.unified_graph_width_with_cell = widened
        self.head = nn.Sequential(
            nn.Linear(widened, self.head_hidden_0),
            nn.LayerNorm(self.head_hidden_0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_hidden_0, self.head_hidden_1),
            nn.ReLU(),
            nn.Linear(self.head_hidden_1, 1),
        )
        # --- forward hooks capturing the mature lower-order patch states -----
        self._patch_encoder_out: torch.Tensor | None = None
        self._center_update_out: torch.Tensor | None = None
        self.patch_encoder.register_forward_hook(self._hook_patch_encoder)
        if self.center_update is not None:
            self.center_update.register_forward_hook(self._hook_center_update)
        self.last_cell_state_norm = 0.0
        self.last_cell_summary_norm = 0.0
        self.last_lower_update_norm = 0.0
        # per-round diagnostics (filled by _cell_summary; reset each forward)
        self.round_cell_state_norms: list[float] = []
        self.round_lower_update_norms: list[float] = []
        self.cell_init_norm = 0.0

    # -- hooks ---------------------------------------------------------------
    def _hook_patch_encoder(self, _module, _inputs, output) -> None:
        self._patch_encoder_out = output

    def _hook_center_update(self, _module, _inputs, output) -> None:
        self._center_update_out = output

    def mature_patch_states(self) -> torch.Tensor:
        if self._patch_encoder_out is None:
            raise RuntimeError("patch_encoder has not run yet")
        if self._center_update_out is None:
            return self._patch_encoder_out
        return self._patch_encoder_out + self._center_update_out

    # -- cell branch ---------------------------------------------------------
    @staticmethod
    def _scatter_mean(
        source: torch.Tensor, index: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        dim_size = int(dim_size)
        if index.numel() == 0 or dim_size <= 0:
            return source.new_zeros((max(dim_size, 0), source.shape[1]))
        total = source.new_zeros((dim_size, source.shape[1]))
        total.index_add_(0, index, source)
        counts = (
            torch.bincount(index, minlength=dim_size)
            .clamp_min(1)
            .to(source.dtype)
            .unsqueeze(1)
        )
        return total / counts

    def cell_branch_enabled(self) -> bool:
        return bool(self.cell_enabled)

    def _cell_summary(self, patch: torch.Tensor, data: Any) -> torch.Tensor:
        n_graphs = int(getattr(data, "num_graphs", 1))
        width = int(self.cell_dim)
        empty = patch.new_zeros((n_graphs, self.cell_summary_width))
        if not self.cell_enabled or not hasattr(data, "cell_edge_patch"):
            return empty
        n_cells_total = int(getattr(data, "num_cells_total", 0))
        if n_cells_total == 0 or data.cell_edge_patch.numel() == 0:
            self.last_cell_state_norm = 0.0
            self.last_cell_summary_norm = 0.0
            self.last_lower_update_norm = 0.0
            return empty

        edge_patch = data.cell_edge_patch
        edge_cell = data.cell_edge_cell
        cell_batch = data.cell_batch
        n_patches = int(patch.shape[0])
        self.round_cell_state_norms = []
        self.round_lower_update_norms = []

        # cell initialization from the boundary patch states (true incidence)
        boundary_mean = self._scatter_mean(patch[edge_patch], edge_cell, n_cells_total)
        cells = self.cell_init(boundary_mean)
        self.cell_init_norm = float(cells.detach().norm(dim=1).mean())

        lower_state_after = patch
        for _round in range(CELL_ROUNDS):
            # ---- lower -> cell (boundary aggregation) ----
            lower_mean = self._scatter_mean(
                lower_state_after[edge_patch], edge_cell, n_cells_total
            )
            cells = self.cell_norm(
                cells + self.cell_update(F.relu(lower_mean))
            )
            # ---- cell -> lower (incidence aggregation) ----
            cell_mean = self._scatter_mean(cells[edge_cell], edge_patch, n_patches)
            alive = (
                torch.bincount(edge_patch, minlength=n_patches) > 0
            ).to(lower_state_after.dtype).unsqueeze(1)
            update = self.patch_update(F.relu(cell_mean))
            lower_state_after = lower_state_after + alive * update
            self.round_cell_state_norms.append(
                float(cells.detach().norm(dim=1).mean()) if cells.numel() else 0.0
            )
            self.round_lower_update_norms.append(
                float((alive * update).detach().norm(dim=1).mean())
            )

        self.last_cell_state_norm = (
            float(cells.detach().norm(dim=1).mean()) if cells.numel() else 0.0
        )
        self.last_lower_update_norm = float(
            (lower_state_after.detach() - patch.detach()).norm(dim=1).mean()
        )

        total = patch.new_zeros((n_graphs, width))
        squared = patch.new_zeros((n_graphs, width))
        total.index_add_(0, cell_batch, cells)
        squared.index_add_(0, cell_batch, cells * cells)
        counts = torch.bincount(cell_batch, minlength=n_graphs)
        denom = counts.clamp_min(1).to(cells.dtype).unsqueeze(1)
        mean = total / denom
        variance = (squared / denom - mean * mean).clamp_min(0.0)
        std = torch.sqrt(variance + 1.0e-8)
        log_count = torch.log1p(counts.to(cells.dtype).unsqueeze(1))
        summary = torch.cat([mean, std, log_count], dim=1)
        has_cell = (counts > 0).to(summary.dtype).unsqueeze(1)
        summary = summary * has_cell
        self.last_cell_summary_norm = float(summary.detach().norm(dim=1).mean())
        return summary

    def encode(self, data: Any) -> torch.Tensor:  # type: ignore[override]
        unified = super().encode(data)
        patch = self.mature_patch_states()
        summary = self._cell_summary(patch, data)
        return torch.cat([unified, summary], dim=1)

    def encode_original(self, data: Any) -> torch.Tensor:
        """Return only the 302D compact-v4 representation (G0.1 check)."""
        return super().encode(data)


# ---------------------------------------------------------------------------
# data encoding
# ---------------------------------------------------------------------------


@dataclass
class CellExtraction:
    cycles: list[tuple[int, ...]]
    n_patches: int


def _cells_from_plain(bundle: dict[str, Any]) -> None:
    """In-place convert a pickled plain cell bundle back to CellExtraction."""
    bundle["cells"] = [
        CellExtraction(
            cycles=[tuple(int(x) for x in cycle) for cycle in item["cycles"]],
            n_patches=int(item["n_patches"]),
        )
        for item in bundle["cells"]
    ]


def _cell_records_from_dataset(dataset: Sequence[Any]) -> list[CellExtraction]:
    out: list[CellExtraction] = []
    for data in dataset:
        cycles = cycles_for_data(data)
        out.append(CellExtraction(cycles=cycles, n_patches=int(data.num_nodes)))
    return out


def _attach_cell_tensors(
    encoded: Sequence[Data],
    cells: Sequence[CellExtraction],
    *,
    mode: str,
    base_seed: int,
) -> dict[str, Any]:
    """Attach per-graph cycle-cell incidence tensors to encoded Data objects.

    ``mode`` is ``"true"`` (real membership) or ``"broken"`` (deterministic
    within-molecule degree-preserving rewiring).
    """
    manifest: list[dict[str, Any]] = []
    for index, (data, cell) in enumerate(zip(encoded, cells)):
        cycles = cell.cycles
        n_cells = len(cycles)
        n_patches = int(cell.n_patches)
        true_edges = incidence_edges(cycles, n_patches)
        if str(mode) == "broken":
            seed = int(base_seed) * 1_000_003 + int(index)
            edges, info = rewire_incidence(true_edges, n_patches, n_cells, seed)
            marginals = _validate_incidence_marginals(
                true_edges, edges, n_patches, n_cells
            )
            removed = len(set(true_edges) - set(edges))
            added = len(set(edges) - set(true_edges))
            changed_cells = sum(
                1
                for c in range(n_cells)
                if {p for p, cc in true_edges if cc == c}
                != {p for p, cc in edges if cc == c}
            )
            entry = {
                "graph_index": int(index),
                "n_cells": int(n_cells),
                "n_patches": int(n_patches),
                "n_edges": int(len(true_edges)),
                "rng_seed": int(seed),
                "swaps": int(info["swaps"]),
                "breakable": not bool(info["unbreakable"]),
                "changed_edges": int(removed),
                "added_edges": int(added),
                "changed_edge_fraction": (
                    float(removed / max(len(true_edges), 1))
                ),
                "changed_cells": int(changed_cells),
                "changed_cell_fraction": (
                    float(changed_cells / max(n_cells, 1)) if n_cells else 0.0
                ),
                "true_hash": _hash_payload(sorted(true_edges)),
                "broken_hash": _hash_payload(sorted(edges)),
                **marginals,
            }
            manifest.append(entry)
        else:
            edges = true_edges
        patch_idx = torch.tensor([e[0] for e in edges], dtype=torch.long)
        cell_idx = torch.tensor([e[1] for e in edges], dtype=torch.long)
        data.cell_edge_patch = patch_idx
        data.cell_edge_cell = cell_idx
        data.num_cells = torch.tensor([n_cells], dtype=torch.long)
    summary = {
        "mode": str(mode),
        "n_graphs": int(len(encoded)),
        "n_cells_total": int(sum(len(c.cycles) for c in cells)),
        "n_edges_total": int(
            sum(len(incidence_edges(c.cycles, c.n_patches)) for c in cells)
        ),
        "manifest": manifest,
    }
    if manifest:
        summary["breakable_graphs"] = int(sum(1 for m in manifest if m["breakable"]))
        summary["unbreakable_graphs"] = int(
            sum(1 for m in manifest if not m["breakable"])
        )
        summary["mean_changed_edge_fraction"] = float(
            np.mean([m["changed_edge_fraction"] for m in manifest])
        )
        multi = [m for m in manifest if m["n_cells"] > 1]
        summary["multi_cell_graphs"] = int(len(multi))
        summary["multi_cell_breakable"] = int(
            sum(1 for m in multi if m["breakable"])
        )
    return summary


def cell_collate(data_list: Sequence[Data]) -> Batch:
    batch = Batch.from_data_list(list(data_list))
    edge_patch: list[torch.Tensor] = []
    edge_cell: list[torch.Tensor] = []
    cell_batch: list[torch.Tensor] = []
    patch_offset = 0
    cell_offset = 0
    for graph_index, data in enumerate(data_list):
        n_patches = int(data.num_nodes)
        n_cells = int(data.num_cells.item())
        if n_cells > 0 and int(data.cell_edge_patch.numel()) > 0:
            edge_patch.append(data.cell_edge_patch + patch_offset)
            edge_cell.append(data.cell_edge_cell + cell_offset)
            cell_batch.append(
                torch.full((n_cells,), graph_index, dtype=torch.long)
            )
        patch_offset += n_patches
        cell_offset += n_cells
    batch.cell_edge_patch = (
        torch.cat(edge_patch) if edge_patch else torch.zeros(0, dtype=torch.long)
    )
    batch.cell_edge_cell = (
        torch.cat(edge_cell) if edge_cell else torch.zeros(0, dtype=torch.long)
    )
    batch.cell_batch = (
        torch.cat(cell_batch) if cell_batch else torch.zeros(0, dtype=torch.long)
    )
    batch.num_cells_total = int(cell_offset)
    return batch


def _make_cell_loader(
    graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int
):
    # NOTE: use torch's DataLoader with the explicit ``cell_collate``.  PyG's
    # ``torch_geometric.loader.DataLoader`` overrides the ``collate_fn``
    # argument, which would silently drop the cycle-cell incidence tensors.
    from torch.utils.data import DataLoader as TorchDataLoader

    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return TorchDataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=cell_collate,
    )


# ---------------------------------------------------------------------------
# config / extraction
# ---------------------------------------------------------------------------


def load_config(path: Path = CANONICAL_V4_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cell_config() -> dict[str, Any]:
    config = load_config()
    config["test_policy"] = "no_test"
    config["model"]["cell_dim"] = int(CELL_DIM)
    config["model"]["max_cycle_len"] = int(MAX_CYCLE_LEN)
    config["model"]["cell_rounds"] = int(CELL_ROUNDS)
    config["parameter_audit"] = False
    config["output"] = {
        "json": str(RESULTS_DIR / "scratch.json"),
        "markdown": str(RESULTS_DIR / "scratch.md"),
    }
    return config


def _cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "cell_records_train.pkl.gz",
        CACHE_DIR / "cell_records_valid.pkl.gz",
        CACHE_DIR / "cell_metadata.json",
    )


def extract_records(force: bool = False):
    """Extract and cache base GraphRecords + cell structures for train/valid."""
    train_path, valid_path, meta_path = _cache_paths()
    if not force and train_path.exists() and valid_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
            with gzip.open(train_path, "rb") as handle:
                bundle = pickle.load(handle)
            with gzip.open(valid_path, "rb") as handle:
                valid_bundle = pickle.load(handle)
            _cells_from_plain(bundle)
            _cells_from_plain(valid_bundle)
            return bundle, valid_bundle, meta

    tokenizer_version = resolve_typed_tokenizer_version(TYPED_TOKENIZER_V1_HISTORICAL)
    topology_mode = "hinge"
    started = time.perf_counter()
    train_ds = _load_zinc(ZINC_ROOT, "train")
    valid_ds = _load_zinc(ZINC_ROOT, "val")
    topo_train = ztopo.matrices_for_split("train", train_ds, topology_mode)[0]
    topo_valid = ztopo.matrices_for_split("valid", valid_ds, topology_mode)[0]
    certificate_cache: dict[bytes, bytes] = {}
    train_records, train_meta = zpp._extract_split(
        train_ds,
        "train",
        certificate_cache,
        topology_mode=topology_mode,
        topology_matrix=topo_train,
        tokenizer_version=tokenizer_version,
    )
    valid_records, valid_meta = zpp._extract_split(
        valid_ds,
        "valid",
        certificate_cache,
        topology_mode=topology_mode,
        topology_matrix=topo_valid,
        tokenizer_version=tokenizer_version,
    )
    train_cells = _cell_records_from_dataset(train_ds)
    valid_cells = _cell_records_from_dataset(valid_ds)
    def _plain(cells: Sequence[CellExtraction]) -> list[dict[str, Any]]:
        return [
            {"cycles": [list(c) for c in item.cycles], "n_patches": int(item.n_patches)}
            for item in cells
        ]

    train_bundle_raw = {"records": train_records, "cells": _plain(train_cells)}
    valid_bundle_raw = {"records": valid_records, "cells": _plain(valid_cells)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(train_bundle_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(valid_bundle_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "tokenizer_version": tokenizer_version,
        "topology_mode": topology_mode,
        "max_cycle_len": int(MAX_CYCLE_LEN),
        "n_train": int(len(train_records)),
        "n_valid": int(len(valid_records)),
        "seconds": float(time.perf_counter() - started),
        "train_mean_centres": float(train_meta["mean_centres"]),
        "valid_mean_centres": float(valid_meta["mean_centres"]),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(meta_path, meta)
    print(
        f"[extract] train={len(train_records)} valid={len(valid_records)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    train_bundle = {"records": train_records, "cells": train_cells}
    valid_bundle = {"records": valid_records, "cells": valid_cells}
    return train_bundle, valid_bundle, meta


def build_encoded_records(mode: str = "true"):
    """Return (train_data, valid_data, audit) with cell tensors attached."""
    train_bundle, valid_bundle, _meta = extract_records()
    config = cell_config()
    enc_train, enc_valid, audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    base_seed = BROKEN_BASE_SEED
    train_summary = _attach_cell_tensors(
        enc_train, train_bundle["cells"], mode=mode, base_seed=base_seed
    )
    valid_summary = _attach_cell_tensors(
        enc_valid, valid_bundle["cells"], mode=mode, base_seed=base_seed
    )
    audit = dict(audit)
    audit["cell_incidence"] = {"train": train_summary, "valid": valid_summary}
    return enc_train, enc_valid, audit


# ---------------------------------------------------------------------------
# diagnostic summaries over cell structures
# ---------------------------------------------------------------------------


def cycle_statistics(cells: Sequence[CellExtraction]) -> dict[str, Any]:
    from collections import Counter

    per_graph = [len(c.cycles) for c in cells]
    sizes: Counter[int] = Counter()
    incidences: list[int] = []
    no = one = multi = 0
    for cell in cells:
        for cycle in cell.cycles:
            sizes[len(cycle)] += 1
        incidences.append(len(incidence_edges(cell.cycles, cell.n_patches)))
        if len(cell.cycles) == 0:
            no += 1
        elif len(cell.cycles) == 1:
            one += 1
        else:
            multi += 1
    per = np.asarray(per_graph, dtype=float)
    return {
        "n_graphs": int(len(cells)),
        "cells_per_graph_mean": float(per.mean()) if len(per) else 0.0,
        "cells_per_graph_median": float(np.median(per)) if len(per) else 0.0,
        "cells_per_graph_max": int(per.max()) if len(per) else 0,
        "total_cells": int(sum(per_graph)),
        "cycle_size_distribution": {str(k): int(v) for k, v in sorted(sizes.items())},
        "incidences_per_graph_mean": float(np.mean(incidences)) if incidences else 0.0,
        "incidences_per_graph_median": float(np.median(incidences)) if incidences else 0.0,
        "incidences_per_graph_max": int(np.max(incidences)) if incidences else 0,
        "graphs_no_cycle": int(no),
        "graphs_one_cycle": int(one),
        "graphs_multiple_cycles": int(multi),
    }


# ---------------------------------------------------------------------------
# parameter audit
# ---------------------------------------------------------------------------


def _build_cell_model(
    *,
    typed_vocabulary_size: int,
    parent_vocabulary_size: int,
    config: Mapping[str, Any],
    seed: int,
    cell_enabled: bool = True,
) -> PatchPathCellModel:
    model_config = config["model"]
    _seed_everything(seed)
    model = PatchPathCellModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        patch_hidden=int(model_config.get("patch_hidden", 48)),
        pair_hidden=int(model_config.get("pair_hidden", 16)),
        token_width=int(model_config.get("token_width", 16)),
        dropout=float(model_config.get("dropout", 0.05)),
        embedding_mode=str(model_config.get("embedding_mode", "hybrid")),
        embedding_rank=int(model_config.get("embedding_rank", 4)),
        hybrid_full_typed_tokens=(
            None
            if model_config.get("hybrid_full_typed_tokens") is None
            else int(model_config["hybrid_full_typed_tokens"])
        ),
        hybrid_full_parent_tokens=(
            None
            if model_config.get("hybrid_full_parent_tokens") is None
            else int(model_config["hybrid_full_parent_tokens"])
        ),
        center_context=bool(model_config.get("center_context", True)),
        center_context_hidden=(
            None
            if model_config.get("center_context_hidden") is None
            else int(model_config["center_context_hidden"])
        ),
        graph_head_hidden_0=(
            None
            if model_config.get("graph_head_hidden_0") is None
            else int(model_config["graph_head_hidden_0"])
        ),
        graph_head_hidden_1=(
            None
            if model_config.get("graph_head_hidden_1") is None
            else int(model_config["graph_head_hidden_1"])
        ),
        topology_mode=str(model_config.get("topology_mode", "hinge")),
        topology_input_width=int(ztopo.raw_width("hinge")),
        topology_hidden_dim=int(model_config.get("topology_hidden_dim", 16)),
        topology_out_dim=int(model_config.get("topology_out_dim", 8)),
        cell_dim=int(CELL_DIM),
        cell_enabled=bool(cell_enabled),
    )
    return model


def parameter_audit() -> dict[str, Any]:
    """Instantiate compact-v4 and compact-v4-cell and decompose parameters."""
    config = cell_config()
    _seed_everything(0)
    base = zpp.PatchPathModel(
        V4_TYPED_VOCAB_WITH_OOV,
        V4_PARENT_VOCAB_WITH_OOV,
        patch_hidden=48,
        pair_hidden=16,
        token_width=16,
        dropout=0.05,
        embedding_mode="hybrid",
        embedding_rank=4,
        hybrid_full_typed_tokens=768,
        hybrid_full_parent_tokens=32,
        center_context=True,
        center_context_hidden=60,
        graph_head_hidden_0=64,
        graph_head_hidden_1=32,
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=16,
        topology_out_dim=8,
    )
    base_total = int(sum(p.numel() for p in base.parameters()))
    model = _build_cell_model(
        typed_vocabulary_size=V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    total = int(sum(p.numel() for p in model.parameters()))

    def block(module: nn.Module | None) -> int:
        if module is None:
            return 0
        return int(sum(p.numel() for p in module.parameters()))

    cell_init_params = block(model.cell_init)
    incidence_operator_params = (
        block(model.cell_update) + block(model.cell_norm) + block(model.patch_update)
    )
    new_head_params = block(model.head)
    base_head_params = block(base.head)
    payload = {
        "baseline_v4_params": base_total,
        "v4_expected_params": V4_PARAMS,
        "baseline_matches_reference": int(base_total) == int(V4_PARAMS),
        "compact_v4_cell_params": total,
        "decomposition": {
            "original_compact_v4_params": int(
                total
                - cell_init_params
                - incidence_operator_params
                - (new_head_params - base_head_params)
            ),
            "cell_initialization_params": int(cell_init_params),
            "incidence_operator_params": int(incidence_operator_params),
            "head_expansion_params": int(new_head_params - base_head_params),
            "new_head_params": int(new_head_params),
            "original_head_params": int(base_head_params),
        },
        "increment_params": int(total - base_total),
        "increment_ratio": float((total - base_total) / max(base_total, 1)),
        "param_ceiling": int(PARAM_CEILING),
        "under_ceiling": bool(total <= PARAM_CEILING),
        "r_original": int(R_ORIGINAL),
        "r_new": int(R_NEW),
        "unified_graph_width": int(model.unified_graph_width_with_cell),
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).detach().cpu().numpy())
            targets.append(batch.y.view(-1).detach().cpu().numpy())
    return (
        np.concatenate(targets) if targets else np.zeros(0),
        np.concatenate(predictions) if predictions else np.zeros(0),
    )


def _mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.size == 0:
        return float("nan")
    return float(np.mean(np.abs(targets - predictions)))


def train_cell(
    *,
    train_data: Sequence[Data],
    valid_data: Sequence[Data],
    typed_vocabulary_size: int,
    parent_vocabulary_size: int,
    config: Mapping[str, Any],
    seed: int,
    tag: str,
    cell_enabled: bool = True,
    save_state: bool = True,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = _build_cell_model(
        typed_vocabulary_size=typed_vocabulary_size,
        parent_vocabulary_size=parent_vocabulary_size,
        config=config,
        seed=seed,
        cell_enabled=cell_enabled,
    ).to(device)
    total_params = int(sum(p.numel() for p in model.parameters()))
    if total_params > PARAM_CEILING:
        raise RuntimeError(
            f"parameter ceiling exceeded: {total_params} > {PARAM_CEILING}"
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(OPTIMIZED_PROTOCOL["learning_rate"]),
        weight_decay=float(OPTIMIZED_PROTOCOL["weight_decay"]),
    )
    loader = _make_cell_loader(
        train_data,
        int(OPTIMIZED_PROTOCOL["batch_size"]),
        True,
        seed + 91011,
    )
    eval_loader = _make_cell_loader(
        valid_data,
        int(OPTIMIZED_PROTOCOL["batch_size"]),
        False,
        seed + 91012,
    )
    if cell_enabled:
        probe = next(iter(loader))
        if int(getattr(probe, "num_cells_total", 0)) <= 0 or not hasattr(
            probe, "cell_batch"
        ):
            raise RuntimeError(
                "cell branch enabled but the cycle-cell incidence tensors are "
                "missing from the collated batch; refusing to train a silently "
                "dormant cell branch"
            )
    steps_per_epoch = int(
        math.ceil(len(train_data) / int(OPTIMIZED_PROTOCOL["batch_size"]))
    )
    cell_param_ids = set()
    for module in (model.cell_init, model.cell_update, model.cell_norm, model.patch_update):
        for parameter in module.parameters():
            cell_param_ids.add(id(parameter))

    patience = int(OPTIMIZED_PROTOCOL["patience"])
    max_epochs = int(OPTIMIZED_PROTOCOL["max_epochs"])
    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        grad_norm_sum = 0.0
        grad_norm_count = 0
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            if cell_param_ids:
                grad_sq = 0.0
                for parameter in model.parameters():
                    if id(parameter) in cell_param_ids and parameter.grad is not None:
                        grad_sq += float(parameter.grad.detach().pow(2).sum())
                grad_norm_sum += math.sqrt(grad_sq)
                grad_norm_count += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(OPTIMIZED_PROTOCOL["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_targets, valid_predictions = _evaluate(model, eval_loader, device)
        valid_mae = _mae(valid_targets, valid_predictions)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_mae),
                "valid_mae": float(valid_mae),
                "lr": float(OPTIMIZED_PROTOCOL["learning_rate"]),
                "optimizer_steps": int(epoch * steps_per_epoch),
                "checkpoint_selected": 0,
                "cell_grad_norm": float(grad_norm_sum / max(grad_norm_count, 1)),
                "cell_state_norm": float(model.last_cell_state_norm),
                "cell_summary_norm": float(model.last_cell_summary_norm),
            }
        )
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs:
            print(
                f"[{tag} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
        if stale >= patience:
            print(
                f"[{tag} seed{seed}] early_stop epoch={epoch} best={best_epoch}",
                flush=True,
            )
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    for row in curve:
        row["checkpoint_selected"] = int(row["epoch"] == best_epoch)
    wall_clock = float(time.perf_counter() - started)

    curve_path = CURVE_DIR / f"{tag}_seed{seed}_curve.csv"
    _write_csv(
        curve_path,
        curve,
        (
            "epoch",
            "train_mae",
            "valid_mae",
            "lr",
            "optimizer_steps",
            "checkpoint_selected",
            "cell_grad_norm",
            "cell_state_norm",
            "cell_summary_norm",
        ),
    )
    state_path = None
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        torch.save(model.state_dict(), state_path)

    valid_targets, valid_predictions = _evaluate(model, eval_loader, device)
    final_valid = _mae(valid_targets, valid_predictions)
    horizon_warning = bool(best_epoch >= max_epochs - max(1, max_epochs // 10))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "seed": int(seed),
        "cell_enabled": bool(cell_enabled),
        "protocol": dict(OPTIMIZED_PROTOCOL),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(final_valid),
        "train_loss_at_best": float(losses[best_epoch - 1]),
        "epochs_run": int(len(losses)),
        "steps_per_epoch": int(steps_per_epoch),
        "optimizer_steps": int(len(losses) * steps_per_epoch),
        "wall_clock_s": wall_clock,
        "early_stopped": bool(len(losses) < max_epochs),
        "horizon_boundary_warning": horizon_warning,
        "parameters": int(total_params),
        "state_path": None if state_path is None else str(state_path),
        "curve_path": str(curve_path),
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(),
        "official_test_loaded": False,
        "valid_predictions": valid_predictions.tolist(),
        "valid_targets": valid_targets.tolist(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "runs" / f"{tag}_seed{seed}.json", summary)
    print(
        f"[{tag} seed{seed}] best_valid={best_mae:.6f} epoch={best_epoch} "
        f"params={total_params} wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# integrity gates
# ---------------------------------------------------------------------------


def _make_synthetic_dataset() -> list[Data]:
    """Small synthetic graphs for the integrity gates (no dataset download)."""
    from tracks.ksvd.code.graph import from_edges, ring_chords

    graphs = [
        ring_chords(6, []),  # benzene-like single 6-cycle
        ring_chords(5, []),  # cyclopentane-like
        from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),  # 4-cycle
        from_edges(6, [(0, 1), (1, 2), (2, 0)]),  # triangle + isolated nodes
        from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),  # path, no cycle
    ]
    data_list: list[Data] = []
    for graph in graphs:
        nodes = list(graph.nodes)
        edges = list(graph.edges())
        bi_edges = edges + [(v, u) for u, v in edges]
        edge_index = torch.tensor(bi_edges, dtype=torch.long).t().contiguous()
        n = len(nodes)
        data = Data(
            edge_index=edge_index,
            x=torch.zeros(n, dtype=torch.long),
            edge_attr=torch.zeros(len(bi_edges), dtype=torch.long),
            y=torch.tensor([0.0]),
            num_nodes=n,
        )
        data_list.append(data)
    return data_list


def _mature_states_for_batch(model: "PatchPathCellModel", batch: Batch) -> torch.Tensor:
    with torch.no_grad():
        model.encode_original(batch)
        return model.mature_patch_states()


def integrity_gates(report: bool = True) -> dict[str, Any]:
    results: dict[str, bool] = {}
    gates: dict[str, Any] = {}

    results["G0.7_r_dimension_399"] = int(R_NEW) == 399 and int(R_ORIGINAL) == 302

    config = cell_config()
    model = _build_cell_model(
        typed_vocabulary_size=V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    model.eval()  # deterministic inference for every numerical gate
    total_params = int(sum(p.numel() for p in model.parameters()))
    results["G0.8_params_under_ceiling"] = total_params <= PARAM_CEILING
    gates["G0.8_total_params"] = total_params

    # G0.2 cycle enumeration invariant to node relabelling
    from tracks.ksvd.code.graph import from_edges

    inv_ok = True
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(6, 12))
        edges = [(i, (i + 1) % n) for i in range(n)]
        for _ in range(int(rng.integers(0, 3))):
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            edges.append((int(a), int(b)))
        graph = from_edges(n, edges)
        reference = induced_cycles(graph, MAX_CYCLE_LEN)
        perm = rng.permutation(n)
        mapping = {i: int(perm[i]) for i in range(n)}
        relabelled = from_edges(n, [(mapping[u], mapping[v]) for u, v in edges])
        inverse = {v: k for k, v in mapping.items()}
        back = sorted(
            tuple(sorted(inverse[x] for x in cycle))
            for cycle in induced_cycles(relabelled, MAX_CYCLE_LEN)
        )
        if back != reference:
            inv_ok = False
            break
    results["G0.2_relabel_invariant"] = inv_ok

    # G0.3 deterministic cycle count
    stable = True
    synthetic = _make_synthetic_dataset()
    for data in synthetic:
        if cycles_for_data(data) != cycles_for_data(data):
            stable = False
    results["G0.3_deterministic_cycle_count"] = stable

    encoded = _encode_synthetic(model, synthetic)
    batch0 = cell_collate([encoded[0]])
    with torch.no_grad():
        r_new = model.encode(batch0)
    results["G0.7_r_dimension_399"] = results["G0.7_r_dimension_399"] and int(
        r_new.shape[1]
    ) == 399
    gates["G0.7_encoded_dim"] = int(r_new.shape[1])

    # G0.1 original v4 pathway numerically identical to compact-v4 (shared
    # weights) with the cell branch present, i.e. the base pathway is untouched.
    base = zpp.PatchPathModel(
        V4_TYPED_VOCAB_WITH_OOV,
        V4_PARENT_VOCAB_WITH_OOV,
        patch_hidden=48,
        pair_hidden=16,
        token_width=16,
        dropout=0.05,
        embedding_mode="hybrid",
        embedding_rank=4,
        hybrid_full_typed_tokens=768,
        hybrid_full_parent_tokens=32,
        center_context=True,
        center_context_hidden=60,
        graph_head_hidden_0=64,
        graph_head_hidden_1=32,
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=16,
        topology_out_dim=8,
    )
    base.eval()
    cell_state = model.state_dict()
    base_state = base.state_dict()
    shared = {
        key: value
        for key, value in cell_state.items()
        if key in base_state and base_state[key].shape == value.shape
    }
    base.load_state_dict(shared, strict=False)
    diff = 0.0
    with torch.no_grad():
        for data in encoded:
            single = cell_collate([data])
            original_cell = model.encode_original(single)
            original_base = base.encode(single)
            diff = max(diff, float((original_cell - original_base).abs().max()))
    results["G0.1_original_pathway_identical"] = diff < 1e-5
    gates["G0.1_max_abs_diff"] = diff

    # G0.4 cell pooling invariant to cycle ordering
    pool_ok = True
    for data in encoded:
        if int(data.num_cells.item()) == 0:
            continue
        single = cell_collate([data])
        patch = _mature_states_for_batch(model, single)
        summary = model._cell_summary(patch, single)
        permuted = _permute_cells(data)
        permuted_batch = cell_collate([permuted])
        summary_permuted = model._cell_summary(patch, permuted_batch)
        if float((summary - summary_permuted).abs().max()) > 1.0e-6:
            pool_ok = False
    results["G0.4_pooling_order_invariant"] = pool_ok

    # G0.5 incidence permutation bookkeeping
    bookkeeping_ok = True
    for data in encoded:
        n = int(data.num_nodes)
        n_cells = int(data.num_cells.item())
        if n_cells > 0:
            if data.cell_edge_patch.numel() and (
                int(data.cell_edge_patch.min()) < 0
                or int(data.cell_edge_patch.max()) >= n
            ):
                bookkeeping_ok = False
            if data.cell_edge_cell.numel() and (
                int(data.cell_edge_cell.min()) < 0
                or int(data.cell_edge_cell.max()) >= n_cells
            ):
                bookkeeping_ok = False
            if data.cell_edge_patch.shape[0] != data.cell_edge_cell.shape[0]:
                bookkeeping_ok = False
    results["G0.5_incidence_bookkeeping"] = bookkeeping_ok

    # G0.6 no-cycle graph forward finite and cell summary exactly zero
    no_cycle_ok = True
    no_cycle_data = next(
        (d for d in encoded if int(d.num_cells.item()) == 0), None
    )
    if no_cycle_data is None:
        no_cycle_ok = False
    else:
        single = cell_collate([no_cycle_data])
        with torch.no_grad():
            out = model(single)
            summary = model._cell_summary(model.mature_patch_states(), single)
        no_cycle_ok = bool(torch.isfinite(out).all()) and bool(
            torch.allclose(summary, torch.zeros_like(summary))
        )
    results["G0.6_no_cycle_finite_zero"] = no_cycle_ok

    # exactly two incidence rounds / no extra message passing / topology kept
    source = Path(__file__).read_text(encoding="utf-8")
    results["G0.9_exactly_two_rounds"] = "CELL_ROUNDS = 2" in source
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    )
    results["G0.10_no_extra_message_passing"] = not any(
        token in import_lines
        for token in ("MessagePassing", "GATConv", "TransformerConv", "MultiheadAttention")
    )
    results["G0.11_global_topology_present"] = bool(
        getattr(model, "topology_encoder", None) is not None
        and str(model.topology_mode) != "none"
    )

    gates["results"] = results
    gates["passed"] = all(results.values())
    if report:
        _write_json(RESULTS_DIR / "integrity_gates.json", gates)
    return gates


def _encode_synthetic(
    model: "PatchPathCellModel", data_list: Sequence[Data]
) -> list[Data]:
    """Attach the minimal base tensors required by the model to synthetic graphs."""
    encoded: list[Data] = []
    for data in data_list:
        graph, _nt, _et = _data_to_graph(data)
        cycles = induced_cycles(graph, MAX_CYCLE_LEN)
        edges = incidence_edges(cycles, int(data.num_nodes))
        n = int(data.num_nodes)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs
            else torch.zeros(2, 0, dtype=torch.long)
        )
        encoded.append(
            Data(
                patch_cont=torch.zeros(n, int(model.shell_width)),
                patch_context=torch.zeros(n, 0),
                typed_token=torch.zeros(n, dtype=torch.long),
                parent_token=torch.zeros(n, dtype=torch.long),
                structural_token=torch.zeros(n, dtype=torch.long),
                structural_coarse=torch.zeros(n, 0),
                pair_index=pair_index,
                pair_relation=torch.zeros(len(pairs), zpp.RELATION_WIDTH),
                pair_bucket=torch.zeros(len(pairs), dtype=torch.long),
                global_context=torch.zeros(1, zpp.GLOBAL_WIDTH),
                topology_features=torch.zeros(1, 25),
                y=torch.tensor([0.0]),
                num_nodes=n,
                cell_edge_patch=torch.tensor(
                    [e[0] for e in edges], dtype=torch.long
                ),
                cell_edge_cell=torch.tensor(
                    [e[1] for e in edges], dtype=torch.long
                ),
                num_cells=torch.tensor([len(cycles)], dtype=torch.long),
            )
        )
    return encoded


def _permute_cells(data: Data) -> Data:
    """Return a copy of a single (unbatched) Data with the cell indices
    reversed (a canonical relabelling of cell identity), keeping incidence
    consistent.  Used only for the pooling-invariance gate."""
    n_cells = int(data.num_cells.item())
    if n_cells < 2:
        return data
    mapping = {i: n_cells - 1 - i for i in range(n_cells)}
    new = copy.copy(data)
    new.cell_edge_cell = torch.tensor(
        [mapping[int(c)] for c in data.cell_edge_cell.tolist()],
        dtype=torch.long,
    )
    new.cell_edge_patch = data.cell_edge_patch.clone()
    new.num_cells = data.num_cells.clone()
    return new


# ---------------------------------------------------------------------------
# baseline inventory
# ---------------------------------------------------------------------------


def baseline_inventory() -> dict[str, Any]:
    payload = {
        "protocol": dict(OPTIMIZED_PROTOCOL),
        "optimized_v4_seed0_valid": V4_SEED0_VALID,
        "optimized_v4_seed1_valid": V4_SEED1_VALID,
        "optimized_v4_seed0_epoch": V4_SEED0_EPOCH,
        "optimized_v4_seed1_epoch": V4_SEED1_EPOCH,
        "optimized_v4_params": V4_PARAMS,
        "v4_seed0_state": str(V4_SEED0_STATE),
        "v4_seed0_state_exists": V4_SEED0_STATE.exists(),
        "v4_seed1_state": str(V4_SEED1_STATE),
        "v4_seed1_state_exists": V4_SEED1_STATE.exists(),
        "v4_seed0_state_sha256": (
            _sha256_file(V4_SEED0_STATE) if V4_SEED0_STATE.exists() else None
        ),
        "v4_seed1_state_sha256": (
            _sha256_file(V4_SEED1_STATE) if V4_SEED1_STATE.exists() else None
        ),
        "canonical_config": str(CANONICAL_V4_CONFIG),
        "canonical_config_sha256": _sha256_file(CANONICAL_V4_CONFIG),
        "tokenizer_version": TYPED_TOKENIZER_V1_HISTORICAL,
        "tokenizer_fingerprint": typed_tokenizer_fingerprint(
            TYPED_TOKENIZER_V1_HISTORICAL, PATCH_RADIUS
        ),
        "topology": {
            "feature_version": ztopo.FEATURE_VERSION,
            "input_width": int(ztopo.raw_width("hinge")),
            "mode": "hinge",
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "baseline_inventory.json", payload)
    return payload


# ---------------------------------------------------------------------------
# cycle construction lock
# ---------------------------------------------------------------------------


def cycle_construction_lock() -> dict[str, Any]:
    _t, valid_bundle, _meta = extract_records()
    stats = cycle_statistics(valid_bundle["cells"])
    payload = {
        "cycle_family": "induced (chordless) simple cycles",
        "enumeration": "exact min-node DFS (structural_context._find_cycles) filtered to chordless",
        "max_cycle_len": int(MAX_CYCLE_LEN),
        "cycle_types": "ALL induced cycles of length 3..8 (no target/residual/valid selection)",
        "canonical_form": "sorted vertex tuple; cell list sorted lexicographically",
        "permutation_invariant": True,
        "target_independent": True,
        "deduplication": "min-node DFS records each simple cycle once; chordless filter deterministic",
        "valid_split_statistics": stats,
        "rationale": (
            "k=8 is the pre-registered bound: it is a strict superset of the "
            "plan's k=6 option, covers all observed ZINC induced ring sizes "
            "(3-8; induced cycles longer than 8 occur in <0.1% of graphs and "
            "are excluded by the fixed bound, not by target performance), and "
            "the bound is saturated at 8 on the valid split. No sweep."
        ),
    }
    _write_json(RESULTS_DIR / "cycle_construction_lock.json", payload)
    return payload


# ---------------------------------------------------------------------------
# architecture lock
# ---------------------------------------------------------------------------


def architecture_lock() -> dict[str, Any]:
    config = cell_config()
    model = _build_cell_model(
        typed_vocabulary_size=V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    payload = {
        "name": "compact-v4-cell",
        "protocol_version": PROTOCOL_VERSION,
        "single_principle_changed": (
            "incidence-preserving structural persistence: explicit persistent "
            "cycle-cell objects (48D) updated over exactly 2 pre-registered "
            "incidence-aware rounds"
        ),
        "base_architecture": {
            "preserved": [
                "patch radius 2",
                "historical aliased rooted-topology tokenizer",
                "patch encoder (146D shell + 16D typed + 8D parent -> 48D)",
                "complete-pair encoder + relation encoder",
                "centre aggregation + one-shot centre update",
                "unary moments 97 + pair moments 165 + global 32",
                "v4 global topology hinge branch (25 -> 16 -> 8)",
                "R_original = 302D",
            ],
            "r_original": int(R_ORIGINAL),
        },
        "cycle_construction": {
            "family": "induced (chordless) cycles",
            "max_cycle_len": int(MAX_CYCLE_LEN),
            "source": "target-independent graph structure",
        },
        "cell_state": {
            "dim": int(CELL_DIM),
            "init": "CellInit(mean of boundary patch states) = Linear(48,48)+LayerNorm+ReLU",
            "identity": (
                "each cell keeps an independent state c_k across both rounds; "
                "no graph-level pooling before both rounds complete"
            ),
        },
        "incidence": {
            "I_PC": (
                "patch (atom-centred lower object) <-> cycle cell; patch index == "
                "centre atom index"
            ),
            "I_QC_used": False,
            "I_QC_note": (
                "the minimal plan does not include a relation<->cell incidence, "
                "so no complete-pair relation is treated as a cycle boundary "
                "relation; the cell boundary is its atoms"
            ),
        },
        "rounds": {
            "T": int(CELL_ROUNDS),
            "definition": "one round = (lower->cell, then cell->lower)",
            "equations": [
                "agg_c = mean_{i in boundary(c)} p_i^{r-1}",
                "c_c^r = LayerNorm_c(c_c^{r-1} + W_c relu(agg_c))",
                "agg_i = mean_{c containing i} c_c^r",
                "p_i^r = p_i^{r-1} + alive_i * W_p relu(agg_i)",
            ],
            "shared_params": "W_c, W_p shared across both rounds; CellInit applied once",
            "rank_interpretation": (
                "operational labels only: lower object = patch-centred structural "
                "object; higher object = explicit cycle cell. The complete-pair "
                "relation q_ij is NOT asserted to be a CW rank-1 edge."
            ),
        },
        "cell_summary": {
            "format": "[mean(c); std(c); log1p(count)]",
            "width": int(CELL_SUMMARY_WIDTH),
            "empty_cycle_representation": "all zeros (mean=std=logcount=0)",
        },
        "final_representation": {
            "R_new": int(R_NEW),
            "composition": "[R_original (302); S_cell (97)]",
        },
        "graph_head": {
            "input_width": int(model.unified_graph_width_with_cell),
            "hidden": [int(model.head_hidden_0), int(model.head_hidden_1)],
            "structure": "unchanged compact-v4 head shape; only input width moved",
        },
        "parameter_budget": {
            "ceiling": int(PARAM_CEILING),
            "target_range": [108000, 112000],
        },
        "training_protocol": dict(OPTIMIZED_PROTOCOL),
        "decision_thresholds": {
            "arch_gate_seed0": ARCH_GATE,
            "arch_gate_seed1": ARCH_GATE_SEED1,
            "arch_mean_gate": ARCH_MEAN_GATE,
            "inc_gate_seed0": INC_GATE,
            "inc_mean_gate": INC_MEAN_GATE,
            "bulk_gate": BULK_GATE,
        },
        "forbidden": [
            "depth sweep (T fixed = 2)",
            "attention / transformer",
            "generic node message passing",
            "per-cycle-length MLPs",
            "removing the global topology branch",
            "target-derived features",
            "official test access",
        ],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "architecture_lock.json", payload)
    return payload


# ---------------------------------------------------------------------------
# compute audit
# ---------------------------------------------------------------------------


def compute_audit() -> dict[str, Any]:
    config = cell_config()
    model = _build_cell_model(
        typed_vocabulary_size=V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    model.eval()
    synthetic = _encode_synthetic(model, _make_synthetic_dataset())
    batch = cell_collate(synthetic)
    operator_calls = {
        "cell_init": 1,
        "cell_update": int(CELL_ROUNDS),
        "patch_update": int(CELL_ROUNDS),
        "lower_aggregation": int(CELL_ROUNDS) + 1,
        "cell_aggregation": int(CELL_ROUNDS),
    }
    with torch.no_grad():
        for _ in range(3):
            model(batch)
        started = time.perf_counter()
        for _ in range(10):
            model(batch)
        forward_ms = (time.perf_counter() - started) / 10.0 * 1000.0
    payload = {
        "synthetic_graphs": len(synthetic),
        "cell_operator_application_counts": operator_calls,
        "synthetic_forward_ms": float(forward_ms),
        "n_cells_synthetic": int(batch.num_cells_total),
        "n_patches_synthetic": int(batch.num_nodes),
        "note": "compute is reported separately from parameter count; see note",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compute_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage orchestration
# ---------------------------------------------------------------------------


def stage1_true_seed0() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records(mode="true")
    config = cell_config()
    summary = train_cell(
        train_data=train_data,
        valid_data=valid_data,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=0,
        tag="true_cell",
        cell_enabled=True,
    )
    baseline = V4_SEED0_VALID
    summary["baseline_v4_valid"] = float(baseline)
    summary["delta_arch0"] = float(baseline - summary["best_valid_mae"])
    summary["arch_gate"] = float(ARCH_GATE)
    summary["passes_arch_gate"] = bool(summary["delta_arch0"] >= ARCH_GATE)
    _write_json(RESULTS_DIR / "stage1_true_seed0.json", summary)
    return summary


def _branch_dependency_check(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Inference-only branch-alive sanity from the saved checkpoint."""
    train_data, valid_data, audit = build_encoded_records(mode="true")
    config = cell_config()
    model = _build_cell_model(
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=0,
    )
    state = torch.load(summary["state_path"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    loader = _make_cell_loader(valid_data, 128, False, 0)
    max_zero_diff = 0.0
    with torch.no_grad():
        for batch in loader:
            base = model(batch)
            # zero the cell summary
            original_encode = model._cell_summary

            def _zero_summary(patch, data):
                return torch.zeros_like(original_encode(patch, data))

            model._cell_summary = _zero_summary  # type: ignore[assignment]
            zeroed = model(batch)
            model._cell_summary = original_encode  # type: ignore[assignment]
            max_zero_diff = max(
                max_zero_diff, float((base - zeroed).abs().max())
            )
            break
    return {
        "zero_cell_summary_max_abs_diff": float(max_zero_diff),
        "branch_alive": bool(max_zero_diff > 1e-3),
    }


def _rare_le5_ratio(
    records: Sequence[Any], train_freq: Mapping[bytes, int]
) -> np.ndarray:
    out = []
    for record in records:
        freqs = [int(train_freq.get(patch.typed_certificate, 0)) for patch in record.patches]
        out.append(float(np.mean(np.asarray(freqs) <= 5)) if freqs else 0.0)
    return np.asarray(out, dtype=np.float64)


def _validate_true_predictions(summary: Mapping[str, Any], valid_data):
    """Return (targets, v4_predictions, true_predictions) on the valid split."""
    from collections import Counter

    train_bundle, valid_bundle, _meta = extract_records()
    train_freq: Counter[bytes] = Counter()
    for record in train_bundle["records"]:
        for patch in record.patches:
            train_freq[patch.typed_certificate] += 1
    train_rare = _rare_le5_ratio(train_bundle["records"], train_freq)
    valid_rare = _rare_le5_ratio(valid_bundle["records"], train_freq)

    targets = np.asarray(summary["valid_targets"], dtype=np.float64)
    true_pred = np.asarray(summary["valid_predictions"], dtype=np.float64)

    base = zpp.PatchPathModel(
        V4_TYPED_VOCAB_WITH_OOV,
        V4_PARENT_VOCAB_WITH_OOV,
        patch_hidden=48,
        pair_hidden=16,
        token_width=16,
        dropout=0.05,
        embedding_mode="hybrid",
        embedding_rank=4,
        hybrid_full_typed_tokens=768,
        hybrid_full_parent_tokens=32,
        center_context=True,
        center_context_hidden=60,
        graph_head_hidden_0=64,
        graph_head_hidden_1=32,
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=16,
        topology_out_dim=8,
    )
    base.load_state_dict(torch.load(V4_SEED0_STATE, map_location="cpu"))
    base.eval()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    device = torch.device("cpu")
    v4_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            v4_pred.append(base(batch).cpu().numpy())
    v4_pred = np.concatenate(v4_pred)
    return targets, v4_pred, true_pred, train_rare, valid_rare


def common_input_bulk() -> dict[str, Any]:
    summary = _read_json(RESULTS_DIR / "stage1_true_seed0.json")
    _train, valid_data, _audit = build_encoded_records(mode="true")
    targets, v4_pred, true_pred, train_rare, valid_rare = _validate_true_predictions(
        summary, valid_data
    )
    threshold = float(np.percentile(train_rare, 80.0))
    bulk = valid_rare < threshold
    v4_mae = float(np.mean(np.abs(targets - v4_pred)))
    true_mae = float(np.mean(np.abs(targets - true_pred)))
    bulk_v4 = float(np.mean(np.abs(targets[bulk] - v4_pred[bulk]))) if bulk.any() else float("nan")
    bulk_true = (
        float(np.mean(np.abs(targets[bulk] - true_pred[bulk]))) if bulk.any() else float("nan")
    )
    payload = {
        "bulk_definition": (
            "target-independent: valid molecules whose train-derived "
            "rare_le5_ratio (fraction of patches with typed-token train "
            "frequency <= 5) is below the 80th percentile of the train split"
        ),
        "rare_le5_threshold": threshold,
        "n_bulk_valid": int(bulk.sum()),
        "n_valid": int(len(valid_rare)),
        "v4_valid_mae": v4_mae,
        "true_cell_valid_mae": true_mae,
        "valid_mae_diff_true_minus_v4": float(true_mae - v4_mae),
        "bulk_v4_mae": bulk_v4,
        "bulk_true_cell_mae": bulk_true,
        "bulk_mae_diff_true_minus_v4": float(bulk_true - bulk_v4),
        "bulk_gate": float(BULK_GATE),
        "bulk_safe": bool((bulk_true - bulk_v4) <= BULK_GATE),
        "v4_reference_valid": float(V4_SEED0_VALID),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "common_input_bulk.json", payload)
    return payload


def mechanism_diagnostics() -> dict[str, Any]:
    """Per-round numerical health of the trained true-cell seed0 branch."""
    summary = _read_json(RESULTS_DIR / "stage1_true_seed0.json")
    _train, valid_data, audit = build_encoded_records(mode="true")
    config = cell_config()
    model = _build_cell_model(
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=0,
    )
    model.load_state_dict(torch.load(summary["state_path"], map_location="cpu"))
    model.eval()
    loader = _make_cell_loader(valid_data, 128, False, 0)
    cell_norms: list[list[float]] = [[], []]
    lower_norms: list[list[float]] = [[], []]
    init_norms: list[float] = []
    summary_norms: list[float] = []
    state_norms: list[float] = []
    with torch.no_grad():
        for batch in loader:
            model(batch)
            for round_index in range(CELL_ROUNDS):
                cell_norms[round_index].append(model.round_cell_state_norms[round_index])
                lower_norms[round_index].append(model.round_lower_update_norms[round_index])
            init_norms.append(model.cell_init_norm)
            summary_norms.append(model.last_cell_summary_norm)
            state_norms.append(model.last_cell_state_norm)
    payload = {
        "checkpoint": summary["state_path"],
        "split": "official-valid",
        "cell_init_norm": float(np.mean(init_norms)),
        "cell_state_norm_by_round": [float(np.mean(x)) for x in cell_norms],
        "lower_update_norm_by_round": [float(np.mean(x)) for x in lower_norms],
        "cell_state_norm_final": float(np.mean(state_norms)),
        "cell_summary_norm": float(np.mean(summary_norms)),
        "rounds": int(CELL_ROUNDS),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism_diagnostics.json", payload)
    return payload


def stage1_decision() -> dict[str, Any]:
    summary = _read_json(RESULTS_DIR / "stage1_true_seed0.json")
    dependency = _branch_dependency_check(summary)
    bulk = common_input_bulk()
    delta = float(summary["delta_arch0"])
    passed = bool(delta >= ARCH_GATE)
    decision = {
        "stage": "stage1_architecture_gate",
        "delta_arch0": delta,
        "threshold": float(ARCH_GATE),
        "branch_dependency": dependency,
        "cell_branch_alive": bool(dependency["branch_alive"]),
        "common_input_bulk_safe": bool(bulk["bulk_safe"]),
        "common_input_bulk": bulk,
        "passed": passed,
        "classification": (
            "ARCHITECTURE PASS - proceed to Stage 2 broken-incidence control"
            if passed
            else "CELL REPRESENTATION NOT SUPPORTED UNDER LOCKED PROTOCOL (Case A stop)"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage1_decision.json", decision)
    return decision


def stage2_broken_seed0() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records(mode="broken")
    config = cell_config()
    summary = train_cell(
        train_data=train_data,
        valid_data=valid_data,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=0,
        tag="broken_cell",
        cell_enabled=True,
    )
    manifest = {
        "mode": "broken",
        "seed": 0,
        "train": audit["cell_incidence"]["train"],
        "valid": audit["cell_incidence"]["valid"],
    }
    _write_json(RESULTS_DIR / "broken_incidence_manifest_seed0.json", manifest)
    true_summary = _read_json(RESULTS_DIR / "stage1_true_seed0.json")
    delta_inc0 = float(summary["best_valid_mae"] - true_summary["best_valid_mae"])
    summary["true_valid"] = float(true_summary["best_valid_mae"])
    summary["delta_inc0"] = delta_inc0
    summary["inc_gate"] = float(INC_GATE)
    summary["passes_inc_gate"] = bool(delta_inc0 >= INC_GATE)
    _write_json(RESULTS_DIR / "stage2_broken_seed0.json", summary)
    decision = {
        "stage": "stage2_mechanism_gate_seed0",
        "delta_arch0": float(true_summary["delta_arch0"]),
        "delta_inc0": delta_inc0,
        "inc_threshold": float(INC_GATE),
        "case": (
            "SEED0 EXPLICIT CYCLE-INCIDENCE SIGNAL (Case C)"
            if delta_inc0 >= INC_GATE
            else "CELL ARCHITECTURE GO - INCIDENCE MECHANISM NOT SUPPORTED (Case B)"
        ),
        "passed": bool(delta_inc0 >= INC_GATE),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage2_mechanism_decision.json", decision)
    return decision


def stage3_true_seed1() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records(mode="true")
    config = cell_config()
    summary = train_cell(
        train_data=train_data,
        valid_data=valid_data,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=1,
        tag="true_cell",
        cell_enabled=True,
    )
    summary["baseline_v4_valid"] = float(V4_SEED1_VALID)
    summary["delta_arch1"] = float(V4_SEED1_VALID - summary["best_valid_mae"])
    _write_json(RESULTS_DIR / "stage3_true_seed1.json", summary)
    return summary


def stage3a_decision() -> dict[str, Any]:
    s0 = _read_json(RESULTS_DIR / "stage1_true_seed0.json")
    s1 = _read_json(RESULTS_DIR / "stage3_true_seed1.json")
    d0 = float(s0["delta_arch0"])
    d1 = float(s1["delta_arch1"])
    mean = (d0 + d1) / 2.0
    passed = bool(d1 >= ARCH_GATE_SEED1 and mean >= ARCH_MEAN_GATE and d0 > 0 and d1 > 0)
    decision = {
        "stage": "stage3a_architecture_replication",
        "delta_arch0": d0,
        "delta_arch1": d1,
        "mean": mean,
        "thresholds": {"seed1": float(ARCH_GATE_SEED1), "mean": float(ARCH_MEAN_GATE)},
        "passed": passed,
        "classification": (
            "ARCHITECTURE REPLICATED - proceed to Stage 3B"
            if passed
            else "ARCHITECTURE REPLICATION INCONCLUSIVE (Case E stop)"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage3a_replication_decision.json", decision)
    return decision


def stage3_broken_seed1() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records(mode="broken")
    config = cell_config()
    summary = train_cell(
        train_data=train_data,
        valid_data=valid_data,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=1,
        tag="broken_cell",
        cell_enabled=True,
    )
    manifest = {
        "mode": "broken",
        "seed": 1,
        "train": audit["cell_incidence"]["train"],
        "valid": audit["cell_incidence"]["valid"],
    }
    _write_json(RESULTS_DIR / "broken_incidence_manifest_seed1.json", manifest)
    true1 = _read_json(RESULTS_DIR / "stage3_true_seed1.json")
    delta_inc1 = float(summary["best_valid_mae"] - true1["best_valid_mae"])
    summary["true_valid"] = float(true1["best_valid_mae"])
    summary["delta_inc1"] = delta_inc1
    _write_json(RESULTS_DIR / "stage3_broken_seed1.json", summary)
    inc0 = float(_read_json(RESULTS_DIR / "stage2_broken_seed0.json")["delta_inc0"])
    mean = (inc0 + delta_inc1) / 2.0
    passed = bool(delta_inc1 > 0 and mean >= INC_MEAN_GATE)
    decision = {
        "stage": "stage3b_mechanism_replication",
        "delta_inc0": inc0,
        "delta_inc1": delta_inc1,
        "mean": mean,
        "passed": passed,
        "classification": (
            "EXPLICIT PERSISTENT CYCLE-CELL INCIDENCE - REPLICATED GO (Case D)"
            if passed
            else "MECHANISM REPLICATION FAILED"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage3b_mechanism_replication.json", decision)
    return decision


def final_decision() -> dict[str, Any]:
    stage1 = _read_json(RESULTS_DIR / "stage1_decision.json")
    payload = {
        "decision_case": None,
        "case_a_architecture_nogo": not bool(stage1["passed"]),
        "notes": [],
        "official_test_loaded": False,
    }
    if not stage1["passed"]:
        payload["decision_case"] = "A"
        payload["decision"] = "CELL REPRESENTATION NO-GO"
        payload["notes"].append(
            "true-cell seed0 did not reach +0.004 over optimized-v4 seed0"
        )
    else:
        stage2 = _read_json(RESULTS_DIR / "stage2_mechanism_decision.json")
        if not stage2["passed"]:
            payload["decision_case"] = "B"
            payload["decision"] = (
                "CELL ARCHITECTURE GO - INCIDENCE MECHANISM NOT SUPPORTED"
            )
        else:
            stage3a_path = RESULTS_DIR / "stage3a_replication_decision.json"
            if not stage3a_path.exists():
                payload["decision_case"] = "C"
                payload["decision"] = "SEED0 EXPLICIT CYCLE-INCIDENCE SIGNAL"
            else:
                stage3a = _read_json(stage3a_path)
                if not stage3a["passed"]:
                    payload["decision_case"] = "E"
                    payload["decision"] = "REPLICATION INCONCLUSIVE"
                else:
                    stage3b_path = RESULTS_DIR / "stage3b_mechanism_replication.json"
                    if not stage3b_path.exists():
                        payload["decision_case"] = "C"
                        payload["decision"] = (
                            "SEED0 SIGNAL; seed1 architecture replicated, "
                            "mechanism replication pending"
                        )
                    else:
                        stage3b = _read_json(stage3b_path)
                        if stage3b["passed"]:
                            payload["decision_case"] = "D"
                            payload["decision"] = (
                                "EXPLICIT PERSISTENT CYCLE-CELL INCIDENCE - "
                                "REPLICATED GO"
                            )
                        else:
                            payload["decision_case"] = "B"
                            payload["decision"] = (
                                "CELL ARCHITECTURE GO - INCIDENCE MECHANISM "
                                "NOT REPLICATED"
                            )
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


def two_seed_summary() -> dict[str, Any]:
    rows = []
    for path, label in (
        (RESULTS_DIR / "stage1_true_seed0.json", "true_cell_seed0"),
        (RESULTS_DIR / "stage2_broken_seed0.json", "broken_cell_seed0"),
        (RESULTS_DIR / "stage3_true_seed1.json", "true_cell_seed1"),
        (RESULTS_DIR / "stage3_broken_seed1.json", "broken_cell_seed1"),
    ):
        if path.exists():
            payload = _read_json(path)
            rows.append(
                {
                    "model": label,
                    "seed": payload.get("seed", ""),
                    "valid_mae": payload.get("best_valid_mae", ""),
                    "best_epoch": payload.get("best_epoch", ""),
                    "params": payload.get("parameters", ""),
                }
            )
    path = RESULTS_DIR / "final_two_seed_summary.csv"
    _write_csv(path, rows, ("model", "seed", "valid_mae", "best_epoch", "params"))
    return {"rows": rows}


def make_figures() -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return {"figures": [], "error": repr(exc)}
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # Figure 1: computation graph schematic
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.9, "compact-v4-cell", ha="center", fontsize=13, weight="bold")
    ax.text(0.12, 0.62, "patch lower objects\n$h'_i$ (48D)", ha="center", fontsize=9)
    ax.text(0.88, 0.62, "cycle cells\n$c_k$ (48D)", ha="center", fontsize=9)
    ax.annotate("", xy=(0.78, 0.6), xytext=(0.22, 0.6), arrowprops={"arrowstyle": "<->"})
    ax.text(0.5, 0.66, "incidence rounds (T = 2)", ha="center", fontsize=9)
    ax.text(0.5, 0.42, "cell moments [mean; std; log1p(count)] = 97D", ha="center", fontsize=9)
    ax.text(0.5, 0.28, "R = [R_original 302 ; S_cell 97] = 399D", ha="center", fontsize=9)
    ax.text(0.5, 0.12, "existing compact-v4 head -> MAE", ha="center", fontsize=9)
    path = FIGURE_DIR / "figure1_computation_graph.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))

    # Figure 2: true vs broken incidence schematic
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    true_edges = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)]
    broken_edges = [(0, 0), (1, 0), (3, 0), (2, 1), (4, 1), (5, 1)]
    for ax, edges, title in (
        (axes[0], true_edges, "true incidence"),
        (axes[1], broken_edges, "broken incidence\n(degree-preserving)"),
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.9, 1.9)
        ax.axis("off")
        for (p, c), color in zip(edges, ["#1f77b4"] * 3 + ["#ff7f0e"] * 3):
            ax.plot([p, 0.5 + c * 3.0], [0.0, 1.0], "-", color=color, alpha=0.6)
        for p in range(6):
            ax.plot(p, 0.0, "o", color="#444")
            ax.text(p, -0.35, f"p{p}", ha="center", fontsize=7)
        for c in range(2):
            ax.plot(0.5 + c * 3.0, 1.0, "s", color="#c44")
            ax.text(0.5 + c * 3.0, 1.25, f"c{c}", ha="center", fontsize=8)
    path = FIGURE_DIR / "figure2_true_vs_broken_incidence.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))

    # Figure 3: valid MAE bars (only models that actually ran)
    labels: list[str] = ["optimized v4"]
    values: list[float] = [V4_SEED0_VALID]
    for path_, label in (
        (RESULTS_DIR / "stage1_true_seed0.json", "true-cell seed0"),
        (RESULTS_DIR / "stage2_broken_seed0.json", "broken-cell seed0"),
        (RESULTS_DIR / "stage3_true_seed1.json", "true-cell seed1"),
        (RESULTS_DIR / "stage3_broken_seed1.json", "broken-cell seed1"),
    ):
        if path_.exists():
            labels.append(label)
            values.append(float(_read_json(path_)["best_valid_mae"]))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(len(labels)), values, color="#4c72b0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("official-valid MAE")
    ax.set_title("compact-v4-cell valid MAE", fontsize=10)
    path = FIGURE_DIR / "figure3_valid_mae.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))

    return {"figures": created}


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def diagnostics() -> dict[str, Any]:
    _t, valid_bundle, _meta = extract_records()
    stats = cycle_statistics(valid_bundle["cells"])
    _write_json(RESULTS_DIR / "cell_statistics_valid.json", stats)
    return stats


def run_all() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    architecture_lock()
    cycle_construction_lock()
    baseline_inventory()
    gate = integrity_gates()
    if not gate["passed"]:
        raise RuntimeError(f"integrity gates failed: {gate['results']}")
    parameter_audit()
    compute_audit()
    diagnostics()
    make_figures()
    print("Stage 0 complete.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "protocol_lock",
            "cycle_lock",
            "extract",
            "param_audit",
            "compute_audit",
            "integrity",
            "baseline",
            "diagnostics",
            "figures",
            "stage1_true_seed0",
            "stage1_decision",
            "stage2_broken_seed0",
            "stage2_decision",
            "stage3_true_seed1",
            "stage3a_decision",
            "stage3_broken_seed1",
            "stage3b_decision",
            "final",
            "two_seed_summary",
            "all",
        ],
    )
    args = parser.parse_args(argv)
    stage = args.stage
    _configure_determinism()
    if stage == "protocol_lock":
        architecture_lock()
    elif stage == "cycle_lock":
        cycle_construction_lock()
    elif stage == "extract":
        print(extract_records()[2])
    elif stage == "param_audit":
        parameter_audit()
    elif stage == "compute_audit":
        compute_audit()
    elif stage == "integrity":
        print(json.dumps(integrity_gates(), indent=2))
    elif stage == "baseline":
        baseline_inventory()
    elif stage == "diagnostics":
        print(diagnostics())
        print(mechanism_diagnostics())
    elif stage == "figures":
        print(make_figures())
    elif stage == "stage1_true_seed0":
        print(stage1_true_seed0()["delta_arch0"])
    elif stage == "stage1_decision":
        print(stage1_decision())
    elif stage == "stage2_broken_seed0":
        print(stage2_broken_seed0())
    elif stage == "stage2_decision":
        print(stage2_broken_seed0())
    elif stage == "stage3_true_seed1":
        print(stage3_true_seed1())
    elif stage == "stage3a_decision":
        print(stage3a_decision())
    elif stage == "stage3_broken_seed1":
        print(stage3_broken_seed1())
    elif stage == "stage3b_decision":
        print(stage3_broken_seed1())
    elif stage == "final":
        print(final_decision())
    elif stage == "two_seed_summary":
        print(two_seed_summary())
    else:
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
