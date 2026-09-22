"""DTX-v0 -- No-Ring Dictionary x Generic Topology Cross (strict-static ZINC).

Round: **ZINC-no-ring-dictionary-topology-cross-v0** (DTX-v0).

One causal question
-------------------
A shared dictionary learns reusable *local chemical environments* from a
label-free static descriptor.  The model is **never told** anything about
rings: the newly added branch reads neither ``ring5`` / ``ring6`` / ``fused`` /
``aromatic`` / ``ring_boundary`` labels, nor cycle membership, cycle length,
cycle rank, cycle-basis or any enumerated cycle object.

Instead every patch/root gets a small **generic, untyped, label-free**
topology role vector ``s_i in R^8`` (shell occupancies, within/between-shell
edge densities, multi-parent and same-shell incidence fractions) computed from
the *untyped radius-2 induced patch*.  The question is whether the **aligned
joint statistic**

    J_align = (1/N) sum_i alpha_i outer s_i        # R^{64 x 8}

"...which learned local dictionary environments occur in which generic
structural roles..." carries task-relevant information that the *marginal-only*
matched control

    J_indep = mu_alpha outer mu_s                   # R^{64 x 8}

cannot represent.  The two arms are otherwise **bit-identical**: same tensors,
same shapes, same initialisation, same data, same protocol.

This round does not re-introduce the deleted typed-token (16D) / parent (8D)
identity channels, does not replace the raw pair backbone, and does not touch
the inherited global topology hinge.  The hinge still enters the graph readout
as before; the new cross branch picks up no explicit ring context.

Strict-static contract (inherited, non-negotiable)
--------------------------------------------------
::

    NO message passing
    NO pair/relation result -> local hidden update
    NO topology / cross result -> local hidden update
    NO recurrence, NO relation refresh, NO second pair evaluation, NO attention

``h_i``, ``alpha_i`` and ``s_i`` are all pure functions of the static patch and
of label-free topology.  ``J`` is a graph-level sufficient statistic; it can
never write back into any local state.

Architecture (frozen; no sweep)
-------------------------------
::

    x_i      = [patch_cont | patch_context]                # 146D static descriptor
    h_i      = SiLU MLP(146 -> 64 -> 48)                   # local state (unary + raw pair)

    x_dict_i = patch_cont with the explicit cycle-rank coordinate removed  # 145D
    z_dict_i = SiLU MLP(145 -> 64 -> 32)
    q_i      = l2_normalize(z_dict_i)
    D_k      = l2_normalize(dictionary_atom_k)             # K = 64, rank = 32
    tau      = 0.05 + 0.95 * sigmoid(tau_logit)            # tau_init = 0.20
    alpha_i  = softmax((q_i @ D_norm^T) / tau)             # 64D

    s_i      = generic_topology_role(untyped radius-2 induced patch)   # 8D

    J_align  = (1/N) sum_i alpha_i outer s_i               # 512D (ARM A)
    J_indep  = mu_alpha outer mu_s                         # 512D (ARM M)
    mu_alpha = (1/N) sum_i alpha_i ; mu_s = (1/N) sum_i s_i
    e_cross  = SiLU Linear(512 -> 32)

    raw pair (S0 style, evaluated ONCE):
        u_i  = pair_projection(h_i)                        # 48 -> 16
        f_ij = [u_i+u_j | |u_i-u_j| | (u_i*u_j)*gate_ij | relation_encoder(r_ij)]
        q_ij = SiLU MLP(64 -> 64 -> 16)

    unary 97 + pair 165 (5 buckets) + global 32 + topology hinge 8
           + mu_alpha 64 + mu_s 8 + e_cross 32 = R in R^406
    y_hat  = GenericReader(406 -> 16 -> 16 -> 1)

Both arms compute ``mu_alpha`` and ``mu_s``; the *only* difference is whether
the 512D cross matrix keeps the within-graph ``alpha_i <-> s_i`` alignment.

Explicitly out of scope / forbidden in this round
--------------------------------------------------
typed-token or parent identity lookups (raw/hashed/corrected/historical/fixed
code/certificate); ring labels or cycle enumeration anywhere inside the new
cross branch; a second dictionary; learned topology prototypes; a K x K
dictionary-pair matrix; SRDA tensor algebra; dictionary residual into ``h_i``;
``gamma``; message passing / centre update / attention / relation refresh /
second pair pass; entropy, sparsity, reconstruction, balance, orthogonality or
top-k auxiliary objectives; ``K`` / ``tau`` / width / topology-feature sweeps;
seed 1+; official ZINC test.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_no_ring_dictionary_topology_cross <stage>

Stages: ``param_audit encode parity integrity smoke train intervene posthoc
analyze all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
from tracks.ksvd.experiments.luyin16.zinc_static_relational_dictionary_algebra import (
    bucket_moments,
    node_moments,
)

# ---------------------------------------------------------------------------
# frozen constants (no sweep)
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
CANONICAL_V4_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

RESULTS_DIR = TRACK_ROOT / "results/zinc_no_ring_dictionary_topology_cross_v0"
CACHE_DIR = RESULTS_DIR / "cache"
RUNS_DIR = RESULTS_DIR / "runs"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"

EXPERIMENT_NAME = "zinc_no_ring_dictionary_topology_cross_v0"
PROTOCOL_VERSION = "zinc_no_ring_dictionary_topology_cross_v0"

# --- local state (identity-free static descriptor, S0 local width) ---------
LOCAL_INPUT_WIDTH = 146  # runtime-attested: patch_cont(146) + patch_context(0)
LOCAL_HIDDEN = 64
LOCAL_DIM = 48
DROPOUT = 0.05

# --- dictionary input: patch_cont minus the explicit cycle-rank coordinate ---
# ``_shell_descriptor`` layout: atom_shell(3 x 28) + bond_shell(6 x 4)
# + root_atom(28) + incident_bonds(4) + scalars(6).  The six scalars are
# ``[log1p(n_nodes), log1p(n_edges), boundary_fraction, cycle_rank/n_nodes,
# root_degree/4, mean_degree/4]``.  Only ``cycle_rank/n_nodes`` is directly
# cycle-derived; it is deleted for the dictionary query.
CYCLE_RANK_SCALAR_OFFSET = 3
CYCLE_RANK_FEATURE_INDEX = (
    3 * zpp.ATOM_CATEGORIES
    + len(zpp._shell_pairs_for_radius(zpp.PATCH_RADIUS)) * zpp.BOND_CATEGORIES
    + zpp.ATOM_CATEGORIES
    + zpp.BOND_CATEGORIES
    + CYCLE_RANK_SCALAR_OFFSET
)  # == 143
DICT_INPUT_WIDTH = LOCAL_INPUT_WIDTH - 1  # == 145

# --- end-to-end dictionary (identical to the frozen SRDA/SDPK dictionary) --
DICT_ENCODER_HIDDEN = 64
DICT_ATOMS = 64
DICT_RANK = 32
DICT_TAU_INIT = 0.20

# --- generic untyped topology role -----------------------------------------
TOPO_ROLE_DIM = 8
TOPO_ROLE_NAMES: tuple[str, ...] = (
    "root_degree_fraction",
    "shell1_occupancy",
    "shell2_occupancy",
    "shell1_edge_density",
    "shell1_shell2_edge_density",
    "shell2_edge_density",
    "shell2_multi_parent_fraction",
    "same_shell_incidence_fraction",
)

# --- raw S0-style pair backbone (unchanged, evaluated once) -----------------
PAIR_HIDDEN = 16
PAIR_ENCODER_HIDDEN = 64
RELATION_HIDDEN = 32

# --- graph-level cross encoder ---------------------------------------------
CROSS_DIM = DICT_ATOMS * TOPO_ROLE_DIM  # 512
CROSS_HIDDEN = 32

# --- graph-level reader -----------------------------------------------------
GRAPH_HEAD_HIDDEN = (16, 16)

# --- pre-registered performance references (absolute context only) ---------
S0_SEED0_BEST = 0.145674
S0_SEED0_SOUP = 0.140794
S0_SEED1_BEST = 0.139389
S0_SEED1_SOUP = 0.136423
SDPK_SEED0_BEST = 0.14219318306347123
SDPK_SEED0_SOUP = 0.13973486851429334
SRDA_SEED0_BEST = 0.15544865403691074
SRDA_SEED0_SOUP = 0.14987310345092555

# --- pre-registered outcome gates (frozen before any formal run) -----------
GATE_ALIGN_SOUP_DELTA = 0.004    # M_soup - A_soup (primary aligned gate)
GATE_ALIGN_BEST_DELTA = 0.003    # M_best - A_best
GATE_ALIGN_ABS_SOUP = 0.1340     # ALIGNED absolute Top-5 soup
GATE_DIRECTIONAL_SOUP_DELTA = 0.003  # 0 < delta < this -> direction only
GATE_MARGINAL_SAFETY = 0.1450    # M_soup above this -> BASE_REGRESSION_CONFOUND
INTERVENTION_MIN_MEAN_SHIFT = 0.010  # a "clear" mechanism intervention
INTERVENTION_NOISE_MULTIPLE = 20.0   # ... and >= 20x the repeat-forward noise floor

MAX_FULL_RUNS = 2
ARMS = ("marginal", "aligned")
POSTHOC_STRATA = (
    "ring_any",
    "ring5",
    "ring6",
    "multi_fused",
    "ring_boundary",
    "non_ring",
)


# ---------------------------------------------------------------------------
# io / determinism helpers (inherited conventions)
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    sdp._write_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return sdp._read_json(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    sdp._write_csv(path, rows)


def _n_params(module: nn.Module | None) -> int:
    return sdp._n_params(module)


def _git_commit() -> str:
    return sdp._git_commit()


def _environment_fingerprint(device: str = "cpu") -> dict[str, Any]:
    return sdp._environment_fingerprint(device)


def _set_deterministic(enabled: bool) -> None:
    """Repo-verified deterministic CUDA regime (same-seed drift is material)."""
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def _seed_everything(seed: int) -> None:
    sdp._seed_everything(int(seed))


def _tau_from_logit(tau_logit: torch.Tensor) -> torch.Tensor:
    return 0.05 + 0.95 * torch.sigmoid(tau_logit)


def _tau_logit_for(tau: float) -> float:
    return sdp._tau_logit_for(tau)


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# generic untyped topology roles (no cycle / ring machinery anywhere)
# ---------------------------------------------------------------------------


def generic_topology_role_from_edges(
    center: int,
    distances: Mapping[int, int],
    edges: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Eight bounded, label-free, permutation-invariant structural quantities.

    Only graph distance classes (0/1/2) and induced edge counts enter.  The
    function has no access to atom types, bond types, cycle objects, cycle
    length, cycle rank, ring membership or node identity:

    ::

        t0  degree(root) / max(1, n_patch - 1)          (root local degree fraction)
        t1  |S1| / max(1, n_patch - 1)                  (shell-1 occupancy)
        t2  |S2| / max(1, n_patch - 1)                  (shell-2 occupancy)
        t3  #edges(S1,S1) / max(1, C(|S1|,2))           (within-S1 edge density)
        t4  #edges(S1,S2) / max(1, |S1|*|S2|)           (S1-S2 edge density)
        t5  #edges(S2,S2) / max(1, C(|S2|,2))           (within-S2 edge density)
        t6  mean_{v in S2} 1[#neighbors(v) in S1 >= 2]  (multi-parent shell-2)
        t7  mean_{v != root} same-shell-neighbors(v)    (same-shell incidence)
                             / max(1, patch_degree(v))

    Every value is a ratio of integers (or of integer means) and therefore
    *exactly* invariant to node relabelling and to patch iteration order: the
    same-shell sums are accumulated over a canonically sorted multiset.
    """
    center = int(center)
    if center not in distances or int(distances[center]) != 0:
        raise ValueError("generic_topology_role requires the centre at distance 0")
    normalized: dict[int, int] = {}
    for node, distance in distances.items():
        distance = int(distance)
        if distance not in (0, 1, 2):
            raise ValueError(f"radius-2 patch distance out of range: {distance}")
        normalized[int(node)] = distance
    n_patch = len(normalized)

    shell_counts = {0: 0, 1: 0, 2: 0}
    for distance in normalized.values():
        shell_counts[distance] += 1

    degree: Counter[int] = Counter()
    same_shell_neighbors: Counter[int] = Counter()
    parent_count: Counter[int] = Counter()
    edges_s1 = 0
    edges_s12 = 0
    edges_s2 = 0
    seen: set[tuple[int, int]] = set()
    for left, right in edges:
        left = int(left)
        right = int(right)
        if left == right:
            continue
        key = (left, right) if left < right else (right, left)
        if key in seen:
            continue
        seen.add(key)
        if left not in normalized or right not in normalized:
            raise ValueError("edge outside the induced patch")
        left_distance = normalized[left]
        right_distance = normalized[right]
        degree[left] += 1
        degree[right] += 1
        if left_distance == right_distance:
            same_shell_neighbors[left] += 1
            same_shell_neighbors[right] += 1
        shell_pair = (
            (left_distance, right_distance)
            if left_distance <= right_distance
            else (right_distance, left_distance)
        )
        if shell_pair == (1, 1):
            edges_s1 += 1
        elif shell_pair == (1, 2):
            edges_s12 += 1
            parent_count[right if right_distance == 2 else left] += 1
        elif shell_pair == (2, 2):
            edges_s2 += 1

    shell1 = shell_counts[1]
    shell2 = shell_counts[2]
    denominator = max(1, n_patch - 1)
    t0 = degree[center] / denominator
    t1 = shell1 / denominator
    t2 = shell2 / denominator
    t3 = edges_s1 / max(1.0, shell1 * (shell1 - 1) / 2.0)
    t4 = edges_s12 / max(1.0, shell1 * shell2)
    t5 = edges_s2 / max(1.0, shell2 * (shell2 - 1) / 2.0)
    if shell2:
        multi_parent = sum(
            1
            for node, distance in normalized.items()
            if distance == 2 and parent_count[node] >= 2
        )
        t6 = multi_parent / shell2
    else:
        t6 = 0.0
    # Canonical accumulation order: node IDs must not influence the float sum.
    same_shell_pairs = sorted(
        (max(1, degree[node]), same_shell_neighbors[node])
        for node, distance in normalized.items()
        if distance > 0
    )
    non_root = len(same_shell_pairs)
    if non_root:
        t7 = (
            sum(
                same / degree_value
                for degree_value, same in same_shell_pairs
            )
            / non_root
        )
    else:
        t7 = 0.0
    return np.asarray([t0, t1, t2, t3, t4, t5, t6, t7], dtype=np.float32)


def generic_topology_role(graph: Any, center: int, distances: Mapping[int, int]) -> np.ndarray:
    """Typed-graph wrapper around :func:`generic_topology_role_from_edges`."""
    induced = graph.induced(set(int(node) for node in distances))
    return generic_topology_role_from_edges(
        int(center), distances, induced.edges()
    )


def topology_roles_for_raw_graph(graph: Any) -> np.ndarray:
    """Per-centre role matrix for a raw graph (centres sorted, patch order fixed)."""
    centers = list(graph.nodes)
    rows = []
    for center in centers:
        distances = zpp._ego_distances(graph, int(center), zpp.PATCH_RADIUS)
        rows.append(generic_topology_role(graph, int(center), distances))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# dictionary-input audit: locate (and delete) the cycle-rank coordinate
# ---------------------------------------------------------------------------


def _descriptor_for_edges(
    n_nodes: int, edges: Sequence[tuple[int, int]], center: int = 0
) -> np.ndarray:
    from tracks.ksvd.code.graph import from_edges

    graph = from_edges(int(n_nodes), [(int(u), int(v)) for u, v in edges])
    distances = zpp._ego_distances(graph, int(center), zpp.PATCH_RADIUS)
    node_types = np.zeros(int(n_nodes), dtype=np.int64)
    edge_types = {
        graph.edge_key(int(u), int(v)): 0 for u, v in edges
    }
    descriptor, _nodes, _boundary = zpp._shell_descriptor(
        graph, int(center), node_types, edge_types, distances, zpp.PATCH_RADIUS
    )
    return descriptor


def cycle_rank_coordinate_audit() -> dict[str, Any]:
    """Prove from the real implementation that index 143 is the cycle rank.

    Synthetic graphs with a known independent-cycle count are pushed through
    the *real* ``_shell_descriptor``; the observed coordinate must equal
    ``max(E - V + 1, 0) / V`` on the induced radius-2 patch.
    """
    cases: list[tuple[str, int, tuple[tuple[int, int], ...], int]] = [
        ("tree_chain", 3, ((0, 1), (1, 2)), 0),
        ("triangle", 3, ((0, 1), (1, 2), (2, 0)), 1),
        ("square", 4, ((0, 1), (1, 2), (2, 3), (3, 0)), 1),
        (
            "fused_triangles",
            5,
            ((0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)),
            2,
        ),
        (
            "hexagon",
            6,
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
            0,
        ),
    ]
    evidence = []
    for name, n_nodes, edges, expected_rank in cases:
        descriptor = _descriptor_for_edges(n_nodes, edges)
        from tracks.ksvd.code.graph import from_edges

        graph = from_edges(n_nodes, list(edges))
        distances = zpp._ego_distances(graph, 0, zpp.PATCH_RADIUS)
        induced = graph.induced(set(distances))
        n_patch = len(distances)
        observed = float(descriptor[CYCLE_RANK_FEATURE_INDEX])
        expected = float(expected_rank) / max(float(n_patch), 1.0)
        evidence.append(
            {
                "case": name,
                "n_patch_nodes": int(n_patch),
                "induced_edges": int(induced.num_edges()),
                "expected_cycle_rank": int(expected_rank),
                "observed_descriptor_value": observed,
                "expected_descriptor_value": expected,
                "match": bool(abs(observed - expected) < 1.0e-7),
            }
        )
    passed = all(row["match"] for row in evidence)
    if not passed:
        raise RuntimeError(f"cycle-rank coordinate audit failed: {evidence}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "feature_index": int(CYCLE_RANK_FEATURE_INDEX),
        "formula": "max(E - V + 1, 0) / V over the induced radius-2 patch",
        "synthetic_evidence": evidence,
        "passed": passed,
        "dictionary_input_width": int(DICT_INPUT_WIDTH),
        "deleted_coordinate": "explicit cycle rank (cyclomatic number)",
        "retained_scalars": [
            "log1p(n_nodes)",
            "log1p(n_edges)",
            "boundary fraction at distance == radius",
            "root degree / 4",
            "mean degree / 4",
        ],
        "other_cycle_derived_coordinates": [],
        "official_test_loaded": False,
    }


def dictionary_input_from_patch_cont(patch_cont: torch.Tensor) -> torch.Tensor:
    """``patch_cont`` (146D) with the explicit cycle-rank coordinate removed."""
    if int(patch_cont.shape[1]) != LOCAL_INPUT_WIDTH:
        raise RuntimeError(
            f"patch_cont width changed: {int(patch_cont.shape[1])} != {LOCAL_INPUT_WIDTH}"
        )
    index = int(CYCLE_RANK_FEATURE_INDEX)
    return torch.cat([patch_cont[:, :index], patch_cont[:, index + 1 :]], dim=1)


# ---------------------------------------------------------------------------
# graph-level invariant statistics
# ---------------------------------------------------------------------------


def _graph_mean(value: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    total = torch.zeros(
        (int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype
    )
    counts = torch.zeros(
        (int(n_graphs), 1), device=value.device, dtype=value.dtype
    )
    total.index_add_(0, batch, value)
    counts.index_add_(
        0,
        batch,
        torch.ones((batch.shape[0], 1), device=value.device, dtype=value.dtype),
    )
    return total / counts.clamp_min(1.0)


def cross_statistics(
    alpha: torch.Tensor, s: torch.Tensor, batch: torch.Tensor, n_graphs: int
) -> dict[str, torch.Tensor]:
    """Marginals + aligned joint + matched marginal-only control.

    ``J_align = mean_i alpha_i outer s_i`` keeps the within-graph alignment;
    ``J_indep = mu_alpha outer mu_s`` keeps only the two marginals.  Both are
    graph-level sufficient statistics: they never touch a local state.
    """
    if alpha.ndim != 2 or s.ndim != 2:
        raise ValueError("alpha and s must be rank-2")
    if int(alpha.shape[0]) != int(s.shape[0]):
        raise ValueError("alpha and s must share the occurrence axis")
    mu_alpha = _graph_mean(alpha, batch, n_graphs)
    mu_s = _graph_mean(s, batch, n_graphs)
    joint = torch.zeros(
        (int(n_graphs), int(alpha.shape[1]), int(s.shape[1])),
        device=alpha.device,
        dtype=alpha.dtype,
    )
    joint.index_add_(0, batch, alpha.unsqueeze(2) * s.unsqueeze(1))
    counts = torch.zeros((int(n_graphs),), device=alpha.device, dtype=alpha.dtype)
    counts.index_add_(
        0, batch, torch.ones((batch.shape[0],), device=alpha.device, dtype=alpha.dtype)
    )
    joint = joint / counts.clamp_min(1.0).view(-1, 1, 1)
    independent = mu_alpha.unsqueeze(2) * mu_s.unsqueeze(1)
    return {
        "mu_alpha": mu_alpha,
        "mu_s": mu_s,
        "J_align": joint,
        "J_indep": independent,
    }


def shuffle_within_graph(
    values: torch.Tensor, batch: torch.Tensor, n_graphs: int, seed: int
) -> torch.Tensor:
    """Permute occurrence rows *within* each graph (graph membership fixed)."""
    generator = torch.Generator().manual_seed(int(seed))
    output = values.clone()
    for graph_id in range(int(n_graphs)):
        index = (batch == graph_id).nonzero(as_tuple=True)[0]
        if int(index.numel()) > 1:
            permutation = torch.randperm(int(index.numel()), generator=generator)
            permutation = permutation.to(values.device)
            output[index] = values[index[permutation]]
    return output


# ---------------------------------------------------------------------------
# end-to-end dictionary assignment
# ---------------------------------------------------------------------------


class DictionaryAssignment(nn.Module):
    """``query -> alpha`` over K forward-normalized dictionary atoms.

    The dictionary is a pure function of the query embedding: no K-SVD, no OMP,
    no reconstruction/entropy/balance/orthogonality term, no top-k, and no
    residual path back into the local state.
    """

    def __init__(
        self,
        rank: int = DICT_RANK,
        atoms: int = DICT_ATOMS,
        tau_init: float = DICT_TAU_INIT,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.atoms_count = int(atoms)
        self.atoms = nn.Parameter(torch.empty(self.atoms_count, self.rank))
        self.tau_logit = nn.Parameter(torch.tensor(_tau_logit_for(tau_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.atoms, std=1.0 / math.sqrt(self.rank))

    def current_tau(self) -> torch.Tensor:
        return _tau_from_logit(self.tau_logit)

    def normalized_atoms(self) -> torch.Tensor:
        return F.normalize(self.atoms, dim=-1, eps=1.0e-8)

    def forward(self, query: torch.Tensor) -> dict[str, torch.Tensor]:
        if int(query.shape[1]) != self.rank:
            raise RuntimeError(
                f"dictionary query width changed: {int(query.shape[1])} != {self.rank}"
            )
        normalized = F.normalize(query, dim=-1, eps=1.0e-8)
        atoms = self.normalized_atoms()
        tau = self.current_tau()
        alpha = F.softmax((normalized @ atoms.t()) / tau, dim=-1)
        return {"alpha": alpha, "query": normalized, "atoms": atoms, "tau": tau}


# ---------------------------------------------------------------------------
# DTX model
# ---------------------------------------------------------------------------


class DTXModel(nn.Module):
    """Strict-static dictionary x generic-topology cross model.

    ``arm`` selects the 512D cross matrix only (``"aligned"`` -> ``J_align``,
    ``"marginal"`` -> ``J_indep``); every tensor, shape and initialisation is
    identical across arms.
    """

    center_context: bool = False
    center_update: None = None

    def __init__(
        self,
        input_width: int = LOCAL_INPUT_WIDTH,
        *,
        arm: str = "aligned",
        local_hidden: int = LOCAL_HIDDEN,
        local_dim: int = LOCAL_DIM,
        dropout: float = DROPOUT,
        dict_encoder_hidden: int = DICT_ENCODER_HIDDEN,
        dict_rank: int = DICT_RANK,
        dict_atoms: int = DICT_ATOMS,
        dict_tau_init: float = DICT_TAU_INIT,
        topo_role_dim: int = TOPO_ROLE_DIM,
        pair_hidden: int = PAIR_HIDDEN,
        pair_encoder_hidden: int = PAIR_ENCODER_HIDDEN,
        relation_hidden: int = RELATION_HIDDEN,
        cross_hidden: int = CROSS_HIDDEN,
        head_hidden: Sequence[int] = GRAPH_HEAD_HIDDEN,
        relation_width: int = zpp.RELATION_WIDTH,
        global_width: int = zpp.GLOBAL_WIDTH,
        topology_width: int = ztopo.raw_width("hinge"),
        topology_hidden: int = 16,
        topology_out: int = 8,
    ) -> None:
        super().__init__()
        if str(arm) not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = str(arm)
        self.input_width = int(input_width)
        self.local_hidden = int(local_hidden)
        self.local_dim = int(local_dim)
        self.dropout = float(dropout)
        self.dict_input_width = int(input_width) - 1
        self.dict_rank = int(dict_rank)
        self.dict_atoms_count = int(dict_atoms)
        self.topo_role_dim = int(topo_role_dim)
        self.pair_hidden = int(pair_hidden)
        self.relation_width = int(relation_width)
        self.global_width = int(global_width)
        self.topology_width = int(topology_width)
        self.cross_dim = int(dict_atoms) * int(topo_role_dim)

        # --- identity-free local encoder (static descriptor only) -----------
        self.local_encoder = nn.Sequential(
            nn.Linear(self.input_width, self.local_hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.local_hidden, self.local_dim),
        )

        # --- dictionary query encoder (cycle-rank coordinate removed) -------
        self.dict_encoder = nn.Sequential(
            nn.Linear(self.dict_input_width, int(dict_encoder_hidden)),
            nn.SiLU(),
            nn.Linear(int(dict_encoder_hidden), self.dict_rank),
        )
        self.dictionary = DictionaryAssignment(
            self.dict_rank, int(dict_atoms), float(dict_tau_init)
        )

        # --- raw S0-style pair backbone (evaluated exactly once) ------------
        self.pair_projection = nn.Linear(self.local_dim, int(pair_hidden), bias=False)
        self.relation_encoder = zpp._MLPBlock(
            self.relation_width,
            max(int(pair_hidden), RELATION_HIDDEN),
            int(pair_hidden),
            self.dropout,
        )
        self.distance_gate = nn.Embedding(zpp.DISTANCE_BUCKETS, int(pair_hidden))
        self.pair_encoder = zpp._MLPBlock(
            4 * int(pair_hidden),
            max(2 * int(pair_hidden), PAIR_ENCODER_HIDDEN),
            int(pair_hidden),
            self.dropout,
        )

        # --- graph-level static channels (inherited) -------------------------
        self.global_encoder = zpp._MLPBlock(
            self.global_width, max(self.local_dim // 2, 32), 32, self.dropout
        )
        self.topology_encoder = nn.Sequential(
            nn.Linear(self.topology_width, int(topology_hidden)),
            nn.ReLU(),
            nn.Linear(int(topology_hidden), int(topology_out)),
        )

        # --- one shallow cross encoder (both arms identical) ------------------
        self.cross_projection = nn.Sequential(
            nn.Linear(self.cross_dim, int(cross_hidden)),
            nn.SiLU(),
        )

        self.unary_width = 2 * self.local_dim + 1
        self.pair_pool_width = zpp.DISTANCE_BUCKETS * (2 * self.pair_hidden + 1)
        self.global_out_width = 32
        self.topology_out_width = int(topology_out)
        self.mu_alpha_width = int(dict_atoms)
        self.mu_s_width = int(topo_role_dim)
        self.cross_out_width = int(cross_hidden)
        self.graph_width = (
            self.unary_width
            + self.pair_pool_width
            + self.global_out_width
            + self.topology_out_width
            + self.mu_alpha_width
            + self.mu_s_width
            + self.cross_out_width
        )
        self.head = GenericReader(self.graph_width, tuple(int(h) for h in head_hidden))

        # Evaluation-only interventions (default None == frozen forward path).
        self.inference_intervention: str | None = None
        self.intervention_seed: int = 20260924

    # -- diagnostics hooks ---------------------------------------------------
    def set_inference_intervention(self, mode: str | None) -> "DTXModel":
        if mode not in (None, "alignment_removal", "topology_shuffle"):
            raise ValueError(f"unknown inference intervention {mode!r}")
        self.inference_intervention = mode
        return self

    # -- forward -------------------------------------------------------------
    def encode(self, data: Any) -> torch.Tensor:
        """Return the unified graph representation ``R`` (input to the sole head).

        Ordering is the contract: the complete local state, the dictionary
        assignment, the topology role and the graph-level cross statistics are
        computed before any pair/relation tensor is touched (and none of them
        is ever refreshed afterwards).
        """
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])

        # --- 1. static local descriptor -> local state (no pair information) --
        x = torch.cat([data.patch_cont, data.patch_context], dim=1)
        if int(x.shape[1]) != self.input_width:
            raise RuntimeError(
                f"static descriptor width changed: {int(x.shape[1])} != {self.input_width}"
            )
        h = self.local_encoder(x)
        unary = node_moments(h, data.batch, n_graphs)

        # --- 2. dictionary assignment (pure function of the cycle-free query) --
        x_dict = dictionary_input_from_patch_cont(x)
        local = self.dictionary(self.dict_encoder(x_dict))
        alpha = local["alpha"]

        # --- 3. generic untyped topology roles (label-free, no ring machine) --
        s = data.generic_topology
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if int(s.shape[1]) != self.topo_role_dim:
            raise RuntimeError(
                f"generic topology role width mismatch: data={int(s.shape[1])} "
                f"model={self.topo_role_dim}"
            )
        if self.inference_intervention == "topology_shuffle":
            s = shuffle_within_graph(
                s, data.batch, n_graphs, int(self.intervention_seed)
            )

        # --- 4. graph-level cross statistics (never write back) ---------------
        statistics = cross_statistics(alpha, s, data.batch, n_graphs)
        mu_alpha = statistics["mu_alpha"]
        mu_s = statistics["mu_s"]
        if self.arm == "aligned" and self.inference_intervention != "alignment_removal":
            cross = statistics["J_align"]
        else:
            cross = statistics["J_indep"]
        e_cross = self.cross_projection(cross.reshape(n_graphs, self.cross_dim))

        # --- 5. raw S0-style pair backbone (static relation trunk) ------------
        source = data.pair_index[0].long()
        target = data.pair_index[1].long()
        projected_left = self.pair_projection(h[source])
        projected_right = self.pair_projection(h[target])
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        relation = self.relation_encoder(data.pair_relation)
        pair_input = torch.cat(
            [
                projected_left + projected_right,
                torch.abs(projected_left - projected_right),
                projected_left * projected_right * gate,
                relation,
            ],
            dim=1,
        )
        q_ij = self.pair_encoder(pair_input)
        pair_pool = bucket_moments(q_ij, data.batch[source], data.pair_bucket, n_graphs)

        # --- 6. graph-level static channels ------------------------------------
        global_hidden = self.global_encoder(global_context)
        topology = data.topology_features
        if topology.ndim == 1:
            topology = topology.unsqueeze(0)
        if int(topology.shape[1]) != self.topology_width:
            raise RuntimeError(
                f"topology width mismatch: data={int(topology.shape[1])} "
                f"model={self.topology_width}"
            )
        topology_hidden = self.topology_encoder(topology)

        return torch.cat(
            [
                unary,
                pair_pool,
                global_hidden,
                topology_hidden,
                mu_alpha,
                mu_s,
                e_cross,
            ],
            dim=1,
        )

    def forward(self, data: Any) -> torch.Tensor:
        return self.head(self.encode(data)).view(-1)


def build_dtx(seed: int = 0, arm: str = "aligned", **kwargs: Any) -> DTXModel:
    """Deterministic construction with full parameter re-initialization."""
    _seed_everything(int(seed))
    return DTXModel(arm=str(arm), **kwargs)


build_model = build_dtx


# ---------------------------------------------------------------------------
# data: cached strict-static encoding + generic topology roles
# ---------------------------------------------------------------------------


def _attach_roles_and_verify(
    records: Sequence[Any],
    encoded: Sequence[Any],
    split: str,
    *,
    patch_standardizer: zpp.Standardizer,
    verify_patch_cont: bool,
) -> dict[str, Any]:
    """Compute per-patch roles from the raw split and attach them to ``Data``.

    Alignment is proven twice: (a) the recomputed induced patch node set must
    equal the cached ``PatchRecord.nodes`` for every patch, and (b) for the
    train split the train-fitted patch standardizer must reproduce the cached
    ``patch_cont`` bit-for-bit.
    """
    dataset = _load_zinc(ZINC_ROOT, split)
    if len(dataset) != len(records):
        raise RuntimeError(
            f"{split} dataset/records length mismatch: {len(dataset)} vs {len(records)}"
        )
    role_rows = 0
    for index, raw in enumerate(dataset):
        graph, _node_types, _edge_types = zpp._data_to_graph(raw)
        centers = list(graph.nodes)
        record = records[index]
        if len(centers) != len(record.patches):
            raise RuntimeError(
                f"{split}[{index}] patch count mismatch: {len(centers)} vs "
                f"{len(record.patches)}"
            )
        rows = np.empty((len(centers), TOPO_ROLE_DIM), dtype=np.float32)
        for patch_index, center in enumerate(centers):
            distances = zpp._ego_distances(graph, int(center), zpp.PATCH_RADIUS)
            if frozenset(int(node) for node in distances) != record.patches[patch_index].nodes:
                raise RuntimeError(
                    f"{split}[{index}] patch {patch_index} node set mismatch"
                )
            rows[patch_index] = generic_topology_role(graph, int(center), distances)
        if not np.isfinite(rows).all():
            raise FloatingPointError(f"{split}[{index}] non-finite topology role")
        encoded[index].generic_topology = torch.from_numpy(rows)
        role_rows += int(rows.shape[0])

    payload: dict[str, Any] = {
        "split": str(split),
        "n_graphs": int(len(records)),
        "role_rows": int(role_rows),
        "role_width": int(TOPO_ROLE_DIM),
        "role_finite": True,
        "patch_node_alignment": True,
    }
    if verify_patch_cont:
        recomputed = patch_standardizer.transform(zpp._patch_matrix(records))
        cached = torch.cat([row.patch_cont for row in encoded], dim=0).numpy()
        max_abs = float(np.abs(recomputed - cached).max())
        payload["patch_cont_recompute_max_abs"] = max_abs
        payload["patch_cont_recompute_bit_identical"] = bool(max_abs == 0.0)
        if max_abs != 0.0:
            raise RuntimeError(
                f"{split} cached patch_cont is not reproduced by the fit standardizer "
                f"(max abs {max_abs})"
            )
    return payload


def prepare_encoded(force: bool = False) -> dict[str, Any]:
    """Extract + encode the official train/valid splits once and cache them.

    Official ZINC **test is never loaded** (``test_policy = no_test``).
    """
    train_path = CACHE_DIR / "encoded_train.pt"
    valid_path = CACHE_DIR / "encoded_valid.pt"
    audit_path = CACHE_DIR / "encoded_audit.json"
    if not force and train_path.exists() and valid_path.exists() and audit_path.exists():
        return _read_json(audit_path)
    train_records, valid_records = sdp.load_train_valid_records()
    config = yaml.safe_load(CANONICAL_V4_CONFIG.read_text(encoding="utf-8"))
    config["test_policy"] = "no_test"
    config["model"]["device"] = "cpu"
    train_data, valid_data, audit = zpp._phase_data(train_records, valid_records, config=config)
    if int(audit["shell_width"]) != LOCAL_INPUT_WIDTH:
        raise RuntimeError(f"shell width drift: {audit['shell_width']}")
    patch_standardizer = zpp.Standardizer.fit(zpp._patch_matrix(train_records))
    train_alignment = _attach_roles_and_verify(
        train_records,
        train_data,
        "train",
        patch_standardizer=patch_standardizer,
        verify_patch_cont=True,
    )
    valid_alignment = _attach_roles_and_verify(
        valid_records,
        valid_data,
        "val",
        patch_standardizer=patch_standardizer,
        verify_patch_cont=False,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(list(train_data), train_path)
    torch.save(list(valid_data), valid_path)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_train": int(len(train_data)),
        "n_valid": int(len(valid_data)),
        "typed_vocabulary_size_with_oov": int(audit["typed_vocabulary_size_with_oov"]),
        "parent_vocabulary_size_with_oov": int(audit["parent_vocabulary_size_with_oov"]),
        "patch_radius": int(audit["patch_radius"]),
        "shell_width": int(audit["shell_width"]),
        "local_input_width": int(LOCAL_INPUT_WIDTH),
        "dictionary_input_width": int(DICT_INPUT_WIDTH),
        "cycle_rank_feature_index": int(CYCLE_RANK_FEATURE_INDEX),
        "cycle_rank_coordinate_audit": cycle_rank_coordinate_audit(),
        "topology_role_width": int(TOPO_ROLE_DIM),
        "topology_role_names": list(TOPO_ROLE_NAMES),
        "train_alignment": train_alignment,
        "valid_alignment": valid_alignment,
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(audit_path, payload)
    return payload


def load_encoded(
    train_subset: int | None = None, valid_subset: int | None = None
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_path = CACHE_DIR / "encoded_train.pt"
    valid_path = CACHE_DIR / "encoded_valid.pt"
    audit_path = CACHE_DIR / "encoded_audit.json"
    if not (train_path.exists() and valid_path.exists() and audit_path.exists()):
        raise FileNotFoundError("encoded cache missing; run the `encode` stage first")
    train_data = list(torch.load(train_path, weights_only=False))
    valid_data = list(torch.load(valid_path, weights_only=False))
    audit = _read_json(audit_path)
    if train_subset is not None:
        train_data = train_data[: int(train_subset)]
    if valid_subset is not None:
        valid_data = valid_data[: int(valid_subset)]
    return train_data, valid_data, audit


def _first_batch(data: Sequence[Any], device: torch.device, limit: int = 128):
    return sdp._first_batch(data, device, limit)


# ---------------------------------------------------------------------------
# runtime width / identity / ring-input audits
# ---------------------------------------------------------------------------


def identity_input_audit(model: DTXModel) -> dict[str, Any]:
    """Hard evidence that no typed/parent identity lookup and no ring cache exists."""
    state_keys = list(model.state_dict().keys())
    embeddings = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding)
    ]
    forbidden = (
        "typed_embedding",
        "parent_embedding",
        "typed_token",
        "parent_token",
        "ring",
        "cycle",
        "structural_token",
    )
    return {
        "state_dict_key_count": int(len(state_keys)),
        "forbidden_key_hits": [
            key
            for key in state_keys
            for needle in forbidden
            if needle in key.lower()
        ],
        "embedding_modules": [
            {"name": name, "rows": int(module.num_embeddings), "dim": int(module.embedding_dim)}
            for name, module in embeddings
        ],
        "only_distance_gate_embedding": bool(
            len(embeddings) == 1
            and embeddings[0][0] == "distance_gate"
            and int(embeddings[0][1].num_embeddings) == zpp.DISTANCE_BUCKETS
        ),
        "uses_typed_token": False,
        "uses_parent_token": False,
        "uses_structural_ring_context": False,
        "reads_generic_topology_roles": True,
    }


def runtime_width_audit(model: DTXModel, batch: Any) -> dict[str, Any]:
    """Attest every runtime width from real tensors on a real batch."""
    captured: dict[str, torch.Tensor] = {}
    dictionary_out: dict[str, torch.Tensor] = {}

    def _cap_h(_m, _i, output):
        captured["h"] = output.detach()

    def _cap_dict(_m, _i, output):
        for key, value in output.items():
            dictionary_out[key] = value.detach()

    def _cap_cross(_m, inputs, _o):
        captured["cross"] = inputs[0].detach()

    def _cap_pair(_m, inputs, _o):
        captured["pair_input"] = inputs[0].detach()

    def _cap_head(_m, inputs, _o):
        captured["graph"] = inputs[0].detach()

    handles = [
        model.local_encoder.register_forward_hook(_cap_h),
        model.dictionary.register_forward_hook(_cap_dict),
        model.cross_projection.register_forward_hook(_cap_cross),
        model.pair_encoder.register_forward_hook(_cap_pair),
        model.head.register_forward_hook(_cap_head),
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _ = model(batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    payload = {
        "local_input_width": int(batch.patch_cont.shape[1] + batch.patch_context.shape[1]),
        "declared_local_input_width": int(model.input_width),
        "local_state_width": int(captured["h"].shape[1]),
        "dictionary_input_width": int(model.dict_input_width),
        "dictionary_query_width": int(dictionary_out["query"].shape[1]),
        "dictionary_assignment_width": int(dictionary_out["alpha"].shape[1]),
        "topology_role_width": int(batch.generic_topology.shape[1]),
        "cross_matrix_width": int(captured["cross"].shape[1]),
        "cross_embedding_width": int(model.cross_out_width),
        "pair_input_width": int(captured["pair_input"].shape[1]),
        "relation_input_width": int(batch.pair_relation.shape[1]),
        "unary_width": int(model.unary_width),
        "pair_pool_width": int(model.pair_pool_width),
        "global_width": int(model.global_out_width),
        "topology_width": int(model.topology_out_width),
        "mu_alpha_width": int(model.mu_alpha_width),
        "mu_s_width": int(model.mu_s_width),
        "graph_width": int(captured["graph"].shape[1]),
        "declared_graph_width": int(model.graph_width),
        "graph_head_hidden": list(GRAPH_HEAD_HIDDEN),
        "arm": str(model.arm),
        "identity_input_audit": identity_input_audit(model),
        "official_test_loaded": False,
    }
    payload["widths_consistent"] = bool(
        payload["local_input_width"] == payload["declared_local_input_width"] == LOCAL_INPUT_WIDTH
        and payload["local_state_width"] == LOCAL_DIM
        and payload["dictionary_input_width"] == DICT_INPUT_WIDTH
        and payload["dictionary_query_width"] == DICT_RANK
        and payload["dictionary_assignment_width"] == DICT_ATOMS
        and payload["topology_role_width"] == TOPO_ROLE_DIM
        and payload["cross_matrix_width"] == CROSS_DIM
        and payload["cross_embedding_width"] == CROSS_HIDDEN
        and payload["pair_input_width"] == 4 * PAIR_HIDDEN
        and payload["graph_width"] == payload["declared_graph_width"]
        == payload["unary_width"]
        + payload["pair_pool_width"]
        + payload["global_width"]
        + payload["topology_width"]
        + payload["mu_alpha_width"]
        + payload["mu_s_width"]
        + payload["cross_embedding_width"]
    )
    return payload


def parameter_audit() -> dict[str, Any]:
    model = build_dtx(0, "aligned")
    total = _n_params(model)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "center_context": bool(model.center_context),
        "center_update_is_none": bool(model.center_update is None),
        "declared_spec": {
            "local": [LOCAL_INPUT_WIDTH, LOCAL_HIDDEN, LOCAL_DIM],
            "dictionary_input": DICT_INPUT_WIDTH,
            "dictionary_encoder": [DICT_INPUT_WIDTH, DICT_ENCODER_HIDDEN, DICT_RANK],
            "dictionary": {"atoms": DICT_ATOMS, "rank": DICT_RANK, "tau_init": DICT_TAU_INIT},
            "topology_role_dim": TOPO_ROLE_DIM,
            "cross_matrix": CROSS_DIM,
            "cross_projection": [CROSS_DIM, CROSS_HIDDEN],
            "raw_pair": {
                "pair_projection": [LOCAL_DIM, PAIR_HIDDEN],
                "pair_input_width": 4 * PAIR_HIDDEN,
                "pair_encoder": [4 * PAIR_HIDDEN, PAIR_ENCODER_HIDDEN, PAIR_HIDDEN],
                "relation_width": zpp.RELATION_WIDTH,
            },
            "graph_head_hidden": list(GRAPH_HEAD_HIDDEN),
            "dropout": DROPOUT,
        },
        "structural_widths": {
            "unary_width": int(model.unary_width),
            "pair_pool_width": int(model.pair_pool_width),
            "global_out_width": int(model.global_out_width),
            "topology_out_width": int(model.topology_out_width),
            "mu_alpha_width": int(model.mu_alpha_width),
            "mu_s_width": int(model.mu_s_width),
            "cross_out_width": int(model.cross_out_width),
            "graph_width": int(model.graph_width),
        },
        "params": {
            "total": total,
            "local_encoder": _n_params(model.local_encoder),
            "dict_encoder": _n_params(model.dict_encoder),
            "dictionary": _n_params(model.dictionary),
            "pair_projection": _n_params(model.pair_projection),
            "relation_encoder": _n_params(model.relation_encoder),
            "distance_gate": _n_params(model.distance_gate),
            "pair_encoder": _n_params(model.pair_encoder),
            "global_encoder": _n_params(model.global_encoder),
            "topology_encoder": _n_params(model.topology_encoder),
            "cross_projection": _n_params(model.cross_projection),
            "head": _n_params(model.head),
        },
        "identity_input_audit": identity_input_audit(model),
        "cycle_rank_coordinate_audit": cycle_rank_coordinate_audit(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# A / M parity
# ---------------------------------------------------------------------------


def parity_gate() -> dict[str, Any]:
    """Arm A and Arm M must differ only in the forward cross-matrix selection."""
    aligned = build_dtx(0, "aligned")
    marginal = build_dtx(0, "marginal")
    aligned_state = aligned.state_dict()
    marginal_state = marginal.state_dict()
    keys_equal = list(aligned_state.keys()) == list(marginal_state.keys())
    shapes_equal = keys_equal and all(
        aligned_state[key].shape == marginal_state[key].shape for key in aligned_state
    )
    init_identical = keys_equal and all(
        torch.equal(aligned_state[key], marginal_state[key]) for key in aligned_state
    )
    total_equal = _n_params(aligned) == _n_params(marginal)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "arm_a_total_params": int(_n_params(aligned)),
        "arm_m_total_params": int(_n_params(marginal)),
        "state_dict_keys_equal": bool(keys_equal),
        "state_dict_shapes_equal": bool(shapes_equal),
        "shared_init_bit_identical": bool(init_identical),
        "total_params_equal": bool(total_equal),
        "only_difference": "forward selection of J_align vs J_indep",
        "official_test_loaded": False,
    }
    payload["passed"] = bool(
        keys_equal and shapes_equal and init_identical and total_equal
    )
    if not payload["passed"]:
        raise RuntimeError(f"arm parity gate failed: {payload}")
    _write_json(RESULTS_DIR / "parity_gates.json", payload)
    return payload


# ---------------------------------------------------------------------------
# strict-static integrity gates
# ---------------------------------------------------------------------------


def _contract_violation(*_args: Any, **_kwargs: Any) -> torch.Tensor:
    raise RuntimeError(
        "strict-static contract violated: _pool_pairs_to_centres was called"
    )


def _trace_forward(model: DTXModel, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    captured: dict[str, Any] = {}
    counters = {"pair_encoder": 0, "relation_encoder": 0, "dict_encoder": 0, "dictionary": 0}

    def _cap_h(_m, _i, output):
        captured["h"] = output.detach().clone()

    def _cap_dict_encoder(_m, _i, output):
        captured["dict_embedding"] = output.detach().clone()
        counters["dict_encoder"] += 1

    def _cap_dictionary(_m, _i, output):
        captured["alpha"] = output["alpha"].detach().clone()
        counters["dictionary"] += 1

    def _cap_cross(_m, inputs, _o):
        captured["cross"] = inputs[0].detach().clone()

    def _cap_pair(_m, inputs, _o):
        captured["pair_input"] = inputs[0].detach().clone()
        counters["pair_encoder"] += 1

    def _cap_relation(_m, _i, _o):
        counters["relation_encoder"] += 1

    handles = [
        model.local_encoder.register_forward_hook(_cap_h),
        model.dict_encoder.register_forward_hook(_cap_dict_encoder),
        model.dictionary.register_forward_hook(_cap_dictionary),
        model.cross_projection.register_forward_hook(_cap_cross),
        model.pair_encoder.register_forward_hook(_cap_pair),
        model.relation_encoder.register_forward_hook(_cap_relation),
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    captured["counters"] = counters
    captured["s"] = batch.generic_topology.detach().clone()
    return prediction, captured


def static_contract_checks(model: DTXModel, batch: Any) -> dict[str, Any]:
    """Programmatic hard checks of the strict-static + no-ring-input contract."""
    results: dict[str, Any] = {}
    model.eval()

    results["center_context_false"] = bool(model.center_context is False)
    results["center_update_is_none"] = bool(model.center_update is None)
    if not results["center_context_false"] or not results["center_update_is_none"]:
        raise RuntimeError("strict-static contract failed: centre update present")

    # (A) historical pair->centre aggregation must be unreachable.
    original = zpp.PatchPathModel._pool_pairs_to_centres
    zpp.PatchPathModel._pool_pairs_to_centres = _contract_violation
    try:
        with torch.no_grad():
            _ = model(batch)
        results["forward_without_pair_to_center"] = True
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original

    # (B) one single pass over the pair kernel / relation trunk / dictionary.
    prediction, trace = _trace_forward(model, batch)
    results["pair_encoder_calls_per_forward"] = int(trace["counters"]["pair_encoder"])
    results["relation_encoder_calls_per_forward"] = int(trace["counters"]["relation_encoder"])
    results["dict_encoder_calls_per_forward"] = int(trace["counters"]["dict_encoder"])
    results["dictionary_calls_per_forward"] = int(trace["counters"]["dictionary"])
    for key, expected in (
        ("pair_encoder", 1),
        ("relation_encoder", 1),
        ("dict_encoder", 1),
        ("dictionary", 1),
    ):
        if trace["counters"][key] != expected:
            raise RuntimeError(f"{key} evaluated {trace['counters'][key]} times per forward")

    # (C) cross statistics must not touch any local state: mutate s and require
    # h_i / alpha_i to be bit-identical while the prediction changes (Arm A).
    mutated = batch.clone()
    generator = torch.Generator().manual_seed(20260924)
    mutated.generic_topology = torch.randn(
        mutated.generic_topology.shape, generator=generator
    ).to(mutated.generic_topology.device)
    _, trace_mutated = _trace_forward(model, mutated)
    results["h_identical_under_topology_mutation"] = bool(
        torch.equal(trace["h"], trace_mutated["h"])
    )
    results["alpha_identical_under_topology_mutation"] = bool(
        torch.equal(trace["alpha"], trace_mutated["alpha"])
    )
    results["dict_embedding_identical_under_topology_mutation"] = bool(
        torch.equal(trace["dict_embedding"], trace_mutated["dict_embedding"])
    )
    results["max_abs_alpha_diff_under_topology_mutation"] = float(
        (trace["alpha"] - trace_mutated["alpha"]).abs().max().item()
    )
    if not results["h_identical_under_topology_mutation"]:
        raise RuntimeError("topology role feeds back into the local state")
    if not results["alpha_identical_under_topology_mutation"]:
        raise RuntimeError("topology role feeds back into the dictionary assignment")

    # (D) marginal control is invariant to the within-graph alignment, while the
    # aligned joint (and its prediction) responds.  ``s_perm`` is a permutation
    # of the occurrence rows inside each graph, so mu_s and mu_alpha are the
    # same multisets; only the pairing alpha_i <-> s_i changes.
    batch_index = batch.batch
    generator = torch.Generator().manual_seed(777)
    permuted_s = batch.generic_topology.clone()
    for graph_id in range(int(batch.global_context.shape[0])):
        index = (batch_index == graph_id).nonzero(as_tuple=True)[0]
        if int(index.numel()) > 1:
            order = torch.randperm(int(index.numel()), generator=generator).to(index.device)
            permuted_s[index] = batch.generic_topology[index[order]]
    permuted_batch = batch.clone()
    permuted_batch.generic_topology = permuted_s

    # Arm M shares the passed model's parameters exactly (only the forward
    # cross-matrix selection differs), so the comparison is weight-matched.
    marginal_model = build_dtx(0, "marginal")
    marginal_model.load_state_dict(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}
    )
    marginal_model.to(batch.patch_cont.device)
    n_graphs = int(batch.global_context.shape[0])
    for arm_model, label in ((marginal_model, "marginal"), (model, "aligned")):
        arm_model.eval()
        with torch.no_grad():
            base_prediction = arm_model(batch)
            perm_prediction = arm_model(permuted_batch)
        base_stats = cross_statistics(
            trace["alpha"], trace["s"], batch.batch, n_graphs
        )
        perm_stats = cross_statistics(
            trace["alpha"], permuted_s, batch.batch, n_graphs
        )
        joint = base_stats["J_align"] if label == "aligned" else base_stats["J_indep"]
        perm_joint = perm_stats["J_align"] if label == "aligned" else perm_stats["J_indep"]
        results[f"{label}_joint_shift_under_alignment_permutation"] = float(
            (joint - perm_joint).abs().max().item()
        )
        results[f"{label}_mu_s_shift_under_alignment_permutation"] = float(
            (base_stats["mu_s"] - perm_stats["mu_s"]).abs().max().item()
        )
        results[f"{label}_mu_alpha_shift_under_alignment_permutation"] = float(
            (base_stats["mu_alpha"] - perm_stats["mu_alpha"]).abs().max().item()
        )
        results[f"{label}_prediction_shift_under_alignment_permutation"] = float(
            (base_prediction - perm_prediction).abs().max().item()
        )
        # The marginal arm's joint is *by construction* the outer product of its
        # own marginals: it cannot see the alignment at all.
        results[f"{label}_joint_is_outer_product"] = bool(
            torch.equal(joint, base_stats["mu_alpha"].unsqueeze(2) * base_stats["mu_s"].unsqueeze(1))
        )
    results["marginal_alignment_invariant"] = bool(
        results["marginal_joint_shift_under_alignment_permutation"] <= 1.0e-5
        and results["marginal_prediction_shift_under_alignment_permutation"] <= 1.0e-5
    )
    results["aligned_alignment_sensitive"] = bool(
        results["aligned_joint_shift_under_alignment_permutation"] >= 1.0e-2
        and results["aligned_prediction_shift_under_alignment_permutation"]
        >= max(1.0e-6, 10.0 * results["marginal_prediction_shift_under_alignment_permutation"])
    )
    if not results["marginal_alignment_invariant"]:
        raise RuntimeError(
            "Arm M responded to within-graph s alignment: "
            f"{results['marginal_prediction_shift_under_alignment_permutation']}"
        )
    if not results["aligned_alignment_sensitive"]:
        raise RuntimeError(
            "Arm A was insensitive to within-graph s alignment: "
            f"{results['aligned_joint_shift_under_alignment_permutation']}"
        )

    # (E) no identity lookup: poison typed/parent tokens; any lookup would raise.
    repeat_prediction, _repeat = _trace_forward(model, batch)
    identity = batch.clone()
    identity.typed_token = torch.full_like(batch.typed_token, 10_000_000)
    identity.parent_token = torch.full_like(batch.parent_token, 10_000_000)
    with torch.no_grad():
        identity_prediction = model(identity)
    repeat_shift = float((prediction - repeat_prediction).abs().max().item())
    results["prediction_shift_repeat_forward_baseline"] = repeat_shift
    results["prediction_shift_under_identity_mutation"] = float(
        (prediction - identity_prediction).abs().max().item()
    )
    results["identity_index_poison_no_lookup"] = True
    results["identity_mutation_zero_effect"] = bool(
        results["prediction_shift_under_identity_mutation"] <= max(1.0e-5, repeat_shift + 1.0e-6)
    )
    if not results["identity_mutation_zero_effect"]:
        raise RuntimeError(
            "typed/parent identity tokens influence the prediction: "
            f"{results['prediction_shift_under_identity_mutation']}"
        )

    # (F) invariance: pair order, endpoint swap, within-graph patch relabel.
    if int(batch.pair_index.shape[1]) > 1:
        order = torch.randperm(
            int(batch.pair_index.shape[1]), generator=torch.Generator().manual_seed(7)
        ).to(batch.pair_index.device)
        shuffled = batch.clone()
        shuffled.pair_index = batch.pair_index[:, order]
        shuffled.pair_relation = batch.pair_relation[order]
        shuffled.pair_bucket = batch.pair_bucket[order]
        with torch.no_grad():
            shuffled_prediction = model(shuffled)
        results["pair_order_max_abs_pred_diff"] = float(
            (repeat_prediction - shuffled_prediction).abs().max().item()
        )
        swapped = batch.clone()
        swapped.pair_index = batch.pair_index.flip(0)
        with torch.no_grad():
            swapped_prediction = model(swapped)
        results["endpoint_swap_max_abs_pred_diff"] = float(
            (repeat_prediction - swapped_prediction).abs().max().item()
        )
    else:
        results["pair_order_max_abs_pred_diff"] = 0.0
        results["endpoint_swap_max_abs_pred_diff"] = 0.0
    results["pair_order_invariant"] = bool(results["pair_order_max_abs_pred_diff"] < 1.0e-4)
    results["endpoint_swap_symmetric"] = bool(results["endpoint_swap_max_abs_pred_diff"] < 1.0e-4)

    generator = torch.Generator().manual_seed(13)
    patch_order = torch.arange(
        int(batch.patch_cont.shape[0]), device=batch.patch_cont.device
    )
    for graph_id in range(n_graphs):
        members = (batch.batch == graph_id).nonzero(as_tuple=True)[0]
        if int(members.numel()) > 1:
            patch_order[members] = members[
                torch.randperm(int(members.numel()), generator=generator).to(members.device)
            ]
    inverse = torch.empty_like(patch_order)
    inverse[patch_order] = torch.arange(patch_order.shape[0], device=patch_order.device)
    relabeled = batch.clone()
    relabeled.patch_cont = batch.patch_cont[patch_order]
    relabeled.patch_context = batch.patch_context[patch_order]
    relabeled.typed_token = batch.typed_token[patch_order]
    relabeled.parent_token = batch.parent_token[patch_order]
    relabeled.structural_token = batch.structural_token[patch_order]
    relabeled.generic_topology = batch.generic_topology[patch_order]
    relabeled.pair_index = inverse[batch.pair_index]
    with torch.no_grad():
        relabeled_prediction = model(relabeled)
    results["patch_relabel_max_abs_pred_diff"] = float(
        (repeat_prediction - relabeled_prediction).abs().max().item()
    )
    results["patch_order_invariant"] = bool(
        results["patch_relabel_max_abs_pred_diff"] < 1.0e-4
    )

    results["identity_input_audit"] = identity_input_audit(model)
    results["passed"] = bool(
        results["forward_without_pair_to_center"]
        and results["pair_encoder_calls_per_forward"] == 1
        and results["relation_encoder_calls_per_forward"] == 1
        and results["dict_encoder_calls_per_forward"] == 1
        and results["dictionary_calls_per_forward"] == 1
        and results["h_identical_under_topology_mutation"]
        and results["alpha_identical_under_topology_mutation"]
        and results["marginal_alignment_invariant"]
        and results["aligned_alignment_sensitive"]
        and results["identity_mutation_zero_effect"]
        and results["pair_order_invariant"]
        and results["endpoint_swap_symmetric"]
        and results["patch_order_invariant"]
        and results["identity_input_audit"]["only_distance_gate_embedding"]
    )
    return results


def gradient_viability_checks(model: DTXModel, batch: Any) -> dict[str, Any]:
    """Real mini-batch backward: every declared component must receive gradient."""
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch).view(-1)
    target = batch.y.view(-1)
    loss = F.l1_loss(prediction, target)
    loss.backward()

    def grad_norm(parameters: Sequence[nn.Parameter]) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return math.sqrt(total)

    payload: dict[str, Any] = {"loss": float(loss.detach())}
    payload["local_encoder_grad_norm"] = grad_norm(list(model.local_encoder.parameters()))
    payload["dict_encoder_grad_norm"] = grad_norm(list(model.dict_encoder.parameters()))
    payload["dict_atoms_grad_norm"] = grad_norm([model.dictionary.atoms])
    payload["tau_grad_norm"] = grad_norm([model.dictionary.tau_logit])
    payload["cross_projection_grad_norm"] = grad_norm(list(model.cross_projection.parameters()))
    payload["pair_projection_grad_norm"] = grad_norm(list(model.pair_projection.parameters()))
    payload["relation_encoder_grad_norm"] = grad_norm(list(model.relation_encoder.parameters()))
    payload["pair_encoder_grad_norm"] = grad_norm(list(model.pair_encoder.parameters()))
    payload["global_encoder_grad_norm"] = grad_norm(list(model.global_encoder.parameters()))
    payload["topology_encoder_grad_norm"] = grad_norm(list(model.topology_encoder.parameters()))
    payload["head_grad_norm"] = grad_norm(list(model.head.parameters()))
    model.eval()
    model.zero_grad(set_to_none=True)

    required_positive = (
        "local_encoder_grad_norm",
        "dict_encoder_grad_norm",
        "dict_atoms_grad_norm",
        "cross_projection_grad_norm",
        "pair_projection_grad_norm",
        "relation_encoder_grad_norm",
        "pair_encoder_grad_norm",
        "head_grad_norm",
    )
    finite_keys = [key for key in payload if key != "loss"]
    payload["all_finite"] = bool(
        all(math.isfinite(float(payload[key])) for key in finite_keys)
    )
    payload["required_positive_gradients"] = list(required_positive)
    payload["required_positive_ok"] = bool(
        all(float(payload[key]) > 0.0 for key in required_positive)
    )
    payload["tau_grad_finite"] = bool(math.isfinite(float(payload["tau_grad_norm"])))
    payload["branch_receives_gradient"] = bool(
        payload["all_finite"] and payload["required_positive_ok"] and payload["tau_grad_finite"]
    )
    return payload


def integrity_stage(valid_subset: int = 64) -> dict[str, Any]:
    _train_data, valid_data, _audit = load_encoded(valid_subset=int(valid_subset))
    batch = _first_batch(valid_data, torch.device("cpu"), limit=min(64, len(valid_data)))
    model = build_dtx(0, "aligned")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_audit": parameter_audit(),
        "parity_gate": parity_gate(),
        "runtime_widths": runtime_width_audit(model, batch),
        "contract": static_contract_checks(model, batch),
        "gradient_viability": gradient_viability_checks(model, batch),
        "official_test_loaded": False,
    }
    payload["passed"] = bool(
        payload["parity_gate"]["passed"]
        and payload["runtime_widths"]["widths_consistent"]
        and payload["runtime_widths"]["identity_input_audit"]["only_distance_gate_embedding"]
        and payload["contract"]["passed"]
        and payload["gradient_viability"]["branch_receives_gradient"]
    )
    _write_json(RESULTS_DIR / "integrity_gates.json", payload)
    return payload


# ---------------------------------------------------------------------------
# dictionary / cross / topology diagnostics (inference only)
# ---------------------------------------------------------------------------


def dictionary_diagnostics(model: DTXModel, loader, device: torch.device) -> dict[str, Any]:
    chunks: list[torch.Tensor] = []
    role_chunks: list[torch.Tensor] = []
    alpha_chunks: list[torch.Tensor] = []

    def _cap_dict(_m, _i, output):
        alpha_chunks.append(output["alpha"].detach())

    def _cap_h(_m, _i, output):
        chunks.append(output.detach())

    model.eval()
    handle_dict = model.dictionary.register_forward_hook(_cap_dict)
    handle_h = model.local_encoder.register_forward_hook(_cap_h)
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                role_chunks.append(batch.generic_topology.detach())
                model(batch)
    finally:
        handle_dict.remove()
        handle_h.remove()

    alpha = torch.cat(alpha_chunks, dim=0)
    roles = torch.cat(role_chunks, dim=0)
    atoms = model.dictionary.normalized_atoms()
    eps = 1.0e-12
    entropy = -(alpha * (alpha + eps).log()).sum(dim=1)
    mean_mass = alpha.mean(dim=0)
    cosine = atoms @ atoms.t()
    off_diagonal = cosine - torch.eye(
        cosine.shape[0], device=cosine.device, dtype=cosine.dtype
    )
    top8 = torch.topk(alpha, k=min(8, alpha.shape[1]), dim=1).values.sum(dim=1)
    return {
        "mean_assignment_entropy": float(entropy.mean()),
        "max_assignment_entropy": float(math.log(DICT_ATOMS)),
        "effective_atom_count": float(torch.exp(entropy.mean())),
        "active_atom_count": int((mean_mass > 1.0e-3).sum()),
        "argmax_used_atoms": int(torch.unique(alpha.argmax(dim=1)).numel()),
        "top8_assignment_mass": float(top8.mean()),
        "max_average_assignment_mass": float(mean_mass.max()),
        "min_average_assignment_mass": float(mean_mass.min()),
        "dictionary_coherence_mean_abs": float(off_diagonal.abs().mean()),
        "dictionary_coherence_max_abs": float(off_diagonal.abs().max()),
        "tau_final": float(model.dictionary.current_tau().detach()),
        "mean_topology_role_norm": float(roles.norm(dim=1).mean()),
        "mean_role_coordinate": [float(value) for value in roles.mean(dim=0)],
        "role_coordinate_std": [float(value) for value in roles.std(dim=0)],
        "n_patches": int(alpha.shape[0]),
    }


def intervention_shifts(
    model: DTXModel,
    loader,
    device: torch.device,
    predictions_normal: np.ndarray,
) -> dict[str, Any]:
    """Inference-only alignment interventions (no gradient, no retraining)."""
    normal = predictions_normal.astype(np.float64)
    payload: dict[str, Any] = {"arm": str(model.arm)}
    # Repeat-forward noise floor (same input, same weights).
    mae_repeat, _t, predictions_repeat = sdp._evaluate_mae(model, loader, device)
    repeat_noise = float(np.abs(normal - predictions_repeat.astype(np.float64)).max())
    payload["repeat_valid_mae"] = float(mae_repeat)
    payload["repeat_noise_max_abs"] = repeat_noise
    for mode in ("alignment_removal", "topology_shuffle"):
        model.set_inference_intervention(mode)
        try:
            mae, _targets, predictions = sdp._evaluate_mae(model, loader, device)
        finally:
            model.set_inference_intervention(None)
        shift = np.abs(normal - predictions.astype(np.float64))
        payload[mode] = {
            "valid_mae_after_intervention": float(mae),
            "mean_abs_prediction_shift": float(shift.mean()),
            "max_abs_prediction_shift": float(shift.max()),
            "frac_shift_gt_1e-6": float((shift > 1.0e-6).mean()),
            "clear_over_noise": bool(
                float(shift.mean()) >= INTERVENTION_MIN_MEAN_SHIFT
                and float(shift.mean())
                >= INTERVENTION_NOISE_MULTIPLE * max(repeat_noise, 1.0e-9)
            ),
        }
    return payload


def intervention_stage(
    arm: str = "aligned",
    which: str = "soup",
    device: str = "cpu",
    valid_subset: int | None = None,
) -> dict[str, Any]:
    if str(arm) not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    device_obj = torch.device(str(device))
    _train_data, valid_data, _audit = load_encoded(None, valid_subset)
    model = build_dtx(0, str(arm))
    state_path = STATE_DIR / f"dtx_{arm}_seed0_{which}_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    model.to(device_obj)
    loader = zpp._make_loader(valid_data, 128, False, 91012)
    _mae, _targets, predictions = sdp._evaluate_mae(model, loader, device_obj)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": str(arm),
        "which": str(which),
        "device": str(device_obj),
        "normal_valid_mae": float(_mae),
        "interventions": intervention_shifts(model, loader, device_obj, predictions),
        "gradient_used": False,
        "used_for_selection": False,
        "official_test_loaded": False,
    }
    payload["mechanism_clear"] = bool(
        payload["interventions"]["alignment_removal"]["clear_over_noise"]
        and payload["interventions"]["topology_shuffle"]["clear_over_noise"]
    )
    _write_json(RESULTS_DIR / f"interventions_{arm}_{which}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# post-hoc ring enrichment (EXPLICIT ring labels; never used for training or
# model selection, never fed to the model, no gradient)
# ---------------------------------------------------------------------------


def _ring_strata_for_split(split: str) -> list[np.ndarray]:
    """Per-graph (n_centres, 6) explicit ring annotations for post-hoc analysis.

    Strata columns follow :data:`POSTHOC_STRATA`:
    ``ring_any / ring5 / ring6 / multi_fused / ring_boundary / non_ring``.
    ``multi_fused`` is ``#covering cycles >= 2`` (any fused / spiro / bridged
    node satisfies this); ``ring_boundary`` is a non-ring node adjacent to a
    ring node.  This path calls the ring enumerator *only here*, after training.
    """
    from tracks.ksvd.experiments.luyin16 import structural_context as sc

    dataset = _load_zinc(ZINC_ROOT, split)
    output: list[np.ndarray] = []
    for raw in dataset:
        graph, _node_types, _edge_types = zpp._data_to_graph(raw)
        cycles = sc._find_cycles(graph, max_cycle_len=8)
        node_cycles: dict[int, list[int]] = defaultdict(list)
        for cycle in cycles:
            for node in cycle:
                node_cycles[int(node)].append(len(cycle))
        ring_nodes = set(node_cycles)
        centers = list(graph.nodes)
        rows = np.zeros((len(centers), len(POSTHOC_STRATA)), dtype=bool)
        for index, center in enumerate(centers):
            lengths = node_cycles.get(int(center), [])
            rows[index, 0] = bool(lengths)
            rows[index, 1] = 5 in lengths
            rows[index, 2] = 6 in lengths
            rows[index, 3] = len(lengths) >= 2
            rows[index, 4] = bool(
                (not lengths)
                and any(int(neighbor) in ring_nodes for neighbor in graph.neighbors(int(center)))
            )
            rows[index, 5] = not lengths
        output.append(rows)
    return output


def posthoc_ring_analysis(
    arm: str = "aligned", which: str = "soup", device: str = "cpu", top_k: int = 8
) -> dict[str, Any]:
    if str(arm) not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    device_obj = torch.device(str(device))
    _train_data, valid_data, _audit = load_encoded(None, None)
    model = build_dtx(0, str(arm))
    state_path = STATE_DIR / f"dtx_{arm}_seed0_{which}_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    model.to(device_obj)
    model.eval()

    alpha_chunks: list[torch.Tensor] = []

    def _cap_dict(_m, _i, output):
        alpha_chunks.append(output["alpha"].detach())

    handle = model.dictionary.register_forward_hook(_cap_dict)
    loader = zpp._make_loader(valid_data, 128, False, 0)
    try:
        with torch.no_grad():
            for batch in loader:
                model(batch.to(device_obj))
    finally:
        handle.remove()
    alpha = torch.cat(alpha_chunks, dim=0).to("cpu", torch.float64).numpy()
    roles = torch.cat([row.generic_topology for row in valid_data], dim=0)
    roles = roles.to("cpu", torch.float64).numpy()
    strata = np.concatenate(_ring_strata_for_split("val"), axis=0)
    if alpha.shape[0] != roles.shape[0] or alpha.shape[0] != strata.shape[0]:
        raise RuntimeError(
            f"post-hoc row mismatch: alpha {alpha.shape} roles {roles.shape} "
            f"strata {strata.shape}"
        )

    global_alpha = alpha.mean(axis=0)
    global_roles = roles.mean(axis=0)
    per_stratum: dict[str, Any] = {}
    joint_delta_by_stratum: dict[str, np.ndarray] = {}
    for column, name in enumerate(POSTHOC_STRATA):
        mask = strata[:, column]
        complement = ~mask
        in_count = int(mask.sum())
        if in_count == 0 or int(complement.sum()) == 0:
            per_stratum[name] = {"coverage": float(mask.mean()), "usable": False}
            continue
        mean_alpha_in = alpha[mask].mean(axis=0)
        mean_alpha_out = alpha[complement].mean(axis=0)
        enrichment = mean_alpha_in - mean_alpha_out
        order = np.argsort(-np.abs(enrichment))[: int(top_k)]
        joint_in = np.einsum("nk,nm->km", alpha[mask], roles[mask]) / in_count
        joint_out = (
            np.einsum("nk,nm->km", alpha[complement], roles[complement])
            / int(complement.sum())
        )
        joint_delta = joint_in - joint_out
        joint_delta_by_stratum[name] = joint_delta
        flat_order = np.argsort(-np.abs(joint_delta).reshape(-1))[: int(top_k)]
        per_stratum[name] = {
            "coverage": float(mask.mean()),
            "usable": True,
            "in_count": in_count,
            "atom_enrichment_top": [
                {
                    "atom": int(atom),
                    "mean_in": float(mean_alpha_in[atom]),
                    "mean_out": float(mean_alpha_out[atom]),
                    "delta": float(enrichment[atom]),
                }
                for atom in order
            ],
            "atom_enrichment_max_abs": float(np.abs(enrichment).max()),
            "role_means_in": [float(value) for value in roles[mask].mean(axis=0)],
            "role_means_out": [float(value) for value in roles[complement].mean(axis=0)],
            "role_delta": [
                float(value)
                for value in roles[mask].mean(axis=0) - roles[complement].mean(axis=0)
            ],
            "joint_enrichment_top": [
                {
                    "atom": int(index // TOPO_ROLE_DIM),
                    "role_coordinate": int(index % TOPO_ROLE_DIM),
                    "role_name": TOPO_ROLE_NAMES[int(index % TOPO_ROLE_DIM)],
                    "delta": float(joint_delta.reshape(-1)[index]),
                }
                for index in flat_order
            ],
            "joint_enrichment_max_abs": float(np.abs(joint_delta).max()),
        }

    # Global (graph-level) aligned joint: which learned environment x generic
    # role pairs are the largest in mean magnitude, and how ring-enriched they
    # are per stratum.
    graphs: list[np.ndarray] = []
    start = 0
    for row in valid_data:
        count = int(row.generic_topology.shape[0])
        slice_alpha = alpha[start : start + count]
        slice_roles = roles[start : start + count]
        graphs.append(np.einsum("nk,nm->km", slice_alpha, slice_roles) / count)
        start += count
    mean_joint = np.mean(np.stack(graphs, axis=0), axis=0)
    top_flat = np.argsort(-np.abs(mean_joint).reshape(-1))[: int(top_k)]
    joint_summary = []
    for index in top_flat:
        atom = int(index // TOPO_ROLE_DIM)
        role_coordinate = int(index % TOPO_ROLE_DIM)
        entry: dict[str, Any] = {
            "atom": atom,
            "role_coordinate": role_coordinate,
            "role_name": TOPO_ROLE_NAMES[role_coordinate],
            "mean_joint": float(mean_joint.reshape(-1)[index]),
        }
        for name in POSTHOC_STRATA:
            delta = joint_delta_by_stratum.get(name)
            if delta is not None:
                entry[f"joint_delta_{name}"] = float(delta.reshape(-1)[index])
        joint_summary.append(entry)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": str(arm),
        "which": str(which),
        "n_patches": int(alpha.shape[0]),
        "global_alpha_mean_max": float(global_alpha.max()),
        "global_role_mean": [float(value) for value in global_roles],
        "stratum_coverage": {
            name: float(strata[:, column].mean())
            for column, name in enumerate(POSTHOC_STRATA)
        },
        "per_stratum": per_stratum,
        "top_aligned_joint_entries": joint_summary,
        "explicit_ring_labels_used": True,
        "post_hoc_only": True,
        "gradient_used": False,
        "used_for_selection": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"posthoc_ring_{arm}_{which}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# training / soup (faithful mirror of the inherited optimized protocol)
# ---------------------------------------------------------------------------


def train_arm(
    *,
    arm: str,
    seed: int = 0,
    device: str = "cpu",
    max_epochs: int = 240,
    train_subset: int | None = None,
    valid_subset: int | None = None,
    tag: str | None = None,
    save_state: bool = True,
    soup: bool = True,
) -> dict[str, Any]:
    if str(arm) not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    device_obj = torch.device(str(device))
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    train_data, valid_data, audit = load_encoded(train_subset, valid_subset)
    if int(audit["local_input_width"]) != LOCAL_INPUT_WIDTH:
        raise RuntimeError("local descriptor width drift")
    if int(audit["dictionary_input_width"]) != DICT_INPUT_WIDTH:
        raise RuntimeError("dictionary input width drift")

    protocol = dict(sdp.OPTIMIZED_PROTOCOL)
    protocol["max_epochs"] = int(max_epochs)
    protocol["early_termination"] = "none"

    model = build_dtx(seed=seed, arm=str(arm)).to(device_obj)
    total_params = _n_params(model)

    first_batch = _first_batch(valid_data, device_obj, limit=min(128, len(valid_data)))
    widths = runtime_width_audit(model, first_batch)
    if not widths["widths_consistent"]:
        raise RuntimeError(f"runtime width audit failed: {widths}")
    contract = static_contract_checks(model, first_batch)
    if not contract["passed"]:
        raise RuntimeError(f"strict-static integrity gates failed: {contract}")
    viability = gradient_viability_checks(model, first_batch)
    if not viability["branch_receives_gradient"]:
        raise RuntimeError(f"DTX branch receives no step-0 gradient: {viability}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    loader = zpp._make_loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = zpp._make_loader(
        valid_data,
        int(protocol["batch_size"]),
        False,
        int(seed) + int(protocol["eval_shuffle_seed_offset"]),
    )
    steps_per_epoch = int(math.ceil(len(train_data) / int(protocol["batch_size"])))
    clip = float(protocol["gradient_clip_norm"])

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}

    started = time.perf_counter()
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae, _, _ = sdp._evaluate_mae(model, eval_loader, device_obj)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_mae": float(train_mae),
            "valid_mae": float(valid_mae),
            "tau": float(model.dictionary.current_tau().detach()),
        }
        curve.append(row)
        epoch_states[epoch] = {
            key: value.detach().to("cpu", copy=True)
            for key, value in model.state_dict().items()
        }
        keep = set(sdp._topk_epochs(curve, "valid_mae", 5))
        for cached_epoch in list(epoch_states):
            if cached_epoch not in keep:
                del epoch_states[cached_epoch]
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().to("cpu", copy=True)
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            print(
                f"[dtx_{arm} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"tau={row['tau']:.4f}",
                flush=True,
            )
    wall_clock = float(time.perf_counter() - started)

    for row in curve:
        row["checkpoint_selected"] = int(row["epoch"] == best_epoch)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device_obj)

    best_valid, valid_targets, best_predictions = sdp._evaluate_mae(
        model, eval_loader, device_obj
    )

    soup_payload: dict[str, Any] = {"available": False}
    soup_state_cpu: dict[str, torch.Tensor] | None = None
    if soup and epoch_states:
        members = sorted(sdp._topk_epochs(curve, "valid_mae", 5))
        keys = list(epoch_states[members[0]].keys())
        soup_state = {
            key: torch.stack([epoch_states[epoch][key].float() for epoch in members]).mean(0)
            for key in keys
        }
        soup_model = build_dtx(seed=seed, arm=str(arm))
        soup_model.load_state_dict(soup_state)
        soup_model.to(device_obj)
        soup_mae, _, soup_predictions = sdp._evaluate_mae(soup_model, eval_loader, device_obj)
        soup_payload = {
            "available": True,
            "members": [int(epoch) for epoch in members],
            "member_valid_mae": [
                float(curve[epoch - 1]["valid_mae"]) for epoch in members
            ],
            "soup_valid_mae": float(soup_mae),
            "soup_predictions": soup_predictions.tolist(),
        }
        soup_state_cpu = {
            key: value.detach().to("cpu", copy=True) for key, value in soup_state.items()
        }
        del soup_model

    diagnostics = dictionary_diagnostics(model, eval_loader, device_obj)
    intervention = intervention_shifts(model, eval_loader, device_obj, best_predictions)

    curve_path = CURVE_DIR / f"dtx_{arm}_seed{seed}_curve.csv"
    _write_csv(curve_path, curve)
    state_paths: dict[str, str] = {}
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        best_path = STATE_DIR / f"dtx_{arm}_seed{seed}_best_state.pt"
        torch.save(model.state_dict(), best_path)
        state_paths["best"] = str(best_path)
        if soup_state_cpu is not None:
            soup_path = STATE_DIR / f"dtx_{arm}_seed{seed}_soup_state.pt"
            torch.save(soup_state_cpu, soup_path)
            state_paths["soup"] = str(soup_path)

    peak_memory_mb = None
    if device_obj.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))

    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "arm": str(arm),
        "seed": int(seed),
        "tag": str(tag or f"dtx_{arm}"),
        "device": str(device_obj),
        "protocol": protocol,
        "max_epochs": int(max_epochs),
        "epochs_run": int(len(losses)),
        "early_terminated": bool(len(losses) < int(max_epochs)),
        "steps_per_epoch": int(steps_per_epoch),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(best_valid),
        "wall_clock_s": float(wall_clock),
        "peak_gpu_memory_mb": peak_memory_mb,
        "parameters": int(total_params),
        "tau_final": float(model.dictionary.current_tau().detach()),
        "runtime_widths": widths,
        "strict_static_contract": contract,
        "gradient_viability": viability,
        "dictionary_diagnostics": diagnostics,
        "intervention": intervention,
        "soup": soup_payload,
        "curve": curve,
        "valid_predictions": best_predictions.tolist(),
        "valid_targets": valid_targets.tolist(),
        "curve_path": str(curve_path),
        "state_paths": state_paths,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(str(device_obj)),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_DIR / f"dtx_{arm}_seed{seed}.json", summary)
    print(
        f"[dtx_{arm} seed{seed}] best_valid={best_mae:.6f} epoch={best_epoch} "
        f"soup={soup_payload.get('soup_valid_mae')} params={total_params} "
        f"wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# smoke / determinism (same seed, GPU0 vs GPU1)
# ---------------------------------------------------------------------------


def smoke_stage(
    device: str = "cuda",
    tag: str = "smoke",
    arm: str = "aligned",
    epochs: int = 2,
    train_subset: int = 256,
    valid_subset: int = 128,
) -> dict[str, Any]:
    summary = train_arm(
        arm=str(arm),
        seed=0,
        device=device,
        max_epochs=int(epochs),
        train_subset=int(train_subset),
        valid_subset=int(valid_subset),
        tag=str(tag),
        save_state=False,
        soup=False,
    )
    model = build_dtx(0, str(arm))
    trace = [
        [int(row["epoch"]), float(row["train_mae"]), float(row["valid_mae"])]
        for row in summary["curve"]
    ]
    parameter_hash = hashlib.sha256()
    for key in sorted(model.state_dict()):
        parameter_hash.update(key.encode())
        parameter_hash.update(model.state_dict()[key].detach().cpu().numpy().tobytes())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "arm": str(arm),
        "device": str(device),
        "epochs": int(epochs),
        "train_subset": int(train_subset),
        "valid_subset": int(valid_subset),
        "trace": trace,
        "parameter_init_sha256": parameter_hash.hexdigest(),
        "best_valid_mae": summary["best_valid_mae"],
        "parameters": summary["parameters"],
        "runtime_widths": summary["runtime_widths"],
        "contract_passed": summary["strict_static_contract"]["passed"],
        "gradient_viability": summary["gradient_viability"],
        "dictionary_diagnostics": summary["dictionary_diagnostics"],
        "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
        "deterministic_algorithms": summary["deterministic_algorithms"],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"smoke_{tag}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# analysis / outcome classification
# ---------------------------------------------------------------------------


def classify_outcome(
    *,
    marginal_best: float,
    marginal_soup: float,
    aligned_best: float,
    aligned_soup: float,
    mechanism_clear: bool | None,
    interventions: Mapping[str, Any] | None,
) -> dict[str, Any]:
    delta_best = float(marginal_best) - float(aligned_best)
    delta_soup = float(marginal_soup) - float(aligned_soup)
    safety = float(marginal_soup) > GATE_MARGINAL_SAFETY
    strong = bool(
        delta_soup >= GATE_ALIGN_SOUP_DELTA
        and delta_best >= GATE_ALIGN_BEST_DELTA
        and float(aligned_soup) <= GATE_ALIGN_ABS_SOUP
        and bool(mechanism_clear)
    )
    if strong:
        case = "A"
        verdict = "NO_RING_DICT_TOPOLOGY_ALIGNMENT_SIGNAL"
    elif delta_soup >= GATE_DIRECTIONAL_SOUP_DELTA:
        case = "B"
        verdict = "ALIGNMENT_SIGNAL_BUT_BASE_NOT_COMPETITIVE"
    elif 0.0 < delta_soup < GATE_DIRECTIONAL_SOUP_DELTA:
        case = "C"
        verdict = "DIRECTIONAL_ONLY"
    else:
        case = "D"
        verdict = "NO_ALIGNED_CROSS_SIGNAL"
    return {
        "marginal_best": float(marginal_best),
        "marginal_soup": float(marginal_soup),
        "aligned_best": float(aligned_best),
        "aligned_soup": float(aligned_soup),
        "delta_align_best": delta_best,
        "delta_align_soup": delta_soup,
        "gate_align_soup_delta": GATE_ALIGN_SOUP_DELTA,
        "gate_align_best_delta": GATE_ALIGN_BEST_DELTA,
        "gate_align_abs_soup": GATE_ALIGN_ABS_SOUP,
        "gate_directional_soup_delta": GATE_DIRECTIONAL_SOUP_DELTA,
        "mechanism_clear": None if mechanism_clear is None else bool(mechanism_clear),
        "interventions": dict(interventions) if interventions else None,
        "marginal_safety_threshold": GATE_MARGINAL_SAFETY,
        "base_regression_confound": bool(safety),
        "case": case,
        "verdict": verdict,
        "seed1_authorized": bool(case == "A"),
        "official_test_loaded": False,
    }


def analyze_stage() -> dict[str, Any]:
    marginal_path = RUNS_DIR / "dtx_marginal_seed0.json"
    aligned_path = RUNS_DIR / "dtx_aligned_seed0.json"
    if not marginal_path.exists() or not aligned_path.exists():
        raise FileNotFoundError(
            f"missing run outputs: {marginal_path} / {aligned_path}"
        )
    marginal = _read_json(marginal_path)
    aligned = _read_json(aligned_path)
    intervention_path = RESULTS_DIR / "interventions_aligned_soup.json"
    interventions = _read_json(intervention_path) if intervention_path.exists() else None
    mechanism_clear = (
        bool(interventions["mechanism_clear"]) if interventions is not None else None
    )
    outcome = classify_outcome(
        marginal_best=marginal["best_valid_mae"],
        marginal_soup=marginal["soup"]["soup_valid_mae"],
        aligned_best=aligned["best_valid_mae"],
        aligned_soup=aligned["soup"]["soup_valid_mae"],
        mechanism_clear=mechanism_clear,
        interventions=interventions,
    )
    seed1_purchased = (RUNS_DIR / "dtx_marginal_seed1.json").exists() or (
        RUNS_DIR / "dtx_aligned_seed1.json"
    ).exists()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "primary_metric": "fixed Top-5 weight soup official-valid MAE",
        "secondary_metric": "best-checkpoint official-valid MAE",
        "causal_comparison": "marginal-only control (Arm M) vs aligned cross (Arm A)",
        "references": {
            "s0_seed0": {"best": S0_SEED0_BEST, "soup": S0_SEED0_SOUP},
            "s0_seed1": {"best": S0_SEED1_BEST, "soup": S0_SEED1_SOUP},
            "sdpk_seed0": {"best": SDPK_SEED0_BEST, "soup": SDPK_SEED0_SOUP},
            "srda_seed0": {"best": SRDA_SEED0_BEST, "soup": SRDA_SEED0_SOUP},
        },
        "marginal": {
            "best_valid_mae": marginal["best_valid_mae"],
            "best_epoch": marginal["best_epoch"],
            "soup_valid_mae": marginal["soup"]["soup_valid_mae"],
            "soup_members": marginal["soup"]["members"],
            "parameters": marginal["parameters"],
            "wall_clock_s": marginal["wall_clock_s"],
            "peak_gpu_memory_mb": marginal.get("peak_gpu_memory_mb"),
            "device": marginal["device"],
            "dict_diagnostics": marginal["dictionary_diagnostics"],
            "tau_final": marginal["tau_final"],
        },
        "aligned": {
            "best_valid_mae": aligned["best_valid_mae"],
            "best_epoch": aligned["best_epoch"],
            "soup_valid_mae": aligned["soup"]["soup_valid_mae"],
            "soup_members": aligned["soup"]["members"],
            "parameters": aligned["parameters"],
            "wall_clock_s": aligned["wall_clock_s"],
            "peak_gpu_memory_mb": aligned.get("peak_gpu_memory_mb"),
            "device": aligned["device"],
            "dict_diagnostics": aligned["dictionary_diagnostics"],
            "tau_final": aligned["tau_final"],
        },
        "outcome": outcome,
        "parity_gate": _read_json(RESULTS_DIR / "parity_gates.json")
        if (RESULTS_DIR / "parity_gates.json").exists()
        else None,
        "budget": {
            "max_full_runs": MAX_FULL_RUNS,
            "full_runs_used": 2,
            "seed1_purchased": bool(seed1_purchased),
            "sweep": False,
            "hpo": False,
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "analysis.json", payload)
    _write_summary_markdown(payload)
    return payload


def _write_summary_markdown(payload: Mapping[str, Any]) -> None:
    outcome = payload["outcome"]
    marginal = payload["marginal"]
    aligned = payload["aligned"]
    lines: list[str] = []
    lines.append(
        "# DTX-v0 -- no-ring dictionary x generic topology cross: results summary"
    )
    lines.append("")
    lines.append(
        "Question: can an **aligned** joint statistic of learned dictionary "
        "assignments and *generic, untyped, label-free* topology roles beat a "
        "matched marginal-only control, without ever reading an explicit ring / "
        "cycle context?"
    )
    lines.append("")
    lines.append("Official ZINC **test was never loaded** in any stage.")
    lines.append("")
    lines.append("## Primary metric — fixed Top-5 weight soup (official-valid MAE)")
    lines.append("")
    lines.append("| arm | best MAE | best epoch | soup MAE | soup members | params | wall s |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(
        f"| MARGINAL (control) | {marginal['best_valid_mae']:.6f} | "
        f"{marginal['best_epoch']} | {marginal['soup_valid_mae']:.6f} | "
        f"{marginal['soup_members']} | {marginal['parameters']} | "
        f"{marginal['wall_clock_s']:.1f} |"
    )
    lines.append(
        f"| ALIGNED (cross) | {aligned['best_valid_mae']:.6f} | "
        f"{aligned['best_epoch']} | {aligned['soup_valid_mae']:.6f} | "
        f"{aligned['soup_members']} | {aligned['parameters']} | "
        f"{aligned['wall_clock_s']:.1f} |"
    )
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    lines.append(f"* Delta_align (soup) = {outcome['delta_align_soup']:+.6f}")
    lines.append(f"* Delta_align (best) = {outcome['delta_align_best']:+.6f}")
    lines.append(f"* case: **{outcome['case']}**")
    lines.append(f"* verdict: **{outcome['verdict']}**")
    lines.append(
        f"* base regression confound (M soup > {outcome['marginal_safety_threshold']}): "
        f"{outcome['base_regression_confound']}"
    )
    lines.append(f"* seed 1 authorized: {outcome['seed1_authorized']}")
    interventions = outcome.get("interventions")
    if interventions:
        lines.append("")
        lines.append("## Alignment-use interventions (Arm A)")
        lines.append("")
        lines.append("| intervention | mean |dpred| | max |dpred| | valid MAE | clear |")
        lines.append("|---|---|---|---|---|")
        for mode in ("alignment_removal", "topology_shuffle"):
            block = interventions["interventions"][mode]
            lines.append(
                f"| {mode} | {block['mean_abs_prediction_shift']:.6f} | "
                f"{block['max_abs_prediction_shift']:.6f} | "
                f"{block['valid_mae_after_intervention']:.6f} | "
                f"{block['clear_over_noise']} |"
            )
        lines.append(
            f"* repeat-forward noise floor (max |dpred|): "
            f"{interventions['interventions']['repeat_noise_max_abs']:.3e}"
        )
    lines.append("")
    lines.append("## Dictionary / topology diagnostics")
    lines.append("")
    for label, block in (("MARGINAL", marginal), ("ALIGNED", aligned)):
        diag = block["dict_diagnostics"]
        lines.append(
            f"* {label}: effective atoms {diag['effective_atom_count']:.2f} / "
            f"argmax-used {diag['argmax_used_atoms']} / top-8 mass "
            f"{diag['top8_assignment_mass']:.3f} / tau {diag['tau_final']:.4f}"
        )
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    budget = payload["budget"]
    lines.append(
        f"* full training runs used: {budget['full_runs_used']} / {budget['max_full_runs']}"
    )
    lines.append(f"* seed 1 purchased: {budget['seed1_purchased']}")
    lines.append("* sweeps / HPO: none")
    lines.append("* official test accessed = false")
    lines.append("")
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_all() -> None:
    parameter_audit()
    prepare_encoded()
    parity_gate()
    integrity_stage()
    analyze_stage()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "param_audit",
            "encode",
            "parity",
            "integrity",
            "smoke",
            "train",
            "intervene",
            "posthoc",
            "analyze",
            "all",
        ],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--arm", default="aligned", choices=list(ARMS))
    parser.add_argument("--which", default="soup", choices=["best", "soup"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--valid-subset", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    _set_deterministic(bool(args.deterministic) or str(args.device).startswith("cuda"))
    if args.stage == "param_audit":
        print(json.dumps(parameter_audit(), indent=2))
    elif args.stage == "encode":
        print(json.dumps(prepare_encoded(), indent=2))
    elif args.stage == "parity":
        print(json.dumps(parity_gate(), indent=2))
    elif args.stage == "integrity":
        print(json.dumps(integrity_stage(), indent=2))
    elif args.stage == "smoke":
        print(
            json.dumps(
                smoke_stage(
                    args.device,
                    tag=str(args.tag or "smoke"),
                    arm=str(args.arm),
                    epochs=int(args.epochs),
                ),
                indent=2,
            )
        )
    elif args.stage == "train":
        summary = train_arm(
            arm=str(args.arm),
            seed=int(args.seed),
            device=str(args.device),
            max_epochs=int(args.epochs),
            train_subset=args.train_subset,
            valid_subset=args.valid_subset,
            tag=args.tag,
        )
        print(json.dumps({k: v for k, v in summary.items() if "predictions" not in k}, indent=2))
    elif args.stage == "intervene":
        print(
            json.dumps(
                intervention_stage(str(args.arm), str(args.which), args.device, args.valid_subset),
                indent=2,
            )
        )
    elif args.stage == "posthoc":
        print(
            json.dumps(
                posthoc_ring_analysis(str(args.arm), str(args.which), args.device),
                indent=2,
            )
        )
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2))
    else:
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
