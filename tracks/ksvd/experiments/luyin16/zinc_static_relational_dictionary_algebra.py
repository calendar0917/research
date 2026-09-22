"""SRDA-v0 -- Static Relational Dictionary Algebra (strict-static ZINC).

Round: **ZINC-static-relational-dictionary-algebra-v0** (SRDA-v0).

One performance-oriented question
---------------------------------
In a *strictly* static graph model (no learned message passing of any kind), if

* the already-proven-unnecessary **typed-token** identity lookup (16D) and
  **parent** identity lookup (8D) are deleted outright, and
* the end-to-end dictionary assignment is made to **define the occurrence-level
  relation algebra directly** (instead of being compressed into a generic
  16D coordinate and handed to a generic pair MLP),

does the absolute ZINC MAE enter a materially better band?

Strict-static contract (non-negotiable, inherited)
--------------------------------------------------
::

    NO message passing
    NO pair/relation state -> centre/patch aggregation -> local hidden update
    NO relation refresh
    NO second pair evaluation
    NO attention / iterative refinement

Every local state is fully computed before any pair/relation information is
read, and that ordering is structural in the code below: ``encode`` computes
``z`` (and the whole unary readout) first and passes only ``alpha`` /
``epsilon`` / static relation features to the occurrence-level pair algebra.

Architecture (frozen; no sweep)
-------------------------------
::

    x_i     = [patch_cont | patch_context]                 # 146D static descriptor
    z_i     = SiLU MLP(146 -> 96 -> 64)                    # local state, per patch
    q_i     = l2_normalize(Wq(z_i))                        # 64 -> 32
    D_k     = l2_normalize(D)                              # K = 64, rank = 32
    tau     = 0.05 + 0.95 * sigmoid(tau_logit)             # tau_init = 0.20
    alpha_i = softmax((q_i @ D_norm^T) / tau)              # 64D assignment
    a_i     = alpha_i @ A                                  # 64 -> 32
    b_i     = alpha_i @ B                                  # 64 -> 32
    q_rec_i = alpha_i @ D_norm                             # 32
    eps_i   = q_i - q_rec_i                                # 32
    e_i     = W_eps(eps_i)                                 # 32 -> 8

    r_ij          -> rel_hidden_ij (32) -> rel_feat_ij (16)
    rel_gate_ij   = 1 + tanh(G(rel_hidden_ij) + bucket_embedding[bucket_ij])

    proto_pair_ij = 0.5 * (a_i * b_j + a_j * b_i)           # 32
    p_dict_ij     = proto_pair_ij * rel_gate_ij             # 32
    p_eps_ij      = [e_i+e_j | |e_i-e_j| | e_i*e_j]         # 24
    pair_input_ij = [p_dict | p_eps | rel_feat]             # 72
    q_ij          = SiLU MLP(72 -> 64 -> 32)                # ONCE per pair

    unary  = mean_std_moments(z)                            # 129
    pair   = 5 distance buckets x mean_std_moments(q_ij)    # 325
    global = GlobalEncoder(62)                              # 32
    topo   = TopologyHinge(25 -> 16 -> 8)                   # 8
    R      = [unary | pair | global | topo]                 # 494
    y_hat  = GenericReader(R, (16, 16))                     # -> 1

The mathematical core is the low-rank contraction

::

    p_dict_ij[a] = sum_{k,l} alpha_i[k] alpha_j[l]
                   0.5 * (A[k,a] B[l,a] + A[l,a] B[k,a]) * g_a(r_ij)

i.e. the dictionary prototypes themselves (not a compressed generic
coordinate) carry the relational interaction, and the 8D residual is the only
allowed endpoint-continuous correction path.

Forbidden: K-SVD / reconstruction / entropy / sparsity / balance / top-k /
orthogonality / contrastive auxiliary objectives; typed-token or parent
identity lookups of any kind (raw, hashed, corrected, historical, fixed-code,
certificate-lookup); patch-internal learned message passing; any raw
``z_i``/``z_j`` bypass into the pair kernel.

Frozen training protocol (inherited optimized strict-static A100 regime)
------------------------------------------------------------------------
Adam, lr 1e-3, weight decay 1e-5, batch 128, 240 epochs, no scheduler, L1
loss, gradient clip 5.0, full horizon (no early termination).

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_static_relational_dictionary_algebra <stage>

Stages: ``param_audit integrity encode smoke train analyze all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)

# ---------------------------------------------------------------------------
# frozen constants (no sweep)
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/zinc_static_relational_dictionary_algebra_v0"
RUNS_DIR = RESULTS_DIR / "runs"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"

EXPERIMENT_NAME = "zinc_static_relational_dictionary_algebra_v0"
PROTOCOL_VERSION = "zinc_static_relational_dictionary_algebra_v0"

# --- local state ------------------------------------------------------------
LOCAL_INPUT_WIDTH = 146  # runtime-attested: patch_cont(146) + patch_context(0)
LOCAL_HIDDEN = 96
LOCAL_DIM = 64
DROPOUT = 0.05

# --- end-to-end dictionary --------------------------------------------------
DICT_ATOMS = 64
DICT_RANK = 32
DICT_TAU_INIT = 0.20

# --- relational dictionary algebra -----------------------------------------
TENSOR_RANK = 32
RESIDUAL_DIM = 8
REL_TRUNK_HIDDEN = 32
REL_FEAT_DIM = 16
PAIR_HIDDEN = 32
PAIR_ENCODER_HIDDEN = 64
PAIR_INPUT_WIDTH = TENSOR_RANK + 3 * RESIDUAL_DIM + REL_FEAT_DIM  # 72

# --- graph-level reader -----------------------------------------------------
GRAPH_HEAD_HIDDEN = (16, 16)

# --- training protocol (inherited optimized strict-static regime) ----------
OPTIMIZED_PROTOCOL: dict[str, Any] = dict(sdp.OPTIMIZED_PROTOCOL)
OPTIMIZED_PROTOCOL["early_termination"] = "none (full 240 epochs)"

# --- pre-registered seed-0 performance gates -------------------------------
S0_SEED0_BEST = 0.145674
S0_SEED0_SOUP = 0.140794
S0_SEED1_BEST = 0.139389
S0_SEED1_SOUP = 0.136423
SDPK_SEED0_BEST = 0.14219318306347123
SDPK_SEED0_SOUP = 0.13973486851429334

GATE_SEED0_SOUP = 0.1340        # absolute Top-5 soup gate (primary)
GATE_SEED0_BEST = 0.1385        # absolute best-checkpoint gate
GATE_SEED1_SOUP = 0.1335        # strong confirmation (absolute)
GATE_SEED1_SOUP_DELTA = 0.0030  # or S0 seed1 soup - SRDA seed1 soup >= this

MAX_FULL_RUNS = 2


# ---------------------------------------------------------------------------
# io / determinism helpers (inherited)
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


def _configure_determinism() -> None:
    sdp._configure_determinism()


def _seed_everything(seed: int) -> None:
    sdp._seed_everything(seed)


def _tau_from_logit(tau_logit: torch.Tensor) -> torch.Tensor:
    return 0.05 + 0.95 * torch.sigmoid(tau_logit)


def _tau_logit_for(tau: float) -> float:
    return sdp._tau_logit_for(tau)


# ---------------------------------------------------------------------------
# graph-level invariant pooling (inherited mean/std/log-count semantics)
# ---------------------------------------------------------------------------


def bucket_moments(
    value: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_bucket: torch.Tensor,
    n_graphs: int,
    buckets: int = zpp.DISTANCE_BUCKETS,
) -> torch.Tensor:
    """Per distance bucket ``[mean | std | log1p(count)]`` over pair rows.

    Semantics are identical to the inherited strict-static ``mean_std`` pool
    (variance clamped at 0, ``sqrt(var + 1e-8)``), evaluated independently for
    every bucket so the graph representation keeps explicit relation mass.
    """
    blocks: list[torch.Tensor] = []
    width = int(value.shape[1])
    for bucket in range(int(buckets)):
        mask = pair_bucket == int(bucket)
        current = value[mask]
        current_batch = pair_batch[mask]
        total = torch.zeros((int(n_graphs), width), device=value.device, dtype=value.dtype)
        squared = torch.zeros_like(total)
        counts = torch.zeros((int(n_graphs), 1), device=value.device, dtype=value.dtype)
        if current.numel():
            total.index_add_(0, current_batch, current)
            squared.index_add_(0, current_batch, current * current)
            counts.index_add_(
                0,
                current_batch,
                torch.ones(
                    (current_batch.shape[0], 1), device=value.device, dtype=value.dtype
                ),
            )
        mean = total / counts.clamp_min(1.0)
        variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
        std = torch.sqrt(variance + 1.0e-8)
        blocks.append(torch.cat([mean, std, torch.log1p(counts)], dim=1))
    return torch.cat(blocks, dim=1)


def node_moments(value: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Graph-level ``[mean | std | log1p(count)]`` over patch rows."""
    total = torch.zeros(
        (int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype
    )
    squared = torch.zeros_like(total)
    counts = torch.zeros((int(n_graphs), 1), device=value.device, dtype=value.dtype)
    total.index_add_(0, batch, value)
    squared.index_add_(0, batch, value * value)
    counts.index_add_(
        0,
        batch,
        torch.ones((batch.shape[0], 1), device=value.device, dtype=value.dtype),
    )
    mean = total / counts.clamp_min(1.0)
    variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
    std = torch.sqrt(variance + 1.0e-8)
    return torch.cat([mean, std, torch.log1p(counts)], dim=1)


def _graph_mean(value: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    total = torch.zeros(
        (int(n_graphs), value.shape[1]), device=value.device, dtype=value.dtype
    )
    counts = torch.zeros((int(n_graphs), 1), device=value.device, dtype=value.dtype)
    total.index_add_(0, batch, value)
    counts.index_add_(
        0,
        batch,
        torch.ones((batch.shape[0], 1), device=value.device, dtype=value.dtype),
    )
    return total / counts.clamp_min(1.0)


# ---------------------------------------------------------------------------
# end-to-end dictionary assignment + prototype factors
# ---------------------------------------------------------------------------


class StaticDictionaryAlgebra(nn.Module):
    """``z -> (alpha, a, b, q_recon, eps)``.

    A pure, single-pass function of the local patch state.  It never sees pair
    or relation information, so it cannot perform message passing and it can
    never be refreshed by a pair result.
    """

    def __init__(
        self,
        local_dim: int = LOCAL_DIM,
        rank: int = DICT_RANK,
        atoms: int = DICT_ATOMS,
        tensor_rank: int = TENSOR_RANK,
        tau_init: float = DICT_TAU_INIT,
    ) -> None:
        super().__init__()
        self.local_dim = int(local_dim)
        self.rank = int(rank)
        self.atoms_count = int(atoms)
        self.tensor_rank = int(tensor_rank)
        self.query = nn.Linear(self.local_dim, self.rank)
        self.atoms = nn.Parameter(torch.empty(self.atoms_count, self.rank))
        self.factor_a = nn.Parameter(torch.empty(self.atoms_count, self.tensor_rank))
        self.factor_b = nn.Parameter(torch.empty(self.atoms_count, self.tensor_rank))
        self.tau_logit = nn.Parameter(torch.tensor(_tau_logit_for(tau_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.query.reset_parameters()
        nn.init.normal_(self.atoms, std=1.0 / math.sqrt(self.rank))
        # Prototype-factor scale (one frozen choice, not a sweep): ``a_i`` and
        # ``b_i`` are probability-weighted reads of the factor rows, so the
        # prototype-pair block scales as ``std^2``.  ``std = 2.0`` puts the
        # prototype-pair block and the narrow residual block at comparable
        # magnitude at initialisation on the real encoded descriptors (see
        # ``integrity_gates.json -> pair_block_scale_at_init``); the round's
        # hypothesis requires the prototype algebra to be the primary
        # relational object, not an order of magnitude below the residual.
        nn.init.normal_(self.factor_a, std=2.0)
        nn.init.normal_(self.factor_b, std=2.0)

    def current_tau(self) -> torch.Tensor:
        return _tau_from_logit(self.tau_logit)

    def normalized_atoms(self) -> torch.Tensor:
        return F.normalize(self.atoms, dim=-1, eps=1.0e-8)

    def alpha_from_z(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = F.normalize(self.query(z), dim=-1, eps=1.0e-8)
        atoms = self.normalized_atoms()
        tau = self.current_tau()
        alpha = F.softmax((query @ atoms.t()) / tau, dim=-1)
        return alpha, tau, atoms

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        alpha, _tau, atoms = self.alpha_from_z(z)
        a = alpha @ self.factor_a
        b = alpha @ self.factor_b
        q_recon = alpha @ atoms
        eps = F.normalize(self.query(z), dim=-1, eps=1.0e-8) - q_recon
        return {"alpha": alpha, "a": a, "b": b, "q_recon": q_recon, "eps": eps}


# ---------------------------------------------------------------------------
# SRDA model
# ---------------------------------------------------------------------------


class SRDAModel(nn.Module):
    """Strict-static relational dictionary algebra over static patch descriptors.

    Deterministic single pass.  ``center_context`` / ``center_update`` exist as
    explicit ``False`` / ``None`` class attributes so the inherited
    strict-static contract test can assert them, and so that monkeypatching the
    historical ``_pool_pairs_to_centres`` to raise cannot affect this model.
    """

    center_context: bool = False
    center_update: None = None

    def __init__(
        self,
        input_width: int = LOCAL_INPUT_WIDTH,
        *,
        local_hidden: int = LOCAL_HIDDEN,
        local_dim: int = LOCAL_DIM,
        dropout: float = DROPOUT,
        dict_atoms: int = DICT_ATOMS,
        dict_rank: int = DICT_RANK,
        dict_tau_init: float = DICT_TAU_INIT,
        tensor_rank: int = TENSOR_RANK,
        residual_dim: int = RESIDUAL_DIM,
        rel_trunk_hidden: int = REL_TRUNK_HIDDEN,
        rel_feat_dim: int = REL_FEAT_DIM,
        pair_hidden: int = PAIR_HIDDEN,
        pair_encoder_hidden: int = PAIR_ENCODER_HIDDEN,
        head_hidden: Sequence[int] = GRAPH_HEAD_HIDDEN,
        relation_width: int = zpp.RELATION_WIDTH,
        global_width: int = zpp.GLOBAL_WIDTH,
        topology_width: int = 25,
        topology_hidden: int = 16,
        topology_out: int = 8,
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.local_hidden = int(local_hidden)
        self.local_dim = int(local_dim)
        self.dropout = float(dropout)
        self.residual_dim = int(residual_dim)
        self.pair_hidden = int(pair_hidden)
        self.relation_width = int(relation_width)
        self.global_width = int(global_width)
        self.topology_width = int(topology_width)

        # --- independent shared local encoder (all patches, no cross-talk) --
        self.local_encoder = nn.Sequential(
            nn.Linear(self.input_width, self.local_hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.local_hidden, self.local_dim),
        )

        # --- end-to-end dictionary assignment + prototype factors ----------
        self.dictionary = StaticDictionaryAlgebra(
            self.local_dim,
            int(dict_rank),
            int(dict_atoms),
            int(tensor_rank),
            float(dict_tau_init),
        )

        # --- narrow within-prototype residual ---------------------------------
        self.residual_proj = nn.Linear(int(dict_rank), self.residual_dim)

        # --- static relation trunk (pair information, read-only) ---------------
        self.relation_trunk = nn.Sequential(
            nn.Linear(self.relation_width, int(rel_trunk_hidden)),
            nn.SiLU(),
        )
        self.relation_gate = nn.Linear(int(rel_trunk_hidden), int(tensor_rank))
        self.bucket_embedding = nn.Embedding(zpp.DISTANCE_BUCKETS, int(tensor_rank))
        nn.init.zeros_(self.bucket_embedding.weight)
        self.relation_feature = nn.Linear(int(rel_trunk_hidden), int(rel_feat_dim))

        # --- one occurrence-level pair kernel (evaluated exactly once) --------
        self.pair_input_width = (
            int(tensor_rank) + 3 * self.residual_dim + int(rel_feat_dim)
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(self.pair_input_width, int(pair_encoder_hidden)),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(pair_encoder_hidden), self.pair_hidden),
        )

        # --- graph-level static channels (inherited pipeline) -----------------
        self.global_encoder = zpp._MLPBlock(
            self.global_width, max(self.local_dim // 2, 32), 32, self.dropout
        )
        self.topology_encoder = nn.Sequential(
            nn.Linear(self.topology_width, int(topology_hidden)),
            nn.ReLU(),
            nn.Linear(int(topology_hidden), int(topology_out)),
        )

        self.unary_width = 2 * self.local_dim + 1
        self.pair_pool_width = zpp.DISTANCE_BUCKETS * (2 * self.pair_hidden + 1)
        self.global_out_width = 32
        self.topology_out_width = int(topology_out)
        self.graph_width = (
            self.unary_width
            + self.pair_pool_width
            + self.global_out_width
            + self.topology_out_width
        )
        self.head = GenericReader(self.graph_width, tuple(int(h) for h in head_hidden))

        # Evaluation-only interventions (default None == frozen forward path).
        self.pair_intervention: str | None = None

    # -- diagnostics hooks ---------------------------------------------------
    def set_pair_intervention(self, mode: str | None) -> "SRDAModel":
        if mode not in (None, "neutral_dict", "zero_eps", "mean_alpha"):
            raise ValueError(f"unknown pair intervention {mode!r}")
        self.pair_intervention = mode
        return self

    # -- forward -------------------------------------------------------------
    def encode(self, data: Any) -> torch.Tensor:
        """Return the unified graph representation ``R`` (input to the sole head).

        Ordering is the contract: the complete local state and the complete
        unary readout are computed before any pair/relation tensor is touched.
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
        z = self.local_encoder(x)
        unary = node_moments(z, data.batch, n_graphs)

        # --- 2. end-to-end dictionary assignment (pure function of z) --------
        local = self.dictionary(z)
        alpha = local["alpha"]
        if self.pair_intervention == "mean_alpha":
            # Evaluation-only: every patch receives its graph-mean assignment,
            # and the whole downstream algebra is recomputed from it.
            alpha = _graph_mean(alpha, data.batch, n_graphs)[data.batch]
            a = alpha @ self.dictionary.factor_a
            b = alpha @ self.dictionary.factor_b
            eps = F.normalize(self.dictionary.query(z), dim=-1, eps=1.0e-8) - (
                alpha @ self.dictionary.normalized_atoms()
            )
        else:
            a = local["a"]
            b = local["b"]
            eps = local["eps"]
        e = self.residual_proj(eps)

        # --- 3. static pair relation trunk (never writes back) ---------------
        source = data.pair_index[0].long()
        target = data.pair_index[1].long()
        rel_hidden = self.relation_trunk(data.pair_relation)
        rel_gate = 1.0 + torch.tanh(
            self.relation_gate(rel_hidden) + self.bucket_embedding(data.pair_bucket)
        )
        rel_feat = self.relation_feature(rel_hidden)

        # --- 4. prototype-pair tensor contraction -----------------------------
        proto_pair = 0.5 * (a[source] * b[target] + a[target] * b[source])
        p_dict = proto_pair * rel_gate
        if self.pair_intervention == "neutral_dict":
            p_dict = torch.zeros_like(p_dict)

        # --- 5. narrow within-prototype residual pair path -------------------
        e_left = e[source]
        e_right = e[target]
        p_eps = torch.cat(
            [e_left + e_right, torch.abs(e_left - e_right), e_left * e_right], dim=1
        )
        if self.pair_intervention == "zero_eps":
            p_eps = torch.zeros_like(p_eps)

        # --- 6. one single-pass occurrence-level pair kernel ------------------
        pair_input = torch.cat([p_dict, p_eps, rel_feat], dim=1)
        q_ij = self.pair_encoder(pair_input)
        pair_pool = bucket_moments(q_ij, data.batch[source], data.pair_bucket, n_graphs)

        # --- 7. graph-level invariant readout --------------------------------
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
        return torch.cat([unary, pair_pool, global_hidden, topology_hidden], dim=1)

    def forward(self, data: Any) -> torch.Tensor:
        return self.head(self.encode(data)).view(-1)


def build_srda(seed: int = 0, **kwargs: Any) -> SRDAModel:
    """Deterministic construction with full parameter re-initialization."""
    _seed_everything(int(seed))
    return SRDAModel(**kwargs)


build_model = build_srda


# ---------------------------------------------------------------------------
# runtime width / identity-channel audit
# ---------------------------------------------------------------------------


def _first_batch(data: Sequence[Any], device: torch.device, limit: int = 128):
    return sdp._first_batch(data, device, limit)


def prepare_encoded(force: bool = False) -> dict[str, Any]:
    return sdp.prepare_encoded(force=force)


def load_encoded(train_subset=None, valid_subset=None):
    return sdp.load_encoded(train_subset, valid_subset)


def runtime_width_audit(model: SRDAModel, batch: Any) -> dict[str, Any]:
    """Attest every runtime width from real tensors on a real batch."""
    captured: dict[str, torch.Tensor] = {}
    dict_out: dict[str, torch.Tensor] = {}

    def _cap_z(_m, _i, output):
        captured["z"] = output.detach()

    def _cap_dict(_m, _i, output):
        for key, value in output.items():
            dict_out[key] = value.detach()

    def _cap_pair(_m, inputs, _o):
        captured["pair_input"] = inputs[0].detach()

    def _cap_head_input(_m, inputs, _o):
        captured["graph"] = inputs[0].detach()

    handles = [
        model.local_encoder.register_forward_hook(_cap_z),
        model.dictionary.register_forward_hook(_cap_dict),
        model.pair_encoder.register_forward_hook(_cap_pair),
        model.head.register_forward_hook(_cap_head_input),
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
        "local_state_width": int(captured["z"].shape[1]),
        "dictionary_assignment_width": int(dict_out["alpha"].shape[1]),
        "prototype_factor_width": int(dict_out["a"].shape[1]),
        "residual_input_width": int(dict_out["eps"].shape[1]),
        "residual_width": int(model.residual_proj.out_features),
        "pair_input_width": int(captured["pair_input"].shape[1]),
        "declared_pair_input_width": int(model.pair_input_width),
        "relation_input_width": int(batch.pair_relation.shape[1]),
        "pair_hidden": int(model.pair_hidden),
        "unary_width": int(model.unary_width),
        "pair_pool_width": int(model.pair_pool_width),
        "global_width": int(model.global_out_width),
        "topology_width": int(model.topology_out_width),
        "graph_width": int(captured["graph"].shape[1]),
        "declared_graph_width": int(model.graph_width),
        "graph_head_hidden": list(GRAPH_HEAD_HIDDEN),
        "identity_channels": identity_channel_audit(model),
        "official_test_loaded": False,
    }
    payload["widths_consistent"] = bool(
        payload["local_input_width"] == payload["declared_local_input_width"] == LOCAL_INPUT_WIDTH
        and payload["local_state_width"] == LOCAL_DIM
        and payload["dictionary_assignment_width"] == DICT_ATOMS
        and payload["prototype_factor_width"] == TENSOR_RANK
        and payload["residual_width"] == RESIDUAL_DIM
        and payload["pair_input_width"] == payload["declared_pair_input_width"] == PAIR_INPUT_WIDTH
        and payload["graph_width"] == payload["declared_graph_width"]
        == payload["unary_width"] + payload["pair_pool_width"]
        + payload["global_width"] + payload["topology_width"]
    )
    return payload


def identity_channel_audit(model: SRDAModel) -> dict[str, Any]:
    """Hard evidence that no typed/parent identity lookup exists anywhere."""
    state_keys = list(model.state_dict().keys())
    embeddings = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding)
    ]
    return {
        "state_dict_keys": state_keys,
        "forbidden_substrings": [
            "typed_embedding",
            "parent_embedding",
            "typed_token",
            "parent_token",
        ],
        "forbidden_key_hits": [
            key
            for key in state_keys
            for needle in (
                "typed_embedding",
                "parent_embedding",
                "typed_token",
                "parent_token",
            )
            if needle in key
        ],
        "embedding_modules": [
            {"name": name, "rows": int(module.num_embeddings), "dim": int(module.embedding_dim)}
            for name, module in embeddings
        ],
        "only_bucket_embedding": bool(
            len(embeddings) == 1
            and embeddings[0][0] == "bucket_embedding"
            and int(embeddings[0][1].num_embeddings) == zpp.DISTANCE_BUCKETS
        ),
        "identity_channel_params": int(
            sum(
                module.weight.numel()
                for name, module in embeddings
                if name != "bucket_embedding"
            )
        ),
        "uses_typed_token": False,
        "uses_parent_token": False,
    }


def parameter_audit() -> dict[str, Any]:
    model = build_srda(0)
    total = _n_params(model)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "center_context": bool(model.center_context),
        "center_update_is_none": bool(model.center_update is None),
        "declared_spec": {
            "local": [LOCAL_INPUT_WIDTH, LOCAL_HIDDEN, LOCAL_DIM],
            "dictionary": {
                "atoms": DICT_ATOMS,
                "rank": DICT_RANK,
                "tau_init": DICT_TAU_INIT,
            },
            "tensor_rank": TENSOR_RANK,
            "residual_dim": RESIDUAL_DIM,
            "relation_trunk": [zpp.RELATION_WIDTH, REL_TRUNK_HIDDEN],
            "relation_feat_dim": REL_FEAT_DIM,
            "pair_input_width": PAIR_INPUT_WIDTH,
            "pair_encoder": [PAIR_INPUT_WIDTH, PAIR_ENCODER_HIDDEN, PAIR_HIDDEN],
            "graph_head_hidden": list(GRAPH_HEAD_HIDDEN),
            "dropout": DROPOUT,
        },
        "structural_widths": {
            "unary_width": int(model.unary_width),
            "pair_pool_width": int(model.pair_pool_width),
            "global_out_width": int(model.global_out_width),
            "topology_out_width": int(model.topology_out_width),
            "graph_width": int(model.graph_width),
        },
        "params": {
            "total": total,
            "local_encoder": _n_params(model.local_encoder),
            "dictionary": _n_params(model.dictionary),
            "residual_proj": _n_params(model.residual_proj),
            "relation_trunk": _n_params(model.relation_trunk),
            "relation_gate": _n_params(model.relation_gate),
            "bucket_embedding": _n_params(model.bucket_embedding),
            "relation_feature": _n_params(model.relation_feature),
            "pair_encoder": _n_params(model.pair_encoder),
            "global_encoder": _n_params(model.global_encoder),
            "topology_encoder": _n_params(model.topology_encoder),
            "head": _n_params(model.head),
        },
        "identity_channel_audit": identity_channel_audit(model),
        "parameter_budget": {
            "soft_target": 100000,
            "within_soft_target": bool(total < 100000),
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# strict-static integrity gates
# ---------------------------------------------------------------------------


def _contract_violation(*_args: Any, **_kwargs: Any) -> torch.Tensor:
    raise RuntimeError(
        "strict-static contract violated: _pool_pairs_to_centres was called"
    )


def _trace_forward(model: SRDAModel, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    captured: dict[str, Any] = {}
    counters = {"pair_encoder": 0, "relation_trunk": 0, "dictionary": 0}

    def _cap_z(_m, _i, output):
        captured["z"] = output.detach().clone()

    def _cap_q(_m, _i, output):
        captured["q_raw"] = output.detach().clone()

    def _cap_dict(_m, _i, output):
        captured["alpha"] = output["alpha"].detach().clone()
        captured["a"] = output["a"].detach().clone()
        captured["b"] = output["b"].detach().clone()
        captured["eps"] = output["eps"].detach().clone()
        counters["dictionary"] += 1

    def _cap_e(_m, _i, output):
        captured["e"] = output.detach().clone()

    def _cap_pair_input(_m, inputs, _o):
        captured["pair_input"] = inputs[0].detach().clone()
        counters["pair_encoder"] += 1

    def _cap_rel(_m, _i, _o):
        counters["relation_trunk"] += 1

    handles = [
        model.local_encoder.register_forward_hook(_cap_z),
        model.dictionary.query.register_forward_hook(_cap_q),
        model.dictionary.register_forward_hook(_cap_dict),
        model.residual_proj.register_forward_hook(_cap_e),
        model.pair_encoder.register_forward_hook(_cap_pair_input),
        model.relation_trunk.register_forward_hook(_cap_rel),
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
    return prediction, captured


def static_contract_checks(model: SRDAModel, batch: Any) -> dict[str, Any]:
    """Programmatic hard checks of the strict-static contract on a real batch."""
    results: dict[str, Any] = {}
    model.eval()

    results["center_context_false"] = bool(model.center_context is False)
    results["center_update_is_none"] = bool(model.center_update is None)
    if not results["center_context_false"] or not results["center_update_is_none"]:
        raise RuntimeError("strict-static contract failed: centre update present")

    # (A) monkeypatch the historical pair->centre aggregation to raise; the
    # SRDA forward must still succeed (and must never touch that code path).
    original = zpp.PatchPathModel._pool_pairs_to_centres
    zpp.PatchPathModel._pool_pairs_to_centres = _contract_violation
    try:
        with torch.no_grad():
            _ = model(batch)
        results["forward_without_pair_to_center"] = True
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original

    # (B) one single pass: dictionary assignment / relation trunk / pair kernel.
    prediction, trace = _trace_forward(model, batch)
    results["pair_encoder_calls_per_forward"] = int(trace["counters"]["pair_encoder"])
    results["relation_trunk_calls_per_forward"] = int(trace["counters"]["relation_trunk"])
    results["dictionary_calls_per_forward"] = int(trace["counters"]["dictionary"])
    if trace["counters"]["pair_encoder"] != 1:
        raise RuntimeError("pair kernel was evaluated more than once per forward")
    if trace["counters"]["relation_trunk"] != 1:
        raise RuntimeError("relation trunk was evaluated more than once per forward")
    if trace["counters"]["dictionary"] != 1:
        raise RuntimeError("dictionary assignment was evaluated more than once per forward")

    # (C) local independence: mutate the static pair relation; z / q / alpha /
    # eps / e must be bit-identical while the prediction does change.
    mutated = batch.clone()
    if int(mutated.pair_relation.shape[0]) > 0:
        generator = torch.Generator(device="cpu").manual_seed(20260923)
        mutated.pair_relation = torch.randn(
            mutated.pair_relation.shape, generator=generator
        ).to(mutated.pair_relation.device)
    _, trace_mut = _trace_forward(model, mutated)
    results["z_identical_under_relation_mutation"] = bool(
        torch.equal(trace["z"], trace_mut["z"])
    )
    results["q_identical_under_relation_mutation"] = bool(
        torch.equal(trace["q_raw"], trace_mut["q_raw"])
    )
    results["alpha_identical_under_relation_mutation"] = bool(
        torch.equal(trace["alpha"], trace_mut["alpha"])
    )
    results["epsilon_identical_under_relation_mutation"] = bool(
        torch.equal(trace["eps"], trace_mut["eps"])
    )
    results["max_abs_z_diff"] = float((trace["z"] - trace_mut["z"]).abs().max().item())
    results["max_abs_alpha_diff"] = float(
        (trace["alpha"] - trace_mut["alpha"]).abs().max().item()
    )
    results["max_abs_epsilon_diff"] = float(
        (trace["eps"] - trace_mut["eps"]).abs().max().item()
    )
    prediction_again, _trace_again = _trace_forward(model, batch)
    with torch.no_grad():
        mutated_prediction = model(mutated)
    results["prediction_mutation_max_abs_diff"] = float(
        (prediction_again - mutated_prediction).abs().max().item()
    )
    results["prediction_changes_under_relation_mutation"] = bool(
        results["prediction_mutation_max_abs_diff"] > 1.0e-8
    )
    if not results["z_identical_under_relation_mutation"]:
        raise RuntimeError("strict-static contract violated: relation feeds back into z")
    if not results["alpha_identical_under_relation_mutation"]:
        raise RuntimeError("dictionary assignment changed under pair mutation")
    if not results["prediction_changes_under_relation_mutation"]:
        raise RuntimeError("relation mutation test is vacuous (prediction unchanged)")

    # (D) no raw endpoint bypass: perturb z inside ker(Wq).  z and the unary
    # readout must change materially, while the pair-kernel input stays at
    # floating-point noise scale (it can only see alpha / eps / rel).
    query_weight = model.dictionary.query.weight.detach()  # (rank, local_dim)
    basis, _ = torch.linalg.qr(query_weight.t())  # (local_dim, rank)
    generator = torch.Generator(device="cpu").manual_seed(4242)
    vector = torch.randn(
        int(query_weight.shape[1]), generator=generator, dtype=query_weight.dtype
    ).to(query_weight.device)
    delta = vector - basis @ (basis.t() @ vector)
    delta = delta / delta.norm().clamp_min(1.0e-12)
    assert float((query_weight @ delta).abs().max().item()) < 1.0e-4

    perturbed_prediction, trace_perturbed = _trace_forward_perturbed(model, batch, delta)
    with torch.no_grad():
        z_clean = trace["z"]
        z_delta = z_clean - trace_perturbed["z"]
        z_shift = float(z_delta.abs().max().item())
        z_relative_shift = float(z_delta.norm() / z_clean.norm().clamp_min(1.0e-12))
        unary_clean = node_moments(z_clean, batch.batch, int(batch.global_context.shape[0]))
        unary_perturbed = node_moments(
            trace_perturbed["z"], batch.batch, int(batch.global_context.shape[0])
        )
        unary_shift = float((unary_clean - unary_perturbed).abs().max().item())
        unary_relative_shift = float(
            (unary_clean - unary_perturbed).norm() / unary_clean.norm().clamp_min(1.0e-12)
        )
    pair_input_shift = float(
        (trace["pair_input"] - trace_perturbed["pair_input"]).abs().max().item()
    )
    pair_input_scale = float(trace["pair_input"].abs().max().item())
    results["no_raw_pair_bypass_z_shift"] = z_shift
    results["no_raw_pair_bypass_z_relative_shift"] = z_relative_shift
    results["no_raw_pair_bypass_unary_shift"] = unary_shift
    results["no_raw_pair_bypass_unary_relative_shift"] = unary_relative_shift
    results["no_raw_pair_bypass_pair_input_shift"] = pair_input_shift
    results["no_raw_pair_bypass_pair_input_relative_shift"] = float(
        pair_input_shift / max(pair_input_scale, 1.0e-12)
    )
    results["no_raw_pair_bypass_prediction_shift"] = float(
        (prediction_again - perturbed_prediction).abs().max().item()
    )
    # A raw ``z_i`` bypass would move the pair input by an O(1) relative shift;
    # a function of (alpha, eps, rel) only can move it by floating-point noise
    # (measured 2e-7 on a real batch, i.e. three orders below the gate).
    results["no_raw_pair_bypass"] = bool(
        z_relative_shift > 0.05
        and unary_relative_shift > 1.0e-2
        and results["no_raw_pair_bypass_pair_input_relative_shift"] < 1.0e-4
    )
    if not results["no_raw_pair_bypass"]:
        raise RuntimeError(
            "pair kernel appears to consume the raw local state: "
            f"{results['no_raw_pair_bypass_pair_input_shift']} vs z shift {z_shift}"
        )

    # (E) exactly one pair kernel and one relation trunk pass, and the pair
    # input is *exactly* the declared 72D block algebra.
    results["pair_input_width"] = int(trace["pair_input"].shape[1])
    results["pair_input_width_ok"] = bool(
        results["pair_input_width"] == PAIR_INPUT_WIDTH
    )
    if not results["pair_input_width_ok"]:
        raise RuntimeError("pair input width drifted")

    # (F) no identity channel: typed / parent tokens have zero effect.  The
    # indices are poisoned out of range, so any embedding lookup on them would
    # raise immediately; the prediction must also stay at execution-noise
    # level (CUDA ``index_add_`` scatter is not bit-deterministic).
    repeat_prediction, _repeat_trace = _trace_forward(model, batch)
    identity = batch.clone()
    identity.typed_token = torch.full_like(batch.typed_token, 10_000_000)
    identity.parent_token = torch.full_like(batch.parent_token, 10_000_000)
    with torch.no_grad():
        identity_prediction = model(identity)
    identity_shift = float(
        (prediction_again - identity_prediction).abs().max().item()
    )
    repeat_shift = float((prediction_again - repeat_prediction).abs().max().item())
    results["prediction_shift_under_identity_mutation"] = identity_shift
    results["prediction_shift_repeat_forward_baseline"] = repeat_shift
    results["identity_index_poison_no_lookup"] = True
    results["identity_mutation_zero_effect"] = bool(
        identity_shift <= max(1.0e-5, 0.01 * results["prediction_mutation_max_abs_diff"])
    )
    if not results["identity_mutation_zero_effect"]:
        raise RuntimeError(
            "typed/parent identity tokens influence the prediction: "
            f"{identity_shift} vs relation shift "
            f"{results['prediction_mutation_max_abs_diff']}"
        )

    # (G) invariances: pair order, endpoint swap, patch relabel.
    if int(batch.pair_index.shape[1]) > 1:
        permutation = torch.randperm(
            int(batch.pair_index.shape[1]), generator=torch.Generator().manual_seed(7)
        ).to(batch.pair_index.device)
        permuted = batch.clone()
        permuted.pair_index = batch.pair_index[:, permutation]
        permuted.pair_relation = batch.pair_relation[permutation]
        permuted.pair_bucket = batch.pair_bucket[permutation]
        with torch.no_grad():
            permuted_prediction = model(permuted)
        results["pair_order_max_abs_pred_diff"] = float(
            (prediction_again - permuted_prediction).abs().max().item()
        )
        swapped = batch.clone()
        swapped.pair_index = batch.pair_index.flip(0)
        with torch.no_grad():
            swapped_prediction = model(swapped)
        results["endpoint_swap_max_abs_pred_diff"] = float(
            (prediction_again - swapped_prediction).abs().max().item()
        )
        results["pair_order_invariant"] = bool(
            results["pair_order_max_abs_pred_diff"] < 1.0e-4
        )
        results["endpoint_swap_symmetric"] = bool(
            results["endpoint_swap_max_abs_pred_diff"] < 1.0e-4
        )
    else:
        results["pair_order_max_abs_pred_diff"] = 0.0
        results["endpoint_swap_max_abs_pred_diff"] = 0.0
        results["pair_order_invariant"] = True
        results["endpoint_swap_symmetric"] = True

    # Patch relabeling must be *within* each graph, so graph membership
    # (``data.batch``) stays valid; the graph-level readout is permutation
    # invariant by construction and this checks it on real pair data.
    generator = torch.Generator().manual_seed(13)
    patch_order = torch.arange(int(batch.patch_cont.shape[0]))
    for graph_id in range(int(batch.global_context.shape[0])):
        members = (batch.batch == graph_id).nonzero(as_tuple=True)[0]
        if int(members.numel()) > 1:
            shuffled = members[
                torch.randperm(int(members.numel()), generator=generator)
            ]
            patch_order[members] = shuffled
    inverse = torch.empty_like(patch_order)
    inverse[patch_order] = torch.arange(patch_order.shape[0])
    relabeled = batch.clone()
    relabeled.patch_cont = batch.patch_cont[patch_order]
    relabeled.patch_context = batch.patch_context[patch_order]
    relabeled.typed_token = batch.typed_token[patch_order]
    relabeled.parent_token = batch.parent_token[patch_order]
    relabeled.structural_token = batch.structural_token[patch_order]
    relabeled.pair_index = inverse[batch.pair_index]
    with torch.no_grad():
        relabeled_prediction = model(relabeled)
    results["patch_relabel_max_abs_pred_diff"] = float(
        (prediction_again - relabeled_prediction).abs().max().item()
    )
    results["patch_order_invariant"] = bool(
        results["patch_relabel_max_abs_pred_diff"] < 1.0e-4
    )

    results["identity_channel_audit"] = identity_channel_audit(model)
    results["passed"] = bool(
        results["forward_without_pair_to_center"]
        and results["pair_encoder_calls_per_forward"] == 1
        and results["relation_trunk_calls_per_forward"] == 1
        and results["dictionary_calls_per_forward"] == 1
        and results["z_identical_under_relation_mutation"]
        and results["alpha_identical_under_relation_mutation"]
        and results["epsilon_identical_under_relation_mutation"]
        and results["prediction_changes_under_relation_mutation"]
        and results["no_raw_pair_bypass"]
        and results["identity_mutation_zero_effect"]
        and results["pair_order_invariant"]
        and results["endpoint_swap_symmetric"]
        and results["patch_order_invariant"]
        and results["identity_channel_audit"]["only_bucket_embedding"]
    )
    return results


def _trace_forward_perturbed(
    model: SRDAModel, batch: Any, delta: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Trace a forward where the local state is perturbed by ``delta``."""

    def _perturb(_m, _i, output):
        return output + delta

    handle = model.local_encoder.register_forward_hook(_perturb)
    try:
        return _trace_forward(model, batch)
    finally:
        handle.remove()


def gradient_viability_checks(model: SRDAModel, batch: Any) -> dict[str, Any]:
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
    payload["wq_grad_norm"] = grad_norm(list(model.dictionary.query.parameters()))
    payload["dict_atoms_grad_norm"] = grad_norm([model.dictionary.atoms])
    payload["factor_a_grad_norm"] = grad_norm([model.dictionary.factor_a])
    payload["factor_b_grad_norm"] = grad_norm([model.dictionary.factor_b])
    payload["tau_grad_norm"] = grad_norm([model.dictionary.tau_logit])
    payload["residual_proj_grad_norm"] = grad_norm(list(model.residual_proj.parameters()))
    payload["relation_trunk_grad_norm"] = grad_norm(list(model.relation_trunk.parameters()))
    payload["relation_gate_grad_norm"] = grad_norm(list(model.relation_gate.parameters()))
    payload["bucket_embedding_grad_norm"] = grad_norm(list(model.bucket_embedding.parameters()))
    payload["relation_feature_grad_norm"] = grad_norm(
        list(model.relation_feature.parameters())
    )
    payload["pair_encoder_grad_norm"] = grad_norm(list(model.pair_encoder.parameters()))
    payload["head_grad_norm"] = grad_norm(list(model.head.parameters()))
    model.eval()
    model.zero_grad(set_to_none=True)

    required_positive = (
        "local_encoder_grad_norm",
        "wq_grad_norm",
        "dict_atoms_grad_norm",
        "factor_a_grad_norm",
        "factor_b_grad_norm",
        "residual_proj_grad_norm",
        "relation_trunk_grad_norm",
        "pair_encoder_grad_norm",
    )
    payload["all_finite"] = bool(
        all(
            math.isfinite(float(value))
            for key, value in payload.items()
            if key not in {"loss", "required_positive_gradients", "required_positive_ok"}
        )
    )
    payload["required_positive_gradients"] = list(required_positive)
    payload["required_positive_ok"] = bool(
        all(float(payload[key]) > 0.0 for key in required_positive)
    )
    payload["branch_receives_gradient"] = bool(
        payload["all_finite"] and payload["required_positive_ok"]
    )
    return payload


def integrity_stage() -> dict[str, Any]:
    _train_data, valid_data, _audit = load_encoded(valid_subset=64)
    batch = _first_batch(valid_data, torch.device("cpu"), limit=64)
    model = build_srda(0)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_audit": parameter_audit(),
        "runtime_widths": runtime_width_audit(model, batch),
        "contract": static_contract_checks(model, batch),
        "gradient_viability": gradient_viability_checks(model, batch),
        "pair_block_scale_at_init": pair_block_scale(model, batch),
        "official_test_loaded": False,
    }
    payload["passed"] = bool(
        payload["runtime_widths"]["widths_consistent"]
        and payload["runtime_widths"]["identity_channels"]["only_bucket_embedding"]
        and payload["contract"]["passed"]
        and payload["gradient_viability"]["branch_receives_gradient"]
    )
    _write_json(RESULTS_DIR / "integrity_gates.json", payload)
    return payload


# ---------------------------------------------------------------------------
# cheap block-scale / dictionary diagnostics (inference only)
# ---------------------------------------------------------------------------


def _trace_pairs(model: SRDAModel, batch: Any) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def _cap(_m, inputs, _o):
        captured["pair_input"] = inputs[0].detach()

    handle = model.pair_encoder.register_forward_hook(_cap)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _ = model(batch)
    finally:
        handle.remove()
        model.train(was_training)
    return captured


def pair_block_scale(model: SRDAModel, batch: Any) -> dict[str, Any]:
    captured = _trace_pairs(model, batch)
    pair_input = captured["pair_input"]
    p_dict = pair_input[:, :TENSOR_RANK]
    p_eps = pair_input[:, TENSOR_RANK : TENSOR_RANK + 3 * RESIDUAL_DIM]
    rel_feat = pair_input[:, TENSOR_RANK + 3 * RESIDUAL_DIM :]
    weight = model.pair_encoder[0].weight.detach()
    return {
        "mean_norm_p_dict": float(p_dict.norm(dim=1).mean()),
        "mean_norm_p_eps": float(p_eps.norm(dim=1).mean()),
        "mean_norm_rel_feat": float(rel_feat.norm(dim=1).mean()),
        "pair_encoder_first_layer_block_weight_norms": {
            "p_dict": float(weight[:, :TENSOR_RANK].norm()),
            "p_eps": float(weight[:, TENSOR_RANK : TENSOR_RANK + 3 * RESIDUAL_DIM].norm()),
            "rel_feat": float(weight[:, TENSOR_RANK + 3 * RESIDUAL_DIM :].norm()),
        },
        "n_pairs": int(pair_input.shape[0]),
    }


def dictionary_diagnostics(model: SRDAModel, loader, device: torch.device) -> dict[str, Any]:
    chunks: list[torch.Tensor] = []
    pair_chunks: list[dict[str, torch.Tensor]] = []

    def _cap_z(_m, _i, output):
        chunks.append(output.detach())

    def _cap_pair(_m, inputs, _o):
        pair_chunks.append({"pair_input": inputs[0].detach()})

    handle_z = model.local_encoder.register_forward_hook(_cap_z)
    handle_pair = model.pair_encoder.register_forward_hook(_cap_pair)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                model(batch)
    finally:
        handle_z.remove()
        handle_pair.remove()

    z = torch.cat(chunks, dim=0)
    alpha, tau, atoms = model.dictionary.alpha_from_z(z)
    eps = 1.0e-12
    entropy = -(alpha * (alpha + eps).log()).sum(dim=1)
    mean_mass = alpha.mean(dim=0)
    cosine = atoms @ atoms.t()
    off_diagonal = cosine - torch.eye(
        cosine.shape[0], device=cosine.device, dtype=cosine.dtype
    )
    top8 = torch.topk(alpha, k=min(8, alpha.shape[1]), dim=1).values.sum(dim=1)
    q_recon = alpha @ atoms
    q = F.normalize(model.dictionary.query(z), dim=-1, eps=1.0e-8)
    residual = q - q_recon

    pair_input = torch.cat([row["pair_input"] for row in pair_chunks], dim=0)
    p_dict = pair_input[:, :TENSOR_RANK]
    p_eps = pair_input[:, TENSOR_RANK : TENSOR_RANK + 3 * RESIDUAL_DIM]
    rel_feat = pair_input[:, TENSOR_RANK + 3 * RESIDUAL_DIM :]
    weight = model.pair_encoder[0].weight.detach()

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
        "tau_final": float(tau),
        "mean_residual_norm": float(residual.norm(dim=1).mean()),
        "mean_e_norm": float(model.residual_proj(residual).norm(dim=1).mean()),
        "mean_norm_p_dict": float(p_dict.norm(dim=1).mean()),
        "mean_norm_p_eps": float(p_eps.norm(dim=1).mean()),
        "mean_norm_rel_feat": float(rel_feat.norm(dim=1).mean()),
        "pair_encoder_first_layer_block_weight_norms": {
            "p_dict": float(weight[:, :TENSOR_RANK].norm()),
            "p_eps": float(weight[:, TENSOR_RANK : TENSOR_RANK + 3 * RESIDUAL_DIM].norm()),
            "rel_feat": float(weight[:, TENSOR_RANK + 3 * RESIDUAL_DIM :].norm()),
        },
        "n_patches": int(z.shape[0]),
        "n_pairs": int(pair_input.shape[0]),
    }


def intervention_shift(
    model: SRDAModel,
    loader,
    device: torch.device,
    predictions_normal: np.ndarray,
) -> dict[str, Any]:
    """Inference-only interventions: dict / residual / assignment ablations."""
    payload: dict[str, Any] = {"applicable": True}
    normal = predictions_normal.astype(np.float64)
    for mode in ("neutral_dict", "zero_eps", "mean_alpha"):
        model.set_pair_intervention(mode)
        try:
            mae, _targets, predictions = sdp._evaluate_mae(model, loader, device)
        finally:
            model.set_pair_intervention(None)
        shift = np.abs(normal - predictions.astype(np.float64))
        payload[mode] = {
            "valid_mae_after_intervention": float(mae),
            "mean_abs_prediction_shift": float(shift.mean()),
            "max_abs_prediction_shift": float(shift.max()),
            "frac_shift_gt_1e-6": float((shift > 1.0e-6).mean()),
        }
    return payload


# ---------------------------------------------------------------------------
# training / soup (faithful mirror of the inherited optimized protocol)
# ---------------------------------------------------------------------------


def train_arm(
    *,
    seed: int = 0,
    device: str = "cpu",
    max_epochs: int = 240,
    train_subset: int | None = None,
    valid_subset: int | None = None,
    tag: str | None = None,
    save_state: bool = True,
    soup: bool = True,
) -> dict[str, Any]:
    device_obj = torch.device(str(device))
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    train_data, valid_data, audit = load_encoded(train_subset, valid_subset)
    if int(audit["typed_vocabulary_size_with_oov"]) != sdp.TYPED_VOCAB_SIZE:
        raise RuntimeError("typed vocabulary drift")
    if int(audit["parent_vocabulary_size_with_oov"]) != sdp.PARENT_VOCAB_SIZE:
        raise RuntimeError("parent vocabulary drift")

    protocol = dict(OPTIMIZED_PROTOCOL)
    protocol["max_epochs"] = int(max_epochs)
    protocol["early_termination"] = "none"

    model = build_srda(seed=seed).to(device_obj)
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
        raise RuntimeError(f"SRDA branch receives no step-0 gradient: {viability}")

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
                f"[{tag or 'srda'} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
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
            key: torch.stack(
                [epoch_states[epoch][key].float() for epoch in members]
            ).mean(0)
            for key in keys
        }
        soup_model = build_srda(seed=seed)
        soup_model.load_state_dict(soup_state)
        soup_model.to(device_obj)
        soup_mae, _, soup_predictions = sdp._evaluate_mae(
            soup_model, eval_loader, device_obj
        )
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
    intervention = intervention_shift(model, eval_loader, device_obj, best_predictions)

    curve_path = CURVE_DIR / f"{tag or 'srda'}_seed{seed}_curve.csv"
    _write_csv(curve_path, curve)
    state_paths: dict[str, str] = {}
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        best_path = STATE_DIR / f"{tag or 'srda'}_seed{seed}_best_state.pt"
        torch.save(model.state_dict(), best_path)
        state_paths["best"] = str(best_path)
        if soup_state_cpu is not None:
            soup_path = STATE_DIR / f"{tag or 'srda'}_seed{seed}_soup_state.pt"
            torch.save(soup_state_cpu, soup_path)
            state_paths["soup"] = str(soup_path)

    peak_memory_mb = None
    if device_obj.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))

    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "seed": int(seed),
        "tag": str(tag or "srda"),
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
        "valid_predictions": best_predictions.tolist(),
        "valid_targets": valid_targets.tolist(),
        "curve_path": str(curve_path),
        "state_paths": state_paths,
        "official_test_loaded": False,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(str(device_obj)),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_DIR / f"{tag or 'srda'}_seed{seed}.json", summary)
    print(
        f"[{tag or 'srda'} seed{seed}] best_valid={best_mae:.6f} epoch={best_epoch} "
        f"soup={soup_payload.get('soup_valid_mae')} params={total_params} "
        f"wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# baseline guard: inherited shared pipeline untouched
# ---------------------------------------------------------------------------


def baseline_guard() -> dict[str, Any]:
    source_files = {
        "zinc_patch_path_pooling.py": TRACK_ROOT
        / "experiments/luyin16/zinc_patch_path_pooling.py",
        "zinc_static_dictionary_pair.py": TRACK_ROOT
        / "experiments/luyin16/zinc_static_dictionary_pair.py",
        "zinc_compact_v4_topology_hinge.yaml": TRACK_ROOT
        / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml",
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_files.items()
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "inherited_pipeline_modified_by_this_round": False,
        "source_sha256": hashes,
        "s0_reference": {"best": S0_SEED0_BEST, "soup": S0_SEED0_SOUP},
        "sdpl_seed0_reference": {"best": SDPK_SEED0_BEST, "soup": SDPK_SEED0_SOUP},
        "note": (
            "SRDA-v0 shares the encoded static-descriptor cache, the "
            "relation preprocessing, the global-context channel, the topology "
            "hinge and the training/checkpoint/soup utilities with the "
            "strict-static S0 pipeline; it re-implements the model so the "
            "no-message-passing contract is structural."
        ),
        "official_test_loaded": False,
    }
    s0_ckpt = sdp.STATE_DIR / "s0_seed0_selection_state.pt"
    if s0_ckpt.exists():
        train_data, valid_data, _audit = load_encoded(valid_subset=None)
        device = torch.device("cpu")
        model_s0 = sdp.build_s0(0)
        model_s0.load_state_dict(
            torch.load(s0_ckpt, map_location="cpu", weights_only=False)
        )
        model_s0.to(device)
        eval_loader = zpp._make_loader(valid_data, 128, False, 91012)
        mae, _t, _p = sdp._evaluate_mae(model_s0, eval_loader, device)
        payload["checkpoint_replay"] = {
            "attempted": True,
            "checkpoint": str(s0_ckpt),
            "replayed_valid_mae": float(mae),
            "recorded_valid_mae": float(S0_SEED0_BEST),
            "abs_diff": float(abs(mae - S0_SEED0_BEST)),
            "reproduced": bool(abs(mae - S0_SEED0_BEST) < 1.0e-6),
        }
    _write_json(RESULTS_DIR / "baseline_guard.json", payload)
    return payload


def smoke_stage(device: str = "cuda") -> dict[str, Any]:
    summary = train_arm(
        seed=0,
        device=device,
        max_epochs=2,
        train_subset=256,
        valid_subset=128,
        tag="smoke",
        save_state=False,
        soup=False,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "best_valid_mae": summary["best_valid_mae"],
        "parameters": summary["parameters"],
        "runtime_widths": summary["runtime_widths"],
        "contract_passed": summary["strict_static_contract"]["passed"],
        "gradient_viability": summary["gradient_viability"],
        "dictionary_diagnostics": summary["dictionary_diagnostics"],
        "intervention": summary["intervention"],
        "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
        "device": summary["device"],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


# ---------------------------------------------------------------------------
# analysis / gates
# ---------------------------------------------------------------------------


def classify_seed0(best: float, soup: float) -> dict[str, Any]:
    go = bool(soup <= GATE_SEED0_SOUP and best <= GATE_SEED0_BEST)
    return {
        "reference_s0_best": float(S0_SEED0_BEST),
        "reference_s0_soup": float(S0_SEED0_SOUP),
        "reference_sdpl_best": float(SDPK_SEED0_BEST),
        "reference_sdpl_soup": float(SDPK_SEED0_SOUP),
        "srda_best": float(best),
        "srda_soup": float(soup),
        "best_improvement_vs_s0": float(S0_SEED0_BEST) - float(best),
        "soup_improvement_vs_s0": float(S0_SEED0_SOUP) - float(soup),
        "best_improvement_vs_sdpl": float(SDPK_SEED0_BEST) - float(best),
        "soup_improvement_vs_sdpl": float(SDPK_SEED0_SOUP) - float(soup),
        "gate_soup_abs": GATE_SEED0_SOUP,
        "gate_best_abs": GATE_SEED0_BEST,
        "go": go,
        "verdict": "GO_SEED1" if go else "SRDA_V0_NO_STRONG_PERFORMANCE_SIGNAL",
    }


def classify_seed1(best: float, soup: float) -> dict[str, Any]:
    soup_delta = float(S0_SEED1_SOUP) - float(soup)
    best_delta = float(S0_SEED1_BEST) - float(best)
    strong = bool(soup <= GATE_SEED1_SOUP)
    directional = bool(
        soup_delta >= GATE_SEED1_SOUP_DELTA and best_delta >= GATE_SEED1_SOUP_DELTA
    )
    if strong:
        verdict = "SEED1_STRONG_CONFIRMATION"
    elif directional:
        verdict = "SEED1_DIRECTIONAL_CONFIRMATION"
    else:
        verdict = "SEED1_NO_CONFIRMATION"
    return {
        "reference_s0_best": float(S0_SEED1_BEST),
        "reference_s0_soup": float(S0_SEED1_SOUP),
        "srda_best": float(best),
        "srda_soup": float(soup),
        "best_improvement_vs_s0": best_delta,
        "soup_improvement_vs_s0": soup_delta,
        "gate_soup_abs": GATE_SEED1_SOUP,
        "gate_soup_delta": GATE_SEED1_SOUP_DELTA,
        "strong": strong,
        "directional": directional,
        "verdict": verdict,
    }


def analyze_stage() -> dict[str, Any]:
    seed0_path = RUNS_DIR / "srda_seed0.json"
    if not seed0_path.exists():
        raise FileNotFoundError(f"missing seed0 run output: {seed0_path}")
    seed0 = _read_json(seed0_path)
    soup0 = seed0["soup"].get("soup_valid_mae")
    seed0_gate = classify_seed0(seed0["best_valid_mae"], soup0)

    seed1 = None
    seed1_gate = None
    seed1_path = RUNS_DIR / "srda_seed1.json"
    if seed1_path.exists():
        seed1 = _read_json(seed1_path)
        seed1_gate = classify_seed1(
            seed1["best_valid_mae"], seed1["soup"].get("soup_valid_mae")
        )

    runs_used = 1 + (1 if seed1 is not None else 0)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "primary_metric": "fixed Top-5 weight soup official-valid MAE",
        "secondary_metric": "best-checkpoint official-valid MAE",
        "seed0_gate": seed0_gate,
        "seed1_gate": seed1_gate,
        "seed0": {
            "best_valid_mae": seed0["best_valid_mae"],
            "best_epoch": seed0["best_epoch"],
            "epochs_run": seed0["epochs_run"],
            "soup_valid_mae": soup0,
            "soup_members": seed0["soup"].get("members"),
            "parameters": seed0["parameters"],
            "wall_clock_s": seed0["wall_clock_s"],
            "peak_gpu_memory_mb": seed0.get("peak_gpu_memory_mb"),
            "device": seed0["device"],
            "git_commit": seed0["git_commit"],
            "contract_passed": seed0["strict_static_contract"]["passed"],
            "runtime_widths": seed0["runtime_widths"],
            "dictionary_diagnostics": seed0["dictionary_diagnostics"],
            "intervention": seed0["intervention"],
        },
        "seed1": None
        if seed1 is None
        else {
            "best_valid_mae": seed1["best_valid_mae"],
            "best_epoch": seed1["best_epoch"],
            "epochs_run": seed1["epochs_run"],
            "soup_valid_mae": seed1["soup"].get("soup_valid_mae"),
            "soup_members": seed1["soup"].get("members"),
            "parameters": seed1["parameters"],
            "wall_clock_s": seed1["wall_clock_s"],
            "peak_gpu_memory_mb": seed1.get("peak_gpu_memory_mb"),
            "device": seed1["device"],
            "git_commit": seed1["git_commit"],
            "contract_passed": seed1["strict_static_contract"]["passed"],
            "runtime_widths": seed1["runtime_widths"],
            "dictionary_diagnostics": seed1["dictionary_diagnostics"],
            "intervention": seed1["intervention"],
        },
        "budget": {
            "max_full_runs": MAX_FULL_RUNS,
            "full_runs_used": runs_used,
            "seed1_purchased": bool(seed1 is not None),
            "matched_control": False,
            "hpo": False,
            "sweep": False,
        },
        "parameter_audit": parameter_audit(),
        "baseline_guard": _read_json(RESULTS_DIR / "baseline_guard.json")
        if (RESULTS_DIR / "baseline_guard.json").exists()
        else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "analysis.json", payload)
    _write_summary_markdown(payload)
    return payload


def _write_summary_markdown(payload: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append(
        "# SRDA-v0 — static relational dictionary algebra: results summary"
    )
    lines.append("")
    lines.append(
        "Question: with the typed-token (16D) and parent (8D) identity channels "
        "deleted, does making the end-to-end dictionary assignment *define* the "
        "occurrence-level relation algebra open a new absolute strict-static "
        "ZINC MAE band?"
    )
    lines.append("")
    lines.append("Official ZINC **test was never loaded** in any stage.")
    lines.append("")
    lines.append("## Primary metric — fixed Top-5 weight soup (official-valid MAE)")
    lines.append("")
    lines.append("| run | best MAE | best epoch | epochs | soup MAE | params | wall s |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(
        f"| S0 seed0 (reference) | {S0_SEED0_BEST:.6f} | - | - | "
        f"{S0_SEED0_SOUP:.6f} | 66228 (typed+parent present) | (historical) |"
    )
    lines.append(
        f"| SDPK-v0 seed0 (reference) | {SDPK_SEED0_BEST:.6f} | 209 | 240 | "
        f"{SDPK_SEED0_SOUP:.6f} | 74996 | 888.1 |"
    )
    s0 = payload["seed0"]
    lines.append(
        f"| **SRDA-v0 seed0** | {s0['best_valid_mae']:.6f} | {s0['best_epoch']} | "
        f"{s0['epochs_run']} | {s0['soup_valid_mae']:.6f} | {s0['parameters']} | "
        f"{s0['wall_clock_s']:.1f} |"
    )
    if payload["seed1"] is not None:
        s1 = payload["seed1"]
        lines.append(
            f"| S0 seed1 (reference) | {S0_SEED1_BEST:.6f} | 233 | 240 | "
            f"{S0_SEED1_SOUP:.6f} | 66228 | (historical) |"
        )
        lines.append(
            f"| **SRDA-v0 seed1** | {s1['best_valid_mae']:.6f} | {s1['best_epoch']} | "
            f"{s1['epochs_run']} | {s1['soup_valid_mae']:.6f} | {s1['parameters']} | "
            f"{s1['wall_clock_s']:.1f} |"
        )
    lines.append("")
    gate0 = payload["seed0_gate"]
    lines.append("## Seed-0 performance gate (both must hold)")
    lines.append("")
    lines.append(f"* Top-5 soup <= {gate0['gate_soup_abs']}: measured {gate0['srda_soup']:.6f}")
    lines.append(f"* best valid <= {gate0['gate_best_abs']}: measured {gate0['srda_best']:.6f}")
    lines.append(
        f"* improvement vs S0 seed0: best {gate0['best_improvement_vs_s0']:+.6f}, "
        f"soup {gate0['soup_improvement_vs_s0']:+.6f}"
    )
    lines.append(
        f"* improvement vs SDPK-v0 seed0: best {gate0['best_improvement_vs_sdpl']:+.6f}, "
        f"soup {gate0['soup_improvement_vs_sdpl']:+.6f}"
    )
    lines.append(f"* verdict: **{gate0['verdict']}**")
    if payload.get("seed1_gate") is not None:
        gate1 = payload["seed1_gate"]
        lines.append("")
        lines.append("## Seed-1 conditional gate")
        lines.append("")
        lines.append(
            f"* soup improvement vs S0 seed1: {gate1['soup_improvement_vs_s0']:+.6f} "
            f"(gate {gate1['gate_soup_delta']}), absolute soup gate {gate1['gate_soup_abs']}"
        )
        lines.append(f"* verdict: **{gate1['verdict']}**")
    lines.append("")
    lines.append("## Runtime widths (attested)")
    lines.append("")
    widths = s0["runtime_widths"]
    for key in (
        "local_input_width",
        "local_state_width",
        "dictionary_assignment_width",
        "prototype_factor_width",
        "residual_input_width",
        "residual_width",
        "pair_input_width",
        "relation_input_width",
        "unary_width",
        "pair_pool_width",
        "global_width",
        "topology_width",
        "graph_width",
    ):
        lines.append(f"* {key}: {widths.get(key)}")
    identity = widths.get("identity_channels", {})
    lines.append(
        f"* identity-channel params: {identity.get('identity_channel_params')} "
        f"(only embedding: {identity.get('only_bucket_embedding')})"
    )
    lines.append("")
    lines.append("## Dictionary / decomposition diagnostics (report only, never a gate)")
    lines.append("")
    diag = s0["dictionary_diagnostics"]
    for key in (
        "mean_assignment_entropy",
        "effective_atom_count",
        "active_atom_count",
        "argmax_used_atoms",
        "top8_assignment_mass",
        "dictionary_coherence_mean_abs",
        "dictionary_coherence_max_abs",
        "tau_final",
        "mean_residual_norm",
        "mean_norm_p_dict",
        "mean_norm_p_eps",
        "mean_norm_rel_feat",
    ):
        lines.append(f"* {key}: {diag.get(key)}")
    lines.append(
        "* pair encoder first-layer block weight norms: "
        f"{diag.get('pair_encoder_first_layer_block_weight_norms')}"
    )
    lines.append("")
    lines.append("## Cheap inference interventions")
    lines.append("")
    lines.append("| intervention | mean |dpred| | max |dpred| | valid MAE after |")
    lines.append("|---|---|---|---|")
    for mode in ("neutral_dict", "zero_eps", "mean_alpha"):
        block = s0["intervention"].get(mode, {})
        lines.append(
            f"| {mode} | {block.get('mean_abs_prediction_shift')} | "
            f"{block.get('max_abs_prediction_shift')} | "
            f"{block.get('valid_mae_after_intervention')} |"
        )
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    budget = payload["budget"]
    lines.append(
        f"* full training runs used: {budget['full_runs_used']} / {budget['max_full_runs']}"
    )
    lines.append(f"* seed 1 purchased: {budget['seed1_purchased']}")
    lines.append("* matched dense/raw control: none (not purchased this round)")
    lines.append("* HPO / sweep: none")
    lines.append("* official test accessed = false")
    lines.append("")
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_all() -> None:
    parameter_audit()
    prepare_encoded()
    baseline_guard()
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
            "integrity",
            "encode",
            "baseline_guard",
            "smoke",
            "train",
            "analyze",
            "all",
        ],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--valid-subset", type=int, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    _configure_determinism()
    if args.stage == "param_audit":
        print(json.dumps(parameter_audit(), indent=2))
    elif args.stage == "integrity":
        print(json.dumps(integrity_stage(), indent=2))
    elif args.stage == "encode":
        print(json.dumps(prepare_encoded(), indent=2))
    elif args.stage == "baseline_guard":
        print(json.dumps(baseline_guard(), indent=2))
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(args.device), indent=2))
    elif args.stage == "train":
        summary = train_arm(
            seed=args.seed,
            device=args.device,
            max_epochs=args.epochs,
            train_subset=args.train_subset,
            valid_subset=args.valid_subset,
            tag=args.tag,
        )
        print(
            json.dumps(
                {k: v for k, v in summary.items() if "predictions" not in k}, indent=2
            )
        )
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2))
    else:
        run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
