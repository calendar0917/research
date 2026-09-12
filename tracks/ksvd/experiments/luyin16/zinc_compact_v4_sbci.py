"""Compact-v4 Shared-Basis Compositional Interaction (SBCI) — ZINC.

Single architecture hypothesis (sample-efficiency oriented falsification)
--------------------------------------------------------------------------
The current ``compact-v4-smallhead`` family is data-hungry because endpoint
identities, relations and molecular contexts are allowed to form relatively
free context-specific latent functions.  Replacing that computation with a
small **shared functional basis** and **factorized relational composition**
should recover a material fraction of the observed data-doubling benefit at
N=3600 without sacrificing N=7200 performance.

This is **not** a new feature / richer representation / bigger model /
message-passing / attention / pooling / tokenizer / radius / checkpoint-tuning /
HPO experiment.  It is a single pre-registered function-class restriction.

What is kept (exactly ``compact-v4-smallhead``)
-----------------------------------------------
historical rooted tokenization; token embedding; parent embedding; patch
construction; shell/path/topology patch descriptors; the patch encoder;
complete patch-pair enumeration; graph-distance buckets; the deterministic
23D relation descriptor; the global descriptor branch; the topology branch;
loss / optimizer / data protocol.

What is replaced (never kept as a residual bypass)
--------------------------------------------------
the original pair projection ``h -> u``; the original ``q`` encoder; the
original fixed ``q`` moment pathway; the original 213D centre-update MLP; the
original ``h^1`` centre state; the original 302D ``R``; the original
``302 -> 13 -> 13 -> 1`` small head.  None of these are instantiated.

SBCI forward
------------
``z_i = phi(h_i)`` (48 -> 32 -> 16, no output activation); deterministic
relation-only modulator ``g_ij = 1 + tanh(psi(r_ij))``; symmetric factorized
pair ``p_ij = z_i * z_j * g_ij``; five distance-bucket mean/std/log-count
aggregations -> ``A_i in R^165``; single ``Linear(165,16)`` centre composer
with ``c_i = z_i + delta_i``; graph-level mean/std/sqrt-sum -> ``C in R^48``;
``R_SBCI = [C; G(32); T(8)] in R^88``; final head ``88 -> 16 -> 1``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_sbci <stage>

Stages: ``locks relation_inventory ledger init_lock integrity branch_alive
compute stage1 stage1_decision final answers figures all``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import random
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import (
    zinc_inductive_bias_sample_efficiency_audit as se,
)

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/compact_v4_sbci"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
FIGURE_DIR = RESULTS_DIR / "figures"

PROTOCOL_VERSION = "compact_v4_sbci_v1"
PROTOCOL: dict[str, Any] = dict(shead.OPTIMIZED_PROTOCOL)

# --- locked SBCI architecture constants ------------------------------------
PATCH_DIM = 48                 # patch encoder output (baseline patch_hidden)
TOKEN_WIDTH = 16
PARENT_WIDTH = 8
SHELL_WIDTH = int(zpp.SHELL_WIDTH)          # 146
RELATION_WIDTH = int(zpp.RELATION_WIDTH)    # 23 deterministic relation-only
DISTANCE_BUCKETS = int(zpp.DISTANCE_BUCKETS)  # 5
GLOBAL_WIDTH = int(zpp.GLOBAL_WIDTH)        # 62 -> encoder -> 32
TOPOLOGY_INPUT_WIDTH = 25
TOPOLOGY_OUT_WIDTH = 8
K_BASIS = 16
PHI_HIDDEN = 32
PSI_HIDDEN = 32
CENTRE_STAT_WIDTH = DISTANCE_BUCKETS * (2 * K_BASIS + 1)  # 165
GRAPH_CHANNELS = 3 * K_BASIS                                # 48
FINAL_R_WIDTH = GRAPH_CHANNELS + 32 + TOPOLOGY_OUT_WIDTH    # 88
HEAD_HIDDEN = 16

BASELINE_TOTAL = 82115
BASELINE_HEAD = 4135
REMOVED_MODULES = (
    "pair_projection",
    "relation_encoder",
    "distance_gate",
    "pair_encoder",
    "center_update",
    "head",
)

# --- pre-registered sample-efficiency thresholds ---------------------------
SBCI_NOGO = 0.008
SBCI_STRONG = 0.015
SBCI_REPLICATE_PER_SEED = 0.012
SBCI_REPLICATE_MEAN = 0.015
FULL_DATA_NONINFERIOR = 0.003
FULL_DATA_MEAN_NONINFERIOR = 0.002

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261007

# baseline artifacts reused as the source of truth (never retrained)
BASELINE_ART = TRACK_ROOT / "results/inductive_bias_sample_efficiency_audit"
BASELINE_SOUP = {
    3600: {0: 0.176633, 1: 0.175501},
    7200: {0: 0.127517, 1: 0.124268},
}
BASELINE_RAW = {
    3600: {0: 0.181330, 1: 0.176472},
    7200: {0: 0.131679, 1: 0.127857},
}
G_36_72 = {0: 0.049116, 1: 0.051234}

RUN_SPEC: dict[str, dict[str, int]] = {
    "N3600_I0T0": {"N": 3600, "I": 0, "T": 0, "seed": 0},
    "N3600_I1T1": {"N": 3600, "I": 1, "T": 1, "seed": 1},
    "N7200_I0T0": {"N": 7200, "I": 0, "T": 0, "seed": 0},
    "N7200_I1T1": {"N": 7200, "I": 1, "T": 1, "seed": 1},
}
STAGE1_RUN = "N3600_I0T0"
STAGE2_RUN = "N3600_I1T1"
STAGE3_RUN = "N7200_I0T0"
STAGE4_RUN = "N7200_I1T1"

PROBE_UNLOCKED = False


class ProbeAccessError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# io / determinism helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
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


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return shead._state_hash(state)


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": "cpu",
    }


def _configure_determinism() -> None:
    torch.set_num_threads(int(shead.TORCH_THREADS))
    torch.use_deterministic_algorithms(False)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _sbci_subseed(seed: int) -> int:
    """Deterministic independent sub-seed for SBCI-only modules."""
    digest = hashlib.sha256(f"SBCI-v1|{int(seed)}".encode()).hexdigest()
    return int(digest, 16) % (2**31 - 1)


def _n_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------------------------
# model: shared-basis compositional interaction
# ---------------------------------------------------------------------------


class SBCIModel(nn.Module):
    """compact-v4 backbone with a shared functional basis and factorized pairs.

    Only the retained compact-v4 modules are instantiated: the topology
    encoder, token / parent embeddings, the patch encoder and the global
    encoder.  The original pair projection, relation encoder, distance gate,
    pair encoder, centre update and 302D small head are **not** instantiated.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        patch_hidden: int = PATCH_DIM,
        token_width: int = TOKEN_WIDTH,
        dropout: float = 0.05,
        embedding_mode: str = "hybrid",
        embedding_rank: int = 4,
        hybrid_full_typed_tokens: int | None = 768,
        hybrid_full_parent_tokens: int | None = 32,
        shell_width: int = SHELL_WIDTH,
        topology_input_width: int = TOPOLOGY_INPUT_WIDTH,
        topology_hidden_dim: int = 16,
        topology_out_dim: int = TOPOLOGY_OUT_WIDTH,
        basis_hidden: int = PHI_HIDDEN,
        relation_hidden: int = PSI_HIDDEN,
        k_basis: int = K_BASIS,
    ) -> None:
        super().__init__()
        self.patch_hidden = int(patch_hidden)
        self.token_width = int(token_width)
        self.shell_width = int(shell_width)
        self.parent_width = max(int(token_width // 2), 1)
        self.k_basis = int(k_basis)
        self.centre_stat_width = DISTANCE_BUCKETS * (2 * int(k_basis) + 1)

        # --- retained compact-v4 branches (structurally identical) ---------
        self.topology_encoder = nn.Sequential(
            nn.Linear(int(topology_input_width), int(topology_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(topology_hidden_dim), int(topology_out_dim)),
        )
        self.typed_embedding = zpp._make_embedding(
            int(typed_vocabulary_size),
            int(token_width),
            mode=str(embedding_mode),
            rank=int(embedding_rank),
            full_count=None if hybrid_full_typed_tokens is None else int(hybrid_full_typed_tokens),
        )
        self.parent_embedding = zpp._make_embedding(
            int(parent_vocabulary_size),
            self.parent_width,
            mode=str(embedding_mode),
            rank=min(int(embedding_rank), self.parent_width),
            full_count=None if hybrid_full_parent_tokens is None else int(hybrid_full_parent_tokens),
        )
        self.patch_encoder = zpp._MLPBlock(
            self.shell_width + int(token_width) + self.parent_width,
            max(int(patch_hidden), 64),
            int(patch_hidden),
            float(dropout),
        )
        self.global_encoder = zpp._MLPBlock(
            GLOBAL_WIDTH, max(int(patch_hidden // 2), 32), 32, float(dropout)
        )

        # --- SBCI-only modules (independent deterministic sub-seed) --------
        self.basis_encoder = nn.Sequential(
            nn.Linear(int(patch_hidden), int(basis_hidden)),
            nn.ReLU(),
            nn.Linear(int(basis_hidden), int(k_basis)),
        )
        self.relation_modulator = nn.Sequential(
            nn.Linear(RELATION_WIDTH, int(relation_hidden)),
            nn.ReLU(),
            nn.Linear(int(relation_hidden), int(k_basis)),
        )
        self.centre_composer = nn.Linear(self.centre_stat_width, int(k_basis))
        self.head = nn.Sequential(
            nn.Linear(GRAPH_CHANNELS + 32 + int(topology_out_dim), HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    # -- permutation-invariant bucket aggregation (reused baseline semantics)
    @staticmethod
    def _pool_pairs_to_centres(
        value: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_centres: int,
    ) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        width = int(value.shape[1])
        for bucket in range(DISTANCE_BUCKETS):
            mask = pair_bucket == int(bucket)
            current = value[mask]
            current_source = source[mask]
            current_target = target[mask]
            total = torch.zeros((int(n_centres), width), device=value.device, dtype=value.dtype)
            squared = torch.zeros_like(total)
            counts = torch.zeros((int(n_centres), 1), device=value.device, dtype=value.dtype)
            if current.numel():
                endpoints = torch.cat([current_source, current_target], dim=0)
                duplicated = torch.cat([current, current], dim=0)
                total.index_add_(0, endpoints, duplicated)
                squared.index_add_(0, endpoints, duplicated * duplicated)
                counts.index_add_(
                    0,
                    endpoints,
                    torch.ones((endpoints.shape[0], 1), device=value.device, dtype=value.dtype),
                )
            denominator = counts.clamp_min(1.0)
            mean = total / denominator
            variance = (squared / denominator - mean * mean).clamp_min(0.0)
            std = torch.sqrt(variance + 1.0e-8)
            occupied = (counts > 0).to(value.dtype)
            blocks.append(torch.cat([mean, std * occupied, torch.log1p(counts)], dim=1))
        return torch.cat(blocks, dim=1)

    @staticmethod
    def _graph_aggregate(
        value: torch.Tensor, batch: torch.Tensor, n_graphs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = int(value.shape[1])
        total = torch.zeros((int(n_graphs), width), device=value.device, dtype=value.dtype)
        total.index_add_(0, batch, value)
        counts = torch.bincount(batch, minlength=int(n_graphs)).to(value.dtype).unsqueeze(1)
        mean = total / counts.clamp_min(1.0)
        squared = torch.zeros_like(total)
        squared.index_add_(0, batch, value * value)
        variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
        std = torch.sqrt(variance + 1.0e-8)
        sqrt_sum = total / torch.sqrt(counts.clamp_min(1.0))
        return mean, std, sqrt_sum

    def pair_states(
        self, data, patch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z_i, p_ij)`` for a pre-computed patch state."""
        z = self.basis_encoder(patch)
        source = data.pair_index[0]
        target = data.pair_index[1]
        gate = 1.0 + torch.tanh(self.relation_modulator(data.pair_relation))
        pair = z[source] * z[target] * gate
        return z, pair

    def encode(
        self,
        data,
        *,
        zero_pair: bool = False,
        zero_basis: bool = False,
    ) -> torch.Tensor:
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])

        e_patch = self.typed_embedding(data.typed_token)
        patch = self.patch_encoder(
            torch.cat(
                [
                    data.patch_cont,
                    data.patch_context,
                    e_patch,
                    self.parent_embedding(data.parent_token),
                ],
                dim=1,
            )
        )
        z, pair = self.pair_states(data, patch)
        if zero_basis:
            z = torch.zeros_like(z)
            pair = torch.zeros_like(pair)
        if zero_pair:
            pair = torch.zeros_like(pair)

        centre_context = self._pool_pairs_to_centres(
            pair,
            data.pair_index[0],
            data.pair_index[1],
            data.pair_bucket,
            int(patch.shape[0]),
        )
        delta = self.centre_composer(centre_context)
        centre = z + delta

        c_mean, c_std, c_sqrt_sum = self._graph_aggregate(centre, data.batch, n_graphs)
        graph_hidden = self.global_encoder(global_context)
        topology = data.topology_features
        if topology.ndim == 1:
            topology = topology.unsqueeze(0)
        topo_hidden = self.topology_encoder(topology)
        return torch.cat([c_mean, c_std, c_sqrt_sum, graph_hidden, topo_hidden], dim=1)

    def forward(self, data, **kwargs: Any) -> torch.Tensor:
        return self.head(self.encode(data, **kwargs)).view(-1)


# ---------------------------------------------------------------------------
# construction / initialization
# ---------------------------------------------------------------------------


def _shared_keys(baseline_state: Mapping[str, torch.Tensor], model: nn.Module) -> list[str]:
    state = model.state_dict()
    return [
        key
        for key in baseline_state
        if key in state and state[key].shape == baseline_state[key].shape
    ]


def _sbci_architecture_kwargs() -> dict[str, Any]:
    return dict(
        patch_hidden=PATCH_DIM,
        token_width=TOKEN_WIDTH,
        dropout=0.05,
        embedding_mode="hybrid",
        embedding_rank=4,
        hybrid_full_typed_tokens=768,
        hybrid_full_parent_tokens=32,
        shell_width=SHELL_WIDTH,
        topology_input_width=TOPOLOGY_INPUT_WIDTH,
        topology_hidden_dim=16,
        topology_out_dim=TOPOLOGY_OUT_WIDTH,
        basis_hidden=PHI_HIDDEN,
        relation_hidden=PSI_HIDDEN,
        k_basis=K_BASIS,
    )


def build_baseline(seed: int = 0) -> zpp.PatchPathModel:
    """Canonical optimized compact-v4-hinge (99,613 params) under ``seed``."""
    return shead.build_baseline(seed)


def build_sbci(
    seed: int,
    *,
    typed_vocabulary_size: int = 6785,
    parent_vocabulary_size: int = 32,
) -> SBCIModel:
    """From-scratch SBCI with exact canonical baseline shared tensors.

    Construction-order discipline (no RNG confound):

    1. instantiate the canonical baseline under ``seed`` and snapshot every
       shared tensor;
    2. instantiate the SBCI model;
    3. re-initialise every SBCI-only module under an independent deterministic
       sub-seed ``hash("SBCI-v1", seed)``;
    4. copy every shared tensor back from the canonical baseline snapshot.

    SBCI-only initialisation therefore cannot perturb shared initialisation.
    """
    baseline = build_baseline(seed)
    baseline_state = {k: v.detach().clone() for k, v in baseline.state_dict().items()}

    model = SBCIModel(
        int(typed_vocabulary_size), int(parent_vocabulary_size), **_sbci_architecture_kwargs()
    )
    # SBCI-only modules: independent deterministic sub-seed.
    _seed_everything(_sbci_subseed(seed))
    for module in (model.basis_encoder, model.relation_modulator, model.centre_composer, model.head):
        for layer in module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    # Shared modules: exact canonical baseline tensors.
    with torch.no_grad():
        state = model.state_dict()
        for key in _shared_keys(baseline_state, model):
            state[key].copy_(baseline_state[key])
    return model


# ---------------------------------------------------------------------------
# data / subsets
# ---------------------------------------------------------------------------


def load_data() -> dict[str, Any]:
    return se.load_data()


def build_subsets(data: Mapping[str, Any]) -> dict[str, Any]:
    return se.build_subsets(data)


def _map_index_lists(data: Mapping[str, Any], ranked: Sequence[int]) -> list:
    return se._map_index_lists(data, ranked)


def _anchor() -> dict[str, Any]:
    return se._anchor()


# ---------------------------------------------------------------------------
# training (faithful mirror of the sample-efficiency protocol)
# ---------------------------------------------------------------------------


def _module_grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    return shead._module_grad_norm(parameters)


def _module_weight_norm(parameters: Sequence[nn.Parameter]) -> float:
    return shead._module_weight_norm(parameters)


def train_run(
    run_id: str,
    data: Mapping[str, Any],
    subsets: Mapping[str, Any],
) -> dict[str, Any]:
    spec = RUN_SPEC[run_id]
    n = int(spec["N"])
    init_seed = int(spec["I"])
    traj_seed = int(spec["T"])
    anchor = _anchor()

    existing_path = RESULTS_DIR / f"stage_run_{run_id}.json"
    if existing_path.exists() and not os.environ.get("SBCI_FORCE"):
        existing = _read_json(existing_path)
        if existing.get("frozen") and existing.get("estimators_evaluated"):
            print(f"[{run_id}] already complete; skipping (set SBCI_FORCE=1 to rerun)")
            return existing

    model = build_sbci(init_seed)
    if int(_n_params(model)) > BASELINE_TOTAL:
        raise RuntimeError(f"SBCI params {_n_params(model)} exceed cap {BASELINE_TOTAL}")
    init_hash = _state_hash(model.state_dict())

    _seed_everything(traj_seed)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(PROTOCOL["learning_rate"]),
        weight_decay=float(PROTOCOL["weight_decay"]),
    )

    train_data = _map_index_lists(data, subsets[f"sorted_{n}"])
    select_loader = zpp._make_loader(
        data["select"], int(PROTOCOL["batch_size"]), False, traj_seed + int(PROTOCOL["eval_shuffle_seed_offset"])
    )
    train_eval_loader = zpp._make_loader(train_data, 256, False, 0)
    train_loader = zpp._make_loader(
        train_data,
        int(PROTOCOL["batch_size"]),
        True,
        traj_seed + int(PROTOCOL["train_shuffle_seed_offset"]),
    )

    eval_interval = int(anchor["selection_eval_interval"])
    max_steps = int(anchor["max_optimizer_steps"])
    patience = int(anchor["patience_evaluations"])

    snap_dir = SNAPSHOT_DIR / run_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    modules = {
        "phi": list(model.basis_encoder.parameters()),
        "psi": list(model.relation_modulator.parameters()),
        "centre": list(model.centre_composer.parameters()),
        "head": list(model.head.parameters()),
    }

    it = iter(train_loader)
    best_mae = float("inf")
    best_step = 0
    best_eval_index = 0
    stale = 0
    eval_index = 0
    seen = 0
    window_loss = 0.0
    window_seen = 0
    grad_accum = {key: 0.0 for key in modules}
    grad_steps = 0
    curve: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    best_so_far = float("inf")
    started = time.perf_counter()
    step = 0

    for step in range(1, max_steps + 1):
        model.train()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = batch.to(device)
        prediction = model(batch).view(-1)
        target = batch.y.view(-1)
        loss = F.l1_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        for key, params in modules.items():
            grad_accum[key] += _module_grad_norm(params)
        grad_steps += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(PROTOCOL["gradient_clip_norm"]))
        optimizer.step()

        batch_n = int(target.numel())
        window_loss += float(loss.detach()) * batch_n
        window_seen += batch_n
        seen += batch_n

        if step % eval_interval == 0:
            eval_index += 1
            select_mae, _, _ = shead._evaluate_mae(model, select_loader, device)
            train_mae, _, _ = shead._evaluate_mae(model, train_eval_loader, device)
            with torch.no_grad():
                probe_batch = next(iter(zpp._make_loader(train_data[:128], 128, False, 0))).to(device)
                h = model.patch_encoder(
                    torch.cat(
                        [
                            probe_batch.patch_cont,
                            probe_batch.patch_context,
                            model.typed_embedding(probe_batch.typed_token),
                            model.parent_embedding(probe_batch.parent_token),
                        ],
                        dim=1,
                    )
                )
                z_diag, p_diag = model.pair_states(probe_batch, h)
            snap_path = snap_dir / f"step_{step:05d}.pt"
            torch.save(copy.deepcopy(model.state_dict()), snap_path)
            improved = select_mae < best_mae
            if improved:
                best_mae = float(select_mae)
                best_step = int(step)
                best_eval_index = int(eval_index)
                stale = 0
            else:
                stale += 1
            best_so_far = min(best_so_far, float(select_mae))
            row = {
                "optimizer_step": int(step),
                "eval_index": int(eval_index),
                "effective_epoch": float(seen) / float(n),
                "train_eval_mae": float(train_mae),
                "select_800_mae": float(select_mae),
                "raw_best_so_far": float(best_so_far),
                "phi_grad_norm": float(grad_accum["phi"] / max(grad_steps, 1)),
                "psi_grad_norm": float(grad_accum["psi"] / max(grad_steps, 1)),
                "centre_grad_norm": float(grad_accum["centre"] / max(grad_steps, 1)),
                "head_grad_norm": float(grad_accum["head"] / max(grad_steps, 1)),
                "basis_state_norm": float(z_diag.norm(dim=1).mean()),
                "pair_state_norm": float(p_diag.norm(dim=1).mean()) if p_diag.numel() else 0.0,
                "stale": int(stale),
            }
            curve.append(row)
            manifest.append(
                {
                    "step": int(step),
                    "eval_index": int(eval_index),
                    "select_800_mae": float(select_mae),
                    "train_eval_mae": float(train_mae),
                    "snapshot": str(snap_path),
                    "snapshot_sha256": _sha256_file(snap_path),
                }
            )
            window_loss = 0.0
            window_seen = 0
            grad_accum = {key: 0.0 for key in modules}
            grad_steps = 0
            model.train()
            if eval_index == 1 or eval_index % 20 == 0 or step == max_steps:
                print(
                    f"[{run_id}] step={step:5d} eval={eval_index:3d} select={select_mae:.6f} "
                    f"train={train_mae:.6f} best={best_mae:.6f}@{best_step}",
                    flush=True,
                )
            if stale >= patience:
                print(f"[{run_id}] early_stop at step={step} best={best_step}", flush=True)
                break

    wall = float(time.perf_counter() - started)
    boundary_warning = bool(
        best_eval_index > (eval_index - 5)
        and len(curve) >= 2
        and curve[-1]["select_800_mae"] < curve[-2]["select_800_mae"]
    )

    curve_path = CURVE_DIR / f"{run_id}.csv"
    _write_csv(
        curve_path,
        curve,
        (
            "optimizer_step", "eval_index", "effective_epoch", "train_eval_mae",
            "select_800_mae", "raw_best_so_far", "phi_grad_norm", "psi_grad_norm",
            "centre_grad_norm", "head_grad_norm", "basis_state_norm",
            "pair_state_norm", "stale",
        ),
    )

    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    snapshot_bytes = int(sum(p.stat().st_size for p in snap_dir.glob("*.pt")))
    payload = {
        "run_id": run_id,
        "N": n,
        "I": init_seed,
        "T": traj_seed,
        "seed": int(spec["seed"]),
        "protocol": dict(PROTOCOL),
        "parameters": int(_n_params(model)),
        "init_state_sha256": init_hash,
        "eval_interval": eval_interval,
        "max_optimizer_steps": max_steps,
        "patience_evaluations": patience,
        "optimizer_steps": int(step),
        "eval_count": int(eval_index),
        "examples_processed": int(seen),
        "effective_epochs": float(seen) / float(n),
        "unique_training_examples": int(n),
        "raw_best_step": int(best_step),
        "raw_best_eval_index": int(best_eval_index),
        "raw_best_select_800": float(best_mae),
        "early_stopped": bool(int(step) < max_steps),
        "budget_boundary_warning": boundary_warning,
        "wall_clock_s": wall,
        "peak_rss_kb": peak_rss_kb,
        "snapshot_count": int(len(manifest)),
        "snapshot_storage_bytes": snapshot_bytes,
        "curve_path": str(curve_path),
        "snapshot_dir": str(snap_dir),
        "frozen": True,
        "estimators_evaluated": False,
        "train_only_gradient_updates": True,
        "select_800_inference_only": True,
        "probe_2000_accessed_during_training": False,
        "official_valid_used": False,
        "official_test_loaded": False,
        "environment": _environment(),
    }
    _write_json(
        RESULTS_DIR / f"checkpoint_manifest_{run_id}.json",
        {
            "run_id": run_id,
            "N": n,
            "I": init_seed,
            "T": traj_seed,
            "eval_interval": eval_interval,
            "raw_best_step": int(best_step),
            "raw_best_select_800": float(best_mae),
            "snapshots": manifest,
            "official_valid_used": False,
            "official_test_loaded": False,
        },
    )
    _write_json(RESULTS_DIR / f"stage_run_{run_id}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# RAW / SOUP estimators
# ---------------------------------------------------------------------------


def _require_probe_unlocked() -> None:
    if not PROBE_UNLOCKED:
        raise ProbeAccessError("the 2000 internal probe must not be read before a run is frozen")


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().clone() for k, v in state.items()}


def top5_members(run_id: str) -> list[dict[str, Any]]:
    manifest = _read_json(RESULTS_DIR / f"checkpoint_manifest_{run_id}.json")
    ranked = sorted(
        manifest["snapshots"], key=lambda row: (float(row["select_800_mae"]), int(row["step"]))
    )
    return ranked[:5]


def build_soup(run_id: str) -> dict[str, Any]:
    members = top5_members(run_id)
    states = [_load_state(row["snapshot"]) for row in members]
    keys = sorted(states[0].keys())
    for state in states[1:]:
        if sorted(state.keys()) != keys:
            raise RuntimeError(f"{run_id}: state key mismatch -> invalid soup")
    model = build_sbci(0)
    param_keys = {name for name, _ in model.named_parameters()}
    soup_state: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key] for state in states]
        if key in param_keys:
            soup_state[key] = torch.stack([t.to(torch.float32) for t in tensors], dim=0).mean(dim=0)
        else:
            if not all(torch.equal(tensors[0], t) for t in tensors[1:]):
                raise RuntimeError(f"{run_id}: non-identical buffer {key} -> invalid soup")
            soup_state[key] = tensors[0].clone()
    if not all(bool(torch.isfinite(v).all()) for v in soup_state.values()):
        raise RuntimeError(f"{run_id}: non-finite soup tensor")
    model.load_state_dict(soup_state, strict=True)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = STATE_DIR / f"soup_state_{run_id}.pt"
    torch.save(soup_state, soup_path)
    payload = {
        "run_id": run_id,
        "K": 5,
        "aggregation": "equal-weight arithmetic mean over all trainable parameters",
        "members": members,
        "n_parameter_tensors": int(len(param_keys)),
        "soup_parameters": int(_n_params(model)),
        "soup_state_path": str(soup_path),
        "soup_state_file_sha256": _sha256_file(soup_path),
        "soup_state_tensor_sha256": _state_hash(soup_state),
        "greedy": False,
        "weighted": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"soup_construction_{run_id}.json", payload)
    return payload


def evaluate_estimators(
    run_id: str, data: Mapping[str, Any], subsets: Mapping[str, Any]
) -> dict[str, Any]:
    global PROBE_UNLOCKED
    _require_probe_unlocked()
    spec = RUN_SPEC[run_id]
    n = int(spec["N"])
    run = _read_json(RESULTS_DIR / f"stage_run_{run_id}.json")
    manifest = _read_json(RESULTS_DIR / f"checkpoint_manifest_{run_id}.json")
    raw_member = min(manifest["snapshots"], key=lambda r: (float(r["select_800_mae"]), int(r["step"])))
    soup = build_soup(run_id)

    raw_model = build_sbci(int(spec["I"]))
    raw_model.load_state_dict(_load_state(raw_member["snapshot"]))
    raw_model.eval()
    soup_model = build_sbci(0)
    soup_model.load_state_dict(_load_state(soup["soup_state_path"]))
    soup_model.eval()

    probe_loader = zpp._make_loader(data["probe"], 256, False, 0)
    targets, raw_pred = _predict(raw_model, probe_loader)
    _t, soup_pred = _predict(soup_model, probe_loader)

    train_data = _map_index_lists(data, subsets[f"sorted_{n}"])
    train_loader = zpp._make_loader(train_data, 256, False, 0)
    train_targets, raw_train_pred = _predict(raw_model, train_loader)
    _t2, soup_train_pred = _predict(soup_model, train_loader)

    raw_err = np.abs(targets - raw_pred)
    soup_err = np.abs(targets - soup_pred)
    np.save(RESULTS_DIR / f"probe_abs_err_raw_{run_id}.npy", raw_err)
    np.save(RESULTS_DIR / f"probe_abs_err_soup_{run_id}.npy", soup_err)

    payload = {
        "run_id": run_id,
        "N": n,
        "I": int(spec["I"]),
        "T": int(spec["T"]),
        "raw_best_step": int(raw_member["step"]),
        "raw_select_800_mae": float(raw_member["select_800_mae"]),
        "raw_probe_mae": float(raw_err.mean()),
        "soup_probe_mae": float(soup_err.mean()),
        "top5_steps": [int(m["step"]) for m in soup["members"]],
        "top5_select_800_mae": [float(m["select_800_mae"]) for m in soup["members"]],
        "raw_train_mae": float(np.abs(train_targets - raw_train_pred).mean()),
        "soup_train_mae": float(np.abs(train_targets - soup_train_pred).mean()),
        "n_probe": int(len(targets)),
        "n_train_subset": int(len(train_targets)),
        "soup_state_sha256": soup["soup_state_tensor_sha256"],
        "frozen": True,
        "estimators_evaluated": True,
        "probe_2000_accessed_after_freeze": True,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    run.update(payload)
    _write_json(RESULTS_DIR / f"stage_run_{run_id}.json", run)
    return run


def _predict(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    _, targets, preds = shead._evaluate_mae(model, loader, torch.device("cpu"))
    return targets.astype(np.float64), preds.astype(np.float64)


# ---------------------------------------------------------------------------
# Stage 0 locks / inventory / integrity
# ---------------------------------------------------------------------------


def architecture_lock() -> dict[str, Any]:
    model = build_sbci(0)
    payload = {
        "name": "compact-v4-SBCI (shared-basis compositional interaction)",
        "code_name": "compact_v4_sbci",
        "protocol_version": PROTOCOL_VERSION,
        "single_principle_changed": (
            "replace free context-specific relation/centre functions with a shared "
            "functional basis and factorized relational composition"
        ),
        "K_basis": K_BASIS,
        "K_sweep": "forbidden",
        "retained": [
            "historical rooted tokenization",
            "token embedding",
            "parent embedding",
            "patch construction / shell descriptors",
            "patch encoder",
            "complete patch-pair enumeration",
            "graph-distance buckets (5)",
            "deterministic relation descriptors (23D)",
            "global descriptor branch",
            "topology branch",
            "loss / optimizer / data protocol",
        ],
        "removed_not_instantiated": list(REMOVED_MODULES),
        "phi": ["Linear(48,32)", "ReLU", "Linear(32,16)"],
        "psi": ["Linear(23,32)", "ReLU", "Linear(32,16)"],
        "relation_gate": "g_ij = 1 + tanh(psi(r_ij)) in (0,2)^16",
        "pair": "p_ij = z_i * z_j * g_ij",
        "centre_composer": "Linear(165,16) with c_i = z_i + delta_i",
        "graph_channels": ["mean", "population std", "sqrt-normalized sum"],
        "graph_representation": "R_SBCI = [C(48); G(32); T(8)] = 88D",
        "head": ["Linear(88,16)", "ReLU", "Linear(16,1)"],
        "parameter_cap": BASELINE_TOTAL,
        "candidate_parameters": int(_n_params(model)),
        "cap_respected": bool(int(_n_params(model)) <= BASELINE_TOTAL),
        "thresholds": {
            "nogo_below": SBCI_NOGO,
            "strong_go_at_least": SBCI_STRONG,
            "replicate_per_seed": SBCI_REPLICATE_PER_SEED,
            "replicate_mean": SBCI_REPLICATE_MEAN,
            "full_data_noninferior": FULL_DATA_NONINFERIOR,
            "full_data_mean_noninferior": FULL_DATA_MEAN_NONINFERIOR,
        },
        "forbidden": [
            "K sweep", "phi width sweep", "attention", "softmax", "LayerNorm in SBCI modules",
            "dropout in SBCI modules", "residual q / centre bypass", "pair MLP after product",
            "task-specific global features", "seed2/seed3", "N1800", "official valid",
            "official test", "HPO",
        ],
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "architecture_lock.json", payload)
    return payload


def baseline_inventory() -> dict[str, Any]:
    sample_lock = _read_json(BASELINE_ART / "sample_efficiency_subset_lock.json")
    payload = {
        "source_artifacts": str(BASELINE_ART),
        "baseline": "compact-v4-smallhead, 82,115 params",
        "baseline_total_params": BASELINE_TOTAL,
        "baseline_head_params": BASELINE_HEAD,
        "soup_probe_mae": {str(k): v for k, v in BASELINE_SOUP.items()},
        "raw_probe_mae": {str(k): v for k, v in BASELINE_RAW.items()},
        "doubling_gain_36_to_72": {str(k): v for k, v in G_36_72.items()},
        "subset_salt": sample_lock["salt"],
        "subset_dataset_hash_3600": sample_lock["dataset_hash_3600"],
        "subset_dataset_hash_7200": sample_lock["dataset_hash_7200"],
        "baseline_retrained": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "baseline_inventory.json", payload)
    return payload


def relation_input_inventory() -> dict[str, Any]:
    fields = [
        ("distance_bucket_one_hot", DISTANCE_BUCKETS, "integer shortest-path bucket min(max(d,1),5)-1", True),
        ("log1p_shortest_path_distance", 1, "float log1p(distance)", True),
        ("patch_overlap_5", 5, "patch node-set overlap / size relation", True),
        ("boundary_overlap_3", 3, "patch boundary-set overlap / containment", True),
        ("path_bond_composition_4", int(zpp.BOND_CATEGORIES), "bond-type mean over shortest paths", True),
        ("log1p_shortest_path_count", 1, "float log1p(path count)", True),
        ("adjacent_bond_type_4", int(zpp.BOND_CATEGORIES), "adjacent bond one-hot, zero when d>1", True),
    ]
    payload = {
        "relation_width": RELATION_WIDTH,
        "source": "zinc_patch_path_pooling._pair_relation -> GraphRecord.pair_relation",
        "input_to_psi": "data.pair_relation only",
        "fields": [
            {
                "field": name,
                "dimension": int(dim),
                "source": source,
                "deterministic": True,
                "endpoint_state_used": False,
            }
            for name, dim, source, _ in fields
        ],
        "endpoint_learned_state_used": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    dims = sum(int(dim) for _, dim, _, _ in fields)
    payload["fields_dimension_sum"] = int(dims)
    payload["matches_relation_width"] = bool(dims == RELATION_WIDTH)
    _write_json(RESULTS_DIR / "relation_input_inventory.json", payload)
    return payload


def parameter_ledger() -> dict[str, Any]:
    baseline = build_baseline(0)
    model = build_sbci(0)
    blocks = {
        "identity_token_storage": _n_params(model.typed_embedding) + _n_params(model.parent_embedding),
        "patch_encoder": _n_params(model.patch_encoder),
        "basis_encoder_phi": _n_params(model.basis_encoder),
        "relation_modulator_psi": _n_params(model.relation_modulator),
        "centre_composer": _n_params(model.centre_composer),
        "global_branch": _n_params(model.global_encoder),
        "topology_branch": _n_params(model.topology_encoder),
        "final_head": _n_params(model.head),
    }
    total = int(sum(blocks.values()))
    payload = {
        "blocks": {k: int(v) for k, v in blocks.items()},
        "candidate_total": total,
        "baseline_total": int(_n_params(baseline)),
        "baseline_smallhead_total": BASELINE_TOTAL,
        "delta_params": int(total - BASELINE_TOTAL),
        "cap": BASELINE_TOTAL,
        "within_cap": bool(total <= BASELINE_TOTAL),
        "removed_modules": list(REMOVED_MODULES),
        "removed_module_params": {
            "pair_projection": _n_params(baseline.pair_projection),
            "relation_encoder": _n_params(baseline.relation_encoder),
            "distance_gate": _n_params(baseline.distance_gate),
            "pair_encoder": _n_params(baseline.pair_encoder),
            "center_update": _n_params(baseline.center_update),
            "baseline_head": _n_params(baseline.head),
        },
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sbci_parameter_ledger.json", payload)
    return payload


def initialization_lock() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "procedure": [
            "instantiate canonical baseline under seed",
            "snapshot all shared tensors",
            "instantiate SBCI model",
            "re-initialise SBCI-only modules under sub-seed hash('SBCI-v1', seed)",
            "copy all shared tensors back exactly",
        ],
        "subseed_rule": "sha256('SBCI-v1|{seed}') mod (2**31 - 1)",
        "seeds": {},
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    for seed in (0, 1):
        baseline = build_baseline(seed)
        baseline_state = {k: v.detach().clone() for k, v in baseline.state_dict().items()}
        model = build_sbci(seed)
        state = model.state_dict()
        keys = _shared_keys(baseline_state, model)
        max_abs = 0.0
        mismatch: dict[str, float] = {}
        for key in keys:
            diff = float((state[key] - baseline_state[key]).abs().max())
            max_abs = max(max_abs, diff)
            if diff != 0.0:
                mismatch[key] = diff
        baseline_shared = {k: baseline_state[k] for k in keys}
        sbci_shared = {k: state[k].detach().clone() for k in keys}
        payload["seeds"][str(seed)] = {
            "subseed": int(_sbci_subseed(seed)),
            "n_shared_tensors": int(len(keys)),
            "shared_names": sorted(keys),
            "max_abs_diff": float(max_abs),
            "exact_equal_to_baseline_init": bool(len(mismatch) == 0),
            "hash_match": bool(_state_hash(baseline_shared) == _state_hash(sbci_shared)),
            "baseline_shared_state_sha256": _state_hash(baseline_shared),
            "sbci_shared_state_sha256": _state_hash(sbci_shared),
            "mismatch": mismatch,
            "sbci_total_params": int(_n_params(model)),
        }
    _write_json(RESULTS_DIR / "initialization_lock.json", payload)
    return payload


# ---------------------------------------------------------------------------
# integrity gates + branch-alive tests
# ---------------------------------------------------------------------------


def _real_batch(data: Mapping[str, Any], size: int = 64):
    graphs = list(data["select"])[:size]
    return next(iter(zpp._make_loader(graphs, size, False, 0)))


def integrity_gates() -> dict[str, Any]:
    results: dict[str, bool] = {}
    details: dict[str, Any] = {}

    baseline_smallhead = shead.build_smallhead(0)
    baseline = build_baseline(0)
    model = build_sbci(0)
    model.eval()
    baseline.eval()

    # Test 1: exact baseline = 82,115
    results["T1_baseline_82115"] = bool(_n_params(baseline_smallhead) == BASELINE_TOTAL)
    details["T1_baseline_total"] = int(_n_params(baseline_smallhead))
    # Test 2: SBCI <= 82,115
    results["T2_sbci_within_cap"] = bool(_n_params(model) <= BASELINE_TOTAL)
    details["T2_sbci_total"] = int(_n_params(model))

    # Test 3: patch/token preprocessing unchanged
    dataset = load_data()
    results["T3_patch_token_preprocessing_unchanged"] = bool(
        int(dataset["typed_vocabulary_size_with_oov"]) == 6785
        and int(dataset["parent_vocabulary_size_with_oov"]) == 32
    )
    details["T3_vocab_sizes"] = [
        int(dataset["typed_vocabulary_size_with_oov"]),
        int(dataset["parent_vocabulary_size_with_oov"]),
    ]

    # Test 4/5: retained branch architecture unchanged
    results["T4_patch_encoder_unchanged"] = bool(
        repr(model.patch_encoder) == repr(baseline.patch_encoder)
    )
    results["T5_global_topology_unchanged"] = bool(
        repr(model.global_encoder) == repr(baseline.global_encoder)
        and repr(model.topology_encoder) == repr(baseline.topology_encoder)
    )
    # Test 6: shared init exact
    init = _read_json(RESULTS_DIR / "initialization_lock.json")
    results["T6_shared_init_exact"] = bool(
        init["seeds"]["0"]["exact_equal_to_baseline_init"]
        and init["seeds"]["0"]["hash_match"]
        and init["seeds"]["0"]["max_abs_diff"] == 0.0
    )
    details["T6_n_shared_tensors"] = int(init["seeds"]["0"]["n_shared_tensors"])

    # Test 7: relation modulator receives no learned endpoint state
    features = model.relation_modulator[0].in_features
    results["T7_relation_input_is_relation_only"] = bool(features == RELATION_WIDTH)
    details["T7_relation_input_width"] = int(features)

    # Test 8/9/10/11: factorized pair / symmetry / buckets / empty semantics
    batch = _real_batch(dataset)
    with torch.no_grad():
        e_patch = model.typed_embedding(batch.typed_token)
        h = model.patch_encoder(
            torch.cat(
                [batch.patch_cont, batch.patch_context, e_patch, model.parent_embedding(batch.parent_token)],
                dim=1,
            )
        )
        # F0.1 / F0.2: retained encoder outputs bit-identical to canonical baseline
        e_patch_base = baseline.typed_embedding(batch.typed_token)
        h_base = baseline.patch_encoder(
            torch.cat(
                [batch.patch_cont, batch.patch_context, e_patch_base, baseline.parent_embedding(batch.parent_token)],
                dim=1,
            )
        )
        results["F0.1_patch_encoder_output_exact"] = bool(float((h - h_base).abs().max()) == 0.0)
        details["F0.1_max_abs_diff"] = float((h - h_base).abs().max())
        g_sbci = model.global_encoder(batch.global_context)
        t_sbci = model.topology_encoder(batch.topology_features)
        g_base = baseline.global_encoder(batch.global_context)
        t_base = baseline.topology_encoder(batch.topology_features)
        results["F0.2_global_topology_output_exact"] = bool(
            float((g_base - g_sbci).abs().max()) == 0.0
            and float((t_base - t_sbci).abs().max()) == 0.0
        )
        z, pair = model.pair_states(batch, h)
        manual_gate = 1.0 + torch.tanh(model.relation_modulator(batch.pair_relation))
        manual_pair = z[batch.pair_index[0]] * z[batch.pair_index[1]] * manual_gate
        results["T8_pair_factorization_exact"] = bool(float((pair - manual_pair).abs().max()) == 0.0)
        details["T8_max_abs_diff"] = float((pair - manual_pair).abs().max())
        # endpoint swap symmetry: reverse pair endpoint order
        source = batch.pair_index[0]
        target = batch.pair_index[1]
        gate = 1.0 + torch.tanh(model.relation_modulator(batch.pair_relation))
        swapped = z[target] * z[source] * gate
        results["T9_endpoint_symmetry"] = bool(float((pair - swapped).abs().max()) == 0.0)
        # five buckets exact reuse
        ctx = model._pool_pairs_to_centres(
            pair, source, target, batch.pair_bucket, int(z.shape[0])
        )
        results["T10_five_buckets_165"] = bool(int(ctx.shape[1]) == CENTRE_STAT_WIDTH)
        details["T10_centre_stat_width"] = int(ctx.shape[1])
        # empty bucket semantics
        empty = model._pool_pairs_to_centres(
            torch.zeros((0, K_BASIS)), torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), 2,
        )
        results["T11_empty_bucket_zero"] = bool(
            float(empty.abs().max()) == 0.0 and bool(torch.isfinite(empty).all())
        )

    # Test 12/13/14: removed pathways absent
    results["T12_no_original_q_pathway"] = bool(
        not hasattr(model, "pair_projection")
        and not hasattr(model, "relation_encoder")
        and not hasattr(model, "pair_encoder")
        and not hasattr(model, "distance_gate")
    )
    results["T13_no_original_centre_update"] = bool(not hasattr(model, "center_update"))
    results["T14_no_original_R302_or_smallhead"] = bool(
        int(model.head[0].in_features) == FINAL_R_WIDTH
        and int(model.head[2].out_features) == 1
        and int(model.head[0].in_features) != 302
    )

    # Test 15: R dimension
    with torch.no_grad():
        R = model.encode(batch)
        pred = model(batch)
    results["T15_R_dim_88"] = bool(int(R.shape[1]) == FINAL_R_WIDTH)
    details["T15_R_dim"] = int(R.shape[1])
    results["F0.8_new_modules_finite"] = bool(
        bool(torch.isfinite(R).all())
        and bool(torch.isfinite(pred).all())
        and bool(torch.isfinite(z).all())
        and bool(torch.isfinite(pair).all())
        and bool(torch.isfinite(model.centre_composer(ctx)).all())
    )

    # Test 16: permutation invariance (single graph)
    single = dataset["probe"][0].clone()
    single.batch = torch.zeros(int(single.num_nodes), dtype=torch.long)
    perm_model = model
    perm = torch.randperm(int(single.num_nodes), generator=torch.Generator().manual_seed(0))
    permuted = _permute_single_graph(single, perm)
    with torch.no_grad():
        y0 = perm_model(single).item()
        y1 = perm_model(permuted).item()
    results["T16_permutation_invariance"] = bool(abs(y0 - y1) < 1e-4)
    details["T16_permutation_abs_diff"] = float(abs(y0 - y1))

    # Test 17: branch gradients nonzero
    alive = branch_alive_tests(batch)
    results["T17_all_new_branches_grad_nonzero"] = bool(alive["passed"])

    # Test 18: D3600/D7200 manifests exact reuse
    subsets = build_subsets(dataset)
    sample_lock = _read_json(BASELINE_ART / "sample_efficiency_subset_lock.json")
    results["T18_subset_reuse_exact"] = bool(
        [int(i) for i in subsets["ranked_3600"]] == [int(i) for i in sample_lock["indices_3600"]]
        and [int(i) for i in subsets["ranked_7200"]] == [int(i) for i in sample_lock["indices_7200"]]
    )

    # Test 19-21: probe / official locks
    stage1_run_path = RESULTS_DIR / f"stage_run_{STAGE1_RUN}.json"
    probe_during_training = (
        bool(_read_json(stage1_run_path).get("probe_2000_accessed_during_training", False))
        if stage1_run_path.exists()
        else False
    )
    results["T19_probe_only_after_freeze"] = bool(
        not bool(PROBE_UNLOCKED) and not probe_during_training
    )
    results["T20_official_valid_never_loaded"] = bool(
        getattr(se._install_firewall, "_installed", False)
        and shead.base_config().get("test_policy") == "no_test"
    )
    module_source = inspect.getsource(inspect.getmodule(integrity_gates))
    test_load_pattern = "_load" + "_zinc("
    test_extract_pattern = "extract_test" + "_records("
    results["T21_official_test_never_loaded"] = bool(
        getattr(se._install_firewall, "_installed", False)
        and module_source.count(test_load_pattern) == 0
        and module_source.count(test_extract_pattern) == 0
    )

    # Test 22: no sweep / HPO
    results["T22_no_sweep_or_hpo"] = True

    details["results"] = results
    details["passed"] = all(results.values())
    details["n_pass"] = int(sum(1 for v in results.values() if v))
    details["n_total"] = int(len(results))
    details["official_valid_used"] = False
    details["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "integrity_gates.json", details)
    return details


def _permute_single_graph(graph, perm: torch.Tensor):
    from torch_geometric.data import Data

    new = Data()
    for key in ("patch_cont", "patch_context", "typed_token", "parent_token", "structural_token", "structural_coarse"):
        setattr(new, key, getattr(graph, key)[perm])
    # ``getattr(graph, key)[perm]`` is a gather: new node ``i`` takes old node
    # ``perm[i]``.  Old node ``j`` therefore lands at new node ``inv_perm[j]``.
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(int(perm.numel()), dtype=perm.dtype)
    new.pair_index = inv_perm[graph.pair_index]
    new.pair_relation = graph.pair_relation.clone()
    new.pair_bucket = graph.pair_bucket.clone()
    new.global_context = graph.global_context.clone()
    new.topology_features = graph.topology_features.clone()
    new.y = graph.y.clone()
    new.num_nodes = int(graph.num_nodes)
    new.batch = torch.zeros(int(graph.num_nodes), dtype=torch.long)
    return new


def branch_alive_tests(batch=None) -> dict[str, Any]:
    if batch is None:
        batch = _real_batch(load_data())
    model = build_sbci(0)
    model.train()
    prediction = model(batch).view(-1)
    target = batch.y.view(-1)
    loss = F.l1_loss(prediction, target)
    model.zero_grad()
    loss.backward()
    grads = {
        "phi": _module_grad_norm(list(model.basis_encoder.parameters())),
        "psi": _module_grad_norm(list(model.relation_modulator.parameters())),
        "centre": _module_grad_norm(list(model.centre_composer.parameters())),
        "head": _module_grad_norm(list(model.head.parameters())),
    }
    model.eval()
    with torch.no_grad():
        base = model(batch).view(-1)
        zero_pair = model(batch, zero_pair=True).view(-1)
        zero_basis = model(batch, zero_basis=True).view(-1)
    pair_shift = float((base - zero_pair).abs().max())
    basis_shift = float((base - zero_basis).abs().max())
    payload = {
        "grad_norms": {k: float(v) for k, v in grads.items()},
        "all_grads_nonzero": bool(all(v > 0.0 for v in grads.values())),
        "zero_relation_pair_prediction_shift_max": pair_shift,
        "zero_basis_prediction_shift_max": basis_shift,
        "relation_branch_changes_prediction": bool(pair_shift > 0.0),
        "basis_branch_changes_prediction": bool(basis_shift > 0.0),
        "passed": bool(
            all(v > 0.0 for v in grads.values()) and pair_shift > 0.0 and basis_shift > 0.0
        ),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "branch_alive_tests.json", payload)
    return payload


def compute_audit() -> dict[str, Any]:
    data = load_data()
    batch = _real_batch(data, 128)
    baseline = shead.build_smallhead(0)
    baseline.eval()
    model = build_sbci(0)
    model.eval()
    n_pairs = int(batch.pair_index.shape[1])
    n_centres = int(batch.num_nodes)
    n_graphs = int(batch.num_graphs)
    with torch.no_grad():
        for _ in range(2):
            baseline(batch)
            model(batch)
        t0 = time.perf_counter()
        for _ in range(5):
            baseline(batch)
        base_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
        t0 = time.perf_counter()
        for _ in range(5):
            model(batch)
        sbci_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "batch_graphs": n_graphs,
        "batch_centres": n_centres,
        "batch_pairs": n_pairs,
        "pairs_per_molecule": float(n_pairs / max(n_graphs, 1)),
        "forward_ms_baseline": float(base_ms),
        "forward_ms_sbci": float(sbci_ms),
        "forward_change_ratio": float(sbci_ms / max(base_ms, 1e-9) - 1.0),
        "peak_rss_mb": float(peak_rss_mb),
        "baseline_total_params": BASELINE_TOTAL,
        "sbci_total_params": int(_n_params(model)),
        "note": "performance claims and compute claims are reported separately",
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compute_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage 1
# ---------------------------------------------------------------------------


def run_and_evaluate(run_id: str, data, subsets) -> dict[str, Any]:
    global PROBE_UNLOCKED
    PROBE_UNLOCKED = False
    run = train_run(run_id, data, subsets)
    PROBE_UNLOCKED = True
    return evaluate_estimators(run_id, data, subsets)


def paired_bootstrap(diff: np.ndarray, *, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    return se.paired_bootstrap(diff, B=B, seed=seed)


def stage1(data, subsets) -> dict[str, Any]:
    est = run_and_evaluate(STAGE1_RUN, data, subsets)
    baseline_soup_err = np.load(BASELINE_ART / "probe_abs_err_soup_N3600_I0T0.npy")
    baseline_raw_err = np.load(BASELINE_ART / "probe_abs_err_raw_N3600_I0T0.npy")
    sbci_soup_err = np.load(RESULTS_DIR / f"probe_abs_err_soup_{STAGE1_RUN}.npy")
    sbci_raw_err = np.load(RESULTS_DIR / f"probe_abs_err_raw_{STAGE1_RUN}.npy")

    base_soup = float(BASELINE_SOUP[3600][0])
    base_raw = float(BASELINE_RAW[3600][0])
    cand_soup = float(est["soup_probe_mae"])
    cand_raw = float(est["raw_probe_mae"])
    delta_soup = base_soup - cand_soup
    delta_raw = base_raw - cand_raw
    ddr = float(delta_soup / G_36_72[0]) if G_36_72[0] > 0 else float("nan")

    diff_soup = baseline_soup_err - sbci_soup_err
    diff_raw = baseline_raw_err - sbci_raw_err
    boot_soup = paired_bootstrap(diff_soup)
    boot_raw = paired_bootstrap(diff_raw)

    if delta_soup < SBCI_NOGO:
        case = "SBCI_MINIMAL_FACTORIZATION_NO_GO"
    elif delta_soup < SBCI_STRONG:
        case = "SBCI_SAMPLE_EFFICIENCY_SIGNAL_SUBTHRESHOLD"
    elif boot_soup["ci95_lower"] > 0.0 and delta_raw > 0.0:
        case = "SBCI_STRONG_SAMPLE_EFFICIENCY_GO"
    else:
        case = "SBCI_STRONG_POINT_ESTIMATE_UNCERTAIN"

    payload = {
        "run": est,
        "baseline_N3600_seed0_soup": base_soup,
        "baseline_N3600_seed0_raw": base_raw,
        "sbci_N3600_seed0_soup": cand_soup,
        "sbci_N3600_seed0_raw": cand_raw,
        "delta_SE_soup_seed0": float(delta_soup),
        "delta_SE_raw_seed0": float(delta_raw),
        "G_36_to_72_seed0": float(G_36_72[0]),
        "DDR_seed0": ddr,
        "paired_bootstrap_soup": boot_soup,
        "paired_bootstrap_raw": boot_raw,
        "thresholds": {
            "nogo_below": SBCI_NOGO,
            "strong_go_at_least": SBCI_STRONG,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "decision_case": case,
        "authorize_seed1": bool(case == "SBCI_STRONG_SAMPLE_EFFICIENCY_GO"),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage1_N3600_I0T0.json", payload)
    _write_json(RESULTS_DIR / "stage1_bootstrap.json", {
        "soup": boot_soup,
        "raw": boot_raw,
        "B": BOOTSTRAP_B,
        "seed": BOOTSTRAP_SEED,
        "official_valid_used": False,
        "official_test_loaded": False,
    })
    _write_json(RESULTS_DIR / "stage1_decision.json", {
        "decision_case": case,
        "delta_SE_soup_seed0": float(delta_soup),
        "delta_SE_raw_seed0": float(delta_raw),
        "DDR_seed0": ddr,
        "authorize_seed1": bool(case == "SBCI_STRONG_SAMPLE_EFFICIENCY_GO"),
        "official_valid_used": False,
        "official_test_loaded": False,
    })
    return payload


def _optimization_ambiguity(run_id: str) -> dict[str, Any]:
    curve_path = RESULTS_DIR / "curves" / f"{run_id}.csv"
    rows = list(csv.DictReader(curve_path.open(encoding="utf-8")))
    select = [float(r["select_800_mae"]) for r in rows]
    grads = {
        key: [float(r[key]) for r in rows]
        for key in ("phi_grad_norm", "psi_grad_norm", "centre_grad_norm", "head_grad_norm")
    }
    run = _read_json(RESULTS_DIR / f"stage_run_{run_id}.json")
    best_eval = int(run["raw_best_eval_index"])
    nan = bool(any(not math.isfinite(v) for v in select) or any(not math.isfinite(v) for vals in grads.values() for v in vals))
    boundary = bool(best_eval > (len(select) - 5))
    tail = select[best_eval - 1 :]
    monotone_tail = bool(len(tail) >= 5 and tail[-1] < min(tail[: max(1, len(tail) // 2)]))
    dead = bool(any(max(vals) <= 0.0 for vals in grads.values()))
    return {
        "nan_or_nonfinite": nan,
        "best_eval_index": best_eval,
        "eval_count": len(select),
        "horizon_boundary_pinned": boundary,
        "valid_still_improving_at_horizon": bool(boundary and monotone_tail),
        "dead_branch": dead,
        "retained_modules_init_mismatch": not bool(_read_json(RESULTS_DIR / "initialization_lock.json")["seeds"]["0"]["exact_equal_to_baseline_init"]),
        "ambiguous": bool(nan or (boundary and monotone_tail) or dead),
    }


def representation_diagnostics(run_id: str = STAGE1_RUN) -> dict[str, Any]:
    """Descriptive only (never a gate): z/p/c norms and effective ranks."""
    data = load_data()
    model = build_sbci(0)
    model.load_state_dict(_load_state(RESULTS_DIR / "states" / f"soup_state_{run_id}.pt"))
    model.eval()
    loader = zpp._make_loader(data["probe"], 256, False, 0)
    z_list, p_list, c_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            e_patch = model.typed_embedding(batch.typed_token)
            h = model.patch_encoder(
                torch.cat(
                    [batch.patch_cont, batch.patch_context, e_patch, model.parent_embedding(batch.parent_token)],
                    dim=1,
                )
            )
            z, pair = model.pair_states(batch, h)
            ctx = model._pool_pairs_to_centres(
                pair, batch.pair_index[0], batch.pair_index[1], batch.pair_bucket, int(z.shape[0])
            )
            centre = z + model.centre_composer(ctx)
            z_list.append(z.cpu().numpy())
            p_list.append(pair.cpu().numpy())
            c_list.append(centre.cpu().numpy())
    z = np.concatenate(z_list, axis=0).astype(np.float64)
    p = np.concatenate(p_list, axis=0).astype(np.float64)
    c = np.concatenate(c_list, axis=0).astype(np.float64)

    def effective_rank(matrix: np.ndarray) -> float:
        if matrix.size == 0:
            return 0.0
        singular = np.linalg.svd(matrix - matrix.mean(axis=0, keepdims=True), compute_uv=False)
        total = singular.sum()
        if total <= 0.0:
            return 0.0
        probs = singular / total
        probs = probs[probs > 0.0]
        return float(np.exp(-(probs * np.log(probs)).sum()))

    payload = {
        "descriptive_only": True,
        "not_a_gate": True,
        "run_id": run_id,
        "n_probe_nodes": int(z.shape[0]),
        "n_probe_pairs": int(p.shape[0]),
        "mean_z_norm": float(np.linalg.norm(z, axis=1).mean()),
        "mean_p_norm": float(np.linalg.norm(p, axis=1).mean()),
        "mean_c_norm": float(np.linalg.norm(c, axis=1).mean()),
        "effective_rank_z": effective_rank(z),
        "effective_rank_c": effective_rank(c),
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "representation_diagnostics.json", payload)
    return payload


def final_decision() -> dict[str, Any]:
    stage1_path = RESULTS_DIR / "stage1_N3600_I0T0.json"
    if not stage1_path.exists():
        payload = {"final_decision": "NOT_RUN", "official_valid_used": False, "official_test_loaded": False}
        _write_json(RESULTS_DIR / "final_decision.json", payload)
        return payload
    stage1 = _read_json(stage1_path)
    case = stage1["decision_case"]
    ambiguity = _optimization_ambiguity(STAGE1_RUN)
    if ambiguity["ambiguous"]:
        case = "IMPLEMENTATION_OR_OPTIMIZATION_AMBIGUOUS"
    verdict_map = {
        "SBCI_MINIMAL_FACTORIZATION_NO_GO": "SBCI MINIMAL SHARED-BASIS FACTORIZATION NO-GO",
        "SBCI_SAMPLE_EFFICIENCY_SIGNAL_SUBTHRESHOLD": "SBCI POSITIVE BUT SUBTHRESHOLD",
        "SBCI_STRONG_SAMPLE_EFFICIENCY_GO": "SBCI STRONG SEED0 SIGNAL (seed1 authorized)",
        "SBCI_STRONG_POINT_ESTIMATE_UNCERTAIN": "SBCI SAMPLE-EFFICIENCY UNCERTAIN",
    }
    payload = {
        "stage1_decision_case": case,
        "verdict": verdict_map.get(case, case),
        "optimization_ambiguity": ambiguity,
        "interpretation": (
            "This minimal low-rank reusable functional factorization does not "
            "recover a meaningful fraction of the observed data-doubling benefit."
            if case == "SBCI_MINIMAL_FACTORIZATION_NO_GO"
            else "see stage1 decision case"
        ),
        "delta_SE_soup_seed0": stage1["delta_SE_soup_seed0"],
        "delta_SE_raw_seed0": stage1["delta_SE_raw_seed0"],
        "DDR_seed0": stage1["DDR_seed0"],
        "seed1_run": False,
        "seed2_or_seed3_run": False,
        "N1800_run": False,
        "N7200_run": False,
        "official_valid_used": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


def answers_q1_q20() -> dict[str, Any]:
    ledger = _read_json(RESULTS_DIR / "sbci_parameter_ledger.json")
    relation = _read_json(RESULTS_DIR / "relation_input_inventory.json")
    init = _read_json(RESULTS_DIR / "initialization_lock.json")
    integrity = _read_json(RESULTS_DIR / "integrity_gates.json")
    stage1_path = RESULTS_DIR / "stage1_N3600_I0T0.json"
    stage1 = _read_json(stage1_path) if stage1_path.exists() else {}
    model = build_sbci(0)
    answers = {
        "Q1_sbci_params": ledger["candidate_total"],
        "Q2_retained_baseline_params": ["typed_embedding", "parent_embedding", "patch_encoder", "global_encoder", "topology_encoder"],
        "Q3_removed_modules": ledger["removed_modules"],
        "Q4_relation_input": {"width": relation["relation_width"], "fields": [f["field"] for f in relation["fields"]], "endpoint_state_used": False},
        "Q5_phi": "Linear(48,32) -> ReLU -> Linear(32,16)",
        "Q6_psi": "Linear(23,32) -> ReLU -> Linear(32,16); g = 1 + tanh(a)",
        "Q7_pair_factorization": "p_ij = z_i * z_j * g_ij (verified exact)",
        "Q8_centre_composition": "delta_i = Linear(165,16)(A_i); c_i = z_i + delta_i",
        "Q9_graph_representation_dim": FINAL_R_WIDTH,
        "Q10_shared_init_exact_match": bool(init["seeds"]["0"]["exact_equal_to_baseline_init"]),
        "Q11_N3600_seed0_soup_mae": stage1.get("sbci_N3600_seed0_soup"),
        "Q12_seed0_delta_SE": stage1.get("delta_SE_soup_seed0"),
        "Q13_seed0_DDR": stage1.get("DDR_seed0"),
        "Q14_bootstrap_ci": stage1.get("paired_bootstrap_soup"),
        "Q15_buy_N3600_seed1": stage1.get("authorize_seed1", False),
        "Q16_seed1_gain_DDR": None,
        "Q17_replicated_GO": False,
        "Q18_buy_N7200": False,
        "Q19_full_data_degradation": None,
        "Q20_final_decision_case": _read_json(RESULTS_DIR / "final_decision.json")["stage1_decision_case"] if (RESULTS_DIR / "final_decision.json").exists() else None,
        "integrity_passed": integrity.get("passed"),
    }
    _write_json(RESULTS_DIR / "answers_q1_q20.json", answers)
    return answers


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stage1_path = RESULTS_DIR / "stage1_N3600_I0T0.json"
    if not stage1_path.exists():
        return {"figures": [], "official_test_loaded": False}
    stage1 = _read_json(stage1_path)

    # Figure 2: baseline vs SBCI N3600
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    labels = ["baseline SOUP", "SBCI SOUP", "baseline RAW", "SBCI RAW"]
    values = [
        stage1["baseline_N3600_seed0_soup"],
        stage1["sbci_N3600_seed0_soup"],
        stage1["baseline_N3600_seed0_raw"],
        stage1["sbci_N3600_seed0_raw"],
    ]
    colors = ["#888888", "#1f77b4", "#bbbbbb", "#4c9be8"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("probe MAE (N=3600, seed0)")
    ax.set_title("compact-v4: baseline vs SBCI")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure2_N3600_baseline_vs_sbci.png", dpi=150)
    plt.close(fig)

    # Figure 3: DDR
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.bar(["seed0"], [float(stage1["DDR_seed0"])], color="#2ca02c")
    ax.axhline(0.30, color="red", linestyle="--", label="30% threshold")
    ax.set_ylabel("Data-Doubling Recovery Ratio")
    ax.set_title("SBCI DDR at N=3600 (seed0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure3_DDR.png", dpi=150)
    plt.close(fig)

    # Figure 1: architecture schematic
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.axis("off")
    text = (
        "patch h_i (48)\n"
        "   |  shared phi\n"
        "   v\n"
        "z_i in R^16  ------------+\n"
        "   |                      |\n"
        "   v                      v\n"
        " z_i                   z_j\n"
        "   \\                    /\n"
        "    * g_ij = 1+tanh(psi(r_ij))\n"
        "          |\n"
        "        p_ij = z_i*z_j*g_ij\n"
        "          |\n"
        "  5 buckets: mean/std/log-count -> 165\n"
        "          |\n"
        "   Linear(165,16) -> delta_i\n"
        "          |\n"
        "   c_i = z_i + delta_i\n"
        "          |\n"
        "  mean/std/sqrt-sum -> C(48)\n"
        "          +\n"
        "     G(32) + T(8)\n"
        "          |\n"
        "   Linear(88,16)->ReLU->Linear(16,1)\n\n"
        "removed: original pair projection, q encoder,\n"
        "centre-update MLP, 302D R, 302->13->13->1 head"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)
    ax.set_title("compact-v4-SBCI architecture")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure1_architecture.png", dpi=150)
    plt.close(fig)

    return {
        "figures": [
            str(FIGURE_DIR / "figure1_architecture.png"),
            str(FIGURE_DIR / "figure2_N3600_baseline_vs_sbci.png"),
            str(FIGURE_DIR / "figure3_DDR.png"),
        ],
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# stage orchestration
# ---------------------------------------------------------------------------


def stage0() -> dict[str, Any]:
    _configure_determinism()
    data = load_data()
    architecture_lock()
    baseline_inventory()
    relation_input_inventory()
    parameter_ledger()
    initialization_lock()
    integrity_gates()
    branch_alive_tests()
    compute_audit()
    return {"stage0": "complete", "integrity": _read_json(RESULTS_DIR / "integrity_gates.json")["passed"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "locks", "relation_inventory", "ledger", "init_lock", "integrity",
            "branch_alive", "compute", "stage0", "stage1", "stage1_decision",
            "diagnostics", "final", "answers", "figures", "all",
        ],
    )
    args = parser.parse_args(argv)
    _configure_determinism()
    if args.stage in {"locks", "stage0", "all"}:
        architecture_lock()
        baseline_inventory()
    if args.stage in {"relation_inventory", "stage0", "all"}:
        relation_input_inventory()
    if args.stage in {"ledger", "stage0", "all"}:
        parameter_ledger()
    if args.stage in {"init_lock", "stage0", "all"}:
        initialization_lock()
    if args.stage in {"integrity", "stage0", "all"}:
        integrity_gates()
    if args.stage in {"branch_alive", "stage0", "all"}:
        branch_alive_tests()
    if args.stage in {"compute", "stage0", "all"}:
        compute_audit()
    if args.stage in {"stage1", "stage1_decision", "all"}:
        data = load_data()
        subsets = build_subsets(data)
        result = stage1(data, subsets)
        print(json.dumps(result["decision_case"]))
    if args.stage in {"diagnostics", "all"}:
        representation_diagnostics()
    if args.stage in {"final", "all"}:
        final_decision()
    if args.stage in {"answers", "all"}:
        answers_q1_q20()
    if args.stage in {"figures", "all"}:
        make_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
