"""P1: compact-v4 learned relation-to-centre composer (minimal falsification).

This stage implements the **single** architecture hypothesis of the P1 task:

    Why must the already-computed learned pair relation ``q_ij`` be compressed
    into the *fixed* summary ``[mean(q); std(q); log1p(count)]`` before it enters
    the centre update?

The hypothesis under test (the only one):

    A small end-to-end learned permutation-invariant relation composer,
    operating *before* centre contextualization, can extract task-useful
    composition structure that fixed coordinatewise moments cannot provide.

What is added
-------------
* ``compact-v4`` base pathway, unchanged: patch -> pair -> centre aggregation
  -> centre update -> unary/pair/global/topology summaries -> ``R = 302D``.
* A new **relation-level** transform ``phi: R^16 -> R^24`` (shared across
  every graph, centre and distance bucket), applied to each individual pair
  state ``q_ij`` *before* pooling::

      z_ib      = (1 / n_ib) * sum_{q in Q_ib} phi(q)         in R^24
      u_ib      = [ z_ib ; log1p(n_ib) ]                       in R^25
      r_ib      = rho(u_ib)                                    in R^33
      a^P1_ib   = a^fixed_ib + r_ib                            (residual)

  with ``rho: R^25 -> R^33`` shared across all buckets.  The five 33D bucket
  summaries are concatenated in the original order, so the centre-update input
  dimension is **exactly unchanged** (``[h_i ; A_i] = [48 ; 165] = 213D``) and
  ``R`` remains **302D**.

What is *not* here
------------------
No relation refresh, no attention/Transformer, no generic message passing, no
change to tokenizer/patch radius/pair definition/relation descriptor/q dim/
pair encoder/number of buckets/centre-update architecture/final pair graph
moments/global descriptors/topology branch/graph head/optimizer/lr/batch/wd/
horizon/patience.  ``q^(0)_ij`` is computed exactly once.  The proven fixed
moments are retained as a residual baseline path.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_learned_centre_composer <stage>

Stages: ``architecture_lock baseline param_audit init_match integrity
compute_audit extract stage1_p1_seed0 stage1_decision bulk
stage2_capacity_control_seed0 stage2_decision
stage3_p1_seed1 stage3a_decision
stage3_capacity_control_seed1 stage3b_decision
answers final two_seed_summary figures all``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import inspect
import json
import math
import pickle
import platform
import random
import resource
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    TYPED_TOKENIZER_V1_HISTORICAL,
    resolve_typed_tokenizer_version,
    typed_tokenizer_fingerprint,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_learned_centre_composer"
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
Q_DIM = 16  # pair_hidden; dimension of q_ij
N_BUCKETS = 5  # integer distance buckets 1,2,3,4,5+
FIXED_BUCKET_WIDTH = 2 * Q_DIM + 1  # 33 = [mean 16 ; pop-std 16 ; log1p(count) 1]
CENTRE_CONTEXT_WIDTH = N_BUCKETS * FIXED_BUCKET_WIDTH  # 165
PHI_HIDDEN = 24
RHO_HIDDEN = 33  # == FIXED_BUCKET_WIDTH (Linear(25,33) -> ReLU -> Linear(33,33))
PSI_HIDDEN = 44  # capacity control: Linear(33,44) -> ReLU -> Linear(44,33)
PATCH_HIDDEN = 48
R_ORIGINAL = 302
R_P1 = R_ORIGINAL  # unchanged
ACTIVATION = "ReLU"
PARAM_CEILING = 103000
CONTROL_PARAM_MISMATCH_TOL = 0.01  # <= 1%

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
ARCH_GATE = 0.004  # Delta_P1,0 = v4_valid0 - p1_valid0
ARCH_GATE_SEED1 = 0.003
ARCH_MEAN_GATE = 0.004
MECH_GATE = 0.0025  # Delta_prepool,0 = MAE(control) - MAE(p1)
MECH_GATE_SEED1 = 0.0
MECH_MEAN_GATE = 0.0025
BULK_GATE = 0.002  # candidate MAE - v4 MAE on target-independent common-input bulk

CACHE_SCHEMA_VERSION = "compact_v4_composer_records_v1"
PROTOCOL_VERSION = "compact_v4_learned_centre_composer_v1"


# ---------------------------------------------------------------------------
# io / determinism helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
# model
# ---------------------------------------------------------------------------


class PatchPathComposerModel(zpp.PatchPathModel):
    """compact-v4 plus a learned pre-pool relation composer (P1 candidate).

    The base ``compact-v4`` pathway is inherited unchanged.  Only
    ``_pool_pairs_to_centres`` is overridden: after computing the *original*
    fixed ``[mean ; pop-std ; log1p(count)]`` summary via ``super()``, a shared
    relation transform ``phi`` is applied to every individual pair state, its
    permutation-invariant mean per ``(centre, bucket)`` is concatenated with the
    bucket log-count and mapped by a shared bottleneck composer ``rho`` into a
    33D residual that is added to the fixed bucket summary.  Empty buckets get
    exactly zero residual.  The centre-update input width is unchanged.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        composer_enabled: bool = True,
        zero_init_rho: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(typed_vocabulary_size, parent_vocabulary_size, **kwargs)
        if int(self.pair_hidden) != Q_DIM:
            raise ValueError(
                f"composer requires pair_hidden == {Q_DIM}; got {self.pair_hidden}"
            )
        self.composer_enabled = bool(composer_enabled)
        self.zero_init_rho = bool(zero_init_rho)
        self.phi = nn.Sequential(
            nn.Linear(Q_DIM, PHI_HIDDEN),
            nn.ReLU(),
            nn.Linear(PHI_HIDDEN, PHI_HIDDEN),
        )
        self.rho = nn.Sequential(
            nn.Linear(PHI_HIDDEN + 1, RHO_HIDDEN),
            nn.ReLU(),
            nn.Linear(RHO_HIDDEN, FIXED_BUCKET_WIDTH),
        )
        if self.zero_init_rho:
            nn.init.zeros_(self.rho[-1].weight)
            nn.init.zeros_(self.rho[-1].bias)
        self._reset_diagnostics()

    # -- diagnostics ---------------------------------------------------------
    def _reset_diagnostics(self) -> None:
        self.diag_residual_sum = 0.0
        self.diag_fixed_sum = 0.0
        self.diag_occupied_sum = 0.0
        self.diag_blocks = 0
        self.diag_bucket_residual_sum = [0.0] * N_BUCKETS
        self.diag_bucket_fixed_sum = [0.0] * N_BUCKETS
        self.diag_bucket_blocks = [0] * N_BUCKETS
        self.diag_phi_sum = 0.0
        self.diag_phi_sq_sum = 0.0
        self.diag_phi_count = 0
        self.last_residual_norm = 0.0
        self.last_fixed_norm = 0.0
        self.last_residual_ratio = 0.0
        self.last_nonempty_rate = 0.0

    def diagnostics(self) -> dict[str, Any]:
        denom = max(self.diag_blocks, 1)
        residual = self.diag_residual_sum / denom
        fixed = self.diag_fixed_sum / denom
        payload: dict[str, Any] = {
            "learned_residual_norm": float(residual),
            "fixed_summary_norm": float(fixed),
            "residual_fixed_ratio": float(residual / max(fixed, 1e-12)),
            "nonempty_bucket_rate": float(self.diag_occupied_sum / denom),
        }
        if any(self.diag_bucket_blocks):
            payload["bucket_residual_norm"] = [
                self.diag_bucket_residual_sum[b] / max(self.diag_bucket_blocks[b], 1)
                for b in range(N_BUCKETS)
            ]
            payload["bucket_fixed_norm"] = [
                self.diag_bucket_fixed_sum[b] / max(self.diag_bucket_blocks[b], 1)
                for b in range(N_BUCKETS)
            ]
        if self.diag_phi_count > 0:
            mean = self.diag_phi_sum / self.diag_phi_count
            var = self.diag_phi_sq_sum / self.diag_phi_count - mean * mean
            payload["phi_output_variance"] = float(max(var, 0.0))
        return payload

    # -- composer pooling ----------------------------------------------------
    def _pool_pairs_to_centres(
        self,
        value: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_centres: int,
    ) -> torch.Tensor:
        fixed = super()._pool_pairs_to_centres(
            value, source, target, pair_bucket, n_centres
        )
        if not self.composer_enabled:
            return fixed
        n_centres = int(n_centres)
        blocks: list[torch.Tensor] = []
        for bucket in range(N_BUCKETS):
            mask = pair_bucket == int(bucket)
            current = value[mask]
            counts = torch.zeros(
                (n_centres, 1), device=value.device, dtype=value.dtype
            )
            total_phi = torch.zeros(
                (n_centres, PHI_HIDDEN), device=value.device, dtype=value.dtype
            )
            if current.numel():
                endpoints = torch.cat([source[mask], target[mask]], dim=0)
                phi_value = self.phi(current)
                duplicated = torch.cat([phi_value, phi_value], dim=0)
                total_phi.index_add_(0, endpoints, duplicated)
                counts.index_add_(
                    0,
                    endpoints,
                    torch.ones(
                        (endpoints.shape[0], 1),
                        device=value.device,
                        dtype=value.dtype,
                    ),
                )
                with torch.no_grad():
                    self.diag_phi_sum += float(phi_value.detach().sum())
                    self.diag_phi_sq_sum += float(phi_value.detach().pow(2).sum())
                    self.diag_phi_count += int(phi_value.numel())
            z = total_phi / counts.clamp_min(1.0)
            log_count = torch.log1p(counts)
            u = torch.cat([z, log_count], dim=1)
            residual = self.rho(u)
            occupied = (counts > 0).to(value.dtype)
            # empty bucket semantics: exactly zero residual, no MLP-bias leak.
            residual = residual * occupied
            fixed_block = fixed[:, bucket * FIXED_BUCKET_WIDTH : (bucket + 1) * FIXED_BUCKET_WIDTH]
            blocks.append(fixed_block + residual)
            with torch.no_grad():
                self.diag_residual_sum += float(residual.norm(dim=1).mean())
                self.diag_fixed_sum += float(fixed_block.norm(dim=1).mean())
                self.diag_occupied_sum += float(occupied.mean())
                self.diag_blocks += 1
                self.diag_bucket_residual_sum[bucket] += float(residual.norm(dim=1).mean())
                self.diag_bucket_fixed_sum[bucket] += float(fixed_block.norm(dim=1).mean())
                self.diag_bucket_blocks[bucket] += 1
                self.last_residual_norm = float(residual.norm(dim=1).mean())
                self.last_fixed_norm = float(fixed_block.norm(dim=1).mean())
                self.last_residual_ratio = self.last_residual_norm / max(
                    self.last_fixed_norm, 1e-12
                )
                self.last_nonempty_rate = float(occupied.mean())
        return torch.cat(blocks, dim=1)


class PatchPathCapacityControlModel(zpp.PatchPathModel):
    """Matched post-pooling capacity control (P1-capacity-control).

    Adds a shared residual MLP ``psi: R^33 -> R^33`` applied to the *already
    fixed* ``[mean ; pop-std ; log1p(count)]`` bucket summary.  It never sees
    individual relation states *beyond* the fixed moments.  Empty buckets are
    forced to zero residual.  Parameter budget is matched to P1 to within 1%.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        control_enabled: bool = True,
        zero_init_psi: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(typed_vocabulary_size, parent_vocabulary_size, **kwargs)
        self.control_enabled = bool(control_enabled)
        self.zero_init_psi = bool(zero_init_psi)
        self.psi = nn.Sequential(
            nn.Linear(FIXED_BUCKET_WIDTH, PSI_HIDDEN),
            nn.ReLU(),
            nn.Linear(PSI_HIDDEN, FIXED_BUCKET_WIDTH),
        )
        if self.zero_init_psi:
            nn.init.zeros_(self.psi[-1].weight)
            nn.init.zeros_(self.psi[-1].bias)
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.diag_residual_sum = 0.0
        self.diag_fixed_sum = 0.0
        self.diag_occupied_sum = 0.0
        self.diag_blocks = 0
        self.last_residual_norm = 0.0
        self.last_fixed_norm = 0.0
        self.last_residual_ratio = 0.0
        self.last_nonempty_rate = 0.0

    def diagnostics(self) -> dict[str, float]:
        denom = max(self.diag_blocks, 1)
        residual = self.diag_residual_sum / denom
        fixed = self.diag_fixed_sum / denom
        return {
            "learned_residual_norm": float(residual),
            "fixed_summary_norm": float(fixed),
            "residual_fixed_ratio": float(residual / max(fixed, 1e-12)),
            "nonempty_bucket_rate": float(self.diag_occupied_sum / denom),
        }

    def _pool_pairs_to_centres(
        self,
        value: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_centres: int,
    ) -> torch.Tensor:
        fixed = super()._pool_pairs_to_centres(
            value, source, target, pair_bucket, n_centres
        )
        if not self.control_enabled:
            return fixed
        blocks: list[torch.Tensor] = []
        for bucket in range(N_BUCKETS):
            fixed_block = fixed[:, bucket * FIXED_BUCKET_WIDTH : (bucket + 1) * FIXED_BUCKET_WIDTH]
            # occupied depends only on log1p(count) > 0, i.e. it is derivable
            # from the fixed summary itself (no access to individual q).
            occupied = (fixed_block[:, -1:] > 0).to(fixed.dtype)
            residual = self.psi(fixed_block) * occupied
            blocks.append(fixed_block + residual)
            with torch.no_grad():
                self.diag_residual_sum += float(residual.norm(dim=1).mean())
                self.diag_fixed_sum += float(fixed_block.norm(dim=1).mean())
                self.diag_occupied_sum += float(occupied.mean())
                self.diag_blocks += 1
                self.last_residual_norm = float(residual.norm(dim=1).mean())
                self.last_fixed_norm = float(fixed_block.norm(dim=1).mean())
                self.last_residual_ratio = self.last_residual_norm / max(
                    self.last_fixed_norm, 1e-12
                )
                self.last_nonempty_rate = float(occupied.mean())
        return torch.cat(blocks, dim=1)


# ---------------------------------------------------------------------------
# config / builders
# ---------------------------------------------------------------------------


def load_config(path: Path = CANONICAL_V4_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def composer_config() -> dict[str, Any]:
    config = load_config()
    config["test_policy"] = "no_test"
    config["parameter_audit"] = False
    config["model"]["device"] = "cpu"
    config["output"] = {
        "json": str(RESULTS_DIR / "scratch.json"),
        "markdown": str(RESULTS_DIR / "scratch.md"),
    }
    return config


def _base_kwargs() -> dict[str, Any]:
    model_config = composer_config()["model"]
    return dict(
        patch_hidden=int(model_config.get("patch_hidden", PATCH_HIDDEN)),
        pair_hidden=int(model_config.get("pair_hidden", Q_DIM)),
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
    )


def build_baseline(seed: int = 0) -> zpp.PatchPathModel:
    _seed_everything(seed)
    return zpp.PatchPathModel(
        V4_TYPED_VOCAB_WITH_OOV, V4_PARENT_VOCAB_WITH_OOV, **_base_kwargs()
    )


def build_p1(
    seed: int = 0,
    *,
    composer_enabled: bool = True,
    typed_vocabulary_size: int = V4_TYPED_VOCAB_WITH_OOV,
    parent_vocabulary_size: int = V4_PARENT_VOCAB_WITH_OOV,
) -> PatchPathComposerModel:
    _seed_everything(seed)
    return PatchPathComposerModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        composer_enabled=composer_enabled,
        zero_init_rho=True,
        **_base_kwargs(),
    )


def build_control(
    seed: int = 0,
    *,
    control_enabled: bool = True,
    typed_vocabulary_size: int = V4_TYPED_VOCAB_WITH_OOV,
    parent_vocabulary_size: int = V4_PARENT_VOCAB_WITH_OOV,
) -> PatchPathCapacityControlModel:
    _seed_everything(seed)
    return PatchPathCapacityControlModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        control_enabled=control_enabled,
        zero_init_psi=True,
        **_base_kwargs(),
    )


def _n_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------------------------
# extraction / encoding
# ---------------------------------------------------------------------------


def _cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "records_train.pkl.gz",
        CACHE_DIR / "records_valid.pkl.gz",
        CACHE_DIR / "metadata.json",
    )


def extract_records(force: bool = False):
    """Extract and cache base GraphRecords for train/valid (official only)."""
    train_path, valid_path, meta_path = _cache_paths()
    if not force and train_path.exists() and valid_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
            with gzip.open(train_path, "rb") as handle:
                train_records = pickle.load(handle)
            with gzip.open(valid_path, "rb") as handle:
                valid_records = pickle.load(handle)
            return train_records, valid_records, meta

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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(train_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(valid_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "tokenizer_version": tokenizer_version,
        "topology_mode": topology_mode,
        "n_train": int(len(train_records)),
        "n_valid": int(len(valid_records)),
        "train_mean_centres": float(train_meta["mean_centres"]),
        "valid_mean_centres": float(valid_meta["mean_centres"]),
        "train_mean_pairs": float(train_meta["mean_pairs"]),
        "valid_mean_pairs": float(valid_meta["mean_pairs"]),
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(meta_path, meta)
    print(
        f"[extract] train={len(train_records)} valid={len(valid_records)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_records, valid_records, meta


def build_encoded_records():
    train_records, valid_records, _meta = extract_records()
    config = composer_config()
    enc_train, enc_valid, audit = zpp._phase_data(
        train_records, valid_records, config=config
    )
    return enc_train, enc_valid, audit


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------


def _evaluate_mae(
    model: nn.Module, loader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    if not targets:
        return float("nan"), np.zeros(0), np.zeros(0)
    t = np.concatenate(targets).astype(np.float64)
    p = np.concatenate(preds).astype(np.float64)
    return float(np.mean(np.abs(t - p))), t, p


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return math.sqrt(total)


def train_model(
    *,
    build_fn: Callable[[int], nn.Module],
    train_data: Sequence[Data],
    valid_data: Sequence[Data],
    seed: int,
    tag: str,
    grad_module_names: Sequence[str] = (),
    save_state: bool = True,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = build_fn(seed).to(device)
    grad_modules = {name: getattr(model, name) for name in grad_module_names}
    total_params = _n_params(model)
    if total_params > PARAM_CEILING:
        raise RuntimeError(
            f"parameter ceiling exceeded: {total_params} > {PARAM_CEILING}"
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(OPTIMIZED_PROTOCOL["learning_rate"]),
        weight_decay=float(OPTIMIZED_PROTOCOL["weight_decay"]),
    )
    loader = zpp._make_loader(
        train_data, int(OPTIMIZED_PROTOCOL["batch_size"]), True, seed + 91011
    )
    eval_loader = zpp._make_loader(
        valid_data, int(OPTIMIZED_PROTOCOL["batch_size"]), False, seed + 91012
    )
    steps_per_epoch = int(
        math.ceil(len(train_data) / int(OPTIMIZED_PROTOCOL["batch_size"]))
    )
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
        if hasattr(model, "_reset_diagnostics"):
            model._reset_diagnostics()  # type: ignore[attr-defined]
        epoch_loss = 0.0
        seen = 0
        grad_norm_sums = {name: 0.0 for name in grad_modules}
        joint_sq = 0.0
        grad_steps = 0
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            batch_joint_sq = 0.0
            for name, module in grad_modules.items():
                norm = _module_grad_norm(module)
                grad_norm_sums[name] += norm
                batch_joint_sq += norm * norm
            joint_sq += math.sqrt(batch_joint_sq)
            grad_steps += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(OPTIMIZED_PROTOCOL["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae, _, _ = _evaluate_mae(model, eval_loader, device)
        diag = model.diagnostics() if hasattr(model, "diagnostics") else {}
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_mae": float(train_mae),
            "valid_mae": float(valid_mae),
            "lr": float(OPTIMIZED_PROTOCOL["learning_rate"]),
            "optimizer_steps": int(epoch * steps_per_epoch),
            "checkpoint_selected": 0,
            "composer_grad_norm": float(joint_sq / max(grad_steps, 1)),
            "phi_grad_norm": float(grad_norm_sums.get("phi", 0.0) / max(grad_steps, 1)),
            "rho_grad_norm": float(grad_norm_sums.get("rho", 0.0) / max(grad_steps, 1)),
            "psi_grad_norm": float(grad_norm_sums.get("psi", 0.0) / max(grad_steps, 1)),
            "learned_residual_norm": float(diag.get("learned_residual_norm", 0.0)),
            "fixed_summary_norm": float(diag.get("fixed_summary_norm", 0.0)),
            "residual_fixed_ratio": float(diag.get("residual_fixed_ratio", 0.0)),
            "nonempty_bucket_rate": float(diag.get("nonempty_bucket_rate", 0.0)),
        }
        curve.append(row)
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
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"res={row['learned_residual_norm']:.4f} "
                f"phi_g={row['phi_grad_norm']:.3e}",
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
            "composer_grad_norm",
            "phi_grad_norm",
            "rho_grad_norm",
            "psi_grad_norm",
            "learned_residual_norm",
            "fixed_summary_norm",
            "residual_fixed_ratio",
            "nonempty_bucket_rate",
        ),
    )
    state_path = None
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        torch.save(model.state_dict(), state_path)

    final_valid, valid_targets, valid_predictions = _evaluate_mae(
        model, eval_loader, device
    )
    diagnostics = model.diagnostics() if hasattr(model, "diagnostics") else {}
    horizon_warning = bool(best_epoch >= max_epochs - max(1, max_epochs // 10))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "seed": int(seed),
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
        "diagnostics_final": diagnostics,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(),
        "official_test_loaded": False,
        "valid_predictions": valid_predictions.tolist(),
        "valid_targets": valid_targets.tolist(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "runs" / f"{tag}_seed{seed}.json", summary)
    print(
        f"[{tag} seed{seed}] best_valid={best_mae:.6f} epoch={best_epoch} "
        f"params={total_params} wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# Stage 0: locks, audits, gates
# ---------------------------------------------------------------------------


def architecture_lock() -> dict[str, Any]:
    config = composer_config()
    model = build_p1(seed=0)
    control = build_control(seed=0)
    p1_total = _n_params(model)
    control_total = _n_params(control)
    base_total = _n_params(build_baseline(seed=0))
    payload = {
        "name": "compact-v4 learned relation-to-centre composer (P1)",
        "protocol_version": PROTOCOL_VERSION,
        "single_principle_changed": (
            "relation composition itself is learned end-to-end (a shared "
            "permutation-invariant phi-pool-rho composer) before centre "
            "contextualization, instead of being fixed to coordinatewise "
            "mean/std/log-count"
        ),
        "q_dim": int(Q_DIM),
        "number_of_buckets": int(N_BUCKETS),
        "fixed_bucket_summary_definition": (
            "record: a^fixed_ib = [mean(q) in R^16 ; population_std(q) in R^16 ; "
            "log1p(count) in R^1] in R^33; empty bucket exactly zero; 5 buckets "
            "concatenated in bucket order => 165D"
        ),
        "phi_architecture": {
            "layers": [f"Linear({Q_DIM},{PHI_HIDDEN})", ACTIVATION, f"Linear({PHI_HIDDEN},{PHI_HIDDEN})"],
            "output_dim": int(PHI_HIDDEN),
            "shared": "across every graph, centre and distance bucket",
            "no": ["attention", "normalization", "dropout", "gating", "bucket-specific parameters"],
        },
        "phi_output_dim": int(PHI_HIDDEN),
        "rho_architecture": {
            "layers": [
                f"Linear({PHI_HIDDEN + 1},{RHO_HIDDEN})",
                ACTIVATION,
                f"Linear({RHO_HIDDEN},{FIXED_BUCKET_WIDTH})",
            ],
            "output_dim": int(FIXED_BUCKET_WIDTH),
            "shared": "across all five buckets",
            "no": ["LayerNorm", "dropout", "residual stack", "extra depth"],
        },
        "activation": ACTIVATION,
        "activation_source": "compact-v4 pair/centre MLP main activation (ReLU) parsed from code",
        "empty_bucket_rule": "n_ib == 0 => r_ib = 0 exactly => a^P1_ib = a^fixed_ib = 0",
        "residual_or_replace": "residual: a^P1_ib = a^fixed_ib + r_ib (fixed moments retained)",
        "zero_init_policy": "rho final Linear initialized to exact zeros (same for control psi)",
        "sharing_policy": (
            "one phi and one rho shared across all graphs/centres/buckets; bucket "
            "identity enters only via concat position; no learned bucket embedding"
        ),
        "R_dimension": int(R_P1),
        "centre_update_input_width": int(PATCH_HIDDEN + CENTRE_CONTEXT_WIDTH),
        "parameter_count": {
            "baseline_v4": int(base_total),
            "p1_total": int(p1_total),
            "p1_added": int(p1_total - base_total),
            "control_total": int(control_total),
            "control_added": int(control_total - base_total),
            "p1_control_added_mismatch_abs": int(abs((p1_total - base_total) - (control_total - base_total))),
            "ceiling": int(PARAM_CEILING),
        },
        "training_protocol": dict(OPTIMIZED_PROTOCOL),
        "decision_thresholds": {
            "arch_gate_seed0": ARCH_GATE,
            "arch_gate_seed1": ARCH_GATE_SEED1,
            "arch_mean_gate": ARCH_MEAN_GATE,
            "mech_gate_seed0": MECH_GATE,
            "mech_mean_gate": MECH_MEAN_GATE,
            "bulk_gate": BULK_GATE,
        },
        "architecture_gate": (
            "Delta_P1,0 = v4_seed0_valid - p1_seed0_valid >= +0.004"
        ),
        "mechanism_gate": (
            "Delta_prepool,0 = MAE(control_seed0) - MAE(p1_seed0) >= +0.0025 "
            "AND architecture gate AND bulk safe AND branch alive"
        ),
        "seed_policy": {
            "stage1": [0],
            "stage3": [1],
            "forbidden": [2, 3],
        },
        "forbidden": [
            "relation refresh (q computed once)",
            "attention / Transformer",
            "generic message passing",
            "change to tokenizer/patch radius/patch encoder/pair definition/relation descriptor/q dim/pair encoder/bucket count/centre-update architecture/final pair moments/global descriptors/topology branch/graph head",
            "adding h_i or endpoints or graph embedding or bucket embedding to phi",
            "re-encoding the raw 23D relation descriptor",
            "removing fixed moments (replace instead of residual)",
            "any sweep (phi width / rho width / sum vs mean / max / activation / bucket embedding / residual scaling)",
            "official test access",
        ],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "architecture_lock.json", payload)
    return payload


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
        "v4_seed0_state_sha256": (
            _sha256_file(V4_SEED0_STATE) if V4_SEED0_STATE.exists() else None
        ),
        "v4_seed1_state": str(V4_SEED1_STATE),
        "v4_seed1_state_exists": V4_SEED1_STATE.exists(),
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
        "historical_pooling_source_sha256": _sha256_text(
            inspect.getsource(zpp.PatchPathModel._pool_pairs_to_centres)
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "baseline_inventory.json", payload)
    return payload


def parameter_audit() -> dict[str, Any]:
    base = build_baseline(seed=0)
    model = build_p1(seed=0)
    control = build_control(seed=0)
    base_total = _n_params(base)
    p1_total = _n_params(model)
    control_total = _n_params(control)
    phi_params = _n_params(model.phi)
    rho_params = _n_params(model.rho)
    psi_params = _n_params(control.psi)
    payload = {
        "baseline_v4_params": int(base_total),
        "v4_expected_params": int(V4_PARAMS),
        "baseline_matches_reference": int(base_total) == int(V4_PARAMS),
        "p1_total_params": int(p1_total),
        "p1_phi_params": int(phi_params),
        "p1_rho_params": int(rho_params),
        "p1_added_params": int(p1_total - base_total),
        "p1_added_expected": int(phi_params + rho_params),
        "control_total_params": int(control_total),
        "control_psi_params": int(psi_params),
        "control_added_params": int(control_total - base_total),
        "p1_control_mismatch_abs": int(
            abs((p1_total - base_total) - (control_total - base_total))
        ),
        "p1_control_mismatch_ratio": float(
            abs((p1_total - base_total) - (control_total - base_total))
            / max(p1_total - base_total, 1)
        ),
        "param_ceiling": int(PARAM_CEILING),
        "under_ceiling": bool(p1_total <= PARAM_CEILING and control_total <= PARAM_CEILING),
        "r_original": int(R_ORIGINAL),
        "r_p1": int(R_P1),
        "q_dim": int(Q_DIM),
        "fixed_bucket_width": int(FIXED_BUCKET_WIDTH),
        "centre_context_width": int(model.center_context_width),
        "centre_update_input_width": int(PATCH_HIDDEN + model.center_context_width),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


def initialization_match() -> dict[str, Any]:
    """Verify P1/control share the exact seed0 baseline initial tensors.

    Constructs the baseline, P1 and control from scratch under seed 0 and
    compares every shared tensor (same name and shape) by SHA-256 and exact
    equality.  If construction RNG order ever diverged, the deterministic
    baseline tensors are explicitly copied back in.
    """
    base = build_baseline(seed=0)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    def _match(build: Callable[[int], nn.Module]) -> dict[str, Any]:
        model = build(0)
        state = model.state_dict()
        shared_keys = [
            k for k in base_state if k in state and state[k].shape == base_state[k].shape
        ]
        exact = True
        max_abs = 0.0
        for key in shared_keys:
            diff = float((state[key] - base_state[key]).abs().max())
            max_abs = max(max_abs, diff)
            if diff != 0.0:
                exact = False
        copied = False
        if not exact:
            mismatch = {
                k: float((state[k] - base_state[k]).abs().max())
                for k in shared_keys
                if float((state[k] - base_state[k]).abs().max()) != 0.0
            }
            with torch.no_grad():
                for key in shared_keys:
                    state[key].copy_(base_state[key])
            copied = True
        else:
            mismatch = {}
        # derive per-block hashes on the shared subset
        shared_hash = _state_hash({k: state[k] for k in shared_keys})
        return {
            "n_shared_tensors": len(shared_keys),
            "exact_equal_to_seed0_baseline": bool(exact),
            "max_abs_diff_initial": float(max_abs),
            "explicit_copy_applied": bool(copied),
            "mismatch": mismatch,
            "shared_state_sha256": shared_hash,
            "total_params": _n_params(model),
        }

    p1 = _match(lambda s: build_p1(s))
    control = _match(lambda s: build_control(s))
    payload = {
        "policy": "from-scratch initialization matching (NOT trained-checkpoint warm start)",
        "seed": 0,
        "baseline_total_params": _n_params(base),
        "p1": p1,
        "control": control,
        "p1_matches_control_shared": bool(
            p1["exact_equal_to_seed0_baseline"] and control["exact_equal_to_seed0_baseline"]
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "initialization_match.json", payload)
    return payload


def _permute_pairs(
    value: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    pair_bucket: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(value.shape[0])
    perm = torch.randperm(n, generator=generator)
    return value[perm], source[perm], target[perm], pair_bucket[perm]


def integrity_gates(report: bool = True) -> dict[str, Any]:
    results: dict[str, bool] = {}
    gates: dict[str, Any] = {}

    base = build_baseline(seed=0)
    model = build_p1(seed=0)
    model.eval()
    control = build_control(seed=0)
    control.eval()

    # G0.1 residual disabled/zeroed == baseline (shared weights).
    base_state = base.state_dict()
    shared = {
        k: v for k, v in model.state_dict().items() if k in base_state and base_state[k].shape == v.shape
    }
    base.load_state_dict(shared, strict=False)
    base.eval()
    gen = torch.Generator().manual_seed(20260911)
    n_centres, n_pairs = 7, 40
    value = torch.randn(n_pairs, Q_DIM, generator=gen)
    source = torch.randint(0, n_centres, (n_pairs,), generator=gen)
    target = torch.randint(0, n_centres, (n_pairs,), generator=gen)
    pair_bucket = torch.randint(0, N_BUCKETS, (n_pairs,), generator=gen)
    with torch.no_grad():
        fixed_parent = zpp.PatchPathModel._pool_pairs_to_centres(
            base, value, source, target, pair_bucket, n_centres
        )
        p1_active = model._pool_pairs_to_centres(
            value, source, target, pair_bucket, n_centres
        )
        model.composer_enabled = False
        p1_disabled = model._pool_pairs_to_centres(
            value, source, target, pair_bucket, n_centres
        )
        model.composer_enabled = True
    zero_init_diff = float((p1_active - fixed_parent).abs().max())
    disabled_diff = float((p1_disabled - fixed_parent).abs().max())
    results["G0.1_zero_init_equals_baseline"] = zero_init_diff == 0.0
    results["G0.1_residual_disabled_equals_baseline"] = disabled_diff == 0.0
    gates["G0.1_zero_init_max_abs_diff"] = zero_init_diff
    gates["G0.1_disabled_max_abs_diff"] = disabled_diff

    # G0.2 fixed centre pooling unchanged (bit-exact vs historical parent).
    results["G0.2_fixed_pooling_unchanged"] = (
        float((p1_disabled - fixed_parent).abs().max()) == 0.0
    )
    gates["G0.2_historical_source_sha256"] = _sha256_text(
        inspect.getsource(zpp.PatchPathModel._pool_pairs_to_centres)
    )
    gates["G0.2_configured_source_sha256"] = _read_json(
        RESULTS_DIR / "baseline_inventory.json"
    )["historical_pooling_source_sha256"]

    # G0.3 q dimension = 16.
    results["G0.3_q_dim_16"] = (
        int(model.pair_hidden) == 16
        and int(model.pair_encoder.layers[0].in_features) == 4 * 16
        and int(model.pair_encoder.layers[-2].out_features) == 16
    )
    gates["G0.3_q_dim"] = int(model.pair_encoder.layers[-2].out_features)

    # G0.4 fixed bucket summary = 33D.
    results["G0.4_fixed_bucket_33"] = (
        int(FIXED_BUCKET_WIDTH) == 33 and int(fixed_parent.shape[1]) == N_BUCKETS * 33
    )
    gates["G0.4_fixed_bucket_width"] = int(FIXED_BUCKET_WIDTH)

    # G0.5 five buckets -> 165D.
    results["G0.5_centre_context_165"] = int(model.center_context_width) == 165
    gates["G0.5_centre_context_width"] = int(model.center_context_width)

    # G0.6 final R = 302D.
    results["G0.6_R_302"] = int(model.unified_graph_width) == 302
    gates["G0.6_R_dimension"] = int(model.unified_graph_width)

    # G0.7 graph head unchanged.
    results["G0.7_graph_head_unchanged"] = bool(
        _state_hash(model.head.state_dict()) == _state_hash(base.head.state_dict())
    )
    gates["G0.7_head_input_width"] = int(model.head[0].in_features)

    # G0.8 topology branch unchanged.
    results["G0.8_topology_unchanged"] = bool(
        model.topology_encoder is not None
        and control.topology_encoder is not None
        and _state_hash(model.topology_encoder.state_dict())
        == _state_hash(base.topology_encoder.state_dict())
    )

    # G0.9 parameter total <= 103K.
    p1_total = _n_params(model)
    results["G0.9_params_under_ceiling"] = bool(p1_total <= PARAM_CEILING)
    gates["G0.9_total_params"] = int(p1_total)

    # G0.10 relation-ordering permutation invariance of the composer output.
    # Temporarily activate rho (its final layer is zero-initialised) so the
    # invariance test exercises the learned computation, not the degenerate
    # zero-residual state.
    rho_backup = model.rho[-1].weight.detach().clone()
    with torch.no_grad():
        model.rho[-1].weight.normal_(mean=0.0, std=0.1, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        ref = model._pool_pairs_to_centres(
            value, source, target, pair_bucket, n_centres
        )
        max_perm_diff = 0.0
        for _ in range(5):
            pv, ps, pt, pb = _permute_pairs(value, source, target, pair_bucket, gen)
            out = model._pool_pairs_to_centres(pv, ps, pt, pb, n_centres)
            max_perm_diff = max(max_perm_diff, float((out - ref).abs().max()))
        composer_active = float((ref - fixed_parent).abs().max())
    with torch.no_grad():
        model.rho[-1].weight.copy_(rho_backup)
    results["G0.10_permutation_invariant"] = max_perm_diff <= 1e-6
    results["G0.10b_composer_actually_active"] = composer_active > 1e-3
    gates["G0.10_max_perm_diff"] = max_perm_diff
    gates["G0.10_composer_active_max_diff"] = composer_active

    # G0.11 empty bucket residual exactly zero (with an active composer, so the
    # non-empty buckets do receive a nonzero learned residual).
    gen2 = torch.Generator().manual_seed(42)
    v2 = torch.randn(10, Q_DIM, generator=gen2)
    s2 = torch.randint(0, n_centres, (10,), generator=gen2)
    t2 = torch.randint(0, n_centres, (10,), generator=gen2)
    b2 = torch.randint(0, N_BUCKETS, (10,), generator=gen2)
    b2 = torch.where(b2 == 3, torch.full_like(b2, 4), b2)
    rho_backup2 = model.rho[-1].weight.detach().clone()
    with torch.no_grad():
        model.rho[-1].weight.normal_(mean=0.0, std=0.1, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        out_p1 = model._pool_pairs_to_centres(v2, s2, t2, b2, n_centres)
        out_fixed = zpp.PatchPathModel._pool_pairs_to_centres(
            base, v2, s2, t2, b2, n_centres
        )
    with torch.no_grad():
        model.rho[-1].weight.copy_(rho_backup2)
    bucket3_p1 = out_p1[:, 3 * 33 : 4 * 33]
    bucket3_fixed = out_fixed[:, 3 * 33 : 4 * 33]
    nonempty_changed = max(
        float((out_p1[:, b * 33 : (b + 1) * 33] - out_fixed[:, b * 33 : (b + 1) * 33]).abs().max())
        for b in range(N_BUCKETS)
        if b != 3
    )
    results["G0.11_empty_bucket_zero_residual"] = bool(
        float((bucket3_p1 - bucket3_fixed).abs().max()) == 0.0
        and float(bucket3_fixed.abs().max()) == 0.0
    )
    results["G0.11c_nonempty_buckets_changed"] = bool(nonempty_changed > 1e-3)
    gates["G0.11_nonempty_max_diff"] = float(nonempty_changed)
    psi_backup = control.psi[-1].weight.detach().clone()
    with torch.no_grad():
        control.psi[-1].weight.normal_(mean=0.0, std=0.1, generator=torch.Generator().manual_seed(13))
        control_active = control._pool_pairs_to_centres(v2, s2, t2, b2, n_centres)
    with torch.no_grad():
        control.psi[-1].weight.copy_(psi_backup)
    results["G0.11b_control_empty_bucket_zero"] = bool(
        float((control_active[:, 3 * 33 : 4 * 33] - bucket3_fixed).abs().max()) == 0.0
    )

    # G0.12 shared baseline init matches seed0 baseline (from initialization_match).
    init_path = RESULTS_DIR / "initialization_match.json"
    if init_path.exists():
        init = _read_json(init_path)
        results["G0.12_init_matches_seed0"] = bool(
            init["p1"]["exact_equal_to_seed0_baseline"]
            and init["control"]["exact_equal_to_seed0_baseline"]
        )
    else:
        init = initialization_match()
        results["G0.12_init_matches_seed0"] = bool(
            init["p1"]["exact_equal_to_seed0_baseline"]
            and init["control"]["exact_equal_to_seed0_baseline"]
        )

    # structural: exactly one q computation stage (no relation refresh).
    encode_source = inspect.getsource(zpp.PatchPathModel.encode)
    results["G0.13_single_q_stage"] = encode_source.count("self.pair_encoder(") == 1

    gates["results"] = results
    gates["passed"] = all(results.values())
    gates["official_test_loaded"] = False
    if report:
        _write_json(RESULTS_DIR / "integrity_gates.json", gates)
    return gates


def compute_audit() -> dict[str, Any]:
    _train, valid_data, _audit = build_encoded_records()

    base = build_baseline(seed=0)
    base.eval()
    model = build_p1(seed=0)
    model.eval()
    batch_graphs = list(valid_data[:128])
    loader = zpp._make_loader(batch_graphs, 128, False, 0)
    batch = next(iter(loader))
    n_pairs = int(batch.pair_index.shape[1])
    n_centres = int(batch.num_nodes)
    n_graphs = int(batch.num_graphs)

    with torch.no_grad():
        for _ in range(2):
            base(batch)
            model(batch)
        t0 = time.perf_counter()
        for _ in range(5):
            base(batch)
        base_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
        t0 = time.perf_counter()
        for _ in range(5):
            model(batch)
        p1_ms = (time.perf_counter() - t0) / 5.0 * 1000.0

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "batch_graphs": int(n_graphs),
        "batch_centres": int(n_centres),
        "batch_pairs": int(n_pairs),
        "pairs_per_molecule": float(n_pairs / max(n_graphs, 1)),
        "phi_calls_per_batch": int(n_pairs),
        "phi_calls_per_molecule": float(n_pairs / max(n_graphs, 1)),
        "forward_ms_baseline": float(base_ms),
        "forward_ms_p1": float(p1_ms),
        "forward_overhead_ms": float(p1_ms - base_ms),
        "forward_overhead_ratio": float(p1_ms / max(base_ms, 1e-9) - 1.0),
        "peak_rss_mb": float(peak_rss_mb),
        "preprocessing_unchanged": True,
        "note": (
            "phi is applied once per pair relation (n_pairs calls); the two "
            "endpoint copies reuse the same phi evaluation.  Compute is reported "
            "separately from parameter count."
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compute_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage1_p1_seed0() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records()
    summary = train_model(
        build_fn=lambda s: build_p1(
            s,
            typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
            parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        ),
        train_data=train_data,
        valid_data=valid_data,
        seed=0,
        tag="p1",
        grad_module_names=("phi", "rho"),
    )
    summary["baseline_v4_valid"] = float(V4_SEED0_VALID)
    summary["delta_arch0"] = float(V4_SEED0_VALID - summary["best_valid_mae"])
    summary["arch_gate"] = float(ARCH_GATE)
    summary["passes_arch_gate"] = bool(summary["delta_arch0"] >= ARCH_GATE)
    _write_json(RESULTS_DIR / "stage1_p1_seed0.json", summary)
    return summary


def composition_diagnostics() -> dict[str, Any]:
    """Descriptive composition diagnostics at the selected P1 seed0 checkpoint.

    Descriptive only: no target-tail mining, no error-quintile mining, no
    post-hoc bucket selection.
    """
    summary = _read_json(RESULTS_DIR / "stage1_p1_seed0.json")
    _train, valid_data, audit = build_encoded_records()
    model = build_p1(
        seed=0,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
    )
    model.load_state_dict(torch.load(summary["state_path"], map_location="cpu"))
    model.eval()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    model._reset_diagnostics()
    with torch.no_grad():
        for batch in loader:
            model.encode(batch)
    diag = model.diagnostics()
    diag["checkpoint"] = summary["state_path"]
    diag["split"] = "official-valid"
    diag["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "composition_diagnostics.json", diag)
    return diag


def branch_diagnostics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Inference-only branch-alive sanity for a saved P1 checkpoint."""
    _train, valid_data, audit = build_encoded_records()
    model = build_p1(
        seed=int(summary["seed"]),
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
    )
    model.load_state_dict(torch.load(summary["state_path"], map_location="cpu"))
    model.eval()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    max_shift = 0.0
    sum_shift = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            full = model(batch)
            model.composer_enabled = False
            zeroed = model(batch)
            model.composer_enabled = True
            diff = (full - zeroed).abs()
            max_shift = max(max_shift, float(diff.max()))
            sum_shift += float(diff.sum())
            count += int(diff.numel())
    diag = model.diagnostics() if hasattr(model, "diagnostics") else {}
    return {
        "residual_zero_max_abs_prediction_shift": float(max_shift),
        "residual_zero_mean_abs_prediction_shift": float(sum_shift / max(count, 1)),
        "branch_alive": bool(max_shift > 1e-3),
        "official_test_loaded": False,
    }


def _rare_le5_ratio(records: Sequence[Any], train_freq: Mapping[bytes, int]) -> np.ndarray:
    out = []
    for record in records:
        freqs = [int(train_freq.get(patch.typed_certificate, 0)) for patch in record.patches]
        out.append(float(np.mean(np.asarray(freqs) <= 5)) if freqs else 0.0)
    return np.asarray(out, dtype=np.float64)


def _v4_predictions(valid_data: Sequence[Data]) -> np.ndarray:
    base = build_baseline(seed=0)
    base.load_state_dict(torch.load(V4_SEED0_STATE, map_location="cpu"))
    base.eval()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(base(batch).view(-1).cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def common_input_bulk(candidate_path: Path | None = None) -> dict[str, Any]:
    candidate_path = candidate_path or (RESULTS_DIR / "stage1_p1_seed0.json")
    summary = _read_json(candidate_path)
    train_records, valid_records, _meta = extract_records()
    _train_data, valid_data, _audit = build_encoded_records()
    from collections import Counter

    train_freq: Counter[bytes] = Counter()
    for record in train_records:
        for patch in record.patches:
            train_freq[patch.typed_certificate] += 1
    train_rare = _rare_le5_ratio(train_records, train_freq)
    valid_rare = _rare_le5_ratio(valid_records, train_freq)

    targets = np.asarray(summary["valid_targets"], dtype=np.float64)
    cand_pred = np.asarray(summary["valid_predictions"], dtype=np.float64)
    v4_pred = _v4_predictions(valid_data)
    threshold = float(np.percentile(train_rare, 80.0))
    bulk = valid_rare < threshold
    v4_mae = float(np.mean(np.abs(targets - v4_pred)))
    cand_mae = float(np.mean(np.abs(targets - cand_pred)))
    bulk_v4 = float(np.mean(np.abs(targets[bulk] - v4_pred[bulk]))) if bulk.any() else float("nan")
    bulk_cand = (
        float(np.mean(np.abs(targets[bulk] - cand_pred[bulk]))) if bulk.any() else float("nan")
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
        "candidate_valid_mae": cand_mae,
        "valid_mae_diff_candidate_minus_v4": float(cand_mae - v4_mae),
        "bulk_v4_mae": bulk_v4,
        "bulk_candidate_mae": bulk_cand,
        "bulk_mae_diff_candidate_minus_v4": float(bulk_cand - bulk_v4),
        "bulk_gate": float(BULK_GATE),
        "bulk_safe": bool((bulk_cand - bulk_v4) <= BULK_GATE),
        "v4_reference_valid": float(V4_SEED0_VALID),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "common_input_bulk.json", payload)
    return payload


def stage1_decision() -> dict[str, Any]:
    summary = _read_json(RESULTS_DIR / "stage1_p1_seed0.json")
    diag = branch_diagnostics(summary)
    bulk = common_input_bulk()
    delta = float(summary["delta_arch0"])
    passed = bool(delta >= ARCH_GATE)
    # optimization-ambiguity check
    curve = list(csv.DictReader((CURVE_DIR / f"p1_seed0_curve.csv").open()))
    valid_series = [float(r["valid_mae"]) for r in curve]
    phi_grads = [float(r["phi_grad_norm"]) for r in curve]
    rho_grads = [float(r["rho_grad_norm"]) for r in curve]
    res_norms = [float(r["learned_residual_norm"]) for r in curve]
    nan_detected = bool(
        any(not math.isfinite(x) for x in valid_series + phi_grads + rho_grads + res_norms)
    )
    best_epoch = int(summary["best_epoch"])
    max_epochs = int(OPTIMIZED_PROTOCOL["max_epochs"])
    boundary = bool(best_epoch >= max_epochs - max(1, max_epochs // 10))
    branch_grad_alive = bool(max(phi_grads) > 0 and max(rho_grads) > 0)
    ambiguous = bool(nan_detected or not branch_grad_alive)
    decision = {
        "stage": "stage1_architecture_gate",
        "delta_arch0": delta,
        "threshold": float(ARCH_GATE),
        "branch_dependency": diag,
        "composer_branch_alive": bool(diag["branch_alive"]),
        "architecture_branch_alive": bool(diag["branch_alive"] and branch_grad_alive),
        "optimization_ambiguous": ambiguous,
        "nan_detected": nan_detected,
        "horizon_boundary_warning": boundary,
        "phi_grad_norm_max": float(max(phi_grads)),
        "rho_grad_norm_max": float(max(rho_grads)),
        "final_residual_norm": float(res_norms[-1]) if res_norms else 0.0,
        "common_input_bulk_safe": bool(bulk["bulk_safe"]),
        "common_input_bulk": bulk,
        "passed": passed,
        "classification": (
            "ARCHITECTURE PASS - proceed to Stage 2 fixed-moment capacity control"
            if passed
            else "P1 MINIMAL LEARNED COMPOSER - NO-GO (Case A stop)"
        ),
        "official_test_loaded": False,
    }
    if ambiguous:
        decision["classification"] = "OPTIMIZATION / IMPLEMENTATION AMBIGUOUS"
    _write_json(RESULTS_DIR / "stage1_decision.json", decision)
    return decision


def stage2_capacity_control_seed0() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records()
    summary = train_model(
        build_fn=lambda s: build_control(
            s,
            typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
            parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        ),
        train_data=train_data,
        valid_data=valid_data,
        seed=0,
        tag="control",
        grad_module_names=("psi",),
    )
    p1 = _read_json(RESULTS_DIR / "stage1_p1_seed0.json")
    delta_prepool = float(summary["best_valid_mae"] - p1["best_valid_mae"])
    summary["p1_valid"] = float(p1["best_valid_mae"])
    summary["control_valid"] = float(summary["best_valid_mae"])
    summary["delta_prepool0"] = delta_prepool
    summary["mech_gate"] = float(MECH_GATE)
    summary["passes_mech_gate"] = bool(delta_prepool >= MECH_GATE)
    _write_json(RESULTS_DIR / "stage2_capacity_control_seed0.json", summary)
    return summary


def stage2_decision() -> dict[str, Any]:
    p1 = _read_json(RESULTS_DIR / "stage1_p1_seed0.json")
    control = _read_json(RESULTS_DIR / "stage2_capacity_control_seed0.json")
    bulk_p1 = _read_json(RESULTS_DIR / "stage1_decision.json")["common_input_bulk"]
    delta_arch0 = float(p1["delta_arch0"])
    delta_prepool0 = float(control["delta_prepool0"])
    bulk_ctrl = common_input_bulk(RESULTS_DIR / "stage2_capacity_control_seed0.json")
    branch_alive = bool(_read_json(RESULTS_DIR / "stage1_decision.json")["architecture_branch_alive"])
    passed = bool(
        delta_arch0 >= ARCH_GATE
        and delta_prepool0 >= MECH_GATE
        and bulk_p1["bulk_safe"]
        and bulk_ctrl["bulk_safe"]
        and branch_alive
    )
    if control["best_valid_mae"] < p1["best_valid_mae"]:
        case = "F - PRE-POOL COMPOSITION HYPOTHESIS NO-GO (control beats P1)"
    elif passed:
        case = "C - SEED0 LEARNED PRE-POOL COMPOSITION SIGNAL"
    else:
        case = "B - MIDDLE-CAPACITY GO; PRE-POOL COMPOSITION NOT ISOLATED"
    decision = {
        "stage": "stage2_mechanism_gate_seed0",
        "delta_arch0": delta_arch0,
        "delta_prepool0": delta_prepool0,
        "mech_threshold": float(MECH_GATE),
        "control_valid": float(control["best_valid_mae"]),
        "p1_valid": float(p1["best_valid_mae"]),
        "bulk_p1_safe": bool(bulk_p1["bulk_safe"]),
        "bulk_control": bulk_ctrl,
        "branch_alive": branch_alive,
        "case": case,
        "passed": passed,
        "classification": case,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage2_mechanism_decision.json", decision)
    return decision


def stage3_p1_seed1() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records()
    summary = train_model(
        build_fn=lambda s: build_p1(
            s,
            typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
            parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        ),
        train_data=train_data,
        valid_data=valid_data,
        seed=1,
        tag="p1",
        grad_module_names=("phi", "rho"),
    )
    summary["baseline_v4_valid"] = float(V4_SEED1_VALID)
    summary["delta_arch1"] = float(V4_SEED1_VALID - summary["best_valid_mae"])
    summary["arch_gate_seed1"] = float(ARCH_GATE_SEED1)
    summary["passes_arch_gate_seed1"] = bool(summary["delta_arch1"] >= ARCH_GATE_SEED1)
    _write_json(RESULTS_DIR / "stage3_p1_seed1.json", summary)
    return summary


def stage3a_decision() -> dict[str, Any]:
    s0 = _read_json(RESULTS_DIR / "stage1_p1_seed0.json")
    s1 = _read_json(RESULTS_DIR / "stage3_p1_seed1.json")
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
            else "REPLICATION INCONCLUSIVE (Case E stop)"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage3a_replication_decision.json", decision)
    return decision


def stage3_capacity_control_seed1() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records()
    summary = train_model(
        build_fn=lambda s: build_control(
            s,
            typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
            parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        ),
        train_data=train_data,
        valid_data=valid_data,
        seed=1,
        tag="control",
        grad_module_names=("psi",),
    )
    p1 = _read_json(RESULTS_DIR / "stage3_p1_seed1.json")
    delta_prepool1 = float(summary["best_valid_mae"] - p1["best_valid_mae"])
    summary["p1_valid"] = float(p1["best_valid_mae"])
    summary["delta_prepool1"] = delta_prepool1
    _write_json(RESULTS_DIR / "stage3_capacity_control_seed1.json", summary)
    return summary


def stage3b_decision() -> dict[str, Any]:
    c0 = _read_json(RESULTS_DIR / "stage2_capacity_control_seed0.json")
    c1 = _read_json(RESULTS_DIR / "stage3_capacity_control_seed1.json")
    d0 = float(c0["delta_prepool0"])
    d1 = float(c1["delta_prepool1"])
    mean = (d0 + d1) / 2.0
    passed = bool(d1 > MECH_GATE_SEED1 and mean >= MECH_MEAN_GATE)
    decision = {
        "stage": "stage3b_mechanism_replication",
        "delta_prepool0": d0,
        "delta_prepool1": d1,
        "mean": mean,
        "passed": passed,
        "classification": (
            "LEARNED RELATION-TO-CENTRE COMPOSITION - REPLICATED GO (Case D)"
            if passed
            else "MECHANISM REPLICATION FAILED"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage3b_mechanism_replication.json", decision)
    return decision


def final_decision() -> dict[str, Any]:
    stage1 = _read_json(RESULTS_DIR / "stage1_decision.json")
    payload: dict[str, Any] = {
        "decision_case": None,
        "official_test_loaded": False,
        "notes": [],
    }
    if stage1.get("optimization_ambiguous"):
        payload["decision_case"] = "F-ambiguous"
        payload["decision"] = "OPTIMIZATION / IMPLEMENTATION AMBIGUOUS"
        payload["notes"].append("no scientific NO-GO registered")
    elif not stage1["passed"]:
        payload["decision_case"] = "A"
        payload["decision"] = "P1 MINIMAL LEARNED COMPOSER - NO-GO"
        payload["notes"].append(
            "delta_arch0 < +0.004; control and seed1 NOT purchased"
        )
    else:
        stage2 = _read_json(RESULTS_DIR / "stage2_mechanism_decision.json")
        if stage2["case"].startswith("F"):
            payload["decision_case"] = "F"
            payload["decision"] = "PRE-POOL COMPOSITION HYPOTHESIS NO-GO"
        elif not stage2["passed"]:
            payload["decision_case"] = "B"
            payload["decision"] = (
                "MIDDLE-CAPACITY GO - PRE-POOL COMPOSITION NOT ISOLATED"
            )
        else:
            stage3a_path = RESULTS_DIR / "stage3a_replication_decision.json"
            if not stage3a_path.exists():
                payload["decision_case"] = "C"
                payload["decision"] = "SEED0 LEARNED PRE-POOL COMPOSITION SIGNAL"
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
                            "SEED0 signal; seed1 architecture replicated; "
                            "mechanism replication pending"
                        )
                    else:
                        stage3b = _read_json(stage3b_path)
                        if stage3b["passed"]:
                            payload["decision_case"] = "D"
                            payload["decision"] = (
                                "LEARNED RELATION-TO-CENTRE COMPOSITION - REPLICATED GO"
                            )
                        else:
                            payload["decision_case"] = "B"
                            payload["decision"] = (
                                "ARCHITECTURE GO - MECHANISM NOT REPLICATED"
                            )
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


def two_seed_summary() -> dict[str, Any]:
    rows = []
    for path, label in (
        (RESULTS_DIR / "stage1_p1_seed0.json", "p1_seed0"),
        (RESULTS_DIR / "stage2_capacity_control_seed0.json", "control_seed0"),
        (RESULTS_DIR / "stage3_p1_seed1.json", "p1_seed1"),
        (RESULTS_DIR / "stage3_capacity_control_seed1.json", "control_seed1"),
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
    _write_csv(
        RESULTS_DIR / "final_two_seed_summary.csv",
        rows,
        ("model", "seed", "valid_mae", "best_epoch", "params"),
    )
    return {"rows": rows}


def answers_q1_q20() -> dict[str, Any]:
    lock = _read_json(RESULTS_DIR / "architecture_lock.json")
    param = _read_json(RESULTS_DIR / "parameter_audit.json")
    s1 = _read_json(RESULTS_DIR / "stage1_p1_seed0.json")
    d1 = _read_json(RESULTS_DIR / "stage1_decision.json")
    final = _read_json(RESULTS_DIR / "final_decision.json")
    control = (
        _read_json(RESULTS_DIR / "stage2_capacity_control_seed0.json")
        if (RESULTS_DIR / "stage2_capacity_control_seed0.json").exists()
        else None
    )
    stage2 = (
        _read_json(RESULTS_DIR / "stage2_mechanism_decision.json")
        if (RESULTS_DIR / "stage2_mechanism_decision.json").exists()
        else None
    )
    answers = {
        "Q1": lock["fixed_bucket_summary_definition"],
        "Q2": lock["q_dim"],
        "Q3": lock["number_of_buckets"],
        "Q4": lock["phi_architecture"],
        "Q5": lock["rho_architecture"],
        "Q6": lock["residual_or_replace"],
        "Q7": param["p1_total_params"],
        "Q8": param["p1_added_params"],
        "Q9": lock["R_dimension"],
        "Q10": {
            "best_valid_mae": s1["best_valid_mae"],
            "best_epoch": s1["best_epoch"],
        },
        "Q11": s1["delta_arch0"],
        "Q12": bool(s1["passes_arch_gate"]),
        "Q13": {
            "branch_alive": d1["architecture_branch_alive"],
            "residual_zero_max_abs_prediction_shift": d1["branch_dependency"][
                "residual_zero_max_abs_prediction_shift"
            ],
            "phi_grad_norm_max": d1["phi_grad_norm_max"],
            "rho_grad_norm_max": d1["rho_grad_norm_max"],
        },
        "Q14": d1["common_input_bulk_safe"],
        "Q15": control is not None,
        "Q16": None if control is None else control["best_valid_mae"],
        "Q17": None if stage2 is None else stage2["delta_prepool0"],
        "Q18": bool(
            stage2 is not None
            and stage2["passed"]
        ),
        "Q19": None,
        "Q20": final["decision_case"],
    }
    _write_json(RESULTS_DIR / "answers_q1_q20.json", answers)
    return answers


def make_figures() -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return {"figures": [], "error": repr(exc)}
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.93, "compact-v4 learned relation-to-centre composer (P1)", ha="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.78, "pairs q_ij (16D, computed once)", ha="center", fontsize=9)
    ax.text(0.5, 0.64, "phi: 16 -> 24 -> 24   (shared across all buckets)", ha="center", fontsize=9)
    ax.text(0.5, 0.52, "per-(centre,bucket) mean  z_ib (24D)", ha="center", fontsize=9)
    ax.text(0.5, 0.40, "u_ib = [z_ib ; log1p(n_ib)] (25D)", ha="center", fontsize=9)
    ax.text(0.5, 0.28, "rho: 25 -> 33 -> 33  (zero-init final)", ha="center", fontsize=9)
    ax.text(0.5, 0.16, "a^P1 = a^fixed + r_ib   (165D, centre update input unchanged)", ha="center", fontsize=9)
    ax.text(0.5, 0.05, "R = 302D -> unchanged head -> MAE", ha="center", fontsize=9)
    path = FIGURE_DIR / "figure1_composer_architecture.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))

    labels = ["v4 seed0"]
    values = [V4_SEED0_VALID]
    for path_, label in (
        (RESULTS_DIR / "stage1_p1_seed0.json", "P1 seed0"),
        (RESULTS_DIR / "stage2_capacity_control_seed0.json", "control seed0"),
        (RESULTS_DIR / "stage3_p1_seed1.json", "P1 seed1"),
        (RESULTS_DIR / "stage3_capacity_control_seed1.json", "control seed1"),
    ):
        if path_.exists():
            labels.append(label)
            values.append(float(_read_json(path_)["best_valid_mae"]))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(len(labels)), values, color="#4c72b0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("official-valid MAE")
    ax.set_title("P1 learned centre composer: valid MAE", fontsize=10)
    path = FIGURE_DIR / "figure2_valid_mae.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))
    return {"figures": created}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_stage0() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    architecture_lock()
    baseline_inventory()
    parameter_audit()
    initialization_match()
    gate = integrity_gates()
    if not gate["passed"]:
        raise RuntimeError(f"integrity gates failed: {gate['results']}")
    compute_audit()
    make_figures()
    print("Stage 0 complete.", flush=True)
    return gate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "architecture_lock",
            "baseline",
            "param_audit",
            "init_match",
            "integrity",
            "compute_audit",
            "extract",
            "stage0",
            "stage1_p1_seed0",
            "stage1_decision",
            "composition_diagnostics",
            "bulk",
            "stage2_capacity_control_seed0",
            "stage2_decision",
            "stage3_p1_seed1",
            "stage3a_decision",
            "stage3_capacity_control_seed1",
            "stage3b_decision",
            "answers",
            "final",
            "two_seed_summary",
            "figures",
            "all",
        ],
    )
    args = parser.parse_args(argv)
    stage = args.stage
    _configure_determinism()
    if stage == "architecture_lock":
        print(json.dumps(architecture_lock(), indent=2))
    elif stage == "baseline":
        print(json.dumps(baseline_inventory(), indent=2))
    elif stage == "param_audit":
        print(json.dumps(parameter_audit(), indent=2))
    elif stage == "init_match":
        print(json.dumps(initialization_match(), indent=2))
    elif stage == "integrity":
        print(json.dumps(integrity_gates(), indent=2))
    elif stage == "compute_audit":
        print(json.dumps(compute_audit(), indent=2))
    elif stage == "extract":
        print(extract_records()[2])
    elif stage == "stage0":
        run_stage0()
    elif stage == "stage1_p1_seed0":
        print(stage1_p1_seed0()["delta_arch0"])
    elif stage == "stage1_decision":
        print(json.dumps(stage1_decision(), indent=2))
    elif stage == "composition_diagnostics":
        print(json.dumps(composition_diagnostics(), indent=2))
    elif stage == "bulk":
        print(json.dumps(common_input_bulk(), indent=2))
    elif stage == "stage2_capacity_control_seed0":
        print(stage2_capacity_control_seed0())
    elif stage == "stage2_decision":
        print(json.dumps(stage2_decision(), indent=2))
    elif stage == "stage3_p1_seed1":
        print(stage3_p1_seed1())
    elif stage == "stage3a_decision":
        print(json.dumps(stage3a_decision(), indent=2))
    elif stage == "stage3_capacity_control_seed1":
        print(stage3_capacity_control_seed1())
    elif stage == "stage3b_decision":
        print(json.dumps(stage3b_decision(), indent=2))
    elif stage == "answers":
        print(json.dumps(answers_q1_q20(), indent=2))
    elif stage == "final":
        print(json.dumps(final_decision(), indent=2))
    elif stage == "two_seed_summary":
        print(two_seed_summary())
    elif stage == "figures":
        print(make_figures())
    else:
        run_stage0()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
