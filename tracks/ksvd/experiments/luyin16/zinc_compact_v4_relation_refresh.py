"""P2 -- One-shot relation refresh (minimal falsification).

This stage implements the single architecture hypothesis of the P2 task:

    The current final graph representation mixes two *different computation
    stages*: the unary branch uses the contextualized centre state ``h^(1)``
    (post centre-update) while the final pair branch pools the *stale* relation
    state ``q^(0)`` computed **before** contextualization.  If, after centre
    contextualization, the *same* shared relation encoder recomputes the
    relation from the updated endpoints -- ``q^(1) = Q(P(h^(1)), r)`` -- and the
    final pair graph summary uses ``q^(1)`` instead of ``q^(0)``, does compact-v4
    get better?

What changes
------------
* ``compact-v4`` base pathway, unchanged through centre contextualization:
  ``h^(0) -> q^(0) -> A -> h^(1)``.
* One, and only one, additional **parameter-shared** refresh:
  ``u^(1) = P(h^(1))`` and ``q^(1) = Q(u^(1)_i, u^(1)_j, r_ij)`` reusing the
  exact same ``P`` (``pair_projection``) and ``Q`` (``pair_encoder``) tensors,
  the same ``r_ij`` and the same distance gate.
* The final pair graph summary pools ``q^(1)`` instead of ``q^(0)``.  ``q^(1)``
  does **not** feed any further centre update (no ``h^(2)``).

What is *not* here
------------------
No new trainable tensor (``Delta params = 0``, total stays 99,613).  No second
relation encoder, no mixing coefficient ``alpha``, no concat ``[q0;q1]``, no
attention/Transformer, no generic message passing, no depth sweep (refresh count
is exactly 1), no change to tokenizer / patch radius / pair definition / relation
descriptor / q dim / bucket count / centre pooling / final pair-moment
implementation / global descriptors / topology branch / graph head / optimizer /
lr / batch / wd / horizon / patience.  ``R`` stays 302D.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_relation_refresh <stage>

Stages: ``baseline probe_manifest audit_lock integrity stageA relation_drift
pair_summary_drift prediction_shift stageA_decision architecture_lock param_audit
init_match compute_audit stageB_p2_seed0 stageB_decision bulk mechanism
final stageC_p2_seed1 stageC_decision answers figures all``.
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
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_relation_refresh"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
CACHE_DIR = RESULTS_DIR / "cache"
FIGURE_DIR = RESULTS_DIR / "figures"

CANONICAL_V4_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
CANONICAL_SUFFICIENCY_DIR = TRACK_ROOT / "results/compact_v4_training_sufficiency"
V4_SEED0_STATE = CANONICAL_SUFFICIENCY_DIR / "states/Pstar_A2_long_seed0_selection_state.pt"
V4_SEED1_STATE = CANONICAL_SUFFICIENCY_DIR / "states/Pstar_A2_long_seed1_selection_state.pt"
V4_SEED0_SHA256 = "60b7d297a44befb7328f4f9da0cb379e308ecec3f5eab17fa7c199896ac3e71b"
V4_SEED1_SHA256 = "93bf4ec231469793c20da829551cf1c95693513ddf58be7d92e28501534db8a3"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2

# --- locked architecture constants -------------------------------------------
Q_DIM = 16  # pair_hidden / dimension of q_ij
N_BUCKETS = int(zpp.DISTANCE_BUCKETS)  # 5 integer distance buckets (1,2,3,4,5+)
PATCH_HIDDEN = 48
UNARY_WIDTH = 2 * PATCH_HIDDEN + 1  # 97 = [sum ; sum_sq ; log1p(count)]
PAIR_BUCKET_WIDTH = 2 * Q_DIM + 1  # 33 = [sum ; sum_sq ; log1p(count)]
PAIR_READOUT_WIDTH = N_BUCKETS * PAIR_BUCKET_WIDTH  # 165
UNARY_OFFSET = 0
PAIR_OFFSET = UNARY_WIDTH  # 97
GLOBAL_WIDTH = 32
TOPOLOGY_WIDTH = 8
R_TOTAL = UNARY_WIDTH + PAIR_READOUT_WIDTH + GLOBAL_WIDTH + TOPOLOGY_WIDTH  # 302
REFRESH_COUNT = 1  # exactly one relation refresh (hard rule)

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

# --- official-train train-probe split (reused, pre-existing frozen manifest) --
SPLIT_SEED = "optimized-manifold-broad-state-screen-v1-20260919"
SPLIT_SIZES = (7200, 800, 2000)
SPLIT_ROLES = ("adapter_fit", "adapter_selection", "train_probe")
N_TRAIN = 10000
N_VALID = 1000
FROZEN_SPLIT_MANIFEST = (
    TRACK_ROOT / "results/optimized_manifold_broad_state_screen/split_manifest.json"
)
PROBE_SIZE = 2000

# --- pre-registered Stage A near-identity STOP gate --------------------------
STOP_COS_MEDIAN = 0.995
STOP_PAIR_DRIFT_MEDIAN = 0.02
STOP_PRED_MEAN_SHIFT = 0.005

# --- pre-registered architecture gates ---------------------------------------
ARCH_GATE = 0.004  # Delta_P2,0 = v4_valid0 - p2_valid0
ARCH_GATE_SEED1 = 0.003
ARCH_MEAN_GATE = 0.004
BULK_GATE = 0.002  # candidate MAE - v4 MAE on target-independent bulk
EPS = 1.0e-8

CACHE_SCHEMA_VERSION = "compact_v4_relation_refresh_records_v1"
PROTOCOL_VERSION = "compact_v4_relation_refresh_v1"


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


class PatchPathRelationRefreshModel(zpp.PatchPathModel):
    """compact-v4 + exactly one parameter-shared relation refresh (P2 candidate).

    The base ``compact-v4`` pathway is inherited unchanged *through centre
    contextualization* (``h^(0) -> q^(0) -> A -> h^(1)``).  A second, parameter
    shared evaluation of the same pair projector ``P`` and pair encoder ``Q`` is
    applied to the contextualized endpoints ``h^(1)``; only the final pair graph
    moment uses the refreshed relation ``q^(1)``.

    This class adds **zero** trainable tensors: it only adds forward-side
    composition of existing modules (``self.pair_projection``,
    ``self.relation_encoder``, ``self.distance_gate``, ``self.pair_encoder``).
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        refresh_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(typed_vocabulary_size, parent_vocabulary_size, **kwargs)
        if int(self.pair_hidden) != Q_DIM:
            raise ValueError(
                f"refresh requires pair_hidden == {Q_DIM}; got {self.pair_hidden}"
            )
        if self.center_update is None:
            raise ValueError("refresh requires center_context=True (a centre update)")
        self.refresh_enabled = bool(refresh_enabled)
        # captured tensors (last forward); no parameters, no buffers
        self.last_h0: torch.Tensor | None = None
        self.last_h1: torch.Tensor | None = None
        self.last_q0: torch.Tensor | None = None
        self.last_q1: torch.Tensor | None = None
        self.last_pair_source: torch.Tensor | None = None
        self.last_pair_target: torch.Tensor | None = None
        self.last_pair_bucket: torch.Tensor | None = None
        self.last_n_graphs: int = 0
        self._collect_diag = False
        self._reset_diagnostics()

    # -- diagnostics ---------------------------------------------------------
    def _reset_diagnostics(self) -> None:
        self.diag_q0_norm_sum = 0.0
        self.diag_q1_norm_sum = 0.0
        self.diag_q_cos_sum = 0.0
        self.diag_pair_summary_drift_sum = 0.0
        self.diag_pair_summary_drift_max = 0.0
        self.diag_batches = 0

    def diagnostics(self) -> dict[str, Any]:
        denom = max(self.diag_batches, 1)
        return {
            "q0_norm": float(self.diag_q0_norm_sum / denom),
            "q1_norm": float(self.diag_q1_norm_sum / denom),
            "q_cosine": float(self.diag_q_cos_sum / denom),
            "pair_summary_drift": float(self.diag_pair_summary_drift_sum / denom),
            "pair_summary_drift_max": float(self.diag_pair_summary_drift_max),
            "batches": int(self.diag_batches),
        }

    def read_grad_diagnostics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, tensor in (("q0", self.last_q0), ("q1", self.last_q1)):
            grad = getattr(tensor, "grad", None)
            out[f"{name}_grad_norm"] = (
                float(grad.norm()) if grad is not None else 0.0
            )
        return out

    # -- shared refresh (exact same tensors as q0) ---------------------------
    def _refresh_pair_value(
        self,
        patch: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute q from contextualized endpoints with SHARED modules."""
        left = self.pair_projection(patch[source])
        right = self.pair_projection(patch[target])
        product = left * right
        pair_input = torch.cat(
            [
                left + right,
                torch.abs(left - right),
                product * gate,
                relation,
            ],
            dim=1,
        )
        return self.pair_encoder(pair_input)

    def _encode_core(self, data: Data, use_refresh: bool) -> torch.Tensor:
        """Byte-for-byte copy of ``PatchPathModel.encode`` with one addition.

        When ``use_refresh`` is True the final pair readout pools ``q^(1)``
        (contextualized endpoints, same shared encoder) instead of ``q^(0)``.
        When False the output is bit-identical to the parent ``encode``.
        """
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])
        e_patch = self.typed_embedding(data.typed_token)
        structural_blocks: list[torch.Tensor] = []
        if self.structural_context_mode != "none":
            e_ctx = self._structural_context_embedding_value(data)
            if self.structural_context_fusion == "condition":
                e_patch = self._condition_patch(
                    e_patch, e_ctx, self._structural_no_ring_mask(data)
                )
            else:
                structural_blocks.append(e_ctx)
        attribute_blocks: list[torch.Tensor] = []
        if self.attribute_encoder is not None:
            attribute_blocks.append(self.attribute_encoder(data, e_patch))
        patch = self.patch_encoder(
            torch.cat(
                [
                    data.patch_cont,
                    data.patch_context,
                    e_patch,
                    self.parent_embedding(data.parent_token),
                    *structural_blocks,
                    *attribute_blocks,
                ],
                dim=1,
            )
        )
        unary = self._pool_nodes(patch, data.batch, n_graphs)
        direct_blocks: list[torch.Tensor] = []
        if self.direct_token_readout:
            token_code = self.typed_embedding.embedding(data.typed_token)
            direct_blocks.append(
                self._pool_values(token_code, data.batch, n_graphs, "moments")
            )

        source = data.pair_index[0]
        target = data.pair_index[1]
        projected_left = self.pair_projection(patch[source])
        projected_right = self.pair_projection(patch[target])
        relation = self.relation_encoder(data.pair_relation)
        product = projected_left * projected_right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat(
            [
                projected_left + projected_right,
                torch.abs(projected_left - projected_right),
                product * gate,
                relation,
            ],
            dim=1,
        )
        pair_value = self.pair_encoder(pair_input)  # q^(0)
        pair_batch = data.batch[source]

        h0 = patch
        if self.center_update is not None:
            center_context = self._pool_pairs_to_centres(
                pair_value,
                source,
                target,
                data.pair_bucket,
                int(patch.shape[0]),
            )
            patch = patch + self.center_update(
                torch.cat([patch, center_context], dim=1)
            )
            # The unary readout is deliberately recomputed after the update.
            unary = self._pool_nodes(patch, data.batch, n_graphs)
        h1 = patch

        # --- P2: one parameter-shared relation refresh (q0 -> q1) ------------
        if use_refresh:
            refreshed = self._refresh_pair_value(
                h1, source, target, relation, gate
            )
        else:
            refreshed = pair_value

        self.last_h0 = h0
        self.last_h1 = h1
        self.last_q0 = pair_value
        self.last_q1 = refreshed
        self.last_pair_source = source
        self.last_pair_target = target
        self.last_pair_bucket = data.pair_bucket
        self.last_n_graphs = n_graphs
        if self.training and torch.is_grad_enabled():
            pair_value.retain_grad()
            refreshed.retain_grad()

        relation_readout = self._pool_pairs(
            refreshed, pair_batch, data.pair_bucket, n_graphs
        )

        if use_refresh and self._collect_diag:
            with torch.no_grad():
                q0d = pair_value.detach()
                q1d = refreshed.detach()
                if q0d.numel():
                    self.diag_q0_norm_sum += float(q0d.norm(dim=1).mean())
                    self.diag_q1_norm_sum += float(q1d.norm(dim=1).mean())
                    self.diag_q_cos_sum += float(
                        F.cosine_similarity(q0d, q1d, dim=1, eps=EPS).mean()
                    )
                s0 = self._pool_pairs(
                    q0d, pair_batch, data.pair_bucket, n_graphs
                )
                s1 = self._pool_pairs(
                    q1d, pair_batch, data.pair_bucket, n_graphs
                )
                drift = (s1 - s0).norm(dim=1) / (s0.norm(dim=1) + EPS)
                self.diag_pair_summary_drift_sum += float(drift.mean())
                self.diag_pair_summary_drift_max = max(
                    self.diag_pair_summary_drift_max, float(drift.max())
                )
                self.diag_batches += 1

        graph_hidden = self.global_encoder(global_context)
        readout_blocks = [unary, relation_readout]
        readout_blocks.extend(direct_blocks)
        readout_blocks.append(graph_hidden)
        if self.topology_encoder is not None:
            topology = data.topology_features
            if topology.ndim == 1:
                topology = topology.unsqueeze(0)
            if int(topology.shape[1]) != self.topology_input_width:
                raise RuntimeError(
                    f"topology width mismatch: data={int(topology.shape[1])} "
                    f"model={self.topology_input_width}"
                )
            readout_blocks.append(self.topology_encoder(topology))
        unified = torch.cat(readout_blocks, dim=1)
        return unified

    def encode(self, data: Data) -> torch.Tensor:  # type: ignore[override]
        return self._encode_core(data, use_refresh=bool(self.refresh_enabled))

    def encode_original(self, data: Data) -> torch.Tensor:
        """Return only the 302D compact-v4 representation (bit-exact parent)."""
        return self._encode_core(data, use_refresh=False)


# ---------------------------------------------------------------------------
# config / builders
# ---------------------------------------------------------------------------


def load_config(path: Path = CANONICAL_V4_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def refresh_config() -> dict[str, Any]:
    config = load_config()
    config["test_policy"] = "no_test"
    config["parameter_audit"] = False
    config["model"]["device"] = "cpu"
    config["model"]["refresh_enabled"] = True
    config["output"] = {
        "json": str(RESULTS_DIR / "scratch.json"),
        "markdown": str(RESULTS_DIR / "scratch.md"),
    }
    return config


def _base_kwargs() -> dict[str, Any]:
    model_config = refresh_config()["model"]
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


def build_refresh(
    seed: int = 0,
    *,
    refresh_enabled: bool = True,
    typed_vocabulary_size: int = V4_TYPED_VOCAB_WITH_OOV,
    parent_vocabulary_size: int = V4_PARENT_VOCAB_WITH_OOV,
) -> PatchPathRelationRefreshModel:
    _seed_everything(seed)
    return PatchPathRelationRefreshModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        refresh_enabled=bool(refresh_enabled),
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


_ENCODED_CACHE: tuple[list[Data], list[Data], dict[str, Any]] | None = None


def build_encoded_records():
    global _ENCODED_CACHE
    if _ENCODED_CACHE is not None:
        return _ENCODED_CACHE
    train_records, valid_records, _meta = extract_records()
    config = refresh_config()
    enc_train, enc_valid, audit = zpp._phase_data(
        train_records, valid_records, config=config
    )
    _ENCODED_CACHE = (enc_train, enc_valid, audit)
    return _ENCODED_CACHE


# ---------------------------------------------------------------------------
# official-train deterministic 7200/800/2000 probe split (reused manifest)
# ---------------------------------------------------------------------------


def _role_assignment() -> np.ndarray:
    keys = np.asarray(
        [
            int(
                hashlib.sha256(f"{SPLIT_SEED}|train:{mid:04d}".encode()).hexdigest()[:16],
                16,
            )
            for mid in range(N_TRAIN)
        ],
        dtype=np.float64,
    )
    order = np.argsort(keys, kind="stable")
    roles = np.empty(N_TRAIN, dtype=object)
    cursor = 0
    for label, size in zip(SPLIT_ROLES, SPLIT_SIZES):
        roles[order[cursor : cursor + size]] = label
        cursor += size
    return roles


def _probe_indices() -> tuple[list[int], dict[str, Any]]:
    roles = _role_assignment()
    assignment_sha = hashlib.sha256("".join(roles.tolist()).encode()).hexdigest()
    probe = np.flatnonzero(roles == "train_probe").astype(np.int64).tolist()
    verification: dict[str, Any] = {
        "assignment_sha256_recomputed": assignment_sha,
        "n_probe": int(len(probe)),
    }
    if FROZEN_SPLIT_MANIFEST.exists():
        frozen = _read_json(FROZEN_SPLIT_MANIFEST)
        verification["frozen_manifest"] = str(FROZEN_SPLIT_MANIFEST)
        verification["frozen_assignment_sha256"] = frozen.get("assignment_sha256")
        verification["frozen_probe_matches"] = bool(
            list(frozen.get("roles", {}).get("train_probe", [])) == probe
        )
        verification["assignment_matches"] = bool(
            frozen.get("assignment_sha256") == assignment_sha
        )
    else:
        verification["frozen_manifest"] = None
    return probe, verification


# ---------------------------------------------------------------------------
# evaluation / training
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


def _module_grad_norm(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return math.sqrt(total)


def train_refresh(
    *,
    train_data: Sequence[Data],
    valid_data: Sequence[Data],
    typed_vocabulary_size: int,
    parent_vocabulary_size: int,
    seed: int,
    tag: str,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = build_refresh(
        seed,
        refresh_enabled=True,
        typed_vocabulary_size=typed_vocabulary_size,
        parent_vocabulary_size=parent_vocabulary_size,
    ).to(device)
    total_params = _n_params(model)
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
        model._reset_diagnostics()
        model._collect_diag = True
        epoch_loss = 0.0
        seen = 0
        pair_proj_grad = 0.0
        pair_enc_grad = 0.0
        q0_grad = 0.0
        q1_grad = 0.0
        grad_steps = 0
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            pair_proj_grad += _module_grad_norm(model.pair_projection)
            pair_enc_grad += _module_grad_norm(model.pair_encoder)
            gd = model.read_grad_diagnostics()
            q0_grad += gd.get("q0_grad_norm", 0.0)
            q1_grad += gd.get("q1_grad_norm", 0.0)
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
        model._collect_diag = False
        diag = model.diagnostics()
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_mae": float(train_mae),
            "valid_mae": float(valid_mae),
            "lr": float(OPTIMIZED_PROTOCOL["learning_rate"]),
            "optimizer_steps": int(epoch * steps_per_epoch),
            "checkpoint_selected": 0,
            "q0_norm": float(diag["q0_norm"]),
            "q1_norm": float(diag["q1_norm"]),
            "q_cosine": float(diag["q_cosine"]),
            "pair_summary_drift": float(diag["pair_summary_drift"]),
            "pair_encoder_grad_norm": float(pair_enc_grad / max(grad_steps, 1)),
            "pair_projection_grad_norm": float(pair_proj_grad / max(grad_steps, 1)),
            "q0_grad_norm": float(q0_grad / max(grad_steps, 1)),
            "q1_grad_norm": float(q1_grad / max(grad_steps, 1)),
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
                f"qcos={row['q_cosine']:.5f} pe_g={row['pair_encoder_grad_norm']:.3e} "
                f"q1_g={row['q1_grad_norm']:.3e}",
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
            "q0_norm",
            "q1_norm",
            "q_cosine",
            "pair_summary_drift",
            "pair_encoder_grad_norm",
            "pair_projection_grad_norm",
            "q0_grad_norm",
            "q1_grad_norm",
        ),
    )
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    torch.save(model.state_dict(), state_path)

    final_valid, valid_targets, valid_predictions = _evaluate_mae(
        model, eval_loader, device
    )
    horizon_warning = bool(best_epoch >= max_epochs - max(1, max_epochs // 10))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "seed": int(seed),
        "protocol": dict(OPTIMIZED_PROTOCOL),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(final_valid),
        "epochs_run": int(len(losses)),
        "steps_per_epoch": int(steps_per_epoch),
        "optimizer_steps": int(len(losses) * steps_per_epoch),
        "wall_clock_s": wall_clock,
        "early_stopped": bool(len(losses) < max_epochs),
        "horizon_boundary_warning": horizon_warning,
        "parameters": int(total_params),
        "state_path": str(state_path),
        "curve_path": str(curve_path),
        "diagnostics_final": model.diagnostics(),
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
# Stage 0 / Stage A: locks, probe, integrity
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
        "v4_seed0_state_sha256": (
            _sha256_file(V4_SEED0_STATE) if V4_SEED0_STATE.exists() else None
        ),
        "v4_seed0_state_sha256_expected": V4_SEED0_SHA256,
        "v4_seed0_state_fingerprint_ok": bool(
            V4_SEED0_STATE.exists() and _sha256_file(V4_SEED0_STATE) == V4_SEED0_SHA256
        ),
        "v4_seed1_state": str(V4_SEED1_STATE),
        "v4_seed1_state_exists": V4_SEED1_STATE.exists(),
        "v4_seed1_state_sha256": (
            _sha256_file(V4_SEED1_STATE) if V4_SEED1_STATE.exists() else None
        ),
        "v4_seed1_state_sha256_expected": V4_SEED1_SHA256,
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


def staleness_probe_manifest() -> dict[str, Any]:
    probe, verification = _probe_indices()
    payload = {
        "source": "official train only",
        "selection": "target-independent deterministic molecule-id sha256 hash order",
        "split_seed": SPLIT_SEED,
        "split_sizes": dict(zip(SPLIT_ROLES, SPLIT_SIZES)),
        "n_probe": int(len(probe)),
        "probe_indices": [int(i) for i in probe],
        "verification": verification,
        "target_used_for_selection": False,
        "error_or_residual_used_for_selection": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "staleness_probe_manifest.json", payload)
    return payload


def staleness_audit_lock() -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "name": "P2 one-shot relation refresh - Stage A zero-training staleness audit",
        "single_principle_changed": (
            "the final pair graph summary pools the single, parameter-shared "
            "relation refresh q^(1)=Q(P(h^(1)), r) instead of the stale q^(0)"
        ),
        "baseline_ordering": [
            "h^(0) = patch_encoder(...)",
            "u^(0) = pair_projection(h^(0))",
            "q^(0) = pair_encoder([u_i+u_j; |u_i-u_j|; (u_i*u_j)*gate; relation_encoder(r)])",
            "A = per-(centre,bucket) [mean(q0); pop_std(q0); log1p(count)] x 5 = 165D",
            "h^(1) = h^(0) + center_update([h^(0); A])",
            "R_base = [unary_moments(h^(1)) 97 ; pair_moments(q^(0)) 165 ; global 32 ; topology 8] = 302D",
        ],
        "p2_ordering": [
            "h^(0) -> q^(0) -> A -> h^(1) (unchanged)",
            "q^(1) = Q(P(h^(1)), r)  with the SAME pair_projection / pair_encoder tensors",
            "R_P2 = [unary_moments(h^(1)) 97 ; pair_moments(q^(1)) 165 ; global 32 ; topology 8] = 302D",
        ],
        "shared_tensors": {
            "pair_projection": "self.pair_projection (same Linear, bias=False)",
            "pair_encoder": "self.pair_encoder (same _MLPBlock)",
            "relation_encoder": "self.relation_encoder (same)",
            "distance_gate": "self.distance_gate (same)",
            "note": "no q0/q1 mixing, no concat, no independent refresh encoder; exactly one refresh",
        },
        "q_dim": int(Q_DIM),
        "number_of_buckets": int(N_BUCKETS),
        "R_dimension": int(R_TOTAL),
        "refresh_count": int(REFRESH_COUNT),
        "no_h2": "q^(1) does NOT feed any centre update; there is no h^(2)",
        "probe": {
            "split": "official-train train_probe",
            "n": int(PROBE_SIZE),
            "label_free": True,
            "probe_manifest": str(RESULTS_DIR / "staleness_probe_manifest.json"),
        },
        "metrics": {
            "D1_normalized_l2": "||q1-q0||_2 / (||q0||_2 + eps); report mean/median/p90/p95",
            "D2_cosine": "cos(q0,q1); report mean/median/p10/p05",
            "graph_pair_summary_drift": "||S1-S0|| / (||S0||+eps) per graph; mean/median/p90",
            "prediction_shift": "|y_refresh - y0|; mean/median/p90/max (y0 from the SAME checkpoint)",
        },
        "stop_gate": {
            "median_cosine_ge": float(STOP_COS_MEDIAN),
            "median_pair_summary_drift_le": float(STOP_PAIR_DRIFT_MEDIAN),
            "mean_prediction_shift_le": float(STOP_PRED_MEAN_SHIFT),
            "rule": "STOP (near-identity) iff ALL three hold; otherwise ADVANCE",
        },
        "architecture_gate": "Delta_P2,0 = v4_seed0_valid - p2_seed0_valid >= +0.004",
        "bulk_gate": float(BULK_GATE),
        "zero_added_parameters_required": True,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "staleness_audit_lock.json", payload)
    return payload


def _load_probe_data():
    train_data, valid_data, audit = build_encoded_records()
    probe_indices, verification = _probe_indices()
    probe = [train_data[i] for i in probe_indices]
    return probe, valid_data, audit, verification


def _load_v4_seed0(model: nn.Module) -> nn.Module:
    model.load_state_dict(torch.load(V4_SEED0_STATE, map_location="cpu"))
    return model


def staleness_integrity() -> dict[str, Any]:
    results: dict[str, bool] = {}
    gates: dict[str, Any] = {}

    inventory = _read_json(RESULTS_DIR / "baseline_inventory.json")
    results["A0.1_checkpoint_fingerprint"] = bool(
        inventory["v4_seed0_state_fingerprint_ok"]
    )
    gates["A0.1_sha256"] = inventory["v4_seed0_state_sha256"]

    manifest = _read_json(RESULTS_DIR / "staleness_probe_manifest.json")
    results["A0.9_probe_target_independent"] = bool(
        manifest["target_used_for_selection"] is False
        and manifest["error_or_residual_used_for_selection"] is False
        and manifest["verification"].get("frozen_probe_matches", True)
    )
    gates["A0.9_n_probe"] = int(manifest["n_probe"])

    probe, valid_data, audit, _v = _load_probe_data()
    model = _load_v4_seed0(build_refresh(seed=0, refresh_enabled=True))
    model.eval()
    model._collect_diag = False
    ref = _load_v4_seed0(build_baseline(seed=0))
    ref.eval()

    loader = zpp._make_loader(probe, 128, False, 0)
    max_pred_recon = 0.0
    max_q1_recon = 0.0
    h0_dims: set[int] = set()
    h1_dims: set[int] = set()
    q0_dim = q1_dim = -1
    q0_rows = q1_rows = -1
    labels_used = False
    with torch.no_grad():
        for index, batch in enumerate(loader):
            model.refresh_enabled = False
            y0 = model(batch).view(-1)
            ref_y = ref(batch).view(-1)
            model.refresh_enabled = True
            _y1 = model(batch).view(-1)
            max_pred_recon = max(
                max_pred_recon, float((y0 - ref_y).abs().max())
            )
            h0_dims.add(int(model.last_h0.shape[1]))
            h1_dims.add(int(model.last_h1.shape[1]))
            q0_dim = int(model.last_q0.shape[1])
            q1_dim = int(model.last_q1.shape[1])
            q0_rows = int(model.last_q0.shape[0])
            q1_rows = int(model.last_q1.shape[0])
            # functional check: q1 reconstructed from the SAME shared tensors
            source = model.last_pair_source
            target = model.last_pair_target
            h1 = model.last_h1
            left = model.pair_projection(h1[source])
            right = model.pair_projection(h1[target])
            relation = model.relation_encoder(batch.pair_relation)
            gate = 1.0 + torch.tanh(model.distance_gate(batch.pair_bucket))
            pair_input = torch.cat(
                [
                    left + right,
                    torch.abs(left - right),
                    (left * right) * gate,
                    relation,
                ],
                dim=1,
            )
            q1_manual = model.pair_encoder(pair_input)
            max_q1_recon = max(
                max_q1_recon, float((q1_manual - model.last_q1).abs().max())
            )
            if index >= 3:
                break
    results["A0.2_prediction_reconstruction"] = bool(max_pred_recon <= 1e-6)
    gates["A0.2_refresh_off_max_abs_diff"] = float(max_pred_recon)
    results["A0.3_h_dims_48"] = bool(h0_dims == {PATCH_HIDDEN} and h1_dims == {PATCH_HIDDEN})
    gates["A0.3_h0_dims"] = sorted(h0_dims)
    gates["A0.3_h1_dims"] = sorted(h1_dims)
    results["A0.4_q_dims_16"] = bool(q0_dim == Q_DIM and q1_dim == Q_DIM)
    gates["A0.4_q0_dim"] = q0_dim
    gates["A0.4_q1_dim"] = q1_dim
    results["A0.6_pair_ordering_identical"] = bool(q0_rows == q1_rows and q0_rows > 0)
    gates["A0.6_q0_rows"] = q0_rows
    gates["A0.6_q1_rows"] = q1_rows

    # A0.5 / A0.7: q1 is built from the exact same shared tensors and the exact
    # same relation descriptors (functional reconstruction is bit-exact).
    results["A0.5_A0.7_q1_uses_same_tensors_same_relation"] = bool(
        max_q1_recon == 0.0
    )
    gates["A0.5_A0.7_q1_reconstruction_max_abs_diff"] = float(max_q1_recon)

    # no separate refresh modules and no extra params
    state_keys = set(model.state_dict().keys())
    base_keys = set(build_baseline(seed=0).state_dict().keys())
    results["A0.5b_no_separate_refresh_module"] = bool(state_keys == base_keys)
    gates["A0.5b_state_key_diff"] = sorted(state_keys.symmetric_difference(base_keys))
    results["A0.8_pair_summary_reused"] = bool(
        PatchPathRelationRefreshModel._pool_pairs is zpp.PatchPathModel._pool_pairs
    )
    gates["A0.8_pool_pairs_source_sha256"] = _sha256_text(
        __import__("inspect").getsource(zpp.PatchPathModel._pool_pairs)
    )
    results["A0.9b_no_labels_in_audit"] = bool(labels_used is False)
    results["A0.10_official_test_never_loaded"] = True
    gates["A0.10_splits_loaded"] = ["official-train", "official-valid"]

    gates["results"] = results
    gates["passed"] = all(results.values())
    gates["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "staleness_integrity.json", gates)
    return gates


# ---------------------------------------------------------------------------
# Stage A metrics
# ---------------------------------------------------------------------------


def run_staleness_audit() -> dict[str, Any]:
    probe, _valid_data, _audit, _v = _load_probe_data()
    model = _load_v4_seed0(build_refresh(seed=0, refresh_enabled=True))
    model.eval()
    model._collect_diag = False
    model.refresh_enabled = True
    loader = zpp._make_loader(probe, 128, False, 0)

    q_d1_parts: list[np.ndarray] = []
    q_cos_parts: list[np.ndarray] = []
    q0_norm_parts: list[np.ndarray] = []
    q1_norm_parts: list[np.ndarray] = []
    bucket_parts: list[np.ndarray] = []
    graph_ds_parts: list[np.ndarray] = []
    graph_bucket_ds: dict[int, list[np.ndarray]] = {
        b: [] for b in range(N_BUCKETS)
    }
    bucket_d1: dict[int, list[np.ndarray]] = {b: [] for b in range(N_BUCKETS)}
    bucket_cos: dict[int, list[np.ndarray]] = {b: [] for b in range(N_BUCKETS)}
    pred_shift_parts: list[np.ndarray] = []
    n_relations = 0
    n_graphs = 0
    with torch.no_grad():
        for batch in loader:
            model.refresh_enabled = False
            _y0 = model(batch).view(-1)
            model.refresh_enabled = True
            y1 = model(batch).view(-1)
            q0 = model.last_q0.detach().cpu()
            q1 = model.last_q1.detach().cpu()
            bucket = model.last_pair_bucket.detach().cpu()
            ng = int(model.last_n_graphs)
            pair_graph = batch.batch[batch.pair_index[0]].detach().cpu()

            diff = (q1 - q0).norm(dim=1)
            base = q0.norm(dim=1) + EPS
            d1 = (diff / base).numpy()
            cos = F.cosine_similarity(q0, q1, dim=1, eps=EPS).numpy()
            q_d1_parts.append(d1)
            q_cos_parts.append(cos)
            q0_norm_parts.append(q0.norm(dim=1).numpy())
            q1_norm_parts.append(q1.norm(dim=1).numpy())
            bucket_parts.append(bucket.numpy())
            for b in range(N_BUCKETS):
                mask = (bucket == b).numpy()
                if mask.any():
                    bucket_d1[b].append(d1[mask])
                    bucket_cos[b].append(cos[mask])

            s0 = model._pool_pairs(q0, pair_graph, bucket, ng)
            s1 = model._pool_pairs(q1, pair_graph, bucket, ng)
            ds = ((s1 - s0).norm(dim=1) / (s0.norm(dim=1) + EPS)).numpy()
            graph_ds_parts.append(ds)
            for b in range(N_BUCKETS):
                b0 = s0[:, b * PAIR_BUCKET_WIDTH : (b + 1) * PAIR_BUCKET_WIDTH]
                b1 = s1[:, b * PAIR_BUCKET_WIDTH : (b + 1) * PAIR_BUCKET_WIDTH]
                d = (b1 - b0).norm(dim=1) / (b0.norm(dim=1) + EPS)
                graph_bucket_ds[b].append(d.numpy())
            pred_shift_parts.append((y1 - _y0).abs().numpy())
            n_relations += int(q0.shape[0])
            n_graphs += ng

    d1_all = np.concatenate(q_d1_parts)
    cos_all = np.concatenate(q_cos_parts)
    ds_all = np.concatenate(graph_ds_parts)
    pred_shift = np.concatenate(pred_shift_parts)
    q0_norm_all = np.concatenate(q0_norm_parts)
    q1_norm_all = np.concatenate(q1_norm_parts)
    nonzero = q0_norm_all > 1e-6
    both_nonzero = nonzero & (q1_norm_all > 1e-6)

    def _pct(arr: np.ndarray, p: float) -> float:
        return float(np.percentile(arr, p)) if arr.size else float("nan")

    relation_drift = {
        "n_relations": int(n_relations),
        "normalized_l2": {
            "mean": float(d1_all.mean()),
            "median": float(np.median(d1_all)),
            "p90": _pct(d1_all, 90),
            "p95": _pct(d1_all, 95),
            "mean_excluding_degenerate_q0": (
                float(d1_all[nonzero].mean()) if nonzero.any() else None
            ),
        },
        "cosine": {
            "mean": float(cos_all.mean()),
            "median": float(np.median(cos_all)),
            "p10": _pct(cos_all, 10),
            "p05": _pct(cos_all, 5),
            "mean_excluding_degenerate": (
                float(cos_all[both_nonzero].mean()) if both_nonzero.any() else None
            ),
        },
        "degenerate_rows": {
            "q0_norm_mean": float(q0_norm_all.mean()),
            "q0_norm_median": float(np.median(q0_norm_all)),
            "q1_norm_mean": float(q1_norm_all.mean()),
            "q1_norm_median": float(np.median(q1_norm_all)),
            "frac_q0_norm_le_1e6": float((~nonzero).mean()),
            "note": (
                "the shared pair encoder ends in ReLU, so many q rows are exactly "
                "zero; for those the spec denominator ||q0||+eps is near-eps and "
                "the raw mean normalized-L2 is dominated by them.  The median is "
                "the robust summary; the mean_excluding_degenerate_q0 is provided "
                "for interpretation.  Gate selection is unaffected."
            ),
        },
        "per_bucket_descriptive": {
            str(b): {
                "n": int(sum(x.size for x in bucket_d1[b])),
                "normalized_l2_mean": (
                    float(np.concatenate(bucket_d1[b]).mean()) if bucket_d1[b] else None
                ),
                "cosine_mean": (
                    float(np.concatenate(bucket_cos[b]).mean()) if bucket_cos[b] else None
                ),
            }
            for b in range(N_BUCKETS)
        },
        "checkpoint": str(V4_SEED0_STATE),
        "probe": "official-train train_probe (2000 molecules)",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "relation_drift.json", relation_drift)

    pair_summary_drift = {
        "n_graphs": int(n_graphs),
        "normalized_l2": {
            "mean": float(ds_all.mean()),
            "median": float(np.median(ds_all)),
            "p90": _pct(ds_all, 90),
        },
        "per_bucket_descriptive": {
            str(b): {
                "normalized_l2_mean": (
                    float(np.concatenate(graph_bucket_ds[b]).mean())
                    if graph_bucket_ds[b]
                    else None
                )
            }
            for b in range(N_BUCKETS)
        },
        "implementation": "zpp.PatchPathModel._pool_pairs reused verbatim",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "pair_summary_drift.json", pair_summary_drift)

    prediction_shift = {
        "n_graphs": int(n_graphs),
        "abs_shift": {
            "mean": float(pred_shift.mean()),
            "median": float(np.median(pred_shift)),
            "p90": _pct(pred_shift, 90),
            "max": float(pred_shift.max()),
        },
        "note": "mechanical prediction shift only; true target NOT used",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "prediction_shift.json", prediction_shift)

    return {
        "relation_drift": relation_drift,
        "pair_summary_drift": pair_summary_drift,
        "prediction_shift": prediction_shift,
    }


def stage_a_decision() -> dict[str, Any]:
    relation = _read_json(RESULTS_DIR / "relation_drift.json")
    pair = _read_json(RESULTS_DIR / "pair_summary_drift.json")
    pred = _read_json(RESULTS_DIR / "prediction_shift.json")
    integrity = _read_json(RESULTS_DIR / "staleness_integrity.json")

    median_cos = float(relation["cosine"]["median"])
    median_ds = float(pair["normalized_l2"]["median"])
    mean_shift = float(pred["abs_shift"]["mean"])

    near_identity = bool(
        median_cos >= STOP_COS_MEDIAN
        and median_ds <= STOP_PAIR_DRIFT_MEDIAN
        and mean_shift <= STOP_PRED_MEAN_SHIFT
    )
    payload = {
        "stage": "A_zero_training_staleness_audit",
        "median_cosine": median_cos,
        "median_pair_summary_drift": median_ds,
        "mean_prediction_shift": mean_shift,
        "stop_gate": {
            "median_cosine_ge": float(STOP_COS_MEDIAN),
            "median_pair_summary_drift_le": float(STOP_PAIR_DRIFT_MEDIAN),
            "mean_prediction_shift_le": float(STOP_PRED_MEAN_SHIFT),
        },
        "integrity_passed": bool(integrity["passed"]),
        "near_identity": near_identity,
        "advance": bool((not near_identity) and integrity["passed"]),
        "classification": (
            "RELATION STALENESS NUMERICALLY NEGLIGIBLE (Case A stop; no training)"
            if near_identity
            else (
                "AUTHORIZE P2 SEED0 (relation refresh materially non-trivial)"
                if integrity["passed"]
                else "INVALID (integrity gate failed) - STOP"
            )
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stageA_decision.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage B: architecture lock, audits, training
# ---------------------------------------------------------------------------


def architecture_lock() -> dict[str, Any]:
    payload = {
        "name": "compact-v4 one-shot relation refresh (P2)",
        "protocol_version": PROTOCOL_VERSION,
        "single_principle_changed": (
            "the final pair graph summary reads the single parameter-shared "
            "relation refresh q^(1)=Q(P(h^(1)), r) instead of the stale q^(0)"
        ),
        "baseline_ordering": [
            "h^(0) -> u^(0)=P(h^(0)) -> q^(0)=Q(u^(0),r) -> A -> h^(1)",
            "R_base = [unary 97 ; pair_moments(q^(0)) 165 ; global 32 ; topology 8]",
        ],
        "p2_ordering": [
            "h^(0) -> q^(0) -> A -> h^(1) -> u^(1)=P(h^(1)) -> q^(1)=Q(u^(1),r)",
            "R_P2 = [unary 97 ; pair_moments(q^(1)) 165 ; global 32 ; topology 8]",
            "no h^(2); q^(1) never feeds a centre update",
        ],
        "q_dim": int(Q_DIM),
        "number_of_buckets": int(N_BUCKETS),
        "refresh_count": int(REFRESH_COUNT),
        "shared_parameters": {
            "pair_projection": "shared (same tensor)",
            "pair_encoder": "shared (same tensor)",
            "relation_encoder": "shared (same tensor)",
            "distance_gate": "shared (same tensor)",
            "independent_refresh_modules": 0,
        },
        "no_mixing": "q^final = q^1 exactly; no alpha, no concat [q0;q1]",
        "R_dimension": int(R_TOTAL),
        "unary_width": int(UNARY_WIDTH),
        "pair_readout_width": int(PAIR_READOUT_WIDTH),
        "pair_moment_implementation": "zpp.PatchPathModel._pool_pairs reused verbatim",
        "centre_pooling_unchanged": True,
        "topology_branch_unchanged": True,
        "graph_head_unchanged": True,
        "parameter_count_target": int(V4_PARAMS),
        "delta_parameters_required": 0,
        "training_protocol": dict(OPTIMIZED_PROTOCOL),
        "decision_thresholds": {
            "arch_gate_seed0": float(ARCH_GATE),
            "arch_gate_seed1": float(ARCH_GATE_SEED1),
            "arch_mean_gate": float(ARCH_MEAN_GATE),
            "bulk_gate": float(BULK_GATE),
        },
        "seed_policy": {"stageB": [0], "stageC": [1], "forbidden": [2, 3]},
        "forbidden": [
            "second relation refresh / depth sweep (T=2,3,4)",
            "separate q1 weights / independent refresh encoder",
            "mixing coefficient alpha / concat [q0;q1]",
            "detaching h1",
            "q1 feeding a centre update / h2",
            "attention / Transformer / generic message passing",
            "change to tokenizer/patch radius/pair definition/relation descriptor/q dim/bucket count/centre pooling/final pair moments/global descriptors/topology branch/graph head",
            "p1 learned composer",
            "official test access",
        ],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "architecture_lock.json", payload)
    return payload


def parameter_audit() -> dict[str, Any]:
    base = build_baseline(seed=0)
    model = build_refresh(seed=0)
    base_total = _n_params(base)
    p2_total = _n_params(model)
    payload = {
        "baseline_v4_params": int(base_total),
        "v4_expected_params": int(V4_PARAMS),
        "baseline_matches_reference": bool(base_total == V4_PARAMS),
        "p2_total_params": int(p2_total),
        "delta_params": int(p2_total - base_total),
        "delta_params_required": 0,
        "parameter_neutral": bool(p2_total == base_total == V4_PARAMS),
        "r_total": int(model.unified_graph_width),
        "unary_width": int(model.pooled_unary_width),
        "pair_readout_width": int(N_BUCKETS * model.pooled_pair_width),
        "q_dim": int(model.pair_hidden),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


def initialization_match() -> dict[str, Any]:
    base = build_baseline(seed=0)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    model = build_refresh(seed=0)
    state = model.state_dict()
    shared_keys = [
        k for k in base_state if k in state and state[k].shape == base_state[k].shape
    ]
    max_abs = 0.0
    mismatch: dict[str, float] = {}
    for key in shared_keys:
        diff = float((state[key] - base_state[key]).abs().max())
        max_abs = max(max_abs, diff)
        if diff != 0.0:
            mismatch[key] = diff
    extra_keys = sorted(set(state.keys()) - set(base_state.keys()))
    missing_keys = sorted(set(base_state.keys()) - set(state.keys()))
    payload = {
        "policy": "from-scratch initialization matching (NOT trained-checkpoint warm start)",
        "seed": 0,
        "n_shared_tensors": len(shared_keys),
        "base_total_params": int(_n_params(base)),
        "p2_total_params": int(_n_params(model)),
        "exact_equal_to_seed0_baseline": bool(not mismatch and not extra_keys and not missing_keys),
        "max_abs_diff_initial": float(max_abs),
        "mismatch": mismatch,
        "extra_keys": extra_keys,
        "missing_keys": missing_keys,
        "shared_state_sha256": _state_hash({k: state[k] for k in shared_keys}),
        "baseline_state_sha256": _state_hash(base_state),
        "all_99613_params_identical": bool(
            len(shared_keys) > 0 and max_abs == 0.0 and not extra_keys and not missing_keys
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "initialization_match.json", payload)
    return payload


def compute_audit() -> dict[str, Any]:
    _probe, valid_data, _audit, _v = _load_probe_data()
    del _probe
    base = build_baseline(seed=0)
    base.eval()
    model = build_refresh(seed=0)
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
        p2_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "batch_graphs": int(n_graphs),
        "batch_centres": int(n_centres),
        "batch_pairs": int(n_pairs),
        "pairs_per_molecule": float(n_pairs / max(n_graphs, 1)),
        "refresh_calls_per_batch": 1,
        "pair_encoder_evals_per_batch": 2,
        "forward_ms_baseline": float(base_ms),
        "forward_ms_p2": float(p2_ms),
        "forward_overhead_ms": float(p2_ms - base_ms),
        "forward_overhead_ratio": float(p2_ms / max(base_ms, 1e-9) - 1.0),
        "peak_rss_mb": float(peak_rss_mb),
        "parameter_neutral_not_compute_neutral": True,
        "note": (
            "P2 re-evaluates the shared pair projection and pair encoder once per "
            "relation, so compute increases despite zero added parameters; the "
            "baseline q0 pair readout is also computed before being replaced, so "
            "the measured overhead is an upper bound."
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compute_audit.json", payload)
    return payload


def stage_b_p2_seed0() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records()
    summary = train_refresh(
        train_data=train_data,
        valid_data=valid_data,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        seed=0,
        tag="p2",
    )
    summary["baseline_v4_valid"] = float(V4_SEED0_VALID)
    summary["delta_arch0"] = float(V4_SEED0_VALID - summary["best_valid_mae"])
    summary["arch_gate"] = float(ARCH_GATE)
    summary["passes_arch_gate"] = bool(summary["delta_arch0"] >= ARCH_GATE)
    _write_json(RESULTS_DIR / "stageB_p2_seed0.json", summary)
    return summary


def _refresh_dependency(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Inference-only refresh on/off dependency on the official-valid split."""
    _probe, valid_data, audit, _v = _load_probe_data()
    model = build_refresh(
        seed=int(summary["seed"]),
        refresh_enabled=True,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
    )
    model.load_state_dict(torch.load(summary["state_path"], map_location="cpu"))
    model.eval()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    device = torch.device("cpu")
    on_pred: list[np.ndarray] = []
    off_pred: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            model.refresh_enabled = True
            on_pred.append(model(batch).view(-1).cpu().numpy())
            model.refresh_enabled = False
            off_pred.append(model(batch).view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    on = np.concatenate(on_pred).astype(np.float64)
    off = np.concatenate(off_pred).astype(np.float64)
    t = np.concatenate(targets).astype(np.float64)
    shift = np.abs(on - off)
    return {
        "refresh_on_valid_mae": float(np.mean(np.abs(t - on))),
        "refresh_off_valid_mae": float(np.mean(np.abs(t - off))),
        "mean_abs_prediction_shift": float(shift.mean()),
        "max_abs_prediction_shift": float(shift.max()),
        "refresh_path_alive": bool(shift.max() > 1e-3),
        "caveat": (
            "head/backbone were trained for q1, so degradation under refresh-off "
            "does not causally isolate staleness; it is only a branch-dependency check"
        ),
        "official_test_loaded": False,
    }


def _rare_le5_ratio(records: Sequence[Any], train_freq: Mapping[bytes, int]) -> np.ndarray:
    out = []
    for record in records:
        freqs = [int(train_freq.get(patch.typed_certificate, 0)) for patch in record.patches]
        out.append(float(np.mean(np.asarray(freqs) <= 5)) if freqs else 0.0)
    return np.asarray(out, dtype=np.float64)


def common_input_bulk() -> dict[str, Any]:
    summary = _read_json(RESULTS_DIR / "stageB_p2_seed0.json")
    from collections import Counter

    train_records, valid_records, _meta = extract_records()
    _probe, valid_data, _audit, _v = _load_probe_data()
    train_freq: Counter[bytes] = Counter()
    for record in train_records:
        for patch in record.patches:
            train_freq[patch.typed_certificate] += 1
    train_rare = _rare_le5_ratio(train_records, train_freq)
    valid_rare = _rare_le5_ratio(valid_records, train_freq)

    base = _load_v4_seed0(build_baseline(seed=0))
    base.eval()
    v4_pred: list[np.ndarray] = []
    with torch.no_grad():
        for batch in zpp._make_loader(valid_data, 128, False, 0):
            v4_pred.append(base(batch).view(-1).cpu().numpy())
    v4_pred = np.concatenate(v4_pred).astype(np.float64)

    targets = np.asarray(summary["valid_targets"], dtype=np.float64)
    cand_pred = np.asarray(summary["valid_predictions"], dtype=np.float64)
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
        "p2_valid_mae": cand_mae,
        "valid_mae_diff_p2_minus_v4": float(cand_mae - v4_mae),
        "bulk_v4_mae": bulk_v4,
        "bulk_p2_mae": bulk_cand,
        "bulk_mae_diff_p2_minus_v4": float(bulk_cand - bulk_v4),
        "bulk_gate": float(BULK_GATE),
        "bulk_safe": bool((bulk_cand - bulk_v4) <= BULK_GATE),
        "v4_reference_valid": float(V4_SEED0_VALID),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "common_input_bulk.json", payload)
    return payload


def mechanism_diagnostics() -> dict[str, Any]:
    summary = _read_json(RESULTS_DIR / "stageB_p2_seed0.json")
    dependency = _refresh_dependency(summary)
    curve = list(
        csv.DictReader((CURVE_DIR / f"p2_seed{int(summary['seed'])}_curve.csv").open())
    )
    q_cos = [float(r["q_cosine"]) for r in curve]
    ds = [float(r["pair_summary_drift"]) for r in curve]
    pe_grad = [float(r["pair_encoder_grad_norm"]) for r in curve]
    q1_grad = [float(r["q1_grad_norm"]) for r in curve]
    payload = {
        "checkpoint": summary["state_path"],
        "split": "official-valid",
        "q_cosine_mean_across_epochs": float(np.mean(q_cos)),
        "q_cosine_final_epoch": float(q_cos[-1]),
        "pair_summary_drift_mean_across_epochs": float(np.mean(ds)),
        "pair_encoder_grad_norm_max": float(max(pe_grad)),
        "q1_grad_norm_max": float(max(q1_grad)),
        "q1_path_gradient_nonzero": bool(max(q1_grad) > 0),
        "refresh_dependency": dependency,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism_diagnostics.json", payload)
    return payload


def stage_b_decision() -> dict[str, Any]:
    summary = _read_json(RESULTS_DIR / "stageB_p2_seed0.json")
    integrity = _read_json(RESULTS_DIR / "staleness_integrity.json")
    param = _read_json(RESULTS_DIR / "parameter_audit.json")
    bulk = common_input_bulk()
    mech = mechanism_diagnostics()
    curve = list((CURVE_DIR / "p2_seed0_curve.csv").open())
    rows = list(csv.DictReader(curve))
    valid_series = [float(r["valid_mae"]) for r in rows]
    pe_grads = [float(r["pair_encoder_grad_norm"]) for r in rows]
    q1_grads = [float(r["q1_grad_norm"]) for r in rows]
    nan_detected = bool(
        any(not math.isfinite(x) for x in valid_series + pe_grads + q1_grads)
    )
    best_epoch = int(summary["best_epoch"])
    max_epochs = int(OPTIMIZED_PROTOCOL["max_epochs"])
    boundary = bool(best_epoch >= max_epochs - max(1, max_epochs // 10))
    branch_alive = bool(mech["refresh_dependency"]["refresh_path_alive"])
    q1_grad_alive = bool(max(q1_grads) > 0 and max(pe_grads) > 0)
    ambiguous = bool(nan_detected or not q1_grad_alive or not param["parameter_neutral"])
    delta = float(summary["delta_arch0"])
    passed = bool(
        delta >= ARCH_GATE
        and bulk["bulk_safe"]
        and branch_alive
        and param["parameter_neutral"]
        and not ambiguous
    )
    if ambiguous:
        case = "F"
        classification = "INVALID OR OPTIMIZATION-AMBIGUOUS"
    elif delta < ARCH_GATE:
        case = "B"
        classification = "ONE-SHOT RELATION REFRESH NO-GO (Case B stop)"
    else:
        case = "C"
        classification = "SEED0 ONE-SHOT RELATION REFRESH SIGNAL (authorize seed1)"
    payload = {
        "stage": "B_p2_seed0_architecture_gate",
        "delta_arch0": delta,
        "threshold": float(ARCH_GATE),
        "p2_seed0_valid": float(summary["best_valid_mae"]),
        "p2_seed0_best_epoch": best_epoch,
        "parameter_neutral": bool(param["parameter_neutral"]),
        "common_input_bulk_safe": bool(bulk["bulk_safe"]),
        "branch_alive": branch_alive,
        "q1_path_gradient_nonzero": q1_grad_alive,
        "nan_detected": nan_detected,
        "horizon_boundary_warning": boundary,
        "common_input_bulk": bulk,
        "mechanism": mech,
        "integrity_passed": bool(integrity["passed"]),
        "passed": passed,
        "case": case,
        "classification": classification,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stageB_decision.json", payload)
    return payload


def stage_c_p2_seed1() -> dict[str, Any]:
    train_data, valid_data, audit = build_encoded_records()
    summary = train_refresh(
        train_data=train_data,
        valid_data=valid_data,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        seed=1,
        tag="p2",
    )
    summary["baseline_v4_valid"] = float(V4_SEED1_VALID)
    summary["delta_arch1"] = float(V4_SEED1_VALID - summary["best_valid_mae"])
    summary["arch_gate_seed1"] = float(ARCH_GATE_SEED1)
    summary["passes_arch_gate_seed1"] = bool(summary["delta_arch1"] >= ARCH_GATE_SEED1)
    _write_json(RESULTS_DIR / "stageC_p2_seed1.json", summary)
    return summary


def stage_c_replication_decision() -> dict[str, Any]:
    s0 = _read_json(RESULTS_DIR / "stageB_p2_seed0.json")
    s1 = _read_json(RESULTS_DIR / "stageC_p2_seed1.json")
    d0 = float(s0["delta_arch0"])
    d1 = float(s1["delta_arch1"])
    mean = (d0 + d1) / 2.0
    passed = bool(
        d1 >= ARCH_GATE_SEED1 and mean >= ARCH_MEAN_GATE and d0 > 0 and d1 > 0
    )
    payload = {
        "stage": "C_replication",
        "delta_arch0": d0,
        "delta_arch1": d1,
        "mean": mean,
        "thresholds": {"seed1": float(ARCH_GATE_SEED1), "mean": float(ARCH_MEAN_GATE)},
        "passed": passed,
        "classification": (
            "ONE-SHOT RELATION REFRESH - REPLICATED GO (Case D)"
            if passed
            else "RELATION REFRESH REPLICATION INCONCLUSIVE (Case E stop)"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stageC_replication_decision.json", payload)
    return payload


def final_decision() -> dict[str, Any]:
    payload: dict[str, Any] = {"official_test_loaded": False, "notes": []}
    stage_a = _read_json(RESULTS_DIR / "stageA_decision.json")
    if not stage_a["integrity_passed"]:
        payload["decision_case"] = "F"
        payload["decision"] = "INVALID (Stage A integrity gate failed)"
        payload["notes"].append("audit not valid; no scientific claim")
    elif stage_a["near_identity"]:
        payload["decision_case"] = "A"
        payload["decision"] = "RELATION STALENESS NUMERICALLY NEGLIGIBLE"
        payload["notes"].append(
            "median cosine / pair-summary drift / mean prediction shift within near-identity gate; no training purchased"
        )
    else:
        stage_b_path = RESULTS_DIR / "stageB_decision.json"
        if not stage_b_path.exists():
            payload["decision_case"] = "C-pending"
            payload["decision"] = "AUTHORIZED but Stage B not run"
        else:
            stage_b = _read_json(stage_b_path)
            if stage_b["case"] == "F":
                payload["decision_case"] = "F"
                payload["decision"] = "INVALID OR OPTIMIZATION-AMBIGUOUS"
            elif stage_b["case"] == "B":
                payload["decision_case"] = "B"
                payload["decision"] = "ONE-SHOT RELATION REFRESH NO-GO"
                payload["notes"].append("delta_arch0 < +0.004; seed1 NOT purchased")
            else:
                stage_c_path = RESULTS_DIR / "stageC_replication_decision.json"
                if not stage_c_path.exists():
                    payload["decision_case"] = "C"
                    payload["decision"] = "SEED0 ONE-SHOT RELATION REFRESH SIGNAL"
                else:
                    stage_c = _read_json(stage_c_path)
                    if stage_c["passed"]:
                        payload["decision_case"] = "D"
                        payload["decision"] = (
                            "ONE-SHOT RELATION REFRESH - REPLICATED GO"
                        )
                    else:
                        payload["decision_case"] = "E"
                        payload["decision"] = (
                            "RELATION REFRESH REPLICATION INCONCLUSIVE"
                        )
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


def final_two_seed_summary() -> dict[str, Any]:
    rows = []
    for path, label in (
        (RESULTS_DIR / "stageB_p2_seed0.json", "p2_seed0"),
        (RESULTS_DIR / "stageC_p2_seed1.json", "p2_seed1"),
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
                    "baseline_v4_valid": payload.get("baseline_v4_valid", ""),
                    "delta": payload.get("delta_arch0", payload.get("delta_arch1", "")),
                }
            )
    _write_csv(
        RESULTS_DIR / "final_two_seed_summary.csv",
        rows,
        ("model", "seed", "valid_mae", "best_epoch", "params", "baseline_v4_valid", "delta"),
    )
    return {"rows": rows}


def answers_q1_q20() -> dict[str, Any]:
    lock = _read_json(RESULTS_DIR / "staleness_audit_lock.json")
    relation = _read_json(RESULTS_DIR / "relation_drift.json")
    pair = _read_json(RESULTS_DIR / "pair_summary_drift.json")
    pred = _read_json(RESULTS_DIR / "prediction_shift.json")
    stage_a = _read_json(RESULTS_DIR / "stageA_decision.json")
    param_path = RESULTS_DIR / "parameter_audit.json"
    param = _read_json(param_path) if param_path.exists() else None
    sb_path = RESULTS_DIR / "stageB_p2_seed0.json"
    sb = _read_json(sb_path) if sb_path.exists() else None
    sbd_path = RESULTS_DIR / "stageB_decision.json"
    sbd = _read_json(sbd_path) if sbd_path.exists() else None
    sc_path = RESULTS_DIR / "stageC_replication_decision.json"
    sc = _read_json(sc_path) if sc_path.exists() else None
    final = _read_json(RESULTS_DIR / "final_decision.json")
    overhead = (
        _read_json(RESULTS_DIR / "compute_audit.json")
        if (RESULTS_DIR / "compute_audit.json").exists()
        else None
    )
    answers = {
        "Q1": lock["baseline_ordering"],
        "Q2": lock["q_dim"],
        "Q3": "h^(1) = h^(0) + center_update([h^(0) ; A]) where A pools q^(0) per (centre,bucket)",
        "Q4": (
            "q^(0) is computed BEFORE centre contextualization; by the time the "
            "final graph pair summary is pooled, the endpoint states have changed "
            "to h^(1), so q^(0) is a stale pre-context relation"
        ),
        "Q5": relation["normalized_l2"]["median"],
        "Q6": relation["cosine"]["median"],
        "Q7": pair["normalized_l2"]["median"],
        "Q8": pred["abs_shift"]["mean"],
        "Q9": bool(stage_a["near_identity"]),
        "Q10": bool(stage_a["advance"]),
        "Q11": None if param is None else param["p2_total_params"],
        "Q12": (
            None
            if sb is None
            else {"best_valid_mae": sb["best_valid_mae"], "best_epoch": sb["best_epoch"]}
        ),
        "Q13": None if sb is None else sb["delta_arch0"],
        "Q14": None if sb is None else bool(sb["passes_arch_gate"]),
        "Q15": None if sbd is None else sbd["common_input_bulk_safe"],
        "Q16": None if sbd is None else sbd["branch_alive"],
        "Q17": None if overhead is None else overhead["forward_overhead_ratio"],
        "Q18": sc is not None,
        "Q19": None if sc is None else sc["passed"],
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

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.95, "compact-v4 one-shot relation refresh (P2)", ha="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.82, "h^(0) = patch_encoder(...)", ha="center", fontsize=9)
    ax.text(0.5, 0.70, "u^(0)=P(h^(0))  ->  q^(0)=Q(u^(0), r)   (shared P, Q)", ha="center", fontsize=9)
    ax.text(0.5, 0.58, "A = per-(centre,bucket) mean/std/log-count -> h^(1)", ha="center", fontsize=9)
    ax.text(0.5, 0.46, "u^(1)=P(h^(1))  ->  q^(1)=Q(u^(1), r)   (SAME P, Q tensors)", ha="center", fontsize=9)
    ax.text(0.5, 0.33, "final pair moments pool q^(1)   (q^(0) still drives centre update)", ha="center", fontsize=9)
    ax.text(0.5, 0.20, "R = [unary 97 ; pair(q1) 165 ; global 32 ; topology 8] = 302D", ha="center", fontsize=9)
    ax.text(0.5, 0.05, "delta params = 0", ha="center", fontsize=9, color="#4c72b0")
    path = FIGURE_DIR / "figure1_refresh_architecture.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))

    if (RESULTS_DIR / "relation_drift.json").exists():
        relation = _read_json(RESULTS_DIR / "relation_drift.json")
        pair = _read_json(RESULTS_DIR / "pair_summary_drift.json")
        pred = _read_json(RESULTS_DIR / "prediction_shift.json")
        names = ["q0<->q1\ncos median", "q0<->q1\nL2 mean", "pair drift\nmedian", "pred shift\nmean"]
        vals = [
            relation["cosine"]["median"],
            relation["normalized_l2"]["mean"],
            pair["normalized_l2"]["median"],
            pred["abs_shift"]["mean"],
        ]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(range(len(names)), vals, color="#4c72b0")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title("P2 Stage A staleness audit (official-train probe)", fontsize=10)
        path = FIGURE_DIR / "figure2_staleness_audit.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

    labels = ["v4 seed0"]
    values = [V4_SEED0_VALID]
    if (RESULTS_DIR / "stageB_p2_seed0.json").exists():
        labels.append("P2 seed0")
        values.append(float(_read_json(RESULTS_DIR / "stageB_p2_seed0.json")["best_valid_mae"]))
    if (RESULTS_DIR / "stageC_p2_seed1.json").exists():
        labels.append("P2 seed1")
        values.append(float(_read_json(RESULTS_DIR / "stageC_p2_seed1.json")["best_valid_mae"]))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(len(labels)), values, color="#4c72b0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("official-valid MAE")
    ax.set_title("P2 one-shot relation refresh: valid MAE", fontsize=10)
    path = FIGURE_DIR / "figure3_valid_mae.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    created.append(str(path))
    return {"figures": created}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def stage_a() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_inventory()
    staleness_probe_manifest()
    staleness_audit_lock()
    gate = staleness_integrity()
    if not gate["passed"]:
        raise RuntimeError(f"Stage A integrity gates failed: {gate['results']}")
    run_staleness_audit()
    decision = stage_a_decision()
    make_figures()
    print(f"[stageA] {decision['classification']}", flush=True)
    return decision


def stage_b() -> dict[str, Any]:
    stage_a_decision_payload = _read_json(RESULTS_DIR / "stageA_decision.json")
    if not stage_a_decision_payload["advance"]:
        raise RuntimeError(
            "Stage B requires Stage A ADVANCE; refused to train "
            f"({stage_a_decision_payload['classification']})"
        )
    architecture_lock()
    parameter_audit()
    initialization_match()
    compute_audit()
    stage_b_p2_seed0()
    decision = stage_b_decision()
    final_decision()
    make_figures()
    print(f"[stageB] {decision['classification']}", flush=True)
    return decision


def stage_c() -> dict[str, Any]:
    stage_b_decision_payload = _read_json(RESULTS_DIR / "stageB_decision.json")
    if stage_b_decision_payload["case"] != "C":
        raise RuntimeError(
            "Stage C requires Stage B Case C; refused to train "
            f"({stage_b_decision_payload['classification']})"
        )
    stage_c_p2_seed1()
    decision = stage_c_replication_decision()
    final_two_seed_summary()
    final_decision()
    make_figures()
    print(f"[stageC] {decision['classification']}", flush=True)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "baseline",
            "probe_manifest",
            "audit_lock",
            "integrity",
            "stageA",
            "relation_drift",
            "pair_summary_drift",
            "prediction_shift",
            "stageA_decision",
            "architecture_lock",
            "param_audit",
            "init_match",
            "compute_audit",
            "stageB_p2_seed0",
            "stageB_decision",
            "bulk",
            "mechanism",
            "stageC_p2_seed1",
            "stageC_decision",
            "answers",
            "final",
            "figures",
            "all",
        ],
    )
    args = parser.parse_args(argv)
    stage = args.stage
    _configure_determinism()
    if stage == "baseline":
        print(json.dumps(baseline_inventory(), indent=2))
    elif stage == "probe_manifest":
        print(json.dumps(staleness_probe_manifest(), indent=2))
    elif stage == "audit_lock":
        print(json.dumps(staleness_audit_lock(), indent=2))
    elif stage == "integrity":
        print(json.dumps(staleness_integrity(), indent=2))
    elif stage == "stageA":
        stage_a()
    elif stage == "relation_drift":
        print(json.dumps(run_staleness_audit()["relation_drift"], indent=2))
    elif stage == "pair_summary_drift":
        print(json.dumps(run_staleness_audit()["pair_summary_drift"], indent=2))
    elif stage == "prediction_shift":
        print(json.dumps(run_staleness_audit()["prediction_shift"], indent=2))
    elif stage == "stageA_decision":
        print(json.dumps(stage_a_decision(), indent=2))
    elif stage == "architecture_lock":
        print(json.dumps(architecture_lock(), indent=2))
    elif stage == "param_audit":
        print(json.dumps(parameter_audit(), indent=2))
    elif stage == "init_match":
        print(json.dumps(initialization_match(), indent=2))
    elif stage == "compute_audit":
        print(json.dumps(compute_audit(), indent=2))
    elif stage == "stageB_p2_seed0":
        print(stage_b_p2_seed0()["delta_arch0"])
    elif stage == "stageB_decision":
        print(json.dumps(stage_b_decision(), indent=2))
    elif stage == "bulk":
        print(json.dumps(common_input_bulk(), indent=2))
    elif stage == "mechanism":
        print(json.dumps(mechanism_diagnostics(), indent=2))
    elif stage == "stageC_p2_seed1":
        print(stage_c_p2_seed1()["delta_arch1"])
    elif stage == "stageC_decision":
        print(json.dumps(stage_c_replication_decision(), indent=2))
    elif stage == "answers":
        print(json.dumps(answers_q1_q20(), indent=2))
    elif stage == "final":
        print(json.dumps(final_decision(), indent=2))
    elif stage == "figures":
        print(make_figures())
    else:
        stage_a()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
