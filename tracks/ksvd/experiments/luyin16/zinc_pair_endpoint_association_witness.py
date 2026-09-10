"""Pair endpoint association witness audit (compact-v4-hinge, ZINC).

This is a **frozen feature witness diagnostic**, not an architecture
benchmark.  It does not modify the backbone, does not train a new backbone and
does not replace the pair encoder.  It answers exactly one question:

    Given the already-frozen compact-v4-hinge states and the existing 302D
    fixed graph vector R, does the *cross-coordinate endpoint association*
    that the current symmetric pair map provably discards carry stable,
    exploitable, task-relevant signed predictive value on real ZINC data?

Current symmetric endpoint features (the pair encoder input, excluding
relation) are

    c_ij = [u_i + u_j, |u_i - u_j|, u_i (*) u_j]

with ``u_i = P(h_i) `` the 16D projected patch state.  ``c_ij`` is permutation
symmetric but not injective on the ordered endpoint vectors: ``u_i=(0,0),
u_j=(1,1)`` and ``u'_i=(0,1), u'_j=(1,0)`` share sum / abs-difference /
coordinatewise product.  The discarded part is the strictly off-diagonal part
of the symmetric endpoint outer interaction

    A_ij = u_i u_j^T + u_j u_i^T,  a_ij = offdiag(A_ij) in R^120 (k < l).

The audit:

* Stage 0  export ``frozen_state_export_v3_pair_endpoint`` (adds ``u_i``,
  raw pair relation) + hard integrity gates (NO adapter training).
* Stage 1  1 frozen OOF backbone seed x 5 outer folds x 1 init:
  B1 (R-only), B2 (diagonal control), E (endpoint witness).  Primary
  mechanism metric ``Delta_R = MAE(B1) - MAE(E)`` and
  ``Delta_diag = MAE(B2) - MAE(E)``.
* Stage 2  second frozen OOF backbone seed (only after a clear advance).

Everything is evaluated on official TRAIN molecules through the existing
5-fold OOF checkpoints.  Official valid/test are never loaded.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_pair_endpoint_association_witness <stage>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_oof_difficulty_audit import (
    CACHE_DIR as OOF_CACHE_DIR,
    FOLD_DIR,
    K_FOLDS,
    _fold_marker,
    _fold_slices,
    _frozen_config,
    _load_fold_npz,
    _load_train_labels,
    _load_train_records,
    _model_config_with_clamps,
    _read_json,
    _verify_records_vs_labels,
)
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import _build_v4_model

REPO_ROOT = zpp.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/pair_endpoint_association_witness"
EXPORT_DIR = RESULTS_DIR / "state_exports"
FIG_DIR = RESULTS_DIR / "figures"

EXPORT_VERSION = "frozen_state_export_v3_pair_endpoint"
LEGACY_EXPORT_VERSIONS = {
    "head_input_v1",
    "frozen_state_export_v1",
    "frozen_state_export_v2_corrected",
}
CONFIG_PATH = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
RARITY_CSV = OOF_CACHE_DIR / "rarity_rows.csv"

# --- frozen v4-hinge shapes (verified from the real model at runtime) ------
R_DIM = 302
PATCH_DIM = 48
PAIR_DIM = 16
RELATION_DIM = zpp.RELATION_WIDTH  # 23
N_BUCKETS = zpp.DISTANCE_BUCKETS  # 5

# --- endpoint witness definition -------------------------------------------
OFFDIAG_DIM = PAIR_DIM * (PAIR_DIM - 1) // 2  # 120
PROJ_DIM = 16
WITNESS_GRAPH_DIM = N_BUCKETS * 2 * PROJ_DIM  # 160 (mean + std per bucket)
DIAG_GRAPH_DIM = N_BUCKETS * 2 * PAIR_DIM  # 160 (same final dimensionality)

# The deterministic random projection is frozen for the whole audit and is
# never learned.  ``projection seed`` -> orthogonal columns.
PROJECTION_SEED = 20260911

# --- collision audit (descriptive, target-free) ----------------------------
KNN_K = 16  # pre-registered local neighbourhood size
AUDIT_POOL_PER_FOLD = 20000
AUDIT_QUERY_PER_FOLD = 2000
AUDIT_RANDOM_SAMPLES = 64
AUDIT_REL_CAND = 64  # relation-nearest candidate set size (version 2)
AUDIT_SEED = 20260911

# --- split / budget (identical to frozen readout sufficiency audit) --------
SPLIT_SEED = "frozen-readout-sufficiency-v2-20260910"
SPLIT_SIZES = (1200, 400, 400)  # adapter-fit / adapter-selection / adapter-evaluation
BACKBONE_SEEDS = (0, 1)
PRIMARY_BACKBONE_SEED = 0
ADAPTER_SEEDS = (0, 1)  # seed 1 used only when Stage 1 is BORDERLINE

# --- adapters (pre-registered, no sweeps) ----------------------------------
# B1 is the historical R-only residual head (R -> 13 -> 13 -> 1, 4,135 params).
# For the larger 462D input, the closest integer hidden width within the +-2%
# budget is a single hidden layer of width 9 ([R;G] -> 9 -> 1, 4,177 params,
# +1.0%).  A two-hidden-layer head at the closest width (9) would be 4,267
# (+3.2%) and is rejected by the pre-registered parameter-budget gate.
RONLY_HIDDEN = 13
RONLY_DEPTH = 2
WITNESS_HIDDEN = 9
WITNESS_DEPTH = 1
ADAPTER_LR = 1.0e-3
ADAPTER_EPOCHS = 400
ADAPTER_PATIENCE = 50
ADAPTER_BATCH = "full"

# --- decision thresholds (pre-registered) ----------------------------------
S1_ADV_DR = 0.002
S1_ADV_DDIAG = 0.0015
S1_ADV_FOLDS = 4
S1_NGO_DR = 0.0005
S1_NGO_DDIAG = 0.0
S1_NGO_WORSE_FOLDS = 2  # endpoint better than R-only in <= 2/5 folds -> NO-GO
FINAL_GO_DR = 0.002
FINAL_GO_DDIAG = 0.0015
FINAL_GO_FOLDS = 4
BULK_MAX_DEGRADATION = 0.002
BULK_RARE_PERCENTILE = 80.0

# --- gate tolerances -------------------------------------------------------
FORWARD_GATE_ATOL = 1.0e-9
PAIR_RECON_ATOL = 1.0e-6
R_RECON_ATOL = 1.0e-4
PRED_RECON_ATOL = 1.0e-5
PERM_INV_ATOL = 1.0e-4
BATCH_INV_ATOL = 1.0e-4


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json_any(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_json(value: Any) -> str:
    blob = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _export_path(fold: int, seed: int) -> Path:
    return EXPORT_DIR / f"{EXPORT_VERSION}_fold{fold}_seed{seed}.npz"


def _random_projection() -> np.ndarray:
    """Deterministic 120x16 matrix with orthonormal columns (QR of Gaussian)."""
    rng = np.random.default_rng(PROJECTION_SEED)
    raw = rng.standard_normal((OFFDIAG_DIM, PROJ_DIM))
    q, _r = np.linalg.qr(raw)
    return np.ascontiguousarray(q[:, :PROJ_DIM].astype(np.float32))


def _projection_fingerprint() -> dict[str, Any]:
    projection = _random_projection()
    return {
        "seed": PROJECTION_SEED,
        "kind": "qr_orthonormal_columns_of_standard_normal",
        "shape": list(projection.shape),
        "sha256": _sha256_array(projection),
        "column_norm_min": float(np.abs(np.linalg.norm(projection, axis=0)).min()),
        "column_norm_max": float(np.abs(np.linalg.norm(projection, axis=0)).max()),
        "gram_max_abs_offdiag": float(
            np.abs(projection.T @ projection - np.eye(PROJ_DIM)).max()
        ),
    }


# ---------------------------------------------------------------------------
# Stage 0: v3 export (v2 + projected endpoint state + raw relation)
# ---------------------------------------------------------------------------

def _capture_states_v3(
    model: nn.Module,
    graphs: Sequence[Any],
    batch_size: int = 128,
) -> dict[str, Any]:
    """Eval forward capturing the v2 states plus ``u_i`` and raw relation.

    ``u_i = pair_projection(patch_i)`` where ``patch_i`` is the patch encoder
    output *before* the centre update (exactly the tensor that enters pair
    feature construction).  We capture the centre-update input (whose first
    ``patch_hidden`` columns are that pre-update patch) and apply the frozen
    projection to it, then verify against the forward pair-projection output.
    """
    loader = DataLoader(list(graphs), batch_size=int(batch_size), shuffle=False)
    head_inputs: list[np.ndarray] = []
    pair_values: list[np.ndarray] = []
    pair_inputs_true: list[np.ndarray] = []
    proj_inputs: list[np.ndarray] = []
    proj_outputs: list[np.ndarray] = []
    pre_patches: list[np.ndarray] = []
    patch_deltas: list[np.ndarray] = []
    global_hidden: list[np.ndarray] = []
    topology_hidden: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    def head_pre_hook(_module, args):
        head_inputs.append(args[0].detach().cpu().numpy())

    def pair_hook(_module, _inputs, output):
        pair_values.append(output.detach().cpu().numpy())

    def pair_input_hook(_module, inputs):
        pair_inputs_true.append(inputs[0].detach().cpu().numpy())

    def proj_hook(_module, inputs, output):
        proj_inputs.append(inputs[0].detach().cpu().numpy())
        proj_outputs.append(output.detach().cpu().numpy())

    def center_hook(_module, inputs, output):
        pre_patches.append(inputs[0].detach().cpu().numpy()[:, : model.patch_hidden])
        patch_deltas.append(output.detach().cpu().numpy())

    def global_hook(_module, _inputs, output):
        global_hidden.append(output.detach().cpu().numpy())

    def topology_hook(_module, _inputs, output):
        topology_hidden.append(output.detach().cpu().numpy())

    handles = [
        model.head[0].register_forward_pre_hook(head_pre_hook),
        model.pair_encoder.register_forward_hook(pair_hook),
        model.pair_encoder.register_forward_pre_hook(pair_input_hook),
        model.pair_projection.register_forward_hook(proj_hook),
        model.center_update.register_forward_hook(center_hook),
        model.global_encoder.register_forward_hook(global_hook),
        model.topology_encoder.register_forward_hook(topology_hook),
    ]

    per_graph: list[dict[str, np.ndarray]] = []
    projected_max_diff = 0.0
    pair_input_max_diff = 0.0
    node_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch.device("cpu"))
            predictions.append(model(batch).cpu().numpy())
            ptr = batch.ptr.cpu().numpy()
            n_graphs = int(batch.num_graphs)
            source = batch.pair_index[0].cpu().numpy()
            target = batch.pair_index[1].cpu().numpy()
            pair_graph = batch.batch[batch.pair_index[0]].cpu().numpy()
            source_graph = batch.batch[batch.pair_index[0]].cpu().numpy()
            target_graph = batch.batch[batch.pair_index[1]].cpu().numpy()
            bucket = batch.pair_bucket.cpu().numpy()
            relation = batch.pair_relation.cpu().numpy()
            pre_patch = pre_patches[-1]
            patch_states = pre_patch + patch_deltas[-1]
            # capture the two true forward projections BEFORE the direct call
            # below (which would itself trigger the projection hook).
            left = proj_outputs[-2]
            right = proj_outputs[-1]
            # u_i = P(pre-update patch_i): exact same linear map the forward uses.
            u_all = model.pair_projection(
                torch.tensor(pre_patch, dtype=torch.float32)
            ).numpy()
            # G0.1: exported u_i == the tensor entering pair feature construction.
            projected_max_diff = max(
                projected_max_diff,
                float(np.abs(u_all[source] - left).max()),
                float(np.abs(u_all[target] - right).max()),
            )
            # G0.4: reconstruct the pair encoder input from exported pieces and
            # compare against the true forward pair input (captured by hook).
            rebuilt = _reconstruct_pair_input(
                u_all,
                source,
                target,
                torch.tensor(relation, dtype=torch.float32),
                torch.tensor(bucket, dtype=torch.long),
                model,
            )
            pair_input_max_diff = max(
                pair_input_max_diff,
                float(np.abs(rebuilt.numpy() - pair_inputs_true[-1]).max()),
            )
            for graph_id in range(n_graphs):
                start, end = int(ptr[graph_id]), int(ptr[graph_id + 1])
                pair_mask = pair_graph == graph_id
                per_graph.append(
                    {
                        "n_patches": np.int64(end - start),
                        "patch_states": patch_states[start:end].astype(np.float32),
                        "projected_patch_state": u_all[start:end].astype(np.float32),
                        "pair_states": pair_values[-1][pair_mask].astype(np.float32),
                        "pair_relation": relation[pair_mask].astype(np.float32),
                        "pair_bucket": bucket[pair_mask].astype(np.int64),
                        "pair_source": (source[pair_mask] + node_offset).astype(np.int64),
                        "pair_target": (target[pair_mask] + node_offset).astype(np.int64),
                        "pair_source_local": (source[pair_mask] - start).astype(np.int64),
                        "pair_target_local": (target[pair_mask] - start).astype(np.int64),
                        "pair_source_graph": source_graph[pair_mask].astype(np.int64),
                        "pair_target_graph": target_graph[pair_mask].astype(np.int64),
                        "global_hidden": global_hidden[-1][graph_id].astype(np.float32),
                        "topology_hidden": topology_hidden[-1][graph_id].astype(np.float32),
                        "R": head_inputs[-1][graph_id].astype(np.float32),
                    }
                )
            node_offset += int(batch.num_nodes)
    for handle in handles:
        handle.remove()

    capture = {
        "R": np.stack([g["R"] for g in per_graph], axis=0),
        "yhat_0": np.concatenate(predictions).astype(np.float64),
        "n_patches": np.asarray([g["n_patches"] for g in per_graph], dtype=np.int64),
        "n_pairs": np.asarray([g["pair_states"].shape[0] for g in per_graph], dtype=np.int64),
        "patch_states": np.concatenate([g["patch_states"] for g in per_graph], axis=0),
        "projected_patch_state": np.concatenate(
            [g["projected_patch_state"] for g in per_graph], axis=0
        ),
        "pair_states": np.concatenate([g["pair_states"] for g in per_graph], axis=0),
        "pair_relation": np.concatenate([g["pair_relation"] for g in per_graph], axis=0),
        "pair_bucket": np.concatenate([g["pair_bucket"] for g in per_graph], axis=0),
        "pair_source": np.concatenate([g["pair_source"] for g in per_graph], axis=0),
        "pair_target": np.concatenate([g["pair_target"] for g in per_graph], axis=0),
        "pair_source_local": np.concatenate([g["pair_source_local"] for g in per_graph], axis=0),
        "pair_target_local": np.concatenate([g["pair_target_local"] for g in per_graph], axis=0),
        "pair_source_graph": np.concatenate([g["pair_source_graph"] for g in per_graph], axis=0),
        "pair_target_graph": np.concatenate([g["pair_target_graph"] for g in per_graph], axis=0),
        "global_hidden": np.stack([g["global_hidden"] for g in per_graph], axis=0),
        "topology_hidden": np.stack([g["topology_hidden"] for g in per_graph], axis=0),
    }
    return {
        "capture": capture,
        "projected_max_diff": float(projected_max_diff),
        "pair_input_max_diff": float(pair_input_max_diff),
    }


def _reconstruct_pair_input(
    u_all: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    relation_raw: torch.Tensor,
    bucket: torch.Tensor,
    model: nn.Module,
) -> torch.Tensor:
    """Rebuild the exact pair encoder input from u_i, u_j, relation, bucket."""
    u = torch.tensor(np.asarray(u_all), dtype=torch.float32)
    left = u[torch.tensor(np.asarray(source), dtype=torch.long)]
    right = u[torch.tensor(np.asarray(target), dtype=torch.long)]
    relation = model.relation_encoder(relation_raw)
    gate = 1.0 + torch.tanh(model.distance_gate(bucket))
    product = left * right
    return torch.cat(
        [
            left + right,
            torch.abs(left - right),
            product * gate,
            relation,
        ],
        dim=1,
    )


def _fold_fingerprint_inputs_v3(fold: int, seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    records = _load_train_records()
    inner_train_idx, inner_valid_idx, holdout_idx = _fold_slices()[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    fold_meta = _read_json(_fold_marker(fold, seed))
    representation = config["representation"]
    typed = zpp._fit_vocabulary(
        inner_train_records, "typed_certificate",
        int(representation.get("max_typed_tokens", 8192)),
        int(representation.get("minimum_typed_frequency", 1)),
    )
    parent = zpp._fit_vocabulary(
        inner_train_records, "parent_certificate",
        int(representation.get("max_parent_tokens", 2048)),
        int(representation.get("minimum_parent_frequency", 1)),
    )
    typed_blob = b"\n".join(k for k, _ in sorted(typed.items(), key=lambda kv: kv[1]))
    parent_blob = b"\n".join(k for k, _ in sorted(parent.items(), key=lambda kv: kv[1]))
    payload = {
        "export_version": EXPORT_VERSION,
        "tokenizer_version": "typed_tokenizer_v1_historical",
        "config_fingerprint": _hash_json(config),
        "vocabulary_fingerprint": {
            "typed_vocabulary_size": int(len(typed)),
            "parent_vocabulary_size": int(len(parent)),
            "typed_sha256": hashlib.sha256(typed_blob).hexdigest(),
            "parent_sha256": hashlib.sha256(parent_blob).hexdigest(),
        },
        "split_fingerprint": {
            "inner_train_sha256": _sha256_array(inner_train_idx),
            "inner_valid_sha256": _sha256_array(inner_valid_idx),
            "holdout_sha256": _sha256_array(holdout_idx),
        },
        "checkpoint_fingerprint": str(fold_meta["state_pt_sha256"]),
        "projection_fingerprint": _projection_fingerprint(),
        "backbone_seed": int(seed),
        "fold": int(fold),
    }
    payload["fingerprint"] = _hash_json(payload)
    return payload


def _build_export(fold: int, seed: int, batch_size: int = 128) -> dict[str, Any]:
    records = _load_train_records()
    labels = _load_train_labels()
    _verify_records_vs_labels(records, labels)
    config = _frozen_config()
    torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", 4)))
    inner_train_idx, _inner_valid_idx, holdout_idx = _fold_slices()[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    holdout_records = [records[int(i)] for i in holdout_idx]
    _encoded_fit, encoded_holdout, audit = zpp._phase_data(
        inner_train_records, holdout_records, config=config
    )
    config_mod = dict(config)
    config_mod["model"] = _model_config_with_clamps(config, audit)
    model = _build_v4_model(config_mod, audit)
    state = torch.load(
        FOLD_DIR / f"fold{fold}_seed{seed}_state.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    if model.head[0].in_features != R_DIM:
        raise AssertionError(f"pre-head R width {model.head[0].in_features} != {R_DIM}")
    if model.patch_hidden != PATCH_DIM or model.pair_hidden != PAIR_DIM:
        raise AssertionError("frozen patch/pair dims changed")

    result = _capture_states_v3(model, encoded_holdout, batch_size=batch_size)
    capture = result["capture"]
    if result["projected_max_diff"] > PAIR_RECON_ATOL:
        raise AssertionError(
            f"fold {fold} seed {seed}: exported u_i deviates from forward "
            f"pair-projection state (max {result['projected_max_diff']})"
        )
    if result["pair_input_max_diff"] > PAIR_RECON_ATOL:
        raise AssertionError(
            f"fold {fold} seed {seed}: reconstructed pair input deviates from "
            f"forward pair input (max {result['pair_input_max_diff']})"
        )
    npz = _load_fold_npz(fold, seed)
    forward_gate = float(np.abs(capture["yhat_0"] - npz["oof_prediction"]).max())
    if forward_gate > FORWARD_GATE_ATOL:
        raise AssertionError(
            f"fold {fold} seed {seed}: forward predictions deviate from fold npz "
            f"(max {forward_gate})"
        )
    target = np.asarray([float(record.y) for record in holdout_records], dtype=np.float64)
    fingerprint = _fold_fingerprint_inputs_v3(fold, seed, config)
    fingerprint["forward_gate_max_diff"] = forward_gate
    fingerprint["projected_gate_max_diff"] = result["projected_max_diff"]
    fingerprint["pair_input_gate_max_diff"] = result["pair_input_max_diff"]
    return {
        "subset_index": holdout_idx.astype(np.int64),
        "target": target,
        "fingerprint": fingerprint,
        **capture,
    }


def _save_export(path: Path, export: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, value in export.items():
        if key == "fingerprint":
            arrays["fingerprint_json"] = np.asarray(json.dumps(_jsonable(value), sort_keys=True))
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_export(path: Path) -> dict[str, Any]:
    """Load a v3 export, refusing any legacy/foreign cache."""
    with np.load(path, allow_pickle=False) as data:
        if "fingerprint_json" not in data.files:
            raise RuntimeError(f"cache {path} has no fingerprint; refusing to load")
        fingerprint = json.loads(str(data["fingerprint_json"]))
        version = fingerprint.get("export_version")
        if version != EXPORT_VERSION:
            raise RuntimeError(
                f"cache {path} export_version={version!r} != required "
                f"{EXPORT_VERSION!r}; refusing to load (legacy versions: "
                f"{sorted(LEGACY_EXPORT_VERSIONS)})"
            )
        export: dict[str, Any] = {"fingerprint": fingerprint}
        for key in data.files:
            if key == "fingerprint_json":
                continue
            export[key] = data[key]
    return export


def stage_export(seeds: Sequence[int] = (PRIMARY_BACKBONE_SEED,), force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    built: dict[str, Any] = {}
    for seed in seeds:
        for fold in range(K_FOLDS):
            path = _export_path(fold, seed)
            if path.exists() and not force:
                built[f"fold{fold}_seed{seed}"] = {"path": str(path), "cached": True}
                continue
            t0 = time.perf_counter()
            export = _build_export(fold, seed)
            _save_export(path, export)
            built[f"fold{fold}_seed{seed}"] = {
                "path": str(path),
                "cached": False,
                "seconds": time.perf_counter() - t0,
                "n": int(len(export["subset_index"])),
                "forward_gate_max_diff": float(export["fingerprint"]["forward_gate_max_diff"]),
                "projected_gate_max_diff": float(export["fingerprint"]["projected_gate_max_diff"]),
                "pair_input_gate_max_diff": float(export["fingerprint"]["pair_input_gate_max_diff"]),
            }
            print(
                f"[export] fold={fold} seed={seed} n={len(export['subset_index'])} "
                f"gate_y={export['fingerprint']['forward_gate_max_diff']:.1e} "
                f"gate_u={export['fingerprint']['projected_gate_max_diff']:.1e} "
                f"gate_pair={export['fingerprint']['pair_input_gate_max_diff']:.1e} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    inventory = checkpoint_inventory()
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    summary = {
        "export_version": EXPORT_VERSION,
        "seeds": [int(s) for s in seeds],
        "exports": built,
        "checkpoint_inventory": inventory,
        "projection_fingerprint": _projection_fingerprint(),
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(RESULTS_DIR / "stage_export.json", summary)
    return summary


def checkpoint_inventory() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            marker = _fold_marker(fold, seed)
            if not marker.exists():
                entries.append({"fold": fold, "seed": seed, "status": "missing"})
                continue
            meta = _read_json(marker)
            entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "status": "present",
                    "state_pt": str(FOLD_DIR / f"fold{fold}_seed{seed}_state.pt"),
                    "state_pt_sha256": meta.get("state_pt_sha256"),
                    "parameters": meta.get("parameters"),
                    "selected_epoch": meta.get("selected_epoch"),
                    "oof_mae": meta.get("oof_mae"),
                }
            )
    present = [e for e in entries if e["status"] == "present"]
    complete_seeds = sorted(
        {e["seed"] for e in present if sum(1 for x in present if x["seed"] == e["seed"]) == K_FOLDS}
    )
    return {
        "search_dir": str(FOLD_DIR),
        "n_entries": len(entries),
        "n_present": len(present),
        "complete_seeds": complete_seeds,
        "entries": entries,
        "frozen_config_path": str(CONFIG_PATH),
        "missing_checkpoint_policy": (
            "STOP and report a missing-checkpoint inventory; never silently "
            "convert the frozen witness diagnostic into fresh backbone training."
        ),
    }


# ---------------------------------------------------------------------------
# pair feature spec
# ---------------------------------------------------------------------------

def write_pair_feature_spec() -> dict[str, Any]:
    config = _frozen_config()
    spec = {
        "model": "compact-v4-hinge (frozen)",
        "patch_state_dim": PATCH_DIM,
        "pair_projection": "Linear(patch_hidden -> pair_hidden, bias=False)",
        "pair_hidden_dim": PAIR_DIM,
        "u_i": "pair_projection(patch_i), 16D, computed on the pre-centre-update patch state",
        "relation_descriptor_width": RELATION_DIM,
        "relation_encoder": "MLPBlock(23 -> 32 -> 16, dropout 0.05)",
        "relation_layout": {
            "distance_bucket_one_hot": N_BUCKETS,
            "log1p_shortest_path_distance": 1,
            "patch_overlap_size_features": 5,
            "boundary_containment_features": 3,
            "path_bond_composition": zpp.BOND_CATEGORIES,
            "log1p_shortest_path_count": 1,
            "adjacent_bond_one_hot": zpp.BOND_CATEGORIES,
        },
        "exact_distance_feature": "relation[5] = log1p(shortest-path distance)",
        "bucket_usage": {
            "n_buckets": N_BUCKETS,
            "definition": "bucket = min(max(distance,1),5)-1 (5 buckets for distances 1,2,3,4,5+)",
        },
        "distance_gate": "Embedding(5 -> 16); gate = 1 + tanh(distance_gate(bucket))",
        "pair_input_width": 4 * PAIR_DIM,
        "pair_input_blocks": [
            "u_i + u_j (16)",
            "abs(u_i - u_j) (16)",
            "(u_i * u_j) * gate (16)",
            "relation_encoder(pair_relation) (16)",
        ],
        "pair_encoder": "MLPBlock(64 -> 64 -> 16, dropout 0.05)",
        "q_ij_dim": PAIR_DIM,
        "readout": (
            "pair_readout=moments -> per distance bucket [sum ; sum of squares ; "
            "log1p(count)] over q_ij, 5 x 33 = 165 of the 302D pre-head R"
        ),
        "center_context": bool(config["model"].get("center_context", False)),
        "center_update": (
            "patch <- patch + MLP([patch ; per-centre per-bucket mean/std/log1p-count "
            "of incident q_ij]) (zero-init final projection)"
        ),
        "endpoint_witness": {
            "offdiag_dim": OFFDIAG_DIM,
            "definition": "strictly upper-triangular (k<l) of A_ij = u_i u_j^T + u_j u_i^T",
            "diagonal_note": "k=l entries are 2*(u_i (*) u_j), already covered by the third block",
            "projection": "deterministic orthonormal 120x16, seed-frozen, not learned",
            "graph_summary": "per distance bucket mean(16) + std(16) -> 5 x 32 = 160",
        },
        "diag_control": {
            "definition": "d_ij = u_i (*) u_j (16D, already directly available to the current pair encoder)",
            "graph_summary": "per distance bucket mean(16) + std(16) -> 5 x 32 = 160",
        },
    }
    _write_json_any(RESULTS_DIR / "pair_feature_spec.json", spec)
    return spec


# ---------------------------------------------------------------------------
# Gate 0 integrity checks
# ---------------------------------------------------------------------------

def _reconstruct_R(export: Mapping[str, Any]) -> np.ndarray:
    n = len(export["subset_index"])
    patch = export["patch_states"]
    pair = export["pair_states"]
    bucket = export["pair_bucket"]
    n_patches = export["n_patches"]
    n_pairs = export["n_pairs"]
    patch_offsets = np.concatenate([[0], np.cumsum(n_patches)])
    pair_offsets = np.concatenate([[0], np.cumsum(n_pairs)])
    unary = np.zeros((n, 2 * PATCH_DIM + 1), dtype=np.float64)
    relation = np.zeros((n, N_BUCKETS * (2 * PAIR_DIM + 1)), dtype=np.float64)
    for position in range(n):
        p0, p1 = int(patch_offsets[position]), int(patch_offsets[position + 1])
        values = patch[p0:p1].astype(np.float64)
        unary[position, :PATCH_DIM] = values.sum(axis=0)
        unary[position, PATCH_DIM: 2 * PATCH_DIM] = (values * values).sum(axis=0)
        unary[position, -1] = np.log1p(p1 - p0)
        q0, q1 = int(pair_offsets[position]), int(pair_offsets[position + 1])
        qvalues = pair[q0:q1].astype(np.float64)
        qbucket = bucket[q0:q1]
        for b in range(N_BUCKETS):
            mask = qbucket == b
            current = qvalues[mask]
            base = b * (2 * PAIR_DIM + 1)
            relation[position, base: base + PAIR_DIM] = current.sum(axis=0)
            relation[position, base + PAIR_DIM: base + 2 * PAIR_DIM] = (current * current).sum(axis=0)
            relation[position, base + 2 * PAIR_DIM] = np.log1p(int(mask.sum()))
    return np.concatenate(
        [unary, relation, export["global_hidden"].astype(np.float64),
         export["topology_hidden"].astype(np.float64)],
        axis=1,
    ).astype(np.float32)


def _gate_pair_grouping(export: Mapping[str, Any]) -> dict[str, Any]:
    n_patches = export["n_patches"]
    expected = n_patches * (n_patches - 1) // 2
    actual = export["n_pairs"]
    return {
        "gate": "G0.1_pair_graph_grouping",
        "n_graphs": int(len(n_patches)),
        "n_mismatch": int(np.sum(expected != actual)),
        "passed": bool(np.array_equal(expected, actual)),
    }


def _gate_projected_state_export(export: Mapping[str, Any]) -> dict[str, Any]:
    """G0.2: each graph's u_i rows correspond one-to-one to its patch nodes."""
    n_patches = export["n_patches"]
    u = export["projected_patch_state"]
    offsets = np.concatenate([[0], np.cumsum(n_patches)])
    ok_rows = int(u.shape[0]) == int(offsets[-1])
    ok_dim = int(u.shape[1]) == PAIR_DIM
    return {
        "gate": "G0.2_projected_state_patch_correspondence",
        "n_patch_rows": int(u.shape[0]),
        "expected_rows": int(offsets[-1]),
        "dim": int(u.shape[1]),
        "passed": bool(ok_rows and ok_dim),
    }


def _gate_pair_endpoint_identity(export: Mapping[str, Any]) -> dict[str, Any]:
    same = export["pair_source_graph"] == export["pair_target_graph"]
    src = export["pair_source"]
    tgt = export["pair_target"]
    n_patches = export["n_patches"]
    pair_offsets = np.concatenate([[0], np.cumsum(export["n_pairs"])])
    patch_offsets = np.concatenate([[0], np.cumsum(n_patches)])
    within = True
    for position in range(len(n_patches)):
        start, end = int(patch_offsets[position]), int(patch_offsets[position + 1])
        q0, q1 = int(pair_offsets[position]), int(pair_offsets[position + 1])
        if q1 > q0:
            within &= bool(src[q0:q1].min() >= start and src[q0:q1].max() < end)
            within &= bool(tgt[q0:q1].min() >= start and tgt[q0:q1].max() < end)
    return {
        "gate": "G0.3_pair_endpoint_identity_and_indexable",
        "n_violations_graph_id": int(np.sum(~same)),
        "indexable": bool(within),
        "passed": bool(np.all(same) and within),
    }


def _gate_pair_coverage(export: Mapping[str, Any]) -> dict[str, Any]:
    n = len(export["subset_index"])
    n_pairs = export["n_pairs"]
    pair_offsets = np.concatenate([[0], np.cumsum(n_pairs)])
    duplicates = 0
    self_loops = 0
    count_mismatch = 0
    n_patches = export["n_patches"]
    src = export["pair_source_local"]
    dst = export["pair_target_local"]
    for position in range(n):
        q0, q1 = int(pair_offsets[position]), int(pair_offsets[position + 1])
        s = src[q0:q1]
        d = dst[q0:q1]
        size = int(n_patches[position])
        self_loops += int(np.sum(s == d))
        keys = np.minimum(s, d).astype(np.int64) * (size + 2) + np.maximum(s, d)
        duplicates += int(len(keys) - len(np.unique(keys)))
        if q1 - q0 != size * (size - 1) // 2:
            count_mismatch += 1
    return {
        "gate": "G0.4_pair_coverage",
        "duplicate_rows": int(duplicates),
        "self_loops": int(self_loops),
        "count_mismatch": int(count_mismatch),
        "passed": bool(duplicates == 0 and self_loops == 0 and count_mismatch == 0),
    }


def _gate_reconstruct_pair_states(export: Mapping[str, Any], model: nn.Module) -> dict[str, Any]:
    """G0.5/G0.6: rebuild pair input from u_i,u_j,relation and match q_ij."""
    u = export["projected_patch_state"]
    source = export["pair_source"]
    target = export["pair_target"]
    relation = torch.tensor(export["pair_relation"], dtype=torch.float32)
    bucket = torch.tensor(export["pair_bucket"], dtype=torch.long)
    with torch.no_grad():
        pair_input = _reconstruct_pair_input(u, source, target, relation, bucket, model)
        q = model.pair_encoder(pair_input).numpy()
    diff = np.abs(q.astype(np.float64) - export["pair_states"].astype(np.float64))
    return {
        "gate": "G0.5_reconstruct_pair_states",
        "pair_input_dim": int(pair_input.shape[1]),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "atol": PAIR_RECON_ATOL,
        "passed": bool(diff.max() <= PAIR_RECON_ATOL),
    }


def _gate_reconstruct_R(export: Mapping[str, Any]) -> dict[str, Any]:
    reconstructed = _reconstruct_R(export)
    diff = np.abs(reconstructed.astype(np.float64) - export["R"].astype(np.float64))
    return {
        "gate": "G0.6_reconstruct_R",
        "max_abs_diff": float(diff.max()),
        "atol": R_RECON_ATOL,
        "passed": bool(diff.max() <= R_RECON_ATOL),
    }


def _gate_reconstruct_prediction(export: Mapping[str, Any], model: nn.Module) -> dict[str, Any]:
    reconstructed = _reconstruct_R(export)
    model.eval()
    with torch.no_grad():
        yhat = model.head(torch.tensor(reconstructed, dtype=torch.float32)).view(-1).numpy()
    diff = np.abs(yhat.astype(np.float64) - export["yhat_0"].astype(np.float64))
    return {
        "gate": "G0.6b_reconstruct_prediction",
        "max_abs_diff": float(diff.max()),
        "atol": PRED_RECON_ATOL,
        "passed": bool(diff.max() <= PRED_RECON_ATOL),
    }


def _load_model_for_fold(fold: int, seed: int) -> nn.Module:
    records = _load_train_records()
    config = _frozen_config()
    inner_train_idx, _iv, holdout_idx = _fold_slices()[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    holdout_records = [records[int(i)] for i in holdout_idx]
    _encoded_fit, _encoded_other, audit = zpp._phase_data(
        inner_train_records, holdout_records, config=config
    )
    config_mod = dict(config)
    config_mod["model"] = _model_config_with_clamps(config, audit)
    model = _build_v4_model(config_mod, audit)
    state = torch.load(FOLD_DIR / f"fold{fold}_seed{seed}_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _fold_transforms(fold: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

    records = _load_train_records()
    config = _frozen_config()
    inner_train_idx = _fold_slices()[fold][0]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    representation = config["representation"]
    transforms = {
        "typed_vocabulary": zpp._fit_vocabulary(
            inner_train_records, "typed_certificate",
            int(representation.get("max_typed_tokens", 8192)),
            int(representation.get("minimum_typed_frequency", 1)),
        ),
        "parent_vocabulary": zpp._fit_vocabulary(
            inner_train_records, "parent_certificate",
            int(representation.get("max_parent_tokens", 2048)),
            int(representation.get("minimum_parent_frequency", 1)),
        ),
        "patch_standardizer": zpp.Standardizer.fit(zpp._patch_matrix(inner_train_records)),
        "context_standardizer": zpp.Standardizer.fit(zpp._context_matrix(inner_train_records)),
    }
    topology_matrix = ztopo.matrices_for_split("train", _load_zinc(zpp.REPO_ROOT / "data/ZINC", "train"), "hinge")[0]
    transforms["topology_standardizer"] = zpp.Standardizer.fit(
        np.stack([record.topology_features for record in inner_train_records], axis=0).astype(np.float32)
    )
    return transforms, {"topology_matrix": topology_matrix, "config": config}


def _encode_with_transforms(records: Sequence[Any], transforms: Mapping[str, Any]) -> list[Any]:
    return zpp._encode_records(
        records,
        transforms["typed_vocabulary"],
        transforms["parent_vocabulary"],
        transforms["patch_standardizer"],
        transforms["context_standardizer"],
        None,
        structural_mode="none",
        structural_vocabulary=None,
        topology_mode="hinge",
        topology_standardizer=transforms["topology_standardizer"],
        attribute_mode="none",
    )


def _witness_for_single_molecule(
    u: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    bucket: np.ndarray,
    projection: np.ndarray,
    a_stat: tuple[np.ndarray, np.ndarray],
    d_stat: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-molecule endpoint / diagonal graph summaries (160D each)."""
    u = np.asarray(u, dtype=np.float64)
    left = u[source]
    right = u[target]
    a = _offdiag_outer(left, right)
    d = left * right
    a_mean, a_scale = a_stat
    d_mean, d_scale = d_stat
    z = ((a - a_mean) / a_scale) @ projection.astype(np.float64)
    d_std_in = (d - d_mean) / d_scale
    return (
        _bucket_mean_std(z, bucket),
        _bucket_mean_std(d_std_in, bucket),
    )


def _offdiag_outer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    outer = left[:, :, None] * right[:, None, :] + right[:, :, None] * left[:, None, :]
    iu = np.triu_indices(left.shape[1], k=1)
    return outer[:, iu[0], iu[1]]


def _bucket_mean_std(values: np.ndarray, bucket: np.ndarray) -> np.ndarray:
    dim = int(values.shape[1])
    out = np.zeros((N_BUCKETS, 2 * dim), dtype=np.float64)
    for b in range(N_BUCKETS):
        mask = bucket == b
        current = values[mask]
        if current.size:
            out[b, :dim] = current.mean(axis=0)
            out[b, dim:] = np.sqrt(current.var(axis=0) + 1.0e-8)
    return out.reshape(-1)


def _gate_node_permutation(fold: int, seed: int, n_molecules: int = 4) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc, global_feature_views
    from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import TYPED_TOKENIZER_V1_HISTORICAL
    from torch_geometric.data import Data as PyGData

    transforms, extra = _fold_transforms(fold)
    topology_matrix = extra["topology_matrix"]
    records = _load_train_records()
    holdout_idx = _fold_slices()[fold][2]
    dataset = _load_zinc(zpp.REPO_ROOT / "data/ZINC", "train")
    model = _load_model_for_fold(fold, seed)
    projection = _random_projection()
    a_stat = (np.zeros(OFFDIAG_DIM), np.ones(OFFDIAG_DIM))
    d_stat = (np.zeros(PAIR_DIM), np.ones(PAIR_DIM))

    rc = np.random.RandomState(0)
    chosen = sorted(np.random.RandomState(7).choice(len(holdout_idx), size=n_molecules, replace=False).tolist())
    diffs = {"R": 0.0, "yhat_0": 0.0, "witness": 0.0}
    for position in chosen:
        subset_index = int(holdout_idx[position])
        original = records[subset_index]
        data = dataset[subset_index]
        n = int(data.num_nodes)
        perm = rc.permutation(n)
        inv = np.argsort(perm)
        edge_index = data.edge_index.detach().cpu().numpy()
        permuted_edge_index = perm[edge_index]
        permuted_data = PyGData(
            x=data.x[torch.tensor(inv, dtype=torch.long)],
            edge_index=torch.tensor(permuted_edge_index, dtype=torch.long),
            edge_attr=data.edge_attr,
            y=data.y,
            num_nodes=n,
        )
        global_context = global_feature_views([permuted_data])["global_all"][0]
        permuted_record = zpp._graph_record(
            permuted_data, global_context, {}, patch_radius=zpp.PATCH_RADIUS,
            context_radius=0, structural_mode="none", topology_features=topology_matrix[subset_index],
            tokenizer_version=TYPED_TOKENIZER_V1_HISTORICAL, attribute_mode="none",
        )
        encoded_original = _encode_with_transforms([original], transforms)[0]
        encoded_permuted = _encode_with_transforms([permuted_record], transforms)[0]
        cap_original = _capture_states_v3(model, [encoded_original], batch_size=1)["capture"]
        cap_permuted = _capture_states_v3(model, [encoded_permuted], batch_size=1)["capture"]
        diffs["R"] = max(diffs["R"], float(np.abs(cap_original["R"] - cap_permuted["R"]).max()))
        diffs["yhat_0"] = max(diffs["yhat_0"], float(np.abs(cap_original["yhat_0"] - cap_permuted["yhat_0"]).max()))
        w_orig, d_orig = _witness_from_capture(cap_original, projection, a_stat, d_stat)
        w_perm, d_perm = _witness_from_capture(cap_permuted, projection, a_stat, d_stat)
        diffs["witness"] = max(
            diffs["witness"],
            float(np.abs(w_orig - w_perm).max()),
            float(np.abs(d_orig - d_perm).max()),
        )
    return {
        "gate": "G0.7_node_permutation_invariance",
        "fold": int(fold),
        "seed": int(seed),
        "n_molecules": n_molecules,
        "max_abs_diff": diffs,
        "atol": PERM_INV_ATOL,
        "passed": bool(max(diffs.values()) <= PERM_INV_ATOL),
    }


def _witness_from_capture(
    capture: Mapping[str, Any],
    projection: np.ndarray,
    a_stat: tuple[np.ndarray, np.ndarray],
    d_stat: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    u = capture["projected_patch_state"]
    return _witness_for_single_molecule(
        u, capture["pair_source"], capture["pair_target"], capture["pair_bucket"],
        projection, a_stat, d_stat,
    )


def _gate_batch_invariance(fold: int, seed: int) -> dict[str, Any]:
    records = _load_train_records()
    config = _frozen_config()
    inner_train_idx, _iv, holdout_idx = _fold_slices()[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    holdout_records = [records[int(i)] for i in holdout_idx]
    _ef, encoded_holdout, audit = zpp._phase_data(inner_train_records, holdout_records, config=config)
    config_mod = dict(config)
    config_mod["model"] = _model_config_with_clamps(config, audit)
    model = _build_v4_model(config_mod, audit)
    state = torch.load(FOLD_DIR / f"fold{fold}_seed{seed}_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    baseline = _capture_states_v3(model, encoded_holdout, batch_size=128)["capture"]
    order = np.arange(len(encoded_holdout))[::-1].copy()
    reordered = [encoded_holdout[int(i)] for i in order]
    shifted = _capture_states_v3(model, reordered, batch_size=97)["capture"]
    inv_order = np.argsort(order)

    offsets_b = np.concatenate([[0], np.cumsum(baseline["n_patches"])])
    offsets_s = np.concatenate([[0], np.cumsum(shifted["n_patches"])])

    def _patch_mean(capture, offsets, positions):
        return np.stack([capture["projected_patch_state"][offsets[p]:offsets[p + 1]].mean(axis=0) for p in positions])

    base_u = _patch_mean(baseline, offsets_b, np.arange(len(encoded_holdout)))
    other_u = _patch_mean(shifted, offsets_s, inv_order)
    diffs = {
        "R": float(np.abs(baseline["R"] - shifted["R"][inv_order]).max()),
        "yhat_0": float(np.abs(baseline["yhat_0"] - shifted["yhat_0"][inv_order]).max()),
        "projected_patch_mean": float(np.abs(base_u - other_u).max()),
    }
    return {
        "gate": "G0.8_batch_invariance",
        "fold": int(fold),
        "seed": int(seed),
        "batch_sizes": [128, 97],
        "max_abs_diff": diffs,
        "atol": BATCH_INV_ATOL,
        "passed": bool(max(diffs.values()) <= BATCH_INV_ATOL),
    }


def run_gate0(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = RESULTS_DIR / "export_integrity_report.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    seeds_to_check = [s for s in BACKBONE_SEEDS if _export_path(0, s).exists()]
    per_fold: dict[str, Any] = {}
    for seed in seeds_to_check:
        for fold in range(K_FOLDS):
            export = load_export(_export_path(fold, seed))
            model = _load_model_for_fold(fold, seed)
            gates = {
                "G0.1": _gate_pair_grouping(export),
                "G0.2": _gate_projected_state_export(export),
                "G0.3": _gate_pair_endpoint_identity(export),
                "G0.4": _gate_pair_coverage(export),
                "G0.5": _gate_reconstruct_pair_states(export, model),
                "G0.6": _gate_reconstruct_R(export),
                "G0.6b": _gate_reconstruct_prediction(export, model),
            }
            per_fold[f"fold{fold}_seed{seed}"] = gates
            print(
                f"[gate0] fold={fold} seed={seed} "
                + " ".join(f"{k}={'PASS' if v['passed'] else 'FAIL'}" for k, v in gates.items()),
                flush=True,
            )
    batch_inv = _gate_batch_invariance(0, PRIMARY_BACKBONE_SEED) if _export_path(0, PRIMARY_BACKBONE_SEED).exists() else {"passed": False}
    perm_inv = _gate_node_permutation(0, PRIMARY_BACKBONE_SEED) if _export_path(0, PRIMARY_BACKBONE_SEED).exists() else {"passed": False}
    all_gates = [g for gates in per_fold.values() for g in gates.values()]
    all_gates.extend([batch_inv, perm_inv])
    failures = [g["gate"] for g in all_gates if not g.get("passed", False)]
    report = {
        "export_version": EXPORT_VERSION,
        "seeds_checked": [int(s) for s in seeds_to_check],
        "per_fold": per_fold,
        "batch_invariance": batch_inv,
        "node_permutation_invariance": perm_inv,
        "n_gates": len(all_gates),
        "failures": failures,
        "all_passed": len(failures) == 0,
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_path, report)
    print(f"[gate0] all_passed={report['all_passed']} failures={failures}", flush=True)
    return report


# ---------------------------------------------------------------------------
# split manifest (identical to frozen readout sufficiency audit)
# ---------------------------------------------------------------------------

def _hash_split(molecule_ids: Sequence[str]) -> np.ndarray:
    keys = np.asarray(
        [int(hashlib.sha256(f"{SPLIT_SEED}|{mid}".encode()).hexdigest()[:16], 16) for mid in molecule_ids],
        dtype=np.float64,
    )
    order = np.argsort(keys, kind="stable")
    roles = np.empty(len(molecule_ids), dtype=object)
    labels = ("adapter_fit", "adapter_selection", "adapter_evaluation")
    cursor = 0
    for label, size in zip(labels, SPLIT_SIZES):
        roles[order[cursor: cursor + size]] = label
        cursor += size
    return roles


def _fold_split_manifest(fold: int) -> dict[str, Any]:
    holdout_idx = _fold_slices()[fold][2]
    molecule_ids = [f"train:{int(i):04d}" for i in holdout_idx]
    roles = _hash_split(molecule_ids)
    rarity = pd.read_csv(RARITY_CSV).set_index("subset_index")
    rare_ratio = rarity.loc[holdout_idx, "rare_le5_ratio"].to_numpy(dtype=np.float64)
    fit_mask = roles == "adapter_fit"
    threshold = float(np.percentile(rare_ratio[fit_mask], BULK_RARE_PERCENTILE))
    eval_mask = roles == "adapter_evaluation"
    bulk = eval_mask & (rare_ratio < threshold)
    molecule_array = np.asarray(molecule_ids)
    return {
        "fold": int(fold),
        "split_seed": SPLIT_SEED,
        "sizes": dict(zip(("adapter_fit", "adapter_selection", "adapter_evaluation"), SPLIT_SIZES)),
        "assignment_sha256": hashlib.sha256("".join(roles.tolist()).encode()).hexdigest(),
        "rare_le5_percentile": BULK_RARE_PERCENTILE,
        "rare_le5_threshold": threshold,
        "n_bulk_eval": int(bulk.sum()),
        "molecule_ids": {
            label: molecule_array[np.flatnonzero(roles == label)].tolist()
            for label in ("adapter_fit", "adapter_selection", "adapter_evaluation")
        },
    }


def stage_splits(force: bool = False) -> dict[str, Any]:
    manifest = {"split_seed": SPLIT_SEED, "sizes": SPLIT_SIZES, "folds": {}}
    for fold in range(K_FOLDS):
        manifest["folds"][f"fold{fold}"] = _fold_split_manifest(fold)
    _write_json_any(RESULTS_DIR / "fold_split_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# witness bank (per fold: endpoint + diagonal graph summaries)
# ---------------------------------------------------------------------------

class FoldWitness:
    def __init__(self, export: Mapping[str, Any], projection: np.ndarray) -> None:
        self.export = export
        self.projection = np.asarray(projection, dtype=np.float32)
        self.subset_index = np.asarray(export["subset_index"], dtype=np.int64)
        self.n_patches = np.asarray(export["n_patches"], dtype=np.int64)
        self.n_pairs = np.asarray(export["n_pairs"], dtype=np.int64)
        self.u = np.asarray(export["projected_patch_state"], dtype=np.float64)
        self.patch_offsets = np.concatenate([[0], np.cumsum(self.n_patches)])
        self.pair_offsets = np.concatenate([[0], np.cumsum(self.n_pairs)])
        self.pair_source = np.asarray(export["pair_source"], dtype=np.int64)
        self.pair_target = np.asarray(export["pair_target"], dtype=np.int64)
        self.pair_bucket = np.asarray(export["pair_bucket"], dtype=np.int64)
        self.R = np.asarray(export["R"], dtype=np.float32)
        self.y = np.asarray(export["target"], dtype=np.float64)
        self.yhat_0 = np.asarray(export["yhat_0"], dtype=np.float64)

    def _molecule_pairs(self, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q0, q1 = int(self.pair_offsets[position]), int(self.pair_offsets[position + 1])
        src = self.pair_source[q0:q1]
        tgt = self.pair_target[q0:q1]
        bucket = self.pair_bucket[q0:q1]
        return src, tgt, bucket

    def fit_statistics(self, fit_positions: np.ndarray) -> tuple[tuple, tuple]:
        a_sum = np.zeros(OFFDIAG_DIM)
        a_sq = np.zeros(OFFDIAG_DIM)
        n_a = 0
        d_sum = np.zeros(PAIR_DIM)
        d_sq = np.zeros(PAIR_DIM)
        n_d = 0
        for position in fit_positions:
            src, tgt, _ = self._molecule_pairs(int(position))
            if src.size == 0:
                continue
            left = self.u[src]
            right = self.u[tgt]
            a = _offdiag_outer(left, right)
            d = left * right
            a_sum += a.sum(axis=0)
            a_sq += (a * a).sum(axis=0)
            n_a += a.shape[0]
            d_sum += d.sum(axis=0)
            d_sq += (d * d).sum(axis=0)
            n_d += d.shape[0]
        a_mean = a_sum / max(n_a, 1)
        a_var = np.maximum(a_sq / max(n_a, 1) - a_mean * a_mean, 0.0)
        a_scale = np.sqrt(a_var + 1e-8)
        a_scale[a_scale < 1e-8] = 1.0
        d_mean = d_sum / max(n_d, 1)
        d_var = np.maximum(d_sq / max(n_d, 1) - d_mean * d_mean, 0.0)
        d_scale = np.sqrt(d_var + 1e-8)
        d_scale[d_scale < 1e-8] = 1.0
        return (a_mean, a_scale), (d_mean, d_scale)

    def build(self, fit_positions: np.ndarray) -> dict[str, np.ndarray]:
        a_stat, d_stat = self.fit_statistics(fit_positions)
        n = len(self.subset_index)
        endpoint = np.zeros((n, WITNESS_GRAPH_DIM), dtype=np.float64)
        diagonal = np.zeros((n, DIAG_GRAPH_DIM), dtype=np.float64)
        for position in range(n):
            src, tgt, bucket = self._molecule_pairs(position)
            if src.size == 0:
                continue
            e, d = _witness_for_single_molecule(self.u, src, tgt, bucket, self.projection, a_stat, d_stat)
            endpoint[position] = e
            diagonal[position] = d
        return {
            "endpoint_graph": endpoint.astype(np.float32),
            "diagonal_graph": diagonal.astype(np.float32),
            "a_mean": a_stat[0].astype(np.float32),
            "a_scale": a_stat[1].astype(np.float32),
            "d_mean": d_stat[0].astype(np.float32),
            "d_scale": d_stat[1].astype(np.float32),
        }


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

class ResidualHead(nn.Module):
    """yhat = yhat_0 + MLP(x).  Hidden width controls the parameter budget."""

    def __init__(self, in_dim: int, hidden: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(int(depth) - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_scale", torch.ones(in_dim))

    def set_standardizer(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.in_mean = torch.tensor(mean, dtype=torch.float32)
        self.in_scale = torch.tensor(scale, dtype=torch.float32)

    def forward(self, x: torch.Tensor, yhat_0: torch.Tensor, use_witness: bool = True) -> torch.Tensor:
        value = x
        if not use_witness and x.shape[1] > R_DIM:
            value = torch.cat([x[:, :R_DIM], torch.zeros_like(x[:, R_DIM:])], dim=1)
        value = (value - self.in_mean) / self.in_scale
        return yhat_0 + self.net(value).squeeze(-1)


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _make_head(seed: int, mode: str) -> ResidualHead:
    torch.manual_seed(int(seed))
    if mode == "ronly":
        return ResidualHead(R_DIM, RONLY_HIDDEN, RONLY_DEPTH)
    if mode in ("diag", "endpoint"):
        return ResidualHead(R_DIM + DIAG_GRAPH_DIM, WITNESS_HIDDEN, WITNESS_DEPTH)
    raise ValueError(mode)


def _standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def _train_adapter(
    model: nn.Module,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    yhat_fit: torch.Tensor,
    x_sel: torch.Tensor,
    y_sel: torch.Tensor,
    yhat_sel: torch.Tensor,
    *,
    epochs: int = ADAPTER_EPOCHS,
    lr: float = ADAPTER_LR,
    patience: int = ADAPTER_PATIENCE,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_state = copy.deepcopy(model.state_dict())
    best_selection = float("inf")
    bad = 0
    grad_alive = False
    epochs_run = 0
    for epoch in range(int(epochs)):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        prediction = model(x_fit, yhat_fit)
        loss = (prediction - y_fit).abs().mean()
        loss.backward()
        if not grad_alive:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grad_alive = bool(grads) and any(float(g.abs().sum()) > 0 for g in grads)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            selection_mae = float((model(x_sel, yhat_sel) - y_sel).abs().mean())
        if selection_mae < best_selection - 1e-9:
            best_selection = selection_mae
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    model.load_state_dict(best_state)
    return {"best_selection_mae": best_selection, "grad_alive": bool(grad_alive), "epochs": epochs_run}


def _run_stage_for_fold(
    fold: int,
    backbone_seed: int,
    adapter_seed: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    export = load_export(_export_path(fold, backbone_seed))
    projection = _random_projection()
    fold_manifest = manifest["folds"][f"fold{fold}"]
    fit_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_fit"]], dtype=np.int64)
    sel_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_selection"]], dtype=np.int64)
    eva_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_evaluation"]], dtype=np.int64)

    witness = FoldWitness(export, projection)
    lookup = {int(v): i for i, v in enumerate(witness.subset_index)}
    fit_pos = np.asarray([lookup[int(v)] for v in fit_ids], dtype=np.int64)
    sel_pos = np.asarray([lookup[int(v)] for v in sel_ids], dtype=np.int64)
    eva_pos = np.asarray([lookup[int(v)] for v in eva_ids], dtype=np.int64)
    graphs = witness.build(fit_pos)

    def tensorize(positions):
        R = torch.tensor(witness.R[positions], dtype=torch.float32)
        y = torch.tensor(witness.y[positions], dtype=torch.float32)
        yhat = torch.tensor(witness.yhat_0[positions], dtype=torch.float32)
        ep = torch.tensor(graphs["endpoint_graph"][positions], dtype=torch.float32)
        dg = torch.tensor(graphs["diagonal_graph"][positions], dtype=torch.float32)
        return R, y, yhat, ep, dg

    R_fit, y_fit, yhat_fit, ep_fit, dg_fit = tensorize(fit_pos)
    R_sel, y_sel, yhat_sel, ep_sel, dg_sel = tensorize(sel_pos)
    R_eva, y_eva, yhat_eva, ep_eva, dg_eva = tensorize(eva_pos)

    r_mean, r_scale = _standardizer(witness.R[fit_pos])
    ep_mean, ep_scale = _standardizer(graphs["endpoint_graph"][fit_pos])
    dg_mean, dg_scale = _standardizer(graphs["diagonal_graph"][fit_pos])

    x_ronly_fit = R_fit
    x_ep_fit = torch.cat([R_fit, ep_fit], dim=1)
    x_dg_fit = torch.cat([R_fit, dg_fit], dim=1)
    x_ronly_sel, x_ep_sel, x_dg_sel = R_sel, torch.cat([R_sel, ep_sel], dim=1), torch.cat([R_sel, dg_sel], dim=1)
    x_ronly_eva, x_ep_eva, x_dg_eva = R_eva, torch.cat([R_eva, ep_eva], dim=1), torch.cat([R_eva, dg_eva], dim=1)

    def make_and_train(x_fit, x_sel, mode, sel_mean_scale):
        model = _make_head(adapter_seed, mode)
        model.set_standardizer(*sel_mean_scale)
        info = _train_adapter(model, x_fit, y_fit, yhat_fit, x_sel, y_sel, yhat_sel)
        return model, info

    ronly_mean_scale = (r_mean, r_scale)
    ep_mean_scale = (
        np.concatenate([r_mean, ep_mean]),
        np.concatenate([r_scale, ep_scale]),
    )
    dg_mean_scale = (
        np.concatenate([r_mean, dg_mean]),
        np.concatenate([r_scale, dg_scale]),
    )

    b1, b1_info = make_and_train(x_ronly_fit, x_ronly_sel, "ronly", ronly_mean_scale)
    b2, b2_info = make_and_train(x_dg_fit, x_dg_sel, "diag", dg_mean_scale)
    e_model, e_info = make_and_train(x_ep_fit, x_ep_sel, "endpoint", ep_mean_scale)

    with torch.no_grad():
        p_b1 = b1(x_ronly_eva, yhat_eva).numpy()
        p_b2 = b2(x_dg_eva, yhat_eva).numpy()
        p_e = e_model(x_ep_eva, yhat_eva).numpy()
        p_e_no = e_model(x_ep_eva, yhat_eva, use_witness=False).numpy()
    y = witness.y[eva_pos]
    yhat0 = witness.yhat_0[eva_pos]
    mae_b0 = float(np.abs(y - yhat0).mean())
    mae_b1 = float(np.abs(y - p_b1).mean())
    mae_b2 = float(np.abs(y - p_b2).mean())
    mae_e = float(np.abs(y - p_e).mean())

    rarity = pd.read_csv(RARITY_CSV).set_index("subset_index")
    rare_ratio = rarity.loc[eva_ids, "rare_le5_ratio"].to_numpy(dtype=np.float64)
    bulk_mask = rare_ratio < float(fold_manifest["rare_le5_threshold"])
    if bulk_mask.sum() >= 10:
        bulk_b0 = float(np.abs(y - yhat0)[bulk_mask].mean())
        bulk_b1 = float(np.abs(y - p_b1)[bulk_mask].mean())
        bulk_b2 = float(np.abs(y - p_b2)[bulk_mask].mean())
        bulk_e = float(np.abs(y - p_e)[bulk_mask].mean())
        bulk_delta_e = bulk_e - bulk_b0
        bulk_delta_b1 = bulk_b1 - bulk_b0
        bulk_delta_b2 = bulk_b2 - bulk_b0
    else:
        bulk_delta_e = bulk_delta_b1 = bulk_delta_b2 = float("nan")

    return {
        "fold": fold,
        "backbone_seed": backbone_seed,
        "adapter_seed": adapter_seed,
        "mae_b0": mae_b0,
        "mae_b1": mae_b1,
        "mae_b2": mae_b2,
        "mae_e": mae_e,
        "delta_R": mae_b1 - mae_e,
        "delta_diag": mae_b2 - mae_e,
        "delta0_E": mae_b0 - mae_e,
        "bulk_degradation_e": bulk_delta_e,
        "bulk_degradation_b1": bulk_delta_b1,
        "bulk_degradation_b2": bulk_delta_b2,
        "n_bulk_eval": int(bulk_mask.sum()),
        "ronly_params": _count_parameters(b1),
        "diag_params": _count_parameters(b2),
        "endpoint_params": _count_parameters(e_model),
        "ronly_selection_mae": b1_info["best_selection_mae"],
        "diag_selection_mae": b2_info["best_selection_mae"],
        "endpoint_selection_mae": e_info["best_selection_mae"],
        "endpoint_grad_alive": e_info["grad_alive"],
        "endpoint_sensitivity_max_abs_diff": float(np.abs(p_e - p_e_no).max()),
        "endpoint_prediction_std": float(p_e.std()),
        "n_eval": int(len(eva_ids)),
        "_molecules": {
            "subset_index": eva_ids,
            "fold": np.full(len(eva_ids), fold, dtype=np.int64),
            "err_b0": np.abs(y - yhat0),
            "err_b1": np.abs(y - p_b1),
            "err_b2": np.abs(y - p_b2),
            "err_e": np.abs(y - p_e),
        },
    }


def _aggregate_stage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    delta_r = frame["delta_R"].to_numpy(dtype=np.float64)
    delta_d = frame["delta_diag"].to_numpy(dtype=np.float64)
    bulk = frame["bulk_degradation_e"].to_numpy(dtype=np.float64)
    bulk = bulk[np.isfinite(bulk)]
    return {
        "mean_delta_R": float(delta_r.mean()),
        "median_delta_R": float(np.median(delta_r)),
        "mean_delta_diag": float(delta_d.mean()),
        "median_delta_diag": float(np.median(delta_d)),
        "positive_folds_delta_R": int(np.sum(delta_r > 0)),
        "nonneg_folds_delta_diag": int(np.sum(delta_d >= 0)),
        "n_folds": int(len(frame)),
        "per_fold_delta_R": delta_r.tolist(),
        "per_fold_delta_diag": delta_d.tolist(),
        "mean_b0": float(frame["mae_b0"].mean()),
        "mean_b1": float(frame["mae_b1"].mean()),
        "mean_b2": float(frame["mae_b2"].mean()),
        "mean_e": float(frame["mae_e"].mean()),
        "mean_bulk_degradation_e": float(bulk.mean()) if bulk.size else None,
        "max_bulk_degradation_e": float(bulk.max()) if bulk.size else None,
        "ronly_params": int(frame["ronly_params"].iloc[0]),
        "diag_params": int(frame["diag_params"].iloc[0]),
        "endpoint_params": int(frame["endpoint_params"].iloc[0]),
        "endpoint_grad_alive_all": bool(frame["endpoint_grad_alive"].all()),
        "endpoint_sensitivity_min": float(frame["endpoint_sensitivity_max_abs_diff"].min()),
    }


def run_stage(
    backbone_seed: int,
    adapter_seed: int,
    tag: str,
    manifest: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_csv = RESULTS_DIR / f"{tag}_fold_results.csv"
    out_boot = RESULTS_DIR / f"{tag}_bootstrap.json"
    if out_csv.exists() and out_boot.exists() and not force:
        cached = json.loads(out_boot.read_text(encoding="utf-8"))
        return {
            "csv": str(out_csv),
            "bootstrap": str(out_boot),
            "aggregate": cached["aggregate"],
            "bootstrap_detail": {"delta_R": cached["delta_R"], "delta_diag": cached["delta_diag"]},
            "cached": True,
        }
    rows = [_run_stage_for_fold(fold, backbone_seed, adapter_seed, manifest) for fold in range(K_FOLDS)]
    pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]).to_csv(out_csv, index=False)
    molecules = [row["_molecules"] for row in rows]
    fold_ids = np.concatenate([m["fold"] for m in molecules])
    delta_r = np.concatenate([m["err_b1"] - m["err_e"] for m in molecules])
    delta_d = np.concatenate([m["err_b2"] - m["err_e"] for m in molecules])
    bootstrap = {
        "backbone_seed": int(backbone_seed),
        "adapter_seed": int(adapter_seed),
        "delta_R": _stratified_bootstrap(delta_r, fold_ids),
        "delta_diag": _stratified_bootstrap(delta_d, fold_ids),
        "aggregate": _aggregate_stage(rows),
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_boot, bootstrap)
    print(
        f"[{tag}] seed={backbone_seed} init={adapter_seed} "
        f"mean dR={bootstrap['aggregate']['mean_delta_R']:+.4f} "
        f"mean ddiag={bootstrap['aggregate']['mean_delta_diag']:+.4f} "
        f"({time.perf_counter() - started:.0f}s)",
        flush=True,
    )
    return {
        "csv": str(out_csv),
        "bootstrap": str(out_boot),
        "aggregate": bootstrap["aggregate"],
        "bootstrap_detail": {"delta_R": bootstrap["delta_R"], "delta_diag": bootstrap["delta_diag"]},
        "cached": False,
    }


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def _stratified_bootstrap(delta: np.ndarray, fold_ids: np.ndarray, n_boot: int = 10000, seed: int = 20260911) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    folds = np.unique(fold_ids)
    index_by_fold = [np.flatnonzero(fold_ids == f) for f in folds]
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        chunks = [delta[idx[rng.integers(0, len(idx), len(idx))]] for idx in index_by_fold]
        means[i] = np.concatenate(chunks).mean()
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_boot": int(n_boot),
        "prob_positive": float((means > 0).mean()),
        "n_molecules": int(len(delta)),
    }


# ---------------------------------------------------------------------------
# representation collision / ambiguity audit
# ---------------------------------------------------------------------------

def _sample_pairs(export: Mapping[str, Any], per_fold: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    u = np.asarray(export["projected_patch_state"], dtype=np.float64)
    src = np.asarray(export["pair_source"], dtype=np.int64)
    tgt = np.asarray(export["pair_target"], dtype=np.int64)
    bucket = np.asarray(export["pair_bucket"], dtype=np.int64)
    rel = np.asarray(export["pair_relation"], dtype=np.float64)
    total = src.shape[0]
    take = min(per_fold, total)
    idx = rng.choice(total, size=take, replace=False)
    left = u[src[idx]]
    right = u[tgt[idx]]
    c = np.concatenate([left + right, np.abs(left - right), left * right], axis=1)
    a = _offdiag_outer(left, right)
    return {
        "c": c,
        "a": a,
        "relation": rel[idx],
        "bucket": bucket[idx],
    }


def _ambiguity_for_pool(pool: Mapping[str, np.ndarray], rng: np.random.Generator) -> dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    c = pool["c"]
    a = pool["a"]
    rel = pool["relation"]
    bucket = pool["bucket"]
    c_mean, c_scale = c.mean(0), c.std(0)
    a_mean, a_scale = a.mean(0), a.std(0)
    r_mean, r_scale = rel.mean(0), rel.std(0)
    c_scale[c_scale < 1e-8] = 1.0
    a_scale[a_scale < 1e-8] = 1.0
    r_scale[r_scale < 1e-8] = 1.0
    c_std = (c - c_mean) / c_scale
    a_std = (a - a_mean) / a_scale
    r_std = (rel - r_mean) / r_scale

    n = c.shape[0]
    n_query = min(AUDIT_QUERY_PER_FOLD, n // 2)
    query_idx = rng.choice(n, size=n_query, replace=False)

    out: dict[str, Any] = {}
    for label in ("same_bucket", "relation_restricted"):
        d_c_nn, d_a_nn, d_a_rand = [], [], []
        for b in range(N_BUCKETS):
            pool_idx = np.flatnonzero(bucket == b)
            if pool_idx.size < KNN_K + 5:
                continue
            q = query_idx[np.isin(query_idx, pool_idx)]
            if q.size == 0:
                continue
            # per-query random a-distance reference (same bucket)
            for qi in q:
                sample = pool_idx
                if sample.size > AUDIT_RANDOM_SAMPLES:
                    sample = rng.choice(sample, size=AUDIT_RANDOM_SAMPLES, replace=False)
                d_a_rand.append(float(np.linalg.norm(a_std[sample] - a_std[qi], axis=1).mean()))

            if label == "same_bucket":
                nn = NearestNeighbors(n_neighbors=min(KNN_K + 1, pool_idx.size)).fit(c_std[pool_idx])
                dist, idx = nn.kneighbors(c_std[q])
                nb = pool_idx[idx[:, 1]]
                d_c_nn.extend(dist[:, 1].tolist())
                d_a_nn.extend(np.linalg.norm(a_std[nb] - a_std[q], axis=1).tolist())
            else:
                k_rel = min(AUDIT_REL_CAND, pool_idx.size)
                nn_r = NearestNeighbors(n_neighbors=k_rel).fit(r_std[pool_idx])
                _, idx_r = nn_r.kneighbors(r_std[q])
                cand_rel = pool_idx[idx_r]
                for i, qi in enumerate(q):
                    cand = cand_rel[i][cand_rel[i] != qi]
                    if cand.size == 0:
                        continue
                    dc = np.linalg.norm(c_std[cand] - c_std[qi], axis=1)
                    j = int(np.argmin(dc))
                    nb = cand[j]
                    d_c_nn.append(float(dc[j]))
                    d_a_nn.append(float(np.linalg.norm(a_std[nb] - a_std[qi])))
        d_c_nn = np.asarray(d_c_nn)
        d_a_nn = np.asarray(d_a_nn)
        d_a_rand = np.asarray(d_a_rand)
        if d_c_nn.size == 0:
            out[label] = {"n": 0}
            continue
        ratio = float(np.median(d_a_nn) / max(np.median(d_a_rand), 1e-12))
        corr = float(np.corrcoef(d_c_nn, d_a_nn)[0, 1]) if d_c_nn.std() > 0 else float("nan")
        close_c = d_c_nn <= np.median(d_c_nn)
        far_a = d_a_nn >= np.median(d_a_rand) if d_a_rand.size else np.zeros_like(close_c)
        sample = rng.choice(d_c_nn.size, size=min(800, d_c_nn.size), replace=False)
        sample = np.sort(sample)
        out[label] = {
            "n": int(d_c_nn.size),
            "median_d_c_nn": float(np.median(d_c_nn)),
            "median_d_a_nn": float(np.median(d_a_nn)),
            "median_d_a_rand": float(np.median(d_a_rand)),
            "median_a_nn_over_rand": ratio,
            "pearson_d_c_nn_vs_d_a_nn": corr,
            "frac_close_c_far_a": float(np.mean(close_c & far_a)),
            "d_c_nn_sample": d_c_nn[sample].tolist(),
            "d_a_nn_sample": d_a_nn[sample].tolist(),
        }

    # Audit B: local conditional variance of a given c-neighbourhood (k = KNN_K)
    cond_vars = []
    total_var_a = float(a_std.var(axis=0).mean())
    for b in range(N_BUCKETS):
        pool_idx = np.flatnonzero(bucket == b)
        if pool_idx.size < KNN_K + 2:
            continue
        q = query_idx[np.isin(query_idx, pool_idx)]
        if q.size == 0:
            continue
        nn = NearestNeighbors(n_neighbors=min(KNN_K + 1, pool_idx.size)).fit(c_std[pool_idx])
        _, idx = nn.kneighbors(c_std[q])
        for row in idx:
            neigh = pool_idx[row[1:]]
            if neigh.size < 2:
                continue
            cond_vars.append(float(a_std[neigh].var(axis=0).mean()))
    cond_vars = np.asarray(cond_vars)
    out["conditional_variance"] = {
        "k": KNN_K,
        "n": int(cond_vars.size),
        "mean_conditional_var_a": float(cond_vars.mean()) if cond_vars.size else float("nan"),
        "total_var_a": total_var_a,
        "normalized_conditional_var": float(cond_vars.mean() / max(total_var_a, 1e-12)) if cond_vars.size else float("nan"),
        "median_conditional_var_a": float(np.median(cond_vars)) if cond_vars.size else float("nan"),
    }
    return out


def run_ambiguity(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = RESULTS_DIR / "representation_ambiguity.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(AUDIT_SEED)
    per_fold: dict[str, Any] = {}
    for fold in range(K_FOLDS):
        export = load_export(_export_path(fold, PRIMARY_BACKBONE_SEED))
        pool = _sample_pairs(export, AUDIT_POOL_PER_FOLD, rng)
        per_fold[f"fold{fold}"] = _ambiguity_for_pool(pool, rng)
        sb = per_fold[f"fold{fold}"]["same_bucket"]
        rr = per_fold[f"fold{fold}"]["relation_restricted"]
        print(
            f"[ambiguity] fold={fold} same_bucket a_nn/rand={sb['median_a_nn_over_rand']:.3f} "
            f"corr={sb['pearson_d_c_nn_vs_d_a_nn']:.3f} | "
            f"relation a_nn/rand={rr['median_a_nn_over_rand']:.3f} "
            f"condvar={per_fold[f'fold{fold}']['conditional_variance']['normalized_conditional_var']:.3f}",
            flush=True,
        )
    # aggregate scalar summaries
    keys = ("median_a_nn_over_rand", "pearson_d_c_nn_vs_d_a_nn", "frac_close_c_far_a")
    aggregate = {}
    for label in ("same_bucket", "relation_restricted"):
        for key in keys:
            values = [per_fold[f"fold{f}"].get(label, {}).get(key) for f in range(K_FOLDS)]
            values = [v for v in values if v is not None and np.isfinite(v)]
            aggregate[f"{label}_{key}_mean"] = float(np.mean(values)) if values else None
    cond = [per_fold[f"fold{f}"]["conditional_variance"]["normalized_conditional_var"] for f in range(K_FOLDS)]
    cond = [v for v in cond if np.isfinite(v)]
    aggregate["conditional_var_normalized_mean"] = float(np.mean(cond)) if cond else None
    report = {
        "definition": {
            "c_ij": "[u_i+u_j, |u_i-u_j|, u_i*u_j] (48D), relation excluded from the collision key",
            "a_ij": "strictly upper-triangular part of A_ij = u_i u_j^T + u_j u_i^T (120D)",
            "knn_k": KNN_K,
            "pool_per_fold": AUDIT_POOL_PER_FOLD,
            "query_per_fold": AUDIT_QUERY_PER_FOLD,
            "audit_seed": AUDIT_SEED,
            "target_used": False,
        },
        "per_fold": per_fold,
        "aggregate": aggregate,
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_path, report)
    return report


# ---------------------------------------------------------------------------
# decision logic
# ---------------------------------------------------------------------------

def _decide_stage1(stage1: Mapping[str, Any]) -> str:
    agg = stage1["aggregate"]
    dr = agg["mean_delta_R"]
    dd = agg["mean_delta_diag"]
    pos_dr = agg["positive_folds_delta_R"]  # folds where E beats R-only (Delta_R > 0)
    # CLEAR NO-GO
    if dr <= S1_NGO_DR or dd <= S1_NGO_DDIAG or pos_dr <= S1_NGO_WORSE_FOLDS:
        return "STAGE1_CLEAR_NO_GO"
    # ADVANCE
    if dr >= S1_ADV_DR and dd >= S1_ADV_DDIAG and pos_dr >= S1_ADV_FOLDS:
        return "STAGE1_ADVANCE"
    return "STAGE1_BORDERLINE"


def _final_decision(
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any] | None,
    stage1b: Mapping[str, Any] | None,
) -> dict[str, Any]:
    s1 = stage1["aggregate"]
    stages = [s1] + ([stage2["aggregate"]] if stage2 is not None else [])
    pooled_dr = float(np.mean([s["mean_delta_R"] for s in stages]))
    pooled_dd = float(np.mean([s["mean_delta_diag"] for s in stages]))
    per_fold_dr = np.mean([s["per_fold_delta_R"] for s in stages], axis=0)
    folds_positive = int(np.sum(per_fold_dr > 0))

    def _delta_r_boot(stage):
        return stage.get("bootstrap_detail", {}).get("delta_R", stage.get("delta_R"))

    if stage2 is not None:
        boot_r = [_delta_r_boot(stage1), _delta_r_boot(stage2)]
        ci_low_positive = all(b["ci95_low"] > 0 for b in boot_r)
    else:
        boot_r = [_delta_r_boot(stage1)]
        ci_low_positive = boot_r[0]["ci95_low"] > 0
    backbone_consistent = all(s["mean_delta_R"] > 0 for s in stages)

    sensitivity_ok = all(s["endpoint_sensitivity_min"] > 0 for s in stages)
    bulk_values = [s.get("max_bulk_degradation_e") for s in stages]
    bulk_values = [float(v) for v in bulk_values if v is not None and np.isfinite(v)]
    bulk_safe = bool(bulk_values) and max(bulk_values) <= BULK_MAX_DEGRADATION

    criteria = {
        "pooled_delta_R_ge_0.002": pooled_dr >= FINAL_GO_DR,
        "pooled_delta_diag_ge_0.0015": pooled_dd >= FINAL_GO_DDIAG,
        "backbone_consistent_positive": backbone_consistent,
        "folds_at_least_4_of_5_positive": folds_positive >= FINAL_GO_FOLDS,
        "paired_ci_lower_positive": ci_low_positive,
        "endpoint_adapter_noncollapsed": bool(sensitivity_ok),
        "bulk_safe_le_0.002": bulk_safe,
    }

    cases: dict[str, Any] = {
        "E_gt_B1": bool(pooled_dr > 0),
        "E_gt_B2": bool(pooled_dd > 0),
    }
    if all(criteria.values()):
        verdict = "GO"
        claim = (
            "Given the frozen compact-v4 representation, symmetric off-diagonal "
            "endpoint association features provide reproducible incremental predictive "
            "value beyond the existing graph vector R and beyond the diagonal/current-"
            "pair-information control."
        )
    elif pooled_dr <= 0:
        verdict = "NO_GO"
        claim = (
            "No stable signed predictive value of the off-diagonal endpoint association "
            "was found beyond R (endpoint witness not better than the R-only control)."
        )
    elif pooled_dd <= 0:
        verdict = "ENDPOINT_ASSOCIATION_NOT_SUPPORTED"
        claim = (
            "The endpoint witness beats R-only but not the matched diagonal/current-pair "
            "information control; the gain is generic extra pair summary, not evidence "
            "for off-diagonal endpoint association."
        )
    else:
        verdict = "INCONCLUSIVE"
        claim = (
            "The endpoint witness shows a positive but sub-threshold / unresolved "
            "incremental signal; the budget was not expanded."
        )

    return {
        "pooled_delta_R": pooled_dr,
        "pooled_delta_diag": pooled_dd,
        "per_fold_delta_R": per_fold_dr.tolist(),
        "positive_folds_delta_R": folds_positive,
        "delta_R_ci95": [boot_r[0]["ci95_low"], boot_r[0]["ci95_high"]],
        "bulk_degradation_e_max": float(max(bulk_values)) if bulk_values else None,
        "criteria": criteria,
        "cases": cases,
        "verdict": verdict,
        "claim": claim,
        "stage1b": stage1b,
        "scope_caveat": (
            "Screening diagnostic on frozen OOF states of the existing backbone. "
            "A NO-GO closes off-diagonal endpoint cross-coordinate association as a "
            "bottleneck for this representation; it does NOT prove the pair encoder is "
            "globally wrong nor that a full outer product is optimal."
        ),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _make_figures(stage1: Mapping[str, Any], stage2: Mapping[str, Any] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RESULTS_DIR / "stage1_fold_results.csv")

    # Figure 1: representation ambiguity — c-neighbour distance vs a-distance
    amb_path = RESULTS_DIR / "representation_ambiguity.json"
    if amb_path.exists():
        amb = json.loads(amb_path.read_text(encoding="utf-8"))
        folds = [f"fold{f}" for f in range(K_FOLDS)]
        f0 = amb["per_fold"]["fold0"]["same_bucket"]
        dc = np.asarray(f0["d_c_nn_sample"])
        da = np.asarray(f0["d_a_nn_sample"])
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        ax = axes[0]
        ax.scatter(dc, da, s=6, alpha=0.35, label="c-nearest pairs (fold 0)")
        ax.axhline(f0["median_d_a_rand"], color="red", linestyle="--", linewidth=1.0,
                   label="random same-bucket a-distance")
        ax.set_xlabel(r"$\|c_i-c_{nn}\|$ (current features)")
        ax.set_ylabel(r"$\|a_i-a_{nn}\|$ (off-diagonal association)")
        ax.set_title("Figure 1a: ambiguity scatter")
        ax.legend(fontsize=8)
        ax = axes[1]
        x = np.arange(K_FOLDS)
        ratio_sb = [amb["per_fold"][f]["same_bucket"]["median_a_nn_over_rand"] for f in folds]
        ratio_rr = [amb["per_fold"][f]["relation_restricted"]["median_a_nn_over_rand"] for f in folds]
        ax.bar(x - 0.2, ratio_sb, width=0.4, label="same bucket")
        ax.bar(x + 0.2, ratio_rr, width=0.4, label="relation-restricted")
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xlabel("outer fold")
        ax.set_ylabel("a-dist(c-nearest) / a-dist(random)")
        ax.set_title("Figure 1b: how much c predicts a")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "figure1_representation_ambiguity.png", dpi=150)
        plt.close(fig)

    # Figure 2: per-fold Delta_R and Delta_diag
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(np.arange(K_FOLDS) - 0.2, frame["delta_R"], width=0.4, label=r"$\Delta_R$ (B1 - E)")
    ax.bar(np.arange(K_FOLDS) + 0.2, frame["delta_diag"], width=0.4, label=r"$\Delta_{diag}$ (B2 - E)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("MAE improvement")
    ax.set_title("Figure 2: per-fold endpoint witness gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_per_fold_delta.png", dpi=150)
    plt.close(fig)

    # Figure 3: only if Stage 2 ran
    if stage2 is not None and (RESULTS_DIR / "stage2_fold_results.csv").exists():
        frame2 = pd.read_csv(RESULTS_DIR / "stage2_fold_results.csv")
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(np.arange(K_FOLDS), frame["delta_R"], marker="o", label="backbone seed 0")
        ax.plot(np.arange(K_FOLDS), frame2["delta_R"], marker="s", label="backbone seed 1")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("outer fold")
        ax.set_ylabel(r"$\Delta_R$")
        ax.set_title("Figure 3: endpoint gain across frozen backbone seeds")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "figure3_stage2_seeds.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Run-all orchestration
# ---------------------------------------------------------------------------

def _run_all(force: bool = False) -> int:
    print("=== checkpoint inventory ===", flush=True)
    inventory = checkpoint_inventory()
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    if not inventory["entries"] or "complete_seeds" not in inventory or not inventory["complete_seeds"]:
        print("no complete frozen checkpoint set; STOP.", flush=True)
        return 2

    print("=== pair feature spec ===", flush=True)
    write_pair_feature_spec()
    print("=== export v3 + gate0 ===", flush=True)
    stage_export(seeds=(PRIMARY_BACKBONE_SEED,), force=force)
    report = run_gate0(force=force)
    if not report.get("all_passed"):
        print("GATE 0 FAILED - measurement invalid; no adapter trained.", flush=True)
        _write_json_any(RESULTS_DIR / "final_decision.json", {
            "verdict": "NO_GO", "reason": "gate0_failed", "failures": report.get("failures"),
        })
        return 3

    print("=== representation ambiguity audit ===", flush=True)
    run_ambiguity(force=force)

    manifest = stage_splits(force=force)
    print("=== stage1 (backbone seed 0, 1 init) ===", flush=True)
    stage1 = run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", manifest, force=force)
    verdict1 = _decide_stage1(stage1)

    stage1b = None
    stage2 = None
    if verdict1 == "STAGE1_CLEAR_NO_GO":
        print("Stage 1 CLEAR NO-GO; second backbone seed NOT spent.", flush=True)
    else:
        if verdict1 == "STAGE1_BORDERLINE":
            print("=== stage1b (second adapter init, same backbone) ===", flush=True)
            stage1b = run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b", manifest, force=force)
            agg_b = stage1b["aggregate"]
            collapses = agg_b["mean_delta_R"] <= 0.0 or np.sign(agg_b["mean_delta_R"]) != np.sign(stage1["aggregate"]["mean_delta_R"])
            if collapses:
                print("Stage 1b signal collapses/flips -> NO-GO, no second backbone.", flush=True)
                verdict1 = "STAGE1_CLEAR_NO_GO"
            else:
                verdict1 = "STAGE1_ADVANCE"
        if verdict1 == "STAGE1_ADVANCE":
            print("=== stage2 (second frozen backbone seed) ===", flush=True)
            stage_export(seeds=(BACKBONE_SEEDS[1],), force=force)
            stage2 = run_stage(BACKBONE_SEEDS[1], 0, "stage2", manifest, force=force)

    decision = _final_decision(stage1, stage2, stage1b)
    decision["stage1_verdict"] = verdict1
    _write_json_any(RESULTS_DIR / "final_decision.json", decision)
    if stage2 is not None:
        pooled = {
            "pooled_delta_R": decision["pooled_delta_R"],
            "pooled_delta_diag": decision["pooled_delta_diag"],
            "stage1": stage1["aggregate"],
            "stage2": stage2["aggregate"],
        }
        _write_json_any(RESULTS_DIR / "pooled_results.json", pooled)
        _write_json_any(RESULTS_DIR / "final_bootstrap.json", {
            "stage1_delta_R": stage1["bootstrap_detail"]["delta_R"],
            "stage2_delta_R": stage2["bootstrap_detail"]["delta_R"],
        })
    _make_figures(stage1, stage2)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["inventory", "spec", "export", "gate0", "splits", "ambiguity", "stage1", "stage1b", "stage2", "figures", "decision", "all"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        print(json.dumps(checkpoint_inventory(), indent=2, sort_keys=True))
        return 0
    if args.stage == "spec":
        write_pair_feature_spec()
        return 0
    if args.stage == "export":
        stage_export(seeds=(PRIMARY_BACKBONE_SEED,), force=args.force)
        return 0
    if args.stage == "gate0":
        report = run_gate0(force=args.force)
        return 0 if report.get("all_passed") else 1
    if args.stage == "splits":
        stage_splits(force=args.force)
        return 0
    if args.stage == "ambiguity":
        run_ambiguity(force=args.force)
        return 0
    if args.stage == "stage1":
        manifest = stage_splits()
        run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", manifest, force=args.force)
        return 0
    if args.stage == "stage1b":
        manifest = stage_splits()
        run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b", manifest, force=args.force)
        return 0
    if args.stage == "stage2":
        manifest = stage_splits()
        stage_export(seeds=(BACKBONE_SEEDS[1],), force=args.force)
        run_stage(BACKBONE_SEEDS[1], 0, "stage2", manifest, force=args.force)
        return 0
    if args.stage == "decision":
        stage1 = json.loads((RESULTS_DIR / "stage1_bootstrap.json").read_text())
        stage2 = json.loads((RESULTS_DIR / "stage2_bootstrap.json").read_text()) if (RESULTS_DIR / "stage2_bootstrap.json").exists() else None
        stage1b = json.loads((RESULTS_DIR / "stage1b_bootstrap.json").read_text()) if (RESULTS_DIR / "stage1b_bootstrap.json").exists() else None
        decision = _final_decision(stage1, stage2, stage1b)
        decision["stage1_verdict"] = _decide_stage1(stage1)
        _write_json_any(RESULTS_DIR / "final_decision.json", decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0
    if args.stage == "figures":
        stage1 = json.loads((RESULTS_DIR / "stage1_bootstrap.json").read_text())
        stage2 = json.loads((RESULTS_DIR / "stage2_bootstrap.json").read_text()) if (RESULTS_DIR / "stage2_bootstrap.json").exists() else None
        _make_figures(stage1, stage2)
        return 0
    if args.stage == "all":
        return _run_all(force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
