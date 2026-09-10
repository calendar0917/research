"""Frozen-set conditional readout sufficiency audit (compact-v4-hinge, ZINC).

This is a **representation / readout sufficiency diagnostic**, not an
architecture benchmark.  It does not modify the main model, does not train a
new backbone and does not replace pooling.  It answers exactly one question:

    Given the already-frozen compact-v4-hinge states and the existing 302D
    fixed graph vector R, is there extra *signed* predictive signal that R
    does not expose but a tiny permutation-invariant set readout can use?

Pipeline (see ``notes/frozen_conditional_readout_sufficiency_audit.md``):

    Stage 0  export repair + integrity gates (NO adapter training)
    Stage 1  1 frozen OOF backbone seed x 5 outer folds x 1 init
    Stage 2  second frozen OOF backbone seed (only if Stage 1 advances)
    B2       marginal-preserving joint-structure destruction (only if S > B1)

Two historical measurement bugs are repaired here and nowhere else:

1. ``zinc_oof_difficulty_audit._forward_capture_fold`` grouped pair rows by
   ``np.diff(batch.pair_index[0])``.  ``pair_index[0]`` is the *source patch
   index within a molecule*, so that grouped by source node, never by
   molecule.  The correct graph id is ``batch.batch[batch.pair_index[0]]``.
2. The old ``head_input`` hook captured ``model.head[0]``'s **output**
   (64D), not the pre-head representation.  The true pre-head graph vector
   is the *input* to ``model.head[0]``: ``R`` in R^302.

Everything is evaluated on official TRAIN molecules only.  The official
validation and test splits are never loaded by this module.

Run:  ``uv run python -m tracks.ksvd.experiments.luyin16.zinc_frozen_readout_sufficiency <stage>``
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
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    TYPED_TOKENIZER_V1_HISTORICAL,
    typed_tokenizer_fingerprint,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _load_zinc,
    global_feature_views,
)
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
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/frozen_readout_sufficiency"
EXPORT_DIR = RESULTS_DIR / "state_exports"
FIG_DIR = RESULTS_DIR / "figures"

EXPORT_VERSION = "frozen_state_export_v2_corrected"
LEGACY_EXPORT_VERSIONS = {"head_input_v1", "frozen_state_export_v1"}
TOKENIZER_VERSION = TYPED_TOKENIZER_V1_HISTORICAL
CONFIG_PATH = (
    REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
)
ZINC_ROOT = REPO_ROOT / "data/ZINC"
RARITY_CSV = OOF_CACHE_DIR / "rarity_rows.csv"

# --- frozen v4-hinge shapes (verified from the real model at runtime) ------
R_DIM = 302
PATCH_DIM = 48
PAIR_DIM = 16
N_BUCKETS = zpp.DISTANCE_BUCKETS  # 5

# --- split / budget --------------------------------------------------------
SPLIT_SEED = "frozen-readout-sufficiency-v2-20260910"
SPLIT_SIZES = (1200, 400, 400)  # adapter-fit / adapter-selection / adapter-evaluation
BACKBONE_SEEDS = (0, 1)
PRIMARY_BACKBONE_SEED = 0

# --- adapter protocol (pre-registered, no sweeps) --------------------------
ADAPTER_SEEDS = (0, 1)  # seed 1 only used when Stage 1 is BORDERLINE
RONLY_HIDDEN = 13
SET_PHI_H_HIDDEN = 16
SET_PHI_H_OUT = 8
SET_PHI_Q_HIDDEN = 16
SET_PHI_Q_OUT = 8
SET_RHO_HIDDEN = 8
ADAPTER_LR = 1.0e-3
ADAPTER_EPOCHS = 400
ADAPTER_PATIENCE = 50
ADAPTER_BATCH = "full"

# --- decision thresholds (pre-registered) ----------------------------------
S1_NGO_D0 = 0.001
S1_NGO_DR = 0.001
S1_ADV_D0 = 0.0025
S1_ADV_DR = 0.0015
FINAL_GO_D0 = 0.003
FINAL_GO_DR = 0.002
BULK_MAX_DEGRADATION = 0.002
BULK_RARE_PERCENTILE = 80.0

# --- gate 0 tolerances -----------------------------------------------------
R_RECON_ATOL = 1.0e-4
PRED_RECON_ATOL = 1.0e-5
BATCH_INV_ATOL = 1.0e-4
PERM_INV_ATOL = 1.0e-4
FORWARD_GATE_ATOL = 1.0e-9


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


def _vocabulary_fingerprint(inner_train_records: Sequence[Any], config: Mapping[str, Any]) -> dict[str, Any]:
    representation = config["representation"]
    typed = zpp._fit_vocabulary(
        inner_train_records,
        "typed_certificate",
        int(representation.get("max_typed_tokens", 8192)),
        int(representation.get("minimum_typed_frequency", 1)),
    )
    parent = zpp._fit_vocabulary(
        inner_train_records,
        "parent_certificate",
        int(representation.get("max_parent_tokens", 2048)),
        int(representation.get("minimum_parent_frequency", 1)),
    )
    typed_blob = b"\n".join(k for k, _ in sorted(typed.items(), key=lambda kv: kv[1]))
    parent_blob = b"\n".join(k for k, _ in sorted(parent.items(), key=lambda kv: kv[1]))
    return {
        "typed_vocabulary_size": int(len(typed)),
        "parent_vocabulary_size": int(len(parent)),
        "typed_sha256": hashlib.sha256(typed_blob).hexdigest(),
        "parent_sha256": hashlib.sha256(parent_blob).hexdigest(),
    }


def _fold_fingerprint_inputs(fold: int, seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    records = _load_train_records()
    slices = _fold_slices()
    inner_train_idx, inner_valid_idx, holdout_idx = slices[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    fold_meta = _read_json(_fold_marker(fold, seed))
    tokenizer_fp = typed_tokenizer_fingerprint(TOKENIZER_VERSION, zpp.PATCH_RADIUS)
    payload = {
        "export_version": EXPORT_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_fingerprint": tokenizer_fp,
        "config_fingerprint": _hash_json(config),
        "vocabulary_fingerprint": _vocabulary_fingerprint(inner_train_records, config),
        "split_fingerprint": {
            "inner_train_sha256": _sha256_array(inner_train_idx),
            "inner_valid_sha256": _sha256_array(inner_valid_idx),
            "holdout_sha256": _sha256_array(holdout_idx),
        },
        "checkpoint_fingerprint": str(fold_meta["state_pt_sha256"]),
        "backbone_seed": int(seed),
        "fold": int(fold),
    }
    payload["fingerprint"] = _hash_json(payload)
    return payload


# ---------------------------------------------------------------------------
# Stage 0: corrected state export
# ---------------------------------------------------------------------------

def _capture_states(
    model: nn.Module,
    graphs: Sequence[Any],
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Eval-mode forward with the *corrected* per-molecule state capture.

    Pair rows are grouped by ``batch.batch[batch.pair_index[0]]`` (graph
    membership), never by the source patch index.  ``R`` is the true input to
    ``model.head[0]`` (the pre-head graph representation).
    """
    loader = DataLoader(list(graphs), batch_size=int(batch_size), shuffle=False)
    head_inputs: list[np.ndarray] = []
    pair_values: list[np.ndarray] = []
    patch_inputs: list[np.ndarray] = []
    patch_deltas: list[np.ndarray] = []
    global_hidden: list[np.ndarray] = []
    topology_hidden: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    def head_pre_hook(_module, args):
        head_inputs.append(args[0].detach().cpu().numpy())

    def pair_hook(_module, _inputs, output):
        pair_values.append(output.detach().cpu().numpy())

    def center_hook(_module, inputs, output):
        patch_inputs.append(inputs[0].detach().cpu().numpy())
        patch_deltas.append(output.detach().cpu().numpy())

    def global_hook(_module, _inputs, output):
        global_hidden.append(output.detach().cpu().numpy())

    def topology_hook(_module, _inputs, output):
        topology_hidden.append(output.detach().cpu().numpy())

    handles = [
        model.head[0].register_forward_pre_hook(head_pre_hook),
        model.pair_encoder.register_forward_hook(pair_hook),
        model.center_update.register_forward_hook(center_hook),
        model.global_encoder.register_forward_hook(global_hook),
        model.topology_encoder.register_forward_hook(topology_hook),
    ]

    per_graph: list[dict[str, np.ndarray]] = []
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
            h_final = patch_inputs[-1][:, : model.patch_hidden] + patch_deltas[-1]
            for graph_id in range(n_graphs):
                start, end = int(ptr[graph_id]), int(ptr[graph_id + 1])
                pair_mask = pair_graph == graph_id
                local_source = source[pair_mask] - start
                local_target = target[pair_mask] - start
                per_graph.append(
                    {
                        "n_patches": np.int64(end - start),
                        "patch_states": h_final[start:end].astype(np.float32),
                        "pair_states": pair_values[-1][pair_mask].astype(np.float32),
                        "pair_bucket": bucket[pair_mask].astype(np.int64),
                        "pair_source": source[pair_mask].astype(np.int64),
                        "pair_target": target[pair_mask].astype(np.int64),
                        "pair_source_local": local_source.astype(np.int64),
                        "pair_target_local": local_target.astype(np.int64),
                        "pair_source_graph": source_graph[pair_mask].astype(np.int64),
                        "pair_target_graph": target_graph[pair_mask].astype(np.int64),
                        "global_hidden": global_hidden[-1][graph_id].astype(np.float32),
                        "topology_hidden": topology_hidden[-1][graph_id].astype(np.float32),
                        "R": head_inputs[-1][graph_id].astype(np.float32),
                    }
                )
    for handle in handles:
        handle.remove()

    n_patches = np.asarray([g["n_patches"] for g in per_graph], dtype=np.int64)
    n_pairs = np.asarray([g["pair_states"].shape[0] for g in per_graph], dtype=np.int64)
    return {
        "R": np.stack([g["R"] for g in per_graph], axis=0),
        "yhat_0": np.concatenate(predictions).astype(np.float64),
        "n_patches": n_patches,
        "n_pairs": n_pairs,
        "patch_states": np.concatenate([g["patch_states"] for g in per_graph], axis=0),
        "pair_states": np.concatenate([g["pair_states"] for g in per_graph], axis=0),
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


def _build_export(fold: int, seed: int, batch_size: int = 128) -> dict[str, Any]:
    records = _load_train_records()
    labels = _load_train_labels()
    _verify_records_vs_labels(records, labels)
    config = _frozen_config()
    torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", 4)))
    inner_train_idx, inner_valid_idx, holdout_idx = _fold_slices()[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    holdout_records = [records[int(i)] for i in holdout_idx]
    _encoded_fit, encoded_holdout, audit = zpp._phase_data(
        inner_train_records, holdout_records, config=config
    )
    config_mod = dict(config)
    config_mod["model"] = _model_config_with_clamps(config, audit)
    model = _build_v4_model(config_mod, audit)
    state = torch.load(
        FOLD_DIR / f"fold{fold}_seed{seed}_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    if model.head[0].in_features != R_DIM:
        raise AssertionError(
            f"pre-head R width {model.head[0].in_features} != frozen R_DIM {R_DIM}"
        )
    if model.patch_hidden != PATCH_DIM or model.pair_hidden != PAIR_DIM:
        raise AssertionError("frozen patch/pair dims changed")
    capture = _capture_states(model, encoded_holdout, batch_size=batch_size)
    npz = _load_fold_npz(fold, seed)
    forward_gate = float(np.abs(capture["yhat_0"] - npz["oof_prediction"]).max())
    if forward_gate > FORWARD_GATE_ATOL:
        raise AssertionError(
            f"fold {fold} seed {seed}: corrected forward predictions deviate from "
            f"fold npz (max diff {forward_gate})"
        )
    target = np.asarray([float(record.y) for record in holdout_records], dtype=np.float64)
    fingerprint = _fold_fingerprint_inputs(fold, seed, config)
    fingerprint["forward_gate_max_diff"] = forward_gate
    export = {
        "subset_index": holdout_idx.astype(np.int64),
        "target": target,
        "fingerprint": fingerprint,
        **capture,
    }
    return export


def _save_export(path: Path, export: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, value in export.items():
        if key == "fingerprint":
            arrays["fingerprint_json"] = np.asarray(
                json.dumps(_jsonable(value), sort_keys=True)
            )
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_export(path: Path) -> dict[str, Any]:
    """Load a corrected state export, refusing any legacy/foreign cache."""
    with np.load(path, allow_pickle=False) as data:
        if "fingerprint_json" not in data.files:
            raise RuntimeError(f"cache {path} has no fingerprint; refusing to load")
        fingerprint = json.loads(str(data["fingerprint_json"]))
        if fingerprint.get("export_version") != EXPORT_VERSION:
            raise RuntimeError(
                f"cache {path} export_version={fingerprint.get('export_version')!r} "
                f"!= required {EXPORT_VERSION!r}; refusing to load"
            )
        export: dict[str, Any] = {"fingerprint": fingerprint}
        for key in data.files:
            if key == "fingerprint_json":
                continue
            export[key] = data[key]
    return export


def stage_export(seeds: Sequence[int] = (0,), force: bool = False) -> dict[str, Any]:
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
                "forward_gate_max_diff": float(
                    export["fingerprint"]["forward_gate_max_diff"]
                ),
            }
            print(
                f"[export] fold={fold} seed={seed} n={len(export['subset_index'])} "
                f"gate={export['fingerprint']['forward_gate_max_diff']:.2e} "
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
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(RESULTS_DIR / "stage_export.json", summary)
    return summary


# ---------------------------------------------------------------------------
# checkpoint inventory
# ---------------------------------------------------------------------------

def checkpoint_inventory() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for seed in BACKBONE_SEEDS:
        for fold in range(K_FOLDS):
            marker = _fold_marker(fold, seed)
            if not marker.exists():
                entries.append(
                    {"fold": fold, "seed": seed, "status": "missing"}
                )
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
                    "typed_vocabulary_size_with_oov": meta.get(
                        "typed_vocabulary_size_with_oov"
                    ),
                    "topology_input_width": meta.get("topology_input_width"),
                }
            )
    present = [e for e in entries if e["status"] == "present"]
    complete_seeds = sorted(
        {
            e["seed"]
            for e in present
            if sum(1 for x in present if x["seed"] == e["seed"]) == K_FOLDS
        }
    )
    return {
        "search_dir": str(FOLD_DIR),
        "n_entries": len(entries),
        "n_present": len(present),
        "complete_seeds": complete_seeds,
        "entries": entries,
        "frozen_config_path": str(CONFIG_PATH),
        "tokenizer_version": TOKENIZER_VERSION,
        "missing_checkpoint_policy": (
            "STOP and report a missing-checkpoint inventory; never silently "
            "convert the frozen diagnostic into fresh backbone training."
        ),
    }


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
            relation[position, base + PAIR_DIM: base + 2 * PAIR_DIM] = (
                current * current
            ).sum(axis=0)
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
    ok = bool(np.array_equal(expected, actual))
    return {
        "gate": "G0.1_pair_graph_grouping",
        "definition": "unordered all-pairs: n_g * (n_g - 1) / 2",
        "n_graphs": int(len(n_patches)),
        "n_mismatch": int(np.sum(expected != actual)),
        "max_abs_diff": int(np.abs(expected - actual).max()) if len(expected) else 0,
        "passed": ok,
    }


def _gate_pair_endpoint_identity(export: Mapping[str, Any]) -> dict[str, Any]:
    same = export["pair_source_graph"] == export["pair_target_graph"]
    return {
        "gate": "G0.2_pair_endpoint_graph_identity",
        "n_pairs": int(len(same)),
        "n_violations": int(np.sum(~same)),
        "passed": bool(np.all(same)),
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
        expected = size * (size - 1) // 2
        if q1 - q0 != expected:
            count_mismatch += 1
    passed = duplicates == 0 and self_loops == 0 and count_mismatch == 0
    return {
        "gate": "G0.3_pair_coverage",
        "n_graphs": int(n),
        "duplicate_rows": int(duplicates),
        "self_loops": int(self_loops),
        "count_mismatch": int(count_mismatch),
        "passed": bool(passed),
    }


def _gate_reconstruct_R(export: Mapping[str, Any]) -> dict[str, Any]:
    reconstructed = _reconstruct_R(export)
    diff = np.abs(reconstructed.astype(np.float64) - export["R"].astype(np.float64))
    return {
        "gate": "G0.4_reconstruct_R",
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
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
        "gate": "G0.5_reconstruct_prediction",
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "atol": PRED_RECON_ATOL,
        "passed": bool(diff.max() <= PRED_RECON_ATOL),
    }


def _load_model_for_fold(fold: int, seed: int) -> nn.Module:
    records = _load_train_records()
    config = _frozen_config()
    inner_train_idx, _inner_valid_idx, holdout_idx = _fold_slices()[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    holdout_records = [records[int(i)] for i in holdout_idx]
    _encoded_fit, _encoded_other, audit = zpp._phase_data(
        inner_train_records, holdout_records, config=config
    )
    config_mod = dict(config)
    config_mod["model"] = _model_config_with_clamps(config, audit)
    model = _build_v4_model(config_mod, audit)
    state = torch.load(
        FOLD_DIR / f"fold{fold}_seed{seed}_state.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _gate_batch_invariance(fold: int, seed: int) -> dict[str, Any]:
    records = _load_train_records()
    config = _frozen_config()
    inner_train_idx, _iv, holdout_idx = _fold_slices()[fold]
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

    baseline = _capture_states(model, encoded_holdout, batch_size=128)

    # reorder graphs (deterministic reversal) at a different batch size
    order = np.arange(len(encoded_holdout))[::-1].copy()
    reordered = [encoded_holdout[int(i)] for i in order]
    shifted = _capture_states(model, reordered, batch_size=97)
    inv_order = np.argsort(order)

    def _snapshot(capture, positions):
        patch_offsets = np.concatenate([[0], np.cumsum(capture["n_patches"])])
        pair_offsets = np.concatenate([[0], np.cumsum(capture["n_pairs"])])
        patches = []
        pairs = []
        for position in positions:
            p0, p1 = int(patch_offsets[position]), int(patch_offsets[position + 1])
            q0, q1 = int(pair_offsets[position]), int(pair_offsets[position + 1])
            patches.append(capture["patch_states"][p0:p1].mean(axis=0))
            pairs.append(
                capture["pair_states"][q0:q1].mean(axis=0)
                if q1 > q0
                else np.zeros(PAIR_DIM, dtype=np.float32)
            )
        return (
            capture["R"][positions],
            capture["yhat_0"][positions],
            np.stack(patches),
            np.stack(pairs),
        )

    base = _snapshot(baseline, np.arange(len(encoded_holdout)))
    other = _snapshot(shifted, inv_order)
    diffs = {
        "R": float(np.abs(base[0] - other[0]).max()),
        "yhat_0": float(np.abs(base[1] - other[1]).max()),
        "patch_mean": float(np.abs(base[2] - other[2]).max()),
        "pair_mean": float(np.abs(base[3] - other[3]).max()),
    }
    return {
        "gate": "G0.6_batch_invariance",
        "fold": int(fold),
        "seed": int(seed),
        "batch_sizes": [128, 97],
        "max_abs_diff": diffs,
        "atol": BATCH_INV_ATOL,
        "passed": bool(max(diffs.values()) <= BATCH_INV_ATOL),
    }


def _fold_transforms(fold: int) -> tuple[dict[str, Any], dict[str, Any]]:
    records = _load_train_records()
    config = _frozen_config()
    inner_train_idx, _iv, _ho = _fold_slices()[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    representation = config["representation"]
    transforms = {
        "typed_vocabulary": zpp._fit_vocabulary(
            inner_train_records,
            "typed_certificate",
            int(representation.get("max_typed_tokens", 8192)),
            int(representation.get("minimum_typed_frequency", 1)),
        ),
        "parent_vocabulary": zpp._fit_vocabulary(
            inner_train_records,
            "parent_certificate",
            int(representation.get("max_parent_tokens", 2048)),
            int(representation.get("minimum_parent_frequency", 1)),
        ),
        "patch_standardizer": zpp.Standardizer.fit(zpp._patch_matrix(inner_train_records)),
        "context_standardizer": zpp.Standardizer.fit(zpp._context_matrix(inner_train_records)),
    }
    topology_matrix = ztopo.matrices_for_split("train", _load_zinc(ZINC_ROOT, "train"), "hinge")[0]
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


def _gate_node_permutation(fold: int, seed: int, n_molecules: int = 4) -> dict[str, Any]:
    transforms, extra = _fold_transforms(fold)
    topology_matrix = extra["topology_matrix"]
    records = _load_train_records()
    holdout_idx = _fold_slices()[fold][2]
    dataset = _load_zinc(ZINC_ROOT, "train")

    model = _load_model_for_fold(fold, seed)
    rc = np.random.RandomState(0)
    chosen = sorted(
        np.random.RandomState(7).choice(len(holdout_idx), size=n_molecules, replace=False).tolist()
    )
    diffs = {"R": 0.0, "yhat_0": 0.0, "summary": 0.0}
    for position in chosen:
        subset_index = int(holdout_idx[position])
        original = records[subset_index]
        data = dataset[subset_index]
        n = int(data.num_nodes)
        perm = rc.permutation(n)
        inv = np.argsort(perm)
        edge_index = data.edge_index.detach().cpu().numpy()
        # relabel node u -> perm[u]: edge columns are remapped by ``perm`` and
        # node-type rows by the inverse ``inv`` (new row p keeps the type of
        # original node inv[p]).  Getting the two directions inconsistent
        # yields a *different* coloured graph, not a relabelling.
        permuted_edge_index = perm[edge_index]
        from torch_geometric.data import Data as PyGData

        permuted_data = PyGData(
            x=data.x[torch.tensor(inv, dtype=torch.long)],
            edge_index=torch.tensor(permuted_edge_index, dtype=torch.long),
            edge_attr=data.edge_attr,
            y=data.y,
            num_nodes=n,
        )
        global_context = global_feature_views([permuted_data])["global_all"][0]
        permuted_record = zpp._graph_record(
            permuted_data,
            global_context,
            {},
            patch_radius=zpp.PATCH_RADIUS,
            context_radius=0,
            structural_mode="none",
            topology_features=topology_matrix[subset_index],
            tokenizer_version=TOKENIZER_VERSION,
            attribute_mode="none",
        )
        encoded_original = _encode_with_transforms([original], transforms)[0]
        encoded_permuted = _encode_with_transforms([permuted_record], transforms)[0]
        cap_original = _capture_states(model, [encoded_original], batch_size=1)
        cap_permuted = _capture_states(model, [encoded_permuted], batch_size=1)
        diffs["R"] = max(diffs["R"], float(np.abs(cap_original["R"] - cap_permuted["R"]).max()))
        diffs["yhat_0"] = max(
            diffs["yhat_0"], float(np.abs(cap_original["yhat_0"] - cap_permuted["yhat_0"]).max())
        )
        patch_mean_orig = cap_original["patch_states"].mean(axis=0)
        patch_mean_perm = cap_permuted["patch_states"].mean(axis=0)
        pair_mean_orig = (
            cap_original["pair_states"].mean(axis=0)
            if cap_original["pair_states"].size
            else np.zeros(PAIR_DIM)
        )
        pair_mean_perm = (
            cap_permuted["pair_states"].mean(axis=0)
            if cap_permuted["pair_states"].size
            else np.zeros(PAIR_DIM)
        )
        diffs["summary"] = max(
            diffs["summary"],
            float(np.abs(patch_mean_orig - patch_mean_perm).max()),
            float(np.abs(pair_mean_orig - pair_mean_perm).max()),
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


def run_gate0(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = RESULTS_DIR / "export_integrity_report.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    # per-fold gates over the primary backbone seed (and any built seed)
    per_fold: dict[str, Any] = {}
    seeds_to_check = [s for s in BACKBONE_SEEDS if _export_path(0, s).exists()]
    for seed in seeds_to_check:
        for fold in range(K_FOLDS):
            export = load_export(_export_path(fold, seed))
            model = _load_model_for_fold(fold, seed)
            gates = {
                "G0.1": _gate_pair_grouping(export),
                "G0.2": _gate_pair_endpoint_identity(export),
                "G0.3": _gate_pair_coverage(export),
                "G0.4": _gate_reconstruct_R(export),
                "G0.5": _gate_reconstruct_prediction(export, model),
            }
            per_fold[f"fold{fold}_seed{seed}"] = gates
            print(
                f"[gate0] fold={fold} seed={seed} "
                + " ".join(f"{k}={'PASS' if v['passed'] else 'FAIL'}" for k, v in gates.items()),
                flush=True,
            )

    batch_inv = _gate_batch_invariance(0, PRIMARY_BACKBONE_SEED) if _export_path(0, PRIMARY_BACKBONE_SEED).exists() else {"passed": False, "reason": "export missing"}
    perm_inv = _gate_node_permutation(0, PRIMARY_BACKBONE_SEED) if _export_path(0, PRIMARY_BACKBONE_SEED).exists() else {"passed": False, "reason": "export missing"}

    all_gates = []
    for gates in per_fold.values():
        all_gates.extend(gates.values())
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
# target-independent 1200/400/400 split
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
    rarity = pd.read_csv(RARITY_CSV)
    rarity = rarity.set_index("subset_index")
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
# FoldStates: build adapter tensors from an export
# ---------------------------------------------------------------------------

class FoldStates:
    def __init__(self, export: Mapping[str, Any]) -> None:
        self.subset_index = np.asarray(export["subset_index"], dtype=np.int64)
        self.R = np.asarray(export["R"], dtype=np.float32)
        self.target = np.asarray(export["target"], dtype=np.float64)
        self.yhat_0 = np.asarray(export["yhat_0"], dtype=np.float64)
        self.n_patches = np.asarray(export["n_patches"], dtype=np.int64)
        self.n_pairs = np.asarray(export["n_pairs"], dtype=np.int64)
        self.patch_states = np.asarray(export["patch_states"], dtype=np.float32)
        self.pair_states = np.asarray(export["pair_states"], dtype=np.float32)
        self.pair_bucket = np.asarray(export["pair_bucket"], dtype=np.int64)
        self.patch_offsets = np.concatenate([[0], np.cumsum(self.n_patches)])
        self.pair_offsets = np.concatenate([[0], np.cumsum(self.n_pairs)])

    def positions_for(self, subset_indices: np.ndarray) -> np.ndarray:
        lookup = {int(v): i for i, v in enumerate(self.subset_index)}
        return np.asarray([lookup[int(v)] for v in subset_indices], dtype=np.int64)

    def tensors(self, positions: np.ndarray) -> dict[str, torch.Tensor]:
        positions = np.asarray(positions, dtype=np.int64)
        n = len(positions)
        patch_rows = []
        pair_rows = []
        patch_graph = []
        pair_graph = []
        pair_bucket = []
        for local, position in enumerate(positions):
            p0, p1 = int(self.patch_offsets[position]), int(self.patch_offsets[position + 1])
            patch_rows.append(self.patch_states[p0:p1])
            patch_graph.append(np.full(p1 - p0, local, dtype=np.int64))
            q0, q1 = int(self.pair_offsets[position]), int(self.pair_offsets[position + 1])
            pair_rows.append(self.pair_states[q0:q1])
            pair_graph.append(np.full(q1 - q0, local, dtype=np.int64))
            pair_bucket.append(self.pair_bucket[q0:q1])
        return {
            "R": torch.tensor(self.R[positions], dtype=torch.float32),
            "y": torch.tensor(self.target[positions], dtype=torch.float32),
            "yhat_0": torch.tensor(self.yhat_0[positions], dtype=torch.float32),
            "patch_states": torch.tensor(np.concatenate(patch_rows, axis=0), dtype=torch.float32),
            "patch_graph": torch.tensor(np.concatenate(patch_graph), dtype=torch.long),
            "pair_states": torch.tensor(np.concatenate(pair_rows, axis=0), dtype=torch.float32),
            "pair_graph": torch.tensor(np.concatenate(pair_graph), dtype=torch.long),
            "pair_bucket": torch.tensor(np.concatenate(pair_bucket), dtype=torch.long),
            "n": n,
        }


def _standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

class RonlyHead(nn.Module):
    """Parameter-matched control B1: reads only R."""

    def __init__(self, r_dim: int = R_DIM, hidden: int = RONLY_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(r_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.r_mean: torch.Tensor | None = None
        self.r_scale: torch.Tensor | None = None

    def set_standardizer(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.r_mean = torch.tensor(mean, dtype=torch.float32)
        self.r_scale = torch.tensor(scale, dtype=torch.float32)

    def forward(self, tensors: Mapping[str, torch.Tensor], use_summary: bool = True) -> torch.Tensor:
        value = tensors["R"]
        if self.r_mean is not None:
            value = (value - self.r_mean) / self.r_scale
        return tensors["yhat_0"] + self.net(value).squeeze(-1)


class SetAdapter(nn.Module):
    """Experimental model S: frozen set states + R -> residual."""

    def __init__(self) -> None:
        super().__init__()
        self.phi_h = nn.Sequential(
            nn.Linear(PATCH_DIM, SET_PHI_H_HIDDEN),
            nn.ReLU(),
            nn.Linear(SET_PHI_H_HIDDEN, SET_PHI_H_OUT),
        )
        self.phi_q = nn.Sequential(
            nn.Linear(PAIR_DIM, SET_PHI_Q_HIDDEN),
            nn.ReLU(),
            nn.Linear(SET_PHI_Q_HIDDEN, SET_PHI_Q_OUT),
        )
        rho_in = R_DIM + SET_PHI_H_OUT + N_BUCKETS * SET_PHI_Q_OUT
        self.rho = nn.Sequential(
            nn.Linear(rho_in, SET_RHO_HIDDEN),
            nn.ReLU(),
            nn.Linear(SET_RHO_HIDDEN, 1),
        )
        for name in ("r_mean", "r_scale", "patch_mean", "patch_scale", "pair_mean", "pair_scale"):
            self.register_buffer(name, None)

    def set_standardizers(self, r, patch, pair) -> None:
        r_mean, r_scale = r
        patch_mean, patch_scale = patch
        pair_mean, pair_scale = pair
        self.r_mean = torch.tensor(r_mean, dtype=torch.float32)
        self.r_scale = torch.tensor(r_scale, dtype=torch.float32)
        self.patch_mean = torch.tensor(patch_mean, dtype=torch.float32)
        self.patch_scale = torch.tensor(patch_scale, dtype=torch.float32)
        self.pair_mean = torch.tensor(pair_mean, dtype=torch.float32)
        self.pair_scale = torch.tensor(pair_scale, dtype=torch.float32)

    def _bucket_means(self, values, pair_graph, n, bucket):
        out = []
        for b in range(N_BUCKETS):
            mask = bucket == b
            current = values[mask]
            graph = pair_graph[mask]
            acc = torch.zeros((n, values.shape[1]), dtype=values.dtype)
            if current.numel():
                acc = acc.index_add(0, graph, current)
                counts = torch.zeros((n, 1), dtype=values.dtype)
                counts = counts.index_add(
                    0, graph, torch.ones((current.shape[0], 1), dtype=values.dtype)
                )
                acc = acc / counts.clamp_min(1.0)
            out.append(acc)
        return torch.cat(out, dim=1)

    def summarize(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        n = int(tensors["n"])
        patch = tensors["patch_states"]
        pair = tensors["pair_states"]
        if self.patch_mean is not None:
            patch = (patch - self.patch_mean) / self.patch_scale
            pair = (pair - self.pair_mean) / self.pair_scale
        ph = self.phi_h(patch)
        acc = torch.zeros((n, SET_PHI_H_OUT), dtype=ph.dtype)
        acc = acc.index_add(0, tensors["patch_graph"], ph)
        counts = torch.zeros((n, 1), dtype=ph.dtype)
        counts = counts.index_add(
            0, tensors["patch_graph"], torch.ones((ph.shape[0], 1), dtype=ph.dtype)
        )
        s_h = acc / counts.clamp_min(1.0)
        pq = self.phi_q(pair)
        s_buckets = self._bucket_means(
            pq, tensors["pair_graph"], n, tensors["pair_bucket"]
        )
        return torch.cat([s_h, s_buckets], dim=1)

    def forward(self, tensors: Mapping[str, torch.Tensor], use_summary: bool = True) -> torch.Tensor:
        R = tensors["R"]
        if self.r_mean is not None:
            R = (R - self.r_mean) / self.r_scale
        if use_summary:
            summary = self.summarize(tensors)
        else:
            summary = torch.zeros(
                (int(tensors["n"]), SET_PHI_H_OUT + N_BUCKETS * SET_PHI_Q_OUT),
                dtype=R.dtype,
            )
        return tensors["yhat_0"] + self.rho(torch.cat([R, summary], dim=1)).squeeze(-1)


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _make_ronly(seed: int) -> RonlyHead:
    torch.manual_seed(int(seed))
    return RonlyHead()


def _make_set(seed: int) -> SetAdapter:
    torch.manual_seed(int(seed))
    return SetAdapter()


def _standardize_all(
    fit: Mapping[str, torch.Tensor],
    others: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    r_mean, r_scale = _standardizer(fit["R"].numpy())
    patch_mean, patch_scale = _standardizer(fit["patch_states"].numpy())
    pair_mean, pair_scale = _standardizer(fit["pair_states"].numpy())
    standardizers = {
        "R": (r_mean, r_scale),
        "patch": (patch_mean, patch_scale),
        "pair": (pair_mean, pair_scale),
    }
    return standardizers


def _train_adapter(
    model: nn.Module,
    fit: Mapping[str, torch.Tensor],
    selection: Mapping[str, torch.Tensor],
    seed: int,
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
    for _epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        prediction = model(fit)
        loss = (prediction - fit["y"]).abs().mean()
        loss.backward()
        if not grad_alive:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            grad_alive = bool(grads) and any(float(g.abs().sum()) > 0 for g in grads)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            selection_mae = float((model(selection) - selection["y"]).abs().mean())
        if selection_mae < best_selection - 1e-9:
            best_selection = selection_mae
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    model.load_state_dict(best_state)
    return {
        "best_selection_mae": best_selection,
        "grad_alive": bool(grad_alive),
        "epochs": int(_epoch + 1),
    }


def _evaluate_adapter(model: nn.Module, tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        prediction = model(tensors)
        with_s = prediction
        no_s = model(tensors, use_summary=False) if isinstance(model, SetAdapter) else None
    mae = float((prediction - tensors["y"]).abs().mean())
    result = {"mae": mae}
    if no_s is not None:
        result["sensitivity_max_abs_diff"] = float((with_s - no_s).abs().max())
        result["prediction_std"] = float(with_s.std())
    return result


# ---------------------------------------------------------------------------
# joint-structure control (B2)
# ---------------------------------------------------------------------------

def _shuffle_states(
    tensors: Mapping[str, torch.Tensor], rng: np.random.RandomState
) -> dict[str, torch.Tensor]:
    """Marginal-preserving per-channel row shuffle within graph (and bucket)."""
    out = dict(tensors)
    patch = tensors["patch_states"].numpy().copy()
    graph = tensors["patch_graph"].numpy()
    for node in np.unique(graph):
        rows = np.flatnonzero(graph == node)
        for channel in range(patch.shape[1]):
            patch[rows, channel] = patch[rows, channel][rng.permutation(len(rows))]
    out["patch_states"] = torch.tensor(patch, dtype=torch.float32)

    pair = tensors["pair_states"].numpy().copy()
    pgraph = tensors["pair_graph"].numpy()
    bucket = tensors["pair_bucket"].numpy()
    for node in np.unique(pgraph):
        for b in range(N_BUCKETS):
            rows = np.flatnonzero((pgraph == node) & (bucket == b))
            for channel in range(pair.shape[1]):
                pair[rows, channel] = pair[rows, channel][rng.permutation(len(rows))]
    out["pair_states"] = torch.tensor(pair, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def _stratified_bootstrap(
    delta: np.ndarray, fold_ids: np.ndarray, n_boot: int = 10000, seed: int = 20260910
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    folds = np.unique(fold_ids)
    index_by_fold = [np.flatnonzero(fold_ids == f) for f in folds]
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        chunks = []
        for idx in index_by_fold:
            pick = rng.integers(0, len(idx), len(idx))
            chunks.append(delta[idx[pick]])
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
# Stage 1 / Stage 2 run
# ---------------------------------------------------------------------------

def _run_stage_for_fold(
    fold: int,
    backbone_seed: int,
    adapter_seed: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    export = load_export(_export_path(fold, backbone_seed))
    states = FoldStates(export)
    fold_manifest = manifest["folds"][f"fold{fold}"]

    fit_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_fit"]], dtype=np.int64)
    sel_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_selection"]], dtype=np.int64)
    eva_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_evaluation"]], dtype=np.int64)

    fit = states.tensors(states.positions_for(fit_ids))
    selection = states.tensors(states.positions_for(sel_ids))
    evaluation = states.tensors(states.positions_for(eva_ids))

    standardizers = _standardize_all(fit, [selection, evaluation])

    # --- B1 R-only ---
    ronly = _make_ronly(adapter_seed)
    ronly.set_standardizer(*standardizers["R"])
    ronly_train = _train_adapter(ronly, fit, selection, adapter_seed)
    _evaluate_adapter(ronly, evaluation)

    # --- S set adapter ---
    setter = _make_set(adapter_seed)
    setter.set_standardizers(
        standardizers["R"], standardizers["patch"], standardizers["pair"]
    )
    set_train = _train_adapter(setter, fit, selection, adapter_seed)
    set_eval = _evaluate_adapter(setter, evaluation)

    y = evaluation["y"].numpy()
    yhat0 = evaluation["yhat_0"].numpy()
    with torch.no_grad():
        ronly_pred = ronly(evaluation).numpy()
        set_pred = setter(evaluation).numpy()
    mae_b0 = float(np.abs(y - yhat0).mean())
    mae_b1 = float(np.abs(y - ronly_pred).mean())
    mae_s = float(np.abs(y - set_pred).mean())

    # common-input bulk safety (target-independent, input-only)
    rarity = pd.read_csv(RARITY_CSV).set_index("subset_index")
    rare_ratio = rarity.loc[eva_ids, "rare_le5_ratio"].to_numpy(dtype=np.float64)
    bulk_mask = rare_ratio < float(fold_manifest["rare_le5_threshold"])
    if bulk_mask.sum() >= 10:
        bulk_b0 = float(np.abs(y - yhat0)[bulk_mask].mean())
        bulk_s = float(np.abs(y - set_pred)[bulk_mask].mean())
        bulk_delta = bulk_s - bulk_b0
        bulk_delta_b1 = float(np.abs(y - ronly_pred)[bulk_mask].mean()) - bulk_b0
    else:
        bulk_delta = float("nan")
        bulk_delta_b1 = float("nan")

    return {
        "fold": fold,
        "backbone_seed": backbone_seed,
        "adapter_seed": adapter_seed,
        "mae_b0": mae_b0,
        "mae_b1": mae_b1,
        "mae_s": mae_s,
        "delta0": mae_b0 - mae_s,
        "delta_R": mae_b1 - mae_s,
        "bulk_degradation_s": bulk_delta,
        "bulk_degradation_b1": bulk_delta_b1,
        "n_bulk_eval": int(bulk_mask.sum()),
        "ronly_params": _count_parameters(ronly),
        "set_params": _count_parameters(setter),
        "ronly_selection_mae": ronly_train["best_selection_mae"],
        "set_selection_mae": set_train["best_selection_mae"],
        "set_grad_alive": set_train["grad_alive"],
        "set_sensitivity_max_abs_diff": set_eval.get("sensitivity_max_abs_diff", float("nan")),
        "set_prediction_std": set_eval.get("prediction_std", float("nan")),
        "n_eval": int(len(eva_ids)),
        # per-molecule errors kept out of the CSV (underscore-prefixed) but
        # used directly for the paired stratified bootstrap.
        "_molecules": {
            "subset_index": eva_ids,
            "fold": np.full(len(eva_ids), fold, dtype=np.int64),
            "err_b0": np.abs(y - yhat0),
            "err_b1": np.abs(y - ronly_pred),
            "err_s": np.abs(y - set_pred),
        },
    }


def _aggregate_stage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame([
        {k: v for k, v in row.items() if not k.startswith("_")} for row in rows
    ])
    delta0 = frame["delta0"].to_numpy(dtype=np.float64)
    delta_r = frame["delta_R"].to_numpy(dtype=np.float64)
    bulk = frame["bulk_degradation_s"].to_numpy(dtype=np.float64)
    bulk = bulk[np.isfinite(bulk)]
    return {
        "mean_delta0": float(np.mean(delta0)),
        "median_delta0": float(np.median(delta0)),
        "mean_delta_R": float(np.mean(delta_r)),
        "median_delta_R": float(np.median(delta_r)),
        "positive_folds_delta0": int(np.sum(delta0 > 0)),
        "positive_folds_delta_R": int(np.sum(delta_r > 0)),
        "n_folds": int(len(frame)),
        "per_fold_delta0": delta0.tolist(),
        "per_fold_delta_R": delta_r.tolist(),
        "mean_b0": float(frame["mae_b0"].mean()),
        "mean_b1": float(frame["mae_b1"].mean()),
        "mean_s": float(frame["mae_s"].mean()),
        "mean_bulk_degradation_s": float(bulk.mean()) if bulk.size else None,
        "max_bulk_degradation_s": float(bulk.max()) if bulk.size else None,
        "ronly_params": int(frame["ronly_params"].iloc[0]),
        "set_params": int(frame["set_params"].iloc[0]),
        "set_grad_alive_all": bool(frame["set_grad_alive"].all()),
        "set_sensitivity_min": float(frame["set_sensitivity_max_abs_diff"].min()),
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
        cached_bootstrap = json.loads(out_boot.read_text(encoding="utf-8"))
        return {
            "csv": str(out_csv),
            "bootstrap": str(out_boot),
            "aggregate": cached_bootstrap["aggregate"],
            "bootstrap_detail": {
                "delta0": cached_bootstrap["delta0"],
                "delta_R": cached_bootstrap["delta_R"],
            },
            "cached": True,
        }

    rows = [
        _run_stage_for_fold(fold, backbone_seed, adapter_seed, manifest)
        for fold in range(K_FOLDS)
    ]
    frame = pd.DataFrame(
        [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    )
    frame.to_csv(out_csv, index=False)

    molecule = [row["_molecules"] for row in rows]
    delta0 = np.concatenate([m["err_b0"] - m["err_s"] for m in molecule])
    delta_r = np.concatenate([m["err_b1"] - m["err_s"] for m in molecule])
    fold_ids = np.concatenate([m["fold"] for m in molecule])
    bootstrap = {
        "backbone_seed": int(backbone_seed),
        "adapter_seed": int(adapter_seed),
        "delta0": _stratified_bootstrap(delta0, fold_ids),
        "delta_R": _stratified_bootstrap(delta_r, fold_ids),
        "aggregate": _aggregate_stage(rows),
        "seconds": time.perf_counter() - started,
    }
    _write_json_any(out_boot, bootstrap)
    print(
        f"[{tag}] seed={backbone_seed} init={adapter_seed} "
        f"mean d0={bootstrap['aggregate']['mean_delta0']:+.4f} "
        f"mean dR={bootstrap['aggregate']['mean_delta_R']:+.4f}"
        f" ({time.perf_counter() - started:.0f}s)",
        flush=True,
    )
    return {
        "csv": str(out_csv),
        "bootstrap": str(out_boot),
        "aggregate": bootstrap["aggregate"],
        "bootstrap_detail": {"delta0": bootstrap["delta0"], "delta_R": bootstrap["delta_R"]},
        "cached": False,
    }


# ---------------------------------------------------------------------------
# B2 joint-structure control
# ---------------------------------------------------------------------------

def run_b2(manifest: Mapping[str, Any], force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    out_csv = RESULTS_DIR / "joint_structure_control.csv"
    if out_csv.exists() and not force:
        return {"csv": str(out_csv), "cached": True}
    rows: list[dict[str, Any]] = []
    for fold in range(K_FOLDS):
        export = load_export(_export_path(fold, PRIMARY_BACKBONE_SEED))
        states = FoldStates(export)
        fold_manifest = manifest["folds"][f"fold{fold}"]
        fit_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_fit"]], dtype=np.int64)
        sel_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_selection"]], dtype=np.int64)
        eva_ids = np.asarray([int(m.split(":")[1]) for m in fold_manifest["molecule_ids"]["adapter_evaluation"]], dtype=np.int64)
        fit = states.tensors(states.positions_for(fit_ids))
        selection = states.tensors(states.positions_for(sel_ids))
        evaluation = states.tensors(states.positions_for(eva_ids))
        rng = np.random.RandomState(20260910 + fold)
        fit_shuf = _shuffle_states(fit, rng)
        sel_shuf = _shuffle_states(selection, rng)
        eva_shuf = _shuffle_states(evaluation, rng)
        standardizers = _standardize_all(fit, [selection, evaluation])
        torch.manual_seed(0)
        setter = SetAdapter()
        setter.set_standardizers(standardizers["R"], standardizers["patch"], standardizers["pair"])
        _train_adapter(setter, fit_shuf, sel_shuf, 0)
        with torch.no_grad():
            pred = setter(eva_shuf).numpy()
        y = evaluation["y"].numpy()
        mae = float(np.abs(y - pred).mean())
        # clean reference (B0, B1, S) recomputed quickly for the same fold
        clean = _run_stage_for_fold(fold, PRIMARY_BACKBONE_SEED, 0, manifest)
        rows.append(
            {
                "fold": fold,
                "mae_b0": clean["mae_b0"],
                "mae_b1": clean["mae_b1"],
                "mae_s_clean": clean["mae_s"],
                "mae_s_shuffled": mae,
                "shuffle_loss": float(clean["mae_s"] - mae) * -1.0,
                "delta_R_clean": clean["delta_R"],
                "delta_R_shuffled": float(clean["mae_b1"] - mae),
            }
        )
        print(
            f"[b2] fold={fold} clean S={clean['mae_s']:.5f} shuffled S={mae:.5f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    return {"csv": str(out_csv), "rows": rows, "seconds": time.perf_counter() - started}


# ---------------------------------------------------------------------------
# decision logic
# ---------------------------------------------------------------------------

def _decide_stage1(stage1: Mapping[str, Any]) -> str:
    agg = stage1["aggregate"]
    d0 = agg["mean_delta0"]
    dr = agg["mean_delta_R"]
    pos_dr = agg["positive_folds_delta_R"]
    if d0 < S1_NGO_D0 and dr <= S1_NGO_DR and pos_dr < 4:
        return "STAGE1_CLEAR_NO_GO"
    if d0 >= S1_ADV_D0 and dr >= S1_ADV_DR and pos_dr >= 4:
        return "STAGE1_ADVANCE"
    return "STAGE1_BORDERLINE"


def _final_decision(
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any] | None,
    stage1b: Mapping[str, Any] | None,
    b2: Mapping[str, Any] | None,
) -> dict[str, Any]:
    s1 = stage1["aggregate"]
    d0 = s1["mean_delta0"]
    dr = s1["mean_delta_R"]

    if stage2 is not None:
        s2 = stage2["aggregate"]
        seeds_mean_dr = [s1["mean_delta_R"], s2["mean_delta_R"]]
        pooled_d0 = (s1["mean_delta0"] + s2["mean_delta0"]) / 2.0
        pooled_dr = (s1["mean_delta_R"] + s2["mean_delta_R"]) / 2.0
        per_fold_dr = np.mean(
            [s1["per_fold_delta_R"], s2["per_fold_delta_R"]], axis=0
        )
        folds_positive = int(np.sum(per_fold_dr > 0))
        boot_dr = stage2["bootstrap_detail"]["delta_R"]
        ci_low_positive = stage1["bootstrap_detail"]["delta_R"]["ci95_low"] > 0 and boot_dr["ci95_low"] > 0
        backbone_consistent = all(v > 0 for v in seeds_mean_dr)
    else:
        pooled_d0 = d0
        pooled_dr = dr
        per_fold_dr = np.asarray(s1["per_fold_delta_R"])
        folds_positive = int(np.sum(per_fold_dr > 0))
        boot_dr = stage1["bootstrap_detail"]["delta_R"]
        ci_low_positive = boot_dr["ci95_low"] > 0
        backbone_consistent = d0 > 0 and dr > 0

    bulk_values = [s1.get("max_bulk_degradation_s")]
    if stage2 is not None:
        bulk_values.append(stage2["aggregate"].get("max_bulk_degradation_s"))
    bulk_values = [float(v) for v in bulk_values if v is not None and np.isfinite(v)]
    bulk_safe = bool(bulk_values) and max(bulk_values) <= BULK_MAX_DEGRADATION

    criteria = {
        "pooled_delta0_ge_0.003": pooled_d0 >= FINAL_GO_D0,
        "pooled_delta_R_ge_0.002": pooled_dr >= FINAL_GO_DR,
        "backbone_consistent_positive": backbone_consistent,
        "folds_at_least_4_of_5_positive": folds_positive >= 4,
        "paired_ci_lower_positive": ci_low_positive,
        "bulk_safe_le_0.002": bulk_safe,
    }
    if all(criteria.values()):
        verdict = "GO"
    elif pooled_dr <= 0 or pooled_dr < FINAL_GO_DR * 0.5 or pooled_d0 < S1_NGO_D0:
        verdict = "NO_GO"
    elif pooled_d0 < FINAL_GO_D0 or not backbone_consistent or folds_positive < 4:
        verdict = "NO_GO"
    else:
        verdict = "INCONCLUSIVE"

    if verdict == "GO":
        claim = (
            "Given frozen compact-v4 states and the existing 302D graph vector, a "
            "small permutation-invariant set adapter provides reproducible incremental "
            "predictive value beyond a parameter-matched R-only residual head."
        )
    elif verdict == "NO_GO":
        claim = (
            "No useful incremental signal was found from a low-capacity nonlinear set "
            "summary of the existing frozen v4 states beyond an equally sized readout "
            "of the existing graph vector."
        )
    else:
        claim = (
            "The frozen set states may carry a small incremental signal but the pooled "
            "estimate is not resolved by the current budget; budget was not expanded."
        )
    return {
        "pooled_delta0": float(pooled_d0),
        "pooled_delta_R": float(pooled_dr),
        "per_fold_delta_R_pooled": per_fold_dr.tolist(),
        "positive_folds_delta_R": folds_positive,
        "delta_R_ci95": [boot_dr["ci95_low"], boot_dr["ci95_high"]],
        "bulk_degradation_s_max": float(max(bulk_values)) if bulk_values else None,
        "criteria": criteria,
        "verdict": verdict,
        "claim": claim,
        "scope_caveat": (
            "This is a screening diagnostic on the frozen OOF states of the existing "
            "backbone.  A NO-GO is about a cheap post-state readout extension only; it "
            "does NOT prove the moment pooling is sufficient, nor that the patch "
            "paradigm has reached its ceiling.  A GO would not prove that moment "
            "pooling definitively loses information (information accessibility and "
            "statistical parameterisation remain entangled)."
        ),
        "stage1b": stage1b,
        "b2": b2,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _make_figures(stage1: Mapping[str, Any], stage2: Mapping[str, Any] | None, b2: Mapping[str, Any] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RESULTS_DIR / "stage1_fold_results.csv")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(np.arange(K_FOLDS) - 0.2, frame["delta0"], width=0.4, label=r"$\Delta_0$")
    ax.bar(np.arange(K_FOLDS) + 0.2, frame["delta_R"], width=0.4, label=r"$\Delta_R$")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("MAE improvement (gold - new)")
    ax.set_title("Figure 1: per-fold set-adapter gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_per_fold_delta.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(K_FOLDS)
    ax.plot(x, frame["mae_b0"], marker="o", label="B0 frozen v4")
    ax.plot(x, frame["mae_b1"], marker="s", label="B1 R-only")
    ax.plot(x, frame["mae_s"], marker="^", label="S set adapter")
    ax.set_xlabel("outer fold")
    ax.set_ylabel("adapter-evaluation MAE")
    ax.set_title("Figure 2: B0 vs B1 vs Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_mae_comparison.png", dpi=150)
    plt.close(fig)

    if b2 is not None and (RESULTS_DIR / "joint_structure_control.csv").exists():
        b2_frame = pd.read_csv(RESULTS_DIR / "joint_structure_control.csv")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(np.arange(K_FOLDS), b2_frame["mae_s_clean"], marker="o", label="clean S")
        ax.plot(np.arange(K_FOLDS), b2_frame["mae_s_shuffled"], marker="x", label="marginal-preserving shuffle")
        ax.set_xlabel("outer fold")
        ax.set_ylabel("adapter-evaluation MAE")
        ax.set_title("Figure 3: joint-structure control")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "figure3_joint_structure_control.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["inventory", "export", "gate0", "splits", "stage1", "stage1b", "stage2", "b2", "figures", "all"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        print(json.dumps(checkpoint_inventory(), indent=2, sort_keys=True))
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
        run_stage(BACKBONE_SEEDS[1], 0, "stage2", manifest, force=args.force)
        return 0
    if args.stage == "b2":
        manifest = stage_splits()
        run_b2(manifest, force=args.force)
        return 0
    if args.stage == "figures":
        stage1 = json.loads((RESULTS_DIR / "stage1_bootstrap.json").read_text())
        stage2 = json.loads((RESULTS_DIR / "stage2_bootstrap.json").read_text()) if (RESULTS_DIR / "stage2_bootstrap.json").exists() else None
        b2 = json.loads((RESULTS_DIR / "b2_summary.json").read_text()) if (RESULTS_DIR / "b2_summary.json").exists() else None
        _make_figures(stage1, stage2, b2)
        return 0
    if args.stage == "all":
        return _run_all(force=args.force)
    return 0


def _run_all(force: bool = False) -> int:
    print("=== checkpoint inventory ===", flush=True)
    inventory = checkpoint_inventory()
    _write_json_any(RESULTS_DIR / "checkpoint_inventory.json", inventory)
    if not inventory["entries"]:
        print("no checkpoints found; STOP (missing checkpoint inventory).")
        return 2

    print("=== export + gate0 ===", flush=True)
    stage_export(seeds=(PRIMARY_BACKBONE_SEED,), force=force)
    report = run_gate0(force=force)
    if not report.get("all_passed"):
        print("GATE 0 FAILED - measurement invalid; no adapter trained.", flush=True)
        return 3

    manifest = stage_splits()
    print("=== stage1 ===", flush=True)
    stage1 = run_stage(PRIMARY_BACKBONE_SEED, 0, "stage1", manifest, force=force)
    verdict1 = _decide_stage1(stage1)

    stage1b = None
    stage2 = None
    b2 = None
    if verdict1 == "STAGE1_CLEAR_NO_GO":
        print("Stage 1 CLEAR NO-GO; no second backbone seed.", flush=True)
    else:
        if verdict1 == "STAGE1_BORDERLINE":
            print("=== stage1b (second adapter init, same backbone) ===", flush=True)
            stage1b = run_stage(PRIMARY_BACKBONE_SEED, 1, "stage1b", manifest, force=force)
            agg_b = stage1b["aggregate"]
            collapses = (
                agg_b["mean_delta_R"] <= 0.0
                or np.sign(agg_b["mean_delta_R"]) != np.sign(stage1["aggregate"]["mean_delta_R"])
            )
            if collapses:
                print("Stage 1b signal collapses/flips -> NO-GO, no second backbone.", flush=True)
                verdict1 = "STAGE1_CLEAR_NO_GO"
            else:
                verdict1 = "STAGE1_ADVANCE"
        if verdict1 == "STAGE1_ADVANCE":
            print("=== stage2 (second frozen backbone seed) ===", flush=True)
            stage_export(seeds=(BACKBONE_SEEDS[1],), force=force)
            stage2 = run_stage(BACKBONE_SEEDS[1], 0, "stage2", manifest, force=force)

    meaningful = stage1["aggregate"]["mean_delta_R"] > 0 and (
        stage2 is None or stage2["aggregate"]["mean_delta_R"] > 0
    )
    if meaningful:
        print("=== b2 joint-structure control ===", flush=True)
        b2 = run_b2(manifest, force=force)

    decision = _final_decision(stage1, stage2, stage1b, b2)
    decision["stage1_verdict"] = verdict1
    _write_json_any(RESULTS_DIR / "final_decision.json", decision)
    _write_json_any(RESULTS_DIR / "stage1_decision.json", {"verdict": verdict1})
    _make_figures(stage1, stage2, b2)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
