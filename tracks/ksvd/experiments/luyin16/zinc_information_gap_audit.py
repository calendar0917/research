"""Information-gap audit of Compact-v2 on ZINC (diagnosis only, no model dev).

Pre-registered question: compact-v2 reaches valid MAE 0.184158 / test
0.135362 with ~99k parameters.  Which information does it systematically
fail to explain?  This module does NOT train any new network beyond the
baseline protocol.  It only

1. reproduces the exact compact-v2 selection-phase model (bit-identity gate
   against the promoted run trace),
2. dumps per-molecule validation + out-of-fold (OOF) train predictions,
3. computes label-free per-molecule structural / patch statistics,
4. evaluates simple linear residual probes (StandardScaler + Ridge) fitted
   on OOF train residuals only, evaluated on validation,
5. writes the per-molecule audit table + diagnostic figures,
6. (separate stage) descriptive train/valid/test split audit.

Leakage rules (hard):

* probe fitting uses ONLY out-of-fold train residuals (5-fold cross-fit of
  the baseline protocol; every fold model is trained exactly like the
  baseline selection phase, with early stopping on the official validation
  split -- identical protocol, so no fold-holdout label is ever seen);
* Ridge alpha is selected by train-internal 5-fold CV only; validation is
  transform + evaluate only;
* the official test split is touched only by the ``split`` stage; its target
  column is used ONLY for clearly-flagged post-hoc descriptive statistics.

Stages:
    features   build GraphRecords for train+valid, cache to disk
    baseline   reproduce selection phase, save per-molecule predictions
    oof        K-fold cross-fitted train predictions (baseline protocol)
    probes     audit table, correlations, binned MAE, ridge probes
    split      descriptive train/valid/test split audit (loads test)
    figures    diagnostic plots from saved probe results
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.structural_context import (
    extract_structural_context,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _all_pair_distances,
    _data_to_graph,
    _load_zinc,
)

REPO_ROOT = zpp.REPO_ROOT
V2_CONFIG_PATH = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml"
)
AUDIT_ROOT = REPO_ROOT / "tracks/ksvd/results/information_gap_audit"
SELECTION_RUN_JSON = (
    REPO_ROOT
    / "tracks/ksvd/runs/2026/09/07/20260907-193612-46c1a12f/artifacts/legacy_full_result.json"
)
EXPECTED_VALID_MAE = 0.18415821571176638
EXPECTED_EPOCH = 56
ATOM_BINS = zpp.ATOM_CATEGORIES
BOND_BINS = zpp.BOND_CATEGORIES
D_BUCKETS = zpp.DISTANCE_BUCKETS
PATCH_RADIUS = zpp.PATCH_RADIUS
MAX_CYCLE_LEN = 10  # bounded simple-cycle definition identical to compact-v3
HASH_DIM = 2048
TOKEN_BASE = 8192  # > any vocabulary size (6785 selection / 7051 refit)
BUCKET_BASE = 8  # >= D_BUCKETS (5)
DEVICE = torch.device("cpu")
OOF_FOLDS = 5
ZINC_ROOT = REPO_ROOT / "data/ZINC"

CACHE_DIR = AUDIT_ROOT / "cache"


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def _write_records(records: Sequence[Any], split: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_DIR / f"graph_records_{split}.pkl.gz", "wb") as handle:
        pickle.dump(list(records), handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_records(split: str) -> list[Any]:
    with gzip.open(CACHE_DIR / f"graph_records_{split}.pkl.gz", "rb") as handle:
        return list(pickle.load(handle))


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _v2_config() -> dict[str, Any]:
    return yaml.safe_load(V2_CONFIG_PATH.read_text(encoding="utf-8"))


def _graph_sidecars(split: str) -> list[tuple[Any, np.ndarray, dict[tuple[int, int], int]]]:
    """(networkx graph, node_types, edge_types) per molecule, cached."""
    cache_path = CACHE_DIR / f"graph_sidecars_{split}.pkl.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rb") as handle:
            return pickle.load(handle)
    dataset = _load_zinc(ZINC_ROOT, {"train": "train", "valid": "val", "test": "test"}[split])
    sidecars = [_data_to_graph(data) for data in dataset]
    with gzip.open(cache_path, "wb") as handle:
        pickle.dump(sidecars, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return sidecars


def _train_counter(train_records: Sequence[Any]) -> Counter:
    counter: Counter[bytes] = Counter()
    for record in train_records:
        for patch in record.patches:
            counter[patch.typed_certificate] += 1
    return counter


# --------------------------------------------------------------------------
# stage: features
# --------------------------------------------------------------------------

def stage_features() -> dict[str, Any]:
    started = time.perf_counter()
    certificate_cache: dict[bytes, bytes] = {}
    result: dict[str, Any] = {}
    for split in ("train", "valid"):
        dataset = _load_zinc(ZINC_ROOT, {"train": "train", "valid": "val"}[split])
        records, metadata = zpp._extract_split(
            dataset,
            split,
            certificate_cache,
            patch_radius=PATCH_RADIUS,
            context_radius=0,
            structural_mode="none",
            max_cycle_len=MAX_CYCLE_LEN,
        )
        _write_records(records, split)
        result[split] = metadata
        print(
            f"[features] {split}: {metadata['n_graphs']} graphs, "
            f"{metadata['mean_centres']:.3f} centres, {metadata['mean_pairs']:.3f} pairs "
            f"({metadata['seconds']:.1f}s)",
            flush=True,
        )
    result["certificate_cache_entries"] = int(len(certificate_cache))
    result["seconds"] = float(time.perf_counter() - started)
    _write_json(AUDIT_ROOT / "stage_features.json", result)
    return result


# --------------------------------------------------------------------------
# shared: prediction + patch-state capture
# --------------------------------------------------------------------------

def _predict_with_states(
    model: torch.nn.Module, data_list: Sequence[Any]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Predictions + per-record final patch states (48D), hook-based.

    The final ``patch`` tensor enters the unary readout and the pair
    projection.  ``center_update`` receives ``cat([patch, centre_context])``,
    so its input columns ``[:patch_hidden]`` are the pre-update patch states
    and its output is the (zero-initialised) delta; the final states are
    ``pre + delta``.
    """
    loader = DataLoader(list(data_list), batch_size=128, shuffle=False)
    captures: list[tuple[np.ndarray, np.ndarray]] = []

    def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        captures.append((inputs[0].detach().cpu().numpy(), output.detach().cpu().numpy()))

    handle = model.center_update.register_forward_hook(hook)
    predictions: list[np.ndarray] = []
    batch_data: list[Any] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            predictions.append(model(batch).cpu().numpy())
            batch_data.append(batch)
    handle.remove()

    flat_pred = np.concatenate(predictions).astype(np.float64)
    record_states: list[np.ndarray] = []
    for capture, batch in zip(captures, batch_data, strict=True):
        pre, delta = capture
        final = pre[:, : model.patch_hidden] + delta
        batch_ids = batch.batch.cpu().numpy()
        boundaries = np.flatnonzero(np.diff(batch_ids)) + 1
        offsets = np.concatenate([[0], boundaries, [len(batch_ids)]])
        for graph_id in range(int(batch.num_graphs)):
            start, end = int(offsets[graph_id]), int(offsets[graph_id + 1])
            record_states.append(final[start:end].astype(np.float64, copy=False))
    return flat_pred, record_states


def _run_selection_phase(
    fit_records: Sequence[Any],
    eval_records: Sequence[Any],
    tag: str,
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    """One baseline-protocol selection run (identical to the real v2 run).

    Returns (phase, model, audit, encoded_eval) where the returned model is
    the best-valid-epoch checkpoint and ``encoded_eval`` is the eval split
    encoded with the phase-fitted transforms (deterministic, reusable).
    """
    config = _v2_config()
    encoded_fit, encoded_eval, audit = zpp._phase_data(
        fit_records, eval_records, config=config
    )
    config = _v2_config()
    # Fold vocabularies can be smaller than the full-train ones; the hybrid
    # embedding's full-table boundary must not exceed the fitted vocabulary
    # (otherwise the model cannot be constructed).  Clamping keeps the fold
    # architecture identical to the baseline wherever possible.
    config = dict(config)
    model_config = dict(config["model"])
    typed_size = int(audit["typed_vocabulary_size_with_oov"])
    parent_size = int(audit["parent_vocabulary_size_with_oov"])
    if model_config.get("hybrid_full_typed_tokens") is not None:
        model_config["hybrid_full_typed_tokens"] = min(
            int(model_config["hybrid_full_typed_tokens"]), typed_size
        )
    if model_config.get("hybrid_full_parent_tokens") is not None:
        model_config["hybrid_full_parent_tokens"] = min(
            int(model_config["hybrid_full_parent_tokens"]), parent_size
        )
    config["model"] = model_config
    phase = zpp._train_phase(
        encoded_fit,
        encoded_eval,
        typed_vocabulary_size=typed_size,
        parent_vocabulary_size=parent_size,
        config=config,
        seed=int(config.get("seed", 0)),
        select_best=True,
        epochs=int(config["model"].get("epochs", 60)),
        shell_width=zpp._shell_width_for_radius(PATCH_RADIUS),
        context_width=0,
        direct_token_readout=False,
        phase_label=tag,
        structural_context_vocabulary_size=0,
    )
    return phase, phase["model"], audit, encoded_eval


# --------------------------------------------------------------------------
# stage: baseline reproduce
# --------------------------------------------------------------------------

def stage_baseline() -> dict[str, Any]:
    torch.set_num_threads(4)
    train_records = _read_records("train")
    valid_records = _read_records("valid")
    phase, model, audit, encoded_valid = _run_selection_phase(
        train_records, valid_records, "validation-selection"
    )
    best_mae = float(phase["best_mae"])
    selected_epoch = int(phase["selected_epoch"])
    print(
        f"[baseline] best valid MAE={best_mae:.12f} epoch={selected_epoch} "
        f"(expected {EXPECTED_VALID_MAE:.12f} / {EXPECTED_EPOCH})",
        flush=True,
    )
    reference = _read_json(SELECTION_RUN_JSON)
    trace_ref = reference["evaluation"]["valid"]["trace"]
    trace_new = phase["trace"]
    if len(trace_new) != len(trace_ref):
        raise RuntimeError("trace length mismatch")
    discrepancy = [
        (int(a["epoch"]), float(a["mae"]), float(b["mae"]))
        for a, b in zip(trace_ref, trace_new, strict=True)
        if abs(float(a["mae"]) - float(b["mae"])) > 1e-12
    ]
    if discrepancy or abs(best_mae - EXPECTED_VALID_MAE) > 1e-12:
        raise RuntimeError(
            f"compact-v2 reproduction mismatch ({len(discrepancy)} epochs differ); "
            f"best {best_mae:.12f} vs expected {EXPECTED_VALID_MAE:.12f}"
        )
    print("[baseline] BIT-IDENTICAL reproduction confirmed", flush=True)
    torch.save(model.state_dict(), AUDIT_ROOT / "selection_model.pt")

    y_valid = np.asarray([record.y for record in valid_records], dtype=np.float64)
    valid_pred, valid_states = _predict_with_states(model, encoded_valid)
    valid_mae = float(mean_absolute_error(y_valid, valid_pred))
    print(f"[baseline] per-molecule valid MAE {valid_mae:.12f}", flush=True)
    _save_npz(
        AUDIT_ROOT / "baseline_valid_predictions.npz",
        y=y_valid,
        prediction=valid_pred,
        signed_residual=y_valid - valid_pred,
        absolute_error=np.abs(y_valid - valid_pred),
    )
    with gzip.open(AUDIT_ROOT / "baseline_valid_states.pkl.gz", "wb") as handle:
        pickle.dump({"states": valid_states}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    y_train = np.asarray([record.y for record in train_records], dtype=np.float64)
    encoded_train_insample, _audit_again, _ = zpp._phase_data(
        train_records, train_records, config=_v2_config()
    )
    train_pred, _insample_states = _predict_with_states(model, encoded_train_insample)
    _save_npz(
        AUDIT_ROOT / "baseline_train_in_sample_predictions.npz",
        y=y_train,
        prediction=train_pred,
        signed_residual=y_train - train_pred,
        absolute_error=np.abs(y_train - train_pred),
        note=np.asarray([b"LEAKY: in-sample fit; not a probe training target"]),
    )
    torch.save(model.state_dict(), AUDIT_ROOT / "selection_model.pt")
    result = {
        "best_valid_mae": best_mae,
        "selected_epoch": selected_epoch,
        "parameters": int(phase["parameters"]),
        "per_molecule_valid_mae": valid_mae,
        "in_sample_train_mae": float(mean_absolute_error(y_train, train_pred)),
        "trace_agrees": True,
        "vocab_size_with_oov": int(audit["typed_vocabulary_size_with_oov"]),
        "valid_typed_coverage": float(
            audit["typed_vocabulary"]["other"]["known_occurrence_fraction"]
        ),
    }
    with gzip.open(AUDIT_ROOT / "baseline_train_in_sample_states.pkl.gz", "wb") as handle:
        pickle.dump({"states": _insample_states}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(AUDIT_ROOT / "stage_baseline.json", result)
    return result


# --------------------------------------------------------------------------
# stage: OOF train residuals
# --------------------------------------------------------------------------

def stage_oof(folds: int = OOF_FOLDS) -> dict[str, Any]:
    torch.set_num_threads(4)
    train_records = _read_records("train")
    valid_records = _read_records("valid")
    n = len(train_records)
    oof_preds = np.zeros(n, dtype=np.float64)
    fold_assign = np.zeros(n, dtype=np.int64)
    oof_states: list[np.ndarray | None] = [None] * n
    started = time.perf_counter()
    for fold in range(int(folds)):
        holdout = np.asarray([i for i in range(n) if i % int(folds) == fold], dtype=np.int64)
        fit_idx = np.asarray([i for i in range(n) if i % int(folds) != fold], dtype=np.int64)
        fit_records = [train_records[int(i)] for i in fit_idx]
        hold_records = [train_records[int(i)] for i in holdout]
        phase, model, audit, _ = _run_selection_phase(fit_records, valid_records, f"oof-fold-{fold}")
        print(
            f"[oof] fold {fold}: best valid MAE {float(phase['best_mae']):.6f} "
            f"epoch {int(phase['selected_epoch'])} ({time.perf_counter()-started:.0f}s)",
            flush=True,
        )
        # holdout is encoded with fold-train-fitted transforms/vocab (identical
        # to how the baseline encodes an eval split).
        config = _v2_config()
        _enc_fit, encoded_hold, _audit2 = zpp._phase_data(
            fit_records, hold_records, config=config
        )
        preds, states = _predict_with_states(model, encoded_hold)
        assert len(preds) == len(holdout)
        for offset, original in enumerate(holdout):
            oof_preds[int(original)] = float(preds[offset])
            oof_states[int(original)] = states[offset]
            fold_assign[int(original)] = int(fold)
    y_train = np.asarray([record.y for record in train_records], dtype=np.float64)
    _save_npz(
        AUDIT_ROOT / "oof_train_predictions.npz",
        y=y_train,
        prediction=oof_preds,
        fit_fold=fold_assign,
        signed_residual=y_train - oof_preds,
        absolute_error=np.abs(y_train - oof_preds),
    )
    with gzip.open(AUDIT_ROOT / "oof_train_states.pkl.gz", "wb") as handle:
        pickle.dump({"states": oof_states}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    result = {
        "oof_mae": float(mean_absolute_error(y_train, oof_preds)),
        "oof_rmse": float(np.sqrt(np.mean((y_train - oof_preds) ** 2))),
        "folds": int(folds),
        "seconds": float(time.perf_counter() - started),
        "note": "train residual target for probes: signed_residual = y_train - oof_prediction",
    }
    _write_json(AUDIT_ROOT / "stage_oof.json", result)
    print(f"[oof] done: train OOF MAE {result['oof_mae']:.6f} in {result['seconds']:.0f}s", flush=True)
    return result


# --------------------------------------------------------------------------
# per-molecule statistics
# --------------------------------------------------------------------------

def _molecule_size_stats(graph: Any) -> dict[str, float]:
    nodes = list(graph.nodes)
    n = int(len(nodes))
    edges = int(graph.num_edges())
    degrees = np.asarray([len(graph.neighbors(node)) for node in nodes], dtype=np.float64)
    seen: set[int] = set()
    comps = 0
    for node in nodes:
        if node in seen:
            continue
        comps += 1
        stack = [node]
        seen.add(node)
        while stack:
            cur = stack.pop()
            for nb in graph.neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    pair_dists, eccentricities = _all_pair_distances(graph)
    diameter = float(np.max(eccentricities)) if eccentricities.size else 0.0
    mean_pair = float(pair_dists.mean()) if pair_dists.size else 0.0
    max_pair = float(pair_dists.max()) if pair_dists.size else 0.0
    branch = int(np.sum(degrees >= 3))
    hist = np.zeros(int(diameter) + 1, dtype=np.float64)
    if pair_dists.size:
        hist = np.bincount(pair_dists.astype(np.int64), minlength=int(diameter) + 1)
        hist = hist.astype(np.float64) / hist.sum()
    return {
        "num_nodes": float(n),
        "num_edges": float(edges),
        "edges_per_node": float(edges / max(n, 1)),
        "diameter": diameter,
        "mean_shortest_path_distance": mean_pair,
        "max_shortest_path_distance": max_pair,
        "eccentricity_mean": float(eccentricities.mean()) if eccentricities.size else 0.0,
        "average_degree": float(degrees.mean()) if degrees.size else 0.0,
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
        "num_branching_nodes": float(branch),
        "branching_node_ratio": float(branch / max(n, 1)),
        "cycle_rank": float(edges - n + comps),
        "num_components": float(comps),
        "density": float(2.0 * edges / max(n * (n - 1), 1)),
        "frac_dist_gt_2": float(float(np.mean(pair_dists > 2)) if pair_dists.size else 0.0),
        "frac_dist_gt_3": float(float(np.mean(pair_dists > 3)) if pair_dists.size else 0.0),
        "frac_dist_gt_4": float(float(np.mean(pair_dists > 4)) if pair_dists.size else 0.0),
        "frac_dist_gt_5": float(float(np.mean(pair_dists > 5)) if pair_dists.size else 0.0),
        **{f"dist_frac_{int(d)}": float(hist[d]) for d in range(min(int(diameter), 12) + 1)},
    }


def _molecule_cycle_stats(
    graph: Any, node_types: np.ndarray, edge_types: Mapping[tuple[int, int], int]
) -> dict[str, float]:
    contexts, cycle_lengths = extract_structural_context(
        graph, node_types, edge_types, max_cycle_len=MAX_CYCLE_LEN
    )
    in_cycle = [1 if ctx.in_cycle else 0 for ctx in contexts.values()]
    lengths = np.asarray(cycle_lengths, dtype=np.float64)
    count_by_len = (
        np.bincount(lengths.astype(np.int64), minlength=MAX_CYCLE_LEN + 1).astype(np.float64)
        if lengths.size
        else np.zeros(MAX_CYCLE_LEN + 1, dtype=np.float64)
    )
    out: dict[str, float] = {
        "num_nodes_in_cycles": float(sum(in_cycle)),
        "fraction_nodes_in_cycles": float(np.mean(in_cycle)) if in_cycle else 0.0,
        "num_simple_cycles_bounded": float(len(cycle_lengths)),
        "cycle_length_min": float(lengths.min()) if lengths.size else 0.0,
        "cycle_length_max": float(lengths.max()) if lengths.size else 0.0,
        "cycle_length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "cycle_length_std": float(lengths.std()) if lengths.size else 0.0,
    }
    for length in range(3, MAX_CYCLE_LEN + 1):
        out[f"cycle_count_len_{length}"] = float(count_by_len[length])
    return out


def _molecule_global_chemistry(
    node_types: np.ndarray, edge_types: Mapping[tuple[int, int], int]
) -> dict[str, float]:
    node_counts = np.bincount(node_types % ATOM_BINS, minlength=ATOM_BINS).astype(np.float64)
    bond_values = np.asarray(list(edge_types.values()), dtype=np.int64)
    bond_counts = (
        np.bincount(bond_values % BOND_BINS, minlength=BOND_BINS).astype(np.float64)
        if bond_values.size
        else np.zeros(BOND_BINS, dtype=np.float64)
    )
    n_total = float(node_counts.sum())
    b_total = float(bond_counts.sum())
    out: dict[str, float] = {}
    for i in range(ATOM_BINS):
        out[f"node_type_{i}_count"] = float(node_counts[i])
        out[f"node_type_{i}_fraction"] = float(node_counts[i] / max(n_total, 1.0))
    for i in range(BOND_BINS):
        out[f"edge_type_{i}_count"] = float(bond_counts[i])
        out[f"edge_type_{i}_fraction"] = float(bond_counts[i] / max(b_total, 1.0))
    out["total_nodes"] = n_total
    out["total_edges"] = b_total
    return out


def _patch_freqs(record: Any, train_freq: Mapping[bytes, int]) -> np.ndarray:
    return np.asarray(
        [train_freq.get(patch.typed_certificate, 0) for patch in record.patches], dtype=np.float64
    )


# --------------------------------------------------------------------------
# continuous pair interaction features
# --------------------------------------------------------------------------

def _continuous_pair_vector(record: Any, h: np.ndarray | None) -> np.ndarray:
    """mean_{(i,j) in bucket} (h_i ⊙ h_j) for each of the 5 buckets (5x48=240D)."""
    if h is None:
        return np.zeros(0, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    src, dst = record.pair_index[0], record.pair_index[1]
    buckets = record.pair_bucket
    blocks = []
    for bucket in range(D_BUCKETS):
        mask = buckets == bucket
        if not mask.any():
            blocks.append(np.zeros(h.shape[1], dtype=np.float64))
            continue
        product = h[src[mask]] * h[dst[mask]]
        blocks.append(product.mean(axis=0))
    return np.concatenate(blocks)


def _continuous_features(
    records: Sequence[Any], state_store: Sequence[np.ndarray | None]
) -> np.ndarray:
    vectors = [
        _continuous_pair_vector(record, state_store[i])
        for i, record in enumerate(records)
    ]
    width = int(vectors[0].shape[0])
    return np.stack(vectors, axis=0).reshape(len(records), width) if width else np.zeros((len(records), 0))


# --------------------------------------------------------------------------
# feature hashing
# --------------------------------------------------------------------------

def _hash_keys(indices: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(indices.shape[0], dtype=np.int64)
    for i, key in enumerate(indices):
        digest = hashlib.md5(int(key).to_bytes(8, "little", signed=False)).digest()
        out[i] = int.from_bytes(digest[:8], "little") % int(dim)
    return out


def _hashed_count_features(
    records: Sequence[Any],
    typed_vocabulary: Mapping[bytes, int],
    *,
    mode: str,
    dim: int,
    transform: str,
    collision_counter: dict[str, Any] | None = None,
) -> np.ndarray:
    """mode: 'unary' | 'pair' | 'pair_relation' | 'pair_oov_exact'.

    'pair_oov_exact' keeps the exact identity of OOV patches (via a
    deterministic hash of the certificate) instead of collapsing every OOV
    patch to token 0; it is an *oracle-upper-bound* variant that gives the
    probe more pair identity information than the baseline possesses.
    """
    rows: list[np.ndarray] = []
    for record in records:
        known = np.asarray(
            [typed_vocabulary.get(p.typed_certificate, 0) for p in record.patches],
            dtype=np.int64,
        )
        if mode == "unary":
            keys = known
        elif mode == "pair_oov_exact":
            tokens = known.astype(np.int64, copy=True)
            for index, patch in enumerate(record.patches):
                if tokens[index] == 0:
                    digest = hashlib.md5(patch.typed_certificate).digest()
                    tokens[index] = 0x10000 + (
                        int.from_bytes(digest[:8], "little") % (2**31 - 0x10000)
                    )
            src, dst = record.pair_index[0], record.pair_index[1]
            a, b = tokens[src], tokens[dst]
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            keys = lo * (2**31) + hi
        else:
            src, dst = record.pair_index[0], record.pair_index[1]
            a, b = known[src], known[dst]
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            if mode == "pair":
                keys = lo * TOKEN_BASE + hi
            else:  # pair_relation
                keys = lo * (BUCKET_BASE * TOKEN_BASE) + record.pair_bucket.astype(np.int64) * TOKEN_BASE + hi
        unique, inverse = np.unique(keys, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        buckets = _hash_keys(unique, dim)
        vector = np.zeros(dim, dtype=np.float64)
        np.add.at(vector, buckets, counts)
        if transform == "presence":
            vector = (vector > 0).astype(np.float64)
        elif transform == "log1p":
            vector = np.log1p(vector)
        else:
            raise ValueError(f"unknown transform {transform!r}")
        rows.append(vector)
        if collision_counter is not None:
            collision_counter["distinct_keys"] += int(len(unique))
            per_bucket = np.bincount(buckets, minlength=dim)
            collision_counter["distinct_key_pairs_colliding"] += int(
                sum(c * (c - 1) // 2 for c in per_bucket if c > 1)
            )
            collision_counter["molecules"] += 1
    return np.stack(rows, axis=0)


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

def _collision_rate(counter: dict[str, Any]) -> dict[str, Any]:
    distinct = int(counter["distinct_keys"])
    colliding = int(counter["distinct_key_pairs_colliding"])
    return {
        "distinct_keys": distinct,
        "colliding_pairs_estimate": colliding,
        "approximate_collision_rate": colliding / max(distinct * (distinct - 1) // 2, 1),
        "molecules_seen": int(counter["molecules"]),
    }


def _fit_evaluate_probe(
    name: str,
    X_train: np.ndarray,
    r_train: np.ndarray,
    X_valid: np.ndarray,
    r_valid: np.ndarray,
    y_valid: np.ndarray,
    y_hat_valid: np.ndarray,
    alpha_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    alphas = alpha_grid if alpha_grid is not None else np.logspace(-4, 6, 41)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_valid = np.asarray(X_valid, dtype=np.float64)
    r_train = np.asarray(r_train, dtype=np.float64)
    r_valid = np.asarray(r_valid, dtype=np.float64)

    if X_train.shape[1] == 0:
        # null control: constant zero predictor; the corrected MAE must
        # equal the baseline MAE exactly (sanity check of the pipeline).
        r_hat = np.zeros(len(r_valid))
        chosen_alpha = float("nan")
        cv_mae = float(mean_absolute_error(r_train, np.zeros_like(r_train)))
    else:
        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xva = scaler.transform(X_valid)
        chosen_alpha, cv_mae = _select_alpha_svd(Xtr, r_train, alphas, folds=5)
        ridge = Ridge(alpha=chosen_alpha)
        ridge.fit(Xtr, r_train)
        r_hat = ridge.predict(Xva)
    baseline_mae = float(mean_absolute_error(y_valid, y_hat_valid))
    # corrected prediction = baseline prediction + probe-estimated residual
    corrected_mae = float(mean_absolute_error(y_valid, y_hat_valid + r_hat))
    return {
        "probe": name,
        "feature_dim": int(X_train.shape[1]),
        "chosen_alpha": chosen_alpha,
        "train_internal_cv_mae": cv_mae,
        "valid_residual_r2": float(r2_score(r_valid, r_hat)),
        "valid_pearson": float(np.corrcoef(r_valid, r_hat)[0, 1]) if np.std(r_hat) > 0 else float("nan"),
        "valid_spearman": float(scipy_stats.spearmanr(r_valid, r_hat).statistic) if np.std(r_hat) > 0 else float("nan"),
        "baseline_valid_mae": baseline_mae,
        "corrected_valid_mae": corrected_mae,
        "delta_mae": float(baseline_mae - corrected_mae),
        "r_hat_mean": float(np.mean(r_hat)),
        "r_hat_std": float(np.std(r_hat)),
        "r_hat_max_abs": float(np.max(np.abs(r_hat))),
    }


def _select_alpha_svd(
    X: np.ndarray, r: np.ndarray, alphas: np.ndarray, folds: int = 5
) -> tuple[float, float]:
    """Train-internal K-fold MAE alpha selection (validation untouched).

    Uses the SVD form of ridge: for a candidate alpha, the prediction on a
    held-out fold is ``X_va V diag(s/(s^2+alpha)) (U^T r)``.  The per-fold
    SVD is computed once; all alphas are evaluated on top of it.
    """
    kf = KFold(n_splits=int(folds), shuffle=False)
    per_alpha = np.zeros(len(alphas))
    for train_idx, valid_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[valid_idx]
        r_tr, r_va = r[train_idx], r[valid_idx]
        U, S, Vt = np.linalg.svd(X_tr, full_matrices=False)
        Utr = U.T @ r_tr
        for i, alpha in enumerate(alphas):
            coef = Vt.T @ (S / (S * S + float(alpha)) * Utr)
            prediction = X_va @ coef
            per_alpha[i] += mean_absolute_error(r_va, prediction)
    per_alpha /= float(folds)
    best = int(np.argmin(per_alpha))
    return float(alphas[best]), float(per_alpha[best])


def stage_probes() -> dict[str, Any]:
    train_records = _read_records("train")
    valid_records = _read_records("valid")
    train_sidecars = _graph_sidecars("train")
    valid_sidecars = _graph_sidecars("valid")
    typed_vocabulary = zpp._fit_vocabulary(train_records, "typed_certificate", 8192, 1)
    train_freq = _train_counter(train_records)
    print(f"[probes] train vocab {len(typed_vocabulary)+1} OOV-inclusive; unique train patches {len(train_freq)}", flush=True)

    valid_npz = _load_npz(AUDIT_ROOT / "baseline_valid_predictions.npz")
    oof_npz = _load_npz(AUDIT_ROOT / "oof_train_predictions.npz")
    y_valid, y_hat_valid = valid_npz["y"], valid_npz["prediction"]
    r_valid = valid_npz["signed_residual"]
    r_train = oof_npz["signed_residual"]
    y_train = oof_npz["y"]
    assert len(r_train) == len(train_records) and len(r_valid) == len(valid_records)
    with gzip.open(AUDIT_ROOT / "oof_train_states.pkl.gz", "rb") as handle:
        oof_states = pickle.load(handle)["states"]
    with gzip.open(AUDIT_ROOT / "baseline_valid_states.pkl.gz", "rb") as handle:
        valid_states = pickle.load(handle)["states"]

    # ---- validation per-molecule audit table ----------------------------
    print("[probes] building validation audit table ...", flush=True)
    rows: list[dict[str, Any]] = []
    for idx, (record, (graph, node_types, edge_types)) in enumerate(
        zip(valid_records, valid_sidecars, strict=True)
    ):
        token_ids = np.asarray(
            [typed_vocabulary.get(p.typed_certificate, 0) for p in record.patches], dtype=np.int64
        )
        freqs = _patch_freqs(record, train_freq)
        n_patches = float(len(freqs))
        row: dict[str, Any] = {
            "molecule_id": f"valid:{idx:04d}",
            "target": float(record.y),
            "prediction": float(y_hat_valid[idx]),
            "signed_residual": float(r_valid[idx]),
            "absolute_error": float(np.abs(r_valid[idx])),
            "num_nodes": float(graph.n),
            "num_patches": n_patches,
            "num_pairs": float(record.pair_index.shape[1]),
            "num_unique_patch_tokens": float(len(set(token_ids.tolist()))),
            "unique_patch_ratio": float(len(set(token_ids.tolist())) / max(n_patches, 1.0)),
            "num_oov_patch_tokens": float(np.sum(token_ids == 0)),
            "oov_patch_ratio": float(np.sum(token_ids == 0) / max(n_patches, 1.0)),
            "min_train_frequency": float(freqs.min()) if freqs.size else 0.0,
            "mean_train_frequency": float(freqs.mean()) if freqs.size else 0.0,
            "median_train_frequency": float(np.median(freqs)) if freqs.size else 0.0,
            "mean_log1p_train_frequency": float(np.log1p(freqs).mean()) if freqs.size else 0.0,
            "mean_inverse_of_1p_frequency": float((1.0 / (1.0 + freqs)).mean()) if freqs.size else 0.0,
        }
        for threshold in (1, 2, 5, 10):
            count = float(np.sum(freqs <= threshold))
            row[f"num_rare_le{threshold}"] = count
            row[f"rare_le{threshold}_ratio"] = count / max(n_patches, 1.0)
        row.update(_molecule_size_stats(graph))
        row.update(_molecule_cycle_stats(graph, node_types, edge_types))
        row.update(_molecule_global_chemistry(node_types, edge_types))
        rows.append(row)
    audit_table = pd.DataFrame(rows)
    audit_table.to_csv(AUDIT_ROOT / "compact_v2_validation_information_gap_audit.csv", index=False)
    print(f"[probes] audit table rows={len(audit_table)} cols={len(audit_table.columns)}", flush=True)

    # ---- correlations (validation) --------------------------------------
    correlation_keys = [
        "oov_patch_ratio", "rare_le1_ratio", "rare_le2_ratio", "rare_le5_ratio",
        "rare_le10_ratio", "min_train_frequency", "mean_train_frequency",
        "mean_log1p_train_frequency", "unique_patch_ratio", "num_unique_patch_tokens",
        "num_patches", "num_pairs", "num_nodes", "num_edges", "diameter",
        "mean_shortest_path_distance", "max_shortest_path_distance", "average_degree",
        "max_degree", "num_branching_nodes", "branching_node_ratio", "cycle_rank",
        "fraction_nodes_in_cycles", "num_simple_cycles_bounded", "cycle_length_mean",
        "num_nodes_in_cycles",
    ]
    errors = audit_table["absolute_error"].to_numpy()
    residuals = audit_table["signed_residual"].to_numpy()
    corr_rows = []
    for key in correlation_keys:
        values = audit_table[key].to_numpy()
        corr_rows.append(
            {
                "statistic": key,
                "pearson_abs_error": float(np.corrcoef(values, errors)[0, 1]),
                "pearson_signed_residual": float(np.corrcoef(values, residuals)[0, 1]),
                "spearman_abs_error": float(scipy_stats.spearmanr(values, errors).statistic),
                "spearman_signed_residual": float(scipy_stats.spearmanr(values, residuals).statistic),
            }
        )
    pd.DataFrame(corr_rows).to_csv(AUDIT_ROOT / "probe_A_B_correlations.csv", index=False)

    # ---- binned MAE ------------------------------------------------------
    def quantile_bins(series: pd.Series, k: int = 5) -> pd.Series:
        try:
            return pd.qcut(series, k, duplicates="drop")
        except ValueError:
            return pd.qcut(series.rank(method="first"), k)

    def bin_table(column: str, fixed_bins: list[float] | None = None) -> pd.DataFrame:
        if fixed_bins is not None:
            binned = pd.cut(audit_table[column], fixed_bins, include_lowest=True)
            counts = binned.value_counts()
            if counts.min() >= 20:
                pass
            else:
                binned = quantile_bins(audit_table[column])
        else:
            binned = quantile_bins(audit_table[column])
        grouped = audit_table.groupby(binned, observed=True)
        out = grouped.agg(
            n_molecules=("signed_residual", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            signed_residual_mean=("signed_residual", "mean"),
        ).reset_index()
        out = out.rename(columns={out.columns[0]: "bin"})
        out["bin"] = out["bin"].astype(str)
        return out[["bin", "n_molecules", "mean_absolute_error", "median_absolute_error", "signed_residual_mean"]]

    bin_tables: dict[str, pd.DataFrame] = {
        "oov_ratio_fixed": bin_table("oov_patch_ratio", [0.0, 1e-9, 0.05, 0.10, 1.0 + 1e-9]),
        "rare_le5_ratio": bin_table("rare_le5_ratio"),
        "min_frequency": bin_table("min_train_frequency"),
        "mean_log1p_frequency": bin_table("mean_log1p_train_frequency"),
        "diameter": bin_table("diameter"),
        "num_nodes": bin_table("num_nodes"),
        "mean_shortest_path_distance": bin_table("mean_shortest_path_distance"),
        "cycle_rank": bin_table("cycle_rank"),
    }
    for name, table in bin_tables.items():
        table.to_csv(AUDIT_ROOT / f"bin_mae_{name}.csv", index=False)

    # ---- probe feature matrices ------------------------------------------
    print("[probes] building OOF train feature matrices ...", flush=True)

    def molecule_feature_matrix(
        records: Sequence[Any],
        sidecars: Sequence[tuple[Any, np.ndarray, dict]],
        state_store: Sequence[np.ndarray | None] | None,
    ) -> dict[str, np.ndarray]:
        out: dict[str, list[np.ndarray]] = {
            "rarity": [], "size": [], "cycle": [], "global_counts": [],
            "global_all_model": [], "continuous_pair": [],
        }
        for i, (record, (graph, node_types, edge_types)) in enumerate(
            zip(records, sidecars, strict=True)
        ):
            token_ids = np.asarray(
                [typed_vocabulary.get(p.typed_certificate, 0) for p in record.patches],
                dtype=np.int64,
            )
            freqs = _patch_freqs(record, train_freq)
            feat = _molecule_feature_vectors(record, graph, node_types, edge_types, token_ids, freqs)
            for key, value in feat.items():
                out[key].append(value)
            if state_store is not None:
                out["continuous_pair"].append(
                    _continuous_pair_vector(record, state_store[i])
                )
        return {key: np.stack(values, axis=0) for key, values in out.items() if values}

    train_matrices = molecule_feature_matrix(train_records, train_sidecars, oof_states)
    valid_matrices = molecule_feature_matrix(valid_records, valid_sidecars, valid_states)
    print(
        f"[probes] feature matrix dims train: "
        + ", ".join(f"{k}={v.shape[1]}D" for k, v in train_matrices.items()),
        flush=True,
    )

    # hashed exact-pair features
    collision_pair: dict[str, Any] = {"distinct_keys": 0, "distinct_key_pairs_colliding": 0, "molecules": 0}
    collision_rel: dict[str, Any] = {"distinct_keys": 0, "distinct_key_pairs_colliding": 0, "molecules": 0}
    collision_unary: dict[str, Any] = {"distinct_keys": 0, "distinct_key_pairs_colliding": 0, "molecules": 0}
    collision_oov: dict[str, Any] = {"distinct_keys": 0, "distinct_key_pairs_colliding": 0, "molecules": 0}
    print("[probes] hashing exact patch-pair counts (train OOF + valid) ...", flush=True)
    pair_train_presence = _hashed_count_features(
        train_records, typed_vocabulary, mode="pair", dim=HASH_DIM, transform="presence", collision_counter=collision_pair
    )
    pair_valid_presence = _hashed_count_features(
        valid_records, typed_vocabulary, mode="pair", dim=HASH_DIM, transform="presence", collision_counter=collision_pair
    )
    pair_train_log = _hashed_count_features(
        train_records, typed_vocabulary, mode="pair", dim=HASH_DIM, transform="log1p", collision_counter=None
    )
    pair_valid_log = _hashed_count_features(
        valid_records, typed_vocabulary, mode="pair", dim=HASH_DIM, transform="log1p", collision_counter=None
    )
    rel_train_presence = _hashed_count_features(
        train_records, typed_vocabulary, mode="pair_relation", dim=HASH_DIM, transform="presence", collision_counter=collision_rel
    )
    rel_valid_presence = _hashed_count_features(
        valid_records, typed_vocabulary, mode="pair_relation", dim=HASH_DIM, transform="presence", collision_counter=collision_rel
    )
    rel_train_log = _hashed_count_features(
        train_records, typed_vocabulary, mode="pair_relation", dim=HASH_DIM, transform="log1p", collision_counter=None
    )
    rel_valid_log = _hashed_count_features(
        valid_records, typed_vocabulary, mode="pair_relation", dim=HASH_DIM, transform="log1p", collision_counter=None
    )
    unary_train = _hashed_count_features(
        train_records, typed_vocabulary, mode="unary", dim=HASH_DIM, transform="log1p", collision_counter=collision_unary
    )
    unary_valid = _hashed_count_features(
        valid_records, typed_vocabulary, mode="unary", dim=HASH_DIM, transform="log1p", collision_counter=None
    )
    pair_oov_train_log = _hashed_count_features(
        train_records, typed_vocabulary, mode="pair_oov_exact", dim=HASH_DIM, transform="log1p", collision_counter=collision_oov
    )
    pair_oov_valid_log = _hashed_count_features(
        valid_records, typed_vocabulary, mode="pair_oov_exact", dim=HASH_DIM, transform="log1p", collision_counter=None
    )
    np.savez_compressed(
        AUDIT_ROOT / "hashed_features.npz",
        pair_train_presence=pair_train_presence, pair_valid_presence=pair_valid_presence,
        pair_train_log=pair_train_log, pair_valid_log=pair_valid_log,
        rel_train_presence=rel_train_presence, rel_valid_presence=rel_valid_presence,
        rel_train_log=rel_train_log, rel_valid_log=rel_valid_log,
        unary_train=unary_train, unary_valid=unary_valid,
        pair_oov_train_log=pair_oov_train_log, pair_oov_valid_log=pair_oov_valid_log,
    )
    collision_summary = {
        "hash_dim": HASH_DIM,
        "pair": _collision_rate(collision_pair),
        "pair_relation": _collision_rate(collision_rel),
        "unary": _collision_rate(collision_unary),
        "pair_oov_exact": _collision_rate(collision_oov),
    }
    _write_json(AUDIT_ROOT / "hash_collisions.json", collision_summary)
    print(f"[probes] collisions: {json.dumps(collision_summary, indent=1)}", flush=True)

    # ---- residual linear probes ------------------------------------------
    ALPHAS = np.logspace(-4, 6, 41)
    probe_specs: list[tuple[str, np.ndarray, np.ndarray, bool]] = [
        # (name, train-feature-key, valid-feature-key, requires_continuous)
        ("null_control", np.zeros((len(r_train), 0)), np.zeros((len(r_valid), 0)), False),
        ("rarity_stats", train_matrices["rarity"], valid_matrices["rarity"], False),
        ("size_long_range", train_matrices["size"], valid_matrices["size"], False),
        ("cycle_stats", train_matrices["cycle"], valid_matrices["cycle"], False),
        ("global_counts", train_matrices["global_counts"], valid_matrices["global_counts"], False),
        ("baseline_global_all_control", train_matrices["global_all_model"], valid_matrices["global_all_model"], False),
        ("unary_token_hash_log1p", unary_train, unary_valid, False),
        ("pair_hash_presence", pair_train_presence, pair_valid_presence, False),
        ("pair_hash_log1p", pair_train_log, pair_valid_log, False),
        ("pair_relation_hash_presence", rel_train_presence, rel_valid_presence, False),
        ("pair_relation_hash_log1p", rel_train_log, rel_valid_log, False),
        ("pair_oov_exact_hash_log1p", pair_oov_train_log, pair_oov_valid_log, False),
    ]
    if "continuous_pair" in train_matrices and train_matrices["continuous_pair"].shape[1]:
        probe_specs.append(
            ("continuous_pair_interaction", train_matrices["continuous_pair"], valid_matrices["continuous_pair"], False)
        )
        # combined pair+relation+continuous (single extra, for Q6 interpretation)
        combined_train = np.concatenate(
            [pair_train_log, rel_train_log, train_matrices["continuous_pair"]], axis=1
        )
        combined_valid = np.concatenate(
            [pair_valid_log, rel_valid_log, valid_matrices["continuous_pair"]], axis=1
        )
        probe_specs.append(("pair_log_rel_log_continuous_combined", combined_train, combined_valid, False))
    results = []
    for name, X_train, X_valid, _flag in probe_specs:
        print(f"[probes] fitting {name} ({X_train.shape[1]}D) ...", flush=True)
        results.append(
            _fit_evaluate_probe(name, X_train, r_train, X_valid, r_valid, y_valid, y_hat_valid, ALPHAS)
        )
    probe_table = pd.DataFrame(results)
    probe_table.to_csv(AUDIT_ROOT / "probe_results.csv", index=False)
    _write_json(AUDIT_ROOT / "probe_results.json", {"probes": results})
    print("\n===== PROBE RESULTS =====", flush=True)
    print(probe_table[["probe", "feature_dim", "valid_residual_r2", "baseline_valid_mae", "corrected_valid_mae", "delta_mae"]].to_string(index=False), flush=True)
    return {"probes": results}


# --------------------------------------------------------------------------
# probe feature vectors (shared by train and valid)
# --------------------------------------------------------------------------

def _molecule_feature_vectors(
    record: Any,
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    token_ids: np.ndarray,
    freqs: np.ndarray,
) -> dict[str, np.ndarray]:
    size = _molecule_size_stats(graph)
    cycle = _molecule_cycle_stats(graph, node_types, edge_types)
    n_patches = float(len(freqs))
    oov = float(np.sum(token_ids == 0))
    rarity = np.asarray(
        [
            oov / max(n_patches, 1.0),
            float(np.sum(freqs <= 1)) / max(n_patches, 1.0),
            float(np.sum(freqs <= 2)) / max(n_patches, 1.0),
            float(np.sum(freqs <= 5)) / max(n_patches, 1.0),
            float(np.sum(freqs <= 10)) / max(n_patches, 1.0),
            float(freqs.min()) if freqs.size else 0.0,
            float(freqs.mean()) if freqs.size else 0.0,
            float(np.median(freqs)) if freqs.size else 0.0,
            float(np.log1p(freqs).mean()) if freqs.size else 0.0,
            float(np.median(np.log1p(freqs))) if freqs.size else 0.0,
            float((1.0 / (1.0 + freqs)).mean()) if freqs.size else 0.0,
            float(len(set(token_ids.tolist()))) / max(n_patches, 1.0),
            float(len(set(token_ids.tolist()))),
            np.log1p(n_patches),
        ],
        dtype=np.float64,
    )
    node_counts = np.bincount(node_types % ATOM_BINS, minlength=ATOM_BINS).astype(np.float64)
    bond_values = np.asarray(list(edge_types.values()), dtype=np.int64)
    bond_counts = (
        np.bincount(bond_values % BOND_BINS, minlength=BOND_BINS).astype(np.float64)
        if bond_values.size
        else np.zeros(BOND_BINS, dtype=np.float64)
    )
    n_total = float(node_counts.sum())
    b_total = float(bond_counts.sum())
    global_counts = np.concatenate(
        [
            node_counts,
            node_counts / max(n_total, 1.0),
            bond_counts,
            bond_counts / max(b_total, 1.0),
            np.asarray([n_total, b_total], dtype=np.float64),
        ]
    )
    size_vec = np.asarray(
        [
            size["num_nodes"], size["num_edges"], size["diameter"],
            size["mean_shortest_path_distance"], size["max_shortest_path_distance"],
            size["average_degree"], size["max_degree"], size["num_branching_nodes"],
            size["branching_node_ratio"], size["cycle_rank"], size["frac_dist_gt_2"],
            size["frac_dist_gt_3"], size["frac_dist_gt_4"], size["frac_dist_gt_5"],
            size["eccentricity_mean"], size["density"], size["num_components"],
            size["edges_per_node"], np.log1p(size["num_nodes"]), np.log1p(size["num_edges"]),
        ],
        dtype=np.float64,
    )
    cycle_vec = np.asarray(
        [
            cycle["num_nodes_in_cycles"], cycle["fraction_nodes_in_cycles"],
            size["cycle_rank"], cycle["num_simple_cycles_bounded"],
            cycle["cycle_length_min"], cycle["cycle_length_max"],
            cycle["cycle_length_mean"], cycle["cycle_length_std"],
        ]
        + [cycle[f"cycle_count_len_{l}"] for l in range(3, MAX_CYCLE_LEN + 1)],
        dtype=np.float64,
    )
    return {
        "rarity": rarity,
        "size": size_vec,
        "cycle": cycle_vec,
        "global_counts": global_counts,
        "global_all_model": np.asarray(record.global_context, dtype=np.float64),
    }


# --------------------------------------------------------------------------
# stage: split audit (loads test; descriptive only)
# --------------------------------------------------------------------------

def stage_split() -> dict[str, Any]:
    train_records = _read_records("train")
    valid_records = _read_records("valid")
    typed_vocabulary = zpp._fit_vocabulary(train_records, "typed_certificate", 8192, 1)
    train_freq = _train_counter(train_records)
    splits = {"train": train_records, "valid": valid_records, "test": None}
    print("[split] extracting test records (descriptive only; no probe retraining)", flush=True)
    certificate_cache: dict[bytes, bytes] = {}
    test_dataset = _load_zinc(ZINC_ROOT, "test")
    test_records, metadata = zpp._extract_split(
        test_dataset, "test", certificate_cache,
        patch_radius=PATCH_RADIUS, context_radius=0,
        structural_mode="none", max_cycle_len=MAX_CYCLE_LEN,
    )
    splits["test"] = test_records
    print(f"[split] test {len(test_records)} graphs ({metadata['seconds']:.1f}s)", flush=True)

    summary_rows: list[dict[str, Any]] = []
    per_split: dict[str, pd.DataFrame] = {}
    for split_name, records in splits.items():
        sidecars = _graph_sidecars(split_name)
        rows = []
        for idx, (record, (graph, node_types, edge_types)) in enumerate(
            zip(records, sidecars, strict=True)
        ):
            token_ids = np.asarray(
                [typed_vocabulary.get(p.typed_certificate, 0) for p in record.patches],
                dtype=np.int64,
            )
            freqs = _patch_freqs(record, train_freq)
            size = _molecule_size_stats(graph)
            stats = {
                "split": split_name,
                "target": float(record.y),
                "num_nodes": size["num_nodes"],
                "num_edges": size["num_edges"],
                "diameter": size["diameter"],
                "mean_shortest_path_distance": size["mean_shortest_path_distance"],
                "cycle_rank": size["cycle_rank"],
                "average_degree": size["average_degree"],
                "branching_node_ratio": size["branching_node_ratio"],
                "oov_patch_ratio": float(np.sum(token_ids == 0) / max(len(token_ids), 1)),
                "num_unique_tokens": float(len(set(token_ids.tolist()))),
                "rare_le1_ratio": float(np.sum(freqs <= 1) / max(len(freqs), 1)),
                "rare_le5_ratio": float(np.sum(freqs <= 5) / max(len(freqs), 1)),
                "rare_le10_ratio": float(np.sum(freqs <= 10) / max(len(freqs), 1)),
                "mean_log1p_freq": float(np.log1p(freqs).mean()) if freqs.size else 0.0,
                "num_pairs": float(record.pair_index.shape[1]),
            }
            node_counts = np.bincount(node_types % ATOM_BINS, minlength=ATOM_BINS).astype(np.float64)
            for i in range(ATOM_BINS):
                stats[f"node_type_frac_{i}"] = float(node_counts[i] / max(node_counts.sum(), 1.0))
            bond_values = np.asarray(list(edge_types.values()), dtype=np.int64)
            bond_counts = (
                np.bincount(bond_values % BOND_BINS, minlength=BOND_BINS).astype(np.float64)
                if bond_values.size
                else np.zeros(BOND_BINS, dtype=np.float64)
            )
            for i in range(BOND_BINS):
                stats[f"edge_type_frac_{i}"] = float(bond_counts[i] / max(bond_counts.sum(), 1.0))
            pair_dists, _ecc = _all_pair_distances(graph)
            hist = np.zeros(9, dtype=np.float64)
            if pair_dists.size:
                hist = np.bincount(np.clip(pair_dists.astype(np.int64), 0, 8), minlength=9).astype(np.float64)
                hist = hist / hist.sum()
            for d in range(9):
                stats[f"pair_dist_frac_{d}"] = float(hist[d])
            rows.append(stats)
        df = pd.DataFrame(rows)
        per_split[split_name] = df
        # summary
        numeric = df.drop(columns=["split"])
        summary_row: dict[str, Any] = {"split": split_name}
        for col in ["num_nodes", "num_edges", "diameter", "mean_shortest_path_distance",
                    "cycle_rank", "oov_patch_ratio", "num_unique_tokens",
                    "rare_le1_ratio", "rare_le5_ratio", "rare_le10_ratio",
                    "mean_log1p_freq", "num_pairs", "average_degree", "branching_node_ratio"]:
            summary_row[f"{col}_mean"] = float(numeric[col].mean())
            summary_row[f"{col}_std"] = float(numeric[col].std())
            summary_row[f"{col}_q25"] = float(numeric[col].quantile(0.25))
            summary_row[f"{col}_median"] = float(numeric[col].median())
            summary_row[f"{col}_q75"] = float(numeric[col].quantile(0.75))
        summary_rows.append(summary_row)
    summary_table = pd.DataFrame(summary_rows).set_index("split")
    summary_table.to_csv(AUDIT_ROOT / "split_audit_summary.csv")
    per_split["test"].to_csv(AUDIT_ROOT / "split_audit_test_rows.csv", index=False)

    # descriptive target distribution (POST-HOC ONLY)
    target_rows = []
    for split_name, df in per_split.items():
        target = df["target"].to_numpy()
        target_rows.append(
            {
                "split": split_name,
                "n": int(len(target)),
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "q25": float(np.quantile(target, 0.25)),
                "median": float(np.median(target)),
                "q75": float(np.quantile(target, 0.75)),
                "max": float(target.max()),
            }
        )
    target_table = pd.DataFrame(target_rows)
    target_table.to_csv(AUDIT_ROOT / "split_audit_target_descriptive.csv", index=False)
    result = {
        "summary": summary_table.to_dict(orient="index"),
        "target_descriptive_post_hoc_only": target_table.to_dict(orient="records"),
        "note": "Test labels used ONLY for post-hoc descriptive target statistics; no probe/model fitted on test.",
    }
    _write_json(AUDIT_ROOT / "stage_split.json", result)
    print(summary_table.to_string(), flush=True)
    print(target_table.to_string(), flush=True)
    return result


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

def stage_figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = AUDIT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(AUDIT_ROOT / "compact_v2_validation_information_gap_audit.csv")
    errors = table["absolute_error"].to_numpy()

    def scatter_with_bins(ax, x, y, xlabel, title, bins=8):
        ax.scatter(x, y, s=6, alpha=0.35, color="tab:blue", label="molecules")
        quant = pd.qcut(table[x.name] if hasattr(x, "name") else x, bins, duplicates="drop")
        binned = table.groupby(quant, observed=True)["absolute_error"].mean()
        centers = [interval.mid for interval in binned.index.categories]
        ax.plot(centers, binned.to_numpy(), color="tab:red", lw=2, label="binned mean AE")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("validation |residual| (absolute error)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    scatter_with_bins(axes[0, 0], table["oov_patch_ratio"], errors, "OOV patch ratio (train vocab)", "Abs error vs OOV patch ratio")
    scatter_with_bins(axes[0, 1], table["diameter"], errors, "graph diameter", "Abs error vs graph diameter")
    scatter_with_bins(axes[1, 0], table["num_nodes"], errors, "node count", "Abs error vs graph size")
    scatter_with_bins(axes[1, 1], table["mean_log1p_train_frequency"], errors, "mean log1p train patch frequency", "Abs error vs patch rarity (log freq)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_error_vs_rarity_size_diameter.png", dpi=120)
    plt.close(fig)

    # error by rarity quantile (binned MAE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, column, title in [
        (axes[0], "min_train_frequency", "min train patch freq"),
        (axes[1], "mean_log1p_train_frequency", "mean log1p train freq"),
        (axes[2], "rare_le5_ratio", "rare (freq<=5) ratio"),
    ]:
        binned_table = pd.read_csv(AUDIT_ROOT / f"bin_mae_{'min_frequency' if column=='min_train_frequency' else 'mean_log1p_frequency' if column=='mean_log1p_train_frequency' else 'rare_le5_ratio'}.csv")
        ax.bar(range(len(binned_table)), binned_table["mean_absolute_error"])
        ax.set_xticks(range(len(binned_table)))
        ax.set_xticklabels([b[:28] for b in binned_table["bin"]], rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("mean abs error")
        ax.set_xlabel("bin")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_error_by_rarity_bins.png", dpi=120)
    plt.close(fig)

    # residual vs best probe prediction
    probe_results = pd.read_csv(AUDIT_ROOT / "probe_results.csv")
    if len(probe_results):
        best = probe_results.loc[probe_results["corrected_valid_mae"].idxmin()]
        print(f"[figures] best probe: {best['probe']} delta {best['delta_mae']:.5f}", flush=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(table["signed_residual"], errors, s=6, alpha=0.3)
    ax.plot([-0.5, 0.5], [0, 0], color="black", lw=0.5)
    ax.set_xlabel("signed residual (y - y_hat)")
    ax.set_ylabel("absolute error")
    ax.set_title("validation residual structure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_residual_structure.png", dpi=120)
    plt.close(fig)
    # binned MAE across size/distance/cycle (quantile-binned, from probes stage)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, fname, title in [
        (axes[0], "bin_mae_num_nodes.csv", "num_nodes"),
        (axes[1], "bin_mae_diameter.csv", "diameter"),
        (axes[2], "bin_mae_mean_shortest_path_distance.csv", "mean SP distance"),
        (axes[3], "bin_mae_cycle_rank.csv", "cycle rank"),
    ]:
        binned_table = pd.read_csv(AUDIT_ROOT / fname)
        ax.bar(range(len(binned_table)), binned_table["mean_absolute_error"])
        ax.set_xticks(range(len(binned_table)))
        ax.set_xticklabels([b[:24] for b in binned_table["bin"]], rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("mean abs error")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_error_by_size_pairdist_cycle_bins.png", dpi=120)
    plt.close(fig)

    # probe comparison: delta MAE vs feature dim
    fig, ax = plt.subplots(figsize=(9, 5))
    probe_results = pd.read_csv(AUDIT_ROOT / "probe_results.csv")
    order = probe_results.sort_values("delta_mae", ascending=False)
    colors = ["tab:red" if d >= 0.008 else "tab:orange" if d >= 0.003 else "tab:gray" for d in order["delta_mae"]]
    ax.barh(range(len(order)), order["delta_mae"], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["probe"], fontsize=7)
    ax.axvline(0.003, color="tab:orange", ls="--", lw=1, label="mild 0.003")
    ax.axvline(0.008, color="tab:red", ls="--", lw=1, label="strong 0.008")
    ax.set_xlabel("delta MAE (baseline - corrected)")
    ax.set_title("Residual probes: validation MAE improvement")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_probe_delta_mae_comparison.png", dpi=120)
    plt.close(fig)
    return {"figures": sorted(p.name for p in fig_dir.iterdir())}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

STAGE_FUNCS = {
    "features": stage_features,
    "baseline": stage_baseline,
    "oof": stage_oof,
    "probes": stage_probes,
    "split": stage_split,
    "figures": stage_figures,
}
STAGES = tuple(STAGE_FUNCS.keys())

# Marker file per stage: if present, the stage is considered complete and is
# skipped unless --force is given.  Stage runs are deterministic: each stage
# produces its artifacts once, and later stages only read them.
STAGE_MARKERS = {
    "features": "stage_features.json",
    "baseline": "stage_baseline.json",
    "oof": "stage_oof.json",
    "probes": "probe_results.json",
    "split": "split_audit_summary.csv",
    "figures": "fig1_error_vs_rarity_size_diameter.png",
}


def _stage_marker(stage: str) -> Path:
    return AUDIT_ROOT / STAGE_MARKERS[stage]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=STAGES, help="audit stage to run"
    )
    parser.add_argument("--folds", type=int, default=OOF_FOLDS, help="OOF folds (default 5)")
    parser.add_argument("--force", action="store_true", help="re-run even if artifacts exist")
    args = parser.parse_args(argv)
    marker = _stage_marker(args.stage)
    if marker.exists() and not args.force:
        print(f"[skip] {args.stage} artifacts already present ({marker.name}); use --force to re-run", flush=True)
        return 0
    if args.stage == "oof":
        result = stage_oof(folds=args.folds)
    else:
        result = STAGE_FUNCS[args.stage]()
    print(json.dumps({"stage": args.stage, "ok": True, "summary": _shorten(result)}, ensure_ascii=False, indent=2))
    return 0


def _shorten(result: Any) -> Any:
    if isinstance(result, dict):
        return {
            key: value
            for key, value in result.items()
            if not isinstance(value, (np.ndarray,)) and len(str(value)) < 400
        }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
