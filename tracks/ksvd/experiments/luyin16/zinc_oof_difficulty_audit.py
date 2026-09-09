"""OOF difficulty / heteroscedasticity confirmation for compact-v4-hinge on ZINC.

Stage mandate (see ``notes/oof_difficulty_heteroscedasticity_audit.md``):

The Post-v4 Residual Audit (validation only) found no second-layer
*representation* signal but a stable, variance-side *difficulty* structure
(rare-patch molecules harder; low-z_SA targets harder; seed disagreement
tracks error magnitude).  This stage is a **diagnostic gate**, not model
development: it re-tests that difficulty structure on the official TRAIN
split through out-of-fold predictions produced by the *frozen* compact-v4-hinge
protocol.  The question is:

> can the model know, from inputs only, how hard a never-trained molecule is?

Everything else (direction/signed bias, residual correction) is out of scope.

Hard rules (enforced in code where possible):

* compact-v4-hinge is frozen: architecture, topology hinge channel, features,
  loss, optimizer, LR, batch size, epochs, early stopping and
  validation-selection logic are reused bit-for-bit from the frozen runner
  code path (``zinc_patch_path_pooling._train_phase`` / ``_phase_data``).
  The only new code is OOF prediction infrastructure.
* the official TEST split is never loaded anywhere in this module (only
  ``_load_zinc(root, "train")`` is ever called); test numbers stay frozen.
* the official VALIDATION split is also not loaded: the frozen benchmark
  numbers and the later v5 gate keep validation as an independent check, and
  every OOF fold selects its checkpoint on an *inner* validation carved from
  its own outer-training folds.
* deterministic outer folds (K=5, fold_seed=0) and deterministic inner
  splits (7200 inner-train / 800 inner-validation per outer fold, ~10%).
* per-fold fitted objects (typed/parent vocabulary, patch/context/topology
  standardizers, rarity frequency counters) are fit on the inner-train
  portion of that outer fold only; the held-out fold never participates in
  training, checkpoint selection, frequency statistics or scaler fitting.
* z_SA / z_logP / z_cycle / frozen group are ATTRIBUTION-ONLY target
  components; they are never candidate difficulty features.

Stages (each idempotent via stage markers; ``--force`` to rerun):

    splits      deterministic 5-fold outer split + inner 7200/800 splits
    fold        one (fold, model_seed) OOF training + held-out prediction
                (parallelisable; requires --fold/--seed)
    states      per-molecule internal-state statistics from saved fold
                checkpoints (gated: predictions must reproduce bit-exactly)
    features    per-molecule feature table (per-fold rarity with inner-train
                counters, static structure, topology raw, attribution)
    table       assemble results/oof_difficulty_audit/oof_per_molecule.csv
    analysis    Tables A/B + rarity bins + partial + Table E + ranking
    predictor   nested cross-fitted StandardScaler+Ridge difficulty predictor
                (Tables C/D)
    figures     figures 1-6
    decision    Q1-Q14 answers + GO/NO-GO decision record
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
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_information_gap_audit import (
    _molecule_cycle_stats,
    _molecule_global_chemistry,
    _molecule_size_stats,
    _train_counter,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
    COMMUNITY_LOGP_MEAN,
    FIT_MU_SA,
    FIT_SIGMA_LOGP,
    FIT_SIGMA_SA,
    _build_v4_model,
)

REPO_ROOT = zpp.REPO_ROOT
OOF_ROOT = REPO_ROOT / "tracks/ksvd/results/oof_difficulty_audit"
CACHE_DIR = OOF_ROOT / "cache"
FOLD_DIR = CACHE_DIR / "folds"
FIG_DIR = OOF_ROOT / "figures"
LOG_DIR = OOF_ROOT / "logs"

CONFIG_PATH = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
)
LABEL_CSV = (
    REPO_ROOT / "tracks/ksvd/results/zinc_long_cycle_audit/label_effective_cycle.csv"
)
# Read-only caches built by the earlier audits (same machine, same extraction
# code paths).  Rebuilt lazily by this module if missing.
V4_RECORDS_CACHE = (
    REPO_ROOT / "tracks/ksvd/results/post_v4_residual_audit/cache/v4_records_train.pkl.gz"
)
SIDECARS_CACHE = (
    REPO_ROOT / "tracks/ksvd/results/information_gap_audit/cache/graph_sidecars_train.pkl.gz"
)
ZINC_ROOT = REPO_ROOT / "data/ZINC"

K_FOLDS = 5
FOLD_SEED = 0
INNER_VALID_SIZE = 800  # 10% of the 8000-molecule outer-training pool
MAX_TYPED_TOKENS = 8192
MIN_TYPED_FREQUENCY = 1
MAX_PARENT_TOKENS = 2048
MIN_PARENT_FREQUENCY = 1
SHELL_WIDTH = zpp.SHELL_WIDTH
MODEL_SEEDS = (0, 1)  # wave 1 = seed 0 only; disagreement needs both


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}



def _rank_quintile_labels(values: np.ndarray, q: int = 5) -> np.ndarray:
    """Quintile labels 0..q-1 on dense ranks (ties broken by position), so
    every bin holds ~n/q molecules even for mass-at-zero distributions."""
    ranks = pd.Series(values).rank(method="first").to_numpy()
    labels = pd.qcut(ranks, q, labels=False)
    if hasattr(labels, "to_numpy"):
        labels = labels.to_numpy()
    return np.asarray(labels, dtype=np.int64)


def _qcut_categorical(values: np.ndarray, q: int = 5):
    """pd.qcut on raw values with ties handled by rank fallback."""
    try:
        return pd.qcut(values, q, duplicates="drop")
    except ValueError:
        ranks = pd.Series(values).rank(method="first")
        return pd.qcut(ranks, q)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _stage_marker(stage: str) -> Path:
    return OOF_ROOT / f"stage_{stage}.json"


def _is_done(stage: str) -> bool:
    return _stage_marker(stage).exists()


def _mark_done(stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["stage"] = stage
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(_stage_marker(stage), result)
    return result


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(scipy_stats.spearmanr(x, y).statistic)


def _mean_finite(values: Sequence[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _std_finite(values: Sequence[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")


def _sign_consistency(values: Sequence[float], k: int = K_FOLDS) -> str:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return "insufficient"
    positive = sum(1 for v in finite if v > 0)
    negative = sum(1 for v in finite if v < 0)
    if positive == len(finite) or negative == len(finite):
        return f"{len(finite)}/{k} same sign"
    return f"{max(positive, negative)}/{k} same sign (flips)"


def _fold_marker(fold: int, seed: int) -> Path:
    return FOLD_DIR / f"fold{fold}_seed{seed}_meta.json"


# ---------------------------------------------------------------------------
# data loading (train split only; val/test never loaded)
# ---------------------------------------------------------------------------

def _load_train_records(force: bool = False) -> list[Any]:
    """GraphRecords (hinge topology) for the official train split, cached.

    Reuses the post-v4 residual audit cache (identical extraction code path:
    ``zpp._extract_split`` with ``topology_mode="hinge"`` over the official
    train dataset).  Rebuild if missing / forced.
    """
    cache_path = CACHE_DIR / "v4_records_train.pkl.gz"
    if not force and cache_path.exists():
        with gzip.open(cache_path, "rb") as handle:
            records = list(pickle.load(handle))
        if len(records) == 10000:
            return records
    if not force and V4_RECORDS_CACHE.exists():
        with gzip.open(V4_RECORDS_CACHE, "rb") as handle:
            records = list(pickle.load(handle))
        assert len(records) == 10000, "post-v4 train record cache has wrong size"
        with gzip.open(cache_path, "wb") as handle:
            pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return records
    dataset = _load_zinc(ZINC_ROOT, "train")
    topo_matrix = ztopo.matrices_for_split("train", dataset, "hinge")[0]
    certificate_cache: dict[bytes, bytes] = {}
    records, _metadata = zpp._extract_split(
        dataset,
        "train",
        certificate_cache,
        topology_mode="hinge",
        topology_matrix=topo_matrix,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wb") as handle:
        pickle.dump(list(records), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return list(records)


def _load_train_sidecars(force: bool = False) -> list[tuple[Any, np.ndarray, dict]]:
    """(nx-like graph, node_types, edge_types) per train molecule, cached."""
    cache_path = CACHE_DIR / "graph_sidecars_train.pkl.gz"
    if not force and cache_path.exists():
        with gzip.open(cache_path, "rb") as handle:
            sidecars = pickle.load(handle)
        if len(sidecars) == 10000:
            return sidecars
    if not force and SIDECARS_CACHE.exists():
        with gzip.open(SIDECARS_CACHE, "rb") as handle:
            sidecars = pickle.load(handle)
        assert len(sidecars) == 10000
        with gzip.open(cache_path, "wb") as handle:
            pickle.dump(sidecars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return sidecars
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph

    dataset = _load_zinc(ZINC_ROOT, "train")
    sidecars = [_data_to_graph(data) for data in dataset]
    with gzip.open(cache_path, "wb") as handle:
        pickle.dump(sidecars, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return sidecars


def _load_train_labels() -> pd.DataFrame:
    """Frozen per-molecule label components for the official train split.

    The label CSV is the frozen long-cycle audit artifact; only its official
    train rows are used, and only for (a) target bit-checks and (b)
    ATTRIBUTION-ONLY columns (z_logP / z_SA / z_cycle / frozen group).
    """
    frame = pd.read_csv(LABEL_CSV)
    frame = frame[frame["split"] == "train"].sort_values("subset_index").reset_index(drop=True)
    assert len(frame) == 10000
    assert (frame["molecule_id"] == [f"train:{i:04d}" for i in range(10000)]).all()
    frame["z_logP"] = (frame["logP"] - COMMUNITY_LOGP_MEAN) / FIT_SIGMA_LOGP
    frame["z_SA"] = (frame["SA"] - FIT_MU_SA) / FIT_SIGMA_SA
    frame["z_cycle"] = frame["label_normalized_cycle_component"]
    frame["y_chem"] = frame["z_logP"] + frame["z_SA"]
    frame["label_excess"] = (-frame["label_effective_cycle_snapped"]).round().clip(lower=0)
    frame["frozen_group"] = np.select(
        [frame["label_excess"] == 0, frame["label_excess"] == 1],
        ["A", "B"],
        default="C",
    )
    return frame


def _verify_records_vs_labels(records: Sequence[Any], labels: pd.DataFrame) -> None:
    y_records = np.asarray([float(record.y) for record in records], dtype=np.float64)
    y_labels = labels["y_stored"].to_numpy(dtype=np.float64)
    if not np.allclose(y_records, y_labels, atol=1e-9, rtol=1e-9):
        bad = int(np.argmax(np.abs(y_records - y_labels)))
        raise AssertionError(
            f"train record y mismatch at {bad}: {y_records[bad]} vs {y_labels[bad]}"
        )


def _frozen_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _model_config_with_clamps(config: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of model config with hybrid full-table counts clamped to the
    fitted vocabulary.

    Identical practice to the v2 OOF audit: the frozen config fixes
    ``hybrid_full_*_tokens`` full-table boundaries (768 typed / 32 parent);
    a fold vocabulary fitted on fewer molecules can be smaller than the full
    boundary (e.g. parent vocab 29 <= 32), and ``_HybridEmbedding`` requires
    ``full_count <= vocabulary_size``.  Clamping ``full_count`` to the fitted
    vocabulary keeps the exact frozen embedding code path (all fitted rows in
    the full table, no rare module) with a data-dependent row count -- the
    same row-count dependence the frozen runner already has across selection
    (10k) vs refit (11k) vocabularies.  Any clamp is recorded per fold in the
    fold metadata so it stays auditable.
    """
    model_config = dict(config["model"])
    typed_size = int(audit["typed_vocabulary_size_with_oov"])
    parent_size = int(audit["parent_vocabulary_size_with_oov"])
    frozen_typed_full = int(model_config["hybrid_full_typed_tokens"])
    frozen_parent_full = int(model_config["hybrid_full_parent_tokens"])
    model_config["hybrid_full_typed_tokens"] = min(frozen_typed_full, typed_size)
    model_config["hybrid_full_parent_tokens"] = min(frozen_parent_full, parent_size)
    model_config["_clamp_note"] = (
        f"frozen full counts typed={frozen_typed_full} parent={frozen_parent_full}; "
        f"used typed={model_config['hybrid_full_typed_tokens']} "
        f"parent={model_config['hybrid_full_parent_tokens']} "
        f"(vocab incl OOV typed={typed_size} parent={parent_size})"
    )
    return model_config


# ---------------------------------------------------------------------------
# stage: splits (deterministic outer folds + inner splits)
# ---------------------------------------------------------------------------

def stage_splits(force: bool = False) -> dict[str, Any]:
    rng = np.random.RandomState(FOLD_SEED)
    permutation = rng.permutation(10000)
    outer_fold = np.zeros(10000, dtype=np.int64)
    for fold in range(K_FOLDS):
        outer_fold[permutation[fold * 2000: (fold + 1) * 2000]] = fold
    assignments = pd.DataFrame(
        {
            "molecule_id": [f"train:{i:04d}" for i in range(10000)],
            "subset_index": np.arange(10000, dtype=np.int64),
            "outer_fold": outer_fold,
        }
    )
    assignments.to_csv(OOF_ROOT / "fold_assignments.csv", index=False)

    inner_rows: list[dict[str, Any]] = []
    inner_summary: dict[str, Any] = {}
    for fold in range(K_FOLDS):
        outer_train = np.flatnonzero(outer_fold != fold)  # 8000 molecules
        inner_rng = np.random.RandomState(12345 + FOLD_SEED * 10000 + fold * 7)
        shuffled = inner_rng.permutation(outer_train)
        inner_train = np.sort(shuffled[: len(outer_train) - INNER_VALID_SIZE])
        inner_valid = np.sort(shuffled[len(outer_train) - INNER_VALID_SIZE:])
        assert len(inner_train) == 7200 and len(inner_valid) == 800
        inner_summary[fold] = {
            "inner_split_seed_formula": "12345 + fold_seed*10000 + fold*7",
            "n_inner_train": int(len(inner_train)),
            "n_inner_valid": int(len(inner_valid)),
            "n_holdout": int(np.sum(outer_fold == fold)),
            "sha256_inner_train": hashlib.sha256(
                inner_train.astype(np.int64).tobytes()
            ).hexdigest()[:16],
            "sha256_inner_valid": hashlib.sha256(
                inner_valid.astype(np.int64).tobytes()
            ).hexdigest()[:16],
        }
        for index in inner_train:
            inner_rows.append(
                {"molecule_id": f"train:{int(index):04d}", "outer_fold": fold,
                 "inner_role": "inner_train"}
            )
        for index in inner_valid:
            inner_rows.append(
                {"molecule_id": f"train:{int(index):04d}", "outer_fold": fold,
                 "inner_role": "inner_valid"}
            )
    pd.DataFrame(inner_rows).to_csv(OOF_ROOT / "inner_splits.csv", index=False)
    summary = {
        "k": K_FOLDS,
        "fold_seed": FOLD_SEED,
        "scheme": (
            "outer: RandomState(fold_seed).permutation(10000) split into "
            "positional 2000-molecule folds; inner: per-fold "
            "RandomState(12345 + fold_seed*10000 + fold*7) permutation of the "
            "8000 outer-training molecules, first 7200 inner-train, last 800 "
            "inner-valid"
        ),
        "counts": {int(f): int(np.sum(outer_fold == f)) for f in range(K_FOLDS)},
        "inner": inner_summary,
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("splits", summary)


def _load_assignments() -> pd.DataFrame:
    return pd.read_csv(OOF_ROOT / "fold_assignments.csv")


def _fold_slices() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per fold: (inner_train_idx, inner_valid_idx, holdout_idx)."""
    assignments = _load_assignments()
    outer = assignments["outer_fold"].to_numpy(dtype=np.int64)
    inner = pd.read_csv(OOF_ROOT / "inner_splits.csv")
    slices: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold in range(K_FOLDS):
        holdout = np.flatnonzero(outer == fold).astype(np.int64)
        sub = inner[inner["outer_fold"] == fold]
        inner_train = sub.loc[sub["inner_role"] == "inner_train", "molecule_id"]
        inner_valid = sub.loc[sub["inner_role"] == "inner_valid", "molecule_id"]
        to_idx = np.vectorize(lambda mid: int(str(mid).split(":")[1]))
        slices.append(
            (
                to_idx(inner_train.to_numpy()).astype(np.int64),
                to_idx(inner_valid.to_numpy()).astype(np.int64),
                holdout,
            )
        )
    return slices


# ---------------------------------------------------------------------------
# stage: fold (one frozen v4-hinge OOF training)
# ---------------------------------------------------------------------------

def _run_fold(fold: int, model_seed: int) -> dict[str, Any]:
    """Train frozen compact-v4-hinge on inner-train (7200), select best epoch
    on inner-validation (800), then predict the held-out fold (2000) once."""
    started = time.perf_counter()
    records = _load_train_records()
    labels = _load_train_labels()
    _verify_records_vs_labels(records, labels)
    config = _frozen_config()
    torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", 4)))
    inner_train_idx, inner_valid_idx, holdout_idx = _fold_slices()[fold]
    inner_train_records = [records[int(i)] for i in inner_train_idx]
    inner_valid_records = [records[int(i)] for i in inner_valid_idx]
    holdout_records = [records[int(i)] for i in holdout_idx]

    # One _phase_data call fits vocab/standardizers on inner-train only and
    # encodes both selection and held-out data with that single fit.
    encoded_fit, encoded_other, audit = zpp._phase_data(
        inner_train_records,
        list(inner_valid_records) + list(holdout_records),
        config=config,
    )
    encoded_valid = encoded_other[: len(inner_valid_records)]
    encoded_holdout = encoded_other[len(inner_valid_records):]

    model_config = _model_config_with_clamps(config, audit)
    config_mod = dict(config)
    config_mod["model"] = dict(model_config)
    phase = zpp._train_phase(
        encoded_fit,
        encoded_valid,
        typed_vocabulary_size=int(audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(audit["parent_vocabulary_size_with_oov"]),
        config=config_mod,
        seed=model_seed,
        select_best=True,
        epochs=int(model_config.get("epochs", 60)),
        shell_width=SHELL_WIDTH,
        context_width=0,
        direct_token_readout=False,
        phase_label=f"oof-fold{fold}-seed{model_seed}",
        structural_context_vocabulary_size=int(
            audit["structural_context_vocabulary_size"]
        ),
        topology_input_width=int(audit["topology"]["input_width"]),
        topology_hidden_dim=int(model_config.get("topology_hidden_dim", 16)),
        topology_out_dim=int(model_config.get("topology_out_dim", 8)),
        topology_shuffle_test=bool(model_config.get("topology_shuffle_test", True)),
    )
    model = phase["model"]
    model.eval()

    # --- OOF predictions on the held-out fold (frozen best-valid state) ---
    loader = DataLoader(list(encoded_holdout), batch_size=int(model_config.get("batch_size", 128)), shuffle=False)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            predictions.append(model(batch).cpu().numpy())
    oof_pred = np.concatenate(predictions).reshape(-1).astype(np.float64)
    assert len(oof_pred) == len(holdout_idx)
    targets = np.asarray([float(record.y) for record in holdout_records], dtype=np.float64)
    signed = targets - oof_pred
    state_path = FOLD_DIR / f"fold{fold}_seed{model_seed}_state.pt"
    torch.save(model.state_dict(), state_path)
    _save_npz(
        FOLD_DIR / f"fold{fold}_seed{model_seed}.npz",
        subset_index=holdout_idx,
        target=targets,
        oof_prediction=oof_pred,
        signed_residual=signed,
        absolute_error=np.abs(signed),
    )

    trace = phase.get("trace") or []
    meta = {
        "fold": int(fold),
        "model_seed": int(model_seed),
        "n_inner_train": int(len(inner_train_idx)),
        "n_inner_valid": int(len(inner_valid_idx)),
        "n_holdout": int(len(holdout_idx)),
        "selected_epoch": int(phase["selected_epoch"]),
        "epochs_run": int(phase["epochs_run"]),
        "best_inner_valid_mae": float(phase["best_mae"]),
        "parameters": int(phase["parameters"]),
        "typed_vocabulary_size_with_oov": int(audit["typed_vocabulary_size_with_oov"]),
        "parent_vocabulary_size_with_oov": int(audit["parent_vocabulary_size_with_oov"]),
        "topology_input_width": int(audit["topology"]["input_width"]),
        "topology_standardizer_fit_graphs": int(audit["topology"]["standardizer_fit_graphs"]),
        "embedding_clamp_note": str(model_config["_clamp_note"]),
        "holdout_typed_known_occurrence_fraction": float(
            audit["typed_vocabulary"]["other"]["known_occurrence_fraction"]
        ),
        "holdout_typed_known_type_fraction": float(
            audit["typed_vocabulary"]["other"]["known_type_fraction"]
        ),
        "oof_mae": float(mean_absolute_error(targets, oof_pred)),
        "mean_signed_residual": float(signed.mean()),
        "mean_abs_error": float(np.abs(signed).mean()),
        "shuffled_topology_inner_valid_mae": float(
            phase["diagnostics"].get("valid_shuffled_topology_mae")
            if phase.get("diagnostics")
            else float("nan")
        ),
        "trace": trace,
        "seconds": float(time.perf_counter() - started),
        "state_pt_sha256": _sha256(state_path),
        "npz_sha256": _sha256(FOLD_DIR / f"fold{fold}_seed{model_seed}.npz"),
    }
    _write_json(_fold_marker(fold, model_seed), meta)
    print(
        f"[fold] fold={fold} seed={model_seed} selected_epoch={meta['selected_epoch']} "
        f"best_inner_valid_mae={meta['best_inner_valid_mae']:.6f} "
        f"oof_mae={meta['oof_mae']:.6f} params={meta['parameters']} "
        f"({meta['seconds']:.0f}s)",
        flush=True,
    )
    return meta


def stage_fold(fold: int, model_seed: int, force: bool = False) -> dict[str, Any]:
    marker = _fold_marker(fold, model_seed)
    if marker.exists() and not force:
        print(f"[skip] fold {fold} seed {model_seed} already done (use --force)")
        return _read_json(marker)
    result = _run_fold(fold, model_seed)
    return result


def _available_seeds() -> list[int]:
    seeds: list[int] = []
    for seed in MODEL_SEEDS:
        if all(_fold_marker(f, seed).exists() for f in range(K_FOLDS)):
            seeds.append(seed)
    return seeds


def _load_fold_npz(fold: int, seed: int) -> dict[str, np.ndarray]:
    return _load_npz(FOLD_DIR / f"fold{fold}_seed{seed}.npz")


# ---------------------------------------------------------------------------
# stage: states (per-molecule internal-state statistics, fold checkpoints)
# ---------------------------------------------------------------------------

def _forward_capture_fold(
    model: torch.nn.Module,
    graphs: Sequence[Any],
    batch_size: int = 128,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, list[int]]]:
    """Single eval-mode forward over ``graphs`` with state capture.

    Returns (predictions in loader order, concatenated per-batch captures,
    per-graph row counts).  Captures: patch pre-update state and centre delta
    (per patch row), pair values (per pair row), graph-level global /
    topology hidden vectors, head input (unified representation) and head
    penultimate representation (one row per graph).
    """
    loader = DataLoader(list(graphs), batch_size=int(batch_size), shuffle=False)
    predictions: list[np.ndarray] = []
    stores: dict[str, list[np.ndarray]] = {
        "patch_pre_update": [],
        "center_delta": [],
        "pair_value": [],
        "global_hidden": [],
        "topology_hidden": [],
        "head_input": [],
        "head_penult": [],
    }
    patch_counts: list[int] = []
    pair_counts: list[int] = []

    def make_hook(key: str):
        def hook(module, inputs, output) -> None:
            stores[key].append(output.detach().cpu().numpy())
        return hook

    def center_hook(module, inputs, output) -> None:
        stores["patch_pre_update"].append(inputs[0].detach().cpu().numpy())
        stores["center_delta"].append(output.detach().cpu().numpy())

    handles = [
        model.center_update.register_forward_hook(center_hook),
        model.pair_encoder.register_forward_hook(make_hook("pair_value")),
        model.global_encoder.register_forward_hook(make_hook("global_hidden")),
        model.topology_encoder.register_forward_hook(make_hook("topology_hidden")),
        model.head[0].register_forward_hook(make_hook("head_input")),
        model.head[5].register_forward_hook(make_hook("head_penult")),
    ]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch.device("cpu"))
            predictions.append(model(batch).cpu().numpy())
            batch_ids = batch.batch.cpu().numpy()
            boundaries = np.flatnonzero(np.diff(batch_ids)) + 1
            offsets = np.concatenate([[0], boundaries, [len(batch_ids)]])
            for graph_id in range(int(batch.num_graphs)):
                start, end = int(offsets[graph_id]), int(offsets[graph_id + 1])
                patch_counts.append(int(end - start))
            pair_ids = batch.pair_index[0].cpu().numpy()
            pair_boundaries = np.flatnonzero(np.diff(pair_ids)) + 1
            pair_offsets = np.concatenate([[0], pair_boundaries, [len(pair_ids)]])
            for graph_id in range(int(batch.num_graphs)):
                p_start, p_end = int(pair_offsets[graph_id]), int(pair_offsets[graph_id + 1])
                pair_counts.append(int(p_end - p_start))
    for handle in handles:
        handle.remove()
    return (
        np.concatenate(predictions).astype(np.float64),
        {key: np.concatenate(values) for key, values in stores.items()},
        {"n_patches": patch_counts, "n_pairs": pair_counts},
    )


def _state_summary_row(
    patch_states: np.ndarray,
    pre_states: np.ndarray,
    delta_states: np.ndarray,
    pair_value: np.ndarray,
    global_hidden: np.ndarray,
    topology_hidden: np.ndarray,
    head_input: np.ndarray,
    head_penult: np.ndarray,
) -> dict[str, float]:
    patch_norms = np.linalg.norm(patch_states, axis=1)
    pre_norms = np.linalg.norm(pre_states, axis=1) if pre_states.size else patch_norms
    delta_norms = np.linalg.norm(delta_states, axis=1) if delta_states.size else np.zeros_like(patch_norms)
    pair_norms = np.linalg.norm(pair_value, axis=1)
    centroid = patch_states.mean(axis=0) if patch_states.size else np.zeros(patch_states.shape[1])
    return {
        "state_patch_norm_mean": float(patch_norms.mean()) if patch_norms.size else 0.0,
        "state_patch_norm_std": float(patch_norms.std()) if patch_norms.size else 0.0,
        "state_patch_absmax_mean": float(np.abs(patch_states).mean()) if patch_states.size else 0.0,
        "state_patch_centroid_norm": float(np.linalg.norm(centroid)),
        "state_patch_dim_variance": float(patch_states.var(axis=0).mean()) if patch_states.size else 0.0,
        "state_pre_norm_mean": float(pre_norms.mean()) if pre_norms.size else 0.0,
        "state_delta_norm_mean": float(delta_norms.mean()) if delta_norms.size else 0.0,
        "state_pair_norm_mean": float(pair_norms.mean()) if pair_norms.size else 0.0,
        "state_pair_norm_std": float(pair_norms.std()) if pair_norms.size else 0.0,
        "state_global_encoder_norm": float(np.linalg.norm(global_hidden)),
        "state_topology_norm": float(np.linalg.norm(topology_hidden)),
        "state_head_input_norm": float(np.linalg.norm(head_input)),
        "state_head_penult_norm": float(np.linalg.norm(head_penult)),
        "n_patches": int(patch_states.shape[0]),
        "n_pairs": int(pair_value.shape[0]),
    }


def stage_states(model_seed: int = 0, force: bool = False) -> dict[str, Any]:
    """Per-molecule internal-state statistics from saved fold checkpoints.

    For every fold, the frozen selection checkpoint is reloaded, the held-out
    fold is re-encoded with the fold's inner-train fit (deterministic), and a
    single capture forward is run.  Gate: predictions must reproduce the fold
    npz bit-exactly before any state row is emitted.
    """
    started = time.perf_counter()
    for fold in range(K_FOLDS):
        if not _fold_marker(fold, model_seed).exists():
            raise AssertionError(f"fold {fold} seed {model_seed} not finished")
    records = _load_train_records()
    labels = _load_train_labels()
    _verify_records_vs_labels(records, labels)
    config = _frozen_config()
    torch.set_num_threads(int(config.get("runtime", {}).get("torch_threads", 4)))
    slices = _fold_slices()

    out_path = CACHE_DIR / f"states_seed{model_seed}.csv"
    if out_path.exists() and not force:
        print(f"[skip] states seed {model_seed} already built")
        return _read_json(CACHE_DIR / f"states_seed{model_seed}_meta.json")

    rows: list[dict[str, Any]] = []
    gates: dict[str, Any] = {}
    for fold in range(K_FOLDS):
        meta = _read_json(_fold_marker(fold, model_seed))
        inner_train_idx, _inner_valid_idx, holdout_idx = slices[fold]
        inner_train_records = [records[int(i)] for i in inner_train_idx]
        holdout_records = [records[int(i)] for i in holdout_idx]
        config = _frozen_config()
        encoded_fit, encoded_holdout, audit = zpp._phase_data(
            inner_train_records, holdout_records, config=config
        )
        del encoded_fit
        if int(audit["typed_vocabulary_size_with_oov"]) != int(
            meta["typed_vocabulary_size_with_oov"]
        ) or int(audit["topology"]["input_width"]) != int(meta["topology_input_width"]):
            raise AssertionError(f"fold {fold}: audit mismatch vs fold run")
        config_mod = dict(config)
        config_mod["model"] = _model_config_with_clamps(config, audit)
        model = _build_v4_model(config_mod, audit)
        state = torch.load(
            FOLD_DIR / f"fold{fold}_seed{model_seed}_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        n_params = int(sum(p.numel() for p in model.parameters()))
        if n_params != int(meta["parameters"]):
            raise AssertionError(f"fold {fold}: params {n_params} != {meta['parameters']}")
        preds, captures, counts = _forward_capture_fold(model, encoded_holdout)
        npz = _load_fold_npz(fold, model_seed)
        max_diff = float(np.abs(preds - npz["oof_prediction"]).max())
        gates[fold] = {
            "max_prediction_diff": max_diff,
            "n_graphs": int(len(counts["n_patches"])),
        }
        if max_diff > 1e-9:
            raise AssertionError(
                f"fold {fold} seed {model_seed}: state forward predictions deviate "
                f"from fold npz (max diff {max_diff})"
            )
        if len(counts["n_patches"]) != len(holdout_idx):
            raise AssertionError(f"fold {fold}: graph count mismatch")
        pre_all = captures["patch_pre_update"][:, : model.patch_hidden]
        delta_all = captures["center_delta"]
        final_all = pre_all + delta_all
        pair_offsets = np.concatenate([[0], np.cumsum(counts["n_pairs"])])
        patch_offsets = np.concatenate([[0], np.cumsum(counts["n_patches"])])
        for graph_pos in range(len(holdout_idx)):
            p0, p1 = int(patch_offsets[graph_pos]), int(patch_offsets[graph_pos + 1])
            q0, q1 = int(pair_offsets[graph_pos]), int(pair_offsets[graph_pos + 1])
            row = _state_summary_row(
                final_all[p0:p1],
                pre_all[p0:p1],
                delta_all[p0:p1],
                captures["pair_value"][q0:q1],
                captures["global_hidden"][graph_pos],
                captures["topology_hidden"][graph_pos],
                captures["head_input"][graph_pos],
                captures["head_penult"][graph_pos],
            )
            row["subset_index"] = int(holdout_idx[graph_pos])
            rows.append(row)
        print(
            f"  fold {fold}: {len(holdout_idx)} state rows (gate max diff "
            f"{max_diff:.2e})",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 10000:
        raise AssertionError(f"states rows {len(frame)} != 10000")
    frame = frame.sort_values("subset_index").reset_index(drop=True)
    frame.to_csv(out_path, index=False)
    summary = {
        "seed": int(model_seed),
        "n": int(len(frame)),
        "gates": gates,
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(CACHE_DIR / f"states_seed{model_seed}_meta.json", summary)
    print(f"[states] seed {model_seed}: {len(frame)} rows in {summary['seconds']:.0f}s", flush=True)
    return summary


# ---------------------------------------------------------------------------
# stage: features (per-molecule rarity / structure / topology / attribution)
# ---------------------------------------------------------------------------

def _patch_token_ids_and_freqs(
    record: Any,
    vocab: Mapping[bytes, int],
    counter: Counter,
) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.asarray(
        [vocab.get(p.typed_certificate, 0) for p in record.patches], dtype=np.int64
    )
    freqs = np.asarray(
        [counter.get(p.typed_certificate, 0) for p in record.patches], dtype=np.float64
    )
    return token_ids, freqs


def stage_features(force: bool = False) -> dict[str, Any]:
    """Per-molecule features for every train molecule, in the fold-of-holdout
    context (per-fold rarity counters are fit on that fold's inner-train)."""
    started = time.perf_counter()
    records = _load_train_records()
    labels = _load_train_labels()
    _verify_records_vs_labels(records, labels)
    slices = _fold_slices()

    static_out = CACHE_DIR / "features_static.csv"
    static_rows: list[dict[str, Any]] = []
    if not static_out.exists() or force:
        sidecars = _load_train_sidecars()
        print("[features] computing static structure rows ...", flush=True)
        for index, (record, (graph, node_types, edge_types)) in enumerate(
            zip(records, sidecars, strict=True)
        ):
            n_patches = float(len(record.patches))
            distinct = len({p.typed_certificate for p in record.patches})
            row: dict[str, Any] = {
                "subset_index": int(index),
                "num_patches": n_patches,
                "num_pairs": float(record.pair_index.shape[1]),
                "num_distinct_certificates": float(distinct),
                "distinct_certificate_ratio": float(distinct / max(n_patches, 1.0)),
            }
            row.update(_molecule_size_stats(graph))
            row.update(_molecule_cycle_stats(graph, node_types, edge_types))
            row.update(_molecule_global_chemistry(node_types, edge_types))
            topo_raw = (
                np.asarray(record.topology_features, dtype=np.float64)
                if record.topology_features is not None
                else np.zeros(0, dtype=np.float64)
            )
            for name, value in zip(ztopo.feature_names("hinge"), topo_raw):
                row[f"topo_{name}"] = float(value)
            static_rows.append(row)
        pd.DataFrame(static_rows).to_csv(static_out, index=False)

    static = pd.read_csv(static_out)
    assert len(static) == 10000

    rarity_out = CACHE_DIR / "rarity_rows.csv"
    rarity_rows: list[dict[str, Any]] = []
    if not rarity_out.exists() or force:
        print("[features] computing per-fold rarity rows ...", flush=True)
        for fold in range(K_FOLDS):
            inner_train_idx, _inner_valid_idx, holdout_idx = slices[fold]
            inner_train_records = [records[int(i)] for i in inner_train_idx]
            holdout_records = [records[int(i)] for i in holdout_idx]
            counter = _train_counter(inner_train_records)
            vocab = zpp._fit_vocabulary(
                inner_train_records,
                "typed_certificate",
                MAX_TYPED_TOKENS,
                MIN_TYPED_FREQUENCY,
            )
            assert len(counter) == len(vocab), "cap must not bind"
            for index, record in zip(holdout_idx, holdout_records, strict=True):
                token_ids, freqs = _patch_token_ids_and_freqs(record, vocab, counter)
                n_patches = float(len(freqs))
                row: dict[str, Any] = {
                    "subset_index": int(index),
                    "outer_fold": int(fold),
                    "num_oov_patch_tokens": float(np.sum(token_ids == 0)),
                    "oov_patch_ratio": float(np.sum(token_ids == 0) / max(n_patches, 1.0)),
                    "num_unique_patch_tokens": float(len(set(token_ids.tolist()))),
                    "unique_patch_ratio": float(len(set(token_ids.tolist())) / max(n_patches, 1.0)),
                    "min_train_frequency": float(freqs.min()) if freqs.size else 0.0,
                    "mean_train_frequency": float(freqs.mean()) if freqs.size else 0.0,
                    "median_train_frequency": float(np.median(freqs)) if freqs.size else 0.0,
                    "mean_log1p_train_frequency": float(np.log1p(freqs).mean()) if freqs.size else 0.0,
                    "mean_inverse_of_1p_frequency": float(
                        (1.0 / (1.0 + freqs)).mean()
                    ) if freqs.size else 0.0,
                }
                for threshold in (1, 2, 5, 10):
                    count = float(np.sum(freqs <= threshold))
                    row[f"num_rare_le{threshold}"] = count
                    row[f"rare_le{threshold}_ratio"] = count / max(n_patches, 1.0)
                rarity_rows.append(row)
            print(f"  fold {fold}: inner-train vocab {len(vocab)} (done)", flush=True)
        rarity = pd.DataFrame(rarity_rows)
        rarity.to_csv(rarity_out, index=False)

    summary = {
        "n_static": int(len(static)),
        "n_rarity": int(pd.read_csv(rarity_out).shape[0]),
        "rarity_scope_note": (
            "per-molecule rarity is computed against the counter of the "
            "inner-train portion (7200) of the molecule's own outer fold; "
            "the held-out fold never contributes to any counter"
        ),
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("features", summary)


# ---------------------------------------------------------------------------
# stage: table (assemble oof_per_molecule.csv)
# ---------------------------------------------------------------------------

def stage_table(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    records = _load_train_records()
    labels = _load_train_labels()
    _verify_records_vs_labels(records, labels)
    assignments = _load_assignments()
    seeds = _available_seeds()
    if not seeds:
        raise AssertionError("no complete model seed available; run fold stage first")

    static = pd.read_csv(CACHE_DIR / "features_static.csv")
    rarity = pd.read_csv(CACHE_DIR / "rarity_rows.csv")
    rarity = rarity.drop(columns=["outer_fold"])  # from the molecule's own fold
    columns = {
        "molecule_id": [f"train:{i:04d}" for i in range(10000)],
        "subset_index": np.arange(10000, dtype=np.int64),
        "outer_fold": assignments["outer_fold"].to_numpy(dtype=np.int64),
        "target": np.asarray([float(r.y) for r in records], dtype=np.float64),
    }
    base = pd.DataFrame(columns)
    for seed in seeds:
        fold_frames: list[pd.DataFrame] = []
        for fold in range(K_FOLDS):
            npz = _load_fold_npz(fold, seed)
            fold_frames.append(
                pd.DataFrame(
                    {
                        "subset_index": npz["subset_index"],
                        f"oof_prediction_seed{seed}": npz["oof_prediction"],
                        f"signed_residual_seed{seed}": npz["signed_residual"],
                        f"absolute_error_seed{seed}": npz["absolute_error"],
                    }
                )
            )
        preds = pd.concat(fold_frames, ignore_index=True).sort_values("subset_index")
        assert len(preds) == 10000
        assert (preds["subset_index"].to_numpy() == np.arange(10000)).all()
        base = base.merge(preds, on="subset_index", how="left", validate="one_to_one")
    pred_cols = [f"oof_prediction_seed{s}" for s in seeds]
    abs_cols = [f"absolute_error_seed{s}" for s in seeds]
    resid_cols = [f"signed_residual_seed{s}" for s in seeds]
    base["mean_prediction"] = base[pred_cols].mean(axis=1)
    base["mean_signed_residual"] = base[resid_cols].mean(axis=1)
    base["mean_absolute_error"] = base[abs_cols].mean(axis=1)
    base["ensemble_absolute_error"] = (base["target"] - base["mean_prediction"]).abs()
    if len(seeds) >= 2:
        base["disagreement"] = base[pred_cols].std(axis=1, ddof=1) if len(seeds) > 2 else (
            base[pred_cols[0]] - base[pred_cols[1]]
        ).abs()
    table = base.merge(
        labels[
            [
                "molecule_id",
                "y_stored",
                "logP",
                "SA",
                "z_logP",
                "z_SA",
                "z_cycle",
                "y_chem",
                "label_excess",
                "frozen_group",
            ]
        ],
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )
    table = table.merge(static, on="subset_index", how="left", validate="one_to_one")
    table = table.merge(rarity, on="subset_index", how="left", validate="one_to_one")
    if (CACHE_DIR / "states_seed0.csv").exists():
        states = pd.read_csv(CACHE_DIR / "states_seed0.csv")
        state_cols = [c for c in states.columns if c != "subset_index"]
        states = states.rename(columns={c: f"{c}_seed0" for c in state_cols})
        table = table.merge(states, on="subset_index", how="left", validate="one_to_one")
    table = table.drop(columns=["y_stored"])
    out_path = OOF_ROOT / "oof_per_molecule.csv"
    table.to_csv(out_path, index=False)
    n_missing = int(table["mean_prediction"].isna().sum())
    assert n_missing == 0
    summary = {
        "n": int(len(table)),
        "seeds": [int(s) for s in seeds],
        "oof_mae_seed0": float(np.mean(table["absolute_error_seed0"].to_numpy()))
        if 0 in seeds else None,
        "ensemble_oof_mae": float(np.mean(table["ensemble_absolute_error"].to_numpy())),
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("table", summary)


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------

def _rank(values: np.ndarray) -> np.ndarray:
    return scipy_stats.rankdata(values)


def _partial_spearman(x: np.ndarray, y: np.ndarray, ctrl: np.ndarray | None) -> float:
    """Partial Spearman of x vs y given control ctrl (rank + OLS residual)."""
    if ctrl is None or ctrl.size == 0 or np.all(ctrl == ctrl[0]):
        return _spearman(x, y)
    rx = _rank(x)
    ry = _rank(y)
    rc = _rank(ctrl)
    design = np.column_stack([np.ones_like(rc), rc])
    res_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    res_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return _pearson(res_x, res_y)


def _table_a_from_table(table: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(K_FOLDS):
        sub = table[table["outer_fold"] == fold]
        row: dict[str, Any] = {"fold": fold, "n": int(len(sub))}
        for seed in seeds:
            row[f"oof_mae_seed{seed}"] = float(np.mean(sub[f"absolute_error_seed{seed}"]))
            row[f"mean_signed_residual_seed{seed}"] = float(np.mean(sub[f"signed_residual_seed{seed}"]))
            row[f"mean_abs_error_seed{seed}"] = float(np.mean(sub[f"absolute_error_seed{seed}"]))
        row["mean_abs_error_over_seeds"] = float(np.mean(sub["mean_absolute_error"]))
        row["mean_signed_residual_over_seeds"] = float(np.mean(sub["mean_signed_residual"]))
        row["median_abs_error_over_seeds"] = float(np.median(sub["mean_absolute_error"]))
        if len(seeds) >= 2:
            row["ensemble_oof_mae"] = float(np.mean(sub["ensemble_absolute_error"]))
        rows.append(row)
    table_a = pd.DataFrame(rows)
    table_a.to_csv(OOF_ROOT / "table_A_fold_results.csv", index=False)
    return table_a


def _error_target(table: pd.DataFrame) -> np.ndarray:
    """Primary per-molecule difficulty target: mean over available seeds of
    the per-seed OOF absolute errors (single seed when only seed 0 exists)."""
    return table["mean_absolute_error"].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# stage: analysis
# ---------------------------------------------------------------------------

CANDIDATE_BLOCKS: dict[str, list[str]] = {
    "rarity": [
        "oov_patch_ratio",
        "rare_le1_ratio",
        "rare_le2_ratio",
        "rare_le5_ratio",
        "rare_le10_ratio",
        "min_train_frequency",
        "mean_train_frequency",
        "median_train_frequency",
        "mean_log1p_train_frequency",
        "mean_inverse_of_1p_frequency",
        "num_unique_patch_tokens",
        "unique_patch_ratio",
    ],
    "structure": [
        "num_nodes",
        "num_edges",
        "num_patches",
        "num_pairs",
        "diameter",
        "mean_shortest_path_distance",
        "max_shortest_path_distance",
        "average_degree",
        "max_degree",
        "num_branching_nodes",
        "branching_node_ratio",
        "cycle_rank",
        "num_components",
        "density",
        "fraction_nodes_in_cycles",
        "num_simple_cycles_bounded",
        "cycle_length_mean",
        "num_nodes_in_cycles",
        "frac_dist_gt_2",
    ],
    "topology_raw": [f"topo_{name}" for name in ztopo.feature_names("hinge")],
    "states": [
        "state_patch_norm_mean_seed0",
        "state_patch_norm_std_seed0",
        "state_patch_absmax_mean_seed0",
        "state_patch_centroid_norm_seed0",
        "state_patch_dim_variance_seed0",
        "state_pre_norm_mean_seed0",
        "state_delta_norm_mean_seed0",
        "state_pair_norm_mean_seed0",
        "state_pair_norm_std_seed0",
        "state_global_encoder_norm_seed0",
        "state_topology_norm_seed0",
        "state_head_input_norm_seed0",
        "state_head_penult_norm_seed0",
    ],
    "disagreement": ["disagreement"],
}

ATTRIBUTION_ONLY = ["z_SA", "z_logP", "z_cycle", "y_chem", "target", "frozen_group", "label_excess"]


def stage_analysis(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(OOF_ROOT / "oof_per_molecule.csv")
    seeds = _available_seeds()
    err = _error_target(table)
    signed = table["mean_signed_residual"].to_numpy(dtype=np.float64)

    # Table A (per fold)
    _table_a_from_table(table, seeds)

    # --- Table B: per-feature per-fold Spearman vs abs error ---
    corr_rows: list[dict[str, Any]] = []
    feature_cols: list[str] = []
    for block in ("rarity", "structure", "topology_raw", "states", "disagreement"):
        for name in CANDIDATE_BLOCKS[block]:
            if name not in table.columns:
                continue
            feature_cols.append(name)
    for name in feature_cols:
        values = table[name].to_numpy(dtype=np.float64)
        per_fold = []
        for fold in range(K_FOLDS):
            mask = table["outer_fold"].to_numpy() == fold
            per_fold.append(_spearman(values[mask], err[mask]))
        row = {
            "feature": name,
            "block": next(b for b in CANDIDATE_BLOCKS if name in CANDIDATE_BLOCKS[b]),
            "overall_spearman": _spearman(values, err),
            "overall_pearson": _pearson(values, err),
            **{f"fold{fold}": per_fold[fold] for fold in range(K_FOLDS)},
            "fold_mean": _mean_finite(per_fold),
            "fold_std": _std_finite(per_fold),
            "consistency": _sign_consistency(per_fold),
        }
        corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OOF_ROOT / "table_B_difficulty_correlations.csv", index=False)

    # --- attribution-only rows (never candidates) ---
    attr_rows = []
    for name in ("z_SA", "z_logP", "z_cycle"):
        values = table[name].to_numpy(dtype=np.float64)
        per_fold = [
            _spearman(values[table["outer_fold"].to_numpy() == fold], err[table["outer_fold"].to_numpy() == fold])
            for fold in range(K_FOLDS)
        ]
        attr_rows.append(
            {
                "component": name,
                "overall_spearman_abs": _spearman(values, err),
                "overall_spearman_signed": _spearman(values, signed),
                "fold_mean_abs": _mean_finite(per_fold),
                "consistency_abs": _sign_consistency(per_fold),
            }
        )
    pd.DataFrame(attr_rows).to_csv(OOF_ROOT / "target_component_attribution.csv", index=False)

    # --- rarity bin table (§13) ---
    rare = table["rare_le5_ratio"].to_numpy(dtype=np.float64)
    fixed_edges = [0.0, 1e-9, 0.01, 0.05, 0.10, 0.20, 1.0]
    # labelled bins are half-open (edge_{k-1}, edge_k] on the labels below:
    # side="left" places a boundary value (e.g. rare==0.05) into the bin whose
    # printed label contains it, i.e. (0.01, 0.05], not the next bin up.
    labels = ["0", "(0,0.01]", "(0.01,0.05]", "(0.05,0.1]", "(0.1,0.2]", ">0.2"]
    bin_index = np.searchsorted(fixed_edges[1:], rare, side="left")
    bin_index = np.clip(bin_index, 0, len(labels) - 1)
    bin_rows = []
    for b in range(len(labels)):
        mask = bin_index == b
        if mask.sum() == 0:
            continue
        bin_rows.append(
            {
                "rarity_bin": labels[b],
                "n": int(mask.sum()),
                "oof_mae": float(np.mean(err[mask])),
                "median_abs_error": float(np.median(err[mask])),
                "signed_residual_mean": float(np.mean(signed[mask])),
                "mean_rare_le5_ratio": float(np.mean(rare[mask])),
            }
        )
    pd.DataFrame(bin_rows).to_csv(OOF_ROOT / "rarity_bin_table.csv", index=False)

    # --- quintile bin tables for selected leading features ---
    def quantile_bin_table(column: str) -> pd.DataFrame:
        values = table[column].to_numpy(dtype=np.float64)
        labels = pd.qcut(values, 5, duplicates="drop") if not np.allclose(values, values[0]) else pd.Series(np.zeros(len(values), dtype=int))
        frame = pd.DataFrame({"bin": labels, "err": err, "signed": signed})
        grouped = frame.groupby("bin", observed=True)
        out = grouped.agg(
            n=("err", "size"),
            oof_mae=("err", "mean"),
            median_abs_error=("err", "median"),
            signed_residual_mean=("signed", "mean"),
        ).reset_index()
        out["bin"] = out["bin"].astype(str)
        return out

    for column in ["rare_le5_ratio", "oov_patch_ratio", "disagreement"]:
        if column not in table.columns:
            continue
        qt = quantile_bin_table(column)
        qt.to_csv(OOF_ROOT / f"quintile_bins_{column}.csv", index=False)

    # --- partial analyses (§29/§30): error ~ rarity/disagreement/complexity ---
    partial_rows: list[dict[str, Any]] = []
    if "disagreement" in table.columns:
        dis = table["disagreement"].to_numpy(dtype=np.float64)
        rare5 = table["rare_le5_ratio"].to_numpy(dtype=np.float64)
        diameter = table["diameter"].to_numpy(dtype=np.float64)
        partial_rows.append(
            {
                "relation": "abs_error ~ rare_le5 | disagreement",
                "partial_spearman": _partial_spearman(err, rare5, dis),
                "uncontrolled_spearman": _spearman(err, rare5),
            }
        )
        partial_rows.append(
            {
                "relation": "abs_error ~ disagreement | rare_le5",
                "partial_spearman": _partial_spearman(err, dis, rare5),
                "uncontrolled_spearman": _spearman(err, dis),
            }
        )
        partial_rows.append(
            {
                "relation": "abs_error ~ diameter | disagreement",
                "partial_spearman": _partial_spearman(err, diameter, dis),
                "uncontrolled_spearman": _spearman(err, diameter),
            }
        )
        partial_rows.append(
            {
                "relation": "abs_error ~ rare_le5 | z_SA (attribution)",
                "partial_spearman": _partial_spearman(err, rare5, table["z_SA"].to_numpy()),
                "uncontrolled_spearman": _spearman(err, rare5),
            }
        )
        partial_rows.append(
            {
                "relation": "abs_error ~ disagreement | z_SA (attribution)",
                "partial_spearman": _partial_spearman(err, dis, table["z_SA"].to_numpy()),
                "uncontrolled_spearman": _spearman(err, dis),
            }
        )
    else:
        rare5 = table["rare_le5_ratio"].to_numpy(dtype=np.float64)
        partial_rows.append(
            {
                "relation": "abs_error ~ rare_le5 | z_SA (attribution)",
                "partial_spearman": _partial_spearman(err, rare5, table["z_SA"].to_numpy()),
                "uncontrolled_spearman": _spearman(err, rare5),
            }
        )
    pd.DataFrame(partial_rows).to_csv(OOF_ROOT / "partial_analysis.csv", index=False)

    # --- signed-residual sanity check for leading features ---
    signed_rows = []
    for name in feature_cols[:25]:
        values = table[name].to_numpy(dtype=np.float64)
        signed_rows.append(
            {
                "feature": name,
                "spearman_signed": _spearman(values, signed),
                "spearman_abs": _spearman(values, err),
            }
        )
    pd.DataFrame(signed_rows).to_csv(OOF_ROOT / "signed_sanity_leading_features.csv", index=False)

    # --- Table E: epistemic signal / ensemble gain (§27) ---
    if len(seeds) >= 2:
        dis = table["disagreement"].to_numpy(dtype=np.float64)
        per_fold_dis = [
            _spearman(
                dis[table["outer_fold"].to_numpy() == fold],
                err[table["outer_fold"].to_numpy() == fold],
            )
            for fold in range(K_FOLDS)
        ]
        individual_mae = float(
            np.mean(
                (table["absolute_error_seed0"].to_numpy() + table["absolute_error_seed1"].to_numpy()) / 2.0
            )
        )
        mae0 = float(np.mean(table["absolute_error_seed0"].to_numpy()))
        mae1 = float(np.mean(table["absolute_error_seed1"].to_numpy()))
        ensemble_mae = float(np.mean(table["ensemble_absolute_error"].to_numpy()))
        table_e = {
            "disagreement_error_spearman_overall": _spearman(dis, err),
            "disagreement_error_spearman_fold_mean": _mean_finite(per_fold_dis),
            "disagreement_error_spearman_fold_std": _std_finite(per_fold_dis),
            "disagreement_error_consistency": _sign_consistency(per_fold_dis),
            "individual_mae_mean": individual_mae,
            "individual_mae_seed0": mae0,
            "individual_mae_seed1": mae1,
            "ensemble_mae": ensemble_mae,
            "ensemble_gain_vs_mean_individual": float((mae0 + mae1) / 2.0 - ensemble_mae),
            "per_fold_disagreement_spearman": {
                f"fold{f}": per_fold_dis[f] for f in range(K_FOLDS)
            },
        }
        # ensemble gain by observed-error quintile
        labels = _rank_quintile_labels(err)
        rows_e = []
        for b in range(5):
            mask = labels == b
            rows_e.append(
                {
                    "difficulty_bin": f"Q{b+1}",
                    "n": int(mask.sum()),
                    "individual_mae": float(
                        np.mean(
                            (table["absolute_error_seed0"].to_numpy()[mask] + table["absolute_error_seed1"].to_numpy()[mask]) / 2.0
                        )
                    ),
                    "ensemble_mae": float(np.mean(table["ensemble_absolute_error"].to_numpy()[mask])),
                    "ensemble_gain": float(
                        np.mean(
                            (table["absolute_error_seed0"].to_numpy()[mask] + table["absolute_error_seed1"].to_numpy()[mask]) / 2.0
                        )
                        - np.mean(table["ensemble_absolute_error"].to_numpy()[mask])
                    ),
                }
            )
        pd.DataFrame(rows_e).to_csv(OOF_ROOT / "ensemble_gain_by_error_bin.csv", index=False)
        _write_json(OOF_ROOT / "table_E_epistemic_signal.json", table_e)
    summary = {
        "n_features": int(len(feature_cols)),
        "seeds": [int(s) for s in seeds],
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("analysis", summary)


# ---------------------------------------------------------------------------
# stage: predictor (nested cross-fitted difficulty predictor)
# ---------------------------------------------------------------------------

def _ridge_cross_fit(
    table: pd.DataFrame,
    feature_cols: Sequence[str],
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Outer-fold nested CV: for fold k, fit StandardScaler+Ridge on folds
    != k (log-target) and predict fold k.  Returns (exp-scale predictions,
    metrics)."""
    X = table[list(feature_cols)].to_numpy(dtype=np.float64)
    folds = table["outer_fold"].to_numpy(dtype=np.int64)
    log_target = np.log(target + 1e-3)
    pred_log = np.full(len(table), np.nan, dtype=np.float64)
    for fold in range(K_FOLDS):
        train_mask = folds != fold
        test_mask = folds == fold
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = log_target[train_mask]
        finite = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        scaler = StandardScaler().fit(X_train[finite])
        model = Ridge(alpha=1.0)
        model.fit(scaler.transform(X_train[finite]), y_train[finite])
        pred_log[test_mask] = model.predict(scaler.transform(X_test))
    pred_exp = np.exp(pred_log) - 1e-3
    pred_exp = np.clip(pred_exp, 0.0, None)
    metrics = {
        "spearman": _spearman(pred_exp, target),
        "pearson": _pearson(pred_exp, target),
        "r2_log_scale": float(r2_score(log_target, pred_log)),
        "r2_abs_scale": float(r2_score(target, pred_exp)),
    }
    return pred_exp, metrics


def stage_predictor(force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(OOF_ROOT / "oof_per_molecule.csv")
    err = _error_target(table)

    variant_defs: dict[str, list[str]] = {
        "rarity_only": CANDIDATE_BLOCKS["rarity"],
        "structure_only": CANDIDATE_BLOCKS["structure"] + CANDIDATE_BLOCKS["topology_raw"],
        "states_only": CANDIDATE_BLOCKS["states"],
        "disagreement_only": CANDIDATE_BLOCKS["disagreement"],
        "model_visible_single": (
            CANDIDATE_BLOCKS["rarity"]
            + CANDIDATE_BLOCKS["structure"]
            + CANDIDATE_BLOCKS["topology_raw"]
            + CANDIDATE_BLOCKS["states"]
        ),
        "all": (
            CANDIDATE_BLOCKS["rarity"]
            + CANDIDATE_BLOCKS["structure"]
            + CANDIDATE_BLOCKS["topology_raw"]
            + CANDIDATE_BLOCKS["states"]
            + CANDIDATE_BLOCKS["disagreement"]
        ),
    }
    results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    quintile_rows: list[dict[str, Any]] = []
    for variant, cols in variant_defs.items():
        available = [c for c in cols if c in table.columns]
        if not available:
            continue
        if variant in ("disagreement_only", "all") and "disagreement" not in table.columns:
            continue
        pred_exp, metrics = _ridge_cross_fit(table, available, err)
        predictions[variant] = pred_exp
        per_fold = []
        folds = table["outer_fold"].to_numpy(dtype=np.int64)
        for fold in range(K_FOLDS):
            mask = folds == fold
            per_fold.append(_spearman(pred_exp[mask], err[mask]))
        quintile = _rank_quintile_labels(pred_exp)
        rows = []
        for b in range(5):
            mask = quintile == b
            rows.append(
                {
                    "variant": variant,
                    "predicted_difficulty_quintile": f"Q{b+1}",
                    "n": int(mask.sum()),
                    "actual_mae": float(np.mean(err[mask])),
                    "median_abs_error": float(np.median(err[mask])),
                }
            )
        quintile_rows.extend(rows)
        hardest = rows[4]["actual_mae"] if len(rows) == 5 else float("nan")
        easiest = rows[0]["actual_mae"] if len(rows) == 5 else float("nan")
        results[variant] = {
            "n_features": int(len(available)),
            "spearman_overall": metrics["spearman"],
            "pearson_overall": metrics["pearson"],
            "r2_log_scale": metrics["r2_log_scale"],
            "r2_abs_scale": metrics["r2_abs_scale"],
            "hardest_easiest_mae_ratio": float(hardest / easiest) if easiest else float("nan"),
            "per_fold_spearman": {f"fold{f}": per_fold[f] for f in range(K_FOLDS)},
            "fold_mean_spearman": _mean_finite(per_fold),
            "fold_std_spearman": _std_finite(per_fold),
            "fold_consistency": _sign_consistency(per_fold),
        }
        print(
            f"[predictor] {variant}: spearman={metrics['spearman']:.4f} "
            f"ratio={results[variant]['hardest_easiest_mae_ratio']:.3f} "
            f"consistency={results[variant]['fold_consistency']}",
            flush=True,
        )
    pd.DataFrame(quintile_rows).to_csv(OOF_ROOT / "table_D_difficulty_quintiles.csv", index=False)
    _save_npz(
        OOF_ROOT / "difficulty_predictions.npz",
        **{f"pred_{v}": predictions[v] for v in predictions},
        absolute_error=err,
    )
    _write_json(OOF_ROOT / "table_C_difficulty_predictor.json", results)
    summary = {"variants": {k: v["spearman_overall"] for k, v in results.items()},
               "seconds": float(time.perf_counter() - started)}
    print(json.dumps(summary, indent=2))
    return _mark_done("predictor", summary)


# ---------------------------------------------------------------------------
# stage: figures
# ---------------------------------------------------------------------------

def stage_figures(force: bool = False) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    table = pd.read_csv(OOF_ROOT / "oof_per_molecule.csv")
    err = _error_target(table)

    def binned_means(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
        bins = _rank_quintile_labels(x, q=n_bins)
        xs, ys, ns = [], [], []
        for b in range(n_bins):
            mask = bins == b
            if mask.sum() == 0:
                continue
            xs.append(float(np.mean(x[mask])))
            ys.append(float(np.mean(y[mask])))
            ns.append(int(mask.sum()))
        return np.asarray(xs), np.asarray(ys), np.asarray(ns)

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4))
    rare5 = table["rare_le5_ratio"].to_numpy(dtype=np.float64)
    xs, ys, ns = binned_means(rare5, err)
    axes[0].plot(xs, ys, "o-")
    axes[0].set_xlabel("rare-le5 patch ratio (binned)")
    axes[0].set_ylabel("OOF abs error")
    axes[0].set_title("Fig1: OOF difficulty vs rare-patch ratio")
    axes[1].scatter(rare5, err, s=4, alpha=0.15)
    axes[1].set_xlabel("rare-le5 patch ratio")
    axes[1].set_ylabel("OOF abs error")
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "fig1_error_vs_rare_ratio.png", dpi=120)

    fig2 = None
    if "disagreement" in table.columns:
        fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
        dis = table["disagreement"].to_numpy(dtype=np.float64)
        xs, ys, ns = binned_means(dis, err)
        axes[0].plot(xs, ys, "o-")
        axes[0].set_xlabel("model disagreement (binned)")
        axes[0].set_ylabel("OOF abs error")
        axes[0].set_title("Fig2: OOF difficulty vs seed disagreement")
        axes[1].scatter(dis, err, s=4, alpha=0.15)
        axes[1].set_xlabel("disagreement")
        axes[1].set_ylabel("OOF abs error")
        fig2.tight_layout()
        fig2.savefig(FIG_DIR / "fig2_error_vs_disagreement.png", dpi=120)

    pred_path = OOF_ROOT / "difficulty_predictions.npz"
    if pred_path.exists():
        preds = _load_npz(pred_path)
        for variant in ("model_visible_single", "all"):
            key = f"pred_{variant}"
            if key not in preds:
                continue
            pred = preds[key]
            fig3, axes = plt.subplots(1, 2, figsize=(11, 4))
            xs, ys, ns = binned_means(pred, err)
            axes[0].plot(xs, ys, "o-")
            axes[0].set_xlabel("predicted difficulty (binned)")
            axes[0].set_ylabel("OOF abs error")
            axes[0].set_title(f"Fig3: predicted vs actual difficulty ({variant})")
            axes[1].scatter(pred, err, s=4, alpha=0.12)
            axes[1].set_xlabel("predicted difficulty")
            axes[1].set_ylabel("OOF abs error")
            fig3.tight_layout()
            fig3.savefig(FIG_DIR / f"fig3_pred_vs_actual_{variant}.png", dpi=120)
            quintiles = pd.read_csv(OOF_ROOT / "table_D_difficulty_quintiles.csv")
            sub = quintiles[quintiles["variant"] == variant]
            fig4, ax = plt.subplots(figsize=(6, 4))
            ax.bar(sub["predicted_difficulty_quintile"], sub["actual_mae"])
            ax.set_xlabel("predicted difficulty quintile")
            ax.set_ylabel("actual MAE")
            ax.set_title(f"Fig4: predicted quintile vs actual MAE ({variant})")
            fig4.tight_layout()
            fig4.savefig(FIG_DIR / f"fig4_quintile_mae_{variant}.png", dpi=120)
            break

    if "disagreement" in table.columns:
        fig5, axes = plt.subplots(1, 2, figsize=(11, 4))
        xs, ys, ns = binned_means(rare5, table["disagreement"].to_numpy(dtype=np.float64))
        axes[0].plot(xs, ys, "o-")
        axes[0].set_xlabel("rare-le5 ratio (binned)")
        axes[0].set_ylabel("disagreement")
        axes[0].set_title("Fig5: rare patches vs disagreement")
        axes[1].scatter(rare5, table["disagreement"].to_numpy(), s=4, alpha=0.15)
        axes[1].set_xlabel("rare-le5 ratio")
        axes[1].set_ylabel("disagreement")
        fig5.tight_layout()
        fig5.savefig(FIG_DIR / "fig5_rare_vs_disagreement.png", dpi=120)

        gain_path = OOF_ROOT / "ensemble_gain_by_error_bin.csv"
        if gain_path.exists():
            gain = pd.read_csv(gain_path)
            fig6, ax = plt.subplots(figsize=(7, 4))
            width = 0.38
            xpos = np.arange(len(gain))
            ax.bar(xpos - width / 2, gain["individual_mae"], width, label="individual")
            ax.bar(xpos + width / 2, gain["ensemble_mae"], width, label="ensemble")
            ax.set_xticks(xpos, gain["difficulty_bin"])
            ax.set_xlabel("observed error quintile")
            ax.set_ylabel("MAE")
            ax.legend()
            ax.set_title("Fig6: ensemble gain by difficulty bin")
            fig6.tight_layout()
            fig6.savefig(FIG_DIR / "fig6_ensemble_gain.png", dpi=120)

    summary = {
        "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("figures", summary)


# ---------------------------------------------------------------------------
# stage: decision
# ---------------------------------------------------------------------------

def stage_decision(force: bool = False) -> dict[str, Any]:
    table = pd.read_csv(OOF_ROOT / "oof_per_molecule.csv")
    seeds = _available_seeds()
    table_c = _read_json(OOF_ROOT / "table_C_difficulty_predictor.json")

    visible = table_c.get("model_visible_single", {})
    spearman_visible = float(visible.get("spearman_overall", float("nan")))
    consistency_visible = str(visible.get("fold_consistency", "n/a"))
    ratio_visible = float(visible.get("hardest_easiest_mae_ratio", float("nan")))

    if "disagreement" in table.columns and (OOF_ROOT / "table_E_epistemic_signal.json").exists():
        table_e = _read_json(OOF_ROOT / "table_E_epistemic_signal.json")
        dis_spearman = float(table_e["disagreement_error_spearman_overall"])
        dis_consistency = str(table_e["disagreement_error_consistency"])
    else:
        dis_spearman = float("nan")
        dis_consistency = "n/a (seed 1 not complete)"

    # difficulty-learnability verdict per pre-registered gates (§24)
    if np.isfinite(spearman_visible):
        if spearman_visible >= 0.40 and "5/5" in consistency_visible and ratio_visible >= 1.5:
            difficulty_verdict = "STRONG GO"
        elif spearman_visible >= 0.25 and "5/5" in consistency_visible:
            difficulty_verdict = "GO"
        elif spearman_visible >= 0.15:
            difficulty_verdict = "MILD"
        else:
            difficulty_verdict = "NO-GO"
    else:
        difficulty_verdict = "INCONCLUSIVE (no model-visible variant ran)"
    if np.isfinite(dis_spearman) and dis_spearman >= 0.30 and "5/5" in dis_consistency:
        epistemic_verdict = "GO"
    elif np.isfinite(dis_spearman) and dis_spearman >= 0.20:
        epistemic_verdict = "MILD"
    elif np.isfinite(dis_spearman):
        epistemic_verdict = "NO-GO"
    else:
        epistemic_verdict = "n/a"

    record = {
        "difficulty_verdict": difficulty_verdict,
        "epistemic_verdict": epistemic_verdict,
        "cross_fitted_spearman_model_visible": spearman_visible,
        "cross_fitted_consistency_model_visible": consistency_visible,
        "hardest_easiest_mae_ratio_model_visible": ratio_visible,
        "disagreement_error_spearman": dis_spearman,
        "disagreement_consistency": dis_consistency,
        "seeds": [int(s) for s in seeds],
        "n": int(len(table)),
    }
    _write_json(OOF_ROOT / "decision_record.json", record)
    print(json.dumps(record, indent=2))
    return _mark_done("decision", record)


def _write_readme() -> None:
    readme = OOF_ROOT / "README.md"
    readme.write_text(
        "# OOF Difficulty / Heteroscedasticity Audit - output manifest\n\n"
        "Diagnostic gate: K-fold OOF predictions on the official TRAIN split with the "
        "frozen compact-v4-hinge protocol (validation-selected checkpoint per outer fold, "
        "inner 7200/800 nested split). Official test and validation splits never loaded.\n\n"
        "See `tracks/ksvd/notes/oof_difficulty_heteroscedasticity_audit.md`.\n"
        "Module: `tracks/ksvd/experiments/luyin16/zinc_oof_difficulty_audit.py`.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "stages",
        nargs="*",
        choices=[
            "splits",
            "fold",
            "states",
            "features",
            "table",
            "analysis",
            "predictor",
            "figures",
            "decision",
        ],
        help="stages to run (default: all non-parallel stages)",
    )
    parser.add_argument("--fold", type=int, default=None, help="outer fold for the fold stage")
    parser.add_argument("--seed", type=int, default=None, help="model seed for fold/states stages")
    parser.add_argument("--force", action="store_true", help="rebuild artifacts")
    args = parser.parse_args(argv)
    OOF_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    available = [
        "splits",
        "fold",
        "states",
        "features",
        "table",
        "analysis",
        "predictor",
        "figures",
        "decision",
    ]
    stages = args.stages or ["splits", "features", "states", "table", "analysis", "predictor", "figures", "decision"]
    if "fold" in stages and (args.fold is None or args.seed is None):
        raise SystemExit("fold stage requires --fold and --seed")
    for stage in stages:
        if stage not in available:
            raise SystemExit(f"unknown stage {stage!r}")
        if stage == "fold":
            started = time.perf_counter()
            print(f"[stage] fold {args.fold} seed {args.seed} ...", flush=True)
            stage_fold(int(args.fold), int(args.seed), force=args.force)
            print(f"[done] fold in {time.perf_counter() - started:.1f}s", flush=True)
            continue
        if args.force or not _is_done(stage):
            started = time.perf_counter()
            print(f"[stage] {stage} ...", flush=True)
            if stage == "states":
                stage_states(model_seed=int(args.seed if args.seed is not None else 0), force=args.force)
            else:
                globals()[f"stage_{stage}"](force=args.force)
            print(f"[done] {stage} in {time.perf_counter() - started:.1f}s", flush=True)
        else:
            print(f"[skip] {stage} already done (use --force)", flush=True)
    _write_readme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
