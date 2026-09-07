"""ZINC transfer of the mentor ``S + T + A`` K-SVD readout.

The MolHIV artifact proxy has a deliberately small, explicit readout:

``S`` (graph composition) + ``T`` (pooled K-SVD reconstruction) + ``A``
(context-conditioned sparse-code mass).  This module applies the same
mechanism to the official PyG ZINC subset.  ZINC already has a cached exact
canonical radius-2 typed patch (840 coordinates) from the earlier exact-patch
experiment; the cache is reused and only the train-only dictionary/readout is
constructed here.

ZINC does not expose OGB's aromatic atom flag.  Consequently the five
contexts are ``ring5``, ``ring6``, ``ring_any``, ``multi_ring`` and
``ring_boundary``.  ``ring_any`` is an explicit ZINC proxy for the aromatic
context and is recorded as such in the manifest; it is not silently presented
as an aromatic label.

The dictionary and feature transforms are fitted on official train only.
Optuna searches XGBoost hyperparameters on three shuffled official-train
folds, one model seed (0), then evaluates frozen parameters on official valid
and on official test after train+valid classifier refitting.  The dictionary
itself remains the train-only dictionary, matching the MolHIV artifact
evaluation protocol.  The current transfer report evaluates ``sta_final``;
the optional covariance block is constructed and cached for later ablation,
but its search is intentionally not run in this report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import yaml
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from ksvd_research.features import build_ring_context_index
from tracks.ksvd.experiments.luyin16 import zinc_ksvd_patch_path_pooling as exact
from tracks.ksvd.experiments.luyin16.molhiv_ksvd_patch_path_pooling import _batch_omp
from tracks.ksvd.experiments.luyin16.mentor_artifact_typed_ksvd import (
    _ksvd_update,
    _normalize_columns,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _data_to_graph,
    _load_zinc,
    _resolve,
    REPO_ROOT,
    source_audit,
)


PATCH_WIDTH = int(exact.PATCH_WIDTH)
N_ATOMS = 64
SPARSITY = 8
TYPED_WIDTH = 3 * PATCH_WIDTH
CONTEXT_NAMES = (
    "ring5",
    "ring6",
    "ring_any",
    "multi_ring",
    "ring_boundary",
)
CONTEXT_WIDTH = len(CONTEXT_NAMES) * N_ATOMS + len(CONTEXT_NAMES)
CROSS_COV_WIDTH = len(CONTEXT_NAMES) * N_ATOMS

DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_mentor_artifact_typed_ksvd.yaml"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sample_raw_rows(records: Sequence[Any], maximum: int, seed: int) -> np.ndarray:
    """Reservoir-sample raw patch descriptors without materialising all patches."""
    limit = int(maximum)
    if limit < 2:
        raise ValueError("dictionary sample must contain at least two patches")
    rng = np.random.default_rng(int(seed))
    reservoir: list[np.ndarray] = []
    seen = 0
    for record in records:
        for patch in record.patches:
            row = np.asarray(patch.shell_descriptor, dtype=np.float32)
            if row.shape != (PATCH_WIDTH,):
                raise ValueError(f"unexpected patch width {row.shape}, expected {(PATCH_WIDTH,)}")
            if seen < limit:
                reservoir.append(row.copy())
            else:
                replacement = int(rng.integers(0, seen + 1))
                if replacement < limit:
                    reservoir[replacement] = row.copy()
            seen += 1
    if len(reservoir) < 2:
        raise RuntimeError("not enough train patches for dictionary learning")
    return np.stack(reservoir, axis=0).astype(np.float32, copy=False)


def fit_dictionary(
    train_records: Sequence[Any],
    *,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    power_iterations: int,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the same RMS/L2-normalised K-SVD convention used by the MolHIV proxy."""
    rows = _sample_raw_rows(train_records, maximum, seed + 1729)
    scale = np.sqrt(np.mean(np.square(rows), axis=0, dtype=np.float64)).astype(np.float32)
    scale = np.maximum(scale, np.float32(1.0e-4))
    scaled = rows / scale[None, :]
    norms = np.maximum(np.linalg.norm(scaled, axis=1), 1.0e-12)
    samples = (scaled / norms[:, None]).T.astype(np.float32, copy=False)
    if int(n_atoms) > samples.shape[1]:
        raise ValueError("dictionary atoms exceed sampled patches")
    _, indices = kmeans_plusplus(
        samples.T,
        n_clusters=int(n_atoms),
        random_state=int(seed),
    )
    initial = _normalize_columns(samples[:, np.asarray(indices, dtype=np.int64)])
    final = initial.copy()
    gram = final.T @ final
    initial_codes, _ = _batch_omp(samples.T, final, int(sparsity), gram=gram)
    initial_error = float(
        np.linalg.norm(samples - final @ initial_codes.T)
        / max(float(np.linalg.norm(samples)), 1.0e-12)
    )
    curve: list[float] = []
    rng = np.random.default_rng(int(seed) + 1009)
    revived_total = 0
    for iteration in range(int(iterations)):
        gram = final.T @ final
        codes, _ = _batch_omp(samples.T, final, int(sparsity), gram=gram)
        final, _, revived = _ksvd_update(
            samples,
            final,
            codes.T,
            power_iterations=int(power_iterations),
            rng=rng,
        )
        recoded, _ = _batch_omp(samples.T, final, int(sparsity), gram=final.T @ final)
        error = float(
            np.linalg.norm(samples - final @ recoded.T)
            / max(float(np.linalg.norm(samples)), 1.0e-12)
        )
        curve.append(error)
        revived_total += int(revived)
        print(
            f"  ZINC K-SVD iteration {iteration + 1}/{iterations}: "
            f"relative reconstruction={error:.6f}, revived={revived}",
            flush=True,
        )
    info = {
        "training_patch_count": int(rows.shape[0]),
        "patch_dimension": PATCH_WIDTH,
        "n_atoms": int(n_atoms),
        "sparsity": int(sparsity),
        "iterations": int(iterations),
        "initial_relative_reconstruction": initial_error,
        "final_relative_reconstruction": curve[-1] if curve else initial_error,
        "reconstruction_curve": curve,
        "revived_atoms_total": int(revived_total),
        "initialization": "sklearn kmeans++ indices over real train patches",
        "coordinate_scaling": "train-only coordinate RMS, then per-patch L2 normalization",
    }
    return initial, final, scale, info


def _zinc_root_context_mask(data: Any, record: Any) -> np.ndarray:
    """Return one context row per centre; aromatic is unavailable in ZINC."""
    graph, _node_types, _edge_types = _data_to_graph(data)
    # ``build_ring_context_index`` accepts OGB atom flags when present.  ZINC
    # has only one categorical atom column, so an all-zero nine-column proxy
    # intentionally supplies no aromatic flag.
    dummy = np.zeros((int(graph.n), 9), dtype=np.int64)
    index = build_ring_context_index(graph, dummy)
    ring_nodes = set(index.ring_nodes)
    boundary_nodes: set[int] = set()
    for left, right in index.boundary_edges:
        if left in ring_nodes and right not in ring_nodes:
            boundary_nodes.add(int(right))
        elif right in ring_nodes and left not in ring_nodes:
            boundary_nodes.add(int(left))
    centres = list(graph.nodes)
    if len(centres) != len(record.patches):
        raise RuntimeError("record centre count does not match dataset graph")
    rows = np.zeros((len(centres), len(CONTEXT_NAMES)), dtype=bool)
    for position, centre in enumerate(centres):
        rows[position] = (
            centre in index.ring5_nodes,
            centre in index.ring6_nodes,
            centre in ring_nodes,
            centre in index.multi_ring_nodes,
            centre in boundary_nodes,
        )
    return rows


def _context_mass(atom_use: np.ndarray, masks: np.ndarray) -> np.ndarray:
    blocks: list[np.ndarray] = []
    coverage: list[float] = []
    for context in range(masks.shape[1]):
        selected = masks[:, context]
        blocks.append(
            atom_use[selected].sum(axis=0) / max(int(atom_use.shape[0]), 1)
            if np.any(selected)
            else np.zeros(atom_use.shape[1], dtype=np.float32)
        )
        coverage.append(float(selected.mean()) if selected.size else 0.0)
    return np.concatenate([*blocks, np.asarray(coverage, dtype=np.float32)]).astype(
        np.float32, copy=False
    )


def _context_cross_cov(atom_use: np.ndarray, masks: np.ndarray) -> np.ndarray:
    values = np.asarray(atom_use, dtype=np.float32)
    indicators = np.asarray(masks, dtype=np.float32)
    if values.shape[0] != indicators.shape[0]:
        raise ValueError(f"atom/context mismatch: {values.shape}/{indicators.shape}")
    if values.shape[0] <= 1:
        return np.zeros(CROSS_COV_WIDTH, dtype=np.float32)
    values -= values.mean(axis=0, keepdims=True)
    indicators -= indicators.mean(axis=0, keepdims=True)
    return ((values.T @ indicators) / float(values.shape[0])).reshape(-1).astype(
        np.float32, copy=False
    )


def _pool(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(TYPED_WIDTH, dtype=np.float32)
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0), values.max(axis=0)]
    ).astype(np.float32, copy=False)


def encode_records(
    records: Sequence[Any],
    context_masks: Sequence[np.ndarray],
    dictionary: np.ndarray,
    scale: np.ndarray,
    sparsity: int,
    *,
    block_records: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Encode patch records and return T, A, cross-cov graph matrices."""
    typed_rows: list[np.ndarray] = []
    context_rows: list[np.ndarray] = []
    cross_rows: list[np.ndarray] = []
    residual_energy = 0.0
    input_energy = 0.0
    dictionary = np.asarray(dictionary, dtype=np.float32)
    gram = dictionary.T @ dictionary
    for block_start in range(0, len(records), int(block_records)):
        block = records[block_start : block_start + int(block_records)]
        values_by_graph: list[np.ndarray] = []
        offsets = [0]
        for record in block:
            values = np.stack(
                [np.asarray(patch.shell_descriptor, dtype=np.float32) for patch in record.patches],
                axis=0,
            )
            values_by_graph.append(values)
            offsets.append(offsets[-1] + int(values.shape[0]))
        packed = np.concatenate(values_by_graph, axis=0)
        normalized_values = packed / scale[None, :]
        norms = np.maximum(np.linalg.norm(normalized_values, axis=1), 1.0e-12)
        normalized_values = normalized_values / norms[:, None]
        codes, _errors = _batch_omp(
            normalized_values,
            dictionary,
            int(sparsity),
            gram=gram,
        )
        reconstruction = (codes @ dictionary.T) * norms[:, None] * scale[None, :]
        # The exact descriptor is a collection of binary slot coordinates.
        reconstruction = np.clip(reconstruction, 0.0, 1.0).astype(np.float32, copy=False)
        residual_energy += float(
            np.square(normalized_values - codes @ dictionary.T).sum(dtype=np.float64)
        )
        input_energy += float(np.square(normalized_values).sum(dtype=np.float64))
        atom_use = np.abs(codes).astype(np.float32, copy=False)
        denominator = atom_use.sum(axis=1, keepdims=True)
        np.divide(atom_use, denominator, out=atom_use, where=denominator > 1.0e-12)
        for local, _record in enumerate(block):
            start = int(offsets[local])
            stop = int(offsets[local + 1])
            typed_rows.append(_pool(reconstruction[start:stop]))
            local_masks = np.asarray(context_masks[block_start + local], dtype=bool)
            local_use = atom_use[start:stop]
            context_rows.append(_context_mass(local_use, local_masks))
            cross_rows.append(_context_cross_cov(local_use, local_masks))
        done = min(block_start + int(block_records), len(records))
        if done % 2000 == 0 or done == len(records):
            print(f"  ZINC encoded {done}/{len(records)} graphs", flush=True)
    audit = {
        "normalized_patch_relative_reconstruction": float(
            np.sqrt(residual_energy / max(input_energy, 1.0e-12))
        ),
        "graphs": int(len(records)),
        "typed_width": TYPED_WIDTH,
        "context_width": CONTEXT_WIDTH,
        "cross_cov_width": CROSS_COV_WIDTH,
    }
    return (
        np.stack(typed_rows).astype(np.float32, copy=False),
        np.stack(context_rows).astype(np.float32, copy=False),
        np.stack(cross_rows).astype(np.float32, copy=False),
        audit,
    )


def _load_or_build_records(root: Path, datasets: Sequence[Any], cache_dir: Path) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    records: dict[str, list[Any]] = {}
    metadata: dict[str, Any] = {}
    certificate_cache: dict[bytes, Any] = {}
    # The existing cache was generated by the exact-patch runner before its
    # base-module hash changed.  Its records are still the same immutable
    # canonical descriptors, so load them with a structural/schema check
    # instead of silently rebuilding ~23k patches per split.
    for name, dataset in zip(("train", "valid", "test"), datasets, strict=True):
        path = cache_dir / f"exact_{name}.pkl"
        if path.exists():
            import __main__
            if not hasattr(__main__, "ExactGraphRecord"):
                setattr(__main__, "ExactGraphRecord", exact.ExactGraphRecord)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, Mapping) or payload.get("schema") != "zinc-exact-ksvd-record-cache-v1":
                raise ValueError(f"invalid exact record cache: {path}")
            rows = payload.get("records")
            info = payload.get("metadata")
            if not isinstance(rows, list) or not isinstance(info, Mapping) or len(rows) != len(dataset):
                raise ValueError(f"exact record cache size/schema mismatch: {path}")
            records[name] = rows
            metadata[name] = {**dict(info), "cache_hit": True, "cache_signature_checked": "schema/size"}
            print(f"ZINC mentor exact record cache hit: {path}", flush=True)
        else:
            rows, info = exact._extract_split(dataset, name, certificate_cache)
            records[name] = rows
            metadata[name] = {**dict(info), "cache_hit": False, "cache_signature_checked": "fresh"}
            exact._save_record_cache(
                path,
                signature="mentor-transfer-cache-v1",
                records=rows,
                metadata=info,
            )
    metadata["certificate_cache_entries"] = int(len(certificate_cache))
    return records, metadata


def _feature_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output = config["output"]
    return (
        _resolve(output["features"]),
        _resolve(output["manifest"]),
        _resolve(output["json"]),
    )


def build_features(config: Mapping[str, Any], datasets: Sequence[Any]) -> dict[str, Any]:
    feature_path, manifest_path, _ = _feature_paths(config)
    if feature_path.exists() and manifest_path.exists():
        print(f"ZINC mentor features cache hit: {feature_path}", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    root = _resolve(config["data"]["root"])
    cache_dir = _resolve(config["runtime"]["record_cache"])
    records, record_meta = _load_or_build_records(root, datasets, cache_dir)
    train_records = records["train"]
    print("Fitting train-only ZINC artifact dictionary", flush=True)
    dictionary_initial, dictionary_final, scale, dictionary_info = fit_dictionary(
        train_records,
        n_atoms=int(config["dictionary"].get("n_atoms", N_ATOMS)),
        sparsity=int(config["dictionary"].get("sparsity", SPARSITY)),
        iterations=int(config["dictionary"].get("iterations", 5)),
        power_iterations=int(config["dictionary"].get("power_iterations", 6)),
        maximum=int(config["dictionary"].get("max_train_patches", 20000)),
        seed=int(config.get("seed", 0)),
    )
    masks: dict[str, list[np.ndarray]] = {"train": [], "valid": [], "test": []}
    for name, dataset in zip(("train", "valid", "test"), datasets, strict=True):
        masks[name] = [_zinc_root_context_mask(data, record) for data, record in zip(dataset, records[name], strict=True)]
        print(f"ZINC context masks complete: {name} ({len(masks[name])} graphs)", flush=True)

    typed: dict[str, np.ndarray] = {}
    context: dict[str, np.ndarray] = {}
    cross: dict[str, np.ndarray] = {}
    encoding_meta: dict[str, Any] = {}
    for name in ("train", "valid", "test"):
        typed[name], context[name], cross[name], encoding_meta[name] = encode_records(
            records[name],
            masks[name],
            dictionary_final,
            scale,
            int(config["dictionary"].get("sparsity", SPARSITY)),
        )

    # ``global_context`` is the existing 62-D, label-free ZINC S block.
    global_context = {
        name: np.stack([record.global_context for record in records[name]], axis=0).astype(
            np.float32, copy=False
        )
        for name in ("train", "valid", "test")
    }
    labels = {
        name: np.asarray([record.y for record in records[name]], dtype=np.float32)
        for name in ("train", "valid", "test")
    }
    arrays: dict[str, np.ndarray] = {}
    for name in ("train", "valid", "test"):
        arrays[f"{name}_s"] = global_context[name]
        arrays[f"{name}_t"] = typed[name]
        arrays[f"{name}_a"] = context[name]
        arrays[f"{name}_cross_cov"] = cross[name]
        arrays[f"{name}_y"] = labels[name]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            dictionary_initial=dictionary_initial,
            dictionary_final=dictionary_final,
            coordinate_scale=scale,
            **arrays,
        )
    temporary.replace(feature_path)
    manifest = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "feature_file": str(feature_path.resolve()),
        "feature_file_sha256": _sha256(feature_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "data": {
            "root": str(root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {name: len(dataset) for name, dataset in zip(("train", "valid", "test"), datasets, strict=True)},
            "source": source_audit(root),
            "test_labels_used_for_features": False,
        },
        "representation": {
            "S": "global_all from existing label-free ZINC graph statistics",
            "S_dimension": int(global_context["train"].shape[1]),
            "T": "mean/std/max pooled reconstructed exact canonical typed patch",
            "patch_dimension": PATCH_WIDTH,
            "T_dimension": TYPED_WIDTH,
            "A": "normalised absolute FINAL K-SVD atom-use mass by root context plus coverage",
            "A_dimension": CONTEXT_WIDTH,
            "cross_cov": "population covariance between normalised atom-use and root-context indicators",
            "cross_cov_dimension": CROSS_COV_WIDTH,
            "sta_final_dimension": int(global_context["train"].shape[1] + TYPED_WIDTH + CONTEXT_WIDTH),
            "sta_cross_cov_dimension": int(global_context["train"].shape[1] + TYPED_WIDTH + CONTEXT_WIDTH + CROSS_COV_WIDTH),
            "context_names": list(CONTEXT_NAMES),
            "ring_any_note": "ZINC has no OGB aromatic atom flag; ring_any is the explicit cycle-membership proxy.",
            "dictionary_fit": "official train patches only",
            "dictionary_atoms": int(config["dictionary"].get("n_atoms", N_ATOMS)),
            "sparsity": int(config["dictionary"].get("sparsity", SPARSITY)),
            "ksvd_iterations": int(config["dictionary"].get("iterations", 5)),
        },
        "record_cache": record_meta,
        "dictionary": dictionary_info,
        "encoding": encoding_meta,
        "leakage_audit": {
            "dictionary_graph_scope": "official train only",
            "coordinate_scale_scope": "official train patches only",
            "context_scope": "label-free graph topology only",
            "test_labels_used": False,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _trial_params(trial: optuna.Trial, config: Mapping[str, Any]) -> dict[str, Any]:
    ranges = config["tuning"]["ranges"]
    return {
        "n_estimators": trial.suggest_int("n_estimators", int(ranges["n_estimators"][0]), int(ranges["n_estimators"][1])),
        "max_depth": trial.suggest_int("max_depth", int(ranges["max_depth"][0]), int(ranges["max_depth"][1])),
        "learning_rate": trial.suggest_float("learning_rate", float(ranges["learning_rate"][0]), float(ranges["learning_rate"][1]), log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", float(ranges["min_child_weight"][0]), float(ranges["min_child_weight"][1]), log=True),
        "subsample": trial.suggest_float("subsample", float(ranges["subsample"][0]), float(ranges["subsample"][1])),
        "colsample_bytree": trial.suggest_float("colsample_bytree", float(ranges["colsample_bytree"][0]), float(ranges["colsample_bytree"][1])),
        "reg_lambda": trial.suggest_float("reg_lambda", float(ranges["reg_lambda"][0]), float(ranges["reg_lambda"][1]), log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", float(ranges["reg_alpha"][0]), float(ranges["reg_alpha"][1]), log=True),
        "gamma": trial.suggest_float("gamma", float(ranges["gamma"][0]), float(ranges["gamma"][1])),
    }


def _fit_mae(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, params: Mapping[str, Any], seed: int, n_jobs: int, objective: str, max_bin: int) -> float:
    model_params = dict(params)
    model_params.update({
        "objective": str(objective),
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_bin": int(max_bin),
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
    })
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)
    return float(mean_absolute_error(y_eval, model.predict(x_eval)))


def _tune_view(
    name: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
    storage: Path,
    *,
    reuse_only: bool = False,
) -> dict[str, Any]:
    tuning = config["tuning"]
    study_name = f"{config['protocol_id']}__{name}"
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(tuning.get("seed", 0)) + (0 if name == "sta_final" else 100003)),
        study_name=study_name,
        storage=f"sqlite:///{storage.resolve()}",
        load_if_exists=True,
    )
    target = int(tuning.get("n_trials", 12))
    def objective(trial: optuna.Trial) -> float:
        params = _trial_params(trial, config)
        scores = [
            _fit_mae(
                matrix[train], labels[train], matrix[valid], labels[valid], params,
                int(config.get("seed", 0)), int(config["runtime"].get("n_jobs", 4)),
                str(config["xgboost"].get("objective", "reg:absoluteerror")),
                int(config["xgboost"].get("max_bin", 256)),
            )
            for train, valid in folds
        ]
        trial.set_user_attr("fold_mae", [float(value) for value in scores])
        return float(np.mean(scores))
    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if reuse_only and complete < target:
        raise RuntimeError(
            f"reuse-only requested for {name}, but only {complete}/{target} "
            "Optuna trials are complete"
        )
    if complete < target:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=target - complete, show_progress_bar=False)
    best = study.best_trial
    return {
        "view": name,
        "dimension": int(matrix.shape[1]),
        "study_name": study_name,
        "storage": str(storage.resolve()),
        "n_trials": target,
        "completed_trials": int(sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)),
        "best_trial": int(best.number),
        "best_cv_mae": float(best.value),
        "best_cv_fold_mae": [float(v) for v in best.user_attrs.get("fold_mae", [])],
        "best_params": dict(best.params),
        "trials": [
            {
                "trial": int(t.number),
                "state": str(t.state),
                "value": None if t.value is None else float(t.value),
                "params": dict(t.params),
            }
            for t in study.trials
        ],
    }


def _evaluate(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, params: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    model_params = dict(params)
    model_params.update({
        "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_bin": int(config["xgboost"].get("max_bin", 256)),
        "random_state": int(config.get("seed", 0)),
        "n_jobs": int(config["runtime"].get("n_jobs", 4)),
    })
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
    return {
        "seed": int(config.get("seed", 0)),
        "fit_rows": int(x_train.shape[0]),
        "score_rows": int(x_eval.shape[0]),
        "mae": float(mean_absolute_error(y_eval, prediction)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    rows = []
    for name, value in result["evaluation"].items():
        rows.append(
            f"| `{name}` | {value['dimension']} | {value['tuning']['best_cv_mae']:.6f} | "
            f"{value['valid']['mae']:.6f} | {value['test_after_train_valid_refit']['mae']:.6f} |"
        )
    return "\n".join(
        [
            f"# {result['protocol_id']}",
            "",
            "Single-seed ZINC transfer of the mentor-aligned S + T + A K-SVD proxy.",
            "",
            f"- seed: `{result['seed']}`; official split sizes: `{result['data']['sizes']}`",
            f"- dictionary: train-only `K={result['representation']['dictionary_atoms']}`, `T={result['representation']['sparsity']}`; exact patch width `{result['representation']['patch_dimension']}`",
            f"- contexts: `{', '.join(result['representation']['context_names'])}`",
            "",
            "| view | dimension | train CV MAE | valid MAE | test MAE after train+valid refit |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            f"- best valid view: `{result['selection']['best_view_by_valid']}`; best valid MAE `{result['selection']['best_valid_mae']:.6f}`",
            f"- selected-view test MAE: `{result['selection']['selected_view_test_mae']:.6f}`",
            "",
            "The ZINC `ring_any` context is a cycle-membership proxy because ZINC has no OGB aromatic atom flag.",
            "",
        ]
    )


def run(config_path: Path, *, reuse_only: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("configuration must be a mapping")
    started = time.perf_counter()
    root = _resolve(config["data"]["root"])
    datasets = tuple(_load_zinc(root, split) for split in ("train", "val", "test"))
    manifest = build_features(config, datasets)
    feature_path, _manifest_path, result_path = _feature_paths(config)
    arrays = _load_features(feature_path)
    views = {
        "sta_final": {
            split: np.concatenate([arrays[f"{split}_s"], arrays[f"{split}_t"], arrays[f"{split}_a"]], axis=1).astype(np.float32, copy=False)
            for split in ("train", "valid", "test")
        },
    }
    labels = {split: np.asarray(arrays[f"{split}_y"], dtype=np.float32) for split in ("train", "valid", "test")}
    kfold = KFold(n_splits=int(config["tuning"].get("n_folds", 3)), shuffle=True, random_state=int(config["tuning"].get("split_seed", 0)))
    folds = [(np.asarray(train, dtype=np.int64), np.asarray(valid, dtype=np.int64)) for train, valid in kfold.split(views["sta_final"]["train"])]
    storage = _resolve(config["output"]["storage"])
    storage.parent.mkdir(parents=True, exist_ok=True)
    tuning: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    for name in ("sta_final",):
        print(f"Optuna tuning ZINC {name} ({config['tuning'].get('n_trials', 12)} trials)", flush=True)
        tuning[name] = _tune_view(
            name,
            views[name]["train"],
            labels["train"],
            folds,
            config,
            storage,
            reuse_only=reuse_only,
        )
        valid_result = _evaluate(views[name]["train"], labels["train"], views[name]["valid"], labels["valid"], tuning[name]["best_params"], config)
        train_valid_x = np.concatenate([views[name]["train"], views[name]["valid"]], axis=0)
        train_valid_y = np.concatenate([labels["train"], labels["valid"]], axis=0)
        test_result = _evaluate(train_valid_x, train_valid_y, views[name]["test"], labels["test"], tuning[name]["best_params"], config)
        evaluation[name] = {
            "dimension": int(views[name]["train"].shape[1]),
            "tuning": tuning[name],
            "valid": valid_result,
            "test_after_train_valid_refit": test_result,
        }
        print(f"  {name}: valid MAE={valid_result['mae']:.6f} test MAE={test_result['mae']:.6f}", flush=True)
    best_view = min(evaluation, key=lambda name: evaluation[name]["valid"]["mae"])
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "seed": int(config.get("seed", 0)),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "data": manifest["data"],
        "representation": manifest["representation"],
        "dictionary": manifest["dictionary"],
        "encoding": manifest["encoding"],
        "leakage_audit": manifest["leakage_audit"],
        "tuning_protocol": {
            "folds": int(config["tuning"].get("n_folds", 3)),
            "split": "official train shuffled KFold only",
            "n_trials_per_view": int(config["tuning"].get("n_trials", 12)),
            "model_seed": int(config.get("seed", 0)),
            "metric": "MAE (lower is better)",
        },
        "evaluation": evaluation,
        "selection": {
            "test_used_for_selection": False,
            "best_view_by_valid": best_view,
            "best_valid_mae": float(evaluation[best_view]["valid"]["mae"]),
            "selected_view_test_mae": float(evaluation[best_view]["test_after_train_valid_refit"]["mae"]),
        },
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "feature_manifest": str(_feature_paths(config)[1].resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(result_path, result)
    markdown_path = _resolve(config["output"]["markdown"])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse a completed Optuna study and forbid starting new trials",
    )
    args = parser.parse_args(argv)
    result = run(_resolve(args.config), reuse_only=bool(args.reuse_existing))
    print(
        json.dumps(
            {
                name: {
                    "valid_mae": row["valid"]["mae"],
                    "test_mae": row["test_after_train_valid_refit"]["mae"],
                }
                for name, row in result["evaluation"].items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
