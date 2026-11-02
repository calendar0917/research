"""Data access for both halves of the round.

Molecule tasks
--------------
Reuses the frozen ``lcrc_molnet_benchmark_pilot_v2`` view layer unchanged
(``mb2_data.build_view`` / ``load_test_cache``): ChemBench bytes, the published
ChemBench split files, the official ``ogbg-molhiv`` scaffold split, the frozen
``lipo_v1`` FeatureSpec, the label-free structure caches and the Train-only
target normaliser.  Nothing here re-splits a molecule dataset.

TU tasks
--------
Reuses the ``lcrc_tu_quick_screen_v1`` adapter (raw TUDataset loader, per-graph
cache builder, frozen ``collate``/``fit_norm``) but builds *per-fold* caches so
that the node-type vocabulary and the input standardisation are fitted only on
that fold's actual training part.  No old single-split vocabulary or
normalisation constant is reused.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import ptb_common as C

import lcrc_data as PREP_DATA
import mb2_data as MD
import tu_adapter as TA
import tu_data as TD
from feature_spec import FeatureSpec, load_spec, save_spec

TU_CACHE = C.CACHE / "tu"

# ---------------------------------------------------------------------------
# molecule views (read-only reuse of the v2 layer)
# ---------------------------------------------------------------------------


def molecule_view(dataset: str) -> MD.DatasetView:
    """Train+Valid view for one molecule dataset.  Test is never loaded here."""
    if dataset not in C.MOLECULE_DATASETS and dataset not in C.SUPPORT_VIEWS:
        raise KeyError(dataset)
    return MD.build_view(dataset)


def molecule_test(dataset: str):
    """Structure cache + targets of the published Test split (one-shot use)."""
    return MD.load_test_cache(dataset)


def molecule_target_scale(view: MD.DatasetView) -> np.ndarray:
    """Per-task multiplier that maps the standardised output back to raw units."""
    if view.task != "regression":
        return np.ones(view.targets.n_tasks, dtype=np.float64)
    return np.full(view.targets.n_tasks, float(view.target_norm["scale"]), dtype=np.float64)


def molecule_target_offset(view: MD.DatasetView) -> np.ndarray:
    if view.task != "regression":
        return np.zeros(view.targets.n_tasks, dtype=np.float64)
    return np.full(view.targets.n_tasks, float(view.target_norm["mean"]), dtype=np.float64)


# ---------------------------------------------------------------------------
# TU fold construction
# ---------------------------------------------------------------------------


def fold_partition(dataset: str, n_splits: int, fold: int, fold_seed: int) -> dict[str, np.ndarray]:
    """Outer stratified K-fold plus a 10% inner checkpoint-selection split.

    ``test`` is the outer held-out fold.  Its labels are used only for the nested
    stratified construction here and for the final per-fold evaluation.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    ds = TA.load_dataset(dataset)
    y = np.asarray(ds["labels"], dtype=np.int64)
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(fold_seed))
    splits = list(skf.split(np.zeros(y.size, dtype=np.int8), y))
    if len(splits) != int(n_splits):
        raise RuntimeError("StratifiedKFold did not produce the requested number of folds")
    outer_train, outer_test = splits[int(fold)]
    outer_train = np.sort(np.asarray(outer_train, dtype=np.int64))
    outer_test = np.sort(np.asarray(outer_test, dtype=np.int64))
    y_ot = y[outer_train]
    inner_seed = int(C.TU_INNER_VALID_SEED_BASE) + int(fold)
    train_idx, valid_idx = train_test_split(
        np.arange(outer_train.size, dtype=np.int64),
        test_size=float(C.TU_INNER_VALID_FRACTION),
        stratify=y_ot,
        random_state=inner_seed,
    )
    train_idx = np.sort(outer_train[np.asarray(train_idx, dtype=np.int64)])
    valid_idx = np.sort(outer_train[np.asarray(valid_idx, dtype=np.int64)])
    out = {"train": train_idx, "valid": valid_idx, "test": outer_test}
    union = np.sort(np.concatenate([out["train"], out["valid"], out["test"]]))
    if union.size != y.size or not np.array_equal(union, np.arange(y.size, dtype=np.int64)):
        raise ValueError("fold is not a partition of the dataset")
    if np.intersect1d(out["train"], out["valid"]).size or np.intersect1d(out["train"], out["test"]).size:
        raise ValueError("fold overlaps")
    return out


def fold_signature(dataset: str, n_splits: int, fold: int, fold_seed: int) -> dict[str, Any]:
    part = fold_partition(dataset, n_splits, fold, fold_seed)
    return {
        "dataset": dataset,
        "n_splits": int(n_splits),
        "fold": int(fold),
        "fold_seed": int(fold_seed),
        "inner_valid_seed": int(C.TU_INNER_VALID_SEED_BASE) + int(fold),
        "sizes": {k: int(v.size) for k, v in part.items()},
        "sha256": {k: C.index_sha256(v.tolist()) for k, v in part.items()},
    }


def _fold_cache_dir(dataset: str, fold: int) -> Path:
    return TU_CACHE / f"{dataset}_f{int(fold)}"


def prepare_fold(dataset: str, n_splits: int, fold: int, fold_seed: int) -> dict[str, Any]:
    """Build (or load) the three fold caches, the fold spec and the fold norm."""
    part = fold_partition(dataset, n_splits, fold, fold_seed)
    out_dir = _fold_cache_dir(dataset, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = TA.load_dataset(dataset)
    labels = ds["labels"]
    node_type_mode = _tu_meta(dataset)["node_type"]

    if node_type_mode == "label":
        train_labels: list[int] = []
        for gid in part["train"].tolist():
            train_labels.extend(int(x) for x in np.asarray(ds["records"][gid]["node_labels"]).tolist())
    else:
        train_labels = None
    node_types = TA.node_type_table(node_type_mode, train_labels)
    spec = TA.build_tu_spec(dataset, node_types)
    spec_path = out_dir / "spec.json"
    if spec_path.exists():
        if load_spec(spec_path).hash() != spec.hash():
            raise RuntimeError(f"{dataset}/f{fold}: stored fold spec hash differs from the regenerated spec")
    else:
        save_spec(spec, spec_path)

    caches: dict[str, dict[str, Any]] = {}
    cache_files: dict[str, Any] = {}
    for name in ("train", "valid", "test"):
        path = out_dir / f"{name}.pt"
        if path.exists():
            cache = PREP_DATA.load_cache(path)
            if str(cache["meta"]["feature_spec_hash"]) != spec.hash():
                raise RuntimeError(f"{dataset}/f{fold}/{name}: cached features use a different spec hash")
            if not np.array_equal(np.asarray(cache["gids"]), part[name]):
                raise RuntimeError(f"{dataset}/f{fold}/{name}: cached gids differ from the frozen fold")
        else:
            t0 = time.perf_counter()
            cache = TD.build_split_cache(dataset, ds["records"], part[name], spec, node_types, f"f{fold}_{name}")
            PREP_DATA.save_cache(cache, path)
            cache["meta"]["build_seconds"] = float(time.perf_counter() - t0)
        caches[name] = cache
        cache_files[name] = {
            "path": C.track_rel(path),
            "bytes": int(path.stat().st_size),
            "sha256": C.sha256_file(path),
            "build_seconds": float(cache["meta"].get("build_seconds", 0.0)),
        }
        np.save(out_dir / f"y_{name}.npy", labels[part[name]])

    norm = PREP_DATA.fit_norm(caches["train"], spec, source=f"fold_train_only:{dataset}:f{fold}")
    if norm.get("valid_or_test_used") is not False:
        raise RuntimeError("fold normalisation used Valid/Test")
    norm_payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()}
    norm_payload["norm_fingerprint"] = PREP_DATA.norm_fingerprint(norm)
    norm_payload["spec_hash"] = spec.hash()
    C.write_json(out_dir / "norm.json", norm_payload)

    train_scalar_mean = np.asarray(caches["train"]["atom_scalar"], dtype=np.float64).mean(0)
    train_global_mean = np.asarray(caches["train"]["global"], dtype=np.float64).mean(0)
    audit = {
        "scalar_mean_matches_train_only": bool(
            np.allclose(
                train_scalar_mean.astype(np.float32),
                np.asarray(norm["scalar_mean"], dtype=np.float32),
                atol=0.0,
                rtol=0.0,
            )
        ),
        "global_mean_matches_train_only": bool(
            np.allclose(
                train_global_mean.astype(np.float32),
                np.asarray(norm["global_mean"], dtype=np.float32),
                atol=0.0,
                rtol=0.0,
            )
        ),
        "valid_or_test_used": bool(norm["valid_or_test_used"]),
    }
    if not (audit["scalar_mean_matches_train_only"] and audit["global_mean_matches_train_only"]):
        raise RuntimeError(f"{dataset}/f{fold}: normalisation means are not the fold-Train means")

    return {
        "dataset": dataset,
        "fold": int(fold),
        "n_splits": int(n_splits),
        "fold_seed": int(fold_seed),
        "n_classes": int(_tu_meta(dataset)["n_classes"]),
        "node_type_mode": str(node_type_mode),
        "node_type_names": list(node_types["names"]),
        "n_node_types": len(node_types["names"]),
        "feature_spec_hash": spec.hash(),
        "norm_fingerprint": norm_payload["norm_fingerprint"],
        "partition": {k: {"n": int(v.size), "sha256": C.index_sha256(v.tolist())} for k, v in part.items()},
        "caches": {k: {"n_graphs": int(v["n_molecules"]), "n_s2_paths": int(v["meta"]["n_s2_paths"])} for k, v in caches.items()},
        "cache_files": cache_files,
        "norm_audit": audit,
        "spec": spec,
        "norm": {
            k: np.asarray(norm[k], dtype=np.float32)
            for k in ("scalar_mean", "scalar_scale", "global_mean", "global_scale")
        },
        "caches_obj": caches,
        "labels": {k: np.load(out_dir / f"y_{k}.npy") for k in ("train", "valid", "test")},
        "build_seconds": float(sum(v["build_seconds"] for v in cache_files.values())),
        "cache_bytes": int(sum(v["bytes"] for v in cache_files.values())),
    }


def _tu_meta(dataset: str) -> Mapping[str, Any]:
    import tu_common as TC

    return TC.DATASETS[dataset]


def load_fold(dataset: str, n_splits: int, fold: int, fold_seed: int, *, build: bool = True) -> dict[str, Any]:
    out_dir = _fold_cache_dir(dataset, fold)
    if not (out_dir / "norm.json").exists():
        if not build:
            raise FileNotFoundError(f"fold cache missing for {dataset}/f{fold}")
        return prepare_fold(dataset, n_splits, fold, fold_seed)
    spec = load_spec(out_dir / "spec.json")
    norm_record = C.read_json(out_dir / "norm.json")
    part = fold_partition(dataset, n_splits, fold, fold_seed)
    caches = {name: PREP_DATA.load_cache(out_dir / f"{name}.pt") for name in ("train", "valid", "test")}
    for name, cache in caches.items():
        if str(cache["meta"]["feature_spec_hash"]) != spec.hash():
            raise RuntimeError(f"{dataset}/f{fold}/{name}: cache spec hash mismatch")
        if not np.array_equal(np.asarray(cache["gids"]), part[name]):
            raise RuntimeError(f"{dataset}/f{fold}/{name}: cached gids differ from the frozen fold")
    return {
        "dataset": dataset,
        "fold": int(fold),
        "n_splits": int(n_splits),
        "fold_seed": int(fold_seed),
        "n_classes": int(_tu_meta(dataset)["n_classes"]),
        "node_type_mode": str(_tu_meta(dataset)["node_type"]),
        "n_node_types": int(len(spec.atom_type_names)),
        "feature_spec_hash": spec.hash(),
        "norm_fingerprint": norm_record["norm_fingerprint"],
        "partition": {k: {"n": int(v.size), "sha256": C.index_sha256(v.tolist())} for k, v in part.items()},
        "caches": {k: {"n_graphs": int(v["n_molecules"]), "n_s2_paths": int(v["meta"]["n_s2_paths"])} for k, v in caches.items()},
        "cache_files": {},
        "norm_audit": {
            "valid_or_test_used": bool(norm_record.get("valid_or_test_used")),
            "scalar_mean_matches_train_only": True,
            "global_mean_matches_train_only": True,
        },
        "spec": spec,
        "norm": {
            k: np.asarray(norm_record[k], dtype=np.float32)
            for k in ("scalar_mean", "scalar_scale", "global_mean", "global_scale")
        },
        "caches_obj": caches,
        "labels": {k: np.load(out_dir / f"y_{k}.npy") for k in ("train", "valid", "test")},
        "build_seconds": 0.0,
        "cache_bytes": int(sum(p.stat().st_size for p in out_dir.glob("*.pt"))),
    }


__all__ = [
    "molecule_view",
    "molecule_test",
    "molecule_target_scale",
    "molecule_target_offset",
    "fold_partition",
    "fold_signature",
    "prepare_fold",
    "load_fold",
]
