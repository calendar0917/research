"""Read-only access to the frozen PCBA assets and the paired batch order.

The split rows are the official OGB row ids, not contiguous ranges.  Every
check below is a refusal path: if the cache, split, labels, feature spec,
normalisation or Test isolation differs from the historical epoch-29 assets,
the round stops instead of training on something else.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lr_common as C
import pc_data as DATA
import bp_data as D
from feature_spec import FeatureSpec, load_spec


@dataclass
class Task:
    spec: FeatureSpec
    cache: Mapping[str, Any]
    labels: np.ndarray
    split: dict[str, np.ndarray]
    train_rows: np.ndarray
    valid_rows: np.ndarray
    test_rows: np.ndarray
    norm: dict[str, Any]
    label_artifact_sha256: str
    split_partition_sha256: str
    cache_identity: dict[str, Any]
    data_audit: dict[str, Any]
    load_seconds: float
    cache_path: Path
    labels_path: Path


def _cache_stats(path: Path) -> dict[str, Any]:
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    return {
        "file_count": len(files),
        "bytes": int(sum(p.stat().st_size for p in files)),
        "names": [p.name for p in files],
    }


def _meta_counts(meta: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "n_molecules",
        "n_atoms_total",
        "n_bonds_total",
        "n_s1_codes",
        "n_s2_paths",
        "feature_spec_hash",
        "split",
        "format",
    ):
        if key in meta:
            out[key] = meta[key]
    return out


def load_task(
    cache_path: Path | str = C.CACHE_PATH,
    labels_path: Path | str = C.LABELS_PATH,
    spec_path: Path | str = C.SPEC_PATH,
    split_path: Path | str = C.SPLIT_PATH,
    *,
    verify_label_sha: bool = True,
) -> Task:
    t0 = time.perf_counter()
    cache_path = Path(cache_path)
    labels_path = Path(labels_path)
    spec = load_spec(spec_path)
    split_payload = C.read_json(split_path)
    split = {
        k: np.asarray(split_payload["rows"][k], dtype=np.int64)
        for k in ("train", "valid", "test")
    }
    if spec.hash() != C.HIST["feature_spec_hash"]:
        raise RuntimeError(f"feature spec hash {spec.hash()} != historical {C.HIST['feature_spec_hash']}")
    expected_sizes = {"train": 350_343, "valid": 43_793, "test": 43_793}
    got_sizes = {k: int(v.size) for k, v in split.items()}
    if got_sizes != expected_sizes:
        raise RuntimeError(f"split sizes {got_sizes} != official {expected_sizes}")
    if str(split_payload.get("partition_sha256")) != C.HIST["split_partition_sha256"]:
        raise RuntimeError(
            f"split partition sha {split_payload.get('partition_sha256')} != frozen "
            f"{C.HIST['split_partition_sha256']}"
        )

    cache = DATA.load_cache(cache_path)
    meta = cache["meta"]
    if int(cache["n_molecules"]) != C.N_MOLECULES:
        raise RuntimeError(f"cache n_molecules {cache['n_molecules']} != {C.N_MOLECULES}")
    if str(meta.get("feature_spec_hash")) != spec.hash():
        raise RuntimeError("cache feature-spec hash does not match the frozen spec")
    if not np.array_equal(np.asarray(cache["gids"]), np.arange(C.N_MOLECULES)):
        raise RuntimeError("cache rows are not the official row order")
    for key, expected in (
        ("n_atoms_total", C.HIST["n_atoms_total"]),
        ("n_bonds_total", C.HIST["n_bonds_total"]),
        ("n_s2_paths", C.HIST["n_s2_paths"]),
    ):
        if key in meta and int(meta[key]) != int(expected):
            raise RuntimeError(f"cache meta {key}={meta[key]} != frozen {expected}")

    labels = np.load(labels_path)
    if labels.ndim != 2 or labels.shape != (C.N_MOLECULES, C.N_TASKS):
        raise RuntimeError(f"label artifact {labels.shape} != {(C.N_MOLECULES, C.N_TASKS)}")
    if labels.dtype != np.float32:
        raise RuntimeError(f"label artifact dtype {labels.dtype} != float32")

    # Test isolation is a gate, not a convention.
    if np.isfinite(labels[split["test"]]).any():
        raise RuntimeError("label artifact exposes official Test targets; refusing to continue")

    train_rows, valid_rows, test_rows = split["train"], split["valid"], split["test"]
    if np.intersect1d(train_rows, valid_rows).size:
        raise RuntimeError("train/valid split rows overlap")
    if np.intersect1d(train_rows, test_rows).size or np.intersect1d(valid_rows, test_rows).size:
        raise RuntimeError("official Test rows overlap train/valid")

    norm = D.fit_norm_rows(cache, spec, train_rows, source="pcba_official_train_only")
    if norm["norm_fingerprint"] != C.HIST["norm_fingerprint"]:
        raise RuntimeError(
            f"Train-only norm fingerprint {norm['norm_fingerprint']} != historical "
            f"{C.HIST['norm_fingerprint']}"
        )

    finite_train = np.isfinite(labels[train_rows])
    finite_valid = np.isfinite(labels[valid_rows])
    train_values = labels[train_rows][finite_train]
    valid_values = labels[valid_rows][finite_valid]
    for name, values in (("train", train_values), ("valid", valid_values)):
        if not np.all((values == 0.0) | (values == 1.0)):
            raise RuntimeError(f"{name} labels are not binary 0/1")
    if int(np.isfinite(labels).sum()) != int(finite_train.sum() + finite_valid.sum()):
        raise RuntimeError("finite labels exist outside the official train/valid rows")

    label_sha = C.sha256_file(labels_path) if verify_label_sha else ""
    if verify_label_sha and label_sha != C.HIST["label_artifact_sha256"]:
        raise RuntimeError(
            f"label artifact sha {label_sha} != historical {C.HIST['label_artifact_sha256']}"
        )

    manifest = C.read_json(C.CACHE_MANIFEST)
    merged = manifest.get("merged", {})
    stats = _cache_stats(cache_path)
    cache_identity = {
        "path": C.track_rel(cache_path),
        "format": "npy-directory-v1",
        "actual_file_count": stats["file_count"],
        "actual_bytes": stats["bytes"],
        "manifest_file_count": merged.get("cache_files"),
        "manifest_bytes": merged.get("cache_bytes"),
        "manifest_cache_sha256": merged.get("cache_sha256"),
        "manifest_feature_spec_hash": merged.get("feature_spec_hash"),
        "feature_spec_hash": spec.hash(),
        "full_tree_rehashed": False,
        "full_tree_rehash_note": (
            "not re-hashed in the training host: hashing 4.5 GB evicts the page cache and "
            "changes the measured mmap read pattern; manifest + file count/bytes + feature spec "
            "are verified instead"
        ),
        "meta": _meta_counts(meta),
    }
    if stats["file_count"] != merged.get("cache_files"):
        raise RuntimeError(f"cache file count {stats['file_count']} != manifest {merged.get('cache_files')}")
    if stats["bytes"] != merged.get("cache_bytes"):
        raise RuntimeError(f"cache bytes {stats['bytes']} != manifest {merged.get('cache_bytes')}")
    if merged.get("cache_sha256") != C.HIST["cache_sha256"]:
        raise RuntimeError("cache manifest sha256 != historical cache sha256")

    data_audit = {
        "dataset": C.DATASET,
        "n_molecules": int(labels.shape[0]),
        "n_tasks": int(labels.shape[1]),
        "split_sizes": got_sizes,
        "split_partition_sha256": str(split_payload.get("partition_sha256")),
        "feature_spec_hash": spec.hash(),
        "norm_fingerprint": norm["norm_fingerprint"],
        "norm_fit_split": norm.get("fit_split"),
        "norm_source": norm.get("source"),
        "norm_fit_rows": int(norm.get("fit_rows", -1)),
        "norm_fit_rows_sha256": norm.get("fit_rows_sha256"),
        "label_artifact_sha256": label_sha,
        "label_artifact_expected_sha256": C.HIST["label_artifact_sha256"],
        "test_rows_all_nan": True,
        "test_rows_finite_entries": 0,
        "train_observed_entries": int(finite_train.sum()),
        "valid_observed_entries": int(finite_valid.sum()),
        "train_total_entries": int(train_rows.size * C.N_TASKS),
        "valid_total_entries": int(valid_rows.size * C.N_TASKS),
        "train_positive_entries": int((train_values == 1.0).sum()),
        "valid_positive_entries": int((valid_values == 1.0).sum()),
        "train_negative_entries": int((train_values == 0.0).sum()),
        "valid_negative_entries": int((valid_values == 0.0).sum()),
        "labels_binary": True,
        "gids_are_official_row_order": True,
        "labels_are_nan_not_zero": "NaN entries are excluded from loss; they are never filled with 0",
        "cache_identity": cache_identity,
    }
    return Task(
        spec=spec,
        cache=cache,
        labels=labels,
        split=split,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        norm=norm,
        label_artifact_sha256=label_sha,
        split_partition_sha256=str(split_payload.get("partition_sha256")),
        cache_identity=cache_identity,
        data_audit=data_audit,
        load_seconds=float(time.perf_counter() - t0),
        cache_path=cache_path,
        labels_path=labels_path,
    )


def make_order_plan(
    train_rows: Sequence[int],
    *,
    batch_size: int = C.BATCH_SIZE,
    seed: int = C.SEED,
    epochs: int = C.TOTAL_EPOCHS,
) -> list[dict[str, Any]]:
    """Exact continuation of ``pc_train._batches``: one CPU generator, 12 epochs.

    ``pc_train._batches`` calls ``torch.randperm(len(rows), generator=generator)``
    once per epoch.  The generator is created with ``manual_seed(0)`` and is
    independent of the model/dropout RNG, so this plan is a pure function of the
    official train rows.
    """
    rows = np.asarray(train_rows, dtype=np.int64)
    generator = torch.Generator().manual_seed(int(seed))
    plan: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(len(rows), generator=generator).numpy()
        shuffled = rows[perm]
        batches = [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]
        plan.append(
            {
                "epoch": int(epoch),
                "rows": shuffled,
                "batches": batches,
                "row_order_sha256": C.sha256_int64(shuffled),
                "batch_plan_sha256": C.sha256_batches(batches),
                "n_batches": len(batches),
                "first_row": int(shuffled[0]),
                "last_row": int(shuffled[-1]),
                "first_batch_rows": [int(x) for x in batches[0]],
                "last_batch_size": int(len(batches[-1])),
            }
        )
    generator_state = generator.get_state()
    plan_sha = C.sha256_bytes(
        json.dumps([p["row_order_sha256"] for p in plan], sort_keys=True).encode()
    )
    for record in plan:
        record["generator_state_sha256"] = C.sha256_bytes(generator_state.numpy().tobytes())
        record["global_plan_sha256"] = plan_sha
    return plan


def order_summary(plan: Sequence[Mapping[str, Any]], *, n_train_rows: int, seed: int = C.SEED) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "batch_size": C.BATCH_SIZE,
        "n_train_rows": int(n_train_rows),
        "steps_per_epoch": int((n_train_rows + C.BATCH_SIZE - 1) // C.BATCH_SIZE),
        "epochs": len(plan),
        "source": "torch.Generator().manual_seed(0) + torch.randperm(len(train_rows)) per epoch",
        "global_plan_sha256": plan[0]["global_plan_sha256"] if plan else "",
        "generator_state_sha256": plan[0]["generator_state_sha256"] if plan else "",
        "epochs_detail": [
            {
                "epoch": int(p["epoch"]),
                "n_batches": int(p["n_batches"]),
                "row_order_sha256": p["row_order_sha256"],
                "batch_plan_sha256": p["batch_plan_sha256"],
                "first_row": int(p["first_row"]),
                "last_row": int(p["last_row"]),
                "first_batch_rows": list(p["first_batch_rows"]),
                "last_batch_size": int(p["last_batch_size"]),
            }
            for p in plan
        ],
    }


def assert_order_isolation(
    plan: Sequence[Mapping[str, Any]],
    train_rows: Sequence[int],
    valid_rows: Sequence[int],
    test_rows: Sequence[int],
) -> dict[str, Any]:
    """Every epoch must be a permutation of train rows and contain no valid/test row."""
    train_sorted = np.sort(np.asarray(train_rows, dtype=np.int64))
    n = C.N_MOLECULES
    valid_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    valid_mask[np.asarray(valid_rows, dtype=np.int64)] = True
    test_mask[np.asarray(test_rows, dtype=np.int64)] = True
    checks = []
    for record in plan:
        rows = np.asarray(record["rows"], dtype=np.int64)
        checks.append(
            {
                "epoch": int(record["epoch"]),
                "is_train_permutation": bool(np.array_equal(np.sort(rows), train_sorted)),
                "n_rows": int(rows.size),
                "n_unique_rows": int(np.unique(rows).size),
                "valid_rows_in_epoch": int(valid_mask[rows].sum()),
                "test_rows_in_epoch": int(test_mask[rows].sum()),
            }
        )
    passed = all(
        c["is_train_permutation"]
        and c["n_rows"] == int(train_sorted.size)
        and c["n_unique_rows"] == int(train_sorted.size)
        and c["valid_rows_in_epoch"] == 0
        and c["test_rows_in_epoch"] == 0
        for c in checks
    )
    return {
        "passed": bool(passed),
        "epochs_checked": len(checks),
        "train_rows": int(train_sorted.size),
        "checks": checks,
    }


__all__ = [
    "Task",
    "assert_order_isolation",
    "load_task",
    "make_order_plan",
    "order_summary",
]
