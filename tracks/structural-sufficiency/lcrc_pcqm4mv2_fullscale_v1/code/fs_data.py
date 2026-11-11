"""Full-Train task assembly for the fullscale round.

Everything data-defining is inherited read-only: the ``lipo_v1`` spec, the
feature/label normalisation, the official evaluator, the Validation cache, the
20k monitor positions and the seed-42 Train permutation.  This module adds the
full 3,378,606-row cache, checks the row plan against the official split and
hand-builds the task dictionary that ``pq_train`` / ``fs_train`` consume.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fs_common as C

FULL_CACHE_NAME = "pcqm_train_full"
FULL_MANIFEST = C.ANALYSIS / "fs_full_cache_manifest.json"


# ---------------------------------------------------------------------------
# cache loading
# ---------------------------------------------------------------------------


def load_cache(path: Path | str, *, with_smiles: bool = False) -> dict[str, Any]:
    """Memory-mapped cache; SMILES/ids only when asked for (they are big)."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    cache: dict[str, Any] = {
        "path": str(path),
        "meta": meta,
        "n_molecules": int(meta["n_molecules"]),
        "split": str(meta["split"]),
    }
    for name in meta["fields"]:
        cache[name] = np.load(path / f"{name}.npy", mmap_mode="r")
    cache["gids"] = np.load(path / "gids.npy", mmap_mode="r")
    if with_smiles:
        with open(path / "molecule_ids.jsonl", encoding="utf-8") as fh:
            cache["molecule_ids"] = [json.loads(line) for line in fh]
        with open(path / "smiles.jsonl", encoding="utf-8") as fh:
            cache["smiles"] = [json.loads(line) for line in fh]
    return cache


# ---------------------------------------------------------------------------
# official split identity (read without rdkit; verified in the build stage)
# ---------------------------------------------------------------------------


def load_split_dict(*, verify_with_ogb: bool = False) -> dict[str, Any]:
    """The official split dict, its file hash and the row-plan evidence."""
    import torch

    split_path = C.PQ.PCQM_ROOT / "split_dict.pt"
    if not split_path.is_file():
        raise FileNotFoundError(f"official split dict missing: {split_path}")
    raw = torch.load(split_path, map_location="cpu", weights_only=False)
    split = {str(k): np.asarray(v, dtype=np.int64).reshape(-1) for k, v in raw.items()}
    if set(split) != set(C.PQ.SPLIT_KEYS):
        raise RuntimeError(f"split keys {sorted(split)} != {sorted(C.PQ.SPLIT_KEYS)}")
    evidence = {}
    for key, rows in split.items():
        evidence[key] = {
            "n": int(rows.size),
            "min": int(rows.min()),
            "max": int(rows.max()),
            "unique": int(np.unique(rows).size),
            "sorted_ascending": bool(np.all(np.diff(rows) > 0)),
            "index_sha256": C.index_hash(rows),
        }
    union = np.sort(np.concatenate([split[k] for k in C.PQ.SPLIT_KEYS]))
    if not np.array_equal(union, np.arange(union.size)):
        raise RuntimeError("official splits do not partition the dataset row ids")
    payload = {
        "split_dict": {
            "path": C.track_rel(split_path),
            "bytes": int(split_path.stat().st_size),
            "sha256": C.sha256_file(split_path),
        },
        "splits": evidence,
        "total_rows": int(union.size),
        "partition_ok": True,
        "n_expected": {
            "train": C.TRAIN_ROWS,
            "valid": C.VALID_ROWS,
            "test-dev": C.TEST_DEV_ROWS,
            "test-challenge": C.TEST_CHALLENGE_ROWS,
        },
        "split": split,
    }
    if verify_with_ogb:
        info = C.PQ_DATA.load_official()
        ogb_split = info["split"]
        for key in C.PQ.SPLIT_KEYS:
            if not np.array_equal(np.asarray(ogb_split[key]), split[key]):
                raise RuntimeError(f"split_dict.pt disagrees with get_idx_split() for {key}")
        payload["ogb_get_idx_split_matches_split_dict"] = True
        payload["ogb_split_evidence"] = info["split_evidence"]
        payload["dataset_files"] = info["files"]
        payload["dataset_n_total"] = int(info["n"])
        payload["release_files"] = info["release_files"]
    payload["size_checks"] = {
        key: int(split[key].size) == int(want)
        for key, want in payload["n_expected"].items()
    }
    payload["status"] = "OK" if all(payload["size_checks"].values()) else "FAILED"
    return payload


# ---------------------------------------------------------------------------
# full Train row plan (the seed-42 permutation, read-only)
# ---------------------------------------------------------------------------


def full_row_plan(*, split: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate ``pcqm_permutation.npy`` as the full official Train row order."""
    if split is None:
        split = load_split_dict()["split"]
    train = np.asarray(split["train"], dtype=np.int64)
    perm_path = C.PARENT_CACHE / "pcqm_permutation.npy"
    subset_path = C.PARENT_CACHE / "pcqm_subset_rows.npy"
    rows = np.load(perm_path).astype(np.int64, copy=False)
    subset = np.load(subset_path).astype(np.int64, copy=False)
    if rows.size != C.TRAIN_ROWS:
        raise RuntimeError(f"permutation has {rows.size} rows != official Train {C.TRAIN_ROWS}")
    checks = {
        "size_is_official_train": bool(rows.size == int(train.size)),
        "unique_rows": bool(np.unique(rows).size == rows.size),
        "all_rows_in_official_train": bool(np.isin(rows, train).all()),
        "set_equals_official_train": bool(
            np.array_equal(np.sort(rows), np.sort(train))
        ),
        "prefix_equals_hist_1m_subset": bool(np.array_equal(rows[: subset.size], subset)),
        "no_valid_overlap": int(
            np.intersect1d(rows, np.asarray(split["valid"], dtype=np.int64), assume_unique=False).size
        )
        == 0,
        "no_test_dev_overlap": int(
            np.intersect1d(rows, np.asarray(split["test-dev"], dtype=np.int64), assume_unique=False).size
        )
        == 0,
        "no_test_challenge_overlap": int(
            np.intersect1d(
                rows, np.asarray(split["test-challenge"], dtype=np.int64), assume_unique=False
            ).size
        )
        == 0,
        "subset_prefix_sha_matches_parent_plan": bool(
            C.index_hash(subset)
            == C.read_json(C.PARENT_ANALYSIS / "pcqm_subset_plan.json")["subsets"]["PCQM_W2_1M"]["rows_sha256"]
        ),
    }
    failed = [k for k, v in checks.items() if not bool(v)]
    if failed:
        raise RuntimeError(f"full Train row plan failed: {failed}")
    return {
        "permutation_file": C.track_rel(perm_path),
        "permutation_sha256": C.sha256_file(perm_path),
        "subset_rows_file": C.track_rel(subset_path),
        "subset_rows_sha256": C.sha256_file(subset_path),
        "n_rows": int(rows.size),
        "n_unique": int(np.unique(rows).size),
        "rows_sha256": C.index_hash(rows),
        "first_row": int(rows[0]),
        "last_row": int(rows[-1]),
        "hist_1m_rows_sha256": C.index_hash(rows[: subset.size]),
        "checks": checks,
        "status": "OK",
        "_rows": rows,
    }


# ---------------------------------------------------------------------------
# task assembly
# ---------------------------------------------------------------------------


def validate_full_cache_identity(cache: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Cache row order and identity vs the full Train plan and the spec."""
    spec_hash = str(cache["meta"].get("feature_spec_hash", ""))
    spec = C.PQ_DATA.FS.load_spec(C.PARENT_ANALYSIS / "pcqm_spec.json")
    gids = np.asarray(cache["gids"], dtype=np.int64)
    plan_rows = np.asarray(plan["_rows"], dtype=np.int64)
    checks = {
        "n_molecules_is_full_train": int(cache["n_molecules"]) == C.TRAIN_ROWS,
        "gids_equal_full_row_plan": bool(np.array_equal(gids, plan_rows)),
        "spec_hash_matches_parent_spec": bool(spec_hash == spec.hash()),
        "labels_present_false": not bool(cache["meta"].get("labels_present", True)),
        "uses_3d_false": not bool(cache["meta"].get("uses_3d_geometry", True)),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"full cache identity failed: {failed}")
    return {"checks": checks, "status": "OK", "gids_sha256": C.index_hash(gids)}


def load_task(run: str, *, verify_hashes: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Assemble the full-Train task for one formal run."""
    if run not in C.RUNS_TABLE:
        raise KeyError(run)
    spec_dict = C.RUNS_TABLE[run]
    t0 = time.perf_counter()
    spec = C.PQ_DATA.FS.load_spec(C.PARENT_ANALYSIS / "pcqm_spec.json")
    feature_norm = C.read_json(C.PARENT_ANALYSIS / "pcqm_feature_norm.json")
    label_norm = C.read_json(C.PARENT_ANALYSIS / "pcqm_label_norm.json")
    monitor_payload = C.read_json(C.PARENT_ANALYSIS / "pcqm_monitor_plan.json")
    label_artifact = C.read_json(C.PARENT_ANALYSIS / "pcqm_label_artifact.json")
    subset_plan_payload = C.read_json(C.PARENT_ANALYSIS / "pcqm_subset_plan.json")

    split_payload = load_split_dict()
    split = split_payload["split"]
    plan = full_row_plan(split=split)

    full_dir = C.CACHE / FULL_CACHE_NAME
    if not (full_dir / "meta.json").is_file():
        raise FileNotFoundError(f"full Train cache missing: {full_dir}")
    if not FULL_MANIFEST.is_file():
        raise FileNotFoundError(f"full Train cache manifest missing: {FULL_MANIFEST}")
    manifest = C.read_json(FULL_MANIFEST)
    if manifest.get("status") != "OK":
        raise RuntimeError(f"full cache manifest status {manifest.get('status')} != OK")

    train_cache = load_cache(full_dir, with_smiles=False)
    valid_cache = load_cache(C.PARENT_CACHE / "pcqm_valid", with_smiles=False)
    identity = validate_full_cache_identity(train_cache, plan)
    if int(valid_cache["n_molecules"]) != len(np.asarray(split["valid"])):
        raise RuntimeError("Validation cache size does not match the official Valid split")

    labels = np.load(C.PARENT_CACHE / "pcqm_labels_trainvalid.npy").astype(np.float64)
    train_gids = np.asarray(train_cache["gids"], dtype=np.int64)
    valid_gids = np.asarray(valid_cache["gids"], dtype=np.int64)
    if C.sha256_file(C.PARENT_CACHE / "pcqm_labels_trainvalid.npy") != label_artifact["artifact_sha256"]:
        raise RuntimeError("label artifact hash does not match its ledger entry")
    if not np.array_equal(valid_gids, np.asarray(split["valid"], dtype=np.int64)):
        raise RuntimeError("Validation cache gids are not the official Valid rows in order")

    if verify_hashes:
        if C.sha256_tree(full_dir) != manifest["cache_sha256"]:
            raise RuntimeError("full Train cache tree hash does not match its manifest")
        if C.sha256_tree(C.PARENT_CACHE / "pcqm_valid") != manifest["valid_tree_sha256"]:
            raise RuntimeError("Validation cache tree hash changed")

    train_y_eV = labels[train_gids]
    if not np.isfinite(train_y_eV).all():
        raise RuntimeError("non-finite Train label on a full-Train row")
    valid_y_eV = labels[valid_gids]
    if not np.isfinite(valid_y_eV).all():
        raise RuntimeError("non-finite Validation label")
    train_y_z = C.PQ_DATA.to_z(train_y_eV, label_norm).astype(np.float32)

    monitor_rows = np.asarray(monitor_payload["positions"], dtype=np.int64)
    if int(monitor_rows.max()) >= int(train_cache["n_molecules"]):
        raise RuntimeError("monitor rows outside the full Train rows")
    monitor_y_eV = train_y_eV[monitor_rows]

    norm = {
        "global_mean": np.asarray(feature_norm["global_mean_list"], dtype=np.float32),
        "global_scale": np.asarray(feature_norm["global_scale_list"], dtype=np.float32),
        "scalar_mean": np.asarray(feature_norm["scalar_mean_list"], dtype=np.float32),
        "scalar_scale": np.asarray(feature_norm["scalar_scale_list"], dtype=np.float32),
    }
    if C.PQ_DATA.PREP_DATA.norm_fingerprint({**feature_norm, **norm}) != feature_norm["norm_fingerprint"]:
        raise RuntimeError("feature normaliser arrays do not match their fingerprint")

    n_rows = int(train_cache["n_molecules"])
    if n_rows != C.TRAIN_ROWS:
        raise RuntimeError("run must use the full official Train set")
    n_intersect_valid = int(np.intersect1d(train_gids, valid_gids, assume_unique=False).size)
    if n_intersect_valid:
        raise RuntimeError("a Validation row is present in the Train rows")
    hidden_hits = {
        key: int(np.intersect1d(train_gids, np.asarray(split[key], dtype=np.int64), assume_unique=False).size)
        for key in ("test-dev", "test-challenge")
    }
    if any(hidden_hits.values()):
        raise RuntimeError(f"hidden Test rows present in the Train rows: {hidden_hits}")

    orders = C.PQ_DATA.epoch_orders(n_rows, seed=C.ORDER_SEED, epochs=int(spec_dict["epochs"]))
    task = {
        "run": run,
        "spec": spec,
        "norm": norm,
        "feature_norm": feature_norm,
        "label_norm": label_norm,
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "valid_rows": np.arange(int(valid_cache["n_molecules"]), dtype=np.int64),
        "valid_y_eV": valid_y_eV,
        "train_y_eV": train_y_eV,
        "train_y_z": train_y_z,
        "monitor_rows": monitor_rows,
        "monitor_y_eV": monitor_y_eV,
        "monitor_plan": {k: v for k, v in monitor_payload.items() if not k.startswith("_")},
        "n_rows": n_rows,
        "steps_epoch": C.steps_per_epoch(n_rows),
        "steps_total": C.total_steps(n_rows),
        "seed": int(spec_dict["seed"]),
        "scale": int(spec_dict["scale"]),
        "model": str(spec_dict["model"]),
        "model_name": str(spec_dict["model_name"]),
        "expected_parameters": int(spec_dict["expected_parameters"]),
        "widths": dict(C.WIDTHS[int(spec_dict["scale"])]),
        "orders": orders,
        "evaluator": C.PQ_DATA._evaluator(),
        "labels_artifact": label_artifact,
        "subset_plan": subset_plan_payload,
        "row_plan": {k: v for k, v in plan.items() if not k.startswith("_")},
        "split_identity": split_payload,
        "cache_manifest": {
            "round_id": C.ROUND_ID,
            "full_cache": {k: v for k, v in manifest.items() if k != "shards"},
            "valid_cache": {
                "path": C.track_rel(C.PARENT_CACHE / "pcqm_valid"),
                "cache_sha256": manifest["valid_tree_sha256"],
                "n_molecules": int(valid_cache["n_molecules"]),
            },
            "train1m_control_sha256": manifest["prefix_control"]["cache_sha256"],
        },
        "data_isolation": {
            "train_rows": int(n_rows),
            "train_rows_last_gid": int(train_gids[-1]),
            "valid_rows": int(valid_gids.size),
            "train_valid_overlap": int(n_intersect_valid),
            "hidden_split_overlap": hidden_hits,
            "train_is_full_official_train": True,
            "train_is_prefix_of_frozen_1m_subset": bool(
                np.array_equal(train_gids[:1_000_000], np.load(C.PARENT_CACHE / "pcqm_subset_rows.npy"))
            ),
            "subset_plan_status": subset_plan_payload["status"],
            "hidden_split_rows_touched": 0,
            "monitor_rows_are_train_rows": bool(monitor_rows.max() < n_rows),
            "full_cache_identity": identity,
        },
        "source_files": {
            "spec": C.track_rel(C.PARENT_ANALYSIS / "pcqm_spec.json"),
            "feature_norm": C.track_rel(C.PARENT_ANALYSIS / "pcqm_feature_norm.json"),
            "label_norm": C.track_rel(C.PARENT_ANALYSIS / "pcqm_label_norm.json"),
            "monitor_plan": C.track_rel(C.PARENT_ANALYSIS / "pcqm_monitor_plan.json"),
            "label_artifact": C.track_rel(C.PARENT_ANALYSIS / "pcqm_label_artifact.json"),
            "valid_cache": C.track_rel(C.PARENT_CACHE / "pcqm_valid"),
            "full_cache": C.track_rel(full_dir),
            "full_cache_manifest": C.track_rel(FULL_MANIFEST),
            "row_plan": C.track_rel(C.PARENT_CACHE / "pcqm_permutation.npy"),
        },
        "load_seconds": float(time.perf_counter() - t0),
    }
    task["spec_hash"] = spec.hash()
    if verbose:
        C.log(
            f"[task:{run}] rows={n_rows} steps_epoch={task['steps_epoch']} "
            f"steps_total={task['steps_total']} valid={valid_gids.size} "
            f"monitor={monitor_rows.size} load={task['load_seconds']:.1f}s"
        )
    return task


def smiles_of(cache: Mapping[str, Any], rows: Sequence[int]) -> list[str]:
    """SMILES of cache rows (requires a cache loaded with ``with_smiles=True``)."""
    smiles = cache.get("smiles")
    if smiles is None or len(smiles) != int(cache["n_molecules"]):
        raise RuntimeError("cache was not loaded with SMILES")
    return [str(smiles[int(r)]) for r in rows]


__all__ = [
    "FULL_CACHE_NAME",
    "FULL_MANIFEST",
    "full_row_plan",
    "load_cache",
    "load_split_dict",
    "load_task",
    "smiles_of",
    "validate_full_cache_identity",
]
