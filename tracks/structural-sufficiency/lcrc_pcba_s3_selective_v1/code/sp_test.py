"""Official Test prediction freeze and the single post-freeze scoring pass.

Two separate subcommands enforce the ordering, exactly as the continuation round
did (``cc_test.py``):

* ``freeze`` loads the three pre-registered checkpoints (W2 best-Valid epoch 24,
  S3 best-Valid epoch 29, SP best-Valid epoch of this round), replays their
  official Validation AP, writes Test logits from the **label-free Test S3
  structure cache** and seals every hash.  It never opens an official Test
  label.
* ``score`` verifies the freeze seal, the selection manifest and every
  prediction SHA256, and only then reads the official OGB Test labels
  (``data/ogb/ogbg_molpcba/raw/graph-label.csv.gz``) through the **inherited**
  ``cc_data.read_official_test_labels`` gate, scoring with the official OGB
  evaluator plus an independent per-task AP recomputation.

The Test split is already exposed (``lcrc_pcba_continuation_recipe_v1``).  This
round's role for Test is ``EXPOSED_TEST_PERFORMANCE_BENCHMARK``: it is a
descriptive benchmark of a checkpoint selected on Validation only, never a blind
generalisation estimate and never a selection signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch  # noqa: E402

import sp_common as C  # noqa: E402
import sp_model as M  # noqa: E402
import bs_model as BSM  # noqa: E402
import s3_model as M3  # noqa: E402
import pc_metrics as MET  # noqa: E402

CC_CODE = C.REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1/code"
if str(CC_CODE) not in sys.path:
    sys.path.insert(0, str(CC_CODE))

import cc_common as CC  # noqa: E402  (the inherited freeze schema / relativiser)
import cc_data as CD  # noqa: E402  (the inherited official Test label gate)

PREDICTIONS = C.ANALYSIS / "test_predictions"
FREEZE_PATH = C.ANALYSIS / "test_prediction_manifest.json"
SELECTION_PATH = C.ANALYSIS / "selected_checkpoint.json"
SCORES_PATH = C.ANALYSIS / "test_scores.json"
PER_TASK_PATH = C.ANALYSIS / "per_task_valid_test.csv"
VALID_REPLAY_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np_sha(array: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def _save_npy_atomic(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(array, dtype=np.float32))
    os.replace(tmp, path)
    return C.sha256_file(path)


def _abs_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else C.REPO / path


def prediction_rows(*, batch_size: int = 256) -> np.ndarray:
    rows = np.asarray(C.load_task_gated().test_rows, dtype=np.int64)
    return rows


# ---------------------------------------------------------------------------
# model restore
# ---------------------------------------------------------------------------


def restore_model(entry: Mapping[str, Any], task: Any, device: torch.device):
    """Rebuild one frozen arm and load its checkpoint state (strict)."""
    path = _abs_path(entry["path"])
    if not path.is_file():
        raise RuntimeError(f"checkpoint missing: {path}")
    sha = C.sha256_file(path)
    if sha != entry["sha256"]:
        raise RuntimeError(f"{entry['label']} checkpoint sha {sha} != recorded {entry['sha256']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != entry["model"]:
        raise RuntimeError(f"{entry['label']} checkpoint model {payload.get('model')} != {entry['model']}")
    if int(payload.get("epoch", -1)) != int(entry["epoch"]):
        raise RuntimeError(f"{entry['label']} checkpoint epoch {payload.get('epoch')} != {entry['epoch']}")
    key = str(entry["model"])
    if key == "W2":
        model, audit = BSM.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device="cpu")
    elif key == "S3":
        model, audit = M3.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device="cpu")
    elif key == "SP":
        model, audit = M.build_model(task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device="cpu")
    else:  # pragma: no cover
        raise KeyError(key)
    state = payload["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{entry['label']} state mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model, payload, audit, sha


@torch.inference_mode()
def predict_rows(
    model: torch.nn.Module,
    task: Any,
    s3: Mapping[str, Any],
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Logits for ``rows``; identical batching for every arm."""
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    out: list[np.ndarray] = []
    for start in range(0, rows.size, int(batch_size)):
        chunk = rows[start : start + int(batch_size)]
        batch = C.collate_with_s3(task, s3, chunk.tolist())
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch)
        if logits.shape[1] != C.N_TASKS:
            raise RuntimeError(f"model returned {logits.shape[1]} outputs, expected {C.N_TASKS}")
        out.append(logits.detach().cpu().numpy().astype(np.float32))
    logits_np = np.concatenate(out, axis=0) if out else np.zeros((0, C.N_TASKS), np.float32)
    if logits_np.shape[0] != rows.size:
        raise RuntimeError(f"prediction rows {logits_np.shape[0]} != {rows.size}")
    if not np.isfinite(logits_np).all():
        raise RuntimeError("non-finite logits")
    return logits_np


def validation_replay(model: torch.nn.Module, task: Any, device: torch.device) -> dict[str, Any]:
    """Replay official Validation AP on the read-only Train+Valid S3 cache."""
    import pc_model as PC

    rows = np.asarray(task.valid_rows, dtype=np.int64)
    logits = predict_rows(model, task, task.s3, rows, device)
    probs = PC.sigmoid(logits)
    report = MET.validation_report(np.asarray(task.labels[rows]), probs)
    report.update(
        {
            "valid_logits_sha256": _np_sha(logits),
            "valid_bce": float(MET.masked_bce_numpy(logits, np.asarray(task.labels[rows]))),
            "evaluated_rows": int(rows.size),
        }
    )
    return report


# ---------------------------------------------------------------------------
# the pre-registered model list
# ---------------------------------------------------------------------------


def candidate_models(*, selection_path: Path = SELECTION_PATH) -> dict[str, dict[str, Any]]:
    """W2-24, S3-29 and the SP best-Valid checkpoint — one selection rule."""
    selection = C.read_json(selection_path)
    if selection.get("test_read") is not False:
        raise RuntimeError("selection manifest must declare test_read=false")
    sp = selection["models"]["SP"]
    if bool(selection.get("selection_uses_test")):
        raise RuntimeError("selection manifest claims test-based selection")
    return {
        "W2_best_valid_epoch24": {
            "label": "W2_best_valid_epoch24",
            "model": "W2",
            "epoch": int(C.W2["best_epoch"]),
            "path": C.track_rel(C.W2_CHECKPOINT_BEST),
            "sha256": C.W2["best_checkpoint_sha256"],
            "valid_ap": float(C.W2["best_valid_ap"]),
            "role": "historical best-Validation checkpoint (selected on Valid)",
        },
        "S3_best_valid_epoch29": {
            "label": "S3_best_valid_epoch29",
            "model": "S3",
            "epoch": int(C.S3["best_epoch"]),
            "path": C.track_rel(C.S3_CHECKPOINT_BEST),
            "sha256": C.S3["best_checkpoint_sha256"],
            "valid_ap": float(C.S3["best_valid_ap"]),
            "role": "historical best-Validation checkpoint (selected on Valid)",
        },
        "SP_best_valid_selected": {
            "label": "SP_best_valid_selected",
            "model": "SP",
            "epoch": int(sp["best_epoch"]),
            "path": sp["path"],
            "sha256": sp["sha256"],
            "valid_ap": float(sp["best_valid_ap"]),
            "role": "this round's best-Validation checkpoint (selected on Valid)",
        },
    }


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# label-free Test structure cache
# ---------------------------------------------------------------------------

TEST_CACHE_RECORD = C.ANALYSIS / "test_cache_build.json"


def run_build_cache(out_dir: Path, *, device: torch.device | None = None) -> dict[str, Any]:
    """Enumerate S3 three-hop paths for the official Test rows. Structure only.

    Labels are never touched: the labels file is not opened, and the resulting
    cache declares ``labels_present=false`` / ``training_scope=false``.  The
    frozen Train+Valid cache is opened read-only and re-hashed before/after.
    """
    del device  # CPU preprocessing; the enumeration is pure numpy
    task = C.load_task_gated()
    train_meta = C.read_json(Path(C.S3_TRAIN_CACHE_DIR) / "meta.json")
    train_sha_before = C.sha256_file(Path(C.S3_TRAIN_CACHE_DIR) / "meta.json")
    t0 = time.perf_counter()
    meta = C.build_s3_test_cache(task, out_dir)
    seconds = time.perf_counter() - t0
    train_sha_after = C.sha256_file(Path(C.S3_TRAIN_CACHE_DIR) / "meta.json")
    if train_sha_before != train_sha_after:
        raise RuntimeError("the frozen Train+Valid cache changed during the Test cache build")
    if train_meta.get("test_rows_present") not in (False, 0):
        raise RuntimeError("the frozen Train+Valid cache claims to contain Test rows")
    cache = C.load_s3_test_cache(out_dir)
    checks = {
        "rows_are_the_official_test_rows": bool(
            np.array_equal(np.sort(np.asarray(cache["rows"], np.int64)), np.sort(np.asarray(task.test_rows, np.int64)))
        ),
        "no_train_or_valid_row": bool(
            int(cache["row_index"][np.asarray(task.train_rows, np.int64)].max()) < 0
            and int(cache["row_index"][np.asarray(task.valid_rows, np.int64)].max()) < 0
        ),
        "labels_absent": bool(cache["meta"]["labels_present"] is False),
        "training_scope_false": bool(cache["meta"]["training_scope"] is False),
        "row_path_count_consistent": bool(
            int(cache["meta"]["n_paths"]) == int(np.asarray(cache["s3_row_paths"]).sum())
        ),
        "atom_bookkeeping_consistent": bool(
            int(cache["row_atom_ptr"][-1]) == int(np.asarray(cache["s3_atom_ptr"]).shape[0]) - 1
        ),
        "feature_spec_matches_frozen": bool(cache["meta"]["feature_spec_hash"] == C.SOURCE["feature_spec_hash"]),
        "split_partition_matches_frozen": bool(
            cache["meta"]["split_partition_sha256"] == C.SOURCE["split_partition_sha256"]
        ),
    }
    audit = {
        **{k: meta[k] for k in ("n_rows", "n_paths", "n_atoms", "dist_counts", "build_seconds", "field_sha256", "rows_sha256")},
        "checks": checks,
        "all_pass": bool(all(checks.values())),
    }
    if not audit["all_pass"]:
        raise RuntimeError(f"Test structure cache failed its invariants: {checks}")
    record = {
        "schema": "lcrc-pcba-s3-selective-test-cache-build-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "out_dir": C.track_rel(out_dir),
        "cpu_seconds": float(seconds),
        "scope": "official Test rows only",
        "labels_read": False,
        "training_cache_sha256_before": train_sha_before,
        "training_cache_sha256_after": train_sha_after,
        "training_cache_untouched": True,
        "training_cache_test_rows_present": bool(train_meta.get("test_rows_present")),
        "audit": audit,
    }
    C.write_json(TEST_CACHE_RECORD, record)
    C.log(
        f"[build-cache] {C.track_rel(out_dir)} rows={audit['n_rows']} paths={audit['n_paths']} "
        f"cpu_seconds={seconds:.1f} all_pass={audit['all_pass']}"
    )
    return record


def run_freeze(
    selection_path: Path,
    out_path: Path,
    predictions_dir: Path,
    device: torch.device,
) -> int:
    task = C.load_task_gated()
    task.s3_test = C.load_s3_test_cache()
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    if not np.array_equal(np.asarray(task.s3_test["rows"], dtype=np.int64), np.sort(test_rows)):
        raise RuntimeError("Test S3 structure cache row ids are not the official Test rows")
    if np.intersect1d(np.asarray(task.s3_test["rows"], np.int64), np.asarray(task.train_rows, np.int64)).size:
        raise RuntimeError("Test structure cache contains Train rows")
    models = candidate_models(selection_path=selection_path)

    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for label, entry in models.items():
        t0 = time.perf_counter()
        model, payload, audit, sha = restore_model(entry, task, device)
        t_load = time.perf_counter() - t0
        t1 = time.perf_counter()
        valid = validation_replay(model, task, device)
        t_valid = time.perf_counter() - t1
        delta = abs(float(valid["ap"]) - float(entry["valid_ap"]))
        if delta > VALID_REPLAY_TOLERANCE:
            raise RuntimeError(
                f"{label} Validation replay {valid['ap']} != recorded {entry['valid_ap']} (delta {delta})"
            )
        t2 = time.perf_counter()
        logits = predict_rows(model, task, task.s3_test, test_rows, device)
        t_test = time.perf_counter() - t2
        logits_path = predictions_dir / f"{label}.npy"
        logits_sha = _save_npy_atomic(logits_path, logits)
        records[label] = {
            "label": label,
            "model": entry["model"],
            "role": entry["role"],
            "epoch": int(entry["epoch"]),
            "checkpoint": entry["path"],
            "checkpoint_sha256": sha,
            "checkpoint_payload_epoch": int(payload.get("epoch", -1)),
            "checkpoint_global_step": int(payload.get("global_step", -1)),
            "checkpoint_selection": payload.get("selection_rule") or payload.get("valid_ap") is not None,
            "valid_ap_recorded": float(entry["valid_ap"]),
            "valid_ap_replay": float(valid["ap"]),
            "valid_ap_abs_delta": float(delta),
            "valid_bce_replay": float(valid["valid_bce"]),
            "valid_per_task_ap": valid["per_task_ap"],
            "valid_observed": valid["observed"],
            "valid_positive": valid["positive"],
            "valid_negative": valid["negative"],
            "valid_logits_sha256": valid["valid_logits_sha256"],
            "n_tasks_scored_valid": int(valid["n_tasks_scored"]),
            "test_logits_path": C.track_rel(logits_path),
            "test_logits_sha256": logits_sha,
            "test_logits_shape": list(logits.shape),
            "test_logits_dtype": "float32",
            "test_logits_min": float(logits.min()),
            "test_logits_max": float(logits.max()),
            "test_logits_mean": float(logits.mean()),
            "test_rows": int(logits.shape[0]),
            "test_n_tasks": int(logits.shape[1]),
            "replay_audit": {
                "parameters": int(audit["total_parameters"]),
                "parameter_check": audit.get("parameter_check"),
                "base_state_sha256": audit.get("base_state_sha256"),
                "model_state_sha256": audit.get("model_state_sha256"),
            },
            "load_seconds": float(t_load),
            "valid_seconds": float(t_valid),
            "test_predict_seconds": float(t_test),
        }
        del model
        C.log(
            f"[freeze] {label} valid {valid['ap']:.9f} (recorded {entry['valid_ap']:.9f}) "
            f"test logits {logits_sha[:12]}"
        )

    rows_path = predictions_dir / "test_row_ids.npy"
    tmp = rows_path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.sort(test_rows))
    os.replace(tmp, rows_path)
    rows_sha = C.sha256_file(rows_path)

    sorted_rows = np.sort(test_rows)
    for label, record in records.items():
        record["test_row_ids_sha256"] = C.sha256_bytes(
            np.ascontiguousarray(sorted_rows, dtype=np.int64).tobytes()
        )

    freeze: dict[str, Any] = {
        "schema": CC.SCHEMA_TEST_FREEZE,
        "round_id": C.ROUND_ID,
        "test_role": "EXPOSED_TEST_PERFORMANCE_BENCHMARK",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_labels_read": False,
        "test_labels_read_before_freeze": False,
        "selection_manifest": C.track_rel(selection_path),
        "selection_manifest_sha256": C.sha256_file(selection_path),
        "test_structure_cache": C.test_cache_identity(),
        "test_train_cache": C.s3_train_cache_identity(),
        "test_rows_path": C.track_rel(rows_path),
        "test_rows_sha256": rows_sha,
        "test_rows_npy_sha256": CC.sha256_int64(sorted_rows),
        "test_rows_int64_sha256": CC.sha256_int64(sorted_rows),
        "n_test_rows": int(sorted_rows.size),
        "n_tasks": int(C.N_TASKS),
        "prediction_batch_size": 256,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "environment": C.environment(),
        "logical_models": {
            label: {
                "prediction": label,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "valid_ap_replay": record["valid_ap_replay"],
            }
            for label, record in records.items()
        },
        "predictions": records,
        "unique_predictions": sorted(records),
        "aliases": {},
        "total_seconds": float(time.perf_counter() - started),
        "note": (
            "Test logits are frozen before any official Test label is opened. "
            "Only sp_test.py score reads raw/graph-label.csv.gz, through the inherited "
            "cc_data.read_official_test_labels gate."
        ),
    }
    freeze["seal_sha256"] = CD.freeze_seal(freeze)
    C.write_json(out_path, freeze)
    check = C.read_json(out_path)
    if check["seal_sha256"] != CD.freeze_seal(check):
        raise RuntimeError("written freeze manifest seal does not verify")
    C.log(f"[freeze] wrote {out_path} seal={freeze['seal_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _load_prediction(record: Mapping[str, Any], predictions_dir: Path) -> np.ndarray:
    path = _abs_path(record["test_logits_path"])
    if path.resolve().parent != Path(predictions_dir).resolve():
        raise RuntimeError(f"prediction outside the prediction directory: {path}")
    if not path.is_file():
        raise RuntimeError(f"frozen logits missing: {path}")
    sha = C.sha256_file(path)
    if sha != record["test_logits_sha256"]:
        raise RuntimeError(f"frozen logits sha {sha} != manifest {record['test_logits_sha256']}")
    arr = np.load(path)
    if list(arr.shape) != list(record["test_logits_shape"]):
        raise RuntimeError(f"{path} shape {arr.shape} != {record['test_logits_shape']}")
    if arr.dtype != np.float32:
        raise RuntimeError(f"{path} dtype {arr.dtype} != float32")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{path} contains non-finite logits")
    return arr


def _wins(a: Sequence[Any], b: Sequence[Any]) -> tuple[int, int, int]:
    up = down = both = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        both += 1
        if float(x) > float(y):
            up += 1
        elif float(x) < float(y):
            down += 1
    return up, down, both


def _stratified(per_task_a: Sequence[Any], per_task_b: Sequence[Any], support: Sequence[int], *, min_support: int) -> dict[str, Any]:
    deltas = []
    improved = 0
    for a, b, s in zip(per_task_a, per_task_b, support):
        if a is None or b is None or int(s) < int(min_support):
            continue
        deltas.append(float(a) - float(b))
        improved += int(float(a) > float(b))
    if not deltas:
        return {"n_tasks": 0, "mean_delta": None, "median_delta": None, "improved": 0}
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "n_tasks": int(arr.size),
        "mean_delta": float(arr.mean()),
        "median_delta": float(np.median(arr)),
        "improved": int(improved),
        "worst_delta": float(arr.min()),
        "best_delta": float(arr.max()),
    }


def run_score(
    freeze_path: Path,
    out_path: Path,
    per_task_path: Path,
    predictions_dir: Path,
) -> int:
    freeze = C.read_json(freeze_path)
    if freeze.get("schema") != CC.SCHEMA_TEST_FREEZE:
        raise RuntimeError("wrong freeze schema")
    if freeze.get("test_labels_read") is not False:
        raise RuntimeError("freeze manifest does not declare test_labels_read=false")
    if freeze.get("seal_sha256") != CD.freeze_seal(freeze):
        raise RuntimeError("freeze seal does not verify")
    selection_path = _abs_path(freeze["selection_manifest"])
    if C.sha256_file(selection_path) != freeze["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest changed after the prediction freeze")

    task = C.load_task_gated()
    test_reference = CD.read_official_test_labels(task, freeze_path=freeze_path)
    y_test = np.asarray(test_reference["labels"])
    if y_test.shape != (C.N_TEST, C.N_TASKS):
        raise RuntimeError(f"official Test label matrix {y_test.shape}")

    records = freeze["predictions"]
    for record in records.values():
        _load_prediction(record, predictions_dir)

    scored: dict[str, Any] = {}
    for label, logical in freeze["logical_models"].items():
        record = records[logical["prediction"]]
        logits = _load_prediction(record, predictions_dir)
        probs = (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)
        official = MET.official_ap(y_test, probs)
        independent = MET.per_task_ap(y_test, probs)
        scored[label] = {
            "model": record["model"],
            "epoch": int(record["epoch"]),
            "role": record["role"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "valid_ap": float(record["valid_ap_replay"]),
            "n_tasks_scored_valid": int(record["n_tasks_scored_valid"]),
            "test_ap_official": float(official["value"]),
            "test_ap_independent": float(independent["mean_over_scored_tasks"]),
            "test_ap_abs_delta": float(abs(float(official["value"]) - float(independent["mean_over_scored_tasks"]))),
            "test_bce": float(MET.masked_bce_numpy(logits, y_test)),
            "test_per_task_ap": independent["per_task"],
            "test_observed": independent["observed"],
            "test_positive": independent["positive"],
            "test_negative": independent["negative"],
            "n_tasks_scored": int(independent["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(independent["n_tasks_with_observed_labels"]),
            "valid_per_task_ap": record["valid_per_task_ap"],
            "valid_observed": record["valid_observed"],
            "valid_positive": record["valid_positive"],
            "valid_negative": record["valid_negative"],
            "valid_test_ap_delta": float(official["value"]) - float(record["valid_ap_replay"]),
            "test_logits_sha256": record["test_logits_sha256"],
        }
        C.log(f"[score] {label} valid {record['valid_ap_replay']:.9f} test {official['value']:.9f}")

    order = ["W2_best_valid_epoch24", "S3_best_valid_epoch29", "SP_best_valid_selected"]
    comparisons: dict[str, Any] = {}
    for new, old in (("SP_best_valid_selected", "S3_best_valid_epoch29"), ("SP_best_valid_selected", "W2_best_valid_epoch24"),
                     ("S3_best_valid_epoch29", "W2_best_valid_epoch24")):
        up, down, both = _wins(scored[new]["test_per_task_ap"], scored[old]["test_per_task_ap"])
        comparisons[f"test_{new}_minus_{old}"] = {
            "delta": scored[new]["test_ap_official"] - scored[old]["test_ap_official"],
            "valid_delta": scored[new]["valid_ap"] - scored[old]["valid_ap"],
            "tasks_improved": up,
            "tasks_degraded": down,
            "tasks_scored_both": both,
        }

    valid_obs = np.isfinite(task.labels[task.valid_rows])
    test_obs = np.isfinite(y_test)
    label_stats = {
        "valid": {
            "rows": int(task.valid_rows.size),
            "observed_entries": int(valid_obs.sum()),
            "positive_entries": int((task.labels[task.valid_rows][valid_obs] == 1.0).sum()),
        },
        "test": {
            "rows": int(task.test_rows.size),
            "observed_entries": int(test_obs.sum()),
            "positive_entries": int((y_test[test_obs] == 1.0).sum()),
        },
    }
    for split in label_stats:
        obs = label_stats[split]["observed_entries"]
        label_stats[split]["positive_rate_observed"] = (
            float(label_stats[split]["positive_entries"] / obs) if obs else None
        )

    support_valid = np.asarray(scored["SP_best_valid_selected"]["valid_positive"], dtype=np.int64)
    support_test = np.asarray(scored["SP_best_valid_selected"]["test_positive"], dtype=np.int64)
    support_stats = {
        "valid_ge20": _stratified(
            scored["SP_best_valid_selected"]["valid_per_task_ap"],
            scored["S3_best_valid_epoch29"]["valid_per_task_ap"],
            support_valid,
            min_support=20,
        ),
        "test_ge20": _stratified(
            scored["SP_best_valid_selected"]["test_per_task_ap"],
            scored["S3_best_valid_epoch29"]["test_per_task_ap"],
            support_test,
            min_support=20,
        ),
        "test_ge5_lt20": _stratified(
            scored["SP_best_valid_selected"]["test_per_task_ap"],
            scored["S3_best_valid_epoch29"]["test_per_task_ap"],
            support_test,
            min_support=5,
        ),
        "test_vs_w2_ge20": _stratified(
            scored["SP_best_valid_selected"]["test_per_task_ap"],
            scored["W2_best_valid_epoch24"]["test_per_task_ap"],
            support_test,
            min_support=20,
        ),
    }

    results: dict[str, Any] = {
        "schema": "lcrc-pcba-s3-selective-test-results-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_role": "EXPOSED_TEST_PERFORMANCE_BENCHMARK",
        "freeze_manifest": C.track_rel(freeze_path),
        "freeze_seal_sha256": freeze["seal_sha256"],
        "selection_manifest": freeze["selection_manifest"],
        "selection_manifest_sha256": freeze["selection_manifest_sha256"],
        "official_evaluator": "ogb.graphproppred.Evaluator('ogbg-molpcba')",
        "independent_evaluator": "sklearn average_precision_score per task with NaN masking",
        "test_label_artifact": {k: v for k, v in test_reference.items() if k != "labels"},
        "models": scored,
        "comparisons": comparisons,
        "support_strata": support_stats,
        "label_stats": label_stats,
        "n_test_rows": int(C.N_TEST),
        "n_tasks": int(C.N_TASKS),
        "n_tasks_scored_test": int(scored["SP_best_valid_selected"]["n_tasks_scored"]),
        "n_tasks_scored_valid": int(scored["SP_best_valid_selected"]["n_tasks_scored_valid"]),
        "test_label_masking": "NaN entries excluded per task; a task is scored only with both classes present",
        "standardisation": "Train-only normalisation inherited from the frozen cache; nothing refitted",
        "checkpoint_selection": "argmax official Validation AP over epochs 1..40, earliest wins ties; no Test score looked at",
        "test_exposure_history": {
            "exposed_by": "lcrc_pcba_continuation_recipe_v1 (2026-11-06) read the official Test labels after its own freeze",
            "this_round_role": "EXPOSED_TEST_PERFORMANCE_BENCHMARK",
            "blind_generalisation_estimate": False,
            "reason": (
                "the split has been read before and the result may influence the next architecture decision, "
                "so this is a benchmark of an already-exposed split, not an independent blind test"
            ),
            "no_test_based_selection": True,
            "no_retraining_after_seeing_test": True,
        },
        "descriptive_only": True,
        "cannot_change_recipe_or_selection": True,
    }
    C.write_json(out_path, results)
    _write_per_task_csv(per_task_path, scored, y_test, label_stats)
    C.log(f"[score] wrote {out_path} and {per_task_path}")
    return 0


def _write_per_task_csv(
    path: Path,
    scored: Mapping[str, Any],
    y_test: np.ndarray,
    label_stats: Mapping[str, Any],
) -> None:
    order = ["W2_best_valid_epoch24", "S3_best_valid_epoch29", "SP_best_valid_selected"]
    order = [x for x in order if x in scored]
    fields = ["task", "valid_observed", "valid_positive", "test_observed", "test_positive", "support_stratum"]
    fields += [f"valid_ap_{label}" for label in order]
    fields += [f"test_ap_{label}" for label in order]
    fields += ["test_delta_SP_minus_S3", "test_delta_SP_minus_W2", "valid_delta_SP_minus_S3"]
    ref = scored["SP_best_valid_selected"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in range(C.N_TASKS):
            support = int(ref["test_positive"][t])
            row: dict[str, Any] = {
                "task": t,
                "valid_observed": int(ref["valid_observed"][t]),
                "valid_positive": int(ref["valid_positive"][t]),
                "test_observed": int(ref["test_observed"][t]),
                "test_positive": support,
                "support_stratum": "test_ge20" if support >= 20 else "test_5_19" if support >= 5 else "test_lt5",
            }
            for label in order:
                ap = scored[label]["valid_per_task_ap"][t]
                row[f"valid_ap_{label}"] = "" if ap is None else float(ap)
            for label in order:
                ap = scored[label]["test_per_task_ap"][t]
                row[f"test_ap_{label}"] = "" if ap is None else float(ap)
            sp = scored["SP_best_valid_selected"]["test_per_task_ap"][t]
            s3 = scored["S3_best_valid_epoch29"]["test_per_task_ap"][t]
            w2 = scored["W2_best_valid_epoch24"]["test_per_task_ap"][t]
            row["test_delta_SP_minus_S3"] = "" if sp is None or s3 is None else float(sp - s3)
            row["test_delta_SP_minus_W2"] = "" if sp is None or w2 is None else float(sp - w2)
            vsp = scored["SP_best_valid_selected"]["valid_per_task_ap"][t]
            vs3 = scored["S3_best_valid_epoch29"]["valid_per_task_ap"][t]
            row["valid_delta_SP_minus_S3"] = "" if vsp is None or vs3 is None else float(vsp - vs3)
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--selection", default=str(SELECTION_PATH))
    fz.add_argument("--out", default=str(FREEZE_PATH))
    fz.add_argument("--predictions-dir", default=str(PREDICTIONS))
    fz.add_argument("--device", default="cuda")
    bc = sub.add_parser("build-cache")
    bc.add_argument("--out", default=str(C.S3_TEST_CACHE_DIR))
    sc = sub.add_parser("score")
    sc.add_argument("--freeze", default=str(FREEZE_PATH))
    sc.add_argument("--out", default=str(SCORES_PATH))
    sc.add_argument("--per-task", default=str(PER_TASK_PATH))
    sc.add_argument("--predictions-dir", default=str(PREDICTIONS))
    args = parser.parse_args(argv)
    if args.mode == "freeze":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return run_freeze(Path(args.selection), Path(args.out), Path(args.predictions_dir), device)
    if args.mode == "build-cache":
        return 0 if run_build_cache(Path(args.out))["audit"]["all_pass"] else 4
    if args.mode == "score":
        return run_score(Path(args.freeze), Path(args.out), Path(args.per_task), Path(args.predictions_dir))
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
