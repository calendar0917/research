"""Official Test prediction freeze and the single post-freeze scoring pass.

Two separate subcommands enforce the ordering:

* ``freeze`` replays Validation AP, writes Test logits for the pre-registered
  models and seals their hashes.  It never opens the official Test labels.
* ``score`` verifies the freeze seal and every prediction SHA256, and only then
  reads ``data/ogb/ogbg_molpcba/raw/graph-label.csv.gz`` for the official Test
  AP.  Scoring is descriptive; it cannot change the model list or the recipe.
"""

from __future__ import annotations

import argparse
import csv
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

import cc_common as C  # noqa: E402
import cc_data as D  # noqa: E402
import cc_restore as R  # noqa: E402
import pc_metrics as MET  # noqa: E402


def _np_sha(array: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def _save_npy_atomic(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(array, dtype=np.float32))
    os.replace(tmp, path)
    return C.sha256_file(path)


def _load_candidate(entry: Mapping[str, Any], task: Any, device: torch.device):
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = C.REPO / path
    sha = C.sha256_file(path)
    if entry.get("sha256") and sha != entry["sha256"]:
        raise RuntimeError(f"candidate {entry.get('label')} sha {sha} != recorded {entry['sha256']}")
    payload = R.load_checkpoint(path)
    payload["_path"] = C.track_rel(path)
    model, _, audit = R.restore_model_and_optimizer(entry["model"], payload, task, device=device)
    return model, payload, audit, sha


def _predict_validation_and_test(
    entry: Mapping[str, Any],
    task: Any,
    device: torch.device,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    model, payload, audit, sha = _load_candidate(entry, task, device)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    valid = D.evaluate_validation(model, task, device, reason="test_freeze_validation_replay")
    t_valid = time.perf_counter() - t1
    expected_valid = float(entry["valid_ap"])
    valid_delta = abs(float(valid["ap"]) - expected_valid)
    if valid_delta > 1e-4:
        raise RuntimeError(
            f"{entry['label']} Validation replay {valid['ap']} != recorded {expected_valid} "
            f"(delta {valid_delta})"
        )
    t2 = time.perf_counter()
    logits = D.predict_rows(model, task, task.test_rows, device, batch_size=256)
    t_test = time.perf_counter() - t2
    if logits.shape != (C.N_TEST, C.N_TASKS):
        raise RuntimeError(f"{entry['label']} Test logits shape {logits.shape}")
    if not np.isfinite(logits).all():
        raise RuntimeError(f"{entry['label']} produced non-finite Test logits")
    return {
        "entry": dict(entry),
        "checkpoint_sha256": sha,
        "valid_ap_replay": float(valid["ap"]),
        "valid_ap_recorded": expected_valid,
        "valid_ap_abs_delta": float(valid_delta),
        "valid_bce_replay": float(valid["valid_bce"]),
        "valid_per_task_ap": valid["per_task_ap"],
        "valid_observed": valid["observed"],
        "valid_positive": valid["positive"],
        "valid_negative": valid["negative"],
        "logits": logits.astype(np.float32),
        "load_seconds": float(t_load),
        "valid_seconds": float(t_valid),
        "test_seconds": float(t_test),
        "restore_audit": audit,
    }


def _canonical_logical_models(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    logical: dict[str, dict[str, Any]] = {}
    for model_id in C.TEST_MODEL_ORDER:
        spec = C.SOURCE[model_id]
        logical[f"{model_id}_epoch40"] = {
            "label": f"{model_id}_epoch40",
            "model": model_id,
            "epoch": C.EPOCH40,
            "recipe": "none",
            "path": C.track_rel(spec["path"]),
            "sha256": spec["sha256"],
            "valid_ap": float(spec["valid_ap"]),
            "role": "historical_epoch40",
        }
    for model_id in C.MODEL_ORDER:
        selected = selection["models"][model_id]["selected"]
        logical[f"{model_id}_selected"] = {
            "label": f"{model_id}_selected",
            "model": model_id,
            "epoch": int(selected["epoch"]),
            "recipe": selected.get("recipe", "none"),
            "path": selected["path"],
            "sha256": selected["sha256"],
            "valid_ap": float(selected["valid_ap"]),
            "role": "validation_selected",
        }
    return logical


def run_freeze(selection_path: Path, out_path: Path, predictions_dir: Path, device: torch.device) -> int:
    task = D.load_and_gate()
    selection = C.read_json(selection_path)
    if selection.get("test_read") is not False:
        raise RuntimeError("selection manifest does not declare test_read=false")
    logical = _canonical_logical_models(selection)
    for label, entry in logical.items():
        if not entry["path"]:
            raise RuntimeError(f"logical model {label} has no checkpoint path")

    # Unique checkpoints by (model, sha256).  Aliases reuse an existing logits file.
    unique: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for label, entry in logical.items():
        key = f"{entry['model']}:{entry['sha256']}"
        if key in unique:
            aliases[label] = unique[key]["label"]
        else:
            unique[key] = entry

    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for key, entry in unique.items():
        res = _predict_validation_and_test(entry, task, device)
        fname = f"{entry['label']}.npy"
        logits_path = predictions_dir / fname
        logits_sha = _save_npy_atomic(logits_path, res["logits"])
        records[entry["label"]] = {
            "label": entry["label"],
            "model": entry["model"],
            "role": entry.get("role"),
            "epoch": int(entry["epoch"]),
            "recipe": entry.get("recipe", "none"),
            "checkpoint": entry["path"],
            "checkpoint_sha256": res["checkpoint_sha256"],
            "valid_ap_recorded": res["valid_ap_recorded"],
            "valid_ap_replay": res["valid_ap_replay"],
            "valid_ap_abs_delta": res["valid_ap_abs_delta"],
            "valid_bce_replay": res["valid_bce_replay"],
            "valid_per_task_ap": res["valid_per_task_ap"],
            "valid_observed": res["valid_observed"],
            "valid_positive": res["valid_positive"],
            "valid_negative": res["valid_negative"],
            "logits_path": C.track_rel(logits_path),
            "logits_sha256": logits_sha,
            "logits_shape": list(res["logits"].shape),
            "logits_dtype": "float32",
            "logits_min": float(res["logits"].min()),
            "logits_max": float(res["logits"].max()),
            "logits_mean": float(res["logits"].mean()),
            "rows": int(res["logits"].shape[0]),
            "n_tasks": int(res["logits"].shape[1]),
            "load_seconds": res["load_seconds"],
            "valid_seconds": res["valid_seconds"],
            "test_seconds": res["test_seconds"],
            "restore_audit": res["restore_audit"],
        }
        C.log(f"[freeze] {entry['label']} valid {res['valid_ap_replay']:.9f} test logits {logits_sha[:12]}")

    rows_path = predictions_dir / "test_row_ids.npy"
    tmp = rows_path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.asarray(task.test_rows, dtype=np.int64))
    os.replace(tmp, rows_path)
    rows_sha = C.sha256_file(rows_path)

    logical_payload: dict[str, Any] = {}
    for label, entry in logical.items():
        if label in aliases:
            base = aliases[label]
            logical_payload[label] = {
                "alias_of": base,
                "checkpoint_sha256": records[base]["checkpoint_sha256"],
                "valid_ap_replay": records[base]["valid_ap_replay"],
            }
        else:
            logical_payload[label] = {
                "prediction": label,
                "checkpoint_sha256": records[label]["checkpoint_sha256"],
                "valid_ap_replay": records[label]["valid_ap_replay"],
            }

    freeze: dict[str, Any] = {
        "schema": C.SCHEMA_TEST_FREEZE,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_labels_read": False,
        "test_labels_read_before_freeze": False,
        "selection_manifest": C.track_rel(selection_path),
        "selection_manifest_sha256": C.sha256_file(selection_path),
        "test_rows_path": C.track_rel(rows_path),
        "test_rows_sha256": rows_sha,
        "n_test_rows": int(len(task.test_rows)),
        "logical_models": logical_payload,
        "predictions": records,
        "unique_predictions": sorted(records),
        "aliases": aliases,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "environment": C.environment(),
        "total_seconds": float(time.perf_counter() - started),
        "total_gpu_hours": float((time.perf_counter() - started) / 3600.0),
        "note": (
            "Test logits are frozen before any official Test label is opened. "
            "Only cc_test.py score reads raw/graph-label.csv.gz."
        ),
    }
    freeze["seal_sha256"] = D.freeze_seal(freeze)
    C.write_json(out_path, freeze)
    # Verify the just-written file round-trips through the seal.
    check = C.read_json(out_path)
    if check["seal_sha256"] != D.freeze_seal(check):
        raise RuntimeError("written freeze manifest seal does not verify")
    C.log(f"[freeze] wrote {out_path} seal={freeze['seal_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# scoring (only after the freeze)
# ---------------------------------------------------------------------------


def _load_prediction(record: Mapping[str, Any], predictions_dir: Path) -> np.ndarray:
    path = Path(str(record["logits_path"]))
    if not path.is_absolute():
        path = C.REPO / path
    if not path.is_file():
        raise RuntimeError(f"frozen logits missing: {path}")
    sha = C.sha256_file(path)
    if sha != record["logits_sha256"]:
        raise RuntimeError(f"frozen logits sha {sha} != manifest {record['logits_sha256']}")
    arr = np.load(path)
    if arr.shape != tuple(record["logits_shape"]):
        raise RuntimeError(f"{path} shape {arr.shape} != {record['logits_shape']}")
    if arr.dtype != np.float32:
        raise RuntimeError(f"{path} dtype {arr.dtype} != float32")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{path} contains non-finite logits")
    return arr


def _count_task_wins(a: Sequence[Any], b: Sequence[Any], *, higher_is_better: bool = True) -> int:
    wins = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if (float(x) > float(y)) if higher_is_better else (float(x) < float(y)):
            wins += 1
    return int(wins)


def run_score(freeze_path: Path, out_path: Path, per_task_path: Path, predictions_dir: Path) -> int:
    freeze = C.read_json(freeze_path)
    if freeze.get("schema") != C.SCHEMA_TEST_FREEZE:
        raise RuntimeError("wrong freeze schema")
    if freeze.get("test_labels_read") is not False:
        raise RuntimeError("freeze manifest does not declare test_labels_read=false")
    if freeze.get("seal_sha256") != D.freeze_seal(freeze):
        raise RuntimeError("freeze seal does not verify")
    task = D.load_and_gate()
    test = D.read_official_test_labels(task, freeze_path=freeze_path)
    y_test = test["labels"]

    # Replay every frozen prediction's SHA before scoring.
    records = freeze["predictions"]
    for label, record in records.items():
        _load_prediction(record, predictions_dir)

    scored: dict[str, Any] = {}
    for label, logical in freeze["logical_models"].items():
        source_label = logical.get("alias_of") or logical.get("prediction")
        if source_label is None or source_label not in records:
            raise RuntimeError(f"logical model {label} has no prediction record")
        record = records[source_label]
        logits = _load_prediction(record, predictions_dir)
        probs = (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)
        official = MET.official_ap(y_test, probs)
        independent = MET.per_task_ap(y_test, probs)
        delta = abs(float(official["value"]) - float(independent["mean_over_scored_tasks"]))
        scored[label] = {
            "model": record["model"],
            "epoch": record["epoch"],
            "recipe": record["recipe"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "prediction": source_label,
            "alias_of": logical.get("alias_of"),
            "valid_ap": record["valid_ap_replay"],
            "test_ap_official": float(official["value"]),
            "test_ap_independent": float(independent["mean_over_scored_tasks"]),
            "test_ap_abs_delta": float(delta),
            "test_bce": float(MET.masked_bce_numpy(logits, y_test)),
            "test_per_task_ap": independent["per_task"],
            "test_observed": independent["observed"],
            "test_positive": independent["positive"],
            "test_negative": independent["negative"],
            "n_tasks_scored": int(independent["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(independent["n_tasks_with_observed_labels"]),
            "valid_per_task_ap": record.get("valid_per_task_ap"),
            "valid_observed": record.get("valid_observed"),
            "valid_positive": record.get("valid_positive"),
            "valid_negative": record.get("valid_negative"),
            "valid_test_ap_delta": float(official["value"]) - float(record["valid_ap_replay"]),
        }
        C.log(f"[score] {label} valid {record['valid_ap_replay']:.9f} test {official['value']:.9f}")

    comparisons: dict[str, Any] = {}
    for model_id in C.MODEL_ORDER:
        base = f"{model_id}_epoch40"
        final = f"{model_id}_selected"
        if final in scored and base in scored:
            comparisons[f"{final}_vs_{base}"] = {
                "test_ap_delta": scored[final]["test_ap_official"] - scored[base]["test_ap_official"],
                "valid_ap_delta": scored[final]["valid_ap"] - scored[base]["valid_ap"],
                "n_tasks_improved": _count_task_wins(
                    scored[final]["test_per_task_ap"], scored[base]["test_per_task_ap"]
                ),
                "n_tasks_degraded": _count_task_wins(
                    scored[base]["test_per_task_ap"], scored[final]["test_per_task_ap"]
                ),
                "n_tasks_scored_both": sum(
                    1
                    for a, b in zip(scored[final]["test_per_task_ap"], scored[base]["test_per_task_ap"])
                    if a is not None and b is not None
                ),
                "same_checkpoint": scored[final]["checkpoint_sha256"] == scored[base]["checkpoint_sha256"],
            }
    for model_id in C.MODEL_ORDER:
        label = f"{model_id}_epoch40"
        if label in scored:
            comparisons[f"{label}_vs_C1_epoch40"] = {
                "test_ap_delta": scored[label]["test_ap_official"] - scored["C_epoch40"]["test_ap_official"],
                "valid_ap_delta": scored[label]["valid_ap"] - scored["C_epoch40"]["valid_ap"],
                "n_tasks_improved": _count_task_wins(
                    scored[label]["test_per_task_ap"], scored["C_epoch40"]["test_per_task_ap"]
                ),
                "n_tasks_degraded": _count_task_wins(
                    scored["C_epoch40"]["test_per_task_ap"], scored[label]["test_per_task_ap"]
                ),
            }

    valid_observed = np.isfinite(task.labels[task.valid_rows])
    valid_positive = (task.labels[task.valid_rows] == 1.0) & valid_observed
    valid_obs = int(valid_observed.sum())
    valid_pos = int(valid_positive.sum())
    test_obs = int(np.isfinite(y_test).sum())
    test_pos = int((y_test == 1.0).sum())

    results: dict[str, Any] = {
        "schema": "lcrc-pcba-continuation-test-results-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "freeze_manifest": C.track_rel(freeze_path),
        "freeze_seal_sha256": freeze["seal_sha256"],
        "official_evaluator": C.DECISION["test"]["official_evaluator"],
        "independent_evaluator": C.DECISION["test"]["independent"],
        "test_label_artifact": {k: v for k, v in test.items() if k != "labels"},
        "models": scored,
        "comparisons": comparisons,
        "valid_label_stats": {
            "rows": int(task.valid_rows.size),
            "observed_entries": valid_obs,
            "positive_entries": valid_pos,
            "positive_rate_observed": float(valid_pos / valid_obs) if valid_obs else None,
        },
        "test_label_stats": {
            "rows": int(task.test_rows.size),
            "observed_entries": test_obs,
            "positive_entries": test_pos,
            "positive_rate_observed": float(test_pos / test_obs) if test_obs else None,
        },
        "valid_test_positive_rate_delta": (
            float(test_pos / test_obs) - float(valid_pos / valid_obs)
            if test_obs and valid_obs
            else None
        ),
        "n_test_rows": int(task.test_rows.size),
        "n_tasks": C.N_TASKS,
        "test_exposure_history": {
            "prior_pcba_test_read_in_repo": False,
            "evidence": (
                "Earlier PCBA rounds' reports state official Test was never read; their "
                "label artifact has Test rows NaN and their protocols forbid Test access."
            ),
            "this_round_read_after_freeze": True,
            "future_use_must_disclose_exposure": True,
        },
        "descriptive_only": True,
        "cannot_change_recipe_or_selection": True,
    }
    C.write_json(out_path, results)
    _write_per_task_csv(per_task_path, scored, y_test, task)
    C.log(f"[score] wrote {out_path} and {per_task_path}")
    return 0


def _write_per_task_csv(
    path: Path,
    scored: Mapping[str, Any],
    y_test: np.ndarray,
    task: Any,
) -> None:
    labels = list(scored)
    # Fixed, stable column order: C1, A/G/V epoch40, A/G/V selected.
    order = [f"{m}_epoch40" for m in C.TEST_MODEL_ORDER] + [
        f"{m}_selected" for m in C.MODEL_ORDER
    ]
    order = [x for x in order if x in scored]
    valid_obs = scored["C_epoch40"]["valid_observed"] if "C_epoch40" in scored else []
    valid_pos = scored["C_epoch40"]["valid_positive"] if "C_epoch40" in scored else []
    test_obs = scored["C_epoch40"]["test_observed"] if "C_epoch40" in scored else []
    test_pos = scored["C_epoch40"]["test_positive"] if "C_epoch40" in scored else []
    fields = [
        "task",
        "valid_observed",
        "valid_positive",
        "valid_positive_rate",
        "test_observed",
        "test_positive",
        "test_positive_rate",
    ]
    for label in order:
        fields.append(f"valid_ap_{label}")
    for label in order:
        fields.append(f"test_ap_{label}")
    for model_id in C.MODEL_ORDER:
        fields.append(f"test_delta_{model_id}_selected_minus_epoch40")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in range(C.N_TASKS):
            row: dict[str, Any] = {
                "task": t,
                "valid_observed": int(valid_obs[t]),
                "valid_positive": int(valid_pos[t]),
                "valid_positive_rate": float(valid_pos[t] / valid_obs[t]) if valid_obs[t] else "",
                "test_observed": int(test_obs[t]),
                "test_positive": int(test_pos[t]),
                "test_positive_rate": float(test_pos[t] / test_obs[t]) if test_obs[t] else "",
            }
            for label in order:
                ap = scored[label].get("valid_per_task_ap")
                ap = None if ap is None else ap[t]
                row[f"valid_ap_{label}"] = "" if ap is None else float(ap)
            for label in order:
                ap = scored[label]["test_per_task_ap"][t]
                row[f"test_ap_{label}"] = "" if ap is None else float(ap)
            for model_id in C.MODEL_ORDER:
                a = scored[f"{model_id}_selected"]["test_per_task_ap"][t]
                b = scored[f"{model_id}_epoch40"]["test_per_task_ap"][t]
                row[f"test_delta_{model_id}_selected_minus_epoch40"] = (
                    "" if a is None or b is None else float(a - b)
                )
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--selection", default=str(C.ANALYSIS / "validation_selection.json"))
    fz.add_argument("--out", default=str(C.ANALYSIS / "test_prediction_freeze.json"))
    fz.add_argument("--predictions-dir", default=str(C.TEST_PREDICTIONS))
    fz.add_argument("--device", default="cuda")
    sc = sub.add_parser("score")
    sc.add_argument("--freeze", default=str(C.ANALYSIS / "test_prediction_freeze.json"))
    sc.add_argument("--out", default=str(C.ANALYSIS / "test_results.json"))
    sc.add_argument("--per-task", default=str(C.ANALYSIS / "per_task_valid_test.csv"))
    sc.add_argument("--predictions-dir", default=str(C.TEST_PREDICTIONS))
    args = parser.parse_args(argv)
    if args.mode == "freeze":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return run_freeze(
            Path(args.selection), Path(args.out), Path(args.predictions_dir), device
        )
    if args.mode == "score":
        return run_score(
            Path(args.freeze), Path(args.out), Path(args.per_task), Path(args.predictions_dir)
        )
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
