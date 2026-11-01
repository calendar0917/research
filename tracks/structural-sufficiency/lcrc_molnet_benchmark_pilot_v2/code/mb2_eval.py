"""One-shot Test evaluation for the three new datasets (BBBP, Tox21, ogbg-molhiv).

Preconditions, all enforced (not merely documented):

* ``analysis/checkpoint_manifest.json`` exists and lists this dataset;
* the on-disk checkpoint sha256 equals the frozen value;
* the recorded code modules have not changed since the freeze;
* rebuilding the Train/Validation structure caches reproduces the frozen
  fingerprints;
* the reloaded best checkpoint reproduces its recorded Validation metric (1e-5);
* the recorded best epoch equals the Validation-curve argbest/argmin;
* no ``test_eval.json`` exists yet for this dataset (one shot).

Only after those checks does the module build the Test structure cache, decode the
Test targets **once** and score them.  Every Test read is appended to
``analysis/test_access_ledger.json`` so the sequence of Test accesses is auditable
after the fact.

Usage
-----
    uv run python .../code/mb2_eval.py --dataset BBBP --device cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import mb2_common as C
import mb2_data as D
import mb2_gate as GATE
import mb2_metrics as MET
import mb2_model as M

import lcrc_data as PREP_DATA


def _manifest_entry(manifest: dict[str, Any], dataset: str) -> dict[str, Any]:
    if dataset not in manifest.get("datasets", {}):
        raise SystemExit(
            f"{dataset} is not in {C.track_rel(C.ANALYSIS / 'checkpoint_manifest.json')}; "
            "the Test evaluation is only allowed after the freeze"
        )
    return manifest["datasets"][dataset]


def _append_test_ledger(entry: dict[str, Any]) -> None:
    path = C.ANALYSIS / "test_access_ledger.json"
    payload = C.read_json(path) if path.exists() else {
        "schema": "lcrc-molnet-v2-test-access-ledger-v1",
        "round_id": C.ROUND_ID,
        "policy": (
            "every Test-label read of the round is appended here with its timestamp, the frozen "
            "manifest hash and the prediction-file hash; the list is append-only"
        ),
        "entries": [],
    }
    payload["entries"].append(entry)
    C.write_json(path, payload)


def evaluate(dataset: str, *, device: torch.device, data_device: torch.device | None = None, force: bool = False) -> dict[str, Any]:
    data_device = data_device or torch.device("cpu")
    manifest_path = C.ANALYSIS / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{C.track_rel(manifest_path)} missing: freeze the P1 checkpoints first")
    manifest = C.read_json(manifest_path)
    entry = _manifest_entry(manifest, dataset)
    out_dir = C.RUNS / f"{dataset}_s{C.TRAIN_SEED}"
    out_path = out_dir / "test_eval.json"
    if out_path.exists() and not force:
        raise SystemExit(f"{C.track_rel(out_path)} already exists: the Test evaluation is one-shot")

    # ---- identity checks before touching Test ------------------------
    checkpoint_path = C.REPO / entry["checkpoint"]["path"]
    actual = C.sha256_file(checkpoint_path)
    if actual != entry["checkpoint"]["sha256"]:
        raise SystemExit(f"{dataset}: checkpoint sha256 {actual} != frozen {entry['checkpoint']['sha256']}")
    frozen_critical = entry.get("code_hashes_scoring_critical") or {
        k: v for k, v in entry["code_hashes"].items() if k in GATE.SCORING_CRITICAL_MODULES
    }
    drift = GATE.drift(frozen_critical, label=f"{dataset} Test gate")
    if drift:
        raise SystemExit(f"{dataset}: a scoring-critical module changed since the freeze: {drift}")

    spec = D.frozen_spec()
    view = D.build_view(dataset, spec=spec)
    if view.train_cache["meta"]["content_fingerprint"] != entry["cache_fingerprints"]["train"]:
        raise SystemExit(f"{dataset}: rebuilt Train cache fingerprint differs from the frozen one")
    if view.valid_cache["meta"]["content_fingerprint"] != entry["cache_fingerprints"]["valid"]:
        raise SystemExit(f"{dataset}: rebuilt Valid cache fingerprint differs from the frozen one")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = M.load_checkpoint_model(
        spec,
        view.train_norm,
        state["state_dict"],
        device,
        dropout=float(state.get("dropout", C.RECIPE["dropout"])),
        n_tasks=int(state.get("n_outputs", 1)),
    )
    if int(state["best_epoch"]) != int(entry["checkpoint"]["best_epoch"]):
        raise SystemExit("checkpoint best_epoch differs from the frozen manifest")

    # ---- Validation re-check ------------------------------------------
    t_valid = time.perf_counter()
    valid_raw = D.predict_matrix(model, view.valid_cache, device)
    valid_metrics = MET.task_metrics(view.task, valid_raw, view.targets.y_valid, view.target_norm, view.task_names)
    valid_seconds = time.perf_counter() - t_valid
    key = MET.primary_name(view.task)
    valid_repro_diff = abs(float(valid_metrics[key]) - float(entry["checkpoint"]["best_metric_value"]))
    if valid_repro_diff > C.REPRO_TOL:
        raise SystemExit(
            f"{dataset}: reloaded Validation {key} {valid_metrics[key]} != frozen "
            f"{entry['checkpoint']['best_metric_value']}"
        )
    curve = np.genfromtxt(C.REPO / entry["artifacts"]["curve"], delimiter=",", names=True)
    col = {"regression": "valid_rmse_raw", "binary": "valid_roc_auc", "multitask": "valid_macro_roc_auc"}[view.task]
    argbest = int(np.argmin(curve[col]) + 1) if view.task == "regression" else int(np.argmax(curve[col]) + 1)
    if argbest != int(entry["checkpoint"]["best_epoch"]):
        raise SystemExit(f"{dataset}: curve argbest {argbest} != frozen best epoch {entry['checkpoint']['best_epoch']}")

    # ---- one-shot Test evaluation -------------------------------------
    t_cache = time.perf_counter()
    test_cache, y_test, test_rows = D.load_test_cache(dataset, spec=spec)
    test_cache_seconds = time.perf_counter() - t_cache
    test_audit = {
        "n": int(test_cache["n_molecules"]),
        "rows_sha256": C.row_sha256(test_rows),
        "rows_sha256_matches_frozen_split_manifest": None,
        "cache_content_fingerprint": test_cache["meta"]["content_fingerprint"],
        "n_atoms": int(np.asarray(test_cache["atom_ptr"])[-1]),
        "n_s1_codes": int(np.asarray(test_cache["nbr_ptr"])[-1]),
        "n_s2_paths": int(np.asarray(test_cache["path_ptr"])[-1]),
        "atom_unknown_fraction": float((np.asarray(test_cache["atom_type"]) == spec.atom_unk_index).mean()),
        "cache_build_seconds": float(test_cache.get("build_seconds", 0.0)),
    }
    split_manifest_path = C.ANALYSIS / f"split_manifest_{dataset}.json"
    if split_manifest_path.exists():
        frozen_rows = C.read_json(split_manifest_path)["rows_sha256"]["test"]
        test_audit["rows_sha256_matches_frozen_split_manifest"] = bool(frozen_rows == test_audit["rows_sha256"])
        if not test_audit["rows_sha256_matches_frozen_split_manifest"]:
            raise SystemExit(f"{dataset}: Test row ids differ from the P0 split manifest")

    train_raw = D.predict_matrix(model, view.train_cache, device)
    train_metrics = MET.task_metrics(view.task, train_raw, view.targets.y_train, view.target_norm, view.task_names)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t_test = time.perf_counter()
    test_raw = D.predict_matrix(model, test_cache, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - t_test
    gpu_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    test_metrics = MET.task_metrics(view.task, test_raw, y_test, view.target_norm, view.task_names)

    ogb_block = None
    if dataset == "ogbg-molhiv":
        ogb_block = MET.ogb_evaluator_roc_auc(y_test.reshape(-1), test_raw[:, 0])

    np.savez(
        out_dir / "test_predictions.npz",
        rows=np.asarray(test_rows, dtype=np.int64),
        raw=np.asarray(test_raw, dtype=np.float64),
        y_true=np.asarray(y_test, dtype=np.float64),
    )

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "dataset": dataset,
        "task": view.task,
        "label_columns": view.label_columns,
        "units": view.units,
        "primary_metric": key,
        "seed": int(C.TRAIN_SEED),
        "split_repeat_index": entry["split_repeat_index"],
        "one_shot": True,
        "evaluated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "hostname": __import__("platform").node(),
        "checkpoint_manifest": {
            "path": C.track_rel(manifest_path),
            "sha256": C.sha256_file(manifest_path),
            "frozen_git_commit": manifest.get("git_commit"),
        },
        "checkpoint": {
            "path": entry["checkpoint"]["path"],
            "sha256": actual,
            "best_epoch": int(state["best_epoch"]),
            "best_metric_name": entry["checkpoint"]["best_metric_name"],
            "best_metric_value": float(entry["checkpoint"]["best_metric_value"]),
            "parameters": int(entry["parameters"]),
            "n_outputs": int(entry["n_outputs"]),
        },
        "preconditions": {
            "checkpoint_sha256_matches_manifest": True,
            "code_hashes_unchanged_since_freeze": True,
            "train_cache_fingerprint_matches": True,
            "valid_cache_fingerprint_matches": True,
            "valid_metric_reproduced": True,
            "valid_metric_reproduced_abs_diff": float(valid_repro_diff),
            "valid_metric_reproduced_tolerance": float(C.REPRO_TOL),
            "valid_metric_reproduced_value": float(valid_metrics[key]),
            "curve_argbest_epoch": argbest,
            "best_epoch_matches_curve": True,
            "test_rows_match_p0_split_manifest": test_audit["rows_sha256_matches_frozen_split_manifest"],
        },
        "metrics": {
            "train": {**train_metrics, "note": "recomputed from the frozen best checkpoint"},
            "valid": {**valid_metrics, "note": "reproduced from the frozen best checkpoint"},
            "test": {**test_metrics, "note": "single one-shot evaluation"},
        },
        "ogb_evaluator": ogb_block,
        "test_split": test_audit,
        "timing": {
            "valid_seconds": float(valid_seconds),
            "test_seconds": float(test_seconds),
            "test_molecules_per_second": float(test_audit["n"] / max(test_seconds, 1e-9)),
            "test_cache_seconds": float(test_cache_seconds),
            "test_cache_build_seconds": float(test_cache.get("build_seconds", 0.0)),
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_memory_allocated_bytes": gpu_peak,
        },
        "predictions": {
            "path": C.track_rel(out_dir / "test_predictions.npz"),
            "sha256": C.sha256_file(out_dir / "test_predictions.npz"),
        },
        "test_labels_read": True,
        "test_used_for_selection": False,
        "comparison_label": "FIRST_TEST_READ",
    }
    C.write_json(out_path, payload)
    _append_test_ledger(
        {
            "kind": "lcrc_test_eval",
            "dataset": dataset,
            "utc": payload["evaluated_utc"],
            "one_shot": True,
            "checkpoint_manifest_sha256": payload["checkpoint_manifest"]["sha256"],
            "checkpoint_sha256": actual,
            "test_rows_sha256": test_audit["rows_sha256"],
            "test_predictions_sha256": payload["predictions"]["sha256"],
            "primary_metric": key,
            "primary_value": float(test_metrics[key]),
            "git_commit": payload["git_commit"],
        }
    )
    C.log(
        f"[{dataset} one-shot Test] {key}={test_metrics[key]:.5f} (valid {valid_metrics[key]:.5f}) "
        f"-> {C.track_rel(out_path)}"
    )
    del data_device, PREP_DATA
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="one-shot Test evaluation of the frozen LCRC checkpoints")
    parser.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--force", action="store_true", help="not permitted for a formal one-shot result")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.dataset in C.EXPOSED_TEST_DATASETS and not args.force:
        raise SystemExit(
            f"{args.dataset}: the Test set was already read in Sprint 01; this round does not re-read it "
            "(the Sprint-01 number is reused read-only, labelled EXPOSED_TEST_EXPLORATORY)"
        )
    evaluate(args.dataset, device=device, force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
