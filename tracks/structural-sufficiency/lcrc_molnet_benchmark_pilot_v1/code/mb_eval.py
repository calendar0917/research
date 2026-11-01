"""P2: one-shot Test evaluation after the checkpoint manifest is frozen.

Preconditions (all enforced, not documented):

* ``analysis/checkpoint_manifest.json`` exists and lists this dataset;
* the on-disk checkpoint sha256 equals the frozen value;
* rebuilding the Train/Validation structure caches reproduces the frozen
  fingerprints (this also proves the remote rebuild is byte-identical to the
  local P0 audit);
* the reloaded best checkpoint reproduces its recorded Validation metric;
* the recorded best epoch equals the Validation curve argmin/argmax;
* no ``test_eval.json`` exists yet for this dataset (one shot).

Only after those checks does the module build the Test structure cache, decode
the Test targets and score them once.  The Test result is written to
``analysis/runs/<dataset>_s0/test_eval.json`` together with the prediction file.

Usage
-----
    uv run python .../code/mb_eval.py --dataset ESOL --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import mb_common as C
import mb_data as D
import mb_model as M
import mb_train as T

import lcrc_data as PREP_DATA


def _manifest_entry(manifest: dict[str, Any], dataset: str) -> dict[str, Any]:
    if dataset not in manifest.get("datasets", {}):
        raise SystemExit(
            f"{dataset} is not in {C.track_rel(C.ANALYSIS / 'checkpoint_manifest.json')}; "
            "P2 Test evaluation is only allowed after the freeze"
        )
    return manifest["datasets"][dataset]


def evaluate(dataset: str, *, device: torch.device, force: bool = False) -> dict[str, Any]:
    manifest_path = C.ANALYSIS / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{C.track_rel(manifest_path)} missing: freeze P1 checkpoints first")
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
    code_hashes_now = T.code_hashes()
    drift = {
        k: (entry["code_hashes"].get(k), v)
        for k, v in code_hashes_now.items()
        if entry["code_hashes"].get(k) not in (None, v)
    }
    if drift:
        raise SystemExit(f"{dataset}: code changed since the freeze: {drift}")

    spec = D.frozen_spec()
    view = D.build_view(dataset, spec=spec)
    if view.train_cache["meta"]["content_fingerprint"] != entry["cache_fingerprints"]["train"]:
        raise SystemExit(f"{dataset}: rebuilt Train cache fingerprint differs from the frozen one")
    if view.valid_cache["meta"]["content_fingerprint"] != entry["cache_fingerprints"]["valid"]:
        raise SystemExit(f"{dataset}: rebuilt Valid cache fingerprint differs from the frozen one")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = M.load_checkpoint_model(spec, view.train_norm, state["state_dict"], device)
    if int(state["best_epoch"]) != int(entry["checkpoint"]["best_epoch"]):
        raise SystemExit("checkpoint best_epoch differs from the frozen manifest")

    # ---- Validation re-check ------------------------------------------
    t_valid = time.perf_counter()
    valid_raw = D.predict_scores(model, view.valid_cache, device)
    valid_metrics = T.task_metrics(view.task, valid_raw, view.y_valid, view.target_norm)
    valid_seconds = time.perf_counter() - t_valid
    key = "rmse" if view.task == "regression" else "roc_auc"
    if abs(float(valid_metrics[key]) - float(entry["checkpoint"]["best_metric_value"])) > 1e-9:
        raise SystemExit(
            f"{dataset}: reloaded Validation {key} {valid_metrics[key]} != frozen "
            f"{entry['checkpoint']['best_metric_value']}"
        )
    curve = np.genfromtxt(C.REPO / entry["artifacts"]["curve"], delimiter=",", names=True)
    col = "valid_rmse_raw" if view.task == "regression" else "valid_roc_auc"
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
        "cache_content_fingerprint": test_cache["meta"]["content_fingerprint"],
        "n_atoms": int(np.asarray(test_cache["atom_ptr"])[-1]),
        "n_s1_codes": int(np.asarray(test_cache["nbr_ptr"])[-1]),
        "n_s2_paths": int(np.asarray(test_cache["path_ptr"])[-1]),
        "atom_unknown_fraction": float(
            (np.asarray(test_cache["atom_type"]) == spec.atom_unk_index).mean()
        ),
        "cache_build_seconds": float(test_cache.get("build_seconds", 0.0)),
    }

    train_raw = D.predict_scores(model, view.train_cache, device)
    train_metrics = T.task_metrics(view.task, train_raw, view.y_train, view.target_norm)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t_test = time.perf_counter()
    test_raw = D.predict_scores(model, test_cache, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - t_test
    gpu_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    test_metrics = T.task_metrics(view.task, test_raw, y_test, view.target_norm)

    np.savez(
        out_dir / "test_predictions.npz",
        rows=np.asarray(test_rows, dtype=np.int64),
        raw=test_raw.astype(np.float64),
        y_true=np.asarray(y_test, dtype=np.float64),
    )

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "dataset": dataset,
        "task": view.task,
        "label_column": view.label_column,
        "units": view.units,
        "seed": int(C.TRAIN_SEED),
        "split_repeat_index": C.SPLIT_REPEAT_INDEX,
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
        },
        "preconditions": {
            "checkpoint_sha256_matches_manifest": True,
            "code_hashes_unchanged_since_freeze": True,
            "train_cache_fingerprint_matches": True,
            "valid_cache_fingerprint_matches": True,
            "valid_metric_reproduced": True,
            "valid_metric_reproduced_value": float(valid_metrics[key]),
            "curve_argbest_epoch": argbest,
            "best_epoch_matches_curve": True,
        },
        "metrics": {
            "train": {**train_metrics, "note": "recomputed from the frozen best checkpoint"},
            "valid": {**valid_metrics, "note": "reproduced from the frozen best checkpoint"},
            "test": {**test_metrics, "note": "single one-shot evaluation"},
        },
        "test_split": test_audit,
        "timing": {
            "valid_seconds": float(valid_seconds),
            "test_seconds": float(test_seconds),
            "test_molecules_per_second": float(test_audit["n"] / max(test_seconds, 1e-9)),
            "test_cache_seconds": float(test_cache_seconds),
            "test_cache_build_seconds": float(test_cache.get("build_seconds", 0.0)),
            "train_predict_seconds": None,
            "note": (
                "test_seconds excludes cache build; cache build is reported separately; "
                "both are inference-only and contain no training"
            ),
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
    }
    C.write_json(out_path, payload)
    C.log(
        f"[{dataset} P2] test {C.DATASETS[dataset]['primary_metric']}="
        f"{test_metrics.get('rmse', test_metrics.get('roc_auc')):.5f} -> {C.track_rel(out_path)}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2 one-shot Test evaluation")
    parser.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--force", action="store_true", help="not permitted for a formal one-shot result")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    evaluate(args.dataset, device=device, force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
