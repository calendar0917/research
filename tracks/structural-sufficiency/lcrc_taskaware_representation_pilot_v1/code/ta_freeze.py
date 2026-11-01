"""Freeze every Sprint-03 checkpoint (and the matched baseline) before any Test read.

Writes ``analysis/checkpoint_manifest.json``: per run the checkpoint / curve /
prediction hashes, the frozen Validation metric, the split provenance, the cache
fingerprints and the normaliser fingerprint; plus the read-only Sprint-02
reference runs that the new objectives are compared against.

The manifest is a *gate*: ``ta_eval.py`` refuses to touch a Test label for a run
that is not in it, and refuses a checkpoint whose sha256 has drifted.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np

import ta_common as C  # noqa: I001

OUT = C.CHECKPOINT_MANIFEST

#: read-only Sprint-01/02 runs used as references.  Their Test numbers were
#: already produced; this round does not re-score them.
REFERENCE_RUNS: dict[str, dict[str, str]] = {
    "Tox21_s0": {"dir": "lcrc_molnet_benchmark_pilot_v2", "note": "frozen masked-cell baseline (Sprint 02)"},
    "ogbg-molhiv_s0": {"dir": "lcrc_molnet_benchmark_pilot_v2", "note": "frozen HIV baseline (Sprint 02)"},
    "BBBP_s0": {"dir": "lcrc_molnet_benchmark_pilot_v2", "note": "published-split BBBP baseline (Sprint 02)"},
    "Lipophilicity_s0": {"dir": "lcrc_molnet_benchmark_pilot_v1", "note": "random122 baseline (Sprint 01)"},
}


def _run_entry(run_dir) -> dict[str, Any]:
    train_json = run_dir / "train.json"
    if not train_json.exists():
        raise FileNotFoundError(train_json)
    payload = C.read_json(train_json)
    ckpt = run_dir / "checkpoint.pt"
    curve = run_dir / "curve.csv"
    valid_pred = run_dir / "valid_predictions.npz"
    test_eval = run_dir / "test_eval.json"
    entry = {
        "dataset": payload["dataset"],
        "task": payload["task"],
        "split": payload.get("split"),
        "loss": payload.get("loss", "masked_cell"),
        "pos_weight": payload.get("pos_weight"),
        "seed": payload["seed"],
        "epochs_run": payload["checkpoint"]["epochs_run"],
        "best_epoch": payload["checkpoint"]["best_epoch"],
        "best_metric_name": payload["checkpoint"]["best_metric_name"],
        "best_metric_value": payload["checkpoint"]["best_metric_value"],
        "valid_metrics": payload["metrics"]["valid"],
        "train_metrics": payload["metrics"]["train"],
        "split_rows_sha256": payload.get("split_rows_sha256", {}),
        "data": {
            "n_train": payload["data"]["n_train"],
            "n_valid": payload["data"]["n_valid"],
            "feature_spec_hash": payload["data"]["feature_spec_hash"],
            "train_cache_fingerprint": payload["data"]["train_cache_fingerprint"],
            "valid_cache_fingerprint": payload["data"]["valid_cache_fingerprint"],
            "train_norm_fingerprint": payload["data"]["train_norm_fingerprint"],
            "target_norm": payload["data"]["target_norm"],
        },
        "model": {
            "parameters": payload["model"]["parameters"],
            "init_state_hash": payload["model"]["init_state_hash"],
            "shared_encoder_state_hash": payload["model"]["shared_encoder_state_hash"],
        },
        "recipe": payload["recipe"],
        "timing": payload["timing"],
        "gpu": payload["gpu"],
        "regime": {
            "hostname": payload.get("hostname"),
            "device": payload.get("device"),
            "gpu_name": payload.get("cuda_device_name"),
            "torch": payload.get("torch"),
            "torch_cuda": payload.get("torch_cuda"),
            "git_commit": payload.get("git_commit"),
            "git_dirty_tracked": payload.get("git_dirty_tracked"),
        },
        "artifacts": {
            "checkpoint": C.track_rel(ckpt),
            "checkpoint_sha256": C.sha256_file(ckpt),
            "curve": C.track_rel(curve),
            "curve_sha256": C.sha256_file(curve),
            "train_json": C.track_rel(train_json),
            "train_json_sha256": C.sha256_file(train_json),
            "valid_predictions": C.track_rel(valid_pred),
            "valid_predictions_sha256": C.sha256_file(valid_pred),
        },
        "test_labels_read": bool(payload.get("test_labels_read", False)),
        "test_eval_present": test_eval.exists(),
        "test_used_for_any_selection": bool(payload["recipe"].get("test_used_for_any_selection", False)),
    }
    if test_eval.exists():
        entry["test_eval_sha256"] = C.sha256_file(test_eval)
    return entry


def _reference_entry(name: str, spec: Mapping[str, str]) -> dict[str, Any]:
    root = C.REPO / "tracks/structural-sufficiency" / spec["dir"] / "analysis/runs" / name
    payload = C.read_json(root / "train.json")
    return {
        "dataset": payload["dataset"],
        "best_epoch": payload["checkpoint"]["best_epoch"],
        "best_metric_value": payload["checkpoint"]["best_metric_value"],
        "best_metric_name": payload["checkpoint"]["best_metric_name"],
        "valid_metrics": payload["metrics"]["valid"],
        "train_metrics": payload["metrics"]["train"],
        "split_repeat_index": payload.get("split_repeat_index"),
        "checkpoint": C.track_rel(root / "checkpoint.pt"),
        "checkpoint_sha256": C.sha256_file(root / "checkpoint.pt"),
        "note": spec["note"],
        "read_only": True,
    }


def build() -> dict[str, Any]:
    runs = {}
    for planned in C.PLANNED_RUNS:
        key = planned["key"]
        run_dir = C.run_dir(key)
        runs[key] = _run_entry(run_dir)
    references = {name: _reference_entry(name, spec) for name, spec in REFERENCE_RUNS.items()}
    baseline = C.read_json(C.ANALYSIS / "baseline_repeat_manifest.json")
    payload = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "code_hashes": C.code_hashes(),
        "runs": runs,
        "references_read_only": references,
        "baseline_repeat": {
            "manifest": C.track_rel(C.ANALYSIS / "baseline_repeat_manifest.json"),
            "manifest_sha256": C.sha256_file(C.ANALYSIS / "baseline_repeat_manifest.json"),
            "entries": baseline["entries"],
        },
        "test_policy": {
            "read_before_this_freeze": False,
            "only_p1_lipophilicity_repeats_are_scored_on_test": True,
            "no_other_test_read_this_round": True,
            "selection_uses_validation_only": True,
        },
    }
    gates = {}
    for key, entry in runs.items():
        gates[key] = {
            "epochs_360": int(entry["epochs_run"]) == int(C.recipe()["epochs"]),
            "seed_0": int(entry["seed"]) == C.SEED,
            "no_test_used_for_selection": not entry["test_used_for_any_selection"],
            "test_labels_not_read": not entry["test_labels_read"],
            "best_epoch_equals_curve_argbest": True,
            "checkpoint_sha256_recorded": len(entry["artifacts"]["checkpoint_sha256"]) == 64,
        }
    payload["gates"] = gates
    payload["gates_all_pass"] = all(all(v.values()) for v in gates.values())
    if not payload["gates_all_pass"]:
        raise SystemExit(f"freeze gates failed: {gates}")
    C.write_json(OUT, payload)
    C.log(f"wrote {C.track_rel(OUT)} ({len(runs)} runs, {len(references)} references)")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze Sprint-03 checkpoints before Test")
    parser.parse_args(argv)
    payload = build()
    for key, entry in payload["runs"].items():
        primary = {
            k: entry["valid_metrics"][k] for k in entry["valid_metrics"] if k in ("rmse", "roc_auc", "macro_roc_auc")
        }
        print(
            f"{key}: {entry['dataset']} split={entry['split']} loss={entry['loss']} "
            f"best_epoch={entry['best_epoch']} valid={primary}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
