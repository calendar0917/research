"""Phase A: one frozen feature extraction (Train + Validation only).

Gates (any failure stops the round):

* the frozen checkpoint SHA-256 and metadata,
* the loaded state dict is bitwise equal to the checkpoint,
* every parameter is frozen and the model is in ``eval()``,
* the ``rho`` input width equals ``model.rho.lins[0].weight.shape[1]`` and the
  frozen ``[g | mean_h | mean_h2 | log1p(n)]`` layout,
* ``rho_in`` equals an independent reconstruction from the batch and ``h``,
* ``logits == contrib + rho`` exactly, and ``rho(rho_in)`` replays ``rho``,
* the replayed Validation AP reproduces ``AP0`` within ``1e-4``,
* the Train macro AP reproduces the frozen diagnostic within ``1e-4``,
* Train/Validation rows are complete, unique, disjoint, and the official Test
  targets are never finite / never indexed,
* the parameter fingerprint is unchanged after extraction.

Outputs: ``runs/feature_cache/`` (git-ignored arrays + manifest) and
``analysis/feature_extraction_audit.json`` / ``analysis/frozen_baseline_replay.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import fr_common as C  # noqa: E402
import fr_metrics as FM  # noqa: E402
import fr_model as MODEL  # noqa: E402
import lr_data as DATA  # noqa: E402
import pc_model as BM  # noqa: E402

CACHE_FILES = {
    "train_rho_in": "train_rho_in.npy",
    "valid_rho_in": "valid_rho_in.npy",
    "train_logits": "train_logits.npy",
    "valid_logits": "valid_logits.npy",
    "train_labels": "train_labels.npy",
    "valid_labels": "valid_labels.npy",
    "train_mask": "train_mask.npy",
    "valid_mask": "valid_mask.npy",
    "standardizer_mean": "standardizer_mean.npy",
    "standardizer_scale": "standardizer_scale.npy",
}


def _save_arrays(arrays: Mapping[str, np.ndarray], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for key, arr in arrays.items():
        path = out_dir / CACHE_FILES[key]
        np.save(path, np.ascontiguousarray(arr))
        records[key] = {
            "path": C.track_rel(path),
            "file": path.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": C.sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return records


def _row_checks(task: Any) -> dict[str, Any]:
    train_rows = np.asarray(task.train_rows, dtype=np.int64)
    valid_rows = np.asarray(task.valid_rows, dtype=np.int64)
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    labels = task.labels
    finite_test = int(np.isfinite(labels[test_rows]).sum())
    train_obs = int(np.isfinite(labels[train_rows]).sum())
    valid_obs = int(np.isfinite(labels[valid_rows]).sum())
    outside = int(np.isfinite(labels).sum() - train_obs - valid_obs)
    checks = {
        "n_train_rows": int(train_rows.size),
        "n_valid_rows": int(valid_rows.size),
        "n_test_rows": int(test_rows.size),
        "train_unique": int(np.unique(train_rows).size),
        "valid_unique": int(np.unique(valid_rows).size),
        "train_valid_overlap": int(np.intersect1d(train_rows, valid_rows).size),
        "train_test_overlap": int(np.intersect1d(train_rows, test_rows).size),
        "valid_test_overlap": int(np.intersect1d(valid_rows, test_rows).size),
        "train_observed_entries": train_obs,
        "valid_observed_entries": valid_obs,
        "test_finite_entries": finite_test,
        "finite_entries_outside_train_valid": outside,
        "train_rows_sha256": C.sha256_int64(train_rows),
        "valid_rows_sha256": C.sha256_int64(valid_rows),
        "test_rows_sha256": C.sha256_int64(test_rows),
    }
    gates = {
        "train_rows_complete": train_rows.size == C.N_TRAIN and checks["train_unique"] == C.N_TRAIN,
        "valid_rows_complete": valid_rows.size == C.N_VALID and checks["valid_unique"] == C.N_VALID,
        "train_valid_disjoint": checks["train_valid_overlap"] == 0,
        "test_disjoint": checks["train_test_overlap"] == 0 and checks["valid_test_overlap"] == 0,
        "test_labels_never_finite": finite_test == 0,
        "no_finite_labels_outside_train_valid": outside == 0,
        "observed_entries_match_data_audit": (
            train_obs == int(task.data_audit["train_observed_entries"])
            and valid_obs == int(task.data_audit["valid_observed_entries"])
        ),
    }
    return {"checks": checks, "gates": {k: bool(v) for k, v in gates.items()}}


def run(
    *,
    device: torch.device,
    out_dir: Path = C.FEATURE_CACHE,
    analysis_dir: Path = C.ANALYSIS,
    soft_seconds: float = float(C.BUDGET["extract_soft_seconds"]),
    batch_size: int = int(C.EXTRACT["batch_size"]),
) -> dict[str, Any]:
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    analysis_dir = Path(analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    inherited = C.assert_inherited_constants()
    task = DATA.load_task()
    model, model_audit = MODEL.load_frozen_model(task, device=device, label="frozen_extract")
    rho_dim = int(model_audit["rho_dimension"]["rho_input_dim"])
    params_before = MODEL.parameter_digests(model)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # ---- Validation first (small), then Train --------------------------------
    valid = MODEL.extract_split(
        model, task, task.valid_rows, device, split="valid", batch_size=batch_size, rho_dim=rho_dim
    )
    train = MODEL.extract_split(
        model, task, task.train_rows, device, split="train", batch_size=batch_size, rho_dim=rho_dim
    )
    extraction_seconds = float(valid["audit"]["seconds"] + train["audit"]["seconds"])

    # ---- independent replay through the historical inference path -----------
    # NOTE (calibration, recorded in PROTOCOL.md §1): the frozen forward uses
    # CUDA atomic ``index_add_`` reductions, so its logits are not bitwise
    # reproducible across calls (measured spread <= 8e-5 absolute; the g and
    # log1p(n) blocks are exactly reproducible, mean_h/mean_h2 are not).  The
    # replay is therefore gated on AP agreement (1e-5) plus a coarse logit
    # sanity bound (1e-3) that a real layout/path bug would exceed by orders of
    # magnitude.  The measured values are always recorded.
    replay = MODEL.replay_logits(model, task, task.valid_rows, device)
    extracted_valid = valid["logits"].cpu().numpy()
    replay_diff = float(np.abs(replay.astype(np.float64) - extracted_valid.astype(np.float64)).max())
    replay_probs = BM.sigmoid(replay)
    replay_ap = FM.official_ap(task.labels[task.valid_rows], replay_probs)["value"]
    replay = None  # release the duplicate validation logits

    # ---- frozen baseline replay ---------------------------------------------
    train_labels = np.ascontiguousarray(task.labels[task.train_rows])
    valid_labels = np.ascontiguousarray(task.labels[task.valid_rows])
    train_mask = np.isfinite(train_labels)
    valid_mask = np.isfinite(valid_labels)

    train_logits_np = train["logits"].cpu().numpy()
    valid_logits_np = extracted_valid
    train_probs = BM.sigmoid(train_logits_np)
    valid_probs = BM.sigmoid(valid_logits_np)

    t_metrics = time.perf_counter()
    official_valid = FM.official_ap(valid_labels, valid_probs)
    valid_tasks = FM.task_table(valid_labels, valid_probs)
    official_train = FM.official_ap(train_labels, train_probs)
    train_tasks = FM.task_table(train_labels, train_probs)
    train_macro = FM.macro_ap_from_tasks(train_tasks["per_task"])
    valid_macro = FM.macro_ap_from_tasks(valid_tasks["per_task"])
    metric_seconds = float(time.perf_counter() - t_metrics)

    train_bce = FM.masked_bce_numpy(train_logits_np, train_labels)
    valid_bce = FM.masked_bce_numpy(valid_logits_np, valid_labels)

    valid_ap = float(official_valid["value"])
    valid_delta = abs(valid_ap - C.AP0)
    train_delta = abs(train_macro - C.TRAIN_AP0)
    replay_ap_delta = abs(float(replay_ap) - valid_ap)

    # ---- Train-only standardiser -------------------------------------------
    stats = FM.fit_standardizer(train["rho_in"])
    layout = model_audit["rho_dimension"]["layout"]
    zero_cols = list(stats["zero_variance_columns"])

    def _columns_owner(idx: int) -> str:
        for name, (lo, hi) in layout.items():
            if int(lo) <= int(idx) < int(hi):
                return str(name)
        return "outside_layout"

    zero_detail = [{"column": int(c), "block": _columns_owner(int(c))} for c in zero_cols]
    mean_np = stats["mean"].cpu().numpy()
    scale_np = stats["scale"].cpu().numpy()

    # The cached arrays stay RAW; the transform is audited on a temporary copy so
    # Phase B can re-apply exactly the same frozen mean/scale from the manifest.
    standardized: dict[str, Any] = {}
    for name in ("train", "valid"):
        raw = {"train": train, "valid": valid}[name]
        tmp = torch.empty_like(raw["rho_in"])
        tmp.copy_(raw["rho_in"])
        FM.standardize_(tmp, stats["mean"], stats["scale"])
        standardized[name] = {
            "sha256": C.sha256_array(tmp.cpu().numpy()),
            "stats": FM.column_stats(tmp),
        }
        del tmp

    # ---- persist ------------------------------------------------------------
    arrays = {
        "train_rho_in": train["rho_in"].cpu().numpy(),
        "valid_rho_in": valid["rho_in"].cpu().numpy(),
        "train_logits": train_logits_np,
        "valid_logits": valid_logits_np,
        "train_labels": train_labels,
        "valid_labels": valid_labels,
        "train_mask": train_mask,
        "valid_mask": valid_mask,
        "standardizer_mean": mean_np,
        "standardizer_scale": scale_np,
    }
    records = _save_arrays(arrays, out_dir)
    del arrays

    rows = _row_checks(task)
    params_after = MODEL.parameter_digests(model)
    grads_present = any(p.grad is not None for p in model.parameters())
    param_gates = {
        "parameters_unchanged": params_after["parameter_fingerprint"]
        == params_before["parameter_fingerprint"]
        == model_audit["parameter_fingerprint"],
        "no_gradients_on_frozen_parameters": not grads_present,
        "all_parameters_require_grad_false": not any(p.requires_grad for p in model.parameters()),
        "model_still_eval": not bool(model.training),
    }

    gates: dict[str, bool] = {
        "checkpoint_bitwise_load": bool(model_audit["checkpoint_load_audit"]["state_dict_bitwise_equal"]),
        "checkpoint_sha256": True,
        "rho_dimension_matches_model": bool(
            model_audit["rho_dimension"]["checks"]["layout_total_matches_rho_first_layer"]
            and rho_dim == C.RHO_IN_DIM_EXPECTED
        ),
        "rho_in_reconstruction": bool(train["audit"]["rho_reconstruction_gate"] and valid["audit"]["rho_reconstruction_gate"]),
        "rho_replay": bool(train["audit"]["rho_replay_gate"] and valid["audit"]["rho_replay_gate"]),
        "logits_identity_contrib_plus_rho": bool(
            train["audit"]["logits_identity_gate"] and valid["audit"]["logits_identity_gate"]
        ),
        "independent_replay_matches_extracted_logits": bool(replay_diff <= C.REPLAY_LOGIT_TOL),
        "independent_replay_matches_valid_ap": bool(replay_ap_delta <= C.REPLAY_AP_TOL),
        "valid_ap_reproduces_ap0": bool(valid_delta <= C.VALID_AP_TOL),
        "train_ap_reproduces_frozen_diagnostic": bool(train_delta <= C.TRAIN_AP_TOL),
        "standardizer_zero_variance_safe": bool(
            all(float(scale_np[c]) == 1.0 for c in zero_cols)
        ),
        "saved_raw_arrays_match_extraction_hashes": bool(
            records["train_rho_in"]["sha256"] == train["audit"]["rho_in_sha256"]
            and records["valid_rho_in"]["sha256"] == valid["audit"]["rho_in_sha256"]
            and records["train_logits"]["sha256"] == train["audit"]["logits_sha256"]
            and records["valid_logits"]["sha256"] == valid["audit"]["logits_sha256"]
        ),
        "standardizer_fit_rows": int(stats["n_rows_fit"]) == C.N_TRAIN,
        "train_valid_disjoint": rows["gates"]["train_valid_disjoint"],
        "test_never_finite_or_indexed": rows["gates"]["test_labels_never_finite"]
        and rows["gates"]["test_disjoint"]
        and rows["gates"]["no_finite_labels_outside_train_valid"],
        "rows_complete": rows["gates"]["train_rows_complete"] and rows["gates"]["valid_rows_complete"],
        "observed_entries": rows["gates"]["observed_entries_match_data_audit"],
        "frozen_parameters_unchanged": all(param_gates.values()),
        "within_soft_deadline": bool(time.perf_counter() - t_start <= float(soft_seconds)),
        "cache_files_written": all(Path(r["path"]).exists() for r in records.values()),
        "test_evaluated_false": True,
    }
    gates = {k: bool(v) for k, v in gates.items()}
    all_passed = bool(all(gates.values()))

    manifest = {
        "schema": "lcrc-pcba-frozen-readout-feature-cache-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
        "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
        "source_epoch": int(C.SOURCE["best_epoch"]),
        "feature_spec_hash": str(task.spec.hash()),
        "norm_fingerprint": str(task.norm["norm_fingerprint"]),
        "split_partition_sha256": str(task.split_partition_sha256),
        "label_artifact_sha256": str(task.label_artifact_sha256),
        "rho_in_dim": int(rho_dim),
        "rho_layout": layout,
        "n_tasks": int(C.N_TASKS),
        "arrays": records,
        "row_order": {
            "train": "official OGB Train rows, ascending order preserved from pcba_split.json",
            "valid": "official OGB Validation rows, ascending order preserved from pcba_split.json",
            "train_rows_sha256": rows["checks"]["train_rows_sha256"],
            "valid_rows_sha256": rows["checks"]["valid_rows_sha256"],
            "train_rho_in_rows_sha256": train["audit"]["rows_sha256"],
            "valid_rho_in_rows_sha256": valid["audit"]["rows_sha256"],
        },
        "cache_content_fingerprints": {
            "train_rho_in_raw_sha256": train["audit"]["rho_in_sha256"],
            "valid_rho_in_raw_sha256": valid["audit"]["rho_in_sha256"],
            "train_logits_sha256": train["audit"]["logits_sha256"],
            "valid_logits_sha256": valid["audit"]["logits_sha256"],
            "train_rho_in_standardized_sha256": standardized["train"]["sha256"],
            "valid_rho_in_standardized_sha256": standardized["valid"]["sha256"],
            "standardizer_mean_sha256": C.sha256_array(mean_np),
            "standardizer_scale_sha256": C.sha256_array(scale_np),
            "cached_arrays_are": "raw rho_in (before standardisation)",
        },
        "standardizer": {
            "fit_split": "train_only",
            "n_rows_fit": int(stats["n_rows_fit"]),
            "ddof": 0,
            "accumulation": stats["accumulation"],
            "zero_variance_epsilon": float(stats["zero_variance_epsilon"]),
            "n_zero_variance_columns": int(stats["n_zero_variance_columns"]),
            "zero_variance_columns": zero_detail,
            "valid_labels_used": False,
            "raw_min_std": float(stats["raw_min_std"]),
            "raw_max_std": float(stats["raw_max_std"]),
        },
        "test_evaluated": False,
        "all_passed": all_passed,
    }
    manifest_path = out_dir / "manifest.json"
    C.write_json(manifest_path, manifest)
    manifest_sha = C.sha256_file(manifest_path)

    extraction_audit = {
        "schema": "lcrc-pcba-frozen-readout-extraction-audit-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "phase": "A_frozen_feature_extraction",
        "all_passed": all_passed,
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not v],
        "inherited": inherited,
        "environment": C.environment(),
        "gpu_occupancy_start": C.gpu_occupancy(),
        "device": str(device),
        "model": model_audit,
        "parameter_fingerprints": {
            "before_extraction": params_before,
            "after_extraction": params_after,
            "gates": param_gates,
            "gradients_present_after_extraction": bool(grads_present),
        },
        "row_checks": rows,
        "extraction": {
            "train": train["audit"],
            "valid": valid["audit"],
            "batch_size": int(batch_size),
            "inference_mode": True,
            "model_eval": True,
            "requires_grad": False,
            "hook": "forward_pre_hook on model.rho (read-only, removed after extraction)",
            "independent_replay_path": "pc_model.predict_logits (batch 256)",
            "independent_replay_max_abs_logit_diff": replay_diff,
            "independent_replay_logit_tolerance": float(C.REPLAY_LOGIT_TOL),
            "independent_replay_logit_gate": bool(replay_diff <= C.REPLAY_LOGIT_TOL),
            "independent_replay_valid_ap": float(replay_ap),
            "independent_replay_valid_ap_delta": replay_ap_delta,
            "independent_replay_ap_tolerance": float(C.REPLAY_AP_TOL),
            "replay_nondeterminism_note": (
                "cross-call logit spread is intrinsic to the frozen encoder's CUDA "
                "index_add_ reductions; the gate is calibrated in "
                "runs/replay_diag/replay_diag.json and PROTOCOL.md §1"
            ),
            "seconds": extraction_seconds,
            "metric_seconds": metric_seconds,
        },
        "standardizer": manifest["standardizer"],
        "standardized_cache_audit": {
            "train": standardized["train"]["stats"],
            "valid": standardized["valid"]["stats"],
            "note": "statistics of the temporary standardised copy; the cached arrays stay raw",
        },
        "cache": {
            "dir": C.track_rel(out_dir),
            "manifest": C.track_rel(manifest_path),
            "manifest_sha256": manifest_sha,
            "arrays": records,
            "total_bytes": int(sum(int(r["bytes"]) for r in records.values())),
        },
        "gpu_peak_during_extraction": C.gpu_peak(device),
        "gpu_seconds": float(time.perf_counter() - t_start),
        "gpu_hours": float((time.perf_counter() - t_start) / 3600.0),
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(analysis_dir / "feature_extraction_audit.json", extraction_audit)

    baseline_replay = {
        "schema": "lcrc-pcba-frozen-readout-baseline-replay-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "phase": "A_frozen_baseline_replay",
        "all_passed": all_passed,
        "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
        "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
        "ap0": float(C.AP0),
        "train_ap0_reference": float(C.TRAIN_AP0),
        "valid": {
            "official_ap": valid_ap,
            "official_ap_source": official_valid["source"],
            "independent_macro_ap": valid_macro,
            "official_minus_independent": float(valid_ap - valid_macro),
            "abs_delta_vs_ap0": float(valid_delta),
            "tolerance": float(C.VALID_AP_TOL),
            "reproduced": bool(valid_delta <= C.VALID_AP_TOL),
            "masked_bce": valid_bce,
            "n_tasks_scored": int(valid_tasks["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(valid_tasks["n_tasks_with_observed_labels"]),
            "per_task_ap": [None if v is None else float(v) for v in valid_tasks["per_task"]],
        },
        "train": {
            "official_ap": float(official_train["value"]),
            "official_ap_source": official_train["source"],
            "macro_ap": float(train_macro),
            "macro_ap_reference": float(C.TRAIN_AP0),
            "abs_delta_vs_reference": float(train_delta),
            "tolerance": float(C.TRAIN_AP_TOL),
            "reproduced": bool(train_delta <= C.TRAIN_AP_TOL),
            "masked_bce": train_bce,
            "n_tasks_scored": int(train_tasks["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(train_tasks["n_tasks_with_observed_labels"]),
            "per_task_ap": [None if v is None else float(v) for v in train_tasks["per_task"]],
            "task_rows": [
                {
                    "task": int(t),
                    "observed": int(train_tasks["observed"][t]),
                    "positive": int(train_tasks["positive"][t]),
                    "negative": int(train_tasks["negative"][t]),
                    "ap": None if train_tasks["per_task"][t] is None else float(train_tasks["per_task"][t]),
                }
                for t in range(C.N_TASKS)
            ],
        },
        "train_minus_valid_ap": float(train_macro - valid_ap),
        "cache": {
            "manifest": C.track_rel(manifest_path),
            "manifest_sha256": manifest_sha,
            "train_logits_sha256": train["audit"]["logits_sha256"],
            "valid_logits_sha256": valid["audit"]["logits_sha256"],
        },
        "row_checks": rows["checks"],
        "gpu_seconds": extraction_audit["gpu_seconds"],
        "gpu_hours": extraction_audit["gpu_hours"],
        "gpu_peak": extraction_audit["gpu_peak_during_extraction"],
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(analysis_dir / "frozen_baseline_replay.json", baseline_replay)

    C.log(
        f"phase A all_passed={all_passed} validAP={valid_ap:.10f} (delta {valid_delta:.3e}) "
        f"trainAP={train_macro:.10f} (delta {train_delta:.3e}) "
        f"rho_dim={rho_dim} zero_var_cols={stats['n_zero_variance_columns']} "
        f"gpu_hours={extraction_audit['gpu_hours']:.6f}"
    )
    return {"all_passed": all_passed, "audit": extraction_audit, "baseline": baseline_replay}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(C.FEATURE_CACHE))
    parser.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    parser.add_argument("--batch-size", type=int, default=int(C.EXTRACT["batch_size"]))
    parser.add_argument("--soft-seconds", type=float, default=float(C.BUDGET["extract_soft_seconds"]))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            Path(args.analysis_dir) / "feature_extraction_audit.json",
            {
                "schema": "lcrc-pcba-frozen-readout-extraction-audit-v1",
                "round_id": C.ROUND_ID,
                "phase": "A_frozen_feature_extraction",
                "all_passed": False,
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        return 2
    try:
        result = run(
            device=device,
            out_dir=Path(args.out_dir),
            analysis_dir=Path(args.analysis_dir),
            soft_seconds=float(args.soft_seconds),
            batch_size=int(args.batch_size),
        )
        return 0 if result["all_passed"] else 3
    except Exception as exc:  # noqa: BLE001 - the failure is the deliverable
        C.write_json(
            Path(args.analysis_dir) / "feature_extraction_audit.json",
            {
                "schema": "lcrc-pcba-frozen-readout-extraction-audit-v1",
                "round_id": C.ROUND_ID,
                "phase": "A_frozen_feature_extraction",
                "all_passed": False,
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"phase A FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
