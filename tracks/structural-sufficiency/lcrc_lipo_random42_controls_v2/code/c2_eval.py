"""Validation-only evaluation, paired bootstrap and inference benchmark.

This module never opens a Test label, Test prediction, Test metric, pre-test
manifest or post-lock analysis file.  Every reported number is recomputed from
a hash-verified Train/Validation artifact.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import c2_common as C
import c2_data as CD
import c2_models as CM
import c2_train as CT

import r42_common as V1C  # noqa: E402
import r42_data as D  # noqa: E402
import r42_labels as L  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402
import lcrc_data  # noqa: E402
import v2_models as V2M  # noqa: E402


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def _check_rows(rows: np.ndarray, expected: Sequence[int], label: str) -> None:
    arr = np.asarray(rows, dtype=np.int64)
    exp = np.asarray(list(expected), dtype=np.int64)
    if arr.shape != exp.shape:
        raise PermissionError(f"{label} row vector length mismatch: {arr.shape} != {exp.shape}")
    if C.rows_sha256_int64(arr) != C.rows_sha256_int64(exp):
        raise PermissionError(f"{label} row hash mismatch")
    if not np.array_equal(arr, exp):
        raise PermissionError(f"{label} row order mismatch")


def _verify_train_predictions(
    train_json_path: Path,
    *,
    y_train: np.ndarray,
    rows_train: Sequence[int],
    expected_metrics: Mapping[str, float],
) -> dict[str, Any]:
    payload = C.read_json(train_json_path)
    info = payload["predictions"]
    path = C.REPO / info["train_path"]
    actual = C.sha256_file(path)
    if actual != info["train_sha256"]:
        raise PermissionError(f"{path} hash differs from its manifest")
    data = _load_npz(path)
    rows = np.asarray(data["rows"], dtype=np.int64)
    pred = np.asarray(data["pred"], dtype=np.float64)
    _check_rows(rows, rows_train, "historical Train3360")
    metrics = MET.all_metrics(pred, y_train)
    for key in ("rmse", "mae", "r2"):
        if abs(float(metrics[key]) - float(expected_metrics[key])) > 1e-12:
            raise PermissionError(f"historical Train {key} does not match the frozen report")
    return {
        "path": C.track_rel(path),
        "sha256": actual,
        "metrics": metrics,
        "matches_frozen_report": True,
    }


def _historical_entry(
    *,
    name: str,
    run_key: str,
    pred_path: Path,
    pred_sha: str,
    checkpoint_sha: str,
    train_json_path: Path,
    params_nominal: int,
    width: int | None,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    rows_train: Sequence[int],
    rows_valid: Sequence[int],
    v1_metrics_row: Mapping[str, Any],
    v1_ledger_entry: Mapping[str, Any],
) -> dict[str, Any]:
    data = _load_npz(pred_path)
    rows = np.asarray(data["rows"], dtype=np.int64)
    pred = np.asarray(data["pred"], dtype=np.float64)
    _check_rows(rows, rows_valid, f"historical {name} Validation420")
    valid_metrics = MET.all_metrics(pred, y_valid)
    for key in ("rmse", "mae", "r2"):
        if abs(float(valid_metrics[key]) - float(v1_metrics_row["valid_metrics"][key])) > 1e-12:
            raise PermissionError(f"historical {name} Validation {key} drifted")
    train_entry = _verify_train_predictions(
        train_json_path,
        y_train=y_train,
        rows_train=rows_train,
        expected_metrics=v1_metrics_row["train_metrics"],
    )
    payload = C.read_json(train_json_path)
    train_block = payload["training"]
    return {
        "name": name,
        "role": "historical_reference_read_only",
        "run_key": run_key,
        "seed": int(payload["seed"]),
        "parameters_nominal": int(params_nominal),
        "parameters_effective": int(params_nominal),
        "parameters_dead": 0,
        "parameters_effective_basis": "no structurally dead inputs identified; historical run not re-audited for per-element gradient coverage",
        "width": width,
        "checkpoint_rule": str(train_block["checkpoint_rule"]),
        "best_epoch": train_block.get("best_epoch"),
        "epochs_run": int(train_block["epochs_run"]),
        "optimizer_steps": int(train_block["optimizer_steps"]),
        "soup_members": train_block.get("soup_members"),
        "train_metrics": train_entry["metrics"],
        "valid_metrics": valid_metrics,
        "train_predictions": train_entry,
        "valid_predictions_sha256": pred_sha,
        "checkpoint_sha256": checkpoint_sha,
        "wall_seconds": float(v1_ledger_entry.get("wall_seconds") or 0.0),
        "gpu_hours_charged": float(v1_ledger_entry.get("gpu_hours_charged") or 0.0),
        "peak_memory_allocated_bytes": v1_ledger_entry.get("peak_memory_allocated_bytes"),
        "inference_seconds_per_valid420_recorded": v1_ledger_entry.get("inference_seconds_per_valid420"),
        "recorded_inference_batch_size": int(payload["timing"]["inference_batch_size"]),
        "source_manifest": C.track_rel(train_json_path),
        "source_manifest_sha256": C.sha256_file(train_json_path),
    }


def _new_entry(
    *,
    run_key: str,
    expected_model: str,
    name: str,
    role: str,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    rows_train: Sequence[int],
    rows_valid: Sequence[int],
    ledger_entry: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = C.RUNS / run_key
    payload = C.read_json(run_dir / "train.json")
    if payload.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError(f"{run_key} protocol mismatch")
    if payload.get("model") != expected_model:
        raise PermissionError(f"{run_key} model mismatch: {payload.get('model')}")
    if payload.get("test_labels_read") is not False or payload.get("test_predictions_read") is not False:
        raise PermissionError(f"{run_key} is not certified Test-free")
    checkpoint_path = C.REPO / payload["checkpoint_path"]
    if C.sha256_file(checkpoint_path) != payload["checkpoint_sha256"]:
        raise PermissionError(f"{run_key} checkpoint hash differs from its manifest")
    valid_path = C.REPO / payload["predictions"]["valid_path"]
    if C.sha256_file(valid_path) != payload["predictions"]["valid_sha256"]:
        raise PermissionError(f"{run_key} Validation prediction hash differs from its manifest")
    train_path = C.REPO / payload["predictions"]["train_path"]
    if C.sha256_file(train_path) != payload["predictions"]["train_sha256"]:
        raise PermissionError(f"{run_key} Train prediction hash differs from its manifest")
    valid_npz = _load_npz(valid_path)
    rows = np.asarray(valid_npz["rows"], dtype=np.int64)
    pred = np.asarray(valid_npz["pred"], dtype=np.float64)
    _check_rows(rows, rows_valid, f"{run_key} Validation420")
    valid_metrics = MET.all_metrics(pred, y_valid)
    train_npz = _load_npz(train_path)
    tr_rows = np.asarray(train_npz["rows"], dtype=np.int64)
    tr_pred = np.asarray(train_npz["pred"], dtype=np.float64)
    _check_rows(tr_rows, rows_train, f"{run_key} Train3360")
    train_metrics = MET.all_metrics(tr_pred, y_train)
    for split_name, computed, frozen in (
        ("train", train_metrics, payload["metrics"]["train"]),
        ("valid", valid_metrics, payload["metrics"]["valid"]),
    ):
        for key in ("rmse", "mae", "r2"):
            if abs(float(computed[key]) - float(frozen[key])) > 1e-12:
                raise PermissionError(f"{run_key} {split_name} {key} does not match its manifest")
    audit = payload["model_audit"]
    if "parameters_total_nominal" in audit:
        nominal = int(audit["parameters_total_nominal"])
        effective = int(audit["parameters_effective"])
    else:
        nominal = int(audit["parameters_total"])
        effective = int(audit.get("parameters_effective", nominal))
    training = payload["training"]
    return {
        "name": name,
        "role": role,
        "run_key": run_key,
        "seed": int(payload["seed"]),
        "parameters_nominal": nominal,
        "parameters_effective": effective,
        "parameters_dead": int(nominal - effective),
        "parameters_effective_basis": "nominal minus structurally dead S2 input columns" if nominal != effective else "all parameters receive gradients",
        "width": audit.get("width"),
        "checkpoint_rule": str(training["checkpoint_rule"]),
        "best_epoch": training.get("best_epoch"),
        "epochs_run": int(training["epochs_run"]),
        "optimizer_steps": int(training["optimizer_steps"]),
        "soup_members": training.get("soup_members"),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "valid_predictions_sha256": payload["predictions"]["valid_sha256"],
        "train_predictions_sha256": payload["predictions"]["train_sha256"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "wall_seconds": float(payload["timing"]["train_wall_seconds"]),
        "gpu_hours_charged": float(ledger_entry.get("gpu_hours_charged") or 0.0),
        "peak_memory_allocated_bytes": payload["gpu"]["peak_memory_allocated_bytes"],
        "inference_seconds_per_valid420_recorded": payload["timing"]["inference_seconds_per_valid420"],
        "recorded_inference_batch_size": int(payload["timing"]["inference_batch_size"]),
        "source_manifest": C.track_rel(run_dir / "train.json"),
        "source_manifest_sha256": C.sha256_file(run_dir / "train.json"),
        "config_hash": payload["config_hash"],
        "s2_off_gates": payload.get("s2_off_gates"),
        "model_audit": audit,
    }


def build_validation_table() -> dict[str, Any]:
    split, labels = CD.load_frozen_split_and_labels()
    y_train = labels.y_train
    y_valid = labels.y_valid
    rows_train = split["rows"]["train"]
    rows_valid = split["rows"]["valid"]
    reuse = C.verify_history()
    v1_metrics = C.read_json(C.V1_METRICS_VALID)
    by_model = {row["model"]: row for row in v1_metrics["results"]}
    v1_ledger = C.read_json(C.V1_GPU_LEDGER)
    v1_entries = {entry["run_key"]: entry for entry in v1_ledger["entries"]}
    if not C.GPU_LEDGER.exists():
        raise PermissionError("new GPU ledger is missing; run ledger before evaluate")
    new_ledger = C.read_json(C.GPU_LEDGER)
    new_entries = {entry["run_key"]: entry for entry in new_ledger["entries"]}

    entries = [
        _historical_entry(
            name="LCRC-A-R3-scratch",
            run_key="lcrc_a_r3_s0",
            pred_path=C.V1_LCRC_VALID,
            pred_sha=C.V1_LCRC_VALID_SHA256,
            checkpoint_sha=C.V1_LCRC_CKPT_SHA256,
            train_json_path=C.V1_LCRC_TRAIN_JSON,
            params_nominal=int(by_model["LCRC-A-R3-scratch"]["parameters_total"]),
            width=None,
            y_train=y_train,
            y_valid=y_valid,
            rows_train=rows_train,
            rows_valid=rows_valid,
            v1_metrics_row=by_model["LCRC-A-R3-scratch"],
            v1_ledger_entry=v1_entries["lcrc_a_r3_s0"],
        ),
        _new_entry(
            run_key="s2off_s0",
            expected_model="LCRC-S2off-capacity-matched",
            name="LCRC-S2off-capacity-matched",
            role="new_control",
            y_train=y_train,
            y_valid=y_valid,
            rows_train=rows_train,
            rows_valid=rows_valid,
            ledger_entry=new_entries["s2off_s0"],
        ),
        _historical_entry(
            name="PNA-native",
            run_key="pna_native_s0",
            pred_path=C.V1_PNA_VALID,
            pred_sha=C.V1_PNA_VALID_SHA256,
            checkpoint_sha=C.V1_PNA_CKPT_SHA256,
            train_json_path=C.V1_PNA_TRAIN_JSON,
            params_nominal=int(by_model["PNA-native"]["parameters_total"]),
            width=256,
            y_train=y_train,
            y_valid=y_valid,
            rows_train=rows_train,
            rows_valid=rows_valid,
            v1_metrics_row=by_model["PNA-native"],
            v1_ledger_entry=v1_entries["pna_native_s0"],
        ),
        _new_entry(
            run_key="pna_small_s45_s0",
            expected_model="PNA-small",
            name="PNA-small",
            role="new_control",
            y_train=y_train,
            y_valid=y_valid,
            rows_train=rows_train,
            rows_valid=rows_valid,
            ledger_entry=new_entries["pna_small_s45_s0"],
        ),
    ]
    main_table = [
        {
            "model": row["name"],
            "role": row["role"],
            "seed": row["seed"],
            "parameters_nominal": row["parameters_nominal"],
            "parameters_effective": row["parameters_effective"],
            "width": row["width"],
            "checkpoint_rule": row["checkpoint_rule"],
            "best_epoch": row["best_epoch"],
            "epochs_run": row["epochs_run"],
            "optimizer_steps": row["optimizer_steps"],
            "valid_rmse": row["valid_metrics"]["rmse"],
            "valid_mae": row["valid_metrics"]["mae"],
            "valid_r2": row["valid_metrics"]["r2"],
            "train_rmse": row["train_metrics"]["rmse"],
            "train_mae": row["train_metrics"]["mae"],
            "train_r2": row["train_metrics"]["r2"],
            "gpu_hours_charged": row["gpu_hours_charged"],
        }
        for row in entries
    ]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "PASS",
        "scope": "train_and_validation_only",
        "units": "raw_logD",
        "test_labels_read": False,
        "test_predictions_read": False,
        "n_train": int(y_train.size),
        "n_valid": int(y_valid.size),
        "valid_rows_sha256": C.rows_sha256_int64(rows_valid),
        "models": entries,
        "main_table": main_table,
        "reuse_audit_sha256": C.sha256_file(C.REUSE_AUDIT) if C.REUSE_AUDIT.exists() else None,
        "v1_metrics_validation_sha256": C.V1_METRICS_VALID_SHA256,
        "new_gpu_ledger_sha256": C.sha256_file(C.GPU_LEDGER),
    }
    C.write_json(C.VALIDATION_TABLE, payload)
    return payload


def run_paired_bootstrap() -> dict[str, Any]:
    split, labels = CD.load_frozen_split_and_labels()
    rows_valid = split["rows"]["valid"]
    y = labels.y_valid
    lcrc = _load_npz(C.V1_LCRC_VALID)
    if C.sha256_file(C.V1_LCRC_VALID) != C.V1_LCRC_VALID_SHA256:
        raise PermissionError("historical LCRC validation prediction hash drifted")
    s2off_run = C.RUNS / "s2off_s0"
    s2off_payload = C.read_json(s2off_run / "train.json")
    s2off_path = C.REPO / s2off_payload["predictions"]["valid_path"]
    if C.sha256_file(s2off_path) != s2off_payload["predictions"]["valid_sha256"]:
        raise PermissionError("S2-off validation prediction hash drifted")
    s2off = _load_npz(s2off_path)
    for name, data in (("LCRC-A", lcrc), ("S2-off", s2off)):
        _check_rows(np.asarray(data["rows"], dtype=np.int64), rows_valid, f"{name} Validation420")
    p_lcrc = np.asarray(lcrc["pred"], dtype=np.float64)
    p_s2off = np.asarray(s2off["pred"], dtype=np.float64)
    e_lcrc = p_lcrc - y
    e_s2off = p_s2off - y
    sq_lcrc = e_lcrc**2
    sq_s2off = e_s2off**2
    abs_lcrc = np.abs(e_lcrc)
    abs_s2off = np.abs(e_s2off)

    def _rmse(err: np.ndarray) -> float:
        return float(np.sqrt(np.mean(err**2)))

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    n = int(y.size)
    delta_rmse = np.empty(C.BOOTSTRAP_RESAMPLES, dtype=np.float64)
    delta_mae = np.empty(C.BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for i in range(C.BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        delta_rmse[i] = _rmse(e_s2off[idx]) - _rmse(e_lcrc[idx])
        delta_mae[i] = float(np.mean(abs_s2off[idx]) - np.mean(abs_lcrc[idx]))
    observed_delta_rmse = _rmse(e_s2off) - _rmse(e_lcrc)
    observed_delta_mae = float(np.mean(abs_s2off) - np.mean(abs_lcrc))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "DESCRIPTIVE_ONLY",
        "scope": "validation420_paired_molecules",
        "test_labels_read": False,
        "test_predictions_read": False,
        "comparison": "LCRC-S2off-capacity-matched minus LCRC-A-R3-scratch",
        "sign_convention": "positive = S2-off has larger error than LCRC-A",
        "n_molecules": n,
        "valid_rows_sha256": C.rows_sha256_int64(rows_valid),
        "resamples": C.BOOTSTRAP_RESAMPLES,
        "seed": C.BOOTSTRAP_SEED,
        "generator": "numpy.random.default_rng(0)",
        "observed": {
            "lcrc_rmse": _rmse(e_lcrc),
            "s2off_rmse": _rmse(e_s2off),
            "delta_rmse": observed_delta_rmse,
            "lcrc_mae": float(np.mean(abs_lcrc)),
            "s2off_mae": float(np.mean(abs_s2off)),
            "delta_mae": observed_delta_mae,
            "mean_squared_error_change": float(np.mean(sq_s2off - sq_lcrc)),
            "mean_absolute_error_change": float(np.mean(abs_s2off - abs_lcrc)),
            "median_absolute_error_change": float(np.median(abs_s2off - abs_lcrc)),
            "s2off_better_count": int(np.sum(abs_s2off < abs_lcrc)),
            "s2off_worse_count": int(np.sum(abs_s2off > abs_lcrc)),
            "tie_count": int(np.sum(abs_s2off == abs_lcrc)),
            "squared_error_correlation": float(np.corrcoef(sq_lcrc, sq_s2off)[0, 1]),
        },
        "bootstrap_delta_rmse": {
            "mean": float(np.mean(delta_rmse)),
            "median": float(np.median(delta_rmse)),
            "ci95_percentile": [float(np.percentile(delta_rmse, 2.5)), float(np.percentile(delta_rmse, 97.5))],
            "p_delta_gt_zero": float(np.mean(delta_rmse > 0.0)),
            "p_delta_le_zero": float(np.mean(delta_rmse <= 0.0)),
            "min": float(np.min(delta_rmse)),
            "max": float(np.max(delta_rmse)),
        },
        "bootstrap_delta_mae": {
            "mean": float(np.mean(delta_mae)),
            "median": float(np.median(delta_mae)),
            "ci95_percentile": [float(np.percentile(delta_mae, 2.5)), float(np.percentile(delta_mae, 97.5))],
            "p_delta_gt_zero": float(np.mean(delta_mae > 0.0)),
            "p_delta_le_zero": float(np.mean(delta_mae <= 0.0)),
        },
        "interpretation_boundary": "descriptive interval only; single seed; no significance or mechanism claim",
    }
    C.write_json(C.PAIRED_BOOTSTRAP, payload)
    return payload


# ---------------------------------------------------------------------------
# inference benchmark
# ---------------------------------------------------------------------------


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise PermissionError(f"{path} has no state_dict")
    state = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return {
        "path": C.track_rel(path),
        "sha256": C.sha256_file(path),
        "checkpoint_keys": sorted(ckpt.keys()),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _chunks(n: int, batch_size: int) -> list[list[int]]:
    return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]


def _predict_chunks(model, chunks, prep: Callable[[list[int]], Any], forward: Callable[[Any], torch.Tensor]) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in chunks:
            batch = prep(chunk)
            pred = forward(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def _time_chunks(
    model,
    chunks,
    prep: Callable[[list[int]], Any],
    forward: Callable[[Any], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model.eval()

    def one_pass() -> tuple[float, float]:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        batches = [prep(chunk) for chunk in chunks]
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        with torch.no_grad():
            for batch in batches:
                forward(batch)
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    for _ in range(int(warmup)):
        one_pass()
    prep_times: list[float] = []
    fwd_times: list[float] = []
    e2e_times: list[float] = []
    for _ in range(int(repeats)):
        p, f = one_pass()
        prep_times.append(p)
        fwd_times.append(f)
        e2e_times.append(p + f)
    return {
        "n_repeats": int(repeats),
        "warmup_repeats": int(warmup),
        "n_validation_batches": len(chunks),
        "batch_sizes": [len(c) for c in chunks],
        "preprocessing_seconds_per_valid420": float(np.mean(prep_times)),
        "preprocessing_seconds_std": float(np.std(prep_times, ddof=0)),
        "preprocessing_seconds_per_batch": float(np.mean(prep_times) / len(chunks)),
        "forward_seconds_per_valid420": float(np.mean(fwd_times)),
        "forward_seconds_std": float(np.std(fwd_times, ddof=0)),
        "forward_seconds_per_batch": float(np.mean(fwd_times) / len(chunks)),
        "end_to_end_seconds_per_valid420": float(np.mean(e2e_times)),
        "end_to_end_seconds_std": float(np.std(e2e_times, ddof=0)),
        "raw_preprocessing_seconds": prep_times,
        "raw_forward_seconds": fwd_times,
        "raw_end_to_end_seconds": e2e_times,
        "preprocessing_definition": "collate/Batch construction + host-to-device transfer",
        "forward_definition": "model forward only; no loss, no backward",
    }


def run_inference_benchmark(*, batch_size: int = 128, warmup: int = 3, repeats: int = 10) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("inference benchmark requires CUDA")
    device = torch.device("cuda", int(torch.cuda.current_device()))
    torch.cuda.set_device(device)
    split, labels = CD.load_frozen_split_and_labels()
    rows_train = split["rows"]["train"]
    rows_valid = split["rows"]["valid"]
    if C.rows_sha256_int64(rows_valid) != C.SPLIT_SHA256["valid"]:
        raise PermissionError("Validation420 row hash drifted")
    views = CD.build_lcrc_views(split, labels)
    valid_cache = views["valid_cache"]
    spec = views["spec"]
    train_norm = views["train_norm"]
    train_data = D.load_structure_data(rows=rows_train)
    valid_data = D.load_structure_data(rows=rows_valid)
    if len(train_data) != C.SPLIT_SIZES["train"] or len(valid_data) != C.SPLIT_SIZES["valid"]:
        raise PermissionError("Train/Validation structure count mismatch")
    chunks = _chunks(len(rows_valid), batch_size)

    s2off_payload = C.read_json(C.RUNS / "s2off_s0/train.json")
    pna_payload = C.read_json(C.RUNS / "pna_small_s45_s0/train.json")
    pna_cfg = CM.pna_small_config()
    expected_degree_sha = "56bc94c1186a1a471b9af7885313ed914f51469f3e40fa99266a20c65721fc40"

    from torch_geometric.data import Batch

    def prep_lcrc(chunk: list[int]):
        batch = lcrc_data.collate(valid_cache, chunk)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def prep_pna(chunk: list[int]):
        return Batch.from_data_list([valid_data[i] for i in chunk]).to(device)

    entries: list[dict[str, Any]] = []

    # ---- historical LCRC-A -------------------------------------------------
    model_lcrc = V2M.prep_model.LCRC(spec, train_norm, V2M.prep_model.DEFAULT_CONFIG)
    ckpt = _load_checkpoint(model_lcrc, C.V1_LCRC_CKPT, device)
    torch.cuda.reset_peak_memory_stats(device)
    pred = _predict_chunks(model_lcrc, chunks, prep_lcrc, lambda b: model_lcrc(b))
    timing = _time_chunks(model_lcrc, chunks, prep_lcrc, lambda b: model_lcrc(b), device=device, warmup=warmup, repeats=repeats)
    peak = int(torch.cuda.max_memory_allocated(device))
    entries.append(
        _benchmark_entry(
            name="LCRC-A-R3-scratch",
            run_key="lcrc_a_r3_s0",
            role="historical_reference",
            model=model_lcrc,
            ckpt=ckpt,
            pred=pred,
            saved_pred_path=C.V1_LCRC_VALID,
            saved_pred_sha=C.V1_LCRC_VALID_SHA256,
            y_valid=labels.y_valid,
            rows_valid=rows_valid,
            timing=timing,
            peak=peak,
            precision="fp32",
            width=None,
        )
    )
    del model_lcrc
    torch.cuda.empty_cache()

    # ---- new S2-off --------------------------------------------------------
    model_s2off, _base, _audit = CM.build_s2off(spec, train_norm, C.TRAIN_SEED, device)
    s2off_ckpt = _load_checkpoint(model_s2off, C.REPO / s2off_payload["checkpoint_path"], device)
    torch.cuda.reset_peak_memory_stats(device)
    pred = _predict_chunks(model_s2off, chunks, prep_lcrc, lambda b: model_s2off(b))
    timing = _time_chunks(model_s2off, chunks, prep_lcrc, lambda b: model_s2off(b), device=device, warmup=warmup, repeats=repeats)
    peak = int(torch.cuda.max_memory_allocated(device))
    entries.append(
        _benchmark_entry(
            name="LCRC-S2off-capacity-matched",
            run_key="s2off_s0",
            role="new_control",
            model=model_s2off,
            ckpt=s2off_ckpt,
            pred=pred,
            saved_pred_path=C.REPO / s2off_payload["predictions"]["valid_path"],
            saved_pred_sha=s2off_payload["predictions"]["valid_sha256"],
            y_valid=labels.y_valid,
            rows_valid=rows_valid,
            timing=timing,
            peak=peak,
            precision="fp32",
            width=None,
        )
    )
    del model_s2off
    torch.cuda.empty_cache()

    # ---- historical PNA-native --------------------------------------------
    with tempfile.TemporaryDirectory(prefix="bench_pna_native_") as tmp_name:
        model_pna, extra_pna = NT.build_pna(
            V1C.PNA_CONFIG,
            device,
            Path(tmp_name),
            train_data,
            int(valid_data[0].x.shape[1]),
            int(valid_data[0].edge_attr.shape[1]),
        )
        if extra_pna.get("histogram_sha256") != expected_degree_sha:
            raise PermissionError("historical PNA degree histogram drifted")
        pna_ckpt = _load_checkpoint(model_pna, C.V1_PNA_CKPT, device)
        torch.cuda.reset_peak_memory_stats(device)
        pred = _predict_chunks(model_pna, chunks, prep_pna, lambda b: pna_forward(model_pna, b, device))
        timing = _time_chunks(
            model_pna,
            chunks,
            prep_pna,
            lambda b: pna_forward(model_pna, b, device),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        entries.append(
            _benchmark_entry(
                name="PNA-native",
                run_key="pna_native_s0",
                role="historical_reference",
                model=model_pna,
                ckpt=pna_ckpt,
                pred=pred,
                saved_pred_path=C.V1_PNA_VALID,
                saved_pred_sha=C.V1_PNA_VALID_SHA256,
                y_valid=labels.y_valid,
                rows_valid=rows_valid,
                timing=timing,
                peak=peak,
                precision="bf16",
                width=256,
            )
        )
        del model_pna
    torch.cuda.empty_cache()

    # ---- new PNA-small -----------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="bench_pna_small_") as tmp_name:
        model_small, extra_small = NT.build_pna(
            pna_cfg,
            device,
            Path(tmp_name),
            train_data,
            int(valid_data[0].x.shape[1]),
            int(valid_data[0].edge_attr.shape[1]),
        )
        if extra_small.get("histogram_sha256") != expected_degree_sha:
            raise PermissionError("PNA-small degree histogram drifted")
        small_ckpt = _load_checkpoint(model_small, C.REPO / pna_payload["checkpoint_path"], device)
        torch.cuda.reset_peak_memory_stats(device)
        pred = _predict_chunks(model_small, chunks, prep_pna, lambda b: pna_forward(model_small, b, device))
        timing = _time_chunks(
            model_small,
            chunks,
            prep_pna,
            lambda b: pna_forward(model_small, b, device),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        entries.append(
            _benchmark_entry(
                name="PNA-small",
                run_key="pna_small_s45_s0",
                role="new_control",
                model=model_small,
                ckpt=small_ckpt,
                pred=pred,
                saved_pred_path=C.REPO / pna_payload["predictions"]["valid_path"],
                saved_pred_sha=pna_payload["predictions"]["valid_sha256"],
                y_valid=labels.y_valid,
                rows_valid=rows_valid,
                timing=timing,
                peak=peak,
                precision="bf16",
                width=int(pna_cfg["gnn_intermediate_dim"]),
            )
        )
        del model_small
    torch.cuda.empty_cache()

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "PASS",
        "scope": "validation420_only",
        "test_labels_read": False,
        "test_predictions_read": False,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "count": int(torch.cuda.device_count()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        },
        "setup": {
            "batch_size": int(batch_size),
            "validation_n": int(len(rows_valid)),
            "validation_batches": [len(c) for c in chunks],
            "warmup_repeats": int(warmup),
            "timed_repeats": int(repeats),
            "same_device": True,
            "same_batch_size": True,
            "same_warmup": True,
            "forward_and_preprocessing_separated": True,
        },
        "models": entries,
        "env": C.env_info(),
    }
    C.write_json(C.INFERENCE_BENCHMARK, payload)
    return payload


def pna_forward(model, batch, device: torch.device):
    with NT._autocast(device, "bf16"):
        _, _, pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    return pred


def _benchmark_entry(
    *,
    name: str,
    run_key: str,
    role: str,
    model,
    ckpt: Mapping[str, Any],
    pred: np.ndarray,
    saved_pred_path: Path,
    saved_pred_sha: str,
    y_valid: np.ndarray,
    rows_valid: Sequence[int],
    timing: Mapping[str, Any],
    peak: int,
    precision: str,
    width: int | None,
) -> dict[str, Any]:
    actual_sha = C.sha256_file(saved_pred_path)
    if actual_sha != saved_pred_sha:
        raise PermissionError(f"{name} saved Validation prediction hash drifted")
    saved = _load_npz(saved_pred_path)
    _check_rows(np.asarray(saved["rows"], dtype=np.int64), rows_valid, f"{name} saved Validation420")
    saved_pred = np.asarray(saved["pred"], dtype=np.float64)
    max_diff = float(np.max(np.abs(pred - saved_pred))) if pred.size else 0.0
    mean_diff = float(np.mean(np.abs(pred - saved_pred))) if pred.size else 0.0
    bench_metrics = MET.all_metrics(pred, y_valid)
    saved_metrics = MET.all_metrics(saved_pred, y_valid)
    if max_diff > 5e-3:
        raise PermissionError(f"{name} benchmark predictions differ from the frozen Validation predictions by {max_diff}")
    return {
        "name": name,
        "run_key": run_key,
        "role": role,
        "parameters_nominal": int(sum(p.numel() for p in model.parameters())),
        "parameters_requires_grad": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "width": width,
        "precision": precision,
        "checkpoint": ckpt,
        "saved_validation_predictions": {
            "path": C.track_rel(saved_pred_path),
            "sha256": actual_sha,
            "max_abs_diff_vs_benchmark": max_diff,
            "mean_abs_diff_vs_benchmark": mean_diff,
            "saved_metrics": saved_metrics,
            "benchmark_metrics": bench_metrics,
            "metrics_match_within_1e_6": all(abs(bench_metrics[k] - saved_metrics[k]) <= 1e-6 for k in ("rmse", "mae", "r2")),
        },
        "timing": dict(timing),
        "peak_memory_allocated_bytes": int(peak),
        "device_name": torch.cuda.get_device_name(torch.device("cuda")),
    }


__all__ = [
    "build_validation_table",
    "run_paired_bootstrap",
    "run_inference_benchmark",
    "pna_forward",
]
