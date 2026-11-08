"""One-shot pre-training audit and the frozen lambda_rank calibration.

This is the only code path that estimates the relative scale of the BCE and
ranking gradients.  It runs **before** the formal trajectory, on the
historical W2 best-Validation checkpoint (epoch 24), on 16 pre-fixed official
**Train** batches.  It never initialises the formal model, never updates a
model, never opens Validation/Test labels for the ratio, and never draws from
the global RNG (model is in ``eval`` mode).

What it writes:

* ``analysis/calibration.json`` -- the frozen ``lambda_rank`` and every input
  to it (batch row hashes, per-batch gradient norms, medians, clipping flag).
* ``analysis/preflight.json`` -- all pre-training checks (W2 identity, epoch-0
  parity, loss properties, lambda=0 equivalence, resume, timing/memory
  projection).

If any check fails, or fewer than 8 calibration batches have a valid ranking
term, the process exits non-zero and the formal training must not be started.
"""

from __future__ import annotations

import argparse
import math
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

import tr_common as C  # noqa: E402
import tr_loss as L  # noqa: E402
import tr_train as T  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

TOL_EPOCH0 = 1e-4
TOL_RESUME = 1e-6


def _grad_norm(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).pow(2).sum().sqrt())


def _state_diff(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    worst = 0.0
    worst_key = None
    mismatch = 0
    for key in keys:
        if key not in a or key not in b:
            mismatch += 1
            continue
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            mismatch += 1
            continue
        if not torch.equal(a[key], b[key]):
            mismatch += 1
            diff = float((a[key].float() - b[key].float()).abs().max())
            if diff > worst:
                worst, worst_key = diff, key
    return {
        "keys": len(keys),
        "mismatched_tensors": int(mismatch),
        "max_abs_diff": float(worst),
        "worst_key": worst_key,
        "exact": bool(mismatch == 0),
    }


def _loss_gradients(logits: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    z = logits.detach().clone().requires_grad_(True)
    bce = L.masked_bce_loss(z, y)
    g_bce = torch.autograd.grad(bce, z)[0]
    rank, stats = L.rank_loss_parts(z, y)
    g_rank = torch.autograd.grad(rank, z)[0]
    return {
        "loss_bce": float(bce.detach()),
        "loss_rank": float(rank.detach()),
        "g_bce": _grad_norm(g_bce),
        "g_rank": _grad_norm(g_rank),
        "n_active": float(stats["n_active_tasks"]),
        "n_no_positive": float(stats["n_no_positive_tasks"]),
        "n_no_negative": float(stats["n_no_negative_tasks"]),
        "implicit_pairs": float(stats["implicit_pairs"]),
    }


def _load_w2_checkpoint() -> tuple[dict[str, Any], str]:
    path = Path(C.W2["checkpoint_best"])
    if not path.is_file():
        raise RuntimeError(f"historical W2 checkpoint missing: {path}")
    sha = C.sha256_file(path)
    if sha != C.W2["checkpoint_best_sha256"]:
        raise RuntimeError(f"W2 checkpoint sha {sha} != frozen {C.W2['checkpoint_best_sha256']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != C.MODEL_KEY:
        raise RuntimeError(f"W2 checkpoint model {payload.get('model')}")
    if int(payload.get("epoch", -1)) != int(C.W2["best_epoch"]):
        raise RuntimeError(f"W2 checkpoint epoch {payload.get('epoch')} != {C.W2['best_epoch']}")
    if abs(float(payload.get("valid_ap", float("nan"))) - C.W2["best_valid_ap"]) > 1e-9:
        raise RuntimeError("W2 checkpoint best Valid AP drifted")
    return payload, sha


def run_preflight(device: torch.device, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    task = C.load_task_gated()
    plan = C.make_plan40(task)
    calib = C.calibration_batches(task, plan)
    combined_rows_sha = calib[0]["combined_rows_sha256"]

    rng_initial = C.capture_rng_state()
    rng_initial_hashes = C.rng_state_hashes(rng_initial)
    occupancy_start = C.gpu_occupancy()

    # -- 1. W2 identity -----------------------------------------------------
    model, audit = C.build_w2(task, device=device)
    state = C.state_cpu(model)
    init_state_sha = C.state_sha256(state)
    rebuild, rebuild_audit = C.build_w2(task, device="cpu")
    rebuild_state = C.state_cpu(rebuild)
    del rebuild
    init_diff = _state_diff(state, rebuild_state)
    checkpoint, checkpoint_sha = _load_w2_checkpoint()
    history_keys = list(checkpoint["model_audit"]["state_keys"])
    init_checks = {
        "parameters_exact": int(audit["total_parameters"]) == C.MODEL_PARAMETERS
        and int(C.parameter_count(model)) == C.MODEL_PARAMETERS,
        "state_sha256_matches_history": init_state_sha == C.W2["init_state_sha256"],
        "trunk_fingerprint_matches_history": audit["trunk_init_fingerprint"]
        == C.W2["trunk_init_fingerprint"],
        "full_trunk_fingerprint_matches_history": audit["full_trunk_init_fingerprint"]
        == C.W2["full_trunk_init_fingerprint"],
        "state_keys_match_history": sorted(state) == sorted(history_keys),
        "tensorwise_matches_historical_builder": bool(init_diff["exact"]),
        "width_audit_all_match": bool(audit["width_audit"]["all_match"]),
    }
    init_audit = {
        "parameters": int(audit["total_parameters"]),
        "expected_parameters": C.MODEL_PARAMETERS,
        "base_parameters": int(audit["base_parameters"]),
        "readout_parameters": int(audit["readout_parameters"]),
        "n_state_tensors": len(state),
        "n_parameter_tensors": int(audit["parameter_tensors"]),
        "state_sha256": init_state_sha,
        "historical_state_sha256": C.W2["init_state_sha256"],
        "trunk_init_fingerprint": audit["trunk_init_fingerprint"],
        "historical_trunk_init_fingerprint": C.W2["trunk_init_fingerprint"],
        "full_trunk_init_fingerprint": audit["full_trunk_init_fingerprint"],
        "historical_full_trunk_init_fingerprint": C.W2["full_trunk_init_fingerprint"],
        "tensorwise_vs_historical_builder": init_diff,
        "state_keys_match_historical_checkpoint": init_checks["state_keys_match_history"],
        "width_audit_all_match": bool(audit["width_audit"]["all_match"]),
        "widths": audit["widths"],
        "checks": init_checks,
        "all_pass": bool(all(init_checks.values())),
    }

    # -- 2. forward unchanged ----------------------------------------------
    stager = PinnedStager(device, depth=2)
    first_rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)
    first_batch = stager.batch(C.BC.PC.collate(task.cache, first_rows.tolist()))
    first_y = stager.labels(task.labels[first_rows])
    calls = C.BSM.count_module_calls(model, first_batch)
    with torch.no_grad():
        logits_first = model(first_batch)
    forward_checks = {
        "output_shape": tuple(logits_first.shape) == (int(first_y.shape[0]), C.N_TASKS),
        "one_atom_encode": calls["atom_encoder"] == 1,
        "one_bond_encode": calls["bond_encoder"] == 1,
        "one_local": calls["local"] == 1,
        "one_f": calls["f"] == 1,
        "one_rho": calls["rho"] == 1,
        "one_readout": calls["readout"] == 1,
        "finite_logits": bool(torch.isfinite(logits_first).all()),
    }
    forward_audit = {
        "module_calls": calls,
        "output_shape": list(logits_first.shape),
        "checks": forward_checks,
        "all_pass": bool(all(forward_checks.values())),
        "forward_source_sha256": C.sha256_bytes(
            __import__("inspect").getsource(C.BSM.BSModel.forward).encode()
        ),
        "encode_source_sha256": C.sha256_bytes(
            __import__("inspect").getsource(C.BSM.BSModel.encode_h).encode()
        ),
        "post_local_source_sha256": C.sha256_bytes(
            __import__("inspect").getsource(C.BSM.BSModel.post_local).encode()
        ),
    }

    # -- 3. one-shot calibration on the historical W2 checkpoint -----------
    cal_model, _cal_audit = C.build_w2(task, device=device)
    missing, unexpected = cal_model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"W2 checkpoint load mismatch: {missing} {unexpected}")
    cal_model.eval()
    rng_before = C.capture_rng_state()
    rng_before_hashes = C.rng_state_hashes(rng_before)
    records: list[dict[str, Any]] = []
    for record in calib:
        rows = record["rows"]
        batch = stager.batch(C.BC.PC.collate(task.cache, rows.tolist()))
        y = stager.labels(task.labels[rows])
        with torch.no_grad():
            logits = cal_model(batch)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("non-finite calibration logits")
        values = _loss_gradients(logits, y)
        ratio = values["g_bce"] / values["g_rank"] if values["g_rank"] > 0 else float("inf")
        records.append(
            {
                "batch_index": int(record["batch_index"]),
                "rows_sha256": record["rows_sha256"],
                "n_rows": int(record["n_rows"]),
                **values,
                "ratio_g_bce_over_g_rank": float(ratio),
            }
        )
    valid = [
        r
        for r in records
        if r["n_active"] > 0
        and math.isfinite(r["g_rank"])
        and r["g_rank"] > 0
        and math.isfinite(r["g_bce"])
        and r["g_bce"] > 0
    ]
    g_bce_median = float(np.median([r["g_bce"] for r in valid])) if valid else float("nan")
    g_rank_median = float(np.median([r["g_rank"] for r in valid])) if valid else float("nan")
    ratio_raw = (
        float(C.CALIBRATION["factor"]) * g_bce_median / g_rank_median
        if g_rank_median > 0
        else float("nan")
    )
    lambda_rank = float(
        min(
            max(ratio_raw, float(C.CALIBRATION["lambda_min"])),
            float(C.CALIBRATION["lambda_max"]),
        )
    )
    clipped = bool(
        lambda_rank != ratio_raw or ratio_raw < float(C.CALIBRATION["lambda_min"]) or ratio_raw > float(C.CALIBRATION["lambda_max"])
    )
    if ratio_raw < float(C.CALIBRATION["lambda_min"]):
        clip_side = "min"
    elif ratio_raw > float(C.CALIBRATION["lambda_max"]):
        clip_side = "max"
    else:
        clip_side = "none"
    calibration_ok = bool(
        len(valid) >= int(C.CALIBRATION["min_valid_batches"])
        and math.isfinite(g_bce_median)
        and g_bce_median > 0
        and math.isfinite(g_rank_median)
        and g_rank_median > 0
        and math.isfinite(lambda_rank)
        and float(C.CALIBRATION["lambda_min"]) <= lambda_rank <= float(C.CALIBRATION["lambda_max"])
    )

    rng_after_cal = C.capture_rng_state()
    rng_after_cal_hashes = C.rng_state_hashes(rng_after_cal)
    rng_unchanged = (
        rng_after_cal_hashes["combined_sha256"] == rng_before_hashes["combined_sha256"]
    )
    del cal_model

    calibration_payload: dict[str, Any] = {
        "schema": C.SCHEMA_CALIBRATION,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "status": "COMPUTED",
        "model": C.MODEL_KEY,
        "model_mode": C.CALIBRATION["model_mode"],
        "checkpoint": {
            "path": C.track_rel(Path(C.W2["checkpoint_best"])),
            "sha256": checkpoint_sha,
            "epoch": int(checkpoint["epoch"]),
            "valid_ap": float(checkpoint["valid_ap"]),
            "selection_role": "historical W2 best-Validation checkpoint, used only for gradient scale",
        },
        "n_batches": int(len(calib)),
        "batch_indices": [int(r["batch_index"]) for r in calib],
        "batch_index_rule": C.CALIBRATION["batch_index_rule"],
        "combined_rows_sha256": combined_rows_sha,
        "batches": records,
        "valid_batches": int(len(valid)),
        "min_valid_batches": int(C.CALIBRATION["min_valid_batches"]),
        "g_bce_median": g_bce_median,
        "g_rank_median": g_rank_median,
        "factor": float(C.CALIBRATION["factor"]),
        "ratio_raw": ratio_raw,
        "lambda_min": float(C.CALIBRATION["lambda_min"]),
        "lambda_max": float(C.CALIBRATION["lambda_max"]),
        "lambda_rank": lambda_rank,
        "clipped": clipped,
        "clip_side": clip_side,
        "calibration_valid": calibration_ok,
        "gradient_norm": C.CALIBRATION["gradient_norm"],
        "train_only": True,
        "test_labels_read": False,
        "rng_unchanged": bool(rng_unchanged),
        "rng_hashes_before_calibration": rng_before_hashes,
        "rng_hashes_after_calibration": rng_after_cal_hashes,
        "rng_hashes_at_process_start": rng_initial_hashes,
        "rng_note": (
            "The pre-calibration hash is taken after building the frozen W2 modules; "
            "module construction itself consumes torch RNG in the historical code, "
            "exactly as in W2/S3/SP. Formal training calls seed_everything(0) after "
            "construction, so the trajectory RNG start is unchanged; the calibration "
            "loop itself must not advance any RNG."
        ),
    }
    calibration_path = out_dir / "calibration.json"
    C.write_json(calibration_path, calibration_payload)
    calibration_sha = C.sha256_file(calibration_path)

    # -- 4. epoch-0 parity --------------------------------------------------
    model.eval()
    rng_before_epoch0 = C.rng_state_hashes(C.capture_rng_state())
    t0 = time.perf_counter()
    with torch.inference_mode():
        report0 = C.BC.SD.evaluate_validation(
            model, task, device, reason="preflight_epoch_0_initialisation"
        )
    epoch0_seconds = float(time.perf_counter() - t0)
    rng_after_epoch0 = C.rng_state_hashes(C.capture_rng_state())
    epoch0 = {
        "valid_ap": float(report0["ap"]),
        "historical_valid_ap": C.W2["epoch0_valid_ap"],
        "valid_ap_abs_delta": abs(float(report0["ap"]) - C.W2["epoch0_valid_ap"]),
        "valid_bce": float(report0["valid_bce"]),
        "historical_valid_bce": C.W2["epoch0_valid_bce"],
        "valid_bce_abs_delta": abs(float(report0["valid_bce"]) - C.W2["epoch0_valid_bce"]),
        "valid_logits_sha256": report0.get("valid_logits_sha256"),
        "n_tasks_scored": int(report0["n_tasks_scored"]),
        "seconds": epoch0_seconds,
        "rng_unchanged": rng_after_epoch0["combined_sha256"]
        == rng_before_epoch0["combined_sha256"],
        "tolerance": TOL_EPOCH0,
    }
    epoch0["checks"] = {
        "ap_matches_history": epoch0["valid_ap_abs_delta"] <= TOL_EPOCH0,
        "bce_matches_history": epoch0["valid_bce_abs_delta"] <= TOL_EPOCH0,
        "rng_unchanged": epoch0["rng_unchanged"],
    }
    epoch0["all_pass"] = bool(all(epoch0["checks"].values()))

    # -- 5. loss property self-check ---------------------------------------
    loss_check = L.self_check()

    # -- 6. lambda=0 is exactly the historical BCE step ---------------------
    state0 = C.state_cpu(model)
    rng0 = C.capture_rng_state()

    def _one_branch(use_total: bool) -> dict[str, Any]:
        model.load_state_dict(state0, strict=True)
        C.restore_rng_state(rng0)
        model.train()
        optimizer, _ = C.make_optimizer(model)
        logits = model(first_batch)
        if use_total:
            loss, parts = L.total_loss_parts(logits, first_y, 0.0)
            value = float(loss)
        else:
            loss = L.masked_bce_loss(logits, first_y)
            value = float(loss)
            parts = {"loss_bce": loss.detach()}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = {
            name: p.grad.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        return {
            "loss": value,
            "loss_bce": float(parts["loss_bce"]),
            "grads": grads,
            "logits": logits.detach().cpu().clone(),
            "logits_sha256": C.sha256_bytes(
                np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32).tobytes()
            ),
        }

    bce_branch = _one_branch(False)
    total_branch = _one_branch(True)
    grad_diff = _state_diff(bce_branch["grads"], total_branch["grads"])
    logits_diff = float(
        (bce_branch["logits"].float() - total_branch["logits"].float()).abs().max()
    )
    lambda_zero = {
        "bce_loss": bce_branch["loss"],
        "total_loss_lambda0": total_branch["loss"],
        "loss_abs_delta": abs(bce_branch["loss"] - total_branch["loss"]),
        "loss_exact": bce_branch["loss"] == total_branch["loss"],
        "grad_max_abs_diff": grad_diff["max_abs_diff"],
        "grad_tensors_compared": grad_diff["keys"],
        "grad_exact": bool(grad_diff["exact"]),
        "forward_logits_exact": bce_branch["logits_sha256"] == total_branch["logits_sha256"],
        "forward_logits_max_abs_diff": logits_diff,
        "loss_tolerance": 1e-7,
        "grad_tolerance": 1e-6,
    }
    lambda_zero["pass"] = bool(
        lambda_zero["loss_abs_delta"] <= float(lambda_zero["loss_tolerance"])
        and grad_diff["max_abs_diff"] <= float(lambda_zero["grad_tolerance"])
        and logits_diff <= float(lambda_zero["grad_tolerance"])
    )

    # -- 7. checkpoint round-trip and resume --------------------------------
    probe_rows = [np.asarray(plan[0]["batches"][i], dtype=np.int64) for i in range(4)]
    probe_batches = [
        stager.batch(C.BC.PC.collate(task.cache, rows.tolist())) for rows in probe_rows
    ]
    probe_labels = [stager.labels(task.labels[rows]) for rows in probe_rows]
    model_a, _ = C.build_w2(task, device=device)
    opt_a, _ = C.make_optimizer(model_a)
    C.seed_everything(int(C.SEED))
    probe_losses_a: list[float] = []
    for i in range(4):
        out = T.train_step(
            model_a,
            opt_a,
            probe_batches[i],
            probe_labels[i],
            global_step=i,
            clip=float(C.RECIPE["gradient_clip"]),
            lambda_rank=1e-4,
        )
        probe_losses_a.append(out["loss_total"])
    state_a = C.state_cpu(model_a)
    opt_a_state = C.optimizer_state_cpu(opt_a)
    rng_a = C.capture_rng_state()

    model_b, _ = C.build_w2(task, device=device)
    opt_b, _ = C.make_optimizer(model_b)
    C.seed_everything(int(C.SEED))
    for i in range(2):
        T.train_step(
            model_b,
            opt_b,
            probe_batches[i],
            probe_labels[i],
            global_step=i,
            clip=float(C.RECIPE["gradient_clip"]),
            lambda_rank=1e-4,
        )
    probe_payload = T.checkpoint_payload(
        epoch=1,
        global_step=2,
        model=model_b,
        optimizer=opt_b,
        rng_state=C.capture_rng_state(),
        plan=C.plan_meta(plan, 1, int(C.SEED)),
        model_audit=audit,
        valid_ap=None,
        valid_report=None,
        kind="audit_resume_probe",
        calibration=calibration_payload,
        calibration_path=calibration_path,
        extra={"test_evaluated": False},
    )
    probe_path = C.RUNS / "audit_resume_probe.pt"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    T.atomic_save(probe_payload, probe_path)
    model_c, _ = C.build_w2(task, device=device)
    opt_c, _ = C.make_optimizer(model_c)
    restore_audit = T.restore(probe_payload, model_c, opt_c)
    probe_losses_c: list[float] = []
    for i in range(2, 4):
        out = T.train_step(
            model_c,
            opt_c,
            probe_batches[i],
            probe_labels[i],
            global_step=i,
            clip=float(C.RECIPE["gradient_clip"]),
            lambda_rank=1e-4,
        )
        probe_losses_c.append(out["loss_total"])
    state_c = C.state_cpu(model_c)
    state_diff = _state_diff(state_a, state_c)

    # Compare AdamW moment tensors too (the dict-of-dicts needs flattening).
    def _flat_opt(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        flat: dict[str, torch.Tensor] = {}
        for param_id, values in state["state"].items():
            for key, value in values.items():
                if torch.is_tensor(value):
                    flat[f"{param_id}.{key}"] = value.detach().cpu()
        return flat

    opt_diff = _state_diff(_flat_opt(opt_a_state), _flat_opt(C.optimizer_state_cpu(opt_c)))
    rng_resume_ok = (
        C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        == C.rng_state_hashes(rng_a)["combined_sha256"]
    )
    resume = {
        "probe_batches": 4,
        "losses_uninterrupted": probe_losses_a,
        "losses_after_resume": [probe_losses_a[0], probe_losses_a[1], *probe_losses_c],
        "step4_loss_uninterrupted": probe_losses_a[-1],
        "step4_loss_after_resume": probe_losses_c[-1],
        "step4_loss_abs_delta": abs(probe_losses_a[-1] - probe_losses_c[-1]),
        "model_state": state_diff,
        "optimizer_state": opt_diff,
        "restore": restore_audit,
        "rng_after_resume_matches": bool(rng_resume_ok),
        "tolerance": TOL_RESUME,
    }
    resume["model_state_max_abs_diff"] = state_diff["max_abs_diff"]
    resume["optimizer_state_max_abs_diff"] = opt_diff["max_abs_diff"]
    resume["state_tolerance"] = TOL_RESUME
    resume["pass"] = bool(
        resume["step4_loss_abs_delta"] <= TOL_RESUME
        and state_diff["max_abs_diff"] <= TOL_RESUME
        and opt_diff["max_abs_diff"] <= TOL_RESUME
        and rng_resume_ok
    )
    del model_a, model_b, model_c, opt_a, opt_b, opt_c
    try:
        probe_path.unlink()
    except FileNotFoundError:
        pass

    # -- 8. timing / memory projection --------------------------------------
    timing_model, _ = C.build_w2(task, device=device)
    timing_opt, _ = C.make_optimizer(timing_model)
    C.seed_everything(int(C.SEED))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    warmup_steps = 5
    timed_steps = 30
    step_times: list[float] = []
    for i in range(warmup_steps + timed_steps):
        rows = np.asarray(plan[0]["batches"][i], dtype=np.int64)
        batch = stager.batch(C.BC.PC.collate(task.cache, rows.tolist()))
        y = stager.labels(task.labels[rows])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        T.train_step(
            timing_model,
            timing_opt,
            batch,
            y,
            global_step=i,
            clip=float(C.RECIPE["gradient_clip"]),
            lambda_rank=lambda_rank,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if i >= warmup_steps:
            step_times.append(float(time.perf_counter() - t))
    arr = np.asarray(step_times, dtype=np.float64)
    step_mean = float(arr.mean())
    step_p50 = float(np.percentile(arr, 50))
    peak = C.gpu_peak(device)
    timing_model.eval()
    t_valid = time.perf_counter()
    with torch.inference_mode():
        C.BC.SD.evaluate_validation(timing_model, task, device, reason="preflight_timing")
    valid_seconds = float(time.perf_counter() - t_valid)
    t_train_eval = time.perf_counter()
    with torch.inference_mode():
        C.BC.SD.evaluate_train(timing_model, task, device)
    train_eval_seconds = float(time.perf_counter() - t_train_eval)
    del timing_model
    train_seconds_projected = step_mean * float(C.STEPS_PER_EPOCH) * float(C.TOTAL_EPOCHS)
    valid_seconds_projected = valid_seconds * float(C.TOTAL_EPOCHS)
    checkpoint_overhead = 2.0 * float(C.TOTAL_EPOCHS)
    projected_seconds = (
        train_seconds_projected
        + valid_seconds_projected
        + train_eval_seconds
        + epoch0_seconds
        + checkpoint_overhead
    )
    projected_hours = projected_seconds / 3600.0
    projection = {
        "step_ms_mean": step_mean * 1e3,
        "step_ms_p50": step_p50 * 1e3,
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "timed_steps": timed_steps,
        "warmup_steps": warmup_steps,
        "epoch_train_seconds_projected": step_mean * float(C.STEPS_PER_EPOCH),
        "valid_seconds_measured": valid_seconds,
        "train_eval_seconds_measured": train_eval_seconds,
        "epoch0_seconds_measured": epoch0_seconds,
        "train_seconds_projected": train_seconds_projected,
        "valid_seconds_projected": valid_seconds_projected,
        "checkpoint_overhead_seconds_projected": checkpoint_overhead,
        "projected_seconds": projected_seconds,
        "projected_gpu_hours": projected_hours,
        "target_gpu_hours": float(C.BUDGET["gpu_target_hours"]),
        "hard_gpu_hours": float(C.BUDGET["gpu_hard_hours"]),
        "within_target": bool(projected_hours <= float(C.BUDGET["gpu_target_hours"])),
        "within_hard_cap": bool(projected_hours <= float(C.BUDGET["gpu_hard_hours"])),
        "gpu_peak_allocated_bytes": int(peak["peak_allocated_bytes"]),
        "gpu_peak_reserved_bytes": int(peak["peak_reserved_bytes"]),
    }

    # -- 9. aggregate -------------------------------------------------------
    checks = {
        **{f"init_{k}": bool(v) for k, v in init_checks.items()},
        **{f"forward_{k}": bool(v) for k, v in forward_checks.items()},
        "plan_sha256_matches_frozen": plan[0]["global_plan_sha256"] == C.GLOBAL_PLAN_SHA256,
        "calibration_valid": calibration_ok,
        "calibration_train_only": True,
        "calibration_test_labels_read_false": calibration_payload["test_labels_read"] is False,
        "calibration_rng_unchanged": bool(rng_unchanged),
        "calibration_lambda_within_bounds": float(C.CALIBRATION["lambda_min"])
        <= lambda_rank
        <= float(C.CALIBRATION["lambda_max"]),
        **{f"epoch0_{k}": bool(v) for k, v in epoch0["checks"].items()},
        "loss_self_check_all_pass": bool(loss_check["all_pass"]),
        "lambda_zero_is_exact_bce": bool(lambda_zero["pass"]),
        "checkpoint_resume_exact": bool(resume["pass"]),
        "projection_within_hard_cap": bool(projection["within_hard_cap"]),
        "no_test_labels_read": True,
    }
    all_pass = bool(all(checks.values()))
    calibration_payload["status"] = "FROZEN" if all_pass else "INVALID"
    calibration_payload["preflight_all_pass"] = all_pass
    C.write_json(calibration_path, calibration_payload)
    calibration_sha = C.sha256_file(calibration_path)

    result = {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "host_facts": C.host_facts(),
        "gpu_occupancy_start": occupancy_start,
        "gpu_occupancy_end": C.gpu_occupancy(),
        "plan": {
            "global_plan_sha256": plan[0]["global_plan_sha256"],
            "steps_per_epoch": int(C.STEPS_PER_EPOCH),
            "n_train": int(C.N_TRAIN),
            "n_valid": int(C.N_VALID),
        },
        "init_audit": init_audit,
        "forward_audit": forward_audit,
        "calibration": {
            "path": C.track_rel(calibration_path),
            "sha256": calibration_sha,
            "lambda_rank": lambda_rank,
            "g_bce_median": g_bce_median,
            "g_rank_median": g_rank_median,
            "ratio_raw": ratio_raw,
            "clipped": clipped,
            "clip_side": clip_side,
            "valid_batches": len(valid),
            "combined_rows_sha256": combined_rows_sha,
            "rng_unchanged": bool(rng_unchanged),
        },
        "epoch0": epoch0,
        "loss_self_check": loss_check,
        "lambda_zero_parity": lambda_zero,
        "resume": resume,
        "timing": projection,
        "checks": checks,
        "all_pass": all_pass,
        "elapsed_seconds": float(time.perf_counter() - started),
        "test_evaluated": False,
        "test_labels_read": False,
    }
    C.write_json(out_dir / "preflight.json", result)
    C.log(
        f"preflight all_pass={all_pass} lambda={lambda_rank:.6g} "
        f"(g_bce={g_bce_median:.6g}, g_rank={g_rank_median:.6g}, valid={len(valid)}) "
        f"projected={projected_hours:.3f} GPU-h"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    try:
        result = run_preflight(device, Path(args.out_dir))
        return 0 if result["all_pass"] else 5
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            Path(args.out_dir) / "preflight_failed.json",
            {
                "schema": C.SCHEMA_PREFLIGHT,
                "round_id": C.ROUND_ID,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "created_utc": C.utcnow(),
                "git_commit": C.git_commit(),
                "test_evaluated": False,
            },
        )
        C.log(f"preflight FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
