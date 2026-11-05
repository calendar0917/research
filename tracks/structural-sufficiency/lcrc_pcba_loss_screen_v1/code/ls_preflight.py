"""GPU preflight for one loss-screen arm.

Run on both assigned GPU pools before formal training.  Each record verifies
the frozen model, the restored AdamW state, the restored RNG state, the exact
Validation AP, the shared batch order, PinnedStager equivalence, the exact
historical M loss, and the finite/reference-matching T loss with a real change
of task weighting.  ``ls_pair_audit.py`` combines the two records.
"""

from __future__ import annotations

import argparse
import copy
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
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import ls_checkpoint as CK  # noqa: E402
import ls_common as C  # noqa: E402
import ls_losses as L  # noqa: E402
import lr_data  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


def _mean_finite(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _labels_equal(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return {
            "equal": False,
            "reason": "shape/dtype mismatch",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    nan_a = torch.isnan(a)
    nan_b = torch.isnan(b)
    nan_mask_equal = bool(torch.equal(nan_a, nan_b))
    finite = torch.isfinite(a) & torch.isfinite(b)
    finite_values_equal = bool(torch.equal(a[finite], b[finite])) if bool(finite.any()) else True
    max_diff = float((a[finite] - b[finite]).abs().max().item()) if bool(finite.any()) else 0.0
    return {
        "equal": bool(nan_mask_equal and finite_values_equal),
        "nan_mask_equal": nan_mask_equal,
        "finite_values_equal": finite_values_equal,
        "n_nan_a": int(nan_a.sum()),
        "n_nan_b": int(nan_b.sum()),
        "max_abs_diff_finite": max_diff,
    }


def _compare_batches(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    fields = []
    all_exact = list(a.keys()) == list(b.keys())
    max_diff = 0.0
    for key in a:
        av, bv = a[key], b.get(key)
        row: dict[str, Any] = {"key": key}
        if torch.is_tensor(av) or torch.is_tensor(bv):
            if not (torch.is_tensor(av) and torch.is_tensor(bv)):
                row.update({"equal": False, "reason": "one side is not a tensor"})
                all_exact = False
            else:
                exact = bool(
                    tuple(av.shape) == tuple(bv.shape)
                    and av.dtype == bv.dtype
                    and torch.equal(av, bv)
                )
                diff = (
                    float((av.float() - bv.float()).abs().max().item())
                    if tuple(av.shape) == tuple(bv.shape)
                    else float("inf")
                )
                row.update(
                    {
                        "shape_equal": tuple(av.shape) == tuple(bv.shape),
                        "dtype_equal": av.dtype == bv.dtype,
                        "equal": exact,
                        "max_abs_diff": diff,
                    }
                )
                max_diff = max(max_diff, diff)
                all_exact = all_exact and exact
        else:
            exact = av == bv
            row.update({"equal": bool(exact), "value_a": av, "value_b": bv})
            all_exact = all_exact and bool(exact)
        fields.append(row)
    return {
        "key_order_equal": list(a.keys()) == list(b.keys()),
        "all_fields_exact_equal": bool(all_exact),
        "max_abs_diff": float(max_diff),
        "fields": fields,
    }


def _grad_stats(model: torch.nn.Module) -> dict[str, Any]:
    max_abs = 0.0
    finite = True
    n = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        n += 1
        if not torch.isfinite(p.grad).all():
            finite = False
        max_abs = max(max_abs, float(p.grad.detach().abs().max().item()))
    return {"n_grad_tensors": n, "all_finite": bool(finite), "max_abs_grad": float(max_abs)}


def _grad_diff(a: torch.nn.Module, b: torch.nn.Module) -> dict[str, Any]:
    max_diff = 0.0
    worst = None
    if [n for n, _ in a.named_parameters()] != [n for n, _ in b.named_parameters()]:
        return {"equal": False, "reason": "parameter name/order mismatch"}
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is None or pb.grad is None:
            if pa.grad is not pb.grad:
                return {"equal": False, "reason": f"gradient missing at {name}"}
            continue
        diff = float((pa.grad - pb.grad).abs().max().item())
        if diff > max_diff:
            max_diff = diff
            worst = name
    return {"equal": bool(max_diff <= 1e-6), "max_abs_diff": float(max_diff), "worst": worst}


def _weight_diff(a: torch.nn.Module, b: torch.nn.Module) -> dict[str, Any]:
    max_diff = 0.0
    worst = None
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        diff = float((pa.detach() - pb.detach()).abs().max().item())
        if diff > max_diff:
            max_diff = diff
            worst = name
    return {"equal": bool(max_diff <= 1e-5), "max_abs_diff": float(max_diff), "worst": worst}


def _clone_with_optimizer(model: torch.nn.Module, source_opt: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    clone = copy.deepcopy(model)
    optimizer = CK.make_optimizer(clone)
    audit = CK.load_optimizer_state(optimizer, source_opt, clone)
    return clone, optimizer, audit


def _pinned_smoke(
    task: Any,
    model: torch.nn.Module,
    source_opt: Mapping[str, Any],
    *,
    device: torch.device,
    plan: Sequence[Mapping[str, Any]],
    rng_start: Mapping[str, Any],
    steps: int = 2,
) -> dict[str, Any]:
    model_pageable = copy.deepcopy(model)
    model_pinned = copy.deepcopy(model)
    opt_pageable = CK.make_optimizer(model_pageable)
    opt_pinned = CK.make_optimizer(model_pinned)
    CK.load_optimizer_state(opt_pageable, source_opt, model_pageable)
    CK.load_optimizer_state(opt_pinned, source_opt, model_pinned)
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    batches = [
        np.asarray(b, dtype=np.int64) for b in plan[0]["batches"][: int(steps)]
    ]
    records: list[dict[str, Any]] = []
    all_batch = True
    all_label = True
    all_loss = True
    all_grad = True
    all_update = True
    all_gnorm = True
    max_logits = 0.0
    max_loss = 0.0
    max_grad = 0.0
    max_update = 0.0
    max_gnorm = 0.0

    for idx, rows in enumerate(batches, start=1):
        C.restore_rng_state(rng_start)
        batch_pageable = BM.to_device(BM.collate(task.cache, rows.tolist()), device)
        y_pageable = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
        C.restore_rng_state(rng_start)
        batch_pinned = stager.batch(BM.collate(task.cache, rows.tolist()))
        y_pinned = stager.labels(task.labels[rows])
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        batch_cmp = _compare_batches(batch_pageable, batch_pinned)
        label_cmp = _labels_equal(y_pageable, y_pinned)

        model_pageable.train()
        model_pinned.train()
        C.restore_rng_state(rng_start)
        logits_a = model_pageable(batch_pageable)
        C.restore_rng_state(rng_start)
        logits_b = model_pinned(batch_pinned)
        logits_diff = float((logits_a - logits_b).abs().max().item())
        loss_a = L.masked_bce_loss(logits_a, y_pageable)
        loss_b = L.masked_bce_loss(logits_b, y_pinned)
        loss_diff = abs(float(loss_a.item()) - float(loss_b.item()))

        opt_pageable.zero_grad(set_to_none=True)
        opt_pinned.zero_grad(set_to_none=True)
        loss_a.backward()
        loss_b.backward()
        gnorm_a = float(torch.nn.utils.clip_grad_norm_(model_pageable.parameters(), clip))
        gnorm_b = float(torch.nn.utils.clip_grad_norm_(model_pinned.parameters(), clip))
        gnorm_diff = abs(gnorm_a - gnorm_b)
        grad_cmp = _grad_diff(model_pageable, model_pinned)
        opt_pageable.step()
        opt_pinned.step()
        update_cmp = _weight_diff(model_pageable, model_pinned)

        records.append(
            {
                "step": idx,
                "rows_sha256": C.sha256_int64(rows),
                "batch": batch_cmp,
                "labels_equal": bool(label_cmp["equal"]),
                "labels_nan_mask_equal": bool(label_cmp["nan_mask_equal"]),
                "labels_n_nan": int(label_cmp["n_nan_a"]),
                "logits_max_abs_diff": float(logits_diff),
                "loss_a": float(loss_a.item()),
                "loss_b": float(loss_b.item()),
                "loss_max_abs_diff": float(loss_diff),
                "grad_norm_a": float(gnorm_a),
                "grad_norm_b": float(gnorm_b),
                "grad_norm_max_abs_diff": float(gnorm_diff),
                "grad": grad_cmp,
                "update": update_cmp,
            }
        )
        all_batch = all_batch and bool(batch_cmp["all_fields_exact_equal"])
        all_label = all_label and bool(label_cmp["equal"])
        all_loss = all_loss and loss_diff <= 1e-6
        all_grad = all_grad and bool(grad_cmp["equal"])
        all_update = all_update and bool(update_cmp["equal"])
        all_gnorm = all_gnorm and gnorm_diff <= 1e-6
        max_logits = max(max_logits, logits_diff)
        max_loss = max(max_loss, loss_diff)
        max_grad = max(max_grad, float(grad_cmp["max_abs_diff"]))
        max_update = max(max_update, float(update_cmp["max_abs_diff"]))
        max_gnorm = max(max_gnorm, gnorm_diff)

    stager.release()
    return {
        "passed": bool(
            all_batch
            and all_label
            and all_loss
            and all_grad
            and all_update
            and all_gnorm
            and max_logits <= 1e-5
        ),
        "steps": len(records),
        "transfer": "pf_pipeline.PinnedStager(depth=2) vs pc_model.to_device (pageable)",
        "batch_fields_exact_equal": bool(all_batch),
        "labels_exact_equal": bool(all_label),
        "loss_within_1e-6": bool(all_loss),
        "gradients_within_1e-6": bool(all_grad),
        "update_within_1e-5": bool(all_update),
        "grad_norm_within_1e-6": bool(all_gnorm),
        "logits_within_1e-5": bool(max_logits <= 1e-5),
        "max_logits_abs_diff": float(max_logits),
        "max_loss_abs_diff": float(max_loss),
        "max_grad_abs_diff": float(max_grad),
        "max_update_abs_diff": float(max_update),
        "max_grad_norm_abs_diff": float(max_gnorm),
        "transfer_counters": dict(stager.counters),
        "records": records,
    }


def _timing_smoke(
    task: Any,
    model: torch.nn.Module,
    source_opt: Mapping[str, Any],
    *,
    device: torch.device,
    plan: Sequence[Mapping[str, Any]],
    group: str,
    rng_start: Mapping[str, Any],
    steps: int = 5,
) -> dict[str, Any]:
    clone, optimizer, _ = _clone_with_optimizer(model, source_opt)
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    n_task = L.task_counts(task.labels[task.train_rows])
    n_task_t = torch.as_tensor(n_task, dtype=torch.float32, device=device)
    batches = [np.asarray(b, dtype=np.int64) for b in plan[0]["batches"][: int(steps)]]
    C.restore_rng_state(rng_start)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    losses = []
    for rows in batches:
        t0 = time.perf_counter()
        batch = stager.batch(BM.collate(task.cache, rows.tolist()))
        y = stager.labels(task.labels[rows])
        clone.train()
        logits = clone(batch)
        if C.GROUPS[group]["loss"] == "masked_bce":
            loss = L.masked_bce_loss(logits, y)
        else:
            loss, _ = L.task_macro_bce_loss(logits, y, n_task_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clone.parameters(), clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t0))
    stager.release()
    arr = np.asarray(times, dtype=np.float64)
    p50 = float(np.percentile(arr, 50)) if arr.size else float("nan")
    return {
        "steps": len(times),
        "group_loss": C.GROUPS[group]["loss"],
        "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
        "step_ms_p50": float(p50 * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
        "steps_per_second": float(len(times) / arr.sum()) if arr.sum() > 0 else float("nan"),
        "projected_epoch_seconds": float(p50 * C.STEPS_PER_EPOCH) if arr.size else float("nan"),
        "projected_6_epoch_seconds": float(p50 * C.STEPS_PER_EPOCH * C.TOTAL_EPOCHS)
        if arr.size
        else float("nan"),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "losses_all_finite": bool(np.isfinite(np.asarray(losses)).all()) if losses else False,
    }


def preflight(
    *,
    group: str,
    out_dir: Path,
    device: torch.device,
    assigned_pool: str,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "lcrc-pcba-loss-screen-preflight-v1",
        "round_id": C.ROUND_ID,
        "group": group,
        "group_spec": C.GROUPS[group],
        "assigned_pool": assigned_pool,
        "device": str(device),
        "environment": C.environment(),
        "gpu_occupancy_start": C.gpu_occupancy(),
        "started_utc": C.utcnow(),
        "test_evaluated": False,
    }
    gates: dict[str, bool] = {}
    try:
        C.assert_inherited_constants()
        task = lr_data.load_task()
        result["data_audit"] = task.data_audit
        result["data_load_seconds"] = float(task.load_seconds)

        checkpoint = CK.load_source_checkpoint()
        result["checkpoint_metadata"] = CK.verify_source_metadata(checkpoint)
        result["source_checkpoint_sha256"] = C.sha256_file(C.SOURCE["checkpoint"])
        gates["checkpoint_sha256"] = (
            result["source_checkpoint_sha256"] == C.SOURCE["checkpoint_sha256"]
        )
        gates["checkpoint_metadata"] = True

        model, model_info, load_audit = CK.build_model_from_checkpoint(
            task, checkpoint, device=device, group=group
        )
        result["initialization"] = load_audit
        result["model_info"] = model_info
        result["model_parameter_fingerprint"] = load_audit["loaded_parameter_fingerprint"]
        gates["model_state_bitwise_equal"] = bool(load_audit["state_dict_bitwise_equal"])
        gates["model_config"] = bool(all(load_audit["model_checks"].values()))

        optimizer = CK.make_optimizer(model)
        opt_audit = CK.load_optimizer_state(
            optimizer, checkpoint["optimizer_state_dict"], model
        )
        result["optimizer_state"] = opt_audit
        result["optimizer_state_fingerprint"] = opt_audit["loaded_optimizer_state_fingerprint"]
        result["source_optimizer_state_fingerprint"] = opt_audit[
            "source_optimizer_state_fingerprint"
        ]
        gates["optimizer_state_identical"] = bool(opt_audit["state_identical_to_source"])

        rng_restore = C.restore_rng_state(checkpoint["rng_state"])
        rng_start = C.capture_rng_state()
        rng_hashes = C.rng_state_hashes(rng_start)
        source_rng_hashes = checkpoint["rng_state_hashes"]
        rng_match = {
            "python_random": rng_hashes["python_random_sha256"]
            == source_rng_hashes["python_random_sha256"],
            "numpy": rng_hashes["numpy_sha256"] == source_rng_hashes["numpy_sha256"],
            "torch_cpu": rng_hashes["torch_cpu_sha256"] == source_rng_hashes["torch_cpu_sha256"],
            "torch_cuda": rng_hashes["torch_cuda_sha256"]
            == source_rng_hashes["torch_cuda_sha256"],
        }
        result["rng"] = {
            "restore": rng_restore,
            "start_hashes": rng_hashes,
            "source_hashes": source_rng_hashes,
            "matches_source": rng_match,
        }
        gates["rng_restored_from_checkpoint"] = bool(
            rng_match["python_random"] and rng_match["numpy"] and rng_match["torch_cpu"]
        )

        # ---- epoch 0: frozen Validation AP ----
        model.eval()
        report0 = CK.evaluate_validation(model, task, device, reason="preflight_epoch0")
        independent0 = _mean_finite(report0["per_task_ap"])
        delta_hist = abs(float(report0["ap"]) - float(C.SOURCE["best_valid_ap"]))
        report0["historical_best_ap"] = float(C.SOURCE["best_valid_ap"])
        report0["abs_delta_vs_historical"] = float(delta_hist)
        report0["tolerance"] = 1e-4
        report0["reproduces_historical"] = bool(delta_hist <= 1e-4)
        result["epoch0_validation"] = report0
        result["epoch0_ap"] = float(report0["ap"])
        result["epoch0_independent_mean_ap"] = independent0
        result["epoch0_official_independent_delta"] = (
            None if independent0 is None else abs(float(report0["ap"]) - float(independent0))
        )
        gates["epoch0_ap_reproduces"] = bool(report0["reproduces_historical"])
        gates["epoch0_official_independent_agree"] = bool(
            independent0 is not None
            and abs(float(report0["ap"]) - float(independent0)) <= 1e-8
        )

        # ---- shared six-epoch order ----
        plan = C.make_order_plan(task.train_rows, epochs=C.TOTAL_EPOCHS)
        plan_repeat = C.make_order_plan(task.train_rows, epochs=C.TOTAL_EPOCHS)
        summary = C.order_summary(plan, n_train_rows=int(task.train_rows.size))
        repeat_equal = all(
            a["row_order_sha256"] == b["row_order_sha256"]
            and a["batch_plan_sha256"] == b["batch_plan_sha256"]
            for a, b in zip(plan, plan_repeat)
        )
        isolation = C.assert_order_isolation(
            plan, task.train_rows, task.valid_rows, task.test_rows
        )
        result["training_order"] = summary
        result["training_order_isolation"] = isolation
        result["training_order_repeat_equal"] = bool(repeat_equal)
        gates["training_order_deterministic"] = bool(repeat_equal)
        gates["no_valid_or_test_rows_in_training"] = bool(isolation["passed"])

        # ---- first batch: same dropout RNG, weights and inputs ----
        rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)
        y_cpu = np.asarray(task.labels[rows], dtype=np.float32)
        n_task = L.task_counts(task.labels[task.train_rows])
        n_task_t = torch.as_tensor(n_task, dtype=torch.float32, device=device)
        batch = BM.to_device(BM.collate(task.cache, rows.tolist()), device)
        y = torch.as_tensor(y_cpu, dtype=torch.float32, device=device)
        model_m = copy.deepcopy(model)
        model_t = copy.deepcopy(model)
        opt_m = CK.make_optimizer(model_m)
        opt_t = CK.make_optimizer(model_t)
        opt_m_audit = CK.load_optimizer_state(opt_m, checkpoint["optimizer_state_dict"], model_m)
        opt_t_audit = CK.load_optimizer_state(opt_t, checkpoint["optimizer_state_dict"], model_t)
        paired_model_fp = {
            "M": BM.state_fingerprint(model_m.state_dict()),
            "T": BM.state_fingerprint(model_t.state_dict()),
        }
        paired_opt_fp = {
            "M": opt_m_audit["loaded_optimizer_state_fingerprint"],
            "T": opt_t_audit["loaded_optimizer_state_fingerprint"],
        }
        result["paired_initial_state"] = {
            "model_fingerprints": paired_model_fp,
            "optimizer_fingerprints": paired_opt_fp,
            "model_equal": bool(paired_model_fp["M"] == paired_model_fp["T"]),
            "optimizer_equal": bool(paired_opt_fp["M"] == paired_opt_fp["T"]),
        }
        gates["paired_models_identical"] = bool(paired_model_fp["M"] == paired_model_fp["T"])
        gates["paired_optimizers_identical"] = bool(paired_opt_fp["M"] == paired_opt_fp["T"])

        model_m.train()
        model_t.train()
        C.restore_rng_state(rng_start)
        logits_m = model_m(batch)
        C.restore_rng_state(rng_start)
        logits_t = model_t(batch)
        logits_diff = float((logits_m - logits_t).abs().max().item())
        loss_m = L.masked_bce_loss(logits_m, y)
        loss_t, raw_t = L.task_macro_bce_loss(logits_t, y, n_task_t)
        m_reference = L.numpy_micro_reference(logits_m.detach().cpu().numpy(), y_cpu)
        t_reference = L.numpy_task_macro_reference(
            logits_t.detach().cpu().numpy(), y_cpu, n_task
        )
        loss_m_diff = abs(float(loss_m) - m_reference)
        loss_t_diff = abs(float(loss_t) - t_reference)

        model_m.zero_grad(set_to_none=True)
        model_t.zero_grad(set_to_none=True)
        loss_m.backward()
        loss_t.backward()
        grad_m = _grad_stats(model_m)
        grad_t = _grad_stats(model_t)
        grad_cmp = _grad_diff(model_m, model_t)
        n_observed = int(np.isfinite(y_cpu).sum())
        t_coeff = (C.N_TRAIN / (len(rows) * C.N_TASKS)) / n_task.astype(np.float64)
        m_coeff = 1.0 / n_observed
        weighting_changed = float(grad_cmp["max_abs_diff"]) > 1e-8
        result["first_batch"] = {
            "rows": rows.tolist(),
            "rows_sha256": C.sha256_int64(rows),
            "labels_sha256": C.sha256_array(y_cpu),
            "n_observed": n_observed,
            "n_task": n_task.tolist(),
            "logits_max_abs_diff_M_vs_T": float(logits_diff),
            "loss_M": float(loss_m),
            "loss_M_reference_numpy": m_reference,
            "loss_M_abs_diff": float(loss_m_diff),
            "loss_T": float(loss_t),
            "loss_T_raw_masked_bce": float(raw_t),
            "loss_T_reference_numpy": t_reference,
            "loss_T_abs_diff": float(loss_t_diff),
            "grad_M": grad_m,
            "grad_T": grad_t,
            "grad_max_abs_diff_T_minus_M": float(grad_cmp["max_abs_diff"]),
            "grad_T_minus_M_within_1e-6": bool(grad_cmp["equal"]),
            "task_weight_changed": bool(weighting_changed),
            "T_per_task_coefficient": t_coeff.tolist(),
            "M_per_entry_coefficient": float(m_coeff),
        }
        gates["M_loss_exact_historical"] = bool(loss_m_diff <= 1e-6)
        gates["T_loss_matches_independent_numpy"] = bool(loss_t_diff <= 1e-6)
        gates["T_loss_and_gradients_finite"] = bool(
            torch.isfinite(loss_t) and grad_t["all_finite"]
        )
        gates["task_weighting_really_changes_gradients"] = bool(weighting_changed)
        gates["paired_dropout_logits_match"] = bool(logits_diff <= 1e-5)

        torch.save(
            {
                "rows_sha256": C.sha256_int64(rows),
                "labels_sha256": C.sha256_array(y_cpu),
                "logits_M": logits_m.detach().cpu(),
                "logits_T": logits_t.detach().cpu(),
            },
            out_dir / "first_batch_logits.pt",
        )

        # ---- PinnedStager equivalence and real update smoke ----
        pinned = _pinned_smoke(
            task,
            model,
            checkpoint["optimizer_state_dict"],
            device=device,
            plan=plan,
            rng_start=rng_start,
            steps=2,
        )
        result["pinned_stager_smoke"] = pinned
        gates["pinned_stager_equivalence"] = bool(pinned["passed"])

        # ---- timing only (not selection) ----
        timing = _timing_smoke(
            task,
            model,
            checkpoint["optimizer_state_dict"],
            device=device,
            plan=plan,
            group=group,
            rng_start=rng_start,
            steps=5,
        )
        result["timing_smoke"] = timing

        # ---- Test isolation / data contract ----
        test_finite = int(np.isfinite(task.labels[task.test_rows]).sum())
        test_isolation = {
            "test_rows_all_nan": bool(test_finite == 0),
            "test_rows_never_in_plan": all(
                np.intersect1d(np.asarray(p["rows"]), task.test_rows).size == 0 for p in plan
            ),
            "split_sizes": {k: int(v.size) for k, v in task.split.items()},
            "n_tasks": int(task.labels.shape[1]),
            "label_artifact_sha256": task.label_artifact_sha256,
            "expected_label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
            "test_evaluated": False,
        }
        result["test_isolation"] = test_isolation
        gates["test_isolation"] = bool(
            test_isolation["test_rows_all_nan"]
            and test_isolation["test_rows_never_in_plan"]
            and test_isolation["n_tasks"] == C.N_TASKS
            and test_isolation["label_artifact_sha256"] == C.SOURCE["label_artifact_sha256"]
        )
        data_contract = {
            "n_molecules": int(task.labels.shape[0]),
            "n_tasks": int(task.labels.shape[1]),
            "gids_are_official_row_order": bool(
                np.array_equal(np.asarray(task.cache["gids"]), np.arange(C.N_MOLECULES))
            ),
            "split_partition_sha256": task.split_partition_sha256,
            "expected_split_partition_sha256": C.SOURCE["split_partition_sha256"],
            "feature_spec_hash": task.spec.hash(),
            "expected_feature_spec_hash": C.SOURCE["feature_spec_hash"],
            "norm_fingerprint": task.norm["norm_fingerprint"],
            "expected_norm_fingerprint": C.SOURCE["norm_fingerprint"],
            "train_observed_entries": task.data_audit["train_observed_entries"],
            "valid_observed_entries": task.data_audit["valid_observed_entries"],
            "test_finite_entries": test_finite,
            "nan_mask_semantics": "NaN is excluded from loss and never filled with 0",
        }
        result["data_contract"] = data_contract
        gates["data_contract"] = bool(
            data_contract["n_molecules"] == C.N_MOLECULES
            and data_contract["n_tasks"] == C.N_TASKS
            and data_contract["gids_are_official_row_order"]
            and data_contract["split_partition_sha256"] == C.SOURCE["split_partition_sha256"]
            and data_contract["feature_spec_hash"] == C.SOURCE["feature_spec_hash"]
            and data_contract["norm_fingerprint"] == C.SOURCE["norm_fingerprint"]
        )

        # ---- fixed recipe / no scheduler ----
        opt_defaults = optimizer.defaults
        recipe_checks = {
            "optimizer_is_adamw": isinstance(optimizer, torch.optim.AdamW),
            "lr_5e-5": abs(float(opt_defaults["lr"]) - float(C.RECIPE["learning_rate"])) == 0.0,
            "betas_equal": tuple(float(x) for x in opt_defaults["betas"])
            == tuple(C.RECIPE["adam_betas"]),
            "eps_equal": float(opt_defaults["eps"]) == float(C.RECIPE["adam_eps"]),
            "weight_decay_equal": float(opt_defaults["weight_decay"]) == float(C.RECIPE["weight_decay"]),
            "dropout_equal": float(getattr(model, "drop").p) == float(C.RECIPE["dropout"]),
            "batch_size_equal": int(C.RECIPE["batch_size"]) == C.BATCH_SIZE,
            "gradient_clip_equal": float(C.RECIPE["gradient_clip"]) == 5.0,
            "no_scheduler": C.RECIPE["scheduler"] == "none",
            "fp32": str(next(model.parameters()).dtype) == "torch.float32",
            "no_autocast": not bool(torch.is_autocast_enabled()),
            "epochs_6": int(C.RECIPE["epochs"]) == 6,
            "eval_epochs": list(C.RECIPE["eval_epochs"]) == [0, 2, 4, 6],
            "adamw_state_restored": not bool(opt_audit["moments_reinitialised"]),
        }
        result["recipe_checks"] = {k: bool(v) for k, v in recipe_checks.items()}
        gates["fixed_recipe"] = bool(all(recipe_checks.values()))

        result["gates"] = {k: bool(v) for k, v in gates.items()}
        result["all_passed"] = bool(all(gates.values()))
        result["gpu_occupancy_end"] = C.gpu_occupancy()
    except Exception as exc:  # noqa: BLE001 - the failure is the deliverable
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        result["gates"] = {k: bool(v) for k, v in gates.items()}
        result["all_passed"] = False
    result["gpu_seconds"] = float(time.perf_counter() - t_start)
    result["gpu_hours"] = float(result["gpu_seconds"] / 3600.0)
    result["finished_utc"] = C.utcnow()
    C.write_json(out_dir / "preflight.json", result)
    C.log(
        f"preflight group={group} all_passed={result['all_passed']} "
        f"epoch0_ap={result.get('epoch0_ap')} gpu_hours={result['gpu_hours']:.6f}"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(C.GROUPS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assigned-pool", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.log("CUDA requested but unavailable")
        return 2
    assigned = args.assigned_pool or str(C.GROUPS[args.group]["assigned_pool"])
    result = preflight(
        group=args.group,
        out_dir=Path(args.out_dir),
        device=device,
        assigned_pool=assigned,
    )
    return 0 if result.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
