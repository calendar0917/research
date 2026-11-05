"""Pre-training correctness gate for one LR-probe arm.

Run this on each assigned GPU *before* any formal training.  It writes a JSON
record; ``lr_audit.py`` then compares the two records and refuses the formal
phase unless every paired gate passes.

Gates
-----
1. frozen inheritance (model widths / AdamW recipe / 128 tasks),
2. historical checkpoint metadata + SHA-256,
3. loaded model state bitwise equals the historical state; same fingerprint,
4. epoch-0 official Validation AP reproduces 0.23449277039313338 within 1e-4,
5. deterministic 12-epoch row order, identical to ``pc_train._batches``,
6. no Validation/Test row appears in any training epoch,
7. PinnedStager batch/label equality and a real one-step gradient/update smoke,
8. official Test rows are all NaN and are never indexed,
9. data contract: official row ids, 128 tasks, NaN mask, Train-only norm.
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


_configure_imports()

import torch  # noqa: E402

import lr_common as C  # noqa: E402
import lr_data  # noqa: E402
import lr_model  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


def _labels_equal(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """NaN-aware label equality: NaN is a mask, not a value to compare."""
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return {
            "equal": False,
            "reason": "shape/dtype mismatch",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "dtype_a": str(a.dtype),
            "dtype_b": str(b.dtype),
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
    all_keys_equal = list(a.keys()) == list(b.keys())
    all_exact = all_keys_equal
    max_diff = 0.0
    for key in a:
        av, bv = a[key], b.get(key)
        row: dict[str, Any] = {"key": key}
        if torch.is_tensor(av) or torch.is_tensor(bv):
            if not (torch.is_tensor(av) and torch.is_tensor(bv)):
                row.update({"equal": False, "reason": "one side is not a tensor"})
                all_exact = False
            else:
                row.update(
                    {
                        "shape_equal": tuple(av.shape) == tuple(bv.shape),
                        "dtype_equal": av.dtype == bv.dtype,
                        "device_a": str(av.device),
                        "device_b": str(bv.device),
                    }
                )
                exact = bool(
                    torch.is_tensor(av)
                    and torch.is_tensor(bv)
                    and tuple(av.shape) == tuple(bv.shape)
                    and av.dtype == bv.dtype
                    and torch.equal(av, bv)
                )
                diff = 0.0
                if torch.is_tensor(av) and torch.is_tensor(bv) and tuple(av.shape) == tuple(bv.shape):
                    diff = float((av.float() - bv.float()).abs().max().item())
                row.update({"equal": exact, "max_abs_diff": diff})
                max_diff = max(max_diff, diff)
                all_exact = all_exact and exact
        else:
            exact = av == bv
            row.update({"equal": bool(exact), "value_a": av, "value_b": bv})
            all_exact = all_exact and bool(exact)
        fields.append(row)
    return {
        "key_order_equal": all_keys_equal,
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
    keys_equal = [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    if not keys_equal:
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


def _pinned_smoke(
    task: Any,
    model: torch.nn.Module,
    *,
    device: torch.device,
    plan: Sequence[Mapping[str, Any]],
    lr: float,
    steps: int,
) -> dict[str, Any]:
    """Real training integration: pageable vs PinnedStager, same first batches."""
    model_pageable = model
    model_pinned = copy.deepcopy(model)
    opt_pageable = lr_model.make_optimizer(model_pageable, learning_rate=lr)
    opt_pinned = lr_model.make_optimizer(model_pinned, learning_rate=lr)
    stager = PinnedStager(device, depth=2)
    clip = float(C.PROBE_RECIPE["gradient_clip"])
    rows_by_step = [np.asarray(b, dtype=np.int64) for b in plan[0]["batches"][: int(steps)]]
    records: list[dict[str, Any]] = []
    all_batch_exact = True
    all_label_exact = True
    all_loss = True
    all_grad = True
    all_update = True
    all_gnorm = True
    max_logits = 0.0
    max_loss = 0.0
    max_grad = 0.0
    max_update = 0.0
    max_gnorm = 0.0

    for idx, rows in enumerate(rows_by_step, start=1):
        # Same dropout mask for both paths: reset the global generators.
        torch.manual_seed(C.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(C.SEED)

        batch_pageable = BM.to_device(BM.collate(task.cache, rows.tolist()), device)
        y_pageable = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
        batch_pinned = stager.batch(BM.collate(task.cache, rows.tolist()))
        y_pinned = stager.labels(task.labels[rows])

        batch_cmp = _compare_batches(batch_pageable, batch_pinned)
        label_cmp = _labels_equal(y_pageable, y_pinned)
        label_exact = bool(label_cmp["equal"])
        label_diff = float(label_cmp.get("max_abs_diff_finite", 0.0))

        model_pageable.train()
        model_pinned.train()
        # Reset immediately before *each* forward so the two models see the
        # same dropout mask.  A single reset before both forwards would let the
        # first forward advance the CUDA generator.
        torch.manual_seed(C.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(C.SEED)
        logits_a = model_pageable(batch_pageable)
        torch.manual_seed(C.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(C.SEED)
        logits_b = model_pinned(batch_pinned)
        logits_diff = float((logits_a - logits_b).abs().max().item())
        loss_a = MET.masked_bce_loss(logits_a, y_pageable)
        loss_b = MET.masked_bce_loss(logits_b, y_pinned)
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

        record = {
            "step": idx,
            "rows_sha256": C.sha256_int64(rows),
            "batch": batch_cmp,
            "labels_exact_equal": label_exact,
            "labels_nan_mask_equal": bool(label_cmp.get("nan_mask_equal", False)),
            "labels_finite_values_equal": bool(label_cmp.get("finite_values_equal", False)),
            "labels_n_nan": int(label_cmp.get("n_nan_a", 0)),
            "labels_max_abs_diff_finite": float(label_diff),
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
        records.append(record)
        all_batch_exact = all_batch_exact and bool(batch_cmp["all_fields_exact_equal"])
        all_label_exact = all_label_exact and label_exact
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
            all_batch_exact
            and all_label_exact
            and all_loss
            and all_grad
            and all_update
            and all_gnorm
            and max_logits <= 1e-5
        ),
        "steps": len(records),
        "transfer": "pf_pipeline.PinnedStager(depth=2) vs pc_model.to_device (pageable)",
        "batch_fields_exact_equal": bool(all_batch_exact),
        "labels_exact_equal": bool(all_label_exact),
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
    *,
    device: torch.device,
    plan: Sequence[Mapping[str, Any]],
    lr: float,
    steps: int,
) -> dict[str, Any]:
    """20 real training steps with the same PinnedStager path, for throughput only."""
    stager = PinnedStager(device, depth=2)
    optimizer = lr_model.make_optimizer(model, learning_rate=lr)
    clip = float(C.PROBE_RECIPE["gradient_clip"])
    model.train()
    torch.manual_seed(C.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.SEED)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    batches = [np.asarray(b, dtype=np.int64) for b in plan[0]["batches"][: int(steps)]]
    times = []
    losses = []
    for rows in batches:
        t0 = time.perf_counter()
        batch = stager.batch(BM.collate(task.cache, rows.tolist()))
        y = stager.labels(task.labels[rows])
        logits = model(batch)
        loss = MET.masked_bce_loss(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
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
        "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
        "step_ms_p50": float(p50 * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
        "steps_per_second": float(len(times) / arr.sum()) if arr.sum() > 0 else float("nan"),
        "projected_epoch_seconds": float(p50 * C.STEPS_PER_EPOCH) if arr.size else float("nan"),
        "projected_12_epoch_seconds": float(p50 * C.STEPS_PER_EPOCH * C.TOTAL_EPOCHS)
        if arr.size
        else float("nan"),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "losses_all_finite": bool(np.isfinite(np.asarray(losses)).all()) if losses else False,
        "note": "throughput smoke only; not used for model selection",
    }


def preflight(
    *,
    group: str,
    out_path: Path,
    device: torch.device,
    assigned_pool: str,
    equivalence_steps: int,
    timing_steps: int,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    result: dict[str, Any] = {
        "schema": "lcrc-pcba-lr-probe-preflight-v1",
        "round_id": C.ROUND_ID,
        "group": group,
        "group_spec": C.GROUPS[group],
        "assigned_pool": assigned_pool,
        "recipe": C.recipe_for_group(group),
        "device": str(device),
        "environment": C.environment(),
        "started_utc": C.utcnow(),
        "test_evaluated": False,
    }
    gates: dict[str, Any] = {}
    try:
        inheritance = C.assert_inherited_constants()
        result["inheritance"] = inheritance
        gates["inheritance"] = True

        task = lr_data.load_task()
        result["data_audit"] = task.data_audit
        result["data_load_seconds"] = float(task.load_seconds)

        checkpoint = lr_model.load_checkpoint()
        result["historical_checkpoint"] = {
            "path": C.track_rel(C.HIST["checkpoint"]),
            "sha256": C.sha256_file(C.HIST["checkpoint"]),
            "metadata": {
                "round_id": checkpoint.get("round_id"),
                "result_label": checkpoint.get("result_label"),
                "best_epoch": checkpoint.get("best_epoch"),
                "best_metric": checkpoint.get("best_metric"),
                "metric": checkpoint.get("metric"),
                "feature_spec_hash": checkpoint.get("feature_spec_hash"),
                "norm_fingerprint": checkpoint.get("norm_fingerprint"),
            },
        }
        gates["checkpoint_metadata"] = True

        model, model_info, load_audit = lr_model.build_from_checkpoint(
            task, checkpoint, device=device, group=group
        )
        result["initialization"] = load_audit
        result["parameter_summary"] = lr_model.parameter_value_summary(model)
        gates["parameter_fingerprint_equal"] = bool(load_audit["parameter_fingerprints_equal"])
        gates["model_config_and_norm_match"] = bool(load_audit["model_info_equal_to_historical"])

        # ---- epoch 0 official Validation AP, before any update ----
        model.eval()
        epoch0 = lr_model.evaluate_validation(model, task, device, reason="preflight_epoch0")
        delta_hist = abs(float(epoch0["ap"]) - C.HIST["best_valid_ap"])
        epoch0["historical_best_ap"] = C.HIST["best_valid_ap"]
        epoch0["abs_delta_vs_historical"] = float(delta_hist)
        epoch0["tolerance"] = 1e-4
        epoch0["reproduces_historical"] = bool(delta_hist <= 1e-4)
        result["epoch0_validation"] = epoch0
        gates["epoch0_ap_reproduces"] = bool(epoch0["reproduces_historical"])

        # ---- deterministic paired order ----
        plan = lr_data.make_order_plan(task.train_rows)
        plan_repeat = lr_data.make_order_plan(task.train_rows)
        summary = lr_data.order_summary(plan, n_train_rows=int(task.train_rows.size))
        repeat_equal = all(
            a["row_order_sha256"] == b["row_order_sha256"]
            and a["batch_plan_sha256"] == b["batch_plan_sha256"]
            for a, b in zip(plan, plan_repeat)
        )
        isolation = lr_data.assert_order_isolation(
            plan, task.train_rows, task.valid_rows, task.test_rows
        )
        result["training_order"] = summary
        result["training_order_isolation"] = isolation
        result["training_order_repeat_equal"] = bool(repeat_equal)
        gates["training_order_deterministic"] = bool(repeat_equal)
        gates["no_valid_or_test_rows_in_training"] = bool(isolation["passed"])

        # ---- PinnedStager equivalence + real one-step smoke ----
        smoke = _pinned_smoke(
            task,
            model,
            device=device,
            plan=plan,
            lr=float(C.GROUPS[group]["learning_rate"]),
            steps=int(equivalence_steps),
        )
        result["pinned_stager_smoke"] = smoke
        gates["pinned_stager_batch_equivalence"] = bool(smoke["batch_fields_exact_equal"])
        gates["pinned_stager_label_equivalence"] = bool(smoke["labels_exact_equal"])
        gates["pinned_stager_gradient_update_smoke"] = bool(
            smoke["loss_within_1e-6"]
            and smoke["gradients_within_1e-6"]
            and smoke["update_within_1e-5"]
            and smoke["grad_norm_within_1e-6"]
        )

        # ---- timing smoke (PinnedStager, same real training path) ----
        timing = _timing_smoke(
            task,
            model,
            device=device,
            plan=plan,
            lr=float(C.GROUPS[group]["learning_rate"]),
            steps=int(timing_steps),
        )
        result["timing_smoke"] = timing

        # ---- fixed recipe / optimizer defaults ----
        opt = lr_model.make_optimizer(model, learning_rate=float(C.GROUPS[group]["learning_rate"]))
        opt_defaults = opt.defaults
        recipe_checks = {
            "optimizer_is_adamw": isinstance(opt, torch.optim.AdamW),
            "lr_is_group_value": abs(float(opt_defaults["lr"]) - float(C.GROUPS[group]["learning_rate"]))
            == 0.0,
            "betas_equal": tuple(float(x) for x in opt_defaults["betas"])
            == tuple(float(x) for x in C.PROBE_RECIPE["adam_betas"]),
            "eps_equal": float(opt_defaults["eps"]) == float(C.PROBE_RECIPE["adam_eps"]),
            "weight_decay_equal": float(opt_defaults["weight_decay"])
            == float(C.PROBE_RECIPE["weight_decay"]),
            "dropout_equal": float(getattr(model, "drop").p) == float(C.PROBE_RECIPE["dropout"]),
            "batch_size_equal": int(C.PROBE_RECIPE["batch_size"]) == C.BATCH_SIZE,
            "gradient_clip_equal": float(C.PROBE_RECIPE["gradient_clip"]) == 5.0,
            "no_scheduler": C.PROBE_RECIPE["scheduler"] == "none",
            "fp32": str(next(model.parameters()).dtype) == "torch.float32",
            "no_autocast": not bool(torch.is_autocast_enabled()),
            "epochs_12": int(C.PROBE_RECIPE["epochs"]) == 12,
            "eval_epochs": list(C.PROBE_RECIPE["eval_epochs"]) == [0, 3, 6, 9, 12],
            "adamw_fresh": True,
        }
        result["recipe_checks"] = recipe_checks
        gates["fixed_recipe"] = bool(all(recipe_checks.values()))

        # ---- Test isolation / data contract ----
        test_isolation = {
            "test_rows_all_nan": bool(not np.isfinite(task.labels[task.test_rows]).any()),
            "test_rows_never_in_plan": all(
                np.intersect1d(np.asarray(p["rows"]), task.test_rows).size == 0 for p in plan
            ),
            "split_sizes": {k: int(v.size) for k, v in task.split.items()},
            "n_tasks": int(task.labels.shape[1]),
            "label_artifact_sha256": task.label_artifact_sha256,
            "expected_label_artifact_sha256": C.HIST["label_artifact_sha256"],
            "test_evaluated": False,
        }
        result["test_isolation"] = test_isolation
        gates["test_isolation"] = bool(
            test_isolation["test_rows_all_nan"]
            and test_isolation["test_rows_never_in_plan"]
            and test_isolation["n_tasks"] == C.N_TASKS
            and test_isolation["label_artifact_sha256"] == C.HIST["label_artifact_sha256"]
        )
        data_contract = {
            "n_molecules": int(task.labels.shape[0]),
            "n_tasks": int(task.labels.shape[1]),
            "gids_are_official_row_order": bool(
                np.array_equal(np.asarray(task.cache["gids"]), np.arange(C.N_MOLECULES))
            ),
            "split_partition_sha256": task.split_partition_sha256,
            "expected_split_partition_sha256": C.HIST["split_partition_sha256"],
            "feature_spec_hash": task.spec.hash(),
            "expected_feature_spec_hash": C.HIST["feature_spec_hash"],
            "norm_fingerprint": task.norm["norm_fingerprint"],
            "expected_norm_fingerprint": C.HIST["norm_fingerprint"],
            "nan_mask_semantics": "NaN is excluded from loss and never filled with 0",
            "train_observed_entries": task.data_audit["train_observed_entries"],
            "valid_observed_entries": task.data_audit["valid_observed_entries"],
        }
        result["data_contract"] = data_contract
        gates["data_contract"] = bool(
            data_contract["n_molecules"] == C.N_MOLECULES
            and data_contract["n_tasks"] == C.N_TASKS
            and data_contract["gids_are_official_row_order"]
            and data_contract["split_partition_sha256"] == C.HIST["split_partition_sha256"]
            and data_contract["feature_spec_hash"] == C.HIST["feature_spec_hash"]
            and data_contract["norm_fingerprint"] == C.HIST["norm_fingerprint"]
        )

    except Exception as exc:  # noqa: BLE001 - the failure is the deliverable
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        result["gates"] = gates
        result["all_passed"] = False
        result["gpu_seconds"] = float(time.perf_counter() - t_start)
        result["gpu_hours"] = float(result["gpu_seconds"] / 3600.0)
        result["finished_utc"] = C.utcnow()
        C.write_json(out_path, result)
        C.log(f"PREFLIGHT FAILED group={group}: {exc}")
        return result

    result["gates"] = gates
    result["all_passed"] = bool(all(bool(v) for v in gates.values()))
    result["gpu_seconds"] = float(time.perf_counter() - t_start)
    result["gpu_hours"] = float(result["gpu_seconds"] / 3600.0)
    result["finished_utc"] = C.utcnow()
    C.write_json(out_path, result)
    C.log(
        f"preflight group={group} all_passed={result['all_passed']} "
        f"epoch0_ap={result.get('epoch0_validation', {}).get('ap')} "
        f"gates={gates} gpu_hours={result['gpu_hours']:.6f}"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(C.GROUPS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assigned-pool", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--equivalence-steps", type=int, default=int(C.BUDGET["preflight_equivalence_steps"]))
    parser.add_argument("--timing-steps", type=int, default=int(C.BUDGET["preflight_smoke_steps"]))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.log("CUDA requested but unavailable")
        return 2
    assigned = args.assigned_pool or str(C.GROUPS[args.group]["assigned_pool"])
    result = preflight(
        group=args.group,
        out_path=Path(args.out),
        device=device,
        assigned_pool=assigned,
        equivalence_steps=int(args.equivalence_steps),
        timing_steps=int(args.timing_steps),
    )
    return 0 if result.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
