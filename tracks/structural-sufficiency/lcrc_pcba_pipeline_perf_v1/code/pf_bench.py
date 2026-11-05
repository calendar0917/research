"""Correctness gates and the 300-step A/B throughput comparison.

A = frozen pipeline (``pc_model.to_device``: pageable, per-tensor, blocking)
B = same pipeline with the *measured* bottleneck replaced (``PinnedStager``:
pinned staging + non-blocking copies)

Nothing else differs between A and B: same frozen ``collate``, same batch order,
same model initialisation, same loss, optimiser, scheduler, FP32 configuration,
same Train/Valid split.  The gates below are the round's correctness threshold:
if any of them fails the optimisation is not used and the round is labelled
``NO_CLEAR_SPEEDUP``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pf_common as PF

import torch  # noqa: E402

import pc_common as PC  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as PCM  # noqa: E402
import pf_collate as PC_C  # noqa: E402
import pf_pipeline as PIPE  # noqa: E402

RECIPE = dict(PC.RECIPE)

#: Tolerances reuse the frozen round's FP32 constants where they exist.
TOL = {
    "device_batch_exact": 0.0,          # bytes must be identical
    "labels_exact": 0.0,
    "logits_max_abs": 1e-6,             # == pc_common.GRAD_TOL
    "loss_max_abs": 1e-6,
    "grad_max_abs": float(PC.GRAD_TOL),  # 1e-6
    "weight_max_abs_one_step": float(PC.GRAD_TOL),
    "weight_max_abs_ten_steps": float(PC.RELOAD_TOL),  # 1e-5
    "single_vs_batch_max_abs": float(PC.INFERENCE_TOL),  # 1e-5
}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(a.size),
        "sum_seconds": float(a.sum()),
        "mean_ms": float(a.mean() * 1e3),
        "p50_ms": float(np.percentile(a, 50) * 1e3),
        "p90_ms": float(np.percentile(a, 90) * 1e3),
        "min_ms": float(a.min() * 1e3),
        "max_ms": float(a.max() * 1e3),
    }


def _build_state(inputs: PF.Inputs, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(PF.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PF.SEED)
    model, info = PCM.build_c1(inputs.spec, inputs.norm, PC.N_TASKS, PF.SEED, device)
    steps_per_epoch = int(math.ceil(len(inputs.train_rows) / PF.BATCH_SIZE))
    total_updates = steps_per_epoch * int(RECIPE["scheduler_horizon_epochs"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(RECIPE["peak_lr"]),
        betas=tuple(float(x) for x in RECIPE["adam_betas"]),
        eps=float(RECIPE["adam_eps"]),
        weight_decay=float(RECIPE["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_updates, 1), eta_min=float(RECIPE["min_lr"])
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "info": info,
        "steps_per_epoch": steps_per_epoch,
    }


def _reset_rng() -> None:
    torch.manual_seed(PF.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PF.SEED)


# ---------------------------------------------------------------------------
# one step, identical arithmetic to pc_train, transfer injected
# ---------------------------------------------------------------------------


def one_step(
    state: Mapping[str, Any],
    inputs: PF.Inputs,
    rows: np.ndarray,
    device: torch.device,
    transfer,
    collate=PC_C.collate_reference,
    *,
    want_diagnostics: bool = False,
) -> dict[str, Any]:
    model, optimizer, scheduler = state["model"], state["optimizer"], state["scheduler"]
    batch_cpu = collate(inputs.cache, rows.tolist())
    batch = transfer.batch(batch_cpu)
    y_rows = inputs.labels[rows]
    y = transfer.labels(y_rows)
    logits = model(batch)
    if logits.shape != (len(rows), PC.N_TASKS):
        raise RuntimeError(f"model output {tuple(logits.shape)} != {(len(rows), PC.N_TASKS)}")
    loss = MET.masked_bce_loss(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(RECIPE["gradient_clip"])))
    if not math.isfinite(gnorm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    scheduler.step()
    loss_value = float(loss.detach().cpu().item())
    out = {"loss": loss_value, "grad_norm": gnorm, "lr": float(optimizer.param_groups[0]["lr"])}
    if want_diagnostics:
        out["logits"] = logits.detach()
        out["loss_tensor"] = loss.detach()
        out["grads"] = {n: (p.grad.detach().clone() if p.grad is not None else None)
                        for n, p in model.named_parameters()}
        out["mask_sha"] = _mask_sha(y_rows)
        out["n_observed"] = int(np.isfinite(y_rows).sum())
    return out


def _mask_sha(y_rows: np.ndarray) -> str:
    mask = np.isfinite(y_rows)
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise tensor equality (NaN-safe; ``torch.equal`` is not)."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    return bool(torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)))


def _field_sha(t: torch.Tensor) -> str:
    a = np.ascontiguousarray(t.detach().cpu().numpy())
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate1_collate_and_device_batch(inputs, batches, device, chosen) -> dict[str, Any]:
    out: dict[str, Any] = {"batches": {}}
    pageable = PIPE.PageableTransfer(device)
    stager = PIPE.PinnedStager(device)
    for name, rows in chosen.items():
        cpu_ref = PC_C.collate_reference(inputs.cache, rows.tolist())
        cpu_mirror, _ = PC_C.collate_instrumented(inputs.cache, rows.tolist())
        collate_equal = all(
            (cpu_ref[k] == cpu_mirror[k]) if not torch.is_tensor(cpu_ref[k])
            else _bit_equal(cpu_ref[k], cpu_mirror[k])
            for k in cpu_ref
        )
        a = pageable.batch(cpu_ref)
        b = stager.batch(cpu_ref)
        _sync()
        fields = {}
        worst = 0.0
        for key in cpu_ref:
            ta, tb = a[key], b[key]
            if not torch.is_tensor(ta):
                continue
            same = ta.device == tb.device and _bit_equal(ta, tb)
            diff = 0.0 if same else float((ta.float() - tb.float()).abs().max().item())
            worst = max(worst, diff)
            fields[key] = {
                "dtype": str(ta.dtype),
                "shape": list(ta.shape),
                "device": str(ta.device),
                "exact_equal": bool(same),
                "sha256": _field_sha(ta),
            }
        ya = pageable.labels(inputs.labels[rows])
        yb = stager.labels(inputs.labels[rows])
        _sync()
        labels_exact = _bit_equal(ya, yb)
        out["batches"][name] = {
            "n_molecules": int(len(rows)),
            "n_atoms": int(cpu_ref["atom_counts"].sum()),
            "n_bonds": int(cpu_ref["bond_counts"].sum()),
            "n_s2": int(cpu_ref["path_cat"].numel()),
            "collate_fields_equal": bool(collate_equal),
            "collate_field_order": list(cpu_ref.keys()),
            "device_fields": fields,
            "device_max_abs_diff": float(worst),
            "labels_exact_equal": bool(labels_exact),
            "labels_nan_mask_sha256": _mask_sha(inputs.labels[rows]),
        }
    out["passed"] = all(
        row["collate_fields_equal"]
        and row["device_max_abs_diff"] <= TOL["device_batch_exact"]
        and row["labels_exact_equal"]
        for row in out["batches"].values()
    )
    out["tolerance"] = {
        "collate_fields_equal": "bitwise (torch.equal) on every field",
        "device_max_abs_diff": TOL["device_batch_exact"],
        "labels_exact_equal": "bitwise",
    }
    return out


def gate2_3_inference(inputs, rows, device) -> dict[str, Any]:
    """Single-graph vs batch consistency, and A vs B forward logits."""
    state = _build_state(inputs, device)
    model = state["model"]
    model.eval()
    pageable = PIPE.PageableTransfer(device)
    stager = PIPE.PinnedStager(device)
    batch_cpu = PC_C.collate_reference(inputs.cache, rows.tolist())
    with torch.no_grad():
        logits_a = model(pageable.batch(batch_cpu))
        logits_b = model(stager.batch(batch_cpu))
        _sync()
        ab_diff = float((logits_a - logits_b).abs().max().item())
        ab_equal = _bit_equal(logits_a, logits_b)
        # per-molecule inference vs the batched result
        single: list[float] = []
        single_equal: list[bool] = []
        for i in range(min(4, len(rows))):
            sub = PC_C.collate_reference(inputs.cache, [int(rows[i])])
            one_b = model(stager.batch(sub))
            _sync()
            single.append(float((one_b[0] - logits_b[i]).abs().max().item()))
            single_equal.append(_bit_equal(one_b[0], logits_b[i]))
    return {
        "logits_shape": list(logits_a.shape),
        "all_finite": bool(torch.isfinite(logits_a).all()),
        "a_vs_b_max_abs_diff": ab_diff,
        "a_vs_b_bitwise_equal": ab_equal,
        "single_vs_batch_max_abs_diff": single,
        "single_vs_batch_bitwise_equal": single_equal,
        "single_vs_batch_n_checked": len(single),
        "tolerance": {
            "a_vs_b": TOL["logits_max_abs"],
            "single_vs_batch": TOL["single_vs_batch_max_abs"],
        },
        "passed": bool(
            torch.isfinite(logits_a).all()
            and ab_diff <= TOL["logits_max_abs"]
            and max(single) <= TOL["single_vs_batch_max_abs"]
        ),
    }


def gate4_one_step(inputs, batches, device) -> dict[str, Any]:
    results = {}
    for name, transfer_cls in (("A_pageable", PIPE.PageableTransfer), ("B_pinned", PIPE.PinnedStager)):
        st = _build_state(inputs, device)
        _reset_rng()
        kwargs = {}
        if transfer_cls is PIPE.PinnedStager:
            kwargs["depth"] = 2
        transfer = transfer_cls(device, **kwargs)
        step = one_step(st, inputs, batches[0], device, transfer, want_diagnostics=True)
        results[name] = {
            "loss": step["loss"],
            "grad_norm": step["grad_norm"],
            "lr": step["lr"],
            "logits_sha256": _field_sha(step["logits"]),
            "grads": step["grads"],
            "weights": {n: p.detach().clone() for n, p in st["model"].named_parameters()},
            "mask_sha256": step["mask_sha"],
            "n_observed": step["n_observed"],
            "transfer_counters": dict(transfer.counters),
        }
        transfer.release()
    a, b = results["A_pageable"], results["B_pinned"]
    grad_diffs = {
        n: (float((a["grads"][n] - b["grads"][n]).abs().max().item())
            if a["grads"][n] is not None and a["grads"][n].numel() else 0.0)
        for n in a["grads"]
    }
    weight_diffs = {
        n: (float((a["weights"][n] - b["weights"][n]).abs().max().item())
            if a["weights"][n].numel() else 0.0)
        for n in a["weights"]
    }
    grad_finite = all(
        bool(torch.isfinite(g).all()) for g in a["grads"].values() if g is not None
    )
    return {
        "loss_a": a["loss"],
        "loss_b": b["loss"],
        "loss_abs_diff": abs(a["loss"] - b["loss"]),
        "grad_norm_a": a["grad_norm"],
        "grad_norm_b": b["grad_norm"],
        "grad_norm_abs_diff": abs(a["grad_norm"] - b["grad_norm"]),
        "logits_bitwise_equal": a["logits_sha256"] == b["logits_sha256"],
        "max_grad_abs_diff": max(grad_diffs.values()) if grad_diffs else 0.0,
        "max_weight_abs_diff_after_one_update": max(weight_diffs.values()) if weight_diffs else 0.0,
        "n_grad_tensors": len(grad_diffs),
        "grads_all_finite": bool(grad_finite),
        "label_mask_sha256": a["mask_sha256"],
        "label_mask_identical": a["mask_sha256"] == b["mask_sha256"],
        "n_observed_labels": a["n_observed"],
        "lr_a": a["lr"],
        "lr_b": b["lr"],
        "transfer_counters": {k: v["transfer_counters"] for k, v in results.items()},
        "tolerance": {
            "loss_max_abs": TOL["loss_max_abs"],
            "grad_max_abs": TOL["grad_max_abs"],
            "weight_max_abs_one_step": TOL["weight_max_abs_one_step"],
        },
        "passed": bool(
            abs(a["loss"] - b["loss"]) <= TOL["loss_max_abs"]
            and max(grad_diffs.values(), default=0.0) <= TOL["grad_max_abs"]
            and max(weight_diffs.values(), default=0.0) <= TOL["weight_max_abs_one_step"]
            and grad_finite
            and a["mask_sha256"] == b["mask_sha256"]
        ),
    }


def gate5_ten_steps(inputs, batches, device) -> dict[str, Any]:
    n = 10
    runs = {}
    for name, cls in (("A_pageable", PIPE.PageableTransfer), ("B_pinned", PIPE.PinnedStager)):
        st = _build_state(inputs, device)
        _reset_rng()
        transfer = cls(device)
        losses = []
        for rows in batches[:n]:
            out = one_step(st, inputs, rows, device, transfer)
            losses.append(out["loss"])
        runs[name] = {
            "losses": losses,
            "weights": {k: v.detach().clone() for k, v in st["model"].state_dict().items()},
        }
        transfer.release()
    a, b = runs["A_pageable"], runs["B_pinned"]
    loss_diffs = [abs(x - y) for x, y in zip(a["losses"], b["losses"])]
    weight_diffs = {
        k: (float((a["weights"][k].float() - b["weights"][k].float()).abs().max().item())
            if a["weights"][k].numel() else 0.0)
        for k in a["weights"]
    }
    return {
        "steps": n,
        "losses_a": a["losses"],
        "losses_b": b["losses"],
        "max_loss_abs_diff": max(loss_diffs),
        "max_weight_abs_diff": max(weight_diffs.values()),
        "final_loss_a": a["losses"][-1],
        "final_loss_b": b["losses"][-1],
        "tolerance": {
            "loss_max_abs": TOL["loss_max_abs"],
            "weight_max_abs_ten_steps": TOL["weight_max_abs_ten_steps"],
        },
        "passed": bool(
            max(loss_diffs) <= TOL["loss_max_abs"]
            and max(weight_diffs.values()) <= TOL["weight_max_abs_ten_steps"]
        ),
    }


def gate6_labels_and_test_isolation(inputs, batches) -> dict[str, Any]:
    train_set = np.zeros(int(inputs.cache["n_molecules"]), dtype=bool)
    train_set[inputs.train_rows] = True
    valid_set = np.zeros_like(train_set)
    valid_set[inputs.valid_rows] = True
    test_set = np.zeros_like(train_set)
    test_set[inputs.test_rows] = True
    used = np.concatenate([np.asarray(b, dtype=np.int64) for b in batches])
    observed = np.isfinite(inputs.labels[used])
    return {
        "n_tasks": int(PC.N_TASKS),
        "label_shape": list(inputs.labels.shape),
        "label_dtype": str(inputs.labels.dtype),
        "test_rows_all_nan": bool(np.isnan(inputs.labels[inputs.test_rows]).all()),
        "test_rows_never_used_in_batches": bool(not test_set[used].any()),
        "all_batch_rows_in_train_split": bool(train_set[used].all()),
        "no_valid_rows_in_batches": bool(not valid_set[used].any()),
        "loss_is_masked_bce": MET.masked_bce_loss.__module__ == "pc_metrics",
        "observed_values_binary": bool(
            np.isin(inputs.labels[used][observed], (0.0, 1.0)).all()
        ),
        "n_observed_entries": int(observed.sum()),
        "n_masked_entries": int((~observed).sum()),
        "nan_policy": "unobserved (NaN) entries are excluded from the loss, never filled with 0",
        "passed": bool(
            int(PC.N_TASKS) == 128
            and np.isnan(inputs.labels[inputs.test_rows]).all()
            and not test_set[used].any()
            and train_set[used].all()
            and np.isin(inputs.labels[used][observed], (0.0, 1.0)).all()
        ),
    }


# ---------------------------------------------------------------------------
# throughput
# ---------------------------------------------------------------------------


def run_throughput(inputs, batches, device, n_steps: int, warmup: int, label: str, cls) -> dict[str, Any]:
    st = _build_state(inputs, device)
    _reset_rng()
    transfer = cls(device)
    _sync()
    for rows in batches[:warmup]:
        one_step(st, inputs, rows, device, transfer)
    _sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall: list[float] = []
    losses: list[float] = []
    t0 = time.perf_counter()
    for rows in batches[warmup : warmup + n_steps]:
        t = time.perf_counter()
        out = one_step(st, inputs, rows, device, transfer)
        wall.append(time.perf_counter() - t)
        losses.append(out["loss"])
    total = time.perf_counter() - t0
    _sync()
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    transfer.release()
    st_ms = _stats(wall)
    return {
        "label": label,
        "transfer": type(transfer).__name__,
        "n_steps": int(n_steps),
        "wall_seconds": float(total),
        "steps_per_second": float(n_steps / total),
        "step_ms": st_ms,
        "total_wall_based_mean_ms": float(total / n_steps * 1e3),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses": losses,
        "gpu_peak_allocated_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "steps_per_epoch": int(st["steps_per_epoch"]),
        "projected_epoch_seconds": float(st["steps_per_epoch"] * (st_ms["mean_ms"] * 1e-3)),
        "transfer_counters": dict(transfer.counters),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", default="all", help="all | correctness | throughput")
    parser.add_argument("--timed-steps", type=int, default=PF.TIMED_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=PF.WARMUP_STEPS)
    parser.add_argument("--recheck-steps", type=int, default=100)
    parser.add_argument("--out", default=str(PF.ANALYSIS / "throughput_comparison.json"))
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    inputs = PF.load_inputs()
    need = max(args.warmup_steps + args.timed_steps + args.recheck_steps, 40)
    batches = PF.fixed_batches(inputs.train_rows, need)
    chosen = PF.diverse_batches(inputs.cache, batches)
    shapes = PF.batch_shape_table(inputs.cache, batches[: args.warmup_steps + args.timed_steps])

    result: dict[str, Any] = {
        "schema": "lcrc-pcba-pipeline-throughput-v1",
        "round_id": PF.ROUND_ID,
        "created_utc": PF.utcnow(),
        "git_commit": PF.git_commit(),
        "environment": PF.environment(),
        "device": str(device),
        "protocol": {
            "batch_size": PF.BATCH_SIZE,
            "seed": PF.SEED,
            "warmup_steps": int(args.warmup_steps),
            "timed_steps": int(args.timed_steps),
            "batch_plan": PF.batch_plan_fingerprint(batches[: args.warmup_steps + args.timed_steps]),
            "measured_batch_shapes": {
                "atoms": _dist(shapes["atoms"]),
                "bonds": _dist(shapes["bonds"]),
                "s2": _dist(shapes["s2"]),
            },
            "variants": {
                "A_original": "pc_model.to_device (pageable, per-tensor, blocking) + torch.as_tensor(labels)",
                "B_optimized": "PinnedStager (pinned staging + non_blocking copies, depth=2)",
            },
            "unchanged": [
                "lcrc_data.collate (frozen)", "batch order", "model init", "loss",
                "optimizer", "scheduler", "fp32", "train/valid split",
            ],
        },
        "inputs": {
            "cache": inputs.identity,
            "labels_sha256": PF.sha256_file(PF.LABELS_PATH),
            "feature_spec_hash": inputs.spec.hash(),
            "norm_fingerprint": inputs.norm["norm_fingerprint"],
        },
    }

    stages = args.stages
    if stages in ("all", "correctness"):
        correctness = {
            "g1_collate_and_device_batch": gate1_collate_and_device_batch(inputs, batches, device, chosen),
            "g2_g3_inference": gate2_3_inference(inputs, batches[0], device),
            "g4_one_step": gate4_one_step(inputs, batches, device),
            "g5_ten_steps": gate5_ten_steps(inputs, batches, device),
            "g6_labels_and_test_isolation": gate6_labels_and_test_isolation(inputs, batches),
        }
        correctness["all_passed"] = bool(
            correctness["g1_collate_and_device_batch"]["passed"]
            and correctness["g2_g3_inference"]["passed"]
            and correctness["g4_one_step"]["passed"]
            and correctness["g5_ten_steps"]["passed"]
            and correctness["g6_labels_and_test_isolation"]["passed"]
        )
        result["correctness"] = correctness
        PF.log(f"correctness gates: all_passed={correctness['all_passed']}")

    if stages in ("all", "throughput"):
        a = run_throughput(
            inputs, batches, device, args.timed_steps, args.warmup_steps, "A_original", PIPE.PageableTransfer
        )
        b = run_throughput(
            inputs, batches, device, args.timed_steps, args.warmup_steps, "B_optimized", PIPE.PinnedStager
        )
        # drift control: repeat A after B on the same batches
        a2 = run_throughput(
            inputs, batches, device, args.recheck_steps, args.warmup_steps, "A_original_recheck",
            PIPE.PageableTransfer,
        )
        speedup_p50 = a["step_ms"]["p50_ms"] / b["step_ms"]["p50_ms"]
        speedup_mean = a["step_ms"]["mean_ms"] / b["step_ms"]["mean_ms"]
        speedup_total = a["steps_per_second"] / b["steps_per_second"]
        result["throughput"] = {
            "A_original": a,
            "B_optimized": b,
            "A_original_recheck": a2,
            "speedup": {
                "p50": float(speedup_p50),
                "mean": float(speedup_mean),
                "wall_total": float(speedup_total),
                "definition": "step time A / step time B; >1 means B is faster",
            },
            "drift_check": {
                "A_p50_ms_first": a["step_ms"]["p50_ms"],
                "A_p50_ms_recheck": a2["step_ms"]["p50_ms"],
                "relative_drift": float(a2["step_ms"]["p50_ms"] / a["step_ms"]["p50_ms"] - 1.0),
                "note": "same host, same batches, after the B pass; guards against clock drift",
            },
            "projected_epoch": {
                "steps_per_epoch": a["steps_per_epoch"],
                "A_seconds": a["projected_epoch_seconds"],
                "B_seconds": b["projected_epoch_seconds"],
                "A_minutes": a["projected_epoch_seconds"] / 60.0,
                "B_minutes": b["projected_epoch_seconds"] / 60.0,
                "note": "pure train-step projection, Validation and logging excluded",
            },
            "losses_identical": bool(
                max(abs(x - y) for x, y in zip(a["losses"], b["losses"])) <= TOL["loss_max_abs"]
            ),
            "max_loss_abs_diff": float(max(abs(x - y) for x, y in zip(a["losses"], b["losses"]))),
            "gpu_memory_delta_bytes": {
                "peak_allocated": b["gpu_peak_allocated_bytes"] - a["gpu_peak_allocated_bytes"],
                "peak_reserved": b["gpu_peak_reserved_bytes"] - a["gpu_peak_reserved_bytes"],
            },
        }
        PF.log(
            f"A {a['step_ms']['p50_ms']:.1f} ms/step ({a['steps_per_second']:.2f} step/s) -> "
            f"B {b['step_ms']['p50_ms']:.1f} ms/step ({b['steps_per_second']:.2f} step/s) "
            f"= {speedup_p50:.3f}x (p50)"
        )

    correctness_ok = result.get("correctness", {}).get("all_passed")
    speedup = result.get("throughput", {}).get("speedup", {}).get("p50")
    result["decision"] = {
        "correctness_all_passed": correctness_ok,
        "speedup_p50": speedup,
        "threshold": 1.5,
        "label": (
            PF.RESULT_LABEL_ACCEPTED
            if (correctness_ok and speedup is not None and speedup >= 1.5)
            else PF.RESULT_LABEL_NO_SPEEDUP
        ),
        "rule": "PERF_OPTIMIZATION_ACCEPTED iff every correctness gate passes and end-to-end step throughput improves >= 1.5x",
    }
    result["finished_utc"] = PF.utcnow()
    PF.write_json(args.out, result)
    PF.log(f"wrote {PF.track_rel(args.out)} label={result['decision']['label']}")
    return 0


def _dist(a: Sequence[int]) -> dict[str, Any]:
    arr = np.asarray(list(a), dtype=np.float64)
    return {
        "min": int(arr.min()),
        "p50": float(np.median(arr)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
