"""P1 training runner: A0 / A1 / A2 / A3 (240 epochs) and P1a (480 epochs).

All arms share the frozen C1 encoder, the same seed-0 initialisation, the same
batch order, the same labels, the same optimiser and the same checkpoint rule.
The count branch is a bias-free zero-initialised ``Linear(2048, 10)`` applied to
``log1p(counts)``.  No count aggregation crosses graph boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import pc_common as C

GUARD = C.install_test_access_guard()

import pc_data as PD  # noqa: E402
import pc_p1_model as PM  # noqa: E402
from bp_data import collate  # noqa: E402


class TimeLimit(RuntimeError):
    pass


def write_curve(path: Path, curve: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)


def batches(rows: np.ndarray, batch_size: int, generator: torch.Generator) -> np.ndarray:
    perm = torch.randperm(len(rows), generator=generator).numpy()
    return np.asarray(rows, dtype=np.int64)[perm]


def recipe_for(arm: str, epochs: int, base_recipe: dict[str, Any]) -> dict[str, Any]:
    recipe = dict(base_recipe)
    recipe.update(
        {
            "optimizer": "AdamW",
            "peak_lr": 3.0e-4,
            "weight_decay": 1.0e-3,
            "adam_betas": (0.9, 0.999),
            "adam_eps": 1e-8,
            "min_lr": 1e-5,
            "dropout": 0.10,
            "batch_size": 64,
            "gradient_clip": 5.0,
            "max_epochs": int(epochs),
            "min_epochs": int(epochs),
            "patience": 10**9,
            "scheduler": "per_step_cosine",
            "scheduler_horizon_epochs": int(epochs),
            "loss": "BCEWithLogitsLoss over 10 independent labels (unweighted)",
            "selection": "max Validation mean average precision (macro AP over 10 labels); tie -> earlier epoch",
            "early_stop": "disabled: fixed epoch budget for every arm",
            "arm": arm,
        }
    )
    return recipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(PM.ARM_SPECS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-seconds", type=float, default=8400.0)
    parser.add_argument("--smoke-epochs", type=int, default=None)
    args = parser.parse_args(argv)
    arm = args.arm
    seed = int(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(out / "run.json", {"status": "RESOURCE_BLOCKED", "reason": "cuda unavailable"})
        return 2
    torch.set_num_threads(8)
    t_start = time.perf_counter()

    task = PD.load_c1_task()
    epochs = int(PM.ARM_SPECS[arm]["epochs"])
    if args.smoke_epochs is not None:
        epochs = int(args.smoke_epochs)
    recipe = recipe_for(arm, epochs, task["recipe"])
    counts_kind = PM.ARM_SPECS[arm]["counts"]
    counts = PM.load_counts(counts_kind) if counts_kind else None
    train_rows = np.asarray(task["train_rows"], dtype=np.int64)
    valid_rows = np.asarray(task["valid_rows"], dtype=np.int64)
    y_train = np.asarray(task["y_train"], dtype=np.float32)
    y_valid = np.asarray(task["y_valid"], dtype=np.float32)
    counts_train = counts["train"] if counts is not None else None
    counts_valid = counts["valid"] if counts is not None else None
    counts_alignment = {
        "train_gids_match_rows": bool(counts is None or np.array_equal(counts["train_gids"], train_rows)),
        "valid_gids_match_rows": bool(counts is None or np.array_equal(counts["valid_gids"], valid_rows)),
        "train_counts_rows": int(counts_train.shape[0]) if counts_train is not None else 0,
        "valid_counts_rows": int(counts_valid.shape[0]) if counts_valid is not None else 0,
        "counts_dim": int(counts_train.shape[1]) if counts_train is not None else 0,
    }
    if not (counts_alignment["train_gids_match_rows"] and counts_alignment["valid_gids_match_rows"]):
        raise RuntimeError(f"count row alignment failure: {counts_alignment}")

    model, model_info = PM.build_arm(task, arm, seed, device)
    structural_gate = {
        "arm": arm,
        "counts_kind": counts_kind,
        "parameters": int(model_info["total_parameters"]),
        "base_parameters": int(model_info["base_parameters"]),
        "branch_parameters": int(model_info["branch_parameters"]),
        "expected_total_parameters": int(model_info["expected_total_parameters"]),
        "parameters_match": bool(model_info["total_parameters"] == model_info["expected_total_parameters"]),
        "base_init_matches_history": bool(model_info["base_init_matches_history"]),
        "base_init_state_hash": model_info["base_init_state_hash"],
        "historical_init_state_hash": model_info["historical_init_state_hash"],
        "feature_spec_hash": task["spec"].hash(),
        "norm_fingerprint": task["norm"]["norm_fingerprint"],
        "cache_identity": task["cache_identity"],
        "counts_alignment": counts_alignment,
    }
    if not structural_gate["parameters_match"] or not structural_gate["base_init_matches_history"]:
        C.write_json(
            out / "run.json",
            {
                "status": "FAILED",
                "reason": "STRUCTURAL_GATE_FAILED",
                "structural_gate": structural_gate,
                "environment": C.environment(),
            },
        )
        return 2

    # Initialisation gate: the zero branch must not change the C1 logits.
    init_gate: dict[str, Any] = {"checked": False}
    first_step_gate: dict[str, Any] = {"checked": False}
    if counts_kind is not None:
        probe_rows = valid_rows[:64]
        probe_batch = PM.to_device(collate(task["valid_cache"], probe_rows.tolist()), device)
        probe_counts = torch.as_tensor(counts_valid[probe_rows], dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            base_logits = model.base(probe_batch)
            full_logits = model(probe_batch, probe_counts)
            branch_delta = model.count_linear(torch.log1p(probe_counts))
        init_gate = {
            "checked": True,
            "branch_weight_all_zero": bool(torch.count_nonzero(model.count_linear.weight).item() == 0),
            "max_abs_output_delta_vs_base": float((full_logits - base_logits).abs().max().item()),
            "max_abs_branch_delta": float(branch_delta.abs().max().item()),
            "probe_graphs": int(probe_rows.size),
            "branch_init_weight_sha256": C.sha256_array(model.count_linear.weight.detach().cpu().numpy()),
        }
        if init_gate["max_abs_output_delta_vs_base"] != 0.0 or not init_gate["branch_weight_all_zero"]:
            C.write_json(
                out / "run.json",
                {"status": "FAILED", "reason": "BRANCH_INIT_GATE_FAILED", "init_gate": init_gate},
            )
            return 2

    # Deterministic shared batch order and training RNG start.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    steps_per_epoch = int(math.ceil(train_rows.size / int(recipe["batch_size"])))
    total_updates = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_updates, 1), eta_min=float(recipe["min_lr"])
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve: list[dict[str, Any]] = []
    curve_path = out / "curve.csv"
    checkpoint_path = out / "checkpoint.pt"
    best_metric = -math.inf
    best_epoch = None
    best_state = None
    best_valid_probs = None
    best_valid_aps: list[float] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    epochs_run = 0
    first_epoch_rows_sha = None
    first_step = True
    try:
        for epoch in range(1, epochs + 1):
            if time.perf_counter() - t_start > args.max_seconds:
                raise TimeLimit(f"exceeded max_seconds before epoch {epoch}")
            model.train()
            t_epoch = time.perf_counter()
            rows_epoch = batches(train_rows, int(recipe["batch_size"]), generator)
            if epoch == 1:
                first_epoch_rows_sha = C.sha256_array(rows_epoch)
            losses: list[float] = []
            for start in range(0, rows_epoch.size, int(recipe["batch_size"])):
                if time.perf_counter() - t_start > args.max_seconds:
                    raise TimeLimit(f"exceeded max_seconds during epoch {epoch}")
                batch_rows = np.asarray(rows_epoch[start : start + int(recipe["batch_size"])], dtype=np.int64)
                batch = PM.to_device(collate(task["train_cache"], batch_rows.tolist()), device)
                cb = None
                if counts_kind is not None:
                    cb = torch.as_tensor(counts_train[batch_rows], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
                if cb is None:
                    logits = model(batch)
                else:
                    logits = model(batch, cb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if first_step and counts_kind is not None:
                    grad = model.count_linear.weight.grad
                    gnorm_branch = float(grad.norm().item()) if grad is not None else 0.0
                    first_step_gate = {
                        "checked": True,
                        "branch_grad_norm": gnorm_branch,
                        "branch_grad_finite": bool(grad is not None and bool(torch.isfinite(grad).all().item())),
                        "branch_grad_nonzero": bool(gnorm_branch > 0.0),
                        "epoch": int(epoch),
                        "batch_rows": int(batch_rows.size),
                    }
                    if not (first_step_gate["branch_grad_finite"] and first_step_gate["branch_grad_nonzero"]):
                        raise RuntimeError(f"branch first-step gradient gate failed: {first_step_gate}")
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
                if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                    raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
                if gnorm > float(recipe["gradient_clip"]):
                    clip_events += 1
                grad_norm_max = max(grad_norm_max, gnorm)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu().item()))
                first_step = False
            train_seconds = time.perf_counter() - t_epoch
            t_eval = time.perf_counter()
            valid_logits = PM.predict_logits(
                model, task["valid_cache"], valid_rows, counts_valid, device, batch_size=256
            )
            valid_probs = PM.sigmoid_np(valid_logits)
            valid_macro, valid_aps = PM.macro_ap(np.asarray(task["y_valid"], dtype=np.float64), valid_probs)
            valid_seconds = time.perf_counter() - t_eval
            improved = valid_macro > best_metric
            if improved:
                best_metric = float(valid_macro)
                best_epoch = int(epoch)
                best_valid_probs = valid_probs
                best_valid_aps = list(valid_aps)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "state_dict": best_state,
                        "arm": arm,
                        "counts_kind": counts_kind,
                        "seed": int(seed),
                        "best_epoch": int(best_epoch),
                        "best_metric": float(best_metric),
                        "model_info": model_info,
                        "recipe": recipe,
                        "feature_spec_hash": task["spec"].hash(),
                        "norm_fingerprint": task["norm"]["norm_fingerprint"],
                        "cache_identity": task["cache_identity"],
                        "test_evaluated": False,
                    },
                    checkpoint_path,
                )
            lr_now = float(optimizer.param_groups[0]["lr"])
            row: dict[str, Any] = {
                "epoch": int(epoch),
                "lr": lr_now,
                "train_seconds": float(train_seconds),
                "valid_seconds": float(valid_seconds),
                "epoch_seconds": float(train_seconds + valid_seconds),
                "elapsed_seconds": float(time.perf_counter() - t_start),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "valid_macro_ap": float(valid_macro),
                "valid_macro_ap_pct": 100.0 * float(valid_macro),
                "is_best": bool(improved),
            }
            for j, name in enumerate(C.LABELS):
                row[f"valid_ap_{name}"] = float(valid_aps[j])
            curve.append(row)
            epochs_run = int(epoch)
            write_curve(curve_path, curve)
            if device.type == "cuda":
                gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
                gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
            if epoch % 10 == 0 or improved:
                C.log(
                    f"{arm} epoch {epoch}/{epochs} loss {row['train_loss']:.5f} "
                    f"valid {100*valid_macro:.3f}%{' *best*' if improved else ''} lr {lr_now:.3e}"
                )
    except TimeLimit as exc:
        C.write_json(
            out / "run.json",
            {
                "status": "RESOURCE_BLOCKED",
                "reason": "PER_TASK_TIME_LIMIT",
                "detail": str(exc),
                "arm": arm,
                "epochs_run": int(epochs_run),
                "curve_path": C.track_rel(curve_path),
                "max_seconds": float(args.max_seconds),
                "test_evaluated": False,
                "guard": GUARD,
                "environment": C.environment(),
            },
        )
        C.log(f"RESOURCE_BLOCKED: {exc}")
        return 3

    if best_state is None or best_epoch is None:
        raise RuntimeError("no checkpoint selected")
    # Re-derive the best epoch from the written curve (earliest on ties).
    metrics = [row["valid_macro_ap"] for row in curve]
    argmax = int(np.argmax(metrics)) + 1
    tie_consistent = int(argmax) == int(best_epoch)
    if not tie_consistent:
        raise RuntimeError(f"selection re-derivation mismatch: {argmax} vs {best_epoch}")

    # Reload the selected checkpoint into a fresh model and independently re-evaluate.
    reloaded, reload_info = PM.build_arm(task, arm, seed, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()}, strict=True)
    reloaded.eval()
    reload_logits = PM.predict_logits(
        reloaded, task["valid_cache"], valid_rows, counts_valid, device, batch_size=256
    )
    reload_probs = PM.sigmoid_np(reload_logits)
    reload_macro, reload_aps = PM.macro_ap(np.asarray(task["y_valid"], dtype=np.float64), reload_probs)
    reload_delta = abs(float(reload_macro) - float(best_metric))
    if reload_delta > 1e-4:
        raise RuntimeError(f"checkpoint reload does not reproduce Validation AP: delta {reload_delta}")

    train_logits = PM.predict_logits(
        reloaded, task["train_cache"], train_rows, counts_train, device, batch_size=256
    )
    train_probs = PM.sigmoid_np(train_logits)
    train_macro, train_aps = PM.macro_ap(np.asarray(task["y_train"], dtype=np.float64), train_probs)
    np.savez_compressed(
        out / "predictions.npz",
        train_probs=train_probs.astype(np.float32),
        valid_probs=reload_probs.astype(np.float32),
        y_train=np.asarray(task["y_train"], dtype=np.float32),
        y_valid=np.asarray(task["y_valid"], dtype=np.float32),
        labels=np.asarray(C.LABELS),
    )

    branch_post_gate: dict[str, Any] = {"checked": False}
    if counts_kind is not None:
        with torch.no_grad():
            cb = torch.as_tensor(counts_valid, dtype=torch.float32, device=device)
            delta = reloaded.count_linear(torch.log1p(cb))
            weight = reloaded.count_linear.weight.detach().cpu().numpy()
        branch_post_gate = {
            "checked": True,
            "weight_norm": float(np.linalg.norm(weight)),
            "weight_abs_max": float(np.abs(weight).max()),
            "weight_sha256": C.sha256_array(weight),
            "weight_fraction_nonzero": float(np.count_nonzero(weight) / weight.size),
            "valid_branch_delta_mean_abs": float(delta.abs().mean().item()),
            "valid_branch_delta_std": float(delta.std().item()),
            "valid_branch_delta_max_abs": float(delta.abs().max().item()),
            "valid_branch_delta_fraction_nonzero": float(torch.count_nonzero(delta).item() / delta.numel()),
            "not_identically_zero": bool(np.any(weight != 0.0)),
        }
        if not branch_post_gate["not_identically_zero"]:
            raise RuntimeError("trained count branch is still identically zero")

    total_seconds = float(time.perf_counter() - t_start)
    selection = {
        "arm": arm,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_valid_macro_ap": float(best_metric),
        "best_valid_macro_ap_pct": 100.0 * float(best_metric),
        "reload_valid_macro_ap": float(reload_macro),
        "reload_delta": float(reload_delta),
        "tie_consistent": bool(tie_consistent),
        "per_label_valid_ap": {name: best_valid_aps[j] for j, name in enumerate(C.LABELS)},
        "checkpoint_sha256": C.sha256_file(checkpoint_path),
        "curve_sha256": C.sha256_file(curve_path),
        "test_evaluated": False,
    }
    C.write_json(out / "selection.json", selection)
    result = {
        "status": "complete_no_test",
        "arm": arm,
        "label": PM.ARM_SPECS[arm]["label"],
        "counts_kind": counts_kind,
        "seed": int(seed),
        "epochs_configured": int(epochs),
        "epochs_run": int(epochs_run),
        "stopped_early": False,
        "best_epoch": int(best_epoch),
        "best_valid_macro_ap": float(best_metric),
        "best_valid_macro_ap_pct": 100.0 * float(best_metric),
        "last_epoch": int(curve[-1]["epoch"]),
        "last_epoch_valid_macro_ap": float(curve[-1]["valid_macro_ap"]),
        "last_epoch_train_loss": float(curve[-1]["train_loss"]),
        "reload_valid_macro_ap": float(reload_macro),
        "reload_delta": float(reload_delta),
        "train_macro_ap": float(train_macro),
        "train_macro_ap_pct": 100.0 * float(train_macro),
        "valid_macro_ap": float(reload_macro),
        "valid_macro_ap_pct": 100.0 * float(reload_macro),
        "per_label_train_ap": {name: train_aps[j] for j, name in enumerate(C.LABELS)},
        "per_label_valid_ap": {name: reload_aps[j] for j, name in enumerate(C.LABELS)},
        "model_info": reload_info,
        "structural_gate": structural_gate,
        "init_gate": init_gate,
        "first_step_gate": first_step_gate,
        "branch_post_gate": branch_post_gate,
        "recipe": recipe,
        "batch_size": int(recipe["batch_size"]),
        "steps_per_epoch": int(steps_per_epoch),
        "total_updates": int(total_updates),
        "first_epoch_rows_sha256": first_epoch_rows_sha,
        "label_hashes": {
            "train": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_train.npy"),
            "valid": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_valid.npy"),
        },
        "feature_spec_hash": task["spec"].hash(),
        "norm_fingerprint": task["norm"]["norm_fingerprint"],
        "cache_identity": task["cache_identity"],
        "selection": selection,
        "curve_sha256": C.sha256_file(curve_path),
        "checkpoint_sha256": C.sha256_file(checkpoint_path),
        "predictions_sha256": C.sha256_file(out / "predictions.npz"),
        "seconds": total_seconds,
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "grad_clip_events": int(clip_events),
        "grad_norm_max": float(grad_norm_max),
        "test_evaluated": False,
        "test_artifacts_opened": 0,
        "guard": GUARD,
        "smoke": bool(args.smoke_epochs is not None),
        "max_seconds": float(args.max_seconds),
        "rr_env": {k: v for k, v in os.environ.items() if k.startswith("RR_")},
        "environment": C.environment(),
        "finished_utc": C.utcnow(),
    }
    C.write_json(out / "run.json", result)
    C.log(
        f"{arm} done: best epoch {best_epoch}, Validation macro AP {100*best_metric:.3f}%, "
        f"Train macro AP {100*train_macro:.3f}%, {total_seconds/3600:.3f} GPU-hours"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
