"""Formal weighted-BCE run: Peptides-func / LCRC-C1 / seed 0 / 240 epochs.

The only intervention versus ``lcrc_peptides_diagnostic_v1/runs/peptide_c1_seed0_240ep``
is the training loss: ordinary masked BCE -> BCEWithLogits with a per-label
``pos_weight = N_negative / N_positive`` computed from the official Train labels.
Architecture, feature caches, normalisation, initialisation, optimiser, schedule,
early stopping and checkpoint selection are unchanged.

Usage (GPU box, repository root):

    uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_p2_train.py \
        --device cuda --out-dir tracks/.../runs/weighted_bce_seed0 --max-seconds 4200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import wb_common as C
import wb_loss as L
import wb_pos_weight as PW

GUARD = C.GUARD

import bp_data as D  # noqa: E402
import bp_model as BM  # noqa: E402
import bp_train as T  # noqa: E402

RECIPE_OVERRIDES = {
    "seed": 0,
    "max_epochs": 240,
    "min_epochs": 80,
    "patience": 40,
    "peak_lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    "dropout": 0.10,
    "batch_size": 64,
    "scheduler_horizon_epochs": 240,
    "min_lr": 1.0e-5,
    "gradient_clip": 5.0,
}
EXPECTED_PARAMETERS = 464739
FIRST_PHASE_END = 120


class TimeLimit(RuntimeError):
    pass


def write_curve(path: Path, curve: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)


def phase_best(curve: list[dict[str, Any]], lo: int, hi: int) -> dict[str, Any] | None:
    rows = [r for r in curve if lo <= r["epoch"] <= hi]
    if not rows:
        return None
    best = max(rows, key=lambda r: r["valid_metric"])
    return {
        "epoch_range": [lo, hi],
        "n_epochs_present": len(rows),
        "best_valid_macro_ap": float(best["valid_metric"]),
        "best_valid_macro_ap_pct": 100.0 * float(best["valid_metric"]),
        "best_epoch": int(best["epoch"]),
        "last_epoch_valid_macro_ap": float(rows[-1]["valid_metric"]),
    }


def _build(data, seed: int, device: torch.device):
    norm_payload = {
        **data.norm,
        "continuous_atom": {
            "dim": int(data.cont_norm.get("dim", 0)),
            "mean": data.cont_norm.get("mean", []),
            "scale": data.cont_norm.get("scale", []),
            "fingerprint": data.cont_norm.get("fingerprint", ""),
        },
    }
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return BM.build_c1(
        data.spec,
        norm_payload,
        data.n_out,
        int(seed),
        device,
        cont_slice=data.cont_slice,
        cont_mean=np.asarray(data.cont_norm.get("mean", []), dtype=np.float32),
        cont_scale=np.asarray(data.cont_norm.get("scale", []), dtype=np.float32),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-seconds", type=float, default=4200.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-weight-json", default=str(C.POS_WEIGHT_JSON))
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=None,
        help="engineering smoke only: cap max_epochs (records smoke=true; never used for the formal run)",
    )
    args = parser.parse_args(argv)

    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(out_dir / "run.json", {"status": "RESOURCE_BLOCKED", "reason": "cuda unavailable"})
        return 2
    if device.type != "cuda":
        C.write_json(out_dir / "run.json", {"status": "FAILED", "reason": "formal run requires cuda"})
        return 2

    t_start = time.perf_counter()
    data = T._make_task("peptide", C.DATASET, seed)
    pos_payload = C.read_json(Path(args.pos_weight_json))
    try:
        pos_check = PW.verify_payload(pos_payload, data.y_train, labels=C.LABELS)
    except PW.PosWeightError as exc:
        C.write_json(
            out_dir / "run.json",
            {
                "status": "FAILED",
                "reason": "POS_WEIGHT_VERIFICATION_FAILED",
                "detail": str(exc),
                "pos_weight_file": C.track_rel(Path(args.pos_weight_json)),
                "environment": C.environment(),
            },
        )
        C.log(f"POS_WEIGHT_VERIFICATION_FAILED: {exc}")
        return 2

    ref_train = np.load(C.TRAIN_LABELS)
    if not np.array_equal(np.asarray(data.y_train), np.asarray(ref_train)):
        C.write_json(
            out_dir / "run.json",
            {"status": "FAILED", "reason": "TRAIN_LABEL_ARRAY_MISMATCH_FAST_V1"},
        )
        return 2
    if not np.isfinite(np.asarray(data.y_train, dtype=np.float64)).all():
        C.write_json(out_dir / "run.json", {"status": "FAILED", "reason": "NONFINITE_TRAIN_LABELS"})
        return 2

    pos_weight = torch.as_tensor(pos_payload["pos_weight"], dtype=torch.float32, device=device)
    pos_weight_used = [float(x) for x in pos_weight.detach().cpu().tolist()]
    pos_weight_file_sha = C.sha256_file(Path(args.pos_weight_json))
    pos_weight_payload_sha = PW.payload_sha256(pos_payload)

    recipe = dict(data.recipe)
    recipe.update({k: v for k, v in RECIPE_OVERRIDES.items() if k != "seed"})
    recipe["scheduler"] = "per_step_cosine"
    recipe["optimizer"] = "AdamW"
    recipe["loss"] = (
        "WeightedBCEWithLogitsLoss: F.binary_cross_entropy_with_logits("
        "logits, targets, pos_weight=Train_Nneg/Npos, reduction='mean')"
    )
    recipe["pos_weight_source"] = "official Train labels only"
    recipe["selection"] = "max Validation mean average precision (macro AP over 10 labels)"
    recipe["early_stop"] = (
        f"after min_epochs={recipe['min_epochs']}, stop when {recipe['patience']} consecutive epochs "
        "do not strictly improve best Validation mean AP"
    )
    if args.smoke_epochs is not None:
        recipe["max_epochs"] = int(args.smoke_epochs)
        recipe["min_epochs"] = int(args.smoke_epochs) + 10**6
        recipe["patience"] = 10**6
    recipe["test_policy"] = "not evaluated in this round; Test is blocked by an audit hook"
    recipe["round_id"] = C.ROUND_ID
    recipe["result_label"] = C.RESULT_LABEL

    if args.smoke_epochs is None:
        expected = {
            "max_epochs": 240,
            "min_epochs": 80,
            "patience": 40,
            "peak_lr": 3.0e-4,
            "weight_decay": 1.0e-3,
            "dropout": 0.10,
            "batch_size": 64,
            "scheduler_horizon_epochs": 240,
            "min_lr": 1.0e-5,
            "gradient_clip": 5.0,
        }
        bad = {k: (recipe.get(k), v) for k, v in expected.items() if recipe.get(k) != v}
        if bad:
            C.write_json(
                out_dir / "run.json",
                {"status": "FAILED", "reason": "RECIPE_MISMATCH", "detail": bad},
            )
            return 2

    model, model_info = _build(data, seed, device)
    params = int(BM.parameter_count(model))
    hist_ckpt = torch.load(C.FASTV1_SEED0_CHECKPOINT, map_location="cpu", weights_only=False)
    hist_run = C.read_json(C.HISTORY_RUN_JSON)
    historical_init_hash = hist_ckpt["model_info"]["init_state_hash"]
    historical_run_init_hash = hist_run["model_info"]["init_state_hash"]
    structural = {
        "parameters": params,
        "expected_parameters": EXPECTED_PARAMETERS,
        "parameters_match_history": params == EXPECTED_PARAMETERS,
        "parameters_match_diagnostic_run": params == int(hist_run["parameters"]),
        "init_state_hash": model_info["init_state_hash"],
        "fast_v1_seed0_init_state_hash": historical_init_hash,
        "diagnostic_run_init_state_hash": historical_run_init_hash,
        "init_state_hash_match_fast_v1_seed0": model_info["init_state_hash"] == historical_init_hash,
        "init_state_hash_match_diagnostic_run": model_info["init_state_hash"] == historical_run_init_hash,
        "local_layout_match": model_info["local_layout"] == hist_ckpt["model_info"]["local_layout"],
        "local_layout_match_diagnostic_run": model_info["local_layout"] == hist_run["model_info"]["local_layout"],
        "feature_spec_hash": data.spec.hash(),
        "diagnostic_feature_spec_hash": hist_run["feature_spec_hash"],
        "spec_hash_match_diagnostic_run": data.spec.hash() == hist_run["feature_spec_hash"],
        "norm_fingerprint": data.norm["norm_fingerprint"],
        "diagnostic_norm_fingerprint": hist_run["norm_fingerprint"],
        "norm_fingerprint_match_diagnostic_run": data.norm["norm_fingerprint"] == hist_run["norm_fingerprint"],
        "cache_identity": data.cache_identity,
        "diagnostic_cache_identity": hist_run["cache_identity"],
        "cache_identity_match_diagnostic_run": dict(data.cache_identity) == dict(hist_run["cache_identity"]),
    }
    structural["all_match"] = bool(
        structural["parameters_match_history"]
        and structural["parameters_match_diagnostic_run"]
        and structural["init_state_hash_match_fast_v1_seed0"]
        and structural["init_state_hash_match_diagnostic_run"]
        and structural["local_layout_match"]
        and structural["local_layout_match_diagnostic_run"]
        and structural["spec_hash_match_diagnostic_run"]
        and structural["norm_fingerprint_match_diagnostic_run"]
        and structural["cache_identity_match_diagnostic_run"]
    )
    if not structural["all_match"]:
        C.write_json(
            out_dir / "run.json",
            {
                "status": "FAILED",
                "reason": "STRUCTURAL_MISMATCH_WITH_HISTORY",
                "structural": structural,
                "environment": C.environment(),
            },
        )
        C.log("STRUCTURAL_MISMATCH_WITH_HISTORY")
        return 2
    C.log(
        f"structural gate PASS: {params} parameters, init hash {model_info['init_state_hash'][:12]} "
        f"matches Fast v1 seed0 and Diagnostic v1"
    )

    batch_sizes = [int(recipe["batch_size"])] + [int(x) for x in recipe["batch_size_fallbacks"]]
    result = None
    for attempt, batch_size in enumerate(batch_sizes):
        if attempt:
            model, model_info = _build(data, seed, device)
        try:
            result = _train(
                data=data,
                model=model,
                model_info=model_info,
                recipe=recipe,
                batch_size=int(batch_size),
                batch_fallback_used=bool(attempt),
                out_dir=out_dir,
                device=device,
                max_seconds=float(args.max_seconds),
                seed=seed,
                structural=structural,
                pos_payload=pos_payload,
                pos_check=pos_check,
                pos_weight=pos_weight,
                pos_weight_used=pos_weight_used,
                pos_weight_file_sha=pos_weight_file_sha,
                pos_weight_payload_sha=pos_weight_payload_sha,
                t_start=t_start,
                smoke=bool(args.smoke_epochs is not None),
            )
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            C.log(f"CUDA OOM at batch {batch_size}; trying the frozen fallback")
            continue
        except TimeLimit as exc:
            C.write_json(
                out_dir / "run.json",
                {
                    "status": "RESOURCE_BLOCKED",
                    "reason": "PER_TASK_TIME_LIMIT",
                    "detail": str(exc),
                    "curve_path": C.track_rel(out_dir / "curve.csv"),
                    "max_seconds": float(args.max_seconds),
                    "test_evaluated": False,
                    "test_access_attempts": list(GUARD["attempts"]),
                    "environment": C.environment(),
                },
            )
            C.log(f"RESOURCE_BLOCKED PER_TASK_TIME_LIMIT: {exc}")
            return 3
    if result is None:
        C.write_json(
            out_dir / "run.json",
            {"status": "RESOURCE_BLOCKED", "reason": "CUDA_OOM_ALL_BATCH_SIZES", "environment": C.environment()},
        )
        return 3
    C.log(
        f"done: best valid macro AP {result['best_valid_macro_ap_pct']:.2f}% at epoch {result['best_epoch']} "
        f"(history {result['history_best_valid_macro_ap_pct']:.2f}%)"
    )
    return 0


def _train(
    *,
    data,
    model,
    model_info,
    recipe,
    batch_size: int,
    batch_fallback_used: bool,
    out_dir: Path,
    device: torch.device,
    max_seconds: float,
    seed: int,
    structural: dict[str, Any],
    pos_payload: dict[str, Any],
    pos_check: dict[str, Any],
    pos_weight: torch.Tensor,
    pos_weight_used: list[float],
    pos_weight_file_sha: str,
    pos_weight_payload_sha: str,
    t_start: float,
    smoke: bool = False,
) -> dict[str, Any]:
    max_epochs = int(recipe["max_epochs"])
    min_epochs = int(recipe["min_epochs"])
    patience = int(recipe["patience"])
    horizon = int(recipe["scheduler_horizon_epochs"])
    steps_per_epoch = int(math.ceil(data.n_train / batch_size))
    total_updates = steps_per_epoch * horizon
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
    generator = torch.Generator().manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    tracker = T.Tracker(data.kind)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    epochs_run = 0
    stopped_early = False
    checkpoint_path = out_dir / "checkpoint.pt"
    curve_path = out_dir / "curve.csv"

    for epoch in range(1, max_epochs + 1):
        if time.perf_counter() - t_start > max_seconds:
            raise TimeLimit(f"exceeded max_seconds before epoch {epoch}")
        model.train()
        t_epoch = time.perf_counter()
        losses: list[float] = []
        epoch_grad_norms: list[float] = []
        epoch_clip_events = 0
        for rows in T._batches(data.train_rows, batch_size, generator):
            if time.perf_counter() - t_start > max_seconds:
                raise TimeLimit(f"exceeded max_seconds during epoch {epoch}")
            batch = T._to_device(D.collate(data.train_cache, rows.tolist()), device)
            y = torch.as_tensor(data.y_train[rows], dtype=torch.float32, device=device)
            logits = model(batch)
            loss = L.weighted_bce_with_logits(logits, y, pos_weight, n_labels=C.N_LABELS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
                epoch_clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            epoch_grad_norms.append(gnorm)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))
        train_seconds = time.perf_counter() - t_epoch
        t_eval = time.perf_counter()
        valid_metrics, _, _ = T.evaluate(
            data.kind, data.dataset, model, data.valid_cache, data.valid_rows, data.y_valid, device,
            n_classes=data.n_out,
        )
        valid_seconds = time.perf_counter() - t_eval
        metric = float(valid_metrics["value"])
        improved = tracker.consider(epoch, metric, float("nan"), valid_metrics)
        if improved:
            tracker.capture(model)
            torch.save(
                {
                    "state_dict": tracker.best_state,
                    "round_id": C.ROUND_ID,
                    "result_label": C.RESULT_LABEL,
                    "kind": data.kind,
                    "dataset": data.dataset,
                    "unit": int(data.unit),
                    "seed": int(seed),
                    "variant": "C1",
                    "n_outputs": int(data.n_out),
                    "best_epoch": int(tracker.best_epoch),
                    "epochs_run_so_far": int(epoch),
                    "best_metric": float(tracker.best_metric),
                    "feature_spec_hash": data.spec.hash(),
                    "norm_fingerprint": data.norm["norm_fingerprint"],
                    "cache_identity": data.cache_identity,
                    "model_info": model_info,
                    "recipe": recipe,
                    "batch_size": int(batch_size),
                    "batch_fallback_used": bool(batch_fallback_used),
                    "loss": recipe["loss"],
                    "pos_weight": pos_weight_used,
                    "pos_weight_file_sha256": pos_weight_file_sha,
                    "pos_weight_payload_sha256": pos_weight_payload_sha,
                    "pos_weight_verification": pos_check,
                    "structural_gate": structural,
                    "saved_at_epoch": int(epoch),
                    "precheck": False,
                    "test_evaluated": False,
                },
                checkpoint_path,
            )
        lr_now = float(optimizer.param_groups[0]["lr"])
        curve.append(
            {
                "epoch": int(epoch),
                "steps": int(steps_per_epoch),
                "lr": lr_now,
                "train_seconds": float(train_seconds),
                "valid_seconds": float(valid_seconds),
                "epoch_seconds": float(train_seconds + valid_seconds),
                "elapsed_seconds": float(time.perf_counter() - t_start),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "grad_norm_mean": float(np.mean(epoch_grad_norms)) if epoch_grad_norms else float("nan"),
                "grad_norm_max": float(np.max(epoch_grad_norms)) if epoch_grad_norms else float("nan"),
                "clip_events_epoch": int(epoch_clip_events),
                "clip_events_cumulative": int(clip_events),
                "is_best": bool(improved),
                "valid_metric": float(metric),
                "valid_per_label_ap": json.dumps(valid_metrics["per_label"]),
            }
        )
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        epochs_run = int(epoch)
        write_curve(curve_path, curve)
        if epoch % 10 == 0 or improved:
            C.log(
                f"epoch {epoch}/{max_epochs} train_loss {curve[-1]['train_loss']:.5f} "
                f"valid {metric:.6f}{' *best*' if improved else ''} lr {lr_now:.3e} "
                f"grad {curve[-1]['grad_norm_mean']:.3f}/{curve[-1]['grad_norm_max']:.3f}"
            )
        if epoch >= min_epochs and epoch - tracker.last_strict_improve >= patience:
            stopped_early = True
            C.log(f"early stop at epoch {epoch}; best epoch {tracker.best_epoch} ({tracker.best_metric:.6f})")
            break

    if tracker.best_epoch is None or tracker.best_state is None:
        raise RuntimeError("no checkpoint selected")
    tie = tracker.tie_check()
    if not tie["consistent"]:
        raise RuntimeError(f"selection re-derivation inconsistent: {tie}")
    total_seconds = time.perf_counter() - t_start
    checkpoint_sha = C.sha256_file(checkpoint_path)

    selection = {
        "best_epoch": int(tracker.best_epoch),
        "best_metric": float(tracker.best_metric),
        "scores": tracker.scores,
        "tie_check": tie,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": C.track_rel(checkpoint_path),
        "selection_finished_utc": C.utcnow(),
        "test_evaluated": False,
        "test_access_attempts": list(GUARD["attempts"]),
    }
    C.write_json(out_dir / "selection.json", selection)

    reloaded, _ = _build(data, seed, device)
    reloaded.load_state_dict({k: v.to(device) for k, v in tracker.best_state.items()}, strict=True)
    reloaded.eval()
    valid_re, valid_logits, _ = T.evaluate(
        data.kind, data.dataset, reloaded, data.valid_cache, data.valid_rows, data.y_valid, device,
        n_classes=data.n_out,
    )
    reload_delta = abs(float(valid_re["value"]) - float(tracker.best_metric))
    if reload_delta > 1e-4:
        raise RuntimeError(f"reloaded checkpoint does not reproduce best Validation AP: delta {reload_delta}")
    train_logits = T.predict_logits(reloaded, data.train_cache, data.train_rows, device)
    train_probs = 1.0 / (1.0 + np.exp(-train_logits.astype(np.float64)))
    valid_probs = 1.0 / (1.0 + np.exp(-valid_logits.astype(np.float64)))
    from sklearn.metrics import average_precision_score

    y_train = np.asarray(data.y_train, dtype=np.float64)
    y_valid = np.asarray(data.y_valid, dtype=np.float64)
    train_ap = [float(average_precision_score(y_train[:, j], train_probs[:, j])) for j in range(C.N_LABELS)]
    valid_ap = [float(average_precision_score(y_valid[:, j], valid_probs[:, j])) for j in range(C.N_LABELS)]
    np.savez_compressed(
        out_dir / "train_valid_predictions.npz",
        train_probs=train_probs,
        valid_probs=valid_probs,
        y_train=y_train,
        y_valid=y_valid,
        labels=np.asarray(C.LABELS),
    )

    hist_run = C.read_json(C.HISTORY_RUN_JSON)
    hist_valid = [float(hist_run["per_label_valid_ap"][name]) for name in C.LABELS]
    hist_train = [float(hist_run["per_label_train_ap"][name]) for name in C.LABELS]
    delta_valid = [100.0 * (valid_ap[j] - hist_valid[j]) for j in range(C.N_LABELS)]
    delta_train = [100.0 * (train_ap[j] - hist_train[j]) for j in range(C.N_LABELS)]
    gap_train = [train_ap[j] - valid_ap[j] for j in range(C.N_LABELS)]
    gap_hist = [hist_train[j] - hist_valid[j] for j in range(C.N_LABELS)]
    best_row = max(curve, key=lambda r: r["valid_metric"])
    phase1 = phase_best(curve, 1, FIRST_PHASE_END)
    phase2 = phase_best(curve, FIRST_PHASE_END + 1, max_epochs)
    overall = phase_best(curve, 1, max_epochs)
    hist_best = float(hist_run["best_valid_macro_ap"])
    hist_best_epoch = int(hist_run["best_epoch"])

    result = {
        "status": "complete_no_test",
        "kind": data.kind,
        "dataset": data.dataset,
        "unit": int(data.unit),
        "seed": int(seed),
        "variant": "C1",
        "n_outputs": int(data.n_out),
        "parameters": int(BM.parameter_count(model)),
        "model_info": model_info,
        "structural_gate": structural,
        "intervention": {
            "only_change": "training loss",
            "loss": "WeightedBCEWithLogitsLoss (per-label pos_weight)",
            "pos_weight_formula": "N_negative_j / N_positive_j on official Train only",
            "pos_weight_source": "official Train split; no Validation or Test statistics",
            "pos_weight_file": C.track_rel(Path(C.POS_WEIGHT_JSON)),
            "pos_weight_file_sha256": pos_weight_file_sha,
            "pos_weight_payload_sha256": pos_weight_payload_sha,
            "pos_weight": pos_weight_used,
            "pos_weight_less_than_one": [
                {"label": C.LABELS[j], "pos_weight": pos_weight_used[j]}
                for j in range(C.N_LABELS)
                if pos_weight_used[j] < 1.0
            ],
            "max_pos_weight": max(pos_weight_used),
            "verification": pos_check,
        },
        "recipe": recipe,
        "batch_size": int(batch_size),
        "batch_fallback_used": bool(batch_fallback_used),
        "strict_single_variable_control": bool(batch_size == 64),
        "epochs_run": int(epochs_run),
        "stopped_early": bool(stopped_early),
        "best_epoch": int(tracker.best_epoch),
        "best_valid_macro_ap": float(tracker.best_metric),
        "best_valid_macro_ap_pct": 100.0 * float(tracker.best_metric),
        "reload_valid_macro_ap": float(valid_re["value"]),
        "reload_valid_delta": reload_delta,
        "valid_macro_ap_per_label_mean": float(np.mean(valid_ap)),
        "train_macro_ap_per_label_mean": float(np.mean(train_ap)),
        "train_valid_macro_gap": float(np.mean(train_ap) - np.mean(valid_ap)),
        "best_epoch_train_loss": float(best_row["train_loss"]),
        "best_epoch_grad_norm_mean": float(best_row["grad_norm_mean"]),
        "best_epoch_grad_norm_max": float(best_row["grad_norm_max"]),
        "last_epoch_train_loss": float(curve[-1]["train_loss"]),
        "last_epoch_valid_macro_ap": float(curve[-1]["valid_metric"]),
        "phase_first_120": phase1,
        "phase_121_240": phase2,
        "overall_best": overall,
        "per_label_train_ap": {name: train_ap[j] for j, name in enumerate(C.LABELS)},
        "per_label_valid_ap": {name: valid_ap[j] for j, name in enumerate(C.LABELS)},
        "per_label_delta_valid_ap_pp_vs_history": {name: delta_valid[j] for j, name in enumerate(C.LABELS)},
        "per_label_delta_train_ap_pp_vs_history": {name: delta_train[j] for j, name in enumerate(C.LABELS)},
        "per_label_train_valid_gap": {name: gap_train[j] for j, name in enumerate(C.LABELS)},
        "per_label_history_train_valid_gap": {name: gap_hist[j] for j, name in enumerate(C.LABELS)},
        "per_label_gap_change_pp": {name: 100.0 * (gap_train[j] - gap_hist[j]) for j, name in enumerate(C.LABELS)},
        "history_reference": {
            "run_json": C.track_rel(C.HISTORY_RUN_JSON),
            "loss": "ordinary BCEWithLogitsLoss",
            "best_epoch": hist_best_epoch,
            "best_valid_macro_ap": hist_best,
            "best_valid_macro_ap_pct": 100.0 * hist_best,
            "train_macro_ap_per_label_mean": float(hist_run["train_macro_ap_per_label_mean"]),
            "train_valid_macro_gap": float(
                hist_run["train_macro_ap_per_label_mean"] - hist_run["best_valid_macro_ap"]
            ),
            "checkpoint_sha256": hist_run["checkpoint_sha256"],
        },
        "delta_vs_history_best_valid_macro_ap_pp": 100.0 * (float(tracker.best_metric) - hist_best),
        "delta_vs_history_train_macro_ap_pp": 100.0 * (
            float(np.mean(train_ap)) - float(hist_run["train_macro_ap_per_label_mean"])
        ),
        "delta_vs_history_train_valid_gap_pp": 100.0 * (
            (float(np.mean(train_ap)) - float(np.mean(valid_ap)))
            - (float(hist_run["train_macro_ap_per_label_mean"]) - float(hist_run["best_valid_macro_ap"]))
        ),
        "curve_sha256": C.sha256_file(curve_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": C.track_rel(checkpoint_path),
        "seconds": float(total_seconds),
        "train_seconds": float(sum(r["train_seconds"] for r in curve)),
        "valid_seconds": float(sum(r["valid_seconds"] for r in curve)),
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "grad_clip_events": int(clip_events),
        "grad_norm_max": float(grad_norm_max),
        "nonfinite_events": 0,
        "cache_identity": data.cache_identity,
        "feature_spec_hash": data.spec.hash(),
        "norm_fingerprint": data.norm["norm_fingerprint"],
        "test_evaluated": False,
        "test_access_attempts": list(GUARD["attempts"]),
        "guard": GUARD,
        "smoke": bool(smoke),
        "max_seconds": float(max_seconds),
        "gpu_budget_hours": 1.25,
        "gpu_hours_used": float(total_seconds) / 3600.0,
        "environment": C.environment(),
        "finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "run.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
