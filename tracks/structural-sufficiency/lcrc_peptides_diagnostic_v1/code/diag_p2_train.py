"""P2 — one longer LCRC-C1 Peptides-func run (seed 0, 240-epoch cosine horizon).

Architecture, input features, S1/S2, readout, loss, batch size and optimiser are
the frozen Fast v1 objects: ``bp_model.build_c1`` + ``bp_data`` caches + the Fast
v1 Peptides-func spec.  The rebuilt model is checked against the historical seed0
checkpoint (parameter count and initialisation hash) before a single step runs.

Only the schedule changes (max 240 epochs, min 80, patience 40, cosine horizon
240).  Test data, Test labels and Test predictions are blocked by an audit hook.

Usage (GPU box, repository root):

    python tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/code/diag_p2_train.py \
        --device cuda --out-dir tracks/.../runs/peptide_c1_seed0_240ep --max-seconds 4200
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import diag_common as C

GUARD = C.install_test_access_guard()

import bp_data as D  # noqa: E402
import bp_metrics as MET  # noqa: E402
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-seconds", type=float, default=4200.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=None,
        help="engineering smoke only: cap max_epochs (records smoke=true; never used for the formal run)",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(out_dir / "run.json", {"status": "RESOURCE_BLOCKED", "reason": "cuda unavailable"})
        return 2

    t_start = time.perf_counter()
    data = T._make_task("peptide", C.DATASET, int(args.seed))
    recipe = dict(data.recipe)
    recipe.update({k: v for k, v in RECIPE_OVERRIDES.items() if k != "seed"})
    recipe["scheduler"] = "per_step_cosine"
    recipe["optimizer"] = "AdamW"
    recipe["loss"] = "BCEWithLogitsLoss over 10 independent labels"
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

    norm_payload = {
        **data.norm,
        "continuous_atom": {
            "dim": int(data.cont_norm.get("dim", 0)),
            "mean": data.cont_norm.get("mean", []),
            "scale": data.cont_norm.get("scale", []),
            "fingerprint": data.cont_norm.get("fingerprint", ""),
        },
    }
    def build() -> tuple[torch.nn.Module, dict[str, Any]]:
        """Fresh frozen C1 initialisation (identical to Fast v1's seed-mapped init)."""
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        return BM.build_c1(
            data.spec,
            norm_payload,
            data.n_out,
            int(args.seed),
            device,
            cont_slice=data.cont_slice,
            cont_mean=np.asarray(data.cont_norm.get("mean", []), dtype=np.float32),
            cont_scale=np.asarray(data.cont_norm.get("scale", []), dtype=np.float32),
        )

    model, model_info = build()
    params = int(BM.parameter_count(model))
    hist_ckpt = torch.load(C.history_checkpoint_path(0), map_location="cpu", weights_only=False)
    structural = {
        "parameters": params,
        "expected_parameters": EXPECTED_PARAMETERS,
        "parameters_match_history": params == int(hist_ckpt["model_info"]["parameters"]),
        "init_state_hash": model_info["init_state_hash"],
        "history_init_state_hash": hist_ckpt["model_info"]["init_state_hash"],
        "init_state_hash_match_history_seed0": model_info["init_state_hash"]
        == hist_ckpt["model_info"]["init_state_hash"],
        "local_layout_match": model_info["local_layout"] == hist_ckpt["model_info"]["local_layout"],
        "feature_spec_hash": data.spec.hash(),
        "history_feature_spec_hash": hist_ckpt["feature_spec_hash"],
        "spec_hash_match": data.spec.hash() == hist_ckpt["feature_spec_hash"],
        "norm_fingerprint": data.norm["norm_fingerprint"],
        "history_norm_fingerprint": hist_ckpt["norm_fingerprint"],
        "norm_fingerprint_match": data.norm["norm_fingerprint"] == hist_ckpt["norm_fingerprint"],
    }
    if not (
        structural["parameters_match_history"]
        and structural["init_state_hash_match_history_seed0"]
        and structural["local_layout_match"]
        and structural["spec_hash_match"]
        and structural["norm_fingerprint_match"]
    ):
        C.write_json(
            out_dir / "run.json",
            {
                "status": "FAILED",
                "reason": "STRUCTURAL_MISMATCH_WITH_FAST_V1",
                "structural": structural,
                "environment": C.environment(),
            },
        )
        return 2
    C.log(f"structural gate PASS: {params} parameters, init hash matches Fast v1 seed0")

    batch_sizes = [int(recipe["batch_size"])] + [int(x) for x in recipe["batch_size_fallbacks"]]
    result = None
    for attempt, batch_size in enumerate(batch_sizes):
        if attempt:
            model, model_info = build()  # a fallback restarts from the frozen initialisation
        try:
            result = _train(
                data=data,
                model=model,
                model_info=model_info,
                recipe=recipe,
                batch_size=int(batch_size),
                out_dir=out_dir,
                device=device,
                max_seconds=float(args.max_seconds),
                seed=int(args.seed),
                structural=structural,
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
    C.log(f"done: best valid macro AP {result['best_valid_macro_ap_pct']:.2f}% at epoch {result['best_epoch']}")
    return 0


def _train(
    *,
    data,
    model,
    model_info,
    recipe,
    batch_size: int,
    out_dir: Path,
    device: torch.device,
    max_seconds: float,
    seed: int,
    structural: dict[str, Any],
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
        for rows in T._batches(data.train_rows, batch_size, generator):
            if time.perf_counter() - t_start > max_seconds:
                raise TimeLimit(f"exceeded max_seconds during epoch {epoch}")
            batch = T._to_device(D.collate(data.train_cache, rows.tolist()), device)
            y = torch.as_tensor(data.y_train[rows], dtype=torch.float32, device=device)
            logits = model(batch)
            loss = MET.masked_bce_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
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
                    "continuous_atom_norm": {
                        "dim": int(data.cont_norm.get("dim", 0)),
                        "mean": data.cont_norm.get("mean", []),
                        "scale": data.cont_norm.get("scale", []),
                        "fingerprint": data.cont_norm.get("fingerprint", ""),
                    },
                    "cache_identity": data.cache_identity,
                    "model_info": model_info,
                    "recipe": recipe,
                    "batch_size": int(batch_size),
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
                "is_best": bool(improved),
                "valid_metric": float(metric),
                "valid_per_label_ap": list(valid_metrics["per_label"]),
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
                f"valid {metric:.6f}{' *best*' if improved else ''} lr {lr_now:.3e}"
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
    }
    C.write_json(out_dir / "selection.json", selection)

    # reload and independently reproduce the selected Validation metric
    reloaded, _ = BM.build_c1(
        data.spec,
        {
            **data.norm,
            "continuous_atom": {
                "dim": int(data.cont_norm.get("dim", 0)),
                "mean": data.cont_norm.get("mean", []),
                "scale": data.cont_norm.get("scale", []),
                "fingerprint": data.cont_norm.get("fingerprint", ""),
            },
        },
        data.n_out,
        int(seed),
        device,
        cont_slice=data.cont_slice,
        cont_mean=np.asarray(data.cont_norm.get("mean", []), dtype=np.float32),
        cont_scale=np.asarray(data.cont_norm.get("scale", []), dtype=np.float32),
    )
    reloaded.load_state_dict({k: v.to(device) for k, v in tracker.best_state.items()}, strict=True)
    reloaded.eval()
    valid_re, valid_logits, valid_probs = T.evaluate(
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

    hist_best = float(C.history_run(0)["best_metric"])
    phase1 = phase_best(curve, 1, FIRST_PHASE_END)
    phase2 = phase_best(curve, FIRST_PHASE_END + 1, max_epochs)
    overall = phase_best(curve, 1, max_epochs)
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
        "recipe": recipe,
        "batch_size": int(batch_size),
        "epochs_run": int(epochs_run),
        "stopped_early": bool(stopped_early),
        "best_epoch": int(tracker.best_epoch),
        "best_valid_macro_ap": float(tracker.best_metric),
        "best_valid_macro_ap_pct": 100.0 * float(tracker.best_metric),
        "reload_valid_macro_ap": float(valid_re["value"]),
        "reload_valid_delta": reload_delta,
        "phase_first_120": phase1,
        "phase_121_240": phase2,
        "overall_best": overall,
        "fast_v1_seed0_best_valid_macro_ap": hist_best,
        "fast_v1_seed0_best_valid_macro_ap_pct": 100.0 * hist_best,
        "delta_vs_fast_v1_seed0_pp": 100.0 * (float(tracker.best_metric) - hist_best),
        "delta_phase1_vs_fast_v1_seed0_pp": None if phase1 is None else 100.0 * (phase1["best_valid_macro_ap"] - hist_best),
        "delta_phase2_vs_phase1_pp": None
        if (phase1 is None or phase2 is None)
        else 100.0 * (phase2["best_valid_macro_ap"] - phase1["best_valid_macro_ap"]),
        "scheduler_note": "cosine horizon is 240 epochs, so the comparison with the 120-epoch Fast v1 seed0 run includes a different LR trajectory",
        "per_label_valid_ap": {name: valid_ap[j] for j, name in enumerate(C.LABELS)},
        "per_label_train_ap": {name: train_ap[j] for j, name in enumerate(C.LABELS)},
        "valid_macro_ap_per_label_mean": float(np.mean(valid_ap)),
        "train_macro_ap_per_label_mean": float(np.mean(train_ap)),
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
        "cache_identity": data.cache_identity,
        "feature_spec_hash": data.spec.hash(),
        "norm_fingerprint": data.norm["norm_fingerprint"],
        "test_evaluated": False,
        "test_artifacts_opened": 0,
        "guard": GUARD,
        "smoke": bool(smoke),
        "max_seconds": float(max_seconds),
        "environment": C.environment(),
        "finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "run.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
