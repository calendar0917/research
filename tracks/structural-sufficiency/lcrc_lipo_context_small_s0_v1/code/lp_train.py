"""One formal training run: one arm (B / S / R1 / R2 / D2 / G), seed 0.

Every completed epoch ends with the *same weights* evaluated on the **full**
Train, Validation and Test splits (independent iterator, RNG snapshot/restore,
``inference_mode``), and the per-epoch metrics are appended to ``curve.csv``.
Test is evaluated live because this round has development-evaluation Test
access; it never selects anything - the checkpoint rule is strictly
"lowest Validation RMSE, ties keep the earlier epoch".

Artifacts per arm (inside the run directory):

``curve.csv``                 per-epoch: RMSE/MAE/bias x 3 splits, timings, state hash
``metrics.json``              best-Val / last-epoch / late-window summary + full curve digest
``checkpoint_best.pt``        best-Val weights (+ provenance)
``checkpoint_last.pt``        final-epoch weights
``pred_*_{best,last}.npz``    per-molecule predictions for Train/Valid/Test
``resume.pt``                 (removed on success) restart state for an engineering retry
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import lp_common as C
import lp_data as D
import lp_model as M

import lcpr_dmpnn as PDN
import lcrc_data as PREP_DATA

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


class BestTracker:
    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_rmse = math.inf
        self.state: dict[str, torch.Tensor] | None = None
        self.improvements: list[int] = []

    def consider(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        rmse = float(metrics["rmse"])
        if self.best_epoch is not None and rmse >= self.best_rmse:
            return False
        self.best_epoch = int(epoch)
        self.best_rmse = rmse
        self.improvements.append(int(epoch))
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# per-arm predictors
# ---------------------------------------------------------------------------


class ArmPredictor:
    """Full-split prediction over the structure caches (C1-family arms)."""

    def __init__(self, model, view: D.View, device):
        self.model = model
        self.view = view
        self.device = device

    def __call__(self, split: str) -> np.ndarray:
        return D.predict_cache(self.model, self.view.caches[split], self.device)


class DmpnnPredictor:
    """Full-split prediction over the chemprop-native feature archive."""

    def __init__(self, model, view: D.View, feat: Mapping[str, np.ndarray], device):
        self.model = model
        self.view = view
        self.feat = feat
        self.device = device

    def __call__(self, split: str) -> np.ndarray:
        rows = [int(r) for r in np.asarray(self.view.rows[split]).tolist()]
        out: list[np.ndarray] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(rows), C.EVAL_BATCH_SIZE):
                    chunk = rows[start : start + C.EVAL_BATCH_SIZE]
                    bmg = PDN._batch(self.feat, chunk, self.device)
                    out.append(self.model(bmg).detach().float().cpu().numpy().reshape(-1))
        finally:
            self.model.train(was_training)
        return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _plan_hash(plan: Mapping[str, Any]) -> str:
    return str(plan["sha256"])


def train_run(
    *,
    arm: str,
    device: torch.device,
    out_dir: Path,
    epochs: int | None = None,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if arm not in C.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    planned_epochs = C.arm_epochs(arm) if epochs is None else int(epochs)
    if not smoke and int(planned_epochs) != C.arm_epochs(arm):
        raise ValueError(f"formal {arm} runs must use {C.arm_epochs(arm)} epochs")

    determinism = C.setup_determinism()
    view = D.build_view()
    n_train = int(view.caches["train"]["n_molecules"])
    batch_size = int(C.C1_RECIPE["batch_size"] if arm != C.DM_ARM else C.DMPNN_RECIPE["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * planned_epochs
    target_norm = view.target_norm

    # ---- model / shared initial state ------------------------------------
    feat = None
    init_report: dict[str, Any] = {}
    if arm == C.DM_ARM:
        with np.load(C.DMPNN_FEATURES, allow_pickle=False) as handle:
            feat = {k: np.asarray(handle[k]) for k in handle.files}
        if int(feat["n_molecules"]) != C.N_MOLECULES:
            raise ValueError("D-MPNN feature archive does not cover every molecule")
        torch.manual_seed(int(C.TRAIN_SEED))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(C.TRAIN_SEED))
        model = PDN.DMPNNReference(d_h=int(C.DMPNN_RECIPE["hidden"]))
        init_report = {"kind": "dmpnn", "feature_version": PDN.FEATURE_VERSION, "smoke": bool(smoke)}
        params_total = int(sum(p.numel() for p in model.parameters()))
        init_state_hash = D.state_hash(model.state_dict())
        predictor = DmpnnPredictor(model, view, feat, device)
    else:
        import dissect_model as _DM

        full_state = _DM.build_full_state(view.spec, view.train_norm, int(C.TRAIN_SEED))
        full_init_hash = D.state_hash(full_state)
        c1_state = _DM.remap_state("C1", full_state)
        model = M.build_arm(view.spec, view.train_norm, arm)
        model.base.load_state_dict(c1_state, strict=True)
        if model.kind == "context":
            init_info = model.init_new_modules()
            drawn_hash = C.sha256_bytes(
                b"".join(np.ascontiguousarray(v.numpy(), dtype=np.float32).tobytes() for _, v in sorted(init_info["drawn"].items()))
            )
            init_report = {
                "kind": "context",
                "private_init_seed": init_info["generator_seed"],
                "private_init_device": init_info["generator_device"],
                "private_init_drawn_hash": drawn_hash,
                "zeroed": init_info["zeroed"],
                "layer_norm_identity": init_info["layer_norm"],
            }
        else:
            init_report = {"kind": "base"}
            drawn_hash = None
        params_total = M.parameter_count(model)
        expected = C.EXPECTED_PARAMS[arm]
        if params_total != expected:
            raise RuntimeError(f"{arm}: trainable parameters {params_total} != frozen {expected}")
        init_report["full_init_state_hash"] = full_init_hash
        init_report["c1_init_state_hash"] = D.state_hash(c1_state)
        if drawn_hash is not None:
            init_report["drawn_hash"] = drawn_hash
        init_state_hash = D.state_hash(model.state_dict())
        model.to(device)
        predictor = ArmPredictor(model, view, device)

    expected = C.EXPECTED_PARAMS.get(arm)
    if expected is not None and params_total != expected:
        raise RuntimeError(f"{arm}: trainable parameters {params_total} != frozen {expected}")
    if arm == C.DM_ARM:
        model.to(device)

    # ---- optimiser / schedule -------------------------------------------
    recipe = dict(C.C1_RECIPE if arm != C.DM_ARM else C.DMPNN_RECIPE)
    torch.manual_seed(int(C.TRAIN_SEED))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(C.TRAIN_SEED))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if arm == C.DM_ARM:
        optimizer = torch.optim.Adam(trainable, lr=float(recipe["optimizer_base_lr"]), weight_decay=float(recipe["weight_decay"]))
        warmup_steps = steps_per_epoch * int(recipe["warmup_epochs"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: PDN._noam_lambda(
                step,
                warmup_steps,
                max(total_updates - warmup_steps, 1),
                float(recipe["init_lr"]),
                float(recipe["peak_lr"]),
                float(recipe["min_lr"]),
            ),
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(recipe["peak_lr"]),
            betas=tuple(float(x) for x in recipe["adam_betas"]),
            eps=float(recipe["adam_eps"]),
            weight_decay=float(recipe["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=float(recipe["min_lr"])
        )

    # ---- shared epoch plan ----------------------------------------------
    plan = D.epoch_plan(n_train, planned_epochs, int(C.TRAIN_SEED))

    y_train_scaled_full = view.y_scaled("train").to(device)
    train_rows_global = np.asarray(view.rows["train"], dtype=np.int64)

    tracker = BestTracker()
    curve: list[dict[str, Any]] = []
    start_epoch = 1
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak = 0
    reference_lr_trace: list[float] = []

    resume_path = out_dir / "resume.pt"
    if resume and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in payload["model"].items()}, strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        curve = payload["curve"]
        start_epoch = int(payload["epoch"]) + 1
        tracker.best_epoch = payload["best_epoch"]
        tracker.best_rmse = payload["best_rmse"]
        tracker.improvements = payload["improvements"]
        tracker.state = payload["best_state"]
        clip_events = int(payload.get("clip_events", 0))
        grad_norm_max = float(payload.get("grad_norm_max", 0.0))
        C.log(f"[{arm}] resuming at epoch {start_epoch} (best {tracker.best_rmse:.6f} @ {tracker.best_epoch})")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    for epoch in range(start_epoch, planned_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = np.asarray(plan["plan"][epoch - 1], dtype=np.int64)
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        abs_sum = 0.0
        sq_sum = 0.0
        seen = 0
        epoch_clips = 0
        epoch_gnorm_max = 0.0
        for rows in batches:
            if arm == C.DM_ARM:
                global_rows = train_rows_global[rows]
                batch = PDN._batch(feat, global_rows, device)
                y = y_train_scaled_full[torch.as_tensor(rows, device=device)]
                pred = model(batch).reshape(-1)
            else:
                batch = D.to_device(PREP_DATA.collate(view.caches["train"], rows), device)
                y = y_train_scaled_full[torch.as_tensor(rows, device=device)]
                pred = model(batch).reshape(-1)
            loss = F.mse_loss(pred, y)
            err = pred - y
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
                epoch_clips += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            epoch_gnorm_max = max(epoch_gnorm_max, gnorm)
            optimizer.step()
            scheduler.step()
            seen += int(y.numel())
        train_seconds = time.perf_counter() - t0
        reference_lr_trace.append(float(optimizer.param_groups[0]["lr"]))

        t_eval = time.perf_counter()
        metrics, _ = D.evaluate_all_splits(predictor, view, device)
        eval_seconds = time.perf_counter() - t_eval

        improved = tracker.consider(epoch, metrics["valid"])
        if improved:
            tracker.capture(model)
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))
        train_mse = float(sq_sum / max(seen, 1))
        curve.append(
            {
                "epoch": epoch,
                "steps": steps_per_epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm_max": float(epoch_gnorm_max),
                "grad_clip_events": int(epoch_clips),
                "train_seconds": float(train_seconds),
                "eval_seconds": float(eval_seconds),
                "is_best": bool(improved),
                "train_mse_standardized_online": train_mse,
                "train_rmse_standardized_online": math.sqrt(train_mse),
                "train_mae_standardized_online": float(abs_sum / max(seen, 1)),
                "state_hash": D.state_hash(model.state_dict()),
                **{f"{split}_rmse": float(metrics[split]["rmse"]) for split in C.SPLITS},
                **{f"{split}_mae": float(metrics[split]["mae"]) for split in C.SPLITS},
                **{f"{split}_bias": float(metrics[split]["bias"]) for split in C.SPLITS},
                **{f"{split}_r2": float(metrics[split]["r2"]) for split in C.SPLITS},
            }
        )
        if epoch % 20 == 0 or epoch == planned_epochs or improved:
            torch.save(
                {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "curve": curve,
                    "best_epoch": tracker.best_epoch,
                    "best_rmse": tracker.best_rmse,
                    "best_state": tracker.state,
                    "improvements": tracker.improvements,
                    "clip_events": clip_events,
                    "grad_norm_max": grad_norm_max,
                },
                resume_path,
            )

    wall = time.perf_counter() - started
    if tracker.best_epoch is None or tracker.state is None:
        raise RuntimeError("training ended without any validation improvement")

    # ---- reload best, re-verify, and score -------------------------------
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last_state_hash = D.state_hash(last_state)
    best_state_hash = D.state_hash(tracker.state)
    model.load_state_dict({k: v.to(device) for k, v in tracker.state.items()})
    best_metrics, best_preds = D.evaluate_all_splits(predictor, view, device, keep_predictions=True)
    recheck = abs(float(best_metrics["valid"]["rmse"]) - float(tracker.best_rmse))
    if recheck > C.REPRO_TOL:
        raise RuntimeError(f"reloaded best Valid RMSE differs by {recheck}")

    curve_valid = np.asarray([row["valid_rmse"] for row in curve], dtype=np.float64)
    argbest = int(np.argmin(curve_valid) + 1)
    if argbest != int(tracker.best_epoch):
        raise RuntimeError(f"curve argbest {argbest} != tracked best {tracker.best_epoch}")

    model.load_state_dict({k: v.to(device) for k, v in last_state.items()})
    last_metrics, last_preds = D.evaluate_all_splits(predictor, view, device, keep_predictions=True)
    last_recorded = {
        split: {
            "rmse": float(curve[-1][f"{split}_rmse"]),
            "mae": float(curve[-1][f"{split}_mae"]),
        }
        for split in C.SPLITS
    }
    for split in C.SPLITS:
        if abs(last_metrics[split]["rmse"] - last_recorded[split]["rmse"]) > C.REPRO_TOL:
            raise RuntimeError(f"last-epoch {split} RMSE replay differs for {arm}")

    # ---- artifacts ------------------------------------------------------
    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    for tag, state, preds, metrics in (
        ("best", tracker.state, best_preds, best_metrics),
        ("last", last_state, last_preds, last_metrics),
    ):
        torch.save(
            {
                "state_dict": state,
                "arm": arm,
                "arm_spec": dict(C.ARM_SPECS[arm]),
                "dataset": C.DATASET,
                "split_method": C.OGB_SPLIT_METHOD,
                "schema": view.spec.hash() if arm != C.DM_ARM else PDN.FEATURE_VERSION,
                "seed": int(C.TRAIN_SEED),
                "protocol_id": C.PROTOCOL_ID,
                "checkpoint": tag,
                "best_epoch": int(tracker.best_epoch),
                "epoch": int(curve[-1]["epoch"]) if tag == "last" else int(tracker.best_epoch),
                "state_hash": best_state_hash if tag == "best" else last_state_hash,
                "init_state_hash": init_state_hash,
                "parameters": params_total,
                "target_norm": dict(target_norm),
                "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
                "epoch_plan_sha256": _plan_hash(plan),
                "metrics": {split: dict(metrics[split]) for split in C.SPLITS},
            },
            out_dir / f"checkpoint_{tag}.pt",
        )
        for split in C.SPLITS:
            np.savez(
                out_dir / f"pred_{split}_{tag}.npz",
                rows=np.asarray(view.rows[split], dtype=np.int64),
                rows_sha256=C.row_sha256(view.rows[split].tolist()),
                standardised=preds[split],
                raw=view.to_raw(preds[split]),
                y_true=view.y[split],
                checkpoint=tag,
                epoch=int(tracker.best_epoch) if tag == "best" else int(curve[-1]["epoch"]),
            )

    late = int(C.late_window(arm))
    late_rows = curve[-late:]
    summary = {
        "best_valid": {
            "epoch": int(tracker.best_epoch),
            **{split: dict(best_metrics[split]) for split in C.SPLITS},
        },
        "last_epoch": {
            "epoch": int(curve[-1]["epoch"]),
            **{split: dict(last_metrics[split]) for split in C.SPLITS},
        },
        "late_window": {
            "window": late,
            "epochs": [int(row["epoch"]) for row in late_rows],
            **{
                f"{split}_{stat}": float(np.mean([row[f"{split}_{stat}"] for row in late_rows]))
                for split in C.SPLITS
                for stat in ("rmse", "mae", "bias")
            },
            "train_rmse_standardized_online": float(
                np.mean([row["train_rmse_standardized_online"] for row in late_rows])
            ),
            "note": "mean of per-epoch metrics over the window; not a prediction ensemble",
        },
    }

    if resume_path.is_file():
        resume_path.unlink()

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "dataset": C.DATASET,
        "task": C.TASK,
        "status": "completed",
        "smoke": bool(smoke),
        "arm": arm,
        "arm_title": C.ARM_TITLES[arm],
        "arm_spec": dict(C.ARM_SPECS[arm]),
        "seed": int(C.TRAIN_SEED),
        "device": str(device),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "determinism": determinism,
        "environment": C.env_info(),
        "code_hashes": C.code_hashes(),
        "data": view.audit(),
        "init": init_report,
        "model": {
            "parameters": params_total,
            "parameters_expected": expected,
            "parameters_match": bool(expected is None or params_total == expected),
            "width": C.WIDTH if arm != C.DM_ARM else int(C.DMPNN_RECIPE["hidden"]),
            "new_parameters": C.NEW_PARAMS if arm in C.CONTEXT_ARMS else 0,
            "context_layers": C.N_CONTEXT_LAYERS if arm in C.CONTEXT_ARMS else 0,
            "source_hop_max": C.ARM_SOURCE_HOPS[arm],
        },
        "recipe": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in recipe.items()},
            "epochs": int(planned_epochs),
            "batch_size": int(batch_size),
            "steps_per_epoch": int(steps_per_epoch),
            "total_updates": int(total_updates),
            "warmup_steps": int(steps_per_epoch * int(C.DMPNN_RECIPE["warmup_epochs"])) if arm == C.DM_ARM else 0,
            "loss": "mse_on_train_standardised_target",
            "checkpoint_rule": "lowest_validation_rmse__strict_improvement__tie_keeps_earlier",
            "test_used_for_selection": False,
            "epoch_plan_sha256": _plan_hash(plan),
            "epoch_plan_generator": plan["generator"],
        },
        "summary": summary,
        "curve": {
            "path": C.track_rel(curve_path),
            "sha256": C.sha256_file(curve_path),
            "n_epochs": len(curve),
            "improvements": list(tracker.improvements),
            "best_state_hash": best_state_hash,
            "last_state_hash": last_state_hash,
            "lr_first": float(reference_lr_trace[0]) if reference_lr_trace else None,
            "lr_last": float(reference_lr_trace[-1]) if reference_lr_trace else None,
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": 0,
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "mean_epoch_seconds": float(np.mean([row["train_seconds"] + row["eval_seconds"] for row in curve])),
            "mean_train_seconds": float(np.mean([row["train_seconds"] for row in curve])),
            "mean_eval_seconds": float(np.mean([row["eval_seconds"] for row in curve])),
            "gpu_peak_bytes": int(gpu_peak),
            "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0) if device.type == "cuda" else 0.0,
        },
        "test_labels_read": True,
        "test_access": "development-evaluation (declared in protocols/lcrc-lipo-context-small-s0-v1.yaml)",
    }
    C.write_json(out_dir / "metrics.json", payload)
    C.log(
        f"[{arm}] best_epoch={tracker.best_epoch}/{len(curve)} "
        f"valid={best_metrics['valid']['rmse']:.6f} test={best_metrics['test']['rmse']:.6f} "
        f"last_test={last_metrics['test']['rmse']:.6f} params={params_total} wall={wall:.1f}s"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LCRC lipophilicity context arms: one formal run")
    parser.add_argument("--arm", required=True, choices=list(C.ARMS))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="smoke only")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted run with the same config")
    args = parser.parse_args(argv)

    if args.smoke:
        if args.epochs is None:
            args.epochs = 2
        if args.epochs > 2:
            raise SystemExit("smoke runs may not exceed 2 epochs")
    elif args.epochs is not None and args.epochs != C.arm_epochs(args.arm):
        raise SystemExit(f"formal {args.arm} runs must use {C.arm_epochs(args.arm)} epochs")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    out_dir = Path(args.out_dir) if args.out_dir else C.RUNS / (f"smoke_{args.arm}" if args.smoke else C.run_name(args.arm))
    payload = train_run(
        arm=args.arm,
        device=device,
        out_dir=out_dir,
        epochs=args.epochs,
        smoke=bool(args.smoke),
        resume=bool(args.resume),
    )
    C.log(f"wrote {C.track_rel(out_dir)}")
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
