"""One fixed training unit: LOCAL3 / DUP2 / NULL3 at seed 0.

Nothing but the correction operator changes with respect to the ordinary C1
baseline: same frozen FULL seed-0 initial state, same ordinary batch order
(shuffle generator 917000), same R3 recipe, one ordinary molecule MSE, one
frozen dropout call per forward, the same checkpoint rule (Valid420 raw RMSE,
strict improvement, tie keeps the earlier epoch).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C
import data as D
import features as F
import metrics as M
import model as MD
import schedule as S

PROCESS_T0 = time.perf_counter()

CORRECTOR_PARAMS = ("W_h.weight", "W_s.weight", "b1", "W2.weight", "b2")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rng_hash(device) -> str:
    h = C.sha256_bytes(torch.get_rng_state().numpy().tobytes())
    if device.type == "cuda":
        h += ":" + C.sha256_bytes(torch.cuda.get_rng_state(device).numpy().tobytes())
    return h


def _reset_rng(seed: int, device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _grad_norm(grads: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.detach().double() ** 2).sum().item())
    return math.sqrt(total)


class BestEpochTracker:
    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_primary = math.inf
        self.best_metrics: dict[str, float] | None = None
        self.improvement_epochs: list[int] = []
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        primary = float(metrics["rmse"])
        if self.best_epoch is not None and not primary < self.best_primary:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = primary
        self.best_metrics = dict(metrics)
        self.improvement_epochs.append(int(epoch))
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("tracker has no captured state")
        return self._state


def _param_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in CORRECTOR_PARAMS:
        p = dict(model.s3.named_parameters())[name]
        out[f"grad_{name}"] = float(p.grad.detach().double().norm().item()) if p.grad is not None else 0.0
    out["grad_corrector_total"] = _grad_norm([p.grad for p in model.s3.parameters()])
    return out


def _degenerate_fraction(model, batch: Mapping[str, Any]) -> float:
    """Fraction of roots whose same-molecule/same-type group carries one distinct s3."""
    key = MD._group_key(batch)
    s = batch["s3"]
    n = int(key.numel())
    if n == 0:
        return 1.0
    degenerate = 0
    uniq, inv = torch.unique(key, return_inverse=True)
    for g in range(int(uniq.numel())):
        idx = torch.nonzero(inv == g, as_tuple=False).view(-1)
        if int(idx.numel()) == 1:
            degenerate += 1
        else:
            sub = s[idx]
            if bool(torch.all(sub == sub[0])):
                degenerate += int(idx.numel())
    return float(degenerate / n)


# ---------------------------------------------------------------------------
# training one arm
# ---------------------------------------------------------------------------


def train_arm(
    *,
    arm: str,
    device: torch.device,
    out_dir: Path,
    epochs: int,
    smoke: bool,
) -> dict[str, Any]:
    if arm not in C.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if not smoke and int(epochs) != int(C.RECIPE["epochs"]):
        raise SystemExit(f"formal runs must use {C.RECIPE['epochs']} epochs")
    if smoke and int(epochs) > 2:
        raise SystemExit("smoke runs may not exceed 2 epochs")
    if not smoke and C.git_dirty():
        raise SystemExit("formal run requested on a dirty tracked worktree; commit first")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = D.load_dataset(include_test=False)
    spec = dataset["spec"]
    train_cache = dataset["train_cache"]
    valid_cache = dataset["valid_cache"]
    train_rows = [int(r) for r in dataset["rows"]["train"]]
    valid_rows = [int(r) for r in dataset["rows"]["valid"]]
    y_train = np.asarray(dataset["y_train"], dtype=np.float64)
    y_valid = np.asarray(dataset["y_valid"], dtype=np.float64)
    train_norm = dataset["train_norm"]
    target_norm = dataset["target_norm"]
    features = F.load_features()
    f_train = F.split_features(features, "train", train_cache)
    f_valid = F.split_features(features, "valid", valid_cache)
    if int(train_cache["n_molecules"]) != C.SPLIT_SIZES[0] or int(valid_cache["n_molecules"]) != C.SPLIT_SIZES[1]:
        raise RuntimeError("split sizes changed")

    order = S.load_schedule()
    perms = order["batch_perms"]
    n_train = len(train_rows)
    batch_size = int(C.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * int(epochs)
    if steps_per_epoch != int(C.RECIPE["steps_per_epoch"]):
        raise RuntimeError(f"steps_per_epoch {steps_per_epoch} != frozen {C.RECIPE['steps_per_epoch']}")

    z_train = (y_train - float(target_norm["mean"])) / float(target_norm["scale"])
    z_train_cache = torch.as_tensor(z_train, dtype=torch.float32)

    model, info = MD.build_model(spec, train_norm, device, arm)
    n_bad = sum(1 for k in ("base_shared_state_equal",) if not info[k]["all_equal"])
    if n_bad:
        raise RuntimeError("remapped base state is not bitwise equal to the FULL state")
    if str(info["init_state_hash"])[:8] != C.BASELINE_EXPECTED["init_state_hash_prefix"]:
        raise RuntimeError(
            f"init state hash {info['init_state_hash']} does not start with the frozen "
            f"{C.BASELINE_EXPECTED['init_state_hash_prefix']}"
        )

    _reset_rng(C.TRAIN_SEED, device)
    init_rng_hash = _rng_hash(device)

    model.eval()
    init_train_std = D.predict_scores(model, train_cache, device, feats=f_train)
    init_valid_std = D.predict_scores(model, valid_cache, device, feats=f_valid)
    model.train()
    init_train_metrics = M.raw_metrics(init_train_std, y_train, target_norm)
    init_valid_metrics = M.raw_metrics(init_valid_std, y_valid, target_norm)

    dropout_calls = {"n": 0}
    model.drop.register_forward_hook(lambda *_: dropout_calls.__setitem__("n", dropout_calls["n"] + 1))

    torch.manual_seed(C.TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(C.TRAIN_SEED)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.RECIPE["peak_lr"]),
        betas=tuple(float(x) for x in C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=float(C.RECIPE["min_lr"])
    )

    tracker = BestEpochTracker()
    curve: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    epoch_rng_hashes: list[dict[str, Any]] = []
    grad_norm_max = 0.0
    clip_events = 0
    nonfinite_events = 0
    correction_stats: list[dict[str, Any]] = []
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        perm = perms[epoch - 1]
        if len(perm) != n_train:
            raise RuntimeError("batch permutation length differs from Train size")
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        lr_start = float(optimizer.param_groups[0]["lr"])
        abs_sum = 0.0
        sq_sum = 0.0
        seen = 0
        epoch_clips = 0
        epoch_gnorm_max = 0.0
        for step_index, rows in enumerate(batches):
            global_step = (epoch - 1) * steps_per_epoch + step_index + 1
            batch = D.to_device(D.collate(train_cache, [int(x) for x in rows], f_train), device)
            z = z_train_cache[list(rows)].to(device)
            pred = model(batch).reshape(-1)
            loss = ((pred - z) ** 2).mean()

            if global_step in C.DIAGNOSTIC_STEPS:
                grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
                diagnostics.append(
                    {
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "loss": float(loss.detach()),
                        "grad_total_l2": float(_grad_norm(grads)),
                        "would_clip_at": float(C.RECIPE["gradient_clip"]),
                        "correction": dict(getattr(model, "s3_last_stats", {})),
                        **_param_grad_norms(model),
                    }
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step in C.DIAGNOSTIC_STEPS:
                diagnostics[-1].update(_param_grad_norms(model))
                diagnostics[-1]["correction_state"] = dict(getattr(model, "s3_last_stats", {}))
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"])))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(gnorm):
                nonfinite_events += 1
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch} step {step_index + 1}")
            if gnorm > float(C.RECIPE["gradient_clip"]):
                clip_events += 1
                epoch_clips += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            epoch_gnorm_max = max(epoch_gnorm_max, gnorm)
            optimizer.step()
            scheduler.step()

            err = pred.detach() - z
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            seen += int(z.numel())
            if epoch == 1:
                epoch_rng_hashes.append({"step": int(global_step), "rng_hash": _rng_hash(device)})
            if global_step in (1, 28, 271):
                correction_stats.append({"step": int(global_step), **dict(getattr(model, "s3_last_stats", {}))})

        train_seconds = time.perf_counter() - t0
        t_eval = time.perf_counter()
        valid_std = D.predict_scores(model, valid_cache, device, feats=f_valid)
        valid_metrics = M.raw_metrics(valid_std, y_valid, target_norm)
        eval_seconds = time.perf_counter() - t_eval
        improved = tracker.consider(epoch, valid_metrics)
        if improved:
            tracker.capture(model)
        train_mse = float(sq_sum / max(seen, 1))
        curve.append(
            {
                "epoch": int(epoch),
                "steps": steps_per_epoch,
                "lr_start": lr_start,
                "lr_end": float(optimizer.param_groups[0]["lr"]),
                "train_mse_standardized": train_mse,
                "train_rmse_standardized": math.sqrt(train_mse),
                "train_mae_standardized": float(abs_sum / max(seen, 1)),
                "valid_rmse_raw": float(valid_metrics["rmse"]),
                "valid_mae_raw": float(valid_metrics["mae"]),
                "valid_r2_raw": float(valid_metrics["r2"]),
                "is_best": bool(improved),
                "grad_norm_max": float(epoch_gnorm_max),
                "grad_clip_events": int(epoch_clips),
                "clip_ratio": float(epoch_clips / max(steps_per_epoch, 1)),
                "train_seconds": float(train_seconds),
                "eval_seconds": float(eval_seconds),
            }
        )
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise RuntimeError("training ended without any improvement")
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_state = tracker.state
    best_state_hash = C.DM.state_fingerprint(best_state)
    last_state_hash = C.DM.state_fingerprint(last_state)
    curve_values = np.asarray([r["valid_rmse_raw"] for r in curve], dtype=np.float64)
    if int(np.argmin(curve_values) + 1) != int(tracker.best_epoch):
        raise RuntimeError("curve argbest != tracked best epoch")

    # ---- best checkpoint: reload, re-verify, predict all splits -------------
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    best_valid_std = D.predict_scores(model, valid_cache, device, feats=f_valid)
    best_valid_metrics = M.raw_metrics(best_valid_std, y_valid, target_norm)
    recheck = abs(float(best_valid_metrics["rmse"]) - float(tracker.best_primary))
    if recheck > C.REPRO_TOL:
        raise RuntimeError(f"reloaded best checkpoint Valid RMSE differs by {recheck}")

    train_std = D.predict_scores(model, train_cache, device, feats=f_train)
    train_metrics = M.raw_metrics(train_std, y_train, target_norm)

    # Test targets are decoded only now, after best checkpoint selection.
    test_cache = D.load_cache("test")
    f_test = F.split_features(features, "test", test_cache)
    src_all = C.source_with_labels("all")
    test_rows = [int(r) for r in src_all["rows"]["test"]]
    if [int(x) for x in np.asarray(test_cache["rows"]).tolist()] != test_rows:
        raise RuntimeError("Test cache rows differ from published split")
    y_test = np.asarray(src_all["y"]["test"], dtype=np.float64)
    test_std = D.predict_scores(model, test_cache, device, feats=f_test)
    test_metrics = M.raw_metrics(test_std, y_test, target_norm)

    final_degenerate = _degenerate_fraction(
        model, D.to_device(D.collate(train_cache, list(range(min(256, int(train_cache["n_molecules"])))), f_train), device)
    )

    # ---- last epoch: one endpoint evaluation only --------------------------
    model.load_state_dict({k: v.to(device) for k, v in last_state.items()})
    model.eval()
    last_train_std = D.predict_scores(model, train_cache, device, feats=f_train)
    last_valid_std = D.predict_scores(model, valid_cache, device, feats=f_valid)
    last_test_std = D.predict_scores(model, test_cache, device, feats=f_test)
    last_train_metrics = M.raw_metrics(last_train_std, y_train, target_norm)
    last_valid_metrics = M.raw_metrics(last_valid_std, y_valid, target_norm)
    last_test_metrics = M.raw_metrics(last_test_std, y_test, target_norm)

    # ---- serialise ---------------------------------------------------------
    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    with open(out_dir / "epoch_log.jsonl", "w", encoding="utf-8") as fh:
        for row in curve:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    C.write_json(out_dir / "diagnostics.json", {"steps": diagnostics, "correction_stats": correction_stats})

    common_ckpt = {
        "model": "LCRC-C1-S3-correction",
        "variant": C.VARIANT,
        "variant_name": C.VARIANT_NAME,
        "feature_spec_hash": spec.hash(),
        "dataset": "Lipophilicity",
        "split_public": "random122",
        "seed": int(C.TRAIN_SEED),
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "parameters": int(info["parameters"]),
        "target_norm": dict(target_norm),
        "train_norm_fingerprint": C.PREP_DATA.norm_fingerprint(train_norm),
        "train_rows_sha256": C.row_sha256(train_rows),
        "epochs_run": len(curve),
        "smoke": bool(smoke),
        "init_state_hash": info["init_state_hash"],
        "full_init_state_hash": info["full_init_state_hash"],
        "corrector_state_hash": info["corrector_state_hash"],
        "s3_norm": dict(features["norm_s3"]),
    }
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"
    torch.save(
        {
            **common_ckpt,
            "state_dict": best_state,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse_raw",
            "best_metric_value": float(tracker.best_primary),
            "state_hash": best_state_hash,
        },
        best_ckpt,
    )
    torch.save(
        {
            **common_ckpt,
            "state_dict": last_state,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse_raw",
            "best_metric_value": float(tracker.best_primary),
            "state_hash": last_state_hash,
        },
        last_ckpt,
    )

    def _npz(path, tag, train_pred, valid_pred, test_pred):
        np.savez(
            path,
            train_rows=np.asarray(train_rows, dtype=np.int64),
            train_y=y_train,
            train_pred_raw=train_pred * float(target_norm["scale"]) + float(target_norm["mean"]),
            train_pred_standardised=train_pred,
            valid_rows=np.asarray(valid_rows, dtype=np.int64),
            valid_y=y_valid,
            valid_pred_raw=valid_pred * float(target_norm["scale"]) + float(target_norm["mean"]),
            valid_pred_standardised=valid_pred,
            test_rows=np.asarray(test_rows, dtype=np.int64),
            test_y=y_test,
            test_pred_raw=test_pred * float(target_norm["scale"]) + float(target_norm["mean"]),
            test_pred_standardised=test_pred,
        )

    _npz(out_dir / "predictions_best.npz", "best", train_std, best_valid_std, test_std)
    _npz(out_dir / "predictions_last.npz", "last", last_train_std, last_valid_std, last_test_std)
    np.savez(
        out_dir / "predictions_initial.npz",
        train_rows=np.asarray(train_rows, dtype=np.int64),
        train_y=y_train,
        train_pred_raw=init_train_std * float(target_norm["scale"]) + float(target_norm["mean"]),
        train_pred_standardised=init_train_std,
        valid_rows=np.asarray(valid_rows, dtype=np.int64),
        valid_y=y_valid,
        valid_pred_raw=init_valid_std * float(target_norm["scale"]) + float(target_norm["mean"]),
        valid_pred_standardised=init_valid_std,
    )

    baseline_ref = C.ANALYSIS / "baseline_reference.json"
    baseline_rng = None
    baseline_device = None
    rng_matches = None
    if baseline_ref.exists():
        ref = C.read_json(baseline_ref)
        baseline_rng = ref.get("rng", {}).get("epoch1_step_hashes")
        baseline_device = ref.get("rng", {}).get("device", "cuda")
        if baseline_rng:
            mine = [x["rng_hash"] for x in epoch_rng_hashes]
            theirs = [x["rng_hash"] for x in baseline_rng]
            # dropout draws from the generator of the *device it runs on*; a CPU
            # trace can never equal a CUDA trace, so only same-device is a check
            if device.type == baseline_device:
                rng_matches = bool(mine == theirs)

    payload: dict[str, Any] = {
        "schema": "lcrc-lipo-s3-locality-train-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "status": "completed",
        "smoke": bool(smoke),
        "arm": arm,
        "arm_description": C.ARM_DESCRIPTIONS[arm],
        "seed": int(C.TRAIN_SEED),
        "model": {
            "variant": C.VARIANT,
            "variant_name": C.VARIANT_NAME,
            "parameters": int(info["parameters"]),
            "parameters_base": int(info["base_parameters"]),
            "parameters_corrector": int(info["corrector_parameters"]),
            "init_state_hash": info["init_state_hash"],
            "full_init_state_hash": info["full_init_state_hash"],
            "corrector_state_hash": info["corrector_state_hash"],
            "shared_init_bitwise_equal_to_full": bool(info["base_shared_state_equal"]["all_equal"]),
            "dropout_calls_in_run": int(dropout_calls["n"]),
            "correction_norm_final": dict(getattr(model, "s3_last_stats", {})),
            "null3_degenerate_fraction_sampled": float(final_degenerate),
        },
        "data": {
            "n_train": n_train,
            "n_valid": len(valid_rows),
            "n_test": len(test_rows),
            "train_rows_sha256": C.row_sha256(train_rows),
            "valid_rows_sha256": C.row_sha256(valid_rows),
            "test_rows_sha256": C.row_sha256(test_rows),
            "train_cache_fingerprint": train_cache["meta"]["upstream_content_fingerprint"],
            "valid_cache_fingerprint": valid_cache["meta"]["upstream_content_fingerprint"],
            "test_cache_fingerprint": test_cache["meta"]["upstream_content_fingerprint"],
            "target_norm": dict(target_norm),
            "train_norm_fingerprint": C.PREP_DATA.norm_fingerprint(train_norm),
            "test_targets_loaded_after_best_checkpoint": True,
            "test_used_for_selection": False,
        },
        "recipe": {
            "optimizer": C.RECIPE["optimizer"],
            "adam_betas": [float(x) for x in C.RECIPE["adam_betas"]],
            "adam_eps": float(C.RECIPE["adam_eps"]),
            "weight_decay": float(C.RECIPE["weight_decay"]),
            "peak_lr": float(C.RECIPE["peak_lr"]),
            "min_lr": float(C.RECIPE["min_lr"]),
            "batch_size": batch_size,
            "epochs_planned": int(epochs),
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "gradient_clip": float(C.RECIPE["gradient_clip"]),
            "dropout": float(C.RECIPE["dropout"]),
            "loss": "mean squared standardised error",
            "checkpoint_rule": "lowest_valid_rmse__strict_improvement__tie_keeps_earlier",
            "early_stopping": False,
            "amp": False,
            "shuffle_generator_seed": int(C.SHUFFLE_SEED),
        },
        "batch_order": {
            "sha256": order["manifest"]["sha256"],
            "baseline_manifest_match": order["manifest"]["baseline_manifest_match"],
        },
        "rng": {
            "device": device.type,
            "baseline_device": baseline_device,
            "initial_hash_after_reset": init_rng_hash,
            "epoch1_step_hashes": epoch_rng_hashes,
            "baseline_epoch1_step_hashes_available": baseline_rng is not None,
            "epoch1_dropout_rng_matches_baseline": rng_matches,
            "comparable_to_baseline": (device.type == baseline_device),
        },
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse_raw",
            "best_metric_value": float(tracker.best_primary),
            "n_improvements": len(tracker.improvement_epochs),
            "best_state_hash": best_state_hash,
            "last_state_hash": last_state_hash,
            "best_path": C.track_rel(best_ckpt),
            "best_sha256": C.sha256_file(best_ckpt),
            "last_path": C.track_rel(last_ckpt),
            "last_sha256": C.sha256_file(last_ckpt),
            "curve_path": C.track_rel(curve_path),
            "curve_sha256": C.sha256_file(curve_path),
            "curve_argbest_epoch": int(np.argmin(curve_values) + 1),
            "best_valid_recheck_abs_diff": float(recheck),
        },
        "metrics": {
            "initial": {"train": init_train_metrics, "valid": init_valid_metrics},
            "best": {"train": train_metrics, "valid": best_valid_metrics, "test": test_metrics},
            "last": {"train": last_train_metrics, "valid": last_valid_metrics, "test": last_test_metrics},
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_clip_ratio": float(clip_events / max(len(curve) * steps_per_epoch, 1)),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": int(nonfinite_events),
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "mean_epoch_seconds": float(np.mean([r["train_seconds"] + r["eval_seconds"] for r in curve])),
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "device_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
            "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0) if device.type == "cuda" else 0.0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "hostname": platform.node(),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "provenance": {
            "git_commit": C.git_commit(),
            "git_dirty_tracked": C.git_dirty(),
            "code_hashes": C.code_hashes(),
        },
        "predictions": {
            "best": C.track_rel(out_dir / "predictions_best.npz"),
            "last": C.track_rel(out_dir / "predictions_last.npz"),
            "initial": C.track_rel(out_dir / "predictions_initial.npz"),
        },
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{arm}] best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train_rmse={train_metrics['rmse']:.5f} valid_rmse={best_valid_metrics['rmse']:.5f} "
        f"test_rmse={test_metrics['rmse']:.5f} wall={wall:.1f}s rng_match={rng_matches}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S3-locality correction training unit")
    ap.add_argument("--arm", choices=list(C.ARMS))
    ap.add_argument("--smoke", action="store_true", help="run LOCAL3 and DUP2 for 2 epochs")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.smoke:
        epochs = 2 if args.epochs is None else int(args.epochs)
        if epochs > 2:
            raise SystemExit("smoke runs may not exceed 2 epochs")
        root = Path(args.out_dir) if args.out_dir else C.RUNS / "smoke"
        for arm in C.STAGE1_ARMS:
            train_arm(arm=arm, device=device, out_dir=root / arm, epochs=epochs, smoke=True)
        return 0
    if not args.arm:
        raise SystemExit("--arm is required unless --smoke is used")
    epochs = int(C.RECIPE["epochs"]) if args.epochs is None else int(args.epochs)
    out_dir = Path(args.out_dir) if args.out_dir else C.run_dir(args.arm)
    train_arm(arm=args.arm, device=device, out_dir=out_dir, epochs=epochs, smoke=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
