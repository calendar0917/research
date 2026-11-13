"""Part-B training unit for the two computational tasks (WC_LOGP / MOLWT).

One formal training per task, on the *original* Train3360 molecules only, with
the frozen ordinary-MSE recipe of the executed ``A_s0`` arm: same LCRC-C1
architecture / 228 674 parameters / seed 0, same rebuilt verified training-start
initialisation (``build_full_state`` -> ``remap_state``), same batch order, same
fixed feature normalisation.  Only the target mean/scale differ per task and are
fitted on that task's **Train original** labels.

The label loader explicitly selects the task and the ``row_id`` and asserts
``source_kind == "original"`` and ``kind == "O"``; rewired-view labels never
enter the optimiser, best-epoch selection or any hyper-parameter.

After the best checkpoint is chosen by Valid RMSE, the script runs exactly one
forward-only evaluation pass over the frozen original Valid/Test sets (for the
in-distribution metrics).  Test labels are only decoded after selection.
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

import round_common as RC

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# label loading (task + row_id explicit; original rows only)
# ---------------------------------------------------------------------------


def select_original_labels(
    rows: Sequence[Mapping[str, Any]], task: str, split: str, split_row_ids: Sequence[int]
) -> dict[int, float]:
    """Select one task's original (un-rewired) labels for one split.

    Pure function (unit-tested): the returned mapping must cover exactly the
    published split row ids, and every selected row must be
    ``source_kind == original`` / ``kind == O`` / ``task == task``.
    """
    if task not in RC.TASKS:
        raise ValueError(f"unknown task {task!r}")
    wanted = set(int(r) for r in split_row_ids)
    out: dict[int, float] = {}
    for r in rows:
        if str(r.get("task")) != task:
            continue
        if str(r.get("source_kind")) != "original" or str(r.get("kind")) != "O":
            continue
        if str(r.get("split")) != split:
            continue
        rid = int(r["row_id"])
        if rid not in wanted:
            continue
        if rid in out:
            raise ValueError(f"duplicate label for row {rid} in task {task}")
        out[rid] = float(r["value"])
    if set(out) != wanted:
        missing = sorted(wanted - set(out))
        extra = sorted(set(out) - wanted)
        raise ValueError(f"label coverage mismatch for {task}/{split}: missing={missing[:5]} extra={extra[:5]}")
    return out


def load_task_labels(task: str, split: str) -> dict[int, float]:
    path = RC.NEW_ANALYSIS / f"{task}_oracle.csv"
    rows = RC.read_csv(path)
    split_rows = [int(x) for x in RC.SC.load_source("none")["rows"][split]]
    return select_original_labels(rows, task, split, split_rows)


def raw_metrics(pred_std: np.ndarray, y_raw: np.ndarray, mean: float, scale: float) -> dict[str, float]:
    pred = np.asarray(pred_std, dtype=np.float64) * float(scale) + float(mean)
    y = np.asarray(y_raw, dtype=np.float64)
    err = pred - y
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(y.size), "mean_target": float(y.mean()), "std_target": float(y.std(ddof=0))}


def batch_order_sha(perms: np.ndarray) -> str:
    return RC.sha256_bytes(np.ascontiguousarray(perms, dtype=np.int64).tobytes())


def build_batch_perms(n_rows: int, epochs: int, seed: int) -> np.ndarray:
    gen = torch.Generator().manual_seed(int(seed))
    out = np.empty((int(epochs), int(n_rows)), dtype=np.int64)
    for ep in range(int(epochs)):
        out[ep] = torch.randperm(int(n_rows), generator=gen).numpy().astype(np.int64)
    return out


class BestEpochTracker:
    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_primary = math.inf
        self.best_metrics: dict[str, float] | None = None
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        primary = float(metrics["rmse"])
        if self.best_epoch is not None and not primary < self.best_primary:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = primary
        self.best_metrics = dict(metrics)
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("tracker has no captured state")
        return self._state


def _predict_std(model, cache, device, batch_size: int = RC.EVAL_BATCH_SIZE) -> np.ndarray:
    rows = list(range(int(cache["n_molecules"])))
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            idx = rows[start : start + batch_size]
            batch = RC.SC.DD.PREP_DATA.collate(cache, idx)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_task(*, task: str, device: torch.device, out_dir: Path) -> dict[str, Any]:
    if task not in RC.TASKS:
        raise ValueError(task)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    RC.log(f"[{task}] start: device={device}")

    spec = RC.frozen_spec()
    norm = RC.frozen_norm()
    ck = RC.SC.load_checkpoint()
    src = RC.SC.load_source("none")
    train_rows = [int(x) for x in src["rows"]["train"]]
    valid_rows = [int(x) for x in src["rows"]["valid"]]
    test_rows = [int(x) for x in src["rows"]["test"]]

    train_cache = RC.load_original_cache("train")
    valid_cache = RC.load_original_cache("valid")
    test_cache = RC.load_original_cache("test")

    train_lab = load_task_labels(task, "train")
    valid_lab = load_task_labels(task, "valid")
    test_lab = load_task_labels(task, "test")
    y_train = np.asarray([train_lab[r] for r in train_rows], dtype=np.float64)
    y_valid = np.asarray([valid_lab[r] for r in valid_rows], dtype=np.float64)
    y_test = np.asarray([test_lab[r] for r in test_rows], dtype=np.float64)

    target_mean = float(y_train.mean())
    target_scale = float(y_train.std(ddof=0))
    if not np.isfinite(target_mean) or not np.isfinite(target_scale) or target_scale <= 0:
        raise RuntimeError("degenerate target normaliser")
    z_train = (y_train - target_mean) / target_scale
    z_train_t = torch.as_tensor(z_train, dtype=torch.float32)

    # ---- model + verified initial state --------------------------------
    built = RC.SC.build_initial_state(spec)
    if not built["matches_recorded"]:
        raise SystemExit(f"[{task}] MISSING_BASELINE: initial state hash does not match A_s0 records")
    init_hash = built["hash"]
    full_init_hash = built["full_hash"]
    model = RC.SC.DM.DissectLCRC(spec, norm, RC.SC.VARIANT)
    model.load_state_dict({k: v.to("cpu") for k, v in built["state"].items()}, strict=True)
    model.to(device)
    params = RC.SC.DM.parameter_count(model)
    if params != RC.SC.EXPECTED_PARAMS:
        raise RuntimeError(f"parameter count {params} != {RC.SC.EXPECTED_PARAMS}")
    if RC.SC.DM.state_fingerprint({k: v.detach().cpu() for k, v in model.state_dict().items()}) != RC.SC.INIT_STATE_HASH:
        raise RuntimeError("model initial state hash differs from the frozen INIT hash")

    # initial (not trained) in-distribution diagnostics
    init_val = raw_metrics(_predict_std(model, valid_cache, device), y_valid, target_mean, target_scale)

    n_train = len(train_rows)
    bs = int(RC.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(n_train / bs)
    epochs = int(RC.RECIPE["epochs"])
    total_steps = steps_per_epoch * epochs
    if steps_per_epoch != int(RC.RECIPE["steps_per_epoch"]) or total_steps != int(RC.RECIPE["total_optimizer_steps"]):
        raise RuntimeError("frozen steps-per-epoch / total steps mismatch")
    perms = build_batch_perms(n_train, epochs, RC.SHUFFLE_SEED)
    order_sha = batch_order_sha(perms)

    torch.manual_seed(RC.TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RC.TRAIN_SEED)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(RC.RECIPE["peak_lr"]),
        betas=tuple(float(x) for x in RC.RECIPE["adam_betas"]),
        eps=float(RC.RECIPE["adam_eps"]), weight_decay=float(RC.RECIPE["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(RC.RECIPE["min_lr"]))

    tracker = BestEpochTracker()
    curve: list[dict[str, Any]] = []
    first_step_checks: dict[str, Any] = {}
    grad_clip_events = 0
    grad_norm_max = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = perms[epoch - 1]
        batches = [perm[i : i + bs] for i in range(0, len(perm), bs)]
        lr_start = float(optimizer.param_groups[0]["lr"])
        sq = 0.0
        ab = 0.0
        seen = 0
        for rows in batches:
            global_step += 1
            idx = [int(x) for x in rows]
            batch = RC.SC.DD.PREP_DATA.collate(train_cache, idx)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            z = z_train_t[list(idx)].to(device)
            pred = model(batch).reshape(-1)
            loss = torch.mean((pred - z) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(RC.RECIPE["gradient_clip"])))
            if global_step == 1:
                grad_l2 = math.sqrt(sum(float((p.grad.detach().double() ** 2).sum()) for p in trainable if p.grad is not None))
                first_step_checks = {
                    "step": 1,
                    "loss_standardised": float(loss.detach()),
                    "loss_finite": bool(torch.isfinite(loss.detach())),
                    "grad_l2_preclip": float(grad_l2),
                    "grad_l2_after_clip": float(gnorm),
                    "grad_finite": bool(np.isfinite(grad_l2) and np.isfinite(gnorm)),
                    "params_finite": bool(all(torch.isfinite(p).all() for p in trainable)),
                }
                if not (first_step_checks["loss_finite"] and first_step_checks["grad_finite"] and first_step_checks["params_finite"]):
                    raise SystemExit(f"[{task}] NONFINITE_FIRST_BATCH: {first_step_checks}")
            if not math.isfinite(float(loss.detach())) or not math.isfinite(gnorm):
                raise SystemExit(f"[{task}] non-finite loss/gradient at step {global_step}")
            if gnorm > float(RC.RECIPE["gradient_clip"]):
                grad_clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            scheduler.step()
            err = pred.detach() - z
            sq += float((err * err).sum())
            ab += float(err.abs().sum())
            seen += int(z.numel())
        valid_metrics = raw_metrics(_predict_std(model, valid_cache, device), y_valid, target_mean, target_scale)
        improved = tracker.consider(epoch, valid_metrics)
        if improved:
            tracker.capture(model)
        curve.append(
            {
                "epoch": epoch,
                "steps": steps_per_epoch,
                "lr_start": lr_start,
                "lr_end": float(optimizer.param_groups[0]["lr"]),
                "train_mse_standardised": float(sq / max(seen, 1)),
                "train_rmse_standardised": float(math.sqrt(sq / max(seen, 1))),
                "valid_rmse_raw": float(valid_metrics["rmse"]),
                "valid_mae_raw": float(valid_metrics["mae"]),
                "is_best": bool(improved),
            }
        )
        if epoch == 1 or epoch % 60 == 0 or epoch == epochs:
            RC.log(f"[{task}] epoch {epoch}/{epochs} train_rmse_std={curve[-1]['train_rmse_standardised']:.4f} "
                   f"valid_rmse_raw={valid_metrics['rmse']:.5f} best={tracker.best_epoch}")
    wall = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise RuntimeError("no improvement during training")
    best_state = tracker.state
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_hash = RC.SC.DM.state_fingerprint(best_state)

    # ---- reload best, re-verify Valid, then decode Test -----------------
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    best_valid = raw_metrics(_predict_std(model, valid_cache, device), y_valid, target_mean, target_scale)
    if abs(best_valid["rmse"] - float(tracker.best_primary)) > RC.REPRO_TOL:
        raise RuntimeError("reloaded best Valid RMSE differs from tracker")
    train_metrics = raw_metrics(_predict_std(model, train_cache, device), y_train, target_mean, target_scale)
    test_metrics = raw_metrics(_predict_std(model, test_cache, device), y_test, target_mean, target_scale)
    # Train mean constant baseline: predicting the Train mean on Test gives RMSE = Test target std.
    test_mean_pred_rmse = float(np.sqrt(np.mean((y_test - target_mean) ** 2)))
    train_mean_baseline = {"test_rmse_predicting_train_mean": test_mean_pred_rmse}

    # ---- serialise ------------------------------------------------------
    with open(out_dir / "curve.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    common_ck = {
        "model": "LCRC-C1", "variant": RC.SC.VARIANT, "variant_name": RC.SC.VARIANT_NAME,
        "feature_spec_hash": spec.hash(), "dataset": "Lipophilicity", "split_public": "random122",
        "seed": int(RC.TRAIN_SEED), "protocol_id": RC.PROTOCOL_ID, "task": task,
        "parameters": params, "target_norm": {"mean": target_mean, "scale": target_scale, "n": int(y_train.size),
                                              "source": "Train original labels only", "valid_or_test_used": False},
        "train_rows_sha256": RC.SC.row_sha256(train_rows),
        "epochs_run": epochs, "init_state_hash": init_hash, "full_init_state_hash": full_init_hash,
    }
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    torch.save({**common_ck, "state_dict": best_state, "best_epoch": int(tracker.best_epoch),
                "best_metric_name": "valid_rmse_raw", "best_metric_value": float(tracker.best_primary),
                "state_hash": best_hash}, best_path)
    torch.save({**common_ck, "state_dict": last_state, "best_epoch": int(tracker.best_epoch),
                "state_hash": RC.SC.DM.state_fingerprint(last_state)}, last_path)

    pred_train = _predict_std(model, train_cache, device)
    pred_valid_all = _predict_std(model, valid_cache, device)
    pred_test = _predict_std(model, test_cache, device)
    np.savez(
        out_dir / "predictions_best.npz",
        train_rows=np.asarray(train_rows, dtype=np.int64), train_y=y_train, train_pred_standardised=pred_train,
        valid_rows=np.asarray(valid_rows, dtype=np.int64), valid_y=y_valid, valid_pred_standardised=pred_valid_all,
        test_rows=np.asarray(test_rows, dtype=np.int64), test_y=y_test, test_pred_standardised=pred_test,
        target_mean=np.float64(target_mean), target_scale=np.float64(target_scale),
    )

    payload = {
        "schema": "lcrc-lipo-oracle-task-train-v1",
        "round_id": RC.ROUND_ID,
        "protocol_id": RC.PROTOCOL_ID,
        "status": "completed",
        "task": task,
        "task_description": RC.TASK_DESCRIPTION[task],
        "units": RC.TASK_UNITS[task],
        "seed": int(RC.TRAIN_SEED),
        "model": {"variant": RC.SC.VARIANT, "parameters": params, "parameters_expected": RC.SC.EXPECTED_PARAMS,
                  "init_state_hash": init_hash, "full_init_state_hash": full_init_hash,
                  "init_state_label": "INIT_TRAINING_START"},
        "label_source": {
            "file": RC.track_rel(RC.NEW_ANALYSIS / f"{task}_oracle.csv"),
            "sha256": RC.sha256_file(RC.NEW_ANALYSIS / f"{task}_oracle.csv"),
            "selection": "source_kind==original, kind==O, task==task, split==split, row_id in published split",
            "rewire_labels_used_in_training": False,
            "other_task_labels_used": False,
        },
        "data": {
            "n_train": n_train, "n_valid": len(valid_rows), "n_test": len(test_rows),
            "train_rows_sha256": RC.SC.row_sha256(train_rows),
            "valid_rows_sha256": RC.SC.row_sha256(valid_rows),
            "test_rows_sha256": RC.SC.row_sha256(test_rows),
            "train_cache_fingerprint": train_cache["meta"].get("content_fingerprint"),
            "valid_cache_fingerprint": valid_cache["meta"].get("content_fingerprint"),
            "test_cache_fingerprint": test_cache["meta"].get("content_fingerprint"),
            "feature_norm": "frozen checkpoint Train-only buffers (never refitted)",
            "train_norm_fingerprint": RC.SC.DD.PREP_DATA.norm_fingerprint(norm),
            "target_norm": {"mean": target_mean, "scale": target_scale, "n": int(y_train.size),
                            "source": "Train original labels only", "valid_or_test_used": False},
            "test_targets_loaded_after_best_checkpoint": True,
            "test_used_for_selection": False,
        },
        "recipe": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in RC.RECIPE.items()},
                   "batch_size": bs, "epochs_run": epochs, "steps_per_epoch": steps_per_epoch,
                   "optimizer_steps": int(epochs * steps_per_epoch), "scheduler": "per_step_cosine",
                   "checkpoint_rule": "lowest_valid_rmse__strict_improvement__tie_keeps_earlier"},
        "batch_order": {"sha256": order_sha, "shuffle_seed": int(RC.SHUFFLE_SEED), "shape": list(perms.shape),
                        "epoch1_first10": [int(x) for x in perms[0, :10]]},
        "metrics": {"initial": {"valid": init_val},
                    "best": {"train": train_metrics, "valid": best_valid, "test": test_metrics},
                    "train_mean_constant_baseline": train_mean_baseline},
        "checkpoint": {"best_epoch": int(tracker.best_epoch), "best_state_hash": best_hash,
                       "best_sha256": RC.sha256_file(best_path), "best_path": RC.track_rel(best_path),
                       "last_sha256": RC.sha256_file(last_path)},
        "anomalies": {"grad_clip_events": int(grad_clip_events),
                      "grad_clip_ratio": float(grad_clip_events / max(epochs * steps_per_epoch, 1)),
                      "grad_norm_max": float(grad_norm_max), "nonfinite_events": 0,
                      "first_step": first_step_checks},
        "timing": {"train_wall_seconds": float(wall), "mean_epoch_seconds": float(wall / epochs)},
        "gpu": {"device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "device_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
                "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0) if device.type == "cuda" else 0.0},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda, "numpy": np.__version__, "hostname": platform.node(),
                        "device": str(device), "cuda_available": bool(torch.cuda.is_available())},
        "provenance": {"git_commit": RC.git_commit(), "git_dirty_tracked": RC.git_dirty(),
                       "code_hashes": RC.code_hashes()},
    }
    RC.write_json(out_dir / "train.json", payload)
    RC.log(f"[{task}] best_epoch={tracker.best_epoch} valid_rmse={best_valid['rmse']:.5f} "
           f"test_rmse={test_metrics['rmse']:.5f} wall={wall:.1f}s")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="oracle task training unit")
    ap.add_argument("--task", choices=list(RC.TASKS), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    out_dir = Path(args.out_dir) if args.out_dir else RC.NEW_RUNS / args.task
    train_task(task=args.task, device=device, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
