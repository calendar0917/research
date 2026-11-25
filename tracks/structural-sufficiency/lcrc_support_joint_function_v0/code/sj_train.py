"""One formal training unit: (task, arm, seed) with full three-split evaluation.

The loop is the frozen reused one, unchanged in every detail that can affect a
comparison:

* ``hiv`` reproduces ``ptb_train_mol.train_unit`` — RNG seeded **before** the
  model is constructed, ``torch.Generator(seed + shuffle_offset)`` permutation,
  batch 128, AdamW(3e-4, wd 1e-3), per-step cosine to 1e-5 over all updates,
  clip 5, ``BCEWithLogitsLoss`` on the raw logits, best strict Valid ROC-AUC.
* ``lipo`` reproduces ``dissect_train.train_run`` — RNG seeded **after** the
  model is constructed and loaded, same generator offset, MSE on the
  train-standardised target, best strict Valid RMSE.

Only two things are added: every epoch also evaluates the Train and Test splits
in full (Train and Test are recorded, never used for selection), and the joint
channel is the arm's parameterisation.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import sj_common as C
import sj_model as M
import sj_support as S


# ---------------------------------------------------------------------------
# validation-only checkpoint selection
# ---------------------------------------------------------------------------


class Tracker:
    """Best-Valid tracker with the frozen tie rules (Test is never consulted)."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.best_epoch: int | None = None
        self.best_primary: float | None = None
        self.best_secondary: float | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.updates: list[int] = []
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, Any]) -> bool:
        primary = float(metrics[self.primary_key])
        secondary = float(metrics[self.secondary_key])
        if self.best_epoch is None:
            better = True
        elif primary == self.best_primary:
            better = (secondary < self.best_secondary) if self.task == "binary" else False
        else:
            better = primary > self.best_primary if self.task == "binary" else primary < self.best_primary
        if not better:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = float(primary)
        self.best_secondary = float(secondary)
        self.best_metrics = dict(metrics)
        self.updates.append(int(epoch))
        return True

    @property
    def primary_key(self) -> str:
        return "roc_auc" if self.task == "binary" else "rmse"

    @property
    def secondary_key(self) -> str:
        return "pr_auc" if self.task == "binary" else "mae"

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("tracker captured no state")
        return self._state

    def rule(self) -> str:
        if self.task == "binary":
            return "highest_validation_roc_auc__tie_lower_validation_bce__then_earlier_epoch"
        return "lowest_validation_rmse__strict_improvement__tie_keeps_earlier_epoch"


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_raw(model: torch.nn.Module, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = to_device(C.PREP_DATA.collate(cache, rows), device)
            pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(len(rows), -1))
    mat = np.concatenate(out, axis=0).astype(np.float64)
    if mat.shape[0] != n:
        raise RuntimeError(f"prediction rows {mat.shape[0]} != {n}")
    return mat


def split_metrics(task: str, raw: np.ndarray, y: np.ndarray, target_norm, names) -> dict[str, Any]:
    return C.MET.task_metrics(task, raw, y, target_norm, names)


def metric_keys(task: str) -> tuple[str, str]:
    return ("roc_auc", "pr_auc") if task == "binary" else ("rmse", "mae")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def task_targets(task_key: str, view: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Targets of the three official splits plus the Test structure cache."""
    t = C.TASKS[task_key]
    if t["family"] == "ptb":
        test_cache, y_test, rows = C.PD.molecule_test(str(t["dataset"]))
        y = {
            "train": np.asarray(view.targets.y_train, dtype=np.float64).reshape(-1),
            "valid": np.asarray(view.targets.y_valid, dtype=np.float64).reshape(-1),
            "test": np.asarray(y_test, dtype=np.float64).reshape(-1),
            "test_rows": np.asarray(rows, dtype=np.int64),
        }
        return y, test_cache
    if t["family"] == "dissect":
        test_cache, y_test, _smiles, rows = C.DD.load_test(int(t["repeat"]))
        y = {
            "train": np.asarray(view.y_train, dtype=np.float64).reshape(-1),
            "valid": np.asarray(view.y_valid, dtype=np.float64).reshape(-1),
            "test": np.asarray(y_test, dtype=np.float64).reshape(-1),
            "test_rows": np.asarray(rows, dtype=np.int64),
        }
        return y, test_cache
    raise KeyError(t["family"])


def loss_targets(task_key: str, view: Any, y_raw: np.ndarray) -> torch.Tensor:
    t = C.TASKS[task_key]
    if t["task"] == "regression":
        tn = view.target_norm
        z = (y_raw - float(tn["mean"])) / float(tn["scale"])
        return torch.as_tensor(z, dtype=torch.float32)
    return torch.as_tensor(y_raw, dtype=torch.float32)


def task_names(view: Any) -> list[str]:
    for attr in ("task_names",):
        if hasattr(view, attr):
            return list(getattr(view, attr))
    if hasattr(view, "task") and getattr(view, "task") == "binary":
        return ["hiv"]
    return ["logD"]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_unit(
    task_key: str,
    arm: str,
    seed: int,
    out_dir: Path,
    device,
    *,
    epochs: int | None = None,
    smoke: bool = False,
    force: bool = False,
    replicate_of: str | None = None,
) -> dict[str, Any]:
    if task_key not in C.TASKS:
        raise KeyError(task_key)
    t = C.TASKS[task_key]
    formal_epochs = int(t["epochs"])
    if arm not in C.ARMS:
        raise KeyError(arm)
    if smoke:
        epochs = int(epochs if epochs is not None else 2)
        if epochs > 3:
            raise SystemExit("smoke runs may use at most 3 epochs")
    else:
        if arm == C.ARM_A and replicate_of is None:
            raise SystemExit("arm A is the reused frozen baseline; use --replicate-of to run a replication probe")
        if epochs is None:
            epochs = formal_epochs
        if int(epochs) != formal_epochs:
            raise ValueError(f"{task_key} formal runs must use {formal_epochs} epochs, got {epochs}")
        if int(seed) != int(C.SEED):
            raise ValueError(f"formal runs use seed {C.SEED}")
    out_dir = Path(out_dir)
    record_path = out_dir / "run.json"
    if record_path.exists() and not force:
        raise RuntimeError(f"{C.track_rel(record_path)} exists; refusing to overwrite a recorded unit")
    out_dir.mkdir(parents=True, exist_ok=True)

    process_start = time.perf_counter()
    view, _tr, _va, _te, spec = S.task_caches(task_key)
    support = S.build_task_support(task_key)
    y, test_cache = task_targets(task_key, view)
    del _te
    train_cache, valid_cache = view.train_cache, view.valid_cache
    names = task_names(view)
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])
    n_test = int(test_cache["n_molecules"])
    if n_train != t["split_sizes"]["train"] or n_valid != t["split_sizes"]["valid"] or n_test != t["split_sizes"]["test"]:
        raise RuntimeError("official split sizes changed")
    for split, cache, expected in (
        ("train", train_cache, view.rows["train"]),
        ("valid", valid_cache, view.rows["valid"]),
        ("test", test_cache, y["test_rows"]),
    ):
        got = [int(x) for x in np.asarray(cache["rows"]).tolist()]
        if got != [int(x) for x in np.asarray(expected).tolist()]:
            raise RuntimeError(f"{split} cache rows differ from the frozen published split")

    recipe = dict(C.RECIPE)
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model, model_info = M.make_arm(task_key, view, support, arm, int(seed), device)
    if C.SEED_TIMING[task_key] == "after_build":
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
    model.to(device)
    params_total = int(sum(p.numel() for p in model.parameters()))
    if not smoke and arm == C.ARM_A:
        expected = int(C.REUSE[task_key]["expected"]["parameters"])
        if params_total != expected:
            raise RuntimeError(f"replication arm A has {params_total} parameters, expected {expected}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(recipe["min_lr"])
    )
    generator = torch.Generator().manual_seed(int(seed) + int(recipe["shuffle_offset"]))

    y_loss = loss_targets(task_key, view, y["train"]).to(device)
    target_norm = view.target_norm
    task = str(t["task"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    tracker = Tracker(task)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    peak_alloc = 0
    train_seconds_total = 0.0
    eval_seconds_total = 0.0
    gpu_seconds_total = 0.0

    t_train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        perm = torch.randperm(n_train, generator=generator).tolist()
        for rows in (perm[i : i + batch_size] for i in range(0, len(perm), batch_size)):
            batch = to_device(C.PREP_DATA.collate(train_cache, rows), device)
            target = y_loss[rows].to(device)
            pred = model(batch).reshape(-1)
            if task == "regression":
                loss = F.mse_loss(pred, target)
            else:
                loss = F.binary_cross_entropy_with_logits(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/grad at epoch {epoch}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            scheduler.step()
        train_seconds = time.perf_counter() - t_ep
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gpu_seconds_total += time.perf_counter() - t_ep
        train_seconds_total += train_seconds

        t_eval = time.perf_counter()
        split_metrics_epoch: dict[str, dict[str, Any]] = {}
        preds: dict[str, np.ndarray] = {}
        for split, cache, key in (("train", train_cache, "train"), ("valid", valid_cache, "valid"), ("test", test_cache, "test")):
            raw = predict_raw(model, cache, device)
            preds[split] = raw
            split_metrics_epoch[split] = split_metrics(task, raw, y[split], target_norm, names)
        eval_seconds = time.perf_counter() - t_eval
        eval_seconds_total += eval_seconds

        improved = tracker.consider(epoch, split_metrics_epoch["valid"])
        if improved:
            tracker.capture(model)
        pk, sk = metric_keys(task)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "train_seconds": float(train_seconds),
            "eval_seconds": float(eval_seconds),
            "is_best": bool(improved),
        }
        for split in ("train", "valid", "test"):
            row[f"{split}_{pk}"] = float(split_metrics_epoch[split][pk])
            row[f"{split}_{sk}"] = float(split_metrics_epoch[split][sk])
        row["valid_loss_reference"] = float(
            split_metrics_epoch["valid"]["bce"] if task == "binary" else split_metrics_epoch["valid"]["rmse"] ** 2
        )
        curve.append(row)
        if device.type == "cuda":
            peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated(device)))
        if smoke or epoch == epochs or improved:
            C.log(
                f"[{task_key}/{arm}/s{seed}] epoch {epoch}/{epochs} "
                f"valid {pk}={row[f'valid_{pk}']:.6f} test {pk}={row[f'test_{pk}']:.6f}"
                f"{' *best*' if improved else ''}"
            )
    train_wall = time.perf_counter() - t_train_start
    if tracker.best_epoch is None:
        raise RuntimeError("no validation improvement recorded")

    best_state = tracker.state
    best_state_hash = C.DM.state_fingerprint(best_state)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    pk, sk = metric_keys(task)
    final_metrics: dict[str, dict[str, Any]] = {}
    final_preds: dict[str, np.ndarray] = {}
    for split, cache in (("train", train_cache), ("valid", valid_cache), ("test", test_cache)):
        raw = predict_raw(model, cache, device)
        final_preds[split] = raw
        final_metrics[split] = split_metrics(task, raw, y[split], target_norm, names)
    recheck = abs(float(final_metrics["valid"][pk]) - float(tracker.best_metrics[pk]))
    if recheck > C.TOL["metric_recheck"]:
        raise RuntimeError(f"reloaded best checkpoint Valid {pk} != recorded (diff {recheck})")

    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    for split in ("train", "valid", "test"):
        np.savez(
            out_dir / f"{split}_predictions.npz",
            rows=np.asarray((train_cache, valid_cache, test_cache)[("train", "valid", "test").index(split)]["rows"], dtype=np.int64),
            pred=final_preds[split],
            y_true=y[split],
        )
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "round_id": C.ROUND_ID,
            "task": task_key,
            "dataset": t["dataset"],
            "variant": t["variant"],
            "arm": arm,
            "seed": int(seed),
            "epochs_run": int(epochs),
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": f"valid_{pk}",
            "best_metric_value": float(tracker.best_primary),
            "selection_rule": tracker.rule(),
            "feature_spec_hash": spec.hash(),
            "norm_fingerprint": C.PREP_DATA.norm_fingerprint(view.train_norm),
            "target_norm": target_norm,
            "recipe": recipe,
            "support_task": support.task_key,
            "support_omega": int(support.n_omega),
            "support_rank": int(support.rank),
            "smoke": bool(smoke),
        },
        checkpoint_path,
    )
    process_wall = time.perf_counter() - process_start
    last20 = curve[-20:]
    tail = {
        f"last20_{split}_{pk}": float(np.mean([r[f"{split}_{pk}"] for r in last20]))
        for split in ("train", "valid", "test")
    }
    tail[f"last20_{sk}"] = float(np.mean([r[f"valid_{sk}"] for r in last20]))
    tail["last20_epochs"] = [int(r["epoch"]) for r in last20]
    record = {
        "schema": "lcrc-support-joint-function-run-v1",
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "env": C.env_info(),
        "task": task_key,
        "dataset": t["dataset"],
        "task_family": task,
        "variant": t["variant"],
        "arm": arm,
        "arm_label": C.ARM_LABEL[arm],
        "seed": int(seed),
        "smoke": bool(smoke),
        "replicate_of": replicate_of,
        "epochs_run": int(epochs),
        "split_sizes": {"train": n_train, "valid": n_valid, "test": n_test},
        "split_rows_sha256": {
            split: C.index_sha256([int(x) for x in np.asarray(cache["rows"]).tolist()])
            for split, cache in (("train", train_cache), ("valid", valid_cache), ("test", test_cache))
        },
        "feature_spec_hash": spec.hash(),
        "norm_fingerprint": C.PREP_DATA.norm_fingerprint(view.train_norm),
        "target_norm": target_norm,
        "model": model_info,
        "parameters": params_total,
        "parameters_breakdown": C.DM.module_breakdown(model),
        "recipe": recipe,
        "support": {
            "omega": int(support.n_omega),
            "rank": int(support.rank),
            "joint_dim": int(support.joint_dim),
            "coverage": support.coverage,
            "artifact_sha256": C.sha256_file(C.ANALYSIS / "support" / f"{task_key}_support.npz"),
        },
        "checkpoint": {
            "path": C.track_rel(checkpoint_path),
            "sha256": C.sha256_file(checkpoint_path),
            "best_epoch": int(tracker.best_epoch),
            "best_state_hash": best_state_hash,
            "improvement_epochs": list(tracker.updates),
            "valid_recheck_abs_diff": float(recheck),
            "selection_uses_test": False,
        },
        "curve": {"path": C.track_rel(curve_path), "sha256": C.sha256_file(curve_path)},
        "metrics": {
            "best_checkpoint": final_metrics,
            "best_checkpoint_label": "best-Valid checkpoint, all three splits evaluated once after training",
        },
        "last20": tail,
        "timing": {
            "process_wall_seconds": float(process_wall),
            "train_wall_seconds": float(train_wall),
            "train_seconds_total": float(train_seconds_total),
            "eval_seconds_total": float(eval_seconds_total),
            "mean_epoch_seconds": float(train_wall / epochs),
            "gpu_seconds": float(gpu_seconds_total),
            "gpu_hours": float(gpu_seconds_total / 3600.0),
            "gpu_peak_allocated_bytes": int(peak_alloc),
            "device": str(device),
            "steps_per_epoch": int(steps_per_epoch),
        },
        "result_label": "DEVELOPMENT_TEST",
    }
    C.write_json(record_path, record)
    C.log(
        f"[{task_key}/{arm}/s{seed}] done best_epoch={tracker.best_epoch} valid_{pk}={tracker.best_primary:.6f} "
        f"test_{pk}={final_metrics['test'][pk]:.6f} wall={process_wall/60:.1f}min gpu_h={gpu_seconds_total/3600:.4f}"
    )
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One joint-function training unit")
    ap.add_argument("--task", required=True, choices=list(C.TASKS))
    ap.add_argument("--arm", required=True, choices=list(C.ARMS))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--replicate-of", default=None, help="tag for an arm-A replication probe")
    args = ap.parse_args(argv)
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.smoke:
        out_dir = C.RUNS / "smoke" / f"{args.task}_{args.arm}_s{args.seed}"
    else:
        out_dir = C.run_dir(args.task, args.arm, int(args.seed))
    record = train_unit(
        args.task,
        args.arm,
        int(args.seed),
        Path(out_dir),
        device,
        epochs=args.epochs,
        smoke=bool(args.smoke),
        force=bool(args.force),
        replicate_of=args.replicate_of,
    )
    print(C.stable_json({k: record[k] for k in ("task", "arm", "seed", "parameters", "last20", "timing")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
