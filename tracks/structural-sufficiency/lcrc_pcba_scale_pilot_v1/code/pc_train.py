"""Formal PCBA seed-0 exploratory training (`SCALE_PILOT`).

One seed, one model, one recipe.  There is no Test access path in this file:
the label artifact has every official Test row set to NaN and the training code
only ever indexes ``split["train"]`` / ``split["valid"]``.

Resource discipline
-------------------
* a soft wall-clock limit stops the loop between epochs,
* a watchdog thread arms a hard stop that is honoured inside the step loop,
* the best Validation-AP checkpoint is written (atomically) *during* training,
  so a kill at any moment leaves the selected checkpoint and the curve on disk,
* the run is labelled ``RESOURCE_LIMITED`` when the limit, not the recipe,
  ended the loop.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import pc_common as C  # noqa: E402
import pc_data as DATA  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
import bp_data as D  # noqa: E402
from feature_spec import load_spec  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "steps_per_epoch",
    "lr",
    "train_loss",
    "valid_ap",
    "n_tasks_with_observed_labels",
    "n_tasks_scored",
    "train_seconds",
    "valid_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "strict_improve",
]


@dataclass
class Task:
    spec: Any
    cache: Mapping[str, Any]
    labels: np.ndarray
    split: dict[str, Any]
    train_rows: np.ndarray
    valid_rows: np.ndarray
    test_rows: np.ndarray
    norm: dict[str, Any]
    load_seconds: float
    cache_path: Path
    labels_path: Path
    split_partition_sha256: str


def load_task(cache_path: Path, labels_path: Path, spec_path: Path, split_path: Path) -> Task:
    t0 = time.perf_counter()
    spec = load_spec(spec_path)
    split_payload = C.read_json(split_path)
    split = {k: np.asarray(split_payload["rows"][k], dtype=np.int64) for k in ("train", "valid", "test")}
    cache = DATA.load_cache(cache_path)
    labels = np.load(labels_path)
    if labels.ndim != 2 or labels.shape[1] != C.N_TASKS:
        raise RuntimeError(f"label artifact {labels.shape} is not [N,{C.N_TASKS}]")
    if int(cache["n_molecules"]) != labels.shape[0]:
        raise RuntimeError(f"cache {cache['n_molecules']} molecules != labels {labels.shape[0]} rows")
    if str(cache["meta"].get("feature_spec_hash")) != spec.hash():
        raise RuntimeError("cache feature-spec hash does not match the loaded frozen spec")
    # The Test rows must be absent from every artifact this code can reach.
    if np.isfinite(labels[split["test"]]).any():
        raise RuntimeError("label artifact exposes official Test targets; refusing to train")
    if not np.array_equal(np.asarray(cache["gids"]), np.arange(int(cache["n_molecules"]))):
        raise RuntimeError("cache rows are not the official row order")
    train_rows, valid_rows = split["train"], split["valid"]
    if np.any(train_rows < 0) or np.any(valid_rows < 0):
        raise RuntimeError("negative split row")
    norm = D.fit_norm_rows(cache, spec, train_rows, source="pcba_official_train_only")
    return Task(
        spec=spec,
        cache=cache,
        labels=labels,
        split=split,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=split["test"],
        norm=norm,
        load_seconds=float(time.perf_counter() - t0),
        cache_path=Path(cache_path),
        labels_path=Path(labels_path),
        split_partition_sha256=str(split_payload["partition_sha256"]),
    )


class Tracker:
    """Best Validation AP, ties -> earlier epoch (strict improvement only)."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_metric = -math.inf
        self.last_strict_improve = 0
        self.scores: list[dict[str, Any]] = []

    def consider(self, epoch: int, metric: float) -> bool:
        strict = self.best_epoch is None or metric > self.best_metric
        self.scores.append({"epoch": int(epoch), "metric": float(metric), "strict_improve": bool(strict)})
        if strict:
            self.best_epoch = int(epoch)
            self.best_metric = float(metric)
            self.last_strict_improve = int(epoch)
        return strict

    def tie_check(self) -> dict[str, Any]:
        best_epoch = None
        best_key = None
        for row in self.scores:
            key = (row["metric"], -row["epoch"])
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = row["epoch"]
        return {
            "argbest_epoch": best_epoch,
            "tracked_epoch": self.best_epoch,
            "consistent": best_epoch == self.best_epoch,
            "rule": "max Validation AP; ties resolved to the earlier epoch",
        }


class Watchdog:
    """Arms a hard-stop flag at ``hard_seconds`` (graceful save, no SIGKILL)."""

    def __init__(self, hard_seconds: float):
        self.hard_seconds = float(hard_seconds)
        self.stop = threading.Event()
        self.fired_at: float | None = None
        self._t0 = time.perf_counter()
        self._timer = threading.Timer(self.hard_seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.fired_at = time.perf_counter() - self._t0
        self.stop.set()

    def cancel(self) -> None:
        self._timer.cancel()


def _batches(rows: np.ndarray, batch_size: int, generator: torch.Generator) -> list[np.ndarray]:
    perm = torch.randperm(len(rows), generator=generator).numpy()
    shuffled = np.asarray(rows, dtype=np.int64)[perm]
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def _write_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CURVE_FIELDS})
    return C.sha256_file(path)


def train(
    task: Task,
    *,
    out_dir: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    soft_seconds: float,
    hard_seconds: float,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model, model_info = BM.build_c1(task.spec, task.norm, C.N_TASKS, int(seed), device)
    params = BM.parameter_count(model)
    recipe = dict(C.RECIPE)
    horizon = int(recipe["scheduler_horizon_epochs"])
    steps_per_epoch = int(math.ceil(len(task.train_rows) / batch_size))
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
    generator = torch.Generator().manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    watchdog = Watchdog(hard_seconds)
    tracker = Tracker()
    curve: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    checkpoint_path = out_dir / "checkpoint.pt"
    selection_path = out_dir / "selection.json"
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_checkpoint_sha = ""
    grad_norm_max = 0.0
    clip_events = 0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    epochs_run = 0
    stop_reason = "MAX_EPOCHS"
    resource_limited = False
    partial_epoch_discarded = False
    finite_failures = 0

    for epoch in range(1, int(max_epochs) + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            resource_limited = True
            break
        if time.perf_counter() - t_start > soft_seconds:
            stop_reason = "SOFT_WALL_LIMIT"
            resource_limited = True
            break
        model.train()
        t_epoch = time.perf_counter()
        losses: list[float] = []
        aborted = False
        for step, rows in enumerate(_batches(task.train_rows, batch_size, generator), start=1):
            if watchdog.stop.is_set():
                aborted = True
                break
            batch = BM.to_device(BM.collate(task.cache, rows.tolist()), device)
            y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
            logits = model(batch)
            if logits.shape != (len(rows), C.N_TASKS):
                raise RuntimeError(f"model output {tuple(logits.shape)} != {(len(rows), C.N_TASKS)}")
            loss = MET.masked_bce_loss(logits, y)
            if not torch.isfinite(loss):
                finite_failures += 1
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}: {float(loss)}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(gnorm):
                finite_failures += 1
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} step {step}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))
        train_seconds = time.perf_counter() - t_epoch
        if aborted:
            partial_epoch_discarded = True
            stop_reason = "HARD_WALL_LIMIT"
            resource_limited = True
            C.log(f"[train] hard wall limit during epoch {epoch}; partial epoch discarded")
            break
        t_eval = time.perf_counter()
        logits_valid = BM.predict_logits(model, task.cache, task.valid_rows, device, batch_size=256)
        probs_valid = BM.sigmoid(logits_valid)
        report = MET.validation_report(task.labels[task.valid_rows], probs_valid)
        valid_seconds = time.perf_counter() - t_eval
        strict = tracker.consider(epoch, float(report["ap"]))
        if strict:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = report
            best_checkpoint_sha = _atomic_torch_save(
                {
                    "state_dict": best_state,
                    "round_id": C.ROUND_ID,
                    "result_label": C.RESULT_LABEL,
                    "dataset": C.DATASET,
                    "variant": C.MODEL_VARIANT,
                    "seed": int(seed),
                    "n_outputs": C.N_TASKS,
                    "best_epoch": int(epoch),
                    "best_metric": float(report["ap"]),
                    "metric": "Validation AP (OGB official Evaluator)",
                    "feature_spec_hash": task.spec.hash(),
                    "norm_fingerprint": task.norm["norm_fingerprint"],
                    "model_info": model_info,
                    "recipe": recipe,
                    "batch_size": int(batch_size),
                    "epochs_run_so_far": int(epoch),
                },
                checkpoint_path,
            )
            C.write_json(
                selection_path,
                {
                    "best_epoch": int(epoch),
                    "best_valid_ap": float(report["ap"]),
                    "checkpoint": C.track_rel(checkpoint_path),
                    "checkpoint_sha256": best_checkpoint_sha,
                    "scores": tracker.scores,
                    "tie_check": tracker.tie_check(),
                    "updated_utc": C.utcnow(),
                    "test_evaluated": False,
                },
            )
        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - t_epoch
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        curve.append(
            {
                "epoch": int(epoch),
                "steps_per_epoch": int(steps_per_epoch),
                "lr": lr_now,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "valid_ap": float(report["ap"]),
                "n_tasks_with_observed_labels": int(report["n_tasks_with_observed_labels"]),
                "n_tasks_scored": int(report["n_tasks_scored"]),
                "train_seconds": float(train_seconds),
                "valid_seconds": float(valid_seconds),
                "epoch_seconds": float(epoch_seconds),
                "cumulative_seconds": float(time.perf_counter() - t_start),
                "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
                "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
                "best_epoch_so_far": int(tracker.best_epoch),
                "best_valid_ap_so_far": float(tracker.best_metric),
                "is_best": bool(strict),
                "strict_improve": bool(strict),
            }
        )
        for t, ap in enumerate(report["per_task_ap"]):
            per_task_rows.append(
                {
                    "epoch": int(epoch),
                    "task": int(t),
                    "ap": "" if ap is None else float(ap),
                    "observed": int(report["observed"][t]),
                    "positive": int(report["positive"][t]),
                    "negative": int(report["negative"][t]),
                }
            )
        # flush per-epoch evidence immediately (survives a later kill)
        _write_curve(curve_path, curve)
        with open(per_task_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["epoch", "task", "ap", "observed", "positive", "negative"])
            writer.writeheader()
            writer.writerows(per_task_rows)
        epochs_run = int(epoch)
        C.log(
            f"[pcba] epoch {epoch}/{max_epochs} loss {np.mean(losses):.5f} validAP {report['ap']:.6f}"
            f" tasks {report['n_tasks_scored']} lr {lr_now:.3e} {epoch_seconds:.1f}s"
            f"{' *best*' if strict else ''}"
        )
        if epoch >= int(min_epochs) and epoch - tracker.last_strict_improve >= int(patience):
            stop_reason = "EARLY_STOP_PATIENCE"
            break
    watchdog.cancel()
    if best_state is None or best_metrics is None or tracker.best_epoch is None:
        raise RuntimeError("training ended without selecting a Validation checkpoint")
    tie = tracker.tie_check()
    if not tie["consistent"]:
        raise RuntimeError(f"selection re-derivation inconsistent: {tie}")
    total_seconds = time.perf_counter() - t_start

    # Independent reload of the selected checkpoint (GPU inference path).
    reloaded, _ = BM.build_c1(task.spec, task.norm, C.N_TASKS, int(seed), device)
    reloaded.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
    logits_re = BM.predict_logits(reloaded, task.cache, task.valid_rows, device, batch_size=256)
    ap_re = MET.official_ap(task.labels[task.valid_rows], BM.sigmoid(logits_re))["value"]
    reload_delta = abs(float(ap_re) - float(tracker.best_metric))
    if reload_delta > 1e-4:
        raise RuntimeError(
            f"reloaded checkpoint does not reproduce the selected Validation AP: "
            f"stored={tracker.best_metric:.12f} reloaded={ap_re:.12f} delta={reload_delta:.3e}"
        )
    result: dict[str, Any] = {
        "status": "RESOURCE_LIMITED" if resource_limited else "COMPLETED",
        "stop_reason": stop_reason,
        "resource_limited": bool(resource_limited),
        "partial_epoch_discarded": bool(partial_epoch_discarded),
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "dataset": C.DATASET,
        "variant": C.MODEL_VARIANT,
        "variant_definition": C.MODEL_VARIANT_DEFINITION,
        "seed": int(seed),
        "n_outputs": C.N_TASKS,
        "parameters": int(params),
        "model_info": model_info,
        "recipe": recipe,
        "batch_size": int(batch_size),
        "steps_per_epoch": int(steps_per_epoch),
        "scheduler_total_updates": int(total_updates),
        "epochs_run": int(epochs_run),
        "max_epochs": int(max_epochs),
        "min_epochs": int(min_epochs),
        "patience": int(patience),
        "best_epoch": int(tracker.best_epoch),
        "best_valid_ap": float(tracker.best_metric),
        "reload_valid_ap": float(ap_re),
        "reload_valid_delta": float(reload_delta),
        "best_valid_ap_source": "OGB official Evaluator(ogbg-molpcba)",
        "best_valid_tasks_scored": int(best_metrics["n_tasks_scored"]),
        "best_valid_tasks_with_labels": int(best_metrics["n_tasks_with_observed_labels"]),
        "curve": C.track_rel(curve_path),
        "curve_sha256": C.sha256_file(curve_path),
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": C.sha256_file(per_task_path),
        "checkpoint": C.track_rel(checkpoint_path),
        "checkpoint_sha256": best_checkpoint_sha,
        "selection": C.track_rel(selection_path),
        "selection_sha256": C.sha256_file(selection_path),
        "scores": tracker.scores,
        "tie_check": tie,
        "seconds": float(total_seconds),
        "gpu_seconds": float(total_seconds),
        "gpu_hours": float(total_seconds / 3600.0),
        "data_load_seconds": float(task.load_seconds),
        "train_seconds": float(sum(r["train_seconds"] for r in curve)),
        "valid_seconds": float(sum(r["valid_seconds"] for r in curve)),
        "seconds_per_epoch_mean": float(np.mean([r["epoch_seconds"] for r in curve])) if curve else 0.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "grad_clip_events": int(clip_events),
        "grad_norm_max": float(grad_norm_max),
        "non_finite_events": int(finite_failures),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "budget": C.BUDGET,
        "cache_identity": {
            "cache": C.track_rel(task.cache_path),
            "cache_sha256": C.sha256_tree(task.cache_path)
        if Path(task.cache_path).is_dir()
        else C.sha256_file(task.cache_path),
        "cache_format": "npy-directory-v1" if Path(task.cache_path).is_dir() else "pickle",
            "feature_spec_hash": task.spec.hash(),
            "norm_fingerprint": task.norm["norm_fingerprint"],
            "split_partition_sha256": task.split_partition_sha256,
            "label_artifact": C.track_rel(task.labels_path),
            "label_artifact_sha256": C.sha256_file(task.labels_path),
        },
        "environment": C.environment(),
        "started_utc": C.utcnow(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
    }
    C.write_json(out_dir / "run.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--batch-size", type=int, default=int(C.RECIPE["batch_size"]))
    parser.add_argument("--max-epochs", type=int, default=int(C.RECIPE["max_epochs"]))
    parser.add_argument("--min-epochs", type=int, default=int(C.RECIPE["min_epochs"]))
    parser.add_argument("--patience", type=int, default=int(C.RECIPE["patience"]))
    parser.add_argument("--soft-seconds", type=float, default=float(C.BUDGET["train_soft_seconds"]))
    parser.add_argument("--hard-seconds", type=float, default=float(C.BUDGET["train_hard_seconds"]))
    parser.add_argument("--cache", default=str(C.CACHE / "pcba_v1"))
    parser.add_argument("--labels", default=str(C.ANALYSIS / "pcba_labels_trainvalid.npy"))
    parser.add_argument("--spec", default=str(C.ANALYSIS / "pcba_spec.json"))
    parser.add_argument("--split", default=str(C.ANALYSIS / "pcba_split.json"))
    parser.add_argument("--out-dir", default=str(C.RUNS / "pcba_seed0"))
    parser.add_argument("--smoke", action="store_true", help="run the short correctness/timing smoke instead")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "status": "RESOURCE_BLOCKED",
                "reason": "CUDA requested but torch.cuda.is_available() is False",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        return 2
    try:
        C.assert_frozen_inheritance()
        task = load_task(Path(args.cache), Path(args.labels), Path(args.spec), Path(args.split))
        result = train(
            task,
            out_dir=out_dir,
            device=device,
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            max_epochs=int(args.max_epochs),
            min_epochs=int(args.min_epochs),
            patience=int(args.patience),
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
        )
        C.log(f"done {C.track_rel(out_dir)} status={result['status']} ap={result['best_valid_ap']:.6f}")
        return 0
    except Exception as exc:  # noqa: BLE001 - recorded for the failure table
        C.write_json(
            out_dir / "run.json",
            {
                "status": "FAILED",
                "reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"FAILED pcba: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
