"""One formal training task or one 5-epoch throughput precheck.

The task identity is ``(kind, dataset, unit)``.  A completed ``run.json`` is
never overwritten: an interrupted attempt is moved aside before a new attempt
starts.  The Test labels are opened only after the selected checkpoint has been
written and its SHA-256 recorded.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _configure_imports() -> None:
    import os
    import sys

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import bp_common as C  # noqa: E402
import bp_data as D  # noqa: E402
import bp_metrics as MET  # noqa: E402
import bp_model as BM  # noqa: E402
import lcrc_data as PREP_DATA  # noqa: E402
from feature_spec import load_spec  # noqa: E402


class TaskTimeout(RuntimeError):
    pass


class OOMBatch(RuntimeError):
    pass


class TaskResourceBlocked(RuntimeError):
    pass


@dataclass
class TaskData:
    kind: str
    dataset: str
    unit: int
    spec: Any
    norm: dict[str, Any]
    n_out: int
    train_cache: Mapping[str, Any]
    valid_cache: Mapping[str, Any]
    train_rows: np.ndarray
    valid_rows: np.ndarray
    test_rows: np.ndarray | None
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test_path: Path | None
    y_test: np.ndarray | None
    cont_slice: tuple[int, int]
    cont_norm: dict[str, Any]
    seed: int
    recipe: dict[str, Any]
    split_record: dict[str, Any]
    cache_identity: dict[str, Any]
    test_cache_loader: Any | None = None

    @property
    def n_train(self) -> int:
        return int(len(self.train_rows))

    @property
    def n_valid(self) -> int:
        return int(len(self.valid_rows))


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _batches(rows: np.ndarray, batch_size: int, generator: torch.Generator) -> list[np.ndarray]:
    perm = torch.randperm(len(rows), generator=generator).numpy()
    shuffled = np.asarray(rows, dtype=np.int64)[perm]
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def predict_logits(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = np.asarray(rows[start : start + batch_size], dtype=np.int64)
            batch = _to_device(D.collate(cache, chunk.tolist()), device)
            out.append(model(batch).detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)
    if logits.shape[0] != len(rows):
        raise RuntimeError(f"prediction rows {logits.shape[0]} != {len(rows)}")
    return logits


def _probs(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("logits must be 2-D")
    if x.shape[1] == 1:
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def evaluate(
    kind: str,
    dataset: str,
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    rows: np.ndarray,
    y_full: np.ndarray,
    device: torch.device,
    *,
    n_classes: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    y = np.asarray(y_full)[rows]
    logits = predict_logits(model, cache, rows, device)
    if kind == "tu":
        if n_classes is None:
            n_classes = int(np.max(y)) + 1
        metrics = MET.classification_metrics(logits, y, int(n_classes))
        probs = _probs(logits)
    elif kind == "mol":
        probs = _sigmoid(logits)
        metrics = MET.ogb_rocauc(dataset, np.asarray(y, dtype=np.float64), probs)
        metrics["n"] = int(len(rows))
    elif kind == "peptide":
        probs = _sigmoid(logits)
        metrics = MET.macro_ap(np.asarray(y, dtype=np.float64), probs)
        metrics["n"] = int(len(rows))
    else:
        raise KeyError(kind)
    return metrics, logits, probs


def _peptide_test_loader() -> tuple[Mapping[str, Any], np.ndarray]:
    cache = D.load_cache(C.CACHE / "peptides_func_test.pt")
    return cache, np.arange(int(cache["n_molecules"]), dtype=np.int64)


def _make_task(kind: str, dataset: str, unit: int) -> TaskData:
    if kind == "tu":
        meta = C.TU_DATASETS[dataset]
        spec = load_spec(C.ANALYSIS / f"{dataset}_spec.json")
        cache = D.load_cache(C.CACHE / f"{dataset}.pt")
        folds = C.read_json(C.ANALYSIS / f"{dataset}_folds.json")
        fold = folds["folds"][int(unit)]
        train_rows = np.asarray(fold["rows"]["train"], dtype=np.int64)
        valid_rows = np.asarray(fold["rows"]["valid"], dtype=np.int64)
        test_rows = np.asarray(fold["rows"]["test"], dtype=np.int64)
        y = np.load(C.ANALYSIS / f"{dataset}_labels.npy")
        norm = D.fit_norm_rows(cache, spec, train_rows, source=f"tu_fold{unit}_train_only")
        cont_start = int(spec.n_atom_types) + 29
        cont_dim = int(spec.atom_feat_dim) - cont_start
        cont_norm = D.fit_continuous_atom_norm(cache, train_rows, start=cont_start, dim=cont_dim)
        return TaskData(
            kind=kind,
            dataset=dataset,
            unit=int(unit),
            spec=spec,
            norm=norm,
            n_out=int(meta["n_classes"]),
            train_cache=cache,
            valid_cache=cache,
            train_rows=train_rows,
            valid_rows=valid_rows,
            test_rows=test_rows,
            y_train=np.asarray(y, dtype=np.int64),
            y_valid=np.asarray(y, dtype=np.int64),
            y_test_path=C.ANALYSIS / f"{dataset}_labels.npy",
            y_test=None,
            cont_slice=(cont_start, cont_start + cont_dim),
            cont_norm=cont_norm,
            seed=C.TU_MODEL_SEED_BASE + int(unit),
            recipe=dict(C.TU_RECIPE),
            split_record=fold,
            cache_identity={
                "cache": C.track_rel(C.CACHE / f"{dataset}.pt"),
                "cache_sha256": C.sha256_file(C.CACHE / f"{dataset}.pt"),
                "feature_spec_hash": spec.hash(),
                "fold_partition_sha256": fold["partition_sha256"],
                "norm_fingerprint": norm["norm_fingerprint"],
                "cont_norm_fingerprint": cont_norm.get("fingerprint", ""),
            },
        )
    if kind == "mol":
        short = dataset.replace("ogbg-", "")
        spec = load_spec(C.ANALYSIS / f"{short}_spec.json")
        cache = D.load_cache(C.CACHE / f"{short}.pt")
        split = C.read_json(C.ANALYSIS / f"{short}_split.json")
        rows = {k: np.asarray(split["rows"][k], dtype=np.int64) for k in ("train", "valid", "test")}
        y = np.load(C.ANALYSIS / f"{short}_labels.npy")
        norm = D.fit_norm_rows(cache, spec, rows["train"], source=f"{short}_official_train_only")
        recipe = dict(C.OGB_RECIPE)
        recipe.update(C.OGB_STOPPING[dataset])
        return TaskData(
            kind=kind,
            dataset=dataset,
            unit=int(unit),
            spec=spec,
            norm=norm,
            n_out=int(split["num_tasks"]),
            train_cache=cache,
            valid_cache=cache,
            train_rows=rows["train"],
            valid_rows=rows["valid"],
            test_rows=rows["test"],
            y_train=np.asarray(y, dtype=np.float64),
            y_valid=np.asarray(y, dtype=np.float64),
            y_test_path=C.ANALYSIS / f"{short}_labels.npy",
            y_test=None,
            cont_slice=(0, 0),
            cont_norm={"dim": 0, "mean": [], "scale": [], "constant_indices": [], "fit_rows": int(rows["train"].size)},
            seed=int(unit),
            recipe=recipe,
            split_record=split,
            cache_identity={
                "cache": C.track_rel(C.CACHE / f"{short}.pt"),
                "cache_sha256": C.sha256_file(C.CACHE / f"{short}.pt"),
                "feature_spec_hash": spec.hash(),
                "split_partition_sha256": split["partition_sha256"],
                "norm_fingerprint": norm["norm_fingerprint"],
            },
        )
    if kind == "peptide":
        if dataset != C.PEPTIDE_DATASET:
            raise KeyError(dataset)
        spec = load_spec(C.ANALYSIS / "peptides_func_spec.json")
        train_cache = D.load_cache(C.CACHE / "peptides_func_train.pt")
        valid_cache = D.load_cache(C.CACHE / "peptides_func_valid.pt")
        train_rows = np.arange(int(train_cache["n_molecules"]), dtype=np.int64)
        valid_rows = np.arange(int(valid_cache["n_molecules"]), dtype=np.int64)
        y_train = np.load(C.ANALYSIS / "peptides_func_y_train.npy").astype(np.float64)
        y_valid = np.load(C.ANALYSIS / "peptides_func_y_valid.npy").astype(np.float64)
        norm = D.fit_norm_rows(train_cache, spec, train_rows, source="peptides_func_official_train_only")
        return TaskData(
            kind=kind,
            dataset=dataset,
            unit=int(unit),
            spec=spec,
            norm=norm,
            n_out=10,
            train_cache=train_cache,
            valid_cache=valid_cache,
            train_rows=train_rows,
            valid_rows=valid_rows,
            test_rows=None,
            y_train=y_train,
            y_valid=y_valid,
            y_test_path=C.ANALYSIS / "peptides_func_y_test.npy",
            y_test=None,
            cont_slice=(0, 0),
            cont_norm={"dim": 0, "mean": [], "scale": [], "constant_indices": [], "fit_rows": int(train_rows.size)},
            seed=int(unit),
            recipe=dict(C.PEPTIDE_RECIPE),
            split_record={
                "source": "official PyG LRGBDataset splits",
                "train": int(train_rows.size),
                "valid": int(valid_rows.size),
                "test": int(C.read_json(C.ANALYSIS / "peptides_func_cache_stats.json")["splits"]["test"]["n_graphs"]),
                "labels_sha256": {
                    "train": C.sha256_file(C.ANALYSIS / "peptides_func_y_train.npy"),
                    "valid": C.sha256_file(C.ANALYSIS / "peptides_func_y_valid.npy"),
                },
            },
            cache_identity={
                "cache_train": C.track_rel(C.CACHE / "peptides_func_train.pt"),
                "cache_train_sha256": C.sha256_file(C.CACHE / "peptides_func_train.pt"),
                "cache_valid_sha256": C.sha256_file(C.CACHE / "peptides_func_valid.pt"),
                "feature_spec_hash": spec.hash(),
                "norm_fingerprint": norm["norm_fingerprint"],
            },
            test_cache_loader=_peptide_test_loader,
        )
    raise KeyError(kind)


class Tracker:
    def __init__(self, kind: str):
        self.kind = kind
        self.best_epoch: int | None = None
        self.best_metric = -math.inf
        self.best_aux = math.inf
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.last_strict_improve = 0
        self.scores: list[dict[str, Any]] = []

    def consider(self, epoch: int, metric: float, aux: float, metrics: Mapping[str, Any]) -> bool:
        if self.kind == "tu":
            better = (
                self.best_epoch is None
                or metric > self.best_metric
                or (metric == self.best_metric and aux < self.best_aux)
            )
            strict = self.best_epoch is None or metric > self.best_metric
        else:
            better = self.best_epoch is None or metric > self.best_metric
            strict = better
        self.scores.append(
            {"epoch": int(epoch), "metric": float(metric), "aux": float(aux), "is_best": bool(better), "strict_improve": bool(strict)}
        )
        if strict:
            self.last_strict_improve = int(epoch)
        if not better:
            return False
        self.best_epoch = int(epoch)
        self.best_metric = float(metric)
        self.best_aux = float(aux)
        self.best_metrics = dict(metrics)
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def tie_check(self) -> dict[str, Any]:
        best_epoch = None
        best_key = None
        for row in self.scores:
            if self.kind == "tu":
                key = (row["metric"], -row["aux"], -row["epoch"])
            else:
                key = (row["metric"], -row["epoch"])
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = row["epoch"]
        return {"argbest_epoch": best_epoch, "tracked_epoch": self.best_epoch, "consistent": best_epoch == self.best_epoch}


def _train_attempt(
    data: TaskData,
    *,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    max_seconds: float,
    precheck_epochs: int | None,
    no_test: bool,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    torch.manual_seed(int(data.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(data.seed))
    norm_payload = {
        **data.norm,
        "continuous_atom": {
            "dim": int(data.cont_norm.get("dim", 0)),
            "mean": data.cont_norm.get("mean", []),
            "scale": data.cont_norm.get("scale", []),
            "fingerprint": data.cont_norm.get("fingerprint", ""),
        },
    }
    model, model_info = BM.build_c1(
        data.spec,
        norm_payload,
        data.n_out,
        data.seed,
        device,
        cont_slice=data.cont_slice,
        cont_mean=np.asarray(data.cont_norm.get("mean", []), dtype=np.float32),
        cont_scale=np.asarray(data.cont_norm.get("scale", []), dtype=np.float32),
    )
    drop_calls = {"n": 0}
    if hasattr(model, "drop"):
        model.drop.register_forward_hook(lambda *_: drop_calls.__setitem__("n", drop_calls["n"] + 1))
    params = int(BM.parameter_count(model))
    recipe = dict(data.recipe)
    max_epochs = int(precheck_epochs if precheck_epochs is not None else recipe["max_epochs"])
    min_epochs = 10**9 if precheck_epochs is not None else int(recipe["min_epochs"])
    patience = 10**9 if precheck_epochs is not None else int(recipe["patience"])
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
    generator = torch.Generator().manual_seed(int(data.seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    tracker = Tracker(data.kind)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    started_optimizer = False
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        if time.perf_counter() - t_start > max_seconds:
            raise TaskTimeout(f"exceeded {max_seconds:.0f}s before epoch {epoch}")
        model.train()
        t_epoch = time.perf_counter()
        losses = []
        for rows in _batches(data.train_rows, batch_size, generator):
            if time.perf_counter() - t_start > max_seconds:
                raise TaskTimeout(f"exceeded {max_seconds:.0f}s during epoch {epoch}")
            try:
                batch = _to_device(D.collate(data.train_cache, rows.tolist()), device)
                if data.kind == "tu":
                    y = torch.as_tensor(data.y_train[rows], dtype=torch.int64, device=device)
                else:
                    y = torch.as_tensor(data.y_train[rows], dtype=torch.float32, device=device)
                logits = model(batch)
                if data.kind == "tu":
                    loss = torch.nn.functional.cross_entropy(logits, y)
                else:
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
                started_optimizer = True
                losses.append(float(loss.detach().cpu().item()))
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                if not started_optimizer:
                    raise OOMBatch(str(exc)) from exc
                raise TaskResourceBlocked(f"CUDA OOM after optimisation started: {exc}") from exc
        train_seconds = time.perf_counter() - t_epoch
        t_eval = time.perf_counter()
        valid_metrics, valid_logits, valid_probs = evaluate(
            data.kind, data.dataset, model, data.valid_cache, data.valid_rows, data.y_valid, device,
            n_classes=data.n_out,
        )
        valid_seconds = time.perf_counter() - t_eval
        if data.kind == "tu":
            metric = float(valid_metrics["accuracy"])
            aux = float(valid_metrics["cross_entropy"])
        else:
            metric = float(valid_metrics["value"])
            aux = float("nan")
        improved = tracker.consider(epoch, metric, aux, valid_metrics)
        if improved:
            tracker.capture(model)
        lr_now = float(optimizer.param_groups[0]["lr"])
        curve.append(
            {
                "epoch": int(epoch),
                "steps": int(steps_per_epoch),
                "lr": lr_now,
                "train_seconds": float(train_seconds),
                "valid_seconds": float(valid_seconds),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "is_best": bool(improved),
                "valid_metric": float(metric),
                "valid_aux": None if not math.isfinite(aux) else float(aux),
            }
        )
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        epochs_run = int(epoch)
        if precheck_epochs is None and epoch >= min_epochs and epoch - tracker.last_strict_improve >= patience:
            C.log(
                f"[{data.kind}/{data.dataset}/{data.unit}] early stop at epoch {epoch}; "
                f"best {tracker.best_epoch} metric {tracker.best_metric:.6f}"
            )
            break
        C.log(
            f"[{data.kind}/{data.dataset}/{data.unit}] epoch {epoch}/{max_epochs} "
            f"train_loss {np.mean(losses) if losses else float('nan'):.5f} valid {metric:.6f}"
            f"{' *best*' if improved else ''}"
        )
    if tracker.best_epoch is None or tracker.best_state is None:
        raise RuntimeError("training ended without selecting a checkpoint")
    tie = tracker.tie_check()
    if not tie["consistent"]:
        raise RuntimeError(f"selection re-derivation inconsistent: {tie}")
    total_seconds = time.perf_counter() - t_start
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    curve_sha = C.sha256_file(curve_path)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": tracker.best_state,
            "round_id": C.ROUND_ID,
            "result_label": C.RESULT_LABEL,
            "kind": data.kind,
            "dataset": data.dataset,
            "unit": int(data.unit),
            "seed": int(data.seed),
            "variant": "C1",
            "n_outputs": int(data.n_out),
            "best_epoch": int(tracker.best_epoch),
            "epochs_run": int(epochs_run),
            "best_metric": float(tracker.best_metric),
            "best_aux": None if not math.isfinite(tracker.best_aux) else float(tracker.best_aux),
            "feature_spec_hash": data.spec.hash(),
            "norm_fingerprint": data.norm["norm_fingerprint"],
            "continuous_atom_norm": norm_payload["continuous_atom"],
            "cache_identity": data.cache_identity,
            "model_info": model_info,
            "recipe": recipe,
            "batch_size": int(batch_size),
            "curve_sha256": curve_sha,
            "precheck": precheck_epochs is not None,
        },
        checkpoint_path,
    )
    checkpoint_sha = C.sha256_file(checkpoint_path)
    selection = {
        "best_epoch": int(tracker.best_epoch),
        "best_metric": float(tracker.best_metric),
        "best_aux": None if not math.isfinite(tracker.best_aux) else float(tracker.best_aux),
        "best_metrics": tracker.best_metrics,
        "scores": tracker.scores,
        "tie_check": tie,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": C.track_rel(checkpoint_path),
        "selection_finished_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(out_dir / "selection.json", selection)
    # Reload the selected checkpoint and independently recompute Valid.
    reloaded, _ = BM.build_c1(
        data.spec,
        norm_payload,
        data.n_out,
        data.seed,
        device,
        cont_slice=data.cont_slice,
        cont_mean=np.asarray(data.cont_norm.get("mean", []), dtype=np.float32),
        cont_scale=np.asarray(data.cont_norm.get("scale", []), dtype=np.float32),
    )
    reloaded.load_state_dict({k: v.to(device) for k, v in tracker.best_state.items()}, strict=True)
    reloaded.eval()
    valid_re, _, _ = evaluate(
        data.kind, data.dataset, reloaded, data.valid_cache, data.valid_rows, data.y_valid, device,
        n_classes=data.n_out,
    )
    if abs(float(valid_re["value"] if data.kind != "tu" else valid_re["accuracy"]) - float(tracker.best_metric)) > 1e-10:
        raise RuntimeError("reloaded checkpoint does not reproduce the selected validation metric")
    if data.kind == "tu":
        np.savez(
            out_dir / "train_predictions.npz",
            rows=np.asarray(data.train_rows, dtype=np.int64),
            logits=predict_logits(reloaded, data.train_cache, data.train_rows, device),
            y_true=np.asarray(data.y_train[data.train_rows]),
        )
        np.savez(
            out_dir / "valid_predictions.npz",
            rows=np.asarray(data.valid_rows, dtype=np.int64),
            logits=predict_logits(reloaded, data.valid_cache, data.valid_rows, device),
            y_true=np.asarray(data.y_valid[data.valid_rows]),
        )
    result: dict[str, Any] = {
        "status": "PRECHECK_ONLY" if precheck_epochs is not None or no_test else "TRAINED_AWAITING_TEST",
        "kind": data.kind,
        "dataset": data.dataset,
        "unit": int(data.unit),
        "seed": int(data.seed),
        "variant": "C1",
        "n_outputs": int(data.n_out),
        "parameters": params,
        "model_info": model_info,
        "recipe": recipe,
        "batch_size": int(batch_size),
        "epochs_run": int(epochs_run),
        "best_epoch": int(tracker.best_epoch),
        "best_metric": float(tracker.best_metric),
        "best_aux": None if not math.isfinite(tracker.best_aux) else float(tracker.best_aux),
        "selection": selection,
        "curve_sha256": curve_sha,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": C.track_rel(checkpoint_path),
        "seconds": float(total_seconds),
        "train_seconds": float(sum(r["train_seconds"] for r in curve)),
        "valid_seconds": float(sum(r["valid_seconds"] for r in curve)),
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "grad_clip_events": int(clip_events),
        "grad_norm_max": float(grad_norm_max),
        "dropout_calls": int(drop_calls["n"]),
        "cache_identity": data.cache_identity,
        "feature_spec_hash": data.spec.hash(),
        "norm_fingerprint": data.norm["norm_fingerprint"],
        "environment": C.environment(),
        "started_utc": C.utcnow(),
        "max_seconds": float(max_seconds),
        "test_evaluated": False,
    }
    if precheck_epochs is not None or no_test:
        C.write_json(out_dir / "run.json", result)
        return result
    # ---- terminal Test evaluation, after checkpoint hash is on disk ----------
    assert C.read_json(out_dir / "selection.json")["checkpoint_sha256"] == checkpoint_sha
    if time.perf_counter() - t_start > max_seconds:
        raise TaskTimeout("exceeded max_seconds before Test evaluation")
    if data.kind == "peptide":
        test_cache, test_rows = data.test_cache_loader()
        y_test_full = np.load(data.y_test_path).astype(np.float64)
    else:
        test_cache = data.valid_cache
        test_rows = np.asarray(data.test_rows, dtype=np.int64)
        y_test_full = np.load(data.y_test_path)
    y_test = np.asarray(y_test_full)[test_rows]
    test_metrics, test_logits, test_probs = evaluate(
        data.kind, data.dataset, reloaded, test_cache, test_rows, y_test_full, device,
        n_classes=data.n_out,
    )
    np.savez(
        out_dir / "test_predictions.npz",
        rows=np.asarray(test_rows, dtype=np.int64),
        logits=test_logits,
        probs=test_probs,
        y_true=np.asarray(y_test),
    )
    result.update(
        {
            "status": "complete",
            "test_metrics": test_metrics,
            "test_evaluated": True,
            "test_evaluated_once_utc": C.utcnow(),
            "test_cache_identity": data.cache_identity
            if data.kind != "peptide"
            else {
                **data.cache_identity,
                "cache_test_sha256": C.sha256_file(C.CACHE / "peptides_func_test.pt"),
            },
            "seconds_including_test": float(time.perf_counter() - t_start),
            "test_predictions_sha256": C.sha256_file(out_dir / "test_predictions.npz"),
        }
    )
    C.write_json(out_dir / "run.json", result)
    return result


def _archive_incomplete(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = out_dir.with_name(out_dir.name + f".incomplete.{stamp}")
    shutil.move(str(out_dir), str(dest))
    C.log(f"archived incomplete attempt to {C.track_rel(dest)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("tu", "mol", "peptide"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--unit", required=True, type=int, help="fold id (TU) or seed (mol/peptide)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--precheck-epochs", type=int, default=None)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=C.BUDGET["per_task_minutes"] * 60.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.RUNS.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else C.result_paths(args.kind, args.dataset, args.unit)
    if out_dir.exists() and (out_dir / "run.json").exists() and not args.force:
        existing = C.read_json(out_dir / "run.json")
        if existing.get("status") == "complete":
            C.log(f"skip completed {C.track_rel(out_dir)}")
            return 0
    if args.out_dir is None and out_dir.exists() and not args.force:
        _archive_incomplete(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "status": "RESOURCE_BLOCKED",
                "reason": "CUDA requested but torch.cuda.is_available() is False",
                "kind": args.kind,
                "dataset": args.dataset,
                "unit": args.unit,
                "environment": C.environment(),
            },
        )
        return 2
    try:
        data = _make_task(args.kind, args.dataset, args.unit)
        batch_sizes = [int(data.recipe["batch_size"])]
        if args.kind == "peptide":
            batch_sizes.extend(int(x) for x in data.recipe["batch_size_fallbacks"])
        last_error: Exception | None = None
        for bs in batch_sizes:
            try:
                C.log(f"start {args.kind}/{args.dataset}/unit{args.unit} batch {bs} device {device}")
                result = _train_attempt(
                    data,
                    out_dir=out_dir,
                    device=device,
                    batch_size=int(bs),
                    max_seconds=float(args.max_seconds),
                    precheck_epochs=args.precheck_epochs,
                    no_test=bool(args.no_test),
                )
                C.log(f"done {C.track_rel(out_dir)} status={result['status']}")
                return 0
            except OOMBatch as exc:
                last_error = exc
                C.log(f"batch {bs} OOM before optimisation; trying next fallback")
                torch.cuda.empty_cache()
                continue
        raise TaskResourceBlocked(f"all peptide batch sizes OOM: {last_error}")
    except TaskTimeout as exc:
        C.write_json(
            out_dir / "run.json",
            {
                "status": "RESOURCE_BLOCKED",
                "reason": "PER_TASK_TIME_LIMIT",
                "detail": str(exc),
                "kind": args.kind,
                "dataset": args.dataset,
                "unit": args.unit,
                "max_seconds": float(args.max_seconds),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"RESOURCE_BLOCKED {args.dataset}/{args.unit}: {exc}")
        return 3
    except TaskResourceBlocked as exc:
        C.write_json(
            out_dir / "run.json",
            {
                "status": "RESOURCE_BLOCKED",
                "reason": "GPU_RESOURCE",
                "detail": str(exc),
                "kind": args.kind,
                "dataset": args.dataset,
                "unit": args.unit,
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"RESOURCE_BLOCKED {args.dataset}/{args.unit}: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001 - recorded for the failure table
        C.write_json(
            out_dir / "run.json",
            {
                "status": "FAILED",
                "reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "kind": args.kind,
                "dataset": args.dataset,
                "unit": args.unit,
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"FAILED {args.dataset}/{args.unit}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
