"""One formal TU unit: ``(dataset, fold, seed, model)``.

Training recipe is the frozen TU quick-screen recipe unchanged: batch 64,
AdamW lr 1e-3 / wd 1e-4, per-step cosine to 1e-5, dropout 0.10, unweighted
CrossEntropyLoss, grad clip 5.0, 200 full epochs, no early stopping, no
EMA/SWA/soup/SSL/augmentation.

Fold protocol (frozen before any formal run, see ``configs/protocol.json``):
outer ``StratifiedKFold(n_splits, shuffle=True, random_state=fold_seed)``; inside
each outer fold a fixed stratified 10% of the outer-training part is the
checkpoint-selection Validation split (``random_state = 20261102 + fold_id``);
the outer held-out fold is the evaluation split.  Both models and all training
seeds share the same folds.

Checkpoint selection: max Valid accuracy; ties -> lower Valid CE; ties ->
earlier epoch.  The held-out fold is evaluated once, only after the selected
checkpoint's SHA-256 has been written.
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

import ptb_common as C
import ptb_data as D

import lcrc_data as PREP_DATA
import tu_common as TC
import tu_model as TM

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# metrics / prediction (frozen TU implementations, imported verbatim in spirit)
# ---------------------------------------------------------------------------


def classification_metrics(logits, y: np.ndarray, n_classes: int) -> dict[str, Any]:
    logits_t = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.int64))
    ce = float(F.cross_entropy(logits_t, y_t, reduction="mean").item())
    pred = logits_t.argmax(dim=1).numpy()
    y = y_t.numpy()
    n = int(y.size)
    acc = float((pred == y).sum() / max(n, 1))
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y.tolist(), pred.tolist()):
        confusion[int(t), int(p)] += 1
    f1s = []
    for c in range(n_classes):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum() - confusion[c, c])
        fn = float(confusion[c, :].sum() - confusion[c, c])
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return {
        "n": n,
        "cross_entropy": ce,
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": [float(x) for x in f1s],
        "confusion_matrix": confusion.tolist(),
    }


def predict_logits(model, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = _to_device(PREP_DATA.collate(cache, rows), device)
            out.append(model(batch).detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)
    if logits.shape[0] != n:
        raise RuntimeError(f"prediction rows {logits.shape[0]} != {n}")
    return logits


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class BestValidTracker:
    """Max Valid accuracy; tie -> lower Valid CE; tie -> earlier epoch."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_acc = -math.inf
        self.best_ce = math.inf
        self.best_metrics: dict[str, Any] | None = None
        self.state: dict[str, torch.Tensor] | None = None
        self.scores: list[dict[str, Any]] = []

    def consider(self, epoch: int, metrics: Mapping[str, Any]) -> bool:
        acc = float(metrics["accuracy"])
        ce = float(metrics["cross_entropy"])
        better = self.best_epoch is None or acc > self.best_acc or (acc == self.best_acc and ce < self.best_ce)
        self.scores.append({"epoch": int(epoch), "accuracy": acc, "cross_entropy": ce, "is_best": bool(better)})
        if not better:
            return False
        self.best_epoch = int(epoch)
        self.best_acc = acc
        self.best_ce = ce
        self.best_metrics = dict(metrics)
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def check_tie_break(self) -> dict[str, Any]:
        best = None
        for row in self.scores:
            key = (row["accuracy"], -row["cross_entropy"], -row["epoch"])
            if best is None or key > best[0]:
                best = (key, row["epoch"])
        return {
            "argbest_epoch": int(best[1]) if best is not None else None,
            "tracked_epoch": self.best_epoch,
            "consistent": bool(best is not None and int(best[1]) == int(self.best_epoch)),
        }


def _model_state(spec, norm, n_classes: int, model_name: str, seed: int) -> dict[str, torch.Tensor]:
    """Frozen shared initialisation (C1 = FULL minus the b_mean columns)."""
    full = TM.build_full_state(spec, norm, n_classes, int(seed))
    return TM.remap_state(model_name, full, spec, n_classes)


def _shared_equal(model_name: str, full: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor], spec) -> dict[str, Any]:
    return TM.shared_state_equal(model_name, full, state, spec)


def train_unit(
    dataset: str,
    model_name: str,
    fold: int,
    seed: int,
    out_dir: Path,
    device: torch.device,
    *,
    n_splits: int,
    fold_seed: int,
    epochs: int | None = None,
    smoke: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if dataset not in C.TU_DATASETS:
        raise KeyError(dataset)
    if model_name not in C.MODELS:
        raise KeyError(f"{model_name} is not one of the two models under test")
    if smoke:
        epochs = int(epochs if epochs is not None else 2)
    else:
        epochs = int(C.TU_EPOCHS) if epochs is None else int(epochs)
    if not smoke and epochs != int(C.TU_EPOCHS):
        raise ValueError(f"formal TU runs must use {C.TU_EPOCHS} epochs")
    out_dir = Path(out_dir)
    run_json = out_dir / "run.json"
    if run_json.exists() and not force:
        raise RuntimeError(f"{C.track_rel(run_json)} exists; refusing to overwrite a recorded unit")
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = D.load_fold(dataset, int(n_splits), int(fold), int(fold_seed))
    spec = prepared["spec"]
    norm = prepared["norm"]
    n_classes = int(prepared["n_classes"])
    train_cache = prepared["caches_obj"]["train"]
    valid_cache = prepared["caches_obj"]["valid"]
    test_cache = prepared["caches_obj"]["test"]
    y_train = np.asarray(prepared["labels"]["train"], dtype=np.int64)
    y_valid = np.asarray(prepared["labels"]["valid"], dtype=np.int64)
    y_test = np.asarray(prepared["labels"]["test"], dtype=np.int64)
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])

    determinism = TC.seed_everything(int(seed))
    tu_name = C.TU_MODEL_NAME[model_name]
    full = TM.build_full_state(spec, norm, n_classes, int(seed))
    init_state = TM.remap_state(tu_name, full, spec, n_classes)
    model = TM.TULCRC(spec, norm, n_classes, tu_name)
    model.load_state_dict(init_state, strict=True)
    model.to(device)
    shared = _shared_equal(tu_name, full, init_state, spec)
    params = TM.parameter_count(model)

    recipe = dict(TC.RECIPE)
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs
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
    generator = torch.Generator().manual_seed(int(seed))

    dropout_calls = {"n": 0}
    hook = model.drop.register_forward_hook(lambda *_: dropout_calls.__setitem__("n", dropout_calls["n"] + 1))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    tracker = BestValidTracker()
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    train_seconds_total = 0.0
    valid_seconds_total = 0.0
    train_wall = 0.0

    t_train = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=generator).tolist()
        t_ep = time.perf_counter()
        for rows in (perm[i : i + batch_size] for i in range(0, n_train, batch_size)):
            batch = _to_device(PREP_DATA.collate(train_cache, rows), device)
            y = torch.as_tensor(y_train[rows], dtype=torch.int64, device=device)
            logits = model(batch)
            loss = F.cross_entropy(logits, y)
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
        train_seconds_total += train_seconds

        t_eval = time.perf_counter()
        train_logits = predict_logits(model, train_cache, device)
        train_metrics = classification_metrics(train_logits, y_train, n_classes)
        valid_logits = predict_logits(model, valid_cache, device)
        valid_metrics = classification_metrics(valid_logits, y_valid, n_classes)
        valid_seconds = time.perf_counter() - t_eval
        valid_seconds_total += valid_seconds
        improved = tracker.consider(epoch, valid_metrics)
        if improved:
            tracker.capture(model)
        curve.append(
            {
                "epoch": int(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_clip_events": int(clip_events),
                "train_seconds": float(train_seconds),
                "valid_seconds": float(valid_seconds),
                "is_best": bool(improved),
                "train_ce": float(train_metrics["cross_entropy"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "valid_ce": float(valid_metrics["cross_entropy"]),
                "valid_accuracy": float(valid_metrics["accuracy"]),
                "valid_macro_f1": float(valid_metrics["macro_f1"]),
            }
        )
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        if smoke or epoch == epochs or improved:
            C.log(
                f"[{dataset}/f{fold}/{model_name}/s{seed}] epoch {epoch}/{epochs} "
                f"valid acc {valid_metrics['accuracy']:.4f} ce {valid_metrics['cross_entropy']:.4f}"
                f"{' *best*' if improved else ''}"
            )
    train_wall = time.perf_counter() - t_train
    hook.remove()

    if tracker.best_epoch is None or tracker.state is None:
        raise RuntimeError("training ended without selecting a checkpoint")
    tie_check = tracker.check_tie_break()
    if not tie_check["consistent"]:
        raise RuntimeError(f"tie-break re-derivation inconsistent: {tie_check}")

    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    curve_sha = C.sha256_file(curve_path)

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": tracker.state,
            "round_id": C.ROUND_ID,
            "dataset": dataset,
            "variant": model_name,
            "fold": int(fold),
            "n_splits": int(n_splits),
            "fold_seed": int(fold_seed),
            "seed": int(seed),
            "epochs_run": int(epochs),
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_accuracy",
            "best_metric_value": float(tracker.best_acc),
            "best_valid_ce": float(tracker.best_ce),
            "selection": TC.RECIPE["selection"],
            "feature_spec_hash": spec.hash(),
            "norm_fingerprint": prepared["norm_fingerprint"],
            "partition_sha256": {k: v["sha256"] for k, v in prepared["partition"].items()},
            "init_state_hash": TM.state_fingerprint(init_state),
            "parameters": int(params),
            "recipe": recipe,
            "model_config": dict(TC.MODEL_CONFIG),
            "smoke": bool(smoke),
        },
        checkpoint_path,
    )
    checkpoint_sha = C.sha256_file(checkpoint_path)
    C.write_json(
        out_dir / "selection.json",
        {
            "dataset": dataset,
            "variant": model_name,
            "fold": int(fold),
            "seed": int(seed),
            "best_epoch": int(tracker.best_epoch),
            "best_valid_accuracy": float(tracker.best_acc),
            "best_valid_ce": float(tracker.best_ce),
            "checkpoint_path": C.track_rel(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "curve_sha256": curve_sha,
            "test_evaluated": False,
            "selection_finished_utc": C.utcnow(),
        },
    )

    # ---- reload the selected state and score Train/Valid/held-out fold -------
    reloaded = TM.TULCRC(spec, norm, n_classes, tu_name)
    reloaded.load_state_dict({k: v.cpu() for k, v in tracker.state.items()}, strict=True)
    reloaded.to(device)
    reloaded.eval()
    train_logits = predict_logits(reloaded, train_cache, device)
    valid_logits = predict_logits(reloaded, valid_cache, device)
    train_metrics = classification_metrics(train_logits, y_train, n_classes)
    valid_metrics = classification_metrics(valid_logits, y_valid, n_classes)
    if abs(float(valid_metrics["accuracy"]) - float(tracker.best_acc)) > 1e-12:
        raise RuntimeError("reloaded checkpoint does not reproduce the selected Valid accuracy")

    np.savez(
        out_dir / "valid_predictions.npz",
        gids=np.asarray(valid_cache["gids"], dtype=np.int64),
        logits=valid_logits,
        y_true=y_valid,
    )
    np.savez(
        out_dir / "train_predictions.npz",
        gids=np.asarray(train_cache["gids"], dtype=np.int64),
        logits=train_logits,
        y_true=y_train,
    )

    test_metrics: dict[str, Any] | None = None
    test_seconds = 0.0
    test_logits = None
    if not smoke:
        if C.read_json(out_dir / "selection.json")["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("selection record changed before the Test read")
        t_test = time.perf_counter()
        test_logits = predict_logits(reloaded, test_cache, device)
        test_metrics = classification_metrics(test_logits, y_test, n_classes)
        test_seconds = time.perf_counter() - t_test
        np.savez(
            out_dir / "test_predictions.npz",
            gids=np.asarray(test_cache["gids"], dtype=np.int64),
            logits=test_logits,
            y_true=y_test,
        )

    serialisable_curve = curve
    record = {
        "schema": "lcrc-ptb-tu-run-v1",
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "env": C.env_info(),
        "dataset": dataset,
        "variant": model_name,
        "model_label": C.MODEL_LABELS[model_name],
        "fold": int(fold),
        "n_splits": int(n_splits),
        "fold_seed": int(fold_seed),
        "seed": int(seed),
        "smoke": bool(smoke),
        "epochs_run": int(epochs),
        "n_classes": n_classes,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": int(test_cache["n_molecules"]),
        "parameters": int(params),
        "node_type_mode": prepared["node_type_mode"],
        "n_node_types": int(prepared["n_node_types"]),
        "feature_spec_hash": spec.hash(),
        "norm_fingerprint": prepared["norm_fingerprint"],
        "partition": prepared["partition"],
        "norm_audit": prepared["norm_audit"],
        "shared_init": shared,
        "determinism": determinism,
        "recipe": recipe,
        "dropout_calls_in_training": int(dropout_calls["n"]),
        "train": train_metrics,
        "validation": valid_metrics,
        "test": test_metrics,
        "gap": {
            "train_minus_valid_accuracy": float(train_metrics["accuracy"] - valid_metrics["accuracy"]),
            "train_minus_valid_ce": float(train_metrics["cross_entropy"] - valid_metrics["cross_entropy"]),
        },
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_valid_accuracy": float(tracker.best_acc),
            "best_valid_ce": float(tracker.best_ce),
            "best_state_hash": TM.state_fingerprint(tracker.state),
            "sha256": checkpoint_sha,
            "path": C.track_rel(checkpoint_path),
            "curve_sha256": curve_sha,
            "tie_break": tie_check,
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": 0,
        },
        "timing": {
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "train_wall_seconds": float(train_wall),
            "train_seconds_total": float(train_seconds_total),
            "valid_seconds_total": float(valid_seconds_total),
            "test_seconds": float(test_seconds),
            "mean_epoch_seconds": float((train_seconds_total + valid_seconds_total) / max(epochs, 1)),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "gpu_hours_estimate": float((train_wall + test_seconds) / 3600.0),
        },
        "curve": serialisable_curve,
        "result_label": "ENGINEERING_SMOKE" if smoke else C.TU_RESULT_LABEL,
    }
    C.write_json(run_json, record)
    C.log(
        f"[{dataset}/f{fold}/{model_name}/s{seed}] done best_epoch={tracker.best_epoch} "
        f"valid={float(tracker.best_acc):.4f} test={None if test_metrics is None else round(float(test_metrics['accuracy']),4)} "
        f"wall={train_wall/60:.1f}min"
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One TU cross-validation unit")
    ap.add_argument("--dataset", required=True, choices=list(C.TU_DATASETS))
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-splits", type=int, required=True)
    ap.add_argument("--fold-seed", type=int, default=20261102)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.smoke:
        out_dir = C.RUNS / "smoke" / C.tu_run_name(args.dataset, args.model, args.fold, args.seed)
    else:
        out_dir = C.tu_run_dir(args.dataset, args.model, args.fold, args.seed)
    record = train_unit(
        args.dataset,
        args.model,
        int(args.fold),
        int(args.seed),
        out_dir,
        torch.device(args.device),
        n_splits=int(args.n_splits),
        fold_seed=int(args.fold_seed),
        epochs=args.epochs,
        smoke=bool(args.smoke),
        force=bool(args.force),
    )
    print(C.stable_json({k: record[k] for k in ("dataset", "variant", "fold", "seed", "checkpoint")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
