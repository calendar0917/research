"""One formal training run for one (dataset, model, seed=0) cell.

The optimisation loop is the frozen round recipe: fp32, AdamW (lr 1e-3,
wd 1e-4, betas (0.9, 0.999), eps 1e-8), batch 64, per-step cosine to 1e-5, 200
full epochs, gradient clip 5.0, unweighted C-class CrossEntropyLoss, no early
stopping, no EMA/SWA/soup/augmentation/SSL.

Checkpoint selection uses Valid accuracy (ties: lower Valid CE, then earlier
epoch).  The Test split is evaluated exactly once, only after the best
checkpoint has been written and its SHA-256 recorded.
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

import tu_common as C  # noqa: E402  (inserts the frozen prep code path)

import lcrc_data as PREP_DATA  # noqa: E402
import tu_data as D  # noqa: E402
import tu_model as M  # noqa: E402

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def classification_metrics(logits: torch.Tensor | np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, Any]:
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


def predict_logits(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
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


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# checkpoint selection
# ---------------------------------------------------------------------------


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
        better = (
            self.best_epoch is None
            or acc > self.best_acc
            or (acc == self.best_acc and ce < self.best_ce)
        )
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
        """Re-derive the selected epoch independently from the recorded scores."""
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


# ---------------------------------------------------------------------------
# formal-run guards
# ---------------------------------------------------------------------------


def validate_run_args(dataset: str, model_name: str, seed: int, epochs: int, smoke: bool) -> None:
    if dataset not in C.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    if model_name not in C.MODEL_NAMES:
        raise ValueError(f"unknown model {model_name!r}")
    if (dataset, model_name) not in C.RUN_MATRIX:
        raise ValueError(f"{(dataset, model_name)} is not one of the 12 authorised runs")
    if not smoke:
        if int(seed) != C.TRAINING_SEED:
            raise ValueError(f"formal runs must use training seed {C.TRAINING_SEED}")
        if int(epochs) != int(C.RECIPE["epochs"]):
            raise ValueError(f"formal runs must use {C.RECIPE['epochs']} epochs")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_run(
    *,
    dataset: str,
    model_name: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    smoke: bool = False,
    epochs: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    epochs = int(C.RECIPE["epochs"]) if epochs is None else int(epochs)
    validate_run_args(dataset, model_name, seed, epochs, smoke)
    out_dir = Path(out_dir)
    run_json = out_dir / "run.json"
    if run_json.exists() and not force:
        raise RuntimeError(f"{C.track_rel(run_json)} already exists; refusing to overwrite a recorded run")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    prepared = D.load_prepared(dataset, splits=("train", "valid"))
    spec = prepared["spec"]
    n_classes = int(prepared["n_classes"])
    train_cache = prepared["caches"]["train"]
    valid_cache = prepared["caches"]["valid"]
    y_train = np.asarray(prepared["labels"]["train"], dtype=np.int64)
    y_valid = np.asarray(prepared["labels"]["valid"], dtype=np.int64)
    if smoke:
        # Short smoke only: Train/Valid, never Test.  The probe rows are drawn
        # per class so the two/three logits are all exercised.
        def _stratified_rows(y: np.ndarray, budget: int) -> np.ndarray:
            classes = np.unique(y)
            per_class = max(1, budget // max(1, classes.size))
            picks: list[int] = []
            for cls in classes.tolist():
                rows_c = np.flatnonzero(y == cls)[:per_class]
                picks.extend(int(x) for x in rows_c.tolist())
            return np.asarray(sorted(picks), dtype=np.int64)

        n_classes_all = int(prepared["n_classes"])
        train_rows = _stratified_rows(y_train, min(int(train_cache["n_molecules"]), 64 * n_classes_all))
        valid_rows = _stratified_rows(y_valid, min(int(valid_cache["n_molecules"]), 32 * n_classes_all))
        y_train = y_train[train_rows]
        y_valid = y_valid[valid_rows]
        train_cache = _subset_cache(train_cache, train_rows, spec)
        valid_cache = _subset_cache(valid_cache, valid_rows, spec)
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])

    determinism = C.seed_everything(int(seed))
    model, model_info = M.build_model(prepared, model_name, int(seed), device)
    params = M.parameter_count(model)

    batch_size = int(C.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.RECIPE["peak_lr"]),
        betas=tuple(float(x) for x in C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(C.RECIPE["min_lr"])
    )
    generator = torch.Generator().manual_seed(int(seed))

    dropout_calls = {"n": 0}
    if hasattr(model, "drop"):
        model.drop.register_forward_hook(lambda *_: dropout_calls.__setitem__("n", dropout_calls["n"] + 1))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    tracker = BestValidTracker()
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    lr_sum = 0.0
    lr_min = math.inf
    lr_max = -math.inf
    train_seconds_total = 0.0
    valid_seconds_total = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=generator).tolist()
        t_epoch = time.perf_counter()
        for rows in (perm[i : i + batch_size] for i in range(0, n_train, batch_size)):
            batch = _to_device(PREP_DATA.collate(train_cache, rows), device)
            y = torch.as_tensor(y_train[rows], dtype=torch.int64, device=device)
            logits = model(batch)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            if gnorm > float(C.RECIPE["gradient_clip"]):
                clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            scheduler.step()
            lr = float(optimizer.param_groups[0]["lr"])
            lr_sum += lr
            lr_min = min(lr_min, lr)
            lr_max = max(lr_max, lr)
        train_seconds = time.perf_counter() - t_epoch
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
                "epoch": epoch,
                "steps": steps_per_epoch,
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
                f"[{dataset}/{model_name}] epoch {epoch}/{epochs} "
                f"train acc {train_metrics['accuracy']:.4f} ce {train_metrics['cross_entropy']:.4f} | "
                f"valid acc {valid_metrics['accuracy']:.4f} ce {valid_metrics['cross_entropy']:.4f} "
                f"{'*best*' if improved else ''}"
            )

    wall_train = time.perf_counter() - t_start
    if tracker.best_epoch is None or tracker.state is None:
        raise RuntimeError("training ended without selecting a checkpoint")
    tie_check = tracker.check_tie_break()
    if not tie_check["consistent"]:
        raise RuntimeError(f"tie-break re-derivation inconsistent: {tie_check}")

    # ---- write the selected checkpoint and record its hash -----------------
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
            "model_name": model_name,
            "variant": model_name,
            "dataset": dataset,
            "n_classes": n_classes,
            "seed": int(seed),
            "epochs_run": epochs,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_accuracy",
            "best_metric_value": float(tracker.best_acc),
            "best_valid_ce": float(tracker.best_ce),
            "selection": C.RECIPE["selection"],
            "feature_spec_hash": spec.hash(),
            "split_partition_sha256": prepared["split"]["partition_sha256"],
            "norm_fingerprint": prepared["norm_record"]["norm_fingerprint"],
            "init_state_hash": model_info["init_state_hash"],
            "parameters": int(params),
            "recipe": C.RECIPE,
            "model_config": C.MODEL_CONFIG if model_name != "GIN" else C.GIN_CONFIG,
            "result_label": C.RESULT_LABEL,
            "smoke": bool(smoke),
            "curve_sha256": curve_sha,
        },
        checkpoint_path,
    )
    checkpoint_sha = C.sha256_file(checkpoint_path)
    selection_record = {
        "best_epoch": int(tracker.best_epoch),
        "best_valid_accuracy": float(tracker.best_acc),
        "best_valid_ce": float(tracker.best_ce),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": C.track_rel(checkpoint_path),
        "selection_finished_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(out_dir / "selection.json", selection_record)

    # ---- reload the selected state and score Train/Valid/Test once ---------
    reloaded = (
        M.TULCRC(spec, prepared["norm"], n_classes, model_name)
        if model_name in ("FULL", "C1")
        else M.GINPlusGlobal(spec, prepared["norm"], n_classes)
    )
    reloaded.load_state_dict({k: v.cpu() for k, v in tracker.state.items()}, strict=True)
    reloaded.to(device)
    reloaded.eval()
    train_logits = predict_logits(reloaded, train_cache, device)
    valid_logits = predict_logits(reloaded, valid_cache, device)
    train_metrics = classification_metrics(train_logits, y_train, n_classes)
    valid_metrics = classification_metrics(valid_logits, y_valid, n_classes)
    valid_recheck = abs(float(valid_metrics["accuracy"]) - float(tracker.best_acc))
    valid_ce_recheck = abs(float(valid_metrics["cross_entropy"]) - float(tracker.best_ce))
    if valid_recheck > 1e-12 or valid_ce_recheck > 1e-6:
        raise RuntimeError(
            f"reloaded checkpoint Valid accuracy {valid_metrics['accuracy']} != {tracker.best_acc}, "
            f"CE {valid_metrics['cross_entropy']} != {tracker.best_ce}"
        )

    np.savez(
        out_dir / "train_predictions.npz",
        gids=np.asarray(train_cache["gids"], dtype=np.int64),
        logits=train_logits,
        probs=torch.softmax(torch.as_tensor(train_logits), dim=1).numpy(),
        y_true=y_train,
    )
    np.savez(
        out_dir / "valid_predictions.npz",
        gids=np.asarray(valid_cache["gids"], dtype=np.int64),
        logits=valid_logits,
        probs=torch.softmax(torch.as_tensor(valid_logits), dim=1).numpy(),
        y_true=y_valid,
    )

    # Test is read exactly now: checkpoint hash is already on disk.
    test_metrics: dict[str, Any] | None = None
    test_seconds = 0.0
    if not smoke:
        assert C.read_json(out_dir / "selection.json")["checkpoint_sha256"] == checkpoint_sha
        t_test = time.perf_counter()
        # Test structures and labels are loaded only now, after the checkpoint
        # hash was written to selection.json.
        test_cache, y_test_raw = D.load_split_cache(dataset, "test")
        y_test = np.asarray(y_test_raw, dtype=np.int64)
        test_logits = predict_logits(reloaded, test_cache, device)
        test_metrics = classification_metrics(test_logits, y_test, n_classes)
        test_seconds = time.perf_counter() - t_test
        np.savez(
            out_dir / "test_predictions.npz",
            gids=np.asarray(test_cache["gids"], dtype=np.int64),
            logits=test_logits,
            probs=torch.softmax(torch.as_tensor(test_logits), dim=1).numpy(),
            y_true=y_test,
        )
        selection_record["test_evaluated"] = True
        selection_record["test_evaluated_utc"] = C.utcnow()
        C.write_json(out_dir / "selection.json", selection_record)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dropout_observed = int(dropout_calls["n"])

    preprocessing_seconds = float(
        sum(prepared["caches"][s]["meta"]["build_seconds"] for s in prepared["caches"])
    )
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "dataset": dataset,
        "model": model_name,
        "model_label": C.MODEL_LABELS[model_name],
        "status": "smoke" if smoke else "completed",
        "smoke": bool(smoke),
        "seed": int(seed),
        "split_seed": C.SPLIT_SEED,
        "n_classes": n_classes,
        "n_train": int(n_train),
        "n_valid": int(n_valid),
        "n_test": int(prepared["split"]["test"]["n"]) if not smoke else None,
        "split": {
            "partition_sha256": prepared["split"]["partition_sha256"],
            "train_index_sha256": prepared["split"]["train"]["index_sha256"],
            "valid_index_sha256": prepared["split"]["valid"]["index_sha256"],
            "test_index_sha256": prepared["split"]["test"]["index_sha256"],
        },
        "feature_spec_hash": spec.hash(),
        "norm_fingerprint": prepared["norm_record"]["norm_fingerprint"],
        "determinism": determinism,
        "model_info": model_info,
        "parameters": int(params),
        "recipe": C.RECIPE,
        "epochs_run": int(len(curve)),
        "steps_per_epoch": int(steps_per_epoch),
        "total_updates": int(total_updates),
        "lr_range": [float(lr_min), float(lr_max)],
        "lr_mean": float(lr_sum / max(1, steps_per_epoch * epochs)),
        "grad_norm_max": float(grad_norm_max),
        "grad_clip_events": int(clip_events),
        "dropout_calls_observed": int(dropout_observed),
        "best_epoch": int(tracker.best_epoch),
        "best_valid_metrics": tracker.best_metrics,
        "tie_break_check": tie_check,
        "train_metrics_at_best": train_metrics,
        "valid_metrics_at_best": valid_metrics,
        "valid_recheck": {
            "accuracy_abs_diff": float(valid_recheck),
            "ce_abs_diff": float(valid_ce_recheck),
        },
        "test_metrics": test_metrics,
        "checkpoint": {
            "path": C.track_rel(checkpoint_path),
            "sha256": checkpoint_sha,
            "bytes": int(checkpoint_path.stat().st_size),
        },
        "curve": {"path": C.track_rel(curve_path), "sha256": curve_sha, "rows": len(curve)},
        "predictions": {
            split: C.track_rel(out_dir / f"{split}_predictions.npz")
            for split in (("train", "valid") if smoke else ("train", "valid", "test"))
        },
        "timing": {
            "preprocessing_seconds": preprocessing_seconds,
            "training_and_eval_seconds": float(wall_train),
            "train_forward_backward_seconds": float(train_seconds_total),
            "epoch_valid_eval_seconds": float(valid_seconds_total),
            "test_eval_seconds": float(test_seconds),
            "process_seconds": float(time.perf_counter() - PROCESS_T0),
        },
        "memory": {
            "peak_cuda_allocated_bytes": int(gpu_peak_alloc),
            "peak_cuda_reserved_bytes": int(gpu_peak_reserved),
            "device": str(device),
        },
        "dataset_stats": prepared["split"],
        "cache_build_seconds": {
            s: float(prepared["caches"][s]["meta"]["build_seconds"]) for s in prepared["caches"]
        },
        "environment": C.environment(),
        "code_state": C.code_state(),
        "recorded_utc": C.utcnow(),
    }
    C.write_json(run_json, payload)
    return payload


def _subset_cache(cache: Mapping[str, Any], rows: np.ndarray, spec) -> dict[str, Any]:
    """Rebuild a cache restricted to leading rows (smoke only, no relabelling)."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0 or rows.min() < 0 or rows.max() >= int(cache["n_molecules"]):
        raise ValueError("smoke subset rows outside cache")
    mols = []
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    for row in rows.tolist():
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n0, n1 = int(nbr_ptr[a0]), int(nbr_ptr[a1])
        p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
        mols.append(
            {
                "n_atoms": np.int64(a1 - a0),
                "n_bonds": np.int64(b1 - b0),
                "atom_feat": np.asarray(cache["atom_feat"])[a0:a1],
                "atom_type": np.asarray(cache["atom_type"])[a0:a1],
                "atom_scalar": np.asarray(cache["atom_scalar"])[a0:a1],
                "bond_feat": np.asarray(cache["bond_feat"])[b0:b1],
                "bond_type": np.asarray(cache["bond_type"])[b0:b1],
                "bond_index": np.asarray(cache["bond_index"])[b0:b1],
                "nbr_ptr": nbr_ptr[a0 : a1 + 1] - n0,
                "nbr_code": np.asarray(cache["nbr_code"])[n0:n1],
                "path_ptr": path_ptr[a0 : a1 + 1] - p0,
                "path_cat": np.asarray(cache["path_cat"])[p0:p1],
                "path_end63": np.asarray(cache["path_end63"])[p0:p1],
                "path_end": np.asarray(cache["path_end"])[p0:p1],
                "path_closure": np.asarray(cache["path_closure"])[p0:p1],
                "global": np.asarray(cache["global"])[row],
                "n_components": np.asarray(cache["n_components"])[row],
                "n_junctions": np.asarray(cache["n_junctions"])[row],
                "n_ring_systems": np.asarray(cache["n_ring_systems"])[row],
                "molecule_id": str(cache["molecule_ids"][row]),
                "smiles": "",
            }
        )
    out = PREP_DATA.concat_molecules(mols, spec, split=f"smoke_{cache['split']}")
    out["gids"] = np.asarray(cache["gids"])[rows]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(C.DATASETS))
    parser.add_argument("--model", required=True, choices=list(C.MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=C.TRAINING_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out) if args.out else C.RUNS / f"{args.dataset}_{args.model}_s{args.seed}"
    payload = train_run(
        dataset=args.dataset,
        model_name=args.model,
        seed=int(args.seed),
        out_dir=out_dir,
        device=torch.device(args.device),
        smoke=bool(args.smoke),
        epochs=args.epochs,
        force=bool(args.force),
    )
    print(C.stable_json({k: payload[k] for k in ("dataset", "model", "status", "best_epoch", "best_valid_metrics")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
