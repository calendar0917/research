"""One formal molecule training unit: ``(dataset, model, seed)``.

Frozen R3 recipe (AdamW, peak LR 3e-4 per-step cosine to 1e-5, wd 1e-3,
batch 128, dropout 0.10 on the shared local representation ``h``, grad clip 5,
full epochs with no early stopping, no EMA/SWA/soup/SSL).  ``ogbg-molhiv`` runs
the identical recipe with a 100-epoch cap; the cap is never extended.

Checkpoint selection uses the task's primary Validation metric only:
* regression -> lowest Valid RMSE (strict improvement, tie keeps the earlier epoch)
* binary     -> highest Valid ROC-AUC (tie: lower Valid BCE, then earlier epoch)
* multitask  -> highest Valid macro ROC-AUC (tie: lower masked BCE, then earlier)

The Test split is *not* touched by this script.  Test scoring happens in
``ptb_score.py`` only after the whole dataset's checkpoint manifest is frozen.
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
import ptb_model as M

import lcrc_data as PREP_DATA
import mb2_metrics as MET


# ---------------------------------------------------------------------------
# validation-only checkpoint selection
# ---------------------------------------------------------------------------


class Tracker:
    """Primary-Validation-metric tracker with the frozen tie-break rules."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.best_epoch: int | None = None
        self.best_primary: float | None = None
        self.best_secondary: float | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.updates: list[int] = []
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, Any]) -> bool:
        primary = float(MET.primary_value(self.task, metrics))
        secondary_key = {"regression": None, "binary": "bce", "multitask": "masked_bce"}[self.task]
        secondary = float(metrics[secondary_key]) if secondary_key else float("nan")
        if self.best_epoch is None:
            better = True
        elif primary == self.best_primary:
            if self.task == "regression":
                better = False  # strict improvement only -> keep the earlier epoch
            else:
                better = secondary < float(self.best_secondary)
        else:
            better = MET.better(self.task, primary, self.best_primary)
        if not better:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = float(primary)
        self.best_secondary = secondary
        self.best_metrics = dict(metrics)
        self.updates.append(int(epoch))
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("tracker captured no state")
        return self._state

    def rule(self) -> str:
        return {
            "regression": "lowest_validation_rmse__strict_improvement__tie_keeps_earlier_epoch",
            "binary": "highest_validation_roc_auc__tie_lower_validation_bce__then_earlier_epoch",
            "multitask": "highest_validation_macro_roc_auc__tie_lower_masked_bce__then_earlier_epoch",
        }[self.task]


def _subset_cache(cache: Mapping[str, Any], rows: Sequence[int], spec) -> dict[str, Any]:
    """Rebuild a cache containing only ``rows`` (smoke only)."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    mols = []
    for g in rows:
        g = int(g)
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        nbr_s = int(nbr_ptr[a0])
        path_s = int(path_ptr[a0])
        mols.append(
            {
                "n_atoms": np.int64(a1 - a0),
                "n_bonds": np.int64(b1 - b0),
                "atom_feat": np.asarray(cache["atom_feat"][a0:a1]),
                "atom_type": np.asarray(cache["atom_type"][a0:a1]),
                "atom_scalar": np.asarray(cache["atom_scalar"][a0:a1]),
                "bond_feat": np.asarray(cache["bond_feat"][b0:b1]),
                "bond_type": np.asarray(cache["bond_type"][b0:b1]),
                "bond_index": np.asarray(cache["bond_index"][b0:b1]),
                "nbr_ptr": np.asarray(nbr_ptr[a0 : a1 + 1]) - nbr_s,
                "nbr_code": np.asarray(cache["nbr_code"][nbr_s : int(nbr_ptr[a1])]),
                "path_ptr": np.asarray(path_ptr[a0 : a1 + 1]) - path_s,
                "path_cat": np.asarray(cache["path_cat"][path_s : int(path_ptr[a1])]),
                "path_end63": np.asarray(cache["path_end63"][path_s : int(path_ptr[a1])]),
                "path_end": np.asarray(cache["path_end"][path_s : int(path_ptr[a1])]),
                "path_closure": np.asarray(cache["path_closure"][path_s : int(path_ptr[a1])]),
                "global": np.asarray(cache["global"][g]),
                "n_components": np.int64(np.asarray(cache["n_components"])[g]),
                "n_junctions": np.int64(np.asarray(cache["n_junctions"])[g]),
                "n_ring_systems": np.int64(np.asarray(cache["n_ring_systems"])[g]),
                "molecule_id": str(cache["molecule_ids"][g]),
                "smiles": str(cache["smiles"][g]) if "smiles" in cache else "",
            }
        )
    out = PREP_DATA.concat_molecules(mols, spec, split="subset")
    out["rows"] = np.asarray([int(np.asarray(cache["rows"])[g]) for g in rows], dtype=np.int64)
    out["meta"]["dataset"] = cache["meta"].get("dataset", "")
    return out


def train_unit(
    dataset: str,
    model_variant: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    *,
    epochs: int | None = None,
    smoke: bool = False,
    force: bool = False,
    subset: bool = True,
) -> dict[str, Any]:
    if dataset not in C.MOLECULE_DATASETS:
        raise KeyError(dataset)
    meta = C.MOLECULE_DATASETS[dataset]
    formal_epochs = int(meta["epochs"])
    if smoke:
        epochs = int(epochs if epochs is not None else 2)
    else:
        if epochs is None:
            epochs = formal_epochs
        if int(epochs) != formal_epochs:
            raise ValueError(f"{dataset} formal runs must use {formal_epochs} epochs, got {epochs}")
        if int(seed) not in C.MOLECULE_SEEDS:
            raise ValueError(f"formal molecule seeds are {C.MOLECULE_SEEDS}")
    out_dir = Path(out_dir)
    run_json = out_dir / "run.json"
    if run_json.exists() and not force:
        raise RuntimeError(f"{C.track_rel(run_json)} exists; refusing to overwrite a recorded unit")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_process = time.perf_counter()
    view = D.molecule_view(dataset)
    spec = view.spec
    task = view.task
    n_tasks = int(view.targets.n_tasks)
    train_cache = view.train_cache
    valid_cache = view.valid_cache
    y_train_raw = np.asarray(view.targets.y_train, dtype=np.float64)
    y_valid_raw = np.asarray(view.targets.y_valid, dtype=np.float64)
    if smoke and subset:
        rows_tr = list(range(min(64 * max(n_tasks, 1), int(train_cache["n_molecules"]))))
        rows_va = list(range(min(32 * max(n_tasks, 1), int(valid_cache["n_molecules"]))))
        train_cache = _subset_cache(train_cache, rows_tr, spec)
        valid_cache = _subset_cache(valid_cache, rows_va, spec)
        y_train_raw = y_train_raw[rows_tr]
        y_valid_raw = y_valid_raw[rows_va]
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model, model_info = M.build_model(spec, view.train_norm, model_variant, int(seed), device, n_tasks=n_tasks)
    params = M.parameter_count(model)

    recipe = dict(C.RECIPE)
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
    generator = torch.Generator().manual_seed(int(seed) + int(recipe["shuffle_offset"]))

    y_train_loss_ref = view.targets.loss_reference(view.target_norm)
    if task == "regression":
        y_train_loss = torch.as_tensor(np.asarray(y_train_loss_ref, dtype=np.float64).reshape(-1), dtype=torch.float32)
    else:
        y_train_loss = torch.as_tensor(np.asarray(y_train_loss_ref, dtype=np.float64), dtype=torch.float32)
    if smoke and subset:
        y_train_loss = y_train_loss[rows_tr]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    tracker = Tracker(task)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    train_seconds_total = 0.0
    valid_seconds_total = 0.0
    skipped_batches = 0

    t_train = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        perm = torch.randperm(n_train, generator=generator).tolist()
        for rows in (perm[i : i + batch_size] for i in range(0, len(perm), batch_size)):
            batch = MD_to_device(PREP_DATA.collate(train_cache, rows), device)
            y = y_train_loss[rows].to(device)
            pred = model(batch)
            if task == "regression":
                loss = F.mse_loss(pred.reshape(-1), y.reshape(-1))
            elif task == "binary":
                loss = F.binary_cross_entropy_with_logits(pred.reshape(-1), y.reshape(-1))
            else:
                mask = torch.isfinite(y)
                if int(mask.sum()) == 0:
                    skipped_batches += 1
                    continue
                per_label = F.binary_cross_entropy_with_logits(pred, torch.nan_to_num(y), reduction="none")
                loss = (per_label * mask).sum() / mask.sum().clamp(min=1.0)
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
        valid_raw = predict_matrix(model, valid_cache, device)
        valid_metrics = MET.task_metrics(task, valid_raw, y_valid_raw, view.target_norm, view.task_names)
        valid_seconds = time.perf_counter() - t_eval
        valid_seconds_total += valid_seconds
        improved = tracker.consider(epoch, valid_metrics)
        if improved:
            tracker.capture(model)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_clip_events": int(clip_events),
            "train_seconds": float(train_seconds),
            "valid_seconds": float(valid_seconds),
            "is_best": bool(improved),
        }
        if task == "regression":
            row["valid_rmse_raw"] = float(valid_metrics["rmse"])
            row["valid_mae_raw"] = float(valid_metrics["mae"])
            row["valid_r2_raw"] = float(valid_metrics["r2"])
        elif task == "binary":
            row["valid_roc_auc"] = float(valid_metrics["roc_auc"])
            row["valid_pr_auc"] = float(valid_metrics["pr_auc"])
            row["valid_bce"] = float(valid_metrics["bce"])
        else:
            row["valid_macro_roc_auc"] = float(valid_metrics["macro_roc_auc"])
            row["valid_macro_pr_auc"] = float(valid_metrics["macro_pr_auc"])
            row["valid_masked_bce"] = float(valid_metrics["masked_bce"])
            row["valid_definable_tasks"] = int(valid_metrics["n_definable_tasks"])
        curve.append(row)
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        if smoke or epoch == epochs or improved:
            key = MET.primary_name(task)
            C.log(
                f"[{dataset}/{model_variant}/s{seed}] epoch {epoch}/{epochs} "
                f"valid {key}={float(valid_metrics[key]):.6f} {'*best*' if improved else ''}"
            )
    train_wall = time.perf_counter() - t_train

    if tracker.best_epoch is None:
        raise RuntimeError("no validation improvement recorded")
    best_state = tracker.state
    best_state_hash = M.state_fingerprint(best_state)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    valid_raw = predict_matrix(model, valid_cache, device)
    valid_metrics = MET.task_metrics(task, valid_raw, y_valid_raw, view.target_norm, view.task_names)
    key = MET.primary_name(task)
    if abs(float(valid_metrics[key]) - float(tracker.best_metrics[key])) > 1e-5:
        raise RuntimeError("reloaded best checkpoint does not reproduce the recorded Validation metric")

    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    np.savez(
        out_dir / "valid_predictions.npz",
        rows=np.asarray(valid_cache["rows"], dtype=np.int64),
        pred=valid_raw,
        y_true=y_valid_raw,
    )
    train_raw = predict_matrix(model, train_cache, device)
    train_metrics = MET.task_metrics(task, train_raw, y_train_raw, view.target_norm, view.task_names)
    np.savez(
        out_dir / "train_predictions.npz",
        rows=np.asarray(train_cache["rows"], dtype=np.int64),
        pred=train_raw,
        y_true=y_train_raw,
    )

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "round_id": C.ROUND_ID,
            "dataset": dataset,
            "variant": model_variant,
            "n_tasks": n_tasks,
            "seed": int(seed),
            "epochs_run": int(epochs),
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": key,
            "best_metric_value": float(tracker.best_primary),
            "selection_rule": tracker.rule(),
            "feature_spec_hash": spec.hash(),
            "norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "target_norm": view.target_norm,
            "init_state_hash": model_info["init_state_hash"],
            "parameters": int(params),
            "recipe": recipe,
            "smoke": bool(smoke),
        },
        checkpoint_path,
    )
    checkpoint_sha = C.sha256_file(checkpoint_path)
    selection = {
        "dataset": dataset,
        "variant": model_variant,
        "seed": int(seed),
        "best_epoch": int(tracker.best_epoch),
        "best_metric_name": key,
        "best_metric_value": float(tracker.best_primary),
        "best_metrics": tracker.best_metrics,
        "selection_rule": tracker.rule(),
        "checkpoint_path": C.track_rel(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "curve_sha256": C.sha256_file(curve_path),
        "test_evaluated": False,
        "selection_finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "selection.json", selection)

    epoch90_100 = [r for r in curve if 90 <= int(r["epoch"]) <= 100]
    tail_key = {"regression": "valid_rmse_raw", "binary": "valid_roc_auc", "multitask": "valid_macro_roc_auc"}[task]
    tail = [float(r[tail_key]) for r in epoch90_100]
    gap = {}
    if task == "regression":
        gap = {
            "train_rmse_raw": float(train_metrics["rmse"]),
            "valid_rmse_raw": float(valid_metrics["rmse"]),
            "train_minus_valid_rmse": float(train_metrics["rmse"] - valid_metrics["rmse"]),
        }
    elif task == "binary":
        gap = {
            "train_roc_auc": float(train_metrics["roc_auc"]),
            "valid_roc_auc": float(valid_metrics["roc_auc"]),
            "train_minus_valid_roc_auc": float(train_metrics["roc_auc"] - valid_metrics["roc_auc"]),
            "train_bce": float(train_metrics["bce"]),
            "valid_bce": float(valid_metrics["bce"]),
        }
    else:
        gap = {
            "train_macro_roc_auc": float(train_metrics["macro_roc_auc"]),
            "valid_macro_roc_auc": float(valid_metrics["macro_roc_auc"]),
            "train_minus_valid_macro_roc_auc": float(
                train_metrics["macro_roc_auc"] - valid_metrics["macro_roc_auc"]
            ),
            "train_masked_bce": float(train_metrics["masked_bce"]),
            "valid_masked_bce": float(valid_metrics["masked_bce"]),
        }

    record = {
        "schema": "lcrc-ptb-molecule-run-v1",
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "env": C.env_info(),
        "dataset": dataset,
        "task": task,
        "variant": model_variant,
        "model_label": C.MODEL_LABELS[model_variant],
        "seed": int(seed),
        "smoke": bool(smoke),
        "epochs_run": int(epochs),
        "epoch_cap": int(C.MOLECULE_DATASETS[dataset]["epochs"]),
        "budget_limited_possible": bool(int(tracker.best_epoch) >= 90 and dataset == "ogbg-molhiv"),
        "n_train": int(n_train),
        "n_valid": int(n_valid),
        "n_tasks": n_tasks,
        "task_names": list(view.task_names),
        "parameters": int(params),
        "model": model_info,
        "recipe": recipe,
        "data": {
            "feature_spec_hash": spec.hash(),
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "target_norm": view.target_norm,
            "source": view.source_meta,
            "rows": {k: {"n": len(v), "sha256": C.index_sha256(v)} for k, v in view.rows.items()},
            "test_labels_read": False,
            "test_cache_built": False,
        },
        "validation": {"metrics": valid_metrics},
        "train": {"metrics": train_metrics},
        "gap": gap,
        "epoch_90_100_valid": {
            "metric": tail_key,
            "values": tail,
            "min": float(min(tail)) if tail else None,
            "max": float(max(tail)) if tail else None,
            "last": float(tail[-1]) if tail else None,
        },
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_state_hash": best_state_hash,
            "sha256": checkpoint_sha,
            "path": C.track_rel(checkpoint_path),
            "n_improvements": int(len(tracker.updates)),
            "improvement_epochs": [int(x) for x in tracker.updates],
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "skipped_batches_no_observed_label": int(skipped_batches),
            "nonfinite_events": 0,
        },
        "timing": {
            "process_wall_seconds": float(time.perf_counter() - t_process),
            "train_wall_seconds": float(train_wall),
            "train_seconds_total": float(train_seconds_total),
            "valid_seconds_total": float(valid_seconds_total),
            "mean_epoch_seconds": float((train_seconds_total + valid_seconds_total) / max(epochs, 1)),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "gpu_hours_estimate": float(train_wall / 3600.0),
        },
        "curve": curve,
        "result_label": "ENGINEERING_SMOKE" if smoke else C.RESULT_LABEL_MOL,
    }
    C.write_json(run_json, record)
    C.log(
        f"[{dataset}/{model_variant}/s{seed}] done best_epoch={tracker.best_epoch} "
        f"{key}={float(tracker.best_primary):.6f} wall={train_wall/60:.1f}min gpu_h={train_wall/3600:.4f}"
    )
    return record


def predict_matrix(model, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = MD_to_device(PREP_DATA.collate(cache, rows), device)
            pred = model(batch)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            out.append(pred.detach().float().cpu().numpy())
    mat = np.concatenate(out, axis=0).astype(np.float64)
    if mat.shape[0] != n:
        raise RuntimeError(f"prediction rows {mat.shape[0]} != {n}")
    return mat


def MD_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One molecule training unit (A0 or C1)")
    ap.add_argument("--dataset", required=True, choices=list(C.MOLECULE_ORDER))
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cost-probe", action="store_true", help="timed smoke on the full training set")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else (
        C.RUNS / "smoke" / C.mol_run_name(args.dataset, args.model, args.seed)
        if args.smoke
        else C.mol_run_dir(args.dataset, args.model, args.seed)
    )
    record = train_unit(
        args.dataset,
        args.model,
        int(args.seed),
        out_dir,
        torch.device(args.device),
        epochs=args.epochs,
        smoke=bool(args.smoke),
        force=bool(args.force),
        subset=not bool(args.cost_probe),
    )
    print(C.stable_json({k: record[k] for k in ("dataset", "variant", "seed", "checkpoint")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
