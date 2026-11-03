"""One formal molecule training unit for one arm and seed.

Mirrors ``ptb_train_mol.train_unit`` step for step (same order of seed calls, same
optimiser/scheduler/generator construction, same single dropout, same clipping,
same 100 epochs, same Validation tracker and tie-breaks).  The only additions are

* a per-batch ``si_feat`` row block for the wide arms (A0 never reads it), and
* extra recorded provenance (branch norms, branch contribution, reload check).

The Test split is never loaded here: ``ptb_data.molecule_view`` returns
Train+Validation only, and this module never imports ``load_test_cache``.
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

import si_common as C
import si_features as F32
import si_model as M

import lcrc_data as PREP_DATA
import mb2_metrics as MET
import ptb_data as PD
import ptb_train_mol as PT


def batch_with_features(cache: Mapping[str, Any], rows: Sequence[int], feats: np.ndarray | None, device):
    batch = PREP_DATA.collate(cache, rows)
    if feats is not None:
        block = np.asarray(feats[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
        batch["si_feat"] = torch.from_numpy(np.ascontiguousarray(block))
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(model, cache: Mapping[str, Any], feats: np.ndarray | None, device, *, batch_size: int = 256) -> np.ndarray:
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = batch_with_features(cache, rows, feats, device)
            pred = model(batch)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            out.append(pred.detach().float().cpu().numpy())
    mat = np.concatenate(out, axis=0).astype(np.float64)
    if mat.shape[0] != n:
        raise RuntimeError(f"prediction rows {mat.shape[0]} != {n}")
    return mat


def train_unit(
    model_kind: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    *,
    feats_train: np.ndarray | None,
    feats_valid: np.ndarray | None,
    feature_meta: Mapping[str, Any],
    epochs: int | None = None,
    smoke: bool = False,
    force: bool = False,
    subset: bool = True,
    tag: str = "",
) -> dict[str, Any]:
    if model_kind not in C.NEURAL:
        raise KeyError(model_kind)
    formal_epochs = int(C.EPOCHS)
    if smoke:
        epochs = int(epochs if epochs is not None else 2)
    else:
        if epochs is None:
            epochs = formal_epochs
        if int(epochs) != formal_epochs:
            raise ValueError(f"{C.DATASET} formal runs must use {formal_epochs} epochs, got {epochs}")
        if int(seed) not in C.SEEDS:
            raise ValueError(f"formal molecule seeds are {C.SEEDS}")
    out_dir = Path(out_dir)
    run_json = out_dir / "run.json"
    if run_json.exists() and not force:
        raise RuntimeError(f"{C.track_rel(run_json)} exists; refusing to overwrite a recorded unit")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_process = time.perf_counter()
    view = PD.molecule_view(C.DATASET)
    spec = view.spec
    task = view.task
    if task != C.TASK:
        raise RuntimeError(f"unexpected task {task}")
    train_cache = view.train_cache
    valid_cache = view.valid_cache
    y_train_raw = np.asarray(view.targets.y_train, dtype=np.float64)
    y_valid_raw = np.asarray(view.targets.y_valid, dtype=np.float64)
    rows_tr = list(range(int(train_cache["n_molecules"])))
    rows_va = list(range(int(valid_cache["n_molecules"])))
    if smoke and subset:
        rows_tr = rows_tr[: min(64, len(rows_tr))]
        rows_va = rows_va[: min(32, len(rows_va))]
        train_cache = PT._subset_cache(train_cache, rows_tr, spec)
        valid_cache = PT._subset_cache(valid_cache, rows_va, spec)
        y_train_raw = y_train_raw[rows_tr]
        y_valid_raw = y_valid_raw[rows_va]
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])
    tr_feats = None if feats_train is None else np.asarray(feats_train, dtype=np.float32)[rows_tr]
    va_feats = None if feats_valid is None else np.asarray(feats_valid, dtype=np.float32)[rows_va]

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model, model_info = M.build_model(model_kind, spec, view.train_norm, int(seed), device)
    params = int(sum(p.numel() for p in model.parameters()))
    if model_kind == C.MODEL_A0:
        if params != C.PC.EXPECTED_PARAMS_SINGLE[C.MODEL_A0]:
            raise RuntimeError(f"A0 parameter count {params} != frozen {C.PC.EXPECTED_PARAMS_SINGLE[C.MODEL_A0]}")
    else:
        expected = C.PC.EXPECTED_PARAMS_SINGLE[C.MODEL_A0] + int(C.BRANCH_DIM[model_kind])
        if params != expected:
            raise RuntimeError(f"{model_kind} parameter count {params} != {expected}")

    recipe = dict(C.PC.RECIPE)
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

    y_loss = view.targets.loss_reference(view.target_norm)
    y_train_loss = torch.as_tensor(np.asarray(y_loss, dtype=np.float64).reshape(-1), dtype=torch.float32)[rows_tr]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    tracker = PT.Tracker(task)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    train_seconds_total = 0.0
    valid_seconds_total = 0.0

    t_train = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        perm = torch.randperm(n_train, generator=generator).tolist()
        for rows in (perm[i : i + batch_size] for i in range(0, len(perm), batch_size)):
            batch = batch_with_features(train_cache, rows, tr_feats, device)
            y = y_train_loss[rows].to(device)
            pred = model(batch)
            loss = F.binary_cross_entropy_with_logits(pred.reshape(-1), y.reshape(-1))
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
        valid_raw = predict(model, valid_cache, va_feats, device)
        valid_metrics = MET.task_metrics(task, valid_raw, y_valid_raw, view.target_norm, view.task_names)
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
                "valid_roc_auc": float(valid_metrics["roc_auc"]),
                "valid_pr_auc": float(valid_metrics["pr_auc"]),
                "valid_bce": float(valid_metrics["bce"]),
            }
        )
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        if smoke or epoch == epochs or improved:
            C.log(
                f"[{C.DATASET}/{model_kind}/s{seed}] epoch {epoch}/{epochs} "
                f"valid roc_auc={float(valid_metrics['roc_auc']):.6f} {'*best*' if improved else ''}"
            )
    train_wall = time.perf_counter() - t_train

    if tracker.best_epoch is None:
        raise RuntimeError("no validation improvement recorded")
    best_state = tracker.state
    best_state_hash = M.state_fingerprint(best_state)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    valid_raw = predict(model, valid_cache, va_feats, device)
    valid_metrics = MET.task_metrics(task, valid_raw, y_valid_raw, view.target_norm, view.task_names)
    if abs(float(valid_metrics["roc_auc"]) - float(tracker.best_metrics["roc_auc"])) > 1e-5:
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
    train_raw = predict(model, train_cache, tr_feats, device)
    train_metrics = MET.task_metrics(task, train_raw, y_train_raw, view.target_norm, view.task_names)
    np.savez(
        out_dir / "train_predictions.npz",
        rows=np.asarray(train_cache["rows"], dtype=np.int64),
        pred=train_raw,
        y_true=y_train_raw,
    )

    branch = M.branch_norms(model)
    contrib = {"train": None, "valid": None}
    if branch is not None:
        contrib["valid"] = M.contribution_stats(model.branch_contribution(va_feats, device))
        contrib["train"] = M.contribution_stats(model.branch_contribution(tr_feats, device))

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "round_id": C.ROUND_ID,
            "dataset": C.DATASET,
            "arm": model_kind,
            "n_tasks": C.N_TASKS,
            "seed": int(seed),
            "epochs_run": int(epochs),
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "roc_auc",
            "best_metric_value": float(tracker.best_primary),
            "selection_rule": tracker.rule(),
            "feature_spec_hash": spec.hash(),
            "norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "target_norm": view.target_norm,
            "init_state_hash": model_info["init_state_hash"],
            "parameters": int(params),
            "branch_dim": int(model_info["branch_dim"]),
            "feature_meta": dict(feature_meta),
            "recipe": recipe,
            "smoke": bool(smoke),
        },
        checkpoint_path,
    )
    checkpoint_sha = C.sha256_file(checkpoint_path)

    # gate 13: a fresh rebuild from the saved file reproduces the Validation row
    reloaded = M.load_model(
        model_kind, spec, view.train_norm, torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"], device
    )
    reload_raw = predict(reloaded, valid_cache, va_feats, device)
    reload_max_abs = float(np.max(np.abs(reload_raw - valid_raw)))

    selection = {
        "dataset": C.DATASET,
        "arm": model_kind,
        "seed": int(seed),
        "best_epoch": int(tracker.best_epoch),
        "best_metric_name": "roc_auc",
        "best_metric_value": float(tracker.best_primary),
        "best_metrics": tracker.best_metrics,
        "selection_rule": tracker.rule(),
        "checkpoint_path": C.track_rel(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "curve_sha256": C.sha256_file(curve_path),
        "test_evaluated": False,
        "reload_check": {
            "max_abs_valid_logit_diff": reload_max_abs,
            "tolerance": 1e-5,
            "pass": bool(reload_max_abs <= 1e-5),
        },
        "selection_finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "selection.json", selection)

    gap = {
        "train_roc_auc": float(train_metrics["roc_auc"]),
        "valid_roc_auc": float(valid_metrics["roc_auc"]),
        "train_minus_valid_roc_auc": float(train_metrics["roc_auc"] - valid_metrics["roc_auc"]),
        "train_bce": float(train_metrics["bce"]),
        "valid_bce": float(valid_metrics["bce"]),
        "train_pr_auc": float(train_metrics["pr_auc"]),
        "valid_pr_auc": float(valid_metrics["pr_auc"]),
    }
    tail = [float(r["valid_roc_auc"]) for r in curve if 90 <= int(r["epoch"]) <= 100]

    record = {
        "schema": "lcrc-sparse-identity-run-v1",
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "env": C.env_info(),
        "dataset": C.DATASET,
        "task": task,
        "arm": model_kind,
        "seed": int(seed),
        "smoke": bool(smoke),
        "tag": str(tag),
        "epochs_run": int(epochs),
        "n_train": int(n_train),
        "n_valid": int(n_valid),
        "parameters": int(params),
        "model": model_info,
        "recipe": recipe,
        "feature_meta": dict(feature_meta),
        "data": {
            "feature_spec_hash": spec.hash(),
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "source": view.source_meta,
            "rows": {k: {"n": len(v), "sha256": C.index_sha256(v)} for k, v in view.rows.items()},
            "test_labels_read": False,
            "test_cache_built": False,
        },
        "validation": {"metrics": valid_metrics},
        "train": {"metrics": train_metrics},
        "gap": gap,
        "epoch_90_100_valid_roc_auc": {
            "values": tail,
            "min": float(min(tail)) if tail else None,
            "max": float(max(tail)) if tail else None,
            "last": float(tail[-1]) if tail else None,
        },
        "branch": {"norms": branch, "contribution": contrib},
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_state_hash": best_state_hash,
            "sha256": checkpoint_sha,
            "path": C.track_rel(checkpoint_path),
            "n_improvements": int(len(tracker.updates)),
            "improvement_epochs": [int(x) for x in tracker.updates],
            "reload_max_abs_valid_logit_diff": reload_max_abs,
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
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
        "result_label": "ENGINEERING_SMOKE" if smoke else C.RESULT_LABEL,
    }
    C.write_json(run_json, record)
    C.log(
        f"[{C.DATASET}/{model_kind}/s{seed}] done best_epoch={tracker.best_epoch} "
        f"roc_auc={float(tracker.best_primary):.6f} wall={train_wall/60:.1f}min gpu_h={train_wall/3600:.4f}"
    )
    return record


def load_arm_features(arm: str) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Rebuild the frozen branch input matrices for Train and Validation."""
    spec = F32.frozen_spec()
    vocab = C.read_json(C.ANALYSIS / "feature_vocabulary.json")
    if vocab.get("status") != "OK":
        raise RuntimeError("feature vocabulary is not feasible/frozen")
    view = PD.molecule_view(C.DATASET)
    feats: dict[str, Any] = {}
    for split, cache in (("train", view.train_cache), ("valid", view.valid_cache)):
        feats[split] = F32.encode_split(cache, spec, vocab)
    fold = F32.fold_buckets(vocab)
    if arm == C.MODEL_A0:
        return None, None, {"arm": arm}
    if arm == C.MODEL_NULL:
        matrices = {}
        stats = {}
        for split in ("train", "valid"):
            null_mat, null_stats = F32.double_edge_swap(feats[split]["binary"], C.NULL_SEED[split])
            scale = (1.0 / np.sqrt(np.maximum(1, null_mat.sum(axis=1)).astype(np.float64))).astype(np.float32)
            matrices[split] = (null_mat.astype(np.float32) * scale[:, None]).astype(np.float32)
            stats[split] = null_stats
        meta = {
            "arm": arm,
            "vocabulary_sha256": vocab["vocabulary_sha256"],
            "null": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} for k, v in stats.items()},
            "null_seeds": {k: int(C.NULL_SEED[k]) for k in ("train", "valid")},
        }
        return matrices["train"], matrices["valid"], meta
    x_train = F32.branch_input(arm, "train", feats, fold)
    x_valid = F32.branch_input(arm, "valid", feats, fold)
    meta = {
        "arm": arm,
        "vocabulary_sha256": vocab["vocabulary_sha256"],
        "branch_dim": int(x_train.shape[1]),
        "branch_matrix_sha256": {"train": C.array_hash_f32(x_train), "valid": C.array_hash_f32(x_valid)},
        "fold_assignment_sha256": fold["assignment_sha256"] if arm == C.MODEL_FOLD else None,
    }
    return x_train, x_valid, meta


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One sparse-identity training unit")
    ap.add_argument("--model", required=True, choices=list(C.NEURAL))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--subset", action="store_true", default=False)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else (
        C.smoke_dir(args.model, args.seed, args.tag or "smoke")
        if args.smoke
        else C.run_dir(args.model, args.seed)
    )
    feats_train, feats_valid, meta = load_arm_features(args.model)
    record = train_unit(
        args.model,
        int(args.seed),
        out_dir,
        torch.device(args.device),
        feats_train=feats_train,
        feats_valid=feats_valid,
        feature_meta=meta,
        epochs=args.epochs,
        smoke=bool(args.smoke),
        force=bool(args.force),
        subset=bool(args.subset),
        tag=args.tag,
    )
    print(C.stable_json({k: record[k] for k in ("dataset", "arm", "seed", "checkpoint")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
