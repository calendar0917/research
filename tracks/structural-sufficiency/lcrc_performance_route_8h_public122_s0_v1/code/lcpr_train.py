"""One fixed training unit for any candidate x dataset.

Common discipline (all arms, all datasets):

* seed 0 only; the batch order is a frozen permutation table (seed 917000), so
  the training shuffle never consumes the model RNG;
* the C1 (or FULL) initial state comes from ``build_full_state(seed=0)`` +
  the frozen ``remap_state``; extension tensors use a private RNG stream and the
  model RNG is reset just before the first update;
* parameter updates use Train labels only; target normalisation, support tables
  and auxiliary targets are fitted on Train only;
* **every epoch** evaluates full Train, Valid and Test in ``eval()``/``no_grad``
  with the RNG state and the module mode saved and restored;
* the checkpoint rule is best Validation only; Test never selects an epoch.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import lcpr_common as C
import lcpr_data as D
import lcpr_model as M

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def regression_metrics(pred_raw: np.ndarray, y_raw: np.ndarray) -> dict[str, float]:
    e = np.asarray(pred_raw, np.float64) - np.asarray(y_raw, np.float64)
    ss = float(np.sum(e * e))
    tot = float(np.sum((np.asarray(y_raw, np.float64) - np.mean(y_raw)) ** 2))
    return {
        "n": int(e.size),
        "rmse": float(math.sqrt(ss / e.size)),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "r2": float(1.0 - ss / tot) if tot > 0 else float("nan"),
        "sse": ss,
    }


def binary_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, np.int64).reshape(-1)
    z = np.asarray(logits, np.float64).reshape(-1)
    p = 1.0 / (1.0 + np.exp(-z))
    bce = float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))
    out = {
        "n": int(y.size),
        "bce": bce,
        "log_loss": bce,
        "positive_rate": float(np.mean(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def split_metrics(task: str, pred_std: np.ndarray, y_raw: np.ndarray, tnorm: Mapping[str, Any]) -> dict[str, Any]:
    if task == "regression":
        pred_raw = pred_std * float(tnorm["scale"]) + float(tnorm["mean"])
        return regression_metrics(pred_raw, y_raw)
    return binary_metrics(pred_std, y_raw)


def primary_value(task: str, metrics: Mapping[str, Any]) -> float:
    return float(metrics["rmse"] if task == "regression" else metrics["roc_auc"])


def is_better(task: str, value: float, best: float | None) -> bool:
    if not math.isfinite(value):
        return False
    if best is None:
        return True
    return value < best if task == "regression" else value > best


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def rng_snapshot(device) -> dict[str, Any]:
    snap = {
        "python": None,
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state(device).clone()
    return snap


def rng_restore(snap: Mapping[str, Any], device) -> None:
    np.random.set_state(snap["numpy"])
    torch.set_rng_state(snap["torch_cpu"])
    if snap["torch_cuda"] is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(snap["torch_cuda"], device)


def rng_hash(device) -> str:
    h = C.sha256_bytes(torch.get_rng_state().numpy().tobytes())
    if device.type == "cuda" and torch.cuda.is_available():
        h += ":" + C.sha256_bytes(torch.cuda.get_rng_state(device).numpy().tobytes())
    return h


# ---------------------------------------------------------------------------
# evaluation (RNG- and mode-preserving)
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_cache(model, cache, device, batch_size: int = C.EVAL_BATCH_SIZE,
                  path_count: np.ndarray | None = None) -> np.ndarray:
    """Full-split prediction; the caller restores RNG/mode."""
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        rows = list(range(start, min(start + batch_size, n)))
        batch = D.to_device(D.collate(cache, rows, path_count=path_count), device)
        out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def evaluate_all(model, caches: Mapping[str, Any], device,
                 path_counts: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    was_training = bool(model.training)
    snap = rng_snapshot(device)
    model.eval()
    try:
        preds = {
            s: predict_cache(model, caches[s], device,
                             path_count=(None if path_counts is None else path_counts.get(s)))
            for s in C.SPLITS
        }
    finally:
        rng_restore(snap, device)
        model.train(was_training)
    return preds


# ---------------------------------------------------------------------------
# recipe / candidate resolution
# ---------------------------------------------------------------------------


def resolve_recipe(name: str, *, recipe_override: str | None = None, epochs_override: int | None = None) -> dict[str, Any]:
    if name == "R*":
        if not recipe_override:
            raise SystemExit("recipe 'R*' requires --recipe (frozen by Stage A first)")
        name = recipe_override
    recipe = dict(C.RECIPES[name])
    if not (name == "D1" or name == "D2"):
        recipe["epochs"] = int(C.BASE_RECIPE["epochs"] if epochs_override is None else epochs_override)
    return recipe


def resolve_epochs(cand: Mapping[str, Any], recipe: Mapping[str, Any], *, epochs_override: int | None) -> int:
    if epochs_override is not None:
        return int(epochs_override)
    if "epochs" in cand:
        return int(cand["epochs"])
    return int(recipe["epochs"])


# ---------------------------------------------------------------------------
# one training unit
# ---------------------------------------------------------------------------


def train_candidate(
    *,
    candidate: str,
    dataset: str,
    device: torch.device,
    out_dir: Path,
    recipe_name: str | None = None,
    epochs_override: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    if candidate not in C.CANDIDATES:
        raise SystemExit(f"unknown candidate {candidate!r}")
    cand = dict(C.CANDIDATES[candidate])
    kind = cand["kind"]
    task_meta = D.dataset_meta(dataset)
    task = task_meta["task"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "dmpnn":
        import lcpr_dmpnn as DMN

        return DMN.train_dmpnn(candidate=candidate, dataset=dataset, device=device,
                               out_dir=out_dir, smoke=smoke)

    recipe = resolve_recipe(cand["recipe"], recipe_override=recipe_name)
    epochs = resolve_epochs(cand, recipe, epochs_override=epochs_override)
    if smoke and epochs > 2:
        epochs = 2
    if not smoke and C.git_dirty():
        raise SystemExit("formal run requested on a dirty tracked worktree; commit first")

    dmpnn_recipe = cand["recipe"] in ("D1", "D2")
    spec = D.frozen_spec()

    # ---- data ----------------------------------------------------------
    tgt = D.targets(dataset, include_test=True)
    caches = {s: D.load_cache(dataset, s, tgt["rows"][s]) for s in C.SPLITS}
    tnorm = tgt["target_norm"]
    perms = D.load_perms(dataset, max(epochs, 1))
    n_train = int(task_meta["sizes"][0])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * int(epochs)

    train_norm = _train_structure_norm(caches["train"], spec)

    chem8 = None
    if M.extra_atom_cols(kind) == 8:
        chem8 = {s: D.chem8_atoms(dataset, s, caches[s]) for s in C.SPLITS}

    support = None
    if kind in M.SHRINK_KINDS:
        support = D.load_support(dataset)
        kappa = float(support["kappa"]) * float(cand.get("kappa_mult", 1.0))
        a_bar = float(support["a_bar_k4"] if float(cand.get("kappa_mult", 1.0)) == 4.0 else support["a_bar_k1"])

    #: evaluation caches carry exactly the same extra columns as training
    eval_caches: dict[str, Any] = {}
    for _s in C.SPLITS:
        _c = caches[_s]
        if chem8 is not None:
            _c = {**_c, "atom_feat": _concat_chem8(_c, chem8[_s])}
        eval_caches[_s] = _c
    path_counts = ({s: support["path_count"][s] for s in C.SPLITS} if support is not None else None)

    aux = None
    aux_train_raw = None
    if kind in M.AUX_KINDS:
        aux = D.load_aux_targets(dataset)
        aux_key = {"a_logp": "wc_logp", "a_mw": "molwt", "a_shuffle": "wc_logp_shuffled"}[kind]
        aux_norm = aux[f"{'wc_logp' if aux_key == 'wc_logp_shuffled' else aux_key}_norm"]
        aux_train_raw = np.asarray(aux[aux_key])[np.asarray(tgt["rows"]["train"], dtype=np.int64)]

    # ---- model ---------------------------------------------------------
    full_state, base_state = M.c1_reference_state_full(spec, train_norm)
    dropout = 0.0 if kind == "m_nodrop" else float(C.BASE_RECIPE["dropout"])
    variant = "A0" if kind in ("a0", "a0_combo") else C.C1_VARIANT
    model = M.build_model(spec, train_norm, kind, variant=variant, dropout=dropout)
    if variant == "A0":
        base_state = full_state
    init_report = M.load_init_state(model, base_state, kind)
    shared = M.shared_state_equal(model, base_state, kind)
    if variant == C.C1_VARIANT and kind not in ("wide_local",) and not shared["all_equal"]:
        raise RuntimeError("shared C1 state is not bitwise equal to the frozen remap")
    if kind == "a0" and int(model.parameter_count()) != C.A0_PARAMS:
        raise RuntimeError(f"A0 parameters {model.parameter_count()} != frozen {C.A0_PARAMS}")
    if kind in ("c1", "m_unbiased", "m_cleanpool", "m_nodrop") and dataset == "Lipophilicity":
        if int(model.parameter_count()) != C.C1_PARAMS:
            raise RuntimeError(f"C1 parameters {model.parameter_count()} != frozen {C.C1_PARAMS}")
    if support is not None:
        model.local.shrink_kappa.fill_(float(kappa))
        model.local.shrink_abar.fill_(float(a_bar))
    model.to(device)
    init_state_hash = C.DM.state_fingerprint({k: v.cpu() for k, v in model.state_dict().items()})

    torch.manual_seed(int(C.TRAIN_SEED))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(C.TRAIN_SEED))
    init_rng = rng_hash(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]), betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]), weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=float(recipe["min_lr"])
    )

    def _batch(split: str, rows: Sequence[int]):
        cache = caches[split]
        if chem8 is not None:
            cache = {**cache, "atom_feat": _concat_chem8(cache, chem8[split])}
        pc = support["path_count"][split] if support is not None else None
        return D.to_device(D.collate(cache, list(rows), path_count=pc), device)

    # ---- training loop -------------------------------------------------
    curve: list[dict[str, Any]] = []
    diag: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    last_state: dict[str, torch.Tensor] | None = None
    clip_events = 0
    gnorm_max = 0.0
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    train_start_rng = rng_hash(device)
    _probe = rng_hash(device)
    evaluate_all(model, eval_caches, device, path_counts)
    eval_probe = {"rng_unchanged_after_full_eval": bool(rng_hash(device) == _probe),
                  "mode_after_full_eval": "train" if model.training else "eval"}
    eval_rng_probe: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        perm = perms[epoch - 1]
        abs_s = sq_s = aux_s = 0.0
        seen = 0
        ep_gnorm = 0.0
        for step_index in range(steps_per_epoch):
            rows = perm[step_index * batch_size : (step_index + 1) * batch_size]
            global_step = (epoch - 1) * steps_per_epoch + step_index + 1
            batch = _batch("train", rows)
            out = model(batch, return_aux=True)
            pred = out["y"]
            if task == "regression":
                z = torch.as_tensor(
                    (tgt["y"]["train"][list(rows)] - float(tnorm["mean"])) / float(tnorm["scale"]),
                    dtype=torch.float32, device=device,
                )
                loss_main = F.mse_loss(pred, z)
                target = z
            else:
                y = torch.as_tensor(tgt["y"]["train"][list(rows)], dtype=torch.float32, device=device)
                loss_main = F.binary_cross_entropy_with_logits(pred, y)
                target = y

            loss_aux = None
            if aux_train_raw is not None:
                raw = torch.as_tensor(aux_train_raw[list(rows)], dtype=torch.float32, device=device)
                z_aux = (raw - float(aux_norm["mean"])) / float(aux_norm["scale"])
                loss_aux = F.mse_loss(out["aux_pred"], z_aux)
                aux_s += float(loss_aux.detach())
            total = loss_main if loss_aux is None else loss_main + 0.1 * loss_aux

            if global_step in (1, 100, 1000, total_steps):
                diag.append({
                    "step": int(global_step), "epoch": int(epoch),
                    **_grad_diagnostics(model, loss_main, loss_aux),
                })

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(total.detach())) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch} step {step_index + 1}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
            gnorm_max = max(gnorm_max, gnorm)
            ep_gnorm = max(ep_gnorm, gnorm)
            optimizer.step()
            scheduler.step()

            if task == "regression":
                e = pred.detach() - z
            else:
                e = pred.detach() - y
            abs_s += float(e.abs().sum())
            sq_s += float((e * e).sum())
            seen += int(e.numel())
        train_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = evaluate_all(model, eval_caches, device, path_counts)
        eval_seconds = time.perf_counter() - t1
        if epoch == 1:
            eval_rng_probe.append({
                "epoch": 1,
                "model_mode_after_eval": "train" if model.training else "eval",
                "prediction_sizes": {s: int(preds[s].size) for s in C.SPLITS},
            })
        metrics = {s: split_metrics(task, preds[s], tgt["y"][s], tnorm) for s in C.SPLITS}
        value = primary_value(task, metrics["valid"])
        improved = is_better(task, value, best_value)
        if improved:
            best_value = value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row = {
            "epoch": int(epoch),
            "steps": int(steps_per_epoch),
            "lr_end": float(optimizer.param_groups[0]["lr"]),
            "train_loss_online": float(sq_s / max(seen, 1)),
            "aux_loss_mean": float(aux_s / max(steps_per_epoch, 1)) if aux is not None else float("nan"),
            "is_best": bool(improved),
            "grad_norm_max": float(ep_gnorm),
            "train_seconds": float(train_seconds),
            "eval_seconds": float(eval_seconds),
        }
        for s in C.SPLITS:
            for k, v in metrics[s].items():
                if isinstance(v, (int, float, np.floating)):
                    row[f"{s}_{k}"] = float(v)
        curve.append(row)
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    if best_epoch is None or best_state is None:
        raise RuntimeError("training ended without any improvement")
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    curve_primary = np.asarray([
        (r["valid_rmse"] if task == "regression" else r["valid_roc_auc"]) for r in curve
    ], dtype=np.float64)
    argbest = int((np.argmin(curve_primary) if task == "regression" else np.argmax(curve_primary)) + 1)
    if argbest != int(best_epoch):
        raise RuntimeError(f"curve argbest {argbest} != tracked best epoch {best_epoch}")

    # best checkpoint: reload, re-verify, save predictions for all splits
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_preds = evaluate_all(model, eval_caches, device, path_counts)
    best_metrics = {s: split_metrics(task, best_preds[s], tgt["y"][s], tnorm) for s in C.SPLITS}
    recheck = abs(primary_value(task, best_metrics["valid"]) - float(best_value))
    if recheck > 1e-9:
        raise RuntimeError(f"reloaded best checkpoint Valid metric differs by {recheck}")

    model.load_state_dict({k: v.to(device) for k, v in last_state.items()})
    last_preds = evaluate_all(model, eval_caches, device, path_counts)
    last_metrics = {s: split_metrics(task, last_preds[s], tgt["y"][s], tnorm) for s in C.SPLITS}

    _write_curve(out_dir / "curve.csv", curve)
    C.write_json(out_dir / "diagnostics.json", {"grad_steps": diag, "eval_rng_probe": [eval_probe] + eval_rng_probe,
                                               "rng": {"init_after_reset": init_rng,
                                                       "train_start": train_start_rng,
                                                       "train_end": rng_hash(device)}})
    torch.save({"state_dict": best_state, "best_epoch": int(best_epoch),
                "best_metric_name": f"valid_{task_meta['primary']}",
                "best_metric_value": float(best_value), "candidate": candidate,
                "dataset": dataset, "parameters": int(model.parameter_count()),
                "target_norm": dict(tnorm), "epochs_run": int(len(curve)),
                "init_state_hash": init_state_hash, "train_rows_sha256": C.row_sha256(tgt["rows"]["train"]),
                "protocol_id": C.PROTOCOL_ID, "recipe": recipe["name"] if "name" in recipe else cand["recipe"]},
               out_dir / "best.pt")
    torch.save({"state_dict": last_state, "best_epoch": int(best_epoch), "epochs_run": int(len(curve)),
                "candidate": candidate, "dataset": dataset}, out_dir / "last.pt")
    np.savez(out_dir / "predictions_best.npz", **{
        f"{s}_{k}": v for s in C.SPLITS
        for k, v in (("rows", np.asarray(tgt["rows"][s], np.int64)), ("y", tgt["y"][s]),
                     ("pred_std", best_preds[s]))
    })
    np.savez(out_dir / "predictions_last.npz", **{
        f"{s}_{k}": v for s in C.SPLITS
        for k, v in (("rows", np.asarray(tgt["rows"][s], np.int64)), ("y", tgt["y"][s]),
                     ("pred_std", last_preds[s]))
    })

    payload = {
        "schema": "lcrc-performance-route-train-v1",
        "round_id": C.ROUND_ID, "protocol_id": C.PROTOCOL_ID,
        "status": "completed", "smoke": bool(smoke),
        "candidate": candidate, "candidate_def": cand, "kind": kind,
        "dataset": dataset, "task": task, "split_method": task_meta["split_method"],
        "units": task_meta["units"],
        "model": {
            "variant": variant, "parameters": int(model.parameter_count()),
            "dropout": float(dropout),
            "shared_init_bitwise_equal": bool(shared["all_equal"]),
            "shared_init_checked": int(shared["n_checked"]),
            "init_copy": init_report, "init_state_hash": init_state_hash,
            "support_kappa": (float(kappa) if support is not None else None),
            "support_a_bar": (float(a_bar) if support is not None else None),
            "kappa_mult": cand.get("kappa_mult"),
        },
        "data": {
            "n_train": int(task_meta["sizes"][0]), "n_valid": int(task_meta["sizes"][1]),
            "n_test": int(task_meta["sizes"][2]),
            "rows_sha256": {s: C.row_sha256(tgt["rows"][s]) for s in C.SPLITS},
            "cache_fingerprint": {s: caches[s]["meta"]["content_fingerprint"] for s in C.SPLITS},
            "target_norm": dict(tnorm),
            "train_norm_fingerprint": C.DD.PREP_DATA.norm_fingerprint(train_norm),
            "test_used_for_selection": False,
        },
        "recipe": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in recipe.items()},
                   "epochs_run": int(len(curve)), "steps_per_epoch": int(steps_per_epoch),
                   "optimizer_steps": int(len(curve) * steps_per_epoch),
                   "shuffle_generator_seed": int(C.SHUFFLE_SEED),
                   "checkpoint_rule": "best Validation only, strict improvement, tie keeps earlier"},
        "metrics": {"best": best_metrics, "last": last_metrics,
                    "initial": None},
        "anomalies": {"grad_clip_events": int(clip_events), "grad_norm_max": float(gnorm_max)},
        "timing": {"train_wall_seconds": float(wall),
                   "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
                   "mean_epoch_seconds": float(np.mean([r["train_seconds"] + r["eval_seconds"] for r in curve]))},
        "gpu": {"device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "peak_memory_allocated_bytes": int(gpu_peak),
                "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0)
                if device.type == "cuda" else 0.0},
        "environment": {**C.env_info(), "device": str(device)},
        "provenance": {"git_commit": C.git_commit(), "git_dirty_tracked": C.git_dirty(),
                       "code_hashes": C.code_hashes()},
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{candidate}@{dataset}] best_epoch={best_epoch}/{len(curve)} "
        f"params={model.parameter_count()} "
        f"valid={primary_value(task, best_metrics['valid']):.6f} "
        f"test={primary_value(task, best_metrics['test']):.6f} wall={wall:.1f}s"
    )
    return payload


def _concat_chem8(cache: Mapping[str, Any], extra: np.ndarray) -> np.ndarray:
    base = np.asarray(cache["atom_feat"], dtype=np.float32)
    add = np.asarray(extra, dtype=np.float32)
    if base.shape[0] != add.shape[0]:
        raise ValueError("chem8 atom count differs from the cache")
    return np.concatenate([base, add], axis=1)


def _train_structure_norm(cache: Mapping[str, Any], spec) -> dict[str, Any]:
    path = C.CACHE / f"train_norm_{cache['dataset']}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    norm = C.DD.PREP_DATA.fit_norm(cache, spec, source=f"{cache['dataset']}_train_structures_only")
    if norm.get("valid_or_test_used") is not False:
        raise ValueError("structure normalisation must be Train-only")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(norm[k], dtype=np.float32)
                      for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")})
    return {k: np.asarray(norm[k], dtype=np.float32)
            for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")}


def _grad_diagnostics(model, loss_main, loss_aux) -> dict[str, Any]:
    params = [p for p in model.parameters() if p.requires_grad]
    g_main = torch.autograd.grad(loss_main, params, retain_graph=True, allow_unused=True)
    if loss_aux is not None and loss_aux.requires_grad:
        g_aux = torch.autograd.grad(loss_aux, params, retain_graph=True, allow_unused=True)
    else:
        g_aux = [None] * len(params)
    n_main = _norm(g_main)
    n_aux = _norm(g_aux)
    dot = sum(float((a.detach().double() * b.detach().double()).sum())
              for a, b in zip(g_main, g_aux) if a is not None and b is not None)
    cos = float(dot / (n_main * n_aux)) if n_main > 0 and n_aux > 0 else None
    return {"loss_main": float(loss_main.detach()),
            "loss_aux": float(loss_aux.detach()) if loss_aux is not None else 0.0,
            "grad_main_l2": float(n_main), "grad_aux_l2": float(n_aux),
            "grad_cosine": cos,
            "aux_over_main_ratio": float(n_aux / n_main) if n_main > 0 else None}


def _norm(grads) -> float:
    t = 0.0
    for g in grads:
        if g is not None:
            t += float((g.detach().double() ** 2).sum())
    return math.sqrt(t)


def _write_curve(path: Path, curve: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in curve:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in curve:
            w.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="train one candidate on one dataset")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--recipe", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    out = Path(args.out_dir) if args.out_dir else C.run_dir(args.candidate, args.dataset)
    train_candidate(candidate=args.candidate, dataset=args.dataset, device=device, out_dir=out,
                    recipe_name=args.recipe, epochs_override=args.epochs, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
