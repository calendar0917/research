"""One training unit: same chain for every split / model / loss / seed.

The optimisation recipe is the frozen R3 one.  The only things that differ
between units are the data (which split and which Fit rows), the architecture,
the loss condition and the seed.

Everything that could leak a holdout label is refused here: only ``fit`` targets
enter the optimiser, only ``select`` targets enter the checkpoint rule, and the
Diagnose/Audit *labels* are never read -- predictions on those partitions are
computed and stored (a forward pass needs no label) so the later scoring step
does not have to re-run 100+ checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import lr_common as C
import lr_data as D
import lr_models as LM
import lr_pairs as P

PROCESS_T0 = time.perf_counter()
DOSE_FREEZE = C.ANALYSIS / "dose_freeze.json"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    err = pred - y
    ss_res = float((err * err).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": int(y.size), "rmse": float(math.sqrt(ss_res / max(y.size, 1))),
            "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def raw_metrics(standardised: np.ndarray, y: np.ndarray, target_norm: Mapping[str, Any]) -> dict[str, float]:
    pred = np.asarray(standardised, dtype=np.float64) * float(target_norm["scale"]) + float(target_norm["mean"])
    return regression_metrics(pred, y)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class BestEpochTracker:
    def __init__(self) -> None:
        self.best_period: int | None = None
        self.best_primary = math.inf
        self.best_metrics: dict[str, float] | None = None
        self.improvement_periods: list[int] = []
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, period: int, metrics: Mapping[str, float]) -> bool:
        primary = float(metrics["rmse"])
        if self.best_period is not None and not primary < self.best_primary:
            return False
        self.best_period = int(period)
        self.best_primary = primary
        self.best_metrics = dict(metrics)
        self.improvement_periods.append(int(period))
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("tracker has no captured state")
        return self._state


def load_dose() -> dict[str, Any]:
    if not DOSE_FREEZE.exists():
        return {"doses": {}, "status": "MISSING"}
    return C.read_json(DOSE_FREEZE)


# ---------------------------------------------------------------------------
# fingerprint features (F only)
# ---------------------------------------------------------------------------


def fingerprint_payload(split: str) -> dict[str, Any]:
    path = C.CACHE / f"fp_{split}.npz"
    if not path.exists():
        raise SystemExit(f"fingerprint cache {path} missing; run lr_prepare.py")
    data = np.load(path)
    out = {k: data[k].astype(np.float32) for k in data.files}
    for key in ("fit", "select", "diagnose", "audit"):
        if key not in out:
            out[key] = None
    return out


def predict_cached(model, cache: Mapping[str, Any], fp_matrix, device, batch_size: int = 256) -> np.ndarray:
    """Prediction helper that also attaches the fingerprint block for ``F``."""
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = D.to_device(D.collate(cache, rows), device)
            if fp_matrix is not None:
                batch["fp"] = torch.from_numpy(
                    fp_matrix[np.asarray(rows, dtype=np.int64)]).to(device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# Fit-subset / norm helpers
# ---------------------------------------------------------------------------


def subset_structure_norm(cache: Mapping[str, Any], positions: Sequence[int], spec) -> dict[str, Any]:
    """Fit-only structure normalisation over an explicit subset of Fit molecules."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    lo, hi = int(atom_ptr[int(positions[0])]), int(atom_ptr[int(positions[-1]) + 1])
    # positions are sorted in the callers; use a mask to stay safe otherwise
    mask = np.zeros(int(cache["n_molecules"]) + 1, dtype=bool)
    for p in positions:
        mask[int(p)] = True
    bounds = np.asarray(cache["atom_ptr"])
    atom_mask = np.zeros(int(bounds[-1]), dtype=bool)
    for p in positions:
        atom_mask[bounds[int(p)]:bounds[int(p) + 1]] = True
    scalars = np.asarray(cache["atom_scalar"])[atom_mask].astype(np.float64)
    globals_ = np.asarray(cache["global"])[np.asarray(list(positions), dtype=np.int64)].astype(np.float64)

    def _scale(std: np.ndarray):
        scale = std.astype(np.float32)
        const = np.where(scale <= 0)[0].tolist()
        scale = scale.copy()
        scale[scale <= 0] = 1.0
        return scale, const

    scalar_scale, scalar_const = _scale(scalars.std(0))
    global_scale, global_const = _scale(globals_.std(0))
    return {
        "source": f"subset_fit_only_n={len(positions)}",
        "valid_or_test_used": False,
        "zero_variance_policy": "constant_fit_features_use_scale_1.0",
        "scalar_constant_indices": scalar_const,
        "global_constant_indices": global_const,
        "scalar_mean": scalars.mean(0).astype(np.float32),
        "scalar_scale": scalar_scale,
        "global_mean": globals_.mean(0).astype(np.float32),
        "global_scale": global_scale,
    }


def subset_target_norm(y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean, scale = float(y.mean()), float(y.std(ddof=0))
    if not (np.isfinite(mean) and np.isfinite(scale) and scale > 0):
        raise ValueError("degenerate subset target distribution")
    return {"mean": mean, "scale": scale, "n": int(y.size), "source": "subset_fit_only"}


def learning_curve_positions(split: str, target_n: int) -> list[int]:
    """Deterministic whole-component subset of the Fit partition.

    Components are ordered by ``sha256('lc-<target_n>-<stable id>')``; whole
    components are admitted while the budget allows, then size-1 components fill
    the remaining quota in the same order.  No label, error or pair status is
    used to choose.
    """
    payload = D.split_payload(split)
    units = [sorted(int(r) for r in u) for u in payload["units_per_partition"]["fit"]]
    rows = payload["rows_by_partition"]["fit"]
    pos = {int(r): i for i, r in enumerate(rows)}
    order = sorted(units, key=lambda u: C.sha256_text(f"lc-{int(target_n)}|{C.stable_id(u)}"))
    chosen: list[list[int]] = []
    total = 0
    rest: list[list[int]] = []
    for unit in order:
        if total + len(unit) <= target_n:
            chosen.append(unit)
            total += len(unit)
        else:
            rest.append(unit)
    for unit in rest:
        if total >= target_n:
            break
        if len(unit) == 1:
            chosen.append(unit)
            total += 1
    if total != target_n:
        raise SystemExit(f"learning-curve subset for {split} reached {total} != {target_n}")
    positions = sorted(pos[r] for unit in chosen for r in unit)
    return positions


# ---------------------------------------------------------------------------
# the step loss
# ---------------------------------------------------------------------------


def auxiliary_term(e_a: torch.Tensor, e_b: torch.Tensor, loss: str) -> torch.Tensor:
    if loss == "R":
        return ((e_b - e_a) ** 2).mean()
    if loss == "W":
        return (e_a**2 + e_b**2).mean()
    if loss == "N":
        return ((e_b - e_a) ** 2).mean()
    if loss == "A":
        return ((e_b - e_a) ** 2).mean()  # computed, then multiplied by the zero coefficient
    raise KeyError(loss)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_unit(
    *,
    rid: str,
    split: str,
    model_kind: str,
    loss: str,
    seed: int,
    stage: str = "main",
    device: torch.device,
    out_dir: Path | None = None,
    smoke_steps: int | None = None,
    lc_size: str | None = None,
    plan_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if model_kind not in C.MODELS:
        raise ValueError(model_kind)
    if loss not in C.LOSSES:
        raise ValueError(loss)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    out_dir = Path(out_dir) if out_dir else C.run_dir(rid)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ---------------------------------------------------------
    spec = D.frozen_spec()
    fit_cache = D.load_cache(split, "fit")
    select_cache = D.load_cache(split, "select")
    rows_fit = [int(r) for r in np.asarray(fit_cache["rows"]).tolist()]
    n_fit_total = len(rows_fit)
    if lc_size is not None:
        target_n = C.LC_LABELS[lc_size]
        positions = learning_curve_positions(split, target_n)
        steps_per_period = C.LC_STEPS[lc_size]
        periods = C.LC_EPOCHS[lc_size]
        train_norm = subset_structure_norm(fit_cache, positions, spec)
        y_fit = D.y_aligned(split, "fit", [rows_fit[p] for p in positions], "train")
        tgt = subset_target_norm(y_fit)
    else:
        positions = list(range(n_fit_total))
        train_norm = D.train_norm(split)
        y_fit = D.y_aligned(split, "fit", rows_fit, "train")
        tgt = D.target_norm(split)
        steps_per_period = math.ceil(len(positions) / C.BATCH)
        periods = C.EPOCHS if split != "ft" else C.EPOCHS
    y_select = D.y_aligned(split, "select", [int(r) for r in np.asarray(select_cache["rows"]).tolist()], "select")
    planned_updates = C.FT_UPDATES if split == "ft" else C.FORMAL_UPDATES
    total_updates = int(smoke_steps) if smoke_steps is not None else int(steps_per_period * periods)
    if total_updates > planned_updates:
        raise SystemExit(f"{rid}: {steps_per_period * periods} updates exceeds the frozen {planned_updates}")

    # ---- model --------------------------------------------------------
    extra = None
    fp_mat = None
    if model_kind == "F":
        fp = fingerprint_payload(split)
        if lc_size is not None:
            fp_fit = fp["fit"][np.asarray(positions, dtype=np.int64)]
            fp_mean = fp_fit.mean(0)
            fp_scale = fp_fit.std(0)
            fp_scale[fp_scale <= 0] = 1.0
        else:
            fp_mean, fp_scale = fp["mean"], fp["scale"]
        extra = {"fp_mean": fp_mean, "fp_scale": fp_scale}
        fp_mat = fp["fit"]
    model = LM.build_model(model_kind, spec, train_norm, int(seed), device, extra)
    params_total = LM.parameter_count(model)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init_hash = C.sha256_bytes(b"".join(
        k.encode() + np.ascontiguousarray(v.numpy(), dtype=np.float32).tobytes()
        for k, v in sorted(init_state.items())
    ))

    # ---- optimisation -------------------------------------------------
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.RECIPE["peak_lr"]),
        betas=tuple(float(x) for x in C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_updates, 1), eta_min=float(C.RECIPE["min_lr"])
    )
    generator = torch.Generator().manual_seed(int(seed) + int(C.SHUFFLE_OFFSET))
    y_fit_scaled = torch.as_tensor((y_fit - float(tgt["mean"])) / float(tgt["scale"]), dtype=torch.float32)
    z_fit = y_fit_scaled.numpy().astype(np.float64)
    pos_to_local = np.full(n_fit_total, -1, dtype=np.int64)
    pos_to_local[np.asarray(positions, dtype=np.int64)] = np.arange(len(positions), dtype=np.int64)

    # ---- frozen pair stream ------------------------------------------
    plan = dict(plan_override) if plan_override is not None else P.get_plan(split, seed, planned_updates)
    dose_all = load_dose()
    lam = 0.0
    if loss != "A":
        lam = float(dose_all.get("doses", {}).get(model_kind, {}).get(loss, float("nan")))
        if not np.isfinite(lam):
            raise SystemExit(f"no frozen dose for {model_kind}/{loss}; run lr_dose.py first")

    table_ = P.table()
    tracker = BestEpochTracker()
    curve: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    grad_norm_max = 0.0
    clip_events = 0
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    step = 0
    period = 0
    batch_iter: list[list[int]] = []
    perm_cursor = 0
    batch_order_probe: list[int] = []
    started = time.perf_counter()
    train_seconds = 0.0
    eval_seconds = 0.0

    def next_batch() -> list[int]:
        nonlocal batch_iter, perm_cursor
        if perm_cursor >= len(batch_iter):
            perm = torch.randperm(len(positions), generator=generator).tolist()
            if not batch_order_probe:
                batch_order_probe.extend(positions[i] for i in perm[:10])
            batch_iter = [[positions[i] for i in perm[j:j + C.BATCH]]
                          for j in range(0, len(perm), C.BATCH)]
            perm_cursor = 0
        out = batch_iter[perm_cursor]
        perm_cursor += 1
        return out

    def add_fp(batch: dict[str, Any], cache_positions: Sequence[int]) -> dict[str, Any]:
        if fp_mat is not None:
            batch["fp"] = torch.from_numpy(fp_mat[np.asarray(list(cache_positions), dtype=np.int64)]).to(device)
        return batch

    model.train()
    while step < total_updates:
        t0 = time.perf_counter()
        rows = next_batch()
        batch = add_fp(D.to_device(D.collate(fit_cache, rows), device), rows)
        y = y_fit_scaled[torch.as_tensor([int(pos_to_local[p]) for p in rows])].to(device)
        pred = model(batch).reshape(-1)
        loss_mol = F.mse_loss(pred, y)

        # auxiliary endpoint forward (eval mode, autograd kept)
        pair_idx = np.asarray(plan["pairs"][step], dtype=np.int64)
        perm = np.asarray(plan["perm"][step], dtype=np.int64)
        ep_rows, idx_a, idx_b, _ = P.endpoint_index(pair_idx, table_.row_a, table_.row_b)
        lookup = {int(r): i for i, r in enumerate(rows_fit)}
        ep_cache_pos = [lookup[int(r)] for r in ep_rows]
        aux_batch = add_fp(D.to_device(D.collate(fit_cache, ep_cache_pos), device), ep_cache_pos)
        was_training = model.training
        model.eval()
        aux_out = model(aux_batch).reshape(-1)
        model.train(was_training)
        z_ep = torch.as_tensor([float(z_fit[int(pos_to_local[c])]) for c in ep_cache_pos],
                               dtype=torch.float32, device=device)
        e = aux_out - z_ep
        ia = torch.as_tensor(idx_a, dtype=torch.long, device=device)
        ib_real = torch.as_tensor(idx_b, dtype=torch.long, device=device)
        ib_null = torch.as_tensor([idx_b[int(k)] for k in perm], dtype=torch.long, device=device)
        ib = ib_null if loss == "N" else ib_real
        aux_value = auxiliary_term(e[ia], e[ib], loss)
        total_loss = loss_mol + lam * aux_value

        if step + 1 in (1, 100, 1000, total_updates):
            params = [p for p in model.parameters() if p.requires_grad]
            g_mol = torch.autograd.grad(loss_mol, params, retain_graph=True, allow_unused=True)
            g_aux = torch.autograd.grad(lam * aux_value if loss != "A" else aux_value, params,
                                        retain_graph=True, allow_unused=True)
            probe_records.append({
                "step": int(step + 1),
                "loss_mol": float(loss_mol.detach()),
                "aux_raw": float(aux_value.detach()),
                "lambda": float(lam),
                "aux_weighted": float(lam * float(aux_value.detach())),
                "train_pair_mse_standardised": float(((e[ia] - e[ib]) ** 2).mean().detach()),
                "grad_norm_mol": float(torch.sqrt(sum((g ** 2).sum() for g in g_mol if g is not None))),
                "grad_norm_aux_raw": float(torch.sqrt(sum((g ** 2).sum() for g in g_aux if g is not None))),
                "n_params_with_aux_grad": int(sum(1 for g in g_aux if g is not None and bool(g.abs().sum() > 0))),
            })
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"])))
        if not math.isfinite(float(total_loss)) or not math.isfinite(gnorm):
            raise RuntimeError(f"non-finite loss/gradient at step {step + 1}")
        if gnorm > float(C.RECIPE["gradient_clip"]):
            clip_events += 1
        grad_norm_max = max(grad_norm_max, gnorm)
        optimizer.step()
        scheduler.step()
        step += 1
        train_seconds += time.perf_counter() - t0

        if step % steps_per_period == 0 or step == total_updates:
            period += 1
            t1 = time.perf_counter()
            select_std = predict_cached(model, select_cache, None if fp_mat is None else fp["select"], device)
            model.train()
            select_metrics = raw_metrics(select_std, y_select, tgt)
            eval_seconds += time.perf_counter() - t1
            improved = tracker.consider(period, select_metrics)
            if improved:
                tracker.capture(model)
            curve.append({
                "period": int(period), "step": int(step), "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm_max": float(gnorm), "train_seconds": float(train_seconds),
                "is_best": bool(improved), "select_rmse_raw": float(select_metrics["rmse"]),
                "select_mae_raw": float(select_metrics["mae"]), "select_r2_raw": float(select_metrics["r2"]),
                "aux_raw": float(aux_value.detach()), "loss_mol": float(loss_mol.detach()),
            })
            if device.type == "cuda":
                gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    if smoke_steps is None and step != total_updates:
        raise RuntimeError(f"expected {total_updates} updates, ran {step}")
    if tracker.best_period is None:
        raise RuntimeError("training ended without any selection improvement")

    wall = time.perf_counter() - started
    best_state = tracker.state
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    select_std = predict_cached(model, select_cache, None if fp_mat is None else fp["select"], device)
    select_metrics = raw_metrics(select_std, y_select, tgt)
    recheck = abs(float(select_metrics["rmse"]) - float(tracker.best_metrics["rmse"]))
    if recheck > 1e-5:
        raise RuntimeError(f"reloaded best Select RMSE differs by {recheck}")
    curve_values = np.asarray([row["select_rmse_raw"] for row in curve], dtype=np.float64)
    if int(np.argmin(curve_values)) != int(tracker.best_period) - 1:
        raise RuntimeError("curve argmin != tracked best period")
    fit_std = predict_cached(model, fit_cache, fp_mat, device)
    fit_metrics = raw_metrics(fit_std, y_fit, tgt)

    # predictions for every partition (no label is read here)
    preds: dict[str, Any] = {}
    for partition in ("fit", "select", "diagnose", "audit"):
        cache_rows = D.rows_by_partition(split)[partition]
        if not cache_rows:
            continue
        if partition in ("fit", "select"):
            cache = fit_cache if partition == "fit" else select_cache
        else:
            cache = D.load_cache(split, partition)
        if int(cache["n_molecules"]) != len(cache_rows):
            raise ValueError(f"{split}/{partition}: cache size mismatch")
        std = predict_cached(model, cache, None if fp_mat is None else fp[partition], device)
        np.savez(out_dir / f"pred_{partition}.npz", rows=np.asarray(cache_rows, dtype=np.int64),
                 standardised=std, target_mean=float(tgt["mean"]), target_scale=float(tgt["scale"]),
                 labels_read=False)
        preds[partition] = C.track_rel(out_dir / f"pred_{partition}.npz")

    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    best_state_hash = C.sha256_bytes(b"".join(
        k.encode() + np.ascontiguousarray(v.numpy(), dtype=np.float32).tobytes()
        for k, v in sorted(best_state.items())
    ))
    final_hash = C.sha256_bytes(b"".join(
        k.encode() + np.ascontiguousarray(v.numpy(), dtype=np.float32).tobytes()
        for k, v in sorted(final_state.items())
    ))
    checkpoint = {
        "state_dict_best": best_state,
        "state_dict_last": final_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "numpy": np.random.get_state(),
            "sampler_generator": generator.get_state(),
        },
        "meta": {
            "run_id": rid, "split": split, "model": model_kind, "loss": loss, "seed": int(seed),
            "stage": stage, "best_period": int(tracker.best_period), "updates": int(step),
            "best_state_hash": best_state_hash, "last_state_hash": final_hash,
            "init_state_hash": init_hash, "parameters": params_total,
            "target_norm": dict(tgt), "lambda": float(lam), "pair_plan_hash": plan["hash"],
            "feature_spec_hash": spec.hash(), "protocol_id": C.ROUND_ID,
            "structure_norm_fingerprint": C.sha256_text(C.stable_json(
                {k: np.asarray(v).tolist() for k, v in train_norm.items()
                 if k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")})),
        },
    }
    atomic_torch_save(checkpoint, out_dir / "checkpoint.pt")

    payload: dict[str, Any] = {
        "run_id": rid, "round": C.ROUND_ID, "stage": stage, "status": "COMPLETE",
        "split": split, "model": model_kind, "loss": loss, "seed": int(seed),
        "smoke": smoke_steps is not None,
        "lc_size": lc_size,
        "labels": {"fit_used_for_optimizer": True, "select_used_for_checkpoint": True,
                   "diagnose_labels_read": False, "audit_labels_read": False},
        "model_info": {"parameters": params_total, "parameters_expected": C.EXPECTED_PARAMS[model_kind],
                       "init_state_hash": init_hash, "kind": C.MODEL_NAMES[model_kind]},
        "data": {"n_fit": len(positions), "n_select": int(select_cache["n_molecules"]),
                 "fit_rows_sha256": C.row_sha256([rows_fit[p] for p in positions]),
                 "select_rows_sha256": C.row_sha256([int(r) for r in np.asarray(select_cache["rows"]).tolist()]),
                 "fit_fingerprint": fit_cache["meta"]["content_fingerprint"],
                 "target_norm": dict(tgt),
                 "structure_norm_fingerprint": checkpoint["meta"]["structure_norm_fingerprint"]},
        "recipe": {"steps_per_period": int(steps_per_period), "periods": int(periods),
                   "updates": int(step), "batch_size": int(C.BATCH), "pair_batch": int(C.PAIR_BATCH),
                   "lambda": float(lam), "pair_plan_hash": plan["hash"],
                   "optimizer": C.RECIPE["optimizer"], "peak_lr": float(C.RECIPE["peak_lr"]),
                   "min_lr": float(C.RECIPE["min_lr"]), "weight_decay": float(C.RECIPE["weight_decay"]),
                   "gradient_clip": float(C.RECIPE["gradient_clip"]),
                   "dropout": float(C.RECIPE["dropout"]),
                   "selection_rule": "lowest_select_rmse__strict__tie_keeps_earlier",
                   "early_stopping": False,
                   "batch_order_probe": batch_order_probe},
        "metrics": {"fit": fit_metrics, "select": select_metrics},
        "checkpoint": {"best_period": int(tracker.best_period), "best_metric": "select_rmse",
                       "best_value": float(tracker.best_primary),
                       "n_improvements": len(tracker.improvement_periods),
                       "best_state_hash": best_state_hash, "last_state_hash": final_hash,
                       "select_recheck_abs_diff": float(recheck),
                       "path": C.track_rel(out_dir / "checkpoint.pt"),
                       "sha256": C.sha256_file(out_dir / "checkpoint.pt"),
                       "curve_path": C.track_rel(curve_path),
                       "curve_sha256": C.sha256_file(curve_path)},
        "predictions": preds,
        "probes": probe_records,
        "anomalies": {"grad_clip_events": int(clip_events), "grad_norm_max": float(grad_norm_max),
                      "nonfinite_events": 0},
        "timing": {"train_wall_seconds": float(train_seconds), "eval_wall_seconds": float(eval_seconds),
                   "run_wall_seconds": float(wall),
                   "process_wall_seconds": float(time.perf_counter() - PROCESS_T0)},
        "gpu": {"device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "peak_memory_allocated_bytes": int(gpu_peak),
                "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0)
                if device.type == "cuda" else 0.0},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda, "hostname": platform.node(),
                        "device": str(device), "rdkit": C.rdkit_version()},
        "provenance": {"git_commit": C.git_commit(), "git_dirty_tracked": C.git_dirty(),
                       "code_hashes": C.code_hashes()},
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(f"[{rid}] best_period={tracker.best_period}/{len(curve)} updates={step} "
          f"fit_rmse={fit_metrics['rmse']:.5f} select_rmse={select_metrics['rmse']:.5f} "
          f"lambda={lam:.4g} wall={payload['timing']['process_wall_seconds']:.1f}s")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LCRC route clarification: one training unit")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", required=True, choices=list(C.SPLITS) + ["ft"])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--loss", required=True, choices=list(C.LOSSES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", default="main")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke-steps", type=int, default=None)
    ap.add_argument("--lc-size", default=None, choices=list(C.LC_LABELS))
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    train_unit(rid=args.run_id, split=args.split, model_kind=args.model, loss=args.loss,
               seed=int(args.seed), stage=args.stage, device=device,
               out_dir=Path(args.out_dir) if args.out_dir else None,
               smoke_steps=args.smoke_steps, lc_size=args.lc_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
