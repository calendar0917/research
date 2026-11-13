"""One fixed training unit: A / R / W / N at seed 0.

The ordinary molecule stream is identical in all four arms: same initial state,
same pre-computed batch order, same dropout RNG reset, one dropout call per
ordinary forward.  R/W/N add one auxiliary forward on the *same* deduplicated
endpoint set in ``model.eval()`` mode, with autograd enabled.  A skips that
forward entirely.

Every step performs exactly one ``total.backward()`` and one
``optimizer.step()``.  The auxiliary term never gets its own optimizer update.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C
import data as D
import losses as L
import metrics as M
import pairs as P
import schedule as S

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rng_hash(device) -> str:
    h = C.sha256_bytes(torch.get_rng_state().numpy().tobytes())
    if device.type == "cuda":
        h += ":" + C.sha256_bytes(torch.cuda.get_rng_state(device).numpy().tobytes())
    return h


def _reset_rng(seed: int, device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _grad_norm(grads: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for g in grads:
        if g is not None:
            total += float((g.detach().double() ** 2).sum().item())
    return math.sqrt(total)


def _grad_dot(a: Sequence[torch.Tensor | None], b: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        if x is not None and y is not None:
            total += float((x.detach().double() * y.detach().double()).sum().item())
    return total


def _diagnostic_grads(
    model: torch.nn.Module,
    loss_mol: torch.Tensor,
    loss_aux: torch.Tensor | None,
    *,
    clip_value: float,
) -> dict[str, Any]:
    params = [p for p in model.parameters() if p.requires_grad]
    g_mol = torch.autograd.grad(loss_mol, params, retain_graph=True, allow_unused=True)
    if loss_aux is not None and bool(loss_aux.requires_grad):
        g_aux = torch.autograd.grad(loss_aux, params, retain_graph=True, allow_unused=True)
    else:
        g_aux = [None] * len(params)
    n_mol = _grad_norm(g_mol)
    n_aux = _grad_norm(g_aux)
    dot = _grad_dot(g_mol, g_aux)
    cosine = None
    if n_mol > 0 and n_aux > 0:
        cosine = float(dot / (n_mol * n_aux))
    total_norm = _grad_norm([None if (a is None and b is None) else (0 if a is None else a) + (0 if b is None else b) for a, b in zip(g_mol, g_aux)])
    return {
        "loss_mol": float(loss_mol.detach()),
        "loss_aux": float(loss_aux.detach()) if loss_aux is not None else 0.0,
        "grad_mol_l2": float(n_mol),
        "grad_aux_l2": float(n_aux),
        "grad_norm_ratio_aux_over_mol": float(n_aux / n_mol) if n_mol > 0 else None,
        "grad_cosine_mol_aux": cosine,
        "grad_total_l2_preclip": float(total_norm),
        "would_clip_at": float(clip_value),
        "aux_grad_identically_zero": bool(n_aux == 0.0),
    }


def _aux_forward(model, cache, unique_idx: Sequence[int], device) -> tuple[torch.Tensor, dict[str, Any]]:
    was_training = bool(model.training)
    model.eval()
    batch = D.to_device(D.collate(cache, list(unique_idx)), device)
    with torch.enable_grad():
        pred = model(batch).reshape(-1)
    model.train() if was_training else model.eval()
    return pred, {"restored_training": bool(model.training == was_training)}


def _render_aux(
    model,
    cache,
    unique_rows: Sequence[int],
    unique_idx: Sequence[int],
    row_to_cache: Mapping[int, int],
    z_global: torch.Tensor,
    device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pred, info = _aux_forward(model, cache, unique_idx, device)
    z = z_global[list(unique_rows)].to(device)
    return pred - z, info


class BestEpochTracker:
    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_primary = math.inf
        self.best_metrics: dict[str, float] | None = None
        self.improvement_epochs: list[int] = []
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        primary = float(metrics["rmse"])
        if self.best_epoch is not None and not primary < self.best_primary:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = primary
        self.best_metrics = dict(metrics)
        self.improvement_epochs.append(int(epoch))
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("tracker has no captured state")
        return self._state


# ---------------------------------------------------------------------------
# training one arm
# ---------------------------------------------------------------------------


def train_arm(
    *,
    arm: str,
    device: torch.device,
    out_dir: Path,
    epochs: int,
    smoke: bool,
) -> dict[str, Any]:
    if arm not in C.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if not smoke and int(epochs) != int(C.RECIPE["epochs"]):
        raise SystemExit(f"formal runs must use {C.RECIPE['epochs']} epochs")
    if smoke and int(epochs) > 2:
        raise SystemExit("smoke runs may not exceed 2 epochs")
    if not smoke and C.git_dirty():
        raise SystemExit("formal run requested on a dirty tracked worktree; commit first")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = D.load_dataset(include_test=False)
    spec = dataset["spec"]
    train_cache = dataset["train_cache"]
    valid_cache = dataset["valid_cache"]
    train_rows = [int(r) for r in dataset["rows"]["train"]]
    valid_rows = [int(r) for r in dataset["rows"]["valid"]]
    y_train = np.asarray(dataset["y_train"], dtype=np.float64)
    y_valid = np.asarray(dataset["y_valid"], dtype=np.float64)
    train_norm = dataset["train_norm"]
    target_norm = dataset["target_norm"]
    if int(train_cache["n_molecules"]) != C.SPLIT_SIZES[0]:
        raise RuntimeError("train cache size changed")
    if int(valid_cache["n_molecules"]) != C.SPLIT_SIZES[1]:
        raise RuntimeError("valid cache size changed")

    schedule = S.load_schedule()
    arrays = schedule["arrays"]
    manifest = schedule["manifest"]
    p_train = sorted(P.p_train(), key=lambda p: str(p["pair_id"]))
    if len(p_train) != int(manifest["p_train"]["n_pairs"]):
        raise RuntimeError("P_train count differs from schedule manifest")
    if C.sha256_text(C.stable_json([str(p["pair_id"]) for p in p_train])) != manifest["p_train"]["pair_ids_sha256"]:
        raise RuntimeError("P_train pair-id hash differs from schedule manifest")

    n_train = len(train_rows)
    batch_size = int(C.RECIPE["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * int(epochs)
    if steps_per_epoch != int(C.RECIPE["steps_per_epoch"]):
        raise RuntimeError(f"steps_per_epoch {steps_per_epoch} != frozen {C.RECIPE['steps_per_epoch']}")

    row_to_cache = {int(r): i for i, r in enumerate(train_rows)}
    z_train = (y_train - float(target_norm["mean"])) / float(target_norm["scale"])
    z_global = torch.zeros(C.N_MOLECULES, dtype=torch.float32)
    z_global[torch.as_tensor(train_rows, dtype=torch.long)] = torch.as_tensor(z_train, dtype=torch.float32)
    z_train_cache = torch.as_tensor(z_train, dtype=torch.float32)

    full_state = C.DM.build_full_state(spec, train_norm, C.TRAIN_SEED)
    init_state = C.DM.remap_state(C.VARIANT, full_state)
    init_hash = C.DM.state_fingerprint(init_state)
    full_init_hash = C.DM.state_fingerprint(full_state)
    shared = C.DM.shared_state_equal(C.VARIANT, full_state, init_state)
    model = C.DM.DissectLCRC(spec, train_norm, C.VARIANT)
    model.load_state_dict(init_state, strict=True)
    model.to(device)
    params_total = C.DM.parameter_count(model)
    if params_total != C.EXPECTED_PARAMS:
        raise RuntimeError(f"{C.VARIANT}: {params_total} parameters != frozen {C.EXPECTED_PARAMS}")

    _reset_rng(C.TRAIN_SEED, device)
    init_rng_hash = _rng_hash(device)

    # Initial-model diagnostics (no Test target is loaded here).
    model.eval()
    init_train_std = D.predict_scores(model, train_cache, device)
    init_valid_std = D.predict_scores(model, valid_cache, device)
    model.train()
    init_train_metrics = M.raw_metrics(init_train_std, y_train, target_norm)
    init_valid_metrics = M.raw_metrics(init_valid_std, y_valid, target_norm)
    init_train_by_row = {int(r): float(v) for r, v in zip(train_rows, init_train_std)}
    init_y_by_row = {int(r): float(v) for r, v in zip(train_rows, y_train)}
    init_pair_metrics = M.pair_metrics(init_train_by_row, init_y_by_row, p_train)

    dropout_calls = {"n": 0}
    if hasattr(model, "drop"):
        model.drop.register_forward_hook(lambda *_: dropout_calls.__setitem__("n", dropout_calls["n"] + 1))

    torch.manual_seed(C.TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(C.TRAIN_SEED)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.RECIPE["peak_lr"]),
        betas=tuple(float(x) for x in C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=float(C.RECIPE["min_lr"])
    )

    tracker = BestEpochTracker()
    curve: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    epoch_rng_hashes: list[dict[str, Any]] = []
    aux_rng_probe: list[dict[str, Any]] = []
    grad_norm_max = 0.0
    clip_events = 0
    nonfinite_events = 0
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    last_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        perm = arrays["batch_perms"][epoch - 1]
        if len(perm) != n_train:
            raise RuntimeError("batch permutation length differs from Train size")
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        lr_start = float(optimizer.param_groups[0]["lr"])
        abs_sum = 0.0
        sq_sum = 0.0
        aux_sum = 0.0
        seen = 0
        epoch_clips = 0
        epoch_gnorm_max = 0.0
        for step_index, rows in enumerate(batches):
            global_step = (epoch - 1) * steps_per_epoch + step_index + 1
            batch = D.to_device(D.collate(train_cache, [int(x) for x in rows]), device)
            z = z_train_cache[list(rows)].to(device)
            pred = model(batch).reshape(-1)
            loss_mol = L.mol_loss(pred, z)
            aux_info: dict[str, Any] = {}
            loss_aux: torch.Tensor | None = None
            if arm != "A":
                real_idx = arrays["real_pair_indices"][global_step - 1]
                step_pairs = [p_train[int(i)] for i in real_idx]
                real_tokens = []
                for p in step_pairs:
                    real_tokens.extend((int(p["row_a"]), int(p["row_b"])))
                unique_rows = sorted(set(real_tokens))
                unique_idx = [row_to_cache[ int(r)] for r in unique_rows]
                pos = {int(r): i for i, r in enumerate(unique_rows)}
                e_aux, aux_info = _render_aux(
                    model, train_cache, unique_rows, unique_idx, row_to_cache, z_global, device
                )
                if arm == "N":
                    u_rows = [int(x) for x in arrays["random_u"][global_step - 1]]
                    v_rows = [int(x) for x in arrays["random_v"][global_step - 1]]
                    if sorted(u_rows + v_rows) != sorted(real_tokens):
                        raise RuntimeError(f"step {global_step}: N endpoint multiset differs from R")
                    u_pos = [pos[r] for r in u_rows]
                    v_pos = [pos[r] for r in v_rows]
                    e_a = e_aux[u_pos]
                    e_b = e_aux[v_pos]
                else:
                    e_a = e_aux[[pos[int(p["row_a"])] for p in step_pairs]]
                    e_b = e_aux[[pos[int(p["row_b"])] for p in step_pairs]]
                loss_aux = L.aux_loss(arm, e_a, e_b)
                aux_sum += float(loss_aux.detach())
                if global_step == 1:
                    before = _rng_hash(device)
                    model.eval()
                    with torch.enable_grad():
                        _ = model(D.to_device(D.collate(train_cache, unique_idx), device))
                    model.train()
                    after = _rng_hash(device)
                    aux_rng_probe.append(
                        {
                            "arm": arm,
                            "step": 1,
                            "rng_unchanged": bool(before == after),
                            "restored_training": bool(model.training),
                            **aux_info,
                        }
                    )
            total = L.total_loss(loss_mol, loss_aux if loss_aux is not None else torch.zeros((), device=device), C.LAMBDA)

            if global_step in C.DIAGNOSTIC_STEPS:
                diagnostics.append(
                    {
                        "step": int(global_step),
                        "epoch": int(epoch),
                        **_diagnostic_grads(
                            model, loss_mol, loss_aux, clip_value=float(C.RECIPE["gradient_clip"])
                        ),
                    }
                )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"])))
            if not math.isfinite(float(total.detach())) or not math.isfinite(gnorm):
                nonfinite_events += 1
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch} step {step_index + 1}")
            if gnorm > float(C.RECIPE["gradient_clip"]):
                clip_events += 1
                epoch_clips += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            epoch_gnorm_max = max(epoch_gnorm_max, gnorm)
            optimizer.step()
            scheduler.step()

            err = pred.detach() - z
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            seen += int(z.numel())
            if epoch == 1:
                epoch_rng_hashes.append({"step": int(global_step), "rng_hash": _rng_hash(device)})

        train_seconds = time.perf_counter() - t0
        t_eval = time.perf_counter()
        valid_std = D.predict_scores(model, valid_cache, device)
        valid_metrics = M.raw_metrics(valid_std, y_valid, target_norm)
        eval_seconds = time.perf_counter() - t_eval
        improved = tracker.consider(epoch, valid_metrics)
        if improved:
            tracker.capture(model)
        train_mse = float(sq_sum / max(seen, 1))
        row = {
            "epoch": int(epoch),
            "steps": steps_per_epoch,
            "lr_start": lr_start,
            "lr_end": float(optimizer.param_groups[0]["lr"]),
            "train_mse_standardized": train_mse,
            "train_rmse_standardized": math.sqrt(train_mse),
            "train_mae_standardized": float(abs_sum / max(seen, 1)),
            "aux_loss_mean": float(aux_sum / max(steps_per_epoch, 1)),
            "valid_rmse_raw": float(valid_metrics["rmse"]),
            "valid_mae_raw": float(valid_metrics["mae"]),
            "valid_r2_raw": float(valid_metrics["r2"]),
            "is_best": bool(improved),
            "grad_norm_max": float(epoch_gnorm_max),
            "grad_clip_events": int(epoch_clips),
            "clip_ratio": float(epoch_clips / max(steps_per_epoch, 1)),
            "train_seconds": float(train_seconds),
            "eval_seconds": float(eval_seconds),
        }
        curve.append(row)
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise RuntimeError("training ended without any improvement")
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_state = tracker.state
    best_state_hash = C.DM.state_fingerprint(best_state)
    last_state_hash = C.DM.state_fingerprint(last_state)

    curve_values = np.asarray([r["valid_rmse_raw"] for r in curve], dtype=np.float64)
    if int(np.argmin(curve_values) + 1) != int(tracker.best_epoch):
        raise RuntimeError("curve argbest != tracked best epoch")

    # ---- best checkpoint: reload, re-verify, predict all splits -------------
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    best_valid_std = D.predict_scores(model, valid_cache, device)
    best_valid_metrics = M.raw_metrics(best_valid_std, y_valid, target_norm)
    recheck = abs(float(best_valid_metrics["rmse"]) - float(tracker.best_primary))
    if recheck > C.REPRO_TOL:
        raise RuntimeError(f"reloaded best checkpoint Valid RMSE differs by {recheck}")

    train_std = D.predict_scores(model, train_cache, device)
    train_metrics = M.raw_metrics(train_std, y_train, target_norm)

    # Test targets are decoded only now, after best checkpoint selection.
    test_cache = D.load_cache("test")
    src_all = C.source_with_labels("all")
    test_rows = [int(r) for r in src_all["rows"]["test"]]
    if [int(x) for x in np.asarray(test_cache["rows"]).tolist()] != test_rows:
        raise RuntimeError("Test cache rows differ from published split")
    y_test = np.asarray(src_all["y"]["test"], dtype=np.float64)
    test_std = D.predict_scores(model, test_cache, device)
    test_metrics = M.raw_metrics(test_std, y_test, target_norm)

    train_by_row = {int(r): float(v) for r, v in zip(train_rows, train_std)}
    y_train_by_row = {int(r): float(v) for r, v in zip(train_rows, y_train)}
    best_pair_metrics = M.pair_metrics(train_by_row, y_train_by_row, p_train)

    # ---- last epoch: one endpoint evaluation only --------------------------
    model.load_state_dict({k: v.to(device) for k, v in last_state.items()})
    model.eval()
    last_train_std = D.predict_scores(model, train_cache, device)
    last_valid_std = D.predict_scores(model, valid_cache, device)
    last_test_std = D.predict_scores(model, test_cache, device)
    last_train_metrics = M.raw_metrics(last_train_std, y_train, target_norm)
    last_valid_metrics = M.raw_metrics(last_valid_std, y_valid, target_norm)
    last_test_metrics = M.raw_metrics(last_test_std, y_test, target_norm)

    # ---- train diagnostics: endpoints vs non-paired molecules --------------
    endpoint_rows = {int(p["row_a"]) for p in p_train} | {int(p["row_b"]) for p in p_train}
    endpoint_idx = [i for i, r in enumerate(train_rows) if int(r) in endpoint_rows]
    other_idx = [i for i, r in enumerate(train_rows) if int(r) not in endpoint_rows]
    train_diag = {
        "n_p_train": len(p_train),
        "n_endpoint_molecules": len(endpoint_rows),
        "endpoint_coverage": float(len(endpoint_rows) / n_train),
        "initial": {
            "train_all": init_train_metrics,
            "pair": {k: v for k, v in init_pair_metrics.items() if k != "arrays"},
            "endpoint_molecules": M.raw_metrics(init_train_std[endpoint_idx], y_train[endpoint_idx], target_norm),
            "nonpaired_molecules": M.raw_metrics(init_train_std[other_idx], y_train[other_idx], target_norm),
        },
        "best": {
            "train_all": train_metrics,
            "pair": {k: v for k, v in best_pair_metrics.items() if k != "arrays"},
            "endpoint_molecules": M.raw_metrics(train_std[endpoint_idx], y_train[endpoint_idx], target_norm),
            "nonpaired_molecules": M.raw_metrics(train_std[other_idx], y_train[other_idx], target_norm),
        },
        "pair_response_error_decreased": bool(
            best_pair_metrics["ce_mae"] < init_pair_metrics["ce_mae"]
        ),
    }

    # ---- serialise ---------------------------------------------------------
    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    with open(out_dir / "epoch_log.jsonl", "w", encoding="utf-8") as fh:
        for row in curve:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    C.write_json(out_dir / "diagnostics.json", {"steps": diagnostics, "aux_rng_probe": aux_rng_probe})
    C.write_json(out_dir / "train_diagnostics.json", train_diag)

    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"
    common_ckpt = {
        "model": "LCRC-A-edit-supervision",
        "variant": C.VARIANT,
        "variant_name": C.VARIANT_NAME,
        "feature_spec_hash": spec.hash(),
        "dataset": "Lipophilicity",
        "split_public": "random122",
        "seed": int(C.TRAIN_SEED),
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "parameters": params_total,
        "target_norm": dict(target_norm),
        "train_norm_fingerprint": C.DD.PREP_DATA.norm_fingerprint(train_norm),
        "train_rows_sha256": C.row_sha256(train_rows),
        "epochs_run": len(curve),
        "smoke": bool(smoke),
        "init_state_hash": init_hash,
        "full_init_state_hash": full_init_hash,
    }
    torch.save(
        {
            **common_ckpt,
            "state_dict": best_state,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse_raw",
            "best_metric_value": float(tracker.best_primary),
            "state_hash": best_state_hash,
        },
        best_ckpt,
    )
    torch.save(
        {
            **common_ckpt,
            "state_dict": last_state,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse_raw",
            "best_metric_value": float(tracker.best_primary),
            "state_hash": last_state_hash,
        },
        last_ckpt,
    )

    np.savez(
        out_dir / "predictions_best.npz",
        train_rows=np.asarray(train_rows, dtype=np.int64),
        train_y=y_train,
        train_pred_standardised=train_std,
        train_pred_raw=train_std * float(target_norm["scale"]) + float(target_norm["mean"]),
        valid_rows=np.asarray(valid_rows, dtype=np.int64),
        valid_y=y_valid,
        valid_pred_standardised=best_valid_std,
        valid_pred_raw=best_valid_std * float(target_norm["scale"]) + float(target_norm["mean"]),
        test_rows=np.asarray(test_rows, dtype=np.int64),
        test_y=y_test,
        test_pred_standardised=test_std,
        test_pred_raw=test_std * float(target_norm["scale"]) + float(target_norm["mean"]),
    )
    np.savez(
        out_dir / "predictions_last.npz",
        train_rows=np.asarray(train_rows, dtype=np.int64),
        train_y=y_train,
        train_pred_standardised=last_train_std,
        train_pred_raw=last_train_std * float(target_norm["scale"]) + float(target_norm["mean"]),
        valid_rows=np.asarray(valid_rows, dtype=np.int64),
        valid_y=y_valid,
        valid_pred_standardised=last_valid_std,
        valid_pred_raw=last_valid_std * float(target_norm["scale"]) + float(target_norm["mean"]),
        test_rows=np.asarray(test_rows, dtype=np.int64),
        test_y=y_test,
        test_pred_standardised=last_test_std,
        test_pred_raw=last_test_std * float(target_norm["scale"]) + float(target_norm["mean"]),
    )
    np.savez(
        out_dir / "predictions_initial.npz",
        train_rows=np.asarray(train_rows, dtype=np.int64),
        train_y=y_train,
        train_pred_standardised=init_train_std,
        train_pred_raw=init_train_std * float(target_norm["scale"]) + float(target_norm["mean"]),
        valid_rows=np.asarray(valid_rows, dtype=np.int64),
        valid_y=y_valid,
        valid_pred_standardised=init_valid_std,
        valid_pred_raw=init_valid_std * float(target_norm["scale"]) + float(target_norm["mean"]),
    )

    payload: dict[str, Any] = {
        "schema": "lcrc-lipo-edit-supervision-train-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "status": "completed",
        "smoke": bool(smoke),
        "arm": arm,
        "arm_description": C.ARM_DESCRIPTIONS[arm],
        "seed": int(C.TRAIN_SEED),
        "model": {
            "variant": C.VARIANT,
            "variant_name": C.VARIANT_NAME,
            "parameters": params_total,
            "parameters_expected": C.EXPECTED_PARAMS,
            "init_state_hash": init_hash,
            "full_init_state_hash": full_init_hash,
            "shared_init_bitwise_equal_to_full": bool(shared["all_equal"]),
            "dropout_calls_in_run": int(dropout_calls["n"]),
        },
        "data": {
            "n_train": n_train,
            "n_valid": len(valid_rows),
            "n_test": len(test_rows),
            "train_rows_sha256": C.row_sha256(train_rows),
            "valid_rows_sha256": C.row_sha256(valid_rows),
            "test_rows_sha256": C.row_sha256(test_rows),
            "train_cache_fingerprint": train_cache["meta"]["content_fingerprint"],
            "valid_cache_fingerprint": valid_cache["meta"]["content_fingerprint"],
            "test_cache_fingerprint": test_cache["meta"]["content_fingerprint"],
            "target_norm": dict(target_norm),
            "train_norm_fingerprint": C.DD.PREP_DATA.norm_fingerprint(train_norm),
            "test_targets_loaded_after_best_checkpoint": True,
            "test_used_for_selection": False,
        },
        "recipe": {
            "optimizer": C.RECIPE["optimizer"],
            "adam_betas": [float(x) for x in C.RECIPE["adam_betas"]],
            "adam_eps": float(C.RECIPE["adam_eps"]),
            "weight_decay": float(C.RECIPE["weight_decay"]),
            "peak_lr": float(C.RECIPE["peak_lr"]),
            "min_lr": float(C.RECIPE["min_lr"]),
            "batch_size": batch_size,
            "epochs_planned": int(epochs),
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "gradient_clip": float(C.RECIPE["gradient_clip"]),
            "dropout": float(C.RECIPE["dropout"]),
            "lambda": float(C.LAMBDA),
            "aux_pairs_per_step": int(C.AUX_PAIRS),
            "loss": "mean squared standardised error + lambda * frozen auxiliary term",
            "checkpoint_rule": "lowest_valid_rmse__strict_improvement__tie_keeps_earlier",
            "early_stopping": False,
            "amp": False,
            "shuffle_generator_seed": int(C.SHUFFLE_SEED),
            "schedule_numpy_seed": int(C.SCHEDULE_SEED),
        },
        "schedule": {
            "manifest_schema": manifest["schema"],
            "p_train_pairs": int(manifest["p_train"]["n_pairs"]),
            "p_train_pair_ids_sha256": manifest["p_train"]["pair_ids_sha256"],
            "real_pair_indices_sha256": manifest["real_pair_indices"]["sha256"],
            "random_u_sha256": manifest["random_pair_rows"]["u_sha256"],
            "random_v_sha256": manifest["random_pair_rows"]["v_sha256"],
            "batch_order_sha256": manifest["batch_order"]["sha256"],
            "n_steps_redrawn": int(manifest["n_steps_redrawn"]),
            "endpoint_multiset_equal_all_steps": bool(manifest["endpoint_multiset_equal_all_steps"]),
            "random_pairs_with_forbidden_edge": int(manifest["random_pairs_with_forbidden_edge"]),
        },
        "rng": {
            "initial_hash_after_reset": init_rng_hash,
            "epoch1_step_hashes": epoch_rng_hashes,
            "aux_rng_probe": aux_rng_probe,
        },
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse_raw",
            "best_metric_value": float(tracker.best_primary),
            "n_improvements": len(tracker.improvement_epochs),
            "best_state_hash": best_state_hash,
            "last_state_hash": last_state_hash,
            "best_path": C.track_rel(best_ckpt),
            "best_sha256": C.sha256_file(best_ckpt),
            "last_path": C.track_rel(last_ckpt),
            "last_sha256": C.sha256_file(last_ckpt),
            "curve_path": C.track_rel(curve_path),
            "curve_sha256": C.sha256_file(curve_path),
            "curve_argbest_epoch": int(np.argmin(curve_values) + 1),
            "best_valid_recheck_abs_diff": float(recheck),
        },
        "metrics": {
            "initial": {"train": init_train_metrics, "valid": init_valid_metrics},
            "best": {"train": train_metrics, "valid": best_valid_metrics, "test": test_metrics},
            "last": {"train": last_train_metrics, "valid": last_valid_metrics, "test": last_test_metrics},
        },
        "train_diagnostics": train_diag,
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_clip_ratio": float(clip_events / max(len(curve) * steps_per_epoch, 1)),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": int(nonfinite_events),
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "mean_epoch_seconds": float(np.mean([r["train_seconds"] + r["eval_seconds"] for r in curve])),
            "mean_step_seconds": float(wall / max(len(curve) * steps_per_epoch, 1)),
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "device_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
            "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0) if device.type == "cuda" else 0.0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "hostname": platform.node(),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "provenance": {
            "git_commit": C.git_commit(),
            "git_dirty_tracked": C.git_dirty(),
            "code_hashes": C.code_hashes(),
        },
        "predictions": {
            "best": C.track_rel(out_dir / "predictions_best.npz"),
            "last": C.track_rel(out_dir / "predictions_last.npz"),
            "initial": C.track_rel(out_dir / "predictions_initial.npz"),
        },
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{arm}] best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train_rmse={train_metrics['rmse']:.5f} valid_rmse={best_valid_metrics['rmse']:.5f} "
        f"test_rmse={test_metrics['rmse']:.5f} pair_ce_mae={best_pair_metrics['ce_mae']:.5f} "
        f"wall={wall:.1f}s"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="edit-supervision training unit")
    ap.add_argument("--arm", choices=list(C.ARMS), help="A / R / W / N")
    ap.add_argument("--smoke-all", action="store_true", help="run all four arms for 2 epochs in this job")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.smoke_all:
        epochs = 2 if args.epochs is None else int(args.epochs)
        if epochs > 2:
            raise SystemExit("smoke runs may not exceed 2 epochs")
        root = Path(args.out_dir) if args.out_dir else C.RUNS / "smoke"
        for arm in C.ARMS:
            train_arm(arm=arm, device=device, out_dir=root / arm, epochs=epochs, smoke=True)
        return 0
    if not args.arm:
        raise SystemExit("--arm is required unless --smoke-all is used")
    epochs = int(C.RECIPE["epochs"]) if args.epochs is None else int(args.epochs)
    out_dir = Path(args.out_dir) if args.out_dir else C.run_dir(args.arm)
    train_arm(arm=args.arm, device=device, out_dir=out_dir, epochs=epochs, smoke=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
