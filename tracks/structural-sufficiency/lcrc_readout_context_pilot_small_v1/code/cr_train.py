"""The frozen R3 training / every-epoch evaluation loop for arms A, B and C.

One recipe for all three arms (copied from the executed ``sq`` R3 loop, not
re-tuned here):

* seed 0, 360 epochs, fixed (no early stopping), batch 128 with the last
  partial batch kept;
* AdamW peak 3e-4 -> min 1e-5, betas (0.9, 0.999), eps 1e-8, wd 1e-3, no warmup,
  per-optimizer-step cosine ``min + 0.5*(peak-min)*(1+cos(pi*t/T))`` with
  ``t`` starting at 0 and ``T = 360 * ceil(N_train / 128)``;
* MSE on the standardised target, global grad-norm clip 5, FP32, no AMP / EMA /
  SWA / soup / augmentation / resampling / distillation / frozen trunk;
* Train batch order from ``np.random.default_rng(0).permutation`` of the Train
  split-local rows, one permutation per epoch (the historical
  ``shuffle_offset=917000`` path is *not* used);
* every epoch a full ``model.eval()`` + ``no_grad`` pass over Train / Val / Test
  at batch 256, in raw target units, with Test evaluated every epoch
  (``test_policy=always``);
* checkpoint selection is minimum Val RMSE only, strictly smaller replaces,
  equal keeps the earlier epoch; epoch 0 is an initialisation diagnostic and can
  never be selected.

Everything an interrupted run needs is written next to the trajectory every
epoch (``resume_state.pt``), and each trajectory has its own atomic
``status.json`` completion marker keyed by a configuration/init/data fingerprint.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

import cr_common as C
import cr_data as D
import cr_models as M

#: Frozen R3 recipe (mirrors ``dissect_common.RECIPE`` / ``sq_lcrc.run_lcrc``).
RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-3,
    "batch_size": 128,
    "eval_batch_size": 256,
    "dropout": M.DROPOUT,
    "gradient_clip": 5.0,
    "epochs": 360,
    "scheduler": "per_step_cosine",
    "warmup_epochs": 0,
    "amp": False,
    "ema": False,
    "swa": False,
    "soup": False,
    "early_stopping": False,
    "selection": "minimum Val RMSE; strictly smaller replaces, ties keep the earlier epoch",
    "shuffle_generator": "np.random.default_rng(0).permutation over Train split-local rows, one per epoch",
    "smoke_epochs": 2,
}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def results_root() -> "Any":
    return C.RESULTS_DIR


def trajectory_dir(dataset: str, arm: str) -> "Any":
    return C.RESULTS_DIR / dataset / arm


def smoke_dir(dataset: str, arm: str) -> "Any":
    return C.RESULTS_DIR / "_smoke" / dataset / arm


def _status_file(path) -> "Any":
    return path / "status.json"


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def _attach_target(batch: Mapping[str, Any], chunk: np.ndarray, y_std: np.ndarray, torch) -> None:
    """Attach the standardised target for *this* chunk (never a placeholder)."""
    batch["y"] = torch.as_tensor(np.asarray(y_std, dtype=np.float32)[chunk], dtype=torch.float32)
    return None


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def eval_rows(bundle: D.Bundle, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cache-local order, official global row ids, raw targets) in ascending row_id order."""
    order, row_ids = bundle.row_ids_sorted(split)
    return order, row_ids, bundle.targets_raw[split][order]


def evaluate(
    model,
    bundle: D.Bundle,
    split: str,
    device,
    *,
    with_ledger: bool = False,
) -> dict[str, Any]:
    """Full split in ``model.eval()`` mode, raw units, rows in ascending official row_id."""
    import torch

    order, row_ids, targets_raw = eval_rows(bundle, split)
    y_std = (targets_raw - bundle.scaler["mean"]) / bundle.scaler["std"]
    cache = bundle.caches[split]
    model.eval()
    preds: list[np.ndarray] = []
    ledger_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, order.size, int(RECIPE["eval_batch_size"])):
            chunk = order[start : start + int(RECIPE["eval_batch_size"])]
            batch = bundle.collate(cache, chunk)
            _attach_target(batch, chunk, y_std, torch)
            batch = _to_device(batch, device)
            if with_ledger:
                pred, parts = model(batch, return_aux=True)
                ledger_rows.append(_ledger_block(parts, batch, chunk))
            else:
                pred = model(batch)
            preds.append(pred.detach().cpu().numpy().astype(np.float64).reshape(-1))
    pred_std = np.concatenate(preds) if preds else np.zeros(0)
    mean, std = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    pred_raw = pred_std * std + mean
    out = {
        "split": split,
        "row_ids": row_ids,
        "y_raw": targets_raw,
        "pred_std": pred_std,
        "pred_raw": pred_raw,
        "rmse_raw": C.rmse(pred_raw, targets_raw),
        "mae_raw": C.mae(pred_raw, targets_raw),
        "seconds": time.perf_counter() - t0,
        "n_rows": int(row_ids.size),
    }
    if with_ledger:
        out["ledger"] = ledger_rows
    return out


def _ledger_block(
    parts: Mapping[str, Any],
    batch: Mapping[str, Any],
    chunk: np.ndarray,
) -> dict[str, np.ndarray]:
    """The per-batch tensors a component ledger needs (numpy, CPU, float64)."""
    def np_(value):
        return value.detach().cpu().numpy().astype(np.float64)

    block = {
        "atom_batch": batch["atom_batch"].detach().cpu().numpy(),
        "atom_counts": batch["atom_counts"].detach().cpu().numpy(),
        "cache_rows": np.asarray(chunk, dtype=np.int64),
        "n_graphs": np.asarray([int(batch["n_graphs"])]),
        "pred_std": np_(parts["pred_std"]),
    }
    for key in (
        "h", "base_root", "sum_base", "u_sum", "intercept", "local_sum", "rho",
        "u_terms", "z_active", "context_root", "context_sums", "a_values",
        "interaction", "interaction_root", "z_context", "hidden", "contrib_atom",
        "mean_h", "mean_h2", "rho_input",
    ):
        if key in parts:
            block[key] = np_(parts[key])
    return block


# ---------------------------------------------------------------------------
# trajectory identity
# ---------------------------------------------------------------------------


def trajectory_signature(
    bundle: D.Bundle,
    arm: str,
    *,
    hidden: int | None,
    epochs: int,
) -> dict[str, Any]:
    payload = {
        "round_id": C.ROUND_ID,
        "dataset": bundle.dataset,
        "arm": arm,
        "seed": C.TRAIN_SEED,
        "epochs": int(epochs),
        "batch_size": int(RECIPE["batch_size"]),
        "eval_batch_size": int(RECIPE["eval_batch_size"]),
        "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items() if not isinstance(v, dict)},
        "hidden": None if hidden is None else int(hidden),
        "n_active_global": int(bundle.active_global["n_active"]),
        "active_indices_sha256": C.sha256_array(np.asarray(bundle.active_global["active_indices"], dtype=np.int64)),
        "graph_cache_sha256": bundle.payload["cache_sha256"],
        "lipo_raw_sha256": bundle.lipo_raw["cache_sha256"],
        "feature_spec_hash": bundle.extra["spec_hash"],
        "norm_fingerprint": bundle.extra["norm_fingerprint"],
        "target_scaler": {k: bundle.scaler[k] for k in ("mean", "std")},
        "row_id_sha256": {split: C.sha256_array(bundle.global_row_ids[split]) for split in C.SPLITS},
        "init_streams": {
            "canonical": C.INIT_SEED_CANONICAL,
            "global": C.INIT_STREAM_GLOBAL,
            "context": C.INIT_STREAM_CONTEXT,
            "local": C.INIT_STREAM_LOCAL,
        },
        "code_sha256": code_hashes(),
    }
    payload["signature"] = C.sha256_json({k: v for k, v in payload.items() if k != "signature"})
    return payload


def code_hashes() -> dict[str, str]:
    out = {}
    for name in ("cr_common.py", "cr_data.py", "cr_models.py", "cr_train.py"):
        path = C.CODE_DIR / name
        if path.is_file():
            out[name] = C.sha256_file(path)
    for name in ("feature_spec.py", "adapters.py", "molecular_graph.py", "lcrc_data.py"):
        path = C.PREP_CODE / name
        if path.is_file():
            out[f"reused/lcrc_cross_task_preparation_v1/{name}"] = C.sha256_file(path)
    for name in ("dissect_common.py", "dissect_model.py"):
        path = C.DISSECT_CODE / name
        if path.is_file():
            out[f"reused/lcrc_lipo_model_dissection_v1/{name}"] = C.sha256_file(path)
    for name in ("sq_common.py", "sq_data.py", "sq_lcrc.py"):
        path = C.SQL_CODE / name
        if path.is_file():
            out[f"reused/sq-pilot-small-v1/{name}"] = C.sha256_file(path)
    return out


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def run_trajectory(
    bundle: D.Bundle,
    arm: str,
    *,
    arm_spec: M.ArmSpec,
    canonical_state: Mapping[str, Any],
    out_dir,
    device,
    epochs: int | None = None,
    resume: bool = True,
    ledger: C.Ledger | None = None,
    purpose: str = "formal",
) -> dict[str, Any]:
    import torch

    out_dir = C.ensure_dir(out_dir)
    epochs = int(RECIPE["epochs"] if epochs is None else epochs)
    context_indices = [D.field_index(bundle.spec, name) for name in C.CONTEXT_FIELDS]
    signature = trajectory_signature(bundle, arm, hidden=arm_spec.hidden, epochs=epochs)
    sig = signature["signature"]

    done = _status_file(out_dir)
    if resume and done.is_file():
        status = C.read_json(done)
        if status.get("state") == "completed" and status.get("signature") == sig:
            summary = C.read_json(out_dir / "trajectory.json")
            summary["resumed_skip"] = True
            return summary

    # ---- construction (identical canonical trunk for every arm) ----------
    C.set_global_seeds(C.TRAIN_SEED)
    model = M.build_arm(
        arm,
        bundle.spec,
        bundle.norm,
        canonical_state,
        n_active=bundle.active_global["n_active"],
        hidden=arm_spec.hidden,
        active_indices=bundle.active_global["active_indices"],
        context_indices=context_indices,
        device="cpu",
    )
    params = M.parameter_report(model)
    trunk_hash = C.sha256_state(
        {k[len(M.ARM_TRUNK_PREFIX):]: v for k, v in model.state_dict().items() if k.startswith(M.ARM_TRUNK_PREFIX)}
    )
    init_audit = {
        "arm": arm,
        "dataset": bundle.dataset,
        "canonical_state_hashes": M.state_hashes(canonical_state),
        "arm_trunk_hash": trunk_hash,
        "arm_trunk_matches_canonical": trunk_hash == M.state_hashes(canonical_state)["trunk_hash"],
        "parameters": params,
        "budget": arm_spec.budget,
        "hidden": arm_spec.hidden,
        "n_active_global": int(bundle.active_global["n_active"]),
        "init_streams": signature["init_streams"],
    }
    if not init_audit["arm_trunk_matches_canonical"]:
        raise RuntimeError(f"{bundle.dataset}/{arm}: trunk init does not match the canonical C1 state")

    state = {
        "epoch": 0,
        "global_step": 0,
        "curves": [],
        "preds": {split: [] for split in C.SPLITS},
        "best": {"valid_rmse_raw": float("inf"), "epoch": None},
        "best_state": None,
        "final_state": None,
        "batch_order_hashes": [],
        "first_batch_rows": [],
        "train_seconds": 0.0,
        "eval_seconds": 0.0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    resume_path = out_dir / "resume_state.pt"
    if resume and resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("signature") == sig:
            model.load_state_dict(saved["model_state"], strict=True)
            state = saved["state"]
            optimizer_state = saved["optimizer_state"]
        else:
            raise RuntimeError(
                f"{bundle.dataset}/{arm}: resume_state.pt signature mismatch "
                f"({saved.get('signature')} != {sig}); refusing to continue a different trajectory"
            )
    else:
        optimizer_state = None

    model = model.to(device)
    train_rows = np.arange(int(bundle.caches["train"]["n_molecules"]), dtype=np.int64)
    if train_rows.size != int(bundle.global_row_ids["train"].size):
        raise RuntimeError("Train split size mismatch")
    batch_size = int(RECIPE["batch_size"])
    steps_per_epoch = int(np.ceil(train_rows.size / batch_size))
    total_steps = epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(RECIPE["peak_lr"]),
        betas=tuple(RECIPE["adam_betas"]),
        eps=float(RECIPE["adam_eps"]),
        weight_decay=float(RECIPE["weight_decay"]),
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # ---- one RNG reset per trajectory: identical dropout stream per arm --
    if state["epoch"] == 0:
        rng = np.random.default_rng(C.TRAIN_SEED)
        seeds = C.set_global_seeds(C.TRAIN_SEED)
        state["rng_hash_at_start"] = C.rng_state_hash()
        state["seed_report"] = seeds
    else:
        C.restore_rng_state(state["rng_payload"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["np_rng_state"]

    device_label = str(device)
    with C.stage_timer(ledger, f"train_{bundle.dataset}_{arm}", device=device_label, purpose=purpose):
        # ---- epoch 0 diagnostic (never selectable) ----------------------
        if state["epoch"] == 0 and not (out_dir / "init_metrics.json").is_file():
            init_metrics = {}
            for split in C.SPLITS:
                out = evaluate(model, bundle, split, device)
                init_metrics[split] = {
                    "rmse_raw": out["rmse_raw"],
                    "mae_raw": out["mae_raw"],
                    "pred_raw": out["pred_raw"],
                }
                C.atomic_write_npz(
                    out_dir / f"init_pred_{split}.npz",
                    {"row_id": out["row_ids"], "y_raw": out["y_raw"], "pred_raw": out["pred_raw"]},
                )
            C.atomic_write_json(
                out_dir / "init_metrics.json",
                {split: {"rmse_raw": v["rmse_raw"], "mae_raw": v["mae_raw"]} for split, v in init_metrics.items()},
            )
            state["init_metrics"] = {s: {"rmse_raw": v["rmse_raw"], "mae_raw": v["mae_raw"]} for s, v in init_metrics.items()}

        for epoch in range(state["epoch"] + 1, epochs + 1):
            t_epoch = time.perf_counter()
            model.train()
            order = rng.permutation(train_rows)
            state["batch_order_hashes"].append(C.sha256_array(order))
            if not state["first_batch_rows"]:
                state["first_batch_rows"] = [int(v) for v in order[:batch_size].tolist()]
            t_train = time.perf_counter()
            loss_sum = 0.0
            loss_weight = 0
            batch_losses: list[float] = []
            for start in range(0, order.size, batch_size):
                chunk = order[start : start + batch_size]
                batch = bundle.collate(bundle.caches["train"], chunk)
                _attach_target(batch, chunk, bundle.y_std["train"], torch)
                batch = _to_device(batch, device)
                lr = float(RECIPE["min_lr"]) + 0.5 * (float(RECIPE["peak_lr"]) - float(RECIPE["min_lr"])) * (
                    1.0 + np.cos(np.pi * state["global_step"] / max(1, total_steps))
                )
                optimizer.param_groups[0]["lr"] = float(lr)
                state["global_step"] += 1
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch)
                target = batch["y"].view(-1)
                loss = torch.nn.functional.mse_loss(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(RECIPE["gradient_clip"]))
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
                loss_sum += float(loss.detach().cpu()) * int(chunk.size)
                loss_weight += int(chunk.size)
            train_seconds = time.perf_counter() - t_train
            state["train_seconds"] += train_seconds

            t_eval = time.perf_counter()
            row: dict[str, Any] = {
                "epoch": int(epoch),
                "update": int(state["global_step"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss_std": float(loss_sum / max(1, loss_weight)),
                "train_loss_std_mean_of_batches": float(np.mean(batch_losses)),
            }
            predictions: dict[str, dict[str, Any]] = {}
            for split in C.SPLITS:
                out = evaluate(model, bundle, split, device)
                predictions[split] = out
                row[f"{split}_rmse_raw"] = out["rmse_raw"]
                row[f"{split}_mae_raw"] = out["mae_raw"]
                state["preds"][split].append(out["pred_raw"].astype(np.float32))
            eval_seconds = time.perf_counter() - t_eval
            state["eval_seconds"] += eval_seconds
            row["train_seconds"] = float(train_seconds)
            row["eval_seconds"] = float(eval_seconds)
            row["elapsed_s"] = float(time.perf_counter() - t_epoch)
            state["curves"].append(row)

            if row["valid_rmse_raw"] < state["best"]["valid_rmse_raw"]:
                state["best"] = {
                    "valid_rmse_raw": float(row["valid_rmse_raw"]),
                    "epoch": int(epoch),
                    "test_rmse_raw": float(row["test_rmse_raw"]),
                    "train_rmse_raw": float(row["train_rmse_raw"]),
                }
                state["best_state"] = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            state["final_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state["epoch"] = int(epoch)
            state["np_rng_state"] = rng.bit_generator.state
            state["rng_payload"] = C.capture_rng_state()

            # ---- durable per-epoch checkpoint (atomic) -------------------
            C.atomic_torch_save(
                resume_path,
                {
                    "signature": sig,
                    "state": state,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                },
            )
            C.write_csv(
                out_dir / "curve.csv",
                state["curves"],
                columns=[
                    "epoch", "update", "lr", "train_loss_std", "train_loss_std_mean_of_batches",
                    "train_rmse_raw", "valid_rmse_raw", "test_rmse_raw",
                    "train_mae_raw", "valid_mae_raw", "test_mae_raw",
                    "train_seconds", "eval_seconds", "elapsed_s",
                ],
            )

    # ---- finalise -------------------------------------------------------
    best_epoch = int(state["best"]["epoch"])
    best_metrics = state["curves"][best_epoch - 1]
    _write_checkpoint(
        out_dir / "checkpoint_best_val.pt",
        state_dict=state["best_state"],
        dataset=bundle.dataset,
        arm=arm,
        epoch=best_epoch,
        metrics=best_metrics,
        bundle=bundle,
        signature=signature,
        init_audit=init_audit,
    )
    _write_checkpoint(
        out_dir / "checkpoint_final.pt",
        state_dict=state["final_state"],
        dataset=bundle.dataset,
        arm=arm,
        epoch=int(state["epoch"]),
        metrics=state["curves"][-1],
        bundle=bundle,
        signature=signature,
        init_audit=init_audit,
    )

    # per-epoch prediction matrices (rows in ascending official row_id order)
    for split in C.SPLITS:
        order, row_ids, targets_raw = eval_rows(bundle, split)
        matrix = np.stack(state["preds"][split], axis=0).astype(np.float32)
        C.atomic_write_npz(
            out_dir / f"pred_matrix_{split}.npz",
            {"pred_raw": matrix, "row_id": row_ids, "y_raw": targets_raw.astype(np.float32), "epoch": np.arange(1, matrix.shape[0] + 1)},
        )

    # best / final per-sample predictions for the three splits
    model.load_state_dict({k: v.to(device) for k, v in state["best_state"].items()}, strict=True)
    best_predictions = {split: evaluate(model, bundle, split, device) for split in C.SPLITS}
    t_infer = time.perf_counter()
    _ = {split: evaluate(model, bundle, split, device) for split in C.SPLITS}
    inference_seconds = time.perf_counter() - t_infer
    model.load_state_dict({k: v.to(device) for k, v in state["final_state"].items()}, strict=True)
    final_predictions = {split: evaluate(model, bundle, split, device) for split in C.SPLITS}
    for tag, bundle_pred in (("best", best_predictions), ("final", final_predictions)):
        arrays = {}
        for split in C.SPLITS:
            out = bundle_pred[split]
            # float64 so the analysis never re-rounds a reported prediction
            arrays[f"{split}_pred_raw"] = out["pred_raw"].astype(np.float64)
            arrays[f"{split}_pred_std"] = out["pred_std"].astype(np.float64)
            arrays[f"{split}_y_raw"] = out["y_raw"].astype(np.float64)
            arrays[f"{split}_row_id"] = out["row_ids"]
        C.atomic_write_npz(out_dir / f"{tag}_predictions.npz", arrays)

    last20 = {
        f"mean_last20_{split}_rmse_raw": float(np.mean([r[f"{split}_rmse_raw"] for r in state["curves"][-20:]]))
        for split in C.SPLITS
    }
    last20.update(
        {
            f"mean_last20_{split}_mae_raw": float(np.mean([r[f"{split}_mae_raw"] for r in state["curves"][-20:]]))
            for split in C.SPLITS
        }
    )
    summary = {
        "round_id": C.ROUND_ID,
        "dataset": bundle.dataset,
        "arm": arm,
        "purpose": purpose,
        "status": "complete",
        "signature": sig,
        "signature_payload": signature,
        "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items()},
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_updates": int(state["global_step"]),
        "n_train": int(train_rows.size),
        "n_valid": int(bundle.global_row_ids["valid"].size),
        "n_test": int(bundle.global_row_ids["test"].size),
        "test_eval_count": int(len(state["curves"])),
        "test_policy": "always",
        "n_parameters": int(params["total"]),
        "readout_parameters": int(params["readout"]),
        "trunk_parameters": int(params["trunk"]),
        "budget": arm_spec.budget,
        "init": {k: v for k, v in init_audit.items() if k != "parameters"},
        "parameter_breakdown": params["breakdown"],
        "init_metrics": state.get("init_metrics"),
        "curves": state["curves"],
        "selected": {
            "epoch": best_epoch,
            **{k: best_metrics[k] for k in best_metrics if k.endswith("_rmse_raw") or k.endswith("_mae_raw")},
            "lr": best_metrics["lr"],
            "update": best_metrics["update"],
        },
        "final": {
            "epoch": int(state["epoch"]),
            **{k: state["curves"][-1][k] for k in state["curves"][-1] if k.endswith("_rmse_raw") or k.endswith("_mae_raw")},
        },
        "last20": last20,
        "best_val_history": {
            "epoch": best_epoch,
            "valid_rmse_raw": float(state["best"]["valid_rmse_raw"]),
        },
        "batch_order_sha256": C.sha256_text("".join(state["batch_order_hashes"])),
        "batch_order_hashes_first3": state["batch_order_hashes"][:3],
        "first_batch_rows": state["first_batch_rows"],
        "rng_hash_at_start": state.get("rng_hash_at_start"),
        "train_seconds": float(state["train_seconds"]),
        "eval_seconds_total": float(state["eval_seconds"]),
        "inference_seconds_full_three_splits": float(inference_seconds),
        "checkpoint_hashes": {
            "checkpoint_best_val.pt": C.sha256_file(out_dir / "checkpoint_best_val.pt"),
            "checkpoint_final.pt": C.sha256_file(out_dir / "checkpoint_final.pt"),
        },
        "environment": C.environment_snapshot(),
        "device": str(device),
        "started_at": state.get("started_at"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    C.atomic_write_json(out_dir / "trajectory.json", summary)
    C.atomic_write_json(
        _status_file(out_dir),
        {
            "state": "completed",
            "signature": sig,
            "dataset": bundle.dataset,
            "arm": arm,
            "epochs": epochs,
            "selected_epoch": best_epoch,
            "valid_rmse_raw": float(state["best"]["valid_rmse_raw"]),
            "test_rmse_raw": float(best_metrics["test_rmse_raw"]),
            "finished_at": summary["finished_at"],
        },
    )
    # resume bookkeeping is no longer needed once the trajectory is complete
    if resume_path.is_file():
        resume_path.unlink()
    return summary


def _write_checkpoint(
    path,
    *,
    state_dict: Mapping[str, Any],
    dataset: str,
    arm: str,
    epoch: int,
    metrics: Mapping[str, Any],
    bundle: D.Bundle,
    signature: Mapping[str, Any],
    init_audit: Mapping[str, Any],
) -> None:
    payload = {
        "round_id": C.ROUND_ID,
        "dataset": dataset,
        "arm": arm,
        "epoch": int(epoch),
        "state_dict": {k: v.detach().cpu().clone() for k, v in state_dict.items()},
        "metrics": dict(metrics),
        "target_scaler": dict(bundle.scaler),
        "norm": {
            "global_mean": np.asarray(bundle.norm["global_mean"]).tolist(),
            "global_scale": np.asarray(bundle.norm["global_scale"]).tolist(),
            "scalar_mean": np.asarray(bundle.norm["scalar_mean"]).tolist(),
            "scalar_scale": np.asarray(bundle.norm["scalar_scale"]).tolist(),
            "fingerprint": bundle.extra["norm_fingerprint"],
        },
        "active_global_indices": list(bundle.active_global["active_indices"]),
        "context_fields": list(C.CONTEXT_FIELDS),
        "context_indices": [D.field_index(bundle.spec, name) for name in C.CONTEXT_FIELDS],
        "signature": signature["signature"],
        "init_audit": dict(init_audit),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    C.atomic_torch_save(path, payload)


def load_arm_from_checkpoint(bundle: D.Bundle, checkpoint_path, device="cpu", *, verify: bool = True):
    """Rebuild one arm from its checkpoint and verify the recorded trunk hash."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arm = payload["arm"]
    context_indices = [D.field_index(bundle.spec, name) for name in C.CONTEXT_FIELDS]
    n_active = int(bundle.active_global["n_active"])
    hidden = None if arm == "A" else int(payload["init_audit"]["hidden"])
    if arm != "A" and payload["init_audit"]["n_active_global"] != n_active:
        raise RuntimeError(f"checkpoint active-global count {payload['init_audit']['n_active_global']} != {n_active}")
    # Build exactly as training did: the canonical C1 state supplies the shared
    # trunk (and, for arm A, the frozen f/rho head); the checkpoint state then
    # overwrites every parameter, so no readout is silently left at init.
    model = M.build_arm(
        arm,
        bundle.spec,
        bundle.norm,
        M.canonical_c1_state(bundle.spec, bundle.norm),
        n_active=n_active,
        hidden=hidden,
        active_indices=bundle.active_global["active_indices"],
        context_indices=context_indices,
        device="cpu",
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    if verify:
        reloaded = C.sha256_state(model.state_dict())
        stored = C.sha256_state(payload["state_dict"])
        if reloaded != stored:
            raise RuntimeError(f"{checkpoint_path}: reloaded state hash {reloaded} != stored {stored}")
        payload["reloaded_state_hash"] = reloaded
        payload["reloaded_matches_stored"] = True
    return model, payload


def recipe_record() -> dict[str, Any]:
    """The frozen recipe as a JSON-safe record for the protocol freeze."""
    out: dict[str, Any] = {}
    for key, value in RECIPE.items():
        out[key] = list(value) if isinstance(value, tuple) else value
    out["test_policy"] = "always"
    out["target_unit_for_reporting"] = "raw (de-standardised once)"
    out["selection_split"] = "valid"
    out["epoch0_role"] = "initialisation diagnostic; can never be selected"
    out["dropout_module_count"] = 1
    return out
