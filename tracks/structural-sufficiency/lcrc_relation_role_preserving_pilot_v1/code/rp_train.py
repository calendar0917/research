"""The frozen Rf training / every-epoch evaluation loop for RR_REAL and RR_CONTROL.

Recipe ``Rf`` copied from ``gnn_route_clarification_8h_v1`` (``gr_common.RECIPES``):

* seed 0, 360 epochs, no early stopping, batch 128 (last partial batch kept);
* AdamW, betas (0.9, 0.999), eps 1e-8, peak LR 1e-3, weight decay 1e-5,
  5 warmup epochs at ``peak * (0.1 + 0.9 * step / warmup_steps)`` and then a
  per-optimizer-step cosine from ``peak`` to 1e-5 over the remaining steps;
* MSE on the standardised target, global gradient-norm clip 5, FP32, no AMP /
  EMA / SWA / soup / augmentation / resampling / distillation;
* Train batch order from ``np.random.default_rng(0).permutation`` of the Train
  split-local rows, one permutation per epoch;
* every epoch a full ``model.eval()`` + ``no_grad`` pass over Train / Valid / Test
  in raw target units (Test is ``DEVELOPMENT_TEST``, evaluated every epoch and
  never used for selection);
* selection: minimum Valid RMSE, strictly smaller replaces, an exact tie keeps
  the earlier epoch; epoch 0 (initialisation) is never selectable.

The only round-specific additions are the role-gamma diagnostics written into
``curve.csv`` (so the learned role scale is auditable epoch by epoch) and the
atom-level contributions written for the frozen best/final states.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

import rp_common as C
import rp_models as M

RECIPE = C.RECIPE
RESULTS_DIR = C.RESULTS_DIR


def steps_per_epoch(n_train: int, batch_size: int | None = None) -> int:
    size = int(batch_size or RECIPE["batch_size"])
    return int(np.ceil(int(n_train) / size))


def total_steps(n_train: int, epochs: int | None = None) -> int:
    epochs = int(RECIPE["epochs"] if epochs is None else epochs)
    return epochs * steps_per_epoch(n_train)


def lr_at(step: int, n_train: int, epochs: int | None = None) -> float:
    """Exact frozen Rf schedule; ``step`` is the 0-based optimizer step."""
    peak = float(RECIPE["peak_lr"])
    min_lr = float(RECIPE["min_lr"])
    spe = steps_per_epoch(n_train)
    total = int(RECIPE["epochs"] if epochs is None else epochs) * spe
    warm = int(RECIPE["warmup_epochs"]) * spe
    if warm > 0 and step < warm:
        return peak * (0.1 + 0.9 * float(step) / float(warm))
    start = step - warm if warm > 0 else step
    span = max(1, total - warm)
    progress = min(1.0, max(0.0, float(start) / float(span)))
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# batching / evaluation
# ---------------------------------------------------------------------------


def _attach_target(batch: Mapping[str, Any], chunk: np.ndarray, y_std: np.ndarray) -> None:
    import torch

    batch["y"] = torch.as_tensor(np.asarray(y_std, dtype=np.float32)[chunk], dtype=torch.float32)


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}


def eval_rows(bundle, task: str, split: str):
    base = bundle.cr
    order, row_ids = base.row_ids_sorted(split)
    return order, row_ids, np.asarray(bundle.targets(task, split), dtype=np.float64)[order]


def evaluate(
    model,
    bundle,
    task: str,
    split: str,
    device,
    *,
    with_atom_contrib: bool = False,
    gammas: Sequence[Any] | None = None,
    role_key: str | None = None,
) -> dict[str, Any]:
    import torch

    order, row_ids, targets_raw = eval_rows(bundle, task, split)
    scaler = bundle.scaler(task)
    y_std = (targets_raw - scaler["mean"]) / scaler["std"]
    model.eval()
    preds: list[np.ndarray] = []
    atom_rows: list[int] = []
    atom_vals: list[np.ndarray] = []
    ledgers: list[dict[str, np.ndarray]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, order.size, int(RECIPE["eval_batch_size"])):
            chunk = order[start : start + int(RECIPE["eval_batch_size"])]
            batch = bundle.collate(split, chunk)
            _attach_target(batch, chunk, y_std)
            batch = _to_device(batch, device)
            if with_atom_contrib:
                pred, parts = model(batch, role_key=role_key, gammas=gammas, return_aux=True)
                ledgers.append(_ledger_block(parts, batch, chunk, scaler))
                rows = np.asarray(chunk, dtype=np.int64).repeat(
                    np.asarray(batch["atom_counts"].detach().cpu().numpy(), dtype=np.int64)
                )
                atom_rows.append(rows)
                atom_vals.append(
                    (parts["contrib_atom"].detach().cpu().numpy().astype(np.float64)) * float(scaler["std"])
                )
            else:
                pred = model(batch, role_key=role_key, gammas=gammas)
            preds.append(pred.detach().cpu().numpy().astype(np.float64).reshape(-1))
    pred_std = np.concatenate(preds) if preds else np.zeros(0)
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    out = {
        "split": split,
        "task": task,
        "row_ids": row_ids,
        "y_raw": targets_raw,
        "pred_std": pred_std,
        "pred_raw": pred_raw,
        "rmse_raw": C.rmse(pred_raw, targets_raw),
        "mae_raw": C.mae(pred_raw, targets_raw),
        "seconds": time.perf_counter() - t0,
        "n_rows": int(row_ids.size),
    }
    if with_atom_contrib:
        out["ledger"] = merge_ledgers(ledgers)
        out["atom_rows"] = np.concatenate(atom_rows) if atom_rows else np.zeros(0, np.int64)
        out["atom_contrib_raw"] = np.concatenate(atom_vals) if atom_vals else np.zeros(0)
    return out


def _np(value):
    return value.detach().cpu().numpy().astype(np.float64)


def _ledger_block(parts: Mapping[str, Any], batch: Mapping[str, Any], chunk: np.ndarray, scaler) -> dict[str, Any]:
    std = float(scaler["std"])
    mean = float(scaler["mean"])
    return {
        "cache_rows": np.asarray(chunk, dtype=np.int64),
        "n_graphs": np.asarray([int(batch["n_graphs"])]),
        "pred_raw": _np(parts["pred_std"]) * std + mean,
        "local_sum_raw": _np(parts["local_sum"]) * std,
        "rho_raw": _np(parts["rho"]) * std,
        "n_atoms": np.asarray(batch["atom_counts"].detach().cpu().numpy(), dtype=np.int64),
        "atom_counts": np.asarray(batch["atom_counts"].detach().cpu().numpy(), dtype=np.int64),
        "gamma_internal": _np(parts["gamma_internal"]),
        "gamma_cross": _np(parts["gamma_cross"]),
    }


def merge_ledgers(blocks: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    rows: list[int] = []
    pred: list[np.ndarray] = []
    local_sum: list[np.ndarray] = []
    rho: list[np.ndarray] = []
    n_atoms: list[np.ndarray] = []
    for block in blocks:
        rows.extend(int(v) for v in block["cache_rows"].tolist())
        pred.append(block["pred_raw"])
        local_sum.append(block["local_sum_raw"])
        rho.append(block["rho_raw"])
        n_atoms.append(block["n_atoms"])
    return {
        "cache_rows": np.asarray(rows, dtype=np.int64),
        "pred_raw": np.concatenate(pred) if pred else np.zeros(0),
        "local_sum_raw": np.concatenate(local_sum) if local_sum else np.zeros(0),
        "rho_raw": np.concatenate(rho) if rho else np.zeros(0),
        "n_atoms": np.concatenate(n_atoms) if n_atoms else np.zeros(0, np.int64),
    }


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def schedule_audit(n_train: int | None = None) -> dict[str, Any]:
    """Compare the local Rf schedule with the shared frozen implementation."""
    n_train = int(C.EXPECTED_SPLIT_SIZES["train"] if n_train is None else n_train)
    steps = int(total_steps(n_train))
    ours = np.asarray([lr_at(step, n_train) for step in range(steps)], dtype=np.float64)
    shared = _shared_lr(n_train, steps)
    if shared is None:
        return {"n_steps": steps, "max_abs_diff": None, "reference": "unavailable"}
    return {
        "n_steps": steps,
        "max_abs_diff": float(np.max(np.abs(ours - shared))),
        "reference": "tracks/structural-sufficiency/gnn_route_clarification_8h_v1/code/gr_common.py:lr_at('Rf', ...)",
        "peak_lr": float(ours.max()),
        "first_step_lr": float(ours[0]),
        "last_step_lr": float(ours[-1]),
    }


def _shared_lr(n_train: int, steps: int) -> np.ndarray | None:
    path = C.ROUTE_DIR / "code/gr_common.py"
    if not path.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("_rp_gr_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return np.asarray(
        [module.lr_at("Rf", step, n_train, epochs=int(RECIPE["epochs"])) for step in range(steps)],
        dtype=np.float64,
    )


def code_hashes() -> dict[str, str]:
    out = {}
    for name in ("rp_common.py", "rp_data.py", "rp_models.py", "rp_train.py"):
        path = C.CODE_DIR / name
        if path.is_file():
            out[name] = C.sha256_file(path)
    for name in ("rr_partition.py", "rr_routing.py", "rr_data.py"):
        path = C.PRED_CODE / name
        if path.is_file():
            out[f"reused/lcrc_relation_interface_routing_pilot_v1/{name}"] = C.sha256_file(path)
    for name in ("cr_models.py", "cr_data.py", "cr_common.py"):
        path = C.RC_CODE / name
        if path.is_file():
            out[f"reused/lcrc_readout_context_pilot_small_v1/{name}"] = C.sha256_file(path)
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
            out[f"reused/structural-query-learning/{name}"] = C.sha256_file(path)
    return out


def trajectory_signature(bundle, task: str, variant: str, *, epochs: int) -> dict[str, Any]:
    payload = {
        "round_id": C.ROUND_ID,
        "dataset": bundle.dataset,
        "task": task,
        "variant": variant,
        "role_key": C.MODEL_ROLE_KEY[variant],
        "role_source": C.MODEL_ROLE_SOURCE[variant],
        "control_salt": C.CONTROL_SALT,
        "seed": C.TRAIN_SEED,
        "epochs": int(epochs),
        "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items()},
        "routing_fingerprints": bundle.routing_fingerprints,
        "routing_cache_sha256": dict(C.EXPECTED_ROUTING_CACHE_SHA256),
        "control_table_sha256": bundle.extra["control_table_sha256"],
        "target_scaler": bundle.scaler(task),
        "target_sha256": {split: C.sha256_array(np.asarray(bundle.targets(task, split))) for split in C.SPLITS},
        "row_id_sha256": {split: C.sha256_array(bundle.global_row_ids[split]) for split in C.SPLITS},
        "graph_cache_sha256": bundle.extra["graph_cache_sha256"],
        "lipo_raw_sha256": bundle.extra["lipo_raw_sha256"],
        "feature_spec_hash": bundle.extra["spec_hash"],
        "norm_fingerprint": bundle.extra["norm_fingerprint"],
        "init_streams": {"canonical": C.INIT_SEED_CANONICAL},
        "code_sha256": code_hashes(),
        "parameter_counts": M.expected_parameter_counts(int(bundle.spec.global_dim)),
    }
    payload["signature"] = C.sha256_json({k: v for k, v in payload.items() if k != "signature"})
    return payload


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def _gamma_stats(state_dict: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("gamma_internal", "gamma_cross"):
        value = state_dict[name].detach().cpu().numpy().astype(np.float64)
        out[f"{name}_mean"] = float(value.mean())
        out[f"{name}_std"] = float(value.std())
        out[f"{name}_min"] = float(value.min())
        out[f"{name}_max"] = float(value.max())
        out[f"{name}_mean_abs_dev_from_one"] = float(np.abs(value - 1.0).mean())
        out[f"{name}_max_abs_dev_from_one"] = float(np.abs(value - 1.0).max())
    out["gamma_mean_abs_dev_from_one"] = 0.5 * (
        out["gamma_internal_mean_abs_dev_from_one"] + out["gamma_cross_mean_abs_dev_from_one"]
    )
    return out


def run_trajectory(
    bundle,
    task: str,
    variant: str,
    *,
    out_dir,
    device,
    epochs: int | None = None,
    resume: bool = True,
    ledger=None,
    purpose: str = "formal",
) -> dict[str, Any]:
    import torch

    out_dir = C.ensure_dir(out_dir)
    epochs = int(RECIPE["epochs"] if epochs is None else epochs)
    signature = trajectory_signature(bundle, task, variant, epochs=epochs)
    sig = signature["signature"]

    status_file = out_dir / "status.json"
    if resume and status_file.is_file():
        status = C.read_json(status_file)
        if status.get("state") == "completed" and status.get("signature") == sig:
            summary = C.read_json(out_dir / "trajectory.json")
            summary["resumed_skip"] = True
            return summary

    canonical = M.canonical_state(bundle.spec, bundle.norm)
    model = M.build_model(variant, bundle.spec, bundle.norm, canonical, device="cpu")
    params = model.parameter_report()
    c1_state = model.canonical_parameters()
    trunk_hash = C.sha256_state(
        {k[len(M.ARM_PREFIX) :]: v for k, v in model.state_dict().items() if k.startswith(M.ARM_PREFIX)}
    )
    expected_trunk = C.sha256_state(canonical, keys=M.CM.trunk_keys(canonical))
    # model keys carry the arm prefix (``trunk.*``); ``canonical`` is already
    # prefix-stripped, so the two sides are compared in the canonical namespace
    canonical_c1 = {
        (k[len(M.ARM_PREFIX):] if k.startswith(M.ARM_PREFIX) else k): canonical[k[len(M.ARM_PREFIX):] if k.startswith(M.ARM_PREFIX) else k]
        for k in c1_state
    }
    model_c1 = {
        (k[len(M.ARM_PREFIX):] if k.startswith(M.ARM_PREFIX) else k): v for k, v in c1_state.items()
    }
    gamma_state = {k: v for k, v in model.state_dict().items() if k.startswith("gamma_")}
    init_audit = {
        "variant": variant,
        "task": task,
        "canonical_state_hashes": M.CM.state_hashes(canonical),
        "trunk_hash": trunk_hash,
        "trunk_matches_canonical": trunk_hash == expected_trunk,
        "parameters": params,
        "expected_parameter_counts": M.expected_parameter_counts(int(bundle.spec.global_dim)),
        "c1_part_hash": C.sha256_state(model_c1),
        "c1_part_matches_canonical": C.sha256_state(model_c1) == C.sha256_state(canonical_c1),
        "gamma_state_hash": C.sha256_state(gamma_state),
        "gamma_all_ones": bool(all(bool((v == 1).all()) for v in gamma_state.values())),
        "init_streams": signature["init_streams"],
    }
    if not init_audit["trunk_matches_canonical"]:
        raise RuntimeError(f"{task}/{variant}: trunk init does not match the canonical C1 state")
    if not init_audit["c1_part_matches_canonical"]:
        raise RuntimeError(f"{task}/{variant}: non-role parameters do not match the canonical C1 state")
    if not init_audit["gamma_all_ones"]:
        raise RuntimeError(f"{task}/{variant}: role gammas are not initialised to one")

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
        if saved.get("signature") != sig:
            raise RuntimeError(
                f"{task}/{variant}: resume_state.pt signature mismatch; refusing to continue a different trajectory"
            )
        model.load_state_dict(saved["model_state"], strict=True)
        state = saved["state"]
        optimizer_state = saved["optimizer_state"]
    else:
        optimizer_state = None

    model = model.to(device)
    train_rows = np.arange(int(bundle.caches["train"]["n_molecules"]), dtype=np.int64)
    batch_size = int(RECIPE["batch_size"])
    spe = steps_per_epoch(train_rows.size, batch_size)
    n_steps = epochs * spe
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(RECIPE["peak_lr"]),
        betas=tuple(RECIPE["adam_betas"]),
        eps=float(RECIPE["adam_eps"]),
        weight_decay=float(RECIPE["weight_decay"]),
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if state["epoch"] == 0:
        rng = np.random.default_rng(C.TRAIN_SEED)
        seeds = C.set_global_seeds(C.TRAIN_SEED)
        state["rng_hash_at_start"] = C.rng_state_hash()
        state["seed_report"] = seeds
    else:
        C.restore_rng_state(state["rng_payload"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["np_rng_state"]

    with C.stage_timer(ledger, f"train_{task}_{variant}", device=str(device), purpose=purpose):
        if state["epoch"] == 0 and not (out_dir / "init_metrics.json").is_file():
            init_metrics = {}
            for split in C.SPLITS:
                out = evaluate(model, bundle, task, split, device)
                init_metrics[split] = {"rmse_raw": out["rmse_raw"], "mae_raw": out["mae_raw"]}
                C.atomic_write_npz(
                    out_dir / f"init_pred_{split}.npz",
                    {"row_id": out["row_ids"], "y_raw": out["y_raw"], "pred_raw": out["pred_raw"]},
                )
            C.atomic_write_json(out_dir / "init_metrics.json", init_metrics)
            state["init_metrics"] = init_metrics

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
                batch = bundle.collate("train", chunk)
                _attach_target(batch, chunk, bundle.target_std(task, "train"))
                batch = _to_device(batch, device)
                lr = lr_at(state["global_step"], train_rows.size, epochs)
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
            for split in C.SPLITS:
                out = evaluate(model, bundle, task, split, device)
                row[f"{split}_rmse_raw"] = out["rmse_raw"]
                row[f"{split}_mae_raw"] = out["mae_raw"]
                state["preds"][split].append(out["pred_raw"].astype(np.float32))
            eval_seconds = time.perf_counter() - t_eval
            state["eval_seconds"] += eval_seconds
            row["train_seconds"] = float(train_seconds)
            row["eval_seconds"] = float(eval_seconds)
            row["elapsed_s"] = float(time.perf_counter() - t_epoch)
            row.update(_gamma_stats(dict(model.named_parameters())))
            state["curves"].append(row)

            if row["valid_rmse_raw"] < state["best"]["valid_rmse_raw"]:
                state["best"] = {
                    "valid_rmse_raw": float(row["valid_rmse_raw"]),
                    "epoch": int(epoch),
                    "test_rmse_raw": float(row["test_rmse_raw"]),
                    "train_rmse_raw": float(row["train_rmse_raw"]),
                }
                state["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state["final_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state["epoch"] = int(epoch)
            state["np_rng_state"] = rng.bit_generator.state
            state["rng_payload"] = C.capture_rng_state()

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
                    "gamma_internal_mean", "gamma_cross_mean",
                    "gamma_internal_mean_abs_dev_from_one", "gamma_cross_mean_abs_dev_from_one",
                    "gamma_mean_abs_dev_from_one",
                    "train_seconds", "eval_seconds", "elapsed_s",
                ],
            )

    best_epoch = int(state["best"]["epoch"])
    best_metrics = state["curves"][best_epoch - 1]
    _write_checkpoint(
        out_dir / "checkpoint_best_val.pt",
        state_dict=state["best_state"],
        task=task,
        variant=variant,
        epoch=best_epoch,
        metrics=best_metrics,
        bundle=bundle,
        signature=signature,
        init_audit=init_audit,
    )
    _write_checkpoint(
        out_dir / "checkpoint_final.pt",
        state_dict=state["final_state"],
        task=task,
        variant=variant,
        epoch=int(state["epoch"]),
        metrics=state["curves"][-1],
        bundle=bundle,
        signature=signature,
        init_audit=init_audit,
    )

    for split in C.SPLITS:
        order, row_ids, targets_raw = eval_rows(bundle, task, split)
        matrix = np.stack(state["preds"][split], axis=0).astype(np.float32)
        C.atomic_write_npz(
            out_dir / f"pred_matrix_{split}.npz",
            {"row_id": row_ids, "y_raw": targets_raw.astype(np.float32), "pred_std_matrix": matrix},
        )

    selected = dict(state["curves"][best_epoch - 1])
    selected["epoch"] = best_epoch
    final = dict(state["curves"][-1])
    last20 = {
        f"mean_last20_{split}_{metric}": float(np.mean([c[f"{split}_{metric}"] for c in state["curves"][-20:]]))
        for split in C.SPLITS
        for metric in ("rmse_raw", "mae_raw")
    }

    # frozen best/final predictions + atom contributions (raw target units)
    best_pred: dict[str, Any] = {}
    final_pred: dict[str, Any] = {}
    model.load_state_dict(state["best_state"], strict=True)
    model = model.to(device)
    for split in C.SPLITS:
        out = evaluate(model, bundle, task, split, device, with_atom_contrib=(split != "train"))
        C.atomic_write_npz(
            out_dir / f"best_predictions_{split}.npz",
            {"row_id": out["row_ids"], "y_raw": out["y_raw"], "pred_raw": out["pred_raw"]},
        )
        best_pred[split] = {"rmse_raw": out["rmse_raw"], "mae_raw": out["mae_raw"]}
        if split != "train":
            C.atomic_write_npz(
                out_dir / f"best_atom_contrib_{split}.npz",
                {
                    "row_id": out["atom_rows"].astype(np.int64),
                    "atom_contrib_raw": out["atom_contrib_raw"].astype(np.float32),
                    "n_atoms": out["ledger"]["n_atoms"],
                },
            )
            C.atomic_write_npz(
                out_dir / f"best_ledger_{split}.npz",
                {
                    "row_id": out["ledger"]["cache_rows"],
                    "pred_raw": out["ledger"]["pred_raw"],
                    "local_sum_raw": out["ledger"]["local_sum_raw"],
                    "rho_raw": out["ledger"]["rho_raw"],
                    "y_raw": out["y_raw"],
                },
            )
    if state["final_state"] is not None:
        model.load_state_dict(state["final_state"], strict=True)
        for split in C.SPLITS:
            out = evaluate(model, bundle, task, split, device)
            C.atomic_write_npz(
                out_dir / f"final_predictions_{split}.npz",
                {"row_id": out["row_ids"], "y_raw": out["y_raw"], "pred_raw": out["pred_raw"]},
            )
            final_pred[split] = {"rmse_raw": out["rmse_raw"], "mae_raw": out["mae_raw"]}

    gamma_best = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state["best_state"].items() if k.startswith("gamma_")}
    gamma_final = {
        k: v.detach().cpu().numpy().astype(np.float32) for k, v in state["final_state"].items() if k.startswith("gamma_")
    }
    C.atomic_write_npz(out_dir / "gamma_best.npz", gamma_best)
    C.atomic_write_npz(out_dir / "gamma_final.npz", gamma_final)

    summary = {
        "round_id": C.ROUND_ID,
        "status": "complete",
        "variant": variant,
        "task": task,
        "role_key": C.MODEL_ROLE_KEY[variant],
        "role_source": C.MODEL_ROLE_SOURCE[variant],
        "dataset": bundle.dataset,
        "seed": C.TRAIN_SEED,
        "epochs": epochs,
        "n_train": int(bundle.split_sizes["train"]),
        "n_valid": int(bundle.split_sizes["valid"]),
        "n_test": int(bundle.split_sizes["test"]),
        "total_updates": int(state["global_step"]),
        "steps_per_epoch": int(spe),
        "n_parameters": int(params["total"]),
        "parameter_breakdown": params["breakdown"],
        "init": init_audit,
        "signature": sig,
        "signature_payload": signature,
        "selected": selected,
        "final": dict(state["curves"][-1]),
        "last20": last20,
        "best_predictions": best_pred,
        "final_predictions": final_pred,
        "curves": state["curves"],
        "batch_order_hashes_first3": state["batch_order_hashes"][:3],
        "batch_order_sha256": C.sha256_json(state["batch_order_hashes"]),
        "first_batch_rows": state["first_batch_rows"],
        "rng_hash_at_start": state.get("rng_hash_at_start"),
        "seed_report": state.get("seed_report"),
        "train_seconds": float(state["train_seconds"]),
        "eval_seconds": float(state["eval_seconds"]),
        "device": str(device),
        "environment": C.environment_snapshot(),
        "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items()},
        "test_policy": RECIPE["test_policy"],
        "started_at": state["started_at"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    C.atomic_write_json(out_dir / "trajectory.json", summary)
    C.atomic_write_json(out_dir / "status.json", {"state": "completed", "signature": sig, "epoch": epochs})
    return summary


def _write_checkpoint(
    path,
    *,
    state_dict,
    task: str,
    variant: str,
    epoch: int,
    metrics: Mapping[str, Any],
    bundle,
    signature: Mapping[str, Any],
    init_audit: Mapping[str, Any],
) -> None:
    C.atomic_torch_save(
        path,
        {
            "round_id": C.ROUND_ID,
            "task": task,
            "variant": variant,
            "role_key": C.MODEL_ROLE_KEY[variant],
            "control_salt": C.CONTROL_SALT,
            "epoch": int(epoch),
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
            "signature": signature["signature"],
            "init": init_audit,
            "n_parameters": int(sum(v.numel() for v in state_dict.values())),
            "dataset": bundle.dataset,
        },
    )


def load_model_from_checkpoint(bundle, task: str, variant: str, checkpoint_path, device="cpu"):
    """Rebuild the model and load one frozen checkpoint (never re-initialised)."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("variant") != variant or payload.get("task") != task:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} is {payload.get('task')}/{payload.get('variant')}, "
            f"expected {task}/{variant}"
        )
    canonical = M.canonical_state(bundle.spec, bundle.norm)
    model = M.build_model(variant, bundle.spec, bundle.norm, canonical, device="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload


def recipe_record() -> dict[str, Any]:
    payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items()}
    payload["lr_schedule_audit"] = schedule_audit()
    return payload


__all__ = [
    "steps_per_epoch",
    "total_steps",
    "lr_at",
    "eval_rows",
    "evaluate",
    "merge_ledgers",
    "schedule_audit",
    "code_hashes",
    "trajectory_signature",
    "run_trajectory",
    "load_model_from_checkpoint",
    "recipe_record",
]
