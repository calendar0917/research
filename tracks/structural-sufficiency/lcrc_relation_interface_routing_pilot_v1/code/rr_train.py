"""The frozen Rf training / every-epoch evaluation loop for R1 and R2.

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

``lr_at`` is checked step-for-step against ``gr_common.lr_at("Rf", ...)`` in
``analysis/routing_invariants.json``.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

import rr_common as C
import rr_models as M

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
    base = bundle.base
    order, row_ids = base.row_ids_sorted(split)
    return order, row_ids, bundle.targets(task, split)[order]


def evaluate(
    model,
    bundle,
    task: str,
    split: str,
    device,
    *,
    with_ledger: bool = False,
) -> dict[str, Any]:
    import torch

    order, row_ids, targets_raw = eval_rows(bundle, task, split)
    scaler = bundle.scaler(task)
    y_std = (targets_raw - scaler["mean"]) / scaler["std"]
    cache = bundle.caches[split]
    model.eval()
    preds: list[np.ndarray] = []
    ledgers: list[dict[str, np.ndarray]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, order.size, int(RECIPE["eval_batch_size"])):
            chunk = order[start : start + int(RECIPE["eval_batch_size"])]
            batch = bundle.collate(split, chunk)
            _attach_target(batch, chunk, y_std)
            batch = _to_device(batch, device)
            if with_ledger:
                pred, parts = model(batch, return_aux=True)
                ledgers.append(_ledger_block(parts, batch, chunk))
            else:
                pred = model(batch)
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
    if with_ledger:
        out["ledger"] = merge_ledgers(ledgers)
    return out


def _np(value):
    return value.detach().cpu().numpy().astype(np.float64)


def _ledger_block(parts: Mapping[str, Any], batch: Mapping[str, Any], chunk: np.ndarray) -> dict[str, Any]:
    return {
        "cache_rows": np.asarray(chunk, dtype=np.int64),
        "n_graphs": np.asarray([int(batch["n_graphs"])]),
        "pred_std": _np(parts["pred_std"]),
        "bias": _np(parts["bias"]),
        "rho": _np(parts["rho"]),
        "block_sum": _np(parts["block_sum"]),
        "iface_sum": _np(parts["iface_sum"]),
        "c_block": _np(parts["c_block"]),
        "c_iface": _np(parts["c_iface"]),
        "block_graph": np.asarray(batch["block_graph"].detach().cpu().numpy()),
        "iface_graph": np.asarray(batch["iface_graph"].detach().cpu().numpy()),
        "iface_a": np.asarray(batch["iface_a"].detach().cpu().numpy()),
        "iface_b": np.asarray(batch["iface_b"].detach().cpu().numpy()),
        "blk_n_bonds": np.asarray(batch["blk_n_bonds"].detach().cpu().numpy()),
        "blk_n_atoms": np.asarray(batch["blk_n_atoms"].detach().cpu().numpy()),
        "n_blocks_per_graph": np.asarray(batch["n_blocks_per_graph"].detach().cpu().numpy()),
        "n_ifaces_per_graph": np.asarray(batch["n_ifaces_per_graph"].detach().cpu().numpy()),
        "z": _np(parts["z"]),
    }


def merge_ledgers(blocks: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Pack per-batch ledgers into one ragged per-molecule ledger.

    ``block_ptr`` / ``iface_ptr`` delimit each molecule's blocks and interfaces;
    ``iface_a_local`` / ``iface_b_local`` are block indices *within the molecule*.
    """
    rows: list[int] = []
    pred_std: list[np.ndarray] = []
    bias: list[np.ndarray] = []
    rho: list[np.ndarray] = []
    block_sum: list[np.ndarray] = []
    iface_sum: list[np.ndarray] = []
    block_ptr = [0]
    iface_ptr = [0]
    block_id: list[np.ndarray] = []
    c_block: list[np.ndarray] = []
    blk_n_bonds: list[np.ndarray] = []
    blk_n_atoms: list[np.ndarray] = []
    iface_a: list[np.ndarray] = []
    iface_b: list[np.ndarray] = []
    c_iface: list[np.ndarray] = []
    z: list[np.ndarray] = []
    for block in blocks:
        n = int(block["n_graphs"][0])
        rows.extend(int(v) for v in block["cache_rows"].tolist())
        pred_std.append(block["pred_std"])
        bias.append(block["bias"])
        rho.append(block["rho"])
        block_sum.append(block["block_sum"])
        iface_sum.append(block["iface_sum"])
        nb_counts = np.asarray(block["n_blocks_per_graph"], dtype=np.int64)
        ni_counts = np.asarray(block["n_ifaces_per_graph"], dtype=np.int64)
        bpos = 0
        ipos = 0
        for g in range(n):
            nb = int(nb_counts[g])
            ni = int(ni_counts[g])
            block_id.append(np.arange(nb, dtype=np.int64))
            c_block.append(block["c_block"][bpos : bpos + nb])
            blk_n_bonds.append(block["blk_n_bonds"][bpos : bpos + nb])
            blk_n_atoms.append(block["blk_n_atoms"][bpos : bpos + nb])
            z.append(block["z"][bpos : bpos + nb])
            iface_a.append(block["iface_a"][ipos : ipos + ni] - bpos)
            iface_b.append(block["iface_b"][ipos : ipos + ni] - bpos)
            c_iface.append(block["c_iface"][ipos : ipos + ni])
            block_ptr.append(block_ptr[-1] + nb)
            iface_ptr.append(iface_ptr[-1] + ni)
            bpos += nb
            ipos += ni
        if bpos != int(np.asarray(block["c_block"]).size) or ipos != int(np.asarray(block["c_iface"]).size):
            raise RuntimeError("ledger block/interface counts do not cover the batch")
    return {
        "cache_rows": np.asarray(rows, dtype=np.int64),
        "pred_std": np.concatenate(pred_std) if pred_std else np.zeros(0),
        "bias": np.concatenate(bias) if bias else np.zeros(0),
        "rho": np.concatenate(rho) if rho else np.zeros(0),
        "block_sum": np.concatenate(block_sum) if block_sum else np.zeros(0),
        "iface_sum": np.concatenate(iface_sum) if iface_sum else np.zeros(0),
        "block_ptr": np.asarray(block_ptr, dtype=np.int64),
        "iface_ptr": np.asarray(iface_ptr, dtype=np.int64),
        "block_id_local": np.concatenate(block_id) if block_id else np.zeros(0, np.int64),
        "c_block": np.concatenate(c_block) if c_block else np.zeros(0),
        "blk_n_bonds": np.concatenate(blk_n_bonds) if blk_n_bonds else np.zeros(0, np.int64),
        "blk_n_atoms": np.concatenate(blk_n_atoms) if blk_n_atoms else np.zeros(0, np.int64),
        "iface_a_local": np.concatenate(iface_a) if iface_a else np.zeros(0, np.int64),
        "iface_b_local": np.concatenate(iface_b) if iface_b else np.zeros(0, np.int64),
        "c_iface": np.concatenate(c_iface) if c_iface else np.zeros(0),
        "z": np.concatenate(z, axis=0) if z else np.zeros((0, C.WIDTH)),
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

    spec = importlib.util.spec_from_file_location("_rr_gr_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return np.asarray(
        [module.lr_at("Rf", step, n_train, epochs=int(RECIPE["epochs"])) for step in range(steps)],
        dtype=np.float64,
    )


def code_hashes() -> dict[str, str]:
    out = {}
    for name in ("rr_common.py", "rr_partition.py", "rr_routing.py", "rr_models.py", "rr_data.py", "rr_train.py"):
        path = C.CODE_DIR / name
        if path.is_file():
            out[name] = C.sha256_file(path)
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
        "seed": C.TRAIN_SEED,
        "epochs": int(epochs),
        "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items()},
        "routing_fingerprints": bundle.routing_fingerprints,
        "target_scaler": bundle.scaler(task),
        "target_sha256": {split: C.sha256_array(bundle.targets(task, split)) for split in C.SPLITS},
        "row_id_sha256": {split: C.sha256_array(bundle.global_row_ids[split]) for split in C.SPLITS},
        "graph_cache_sha256": bundle.extra["graph_cache_sha256"],
        "lipo_raw_sha256": bundle.extra["lipo_raw_sha256"],
        "feature_spec_hash": bundle.extra["spec_hash"],
        "norm_fingerprint": bundle.extra["norm_fingerprint"],
        "init_streams": {
            "canonical": C.INIT_SEED_CANONICAL,
            "block": C.INIT_STREAM_BLOCK,
            "iface": C.INIT_STREAM_IFACE,
            "iface_readout": C.INIT_STREAM_IFACE + 1,
            "rho": C.INIT_STREAM_RHO,
            "context": C.INIT_STREAM_CONTEXT,
        },
        "code_sha256": code_hashes(),
        "parameter_counts": M.expected_parameter_counts(int(bundle.spec.global_dim)),
    }
    payload["signature"] = C.sha256_json({k: v for k, v in payload.items() if k != "signature"})
    return payload


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


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

    canonical = M.canonical_trunk_state(bundle.spec, bundle.norm)
    model = M.build_model(variant, bundle.spec, bundle.norm, canonical, device="cpu")
    params = model.parameter_report()
    trunk_hash = C.sha256_state(
        {k[len(M.ARM_PREFIX) :]: v for k, v in model.state_dict().items() if k.startswith(M.ARM_PREFIX)}
    )
    expected_trunk = C.sha256_state(canonical, keys=M.CM.trunk_keys(canonical))
    init_audit = {
        "variant": variant,
        "task": task,
        "canonical_state_hashes": M.CM.state_hashes(canonical),
        "trunk_hash": trunk_hash,
        "trunk_matches_canonical": trunk_hash == expected_trunk,
        "parameters": params,
        "expected_parameter_counts": M.expected_parameter_counts(int(bundle.spec.global_dim)),
        "readout_state_hash": C.sha256_state(model.readout_state()),
        "init_streams": signature["init_streams"],
    }
    if not init_audit["trunk_matches_canonical"]:
        raise RuntimeError(f"{task}/{variant}: trunk init does not match the canonical C1 state")

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
            {
                "pred_raw": matrix,
                "row_id": row_ids,
                "y_raw": targets_raw.astype(np.float32),
                "epoch": np.arange(1, matrix.shape[0] + 1),
            },
        )

    model.load_state_dict({k: v.to(device) for k, v in state["best_state"].items()}, strict=True)
    best_outputs = {split: evaluate(model, bundle, task, split, device, with_ledger=True) for split in C.SPLITS}
    model.load_state_dict({k: v.to(device) for k, v in state["final_state"].items()}, strict=True)
    final_outputs = {split: evaluate(model, bundle, task, split, device) for split in C.SPLITS}
    for tag, outputs in (("best", best_outputs), ("final", final_outputs)):
        arrays = {}
        for split in C.SPLITS:
            out = outputs[split]
            arrays[f"{split}_pred_raw"] = out["pred_raw"].astype(np.float64)
            arrays[f"{split}_pred_std"] = out["pred_std"].astype(np.float64)
            arrays[f"{split}_y_raw"] = out["y_raw"].astype(np.float64)
            arrays[f"{split}_row_id"] = out["row_ids"]
        C.atomic_write_npz(out_dir / f"{tag}_predictions.npz", arrays)
    for split in C.SPLITS:
        out = best_outputs[split]
        ledger_payload = {"row_id": out["row_ids"], "y_raw": out["y_raw"], "pred_raw": out["pred_raw"]}
        ledger_payload.update({f"ldg_{k}": v for k, v in out["ledger"].items()})
        C.atomic_write_npz(out_dir / f"ledger_{split}.npz", ledger_payload)

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
        "task": task,
        "variant": variant,
        "purpose": purpose,
        "status": "complete",
        "signature": sig,
        "signature_payload": signature,
        "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items()},
        "epochs": epochs,
        "steps_per_epoch": spe,
        "total_updates": int(state["global_step"]),
        "n_train": int(train_rows.size),
        "n_valid": int(len(bundle.global_row_ids["valid"])),
        "n_test": int(len(bundle.global_row_ids["test"])),
        "test_eval_count": int(len(state["curves"])),
        "test_policy": "always (DEVELOPMENT_TEST)",
        "n_parameters": int(params["total"]),
        "parameter_breakdown": params["breakdown"],
        "expected_parameter_counts": init_audit["expected_parameter_counts"],
        "init": {k: v for k, v in init_audit.items() if k != "parameters"},
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
            **{
                k: state["curves"][-1][k]
                for k in state["curves"][-1]
                if k.endswith("_rmse_raw") or k.endswith("_mae_raw")
            },
        },
        "last20": last20,
        "batch_order_sha256": C.sha256_text("".join(state["batch_order_hashes"])),
        "batch_order_hashes_first3": state["batch_order_hashes"][:3],
        "first_batch_rows": state["first_batch_rows"],
        "rng_hash_at_start": state.get("rng_hash_at_start"),
        "train_seconds": float(state["train_seconds"]),
        "eval_seconds_total": float(state["eval_seconds"]),
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
        status_file,
        {
            "state": "completed",
            "signature": sig,
            "task": task,
            "variant": variant,
            "epochs": epochs,
            "selected_epoch": best_epoch,
            "valid_rmse_raw": float(state["best"]["valid_rmse_raw"]),
            "test_rmse_raw": float(best_metrics["test_rmse_raw"]),
            "finished_at": summary["finished_at"],
        },
    )
    if resume_path.is_file():
        resume_path.unlink()
    return summary


def _write_checkpoint(
    path,
    *,
    state_dict: Mapping[str, Any],
    task: str,
    variant: str,
    epoch: int,
    metrics: Mapping[str, Any],
    bundle,
    signature: Mapping[str, Any],
    init_audit: Mapping[str, Any],
) -> None:
    payload = {
        "round_id": C.ROUND_ID,
        "dataset": bundle.dataset,
        "task": task,
        "variant": variant,
        "epoch": int(epoch),
        "state_dict": {k: v.detach().cpu().clone() for k, v in state_dict.items()},
        "metrics": dict(metrics),
        "target_scaler": dict(bundle.scaler(task)),
        "norm": {
            "global_mean": np.asarray(bundle.norm["global_mean"]).tolist(),
            "global_scale": np.asarray(bundle.norm["global_scale"]).tolist(),
            "scalar_mean": np.asarray(bundle.norm["scalar_mean"]).tolist(),
            "scalar_scale": np.asarray(bundle.norm["scalar_scale"]).tolist(),
            "fingerprint": bundle.extra["norm_fingerprint"],
        },
        "routing_fingerprints": dict(bundle.routing_fingerprints),
        "signature": signature["signature"],
        "init_audit": dict(init_audit),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    C.atomic_torch_save(path, payload)


def load_model_from_checkpoint(bundle, task: str, variant: str, checkpoint_path, device="cpu"):
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["task"] != task or payload["variant"] != variant:
        raise RuntimeError(
            f"{checkpoint_path}: task/variant {payload['task']}/{payload['variant']} != {task}/{variant}"
        )
    if dict(payload["target_scaler"]) != dict(bundle.scaler(task)):
        raise RuntimeError(f"{checkpoint_path}: target scaler does not match the task bundle")
    canonical = M.canonical_trunk_state(bundle.spec, bundle.norm)
    model = M.build_model(variant, bundle.spec, bundle.norm, canonical, device="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    reloaded = C.sha256_state(model.state_dict())
    stored = C.sha256_state(payload["state_dict"])
    if reloaded != stored:
        raise RuntimeError(f"{checkpoint_path}: reloaded state hash mismatch")
    payload["reloaded_state_hash"] = reloaded
    return model, payload


def recipe_record() -> dict[str, Any]:
    out = {key: (list(value) if isinstance(value, tuple) else value) for key, value in RECIPE.items()}
    out["target_unit_for_reporting"] = "raw (de-standardised once)"
    out["selection_split"] = "valid"
    out["epoch0_role"] = "initialisation diagnostic; never selectable"
    out["dropout_module_count"] = 1
    out["lr_schedule_reference"] = "gnn_route_clarification_8h_v1 gr_common.lr_at('Rf', ...)"
    return out


__all__ = [
    "RECIPE",
    "steps_per_epoch",
    "total_steps",
    "lr_at",
    "eval_rows",
    "evaluate",
    "merge_ledgers",
    "run_trajectory",
    "load_model_from_checkpoint",
    "trajectory_signature",
    "recipe_record",
    "code_hashes",
]
