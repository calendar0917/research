"""Training loop for the follow-up unit.

``rd_train.train_arm`` (parent round, frozen) constructs the model itself via
``rd_model.build_model``.  This module reproduces that loop line for line and
changes exactly one thing: the model is built through an injected factory, so
the same loop can drive RESP (for an equivalence check) and JOINT.

Everything else -- ordering, learning-rate schedule, per-molecule view loss,
gradient clipping, per-epoch full Train/Val/Test evaluation, best-Val rule --
is unchanged and imported from the parent module:
``rd_train.lr_schedule``, ``rd_train.molecule_loss``,
``rd_train.evaluate_split``, ``rd_train.train_order``.

The 1-epoch RESP reproduction against the committed S0 ``curve.csv``
(``fu_checks.py --reproduce-epoch1``) is the check that this copy did not drift.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Mapping

import numpy as np
import torch

import fu_common as FC  # noqa: F401
import fu_model as FM
import rd_common as RC
import rd_model as RM
import rd_train as RT


def train_arm(
    *,
    dataset: RT.Dataset,
    arm: str,
    config: Mapping[str, Any],
    device: str,
    init_state: Mapping[str, torch.Tensor],
    order: np.ndarray,
    epochs: int,
    log=print,
    eval_batch_size: int = 256,
    model_factory=FM.build_model,
) -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    model_config = FM.model_config(config, arm)
    model = model_factory(dataset.dims, model_config, init_state).to(device)
    model_hash = RM.state_hash(model.state_dict())
    if model_hash != RM.state_hash(init_state):
        raise RuntimeError("model did not load the canonical init state exactly")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["peak_lr"]),
        weight_decay=float(config["weight_decay"]),
        betas=tuple(float(b) for b in config["betas"]),
        eps=float(config["eps"]),
    )
    train_split = dataset.splits["train"]
    n_train = len(train_split.row_id)
    batch_size = int(config["batch_size"])
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
    lr_at = RT.lr_schedule(config, steps_per_epoch, epochs)
    grad_clip = float(config["grad_clip"])

    curves: list[dict[str, Any]] = []
    best = {"valid_rmse_raw": float("inf"), "epoch": -1}
    best_state: dict[str, torch.Tensor] | None = None
    best_pred: dict[str, Any] = {}
    final_pred: dict[str, Any] = {}
    global_step = 0
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        perm = order[epoch - 1]
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            mol_idx = perm[start:start + batch_size]
            if mol_idx.size == 0:
                continue
            optimizer.param_groups[0]["lr"] = lr_at(global_step)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = RT.molecule_loss(model, dataset, train_split, mol_idx, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "arm": arm,
            "train_loss": epoch_loss / max(1, n_batches),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for split_name in ("train", "valid", "test"):
            pred = RT.evaluate_split(model, dataset, split_name, device, eval_batch_size)
            predictions[split_name] = pred
            row[f"{split_name}_rmse_raw"] = RC.rmse(pred["pred_raw"], pred["target_raw"])
            row[f"{split_name}_mae_raw"] = RC.mae(pred["pred_raw"], pred["target_raw"])
        row["seconds"] = time.perf_counter() - t0
        row["eval_batch_reductions"] = int(RT._EVAL_BATCH_STATE.get("reductions", 0))
        curves.append(row)
        if row["valid_rmse_raw"] < best["valid_rmse_raw"]:
            best = {"valid_rmse_raw": float(row["valid_rmse_raw"]), "epoch": int(epoch)}
            best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
            best_pred = copy.deepcopy(predictions)
        final_pred = copy.deepcopy(predictions)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            log(
                f"[{dataset.task}:{arm}] epoch {epoch}/{epochs} "
                f"train {row['train_rmse_raw']:.4f} valid {row['valid_rmse_raw']:.4f} "
                f"test {row['test_rmse_raw']:.4f} ({row['seconds']:.1f}s)"
            )
    seconds = time.perf_counter() - t0
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    return {
        "arm": arm,
        "curves": curves,
        "best": best,
        "best_state": best_state,
        "final_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "best_predictions": best_pred,
        "final_predictions": final_pred,
        "seconds": seconds,
        "steps_per_epoch": steps_per_epoch,
        "global_step": global_step,
        "n_parameters": RM.n_parameters(model),
        "init_state_hash": RM.state_hash(init_state),
        "loaded_state_hash": model_hash,
        "eval_batch_size": eval_batch_size,
    }
