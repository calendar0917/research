"""One frozen training loop shared by all three families and all three targets.

The loop implements exactly the loss definition of PROTOCOL.md section 6::

    e_i        = f_std(G_i) - y_std_i           (same Fit scaler for every arm)
    L0         = mean over the base minibatch of e_train^2
    LE         = 0.5 * E_q[e_eval_A^2 + e_eval_B^2]
    LD         = 0.5 * E_q[(e_eval_B - e_eval_A)^2]
    BASE       L = L0
    ENDPOINT   L = (L0 + lambda*LE) / (1 + lambda)
    DELTA      L = (L0 + lambda*LD) / (1 + lambda)

The auxiliary pair forward runs with the module temporarily in ``eval()`` mode
(so BatchNorm buffers are not updated and dropout consumes no RNG) but with
autograd enabled.  Both losses are combined **before** a single backward pass.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import sr_common as C

TARGETS = ("BASE", "ENDPOINT", "DELTA")


def lr_at_fgcr(step: int, steps_per_epoch: int, epochs: int,
               peak: float = 3e-4, minimum: float = 1e-5, warmup_epochs: int = 5) -> float:
    warmup = int(warmup_epochs) * int(steps_per_epoch)
    total = int(epochs) * int(steps_per_epoch)
    if warmup > 0 and step < warmup:
        return float(peak) * float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress))


def lr_at_linear_warmup(step: int, steps_per_epoch: int, epochs: int,
                        peak: float = 1e-3, minimum: float = 1e-5,
                        warmup_epochs: int = 5, start_fraction: float = 0.1) -> float:
    """The frozen ``Rf`` / ``rd_small_s0`` schedule (step is 0-based).

    ``gr_common.lr_at``: warmup is ``peak*(0.1 + 0.9*step/warm)``; ``rd_train``
    uses ``1e-4 + (peak-1e-4)*step/warm`` which is the same ramp for
    peak=1e-3.  Both then take a cosine from ``peak`` to ``minimum``.
    """
    warmup = int(warmup_epochs) * int(steps_per_epoch)
    total = int(epochs) * int(steps_per_epoch)
    if warmup > 0 and step < warmup:
        return float(peak) * (float(start_fraction) + (1.0 - float(start_fraction))
                              * float(step) / float(warmup))
    start = step - warmup if warmup > 0 else step
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, float(start) / float(span)))
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress))


class Arm:
    """A trainable model plus the three forward entry points the loop needs."""

    def __init__(self, *, family: str, model: torch.nn.Module, device: Any,
                 base_loss: Callable[[np.ndarray], torch.Tensor],
                 aux_predict: Callable[[np.ndarray], torch.Tensor],
                 predict: Callable[[np.ndarray], np.ndarray],
                 lr_at: Callable[[int, int, int], float],
                 recipe: Mapping[str, Any],
                 extra: Mapping[str, Any] | None = None) -> None:
        self.family = family
        self.model = model
        self.device = device
        self.base_loss = base_loss
        self.aux_predict = aux_predict
        self.predict = predict
        self.lr_at = lr_at
        self.recipe = dict(recipe)
        self.extra = dict(extra or {})


def _to_device(batch):
    return batch


def train_one(*, arm: Arm, target: str, train_rows: np.ndarray, order: np.ndarray,
              schedule: np.ndarray, labels_std: np.ndarray, scaler: Mapping[str, float],
              eval_rows: np.ndarray, epochs: int, out_dir: Path, eval_extra_rows: Mapping[str, np.ndarray] | None = None,
              log: Callable[[str], None] = print, checkpoint_extra: Mapping[str, Any] | None = None
              ) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(target)
    recipe = arm.recipe
    model = arm.model.to(arm.device)
    batch_size = int(recipe["batch_molecules"])
    eval_batch = int(recipe["eval_batch_molecules"])
    steps_per_epoch = int(math.ceil(train_rows.size / batch_size))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["peak_lr"]),
        betas=tuple(float(b) for b in recipe.get("betas", (0.9, 0.999))),
        eps=float(recipe.get("eps", 1e-8)),
        weight_decay=float(recipe["weight_decay"]))
    y_std_t = torch.as_tensor(np.asarray(labels_std, dtype=np.float32), device=arm.device)
    grad_clip = float(recipe["grad_clip"])
    lam = float(C.LAMBDA)

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = np.asarray(eval_rows, dtype=np.int64)
    curves: list[dict[str, Any]] = []
    best = {"rmse": float("inf"), "epoch": -1, "state": None, "pred": None}
    final_state = None
    final_pred = None
    step = 0
    aux_calls = 0
    t0 = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        t_epoch = time.perf_counter()
        model.train()
        perm = order[epoch - 1]
        rows_epoch = train_rows[perm]
        losses: list[float] = []
        base_parts: list[float] = []
        aux_parts: list[float] = []
        for start in range(0, rows_epoch.size, batch_size):
            chunk = rows_epoch[start:start + batch_size]
            for group in optimizer.param_groups:
                group["lr"] = float(arm.lr_at(step, steps_per_epoch, int(epochs)))
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss0 = arm.base_loss(chunk)
            if target == "BASE":
                loss = loss0
                aux_value = 0.0
            else:
                pairs = schedule[min(step - 1, schedule.shape[0] - 1)]
                # pair endpoints come from the frozen catalogue rows
                ends = np.stack([arm.extra["pair_row_a"][pairs],
                                 arm.extra["pair_row_b"][pairs]], axis=1).reshape(-1)
                uniq, inv = np.unique(ends, return_inverse=True)
                was_training = model.training
                model.train(False)
                pred = arm.aux_predict(uniq)
                model.train(was_training)
                aux_calls += 1
                residual = pred[torch.as_tensor(inv, device=arm.device)] - y_std_t[
                    torch.as_tensor(ends, device=arm.device)]
                residual = residual.view(pairs.size, 2)
                e_a = residual[:, 0]
                e_b = residual[:, 1]
                if target == "ENDPOINT":
                    aux_value_t = 0.5 * (e_a.pow(2) + e_b.pow(2)).mean()
                else:
                    aux_value_t = 0.5 * (e_b - e_a).pow(2).mean()
                loss = (loss0 + lam * aux_value_t) / (1.0 + lam)
                aux_value = float(aux_value_t.detach().cpu().item())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
            if target != "BASE" and not loss.requires_grad:
                raise RuntimeError("auxiliary path lost the autograd graph")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            base_parts.append(float(loss0.detach().cpu().item()))
            aux_parts.append(aux_value)

        # per-epoch checkpoint selection on the frozen evaluation rows
        pred_std = arm.predict(eval_rows)
        pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
        target_raw = arm.extra["eval_labels_raw"]
        epoch_rmse = C.rmse(pred_raw, target_raw)
        epoch_mae = C.mae(pred_raw, target_raw)
        row = {
            "epoch": epoch, "target": target, "family": arm.family,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)),
            "base_loss": float(np.mean(base_parts)),
            "aux_loss": float(np.mean(aux_parts)),
            "eval_rmse": epoch_rmse, "eval_mae": epoch_mae,
            "epoch_seconds": time.perf_counter() - t_epoch,
        }
        curves.append(row)
        if epoch_rmse < best["rmse"]:
            best = {"rmse": epoch_rmse, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "pred": pred_raw.copy()}
        final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        final_pred = pred_raw.copy()
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            log(f"[{arm.family}:{target}] epoch {epoch}/{epochs} "
                f"loss {row['train_loss']:.5f} (base {row['base_loss']:.5f} "
                f"aux {row['aux_loss']:.5f}) eval {epoch_rmse:.6f}/{epoch_mae:.6f} "
                f"({row['epoch_seconds']:.1f}s)")

    seconds = time.perf_counter() - t0
    last20 = curves[-20:]
    summary = {
        "family": arm.family, "target": target, "epochs": int(epochs),
        "steps_per_epoch": steps_per_epoch, "global_step": step,
        "seconds": seconds, "aux_forward_calls": aux_calls,
        "best_epoch": int(best["epoch"]), "best_eval_rmse": float(best["rmse"]),
        "last20_eval_rmse": float(np.mean([r["eval_rmse"] for r in last20])),
        "last20_eval_mae": float(np.mean([r["eval_mae"] for r in last20])),
        "final_eval_rmse": float(curves[-1]["eval_rmse"]),
        "final_eval_mae": float(curves[-1]["eval_mae"]),
        "recipe": dict(recipe), "lambda": lam,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "extra": dict(arm.extra.get("meta", {})),
    }
    _save(out_dir, "best.pt", best["state"], best["epoch"], summary, checkpoint_extra)
    _save(out_dir, "final.pt", final_state, int(epochs), summary, checkpoint_extra)
    return {
        "curves": curves, "summary": summary,
        "best_pred_raw": best["pred"], "final_pred_raw": final_pred,
        "best_state": best["state"], "final_state": final_state,
    }


def _save(out_dir: Path, name: str, state, epoch: int, summary: Mapping[str, Any],
          extra: Mapping[str, Any] | None) -> None:
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in state.items()},
                "epoch": int(epoch), "summary": C.jsonable(summary),
                "extra": C.jsonable(dict(extra or {}))},
               out_dir / name)


def evaluate_rows(arm: Arm, rows: np.ndarray, scaler: Mapping[str, float]) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    pred_std = arm.predict(rows)
    return {
        "row_ids": rows,
        "pred_std": pred_std,
        "pred_raw": pred_std * float(scaler["std"]) + float(scaler["mean"]),
    }
