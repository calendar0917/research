"""Shared core for the two P3 arms of ``zinc_p3_sam_v1``.

C  P3-Control : the frozen P3 graph (``P3NullModel``) trained with the original
                recipe-B AdamW loop (one forward / one backward per update).
S  P3-SAM     : the *same* model, initial state, data, batches and schedule,
                trained with vanilla global-L2 SAM (two forward / two backward
                per outer update, one AdamW step).

Implementation notes that matter for interpretation:

* The first-pass gradient ``g1`` is the raw L1 gradient.  It is **not** clipped;
  it only defines the ascent direction ``eps_p = rho * g1_p / (||g1|| + 1e-12)``.
* ``eps`` is built from ``g1.detach()``; the second pass runs at ``theta + eps``
  with no graph to ``eps`` (no second-order term).
* The original parameters are saved with ``clone`` and restored with ``copy_``
  before any clipping / optimizer step, so no ``+eps - eps`` drift can occur.
* One AdamW step and one LR-schedule advance per outer update; the second
  forward is never a second optimizer update.
* Non-finite loss/gradient/norm restores ``theta`` and raises; nothing perturbed
  can reach ``best.pt`` or evaluation.

The training loop is deliberately a single implementation used by the control
arm as well, so both arms share collate, row order, evaluation and checkpointing.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_V2), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402

EXP = _HERE.parent
CONFIG_PATH = EXP / "configs" / "p3_sam_v1.json"
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
PROTOCOL = "zinc-p3-sam-v1"
MODEL_KEY = "P3N"
CANDIDATE = "P3N"


# ---------------------------------------------------------------------------
# config / data / model
# ---------------------------------------------------------------------------


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    cfg = C.read_json(path)
    cfg["_config_path"] = str(Path(path))
    cfg["_config_sha256"] = C.sha256_file(Path(path))
    return cfg


def load_split(split: str) -> tuple[dict, np.ndarray, np.ndarray]:
    cache = D.load_cache(split)
    y = np.asarray(cache["labels"], dtype=np.float64)
    return cache, y, np.arange(len(y), dtype=np.int64)


def build_model(norm: Mapping[str, object], device) -> tuple[torch.nn.Module, dict]:
    """Build P3N exactly as ``ad2_models.build_model('P3N', norm, seed=0)``."""
    return A2.build_model(MODEL_KEY, norm, 0, device)


def optimizable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def make_optimizer(model: torch.nn.Module, cfg: Mapping[str, Any]) -> torch.optim.AdamW:
    opt_cfg = cfg["optimizer"]
    params = optimizable_params(model)
    if len(params) != sum(1 for _ in model.parameters()):
        raise RuntimeError("some parameters do not require grad")
    return torch.optim.AdamW(
        params,
        lr=float(opt_cfg["lr"]),
        betas=tuple(float(b) for b in opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )


def schedule_lr(step: int, total_updates: int, cfg: Mapping[str, Any]) -> float:
    opt_cfg, sch = cfg["optimizer"], cfg["schedule"]
    return C.cosine_with_warmup_lr(
        step, total_updates, int(sch["warmup_updates"]),
        float(opt_cfg["lr"]), float(opt_cfg["min_lr"]),
    )


def set_lr(optimizer: torch.optim.Optimizer, step: int, total_updates: int,
           cfg: Mapping[str, Any]) -> float:
    lr = schedule_lr(step, total_updates, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def order_sha256(order: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(order, dtype=np.int64).tobytes()).hexdigest()


def epoch_order(gen: torch.Generator, n: int) -> np.ndarray:
    return torch.randperm(int(n), generator=gen).numpy()


# ---------------------------------------------------------------------------
# SAM primitives
# ---------------------------------------------------------------------------


def grad_global_norm(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """Raw global L2 norm of all non-empty gradients, float64 accumulation."""
    total: torch.Tensor | None = None
    for p in params:
        g = p.grad
        if g is None:
            continue
        term = g.detach().double().pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError("no gradients present")
    return total.sqrt()


def sam_epsilon(params: Sequence[torch.nn.Parameter], g1_norm: torch.Tensor,
                rho: float) -> list[torch.Tensor | None]:
    """``eps_p = rho * g1_p / (||g1|| + 1e-12)``, detached from any graph."""
    scale = float(rho) / (g1_norm + 1e-12)
    eps: list[torch.Tensor | None] = []
    for p in params:
        g = p.grad
        eps.append(None if g is None else g.detach() * scale)
    return eps


def epsilon_norm(eps: Sequence[torch.Tensor | None]) -> torch.Tensor:
    total: torch.Tensor | None = None
    for e in eps:
        if e is None:
            continue
        term = e.double().pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError("empty epsilon")
    return total.sqrt()


def save_params(params: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def restore_params(params: Sequence[torch.nn.Parameter],
                   backup: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, b in zip(params, backup):
            p.copy_(b)


def apply_perturbation(params: Sequence[torch.nn.Parameter],
                       eps: Sequence[torch.Tensor | None]) -> None:
    with torch.no_grad():
        for p, e in zip(params, eps):
            if e is not None:
                p.add_(e)


def params_match_perturbed(params: Sequence[torch.nn.Parameter],
                           backup: Sequence[torch.Tensor],
                           eps: Sequence[torch.Tensor | None]) -> bool:
    with torch.no_grad():
        for p, b, e in zip(params, backup, eps):
            if e is None:
                continue
            if not torch.equal(p.detach(), b + e):
                return False
    return True


def params_match_backup(params: Sequence[torch.nn.Parameter],
                        backup: Sequence[torch.Tensor]) -> bool:
    with torch.no_grad():
        return all(torch.equal(p.detach(), b) for p, b in zip(params, backup))


# ---------------------------------------------------------------------------
# one outer update
# ---------------------------------------------------------------------------


def _check_finite(value: torch.Tensor, what: str, epoch: int) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise RuntimeError(f"non-finite {what} at epoch {epoch}: {v!r}")
    return v


def update_control(model, optimizer, params, batch, target, *, clip: float,
                   epoch: int) -> dict:
    """One recipe-B AdamW update: forward, raw L1, backward, clip, return pred."""
    pred = model(batch)
    loss = F.l1_loss(pred.view(-1), target.view(-1))
    _check_finite(loss.detach(), "control loss", epoch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    g_norm = _check_finite(grad_global_norm(params), "control grad norm", epoch)
    torch.nn.utils.clip_grad_norm_(params, float(clip))
    return {
        "pred": pred.detach(),
        "loss": float(loss.detach()),
        "g1_norm": float(g_norm),
        "g2_norm": None,
        "eps_norm": None,
        "clipped": None,
        "perturb_exact": None,
        "restore_exact": None,
    }


def update_sam(model, optimizer, params, batch, target, *, rho: float, clip: float,
               epoch: int, check: bool = False) -> dict:
    """One vanilla global-L2 SAM outer update (one AdamW step is *not* taken here)."""
    optimizer.zero_grad(set_to_none=True)
    pred = model(batch)
    loss = F.l1_loss(pred.view(-1), target.view(-1))
    _check_finite(loss.detach(), "SAM first-pass loss", epoch)
    loss.backward()

    backup: list[torch.Tensor] | None = None
    try:
        g1_norm = grad_global_norm(params)
        _check_finite(g1_norm, "SAM first-pass grad norm", epoch)
        eps = sam_epsilon(params, g1_norm, rho)
        eps_norm = _check_finite(epsilon_norm(eps), "SAM epsilon norm", epoch)
        backup = save_params(params)
        apply_perturbation(params, eps)
        perturb_exact = params_match_perturbed(params, backup, eps) if check else None

        optimizer.zero_grad(set_to_none=True)
        pred2 = model(batch)
        loss2 = F.l1_loss(pred2.view(-1), target.view(-1))
        _check_finite(loss2.detach(), "SAM second-pass loss", epoch)
        loss2.backward()
        g2_norm = _check_finite(grad_global_norm(params), "SAM second-pass grad norm", epoch)
    finally:
        if backup is not None:
            restore_params(params, backup)

    restore_exact = params_match_backup(params, backup) if check else None
    torch.nn.utils.clip_grad_norm_(params, float(clip))
    return {
        "pred": pred.detach(),
        "pred_perturbed": pred2.detach(),
        "loss": float(loss.detach()),
        "loss_perturbed": float(loss2.detach()),
        "g1_norm": float(g1_norm),
        "g2_norm": float(g2_norm),
        "eps_norm": float(eps_norm),
        "clipped": bool(float(g2_norm) > float(clip)),
        "perturb_exact": perturb_exact,
        "restore_exact": restore_exact,
    }


# ---------------------------------------------------------------------------
# epoch
# ---------------------------------------------------------------------------


def _sample_check(global_step: int) -> bool:
    """A handful of sampled perturb/restore verifications, not per-step logs."""
    return global_step <= 3 or global_step % 2000 == 0


def train_epoch(model, optimizer, params, cache, rows, y, gen, *, arm: str,
                cfg: Mapping[str, Any], epoch: int, total_updates: int,
                global_step: int, collate_fn, device) -> tuple[dict, int]:
    model.train()
    batch_size = int(cfg["schedule"]["batch_size"])
    clip = float(cfg["optimizer"]["grad_clip"])
    rho = float(cfg["sam"]["rho"])
    order = epoch_order(gen, len(rows))
    total_err, pert_err, n_mol = 0.0, 0.0, 0
    g1_norms, g2_norms, eps_norms = [], [], []
    g1_over_clip = 0
    perturb_checks = {"checked": 0, "exact": 0}
    restore_checks = {"checked": 0, "exact": 0}
    eps_dev: list[float] = []
    lr = 0.0
    t0 = time.perf_counter()
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        batch = T1._to_device(collate_fn(cache, chunk), device)
        target = torch.as_tensor(y[chunk], dtype=torch.float32, device=device)
        step_index = global_step + 1
        check = _sample_check(step_index) if arm == "S" else False
        if arm == "C":
            rec = update_control(model, optimizer, params, batch, target,
                                 clip=clip, epoch=epoch)
        elif arm == "S":
            rec = update_sam(model, optimizer, params, batch, target, rho=rho,
                             clip=clip, epoch=epoch, check=check)
        else:
            raise ValueError(f"unknown arm {arm!r}")

        g1_norms.append(rec["g1_norm"])
        g1_over_clip += int(rec["g1_norm"] > clip)
        if rec["g2_norm"] is not None:
            g2_norms.append(rec["g2_norm"])
            eps_norms.append(rec["eps_norm"])
            if rec["eps_norm"] > 0.0:
                eps_dev.append(abs(rec["eps_norm"] - rho) / rho)
            if check:
                perturb_checks["checked"] += 1
                perturb_checks["exact"] += int(bool(rec["perturb_exact"]))
                restore_checks["checked"] += 1
                restore_checks["exact"] += int(bool(rec["restore_exact"]))

        total_err += float((rec["pred"].view(-1) - target).abs().sum())
        if "pred_perturbed" in rec:
            pert_err += float((rec["pred_perturbed"].view(-1) - target).abs().sum())
        n_mol += len(chunk)
        global_step = step_index
        lr = set_lr(optimizer, global_step, total_updates, cfg)
        optimizer.step()
    seconds = time.perf_counter() - t0
    if global_step != epoch * int(cfg["schedule"]["steps_per_epoch"]):
        raise RuntimeError(f"outer update counter drifted: {global_step}")
    rec = {
        "epoch": int(epoch),
        "online_train_mae": float(total_err / max(n_mol, 1)),
        "perturbed_train_mae": (float(pert_err / max(n_mol, 1)) if arm == "S" else None),
        "lr": float(lr),
        "seconds": float(seconds),
        "steps": int(len(range(0, len(order), batch_size))),
        "order_sha256": order_sha256(order),
        "g1_norm_mean": float(np.mean(g1_norms)),
        "g1_norm_max": float(np.max(g1_norms)),
        "g1_over_clip_updates": int(g1_over_clip),
        "g2_norm_mean": float(np.mean(g2_norms)) if g2_norms else None,
        "g2_norm_max": float(np.max(g2_norms)) if g2_norms else None,
        "g2_clip_fraction": (float(np.mean([g > clip for g in g2_norms])) if g2_norms else None),
        "eps_norm_mean": float(np.mean(eps_norms)) if eps_norms else None,
        "eps_norm_rel_dev_max": float(np.max(eps_dev)) if eps_dev else None,
        "perturb_checks_checked": int(perturb_checks["checked"]),
        "perturb_checks_exact": int(perturb_checks["exact"]),
        "restore_checks_checked": int(restore_checks["checked"]),
        "restore_checks_exact": int(restore_checks["exact"]),
    }
    return rec, global_step


def collate_p3(cache, rows, candidate: str | None = None, *, valid: bool = False):
    """The unchanged v1 collate for the P3 batch layout (candidate is implicit)."""
    return T1.collate(cache, rows, CANDIDATE, valid=valid)


def evaluate(model, cache, rows, device, batch_size: int) -> tuple[np.ndarray, float]:
    return T1.evaluate(model, cache, rows, candidate=CANDIDATE, device=device,
                       batch_size=batch_size, collate_fn=collate_p3)
