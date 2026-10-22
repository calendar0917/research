"""Shared core for the two single-model arms of ``zinc_local_factorization_v1``.

``D`` = dense content MLP, ``F`` = factorized low-order combination.  Both arms
use the identical ``bf_data`` objects (S3 / S4 / F4 rooted typed fragments, full
typed induced adjacency, multiplicity), the identical root-preserving
permutation average, the identical per-root/family 198-dim statistics, and the
identical P3 carrier, initialisation protocol, batch order and recipe-B
schedule.  Only ``phi`` differs.

This module intentionally holds one training loop, one evaluator and one
schedule implementation, used by both arms, so a difference between arms cannot
come from the harness.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_BF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bf_data as BFD  # noqa: E402
import lf_models as LM  # noqa: E402

EXP = _HERE.parent
CONFIG_PATH = EXP / "configs" / "lf_content_v1.json"
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
PROTOCOL = "zinc-local-factorization-v1"

SEED = 0
SHUFFLE_SEED_OFFSET = 101

_TABLES: dict[str, LM.PatternTable] = {}
_TABLES_DEV: dict[tuple[str, str], LM.PatternTable] = {}


# ---------------------------------------------------------------------------
# config / data
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


def collate_lf(cache, rows, candidate: str | None = None, *, valid: bool = False):
    """``bf_data.collate_bf`` plus the split tag used to select the pattern table."""
    batch = BFD.collate_bf(cache, rows, candidate or "P3BF", valid=valid)
    batch["split"] = str(cache["split"])
    return batch


def pattern_table(split: str) -> LM.PatternTable:
    """Static (content-keyed) permutation table for one split, built once per process."""
    if split not in _TABLES:
        cache = BFD.load_cache(split)
        _TABLES[split] = LM.build_pattern_table(cache, split=split)
    return _TABLES[split]


def attach_table(model: LM.LocalFactorModel, split: str, device) -> LM.PatternTable:
    key = (split, str(device))
    if key not in _TABLES_DEV:
        _TABLES_DEV[key] = pattern_table(split).to(device)
    table = _TABLES_DEV[key]
    model.set_table(table)
    return table


# ---------------------------------------------------------------------------
# optimiser / schedule
# ---------------------------------------------------------------------------


def optimizable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def make_optimizer(model: torch.nn.Module, cfg: Mapping[str, Any],
                   total_updates: int) -> torch.optim.AdamW:
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
        step, int(total_updates), int(sch["warmup_updates"]),
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
# evaluation
# ---------------------------------------------------------------------------


def evaluate(model: LM.LocalFactorModel, cache, rows, device, batch_size: int,
             split: str | None = None) -> tuple[np.ndarray, float]:
    split = str(cache["split"] if split is None else split)
    attach_table(model, split, device)
    model.eval()
    out = []
    with C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = T1._to_device(collate_lf(cache, chunk, valid=True), device)
            pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    return pred, C.mae(pred, y)


# ---------------------------------------------------------------------------
# one epoch
# ---------------------------------------------------------------------------


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().double().pow(2).sum())
    return math.sqrt(total)


def _l2(module_or_tensor) -> float:
    if isinstance(module_or_tensor, torch.Tensor):
        return float(module_or_tensor.detach().double().pow(2).sum().sqrt())
    total = 0.0
    for p in module_or_tensor.parameters():
        total += float(p.detach().double().pow(2).sum())
    return math.sqrt(total)


def _snapshot(params) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _delta_norm(params, snap) -> float:
    total = 0.0
    for p, s in zip(params, snap):
        total += float((p.detach() - s).double().pow(2).sum())
    return math.sqrt(total)


def step_once(model, optimizer, batch, target, *, cfg, epoch: int) -> dict:
    """One recipe-B AdamW update (single forward / backward / clip / step)."""
    pred = model(batch)
    loss = F.l1_loss(pred.view(-1), target.view(-1))
    if not math.isfinite(float(loss.detach())):
        raise RuntimeError(f"non-finite loss at epoch {epoch}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    g_norm = _grad_norm(optimizable_params(model))
    if not math.isfinite(g_norm):
        raise RuntimeError(f"non-finite grad norm at epoch {epoch}")
    torch.nn.utils.clip_grad_norm_(optimizable_params(model), float(cfg["optimizer"]["grad_clip"]))
    return {"pred": pred.detach(), "loss": float(loss.detach()), "grad_norm": g_norm}


def branch_port_stats(model: LM.LocalFactorModel, batch) -> dict:
    """Cheap magnitude probe of the content port (no gradient, one extra branch pass)."""
    with torch.no_grad():
        s = model.branch(batch, model.table)              # [n_atoms, 198]
        port = model.w_s(s)                               # [n_atoms, 128]
        return {
            "s_rms": float(s.double().pow(2).mean().sqrt()),
            "port_rms": float(port.double().pow(2).mean().sqrt()),
            "port_max_abs": float(port.abs().max()),
        }


def train_epoch(model: LM.LocalFactorModel, optimizer, cache, rows, y, gen, *,
                cfg: Mapping[str, Any], epoch: int, total_updates: int,
                global_step: int, device) -> tuple[dict, int]:
    model.train()
    attach_table(model, str(cache["split"]), device)
    batch_size = int(cfg["schedule"]["batch_size"])
    order = epoch_order(gen, len(rows))
    phi_params = list(model.branch.phi.parameters())
    phi_snap = _snapshot(phi_params)
    total_err, n_mol, grad_norms = 0.0, 0, []
    lr = 0.0
    last_stats: dict | None = None
    t0 = time.perf_counter()
    positions = list(range(0, len(order), batch_size))
    for bi, start in enumerate(positions):
        chunk = order[start:start + batch_size]
        batch = T1._to_device(collate_lf(cache, chunk), device)
        target = torch.as_tensor(y[chunk], dtype=torch.float32, device=device)
        step_index = global_step + 1
        rec = step_once(model, optimizer, batch, target, cfg=cfg, epoch=epoch)
        total_err += float((rec["pred"].view(-1) - target).abs().sum())
        n_mol += len(chunk)
        grad_norms.append(rec["grad_norm"])
        global_step = step_index
        lr = set_lr(optimizer, global_step, total_updates, cfg)
        optimizer.step()
        if bi == len(positions) - 1:
            with torch.no_grad():
                last_stats = branch_port_stats(model, batch)
    seconds = time.perf_counter() - t0
    if global_step != epoch * len(positions):
        raise RuntimeError(f"outer update counter drifted: {global_step}")
    rec = {
        "epoch": int(epoch),
        "online_train_mae": float(total_err / max(n_mol, 1)),
        "lr": float(lr),
        "seconds": float(seconds),
        "steps": int(len(positions)),
        "order_sha256": order_sha256(order),
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_max": float(np.max(grad_norms)),
        "grad_clip_fraction": float(np.mean([g > float(cfg["optimizer"]["grad_clip"])
                                             for g in grad_norms])),
        "w_s_norm": _l2(model.w_s.weight),
        "phi_norm": _l2(model.branch.phi),
        "phi_epoch_delta": _delta_norm(phi_params, phi_snap),
        "atom_emb_norm": _l2(model.branch.atom_emb.weight),
        **(last_stats or {}),
    }
    return rec, global_step


# ---------------------------------------------------------------------------
# run helpers
# ---------------------------------------------------------------------------


def build(arm: str, norm, device, seed: int = SEED):
    return LM.build_model(arm, norm, seed=seed, device=device)


def init_hashes(model: LM.LocalFactorModel) -> dict:
    return {
        "init_state_sha256": C.state_hash(model),
        "public_state_sha256": LM.hash_tensors(LM.public_state(model)),
        "shared_embedding_sha256": LM.hash_tensors(LM.shared_embedding_state(model)),
        "w_s_zero": bool(model.w_s.weight.abs().sum().item() == 0.0),
    }


def probe_signature(model: LM.LocalFactorModel, cache, rows, device, split: str,
                    batch_size: int = 64) -> dict:
    """Deterministic prediction digest used to compare initial functions."""
    pred, mae = evaluate(model, cache, rows, device, batch_size, split=split)
    return {"n": int(len(pred)), "pred_sha256": C.sha256_array(pred), "mae": float(mae)}
