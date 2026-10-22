"""Shared pipeline for the three arms of ``zinc_shared_factor_v1`` (D, T, N).

This module owns *one* training loop, evaluator, optimiser/schedule and batch
order, used by all arms, so an arm difference cannot come from the harness.

Exactly two things are reused from ``zinc_local_factorization_v1``:

* ``lf_core`` for the collate, static pattern table, optimiser, per-step cosine
  schedule, epoch order generator, evaluator and the single AdamW update
  (:func:`lf_core.step_once`, i.e. forward -> raw-y L1 -> backward -> grad-norm
  clip 5 -> AdamW step);
* ``lf_models.ContentBranch("D")`` (via :class:`sf_models.SFModel`) for the D
  arm, so D *is* the previous round's DensePhi arm on the same training path.

``train_epoch`` below is a copy of ``lf_core.train_epoch`` with the same
mathematics and ordering; the only edit is that the logged parameter probes are
arm aware (D has no atomic state; T/N have no content embedding table) and the
counter/consistency assertions are identical.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_LF = _SS / "zinc_local_factorization_v1" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_BF), str(_LF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_train as T1  # noqa: E402
import lf_core as LC  # noqa: E402
import lf_models as LM  # noqa: E402
import sf_models as SF  # noqa: E402

EXP = _HERE.parent
CONFIG_PATH = EXP / "configs" / "sf_atomic_v1.json"
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
PROTOCOL = "zinc-shared-factor-v1"

SEEDS = (0, 1)
SHUFFLE_SEED_OFFSET = 101

# Re-export the pieces that are literally shared with the previous round.
collate_lf = LC.collate_lf
attach_table = LC.attach_table
pattern_table = LC.pattern_table
evaluate = LC.evaluate
make_optimizer = LC.make_optimizer
schedule_lr = LC.schedule_lr
set_lr = LC.set_lr
order_sha256 = LC.order_sha256
epoch_order = LC.epoch_order
step_once = LC.step_once
branch_port_stats = LC.branch_port_stats
load_split = LC.load_split


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    cfg = C.read_json(path)
    cfg["_config_path"] = str(Path(path))
    cfg["_config_sha256"] = C.sha256_file(Path(path))
    return cfg


# ---------------------------------------------------------------------------
# arm-aware probes (logging only; no effect on the update)
# ---------------------------------------------------------------------------


def content_modules(model) -> list[torch.nn.Module]:
    if model.arm == "D":
        return [model.branch.phi]
    return [model.branch.state, model.branch.l1, model.branch.l2]


def content_params(model) -> list[torch.nn.Parameter]:
    out: list[torch.nn.Parameter] = []
    for module in content_modules(model):
        out.extend(list(module.parameters()))
    return out


def _l2(module_or_tensor) -> float:
    return LC._l2(module_or_tensor)


def _snapshot(params) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _delta_norm(params, snap) -> float:
    return LC._delta_norm(params, snap)


# ---------------------------------------------------------------------------
# one epoch
# ---------------------------------------------------------------------------


def train_epoch(model, optimizer, cache, rows, y, gen, *, cfg: Mapping[str, Any],
                epoch: int, total_updates: int, global_step: int, device) -> tuple[dict, int]:
    model.train()
    attach_table(model, str(cache["split"]), device)
    batch_size = int(cfg["schedule"]["batch_size"])
    order = epoch_order(gen, len(rows))
    params = content_params(model)
    snap = _snapshot(params)
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
        "content_norm": _l2_params(params),
        "content_epoch_delta": _delta_norm(params, snap),
        "atom_emb_norm": float(_l2(model.branch.atom_emb.weight))
        if model.arm == "D" else 0.0,
        **(last_stats or {}),
    }
    return rec, global_step


def _l2_params(params) -> float:
    total = 0.0
    for p in params:
        total += float(p.detach().double().pow(2).sum())
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# build / hashes
# ---------------------------------------------------------------------------


def build(arm: str, norm, device, seed: int = 0):
    return SF.build_model(arm, norm, seed=seed, device=device)


def init_hashes(model) -> dict:
    state = SF.content_state(model)
    return {
        "init_state_sha256": C.state_hash(model),
        "public_state_sha256": LM.hash_tensors(LM.public_state(model)),
        "content_state_sha256": SF.hash_tensors(state),
        "w_s_zero": bool(model.w_s.weight.abs().sum().item() == 0.0),
    }


def probe_signature(model, cache, rows, device, split: str, batch_size: int = 64) -> dict:
    pred, mae = evaluate(model, cache, rows, device, batch_size, split=split)
    return {"n": int(len(pred)), "pred_sha256": C.sha256_array(pred), "mae": float(mae)}
