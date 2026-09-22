#!/usr/bin/env python
"""TCCD-v2 GRAD-CONT: label-free ordinal local-geometry regularizer.

Single diagnostic intervention on top of the frozen TCCD-v2 Prototype-REL
architecture. Pre-registration:
``tracks/ksvd/notes/tccd_v2_gradcont_preregistration.md``.

Architecture, optimizer, split, stopping protocol and all existing
regularizers are reused unchanged. The only change is one added label-free
ordinal term on ``z``:

    L_grad = mean softplus(cos(z_i, z_far) - cos(z_i, z_close))

over deterministic cross-molecule training triplets ``close ~< far`` in the
Pareto partial order on ``(d1, d2)`` edit counts defined in
``tracks/ksvd/code/tccd_v2_local_chem.py``. ``lambda_grad`` is calibrated once
from detached initial task/ordinal magnitudes (0.05 initial task share) and
then frozen. No scalar chemical distance, no shell weighting, no margin.

The chemistry decode and ordinal relation are shared with the audit through
``tccd_v2_local_chem`` so training and audit use one definition.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V
from tracks.ksvd.code import tccd_v2_local_chem as LC

PROTOCOL_VERSION = "tccd_v2_gradcont"
RESULTS_DIR = V.RESULTS_DIR.parent / "tccd_v2_gradcont"
CACHE_DIR = RESULTS_DIR / "cache"

PAIR_SEED = LC.ORDINAL_SEED
LAMBDA_SHARE = 0.05
CTYPES = ("same_d1", "same_d2", "both")


# ===========================================================================
# ordinal loss + training loop
# ===========================================================================
def _gather_rows(trip_by_pos: Sequence[np.ndarray], positions: Sequence[int]) -> np.ndarray:
    chunks = [trip_by_pos[int(p)] for p in positions if trip_by_pos[int(p)].size]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def _trips_by_position(anchor_pos: np.ndarray, n_positions: int) -> list[np.ndarray]:
    out: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_positions)]
    order = np.argsort(anchor_pos, kind="stable")
    if order.size == 0:
        return out
    ua, starts = np.unique(anchor_pos[order], return_index=True)
    for k, pos in enumerate(ua):
        st = int(starts[k])
        en = int(starts[k + 1]) if k + 1 < len(starts) else int(order.size)
        out[int(pos)] = order[st:en]
    return out


def _patch_matrix(records: Sequence[Mapping[str, Any]], train_indices: Sequence[int], device: str):
    torch = V._torch()
    X = np.concatenate(
        [np.asarray(records[int(gi)]["X"], dtype=np.float32) for gi in train_indices], axis=0
    )
    return torch.as_tensor(X, dtype=torch.float32, device=device)


def _ordinal_loss(za, zj, zk):
    torch = V._torch()
    F = torch.nn.functional
    za = F.normalize(za, dim=-1, eps=V.EPS)
    zj = F.normalize(zj, dim=-1, eps=V.EPS)
    zk = F.normalize(zk, dim=-1, eps=V.EPS)
    s_close = (za * zj).sum(dim=-1)
    s_far = (za * zk).sum(dim=-1)
    return F.softplus(s_far - s_close).mean()


def _ordinal_loss_rows(model, Xpatch, tri_a, tri_c, tri_f):
    za = Xpatch.index_select(0, tri_a) @ model.W
    zc = Xpatch.index_select(0, tri_c) @ model.W
    zf = Xpatch.index_select(0, tri_f) @ model.W
    return _ordinal_loss(za, zc, zf)


def initial_regularization_ordinal(
    model,
    records_train,
    train_indices,
    device,
    *,
    batch,
    Xpatch,
    tri_a,
    tri_c,
    tri_f,
    trip_by_pos,
    log=print,
) -> dict[str, Any]:
    """Reuse the frozen TCCD-v2 calibration, then calibrate lambda_grad once."""
    base = V._initial_regularization(model, records_train, train_indices, device, batch=batch, log=log)
    positions = list(range(min(int(batch), len(train_indices))))
    rows = _gather_rows(trip_by_pos, positions)
    torch = V._torch()
    with torch.no_grad():
        if rows.size:
            rt = torch.as_tensor(rows, dtype=torch.long, device=device)
            grad = float(_ordinal_loss_rows(model, Xpatch, tri_a[rt], tri_c[rt], tri_f[rt]))
        else:
            grad = 0.0
    lam = LAMBDA_SHARE * float(base["initial_task"]) / max(grad, T.EPS)
    base.update(
        {
            "initial_grad": grad,
            "lambda_grad": float(lam),
            "lambda_grad_share": LAMBDA_SHARE,
            "initial_grad_contribution": float(lam * grad),
            "calibration_triplets": int(rows.size),
        }
    )
    log(
        "  [grad-calibration] "
        + str(
            {
                "initial_task": round(base["initial_task"], 6),
                "initial_grad": round(grad, 6),
                "lambda_grad": round(lam, 6),
                "initial_grad_contribution": round(lam * grad, 6),
                "calibration_triplets": int(rows.size),
            }
        )
    )
    return base


@dataclass
class GradContTrainResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    train_history: list[dict[str, Any]] = field(default_factory=list)
    state_best: Any = None
    state_soup: Any = None
    regularization: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    peak_mem_mb: float | None = None


def train_model_gradcont(
    model,
    records_train,
    records_dev,
    train_indices,
    dev_indices,
    device,
    *,
    triplets: Mapping[str, Any],
    seed: int = 0,
    max_epochs: int = T.MAX_EPOCHS,
    patience: int = T.PATIENCE,
    batch: int = T.BATCH,
    lr: float = T.LR,
    wd: float = T.WD,
    clip: float = T.CLIP,
    lambda_grad: float | None = None,
    log=print,
) -> GradContTrainResult:
    """TCCD-v2 Prototype-REL training plus one frozen ordinal-geometry term.

    With ``lambda_grad=0`` every operation is identical to ``tccd_v2.train_model``
    for the same seed/init (the ordinal term is skipped and consumes no RNG).
    """
    torch = V._torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    Xpatch = _patch_matrix(records_train, train_indices, device)
    trip_by_pos = _trips_by_position(np.asarray(triplets["anchor_pos"], dtype=np.int64), len(train_indices))
    tri_a = torch.as_tensor(np.asarray(triplets["anchor"], dtype=np.int64), dtype=torch.long, device=device)
    tri_c = torch.as_tensor(np.asarray(triplets["closer"], dtype=np.int64), dtype=torch.long, device=device)
    tri_f = torch.as_tensor(np.asarray(triplets["farther"], dtype=np.int64), dtype=torch.long, device=device)

    reg = initial_regularization_ordinal(
        model,
        records_train,
        train_indices,
        device,
        batch=batch,
        Xpatch=Xpatch,
        tri_a=tri_a,
        tri_c=tri_c,
        tri_f=tri_f,
        trip_by_pos=trip_by_pos,
        log=log,
    )
    lam_local = float(reg["lambda_local"])
    lam_balance = float(reg["lambda_balance"])
    lam_grad = float(reg["lambda_grad"]) if lambda_grad is None else float(lambda_grad)
    reg["lambda_grad_applied"] = lam_grad
    reg["initial_grad_contribution_applied"] = lam_grad * float(reg["initial_grad"])

    best = math.inf
    best_epoch = -1
    best_state = None
    top: list[tuple[float, int, Any]] = []
    history: list[dict[str, Any]] = []
    stale = 0
    t0 = time.time()
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(train_indices))
        ep_total = ep_task = ep_local = ep_balance = ep_grad = 0.0
        ep_n = 0
        for start in range(0, len(order), batch):
            positions = order[start : start + batch]
            sel = [train_indices[i] for i in positions]
            b = V.make_batch(records_train, sel, device)
            opt.zero_grad(set_to_none=True)
            pred, C_flat, _ = model.forward_padded(b)
            valid = b["valid"].reshape(-1)
            task = (pred - b["y"]).abs().mean()
            local = V.local_entropy(C_flat, valid)
            balance = V.balance_kl(C_flat, valid)
            if lam_grad > 0.0:
                rows = _gather_rows(trip_by_pos, positions)
                if rows.size:
                    rt = torch.as_tensor(rows, dtype=torch.long, device=device)
                    grad = _ordinal_loss_rows(model, Xpatch, tri_a[rt], tri_c[rt], tri_f[rt])
                else:
                    grad = torch.zeros((), device=device)
            else:
                grad = torch.zeros((), device=device)
            loss = task + lam_local * local + lam_balance * balance + lam_grad * grad
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            model.renormalize_()
            count = len(sel)
            ep_total += float(loss.detach()) * count
            ep_task += float(task.detach()) * count
            ep_local += float(local.detach()) * count
            ep_balance += float(balance.detach()) * count
            ep_grad += float(grad.detach()) * count
            ep_n += count
        dev_mae = V.evaluate_mae(model, records_dev, dev_indices, device, batch=64)
        row = {
            "epoch": epoch,
            "train_loss": ep_total / max(ep_n, 1),
            "train_task": ep_task / max(ep_n, 1),
            "train_local_entropy": ep_local / max(ep_n, 1),
            "train_balance_kl": ep_balance / max(ep_n, 1),
            "train_grad": ep_grad / max(ep_n, 1),
            "valid": dev_mae,
            "temperature": float(model.temperature().detach()),
        }
        history.append(row)
        log(
            f"  epoch={epoch:03d} train={row['train_loss']:.6f} task={row['train_task']:.6f} "
            f"grad={row['train_grad']:.6f} valid={dev_mae:.6f} tau={row['temperature']:.5f} best@{best_epoch}"
        )
        if dev_mae < best - 1e-9:
            best = dev_mae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if len(top) < T.TOP_K_SOUP or dev_mae < max(t[0] for t in top):
            top.append((dev_mae, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}))
            top.sort(key=lambda t: t[0])
            top = top[: T.TOP_K_SOUP]
        if stale >= int(patience):
            log(f"  early stop at epoch {epoch}")
            break

    soup = None
    soup_mae = None
    members: list[int] = []
    if top:
        members = [e for _, e, _ in top]
        keys = top[0][2].keys()
        soup = {k: sum(st[k].float() for _, _, st in top) / len(top) for k in keys}
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(soup)
        soup_mae = V.evaluate_mae(model, records_dev, dev_indices, device, batch=64)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return GradContTrainResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        train_history=history,
        state_best=best_state,
        state_soup=soup,
        regularization=reg,
        wall_s=time.time() - t0,
    )


def triplets_cache_paths(seed: int = PAIR_SEED) -> tuple[Path, Path]:
    return (
        CACHE_DIR / f"gradcont_triplets_seed{seed}.npz",
        CACHE_DIR / f"gradcont_triplets_seed{seed}_stats.json",
    )


def save_triplets(triplets: Mapping[str, Any], seed: int = PAIR_SEED) -> tuple[Path, Path]:
    npz, js = triplets_cache_paths(seed)
    LC.save_triplets(triplets, npz)
    js.write_text(json.dumps(triplets["stats"], indent=2, sort_keys=True, default=float))
    return npz, js


def load_triplets(seed: int = PAIR_SEED) -> dict[str, Any] | None:
    npz, js = triplets_cache_paths(seed)
    if not npz.exists() or not js.exists():
        return None
    obj = LC.load_triplets(npz)
    obj["stats"] = json.loads(js.read_text())
    return obj
