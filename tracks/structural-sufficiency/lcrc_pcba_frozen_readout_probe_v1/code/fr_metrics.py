"""Numerical helpers for the frozen-readout probe (all CPU-testable).

Nothing here touches the frozen encoder.  The metric functions are thin wrappers
over the already-audited ``pc_metrics`` module so that the frozen baseline and
every residual-head epoch are scored by exactly the same code path that produced
the historical numbers.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import fr_common as C
import pc_metrics as MET

# ---------------------------------------------------------------------------
# standardisation (Train-only statistics, safe zero-variance handling)
# ---------------------------------------------------------------------------

ZERO_VARIANCE_EPS = 1e-12


def fit_standardizer(x: torch.Tensor, *, chunk: int = 65_536) -> dict[str, Any]:
    """Per-column mean/std of ``x`` (Train rows only), fp64 accumulation.

    ``x`` is not modified.  Returns float32 tensors ``mean``/``scale`` on the
    same device plus the audit block.  Columns whose population std is
    ``<= 1e-12`` get ``scale = 1.0`` and are listed explicitly, so a constant
    column stays constant instead of exploding.
    """
    if x.ndim != 2:
        raise ValueError(f"standardizer expects a 2-D tensor, got {tuple(x.shape)}")
    n, d = int(x.shape[0]), int(x.shape[1])
    if n <= 0:
        raise ValueError("cannot fit a standardizer on zero rows")
    total = torch.zeros(d, dtype=torch.float64, device=x.device)
    total_sq = torch.zeros(d, dtype=torch.float64, device=x.device)
    for start in range(0, n, int(chunk)):
        block = x[start : start + int(chunk)]
        total += torch.sum(block, dim=0, dtype=torch.float64)
        total_sq += torch.sum(block * block, dim=0, dtype=torch.float64)
    mean = total / float(n)
    var = total_sq / float(n) - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)
    zero = std <= ZERO_VARIANCE_EPS
    scale = torch.where(zero, torch.ones_like(std), std)
    mean32 = mean.to(torch.float32)
    scale32 = scale.to(torch.float32)
    return {
        "mean": mean32,
        "scale": scale32,
        "n_rows_fit": n,
        "dim": d,
        "mean_ddof": 0,
        "accumulation": "float64 sums over float32 blocks (elementwise square in float32)",
        "zero_variance_epsilon": ZERO_VARIANCE_EPS,
        "zero_variance_columns": [int(i) for i in torch.nonzero(zero, as_tuple=False).flatten().tolist()],
        "n_zero_variance_columns": int(zero.sum().item()),
        "raw_min_std": float(std.min().item()),
        "raw_max_std": float(std.max().item()),
        "scale_is_one_where_zero_variance": bool(torch.all(scale32[zero] == 1.0).item())
        if bool(zero.any())
        else True,
    }


def standardize_(x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, *, chunk: int = 65_536) -> torch.Tensor:
    """In-place standardisation of ``x`` in chunks; returns ``x`` itself."""
    if x.ndim != 2 or x.shape[1] != mean.numel() or mean.numel() != scale.numel():
        raise ValueError(
            f"standardize shape mismatch: x={tuple(x.shape)} mean={tuple(mean.shape)} scale={tuple(scale.shape)}"
        )
    mean = mean.to(device=x.device, dtype=x.dtype)
    scale = scale.to(device=x.device, dtype=x.dtype)
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(chunk)):
            stop = min(start + int(chunk), int(x.shape[0]))
            block = x[start:stop]
            block.sub_(mean).div_(scale)
    return x


def column_stats(x: torch.Tensor, *, chunk: int = 65_536) -> dict[str, Any]:
    """Mean/std/max-abs summary used to audit the standardised cache."""
    n, d = int(x.shape[0]), int(x.shape[1])
    total = torch.zeros(d, dtype=torch.float64, device=x.device)
    total_sq = torch.zeros(d, dtype=torch.float64, device=x.device)
    max_abs = 0.0
    for start in range(0, n, int(chunk)):
        block = x[start : start + int(chunk)]
        total += torch.sum(block, dim=0, dtype=torch.float64)
        total_sq += torch.sum(block * block, dim=0, dtype=torch.float64)
        max_abs = max(max_abs, float(block.abs().max().item()))
    mean = total / float(n)
    var = torch.clamp(total_sq / float(n) - mean * mean, min=0.0)
    return {
        "rows": n,
        "dim": d,
        "column_mean_abs_max": float(mean.abs().max().item()),
        "column_std_min": float(torch.sqrt(var).min().item()),
        "column_std_max": float(torch.sqrt(var).max().item()),
        "abs_max": float(max_abs),
    }


# ---------------------------------------------------------------------------
# masked BCE (every observed entry equally weighted)
# ---------------------------------------------------------------------------


def masked_bce_torch(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``BCEWithLogits`` over observed entries only; NaN targets are excluded."""
    if logits.shape != y.shape:
        raise ValueError(f"BCE logits {tuple(logits.shape)} != target {tuple(y.shape)}")
    mask = torch.isfinite(y)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask])


def masked_bce_numpy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(MET.masked_bce_numpy(logits, y))


# ---------------------------------------------------------------------------
# AP / task-level tables
# ---------------------------------------------------------------------------


def task_table(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Per-task AP + coverage via the inherited ``pc_metrics`` implementation."""
    return MET.per_task_ap(y, probs)


def official_ap(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    return MET.official_ap(y, probs)


def macro_ap_from_tasks(per_task: Sequence[float | None]) -> float:
    vals = [float(v) for v in per_task if v is not None and np.isfinite(v)]
    if not vals:
        raise ValueError("no finite per-task AP")
    return float(np.mean(vals))


def delta_summary(base: Sequence[float | None], new: Sequence[float | None]) -> dict[str, Any]:
    """Paired per-task AP comparison over tasks scorable in both columns."""
    if len(base) != len(new):
        raise ValueError(f"delta length mismatch {len(base)} != {len(new)}")
    pairs = [
        (t, float(b), float(n))
        for t, (b, n) in enumerate(zip(base, new))
        if b is not None and n is not None and np.isfinite(b) and np.isfinite(n)
    ]
    if not pairs:
        raise ValueError("no paired scorable tasks")
    deltas = np.asarray([n - b for _, b, n in pairs], dtype=np.float64)
    order = np.argsort(-deltas)
    total = float(deltas.sum())
    top1 = float(deltas[order[:1]].sum())
    top5 = float(deltas[order[:5]].sum())
    worst = np.argsort(deltas)
    return {
        "n_paired_tasks": len(pairs),
        "n_improved": int((deltas > 0).sum()),
        "n_worse": int((deltas < 0).sum()),
        "n_tied": int((deltas == 0).sum()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "sum_delta": total,
        "min_delta": float(deltas.min()),
        "max_delta": float(deltas.max()),
        "top1_delta_share_of_sum": None if total == 0 else float(top1 / total),
        "top5_delta_share_of_sum": None if total == 0 else float(top5 / total),
        "tasks": [int(pairs[i][0]) for i in order],
        "deltas_by_task": [float(deltas[i]) for i in order],
        "largest_delta_task": int(pairs[int(order[0])][0]),
        "largest_delta": float(deltas[order[0]]),
        "smallest_delta_task": int(pairs[int(worst[0])][0]),
        "smallest_delta": float(deltas[worst[0]]),
    }


def gap_summary(train_ap: float, valid_ap: float) -> dict[str, float]:
    return {"train_ap": float(train_ap), "valid_ap": float(valid_ap), "train_minus_valid": float(train_ap) - float(valid_ap)}


# ---------------------------------------------------------------------------
# paired row order (identical for both arms)
# ---------------------------------------------------------------------------


def row_plan(
    n_rows: int,
    *,
    batch_size: int = C.BATCH_SIZE,
    epochs: int = C.EPOCHS,
    seed: int = C.SEED,
) -> dict[str, Any]:
    """One CPU generator, ``epochs`` permutations, fixed batch size."""
    n_rows = int(n_rows)
    generator = torch.Generator().manual_seed(int(seed))
    perms = []
    batch_hashes = []
    for _ in range(int(epochs)):
        perm = torch.randperm(n_rows, generator=generator).numpy().astype(np.int64)
        batches = [perm[i : i + int(batch_size)] for i in range(0, n_rows, int(batch_size))]
        perms.append(perm)
        batch_hashes.append(C.sha256_batches(batches))
    row_hashes = [C.sha256_int64(p) for p in perms]
    global_sha = C.sha256_bytes("|".join(row_hashes).encode())
    return {
        "perms": perms,
        "n_rows": n_rows,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "seed": int(seed),
        "steps_per_epoch": int(math.ceil(n_rows / int(batch_size))),
        "row_order_sha256": row_hashes,
        "batch_plan_sha256": batch_hashes,
        "global_plan_sha256": global_sha,
        "source": "torch.Generator().manual_seed(seed) + torch.randperm(n_rows) per epoch",
    }


def plan_isolated(plan: Mapping[str, Any], forbidden_rows: Iterable[int], n_total: int) -> dict[str, Any]:
    """Check that no forbidden (Validation/Test) row can enter the training order."""
    mask = np.zeros(int(n_total), dtype=bool)
    mask[np.asarray(list(forbidden_rows), dtype=np.int64)] = True
    bad = [(i, int(mask[p].sum())) for i, p in enumerate(plan["perms"]) if int(mask[p].sum())]
    return {
        "passed": not bad,
        "epochs_checked": len(plan["perms"]),
        "forbidden_rows": int(mask.sum()),
        "violations": [{"epoch": i + 1, "n_forbidden": n} for i, n in bad],
    }


# ---------------------------------------------------------------------------
# residual heads
# ---------------------------------------------------------------------------


def head_parameter_fingerprint(head: torch.nn.Module) -> dict[str, Any]:
    rows = []
    for name, param in head.named_parameters():
        arr = param.detach().cpu().contiguous().numpy()
        rows.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": C.sha256_array(arr),
            }
        )
    h = C.sha256_bytes("|".join(f"{r['name']}:{r['sha256']}" for r in rows).encode())
    return {
        "n_tensors": len(rows),
        "n_parameters": int(sum(int(np.prod(r["shape"])) for r in rows)),
        "combined_sha256": h,
        "tensors": rows,
    }


def build_head(arm: str, d: int, n_tasks: int = C.N_TASKS, *, seed: int = C.SEED) -> torch.nn.Module:
    """Deterministic residual head; the output layer is exactly zero at init."""
    if arm not in C.ARMS:
        raise KeyError(f"unknown arm {arm!r}")
    C.seed_everything(int(seed))
    if arm == "L":
        head: torch.nn.Module = torch.nn.Linear(int(d), int(n_tasks))
        last = head
    elif arm == "N":
        first = torch.nn.Linear(int(d), int(n_tasks))
        last = torch.nn.Linear(int(n_tasks), int(n_tasks))
        head = torch.nn.Sequential(first, torch.nn.SiLU(), last)
    else:  # pragma: no cover - guarded above
        raise KeyError(arm)
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    return head


def zero_init_audit(head: torch.nn.Module, x_sample: torch.Tensor, frozen_sample: torch.Tensor) -> dict[str, Any]:
    """The initial residual must be exactly zero and the initial logits exactly frozen."""
    head.eval()
    with torch.no_grad():
        resid = head(x_sample)
        new_logits = frozen_sample + resid
    resid_abs_max = float(resid.abs().max().item()) if resid.numel() else 0.0
    diff = (new_logits - frozen_sample).abs()
    return {
        "residual_abs_max": resid_abs_max,
        "residual_is_exactly_zero": bool(resid_abs_max == 0.0),
        "logits_bitwise_equal_frozen": bool(torch.equal(new_logits, frozen_sample)),
        "logits_max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "sample_rows": int(x_sample.shape[0]),
    }


__all__ = [
    "ZERO_VARIANCE_EPS",
    "build_head",
    "column_stats",
    "delta_summary",
    "fit_standardizer",
    "gap_summary",
    "head_parameter_fingerprint",
    "macro_ap_from_tasks",
    "masked_bce_numpy",
    "masked_bce_torch",
    "official_ap",
    "plan_isolated",
    "row_plan",
    "standardize_",
    "task_table",
    "zero_init_audit",
]
