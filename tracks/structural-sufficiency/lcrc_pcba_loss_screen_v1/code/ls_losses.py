"""Frozen loss definitions and independent CPU numerical checks.

The control loss **M** is the historical ``pc_metrics.masked_bce_loss``
verbatim.  The experimental loss **T** is the pre-registered task-macro
estimator

    L_T = N / (B * T) * sum_{i,t} (m_it * ell_it / n_t)

where ``N`` is the number of official Train molecules, ``B`` the actual
minibatch size, ``T = 128``, ``m_it`` the observed-label mask, ``ell_it`` the
unreduced BCEWithLogits, and ``n_t`` the number of observed Train labels for
task ``t``.  ``n_t`` is a fixed vector computed from the complete official
Train split only.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

import ls_common as C

# The inherited implementation is imported, not reimplemented, so the control
# arm cannot silently drift.
import pc_metrics as MET


def task_counts(labels_train: np.ndarray) -> np.ndarray:
    """Number of observed Train labels per task (``n_t``), from Train only."""
    labels_train = np.asarray(labels_train)
    if labels_train.ndim != 2 or labels_train.shape[1] != C.N_TASKS:
        raise ValueError(f"Train labels {labels_train.shape} != (N, {C.N_TASKS})")
    counts = np.isfinite(labels_train).sum(axis=0).astype(np.int64)
    if counts.shape != (C.N_TASKS,):
        raise RuntimeError(f"task-count vector {counts.shape} != ({C.N_TASKS},)")
    if int(counts.sum()) <= 0:
        raise RuntimeError("Train split has no observed labels")
    if not np.all(counts > 0):
        missing = np.flatnonzero(counts <= 0).tolist()
        raise RuntimeError(f"tasks with zero observed Train labels: {missing}")
    return counts


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Control loss M: the exact inherited masked BCE."""
    return MET.masked_bce_loss(logits, y)


def task_macro_bce_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_task: torch.Tensor | np.ndarray,
    *,
    n_train: int = C.N_TRAIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental loss T plus the unified raw masked BCE for logging.

    Returns ``(task_macro_loss, raw_masked_bce)``.  Invalid labels are replaced
    by a safe value *before* the elementwise BCE; the mask is applied only
    afterwards.  No NaN can enter the loss or its gradient.
    """
    if logits.shape != y.shape:
        raise ValueError(f"BCE logits {tuple(logits.shape)} != target {tuple(y.shape)}")
    if logits.ndim != 2:
        raise ValueError(f"expected [B, T] logits, got {tuple(logits.shape)}")
    B, T = int(logits.shape[0]), int(logits.shape[1])
    if B <= 0:
        raise ValueError("empty minibatch")
    if int(n_train) <= 0:
        raise ValueError(f"n_train {n_train} must be positive")

    n_task_t = torch.as_tensor(n_task, dtype=logits.dtype, device=logits.device)
    if tuple(n_task_t.shape) != (T,):
        raise ValueError(f"n_task shape {tuple(n_task_t.shape)} != ({T},)")
    if not bool(torch.isfinite(n_task_t).all()) or not bool((n_task_t > 0).all()):
        raise ValueError("n_task must be finite and strictly positive for every task")

    mask = torch.isfinite(y)
    # Safe replacement first; the mask is applied after the elementwise BCE.
    y_safe = torch.where(mask, y, torch.zeros_like(y))
    ell = F.binary_cross_entropy_with_logits(logits, y_safe, reduction="none")
    if not bool(torch.isfinite(ell[mask]).all()):
        raise RuntimeError("non-finite elementwise BCE on observed entries")

    coefficient = float(n_train) / float(B * T)
    weights = coefficient * (mask.to(dtype=ell.dtype) / n_task_t.view(1, T))
    loss = (ell * weights).sum()
    observed = mask.sum()
    raw_masked = (ell * mask.to(dtype=ell.dtype)).sum() / observed.clamp_min(1)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"non-finite task-macro loss: {float(loss)}")
    if not bool(torch.isfinite(raw_masked)):
        raise RuntimeError(f"non-finite raw masked BCE: {float(raw_masked)}")
    return loss, raw_masked


def _numpy_bce(logits: np.ndarray, y_safe: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(y_safe, dtype=np.float64)
    return np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z)))


def numpy_task_macro_reference(
    logits: np.ndarray,
    y: np.ndarray,
    n_task: Iterable[int],
    *,
    n_train: int = C.N_TRAIN,
) -> float:
    """Independent float64 reference for the T minibatch formula."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_task = np.asarray(list(n_task), dtype=np.float64)
    mask = np.isfinite(y)
    y_safe = np.where(mask, y, 0.0)
    ell = _numpy_bce(logits, y_safe)
    B, T = logits.shape
    value = float(n_train) / float(B * T) * float(np.sum(mask * ell / n_task.reshape(1, T)))
    return value


def numpy_full_task_average(
    logits: np.ndarray,
    y: np.ndarray,
    n_task: Iterable[int],
) -> float:
    """Full-dataset task-macro objective: mean_t mean over observed labels."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_task = np.asarray(list(n_task), dtype=np.float64)
    mask = np.isfinite(y)
    ell = _numpy_bce(logits, np.where(mask, y, 0.0))
    T = logits.shape[1]
    per_task = []
    for t in range(T):
        m = mask[:, t]
        if not bool(m.any()):
            raise ValueError(f"task {t} has no observed entries in the full data")
        per_task.append(float(np.sum(ell[:, t] * m) / n_task[t]))
    return float(np.mean(per_task))


def numpy_micro_reference(logits: np.ndarray, y: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y)
    ell = _numpy_bce(logits, np.where(mask, y, 0.0))
    if not bool(mask.any()):
        return 0.0
    return float(np.sum(ell[mask]) / np.sum(mask))


def _tensor_close(a: float, b: float, tol: float = 1e-6) -> bool:
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return bool(abs(float(a) - float(b)) <= tol * scale)


def _relative_abs_diff(a: float, b: float) -> float:
    return float(abs(float(a) - float(b)) / max(1.0, abs(float(a)), abs(float(b))))


def cpu_validation_suite() -> dict[str, Any]:
    """Independent CPU checks required before GPU smoke/formal training."""
    torch.manual_seed(7)
    cases: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- case 1
    # Heterogeneous coverage, one task completely unlabelled in the batch,
    # positive and negative examples inside the same task, NaN-safe.
    y1 = torch.tensor(
        [
            [1.0, 0.0, float("nan"), 1.0],
            [0.0, 1.0, float("nan"), float("nan")],
            [float("nan"), 0.0, float("nan"), 0.0],
            [1.0, float("nan"), float("nan"), 1.0],
            [0.0, 1.0, float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    logits1 = torch.tensor(
        [
            [0.2, -1.1, 0.7, 2.0],
            [-0.4, 0.8, -0.3, -1.5],
            [1.3, -0.2, 0.1, -0.7],
            [0.9, 1.1, -0.4, 0.3],
            [-1.2, 0.4, 0.6, 0.8],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    n1 = np.array([10, 20, 30, 40], dtype=np.int64)
    loss1, raw1 = task_macro_bce_loss(logits1, y1, n1)
    ref1 = numpy_task_macro_reference(logits1.detach().numpy(), y1.numpy(), n1)
    loss1.backward()
    grad1 = logits1.grad.detach()
    cases.append(
        {
            "name": "heterogeneous_batch_with_fully_unlabelled_task",
            "n_task": n1.tolist(),
            "batch_size": int(logits1.shape[0]),
            "loss": float(loss1),
            "raw_masked_bce": float(raw1),
            "numpy_reference": ref1,
            "abs_diff": abs(float(loss1) - ref1),
            "relative_abs_diff": _relative_abs_diff(float(loss1), ref1),
            "loss_finite": bool(torch.isfinite(loss1)),
            "grad_all_finite": bool(torch.isfinite(grad1).all()),
            "grad_fully_unlabelled_task_max_abs": float(grad1[:, 2].abs().max()),
            "same_task_has_positive_and_negative": bool(
                (y1[:, 0] == 1).any() and (y1[:, 0] == 0).any()
            ),
        }
    )

    # ---------------------------------------------------------------- case 2
    # Different actual batch sizes; the formula must use B, not a constant.
    n2 = np.array([5, 7, 11, 13], dtype=np.int64)
    y2 = torch.tensor(
        [
            [1.0, 0.0, float("nan"), 1.0],
            [0.0, 1.0, 1.0, float("nan")],
            [float("nan"), 0.0, 1.0, 0.0],
            [1.0, 1.0, float("nan"), 1.0],
            [0.0, 0.0, 1.0, float("nan")],
            [1.0, float("nan"), 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, float("nan"), 0.0],
        ],
        dtype=torch.float32,
    )
    logits2 = torch.linspace(-1.5, 1.5, steps=32, dtype=torch.float32).reshape(8, 4)
    batch_rows = [(0, 4), (4, 8)]
    batch_report = []
    for start, stop in batch_rows:
        lb = logits2[start:stop].clone().requires_grad_(True)
        yb = y2[start:stop]
        loss, raw = task_macro_bce_loss(lb, yb, n2)
        ref = numpy_task_macro_reference(lb.detach().numpy(), yb.numpy(), n2)
        batch_report.append(
            {
                "batch_size": int(stop - start),
                "loss": float(loss),
                "raw_masked_bce": float(raw),
                "numpy_reference": ref,
                "abs_diff": abs(float(loss) - ref),
                "relative_abs_diff": _relative_abs_diff(float(loss), ref),
            }
        )
    cases.append(
        {
            "name": "different_batch_sizes",
            "n_task": n2.tolist(),
            "batches": batch_report,
            "all_finite": all(np.isfinite([b["loss"], b["raw_masked_bce"]]).all() for b in batch_report),
            "all_match_numpy": all(b["relative_abs_diff"] <= 1e-6 for b in batch_report),
        }
    )

    # ---------------------------------------------------------------- case 3
    # Equal observation coverage across tasks: the full-data task-macro
    # objective equals the full-data micro objective.
    N3, T3 = 8, 3
    y3_np = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )
    mask3 = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    )
    y3_np = np.where(mask3, y3_np, np.nan)
    logits3_np = np.linspace(-1.0, 1.0, num=N3 * T3).reshape(N3, T3)
    n3 = np.isfinite(y3_np).sum(axis=0).astype(np.int64)
    assert len(set(n3.tolist())) == 1, n3
    full_task = numpy_full_task_average(logits3_np, y3_np, n3)
    full_micro = numpy_micro_reference(logits3_np, y3_np)
    cases.append(
        {
            "name": "equal_coverage_full_data_equivalence",
            "n_task": n3.tolist(),
            "full_task_macro": full_task,
            "full_micro": full_micro,
            "abs_diff": abs(full_task - full_micro),
            "equal": bool(abs(full_task - full_micro) <= 1e-12),
        }
    )

    # ---------------------------------------------------------------- case 4
    # Batch-size-weighted aggregation of minibatch T losses must reconstruct
    # the full-data task average, for unequal batch sizes.
    N4, T4 = 10, 3
    rng = np.random.default_rng(20261105)
    y4_np = rng.integers(0, 2, size=(N4, T4)).astype(np.float64)
    mask4 = rng.random((N4, T4)) > 0.25
    # Guarantee every task has at least one observed entry.
    mask4[0, :] = True
    y4_np = np.where(mask4, y4_np, np.nan)
    logits4_np = rng.normal(0.0, 1.0, size=(N4, T4))
    n4 = np.isfinite(y4_np).sum(axis=0).astype(np.int64)
    full4 = numpy_full_task_average(logits4_np, y4_np, n4)
    bounds = [(0, 3), (3, 7), (7, 10)]
    weighted = 0.0
    aggregate_rows = []
    for start, stop in bounds:
        B = stop - start
        lb = torch.tensor(logits4_np[start:stop], dtype=torch.float32)
        yb = torch.tensor(y4_np[start:stop], dtype=torch.float32)
        loss, raw = task_macro_bce_loss(lb, yb, n4, n_train=N4)
        weighted += float(loss) * B / N4
        aggregate_rows.append(
            {"start": start, "stop": stop, "batch_size": B, "loss": float(loss), "raw": float(raw)}
        )
    cases.append(
        {
            "name": "batch_size_weighted_aggregation",
            "n_task": n4.tolist(),
            "batches": aggregate_rows,
            "weighted_minibatch_average": weighted,
            "full_task_average": full4,
            "abs_diff": abs(weighted - full4),
            "relative_abs_diff": _relative_abs_diff(weighted, full4),
            "equal": bool(_tensor_close(weighted, full4, 1e-6)),
            "rule": "sum_b (B_b / N) * L_T(batch_b) equals mean_t mean_observed BCE",
        }
    )

    # ---------------------------------------------------------------- case 5
    # Gradient propagation and a real change of task weighting (not logging).
    y5 = torch.tensor(
        [
            [1.0, 0.0, float("nan")],
            [0.0, 1.0, 1.0],
            [1.0, float("nan"), 0.0],
            [0.0, 1.0, float("nan")],
        ],
        dtype=torch.float32,
    )
    logits5_np = np.array(
        [[0.2, -0.5, 0.1], [-0.3, 0.7, -0.2], [0.6, 0.1, -0.8], [-0.1, -0.4, 0.5]],
        dtype=np.float32,
    )
    n5 = np.array([9, 13, 17], dtype=np.int64)
    logits5_m = torch.tensor(logits5_np, requires_grad=True)
    logits5_t = torch.tensor(logits5_np, requires_grad=True)
    loss5_m = masked_bce_loss(logits5_m, y5)
    loss5_t, raw5_t = task_macro_bce_loss(logits5_t, y5, n5)
    loss5_m.backward()
    loss5_t.backward()
    grad5_m = logits5_m.grad.detach()
    grad5_t = logits5_t.grad.detach()
    # T per-observed-entry coefficient is N/(B*T*n_t), not the micro 1/n_observed.
    n_observed = int(torch.isfinite(y5).sum())
    t_coeff = (C.N_TRAIN / (4 * 3)) / n5.astype(np.float64)
    m_coeff = 1.0 / n_observed
    cases.append(
        {
            "name": "gradient_and_real_task_weighting_change",
            "n_task": n5.tolist(),
            "loss_M": float(loss5_m),
            "loss_T": float(loss5_t),
            "raw_T": float(raw5_t),
            "grad_M_all_finite": bool(torch.isfinite(grad5_m).all()),
            "grad_T_all_finite": bool(torch.isfinite(grad5_t).all()),
            "grad_max_abs_diff_T_minus_M": float((grad5_t - grad5_m).abs().max()),
            "T_per_task_coefficient": t_coeff.tolist(),
            "M_per_entry_coefficient": float(m_coeff),
            "task_weights_differ": bool((grad5_t - grad5_m).abs().max() > 1e-8),
            "T_matches_numpy": _tensor_close(
                float(loss5_t),
                numpy_task_macro_reference(logits5_np, y5.numpy(), n5),
            ),
        }
    )

    # ---------------------------------------------------------------- case 6
    # All-observed batch and n_t == N: T reduces to the micro mean.
    y6 = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    logits6 = torch.tensor(
        [[0.2, -0.5, 0.1], [-0.3, 0.7, -0.2], [0.6, 0.1, -0.8], [-0.1, -0.4, 0.5]],
        dtype=torch.float32,
    )
    n6 = np.array([C.N_TRAIN, C.N_TRAIN, C.N_TRAIN], dtype=np.int64)
    loss6_t, _ = task_macro_bce_loss(logits6, y6, n6)
    loss6_m = masked_bce_loss(logits6, y6)
    cases.append(
        {
            "name": "fully_observed_n_equals_N_equivalence",
            "loss_T": float(loss6_t),
            "loss_M": float(loss6_m),
            "abs_diff": abs(float(loss6_t) - float(loss6_m)),
            "equal": bool(abs(float(loss6_t) - float(loss6_m)) <= 1e-6),
        }
    )

    checks = {
        "heterogeneous_batch_matches_numpy": cases[0]["relative_abs_diff"] <= 1e-6,
        "fully_unlabelled_task_zero_gradient": cases[0]["grad_fully_unlabelled_task_max_abs"] == 0.0,
        "heterogeneous_batch_grad_finite": cases[0]["grad_all_finite"],
        "different_batch_sizes_finite_and_match": cases[1]["all_finite"] and cases[1]["all_match_numpy"],
        "equal_coverage_equivalence": cases[2]["equal"],
        "batch_weighted_aggregation": cases[3]["equal"],
        "gradient_finite_both_losses": cases[4]["grad_M_all_finite"] and cases[4]["grad_T_all_finite"],
        "task_weighting_really_changes": cases[4]["task_weights_differ"],
        "T_matches_independent_numpy": cases[4]["T_matches_numpy"],
        "T_reduces_to_M_when_fully_observed": cases[5]["equal"],
    }
    return {
        "schema": "lcrc-pcba-loss-screen-cpu-validation-v1",
        "round_id": C.ROUND_ID,
        "formula": "L_T = N / (B*T) * sum_{i,t}(m_it * ell_it / n_t)",
        "constants": {
            "N": C.N_TRAIN,
            "T": C.N_TASKS,
            "loss_M": "pc_metrics.masked_bce_loss (historical, unchanged)",
            "loss_T": "task_macro_bce_loss (pre-registered)",
        },
        "cases": cases,
        "checks": {k: bool(v) for k, v in checks.items()},
        "all_passed": bool(all(checks.values())),
        "notes": [
            "NaN labels are replaced by 0 before elementwise BCE and then masked.",
            "n_t is supplied by the caller from the complete official Train split.",
            "No positive-class weight, focal term or clipping is applied.",
            "The T coefficient changes task weights only; it does not address "
            "within-task positive/negative imbalance.",
        ],
    }


__all__ = [
    "cpu_validation_suite",
    "masked_bce_loss",
    "numpy_full_task_average",
    "numpy_micro_reference",
    "numpy_task_macro_reference",
    "task_counts",
    "task_macro_bce_loss",
]
