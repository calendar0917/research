"""P3 -- the two task-objective candidates, their toy-batch probes and gradients.

``masked_cell`` (frozen, Sprint 02)
    ``sum_{observed} BCE / n_observed`` -- every observed (molecule, endpoint)
    cell has the same weight, so an endpoint with many assayed molecules
    dominates a rare endpoint.

``equal_task`` (new, Tox21 only)
    per-endpoint mean BCE over that endpoint's observed cells, then an
    unweighted mean over the endpoints that have at least one observed cell
    *in this batch*.  Empty endpoints are excluded from the denominator and are
    never filled with a fake zero loss.

``sqrt_pos_weight`` (new, ogbg-molhiv only)
    ``BCEWithLogitsLoss(pos_weight=min(5, sqrt(n_neg/n_pos)))`` with the counts
    taken from the official **Train** split only.

Everything in this module is CPU-only and label-shape agnostic; the probes are
run on hand-built toy batches so the behaviour on different missing-label
patterns is observable, not assumed.

Usage
-----
    uv run python code/ta_losses.py            # run the toy probes, print the table
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import ta_common as C


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------


def masked_cell_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Frozen Tox21 objective: mean BCE over observed cells."""
    mask = torch.isfinite(y)
    if int(mask.sum()) == 0:
        raise ValueError("no observed label in the batch")
    per_cell = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
    return (per_cell * mask).sum() / mask.sum().clamp(min=1.0)


def equal_task_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-task mean BCE, then an unweighted mean over tasks observed in the batch."""
    if logits.ndim != 2:
        raise ValueError("equal_task_loss expects [n_rows, n_tasks] logits")
    mask = torch.isfinite(y)
    per_cell = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
    per_cell = per_cell * mask
    counts = mask.sum(dim=0)  # [n_tasks]
    present = counts > 0
    if int(present.sum()) == 0:
        raise ValueError("no observed label in the batch")
    per_task = per_cell.sum(dim=0) / counts.clamp(min=1.0)
    return per_task[present].mean()


def equal_task_weights(y: torch.Tensor) -> torch.Tensor:
    """Per-cell weight of ``equal_task_loss`` (for the analytic gradient check)."""
    mask = torch.isfinite(y)
    counts = mask.sum(dim=0)
    present = counts > 0
    n_present = present.sum()
    if int(n_present) == 0:
        raise ValueError("no observed label in the batch")
    weights = torch.zeros_like(mask, dtype=torch.float32)
    for t in range(y.shape[1]):
        if counts[t] > 0:
            weights[:, t] = mask[:, t].float() / counts[t].float() / n_present.float()
    return weights


def masked_cell_weights(y: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(y)
    total = mask.sum()
    if int(total) == 0:
        raise ValueError("no observed label in the batch")
    return mask.float() / total.float()


def sqrt_pos_weight(y_train: np.ndarray, cap: float = 5.0) -> dict[str, Any]:
    """``min(cap, sqrt(n_neg/n_pos))`` from Train labels only."""
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y)
    yy = y[mask]
    if not np.all(np.isin(yy, (0.0, 1.0))):
        raise ValueError("pos_weight requires binary labels")
    n_pos = float(yy.sum())
    n_neg = float(yy.size - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError("both classes are required to compute pos_weight")
    raw = math.sqrt(n_neg / n_pos)
    value = float(min(cap, raw))
    return {
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "positive_rate": float(n_pos / yy.size),
        "raw_sqrt_ratio": float(raw),
        "capped": bool(raw > cap),
        "cap": float(cap),
        "pos_weight": value,
        "formula": "min(cap, sqrt(n_negative / n_positive))",
        "source": "official Train split labels only",
    }


# ---------------------------------------------------------------------------
# toy-batch probes
# ---------------------------------------------------------------------------


def _toy_cases() -> dict[str, dict[str, Any]]:
    """Hand-built batches with different missing-label patterns.

    The toy values are chosen so the two objectives are analytically different:
    task 0 has 3 observed cells (2 positive), task 1 has 1 observed cell, and
    task 2 is entirely missing in some cases.
    """
    logits = torch.tensor(
        [
            [0.4, -1.2, 0.0],
            [1.1, 0.3, -0.5],
            [-0.7, 0.2, 1.3],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [
            [1.0, 0.0, float("nan")],
            [0.0, float("nan"), float("nan")],
            [1.0, float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    y_dense = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y_empty_task = torch.tensor(
        [
            [1.0, float("nan"), float("nan")],
            [0.0, 1.0, float("nan")],
            [1.0, 0.0, float("nan")],
        ],
        dtype=torch.float32,
    )
    y_single_task = torch.tensor(
        [
            [1.0, float("nan"), float("nan")],
            [0.0, float("nan"), float("nan")],
            [1.0, float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    return {
        "sparse_two_tasks": {"logits": logits, "y": y},
        "dense_three_tasks": {"logits": logits, "y": y_dense},
        "empty_third_task": {"logits": logits, "y": y_empty_task},
        "only_first_task": {"logits": logits, "y": y_single_task},
    }


def _analytic_grad(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """dL/dz = (sigmoid(z) - y_with_nan_to_0) * weight, for BCEWithLogits."""
    return (torch.sigmoid(logits) - torch.nan_to_num(y)) * weights


def toy_probe() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, case in _toy_cases().items():
        logits = case["logits"].clone().requires_grad_(True)
        y = case["y"]
        mask = torch.isfinite(y)
        n_cells = int(mask.sum())
        n_tasks = int((mask.sum(dim=0) > 0).sum())
        l_cell = masked_cell_loss(logits, y)
        l_task = equal_task_loss(logits, y)
        g_cell, = torch.autograd.grad(l_cell, logits, retain_graph=True)
        g_task, = torch.autograd.grad(l_task, logits, retain_graph=True)
        analytic_cell = _analytic_grad(logits.detach(), y, masked_cell_weights(y))
        analytic_task = _analytic_grad(logits.detach(), y, equal_task_weights(y))
        rows.append(
            {
                "case": name,
                "n_observed_cells": n_cells,
                "n_tasks_with_an_observed_cell": n_tasks,
                "masked_cell_loss": float(l_cell),
                "equal_task_loss": float(l_task),
                "difference": float(l_task - l_cell),
                "losses_differ": bool(abs(float(l_task - l_cell)) > 1e-7),
                "tasks_missing_pattern": mask.sum(dim=0).tolist(),
                "grad_masked_cell_matches_analytic": bool(
                    torch.allclose(g_cell, analytic_cell, atol=1e-6)
                ),
                "grad_equal_task_matches_analytic": bool(
                    torch.allclose(g_task, analytic_task, atol=1e-6)
                ),
                "equal_task_cell_weighting": [
                    [round(float(v), 6) for v in row] for row in equal_task_weights(y).tolist()
                ],
                "masked_cell_cell_weighting": [
                    [round(float(v), 6) for v in row] for row in masked_cell_weights(y).tolist()
                ],
            }
        )
    return rows


def run(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import mb2_source as S

    src = S.load_dataset("ogbg-molhiv", labels="train_valid")
    hiv = sqrt_pos_weight(np.asarray(src["y"]["train"], dtype=np.float64).reshape(-1), cap=5.0)
    payload = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "tox21_equal_task_loss": dict(C.TOX21_EQUAL_TASK),
        "hiv_sqrt_pos_weight": {**dict(C.HIV_SQRT_POS_WEIGHT), **hiv},
        "toy_probe": toy_probe(),
        "gates": {
            "two_losses_differ_on_some_toy_batch": any(r["losses_differ"] for r in toy_probe()),
            "gradients_match_analytic": all(
                r["grad_masked_cell_matches_analytic"] and r["grad_equal_task_matches_analytic"]
                for r in toy_probe()
            ),
            "empty_task_excluded": next(
                r for r in toy_probe() if r["case"] == "empty_third_task"
            )["n_tasks_with_an_observed_cell"]
            == 2,
            "single_task_normalisation": abs(
                next(r for r in toy_probe() if r["case"] == "only_first_task")["equal_task_loss"]
                - next(r for r in toy_probe() if r["case"] == "only_first_task")["masked_cell_loss"]
            )
            < 1e-6,
            "hiv_pos_weight_within_cap": hiv["pos_weight"] <= hiv["cap"],
        },
        "environment": C.env_info(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3 loss probes (CPU, toy batches)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    payload = run()
    out = C.ANALYSIS / "taskaware_loss_probes_p0.json"
    if args.write:
        C.write_json(out, payload)
        C.log(f"wrote {C.track_rel(out)}")
    print("| case | cells | tasks | masked_cell | equal_task | differ | grad ok |")
    print("|---|---:|---:|---:|---:|---|---|")
    for r in payload["toy_probe"]:
        print(
            f"| {r['case']} | {r['n_observed_cells']} | {r['n_tasks_with_an_observed_cell']} | "
            f"{r['masked_cell_loss']:.6f} | {r['equal_task_loss']:.6f} | {r['losses_differ']} | "
            f"{r['grad_masked_cell_matches_analytic'] and r['grad_equal_task_matches_analytic']} |"
        )
    h = payload["hiv_sqrt_pos_weight"]
    print(
        f"\nHIV pos_weight = min({h['cap']}, sqrt({h['n_negative']}/{h['n_positive']})) = "
        f"{h['pos_weight']:.6f} (raw {h['raw_sqrt_ratio']:.6f}, capped={h['capped']})"
    )
    print("\ngates:", payload["gates"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
