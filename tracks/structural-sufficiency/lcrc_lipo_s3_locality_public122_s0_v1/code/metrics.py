"""Independent NumPy metrics and the paired component bootstrap.

All performance numbers are in original logD units (the standardised -> raw
conversion happens exactly once, here).

Evaluation components are *label-free*: the baseline round's ``eval_component``
(a structural edit/identity component over all 4200 molecules) is merged with
this round's "complete C1 input is identical" molecule groups.  They are used
only for resampling; they never enter training and never read a target.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import common as C


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if pred.size != y.size:
        raise ValueError("prediction/target size mismatch")
    err = pred - y
    sse = float((err * err).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "rmse": float(math.sqrt(sse / max(y.size, 1))),
        "mae": float(np.abs(err).mean()) if y.size else float("nan"),
        "sse": sse,
        "bias": float(err.mean()) if y.size else float("nan"),
        "r2": float(1.0 - sse / ss_tot) if ss_tot > 0 else float("nan"),
    }


def raw_metrics(pred_standardised: np.ndarray, y: np.ndarray, target_norm: Mapping[str, Any]) -> dict[str, float]:
    pred = np.asarray(pred_standardised, dtype=np.float64) * float(target_norm["scale"]) + float(target_norm["mean"])
    return regression_metrics(pred, y)


# ---------------------------------------------------------------------------
# label-free evaluation components
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {int(x): int(x) for x in items}

    def find(self, x: int) -> int:
        parent = self.parent
        root = int(x)
        while parent[root] != root:
            root = parent[root]
        while parent[int(x)] != root:
            parent[int(x)], x = root, parent[int(x)]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_eval_components(
    rows: Sequence[int],
    baseline_component: Mapping[int, int],
    same_input_groups: Sequence[Sequence[int]],
) -> dict[int, int]:
    """Merge the baseline structural component with identical-input groups."""
    uf = UnionFind(rows)
    by_component: dict[int, list[int]] = {}
    for r in rows:
        by_component.setdefault(int(baseline_component[int(r)]), []).append(int(r))
    for members in by_component.values():
        for a, b in zip(members, members[1:]):
            uf.union(a, b)
    for group in same_input_groups:
        members = [int(r) for r in group]
        for a, b in zip(members, members[1:]):
            uf.union(a, b)
    # deterministic component ids: the smallest row id in the component
    return {int(r): uf.find(int(r)) for r in rows}


def component_members(component: Mapping[int, int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r, c in component.items():
        out.setdefault(int(c), []).append(int(r))
    for c in out:
        out[c].sort()
    return out


# ---------------------------------------------------------------------------
# paired component bootstrap
# ---------------------------------------------------------------------------


def bootstrap_indices(n_components: int, replicates: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, int(n_components), size=(int(replicates), int(n_components)))


def _component_sums(
    component: Mapping[int, int],
    rows: Sequence[int],
    pred: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    comp = np.asarray([int(component[int(r)]) for r in rows], dtype=np.int64)
    uniq = np.asarray(sorted(set(comp.tolist())), dtype=np.int64)
    index = {int(c): i for i, c in enumerate(uniq.tolist())}
    pos = np.asarray([index[int(c)] for c in comp.tolist()], dtype=np.int64)
    sse = np.zeros(uniq.size, dtype=np.float64)
    sae = np.zeros(uniq.size, dtype=np.float64)
    cnt = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(sse, pos, err * err)
    np.add.at(sae, pos, np.abs(err))
    np.add.at(cnt, pos, 1.0)
    return {"components": uniq, "sse": sse, "sae": sae, "n": cnt}


def bootstrap_split(
    *,
    predictions: Mapping[str, np.ndarray],
    rows: Sequence[int],
    y: np.ndarray,
    component: Mapping[int, int],
    rng: np.random.Generator,
    replicates: int = C.BOOTSTRAP_REPLICATES,
    index: np.ndarray | None = None,
) -> dict[str, Any]:
    """One shared resampling of components; every model uses the same draws."""
    sums = {name: _component_sums(component, rows, pred, y) for name, pred in predictions.items()}
    comps = next(iter(sums.values()))["components"]
    if index is None:
        index = bootstrap_indices(comps.size, replicates, rng)
    out: dict[str, Any] = {"n_components": int(comps.size), "component_ids": comps.tolist()}
    for name, s in sums.items():
        sse = s["sse"][index].sum(axis=1)
        cnt = s["n"][index].sum(axis=1)
        rmse = np.sqrt(sse / np.maximum(cnt, 1.0))
        sae = s["sae"][index].sum(axis=1)
        mae = sae / np.maximum(cnt, 1.0)
        out[name] = {"rmse": rmse, "mae": mae}
    return out


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(vals, 100.0 * (1.0 - level) / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 + level) / 2.0))
    return lo, hi


def gain_summary(
    *,
    reference: str,
    candidate: str,
    metric: str,
    point_reference: float,
    point_candidate: float,
    boots_reference: np.ndarray,
    boots_candidate: np.ndarray,
) -> dict[str, Any]:
    gain = float(point_reference) - float(point_candidate)
    boots = np.asarray(boots_reference, dtype=np.float64) - np.asarray(boots_candidate, dtype=np.float64)
    lo, hi = percentile_ci(boots)
    return {
        "reference": reference,
        "candidate": candidate,
        "metric": metric,
        "reference_value": float(point_reference),
        "candidate_value": float(point_candidate),
        "gain": gain,
        "gain_ci95_lower": lo,
        "gain_ci95_upper": hi,
        "ci_lower_gt_zero": bool(np.isfinite(lo) and lo > 0.0),
        "ci_upper_lt_zero": bool(np.isfinite(hi) and hi < 0.0),
        "ci_crosses_zero": bool((not np.isfinite(lo)) or (not np.isfinite(hi)) or lo <= 0.0 <= hi),
        "n_bootstrap": int(boots.size),
        "gain_sign": "positive" if gain > 0 else ("negative" if gain < 0 else "zero"),
    }


__all__ = [
    "regression_metrics",
    "raw_metrics",
    "UnionFind",
    "build_eval_components",
    "component_members",
    "bootstrap_indices",
    "bootstrap_split",
    "percentile_ci",
    "gain_summary",
]
