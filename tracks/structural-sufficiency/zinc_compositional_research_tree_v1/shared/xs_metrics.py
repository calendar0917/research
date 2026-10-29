"""Frozen evaluation metrics and slices (task §6 result reporting).

Definitions are frozen before any GPU run:

* ``ordinary``        : ``y > -3``   (outside the artificial cycle-penalty tail)
* ``tail``            : ``y <= -3``
* ``any_unseen``      : at least one atom whose exact typed radius-2 environment
                        never occurs in Fit8k (same construction as the frozen
                        SSRC audits)
* ``unseen_ordinary`` : ``any_unseen & ordinary``
* ``seen_ordinary``   : ``~any_unseen & ordinary``
* ``minus_row172``    : drops the row at fixed position 172 of the frozen
                        Monitor row list (a fixed identity, never chosen by y)
* ``minus_fixed_worst10`` : drops the 10 Monitor rows with the largest absolute
                        error of the frozen DCC-L Fit8k seed-0 best-Monitor
                        checkpoint (fixed before any run of this round)
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# structural support (identical construction to the frozen SSRC audits)
# ---------------------------------------------------------------------------


def molecules(cache: Mapping[str, Any]) -> list[tuple[np.ndarray, list]]:
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    out = []
    for m in range(len(ap) - 1):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        out.append((at[a0:a1], [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e]))
                                for e in range(b0, b1)]))
    return out


def radius2_support(fit_graphs, query_graphs) -> dict:
    def signature_sets(graphs):
        per_atom = []
        for atoms, bonds in graphs:
            nbrs: list[list[tuple[int, int]]] = [[] for _ in range(len(atoms))]
            for u, v, t in bonds:
                nbrs[u].append((t, v))
                nbrs[v].append((t, u))
            one = [f"r1({int(atoms[i])};"
                   + ",".join(sorted(f"{t}-{int(atoms[j])}" for t, j in nbrs[i])) + ")"
                   for i in range(len(atoms))]
            two = [f"r2({one[i]};"
                   + ",".join(sorted(f"{t}-{one[j]}" for t, j in nbrs[i])) + ")"
                   for i in range(len(atoms))]
            per_atom.append(two)
        return per_atom

    freq: dict[str, int] = {}
    for sigs in signature_sets(fit_graphs):
        for s in sigs:
            freq[s] = freq.get(s, 0) + 1
    unseen, low4, supported, counts = [], [], [], []
    for sigs in signature_sets(query_graphs):
        if not sigs:
            unseen.append(0.0); low4.append(0.0); supported.append(0.0); counts.append([])
            continue
        c = [freq.get(s, 0) for s in sigs]
        unseen.append(float(np.mean([x == 0 for x in c])))
        low4.append(float(np.mean([1 <= x <= 4 for x in c])))
        supported.append(float(np.mean([x >= 5 for x in c])))
        counts.append(c)
    return {
        "unseen_frac": np.asarray(unseen),
        "low4_frac": np.asarray(low4),
        "supported_frac": np.asarray(supported),
        "all_support": counts,
        "fit_vocab_size": int(len(freq)),
    }


def support_tiers(fit_cache, fit_rows, query_cache, query_rows) -> dict:
    fit_graphs = [molecules(fit_cache)[int(r)] for r in np.asarray(fit_rows)]
    query_graphs = [molecules(query_cache)[int(r)] for r in np.asarray(query_rows)]
    sup = radius2_support(fit_graphs, query_graphs)
    unseen = sup["unseen_frac"] > 0.0
    if "labels" in query_cache and query_rows is not None:
        y = np.asarray(query_cache["labels"], dtype=np.float64)[np.asarray(query_rows)]
    else:
        y = np.zeros(len(query_graphs))
    ordinary = y > -3.0
    return {
        "unseen_frac": sup["unseen_frac"],
        "any_unseen": unseen,
        "ordinary": ordinary,
        "tail": ~ordinary,
        "unseen_ordinary": unseen & ordinary,
        "seen_ordinary": (~unseen) & ordinary,
        "fit_vocab_size": sup["fit_vocab_size"],
    }


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------


def mae(pred, y, mask=None) -> float | None:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if mask is not None:
        pred, y = pred[np.asarray(mask, bool)], y[np.asarray(mask, bool)]
    if pred.size == 0:
        return None
    return float(np.mean(np.abs(pred - y)))


def slice_table(pred, y, *, tiers, drop_rows: Sequence[int] = (), row_index: int | None = None) -> dict:
    keep = np.ones(len(y), dtype=bool)
    rows = np.asarray(drop_rows, dtype=np.int64)
    if rows.size:
        keep[rows] = False
    out = {}
    out["full"] = mae(pred, y, keep)
    out["ordinary"] = mae(pred, y, keep & tiers["ordinary"])
    out["tail"] = mae(pred, y, keep & tiers["tail"])
    out["any_unseen"] = mae(pred, y, keep & tiers["any_unseen"])
    out["unseen_ordinary"] = mae(pred, y, keep & tiers["unseen_ordinary"])
    out["seen_ordinary"] = mae(pred, y, keep & tiers["seen_ordinary"])
    if row_index is not None:
        m = keep.copy()
        m[row_index] = False
        out["minus_row172"] = mae(pred, y, m)
    return out


def curve_stats(curve: Sequence[Mapping[str, Any]]) -> dict:
    vals = np.asarray([float(e["monitor_mae"]) for e in curve])
    out = {
        "best": float(vals.min()),
        "best_epoch": int(np.argmin(vals) + 1),
        "final": float(vals[-1]),
    }
    if len(vals) >= 20:
        roll = [float(np.median(vals[i - 20:i])) for i in range(20, len(vals) + 1)]
        j = int(np.argmin(roll))
        out["rolling20_min_median"] = float(roll[j])
        out["rolling20_epoch"] = int(j + 20)
    else:
        out["rolling20_min_median"] = None
    if len(vals) >= 100:
        out["tail100_median"] = float(np.median(vals[-100:]))
    else:
        out["tail100_median"] = None
    return out


def paired_delta(pred_a, pred_b, y, mask=None, *, n_boot: int = 5000, seed: int = 0) -> dict:
    """``mean|a-y| - mean|b-y|`` with a paired bootstrap CI (negative = a better)."""
    a = np.asarray(pred_a, dtype=np.float64)
    b = np.asarray(pred_b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, bool)
        a, b, y = a[m], b[m], y[m]
    n = len(y)
    if n == 0:
        return {"n": 0, "delta": None, "ci95": None}
    d = np.abs(a - y) - np.abs(b - y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(1)
    return {
        "n": int(n),
        "delta": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "win_rate": float(np.mean(d < 0)),
    }
