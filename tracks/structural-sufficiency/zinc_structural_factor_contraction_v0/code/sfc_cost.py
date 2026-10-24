"""T2 — exact elimination cost on the 10,000 official ZINC-subset **train** structures.

Data policy
-----------
Reads ``data/ZINC/raw/train.pickle`` + ``data/ZINC/raw/train.index`` through a
**label-blocking mapping** that raises ``PermissionError`` on the target key.
Only ``num_atom``, ``atom_type`` and ``bond_type`` are read.  The PyG
``train.pt`` tensor cache is deliberately *not* used because it materialises
``y``.  Official valid/test are never opened.  Labels are never read.

What is measured (per graph, not averaged away)
-----------------------------------------------
* ``n``, ``m``, cyclomatic number;
* induced width of the min-fill and min-degree elimination orders, and the
  best (smallest) of the two;
* max intermediate factor size (bucket elements and message elements);
* arithmetic op count ``sum_steps 2^|union_scope|`` per channel;
* measured peak bytes held live during the elimination (C=1);
* wall time of one forward and of forward+backward at ``C = 1, 8, 16``.

Everything is reported as a distribution (median / p90 / p95 / p99 / max) **and**
as total + worst graph.  A mean is never used to hide the hardest molecule.

Also reported
-------------
* projected cost of one full train epoch (10,000 graphs, forward+backward) at
  ``K = 8`` and ``K = 16``;
* the count of graphs whose *single-graph* cost exceeds a stated budget;
* a **high-treewidth stress test** on synthetic dense graphs, so the ZINC numbers
  are not mistaken for a general statement about the operator's cost.

Run:
  uv run python tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/code/sfc_cost.py
"""

from __future__ import annotations

import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sfc_core as SC  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
REPO = EXP.parents[2]
ZINC_RAW = REPO / "data" / "ZINC" / "raw"
LABEL_KEY = "logP_SA_cycle_normalized"

N_ATOM_CATEGORIES = 28
N_BOND_SLOTS = 4
CHANNELS = (1, 8, 16)
BUDGET_PER_GRAPH_MS = 50.0        # "acceptable" single-graph budget, declared up front


# ---------------------------------------------------------------------------
# label-blocked reader
# ---------------------------------------------------------------------------


class LabelBlocked(dict):
    """Dict view that refuses to return the target column and records every read."""

    def __init__(self, payload):
        super().__init__(payload)
        self.reads: list[str] = []

    def __getitem__(self, key):
        if key == LABEL_KEY:
            raise PermissionError(f"refusing to read label column {key!r}")
        self.reads.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == LABEL_KEY:
            raise PermissionError(f"refusing to read label column {key!r}")
        return super().get(key, default)


def load_train_structures(limit: int | None = None) -> tuple[list[SC.TypedGraph], dict]:
    idx = [int(x) for x in (ZINC_RAW / "train.index").read_text().strip().split(",")]
    if limit is not None:
        idx = idx[:limit]
    t0 = time.perf_counter()
    with open(ZINC_RAW / "train.pickle", "rb") as fh:
        mols = pickle.load(fh)
    load_s = time.perf_counter() - t0

    reads: set[str] = set()
    graphs: list[SC.TypedGraph] = []
    for i in idx:
        mol = LabelBlocked(mols[i])
        n = int(mol["num_atom"])
        atom = np.asarray(mol["atom_type"], dtype=np.int64)
        bt = np.asarray(mol["bond_type"], dtype=np.int64)
        reads.update(mol.reads)
        if atom.shape != (n,) or bt.shape != (n, n):
            raise RuntimeError(f"row {i}: inconsistent raw shapes")
        u, v = np.nonzero(np.triu(bt, 1))
        graphs.append(SC.TypedGraph(
            n=n, atom=atom,
            edge_u=u.astype(np.int64), edge_v=v.astype(np.int64),
            edge_b=bt[u, v].astype(np.int64)))
    if LABEL_KEY in reads:
        raise RuntimeError("label column was read")
    meta = {
        "source": "data/ZINC/raw/train.pickle + train.index",
        "n_requested": len(idx),
        "n_loaded": len(graphs),
        "keys_read": sorted(reads),
        "label_column_read": False,
        "pickle_load_seconds": load_s,
        "valid_or_test_read": False,
        "pyg_processed_cache_used": False,
    }
    return graphs, meta


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def dist(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "p999": float(np.percentile(a, 99.9)),
        "max": float(a.max()),
        "total": float(a.sum()),
    }


def _params(C: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(N_ATOM_CATEGORIES, C, 2, generator=g, dtype=torch.float64)
    S = torch.randn(N_BOND_SLOTS, C, 3, generator=g, dtype=torch.float64)
    return A, S


# ---------------------------------------------------------------------------
# structural cost (order-independent of wall clock)
# ---------------------------------------------------------------------------


def structural_costs(graphs: list[SC.TypedGraph]) -> dict:
    rows = []
    t0 = time.perf_counter()
    for g in graphs:
        edges = g.edges()
        p_fill = SC.build_plan(g.n, edges, SC.order_min_fill(g.n, edges))
        p_deg = SC.build_plan(g.n, edges, SC.order_min_degree(g.n, edges))
        best = p_fill if p_fill.max_bucket_elems <= p_deg.max_bucket_elems else p_deg
        rows.append({
            "n": g.n, "m": g.m, "cyclomatic": g.cyclomatic(),
            "width_min_fill": p_fill.induced_width,
            "width_min_degree": p_deg.induced_width,
            "width_best": best.induced_width,
            "max_bucket_elems_best": best.max_bucket_elems,
            "max_message_elems_best": best.max_message_elems,
            "arith_ops_best": best.arith_ops,
            "n_steps_best": len(best.steps),
        })
    plan_s = time.perf_counter() - t0
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    widths = np.asarray(cols["width_best"])
    return {
        "per_graph_plan_seconds": plan_s / max(len(graphs), 1),
        "plan_all_seconds": plan_s,
        "n": dist(cols["n"]),
        "m": dist(cols["m"]),
        "cyclomatic": dist(cols["cyclomatic"]),
        "width_min_fill": dist(cols["width_min_fill"]),
        "width_min_degree": dist(cols["width_min_degree"]),
        "width_best": dist(cols["width_best"]),
        "width_best_histogram": {str(int(k)): int(v) for k, v in
                                 zip(*np.unique(widths, return_counts=True))},
        "max_bucket_elems_best": dist(cols["max_bucket_elems_best"]),
        "max_message_elems_best": dist(cols["max_message_elems_best"]),
        "arith_ops_best_per_channel": dist(cols["arith_ops_best"]),
        "hardest_graphs": sorted(rows, key=lambda r: (-r["arith_ops_best"], -r["n"]))[:10],
    }


# ---------------------------------------------------------------------------
# measured wall time + peak memory
# ---------------------------------------------------------------------------


def measure_time_and_memory(graphs: list[SC.TypedGraph], C: int,
                            do_backward: bool, seed: int = 0) -> dict:
    A, S = _params(C, seed=seed)
    A.requires_grad_(do_backward)
    S.requires_grad_(do_backward)
    fwd_ms: list[float] = []
    peak_bytes: list[float] = []
    plans = [SC.build_plan(g.n, g.edges(), SC.order_min_fill(g.n, g.edges()))
             for g in graphs]
    for g, plan in zip(graphs, plans):
        t0 = time.perf_counter()
        out = SC.sfc_forward(g, A, S, plan=plan, with_marginals=do_backward,
                             track_memory=True)
        if do_backward:
            (out.H.sum() + out.T.sum()).backward()
            A.grad = None
            S.grad = None
        fwd_ms.append((time.perf_counter() - t0) * 1e3)
        peak_bytes.append(out.stats["peak_bytes_measured"])
    return {
        "C": C,
        "backward": do_backward,
        "ms": dist(fwd_ms),
        "peak_bytes": dist(peak_bytes),
        "slowest_graphs": sorted(
            [{"n": g.n, "m": g.m, "cyclomatic": g.cyclomatic(),
              "ms": float(ms), "peak_bytes": float(pb)}
             for g, ms, pb in zip(graphs, fwd_ms, peak_bytes)],
            key=lambda r: -r["ms"])[:10],
        "n_over_budget": int(sum(1 for v in fwd_ms if v > BUDGET_PER_GRAPH_MS)),
        "budget_ms": BUDGET_PER_GRAPH_MS,
    }


# ---------------------------------------------------------------------------
# high-treewidth stress test (regime boundary, not a ZINC claim)
# ---------------------------------------------------------------------------


def stress_high_treewidth() -> dict:
    rng = np.random.default_rng(0)
    rows = []
    for n in (10, 14, 18, 22, 26, 30, 36):
        for density in (0.10, 0.20, 0.35, 0.50):
            eu, ev = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        eu.append(i)
                        ev.append(j)
            g = SC.TypedGraph(n, np.zeros(n, np.int64),
                              np.asarray(eu, np.int64), np.asarray(ev, np.int64),
                              np.ones(len(eu), np.int64))
            p_fill = SC.build_plan(g.n, g.edges(), SC.order_min_fill(g.n, g.edges()))
            p_deg = SC.build_plan(g.n, g.edges(), SC.order_min_degree(g.n, g.edges()))
            best = min(p_fill, p_deg, key=lambda p: p.max_bucket_elems)
            rows.append({
                "n": n, "density": density, "m": g.m,
                "width_min_fill": p_fill.induced_width,
                "width_min_degree": p_deg.induced_width,
                "width_best": best.induced_width,
                "max_bucket_elems_best": best.max_bucket_elems,
                "C16_max_bucket_floats": 16 * best.max_bucket_elems,
                "C16_max_bucket_MB_float64": 16 * best.max_bucket_elems * 8 / 2**20,
            })
    return {
        "note": ("synthetic dense graphs, NOT ZINC; bounds the regime where the "
                 "ZINC result stops applying"),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    torch.set_num_threads(1)
    t_all = time.perf_counter()
    graphs, meta = load_train_structures()
    struct = structural_costs(graphs)

    timings = {}
    for C in CHANNELS:
        timings[f"C{C}_forward"] = measure_time_and_memory(graphs, C, do_backward=False)
        timings[f"C{C}_forward_backward"] = measure_time_and_memory(
            graphs, C, do_backward=True)

    n = len(graphs)
    projections = {}
    for C in CHANNELS:
        fwd = timings[f"C{C}_forward"]["ms"]
        fb = timings[f"C{C}_forward_backward"]["ms"]
        projections[f"K{C}"] = {
            "one_epoch_forward_seconds_total": fwd["total"] / 1e3,
            "one_epoch_forward_backward_seconds_total": fb["total"] / 1e3,
            "one_epoch_fb_p95_bound_seconds": n * fb["p95"] / 1e3,
            "one_epoch_fb_max_bound_seconds": n * fb["max"] / 1e3,
            "note": ("p95 / max bounds assume every graph in the epoch costs the "
                     "p95 / max observed graph; total is the measured sum"),
        }

    payload = {
        "protocol": "sfc-v0-cpu-audit",
        "stage": "T2-real-graph-cost",
        "data": meta,
        "device": "cpu",
        "torch_threads": int(torch.get_num_threads()),
        "dtype": "float64",
        "structural": struct,
        "timings": timings,
        "epoch_projection": projections,
        "stress_high_treewidth": stress_high_treewidth(),
        "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "seconds_total": time.perf_counter() - t_all,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS / "t2_cost.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    w = struct["width_best"]
    print(f"graphs={n} width_best median={w['median']:.0f} p99={w['p99']:.0f} max={w['max']:.0f}")
    for C in CHANNELS:
        fb = timings[f"C{C}_forward_backward"]["ms"]
        print(f"C={C:2d} fwd+bwd median={fb['median']:.2f} ms  p99={fb['p99']:.2f}  "
              f"max={fb['max']:.2f}  total={fb['total']/1e3:.1f} s")
    print(f"-> {out}  ({payload['seconds_total']:.1f}s, rss {payload['rss_peak_mb']:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
