"""Cost preflight for the corrected SFC-v0 replay (CPU only, no experiment).

Measures the per-graph cost of the *corrected* feature path
(``H`` and ``T`` from the exact marginals, with the second-order gradient needed
by ``T``) and projects the full replayed protocol:

* 5 arms x 2 seeds x 2 tasks, each 400 steps x batch 64;
* train/unseen evaluation with the same features (3000 + 1500 graphs per arm);
* dataset assembly (6000 graphs/task) and the auxiliary fit diagnostic
  (200 graphs x 60 steps/task).

No ZINC data is touched.  The preflight decides whether the replay fits a
reasonable CPU budget; it does not change any hyper-parameter.

Run:
  uv run python tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/code/sfc_cost_preflight.py
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import sfc_synth as SY

EXP = Path(__file__).resolve().parent.parent
ANALYSIS = EXP / "analysis"
BENCH_ITERS = 10


def _bench(graphs, A, S, need_grad: bool, iters: int = BENCH_ITERS) -> dict:
    for _ in range(2):                      # warm-up
        SY._sfc_batch(graphs, A, S, need_grad)
    t0 = time.perf_counter()
    for _ in range(iters):
        SY._sfc_batch(graphs, A, S, need_grad)
    sec = (time.perf_counter() - t0) / iters
    return {"batch_graphs": len(graphs), "seconds_per_batch": sec,
            "ms_per_graph": 1e3 * sec / max(len(graphs), 1)}


def main() -> int:
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    rows = {}
    total_forward = 0.0
    total_train = 0.0
    for task in ("reach", "junction"):
        t_as = time.perf_counter()
        split = SY.assemble(task, n_seen=96, n_unseen=48, seed=0)
        as_sec = time.perf_counter() - t_as
        graphs = split["graphs_train"]
        plan = split["sfc_plan"]
        A = torch.as_tensor(plan["A"], dtype=torch.float64)
        S = torch.as_tensor(plan["S"], dtype=torch.float64)
        train = _bench(graphs, A.clone().requires_grad_(True),
                       S.clone().requires_grad_(True), True)
        ev = _bench(graphs, A, S, False)
        n_graphs_per_assemble = 96 + 48
        rows[task] = {
            "assembly_seconds_for_144_graphs": as_sec,
            "assembly_ms_per_graph": 1e3 * as_sec / n_graphs_per_assemble,
            "train_feature": train,
            "eval_feature": ev,
        }
        # projections use the replayed counts, not the preflight split size
        total_train += 400 * 64 * train["ms_per_graph"] / 1e3 * 8      # S/SP x 2 tasks x 2 seeds
        total_forward += (3000 + 1500) * ev["ms_per_graph"] / 1e3 * 8  # eval, same arms
        total_forward += 4500 * rows[task]["assembly_ms_per_graph"] / 1e3 * 2  # assembly (2 tasks)
        total_forward += 60 * 200 * train["ms_per_graph"] / 1e3 * 2    # fit diagnostic
    projection = {
        "training_feature_seconds_S_and_SP": total_train,
        "other_feature_seconds_eval_assembly_diagnostic": total_forward,
        "projected_feature_seconds_total": total_train + total_forward,
        "assumed_counts": {
            "train_graph_forwards_with_backward": 400 * 64 * 8,
            "eval_graph_forwards": (3000 + 1500) * 8,
            "assembly_graph_forwards": 4500 * 2,
            "fit_diagnostic_graph_forwards_with_backward": 60 * 200 * 2,
        },
        "note": ("readout / collate / optimiser overhead is excluded; the measured "
                 "full-run wall clock is recorded in t3_synth_corrected.json"),
    }
    payload = {
        "protocol": "sfc-v0-cpu-audit",
        "stage": "T0-cost-preflight-corrected",
        "device": "cpu",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
        "torch_threads": int(torch.get_num_threads()),
        "dtype": "float64",
        "per_task": rows,
        "projection": projection,
        "seconds": time.perf_counter() - t0,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS / "t0_cost_preflight.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(projection, indent=2))
    print(f"-> {out} ({payload['seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
