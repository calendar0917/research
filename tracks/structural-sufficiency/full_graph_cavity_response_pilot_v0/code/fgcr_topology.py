"""FGCR-v0 full-corpus topology preflight.

Builds the exact min-fill junction plan for every official row (4200 graphs),
records induced-width statistics and the exact state-table cost, and refuses to
continue if any induced width exceeds 8.  No labels are used for any decision.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import fgcr_common as C
import fgcr_data as D
import fgcr_inference as I


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "topology.json"))
    parser.add_argument("--widths-out", default=str(C.ANALYSIS_DIR / "widths.npy"))
    parser.add_argument("--smax", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=8)
    args = parser.parse_args()

    t0 = time.perf_counter()
    bundle = D.load_bundle()
    graphs = D.graphs_for_rows(bundle, range(bundle.n_graphs))
    graph_pairs = [(int(g["n_atoms"]), list(zip(g["bond_u"].tolist(), g["bond_v"].tolist())))
                   for g in graphs]

    widths = np.asarray([I.min_fill_width(n, edges) for n, edges in graph_pairs],
                        dtype=np.int64)
    width_seconds = time.perf_counter() - t0
    stats = {
        "n_graphs": int(widths.size),
        "width_median": float(np.median(widths)),
        "width_p95": float(np.percentile(widths, 95)),
        "width_p99": float(np.percentile(widths, 99)),
        "width_max": int(widths.max()),
        "width_counts": {int(k): int(v) for k, v in
                         zip(*np.unique(widths, return_counts=True))},
        "max_clique_size": int(widths.max()) + 1,
        "max_clique_state_table": int(2 ** (int(widths.max()) + 1)),
        "n_width_gt_8": int((widths > int(args.max_width)).sum()),
        "width_seconds": float(width_seconds),
        "cache_sha256": bundle.cache_sha256,
        "data_audit": bundle.audit,
    }
    Path(args.widths_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.widths_out, widths)

    if stats["n_width_gt_8"] > 0:
        stats["status"] = "BLOCKED"
        stats["blocked_reason"] = (
            f"{stats['n_width_gt_8']} graph(s) have induced width > {args.max_width}; "
            "exact inference would require a clique state table larger than 2^9"
        )
        C.write_json(args.out, stats)
        print(f"BLOCKED: {stats['blocked_reason']}")
        return 2

    # Build the actual plan used by training, with the frozen smax.
    t1 = time.perf_counter()
    plan = I.build_corpus_plan(graph_pairs, smax=int(args.smax))
    plan_seconds = time.perf_counter() - t1
    plan_stats = I.plan_stats(plan)
    stats["plan"] = plan_stats
    stats["plan_build_seconds"] = float(plan_seconds)
    stats["total_seconds"] = float(time.perf_counter() - t0)

    if plan_stats["max_width"] != stats["width_max"]:
        raise RuntimeError(
            f"plan max width {plan_stats['max_width']} != independent min-fill "
            f"width {stats['width_max']}")
    if plan_stats["n_cliques"] != plan_stats["n_atoms"]:
        raise RuntimeError("expected one elimination clique per atom")
    if plan_stats["max_clique_size"] != stats["max_clique_size"]:
        raise RuntimeError("plan max clique size differs from width + 1")

    # Exact state-table work and per-split totals (topology only).
    scope_sizes = np.asarray(plan.clique_scope_size.detach().cpu().numpy(), dtype=np.int64)
    state_entries_per_clique = np.power(2, scope_sizes, dtype=np.int64)
    stats["exact_cost"] = {
        "clique_state_entries_total": int(state_entries_per_clique.sum()),
        "clique_state_entries_max_per_clique": int(state_entries_per_clique.max()),
        "smax_state_table": int(2 ** int(plan.smax)),
        "channels": int(C.CHANNELS),
        "float32_bytes_per_clique_channel_table_smax": int(4 * 2 ** int(plan.smax)),
        "total_atoms": int(plan_stats["n_atoms"]),
        "total_physical_bonds": int(plan_stats["n_physical_bonds"]),
        "total_cliques": int(plan_stats["n_cliques"]),
    }
    stats["splits"] = {}
    for split in ("train", "valid", "test"):
        rows = bundle.split_rows(split)
        widths_split = widths[rows]
        stats["splits"][split] = {
            "n_graphs": int(rows.size),
            "width_median": float(np.median(widths_split)),
            "width_p95": float(np.percentile(widths_split, 95)),
            "width_p99": float(np.percentile(widths_split, 99)),
            "width_max": int(widths_split.max()),
            "n_atoms": int(bundle.graph_n_atoms[rows].sum()),
            "n_physical_bonds": int(bundle.graph_n_bonds[rows].sum()),
        }

    stats["status"] = "PASS"
    stats["all_graphs_covered"] = bool(widths.size == 4200 and plan.n_graphs == 4200)
    stats["note"] = (
        "Every official row has an exact junction plan. No graph is dropped, "
        "truncated, sampled or approximated."
    )
    C.write_json(args.out, stats)
    print(f"width median={stats['width_median']} p95={stats['width_p95']} "
          f"p99={stats['width_p99']} max={stats['width_max']} "
          f"state_table={stats['max_clique_state_table']}")
    print(f"plan cliques={plan_stats['n_cliques']} build={plan_seconds:.2f}s "
          f"status={stats['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())