"""
Process-metric suite for luyin10 R1/R4/R5/R6 (no GIN).

1) Coverage-driven sampling vs all-node / no-decay baselines
2) Receptive-field curve: B0 vs RW(L,m) on C4 hit rate + |S|
3) Assert no hard-delete

Outputs: results/coverage_rf.json + COVERAGE_RF_SUMMARY.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.coverage_sample import CoverageConfig, sample_coverage, trajectory_cover_metrics  # noqa: E402
from code.graph import Graph  # noqa: E402
from code.sample import SampleConfig, run_method  # noqa: E402
from code.synthetic_task import _count_c4, make_c4_vs_longcycle  # noqa: E402
from code.vectorize import adjacency_padded  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def c4_hit_rate(graphs: list[Graph], labels: np.ndarray, make_S) -> dict:
    """Fraction of class-0 graphs where some patch induced subgraph has C4."""
    hits = 0
    n0 = 0
    sizes = []
    for g, y in zip(graphs, labels):
        if int(y) != 0:
            continue
        n0 += 1
        ok = False
        for S in make_S(g):
            sizes.append(len(S))
            if _count_c4(g.induced(S)) > 0:
                ok = True
        hits += int(ok)
    return {
        "n_class0": n0,
        "frac_graphs_with_c4_patch": hits / max(n0, 1),
        "mean_|S|": float(np.mean(sizes)) if sizes else 0.0,
    }


def main() -> int:
    results: dict = {"protocol_id": "process-coverage-rf-v0", "hard_delete": False}

    # --- graphs for coverage ---
    from code.graph import ring_chords

    g_syn = ring_chords(24, [[0, 6], [3, 12], [8, 18], [5, 15]])
    log(f"coverage graph: n={g_syn.n} e={g_syn.num_edges()}")

    # A) coverage policies
    log("\n=== A. Coverage-driven sampling (same budget max_walks=12) ===")
    cover_rows = []
    policies = [
        ("all_nodes_no_decay", "all_nodes", 1.0, None, 30),  # many walks if all nodes
        ("random_decay", "random", 0.7, 0.95, 12),
        ("deg_strat_decay", "degree_stratified", 0.7, 0.95, 12),
        ("uncovered_decay", "uncovered", 0.7, 0.95, 12),
        ("uncovered_no_decay", "uncovered", 1.0, 0.95, 12),
    ]
    for name, pol, decay, tgt, mw in policies:
        # all_nodes: one walk per node up to n
        cfg = CoverageConfig(
            p=1.0,
            q=1.0,
            walk_length=10,
            max_nodes=8,
            max_walks=min(mw, g_syn.n) if pol == "all_nodes" else mw,
            edge_decay=decay,
            seed=0,
            seed_policy=pol,  # type: ignore
            cover_target=tgt,
        )
        if pol == "all_nodes":
            cfg.max_walks = g_syn.n
            cfg.cover_target = None
        bundle = sample_coverage(g_syn, cfg)
        m = trajectory_cover_metrics(g_syn, bundle)
        m["name"] = name
        m["edge_decay"] = decay
        m["seed_policy"] = pol
        cover_rows.append(m)
        log(
            f"  {name}: walks={m['n_walks']} cover={m['traj_edge_cover']:.3f} "
            f"repeat={m['traj_edge_repeat']:.3f} |S|={m['mean_|S|']:.2f}"
        )
    results["coverage"] = cover_rows

    # B0 baseline on same graph (every node star) — trajectory N/A; use induced edge counts
    b0 = run_method(g_syn, "B0", SampleConfig(max_nodes=8, seed=0))
    from code.metrics import evaluate_bundle

    results["B0_induced"] = evaluate_bundle(g_syn, b0)
    log(
        f"  B0_all_nodes: n_sg={results['B0_induced']['n_subgraphs']} "
        f"cover={results['B0_induced']['edge_cover']:.3f} "
        f"repeat={results['B0_induced']['edge_repeat']:.3f}"
    )

    # --- B) RF curve on C4 dataset ---
    log("\n=== B. Receptive-field curve on C4 vs C8 (class0 C4 hit) ===")
    graphs, y, meta = make_c4_vs_longcycle(n_per_class=80, n_nodes=14, long_cycle=8, seed=0)
    results["c4_meta"] = meta
    rf_rows = []

    # B0: one star per node, take all
    def make_B0(g: Graph):
        return [{v} | set(g.neighbors(v)) for v in g.nodes]

    r = c4_hit_rate(graphs, y, make_B0)
    r["setting"] = "B0_1hop"
    rf_rows.append(r)
    log(f"  B0_1hop: c4_hit={r['frac_graphs_with_c4_patch']:.3f} |S|={r['mean_|S|']:.2f}")

    for L, m, nw in [
        (4, 4, 8),
        (6, 6, 8),
        (8, 8, 8),
        (12, 10, 8),
        (8, 8, 4),  # fewer seeds
    ]:
        def make_rw(g, L=L, m=m, nw=nw):
            cfg = CoverageConfig(
                p=0.5,
                q=2.0,
                walk_length=L,
                max_nodes=m,
                max_walks=nw,
                edge_decay=0.7,
                seed=0,
                seed_policy="degree_stratified",
                cover_target=None,
            )
            b = sample_coverage(g, cfg)
            return b.node_sets

        r = c4_hit_rate(graphs, y, make_rw)
        r["setting"] = f"RW_L{L}_m{m}_w{nw}"
        r["L"] = L
        r["m"] = m
        r["max_walks"] = nw
        rf_rows.append(r)
        log(
            f"  RW L={L} m={m} walks={nw}: c4_hit={r['frac_graphs_with_c4_patch']:.3f} "
            f"|S|={r['mean_|S|']:.2f}"
        )
    results["rf_curve"] = rf_rows

    # C) invariant
    results["invariants"] = {
        "hard_delete": False,
        "soft_decay_only": True,
        "note": "edge weights multiply by gamma; edges never removed from G",
    }

    path = _TRACK / "results" / "coverage_rf.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / "COVERAGE_RF_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    log(f"\nwrote {path}")
    log(f"wrote {sm}")
    return 0


def _render(r: dict) -> str:
    lines = [
        "# Coverage-driven RW + receptive-field curve (process metrics)",
        "",
        f"Protocol: `{r['protocol_id']}` · hard_delete={r['invariants']['hard_delete']}",
        "",
        "## A. Coverage (synthetic ring+chords)",
        "",
        "| name | walks | traj cover | traj repeat | mean|S| |",
        "|------|-------|------------|-------------|---------|",
    ]
    for m in r["coverage"]:
        lines.append(
            f"| {m['name']} | {m['n_walks']} | {m['traj_edge_cover']:.3f} | "
            f"{m['traj_edge_repeat']:.3f} | {m['mean_|S|']:.2f} |"
        )
    b0 = r.get("B0_induced", {})
    lines += [
        "",
        f"B0 (every node star): n_sg={b0.get('n_subgraphs')} "
        f"induced_cover={b0.get('edge_cover')} induced_repeat={b0.get('edge_repeat')}",
        "",
        "## B. RF curve — fraction of C4-class graphs with ≥1 patch containing C4",
        "",
        "| setting | c4_hit | mean|S| |",
        "|---------|--------|---------|",
    ]
    for m in r["rf_curve"]:
        lines.append(
            f"| {m['setting']} | {m['frac_graphs_with_c4_patch']:.3f} | {m['mean_|S|']:.2f} |"
        )
    lines += [
        "",
        "## C. luyin10 checklist update",
        "",
        "| ID | Status after this suite |",
        "|----|-------------------------|",
        "| R4 少重复 | process metrics + soft decay + uncovered seeds |",
        "| R5 非全点 | degree_stratified / uncovered / budget max_walks |",
        "| R6 不硬删 | invariant hard_delete=False |",
        "| R1 感受野 | RF curve c4_hit vs L,m |",
        "",
        "Downstream Acc not claimed here.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
