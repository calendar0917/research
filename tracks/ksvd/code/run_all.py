from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.classify import eval_logistic_cv  # noqa: E402
from code.graph import ring_chords  # noqa: E402
from code.metrics import evaluate_bundle  # noqa: E402
from code.pipeline import PipelineConfig, embed_dataset, sample_Y, structure_embedding  # noqa: E402
from code.sample import SampleConfig, run_method  # noqa: E402
from code.synthetic_task import make_ring_vs_path  # noqa: E402


def stage0(results: dict[str, Any]) -> None:
    g = ring_chords(20, [[0, 5], [3, 10], [7, 15]])
    methods = ["B0", "B1", "M0"]
    pq_grid = [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)]
    rows = []
    for p, q in pq_grid:
        for method in methods:
            for seed in [0, 1, 2]:
                cfg = SampleConfig(
                    p=p,
                    q=q,
                    walk_length=8,
                    max_nodes=12,
                    num_walks=30,
                    edge_decay=0.7 if method == "M0" else 1.0,
                    no_backtrack=True,
                    seed=seed,
                )
                if method == "B1":
                    cfg.p, cfg.q, cfg.edge_decay = 1.0, 1.0, 1.0
                bundle = run_method(g, method, cfg)
                row = evaluate_bundle(g, bundle)
                row.update({"seed": seed, "p": p, "q": q})
                rows.append(row)
    # summary default p=q=1
    def mean_for(method: str, p: float = 1.0, q: float = 1.0) -> dict:
        rs = [r for r in rows if r["method"] == method and r["p"] == p and r["q"] == q]
        keys = ["edge_cover", "edge_count_per_sg", "mean_|S|", "edge_repeat"]
        return {k: float(np.mean([r[k] for r in rs])) for k in keys} | {"n": len(rs)}

    results["stage0"] = {
        "protocol_id": "stage0-sample-v0",
        "graph": {"n": g.n, "edges": g.num_edges()},
        "per_run": rows,
        "summary_pq1": {m: mean_for(m) for m in methods},
        "summary_p0.5_q2": {m: mean_for(m, 0.5, 2.0) for m in methods},
    }


def stage1(results: dict[str, Any]) -> None:
    g = ring_chords(20, [[0, 5], [3, 10], [7, 15]])
    out = {}
    for method in ["B0", "B1", "M0"]:
        cfg = PipelineConfig(
            method=method,
            p=1.0,
            q=1.0 if method != "M0" else 0.5,
            walk_length=8,
            max_nodes=10,
            num_walks=25,
            edge_decay=0.7 if method == "M0" else 1.0,
            seed=0,
            n_atoms=10,
            T=3,
            T_min=2,
            ksvd_iter=8,
        )
        if method == "B1":
            cfg.p, cfg.q = 1.0, 1.0
        Y, meta = sample_Y(g, cfg)
        from code.ksvd import ksvd

        D, X, info = ksvd(Y, n_atoms=cfg.n_atoms, T=cfg.T, n_iter=cfg.ksvd_iter, seed=0, T_min=cfg.T_min)
        out[method] = {**meta, **info, "Y_shape": list(Y.shape)}
    results["stage1"] = {
        "protocol_id": "stage1-ksvd-smoke-v0",
        "methods": out,
    }


def stage2_synthetic(results: dict[str, Any]) -> None:
    graphs, y = make_ring_vs_path(n_per_class=50, n_nodes=12, seed=0)
    method_rows = {}
    for method, p, q, decay in [
        ("B0", 1.0, 1.0, 1.0),
        ("B1", 1.0, 1.0, 1.0),
        ("M0", 0.5, 2.0, 0.7),
    ]:
        cfg = PipelineConfig(
            method=method,
            p=p,
            q=q,
            walk_length=8,
            max_nodes=10,
            num_walks=15,
            edge_decay=decay,
            seed=0,
            n_atoms=8,
            T=3,
            T_min=2,
            ksvd_iter=6,
        )
        t0 = time.time()
        X, infos = embed_dataset(graphs, cfg)
        clf = eval_logistic_cv(X, y, n_splits=5, seed=0)
        recon = float(np.mean([i.get("recon_rel", 1.0) for i in infos]))
        atoms = float(np.mean([i.get("atoms_used", 0) for i in infos]))
        method_rows[method] = {
            **clf,
            "mean_recon_rel": recon,
            "mean_atoms_used": atoms,
            "emb_dim": int(X.shape[1]),
            "seconds": time.time() - t0,
        }
    # degree baseline
    deg_feats = []
    for g in graphs:
        degs = [len(g.neighbors(u)) for u in g.nodes]
        deg_feats.append(
            [
                float(np.mean(degs)),
                float(np.std(degs) if len(degs) > 1 else 0.0),
                float(np.max(degs)),
                float(g.n),
                float(g.num_edges()),
            ]
        )
    Xd = np.array(deg_feats, dtype=np.float64)
    method_rows["degree_stats"] = eval_logistic_cv(Xd, y, n_splits=5, seed=0)

    results["stage2_synthetic"] = {
        "protocol_id": "stage2a-synth-ringpath-v0",
        "task": "cycle(+noise) vs path",
        "n_graphs": len(graphs),
        "methods": method_rows,
    }


def stage2_tud(results: dict[str, Any], name: str = "MUTAG") -> None:
    try:
        from code.data_tud import load_tud
    except Exception as e:
        results["stage2_tud"] = {"error": f"import failed: {e}"}
        return
    try:
        graphs, y, meta = load_tud(name)
    except Exception as e:
        results["stage2_tud"] = {"error": f"load failed: {e}", "dataset": name}
        return

    # limit size for runtime if huge
    if len(graphs) > 300:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(graphs), size=300, replace=False)
        graphs = [graphs[i] for i in idx]
        y = y[idx]
        meta["subsampled"] = 300

    method_rows = {}
    for method, p, q, decay in [
        ("B0", 1.0, 1.0, 1.0),
        ("B1", 1.0, 1.0, 1.0),
        ("M0", 0.5, 2.0, 0.7),
    ]:
        cfg = PipelineConfig(
            method=method,
            p=p,
            q=q,
            walk_length=6,
            max_nodes=10,
            num_walks=12,
            edge_decay=decay,
            seed=0,
            n_atoms=8,
            T=3,
            T_min=2,
            ksvd_iter=5,
        )
        t0 = time.time()
        X, infos = embed_dataset(graphs, cfg)
        clf = eval_logistic_cv(X, y, n_splits=10, seed=0)
        method_rows[method] = {
            **clf,
            "mean_recon_rel": float(np.mean([i.get("recon_rel", 1.0) for i in infos])),
            "mean_atoms_used": float(np.mean([i.get("atoms_used", 0) for i in infos])),
            "seconds": time.time() - t0,
        }
    # degree baseline
    deg_feats = []
    for g in graphs:
        degs = [len(g.neighbors(u)) for u in g.nodes] or [0]
        deg_feats.append(
            [float(np.mean(degs)), float(np.max(degs)), float(g.n), float(g.num_edges())]
        )
    method_rows["degree_stats"] = eval_logistic_cv(
        np.array(deg_feats, dtype=np.float64), y, n_splits=10, seed=0
    )

    results["stage2_tud"] = {
        "protocol_id": "stage2a-tud-skfold-v0",
        "dataset": meta,
        "methods": method_rows,
        "note": "NOT comparable to CIN/GIN paper Acc (Xu max-val protocol)",
    }


def stage3_ablation(results: dict[str, Any]) -> None:
    """Light p,q grid on synthetic task for M0 only."""
    graphs, y = make_ring_vs_path(n_per_class=40, n_nodes=12, seed=1)
    grid = []
    for p in [0.5, 1.0, 2.0]:
        for q in [0.5, 1.0, 2.0]:
            cfg = PipelineConfig(
                method="M0",
                p=p,
                q=q,
                walk_length=8,
                max_nodes=10,
                num_walks=12,
                edge_decay=0.7,
                seed=0,
                n_atoms=8,
                T=3,
                T_min=2,
                ksvd_iter=5,
            )
            X, infos = embed_dataset(graphs, cfg)
            clf = eval_logistic_cv(X, y, n_splits=5, seed=0)
            grid.append(
                {
                    "p": p,
                    "q": q,
                    "acc_mean": clf["acc_mean"],
                    "acc_std": clf["acc_std"],
                    "recon": float(np.mean([i["recon_rel"] for i in infos])),
                }
            )
    best = max(grid, key=lambda r: r["acc_mean"])
    results["stage3_ablation"] = {
        "protocol_id": "stage3-pq-ablate-v0",
        "grid": grid,
        "best": best,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tud", action="store_true")
    ap.add_argument("--tud", default="MUTAG")
    ap.add_argument("--out", type=Path, default=_TRACK / "results" / "full_pipeline.json")
    args = ap.parse_args()

    results: dict[str, Any] = {"track": "ksvd", "status": "running"}
    t_all = time.time()

    print("=== Stage 0: sampling metrics ===")
    stage0(results)
    print(json.dumps(results["stage0"]["summary_pq1"], indent=2))

    print("=== Stage 1: KSVD smoke ===")
    stage1(results)
    for m, info in results["stage1"]["methods"].items():
        print(f"  {m}: recon={info['recon_rel']:.4f} atoms={info['atoms_used']} nnz={info['mean_nnz']:.2f}")

    print("=== Stage 2a: synthetic cycle vs path ===")
    stage2_synthetic(results)
    for m, info in results["stage2_synthetic"]["methods"].items():
        if "acc_mean" in info:
            print(f"  {m}: acc={info['acc_mean']:.3f}±{info.get('acc_std', 0):.3f}")

    if not args.skip_tud:
        print(f"=== Stage 2a: TUD {args.tud} ===")
        stage2_tud(results, args.tud)
        tud = results.get("stage2_tud", {})
        if "error" in tud:
            print("  ERROR", tud["error"])
        else:
            for m, info in tud.get("methods", {}).items():
                if "acc_mean" in info:
                    print(f"  {m}: acc={info['acc_mean']:.3f}±{info['acc_std']:.3f} ({info.get('seconds', 0):.1f}s)")

    print("=== Stage 3: p,q ablation (synthetic) ===")
    stage3_ablation(results)
    print("  best", results["stage3_ablation"]["best"])

    results["status"] = "done"
    results["total_seconds"] = time.time() - t_all
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # human summary
    summary_path = args.out.parent / "SUMMARY.md"
    summary_path.write_text(_render_summary(results), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    print(f"total {results['total_seconds']:.1f}s")
    return 0


def _render_summary(r: dict[str, Any]) -> str:
    lines = [
        "# KSVD pipeline results (auto)",
        "",
        f"Status: **{r.get('status')}** · wall {r.get('total_seconds', 0):.1f}s",
        "",
        "## Stage 0 — sampling (`stage0-sample-v0`)",
        "",
        "Synthetic ring+chords n=20. Metrics: cover / count_per_sg / mean|S|.",
        "",
        "```",
        json.dumps(r.get("stage0", {}).get("summary_pq1", {}), indent=2),
        "```",
        "",
        "## Stage 1 — KSVD smoke (`stage1-ksvd-smoke-v0`)",
        "",
    ]
    for m, info in r.get("stage1", {}).get("methods", {}).items():
        lines.append(
            f"- **{m}**: recon_rel={info.get('recon_rel'):.4f}, atoms_used={info.get('atoms_used')}, mean_nnz={info.get('mean_nnz'):.2f}"
        )
    lines += ["", "## Stage 2a — synthetic cycle vs path", ""]
    for m, info in r.get("stage2_synthetic", {}).get("methods", {}).items():
        if "acc_mean" in info:
            lines.append(
                f"- **{m}**: Acc {info['acc_mean']:.3f} ± {info.get('acc_std', 0):.3f}"
                + (f" · recon {info['mean_recon_rel']:.3f}" if "mean_recon_rel" in info else "")
            )
    lines += ["", "## Stage 2a — TUD (sklearn 10-fold mean, **not** CIN paper protocol)", ""]
    tud = r.get("stage2_tud", {})
    if "error" in tud:
        lines.append(f"ERROR: {tud['error']}")
    else:
        lines.append(f"Dataset: {tud.get('dataset', {})}")
        for m, info in tud.get("methods", {}).items():
            if "acc_mean" in info:
                lines.append(f"- **{m}**: Acc {info['acc_mean']:.3f} ± {info['acc_std']:.3f}")
    lines += ["", "## Stage 3 — M0 p,q grid (synthetic)", ""]
    ab = r.get("stage3_ablation", {})
    if ab:
        lines.append(f"Best: {ab.get('best')}")
        lines.append("")
        lines.append("| p | q | acc | recon |")
        lines.append("|---|---|-----|-------|")
        for row in ab.get("grid", []):
            lines.append(
                f"| {row['p']} | {row['q']} | {row['acc_mean']:.3f} | {row['recon']:.3f} |"
            )
    lines += [
        "",
        "## Not done / next",
        "",
        "- **ogbg-molhiv** (`ogb-molhiv-v0`): needs `ogb` package + download; not run here.",
        "- Do **not** compare TUD Acc above to CIN Table 2 (different protocol).",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
