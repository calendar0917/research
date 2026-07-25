"""
Systematic design ablation on molhiv (medium n, relative ranking).

Protocol: molhiv-design-ablation-v0
Goal: which *link* (sample / RF / pool / dict) moves valid AUC vs degree.

Usage:
  python -m code.run_molhiv_design_ablation --max-graphs 6000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import degree_hist_features, load_molhiv  # noqa: E402
from code.graph_level import (  # noqa: E402
    GraphLevelConfig,
    encode_graph,
    learn_shared_D_graph_level,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def fit_auc(Xtr, ytr, Xva, yva, Xte, yte, seed: int = 0) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, random_state=seed, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)

    def _auc(X, y):
        if len(np.unique(y)) < 2:
            return float("nan")
        proba = clf.predict_proba(X)
        classes = list(clf.named_steps["logisticregression"].classes_)
        if 1.0 in classes:
            col = classes.index(1.0)
        elif 1 in classes:
            col = classes.index(1)
        else:
            col = int(np.argmax(classes))
        return float(roc_auc_score(y, proba[:, col]))

    return {
        "train_auc": _auc(Xtr, ytr),
        "valid_auc": _auc(Xva, yva),
        "test_auc": _auc(Xte, yte),
    }


def encode_indices(graphs, indices, D, cfg, mode: str) -> np.ndarray:
    rows = []
    for i in indices:
        s, _ = encode_graph(graphs[int(i)], D, cfg, mode=mode, seed=cfg.seed + int(i) * 17)
        rows.append(s)
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for j, r in enumerate(rows):
        X[j, : r.shape[0]] = r
    return X


def process_features(graphs, cfg: GraphLevelConfig, mode: str) -> np.ndarray:
    """Cheap process-metric vector (negative control: cover alone shouldn't classify)."""
    from code.graph_level import sample_patches_B0, sample_patches_graph_level

    rows = []
    for i, g in enumerate(graphs):
        if mode == "B0":
            b = sample_patches_B0(g, cfg.max_nodes, cfg.seed + i)
            from code.metrics import evaluate_bundle

            m = evaluate_bundle(g, b)
            rows.append(
                [
                    m.get("edge_cover", 0.0),
                    m.get("edge_repeat", 0.0),
                    m.get("n_subgraphs", 0.0),
                    m.get("mean_|S|", 0.0),
                    float(g.n),
                    float(g.num_edges()),
                ]
            )
        else:
            b, met = sample_patches_graph_level(g, cfg, seed=cfg.seed + i)
            rows.append(
                [
                    met.get("traj_edge_cover", 0.0),
                    met.get("traj_edge_repeat", 0.0),
                    met.get("n_walks", 0.0),
                    float(np.mean([len(s) for s in b.node_sets]) if b.node_sets else 0),
                    float(g.n),
                    float(g.num_edges()),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def base_cfg(seed: int) -> GraphLevelConfig:
    return GraphLevelConfig(
        p=0.5,
        q=2.0,
        walk_length=8,
        max_nodes=8,
        max_walks=12,
        edge_decay=0.7,
        seed_policy="degree_stratified",
        cover_target=0.95,
        n_atoms=16,
        T=3,
        T_min=2,
        ksvd_iter=6,
        readout_mode="pool",
        pool="max",
        seed=seed,
        max_train_patches=6000,
    )


def design_grid(seed: int) -> list[dict[str, Any]]:
    """
    Named designs. Same link changes share group for ranking.
    mode: B0 | coverage
    """
    b = base_cfg(seed)
    specs: list[dict[str, Any]] = []

    def add(name: str, group: str, mode: str, **kw):
        cfg = replace(b, **kw) if kw else replace(b)
        specs.append({"name": name, "group": group, "mode": mode, "cfg": cfg})

    # --- A pool (fixed default coverage) ---
    for pool in ("mean", "max", "attn"):
        add(f"pool_{pool}", "pool", "coverage", pool=pool, readout_mode="pool")
    add("pool_rich", "pool", "coverage", pool="mean", readout_mode="rich")

    # --- B sampler ---
    add("samp_B0", "sampler", "B0", pool="max", readout_mode="pool")
    add("samp_cov", "sampler", "coverage", pool="max", readout_mode="pool")
    add(
        "samp_cov_no_earlystop",
        "sampler",
        "coverage",
        pool="max",
        cover_target=None,
        max_walks=12,
    )
    add(
        "samp_cov_uncovered",
        "sampler",
        "coverage",
        pool="max",
        seed_policy="uncovered",
    )

    # --- C receptive field ---
    add("rf_L4m4", "rf", "coverage", walk_length=4, max_nodes=4, pool="max")
    add("rf_L8m8", "rf", "coverage", walk_length=8, max_nodes=8, pool="max")
    add(
        "rf_L12m12",
        "rf",
        "coverage",
        walk_length=12,
        max_nodes=12,
        max_walks=16,
        pool="max",
    )

    # --- D walk bias p,q ---
    add("bias_BFS", "bias", "coverage", p=0.5, q=2.0, pool="max")
    add("bias_DFS", "bias", "coverage", p=2.0, q=0.5, pool="max")
    add("bias_neutral", "bias", "coverage", p=1.0, q=1.0, pool="max")

    # --- E dictionary ---
    add("dict_A8T2", "dict", "coverage", n_atoms=8, T=2, T_min=1, pool="max")
    add("dict_A16T3", "dict", "coverage", n_atoms=16, T=3, T_min=2, pool="max")
    add("dict_A24T4", "dict", "coverage", n_atoms=24, T=4, T_min=2, pool="max")

    return specs


def cfg_key(mode: str, cfg: GraphLevelConfig) -> str:
    """Configs that share the same sampling+dict can share D (pool may differ)."""
    return json.dumps(
        {
            "mode": mode,
            "p": cfg.p,
            "q": cfg.q,
            "L": cfg.walk_length,
            "m": cfg.max_nodes,
            "mw": cfg.max_walks,
            "decay": cfg.edge_decay,
            "seed_policy": cfg.seed_policy,
            "cover": cfg.cover_target,
            "atoms": cfg.n_atoms,
            "T": cfg.T,
            "Tmin": cfg.T_min,
            "ksvd_iter": cfg.ksvd_iter,
            "seed": cfg.seed,
            "max_patches": cfg.max_train_patches,
        },
        sort_keys=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", type=str, default="all", help="comma groups or all")
    args = ap.parse_args()

    t0 = time.time()
    log(f"Load molhiv max_graphs={args.max_graphs} seed={args.seed}")
    bundle = load_molhiv(max_graphs=args.max_graphs, seed=args.seed)
    graphs, y = bundle.graphs, bundle.y
    tr, va, te = bundle.split["train"], bundle.split["valid"], bundle.split["test"]
    log(f"  {bundle.meta}")
    log(f"  pos train/val/test={y[tr].sum():.0f}/{y[va].sum():.0f}/{y[te].sum():.0f}")

    results: dict[str, Any] = {
        "protocol_id": "molhiv-design-ablation-v0",
        "meta": bundle.meta,
        "seed": args.seed,
        "baselines": {},
        "designs": {},
        "by_group_best": {},
    }

    # baselines
    log("Baseline: degree")
    Xdeg = degree_hist_features(graphs)
    results["baselines"]["degree"] = fit_auc(Xdeg[tr], y[tr], Xdeg[va], y[va], Xdeg[te], y[te], args.seed)
    log(f"  degree val={results['baselines']['degree']['valid_auc']:.4f} test={results['baselines']['degree']['test_auc']:.4f}")

    log("Baseline: size (n, |E|)")
    Xsz = np.array([[g.n, g.num_edges()] for g in graphs], dtype=np.float64)
    results["baselines"]["size"] = fit_auc(Xsz[tr], y[tr], Xsz[va], y[va], Xsz[te], y[te], args.seed)
    log(f"  size val={results['baselines']['size']['valid_auc']:.4f} test={results['baselines']['size']['test_auc']:.4f}")

    log("Baseline: process metrics (coverage default)")
    Xp = process_features(graphs, base_cfg(args.seed), "coverage")
    results["baselines"]["process_cov"] = fit_auc(Xp[tr], y[tr], Xp[va], y[va], Xp[te], y[te], args.seed)
    log(
        f"  process val={results['baselines']['process_cov']['valid_auc']:.4f} "
        f"test={results['baselines']['process_cov']['test_auc']:.4f}"
    )

    want = None if args.groups == "all" else set(args.groups.split(","))
    specs = design_grid(args.seed)
    if want:
        specs = [s for s in specs if s["group"] in want]

    # share D across same sampling/dict
    D_cache: dict[str, tuple[np.ndarray, dict]] = {}

    for i, spec in enumerate(specs):
        name, group, mode, cfg = spec["name"], spec["group"], spec["mode"], spec["cfg"]
        log(f"[{i+1}/{len(specs)}] {name} (group={group}, mode={mode})")
        key = cfg_key(mode, cfg)
        if key not in D_cache:
            D, dinfo = learn_shared_D_graph_level(graphs, tr, cfg, mode=mode)
            D_cache[key] = (D, dinfo)
            log(
                f"  learn D recon={dinfo.get('recon_rel'):.4f} "
                f"patches={dinfo.get('n_patches_used')} cover={dinfo.get('mean_traj_cover')}"
            )
        else:
            D, dinfo = D_cache[key]
            log("  reuse D")

        Xtr = encode_indices(graphs, tr, D, cfg, mode)
        Xva = encode_indices(graphs, va, D, cfg, mode)
        Xte = encode_indices(graphs, te, D, cfg, mode)
        auc = fit_auc(Xtr, y[tr], Xva, y[va], Xte, y[te], args.seed)
        row = {
            **auc,
            "group": group,
            "mode": mode,
            "dim": int(Xtr.shape[1]),
            "recon_rel": dinfo.get("recon_rel"),
            "mean_traj_cover": dinfo.get("mean_traj_cover"),
            "mean_n_walks": dinfo.get("mean_n_walks"),
            "cfg": {
                "p": cfg.p,
                "q": cfg.q,
                "L": cfg.walk_length,
                "m": cfg.max_nodes,
                "pool": cfg.pool,
                "readout": cfg.readout_mode,
                "n_atoms": cfg.n_atoms,
                "T": cfg.T,
                "seed_policy": cfg.seed_policy,
                "cover_target": cfg.cover_target,
            },
        }
        results["designs"][name] = row
        log(f"  val={auc['valid_auc']:.4f} test={auc['test_auc']:.4f} dim={Xtr.shape[1]}")

    # rank
    ranked = sorted(
        results["designs"].items(),
        key=lambda kv: (kv[1]["valid_auc"] if kv[1]["valid_auc"] == kv[1]["valid_auc"] else -1),
        reverse=True,
    )
    results["ranked_by_valid"] = [
        {"name": n, "valid_auc": v["valid_auc"], "test_auc": v["test_auc"], "group": v["group"]}
        for n, v in ranked
    ]
    for g in sorted({s["group"] for s in specs}):
        cand = [(n, v) for n, v in results["designs"].items() if v["group"] == g]
        if not cand:
            continue
        best = max(cand, key=lambda kv: kv[1]["valid_auc"])
        results["by_group_best"][g] = {
            "name": best[0],
            "valid_auc": best[1]["valid_auc"],
            "test_auc": best[1]["test_auc"],
        }

    results["elapsed_sec"] = round(time.time() - t0, 2)
    out_dir = _TRACK / "results" / "molhiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"design_ablation_n{bundle.meta['n_used']}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # markdown
    deg_v = results["baselines"]["degree"]["valid_auc"]
    lines = [
        "# molhiv design ablation",
        "",
        f"Protocol: `molhiv-design-ablation-v0` · n={bundle.meta['n_used']} · seed={args.seed}",
        "",
        "## Baselines",
        "",
        "| name | valid AUC | test AUC |",
        "|------|-----------|----------|",
    ]
    for k, v in results["baselines"].items():
        lines.append(f"| {k} | {v['valid_auc']:.4f} | {v['test_auc']:.4f} |")
    lines += [
        "",
        f"Degree valid = **{deg_v:.4f}** (designs should beat this to claim structure signal)",
        "",
        "## All designs (by valid AUC)",
        "",
        "| rank | name | group | valid | test | Δval−deg |",
        "|------|------|-------|-------|------|----------|",
    ]
    for r, item in enumerate(results["ranked_by_valid"], 1):
        dlt = item["valid_auc"] - deg_v
        lines.append(
            f"| {r} | {item['name']} | {item['group']} | {item['valid_auc']:.4f} | "
            f"{item['test_auc']:.4f} | {dlt:+.3f} |"
        )
    lines += ["", "## Best per group", ""]
    for g, v in results["by_group_best"].items():
        lines.append(f"- **{g}**: `{v['name']}` val={v['valid_auc']:.4f} test={v['test_auc']:.4f}")
    lines += [
        "",
        "## How to read",
        "",
        "- Primary: **valid AUC** ranking (scaffold; avoid test shopping).",
        "- Viable direction: consistently **> degree** on valid, and not only high test.",
        "- If process_cov ≈ best design → sampling stats leak, not KSVD structure.",
        "",
        f"Elapsed: {results['elapsed_sec']}s",
        "",
    ]
    mpath = out_dir / f"DESIGN_ABLATION_n{bundle.meta['n_used']}_SUMMARY.md"
    mpath.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {jpath}")
    log(f"Wrote {mpath}")
    log("Top-5 by valid:")
    for item in results["ranked_by_valid"][:5]:
        log(f"  {item['name']}: val={item['valid_auc']:.4f} test={item['test_auc']:.4f}")


if __name__ == "__main__":
    main()
