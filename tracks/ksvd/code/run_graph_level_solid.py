"""
Solidify graph-level RW sampling (luyin10 main line).

1) Formal default algorithm (documented + fixed config)
2) Process table on many graphs: cover / repeat / n_walks / |S|
3) Closed loop: sample → shared D → readout → C4 graph classification

No GIN. No node-level fusion.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.graph import ring_chords  # noqa: E402
from code.graph_level import (  # noqa: E402
    GraphLevelConfig,
    encode_dataset,
    encode_graph,
    learn_shared_D_graph_level,
    sample_patches_B0,
    sample_patches_graph_level,
    bundle_to_Y,
)
from code.coverage_sample import trajectory_cover_metrics  # noqa: E402
from code.ksvd import ksvd, readout_X, _omp  # noqa: E402
from code.metrics import evaluate_bundle  # noqa: E402
from code.synthetic_task import (  # noqa: E402
    degree_features,
    make_c4_vs_longcycle,
    make_triangle_vs_longcycle,
)


def log(msg: str) -> None:
    print(msg, flush=True)


DEFAULT = GraphLevelConfig(
    p=0.5,
    q=2.0,
    walk_length=8,
    max_nodes=8,
    max_walks=12,
    edge_decay=0.7,
    seed_policy="degree_stratified",
    cover_target=0.95,
    n_atoms=12,
    T=3,
    T_min=2,
    ksvd_iter=6,
    readout_mode="rich",
    seed=0,
)


def process_table(graphs, cfg: GraphLevelConfig, mode: str, tag: str) -> dict:
    covers, repeats, nwalks, sizes = [], [], [], []
    for i, g in enumerate(graphs):
        if mode == "B0":
            b = sample_patches_B0(g, cfg.max_nodes, cfg.seed + i)
            # induced metrics
            m = evaluate_bundle(g, b)
            covers.append(m["edge_cover"])
            repeats.append(m["edge_repeat"])
            nwalks.append(m["n_subgraphs"])
            sizes.append(m["mean_|S|"])
        else:
            b, met = sample_patches_graph_level(g, cfg, seed=cfg.seed + i)
            covers.append(met["traj_edge_cover"])
            repeats.append(met["traj_edge_repeat"])
            nwalks.append(met["n_walks"])
            sizes.append(met["mean_|S|"])
    return {
        "tag": tag,
        "mode": mode,
        "n_graphs": len(graphs),
        "cover_mean": float(np.mean(covers)),
        "cover_std": float(np.std(covers)),
        "repeat_mean": float(np.mean(repeats)),
        "repeat_std": float(np.std(repeats)),
        "n_walks_mean": float(np.mean(nwalks)),
        "n_walks_std": float(np.std(nwalks)),
        "mean_|S|_mean": float(np.mean(sizes)),
        "budget_max_walks": cfg.max_walks if mode != "B0" else None,
    }


def shared_dict_cv_acc(graphs, y, cfg: GraphLevelConfig, mode: str, n_splits=5) -> dict:
    """Train-only D each fold; graph embedding → LR."""
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(n_splits, int(counts.min()), len(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    proc = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        log(f"  [{mode}] fold {fold}/{n_splits-1} train={len(tr)} test={len(te)}")
        D, dmeta = learn_shared_D_graph_level(graphs, tr, cfg, mode=mode)
        log(
            f"    D: patches={dmeta.get('n_patches_used')} "
            f"cover={dmeta.get('mean_traj_cover', float('nan')):.3f} "
            f"repeat={dmeta.get('mean_traj_repeat', float('nan')):.3f} "
            f"recon={dmeta.get('recon_rel', 0):.4f}"
        )
        X, metas = encode_dataset(graphs, D, cfg, mode=mode)
        # z-score on train
        mu, sig = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xz = (X - mu) / sig
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=fold),
        )
        clf.fit(Xz[tr], y[tr])
        acc = float(clf.score(Xz[te], y[te]))
        scores.append(acc)
        proc.append(
            {
                "fold": fold,
                "acc": acc,
                "test_cover": float(np.nanmean([metas[i].get("traj_edge_cover", np.nan) for i in te])),
                "test_repeat": float(np.nanmean([metas[i].get("traj_edge_repeat", np.nan) for i in te])),
                "test_recon": float(np.mean([metas[i]["recon_rel"] for i in te])),
            }
        )
        log(f"    acc={acc*100:.1f}%  test_recon={proc[-1]['test_recon']:.4f}")
    return {
        "mode": mode,
        "acc_mean": float(np.mean(scores)),
        "acc_std": float(np.std(scores)),
        "acc_percent_mean": float(np.mean(scores) * 100),
        "acc_percent_std": float(np.std(scores) * 100),
        "fold_details": proc,
        "seconds": time.time() - t0,
        "n_splits": n_splits,
    }


def main() -> int:
    cfg = DEFAULT
    results = {
        "protocol_id": "graph-level-solid-v0",
        "default_algorithm": {
            "name": "CoverageRW-KSVD-Readout",
            "steps": [
                "1. Seeds: degree_stratified (not all nodes)",
                "2. Each seed: node2vec walk (p,q), soft decay on traversed edges",
                "3. Cap nodes → induced G[S] as patch",
                "4. Stop if traj_edge_cover>=target or max_walks",
                "5. Never hard-delete edges",
                "6. Train: stack patches → KSVD shared D",
                "7. Per graph: encode patches with D → energy/rich readout → s_G",
                "8. Downstream: LR on s_G (synthetic graph classification)",
            ],
            "config": {
                "p": cfg.p,
                "q": cfg.q,
                "L": cfg.walk_length,
                "m": cfg.max_nodes,
                "max_walks": cfg.max_walks,
                "edge_decay": cfg.edge_decay,
                "seed_policy": cfg.seed_policy,
                "cover_target": cfg.cover_target,
                "n_atoms": cfg.n_atoms,
                "T": cfg.T,
            },
        },
        "note": "Graph-level only. Coverage metrics are graph-level. Not node classification.",
    }

    # ----- 2) process tables on many graphs -----
    log("=== Process metrics on many graphs ===")
    g_c4, y_c4, meta_c4 = make_c4_vs_longcycle(100, 14, 8, seed=0)
    g_tri, y_tri, meta_tri = make_triangle_vs_longcycle(80, 16, seed=1)
    # extra diverse synthetics
    extra = [ring_chords(20, [[0, 5], [3, 10], [7, 15]]) for _ in range(30)]
    for i in range(30):
        # vary slightly via different chord sets
        extra[i] = ring_chords(20 + (i % 5), [[0, 4 + i % 3], [2, 9], [5, 12]])

    proc = []
    for name, graphs, mode in [
        ("c4_set_coverage", g_c4, "coverage"),
        ("c4_set_B0", g_c4, "B0"),
        ("tri_set_coverage", g_tri, "coverage"),
        ("tri_set_B0", g_tri, "B0"),
        ("ring_family_coverage", extra, "coverage"),
        ("ring_family_B0", extra, "B0"),
    ]:
        log(f"  process {name} n={len(graphs)} ...")
        row = process_table(graphs, cfg, mode, name)
        proc.append(row)
        log(
            f"    cover={row['cover_mean']:.3f}±{row['cover_std']:.3f} "
            f"repeat={row['repeat_mean']:.3f}±{row['repeat_std']:.3f} "
            f"walks={row['n_walks_mean']:.1f} |S|={row['mean_|S|_mean']:.2f}"
        )
    results["process_tables"] = proc
    results["datasets_meta"] = {"c4": meta_c4, "triangle": meta_tri, "ring_family_n": len(extra)}

    # ----- 3) closed loop classification -----
    log("\n=== Closed loop: sample → KSVD → readout → classify (C4 vs C8) ===")
    # degree baseline
    Xdeg = degree_features(g_c4)
    deg_scores = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
        Xdeg,
        y_c4,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    results["baselines"] = {
        "degree": {
            "acc_mean": float(deg_scores.mean()),
            "acc_std": float(deg_scores.std()),
            "acc_percent_mean": float(deg_scores.mean() * 100),
        }
    }
    log(
        f"  degree: {results['baselines']['degree']['acc_percent_mean']:.1f}%"
        f"±{results['baselines']['degree']['acc_std']*100:.1f}"
    )

    results["closed_loop"] = {}
    for mode in ["B0", "coverage"]:
        log(f"\n--- closed loop mode={mode} ---")
        results["closed_loop"][mode] = shared_dict_cv_acc(g_c4, y_c4, cfg, mode=mode, n_splits=5)
        cl = results["closed_loop"][mode]
        log(
            f"  >> {mode}: Acc={cl['acc_percent_mean']:.1f}±{cl['acc_percent_std']:.1f}% "
            f"({cl['seconds']:.1f}s)"
        )

    # optional second task
    log("\n=== Closed loop on triangle vs longcycle (control) ===")
    results["closed_loop_triangle"] = {}
    for mode in ["B0", "coverage"]:
        log(f"\n--- triangle mode={mode} ---")
        results["closed_loop_triangle"][mode] = shared_dict_cv_acc(
            g_tri, y_tri, cfg, mode=mode, n_splits=5
        )
        cl = results["closed_loop_triangle"][mode]
        log(f"  >> {mode}: Acc={cl['acc_percent_mean']:.1f}±{cl['acc_percent_std']:.1f}%")

    path = _TRACK / "results" / "graph_level_solid.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / "GRAPH_LEVEL_SOLID_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    # algorithm doc
    alg = _TRACK / "notes" / "graph_level_algorithm.md"
    alg.write_text(_alg_doc(results), encoding="utf-8")
    log(f"\nwrote {path}")
    log(f"wrote {sm}")
    log(f"wrote {alg}")
    return 0


def _render(r: dict) -> str:
    lines = [
        "# Graph-level RW sampling — solidification",
        "",
        f"Protocol: `{r['protocol_id']}`",
        "",
        "## 1. Default algorithm",
        "",
        f"**Name:** {r['default_algorithm']['name']}",
        "",
    ]
    for s in r["default_algorithm"]["steps"]:
        lines.append(f"- {s}")
    lines += [
        "",
        f"Config: `{r['default_algorithm']['config']}`",
        "",
        "## 2. Process metrics (many graphs)",
        "",
        "| tag | n | cover | repeat | walks | mean|S| |",
        "|-----|---|-------|--------|-------|---------|",
    ]
    for p in r["process_tables"]:
        lines.append(
            f"| {p['tag']} | {p['n_graphs']} | "
            f"{p['cover_mean']:.3f}±{p['cover_std']:.3f} | "
            f"{p['repeat_mean']:.3f}±{p['repeat_std']:.3f} | "
            f"{p['n_walks_mean']:.1f} | {p['mean_|S|_mean']:.2f} |"
        )
    lines += [
        "",
        "## 3. Closed loop Acc (C4 vs C8, shared D, 5-fold)",
        "",
        f"Degree baseline: **{r['baselines']['degree']['acc_percent_mean']:.1f}%**",
        "",
        "| mode | Acc % |",
        "|------|-------|",
    ]
    for mode, cl in r["closed_loop"].items():
        lines.append(
            f"| {mode} | {cl['acc_percent_mean']:.1f} ± {cl['acc_percent_std']:.1f} |"
        )
    lines += [
        "",
        "### Triangle task (control)",
        "",
        "| mode | Acc % |",
        "|------|-------|",
    ]
    for mode, cl in r.get("closed_loop_triangle", {}).items():
        lines.append(
            f"| {mode} | {cl['acc_percent_mean']:.1f} ± {cl['acc_percent_std']:.1f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **Process:** coverage mode should use fewer walks than all-node B0 while keeping high cover; lower repeat is good **for graph-level corpus**.",
        "- **Closed loop:** coverage should beat B0 on C4 if sampling+KSVD capture multi-hop rings.",
        "- Triangle task may be less RF-sensitive; report honestly.",
        "- This does **not** claim MUTAG/GIN gains; graph-level only.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _alg_doc(r: dict) -> str:
    c = r["default_algorithm"]["config"]
    return f"""# 图级默认算法：CoverageRW-KSVD-Readout

> 对齐 luyin10：**一系列子图代表图 → KSVD → readout → 图分类**。  
> 与节点级 `node_struct`（每点 \(s_v\)）分离。

## 输入输出

- 输入：无向图 \(G=(V,E)\)
- 输出：图向量 \(s_G\\in\\mathbb{{R}}^d\\)；训练时还有共享字典 \(D\)

## 算法步骤

1. **种子**（非全点）：`degree_stratified`，预算 `max_walks={c['max_walks']}`  
2. **游走**：node2vec，`p={c['p']}`, `q={c['q']}`, `L={c['L']}`，不立刻回头  
3. **软降权**：轨迹边 `w ← w × {c['edge_decay']}`；**永不删边**  
4. **Patch**：访问节点去重 cap 到 `m={c['m']}` → **诱导子图** \(G[S]\)  
5. **早停**：轨迹边覆盖率 ≥ `{c['cover_target']}` 或用尽预算  
6. **学字典**（仅 train 图）：所有 patch 向量堆成 \(Y\\)，KSVD 得 \(D\\)（atoms={c['n_atoms']}, T={c['T']}）  
7. **编码一图**：同样采样得 \(Y_g\\)，\(x_j=\\mathrm{{OMP}}(D,y_j)\\)，  
   \(s_G = \\mathrm{{readout}}(X)\\) 拼 energy/usage（录音「能量」）  
8. **分类**：\(s_G\\) → 标准化 + LogisticRegression  

## 过程指标（图级）

| 指标 | 含义 |
|------|------|
| traj_edge_cover | 轨迹边覆盖率 |
| traj_edge_repeat | 多余重复质量 |
| n_walks | 实际 walk 数 |
| mean \\|S\\| | patch 规模 |

## 成功标准（本轨）

| 层 | 标准 |
|----|------|
| 过程 | 同 cover 下 walks 少于全点 B0；repeat 不爆炸 |
| 闭环 | C4 任务上 coverage 模式 Acc **明显高于** B0 与随机 |

## 明确不做

- 节点分类、GIN 融合（另模块）  
- 用低 repeat 约束节点级 \(s_v\\)（会抽空节点邻域）  

## 复现

```bash
python -m code.run_graph_level_solid
```
"""


if __name__ == "__main__":
    raise SystemExit(main())
