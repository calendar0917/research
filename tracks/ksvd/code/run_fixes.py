from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.classify import eval_logistic_cv  # noqa: E402
from code.data_tud import load_tud, node_feature_readout  # noqa: E402
from code.pipeline import (  # noqa: E402
    PipelineConfig,
    embed_dataset,
    embed_dataset_shared_dict,
    graph_oracle_features,
    patch_pool_features,
)
from code.synthetic_task import degree_features, make_triangle_vs_longcycle  # noqa: E402


def _pad_stack(rows: list[np.ndarray]) -> np.ndarray:
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, : r.shape[0]] = r
    return X


def shared_dict_cv(
    graphs,
    y,
    cfg: PipelineConfig,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Proper CV: D learned only on train folds."""
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(n_splits, int(counts.min()), len(y))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        X, meta = embed_dataset_shared_dict(graphs, cfg, train_idx=tr)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed + fold),
        )
        clf.fit(X[tr], y[tr])
        scores.append(float(clf.score(X[te], y[te])))
    return {
        "acc_mean": float(np.mean(scores)),
        "acc_std": float(np.std(scores)),
        "n_splits": n_splits,
        "scores": scores,
        "seconds": time.time() - t0,
        "protocol": "shared_dict_D_train_only",
    }


def run_synth() -> dict:
    graphs, y, meta = make_triangle_vs_longcycle(n_per_class=100, n_nodes=16, seed=0)
    print("=== Synth triangle vs long-cycle ===")
    out: dict = {"meta": meta, "baselines": {}, "ksvd": {}}

    Xdeg = degree_features(graphs)
    Xorc = graph_oracle_features(graphs)
    out["baselines"]["degree"] = eval_logistic_cv(Xdeg, y, n_splits=5, seed=0)
    out["baselines"]["oracle_tri_stats"] = eval_logistic_cv(Xorc, y, n_splits=5, seed=0)
    print(f"  degree: {out['baselines']['degree']['acc_mean']:.3f}")
    print(f"  oracle triangle stats: {out['baselines']['oracle_tri_stats']['acc_mean']:.3f}")

    # patch pool no KSVD (BFS order + local stats)
    cfg_pool = PipelineConfig(
        method="B0",
        max_nodes=10,
        num_walks=16,
        walk_length=8,
        seed=0,
        order_mode="bfs",
        append_local_stats=True,
    )
    for method in ["B0", "B1", "M0"]:
        cfg_pool.method = method
        if method == "M0":
            cfg_pool.p, cfg_pool.q, cfg_pool.edge_decay = 0.5, 2.0, 0.7
        else:
            cfg_pool.p, cfg_pool.q, cfg_pool.edge_decay = 1.0, 1.0, 1.0
        Xp = patch_pool_features(graphs, cfg_pool)
        out["baselines"][f"patch_pool_{method}"] = eval_logistic_cv(Xp, y, n_splits=5, seed=0)
        print(
            f"  patch_pool {method}: {out['baselines'][f'patch_pool_{method}']['acc_mean']:.3f}"
        )

    # KSVD variants
    variants = [
        ("per_graph_degree_order", dict(order_mode="degree", shared=False)),
        ("per_graph_bfs", dict(order_mode="bfs", shared=False)),
        ("per_graph_bfs+localstats_in_Y", dict(order_mode="bfs", shared=False, append_local_stats=True)),
        ("shared_bfs", dict(order_mode="bfs", shared=True)),
        ("shared_bfs+localstats", dict(order_mode="bfs", shared=True, append_local_stats=True)),
    ]
    for name, opts in variants:
        shared = opts.pop("shared")
        cfg = PipelineConfig(
            method="M0",
            p=0.5,
            q=2.0,
            walk_length=10,
            max_nodes=12,
            num_walks=18,
            edge_decay=0.7,
            seed=0,
            n_atoms=12,
            T=3,
            T_min=2,
            ksvd_iter=6,
            **opts,
        )
        t0 = time.time()
        if shared:
            res = shared_dict_cv(graphs, y, cfg, n_splits=5, seed=0)
        else:
            X, infos = embed_dataset(graphs, cfg)
            res = eval_logistic_cv(X, y, n_splits=5, seed=0)
            res["mean_recon"] = float(np.mean([i["recon_rel"] for i in infos]))
            res["seconds"] = time.time() - t0
        out["ksvd"][name] = res
        print(f"  ksvd {name}: {res['acc_mean']:.3f}±{res['acc_std']:.3f} ({res.get('seconds', 0):.1f}s)")

    # also B0 shared
    cfg_b0 = PipelineConfig(
        method="B0",
        max_nodes=10,
        num_walks=16,
        seed=0,
        n_atoms=12,
        T=3,
        T_min=2,
        ksvd_iter=6,
        order_mode="bfs",
    )
    out["ksvd"]["shared_B0_bfs"] = shared_dict_cv(graphs, y, cfg_b0, n_splits=5, seed=0)
    print(
        f"  ksvd shared_B0_bfs: {out['ksvd']['shared_B0_bfs']['acc_mean']:.3f}±"
        f"{out['ksvd']['shared_B0_bfs']['acc_std']:.3f}"
    )
    return out


def run_mutag() -> dict:
    graphs, y, node_feats, meta = load_tud("MUTAG")
    print("=== MUTAG fixes ===")
    attr = _pad_stack([node_feature_readout(x) for x in node_feats])
    deg = degree_features(graphs)
    out: dict = {"meta": meta, "baselines": {}, "ksvd": {}}
    out["baselines"]["attr"] = eval_logistic_cv(attr, y, n_splits=10, seed=0)
    out["baselines"]["degree"] = eval_logistic_cv(deg, y, n_splits=10, seed=0)
    out["baselines"]["oracle"] = eval_logistic_cv(graph_oracle_features(graphs), y, n_splits=10, seed=0)
    print(f"  attr {out['baselines']['attr']['acc_mean']:.3f} deg {out['baselines']['degree']['acc_mean']:.3f}")

    cfg = PipelineConfig(
        method="M0",
        p=0.5,
        q=2.0,
        walk_length=6,
        max_nodes=10,
        num_walks=12,
        edge_decay=0.7,
        seed=0,
        n_atoms=10,
        T=3,
        T_min=2,
        ksvd_iter=5,
        order_mode="bfs",
    )
    # shared struct
    res = shared_dict_cv(graphs, y, cfg, n_splits=10, seed=0)
    out["ksvd"]["shared_M0"] = res
    print(f"  shared_M0 struct: {res['acc_mean']:.3f}")

    # re-embed once with all data D for fusion diagnostic (slightly optimistic) + report
    X, meta_e = embed_dataset_shared_dict(graphs, cfg, train_idx=np.arange(len(graphs)))
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    def cv(X):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
        sc = cross_val_score(
            clf, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=0), scoring="accuracy"
        )
        return {"acc_mean": float(sc.mean()), "acc_std": float(sc.std()), "note": "D_on_all_optimistic"}

    out["ksvd"]["shared_M0_allD_struct"] = cv(X)
    out["ksvd"]["shared_M0_allD_s+attr"] = cv(np.hstack([X, attr]))
    out["ksvd"]["shared_M0_allD_s+attr+deg"] = cv(np.hstack([X, attr, deg]))
    # patch pool + attr
    Xp = patch_pool_features(graphs, cfg)
    out["baselines"]["patch_pool_M0"] = eval_logistic_cv(Xp, y, n_splits=10, seed=0)
    out["baselines"]["patch_pool_M0+attr"] = eval_logistic_cv(np.hstack([Xp, attr]), y, n_splits=10, seed=0)
    for k in [
        "shared_M0_allD_struct",
        "shared_M0_allD_s+attr",
        "shared_M0_allD_s+attr+deg",
        "patch_pool_M0",
        "patch_pool_M0+attr",
    ]:
        src = out["ksvd"] if k.startswith("shared") else out["baselines"]
        print(f"  {k}: {src[k]['acc_mean']:.3f}")
    return out


def main() -> int:
    results = {"experiment": "fixes_v1"}
    results["synth"] = run_synth()
    results["mutag"] = run_mutag()
    path = _TRACK / "results" / "fixes.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / "FIXES_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    print(f"wrote {path}\nwrote {sm}")
    return 0


def _render(r: dict) -> str:
    s = r["synth"]
    m = r["mutag"]
    lines = [
        "# Fix experiments v1",
        "",
        "## Fixes implemented",
        "",
        "1. **BFS node order** in induced pad (vs degree-only)",
        "2. **Shared dictionary** (train-only D in CV)",
        "3. **Oracle** whole-graph triangle stats",
        "4. **Patch-pool** (no KSVD): mean of patch adj+local stats",
        "5. Optional **local stats in Y** before KSVD",
        "",
        "## Synth: triangle vs long-cycle",
        "",
        f"Meta: `{s['meta']}`",
        "",
        "### Baselines",
        "",
        "| name | Acc |",
        "|------|-----|",
    ]
    for k, v in s["baselines"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += ["", "### KSVD variants", "", "| name | Acc |", "|------|-----|"]
    for k, v in s["ksvd"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += [
        "",
        "## MUTAG",
        "",
        f"Meta: `{m['meta']}`",
        "",
        "### Baselines",
        "",
        "| name | Acc |",
        "|------|-----|",
    ]
    for k, v in m["baselines"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += ["", "### KSVD / fusion", "", "| name | Acc |", "|------|-----|"]
    for k, v in m["ksvd"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += [
        "",
        "## How to read",
        "",
        "- If **oracle ~1.0** on synth: labels are linearly separable by triangle count.",
        "- If **patch_pool >> degree** but **KSVD low**: sampling OK, dictionary/readout loses signal.",
        "- If **shared >> per-graph**: cross-graph alignment was the bottleneck.",
        "- MUTAG: compare shared_M0 (honest CV) vs attr baseline.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
