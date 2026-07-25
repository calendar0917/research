from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.classify import eval_logistic_cv  # noqa: E402
from code.pipeline import (  # noqa: E402
    PipelineConfig,
    embed_dataset_shared_dict,
    graph_oracle_features,
    patch_pool_features,
)
from code.synthetic_task import (  # noqa: E402
    _count_c4,
    degree_features,
    make_c4_vs_longcycle,
)
from code.vectorize import adjacency_padded, flatten_upper, subgraph_extra_features  # noqa: E402
from code.sample import SampleConfig, run_method  # noqa: E402


def shared_dict_cv(graphs, y, cfg: PipelineConfig, n_splits=5, seed=0) -> dict:
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(n_splits, int(counts.min()), len(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        X, _ = embed_dataset_shared_dict(graphs, cfg, train_idx=tr)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed + fold),
        )
        clf.fit(X[tr], y[tr])
        scores.append(float(clf.score(X[te], y[te])))
    return {
        "acc_mean": float(np.mean(scores)),
        "acc_std": float(np.std(scores)),
        "n_splits": n_splits,
        "seconds": time.time() - t0,
        "scores": scores,
    }


def b0_strict_onehop_features(graphs) -> np.ndarray:
    """
    Only pure 1-hop stars: for each v, S={v}∪N(v), vectorize WITHOUT max_nodes expansion.
    Pool mean over v. max_nodes set to max |S| which is typically 3 on cycle verts.
    """
    rows = []
    for g in graphs:
        cols = []
        # use m=4 pad so C4 *could* fit if we had 4 nodes — but S only has 3 on cycle
        for v in g.nodes:
            S = {v} | set(g.neighbors(v))
            A = adjacency_padded(g, S, m=4, order_mode="bfs")
            vec = flatten_upper(A) + subgraph_extra_features(g, S)
            cols.append(vec)
        rows.append(np.mean(cols, axis=0))
    return np.array(rows, dtype=np.float64)


def main() -> int:
    print("=== C4 vs C8 (+trees), same n,|E| ===")
    graphs, y, meta = make_c4_vs_longcycle(
        n_per_class=120, n_nodes=14, long_cycle=8, seed=0
    )
    print("meta", meta)
    # sanity: label vs c4
    c4 = np.array([_count_c4(g) for g in graphs])
    print(
        f"  mean c4 | y=0: {c4[y==0].mean():.2f} | y=1: {c4[y==1].mean():.2f}"
    )

    results: dict = {"meta": meta, "baselines": {}, "methods": {}}

    results["baselines"]["degree"] = eval_logistic_cv(degree_features(graphs), y, 5, 0)
    results["baselines"]["oracle_graph"] = eval_logistic_cv(
        graph_oracle_features(graphs), y, 5, 0
    )
    results["baselines"]["c4_count_only"] = eval_logistic_cv(
        c4.reshape(-1, 1).astype(np.float64), y, 5, 0
    )
    results["baselines"]["b0_strict_1hop_pool"] = eval_logistic_cv(
        b0_strict_onehop_features(graphs), y, 5, 0
    )
    for k, v in results["baselines"].items():
        print(f"  base {k}: {v['acc_mean']:.3f}±{v['acc_std']:.3f}")

    # B0 with various max_nodes (implementation B0 uses stars, cap m)
    configs = [
        ("B0_m4", "B0", 1.0, 1.0, 1.0, 6, 4, 20),
        ("B0_m8", "B0", 1.0, 1.0, 1.0, 6, 8, 20),
        ("B1_L4_m6", "B1", 1.0, 1.0, 1.0, 4, 6, 40),
        ("B1_L8_m8", "B1", 1.0, 1.0, 1.0, 8, 8, 40),
        ("M0_L8_m8", "M0", 0.5, 2.0, 0.7, 8, 8, 40),
        ("M0_L12_m10", "M0", 1.0, 0.5, 0.7, 12, 10, 40),
    ]

    for name, method, p, q, decay, L, m, nw in configs:
        cfg = PipelineConfig(
            method=method,
            p=p,
            q=q,
            walk_length=L,
            max_nodes=m,
            num_walks=nw,
            edge_decay=decay,
            seed=0,
            n_atoms=16,
            T=4,
            T_min=2,
            ksvd_iter=6,
            order_mode="bfs",
            readout_mode="rich",
            append_local_stats=False,  # pure adjacency patches
        )
        # patch pool with local stats OFF for fair — actually pool uses append True in patch_pool_features
        # use pure adj pool:
        Xp = []
        for i, g in enumerate(graphs):
            c = PipelineConfig(**{**cfg.__dict__, "seed": cfg.seed + i * 17, "append_local_stats": False})
            from code.pipeline import sample_Y

            Y, _ = sample_Y(g, c)
            if Y.shape[1] == 0:
                Xp.append(np.zeros(Y.shape[0] or 1))
            else:
                Xp.append(Y.mean(axis=1))
        d = max(x.shape[0] for x in Xp)
        Xpool = np.zeros((len(Xp), d))
        for i, x in enumerate(Xp):
            Xpool[i, : x.shape[0]] = x
        results["baselines"][f"pool_{name}"] = eval_logistic_cv(Xpool, y, 5, 0)
        results["methods"][f"ksvd_{name}"] = shared_dict_cv(graphs, y, cfg, 5, 0)
        print(
            f"  {name}: pool={results['baselines'][f'pool_{name}']['acc_mean']:.3f} "
            f"ksvd={results['methods'][f'ksvd_{name}']['acc_mean']:.3f}±"
            f"{results['methods'][f'ksvd_{name}']['acc_std']:.3f} "
            f"({results['methods'][f'ksvd_{name}']['seconds']:.1f}s)"
        )

    # oracle: does any patch of size 4 contain C4 in class0?
    hit = 0
    for g, lab in zip(graphs, y):
        if lab != 0:
            continue
        sc = SampleConfig(walk_length=8, max_nodes=4, num_walks=30, seed=0, edge_decay=1.0)
        bundle = run_method(g, "B1", sc)
        ok = False
        for S in bundle.node_sets:
            if _count_c4(g.induced(S)) > 0:
                ok = True
                break
        hit += int(ok)
    results["diag"] = {
        "class0_frac_with_some_RW_patch_containing_c4": hit / max(int((y == 0).sum()), 1)
    }
    print("  diag", results["diag"])

    path = _TRACK / "results" / "c4.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / "C4_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    print(f"wrote {path}\nwrote {sm}")
    return 0


def _render(r: dict) -> str:
    lines = [
        "# C4 vs long cycle — multi-hop probe",
        "",
        f"Meta: `{r['meta']}`",
        "",
        "## Baselines",
        "",
        "| name | Acc |",
        "|------|-----|",
    ]
    for k, v in r["baselines"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += ["", "## Shared-dict KSVD", "", "| name | Acc |", "|------|-----|"]
    for k, v in r["methods"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += [
        "",
        f"## Diag",
        "",
        f"`{r.get('diag')}`",
        "",
        "## Expected pattern (if task works)",
        "",
        "- `b0_strict_1hop_pool` / `B0_m4` ≈ 0.5–0.65 (cannot see C4)",
        "- `B1_L8_m8` / `M0_*` higher if RW covers cycle",
        "- `c4_count_only` ≈ 1.0 (label check)",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
