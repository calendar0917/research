from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.classify import eval_logistic_cv  # noqa: E402
from code.data_tud import load_tud, node_feature_readout  # noqa: E402
from code.pipeline import (  # noqa: E402
    PipelineConfig,
    embed_dataset_shared_dict,
    graph_oracle_features,
    patch_pool_features,
)
from code.synthetic_task import (  # noqa: E402
    degree_features,
    make_distant_wedge,
    make_triangle_vs_longcycle,
)


def _pad_stack(rows: list[np.ndarray]) -> np.ndarray:
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, : r.shape[0]] = r
    return X


def _cv_acc(X, y, n_splits=5, seed=0, head="lr") -> dict:
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(n_splits, int(counts.min()), len(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if head == "lr":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed),
        )
    else:
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=400,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.15,
            ),
        )
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return {
        "acc_mean": float(scores.mean()),
        "acc_std": float(scores.std()),
        "n_splits": n_splits,
        "head": head,
        "scores": scores.tolist(),
    }


def shared_dict_cv(graphs, y, cfg: PipelineConfig, n_splits=5, seed=0, head="lr") -> dict:
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(n_splits, int(counts.min()), len(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        X, _ = embed_dataset_shared_dict(graphs, cfg, train_idx=tr)
        if head == "lr":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, random_state=seed + fold),
            )
        else:
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=400,
                    random_state=seed + fold,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            )
        clf.fit(X[tr], y[tr])
        scores.append(float(clf.score(X[te], y[te])))
    return {
        "acc_mean": float(np.mean(scores)),
        "acc_std": float(np.std(scores)),
        "n_splits": n_splits,
        "head": head,
        "seconds": time.time() - t0,
        "scores": scores,
        "protocol": "shared_dict_train_only",
    }


def run_distant() -> dict:
    print("=== Distant triangles (multi-hop) ===")
    graphs, y, meta = make_distant_wedge(
        n_per_class=100, n_nodes=24, seed=0, path_len=6
    )
    out: dict = {"meta": meta, "baselines": {}, "methods": {}}
    out["baselines"]["degree"] = _cv_acc(degree_features(graphs), y, 5, 0)
    out["baselines"]["oracle"] = _cv_acc(graph_oracle_features(graphs), y, 5, 0)
    print(f"  degree {out['baselines']['degree']['acc_mean']:.3f}")
    print(f"  oracle {out['baselines']['oracle']['acc_mean']:.3f}")

    for method, p, q, decay, L, mnodes, nw in [
        ("B0", 1.0, 1.0, 1.0, 6, 6, 24),  # small max_nodes ~1-hopish
        ("B0_wide", 1.0, 1.0, 1.0, 6, 14, 24),
        ("B1", 1.0, 1.0, 1.0, 12, 14, 30),
        ("M0_bfs", 1.0, 2.0, 0.7, 12, 14, 30),
        ("M0_dfs", 1.0, 0.5, 0.7, 12, 14, 30),
    ]:
        base_method = "B0" if method.startswith("B0") else ("B1" if method == "B1" else "M0")
        cfg = PipelineConfig(
            method=base_method,
            p=p,
            q=q,
            walk_length=L,
            max_nodes=mnodes,
            num_walks=nw,
            edge_decay=decay,
            seed=0,
            n_atoms=14,
            T=3,
            T_min=2,
            ksvd_iter=6,
            order_mode="bfs",
            readout_mode="rich",
        )
        # patch pool
        Xp = patch_pool_features(graphs, cfg)
        out["baselines"][f"pool_{method}"] = _cv_acc(Xp, y, 5, 0)
        # shared ksvd basic vs rich
        cfg.readout_mode = "basic"
        out["methods"][f"{method}_shared_basic"] = shared_dict_cv(
            graphs, y, cfg, 5, 0, head="lr"
        )
        cfg.readout_mode = "rich"
        out["methods"][f"{method}_shared_rich"] = shared_dict_cv(
            graphs, y, cfg, 5, 0, head="lr"
        )
        cfg.readout_mode = "rich"
        out["methods"][f"{method}_shared_rich_mlp"] = shared_dict_cv(
            graphs, y, cfg, 5, 0, head="mlp"
        )
        print(
            f"  {method}: pool={out['baselines'][f'pool_{method}']['acc_mean']:.3f} "
            f"ksvd_basic={out['methods'][f'{method}_shared_basic']['acc_mean']:.3f} "
            f"rich={out['methods'][f'{method}_shared_rich']['acc_mean']:.3f} "
            f"mlp={out['methods'][f'{method}_shared_rich_mlp']['acc_mean']:.3f}"
        )
    return out


def run_triangle_readout() -> dict:
    """Confirm rich readout on previous synth task."""
    print("=== Triangle task: readout ablate ===")
    graphs, y, meta = make_triangle_vs_longcycle(100, 16, 0)
    out = {"meta": meta, "methods": {}}
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
        order_mode="bfs",
    )
    for mode in ["basic", "rich"]:
        cfg.readout_mode = mode
        out["methods"][f"shared_{mode}_lr"] = shared_dict_cv(graphs, y, cfg, 5, 0, "lr")
        out["methods"][f"shared_{mode}_mlp"] = shared_dict_cv(graphs, y, cfg, 5, 0, "mlp")
        print(
            f"  {mode} lr={out['methods'][f'shared_{mode}_lr']['acc_mean']:.3f} "
            f"mlp={out['methods'][f'shared_{mode}_mlp']['acc_mean']:.3f}"
        )
    return out


def run_mutag_gate() -> dict:
    """Structure residual: concat vs gate (scale struct by learned-free heuristic)."""
    print("=== MUTAG gated fusion ===")
    graphs, y, node_feats, meta = load_tud("MUTAG")
    attr = _pad_stack([node_feature_readout(x) for x in node_feats])
    deg = degree_features(graphs)
    out: dict = {"meta": meta, "baselines": {}, "methods": {}}
    out["baselines"]["attr"] = _cv_acc(attr, y, 10, 0)
    out["baselines"]["degree"] = _cv_acc(deg, y, 10, 0)
    out["baselines"]["attr+deg"] = _cv_acc(np.hstack([attr, deg]), y, 10, 0)

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
        readout_mode="rich",
    )
    # honest shared CV with fusion inside fold
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(10, int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = {
        "struct": [],
        "concat": [],
        "gate_l2": [],
        "mlp_concat": [],
    }
    t0 = time.time()
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        Xs, _ = embed_dataset_shared_dict(graphs, cfg, train_idx=tr)
        # normalize struct energy per graph for gate
        s_norm = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8
        # gate: alpha = sigmoid(a - b * ||s||) fixed heuristic — downweight large noisy struct
        # learn alpha on train via 1D: use mean attr as proxy — simpler: scale struct to match attr energy
        attr_e = np.linalg.norm(attr, axis=1, keepdims=True) + 1e-8
        Xs_scaled = Xs * (attr_e / s_norm) * 0.25  # soft scale
        Xc = np.hstack([Xs, attr])
        Xg = np.hstack([Xs_scaled, attr])

        def fit_score(X, head="lr"):
            if head == "lr":
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=3000, random_state=fold),
                )
            else:
                clf = make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        max_iter=300,
                        random_state=fold,
                        early_stopping=True,
                        validation_fraction=0.15,
                    ),
                )
            clf.fit(X[tr], y[tr])
            return float(clf.score(X[te], y[te]))

        scores["struct"].append(fit_score(Xs))
        scores["concat"].append(fit_score(Xc))
        scores["gate_l2"].append(fit_score(Xg))
        scores["mlp_concat"].append(fit_score(Xc, head="mlp"))

    for k, sc in scores.items():
        out["methods"][k] = {
            "acc_mean": float(np.mean(sc)),
            "acc_std": float(np.std(sc)),
            "scores": sc,
        }
        print(f"  {k}: {out['methods'][k]['acc_mean']:.3f}±{out['methods'][k]['acc_std']:.3f}")
    out["seconds"] = time.time() - t0
    return out


def main() -> int:
    results = {"experiment": "next_v1"}
    results["triangle_readout"] = run_triangle_readout()
    results["distant"] = run_distant()
    results["mutag_gate"] = run_mutag_gate()
    path = _TRACK / "results" / "next.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / "NEXT_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    print(f"wrote {path}\nwrote {sm}")
    return 0


def _render(r: dict) -> str:
    lines = [
        "# Next experiments (readout + multi-hop + MUTAG gate)",
        "",
        "## A. Triangle task — readout",
        "",
        "| variant | Acc |",
        "|---------|-----|",
    ]
    for k, v in r["triangle_readout"]["methods"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v['acc_std']:.3f} |")
    lines += [
        "",
        "## B. Distant triangles (path_len=6)",
        "",
        f"Meta: `{r['distant']['meta']}`",
        "",
        "### Baselines",
        "",
        "| name | Acc |",
        "|------|-----|",
    ]
    for k, v in r["distant"]["baselines"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += ["", "### Methods", "", "| name | Acc |", "|------|-----|"]
    for k, v in r["distant"]["methods"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += [
        "",
        "## C. MUTAG fusion (honest shared D)",
        "",
        "### Baselines",
        "",
        "| name | Acc |",
        "|------|-----|",
    ]
    for k, v in r["mutag_gate"]["baselines"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += ["", "### Methods", "", "| name | Acc |", "|------|-----|"]
    for k, v in r["mutag_gate"]["methods"].items():
        lines.append(f"| {k} | {v['acc_mean']:.3f} ± {v.get('acc_std', 0):.3f} |")
    lines += [
        "",
        "## Reading guide",
        "",
        "- Distant: if B0_wide ≈ M0 but B0(small m) low → need larger receptive field.",
        "- If M0 > B0_wide → RW bias helps.",
        "- rich readout should ≥ basic on shared dict.",
        "- MUTAG gate should not hurt attr; ideal ≥ attr baseline.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
