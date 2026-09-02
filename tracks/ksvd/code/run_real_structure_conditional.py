"""Conditional KSVD audit on real structure-only graph datasets.

The primary question is deliberately narrower than raw benchmark accuracy:

    Does a patch representation add predictive information after controlling for
    graph size, density, degree distribution, triangles, and components?

Every dictionary/PCA transform is fitted on the outer-training fold.  Classifier
regularization is selected by inner CV using only the outer-training labels.
The unsupervised outer-train representation is held fixed during inner CV; the
outer test fold remains untouched.

Example:
  cd tracks/ksvd
  PY=/home/calendar/.conda/envs/gsn-official/bin/python
  $PY -m code.run_real_structure_conditional \
      --datasets IMDB-BINARY IMDB-MULTI --variants cleaned --methods B0 \
      --split-seeds 0 1 2 --n-splits 5 --n-iter 10
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud
from .ksvd import ksvd
from .run_real_structure_ksvd import (
    _atomic_json,
    _encode_dictionary,
    _encode_pca,
    _subset_per_class,
    clustered_real_dictionary,
    degree_hist_features,
    graph_basic_features,
    prepare_graphs,
    random_real_dictionary,
)
from .sampling_route import MatchedSamplingConfig, relation_graph_readout

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "real_structure_conditional_20260730.json"
DEFAULT_MD = ROOT / "docs" / "REAL_STRUCTURE_CONDITIONAL_20260730.md"

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _score_estimator(clf: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = clf.predict(X)
    prob = clf.predict_proba(X)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    try:
        if prob.shape[1] == 2:
            out["roc_auc"] = float(roc_auc_score(y, prob[:, 1]))
        else:
            out["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y, prob, multi_class="ovr", average="weighted")
            )
    except ValueError:
        pass
    return out


def _fit_score_nested(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int,
    inner_splits: int,
) -> tuple[dict[str, float], float]:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=seed, solver="lbfgs"),
    )
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 47)
    search = GridSearchCV(
        pipe,
        {"logisticregression__C": list(C_GRID)},
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(Xtr, ytr)
    return _score_estimator(search.best_estimator_, Xte, yte), float(search.best_params_["logisticregression__C"])


def _residualize(
    stats: np.ndarray,
    feature: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the train-fold linear component of a feature block explained by controls."""
    ss = StandardScaler().fit(stats[tr])
    fs = StandardScaler().fit(feature[tr])
    Str, Ste = ss.transform(stats[tr]), ss.transform(stats[te])
    Ftr, Fte = fs.transform(feature[tr]), fs.transform(feature[te])
    reg = Ridge(alpha=alpha).fit(Str, Ftr)
    return Ftr - reg.predict(Str), Fte - reg.predict(Ste)


def _summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(folds[0]["score"])
    metrics = sorted(folds[0]["score"][methods[0]])
    out: dict[str, Any] = {"methods": {}, "paired_delta_vs_stats": {}}
    for method in methods:
        out["methods"][method] = {}
        for metric in metrics:
            if metric not in folds[0]["score"][method]:
                continue
            vals = np.asarray([f["score"][method][metric] for f in folds], dtype=np.float64)
            out["methods"][method][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "values": vals.tolist(),
            }
        out["methods"][method]["selected_C"] = [float(f["selected_C"][method]) for f in folds]

    baseline = "stats_degree"
    base_metrics = out["methods"][baseline]
    for method in methods:
        if method == baseline:
            continue
        out["paired_delta_vs_stats"][method] = {}
        for metric in metrics:
            if metric not in base_metrics or metric not in out["methods"][method]:
                continue
            delta = np.asarray(
                [f["score"][method][metric] - f["score"][baseline][metric] for f in folds],
                dtype=np.float64,
            )
            seed_delta = []
            for seed in sorted({int(f["split_seed"]) for f in folds}):
                vals = [
                    f["score"][method][metric] - f["score"][baseline][metric]
                    for f in folds
                    if int(f["split_seed"]) == seed
                ]
                seed_delta.append({"split_seed": seed, "mean_delta": float(np.mean(vals))})
            out["paired_delta_vs_stats"][method][metric] = {
                "mean": float(delta.mean()),
                "std": float(delta.std()),
                "wins": int(np.sum(delta > 1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "losses": int(np.sum(delta < -1e-12)),
                "values": delta.tolist(),
                "by_split_seed": seed_delta,
                "seed_wins": int(sum(x["mean_delta"] > 1e-12 for x in seed_delta)),
            }
    return out


def evaluate(
    graphs: list[Any],
    y: np.ndarray,
    prepared: dict[str, Any],
    *,
    n_splits: int,
    inner_splits: int,
    split_seeds: list[int],
    n_atoms: int,
    T: int,
    n_iter: int,
    max_train_patches: int,
    residual_alpha: float,
) -> dict[str, Any]:
    Ys: list[np.ndarray] = prepared["Ys"]
    channels: list[dict[str, np.ndarray]] = prepared["channels"]
    basics = np.stack([graph_basic_features(g) for g in graphs])
    degrees = np.stack([degree_hist_features(g) for g in graphs])
    stats = np.concatenate([basics, degrees], axis=1)
    raw = np.stack([Y.mean(axis=1) for Y in Ys])
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    relation_graph = np.stack([relation_graph_readout(c) for c in channels])

    folds: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y)):
            fit_seed = split_seed * 1000 + fold
            Ytr = np.concatenate([Ys[int(i)] for i in tr], axis=1)
            if Ytr.shape[1] > max_train_patches:
                rng = np.random.default_rng(fit_seed)
                keep = rng.choice(Ytr.shape[1], max_train_patches, replace=False)
                Yfit = Ytr[:, keep]
            else:
                Yfit = Ytr
            k = min(n_atoms, Yfit.shape[0], Yfit.shape[1])
            Dksvd, _, dinfo = ksvd(
                Yfit,
                n_atoms=k,
                T=min(T, k),
                T_min=min(T, 2),
                n_iter=n_iter,
                seed=fit_seed,
            )
            Drandom = random_real_dictionary(Yfit, k, fit_seed + 101)
            Dreal, _ = clustered_real_dictionary(Yfit, k, fit_seed + 211)
            pca = PCA(n_components=k, random_state=fit_seed).fit(Yfit.T)

            ksvd_feat = _encode_dictionary(Dksvd, Ys, channels, min(T, k), fit_seed + 100_003)
            random = _encode_dictionary(Drandom, Ys, channels, min(T, k), fit_seed + 200_003)
            real = _encode_dictionary(Dreal, Ys, channels, min(T, k), fit_seed + 300_003)
            pca_content, _ = _encode_pca(pca, Ys)

            additions = {
                "raw_patch_mean": raw,
                "pca_content": pca_content,
                "random_dictionary_content": random["content"],
                "real_patch_dictionary_content": real["content"],
                "ksvd_content": ksvd_feat["content"],
                "relation_graph_only": relation_graph,
                "raw_patch_mean_graph": np.concatenate([raw, relation_graph], axis=1),
                "pca_content_graph": np.concatenate([pca_content, relation_graph], axis=1),
                "random_dictionary_content_graph": random["content_graph"],
                "real_patch_dictionary_content_graph": real["content_graph"],
                "ksvd_content_graph": ksvd_feat["content_graph"],
                "ksvd_content_true_relation": ksvd_feat["content_true"],
                "ksvd_content_shuffled_relation": ksvd_feat["content_shuffled"],
            }
            blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {
                "stats_degree": (stats[tr], stats[te])
            }
            for name, feat in additions.items():
                blocks[f"stats_plus_{name}"] = (
                    np.concatenate([stats[tr], feat[tr]], axis=1),
                    np.concatenate([stats[te], feat[te]], axis=1),
                )
                rtr, rte = _residualize(stats, feat, tr, te, residual_alpha)
                blocks[f"residual_{name}"] = (rtr, rte)

            score: dict[str, Any] = {}
            selected_C: dict[str, float] = {}
            for name, (Xtr, Xte) in blocks.items():
                score[name], selected_C[name] = _fit_score_nested(
                    Xtr, y[tr], Xte, y[te], fit_seed, inner_splits
                )
            folds.append(
                {
                    "split_seed": split_seed,
                    "fold": fold,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_dictionary_patches": int(Yfit.shape[1]),
                    "ksvd": dinfo,
                    "score": score,
                    "selected_C": selected_C,
                }
            )
            print(f"  fold seed={split_seed} fold={fold} done", flush=True)
    return {
        "n_graphs": len(graphs),
        "class_counts": {str(int(c)): int(np.sum(y == c)) for c in np.unique(y)},
        "folds": folds,
        "summary": _summary(folds),
    }


def render_markdown(result: dict[str, Any]) -> str:
    c = result["config"]
    lines = [
        "# 控制 size/degree 后的 KSVD 条件增益审计（2026-07-30）",
        "",
        "> 核心问题：patch/字典特征是否提供超出 graph size、density、degree histogram、triangle、component 等简单统计的信息。所有字典/PCA 仅在 outer-train 拟合；分类器 C 仅用 outer-train 的 inner CV 选择。",
        "",
        "## 协议",
        "",
        f"- outer CV：{c['n_splits']}-fold × seeds {c['split_seeds']}；inner CV：{c['inner_splits']}-fold。",
        f"- atoms={c['n_atoms']}，T={c['T']}，KSVD iterations={c['n_iter']}，C grid={c['C_grid']}。",
        f"- residual 特征：先用 outer-train 上的 Ridge(alpha={c['residual_alpha']}) 从 patch block 中线性回归掉 stats，再分类。",
        "- `stats_plus_*` 是主要条件增益判断；`residual_*` 是辅助诊断，不应单独替代 paired delta。",
        "",
    ]
    for tag, block in result["experiments"].items():
        s = block["summary"]
        lines += [f"## {tag}", "", f"图数：{block['n_graphs']}；类别：`{block['class_counts']}`。", ""]
        metric = "balanced_accuracy"
        base = s["methods"]["stats_degree"][metric]["mean"]
        lines += [
            f"stats+degree baseline balanced accuracy：**{base:.4f}**。",
            "",
            "| 新增表征 | stats+feature | paired Δ vs stats | fold W/T/L | seed wins | residual-only |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        preferred = [
            "raw_patch_mean",
            "pca_content",
            "random_dictionary_content",
            "real_patch_dictionary_content",
            "ksvd_content",
            "relation_graph_only",
            "raw_patch_mean_graph",
            "pca_content_graph",
            "random_dictionary_content_graph",
            "real_patch_dictionary_content_graph",
            "ksvd_content_graph",
            "ksvd_content_true_relation",
            "ksvd_content_shuffled_relation",
        ]
        for name in preferred:
            plus = f"stats_plus_{name}"
            residual = f"residual_{name}"
            if plus not in s["methods"]:
                continue
            d = s["paired_delta_vs_stats"][plus][metric]
            wtl = f"{d['wins']}/{d['ties']}/{d['losses']}"
            lines.append(
                f"| {name} | {s['methods'][plus][metric]['mean']:.4f} | {d['mean']:+.4f} | {wtl} | {d['seed_wins']}/{len(d['by_split_seed'])} | {s['methods'][residual][metric]['mean']:.4f} |"
            )
        true_key = "stats_plus_ksvd_content_true_relation"
        shuf_key = "stats_plus_ksvd_content_shuffled_relation"
        true_vals = np.asarray(s["methods"][true_key][metric]["values"])
        shuf_vals = np.asarray(s["methods"][shuf_key][metric]["values"])
        td = true_vals - shuf_vals
        values = lambda key: np.asarray(s["methods"][key][metric]["values"])
        comparisons = [
            ("KSVD content − PCA", values("stats_plus_ksvd_content") - values("stats_plus_pca_content")),
            ("KSVD content − real-patch dictionary", values("stats_plus_ksvd_content") - values("stats_plus_real_patch_dictionary_content")),
            ("KSVD content+graph − raw+same graph", values("stats_plus_ksvd_content_graph") - values("stats_plus_raw_patch_mean_graph")),
            ("KSVD content+graph − PCA+same graph", values("stats_plus_ksvd_content_graph") - values("stats_plus_pca_content_graph")),
            ("KSVD content+graph − real-patch+same graph", values("stats_plus_ksvd_content_graph") - values("stats_plus_real_patch_dictionary_content_graph")),
            ("true atom-position relation − content+graph", true_vals - values("stats_plus_ksvd_content_graph")),
        ]
        lines += [
            "",
            f"- true relation − shuffled relation：{td.mean():+.4f}，fold wins={int(np.sum(td > 1e-12))}/{len(td)}。",
        ]
        for label, delta in comparisons:
            seed_means = [float(delta[i * c["n_splits"] : (i + 1) * c["n_splits"]].mean()) for i in range(len(c["split_seeds"]))]
            lines.append(
                f"- {label}：{delta.mean():+.4f}，fold wins={int(np.sum(delta > 1e-12))}/{len(delta)}，seed means={[round(x, 4) for x in seed_means]}。"
            )
        lines += [
            "- 解释时优先看 paired delta 是否跨 split seeds 同方向，并要求 KSVD 同时优于 raw/PCA/real-patch，而不是只优于 stats。",
            "",
        ]
    lines += [
        "## 判断口径",
        "",
        "1. `stats_plus_*` 相对 stats 的正增益，只能说明该 patch block 含额外信息；若 raw/PCA/real-patch 同样或更强，不能归因于 KSVD。",
        "2. `true relation > shuffled relation` 只是必要条件；还必须比较 `true relation` 与较低容量的 `content+graph`，否则可能只是 shuffle 更坏，而不是 exact binding 真正有益。",
        "3. residual-only 接近随机不否定联合条件信息，但若 stats+feature 也无稳定增益，则该表征没有通过当前 conditional test。",
        "4. 重复 CV folds 不是独立样本；结论以 split-seed 方向一致性和跨数据集复现为主，不把 fold 数当显著性样本量。",
        "",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["IMDB-BINARY", "IMDB-MULTI"])
    ap.add_argument("--variants", nargs="+", choices=["raw", "cleaned"], default=["cleaned"])
    ap.add_argument("--methods", nargs="+", choices=["B0", "R2", "UniformRW", "CoverageRW", "Path", "PPR"], default=["B0"])
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sampling-seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--n-patches", type=int, default=16)
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--walk-length", type=int, default=24)
    ap.add_argument("--n-atoms", type=int, default=16)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--max-train-patches", type=int, default=4000)
    ap.add_argument("--residual-alpha", type=float, default=1.0)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MatchedSamplingConfig(
        n_patches=args.n_patches,
        max_nodes=args.max_nodes,
        walk_length=args.walk_length,
        seed=args.sampling_seed,
    )
    result: dict[str, Any] = {
        "config": {
            "datasets": args.datasets,
            "variants": args.variants,
            "methods": args.methods,
            "split_seeds": args.split_seeds,
            "sampling_seed": args.sampling_seed,
            "n_splits": args.n_splits,
            "inner_splits": args.inner_splits,
            "n_patches": args.n_patches,
            "max_nodes": args.max_nodes,
            "walk_length": args.walk_length,
            "n_atoms": args.n_atoms,
            "T": args.T,
            "n_iter": args.n_iter,
            "max_train_patches": args.max_train_patches,
            "residual_alpha": args.residual_alpha,
            "limit_per_class": args.limit_per_class,
            "C_grid": list(C_GRID),
        },
        "experiments": {},
    }
    for dataset in args.datasets:
        for variant in args.variants:
            cleaned = variant == "cleaned"
            graphs, y, _node_feats, meta = load_tud(dataset, cleaned=cleaned, structure_only=True)
            graphs, y = _subset_per_class(graphs, y, args.limit_per_class, seed=0)
            for method in args.methods:
                tag = f"{dataset}/{variant}/{method}"
                print(f"[conditional] {tag}: n={len(graphs)}", flush=True)
                prepared = prepare_graphs(
                    dataset,
                    cleaned,
                    graphs,
                    method,
                    cfg,
                    args.sampling_seed,
                    args.limit_per_class,
                    use_cache=not args.no_cache,
                )
                block = evaluate(
                    graphs,
                    y,
                    prepared,
                    n_splits=args.n_splits,
                    inner_splits=args.inner_splits,
                    split_seeds=args.split_seeds,
                    n_atoms=args.n_atoms,
                    T=args.T,
                    n_iter=args.n_iter,
                    max_train_patches=args.max_train_patches,
                    residual_alpha=args.residual_alpha,
                )
                block["dataset_meta"] = meta
                result["experiments"][tag] = block
                _atomic_json(args.output_json, result)
                args.output_md.parent.mkdir(parents=True, exist_ok=True)
                args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
