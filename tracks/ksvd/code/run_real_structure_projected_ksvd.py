"""Bounded go/no-go experiment for empirically projected K-SVD.

This runner compares three legal/shared vocabulary mechanisms under the same
outer-fold protocol:

1. clustered real-patch medoids;
2. ordinary K-SVD followed by one joint projection to distinct real patches;
3. K-SVD projected to distinct real patches after every dictionary update.

Raw patch means, PCA, unconstrained K-SVD, graph statistics, and the common
patch-relation topology block are retained as controls.  Every unsupervised
transform is fitted on the outer-training fold only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .ksvd import ksvd
from .projected_ksvd import iterative_projected_ksvd, project_dictionary_to_real_patches
from .run_real_structure_conditional import _fit_score_nested, _residualize, _summary
from .run_real_structure_ksvd import (
    _atomic_json,
    _encode_dictionary,
    _encode_pca,
    _subset_per_class,
    clustered_real_dictionary,
    degree_hist_features,
    graph_basic_features,
    prepare_graphs,
)
from .sampling_route import MatchedSamplingConfig, relation_graph_readout, sparse_code_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "real_structure_projected_ksvd_20260730.json"
DEFAULT_MD = ROOT / "docs" / "REAL_STRUCTURE_PROJECTED_KSVD_20260730.md"

SPECS = {
    "IMDB-BINARY": {"dataset": "IMDB-BINARY", "variant": "cleaned", "method": "B0", "max_nodes": 12},
    "IMDB-MULTI": {"dataset": "IMDB-MULTI", "variant": "cleaned", "method": "B0", "max_nodes": 12},
    "REDDIT-BINARY": {"dataset": "REDDIT-BINARY", "variant": "raw", "method": "R2", "max_nodes": 24},
}


def _frobenius_reconstruction(D: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.linalg.norm(Y - D @ X, "fro") / max(np.linalg.norm(Y, "fro"), 1e-12))


def _seed_means(values: np.ndarray, n_splits: int, split_seeds: list[int]) -> list[float]:
    return [
        float(values[i * n_splits : (i + 1) * n_splits].mean())
        for i in range(len(split_seeds))
    ]


def _paired(values_a: np.ndarray, values_b: np.ndarray, n_splits: int, split_seeds: list[int]) -> dict[str, Any]:
    delta = values_a - values_b
    seed_means = _seed_means(delta, n_splits, split_seeds)
    return {
        "mean": float(delta.mean()),
        "std": float(delta.std()),
        "wins": int(np.sum(delta > 1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
        "seed_means": seed_means,
        "seed_wins": int(np.sum(np.asarray(seed_means) > 1e-12)),
    }


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
    stats = np.concatenate(
        [
            np.stack([graph_basic_features(g) for g in graphs]),
            np.stack([degree_hist_features(g) for g in graphs]),
        ],
        axis=1,
    )
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
            T_eff = min(T, k)

            Dfree, Xfree, free_info = ksvd(
                Yfit,
                n_atoms=k,
                T=T_eff,
                T_min=min(T_eff, 2),
                n_iter=n_iter,
                seed=fit_seed,
            )
            Dfinal, final_projection = project_dictionary_to_real_patches(Dfree, Yfit)
            Diter, Xiter, iterative_info = iterative_projected_ksvd(
                Yfit,
                n_atoms=k,
                T=T_eff,
                T_min=min(T_eff, 2),
                n_iter=n_iter,
                seed=fit_seed,
            )
            Dreal, real_indices = clustered_real_dictionary(Yfit, k, fit_seed + 211)
            pca = PCA(n_components=k, random_state=fit_seed).fit(Yfit.T)

            free = _encode_dictionary(Dfree, Ys, channels, T_eff, fit_seed + 100_003)
            final = _encode_dictionary(Dfinal, Ys, channels, T_eff, fit_seed + 200_003)
            iterative = _encode_dictionary(Diter, Ys, channels, T_eff, fit_seed + 300_003)
            real = _encode_dictionary(Dreal, Ys, channels, T_eff, fit_seed + 400_003)
            pca_content, _ = _encode_pca(pca, Ys)

            additions = {
                "raw_patch_mean": raw,
                "pca_content": pca_content,
                "real_patch_dictionary_content": real["content"],
                "ksvd_content": free["content"],
                "final_projected_content": final["content"],
                "iterative_projected_content": iterative["content"],
                "relation_graph_only": relation_graph,
                "raw_patch_mean_graph": np.concatenate([raw, relation_graph], axis=1),
                "pca_content_graph": np.concatenate([pca_content, relation_graph], axis=1),
                "real_patch_dictionary_content_graph": real["content_graph"],
                "ksvd_content_graph": free["content_graph"],
                "final_projected_content_graph": final["content_graph"],
                "iterative_projected_content_graph": iterative["content_graph"],
            }
            blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {
                "stats_degree": (stats[tr], stats[te])
            }
            for name, feature in additions.items():
                blocks[f"stats_plus_{name}"] = (
                    np.concatenate([stats[tr], feature[tr]], axis=1),
                    np.concatenate([stats[te], feature[te]], axis=1),
                )
                blocks[f"residual_{name}"] = _residualize(stats, feature, tr, te, residual_alpha)

            score: dict[str, Any] = {}
            selected_C: dict[str, float] = {}
            for name, (Xtr, Xte) in blocks.items():
                score[name], selected_C[name] = _fit_score_nested(
                    Xtr, y[tr], Xte, y[te], fit_seed, inner_splits
                )

            # Fit-matrix reconstruction is diagnostic only; classification is
            # the pre-registered promotion criterion.
            Xfinal, _final_patch_errors = sparse_code_matrix(Dfinal, Yfit, T_eff)
            folds.append(
                {
                    "split_seed": split_seed,
                    "fold": fold,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_dictionary_patches": int(Yfit.shape[1]),
                    "dictionary": {
                        "unconstrained_ksvd": free_info,
                        "clustered_real_patch": {
                            "projectability": 1.0,
                            "selected_training_indices": [int(i) for i in real_indices],
                        },
                        "final_projected_ksvd": {
                            **final_projection,
                            "recon_rel": _frobenius_reconstruction(Dfinal, Xfinal, Yfit),
                            "unconstrained_recon_rel": _frobenius_reconstruction(Dfree, Xfree, Yfit),
                        },
                        "iterative_projected_ksvd": iterative_info,
                    },
                    "score": score,
                    "selected_C": selected_C,
                }
            )
            print(f"  fold seed={split_seed} fold={fold} done", flush=True)

    summary = _summary(folds)
    metric = "balanced_accuracy"
    vals = lambda name: np.asarray(summary["methods"][f"stats_plus_{name}"][metric]["values"])
    comparisons = {
        "final_projected_minus_real": _paired(vals("final_projected_content"), vals("real_patch_dictionary_content"), n_splits, split_seeds),
        "iterative_projected_minus_real": _paired(vals("iterative_projected_content"), vals("real_patch_dictionary_content"), n_splits, split_seeds),
        "final_projected_minus_raw": _paired(vals("final_projected_content"), vals("raw_patch_mean"), n_splits, split_seeds),
        "iterative_projected_minus_raw": _paired(vals("iterative_projected_content"), vals("raw_patch_mean"), n_splits, split_seeds),
        "final_projected_minus_pca": _paired(vals("final_projected_content"), vals("pca_content"), n_splits, split_seeds),
        "iterative_projected_minus_pca": _paired(vals("iterative_projected_content"), vals("pca_content"), n_splits, split_seeds),
        "final_projected_minus_unconstrained": _paired(vals("final_projected_content"), vals("ksvd_content"), n_splits, split_seeds),
        "iterative_projected_minus_unconstrained": _paired(vals("iterative_projected_content"), vals("ksvd_content"), n_splits, split_seeds),
        "final_projected_graph_minus_real_graph": _paired(vals("final_projected_content_graph"), vals("real_patch_dictionary_content_graph"), n_splits, split_seeds),
        "iterative_projected_graph_minus_real_graph": _paired(vals("iterative_projected_content_graph"), vals("real_patch_dictionary_content_graph"), n_splits, split_seeds),
        "final_projected_graph_minus_relation_graph": _paired(vals("final_projected_content_graph"), vals("relation_graph_only"), n_splits, split_seeds),
        "iterative_projected_graph_minus_relation_graph": _paired(vals("iterative_projected_content_graph"), vals("relation_graph_only"), n_splits, split_seeds),
    }
    return {
        "n_graphs": len(graphs),
        "class_counts": {str(int(c)): int(np.sum(y == c)) for c in np.unique(y)},
        "folds": folds,
        "summary": summary,
        "comparisons": comparisons,
    }


def render_markdown(result: dict[str, Any]) -> str:
    c = result["config"]
    lines = [
        "# Projected KSVD 有界 Go/No-Go 实验（2026-07-30）",
        "",
        "> 问题：把自由 KSVD atom 约束为训练折中的 distinct real patches 后，能否获得稳定、跨数据集、且不只是重构更好的任务增益？",
        "",
        "## 协议",
        "",
        f"- outer CV：{c['n_splits']}-fold × split seeds {c['split_seeds']}；inner CV：{c['inner_splits']}-fold 选择 logistic C。",
        f"- atoms={c['n_atoms']}，T={c['T']}，iterations={c['n_iter']}，每折最多 {c['max_train_patches']} 个训练 patches。",
        "- 所有 PCA/字典只在 outer-train 拟合；所有图均 structure-only。",
        "- `final-projected`：普通 KSVD 完成后做一次 Hungarian joint projection。",
        "- `iterative-projected`：每次 dictionary update 后投影，并重新 OMP。",
        "- projection 使用 absolute cosine，但输出 atom 本身始终是 normalized real training patch；同时要求 vector signature distinct。",
        "- 晋级标准：projectability=1，且至少两个结构差异明显的数据集上，跨 3 split seeds 稳定优于 clustered-real-patch、PCA、raw，并在控制 stats 后保留增益。",
        "",
    ]
    preferred = [
        "raw_patch_mean", "pca_content", "real_patch_dictionary_content", "ksvd_content",
        "final_projected_content", "iterative_projected_content", "relation_graph_only",
        "real_patch_dictionary_content_graph", "ksvd_content_graph",
        "final_projected_content_graph", "iterative_projected_content_graph",
    ]
    labels = {
        "raw_patch_mean": "raw mean", "pca_content": "PCA", "real_patch_dictionary_content": "clustered real-patch",
        "ksvd_content": "unconstrained KSVD", "final_projected_content": "final-projected KSVD",
        "iterative_projected_content": "iterative-projected KSVD", "relation_graph_only": "relation graph only",
        "real_patch_dictionary_content_graph": "real-patch + graph", "ksvd_content_graph": "KSVD + graph",
        "final_projected_content_graph": "final-projected + graph", "iterative_projected_content_graph": "iterative-projected + graph",
    }
    for tag, block in result["experiments"].items():
        s = block["summary"]
        metric = "balanced_accuracy"
        lines += [f"## {tag}", "", f"图数：{block['n_graphs']}；类别：`{block['class_counts']}`。", ""]
        lines += [
            "| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in preferred:
            key = f"stats_plus_{name}"
            d = s["paired_delta_vs_stats"][key][metric]
            lines.append(
                f"| {labels[name]} | {s['methods'][key][metric]['mean']:.4f} | {d['mean']:+.4f} | {d['wins']}/{d['ties']}/{d['losses']} | {d['seed_wins']}/{len(d['by_split_seed'])} |"
            )
        lines += ["", "关键 paired comparisons（均已包含 stats）：", ""]
        for name, comp in block["comparisons"].items():
            lines.append(
                f"- `{name}`：{comp['mean']:+.4f}，fold W/T/L={comp['wins']}/{comp['ties']}/{comp['losses']}，seed means={[round(x, 4) for x in comp['seed_means']]}。"
            )
        dictionary_rows = [
            ("unconstrained", "unconstrained_ksvd"),
            ("final-projected", "final_projected_ksvd"),
            ("iterative-projected", "iterative_projected_ksvd"),
        ]
        lines += ["", "字典训练折诊断（所有 outer folds 的 mean）：", "", "| 字典 | projectability | reconstruction |", "|---|---:|---:|"]
        for label, key in dictionary_rows:
            infos = [fold["dictionary"][key] for fold in block["folds"]]
            projectability = np.mean([float(info.get("projectability", np.nan)) for info in infos])
            recon = np.mean([float(info["recon_rel"]) for info in infos])
            lines.append(f"| {label} | {projectability:.3f} | {recon:.4f} |")
        lines.append("")

    # Mechanical pre-registration check.  A method passes a dataset only when
    # its content block beats all three baselines in every split-seed mean.
    pass_map: dict[str, list[str]] = {"final_projected": [], "iterative_projected": []}
    for tag, block in result["experiments"].items():
        for method in pass_map:
            keys = [f"{method}_minus_real", f"{method}_minus_raw", f"{method}_minus_pca"]
            if all(block["comparisons"][key]["seed_wins"] == len(c["split_seeds"]) for key in keys):
                pass_map[method].append(tag)
    lines += ["## 预注册门槛判定", ""]
    for method, passed in pass_map.items():
        lines.append(f"- `{method}` 在全部三个 baseline 上均 3/3 seed 正向的数据集：{passed or '无'}。")
    passed_any = any(len(v) >= 2 for v in pass_map.values())
    lines += [
        "",
        f"**最终判定：{'GO' if passed_any else 'NO-GO'}。**",
        "",
        "注意：重构改善不能覆盖分类门槛失败；relation graph 的结果也不能归因于 KSVD vocabulary。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", nargs="+", choices=sorted(SPECS), default=list(SPECS))
    ap.add_argument("--feature-mode", choices=["wl", "canonical", "rooted_canonical"], default="wl")
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sampling-seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--n-patches", type=int, default=16)
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
    result: dict[str, Any] = {
        "config": {
            "settings": args.settings,
            "feature_mode": args.feature_mode,
            "split_seeds": args.split_seeds,
            "sampling_seed": args.sampling_seed,
            "n_splits": args.n_splits,
            "inner_splits": args.inner_splits,
            "n_patches": args.n_patches,
            "walk_length": args.walk_length,
            "n_atoms": args.n_atoms,
            "T": args.T,
            "n_iter": args.n_iter,
            "max_train_patches": args.max_train_patches,
            "residual_alpha": args.residual_alpha,
            "limit_per_class": args.limit_per_class,
        },
        "experiments": {},
    }
    for setting in args.settings:
        spec = SPECS[setting]
        cleaned = spec["variant"] == "cleaned"
        graphs, y, _features, meta = load_tud(spec["dataset"], cleaned=cleaned, structure_only=True)
        graphs, y = _subset_per_class(graphs, y, args.limit_per_class, seed=0)
        cfg = MatchedSamplingConfig(
            n_patches=args.n_patches,
            max_nodes=spec["max_nodes"],
            walk_length=args.walk_length,
            seed=args.sampling_seed,
        )
        tag = f"{spec['dataset']}/{spec['variant']}/{spec['method']}/max_nodes={spec['max_nodes']}/{args.feature_mode}"
        print(f"[projected-ksvd] {tag}: n={len(graphs)}", flush=True)
        prepared = prepare_graphs(
            spec["dataset"], cleaned, graphs, spec["method"], cfg,
            args.sampling_seed, args.limit_per_class, use_cache=not args.no_cache,
        )
        if args.feature_mode != "wl":
            from .run_real_structure_nonwl_audit import build_representations
            nonwl = build_representations(
                setting, graphs, prepared, cfg,
                sampling_seed=args.sampling_seed,
                limit_per_class=args.limit_per_class,
                use_cache=not args.no_cache,
            )
            prepared = {**prepared, "Ys": nonwl["Ys"][args.feature_mode]}
        block = evaluate(
            graphs, y, prepared,
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
