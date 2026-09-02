"""Real, structure-only TU benchmark for the KSVD research question.

This runner is intentionally a mechanism audit rather than a leaderboard script.
It uses real social-network graph datasets without node/edge attributes and asks:

1. Does train-fold KSVD improve over the raw permutation-invariant patch mean?
2. Does it improve over equally sized PCA, random-column, and real-patch dictionaries?
3. Does patch identity-to-position alignment beat a shuffled alignment?
4. Do conclusions survive removal of isomorphic duplicate graphs (cleaned TU data)?

All dictionaries and PCA bases are fitted inside the training fold.

Example:
  cd tracks/ksvd
  PY=/home/calendar/.conda/envs/gsn-official/bin/python
  $PY -m code.run_real_structure_ksvd \
      --datasets IMDB-BINARY IMDB-MULTI --variants raw cleaned \
      --methods B0 R2 --split-seeds 0 1 2
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud
from .graph import Graph
from .ksvd import ksvd
from .sampling_route import (
    MatchedSamplingConfig,
    PatchCollection,
    content_readout,
    relation_channels,
    relation_graph_readout,
    relation_readout,
    sample_matched,
    sparse_code_matrix,
    vectorize_collection,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "real_structure_ksvd_20260730.json"
DEFAULT_MD = ROOT / "docs" / "REAL_STRUCTURE_KSVD_20260730.md"
CACHE_DIR = ROOT / "results" / "cache_real_structure_20260730"


def _atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _components(g: Graph) -> tuple[int, int]:
    seen: set[int] = set()
    sizes: list[int] = []
    for start in g.nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in g.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        sizes.append(size)
    return len(sizes), max(sizes, default=0)


def graph_basic_features(g: Graph) -> np.ndarray:
    """Low-capacity controls that expose size/degree/triangle shortcuts."""
    n = g.n
    m = g.num_edges()
    deg = np.asarray([len(g.neighbors(u)) for u in g.nodes], dtype=np.float64)
    if deg.size == 0:
        deg = np.zeros(1, dtype=np.float64)
    comp, largest = _components(g)
    triangles3 = 0
    for u, v in g.edges():
        triangles3 += len(g.neighbors(u) & g.neighbors(v))
    triangles = triangles3 / 3.0
    wedges = float(np.sum(deg * np.maximum(deg - 1.0, 0.0) / 2.0))
    transitivity = 3.0 * triangles / max(wedges, 1.0)
    density = 2.0 * m / max(n * (n - 1), 1)
    cycle_rank = m - n + comp
    return np.asarray(
        [
            np.log1p(n),
            np.log1p(m),
            density,
            deg.mean(),
            deg.std(),
            deg.max(),
            *np.quantile(deg, [0.25, 0.5, 0.75]).tolist(),
            float(np.mean(deg == 0)),
            float(comp),
            largest / max(n, 1),
            np.log1p(triangles),
            transitivity,
            float(cycle_rank) / max(n, 1),
        ],
        dtype=np.float64,
    )


def degree_hist_features(g: Graph, max_degree_bin: int = 20) -> np.ndarray:
    deg = np.asarray([len(g.neighbors(u)) for u in g.nodes], dtype=np.int64)
    clipped = np.minimum(deg, max_degree_bin)
    hist = np.bincount(clipped, minlength=max_degree_bin + 1).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    return hist


def _subset_per_class(
    graphs: list[Graph], y: np.ndarray, limit_per_class: int | None, seed: int
) -> tuple[list[Graph], np.ndarray]:
    if limit_per_class is None:
        return graphs, y
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        keep.extend(int(i) for i in idx[:limit_per_class])
    keep = sorted(keep)
    return [graphs[i] for i in keep], y[np.asarray(keep, dtype=np.int64)]


def _cache_path(
    dataset: str,
    cleaned: bool,
    method: str,
    cfg: MatchedSamplingConfig,
    sampling_seed: int,
    limit_per_class: int | None,
) -> Path:
    payload = {
        "dataset": dataset,
        "cleaned": cleaned,
        "method": method,
        "cfg": asdict(cfg),
        "sampling_seed": sampling_seed,
        "limit_per_class": limit_per_class,
        "feature": "wl",
        "center_policy": "all_if_small_else_uniform_without_replacement",
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    variant = "cleaned" if cleaned else "raw"
    return CACHE_DIR / f"{dataset}_{variant}_{method}_s{sampling_seed}_{key}.pkl"


def prepare_graphs(
    dataset: str,
    cleaned: bool,
    graphs: list[Graph],
    method: str,
    cfg: MatchedSamplingConfig,
    sampling_seed: int,
    limit_per_class: int | None,
    use_cache: bool,
) -> dict[str, Any]:
    path = _cache_path(dataset, cleaned, method, cfg, sampling_seed, limit_per_class)
    if use_cache and path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    pcs: list[PatchCollection] = []
    ys: list[np.ndarray] = []
    channels: list[dict[str, np.ndarray]] = []
    for i, g in enumerate(graphs):
        # Do not duplicate centers in small graphs. This differs deliberately
        # from the historical matched-sampler probe where patch count was fixed.
        graph_cfg = replace(
            cfg,
            n_patches=min(cfg.n_patches, max(g.n, 1)),
            seed=sampling_seed * 1_000_003 + i * 9_973 + 17,
        )
        pc = sample_matched(g, method, graph_cfg)
        Y = vectorize_collection(g, pc, graph_cfg.max_nodes, "wl", None)
        pcs.append(pc)
        ys.append(Y)
        channels.append(relation_channels(g, pc))

    obj = {"pcs": pcs, "Ys": ys, "channels": channels}
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj


def _normalize_columns(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64).copy()
    denom = np.linalg.norm(D, axis=0, keepdims=True)
    denom[denom < 1e-12] = 1.0
    return D / denom


def random_real_dictionary(Y: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(Y.shape[1], size=n_atoms, replace=Y.shape[1] < n_atoms)
    return _normalize_columns(Y[:, idx])


def clustered_real_dictionary(Y: np.ndarray, n_atoms: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Learn clusters, but keep each atom equal to an actual training patch."""
    km = MiniBatchKMeans(
        n_clusters=n_atoms,
        random_state=seed,
        batch_size=min(1024, max(64, Y.shape[1])),
        n_init=5,
        max_iter=200,
    )
    samples = Y.T
    km.fit(samples)
    chosen: list[int] = []
    available = np.ones(samples.shape[0], dtype=bool)
    for center in km.cluster_centers_:
        dist = np.sum((samples - center[None, :]) ** 2, axis=1)
        masked = np.where(available, dist, np.inf)
        j = int(np.argmin(masked))
        if not np.isfinite(masked[j]):
            j = int(np.argmin(dist))
        chosen.append(j)
        available[j] = False
    return _normalize_columns(Y[:, chosen]), chosen


def _encode_dictionary(
    D: np.ndarray,
    Ys: list[np.ndarray],
    channels: list[dict[str, np.ndarray]],
    T: int,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    content: list[np.ndarray] = []
    content_graph: list[np.ndarray] = []
    content_true: list[np.ndarray] = []
    content_shuffled: list[np.ndarray] = []
    reconstruction: list[float] = []
    for i, Y in enumerate(Ys):
        X, errors = sparse_code_matrix(D, Y, T)
        c = content_readout(X, errors)
        rg = relation_graph_readout(channels[i])
        rt = relation_readout(X, channels[i], include_topology=False)
        rng = np.random.default_rng(shuffle_seed + i * 997)
        rs = relation_readout(
            X,
            channels[i],
            assignment_permutation=rng.permutation(X.shape[1]),
            include_topology=False,
        )
        content.append(c)
        content_graph.append(np.concatenate([c, rg]))
        content_true.append(np.concatenate([c, rg, rt]))
        content_shuffled.append(np.concatenate([c, rg, rs]))
        reconstruction.append(float(errors.mean()))
    return {
        "content": np.stack(content),
        "content_graph": np.stack(content_graph),
        "content_true": np.stack(content_true),
        "content_shuffled": np.stack(content_shuffled),
        "reconstruction": np.asarray(reconstruction, dtype=np.float64),
    }


def _encode_pca(
    pca: PCA,
    Ys: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    feats: list[np.ndarray] = []
    recs: list[float] = []
    for Y in Ys:
        samples = Y.T
        Z = pca.transform(samples)
        R = samples - pca.inverse_transform(Z)
        denom = np.maximum(np.linalg.norm(samples, axis=1), 1e-12)
        errors = np.linalg.norm(R, axis=1) / denom
        feats.append(content_readout(Z.T, errors))
        recs.append(float(errors.mean()))
    return np.stack(feats), np.asarray(recs, dtype=np.float64)


def _fit_score(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int,
) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, random_state=seed, solver="lbfgs"),
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)
    out = {
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
    }
    try:
        if prob.shape[1] == 2:
            out["roc_auc"] = float(roc_auc_score(yte, prob[:, 1]))
        else:
            out["roc_auc_ovr_weighted"] = float(
                roc_auc_score(yte, prob, multi_class="ovr", average="weighted")
            )
    except ValueError:
        pass
    return out


def matched_dictionary_cosine(A: np.ndarray, B: np.ndarray) -> float:
    S = np.abs(_normalize_columns(A).T @ _normalize_columns(B))
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(folds[0]["score"])
    summary: dict[str, Any] = {}
    for method in methods:
        metrics = sorted(folds[0]["score"][method])
        summary[method] = {}
        for metric in metrics:
            vals = np.asarray([f["score"][method][metric] for f in folds], dtype=np.float64)
            summary[method][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "values": vals.tolist(),
            }
    for key in ["ksvd_test_reconstruction", "pca_test_reconstruction", "real_patch_test_reconstruction"]:
        vals = np.asarray([f[key] for f in folds], dtype=np.float64)
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return summary


def evaluate(
    graphs: list[Graph],
    y: np.ndarray,
    prepared: dict[str, Any],
    n_splits: int,
    split_seeds: list[int],
    n_atoms: int,
    T: int,
    n_iter: int,
    max_train_patches: int,
) -> dict[str, Any]:
    Ys: list[np.ndarray] = prepared["Ys"]
    channels: list[dict[str, np.ndarray]] = prepared["channels"]
    basics = np.stack([graph_basic_features(g) for g in graphs])
    degrees = np.stack([degree_hist_features(g) for g in graphs])
    stats_degree = np.concatenate([basics, degrees], axis=1)
    raw_mean = np.stack([Y.mean(axis=1) for Y in Ys])
    raw_mean /= np.maximum(np.linalg.norm(raw_mean, axis=1, keepdims=True), 1e-12)
    relation_graph = np.stack([relation_graph_readout(c) for c in channels])

    folds: list[dict[str, Any]] = []
    learned_dicts: list[np.ndarray] = []
    real_dicts: list[np.ndarray] = []
    for split_seed in split_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
            Ytr = np.concatenate([Ys[int(i)] for i in tr], axis=1)
            fit_seed = split_seed * 1000 + fold
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
            Dreal, real_idx = clustered_real_dictionary(Yfit, k, fit_seed + 211)
            pca = PCA(n_components=k, random_state=fit_seed)
            pca.fit(Yfit.T)

            learned_dicts.append(Dksvd)
            real_dicts.append(Dreal)
            ksvd_feat = _encode_dictionary(Dksvd, Ys, channels, min(T, k), fit_seed + 100_003)
            random_feat = _encode_dictionary(Drandom, Ys, channels, min(T, k), fit_seed + 200_003)
            real_feat = _encode_dictionary(Dreal, Ys, channels, min(T, k), fit_seed + 300_003)
            pca_content, pca_recon = _encode_pca(pca, Ys)

            feature_blocks = {
                "size_degree_stats": basics,
                "degree_hist": degrees,
                "stats_degree": stats_degree,
                "raw_patch_mean": raw_mean,
                "relation_graph_only": relation_graph,
                "pca_content": pca_content,
                "random_dictionary_content": random_feat["content"],
                "real_patch_dictionary_content": real_feat["content"],
                "ksvd_content": ksvd_feat["content"],
                "ksvd_content_graph": ksvd_feat["content_graph"],
                "ksvd_content_true_relation": ksvd_feat["content_true"],
                "ksvd_content_shuffled_relation": ksvd_feat["content_shuffled"],
            }
            score = {
                name: _fit_score(X[tr], y[tr], X[te], y[te], fit_seed)
                for name, X in feature_blocks.items()
            }
            folds.append(
                {
                    "split_seed": split_seed,
                    "fold": fold,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_dictionary_patches": int(Yfit.shape[1]),
                    "ksvd": dinfo,
                    "real_patch_indices_in_fit_matrix": real_idx,
                    "ksvd_test_reconstruction": float(ksvd_feat["reconstruction"][te].mean()),
                    "pca_test_reconstruction": float(pca_recon[te].mean()),
                    "real_patch_test_reconstruction": float(real_feat["reconstruction"][te].mean()),
                    "score": score,
                }
            )

    def stability(dicts: list[np.ndarray]) -> dict[str, Any]:
        vals = [
            matched_dictionary_cosine(dicts[i], dicts[j])
            for i in range(len(dicts))
            for j in range(i + 1, len(dicts))
        ]
        return {
            "mean_matched_absolute_cosine": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
            "n_pairs": len(vals),
        }

    return {
        "n_graphs": len(graphs),
        "class_counts": {str(int(c)): int(np.sum(y == c)) for c in np.unique(y)},
        "folds": folds,
        "summary": summarize_folds(folds),
        "dictionary_stability": {
            "ksvd": stability(learned_dicts),
            "clustered_real_patch": stability(real_dicts),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    cfg = result["config"]
    lines = [
        "# 真实纯结构数据上的 KSVD 前置审计（2026-07-30）",
        "",
        "> 目标不是追 TUData leaderboard，而是在无节点/边属性的真实图上，严格区分简单统计、原始 patch 表示、普通降维、真实 patch 字典与 KSVD。所有数据依赖变换只在训练折拟合。",
        "",
        "## 协议",
        "",
        f"- 数据集：`{', '.join(cfg['datasets'])}`；版本：`{', '.join(cfg['variants'])}`。",
        f"- patch：`{', '.join(cfg['methods'])}`；每图最多 {cfg['n_patches']} 个中心，每个 patch 最多 {cfg['max_nodes']} 个节点。",
        f"- 字典：{cfg['n_atoms']} atoms，稀疏度 T={cfg['T']}，KSVD iterations={cfg['n_iter']}。",
        f"- 外层评估：{cfg['n_splits']}-fold × split seeds {cfg['split_seeds']}。",
        "- `raw` 与 `cleaned` 同时报告，用于检查同构重复图造成的结果膨胀。",
        "- `real_patch_dictionary` 的每个 atom 是真实训练 patch；KSVD atom 仍是 WL 特征空间中的欧氏向量，不自动等价于合法图。",
        "",
    ]
    for key, block in result["experiments"].items():
        lines += [f"## {key}", "", f"图数：{block['n_graphs']}；类别：`{block['class_counts']}`。", ""]
        summary = block["summary"]
        metric = "balanced_accuracy"
        lines += ["| 表征 | Balanced Acc | Accuracy |", "|---|---:|---:|"]
        order = [
            "size_degree_stats",
            "degree_hist",
            "stats_degree",
            "relation_graph_only",
            "raw_patch_mean",
            "pca_content",
            "random_dictionary_content",
            "real_patch_dictionary_content",
            "ksvd_content",
            "ksvd_content_graph",
            "ksvd_content_true_relation",
            "ksvd_content_shuffled_relation",
        ]
        for name in order:
            if name not in summary:
                continue
            b = summary[name]
            ba = b[metric]
            ac = b["accuracy"]
            lines.append(f"| {name} | {ba['mean']:.4f} ± {ba['std']:.4f} | {ac['mean']:.4f} ± {ac['std']:.4f} |")
        lines += [
            "",
            "重构误差（test graph mean）：",
            "",
            f"- KSVD：{summary['ksvd_test_reconstruction']['mean']:.4f} ± {summary['ksvd_test_reconstruction']['std']:.4f}",
            f"- PCA：{summary['pca_test_reconstruction']['mean']:.4f} ± {summary['pca_test_reconstruction']['std']:.4f}",
            f"- clustered real patch：{summary['real_patch_test_reconstruction']['mean']:.4f} ± {summary['real_patch_test_reconstruction']['std']:.4f}",
            "",
            "字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：",
            "",
            f"- KSVD：`{block['dictionary_stability']['ksvd']}`",
            f"- clustered real patch：`{block['dictionary_stability']['clustered_real_patch']}`",
            "",
        ]
    lines += [
        "## 解释纪律",
        "",
        "1. KSVD 只有稳定优于 `raw_patch_mean`、PCA、random dictionary 和 clustered real-patch dictionary，才支持“字典学习本身提供额外信息”。",
        "2. `true_relation` 必须稳定优于 `shuffled_relation`，才能支持当前 sparse atom identity 与 patch 位置的正确绑定有用。",
        "3. raw 好、cleaned 明显下降时，应首先解释为同构重复/数据集偏差，而不是模型表达力。",
        "4. 即便 KSVD 分类更好，当前 WL 向量 atom 仍不自动满足合法图、精确可解码或 exact occurrence；这些需要单独验证。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["IMDB-BINARY", "IMDB-MULTI"])
    ap.add_argument("--variants", nargs="+", choices=["raw", "cleaned"], default=["raw", "cleaned"])
    ap.add_argument("--methods", nargs="+", choices=["B0", "R2", "UniformRW", "CoverageRW", "Path", "PPR"], default=["B0", "R2"])
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sampling-seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-patches", type=int, default=16)
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--walk-length", type=int, default=24)
    ap.add_argument("--n-atoms", type=int, default=16)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--max-train-patches", type=int, default=6000)
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
            "n_patches": args.n_patches,
            "max_nodes": args.max_nodes,
            "walk_length": args.walk_length,
            "n_atoms": args.n_atoms,
            "T": args.T,
            "n_iter": args.n_iter,
            "max_train_patches": args.max_train_patches,
            "limit_per_class": args.limit_per_class,
        },
        "experiments": {},
    }
    for dataset in args.datasets:
        for variant in args.variants:
            cleaned = variant == "cleaned"
            graphs, y, _node_feats, meta = load_tud(
                dataset, cleaned=cleaned, structure_only=True
            )
            graphs, y = _subset_per_class(graphs, y, args.limit_per_class, seed=0)
            for method in args.methods:
                tag = f"{dataset}/{variant}/{method}"
                print(f"[run] {tag}: n={len(graphs)}", flush=True)
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
                    split_seeds=args.split_seeds,
                    n_atoms=args.n_atoms,
                    T=args.T,
                    n_iter=args.n_iter,
                    max_train_patches=args.max_train_patches,
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
