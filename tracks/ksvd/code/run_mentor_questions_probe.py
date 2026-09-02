"""Small, leakage-controlled probes for the questions in luyin10/luyin11.

The goal is mechanism diagnosis, not TU benchmark reporting:
1) reconstruction error versus graph-level usefulness;
2) node-permutation invariance of patch vectorization;
3) random walk as one sampler rather than the method core;
4) loss of relations when patches are pooled as an unordered bag;
5) whether node labels can be included without abandoning invariance.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud, node_feature_readout
from .graph import Graph, from_edges
from .ksvd import ksvd
from .sample import SampleBundle, SampleConfig, run_method
from .vectorize import (
    adjacency_padded,
    flatten_upper,
    labeled_wl_patch_features,
    wl_patch_features,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "mentor_questions_probe_20260729.json"
OUT_MD = ROOT / "docs" / "MENTOR_QUESTIONS_EVIDENCE_20260729.md"


def radius_sets(g: Graph, radius: int) -> list[set[int]]:
    out: list[set[int]] = []
    for root in g.nodes:
        seen = {root}
        frontier = {root}
        for _ in range(radius):
            nxt: set[int] = set()
            for u in frontier:
                nxt.update(g.neighbors(u))
            nxt -= seen
            seen |= nxt
            frontier = nxt
        out.append(seen)
    return out


def patch_sets(g: Graph, mode: str, seed: int) -> list[set[int]]:
    if mode == "B0":
        return radius_sets(g, 1)
    if mode == "R2":
        return radius_sets(g, 2)
    if mode == "RW":
        cfg = SampleConfig(
            p=1.0,
            q=1.0,
            walk_length=7,
            max_nodes=8,
            num_walks=12,
            edge_decay=1.0,
            no_backtrack=True,
            seed=seed,
        )
        return run_method(g, "B1", cfg).node_sets
    raise ValueError(mode)


def vectorize_sets(
    g: Graph,
    sets: list[set[int]],
    max_nodes: int,
    feature_mode: str,
    node_feat: np.ndarray | None = None,
) -> np.ndarray:
    cols: list[np.ndarray] = []
    for S in sets:
        if feature_mode == "wl":
            y = np.asarray(wl_patch_features(g, S, max_nodes=max_nodes), dtype=np.float64)
        elif feature_mode == "labeled_wl":
            y = np.asarray(
                labeled_wl_patch_features(
                    g,
                    S,
                    max_nodes=max_nodes,
                    node_feat=node_feat,
                    edge_feat=None,
                    atom_bins=32,
                    bond_bins=4,
                    wl_bins=64,
                    n_iter=3,
                ),
                dtype=np.float64,
            )
        else:
            raise ValueError(feature_mode)
        norm = float(np.linalg.norm(y))
        if norm > 1e-12:
            cols.append(y / norm)
    if not cols:
        raise RuntimeError("no nonzero patches")
    return np.stack(cols, axis=1)


def sparse_code(D: np.ndarray, Y: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    # Import the tested OMP implementation but keep this probe independent of
    # graph_level.py readout changes.
    from .ksvd import _omp

    X = np.zeros((D.shape[1], Y.shape[1]), dtype=np.float64)
    for j in range(Y.shape[1]):
        X[:, j] = _omp(D, Y[:, j], min(T, D.shape[1]))
    R = Y - D @ X
    denom = np.maximum(np.linalg.norm(Y, axis=0), 1e-12)
    err = np.linalg.norm(R, axis=0) / denom
    return X, err


def graph_readouts(X: np.ndarray, err: np.ndarray) -> dict[str, np.ndarray]:
    A = np.abs(X)
    k, n = A.shape
    mean = A.mean(axis=1)
    maxv = A.max(axis=1)
    std = A.std(axis=1)
    usage = (A > 1e-10).mean(axis=1)
    q75 = np.quantile(A, 0.75, axis=1)
    energy = (X * X).mean(axis=1)
    signed = X.mean(axis=1)
    winner = np.bincount(np.argmax(A, axis=0), minlength=k).astype(np.float64) / n
    code = np.concatenate([mean, maxv, std, usage, q75, energy, signed, winner])
    recon = np.asarray(
        [err.mean(), err.std(), *np.quantile(err, [0.50, 0.75, 0.90]), err.max(), n, np.log1p(n)],
        dtype=np.float64,
    )
    return {"code": code, "recon": recon, "code_recon": np.concatenate([code, recon])}


def fit_score(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, seed: int) -> float:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, random_state=seed),
    )
    clf.fit(Xtr, ytr)
    return float(accuracy_score(yte, clf.predict(Xte)))


def cv_ksvd(
    graphs: list[Graph],
    y: np.ndarray,
    node_feats: list[np.ndarray | None] | None,
    *,
    patch_mode: str,
    feature_mode: str,
    n_atoms: int,
    T: int,
    n_iter: int = 5,
    n_splits: int = 5,
    split_seed: int = 123,
    include_attr: bool = False,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    fold_rows: list[dict] = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        Ys: dict[int, np.ndarray] = {}
        for i, g in enumerate(graphs):
            nf = None if node_feats is None else node_feats[i]
            Ys[i] = vectorize_sets(
                g,
                patch_sets(g, patch_mode, seed=split_seed + i * 17),
                max_nodes=8,
                feature_mode=feature_mode,
                node_feat=nf,
            )
        Ytr = np.concatenate([Ys[int(i)] for i in tr], axis=1)
        D, _, dinfo = ksvd(
            Ytr,
            n_atoms=n_atoms,
            T=T,
            T_min=min(2, T),
            n_iter=n_iter,
            seed=1000 + fold,
        )
        feats: dict[str, list[np.ndarray]] = {"code": [], "recon": [], "code_recon": []}
        graph_err: list[float] = []
        for i in range(len(graphs)):
            X, err = sparse_code(D, Ys[i], T)
            r = graph_readouts(X, err)
            attr = (
                node_feature_readout(None if node_feats is None else node_feats[i])
                if include_attr
                else np.zeros(0, dtype=np.float64)
            )
            for key in feats:
                feats[key].append(np.concatenate([r[key], attr]))
            graph_err.append(float(err.mean()))
        acc = {}
        for key, rows in feats.items():
            M = np.stack(rows)
            acc[key] = fit_score(M[tr], y[tr], M[te], y[te], seed=fold)
        fold_rows.append(
            {
                "fold": fold,
                "accuracy": acc,
                "train_recon_mean": float(np.mean([graph_err[int(i)] for i in tr])),
                "test_recon_mean": float(np.mean([graph_err[int(i)] for i in te])),
                "dictionary_recon_train": float(dinfo["recon_rel"]),
            }
        )
    return {
        "config": {
            "patch_mode": patch_mode,
            "feature_mode": feature_mode,
            "n_atoms": n_atoms,
            "T": T,
            "n_iter": n_iter,
            "n_splits": n_splits,
            "split_seed": split_seed,
            "include_attr": include_attr,
            "dictionary_fit": "training fold patches only",
        },
        "folds": fold_rows,
        "mean_accuracy": {
            key: float(np.mean([r["accuracy"][key] for r in fold_rows]))
            for key in ["code", "recon", "code_recon"]
        },
        "std_accuracy": {
            key: float(np.std([r["accuracy"][key] for r in fold_rows]))
            for key in ["code", "recon", "code_recon"]
        },
        "mean_test_recon": float(np.mean([r["test_recon_mean"] for r in fold_rows])),
    }


def relabel(g: Graph, node_feat: np.ndarray | None, rng: np.random.Generator) -> tuple[Graph, np.ndarray | None]:
    perm = rng.permutation(g.n)  # old -> new
    edges = [(int(perm[u]), int(perm[v])) for u, v in g.edges()]
    gp = from_edges(g.n, edges)
    if node_feat is None:
        return gp, None
    xp = np.zeros_like(node_feat)
    for old in range(g.n):
        xp[int(perm[old])] = node_feat[old]
    return gp, xp


def permutation_probe(graphs: list[Graph], node_feats: list[np.ndarray | None]) -> dict:
    rng = np.random.default_rng(20260729)
    diffs = {"bfs": [], "degree": [], "bfs_plus_degree": [], "wl": [], "labeled_wl": []}
    exact = {k: [] for k in diffs}
    for gi in range(min(80, len(graphs))):
        g, x = graphs[gi], node_feats[gi]
        S = set(g.nodes)
        base = {
            "bfs": np.asarray(flatten_upper(adjacency_padded(g, S, 32, "bfs"))),
            "degree": np.asarray(flatten_upper(adjacency_padded(g, S, 32, "degree"))),
            "wl": np.asarray(wl_patch_features(g, S, 32, n_bins=128)),
            "labeled_wl": np.asarray(
                labeled_wl_patch_features(
                    g, S, 32, x, None, atom_bins=32, bond_bins=4, wl_bins=128
                )
            ),
        }
        base["bfs_plus_degree"] = np.concatenate([base["bfs"], base["degree"]])
        for _ in range(8):
            gp, xp = relabel(g, x, rng)
            Sp = set(gp.nodes)
            cur = {
                "bfs": np.asarray(flatten_upper(adjacency_padded(gp, Sp, 32, "bfs"))),
                "degree": np.asarray(flatten_upper(adjacency_padded(gp, Sp, 32, "degree"))),
                "wl": np.asarray(wl_patch_features(gp, Sp, 32, n_bins=128)),
                "labeled_wl": np.asarray(
                    labeled_wl_patch_features(
                        gp, Sp, 32, xp, None, atom_bins=32, bond_bins=4, wl_bins=128
                    )
                ),
            }
            cur["bfs_plus_degree"] = np.concatenate([cur["bfs"], cur["degree"]])
            for key in diffs:
                den = max(float(np.linalg.norm(base[key])), 1e-12)
                d = float(np.linalg.norm(base[key] - cur[key]) / den)
                diffs[key].append(d)
                exact[key].append(bool(np.array_equal(base[key], cur[key])))
    return {
        key: {
            "mean_relative_change": float(np.mean(diffs[key])),
            "median_relative_change": float(np.median(diffs[key])),
            "exact_match_rate": float(np.mean(exact[key])),
            "n_comparisons": len(diffs[key]),
        }
        for key in diffs
    }


def cube_graph() -> Graph:
    edges = []
    for u in range(8):
        for bit in [1, 2, 4]:
            v = u ^ bit
            if u < v:
                edges.append((u, v))
    return from_edges(8, edges)


def mobius_ladder_8() -> Graph:
    edges = [(i, (i + 1) % 8) for i in range(8)]
    edges += [(i, i + 4) for i in range(4)]
    return from_edges(8, edges)


def canonical_column_multiset(Y: np.ndarray) -> list[bytes]:
    return sorted(np.ascontiguousarray(Y[:, j]).tobytes() for j in range(Y.shape[1]))


def relation_and_sampler_probe() -> dict:
    base = [cube_graph(), mobius_ladder_8()]
    # Verify the intended positive/negative control directly on patch multisets.
    equality = {}
    for mode in ["B0", "R2"]:
        Ys = [vectorize_sets(g, patch_sets(g, mode, 0), 8, "wl") for g in base]
        equality[mode] = canonical_column_multiset(Ys[0]) == canonical_column_multiset(Ys[1])

    rng = np.random.default_rng(77)
    graphs: list[Graph] = []
    labels: list[int] = []
    for cls, g in enumerate(base):
        for _ in range(60):
            gp, _ = relabel(g, None, rng)
            graphs.append(gp)
            labels.append(cls)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    graphs = [graphs[int(i)] for i in order]
    y = y[order]
    results = {}
    for mode in ["B0", "R2", "RW"]:
        results[mode] = cv_ksvd(
            graphs,
            y,
            None,
            patch_mode=mode,
            feature_mode="wl",
            n_atoms=8,
            T=2,
            n_iter=4,
            n_splits=5,
            split_seed=31,
        )
    return {
        "task": "cube versus 8-node Mobius ladder; both connected, 3-regular, triangle-free",
        "why": "Every radius-1 ego patch is the same K1,3 star, but the global arrangement differs.",
        "base_patch_multiset_equal": equality,
        "results": results,
    }


def attr_baseline(node_feats: list[np.ndarray | None], y: np.ndarray) -> dict:
    X = np.stack([node_feature_readout(x) for x in node_feats])
    cv = StratifiedKFold(5, shuffle=True, random_state=123)
    scores = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        scores.append(fit_score(X[tr], y[tr], X[te], y[te], seed=fold))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "scores": scores}


def render_md(r: dict) -> str:
    p = r["permutation_invariance"]
    rel = r["patch_relation_and_sampler"]
    lines = [
        "# 导师问题的小型证据实验（2026-07-29）",
        "",
        "> 目的：回答 `docs/luyin/luyin10.txt` 与 `luyin11.txt` 中的机制问题，不作为 TUData 排行榜结果。所有 MUTAG 字典均只在训练折拟合。",
        "",
        "## 1. 节点置换不变性",
        "",
        "| patch 表示 | 随机重编号后的平均相对变化 | 完全相同比例 |",
        "|---|---:|---:|",
    ]
    for key in ["bfs", "degree", "bfs_plus_degree", "wl", "labeled_wl"]:
        lines.append(
            f"| {key} | {p[key]['mean_relative_change']:.6f} | {p[key]['exact_match_rate']:.3f} |"
        )
    lines += [
        "",
        "结论：按度/BFS 排序（即使把两种排序拼起来）不是严格的不变表示；WL 直方图及带节点标签的 WL 表示在该测试中严格不变。",
        "",
        "## 2. 重建误差与图分类不是同一个目标（MUTAG, 5-fold）",
        "",
        "| D | test patch 重建误差↓ | 稀疏码读出 Acc | 仅重建统计 Acc | 稀疏码+重建 Acc |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in r["mutag_reconstruction_sweep"]:
        lines.append(
            f"| {row['config']['n_atoms']} | {row['mean_test_recon']:.4f} | "
            f"{row['mean_accuracy']['code']:.3f} | {row['mean_accuracy']['recon']:.3f} | "
            f"{row['mean_accuracy']['code_recon']:.3f} |"
        )
    lines += [
        "",
        "## 3. patch 之间的关系，以及随机游走的角色",
        "",
        f"任务：{rel['task']}。{rel['why']}",
        "",
        f"- 一阶 patch 多重集合是否完全相同：`{rel['base_patch_multiset_equal']['B0']}`",
        f"- 二阶 patch 多重集合是否完全相同：`{rel['base_patch_multiset_equal']['R2']}`",
        "",
        "| patch 构造 | 稀疏码读出 Acc | test 重建误差 |",
        "|---|---:|---:|",
    ]
    for mode in ["B0", "R2", "RW"]:
        row = rel["results"][mode]
        lines.append(
            f"| {mode} | {row['mean_accuracy']['code']:.3f} | {row['mean_test_recon']:.4f} |"
        )
    lines += [
        "",
        "结论：一阶 patch 的‘袋子’确实会丢掉 patch 如何连接的信息；扩大 patch 后可以区分。随机游走能做到，但确定性的二阶邻域也能做到，因此随机游走只是候选采样器，不应被当作核心假设。",
        "",
        "## 4. 节点特征是否可以进入字典（MUTAG, 5-fold）",
        "",
        f"- 仅全图节点属性统计：{r['mutag_node_features']['attribute_baseline']['mean']:.3f}",
        f"- 只用拓扑 patch 的 KSVD：{r['mutag_node_features']['topology_only']['mean_accuracy']['code']:.3f}",
        f"- patch 中加入节点标签后：{r['mutag_node_features']['labeled_patch']['mean_accuracy']['code']:.3f}",
        f"- 标签 patch 稀疏码 + 全图节点属性：{r['mutag_node_features']['labeled_patch_plus_attr']['mean_accuracy']['code']:.3f}",
        "",
        "这里的重点不是哪一个小数据集数字最高，而是：节点标签可以在保持置换不变的前提下进入 patch；但‘能重建节点标签’仍不等于‘得到任务相关的图表示’。",
        "",
        "## 协议说明",
        "",
        "- 5-fold stratified CV；固定逻辑回归；不做超参搜索。",
        "- 每一折的共享字典只用训练折 patch 学习。",
        "- 结果用于机制判断，不与论文里的 nested CV / 10-fold benchmark 数字直接比较。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    started = time.time()
    graphs, y, node_feats, meta = load_tud("MUTAG")
    print("loaded", meta, flush=True)

    result: dict = {
        "experiment": "mentor_questions_probe_v1",
        "date": "2026-07-29",
        "sources": ["docs/luyin/luyin10.txt", "docs/luyin/luyin11.txt"],
        "mutag_meta": meta,
    }

    print("[1/4] permutation probe", flush=True)
    result["permutation_invariance"] = permutation_probe(graphs, node_feats)

    print("[2/4] reconstruction sweep", flush=True)
    sweep = []
    for D in [4, 8, 16, 32]:
        row = cv_ksvd(
            graphs,
            y,
            node_feats,
            patch_mode="B0",
            feature_mode="wl",
            n_atoms=D,
            T=min(3, D),
            n_iter=5,
            n_splits=5,
            split_seed=123,
        )
        sweep.append(row)
        print(" D", D, row["mean_test_recon"], row["mean_accuracy"], flush=True)
    result["mutag_reconstruction_sweep"] = sweep

    print("[3/4] relation/sampler probe", flush=True)
    result["patch_relation_and_sampler"] = relation_and_sampler_probe()

    print("[4/4] node-feature probe", flush=True)
    topo = cv_ksvd(
        graphs, y, node_feats, patch_mode="B0", feature_mode="wl",
        n_atoms=12, T=3, n_iter=5, n_splits=5, split_seed=123,
    )
    labeled = cv_ksvd(
        graphs, y, node_feats, patch_mode="B0", feature_mode="labeled_wl",
        n_atoms=12, T=3, n_iter=5, n_splits=5, split_seed=123,
    )
    labeled_attr = cv_ksvd(
        graphs, y, node_feats, patch_mode="B0", feature_mode="labeled_wl",
        n_atoms=12, T=3, n_iter=5, n_splits=5, split_seed=123, include_attr=True,
    )
    result["mutag_node_features"] = {
        "attribute_baseline": attr_baseline(node_feats, y),
        "topology_only": topo,
        "labeled_patch": labeled,
        "labeled_patch_plus_attr": labeled_attr,
    }
    result["elapsed_sec"] = float(time.time() - started)

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(result), encoding="utf-8")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("elapsed", result["elapsed_sec"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
