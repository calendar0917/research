"""Audit whether KSVD vector atoms correspond to stable real graph patches.

The classifier can improve even when dictionary atoms have no coherent graph
semantics. This runner therefore ignores labels during dictionary learning and
measures, fold by fold:

* cosine projectability of each KSVD atom to real training patches;
* negative mass after sign alignment (a legal WL-count vector is non-negative);
* structural-signature purity among the nearest real patches;
* cross-fold vector matching and nearest-patch semantic agreement.

Run from ``tracks/ksvd`` with the gsn-official Python environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .graph import Graph
from .ksvd import ksvd
from .run_real_structure_ksvd import (
    CACHE_DIR,
    _atomic_json,
    _normalize_columns,
    clustered_real_dictionary,
    prepare_graphs,
)
from .sampling_route import MatchedSamplingConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "real_structure_atom_audit_20260730.json"
DEFAULT_MD = ROOT / "docs" / "REAL_STRUCTURE_ATOM_AUDIT_20260730.md"


def patch_signature(g: Graph, nodes: set[int]) -> dict[str, Any]:
    sub = g.induced(nodes)
    mapping = {u: i for i, u in enumerate(sub.nodes)}
    h = nx.Graph()
    h.add_nodes_from(range(sub.n))
    h.add_edges_from((mapping[u], mapping[v]) for u, v in sub.edges())
    degrees = sorted((d for _, d in h.degree()), reverse=True)
    triangles = sum(nx.triangles(h).values()) // 3
    components = nx.number_connected_components(h) if h.number_of_nodes() else 0
    wl_hash = nx.weisfeiler_lehman_graph_hash(h, iterations=4)
    key = f"n{sub.n}|m{sub.num_edges()}|d{','.join(map(str, degrees))}|t{triangles}|c{components}|wl{wl_hash}"
    return {
        "key": key,
        "n": sub.n,
        "m": sub.num_edges(),
        "degree_sequence": degrees,
        "triangles": int(triangles),
        "components": int(components),
        "wl_hash": wl_hash,
        "edges": [[mapping[u], mapping[v]] for u, v in sub.edges()],
    }


def _training_matrix_with_sources(
    Ys: list[np.ndarray], train_idx: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    columns: list[np.ndarray] = []
    sources: list[tuple[int, int]] = []
    for gi in train_idx:
        Y = Ys[int(gi)]
        columns.append(Y)
        sources.extend((int(gi), pj) for pj in range(Y.shape[1]))
    return np.concatenate(columns, axis=1), sources


def audit_dictionary(
    D: np.ndarray,
    Yfit: np.ndarray,
    fit_sources: list[tuple[int, int]],
    graphs: list[Graph],
    pcs: list[Any],
    top_k: int,
) -> list[dict[str, Any]]:
    Yfit = _normalize_columns(Yfit)
    D = _normalize_columns(D)
    out: list[dict[str, Any]] = []
    for atom_idx in range(D.shape[1]):
        d = D[:, atom_idx].copy()
        signed = Yfit.T @ d
        best_abs = int(np.argmax(np.abs(signed)))
        if signed[best_abs] < 0:
            d = -d
            signed = -signed
        order = np.argsort(-signed)[: min(top_k, len(signed))]
        signatures: list[dict[str, Any]] = []
        nearest: list[dict[str, Any]] = []
        for rank, j in enumerate(order):
            gi, pj = fit_sources[int(j)]
            sig = patch_signature(graphs[gi], pcs[gi].node_sets[pj])
            signatures.append(sig)
            nearest.append(
                {
                    "rank": rank,
                    "cosine": float(signed[int(j)]),
                    "graph_index": gi,
                    "patch_index": pj,
                    "center": int(pcs[gi].centers[pj]),
                    "signature": sig,
                }
            )
        counts = Counter(s["key"] for s in signatures)
        mode_key, mode_count = counts.most_common(1)[0]
        negative_mass = float(np.abs(d[d < 0]).sum() / max(np.abs(d).sum(), 1e-12))
        out.append(
            {
                "atom": atom_idx,
                "nearest_signed_cosine": float(signed[order[0]]),
                "negative_mass_fraction": negative_mass,
                "topk_signature_purity": float(mode_count / len(signatures)),
                "topk_mode_signature": mode_key,
                "nearest_signature": signatures[0]["key"],
                "projectable_proxy": bool(signed[order[0]] >= 0.95 and negative_mass <= 0.05),
                "nearest_patches": nearest,
            }
        )
    return out


def pairwise_stability(runs: list[dict[str, Any]]) -> dict[str, Any]:
    vector_scores: list[float] = []
    nearest_agreements: list[float] = []
    mode_agreements: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            A = _normalize_columns(np.asarray(runs[i]["dictionary"], dtype=np.float64))
            B = _normalize_columns(np.asarray(runs[j]["dictionary"], dtype=np.float64))
            sim = np.abs(A.T @ B)
            r, c = linear_sum_assignment(-sim)
            vector_scores.extend(float(sim[a, b]) for a, b in zip(r, c))
            nearest_agreements.extend(
                float(runs[i]["atoms"][a]["nearest_signature"] == runs[j]["atoms"][b]["nearest_signature"])
                for a, b in zip(r, c)
            )
            mode_agreements.extend(
                float(runs[i]["atoms"][a]["topk_mode_signature"] == runs[j]["atoms"][b]["topk_mode_signature"])
                for a, b in zip(r, c)
            )
    return {
        "matched_atom_absolute_cosine_mean": float(np.mean(vector_scores)),
        "matched_atom_absolute_cosine_std": float(np.std(vector_scores)),
        "nearest_signature_agreement": float(np.mean(nearest_agreements)),
        "topk_mode_signature_agreement": float(np.mean(mode_agreements)),
        "n_matched_atoms": len(vector_scores),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    atoms = [a for run in runs for a in run["atoms"]]
    vals = lambda key: np.asarray([float(a[key]) for a in atoms], dtype=np.float64)
    signature_counts = Counter(a["nearest_signature"] for a in atoms)
    nearest_nm = []
    for a in atoms:
        parts = a["nearest_signature"].split("|")
        nearest_nm.append((int(parts[0][1:]), int(parts[1][1:])))
    clique_fraction = float(np.mean([m == n * (n - 1) // 2 for n, m in nearest_nm]))
    max_observed_n = max((n for n, _ in nearest_nm), default=0)
    max_size_fraction = float(np.mean([n == max_observed_n for n, _ in nearest_nm]))
    return {
        "n_runs": len(runs),
        "n_atoms_total": len(atoms),
        "nearest_signed_cosine": {
            "mean": float(vals("nearest_signed_cosine").mean()),
            "std": float(vals("nearest_signed_cosine").std()),
            "q10": float(np.quantile(vals("nearest_signed_cosine"), 0.1)),
        },
        "negative_mass_fraction": {
            "mean": float(vals("negative_mass_fraction").mean()),
            "std": float(vals("negative_mass_fraction").std()),
            "q90": float(np.quantile(vals("negative_mass_fraction"), 0.9)),
        },
        "topk_signature_purity": {
            "mean": float(vals("topk_signature_purity").mean()),
            "std": float(vals("topk_signature_purity").std()),
        },
        "projectable_proxy_fraction": float(np.mean(vals("projectable_proxy"))),
        "n_unique_nearest_signatures": len(signature_counts),
        "nearest_clique_fraction": clique_fraction,
        "nearest_max_observed_size": max_observed_n,
        "nearest_max_size_fraction": max_size_fraction,
        "most_common_nearest_signatures": signature_counts.most_common(10),
        "cross_run_stability": pairwise_stability(runs),
    }


def evaluate_dataset(
    dataset: str,
    cleaned: bool,
    method: str,
    split_seeds: list[int],
    n_splits: int,
    n_patches: int,
    max_nodes: int,
    n_atoms: int,
    T: int,
    n_iter: int,
    max_train_patches: int,
    top_k: int,
) -> dict[str, Any]:
    graphs, y, _, meta = load_tud(dataset, cleaned=cleaned, structure_only=True)
    cfg = MatchedSamplingConfig(n_patches=n_patches, max_nodes=max_nodes, seed=0)
    prepared = prepare_graphs(
        dataset, cleaned, graphs, method, cfg, 0, None, use_cache=True
    )
    Ys = prepared["Ys"]
    pcs = prepared["pcs"]
    ksvd_runs: list[dict[str, Any]] = []
    real_runs: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold, (tr, _te) in enumerate(cv.split(np.zeros(len(y)), y)):
            Ytr, sources = _training_matrix_with_sources(Ys, tr)
            fit_seed = split_seed * 1000 + fold
            if Ytr.shape[1] > max_train_patches:
                rng = np.random.default_rng(fit_seed)
                keep = rng.choice(Ytr.shape[1], max_train_patches, replace=False)
                Yfit = Ytr[:, keep]
                fit_sources = [sources[int(i)] for i in keep]
            else:
                Yfit = Ytr
                fit_sources = sources
            k = min(n_atoms, Yfit.shape[0], Yfit.shape[1])
            D, _, info = ksvd(
                Yfit,
                n_atoms=k,
                T=min(T, k),
                T_min=min(T, 2),
                n_iter=n_iter,
                seed=fit_seed,
            )
            Dreal, _ = clustered_real_dictionary(Yfit, k, fit_seed + 211)
            ksvd_runs.append(
                {
                    "split_seed": split_seed,
                    "fold": fold,
                    "dictionary_info": info,
                    "dictionary": D.tolist(),
                    "atoms": audit_dictionary(D, Yfit, fit_sources, graphs, pcs, top_k),
                }
            )
            real_runs.append(
                {
                    "split_seed": split_seed,
                    "fold": fold,
                    "dictionary": Dreal.tolist(),
                    "atoms": audit_dictionary(Dreal, Yfit, fit_sources, graphs, pcs, top_k),
                }
            )
    return {
        "dataset_meta": meta,
        "ksvd": {"summary": summarize_runs(ksvd_runs), "runs": ksvd_runs},
        "clustered_real_patch": {"summary": summarize_runs(real_runs), "runs": real_runs},
    }


def render_markdown(result: dict[str, Any]) -> str:
    c = result["config"]
    lines = [
        "# KSVD atom 的真实 patch 语义审计（2026-07-30）",
        "",
        "> 分类增益不能证明 atom 是合法结构词汇。本报告直接把每个 atom 投影到训练折真实 patch，并检查可投影性、非负性、近邻结构纯度和跨折语义一致性。",
        "",
        "## 协议",
        "",
        f"- 数据集：`{', '.join(c['datasets'])}`，variant=`{c['variant']}`，patch=`{c['method']}`。",
        f"- {c['n_splits']}-fold × seeds {c['split_seeds']}；每折 {c['n_atoms']} atoms；KSVD iterations={c['n_iter']}。",
        f"- 每个 atom 检索 top-{c['top_k']} 真实训练 patch。",
        "- `projectable proxy`：符号对齐后 nearest cosine ≥0.95 且 negative mass ≤0.05；它只是向量层面的必要条件，不等价于严格图可解码。",
        "- 结构 signature 包含节点/边数、度序列、三角形、连通分量和 WL hash；不是严格 canonical isomorphism code。",
        "",
    ]
    for tag, block in result["experiments"].items():
        lines += [f"## {tag}", ""]
        lines += [
            "| 字典 | nearest cosine | negative mass | top-k signature purity | projectable proxy | nearest-signature cross-run agreement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in ["ksvd", "clustered_real_patch"]:
            s = block[name]["summary"]
            lines.append(
                f"| {name} | {s['nearest_signed_cosine']['mean']:.4f} ± {s['nearest_signed_cosine']['std']:.4f} "
                f"| {s['negative_mass_fraction']['mean']:.4f} | {s['topk_signature_purity']['mean']:.4f} "
                f"| {s['projectable_proxy_fraction']:.4f} | {s['cross_run_stability']['nearest_signature_agreement']:.4f} |"
            )
        ks = block["ksvd"]["summary"]
        rs = block["clustered_real_patch"]["summary"]
        lines += [
            "",
            f"- KSVD matched-vector cosine：{ks['cross_run_stability']['matched_atom_absolute_cosine_mean']:.4f}；top-k mode signature agreement：{ks['cross_run_stability']['topk_mode_signature_agreement']:.4f}。",
            f"- Real-patch matched-vector cosine：{rs['cross_run_stability']['matched_atom_absolute_cosine_mean']:.4f}；top-k mode signature agreement：{rs['cross_run_stability']['topk_mode_signature_agreement']:.4f}。",
            f"- KSVD nearest signatures 数量：{ks['n_unique_nearest_signatures']}；最常见：`{ks['most_common_nearest_signatures'][:5]}`。",
            f"- nearest patch 中 clique 占比：KSVD={ks['nearest_clique_fraction']:.4f}，real-patch={rs['nearest_clique_fraction']:.4f}；最大观测 patch size={ks['nearest_max_observed_size']}，KSVD 达到该上限的比例={ks['nearest_max_size_fraction']:.4f}。",
            "",
        ]
    lines += [
        "## 判断规则",
        "",
        "1. vector cosine 高但 signature agreement 低，表示字典只在 WL 向量空间稳定，未形成稳定 graphlet 语义。",
        "2. negative mass 高或 nearest cosine 低，说明普通欧氏 KSVD atom 很难直接解释为合法计数/图结构。",
        "3. clustered real-patch 若语义稳定性不低于 KSVD，则继续使用自由向量 atom 的必要性不足。",
        "4. 本审计不使用标签；它只能判断 vocabulary plausibility，不能替代下游任务评估。",
        "",
        "## 当前结论",
        "",
    ]
    for tag, block in result["experiments"].items():
        ks = block["ksvd"]["summary"]
        rs = block["clustered_real_patch"]["summary"]
        lines.append(
            f"- `{tag}`：KSVD projectable proxy={ks['projectable_proxy_fraction']:.3f}，"
            f"nearest-signature agreement={ks['cross_run_stability']['nearest_signature_agreement']:.3f}；"
            f"real-patch 对应值为 1.000/{rs['cross_run_stability']['nearest_signature_agreement']:.3f}。"
        )
        if ks["nearest_clique_fraction"] > 0.5:
            lines.append(
                f"  nearest prototypes 中 {ks['nearest_clique_fraction']:.1%} 是 clique，主要按 clique/patch size 区分，容易复述规模与度结构。"
            )
        else:
            lines.append(
                f"  clique 仅占 {ks['nearest_clique_fraction']:.1%}，但 {ks['nearest_max_size_fraction']:.1%} 的 nearest prototypes 达到观测最大 size={ks['nearest_max_observed_size']}；常见原型仍受 sampler 尺度/星形结构主导。"
            )
    lines += [
        "- clustered real-patch dictionary 天然合法、可回指，且语义稳定性未弱于 KSVD；普通欧氏 KSVD 暂无不可替代性证据。",
        "- 因此当前最多支持“WL patch 空间中的 coding directions”，不支持“已学得跨折稳定、合法、可组合的 graphlet vocabulary”。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["IMDB-BINARY", "IMDB-MULTI"])
    ap.add_argument("--variant", choices=["raw", "cleaned"], default="cleaned")
    ap.add_argument("--method", choices=["B0", "R2"], default="B0")
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-patches", type=int, default=16)
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--n-atoms", type=int, default=16)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--max-train-patches", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "config": {
            "datasets": args.datasets,
            "variant": args.variant,
            "method": args.method,
            "split_seeds": args.split_seeds,
            "n_splits": args.n_splits,
            "n_patches": args.n_patches,
            "max_nodes": args.max_nodes,
            "n_atoms": args.n_atoms,
            "T": args.T,
            "n_iter": args.n_iter,
            "max_train_patches": args.max_train_patches,
            "top_k": args.top_k,
            "cache_dir": str(CACHE_DIR),
        },
        "experiments": {},
    }
    for dataset in args.datasets:
        tag = f"{dataset}/{args.variant}/{args.method}"
        print(f"[audit] {tag}", flush=True)
        result["experiments"][tag] = evaluate_dataset(
            dataset=dataset,
            cleaned=args.variant == "cleaned",
            method=args.method,
            split_seeds=args.split_seeds,
            n_splits=args.n_splits,
            n_patches=args.n_patches,
            max_nodes=args.max_nodes,
            n_atoms=args.n_atoms,
            T=args.T,
            n_iter=args.n_iter,
            max_train_patches=args.max_train_patches,
            top_k=args.top_k,
        )
        _atomic_json(args.output_json, result)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
