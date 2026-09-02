"""Compare exact-canonical and WL projected-KSVD results fold by fold.

This script performs no model fitting. It combines result JSON files produced by
``run_real_structure_projected_ksvd`` under the same outer splits and reports
paired changes caused only by replacing the patch vectorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WL = ROOT / "results" / "real_structure_projected_ksvd_20260730.json"
DEFAULT_CANONICAL = [
    ROOT / "results" / "canonical_projected_imdb_binary_20260730.json",
    ROOT / "results" / "canonical_projected_imdb_multi_20260730.json",
    ROOT / "results" / "canonical_projected_reddit_binary_20260730.json",
]
DEFAULT_JSON = ROOT / "results" / "canonical_vs_wl_projected_ksvd_20260730.json"
DEFAULT_MD = ROOT / "docs" / "CANONICAL_VS_WL_PROJECTED_KSVD_20260730.md"

METHODS = {
    "stats_plus_raw_patch_mean": "raw mean",
    "stats_plus_pca_content": "PCA",
    "stats_plus_real_patch_dictionary_content": "clustered real-patch",
    "stats_plus_ksvd_content": "unconstrained KSVD",
    "stats_plus_final_projected_content": "final-projected KSVD",
    "stats_plus_iterative_projected_content": "iterative-projected KSVD",
    "stats_plus_relation_graph_only": "relation graph only (unchanged control)",
    "stats_plus_real_patch_dictionary_content_graph": "real-patch + relation graph",
    "stats_plus_ksvd_content_graph": "KSVD + relation graph",
    "stats_plus_final_projected_content_graph": "final-projected + relation graph",
    "stats_plus_iterative_projected_content_graph": "iterative-projected + relation graph",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_tag(tag: str) -> str:
    for suffix in ("/canonical", "/rooted_canonical", "/wl"):
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def _score(fold: dict[str, Any], method: str) -> float:
    value = fold["score"][method]
    if isinstance(value, dict):
        return float(value["balanced_accuracy"])
    return float(value)


def _paired(canonical: np.ndarray, wl: np.ndarray, n_splits: int) -> dict[str, Any]:
    delta = canonical - wl
    n_seeds = len(delta) // n_splits
    seed_means = [float(delta[i * n_splits:(i + 1) * n_splits].mean()) for i in range(n_seeds)]
    return {
        "wl_mean": float(wl.mean()),
        "canonical_mean": float(canonical.mean()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "wins": int(np.sum(delta > 1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
        "seed_means": seed_means,
        "seed_wins": int(np.sum(np.asarray(seed_means) > 1e-12)),
    }


def compare(wl_result: dict[str, Any], canonical_results: list[dict[str, Any]]) -> dict[str, Any]:
    wl_blocks = {_base_tag(tag): block for tag, block in wl_result["experiments"].items()}
    canonical_blocks: dict[str, Any] = {}
    canonical_config: dict[str, Any] | None = None
    for result in canonical_results:
        cfg = result["config"]
        if cfg.get("feature_mode") != "canonical":
            raise ValueError("all comparison inputs must use feature_mode=canonical")
        canonical_config = canonical_config or cfg
        for tag, block in result["experiments"].items():
            canonical_blocks[_base_tag(tag)] = block

    missing = sorted(set(wl_blocks) - set(canonical_blocks))
    if missing:
        raise ValueError(f"missing canonical results for: {missing}")

    n_splits = int(wl_result["config"]["n_splits"])
    out: dict[str, Any] = {
        "wl_config": wl_result["config"],
        "canonical_config": canonical_config,
        "datasets": {},
    }
    for tag, wl_block in wl_blocks.items():
        can_block = canonical_blocks[tag]
        wl_folds = {(int(f["split_seed"]), int(f["fold"])): f for f in wl_block["folds"]}
        can_folds = {(int(f["split_seed"]), int(f["fold"])): f for f in can_block["folds"]}
        if wl_folds.keys() != can_folds.keys():
            raise ValueError(f"outer folds differ for {tag}")
        keys = sorted(wl_folds)
        methods: dict[str, Any] = {}
        for method in METHODS:
            wl_values = np.asarray([_score(wl_folds[k], method) for k in keys])
            can_values = np.asarray([_score(can_folds[k], method) for k in keys])
            methods[method] = _paired(can_values, wl_values, n_splits)
        out["datasets"][tag] = {"n_graphs": can_block["n_graphs"], "methods": methods}
    return out


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Exact canonical adjacency vs WL：Projected-KSVD 配对复核（2026-07-30）",
        "",
        "> 唯一改变是 patch 向量化：从 WL histogram 改为 nauty canonical labeling 后的完整邻接上三角 + node mask。采样、outer split、字典规模、稀疏度、训练迭代和分类协议保持一致。",
        "",
        "## 判读问题",
        "",
        "1. 若 canonical 显著改善同一种 KSVD 方法，说明 WL 的表示几何可能限制了字典学习，即使观察到的真实 patches 没有 WL type collision。",
        "2. 若 canonical 对 raw、PCA 和 KSVD 都不改善，或 KSVD 没有相对 canonical 自身 baseline 获得稳定优势，则此前 no-go 不能归因于 WL 信息损失。",
        "3. `relation graph only` 不依赖 patch 向量，理论上应保持不变；它用于检查两次实验的 split 与评估是否对齐。",
        "",
    ]
    for tag, block in result["datasets"].items():
        lines += [f"## {tag}", "", f"图数：{block['n_graphs']}。", ""]
        lines += [
            "| 方法 | WL | canonical | canonical − WL | fold W/T/L | seed means |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for method, label in METHODS.items():
            d = block["methods"][method]
            seeds = ", ".join(f"{x:+.4f}" for x in d["seed_means"])
            lines.append(
                f"| {label} | {d['wl_mean']:.4f} | {d['canonical_mean']:.4f} | "
                f"{d['delta_mean']:+.4f} | {d['wins']}/{d['ties']}/{d['losses']} | [{seeds}] |"
            )
        lines.append("")

    projected = (
        "stats_plus_final_projected_content",
        "stats_plus_iterative_projected_content",
    )
    stable = []
    for tag, block in result["datasets"].items():
        for method in projected:
            d = block["methods"][method]
            if d["seed_wins"] == len(d["seed_means"]):
                stable.append((tag, METHODS[method], d["delta_mean"]))
    lines += ["## 跨表示结论", ""]
    if stable:
        for tag, method, delta in stable:
            lines.append(f"- {tag} 的 {method} 在 3/3 seeds 优于 WL 对应方法，平均变化 {delta:+.4f}。")
    else:
        lines.append("- 没有 canonical projected-KSVD 方法在任一数据集上达到相对 WL 对应方法的 3/3 seed 稳定提升。")
    lines += [
        "",
        "最终是否值得继续 rooted canonical，除跨表示变化外，还必须结合 canonical 报告中相对 raw/PCA/real-patch baseline 的结果判断。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wl-json", type=Path, default=DEFAULT_WL)
    ap.add_argument("--canonical-json", nargs="+", type=Path, default=DEFAULT_CANONICAL)
    ap.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(_load(args.wl_json), [_load(p) for p in args.canonical_json])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
