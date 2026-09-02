"""Build the final WL vs exact canonical vs rooted-canonical KSVD summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_canonical_projected_comparison import METHODS, _base_tag, _score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = {
    "wl": [ROOT / "results" / "real_structure_projected_ksvd_20260730.json"],
    "canonical": [
        ROOT / "results" / "canonical_projected_imdb_binary_20260730.json",
        ROOT / "results" / "canonical_projected_imdb_multi_20260730.json",
        ROOT / "results" / "canonical_projected_reddit_binary_20260730.json",
    ],
    "rooted_canonical": [
        ROOT / "results" / "rooted_canonical_projected_imdb_binary_20260730.json",
        ROOT / "results" / "rooted_canonical_projected_imdb_multi_20260730.json",
        ROOT / "results" / "rooted_canonical_projected_reddit_binary_20260730.json",
    ],
}
DEFAULT_JSON = ROOT / "results" / "nonwl_projected_summary_20260730.json"
DEFAULT_MD = ROOT / "docs" / "NONWL_PROJECTED_KSVD_FINAL_20260730.md"

SHORT_METHODS = {
    "stats_plus_raw_patch_mean": "raw",
    "stats_plus_pca_content": "PCA",
    "stats_plus_real_patch_dictionary_content": "real-patch",
    "stats_plus_ksvd_content": "KSVD",
    "stats_plus_final_projected_content": "final-projected",
    "stats_plus_iterative_projected_content": "iterative-projected",
    "stats_plus_relation_graph_only": "relation-only",
    "stats_plus_final_projected_content_graph": "final-projected+graph",
}


def _load_blocks(paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    blocks: dict[str, Any] = {}
    config: dict[str, Any] | None = None
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        config = config or result["config"]
        for tag, block in result["experiments"].items():
            blocks[_base_tag(tag)] = block
    if config is None:
        raise ValueError("no result files")
    return config, blocks


def _fold_values(block: dict[str, Any], method: str) -> tuple[list[tuple[int, int]], np.ndarray]:
    folds = sorted(block["folds"], key=lambda f: (int(f["split_seed"]), int(f["fold"])))
    keys = [(int(f["split_seed"]), int(f["fold"])) for f in folds]
    return keys, np.asarray([_score(f, method) for f in folds])


def _delta(a: np.ndarray, b: np.ndarray, n_splits: int) -> dict[str, Any]:
    d = a - b
    n_seeds = len(d) // n_splits
    seed_means = [float(d[i*n_splits:(i+1)*n_splits].mean()) for i in range(n_seeds)]
    return {
        "mean": float(d.mean()),
        "wins": int(np.sum(d > 1e-12)),
        "ties": int(np.sum(np.abs(d) <= 1e-12)),
        "losses": int(np.sum(d < -1e-12)),
        "seed_means": seed_means,
        "seed_wins": int(np.sum(np.asarray(seed_means) > 1e-12)),
    }


def build(mode_inputs: dict[str, list[Path]]) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    mode_blocks: dict[str, dict[str, Any]] = {}
    for mode, paths in mode_inputs.items():
        configs[mode], mode_blocks[mode] = _load_blocks(paths)
    datasets = sorted(mode_blocks["wl"])
    for mode, blocks in mode_blocks.items():
        if set(blocks) != set(datasets):
            raise ValueError(f"dataset mismatch for {mode}")
    n_splits = int(configs["wl"]["n_splits"])

    out: dict[str, Any] = {"configs": configs, "datasets": {}}
    for tag in datasets:
        modes: dict[str, Any] = {}
        reference_keys: list[tuple[int, int]] | None = None
        values_by_mode: dict[str, dict[str, np.ndarray]] = {}
        for mode, blocks in mode_blocks.items():
            values_by_mode[mode] = {}
            means = {}
            for method in SHORT_METHODS:
                keys, values = _fold_values(blocks[tag], method)
                if reference_keys is None:
                    reference_keys = keys
                elif keys != reference_keys:
                    raise ValueError(f"fold mismatch: {tag}/{mode}")
                values_by_mode[mode][method] = values
                means[method] = float(values.mean())
            modes[mode] = {"means": means}

        comparisons: dict[str, Any] = {}
        for mode in ("canonical", "rooted_canonical"):
            comparisons[f"{mode}_minus_wl"] = {
                method: _delta(values_by_mode[mode][method], values_by_mode["wl"][method], n_splits)
                for method in SHORT_METHODS
            }
        comparisons["rooted_minus_canonical"] = {
            method: _delta(values_by_mode["rooted_canonical"][method], values_by_mode["canonical"][method], n_splits)
            for method in SHORT_METHODS
        }
        projected = {
            "final_projected": "stats_plus_final_projected_content",
            "iterative_projected": "stats_plus_iterative_projected_content",
        }
        baselines = {
            "raw": "stats_plus_raw_patch_mean",
            "pca": "stats_plus_pca_content",
            "real": "stats_plus_real_patch_dictionary_content",
        }
        within_mode: dict[str, Any] = {}
        for mode in mode_blocks:
            within_mode[mode] = {}
            for projected_name, projected_key in projected.items():
                baseline_deltas = {
                    baseline_name: _delta(
                        values_by_mode[mode][projected_key],
                        values_by_mode[mode][baseline_key],
                        n_splits,
                    )
                    for baseline_name, baseline_key in baselines.items()
                }
                within_mode[mode][projected_name] = {
                    "baselines": baseline_deltas,
                    "passes_dataset": all(
                        d["seed_wins"] == len(d["seed_means"])
                        for d in baseline_deltas.values()
                    ),
                }
        out["datasets"][tag] = {
            "n_graphs": mode_blocks["wl"][tag]["n_graphs"],
            "modes": modes,
            "comparisons": comparisons,
            "within_mode": within_mode,
        }

    promotion: dict[str, Any] = {}
    for mode in mode_blocks:
        promotion[mode] = {}
        for projected_name in ("final_projected", "iterative_projected"):
            passed_datasets = [
                tag for tag, block in out["datasets"].items()
                if block["within_mode"][mode][projected_name]["passes_dataset"]
            ]
            promotion[mode][projected_name] = {
                "passed_datasets": passed_datasets,
                "go": len(passed_datasets) >= 2,
            }
    out["promotion"] = promotion
    return out


def render(result: dict[str, Any]) -> str:
    lines = [
        "# 不使用 WL 的最终 KSVD 复核（2026-07-30）",
        "",
        "> 比较 WL histogram、exact canonical adjacency、root-preserving exact canonical adjacency。三者使用相同 patches、outer folds、训练预算和下游评估；后两者都不以 WL 作为最终结构特征。",
        "",
        "## 表示含义",
        "",
        "- `WL`：度/度对统计与 3 轮 1-WL histogram；置换不变，但理论上有损。",
        "- `canonical`：nauty 规范标号后的完整邻接上三角与 node mask；对无根 induced patch 在同构意义下无损。",
        "- `rooted`：额外约束 sampler center 必须在同构下被保留，区分相同 induced graph 中不同中心角色。",
        "- 多维启发式排序没有作为主方案：tie 最终若由 node id 打破便不置换不变；exact canonical labeling 才是严格的无 WL 对照。",
        "",
        "所有数值均为 `stats + feature` 的 3 seeds × 5 folds balanced accuracy。",
        "",
    ]
    table_methods = list(SHORT_METHODS)
    for tag, block in result["datasets"].items():
        lines += [f"## {tag}", "", f"图数：{block['n_graphs']}。", ""]
        lines += ["| 方法 | WL | canonical | rooted |", "|---|---:|---:|---:|"]
        for method in table_methods:
            means = {mode: block["modes"][mode]["means"][method] for mode in ("wl", "canonical", "rooted_canonical")}
            lines.append(
                f"| {SHORT_METHODS[method]} | {means['wl']:.4f} | {means['canonical']:.4f} | {means['rooted_canonical']:.4f} |"
            )
        lines += ["", "Projected 方法的跨表示 paired 变化：", ""]
        lines += ["| 比较 | 方法 | mean Δ | fold W/T/L | seed means |", "|---|---|---:|---:|---|"]
        for comp_name in ("canonical_minus_wl", "rooted_canonical_minus_wl", "rooted_minus_canonical"):
            for method in ("stats_plus_final_projected_content", "stats_plus_iterative_projected_content"):
                d = block["comparisons"][comp_name][method]
                seeds = ", ".join(f"{x:+.4f}" for x in d["seed_means"])
                lines.append(
                    f"| {comp_name} | {SHORT_METHODS[method]} | {d['mean']:+.4f} | "
                    f"{d['wins']}/{d['ties']}/{d['losses']} | [{seeds}] |"
                )
        lines.append("")

    lines += ["## 预注册式最终判定", ""]
    any_go = False
    for mode in ("wl", "canonical", "rooted_canonical"):
        for method in ("final_projected", "iterative_projected"):
            decision = result["promotion"][mode][method]
            passed = decision["passed_datasets"] or "无"
            lines.append(f"- `{mode}/{method}` 对自身 raw、PCA、real-patch 均 3/3 seed 正向的数据集：{passed}。")
            any_go = any_go or bool(decision["go"])
    lines += [
        "",
        f"**最终判定：{'GO' if any_go else 'NO-GO'}。**",
        "",
        "门槛要求同一个 projected 方法至少通过两个结构差异明显的数据集；单一数据集提升、仅优于 WL 对应实现、或不超过 relation-only 均不足以支持 KSVD 主线。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = build(DEFAULT_INPUTS)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(result), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
