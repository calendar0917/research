"""Audit whether a real-data patch sampler produces a meaningful vocabulary substrate.

A dictionary learner cannot discover rich motifs if the sampled patch population
collapses to edges, cliques, or the hard maximum-size cap.  This label-free audit
reports patch size/shape distributions and within-graph repetition before any
KSVD fitting.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .data_tud import load_tud
from .run_real_structure_atom_audit import patch_signature
from .run_real_structure_ksvd import _atomic_json, prepare_graphs
from .sampling_route import MatchedSamplingConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "real_structure_patch_audit_20260730.json"
DEFAULT_MD = ROOT / "docs" / "REAL_STRUCTURE_PATCH_AUDIT_20260730.md"


def audit(graphs: list[Any], pcs: list[Any], max_nodes: int) -> dict[str, Any]:
    sizes: list[int] = []
    edges: list[int] = []
    densities: list[float] = []
    signatures: list[str] = []
    per_graph_unique_fraction: list[float] = []
    clique = tree = edge_only = capped = 0
    for g, pc in zip(graphs, pcs):
        graph_sigs: list[str] = []
        for nodes in pc.node_sets:
            sig = patch_signature(g, nodes)
            n, m = int(sig["n"]), int(sig["m"])
            key = str(sig["key"])
            sizes.append(n)
            edges.append(m)
            densities.append(2.0 * m / max(n * (n - 1), 1))
            signatures.append(key)
            graph_sigs.append(key)
            clique += int(m == n * (n - 1) // 2)
            tree += int(n > 0 and m == n - 1)
            edge_only += int(n == 2 and m == 1)
            capped += int(n == max_nodes)
        if graph_sigs:
            per_graph_unique_fraction.append(len(set(graph_sigs)) / len(graph_sigs))
    count = Counter(signatures)
    probs = np.asarray(list(count.values()), dtype=np.float64) / max(len(signatures), 1)
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-15))).sum())
    normalized_entropy = entropy / max(np.log(max(len(count), 2)), 1e-12)
    arr = np.asarray(sizes, dtype=np.float64)
    return {
        "n_patches": len(signatures),
        "size": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "q25": float(np.quantile(arr, 0.25)),
            "median": float(np.quantile(arr, 0.5)),
            "q75": float(np.quantile(arr, 0.75)),
            "q90": float(np.quantile(arr, 0.9)),
            "max": int(arr.max()),
            "counts": {str(int(k)): int(v) for k, v in zip(*np.unique(arr.astype(int), return_counts=True))},
        },
        "mean_edges": float(np.mean(edges)),
        "mean_density": float(np.mean(densities)),
        "edge_only_fraction": edge_only / max(len(signatures), 1),
        "clique_fraction": clique / max(len(signatures), 1),
        "tree_fraction": tree / max(len(signatures), 1),
        "cap_fraction": capped / max(len(signatures), 1),
        "n_unique_signatures": len(count),
        "signature_entropy": entropy,
        "normalized_signature_entropy": normalized_entropy,
        "top10_signature_mass": sum(v for _, v in count.most_common(10)) / max(len(signatures), 1),
        "mean_within_graph_unique_fraction": float(np.mean(per_graph_unique_fraction)),
        "most_common_signatures": count.most_common(10),
    }


def render(result: dict[str, Any]) -> str:
    c = result["config"]
    lines = [
        "# 真实纯结构数据的 patch substrate 审计（2026-07-30）",
        "",
        "> 在解释 KSVD 前，先检查 sampler 是否给出了足够丰富的 motif population。若 patch 主要是单边、clique 或硬截断样本，字典失败不能单独归因于 KSVD，字典成功也可能只是规模/度原型量化。",
        "",
        f"每图最多 {c['n_patches']} patches；每 patch 最多 {c['max_nodes']} nodes；sampling seed={c['sampling_seed']}。",
        "",
        "| 设置 | patches | size mean/median | edge-only | clique | tree | at cap | unique signatures | top-10 mass | within-graph unique |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, s in result["experiments"].items():
        lines.append(
            f"| {tag} | {s['n_patches']} | {s['size']['mean']:.2f}/{s['size']['median']:.1f} | "
            f"{s['edge_only_fraction']:.3f} | {s['clique_fraction']:.3f} | {s['tree_fraction']:.3f} | "
            f"{s['cap_fraction']:.3f} | {s['n_unique_signatures']} | {s['top10_signature_mass']:.3f} | "
            f"{s['mean_within_graph_unique_fraction']:.3f} |"
        )
    lines += ["", "## 详细分布", ""]
    for tag, s in result["experiments"].items():
        lines += [
            f"### {tag}",
            "",
            f"- size counts：`{s['size']['counts']}`；q75={s['size']['q75']:.1f}，q90={s['size']['q90']:.1f}。",
            f"- signature entropy={s['signature_entropy']:.3f}，normalized={s['normalized_signature_entropy']:.3f}。",
            f"- most common signatures：`{s['most_common_signatures'][:5]}`。",
            "",
        ]
    lines += [
        "## 判断口径",
        "",
        "1. edge-only/clique fraction 高：vocabulary 容易退化为 patch size/degree prototypes。",
        "2. cap fraction 高：结果对 `max_nodes` 敏感，需做尺度复核。",
        "3. within-graph unique fraction 低：增加 patch 数主要产生重复样本，不会增加结构覆盖。",
        "4. 该审计无标签，只判断 sampler 是否为 vocabulary learning 提供了合理 substrate。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", nargs="+", required=True, help="DATASET:variant:method, e.g. IMDB-BINARY:cleaned:B0")
    ap.add_argument("--n-patches", type=int, default=16)
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--walk-length", type=int, default=24)
    ap.add_argument("--sampling-seed", type=int, default=0)
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
            **{k: v for k, v in vars(args).items() if k not in {"output_json", "output_md"}},
        },
        "experiments": {},
    }
    for spec in args.settings:
        dataset, variant, method = spec.split(":")
        cleaned = variant == "cleaned"
        graphs, _y, _x, meta = load_tud(dataset, cleaned=cleaned, structure_only=True)
        prepared = prepare_graphs(dataset, cleaned, graphs, method, cfg, args.sampling_seed, None, use_cache=True)
        tag = f"{dataset}/{variant}/{method}"
        print(f"[patch-audit] {tag}", flush=True)
        block = audit(graphs, prepared["pcs"], args.max_nodes)
        block["dataset_meta"] = meta
        result["experiments"][tag] = block
        _atomic_json(args.output_json, result)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render(result), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
