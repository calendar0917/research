"""Audit whether the current WL patch vector loses useful exact structure.

The non-WL controls use nauty canonical labeling and retain the complete padded
adjacency matrix up to isomorphism.  The rooted variant additionally preserves
the sampler's distinguished patch center.  Before any dictionary learning, the
runner measures exact representation collisions and compares graph-level raw
patch means after controlling for simple graph statistics.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .run_real_structure_conditional import _fit_score_nested
from .run_real_structure_ksvd import (
    _atomic_json,
    _subset_per_class,
    degree_hist_features,
    graph_basic_features,
    prepare_graphs,
)
from .sampling_route import MatchedSamplingConfig, vectorize_collection

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "real_structure_nonwl_audit_20260730.json"
DEFAULT_MD = ROOT / "docs" / "REAL_STRUCTURE_NONWL_AUDIT_20260730.md"
CACHE_DIR = ROOT / "results" / "cache_real_structure_nonwl_20260730"

SPECS = {
    "IMDB-BINARY": {"dataset": "IMDB-BINARY", "variant": "cleaned", "method": "B0", "max_nodes": 12},
    "IMDB-MULTI": {"dataset": "IMDB-MULTI", "variant": "cleaned", "method": "B0", "max_nodes": 12},
    "REDDIT-BINARY": {"dataset": "REDDIT-BINARY", "variant": "raw", "method": "R2", "max_nodes": 24},
}


def _key(column: np.ndarray) -> bytes:
    return np.ascontiguousarray(column, dtype=np.float64).tobytes()


def _collision_stats(coarse_keys: list[bytes], exact_keys: list[bytes]) -> dict[str, Any]:
    buckets: dict[bytes, set[bytes]] = {}
    for coarse, exact in zip(coarse_keys, exact_keys):
        buckets.setdefault(coarse, set()).add(exact)
    ambiguous = {key for key, values in buckets.items() if len(values) > 1}
    occurrence_ambiguous = sum(key in ambiguous for key in coarse_keys)
    exact_in_ambiguous: set[bytes] = set()
    for key in ambiguous:
        exact_in_ambiguous.update(buckets[key])
    counts = [len(values) for values in buckets.values()]
    return {
        "n_occurrences": len(coarse_keys),
        "n_coarse_signatures": len(buckets),
        "n_exact_signatures": len(set(exact_keys)),
        "ambiguous_coarse_signatures": len(ambiguous),
        "occurrence_ambiguous_fraction": occurrence_ambiguous / max(len(coarse_keys), 1),
        "exact_types_in_ambiguous_fraction": len(exact_in_ambiguous) / max(len(set(exact_keys)), 1),
        "mean_exact_types_per_coarse": float(np.mean(counts)) if counts else 0.0,
        "max_exact_types_per_coarse": max(counts, default=0),
    }


def _cache_path(setting: str, n_patches: int, sampling_seed: int, limit: int | None) -> Path:
    safe = setting.lower().replace("-", "_")
    return CACHE_DIR / f"{safe}_p{n_patches}_seed{sampling_seed}_limit{limit}.pkl"


def build_representations(
    setting: str,
    graphs: list[Any],
    prepared: dict[str, Any],
    cfg: MatchedSamplingConfig,
    *,
    sampling_seed: int,
    limit_per_class: int | None,
    use_cache: bool,
) -> dict[str, Any]:
    path = _cache_path(setting, cfg.n_patches, sampling_seed, limit_per_class)
    if use_cache and path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    modes = ["wl", "canonical", "rooted_canonical"]
    Ys: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    keys: dict[str, list[bytes]] = {mode: [] for mode in modes}
    for i, (g, pc) in enumerate(zip(graphs, prepared["pcs"])):
        if i % 200 == 0:
            print(f"  canonicalized {i}/{len(graphs)} graphs", flush=True)
        for mode in modes:
            Y = prepared["Ys"][i] if mode == "wl" else vectorize_collection(g, pc, cfg.max_nodes, mode)
            Ys[mode].append(Y)
            keys[mode].extend(_key(Y[:, j]) for j in range(Y.shape[1]))
    obj = {"Ys": Ys, "keys": keys}
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj


def evaluate_raw(
    graphs: list[Any],
    y: np.ndarray,
    representations: dict[str, Any],
    *,
    n_splits: int,
    inner_splits: int,
    split_seeds: list[int],
) -> dict[str, Any]:
    stats = np.concatenate(
        [
            np.stack([graph_basic_features(g) for g in graphs]),
            np.stack([degree_hist_features(g) for g in graphs]),
        ],
        axis=1,
    )
    raw: dict[str, np.ndarray] = {}
    for mode, Ys in representations["Ys"].items():
        feature = np.stack([Y.mean(axis=1) for Y in Ys])
        feature /= np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
        raw[mode] = feature

    folds: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y)):
            fit_seed = split_seed * 1000 + fold
            score: dict[str, Any] = {}
            selected_C: dict[str, float] = {}
            blocks = {"stats": (stats[tr], stats[te])}
            for mode, feature in raw.items():
                blocks[f"stats_plus_{mode}_raw"] = (
                    np.concatenate([stats[tr], feature[tr]], axis=1),
                    np.concatenate([stats[te], feature[te]], axis=1),
                )
            for name, (Xtr, Xte) in blocks.items():
                score[name], selected_C[name] = _fit_score_nested(
                    Xtr, y[tr], Xte, y[te], fit_seed, inner_splits
                )
            folds.append({
                "split_seed": split_seed,
                "fold": fold,
                "score": score,
                "selected_C": selected_C,
            })
    metric = "balanced_accuracy"
    methods = sorted(folds[0]["score"])
    summary: dict[str, Any] = {}
    for method in methods:
        vals = np.asarray([f["score"][method][metric] for f in folds], dtype=np.float64)
        seed_means = []
        for seed in split_seeds:
            selected = [
                f["score"][method][metric]
                for f in folds if int(f["split_seed"]) == seed
            ]
            seed_means.append(float(np.mean(selected)))
        summary[method] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "values": vals.tolist(),
            "seed_means": seed_means,
        }
    wl = np.asarray(summary["stats_plus_wl_raw"]["values"])
    for mode in ("canonical", "rooted_canonical"):
        vals = np.asarray(summary[f"stats_plus_{mode}_raw"]["values"])
        delta = vals - wl
        seed_delta = [
            float(delta[i * n_splits:(i + 1) * n_splits].mean())
            for i in range(len(split_seeds))
        ]
        summary[f"{mode}_minus_wl"] = {
            "mean": float(delta.mean()),
            "wins": int(np.sum(delta > 1e-12)),
            "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "losses": int(np.sum(delta < -1e-12)),
            "seed_means": seed_delta,
        }
    return {"folds": folds, "summary": summary}


def render_markdown(result: dict[str, Any]) -> str:
    c = result["config"]
    lines = [
        "# 不使用 WL 的 exact canonical patch 审计（2026-07-30）",
        "",
        "> 目的：区分 KSVD 的失败是否可能由当前 WL histogram 的信息损失导致。Canonical 表示用 nauty 规范化完整邻接矩阵，不依赖节点编号；rooted canonical 额外保留 sampler center。",
        "",
        "## 表示定义",
        "",
        "- `WL`：当前 degree histogram + degree-pair histogram + 3轮 hashed 1-WL histogram。",
        "- `canonical`：完整 induced adjacency 的 exact canonical labeling + node mask；不使用 WL 作为最终特征。",
        "- `rooted canonical`：canonical graph 同时把 patch center 作为 singleton color，保留中心在子图中的角色。",
        "- 多指标排序不是严格替代：只要仍有 tie 并由 node id 决定，表示就不是置换不变；canonical labeling 才能在保留完整邻接的同时解决该问题。",
        "",
        f"分类协议：{c['n_splits']}-fold × seeds {c['split_seeds']}，inner {c['inner_splits']}-fold；此处先比较 raw patch mean，不训练 KSVD。",
        "",
    ]
    for tag, block in result["experiments"].items():
        col = block["collisions"]
        s = block["raw_evaluation"]["summary"]
        lines += [f"## {tag}", "", "### 信息损失审计", ""]
        lines += [
            "| coarse → exact | coarse signatures | exact signatures | ambiguous occurrences | exact types affected | max exact/coarse |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, label in [
            ("wl_to_unrooted", "WL → exact unrooted"),
            ("unrooted_to_rooted", "unrooted → exact rooted"),
            ("wl_to_rooted", "WL → exact rooted"),
        ]:
            x = col[name]
            lines.append(
                f"| {label} | {x['n_coarse_signatures']} | {x['n_exact_signatures']} | {x['occurrence_ambiguous_fraction']:.3f} | {x['exact_types_in_ambiguous_fraction']:.3f} | {x['max_exact_types_per_coarse']} |"
            )
        lines += ["", "### 控制 stats 后的 raw patch mean", ""]
        lines += [
            "| 表示 | balanced accuracy | seed means |",
            "|---|---:|---:|",
            f"| stats only | {s['stats']['mean']:.4f} | {[round(x,4) for x in s['stats']['seed_means']]} |",
            f"| WL raw | {s['stats_plus_wl_raw']['mean']:.4f} | {[round(x,4) for x in s['stats_plus_wl_raw']['seed_means']]} |",
            f"| canonical raw | {s['stats_plus_canonical_raw']['mean']:.4f} | {[round(x,4) for x in s['stats_plus_canonical_raw']['seed_means']]} |",
            f"| rooted canonical raw | {s['stats_plus_rooted_canonical_raw']['mean']:.4f} | {[round(x,4) for x in s['stats_plus_rooted_canonical_raw']['seed_means']]} |",
            "",
        ]
        for mode in ("canonical", "rooted_canonical"):
            d = s[f"{mode}_minus_wl"]
            lines.append(
                f"- `{mode} − WL`：{d['mean']:+.4f}，fold W/T/L={d['wins']}/{d['ties']}/{d['losses']}，seed means={[round(x,4) for x in d['seed_means']]}。"
            )
        lines.append("")
    lines += [
        "## 决策规则",
        "",
        "- 若 canonical 显著减少 collision，且 raw 表征跨 seeds 稳定优于 WL，则再运行完整 projected-KSVD。",
        "- 若 exact canonical 仍不优于 WL，则不能继续把 KSVD 失败主要归因于 WL 信息损失。",
        "- rooted 优于 unrooted 时，问题更可能是当前表示丢失 patch center，而不只是 1-WL 表达力不足。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", nargs="+", choices=sorted(SPECS), default=list(SPECS))
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sampling-seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--n-patches", type=int, default=16)
    ap.add_argument("--walk-length", type=int, default=24)
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
            "split_seeds": args.split_seeds,
            "sampling_seed": args.sampling_seed,
            "n_splits": args.n_splits,
            "inner_splits": args.inner_splits,
            "n_patches": args.n_patches,
            "walk_length": args.walk_length,
            "limit_per_class": args.limit_per_class,
            "pynauty_version": "2.8.8.1",
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
        tag = f"{spec['dataset']}/{spec['variant']}/{spec['method']}/max_nodes={spec['max_nodes']}"
        print(f"[non-WL audit] {tag}: n={len(graphs)}", flush=True)
        prepared = prepare_graphs(
            spec["dataset"], cleaned, graphs, spec["method"], cfg,
            args.sampling_seed, args.limit_per_class, use_cache=not args.no_cache,
        )
        representations = build_representations(
            setting, graphs, prepared, cfg,
            sampling_seed=args.sampling_seed,
            limit_per_class=args.limit_per_class,
            use_cache=not args.no_cache,
        )
        keys = representations["keys"]
        block = {
            "dataset_meta": meta,
            "n_graphs": len(graphs),
            "class_counts": {str(int(c)): int(np.sum(y == c)) for c in np.unique(y)},
            "collisions": {
                "wl_to_unrooted": _collision_stats(keys["wl"], keys["canonical"]),
                "unrooted_to_rooted": _collision_stats(keys["canonical"], keys["rooted_canonical"]),
                "wl_to_rooted": _collision_stats(keys["wl"], keys["rooted_canonical"]),
            },
            "raw_evaluation": evaluate_raw(
                graphs, y, representations,
                n_splits=args.n_splits,
                inner_splits=args.inner_splits,
                split_seeds=args.split_seeds,
            ),
        }
        result["experiments"][tag] = block
        _atomic_json(args.output_json, result)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
