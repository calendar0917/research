"""Does Beam8 add frozen-residual information after global graph statistics?"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data_tud import load_tud
from .run_attributed_beam8_bag_specificity_controls import (
    _bag,
    _delta,
    _incidence_rows,
    _pad_rows,
    _random_dictionary,
)
from .run_attributed_beam8_frozen_binding_shuffle_audit import (
    PARITY_TOLERANCE,
    _attach_cached,
    _cache_base,
    _cached_score,
    _select_cached_head_epoch,
    _train_cached_head,
)
from .run_attributed_beam8_frozen_gine_residual import ResidualHead, _train_base
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_classification import _normalize_nodes
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _encode,
    _fit_dictionary,
)
from .run_luyin14_edge_aware_joint import _attach_edge, _load_edge_pyg


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_CONDITIONAL_INCREMENT_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
REFERENCE = RESULT_DIR / "attributed_beam8_bag_specificity_controls_20260814.json"
DEFAULT_JSON = RESULT_DIR / "attributed_beam8_conditional_increment_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ATTRIBUTED_BEAM8_CONDITIONAL_INCREMENT_20260814.md"
SECOND_VARIANTS = (
    "GLOBAL_PLUS_BEAM8_FULL",
    "GLOBAL_PLUS_BEAM8_CODE_ONLY",
    "GLOBAL_PLUS_BEAM8_HIST_ONLY",
    "GLOBAL_PLUS_BEAM8_RANDOM_DICTIONARY",
)
VARIANTS = ("GINE_FROZEN", "GLOBAL_STATS_STAGE1", *SECOND_VARIANTS)


@torch.no_grad()
def _offset_cache(
    cached: Sequence[dict[str, torch.Tensor]],
    rows: Sequence[np.ndarray],
    head: ResidualHead,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Freeze a residual into cached logits while retaining the same node states."""
    head.eval()
    data = _attach_cached(cached, rows)
    output: list[dict[str, torch.Tensor]] = []
    for batch in DataLoader(data, batch_size=128, shuffle=False):
        batch = batch.to(device)
        delta = head(batch.x, batch.s, batch.batch)
        ptr = batch.ptr.detach().cpu().numpy()
        states = batch.x.detach().cpu()
        logits = (batch.base_logits + delta).detach().cpu()
        labels = batch.y.view(-1).detach().cpu()
        for index in range(int(batch.num_graphs)):
            output.append(
                {
                    "states": states[int(ptr[index]) : int(ptr[index + 1])].clone(),
                    "logits": logits[index : index + 1].clone(),
                    "label": labels[index : index + 1].clone(),
                }
            )
    if len(output) != len(cached):
        raise RuntimeError(f"offset cached {len(output)} graphs, expected {len(cached)}")
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference_units = {
        (int(unit["split_seed"]), int(unit["model_seed"]), int(unit["fold_index"])): unit
        for unit in reference["units"]
    }
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("the graph and PyG loaders disagree on labels")

    items, typed_graphs = [], []
    for index, (graph, label, node_features, data) in enumerate(zip(graphs, labels, features, raw)):
        typed = _typed_adjacency(data, edge_dim)
        typed_graphs.append(typed)
        items.append(
            prepare_attributed_beam_graph(
                index, graph, int(label), node_features, typed, edge_dim=edge_dim
            )
        )
        if (index + 1) % 500 == 0 or index + 1 == len(graphs):
            print(f"prepare {index + 1}/{len(graphs)}", flush=True)

    base_data = _attach_edge(raw, None)
    input_dim = int(raw[0].x.shape[1])
    units = []
    for split_seed in (3, 4):
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            split_fold_seed = split_seed * 1000 + fold
            inner_train, validation = train_test_split(
                train,
                test_size=0.2,
                stratify=labels[train],
                random_state=1729 + split_fold_seed,
            )
            dictionary = _fit_dictionary(
                items,
                train,
                n_atoms=24,
                sparsity=3,
                iterations=5,
                max_train_patches=3000,
                seed=split_fold_seed,
            )
            random_dictionary, random_mean = _random_dictionary(
                items,
                train,
                n_atoms=24,
                max_train_patches=3000,
                seed=split_fold_seed,
            )
            if not np.allclose(random_mean, dictionary["mean"]):
                raise RuntimeError("random and deterministic dictionaries use different train pools")

            deterministic_codes = [
                _encode(item, dictionary["initial"], dictionary["mean"], 3) for item in items
            ]
            random_codes = [_encode(item, random_dictionary, random_mean, 3) for item in items]
            token_families = {
                "GLOBAL_PLUS_BEAM8_FULL": [
                    np.concatenate([code, item.node_histograms], axis=1)
                    for code, item in zip(deterministic_codes, items)
                ],
                "GLOBAL_PLUS_BEAM8_CODE_ONLY": deterministic_codes,
                "GLOBAL_PLUS_BEAM8_HIST_ONLY": [item.node_histograms for item in items],
                "GLOBAL_PLUS_BEAM8_RANDOM_DICTIONARY": [
                    np.concatenate([code, item.node_histograms], axis=1)
                    for code, item in zip(random_codes, items)
                ],
            }
            raw_rows = {
                variant: _incidence_rows(items, typed_graphs, features, graphs, tokens)
                for variant, tokens in token_families.items()
            }
            full_dim = raw_rows["GLOBAL_PLUS_BEAM8_FULL"][0].shape[1]
            second_rows = {
                variant: _bag(_normalize_nodes(_pad_rows(rows, full_dim), train))
                for variant, rows in raw_rows.items()
            }
            global_raw = []
            for item, graph in zip(items, graphs):
                summary = np.concatenate([item.graph_features, item.stats])
                global_raw.append(np.repeat(summary[None, :], graph.n, axis=0))
            global_rows = _bag(_normalize_nodes(_pad_rows(global_raw, full_dim), train))

            for model_seed in (0, 1, 2):
                key = (split_seed, model_seed, fold)
                reference_unit = reference_units[key]
                base_epoch = int(reference_unit["selected_epochs"]["base"])
                seed = model_seed * 1000 + fold
                inner_base = _train_base(
                    base_data,
                    inner_train,
                    epochs=base_epoch,
                    seed=seed,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    classes=classes,
                    device=device,
                )
                full_base = _train_base(
                    base_data,
                    train,
                    epochs=base_epoch,
                    seed=seed,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    classes=classes,
                    device=device,
                )
                inner_cache = _cache_base(inner_base, base_data, device)
                full_cache = _cache_base(full_base, base_data, device)
                zero_rows = [np.zeros((graph.n, 1), dtype=np.float32) for graph in graphs]
                gine_score = _cached_score(
                    None, _attach_cached(full_cache, zero_rows), test, device
                )["balanced_accuracy"]
                reference_gine = float(reference_unit["scores"]["GINE_FROZEN"])
                gine_error = abs(gine_score - reference_gine)
                if gine_error > PARITY_TOLERANCE:
                    raise RuntimeError(f"GINE parity failed for {key}: {gine_error}")

                struct_dim = int(global_rows[0].shape[1])
                global_epoch = _select_cached_head_epoch(
                    _attach_cached(inner_cache, global_rows),
                    inner_train,
                    validation,
                    seed=seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                inner_global = _train_cached_head(
                    _attach_cached(inner_cache, global_rows),
                    inner_train,
                    epochs=global_epoch,
                    seed=seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                full_global = _train_cached_head(
                    _attach_cached(full_cache, global_rows),
                    train,
                    epochs=global_epoch,
                    seed=seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                inner_offset = _offset_cache(inner_cache, global_rows, inner_global, device)
                full_offset = _offset_cache(full_cache, global_rows, full_global, device)
                global_score = _cached_score(
                    None, _attach_cached(full_offset, zero_rows), test, device
                )["balanced_accuracy"]
                reference_global = float(reference_unit["scores"]["GLOBAL_STATS"])
                global_error = abs(global_score - reference_global)
                if global_error > PARITY_TOLERANCE:
                    raise RuntimeError(f"GLOBAL parity failed for {key}: {global_error}")

                scores = {
                    "GINE_FROZEN": gine_score,
                    "GLOBAL_STATS_STAGE1": global_score,
                }
                selected_epochs = {"base": base_epoch, "GLOBAL_STATS_STAGE1": global_epoch}
                for variant in SECOND_VARIANTS:
                    inner_data = _attach_cached(inner_offset, second_rows[variant])
                    full_data = _attach_cached(full_offset, second_rows[variant])
                    second_epoch = _select_cached_head_epoch(
                        inner_data,
                        inner_train,
                        validation,
                        seed=seed,
                        struct_dim=struct_dim,
                        classes=classes,
                        device=device,
                    )
                    second_head = _train_cached_head(
                        full_data,
                        train,
                        epochs=second_epoch,
                        seed=seed,
                        struct_dim=struct_dim,
                        classes=classes,
                        device=device,
                    )
                    scores[variant] = _cached_score(second_head, full_data, test, device)[
                        "balanced_accuracy"
                    ]
                    selected_epochs[variant] = second_epoch
                units.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "fold_index": fold,
                        "scores": scores,
                        "selected_epochs": selected_epochs,
                        "parity": {"gine_error": gine_error, "global_error": global_error},
                    }
                )
                print(
                    f"split={split_seed} model={model_seed} fold={fold} "
                    + " ".join(f"{name}={scores[name]:.4f}" for name in VARIANTS),
                    flush=True,
                )

    variants = {
        variant: {
            "mean": float(np.mean([unit["scores"][variant] for unit in units])),
            "std": float(np.std([unit["scores"][variant] for unit in units])),
        }
        for variant in VARIANTS
    }
    paired = {
        "full_vs_global": _delta(units, "GLOBAL_PLUS_BEAM8_FULL", "GLOBAL_STATS_STAGE1"),
        "full_vs_code_only": _delta(
            units, "GLOBAL_PLUS_BEAM8_FULL", "GLOBAL_PLUS_BEAM8_CODE_ONLY"
        ),
        "full_vs_hist_only": _delta(
            units, "GLOBAL_PLUS_BEAM8_FULL", "GLOBAL_PLUS_BEAM8_HIST_ONLY"
        ),
        "full_vs_random_dictionary": _delta(
            units,
            "GLOBAL_PLUS_BEAM8_FULL",
            "GLOBAL_PLUS_BEAM8_RANDOM_DICTIONARY",
        ),
    }
    full_global = paired["full_vs_global"]
    checks = {
        "increment": full_global["mean"] >= 0.005 and full_global["wins"] >= 12,
        "beats_hist_only": paired["full_vs_hist_only"]["mean"] >= 0.0025
        and paired["full_vs_hist_only"]["wins"] >= 11,
        "beats_random_dictionary": paired["full_vs_random_dictionary"]["mean"] >= 0.0025
        and paired["full_vs_random_dictionary"]["wins"] >= 11,
        "both_splits": all(value > 0 for value in full_global["split_means"].values()),
        "model_majority": int(
            np.sum(np.asarray(list(full_global["model_means"].values())) > 0)
        )
        >= 2,
        "parity": all(
            unit["parity"]["gine_error"] <= PARITY_TOLERANCE
            and unit["parity"]["global_error"] <= PARITY_TOLERANCE
            for unit in units
        ),
    }
    decision = (
        "BEAM8_ADDS_INFORMATION_BEYOND_GLOBAL_STATS_EXPLORATORY"
        if all(checks.values())
        else "BEAM8_INCREMENT_BEYOND_GLOBAL_STATS_NOT_ESTABLISHED"
    )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "config": {"split_seeds": [3, 4], "model_seeds": [0, 1, 2]},
        "units": units,
        "summary": {
            "variants": variants,
            "paired": paired,
            "checks": checks,
            "decision": decision,
        },
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Beam8 conditional increment after global statistics",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | balanced accuracy over 18 units |",
        "|---|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(f"| {variant} | {row['mean']:.4f} ± {row['std']:.4f} |")
    lines.extend(
        [
            "",
            "## Paired attribution",
            "",
            "| comparison | mean | W/T/L | split3/4 | model0/1/2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["paired"].items():
        split_text = " / ".join(
            f"{row['split_means'][str(seed)]:+.4f}" for seed in (3, 4)
        )
        model_text = " / ".join(
            f"{row['model_means'][str(seed)]:+.4f}" for seed in (0, 1, 2)
        )
        lines.append(
            f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {split_text} | {model_text} |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 第一阶段先冻结 GLOBAL_STATS，第二阶段才读取 Beam8，因此 FULL−GLOBAL 是条件增量。",
            "- 本轮是结果可见后的机制探索；通过时仍需未见 split 或外部 attributed TUD 确认。",
            "- 失败表示当前 Beam8 BAG 未证明具有普通图统计之外的分类价值。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
