"""Strict full-node-attribute prescreen before applying Beam8 to TU ENZYMES."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from .data_tud import load_tud, node_feature_readout
from .run_attributed_beam8_frozen_binding_shuffle_audit import (
    _attach_cached,
    _cached_score,
    _select_cached_head_epoch,
    _train_cached_head,
)
from .run_attributed_beam8_node_incidence_classification import _normalize_nodes
from .run_beam8_nci1_chain_classification import ROOT, _atomic_json, _atomic_text
from .run_real_structure_ksvd import graph_basic_features


PROTOCOL = "tracks/ksvd/docs/KSVD_ENZYMES_FULL_ATTRIBUTE_PRESCREEN_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_ROOT = ROOT / "data/TUD"
DEFAULT_JSON = RESULT_DIR / "enzymes_full_attribute_prescreen_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ENZYMES_FULL_ATTRIBUTE_PRESCREEN_20260814.md"
VARIANTS = (
    "GLOBAL_STATS_LINEAR",
    "GIN_LABEL_ONLY",
    "GIN_FULL_ATTRIBUTES",
    "GIN_FULL_PLUS_GLOBAL",
)


def _mlp(input_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
    )


class GINClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        classes: int,
        *,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            width = input_dim if layer == 0 else hidden
            self.convs.append(GINConv(_mlp(width, hidden), train_eps=False))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.predictors = nn.ModuleList(
            [nn.Linear(input_dim, classes)]
            + [nn.Linear(hidden, classes) for _ in range(layers)]
        )

    def forward_states(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        h = data.x
        representations = [h]
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, data.edge_index)))
            representations.append(h)
        score = 0.0
        for predictor, representation in zip(self.predictors, representations):
            logits = predictor(global_add_pool(representation, data.batch))
            score = score + F.dropout(logits, p=self.dropout, training=self.training)
        return score, h

    def forward(self, data: Data) -> torch.Tensor:
        return self.forward_states(data)[0]


def _load_raw(root: Path) -> tuple[list[Data], np.ndarray, int, int]:
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root=str(root), name="ENZYMES", use_node_attr=True)
    labels = np.asarray([int(data.y.view(-1)[0]) for data in dataset], dtype=np.int64)
    unique = sorted(set(labels.tolist()))
    remap = {value: index for index, value in enumerate(unique)}
    labels = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    raw = []
    for index, data in enumerate(dataset):
        raw.append(
            Data(
                x=data.x.float(),
                edge_index=data.edge_index,
                y=torch.tensor([int(labels[index])], dtype=torch.long),
                num_nodes=int(data.num_nodes),
            )
        )
    return raw, labels, int(dataset.num_node_attributes), int(dataset.num_node_labels)


def _normalize_raw(
    raw: Sequence[Data], train: Sequence[int], *, attribute_dim: int, label_only: bool
) -> list[Data]:
    if label_only:
        return [
            Data(
                x=data.x[:, attribute_dim:].clone(),
                edge_index=data.edge_index,
                y=data.y.clone(),
                num_nodes=data.num_nodes,
            )
            for data in raw
        ]
    train_values = torch.cat([raw[int(index)].x[:, :attribute_dim] for index in train], dim=0)
    mean = train_values.mean(dim=0, keepdim=True)
    scale = train_values.std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    output = []
    for data in raw:
        continuous = (data.x[:, :attribute_dim] - mean) / scale
        labels = data.x[:, attribute_dim:]
        output.append(
            Data(
                x=torch.cat([continuous, labels], dim=1),
                edge_index=data.edge_index,
                y=data.y.clone(),
                num_nodes=data.num_nodes,
            )
        )
    return output


@torch.no_grad()
def _score(
    model: GINClassifier,
    data: Sequence[Data],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    prediction, labels = [], []
    for batch in DataLoader([data[int(index)] for index in indices], batch_size=128):
        batch = batch.to(device)
        prediction.extend(model(batch).argmax(1).cpu().numpy().tolist())
        labels.extend(batch.y.view(-1).cpu().numpy().tolist())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
    }


def _new_model(input_dim: int, classes: int, seed: int, device: torch.device) -> GINClassifier:
    torch.manual_seed(seed)
    return GINClassifier(input_dim, classes).to(device)


def _train(
    data: Sequence[Data],
    indices: Sequence[int],
    *,
    epochs: int,
    seed: int,
    classes: int,
    device: torch.device,
) -> GINClassifier:
    model = _new_model(int(data[0].x.shape[1]), classes, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 11)
    loader = DataLoader(
        [data[int(index)] for index in indices],
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    for _epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
    return model


def _select_epoch(
    data: Sequence[Data],
    train: Sequence[int],
    validation: Sequence[int],
    *,
    seed: int,
    classes: int,
    device: torch.device,
) -> int:
    model = _new_model(int(data[0].x.shape[1]), classes, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 11)
    loader = DataLoader(
        [data[int(index)] for index in train],
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    best_epoch, best_score, stale = 1, -1.0, 0
    for epoch in range(1, 101):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
        score = _score(model, data, validation, device)["balanced_accuracy"]
        if score > best_score + 1e-12:
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= 20:
            break
    return best_epoch


@torch.no_grad()
def _cache(
    model: GINClassifier, data: Sequence[Data], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    model.eval()
    cached: list[dict[str, torch.Tensor]] = []
    for batch in DataLoader(list(data), batch_size=128, shuffle=False):
        batch = batch.to(device)
        logits, states = model.forward_states(batch)
        ptr = batch.ptr.detach().cpu().numpy()
        states = states.detach().cpu()
        logits = logits.detach().cpu()
        labels = batch.y.view(-1).detach().cpu()
        for index in range(int(batch.num_graphs)):
            cached.append(
                {
                    "states": states[int(ptr[index]) : int(ptr[index + 1])].clone(),
                    "logits": logits[index : index + 1].clone(),
                    "label": labels[index : index + 1].clone(),
                }
            )
    return cached


def _delta(units: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray([unit["scores"][left] - unit["scores"][right] for unit in units])
    split_means = {
        str(seed): float(np.mean([value for value, unit in zip(values, units) if unit["split_seed"] == seed]))
        for seed in (0, 1, 2)
    }
    model_means = {
        str(seed): float(np.mean([value for value, unit in zip(values, units) if unit["model_seed"] == seed]))
        for seed in (0, 1, 2)
    }
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "split_means": split_means,
        "model_means": model_means,
        "values": values.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    graphs, labels, features, metadata = load_tud(
        "ENZYMES", args.dataset_root, use_node_attr=True
    )
    raw, raw_labels, attribute_dim, label_dim = _load_raw(args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("load_tud and PyG labels disagree")
    if attribute_dim != 18 or label_dim != 3 or features[0].shape[1] != 21:
        raise RuntimeError(
            f"unexpected ENZYMES dimensions: attr={attribute_dim}, label={label_dim}, total={features[0].shape[1]}"
        )

    summaries = np.stack(
        [
            np.concatenate([node_feature_readout(node_features), graph_basic_features(graph)])
            for graph, node_features in zip(graphs, features)
        ]
    )
    global_raw_rows = [
        np.repeat(summary[None, :], graph.n, axis=0)
        for summary, graph in zip(summaries, graphs)
    ]
    classes = int(len(np.unique(labels)))
    units = []
    for split_seed in (0, 1, 2):
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            inner_train, validation = train_test_split(
                train,
                test_size=0.2,
                stratify=labels[train],
                random_state=1729 + split_seed * 1000 + fold,
            )
            linear = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=split_seed * 1000 + fold,
                ),
            )
            linear.fit(summaries[train], labels[train])
            global_linear = float(
                balanced_accuracy_score(labels[test], linear.predict(summaries[test]))
            )

            label_inner = _normalize_raw(
                raw, inner_train, attribute_dim=attribute_dim, label_only=True
            )
            label_full = _normalize_raw(raw, train, attribute_dim=attribute_dim, label_only=True)
            full_inner = _normalize_raw(
                raw, inner_train, attribute_dim=attribute_dim, label_only=False
            )
            full_outer = _normalize_raw(raw, train, attribute_dim=attribute_dim, label_only=False)
            inner_global_rows = _normalize_nodes(global_raw_rows, inner_train)
            outer_global_rows = _normalize_nodes(global_raw_rows, train)

            for model_seed in (0, 1, 2):
                seed = model_seed * 1000 + fold
                label_epoch = _select_epoch(
                    label_inner,
                    inner_train,
                    validation,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                label_model = _train(
                    label_full,
                    train,
                    epochs=label_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                label_score = _score(label_model, label_full, test, device)[
                    "balanced_accuracy"
                ]

                full_epoch = _select_epoch(
                    full_inner,
                    inner_train,
                    validation,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                inner_model = _train(
                    full_inner,
                    inner_train,
                    epochs=full_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                full_model = _train(
                    full_outer,
                    train,
                    epochs=full_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                full_score = _score(full_model, full_outer, test, device)[
                    "balanced_accuracy"
                ]
                inner_cache = _cache(inner_model, full_inner, device)
                outer_cache = _cache(full_model, full_outer, device)
                inner_data = _attach_cached(inner_cache, inner_global_rows)
                outer_data = _attach_cached(outer_cache, outer_global_rows)
                struct_dim = int(outer_data[0].s.shape[1])
                residual_epoch = _select_cached_head_epoch(
                    inner_data,
                    inner_train,
                    validation,
                    seed=seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                head = _train_cached_head(
                    outer_data,
                    train,
                    epochs=residual_epoch,
                    seed=seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                combined_score = _cached_score(head, outer_data, test, device)[
                    "balanced_accuracy"
                ]
                scores = {
                    "GLOBAL_STATS_LINEAR": global_linear,
                    "GIN_LABEL_ONLY": label_score,
                    "GIN_FULL_ATTRIBUTES": full_score,
                    "GIN_FULL_PLUS_GLOBAL": combined_score,
                }
                units.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "fold_index": fold,
                        "scores": scores,
                        "selected_epochs": {
                            "label_gin": label_epoch,
                            "full_gin": full_epoch,
                            "global_residual": residual_epoch,
                        },
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
        "full_vs_global": _delta(units, "GIN_FULL_ATTRIBUTES", "GLOBAL_STATS_LINEAR"),
        "full_vs_label_only": _delta(units, "GIN_FULL_ATTRIBUTES", "GIN_LABEL_ONLY"),
        "combined_vs_global": _delta(units, "GIN_FULL_PLUS_GLOBAL", "GLOBAL_STATS_LINEAR"),
        "combined_vs_full": _delta(units, "GIN_FULL_PLUS_GLOBAL", "GIN_FULL_ATTRIBUTES"),
    }
    sizes = np.asarray([graph.n for graph in graphs])
    checks = {
        "full_beats_global": paired["full_vs_global"]["mean"] >= 0.03
        and paired["full_vs_global"]["wins"] >= 18,
        "continuous_attributes_help": paired["full_vs_label_only"]["mean"] >= 0.03
        and paired["full_vs_label_only"]["wins"] >= 18,
        "all_split_means_positive": all(
            value > 0 for value in paired["full_vs_global"]["split_means"].values()
        ),
        "model_majority_positive": int(
            np.sum(np.asarray(list(paired["full_vs_global"]["model_means"].values())) > 0)
        )
        >= 2,
        "combined_beats_global": paired["combined_vs_global"]["mean"] >= 0.03,
        "dimensions": attribute_dim == 18 and label_dim == 3,
        "multi_patch_size_proxy": float(np.mean(sizes > 16)) >= 0.50,
    }
    decision = (
        "ENZYMES_ADVANCE_TO_BEAM8_ATTRIBUTED_SCREEN"
        if all(checks.values())
        else "ENZYMES_DO_NOT_ADVANCE_TO_BEAM8"
    )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata
        | {
            "continuous_attribute_dim": attribute_dim,
            "discrete_label_dim": label_dim,
            "fraction_nodes_gt16": float(np.mean(sizes > 16)),
        },
        "config": {"split_seeds": [0, 1, 2], "model_seeds": [0, 1, 2]},
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
        "# TU ENZYMES full-node-attribute prescreen",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | balanced accuracy over 27 units |",
        "|---|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(f"| {variant} | {row['mean']:.4f} ± {row['std']:.4f} |")
    lines.extend(
        [
            "",
            "## Paired screening",
            "",
            "| comparison | mean | W/T/L | split0/1/2 | model0/1/2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["paired"].items():
        split_text = " / ".join(
            f"{row['split_means'][str(seed)]:+.4f}" for seed in (0, 1, 2)
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
            "- GIN 读取 18 continuous + 3 discrete；连续属性 train-only 标准化。",
            "- 后续 canonicalization 只能读取 3 维离散 labels，不能把连续属性 argmax 当颜色。",
            "- 本轮只决定是否进入低容量 Beam8 matched controls，不构成 Beam8 分类证据。",
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
