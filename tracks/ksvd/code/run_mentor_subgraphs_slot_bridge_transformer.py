"""Slot-level shared-node bridge Transformer on mentor 50-node subgraphs.

Each Beam8 patch is kept as eight canonical slot tokens instead of one pooled
descriptor.  A target patch is removed completely.  Visible patches are
encoded locally, then globally, and exact repeated-node occurrences transfer
slot states back into the target slots.  The decoder predicts the target
patch's full upper-triangular adjacency.

Matched controls delete the bridge or cyclically assign the exact bridge-state
multiset to the wrong target slots.  No source IDs are embedded, no graph
labels, KSVD, GIN or GINE are used.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data_mentor_subgraphs import (
    default_cache_path,
    default_source_path,
    density_stratum,
    load_bundle,
    select_pilot_indices,
)
from .mentor_grouped_splits import audit_fold_partition, balanced_group_folds
from .overlap_stitching import CoverExample
from .run_beam8_coverage_operating_point_audit import GEOMETRIES
from .run_mentor_subgraphs_patch_transformer import (
    COVER_SEED,
    GEOMETRY,
    MAXIMUM_PATCHES,
    N_SPLITS,
    PILOT_SEED,
    SPLIT_SEED,
    VIEW,
    prepare_examples,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = (
    ROOT
    / "tracks/ksvd/results/mentor_subgraphs/slot_bridge_transformer_pilot_20260817.json"
)
DEFAULT_REPORT = (
    ROOT
    / "tracks/ksvd/results/mentor_subgraphs/SLOT_BRIDGE_TRANSFORMER_PILOT_20260817.md"
)
PROTOCOL = "tracks/ksvd/docs/KSVD_MENTOR_SLOT_BRIDGE_TRANSFORMER_PROTOCOL_20260817.md"
BRANCHES = ("NO_BRIDGE", "TRUE_BRIDGE", "SHUFFLED_BRIDGE")


@dataclass(frozen=True)
class SlotBridgeSample:
    graph_index: int
    stratum: str
    target_index: int
    context_features: np.ndarray  # context patches, slots, features
    bridge_weights: np.ndarray  # target slots, flattened context slots
    target_edges: np.ndarray


def slot_features(example: CoverExample, patch_size: int) -> np.ndarray:
    rows = []
    for patch in example.cover.patches:
        adjacency = np.asarray(patch.adjacency, dtype=np.float32)
        if adjacency.shape != (patch_size, patch_size):
            raise ValueError("slot bridge requires fixed-size patches")
        degree = adjacency.sum(axis=1, keepdims=True) / max(patch_size - 1, 1)
        center = np.asarray(
            [[float(int(node) == int(patch.center))] for node in patch.node_ids],
            dtype=np.float32,
        )
        rows.append(np.concatenate([adjacency, degree, center], axis=1))
    return np.stack(rows, axis=0)


def target_upper_edges(example: CoverExample, target_index: int) -> np.ndarray:
    adjacency = np.asarray(example.cover.patches[target_index].adjacency, dtype=np.float32)
    left, right = np.triu_indices(adjacency.shape[0], k=1)
    return adjacency[left, right].astype(np.float32)


def exact_bridge_weights(
    example: CoverExample, target_index: int, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    context_indices = np.asarray(
        [index for index in range(len(example.cover.patches)) if index != target_index],
        dtype=np.int64,
    )
    target = example.cover.patches[target_index]
    weights = np.zeros(
        (patch_size, len(context_indices) * patch_size), dtype=np.float32
    )
    for target_slot, node in enumerate(target.node_ids):
        for context_position, patch_index in enumerate(context_indices):
            context = example.cover.patches[int(patch_index)]
            for context_slot, context_node in enumerate(context.node_ids):
                if int(context_node) == int(node):
                    weights[
                        target_slot, context_position * patch_size + context_slot
                    ] = 1.0
        total = float(weights[target_slot].sum())
        if total > 0.0:
            weights[target_slot] /= total
    return context_indices, weights


def _shuffle_bridge_rows(
    weights: np.ndarray, *, target_index: int, graph_index: int
) -> np.ndarray:
    if weights.shape[0] < 2:
        return weights.copy()
    shift = 1 + (int(target_index) + int(graph_index)) % (weights.shape[0] - 1)
    return np.roll(weights, shift=shift, axis=0).copy()


def build_samples(
    examples: Sequence[CoverExample], *, branch: str, patch_size: int
) -> list[SlotBridgeSample]:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch: {branch}")
    samples = []
    for example in examples:
        features = slot_features(example, patch_size)
        for target_index in range(features.shape[0]):
            context_indices, bridge = exact_bridge_weights(
                example, target_index, patch_size
            )
            if branch == "NO_BRIDGE":
                bridge = np.zeros_like(bridge)
            elif branch == "SHUFFLED_BRIDGE":
                bridge = _shuffle_bridge_rows(
                    bridge,
                    target_index=target_index,
                    graph_index=int(example.graph_index),
                )
            samples.append(
                SlotBridgeSample(
                    graph_index=int(example.graph_index),
                    stratum=str(example.family),
                    target_index=int(target_index),
                    context_features=features[context_indices].astype(np.float32),
                    bridge_weights=bridge.astype(np.float32),
                    target_edges=target_upper_edges(example, target_index),
                )
            )
    return samples


def _import_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required") from exc
    return torch, nn, DataLoader


def make_collate(torch: Any, patch_size: int):
    def collate(batch: Sequence[SlotBridgeSample]) -> dict[str, Any]:
        max_context = max(sample.context_features.shape[0] for sample in batch)
        feature_dim = batch[0].context_features.shape[2]
        edge_dim = batch[0].target_edges.shape[0]
        context = np.zeros(
            (len(batch), max_context, patch_size, feature_dim), dtype=np.float32
        )
        valid = np.zeros((len(batch), max_context), dtype=bool)
        bridge = np.zeros(
            (len(batch), patch_size, max_context * patch_size), dtype=np.float32
        )
        targets = np.zeros((len(batch), edge_dim), dtype=np.float32)
        weights = np.zeros(len(batch), dtype=np.float32)
        graph_indices = np.zeros(len(batch), dtype=np.int64)
        strata: list[str] = []
        for row, sample in enumerate(batch):
            count = sample.context_features.shape[0]
            context[row, :count] = sample.context_features
            valid[row, :count] = True
            bridge[row, :, : count * patch_size] = sample.bridge_weights
            targets[row] = sample.target_edges
            weights[row] = 1.0 / float(count + 1)
            graph_indices[row] = sample.graph_index
            strata.append(sample.stratum)
        return {
            "context": torch.from_numpy(context),
            "valid": torch.from_numpy(valid),
            "bridge": torch.from_numpy(bridge),
            "targets": torch.from_numpy(targets),
            "weights": torch.from_numpy(weights),
            "graph_indices": graph_indices,
            "strata": strata,
        }

    return collate


def build_model(
    nn: Any,
    torch: Any,
    *,
    patch_size: int,
    feature_dim: int,
    hidden: int,
    heads: int,
    local_layers: int,
    global_layers: int,
    dropout: float,
) -> Any:
    if hidden % heads:
        raise ValueError("hidden size must be divisible by heads")
    edge_pairs = tuple(zip(*np.triu_indices(patch_size, k=1)))

    class SlotBridgeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.slot_input = nn.Linear(feature_dim, hidden)
            self.slot_embedding = nn.Embedding(patch_size, hidden)
            local_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=4 * hidden,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.local = nn.TransformerEncoder(local_layer, num_layers=local_layers)
            global_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=4 * hidden,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.global_encoder = nn.TransformerEncoder(
                global_layer, num_layers=global_layers
            )
            self.target_mask = nn.Parameter(torch.zeros(hidden))
            nn.init.normal_(self.target_mask, std=0.02)
            self.target_fuse = nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.edge_decoder = nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, context: Any, valid: Any, bridge: Any) -> Any:
            batch, patch_count, slots, _features = context.shape
            slot_ids = torch.arange(slots, device=context.device)
            values = self.slot_input(context) + self.slot_embedding(slot_ids)[
                None, None, :, :
            ]
            values = self.local(values.reshape(batch * patch_count, slots, hidden))
            values = values.reshape(batch, patch_count, slots, hidden)
            values = values * valid[:, :, None, None]

            patch_values = values.mean(dim=2)
            patch_values = self.global_encoder(
                patch_values, src_key_padding_mask=~valid
            )
            patch_values = patch_values * valid[:, :, None]
            global_context = patch_values.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)

            flat_slots = values.reshape(batch, patch_count * slots, hidden)
            bridge_state = torch.bmm(bridge, flat_slots)
            target_slot_embedding = self.slot_embedding(slot_ids)[None, :, :].expand(
                batch, -1, -1
            )
            global_slots = global_context[:, None, :].expand(-1, slots, -1)
            mask_slots = self.target_mask[None, None, :].expand(batch, slots, -1)
            target_slots = self.target_fuse(
                torch.cat([target_slot_embedding + mask_slots, bridge_state, global_slots], dim=-1)
            )
            pair_rows = []
            for left, right in edge_pairs:
                left_value = target_slots[:, int(left)]
                right_value = target_slots[:, int(right)]
                pair_rows.append(
                    self.edge_decoder(
                        torch.cat(
                            [left_value, right_value, left_value * right_value], dim=-1
                        )
                    )
                )
            return torch.cat(pair_rows, dim=1)

    return SlotBridgeTransformer()


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def train_and_evaluate(
    train_samples: Sequence[SlotBridgeSample],
    test_samples: Sequence[SlotBridgeSample],
    *,
    patch_size: int,
    seed: int,
    hidden: int,
    heads: int,
    local_layers: int,
    global_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    torch, nn, DataLoader = _import_torch()
    _seed_everything(torch, seed)
    collate = make_collate(torch, patch_size)
    generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        list(train_samples),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        list(test_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = build_model(
        nn,
        torch,
        patch_size=patch_size,
        feature_dim=train_samples[0].context_features.shape[2],
        hidden=hidden,
        heads=heads,
        local_layers=local_layers,
        global_layers=global_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positive = sum(float(sample.target_edges.sum()) for sample in train_samples)
    total = sum(float(sample.target_edges.size) for sample in train_samples)
    negative = total - positive
    pos_weight = torch.tensor(negative / max(positive, 1.0), dtype=torch.float32)
    curve = []
    for _epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["context"], batch["valid"], batch["bridge"])
            element = nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch["targets"],
                pos_weight=pos_weight,
                reduction="none",
            ).mean(dim=1)
            loss = (element * batch["weights"]).sum() / batch["weights"].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float((element.detach() * batch["weights"]).sum())
            weight_sum += float(batch["weights"].sum())
        curve.append(loss_sum / max(weight_sum, 1e-12))

    predictions = []
    targets = []
    graph_indices: list[int] = []
    strata: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["context"], batch["valid"], batch["bridge"])
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(batch["targets"].cpu().numpy())
            graph_indices.extend(int(value) for value in batch["graph_indices"])
            strata.extend(str(value) for value in batch["strata"])
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    graph_values = np.asarray(graph_indices, dtype=np.int64)
    per_graph = []
    for graph_index in sorted(set(graph_indices)):
        mask = graph_values == graph_index
        truth = target[mask]
        probability = prediction[mask]
        binary = probability >= 0.5
        tp = int(np.count_nonzero(binary & (truth == 1)))
        fp = int(np.count_nonzero(binary & (truth == 0)))
        fn = int(np.count_nonzero((~binary) & (truth == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        graph_strata = {strata[index] for index in np.flatnonzero(mask)}
        per_graph.append(
            {
                "graph_index": int(graph_index),
                "stratum": next(iter(graph_strata)),
                "rmse": float(np.sqrt(np.mean((probability - truth) ** 2))),
                "f1": float(f1),
            }
        )
    return {
        "graph_balanced_rmse": float(np.mean([row["rmse"] for row in per_graph])),
        "graph_balanced_f1": float(np.mean([row["f1"] for row in per_graph])),
        "per_graph": per_graph,
        "training_curve": curve,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _aggregate(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for branch in BRANCHES:
        metrics[branch] = {
            "rmse": float(
                np.mean(
                    [fold["branches"][branch]["graph_balanced_rmse"] for fold in folds]
                )
            ),
            "f1": float(
                np.mean(
                    [fold["branches"][branch]["graph_balanced_f1"] for fold in folds]
                )
            ),
        }
    true = metrics["TRUE_BRIDGE"]["rmse"]
    no_bridge = metrics["NO_BRIDGE"]["rmse"]
    shuffled = metrics["SHUFFLED_BRIDGE"]["rmse"]
    true_no = (no_bridge - true) / max(no_bridge, 1e-12)
    true_shuffled = (shuffled - true) / max(shuffled, 1e-12)
    wins_no = sum(
        fold["branches"]["TRUE_BRIDGE"]["graph_balanced_rmse"]
        < fold["branches"]["NO_BRIDGE"]["graph_balanced_rmse"]
        for fold in folds
    )
    wins_shuffled = sum(
        fold["branches"]["TRUE_BRIDGE"]["graph_balanced_rmse"]
        < fold["branches"]["SHUFFLED_BRIDGE"]["graph_balanced_rmse"]
        for fold in folds
    )
    return {
        "metrics": metrics,
        "true_vs_no_bridge_reduction": float(true_no),
        "true_vs_shuffled_reduction": float(true_shuffled),
        "true_wins_no_bridge_folds": int(wins_no),
        "true_wins_shuffled_folds": int(wins_shuffled),
        "gate": bool(
            true_no >= 0.02
            and true_shuffled >= 0.02
            and wins_no >= 2
            and wins_shuffled >= 2
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# 槽位—共享节点桥 Transformer 首轮结果",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 门槛：`{decision['gate']}`",
        "",
        "| 分支 | RMSE | F1 |",
        "|---|---:|---:|",
    ]
    for branch in BRANCHES:
        row = decision["metrics"][branch]
        lines.append(f"| {branch} | {row['rmse']:.6f} | {row['f1']:.6f} |")
    lines.extend(
        [
            "",
            f"- 正确桥相对无桥改善：`{decision['true_vs_no_bridge_reduction']:.4%}`，赢 `{decision['true_wins_no_bridge_folds']}/3` 折；",
            f"- 正确桥相对打乱桥改善：`{decision['true_vs_shuffled_reduction']:.4%}`，赢 `{decision['true_wins_shuffled_folds']}/3` 折；",
            "- 目标是完整 8 节点局部块邻接，而不是统计描述。",
            "- 本轮不使用图标签、KSVD、GIN/GINE 或源节点编号嵌入。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Slot-level Beam8 bridge Transformer")
    parser.add_argument("--source", type=Path, default=default_source_path())
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--pilot-seed", type=int, default=PILOT_SEED)
    parser.add_argument("--cover-seed", type=int, default=COVER_SEED)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--model-seed", type=int, default=20260817)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--local-layers", type=int, default=1)
    parser.add_argument("--global-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    bundle = load_bundle(
        source_pkl=args.source,
        cache_path=args.cache or default_cache_path(args.source),
        force=False,
    )
    selection = select_pilot_indices(bundle, n_pilot=args.pilot_size, seed=args.pilot_seed)
    indices = selection.indices
    all_strata = density_stratum(bundle.avg_degrees)
    prepared, missing = prepare_examples(
        bundle,
        indices,
        all_strata,
        cover_seed=args.cover_seed,
        maximum_patches=MAXIMUM_PATCHES,
    )
    if missing:
        raise RuntimeError(f"missing covers: {missing[:3]}")
    folds = balanced_group_folds(
        indices,
        all_strata[indices],
        bundle.roots[indices],
        n_splits=N_SPLITS,
        seed=args.split_seed,
        view=VIEW,
    )
    fold_audit = audit_fold_partition(
        folds,
        indices,
        groups_array_by_index=dict(zip(indices, bundle.roots[indices])),
        require_group_integrity=True,
    )
    if not fold_audit["passed"]:
        raise RuntimeError("fold audit failed")
    patch_size, _overlap = GEOMETRIES[GEOMETRY]
    fold_rows = []
    for fold in folds:
        train_examples = [prepared[int(index)] for index in fold.train_indices]
        test_examples = [prepared[int(index)] for index in fold.test_indices]
        branches = {}
        for branch in BRANCHES:
            branches[branch] = train_and_evaluate(
                build_samples(train_examples, branch=branch, patch_size=patch_size),
                build_samples(test_examples, branch=branch, patch_size=patch_size),
                patch_size=patch_size,
                seed=args.model_seed + int(fold.fold_index) * 101,
                hidden=args.hidden,
                heads=args.heads,
                local_layers=args.local_layers,
                global_layers=args.global_layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            print(
                f"fold={fold.fold_index} branch={branch} "
                f"rmse={branches[branch]['graph_balanced_rmse']:.6f} "
                f"f1={branches[branch]['graph_balanced_f1']:.6f}",
                flush=True,
            )
        fold_rows.append(
            {
                "fold_index": int(fold.fold_index),
                "train_graph_count": len(train_examples),
                "test_graph_count": len(test_examples),
                "branches": branches,
            }
        )
    decision = _aggregate(fold_rows)
    payload = {
        "protocol": PROTOCOL,
        "selection": selection.summary(),
        "fold_audit": fold_audit,
        "config": {
            **{
                key: value
                for key, value in vars(args).items()
                if key not in {"source", "cache", "json", "report"}
            },
            "source": str(args.source),
            "cache": None if args.cache is None else str(args.cache),
            "geometry": GEOMETRY,
            "branches": list(BRANCHES),
            "labels_used": False,
            "ksvd_used": False,
            "gnn_used": False,
        },
        "folds": fold_rows,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
