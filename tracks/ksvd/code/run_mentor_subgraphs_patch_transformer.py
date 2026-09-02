"""Relation-aware patch Transformer on the mentor 50-node subgraph bank.

This is a deliberately GNN-free masked-patch experiment.  Beam8 patches are
the computational units.  A target patch token is replaced by a learned mask,
and the model predicts its permutation-invariant structural descriptor from
the remaining patch tokens.

Matched branches:

* ``NO_RELATION``: identical Transformer capacity, zero pair features;
* ``TRUE_RELATION``: exact shared-slot map + overlap + center distance;
* ``SHUFFLED_RELATION``: the same relation tensor with context tokens bound to
  the wrong patch positions.

No graph labels, source node IDs, KSVD codes, residual edges, GIN or GINE are
used.  Splits and Beam8 covers match the earlier mentor-bank relation pilot.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .beam8_coverage_operating_point import (
    prefix_coverage_trajectory,
    select_operating_checkpoints,
)
from .canonical_slots import reorder_cover_structurally
from .data_mentor_subgraphs import (
    STRATUM_NAMES,
    default_cache_path,
    default_source_path,
    density_stratum,
    load_bundle,
    select_pilot_indices,
)
from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .mentor_grouped_splits import audit_fold_partition, balanced_group_folds
from .overlap_cover import _make_cover, patch_budget
from .overlap_stitching import CoverExample, make_cover_example
from .run_beam8_coverage_operating_point_audit import GEOMETRIES
from .run_patch_relation_representation_audit import (
    EPS,
    all_pairs_shortest_paths,
    patch_invariant_descriptor,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = (
    ROOT
    / "tracks/ksvd/results/mentor_subgraphs/patch_transformer_pilot_20260817.json"
)
DEFAULT_REPORT = (
    ROOT
    / "tracks/ksvd/results/mentor_subgraphs/PATCH_TRANSFORMER_PILOT_20260817.md"
)
PROTOCOL = "tracks/ksvd/docs/KSVD_MENTOR_PATCH_TRANSFORMER_PROTOCOL_20260817.md"

GEOMETRY = "s8_o2"
CHECKPOINT = "BASE"
PILOT_SEED = 20260806
COVER_SEED = 970201
SPLIT_SEED = 20260807
N_SPLITS = 3
VIEW = "root_candidate_grouped"
MAXIMUM_PATCHES = 60
RETAINED_BEAM = 8
CANDIDATE_RESTARTS = 1
MULTIPLIER = 1.5
BRANCHES = ("NO_RELATION", "TRUE_RELATION", "SHUFFLED_RELATION")
DIAGNOSTIC_BRANCHES = ("RELATION_ONLY",)


@dataclass(frozen=True)
class MaskedSample:
    graph_index: int
    stratum: str
    target_index: int
    tokens: np.ndarray
    relations: np.ndarray
    target: np.ndarray


def invariant_tokens(example: CoverExample) -> np.ndarray:
    return np.stack(
        [patch_invariant_descriptor(patch.adjacency) for patch in example.cover.patches],
        axis=0,
    ).astype(np.float32)


def exact_pair_relations(example: CoverExample, patch_size: int) -> np.ndarray:
    """Pair tensor: exact slot map, overlap fraction, exp(-center distance)."""
    patches = example.cover.patches
    count = len(patches)
    relation_dim = patch_size * patch_size + 2
    output = np.zeros((count, count, relation_dim), dtype=np.float32)
    distances = all_pairs_shortest_paths(example.adjacency)
    for left_index, left in enumerate(patches):
        if len(left.node_ids) != patch_size:
            raise ValueError("patch Transformer requires fixed-size cover patches")
        for right_index, right in enumerate(patches):
            if len(right.node_ids) != patch_size:
                raise ValueError("patch Transformer requires fixed-size cover patches")
            right_slots = {int(node): slot for slot, node in enumerate(right.node_ids)}
            shared = 0
            for left_slot, node in enumerate(left.node_ids):
                right_slot = right_slots.get(int(node))
                if right_slot is not None:
                    output[
                        left_index,
                        right_index,
                        left_slot * patch_size + int(right_slot),
                    ] = 1.0
                    shared += 1
            output[left_index, right_index, -2] = shared / float(patch_size)
            distance = float(distances[int(left.center), int(right.center)])
            output[left_index, right_index, -1] = math.exp(-distance)
    return output


def _context_permutation(count: int, target_index: int, graph_index: int) -> np.ndarray:
    """Deterministic derangement of context tokens while preserving target slot."""
    permutation = np.arange(count, dtype=np.int64)
    context = np.asarray(
        [index for index in range(count) if index != target_index], dtype=np.int64
    )
    if context.size > 1:
        shift = 1 + (int(target_index) + int(graph_index)) % (context.size - 1)
        permutation[context] = np.roll(context, shift)
    return permutation


def build_samples(
    examples: Sequence[CoverExample],
    *,
    branch: str,
    patch_size: int,
) -> list[MaskedSample]:
    if branch not in (*BRANCHES, *DIAGNOSTIC_BRANCHES):
        raise ValueError(f"unknown branch: {branch}")
    samples: list[MaskedSample] = []
    for example in examples:
        tokens = invariant_tokens(example)
        targets = tokens.copy()
        relations = exact_pair_relations(example, patch_size)
        if branch == "NO_RELATION":
            relations = np.zeros_like(relations)
        if branch == "RELATION_ONLY":
            tokens = np.zeros_like(tokens)
        for target_index in range(tokens.shape[0]):
            values = tokens
            if branch == "SHUFFLED_RELATION":
                permutation = _context_permutation(
                    tokens.shape[0], target_index, int(example.graph_index)
                )
                values = tokens[permutation]
                if not np.array_equal(values[target_index], tokens[target_index]):
                    raise AssertionError("shuffled branch moved the masked target token")
            samples.append(
                MaskedSample(
                    graph_index=int(example.graph_index),
                    stratum=str(example.family),
                    target_index=int(target_index),
                    tokens=np.asarray(values, dtype=np.float32),
                    relations=np.asarray(relations, dtype=np.float32),
                    target=np.asarray(targets[target_index], dtype=np.float32),
                )
            )
    return samples


def fit_standardizer(examples: Sequence[CoverExample]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([invariant_tokens(example) for example in examples], axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0, ddof=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def standardize_samples(
    samples: Sequence[MaskedSample], mean: np.ndarray, scale: np.ndarray
) -> list[MaskedSample]:
    return [
        MaskedSample(
            graph_index=sample.graph_index,
            stratum=sample.stratum,
            target_index=sample.target_index,
            tokens=((sample.tokens - mean) / scale).astype(np.float32),
            relations=sample.relations,
            target=((sample.target - mean) / scale).astype(np.float32),
        )
        for sample in samples
    ]


def _import_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required; run with `uv run --with numpy --with torch ...`"
        ) from exc
    return torch, nn, DataLoader


def make_collate(torch: Any):
    def collate(batch: Sequence[MaskedSample]) -> dict[str, Any]:
        max_length = max(sample.tokens.shape[0] for sample in batch)
        token_dim = batch[0].tokens.shape[1]
        relation_dim = batch[0].relations.shape[2]
        tokens = np.zeros((len(batch), max_length, token_dim), dtype=np.float32)
        relations = np.zeros(
            (len(batch), max_length, max_length, relation_dim), dtype=np.float32
        )
        valid = np.zeros((len(batch), max_length), dtype=bool)
        targets = np.zeros((len(batch), token_dim), dtype=np.float32)
        target_indices = np.zeros(len(batch), dtype=np.int64)
        weights = np.zeros(len(batch), dtype=np.float32)
        graph_indices = np.zeros(len(batch), dtype=np.int64)
        strata: list[str] = []
        for row, sample in enumerate(batch):
            length = sample.tokens.shape[0]
            tokens[row, :length] = sample.tokens
            relations[row, :length, :length] = sample.relations
            valid[row, :length] = True
            targets[row] = sample.target
            target_indices[row] = sample.target_index
            weights[row] = 1.0 / float(length)
            graph_indices[row] = sample.graph_index
            strata.append(sample.stratum)
        return {
            "tokens": torch.from_numpy(tokens),
            "relations": torch.from_numpy(relations),
            "valid": torch.from_numpy(valid),
            "targets": torch.from_numpy(targets),
            "target_indices": torch.from_numpy(target_indices),
            "weights": torch.from_numpy(weights),
            "graph_indices": graph_indices,
            "strata": strata,
        }

    return collate


def build_model(
    nn: Any,
    torch: Any,
    *,
    token_dim: int,
    relation_dim: int,
    hidden: int,
    heads: int,
    layers: int,
    dropout: float,
) -> Any:
    if hidden % heads:
        raise ValueError("hidden size must be divisible by attention heads")

    class RelationAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = heads
            self.head_dim = hidden // heads
            self.qkv = nn.Linear(hidden, 3 * hidden)
            self.relation_bias = nn.Linear(relation_dim, heads, bias=False)
            self.output = nn.Linear(hidden, hidden)
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: Any, relations: Any, valid: Any) -> Any:
            batch, length, _dimension = values.shape
            qkv = self.qkv(values).reshape(
                batch, length, 3, self.heads, self.head_dim
            )
            query, key, value = qkv.unbind(dim=2)
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
                self.head_dim
            )
            bias = self.relation_bias(relations).permute(0, 3, 1, 2)
            scores = scores + bias
            scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
            attention = self.dropout(torch.softmax(scores, dim=-1))
            context = torch.matmul(attention, value)
            context = context.permute(0, 2, 1, 3).reshape(batch, length, hidden)
            return self.output(context)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden)
            self.attention = RelationAttention()
            self.norm2 = nn.LayerNorm(hidden)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden, 4 * hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden, hidden),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: Any, relations: Any, valid: Any) -> Any:
            values = values + self.dropout(
                self.attention(self.norm1(values), relations, valid)
            )
            values = values + self.dropout(self.feedforward(self.norm2(values)))
            return values * valid[:, :, None]

    class PatchTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(token_dim, hidden)
            self.mask_token = nn.Parameter(torch.zeros(hidden))
            nn.init.normal_(self.mask_token, std=0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.output = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, token_dim),
            )

        def forward(
            self, tokens: Any, relations: Any, valid: Any, target_indices: Any
        ) -> Any:
            values = self.input(tokens)
            rows = torch.arange(values.shape[0], device=values.device)
            values = values.clone()
            values[rows, target_indices] = self.mask_token
            for block in self.blocks:
                values = block(values, relations, valid)
            return self.output(values[rows, target_indices])

    return PatchTransformer()


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def train_and_evaluate(
    train_samples: Sequence[MaskedSample],
    test_samples: Sequence[MaskedSample],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    seed: int,
    hidden: int,
    heads: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    torch, nn, DataLoader = _import_torch()
    _seed_everything(torch, seed)
    collate = make_collate(torch)
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
    token_dim = train_samples[0].tokens.shape[1]
    relation_dim = train_samples[0].relations.shape[2]
    model = build_model(
        nn,
        torch,
        token_dim=token_dim,
        relation_dim=relation_dim,
        hidden=hidden,
        heads=heads,
        layers=layers,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    curve: list[float] = []
    model.train()
    for _epoch in range(epochs):
        total = 0.0
        weight_total = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["tokens"],
                batch["relations"],
                batch["valid"],
                batch["target_indices"],
            )
            per_sample = ((prediction - batch["targets"]) ** 2).mean(dim=1)
            loss = (per_sample * batch["weights"]).sum() / batch["weights"].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float((per_sample.detach() * batch["weights"]).sum())
            weight_total += float(batch["weights"].sum())
        curve.append(total / max(weight_total, EPS))

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    graph_indices: list[int] = []
    strata: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            prediction = model(
                batch["tokens"],
                batch["relations"],
                batch["valid"],
                batch["target_indices"],
            ).cpu().numpy()
            predictions.append(prediction * scale + mean)
            targets.append(batch["targets"].cpu().numpy() * scale + mean)
            graph_indices.extend(int(value) for value in batch["graph_indices"])
            strata.extend(str(value) for value in batch["strata"])
    predicted = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    graph_values = np.asarray(graph_indices, dtype=np.int64)
    per_graph = []
    for graph_index in sorted(set(graph_indices)):
        mask = graph_values == graph_index
        rmse = float(np.sqrt(np.mean((predicted[mask] - target[mask]) ** 2)))
        graph_strata = {strata[index] for index in np.flatnonzero(mask)}
        if len(graph_strata) != 1:
            raise AssertionError("one graph mapped to multiple density strata")
        per_graph.append(
            {
                "graph_index": int(graph_index),
                "stratum": next(iter(graph_strata)),
                "rmse": rmse,
            }
        )
    return {
        "graph_balanced_rmse": float(np.mean([row["rmse"] for row in per_graph])),
        "per_graph": per_graph,
        "training_curve": curve,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def prepare_examples(
    bundle: Any,
    pilot_indices: np.ndarray,
    all_strata: np.ndarray,
    *,
    cover_seed: int,
    maximum_patches: int,
) -> tuple[dict[int, CoverExample], list[dict[str, Any]]]:
    patch_size, overlap = GEOMETRIES[GEOMETRY]
    geometry_index = tuple(GEOMETRIES).index(GEOMETRY)
    prepared: dict[int, CoverExample] = {}
    missing: list[dict[str, Any]] = []
    for position, raw_source_index in enumerate(pilot_indices):
        source_index = int(raw_source_index)
        adjacency = bundle.adjacency[source_index].astype(np.int8, copy=False)
        stable = reorder_by_stable_ids(adjacency, compute_global_wl_ids(adjacency))
        base_budget = patch_budget(
            stable,
            patch_size=patch_size,
            target_overlap=overlap,
            edge_capacity_multiplier=MULTIPLIER,
        )
        sampler_seed = int(
            np.random.SeedSequence([cover_seed, source_index, geometry_index])
            .generate_state(1, dtype=np.uint32)[0]
        )
        try:
            cover = sample_marginal_candidate_cover(
                stable,
                np.random.default_rng(sampler_seed),
                n_patches=maximum_patches,
                patch_size=patch_size,
                target_overlap=overlap,
                retained_beam=RETAINED_BEAM,
                candidate_restarts=CANDIDATE_RESTARTS,
                allow_partial=True,
            )
            trajectory = prefix_coverage_trajectory(
                stable,
                cover,
                patch_size=patch_size,
                overlap=overlap,
                maximum_patches=maximum_patches,
            )
            selected = select_operating_checkpoints(
                trajectory, base_patch_count=base_budget
            )[CHECKPOINT]
            if selected is None:
                raise RuntimeError("BASE checkpoint unreachable")
            count = int(selected["patch_count"])
            prefix = _make_cover(
                f"{GEOMETRY}_{CHECKPOINT}",
                cover.patches[:count],
                cover.segment_ids[:count],
                cover.target_edges[:count],
                cover.bridge_lengths[:count],
            )
            ordered, _diagnostics = reorder_cover_structurally(
                stable, prefix, "rooted_canonical"
            )
            prepared[source_index] = make_cover_example(
                source_index,
                STRATUM_NAMES[int(all_strata[source_index])],
                int(round(float(bundle.avg_degrees[source_index]))),
                stable,
                ordered,
            )
        except Exception as exc:  # noqa: BLE001 - recorded per graph
            missing.append(
                {
                    "source_index": source_index,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if (position + 1) % 25 == 0 or position + 1 == len(pilot_indices):
            print(f"prepared={position + 1}/{len(pilot_indices)}", flush=True)
    return prepared, missing


def _aggregate(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    branch_means = {
        branch: float(np.mean([fold["branches"][branch]["graph_balanced_rmse"] for fold in folds]))
        for branch in BRANCHES
    }
    true_value = branch_means["TRUE_RELATION"]
    no_relation = branch_means["NO_RELATION"]
    shuffled = branch_means["SHUFFLED_RELATION"]
    wins_no_relation = sum(
        fold["branches"]["TRUE_RELATION"]["graph_balanced_rmse"]
        < fold["branches"]["NO_RELATION"]["graph_balanced_rmse"]
        for fold in folds
    )
    wins_shuffled = sum(
        fold["branches"]["TRUE_RELATION"]["graph_balanced_rmse"]
        < fold["branches"]["SHUFFLED_RELATION"]["graph_balanced_rmse"]
        for fold in folds
    )
    true_vs_no = (no_relation - true_value) / max(no_relation, EPS)
    true_vs_shuffled = (shuffled - true_value) / max(shuffled, EPS)
    return {
        "branch_means": branch_means,
        "true_vs_no_relation_reduction": float(true_vs_no),
        "true_vs_shuffled_reduction": float(true_vs_shuffled),
        "true_wins_no_relation_folds": int(wins_no_relation),
        "true_wins_shuffled_folds": int(wins_shuffled),
        "exploratory_gate": bool(
            true_vs_no >= 0.02
            and true_vs_shuffled >= 0.02
            and wins_no_relation >= 2
            and wins_shuffled >= 2
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# 导师 50 节点子图：局部块关系 Transformer 首轮探索",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 探索门槛：`{decision['exploratory_gate']}`",
        "",
        "## 设置",
        "",
        f"- 图数：`{payload['selection']['size']}`；有效覆盖：`{payload['data']['prepared_graphs']}`；",
        f"- 划分：`{payload['config']['n_splits']}` 折根节点分组；",
        f"- 局部块：`{payload['config']['geometry']}` / `{payload['config']['checkpoint']}`；",
        f"- 输入：未压缩的不变局部结构描述；KSVD、GIN/GINE、图标签均未使用；",
        f"- 关系：精确共享槽位图 + 重叠比例 + 中心距离；",
        f"- 模型：hidden=`{payload['config']['hidden']}`，heads=`{payload['config']['heads']}`，layers=`{payload['config']['layers']}`，epochs=`{payload['config']['epochs']}`。",
        "",
        "## 结果",
        "",
        "| 分支 | 三折平均 RMSE |",
        "|---|---:|",
    ]
    for branch in BRANCHES:
        lines.append(
            f"| {branch} | {decision['branch_means'][branch]:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- 正确关系相对无关系改善：`{decision['true_vs_no_relation_reduction']:.4%}`，赢 `{decision['true_wins_no_relation_folds']}/3` 折；",
            f"- 正确关系相对打乱关系改善：`{decision['true_vs_shuffled_reduction']:.4%}`，赢 `{decision['true_wins_shuffled_folds']}/3` 折；",
            "",
            "## 解释边界",
            "",
            "- 这是遮蔽局部块重构，不是图分类结果。",
            "- 三个分支模型容量、初始化、轮数完全相同；无关系分支只是把关系张量置零。",
            "- 打乱分支保留目标位置和关系边际，只错配其他局部块的内容与位置。",
            "- 只有正确关系稳定优于无关系和打乱关系，才值得扩展到 500 图及分子数据。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GNN-free relation-aware patch Transformer on mentor subgraphs"
    )
    parser.add_argument("--source", type=Path, default=default_source_path())
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--pilot-seed", type=int, default=PILOT_SEED)
    parser.add_argument("--cover-seed", type=int, default=COVER_SEED)
    parser.add_argument("--maximum-patches", type=int, default=MAXIMUM_PATCHES)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--model-seed", type=int, default=20260817)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    bundle = load_bundle(
        source_pkl=args.source,
        cache_path=args.cache or default_cache_path(args.source),
        force=args.force_cache,
    )
    selection = select_pilot_indices(
        bundle, n_pilot=args.pilot_size, seed=args.pilot_seed
    )
    pilot_indices = selection.indices
    all_strata = density_stratum(bundle.avg_degrees)
    strata_values = all_strata[pilot_indices]
    root_values = bundle.roots[pilot_indices]
    prepared, missing = prepare_examples(
        bundle,
        pilot_indices,
        all_strata,
        cover_seed=args.cover_seed,
        maximum_patches=args.maximum_patches,
    )
    if len(prepared) != len(pilot_indices):
        raise RuntimeError(f"missing covers for {len(missing)} selected graphs")

    folds = balanced_group_folds(
        pilot_indices,
        strata_values,
        root_values,
        n_splits=N_SPLITS,
        seed=args.split_seed,
        view=VIEW,
    )
    fold_audit = audit_fold_partition(
        folds,
        pilot_indices,
        groups_array_by_index=dict(zip(pilot_indices, root_values)),
        require_group_integrity=True,
    )
    if not fold_audit["passed"]:
        raise RuntimeError("fold integrity audit failed")

    patch_size, _overlap = GEOMETRIES[GEOMETRY]
    fold_payloads = []
    for fold in folds:
        train_examples = [prepared[int(index)] for index in fold.train_indices]
        test_examples = [prepared[int(index)] for index in fold.test_indices]
        mean, scale = fit_standardizer(train_examples)
        branches: dict[str, Any] = {}
        for branch in BRANCHES:
            train_samples = standardize_samples(
                build_samples(train_examples, branch=branch, patch_size=patch_size),
                mean,
                scale,
            )
            test_samples = standardize_samples(
                build_samples(test_examples, branch=branch, patch_size=patch_size),
                mean,
                scale,
            )
            branches[branch] = train_and_evaluate(
                train_samples,
                test_samples,
                mean=mean,
                scale=scale,
                seed=args.model_seed + int(fold.fold_index) * 101,
                hidden=args.hidden,
                heads=args.heads,
                layers=args.layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            print(
                f"fold={fold.fold_index} branch={branch} "
                f"rmse={branches[branch]['graph_balanced_rmse']:.6f}",
                flush=True,
            )
        fold_payloads.append(
            {
                "fold_index": int(fold.fold_index),
                "train_graph_count": len(train_examples),
                "test_graph_count": len(test_examples),
                "branches": branches,
            }
        )

    decision = _aggregate(fold_payloads)
    payload = {
        "protocol": PROTOCOL,
        "data": {
            "cache_graph_count": bundle.n_graphs,
            "prepared_graphs": len(prepared),
            "missing": missing,
        },
        "selection": selection.summary(),
        "fold_audit": fold_audit,
        "config": {
            "geometry": GEOMETRY,
            "checkpoint": CHECKPOINT,
            "n_splits": N_SPLITS,
            "view": VIEW,
            "split_seed": args.split_seed,
            "model_seed": args.model_seed,
            "hidden": args.hidden,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "branches": list(BRANCHES),
            "labels_used": False,
            "ksvd_used": False,
            "gnn_used": False,
        },
        "folds": fold_payloads,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
