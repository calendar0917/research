"""Label-free masked-chemistry gate for single- and multi-cover Beam8.

The model reconstructs masked OGB atom fields exclusively from overlapping
patch contexts.  HIV labels are never attached to a training/evaluation item.
All molecules come from one official-train internal scaffold fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.molhiv_beam8_incidence import (
    _atomic_number_one_hot,
    _feature_row_colors,
    typed_bond_adjacency,
)
from code.overlap_cover import patch_budget
from code.run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from code.run_beam8_nci1_chain_classification import _adjacency, _components
from code.run_molhiv_beam8_cover_diversity_audit import (
    _canonical_components,
    _random_bfs_patch,
)


VARIANTS = (
    "single_beam",
    "multi_beam",
    "multi_beam_shuffled",
    "multi_random_bfs",
    "multi_balanced_bfs",
)
DEFAULT_OUTPUT = Path(
    "tracks/ksvd/results/molhiv/"
    "molhiv_beam8_masked_chemistry_gate_20260816.json"
)


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _limit(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if limit <= 0 or len(values) <= limit:
        return values.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(values, size=limit, replace=False)).astype(np.int64)


def _nonidentity_permutation(size: int, seed: int) -> np.ndarray:
    if size <= 1:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(size)
    if np.array_equal(permutation, np.arange(size)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def random_bfs_cover_slots(
    adjacency: np.ndarray,
    typed: np.ndarray,
    canonical_types: np.ndarray,
    *,
    seed: int,
    patch_size: int,
    overlap: int,
    edge_capacity_multiplier: float,
) -> tuple[tuple[int, ...], ...]:
    output: list[tuple[int, ...]] = []
    for component_offset, (ordered_global, binary) in enumerate(
        _canonical_components(adjacency, typed, canonical_types)
    ):
        if len(ordered_global) <= patch_size:
            output.append(tuple(int(node) for node in ordered_global))
            continue
        budget = patch_budget(
            binary,
            patch_size=patch_size,
            target_overlap=overlap,
            edge_capacity_multiplier=edge_capacity_multiplier,
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, component_offset]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        roots: list[int] = []
        while len(roots) < budget:
            roots.extend(int(node) for node in rng.permutation(len(binary)))
        for root in roots[:budget]:
            local = _random_bfs_patch(binary, root, rng, patch_size)
            output.append(tuple(int(ordered_global[node]) for node in local))
    return tuple(output)


def balanced_bfs_cover_slots(
    adjacency: np.ndarray,
    typed: np.ndarray,
    canonical_types: np.ndarray,
    *,
    seed: int,
    patch_size: int,
    overlap: int,
    edge_capacity_multiplier: float,
    candidate_multiplier: int = 16,
) -> tuple[tuple[int, ...], ...]:
    """Select randomized-BFS candidates by marginal occurrence balance.

    The objective is topology-only.  It rewards previously unseen nodes and
    edges first, then low-occurrence nodes/edges and patch-set novelty.  This
    drops the continuous-chain constraint that made marginal Beam less diverse.
    """

    output: list[tuple[int, ...]] = []
    for component_offset, (ordered_global, binary) in enumerate(
        _canonical_components(adjacency, typed, canonical_types)
    ):
        if len(ordered_global) <= patch_size:
            output.append(tuple(int(node) for node in ordered_global))
            continue
        budget = patch_budget(
            binary,
            patch_size=patch_size,
            target_overlap=overlap,
            edge_capacity_multiplier=edge_capacity_multiplier,
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, component_offset, 991]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        target_candidates = max(budget * candidate_multiplier, budget)
        candidates: list[tuple[int, ...]] = []
        signatures: set[tuple[int, ...]] = set()
        roots: list[int] = []
        while len(roots) < target_candidates * 2:
            roots.extend(int(node) for node in rng.permutation(len(binary)))
        for root in roots:
            patch = _random_bfs_patch(binary, root, rng, patch_size)
            signature = tuple(sorted(patch))
            if signature in signatures:
                continue
            signatures.add(signature)
            candidates.append(patch)
            if len(candidates) >= target_candidates:
                break
        if not candidates:
            raise RuntimeError("balanced BFS produced no candidates")

        node_count = np.zeros(len(binary), dtype=np.int64)
        edge_list = [
            (left, right)
            for left in range(len(binary))
            for right in range(left + 1, len(binary))
            if binary[left, right] != 0
        ]
        edge_lookup = {edge: offset for offset, edge in enumerate(edge_list)}
        edge_count = np.zeros(len(edge_list), dtype=np.int64)
        selected: list[tuple[int, ...]] = []
        remaining = list(candidates)
        while len(selected) < budget:
            scored = []
            for candidate in remaining:
                nodes = set(candidate)
                edges = [
                    edge_lookup[(left, right)]
                    for left, right in edge_list
                    if left in nodes and right in nodes
                ]
                new_nodes = int(np.count_nonzero(node_count[list(nodes)] == 0))
                new_edges = int(np.count_nonzero(edge_count[edges] == 0)) if edges else 0
                node_balance = float(np.sum(1.0 / (1.0 + node_count[list(nodes)])))
                edge_balance = (
                    float(np.sum(1.0 / (1.0 + edge_count[edges]))) if edges else 0.0
                )
                maximum_jaccard = max(
                    (
                        len(nodes & set(previous)) / len(nodes | set(previous))
                        for previous in selected
                    ),
                    default=0.0,
                )
                score = (
                    new_nodes,
                    new_edges,
                    node_balance,
                    edge_balance,
                    -maximum_jaccard,
                )
                scored.append((score, candidate, edges))
            best = max(score for score, _candidate, _edges in scored)
            choices = [item for item in scored if item[0] == best]
            _score, chosen, chosen_edges = choices[int(rng.integers(len(choices)))]
            selected.append(chosen)
            node_count[list(set(chosen))] += 1
            if chosen_edges:
                edge_count[chosen_edges] += 1
            remaining.remove(chosen)
            if not remaining and len(selected) < budget:
                remaining = list(candidates)
        output.extend(
            tuple(int(ordered_global[node]) for node in patch) for patch in selected
        )
    return tuple(output)


def build_patch_context_data(
    graph_index: int,
    atom_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    covers: Sequence[Sequence[Sequence[int]]],
    *,
    patch_size: int,
    context_groups: int,
    shuffle_seed: int,
) -> Any:
    """Make a compact HeteroData containing only atom/patch chemistry context."""

    import torch
    from torch_geometric.data import HeteroData

    atoms = np.asarray(atom_features, dtype=np.int64)
    slot_pairs = {
        (left, right): offset
        for offset, (left, right) in enumerate(
            (pair for left in range(patch_size) for pair in ((left, right) for right in range(left + 1, patch_size)))
        )
    }
    incidence_patch: list[int] = []
    incidence_atom: list[int] = []
    incidence_slot: list[int] = []
    internal_patch: list[int] = []
    internal_pair: list[int] = []
    internal_features: list[np.ndarray] = []
    patch_offset = 0
    for cover in covers:
        for nodes in cover:
            patch = patch_offset
            patch_offset += 1
            node_values = tuple(int(node) for node in nodes)
            for slot, node in enumerate(node_values):
                incidence_patch.append(patch)
                incidence_atom.append(node)
                incidence_slot.append(slot)
            for left_slot in range(len(node_values)):
                for right_slot in range(left_slot + 1, len(node_values)):
                    left, right = node_values[left_slot], node_values[right_slot]
                    key = tuple(sorted((left, right)))
                    if key not in edge_features:
                        continue
                    internal_patch.append(patch)
                    internal_pair.append(slot_pairs[(left_slot, right_slot)])
                    internal_features.append(np.asarray(edge_features[key], dtype=np.int64))

    data = HeteroData()
    data["atom"].x = torch.tensor(atoms, dtype=torch.long)
    data["atom"].num_nodes = len(atoms)
    group_offset = int(
        hashlib.sha256(f"{shuffle_seed}:{graph_index}".encode()).digest()[0]
    ) % context_groups
    data["atom"].context_group = (
        torch.arange(len(atoms), dtype=torch.long) + group_offset
    ) % context_groups
    data["patch"].num_nodes = patch_offset
    data["patch", "contains", "atom"].edge_index = torch.tensor(
        [incidence_patch, incidence_atom], dtype=torch.long
    ).reshape(2, -1)
    data["patch", "contains", "atom"].slot = torch.tensor(
        incidence_slot, dtype=torch.long
    )
    internal_tensor = torch.tensor(internal_patch, dtype=torch.long)
    data["patch", "internal", "patch"].edge_index = torch.stack(
        [internal_tensor, internal_tensor]
    )
    data["patch", "internal", "patch"].pair = torch.tensor(
        internal_pair, dtype=torch.long
    )
    bond_width = len(next(iter(edge_features.values()))) if edge_features else 3
    data["patch", "internal", "patch"].edge_attr = torch.tensor(
        np.stack(internal_features)
        if internal_features
        else np.empty((0, bond_width), dtype=np.int64),
        dtype=torch.long,
    )
    permutation = _nonidentity_permutation(
        patch_offset, shuffle_seed ^ ((graph_index + 1) * 0x9E3779B1)
    )
    data["patch", "context_shuffle", "patch"].edge_index = torch.tensor(
        np.stack([permutation, np.arange(patch_offset, dtype=np.int64)]),
        dtype=torch.long,
    )
    return data


def classify(results: Mapping[str, Any]) -> dict[str, Any]:
    if "multi_balanced_bfs" in results and "multi_random_bfs" in results:
        balanced = results["multi_balanced_bfs"]["heldout"]
        random_bfs = results["multi_random_bfs"]["heldout"]
        balanced_effects = {
            "balanced_vs_random_bfs_ce_reduction": float(
                random_bfs["mean_cross_entropy"] - balanced["mean_cross_entropy"]
            ),
            "balanced_vs_random_bfs_accuracy_gain": float(
                balanced["mean_field_accuracy"] - random_bfs["mean_field_accuracy"]
            ),
        }
    else:
        balanced_effects = {}
    required = {
        "single_beam",
        "multi_beam",
        "multi_beam_shuffled",
        "multi_random_bfs",
    }
    if not required.issubset(results):
        ce_gain = balanced_effects.get("balanced_vs_random_bfs_ce_reduction", 0.0)
        accuracy_gain = balanced_effects.get(
            "balanced_vs_random_bfs_accuracy_gain", 0.0
        )
        return {
            "classification": (
                "BALANCED_BFS_CONTEXT_PASS"
                if ce_gain > 0.005 and accuracy_gain > 0.005
                else "BALANCED_BFS_CONTEXT_REJECTED"
            ),
            "gates": {
                "balanced_vs_random_bfs": ce_gain > 0.005
                and accuracy_gain > 0.005
            },
            "effects": balanced_effects,
        }
    single = results["single_beam"]["heldout"]
    multi = results["multi_beam"]["heldout"]
    shuffled = results["multi_beam_shuffled"]["heldout"]
    random_bfs = results["multi_random_bfs"]["heldout"]
    effects = {
        "multi_vs_single_ce_reduction": float(single["mean_cross_entropy"] - multi["mean_cross_entropy"]),
        "aligned_vs_shuffled_ce_reduction": float(shuffled["mean_cross_entropy"] - multi["mean_cross_entropy"]),
        "beam_vs_random_bfs_ce_reduction": float(random_bfs["mean_cross_entropy"] - multi["mean_cross_entropy"]),
        "multi_vs_single_accuracy_gain": float(multi["mean_field_accuracy"] - single["mean_field_accuracy"]),
        "aligned_vs_shuffled_accuracy_gain": float(multi["mean_field_accuracy"] - shuffled["mean_field_accuracy"]),
        "beam_vs_random_bfs_accuracy_gain": float(multi["mean_field_accuracy"] - random_bfs["mean_field_accuracy"]),
    }
    pass_multi = effects["multi_vs_single_ce_reduction"] > 0.005
    pass_alignment = effects["aligned_vs_shuffled_ce_reduction"] > 0.005
    pass_beam = effects["beam_vs_random_bfs_ce_reduction"] > 0.005
    if pass_multi and pass_alignment and pass_beam:
        label = "MULTIBEAM_MASKED_CHEMISTRY_PASS"
    elif pass_alignment and pass_beam:
        label = "BEAM_CONTEXT_PASS_WITHOUT_MULTICOVER_GAIN"
    elif pass_alignment:
        label = "ALIGNMENT_SIGNAL_NOT_BEAM_SPECIFIC"
    else:
        label = "NO_ALIGNED_BEAM_CHEMISTRY_SIGNAL"
    return {
        "classification": label,
        "gates": {
            "multi_cover": pass_multi,
            "alignment": pass_alignment,
            "beam_vs_random_bfs": pass_beam,
        },
        "effects": {**effects, **balanced_effects},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-cache",
        default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=8000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, default=1000)
    parser.add_argument("--limit-seed", type=int, default=20260816)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--cover-seeds", type=int, nargs="+", default=[20260813, 20260814, 20260815, 20260816])
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--cover-canonicalization",
        choices=("topology_only", "full_chemistry"),
        default="topology_only",
        help=(
            "topology_only prevents masked atom categories and bond categories "
            "from leaking through cover membership or canonical slot IDs"
        ),
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mask-prob", type=float, default=0.25)
    parser.add_argument("--context-groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="optional directory for fold-specific pretrained model states",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selected_variants = tuple(
        value.strip() for value in args.variants.split(",") if value.strip()
    )
    unknown_variants = sorted(set(selected_variants) - set(VARIANTS))
    if unknown_variants or not selected_variants:
        raise ValueError(f"invalid variants: {unknown_variants}")
    if args.fold not in (0, 1, 2) or len(args.cover_seeds) < 2:
        raise ValueError("need a valid fold and at least two cover seeds")
    if not 0.0 < args.mask_prob < 1.0 or args.context_groups < 2:
        raise ValueError("invalid masking configuration")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred.mol_encoder import BondEncoder
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
        from torch_geometric.loader import DataLoader
        from torch_geometric.utils import scatter
    except ImportError as exc:
        raise RuntimeError(f"masked chemistry gate dependencies are missing: {exc}") from exc

    started = time.time()
    _seed_everything(args.seed, torch)
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise AssertionError("categorical atom and bond features are required")
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)
    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64),
            np.asarray(bundle.meta["original_indices"], dtype=np.int64),
        ):
            raise ValueError("fold cache original indices mismatch")
    if set(fit_indices) | set(heldout_indices) != set(official_train):
        raise AssertionError("internal fold does not partition official train")
    forbidden = set(official_valid) | set(official_test)
    if (set(fit_indices) | set(heldout_indices)) & forbidden:
        raise AssertionError("official valid/test leakage")
    fit_indices = _limit(fit_indices, args.limit_per_split, args.limit_seed + 1)
    heldout_indices = _limit(heldout_indices, args.limit_per_split, args.limit_seed + 2)
    selected = np.concatenate([fit_indices, heldout_indices])

    atom_dims = tuple(int(value) for value in get_atom_feature_dims())
    bond_dims = tuple(int(value) for value in get_bond_feature_dims())
    data_by_variant: dict[str, dict[int, Any]] = {
        variant: {} for variant in selected_variants
    }
    cover_counts: dict[str, list[int]] = {
        variant: [] for variant in selected_variants
    }
    for count, raw_index in enumerate(selected, 1):
        graph_index = int(raw_index)
        graph = bundle.graphs[graph_index]
        atoms = np.asarray(bundle.node_feats[graph_index], dtype=np.int64)
        edge_features = bundle.edge_feats[graph_index]
        typed = typed_bond_adjacency(graph, edge_features, bond_dims)
        atomic_one_hot = _atomic_number_one_hot(atoms, atom_dims[0])
        adjacency = _adjacency(graph)
        if args.cover_canonicalization == "topology_only":
            cover_typed = (adjacency != 0).astype(np.int16)
            canonical_types = np.zeros(graph.n, dtype=np.int64)
            cover_edge_dim = 1
        else:
            cover_typed = typed
            canonical_types = _feature_row_colors(atoms)
            cover_edge_dim = int(np.prod(bond_dims))
        beam_covers = []
        random_covers = []
        balanced_covers = []
        need_beam = bool(
            {"single_beam", "multi_beam", "multi_beam_shuffled"}
            & set(selected_variants)
        )
        need_random = "multi_random_bfs" in selected_variants
        need_balanced = "multi_balanced_bfs" in selected_variants
        for cover_seed in args.cover_seeds:
            if need_beam:
                beam = prepare_attributed_beam_graph(
                    graph_index,
                    graph,
                    0,
                    atomic_one_hot,
                    cover_typed,
                    patch_size=args.patch_size,
                    overlap=args.overlap,
                    retained_beam=args.retained_beam,
                    edge_capacity_multiplier=args.edge_capacity_multiplier,
                    seed=int(cover_seed),
                    edge_dim=cover_edge_dim,
                    canonical_node_features=atomic_one_hot,
                    canonical_node_types=canonical_types,
                )
                beam_covers.append(beam.slot_nodes)
            if need_random:
                random_covers.append(
                    random_bfs_cover_slots(
                        adjacency,
                        cover_typed,
                        canonical_types,
                        seed=int(cover_seed),
                        patch_size=args.patch_size,
                        overlap=args.overlap,
                        edge_capacity_multiplier=args.edge_capacity_multiplier,
                    )
                )
            if need_balanced:
                balanced_covers.append(
                    balanced_bfs_cover_slots(
                        adjacency,
                        cover_typed,
                        canonical_types,
                        seed=int(cover_seed),
                        patch_size=args.patch_size,
                        overlap=args.overlap,
                        edge_capacity_multiplier=args.edge_capacity_multiplier,
                    )
                )
        covers_by_variant = {
            "single_beam": beam_covers[:1],
            "multi_beam": beam_covers,
            "multi_beam_shuffled": beam_covers,
            "multi_random_bfs": random_covers,
            "multi_balanced_bfs": balanced_covers,
        }
        for variant, covers in covers_by_variant.items():
            if variant not in selected_variants:
                continue
            item = build_patch_context_data(
                graph_index,
                atoms,
                edge_features,
                covers,
                patch_size=args.patch_size,
                context_groups=args.context_groups,
                shuffle_seed=args.seed + 314159,
            )
            data_by_variant[variant][graph_index] = item
            cover_counts[variant].append(int(item["patch"].num_nodes))
        if count == 1 or count % 100 == 0 or count == len(selected):
            print(f"prepared label-free patch contexts {count}/{len(selected)}", flush=True)

    pair_count = args.patch_size * (args.patch_size - 1) // 2

    class MaskableAtomEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(dim + 1, args.hidden) for dim in atom_dims]
            )
            for embedding in self.embeddings:
                nn.init.xavier_uniform_(embedding.weight)

        def forward(self, values: Any, mask: Any) -> Any:
            output = 0.0
            for field, (dim, embedding) in enumerate(zip(atom_dims, self.embeddings)):
                indices = values[:, field].clone()
                indices[mask] = dim
                output = output + embedding(indices)
            return output

    class PatchContextReconstructor(nn.Module):
        def __init__(self, shuffled: bool) -> None:
            super().__init__()
            self.shuffled = shuffled
            self.atom_encoder = MaskableAtomEncoder()
            self.bond_encoder = BondEncoder(args.hidden)
            self.slot_embedding = nn.Embedding(args.patch_size, args.hidden)
            self.pair_embedding = nn.Embedding(pair_count, args.hidden)
            self.patch_update = nn.Sequential(
                nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden)
            )
            self.message = nn.Sequential(
                nn.Linear(2 * args.hidden, args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden)
            )
            self.heads = nn.ModuleList([nn.Linear(args.hidden, dim) for dim in atom_dims])

        def forward(self, data: Any, mask: Any) -> list[Any]:
            atom_h = self.atom_encoder(data["atom"].x, mask)
            contains = data["patch", "contains", "atom"]
            patch_index, atom_index = contains.edge_index
            slot_h = self.slot_embedding(contains.slot)
            patch_count = int(data["patch"].num_nodes)
            patch_h = scatter(
                atom_h[atom_index] + slot_h,
                patch_index,
                dim=0,
                dim_size=patch_count,
                reduce="sum",
            )
            patch_degree = scatter(
                torch.ones_like(patch_index, dtype=atom_h.dtype),
                patch_index,
                dim=0,
                dim_size=patch_count,
                reduce="sum",
            ).clamp_min(1.0)
            patch_h = patch_h / torch.sqrt(patch_degree).unsqueeze(-1)
            internal = data["patch", "internal", "patch"]
            if internal.edge_index.shape[1] > 0:
                internal_patch = internal.edge_index[0]
                internal_h = self.bond_encoder(internal.edge_attr) + self.pair_embedding(internal.pair)
                internal_sum = scatter(
                    internal_h,
                    internal_patch,
                    dim=0,
                    dim_size=patch_count,
                    reduce="sum",
                )
                internal_degree = scatter(
                    torch.ones_like(internal_patch, dtype=atom_h.dtype),
                    internal_patch,
                    dim=0,
                    dim_size=patch_count,
                    reduce="sum",
                ).clamp_min(1.0)
                patch_h = patch_h + internal_sum / torch.sqrt(internal_degree).unsqueeze(-1)
            patch_h = self.patch_update(patch_h)
            if self.shuffled:
                source, target = data["patch", "context_shuffle", "patch"].edge_index
                shuffled_h = torch.zeros_like(patch_h)
                shuffled_h[target] = patch_h[source]
                patch_h = shuffled_h
            messages = self.message(torch.cat([patch_h[patch_index], slot_h], dim=1))
            atom_context = scatter(
                messages,
                atom_index,
                dim=0,
                dim_size=len(atom_h),
                reduce="mean",
            )
            return [head(atom_context) for head in self.heads]

    device = torch.device(args.device)

    def make_loader(variant: str, indices: Sequence[int], shuffle: bool) -> Any:
        generator = torch.Generator().manual_seed(args.seed + 2718) if shuffle else None
        return DataLoader(
            [data_by_variant[variant][int(index)] for index in indices],
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    @torch.no_grad()
    def evaluate(model: Any, variant: str, indices: Sequence[int]) -> dict[str, Any]:
        model.eval()
        loss_sum = np.zeros(len(atom_dims), dtype=np.float64)
        correct = np.zeros(len(atom_dims), dtype=np.int64)
        total = np.zeros(len(atom_dims), dtype=np.int64)
        for data in make_loader(variant, indices, False):
            data = data.to(device)
            for group in range(args.context_groups):
                mask = data["atom"].context_group == group
                logits = model(data, mask)
                for field, logit in enumerate(logits):
                    target = data["atom"].x[mask, field]
                    loss_sum[field] += float(
                        F.cross_entropy(logit[mask], target, reduction="sum")
                    )
                    correct[field] += int((logit[mask].argmax(dim=1) == target).sum())
                    total[field] += int(target.numel())
        field_loss = loss_sum / np.maximum(total, 1)
        field_accuracy = correct / np.maximum(total, 1)
        return {
            "mean_cross_entropy": float(field_loss.mean()),
            "atomic_number_cross_entropy": float(field_loss[0]),
            "mean_field_accuracy": float(field_accuracy.mean()),
            "atomic_number_accuracy": float(field_accuracy[0]),
            "field_cross_entropy": field_loss.tolist(),
            "field_accuracy": field_accuracy.tolist(),
            "atoms": int(total[0]),
        }

    results: dict[str, Any] = {}
    for variant in selected_variants:
        _seed_everything(args.seed, torch)
        model = PatchContextReconstructor(variant == "multi_beam_shuffled").to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        mask_generator = torch.Generator(device=device).manual_seed(args.seed + 1618)
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            batches = 0
            for data in make_loader(variant, fit_indices, True):
                data = data.to(device)
                mask = torch.rand(
                    data["atom"].x.shape[0], generator=mask_generator, device=device
                ) < args.mask_prob
                if not bool(mask.any()):
                    mask[0] = True
                logits = model(data, mask)
                losses = [
                    F.cross_entropy(logit[mask], data["atom"].x[mask, field])
                    for field, logit in enumerate(logits)
                ]
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1
            history.append(total_loss / max(batches, 1))
            print(
                f"variant={variant} epoch={epoch:02d} loss={history[-1]:.5f}",
                flush=True,
            )
        results[variant] = {
            "fit": evaluate(model, variant, fit_indices),
            "heldout": evaluate(model, variant, heldout_indices),
            "history": history,
            "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = args.checkpoint_dir / f"{variant}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": variant,
                    "atom_feature_dims": atom_dims,
                    "bond_feature_dims": bond_dims,
                    "hidden": args.hidden,
                    "fold": args.fold,
                    "fit_indices": fit_indices,
                    "cover_canonicalization": args.cover_canonicalization,
                    "official_valid_evaluations": 0,
                    "official_test_evaluations": 0,
                },
                checkpoint_path,
            )
            results[variant]["checkpoint"] = str(checkpoint_path)
        print(json.dumps({"variant": variant, **results[variant]["heldout"]}), flush=True)

    decision = classify(results)
    payload = {
        "protocol_id": "molhiv-fold-only-multicover-masked-chemistry-gate-v1",
        "date": "2026-08-16",
        "scope": "official-train internal scaffold fold only",
        "labels_used": False,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "config": {
            **vars(args),
            "variants": list(selected_variants),
            "output": str(args.output),
            "checkpoint_dir": (
                None if args.checkpoint_dir is None else str(args.checkpoint_dir)
            ),
        },
        "split": {
            "fit_graphs": len(fit_indices),
            "heldout_graphs": len(heldout_indices),
            "fit_intersection_official_valid_test": 0,
            "heldout_intersection_official_valid_test": 0,
        },
        "mean_patch_counts": {
            variant: float(np.mean(values)) for variant, values in cover_counts.items()
        },
        "results": results,
        "decision": decision,
        "elapsed_seconds": float(time.time() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
