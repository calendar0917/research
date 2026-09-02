"""Official-train scaffold pilot for explicit typed Beam8 incidence on MolHIV.

This runner compares a matched original-atom GINE with Beam8 interfaces that
retain patch instances.  It never evaluates OGB official-valid or official-test.
The first stage is intentionally dictionary free so that any gain can be
attributed to the atom--patch / patch--patch topology rather than KSVD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.molhiv_beam8_incidence import (
    build_molhiv_beam8_incidence,
    incidence_summary,
    to_heterodata,
)


VARIANTS = (
    "gine",
    "node_meanmax",
    "incidence_no_chain",
    "incidence_base_only",
    "incidence_completion_only",
    "incidence_split",
    "incidence_split_shuffled",
    "true_chain",
    "true_chain_base_only",
    "true_chain_overlap_only",
    "patch_shuffled",
    "chain_shuffled",
    "mapping_shuffled",
    "overlap_graph",
    "overlap_mapping_shuffled",
    "base_overlap_graph",
    "base_overlap_mapping_shuffled",
    "bond_endpoint",
    "bond_endpoint_shuffled",
    "bond_endpoint_no_patch",
    "bond_endpoint_graph_context",
    "bond_endpoint_leave_one_out",
    "bond_endpoint_slot_context",
    "bond_endpoint_structure_context",
    "bond_endpoint_structure_context_shuffled",
    "bond_endpoint_internal_context",
    "bond_endpoint_position_context",
    "bond_endpoint_no_canonical",
    "bond_endpoint_base_only",
    "bond_endpoint_base_only_shuffled",
    "bond_endpoint_bag",
    "bond_endpoint_bag_shuffled",
    "bond_edge_update",
    "bond_edge_update_shuffled",
    "bag",
)


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


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


def _stratified_limit(
    indices: np.ndarray, labels: np.ndarray, limit: int, seed: int
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit <= 0 or len(indices) <= limit:
        return indices.copy()
    rng = np.random.default_rng(seed)
    positive = indices[labels[indices] > 0.5]
    negative = indices[labels[indices] <= 0.5]
    n_positive = max(1, int(round(limit * len(positive) / max(len(indices), 1))))
    n_positive = min(n_positive, len(positive), limit - 1)
    n_negative = min(limit - n_positive, len(negative))
    n_positive = min(limit - n_negative, len(positive))
    chosen = np.concatenate(
        [
            rng.choice(positive, n_positive, replace=False),
            rng.choice(negative, n_negative, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen.astype(np.int64)


def _parse_variants(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    unknown = sorted(set(values) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    if not values:
        raise ValueError("at least one variant is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-cache",
        default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=8000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--limit-seed", type=int, default=0)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--branch-dropout", type=float, default=0.0)
    parser.add_argument("--fusion-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--positive-weight",
        type=float,
        default=1.0,
        help="BCE positive weight; use 0 for the fit-fold negative/positive ratio",
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument(
        "--coverage-checkpoint",
        choices=("base", "fair95", "edge100"),
        default="edge100",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        default="tracks/ksvd/results/molhiv/beam8_incidence_pilot.json",
    )
    args = parser.parse_args()
    variants = _parse_variants(args.variants)
    if args.fold < 0 or args.fold >= 3:
        raise ValueError("--fold must be 0, 1, or 2")
    if min(
        args.epochs,
        args.batch_size,
        args.hidden,
        args.layers,
        args.patch_size,
        args.retained_beam,
    ) <= 0:
        raise ValueError("positive model and Beam8 dimensions are required")
    if args.overlap < 0 or args.overlap >= args.patch_size:
        raise ValueError("overlap must be in [0, patch_size)")
    if not 0.0 <= args.branch_dropout < 1.0:
        raise ValueError("branch-dropout must be in [0, 1)")
    if args.fusion_scale < 0.0:
        raise ValueError("fusion-scale must be non-negative")
    if args.positive_weight < 0.0:
        raise ValueError("positive-weight must be non-negative")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv, global_mean_pool
        from torch_geometric.utils import scatter
    except ImportError as exc:
        raise RuntimeError(
            "MolHIV Beam8 training requires torch, torch-geometric, and ogb; "
            "install tracks/ksvd/configs/requirements-molhiv.txt"
        ) from exc

    started = time.time()
    device = torch.device(args.device)
    _seed_everything(args.seed, torch)
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise AssertionError("MolHIV atom/bond features are required")
    labels = np.asarray(bundle.y, dtype=np.float64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(
            folds[f"fold_{args.fold}_train_indices"], dtype=np.int64
        )
        heldout_indices = np.asarray(
            folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64
        )
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64), expected_original
        ):
            raise ValueError("fold cache original_indices mismatch")
        if "official_train_indices" in folds.files and not np.array_equal(
            np.asarray(folds["official_train_indices"], dtype=np.int64), official_train
        ):
            raise ValueError("fold cache official_train_indices mismatch")

    if set(fit_indices) | set(heldout_indices) != set(official_train):
        raise AssertionError("scaffold fold does not partition official train")
    if set(fit_indices) & set(heldout_indices):
        raise AssertionError("fit and held-out scaffold folds overlap")
    if set(fit_indices) & (set(official_valid) | set(official_test)):
        raise AssertionError("fit fold leaks official valid/test")
    if set(heldout_indices) & (set(official_valid) | set(official_test)):
        raise AssertionError("held-out fold leaks official valid/test")

    fit_indices = _stratified_limit(
        fit_indices, labels, args.limit_per_split, args.limit_seed + 101
    )
    heldout_indices = _stratified_limit(
        heldout_indices, labels, args.limit_per_split, args.limit_seed + 202
    )
    selected_indices = np.unique(np.concatenate([fit_indices, heldout_indices]))
    fit_positive = int(labels[fit_indices].sum())
    if fit_positive <= 0:
        raise ValueError("fit split must contain at least one positive graph")
    resolved_positive_weight = (
        float(args.positive_weight)
        if args.positive_weight > 0.0
        else float((len(fit_indices) - fit_positive) / fit_positive)
    )
    atom_dims = tuple(int(value) for value in get_atom_feature_dims())
    bond_dims = tuple(int(value) for value in get_bond_feature_dims())

    incidence_items = {}
    pyg_data = {}
    for count, raw_index in enumerate(selected_indices, 1):
        graph_index = int(raw_index)
        item = build_molhiv_beam8_incidence(
            graph_index,
            bundle.graphs[graph_index],
            labels[graph_index],
            bundle.node_feats[graph_index],
            bundle.edge_feats[graph_index],
            atom_feature_dims=atom_dims,
            bond_feature_dims=bond_dims,
            patch_size=args.patch_size,
            overlap=args.overlap,
            retained_beam=args.retained_beam,
            seed=20260815,
            coverage_checkpoint=args.coverage_checkpoint,
        )
        incidence_items[graph_index] = item
        pyg_data[graph_index] = to_heterodata(item)
        if count == 1 or count % 100 == 0 or count == len(selected_indices):
            print(f"prepared Beam8 molecules {count}/{len(selected_indices)}", flush=True)

    first = pyg_data[int(selected_indices[0])]
    position_dim = int(first["patch"].position.shape[1])

    class Beam8IncidenceGINE(nn.Module):
        def __init__(self, variant: str) -> None:
            super().__init__()
            self.variant = variant
            self.atom_encoder = AtomEncoder(args.hidden)
            self.bond_encoder = BondEncoder(args.hidden)
            self.atom_convs = nn.ModuleList()
            self.atom_norms = nn.ModuleList()
            for _ in range(args.layers):
                mlp = nn.Sequential(
                    nn.Linear(args.hidden, args.hidden),
                    nn.ReLU(),
                    nn.Linear(args.hidden, args.hidden),
                )
                self.atom_convs.append(GINEConv(mlp, train_eps=True))
                self.atom_norms.append(nn.BatchNorm1d(args.hidden))

            self.slot_embedding = nn.Embedding(args.patch_size, args.hidden)
            self.pair_embedding = nn.Embedding(
                args.patch_size * (args.patch_size - 1) // 2, args.hidden
            )
            self.slot_update = nn.Sequential(
                nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden)
            )
            self.internal_update = nn.Sequential(
                nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden)
            )
            self.position_update = nn.Sequential(
                nn.Linear(position_dim, args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden)
            )
            self.patch_role_embedding = nn.Embedding(2, args.hidden)
            self.patch_norm = nn.LayerNorm(args.hidden)
            self.chain_role = nn.Linear(2, args.hidden)
            self.chain_message = nn.Sequential(
                nn.Linear(2 * args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.chain_output = nn.Linear(args.hidden, args.hidden, bias=False)
            nn.init.zeros_(self.chain_output.weight)

            self.flag_projection = nn.Linear(3, args.hidden)
            self.incidence_message = nn.Sequential(
                nn.Linear(3 * args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.incidence_gate = nn.Linear(3 * args.hidden, 1)
            self.meanmax_projection = nn.Sequential(
                nn.Linear(2 * args.hidden, args.hidden), nn.ReLU()
            )
            self.bag_projection = nn.Sequential(
                nn.Linear(args.hidden, args.hidden), nn.ReLU()
            )
            self.fusion_output = nn.Linear(args.hidden, args.hidden, bias=False)
            nn.init.zeros_(self.fusion_output.weight)
            self.edge_context_update = nn.Sequential(
                nn.Linear(args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.edge_fusion_output = nn.Linear(
                args.hidden, args.hidden, bias=False
            )
            nn.init.zeros_(self.edge_fusion_output.weight)
            self.completion_gate = nn.Parameter(torch.zeros(()))

            self.jk_gates = nn.Parameter(torch.zeros(max(args.layers - 1, 0)))
            self.head = nn.Linear(args.hidden, 1)

        def _patch_states(self, data: Any, atom_h: Any) -> tuple[Any, Any, Any, Any]:
            mask = data["patch"].slot_mask.unsqueeze(-1)
            n_patches = int(mask.shape[0])
            incidence = data["patch", "contains", "atom"]
            patch_index, atom_index = incidence.edge_index
            flat_slot = patch_index * args.patch_size + incidence.slot
            slot_h = scatter(
                atom_h[atom_index],
                flat_slot,
                dim=0,
                dim_size=n_patches * args.patch_size,
                reduce="sum",
            ).reshape(n_patches, args.patch_size, args.hidden)
            slot_ids = torch.arange(args.patch_size, device=atom_h.device)
            slot_h = self.slot_update(
                slot_h + self.slot_embedding(slot_ids).unsqueeze(0)
            )
            slot_h = slot_h * mask

            internal_store = data["patch", "internal_bond", "patch"]
            internal_summary = atom_h.new_zeros((n_patches, args.hidden))
            if internal_store.edge_index.shape[1] > 0:
                patch_index = internal_store.edge_index[0]
                bond = self.bond_encoder(internal_store.edge_attr)
                pair = self.pair_embedding(internal_store.pair)
                messages = self.internal_update(bond + pair)
                internal_summary = scatter(
                    messages, patch_index, dim=0, dim_size=n_patches, reduce="sum"
                )
                degree = scatter(
                    torch.ones_like(patch_index, dtype=atom_h.dtype),
                    patch_index,
                    dim=0,
                    dim_size=n_patches,
                    reduce="sum",
                ).clamp_min(1.0)
                internal_summary = internal_summary / torch.sqrt(degree).unsqueeze(-1)
            position = self.position_update(data["patch"].position)
            position = position + self.patch_role_embedding(
                data["patch"].is_completion.long()
            )
            return slot_h, mask, internal_summary, position

        def _pool_patch_states(
            self, slot_h: Any, mask: Any, internal_summary: Any, position: Any
        ) -> Any:
            slot_count = mask.sum(dim=1).clamp_min(1.0)
            slot_summary = (slot_h * mask).sum(dim=1) / torch.sqrt(slot_count)
            return self.patch_norm(slot_summary + internal_summary + position)

        def _shuffle_patch_states(self, values: Any, data: Any) -> Any:
            edge_index = data["patch", "token_shuffle", "patch"].edge_index
            source, target = edge_index
            output = torch.zeros_like(values)
            output[target] = values[source]
            return output

        def _chain_update(self, slot_h: Any, data: Any, relation: str) -> Any:
            store = data["patch", relation, "patch"]
            if store.edge_index.shape[1] == 0:
                return slot_h
            source, target = store.edge_index
            mapping = store.edge_attr[:, 2:].reshape(-1, args.patch_size, args.patch_size)
            aligned = torch.einsum("eij,eih->ejh", mapping, slot_h[source])
            shared_mask = mapping.sum(dim=1).clamp_max(1.0).unsqueeze(-1)
            role = self.chain_role(store.edge_attr[:, :2]).unsqueeze(1)
            message = self.chain_message(
                torch.cat([aligned, role.expand_as(aligned)], dim=2)
            ) * shared_mask
            aggregate = scatter(
                message, target, dim=0, dim_size=len(slot_h), reduce="sum"
            )
            degree = scatter(
                shared_mask.squeeze(-1),
                target,
                dim=0,
                dim_size=len(slot_h),
                reduce="sum",
            ).clamp_min(1.0)
            return slot_h + self.chain_output(
                aggregate / torch.sqrt(degree).unsqueeze(-1)
            )

        def _fusion_context(
            self, atom_h: Any, patch_h: Any, slot_h: Any, data: Any
        ) -> Any:
            atom_count = int(atom_h.shape[0])
            incidence = data["patch", "contains", "atom"]
            patch_index, atom_index = incidence.edge_index
            if self.variant == "bag":
                return self._bag_context(patch_h, data)
            if self.variant == "node_meanmax":
                mean = scatter(
                    patch_h[patch_index], atom_index, dim=0, dim_size=atom_count, reduce="mean"
                )
                maximum = scatter(
                    patch_h[patch_index], atom_index, dim=0, dim_size=atom_count, reduce="max"
                )
                return self.meanmax_projection(torch.cat([mean, maximum], dim=1))

            role = (
                slot_h[patch_index, incidence.slot]
                + self.flag_projection(incidence.flags)
                + self.patch_role_embedding(
                    data["patch"].is_completion[patch_index].long()
                )
            )
            joined = torch.cat(
                [atom_h[atom_index], patch_h[patch_index], role], dim=1
            )
            message = self.incidence_message(joined) * torch.sigmoid(
                self.incidence_gate(joined)
            )
            def aggregate(selected: Any) -> Any:
                chosen_message = message[selected]
                chosen_atom = atom_index[selected]
                total = scatter(
                    chosen_message,
                    chosen_atom,
                    dim=0,
                    dim_size=atom_count,
                    reduce="sum",
                )
                degree = scatter(
                    torch.ones_like(chosen_atom, dtype=atom_h.dtype),
                    chosen_atom,
                    dim=0,
                    dim_size=atom_count,
                    reduce="sum",
                ).clamp_min(1.0)
                return total / torch.sqrt(degree).unsqueeze(-1)

            completion = data["patch"].is_completion[patch_index]
            if self.variant in {
                "incidence_base_only",
                "true_chain_base_only",
                "base_overlap_graph",
                "base_overlap_mapping_shuffled",
            }:
                return aggregate(~completion)
            if self.variant == "incidence_completion_only":
                return aggregate(completion)
            if self.variant == "true_chain_overlap_only":
                overlap_incidence = incidence.flags[:, 1:].sum(dim=1) > 0
                return aggregate(overlap_incidence)
            if self.variant in {"incidence_split", "incidence_split_shuffled"}:
                base = aggregate(~completion)
                extra = aggregate(completion)
                return base + torch.tanh(self.completion_gate) * extra
            return aggregate(torch.ones_like(completion, dtype=torch.bool))

        def _bag_context(self, patch_h: Any, data: Any) -> Any:
            graph_patch = global_mean_pool(patch_h, data["patch"].batch)
            return self.bag_projection(graph_patch[data["atom"].batch])

        def _bond_endpoint_context(
            self,
            atom_h: Any,
            patch_h: Any,
            slot_h: Any,
            mask: Any,
            internal_summary: Any,
            position: Any,
            data: Any,
            *,
            base_only: bool,
            context_mode: str,
            use_canonical_role: bool,
        ) -> Any:
            atom_count = int(atom_h.shape[0])
            store = data["patch", "bond_endpoint", "atom"]
            patch_index, atom_index = store.edge_index
            role = self.bond_encoder(store.edge_attr) + self.patch_role_embedding(
                data["patch"].is_completion[patch_index].long()
            )
            if use_canonical_role:
                role = (
                    role
                    + self.pair_embedding(store.pair)
                    + self.slot_embedding(store.slot)
                )
            if context_mode == "patch":
                patch_context = patch_h[patch_index]
            elif context_mode == "none":
                patch_context = torch.zeros_like(patch_h[patch_index])
            elif context_mode == "graph":
                graph_context = global_mean_pool(atom_h, data["atom"].batch)
                patch_context = graph_context[data["patch"].batch[patch_index]]
            elif context_mode == "leave_one_out":
                slot_total = (slot_h * mask).sum(dim=1)
                remaining_count = (mask.sum(dim=1) - 1.0).clamp_min(1.0)
                slot_summary = (
                    slot_total[patch_index]
                    - slot_h[patch_index, store.slot]
                ) / torch.sqrt(remaining_count[patch_index])
                patch_context = self.patch_norm(
                    slot_summary
                    + internal_summary[patch_index]
                    + position[patch_index]
                )
            elif context_mode == "slot":
                slot_count = mask.sum(dim=1).clamp_min(1.0)
                slot_summary = (slot_h * mask).sum(dim=1) / torch.sqrt(slot_count)
                patch_context = self.patch_norm(slot_summary)[patch_index]
            elif context_mode == "structure":
                patch_context = self.patch_norm(
                    internal_summary + position
                )[patch_index]
            elif context_mode == "internal":
                patch_context = self.patch_norm(internal_summary)[patch_index]
            elif context_mode == "position":
                patch_context = self.patch_norm(position)[patch_index]
            else:
                raise ValueError(f"unknown bond endpoint context mode: {context_mode}")
            joined = torch.cat(
                [atom_h[atom_index], patch_context, role], dim=1
            )
            message = self.incidence_message(joined) * torch.sigmoid(
                self.incidence_gate(joined)
            )
            if base_only:
                selected = ~data["patch"].is_completion[patch_index]
                message = message[selected]
                atom_index = atom_index[selected]
            aggregate = scatter(
                message,
                atom_index,
                dim=0,
                dim_size=atom_count,
                reduce="sum",
            )
            degree = scatter(
                torch.ones_like(atom_index, dtype=atom_h.dtype),
                atom_index,
                dim=0,
                dim_size=atom_count,
                reduce="sum",
            ).clamp_min(1.0)
            return aggregate / torch.sqrt(degree).unsqueeze(-1)

        def _bond_edge_states(
            self, atom_edge: Any, patch_h: Any, data: Any, edge_index: Any
        ) -> Any:
            store = data["patch", "bond_endpoint", "atom"]
            if store.edge_index.shape[1] == 0:
                return atom_edge
            if store.edge_index.shape[1] % 2 != 0:
                raise RuntimeError("bond endpoint incidences must occur in pairs")
            patch_pair = store.edge_index[0].reshape(-1, 2)
            if not torch.equal(patch_pair[:, 0], patch_pair[:, 1]):
                raise RuntimeError("paired bond endpoints must share one patch")
            endpoint_pair = store.edge_index[1].reshape(-1, 2)
            patch_index = patch_pair[:, 0]
            bond = self.bond_encoder(store.edge_attr[::2])
            pair = self.pair_embedding(store.pair[::2])
            occurrence = self.edge_context_update(
                patch_h[patch_index] + bond + pair
            )
            directed_source = endpoint_pair[:, [0, 1]].reshape(-1)
            directed_target = endpoint_pair[:, [1, 0]].reshape(-1)
            directed_message = occurrence[:, None, :].expand(-1, 2, -1).reshape(
                -1, args.hidden
            )
            atom_count = int(data["atom"].num_nodes)
            edge_keys = edge_index[0] * atom_count + edge_index[1]
            ordered_keys, order = torch.sort(edge_keys)
            occurrence_keys = directed_source * atom_count + directed_target
            locations = torch.searchsorted(ordered_keys, occurrence_keys)
            if torch.any(locations >= len(ordered_keys)) or not torch.equal(
                ordered_keys[locations], occurrence_keys
            ):
                raise RuntimeError("patch bond occurrence is missing from atom edges")
            atom_edge_position = order[locations]
            aggregate = scatter(
                directed_message,
                atom_edge_position,
                dim=0,
                dim_size=len(atom_edge),
                reduce="sum",
            )
            degree = scatter(
                torch.ones_like(atom_edge_position, dtype=atom_edge.dtype),
                atom_edge_position,
                dim=0,
                dim_size=len(atom_edge),
                reduce="sum",
            ).clamp_min(1.0)
            return atom_edge + self.edge_fusion_output(
                aggregate / torch.sqrt(degree).unsqueeze(-1)
            )

        def forward(self, data: Any) -> Any:
            atom_h = self.atom_encoder(data["atom"].x)
            atom_edge = self.bond_encoder(data["atom", "bond", "atom"].edge_attr)
            edge_index = data["atom", "bond", "atom"].edge_index
            states = []
            for layer, (conv, norm) in enumerate(zip(self.atom_convs, self.atom_norms)):
                atom_h = F.relu(norm(conv(atom_h, edge_index, atom_edge)))
                if layer == 0 and self.variant != "gine":
                    slot_h, mask, internal_summary, position = self._patch_states(
                        data, atom_h
                    )
                    chain_relations = {
                        "true_chain": "chain",
                        "true_chain_base_only": "chain",
                        "true_chain_overlap_only": "chain",
                        "chain_shuffled": "chain_shuffled",
                        "mapping_shuffled": "mapping_shuffled",
                        "overlap_graph": "overlap",
                        "overlap_mapping_shuffled": "overlap_mapping_shuffled",
                        "base_overlap_graph": "base_overlap",
                        "base_overlap_mapping_shuffled": "base_overlap_mapping_shuffled",
                    }
                    if self.variant in chain_relations:
                        slot_h = self._chain_update(
                            slot_h, data, relation=chain_relations[self.variant]
                        )
                    patch_h = self._pool_patch_states(
                        slot_h, mask, internal_summary, position
                    )
                    if self.variant in {
                        "patch_shuffled",
                        "incidence_split_shuffled",
                        "bond_endpoint_shuffled",
                        "bond_endpoint_structure_context_shuffled",
                        "bond_endpoint_internal_context",
                        "bond_endpoint_position_context",
                        "bond_endpoint_base_only_shuffled",
                        "bond_endpoint_bag_shuffled",
                        "bond_edge_update_shuffled",
                    }:
                        slot_h = self._shuffle_patch_states(slot_h, data)
                        patch_h = self._shuffle_patch_states(patch_h, data)
                        if self.variant == "bond_endpoint_structure_context_shuffled":
                            internal_summary = self._shuffle_patch_states(
                                internal_summary, data
                            )
                            position = self._shuffle_patch_states(position, data)
                    bond_variants = {
                        "bond_endpoint",
                        "bond_endpoint_shuffled",
                        "bond_endpoint_no_patch",
                        "bond_endpoint_graph_context",
                        "bond_endpoint_leave_one_out",
                        "bond_endpoint_slot_context",
                        "bond_endpoint_structure_context",
                        "bond_endpoint_structure_context_shuffled",
                        "bond_endpoint_internal_context",
                        "bond_endpoint_position_context",
                        "bond_endpoint_no_canonical",
                        "bond_endpoint_base_only",
                        "bond_endpoint_base_only_shuffled",
                        "bond_endpoint_bag",
                        "bond_endpoint_bag_shuffled",
                    }
                    if self.variant in {"bond_edge_update", "bond_edge_update_shuffled"}:
                        atom_edge = self._bond_edge_states(
                            atom_edge, patch_h, data, edge_index
                        )
                        context = None
                    elif self.variant in bond_variants:
                        context = self._bond_endpoint_context(
                            atom_h,
                            patch_h,
                            slot_h,
                            mask,
                            internal_summary,
                            position,
                            data,
                            base_only="base_only" in self.variant,
                            context_mode=(
                                "none"
                                if self.variant == "bond_endpoint_no_patch"
                                else "graph"
                                if self.variant == "bond_endpoint_graph_context"
                                else "leave_one_out"
                                if self.variant == "bond_endpoint_leave_one_out"
                                else "slot"
                                if self.variant == "bond_endpoint_slot_context"
                                else "structure"
                                if self.variant in {
                                    "bond_endpoint_structure_context",
                                    "bond_endpoint_structure_context_shuffled",
                                }
                                else "internal"
                                if self.variant == "bond_endpoint_internal_context"
                                else "position"
                                if self.variant == "bond_endpoint_position_context"
                                else "patch"
                            ),
                            use_canonical_role=(
                                self.variant != "bond_endpoint_no_canonical"
                            ),
                        )
                        if "_bag" in self.variant:
                            context = (
                                context + self._bag_context(patch_h, data)
                            ) / np.sqrt(2.0)
                    else:
                        context = self._fusion_context(atom_h, patch_h, slot_h, data)
                    if context is not None:
                        context = F.dropout(
                            context, p=args.branch_dropout, training=self.training
                        )
                        atom_h = atom_h + args.fusion_scale * self.fusion_output(context)
                atom_h = F.dropout(atom_h, p=args.dropout, training=self.training)
                states.append(global_mean_pool(atom_h, data["atom"].batch))
            graph_h = states[-1]
            for gate, earlier in zip(self.jk_gates, states[:-1]):
                graph_h = graph_h + torch.tanh(gate) * earlier
            return self.head(graph_h).view(-1)

    def make_loader(indices: Sequence[int], *, shuffle: bool, offset: int) -> Any:
        generator = None
        if shuffle:
            generator = torch.Generator().manual_seed(args.seed + 9173 + offset)
        return DataLoader(
            [pyg_data[int(index)] for index in indices],
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=args.num_workers,
        )

    @torch.no_grad()
    def evaluate(model: Any, indices: Sequence[int], offset: int) -> dict[str, Any]:
        model.eval()
        ys: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        for batch in make_loader(indices, shuffle=False, offset=offset):
            batch = batch.to(device)
            logits.append(model(batch).cpu().numpy())
            ys.append(batch.y.view(-1).cpu().numpy())
        y = np.concatenate(ys).astype(np.float64)
        raw = np.concatenate(logits).astype(np.float64)
        probability = 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))
        return {
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "probability_sha256": _sha256(probability),
        }

    results: dict[str, Any] = {}
    for variant_index, variant in enumerate(variants):
        _seed_everything(args.seed, torch)
        model = Beam8IncidenceGINE(variant).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        # Every control sees the identical minibatch order.
        train_loader = make_loader(fit_indices, shuffle=True, offset=0)
        history = []
        positive_weight = torch.tensor(resolved_positive_weight, device=device)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            n_seen = 0
            for batch in train_loader:
                batch = batch.to(device)
                output = model(batch)
                y = batch.y.view(-1).float()
                loss = F.binary_cross_entropy_with_logits(
                    output, y, pos_weight=positive_weight
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(y)
                n_seen += len(y)
            history.append({"epoch": epoch, "train_loss": total_loss / max(n_seen, 1)})
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(
                    f"variant={variant} epoch={epoch:02d} "
                    f"loss={history[-1]['train_loss']:.6f}",
                    flush=True,
                )

        fit_metrics = evaluate(model, fit_indices, 100 + variant_index)
        heldout_metrics = evaluate(model, heldout_indices, 200 + variant_index)
        results[variant] = {
            "fit": fit_metrics,
            "heldout": heldout_metrics,
            "history": history,
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "fusion_output_norm": float(model.fusion_output.weight.norm().item()),
            "chain_output_norm": float(model.chain_output.weight.norm().item()),
            "edge_fusion_output_norm": float(
                model.edge_fusion_output.weight.norm().item()
            ),
            "completion_gate": float(
                torch.tanh(model.completion_gate).detach().cpu().item()
            ),
            "jk_gates": torch.tanh(model.jk_gates).detach().cpu().tolist(),
        }
        print(
            json.dumps(
                {
                    "variant": variant,
                    "fit_auc": fit_metrics["roc_auc"],
                    "heldout_auc": heldout_metrics["roc_auc"],
                    "heldout_ap": heldout_metrics["average_precision"],
                }
            ),
            flush=True,
        )

    paired = {}
    if "gine" in results:
        baseline_auc = results["gine"]["heldout"]["roc_auc"]
        baseline_ap = results["gine"]["heldout"]["average_precision"]
        for variant in variants:
            paired[variant] = {
                "roc_auc_delta_vs_gine": float(
                    results[variant]["heldout"]["roc_auc"] - baseline_auc
                ),
                "average_precision_delta_vs_gine": float(
                    results[variant]["heldout"]["average_precision"] - baseline_ap
                ),
            }

    item_list = [incidence_items[int(index)] for index in selected_indices]
    report = {
        "protocol_id": "molhiv-typed-beam8-explicit-incidence-scaffold-pilot-v3",
        "date": "2026-08-15",
        "scope": "OGB official-train only; one held-out Bemis-Murcko scaffold fold",
        "hypothesis": (
            "explicit role-conditioned atom--patch incidence and typed Beam8 chain "
            "relations retain useful information lost by raw patch mean/max readout"
        ),
        "architecture": {
            "patch_input": "first-layer GINE atom states scattered to canonical slots",
            "chain_update": "exact shared-slot correspondence before patch pooling",
            "atom_fusion": "canonical-slot-conditioned patch-to-atom residual",
            "completion_chain_policy": "independent segments; no synthetic chain edges",
        },
        "config": vars(args)
        | {
            "variants": list(variants),
            "resolved_positive_weight": resolved_positive_weight,
        },
        "dataset": bundle.meta,
        "fold": {
            "index": args.fold,
            "n_fit": int(len(fit_indices)),
            "n_fit_positive": int(labels[fit_indices].sum()),
            "n_heldout": int(len(heldout_indices)),
            "n_heldout_positive": int(labels[heldout_indices].sum()),
            "fit_indices_sha256": _sha256(fit_indices),
            "heldout_indices_sha256": _sha256(heldout_indices),
        },
        "beam8": incidence_summary(item_list),
        "results": results,
        "paired": paired,
        "isolation": {
            "encoded_official_valid_graphs": 0,
            "encoded_official_test_graphs": 0,
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "dictionary_or_ksvd_used": False,
        },
        "elapsed_sec": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "paired": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()
