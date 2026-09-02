"""Official-train scaffold audit of CIN-small with a Beam8 residual branch.

This is a dependency-light implementation of the molecular Sparse CIN operator
used by CWN: atom, bond and induced-cycle cells; boundary aggregation; and
same-dimensional upper adjacency conditioned on the shared coboundary.  It is
not presented as a bit-for-bit reproduction of the historical PyG-1.6 code.

No official-valid or official-test graph is encoded or evaluated here.
"""

from __future__ import annotations

import argparse
import json
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
from code.molhiv_beam8_incidence import build_molhiv_beam8_incidence, incidence_summary
from code.molhiv_cin_beam8 import ring_complex_summary, to_cin_beam8_heterodata
from code.run_molhiv_beam8_incidence import _seed_everything, _sha256, _stratified_limit


VARIANTS = (
    "cin",
    "cin_no_rings",
    "cin_ring_shuffled",
    "cin_beam8",
    "cin_beam8_shuffled",
    "cin_beam8_no_patch",
    "cin_beam8_bond",
    "cin_beam8_bond_shuffled",
    "cin_beam8_bond_bag",
    "cin_beam8_bond_no_patch",
    "cin_beam8_aux",
    "cin_beam8_aux_shuffled",
    "cin_beam8_node_aux",
    "cin_beam8_node_aux_shuffled",
    "cin_beam8_node_aux_patch",
    "cin_beam8_node_aux_patch_shuffled",
    "cin_beam8_node_aux_local",
    "cin_beam8_multiaux",
    "cin_beam8_multiaux_shuffled",
    "cin_ssl_beam",
    "cin_ssl_beam_shuffled",
    "cin_ssl_random_bfs",
    "cin_ssl_balanced_bfs",
)


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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--max-ring-size", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument(
        "--coverage-checkpoint", choices=("base", "fair95", "edge100"), default="edge100"
    )
    parser.add_argument("--fusion-scale", type=float, default=1.0)
    parser.add_argument("--branch-dropout", type=float, default=0.0)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument(
        "--aux-decay-epochs",
        type=int,
        default=0,
        help="linearly decay auxiliary weight to zero over this many epochs; 0 is constant",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ssl-beam-checkpoint", type=Path, default=None)
    parser.add_argument("--ssl-beam-shuffled-checkpoint", type=Path, default=None)
    parser.add_argument("--ssl-random-bfs-checkpoint", type=Path, default=None)
    parser.add_argument("--ssl-balanced-bfs-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--output",
        default="tracks/ksvd/results/molhiv/cin_beam8_internal_pilot.json",
    )
    args = parser.parse_args()
    variants = _parse_variants(args.variants)
    if args.fold not in (0, 1, 2):
        raise ValueError("--fold must be 0, 1, or 2")
    if min(
        args.epochs,
        args.batch_size,
        args.hidden,
        args.layers,
        args.max_ring_size,
        args.patch_size,
        args.retained_beam,
    ) <= 0:
        raise ValueError("model, ring and Beam8 dimensions must be positive")
    if args.max_ring_size < 3:
        raise ValueError("--max-ring-size must be at least 3")
    if args.overlap < 0 or args.overlap >= args.patch_size:
        raise ValueError("overlap must be in [0, patch_size)")
    if not 0.0 <= args.dropout < 1.0 or not 0.0 <= args.branch_dropout < 1.0:
        raise ValueError("dropout values must be in [0, 1)")
    if (
        args.positive_weight < 0.0
        or args.fusion_scale < 0.0
        or args.aux_weight < 0.0
        or args.aux_decay_epochs < 0
    ):
        raise ValueError("loss and fusion scales must be non-negative")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
        from torch_geometric.loader import DataLoader
        from torch_geometric.utils import scatter
    except ImportError as exc:
        raise RuntimeError("CIN--Beam8 training requires torch, PyG and OGB") from exc

    started = time.time()
    device = torch.device(args.device)
    _seed_everything(args.seed, torch)
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise AssertionError("MolHIV categorical atom/bond features are required")
    labels = np.asarray(bundle.y, dtype=np.float64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64), expected_original
        ):
            raise ValueError("fold cache original_indices mismatch")
        if "official_train_indices" in folds.files and not np.array_equal(
            np.asarray(folds["official_train_indices"], dtype=np.int64), official_train
        ):
            raise ValueError("fold cache official_train_indices mismatch")
    if set(fit_indices) | set(heldout_indices) != set(official_train):
        raise AssertionError("internal scaffold fold does not partition official train")
    if set(fit_indices) & set(heldout_indices):
        raise AssertionError("fit and heldout folds overlap")
    forbidden = set(official_valid) | set(official_test)
    if (set(fit_indices) | set(heldout_indices)) & forbidden:
        raise AssertionError("official valid/test leakage")

    fit_indices = _stratified_limit(
        fit_indices, labels, args.limit_per_split, args.limit_seed + 101
    )
    heldout_indices = _stratified_limit(
        heldout_indices, labels, args.limit_per_split, args.limit_seed + 202
    )
    selected_indices = np.unique(np.concatenate([fit_indices, heldout_indices]))
    fit_positive = int(labels[fit_indices].sum())
    if fit_positive == 0:
        raise ValueError("fit fold has no positives")
    resolved_positive_weight = (
        (len(fit_indices) - fit_positive) / fit_positive
        if args.positive_weight == 0.0
        else args.positive_weight
    )

    atom_dims = tuple(int(value) for value in get_atom_feature_dims())
    bond_dims = tuple(int(value) for value in get_bond_feature_dims())
    incidence_items: dict[int, Any] = {}
    pyg_data: dict[int, Any] = {}
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
        pyg_data[graph_index] = to_cin_beam8_heterodata(
            item,
            bundle.graphs[graph_index],
            bundle.edge_feats[graph_index],
            max_ring_size=args.max_ring_size,
            shuffle_seed=20260815,
        )
        if count == 1 or count % 100 == 0 or count == len(selected_indices):
            print(f"prepared CIN--Beam8 molecules {count}/{len(selected_indices)}", flush=True)

    first = pyg_data[int(selected_indices[0])]
    position_dim = int(first["patch"].position.shape[1])

    class SafeBatchNorm1d(nn.BatchNorm1d):
        def forward(self, values: Any) -> Any:
            if values.shape[0] == 0:
                return values
            if self.training and values.shape[0] == 1:
                return F.batch_norm(
                    values,
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    False,
                    self.momentum,
                    self.eps,
                )
            return super().forward(values)

    def update_mlp(input_dim: int) -> Any:
        return nn.Sequential(
            nn.Linear(input_dim, args.hidden),
            SafeBatchNorm1d(args.hidden),
            nn.ReLU(),
            nn.Linear(args.hidden, args.hidden),
            SafeBatchNorm1d(args.hidden),
            nn.ReLU(),
        )

    class SparseCINLevel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_message = nn.Sequential(
                nn.Linear(2 * args.hidden, args.hidden), nn.ReLU()
            )
            self.up_update = update_mlp(args.hidden)
            self.boundary_update = update_mlp(args.hidden)
            self.combine = nn.Sequential(
                nn.Linear(2 * args.hidden, args.hidden),
                SafeBatchNorm1d(args.hidden),
                nn.ReLU(),
            )

        def forward(
            self,
            values: Any,
            up_index: Any,
            upper_values: Any,
            shared_index: Any,
            boundary_index: Any,
            lower_values: Any,
        ) -> Any:
            if values.shape[0] == 0:
                return values
            up = torch.zeros_like(values)
            if up_index.shape[1] > 0:
                source, target = up_index
                message = self.up_message(
                    torch.cat([values[source], upper_values[shared_index]], dim=1)
                )
                up = scatter(
                    message, target, dim=0, dim_size=len(values), reduce="sum"
                )
            boundary = torch.zeros_like(values)
            if boundary_index.shape[1] > 0:
                lower, target = boundary_index
                boundary = scatter(
                    lower_values[lower],
                    target,
                    dim=0,
                    dim_size=len(values),
                    reduce="sum",
                )
            up = self.up_update(up + values)
            boundary = self.boundary_update(boundary + values)
            return self.combine(torch.cat([up, boundary], dim=1))

    class SparseCINLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.levels = nn.ModuleList([SparseCINLevel() for _ in range(3)])

        def forward(
            self, atom_h: Any, bond_h: Any, ring_h: Any, data: Any, *, use_rings: bool
        ) -> tuple[Any, Any, Any]:
            atom_up = data["atom", "cell_upper", "atom"]
            atom_graph = data["atom"].batch[atom_up.edge_index[0]]
            atom_shared = atom_up.shared + data["bond_cell"].ptr[atom_graph]
            empty_index = atom_up.edge_index.new_empty((2, 0))
            atom_new = self.levels[0](
                atom_h,
                atom_up.edge_index,
                bond_h,
                atom_shared,
                empty_index,
                atom_h.new_empty((0, args.hidden)),
            )

            bond_boundary = data["atom", "cell_boundary", "bond_cell"].edge_index
            bond_up_store = data["bond_cell", "cell_upper", "bond_cell"]
            if use_rings and bond_up_store.edge_index.shape[1] > 0:
                bond_graph = data["bond_cell"].batch[bond_up_store.edge_index[0]]
                bond_shared = bond_up_store.shared + data["ring"].ptr[bond_graph]
                bond_up = bond_up_store.edge_index
            else:
                bond_up = empty_index
                bond_shared = empty_index.new_empty((0,))
            bond_new = self.levels[1](
                bond_h,
                bond_up,
                ring_h,
                bond_shared,
                bond_boundary,
                atom_h,
            )

            ring_boundary = data["bond_cell", "cell_boundary", "ring"].edge_index
            ring_new = self.levels[2](
                ring_h,
                empty_index,
                ring_h,
                empty_index.new_empty((0,)),
                ring_boundary,
                bond_h,
            )
            return atom_new, bond_new, ring_new

    class CINBeam8(nn.Module):
        def __init__(self, variant: str) -> None:
            super().__init__()
            self.variant = variant
            self.atom_encoder = AtomEncoder(args.hidden)
            self.bond_encoder = BondEncoder(args.hidden)
            self.layers = nn.ModuleList([SparseCINLayer() for _ in range(args.layers)])
            self.readout_projection = nn.ModuleList(
                [nn.Linear(args.hidden, 2 * args.hidden) for _ in range(3)]
            )
            self.head = nn.Linear(2 * args.hidden, 1)

            # Beam8 structure-endpoint branch.  These modules exist in every
            # variant so the CIN core receives identical random initialization.
            self.slot_embedding = nn.Embedding(args.patch_size, args.hidden)
            self.pair_embedding = nn.Embedding(
                args.patch_size * (args.patch_size - 1) // 2, args.hidden
            )
            self.internal_update = nn.Sequential(
                nn.Linear(args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.position_update = nn.Sequential(
                nn.Linear(position_dim, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.patch_role_embedding = nn.Embedding(2, args.hidden)
            self.patch_norm = nn.LayerNorm(args.hidden)
            self.endpoint_message = nn.Sequential(
                nn.Linear(3 * args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.endpoint_gate = nn.Linear(3 * args.hidden, 1)
            self.beam_fusion = nn.Linear(args.hidden, args.hidden, bias=False)
            nn.init.zeros_(self.beam_fusion.weight)
            self.bond_endpoint_message = nn.Sequential(
                nn.Linear(3 * args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.bond_endpoint_gate = nn.Linear(3 * args.hidden, 1)
            self.beam_bond_fusion = nn.Linear(args.hidden, args.hidden, bias=False)
            nn.init.zeros_(self.beam_bond_fusion.weight)
            pair_count = args.patch_size * (args.patch_size - 1) // 2
            self.aux_position_head = nn.Linear(args.hidden, position_dim)
            self.aux_pair_head = nn.Linear(args.hidden, pair_count)
            self.aux_completion_head = nn.Linear(args.hidden, 1)
            self.node_aux_position_head = nn.Linear(args.hidden, position_dim)
            self.node_aux_slot_head = nn.Linear(args.hidden, args.patch_size)
            self.node_aux_flags_head = nn.Linear(args.hidden, 3)
            self.node_aux_completion_head = nn.Linear(args.hidden, 1)
            self.node_aux_count_head = nn.Linear(args.hidden, 1)
            self.auxiliary_loss = None

        @staticmethod
        def _shuffle(values: Any, relation: Any) -> Any:
            source, target = relation.edge_index
            output = torch.zeros_like(values)
            output[target] = values[source]
            return output

        def _beam_context(self, atom_h: Any, data: Any, *, shuffled: bool, no_patch: bool) -> Any:
            patch_count = int(data["patch"].num_nodes)
            internal_store = data["patch", "internal_bond", "patch"]
            internal = atom_h.new_zeros((patch_count, args.hidden))
            if internal_store.edge_index.shape[1] > 0:
                patch_index = internal_store.edge_index[0]
                message = self.internal_update(
                    self.bond_encoder(internal_store.edge_attr)
                    + self.pair_embedding(internal_store.pair)
                )
                internal = scatter(
                    message, patch_index, dim=0, dim_size=patch_count, reduce="sum"
                )
                degree = scatter(
                    torch.ones_like(patch_index, dtype=atom_h.dtype),
                    patch_index,
                    dim=0,
                    dim_size=patch_count,
                    reduce="sum",
                ).clamp_min(1.0)
                internal = internal / torch.sqrt(degree).unsqueeze(-1)
            position = self.position_update(data["patch"].position)
            position = position + self.patch_role_embedding(
                data["patch"].is_completion.long()
            )
            if shuffled:
                relation = data["patch", "token_shuffle", "patch"]
                internal = self._shuffle(internal, relation)
                position = self._shuffle(position, relation)
            patch_h = self.patch_norm(internal + position)

            if bag:
                # A strict graph-level control: every bond receives the same
                # graph patch summary, conditioned only on its own CIN state.
                # Do not traverse endpoint incidence, canonical pair/slot, or
                # completion role here; each would retain localized binding.
                graph_count = int(data.y.numel())
                graph_context = scatter(
                    patch_h,
                    data["patch"].batch,
                    dim=0,
                    dim_size=graph_count,
                    reduce="mean",
                )
                empty_role = torch.zeros_like(bond_h)
                joined = torch.cat(
                    [
                        bond_h,
                        graph_context[data["bond_cell"].batch],
                        empty_role,
                    ],
                    dim=1,
                )
                return self.bond_endpoint_message(joined) * torch.sigmoid(
                    self.bond_endpoint_gate(joined)
                )

            store = data["patch", "bond_endpoint", "atom"]
            patch_index, atom_index = store.edge_index
            role = (
                self.bond_encoder(store.edge_attr)
                + self.pair_embedding(store.pair)
                + self.slot_embedding(store.slot)
                + self.patch_role_embedding(
                    data["patch"].is_completion[patch_index].long()
                )
            )
            patch_context = (
                torch.zeros_like(patch_h[patch_index])
                if no_patch
                else patch_h[patch_index]
            )
            joined = torch.cat([atom_h[atom_index], patch_context, role], dim=1)
            message = self.endpoint_message(joined) * torch.sigmoid(
                self.endpoint_gate(joined)
            )
            aggregate = scatter(
                message,
                atom_index,
                dim=0,
                dim_size=len(atom_h),
                reduce="sum",
            )
            degree = scatter(
                torch.ones_like(atom_index, dtype=atom_h.dtype),
                atom_index,
                dim=0,
                dim_size=len(atom_h),
                reduce="sum",
            ).clamp_min(1.0)
            return aggregate / torch.sqrt(degree).unsqueeze(-1)

        def _beam_bond_context(
            self,
            bond_h: Any,
            data: Any,
            *,
            shuffled: bool,
            bag: bool,
            no_patch: bool,
        ) -> Any:
            patch_count = int(data["patch"].num_nodes)
            internal_store = data["patch", "internal_bond", "patch"]
            internal = bond_h.new_zeros((patch_count, args.hidden))
            if internal_store.edge_index.shape[1] > 0:
                patch_index = internal_store.edge_index[0]
                message = self.internal_update(
                    self.bond_encoder(internal_store.edge_attr)
                    + self.pair_embedding(internal_store.pair)
                )
                internal = scatter(
                    message, patch_index, dim=0, dim_size=patch_count, reduce="sum"
                )
                degree = scatter(
                    torch.ones_like(patch_index, dtype=bond_h.dtype),
                    patch_index,
                    dim=0,
                    dim_size=patch_count,
                    reduce="sum",
                ).clamp_min(1.0)
                internal = internal / torch.sqrt(degree).unsqueeze(-1)
            position = self.position_update(data["patch"].position)
            position = position + self.patch_role_embedding(
                data["patch"].is_completion.long()
            )
            if shuffled:
                relation = data["patch", "token_shuffle", "patch"]
                internal = self._shuffle(internal, relation)
                position = self._shuffle(position, relation)
            patch_h = self.patch_norm(internal + position)

            store = data["patch", "bond_endpoint", "atom"]
            if store.edge_index.shape[1] == 0:
                return torch.zeros_like(bond_h)
            if store.edge_index.shape[1] % 2:
                raise ValueError("patch bond endpoints must occur in pairs")
            patch_rows = store.edge_index[0].reshape(-1, 2)
            bond_rows = store.bond_cell.reshape(-1, 2)
            pair_rows = store.pair.reshape(-1, 2)
            if not torch.all(patch_rows[:, 0] == patch_rows[:, 1]):
                raise ValueError("paired endpoints must belong to the same patch")
            if not torch.all(bond_rows[:, 0] == bond_rows[:, 1]):
                raise ValueError("paired endpoints must map to the same bond cell")
            if not torch.all(pair_rows[:, 0] == pair_rows[:, 1]):
                raise ValueError("paired endpoints must share a canonical bond pair")

            patch_index = patch_rows[:, 0]
            bond_index = bond_rows[:, 0]
            first = torch.arange(0, store.edge_index.shape[1], 2, device=bond_h.device)
            slot_rows = store.slot.reshape(-1, 2)
            role = (
                self.bond_encoder(store.edge_attr[first])
                + self.pair_embedding(pair_rows[:, 0])
                + 0.5
                * (
                    self.slot_embedding(slot_rows[:, 0])
                    + self.slot_embedding(slot_rows[:, 1])
                )
                + self.patch_role_embedding(
                    data["patch"].is_completion[patch_index].long()
                )
            )
            patch_context = (
                torch.zeros_like(patch_h[patch_index])
                if no_patch
                else patch_h[patch_index]
            )
            joined = torch.cat([bond_h[bond_index], patch_context, role], dim=1)
            message = self.bond_endpoint_message(joined) * torch.sigmoid(
                self.bond_endpoint_gate(joined)
            )
            aggregate = scatter(
                message,
                bond_index,
                dim=0,
                dim_size=len(bond_h),
                reduce="sum",
            )
            degree = scatter(
                torch.ones_like(bond_index, dtype=bond_h.dtype),
                bond_index,
                dim=0,
                dim_size=len(bond_h),
                reduce="sum",
            ).clamp_min(1.0)
            return aggregate / torch.sqrt(degree).unsqueeze(-1)

        def _beam_auxiliary_loss(
            self, bond_h: Any, data: Any, *, shuffled: bool
        ) -> Any:
            internal_store = data["patch", "internal_bond", "patch"]
            endpoint_store = data["patch", "bond_endpoint", "atom"]
            patch_count = int(data["patch"].num_nodes)
            if internal_store.edge_index.shape[1] == 0:
                return bond_h.sum() * 0.0
            if endpoint_store.edge_index.shape[1] != 2 * internal_store.edge_index.shape[1]:
                raise ValueError("internal bonds and paired endpoint rows disagree")
            bond_rows = endpoint_store.bond_cell.reshape(-1, 2)
            if not torch.all(bond_rows[:, 0] == bond_rows[:, 1]):
                raise ValueError("paired endpoints must map to one bond cell")
            patch_index = internal_store.edge_index[0]
            bond_index = bond_rows[:, 0]
            patch_h = scatter(
                bond_h[bond_index],
                patch_index,
                dim=0,
                dim_size=patch_count,
                reduce="mean",
            )

            position_target = data["patch"].position
            pair_count = args.patch_size * (args.patch_size - 1) // 2
            pair_target = bond_h.new_zeros((patch_count, pair_count))
            pair_target[patch_index, internal_store.pair] = 1.0
            completion_target = data["patch"].is_completion.float().view(-1, 1)
            if shuffled:
                relation = data["patch", "token_shuffle", "patch"]
                position_target = self._shuffle(position_target, relation)
                pair_target = self._shuffle(pair_target, relation)
                completion_target = self._shuffle(completion_target, relation)

            position_loss = F.smooth_l1_loss(
                self.aux_position_head(patch_h), position_target
            )
            pair_loss = F.binary_cross_entropy_with_logits(
                self.aux_pair_head(patch_h), pair_target
            )
            completion_loss = F.binary_cross_entropy_with_logits(
                self.aux_completion_head(patch_h), completion_target
            )
            return position_loss + pair_loss + completion_loss

        def _beam_node_auxiliary_loss(
            self,
            atom_h: Any,
            data: Any,
            *,
            shuffled: bool,
            objective: str = "all",
        ) -> Any:
            """Reconstruct Beam8 role distributions from first-layer atom states.

            The targets are aggregated to original atoms, so the Beam8 view is
            used only as label-free training supervision and is absent at
            inference.  The shuffled control permutes patch-level position and
            completion attributes while preserving atom/patch degree, slot and
            local chain-role marginals.
            """

            store = data["patch", "contains", "atom"]
            if store.edge_index.shape[1] == 0:
                return atom_h.sum() * 0.0
            patch_index, atom_index = store.edge_index
            atom_count = len(atom_h)
            position = data["patch"].position
            completion = data["patch"].is_completion.float().view(-1, 1)
            if shuffled:
                relation = data["patch", "token_shuffle", "patch"]
                position = self._shuffle(position, relation)
                completion = self._shuffle(completion, relation)

            slot_rows = F.one_hot(
                store.slot, num_classes=args.patch_size
            ).to(dtype=atom_h.dtype)
            ones = torch.ones_like(atom_index, dtype=atom_h.dtype)
            degree = scatter(
                ones,
                atom_index,
                dim=0,
                dim_size=atom_count,
                reduce="sum",
            ).clamp_min(1.0)

            def incidence_mean(values: Any) -> Any:
                aggregate = scatter(
                    values,
                    atom_index,
                    dim=0,
                    dim_size=atom_count,
                    reduce="sum",
                )
                return aggregate / degree.unsqueeze(-1)

            position_target = incidence_mean(position[patch_index])
            slot_target = incidence_mean(slot_rows)
            flags_target = incidence_mean(store.flags)
            completion_target = incidence_mean(completion[patch_index])
            count_target = torch.log1p(degree).view(-1, 1)

            position_loss = F.smooth_l1_loss(
                self.node_aux_position_head(atom_h), position_target
            )
            slot_loss = F.binary_cross_entropy_with_logits(
                self.node_aux_slot_head(atom_h), slot_target
            )
            flags_loss = F.binary_cross_entropy_with_logits(
                self.node_aux_flags_head(atom_h), flags_target
            )
            completion_loss = F.binary_cross_entropy_with_logits(
                self.node_aux_completion_head(atom_h), completion_target
            )
            count_loss = F.smooth_l1_loss(
                self.node_aux_count_head(atom_h), count_target
            )
            patch_loss = position_loss + completion_loss
            local_loss = slot_loss + flags_loss + count_loss
            if objective == "patch":
                return patch_loss
            if objective == "local":
                return local_loss
            if objective != "all":
                raise ValueError(f"unknown node auxiliary objective: {objective}")
            return patch_loss + local_loss

        def forward(self, data: Any) -> Any:
            self.auxiliary_loss = None
            atom_h = self.atom_encoder(data["atom"].x)
            bond_h = self.bond_encoder(data["bond_cell"].x)
            ring_boundary = data["bond_cell", "cell_boundary", "ring"].edge_index
            ring_h = atom_h.new_zeros((int(data["ring"].num_nodes), args.hidden))
            if ring_boundary.shape[1] > 0:
                lower, target = ring_boundary
                ring_h = scatter(
                    bond_h[lower],
                    target,
                    dim=0,
                    dim_size=int(data["ring"].num_nodes),
                    reduce="sum",
                ) / 2.0

            ring_shuffled = self.variant == "cin_ring_shuffled"
            if ring_shuffled:
                ring_h = self._shuffle(
                    ring_h, data["ring", "token_shuffle", "ring"]
                )
            atom_h = F.dropout(atom_h, p=0.0, training=self.training)
            bond_h = F.dropout(bond_h, p=0.0, training=self.training)
            ring_h = F.dropout(ring_h, p=0.0, training=self.training)

            use_rings = self.variant != "cin_no_rings"
            for layer_index, layer in enumerate(self.layers):
                atom_h, bond_h, ring_h = layer(
                    atom_h, bond_h, ring_h, data, use_rings=use_rings
                )
                if ring_shuffled:
                    ring_h = self._shuffle(
                        ring_h, data["ring", "token_shuffle", "ring"]
                    )
                atom_beam_variants = {
                    "cin_beam8",
                    "cin_beam8_shuffled",
                    "cin_beam8_no_patch",
                }
                bond_beam_variants = {
                    "cin_beam8_bond",
                    "cin_beam8_bond_shuffled",
                    "cin_beam8_bond_bag",
                    "cin_beam8_bond_no_patch",
                }
                if layer_index == 0 and self.variant in atom_beam_variants:
                    context = self._beam_context(
                        atom_h,
                        data,
                        shuffled=self.variant == "cin_beam8_shuffled",
                        no_patch=self.variant == "cin_beam8_no_patch",
                    )
                    context = F.dropout(
                        context, p=args.branch_dropout, training=self.training
                    )
                    atom_h = atom_h + args.fusion_scale * self.beam_fusion(context)
                if layer_index == 0 and self.variant in bond_beam_variants:
                    context = self._beam_bond_context(
                        bond_h,
                        data,
                        shuffled=self.variant == "cin_beam8_bond_shuffled",
                        bag=self.variant == "cin_beam8_bond_bag",
                        no_patch=self.variant == "cin_beam8_bond_no_patch",
                    )
                    context = F.dropout(
                        context, p=args.branch_dropout, training=self.training
                    )
                    bond_h = bond_h + args.fusion_scale * self.beam_bond_fusion(
                        context
                    )
                if layer_index == 0 and self.variant in {
                    "cin_beam8_aux",
                    "cin_beam8_aux_shuffled",
                    "cin_beam8_multiaux",
                    "cin_beam8_multiaux_shuffled",
                }:
                    self.auxiliary_loss = self._beam_auxiliary_loss(
                        bond_h,
                        data,
                        shuffled=self.variant in {
                            "cin_beam8_aux_shuffled",
                            "cin_beam8_multiaux_shuffled",
                        },
                    )
                if layer_index == 0 and self.variant in {
                    "cin_beam8_node_aux",
                    "cin_beam8_node_aux_shuffled",
                    "cin_beam8_node_aux_patch",
                    "cin_beam8_node_aux_patch_shuffled",
                    "cin_beam8_node_aux_local",
                    "cin_beam8_multiaux",
                    "cin_beam8_multiaux_shuffled",
                }:
                    node_auxiliary_loss = self._beam_node_auxiliary_loss(
                        atom_h,
                        data,
                        shuffled=self.variant in {
                            "cin_beam8_node_aux_shuffled",
                            "cin_beam8_node_aux_patch_shuffled",
                            "cin_beam8_multiaux_shuffled",
                        },
                        objective=(
                            "patch"
                            if self.variant in {
                                "cin_beam8_node_aux_patch",
                                "cin_beam8_node_aux_patch_shuffled",
                            }
                            else "local"
                            if self.variant == "cin_beam8_node_aux_local"
                            else "all"
                        ),
                    )
                    if self.auxiliary_loss is None:
                        self.auxiliary_loss = node_auxiliary_loss
                    else:
                        self.auxiliary_loss = 0.5 * (
                            self.auxiliary_loss + node_auxiliary_loss
                        )
                atom_h = F.dropout(atom_h, p=args.dropout, training=self.training)
                bond_h = F.dropout(bond_h, p=args.dropout, training=self.training)
                ring_h = F.dropout(ring_h, p=args.dropout, training=self.training)

            graph_count = int(data.y.numel())
            states = []
            for dim, (values, batch) in enumerate(
                (
                    (atom_h, data["atom"].batch),
                    (bond_h, data["bond_cell"].batch),
                    (ring_h, data["ring"].batch),
                )
            ):
                if dim == 2 and not use_rings:
                    continue
                pooled = scatter(
                    values,
                    batch,
                    dim=0,
                    dim_size=graph_count,
                    reduce="mean",
                )
                states.append(F.relu(self.readout_projection[dim](pooled)))
            graph_h = torch.stack(states, dim=0).sum(dim=0)
            graph_h = F.dropout(graph_h, p=args.dropout, training=self.training)
            return self.head(graph_h).view(-1)

    ssl_checkpoint_by_variant = {
        "cin_ssl_beam": args.ssl_beam_checkpoint,
        "cin_ssl_beam_shuffled": args.ssl_beam_shuffled_checkpoint,
        "cin_ssl_random_bfs": args.ssl_random_bfs_checkpoint,
        "cin_ssl_balanced_bfs": args.ssl_balanced_bfs_checkpoint,
    }

    def initialize_chemistry_embeddings(
        model: Any, checkpoint_path: Path, expected_ssl_variant: str
    ) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("variant") != expected_ssl_variant:
            raise ValueError(
                f"SSL checkpoint variant mismatch: expected {expected_ssl_variant}, "
                f"got {checkpoint.get('variant')}"
            )
        if int(checkpoint.get("hidden", -1)) != args.hidden:
            raise ValueError("SSL checkpoint hidden dimension mismatch")
        if int(checkpoint.get("fold", -1)) != args.fold:
            raise ValueError("SSL checkpoint scaffold fold mismatch")
        if checkpoint.get("cover_canonicalization") != "topology_only":
            raise ValueError("only topology-only SSL checkpoints may initialize CIN")
        state = checkpoint["state_dict"]
        copied = []
        with torch.no_grad():
            for field, embedding in enumerate(model.atom_encoder.atom_embedding_list):
                source = state[f"atom_encoder.embeddings.{field}.weight"]
                source = source[: embedding.weight.shape[0]]
                if source.shape != embedding.weight.shape:
                    raise ValueError("SSL atom embedding shape mismatch")
                embedding.weight.copy_(source)
                copied.append(f"atom_embedding_{field}")
            for field, embedding in enumerate(model.bond_encoder.bond_embedding_list):
                source = state[f"bond_encoder.bond_embedding_list.{field}.weight"]
                if source.shape != embedding.weight.shape:
                    raise ValueError("SSL bond embedding shape mismatch")
                embedding.weight.copy_(source)
                copied.append(f"bond_embedding_{field}")
        fit_values = np.asarray(checkpoint["fit_indices"], dtype=np.int64)
        if not set(fit_values).issubset(set(fit_indices)):
            # Fine-tuning may use a limited subset different from the SSL
            # reservoir, but the checkpoint must remain inside the fold fit set.
            with np.load(args.fold_cache, allow_pickle=False) as folds:
                full_fit = np.asarray(
                    folds[f"fold_{args.fold}_train_indices"], dtype=np.int64
                )
            if not set(fit_values).issubset(set(full_fit)):
                raise ValueError("SSL checkpoint contains non-fit-fold graphs")
        return {
            "checkpoint": str(checkpoint_path),
            "ssl_variant": expected_ssl_variant,
            "ssl_fit_graphs": int(len(fit_values)),
            "copied_parameters": copied,
            "cover_canonicalization": checkpoint["cover_canonicalization"],
        }

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
        result = {
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "probability_sha256": _sha256(probability),
        }
        if args.save_predictions:
            result.update(
                {
                    "graph_indices": [int(index) for index in indices],
                    "labels": y.tolist(),
                    "probabilities": probability.tolist(),
                }
            )
        return result

    results: dict[str, Any] = {}
    for variant_index, variant in enumerate(variants):
        _seed_everything(args.seed, torch)
        model = CINBeam8(variant).to(device)
        ssl_initialization = None
        if variant in ssl_checkpoint_by_variant:
            checkpoint_path = ssl_checkpoint_by_variant[variant]
            if checkpoint_path is None:
                raise ValueError(f"{variant} requires its SSL checkpoint argument")
            expected = {
                "cin_ssl_beam": "multi_beam",
                "cin_ssl_beam_shuffled": "multi_beam_shuffled",
                "cin_ssl_random_bfs": "multi_random_bfs",
                "cin_ssl_balanced_bfs": "multi_balanced_bfs",
            }[variant]
            ssl_initialization = initialize_chemistry_embeddings(
                model, checkpoint_path, expected
            )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        train_loader = make_loader(fit_indices, shuffle=True, offset=0)
        positive_weight = torch.tensor(resolved_positive_weight, device=device)
        history = []
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
                effective_aux_weight = args.aux_weight
                if args.aux_decay_epochs > 0:
                    effective_aux_weight *= max(
                        0.0,
                        1.0 - (epoch - 1) / float(args.aux_decay_epochs),
                    )
                if model.auxiliary_loss is not None:
                    loss = loss + effective_aux_weight * model.auxiliary_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(y)
                n_seen += len(y)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(n_seen, 1),
                    "effective_aux_weight": effective_aux_weight,
                }
            )
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"variant={variant} epoch={epoch:03d} "
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
            "beam_fusion_norm": float(model.beam_fusion.weight.norm().item()),
            "beam_bond_fusion_norm": float(
                model.beam_bond_fusion.weight.norm().item()
            ),
            "ssl_initialization": ssl_initialization,
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

    paired: dict[str, Any] = {}
    if "cin" in results:
        for variant in variants:
            paired[variant] = {
                "roc_auc_delta_vs_cin": (
                    results[variant]["heldout"]["roc_auc"]
                    - results["cin"]["heldout"]["roc_auc"]
                ),
                "average_precision_delta_vs_cin": (
                    results[variant]["heldout"]["average_precision"]
                    - results["cin"]["heldout"]["average_precision"]
                ),
            }
    output = {
        "protocol_id": "molhiv-official-train-internal-cin-beam8-v1",
        "date": "2026-08-15",
        "scope": "official-train internal scaffold folds only",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "official_cwn_commit": "c4ddd24e251929f934f8a2467da98fdba376864d",
        "implementation_note": (
            "PyG-2 dependency-light Sparse CIN-compatible operator; not a bit-for-bit "
            "reproduction of the historical PyG-1.6 implementation"
        ),
        "config": {
            **vars(args),
            "variants": list(variants),
            "resolved_positive_weight": resolved_positive_weight,
            "ssl_beam_checkpoint": (
                None
                if args.ssl_beam_checkpoint is None
                else str(args.ssl_beam_checkpoint)
            ),
            "ssl_beam_shuffled_checkpoint": (
                None
                if args.ssl_beam_shuffled_checkpoint is None
                else str(args.ssl_beam_shuffled_checkpoint)
            ),
            "ssl_random_bfs_checkpoint": (
                None
                if args.ssl_random_bfs_checkpoint is None
                else str(args.ssl_random_bfs_checkpoint)
            ),
            "ssl_balanced_bfs_checkpoint": (
                None
                if args.ssl_balanced_bfs_checkpoint is None
                else str(args.ssl_balanced_bfs_checkpoint)
            ),
        },
        "split": {
            "fit_graphs": len(fit_indices),
            "fit_positive": fit_positive,
            "heldout_graphs": len(heldout_indices),
            "heldout_positive": int(labels[heldout_indices].sum()),
        },
        "beam8_summary": incidence_summary(list(incidence_items.values())),
        "ring_summary": ring_complex_summary(list(pyg_data.values())),
        "results": results,
        "paired": paired,
        "elapsed_sec": time.time() - started,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
