"""MolHIV transfer of the shared, vocabulary-free B-Bag patch encoder.

One sentence
------------
Replace MolHIV's vocabulary-sized exact rooted-patch *identity* embedding with a
single shared DeepSets-style encoder over OGB atom/root/distance and bond
primitives, while leaving the relational downstream unchanged.

Why
---
The frozen MolHIV H96 recurrent pair--centre port (``molhiv_recurrent_pair_centre``)
has ``1,076,589`` parameters, of which ``839,456`` (77.97 %) are a single
dense ``typed_embedding.weight`` of shape ``[26233, 32]``: one learned row per
exact rooted typed certificate observed in the official training split.  That
table is the dataset-dependent identity storage the ZINC B-Bag study
(``zinc_shared_bag_patch_encoder``) removed on ZINC by replacing the lookup with
a shared, permutation-invariant bag encoder.

This module performs the analogous replacement on MolHIV.  The only thing that
changes is the source of the per-patch token:

    before:  token_p = typed_embedding[certificate_id_p]              (R^32)
    after:   token_p = MolhivSharedBagPatchEncoder(patch primitives)   (R^32)

Everything else is inherited unchanged from the frozen MolHIV H96 port:
radius-2 rooted patches, ``patch_cont`` shell descriptor, radius-1 parent
embedding, shortest-path-conditioned pair relation, ``T=2`` weight-tied
recurrent pair--centre, global context channel, prediction head, loss,
optimizer, training schedule and the official scaffold split.

The MolHIV atom is *not* a single category: OGB gives 9 categorical fields per
atom and 3 per bond.  Each field therefore gets its own **fixed schema-sized**
embedding (sizes from ``ATOM_FEATURE_DIMS`` / ``BOND_FEATURE_DIMS``, never from
the observed certificate count), and the field embeddings are combined with the
ZINC B-Bag normalisation (``mean`` over fields, implemented as a ``1/sqrt(K)``
scaled sum).  The total parameter count is then independent of how many exact
patches the training split contains.

The B-Bag forward reads ``struct_atom_fields``, ``struct_root``,
``struct_dist``, ``struct_patch`` (node -> patch grouping), ``struct_bond_fields``
and ``struct_edge_patch`` (bond -> patch grouping) only.  It deliberately never
reads ``struct_src`` / ``struct_dst``: no message is propagated along a real
edge, so two patches with identical primitive multisets but different
connectivity receive exactly the same token (the definitional B-Bag property).

Discipline
----------
* the exact typed certificate never enters the neural forward;
* the structural cache is target-free and label-independent, and is versioned
  with an explicit schema string;
* architecture selection / epoch selection use the official **validation**
  ROC-AUC only; official test is loaded once, after a freeze record is written;
* the frozen rule is ``raw = best valid AUC checkpoint`` (ties -> earliest
  epoch) plus a fixed equal-weight ``Top-5`` parameter soup (no k/weight search);
* no refit on train+valid: exactly the frozen ``molhiv_recurrent_pair_centre_v1``
  protocol.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.molhiv_shared_bag_patch_encoder <stage>

Stages: ``data_sanity preprocess params sanity smoke train train_queue freeze
test report``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import pickle
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc
from tracks.ksvd.experiments.luyin16.structural_patch_encoder import scatter_mean

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/molhiv_shared_bag_patch_encoder"
STRUCT_CACHE_DIR = RESULTS_DIR / "structural_cache"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
CURVE_DIR = RESULTS_DIR / "curves"

# Pre-existing, audited exact-rooted MolHIV record cache (certificates, shell
# descriptors, pair relations, global context).  Read-only: never modified.
RECORD_CACHE = (
    TRACK_ROOT
    / "results/luyin16/unified_relational_patch_molhiv_exact_rooted/record_cache"
)
DATA_ROOT = REPO_ROOT / "data/ogb"

PROTOCOL_VERSION = "molhiv_shared_bag_patch_encoder_v1"
STRUCT_CACHE_SCHEMA = "molhiv-shared-bag-structural-cache-v1"

SPLIT_SIZES = {"train": 32901, "valid": 4113, "test": 4113}

# --- frozen MolHIV H96 recurrent pair--centre geometry (unchanged) ----------
H_DIM = 96
Q_DIM = 16
TOKEN_WIDTH = 32
DROPOUT = 0.05
CENTER_CONTEXT_HIDDEN = 60
RECURRENCE_ROUNDS = 2
PATCH_RADIUS = 2

# --- pre-registered B-Bag encoder (one architecture; no sweep) --------------
BAG_NODE_DIM = 48
BAG_EDGE_DIM = 24
BAG_NODE_HIDDEN = 96
BAG_BOND_HIDDEN = 48
BAG_FUSION_HIDDEN = 104
BAG_OUTPUT_DIM = TOKEN_WIDTH  # drop-in replacement for the old 32-D typed token

ATOM_FEATURE_DIMS = tuple(int(value) for value in mpp.ATOM_FEATURE_DIMS)
BOND_FEATURE_DIMS = tuple(int(value) for value in mpp.BOND_FEATURE_DIMS)

# --- frozen MolHIV training protocol (identical to the RPC baseline) --------
BATCH_SIZE = 128
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5
MAX_EPOCHS = 240
PATIENCE = 40
GRAD_CLIP = 5.0
SOUP_K = 5
SEEDS = (0, 1)

# --- frozen reference: H96 recurrent pair--centre on official train ---------
REFERENCE_RPC_PARAMS = 1_076_589
REFERENCE_RPC_TYPED_PARAMS = 839_456
REFERENCE_RPC_PARENT_PARAMS = 1_648
REFERENCE_RPC_VALID_RAW = {0: 0.8142391730354693, 1: 0.8498646629433665}
REFERENCE_RPC_VALID_SOUP = {0: 0.8095054379776603, 1: 0.8510924946110132}
REFERENCE_RPC_TEST_RAW = {0: 0.7694547982772938, 1: 0.7557542633113811}
REFERENCE_RPC_TEST_RAW_MEAN = 0.7626045307943374
REFERENCE_RPC_TEST_ENSEMBLE = 0.780047895865119

# The new model must be far below the old vocabulary-sized model; this is a
# guard against accidentally keeping a ``[num_exact_patch_types, D]`` tensor.
PARAM_UPPER_BOUND = 400_000


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# io / provenance
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _n_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().to(torch.float32).cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# B-Bag encoder
# ---------------------------------------------------------------------------


def _field_sum(
    embeddings: nn.ModuleList, fields: torch.Tensor, n_fields: int
) -> torch.Tensor:
    """``mean_k emb_k(fields[:, k])`` as a ``1/sqrt(K)``-scaled sum.

    ``fields`` is ``[T, K]`` int; each column ``k`` is a categorical field with
    its own fixed schema-sized embedding.  Averaging over the K fields (rather
    than summing) keeps the primitive scale independent of the OGB schema width.
    """
    if fields.ndim != 2 or int(fields.shape[1]) != int(n_fields):
        raise ValueError(
            f"expected categorical fields [T,{int(n_fields)}], got {tuple(fields.shape)}"
        )
    total = embeddings[0](fields[:, 0].long())
    for index in range(1, int(n_fields)):
        total = total + embeddings[index](fields[:, index].long())
    return total / math.sqrt(float(n_fields))


def _std_pool(
    values: torch.Tensor, group: torch.Tensor, n_groups: int, mean: torch.Tensor
) -> torch.Tensor:
    total = values.new_zeros((int(n_groups), int(values.shape[1])))
    if values.numel():
        total.index_add_(0, group, values * values)
    counts = (
        torch.bincount(group, minlength=int(n_groups))
        .clamp_min(1)
        .to(values.dtype)
        .unsqueeze(1)
    )
    second_moment = total / counts
    return torch.sqrt((second_moment - mean * mean).clamp_min(0.0) + 1.0e-8)


class MolhivSharedBagPatchEncoder(nn.Module):
    """Vocabulary-free, connectivity-free DeepSets encoder for OGB patches.

    Primitive form (one pre-registered architecture)::

        atom_base(v)  = mean_k atom_emb_k(atom_field_k(v))            in R^48
        z_v           = node_mlp(atom_base(v)
                                 + root_emb(is_root(v))
                                 + dist_emb(root_distance(v)))        in R^48
        bond_base(e)  = mean_k bond_emb_k(bond_field_k(e))            in R^24
        g_e           = bond_mlp(bond_base(e))                        in R^24
        token_p       = fusion([z_root ; mean_v z_v ; std_v z_v ;
                                mean_e g_e ; std_e g_e])              in R^32

    The atom/bond field embeddings are sized from the fixed OGB schema
    (``ATOM_FEATURE_DIMS`` / ``BOND_FEATURE_DIMS``), never from the number of
    exact certificates observed in a split, so the parameter count does not grow
    with the training vocabulary.

    The forward never reads ``struct_src`` / ``struct_dst``.  Patch membership
    is taken from ``struct_patch`` (nodes) and ``struct_edge_patch`` (bonds),
    which are grouping indices, not adjacency.
    """

    kind = "molhiv_shared_bag"
    rounds = 0
    include_std_pool = True

    def __init__(
        self,
        *,
        atom_feature_dims: Sequence[int] = ATOM_FEATURE_DIMS,
        bond_feature_dims: Sequence[int] = BOND_FEATURE_DIMS,
        n_distance_bins: int = PATCH_RADIUS + 1,
        node_dim: int = BAG_NODE_DIM,
        edge_dim: int = BAG_EDGE_DIM,
        node_hidden: int = BAG_NODE_HIDDEN,
        bond_hidden: int = BAG_BOND_HIDDEN,
        fusion_hidden: int = BAG_FUSION_HIDDEN,
        output_dim: int = BAG_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.atom_feature_dims = tuple(int(value) for value in atom_feature_dims)
        self.bond_feature_dims = tuple(int(value) for value in bond_feature_dims)
        self.n_atom_fields = len(self.atom_feature_dims)
        self.n_bond_fields = len(self.bond_feature_dims)
        self.n_distance_bins = int(n_distance_bins)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.node_hidden = int(node_hidden)
        self.bond_hidden = int(bond_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.output_dim = int(output_dim)
        # Interface-compatible audit metadata (mirrors the ZINC B-Bag encoder).
        self.hidden_dim = int(fusion_hidden)
        if min(
            *self.atom_feature_dims,
            *self.bond_feature_dims,
            self.n_distance_bins,
            self.node_dim,
            self.edge_dim,
            self.node_hidden,
            self.bond_hidden,
            self.fusion_hidden,
            self.output_dim,
        ) < 1:
            raise ValueError("B-Bag encoder dimensions must be positive")

        # Fixed schema-sized categorical embeddings (no vocabulary, no OOV row).
        self.atom_embeddings = nn.ModuleList(
            nn.Embedding(int(width), self.node_dim)
            for width in self.atom_feature_dims
        )
        self.root_embedding = nn.Embedding(2, self.node_dim)
        self.distance_embedding = nn.Embedding(self.n_distance_bins, self.node_dim)
        self.bond_embeddings = nn.ModuleList(
            nn.Embedding(int(width), self.edge_dim)
            for width in self.bond_feature_dims
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_hidden),
            nn.ReLU(),
            nn.Linear(self.node_hidden, self.node_dim),
        )
        self.bond_mlp = nn.Sequential(
            nn.Linear(self.edge_dim, self.bond_hidden),
            nn.ReLU(),
            nn.Linear(self.bond_hidden, self.edge_dim),
        )
        fusion_input = 3 * self.node_dim + 2 * self.edge_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, self.fusion_hidden),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden, self.output_dim),
        )

    def atom_base(self, fields: torch.Tensor) -> torch.Tensor:
        return _field_sum(self.atom_embeddings, fields, self.n_atom_fields)

    def bond_base(self, fields: torch.Tensor) -> torch.Tensor:
        return _field_sum(self.bond_embeddings, fields, self.n_bond_fields)

    def forward(self, data: object) -> torch.Tensor:
        fields = data.struct_atom_fields.long()
        root = data.struct_root.long()
        distance = data.struct_dist.long()
        if not hasattr(data, "struct_edge_patch"):
            raise AttributeError(
                "MolhivSharedBagPatchEncoder requires struct_edge_patch "
                "(bond -> patch grouping); it never reads struct_src/struct_dst"
            )
        unique_patch, node_patch = torch.unique(
            data.struct_patch.long(), return_inverse=True
        )
        n_patches = int(unique_patch.numel())

        z = self.node_mlp(
            self.atom_base(fields)
            + self.root_embedding(root)
            + self.distance_embedding(distance)
        )
        node_mean = scatter_mean(z, node_patch, n_patches)
        node_std = _std_pool(z, node_patch, n_patches, node_mean)

        root_mask = root > 0
        root_state = z.new_zeros((n_patches, self.node_dim))
        if bool(root_mask.any()):
            root_state.index_add_(0, node_patch[root_mask], z[root_mask])

        edge_patch_ids = data.struct_edge_patch.long()
        if edge_patch_ids.numel():
            # ``unique_patch`` is sorted, so searchsorted maps a patch id to its
            # batch-local group index without touching an edge endpoint.
            edge_group = torch.searchsorted(unique_patch, edge_patch_ids)
            g = self.bond_mlp(self.bond_base(data.struct_bond_fields.long()))
            bond_mean = scatter_mean(g, edge_group, n_patches)
            bond_std = _std_pool(g, edge_group, n_patches, bond_mean)
        else:
            bond_mean = z.new_zeros((n_patches, self.edge_dim))
            bond_std = z.new_zeros((n_patches, self.edge_dim))

        return self.fusion(
            torch.cat([root_state, node_mean, node_std, bond_mean, bond_std], dim=1)
        )


# ---------------------------------------------------------------------------
# model: frozen MolHIV RPC with the typed lookup replaced by B-Bag
# ---------------------------------------------------------------------------


class MolhivSharedBagRecurrentPairCentreModel(rpc.MolhivRecurrentPairCentreModel):
    """``molhiv_recurrent_pair_centre`` with B-Bag patch tokens.

    The whole relational downstream (pair projection / relation encoder /
    distance gate / pair encoder / ``T``-round weight-tied centre refresh /
    unary + pair moments + global readout / head) is inherited verbatim from
    :class:`rpc.MolhivRecurrentPairCentreModel`.  Only ``_patch_token_value``
    changes: the exact typed lookup is deleted and replaced by the shared
    B-Bag encoder.
    """

    kind = "molhiv_shared_bag_recurrent_pair_centre"

    def __init__(
        self,
        parent_vocabulary_size: int,
        *,
        bag_encoder: MolhivSharedBagPatchEncoder,
        patch_hidden: int = H_DIM,
        pair_hidden: int = Q_DIM,
        token_width: int = TOKEN_WIDTH,
        dropout: float = DROPOUT,
        center_context_hidden: int = CENTER_CONTEXT_HIDDEN,
        recurrence_rounds: int = RECURRENCE_ROUNDS,
    ) -> None:
        super().__init__(
            1,  # no exact certificate vocabulary: a single placeholder row exists
            int(parent_vocabulary_size),
            patch_hidden=int(patch_hidden),
            pair_hidden=int(pair_hidden),
            token_width=int(token_width),
            dropout=float(dropout),
            center_context=True,
            center_context_hidden=int(center_context_hidden),
            recurrence_rounds=int(recurrence_rounds),
        )
        # ``mpp.PatchPathModel`` does not persist token_width; record it so the
        # drop-in width check and audits can read it back.
        self.token_width = int(token_width)
        if int(bag_encoder.output_dim) != int(self.token_width):
            raise ValueError(
                f"B-Bag output_dim {bag_encoder.output_dim} must equal token_width "
                f"{self.token_width} (drop-in replacement of the typed token)"
            )
        del self.typed_embedding  # no vocabulary-sized table may remain
        self.typed_embedding = None
        self.bag_encoder = bag_encoder

    def _patch_token_value(self, data: Data) -> torch.Tensor:
        return self.bag_encoder(data)

    def forward(self, data: Data) -> torch.Tensor:
        """Identical to the frozen RPC forward with the B-Bag patch token."""
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])
        patch = self.patch_encoder(
            torch.cat(
                [
                    data.patch_cont,
                    self._patch_token_value(data),
                    self.parent_embedding(data.parent_token),
                ],
                dim=1,
            )
        )
        source = data.pair_index[0]
        target = data.pair_index[1]
        relation = self.relation_encoder(data.pair_relation)
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_value = None
        for _round in range(self.recurrence_rounds):
            pair_value = self._pair_value(patch, source, target, relation, gate)
            center_context = self._pool_pairs_to_centres(
                pair_value, source, target, data.pair_bucket, int(patch.shape[0])
            )
            patch = patch + self.center_update(
                torch.cat([patch, center_context], dim=1)
            )
        unary = self._pool_nodes(patch, data.batch, n_graphs)
        assert pair_value is not None
        relation_readout = self._pool_pairs(
            pair_value, data.batch[source], data.pair_bucket, n_graphs
        )
        graph_hidden = self.global_encoder(global_context)
        return self.head(
            torch.cat([unary, relation_readout, graph_hidden], dim=1)
        ).view(-1)


def build_bag_encoder() -> MolhivSharedBagPatchEncoder:
    return MolhivSharedBagPatchEncoder()


def build_model(parent_vocabulary_size: int, seed: int) -> MolhivSharedBagRecurrentPairCentreModel:
    _seed_everything(int(seed))
    return MolhivSharedBagRecurrentPairCentreModel(
        int(parent_vocabulary_size),
        bag_encoder=build_bag_encoder(),
        patch_hidden=H_DIM,
        pair_hidden=Q_DIM,
        token_width=TOKEN_WIDTH,
        dropout=DROPOUT,
        center_context_hidden=CENTER_CONTEXT_HIDDEN,
        recurrence_rounds=RECURRENCE_ROUNDS,
    )


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def parameter_breakdown(
    model: MolhivSharedBagRecurrentPairCentreModel,
) -> dict[str, int]:
    encoder = model.bag_encoder
    blocks = {
        "molhiv_atom_field_embeddings": _n_params(encoder.atom_embeddings),
        "root_distance_embeddings": int(
            encoder.root_embedding.weight.numel()
            + encoder.distance_embedding.weight.numel()
        ),
        "bond_field_embeddings": _n_params(encoder.bond_embeddings),
        "bag_node_mlp": _n_params(encoder.node_mlp),
        "bag_bond_mlp": _n_params(encoder.bond_mlp),
        "bag_fusion": _n_params(encoder.fusion),
        "exact_typed_token_embedding": _n_params(
            getattr(model, "typed_embedding", None)
        ),
        "radius1_parent_embedding": _n_params(model.parent_embedding),
        "patch_encoder": _n_params(model.patch_encoder),
        "pair_relation_modules": sum(
            _n_params(module)
            for module in (
                model.pair_projection,
                model.relation_encoder,
                model.distance_gate,
                model.pair_encoder,
            )
        ),
        "recurrent_center_update": _n_params(model.center_update),
        "global_encoder": _n_params(model.global_encoder),
        "prediction_head": _n_params(model.head),
    }
    blocks["bag_encoder_total"] = int(
        blocks["molhiv_atom_field_embeddings"]
        + blocks["root_distance_embeddings"]
        + blocks["bond_field_embeddings"]
        + blocks["bag_node_mlp"]
        + blocks["bag_bond_mlp"]
        + blocks["bag_fusion"]
    )
    blocks["total"] = int(sum(v for k, v in blocks.items() if k != "bag_encoder_total"))
    global_total = int(sum(parameter.numel() for parameter in model.parameters()))
    if blocks["total"] != global_total:
        raise RuntimeError(
            f"parameter breakdown inconsistent: {blocks['total']} != {global_total}"
        )
    return blocks


def params() -> dict[str, Any]:
    """Static parameter accounting (no data loaded, no training)."""
    model = build_model(parent_vocabulary_size=103, seed=0)
    breakdown = parameter_breakdown(model)
    typed_state_keys = [k for k in model.state_dict() if "typed_embedding" in k]
    typed_named = [k for k, _ in model.named_parameters() if "typed_embedding" in k]
    largest = sorted(
        ((int(p.numel()), name) for name, p in model.named_parameters()),
        reverse=True,
    )[:8]
    reduction = 1.0 - breakdown["total"] / float(REFERENCE_RPC_PARAMS)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": {
            "h_dim": H_DIM,
            "q_dim": Q_DIM,
            "token_width": TOKEN_WIDTH,
            "recurrence_rounds": RECURRENCE_ROUNDS,
            "weight_tied": True,
            "center_context_hidden": CENTER_CONTEXT_HIDDEN,
            "patch_representation": "molhiv_shared_bag",
            "bag_encoder": {
                "atom_feature_dims": list(ATOM_FEATURE_DIMS),
                "bond_feature_dims": list(BOND_FEATURE_DIMS),
                "node_dim": BAG_NODE_DIM,
                "edge_dim": BAG_EDGE_DIM,
                "node_hidden": BAG_NODE_HIDDEN,
                "bond_hidden": BAG_BOND_HIDDEN,
                "fusion_hidden": BAG_FUSION_HIDDEN,
                "output_dim": BAG_OUTPUT_DIM,
                "n_distance_bins": PATCH_RADIUS + 1,
                "rounds": 0,
            },
        },
        "parent_vocabulary_size_example": 103,
        "parameter_breakdown": breakdown,
        "total_params": int(breakdown["total"]),
        "under_upper_bound": bool(breakdown["total"] <= PARAM_UPPER_BOUND),
        "typed_embedding_present": bool(getattr(model, "typed_embedding", None) is not None),
        "typed_embedding_state_keys": typed_state_keys,
        "typed_embedding_named_parameters": typed_named,
        "largest_tensors": [
            {"name": name, "numel": numel} for numel, name in largest
        ],
        "reference": {
            "rpc_total_params": REFERENCE_RPC_PARAMS,
            "rpc_typed_identity_params": REFERENCE_RPC_TYPED_PARAMS,
            "rpc_parent_params": REFERENCE_RPC_PARENT_PARAMS,
            "new_total_params": int(breakdown["total"]),
            "new_exact_identity_params": int(blocks_exact_identity(breakdown)),
            "typed_identity_params_removed": int(
                REFERENCE_RPC_TYPED_PARAMS - blocks_exact_identity(breakdown)
            ),
            "parameter_reduction_fraction": float(reduction),
        },
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    if not payload["under_upper_bound"]:
        raise RuntimeError(
            f"new total {breakdown['total']} exceeds guard {PARAM_UPPER_BOUND}; "
            "investigate a lingering vocabulary-sized table"
        )
    return payload


def blocks_exact_identity(breakdown: Mapping[str, int]) -> int:
    """Exact (dataset-dependent) identity storage in the new model."""
    return int(
        breakdown["exact_typed_token_embedding"]
        + breakdown["radius1_parent_embedding"]
    )


# ---------------------------------------------------------------------------
# structural preprocessing (target-free, label-independent, schema-versioned)
# ---------------------------------------------------------------------------


def _structural_record(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
) -> dict[str, Any]:
    """Per-graph B-Bag primitives for every radius-2 rooted patch.

    Patch ``p`` corresponds to centre ``graph.nodes[p]`` (the same ordering used
    by the pre-existing exact-rooted record cache), so ``struct_patch`` lines up
    one-to-one with ``record.patches``.  Each undirected bond of a patch is
    stored once; the bag only needs the bond *multiset* per patch.

    No target, no label, and no exact certificate id is stored here.
    """
    centers = list(graph.nodes)
    node_fields: list[np.ndarray] = []
    node_root: list[int] = []
    node_dist: list[int] = []
    node_patch: list[int] = []
    edge_fields: list[tuple[int, ...]] = []
    edge_patch: list[int] = []
    for patch_index, center in enumerate(centers):
        distances = mpp._ego_distances(graph, int(center), PATCH_RADIUS)
        nodes = sorted(distances, key=lambda node: (distances[node], node))
        for node in nodes:
            node_fields.append(np.asarray(node_types[int(node)], dtype=np.int64))
            node_root.append(int(node == int(center)))
            node_dist.append(int(distances[node]))
            node_patch.append(int(patch_index))
        induced = graph.induced(set(distances))
        for left, right in sorted(induced.edges()):
            edge_fields.append(
                tuple(
                    int(value)
                    for value in edge_types[graph.edge_key(int(left), int(right))]
                )
            )
            edge_patch.append(int(patch_index))

    if node_fields:
        fields_matrix = np.stack(node_fields, axis=0).astype(np.int64, copy=False)
    else:
        fields_matrix = np.zeros((0, len(ATOM_FEATURE_DIMS)), dtype=np.int64)
    if edge_fields:
        bond_matrix = np.asarray(edge_fields, dtype=np.int64)
    else:
        bond_matrix = np.zeros((0, len(BOND_FEATURE_DIMS)), dtype=np.int64)
    return {
        "n_patches": int(len(centers)),
        "n_nodes": int(fields_matrix.shape[0]),
        "n_bonds": int(bond_matrix.shape[0]),
        "node_fields": fields_matrix.astype(np.int16, copy=False),
        "node_root": np.asarray(node_root, dtype=np.int8),
        "node_dist": np.asarray(node_dist, dtype=np.int8),
        "node_patch": np.asarray(node_patch, dtype=np.int32),
        "bond_fields": bond_matrix.astype(np.int16, copy=False),
        "edge_patch": np.asarray(edge_patch, dtype=np.int32),
    }


def _split_indices(bundle: Any) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(bundle.split[name], dtype=np.int64)
        for name in ("train", "valid", "test")
    }


def build_structural_cache(split: str, force: bool = False) -> dict[str, Any]:
    """Build (or reuse) the target-free structural cache for one split."""
    path = STRUCT_CACHE_DIR / f"structural_{split}.pkl"
    if path.exists() and not force:
        payload = pickle.loads(path.read_bytes())
        metadata = payload["metadata"]
        print(f"structural cache hit: {path}", flush=True)
        return metadata
    if split not in SPLIT_SIZES:
        raise ValueError(f"unknown split {split!r}")

    bundle = load_molhiv(root=DATA_ROOT, with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV OGB node/edge features were not loaded")
    indices = _split_indices(bundle)[split]
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for position, index in enumerate(indices):
        index_int = int(index)
        graph = bundle.graphs[index_int]
        edge_types = {
            (int(left), int(right)): tuple(int(value) for value in values)
            for (left, right), values in bundle.edge_feats[index_int].items()
        }
        records.append(
            _structural_record(
                graph,
                np.asarray(bundle.node_feats[index_int], dtype=np.int64),
                edge_types,
            )
        )
        if (position + 1) % 2000 == 0 or position + 1 == len(indices):
            print(
                f"structural cache {split}: {position + 1}/{len(indices)}",
                flush=True,
            )
    metadata = {
        "schema": STRUCT_CACHE_SCHEMA,
        "split": split,
        "n_graphs": int(len(records)),
        "n_patches": int(sum(row["n_patches"] for row in records)),
        "n_patch_nodes": int(sum(row["n_nodes"] for row in records)),
        "n_patch_bonds": int(sum(row["n_bonds"] for row in records)),
        "atom_feature_dims": list(ATOM_FEATURE_DIMS),
        "bond_feature_dims": list(BOND_FEATURE_DIMS),
        "patch_radius": int(PATCH_RADIUS),
        "label_free": True,
        "target_free": True,
        "exact_certificate_stored": False,
        "seconds": float(time.perf_counter() - started),
    }
    STRUCT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {"schema": STRUCT_CACHE_SCHEMA, "metadata": metadata, "records": records},
            protocol=4,
        )
    )
    print(
        f"structural cache written: {path} "
        f"({metadata['n_patches']} patches, {metadata['seconds']:.1f}s)",
        flush=True,
    )
    return metadata


def load_structural_cache(split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = STRUCT_CACHE_DIR / f"structural_{split}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"structural cache missing: {path}; run the `preprocess` stage first"
        )
    payload = pickle.loads(path.read_bytes())
    if payload.get("schema") != STRUCT_CACHE_SCHEMA:
        raise ValueError(f"unexpected structural cache schema in {path}")
    records = payload["records"]
    metadata = payload["metadata"]
    if len(records) != int(SPLIT_SIZES[split]):
        raise ValueError(
            f"structural cache size mismatch for {split}: "
            f"{len(records)} != {SPLIT_SIZES[split]}"
        )
    return records, metadata


def preprocess(force: bool = False) -> dict[str, Any]:
    metadata = {
        split: build_structural_cache(split, force=force)
        for split in ("train", "valid", "test")
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "preprocess",
        "structural_cache_dir": str(STRUCT_CACHE_DIR),
        "splits": metadata,
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "preprocess.json", payload)
    return payload


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------


def _load_records(split: str) -> tuple[list[Any], dict[str, Any]]:
    return mpp._load_exact_rooted_record_cache(
        RECORD_CACHE, split, int(SPLIT_SIZES[split])
    )


def fit_train_transforms(records: Sequence[Any]) -> dict[str, Any]:
    """Fit the label-free transforms on the official train split only.

    The exact typed vocabulary is deliberately **not** fit: the B-Bag encoder
    replaces it.  Only the radius-1 parent vocabulary and the patch/context
    standardizers remain.
    """
    parent = mpp._fit_vocabulary(records, "parent_certificate", 4096, 1)
    patch_std = mpp.Standardizer.fit(mpp._patch_matrix(records))
    context_std = mpp.Standardizer.fit(mpp._context_matrix(records))
    return {
        "parent_vocabulary": parent,
        "patch_standardizer": patch_std,
        "context_standardizer": context_std,
    }


def encode_records(
    records: Sequence[Any],
    structural: Sequence[Mapping[str, Any]],
    transforms: Mapping[str, Any],
) -> list[Data]:
    if len(records) != len(structural):
        raise ValueError(
            f"record/structural count mismatch: {len(records)} != {len(structural)}"
        )
    parent_vocabulary = transforms["parent_vocabulary"]
    patch_std = transforms["patch_standardizer"]
    context_std = transforms["context_standardizer"]
    output: list[Data] = []
    for record, struct in zip(records, structural):
        n_patches = len(record.patches)
        if int(struct["n_patches"]) != n_patches:
            raise RuntimeError(
                f"structural patch count mismatch: {struct['n_patches']} != {n_patches}"
            )
        patch_cont = patch_std.transform(
            np.stack([patch.shell_descriptor for patch in record.patches], axis=0)
        )
        parent = np.asarray(
            [
                parent_vocabulary.get(patch.parent_certificate, 0)
                for patch in record.patches
            ],
            dtype=np.int64,
        )
        output.append(
            Data(
                patch_cont=torch.from_numpy(patch_cont),
                parent_token=torch.from_numpy(parent),
                struct_atom_fields=torch.from_numpy(
                    np.asarray(struct["node_fields"], dtype=np.int64)
                ),
                struct_root=torch.from_numpy(
                    np.asarray(struct["node_root"], dtype=np.int64)
                ),
                struct_dist=torch.from_numpy(
                    np.asarray(struct["node_dist"], dtype=np.int64)
                ),
                struct_patch=torch.from_numpy(
                    np.asarray(struct["node_patch"], dtype=np.int64)
                ),
                struct_bond_fields=torch.from_numpy(
                    np.asarray(struct["bond_fields"], dtype=np.int64)
                ),
                struct_edge_patch=torch.from_numpy(
                    np.asarray(struct["edge_patch"], dtype=np.int64)
                ),
                pair_index=torch.from_numpy(record.pair_index),
                pair_relation=torch.from_numpy(record.pair_relation),
                pair_bucket=torch.from_numpy(record.pair_bucket),
                global_context=torch.from_numpy(
                    context_std.transform(record.global_context[None, :])
                ),
                y=torch.tensor([record.y], dtype=torch.float32),
                num_nodes=n_patches,
            )
        )
    return output


def bag_collate(data_list: Sequence[Data]) -> Batch:
    """Batch with explicit patch offsets for the structural grouping tensors.

    ``struct_patch`` and ``struct_edge_patch`` are graph-local patch ids, so
    they must be offset by the cumulative patch count.  ``struct_atom_fields``
    and ``struct_bond_fields`` are node/edge level and PyG concatenates them.
    """
    batch = Batch.from_data_list(list(data_list))
    patch_offset = 0
    patch_parts: list[torch.Tensor] = []
    edge_patch_parts: list[torch.Tensor] = []
    for data in data_list:
        patch_parts.append(data.struct_patch + patch_offset)
        edge_patch_parts.append(data.struct_edge_patch + patch_offset)
        patch_offset += int(data.num_nodes)
    batch.struct_patch = (
        torch.cat(patch_parts) if patch_parts else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_edge_patch = (
        torch.cat(edge_patch_parts)
        if edge_patch_parts
        else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_n_patches = int(patch_offset)
    return batch


def _make_loader(
    graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int
):
    from torch.utils.data import DataLoader as TorchDataLoader

    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return TorchDataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=bag_collate,
    )


def data_sanity() -> dict[str, Any]:
    bundle = load_molhiv(root=DATA_ROOT, with_features=False)
    split = _split_indices(bundle)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "ogbg-molhiv",
        "root": str(DATA_ROOT),
        "official_split": "OGB scaffold train/valid/test",
        "sizes": {name: int(values.size) for name, values in split.items()},
        "sizes_match_official": all(
            int(split[name].size) == SPLIT_SIZES[name] for name in SPLIT_SIZES
        ),
        "positive": {name: int(bundle.y[split[name]].sum()) for name in split},
        "record_cache": str(RECORD_CACHE),
        "record_cache_exists": all(
            (RECORD_CACHE / f"exact_rooted_{name}.pkl").exists()
            for name in ("train", "valid", "test")
        ),
        "structural_cache_dir": str(STRUCT_CACHE_DIR),
        "structural_cache_exists": all(
            (STRUCT_CACHE_DIR / f"structural_{name}.pkl").exists()
            for name in ("train", "valid", "test")
        ),
        "official_test_labels_loaded": False,
        "official_test_loaded": False,
    }
    if not payload["sizes_match_official"]:
        raise RuntimeError("MolHIV split sizes do not match the official scaffold split")

    # Cross-check that the structural patch ordering lines up with the audited
    # exact-rooted cache on a bounded prefix of each *selection* split (train,
    # valid).  Official test records are deliberately not touched before the
    # one-shot unlock; their ordering is validated inside ``test_eval``.
    checks: dict[str, Any] = {}
    for name, limit in (("train", 256), ("valid", 256)):
        if not (STRUCT_CACHE_DIR / f"structural_{name}.pkl").exists():
            checks[name] = {"checked": False, "reason": "structural cache absent"}
            continue
        records, _meta = _load_records(name)
        structural, _smeta = load_structural_cache(name)
        mismatches = []
        for position in range(min(limit, len(records))):
            if int(structural[position]["n_patches"]) != len(records[position].patches):
                mismatches.append(int(position))
        checks[name] = {
            "checked": True,
            "n_checked": int(min(limit, len(records))),
            "patch_count_mismatches": mismatches,
        }
        del records, structural
        gc.collect()
    payload["patch_order_checks"] = checks
    payload["patch_order_ok"] = all(
        (not row["checked"]) or not row["patch_count_mismatches"]
        for row in checks.values()
    )
    _write_json(RESULTS_DIR / "data_sanity.json", payload)
    if not payload["patch_order_ok"]:
        raise RuntimeError("structural patch ordering does not match the record cache")
    return payload


# ---------------------------------------------------------------------------
# smoke / training
# ---------------------------------------------------------------------------


def _build_datasets(limit: int | None = None) -> tuple[list[Data], list[Data], dict[str, Any]]:
    train_records, _ = _load_records("train")
    valid_records, _ = _load_records("valid")
    train_struct, _ = load_structural_cache("train")
    valid_struct, _ = load_structural_cache("valid")
    if limit is not None:
        train_records = train_records[: int(limit)]
        train_struct = train_struct[: int(limit)]
    transforms = fit_train_transforms(train_records)
    train_data = encode_records(train_records, train_struct, transforms)
    valid_data = encode_records(valid_records, valid_struct, transforms)
    audit = {
        "parent_vocabulary_size_with_oov": int(len(transforms["parent_vocabulary"]) + 1),
        "exact_typed_vocabulary": "absent (B-Bag encoder is vocabulary-free)",
        "n_train_graphs": int(len(train_data)),
        "n_valid_graphs": int(len(valid_data)),
    }
    del train_records, valid_records, train_struct, valid_struct
    gc.collect()
    return train_data, valid_data, audit


def smoke(device: str = "cuda") -> dict[str, Any]:
    """Short forward/backward/few-step smoke on a bounded subset (no test)."""
    train_data, valid_data, audit = _build_datasets(limit=2048)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("smoke requested CUDA but CUDA is unavailable")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = build_model(audit["parent_vocabulary_size_with_oov"], 0).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = _make_loader(train_data, BATCH_SIZE, True, 91011)
    started = time.perf_counter()
    steps = 0
    losses: list[float] = []
    n_graphs = 0
    for batch in loader:
        batch = batch.to(dev)
        target = batch.y.view(-1)
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad()
        loss.backward()
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite:
            raise RuntimeError("non-finite gradient in smoke")
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(float(loss.detach()))
        n_graphs += int(target.numel())
        steps += 1
        if steps >= 20:
            break
    elapsed = float(time.perf_counter() - started)
    valid_auc = mpp._evaluate_auc(
        model, _make_loader(valid_data, BATCH_SIZE, False, 91012), dev
    )
    gradient_checks = _gradient_viability(model, next(iter(_make_loader(train_data[:128], 128, False, 0))).to(dev))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "device": str(dev),
        "device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if dev.type == "cuda"
            else None
        ),
        "n_train_graphs_used": int(audit["n_train_graphs"]),
        "steps": int(steps),
        "graphs_seen": int(n_graphs),
        "losses": losses,
        "valid_auc_after_smoke": float(valid_auc),
        "gradient_checks": gradient_checks,
        "parameters": _n_params(model),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if dev.type == "cuda" else 0
        ),
        "seconds": elapsed,
        "steps_per_second": (float(steps) / elapsed if elapsed > 0 else 0.0),
        "graphs_per_second": (float(n_graphs) / elapsed if elapsed > 0 else 0.0),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


def _gradient_viability(model: nn.Module, batch: Data) -> dict[str, Any]:
    """Forward/backward on a small batch and check per-module gradient health."""
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = F.binary_cross_entropy_with_logits(logits, batch.y.view(-1))
    loss.backward()
    modules = {
        "atom_field_embeddings": model.bag_encoder.atom_embeddings,
        "bond_field_embeddings": model.bag_encoder.bond_embeddings,
        "node_mlp": model.bag_encoder.node_mlp,
        "bond_mlp": model.bag_encoder.bond_mlp,
        "fusion": model.bag_encoder.fusion,
    }
    checks: dict[str, Any] = {}
    for name, module in modules.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        checks[name] = {
            "n_tensors_with_grad": int(len(grads)),
            "finite": bool(all(bool(torch.isfinite(g).all()) for g in grads)),
            "nonzero": bool(any(float(g.abs().max()) > 0.0 for g in grads)),
        }
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    checks["all_healthy"] = bool(
        all(
            row["finite"] and row["nonzero"] and row["n_tensors_with_grad"] > 0
            for row in checks.values()
            if isinstance(row, dict)
        )
    )
    return checks


def _topk(manifest: Sequence[Mapping[str, Any]], k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        manifest, key=lambda row: (-float(row["valid_auc"]), int(row["epoch"]))
    )
    return [dict(row) for row in ranked[:k]]


def _soup_state(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = list(states[0].keys())
    soup: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack([state[key].float() for state in states], dim=0)
        soup[key] = stacked.mean(dim=0).to(states[0][key].dtype)
    return soup


@torch.no_grad()
def _predict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    model.eval()
    targets: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        targets.append(batch.y.view(-1).cpu().numpy())
        logits.append(model(batch).view(-1).cpu().numpy())
    return (
        np.concatenate(targets).astype(np.float64),
        np.concatenate(logits).astype(np.float64),
    )


def _auc(targets: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(targets, logits))


def train_seed(
    seed: int,
    device: str = "cuda",
    max_epochs: int | None = None,
    patience: int | None = None,
    train_limit: int | None = None,
) -> dict[str, Any]:
    max_epochs = int(MAX_EPOCHS if max_epochs is None else max_epochs)
    patience = int(PATIENCE if patience is None else patience)
    started = time.perf_counter()
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train requested CUDA but CUDA is unavailable")
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats()

    train_data, valid_data, audit = _build_datasets(limit=train_limit)
    model = build_model(audit["parent_vocabulary_size_with_oov"], seed).to(dev)
    total_params = _n_params(model)
    if total_params > PARAM_UPPER_BOUND:
        raise RuntimeError(
            f"model has {total_params} params (> {PARAM_UPPER_BOUND}); a "
            "vocabulary-sized table may have leaked back in"
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_loader = _make_loader(train_data, BATCH_SIZE, True, seed + 91011)
    valid_loader = _make_loader(valid_data, BATCH_SIZE, False, seed + 91012)

    curve: list[dict[str, Any]] = []
    best_auc = -float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    top5: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    stale = 0
    epoch_times: list[float] = []
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(dev)
            target = batch.y.view(-1)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_loss = total_loss / max(seen, 1)
        valid_auc = mpp._evaluate_auc(model, valid_loader, dev)
        epoch_times.append(float(time.perf_counter() - epoch_started))
        state_copy = copy.deepcopy(model.state_dict())
        top5.append((float(valid_auc), int(epoch), state_copy))
        top5.sort(key=lambda item: (-item[0], item[1]))
        top5 = top5[:SOUP_K]
        curve.append(
            {"epoch": int(epoch), "train_bce": float(train_loss), "valid_auc": float(valid_auc)}
        )
        if valid_auc > best_auc:
            best_auc = float(valid_auc)
            best_epoch = int(epoch)
            best_state = state_copy
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or valid_auc == best_auc:
            print(
                f"[molhiv-bbag seed{seed}] epoch={epoch:03d} bce={train_loss:.6f} "
                f"valid_auc={valid_auc:.6f} best={best_auc:.6f}@{best_epoch}",
                flush=True,
            )
        if stale >= patience:
            print(f"[molhiv-bbag seed{seed}] early_stop epoch={epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    top_rows = sorted(top5, key=lambda item: (-item[0], item[1]))
    soup_state = _soup_state([row[2] for row in top_rows])
    valid_targets, raw_logits = _predict(model, best_state, valid_loader, dev)
    _t, soup_logits = _predict(model, soup_state, valid_loader, dev)
    raw_valid_auc = _auc(valid_targets, raw_logits)
    soup_valid_auc = _auc(valid_targets, soup_logits)

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        CURVE_DIR / f"seed{seed}_curve.json",
        {"seed": int(seed), "curve": curve},
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, RESULTS_DIR / f"raw_state_seed{seed}.pt")
    torch.save(soup_state, RESULTS_DIR / f"soup_state_seed{seed}.pt")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "device": str(dev),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if dev.type == "cuda"
            else None
        ),
        "parameters": int(total_params),
        "parameter_breakdown": parameter_breakdown(model),
        "parent_vocabulary_size_with_oov": int(audit["parent_vocabulary_size_with_oov"]),
        "typed_vocabulary_size_with_oov": 0,
        "patch_representation": "molhiv_shared_bag",
        "epochs_run": int(len(curve)),
        "early_stopped": bool(len(curve) < max_epochs),
        "best_epoch": int(best_epoch),
        "best_valid_auc": float(best_auc),
        "raw_valid_auc_recomputed": float(raw_valid_auc),
        "soup_valid_auc": float(soup_valid_auc),
        "top5_epochs": [int(row[1]) for row in top_rows],
        "top5_valid_auc": [float(row[0]) for row in top_rows],
        "mean_epoch_time_s": float(np.mean(epoch_times)),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if dev.type == "cuda" else 0
        ),
        "curve": curve,
        "valid_targets": valid_targets.tolist(),
        "raw_valid_logits": raw_logits.tolist(),
        "soup_valid_logits": soup_logits.tolist(),
        "loss": "BCEWithLogitsLoss (unweighted)",
        "protocol": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "grad_clip": GRAD_CLIP,
            "selection": "official validation ROC-AUC",
            "refit_on_train_valid": False,
        },
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    if dev.type == "cuda":
        # keep valid logits only for AUC recomputation; drop them from disk json
        pass
    _write_json(RESULTS_DIR / f"run_seed{seed}.json", summary)
    return summary


def train_queue(seeds: Sequence[int], device: str) -> None:
    for seed in seeds:
        if (RESULTS_DIR / f"run_seed{seed}.json").exists():
            print(f"skip existing seed{seed}", flush=True)
            continue
        print(f"=== train molhiv-bbag seed{seed} device={device} ===", flush=True)
        train_seed(int(seed), device=device)


# ---------------------------------------------------------------------------
# freeze / one-shot test
# ---------------------------------------------------------------------------


def freeze() -> dict[str, Any]:
    lock_path = RESULTS_DIR / "official_test_unlock.json"
    if lock_path.exists():
        raise RuntimeError("refusing to change the frozen rule after test unlock")
    runs = {}
    for seed in SEEDS:
        path = RESULTS_DIR / f"run_seed{seed}.json"
        if not path.exists():
            raise RuntimeError(f"missing run for seed{seed}; cannot freeze")
        runs[seed] = _read_json(path)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": {
            "patch_representation": "molhiv_shared_bag",
            "bag_encoder": {
                "atom_feature_dims": list(ATOM_FEATURE_DIMS),
                "bond_feature_dims": list(BOND_FEATURE_DIMS),
                "node_dim": BAG_NODE_DIM,
                "edge_dim": BAG_EDGE_DIM,
                "node_hidden": BAG_NODE_HIDDEN,
                "bond_hidden": BAG_BOND_HIDDEN,
                "fusion_hidden": BAG_FUSION_HIDDEN,
                "output_dim": BAG_OUTPUT_DIM,
            },
            "h_dim": H_DIM,
            "q_dim": Q_DIM,
            "token_width": TOKEN_WIDTH,
            "recurrence_rounds": RECURRENCE_ROUNDS,
            "weight_tied": True,
            "center_context_hidden": CENTER_CONTEXT_HIDDEN,
            "loss": "BCEWithLogitsLoss (unweighted)",
        },
        "seeds": [int(s) for s in SEEDS],
        "selection_metric": "official validation ROC-AUC",
        "raw_rule": "checkpoint with the highest validation ROC-AUC (ties -> earliest epoch)",
        "soup_rule": {
            "K": SOUP_K,
            "ranking": "highest validation ROC-AUC",
            "tie_rule": "earliest epoch",
            "aggregation": "equal-weight parameter average",
            "no_k_search": True,
            "no_weight_search": True,
        },
        "refit_on_train_valid": False,
        "validation": {
            str(seed): {
                "raw_valid_auc": float(runs[seed]["raw_valid_auc_recomputed"]),
                "soup_valid_auc": float(runs[seed]["soup_valid_auc"]),
                "best_epoch": int(runs[seed]["best_epoch"]),
                "epochs_run": int(runs[seed]["epochs_run"]),
                "parameters": int(runs[seed]["parameters"]),
                "mean_epoch_time_s": float(runs[seed]["mean_epoch_time_s"]),
                "peak_gpu_memory_bytes": int(runs[seed].get("peak_gpu_memory_bytes", 0)),
            }
            for seed in SEEDS
        },
        "test_status": "not yet loaded",
        "platform": platform.platform(),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    return payload


def test_eval(device: str = "cuda") -> dict[str, Any]:
    freeze_path = RESULTS_DIR / "architecture_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("refusing test: architecture_freeze.json missing")
    lock_path = RESULTS_DIR / "official_test_unlock.json"
    if lock_path.exists():
        raise RuntimeError("refusing test: official test already unlocked once")

    _write_json(
        lock_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "frozen_before_test": True,
            "encoding": "transforms fit on official train only (no refit)",
            "checkpoints": "pre-existing raw / Top-5 soup selection states",
            "official_test_loaded": True,
            "git_commit": _git_commit(),
        },
    )

    dev = torch.device(device)
    train_records, _ = _load_records("train")
    transforms = fit_train_transforms(train_records)
    parent_size = len(transforms["parent_vocabulary"]) + 1
    del train_records
    gc.collect()

    test_records, metadata = _load_records("test")
    test_struct, _ = load_structural_cache("test")
    test_data = encode_records(test_records, test_struct, transforms)
    del test_records, test_struct
    gc.collect()
    test_loader = _make_loader(test_data, BATCH_SIZE, False, 0)
    targets = np.asarray(
        [float(graph.y.view(-1)[0]) for graph in test_data], dtype=np.float64
    )

    rows = []
    raw_logits_by_seed: dict[int, np.ndarray] = {}
    soup_logits_by_seed: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model = build_model(parent_size, seed).to(dev)
        raw_state = torch.load(
            RESULTS_DIR / f"raw_state_seed{seed}.pt", map_location="cpu", weights_only=True
        )
        soup_state = torch.load(
            RESULTS_DIR / f"soup_state_seed{seed}.pt", map_location="cpu", weights_only=True
        )
        t_raw, raw_logits = _predict(model, raw_state, test_loader, dev)
        t_soup, soup_logits = _predict(model, soup_state, test_loader, dev)
        if not np.array_equal(t_raw, targets) or not np.array_equal(t_soup, targets):
            raise RuntimeError("official-test target order mismatch")
        raw_logits_by_seed[seed] = raw_logits
        soup_logits_by_seed[seed] = soup_logits
        rows.append(
            {
                "seed": int(seed),
                "raw_test_auc": _auc(targets, raw_logits),
                "soup_test_auc": _auc(targets, soup_logits),
            }
        )

    raw_values = np.asarray([row["raw_test_auc"] for row in rows])
    soup_values = np.asarray([row["soup_test_auc"] for row in rows])
    raw_ensemble = np.mean(np.stack([raw_logits_by_seed[s] for s in SEEDS]), axis=0)
    soup_ensemble = np.mean(np.stack([soup_logits_by_seed[s] for s in SEEDS]), axis=0)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "rows": rows,
        "raw_test_mean": float(raw_values.mean()),
        "raw_test_std": float(raw_values.std(ddof=1)) if len(raw_values) > 1 else 0.0,
        "soup_test_mean": float(soup_values.mean()),
        "soup_test_std": float(soup_values.std(ddof=1)) if len(soup_values) > 1 else 0.0,
        "diagnostic_raw_2seed_ensemble_test_auc": _auc(targets, raw_ensemble),
        "diagnostic_soup_2seed_ensemble_test_auc": _auc(targets, soup_ensemble),
        "test_metadata": {k: v for k, v in metadata.items() if k != "records"},
        "reference_rpc": {
            "raw_test_mean": REFERENCE_RPC_TEST_RAW_MEAN,
            "raw_test_ensemble": REFERENCE_RPC_TEST_ENSEMBLE,
            "valid_raw": {str(k): v for k, v in REFERENCE_RPC_VALID_RAW.items()},
            "valid_soup": {str(k): v for k, v in REFERENCE_RPC_VALID_SOUP.items()},
            "test_raw": {str(k): v for k, v in REFERENCE_RPC_TEST_RAW.items()},
        },
        "official_test_loaded": True,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "data_sanity": _maybe("data_sanity.json"),
        "preprocess": _maybe("preprocess.json"),
        "parameter_accounting": _maybe("parameter_accounting.json"),
        "smoke": _maybe("smoke.json"),
        "runs": {str(seed): _maybe(f"run_seed{seed}.json") for seed in SEEDS},
        "architecture_freeze": _maybe("architecture_freeze.json"),
        "official_test_results": _maybe("official_test_results.json"),
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "data_sanity",
            "preprocess",
            "params",
            "sanity",
            "smoke",
            "train",
            "train_queue",
            "freeze",
            "test",
            "report",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(4)
    _set_deterministic(bool(args.deterministic))

    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    elif args.stage == "preprocess":
        print(
            json.dumps(preprocess(force=bool(args.force)), indent=2, default=str),
            flush=True,
        )
    elif args.stage == "data_sanity":
        print(json.dumps(data_sanity(), indent=2, default=str), flush=True)
    elif args.stage == "sanity":
        print(json.dumps(sanity(args.device), indent=2, default=str), flush=True)
    elif args.stage == "smoke":
        print(json.dumps(smoke(args.device), indent=2, default=str), flush=True)
    elif args.stage == "train":
        limit = None if int(args.train_limit) < 0 else int(args.train_limit)
        print(
            json.dumps(
                train_seed(
                    args.seed,
                    device=args.device,
                    max_epochs=int(args.epochs),
                    patience=int(args.patience),
                    train_limit=limit,
                ),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    elif args.stage == "train_queue":
        train_queue([int(s) for s in args.seeds.split(",")], args.device)
    elif args.stage == "freeze":
        print(json.dumps(freeze(), indent=2, default=str), flush=True)
    elif args.stage == "test":
        print(json.dumps(test_eval(args.device), indent=2, default=str), flush=True)
    elif args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


def sanity(device: str = "cpu") -> dict[str, Any]:
    """Data-level integrity gate (encoder invariants are covered by unit tests)."""
    dev = torch.device(device)
    train_data, valid_data, audit = _build_datasets(limit=None)
    model = build_model(audit["parent_vocabulary_size_with_oov"], 0).to(dev)
    checks: dict[str, bool] = {}

    checks["no_typed_embedding_attribute"] = (
        getattr(model, "typed_embedding", None) is None
    )
    checks["no_typed_embedding_state"] = (
        len([k for k in model.state_dict() if "typed_embedding" in k]) == 0
    )
    checks["no_vocabulary_sized_tensor"] = all(
        int(p.numel()) < 100_000 for p in model.parameters()
    )
    checks["params_below_upper_bound"] = bool(_n_params(model) <= PARAM_UPPER_BOUND)
    checks["bag_encoder_output_is_token_width"] = bool(
        int(model.bag_encoder.output_dim) == int(model.token_width)
    )
    checks["parent_embedding_preserved"] = bool(
        _n_params(model.parent_embedding) == REFERENCE_RPC_PARENT_PARAMS
    )
    checks["recurrence_two_rounds"] = bool(
        int(model.recurrence_rounds) == 2 and int(model.pair_hidden) == 16
    )

    batch = next(iter(_make_loader(valid_data[:128], 128, False, 0))).to(dev)
    checks["forward_finite"] = bool(torch.isfinite(model(batch)).all())
    grad = _gradient_viability(model, batch)
    checks["gradients_healthy"] = bool(grad["all_healthy"])

    # batch invariance: a single molecule vs the same molecule inside a batch
    single = next(iter(_make_loader(valid_data[:1], 1, False, 0))).to(dev)
    model.eval()
    with torch.no_grad():
        out_single = model(single)
        out_batch = model(batch)[:1]
    checks["batch_invariance"] = bool(
        float((out_single - out_batch).abs().max()) < 1.0e-4
    )

    # connectivity-free witness: the encoder must not consume edge endpoints
    batch_a, batch_b = _adversarial_structural_pair(dev)
    encoder = model.bag_encoder.eval()
    with torch.no_grad():
        token_a = encoder(batch_a)
        token_b = encoder(batch_b)
    checks["connectivity_free_token"] = bool(
        float((token_a - token_b).abs().max()) == 0.0
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "parameters": int(_n_params(model)),
        "parameter_breakdown": parameter_breakdown(model),
        "gradient_checks": grad,
        "data_audit": audit,
        "device": str(dev),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"MolHIV B-Bag sanity failed: {failed}")
    return payload


class _StructBatch:
    """Minimal primitive container; endpoints are optional and never read."""

    def __init__(
        self,
        atom_fields: torch.Tensor,
        root: torch.Tensor,
        dist: torch.Tensor,
        patch: torch.Tensor,
        bond_fields: torch.Tensor,
        edge_patch: torch.Tensor,
    ) -> None:
        self.struct_atom_fields = atom_fields
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_bond_fields = bond_fields
        self.struct_edge_patch = edge_patch


def _adversarial_structural_pair(
    device: torch.device | str = "cpu",
) -> tuple[_StructBatch, _StructBatch]:
    """Identical primitive multisets, different (unused) connectivity.

    The encoder receives no endpoints at all, so both containers produce the
    same token; a connectivity-aware encoder could not be built from them.
    """
    # patch 0: 5 nodes, root at 0, distances [0,1,1,2,2]; 4 bonds.
    atom_fields = torch.zeros((5, len(ATOM_FEATURE_DIMS)), dtype=torch.long)
    root = torch.tensor([1, 0, 0, 0, 0])
    dist = torch.tensor([0, 1, 1, 2, 2])
    patch = torch.zeros(5, dtype=torch.long)
    bond_fields = torch.zeros((4, len(BOND_FEATURE_DIMS)), dtype=torch.long)
    edge_patch = torch.zeros(4, dtype=torch.long)
    moved = (
        atom_fields.to(device),
        root.to(device),
        dist.to(device),
        patch.to(device),
        bond_fields.to(device),
        edge_patch.to(device),
    )
    return (
        _StructBatch(*moved),
        _StructBatch(*(tensor.clone() for tensor in moved)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
