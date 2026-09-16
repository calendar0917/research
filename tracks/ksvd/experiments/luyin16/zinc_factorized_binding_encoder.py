"""ZINC Factorized Structure--Attribute Binding (FSAB) encoder.

Research question
-----------------
Inside each rooted local reference frame (the radius-2 all-centre patch), can a
strong molecular model explicitly factorize the local information into

    S_v : pure topology / structural role,
    A_v : attribute marginals only,
    B_v : centered structure--attribute correspondence / binding,

and then feed ``[S; A; B]`` into the *existing* relational reasoning **without
sacrificing the predictive performance of the current B-Full / recurrent
model**?

The radius-2 patch is used only as a *rooted local reference frame*.  It is not
a motif, not a learned object and not a discovered structure.  There is no
learned motif vocabulary, no adaptive subgraph, no support selection, no
structure composition and no structure discovery.

Separation is enforced by **input access**, not by hoping three MLPs learn the
right semantics:

    S  reads root flag + root distance + untyped adjacency.  Never atom type,
       never bond type, never a typed certificate.
    A  reads atom / bond categories grouped by patch membership.  Never the
       root flag, never the root distance, never a degree, never a structural
       role, never ``struct_src`` / ``struct_dst`` and never adjacency
       propagation.  In particular it never reads the root atom separately.
    B  is the only channel that sees a structural role and the attribute of the
       *same* node / edge, and it centers both sides inside the patch before a
       low-rank product (``P(S,A) - P(S)P(A)`` in spirit).

Everything downstream is inherited bit-exactly from the frozen cell-A /
B-Full geometry: radius-2 patch extraction, the relation system, the pair
descriptors, the ``T=2`` weight-tied recurrent pair--centre, the parent
embedding, the global / topology channels, the graph head, the optimizer, LR,
weight decay, batch size, gradient clipping and the Top-5 soup.

Only the ZINC regression task loss (L1 / MAE) is optimized.  There is no
distillation, no teacher, no contrastive / shuffle / binding auxiliary loss, no
role or atom prediction, no reconstruction, no orthogonality / HSIC / MI loss,
no rank or sparsity loss.  Attribute shuffle is an **evaluation-only**
diagnostic; it is never differentiated through.

Frozen references (never retrained here; official valid, 2-seed Top-5 soup):

    B-Bag  0.12380552224389975   (84,511 params; encoder 35,168)
    B-Full 0.11897220489243046   (84,495 params; encoder 35,152)
    A2     0.12191404939390486

Official ZINC test is **never** loaded.

Pre-registered seed0 performance guard: FSAB seed0 Top-5 soup valid must not be
worse than the matched B-Bag seed0 soup by more than 0.002 MAE.

Stages: ``params preprocess sanity smoke train soup train_queue repro
diagnostics witness interventions decide report``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_factorized_binding_encoder <stage>
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)
from tracks.ksvd.experiments.luyin16.structural_patch_encoder import (
    FactorizedStructureAttributeBindingEncoder,
)
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
    base_config as _v4_base_config,
)

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/factorized_binding_encoder"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "factorized_binding_encoder_v1"

# --- frozen cell-A geometry (unchanged) --------------------------------------
CELL = "A"
H_DIM = 64
Q_DIM = 16
T_ROUNDS = 2
PATCH_ENCODER_HIDDEN = 64
GLOBAL_ENCODER_HIDDEN = 32
TYPED_VOCABULARY_SIZE = 6785
PARENT_VOCABULARY_SIZE = 32
TOKEN_WIDTH = 16

# --- pre-registered FSAB geometry (one architecture; no sweep) ---------------
FSAB_ROLE_DIM = 32
FSAB_S_DIM = 16
FSAB_A_DIM = 16
FSAB_B_DIM = 16
FSAB_ATTR_DIM = 16
FSAB_S_HIDDEN = 32
FSAB_A_HIDDEN = 32
FSAB_B_HIDDEN = 32
FSAB_FUSION_HIDDEN = 32
FSAB_ROUNDS = 2
FSAB_ACTIVATION = "silu"

# --- references (frozen; never retrained in this study) -----------------------
REFERENCE_BBAG_SOUP_VALID = 0.12380552224389975
REFERENCE_BBAG_SOUP_PER_SEED = {
    0: 0.12738177864899625,
    1: 0.12022926583880325,
}
REFERENCE_BBAG_PARAMS = 84511
REFERENCE_BBAG_ENCODER_PARAMS = 35168

REFERENCE_BFULL_SOUP_VALID = 0.11897220489243046
REFERENCE_BFULL_SOUP_PER_SEED = {
    0: 0.11981802638241788,
    1: 0.11812638340244302,
}
REFERENCE_BFULL_PARAMS = 84495
REFERENCE_BFULL_ENCODER_PARAMS = 35152

REFERENCE_A2_SOUP_VALID = 0.12191404939390486
REFERENCE_A2_SOUP_PER_SEED = {
    0: 0.12169362585240742,
    1: 0.1221344729354023,
}

PARAM_UPPER = 120000
STRONG_GATE = 0.002
REGRESSION_GATE = 0.003
# Prompt section 26: seed0 Top-5 soup must not be more than 0.002 worse than the
# matched (same-seed) B-Bag reference.
SEED0_GUARD = REFERENCE_BBAG_SOUP_PER_SEED[0] + 0.002
# Mechanism viability: the true-vs-shuffle binding gap must exceed numerical
# noise aggregate.
BINDING_GAP_FLOOR = 1.0e-4
PURITY_TOLERANCE = 1.0e-5

SEEDS = (0, 1)
DIAG_BATCHES = 128
WITNESS_BATCHES = 64
SHUFFLE_REPEATS = 3

DETERMINISTIC = False

_write_json = sspe._write_json
_read_json = sspe._read_json
_git_commit = sspe._git_commit
_environment_fingerprint = sspe._environment_fingerprint


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# --- static audit vocabulary (brief section 19 I) ---------------------------
_LOSS_CALL_NAMES = {
    "l1_loss",
    "mse_loss",
    "smooth_l1_loss",
    "huber_loss",
    "nll_loss",
    "cross_entropy",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "kl_div",
    "triplet_margin_loss",
    "cosine_embedding_loss",
    "hsic",
}
_ATTENTION_CALL_NAMES = {
    "softmax",
    "log_softmax",
    "scaled_dot_product_attention",
    "multi_head_attention_forward",
    "MultiheadAttention",
    "MultiheadAttentionForward",
}
_DISTILLATION_CALL_NAMES = {
    "distillation_loss",
    "kd_loss",
    "teacher_logits",
    "teacher_forward",
}


def _called_names(source: str) -> set[str]:
    """Return every called function name appearing in ``source`` (AST-based)."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _loss_call_names(source: str) -> set[str]:
    return _called_names(source) & _LOSS_CALL_NAMES


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_candidate(seed: int = 0) -> torch.nn.Module:
    """Cell-A geometry with the local token produced by the FSAB encoder.

    The canonical frozen baseline is instantiated first and every shared tensor
    whose shape is unchanged is copied bit-exactly (identical construction order
    to ``sspe.build_candidate`` / ``zinc_shared_bag_patch_encoder``).  The only
    difference is ``patch_representation="factorized_binding"``: there is no
    ``typed_embedding`` and the patch token is the FSAB ``e_struct``.
    """
    baseline = shead.build_baseline(int(seed))
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }
    shead._seed_everything(int(seed))
    kwargs = shead._base_kwargs()
    kwargs["patch_hidden"] = H_DIM
    kwargs["patch_encoder_hidden"] = PATCH_ENCODER_HIDDEN
    kwargs["global_encoder_hidden"] = GLOBAL_ENCODER_HIDDEN
    kwargs.pop("pair_hidden", None)
    with cap._construction_guards(H_DIM, Q_DIM):
        model = rec.PatchPathRecurrentPairCentreModel(
            TYPED_VOCABULARY_SIZE,
            PARENT_VOCABULARY_SIZE,
            pair_hidden=Q_DIM,
            recurrence_rounds=rec.RECURRENCE_ROUNDS,
            recurrence_enabled=True,
            recurrence_mode="refresh",
            patch_representation="factorized_binding",
            fsab_role_dim=FSAB_ROLE_DIM,
            fsab_s_dim=FSAB_S_DIM,
            fsab_a_dim=FSAB_A_DIM,
            fsab_b_dim=FSAB_B_DIM,
            fsab_attr_dim=FSAB_ATTR_DIM,
            fsab_s_hidden=FSAB_S_HIDDEN,
            fsab_a_hidden=FSAB_A_HIDDEN,
            fsab_b_hidden=FSAB_B_HIDDEN,
            fsab_fusion_hidden=FSAB_FUSION_HIDDEN,
            fsab_rounds=FSAB_ROUNDS,
            fsab_activation=FSAB_ACTIVATION,
            **kwargs,
        )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in baseline_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
    torch.manual_seed(int(shead.SMALL_HEAD_SEED))
    model.head = shead.GenericReader(
        int(model.unified_graph_width), shead.SMALL_HEAD_HIDDEN
    )
    return model


def _fsab_breakdown(encoder: FactorizedStructureAttributeBindingEncoder) -> dict[str, int]:
    return {
        "structure_stream": int(
            _n_params(encoder.root_embedding)
            + _n_params(encoder.distance_embedding)
            + encoder.topology_base.numel()
            + _n_params(encoder.message)
            + _n_params(encoder.update)
            + _n_params(encoder.structure_pool)
        ),
        "attribute_stream": int(
            _n_params(encoder.atom_embedding)
            + _n_params(encoder.bond_embedding)
            + _n_params(encoder.atom_mlp)
            + _n_params(encoder.bond_mlp)
            + _n_params(encoder.attribute_fuse)
        ),
        "binding_stream": int(
            _n_params(encoder.node_role_projection)
            + _n_params(encoder.node_attribute_projection)
            + _n_params(encoder.edge_role_mlp)
            + _n_params(encoder.edge_attribute_mlp)
            + _n_params(encoder.edge_role_projection)
            + _n_params(encoder.edge_attribute_projection)
            + _n_params(encoder.binding_fuse)
        ),
        "fusion": int(
            _n_params(encoder.structure_weight)
            + _n_params(encoder.attribute_weight)
            + _n_params(encoder.binding_weight)
            + encoder.fusion_bias.numel()
            + _n_params(encoder.fusion_mlp)
        ),
    }


# ---------------------------------------------------------------------------
# structural tensors / batching
# ---------------------------------------------------------------------------


def _attach_struct_tensors_fsab(encoded: Sequence[Data], graphs: Sequence[Any]) -> None:
    """Attach FSAB primitives plus the explicit bond -> patch grouping."""
    for data, graph in zip(encoded, graphs):
        if int(data.num_nodes) != graph.n_patches:
            raise RuntimeError("encoded / patch-graph patch count mismatch")
        node_counts = np.bincount(graph.patch, minlength=graph.n_patches).astype(
            np.int64
        )
        node_start = np.concatenate([[0], np.cumsum(node_counts)[:-1]]).astype(
            np.int64
        )
        data.struct_atom = torch.from_numpy(graph.atom)
        data.struct_root = torch.from_numpy(graph.root)
        data.struct_dist = torch.from_numpy(graph.dist)
        data.struct_patch = torch.from_numpy(graph.patch)
        data.struct_src = torch.from_numpy(graph.src + node_start[graph.edge_patch])
        data.struct_dst = torch.from_numpy(graph.dst + node_start[graph.edge_patch])
        data.struct_bond = torch.from_numpy(graph.bond)
        data.struct_edge_patch = torch.from_numpy(np.asarray(graph.edge_patch))


def build_encoded_records():
    """Return (train_data, valid_data, audit) with FSAB structural tensors."""
    train_bundle, valid_bundle, _meta = sspe.extract_records()
    config = _v4_base_config()
    config["model"]["device"] = "cpu"
    enc_train, enc_valid, audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    _attach_struct_tensors_fsab(enc_train, train_bundle["graphs"])
    _attach_struct_tensors_fsab(enc_valid, valid_bundle["graphs"])
    audit = dict(audit)
    audit["patch_representation"] = "factorized_binding"
    audit["fsab_encoder"] = {
        "role_dim": FSAB_ROLE_DIM,
        "s_dim": FSAB_S_DIM,
        "a_dim": FSAB_A_DIM,
        "b_dim": FSAB_B_DIM,
        "attr_dim": FSAB_ATTR_DIM,
        "rounds": FSAB_ROUNDS,
        "output_dim": TOKEN_WIDTH,
        "activation": FSAB_ACTIVATION,
    }
    audit["official_test_loaded"] = False
    return enc_train, enc_valid, audit


def fsab_collate(data_list: Sequence[Data]) -> Any:
    """Batch with explicit offsets for the FSAB structural tensors."""
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(list(data_list))
    node_offset = 0
    patch_offset = 0
    patch_parts: list[torch.Tensor] = []
    edge_patch_parts: list[torch.Tensor] = []
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    for data in data_list:
        n_patches = int(data.num_nodes)
        patch_parts.append(data.struct_patch + patch_offset)
        edge_patch_parts.append(data.struct_edge_patch + patch_offset)
        src_parts.append(data.struct_src + node_offset)
        dst_parts.append(data.struct_dst + node_offset)
        node_offset += int(data.struct_atom.numel())
        patch_offset += n_patches
    batch.struct_patch = (
        torch.cat(patch_parts) if patch_parts else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_edge_patch = (
        torch.cat(edge_patch_parts)
        if edge_patch_parts
        else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_src = (
        torch.cat(src_parts) if src_parts else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_dst = (
        torch.cat(dst_parts) if dst_parts else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_n_patches = int(patch_offset)
    return batch


def _make_fsab_loader(
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
        collate_fn=fsab_collate,
    )


def _evaluate_mae(model: nn.Module, loader, device: torch.device) -> float:
    return sspe._evaluate_mae(model, loader, device)


def _selection_loader(valid_data: Sequence[Data]):
    return _make_fsab_loader(valid_data, 128, False, 0)


# ---------------------------------------------------------------------------
# attribute-assignment shuffle (evaluation-only diagnostic)
# ---------------------------------------------------------------------------


def _within_group_rotation(
    group: np.ndarray, seed: int
) -> np.ndarray:
    """Return, for each row, the source row of a per-group cyclic rotation.

    The rotation is a permutation inside every group, so every group keeps its
    exact value multiset while the assignment to positions changes.
    """
    group = np.asarray(group, dtype=np.int64)
    n = int(group.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((np.arange(n), group))
    ordered_group = group[order]
    # group boundaries in the sorted order
    change = np.ones(n, dtype=bool)
    change[1:] = ordered_group[1:] != ordered_group[:-1]
    group_start_ordered = np.where(change, np.arange(n), 0)
    np.maximum.accumulate(group_start_ordered, out=group_start_ordered)
    sizes_ordered = np.zeros(n, dtype=np.int64)
    # size of each sorted run
    run_id_ordered = np.cumsum(change) - 1
    counts = np.bincount(run_id_ordered)
    sizes_ordered = counts[run_id_ordered]
    # deterministic per-run shift in [0, size)
    run_sizes = np.bincount(run_id_ordered)
    rng = np.random.default_rng(int(seed))
    run_shift = rng.integers(0, np.maximum(run_sizes, 1))
    shift_ordered = run_shift[run_id_ordered]
    local = np.arange(n) - group_start_ordered
    source_ordered = group_start_ordered + ((local + shift_ordered) % sizes_ordered)
    source = np.empty(n, dtype=np.int64)
    source[order] = order[source_ordered]
    return source


def shuffle_attribute_assignment(
    batch: Any, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (shuffled_atom, shuffled_bond) preserving per-patch multisets.

    Topology is untouched; only the attribute -> structural-role assignment
    changes inside each patch.
    """
    patch = batch.struct_patch.detach().cpu().numpy().astype(np.int64)
    atom = batch.struct_atom.detach().cpu()
    atom_source = _within_group_rotation(patch, seed)
    shuffled_atom = atom[torch.from_numpy(atom_source)]

    bond = batch.struct_bond.detach().cpu()
    edge_patch = batch.struct_edge_patch.detach().cpu().numpy().astype(np.int64)
    bond_source = _within_group_rotation(edge_patch, seed + 977)
    shuffled_bond = (
        bond[torch.from_numpy(bond_source)] if bond.numel() else bond.clone()
    )
    return shuffled_atom, shuffled_bond


class _FSABBatch:
    """Plain container for encoder-level audits (no PyG machinery)."""

    def __init__(self, atom, root, dist, patch, edge_patch, bond, src, dst) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_edge_patch = edge_patch
        self.struct_bond = bond
        self.struct_src = src
        self.struct_dst = dst


def _relabel(batch: _FSABBatch, perm: torch.Tensor) -> _FSABBatch:
    """Relabel patch nodes by ``perm`` (edges follow the node labels)."""
    n = int(batch.struct_atom.numel())
    inverse = torch.empty(n, dtype=torch.long)
    inverse[perm] = torch.arange(n)
    return _FSABBatch(
        atom=batch.struct_atom[perm],
        root=batch.struct_root[perm],
        dist=batch.struct_dist[perm],
        patch=batch.struct_patch[perm],
        edge_patch=batch.struct_edge_patch.clone(),
        bond=batch.struct_bond.clone(),
        src=inverse[batch.struct_src],
        dst=inverse[batch.struct_dst],
    )


def _reorder_directed_edges(batch: _FSABBatch, perm: torch.Tensor) -> _FSABBatch:
    """Permute the directed-edge list (and flip every direction)."""
    src = batch.struct_dst[perm]
    dst = batch.struct_src[perm]
    return _FSABBatch(
        atom=batch.struct_atom.clone(),
        root=batch.struct_root.clone(),
        dist=batch.struct_dist.clone(),
        patch=batch.struct_patch.clone(),
        edge_patch=batch.struct_edge_patch[perm],
        bond=batch.struct_bond[perm],
        src=src,
        dst=dst,
    )


def _synthetic_patch(
    atom: Sequence[int],
    root_index: int,
    dist: Sequence[int],
    edges: Sequence[tuple[int, int]],
    bond_types: Sequence[int],
) -> _FSABBatch:
    n = len(atom)
    root = [0] * n
    root[root_index] = 1
    both = list(edges) + [(v, u) for u, v in edges]
    src = torch.tensor([u for u, _ in both], dtype=torch.long)
    dst = torch.tensor([v for _, v in both], dtype=torch.long)
    # every undirected bond keeps the same type in both directions
    edge_bond = list(bond_types) + list(bond_types)
    return _FSABBatch(
        atom=torch.tensor(list(atom), dtype=torch.long),
        root=torch.tensor(root, dtype=torch.long),
        dist=torch.tensor(list(dist), dtype=torch.long),
        patch=torch.zeros(n, dtype=torch.long),
        edge_patch=torch.zeros(len(both), dtype=torch.long),
        bond=torch.tensor(edge_bond, dtype=torch.long),
        src=src,
        dst=dst,
    )


def _binding_witness_pair() -> tuple[_FSABBatch, _FSABBatch, _FSABBatch]:
    """Same topology / root / marginals, different attribute assignment.

    Returns ``(reference, swapped_node_attributes, swapped_bond_attributes)``.
    """
    reference = _synthetic_patch(
        atom=[0, 1, 2],
        root_index=0,
        dist=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        bond_types=[0, 1],
    )
    node_swap = _synthetic_patch(
        atom=[0, 2, 1],
        root_index=0,
        dist=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        bond_types=[0, 1],
    )
    bond_swap = _synthetic_patch(
        atom=[0, 1, 2],
        root_index=0,
        dist=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        bond_types=[1, 0],
    )
    return reference, node_swap, bond_swap


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    baseline = cd.build_cell(CELL, 0)
    bfull = sspe.build_candidate(0)
    candidate = build_candidate(0)
    base_total = _n_params(baseline)
    cand_total = _n_params(candidate)
    if base_total != sspe.REFERENCE_PARAMS:
        raise RuntimeError(
            f"cell-A baseline params changed: {base_total} != {sspe.REFERENCE_PARAMS}"
        )
    audit = zpp.audit_parameters(candidate)
    encoder = candidate.structural_encoder
    breakdown = _fsab_breakdown(encoder)
    encoder_total = _n_params(encoder)
    typed_keys = [key for key in candidate.state_dict() if "typed_embedding" in key]
    # Repo convention: dataset-dependent identity/vocabulary storage means the
    # vocab-sized learned lookups.  Both are absent here (the FSAB encoder is
    # vocabulary-free).  ``parent_embedding`` is the 32-row table inherited
    # bit-exactly from the frozen cell-A / B-Full baseline and is reported
    # separately for transparency.
    dataset_dependent = sum(
        parameter.numel()
        for name, parameter in candidate.named_parameters()
        if "typed_embedding" in name or "structural_context_embedding" in name
    )
    inherited_parent = sum(
        parameter.numel()
        for name, parameter in candidate.named_parameters()
        if name.startswith("parent_embedding")
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cell": CELL,
        "baseline_params": int(base_total),
        "bfull_params": int(_n_params(bfull)),
        "bbag_params": int(REFERENCE_BBAG_PARAMS),
        "candidate_params": int(cand_total),
        "candidate_minus_bfull": int(cand_total - _n_params(bfull)),
        "params_in_range": bool(cand_total <= PARAM_UPPER),
        "fsab_encoder_params": int(encoder_total),
        "bfull_encoder_params": int(_n_params(bfull.structural_encoder)),
        "bbag_encoder_params": int(REFERENCE_BBAG_ENCODER_PARAMS),
        "fsab_stream_breakdown": breakdown,
        "downstream_params": int(cand_total - encoder_total),
        "candidate_has_typed_embedding": bool(
            getattr(candidate, "typed_embedding", None) is not None
        ),
        "candidate_typed_state_keys": typed_keys,
        "dataset_dependent_vocabulary_params": int(dataset_dependent),
        "inherited_parent_embedding_params": int(inherited_parent),
        "candidate_block_params": audit["blocks"],
        "candidate_dimensions": audit["dimensions"],
        "q_dim": int(candidate.pair_hidden),
        "h_dim": int(candidate.patch_hidden),
        "recurrence_rounds": int(candidate.recurrence_rounds),
        "recurrence_mode": str(candidate.recurrence_mode),
        "unified_graph_width": int(candidate.unified_graph_width),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# preprocessing / sanity
# ---------------------------------------------------------------------------


def preprocess(force: bool = False) -> dict[str, Any]:
    """Materialise (and cache) the shared patch split + FSAB tensors."""
    train_data, valid_data, audit = build_encoded_records()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_train": int(len(train_data)),
        "n_valid": int(len(valid_data)),
        "patch_radius": PATCH_RADIUS,
        "audit": audit,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "preprocess.json", payload)
    return payload


def sanity() -> dict[str, Any]:
    """Correctness / purity / invariance audit (brief section 19)."""
    torch.set_num_threads(4)
    device = torch.device("cpu")
    train_data, valid_data, _audit = build_encoded_records()
    loader = _make_fsab_loader(list(valid_data)[:96], 96, False, 0)
    batch = next(iter(loader)).to(device)

    model = build_candidate(0).to(device)
    encoder = model.structural_encoder
    encoder.eval()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # --- architecture / bookkeeping --------------------------------------
    checks["no_vocab_sized_typed_embedding"] = not any(
        "typed_embedding" in key for key in model.state_dict()
    )
    checks["fsab_encoder_present"] = isinstance(
        encoder, FactorizedStructureAttributeBindingEncoder
    )
    checks["q16_h64_T2"] = (
        int(model.pair_hidden) == 16
        and int(model.patch_hidden) == 64
        and int(model.recurrence_rounds) == 2
    )
    counts = model.module_call_counts(batch)
    checks["recurrent_weight_tying"] = counts == {
        "pair_projection": 4,
        "pair_encoder": 2,
        "center_update": 2,
    }
    checks["output_width_16"] = int(encoder.output_dim) == 16
    total_params = _n_params(model)
    checks["params_in_range"] = bool(total_params <= PARAM_UPPER)
    details["candidate_params"] = int(total_params)
    details["fsab_encoder_params"] = int(_n_params(encoder))

    # --- gradient viability (H) ------------------------------------------
    model.train()
    out = model(batch)
    loss = F.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    checks["forward_finite"] = bool(torch.isfinite(out).all())
    grad_norms = _fsab_grad_norms(model)
    details["init_grad_norms"] = grad_norms
    checks["backward_finite_nonzero"] = bool(
        all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
    ) and any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in model.parameters()
    )
    for name, key in (
        ("S", "structure"),
        ("A", "attribute"),
        ("B_node", "binding_node"),
        ("B_edge", "binding_edge"),
        ("fusion", "fusion"),
    ):
        checks[f"grad_{key}_nonzero"] = bool(grad_norms[f"{key}_grad_norm"] > 0.0)
    model.zero_grad(set_to_none=True)
    encoder.eval()

    # --- channel purity + witness (A/B/C/D/F) ----------------------------
    channels_ref = {
        key: value.detach().clone()
        for key, value in encoder.forward_channels(batch).items()
        if key in {"structure", "attributes", "binding", "output"}
    }
    atom = batch.struct_atom.clone()
    root = batch.struct_root.clone()
    dist = batch.struct_dist.clone()
    patch = batch.struct_patch.clone()
    edge_patch = batch.struct_edge_patch.clone()
    bond = batch.struct_bond.clone()
    src = batch.struct_src.clone()
    dst = batch.struct_dst.clone()

    # (A) node relabel invariance
    n_nodes = int(atom.shape[0])
    perm = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(20260930))
    inverse = torch.empty(n_nodes, dtype=torch.long)
    inverse[perm] = torch.arange(n_nodes)
    with torch.no_grad():
        channels_perm = encoder.forward_channels(
            _FSABBatch(
                atom[perm], root[perm], dist[perm], patch[perm], edge_patch, bond,
                inverse[src], inverse[dst],
            )
        )
    relabel_delta = {
        key: float((channels_ref[key] - channels_perm[key]).abs().max())
        for key in channels_ref
    }
    details["relabel_delta"] = relabel_delta
    checks["node_relabel_invariant"] = all(
        value < 1.0e-5 for value in relabel_delta.values()
    )

    # (G) batch invariance: single graph vs batched graph
    single_loader = _make_fsab_loader(list(valid_data)[:96], 1, False, 0)
    single_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for sub in single_loader:
            single_rows.append(encoder(sub).detach().view(-1))
    single = torch.cat(single_rows)
    batched = encoder(batch).detach().view(-1)
    details["batch_invariance_max_abs"] = float((single - batched).abs().max())
    checks["batch_invariant"] = bool(
        float((single - batched).abs().max()) < 1.0e-4
    )

    # (E) attribute permutation null: shuffle preserves S and A exactly
    shuffled_atom, shuffled_bond = shuffle_attribute_assignment(batch, 20260930)
    with torch.no_grad():
        channels_shuffle = encoder.forward_channels(
            _FSABBatch(
                shuffled_atom, root, dist, patch, edge_patch, shuffled_bond, src, dst
            )
        )
    details["shuffle_delta_S"] = float(
        (channels_ref["structure"] - channels_shuffle["structure"]).abs().max()
    )
    details["shuffle_delta_A"] = float(
        (channels_ref["attributes"] - channels_shuffle["attributes"]).abs().max()
    )
    details["shuffle_delta_B"] = float(
        (channels_ref["binding"] - channels_shuffle["binding"]).abs().max()
    )
    checks["shuffle_S_exactly_unchanged"] = bool(details["shuffle_delta_S"] == 0.0)
    checks["shuffle_A_unchanged_within_float_noise"] = bool(
        details["shuffle_delta_A"] <= PURITY_TOLERANCE
    )
    checks["shuffle_B_changes"] = bool(details["shuffle_delta_B"] > 1.0e-6)

    # (D) binding witness: same topology / marginals, different assignment
    witness_ref, witness_node, witness_bond = _binding_witness_pair()
    with torch.no_grad():
        ch_w_ref = encoder.forward_channels(witness_ref)
        ch_w_node = encoder.forward_channels(witness_node)
        ch_w_bond = encoder.forward_channels(witness_bond)
    witness = {
        "node_swap_S": float(
            (ch_w_ref["structure"] - ch_w_node["structure"]).abs().max()
        ),
        "node_swap_A": float(
            (ch_w_ref["attributes"] - ch_w_node["attributes"]).abs().max()
        ),
        "node_swap_B": float(
            (ch_w_ref["binding"] - ch_w_node["binding"]).abs().max()
        ),
        "bond_swap_S": float(
            (ch_w_ref["structure"] - ch_w_bond["structure"]).abs().max()
        ),
        "bond_swap_A": float(
            (ch_w_ref["attributes"] - ch_w_bond["attributes"]).abs().max()
        ),
        "bond_swap_B": float(
            (ch_w_ref["binding"] - ch_w_bond["binding"]).abs().max()
        ),
    }
    details["witness"] = witness
    checks["witness_S_unchanged"] = bool(
        witness["node_swap_S"] == 0.0 and witness["bond_swap_S"] == 0.0
    )
    checks["witness_A_unchanged_within_float_noise"] = bool(
        witness["node_swap_A"] <= PURITY_TOLERANCE
        and witness["bond_swap_A"] <= PURITY_TOLERANCE
    )
    checks["witness_B_changes"] = bool(
        witness["node_swap_B"] > 1.0e-6 and witness["bond_swap_B"] > 1.0e-6
    )

    # (C) attribute marginal purity: S must not react to any attribute change
    atom_all = torch.full_like(atom, 1)
    bond_all = torch.full_like(bond, 2)
    with torch.no_grad():
        channels_attr = encoder.forward_channels(
            _FSABBatch(atom_all, root, dist, patch, edge_patch, bond_all, src, dst)
        )
    details["attribute_change_delta_S"] = float(
        (channels_ref["structure"] - channels_attr["structure"]).abs().max()
    )
    checks["S_ignores_attributes"] = bool(details["attribute_change_delta_S"] == 0.0)
    checks["A_reacts_to_attributes"] = bool(
        float(
            (channels_ref["attributes"] - channels_attr["attributes"]).abs().max()
        )
        > 1.0e-6
    )

    # (B) structure purity: A must not react to any topology change
    dist_new = dist.clone()
    dist_new[0] = (int(dist_new[0].item()) + 1) % 3
    with torch.no_grad():
        channels_topo = encoder.forward_channels(
            _FSABBatch(atom, root, dist_new, patch, edge_patch, bond, src, dst)
        )
    details["topology_change_delta_A"] = float(
        (channels_ref["attributes"] - channels_topo["attributes"]).abs().max()
    )
    checks["A_ignores_topology"] = bool(details["topology_change_delta_A"] == 0.0)

    # (F) edge-binding correctness: bond-type shuffle keeps A, changes B edge
    _shuffled_atom_unused, shuffled_bond_only = shuffle_attribute_assignment(
        batch, 20260930
    )
    with torch.no_grad():
        ch_edge = encoder.forward_channels(
            _FSABBatch(
                atom, root, dist, patch, edge_patch, shuffled_bond_only, src, dst
            )
        )
    details["edge_shuffle_delta_A"] = float(
        (channels_ref["attributes"] - ch_edge["attributes"]).abs().max()
    )
    details["edge_shuffle_delta_B"] = float(
        (channels_ref["binding"] - ch_edge["binding"]).abs().max()
    )
    checks["edge_shuffle_A_unchanged"] = bool(
        details["edge_shuffle_delta_A"] <= PURITY_TOLERANCE
    )
    checks["edge_shuffle_B_changes"] = bool(details["edge_shuffle_delta_B"] > 1.0e-6)

    # (I) static audit: no attention / prohibited losses in the encoder path
    import inspect as _inspect
    import textwrap as _textwrap

    encoder_source = _textwrap.dedent(
        _inspect.getsource(FactorizedStructureAttributeBindingEncoder)
    )
    runner_source = Path(__file__).read_text(encoding="utf-8")
    encoder_loss_calls = _loss_call_names(encoder_source)
    runner_calls = _called_names(runner_source)
    runner_loss_calls = runner_calls & _LOSS_CALL_NAMES
    details["encoder_loss_calls"] = sorted(encoder_loss_calls)
    details["runner_loss_calls"] = sorted(runner_loss_calls)
    checks["fsab_has_no_loss_calls"] = bool(len(encoder_loss_calls) == 0)
    checks["no_attention_calls"] = bool(
        not (_called_names(encoder_source) & _ATTENTION_CALL_NAMES)
        and not (runner_calls & _ATTENTION_CALL_NAMES)
    )
    checks["no_auxiliary_loss_calls"] = bool(
        len(runner_loss_calls - {"l1_loss"}) == 0
        and len(encoder_loss_calls) == 0
    )
    checks["no_distillation_calls"] = bool(
        not (runner_calls & _DISTILLATION_CALL_NAMES)
        and not (_called_names(encoder_source) & _DISTILLATION_CALL_NAMES)
    )
    checks["train_loss_is_l1_only"] = bool(runner_loss_calls == {"l1_loss"})

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "details": details,
        "n_valid_molecules_used": 96,
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint("cpu"),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"FSAB sanity failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# diagnostics helpers
# ---------------------------------------------------------------------------


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return float(math.sqrt(total))


def _fsab_grad_norms(model: nn.Module) -> dict[str, float]:
    encoder = model.structural_encoder
    return {
        "structure_grad_norm": _module_grad_norm(encoder.message)
        + _module_grad_norm(encoder.update)
        + _module_grad_norm(encoder.structure_pool),
        "attribute_grad_norm": _module_grad_norm(encoder.atom_mlp)
        + _module_grad_norm(encoder.bond_mlp)
        + _module_grad_norm(encoder.attribute_fuse),
        "binding_node_grad_norm": _module_grad_norm(encoder.node_role_projection)
        + _module_grad_norm(encoder.node_attribute_projection),
        "binding_edge_grad_norm": _module_grad_norm(encoder.edge_role_mlp)
        + _module_grad_norm(encoder.edge_attribute_mlp)
        + _module_grad_norm(encoder.edge_role_projection)
        + _module_grad_norm(encoder.edge_attribute_projection),
        "binding_grad_norm": _module_grad_norm(encoder.binding_fuse),
        "fusion_grad_norm": _module_grad_norm(encoder.structure_weight)
        + _module_grad_norm(encoder.attribute_weight)
        + _module_grad_norm(encoder.binding_weight)
        + _module_grad_norm(encoder.fusion_mlp),
        "patch_encoder_grad_norm": _module_grad_norm(model.patch_encoder),
    }


def _collect_channel_stats(
    encoder: nn.Module, batches: Sequence[Any]
) -> dict[str, float]:
    encoder.capture_diagnostics = True
    accumulators: dict[str, float] = {}
    weights: dict[str, float] = {}
    try:
        for batch in batches:
            with torch.no_grad():
                encoder(batch)
            stats = encoder.last_stats
            weight = float(stats.get("n_patches", 0) or 1)
            for key, value in stats.items():
                if not isinstance(value, (int, float)):
                    continue
                accumulators[key] = accumulators.get(key, 0.0) + float(value) * weight
                weights[key] = weights.get(key, 0.0) + weight
    finally:
        encoder.capture_diagnostics = False
    return {
        key: accumulators[key] / max(weights[key], 1.0e-12) for key in accumulators
    }


def _channel_matrix(encoder: nn.Module, batches: Sequence[Any], channel: str) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for batch in batches:
        with torch.no_grad():
            blocks.append(encoder.forward_channels(batch)[channel].cpu().numpy())
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 1))


def _effective_rank(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"effective_rank": 0.0, "participation_ratio": 0.0}
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    if total <= 0.0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0}
    probability = energy / total
    probability = probability[probability > 0.0]
    entropy = float(-(probability * np.log(probability)).sum())
    return {
        "effective_rank": float(math.exp(entropy)),
        "participation_ratio": float(total * total / float((energy**2).sum())),
        "top_singular_fraction": float(energy[0] / total),
    }


def _parameter_norm(module: nn.Module) -> float:
    return float(
        math.sqrt(sum(float(p.detach().pow(2).sum()) for p in module.parameters()))
    )


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def train_seed(
    seed: int,
    device: str = "cpu",
    tag: str = "fsab",
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train_data, valid_data, _audit = build_encoded_records()
    protocol = dict(shead.OPTIMIZED_PROTOCOL)
    if protocol_override:
        protocol.update(protocol_override)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats()
    model = build_candidate(int(seed)).to(device_obj)
    total_params = _n_params(model)
    if total_params > PARAM_UPPER:
        raise RuntimeError(f"candidate params {total_params} exceed budget")

    diag_loader = _make_fsab_loader(
        list(valid_data)[:DIAG_BATCHES], 128, False, 0
    )
    diag_batches = [b.to(device_obj) for b in diag_loader]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    loader = _make_fsab_loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = _make_fsab_loader(
        valid_data,
        int(protocol["batch_size"]),
        False,
        int(seed) + int(protocol["eval_shuffle_seed_offset"]),
    )
    steps_per_epoch = int(math.ceil(len(train_data) / int(protocol["batch_size"])))
    patience = int(protocol["patience"])
    max_epochs = int(protocol["max_epochs"])

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    top5: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        last_grads: dict[str, float] = {}
        n_train_batches = len(loader)
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            if batch_index == n_train_batches - 1:
                last_grads = _fsab_grad_norms(model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(protocol["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae = _evaluate_mae(model, eval_loader, device_obj)
        stats = _collect_channel_stats(model.structural_encoder, diag_batches)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_mae),
                "valid_mae": float(valid_mae),
                **{
                    key: float(value)
                    for key, value in stats.items()
                    if key
                    in {
                        "S_norm_mean",
                        "S_norm_std",
                        "A_norm_mean",
                        "A_norm_std",
                        "B_norm_mean",
                        "B_norm_std",
                        "e_struct_norm_mean",
                        "e_struct_norm_std",
                        "contribution_W_S_S",
                        "contribution_W_A_A",
                        "contribution_W_B_B",
                    }
                },
                **{key: float(value) for key, value in last_grads.items()},
                "lr": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": int(epoch * steps_per_epoch),
                "checkpoint_selected": 0,
            }
        )
        state_copy = copy.deepcopy(model.state_dict())
        top5.append((float(valid_mae), int(epoch), state_copy))
        top5.sort(key=lambda item: (item[0], item[1]))
        top5 = top5[:5]
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs:
            print(
                f"[{tag} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"|S|={stats.get('S_norm_mean', 0.0):.3f} "
                f"|A|={stats.get('A_norm_mean', 0.0):.3f} "
                f"|B|={stats.get('B_norm_mean', 0.0):.4f} "
                f"gS={last_grads.get('structure_grad_norm', 0.0):.2e} "
                f"gA={last_grads.get('attribute_grad_norm', 0.0):.2e} "
                f"gB={last_grads.get('binding_grad_norm', 0.0):.2e}",
                flush=True,
            )
        if stale >= patience:
            print(
                f"[{tag} seed{seed}] early_stop epoch={epoch} best={best_epoch}",
                flush=True,
            )
            break
    wall_clock = float(time.perf_counter() - started)
    if best_state is not None:
        model.load_state_dict(best_state)
    for row in curve:
        row["checkpoint_selected"] = int(row["epoch"] == best_epoch)

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = CURVE_DIR / f"{tag}_seed{seed}_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        for row in curve:
            writer.writerow(row)
    selection_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    torch.save(model.state_dict(), selection_path)
    torch.save(
        [{"valid_mae": float(v), "epoch": int(e), "state": s} for v, e, s in top5],
        SOUP_DIR / f"{tag}_seed{seed}_top5_states.pt",
    )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "seed": int(seed),
        "protocol": protocol,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "train_loss_at_best": float(losses[best_epoch - 1]),
        "epochs_run": int(len(losses)),
        "steps_per_epoch": int(steps_per_epoch),
        "optimizer_steps": int(len(losses) * steps_per_epoch),
        "wall_clock_s": wall_clock,
        "epoch_time_s": float(wall_clock) / max(len(losses), 1),
        "early_stopped": bool(len(losses) < max_epochs),
        "parameters": int(total_params),
        "cell": CELL,
        "h_dim": H_DIM,
        "q_dim": Q_DIM,
        "recurrence_rounds": T_ROUNDS,
        "patch_representation": "factorized_binding",
        "fsab_encoder_params": int(_n_params(model.structural_encoder)),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device_obj.type == "cuda"
            else 0
        ),
        "state_path": str(selection_path),
        "curve_path": str(curve_path),
        "git_commit": _git_commit(),
        "environment": _environment_fingerprint(str(device)),
        "official_test_loaded": False,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_DIR / f"{tag}_seed{seed}.json", summary)
    print(
        f"[{tag} seed{seed}] best_valid={best_mae:.6f} epoch={best_epoch} "
        f"params={total_params} wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


def soup_seed(seed: int, tag: str = "fsab") -> dict[str, Any]:
    entries = torch.load(
        SOUP_DIR / f"{tag}_seed{seed}_top5_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    if len(entries) < 5:
        raise RuntimeError(f"need >= 5 snapshots for top-5 soup, got {len(entries)}")
    ranked = sorted(
        entries, key=lambda row: (float(row["valid_mae"]), int(row["epoch"]))
    )[:5]
    states = [row["state"] for row in ranked]
    keys = list(states[0].keys())
    soup_state = {
        key: torch.stack([state[key].float() for state in states], dim=0)
        .mean(dim=0)
        .to(states[0][key].dtype)
        for key in keys
    }
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    _train, valid_data, _ = build_encoded_records()
    loader = _selection_loader(valid_data)
    model = build_candidate(seed)
    selection_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    targets, best_preds = vd._predict_state(
        model,
        torch.load(selection_path, map_location="cpu", weights_only=True),
        loader,
    )
    _t, soup_preds = vd._predict_state(model, soup_state, loader)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": tag,
        "seed": int(seed),
        "top5_epochs": [int(row["epoch"]) for row in ranked],
        "top5_valid_mae": [float(row["valid_mae"]) for row in ranked],
        "best_checkpoint_valid_mae": vd._mae(targets, best_preds),
        "top5_soup_valid_mae": vd._mae(targets, soup_preds),
        "soup_improvement_over_best": vd._mae(targets, best_preds)
        - vd._mae(targets, soup_preds),
        "soup_state_path": str(soup_path),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    return payload


def train_queue(seeds: Sequence[int], device: str, tag: str = "fsab") -> None:
    for seed in seeds:
        if (RUNS_DIR / f"{tag}_seed{seed}.json").exists() and (
            SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
        ).exists():
            print(f"skip existing seed{seed}", flush=True)
            continue
        print(f"=== train seed{seed} device={device} ===", flush=True)
        train_seed(int(seed), device=device, tag=tag)
        soup_seed(int(seed), tag=tag)


def smoke(device: str = "cpu", steps: int = 5) -> dict[str, Any]:
    """GPU smoke: forward / backward / optimizer steps / channel liveness."""
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_data, valid_data, _audit = build_encoded_records()
    model = build_candidate(0).to(device_obj)
    loader = _make_fsab_loader(list(valid_data)[:64], 64, False, 0)
    batch = next(iter(loader)).to(device_obj)
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    losses: list[float] = []
    finite = True
    model.train()
    for _ in range(int(steps)):
        prediction = model(batch).view(-1)
        target = batch.y.view(-1)
        loss = F.l1_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        finite = finite and bool(torch.isfinite(loss).all())
    stats = _collect_channel_stats(model.structural_encoder, [batch])
    grads = _fsab_grad_norms(model)
    matrix = _channel_matrix(model.structural_encoder, [batch], "output")
    rank = _effective_rank(matrix)
    channels = model.structural_encoder.forward_channels(batch)
    channel_std = {
        "S": float(channels["structure"].std(dim=0).mean()),
        "A": float(channels["attributes"].std(dim=0).mean()),
        "B": float(channels["binding"].std(dim=0).mean()),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "device": str(device),
        "steps": int(steps),
        "losses": losses,
        "losses_finite": bool(finite),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "channel_stats": stats,
        "gradient_norms": grads,
        "output_effective_rank": rank,
        "structure_alive": bool(
            channel_std["S"] > 0.0 and grads["structure_grad_norm"] > 0.0
        ),
        "attribute_alive": bool(
            channel_std["A"] > 0.0 and grads["attribute_grad_norm"] > 0.0
        ),
        "binding_alive": bool(
            channel_std["B"] > 0.0
            and (
                grads["binding_node_grad_norm"] > 0.0
                or grads["binding_edge_grad_norm"] > 0.0
            )
        ),
        "output_not_constant": bool(rank["effective_rank"] > 0.1),
        "channel_std_S": channel_std["S"],
        "channel_std_A": channel_std["A"],
        "channel_std_B": channel_std["B"],
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device_obj.type == "cuda"
            else 0
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return sspe._state_sha256(state)


def repro(seed: int, epochs: int, device: str, run_tag: str = "default") -> dict[str, Any]:
    global CURVE_DIR, STATE_DIR, RUNS_DIR, SOUP_DIR
    tag = str(run_tag or "default").replace("/", "_")
    saved = (CURVE_DIR, STATE_DIR, RUNS_DIR, SOUP_DIR)
    root = RESULTS_DIR / "repro" / tag
    CURVE_DIR, STATE_DIR, RUNS_DIR, SOUP_DIR = (
        root / "curves",
        root / "states",
        root / "runs",
        root / "soup_states",
    )
    try:
        summary = train_seed(
            int(seed),
            device=device,
            tag=f"repro_{tag}",
            protocol_override={"max_epochs": int(epochs), "patience": int(epochs)},
        )
        state = torch.load(
            STATE_DIR / f"repro_{tag}_seed{seed}_selection_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "gpu_reproducibility_sanity",
            "seed": int(seed),
            "epochs": int(epochs),
            "run_tag": tag,
            "device": str(device),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "best_valid_mae": float(summary["best_valid_mae"]),
            "best_epoch": int(summary["best_epoch"]),
            "epochs_run": int(summary["epochs_run"]),
            "wall_clock_s": float(summary["wall_clock_s"]),
            "selection_state_sha256": _state_sha256(state),
            "official_test_loaded": False,
        }
    finally:
        CURVE_DIR, STATE_DIR, RUNS_DIR, SOUP_DIR = saved
    _write_json(RESULTS_DIR / f"repro_{tag}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# diagnostics / witness / interventions
# ---------------------------------------------------------------------------


def _diagnostic_batches(count: int = DIAG_BATCHES) -> list[Any]:
    _train, valid_data, _audit = build_encoded_records()
    loader = _make_fsab_loader(list(valid_data)[:count], 128, False, 0)
    return [b.to("cpu") for b in loader]


def diagnostics(tag: str = "fsab") -> dict[str, Any]:
    device = torch.device("cpu")
    batches = _diagnostic_batches(64)
    viability: dict[str, Any] = {}
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "patch_representation": "factorized_binding",
        "per_seed": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        if not state_path.exists():
            continue
        model = build_candidate(seed).to(device)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        encoder = model.structural_encoder
        encoder.capture_diagnostics = True
        s_matrix = _channel_matrix(encoder, batches, "structure")
        a_matrix = _channel_matrix(encoder, batches, "attributes")
        b_matrix = _channel_matrix(encoder, batches, "binding")
        e_matrix = _channel_matrix(encoder, batches, "output")
        stats = _collect_channel_stats(encoder, batches)
        dead_fraction = float(
            np.mean(
                [
                    float((p.detach().abs() < 1.0e-30).float().mean())
                    for p in encoder.parameters()
                ]
            )
        )
        entry = {
            "S_norm_mean": stats.get("S_norm_mean", 0.0),
            "S_norm_std": stats.get("S_norm_std", 0.0),
            "A_norm_mean": stats.get("A_norm_mean", 0.0),
            "A_norm_std": stats.get("A_norm_std", 0.0),
            "B_norm_mean": stats.get("B_norm_mean", 0.0),
            "B_norm_std": stats.get("B_norm_std", 0.0),
            "e_struct_norm_mean": stats.get("e_struct_norm_mean", 0.0),
            "e_struct_norm_std": stats.get("e_struct_norm_std", 0.0),
            "contribution_W_S_S": stats.get("contribution_W_S_S", 0.0),
            "contribution_W_A_A": stats.get("contribution_W_A_A", 0.0),
            "contribution_W_B_B": stats.get("contribution_W_B_B", 0.0),
            "S_rank": _effective_rank(s_matrix),
            "A_rank": _effective_rank(a_matrix),
            "B_rank": _effective_rank(b_matrix),
            "e_struct_rank": _effective_rank(e_matrix),
            "parameter_norm_S": _parameter_norm(encoder.message)
            + _parameter_norm(encoder.update)
            + _parameter_norm(encoder.structure_pool),
            "parameter_norm_A": _parameter_norm(encoder.atom_mlp)
            + _parameter_norm(encoder.bond_mlp)
            + _parameter_norm(encoder.attribute_fuse),
            "parameter_norm_B": _parameter_norm(encoder.node_role_projection)
            + _parameter_norm(encoder.node_attribute_projection)
            + _parameter_norm(encoder.edge_role_mlp)
            + _parameter_norm(encoder.edge_attribute_mlp)
            + _parameter_norm(encoder.edge_role_projection)
            + _parameter_norm(encoder.edge_attribute_projection)
            + _parameter_norm(encoder.binding_fuse),
            "dead_parameter_fraction": dead_fraction,
        }
        payload["per_seed"][str(seed)] = entry
        viability[str(seed)] = _channel_viability(entry)
    payload["viability"] = viability
    payload["channels_alive"] = bool(viability) and all(
        entry["alive"] for entry in viability.values()
    )
    _write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


def _channel_viability(entry: Mapping[str, Any]) -> dict[str, Any]:
    def _alive(prefix: str) -> bool:
        return bool(
            float(entry.get(f"{prefix}_norm_std", 0.0)) > 0.0
            and float(entry.get(f"{prefix}_rank", {}).get("effective_rank", 0.0))
            > 0.0
            and float(entry.get(f"parameter_norm_{prefix}", 0.0)) > 0.0
        )

    return {
        "S_alive": _alive("S"),
        "A_alive": _alive("A"),
        "B_alive": _alive("B"),
        "alive": bool(_alive("S") and _alive("A") and _alive("B")),
        "dead_parameter_fraction": float(entry.get("dead_parameter_fraction", 0.0)),
    }


def witness(seed: int = 0, tag: str = "fsab") -> dict[str, Any]:
    """True-vs-shuffle S/A/B/e_struct deltas on fixed valid witness patches."""
    device = torch.device("cpu")
    state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"missing selection state {state_path}")
    model = build_candidate(seed).to(device)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    encoder = model.structural_encoder
    batches = _diagnostic_batches(WITNESS_BATCHES)

    def _channel_values(batch, shuffled: bool, shuffle_seed: int):
        if shuffled:
            shuffled_atom, shuffled_bond = shuffle_attribute_assignment(
                batch, shuffle_seed
            )
            view = _FSABBatch(
                shuffled_atom,
                batch.struct_root,
                batch.struct_dist,
                batch.struct_patch,
                batch.struct_edge_patch,
                shuffled_bond,
                batch.struct_src,
                batch.struct_dst,
            )
        else:
            view = batch
        channels = encoder.forward_channels(view)
        return {
            "S": channels["structure"].detach(),
            "A": channels["attributes"].detach(),
            "B": channels["binding"].detach(),
            "e": channels["output"].detach(),
        }

    def _prediction(batch, shuffled: bool, shuffle_seed: int):
        if shuffled:
            shuffled_atom, shuffled_bond = shuffle_attribute_assignment(
                batch, shuffle_seed
            )
            view_batch = batch.clone()
            view_batch.struct_atom = shuffled_atom
            view_batch.struct_bond = shuffled_bond
        else:
            view_batch = batch
        with torch.no_grad():
            return model(view_batch).view(-1).detach()

    deltas: dict[str, list[float]] = {key: [] for key in ("S", "A", "B", "e", "pred")}
    max_deltas: dict[str, list[float]] = {key: [] for key in deltas}
    for batch in batches:
        true = _channel_values(batch, False, 0)
        true_pred = _prediction(batch, False, 0)
        true["pred"] = true_pred
        for repeat in range(SHUFFLE_REPEATS):
            shuffled = _channel_values(batch, True, 20260930 + repeat)
            shuffled["pred"] = _prediction(batch, True, 20260930 + repeat)
            for key in deltas:
                diff = (true[key] - shuffled[key]).abs()
                deltas[key].append(float(diff.mean()))
                max_deltas[key].append(float(diff.max()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "n_batches": len(batches),
        "shuffle_repeats": SHUFFLE_REPEATS,
        "mean_abs_delta": {key: float(np.mean(value)) for key, value in deltas.items()},
        "max_abs_delta": {
            key: float(np.mean(value)) for key, value in max_deltas.items()
        },
        "global_max_abs_delta": {
            key: float(np.max(value)) for key, value in max_deltas.items()
        },
        "S_delta_is_numerical_noise": bool(
            float(np.max(max_deltas["S"])) <= PURITY_TOLERANCE
        ),
        "A_delta_is_numerical_noise": bool(
            float(np.max(max_deltas["A"])) <= PURITY_TOLERANCE
        ),
        "B_delta_above_noise": bool(
            float(np.mean(deltas["B"])) >= BINDING_GAP_FLOOR
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"witness_seed{seed}.json", payload)
    return payload


def interventions(seed: int = 0, tag: str = "fsab") -> dict[str, Any]:
    """Frozen-checkpoint channel interventions and shuffled-B prediction."""
    device = torch.device("cpu")
    state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"missing selection state {state_path}")
    model = build_candidate(seed).to(device)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    encoder = model.structural_encoder

    _train, valid_data, _audit = build_encoded_records()
    loader = _make_fsab_loader(valid_data, 128, False, 0)
    targets: list[torch.Tensor] = []
    predictions = {
        "true": [],
        "no_S": [],
        "no_A": [],
        "no_B": [],
    }
    shuffle_seeds = {}
    for repeat in range(SHUFFLE_REPEATS):
        shuffle_seeds[f"shuffle_B_{repeat}"] = []
    encoder.set_intervention(None)
    with torch.no_grad():
        for batch in loader:
            targets.append(batch.y.view(-1).detach())
            predictions["true"].append(model(batch).view(-1).detach())
            for name, channel in (("no_S", "S"), ("no_A", "A"), ("no_B", "B")):
                encoder.set_intervention({channel})
                predictions[name].append(model(batch).view(-1).detach())
            encoder.set_intervention(None)
            for repeat in range(SHUFFLE_REPEATS):
                shuffled_atom, shuffled_bond = shuffle_attribute_assignment(
                    batch, 20260930 + repeat
                )
                shuffled_batch = batch.clone()
                shuffled_batch.struct_atom = shuffled_atom
                shuffled_batch.struct_bond = shuffled_bond
                shuffle_seeds[f"shuffle_B_{repeat}"].append(
                    model(shuffled_batch).view(-1).detach()
                )
    encoder.set_intervention(None)
    target = torch.cat(targets).cpu().numpy().astype(np.float64)

    def _stack_mae(rows: Sequence[torch.Tensor]) -> float:
        values = torch.cat(list(rows)).cpu().numpy().astype(np.float64)
        return float(np.mean(np.abs(target - values)))

    mae = {name: _stack_mae(value) for name, value in predictions.items()}
    shuffle_mae = {
        name: _stack_mae(value) for name, value in shuffle_seeds.items()
    }
    shuffle_mean = float(np.mean(list(shuffle_mae.values())))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "valid_mae_true": mae["true"],
        "valid_mae_no_S": mae["no_S"],
        "valid_mae_no_A": mae["no_A"],
        "valid_mae_no_B": mae["no_B"],
        "valid_mae_shuffle_B_per_repeat": shuffle_mae,
        "valid_mae_shuffle_B_mean": shuffle_mean,
        "delta_no_S": mae["no_S"] - mae["true"],
        "delta_no_A": mae["no_A"] - mae["true"],
        "delta_no_B": mae["no_B"] - mae["true"],
        "delta_shuffle_B": shuffle_mean - mae["true"],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"interventions_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision / report
# ---------------------------------------------------------------------------


def decide(tag: str = "fsab") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for seed in SEEDS:
        soup_path = RESULTS_DIR / f"soup_{tag}_seed{seed}.json"
        run_path = RUNS_DIR / f"{tag}_seed{seed}.json"
        if not soup_path.exists() or not run_path.exists():
            continue
        soup = _read_json(soup_path)
        run = _read_json(run_path)
        rows[str(seed)] = {
            "raw_best_valid": float(soup["best_checkpoint_valid_mae"]),
            "soup_valid": float(soup["top5_soup_valid_mae"]),
            "best_epoch": int(run["best_epoch"]),
            "epochs_run": int(run["epochs_run"]),
            "top5_epochs": soup["top5_epochs"],
            "wall_clock_s": float(run["wall_clock_s"]),
            "peak_gpu_memory_bytes": int(run.get("peak_gpu_memory_bytes", 0)),
        }
    diag_path = RESULTS_DIR / "diagnostics.json"
    diag = _read_json(diag_path) if diag_path.exists() else {"per_seed": {}}
    channels_alive = bool(diag.get("channels_alive", False))
    witness_seed0_path = RESULTS_DIR / "witness_seed0.json"
    witness_seed0 = (
        _read_json(witness_seed0_path) if witness_seed0_path.exists() else None
    )
    mechanism_ok = bool(
        witness_seed0 is not None
        and witness_seed0["S_delta_is_numerical_noise"]
        and witness_seed0["A_delta_is_numerical_noise"]
        and witness_seed0["B_delta_above_noise"]
    )

    if len(rows) < 2:
        seed0 = rows.get("0")
        guard_pass = bool(
            seed0 is not None and float(seed0["soup_valid"]) <= SEED0_GUARD
        )
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "SEED0_ONLY",
            "per_seed": rows,
            "seed0_guard_threshold": SEED0_GUARD,
            "seed0_guard_pass": guard_pass,
            "seed1_authorized": bool(guard_pass and channels_alive and mechanism_ok),
            "channels_alive": channels_alive,
            "mechanism_ok": mechanism_ok,
            "official_test_loaded": False,
        }
        _write_json(RESULTS_DIR / "decision.json", payload)
        return payload

    soup_mean = float(np.mean([rows[str(s)]["soup_valid"] for s in SEEDS]))
    raw_mean = float(np.mean([rows[str(s)]["raw_best_valid"] for s in SEEDS]))
    delta_bag = soup_mean - REFERENCE_BBAG_SOUP_VALID
    delta_bfull = soup_mean - REFERENCE_BFULL_SOUP_VALID
    delta_a2 = soup_mean - REFERENCE_A2_SOUP_VALID
    delta_bag_per_seed = {
        str(seed): rows[str(seed)]["soup_valid"] - REFERENCE_BBAG_SOUP_PER_SEED[seed]
        for seed in SEEDS
    }
    delta_bfull_per_seed = {
        str(seed): rows[str(seed)]["soup_valid"] - REFERENCE_BFULL_SOUP_PER_SEED[seed]
        for seed in SEEDS
    }

    if channels_alive and delta_bfull <= 0.002:
        case = "A_fsab_alive_near_or_above_bfull"
    elif channels_alive and delta_bag < 0 < delta_bfull:
        case = "B_fsab_alive_above_bbag_below_bfull"
    elif channels_alive and abs(delta_bag) < STRONG_GATE:
        case = "C_fsab_alive_tied_with_bbag"
    elif not channels_alive:
        case = "E_binding_channel_collapsed"
    elif delta_bag >= REGRESSION_GATE:
        case = "bad_regression"
    else:
        case = "D_fsab_weak_capacity"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference": {
            "bbag_2seed_soup_valid": REFERENCE_BBAG_SOUP_VALID,
            "bfull_2seed_soup_valid": REFERENCE_BFULL_SOUP_VALID,
            "a2_2seed_soup_valid": REFERENCE_A2_SOUP_VALID,
            "bbag_params": REFERENCE_BBAG_PARAMS,
            "bfull_params": REFERENCE_BFULL_PARAMS,
        },
        "per_seed": rows,
        "candidate_2seed_soup_valid_mean": soup_mean,
        "candidate_2seed_raw_valid_mean": raw_mean,
        "delta_soup_mean_vs_bbag": float(delta_bag),
        "delta_soup_mean_vs_bfull": float(delta_bfull),
        "delta_soup_mean_vs_a2": float(delta_a2),
        "delta_per_seed_vs_bbag": delta_bag_per_seed,
        "delta_per_seed_vs_bfull": delta_bfull_per_seed,
        "channels_alive": channels_alive,
        "mechanism_ok": mechanism_ok,
        "case": case,
        "candidate_params": int(
            _read_json(RESULTS_DIR / "parameter_accounting.json")["candidate_params"]
        ),
        "vocab_independent": True,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "parameter_accounting": _maybe("parameter_accounting.json"),
        "sanity": (lambda d: None if d is None else d["checks"])(_maybe("sanity.json")),
        "smoke": _maybe("smoke.json"),
        "diagnostics": _maybe("diagnostics.json"),
        "witness_seed0": _maybe("witness_seed0.json"),
        "witness_seed1": _maybe("witness_seed1.json"),
        "interventions_seed0": _maybe("interventions_seed0.json"),
        "interventions_seed1": _maybe("interventions_seed1.json"),
        "decision": _maybe("decision.json"),
        "official_test_loaded": False,
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
            "params",
            "preprocess",
            "sanity",
            "smoke",
            "train",
            "train_queue",
            "repro",
            "soup",
            "diagnostics",
            "witness",
            "interventions",
            "decide",
            "report",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="fsab")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    global DETERMINISTIC
    DETERMINISTIC = bool(args.deterministic)
    torch.set_num_threads(4)
    _set_deterministic(DETERMINISTIC)

    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    elif args.stage == "preprocess":
        print(
            json.dumps(preprocess(force=bool(args.force)), indent=2, default=str),
            flush=True,
        )
    elif args.stage == "sanity":
        print(json.dumps(sanity()["checks"], indent=2, default=str), flush=True)
    elif args.stage == "smoke":
        print(
            json.dumps(smoke(device=args.device, steps=args.steps), indent=2, default=str),
            flush=True,
        )
    elif args.stage == "train":
        print(
            json.dumps(
                train_seed(args.seed, device=args.device, tag=args.tag),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    elif args.stage == "train_queue":
        train_queue(
            [int(s) for s in args.seeds.split(",") if s != ""],
            args.device,
            tag=args.tag,
        )
    elif args.stage == "soup":
        print(
            json.dumps(soup_seed(args.seed, tag=args.tag), indent=2, default=str),
            flush=True,
        )
    elif args.stage == "repro":
        print(
            json.dumps(
                repro(args.seed, args.epochs, args.device, run_tag=args.run_tag or "default"),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    elif args.stage == "diagnostics":
        print(json.dumps(diagnostics(tag=args.tag), indent=2, default=str), flush=True)
    elif args.stage == "witness":
        print(
            json.dumps(witness(seed=args.seed, tag=args.tag), indent=2, default=str),
            flush=True,
        )
    elif args.stage == "interventions":
        print(
            json.dumps(
                interventions(seed=args.seed, tag=args.tag), indent=2, default=str
            ),
            flush=True,
        )
    elif args.stage == "decide":
        print(json.dumps(decide(tag=args.tag), indent=2, default=str), flush=True)
    elif args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
