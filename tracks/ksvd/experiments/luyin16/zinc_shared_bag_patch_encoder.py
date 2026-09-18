"""ZINC shared permutation-invariant BAG patch encoder (connectivity ablation).

Question
--------
The completed shared connectivity-aware encoder (``shared_structural`` / B-full)
improves the deterministic-A100 cell-A ZINC validation MAE
(2-seed Top-5 soup ``0.126368`` -> ``0.118972``) while its 16-D ``e_struct`` is
nearly rank-1.  It is therefore unclear whether the gain comes from the *real
rooted-patch connectivity* or merely from a shared, low-dimensional
compositional function of the patch primitives.

This experiment builds the single missing control, **B-bag**: an encoder that
sees exactly the same input primitives as B-full -- atom type, root flag,
root-distance and bond type -- with the same output width (16) and a matched
shared-parameter budget, but which **never uses node-node adjacency**.  It
treats a patch as a bag of node primitives plus a bag of bond-type primitives
and aggregates them with a permutation-invariant DeepSets pooling.

    B-bag:   primitives -> shared node MLP -> mean/std/root pooling
                          + shared bond MLP -> mean/std pooling
                          -> fusion -> e_bag in R^16
    B-full:  the same primitives -> 2 rounds of edge-aware message passing over
             the REAL patch connectivity -> [root; mean; std] -> e_struct

Nothing downstream changes: radius-2 patch definition, relation system, pair
descriptors, ``T=2`` weight-tied recurrent pair--centre, parent embedding,
global/topology channels, R width (334), graph head and the frozen optimized
training protocol are all inherited bit-exactly from cell A / B-full.  Only the
patch representation differs.

The reference points for the mechanism decomposition are the existing frozen
results (no retraining):

    A2  (no typed identity + capacity-matched shared adapter)
        2-seed Top-5 soup valid = 0.121914
    B-full (shared connectivity-aware encoder)
        2-seed Top-5 soup valid = 0.118972

Official ZINC test is **never** loaded by the selection/mechanism stages; it is
read exactly once by the terminal ``test`` stage (user-authorised one-shot
closure), which never re-selects an epoch or changes the frozen soup rule.

Stages: ``params preprocess sanity train soup train_queue repro diagnostics
decide report test``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_shared_bag_patch_encoder <stage>
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
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
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_training_sufficiency as ztraining,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as sspe
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
    base_config as _v4_base_config,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/shared_bag_patch_encoder"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "shared_bag_patch_encoder_v1"

# The patch-graph proposal is shared with B-full.  The cache is target-free and
# lives in the completed shared-structural experiment; reusing it verbatim
# guarantees an identical patch split / preprocessing between B-full and B-bag.
CACHE_DIR = sspe.CACHE_DIR
CACHE_SCHEMA_VERSION = sspe.CACHE_SCHEMA_VERSION

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

# --- pre-registered connectivity-free bag encoder (one architecture) ---------
# Embedding widths are copied from B-full (node 48 / bond 24).  The MLP and
# fusion widths were fixed once (no sweep) so the structural encoder budget
# matches B-full's 35,152-param encoder; see ``parameter_accounting.json``.
BAG_NODE_DIM = 48
BAG_EDGE_DIM = 24
BAG_NODE_HIDDEN = 96
BAG_BOND_HIDDEN = 48
BAG_FUSION_HIDDEN = 104

# --- reference results (frozen; never retrained in this study) ---------------
# B-full: results/shared_structural_patch_encoder/decision.json
REFERENCE_BFULL_SOUP_VALID = 0.11897220489243046
REFERENCE_BFULL_SOUP_PER_SEED = {
    0: 0.11981802638241788,
    1: 0.11812638340244302,
}
REFERENCE_BFULL_PARAMS = 84495
REFERENCE_BFULL_ENCODER_PARAMS = 35152
# A2: results/compact_v4_identity_capacity_control/decision.json
REFERENCE_A2_SOUP_VALID = 0.12191404939390486
REFERENCE_A2_SOUP_PER_SEED = {
    0: 0.12169362585240742,
    1: 0.1221344729354023,
}
REFERENCE_A2_PARAMS = 85740

PARAM_LOWER = 80000
PARAM_UPPER = 90000
# total must be within +/-1% of the B-full candidate (84,495).
BUDGET_MATCH_TOLERANCE = 0.01

STRONG_GATE = 0.002
REGRESSION_GATE = 0.003
DECOMPOSITION_SCALE = 0.002

SEEDS = (0, 1)

DETERMINISTIC = False

_write_json = sspe._write_json
_read_json = sspe._read_json
_git_commit = sspe._git_commit
_environment_fingerprint = sspe._environment_fingerprint
_set_deterministic = sspe._set_deterministic
_n_params = sspe._n_params
_effective_rank = sspe._effective_rank
_evaluate_mae = sspe._evaluate_mae
_predict_state = vd._predict_state
_mae = vd._mae
_state_sha256 = sspe._state_sha256


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_candidate(seed: int = 0) -> torch.nn.Module:
    """Cell-A geometry with the typed lookup replaced by the bag encoder.

    Construction order mirrors ``sspe.build_candidate`` exactly so every shared
    tensor is copied bit-exactly from the matched cell-A initialisation and the
    small head is re-drawn from the historical ``head_seed=0`` stream.  The only
    difference is ``patch_representation="shared_bag"``.
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
            patch_representation="shared_bag",
            structural_node_dim=BAG_NODE_DIM,
            structural_edge_dim=BAG_EDGE_DIM,
            bag_node_hidden=BAG_NODE_HIDDEN,
            bag_bond_hidden=BAG_BOND_HIDDEN,
            bag_fusion_hidden=BAG_FUSION_HIDDEN,
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


# ---------------------------------------------------------------------------
# structural tensors / batching
# ---------------------------------------------------------------------------


def _attach_struct_tensors_bag(encoded: Sequence[Data], graphs: Sequence[Any]) -> None:
    """Attach primitives + explicit bond->patch grouping (no endpoint needed)."""
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
        # explicit bond -> patch grouping so the bag forward never reads an
        # edge endpoint (struct_src / struct_dst).
        data.struct_edge_patch = torch.from_numpy(np.asarray(graph.edge_patch))


def build_encoded_records():
    """Return (train_data, valid_data, audit) with bag structural tensors."""
    train_bundle, valid_bundle, _meta = sspe.extract_records()
    config = _v4_base_config()
    config["model"]["device"] = "cpu"
    enc_train, enc_valid, audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    _attach_struct_tensors_bag(enc_train, train_bundle["graphs"])
    _attach_struct_tensors_bag(enc_valid, valid_bundle["graphs"])
    audit = dict(audit)
    audit["patch_representation"] = "shared_bag"
    audit["bag_encoder"] = {
        "node_dim": BAG_NODE_DIM,
        "edge_dim": BAG_EDGE_DIM,
        "node_hidden": BAG_NODE_HIDDEN,
        "bond_hidden": BAG_BOND_HIDDEN,
        "fusion_hidden": BAG_FUSION_HIDDEN,
        "output_dim": TOKEN_WIDTH,
        "rounds": 0,
    }
    audit["official_test_loaded"] = False
    return enc_train, enc_valid, audit


def bag_collate(data_list: Sequence[Data]) -> Any:
    """Batch with explicit offsets for the bag structural tensors.

    ``struct_patch`` and ``struct_edge_patch`` are graph-local patch ids and
    are offset by the cumulative patch count; ``struct_src`` / ``struct_dst``
    (used only by the B-full diagnostic comparisons) are offset by the
    cumulative node count.
    """
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


def _make_bag_loader(
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


def _selection_loader(valid_data: Sequence[Data]):
    return _make_bag_loader(valid_data, 128, False, 0)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    baseline = cd.build_cell(CELL, 0)
    reference = sspe.build_candidate(0)
    candidate = build_candidate(0)
    base_total = _n_params(baseline)
    cand_total = _n_params(candidate)
    ref_total = _n_params(reference)
    if base_total != sspe.REFERENCE_PARAMS:
        raise RuntimeError(
            f"cell-A baseline params changed: {base_total} != {sspe.REFERENCE_PARAMS}"
        )
    if ref_total != REFERENCE_BFULL_PARAMS:
        raise RuntimeError(
            f"B-full params changed: {ref_total} != {REFERENCE_BFULL_PARAMS}"
        )
    encoder_params = _n_params(candidate.structural_encoder)
    relative = abs(cand_total - REFERENCE_BFULL_PARAMS) / REFERENCE_BFULL_PARAMS
    audit = zpp.audit_parameters(candidate)
    typed_keys = [key for key in candidate.state_dict() if "typed_embedding" in key]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cell": CELL,
        "baseline_params": int(base_total),
        "bfull_params": int(ref_total),
        "candidate_params": int(cand_total),
        "candidate_minus_bfull": int(cand_total - REFERENCE_BFULL_PARAMS),
        "candidate_relative_budget_gap": float(relative),
        "budget_matched_within_1pct": bool(relative <= BUDGET_MATCH_TOLERANCE),
        "params_in_range": bool(PARAM_LOWER <= cand_total <= PARAM_UPPER),
        "bag_encoder_params": int(encoder_params),
        "bfull_encoder_params": int(
            _n_params(reference.structural_encoder)
        ),
        "encoder_params_minus_bfull": int(
            encoder_params - REFERENCE_BFULL_ENCODER_PARAMS
        ),
        "candidate_has_typed_embedding": bool(
            getattr(candidate, "typed_embedding", None) is not None
        ),
        "candidate_typed_state_keys": typed_keys,
        "candidate_block_params": audit["blocks"],
        "candidate_dimensions": audit["dimensions"],
        "baseline_block_params": zpp.audit_parameters(baseline)["blocks"],
        "bfull_block_params": zpp.audit_parameters(reference)["blocks"],
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
# integrity / mechanism sanity
# ---------------------------------------------------------------------------


class _BagBatch:
    """Bag-only primitive container (deliberately has no edge endpoints)."""

    def __init__(self, atom, root, dist, patch, edge_patch, bond) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_edge_patch = edge_patch
        self.struct_bond = bond


class _FullBatch(_BagBatch):
    """Same primitives plus real endpoints, for the B-full comparison."""

    def __init__(self, atom, root, dist, patch, edge_patch, bond, src, dst) -> None:
        super().__init__(atom, root, dist, patch, edge_patch, bond)
        self.struct_src = src
        self.struct_dst = dst


def _adversarial_connectivity_pair():
    """Two patches with identical primitive multisets, different connectivity.

    Both are rooted trees on 5 nodes with atom all-0, the root at node 0 and
    the SAME distance labelling ``[0, 1, 1, 2, 2]``.  The bond-type multiset is
    ``{0, 0, 0, 0}`` in both.  Only the edge endpoints differ, so only a
    connectivity-aware encoder can tell them apart.
    """
    atom = torch.tensor([0, 0, 0, 0, 0])
    root = torch.tensor([1, 0, 0, 0, 0])
    dist = torch.tensor([0, 1, 1, 2, 2])
    patch = torch.zeros(5, dtype=torch.long)

    # A: root -> {1, 2}; 1 -> 3 ; 2 -> 4   (both depth-1 children branch)
    undirected_a = [(0, 1), (0, 2), (1, 3), (2, 4)]
    # B: root -> {1, 2}; 2 -> {3, 4}       (one depth-1 child branches twice)
    undirected_b = [(0, 1), (0, 2), (2, 3), (2, 4)]

    def directed(edges):
        both = edges + [(v, u) for u, v in edges]
        src = torch.tensor([u for u, _ in both], dtype=torch.long)
        dst = torch.tensor([v for _, v in both], dtype=torch.long)
        bond = torch.zeros(len(both), dtype=torch.long)
        edge_patch = torch.zeros(len(both), dtype=torch.long)
        return src, dst, bond, edge_patch

    src_a, dst_a, bond_a, ep_a = directed(undirected_a)
    src_b, dst_b, bond_b, ep_b = directed(undirected_b)
    batch_a = _FullBatch(atom, root, dist, patch, ep_a, bond_a, src_a, dst_a)
    batch_b = _FullBatch(atom, root, dist, patch, ep_b, bond_b, src_b, dst_b)
    return batch_a, batch_b


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    train_data, valid_data, _audit = build_encoded_records()
    loader = _make_bag_loader(list(valid_data)[:128], 128, False, 0)
    batch = next(iter(loader)).to(device)

    model = build_candidate(0).to(device)
    baseline = cd.build_cell(CELL, 0).to(device)
    bfull = sspe.build_candidate(0).to(device)

    checks: dict[str, bool] = {}

    # (9) no vocab-sized typed lookup anywhere in the model.
    typed_state = [k for k in model.state_dict() if "typed_embedding" in k]
    typed_named = [k for k, _ in model.named_parameters() if "typed_embedding" in k]
    checks["no_vocab_sized_typed_embedding_weight"] = (
        len(typed_state) == 0 and len(typed_named) == 0
    )
    checks["bag_encoder_present"] = bool(
        getattr(model, "structural_encoder", None) is not None
    )
    # (8) parent path preserved.
    checks["parent_embedding_present"] = hasattr(model, "parent_embedding") and (
        _n_params(model.parent_embedding) == _n_params(baseline.parent_embedding)
    )
    # (11) q=16, h=64, T=2.
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
    # (12) parameter budget vs B-full.
    total = _n_params(model)
    checks["params_in_range"] = bool(PARAM_LOWER <= total <= PARAM_UPPER)
    checks["params_match_bfull_within_1pct"] = bool(
        abs(total - REFERENCE_BFULL_PARAMS) / REFERENCE_BFULL_PARAMS
        <= BUDGET_MATCH_TOLERANCE
    )
    # (10) output width.
    encoder = model.structural_encoder
    checks["output_width_16"] = bool(int(encoder.output_dim) == 16)

    # (13) forward / backward finite.
    model.train()
    out = model(batch)
    loss = F.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    checks["forward_finite"] = bool(torch.isfinite(out).all())
    checks["backward_finite_nonzero"] = all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    ) and any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in model.parameters()
    )
    model.zero_grad(set_to_none=True)

    # --- encoder-level invariance / sensitivity on a real batch ------------
    encoder.eval()
    with torch.no_grad():
        e_ref = encoder(batch)

    atom = batch.struct_atom.clone()
    root = batch.struct_root.clone()
    dist = batch.struct_dist.clone()
    patch = batch.struct_patch.clone()
    edge_patch = batch.struct_edge_patch.clone()
    bond = batch.struct_bond.clone()
    n_nodes = int(atom.shape[0])
    perm = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(20260928))
    with torch.no_grad():
        e_perm = encoder(
            _BagBatch(atom[perm], root[perm], dist[perm], patch[perm], edge_patch, bond)
        )
    checks["node_relabel_invariant"] = bool(
        float((e_ref - e_perm).abs().max()) < 1.0e-5
    )

    # (2) atom type change.
    atom2 = atom.clone()
    atom2[0] = (int(atom2[0].item()) + 1) % int(encoder.atom_categories)
    with torch.no_grad():
        e_atom = encoder(_BagBatch(atom2, root, dist, patch, edge_patch, bond))
    checks["atom_type_changes_representation"] = bool(
        float((e_ref - e_atom).abs().max()) > 1.0e-6
    )

    # (3) bond-type multiset change.
    bond2 = bond.clone()
    if bond2.numel():
        bond2[0] = (int(bond2[0].item()) + 1) % int(encoder.bond_categories)
    with torch.no_grad():
        e_bond = encoder(_BagBatch(atom, root, dist, patch, edge_patch, bond2))
    checks["bond_type_multiset_changes_representation"] = bool(
        float((e_ref - e_bond).abs().max()) > 1.0e-6
    )

    # (4) root designation change.
    root3 = root.clone()
    root_indices = torch.nonzero(root3 > 0, as_tuple=False).view(-1)
    patch0_root = int(root_indices[0].item())
    same_patch = torch.nonzero(
        patch == patch[patch0_root], as_tuple=False
    ).view(-1)
    alternative = [int(i) for i in same_patch.tolist() if int(i) != patch0_root]
    if alternative:
        root3[patch0_root] = 0
        root3[alternative[0]] = 1
    with torch.no_grad():
        e_root = encoder(_BagBatch(atom, root3, dist, patch, edge_patch, bond))
    checks["root_designation_changes_representation"] = bool(
        float((e_ref - e_root).abs().max()) > 1.0e-6
    )

    # (5) root-distance multiset change.
    dist5 = dist.clone()
    dist5[0] = (int(dist5[0].item()) + 1) % int(encoder.n_distance_bins)
    with torch.no_grad():
        e_dist = encoder(_BagBatch(atom, root, dist5, patch, edge_patch, bond))
    checks["root_distance_multiset_changes_representation"] = bool(
        float((e_ref - e_dist).abs().max()) > 1.0e-6
    )

    # (6)+(7) the integrity gate: identical primitives, different connectivity.
    batch_a, batch_b = _adversarial_connectivity_pair()
    encoder.eval()
    bfull.structural_encoder.eval()
    with torch.no_grad():
        bag_a = encoder(batch_a)
        bag_b = encoder(batch_b)
        full_a = bfull.structural_encoder(batch_a)
        full_b = bfull.structural_encoder(batch_b)
    primitives_identical = bool(
        torch.equal(batch_a.struct_atom, batch_b.struct_atom)
        and torch.equal(batch_a.struct_root, batch_b.struct_root)
        and torch.equal(batch_a.struct_dist, batch_b.struct_dist)
        and sorted(batch_a.struct_bond.tolist())
        == sorted(batch_b.struct_bond.tolist())
        and batch_a.struct_src.tolist() != batch_b.struct_src.tolist()
    )
    checks["adversarial_pair_primitives_identical"] = primitives_identical
    checks["bag_connectivity_invariant"] = bool(
        float((bag_a - bag_b).abs().max()) == 0.0
    )
    checks["bfull_connectivity_sensitive"] = bool(
        float((full_a - full_b).abs().max()) > 1.0e-6
    )

    # (8) the bag forward never touches edge endpoints.
    try:
        with torch.no_grad():
            _ = encoder(_BagBatch(atom, root, dist, patch, edge_patch, bond))
        checks["bag_forward_without_endpoints"] = True
    except Exception:  # pragma: no cover - defensive
        checks["bag_forward_without_endpoints"] = False

    # model-level: clearing the certificate vocabulary id changes nothing.
    model.eval()
    original_token = batch.typed_token
    batch.typed_token = torch.full_like(original_token, 0)
    with torch.no_grad():
        out_no_tokens = model.encode(batch)
    batch.typed_token = original_token
    with torch.no_grad():
        out_tokens = model.encode(batch)
    checks["no_certificate_dependence"] = bool(
        float((out_no_tokens - out_tokens).abs().max()) == 0.0
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_params": int(total),
        "bfull_params": int(REFERENCE_BFULL_PARAMS),
        "bag_encoder_params": int(_n_params(encoder)),
        "bfull_encoder_params": int(_n_params(bfull.structural_encoder)),
        "adversarial_pair": {
            "connectivity_changed_only": bool(primitives_identical),
            "bag_max_abs_delta": float((bag_a - bag_b).abs().max()),
            "bfull_max_abs_delta": float((full_a - full_b).abs().max()),
        },
        "split": "first 128 official-valid molecules",
        "environment": _environment_fingerprint("cpu"),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"shared bag sanity failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def train_seed(
    seed: int,
    device: str = "cpu",
    tag: str = "sbpe",
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
    if not PARAM_LOWER <= total_params <= PARAM_UPPER:
        raise RuntimeError(f"candidate params {total_params} outside budget")
    if abs(total_params - REFERENCE_BFULL_PARAMS) / REFERENCE_BFULL_PARAMS > (
        BUDGET_MATCH_TOLERANCE
    ):
        raise RuntimeError(
            f"candidate params {total_params} not budget-matched to "
            f"B-full {REFERENCE_BFULL_PARAMS}"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    loader = _make_bag_loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = _make_bag_loader(
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
        for batch in loader:
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(protocol["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae = _evaluate_mae(model, eval_loader, device_obj)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_mae),
                "valid_mae": float(valid_mae),
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
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch}",
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
        [
            {"valid_mae": float(v), "epoch": int(e), "state": s}
            for v, e, s in top5
        ],
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
        "patch_representation": "shared_bag",
        "bag_encoder_params": int(_n_params(model.structural_encoder)),
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


def soup_seed(seed: int, tag: str = "sbpe") -> dict[str, Any]:
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
    targets, best_preds = _predict_state(
        model,
        torch.load(selection_path, map_location="cpu", weights_only=True),
        loader,
    )
    _t, soup_preds = _predict_state(model, soup_state, loader)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": tag,
        "seed": int(seed),
        "top5_epochs": [int(row["epoch"]) for row in ranked],
        "top5_valid_mae": [float(row["valid_mae"]) for row in ranked],
        "best_checkpoint_valid_mae": _mae(targets, best_preds),
        "top5_soup_valid_mae": _mae(targets, soup_preds),
        "soup_improvement_over_best": _mae(targets, best_preds)
        - _mae(targets, soup_preds),
        "soup_state_path": str(soup_path),
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
        "valid_targets": targets.tolist(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    return payload


def train_queue(seeds: Sequence[int], device: str, tag: str = "sbpe") -> None:
    for seed in seeds:
        if (RUNS_DIR / f"{tag}_seed{seed}.json").exists() and (
            SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
        ).exists():
            print(f"skip existing seed{seed}", flush=True)
            continue
        print(f"=== train seed{seed} device={device} ===", flush=True)
        train_seed(int(seed), device=device, tag=tag)
        soup_seed(int(seed), tag=tag)


def repro(
    seed: int, epochs: int, device: str, run_tag: str = "default"
) -> dict[str, Any]:
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
            protocol_override={
                "max_epochs": int(epochs),
                "patience": int(epochs),
            },
        )
        state = torch.load(
            STATE_DIR / f"repro_{tag}_seed{seed}_selection_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        state_hash = _state_sha256(state)
        curve_path = CURVE_DIR / f"repro_{tag}_seed{seed}_curve.csv"
        valid_curve: list[float] = []
        if curve_path.exists():
            for line in curve_path.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split(",")
                if len(parts) > 2:
                    valid_curve.append(float(parts[2]))
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
            "valid_mae_curve": valid_curve,
            "selection_state_sha256": state_hash,
            "official_test_loaded": False,
        }
    finally:
        CURVE_DIR, STATE_DIR, RUNS_DIR, SOUP_DIR = saved
    _write_json(RESULTS_DIR / f"repro_{tag}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def diagnostics(tag: str = "sbpe") -> dict[str, Any]:
    device = torch.device("cpu")
    _train, valid_data, _audit = build_encoded_records()
    loader = _make_bag_loader(list(valid_data)[:256], 128, False, 0)
    batches = [b.to(device) for b in loader]

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": {},
        "patch_representation": "shared_bag",
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        model = build_candidate(seed).to(device)
        model.load_state_dict(
            torch.load(
                STATE_DIR / f"{tag}_seed{seed}_selection_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.capture_diagnostics = True
        encoder_blocks: list[np.ndarray] = []
        h0_blocks: list[np.ndarray] = []
        for batch in batches:
            with torch.no_grad():
                model.encode(batch)
                encoder_blocks.append(
                    model.structural_encoder(batch).cpu().numpy()
                )
            if model.last_h0 is not None:
                h0_blocks.append(model.last_h0.cpu().numpy())
        encoder_matrix = np.concatenate(encoder_blocks, axis=0)
        h0_matrix = (
            np.concatenate(h0_blocks, axis=0) if h0_blocks else np.zeros((0, 1))
        )
        payload["per_seed"][str(seed)] = {
            "bag_encoder_rank": _effective_rank(encoder_matrix),
            "patch_h0_rank": _effective_rank(h0_matrix),
            "n_encoder_samples": int(encoder_matrix.shape[0]),
            "n_h0_samples": int(h0_matrix.shape[0]),
        }
    _write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# one-shot official-test closure
# ---------------------------------------------------------------------------


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
    return mean, std


def terminal_test(tag: str = "sbpe") -> dict[str, Any]:
    """Load the official ZINC test split exactly once and evaluate frozen bags.

    This is the single authorised official-test read for the B-bag control.  It
    reuses the pre-existing per-seed Top-5 soup states (frozen by validation),
    never re-selects an epoch, never changes the soup rule, and writes a
    one-shot unlock marker before the test split is touched.  The 2-seed soup is
    the equal-weight mean of the per-seed soup test MAEs (matching the
    validation definition); the prediction ensemble is reported only as a
    diagnostic.
    """
    unlock_path = RESULTS_DIR / "official_test_unlock.json"
    if unlock_path.exists():
        raise RuntimeError(
            "official test already unlocked once; refusing to re-read it"
        )

    soup_rows = {
        seed: _read_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json")
        for seed in SEEDS
    }
    run_rows = {
        seed: _read_json(RUNS_DIR / f"{tag}_seed{seed}.json") for seed in SEEDS
    }
    _write_json(
        RESULTS_DIR / "official_test_freeze.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "role": "terminal official-test closure for the B-bag mechanism control",
            "architecture": "shared_bag",
            "h_dim": H_DIM,
            "q_dim": Q_DIM,
            "recurrence_rounds": T_ROUNDS,
            "params": int(run_rows[0]["parameters"]),
            "seeds": [int(s) for s in SEEDS],
            "soup_rule": {
                "K": 5,
                "ranking": "lowest official-valid MAE",
                "tie_rule": "earliest epoch",
                "weight_aggregation": "equal-weight arithmetic parameter mean",
                "frozen_before_test": True,
            },
            "validation": {
                "soup_valid_mae": {
                    str(s): float(soup_rows[s]["top5_soup_valid_mae"])
                    for s in SEEDS
                },
                "raw_valid_mae": {
                    str(s): float(soup_rows[s]["best_checkpoint_valid_mae"])
                    for s in SEEDS
                },
                "top5_epochs": {str(s): soup_rows[s]["top5_epochs"] for s in SEEDS},
                "best_epochs": {str(s): int(run_rows[s]["best_epoch"]) for s in SEEDS},
            },
            "test_loaded_at_freeze_time": False,
            "official_test_loaded": False,
        },
    )
    _write_json(
        unlock_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "frozen_before_test": True,
            "encoding": "transforms fit on official train only (identical to selection)",
            "checkpoints": "pre-existing seed0/seed1 selection states and fixed Top-5 soups",
            "test_used_for_selection_or_tuning": False,
            "official_test_loaded": True,
            "git_commit": _git_commit(),
        },
    )

    # --- first and only official-test load ---------------------------------
    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    config["model"]["device"] = "cpu"
    _enc_train, enc_test, audit = zpp._phase_data(
        train_records, test_records, config=config
    )
    del _enc_train
    test_ds = _load_zinc(ZINC_ROOT, "test")
    test_graphs = sspe._patch_graphs_from_dataset(test_ds)
    if len(test_graphs) != len(enc_test):
        raise RuntimeError("test patch graph / record molecule count mismatch")
    for graph, data in zip(test_graphs, enc_test):
        if graph.n_patches != int(data.num_nodes):
            raise RuntimeError("test patch count mismatch")
    _attach_struct_tensors_bag(enc_test, test_graphs)

    loader = _make_bag_loader(enc_test, 128, False, 0)
    targets = np.asarray(
        [float(graph.y.view(-1)[0]) for graph in enc_test], dtype=np.float64
    )

    rows: list[dict[str, Any]] = []
    raw_preds: dict[int, np.ndarray] = {}
    soup_preds: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model = build_candidate(int(seed))
        _t_raw, raw_pred = _predict_state(
            model,
            torch.load(
                STATE_DIR / f"{tag}_seed{seed}_selection_state.pt",
                map_location="cpu",
                weights_only=True,
            ),
            loader,
        )
        _t_soup, soup_pred = _predict_state(
            model,
            torch.load(
                SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt",
                map_location="cpu",
                weights_only=True,
            ),
            loader,
        )
        if not np.array_equal(_t_raw, targets) or not np.array_equal(
            _t_soup, targets
        ):
            raise RuntimeError("official-test target order mismatch")
        raw_preds[int(seed)] = raw_pred
        soup_preds[int(seed)] = soup_pred
        rows.append(
            {
                "seed": int(seed),
                "raw_selection_test_mae": _mae(targets, raw_pred),
                "soup_test_mae": _mae(targets, soup_pred),
            }
        )

    raw_values = np.array([row["raw_selection_test_mae"] for row in rows])
    soup_values = np.array([row["soup_test_mae"] for row in rows])
    raw_mean, raw_std = _mean_std(raw_values)
    soup_mean, soup_std = _mean_std(soup_values)
    raw_ensemble = np.mean(np.stack([raw_preds[s] for s in SEEDS], axis=0), axis=0)
    soup_ensemble = np.mean(np.stack([soup_preds[s] for s in SEEDS], axis=0), axis=0)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": "official-test terminal evaluation of the B-bag mechanism control",
        "n_test": int(targets.shape[0]),
        "rows": rows,
        "soup_test_per_seed": {str(r["seed"]): r["soup_test_mae"] for r in rows},
        "raw_test_per_seed": {
            str(r["seed"]): r["raw_selection_test_mae"] for r in rows
        },
        "soup_test_mean": soup_mean,
        "soup_test_std": soup_std,
        "raw_test_mean": raw_mean,
        "raw_test_std": raw_std,
        "diagnostic_soup_2seed_prediction_ensemble_test_mae": _mae(
            targets, soup_ensemble
        ),
        "diagnostic_raw_2seed_prediction_ensemble_test_mae": _mae(
            targets, raw_ensemble
        ),
        "diagnostic_note": (
            "the 2-seed prediction ensemble is a diagnostic, NOT a single-model "
            "result; the reported 2-seed soup is the equal-weight mean of the "
            "per-seed Top-5 soup test MAEs, matching the validation definition"
        ),
        "official_test_evaluated": True,
        "test_used_for_selection_or_tuning": False,
        "audit": {key: audit[key] for key in sorted(audit) if key != "topology"},
        "official_test_loaded": True,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision / mechanism decomposition
# ---------------------------------------------------------------------------


def decompose(tag: str = "sbpe") -> dict[str, Any]:
    """A2 -> B-bag -> B-full mechanism decomposition (primary result)."""
    soup_per_seed = {
        seed: float(_read_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json")[
            "top5_soup_valid_mae"
        ])
        for seed in SEEDS
    }
    bag_mean = float(np.mean([soup_per_seed[s] for s in SEEDS]))
    delta_composition = REFERENCE_A2_SOUP_VALID - bag_mean
    delta_connectivity = bag_mean - REFERENCE_BFULL_SOUP_VALID
    delta_total = REFERENCE_A2_SOUP_VALID - REFERENCE_BFULL_SOUP_VALID

    comp_per_seed = {
        str(s): REFERENCE_A2_SOUP_PER_SEED[s] - soup_per_seed[s] for s in SEEDS
    }
    conn_per_seed = {
        str(s): soup_per_seed[s] - REFERENCE_BFULL_SOUP_PER_SEED[s] for s in SEEDS
    }
    comp_dirs = {np.sign(v) for v in comp_per_seed.values()}
    conn_dirs = {np.sign(v) for v in conn_per_seed.values()}

    if (
        abs(delta_composition) < DECOMPOSITION_SCALE
        and delta_connectivity >= DECOMPOSITION_SCALE
        and len(conn_dirs) == 1
    ):
        case = "C1_connectivity_supported"
    elif (
        delta_composition >= DECOMPOSITION_SCALE
        and abs(delta_connectivity) < DECOMPOSITION_SCALE
    ):
        case = "C2_composition_sufficient"
    elif (
        delta_composition > 0
        and delta_connectivity > 0
        and REFERENCE_A2_SOUP_VALID
        > bag_mean
        > REFERENCE_BFULL_SOUP_VALID
        and max(delta_composition, delta_connectivity) >= DECOMPOSITION_SCALE
    ):
        case = "C3_both_contribute"
    else:
        case = "C4_no_robust_decomposition"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed_bag_soup_valid": soup_per_seed,
        "bag_2seed_soup_mean": bag_mean,
        "reference_a2_soup_mean": REFERENCE_A2_SOUP_VALID,
        "reference_bfull_soup_mean": REFERENCE_BFULL_SOUP_VALID,
        "delta_composition_a2_minus_bag": float(delta_composition),
        "delta_connectivity_bag_minus_bfull": float(delta_connectivity),
        "delta_total_a2_minus_bfull": float(delta_total),
        "delta_composition_per_seed": comp_per_seed,
        "delta_connectivity_per_seed": conn_per_seed,
        "delta_composition_same_direction": bool(len(comp_dirs) == 1),
        "delta_connectivity_same_direction": bool(len(conn_dirs) == 1),
        "case": case,
        "decomposition_scale": DECOMPOSITION_SCALE,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decomposition.json", payload)
    return payload


def decide(tag: str = "sbpe") -> dict[str, Any]:
    rows = {}
    for seed in SEEDS:
        soup = _read_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json")
        run = _read_json(RUNS_DIR / f"{tag}_seed{seed}.json")
        rows[str(seed)] = {
            "raw_best_valid": float(soup["best_checkpoint_valid_mae"]),
            "soup_valid": float(soup["top5_soup_valid_mae"]),
            "best_epoch": int(run["best_epoch"]),
            "epochs_run": int(run["epochs_run"]),
            "train_at_best": float(run["train_loss_at_best"]),
            "top5_epochs": soup["top5_epochs"],
            "wall_clock_s": float(run["wall_clock_s"]),
            "peak_gpu_memory_bytes": int(run.get("peak_gpu_memory_bytes", 0)),
        }
    soup_mean = float(np.mean([rows[str(s)]["soup_valid"] for s in SEEDS]))
    raw_mean = float(np.mean([rows[str(s)]["raw_best_valid"] for s in SEEDS]))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference": {
            "a2_2seed_soup_valid": REFERENCE_A2_SOUP_VALID,
            "bfull_2seed_soup_valid": REFERENCE_BFULL_SOUP_VALID,
            "bfull_params": REFERENCE_BFULL_PARAMS,
        },
        "per_seed": rows,
        "candidate_2seed_soup_valid_mean": soup_mean,
        "candidate_2seed_raw_valid_mean": raw_mean,
        "candidate_params": int(
            _read_json(RESULTS_DIR / "parameter_accounting.json")[
                "candidate_params"
            ]
        ),
        "vocab_independent": True,
        "no_typed_lookup": True,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / f"{name}.json"
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "params": _maybe("parameter_accounting"),
        "sanity": _maybe("sanity"),
        "diagnostics": _maybe("diagnostics"),
        "decision": _maybe("decision"),
        "decomposition": _maybe("decomposition"),
        "official_test_freeze": _maybe("official_test_freeze"),
        "official_test_results": _maybe("official_test_results"),
        "soup": {
            str(seed): _maybe(f"soup_sbpe_seed{seed}") for seed in SEEDS
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "params",
            "preprocess",
            "sanity",
            "train",
            "soup",
            "train_queue",
            "repro",
            "diagnostics",
            "decompose",
            "decide",
            "report",
            "test",
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="sbpe")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    global DETERMINISTIC
    DETERMINISTIC = bool(args.deterministic)
    _set_deterministic(DETERMINISTIC)
    torch.set_num_threads(4)

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if args.stage == "params":
        print(params(), flush=True)
    elif args.stage == "preprocess":
        sspe.extract_records(force=bool(args.force))
    elif args.stage == "sanity":
        print(sanity(), flush=True)
    elif args.stage == "train":
        print(train_seed(args.seed, device=args.device, tag=args.tag), flush=True)
    elif args.stage == "train_queue":
        train_queue(seeds, device=args.device, tag=args.tag)
    elif args.stage == "soup":
        print(soup_seed(args.seed, tag=args.tag), flush=True)
    elif args.stage == "repro":
        print(
            repro(args.seed, args.epochs, args.device, run_tag=args.run_tag),
            flush=True,
        )
    elif args.stage == "diagnostics":
        print(diagnostics(tag=args.tag), flush=True)
    elif args.stage == "decompose":
        print(decompose(tag=args.tag), flush=True)
    elif args.stage == "decide":
        print(decide(tag=args.tag), flush=True)
    elif args.stage == "report":
        print(report(), flush=True)
    elif args.stage == "test":
        print(terminal_test(tag=args.tag), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
