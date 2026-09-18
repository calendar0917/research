"""ZINC Z1 / Local Adaptive Structure-Binding (ASB) cell.

Research question
-----------------
The connectivity-free shared bag encoder (B-bag) and the shared connectivity-
aware encoder (B-full) both emit a single 16-D ``e_struct`` per rooted typed
radius-2 patch.  B-bag never reads the patch's node--node adjacency; B-full
propagates messages over the *whole* patch.  Neither lets the model decide
*which* part of the patch is the task-relevant structure.

This experiment implements **Z1 / Local ASB**: a single shared cell that

1. forms an explicit, connected support ``S_i`` (with the real induced bond set)
   *inside* each existing radius-2 patch by activating structure--attribute
   bindings, and
2. performs the patch computation on that learned support, using the *same*
   binding states as the messages.

The radius-2 patch is the search region; the learned support is the actual
structure.  There is no outer context -> support revision (that is Z2).

Degeneration contract
---------------------
With all support gates open and the three perturbation paths at zero the cell
reduces exactly to B-bag.  The natural initialisation is *near*-function-
preserving (tiny final-layer weights + open gates) so gradients reach the gate,
binding, message and update modules from step one without making B-bag a
permanent bypass.

Inherited machinery (unchanged): radius-2 patch extraction, pair relation
system, pair descriptors, ``T=2`` weight-tied recurrent pair--centre, parent
embedding, global/topology channels, graph head, optimizer / LR / weight decay /
batch size / gradient clipping and the Top-5 soup protocol.  The only variable
is the patch representation (B-bag -> ASB cell).

References (frozen, never retrained here):
    B-bag  2-seed Top-5 soup valid = 0.12380552224389975
    B-full 2-seed Top-5 soup valid = 0.11897220489243046
    A2     2-seed Top-5 soup valid = 0.12191404939390486

Official ZINC test is **never** loaded.

Stages: ``params preprocess sanity train soup train_queue repro diagnostics
support_audit decide report``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_adaptive_structure_binding_cell <stage>
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
    zinc_shared_bag_patch_encoder as sbpe,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
    base_config as _v4_base_config,
)
from tracks.ksvd.experiments.luyin16.structural_patch_encoder import (
    AdaptiveStructureBindingEncoder,
    SharedBagPatchEncoder,
)

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/adaptive_structure_binding_cell"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "adaptive_structure_binding_cell_z1_v1"

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

# --- pre-registered ASB cell geometry (one architecture; no sweep) -----------
ASB_NODE_DIM = 48
ASB_EDGE_DIM = 24
ASB_NODE_HIDDEN = 96          # == B-bag node hidden
ASB_BOND_HIDDEN = 48          # == B-bag bond hidden (shape-compatible copy)
ASB_BIND_HIDDEN = 32          # binding-delta / bind-update hidden
ASB_GATE_HIDDEN = 24
ASB_MESSAGE_HIDDEN = 32
ASB_UPDATE_HIDDEN = 32
ASB_BIND_UPDATE_HIDDEN = 32
ASB_FUSION_HIDDEN = 104       # == B-bag fusion hidden
ASB_PERTURB_INIT = 1.0e-3     # tiny final-layer std of the perturbation paths
ASB_GATE_BIAS_INIT = 3.0      # sigmoid(3.0) ~ 0.953 -> supports start open
ASB_GATE_WEIGHT_INIT_STD = 1.0e-2

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

PARAM_LOWER = 80000
PARAM_UPPER = 120000

STRONG_GATE = 0.002
REGRESSION_GATE = 0.003
NONTRIVIAL_FRACTION_MARGIN = 1.0e-3  # support is non-trivial below 0.999 full

SEEDS = (0, 1)
SUPPORT_EXAMPLE_MOLECULES = 8
SUPPORT_NUM_QUANTILES = (0.1, 0.5, 0.9)

DETERMINISTIC = False

_write_json = sbpe._write_json
_read_json = sbpe._read_json
_git_commit = sbpe._git_commit
_environment_fingerprint = sbpe._environment_fingerprint
_set_deterministic = sspe._set_deterministic
_n_params = sbpe._n_params
_effective_rank = sspe._effective_rank
_predict_state = vd._predict_state
_mae = vd._mae
_state_sha256 = sspe._state_sha256


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_candidate(seed: int = 0) -> torch.nn.Module:
    """Cell-A geometry with the typed lookup replaced by the ASB cell.

    Construction order mirrors ``sspe.build_candidate`` / ``sbpe.build_candidate``
    exactly so every shared tensor is copied bit-exactly from the matched
    cell-A initialisation and the small head is re-drawn from the historical
    ``head_seed=0`` stream.  The structural encoder's B-bag-compatible attribute
    maps are then copied from the frozen B-bag initialisation so the ASB cell
    starts near the B-bag function.
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
            patch_representation="adaptive_structure_binding",
            structural_node_dim=ASB_NODE_DIM,
            structural_edge_dim=ASB_EDGE_DIM,
            bag_node_hidden=ASB_NODE_HIDDEN,
            bag_fusion_hidden=ASB_FUSION_HIDDEN,
            asb_bond_hidden=ASB_BOND_HIDDEN,
            asb_bind_hidden=ASB_BIND_HIDDEN,
            asb_gate_hidden=ASB_GATE_HIDDEN,
            asb_message_hidden=ASB_MESSAGE_HIDDEN,
            asb_update_hidden=ASB_UPDATE_HIDDEN,
            asb_bind_update_hidden=ASB_BIND_UPDATE_HIDDEN,
            asb_perturb_init=ASB_PERTURB_INIT,
            **kwargs,
        )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in baseline_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
    # near-function-preserving initialisation: copy the B-bag attribute maps.
    bbag_encoder = sbpe.build_candidate(int(seed)).structural_encoder
    model.structural_encoder.load_attribute_pretrained(bbag_encoder)
    torch.manual_seed(int(shead.SMALL_HEAD_SEED))
    model.head = shead.GenericReader(
        int(model.unified_graph_width), shead.SMALL_HEAD_HIDDEN
    )
    return model


def bbag_reference_encoder(seed: int = 0) -> SharedBagPatchEncoder:
    """The frozen B-bag structural encoder for the matched seed."""
    return sbpe.build_candidate(int(seed)).structural_encoder


# ---------------------------------------------------------------------------
# data / batching (reuse the shared structural preprocessing cache verbatim)
# ---------------------------------------------------------------------------


def build_encoded_records():
    train_bundle, valid_bundle, _meta = sspe.extract_records()
    config = _v4_base_config()
    config["model"]["device"] = "cpu"
    train_data, valid_data, audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    sspe._attach_struct_tensors(train_data, train_bundle["graphs"])
    sspe._attach_struct_tensors(valid_data, valid_bundle["graphs"])
    # ``struct_edge_patch`` is required by the connectivity-free B-bag encoder
    # used as the degeneration reference; it is a grouping index, not adjacency.
    for encoded, graphs in ((train_data, train_bundle["graphs"]),
                             (valid_data, valid_bundle["graphs"])):
        for data, graph in zip(encoded, graphs):
            data.struct_edge_patch = torch.from_numpy(graph.edge_patch)
    audit = dict(audit)
    audit["patch_representation"] = "adaptive_structure_binding"
    audit["asb_cell"] = {
        "node_dim": ASB_NODE_DIM,
        "edge_dim": ASB_EDGE_DIM,
        "node_hidden": ASB_NODE_HIDDEN,
        "bond_hidden": ASB_BOND_HIDDEN,
        "bind_hidden": ASB_BIND_HIDDEN,
        "gate_hidden": ASB_GATE_HIDDEN,
        "message_hidden": ASB_MESSAGE_HIDDEN,
        "update_hidden": ASB_UPDATE_HIDDEN,
        "bind_update_hidden": ASB_BIND_UPDATE_HIDDEN,
        "fusion_hidden": ASB_FUSION_HIDDEN,
        "output_dim": TOKEN_WIDTH,
        "perturb_init": ASB_PERTURB_INIT,
        "gate_bias_init": ASB_GATE_BIAS_INIT,
    }
    audit["official_test_loaded"] = False
    return train_data, valid_data, audit


def asb_collate(data_list: Sequence[Any]) -> Any:
    """Batch the structural tensors (patch / node / edge-patch offsets)."""
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(list(data_list))
    node_offset = 0
    patch_offset = 0
    patch_parts: list[torch.Tensor] = []
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    edge_patch_parts: list[torch.Tensor] = []
    patch_batch_parts: list[torch.Tensor] = []
    for graph_index, data in enumerate(data_list):
        n_patches = int(data.num_nodes)
        patch_parts.append(data.struct_patch + patch_offset)
        edge_patch_parts.append(data.struct_edge_patch + patch_offset)
        src_parts.append(data.struct_src + node_offset)
        dst_parts.append(data.struct_dst + node_offset)
        patch_batch_parts.append(
            torch.full((n_patches,), int(graph_index), dtype=torch.long)
        )
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
    batch.struct_patch_batch = (
        torch.cat(patch_batch_parts)
        if patch_batch_parts
        else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_n_patches = int(patch_offset)
    return batch


def _make_struct_loader(
    graphs: Sequence[Any], batch_size: int, shuffle: bool, seed: int
):
    from torch.utils.data import DataLoader as TorchDataLoader

    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return TorchDataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=asb_collate,
    )
_evaluate_mae = sspe._evaluate_mae


def _selection_loader(valid_data: Sequence[Any]):
    return _make_struct_loader(valid_data, 128, False, 0)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def _encoder_breakdown(encoder: AdaptiveStructureBindingEncoder) -> dict[str, int]:
    return {
        "atom_root_distance_bond_embeddings": int(
            _n_params(encoder.atom_embedding)
            + _n_params(encoder.root_embedding)
            + _n_params(encoder.distance_embedding)
            + _n_params(encoder.bond_embedding)
        ),
        "node_mlp": int(_n_params(encoder.node_mlp)),
        "bond_primitive_mlp": int(_n_params(encoder.bond_mlp)),
        "binding_delta_mlp": int(_n_params(encoder.bind_delta_mlp)),
        "gate_mlp": int(_n_params(encoder.gate_mlp)),
        "message_mlp": int(_n_params(encoder.message_mlp)),
        "update_mlp": int(_n_params(encoder.update_mlp)),
        "bind_update_mlp": int(_n_params(encoder.bind_update)),
        "fusion": int(_n_params(encoder.fusion)),
    }


def _dataset_dependent_params(model: nn.Module) -> int:
    """Parameters whose shape scales with the dataset vocabulary (must be 0)."""
    total = 0
    for name, parameter in model.named_parameters():
        if "typed_embedding" in name or "structural_context_embedding" in name:
            total += int(parameter.numel())
    return int(total)


def params() -> dict[str, Any]:
    baseline = cd.build_cell(CELL, 0)
    reference_bfull = sspe.build_candidate(0)
    reference_bbag = sbpe.build_candidate(0)
    candidate = build_candidate(0)
    base_total = _n_params(baseline)
    cand_total = _n_params(candidate)
    if base_total != sspe.REFERENCE_PARAMS:
        raise RuntimeError(
            f"cell-A baseline params changed: {base_total} != {sspe.REFERENCE_PARAMS}"
        )
    if _n_params(reference_bfull) != REFERENCE_BFULL_PARAMS:
        raise RuntimeError("B-full params changed")
    if _n_params(reference_bbag) != REFERENCE_BBAG_PARAMS:
        raise RuntimeError("B-bag params changed")
    encoder = candidate.structural_encoder
    breakdown = _encoder_breakdown(encoder)
    if sum(breakdown.values()) != _n_params(encoder):
        raise RuntimeError("ASB encoder breakdown does not sum to encoder params")
    typed_keys = [key for key in candidate.state_dict() if "typed_embedding" in key]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cell": CELL,
        "baseline_params": int(base_total),
        "bfull_params": int(_n_params(reference_bfull)),
        "bbag_params": int(_n_params(reference_bbag)),
        "candidate_params": int(cand_total),
        "candidate_minus_bbag": int(cand_total - REFERENCE_BBAG_PARAMS),
        "candidate_minus_bfull": int(cand_total - REFERENCE_BFULL_PARAMS),
        "params_in_range": bool(PARAM_LOWER <= cand_total <= PARAM_UPPER),
        "target_upper_bound": int(PARAM_UPPER),
        "asb_encoder_params": int(_n_params(encoder)),
        "bbag_encoder_params": int(_n_params(reference_bbag.structural_encoder)),
        "bfull_encoder_params": int(_n_params(reference_bfull.structural_encoder)),
        "asb_minus_bbag_encoder": int(
            _n_params(encoder) - _n_params(reference_bbag.structural_encoder)
        ),
        "asb_encoder_breakdown": breakdown,
        "downstream_inherited_params": int(cand_total - _n_params(encoder)),
        "dataset_dependent_params": int(_dataset_dependent_params(candidate)),
        "candidate_has_typed_embedding": bool(
            getattr(candidate, "typed_embedding", None) is not None
        ),
        "candidate_typed_state_keys": typed_keys,
        "candidate_block_params": zpp.audit_parameters(candidate)["blocks"],
        "candidate_dimensions": zpp.audit_parameters(candidate)["dimensions"],
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
# support diagnostics helpers
# ---------------------------------------------------------------------------


def _collect_support_stats(encoder: nn.Module, batches: Sequence[Any]) -> dict[str, Any]:
    """Aggregate the ASB sufficient statistics over a fixed set of batches."""
    encoder.eval()
    acc: dict[str, float] = {}
    support_sizes: list[np.ndarray] = []
    patch_counts: list[np.ndarray] = []
    output_norms: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            out = encoder(batch)
            stats = encoder.last_stats
            for key in (
                "n_patches",
                "n_nodes",
                "n_selected_nodes",
                "n_dist1",
                "n_dist1_selected",
                "n_dist2",
                "n_dist2_selected",
                "n_edges",
                "n_selected_edges",
                "gate_sum",
                "gate_sq_sum",
                "gate_count",
                "gate_entropy_sum",
                "node_state_norm_sum",
                "binding_state_norm_sum",
                "message_norm_sum",
                "update_norm_sum",
            ):
                acc[key] = acc.get(key, 0.0) + float(stats[key])
            support_sizes.append(stats["support_sizes"].cpu().numpy())
            patch_counts.append(stats["patch_node_counts"].cpu().numpy())
            output_norms.append(out.detach().cpu().norm(dim=1).numpy())

    support = np.concatenate(support_sizes) if support_sizes else np.zeros(0)
    patch = np.concatenate(patch_counts) if patch_counts else np.zeros(0)
    out_norm = np.concatenate(output_norms) if output_norms else np.zeros(0)
    gate_count = max(acc.get("gate_count", 0.0), 1.0)
    gate_mean = acc.get("gate_sum", 0.0) / gate_count
    gate_var = max(acc.get("gate_sq_sum", 0.0) / gate_count - gate_mean**2, 0.0)
    n_nodes = max(acc.get("n_nodes", 0.0), 1.0)
    n_edges = max(acc.get("n_edges", 0.0), 1.0)
    nontrivial = (
        patch - support > 0.5
    ) if patch.size else np.zeros(0, dtype=bool)
    return {
        "n_patches": int(acc.get("n_patches", 0.0)),
        "selected_node_fraction": float(
            acc.get("n_selected_nodes", 0.0) / n_nodes
        ),
        "selected_edge_fraction": float(
            acc.get("n_selected_edges", 0.0) / n_edges
        ),
        "distance1_selection_rate": float(
            acc.get("n_dist1_selected", 0.0) / max(acc.get("n_dist1", 0.0), 1.0)
        ),
        "distance2_selection_rate": float(
            acc.get("n_dist2_selected", 0.0) / max(acc.get("n_dist2", 0.0), 1.0)
        ),
        "support_size_mean": float(support.mean()) if support.size else 0.0,
        "support_size_std": float(support.std()) if support.size else 0.0,
        "support_size_min": float(support.min()) if support.size else 0.0,
        "support_size_max": float(support.max()) if support.size else 0.0,
        "support_size_quantiles": {
            str(q): float(np.quantile(support, q)) if support.size else 0.0
            for q in SUPPORT_NUM_QUANTILES
        },
        "patch_size_mean": float(patch.mean()) if patch.size else 0.0,
        "patch_minus_support_mean": (
            float((patch - support).mean()) if patch.size else 0.0
        ),
        "patches_with_nontrivial_support_fraction": (
            float(nontrivial.mean()) if nontrivial.size else 0.0
        ),
        "support_varies_across_patches": bool(
            support.size > 1 and float(support.std()) > 0.0
        ),
        "gate_probability_mean": float(gate_mean),
        "gate_probability_std": float(math.sqrt(gate_var)),
        "gate_entropy": float(acc.get("gate_entropy_sum", 0.0) / gate_count),
        "binding_state_norm_mean": float(
            acc.get("binding_state_norm_sum", 0.0) / n_edges
        ),
        "message_norm_mean": float(acc.get("message_norm_sum", 0.0) / n_nodes),
        "node_state_norm_mean": float(
            acc.get("node_state_norm_sum", 0.0) / n_nodes
        ),
        "update_norm_mean": float(acc.get("update_norm_sum", 0.0) / n_nodes),
        "output_token_norm_mean": float(out_norm.mean()) if out_norm.size else 0.0,
    }


def _asb_module_grad_norms(model: nn.Module) -> dict[str, float]:
    encoder = model.structural_encoder
    if not isinstance(encoder, AdaptiveStructureBindingEncoder):
        return {}
    out: dict[str, float] = {}
    for name in (
        "gate_mlp",
        "bind_delta_mlp",
        "message_mlp",
        "update_mlp",
        "bind_update",
        "node_mlp",
        "bond_mlp",
        "fusion",
        "atom_embedding",
    ):
        module = getattr(encoder, name)
        total = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        out[f"{name}_grad_norm"] = float(math.sqrt(total))
    return out


# ---------------------------------------------------------------------------
# explicit learned-support examples
# ---------------------------------------------------------------------------


def _fixed_molecule_batch(valid_data: Sequence[Any]):
    loader = _make_struct_loader(
        list(valid_data)[:SUPPORT_EXAMPLE_MOLECULES],
        SUPPORT_EXAMPLE_MOLECULES,
        False,
        0,
    )
    return next(iter(loader))


def _support_examples(model: nn.Module, batch: Any, max_patches: int = 24) -> dict[str, Any]:
    """Extract readable explicit learned supports for a fixed valid batch."""
    encoder = model.structural_encoder
    encoder.record_support = True
    try:
        with torch.no_grad():
            encoder(batch)
        support = encoder.last_support
    finally:
        encoder.record_support = False
    selected = support["selected"].bool().numpy()
    distance = support["distance"].numpy()
    node_patch = support["node_patch"].numpy()
    src = support["src"].numpy()
    dst = support["dst"].numpy()
    sel_edge = support["selected_edge"].bool().numpy()
    n_patches = int(support["n_patches"])
    patch_batch = getattr(batch, "struct_patch_batch", None)
    patch_batch = (
        patch_batch.detach().cpu().numpy()
        if patch_batch is not None
        else np.zeros(n_patches, dtype=np.int64)
    )
    examples = []
    for patch in range(min(n_patches, max_patches)):
        node_idx = np.nonzero(node_patch == patch)[0]
        local = {int(node): i for i, node in enumerate(node_idx)}
        selected_local = sorted(local[int(n)] for n in node_idx if selected[n])
        local_distance = sorted(
            int(distance[n]) for n in node_idx if selected[n]
        )
        root_local = [
            local[int(n)] for n in node_idx if int(distance[n]) == 0
        ]
        edge_pairs = []
        for edge_index in range(len(src)):
            if not sel_edge[edge_index]:
                continue
            left = int(src[edge_index])
            right = int(dst[edge_index])
            if left in local and right in local:
                edge_pairs.append(tuple(sorted([local[left], local[right]])))
        edge_pairs = sorted(set(edge_pairs))
        examples.append(
            {
                "molecule": int(patch_batch[patch]) if patch < len(patch_batch) else 0,
                "patch": int(patch),
                "n_patch_nodes": int(len(node_idx)),
                "root_local": root_local,
                "selected_local_nodes": selected_local,
                "selected_local_distances": local_distance,
                "selected_local_edges": [list(pair) for pair in edge_pairs],
                "selected_fraction": (
                    float(len(selected_local) / len(node_idx)) if node_idx.size else 0.0
                ),
            }
        )
    return {
        "n_patches": n_patches,
        "n_examples": len(examples),
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# integrity / sanity
# ---------------------------------------------------------------------------


class _StructBatch:
    def __init__(self, atom, root, dist, patch, src, dst, bond) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_src = src
        self.struct_dst = dst
        self.struct_bond = bond


def _force_full_gate(encoder: AdaptiveStructureBindingEncoder) -> None:
    """Force exactly-open gates and exactly-zero perturbation paths."""
    with torch.no_grad():
        encoder.gate_mlp[-1].bias.fill_(60.0)
        encoder.gate_mlp[-1].weight.zero_()
        for module in (encoder.bind_delta_mlp, encoder.update_mlp, encoder.bind_update):
            module[-1].weight.zero_()
            module[-1].bias.zero_()


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    train_data, valid_data, _audit = build_encoded_records()
    loader = _make_struct_loader(list(valid_data)[:128], 128, False, 0)
    batch = next(iter(loader)).to(device)

    model = build_candidate(0).to(device)
    baseline = cd.build_cell(CELL, 0).to(device)
    bbag = sbpe.build_candidate(0).to(device)
    checks: dict[str, bool] = {}

    typed_state = [k for k in model.state_dict() if "typed_embedding" in k]
    checks["no_vocab_sized_typed_embedding_weight"] = len(typed_state) == 0
    checks["asb_encoder_present"] = isinstance(
        model.structural_encoder, AdaptiveStructureBindingEncoder
    )
    checks["parent_embedding_present"] = _n_params(
        model.parent_embedding
    ) == _n_params(baseline.parent_embedding)
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
    total = _n_params(model)
    checks["params_in_range"] = bool(PARAM_LOWER <= total <= PARAM_UPPER)
    checks["output_width_16"] = int(model.structural_encoder.output_dim) == 16
    checks["dataset_dependent_params_zero"] = (
        _dataset_dependent_params(model) == 0
    )

    # forward / backward finite and gradients alive on the new modules.
    model.train()
    out = model(batch)
    loss = F.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    grads = _asb_module_grad_norms(model)
    checks["forward_finite"] = bool(torch.isfinite(out).all())
    checks["backward_finite"] = all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    checks["gate_gradient_alive"] = grads.get("gate_mlp_grad_norm", 0.0) > 0.0
    checks["binding_gradient_alive"] = (
        grads.get("bind_delta_mlp_grad_norm", 0.0) > 0.0
    )
    checks["message_gradient_alive"] = (
        grads.get("message_mlp_grad_norm", 0.0) > 0.0
    )
    checks["update_gradient_alive"] = grads.get("update_mlp_grad_norm", 0.0) > 0.0
    checks["bind_update_gradient_alive"] = (
        grads.get("bind_update_grad_norm", 0.0) > 0.0
    )

    # natural initialisation is close to B-bag
    asb_encoder = model.structural_encoder
    asb_encoder.eval()
    bbag_encoder = bbag.structural_encoder.eval()
    with torch.no_grad():
        e_asb = asb_encoder(batch)
        e_bbag = bbag_encoder(batch)
    init_delta = float((e_asb - e_bbag).abs().max())
    checks["init_close_to_bbag"] = bool(init_delta < 1.0e-2)

    # full-gate degeneration is (near) exact
    degenerate = copy.deepcopy(asb_encoder).eval()
    _force_full_gate(degenerate)
    with torch.no_grad():
        e_deg = degenerate(batch)
    degenerate_delta = float((e_deg - e_bbag).abs().max())
    checks["full_gate_degeneration_exact"] = bool(degenerate_delta < 1.0e-5)

    model.zero_grad(set_to_none=True)

    # permutation invariance on the real batch
    atom = batch.struct_atom.clone()
    root = batch.struct_root.clone()
    dist = batch.struct_dist.clone()
    patch = batch.struct_patch.clone()
    src = batch.struct_src.clone()
    dst = batch.struct_dst.clone()
    bond = batch.struct_bond.clone()
    n_nodes = int(atom.shape[0])
    perm = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(20260929))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n_nodes)
    with torch.no_grad():
        e_perm = asb_encoder(
            _StructBatch(
                atom[perm], root[perm], dist[perm], patch[perm],
                inverse[src], inverse[dst], bond,
            )
        )
    checks["node_relabel_invariant"] = bool(
        float((e_asb - e_perm).abs().max()) < 1.0e-5
    )

    # connectivity of every hard support on the real batch
    asb_encoder.record_support = True
    with torch.no_grad():
        asb_encoder(batch)
    support = asb_encoder.last_support
    asb_encoder.record_support = False
    checks["supports_connected_and_rooted"] = _all_supports_connected(support)

    # support is non-trivial at init (gates open -> full support)
    stats = _collect_support_stats(asb_encoder, [batch])
    checks["init_support_full_by_construction"] = bool(
        stats["selected_node_fraction"] > 0.999
    )

    # batch invariance: one molecule alone vs inside the 128-molecule batch
    single = next(iter(_make_struct_loader(list(valid_data)[:1], 1, False, 0)))
    with torch.no_grad():
        e_single = asb_encoder(single.to(device))
    molecule_zero = batch.struct_patch_batch == 0
    batch_invariance_delta = float(
        (e_asb[molecule_zero] - e_single).abs().max()
    )
    checks["batch_invariance"] = bool(batch_invariance_delta < 1.0e-5)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_params": int(total),
        "asb_encoder_params": int(_n_params(asb_encoder)),
        "bbag_encoder_params": int(_n_params(bbag_encoder)),
        "bfull_params": int(REFERENCE_BFULL_PARAMS),
        "initialisation_max_abs_delta_vs_bbag": init_delta,
        "full_gate_degeneration_max_abs_delta_vs_bbag": degenerate_delta,
        "batch_invariance_max_abs_delta": batch_invariance_delta,
        "init_support_stats": stats,
        "gradient_norms": grads,
        "split": "first 128 official-valid molecules",
        "environment": _environment_fingerprint("cpu"),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"ASB sanity failed: {failed}")
    return payload


def _all_supports_connected(support: Mapping[str, Any]) -> bool:
    selected = support["selected"].bool().numpy()
    node_patch = support["node_patch"].numpy()
    src = support["src"].numpy()
    dst = support["dst"].numpy()
    sel_edge = support["selected_edge"].bool().numpy()
    distance = support["distance"].numpy()
    n_patches = int(support["n_patches"])

    # one pass over the selected induced edges builds the global adjacency
    adjacency: dict[int, set[int]] = {}
    for edge_index in range(len(src)):
        if not sel_edge[edge_index]:
            continue
        left = int(src[edge_index])
        right = int(dst[edge_index])
        if selected[left] and selected[right]:
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)

    for patch in range(n_patches):
        nodes = np.nonzero(node_patch == patch)[0]
        selected_nodes = [int(n) for n in nodes if selected[n]]
        if not selected_nodes:
            return False
        root_nodes = [int(n) for n in nodes if int(distance[n]) == 0]
        if len(root_nodes) != 1:
            return False
        seen = {root_nodes[0]}
        stack = [root_nodes[0]]
        while stack:
            node = stack.pop()
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if set(selected_nodes) - seen:
            return False
    return True


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def train_seed(
    seed: int,
    device: str = "cpu",
    tag: str = "asb",
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

    # fixed diagnostic material (never shuffled, never used for selection)
    diag_loader = _make_struct_loader(list(valid_data)[:128], 128, False, 0)
    diag_batches = [b.to(device_obj) for b in diag_loader]
    fixed_batch = _fixed_molecule_batch(valid_data).to(device_obj)
    support_examples: dict[str, Any] = {
        "initial": _support_examples(model, fixed_batch)
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    loader = _make_struct_loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = _make_struct_loader(
        valid_data,
        int(protocol["batch_size"]),
        False,
        int(seed) + int(protocol["eval_shuffle_seed_offset"]),
    )
    steps_per_epoch = int(math.ceil(len(train_data) / int(protocol["batch_size"])))
    patience = int(protocol["patience"])
    max_epochs = int(protocol["max_epochs"])
    mid_epoch = max(1, max_epochs // 2)

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
                last_grads = _asb_module_grad_norms(model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(protocol["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae = _evaluate_mae(model, eval_loader, device_obj)
        support_stats = _collect_support_stats(model.structural_encoder, diag_batches)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(train_mae),
                "valid_mae": float(valid_mae),
                "selected_node_fraction": float(
                    support_stats["selected_node_fraction"]
                ),
                "selected_edge_fraction": float(
                    support_stats["selected_edge_fraction"]
                ),
                "support_size_mean": float(support_stats["support_size_mean"]),
                "support_size_std": float(support_stats["support_size_std"]),
                "distance1_selection_rate": float(
                    support_stats["distance1_selection_rate"]
                ),
                "distance2_selection_rate": float(
                    support_stats["distance2_selection_rate"]
                ),
                "patches_with_nontrivial_support_fraction": float(
                    support_stats["patches_with_nontrivial_support_fraction"]
                ),
                "gate_probability_mean": float(
                    support_stats["gate_probability_mean"]
                ),
                "gate_probability_std": float(
                    support_stats["gate_probability_std"]
                ),
                "gate_entropy": float(support_stats["gate_entropy"]),
                "binding_state_norm_mean": float(
                    support_stats["binding_state_norm_mean"]
                ),
                "message_norm_mean": float(support_stats["message_norm_mean"]),
                "node_state_norm_mean": float(
                    support_stats["node_state_norm_mean"]
                ),
                "update_norm_mean": float(support_stats["update_norm_mean"]),
                "output_token_norm_mean": float(
                    support_stats["output_token_norm_mean"]
                ),
                "gate_grad_norm": float(last_grads.get("gate_mlp_grad_norm", 0.0)),
                "binding_grad_norm": float(
                    last_grads.get("bind_delta_mlp_grad_norm", 0.0)
                ),
                "message_grad_norm": float(
                    last_grads.get("message_mlp_grad_norm", 0.0)
                ),
                "update_grad_norm": float(
                    last_grads.get("update_mlp_grad_norm", 0.0)
                ),
                "bind_update_grad_norm": float(
                    last_grads.get("bind_update_grad_norm", 0.0)
                ),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": int(epoch * steps_per_epoch),
                "checkpoint_selected": 0,
            }
        )
        if epoch == 1:
            support_examples["early"] = _support_examples(model, fixed_batch)
        if epoch == mid_epoch:
            support_examples["middle"] = _support_examples(model, fixed_batch)
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
                f"support={support_stats['selected_node_fraction']:.4f} "
                f"gate={support_stats['gate_probability_mean']:.4f}",
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
    support_examples["best"] = _support_examples(model, fixed_batch)
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
    _write_json(
        RESULTS_DIR / f"support_examples_{tag}_seed{seed}.json",
        support_examples,
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
        "patch_representation": "adaptive_structure_binding",
        "asb_encoder_params": int(_n_params(model.structural_encoder)),
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


def soup_seed(seed: int, tag: str = "asb") -> dict[str, Any]:
    entries = torch.load(
        SOUP_DIR / f"{tag}_seed{seed}_top5_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    if len(entries) < 5:
        raise RuntimeError(f"need >= 5 snapshots for top-5 soup, got {len(entries)}")
    ranked = sorted(entries, key=lambda row: (float(row["valid_mae"]), int(row["epoch"])))[
        :5
    ]
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


def train_queue(seeds: Sequence[int], device: str, tag: str = "asb") -> None:
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


def diagnostics(tag: str = "asb") -> dict[str, Any]:
    device = torch.device("cpu")
    _train, valid_data, _audit = build_encoded_records()
    loader = _make_struct_loader(list(valid_data)[:256], 128, False, 0)
    batches = [b.to(device) for b in loader]

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": {},
        "patch_representation": "adaptive_structure_binding",
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        model = build_candidate(seed).to(device)
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        if not state_path.exists():
            continue
        model.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=True)
        )
        encoder = model.structural_encoder
        encoder.eval()
        support_stats = _collect_support_stats(encoder, batches)
        encoder_blocks: list[np.ndarray] = []
        with torch.no_grad():
            for batch in batches:
                encoder_blocks.append(encoder(batch).cpu().numpy())
        encoder_matrix = np.concatenate(encoder_blocks, axis=0)
        payload["per_seed"][str(seed)] = {
            "support": support_stats,
            "asb_encoder_rank": _effective_rank(encoder_matrix),
            "n_encoder_samples": int(encoder_matrix.shape[0]),
        }
    _write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


def support_audit(tag: str = "asb") -> dict[str, Any]:
    """Readable explicit learned supports at fixed epochs on fixed molecules."""
    device = torch.device("cpu")
    _train, valid_data, _audit = build_encoded_records()
    fixed_batch = _fixed_molecule_batch(valid_data).to(device)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "molecules": SUPPORT_EXAMPLE_MOLECULES,
        "per_seed": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        if not state_path.exists():
            continue
        model = build_candidate(seed).to(device)
        entry: dict[str, Any] = {
            "initial": _support_examples(model, fixed_batch),
        }
        model.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=True)
        )
        entry["best"] = _support_examples(model, fixed_batch)
        existing = RESULTS_DIR / f"support_examples_{tag}_seed{seed}.json"
        if existing.exists():
            data = _read_json(existing)
            for key in ("early", "middle"):
                if key in data:
                    entry[key] = data[key]
        payload["per_seed"][str(seed)] = entry
    _write_json(RESULTS_DIR / "support_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(tag: str = "asb") -> dict[str, Any]:
    rows = {}
    support_by_seed: dict[str, Any] = {}
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
    if diag_path.exists():
        diag = _read_json(diag_path)
        for seed, value in diag.get("per_seed", {}).items():
            support_by_seed[str(seed)] = value.get("support", {})

    if len(rows) < 2:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "INCOMPLETE",
            "per_seed": rows,
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
    same_direction_bag = all(v < 0 for v in delta_bag_per_seed.values()) or all(
        v > 0 for v in delta_bag_per_seed.values()
    )

    # mechanism liveness from the selection checkpoints
    fractions = {
        str(s): float(support_by_seed[str(s)]["selected_node_fraction"])
        for s in SEEDS
        if str(s) in support_by_seed
    }
    gate_stds = {
        str(s): float(support_by_seed[str(s)]["gate_probability_std"])
        for s in SEEDS
        if str(s) in support_by_seed
    }
    message_norms = {
        str(s): float(support_by_seed[str(s)]["message_norm_mean"])
        for s in SEEDS
        if str(s) in support_by_seed
    }
    binding_norms = {
        str(s): float(support_by_seed[str(s)]["binding_state_norm_mean"])
        for s in SEEDS
        if str(s) in support_by_seed
    }
    ranks = {
        str(s): float(diag.get("per_seed", {}).get(str(s), {}).get(
            "asb_encoder_rank", {}
        ).get("effective_rank", 1.0))
        for s in SEEDS
    } if diag_path.exists() else {}
    per_seed_nontrivial = {
        s: fractions[s] < 1.0 - NONTRIVIAL_FRACTION_MARGIN for s in fractions
    }
    any_nontrivial = any(per_seed_nontrivial.values())
    all_full = bool(fractions) and all(
        f >= 1.0 - NONTRIVIAL_FRACTION_MARGIN for f in fractions.values()
    )
    # ``branch_dead``: the binding / message branch carries no signal at the
    # selected checkpoint (exactly-zero message or a constant encoder output).
    message_dead = {
        s: message_norms.get(s, 0.0) < 1.0e-6 for s in message_norms
    }
    encoder_collapsed = {
        s: ranks.get(s, 1.0) < 0.5 for s in ranks
    }
    gate_input_independent = {
        s: gate_stds.get(s, 1.0) < 1.0e-4 for s in gate_stds
    }
    branch_dead = bool(message_dead) and all(message_dead.values())
    collapse = branch_dead or (
        bool(encoder_collapsed) and all(encoder_collapsed.values())
    )

    if collapse:
        case = "E_branch_or_structure_collapse"
    elif all_full and all(
        support_by_seed[s]["gate_probability_mean"] > 0.999
        for s in support_by_seed
    ):
        case = "D_gates_always_on"
    elif delta_bag <= -STRONG_GATE and same_direction_bag and any_nontrivial:
        case = "A_go_to_z2"
    elif abs(delta_bag) < STRONG_GATE:
        case = "B_neutral_signal"
    elif delta_bag >= REGRESSION_GATE and same_direction_bag:
        case = "C_regression_but_active"
    else:
        case = "E_inconclusive"

    non_trivial = any_nontrivial
    gates_collapsed_open = all_full and all(
        support_by_seed[s]["gate_probability_mean"] > 0.999 for s in support_by_seed
    )
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
        "same_direction_vs_bbag": bool(same_direction_bag),
        "support_selected_fraction_per_seed": fractions,
        "gate_probability_mean_per_seed": {
            str(s): support_by_seed.get(str(s), {}).get("gate_probability_mean")
            for s in SEEDS
        },
        "gate_probability_std_per_seed": gate_stds,
        "message_norm_mean_per_seed": message_norms,
        "binding_state_norm_mean_per_seed": binding_norms,
        "asb_encoder_effective_rank_per_seed": ranks,
        "support_nontrivial_per_seed": per_seed_nontrivial,
        "support_nontrivial": bool(non_trivial),
        "gates_collapsed_open": bool(gates_collapsed_open),
        "binding_message_branch_dead": bool(branch_dead),
        "asb_encoder_collapsed": bool(collapse),
        "gate_input_independent": gate_input_independent,
        "case": case,
        "candidate_params": int(
            _read_json(RESULTS_DIR / "parameter_accounting.json")["candidate_params"]
        ),
        "vocab_independent": True,
        "dataset_dependent_params_zero": True,
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
        "params": _maybe("parameter_accounting.json"),
        "sanity": _maybe("sanity.json"),
        "diagnostics": _maybe("diagnostics.json"),
        "decision": _maybe("decision.json"),
        "support_audit": _maybe("support_audit.json"),
        "soup": {str(seed): _maybe(f"soup_asb_seed{seed}.json") for seed in SEEDS},
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
            "support_audit",
            "decide",
            "report",
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="asb")
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
    elif args.stage == "support_audit":
        print(support_audit(tag=args.tag), flush=True)
    elif args.stage == "decide":
        print(decide(tag=args.tag), flush=True)
    elif args.stage == "report":
        print(report(), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
