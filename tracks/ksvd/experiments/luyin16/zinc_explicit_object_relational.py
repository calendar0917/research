"""ZINC explicit objects + explicit relations + T_obj=2 recurrent reasoning.

Question
--------
B-full replaces the historical typed certificate lookup with a shared,
connectivity-aware two-round message-passing encoder over the real rooted
typed radius-2 patch graph (``patch_representation="shared_structural"``).
Its 16-D ``e_struct`` is effectively rank-1.  The previous experiment
(``explicit_basis_rank1``) enumerated the exact B0/B1/B2 basis but pooled
immediately and only matched the connectivity-free B-bag: fixed low-order
support-local valuation cannot carry connectivity.

This experiment tests exactly one pre-registered hypothesis:

    The objects are explicit.  Their relations are explicit.  Neural
    computation only refines states over those explicit relations.

It is *not* hidden message passing: object existence and the sparse
support-derived relation graph are fixed before any parameter is evaluated.
The honest classification is *neural relational reasoning over explicitly
enumerated, support-traceable structural objects and relations*.

Exactly one pre-registered architecture::

    B0/B1/B2 explicit objects            (unchanged from the basis experiment)
    relation graph = overlap / bond-overlap / adjacent   (support-derived, sparse)
    h_i^0 = E_{order(i)}(explicit object)                in R^8
    q_ij^t = Q(h_i^t, h_j^t, r_ij)                       in R^8
    A_i^t = fixed per-family invariant stats of q
    h_i^{t+1} = h_i^t + U([h_i^t, A_i^t])                (U tied across rounds)
    T_obj = 2
    s(P) = fusion(late per-order pooling of h^2)
    e_struct(P) = b + s(P) * v                           (rank-1 by construction)

No attention / softmax / Transformer / GRU / LSTM / learned support or
existence gate / certificate or token lookup / dense inner graph.  Every
thing downstream is inherited bit-exactly from cell A / B-full.  Official
ZINC test is **never** loaded.

Stages: ``params preprocess sanity train soup train_queue repro diagnostics
mechanism profile decide report``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_explicit_object_relational <stage>
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import inspect
import json
import math
import pickle
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_pair_centre as rec
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as sspe
from tracks.ksvd.experiments.luyin16 import zinc_structural_encoder_rank1_audit as r1a
from tracks.ksvd.experiments.luyin16.explicit_structural_basis import (
    ExplicitBasisGraph,
    ExplicitStructuralBasisEncoder,
    atom_object_support,
    bond_object_bond_support,
    bond_object_support,
    build_explicit_basis_graph,
    expected_triple_count,
    object_count,
    triple_object_bond_support,
    triple_object_center,
    triple_object_support,
)
from tracks.ksvd.experiments.luyin16.explicit_object_relational import (
    FAMILY_ADJACENT,
    FAMILY_BOND_OVERLAP,
    FAMILY_NAMES,
    FAMILY_OVERLAP,
    N_FAMILIES,
    OBJ_DIM,
    Q_OBJ_DIM,
    REL_FEATURE_DIM,
    T_OBJ,
    ExplicitObjectRelationalEncoder,
    ObjectRelationGraph,
    build_object_relation_graph,
    relation_counts,
)

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/explicit_object_relational"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"
CACHE_DIR = RESULTS_DIR / "cache"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "explicit_object_relational_v1"
CACHE_SCHEMA_VERSION = "explicit_object_relational_v1"

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

# --- pre-registered explicit-basis valuation (one architecture; no sweep) ----
# Widths were chosen ONCE by parameter accounting so the encoder budget matches
# B-full's 35,152-param structural encoder (see ``parameter_accounting.json``).
# They were not selected on validation.
ESB_NODE_DIM = 20
ESB_EDGE_DIM = 16
ESB_N_DEGREE_BINS = 8
ESB_VALUATION_HIDDEN = 184
ESB_FUSION_HIDDEN = 184

# --- pre-registered explicit-object relational architecture (no sweep) -------
# object_hidden=100, update_hidden=79, fusion_hidden=136 gives exactly 35,152
# encoder parameters == B-full's encoder (total 84,495 == B-full), chosen once
# by accounting only (no validation sweep).
EOR_OBJ_DIM = OBJ_DIM
EOR_REL_HIDDEN = 64
EOR_REL_LATENT_DIM = 16
EOR_Q_HIDDEN = 64
EOR_Q_OBJ_DIM = Q_OBJ_DIM
EOR_OBJECT_HIDDEN = 100
EOR_UPDATE_HIDDEN = 79
EOR_FUSION_HIDDEN = 136
EOR_ROUNDS = T_OBJ

# --- references (frozen) -----------------------------------------------------
REFERENCE_A2_SOUP_VALID = 0.12191404939390486
REFERENCE_A2_SOUP_PER_SEED = {
    0: 0.12169362585240742,
    1: 0.1221344729354023,
}
REFERENCE_BFULL_SOUP_VALID = 0.11897220489243046
REFERENCE_BFULL_SOUP_PER_SEED = {
    0: 0.11981802638241788,
    1: 0.11812638340244302,
}
REFERENCE_BFULL_RAW_PER_SEED = {0: 0.125324, 1: 0.124394}
REFERENCE_BFULL_PARAMS = 84495
REFERENCE_BFULL_ENCODER_PARAMS = 35152
REFERENCE_BBAG_SOUP_PER_SEED = {1: 0.12022926583880325}
# previous explicit-basis early-pooling experiment (seed1 Top-5 soup)
REFERENCE_EXPLICIT_BASIS_SOUP_PER_SEED = {1: 0.12019429576670518}
REFERENCE_EXPLICIT_BASIS_SOUP_VALID = 0.12019429576670518

PARAM_LOWER = 80000
PARAM_UPPER = 90000
BUDGET_MATCH_TOLERANCE = 0.01

STRONG_GATE = 0.002
METHOD_GATE = 0.001
REGRESSION_GATE = 0.003

# one promotion seed only (pre-registered)
PROMOTION_SEED = 1
SEEDS = (PROMOTION_SEED,)

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
# basis preprocessing / cache
# ---------------------------------------------------------------------------


def _esb_cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "explicit_basis_train.pkl.gz",
        CACHE_DIR / "explicit_basis_valid.pkl.gz",
        CACHE_DIR / "explicit_basis_metadata.json",
    )


def _esb_to_plain(graph: ExplicitBasisGraph) -> dict[str, Any]:
    return {
        "patch": graph.patch,
        "degree": graph.degree,
        "l1_u": graph.l1_u,
        "l1_v": graph.l1_v,
        "l1_bond": graph.l1_bond,
        "l1_patch": graph.l1_patch,
        "b2_center": graph.b2_center,
        "b2_u": graph.b2_u,
        "b2_w": graph.b2_w,
        "b2_l1_cu": graph.b2_l1_cu,
        "b2_l1_cw": graph.b2_l1_cw,
        "b2_l1_closure": graph.b2_l1_closure,
        "b2_bond_cu": graph.b2_bond_cu,
        "b2_bond_cw": graph.b2_bond_cw,
        "b2_closure": graph.b2_closure,
        "b2_patch": graph.b2_patch,
        "n_atoms": int(graph.n_atoms),
        "n_bonds": int(graph.n_bonds),
        "n_triples": int(graph.n_triples),
        "n_patches": int(graph.n_patches),
    }


def _esb_from_plain(item: Mapping[str, Any]) -> ExplicitBasisGraph:
    return ExplicitBasisGraph(**dict(item))


def _basis_stats(graphs: Sequence[ExplicitBasisGraph]) -> dict[str, Any]:
    n_atoms = np.asarray([g.n_atoms for g in graphs], dtype=np.float64)
    n_bonds = np.asarray([g.n_bonds for g in graphs], dtype=np.float64)
    n_triples = np.asarray([g.n_triples for g in graphs], dtype=np.float64)
    n_patches = np.asarray([g.n_patches for g in graphs], dtype=np.float64)
    safe = np.clip(n_patches, 1, None)
    return {
        "n_molecules": int(len(graphs)),
        "total_atoms": int(n_atoms.sum()),
        "total_bonds": int(n_bonds.sum()),
        "total_triples": int(n_triples.sum()),
        "total_patches": int(n_patches.sum()),
        "mean_atoms_per_patch": float(n_atoms.sum() / safe.sum()),
        "mean_bonds_per_patch": float(n_bonds.sum() / safe.sum()),
        "mean_triples_per_patch": float(n_triples.sum() / safe.sum()),
        "max_triples_per_molecule": int(n_triples.max()) if n_triples.size else 0,
        "mean_objects_per_patch": float(
            (n_atoms.sum() + n_bonds.sum() + n_triples.sum()) / safe.sum()
        ),
    }


def explicit_basis() -> tuple[
    list[ExplicitBasisGraph], list[ExplicitBasisGraph], dict[str, Any]
]:
    """Build (and cache) the explicit B0/B1/B2 basis for train and valid."""
    train_path, valid_path, meta_path = _esb_cache_paths()
    if train_path.exists() and valid_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
            with gzip.open(train_path, "rb") as handle:
                train_items = pickle.load(handle)
            with gzip.open(valid_path, "rb") as handle:
                valid_items = pickle.load(handle)
            return (
                [_esb_from_plain(item) for item in train_items],
                [_esb_from_plain(item) for item in valid_items],
                meta,
            )

    started = time.perf_counter()
    train_bundle, valid_bundle, _base_meta = sspe.extract_records()
    train_basis = [
        build_explicit_basis_graph(
            graph.patch,
            graph.src,
            graph.dst,
            graph.bond,
            graph.edge_patch,
            graph.n_patches,
        )
        for graph in train_bundle["graphs"]
    ]
    valid_basis = [
        build_explicit_basis_graph(
            graph.patch,
            graph.src,
            graph.dst,
            graph.bond,
            graph.edge_patch,
            graph.n_patches,
        )
        for graph in valid_bundle["graphs"]
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(
            [_esb_to_plain(g) for g in train_basis],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(
            [_esb_to_plain(g) for g in valid_basis],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "patch_radius": PATCH_RADIUS,
        "orders": [0, 1, 2],
        "n_train_molecules": int(len(train_basis)),
        "n_valid_molecules": int(len(valid_basis)),
        "train": _basis_stats(train_basis),
        "valid": _basis_stats(valid_basis),
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    _write_json(meta_path, meta)
    print(
        f"[basis] train={len(train_basis)} valid={len(valid_basis)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_basis, valid_basis, meta


def _eor_cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "object_relation_train.pkl.gz",
        CACHE_DIR / "object_relation_valid.pkl.gz",
        CACHE_DIR / "object_relation_metadata.json",
    )


def _relation_to_plain(relation: ObjectRelationGraph) -> dict[str, Any]:
    return {
        "edge_i": relation.edge_i,
        "edge_j": relation.edge_j,
        "family": relation.family,
        "feature": relation.feature,
        "n_atoms": int(relation.n_atoms),
        "n_bonds": int(relation.n_bonds),
        "n_triples": int(relation.n_triples),
        "n_objects": int(relation.n_objects),
    }


def _relation_from_plain(item: Mapping[str, Any]) -> ObjectRelationGraph:
    return ObjectRelationGraph(**dict(item))


def _relation_stats(relations: Sequence[ObjectRelationGraph]) -> dict[str, Any]:
    n_objects = np.asarray([r.n_objects for r in relations], dtype=np.float64)
    n_rel = np.asarray([r.edge_i.shape[0] for r in relations], dtype=np.float64)
    ratio = np.divide(n_rel, n_objects, out=np.zeros_like(n_rel), where=n_objects > 0)
    families = np.zeros(N_FAMILIES, dtype=np.float64)
    for relation in relations:
        for fam in range(N_FAMILIES):
            families[fam] += float((relation.family == fam).sum())
    return {
        "n_molecules": int(len(relations)),
        "total_objects": float(n_objects.sum()),
        "total_relations": float(n_rel.sum()),
        "mean_relations_per_molecule": float(n_rel.mean()) if n_rel.size else 0.0,
        "p95_relations_per_molecule": float(np.percentile(n_rel, 95)) if n_rel.size else 0.0,
        "max_relations_per_molecule": float(n_rel.max()) if n_rel.size else 0.0,
        "mean_edge_object_ratio": float(ratio.mean()) if ratio.size else 0.0,
        "p95_edge_object_ratio": float(np.percentile(ratio, 95)) if ratio.size else 0.0,
        "max_edge_object_ratio": float(ratio.max()) if ratio.size else 0.0,
        "family_totals": {
            FAMILY_NAMES[fam]: float(families[fam]) for fam in range(N_FAMILIES)
        },
    }


def object_relation(
    encoded: Sequence[Data], basis: Sequence[ExplicitBasisGraph]
) -> tuple[list[ObjectRelationGraph], dict[str, Any]]:
    """Build (and cache) the explicit relation graph for one split.

    Relations depend only on the deterministic basis and the patch root, so a
    single cache is shared by train and valid.  Root atoms are read from the
    encoded records (target-free).
    """
    relations: list[ObjectRelationGraph] = []
    for data, graph in zip(encoded, basis):
        root = np.asarray(data.struct_root.numpy(), dtype=np.int64)
        relations.append(build_object_relation_graph(graph, root_atoms=root))
    return relations, _relation_stats(relations)


def explicit_relations() -> tuple[
    list[ObjectRelationGraph], list[ObjectRelationGraph], dict[str, Any]
]:
    """Build (and cache) the explicit relation graph for train and valid."""
    train_path, valid_path, meta_path = _eor_cache_paths()
    if train_path.exists() and valid_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
            with gzip.open(train_path, "rb") as handle:
                train_items = pickle.load(handle)
            with gzip.open(valid_path, "rb") as handle:
                valid_items = pickle.load(handle)
            return (
                [_relation_from_plain(item) for item in train_items],
                [_relation_from_plain(item) for item in valid_items],
                meta,
            )
    train_bundle, valid_bundle, _base_meta = sspe.extract_records()
    train_basis, valid_basis, _bmeta = explicit_basis()
    config = sspe._v4_base_config()
    config["model"]["device"] = "cpu"
    enc_train, enc_valid, _audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    sspe._attach_struct_tensors(enc_train, train_bundle["graphs"])
    sspe._attach_struct_tensors(enc_valid, valid_bundle["graphs"])
    started = time.perf_counter()
    train_rel, train_stats = object_relation(enc_train, train_basis)
    valid_rel, valid_stats = object_relation(enc_valid, valid_basis)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(
            [_relation_to_plain(r) for r in train_rel],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(
            [_relation_to_plain(r) for r in valid_rel],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "patch_radius": PATCH_RADIUS,
        "relation_feature_dim": REL_FEATURE_DIM,
        "relation_families": list(FAMILY_NAMES),
        "n_train_molecules": int(len(train_rel)),
        "n_valid_molecules": int(len(valid_rel)),
        "train": train_stats,
        "valid": valid_stats,
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    _write_json(meta_path, meta)
    print(
        f"[relations] train={len(train_rel)} valid={len(valid_rel)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_rel, valid_rel, meta


def _attach_basis_tensors(
    encoded: Sequence[Data], basis: Sequence[ExplicitBasisGraph]
) -> None:
    """Attach the precomputed explicit basis tensors to the encoded ``Data``."""
    for data, graph in zip(encoded, basis):
        if int(data.num_nodes) != graph.n_patches:
            raise RuntimeError("encoded / basis patch count mismatch")
        if int(data.struct_atom.numel()) != graph.n_atoms:
            raise RuntimeError("encoded / basis atom count mismatch")
        data.esb_degree = torch.from_numpy(graph.degree)
        data.esb_l1_u = torch.from_numpy(graph.l1_u)
        data.esb_l1_v = torch.from_numpy(graph.l1_v)
        data.esb_l1_bond = torch.from_numpy(graph.l1_bond)
        data.esb_l1_patch = torch.from_numpy(graph.l1_patch)
        data.esb_b2_center = torch.from_numpy(graph.b2_center)
        data.esb_b2_u = torch.from_numpy(graph.b2_u)
        data.esb_b2_w = torch.from_numpy(graph.b2_w)
        data.esb_b2_l1_cu = torch.from_numpy(graph.b2_l1_cu)
        data.esb_b2_l1_cw = torch.from_numpy(graph.b2_l1_cw)
        data.esb_b2_l1_closure = torch.from_numpy(graph.b2_l1_closure)
        data.esb_b2_bond_cu = torch.from_numpy(graph.b2_bond_cu)
        data.esb_b2_bond_cw = torch.from_numpy(graph.b2_bond_cw)
        data.esb_b2_closure = torch.from_numpy(graph.b2_closure)
        data.esb_b2_patch = torch.from_numpy(graph.b2_patch)


def _attach_relation_tensors(
    encoded: Sequence[Data], relations: Sequence[ObjectRelationGraph]
) -> None:
    """Attach the precomputed explicit relation tensors to the encoded ``Data``."""
    for data, relation in zip(encoded, relations):
        if int(data.struct_atom.numel()) != relation.n_atoms:
            raise RuntimeError("encoded / relation atom count mismatch")
        if int(data.esb_l1_u.numel()) != relation.n_bonds:
            raise RuntimeError("encoded / relation bond count mismatch")
        if int(data.esb_b2_center.numel()) != relation.n_triples:
            raise RuntimeError("encoded / relation triple count mismatch")
        data.eor_edge_i = torch.from_numpy(relation.edge_i)
        data.eor_edge_j = torch.from_numpy(relation.edge_j)
        data.eor_family = torch.from_numpy(relation.family)
        data.eor_feat = torch.from_numpy(relation.feature)


def build_encoded_records():
    """Return (train_data, valid_data, audit) with basis + relations attached."""
    train_bundle, valid_bundle, _meta = sspe.extract_records()
    config = sspe._v4_base_config()
    config["model"]["device"] = "cpu"
    enc_train, enc_valid, audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    sspe._attach_struct_tensors(enc_train, train_bundle["graphs"])
    sspe._attach_struct_tensors(enc_valid, valid_bundle["graphs"])
    train_basis, valid_basis, _bmeta = explicit_basis()
    _attach_basis_tensors(enc_train, train_basis)
    _attach_basis_tensors(enc_valid, valid_basis)
    train_rel, valid_rel, _rmeta = explicit_relations()
    _attach_relation_tensors(enc_train, train_rel)
    _attach_relation_tensors(enc_valid, valid_rel)
    audit = dict(audit)
    audit["patch_representation"] = "explicit_object_relational"
    audit["explicit_object_relational"] = {
        "obj_dim": EOR_OBJ_DIM,
        "rel_feature_dim": REL_FEATURE_DIM,
        "rel_hidden": EOR_REL_HIDDEN,
        "rel_latent_dim": EOR_REL_LATENT_DIM,
        "q_hidden": EOR_Q_HIDDEN,
        "q_obj_dim": EOR_Q_OBJ_DIM,
        "object_hidden": EOR_OBJECT_HIDDEN,
        "update_hidden": EOR_UPDATE_HIDDEN,
        "fusion_hidden": EOR_FUSION_HIDDEN,
        "t_obj": EOR_ROUNDS,
        "output_dim": TOKEN_WIDTH,
        "relation_families": list(FAMILY_NAMES),
        "channel": "b + s(P) * v (rank-1 by construction)",
    }
    audit["official_test_loaded"] = False
    return enc_train, enc_valid, audit


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_candidate(seed: int = 0) -> torch.nn.Module:
    """Cell-A geometry with the typed lookup replaced by explicit object relations.

    Construction order mirrors ``sspe.build_candidate`` exactly so every shared
    tensor is copied bit-exactly and the small head is re-drawn from the
    historical stream; only ``patch_representation`` differs.
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
            patch_representation="explicit_object_relational",
            esb_n_degree_bins=ESB_N_DEGREE_BINS,
            eor_obj_dim=EOR_OBJ_DIM,
            eor_rel_hidden=EOR_REL_HIDDEN,
            eor_rel_latent_dim=EOR_REL_LATENT_DIM,
            eor_q_hidden=EOR_Q_HIDDEN,
            eor_q_obj_dim=EOR_Q_OBJ_DIM,
            eor_object_hidden=EOR_OBJECT_HIDDEN,
            eor_update_hidden=EOR_UPDATE_HIDDEN,
            eor_fusion_hidden=EOR_FUSION_HIDDEN,
            eor_rounds=EOR_ROUNDS,
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
# collate / loaders
# ---------------------------------------------------------------------------


def esb_collate(data_list: Sequence[Data]) -> Any:
    """Batch the explicit basis tensors with the correct graph offsets.

    B0/B1/B2 atom ids are offsets into the molecule-local atom block, patch ids
    are offsets into the molecule-local patch block.  ``struct_src`` /
    ``struct_dst`` are offset by ``sspe.struct_collate`` but the encoder never
    reads them.
    """
    batch = sspe.struct_collate(data_list)
    n_offset = 0
    patch_offset = 0
    l1_u: list[torch.Tensor] = []
    l1_v: list[torch.Tensor] = []
    l1_bond: list[torch.Tensor] = []
    l1_patch: list[torch.Tensor] = []
    b2_center: list[torch.Tensor] = []
    b2_u: list[torch.Tensor] = []
    b2_w: list[torch.Tensor] = []
    b2_bond_cu: list[torch.Tensor] = []
    b2_bond_cw: list[torch.Tensor] = []
    b2_closure: list[torch.Tensor] = []
    b2_patch: list[torch.Tensor] = []
    eor_edge_i: list[torch.Tensor] = []
    eor_edge_j: list[torch.Tensor] = []
    eor_family: list[torch.Tensor] = []
    eor_feat: list[torch.Tensor] = []
    object_offset = 0
    for data in data_list:
        n_patches = int(data.num_nodes)
        n_atoms = int(data.struct_atom.numel())
        l1_u.append(data.esb_l1_u + n_offset)
        l1_v.append(data.esb_l1_v + n_offset)
        l1_bond.append(data.esb_l1_bond)
        l1_patch.append(data.esb_l1_patch + patch_offset)
        b2_center.append(data.esb_b2_center + n_offset)
        b2_u.append(data.esb_b2_u + n_offset)
        b2_w.append(data.esb_b2_w + n_offset)
        b2_bond_cu.append(data.esb_b2_bond_cu)
        b2_bond_cw.append(data.esb_b2_bond_cw)
        b2_closure.append(data.esb_b2_closure)
        b2_patch.append(data.esb_b2_patch + patch_offset)
        n_objects = (
            n_atoms
            + int(data.esb_l1_u.numel())
            + int(data.esb_b2_center.numel())
        )
        eor_edge_i.append(data.eor_edge_i + object_offset)
        eor_edge_j.append(data.eor_edge_j + object_offset)
        eor_family.append(data.eor_family)
        eor_feat.append(data.eor_feat)
        n_offset += n_atoms
        patch_offset += n_patches
        object_offset += n_objects
    # B1 identity ids are not consumed by the encoder; the B2 support /
    # provenance diagnostics run per molecule on the precomputed tables.
    batch.esb_l1_u = torch.cat(l1_u) if l1_u else torch.zeros(0, dtype=torch.long)
    batch.esb_l1_v = torch.cat(l1_v) if l1_v else torch.zeros(0, dtype=torch.long)
    batch.esb_l1_bond = (
        torch.cat(l1_bond) if l1_bond else torch.zeros(0, dtype=torch.long)
    )
    batch.esb_l1_patch = (
        torch.cat(l1_patch) if l1_patch else torch.zeros(0, dtype=torch.long)
    )
    cat = lambda parts, width=0: (  # noqa: E731
        torch.cat(parts) if parts else torch.zeros((0, width), dtype=torch.long)
    )
    batch.esb_b2_center = cat(b2_center)
    batch.esb_b2_u = cat(b2_u)
    batch.esb_b2_w = cat(b2_w)
    batch.esb_b2_bond_cu = cat(b2_bond_cu)
    batch.esb_b2_bond_cw = cat(b2_bond_cw)
    batch.esb_b2_closure = cat(b2_closure)
    batch.esb_b2_patch = cat(b2_patch)
    batch.eor_edge_i = cat(eor_edge_i)
    batch.eor_edge_j = cat(eor_edge_j)
    batch.eor_family = cat(eor_family)
    batch.eor_feat = (
        torch.cat(eor_feat)
        if eor_feat
        else torch.zeros((0, REL_FEATURE_DIM), dtype=torch.float32)
    )
    return batch


def _make_esb_loader(
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
        collate_fn=esb_collate,
    )


def _selection_loader(valid_data: Sequence[Data]):
    return _make_esb_loader(valid_data, 128, False, 0)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    baseline = cd.build_cell(CELL, 0)
    reference = sspe.build_candidate(0)
    candidate = build_candidate(0)
    base_total = _n_params(baseline)
    ref_total = _n_params(reference)
    cand_total = _n_params(candidate)
    if base_total != sspe.REFERENCE_PARAMS:
        raise RuntimeError(
            f"cell-A baseline params changed: {base_total} != {sspe.REFERENCE_PARAMS}"
        )
    if ref_total != REFERENCE_BFULL_PARAMS:
        raise RuntimeError(
            f"B-full params changed: {ref_total} != {REFERENCE_BFULL_PARAMS}"
        )
    encoder = candidate.structural_encoder
    encoder_params = _n_params(encoder)
    relative = abs(cand_total - REFERENCE_BFULL_PARAMS) / REFERENCE_BFULL_PARAMS
    audit = zpp.audit_parameters(candidate)
    typed_keys = [key for key in candidate.state_dict() if "typed_embedding" in key]

    encoder_blocks = {
        "primitive_embeddings": int(
            _n_params(encoder.atom_embedding)
            + _n_params(encoder.root_embedding)
            + _n_params(encoder.distance_embedding)
            + _n_params(encoder.degree_embedding)
            + _n_params(encoder.boundary_embedding)
            + _n_params(encoder.bond_embedding)
        ),
        "object_encoders_E0_E1_E2": int(
            _n_params(encoder.order0_encoder)
            + _n_params(encoder.order1_encoder)
            + _n_params(encoder.order2_encoder)
        ),
        "inner_relation_encoder": int(_n_params(encoder.relation_encoder)),
        "inner_q_encoder": int(_n_params(encoder.q_encoder)),
        "inner_update_U": int(_n_params(encoder.update)),
        "late_fusion": int(_n_params(encoder.fusion)),
        "rank1_projection": int(
            encoder.rank1_bias.numel() + encoder.rank1_direction.numel()
        ),
    }
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
        "encoder_params": int(encoder_params),
        "encoder_params_minus_bfull_encoder": int(
            encoder_params - REFERENCE_BFULL_ENCODER_PARAMS
        ),
        "encoder_blocks": encoder_blocks,
        "encoder_dimensions": {
            "obj_dim": int(encoder.obj_dim),
            "n_degree_bins": int(encoder.n_degree_bins),
            "rel_feature_dim": int(REL_FEATURE_DIM),
            "rel_hidden": int(encoder.rel_hidden),
            "rel_latent_dim": int(encoder.rel_latent_dim),
            "q_hidden": int(encoder.q_hidden),
            "q_obj_dim": int(encoder.q_obj_dim),
            "object_hidden": int(encoder.object_hidden),
            "update_hidden": int(encoder.update_hidden),
            "fusion_hidden": int(encoder.fusion_hidden),
            "aggregate_dim": int(encoder.aggregate_dim),
            "triple_input": int(encoder.triple_input),
            "t_obj": int(encoder.t_obj),
            "output_dim": int(encoder.output_dim),
        },
        "released_typed_lookup_params": int(
            _n_params(baseline.typed_embedding)
            if baseline.typed_embedding is not None
            else 0
        ),
        "candidate_has_typed_embedding": bool(
            getattr(candidate, "typed_embedding", None) is not None
        ),
        "candidate_typed_state_keys": typed_keys,
        "candidate_block_params": audit["blocks"],
        "candidate_dimensions": audit["dimensions"],
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
# integrity / adversarial sanity
# ---------------------------------------------------------------------------


class _BasisBatch:
    """Minimal explicit-basis container for unit / adversarial checks."""

    def __init__(
        self,
        atom,
        root,
        dist,
        degree,
        patch,
        l1_u,
        l1_v,
        l1_bond,
        l1_patch,
        b2_center,
        b2_u,
        b2_w,
        b2_bond_cu,
        b2_bond_cw,
        b2_closure,
        b2_patch,
        n_patches: int,
        eor_edge_i=None,
        eor_edge_j=None,
        eor_family=None,
        eor_feat=None,
    ) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.esb_degree = degree
        self.struct_patch = patch
        self.esb_l1_u = l1_u
        self.esb_l1_v = l1_v
        self.esb_l1_bond = l1_bond
        self.esb_l1_patch = l1_patch
        self.esb_b2_center = b2_center
        self.esb_b2_u = b2_u
        self.esb_b2_w = b2_w
        self.esb_b2_bond_cu = b2_bond_cu
        self.esb_b2_bond_cw = b2_bond_cw
        self.esb_b2_closure = b2_closure
        self.esb_b2_patch = b2_patch
        self.struct_n_patches = int(n_patches)
        self.eor_edge_i = eor_edge_i
        self.eor_edge_j = eor_edge_j
        self.eor_family = eor_family
        self.eor_feat = eor_feat


def basis_batch_from_graph(
    graph: ExplicitBasisGraph,
    *,
    atom: np.ndarray | None = None,
    root: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> _BasisBatch:
    n = int(graph.n_atoms)
    if atom is None:
        atom = np.zeros(n, dtype=np.int64)
    if root is None:
        root = np.zeros(n, dtype=np.int64)
        if n:
            root[0] = 1
    if dist is None:
        dist = np.zeros(n, dtype=np.int64)
    relation = build_object_relation_graph(
        graph, root_atoms=np.asarray(root, dtype=np.int64)
    )
    to_t = lambda array: torch.from_numpy(np.asarray(array, dtype=np.int64))  # noqa: E731
    return _BasisBatch(
        atom=to_t(atom),
        root=to_t(root),
        dist=to_t(dist),
        degree=torch.from_numpy(graph.degree),
        patch=torch.from_numpy(graph.patch),
        l1_u=torch.from_numpy(graph.l1_u),
        l1_v=torch.from_numpy(graph.l1_v),
        l1_bond=torch.from_numpy(graph.l1_bond),
        l1_patch=torch.from_numpy(graph.l1_patch),
        b2_center=torch.from_numpy(graph.b2_center),
        b2_u=torch.from_numpy(graph.b2_u),
        b2_w=torch.from_numpy(graph.b2_w),
        b2_bond_cu=torch.from_numpy(graph.b2_bond_cu),
        b2_bond_cw=torch.from_numpy(graph.b2_bond_cw),
        b2_closure=torch.from_numpy(graph.b2_closure),
        b2_patch=torch.from_numpy(graph.b2_patch),
        n_patches=graph.n_patches,
        eor_edge_i=torch.from_numpy(relation.edge_i),
        eor_edge_j=torch.from_numpy(relation.edge_j),
        eor_family=torch.from_numpy(relation.family),
        eor_feat=torch.from_numpy(relation.feature),
    )


def _hand_graph(
    undirected: Sequence[tuple[int, int]],
    n_atoms: int,
    *,
    bonds: Mapping[tuple[int, int], int] | None = None,
) -> ExplicitBasisGraph:
    both = list(undirected) + [(v, u) for u, v in undirected]
    src = np.asarray([u for u, _ in both], dtype=np.int64)
    dst = np.asarray([v for _, v in both], dtype=np.int64)
    if bonds is None:
        bond = np.zeros(len(both), dtype=np.int64)
    else:
        bond = np.asarray(
            [bonds[(min(u, v), max(u, v))] for u, v in both], dtype=np.int64
        )
    patch = np.zeros(int(n_atoms), dtype=np.int64)
    edge_patch = np.zeros(len(both), dtype=np.int64)
    return build_explicit_basis_graph(patch, src, dst, bond, edge_patch, 1)


def _breadth_first_distances(
    undirected: Sequence[tuple[int, int]], n_atoms: int, root: int
) -> np.ndarray:
    adjacency: dict[int, set[int]] = {node: set() for node in range(int(n_atoms))}
    for u, v in undirected:
        adjacency[u].add(v)
        adjacency[v].add(u)
    distances = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbour in sorted(adjacency[node]):
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return np.asarray(
        [distances.get(node, 0) for node in range(int(n_atoms))], dtype=np.int64
    )


def _basis_signature(graph: ExplicitBasisGraph) -> tuple:
    return (
        tuple(
            sorted(
                zip(
                    graph.b2_center.tolist(),
                    graph.b2_u.tolist(),
                    graph.b2_w.tolist(),
                    graph.b2_bond_cu.tolist(),
                    graph.b2_bond_cw.tolist(),
                    graph.b2_closure.tolist(),
                )
            )
        ),
    )


def _relation_signature(relation: ObjectRelationGraph) -> tuple:
    """Deterministic signature of the relation graph (topology + descriptors)."""
    rows = sorted(
        (
            int(i),
            int(j),
            int(fam),
            tuple(np.round(relation.feature[index], 4).tolist()),
        )
        for index, (i, j, fam) in enumerate(
            zip(relation.edge_i, relation.edge_j, relation.family)
        )
    )
    return (int(relation.n_objects), tuple(rows))


def _rewire_relations(
    edge_i: torch.Tensor, edge_j: torch.Tensor, n_switches: int = 64
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Deterministic 2-switch of the relation graph (object set unchanged).

    Pairs consecutive undirected relation edges ``(a,b)``, ``(c,d)`` with four
    distinct endpoints and replaces them by ``(a,c)``, ``(b,d)`` when neither
    new edge already exists.  Object identities, object features and the number
    of relation edges are preserved; only the relation topology changes.
    """
    ei = [int(v) for v in edge_i.tolist()]
    ej = [int(v) for v in edge_j.tolist()]
    existing = {(min(a, b), max(a, b)) for a, b in zip(ei, ej)}
    done = 0
    for index in range(len(ei)):
        if done >= int(n_switches):
            break
        for other in range(index + 1, len(ei)):
            a, b = ei[index], ej[index]
            c, d = ei[other], ej[other]
            if len({a, b, c, d}) != 4:
                continue
            n1 = (min(a, c), max(a, c))
            n2 = (min(b, d), max(b, d))
            if (
                n1 not in existing
                and n2 not in existing
                and n1[0] != n1[1]
                and n2[0] != n2[1]
            ):
                existing.discard((min(a, b), max(a, b)))
                existing.discard((min(c, d), max(c, d)))
                existing.add(n1)
                existing.add(n2)
                ei[index], ej[index] = n1
                ei[other], ej[other] = n2
                done += 1
                break
    return (
        torch.tensor(ei, dtype=torch.long),
        torch.tensor(ej, dtype=torch.long),
        done,
    )


def _adversarial_connectivity_pair():
    """Same B0/B1 primitive multisets, different connectivity -> different B2.

    Both are seven-atom graphs on one atom type and one bond type with the SAME
    degree multiset ``{4,4,4,4,4,4,2}``, the SAME rooted distance multiset and
    the SAME B1 symmetric endpoint-primitive multiset, but a different edge set,
    hence a different explicit B2 basis and a different relation graph.
    """
    graph_a_edges = [
        (0, 1), (0, 2), (0, 4), (0, 5), (1, 2), (1, 5), (1, 6),
        (2, 4), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6),
    ]
    graph_b_edges = [
        (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4),
        (1, 6), (2, 6), (3, 4), (3, 5), (4, 5), (5, 6),
    ]
    graph_a = _hand_graph(graph_a_edges, 7)
    graph_b = _hand_graph(graph_b_edges, 7)
    dist_a = _breadth_first_distances(graph_a_edges, 7, 0)
    dist_b = _breadth_first_distances(graph_b_edges, 7, 0)
    root = np.zeros(7, dtype=np.int64)
    root[0] = 1
    batch_a = basis_batch_from_graph(graph_a, root=root, dist=dist_a)
    batch_b = basis_batch_from_graph(graph_b, root=root, dist=dist_b)
    primitive_a = (
        tuple(sorted(graph_a.degree.tolist())),
        tuple(sorted(graph_a.l1_bond.tolist())),
        tuple(sorted(dist_a.tolist())),
    )
    primitive_b = (
        tuple(sorted(graph_b.degree.tolist())),
        tuple(sorted(graph_b.l1_bond.tolist())),
        tuple(sorted(dist_b.tolist())),
    )
    return graph_a, graph_b, batch_a, batch_b, primitive_a, primitive_b


def _endpoint_primitive_multiset(graph: ExplicitBasisGraph, dist: np.ndarray):
    """Sorted multiset of B1 symmetric endpoint primitive tuples."""
    rows = []
    for index in range(graph.n_bonds):
        u = int(graph.l1_u[index])
        v = int(graph.l1_v[index])
        rows.append(
            tuple(
                sorted(
                    [
                        (int(graph.degree[u]), int(dist[u])),
                        (int(graph.degree[v]), int(dist[v])),
                    ]
                )
                + [int(graph.l1_bond[index])]
            )
        )
    return tuple(sorted(rows))


def _triple_center0_closure(graph: ExplicitBasisGraph) -> int:
    """Closure bond type of the B2 object centred on atom 0, if any."""
    for index in range(graph.n_triples):
        if int(graph.b2_center[index]) == 0:
            return int(graph.b2_closure[index])
    return -1


def _relation_lookup(
    relation: ObjectRelationGraph, i: int, j: int
) -> tuple[int, np.ndarray] | None:
    key = (min(int(i), int(j)), max(int(i), int(j)))
    for index in range(relation.edge_i.shape[0]):
        pair = (int(relation.edge_i[index]), int(relation.edge_j[index]))
        if pair == key:
            return int(relation.family[index]), relation.feature[index]
    return None


class _CloneBatch:
    """Attribute container copy for object-state perturbation checks."""


def _clone_batch(batch: Any) -> Any:
    clone = _CloneBatch()
    for name in (
        "struct_atom",
        "struct_root",
        "struct_dist",
        "struct_patch",
        "esb_degree",
        "esb_l1_u",
        "esb_l1_v",
        "esb_l1_bond",
        "esb_l1_patch",
        "esb_b2_center",
        "esb_b2_u",
        "esb_b2_w",
        "esb_b2_bond_cu",
        "esb_b2_bond_cw",
        "esb_b2_closure",
        "esb_b2_patch",
        "eor_edge_i",
        "eor_edge_j",
        "eor_family",
        "eor_feat",
    ):
        setattr(clone, name, getattr(batch, name))
    clone.struct_n_patches = batch.struct_n_patches
    return clone


def _with_swapped(batch: Any, which: str) -> Any:
    clone = _clone_batch(batch)
    if which == "b1":
        clone.esb_l1_u = batch.esb_l1_v
        clone.esb_l1_v = batch.esb_l1_u
    elif which == "b2":
        clone.esb_b2_u = batch.esb_b2_w
        clone.esb_b2_w = batch.esb_b2_u
        clone.esb_b2_bond_cu = batch.esb_b2_bond_cw
        clone.esb_b2_bond_cw = batch.esb_b2_bond_cu
    elif which == "relation":
        clone.eor_edge_i = batch.eor_edge_j
        clone.eor_edge_j = batch.eor_edge_i
    else:  # pragma: no cover - defensive
        raise ValueError(which)
    return clone


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    train_data, valid_data, _audit = build_encoded_records()
    loader = _make_esb_loader(list(valid_data)[:128], 128, False, 0)
    batch = next(iter(loader)).to(device)

    model = build_candidate(0).to(device)
    baseline = cd.build_cell(CELL, 0).to(device)
    encoder = model.structural_encoder
    checks: dict[str, bool] = {}

    # --- frozen geometry / outer backbone untouched -------------------------
    checks["output_width_16"] = bool(int(encoder.output_dim) == 16)
    checks["inner_T_obj_2"] = bool(int(encoder.t_obj) == 2 and int(encoder.rounds) == 2)
    checks["obj_widths_fixed"] = bool(
        int(encoder.obj_dim) == 8 and int(encoder.q_obj_dim) == 8
    )
    checks["q16_h64_outerT2"] = (
        int(model.pair_hidden) == 16
        and int(model.patch_hidden) == 64
        and int(model.recurrence_rounds) == 2
    )
    checks["parent_embedding_present"] = hasattr(model, "parent_embedding") and (
        _n_params(model.parent_embedding) == _n_params(baseline.parent_embedding)
    )
    counts = model.module_call_counts(batch)
    checks["outer_recurrent_weight_tying"] = counts == {
        "pair_projection": 4,
        "pair_encoder": 2,
        "center_update": 2,
    }
    # Q and U are single modules reused in both inner rounds (weight tying).
    encoder_module_names = [name for name, _ in encoder.named_modules()]
    checks["inner_Q_tied_across_rounds"] = (
        sum(1 for name in encoder_module_names if name == "q_encoder") == 1
        and sum(1 for name in encoder_module_names if name.startswith("q_encoder_")) == 0
    )
    checks["inner_U_tied_across_rounds"] = (
        sum(1 for name in encoder_module_names if name == "update") == 1
        and sum(1 for name in encoder_module_names if name.startswith("update_")) == 0
    )

    # --- no vocabulary-sized lookup / certificate dependence ----------------
    typed_state = [k for k in model.state_dict() if "typed_embedding" in k]
    typed_named = [k for k, _ in model.named_parameters() if "typed_embedding" in k]
    checks["no_vocab_sized_typed_embedding_weight"] = (
        len(typed_state) == 0 and len(typed_named) == 0
    )
    checks["explicit_object_relational_encoder_present"] = bool(
        getattr(encoder, "kind", "") == "explicit_object_relational"
    )
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

    # --- no message passing / attention / learned support gate --------------
    forbidden = (
        "gineconv",
        "gcnconv",
        "graphconv",
        "messagepassing",
        "transformer",
        "multiheadattention",
    )
    checks["no_message_passing_modules"] = not any(
        "message" in name for name in encoder_module_names
    ) and not any(
        any(token in type(module).__name__.lower() for token in forbidden)
        for module in encoder.modules()
    )
    checks["no_attention"] = not any(
        "attention" in name for name in encoder_module_names
    ) and not any(
        "attention" in type(module).__name__.lower() for module in encoder.modules()
    )
    forward_source = inspect.getsource(
        ExplicitObjectRelationalEncoder.forward
    ).lower()
    checks["no_edge_endpoints_in_forward"] = (
        "struct_src" not in forward_source and "struct_dst" not in forward_source
    )
    class_source = inspect.getsource(ExplicitObjectRelationalEncoder).lower()
    checks["no_learned_support_gate"] = not any(
        token in class_source
        for token in (
            "sigmoid(",
            "nn.sigmoid",
            "softmax(",
            "nn.softmax",
            "topk(",
            "gumbel",
        )
    )

    # --- exact supports / object counts (unchanged basis) -------------------
    _train_basis, valid_basis, _meta = explicit_basis()
    graph = valid_basis[0]
    atom_ok = all(
        atom_object_support(graph, i) == frozenset({int(i)})
        for i in range(graph.n_atoms)
    )
    bond_ok = True
    triple_ok = True
    center_ok = True
    for index in range(graph.n_bonds):
        if bond_object_support(graph, index) != frozenset(
            {int(graph.l1_u[index]), int(graph.l1_v[index])}
        ) or bond_object_bond_support(graph, index) != frozenset({index}):
            bond_ok = False
    for index in range(graph.n_triples):
        if triple_object_support(graph, index) != frozenset(
            {
                int(graph.b2_center[index]),
                int(graph.b2_u[index]),
                int(graph.b2_w[index]),
            }
        ):
            triple_ok = False
        if triple_object_center(graph, index) != int(graph.b2_center[index]):
            center_ok = False
        incident = {
            int(graph.b2_l1_cu[index]),
            int(graph.b2_l1_cw[index]),
        }
        closure = int(graph.b2_l1_closure[index])
        if closure >= 0:
            incident.add(closure)
        if triple_object_bond_support(graph, index) != frozenset(incident):
            triple_ok = False
    checks["atom_object_support_exact"] = bool(atom_ok)
    checks["bond_object_support_exact"] = bool(bond_ok)
    checks["triple_object_support_exact"] = bool(triple_ok)
    checks["triple_center_explicit"] = bool(center_ok)
    count_bonds_ok = True
    count_triples_ok = True
    for item in valid_basis[:64]:
        if object_count(item, 1) != int(item.n_bonds):
            count_bonds_ok = False
        if object_count(item, 2) != expected_triple_count(item):
            count_triples_ok = False
    checks["bond_count_equals_real_bonds"] = bool(count_bonds_ok)
    checks["triple_count_equals_sum_choose_degree"] = bool(count_triples_ok)

    # --- relation graph integrity on a real batch ---------------------------
    _train_rel, valid_rel, _rmeta = explicit_relations()
    relation = valid_rel[0]
    obj_patch = np.concatenate(
        [
            valid_basis[0].patch,
            valid_basis[0].l1_patch,
            valid_basis[0].b2_patch,
        ]
    )
    within_patch = bool(
        np.all(obj_patch[relation.edge_i] == obj_patch[relation.edge_j])
    ) if relation.edge_i.size else True
    no_self_loop = bool(np.all(relation.edge_i < relation.edge_j))
    valid_family = bool(
        np.all((relation.family >= 0) & (relation.family < N_FAMILIES))
    )
    feature_width = bool(
        relation.feature.shape[1] == REL_FEATURE_DIM if relation.feature.size else True
    )
    checks["relations_within_patch"] = within_patch
    checks["relations_no_self_loops"] = no_self_loop
    checks["relations_valid_family"] = valid_family
    checks["relation_feature_width"] = feature_width
    # determinism: rebuilding yields the identical graph
    rebuilt = build_object_relation_graph(
        valid_basis[0],
        root_atoms=np.asarray(
            list(valid_data)[0].struct_root.numpy(), dtype=np.int64
        ),
    )
    checks["relation_graph_deterministic"] = bool(
        np.array_equal(rebuilt.edge_i, relation.edge_i)
        and np.array_equal(rebuilt.edge_j, relation.edge_j)
        and np.array_equal(rebuilt.family, relation.family)
        and np.array_equal(rebuilt.feature, relation.feature)
    )
    # sparsity: never close to complete
    ratio = (
        float(relation.edge_i.size) / float(relation.n_objects)
        if relation.n_objects
        else 0.0
    )
    checks["relation_graph_sparse"] = bool(
        float(_rmeta["valid"]["mean_edge_object_ratio"]) < 20.0
        and float(_rmeta["valid"]["max_edge_object_ratio"]) < 40.0
    )

    # --- explicit relation semantics on a hand path graph -------------------
    path_edges = [(0, 1), (1, 2)]
    path_graph = _hand_graph(path_edges, 3)
    path_root = np.zeros(3, dtype=np.int64)
    path_root[0] = 1
    path_rel = build_object_relation_graph(
        path_graph,
        root_atoms=path_root,
    )
    # object ids: 0,1,2 = B0; 3 = bond(0,1); 4 = bond(1,2); 5 = B2 centred 1
    adjacent_edge = _relation_lookup(path_rel, 0, 1)
    bond_overlap_edge = _relation_lookup(path_rel, 3, 5)
    containment_edge = _relation_lookup(path_rel, 0, 5)
    centre_edge = _relation_lookup(path_rel, 1, 5)
    checks["family_adjacent_correct"] = bool(
        adjacent_edge is not None and adjacent_edge[0] == FAMILY_ADJACENT
    )
    checks["family_bond_overlap_correct"] = bool(
        bond_overlap_edge is not None
        and bond_overlap_edge[0] == FAMILY_BOND_OVERLAP
    )
    checks["containment_descriptor_correct"] = bool(
        containment_edge is not None and float(containment_edge[1][8]) == 1.0
    )
    checks["share_centre_descriptor_correct"] = bool(
        centre_edge is not None and float(centre_edge[1][9]) == 1.0
    )
    checks["no_relation_when_supports_disjoint"] = bool(
        _relation_lookup(path_rel, 0, 2) is None
    )
    # graph distance: on the path 0-1-2, the B0(0)/B1(1,2) relation is adjacent
    # (atom 0 not bonded to support {1,2}? 0-1 is a bond) -> overlap family, and
    # B0(0)/B2 support contains 0 so overlap; check min-distance bucket of the
    # disjoint adjacent pair on a larger path instead.
    dist_edges = [(0, 1), (1, 2), (2, 3)]
    dist_graph = _hand_graph(dist_edges, 4)
    dist_root = np.zeros(4, dtype=np.int64)
    dist_root[0] = 1
    dist_rel = build_object_relation_graph(dist_graph, root_atoms=dist_root)
    # B0(0) and B0(2) supports are disjoint and NOT bonded -> no edge.
    checks["graph_distance_no_false_edge"] = bool(
        _relation_lookup(dist_rel, 0, 2) is None
    )

    # --- closure edge changes the B2 explicit feature -----------------------
    path2 = _hand_graph([(0, 1), (0, 2)], 3)
    closed = _hand_graph([(0, 1), (0, 2), (1, 2)], 3)
    checks["closure_changes_b2_feature"] = bool(
        _triple_center0_closure(path2) == -1
        and _triple_center0_closure(closed) >= 0
    )

    # --- endpoint-swap invariance of the relational output ------------------
    encoder.eval()
    with torch.no_grad():
        reference = encoder(batch)
        swap_b1 = _with_swapped(batch, "b1")
        out_b1 = encoder(swap_b1)
        swap_b2 = _with_swapped(batch, "b2")
        out_b2 = encoder(swap_b2)
        swap_rel = _with_swapped(batch, "relation")
        out_rel = encoder(swap_rel)
    relation_swap_delta = float((reference - out_rel).abs().max())
    checks["bond_endpoint_swap_invariant"] = bool(
        float((reference - out_b1).abs().max()) < 1.0e-6
    )
    checks["triple_endpoint_swap_invariant"] = bool(
        float((reference - out_b2).abs().max()) < 1.0e-6
    )
    # Endpoint swap of an undirected relation changes only the scatter-add
    # summation order, so the tolerance is looser than the exact-symmetry cases.
    checks["relation_endpoint_swap_invariant"] = bool(relation_swap_delta < 1.0e-4)

    # --- late pooling after T_obj = 2, not before ---------------------------
    with torch.no_grad():
        full_details = encoder(batch, return_details=True)
        encoder._rounds = 1
        with torch.no_grad():
            round1 = encoder(batch)
        encoder._rounds = 2
    checks["uses_second_round"] = bool(
        float((full_details["e_struct"] - round1).abs().max()) > 1.0e-8
    )
    checks["pooling_after_all_rounds"] = bool(
        len(full_details["h_rounds"]) == 3
    )

    # --- rank-1 structural channel by construction --------------------------
    e_struct = full_details["e_struct"]
    delta = e_struct - encoder.rank1_bias.unsqueeze(0)
    delta = delta - delta.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(delta)
    checks["rank1_channel_by_construction"] = bool(
        int((singular > 1.0e-6 * float(singular[0].clamp_min(1.0e-12))).sum()) <= 1
    )

    # --- relation-topology intervention witness -----------------------------
    # Same molecular graph => identical explicit objects => the early
    # explicit-basis pooling is *exactly* unchanged; only the support-derived
    # relation topology is rewired by a deterministic 2-switch.  This isolates
    # "relations among explicit objects" from "new object features".
    esb_encoder = ExplicitStructuralBasisEncoder(
        atom_categories=28,
        bond_categories=4,
        n_distance_bins=PATCH_RADIUS + 1,
        n_degree_bins=ESB_N_DEGREE_BINS,
        node_dim=ESB_NODE_DIM,
        edge_dim=ESB_EDGE_DIM,
        valuation_hidden=ESB_VALUATION_HIDDEN,
        fusion_hidden=ESB_FUSION_HIDDEN,
        output_dim=TOKEN_WIDTH,
    ).eval()
    rewired = _clone_batch(batch)
    new_i, new_j, n_switches = _rewire_relations(
        batch.eor_edge_i, batch.eor_edge_j, n_switches=64
    )
    rewired.eor_edge_i = new_i
    rewired.eor_edge_j = new_j
    with torch.no_grad():
        early_reference = esb_encoder(batch)
        early_rewired = esb_encoder(rewired)
        late_reference = encoder(batch)
        late_rewired = encoder(rewired)
    early_pool_delta = float((early_reference - early_rewired).abs().max())
    intervention_late_delta = float((late_reference - late_rewired).abs().max())
    relation_topology_changed = bool(n_switches > 0)
    checks["witness_early_pooling_identical"] = bool(early_pool_delta == 0.0)
    checks["witness_relation_topology_differs"] = bool(relation_topology_changed)
    checks["witness_late_relational_output_differs"] = bool(
        intervention_late_delta > 1.0e-6
    )

    # --- connectivity witness (near-identical early pooling, different B2) ---
    (
        graph_a,
        graph_b,
        batch_a,
        batch_b,
        primitive_a,
        primitive_b,
    ) = _adversarial_connectivity_pair()
    dist_a = np.asarray(batch_a.struct_dist.numpy(), dtype=np.int64)
    dist_b = np.asarray(batch_b.struct_dist.numpy(), dtype=np.int64)
    primitives_match = primitive_a == primitive_b
    endpoint_match = _endpoint_primitive_multiset(
        graph_a, dist_a
    ) == _endpoint_primitive_multiset(graph_b, dist_b)
    b2_differs = _basis_signature(graph_a) != _basis_signature(graph_b)
    with torch.no_grad():
        pair_early_a = esb_encoder(batch_a)
        pair_early_b = esb_encoder(batch_b)
        out_a = encoder(batch_a)
        out_b = encoder(batch_b)
    pair_early_delta = float((pair_early_a - pair_early_b).abs().max())
    pair_late_delta = float((out_a - out_b).abs().max())
    checks["adversarial_primitives_match"] = bool(primitives_match)
    checks["adversarial_b1_primitive_multiset_match"] = bool(endpoint_match)
    checks["adversarial_b2_basis_differs"] = bool(b2_differs)
    checks["adversarial_output_differs"] = bool(pair_late_delta > 1.0e-6)

    # --- node relabeling invariance -----------------------------------------
    tri_edges = [(0, 1), (1, 2), (0, 2)]
    tri_graph = _hand_graph(tri_edges, 3)
    with torch.no_grad():
        relabel_reference = encoder(
            basis_batch_from_graph(
                tri_graph, dist=np.asarray([0, 1, 2], dtype=np.int64)
            )
        )
    permutation = {0: 2, 1: 0, 2: 1}
    relabelled = [
        (permutation[u], permutation[v]) for u, v in tri_edges
    ]
    relabelled_graph = _hand_graph(relabelled, 3)
    root = np.zeros(3, dtype=np.int64)
    root[permutation[0]] = 1
    dist = np.zeros(3, dtype=np.int64)
    dist[permutation[0]] = 0
    dist[permutation[1]] = 1
    dist[permutation[2]] = 2
    with torch.no_grad():
        relabelled_out = encoder(
            basis_batch_from_graph(relabelled_graph, root=root, dist=dist)
        )
    checks["node_relabel_invariant"] = bool(
        float((relabel_reference - relabelled_out).abs().max()) < 1.0e-6
    )

    # --- parameter budget + forward/backward finite -------------------------
    total = _n_params(model)
    checks["params_in_range"] = bool(PARAM_LOWER <= total <= PARAM_UPPER)
    checks["params_match_bfull_within_1pct"] = bool(
        abs(total - REFERENCE_BFULL_PARAMS) / REFERENCE_BFULL_PARAMS
        <= BUDGET_MATCH_TOLERANCE
    )
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

    # --- deterministic forward smoke ----------------------------------------
    model.eval()
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
    checks["deterministic_forward_smoke"] = bool(
        float((first - second).abs().max()) == 0.0
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failed": [name for name, ok in checks.items() if not ok],
        "candidate_params": int(total),
        "bfull_params": int(REFERENCE_BFULL_PARAMS),
        "encoder_params": int(_n_params(encoder)),
        "relation_sparsity": {
            "first_valid_relation_edge_object_ratio": float(ratio),
            "n_objects": int(relation.n_objects),
            "n_relations": int(relation.edge_i.size),
        },
        "relation_topology_intervention_witness": {
            "early_pooling_delta": float(early_pool_delta),
            "n_relation_2_switches": int(n_switches),
            "relation_topology_changed": bool(relation_topology_changed),
            "late_relational_output_delta": float(intervention_late_delta),
        },
        "connectivity_witness": {
            "primitives_match": bool(primitives_match),
            "b1_primitive_multiset_match": bool(endpoint_match),
            "b2_basis_differs": bool(b2_differs),
            "early_pooling_delta": float(pair_early_delta),
            "late_relational_output_delta": float(pair_late_delta),
            "tree_a_triples": int(graph_a.n_triples),
            "tree_b_triples": int(graph_b.n_triples),
        },
        "split": "first 128 official-valid molecules",
        "environment": _environment_fingerprint("cpu"),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        raise RuntimeError(
            f"explicit object relational sanity failed: {payload['failed']}"
        )
    return payload

# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def train_seed(
    seed: int,
    device: str = "cpu",
    tag: str = "eor",
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
    loader = _make_esb_loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = _make_esb_loader(
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
        "patch_representation": "explicit_object_relational",
        "encoder_params": int(_n_params(model.structural_encoder)),
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


def soup_seed(seed: int, tag: str = "eor") -> dict[str, Any]:
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


def train_queue(seeds: Sequence[int], device: str, tag: str = "eor") -> None:
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
# structural diagnostics
# ---------------------------------------------------------------------------


def _load_encoder_state(model: nn.Module, seed: int, tag: str) -> nn.Module:
    model.load_state_dict(
        torch.load(
            STATE_DIR / f"{tag}_seed{seed}_selection_state.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    return model


def _collect_object_details(encoder: nn.Module, loader, device: torch.device) -> dict[str, Any]:
    """Concatenate explicit object/relation states over a loader.

    Returns per-round object states ``h^0, h^1, h^2``, per-round relation
    latents ``q^0, q^1``, the object order vector, the per-round update norms,
    the patch scalar ``s(P)`` and the rank-1 channel ``e_struct``.
    """
    h_rounds: list[list[np.ndarray]] = [[], [], []]
    q_rounds: list[list[np.ndarray]] = [[], []]
    orders: list[np.ndarray] = []
    update_sums = [0.0, 0.0]
    n_batches = 0
    s_blocks: list[np.ndarray] = []
    e_blocks: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            details = encoder(batch, return_details=True)
            s_blocks.append(details["s"].cpu().numpy())
            e_blocks.append(details["e_struct"].cpu().numpy())
            if "h_rounds" not in details:
                continue
            for index, tensor in enumerate(details["h_rounds"]):
                h_rounds[index].append(tensor.cpu().numpy())
            for index, tensor in enumerate(details["q_rounds"]):
                q_rounds[index].append(tensor.cpu().numpy())
            orders.append(details["obj_order"].cpu().numpy())
            for index, value in enumerate(details["update_magnitudes"]):
                update_sums[index] += float(value)
            n_batches += 1
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)  # noqa: E731
    return {
        "h_rounds": [cat(parts) for parts in h_rounds],
        "q_rounds": [cat(parts) for parts in q_rounds],
        "obj_order": cat(orders),
        "update_norm_sum": [float(v) for v in update_sums],
        "n_batches": int(n_batches),
        "s": cat(s_blocks),
        "e_struct": (
            np.concatenate(e_blocks, axis=0) if e_blocks else np.zeros((0, 16))
        ),
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"n": 0}
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
    }


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        return {"n": int(array.shape[0]) if array.ndim else 0}
    return {
        "n": int(array.shape[0]),
        "dim": int(array.shape[1]),
        "per_dim_std_mean": float(array.std(axis=0).mean()),
        "norm_mean": float(np.linalg.norm(array, axis=1).mean()),
        **_effective_rank(array),
    }


def _update_relative(
    h_rounds: Sequence[np.ndarray], orders: np.ndarray
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for order in range(3):
        mask = orders == order
        if not bool(mask.any()):
            continue
        entry: dict[str, float] = {}
        for round_index in range(len(h_rounds) - 1):
            previous = h_rounds[round_index][mask]
            current = h_rounds[round_index + 1][mask]
            denominator = float(np.linalg.norm(previous, axis=1).mean())
            delta = float(np.linalg.norm(current - previous, axis=1).mean())
            entry[f"round_{round_index}_relative"] = delta / max(denominator, 1.0e-12)
        out[f"B{order}"] = entry
    return out


def diagnostics(tag: str = "eor", n_molecules: int = 256) -> dict[str, Any]:
    device = torch.device("cpu")
    _train, valid_data, _audit = build_encoded_records()
    _train_basis, valid_basis, _meta = explicit_basis()
    subset = list(valid_data)[: int(n_molecules)]
    subset_basis = list(valid_basis)[: int(n_molecules)]
    loader = _make_esb_loader(subset, 128, False, 0)

    counts = {
        "atoms": np.asarray([g.n_atoms for g in subset_basis], dtype=np.float64),
        "bonds": np.asarray([g.n_bonds for g in subset_basis], dtype=np.float64),
        "triples": np.asarray([g.n_triples for g in subset_basis], dtype=np.float64),
        "patches": np.asarray([g.n_patches for g in subset_basis], dtype=np.float64),
    }
    relation_counts_subset = [relation_counts(g) for g in subset_basis]
    n_objects = np.asarray([r["objects"] for r in relation_counts_subset])
    n_relations = np.asarray([r["relations"] for r in relation_counts_subset])
    total_patches = float(counts["patches"].sum())
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "patch_representation": "explicit_object_relational",
        "n_molecules": int(len(subset)),
        "n_patches": int(total_patches),
        "objects_per_patch": {
            "atoms": float(counts["atoms"].sum() / max(total_patches, 1.0)),
            "bonds": float(counts["bonds"].sum() / max(total_patches, 1.0)),
            "triples": float(counts["triples"].sum() / max(total_patches, 1.0)),
            "objects": float(n_objects.sum() / max(total_patches, 1.0)),
        },
        "relations_per_patch": {
            "mean": float(n_relations.sum() / max(total_patches, 1.0)),
            "p95_per_molecule": float(np.percentile(n_relations, 95)),
            "max_per_molecule": float(n_relations.max()),
            "mean_edge_object_ratio": float(
                np.divide(
                    n_relations,
                    n_objects,
                    out=np.zeros_like(n_relations),
                    where=n_objects > 0,
                ).mean()
            ),
            "family_totals": {
                FAMILY_NAMES[fam]: float(
                    sum(r[f"family_{FAMILY_NAMES[fam]}"] for r in relation_counts_subset)
                )
                for fam in range(N_FAMILIES)
            },
        },
        "per_seed": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        model = build_candidate(seed).to(device)
        encoder = model.structural_encoder
        _load_encoder_state(model, seed, tag)
        collected = _collect_object_details(encoder, loader, device)
        orders = collected["obj_order"]
        h_rounds = collected["h_rounds"]
        per_round: dict[str, Any] = {}
        for round_index, array in enumerate(h_rounds):
            entry = {"all": _array_stats(array)}
            for order in range(3):
                entry[f"B{order}"] = _array_stats(array[orders == order])
            per_round[str(round_index)] = entry
        q_stats = {
            str(index): _array_stats(array)
            for index, array in enumerate(collected["q_rounds"])
        }
        s = collected["s"]
        payload["per_seed"][str(seed)] = {
            "h_rounds": per_round,
            "q_rounds": q_stats,
            "update_relative": _update_relative(h_rounds, orders),
            "s_patch": _distribution(s),
            "e_struct_rank": _effective_rank(collected["e_struct"]),
            "e_struct_norm_mean": float(
                np.linalg.norm(collected["e_struct"], axis=1).mean()
            ),
            "e_struct_norm_std": float(
                np.linalg.norm(collected["e_struct"], axis=1).std()
            ),
            "s_collapse": bool(float(np.std(s)) < 1.0e-6),
        }
    _write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# frozen mechanism diagnostics (relation ablation / round ablation / PC1)
# ---------------------------------------------------------------------------


def _s_from_encoder(encoder: nn.Module, loader, device: torch.device) -> np.ndarray:
    blocks: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            details = encoder(batch, return_details=True)
            blocks.append(details["s"].cpu().numpy())
    return np.concatenate(blocks) if blocks else np.zeros(0)


def _evaluate_with_hook(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    loader,
    device: torch.device,
    *,
    ablate_relations: bool = False,
    rounds: int | None = None,
    family_mask: Sequence[bool] | None = None,
) -> float:
    """Frozen inference diagnostic; never retrains and never selects a model."""
    model.load_state_dict(state)
    encoder = model.structural_encoder
    previous = (
        bool(encoder._ablate_relations),
        int(encoder._rounds),
        tuple(encoder._family_mask),
    )
    if ablate_relations:
        encoder._ablate_relations = True
    if rounds is not None:
        encoder._rounds = int(rounds)
    if family_mask is not None:
        encoder._family_mask = tuple(bool(value) for value in family_mask)
    try:
        return _evaluate_mae(model, loader, device)
    finally:
        (
            encoder._ablate_relations,
            encoder._rounds,
            encoder._family_mask,
        ) = previous


def mechanism(tag: str = "eor", device: str = "auto") -> dict[str, Any]:
    device_obj = torch.device(
        "cuda"
        if (device == "auto" and torch.cuda.is_available())
        else ("cuda" if device == "cuda" else "cpu")
    )
    _train, valid_data, _audit = build_encoded_records()
    loader = _selection_loader(valid_data)
    seed = PROMOTION_SEED
    model = build_candidate(seed).to(device_obj)
    selection_state = torch.load(
        STATE_DIR / f"{tag}_seed{seed}_selection_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    soup_state = torch.load(
        SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt",
        map_location="cpu",
        weights_only=True,
    )

    # --- initialization vs trained object/relation state --------------------
    fresh_encoder = build_candidate(seed).structural_encoder
    valid_subset = list(valid_data)[:256]
    sub_loader = _make_esb_loader(valid_subset, 128, False, 0)
    initial = _collect_object_details(fresh_encoder, sub_loader, torch.device("cpu"))
    _load_encoder_state(model, seed, tag)
    trained = _collect_object_details(
        model.structural_encoder, sub_loader, device_obj
    )

    def _state_summary(collected: Mapping[str, Any]) -> dict[str, Any]:
        orders = collected["obj_order"]
        return {
            "h_rounds": {
                str(index): {
                    "all": _array_stats(array),
                    **{
                        f"B{order}": _array_stats(array[orders == order])
                        for order in range(3)
                    },
                }
                for index, array in enumerate(collected["h_rounds"])
            },
            "q_rounds": {
                str(index): _array_stats(array)
                for index, array in enumerate(collected["q_rounds"])
            },
            "update_relative": _update_relative(
                collected["h_rounds"], collected["obj_order"]
            ),
            "s_patch": _distribution(collected["s"]),
            "e_struct_rank": _effective_rank(collected["e_struct"]),
            "s_collapse": bool(float(np.std(collected["s"])) < 1.0e-6),
        }

    valuation_payload = {
        "initial": _state_summary(initial),
        "trained": _state_summary(trained),
    }

    # --- frozen relation / round / family ablations on official valid -------
    raw_base = _evaluate_with_hook(model, selection_state, loader, device_obj)
    soup_base = _evaluate_with_hook(model, soup_state, loader, device_obj)

    def _pair(**kwargs) -> dict[str, Any]:
        raw = _evaluate_with_hook(model, selection_state, loader, device_obj, **kwargs)
        soup = _evaluate_with_hook(model, soup_state, loader, device_obj, **kwargs)
        return {
            "raw_valid": float(raw),
            "soup_valid": float(soup),
            "delta_raw_vs_baseline": float(raw - raw_base),
            "delta_soup_vs_baseline": float(soup - soup_base),
        }

    relation_zero = _pair(ablate_relations=True)
    round_one_only = _pair(rounds=1)
    family_ablation: dict[str, Any] = {}
    for fam in range(N_FAMILIES):
        mask = [True, True, True]
        mask[fam] = False
        family_ablation[FAMILY_NAMES[fam]] = _pair(family_mask=tuple(mask))

    # --- candidate scalar s(P) vs frozen B-full PC1 -------------------------
    s_valid = _s_from_encoder(model.structural_encoder, loader, device_obj)
    pc1_payload: dict[str, Any] = {"available": False}
    collect_dir = TRACK_ROOT / "results/structural_encoder_rank1_audit/collect"
    direction_path = (
        TRACK_ROOT
        / "results/structural_encoder_rank1_audit"
        / f"pc1_direction_seed{seed}.npy"
    )
    mean_path = (
        TRACK_ROOT
        / "results/structural_encoder_rank1_audit"
        / f"pc1_mean_seed{seed}.npy"
    )
    e_bfull_path = collect_dir / f"e_struct_valid_seed{seed}.npy"
    if direction_path.exists() and mean_path.exists() and e_bfull_path.exists():
        from scipy.stats import pearsonr, spearmanr

        direction = np.load(direction_path)
        centre = np.load(mean_path)
        e_bfull = np.load(e_bfull_path).astype(np.float64)
        if e_bfull.shape[0] == s_valid.shape[0]:
            pc1_score = (e_bfull - centre) @ direction
            rho = float(spearmanr(s_valid, pc1_score).statistic)
            r = float(pearsonr(s_valid, pc1_score).statistic)
            pc1_payload = {
                "available": True,
                "n_patches": int(s_valid.size),
                "spearman_rho": rho,
                "pearson_r": r,
                "linear_r2": float(r * r),
                "abs_spearman": abs(rho),
            }
        else:
            pc1_payload = {
                "available": False,
                "reason": "patch occurrence count mismatch",
                "candidate": int(s_valid.size),
                "bfull": int(e_bfull.shape[0]),
            }

    # --- candidate scalar s(P) vs previous explicit-basis scalar ------------
    basis_scalar_payload: dict[str, Any] = {"available": False}
    try:
        from tracks.ksvd.experiments.luyin16 import (
            zinc_explicit_structural_basis as esbdrv,
        )

        basis_state_path = esbdrv.STATE_DIR / f"esb_seed{seed}_selection_state.pt"
        if basis_state_path.exists():
            basis_model = esbdrv.build_candidate(seed).to(device_obj)
            basis_model.load_state_dict(
                torch.load(basis_state_path, map_location="cpu", weights_only=True)
            )
            s_basis = _s_from_encoder(
                basis_model.structural_encoder, loader, device_obj
            )
            if s_basis.shape[0] == s_valid.shape[0]:
                from scipy.stats import pearsonr, spearmanr

                rho = float(spearmanr(s_valid, s_basis).statistic)
                r = float(pearsonr(s_valid, s_basis).statistic)
                basis_scalar_payload = {
                    "available": True,
                    "n_patches": int(s_valid.size),
                    "spearman_rho": rho,
                    "pearson_r": r,
                    "linear_r2": float(r * r),
                    "basis_state_path": str(basis_state_path),
                }
            else:
                basis_scalar_payload = {
                    "available": False,
                    "reason": "patch occurrence count mismatch",
                    "candidate": int(s_valid.size),
                    "basis": int(s_basis.shape[0]),
                }
        else:
            basis_scalar_payload = {
                "available": False,
                "reason": "previous explicit-basis selection state not present",
                "path": str(basis_state_path),
            }
    except Exception as error:  # pragma: no cover - diagnostic only
        basis_scalar_payload = {"available": False, "reason": repr(error)}

    # --- existing deterministic descriptors -> s(P) ridge (alpha = 1) ------
    descriptor_payload: dict[str, Any] = {"available": False}
    x_train_path = collect_dir / "descriptors_train.npy"
    x_valid_path = collect_dir / "descriptors_valid.npy"
    if x_train_path.exists() and x_valid_path.exists():
        train_data, _valid, _audit2 = build_encoded_records()
        train_loader = _make_esb_loader(list(train_data), 128, False, 0)
        s_train = _s_from_encoder(model.structural_encoder, train_loader, device_obj)
        x_train = np.load(x_train_path).astype(np.float64)
        x_valid = np.load(x_valid_path).astype(np.float64)
        if x_train.shape[0] == s_train.size and x_valid.shape[0] == s_valid.size:
            train_x, valid_x = r1a._standardize(x_train, x_valid)
            descriptor_payload = {
                "available": True,
                **(r1a._ridge_r2(train_x, s_train, valid_x, s_valid)),
            }
        else:
            descriptor_payload = {
                "available": False,
                "reason": "descriptor / patch occurrence count mismatch",
                "train": [int(x_train.shape[0]), int(s_train.size)],
                "valid": [int(x_valid.shape[0]), int(s_valid.size)],
            }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "valuation": valuation_payload,
        "relation_zero_ablation": relation_zero,
        "round_one_only_ablation": round_one_only,
        "family_ablation": {
            "baseline_raw_valid": float(raw_base),
            "baseline_soup_valid": float(soup_base),
            "masks": family_ablation,
        },
        "pc1_vs_bfull": pc1_payload,
        "scalar_vs_previous_explicit_basis": basis_scalar_payload,
        "ridge_from_descriptors": descriptor_payload,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism.json", payload)
    return payload


# ---------------------------------------------------------------------------
# compute profile
# ---------------------------------------------------------------------------


def profile(n_batches: int = 10, device: str = "cpu") -> dict[str, Any]:
    _train, valid_data, _audit = build_encoded_records()
    loader = _make_esb_loader(list(valid_data)[:1280], 128, False, 0)
    device_obj = torch.device(device)
    eor = build_candidate(0).structural_encoder.to(device_obj).eval()
    bfull = sspe.build_candidate(0).structural_encoder.to(device_obj).eval()
    explicit = None
    try:
        from tracks.ksvd.experiments.luyin16 import (
            zinc_explicit_structural_basis as esbdrv,
        )

        explicit = esbdrv.build_candidate(0).structural_encoder.to(device_obj).eval()
    except Exception:  # pragma: no cover - diagnostic only
        explicit = None

    batches = []
    counts = {"atoms": 0, "bonds": 0, "triples": 0, "patches": 0, "relations": 0}
    for index, batch in enumerate(loader):
        if index >= n_batches:
            break
        batch = batch.to(device_obj)
        batches.append(batch)
        counts["atoms"] += int(batch.struct_atom.numel())
        counts["bonds"] += int(batch.esb_l1_u.numel())
        counts["triples"] += int(batch.esb_b2_center.numel())
        counts["patches"] += int(batch.struct_n_patches)
        counts["relations"] += int(batch.eor_edge_i.numel())

    def _time(module, batch):
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                module(batch)
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - started) / 3.0

    eor_latency = float(np.mean([_time(eor, b) for b in batches])) if batches else 0.0
    bfull_latency = (
        float(np.mean([_time(bfull, b) for b in batches])) if batches else 0.0
    )
    explicit_latency = (
        float(np.mean([_time(explicit, b) for b in batches])) if batches else 0.0
    )
    patches = max(counts["patches"], 1)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "device": str(device),
        "n_batches": int(len(batches)),
        "batch_size": 128,
        "eor_forward_latency_s": eor_latency,
        "bfull_forward_latency_s": bfull_latency,
        "explicit_basis_forward_latency_s": explicit_latency,
        "eor_latency_vs_bfull": (
            float(eor_latency / bfull_latency) if bfull_latency else 0.0
        ),
        "eor_latency_vs_explicit_basis": (
            float(eor_latency / explicit_latency) if explicit_latency else 0.0
        ),
        "mean_atoms_per_patch": float(counts["atoms"] / patches),
        "mean_bonds_per_patch": float(counts["bonds"] / patches),
        "mean_triples_per_patch": float(counts["triples"] / patches),
        "mean_objects_per_patch": float(
            (counts["atoms"] + counts["bonds"] + counts["triples"]) / patches
        ),
        "mean_relations_per_patch": float(counts["relations"] / patches),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device_obj.type == "cuda"
            else 0
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compute_profile.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(tag: str = "eor") -> dict[str, Any]:
    seed = PROMOTION_SEED
    soup = _read_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json")
    run = _read_json(RUNS_DIR / f"{tag}_seed{seed}.json")
    soup_valid = float(soup["top5_soup_valid_mae"])
    raw_valid = float(soup["best_checkpoint_valid_mae"])
    reference = REFERENCE_BFULL_SOUP_PER_SEED[seed]
    previous = REFERENCE_EXPLICIT_BASIS_SOUP_PER_SEED[seed]
    delta_bfull = soup_valid - reference
    delta_late_relation = previous - soup_valid

    diag = (
        _read_json(RESULTS_DIR / "diagnostics.json")
        if (RESULTS_DIR / "diagnostics.json").exists()
        else None
    )
    s_collapse = bool(
        diag["per_seed"][str(seed)]["s_collapse"] if diag is not None else True
    )
    profile_payload = (
        _read_json(RESULTS_DIR / "compute_profile.json")
        if (RESULTS_DIR / "compute_profile.json").exists()
        else None
    )
    relation_sparse = True
    if diag is not None:
        ratio = float(diag["relations_per_patch"]["mean_edge_object_ratio"])
        relation_sparse = bool(ratio < 20.0)
    compute_reasonable = True
    if profile_payload is not None:
        compute_reasonable = bool(
            float(profile_payload.get("eor_latency_vs_bfull", 0.0)) <= 3.0
        )

    strong_go = soup_valid <= 0.117126
    method_go = 0.117126 < soup_valid <= 0.119126
    partial_go = 0.119126 < soup_valid < previous
    if s_collapse:
        verdict = "STOP"
    elif strong_go:
        verdict = "STRONG_GO"
    elif method_go and relation_sparse and compute_reasonable:
        verdict = "METHOD_GO"
    elif partial_go:
        verdict = "PARTIAL_GO"
    else:
        verdict = "STOP"
    better_than_previous = bool(soup_valid < previous)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "promotion_seed": int(seed),
        "single_seed": True,
        "seed1_soup_valid": soup_valid,
        "seed1_raw_valid": raw_valid,
        "reference_bfull_seed1_soup": float(reference),
        "reference_previous_explicit_basis_soup": float(previous),
        "reference_a2_seed1_soup": float(REFERENCE_A2_SOUP_PER_SEED[seed]),
        "reference_bbag_seed1_soup": float(REFERENCE_BBAG_SOUP_PER_SEED[seed]),
        "delta_seed1_soup_vs_bfull": float(delta_bfull),
        "delta_late_relation_vs_previous_explicit_basis": float(delta_late_relation),
        "delta_gap_to_bfull": float(delta_bfull),
        "delta_seed1_soup_vs_a2": float(soup_valid - REFERENCE_A2_SOUP_PER_SEED[seed]),
        "delta_seed1_soup_vs_bbag": float(
            soup_valid - REFERENCE_BBAG_SOUP_PER_SEED[seed]
        ),
        "gates": {
            "strong_go_upper": 0.117126,
            "method_go_upper": 0.119126,
            "partial_go_previous_explicit_basis": float(previous),
        },
        "s_collapse": s_collapse,
        "relation_graph_sparse": relation_sparse,
        "compute_reasonable": compute_reasonable,
        "better_than_previous_explicit_basis": better_than_previous,
        "better_than_bbag": bool(soup_valid < REFERENCE_BBAG_SOUP_PER_SEED[seed]),
        "verdict": verdict,
        "second_seed_authorized": bool(verdict in {"STRONG_GO", "METHOD_GO"}),
        "candidate_params": int(
            _read_json(RESULTS_DIR / "parameter_accounting.json")["candidate_params"]
        ),
        "wall_clock_s": float(run["wall_clock_s"]),
        "epochs_run": int(run["epochs_run"]),
        "peak_gpu_memory_bytes": int(run.get("peak_gpu_memory_bytes", 0)),
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
        "git_commit": _git_commit(),
        "parameter_accounting": _maybe("parameter_accounting"),
        "sanity": _maybe("sanity"),
        "diagnostics": _maybe("diagnostics"),
        "mechanism": _maybe("mechanism"),
        "compute_profile": _maybe("compute_profile"),
        "decision": _maybe("decision"),
        "soup": {str(seed): _maybe(f"soup_eor_seed{seed}") for seed in SEEDS},
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
            "mechanism",
            "profile",
            "decide",
            "report",
        ),
    )
    parser.add_argument("--seed", type=int, default=PROMOTION_SEED)
    parser.add_argument("--seeds", type=str, default=str(PROMOTION_SEED))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="eor")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--molecules", type=int, default=256)
    args = parser.parse_args(argv)

    global DETERMINISTIC
    DETERMINISTIC = bool(args.deterministic)
    _set_deterministic(DETERMINISTIC)
    torch.set_num_threads(4)

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    elif args.stage == "preprocess":
        explicit_basis()
        explicit_relations()
        print(
            json.dumps(
                {
                    "explicit_basis": _read_json(_esb_cache_paths()[2]),
                    "object_relation": _read_json(_eor_cache_paths()[2]),
                },
                indent=2,
            ),
            flush=True,
        )
    elif args.stage == "sanity":
        print(json.dumps(sanity(), indent=2, default=str), flush=True)
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
        train_queue(seeds, device=args.device, tag=args.tag)
    elif args.stage == "soup":
        print(
            json.dumps(soup_seed(args.seed, tag=args.tag), indent=2, default=str),
            flush=True,
        )
    elif args.stage == "repro":
        print(
            json.dumps(
                repro(args.seed, args.epochs, args.device, run_tag=args.run_tag),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    elif args.stage == "diagnostics":
        print(
            json.dumps(
                diagnostics(tag=args.tag, n_molecules=args.molecules),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    elif args.stage == "mechanism":
        print(
            json.dumps(
                mechanism(tag=args.tag, device=args.device),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    elif args.stage == "profile":
        print(
            json.dumps(profile(device=args.device), indent=2, default=str),
            flush=True,
        )
    elif args.stage == "decide":
        print(json.dumps(decide(tag=args.tag), indent=2, default=str), flush=True)
    elif args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
