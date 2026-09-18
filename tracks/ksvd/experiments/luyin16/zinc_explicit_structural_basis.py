"""ZINC explicit structural basis with learned scalar valuation (rank-1 channel).

Question
--------
B-full replaces the historical typed certificate lookup with a shared,
connectivity-aware two-round message-passing encoder over the real rooted
typed radius-2 patch graph (``patch_representation="shared_structural"``).
A zero-training mechanism audit showed that its 16-D ``e_struct`` is
*effectively rank-1* (effective rank ~1.1-1.2; a frozen rank-1 reconstruction
changes the 2-seed soup by ~1.6e-5), and the previous explicit-support
*composition* composer failed because its learned activity gate saturated.

This experiment tests a deliberately narrower representation hypothesis:

    The graph defines **what** structural objects exist; the network only
    learns **what each explicitly present object is worth for the task**.

Exactly one pre-registered architecture (no attention, no Transformer, no
learned dictionary, no soft assignment, no learned support/existence gate, no
message passing inside the structural module)::

    B0  atom objects              support = {v}
    B1  bond objects              support = {u, v},       bonds = {(u, v)}
    B2  centred 3-atom objects    support = {u, v, w}     centre = v,
                                  bonds = {(v, u), (v, w)} (+ closure (u, w))
    valuation  t_r = f_r(explicit object) in R    (shared per order, signed)
    statistics A(P) = [mean, std, log1p(count)] x 3 orders  in R^9
    scalar     s(P) = fusion(A(P)) in R
    channel    e_struct(P) = b + s(P) * v         with trainable b, v in R^16

``e_struct`` is rank-1 by construction.  Everything downstream is inherited
bit-exactly from cell A / B-full: radius-2 patch definition, relation system,
pair descriptors, T=2 weight-tied recurrent pair--centre, parent embedding,
global/topology channels, R=334, the fixed small head and the frozen optimized
protocol.  Official ZINC test is **never** loaded.

Stages: ``params preprocess sanity train soup train_queue repro diagnostics
mechanism profile decide report``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_explicit_structural_basis <stage>
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

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/explicit_structural_basis"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"
CACHE_DIR = RESULTS_DIR / "cache"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "explicit_structural_basis_v1"
CACHE_SCHEMA_VERSION = "explicit_structural_basis_v1"

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


def build_encoded_records():
    """Return (train_data, valid_data, audit) with the explicit basis attached."""
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
    audit = dict(audit)
    audit["patch_representation"] = "explicit_basis_rank1"
    audit["explicit_structural_basis"] = {
        "node_dim": ESB_NODE_DIM,
        "edge_dim": ESB_EDGE_DIM,
        "n_degree_bins": ESB_N_DEGREE_BINS,
        "valuation_hidden": ESB_VALUATION_HIDDEN,
        "fusion_hidden": ESB_FUSION_HIDDEN,
        "output_dim": TOKEN_WIDTH,
        "orders": [0, 1, 2],
        "channel": "b + s(P) * v (rank-1 by construction)",
    }
    audit["official_test_loaded"] = False
    return enc_train, enc_valid, audit


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_candidate(seed: int = 0) -> torch.nn.Module:
    """Cell-A geometry with the typed lookup replaced by the explicit basis.

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
            patch_representation="explicit_basis_rank1",
            esb_node_dim=ESB_NODE_DIM,
            esb_edge_dim=ESB_EDGE_DIM,
            esb_n_degree_bins=ESB_N_DEGREE_BINS,
            esb_valuation_hidden=ESB_VALUATION_HIDDEN,
            esb_fusion_hidden=ESB_FUSION_HIDDEN,
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
        n_offset += n_atoms
        patch_offset += n_patches
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
        "atom_valuation": int(_n_params(encoder.atom_valuation)),
        "bond_valuation": int(_n_params(encoder.bond_valuation)),
        "triple_valuation": int(_n_params(encoder.triple_valuation)),
        "scalar_fusion": int(_n_params(encoder.fusion)),
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
            "node_dim": int(encoder.node_dim),
            "edge_dim": int(encoder.edge_dim),
            "n_degree_bins": int(encoder.n_degree_bins),
            "valuation_hidden": int(encoder.valuation_hidden),
            "fusion_hidden": int(encoder.fusion_hidden),
            "output_dim": int(encoder.output_dim),
            "triple_input": int(encoder.triple_input),
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


def _adversarial_connectivity_pair():
    """Same B0/B1 primitive multisets, different connectivity -> different B2.

    Both are seven-atom graphs on one atom type and one bond type with the SAME
    degree multiset ``{4,4,4,4,4,4,2}`` and the SAME rooted distance multiset
    ``{0,1,1,1,1,2,2}``, the SAME B1 symmetric endpoint-primitive multiset, but
    a different edge set.  A permutation-invariant primitive bag cannot tell
    them apart; the explicit centred B2 basis does, and the pooled patch output
    moves.
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


def _descriptor_coincident_pair():
    """Six-atom trees: B0/B1 match and B2 *identified* objects differ, but the
    B2 descriptor multiset coincides (pooled output is representation-equal).

    This is the honest boundary of the connectivity witness: object existence
    differs, but a mean/std statistic over the object population can still
    coincide when the population changes by a relabelling only.
    """
    tree_a = [(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)]
    tree_b = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    graph_a = _hand_graph(tree_a, 6)
    graph_b = _hand_graph(tree_b, 6)
    root = np.zeros(6, dtype=np.int64)
    root[0] = 1
    batch_a = basis_batch_from_graph(
        graph_a, root=root, dist=_breadth_first_distances(tree_a, 6, 0)
    )
    batch_b = basis_batch_from_graph(
        graph_b, root=root, dist=_breadth_first_distances(tree_b, 6, 0)
    )
    return graph_a, graph_b, batch_a, batch_b


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


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    train_data, valid_data, _audit = build_encoded_records()
    loader = _make_esb_loader(list(valid_data)[:128], 128, False, 0)
    batch = next(iter(loader)).to(device)

    model = build_candidate(0).to(device)
    baseline = cd.build_cell(CELL, 0).to(device)
    encoder = model.structural_encoder
    checks: dict[str, bool] = {}

    # (18) output width / (19) frozen geometry.
    checks["output_width_16"] = bool(int(encoder.output_dim) == 16)
    checks["no_message_passing_rounds"] = bool(int(encoder.rounds) == 0)
    checks["q16_h64_T2"] = (
        int(model.pair_hidden) == 16
        and int(model.patch_hidden) == 64
        and int(model.recurrence_rounds) == 2
    )
    checks["parent_embedding_present"] = hasattr(model, "parent_embedding") and (
        _n_params(model.parent_embedding) == _n_params(baseline.parent_embedding)
    )
    counts = model.module_call_counts(batch)
    checks["recurrent_weight_tying"] = counts == {
        "pair_projection": 4,
        "pair_encoder": 2,
        "center_update": 2,
    }

    # (15) no vocabulary-sized lookup / certificate dependence.
    typed_state = [k for k in model.state_dict() if "typed_embedding" in k]
    typed_named = [k for k, _ in model.named_parameters() if "typed_embedding" in k]
    checks["no_vocab_sized_typed_embedding_weight"] = (
        len(typed_state) == 0 and len(typed_named) == 0
    )
    checks["explicit_basis_encoder_present"] = bool(
        getattr(encoder, "kind", "") == "explicit_basis_rank1"
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

    # (13)/(14) no message passing / no attention modules anywhere.
    forbidden = (
        "gineconv",
        "gcnconv",
        "graphconv",
        "messagepassing",
        "transformer",
        "multiheadattention",
    )
    module_names = [name for name, _ in encoder.named_modules()]
    checks["no_message_passing_modules"] = not any(
        "message" in name for name in module_names
    ) and not any(
        any(token in type(module).__name__.lower() for token in forbidden)
        for module in encoder.modules()
    )
    checks["no_attention"] = not any(
        "attention" in name for name in module_names
    ) and not any(
        "attention" in type(module).__name__.lower() for module in encoder.modules()
    )
    forward_source = inspect.getsource(
        ExplicitStructuralBasisEncoder.forward
    ).lower()
    checks["no_edge_endpoints_in_forward"] = (
        "struct_src" not in forward_source and "struct_dst" not in forward_source
    )
    # (16) no learned support / existence gate: no sigmoid / softmax / topk.
    class_source = inspect.getsource(ExplicitStructuralBasisEncoder).lower()
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

    # (2)/(3)/(4) exact supports on a real molecule.
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

    # (5)/(6) object counts are exactly topology-determined over the split.
    count_bonds_ok = True
    count_triples_ok = True
    for item in valid_basis[:64]:
        if object_count(item, 1) != int(item.n_bonds):
            count_bonds_ok = False
        if object_count(item, 2) != expected_triple_count(item):
            count_triples_ok = False
    checks["bond_count_equals_real_bonds"] = bool(count_bonds_ok)
    checks["triple_count_equals_sum_choose_degree"] = bool(count_triples_ok)

    # (7)/(8) endpoint-swap invariance of the object valuations.
    encoder.eval()
    with torch.no_grad():
        reference = encoder(batch)
        swap_b1 = _with_swapped(batch, "b1")
        out_b1 = encoder(swap_b1)
        swap_b2 = _with_swapped(batch, "b2")
        out_b2 = encoder(swap_b2)
    checks["bond_endpoint_swap_invariant"] = bool(
        float((reference - out_b1).abs().max()) < 1.0e-6
    )
    checks["triple_endpoint_swap_invariant"] = bool(
        float((reference - out_b2).abs().max()) < 1.0e-6
    )

    # (9)/(10) changing an atom / bond type changes the matching valuation.
    with torch.no_grad():
        details = encoder(batch, return_details=True)
        atom_batch = _clone_batch(batch)
        atom_batch.struct_atom = atom_batch.struct_atom.clone()
        atom_batch.struct_atom[0] = (int(atom_batch.struct_atom[0]) + 1) % 28
        atom_details = encoder(atom_batch, return_details=True)
        bond_batch = _clone_batch(batch)
        bond_batch.esb_l1_bond = bond_batch.esb_l1_bond.clone()
        if bond_batch.esb_l1_bond.numel():
            bond_batch.esb_l1_bond[0] = (
                int(bond_batch.esb_l1_bond[0]) + 1
            ) % 4
        bond_details = encoder(bond_batch, return_details=True)
        triple_batch = _clone_batch(batch)
        triple_batch.esb_b2_bond_cu = triple_batch.esb_b2_bond_cu.clone()
        if triple_batch.esb_b2_bond_cu.numel():
            triple_batch.esb_b2_bond_cu[0] = (
                int(triple_batch.esb_b2_bond_cu[0]) + 1
            ) % 4
        triple_details = encoder(triple_batch, return_details=True)
    checks["atom_type_changes_atom_valuation"] = bool(
        float((atom_details["t0"] - details["t0"]).abs().max()) > 1.0e-6
    )
    checks["bond_type_changes_bond_valuation"] = bool(
        float((bond_details["t1"] - details["t1"]).abs().max()) > 1.0e-6
    )
    checks["bond_type_changes_triple_valuation"] = bool(
        float((triple_details["t2"] - details["t2"]).abs().max()) > 1.0e-6
    )

    # (11) adversarial connectivity: same primitive multisets, different B2.
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
        out_a = encoder(batch_a)
        out_b = encoder(batch_b)
    checks["adversarial_primitives_match"] = bool(primitives_match)
    checks["adversarial_b1_primitive_multiset_match"] = bool(endpoint_match)
    checks["adversarial_b2_basis_differs"] = bool(b2_differs)
    checks["adversarial_output_differs"] = bool(
        float((out_a - out_b).abs().max()) > 1.0e-6
    )
    # Honest boundary: a `descriptor-coincident` witness where identified B2
    # objects differ but their descriptor multiset coincides.
    (dc_a, dc_b, dc_batch_a, dc_batch_b) = _descriptor_coincident_pair()
    with torch.no_grad():
        dc_out_a = encoder(dc_batch_a)
        dc_out_b = encoder(dc_batch_b)
    descriptor_coincident = {
        "b2_identified_basis_differs": bool(
            _basis_signature(dc_a) != _basis_signature(dc_b)
        ),
        "pooled_output_delta": float((dc_out_a - dc_out_b).abs().max()),
    }

    # (12) closure edge appears / disappears changes the B2 explicit features.
    path = _hand_graph([(0, 1), (0, 2)], 3)
    closed = _hand_graph([(0, 1), (0, 2), (1, 2)], 3)
    path_center0 = _triple_center0_closure(path)
    closed_center0 = _triple_center0_closure(closed)
    checks["closure_changes_b2_feature"] = bool(
        path_center0 != closed_center0
        and path_center0 == -1
        and closed_center0 >= 0
    )

    # (1) node relabeling preserves the patch output.
    path_graph = _hand_graph([(0, 1), (1, 2), (0, 2)], 3)
    with torch.no_grad():
        relabel_reference = encoder(
            basis_batch_from_graph(
                path_graph,
                dist=np.asarray([0, 1, 2], dtype=np.int64),
            )
        )
    permutation = {0: 2, 1: 0, 2: 1}
    relabelled = [
        (permutation[u], permutation[v]) for u, v in [(0, 1), (1, 2), (0, 2)]
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

    # (17) rank-1 structural channel by construction.
    with torch.no_grad():
        e_struct = encoder(batch)
    delta = e_struct - encoder.rank1_bias.unsqueeze(0)
    delta = delta - delta.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(delta)
    checks["rank1_channel_by_construction"] = bool(
        int((singular > 1.0e-6 * float(singular[0].clamp_min(1.0e-12))).sum()) <= 1
    )

    # (20) parameter budget + forward/backward finite.
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

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failed": [name for name, ok in checks.items() if not ok],
        "candidate_params": int(total),
        "bfull_params": int(REFERENCE_BFULL_PARAMS),
        "encoder_params": int(_n_params(encoder)),
        "adversarial": {
            "primitives_match": bool(primitives_match),
            "b1_primitive_multiset_match": bool(endpoint_match),
            "b2_basis_differs": bool(b2_differs),
            "output_delta": float((out_a - out_b).abs().max()),
            "tree_a_triples": int(graph_a.n_triples),
            "tree_b_triples": int(graph_b.n_triples),
        },
        "descriptor_coincident_pair": descriptor_coincident,
        "split": "first 128 official-valid molecules",
        "environment": _environment_fingerprint("cpu"),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"explicit basis sanity failed: {payload['failed']}")
    return payload


def _triple_center0_closure(graph: ExplicitBasisGraph) -> int:
    """Closure bond type of the B2 object centred on atom 0, if any."""
    for index in range(graph.n_triples):
        if int(graph.b2_center[index]) == 0:
            return int(graph.b2_closure[index])
    return -1


class _CloneBatch:
    """Attribute container copy for object-valuation perturbation checks."""


def _clone_batch(batch: Any) -> Any:
    clone = _CloneBatch()
    clone.struct_atom = batch.struct_atom
    clone.struct_root = batch.struct_root
    clone.struct_dist = batch.struct_dist
    clone.struct_patch = batch.struct_patch
    clone.esb_degree = batch.esb_degree
    clone.esb_l1_u = batch.esb_l1_u
    clone.esb_l1_v = batch.esb_l1_v
    clone.esb_l1_bond = batch.esb_l1_bond
    clone.esb_l1_patch = batch.esb_l1_patch
    clone.esb_b2_center = batch.esb_b2_center
    clone.esb_b2_u = batch.esb_b2_u
    clone.esb_b2_w = batch.esb_b2_w
    clone.esb_b2_bond_cu = batch.esb_b2_bond_cu
    clone.esb_b2_bond_cw = batch.esb_b2_bond_cw
    clone.esb_b2_closure = batch.esb_b2_closure
    clone.esb_b2_patch = batch.esb_b2_patch
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
    else:  # pragma: no cover - defensive
        raise ValueError(which)
    return clone


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def train_seed(
    seed: int,
    device: str = "cpu",
    tag: str = "esb",
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
        "patch_representation": "explicit_basis_rank1",
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


def soup_seed(seed: int, tag: str = "esb") -> dict[str, Any]:
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


def train_queue(seeds: Sequence[int], device: str, tag: str = "esb") -> None:
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


def _collect_details(encoder: nn.Module, loader, device: torch.device) -> dict[str, Any]:
    """Concatenate t0/t1/t2/s over a loader (patch-occurrence order)."""
    blocks: dict[str, list[np.ndarray]] = {"t0": [], "t1": [], "t2": [], "s": []}
    e_blocks: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            details = encoder(batch, return_details=True)
            for key in blocks:
                blocks[key].append(details[key].cpu().numpy())
            e_blocks.append(details["e_struct"].cpu().numpy())
    return {
        **{key: np.concatenate(value) for key, value in blocks.items()},
        "e_struct": np.concatenate(e_blocks, axis=0),
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


def diagnostics(tag: str = "esb", n_molecules: int = 256) -> dict[str, Any]:
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
    total_patches = float(counts["patches"].sum())
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "patch_representation": "explicit_basis_rank1",
        "n_molecules": int(len(subset)),
        "n_patches": int(total_patches),
        "objects_per_patch": {
            "atoms": float(counts["atoms"].sum() / max(total_patches, 1.0)),
            "bonds": float(counts["bonds"].sum() / max(total_patches, 1.0)),
            "triples": float(counts["triples"].sum() / max(total_patches, 1.0)),
        },
        "per_seed": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        model = build_candidate(seed).to(device)
        encoder = model.structural_encoder
        _load_encoder_state(model, seed, tag)
        collected = _collect_details(encoder, loader, device)
        rank = _effective_rank(collected["e_struct"])
        s = collected["s"]
        payload["per_seed"][str(seed)] = {
            "t_atom": _distribution(collected["t0"]),
            "t_bond": _distribution(collected["t1"]),
            "t_triple": _distribution(collected["t2"]),
            "s_patch": _distribution(s),
            "e_struct_rank": rank,
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
# frozen mechanism diagnostics (order ablation / PC1 / descriptors)
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


def _evaluate_with_ablation(
    model: nn.Module, state: Mapping[str, torch.Tensor], loader, device: torch.device, orders
) -> float:
    model.load_state_dict(state)
    model.structural_encoder._ablate_orders = tuple(int(o) for o in orders)
    result = _evaluate_mae(model, loader, device)
    model.structural_encoder._ablate_orders = ()
    return result


def mechanism(tag: str = "esb", device: str = "auto") -> dict[str, Any]:
    device_obj = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available()) else (
            "cuda" if device == "cuda" else "cpu"
        )
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

    # --- initialization vs trained valuation statistics --------------------
    fresh = build_candidate(seed)
    fresh_encoder = fresh.structural_encoder
    valid_subset = list(valid_data)[:256]
    sub_loader = _make_esb_loader(valid_subset, 128, False, 0)
    initial = _collect_details(fresh_encoder, sub_loader, torch.device("cpu"))

    _load_encoder_state(model, seed, tag)
    trained = _collect_details(
        model.structural_encoder, sub_loader, device_obj
    )
    valuation_payload = {
        "initial": {
            "t_atom": _distribution(initial["t0"]),
            "t_bond": _distribution(initial["t1"]),
            "t_triple": _distribution(initial["t2"]),
            "s_patch": _distribution(initial["s"]),
            "e_struct_rank": _effective_rank(initial["e_struct"]),
            "e_struct_norm_mean": float(
                np.linalg.norm(initial["e_struct"], axis=1).mean()
            ),
        },
        "trained": {
            "t_atom": _distribution(trained["t0"]),
            "t_bond": _distribution(trained["t1"]),
            "t_triple": _distribution(trained["t2"]),
            "s_patch": _distribution(trained["s"]),
            "e_struct_rank": _effective_rank(trained["e_struct"]),
            "e_struct_norm_mean": float(
                np.linalg.norm(trained["e_struct"], axis=1).mean()
            ),
            "s_collapse": bool(float(np.std(trained["s"])) < 1.0e-6),
        },
    }

    # --- frozen order ablation on official valid (no retraining) ------------
    raw_mae = _evaluate_with_ablation(model, selection_state, loader, device_obj, ())
    soup_mae = _evaluate_with_ablation(model, soup_state, loader, device_obj, ())
    ablation: dict[str, Any] = {
        "baseline_raw_valid": float(raw_mae),
        "baseline_soup_valid": float(soup_mae),
        "per_order": {},
    }
    for order in (0, 1, 2):
        raw_ab = _evaluate_with_ablation(
            model, selection_state, loader, device_obj, (order,)
        )
        soup_ab = _evaluate_with_ablation(
            model, soup_state, loader, device_obj, (order,)
        )
        ablation["per_order"][f"B{order}"] = {
            "raw_valid": float(raw_ab),
            "soup_valid": float(soup_ab),
            "delta_raw_vs_baseline": float(raw_ab - raw_mae),
            "delta_soup_vs_baseline": float(soup_ab - soup_mae),
        }

    # --- candidate scalar s(P) vs frozen B-full PC1 -------------------------
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
        s_valid = _s_from_encoder(model.structural_encoder, loader, device_obj)
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

    # --- existing deterministic descriptors -> s(P) ridge (alpha = 1) ------
    descriptor_payload: dict[str, Any] = {"available": False}
    x_train_path = collect_dir / "descriptors_train.npy"
    x_valid_path = collect_dir / "descriptors_valid.npy"
    if x_train_path.exists() and x_valid_path.exists():
        train_data, _valid, _audit2 = build_encoded_records()
        train_loader = _make_esb_loader(list(train_data), 128, False, 0)
        s_train = _s_from_encoder(model.structural_encoder, train_loader, device_obj)
        s_valid = _s_from_encoder(model.structural_encoder, loader, device_obj)
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
        "order_ablation": ablation,
        "pc1_vs_bfull": pc1_payload,
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
    esb = build_candidate(0).structural_encoder.to(device_obj).eval()
    bfull = sspe.build_candidate(0).structural_encoder.to(device_obj).eval()

    batches = []
    counts = {"atoms": 0, "bonds": 0, "triples": 0, "patches": 0}
    for index, batch in enumerate(loader):
        if index >= n_batches:
            break
        batch = batch.to(device_obj)
        batches.append(batch)
        counts["atoms"] += int(batch.struct_atom.numel())
        counts["bonds"] += int(batch.esb_l1_u.numel())
        counts["triples"] += int(batch.esb_b2_center.numel())
        counts["patches"] += int(batch.struct_n_patches)

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

    esb_latency = float(np.mean([_time(esb, b) for b in batches])) if batches else 0.0
    bfull_latency = (
        float(np.mean([_time(bfull, b) for b in batches])) if batches else 0.0
    )
    patches = max(counts["patches"], 1)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "device": str(device),
        "n_batches": int(len(batches)),
        "batch_size": 128,
        "esb_forward_latency_s": esb_latency,
        "bfull_forward_latency_s": bfull_latency,
        "esb_latency_ratio": (
            float(esb_latency / bfull_latency) if bfull_latency else 0.0
        ),
        "mean_atoms_per_patch": float(counts["atoms"] / patches),
        "mean_bonds_per_patch": float(counts["bonds"] / patches),
        "mean_triples_per_patch": float(counts["triples"] / patches),
        "mean_objects_per_patch": float(
            (counts["atoms"] + counts["bonds"] + counts["triples"]) / patches
        ),
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


def decide(tag: str = "esb") -> dict[str, Any]:
    seed = PROMOTION_SEED
    soup = _read_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json")
    run = _read_json(RUNS_DIR / f"{tag}_seed{seed}.json")
    soup_valid = float(soup["top5_soup_valid_mae"])
    raw_valid = float(soup["best_checkpoint_valid_mae"])
    reference = REFERENCE_BFULL_SOUP_PER_SEED[seed]
    delta = soup_valid - reference

    diag = (
        _read_json(RESULTS_DIR / "diagnostics.json")
        if (RESULTS_DIR / "diagnostics.json").exists()
        else None
    )
    s_collapse = bool(
        diag["per_seed"][str(seed)]["s_collapse"] if diag is not None else True
    )

    if s_collapse:
        verdict = "STOP"
    elif delta <= -STRONG_GATE:
        verdict = "STRONG_GO"
    elif -METHOD_GATE <= delta <= METHOD_GATE:
        verdict = "METHOD_GO"
    elif METHOD_GATE < delta <= REGRESSION_GATE:
        verdict = (
            "PARTIAL_GO"
            if soup_valid < REFERENCE_BBAG_SOUP_PER_SEED[seed]
            else "WEAK_AMBIGUOUS"
        )
    else:
        verdict = "STOP"
    better_than_bbag = bool(soup_valid < REFERENCE_BBAG_SOUP_PER_SEED[seed])

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "promotion_seed": int(seed),
        "single_seed": True,
        "seed1_soup_valid": soup_valid,
        "seed1_raw_valid": raw_valid,
        "reference_bfull_seed1_soup": float(reference),
        "delta_seed1_soup_vs_bfull": float(delta),
        "reference_a2_seed1_soup": float(REFERENCE_A2_SOUP_PER_SEED[seed]),
        "reference_bbag_seed1_soup": float(REFERENCE_BBAG_SOUP_PER_SEED[seed]),
        "delta_seed1_soup_vs_a2": float(soup_valid - REFERENCE_A2_SOUP_PER_SEED[seed]),
        "delta_seed1_soup_vs_bbag": float(
            soup_valid - REFERENCE_BBAG_SOUP_PER_SEED[seed]
        ),
        "gates": {
            "strong_go": -STRONG_GATE,
            "method_go_abs": METHOD_GATE,
            "partial_go_upper": REGRESSION_GATE,
        },
        "s_collapse": s_collapse,
        "better_than_bbag": better_than_bbag,
        "verdict": verdict,
        "second_seed_authorized": bool(
            verdict in {"STRONG_GO", "METHOD_GO"}
        ),
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
        "soup": {str(seed): _maybe(f"soup_esb_seed{seed}") for seed in SEEDS},
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
    parser.add_argument("--tag", type=str, default="esb")
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
        print(json.dumps(_read_json(_esb_cache_paths()[2]), indent=2), flush=True)
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
