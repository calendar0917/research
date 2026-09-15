"""ZINC explicit-support structural composer (vocabulary-free).

Question
--------
The completed shared structural encoder (B-full) derives ``e_struct in R^16`` by
propagating hidden node states over the real rooted typed patch graph, and a
zero-training audit showed that its output is effectively rank-1.  The B-bag
control (same primitives, no adjacency at all) then measured how much of the
gain depends on real connectivity.

This experiment tests a **representation hypothesis**, not a wider encoder:

    Do not encode the whole patch implicitly through GNN hidden-state
    propagation.  Instead maintain explicit structural objects with exact
    atom / bond supports and learn only *which legal local objects are worth
    composing*.

    round 0 : atom primitives
    round 1 : legal atom+atom objects (one per real bond)
    round 2 : legal object+object objects (overlap or a real new connecting
              bond, union <= 4 atoms)
    pooling : activity-weighted mean / std + log count -> fusion -> e_struct

Core principle: **topology defines which compositions are legal; neural
parameters learn which legal compositions are useful.**  Supports are discrete,
exact and deterministic; only the scalar activity and the composition content
are learned.  The composer never reads ``struct_src`` / ``struct_dst`` and never
performs node-to-node message passing.

Everything downstream of ``e_struct`` is inherited bit-exactly from cell A /
B-full: radius-2 patch definition, relation system, pair descriptors, ``T=2``
weight-tied recurrent pair--centre, parent embedding, global/topology channels,
R width (334), small head and the frozen deterministic training protocol.

Matched reference points (frozen; never retrained here):

    A2      (no typed identity + capacity-matched shared adapter)
            2-seed Top-5 soup valid = 0.121914
    B-full  (shared connectivity-aware encoder)
            seed0 soup 0.119818, seed1 soup 0.118126, 2-seed 0.118972, 84,495 params
    B-bag   (connectivity-free DeepSets ablation; seed1 available)
            seed1 soup 0.120229

Official ZINC test is **never** loaded.  One promotion seed only (seed1 by
pre-registration), no width / round / temperature sweep.

Stages: ``params preprocess sanity train soup train_queue repro diagnostics
profile relations decide report``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_explicit_support_composer <stage>
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
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
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as sspe
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.explicit_support_composer import (
    ExplicitCandidateGraph,
    build_explicit_candidate_graph,
    expand_atom_support,
    expand_bond_support,
    parents,
)

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/explicit_support_composer"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"
CACHE_DIR = RESULTS_DIR / "cache"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "explicit_support_composer_v1"
CACHE_SCHEMA_VERSION = "explicit_support_candidates_v1"

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

# --- pre-registered explicit composer (one architecture; no sweep) -----------
# Object latent width is deliberately small (rank audit: the structural
# manifold is near rank-1).  Widths were chosen once so the composer budget
# matches B-full's 35,152-param encoder (see parameter_accounting.json).
COMPOSER_LATENT_DIM = 8
COMPOSER_EDGE_DIM = 8
COMPOSER_N_DEGREE_BINS = 6
COMPOSER_CANDIDATE_HIDDEN = 200
COMPOSER_FUSION_HIDDEN = 199
COMPOSER_MAX_CONN = 4
COMPOSER_MAX_UNION_SIZE = 4
COMPOSER_ROUNDS = 2

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
# candidate preprocessing / cache
# ---------------------------------------------------------------------------


def _exo_cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "explicit_candidates_train.pkl.gz",
        CACHE_DIR / "explicit_candidates_valid.pkl.gz",
        CACHE_DIR / "explicit_candidates_metadata.json",
    )


def _exo_to_plain(graph: ExplicitCandidateGraph) -> dict[str, Any]:
    return {
        "patch": graph.patch,
        "degree": graph.degree,
        "l1_u": graph.l1_u,
        "l1_v": graph.l1_v,
        "l1_bond": graph.l1_bond,
        "l1_patch": graph.l1_patch,
        "l2_a": graph.l2_a,
        "l2_b": graph.l2_b,
        "l2_patch": graph.l2_patch,
        "l2_overlap": graph.l2_overlap,
        "l2_conn": graph.l2_conn,
        "n_atoms": int(graph.n_atoms),
        "n_level1": int(graph.n_level1),
        "n_level2": int(graph.n_level2),
        "n_patches": int(graph.n_patches),
    }


def _exo_from_plain(item: Mapping[str, Any]) -> ExplicitCandidateGraph:
    return ExplicitCandidateGraph(**dict(item))


def explicit_candidates() -> tuple[
    list[ExplicitCandidateGraph], list[ExplicitCandidateGraph], dict[str, Any]
]:
    """Build (and cache) the explicit legal-composition search space."""
    import gzip

    train_path, valid_path, meta_path = _exo_cache_paths()
    if train_path.exists() and valid_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
            with gzip.open(train_path, "rb") as handle:
                train_items = pickle.load(handle)
            with gzip.open(valid_path, "rb") as handle:
                valid_items = pickle.load(handle)
            return (
                [_exo_from_plain(item) for item in train_items],
                [_exo_from_plain(item) for item in valid_items],
                meta,
            )

    started = time.perf_counter()
    train_bundle, valid_bundle, _base_meta = sspe.extract_records()
    train_graphs = [
        build_explicit_candidate_graph(
            graph.patch,
            graph.src,
            graph.dst,
            graph.bond,
            graph.edge_patch,
            graph.n_patches,
            max_union_size=COMPOSER_MAX_UNION_SIZE,
            max_conn=COMPOSER_MAX_CONN,
        )
        for graph in train_bundle["graphs"]
    ]
    valid_graphs = [
        build_explicit_candidate_graph(
            graph.patch,
            graph.src,
            graph.dst,
            graph.bond,
            graph.edge_patch,
            graph.n_patches,
            max_union_size=COMPOSER_MAX_UNION_SIZE,
            max_conn=COMPOSER_MAX_CONN,
        )
        for graph in valid_bundle["graphs"]
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(
            [_exo_to_plain(g) for g in train_graphs],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(
            [_exo_to_plain(g) for g in valid_graphs],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "patch_radius": PATCH_RADIUS,
        "max_union_size": COMPOSER_MAX_UNION_SIZE,
        "max_conn": COMPOSER_MAX_CONN,
        "rounds": COMPOSER_ROUNDS,
        "n_train_molecules": int(len(train_graphs)),
        "n_valid_molecules": int(len(valid_graphs)),
        "train": _candidate_stats(train_graphs),
        "valid": _candidate_stats(valid_graphs),
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    _write_json(meta_path, meta)
    print(
        f"[explicit] train={len(train_graphs)} valid={len(valid_graphs)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_graphs, valid_graphs, meta


def _candidate_stats(graphs: Sequence[ExplicitCandidateGraph]) -> dict[str, Any]:
    n_atoms = np.asarray([g.n_atoms for g in graphs], dtype=np.float64)
    n_l1 = np.asarray([g.n_level1 for g in graphs], dtype=np.float64)
    n_l2 = np.asarray([g.n_level2 for g in graphs], dtype=np.float64)
    n_patches = np.asarray([g.n_patches for g in graphs], dtype=np.float64)
    safe_patches = np.clip(n_patches, 1, None)
    return {
        "n_molecules": int(len(graphs)),
        "total_atoms": int(n_atoms.sum()),
        "total_patches": int(n_patches.sum()),
        "total_level1_objects": int(n_l1.sum()),
        "total_level2_objects": int(n_l2.sum()),
        "mean_atoms_per_patch": float(n_atoms.sum() / safe_patches.sum()),
        "mean_level1_per_patch": float(n_l1.sum() / safe_patches.sum()),
        "mean_level2_per_patch": float(n_l2.sum() / safe_patches.sum()),
        "max_level2_per_molecule": int(n_l2.max()) if n_l2.size else 0,
    }


def _attach_explicit_tensors(
    encoded: Sequence[Data], candidates: Sequence[ExplicitCandidateGraph]
) -> None:
    for data, graph in zip(encoded, candidates):
        if int(data.num_nodes) != graph.n_patches:
            raise RuntimeError("encoded / candidate patch count mismatch")
        if int(data.struct_atom.numel()) != graph.n_atoms:
            raise RuntimeError("encoded / candidate atom count mismatch")
        data.struct_degree = torch.from_numpy(graph.degree)
        data.exo_l1_u = torch.from_numpy(graph.l1_u)
        data.exo_l1_v = torch.from_numpy(graph.l1_v)
        data.exo_l1_bond = torch.from_numpy(graph.l1_bond)
        data.exo_l1_patch = torch.from_numpy(graph.l1_patch)
        data.exo_l2_a = torch.from_numpy(graph.l2_a)
        data.exo_l2_b = torch.from_numpy(graph.l2_b)
        data.exo_l2_patch = torch.from_numpy(graph.l2_patch)
        data.exo_l2_overlap = torch.from_numpy(graph.l2_overlap)
        data.exo_l2_conn = torch.from_numpy(graph.l2_conn)


def build_encoded_records():
    """Return (train_data, valid_data, audit) with explicit-object tensors."""
    train_bundle, valid_bundle, _meta = sspe.extract_records()
    config = sspe._v4_base_config()
    config["model"]["device"] = "cpu"
    enc_train, enc_valid, audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    sspe._attach_struct_tensors(enc_train, train_bundle["graphs"])
    sspe._attach_struct_tensors(enc_valid, valid_bundle["graphs"])
    train_candidates, valid_candidates, _cmeta = explicit_candidates()
    _attach_explicit_tensors(enc_train, train_candidates)
    _attach_explicit_tensors(enc_valid, valid_candidates)
    audit = dict(audit)
    audit["patch_representation"] = "explicit_composer"
    audit["explicit_composer"] = {
        "latent_dim": COMPOSER_LATENT_DIM,
        "edge_dim": COMPOSER_EDGE_DIM,
        "n_degree_bins": COMPOSER_N_DEGREE_BINS,
        "candidate_hidden": COMPOSER_CANDIDATE_HIDDEN,
        "fusion_hidden": COMPOSER_FUSION_HIDDEN,
        "output_dim": TOKEN_WIDTH,
        "max_conn": COMPOSER_MAX_CONN,
        "max_union_size": COMPOSER_MAX_UNION_SIZE,
        "rounds": COMPOSER_ROUNDS,
    }
    audit["official_test_loaded"] = False
    return enc_train, enc_valid, audit


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_candidate(seed: int = 0) -> torch.nn.Module:
    """Cell-A geometry with the typed lookup replaced by the composer.

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
            patch_representation="explicit_composer",
            composer_latent_dim=COMPOSER_LATENT_DIM,
            composer_edge_dim=COMPOSER_EDGE_DIM,
            composer_n_degree_bins=COMPOSER_N_DEGREE_BINS,
            composer_candidate_hidden=COMPOSER_CANDIDATE_HIDDEN,
            composer_fusion_hidden=COMPOSER_FUSION_HIDDEN,
            composer_max_conn=COMPOSER_MAX_CONN,
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


def exo_collate(data_list: Sequence[Data]) -> Any:
    """Batch explicit-object tensors with the correct graph offsets.

    Level-0 object ids are node offsets; level-1 object ids are bond offsets;
    patch ids are patch offsets.  ``struct_src`` / ``struct_dst`` are offset by
    ``sspe.struct_collate`` but the composer never reads them.
    """
    batch = sspe.struct_collate(data_list)
    n_offset = 0
    b_offset = 0
    patch_offset = 0
    l1_u: list[torch.Tensor] = []
    l1_v: list[torch.Tensor] = []
    l1_bond: list[torch.Tensor] = []
    l1_patch: list[torch.Tensor] = []
    l2_a: list[torch.Tensor] = []
    l2_b: list[torch.Tensor] = []
    l2_patch: list[torch.Tensor] = []
    l2_overlap: list[torch.Tensor] = []
    l2_conn: list[torch.Tensor] = []
    for data in data_list:
        n_patches = int(data.num_nodes)
        n_atoms = int(data.struct_atom.numel())
        n_bonds = int(data.exo_l1_u.numel())
        l1_u.append(data.exo_l1_u + n_offset)
        l1_v.append(data.exo_l1_v + n_offset)
        l1_bond.append(data.exo_l1_bond)
        l1_patch.append(data.exo_l1_patch + patch_offset)
        l2_a.append(data.exo_l2_a + b_offset)
        l2_b.append(data.exo_l2_b + b_offset)
        l2_patch.append(data.exo_l2_patch + patch_offset)
        l2_overlap.append(data.exo_l2_overlap)
        conn = data.exo_l2_conn.clone()
        mask = conn >= 0
        conn[mask] = conn[mask] + b_offset
        l2_conn.append(conn)
        n_offset += n_atoms
        b_offset += n_bonds
        patch_offset += n_patches
    batch.exo_l1_u = (
        torch.cat(l1_u) if l1_u else torch.zeros(0, dtype=torch.long)
    )
    batch.exo_l1_v = (
        torch.cat(l1_v) if l1_v else torch.zeros(0, dtype=torch.long)
    )
    batch.exo_l1_bond = (
        torch.cat(l1_bond) if l1_bond else torch.zeros(0, dtype=torch.long)
    )
    batch.exo_l1_patch = (
        torch.cat(l1_patch) if l1_patch else torch.zeros(0, dtype=torch.long)
    )
    batch.exo_l2_a = torch.cat(l2_a) if l2_a else torch.zeros(0, dtype=torch.long)
    batch.exo_l2_b = torch.cat(l2_b) if l2_b else torch.zeros(0, dtype=torch.long)
    batch.exo_l2_patch = (
        torch.cat(l2_patch) if l2_patch else torch.zeros(0, dtype=torch.long)
    )
    batch.exo_l2_overlap = (
        torch.cat(l2_overlap) if l2_overlap else torch.zeros(0, dtype=torch.long)
    )
    batch.exo_l2_conn = (
        torch.cat(l2_conn)
        if l2_conn
        else torch.zeros((0, COMPOSER_MAX_CONN), dtype=torch.long)
    )
    return batch


def _make_exo_loader(
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
        collate_fn=exo_collate,
    )


def _selection_loader(valid_data: Sequence[Data]):
    return _make_exo_loader(valid_data, 128, False, 0)


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

    composer_blocks = {
        "primitive_embeddings": int(
            _n_params(encoder.atom_embedding)
            + _n_params(encoder.root_embedding)
            + _n_params(encoder.distance_embedding)
            + _n_params(encoder.degree_embedding)
            + _n_params(encoder.bond_embedding)
        ),
        "level1_scorer": int(_n_params(encoder.level1_score)),
        "level1_content": int(_n_params(encoder.level1_content)),
        "level2_scorer": int(_n_params(encoder.level2_score)),
        "level2_content": int(_n_params(encoder.level2_content)),
        "object_fusion": int(_n_params(encoder.fusion)),
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
        "composer_params": int(encoder_params),
        "composer_params_minus_bfull_encoder": int(
            encoder_params - REFERENCE_BFULL_ENCODER_PARAMS
        ),
        "composer_blocks": composer_blocks,
        "composer_dimensions": {
            "latent_dim": int(encoder.latent_dim),
            "edge_dim": int(encoder.edge_dim),
            "candidate_hidden": int(encoder.candidate_hidden),
            "fusion_hidden": int(encoder.fusion_hidden),
            "rounds": int(encoder.rounds),
            "output_dim": int(encoder.output_dim),
        },
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


class _ExoBatch:
    """Minimal explicit-object container for unit / adversarial checks."""

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
        l2_a,
        l2_b,
        l2_patch,
        l2_overlap,
        l2_conn,
        n_patches: int,
    ) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_degree = degree
        self.struct_patch = patch
        self.exo_l1_u = l1_u
        self.exo_l1_v = l1_v
        self.exo_l1_bond = l1_bond
        self.exo_l1_patch = l1_patch
        self.exo_l2_a = l2_a
        self.exo_l2_b = l2_b
        self.exo_l2_patch = l2_patch
        self.exo_l2_overlap = l2_overlap
        self.exo_l2_conn = l2_conn
        self.struct_n_patches = int(n_patches)


def _exo_batch_from_graph(graph: ExplicitCandidateGraph) -> _ExoBatch:
    n_atoms = int(graph.n_atoms)
    root = torch.zeros(n_atoms, dtype=torch.long)
    if n_atoms:
        root[0] = 1
    return _ExoBatch(
        atom=torch.zeros(n_atoms, dtype=torch.long),
        root=root,
        dist=torch.zeros(n_atoms, dtype=torch.long),
        degree=torch.from_numpy(graph.degree),
        patch=torch.from_numpy(graph.patch),
        l1_u=torch.from_numpy(graph.l1_u),
        l1_v=torch.from_numpy(graph.l1_v),
        l1_bond=torch.from_numpy(graph.l1_bond),
        l1_patch=torch.from_numpy(graph.l1_patch),
        l2_a=torch.from_numpy(graph.l2_a),
        l2_b=torch.from_numpy(graph.l2_b),
        l2_patch=torch.from_numpy(graph.l2_patch),
        l2_overlap=torch.from_numpy(graph.l2_overlap),
        l2_conn=torch.from_numpy(graph.l2_conn),
        n_patches=graph.n_patches,
    )


def _adversarial_connectivity_pair() -> tuple[Any, Any, tuple, tuple]:
    """Same atom/root/distance/degree/bond multiset, different connectivity.

    Both are 6-node rooted trees on atom type 0 with bond type 0, the root at
    node 0, the SAME rooted distance profile ``[0,1,1,2,2,2]`` and the SAME
    degree multiset ``{3,2,2,1,1,1}``.  Only the edge endpoints differ, so a
    permutation-invariant primitive-bag encoder cannot tell them apart while a
    connectivity-defined composition search space can.
    """
    n_nodes = 6
    root_node = 0
    patch = np.zeros(n_nodes, dtype=np.int64)
    # A: 0-1, 0-2, 1-3, 2-4, 2-5   (the degree-3 node is atom 2)
    undirected_a = [(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)]
    # B: 0-1, 0-2, 1-3, 1-4, 2-5   (the degree-3 node is atom 1)
    undirected_b = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]

    def build(undirected):
        both = undirected + [(v, u) for u, v in undirected]
        src = np.asarray([u for u, _ in both], dtype=np.int64)
        dst = np.asarray([v for _, v in both], dtype=np.int64)
        bond = np.zeros(len(both), dtype=np.int64)
        edge_patch = np.zeros(len(both), dtype=np.int64)
        graph = build_explicit_candidate_graph(
            patch,
            src,
            dst,
            bond,
            edge_patch,
            1,
            max_union_size=COMPOSER_MAX_UNION_SIZE,
            max_conn=COMPOSER_MAX_CONN,
        )
        adjacency: dict[int, set[int]] = {node: set() for node in range(n_nodes)}
        for left, right in undirected:
            adjacency[left].add(right)
            adjacency[right].add(left)
        distances = {root_node: 0}
        queue = [root_node]
        while queue:
            node = queue.pop(0)
            for neighbour in sorted(adjacency[node]):
                if neighbour not in distances:
                    distances[neighbour] = distances[node] + 1
                    queue.append(neighbour)
        dist = np.asarray([distances[node] for node in range(n_nodes)], dtype=np.int64)
        root_flag = np.zeros(n_nodes, dtype=np.int64)
        root_flag[root_node] = 1
        atom = np.zeros(n_nodes, dtype=np.int64)
        signature = (
            tuple(
                sorted(
                    zip(
                        atom.tolist(),
                        root_flag.tolist(),
                        dist.tolist(),
                        graph.degree.tolist(),
                    )
                )
            ),
            tuple(sorted(bond.tolist())),
        )
        batch = _ExoBatch(
            atom=torch.from_numpy(atom),
            root=torch.from_numpy(root_flag),
            dist=torch.from_numpy(dist),
            degree=torch.from_numpy(graph.degree),
            patch=torch.from_numpy(graph.patch),
            l1_u=torch.from_numpy(graph.l1_u),
            l1_v=torch.from_numpy(graph.l1_v),
            l1_bond=torch.from_numpy(graph.l1_bond),
            l1_patch=torch.from_numpy(graph.l1_patch),
            l2_a=torch.from_numpy(graph.l2_a),
            l2_b=torch.from_numpy(graph.l2_b),
            l2_patch=torch.from_numpy(graph.l2_patch),
            l2_overlap=torch.from_numpy(graph.l2_overlap),
            l2_conn=torch.from_numpy(graph.l2_conn),
            n_patches=graph.n_patches,
        )
        return graph, batch, signature

    graph_a, batch_a, signature_a = build(undirected_a)
    graph_b, batch_b, signature_b = build(undirected_b)
    return graph_a, graph_b, (batch_a, signature_a), (batch_b, signature_b)


def _typed_bond_path_pair() -> tuple[Any, Any]:
    """Same primitives and same topology, different bond-type placement.

    Both are the 3-node path ``0-1-2`` with atom type 0 everywhere, root at
    node 0, identical distance profile ``[0,1,2]``/degree profile and the same
    bond-type multiset ``{0,1}``.  Only *which* edge carries the type-1 bond
    differs, so a primitive-bag encoder is invariant while any encoder that
    actually composes typed local relations moves.
    """
    undirected = [(0, 1), (1, 2)]
    dist = np.asarray([0, 1, 2], dtype=np.int64)
    root = np.asarray([1, 0, 0], dtype=np.int64)
    atom = np.zeros(3, dtype=np.int64)

    def build(type_by_edge):
        both = undirected + [(v, u) for u, v in undirected]
        src = np.asarray([u for u, _ in both], dtype=np.int64)
        dst = np.asarray([v for _, v in both], dtype=np.int64)
        bond = np.asarray(
            [type_by_edge[(min(u, v), max(u, v))] for u, v in both],
            dtype=np.int64,
        )
        edge_patch = np.zeros(len(both), dtype=np.int64)
        graph = build_explicit_candidate_graph(
            np.zeros(3, dtype=np.int64),
            src,
            dst,
            bond,
            edge_patch,
            1,
            max_union_size=COMPOSER_MAX_UNION_SIZE,
            max_conn=COMPOSER_MAX_CONN,
        )
        batch = _ExoBatch(
            atom=torch.from_numpy(atom),
            root=torch.from_numpy(root),
            dist=torch.from_numpy(dist),
            degree=torch.from_numpy(graph.degree),
            patch=torch.from_numpy(graph.patch),
            l1_u=torch.from_numpy(graph.l1_u),
            l1_v=torch.from_numpy(graph.l1_v),
            l1_bond=torch.from_numpy(graph.l1_bond),
            l1_patch=torch.from_numpy(graph.l1_patch),
            l2_a=torch.from_numpy(graph.l2_a),
            l2_b=torch.from_numpy(graph.l2_b),
            l2_patch=torch.from_numpy(graph.l2_patch),
            l2_overlap=torch.from_numpy(graph.l2_overlap),
            l2_conn=torch.from_numpy(graph.l2_conn),
            n_patches=graph.n_patches,
        )
        return graph, batch

    graph_a, batch_a = build({(0, 1): 1, (1, 2): 0})
    graph_b, batch_b = build({(0, 1): 0, (1, 2): 1})
    return batch_a, batch_b


def _graph_signature(graph: ExplicitCandidateGraph) -> tuple:
    return (
        tuple(graph.l1_u.tolist()),
        tuple(graph.l1_v.tolist()),
        tuple(graph.l2_a.tolist()),
        tuple(graph.l2_b.tolist()),
        tuple(graph.l2_overlap.tolist()),
        tuple(graph.l2_conn.ravel().tolist()),
    )


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    train_data, valid_data, _audit = build_encoded_records()
    loader = _make_exo_loader(list(valid_data)[:128], 128, False, 0)
    batch = next(iter(loader)).to(device)

    model = build_candidate(0).to(device)
    baseline = cd.build_cell(CELL, 0).to(device)
    encoder = model.structural_encoder
    checks: dict[str, bool] = {}

    # (13) no vocab-sized typed lookup.
    typed_state = [k for k in model.state_dict() if "typed_embedding" in k]
    typed_named = [k for k, _ in model.named_parameters() if "typed_embedding" in k]
    checks["no_vocab_sized_typed_embedding_weight"] = (
        len(typed_state) == 0 and len(typed_named) == 0
    )
    checks["explicit_composer_present"] = bool(
        getattr(encoder, "kind", "") == "explicit_support_composer"
    )
    # (10) unseen certificate needs no vocabulary: zeroing typed_token is a no-op.
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
    # (14) h/q/T preserved.
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
    # (13) output width.
    checks["output_width_16"] = bool(int(encoder.output_dim) == 16)
    checks["explicit_rounds_2"] = bool(int(encoder.rounds) == 2)
    # (1)/(2)/(3)/(4)/(5) support + provenance exactness on the real batch.
    train_candidates, valid_candidates, _ = explicit_candidates()
    graph = valid_candidates[0]
    support_ok = True
    bond_ok = True
    union_ok = True
    provenance_ok = True
    for index in range(graph.n_level1):
        atoms = expand_atom_support(graph, 1, index)
        if atoms != frozenset({int(graph.l1_u[index]), int(graph.l1_v[index])}):
            support_ok = False
        if expand_bond_support(graph, 1, index) != frozenset({index}):
            bond_ok = False
        if parents(graph, 1, index) != (
            (0, int(graph.l1_u[index])),
            (0, int(graph.l1_v[index])),
        ):
            provenance_ok = False
    for index in range(graph.n_level2):
        left = expand_atom_support(graph, 1, int(graph.l2_a[index]))
        right = expand_atom_support(graph, 1, int(graph.l2_b[index]))
        if expand_atom_support(graph, 2, index) != left | right:
            union_ok = False
        left_b = expand_bond_support(graph, 1, int(graph.l2_a[index]))
        right_b = expand_bond_support(graph, 1, int(graph.l2_b[index]))
        connecting = {
            int(value) for value in graph.l2_conn[index].tolist() if value >= 0
        }
        if expand_bond_support(graph, 2, index) != left_b | right_b | connecting:
            bond_ok = False
        overlap = len(left & right)
        if overlap != int(graph.l2_overlap[index]):
            union_ok = False
        if int(graph.l2_overlap[index]) != len(left & right):
            union_ok = False
    checks["atom_support_recoverable"] = bool(support_ok)
    checks["bond_support_recoverable"] = bool(bond_ok)
    checks["composition_support_is_parent_union"] = bool(union_ok)
    checks["provenance_chain_to_primitives"] = bool(provenance_ok)
    # (6) only topology-legal pairs compose.
    checks["only_legal_pairs"] = bool(
        np.all(graph.l2_overlap >= 0)
        and np.all(
            (graph.l2_overlap > 0)
            | (graph.l2_conn >= 0).any(axis=1)
        )
    )
    # (11)/(12) overlap allowed + containment deterministically computable.
    overlap_present = bool(np.any(graph.l2_overlap > 0))
    checks["overlap_allowed_and_recorded"] = bool(
        overlap_present and np.all(graph.l2_overlap >= 0)
    )
    containment_ok = True
    contained_any = False
    for index in range(graph.n_level2):
        child = expand_atom_support(graph, 2, index)
        left = expand_atom_support(graph, 1, int(graph.l2_a[index]))
        right = expand_atom_support(graph, 1, int(graph.l2_b[index]))
        # containment is a deterministic function of the exact supports
        if not (left <= child and right <= child):
            containment_ok = False
        if left < child or right < child:
            contained_any = True
    checks["containment_deterministic"] = bool(containment_ok)
    checks["containment_relation_present"] = bool(contained_any)

    # (7) modifying connectivity changes the legal candidate set (adversarial).
    graph_a, graph_b, pair_a, pair_b = _adversarial_connectivity_pair()
    batch_a, signature_a = pair_a
    batch_b, signature_b = pair_b
    same_primitives = bool(signature_a == signature_b)
    checks["adversarial_pair_primitives_identical"] = same_primitives
    signatures_differ = _graph_signature(graph_a) != _graph_signature(graph_b)
    checks["connectivity_changes_candidate_set"] = bool(signatures_differ)

    encoder.eval()
    with torch.no_grad():
        e_a = encoder(batch_a)
        e_b = encoder(batch_b)
    tree_pair_delta = float((e_a - e_b).abs().max())
    # Typed connectivity sensitivity: same primitives AND same topology, but a
    # different typed bond placement.  The candidate SET is identical, so only
    # an encoder that actually composes typed local relations can move.
    typed_a, typed_b = _typed_bond_path_pair()
    with torch.no_grad():
        et_a = encoder(typed_a)
        et_b = encoder(typed_b)
    typed_pair_delta = float((et_a - et_b).abs().max())
    checks["typed_connectivity_changes_patch_output"] = bool(typed_pair_delta > 1.0e-6)

    # (8) no GNN / message passing: the composer must not read src/dst.
    class _NoEndpoints:
        """Wrapper that raises on any access to edge endpoints."""

        def __init__(self, batch):
            self._batch = batch

        def __getattr__(self, name):
            if name in {"struct_src", "struct_dst"}:
                raise AttributeError(name)
            return getattr(self._batch, name)

    try:
        with torch.no_grad():
            _ = encoder(_NoEndpoints(batch))
        checks["forward_without_edge_endpoints"] = True
    except Exception:
        checks["forward_without_edge_endpoints"] = False

    # (9) parameter budget vs B-full.
    total = _n_params(model)
    checks["params_in_range"] = bool(PARAM_LOWER <= total <= PARAM_UPPER)
    checks["params_match_bfull_within_1pct"] = bool(
        abs(total - REFERENCE_BFULL_PARAMS) / REFERENCE_BFULL_PARAMS
        <= BUDGET_MATCH_TOLERANCE
    )

    # (15) forward / backward finite.
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
        "candidate_params": int(total),
        "bfull_params": int(REFERENCE_BFULL_PARAMS),
        "composer_params": int(_n_params(encoder)),
        "adversarial_pair": {
            "connectivity_changed_only": bool(same_primitives),
            "candidate_signature_differs": bool(signatures_differ),
            "tree_pair_pooled_delta": float(tree_pair_delta),
            "typed_pair_pooled_delta": float(typed_pair_delta),
        },
        "split": "first 128 official-valid molecules",
        "environment": _environment_fingerprint("cpu"),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"explicit composer sanity failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def train_seed(
    seed: int,
    device: str = "cpu",
    tag: str = "exco",
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
    loader = _make_exo_loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = _make_exo_loader(
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
        "patch_representation": "explicit_composer",
        "composer_params": int(_n_params(model.structural_encoder)),
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


def soup_seed(seed: int, tag: str = "exco") -> dict[str, Any]:
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


def train_queue(seeds: Sequence[int], device: str, tag: str = "exco") -> None:
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


def _patch_atom_sets(graph: ExplicitCandidateGraph) -> list[set[int]]:
    """Exact atom set of every patch (including isolated atoms)."""
    sets: list[set[int]] = [set() for _ in range(graph.n_patches)]
    for atom in range(graph.n_atoms):
        sets[int(graph.patch[atom])].add(int(atom))
    return sets


def _support_stats_for_details(
    graph: ExplicitCandidateGraph, details: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-molecule structural diagnostics of the selected objects.

    The ``0.5`` activity threshold is used **for diagnostics only**; it never
    changes the forward pass.
    """
    all_patch = details["all_patch"].cpu().numpy()
    all_level = details["all_level"].cpu().numpy()
    selected = details["selected"].cpu().numpy() > 0.5
    n_level1 = int(graph.n_level1)
    n_level2 = int(graph.n_level2)
    n_atoms = int(graph.n_atoms)
    activity1 = details["activity1"].cpu().numpy()
    activity2 = details["activity2"].cpu().numpy()

    round1_counts = np.bincount(graph.l1_patch, minlength=graph.n_patches)
    round2_counts = np.bincount(graph.l2_patch, minlength=graph.n_patches)
    round1_activity: list[float] = []
    for p in range(graph.n_patches):
        rows = np.nonzero(graph.l1_patch == p)[0]
        round1_activity.append(
            float(activity1[rows].mean()) if rows.size else float("nan")
        )
    round2_activity: list[float] = []
    for p in range(graph.n_patches):
        rows = np.nonzero(graph.l2_patch == p)[0]
        round2_activity.append(
            float(activity2[rows].mean()) if rows.size else float("nan")
        )

    selected_by_patch: list[list[tuple[int, int]]] = [
        [] for _ in range(graph.n_patches)
    ]
    for index in range(int(all_patch.shape[0])):
        if not selected[index]:
            continue
        level = int(all_level[index])
        if level == 0:
            local = index
        elif level == 1:
            local = index - n_atoms
        else:
            local = index - n_atoms - n_level1
        selected_by_patch[int(all_patch[index])].append((level, local))

    patch_atoms = _patch_atom_sets(graph)
    support_sizes: list[int] = []
    atom_coverage: list[float] = []
    bond_coverage: list[float] = []
    active_total = 0
    selected_total = 0
    overlap_pairs = 0
    containment_pairs = 0
    total_pairs = 0
    duplicates = 0
    singletons = 0
    whole_patch = 0
    for p in range(graph.n_patches):
        objects = selected_by_patch[p]
        if not objects:
            continue
        supports: list[frozenset[int]] = []
        bond_supports: list[frozenset[int]] = []
        for level, local in objects:
            supports.append(expand_atom_support(graph, level, local))
            bond_supports.append(expand_bond_support(graph, level, local))
            if level == 1:
                value = float(activity1[local])
            elif level == 2:
                value = float(activity2[local])
            else:
                value = 1.0
            if value >= 0.5:
                active_total += 1
        selected_total += len(supports)
        for support in supports:
            support_sizes.append(int(len(support)))
            if len(support) == 1:
                singletons += 1
            if patch_atoms[p] and set(support) == patch_atoms[p]:
                whole_patch += 1
        if patch_atoms[p]:
            union = set()
            for support in supports:
                union |= set(support)
            atom_coverage.append(
                float(len(union & patch_atoms[p]) / len(patch_atoms[p]))
            )
        bonds_here = {
            index
            for index in range(n_level1)
            if int(graph.l1_patch[index]) == p
        }
        if bonds_here:
            union_b = set()
            for support in bond_supports:
                union_b |= set(support)
            bond_coverage.append(
                float(len(union_b & bonds_here) / len(bonds_here))
            )
        seen: set[frozenset[int]] = set()
        for support in supports:
            if support in seen:
                duplicates += 1
            seen.add(support)
        for i in range(len(supports)):
            for j in range(i + 1, len(supports)):
                total_pairs += 1
                if supports[i] & supports[j]:
                    overlap_pairs += 1
                if supports[i] < supports[j] or supports[j] < supports[i]:
                    containment_pairs += 1

    return {
        "patch_count": int(graph.n_patches),
        "round1_candidate_total": int(round1_counts.sum()),
        "round2_candidate_total": int(round2_counts.sum()),
        "mean_round1_candidates": float(round1_counts.mean())
        if graph.n_patches
        else 0.0,
        "mean_round2_candidates": float(round2_counts.mean())
        if graph.n_patches
        else 0.0,
        "mean_round1_activity": float(np.nanmean(round1_activity))
        if round1_activity
        else 0.0,
        "mean_round2_activity": float(np.nanmean(round2_activity))
        if round2_activity
        else 0.0,
        "active_objects": int(active_total),
        "selected_objects": int(selected_total),
        "support_size_histogram": np.bincount(
            np.asarray(support_sizes, dtype=np.int64)
        ).tolist()
        if support_sizes
        else [],
        "atom_coverage": float(np.mean(atom_coverage)) if atom_coverage else 0.0,
        "bond_coverage": float(np.mean(bond_coverage)) if bond_coverage else 0.0,
        "overlap_pairs": int(overlap_pairs),
        "containment_pairs": int(containment_pairs),
        "total_pairs": int(total_pairs),
        "duplicate_supports": int(duplicates),
        "singleton_supports": int(singletons),
        "whole_patch_supports": int(whole_patch),
    }


def _collapse_flags(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Section-13 collapse checks (diagnostic only; never affects forward)."""
    total = max(int(stats["selected_objects"]), 1)
    singleton_fraction = float(stats["singleton_supports"]) / total
    whole_fraction = float(stats["whole_patch_supports"]) / total
    duplicate_fraction = float(stats["duplicate_supports"]) / total
    collapse_a = singleton_fraction > 0.9
    collapse_b = whole_fraction > 0.9
    collapse_c = bool(stats["activity_std"] < 0.01)
    collapse_d = duplicate_fraction > 0.5
    return {
        "collapse_A_singleton_atoms": bool(collapse_a),
        "collapse_B_whole_patch": bool(collapse_b),
        "collapse_C_constant_activity": bool(collapse_c),
        "collapse_D_duplicate_supports": bool(collapse_d),
        "singleton_support_fraction": float(singleton_fraction),
        "whole_patch_support_fraction": float(whole_fraction),
        "duplicate_support_rate": float(duplicate_fraction),
        "any_collapse": bool(collapse_a or collapse_b or collapse_c or collapse_d),
    }


def _aggregate_diagnostics(per_molecule: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _mean(key: str) -> float:
        values = [float(row[key]) for row in per_molecule]
        return float(np.mean(values)) if values else 0.0

    histogram = np.zeros(1, dtype=np.int64)
    for row in per_molecule:
        arr = np.asarray(row["support_size_histogram"], dtype=np.int64)
        if arr.size > histogram.size:
            histogram = np.concatenate(
                [histogram, np.zeros(arr.size - histogram.size, dtype=np.int64)]
            )
        histogram[: arr.size] += arr

    total_pairs = int(sum(int(row["total_pairs"]) for row in per_molecule))
    overlap_pairs = int(sum(int(row["overlap_pairs"]) for row in per_molecule))
    containment_pairs = int(
        sum(int(row["containment_pairs"]) for row in per_molecule)
    )
    round1_total = int(sum(int(row["round1_candidate_total"]) for row in per_molecule))
    round2_total = int(sum(int(row["round2_candidate_total"]) for row in per_molecule))
    return {
        "n_molecules": int(len(per_molecule)),
        "mean_round1_candidates_per_patch": _mean("mean_round1_candidates"),
        "mean_round2_candidates_per_patch": _mean("mean_round2_candidates"),
        "mean_round1_activity": _mean("mean_round1_activity"),
        "mean_round2_activity": _mean("mean_round2_activity"),
        "round1_candidate_total": int(round1_total),
        "round2_candidate_total": int(round2_total),
        "mean_selected_objects_per_molecule": _mean("selected_objects"),
        "mean_active_objects_per_molecule": _mean("active_objects"),
        "mean_atom_coverage": _mean("atom_coverage"),
        "mean_bond_coverage": _mean("bond_coverage"),
        "pairwise_overlap_rate": float(overlap_pairs / total_pairs)
        if total_pairs
        else 0.0,
        "containment_rate": float(containment_pairs / total_pairs)
        if total_pairs
        else 0.0,
        "total_object_pairs": int(total_pairs),
        "support_size_histogram": histogram.tolist(),
        "duplicate_supports_total": int(
            sum(int(row["duplicate_supports"]) for row in per_molecule)
        ),
        "singleton_supports_total": int(
            sum(int(row["singleton_supports"]) for row in per_molecule)
        ),
        "whole_patch_supports_total": int(
            sum(int(row["whole_patch_supports"]) for row in per_molecule)
        ),
        "selected_objects_total": int(
            sum(int(row["selected_objects"]) for row in per_molecule)
        ),
    }


def diagnostics(tag: str = "exco", n_molecules: int = 256) -> dict[str, Any]:
    device = torch.device("cpu")
    _train, valid_data, _audit = build_encoded_records()
    _train_candidates, valid_candidates, _meta = explicit_candidates()
    subset_data = list(valid_data)[: int(n_molecules)]
    subset_graphs = list(valid_candidates)[: int(n_molecules)]

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "patch_representation": "explicit_composer",
        "n_molecules": int(len(subset_data)),
        "per_seed": {},
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
        encoder = model.structural_encoder
        encoder.eval()
        per_molecule: list[dict[str, Any]] = []
        activity_values: list[np.ndarray] = []
        encoder_outputs: list[np.ndarray] = []
        for data, graph in zip(subset_data, subset_graphs):
            batch = exo_collate([data])
            with torch.no_grad():
                details = encoder(batch, return_details=True)
            per_molecule.append(_support_stats_for_details(graph, details))
            if details["activity1"].numel():
                activity_values.append(details["activity1"].cpu().numpy())
            if details["activity2"].numel():
                activity_values.append(details["activity2"].cpu().numpy())
            encoder_outputs.append(details["e_struct"].cpu().numpy())
        activity = np.concatenate(activity_values) if activity_values else np.zeros(1)
        encoder_matrix = np.concatenate(encoder_outputs, axis=0)
        stats = _aggregate_diagnostics(per_molecule)
        stats["activity_mean"] = float(activity.mean())
        stats["activity_std"] = float(activity.std())
        stats["activity_min"] = float(activity.min())
        stats["activity_max"] = float(activity.max())
        collapse = _collapse_flags(stats)
        payload["per_seed"][str(seed)] = {
            "composer_output_rank": _effective_rank(encoder_matrix),
            "n_composer_samples": int(encoder_matrix.shape[0]),
            "structural_diagnostics": stats,
            "collapse": collapse,
        }
    _write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# relation construction preview (zero training, section 18)
# ---------------------------------------------------------------------------


def _patch_atom_graph(graph: ExplicitCandidateGraph, patch: int):
    """Adjacency (atom-level BFS distance) inside one patch."""
    adjacency: dict[int, set[int]] = {}
    for index in range(graph.n_level1):
        if int(graph.l1_patch[index]) != int(patch):
            continue
        u = int(graph.l1_u[index])
        v = int(graph.l1_v[index])
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    return adjacency


def _bfs_distance(adjacency: Mapping[int, set[int]], source: int) -> dict[int, int]:
    distances = {source: 0}
    queue = [source]
    while queue:
        node = queue.pop(0)
        for neighbour in adjacency.get(node, ()):  # pragma: no cover - simple
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def relations_preview(n_molecules: int = 128) -> dict[str, Any]:
    """Zero-training preview: can learned objects yield explicit relations?"""
    _train, valid_data, _audit = build_encoded_records()
    _train_candidates, valid_candidates, _meta = explicit_candidates()
    subset_graphs = list(valid_candidates)[: int(n_molecules)]
    n_pairs = 0
    n_overlap = 0
    n_containment = 0
    n_adjacent = 0
    n_boundary = 0
    distance_hist: dict[int, int] = {}
    support_sizes: list[int] = []
    deterministic = True
    for graph in subset_graphs:
        for patch in range(graph.n_patches):
            level1 = [
                index
                for index in range(graph.n_level1)
                if int(graph.l1_patch[index]) == patch
            ]
            level2 = [
                index
                for index in range(graph.n_level2)
                if int(graph.l2_patch[index]) == patch
            ]
            objects: list[tuple[int, int]] = [(1, index) for index in level1]
            objects += [(2, index) for index in level2]
            supports = [expand_atom_support(graph, level, index) for level, index in objects]
            adjacency = _patch_atom_graph(graph, patch)
            for support in supports:
                support_sizes.append(len(support))
            for i in range(len(supports)):
                for j in range(i + 1, len(supports)):
                    n_pairs += 1
                    a = supports[i]
                    b = supports[j]
                    if a & b:
                        n_overlap += 1
                    if a < b or b < a:
                        n_containment += 1
                    adjacent = False
                    best = None
                    for left in a:
                        distances = _bfs_distance(adjacency, left)
                        for right in b:
                            if right in distances:
                                adjacent = True
                                value = distances[right]
                                if best is None or value < best:
                                    best = value
                    if adjacent:
                        # ``best`` is the minimum atom-level graph distance
                        # between the two supports: 0 == overlap, 1 == directly
                        # bonded (boundary sharing), >=2 == merely connected.
                        if best is not None and best <= 1:
                            n_adjacent += 1
                            if best == 1 and not (a & b):
                                n_boundary += 1
                        distance_hist[best] = distance_hist.get(best, 0) + 1
                    # determinism re-check: recompute support and compare
                    if (
                        expand_atom_support(graph, objects[i][0], objects[i][1]) != a
                        or expand_atom_support(graph, objects[j][0], objects[j][1]) != b
                    ):
                        deterministic = False
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_molecules": int(n_molecules),
        "n_object_pairs": int(n_pairs),
        "overlap_rate": float(n_overlap / n_pairs) if n_pairs else 0.0,
        "containment_rate": float(n_containment / n_pairs) if n_pairs else 0.0,
        "adjacency_rate": float(n_adjacent / n_pairs) if n_pairs else 0.0,
        "boundary_sharing_rate": float(n_boundary / n_pairs) if n_pairs else 0.0,
        "graph_distance_histogram": {
            str(key): int(value) for key, value in sorted(distance_hist.items())
        },
        "support_size_histogram": np.bincount(
            np.asarray(support_sizes, dtype=np.int64)
        ).tolist()
        if support_sizes
        else [],
        "relations_deterministic": bool(deterministic),
        "feasible_for_outer_relations": bool(
            deterministic and n_pairs > 0 and n_overlap + n_adjacent > 0
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "relations_preview.json", payload)
    return payload


# ---------------------------------------------------------------------------
# compute profile (section 17)
# ---------------------------------------------------------------------------


def profile(n_batches: int = 10, device: str = "cpu") -> dict[str, Any]:
    _train, valid_data, _audit = build_encoded_records()
    loader = _make_exo_loader(list(valid_data)[:1280], 128, False, 0)
    device_obj = torch.device(device)
    composer = build_candidate(0).structural_encoder.to(device_obj).eval()
    bfull = sspe.build_candidate(0).structural_encoder.to(device_obj).eval()

    batches = []
    object_counts = {"level1": [], "level2": [], "selected": []}
    for index, batch in enumerate(loader):
        if index >= n_batches:
            break
        batch = batch.to(device_obj)
        batches.append(batch)
        object_counts["level1"].append(int(batch.exo_l1_u.numel()))
        object_counts["level2"].append(int(batch.exo_l2_a.numel()))
        with torch.no_grad():
            details = composer(batch, return_details=True)
        object_counts["selected"].append(
            int(details["selected"].sum().item())
        )

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

    composer_latency = (
        float(np.mean([_time(composer, b) for b in batches])) if batches else 0.0
    )
    bfull_latency = (
        float(np.mean([_time(bfull, b) for b in batches])) if batches else 0.0
    )
    total_patches = sum(int(b.struct_n_patches) for b in batches)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "device": str(device),
        "n_batches": int(len(batches)),
        "batch_size": 128,
        "composer_forward_latency_s": composer_latency,
        "bfull_forward_latency_s": bfull_latency,
        "composer_latency_ratio": (
            float(composer_latency / bfull_latency) if bfull_latency else 0.0
        ),
        "mean_level1_objects_per_patch": float(
            np.sum(object_counts["level1"]) / max(total_patches, 1)
        ),
        "mean_level2_objects_per_patch": float(
            np.sum(object_counts["level2"]) / max(total_patches, 1)
        ),
        "mean_selected_objects_per_patch": float(
            np.sum(object_counts["selected"]) / max(total_patches, 1)
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


def decide(tag: str = "exco") -> dict[str, Any]:
    seed = PROMOTION_SEED
    soup = _read_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json")
    run = _read_json(RUNS_DIR / f"{tag}_seed{seed}.json")
    soup_valid = float(soup["top5_soup_valid_mae"])
    raw_valid = float(soup["best_checkpoint_valid_mae"])
    reference = REFERENCE_BFULL_SOUP_PER_SEED[seed]
    delta = soup_valid - reference

    diagnostics_payload = (
        _read_json(RESULTS_DIR / "diagnostics.json")
        if (RESULTS_DIR / "diagnostics.json").exists()
        else None
    )
    collapse = None
    if diagnostics_payload:
        collapse = diagnostics_payload["per_seed"][str(seed)]["collapse"]

    method_conditions = {
        "support_provenance_explicit": True,
        "no_collapse": bool(collapse is not None and not collapse["any_collapse"]),
        "vocab_free": True,
    }
    if delta <= -STRONG_GATE and method_conditions["no_collapse"]:
        verdict = "STRONG_GO"
    elif abs(delta) <= METHOD_GATE and all(method_conditions.values()):
        verdict = "METHOD_GO"
    elif delta >= REGRESSION_GATE or not method_conditions["no_collapse"]:
        verdict = "STOP"
    else:
        verdict = "WEAK_AMBIGUOUS"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "promotion_seed": int(seed),
        "single_seed": True,
        "seed1_soup_valid": float(soup["soup_valid"] if "soup_valid" in soup else soup_valid),
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
            "method_go": METHOD_GATE,
            "regression": REGRESSION_GATE,
        },
        "method_conditions": method_conditions,
        "collapse": collapse,
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
        "compute_profile": _maybe("compute_profile"),
        "relations_preview": _maybe("relations_preview"),
        "decision": _maybe("decision"),
        "soup": {str(seed): _maybe(f"soup_exco_seed{seed}") for seed in SEEDS},
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
            "profile",
            "relations",
            "decide",
            "report",
        ),
    )
    parser.add_argument("--seed", type=int, default=PROMOTION_SEED)
    parser.add_argument("--seeds", type=str, default=str(PROMOTION_SEED))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="exco")
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
        explicit_candidates()
        print(json.dumps(_read_json(_exo_cache_paths()[2]), indent=2), flush=True)
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
    elif args.stage == "profile":
        print(
            json.dumps(profile(device=args.device), indent=2, default=str),
            flush=True,
        )
    elif args.stage == "relations":
        print(
            json.dumps(relations_preview(), indent=2, default=str), flush=True
        )
    elif args.stage == "decide":
        print(json.dumps(decide(tag=args.tag), indent=2, default=str), flush=True)
    elif args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
