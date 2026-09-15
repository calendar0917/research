"""ZINC shared connectivity-aware structural patch encoder (vocabulary-free).

Question
--------
The canonical compact-v4 / cell-A patch representation turns every rooted typed
radius-2 patch into a categorical certificate, maps it to a vocabulary id and
looks up a learned row (``typed_embedding``; 36,420 of cell A's 85,763
parameters).  This is opaque categorical memory: it cannot represent an unseen
certificate without a row and its parameter count grows with the certificate
vocabulary.

This experiment tests a different hypothesis:

    Completely remove the vocab-sized typed lookup and derive the patch
    representation directly from each rooted typed patch's real internal
    connectivity and semantic attributes, through a single encoder shared by
    every patch.

The candidate keeps the cell-A architecture everywhere else: radius-2 patch
definition, relation system, pair descriptors, T=2 weight-tied recurrent
pair--centre, parent embedding, global/topology channels, and the frozen
optimized training protocol.  Only the patch-token pathway changes:

    typed certificate -> vocabulary id -> learned row        (cell A)
    atom/root/distance/bond primitives -> 2 rounds of shared edge-aware
        message passing -> permutation-invariant pooling -> e_struct in R^16
                                                              (candidate)

This is **not** the historical v6 attribute branch: v6 pooled (type, role)
primitives independently and *kept* the coarse typed lookup.  It is not SBCI
low-rank relational composition, not a rare/OOV KNN or residual-exact-token
scheme, and not a learned token dictionary: the encoder never looks up an exact
patch identity and its parameters are independent of the certificate
vocabulary.

Official ZINC test is **never** loaded.  The reference is the deterministic-A100
cell-A 2-seed fixed Top-5 soup validation MAE (``0.126368``).

Stages: ``params preprocess sanity train soup train_queue diagnostics decide
molhiv report``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_shared_structural_patch_encoder <stage>
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import time
from dataclasses import dataclass
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
from tracks.ksvd.experiments.luyin16 import zinc_post_v4_residual_audit as postv4
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
    base_config as _v4_base_config,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _data_to_graph,
    _load_zinc,
)

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/shared_structural_patch_encoder"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"
CACHE_DIR = RESULTS_DIR / "cache"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "shared_structural_patch_encoder_v1"
CACHE_SCHEMA_VERSION = "shared_structural_patch_graphs_v2"

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

# --- pre-registered shared structural encoder (one architecture) -------------
STRUCT_NODE_DIM = 48
STRUCT_EDGE_DIM = 24
STRUCT_HIDDEN_DIM = 48
STRUCT_ROUNDS = 2
STRUCT_INCLUDE_STD_POOL = True

# --- reference + decision gates ----------------------------------------------
REFERENCE_SOUP_VALID = 0.1263680279762484  # deterministic-A100 cell A, 2-seed soup
REFERENCE_RAW_VALID = {0: 0.129710, 1: 0.131415}
REFERENCE_SOUP_VALID_PER_SEED = {0: 0.124704, 1: 0.128032}
REFERENCE_PARAMS = 85763

PARAM_LOWER = 80000
PARAM_UPPER = 90000

STRONG_GATE = 0.002
REGRESSION_GATE = 0.003

SEEDS = (0, 1)

DETERMINISTIC = False

# MolHIV typed lookup to be replaced (source:
# results/molhiv_parameter_attribution/parameter_breakdown.json).
MOLHIV_TYPED_LOOKUP_PARAMS = 26233 * 32  # 839,456
MOLHIV_TYPED_TOKEN_WIDTH = 32


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    if head.exists():
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        return text
    return "unknown"


def _environment_fingerprint(device: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_candidate(seed: int = 0) -> torch.nn.Module:
    """Cell-A geometry with the typed lookup replaced by the shared encoder.

    Construction order mirrors ``cd.build_cell("A", seed)`` exactly: the
    canonical baseline is instantiated first, every shared tensor whose shape is
    unchanged is copied bit-exactly, and the fixed small head is re-drawn from
    the historical ``head_seed=0`` stream.  The only difference is that the
    model is built with ``patch_representation="shared_structural"`` so it
    contains no ``typed_embedding`` and a shared two-round structural encoder
    instead.
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
            patch_representation="shared_structural",
            structural_node_dim=STRUCT_NODE_DIM,
            structural_edge_dim=STRUCT_EDGE_DIM,
            structural_hidden_dim=STRUCT_HIDDEN_DIM,
            structural_rounds=STRUCT_ROUNDS,
            structural_include_std_pool=STRUCT_INCLUDE_STD_POOL,
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


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# patch-graph preprocessing / cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchGraph:
    """Local (0-based within each patch) rooted typed patch graph."""

    atom: np.ndarray  # [T] int64
    root: np.ndarray  # [T] int64 (1 for the root, else 0)
    dist: np.ndarray  # [T] int64 distance from the root
    patch: np.ndarray  # [T] int64 local patch index
    src: np.ndarray  # [E] int64 local node index within the owning patch
    dst: np.ndarray  # [E] int64 local node index within the owning patch
    bond: np.ndarray  # [E] int64
    edge_patch: np.ndarray  # [E] int64 local patch index
    n_patches: int
    n_nodes: int
    n_edges: int


def _patch_graphs_from_dataset(
    dataset: Sequence[Any], radius: int = PATCH_RADIUS
) -> list[PatchGraph]:
    """Rebuild the real rooted typed patch graph for every patch.

    Uses the *same* graph construction, root enumeration order and radius as
    ``zpp._graph_record`` / ``typed_patch_tokenizer.build_colored_incidence`` so
    the patch order matches the encoded ``GraphRecord`` patch order.  No
    certificate, no vocabulary and no target is used.
    """
    radius = int(radius)
    output: list[PatchGraph] = []
    for data in dataset:
        graph, node_types, edge_types = _data_to_graph(data)
        centers = list(graph.nodes)
        atom_parts: list[np.ndarray] = []
        root_parts: list[np.ndarray] = []
        dist_parts: list[np.ndarray] = []
        patch_parts: list[np.ndarray] = []
        src_parts: list[np.ndarray] = []
        dst_parts: list[np.ndarray] = []
        bond_parts: list[np.ndarray] = []
        edge_patch_parts: list[np.ndarray] = []
        for patch_index, center in enumerate(centers):
            distances = zpp._ego_distances(graph, int(center), radius)
            nodes = sorted(distances)
            local = {node: index for index, node in enumerate(nodes)}
            n_local = len(nodes)
            atom_parts.append(
                np.asarray([int(node_types[node]) for node in nodes], dtype=np.int64)
            )
            root_parts.append(
                np.asarray([1 if node == int(center) else 0 for node in nodes], dtype=np.int64)
            )
            dist_parts.append(
                np.asarray([int(distances[node]) for node in nodes], dtype=np.int64)
            )
            patch_parts.append(np.full(n_local, patch_index, dtype=np.int64))
            induced = graph.induced(set(nodes))
            for left, right in sorted(induced.edges()):
                left, right = int(left), int(right)
                bond = int(edge_types[graph.edge_key(left, right)])
                for u, v in ((left, right), (right, left)):
                    src_parts.append(np.asarray([local[u]], dtype=np.int64))
                    dst_parts.append(np.asarray([local[v]], dtype=np.int64))
                    bond_parts.append(np.asarray([bond], dtype=np.int64))
                    edge_patch_parts.append(
                        np.asarray([patch_index], dtype=np.int64)
                    )
        atom = (
            np.concatenate(atom_parts) if atom_parts else np.zeros(0, dtype=np.int64)
        )
        root = (
            np.concatenate(root_parts) if root_parts else np.zeros(0, dtype=np.int64)
        )
        dist = (
            np.concatenate(dist_parts) if dist_parts else np.zeros(0, dtype=np.int64)
        )
        patch = (
            np.concatenate(patch_parts) if patch_parts else np.zeros(0, dtype=np.int64)
        )
        src = np.concatenate(src_parts) if src_parts else np.zeros(0, dtype=np.int64)
        dst = np.concatenate(dst_parts) if dst_parts else np.zeros(0, dtype=np.int64)
        bond = (
            np.concatenate(bond_parts) if bond_parts else np.zeros(0, dtype=np.int64)
        )
        edge_patch = (
            np.concatenate(edge_patch_parts)
            if edge_patch_parts
            else np.zeros(0, dtype=np.int64)
        )
        output.append(
            PatchGraph(
                atom=atom,
                root=root,
                dist=dist,
                patch=patch,
                src=src,
                dst=dst,
                bond=bond,
                edge_patch=edge_patch,
                n_patches=len(centers),
                n_nodes=int(atom.shape[0]),
                n_edges=int(src.shape[0]),
            )
        )
    return output


def _graphs_to_plain(graphs: Sequence[PatchGraph]) -> list[dict[str, Any]]:
    """Serialize patch graphs as plain arrays so the cache is import-safe.

    Pickling dataclass instances defined in a module executed via ``python -m``
    records ``__main__.PatchGraph`` and cannot be reloaded from another entry
    point.  Plain dicts avoid that fragility entirely.
    """
    return [
        {
            "atom": graph.atom,
            "root": graph.root,
            "dist": graph.dist,
            "patch": graph.patch,
            "src": graph.src,
            "dst": graph.dst,
            "bond": graph.bond,
            "edge_patch": graph.edge_patch,
            "n_patches": int(graph.n_patches),
            "n_nodes": int(graph.n_nodes),
            "n_edges": int(graph.n_edges),
        }
        for graph in graphs
    ]


def _graphs_from_plain(items: Sequence[Mapping[str, Any]]) -> list[PatchGraph]:
    return [PatchGraph(**dict(item)) for item in items]


def _cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "struct_graphs_train.pkl.gz",
        CACHE_DIR / "struct_graphs_valid.pkl.gz",
        CACHE_DIR / "cache_metadata.json",
    )


def _patch_graph_stats(graphs: Sequence[PatchGraph]) -> dict[str, Any]:
    n_nodes = np.asarray([g.n_nodes for g in graphs], dtype=np.float64)
    per_patch = np.asarray(
        [
            np.bincount(g.patch, minlength=g.n_patches).mean()
            for g in graphs
            if g.n_patches
        ],
        dtype=np.float64,
    )
    return {
        "n_graphs": int(len(graphs)),
        "total_patches": int(sum(g.n_patches for g in graphs)),
        "total_patch_nodes": int(sum(g.n_nodes for g in graphs)),
        "total_patch_edges": int(sum(g.n_edges for g in graphs)),
        "mean_patch_nodes": float(per_patch.mean()) if per_patch.size else 0.0,
        "max_patch_nodes": int(
            max(
                (
                    int(np.bincount(g.patch, minlength=g.n_patches).max())
                    for g in graphs
                    if g.n_patches
                ),
                default=0,
            )
        ),
        "patch_node_histogram": np.bincount(n_nodes.astype(np.int64)).tolist()
        if n_nodes.size
        else [],
    }


def extract_records(force: bool = False):
    """Extract (and cache) the base GraphRecords and the patch graphs."""
    train_path, valid_path, meta_path = _cache_paths()
    if not force and train_path.exists() and valid_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
            with gzip.open(train_path, "rb") as handle:
                train_bundle = pickle.load(handle)
            with gzip.open(valid_path, "rb") as handle:
                valid_bundle = pickle.load(handle)
            train_bundle["graphs"] = _graphs_from_plain(train_bundle["graphs"])
            valid_bundle["graphs"] = _graphs_from_plain(valid_bundle["graphs"])
            return train_bundle, valid_bundle, meta

    started = time.perf_counter()
    train_records, valid_records = postv4._extract_v4_records()
    train_ds = _load_zinc(ZINC_ROOT, "train")
    valid_ds = _load_zinc(ZINC_ROOT, "val")
    train_graphs = _patch_graphs_from_dataset(train_ds)
    valid_graphs = _patch_graphs_from_dataset(valid_ds)

    if len(train_graphs) != len(train_records) or len(valid_graphs) != len(
        valid_records
    ):
        raise RuntimeError("patch graph / record molecule count mismatch")
    for graphs, records, split in (
        (train_graphs, train_records, "train"),
        (valid_graphs, valid_records, "valid"),
    ):
        for index, (graph, record) in enumerate(zip(graphs, records)):
            if graph.n_patches != len(record.patches):
                raise RuntimeError(
                    f"{split} molecule {index}: patch count mismatch "
                    f"{graph.n_patches} != {len(record.patches)}"
                )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_bundle = {"records": train_records, "graphs": _graphs_to_plain(train_graphs)}
    valid_bundle = {"records": valid_records, "graphs": _graphs_to_plain(valid_graphs)}
    with gzip.open(train_path, "wb") as handle:
        pickle.dump(train_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(valid_path, "wb") as handle:
        pickle.dump(valid_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "patch_radius": PATCH_RADIUS,
        "n_train_molecules": int(len(train_records)),
        "n_valid_molecules": int(len(valid_records)),
        "train": _patch_graph_stats(train_graphs),
        "valid": _patch_graph_stats(valid_graphs),
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    _write_json(meta_path, meta)
    print(
        f"[extract] train={len(train_records)} valid={len(valid_records)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_bundle, valid_bundle, meta


def _attach_struct_tensors(
    encoded: Sequence[Data], graphs: Sequence[PatchGraph]
) -> None:
    """Attach graph-local structural tensors to the encoded ``Data`` objects.

    Patch ids are local (``0..n_patches-1``) and edge endpoints are local node
    indices relative to the owning graph's node block.  The custom collate used
    by this experiment offsets them by the cumulative batch offsets, so the
    default PyG node/edge increments are never relied on.
    """
    for data, graph in zip(encoded, graphs):
        if int(data.num_nodes) != graph.n_patches:
            raise RuntimeError("encoded / patch-graph patch count mismatch")
        node_counts = np.bincount(graph.patch, minlength=graph.n_patches).astype(
            np.int64
        )
        node_start = np.concatenate([[0], np.cumsum(node_counts)[:-1]]).astype(np.int64)
        data.struct_atom = torch.from_numpy(graph.atom)
        data.struct_root = torch.from_numpy(graph.root)
        data.struct_dist = torch.from_numpy(graph.dist)
        data.struct_patch = torch.from_numpy(graph.patch)
        data.struct_src = torch.from_numpy(graph.src + node_start[graph.edge_patch])
        data.struct_dst = torch.from_numpy(graph.dst + node_start[graph.edge_patch])
        data.struct_bond = torch.from_numpy(graph.bond)


def build_encoded_records():
    """Return (train_data, valid_data, audit) with structural tensors attached."""
    train_bundle, valid_bundle, _meta = extract_records()
    config = _v4_base_config()
    config["model"]["device"] = "cpu"
    enc_train, enc_valid, audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    _attach_struct_tensors(enc_train, train_bundle["graphs"])
    _attach_struct_tensors(enc_valid, valid_bundle["graphs"])
    audit = dict(audit)
    audit["patch_representation"] = "shared_structural"
    audit["structural_encoder"] = {
        "node_dim": STRUCT_NODE_DIM,
        "edge_dim": STRUCT_EDGE_DIM,
        "hidden_dim": STRUCT_HIDDEN_DIM,
        "rounds": STRUCT_ROUNDS,
        "output_dim": TOKEN_WIDTH,
        "include_std_pool": STRUCT_INCLUDE_STD_POOL,
    }
    audit["official_test_loaded"] = False
    return enc_train, enc_valid, audit


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    baseline = cd.build_cell(CELL, 0)
    candidate = build_candidate(0)
    base_total = _n_params(baseline)
    cand_total = _n_params(candidate)
    if base_total != REFERENCE_PARAMS:
        raise RuntimeError(
            f"cell-A baseline params changed: {base_total} != {REFERENCE_PARAMS}"
        )
    audit = zpp.audit_parameters(candidate)
    typed_keys = [key for key in candidate.state_dict() if "typed_embedding" in key]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cell": CELL,
        "baseline_params": int(base_total),
        "candidate_params": int(cand_total),
        "params_in_range": bool(PARAM_LOWER <= cand_total <= PARAM_UPPER),
        "structural_encoder_params": int(
            _n_params(candidate.structural_encoder)
        ),
        "released_typed_lookup_params": int(
            _n_params(baseline.typed_embedding) if baseline.typed_embedding is not None else 0
        ),
        "candidate_has_typed_embedding": bool(
            getattr(candidate, "typed_embedding", None) is not None
        ),
        "candidate_typed_state_keys": typed_keys,
        "candidate_block_params": audit["blocks"],
        "candidate_dimensions": audit["dimensions"],
        "baseline_block_params": zpp.audit_parameters(baseline)["blocks"],
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
# sanity / mechanism tests
# ---------------------------------------------------------------------------


def _encode_candidate_batch(model: nn.Module, batch: Data) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model.encode(batch)


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    train_data, valid_data, audit = build_encoded_records()
    loader = _make_struct_loader(list(valid_data)[:128], 128, False, 0)
    batch = next(iter(loader)).to(device)

    model = build_candidate(0).to(device)
    baseline = cd.build_cell(CELL, 0).to(device)

    checks: dict[str, bool] = {}

    # (7) no vocab-sized typed lookup anywhere in the model.
    typed_state = [k for k in model.state_dict() if "typed_embedding" in k]
    typed_named = [k for k, _ in model.named_parameters() if "typed_embedding" in k]
    checks["no_vocab_sized_typed_embedding_weight"] = (
        len(typed_state) == 0 and len(typed_named) == 0
    )
    checks["structural_encoder_present"] = bool(
        getattr(model, "structural_encoder", None) is not None
    )
    # (8) parent path preserved.
    checks["parent_embedding_present"] = hasattr(model, "parent_embedding") and (
        _n_params(model.parent_embedding)
        == _n_params(baseline.parent_embedding)
    )
    # (9) q=16, h=64, T=2.
    checks["q16_h64_T2"] = (
        int(model.pair_hidden) == 16
        and int(model.patch_hidden) == 64
        and int(model.recurrence_rounds) == 2
    )
    # (10) recurrent weight tying (shared modules called once per round).
    counts = model.module_call_counts(batch)
    checks["recurrent_weight_tying"] = counts == {
        "pair_projection": 4,
        "pair_encoder": 2,
        "center_update": 2,
    }
    # (12) parameter budget.
    total = _n_params(model)
    checks["params_in_range"] = bool(PARAM_LOWER <= total <= PARAM_UPPER)
    checks["param_minus_baseline_small"] = abs(total - REFERENCE_PARAMS) < 15000

    # (11) forward / backward finite.
    model.train()
    out = model(batch)
    loss = F.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    checks["forward_finite"] = bool(torch.isfinite(out).all())
    checks["backward_finite_nonzero"] = all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    ) and any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in model.parameters()
    )
    model.zero_grad(set_to_none=True)

    # --- structural-encoder invariance unit checks on a real batch ---------
    encoder = model.structural_encoder
    encoder.eval()
    with torch.no_grad():
        e_ref = encoder(batch)

    # (1)+(5) relabel patch nodes consistently -> identical e_struct.
    atom = batch.struct_atom.clone()
    root = batch.struct_root.clone()
    dist = batch.struct_dist.clone()
    patch = batch.struct_patch.clone()
    src = batch.struct_src.clone()
    dst = batch.struct_dst.clone()
    bond = batch.struct_bond.clone()
    # deterministic per-node permutation within each patch
    n_nodes = int(atom.shape[0])
    perm = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(20260927))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n_nodes)
    perm_repr = _StructBatch(
        atom=atom[perm],
        root=root[perm],
        dist=dist[perm],
        patch=patch[perm],
        src=inverse[src],
        dst=inverse[dst],
        bond=bond,
    )
    with torch.no_grad():
        e_perm = encoder(perm_repr)
    checks["node_relabel_invariant"] = bool(
        float((e_ref - e_perm).abs().max()) < 1.0e-5
    )

    # (2) atom type change changes the representation.
    atom2 = atom.clone()
    first = int(atom2[0].item())
    atom2[0] = (first + 1) % int(encoder.atom_categories)
    with torch.no_grad():
        e_atom = encoder(
            _StructBatch(atom2, root, dist, patch, src, dst, bond)
        )
    checks["atom_type_changes_representation"] = bool(
        float((e_ref - e_atom).abs().max()) > 1.0e-6
    )

    # (3) bond type change changes the representation.
    bond2 = bond.clone()
    if bond2.numel():
        first_bond = int(bond2[0].item())
        bond2[0] = (first_bond + 1) % int(encoder.bond_categories)
    with torch.no_grad():
        e_bond = encoder(_StructBatch(atom, root, dist, patch, src, dst, bond2))
    checks["bond_type_changes_representation"] = bool(
        float((e_ref - e_bond).abs().max()) > 1.0e-6
    )

    # (4) root position change changes the rooted representation.
    root3 = root.clone()
    root_indices = torch.nonzero(root3 > 0, as_tuple=False).view(-1)
    # move the root of patch 0 to a different node in the same patch
    patch0_root = int(root_indices[0].item())
    same_patch = torch.nonzero(patch == patch[patch0_root], as_tuple=False).view(-1)
    alternative = [int(i) for i in same_patch.tolist() if int(i) != patch0_root]
    if alternative:
        root3[patch0_root] = 0
        root3[alternative[0]] = 1
    with torch.no_grad():
        e_root = encoder(_StructBatch(atom, root3, dist, patch, src, dst, bond))
    checks["root_position_changes_representation"] = bool(
        float((e_ref - e_root).abs().max()) > 1.0e-6
    )

    # (6) unseen certificate needs no vocabulary row: clearing the certificate /
    # vocabulary ids (typed_token) leaves the structural forward unchanged.
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

    device_info = _environment_fingerprint("cpu")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_params": int(total),
        "baseline_params": int(_n_params(baseline)),
        "structural_encoder_params": int(_n_params(encoder)),
        "split": "first 128 official-valid molecules",
        "environment": device_info,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    # patch-graph audit hash so the cache is identifiable
    payload["cache_metadata"] = _read_json(CACHE_DIR / "cache_metadata.json") if (
        CACHE_DIR / "cache_metadata.json"
    ).exists() else None
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"shared structural sanity failed: {failed}")
    return payload


class _StructBatch:
    """Minimal attribute container for structural-encoder invariance tests."""

    def __init__(self, atom, root, dist, patch, src, dst, bond) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_src = src
        self.struct_dst = dst
        self.struct_bond = bond


# ---------------------------------------------------------------------------
# custom batching for the structural graph tensors
# ---------------------------------------------------------------------------


def struct_collate(data_list: Sequence[Data]) -> Any:
    """Batch with explicit offsets for the structural patch-graph tensors.

    PyG's default collate leaves unknown keys un-offset.  The structural tensor
    ids are graph-local, so they are offset here by the cumulative node / patch
    counts of the graphs in the batch.
    """
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(list(data_list))
    node_offset = 0
    patch_offset = 0
    patch_parts: list[torch.Tensor] = []
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    patch_batch_parts: list[torch.Tensor] = []
    for graph_index, data in enumerate(data_list):
        n_patches = int(data.num_nodes)
        patch_parts.append(data.struct_patch + patch_offset)
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
        collate_fn=struct_collate,
    )


def _evaluate_mae(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            total += float((prediction - target).abs().sum())
            seen += int(target.numel())
    return float(total / max(seen, 1))


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def _selection_loader(valid_data: Sequence[Data]):
    return _make_struct_loader(valid_data, 128, False, 0)


def train_seed(
    seed: int,
    device: str = "cpu",
    tag: str = "sspe",
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
        writer = csv.DictWriter(
            handle, fieldnames=list(curve[0].keys())
        )
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

    train_at_best = float(losses[best_epoch - 1])
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "seed": int(seed),
        "protocol": protocol,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "train_loss_at_best": train_at_best,
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
        "patch_representation": "shared_structural",
        "structural_encoder_params": int(_n_params(model.structural_encoder)),
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


def soup_seed(seed: int, tag: str = "sspe") -> dict[str, Any]:
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
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
        "valid_targets": targets.tolist(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    return payload


def train_queue(seeds: Sequence[int], device: str, tag: str = "sspe") -> None:
    for seed in seeds:
        if (RUNS_DIR / f"{tag}_seed{seed}.json").exists() and (
            SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
        ).exists():
            print(f"skip existing seed{seed}", flush=True)
            continue
        print(f"=== train seed{seed} device={device} ===", flush=True)
        train_seed(int(seed), device=device, tag=tag)
        soup_seed(int(seed), tag=tag)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().to(torch.float32).cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def repro(
    seed: int, epochs: int, device: str, run_tag: str = "default"
) -> dict[str, Any]:
    """Short deterministic GPU sanity for one seed (isolated result subtree)."""
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
    effective = float(math.exp(entropy))
    participation = float(total * total / float((energy**2).sum()))
    return {
        "effective_rank": effective,
        "participation_ratio": participation,
        "top_singular_fraction": float(energy[0] / total),
    }


def diagnostics(tag: str = "sspe") -> dict[str, Any]:
    device = torch.device("cpu")
    _train, valid_data, audit = build_encoded_records()
    loader = _make_struct_loader(list(valid_data)[:256], 128, False, 0)
    batches = [b.to(device) for b in loader]

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": {},
        "patch_representation": "shared_structural",
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
        h0_matrix = np.concatenate(h0_blocks, axis=0) if h0_blocks else np.zeros((0, 1))
        payload["per_seed"][str(seed)] = {
            "struct_encoder_rank": _effective_rank(encoder_matrix),
            "patch_h0_rank": _effective_rank(h0_matrix),
            "n_encoder_samples": int(encoder_matrix.shape[0]),
            "n_h0_samples": int(h0_matrix.shape[0]),
        }
    _write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(tag: str = "sspe") -> dict[str, Any]:
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
    delta_per_seed = {
        str(seed): rows[str(seed)]["soup_valid"]
        - REFERENCE_SOUP_VALID_PER_SEED[seed]
        for seed in SEEDS
    }
    delta = soup_mean - REFERENCE_SOUP_VALID
    same_direction = all(value < 0 for value in delta_per_seed.values()) or all(
        value > 0 for value in delta_per_seed.values()
    )
    if delta <= -STRONG_GATE and same_direction:
        verdict = "STRONG_SUCCESS"
    elif abs(delta) < STRONG_GATE:
        verdict = "PERFORMANCE_NEUTRAL_SUCCESS"
    elif delta >= REGRESSION_GATE and same_direction:
        verdict = "FAILURE"
    else:
        verdict = "INCONCLUSIVE_SIGN_FLIP"

    params_payload = _read_json(RESULTS_DIR / "parameter_accounting.json")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference": {
            "cell_A_2seed_soup_valid": REFERENCE_SOUP_VALID,
            "cell_A_params": REFERENCE_PARAMS,
        },
        "per_seed": rows,
        "candidate_2seed_soup_valid_mean": soup_mean,
        "candidate_2seed_raw_valid_mean": raw_mean,
        "delta_soup_mean_vs_reference": delta,
        "delta_per_seed_vs_reference": delta_per_seed,
        "same_direction": bool(same_direction),
        "verdict": verdict,
        "candidate_params": int(params_payload["candidate_params"]),
        "vocab_independent": True,
        "no_typed_lookup": bool(not params_payload["candidate_has_typed_embedding"]),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


# ---------------------------------------------------------------------------
# MolHIV projection (accounting only; no training)
# ---------------------------------------------------------------------------


def molhiv_projection() -> dict[str, Any]:
    source = (
        TRACK_ROOT
        / "results/molhiv_parameter_attribution/parameter_breakdown.json"
    )
    total = None
    typed = None
    if source.exists():
        breakdown = _read_json(source)
        total = int(breakdown["total_params"])
        for tensor in breakdown.get("top20_tensors", []):
            name = str(tensor.get("name", ""))
            if name.startswith("typed_embedding"):
                typed = int(tensor.get("numel", tensor.get("parameters", 0)))
        if typed is None:
            typed = MOLHIV_TYPED_LOOKUP_PARAMS
    total = int(total if total is not None else 1076589)
    typed = int(typed if typed is not None else MOLHIV_TYPED_LOOKUP_PARAMS)
    encoder = _n_params(
        _molhiv_projected_encoder()
    )
    projected = total - typed + encoder
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "parameter_accounting_only",
        "no_performance_claim": True,
        "no_molhiv_training": True,
        "source_file": str(source),
        "molhiv_total_params": total,
        "replaced_typed_lookup_params": typed,
        "structural_encoder_params_output32": encoder,
        "projected_total_params": int(projected),
        "delta_params": int(projected - total),
        "assumptions": [
            "typed_embedding table [26233,32] removed",
            "shared structural encoder output width = 32 to keep the "
            "MolHIV patch encoder input width unchanged",
            "all other MolHIV modules unchanged",
        ],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "molhiv_projection.json", payload)
    return payload


def _molhiv_projected_encoder() -> nn.Module:
    from tracks.ksvd.experiments.luyin16.structural_patch_encoder import (
        SharedStructuralPatchEncoder,
    )

    return SharedStructuralPatchEncoder(
        atom_categories=zpp.ATOM_CATEGORIES,
        bond_categories=zpp.BOND_CATEGORIES,
        n_distance_bins=PATCH_RADIUS + 1,
        node_dim=STRUCT_NODE_DIM,
        edge_dim=STRUCT_EDGE_DIM,
        hidden_dim=STRUCT_HIDDEN_DIM,
        output_dim=MOLHIV_TYPED_TOKEN_WIDTH,
        rounds=STRUCT_ROUNDS,
        include_std_pool=STRUCT_INCLUDE_STD_POOL,
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "parameter_accounting": _maybe("parameter_accounting.json"),
        "cache_metadata": _maybe("cache_metadata.json"),
        "sanity": (lambda d: None if d is None else d["checks"])(_maybe("sanity.json")),
        "diagnostics": _maybe("diagnostics.json"),
        "decision": _maybe("decision.json"),
        "molhiv_projection": _maybe("molhiv_projection.json"),
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
            "train",
            "train_queue",
            "repro",
            "soup",
            "diagnostics",
            "decide",
            "molhiv",
            "report",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="sspe")
    parser.add_argument("--epochs", type=int, default=6)
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
    if args.stage == "preprocess":
        _train, _valid, meta = extract_records(force=bool(args.force))
        print(json.dumps(meta, indent=2, default=str), flush=True)
    if args.stage == "sanity":
        print(json.dumps(sanity()["checks"], indent=2, default=str), flush=True)
    if args.stage == "train":
        print(
            json.dumps(
                train_seed(args.seed, device=args.device, tag=args.tag),
                indent=2,
                default=str,
            )
        )
    if args.stage == "train_queue":
        train_queue(
            [int(s) for s in args.seeds.split(",") if s != ""],
            args.device,
            tag=args.tag,
        )
    if args.stage == "soup":
        print(
            json.dumps(soup_seed(args.seed, tag=args.tag), indent=2, default=str),
            flush=True,
        )
    if args.stage == "repro":
        print(
            json.dumps(
                repro(args.seed, args.epochs, args.device, args.run_tag),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "diagnostics":
        print(
            json.dumps(diagnostics(tag=args.tag), indent=2, default=str),
            flush=True,
        )
    if args.stage == "decide":
        print(json.dumps(decide(tag=args.tag), indent=2, default=str), flush=True)
    if args.stage == "molhiv":
        print(json.dumps(molhiv_projection(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
