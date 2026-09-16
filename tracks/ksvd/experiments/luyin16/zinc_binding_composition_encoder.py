"""ZINC Binding Composition Encoder (BCE) — composition study, not attention.

Research question
-----------------
Can a model start from the *minimal explicit structure--attribute bindings*
(atom primitives and real-bond bindings) and, through one composition operator
shared across every patch, recursively form **higher-order explicit objects**
(connected induced supports with their exact atom/bond support and provenance),
then use those objects for ZINC prediction?

This is *not* GAT / routing / attention: no existing edge is given a scalar
weight, no softmax over decompositions, no gate, no hard/soft mask and no top-k.
The object contract is

    Object = (exact support, learned state, provenance)

and the data flow is a single structural path::

    atom attributes + root role + root distance
            -> atom primitives
    real bond + endpoint attributes
            -> bond-binding primitives
    every legal unordered parent decomposition of every connected induced
    support (|S| = 3, 4)
            -> shared symmetric Compose -> candidate object
            -> invariant (mean, std, count) aggregation -> SupportUpdate
            -> higher-order object state
    invariant object pooling (root state / object mean / object std / count)
            -> e_struct in R^16
            -> existing frozen downstream

There is no B-Bag or B-Full bypass: the higher-order objects *are* the patch
representation.  B-Bag (shared bag) and B-Full (shared structural) remain frozen
external references.

Inherited unchanged: radius-2 patch extraction, pair relation system, pair
descriptors, ``T=2`` weight-tied recurrent pair--centre, global/topology
channels, graph head, optimizer, LR, weight decay, batch size, gradient
clipping, training horizon, checkpoint selection and the Top-5 soup.

Frozen references (never retrained here; official valid, 2-seed Top-5 soup):
    B-Bag  0.12380552224389975
    B-Full 0.11897220489243046
    A2     0.12191404939390486

Official ZINC test is **never** loaded.

Stages: ``params preprocess candidate_audit sanity smoke train soup train_queue
repro diagnostics disagreement support_examples decide report``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_binding_composition_encoder <stage>
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
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
from tracks.ksvd.experiments.luyin16.binding_composition_support import (
    MAX_SUPPORT_SIZE,
    SupportChart,
    build_chart,
    chart_from_plain,
    chart_to_plain,
)
from tracks.ksvd.experiments.luyin16.structural_patch_encoder import (
    BindingCompositionEncoder,
)
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency import (
    base_config as _v4_base_config,
)

# ---------------------------------------------------------------------------
# layout / frozen constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/binding_composition_encoder"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SOUP_DIR = RESULTS_DIR / "soup_states"
CACHE_DIR = RESULTS_DIR / "cache"

ZINC_ROOT = REPO_ROOT / "data/ZINC"
PATCH_RADIUS = 2
PROTOCOL_VERSION = "binding_composition_encoder_v1"
CACHE_SCHEMA_VERSION = "binding_composition_support_chart_v1"

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

# --- pre-registered BCE geometry (one architecture; no sweep) ----------------
BCE_OBJECT_DIM = 32
BCE_COMPOSE_HIDDEN = 48
BCE_UPDATE_HIDDEN = 48
BCE_FUSION_HIDDEN = 48
BCE_ACTIVATION = "silu"

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
SEED0_GUARD = REFERENCE_BBAG_SOUP_VALID + 0.003

SEEDS = (0, 1)
AUDIT_SUBSET_MOLECULES = 512
DIAG_BATCHES = 128

DETERMINISTIC = False

_write_json = sspe._write_json
_read_json = sspe._read_json
_git_commit = sspe._git_commit
_environment_fingerprint = sspe._environment_fingerprint
_patch_graph_stats = sspe._patch_graph_stats


def _set_deterministic(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_candidate(seed: int = 0) -> torch.nn.Module:
    """Cell-A geometry with the patch token produced by the BCE encoder.

    The canonical frozen baseline is instantiated first; every shared tensor
    whose shape is unchanged is copied bit-exactly.  The only difference is
    ``patch_representation="binding_composition"``: there is no
    ``typed_embedding`` and the patch token is the BCE ``e_struct``.
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
            patch_representation="binding_composition",
            bce_object_dim=BCE_OBJECT_DIM,
            bce_compose_hidden=BCE_COMPOSE_HIDDEN,
            bce_update_hidden=BCE_UPDATE_HIDDEN,
            bce_fusion_hidden=BCE_FUSION_HIDDEN,
            bce_max_support_size=MAX_SUPPORT_SIZE,
            bce_activation=BCE_ACTIVATION,
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


def _encoder_breakdown(encoder: BindingCompositionEncoder) -> dict[str, int]:
    return {
        "atom_root_distance_bond_embeddings": int(
            _n_params(encoder.atom_embedding)
            + _n_params(encoder.root_embedding)
            + _n_params(encoder.distance_embedding)
            + _n_params(encoder.bond_embedding)
        ),
        "atom_mlp": int(_n_params(encoder.atom_mlp)),
        "bond_binding_mlp": int(_n_params(encoder.bind_mlp)),
        "compose": int(_n_params(encoder.compose)),
        "support_update": int(_n_params(encoder.support_update)),
        "support_size_embedding": int(_n_params(encoder.size_embedding)),
        "patch_fusion": int(_n_params(encoder.fusion)),
    }


# ---------------------------------------------------------------------------
# support-chart preprocessing / cache
# ---------------------------------------------------------------------------


def _charts_from_graphs(graphs: Sequence[Any]) -> list[SupportChart]:
    """Build the target-free support chart for every molecule."""
    charts: list[SupportChart] = []
    for graph in graphs:
        node_counts = np.bincount(graph.patch, minlength=graph.n_patches).astype(
            np.int64
        )
        node_start = np.concatenate([[0], np.cumsum(node_counts)[:-1]]).astype(
            np.int64
        )
        offset = node_start[graph.edge_patch]
        edge_u = graph.src + offset
        edge_v = graph.dst + offset
        keep = edge_u < edge_v
        charts.append(
            build_chart(
                n_nodes=graph.n_nodes,
                patch_of_node=graph.patch,
                n_patches=graph.n_patches,
                edge_u=edge_u[keep],
                edge_v=edge_v[keep],
                edge_patch=graph.edge_patch[keep],
                root_of_node=graph.root.astype(bool),
                max_support_size=MAX_SUPPORT_SIZE,
            )
        )
    return charts


def _chart_cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "support_charts_train.pkl.gz",
        CACHE_DIR / "support_charts_valid.pkl.gz",
        CACHE_DIR / "cache_metadata.json",
    )


def _chart_split_stats(charts: Sequence[SupportChart]) -> dict[str, Any]:
    supports_per_patch_parts: list[np.ndarray] = []
    decompositions_per_patch_parts: list[np.ndarray] = []
    for chart in charts:
        if not chart.n_supports:
            continue
        n_patches = int(chart.sup_patch.max()) + 1
        supports_per_patch_parts.append(
            np.bincount(chart.sup_patch, minlength=n_patches).astype(np.float64)
        )
        if chart.n_decompositions:
            child_patch = chart.sup_patch[chart.dec_child]
            decompositions_per_patch_parts.append(
                np.bincount(child_patch, minlength=n_patches).astype(np.float64)
            )
        else:
            decompositions_per_patch_parts.append(np.zeros(n_patches))
    supports_per_patch = (
        np.concatenate(supports_per_patch_parts)
        if supports_per_patch_parts
        else np.zeros(0)
    )
    decompositions_per_patch = (
        np.concatenate(decompositions_per_patch_parts)
        if decompositions_per_patch_parts
        else np.zeros(0)
    )
    by_size = {
        str(size): int(
            sum(int((chart.sup_size == size).sum()) for chart in charts)
        )
        for size in range(1, MAX_SUPPORT_SIZE + 1)
    }
    total_supports = int(sum(chart.n_supports for chart in charts))
    total_decompositions = int(sum(chart.n_decompositions for chart in charts))

    def _percentiles(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        }

    return {
        "n_molecules": int(len(charts)),
        "n_patches": int(supports_per_patch.size),
        "n_supports": total_supports,
        "n_decompositions": total_decompositions,
        "supports_per_patch": _percentiles(supports_per_patch),
        "decompositions_per_patch": _percentiles(decompositions_per_patch),
        "supports_by_size": by_size,
    }


def extract_charts(
    force: bool = False, subset: int | None = None
) -> tuple[list[SupportChart], list[SupportChart], dict[str, Any]]:
    """Build (and cache) the target-free support charts for train / valid."""
    if subset is None:
        train_path, valid_path, meta_path = _chart_cache_paths()
        if (
            not force
            and train_path.exists()
            and valid_path.exists()
            and meta_path.exists()
        ):
            meta = _read_json(meta_path)
            if meta.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
                with gzip.open(train_path, "rb") as handle:
                    train_items = pickle.load(handle)
                with gzip.open(valid_path, "rb") as handle:
                    valid_items = pickle.load(handle)
                return (
                    [chart_from_plain(item) for item in train_items],
                    [chart_from_plain(item) for item in valid_items],
                    meta,
                )

    started = time.perf_counter()
    train_bundle, valid_bundle, _base_meta = sspe.extract_records()
    train_graphs = train_bundle["graphs"]
    valid_graphs = valid_bundle["graphs"]
    if subset is not None:
        train_graphs = train_graphs[: int(subset)]
        valid_graphs = valid_graphs[: int(subset)]
    train_charts = _charts_from_graphs(train_graphs)
    valid_charts = _charts_from_graphs(valid_graphs)
    meta = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "patch_radius": PATCH_RADIUS,
        "max_support_size": MAX_SUPPORT_SIZE,
        "subset": None if subset is None else int(subset),
        "train": _chart_split_stats(train_charts),
        "valid": _chart_split_stats(valid_charts),
        "seconds": float(time.perf_counter() - started),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    if subset is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with gzip.open(train_path, "wb") as handle:
            pickle.dump(
                [chart_to_plain(chart) for chart in train_charts],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        with gzip.open(valid_path, "wb") as handle:
            pickle.dump(
                [chart_to_plain(chart) for chart in valid_charts],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        _write_json(meta_path, meta)
    print(
        f"[charts] train={len(train_charts)} valid={len(valid_charts)} "
        f"({meta['seconds']:.1f}s)",
        flush=True,
    )
    return train_charts, valid_charts, meta


def _attach_chart_tensors(
    encoded: Sequence[Data], charts: Sequence[SupportChart]
) -> None:
    for data, chart in zip(encoded, charts):
        data.sup_size = torch.from_numpy(chart.sup_size)
        data.sup_root = torch.from_numpy(chart.sup_root)
        data.sup_patch = torch.from_numpy(chart.sup_patch)
        data.sup_single_node = torch.from_numpy(chart.sup_single_node)
        data.sup_single_bond = torch.from_numpy(chart.sup_single_bond)
        data.sup_node_ptr = torch.from_numpy(chart.sup_node_ptr)
        data.sup_node_idx = torch.from_numpy(chart.sup_node_idx)
        data.sup_edge_ptr = torch.from_numpy(chart.sup_edge_ptr)
        data.sup_edge_idx = torch.from_numpy(chart.sup_edge_idx)
        data.dec_child = torch.from_numpy(chart.dec_child)
        data.dec_parent_a = torch.from_numpy(chart.dec_parent_a)
        data.dec_parent_b = torch.from_numpy(chart.dec_parent_b)
        data.dec_overlap_ptr = torch.from_numpy(chart.dec_overlap_ptr)
        data.dec_overlap_idx = torch.from_numpy(chart.dec_overlap_idx)
        data.dec_cross_ptr = torch.from_numpy(chart.dec_cross_ptr)
        data.dec_cross_idx = torch.from_numpy(chart.dec_cross_idx)


def build_encoded_records():
    """Return (train_data, valid_data, audit) with base + chart tensors."""
    train_bundle, valid_bundle, _meta = sspe.extract_records()
    config = _v4_base_config()
    config["model"]["device"] = "cpu"
    enc_train, enc_valid, audit = zpp._phase_data(
        train_bundle["records"], valid_bundle["records"], config=config
    )
    sspe._attach_struct_tensors(enc_train, train_bundle["graphs"])
    sspe._attach_struct_tensors(enc_valid, valid_bundle["graphs"])
    train_charts, valid_charts, chart_meta = extract_charts()
    _attach_chart_tensors(enc_train, train_charts)
    _attach_chart_tensors(enc_valid, valid_charts)
    audit = dict(audit)
    audit["patch_representation"] = "binding_composition"
    audit["binding_composition_encoder"] = {
        "object_dim": BCE_OBJECT_DIM,
        "compose_hidden": BCE_COMPOSE_HIDDEN,
        "update_hidden": BCE_UPDATE_HIDDEN,
        "fusion_hidden": BCE_FUSION_HIDDEN,
        "output_dim": TOKEN_WIDTH,
        "max_support_size": MAX_SUPPORT_SIZE,
        "activation": BCE_ACTIVATION,
    }
    audit["support_chart"] = chart_meta
    audit["official_test_loaded"] = False
    return enc_train, enc_valid, audit


# ---------------------------------------------------------------------------
# structured batching
# ---------------------------------------------------------------------------

_STRUCT_KEYS = (
    "struct_atom",
    "struct_root",
    "struct_dist",
    "struct_patch",
    "struct_src",
    "struct_dst",
    "struct_bond",
    "sup_size",
    "sup_root",
    "sup_patch",
    "sup_single_node",
    "sup_single_bond",
    "sup_node_ptr",
    "sup_node_idx",
    "sup_edge_ptr",
    "sup_edge_idx",
    "dec_child",
    "dec_parent_a",
    "dec_parent_b",
    "dec_overlap_ptr",
    "dec_overlap_idx",
    "dec_cross_ptr",
    "dec_cross_idx",
)


def _offset_concat(items: Sequence[Any]) -> dict[str, torch.Tensor]:
    """Concatenate the structural / chart tensors with explicit offsets.

    Node ids, bond ids, patch ids and support ids are graph-local, so they are
    offset by the cumulative counts of the items already in the batch.  CSR
    pointers are rebuilt without the duplicate seam entry: each item contributes
    ``ptr[1:]`` shifted by its cumulative value offset and a single leading zero
    is prepended.  Sentinel ``-1`` entries are preserved.
    """
    plain_keys = (
        "struct_atom",
        "struct_root",
        "struct_dist",
        "struct_patch",
        "struct_src",
        "struct_dst",
        "struct_bond",
        "sup_size",
        "sup_root",
        "sup_patch",
        "sup_single_node",
        "sup_single_bond",
        "sup_node_idx",
        "sup_edge_idx",
        "dec_child",
        "dec_parent_a",
        "dec_parent_b",
        "dec_overlap_idx",
        "dec_cross_idx",
    )
    pointer_keys = (
        "sup_node_ptr",
        "sup_edge_ptr",
        "dec_overlap_ptr",
        "dec_cross_ptr",
    )
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in plain_keys}
    pointers: dict[str, list[torch.Tensor]] = {key: [] for key in pointer_keys}
    node_off = bond_off = patch_off = sup_off = 0
    node_val_off = edge_val_off = overlap_val_off = cross_val_off = 0

    def _shift_positive(value: torch.Tensor, offset: int) -> torch.Tensor:
        if offset == 0:
            return value
        return torch.where(value >= 0, value + offset, value)

    for data in items:
        parts["struct_atom"].append(data.struct_atom)
        parts["struct_root"].append(data.struct_root)
        parts["struct_dist"].append(data.struct_dist)
        parts["struct_patch"].append(data.struct_patch + patch_off)
        parts["struct_src"].append(data.struct_src + node_off)
        parts["struct_dst"].append(data.struct_dst + node_off)
        parts["struct_bond"].append(data.struct_bond)
        parts["sup_size"].append(data.sup_size)
        parts["sup_root"].append(data.sup_root)
        parts["sup_patch"].append(data.sup_patch + patch_off)
        parts["sup_single_node"].append(
            _shift_positive(data.sup_single_node, node_off)
        )
        parts["sup_single_bond"].append(
            _shift_positive(data.sup_single_bond, bond_off)
        )
        parts["sup_node_idx"].append(data.sup_node_idx + node_off)
        parts["sup_edge_idx"].append(data.sup_edge_idx + bond_off)
        parts["dec_child"].append(data.dec_child + sup_off)
        parts["dec_parent_a"].append(data.dec_parent_a + sup_off)
        parts["dec_parent_b"].append(data.dec_parent_b + sup_off)
        parts["dec_overlap_idx"].append(data.dec_overlap_idx + node_off)
        parts["dec_cross_idx"].append(data.dec_cross_idx + bond_off)
        pointers["sup_node_ptr"].append(data.sup_node_ptr[1:] + node_val_off)
        pointers["sup_edge_ptr"].append(data.sup_edge_ptr[1:] + edge_val_off)
        pointers["dec_overlap_ptr"].append(
            data.dec_overlap_ptr[1:] + overlap_val_off
        )
        pointers["dec_cross_ptr"].append(
            data.dec_cross_ptr[1:] + cross_val_off
        )

        node_off += int(data.struct_atom.numel())
        bond_off += int(data.struct_bond.numel()) // 2
        patch_off += int(data.struct_patch.max()) + 1
        sup_off += int(data.sup_size.numel())
        node_val_off += int(data.sup_node_idx.numel())
        edge_val_off += int(data.sup_edge_idx.numel())
        overlap_val_off += int(data.dec_overlap_idx.numel())
        cross_val_off += int(data.dec_cross_idx.numel())

    output: dict[str, torch.Tensor] = {}
    for key, value in parts.items():
        output[key] = (
            torch.cat(value) if value else torch.zeros(0, dtype=torch.long)
        )
    zero = torch.zeros(1, dtype=torch.long)
    for key, value in pointers.items():
        output[key] = (
            torch.cat([zero] + value)
            if value
            else torch.zeros(1, dtype=torch.long)
        )
    return output


def bc_collate(data_list: Sequence[Any]) -> Any:
    """Batch with explicit offsets for the base and support-chart tensors."""
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(list(data_list))
    fields = _offset_concat(list(data_list))
    for key, value in fields.items():
        setattr(batch, key, value)
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
        collate_fn=bc_collate,
    )


def _selection_loader(valid_data: Sequence[Any]):
    return _make_struct_loader(valid_data, 128, False, 0)


_evaluate_mae = sspe._evaluate_mae


# ---------------------------------------------------------------------------
# synthetic single-graph batches (correctness tests / sanity)
# ---------------------------------------------------------------------------


class _PlainGraph:
    """Minimal structural container for synthetic single-patch tests."""

    def __init__(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


def synthetic_graph(
    atom: Sequence[int],
    root_index: int,
    dist: Sequence[int],
    edges: Sequence[tuple[int, int]],
    bond_types: Sequence[int] | None = None,
) -> _PlainGraph:
    """Build one single-patch structural container via the real chart builder."""
    n = len(atom)
    if bond_types is None:
        bond_types = [0] * len(edges)
    undirected = sorted(
        ((int(min(u, v)), int(max(u, v))), int(bt))
        for (u, v), bt in zip(edges, bond_types)
    )
    src: list[int] = []
    dst: list[int] = []
    bond: list[int] = []
    for (u, v), bond_type in undirected:
        src.extend([u, v])
        dst.extend([v, u])
        bond.extend([int(bond_type), int(bond_type)])
    undirected = [pair for pair, _ in undirected]
    root = np.zeros(n, dtype=np.int64)
    root[int(root_index)] = 1
    chart = build_chart(
        n_nodes=n,
        patch_of_node=np.zeros(n, dtype=np.int64),
        n_patches=1,
        edge_u=np.asarray([u for u, _ in undirected], dtype=np.int64),
        edge_v=np.asarray([v for _, v in undirected], dtype=np.int64),
        edge_patch=np.zeros(len(undirected), dtype=np.int64),
        root_of_node=root.astype(bool),
        max_support_size=MAX_SUPPORT_SIZE,
    )
    fields = {
        "struct_atom": torch.tensor(atom, dtype=torch.long),
        "struct_root": torch.tensor(root, dtype=torch.long),
        "struct_dist": torch.tensor(dist, dtype=torch.long),
        "struct_patch": torch.zeros(n, dtype=torch.long),
        "struct_src": torch.tensor(src, dtype=torch.long),
        "struct_dst": torch.tensor(dst, dtype=torch.long),
        "struct_bond": torch.tensor(bond, dtype=torch.long),
        "sup_size": torch.from_numpy(chart.sup_size),
        "sup_root": torch.from_numpy(chart.sup_root),
        "sup_patch": torch.from_numpy(chart.sup_patch),
        "sup_single_node": torch.from_numpy(chart.sup_single_node),
        "sup_single_bond": torch.from_numpy(chart.sup_single_bond),
        "sup_node_ptr": torch.from_numpy(chart.sup_node_ptr),
        "sup_node_idx": torch.from_numpy(chart.sup_node_idx),
        "sup_edge_ptr": torch.from_numpy(chart.sup_edge_ptr),
        "sup_edge_idx": torch.from_numpy(chart.sup_edge_idx),
        "dec_child": torch.from_numpy(chart.dec_child),
        "dec_parent_a": torch.from_numpy(chart.dec_parent_a),
        "dec_parent_b": torch.from_numpy(chart.dec_parent_b),
        "dec_overlap_ptr": torch.from_numpy(chart.dec_overlap_ptr),
        "dec_overlap_idx": torch.from_numpy(chart.dec_overlap_idx),
        "dec_cross_ptr": torch.from_numpy(chart.dec_cross_ptr),
        "dec_cross_idx": torch.from_numpy(chart.dec_cross_idx),
    }
    return _PlainGraph(**fields)


def synthetic_batch(graphs: Sequence[_PlainGraph]) -> _PlainGraph:
    fields = _offset_concat(list(graphs))
    return _PlainGraph(**fields)


def relabel_graph(graph: _PlainGraph, perm: torch.Tensor) -> _PlainGraph:
    """Relabel the patch nodes consistently (all node-indexed arrays)."""
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(int(perm.numel()))
    fields = {
        "struct_atom": graph.struct_atom[perm],
        "struct_root": graph.struct_root[perm],
        "struct_dist": graph.struct_dist[perm],
        "struct_patch": graph.struct_patch[perm],
        # both endpoints must be mapped through the same inverse permutation
        "struct_src": inverse[graph.struct_src],
        "struct_dst": inverse[graph.struct_dst],
        "struct_bond": graph.struct_bond,
        "sup_size": graph.sup_size,
        "sup_root": graph.sup_root,
        "sup_patch": graph.sup_patch,
        "sup_single_node": torch.where(
            graph.sup_single_node >= 0,
            inverse[graph.sup_single_node.clamp_min(0)],
            graph.sup_single_node,
        ),
        "sup_single_bond": graph.sup_single_bond,
        "sup_node_ptr": graph.sup_node_ptr,
        "sup_node_idx": inverse[graph.sup_node_idx],
        "sup_edge_ptr": graph.sup_edge_ptr,
        "sup_edge_idx": graph.sup_edge_idx,
        "dec_child": graph.dec_child,
        "dec_parent_a": graph.dec_parent_a,
        "dec_parent_b": graph.dec_parent_b,
        "dec_overlap_ptr": graph.dec_overlap_ptr,
        "dec_overlap_idx": inverse[graph.dec_overlap_idx],
        "dec_cross_ptr": graph.dec_cross_ptr,
        "dec_cross_idx": graph.dec_cross_idx,
    }
    return _PlainGraph(**fields)


def reorder_edges(graph: _PlainGraph) -> _PlainGraph:
    """Reverse the storage order of the undirected bonds (order invariance)."""
    n_bonds = int(graph.struct_bond.numel()) // 2
    order = torch.arange(n_bonds - 1, -1, -1, dtype=torch.long)
    src = graph.struct_src.view(n_bonds, 2)
    dst = graph.struct_dst.view(n_bonds, 2)
    bond = graph.struct_bond.view(n_bonds, 2)
    remap = torch.empty(n_bonds, dtype=torch.long)
    remap[order] = torch.arange(n_bonds)
    fields = {
        key: getattr(graph, key) for key in _STRUCT_KEYS
    }
    fields["struct_src"] = src[order].reshape(-1)
    fields["struct_dst"] = dst[order].reshape(-1)
    fields["struct_bond"] = bond[order].reshape(-1)
    for key in ("sup_single_bond", "sup_edge_idx", "dec_cross_idx"):
        value = getattr(graph, key)
        fields[key] = torch.where(value >= 0, remap[value.clamp_min(0)], value)
    return _PlainGraph(**fields)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    baseline = cd.build_cell(CELL, 0)
    candidate = build_candidate(0)
    base_total = _n_params(baseline)
    cand_total = _n_params(candidate)
    if base_total != 85763:
        raise RuntimeError(f"cell-A baseline params changed: {base_total} != 85763")
    encoder = candidate.structural_encoder
    breakdown = _encoder_breakdown(encoder)
    audit = zpp.audit_parameters(candidate)
    typed_keys = [key for key in candidate.state_dict() if "typed_embedding" in key]
    dataset_dependent = 0  # all embeddings are fixed categorical feature maps
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cell": CELL,
        "baseline_params": int(base_total),
        "candidate_params": int(cand_total),
        "params_in_range": bool(cand_total <= PARAM_UPPER),
        "binding_composition_encoder_params": int(_n_params(encoder)),
        "encoder_breakdown": breakdown,
        "inherited_downstream_params": int(cand_total - _n_params(encoder)),
        "dataset_dependent_params": int(dataset_dependent),
        "candidate_has_typed_embedding": bool(
            getattr(candidate, "typed_embedding", None) is not None
        ),
        "candidate_typed_state_keys": typed_keys,
        "candidate_block_params": audit["blocks"],
        "candidate_dimensions": audit["dimensions"],
        "q_dim": int(candidate.pair_hidden),
        "h_dim": int(candidate.patch_hidden),
        "recurrence_rounds": int(candidate.recurrence_rounds),
        "recurrence_mode": str(candidate.recurrence_mode),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# candidate / cache audit
# ---------------------------------------------------------------------------


def preprocess(force: bool = False) -> dict[str, Any]:
    _train, _valid, meta = extract_charts(force=force)
    return meta


def candidate_audit(subset: int | None = None) -> dict[str, Any]:
    """Target-free support / decomposition scale + forward-cost audit."""
    started = time.perf_counter()
    if subset is not None:
        train_charts, valid_charts, meta = extract_charts(
            force=True, subset=int(subset)
        )
    else:
        train_charts, valid_charts, meta = extract_charts(force=False)
    stats = {
        "protocol_version": PROTOCOL_VERSION,
        "subset_molecules": None if subset is None else int(subset),
        "chart_meta": meta,
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }

    # forward-cost probe on a real (small) batch if the full cache exists
    if subset is None:
        try:
            train_data, valid_data, _audit = build_encoded_records()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_candidate(0).to(device)
            loader = _make_struct_loader(list(valid_data)[:32], 32, False, 0)
            batch = next(iter(loader)).to(device)
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = model(batch)
            loss = F.l1_loss(out.view(-1), batch.y.view(-1))
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            stats["forward_backward_seconds_32_molecules"] = float(
                time.perf_counter() - t0
            )
            stats["batch_decompositions"] = int(batch.dec_child.numel())
            stats["batch_supports"] = int(batch.sup_size.numel())
            stats["candidate_params"] = int(_n_params(model))
            if device.type == "cuda":
                stats["peak_gpu_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
            else:
                stats["peak_gpu_memory_bytes"] = 0
            stats["probe_ok"] = True
        except Exception as exc:  # pragma: no cover - audit must not crash
            stats["probe_ok"] = False
            stats["probe_error"] = repr(exc)
    _write_json(RESULTS_DIR / "candidate_audit.json", stats)
    return stats


# ---------------------------------------------------------------------------
# sanity
# ---------------------------------------------------------------------------

_SANITY_GRAPHS = {
    "single_edge": dict(
        atom=[0, 1], root_index=0, dist=[0, 1], edges=[(0, 1)], bond_types=[0]
    ),
    "path3": dict(
        atom=[0, 1, 2],
        root_index=0,
        dist=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        bond_types=[0, 1],
    ),
    "triangle": dict(
        atom=[0, 1, 2],
        root_index=0,
        dist=[0, 1, 1],
        edges=[(0, 1), (0, 2), (1, 2)],
        bond_types=[0, 1, 2],
    ),
    "branch": dict(
        atom=[0, 1, 2, 3, 4],
        root_index=0,
        dist=[0, 1, 1, 2, 2],
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
        bond_types=[0, 1, 0, 2],
    ),
    "four_cycle": dict(
        atom=[0, 1, 2, 3],
        root_index=0,
        dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        bond_types=[0, 1, 2, 0],
    ),
    "ambiguous": dict(
        atom=[0, 1, 2, 3],
        root_index=0,
        dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
        bond_types=[0, 1, 2, 0, 1],
    ),
}


def _encoder(seed: int = 0) -> BindingCompositionEncoder:
    torch.manual_seed(int(seed))
    return BindingCompositionEncoder(
        object_dim=BCE_OBJECT_DIM,
        compose_hidden=BCE_COMPOSE_HIDDEN,
        update_hidden=BCE_UPDATE_HIDDEN,
        fusion_hidden=BCE_FUSION_HIDDEN,
        output_dim=TOKEN_WIDTH,
        max_support_size=MAX_SUPPORT_SIZE,
        activation=BCE_ACTIVATION,
    ).eval()


def sanity() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    encoder = _encoder(0)
    device = torch.device("cpu")

    # --- structural invariants on synthetic patches ------------------------
    invariances: dict[str, float] = {}
    for name, spec in _SANITY_GRAPHS.items():
        graph = synthetic_graph(**spec)
        with torch.no_grad():
            reference = encoder(graph)
        n = int(graph.struct_atom.numel())
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(17))
        with torch.no_grad():
            relabelled = encoder(relabel_graph(graph, perm))
        with torch.no_grad():
            reordered = encoder(reorder_edges(graph))
        invariances[f"{name}_relabel"] = float(
            (reference - relabelled).abs().max()
        )
        invariances[f"{name}_edge_order"] = float(
            (reference - reordered).abs().max()
        )
    checks["node_relabel_invariant"] = all(
        value < 1.0e-5
        for key, value in invariances.items()
        if key.endswith("_relabel")
    )
    checks["edge_order_invariant"] = all(
        value < 1.0e-5
        for key, value in invariances.items()
        if key.endswith("_edge_order")
    )

    # --- binding sensitivity (atom / bond changes must move e_struct) ------
    base = synthetic_graph(**_SANITY_GRAPHS["path3"])
    with torch.no_grad():
        e_ref = encoder(base)
    atom_changed = synthetic_graph(
        atom=[0, 3, 2], root_index=0, dist=[0, 1, 2], edges=[(0, 1), (1, 2)],
        bond_types=[0, 1],
    )
    with torch.no_grad():
        e_atom = encoder(atom_changed)
    bond_changed = synthetic_graph(
        atom=[0, 1, 2], root_index=0, dist=[0, 1, 2], edges=[(0, 1), (1, 2)],
        bond_types=[2, 1],
    )
    with torch.no_grad():
        e_bond = encoder(bond_changed)
    checks["atom_type_changes_output"] = bool(
        float((e_ref - e_atom).abs().max()) > 1.0e-6
    )
    checks["bond_type_changes_output"] = bool(
        float((e_ref - e_bond).abs().max()) > 1.0e-6
    )

    # --- connectivity sensitivity: same multisets, different connectivity --
    left = synthetic_graph(
        atom=[0, 1, 2, 3], root_index=0, dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (1, 3)], bond_types=[0, 0, 0],
    )
    right = synthetic_graph(
        atom=[0, 1, 2, 3], root_index=0, dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (2, 3)], bond_types=[0, 0, 0],
    )
    with torch.no_grad():
        e_left = encoder(left)
        e_right = encoder(right)
    checks["connectivity_changes_output"] = bool(
        float((e_left - e_right).abs().max()) > 1.0e-6
    )

    # --- edge-type multiset identical but connectivity different -----------
    checks["no_gat_or_attention"] = bool(
        not any(
            "attention" in name.lower()
            or "gate" in name.lower()
            or "score" in name.lower()
            for name, _ in encoder.named_modules()
        )
    )
    checks["no_vocabulary_parameters"] = bool(
        all(
            parameter.shape[0] <= 64
            for name, parameter in encoder.named_parameters()
            if parameter.ndim >= 2 and "embedding" in name
        )
    )

    # --- gradient viability -------------------------------------------------
    encoder.train()
    graph = synthetic_graph(**_SANITY_GRAPHS["ambiguous"])
    out = encoder(graph)
    loss = out.pow(2).mean()
    loss.backward()
    grads = {
        "atom_mlp": _module_grad_norm(encoder.atom_mlp),
        "bind_mlp": _module_grad_norm(encoder.bind_mlp),
        "compose": _module_grad_norm(encoder.compose),
        "support_update": _module_grad_norm(encoder.support_update),
        "fusion": _module_grad_norm(encoder.fusion),
    }
    checks["compose_grad_nonzero"] = grads["compose"] > 0.0
    checks["binding_grad_nonzero"] = grads["bind_mlp"] > 0.0
    checks["support_update_grad_nonzero"] = grads["support_update"] > 0.0
    checks["fusion_grad_nonzero"] = grads["fusion"] > 0.0
    checks["atom_grad_nonzero"] = grads["atom_mlp"] > 0.0

    # --- batch invariance ---------------------------------------------------
    combined = synthetic_batch([graph, left, right])
    encoder.eval()
    with torch.no_grad():
        batched = encoder(combined)
    with torch.no_grad():
        singles = torch.cat([encoder(graph), encoder(left), encoder(right)])
    checks["batch_invariance"] = bool(
        float((batched - singles).abs().max()) < 1.0e-5
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "invariance_max_abs_diff": invariances,
        "gradient_norms": grads,
        "encoder_params": int(_n_params(encoder)),
        "environment": _environment_fingerprint(str(device)),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"BCE sanity failed: {failed}")
    return payload


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return float(math.sqrt(total))


# ---------------------------------------------------------------------------
# mechanism diagnostics helpers
# ---------------------------------------------------------------------------


def _bce_module_grad_norms(model: nn.Module) -> dict[str, float]:
    encoder = model.structural_encoder
    return {
        "atom_mlp_grad_norm": _module_grad_norm(encoder.atom_mlp),
        "bind_mlp_grad_norm": _module_grad_norm(encoder.bind_mlp),
        "compose_grad_norm": _module_grad_norm(encoder.compose),
        "support_update_grad_norm": _module_grad_norm(encoder.support_update),
        "size_embedding_grad_norm": _module_grad_norm(encoder.size_embedding),
        "fusion_grad_norm": _module_grad_norm(encoder.fusion),
        "patch_encoder_grad_norm": _module_grad_norm(model.patch_encoder),
    }


_STAT_MEAN_SUFFIXES = (
    "object_norm_mean_size{s}",
    "object_norm_std_size{s}",
)


def _collect_mechanism_stats(
    encoder: BindingCompositionEncoder, batches: Sequence[Any]
) -> dict[str, float]:
    encoder.capture_diagnostics = True
    accumulators: dict[str, float] = {}
    weights: dict[str, float] = {}
    try:
        for batch in batches:
            with torch.no_grad():
                encoder(batch)
            stats = encoder.last_stats
            weight = float(stats.get("n_supports", 0) or 1)
            for key, value in stats.items():
                if not isinstance(value, (int, float)):
                    continue
                accumulators[key] = accumulators.get(key, 0.0) + float(value) * weight
                weights[key] = weights.get(key, 0.0) + weight
    finally:
        encoder.capture_diagnostics = False
    return {
        key: accumulators[key] / max(weights[key], 1.0e-12)
        for key in accumulators
    }


def _encoder_output_matrix(
    encoder: BindingCompositionEncoder, batches: Sequence[Any]
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for batch in batches:
        with torch.no_grad():
            blocks.append(encoder(batch).cpu().numpy())
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, TOKEN_WIDTH))


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


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def _selection_loader_valid(valid_data: Sequence[Any]):
    return _make_struct_loader(valid_data, 128, False, 0)


def train_seed(
    seed: int,
    device: str = "cpu",
    tag: str = "bce",
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

    diag_loader = _make_struct_loader(list(valid_data)[:DIAG_BATCHES], 128, False, 0)
    diag_batches = [b.to(device_obj) for b in diag_loader]

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
                last_grads = _bce_module_grad_norms(model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(protocol["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae = _evaluate_mae(model, eval_loader, device_obj)
        stats = _collect_mechanism_stats(model.structural_encoder, diag_batches)
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
                        "object_norm_mean_size1",
                        "object_norm_mean_size2",
                        "object_norm_mean_size3",
                        "object_norm_mean_size4",
                        "object_norm_std_size2",
                        "object_norm_std_size3",
                        "object_norm_std_size4",
                        "candidate_norm_mean",
                        "candidate_norm_std",
                        "candidate_across_decomposition_std_mean",
                        "composition_disagreement_mean",
                        "multi_decomposition_candidate_count",
                        "higher_object_norm_mean",
                        "output_norm_mean",
                        "output_norm_std",
                        "support_count_size3",
                        "support_count_size4",
                    }
                },
                **{
                    f"{key}": float(value)
                    for key, value in last_grads.items()
                },
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
                f"compose_grad={last_grads.get('compose_grad_norm', 0.0):.3e} "
                f"disagree={stats.get('composition_disagreement_mean', 0.0):.4f}",
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
        "patch_representation": "binding_composition",
        "binding_composition_encoder_params": int(
            _n_params(model.structural_encoder)
        ),
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


def soup_seed(seed: int, tag: str = "bce") -> dict[str, Any]:
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
    loader = _selection_loader_valid(valid_data)
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


def train_queue(seeds: Sequence[int], device: str, tag: str = "bce") -> None:
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
    """GPU smoke: forward / backward / optimizer steps / mechanism liveness."""
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_data, valid_data, _audit = build_encoded_records()
    model = build_candidate(0).to(device_obj)
    loader = _make_struct_loader(list(valid_data)[:64], 64, False, 0)
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
    stats = _collect_mechanism_stats(
        model.structural_encoder, [batch]
    )
    grads = _bce_module_grad_norms(model)
    stats["gradient_norms"] = grads
    matrix = _encoder_output_matrix(model.structural_encoder, [batch])
    rank = _effective_rank(matrix)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "device": str(device),
        "steps": int(steps),
        "losses": losses,
        "losses_finite": bool(finite),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "mechanism_stats": stats,
        "output_effective_rank": rank,
        "compose_grad_nonzero": grads["compose_grad_norm"] > 0.0,
        "support_update_grad_nonzero": grads["support_update_grad_norm"] > 0.0,
        "binding_grad_nonzero": grads["bind_mlp_grad_norm"] > 0.0,
        "fusion_grad_nonzero": grads["fusion_grad_norm"] > 0.0,
        "object_states_nontrivial": bool(
            stats.get("object_norm_std_size3", 0.0) > 0.0
            or stats.get("object_norm_std_size4", 0.0) > 0.0
        ),
        "composition_disagreement_nonzero": bool(
            stats.get("composition_disagreement_mean", 0.0) > 0.0
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device_obj.type == "cuda"
            else 0
        ),
        "output_not_constant": bool(rank["effective_rank"] > 0.1),
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
# diagnostics / disagreement / provenance examples
# ---------------------------------------------------------------------------


def _diagnostic_batches(count: int = 256) -> list[Any]:
    _train, valid_data, _audit = build_encoded_records()
    loader = _make_struct_loader(
        list(valid_data)[:count], 128, False, 0
    )
    return [b.to("cpu") for b in loader]


def diagnostics(tag: str = "bce") -> dict[str, Any]:
    device = torch.device("cpu")
    batches = _diagnostic_batches(256)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "patch_representation": "binding_composition",
        "per_seed": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        if not state_path.exists():
            continue
        model = build_candidate(seed).to(device)
        model.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=True)
        )
        encoder = model.structural_encoder
        matrix = _encoder_output_matrix(encoder, batches)
        stats = _collect_mechanism_stats(encoder, batches)
        # gradient viability at the selection checkpoint (single real backward)
        model.train()
        diagnostic_batch = batches[0]
        prediction = model(diagnostic_batch).view(-1)
        loss = F.l1_loss(prediction, diagnostic_batch.y.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = _bce_module_grad_norms(model)
        model.zero_grad(set_to_none=True)
        payload["per_seed"][str(seed)] = {
            "output_effective_rank": _effective_rank(matrix),
            "mechanism_stats": stats,
            "gradient_norms": grads,
            "n_output_samples": int(matrix.shape[0]),
        }
    _write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


def disagreement(tag: str = "bce") -> dict[str, Any]:
    """Per-support composition disagreement at the selection checkpoints.

    For every support with more than one legal decomposition the encoder records
    ``mean_j ||c_j - mean_j c_j||``; this is the stable "did Compose actually use
    the composition" statistic (much more direct than a final output rank).
    """
    device = torch.device("cpu")
    batches = _diagnostic_batches(128)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        if not state_path.exists():
            continue
        model = build_candidate(seed).to(device)
        model.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=True)
        )
        encoder = model.structural_encoder
        per_size: dict[str, list[float]] = {}
        multi = 0
        single = 0
        total = 0
        for batch in batches:
            with torch.no_grad():
                encoder(batch)
            counts = encoder.last_support_counts.cpu().numpy()
            spread = encoder.last_support_disagreement.cpu().numpy()
            sizes = batch.sup_size.cpu().numpy()
            multi += int((counts > 1).sum())
            single += int((counts == 1).sum())
            total += int(counts.shape[0])
            for size in range(3, MAX_SUPPORT_SIZE + 1):
                mask = (sizes == size) & (counts > 1)
                if mask.any():
                    per_size.setdefault(str(size), []).extend(
                        spread[mask].tolist()
                    )
        payload["per_seed"][str(seed)] = {
            "total_supports": int(total),
            "multi_decomposition_supports": int(multi),
            "single_decomposition_supports": int(single),
            "disagreement_by_size": {
                size: {
                    "n": int(len(values)),
                    "mean": float(np.mean(values)) if values else 0.0,
                    "p50": float(np.percentile(values, 50)) if values else 0.0,
                    "p90": float(np.percentile(values, 90)) if values else 0.0,
                    "max": float(np.max(values)) if values else 0.0,
                }
                for size, values in per_size.items()
            },
        }
    _write_json(RESULTS_DIR / "disagreement.json", payload)
    return payload


def support_examples(tag: str = "bce", max_patches: int = 6) -> dict[str, Any]:
    """Explicit support / provenance examples on fixed valid molecules."""
    device = torch.device("cpu")
    _train, valid_data, _audit = build_encoded_records()
    loader = _make_struct_loader(list(valid_data)[:1], 1, False, 0)
    batch = next(iter(loader)).to(device)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
        if not state_path.exists():
            continue
        model = build_candidate(seed).to(device)
        model.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=True)
        )
        encoder = model.structural_encoder
        with torch.no_grad():
            encoder(batch)
        states = encoder.last_support_states
        payload["per_seed"][str(seed)] = _describe_supports(
            batch, states, max_patches=int(max_patches)
        )
    _write_json(RESULTS_DIR / "support_examples.json", payload)
    return payload


def _describe_supports(batch: Any, states: torch.Tensor, max_patches: int) -> Any:
    sup_patch = batch.sup_patch.cpu().numpy()
    sup_size = batch.sup_size.cpu().numpy()
    sup_root = batch.sup_root.cpu().numpy()
    node_ptr = batch.sup_node_ptr.cpu().numpy()
    node_idx = batch.sup_node_idx.cpu().numpy()
    edge_ptr = batch.sup_edge_ptr.cpu().numpy()
    edge_idx = batch.sup_edge_idx.cpu().numpy()
    dec_child = batch.dec_child.cpu().numpy()
    dec_parent_a = batch.dec_parent_a.cpu().numpy()
    dec_parent_b = batch.dec_parent_b.cpu().numpy()
    atom = batch.struct_atom.cpu().numpy()
    bond = batch.struct_bond.cpu().numpy()
    states_np = states.detach().cpu().numpy()
    parents: dict[int, list[list[int]]] = {}
    for child, parent_a, parent_b in zip(dec_child, dec_parent_a, dec_parent_b):
        parents.setdefault(int(child), []).append([int(parent_a), int(parent_b)])

    patches = sorted(set(sup_patch.tolist()))[: int(max_patches)]
    described = []
    for patch in patches:
        indices = [i for i in range(len(sup_size)) if int(sup_patch[i]) == patch]
        supports = []
        for index in indices:
            node_ids = [
                int(v) for v in node_idx[node_ptr[index] : node_ptr[index + 1]]
            ]
            edge_ids = [
                int(v) for v in edge_idx[edge_ptr[index] : edge_ptr[index + 1]]
            ]
            supports.append(
                {
                    "support_index": int(index),
                    "size": int(sup_size[index]),
                    "contains_root": bool(sup_root[index]),
                    "node_ids": node_ids,
                    "atom_types": [int(atom[v]) for v in node_ids],
                    "induced_bond_ids": edge_ids,
                    "induced_bond_types": [int(bond[2 * b]) for b in edge_ids],
                    "state_norm": float(np.linalg.norm(states_np[index])),
                    "decompositions": parents.get(int(index), []),
                }
            )
        described.append({"patch": int(patch), "supports": supports})
    return described


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def _mechanism_viability(diag: Mapping[str, Any]) -> dict[str, Any]:
    """Section-22 mechanism viability from the selection-checkpoint diagnostics."""
    per_seed = diag.get("per_seed", {})
    result: dict[str, Any] = {}
    for seed, entry in per_seed.items():
        stats = entry.get("mechanism_stats", {})
        grads = entry.get("gradient_norms", {})
        rank = entry.get("output_effective_rank", {})
        result[seed] = {
            "compose_grad_nonzero": bool(
                grads.get("compose_grad_norm", 0.0) > 0.0
            ),
            "support_update_grad_nonzero": bool(
                grads.get("support_update_grad_norm", 0.0) > 0.0
            ),
            "binding_grad_nonzero": bool(grads.get("bind_mlp_grad_norm", 0.0) > 0.0),
            "fusion_grad_nonzero": bool(grads.get("fusion_grad_norm", 0.0) > 0.0),
            "object_states_nontrivial": bool(
                stats.get("object_norm_std_size3", 0.0) > 0.0
                or stats.get("object_norm_std_size4", 0.0) > 0.0
            ),
            "composition_disagreement_nonzero": bool(
                stats.get("composition_disagreement_mean", 0.0) > 0.0
            ),
            "output_not_constant": bool(
                rank.get("effective_rank", 0.0) > 0.1
            ),
            "output_effective_rank": float(rank.get("effective_rank", 0.0)),
        }
        result[seed]["viable"] = bool(
            all(
                result[seed][key]
                for key in (
                    "compose_grad_nonzero",
                    "support_update_grad_nonzero",
                    "binding_grad_nonzero",
                    "fusion_grad_nonzero",
                    "object_states_nontrivial",
                    "composition_disagreement_nonzero",
                    "output_not_constant",
                )
            )
        )
    return result


def decide(tag: str = "bce") -> dict[str, Any]:
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
    viability = _mechanism_viability(diag)

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
            "mechanism_viability": viability,
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
    alive = {
        seed: entry["viable"] for seed, entry in viability.items()
    }
    composition_alive = bool(alive) and all(alive.values())

    if composition_alive and delta_bfull <= 0.0:
        case = "A_composition_alive_at_or_above_bfull"
    elif composition_alive and delta_bag < 0 < delta_bfull:
        case = "B_composition_alive_above_bbag_below_bfull"
    elif composition_alive and abs(delta_bag) < STRONG_GATE:
        case = "C_composition_alive_tied_with_bbag"
    elif not composition_alive:
        case = "D_composition_branch_dead_or_constant"
    elif delta_bag >= REGRESSION_GATE and same_direction_bag:
        case = "bad_regression"
    else:
        case = "E_inconclusive"

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
        "mechanism_viability": viability,
        "composition_alive": composition_alive,
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
        "candidate_audit": _maybe("candidate_audit.json"),
        "sanity": (lambda d: None if d is None else d["checks"])(_maybe("sanity.json")),
        "smoke": _maybe("smoke.json"),
        "diagnostics": _maybe("diagnostics.json"),
        "disagreement": _maybe("disagreement.json"),
        "decision": _maybe("decision.json"),
        "support_examples": _maybe("support_examples.json"),
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
            "candidate_audit",
            "sanity",
            "smoke",
            "train",
            "train_queue",
            "repro",
            "soup",
            "diagnostics",
            "disagreement",
            "support_examples",
            "decide",
            "report",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="bce")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--subset", type=int, default=None)
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
        print(
            json.dumps(preprocess(force=bool(args.force)), indent=2, default=str),
            flush=True,
        )
    if args.stage == "candidate_audit":
        print(
            json.dumps(candidate_audit(subset=args.subset), indent=2, default=str),
            flush=True,
        )
    if args.stage == "sanity":
        print(json.dumps(sanity()["checks"], indent=2, default=str), flush=True)
    if args.stage == "smoke":
        print(
            json.dumps(
                smoke(device=args.device, steps=args.steps),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "train":
        print(
            json.dumps(
                train_seed(args.seed, device=args.device, tag=args.tag),
                indent=2,
                default=str,
            ),
            flush=True,
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
                repro(
                    args.seed,
                    args.epochs,
                    args.device,
                    run_tag=args.run_tag or "default",
                ),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "diagnostics":
        print(json.dumps(diagnostics(tag=args.tag), indent=2, default=str), flush=True)
    if args.stage == "disagreement":
        print(json.dumps(disagreement(tag=args.tag), indent=2, default=str), flush=True)
    if args.stage == "support_examples":
        print(
            json.dumps(support_examples(tag=args.tag), indent=2, default=str),
            flush=True,
        )
    if args.stage == "decide":
        print(json.dumps(decide(tag=args.tag), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
