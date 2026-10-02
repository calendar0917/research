"""Typed-cycle object screen (Phase A) — label-free train-only feature extraction.

Round ``zinc_e2e_dictenv_typed_cycle_v1`` (Workstream Z, ZINC).
Pre-registration: ``tracks/ksvd/notes/zinc_e2e_dictenv_typed_cycle_v1_preregistration.md``.

This module builds the **cheap, label-free** diagnostic features used to decide
whether a typed static-cycle object is worth one formal training purchase.  It
does **not** implement the model; the model lives in
``e2e_dictenv_typed_cycle_v1.py``.

Two raw arm matrices (both deterministic functions of the raw graph, never of
the labels):

* ``X_base`` (573 dims)
    1. per-root raw ``Sem110`` graph-level sum + sum-of-squares (220)
    2. per-root raw ``phi65`` graph-level sum + sum-of-squares (130)
    3. raw ``global62`` + cached ``topology25`` (87)
    4. untyped chordless-ring length-3..10 counts (8)
    5. ``edge_type_counts`` fixed 128-dim atom-pair x bond-type counts (128)
* ``X_typed`` (1040 dims)
    the vendored ``cheap_probe_features.ring_features`` typed block: per-length
    atom/bond composition and second moments, cyclic bond-pair separation and a
    fixed hash of partial typed patterns (partial mode only, no complete-ring
    token, no label-based vocabulary).

The object census (per-graph object count, length distribution, 2L dihedral
views, acyclic fraction, p95/p99/max, timing) is recorded alongside so the
resource decision is auditable.  Official valid/test are never read: only the
official train split is instantiated and the test path is blocked.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# vendored source-derived NumPy reference (feature authority for the screen)
# ---------------------------------------------------------------------------
REFERENCE_DIR = (
    Path(__file__).resolve().parent / "typed_cycle_reference" / "v1_20261002"
)
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

import typed_cycle_reference as _tcr  # noqa: E402
import cheap_probe_features as _cpf  # noqa: E402

from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as _fsar  # noqa: E402
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as _ztf  # noqa: E402
from tracks.ksvd.experiments.luyin16.identity_incremental_information_audit import (  # noqa: E402
    _raw_zinc_certificate,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (  # noqa: E402
    _data_to_graph,
    global_feature_views,
)

PROTOCOL_VERSION = "typed_cycle_probe_v1"

#: frozen feature widths (closed form; asserted against every row).
SEM110_DIM = 110
PHI65_DIM = 65
GLOBAL_DIM = 62
TOPOLOGY_DIM = 25
UNTYPED_RING_DIM = 8
EDGE_TYPE_DIM = 128
BASE_DIM = (
    2 * SEM110_DIM
    + 2 * PHI65_DIM
    + GLOBAL_DIM
    + TOPOLOGY_DIM
    + UNTYPED_RING_DIM
    + EDGE_TYPE_DIM
)  # 573
TYPED_RING_DIM = int(_cpf.TYPED_RING_WIDTH)  # 1040
MIN_CYCLE_LEN, MAX_CYCLE_LEN = 3, 10

ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4

OFFICIAL_TRAIN_SIZE = 10_000
OFFICIAL_VALID_SIZE = 1_000
OFFICIAL_TEST_SIZE = 1_000


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


# ---------------------------------------------------------------------------
# raw graph conversion (repo graph -> vendored reference typed graph)
# ---------------------------------------------------------------------------
def to_reference_graph(graph: Any, node_types: np.ndarray, edge_types: Mapping[tuple[int, int], int]) -> Any:
    """Convert a repo ``Graph`` + type arrays into the reference typed graph.

    Directed duplicates are collapsed to unique undirected edges; every edge
    must be present in ``edge_types``; a self loop is rejected.
    """
    nodes = list(graph.nodes)
    atoms = tuple(int(node_types[node]) for node in nodes)
    edges: list[tuple[int, int, int]] = []
    for u, v in graph.edges():
        if int(u) == int(v):
            raise RuntimeError("self loop in ZINC raw graph")
        key = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        if key not in edge_types:
            raise RuntimeError(f"missing bond type for edge {key}")
        edges.append((key[0], key[1], int(edge_types[key])))
    edges.sort()
    return _tcr.Graph(atoms, tuple(edges))


# ---------------------------------------------------------------------------
# per-graph raw summaries
# ---------------------------------------------------------------------------
def sem110_summary(graph: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = _tcr.semantic110(graph)  # [n, 110]
    if rows.shape[1] != SEM110_DIM:
        raise RuntimeError(f"Sem110 width {rows.shape[1]} != {SEM110_DIM}")
    return rows.sum(axis=0), (rows * rows).sum(axis=0)


def phi65_summary(graph: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = _fsar.build_phi(graph)  # [n, 65]
    if rows.shape[1] != PHI65_DIM:
        raise RuntimeError(f"phi65 width {rows.shape[1]} != {PHI65_DIM}")
    return rows.sum(axis=0), (rows * rows).sum(axis=0)


def ring_census(graph: Any) -> dict[str, Any]:
    """Enumerate chordless cycles once and describe the object population.

    Uses the vendored ``chordless_cycles`` early-chord-pruning enumerator.  A
    length-``L`` ring contributes ``2L`` dihedral views; the census reports the
    exact integer view multiset size.
    """
    cycles = _tcr.chordless_cycles(graph, max_length=MAX_CYCLE_LEN)
    length_counts = np.zeros(UNTYPED_RING_DIM, dtype=np.float64)
    n_views = 0
    for cycle in cycles:
        length = len(cycle)
        length_counts[length - MIN_CYCLE_LEN] += 1.0
        n_views += 2 * length
    return {
        "n_cycles": int(len(cycles)),
        "n_views": int(n_views),
        "length_counts": length_counts,
        "max_length": int(length_counts.nonzero()[0].max() + MIN_CYCLE_LEN) if len(cycles) else 0,
    }


def graph_feature_row(
    data: Any, topology_row: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(base573, typed1040, census)`` for one official-train molecule."""
    graph, node_types, edge_types = _data_to_graph(data)
    ref_graph = to_reference_graph(graph, node_types, edge_types)

    sem_sum, sem_sumsq = sem110_summary(ref_graph)
    phi_sum, phi_sumsq = phi65_summary(graph)

    global_row = global_feature_views([data])["global_all"][0]
    if global_row.shape[0] != GLOBAL_DIM:
        raise RuntimeError(f"global width {global_row.shape[0]} != {GLOBAL_DIM}")
    if topology_row is None:
        topology_row = np.zeros(TOPOLOGY_DIM, dtype=np.float64)
    topology_row = np.asarray(topology_row, dtype=np.float64).reshape(-1)
    if topology_row.shape[0] != TOPOLOGY_DIM:
        raise RuntimeError(f"topology width {topology_row.shape[0]} != {TOPOLOGY_DIM}")

    untyped, typed = _cpf.ring_features(ref_graph)
    edge_counts = _cpf.edge_type_counts(ref_graph)

    base = np.concatenate(
        [
            sem_sum,
            sem_sumsq,
            phi_sum,
            phi_sumsq,
            np.asarray(global_row, dtype=np.float64),
            topology_row,
            np.asarray(untyped, dtype=np.float64),
            np.asarray(edge_counts, dtype=np.float64),
        ]
    ).astype(np.float64)
    typed = np.asarray(typed, dtype=np.float64).reshape(-1)
    if base.shape[0] != BASE_DIM:
        raise RuntimeError(f"X_base width {base.shape[0]} != {BASE_DIM}")
    if typed.shape[0] != TYPED_RING_DIM:
        raise RuntimeError(f"X_typed width {typed.shape[0]} != {TYPED_RING_DIM}")

    census = ring_census(ref_graph)
    census["n_nodes"] = int(graph.n)
    census["n_edges"] = int(graph.num_edges())
    return base, typed, census


# ---------------------------------------------------------------------------
# dataset build (train only)
# ---------------------------------------------------------------------------
def load_official_train(root: Path):
    """Instantiate the official ZINC ``subset=True`` **train split only**.

    PyG's ``ZINC(split="train")`` reads ``raw/train.index`` + ``train.pickle``;
    it never instantiates valid/test.  Test access is additionally guarded by
    the caller and by the probe CLI.
    """
    from torch_geometric.datasets import ZINC

    return ZINC(root=str(root), subset=True, split="train")


def topology_matrix_train(root: Path, dataset: Sequence[Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load (never recompute) the verified ``topology25`` hinge cache for train."""
    frame = _ztf.load_cache("train")
    if frame is None:
        raise RuntimeError("verified topology cache missing; refusing to rebuild silently")
    if len(frame) != OFFICIAL_TRAIN_SIZE:
        raise RuntimeError(
            f"topology cache length {len(frame)} != official train {OFFICIAL_TRAIN_SIZE}"
        )
    if len(dataset) > len(frame):
        raise RuntimeError(
            f"dataset length {len(dataset)} exceeds topology cache {len(frame)}"
        )
    matrix = np.stack(
        [_ztf.raw_vector(frame.iloc[index], "hinge") for index in range(len(frame))]
    ).astype(np.float64)
    if len(dataset) < len(frame):
        # smoke / budget-limit prefix: the official split order is unchanged.
        matrix = matrix[: len(dataset)]
    meta_path = _ztf.CACHE_ROOT / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta = dict(meta)
    meta["loaded_rows"] = int(len(matrix))
    return matrix, meta


def group_ids_train(dataset: Sequence[Any], *, limit: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Verified canonical molecule keys (pynauty certificate + coloured sequence).

    Reuses ``identity_incremental_information_audit._raw_zinc_certificate`` —
    an existing, repository-verified canonical key — so duplicate molecules
    share a group and cannot leak across folds.  No new isomorphism machinery
    is introduced.
    """
    stop = len(dataset) if limit is None else min(int(limit), len(dataset))
    digest_to_id: dict[bytes, int] = {}
    ids = np.empty(stop, dtype=np.int64)
    for index in range(stop):
        digest = hashlib.sha256(_raw_zinc_certificate(dataset[index])).digest()
        if digest not in digest_to_id:
            digest_to_id[digest] = len(digest_to_id)
        ids[index] = digest_to_id[digest]
    n_unique = int(len(digest_to_id))
    return ids, {
        "key": "pynauty_certificate+coloured_incidence_sequence",
        "source": "tracks/ksvd/experiments/luyin16/identity_incremental_information_audit.py::_raw_zinc_certificate",
        "n_rows": int(stop),
        "n_unique_groups": n_unique,
        "duplicate_rows": int(stop - n_unique),
        "duplicate_fraction": float((stop - n_unique) / max(stop, 1)),
    }


def build_probe(
    root: Path,
    *,
    limit: int | None = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Extract ``X_base`` / ``X_typed`` / ``y`` / ``group_ids`` from official train."""
    started = time.perf_counter()
    dataset = load_official_train(root)
    if limit is not None:
        dataset = dataset[: int(limit)]
    if len(dataset) != OFFICIAL_TRAIN_SIZE and limit is None:
        raise RuntimeError(f"official train size {len(dataset)} != {OFFICIAL_TRAIN_SIZE}")

    topology, topology_meta = topology_matrix_train(root, dataset)
    build_started = time.perf_counter()
    base_rows: list[np.ndarray] = []
    typed_rows: list[np.ndarray] = []
    ys: list[float] = []
    per_graph_cycles: list[int] = []
    per_graph_views: list[int] = []
    length_totals = np.zeros(UNTYPED_RING_DIM, dtype=np.int64)
    acyclic = 0
    for index, data in enumerate(dataset):
        base, typed, census = graph_feature_row(data, topology[index])
        base_rows.append(base)
        typed_rows.append(typed)
        ys.append(float(data.y.view(-1)[0]))
        per_graph_cycles.append(int(census["n_cycles"]))
        per_graph_views.append(int(census["n_views"]))
        length_totals += census["length_counts"].astype(np.int64)
        if census["n_cycles"] == 0:
            acyclic += 1
        if index and index % progress_every == 0:
            print(
                f"[typed-cycle-probe] {index}/{len(dataset)} "
                f"elapsed={time.perf_counter() - build_started:.1f}s",
                flush=True,
            )
    feature_seconds = time.perf_counter() - build_started

    X_base = np.stack(base_rows).astype(np.float32)
    X_typed = np.stack(typed_rows).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    groups, group_meta = group_ids_train(dataset, limit=limit)

    counts = np.asarray(per_graph_cycles, dtype=np.float64)
    views = np.asarray(per_graph_views, dtype=np.float64)
    census = {
        "official_test_loaded": False,
        "n_graphs": int(len(y)),
        "feature_seconds": float(feature_seconds),
        "object_seconds_per_graph": float(feature_seconds / max(len(y), 1)),
        "total_cycles": int(counts.sum()),
        "total_dihedral_views": int(views.sum()),
        "cycles_per_graph": {
            "mean": float(counts.mean()),
            "p50": float(np.percentile(counts, 50)),
            "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)),
            "max": float(counts.max()),
        },
        "views_per_graph": {
            "mean": float(views.mean()),
            "p95": float(np.percentile(views, 95)),
            "p99": float(np.percentile(views, 99)),
            "max": float(views.max()),
        },
        "length_distribution": {
            str(MIN_CYCLE_LEN + offset): int(length_totals[offset])
            for offset in range(UNTYPED_RING_DIM)
        },
        "acyclic_graphs": int(acyclic),
        "acyclic_fraction": float(acyclic / max(len(y), 1)),
        "max_cycle_length": MAX_CYCLE_LEN,
        "enumerator": "typed_cycle_reference.chordless_cycles (early chord pruning, dedup)",
    }
    return {
        "X_base": X_base,
        "X_typed": X_typed,
        "y": y,
        "group_ids": groups,
        "group_meta": group_meta,
        "census": census,
        "topology_cache": topology_meta,
        "wall_seconds": float(time.perf_counter() - started),
    }


def split_fingerprint(root: Path) -> dict[str, Any]:
    from ksvd_research.runtime.fingerprints import zinc_fingerprints

    return zinc_fingerprints(
        root,
        expected_sizes={
            "train": OFFICIAL_TRAIN_SIZE,
            "val": OFFICIAL_VALID_SIZE,
            "test": OFFICIAL_TEST_SIZE,
        },
    )


def save_probe_npz(path: Path, payload: Mapping[str, Any]) -> str:
    """Write the probe NPZ (no object dtype) and return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X_base=np.asarray(payload["X_base"], dtype=np.float32),
        X_typed=np.asarray(payload["X_typed"], dtype=np.float32),
        y=np.asarray(payload["y"], dtype=np.float32),
        group_ids=np.asarray(payload["group_ids"], dtype=np.int64),
        split=np.asarray("train"),
        official_test_loaded=np.asarray(False),
        label_free_features=np.asarray(True),
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PROTOCOL_VERSION",
    "BASE_DIM",
    "TYPED_RING_DIM",
    "SEM110_DIM",
    "PHI65_DIM",
    "GLOBAL_DIM",
    "TOPOLOGY_DIM",
    "UNTYPED_RING_DIM",
    "EDGE_TYPE_DIM",
    "REFERENCE_DIR",
    "official_test_blocker",
    "to_reference_graph",
    "sem110_summary",
    "phi65_summary",
    "ring_census",
    "graph_feature_row",
    "load_official_train",
    "topology_matrix_train",
    "group_ids_train",
    "build_probe",
    "split_fingerprint",
    "save_probe_npz",
    "sha256_file",
]
