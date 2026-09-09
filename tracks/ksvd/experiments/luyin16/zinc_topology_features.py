"""Compact-v4: permutation-invariant global topology features for ZINC.

Purpose
-------
Compact-v2 is a fixed-radius (radius-2) patch representation.  It cannot
explicitly express graph-scale closure (how far one must walk until the graph
closes).  The long-cycle audit (``notes/zinc_long_cycle_audit.md``) showed
that a *monotone* function of permutation-invariant global cycle statistics
recovers up to ΔMAE +0.0117 on validation (isotonic oracle), while every
*linear* probe fails.  This module defines the raw, handcrafted, small
(<= 16D for v4-spectrum) global topology feature vector that the
``GlobalTopologyEncoder`` consumes; it is a **mechanism-validation feature
set**, not a new architecture and not a reconstruction of the ZINC target
formula.

Explicitly NOT here
-------------------
* no atom counts / edge counts / RDKit descriptors / target formula;
* no ``max(nx.cycle_basis)`` (proven order-dependent in the audit);
* no normalized cycle penalty, no target-definition cycle oracle.

Definitions (all graph invariants)
----------------------------------
* ``longest_simple_cycle`` L: length of the longest simple cycle; exact
  (the audit verified exact longest simple cycle is stable under node
  relabelling, 5x10 perms, 0 unstable).  Reused from the audit's verified
  cache ``results/zinc_long_cycle_audit/exact_longest_all.csv`` and
  cross-checked against an independent full-cycle enumeration.
* cycle spectrum: the exact number of simple cycles of length k, k = 3..10,
  and the number with length > 10.  All simple cycles are enumerated exactly
  (measured: ZINC-subset graphs have <= 38 simple cycles, max length 26, so
  enumeration is cheap and complete — no exponential blow-up; the counting
  definition is exact, not a basis/cut approximation).
* minimum cycle basis statistics (``nx.minimum_cycle_basis``): count, max
  length, mean length, total length.  The audit verified min-basis summary
  statistics are permutation-stable; an additional per-graph robustness check
  (3 random relabellings) is run at cache-build time and recorded.
* cycle rank: E - V + C (C = number of connected components).

Raw vectors (per mode, all values are raw, unstandardized)
----------------------------------------------------------
* ``longest``       : [L, cycle_rank]                                    (2D)
* ``spectrum``      : [L, n3..n10, n_gt10, mcb_count, mcb_max, mcb_mean,
                       mcb_total, cycle_rank]                            (15D)
* ``hinge``         : spectrum + [L, L^2, ReLU(L-3)..ReLU(L-10)]         (25D)
* ``capacity_control``: zeros(width)  (same shape/parameters as the mode it
                       controls against; carries no topology information)
* ``none``          : empty

The encoder input is standardized train-only (protocol: every fitted
transform is fit on official train in the selection phase; refit on
train+valid for the terminal phase).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from tracks.ksvd.experiments.luyin16.structural_context import _find_cycles

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIT_ROOT = REPO_ROOT / "tracks/ksvd/results/zinc_long_cycle_audit"
CACHE_ROOT = REPO_ROOT / "tracks/ksvd/results/zinc_topology_cache"
EXACT_LONGEST_AUDIT_CSV = AUDIT_ROOT / "exact_longest_all.csv"

FEATURE_VERSION = "v4-1"

SPECTRUM_MIN_LEN = 3
SPECTRUM_MAX_LEN = 10
HINGE_THRESHOLDS = tuple(range(3, 11))  # ReLU(L - 3) .. ReLU(L - 10)

# Column layout of the per-molecule cache DataFrame.
BASE_COLUMNS = (
    "n_nodes",
    "n_edges",
    "cycle_rank",
    "longest_simple_cycle",
    "audit_longest_simple_cycle",
    "audit_longest_status",
    "longest_matches_audit",
    "n_cycles_total",
    "n_cycle_len_3",
    "n_cycle_len_4",
    "n_cycle_len_5",
    "n_cycle_len_6",
    "n_cycle_len_7",
    "n_cycle_len_8",
    "n_cycle_len_9",
    "n_cycle_len_10",
    "n_cycle_len_gt10",
    "mcb_count",
    "mcb_max_length",
    "mcb_mean_length",
    "mcb_total_length",
    "mcb_perm_consistent",
)


def _connected_components(graph: Any) -> int:
    seen: set[int] = set()
    components = 0
    for node in graph.nodes:
        node = int(node)
        if node in seen:
            continue
        components += 1
        stack = [node]
        seen.add(node)
        while stack:
            current = stack.pop()
            for neighbor in graph.neighbors(current):
                neighbor = int(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return components


def cycle_rank(graph: Any) -> int:
    """E - V + C (cycle rank / cyclomatic number of the graph)."""
    n_nodes = int(len(graph.nodes))
    n_edges = int(sum(1 for _ in graph.edges()))
    return int(n_edges - n_nodes + _connected_components(graph))


def _to_networkx(graph: Any) -> nx.Graph:
    out = nx.Graph()
    out.add_nodes_from(int(node) for node in graph.nodes)
    out.add_edges_from(
        (int(u), int(v)) for u, v in graph.edges()
    )
    return out


def cycle_spectrum(graph: Any) -> dict[int, int]:
    """Exact count of simple cycles per length (permutation-invariant).

    Uses the tested exact min-node DFS enumerator from Compact-v3
    (``structural_context._find_cycles``) with an unbounded length cap
    (>= number of nodes, so every simple cycle is enumerated).  Node IDs
    appear only inside the DFS as an enumeration device; ``_canonical_cycle``
    deduplicates each cycle into an ID-free canonical form, so the resulting
    cycle *set* is a graph invariant, hence the per-length counts are
    permutation-invariant.
    """
    cycles = _find_cycles(graph, max(len(graph.nodes) + 1, SPECTRUM_MAX_LEN + 1))
    counts: dict[int, int] = {}
    for cycle in cycles:
        length = int(len(cycle))
        counts[length] = int(counts.get(length, 0) + 1)
    return counts


def mcb_statistics(graph: Any) -> dict[str, float]:
    """Summary statistics of ``nx.minimum_cycle_basis`` (permutation-invariant).

    The audit verified (20 perms x 20 molecules on the extremes, all stable)
    that the minimum-cycle-basis *summary statistics* are permutation-stable;
    the cache build additionally runs a 3-permutation robustness check per
    molecule (column ``mcb_perm_consistent``).
    """
    basis = nx.minimum_cycle_basis(_to_networkx(graph))
    lengths = [int(len(cycle)) for cycle in basis]
    if not lengths:
        return {
            "mcb_count": 0.0,
            "mcb_max_length": 0.0,
            "mcb_mean_length": 0.0,
            "mcb_total_length": 0.0,
            "mcb_perm_consistent": 1.0,
        }
    return {
        "mcb_count": float(len(lengths)),
        "mcb_max_length": float(max(lengths)),
        "mcb_mean_length": float(np.mean(lengths)),
        "mcb_total_length": float(sum(lengths)),
        "mcb_perm_consistent": 1.0,
    }


def compute_features(graph: Any, *, perm_check: bool = False) -> dict[str, float]:
    """Raw invariant topology features for one graph (dict with BASE_COLUMNS)."""
    spectrum = cycle_spectrum(graph)
    lengths = sorted(spectrum)
    longest = float(max(lengths, default=0))
    mcb = mcb_statistics(graph)
    row: dict[str, float] = {
        "n_nodes": float(len(graph.nodes)),
        "n_edges": float(sum(1 for _ in graph.edges())),
        "cycle_rank": float(cycle_rank(graph)),
        "longest_simple_cycle": longest,
        "audit_longest_simple_cycle": np.nan,
        "audit_longest_status": np.nan,
        "longest_matches_audit": np.nan,
        "n_cycles_total": float(sum(spectrum.values())),
        "n_cycle_len_gt10": float(sum(v for k, v in spectrum.items() if k > 10)),
    }
    for k in range(SPECTRUM_MIN_LEN, SPECTRUM_MAX_LEN + 1):
        row[f"n_cycle_len_{k}"] = float(spectrum.get(k, 0))
    row.update(mcb)
    if perm_check:
        row["mcb_perm_consistent"] = _mcb_perm_consistent(graph)
    return {key: float(row[key]) for key in BASE_COLUMNS}


def _relabel_graph(graph: Any, permutation: Sequence[int]) -> Any:
    """Return a copy of ``graph`` with node IDs permuted (labels 0..n-1)."""
    from tracks.ksvd.code.graph import from_edges

    mapping = {int(node): int(p) for node, p in zip(graph.nodes, permutation)}
    edges = [(mapping[int(u)], mapping[int(v)]) for u, v in graph.edges()]
    return from_edges(len(graph.nodes), edges)


def _mcb_perm_consistent(graph: Any, n_perms: int = 3) -> float:
    """1.0 iff MCB summary stats are identical under n random relabellings."""
    rng = np.random.default_rng(0)
    reference = mcb_statistics(graph)
    width = len(graph.nodes)
    for _ in range(int(n_perms)):
        permutation = rng.permutation(width).tolist()
        relabelled = _relabel_graph(graph, permutation)
        other = mcb_statistics(relabelled)
        for key in ("mcb_count", "mcb_max_length", "mcb_mean_length", "mcb_total_length"):
            if abs(reference[key] - other[key]) > 1e-9:
                if key == "mcb_mean_length":
                    # mean can differ by float rounding when count==0; keep
                    # the stricter check on the integer-valued stats.
                    if reference["mcb_count"] == 0 and other["mcb_count"] == 0:
                        continue
                return 0.0
    return 1.0


def _audit_longest_table() -> dict[tuple[str, int], tuple[float, float]]:
    if not EXACT_LONGEST_AUDIT_CSV.exists():
        return {}
    table: dict[tuple[str, int], tuple[float, float]] = {}
    with open(EXACT_LONGEST_AUDIT_CSV, newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle):
            split = str(entry["split"])
            index = int(entry["idx"])
            table[(split, index)] = (
                float(entry["longest_simple_cycle_length"]),
                float(entry["longest_simple_cycle_status"]),
            )
    return table


def build_cache(split: str, dataset: Sequence[Any], *, force: bool = False) -> pd.DataFrame:
    """Compute (or load) the per-molecule topology cache DataFrame.

    Reuses the audit's verified ``exact_longest_all.csv`` for L_max and
    cross-checks the independent enumeration against it.
    """
    if not force:
        cached = load_cache(split)
        if cached is not None:
            return cached
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph

    audit_table = _audit_longest_table()
    split_key = "valid" if split == "val" else ("test" if split == "test" else split)
    started = time.perf_counter()
    rows: list[dict[str, float]] = []
    for index, data in enumerate(dataset):
        graph, _, _ = _data_to_graph(data)
        row = compute_features(graph, perm_check=False)
        audit = audit_table.get((split_key, index))
        if audit is not None:
            row["audit_longest_simple_cycle"] = audit[0]
            row["audit_longest_status"] = audit[1]
            row["longest_matches_audit"] = 1.0 if row["longest_simple_cycle"] == audit[0] else 0.0
        row["mcb_perm_consistent"] = _mcb_perm_consistent(graph)
        row["_split"] = split_key
        row["_idx"] = float(index)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.set_index(["_split", "_idx"])
    frame.index.names = ["split", "idx"]
    _write_cache(split_key, frame, seconds=time.perf_counter() - started)
    return _reread(split_key, frame)


def _reread(split: str, frame: pd.DataFrame | None = None) -> pd.DataFrame | None:
    path = _cache_path(split)
    if path is None or not path.exists():
        return frame
    return pd.read_csv(path, index_col=[0, 1])


def _cache_path(split: str) -> Path | None:
    if split not in ("train", "valid", "test"):
        return None
    return CACHE_ROOT / f"{split}_topology_features.csv"


def _write_cache(split: str, frame: pd.DataFrame, *, seconds: float) -> None:
    path = _cache_path(split)
    if path is None:
        return
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = frame.reset_index()
    payload.to_csv(path, index=False)
    meta_path = CACHE_ROOT / "meta.json"
    meta: dict[str, Any] = {"version": FEATURE_VERSION}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {"version": FEATURE_VERSION}
    stats = meta.setdefault("split_stats", {})
    stats[str(split)] = {
        "n_molecules": int(len(frame)),
        "seconds": float(seconds),
        "mean_seconds_per_graph": float(seconds / max(len(frame), 1)),
        "cache_bytes": int(path.stat().st_size),
        "matches_audit": int(frame["longest_matches_audit"].sum()) if "longest_matches_audit" in frame else None,
        "mcb_perm_consistent": (
            float(frame["mcb_perm_consistent"].mean()) if "mcb_perm_consistent" in frame else None
        ),
        "mcb_inconsistent_n": (
            int((frame["mcb_perm_consistent"] < 1.0).sum()) if "mcb_perm_consistent" in frame else None
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_cache(split: str) -> pd.DataFrame | None:
    path = _cache_path(split)
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path, index_col=[0, 1])
    if "longest_simple_cycle" not in frame.columns:
        return None
    return frame


def raw_width(mode: str, *, input_width_hint: int | None = None) -> int:
    mode = str(mode)
    if mode == "none":
        return 0
    if mode == "longest":
        return 2
    if mode == "spectrum":
        return 15
    if mode == "hinge":
        return 25
    if mode == "capacity_control":
        if input_width_hint is None or int(input_width_hint) < 1:
            raise ValueError(
                "capacity_control requires an explicit positive topology_input_width"
            )
        return int(input_width_hint)
    raise ValueError(f"unknown topology_mode={mode!r}")


def raw_vector(
    frame_row: Mapping[str, float], mode: str, *, input_width: int | None = None
) -> np.ndarray:
    """Raw (unstandardized) topology feature vector for one molecule."""
    mode = str(mode)
    width = raw_width(mode, input_width_hint=input_width)
    if mode == "none":
        return np.zeros((0,), dtype=np.float32)
    longest = float(frame_row["longest_simple_cycle"])
    rank = float(frame_row["cycle_rank"])
    if mode == "longest":
        return np.asarray([longest, rank], dtype=np.float32)
    if mode == "capacity_control":
        return np.zeros((width,), dtype=np.float32)
    spectrum = np.asarray(
        [float(frame_row[f"n_cycle_len_{k}"]) for k in range(3, 11)] + [float(frame_row["n_cycle_len_gt10"])],
        dtype=np.float32,
    )
    mcb = np.asarray(
        [
            float(frame_row["mcb_count"]),
            float(frame_row["mcb_max_length"]),
            float(frame_row["mcb_mean_length"]),
            float(frame_row["mcb_total_length"]),
        ],
        dtype=np.float32,
    )
    base = np.concatenate([
        np.asarray([longest], dtype=np.float32),
        spectrum,
        mcb,
        np.asarray([rank], dtype=np.float32),
    ])
    if mode == "spectrum":
        return base
    if mode == "hinge":
        hinge = np.asarray(
            [float(max(longest - t, 0.0)) for t in HINGE_THRESHOLDS],
            dtype=np.float32,
        )
        return np.concatenate([
            base,
            np.asarray([longest, longest * longest], dtype=np.float32),
            hinge,
        ])
    raise ValueError(f"unknown topology_mode={mode!r}")


def feature_names(mode: str, *, input_width: int | None = None) -> list[str]:
    mode = str(mode)
    if mode == "none":
        return []
    if mode == "longest":
        return ["longest_simple_cycle", "cycle_rank"]
    spectrum_names = [f"n_cycle_len_{k}" for k in range(3, 11)] + ["n_cycle_len_gt10"]
    mcb_names = ["mcb_count", "mcb_max_length", "mcb_mean_length", "mcb_total_length"]
    base = ["longest_simple_cycle"] + spectrum_names + mcb_names + ["cycle_rank"]
    if mode == "spectrum":
        return base
    if mode == "hinge":
        return base + ["longest", "longest_squared"] + [
            f"hinge_{t}" for t in HINGE_THRESHOLDS
        ]
    if mode == "capacity_control":
        return [f"zero_{i}" for i in range(raw_width(mode, input_width_hint=input_width))]
    raise ValueError(f"unknown topology_mode={mode!r}")


def matrices_for_split(
    split: str,
    dataset: Sequence[Any],
    mode: str,
    *,
    force: bool = False,
    input_width: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Return (raw feature matrix (n, d), cache frame, metadata)."""
    frame = build_cache(split, dataset, force=force)
    width = raw_width(mode, input_width_hint=input_width)
    vectors = [raw_vector(frame.iloc[index], mode, input_width=width) for index in range(len(frame))]
    matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    metadata = {
        "mode": str(mode),
        "input_width": int(matrix.shape[1]),
        "n_graphs": int(len(frame)),
        "cache": _cache_meta(),
    }
    return matrix, frame, metadata


def _cache_meta() -> dict[str, Any]:
    meta_path = CACHE_ROOT / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
