"""Fixed base query library ``B`` and the query-feature matrix assembler.

``B`` is built from **Train structure only** (no labels at all):

* every node primitive over the Train-observed atom/bond vocabularies;
* every ``NBR(ATOM_EQ(0,a), b, m)`` with ``m in {1,2}``;
* every ``AND(ATOM_EQ(0,a), NBR(ATOM_EQ(0,a2), b, m))`` with ``m in {1,2}``.

Filtering: keep a query only if it matches at least 8 Train molecules and at
least one of its four graph-level feature columns is non-constant on Train.
Then drop Train-equivalent duplicates (identical root truth vectors on Train;
keep the shorter AST, then the lexicographically smaller canonical string) and
cap at 512 by keeping every surviving node primitive first and then filling by
descending Train molecule coverage with canonical-string tie-breaks.  Every
dropped count is recorded (``filtered_support``, ``filtered_constant``,
``dropped_train_equivalents``, ``excluded_by_cap``).

The graph feature of a query is ``[c, p, e, log1p(c)]``: matched roots, root
density, existence indicator and a log count.  Everything is dense ``float32``
with genuine zeros (never a sparse matrix, so "0" is never "missing").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import sq_ast as A
from sq_ast import PackedGraph

BASE_BANK_CAP = 512
MIN_TRAIN_MOLECULES = 8
BASE_COLUMN_NAMES: tuple[str, ...] = (
    "n_atoms",
    "n_bonds",
    "n_components",
    "cycle_rank",
    "log1p_n_atoms",
)


# ---------------------------------------------------------------------------
# root-truth mask cache (bounded)
# ---------------------------------------------------------------------------


DEFAULT_MASK_CACHE_MB = 512.0


def mask_cache_limit_from_env() -> int:
    """Byte budget for the root-truth LRU (``SQ_MASK_CACHE_MB``, default 512)."""
    raw = os.environ.get("SQ_MASK_CACHE_MB", str(DEFAULT_MASK_CACHE_MB)).strip()
    try:
        megabytes = float(raw)
    except ValueError:
        megabytes = DEFAULT_MASK_CACHE_MB
    return max(1, int(megabytes * 1024 * 1024))


@dataclass
class MaskCache:
    """Small LRU over root-truth masks, bounded in bytes (default 512 MiB)."""

    limit_bytes: int = field(default_factory=mask_cache_limit_from_env)
    _store: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    _order: list[tuple[str, str]] = field(default_factory=list)
    _bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def get(self, split: str, ast: tuple) -> np.ndarray | None:
        key = (split, A.canon(ast))
        mask = self._store.get(key)
        if mask is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            self._order.remove(key)
        except ValueError:  # pragma: no cover
            pass
        self._order.append(key)
        return mask

    def put(self, split: str, ast: tuple, mask: np.ndarray) -> None:
        key = (split, A.canon(ast))
        if key in self._store:
            return
        size = int(mask.nbytes)
        if size > self.limit_bytes:
            return
        while self._bytes + size > self.limit_bytes and self._order:
            old = self._order.pop(0)
            self._bytes -= int(self._store.pop(old).nbytes)
            self.evictions += 1
        self._store[key] = mask
        self._order.append(key)
        self._bytes += size

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._store),
            "bytes": self._bytes,
            "limit_bytes": self.limit_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


def root_mask(packed: PackedGraph, ast: tuple, cache: MaskCache | None = None) -> np.ndarray:
    if cache is not None:
        cached = cache.get(packed.split, ast)
        if cached is not None:
            return cached
    mask = A.evaluate(packed, ast)
    if cache is not None:
        cache.put(packed.split, ast, mask)
    return mask


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def node_primitives(vocabulary: Mapping[str, Any]) -> list[tuple]:
    """All node primitives with Train-observed parameter values, canonical order."""
    out: list[tuple] = [A.TRUE()]
    for j in range(len(vocabulary["atom_values"])):
        for value in vocabulary["atom_values"][j]:
            out.append(A.ATOM_EQ(j, value))
    for k in range(0, 6):
        out.append(A.DEG_EQ(k))
    out.append(A.DEG_GE(6))
    out.append(A.IN_CYCLE())
    for k in range(3, 9):
        out.append(A.MIN_RING_EQ(k))
    out.append(A.MIN_RING_GE(9))
    out.append(A.ACYCLIC())
    return sorted(set(out), key=A.canon)


def bond_predicates(vocabulary: Mapping[str, Any]) -> list[tuple]:
    """ANY plus every BOND_EQ(j,k) with a Train-observed value."""
    out: list[tuple] = [A.ANY_BOND_PRED()]
    for j in range(len(vocabulary["bond_values"])):
        for value in vocabulary["bond_values"][j]:
            out.append(A.BOND_EQ(j, value))
    return sorted(set(out), key=A._bond_name)


def candidate_base_bank(vocabulary: Mapping[str, Any]) -> list[tuple]:
    """The untrimmed B candidate pool (Train vocabulary, no labels)."""
    bonds = bond_predicates(vocabulary)
    elements = vocabulary["atom_values"][0]
    candidates: list[tuple] = list(node_primitives(vocabulary))
    for a in elements:
        for bond in bonds:
            for m in (1, 2):
                candidates.append(A.NBR(A.ATOM_EQ(0, a), bond, m))
    for a in elements:
        for a2 in elements:
            for bond in bonds:
                for m in (1, 2):
                    candidates.append(A.AND(A.ATOM_EQ(0, a), A.NBR(A.ATOM_EQ(0, a2), bond, m)))
    # dedup exact ASTs while keeping canonical order
    unique: dict[str, tuple] = {}
    for ast in candidates:
        unique.setdefault(A.canon(ast), ast)
    return [unique[key] for key in sorted(unique)]


# ---------------------------------------------------------------------------
# query records
# ---------------------------------------------------------------------------


@dataclass
class QueryRecord:
    ast: tuple
    label: str
    source: str           # "B" | "discovered" | "random"
    outer_round: int
    train_support: int    # number of Train molecules with >=1 match
    train_gain: float | None = None
    node_mask_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ast": A.canon(self.ast),
            "source": self.source,
            "outer_round": int(self.outer_round),
            "train_support": int(self.train_support),
            "train_gain": None if self.train_gain is None else float(self.train_gain),
            "nodes": A.ast_nodes(self.ast),
            "hops": A.ast_hops(self.ast),
            "cost": A.ast_cost(self.ast),
        }


def validate_query_on_train(
    packed: PackedGraph,
    ast: tuple,
    *,
    min_molecules: int = MIN_TRAIN_MOLECULES,
    cache: MaskCache | None = None,
) -> tuple[bool, str, np.ndarray | None]:
    """Support / constancy check used by B, A-selection and R construction."""
    if not A.validates(ast):
        return False, "invalid_ast", None
    mask = root_mask(packed, ast, cache)
    counts = packed.graph_counts(mask)
    support = int((counts > 0).sum())
    if support < min_molecules:
        return False, "support", mask
    features = A.query_features(packed, mask)
    if np.all(features == features[0:1, :]):
        return False, "constant", mask
    return True, "ok", mask


# ---------------------------------------------------------------------------
# B construction
# ---------------------------------------------------------------------------


def build_base_bank(
    train_packed: PackedGraph,
    vocabulary: Mapping[str, Any],
    *,
    cap: int = BASE_BANK_CAP,
    min_molecules: int = MIN_TRAIN_MOLECULES,
    cache: MaskCache | None = None,
) -> tuple[list[QueryRecord], dict[str, Any]]:
    candidates = candidate_base_bank(vocabulary)
    stats: dict[str, Any] = {
        "n_candidates": len(candidates),
        "min_train_molecules": int(min_molecules),
        "cap": int(cap),
        "filtered_support": 0,
        "filtered_constant": 0,
        "filtered_invalid": 0,
        "dropped_train_equivalents": 0,
        "excluded_by_cap": 0,
    }
    survivors: list[tuple[tuple, np.ndarray, int]] = []
    for ast in candidates:
        ok, reason, mask = validate_query_on_train(
            train_packed, ast, min_molecules=min_molecules, cache=cache
        )
        if not ok:
            if reason == "support":
                stats["filtered_support"] += 1
            elif reason == "constant":
                stats["filtered_constant"] += 1
            else:
                stats["filtered_invalid"] += 1
            continue
        counts = train_packed.graph_counts(mask)
        survivors.append((ast, mask, int((counts > 0).sum())))

    # Train-equivalence dedup: identical root truth vectors on Train only.
    kept: dict[str, tuple[tuple, np.ndarray, int]] = {}
    for ast, mask, support in survivors:
        key = mask.tobytes()
        current = kept.get(key)
        if current is None:
            kept[key] = (ast, mask, support)
            continue
        better = _prefer(ast, current[0])
        if better is current[0]:  # pragma: no cover - defensive
            continue
        stats["dropped_train_equivalents"] += 1
        kept[key] = (ast, mask, support)
    deduped = list(kept.values())

    node_like = [rec for rec in deduped if A.is_node_primitive(rec[0])]
    others = [rec for rec in deduped if not A.is_node_primitive(rec[0])]
    others.sort(key=lambda rec: (-rec[2], A.canon(rec[0])))
    ordered = node_like + others
    if len(ordered) > cap:
        stats["excluded_by_cap"] = len(ordered) - cap
        ordered = ordered[:cap]

    records = [
        QueryRecord(
            ast=ast,
            label=f"B{idx:03d}",
            source="B",
            outer_round=0,
            train_support=int(support),
            train_gain=None,
        )
        for idx, (ast, _mask, support) in enumerate(ordered)
    ]
    stats.update(
        {
            "n_survivors_before_dedup": len(survivors),
            "n_after_dedup": len(deduped),
            "n_node_primitives_kept": len(node_like),
            "bank_size": len(records),
            "n_bond_predicates": len(bond_predicates(vocabulary)),
            "n_node_primitives": len(node_primitives(vocabulary)),
        }
    )
    return records, stats


def _prefer(new_ast: tuple, old_ast: tuple) -> tuple:
    new_key = (A.ast_nodes(new_ast), A.canon(new_ast))
    old_key = (A.ast_nodes(old_ast), A.canon(old_ast))
    return new_ast if new_key < old_key else old_ast


# ---------------------------------------------------------------------------
# design matrices
# ---------------------------------------------------------------------------


def query_feature_names(records: Sequence[QueryRecord]) -> list[str]:
    names: list[str] = []
    for record in records:
        for transform in A.FEATURE_TRANSFORMS:
            names.append(f"{record.label}::{transform}")
    return names


def design_matrix(
    packed: PackedGraph,
    records: Sequence[QueryRecord],
    *,
    cache: MaskCache | None = None,
    include_base: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Dense ``float32`` matrix: 5 shared base columns + 4 columns per query."""
    blocks: list[np.ndarray] = []
    names: list[str] = []
    if include_base:
        base = packed.base_columns()
        blocks.append(np.stack([base[name] for name in BASE_COLUMN_NAMES], axis=1))
        names.extend(f"base::{name}" for name in BASE_COLUMN_NAMES)
    for record in records:
        mask = root_mask(packed, record.ast, cache)
        blocks.append(A.query_features(packed, mask).astype(np.float64))
        for transform in A.FEATURE_TRANSFORMS:
            names.append(f"{record.label}::{transform}")
    if not blocks:
        return np.zeros((packed.n_graphs, 0), dtype=np.float32), names
    matrix = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("query feature matrix contains non-finite values")
    return matrix, names


def query_columns(
    packed: PackedGraph, ast: tuple, *, cache: MaskCache | None = None
) -> np.ndarray:
    """The four graph-level columns ``[c, p, e, log1p(c)]`` for one query."""
    return A.query_features(packed, root_mask(packed, ast, cache))
