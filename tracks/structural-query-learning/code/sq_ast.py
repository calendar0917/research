"""The discrete structural query language (AST + packed-graph executor).

A query is a Boolean predicate on nodes.  It is represented as a hashable
Python tuple, so equality/dedup/canonical ordering are exact and cheap.  The
``PackedGraph`` below is the *only* execution context: every graph of a split
is concatenated with an offset table, so one query evaluation is a handful of
vectorised numpy operations.  Queries never see labels and never cross molecule
boundaries: every operator is either per-node, per-physical-bond, or a
per-graph segment sum.

Language (frozen by the round brief):

    primitives : TRUE, ATOM_EQ(j,k), DEG_EQ(k), DEG_GE(k), IN_CYCLE,
                 MIN_RING_EQ(k), MIN_RING_GE(k), ACYCLIC
    bonds      : ANY, BOND_EQ(j,k)
    operators  : AND(P,Q[,R]), NBR(P,b,m>=1), BRANCH(P,Q,b1,b2)

Semantics pinned here (see tests):

* ``NBR(P,b,m)`` is a *counting* modality over the physical neighbours of the
  root: at least ``m`` neighbours ``u`` with ``b(v,u)`` and ``P(u)``.  Recursion
  is logical existence/counting, i.e. the same neighbour may serve several
  levels; no induction, no simple-path assumption.
* ``BRANCH(P,Q,b1,b2)`` requires TWO DISTINCT neighbours: a witness is a pair
  ``u != w`` with ``b1(v,u) & P(u)`` and ``b2(v,w) & Q(w)``.  Implemented
  exactly through neighbour counts and the joint overlap count:

      ok = (cp > 0) & (cq > 0) & (cp + cq - cpq >= 2)

  where ``cpq`` counts neighbours satisfying both (b1 & P) and (b2 & Q).  The
  overlap must respect the *conjunction* of the two bond predicates, so a
  generic ``ANY`` and a concrete ``BOND_EQ`` intersect correctly.
* A physical undirected bond is stored once.  Neighbourhoods are expanded to
  both directions inside the executor; degree is the number of physical bonds.
* Ring information only enters through the explicit ring primitives.

Limits: at most 3 nested NBR/BRANCH hops and at most 9 AST nodes.  Leaves and
operators cost 1 node each; bond predicates and ``m`` are operator parameters
and cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MAX_NODES = 9
MAX_HOPS = 3

ANY_BOND: tuple[str, ...] = ("ANY",)

NODE_KINDS = frozenset(
    {
        "TRUE",
        "ATOM_EQ",
        "DEG_EQ",
        "DEG_GE",
        "IN_CYCLE",
        "MIN_RING_EQ",
        "MIN_RING_GE",
        "ACYCLIC",
    }
)
OP_KINDS = frozenset({"AND", "NBR", "BRANCH"})


# ---------------------------------------------------------------------------
# AST construction
# ---------------------------------------------------------------------------


def TRUE() -> tuple:
    return ("TRUE",)


def ATOM_EQ(j: int, k: int) -> tuple:
    return ("ATOM_EQ", int(j), int(k))


def DEG_EQ(k: int) -> tuple:
    return ("DEG_EQ", int(k))


def DEG_GE(k: int) -> tuple:
    return ("DEG_GE", int(k))


def IN_CYCLE() -> tuple:
    return ("IN_CYCLE",)


def MIN_RING_EQ(k: int) -> tuple:
    return ("MIN_RING_EQ", int(k))


def MIN_RING_GE(k: int) -> tuple:
    return ("MIN_RING_GE", int(k))


def ACYCLIC() -> tuple:
    return ("ACYCLIC",)


def ANY_BOND_PRED() -> tuple:
    return ("ANY",)


def BOND_EQ(j: int, k: int) -> tuple:
    return ("BOND_EQ", int(j), int(k))


def AND(*children: tuple) -> tuple:
    """N-ary conjunction, canonicalised (sorted + deduplicated)."""
    flat: list[tuple] = []
    for child in children:
        if child[0] == "AND":
            flat.extend(child[1:])
        else:
            flat.append(child)
    unique = sorted(set(flat), key=canon)
    if len(unique) == 1:
        return unique[0]
    return ("AND", *unique)


def NBR(parent: tuple, bond: tuple, m: int) -> tuple:
    return ("NBR", parent, bond, int(m))


def BRANCH(parent_p: tuple, parent_q: tuple, b1: tuple, b2: tuple) -> tuple:
    return ("BRANCH", parent_p, parent_q, b1, b2)


def is_node_primitive(ast: tuple) -> bool:
    return ast[0] in NODE_KINDS


def is_bond_pred(bond: tuple) -> bool:
    return bond[0] in ("ANY", "BOND_EQ")


def validates(ast: tuple) -> bool:
    """Structural validity of an AST (kinds, arity, node/hop budget)."""
    try:
        _check_shapes(ast)
    except (TypeError, IndexError, ValueError):
        return False
    return ast_nodes(ast) <= MAX_NODES and ast_hops(ast) <= MAX_HOPS


def _check_shapes(ast) -> None:
    if not isinstance(ast, tuple) or not ast:
        raise TypeError("AST node must be a non-empty tuple")
    kind = ast[0]
    if kind in NODE_KINDS:
        if kind == "ATOM_EQ":
            if len(ast) != 3 or not isinstance(ast[1], int) or not isinstance(ast[2], int):
                raise ValueError("ATOM_EQ expects two integer arguments (column, value)")
        elif kind in ("DEG_EQ", "DEG_GE", "MIN_RING_EQ", "MIN_RING_GE"):
            if len(ast) != 2 or not isinstance(ast[1], int):
                raise ValueError(f"{kind} expects one integer argument")
        elif len(ast) != 1:
            raise ValueError(f"{kind} takes no arguments")
        return
    if kind == "AND":
        if len(ast) not in (3, 4):
            raise ValueError("AND takes 2 or 3 children")
        for child in ast[1:]:
            _check_shapes(child)
        return
    if kind == "NBR":
        if len(ast) != 4:
            raise ValueError("NBR expects (parent, bond, m)")
        parent, bond, m = ast[1], ast[2], ast[3]
        _check_shapes(parent)
        if not is_bond_pred(bond):
            raise ValueError("NBR bond predicate must be ANY or BOND_EQ")
        if not isinstance(m, int) or m < 1:
            raise ValueError("NBR m must be an integer >= 1")
        return
    if kind == "BRANCH":
        if len(ast) != 5:
            raise ValueError("BRANCH expects (P, Q, b1, b2)")
        _check_shapes(ast[1])
        _check_shapes(ast[2])
        if not is_bond_pred(ast[3]) or not is_bond_pred(ast[4]):
            raise ValueError("BRANCH bond predicates must be ANY or BOND_EQ")
        return
    raise ValueError(f"unknown AST kind {kind!r}")


def ast_nodes(ast: tuple) -> int:
    kind = ast[0]
    if kind in NODE_KINDS:
        return 1
    if kind == "AND":
        return 1 + sum(ast_nodes(child) for child in ast[1:])
    if kind == "NBR":
        return 1 + ast_nodes(ast[1])
    if kind == "BRANCH":
        return 1 + ast_nodes(ast[1]) + ast_nodes(ast[2])
    raise ValueError(f"unknown AST kind {kind!r}")


def ast_hops(ast: tuple) -> int:
    kind = ast[0]
    if kind in NODE_KINDS:
        return 0
    if kind == "AND":
        return max(ast_hops(child) for child in ast[1:])
    if kind == "NBR":
        return 1 + ast_hops(ast[1])
    if kind == "BRANCH":
        return 1 + max(ast_hops(ast[1]), ast_hops(ast[2]))
    raise ValueError(f"unknown AST kind {kind!r}")


def ast_cost(ast: tuple) -> int:
    """Cost used in the complexity penalty: ``1 + 0.05*(nodes-1)``."""
    return ast_nodes(ast)


def _bond_name(bond: tuple) -> str:
    if bond[0] == "ANY":
        return "ANY"
    return f"BOND_EQ({bond[1]},{bond[2]})"


def bond_from_json(node: Mapping[str, Any]) -> tuple:
    """Inverse of ``sq_explain._bond_to_json`` (exported artefacts stay executable)."""
    op = str(node["op"])
    if op == "ANY":
        return ANY_BOND_PRED()
    if op == "BOND_EQ":
        return BOND_EQ(int(node["bond_column"]), int(node["value"]))
    raise ValueError(f"unknown bond predicate in export: {op!r}")


def from_json(node: Mapping[str, Any]) -> tuple:
    """Rebuild an AST tuple from the exported ``ast_tree`` JSON.

    Every exported query is re-parsed with this function before a run is
    allowed to claim "executable export": the JSON is the artefact, not a
    pretty-printed convenience.
    """
    op = str(node["op"])
    if op == "ATOM_EQ":
        return ATOM_EQ(int(node["atom_column"]), int(node["value"]))
    if op in ("DEG_EQ", "DEG_GE", "MIN_RING_EQ", "MIN_RING_GE"):
        return (op, int(node["value"]))
    if op in NODE_KINDS:
        return (op,)
    if op == "AND":
        return AND(*[from_json(child) for child in node["children"]])
    if op == "NBR":
        return NBR(from_json(node["parent"]), bond_from_json(node["bond"]), int(node["m"]))
    if op == "BRANCH":
        return BRANCH(
            from_json(node["parent_p"]),
            from_json(node["parent_q"]),
            bond_from_json(node["bond_p"]),
            bond_from_json(node["bond_q"]),
        )
    raise ValueError(f"unknown AST operator in export: {op!r}")


def canon(ast: tuple) -> str:
    kind = ast[0]
    if kind == "TRUE":
        return "TRUE"
    if kind == "ATOM_EQ":
        return f"ATOM_EQ({ast[1]},{ast[2]})"
    if kind in ("DEG_EQ", "DEG_GE", "MIN_RING_EQ", "MIN_RING_GE"):
        return f"{kind}({ast[1]})"
    if kind in ("IN_CYCLE", "ACYCLIC"):
        return kind
    if kind == "AND":
        return "AND(" + ",".join(canon(child) for child in ast[1:]) + ")"
    if kind == "NBR":
        return f"NBR({canon(ast[1])},{_bond_name(ast[2])},{ast[3]})"
    if kind == "BRANCH":
        return (
            f"BRANCH({canon(ast[1])},{canon(ast[2])},"
            f"{_bond_name(ast[3])},{_bond_name(ast[4])})"
        )
    raise ValueError(f"unknown AST kind {kind!r}")


def leaf_params(ast: tuple) -> list[tuple[str, tuple]]:
    """All primitive leaves in canonical order (for the R random remapping)."""
    out: list[tuple[str, tuple]] = []

    def walk(node: tuple) -> None:
        kind = node[0]
        if kind in NODE_KINDS:
            out.append(("node", node))
            return
        if kind == "AND":
            for child in node[1:]:
                walk(child)
            return
        if kind == "NBR":
            walk(node[1])
            if node[2][0] == "BOND_EQ":
                out.append(("bond", node[2]))
            return
        if kind == "BRANCH":
            walk(node[1])
            walk(node[2])
            for bond in (node[3], node[4]):
                if bond[0] == "BOND_EQ":
                    out.append(("bond", bond))
            return

    walk(ast)
    return out


def atom_columns(ast: tuple) -> set[int]:
    """Atom feature columns referenced by ATOM_EQ leaves."""
    return {node[2] for kind, node in leaf_params(ast) if kind == "node" and node[0] == "ATOM_EQ"}


def bond_columns(ast: tuple) -> set[int]:
    return {bond[1] for kind, bond in leaf_params(ast) if kind == "bond"}


def iter_operators(ast: tuple) -> Iterable[tuple]:
    yield ast
    kind = ast[0]
    if kind == "AND":
        for child in ast[1:]:
            yield from iter_operators(child)
    elif kind == "NBR":
        yield from iter_operators(ast[1])
    elif kind == "BRANCH":
        yield from iter_operators(ast[1])
        yield from iter_operators(ast[2])


def operator_shape(ast: tuple) -> dict[str, Any]:
    """Operator census (AND/NBR/BRANCH, hop and node budget) for R matching."""
    counts = {"AND": 0, "NBR": 0, "BRANCH": 0}
    ms: list[int] = []
    for node in iter_operators(ast):
        if node[0] in counts:
            counts[node[0]] += 1
        if node[0] == "NBR":
            ms.append(int(node[3]))
    return {
        "counts": counts,
        "m_values": sorted(ms),
        "nodes": ast_nodes(ast),
        "hops": ast_hops(ast),
    }


# ---------------------------------------------------------------------------
# packed graphs
# ---------------------------------------------------------------------------


@dataclass
class PackedGraph:
    """All molecules of one split concatenated, with physical bonds stored once."""

    dataset: str
    split: str
    n_graphs: int
    n_nodes: int
    node_graph: np.ndarray          # (n_nodes,) int32 -> graph index
    atom_feat: np.ndarray           # (n_nodes, 9) int16
    degree: np.ndarray              # (n_nodes,) int16
    min_ring: np.ndarray            # (n_nodes,) int16; 0 acyclic, 3..8, 9 => >=9
    in_cycle: np.ndarray            # (n_nodes,) bool
    n_bonds: int
    bond_u: np.ndarray              # (n_bonds,) int32 with u < v
    bond_v: np.ndarray
    bond_feat: np.ndarray           # (n_bonds, 3) int16
    graph_n_atoms: np.ndarray       # (n_graphs,) int32
    graph_n_bonds: np.ndarray
    graph_n_components: np.ndarray
    graph_cycle_rank: np.ndarray
    row_ids: np.ndarray             # (n_graphs,) int64 official row id
    labels: np.ndarray              # (n_graphs,) float32 raw units
    smiles: list[str]

    # -- derived helpers ---------------------------------------------------
    @property
    def n_atom_columns(self) -> int:
        return int(self.atom_feat.shape[1])

    @property
    def n_bond_columns(self) -> int:
        return int(self.bond_feat.shape[1])

    def base_columns(self) -> dict[str, np.ndarray]:
        n_atoms = self.graph_n_atoms.astype(np.float64)
        return {
            "n_atoms": n_atoms,
            "n_bonds": self.graph_n_bonds.astype(np.float64),
            "n_components": self.graph_n_components.astype(np.float64),
            "cycle_rank": self.graph_cycle_rank.astype(np.float64),
            "log1p_n_atoms": np.log1p(n_atoms),
        }

    def graph_counts(self, node_mask: np.ndarray) -> np.ndarray:
        """Matched-root count per graph (exact integer counts)."""
        return np.bincount(
            self.node_graph,
            weights=np.asarray(node_mask, dtype=np.float32),
            minlength=self.n_graphs,
        ).astype(np.int64)

    def neighbors(self, node: int) -> list[tuple[int, int]]:
        """(neighbour, bond_record_index) for one node, both directions."""
        out: list[tuple[int, int]] = []
        for idx in range(self.n_bonds):
            if self.bond_u[idx] == node:
                out.append((int(self.bond_v[idx]), idx))
            elif self.bond_v[idx] == node:
                out.append((int(self.bond_u[idx]), idx))
        return out

    def graph_nodes(self, graph: int) -> np.ndarray:
        return np.nonzero(self.node_graph == graph)[0]


def bond_ok(packed: PackedGraph, bond: tuple) -> np.ndarray:
    """Boolean mask over physical bonds (True for ANY)."""
    if bond[0] == "ANY":
        return np.ones(packed.n_bonds, dtype=bool)
    return packed.bond_feat[:, bond[1]] == bond[2]


class QueryError(RuntimeError):
    pass


def evaluate(packed: PackedGraph, ast: tuple) -> np.ndarray:
    """Evaluate a query on a packed split; returns the per-node truth mask."""
    if not validates(ast):
        raise QueryError(f"invalid AST (nodes>{MAX_NODES} or hops>{MAX_HOPS}): {canon(ast)}")
    return _eval(packed, ast)


def _eval(packed: PackedGraph, ast: tuple) -> np.ndarray:
    kind = ast[0]
    if kind == "TRUE":
        return np.ones(packed.n_nodes, dtype=bool)
    if kind == "ATOM_EQ":
        return packed.atom_feat[:, ast[1]] == ast[2]
    if kind == "DEG_EQ":
        return packed.degree == ast[1]
    if kind == "DEG_GE":
        return packed.degree >= ast[1]
    if kind == "IN_CYCLE":
        return packed.in_cycle
    if kind == "MIN_RING_EQ":
        return packed.min_ring == ast[1]
    if kind == "MIN_RING_GE":
        return packed.min_ring >= ast[1]
    if kind == "ACYCLIC":
        return packed.min_ring == 0
    if kind == "AND":
        out = np.ones(packed.n_nodes, dtype=bool)
        for child in ast[1:]:
            out &= _eval(packed, child)
        return out
    if kind == "NBR":
        counts = _nbr_counts(packed, _eval(packed, ast[1]), ast[2])
        return counts >= int(ast[3])
    if kind == "BRANCH":
        return _branch_mask(packed, ast[1], ast[2], ast[3], ast[4])
    raise QueryError(f"unknown AST kind {kind!r}")


def _nbr_counts(packed: PackedGraph, parent_mask: np.ndarray, bond: tuple) -> np.ndarray:
    """Per-root count of neighbours u with bond(v,u) and P(u)."""
    counts = np.zeros(packed.n_nodes, dtype=np.int32)
    if packed.n_bonds == 0:
        return counts
    ok = bond_ok(packed, bond)
    from_u = parent_mask[packed.bond_u] & ok
    if from_u.any():
        counts += np.bincount(
            packed.bond_v[from_u], minlength=packed.n_nodes
        ).astype(np.int32)
    from_v = parent_mask[packed.bond_v] & ok
    if from_v.any():
        counts += np.bincount(
            packed.bond_u[from_v], minlength=packed.n_nodes
        ).astype(np.int32)
    return counts


def _branch_mask(
    packed: PackedGraph, p: tuple, q: tuple, b1: tuple, b2: tuple
) -> np.ndarray:
    mask_p = _eval(packed, p)
    mask_q = _eval(packed, q)
    cp = _nbr_counts(packed, mask_p, b1)
    cq = _nbr_counts(packed, mask_q, b2)
    # Joint overlap: a neighbour must satisfy P AND Q AND b1 AND b2.
    bond1 = bond_ok(packed, b1)
    bond2 = bond_ok(packed, b2)
    joint = np.zeros(packed.n_nodes, dtype=np.int32)
    if packed.n_bonds:
        from_u = mask_p[packed.bond_u] & mask_q[packed.bond_u] & bond1 & bond2
        if from_u.any():
            joint += np.bincount(packed.bond_v[from_u], minlength=packed.n_nodes).astype(np.int32)
        from_v = mask_p[packed.bond_v] & mask_q[packed.bond_v] & bond1 & bond2
        if from_v.any():
            joint += np.bincount(packed.bond_u[from_v], minlength=packed.n_nodes).astype(np.int32)
    return (cp > 0) & (cq > 0) & ((cp + cq - joint) >= 2)


# ---------------------------------------------------------------------------
# explanation witnesses
# ---------------------------------------------------------------------------


def witness(packed: PackedGraph, ast: tuple, root: int) -> dict[str, Any]:
    """Structural witness that ``root`` matches ``ast`` (raises if it does not)."""
    if not bool(_eval_single(packed, ast, root)):
        raise QueryError(f"root {root} does not match {canon(ast)}")
    return _witness(packed, ast, root)


def _eval_single(packed: PackedGraph, ast: tuple, node: int) -> bool:
    kind = ast[0]
    if kind == "TRUE":
        return True
    if kind == "ATOM_EQ":
        return int(packed.atom_feat[node, ast[1]]) == ast[2]
    if kind == "DEG_EQ":
        return int(packed.degree[node]) == ast[1]
    if kind == "DEG_GE":
        return int(packed.degree[node]) >= ast[1]
    if kind == "IN_CYCLE":
        return bool(packed.in_cycle[node])
    if kind == "MIN_RING_EQ":
        return int(packed.min_ring[node]) == ast[1]
    if kind == "MIN_RING_GE":
        return int(packed.min_ring[node]) >= ast[1]
    if kind == "ACYCLIC":
        return int(packed.min_ring[node]) == 0
    if kind == "AND":
        return all(_eval_single(packed, child, node) for child in ast[1:])
    if kind == "NBR":
        hits = _nbr_hits(packed, ast[1], ast[2], node)
        return len(hits) >= int(ast[3])
    if kind == "BRANCH":
        return _branch_witness(packed, ast[1], ast[2], ast[3], ast[4], node) is not None
    raise QueryError(f"unknown AST kind {kind!r}")


def _nbr_hits(packed: PackedGraph, parent: tuple, bond: tuple, node: int) -> list[int]:
    hits: list[int] = []
    for neighbor, bond_idx in packed.neighbors(node):
        if bond[0] == "BOND_EQ" and int(packed.bond_feat[bond_idx, bond[1]]) != bond[2]:
            continue
        if _eval_single(packed, parent, neighbor):
            hits.append(int(neighbor))
    return hits


def _branch_witness(
    packed: PackedGraph, p: tuple, q: tuple, b1: tuple, b2: tuple, node: int
) -> tuple[int, int] | None:
    left = _nbr_hits(packed, p, b1, node)
    right = _nbr_hits(packed, q, b2, node)
    for u in left:
        for w in right:
            if u != w:
                return int(u), int(w)
    return None


def _witness(packed: PackedGraph, ast: tuple, node: int) -> dict[str, Any]:
    kind = ast[0]
    payload: dict[str, Any] = {
        "root": int(node),
        "query": canon(ast),
        "atom_features": [int(v) for v in packed.atom_feat[node]],
        "degree": int(packed.degree[node]),
        "min_ring": int(packed.min_ring[node]),
        "in_cycle": bool(packed.in_cycle[node]),
    }
    if kind == "AND":
        payload["children"] = [_witness(packed, child, node) for child in ast[1:]]
    elif kind == "NBR":
        hits = _nbr_hits(packed, ast[1], ast[2], node)
        payload["nbr_hits"] = [
            {"neighbor": int(u), "child": _witness(packed, ast[1], int(u))} for u in hits
        ]
        payload["m"] = int(ast[3])
        payload["bond"] = _bond_name(ast[2])
    elif kind == "BRANCH":
        pair = _branch_witness(packed, ast[1], ast[2], ast[3], ast[4], node)
        if pair is None:  # pragma: no cover - guarded by the caller
            raise QueryError("branch witness disappeared")
        u, w = pair
        payload["branch"] = {
            "u_neighbor": int(u),
            "w_neighbor": int(w),
            "u_witness": _witness(packed, ast[1], int(u)),
            "w_witness": _witness(packed, ast[2], int(w)),
            "b1": _bond_name(ast[3]),
            "b2": _bond_name(ast[4]),
        }
    return payload


# ---------------------------------------------------------------------------
# feature transforms (four columns per query)
# ---------------------------------------------------------------------------

FEATURE_TRANSFORMS: tuple[str, ...] = ("matched_roots", "density", "exists", "log1p_count")


def query_features(packed: PackedGraph, node_mask: np.ndarray) -> np.ndarray:
    """Dense float32 (n_graphs, 4) = [c, p, e, log1p(c)] for one query."""
    counts = packed.graph_counts(node_mask).astype(np.float64)
    n_atoms = packed.graph_n_atoms.astype(np.float64)
    density = counts / n_atoms
    exists = (counts > 0).astype(np.float64)
    log_counts = np.log1p(counts)
    return np.stack([counts, density, exists, log_counts], axis=1).astype(np.float32)


def monotone_duplicate(node_mask_a: np.ndarray, node_mask_b: np.ndarray) -> bool:
    """True when two root-truth vectors are identical (Train-equivalence)."""
    return bool(np.array_equal(node_mask_a, node_mask_b))
