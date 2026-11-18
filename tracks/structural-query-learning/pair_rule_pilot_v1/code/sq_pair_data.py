"""Original-graph objects, OGB decoding and the explicit structural counters.

Everything here operates on *one molecule at a time*; it deliberately shares no
precomputed matrix with the fit.  That property is what makes the rule replay
audit independent from the training artefacts.

Rule type of an atom ``v`` is the integer tuple

    t(v) = (atom_feat[0], atom_feat[3], atom_feat[7], atom_feat[8], role)

i.e. OGB *category indices* for element, formal charge, aromatic and in-ring,
plus a role for the C=O / O=C double-bond pair.  Keeping category indices (not
decoded strings) means every key is exactly the stored OGB column value.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import sq_pair_common as C

import sq_ast as A
import sq_data as D

BOND_TYPE_DOUBLE = 1  # OGB possible_bond_type_list[1] == "DOUBLE" (asserted below)


# ---------------------------------------------------------------------------
# OGB decoding tables
# ---------------------------------------------------------------------------


def ogb_decoding() -> dict[str, Any]:
    from ogb.utils.features import allowable_features

    atom_lists = {
        "atomic_num": list(allowable_features["possible_atomic_num_list"]),
        "chirality": list(allowable_features["possible_chirality_list"]),
        "degree": list(allowable_features["possible_degree_list"]),
        "formal_charge": list(allowable_features["possible_formal_charge_list"]),
        "num_hs": list(allowable_features["possible_numH_list"]),
        "num_radical_electrons": list(allowable_features["possible_number_radical_e_list"]),
        "hybridization": list(allowable_features["possible_hybridization_list"]),
        "is_aromatic": list(allowable_features["possible_is_aromatic_list"]),
        "is_in_ring": list(allowable_features["possible_is_in_ring_list"]),
    }
    bond_lists = {
        "bond_type": list(allowable_features["possible_bond_type_list"]),
        "bond_stereo": list(allowable_features["possible_bond_stereo_list"]),
        # The frozen cache's third bond column is the OGB conjugation code, even
        # though the reused ``sq_data.BOND_COLUMN_NAMES`` label says otherwise.
        "bond_conjugated": list(allowable_features["possible_is_conjugated_list"]),
    }
    double_index = int(bond_lists["bond_type"].index("DOUBLE"))
    carbon_index = int(atom_lists["atomic_num"].index(6))
    oxygen_index = int(atom_lists["atomic_num"].index(8))
    if double_index != BOND_TYPE_DOUBLE:
        raise RuntimeError("OGB DOUBLE bond index changed; the frozen role rule is invalid")
    return {
        "atom": atom_lists,
        "bond": bond_lists,
        "double_bond_index": double_index,
        "carbon_atomic_num_index": carbon_index,
        "oxygen_atomic_num_index": oxygen_index,
        "atom_rule_columns": [0, 3, 7, 8],
        "atom_rule_names": ["element", "formal_charge", "is_aromatic", "is_in_ring"],
        "bond_columns": [0, 1, 2],
        "bond_column_names": ["bond_type", "bond_stereo", "bond_conjugated"],
        "role_names": dict(C.ROLE_NAMES),
    }


def decode_atom_type(decoding: Mapping[str, Any], t: Sequence[int]) -> dict[str, Any]:
    atom = decoding["atom"]
    element_index, charge_index, aromatic_index, ring_index, role = (int(x) for x in t)
    return {
        "element": atom["atomic_num"][element_index],
        "formal_charge": atom["formal_charge"][charge_index],
        "is_aromatic": bool(atom["is_aromatic"][aromatic_index]),
        "is_in_ring": bool(atom["is_in_ring"][ring_index]),
        "role": decoding["role_names"][int(role)],
        "raw": [element_index, charge_index, aromatic_index, ring_index, role],
    }


def decode_bond(decoding: Mapping[str, Any], feat: Sequence[int]) -> dict[str, Any]:
    bond = decoding["bond"]
    return {
        "bond_type": bond["bond_type"][int(feat[0])],
        "bond_stereo": bond["bond_stereo"][int(feat[1])],
        "bond_conjugated": bool(bond["bond_conjugated"][int(feat[2])]),
        "raw": [int(x) for x in feat],
    }


def validate_bond_columns(bond_feat: np.ndarray, decoding: Mapping[str, Any]) -> dict[str, Any]:
    arr = np.asarray(bond_feat)
    maxima = [int(arr[:, j].max()) if arr.size else -1 for j in range(arr.shape[1])]
    limits = [
        len(decoding["bond"]["bond_type"]),
        len(decoding["bond"]["bond_stereo"]),
        len(decoding["bond"]["bond_conjugated"]),
    ]
    ok = all(0 <= maxima[j] < limits[j] for j in range(3))
    if not ok:
        raise ValueError(f"bond category index outside the OGB decoding table: {maxima} vs {limits}")
    return {"maxima": maxima, "limits": limits, "valid": True}


# ---------------------------------------------------------------------------
# molecule object
# ---------------------------------------------------------------------------


@dataclass
class MolGraph:
    row_id: int
    smiles: str
    n_atoms: int
    n_bonds: int
    n_components: int
    cycle_rank: int
    atom_feat: np.ndarray      # (n, 9) int16, OGB category indices
    bond_u: np.ndarray         # (m,) int32 local, u < v
    bond_v: np.ndarray
    bond_feat: np.ndarray      # (m, 3) int16
    degree: np.ndarray         # (n,) int16
    min_ring: np.ndarray       # (n,) int16
    in_cycle: np.ndarray       # (n,) bool

    # -- derived cached structure -----------------------------------------
    def __post_init__(self) -> None:
        if self.n_atoms != int(self.atom_feat.shape[0]):
            raise ValueError("atom_feat does not match n_atoms")
        if self.n_bonds != int(self.bond_u.size):
            raise ValueError("bond_u does not match n_bonds")
        if self.bond_v.size != self.bond_u.size or self.bond_feat.shape[0] != self.bond_u.size:
            raise ValueError("bond arrays disagree")
        if self.bond_u.size and (self.bond_u >= self.bond_v).any():
            raise ValueError("physical bonds must be stored once with u < v")
        self._adj: list[list[tuple[int, tuple[int, int, int]]]] | None = None
        self._adj_simple: list[list[int]] | None = None
        self._types: list[tuple[int, int, int, int, int]] | None = None
        self._role: np.ndarray | None = None

    @property
    def adj(self) -> list[list[tuple[int, tuple[int, int, int]]]]:
        if self._adj is None:
            adj: list[list[tuple[int, tuple[int, int, int]]]] = [[] for _ in range(self.n_atoms)]
            for k in range(self.n_bonds):
                u = int(self.bond_u[k])
                v = int(self.bond_v[k])
                feat = tuple(int(x) for x in self.bond_feat[k])
                adj[u].append((v, feat))
                adj[v].append((u, feat))
            self._adj = adj
        return self._adj

    @property
    def adj_simple(self) -> list[list[int]]:
        if self._adj_simple is None:
            self._adj_simple = [[nbr for nbr, _ in row] for row in self.adj]
        return self._adj_simple

    @property
    def role(self) -> np.ndarray:
        if self._role is None:
            roles = np.zeros(self.n_atoms, dtype=np.int8)
            carbon = None
            oxygen = None
            from ogb.utils.features import allowable_features

            carbon = int(allowable_features["possible_atomic_num_list"].index(6))
            oxygen = int(allowable_features["possible_atomic_num_list"].index(8))
            for k in range(self.n_bonds):
                if int(self.bond_feat[k, 0]) != BOND_TYPE_DOUBLE:
                    continue
                u = int(self.bond_u[k])
                v = int(self.bond_v[k])
                eu = int(self.atom_feat[u, 0])
                ev = int(self.atom_feat[v, 0])
                if eu == carbon and ev == oxygen:
                    roles[u] = C.ROLE_CARBOXYL_C
                    roles[v] = C.ROLE_CARBONYL_O
                elif eu == oxygen and ev == carbon:
                    roles[u] = C.ROLE_CARBONYL_O
                    roles[v] = C.ROLE_CARBOXYL_C
            self._role = roles
        return self._role

    @property
    def types(self) -> list[tuple[int, int, int, int, int]]:
        if self._types is None:
            roles = self.role
            self._types = [
                (
                    int(self.atom_feat[i, 0]),
                    int(self.atom_feat[i, 3]),
                    int(self.atom_feat[i, 7]),
                    int(self.atom_feat[i, 8]),
                    int(roles[i]),
                )
                for i in range(self.n_atoms)
            ]
        return self._types

    def distances(self, source: int) -> list[int]:
        dist = [-1] * self.n_atoms
        dist[int(source)] = 0
        queue = deque([int(source)])
        adj = self.adj_simple
        while queue:
            node = queue.popleft()
            d = dist[node] + 1
            for nbr in adj[node]:
                if dist[nbr] < 0:
                    dist[nbr] = d
                    queue.append(nbr)
        return dist

    def packed(self) -> A.PackedGraph:
        n = self.n_atoms
        return A.PackedGraph(
            dataset="single",
            split="single",
            n_graphs=1,
            n_nodes=n,
            node_graph=np.zeros(n, dtype=np.int32),
            atom_feat=np.asarray(self.atom_feat, dtype=np.int16),
            degree=np.asarray(self.degree, dtype=np.int16),
            min_ring=np.asarray(self.min_ring, dtype=np.int16),
            in_cycle=np.asarray(self.in_cycle, dtype=bool),
            n_bonds=self.n_bonds,
            bond_u=np.asarray(self.bond_u, dtype=np.int32),
            bond_v=np.asarray(self.bond_v, dtype=np.int32),
            bond_feat=np.asarray(self.bond_feat, dtype=np.int16),
            graph_n_atoms=np.asarray([n], dtype=np.int32),
            graph_n_bonds=np.asarray([self.n_bonds], dtype=np.int32),
            graph_n_components=np.asarray([self.n_components], dtype=np.int32),
            graph_cycle_rank=np.asarray([self.cycle_rank], dtype=np.int32),
            row_ids=np.asarray([self.row_id], dtype=np.int64),
            labels=np.zeros(1, dtype=np.float32),
            smiles=[self.smiles],
        )


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _union_components(n_atoms: int, u: np.ndarray, v: np.ndarray) -> int:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(u.tolist(), v.tolist()):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n_atoms)})


def graph_from_arrays(
    *,
    atom_feat: np.ndarray,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_feat: np.ndarray,
    row_id: int = -1,
    smiles: str = "",
    n_components: int | None = None,
    cycle_rank: int | None = None,
    degree: np.ndarray | None = None,
    min_ring: np.ndarray | None = None,
    in_cycle: np.ndarray | None = None,
) -> MolGraph:
    atom_feat = np.asarray(atom_feat, dtype=np.int16)
    n = int(atom_feat.shape[0])
    u = np.asarray(bond_u, dtype=np.int64)
    v = np.asarray(bond_v, dtype=np.int64)
    feat = np.asarray(bond_feat, dtype=np.int16).reshape(-1, 3)
    if u.size:
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        order = np.lexsort((hi, lo))
        u = lo[order].astype(np.int32)
        v = hi[order].astype(np.int32)
        feat = feat[order]
    else:
        u = np.zeros(0, dtype=np.int32)
        v = np.zeros(0, dtype=np.int32)
        feat = np.zeros((0, 3), dtype=np.int16)
    if degree is None:
        degree = np.zeros(n, dtype=np.int16)
        for a, b in zip(u.tolist(), v.tolist()):
            degree[a] += 1
            degree[b] += 1
    if n_components is None:
        n_components = _union_components(n, u, v)
    if cycle_rank is None:
        cycle_rank = int(u.size - n + int(n_components))
    if min_ring is None or in_cycle is None:
        rings, cycles = D.min_ring_buckets(n, u, v)
        min_ring = rings if min_ring is None else min_ring
        in_cycle = cycles if in_cycle is None else in_cycle
    return MolGraph(
        row_id=int(row_id),
        smiles=str(smiles),
        n_atoms=n,
        n_bonds=int(u.size),
        n_components=int(n_components),
        cycle_rank=int(cycle_rank),
        atom_feat=atom_feat,
        bond_u=np.asarray(u, dtype=np.int32),
        bond_v=np.asarray(v, dtype=np.int32),
        bond_feat=feat,
        degree=np.asarray(degree, dtype=np.int16),
        min_ring=np.asarray(min_ring, dtype=np.int16),
        in_cycle=np.asarray(in_cycle, dtype=bool),
    )


def graph_from_payload(payload: Mapping[str, Any], row: int) -> MolGraph:
    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    return MolGraph(
        row_id=int(row),
        smiles=str(payload["smiles"][row]),
        n_atoms=int(a1 - a0),
        n_bonds=int(b1 - b0),
        n_components=int(np.asarray(payload["graph_n_components"])[row]),
        cycle_rank=int(np.asarray(payload["graph_cycle_rank"])[row]),
        atom_feat=np.asarray(payload["atom_feat"])[a0:a1].astype(np.int16),
        bond_u=np.asarray(payload["bond_u"])[b0:b1].astype(np.int32),
        bond_v=np.asarray(payload["bond_v"])[b0:b1].astype(np.int32),
        bond_feat=np.asarray(payload["bond_feat"])[b0:b1].astype(np.int16),
        degree=np.asarray(payload["degree"])[a0:a1].astype(np.int16),
        min_ring=np.asarray(payload["min_ring"])[a0:a1].astype(np.int16),
        in_cycle=np.asarray(payload["in_cycle"])[a0:a1].astype(bool),
    )


def relabel(g: MolGraph, perm: Sequence[int]) -> MolGraph:
    """Return a copy whose atom ``i`` is the old atom ``perm[i]``."""
    perm_arr = np.asarray(perm, dtype=np.int64)
    if perm_arr.size != g.n_atoms or sorted(perm_arr.tolist()) != list(range(g.n_atoms)):
        raise ValueError("relabelling permutation must be a permutation of 0..n-1")
    inverse = np.empty(g.n_atoms, dtype=np.int64)
    inverse[perm_arr] = np.arange(g.n_atoms, dtype=np.int64)
    atom_feat = g.atom_feat[perm_arr]
    u = inverse[g.bond_u]
    v = inverse[g.bond_v]
    return graph_from_arrays(
        atom_feat=atom_feat,
        bond_u=u,
        bond_v=v,
        bond_feat=g.bond_feat,
        row_id=g.row_id,
        smiles=g.smiles,
    )


# ---------------------------------------------------------------------------
# canonical keys
# ---------------------------------------------------------------------------


def type_name(t: Sequence[int]) -> str:
    return ",".join(str(int(x)) for x in t)


def type_pair(types: Sequence[tuple[int, ...]], u: int, v: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    a = tuple(int(x) for x in types[u])
    b = tuple(int(x) for x in types[v])
    return (a, b) if a <= b else (b, a)


def type_pair_name(tp: Sequence[Sequence[int]]) -> str:
    a, b = tp
    return f"TP|{type_name(a)}|{type_name(b)}"


def distance_bin(distance: int) -> str:
    d = int(distance)
    if d < 0:
        return "unreachable"
    if d <= 8:
        return str(d)
    return "9+"


def bin_key(bin_name: str) -> str:
    return f"D|{bin_name}"


def s2_full_key(
    tu: Sequence[int],
    tv: Sequence[int],
    tw: Sequence[int],
    bu: Sequence[int],
    bw: Sequence[int],
    adjacent: bool,
) -> tuple[Any, ...]:
    return (
        tuple(int(x) for x in tu),
        tuple(int(x) for x in tv),
        tuple(int(x) for x in tw),
        tuple(int(x) for x in bu),
        tuple(int(x) for x in bw),
        int(bool(adjacent)),
    )


def s2_canonical_key(
    tu: Sequence[int],
    tv: Sequence[int],
    tw: Sequence[int],
    bu: Sequence[int],
    bw: Sequence[int],
    adjacent: bool,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    forward = s2_full_key(tu, tv, tw, bu, bw, adjacent)
    reverse = s2_full_key(tw, tv, tu, bw, bu, adjacent)
    chosen = forward if forward <= reverse else reverse
    name = (
        f"S2|{type_name(chosen[0])}|{type_name(chosen[1])}|{type_name(chosen[2])}"
        f"|{','.join(str(x) for x in chosen[3])}|{','.join(str(x) for x in chosen[4])}|{chosen[5]}"
    )
    meta = {
        "name": name,
        "type_u": list(chosen[0]),
        "type_center": list(chosen[1]),
        "type_w": list(chosen[2]),
        "bond_vu": list(chosen[3]),
        "bond_vw": list(chosen[4]),
        "u_w_adjacent": bool(chosen[5]),
        "full_key": [list(x) if isinstance(x, tuple) else x for x in chosen],
    }
    return chosen, meta


# ---------------------------------------------------------------------------
# features of one molecule
# ---------------------------------------------------------------------------


def named_counts(g: MolGraph) -> dict[str, int]:
    return {
        "n_atoms": int(g.n_atoms),
        "n_bonds": int(g.n_bonds),
        "n_components": int(g.n_components),
        "cycle_rank": int(g.cycle_rank),
    }


def s2_counts(g: MolGraph) -> dict[str, dict[str, Any]]:
    """Canonical typed two-bond paths, each undirected path counted once."""
    types = g.types
    adj = g.adj
    out: dict[str, dict[str, Any]] = {}
    for center in range(g.n_atoms):
        neighbours = adj[center]
        for i in range(len(neighbours)):
            u, bu = neighbours[i]
            for j in range(i + 1, len(neighbours)):
                w, bw = neighbours[j]
                if u == w:
                    continue
                adjacent = w in g.adj_simple[u]
                _, meta = s2_canonical_key(
                    types[u], types[center], types[w], bu, bw, adjacent
                )
                name = meta["name"]
                entry = out.get(name)
                if entry is None:
                    entry = {"count": 0, "meta": meta}
                    out[name] = entry
                entry["count"] += 1
    return out


def pair_features(g: MolGraph) -> dict[str, Any]:
    """Type-pair totals, distance marginals and finite (type, distance) counts."""
    types = g.types
    n = g.n_atoms
    total_pairs = n * (n - 1) // 2
    bin_counts: dict[str, int] = {b: 0 for b in C.DISTANCE_BINS}
    type_counts: dict[str, int] = {}
    ext_counts: dict[str, int] = {}
    type_meta: dict[str, Any] = {}
    for source in range(n):
        dist = g.distances(source)
        for target in range(source + 1, n):
            d = int(dist[target])
            b = distance_bin(d)
            bin_counts[b] += 1
            tp = type_pair(types, source, target)
            tp_name = type_pair_name(tp)
            type_counts[tp_name] = type_counts.get(tp_name, 0) + 1
            if tp_name not in type_meta:
                type_meta[tp_name] = {"name": tp_name, "type_a": list(tp[0]), "type_b": list(tp[1])}
            if b in C.EXTENSION_BINS:
                key = f"{tp_name}|{b}"
                ext_counts[key] = ext_counts.get(key, 0) + 1
    finite_ge3 = sum(bin_counts[b] for b in C.EXTENSION_BINS)
    checks = {
        "total_pairs": int(total_pairs),
        "sum_bins": int(sum(bin_counts.values())),
        "sum_type_pairs": int(sum(type_counts.values())),
        "finite_margin": int(finite_ge3),
        "sum_ext": int(sum(ext_counts.values())),
    }
    checks["ok"] = bool(
        checks["sum_bins"] == total_pairs
        and checks["sum_type_pairs"] == total_pairs
        and checks["finite_margin"] == total_pairs - bin_counts["1"] - bin_counts["2"] - bin_counts["unreachable"]
        and checks["sum_ext"] == checks["finite_margin"]
    )
    return {
        "named": named_counts(g),
        "s2": s2_counts(g),
        "bin_counts": bin_counts,
        "type_counts": type_counts,
        "type_meta": type_meta,
        "ext_counts": ext_counts,
        "checks": checks,
    }


def group_count_for(graph_features: Mapping[str, Any], kind: str, key: str) -> int:
    if kind == "graph_count":
        return int(graph_features["named"][key])
    if kind == "s2_path":
        entry = graph_features["s2"].get(key)
        return 0 if entry is None else int(entry["count"])
    if kind == "type_pair_total":
        return int(graph_features["type_counts"].get(key, 0))
    if kind == "distance_bin":
        return int(graph_features["bin_counts"][key])
    if kind == "relation_real":
        return int(graph_features["ext_counts"].get(key, 0))
    if kind == "b_query":
        raise ValueError("b_query counts need the AST evaluator, not graph_features")
    raise ValueError(f"unknown group kind {kind!r}")


def relation_slot_key(tp_name: str, bin_name: str) -> str:
    return f"{tp_name}|{bin_name}"


def _split_relation_key(key: str) -> tuple[str, str]:
    # type-pair names contain '|' themselves; the distance bin is the final part
    head, _, tail = key.rpartition("|")
    return head, tail


def support_details(g: MolGraph, kind: str, key: str, *, limit: int = 4) -> dict[str, Any]:
    """Explain the support of one group on one molecule (never used for fitting)."""
    if kind == "graph_count":
        value = int(named_counts(g)[key])
        return {"kind": kind, "key": key, "count": value}
    if kind == "s2_path":
        items = []
        for center in range(g.n_atoms):
            neighbours = g.adj[center]
            for i in range(len(neighbours)):
                u, bu = neighbours[i]
                for j in range(i + 1, len(neighbours)):
                    w, bw = neighbours[j]
                    if u == w:
                        continue
                    _, meta = s2_canonical_key(
                        g.types[u], g.types[center], g.types[w], bu, bw, w in g.adj_simple[u]
                    )
                    if meta["name"] == key:
                        items.append({"center": int(center), "u": int(u), "w": int(w)})
        return {"kind": kind, "key": key, "count": len(items), "items": items[:limit]}
    if kind in ("type_pair_total", "distance_bin", "relation_real"):
        if kind == "relation_real":
            tp_name, bin_name = _split_relation_key(key)
            want_bin = bin_name
        elif kind == "distance_bin":
            tp_name, want_bin = None, key
        else:
            tp_name, want_bin = key, None
        items = []
        for source in range(g.n_atoms):
            dist = g.distances(source)
            for target in range(source + 1, g.n_atoms):
                b = distance_bin(dist[target])
                tp = type_pair(g.types, source, target)
                if tp_name is not None and type_pair_name(tp) != tp_name:
                    continue
                if want_bin is not None and b != want_bin:
                    continue
                items.append({"u": int(source), "v": int(target), "distance_bin": b})
        return {"kind": kind, "key": key, "count": len(items), "items": items[:limit]}
    if kind == "b_query":
        roots = b_query_roots(g, A.from_json(json.loads(key)))
        return {"kind": kind, "key": key, "count": len(roots), "root_nodes": roots[:limit]}
    raise ValueError(f"unknown group kind {kind!r}")


# ---------------------------------------------------------------------------
# B query support on one molecule (independent from the packed design matrix)
# ---------------------------------------------------------------------------


def b_query_roots(g: MolGraph, ast: tuple) -> list[int]:
    mask = A.evaluate(g.packed(), ast)
    return [int(i) for i in np.flatnonzero(mask).tolist()]


def b_query_count(g: MolGraph, ast: tuple) -> int:
    return len(b_query_roots(g, ast))


# ---------------------------------------------------------------------------
# synthetic controls
# ---------------------------------------------------------------------------


def _plain_atom(element_index: int, *, in_ring: int, aromatic: int = 0, charge: int = 6) -> list[int]:
    # columns: atomic_num, chirality, degree, formal_charge, num_hs,
    #          radicals, hybridization, aromatic, in_ring
    return [int(element_index), 0, 2, int(charge), 0, 0, 2, int(aromatic), int(in_ring)]


def ring_graph(
    n: int,
    *,
    oxygen_position: int,
    nitrogen_position: int,
    carbon_index: int | None = None,
    oxygen_index: int | None = None,
    nitrogen_index: int | None = None,
) -> MolGraph:
    """A plain ``n``-ring with exactly one O-type and one N-type anchor."""
    decoding = ogb_decoding()
    if carbon_index is None:
        carbon_index = decoding["carbon_atomic_num_index"]
    if oxygen_index is None:
        oxygen_index = decoding["oxygen_atomic_num_index"]
    if nitrogen_index is None:
        nitrogen_index = int(decoding["atom"]["atomic_num"].index(7))
    atom_feat = np.asarray([_plain_atom(carbon_index, in_ring=1) for _ in range(n)], dtype=np.int16)
    atom_feat[oxygen_position] = np.asarray(_plain_atom(oxygen_index, in_ring=1), dtype=np.int16)
    atom_feat[nitrogen_position] = np.asarray(_plain_atom(nitrogen_index, in_ring=1), dtype=np.int16)
    u = list(range(n)) + list(range(1, n)) + [0]
    v = list(range(1, n)) + [0] + list(range(n - 1))
    # every ring bond exactly once with u < v
    bonds = sorted({(min(a, b), max(a, b)) for a, b in zip(u, v)})
    bond_u = np.asarray([a for a, _ in bonds], dtype=np.int32)
    bond_v = np.asarray([b for _, b in bonds], dtype=np.int32)
    bond_feat = np.zeros((len(bonds), 3), dtype=np.int16)  # SINGLE, no stereo, not conjugated
    return graph_from_arrays(
        atom_feat=atom_feat,
        bond_u=bond_u,
        bond_v=bond_v,
        bond_feat=bond_feat,
        row_id=-1,
        smiles=f"synthetic-ring18-O{oxygen_position}-N{nitrogen_position}",
    )


def chain_graph(n: int) -> MolGraph:
    """A plain carbon chain of ``n`` atoms (n-1 bonds)."""
    decoding = ogb_decoding()
    carbon = decoding["carbon_atomic_num_index"]
    atom_feat = np.asarray([_plain_atom(carbon, in_ring=0) for _ in range(n)], dtype=np.int16)
    bonds = [(i, i + 1) for i in range(n - 1)]
    u = np.asarray([a for a, _ in bonds], dtype=np.int32)
    v = np.asarray([b for _, b in bonds], dtype=np.int32)
    bond_feat = np.zeros((len(bonds), 3), dtype=np.int16)
    return graph_from_arrays(
        atom_feat=atom_feat,
        bond_u=u,
        bond_v=v,
        bond_feat=bond_feat,
        row_id=-1,
        smiles=f"synthetic-chain-{n}",
    )


def chain_with_isolated(n_chain: int) -> MolGraph:
    """A carbon chain plus one isolated carbon atom (two components)."""
    g = chain_graph(n_chain)
    decoding = ogb_decoding()
    carbon = decoding["carbon_atomic_num_index"]
    atom_feat = np.vstack([g.atom_feat, np.asarray([_plain_atom(carbon, in_ring=0)], dtype=np.int16)])
    return graph_from_arrays(
        atom_feat=atom_feat,
        bond_u=g.bond_u,
        bond_v=g.bond_v,
        bond_feat=g.bond_feat,
        row_id=-1,
        smiles=f"synthetic-chain-{n_chain}+isolated",
    )
