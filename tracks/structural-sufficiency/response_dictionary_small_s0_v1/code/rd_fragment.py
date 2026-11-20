"""Real single-bond bridge cuts and exact coloured-graph fragment identity.

Every molecule is left intact in the formal data; this module only *derives*
views.  A view is a pair ``(S, C)`` obtained by deleting one non-ring single
bridge of the physical OGB bond graph.  ``S`` is the smaller side (1..8 heavy
atoms) and ``C`` is its complement in the whole molecule (other connected
components included).  The removed bond attributes and both ports are kept so
the original attributed graph can be reassembled exactly.

Identity uses a pynauty certificate of the coloured incidence graph:

* atom vertices carry all nine raw OGB atom columns plus a binary port colour;
* each physical bond becomes an auxiliary vertex carrying all three raw bond
  columns and is joined to its two atom vertices.

The certificate is exact: two fragments share a key iff their coloured
incidence graphs are isomorphic.  Ordinary hashes are never used as identity.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import pynauty

import rd_common as RC

ATOM_COLUMNS = 9
BOND_COLUMNS = 3


# ---------------------------------------------------------------------------
# graph primitives on a physical bond graph
# ---------------------------------------------------------------------------


def adjacency(n_atoms: int, u: np.ndarray, v: np.ndarray) -> list[list[tuple[int, int]]]:
    """Adjacency list of ``(neighbour, physical_bond_index)``."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for k, (a, b) in enumerate(zip(u.tolist(), v.tolist())):
        adj[a].append((b, k))
        adj[b].append((a, k))
    return adj


def find_bridges(n_atoms: int, u: np.ndarray, v: np.ndarray) -> list[int]:
    """Indices of bridges in a simple undirected graph (iterative Tarjan)."""
    adj = adjacency(n_atoms, u, v)
    tin = np.full(n_atoms, -1, dtype=np.int64)
    low = np.zeros(n_atoms, dtype=np.int64)
    bridges: list[int] = []
    timer = 0
    for root in range(n_atoms):
        if tin[root] != -1:
            continue
        tin[root] = low[root] = timer
        timer += 1
        # stack entries: (node, parent_edge, next_adj_index)
        stack: list[list[int]] = [[root, -1, 0]]
        while stack:
            node, parent_edge, pos = stack[-1]
            if pos < len(adj[node]):
                nxt, edge_id = adj[node][pos]
                stack[-1][2] += 1
                if edge_id == parent_edge:
                    continue
                if tin[nxt] == -1:
                    tin[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append([nxt, edge_id, 0])
                else:
                    low[node] = min(low[node], tin[nxt])
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] > tin[parent]:
                        bridges.append(int(parent_edge))
    return sorted(set(bridges))


def component_labels(
    n_atoms: int,
    u: np.ndarray,
    v: np.ndarray,
    *,
    skip_edge: int | None = None,
) -> np.ndarray:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, (a, b) in enumerate(zip(u.tolist(), v.tolist())):
        if skip_edge is not None and k == skip_edge:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(n_atoms)]
    remap = {root: i for i, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[r] for r in roots], dtype=np.int32)


# ---------------------------------------------------------------------------
# exact coloured-graph identity
# ---------------------------------------------------------------------------


def fragment_certificate(
    atom_feat: np.ndarray,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_feat: np.ndarray,
    *,
    port_local: int | None,
) -> str:
    """Exact, relabelling-invariant serialization of the coloured incidence graph.

    ``pynauty.certificate`` only encodes the uncoloured adjacency, so the vertex
    colours are restored through the canonical labelling: vertex colours and
    canonical edge pairs are serialized in canonical-position order.  Two
    fragments share this string iff they are isomorphic *as coloured graphs*.
    """
    n_atoms = int(atom_feat.shape[0])
    n_bonds = int(bond_u.shape[0])
    n = n_atoms + n_bonds
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for k, (a, b) in enumerate(zip(bond_u.tolist(), bond_v.tolist())):
        bond_node = n_atoms + k
        adj[int(a)].add(bond_node)
        adj[int(b)].add(bond_node)
        adj[bond_node].add(int(a))
        adj[bond_node].add(int(b))
    colours: dict[int, tuple[int, ...]] = {}
    for i in range(n_atoms):
        port = 1 if port_local is not None and i == int(port_local) else 0
        colours[i] = (0, *[int(x) for x in atom_feat[i]], port)
    for k in range(n_bonds):
        colours[n_atoms + k] = (1, *[int(x) for x in bond_feat[k]])
    color_classes: dict[tuple[int, ...], set[int]] = {}
    for vertex, colour in colours.items():
        color_classes.setdefault(colour, set()).add(vertex)
    coloring = [color_classes[key] for key in sorted(color_classes)]
    graph = pynauty.Graph(
        number_of_vertices=n,
        directed=False,
        adjacency_dict={node: sorted(nbrs) for node, nbrs in adj.items()},
        vertex_coloring=coloring,
    )
    order = pynauty.canon_label(graph)  # canonical position -> original vertex
    position = {vertex: index for index, vertex in enumerate(order)}
    node_part = ";".join(",".join(str(x) for x in colours[vertex]) for vertex in order)
    edge_keys = []
    for k, (a, b) in enumerate(zip(bond_u.tolist(), bond_v.tolist())):
        bond_node = n_atoms + k
        edge_keys.append(tuple(sorted((position[int(a)], position[bond_node]))))
        edge_keys.append(tuple(sorted((position[int(b)], position[bond_node]))))
    edge_part = ";".join(f"{a}-{b}" for a, b in sorted(set(edge_keys)))
    return "N|" + node_part + "|E|" + edge_part


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


@dataclass
class Fragment:
    atom_feat: np.ndarray
    internal_degree: np.ndarray
    port_local: int | None
    bond_u: np.ndarray
    bond_v: np.ndarray
    bond_feat: np.ndarray
    key: str          # SHA-256 of the exact canonical serialization
    canonical: str    # full serialization, kept only during preparation


@dataclass
class View:
    s: Fragment
    c: Fragment
    cut_bond_feat: tuple[int, int, int]
    cut_bond_u_global: int
    cut_bond_v_global: int
    s_atoms_global: tuple[int, ...]
    c_atoms_global: tuple[int, ...]


def _induce(
    atoms: Sequence[int],
    atom_feat: np.ndarray,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_feat: np.ndarray,
    *,
    port_global: int,
) -> Fragment:
    atoms_sorted = np.asarray(sorted(int(a) for a in atoms), dtype=np.int64)
    remap = {int(g): i for i, g in enumerate(atoms_sorted.tolist())}
    sub_feat = np.asarray(atom_feat[atoms_sorted], dtype=np.int16)
    in_set = set(remap)
    bu: list[int] = []
    bv: list[int] = []
    bf: list[int] = []
    for k, (a, b) in enumerate(zip(bond_u.tolist(), bond_v.tolist())):
        if a in in_set and b in in_set:
            bu.append(remap[int(a)])
            bv.append(remap[int(b)])
            bf.append(k)
    bu_arr = np.asarray(bu, dtype=np.int32)
    bv_arr = np.asarray(bv, dtype=np.int32)
    bf_arr = np.asarray(bond_feat[bf], dtype=np.int16).reshape(-1, BOND_COLUMNS)
    internal = np.zeros(len(atoms_sorted), dtype=np.int16)
    for a, b in zip(bu_arr.tolist(), bv_arr.tolist()):
        internal[a] += 1
        internal[b] += 1
    port_local = remap.get(int(port_global))
    if port_local is None:
        raise ValueError("port atom is outside the induced fragment")
    canonical = fragment_certificate(
        sub_feat, bu_arr, bv_arr, bf_arr, port_local=port_local
    )
    return Fragment(
        atom_feat=sub_feat,
        internal_degree=internal,
        port_local=int(port_local),
        bond_u=bu_arr,
        bond_v=bv_arr,
        bond_feat=bf_arr,
        key=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        canonical=canonical,
    )


def whole_molecule_fragment(
    atom_feat: np.ndarray,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_feat: np.ndarray,
) -> Fragment:
    n = int(atom_feat.shape[0])
    internal = np.zeros(n, dtype=np.int16)
    for a, b in zip(bond_u.tolist(), bond_v.tolist()):
        internal[a] += 1
        internal[b] += 1
    canonical = fragment_certificate(
        np.asarray(atom_feat, dtype=np.int16),
        np.asarray(bond_u, dtype=np.int32),
        np.asarray(bond_v, dtype=np.int32),
        np.asarray(bond_feat, dtype=np.int16),
        port_local=None,
    )
    return Fragment(
        atom_feat=np.asarray(atom_feat, dtype=np.int16),
        internal_degree=internal,
        port_local=None,
        bond_u=np.asarray(bond_u, dtype=np.int32),
        bond_v=np.asarray(bond_v, dtype=np.int32),
        bond_feat=np.asarray(bond_feat, dtype=np.int16),
        key=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        canonical=canonical,
    )


def enumerate_views(
    n_atoms: int,
    atom_feat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bond_feat: np.ndarray,
    *,
    max_fragment_atoms: int = 8,
) -> tuple[list[View], dict[str, int]]:
    """All distinct real single-bond bridge cuts (with both equal-size sides)."""
    stats = {
        "n_bridges": 0,
        "n_single_nonring_bridges": 0,
        "n_rejected_large_small_side": 0,
        "n_views": 0,
    }
    bridges = find_bridges(n_atoms, u, v)
    stats["n_bridges"] = len(bridges)
    views: list[View] = []
    for edge_id in bridges:
        if int(bond_feat[edge_id, 0]) != 0 or int(bond_feat[edge_id, 2]) != 0:
            continue
        stats["n_single_nonring_bridges"] += 1
        a = int(u[edge_id])
        b = int(v[edge_id])
        labels = component_labels(n_atoms, u, v, skip_edge=edge_id)
        side_a = np.flatnonzero(labels == labels[a])
        side_b = np.flatnonzero(labels == labels[b])
        orientations: list[tuple[np.ndarray, int, int]] = []
        if len(side_a) <= len(side_b):
            orientations.append((side_a, a, b))
        if len(side_b) <= len(side_a):
            orientations.append((side_b, b, a))
        for side, port_s, port_c in orientations:
            if not (1 <= len(side) <= max_fragment_atoms):
                stats["n_rejected_large_small_side"] += 1
                continue
            complement = np.asarray(
                [i for i in range(n_atoms) if i not in set(side.tolist())], dtype=np.int64
            )
            frag_s = _induce(
                side, atom_feat, u, v, bond_feat, port_global=port_s
            )
            frag_c = _induce(
                complement, atom_feat, u, v, bond_feat, port_global=port_c
            )
            views.append(
                View(
                    s=frag_s,
                    c=frag_c,
                    cut_bond_feat=tuple(int(x) for x in bond_feat[edge_id]),
                    cut_bond_u_global=int(u[edge_id]),
                    cut_bond_v_global=int(v[edge_id]),
                    s_atoms_global=tuple(int(x) for x in sorted(side.tolist())),
                    c_atoms_global=tuple(int(x) for x in complement.tolist()),
                )
            )
    views.sort(key=lambda view: (view.s.atom_feat.shape[0], view.s.key, view.c.key, view.cut_bond_feat))
    stats["n_views"] = len(views)
    return views, stats


def select_views(views: Sequence[View], max_views: int = 4) -> list[View]:
    m = len(views)
    if m == 0:
        return []
    q = min(int(max_views), m)
    if q == 1:
        return [views[0]]
    indices = [int(np.floor(j * (m - 1) / (q - 1))) for j in range(q)]
    return [views[i] for i in indices]


# ---------------------------------------------------------------------------
# packed fragment store and view tables
# ---------------------------------------------------------------------------


def _cat(parts, shape, dtype):
    if parts:
        return np.concatenate(parts, axis=0).astype(dtype)
    return np.zeros(shape, dtype=dtype)


class FragmentStore:
    """Deduplicated exact fragments plus per-molecule fixed view lists."""

    def __init__(self) -> None:
        self._by_key: dict[str, int] = {}
        self._canonical: dict[str, str] = {}
        self.n_hash_matches = 0
        self.atom_feat: list[np.ndarray] = []
        self.internal_degree: list[np.ndarray] = []
        self.port_local: list[int] = []
        self.bond_u: list[np.ndarray] = []
        self.bond_v: list[np.ndarray] = []
        self.bond_feat: list[np.ndarray] = []
        self.keys: list[str] = []

    def add(self, fragment: Fragment) -> int:
        existing = self._by_key.get(fragment.key)
        if existing is not None:
            self.n_hash_matches += 1
            if self._canonical[fragment.key] != fragment.canonical:
                raise AssertionError(
                    "canonical serialization hash collision: refusing to merge distinct fragments"
                )
            return existing
        index = len(self.keys)
        self._by_key[fragment.key] = index
        self._canonical[fragment.key] = fragment.canonical
        self.keys.append(fragment.key)
        self.atom_feat.append(np.asarray(fragment.atom_feat, dtype=np.int16))
        self.internal_degree.append(np.asarray(fragment.internal_degree, dtype=np.int16))
        self.port_local.append(-1 if fragment.port_local is None else int(fragment.port_local))
        self.bond_u.append(np.asarray(fragment.bond_u, dtype=np.int32))
        self.bond_v.append(np.asarray(fragment.bond_v, dtype=np.int32))
        self.bond_feat.append(np.asarray(fragment.bond_feat, dtype=np.int16))
        return index

    def __len__(self) -> int:
        return len(self.keys)

    def export(self) -> dict[str, np.ndarray]:
        return {
            "frag_key": np.asarray(self.keys, dtype=str),
            "frag_n_atoms": np.asarray([a.shape[0] for a in self.atom_feat], dtype=np.int32),
            "frag_atom_ptr": np.concatenate([[0], np.cumsum([a.shape[0] for a in self.atom_feat])]).astype(np.int64),
            "frag_atom_feat": _cat(self.atom_feat, (0, ATOM_COLUMNS), np.int16),
            "frag_internal_degree": _cat(self.internal_degree, (0,), np.int16),
            "frag_port_local": np.asarray(self.port_local, dtype=np.int32),
            "frag_n_bonds": np.asarray([a.shape[0] for a in self.bond_u], dtype=np.int32),
            "frag_bond_ptr": np.concatenate([[0], np.cumsum([a.shape[0] for a in self.bond_u])]).astype(np.int64),
            "frag_bond_u": _cat(self.bond_u, (0,), np.int32),
            "frag_bond_v": _cat(self.bond_v, (0,), np.int32),
            "frag_bond_feat": _cat(self.bond_feat, (0, BOND_COLUMNS), np.int16),
        }


def prepare_split(
    payload: Mapping[str, Any],
    split: str,
    store: FragmentStore,
    *,
    max_views: int = 4,
    max_fragment_atoms: int = 8,
    audit: bool = True,
) -> dict[str, Any]:
    indices = np.asarray(payload[f"{split}_idx"], dtype=np.int64)
    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    atom_all = np.asarray(payload["atom_feat"])
    bond_u_all = np.asarray(payload["bond_u"], dtype=np.int32)
    bond_v_all = np.asarray(payload["bond_v"], dtype=np.int32)
    bond_feat_all = np.asarray(payload["bond_feat"])

    view_s: list[int] = []
    view_c: list[int] = []
    view_bond_feat: list[tuple[int, int, int]] = []
    view_mol: list[int] = []
    mol_view_ptr = [0]
    mol_fallback: list[int] = []
    mol_rows = np.asarray(payload["row_ids"])[indices].astype(np.int64)
    mol_labels = np.asarray(payload["labels"])[indices].astype(np.float32)
    mol_n_atoms = np.asarray(payload["graph_n_atoms"])[indices].astype(np.int32)
    split_stats = {
        "n_molecules": int(len(indices)),
        "n_views_total": 0,
        "n_molecules_with_views": 0,
        "n_molecules_fallback": 0,
        "n_duplicate_equal_orientation_records": 0,
        "bridge_stats": {},
        "fallback_row_ids": [],
    }
    reassembly_checked = 0
    for local, graph_id in enumerate(indices.tolist()):
        a0, a1 = int(node_ptr[graph_id]), int(node_ptr[graph_id + 1])
        b0, b1 = int(bond_ptr[graph_id]), int(bond_ptr[graph_id + 1])
        atom_feat = np.asarray(atom_all[a0:a1], dtype=np.int16)
        # The cache stores graph-local bond endpoints (``packed_split`` adds the
        # offset); no offset arithmetic here.
        u = np.asarray(bond_u_all[b0:b1], dtype=np.int32)
        v = np.asarray(bond_v_all[b0:b1], dtype=np.int32)
        if u.size and (int(u.max()) >= int(a1 - a0) or int(v.max()) >= int(a1 - a0)):
            raise ValueError(f"graph {graph_id}: bond endpoint outside the molecule")
        bf = np.asarray(bond_feat_all[b0:b1], dtype=np.int16)
        views, stats = enumerate_views(
            int(a1 - a0), atom_feat, u, v, bf, max_fragment_atoms=max_fragment_atoms
        )
        for key, value in stats.items():
            split_stats["bridge_stats"][key] = split_stats["bridge_stats"].get(key, 0) + value
        chosen = select_views(views, max_views=max_views)
        if audit and chosen:
            verify_reassembly(atom_feat, u, v, bf, chosen[0])
            reassembly_checked += 1
        if not chosen:
            fallback = whole_molecule_fragment(atom_feat, u, v, bf)
            frag_id = store.add(fallback)
            mol_fallback.append(frag_id)
            split_stats["n_molecules_fallback"] += 1
            split_stats["fallback_row_ids"].append(int(mol_rows[local]))
        else:
            mol_fallback.append(-1)
            split_stats["n_molecules_with_views"] += 1
            seen: dict[tuple[int, int, tuple[int, int, int]], int] = {}
            for view in chosen:
                s_id = store.add(view.s)
                c_id = store.add(view.c)
                key = (s_id, c_id, view.cut_bond_feat)
                if key in seen:
                    split_stats["n_duplicate_equal_orientation_records"] += 1
                seen[key] = seen.get(key, 0) + 1
                view_s.append(s_id)
                view_c.append(c_id)
                view_bond_feat.append(view.cut_bond_feat)
                view_mol.append(local)
            split_stats["n_views_total"] += len(chosen)
        mol_view_ptr.append(len(view_mol))
    split_stats["n_reassembly_checked"] = reassembly_checked
    split_stats["fragments_after_split"] = len(store)
    return {
        "mol_row_id": mol_rows,
        "mol_label_raw": mol_labels,
        "mol_n_atoms": mol_n_atoms,
        "mol_view_ptr": np.asarray(mol_view_ptr, dtype=np.int64),
        "view_s": np.asarray(view_s, dtype=np.int32),
        "view_c": np.asarray(view_c, dtype=np.int32),
        "view_bond_feat": np.asarray(view_bond_feat, dtype=np.int16).reshape(-1, BOND_COLUMNS),
        "view_mol": np.asarray(view_mol, dtype=np.int64),
        "mol_fallback": np.asarray(mol_fallback, dtype=np.int32),
        "stats": split_stats,
    }


def verify_reassembly(
    atom_feat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bond_feat: np.ndarray,
    view: View,
) -> None:
    """Raise unless S + C + the cut bond restores the original attributed graph."""
    s_atoms = set(view.s_atoms_global)
    c_atoms = set(view.c_atoms_global)
    if s_atoms & c_atoms or s_atoms | c_atoms != set(range(atom_feat.shape[0])):
        raise AssertionError("S/C partition does not cover the molecule")
    cut = (view.cut_bond_u_global, view.cut_bond_v_global)
    expected = {(int(a), int(b)): tuple(int(x) for x in bond_feat[k]) for k, (a, b) in enumerate(zip(u, v))}
    cut_feat = expected.pop(cut)
    recovered: dict[tuple[int, int], tuple[int, ...]] = {}
    for a, b, f in zip(view.s.bond_u, view.s.bond_v, view.s.bond_feat):
        recovered[(view.s_atoms_global[int(a)], view.s_atoms_global[int(b)])] = tuple(int(x) for x in f)
    for a, b, f in zip(view.c.bond_u, view.c.bond_v, view.c.bond_feat):
        recovered[(view.c_atoms_global[int(a)], view.c_atoms_global[int(b)])] = tuple(int(x) for x in f)
    if recovered != expected:
        raise AssertionError("S + C do not reproduce the original bonds")
    if cut_feat != view.cut_bond_feat:
        raise AssertionError("cut bond attributes changed")
    # The cut bond must join the two declared ports.
    if {view.s_atoms_global[view.s.port_local], view.c_atoms_global[view.c.port_local]} != set(cut):
        raise AssertionError("ports do not correspond to the cut bond")
    # Atom attributes and ports.
    for local, attrs in enumerate(view.s.atom_feat):
        global_id = view.s_atoms_global[local]
        if tuple(int(x) for x in attrs) != tuple(int(x) for x in atom_feat[global_id]):
            raise AssertionError("S atom attributes changed")
    for local, attrs in enumerate(view.c.atom_feat):
        global_id = view.c_atoms_global[local]
        if tuple(int(x) for x in attrs) != tuple(int(x) for x in atom_feat[global_id]):
            raise AssertionError("C atom attributes changed")
    if view.s.port_local is None or view.c.port_local is None:
        raise AssertionError("a real cut must have both ports")


def prepare_dataset_cache(
    dataset: str,
    *,
    max_views: int = 4,
    max_fragment_atoms: int = 8,
) -> dict[str, Any]:
    payload = RC.dataset_cache(dataset)
    store = FragmentStore()
    splits: dict[str, Any] = {}
    t0 = time.perf_counter()
    for split in ("train", "valid", "test"):
        splits[split] = prepare_split(
            payload,
            split,
            store,
            max_views=max_views,
            max_fragment_atoms=max_fragment_atoms,
            audit=True,
        )
        splits[split]["stats"]["fragment_store_size"] = len(store)
    arrays = store.export()
    arrays["canonical_hash_matches_verified"] = np.asarray([int(store.n_hash_matches)], dtype=np.int64)
    for split in ("train", "valid", "test"):
        meta = splits[split].pop("stats")
        for key, value in splits[split].items():
            arrays[f"{split}_{key}"] = value
        arrays[f"{split}_meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))
    arrays["dataset"] = np.asarray(dataset)
    arrays["cache_sha256"] = np.asarray(payload["cache_sha256"])
    arrays["split_sha256"] = np.asarray(
        json.dumps(
            {
                split: RC.sha256_array(np.asarray(payload[f"{split}_idx"], dtype=np.int64))
                for split in ("train", "valid", "test")
            },
            sort_keys=True,
        )
    )
    arrays["scaler_json"] = np.asarray(json.dumps(RC.target_scaler(payload), sort_keys=True))
    arrays["prep_seconds"] = np.asarray([time.perf_counter() - t0], dtype=np.float64)
    arrays["max_views"] = np.asarray([max_views], dtype=np.int32)
    arrays["max_fragment_atoms"] = np.asarray([max_fragment_atoms], dtype=np.int32)
    return arrays


def save_dataset_cache(dataset: str, arrays: Mapping[str, Any]) -> Path:
    path = RC.DATA_DIR / f"{dataset.replace('-', '_')}_rd_frag_v1.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: value for key, value in arrays.items()})
    return path


def load_dataset_cache(dataset: str, *, verify_sha: str | None = None) -> dict[str, Any]:
    path = RC.DATA_DIR / f"{dataset.replace('-', '_')}_rd_frag_v1.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing fragment cache {path}; run rd_prepare first")
    digest = RC.sha256_file(path)
    if verify_sha is not None and digest != verify_sha:
        raise ValueError(f"fragment cache sha mismatch: {digest} != {verify_sha}")
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["fragment_cache_sha256"] = digest
    payload["fragment_cache_path"] = str(path)
    for split in ("train", "valid", "test"):
        payload[f"{split}_meta"] = json.loads(str(payload[f"{split}_meta_json"]))
        for key in ("mol_view_ptr", "view_s", "view_c", "view_mol", "mol_fallback",
                    "mol_row_id", "mol_label_raw", "mol_n_atoms", "view_bond_feat"):
            payload[f"{split}_{key}"] = np.asarray(payload[f"{split}_{key}"])
    payload["scaler"] = json.loads(str(payload["scaler_json"]))
    payload["frag_key"] = [str(x) for x in payload["frag_key"].tolist()]
    for key in ("frag_n_atoms", "frag_atom_ptr", "frag_atom_feat", "frag_internal_degree",
                "frag_port_local", "frag_n_bonds", "frag_bond_ptr", "frag_bond_u",
                "frag_bond_v", "frag_bond_feat"):
        payload[key] = np.asarray(payload[key])
    return payload


def fragment_counts_train(payload: Mapping[str, Any]) -> dict[str, int]:
    """Distinct Train molecules per exact fragment key and per split role."""
    counts: dict[str, int] = {}
    n_mol = int(payload["train_mol_row_id"].shape[0])
    ptr = np.asarray(payload["train_mol_view_ptr"])
    s_ids = np.asarray(payload["train_view_s"])
    c_ids = np.asarray(payload["train_view_c"])
    keys = payload["frag_key"]
    for mol in range(n_mol):
        seen: set[str] = set()
        for view in range(int(ptr[mol]), int(ptr[mol + 1])):
            seen.add(keys[int(s_ids[view])])
            seen.add(keys[int(c_ids[view])])
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    return counts
