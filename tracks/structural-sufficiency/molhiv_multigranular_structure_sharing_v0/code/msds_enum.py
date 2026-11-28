"""Phase A core: exact connected induced subgraph enumeration + exact typed canonical keys.

Subgraph objects (PROTOCOL.md §4.1)
-----------------------------------
* ``size-2``/``size-3``/``size-4``: **every** connected induced subgraph with 2/3/4
  nodes of the molecule (not a tokenisation, not a partition).
* ``cycle-5``/``cycle-6``: every **chordless (induced) cycle** with 5/6 nodes.  This
  is a restricted family: it is *not* all connected induced 5-/6-node subgraphs.

Each instance is the molecule's real node set plus every internal bond.

Canonicalisation
----------------
A type is the isomorphism class of the *attributed* graph (node attributes + bond
attributes + the real internal adjacency) under node relabelling.  The key is

    min over admissible labelings of ( node attributes in order,
                                        sorted internal bonds with bond attributes )

where the admissible labelings are all orderings consistent with a canonically
ordered colour partition produced by attribute-and-bond-aware 1-WL colour
refinement.  Because the serialised object *is* a fully labelled attribute graph,
the minimum is a complete invariant: two instances share a key iff they are
isomorphic as attributed graphs.  The colour partition is itself
isomorphism-invariant, so the candidate set of labelings is too.

No atom index, molecule id, label or model output is part of a type identity.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

import msds_common as C


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------


def adjacency(n_atoms: int, bond_u: np.ndarray, bond_v: np.ndarray) -> list[list[int]]:
    nbr: list[list[int]] = [[] for _ in range(n_atoms)]
    for u, v in zip(bond_u.tolist(), bond_v.tolist()):
        nbr[u].append(v)
        nbr[v].append(u)
    return nbr


def _connected_subsets(nbr: Sequence[Sequence[int]], nodes: Sequence[int], k: int, root: int) -> Iterator[tuple[int, ...]]:
    """ESU-style connected connected-subset enumeration rooted at the minimum node."""
    ext = [u for u in nbr[root] if u > root]
    yield from _esu(nbr, (root,), ext, k, root)


def _esu(
    nbr: Sequence[Sequence[int]], current: tuple[int, ...], ext: list[int], k: int, root: int
) -> Iterator[tuple[int, ...]]:
    if len(current) == k:
        yield current
        return
    remaining = list(ext)
    while remaining:
        u = remaining.pop()
        extra = [
            w
            for w in nbr[u]
            if w > root and w not in current and w not in remaining
        ]
        yield from _esu(nbr, current + (u,), remaining + extra, k, root)


def enumerate_family_connected(
    nbr: Sequence[Sequence[int]], n_atoms: int, k: int
) -> list[tuple[int, ...]]:
    """All connected induced k-node subsets, deduplicated by node set."""
    seen: dict[frozenset[int], tuple[int, ...]] = {}
    for root in range(n_atoms):
        for sub in _connected_subsets(nbr, (), k, root):
            fs = frozenset(sub)
            if fs not in seen:
                seen[fs] = sub
    return list(seen.values())


def enumerate_induced_cycles(
    nbr: Sequence[Sequence[int]], n_atoms: int, k: int
) -> list[tuple[int, ...]]:
    """All chordless cycles of length ``k`` (5 or 6), deduplicated by node set."""
    adj_set = [set(n) for n in nbr]
    found: dict[frozenset[int], tuple[int, ...]] = {}

    def dfs(path: list[int], cur: int) -> None:
        depth = len(path)
        if depth == k:
            if root in adj_set[cur]:
                fs = frozenset(path)
                if fs not in found:
                    found[fs] = tuple(path)
            return
        closing = depth == k - 1
        # chordless: a new vertex may only touch the current endpoint.  The last
        # vertex of a cycle must of course touch the root, so the root is exempt
        # for the closing step only.
        checked = path[1:-1] if closing else path[:-1]
        for nxt in nbr[cur]:
            if nxt <= root or nxt in path:
                continue
            if any(nxt in adj_set[p] for p in checked):
                continue
            if closing and root not in adj_set[nxt]:
                continue
            path.append(nxt)
            dfs(path, nxt)
            path.pop()

    for root in range(n_atoms):
        dfs([root], root)
    return list(found.values())


def instances_for_molecule(
    n_atoms: int, bond_u: np.ndarray, bond_v: np.ndarray
) -> list[tuple[int, tuple[int, ...]]]:
    """All Phase A instances of one molecule, as ``(family_index, node_tuple)``."""
    nbr = adjacency(n_atoms, bond_u, bond_v)
    out: list[tuple[int, tuple[int, ...]]] = []
    for fam_idx, family in enumerate(C.FAMILIES):
        k = C.FAMILY_SIZE[family]
        if family.startswith("size"):
            subs = enumerate_family_connected(nbr, n_atoms, k)
        else:
            subs = enumerate_induced_cycles(nbr, n_atoms, k)
        out.extend((fam_idx, tuple(sorted(s))) for s in subs)
    return out


# ---------------------------------------------------------------------------
# canonicalisation
# ---------------------------------------------------------------------------


def _rank(values: Sequence[Any]) -> list[int]:
    uniq = sorted(set(values))
    index = {v: i for i, v in enumerate(uniq)}
    return [index[v] for v in values]


def _refine(attrs: Sequence[tuple], neighbours: Sequence[Sequence[tuple[int, tuple]]]) -> list[int]:
    colors = _rank(attrs)
    while True:
        sigs = [
            (colors[i], tuple(sorted((colors[j], battr) for j, battr in neighbours[i])))
            for i in range(len(attrs))
        ]
        new = _rank(sigs)
        if new == colors:
            return colors
        colors = new


def canonical_key(
    nodes: Sequence[int],
    atom_feat: np.ndarray,
    bond_attr: dict[tuple[int, int], tuple],
    atom_columns: Sequence[int],
    bond_columns: Sequence[int],
) -> tuple:
    """Exact canonical key of the attributed induced subgraph (complete invariant)."""
    k = len(nodes)
    local = {v: i for i, v in enumerate(nodes)}
    attrs = [tuple(int(atom_feat[v][c]) for c in atom_columns) for v in nodes]
    adj_pairs: list[tuple[int, int, tuple]] = []
    for (a, b), battr in bond_attr.items():
        i, j = local[a], local[b]
        adj_pairs.append((min(i, j), max(i, j), tuple(int(battr[c]) for c in bond_columns)))
    neighbours: list[list[tuple[int, tuple]]] = [[] for _ in range(k)]
    for i, j, battr in adj_pairs:
        neighbours[i].append((j, battr))
        neighbours[j].append((i, battr))
    colors = _refine(attrs, neighbours)

    n_colors = max(colors) + 1
    cells: list[list[int]] = [[] for _ in range(n_colors)]
    for i, c in enumerate(colors):
        cells[c].append(i)
    cells = [c for c in cells if c]

    best: tuple | None = None
    perms = [list(itertools.permutations(cell)) for cell in cells]
    for combo in itertools.product(*perms):
        order = [v for group in combo for v in group]
        pos = {v: p for p, v in enumerate(order)}
        node_part = tuple(attrs[v] for v in order)
        edge_part = tuple(
            sorted((min(pos[i], pos[j]), max(pos[i], pos[j]), battr) for i, j, battr in adj_pairs)
        )
        candidate = (node_part, edge_part)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def key_to_string(key: tuple) -> str:
    """Compact stable text form of a canonical key (for registries / JSON)."""
    node_part, edge_part = key
    node_text = ";".join(",".join(str(x) for x in node) for node in node_part)
    edge_text = ";".join(f"{i}-{j}:{','.join(str(x) for x in e)}" for i, j, e in edge_part)
    return f"N[{node_text}]E[{edge_text}]"


def instance_key(
    molecules: Any,
    graph_id: int,
    nodes: Sequence[int],
    *,
    granularity: str,
    adjacency_map: dict[int, dict[int, tuple]] | None = None,
) -> tuple:
    """Canonical key of one instance at ``COARSE`` or ``FULL`` granularity."""
    atom_feat = molecules.atom_feat[int(molecules.atom_offsets[graph_id]) : int(molecules.atom_offsets[graph_id + 1])]
    if adjacency_map is None:
        raise ValueError("adjacency_map is required")
    bond_attr = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if b in adjacency_map.get(a, {}):
                bond_attr[(a, b)] = adjacency_map[a][b]
    if granularity == "COARSE":
        atom_columns = C.COARSE_ATOM_COLUMNS
        bond_columns = C.COARSE_BOND_COLUMNS
    elif granularity == "FULL":
        atom_columns = tuple(range(atom_feat.shape[1]))
        bond_columns = tuple(range(3))
    else:
        raise ValueError(granularity)
    return canonical_key(nodes, atom_feat, bond_attr, atom_columns, bond_columns)


# ---------------------------------------------------------------------------
# independent brute-force reference (used by the tests, deliberately naive)
# ---------------------------------------------------------------------------


def brute_force_instances(n_atoms: int, bonds: Sequence[tuple[int, int]]) -> list[tuple[int, tuple[int, ...]]]:
    """Every k-subset, keeping only connected ones; cycles additionally chordless."""
    edges = {tuple(sorted(b)) for b in bonds}
    out: list[tuple[int, tuple[int, ...]]] = []
    for fam_idx, family in enumerate(C.FAMILIES):
        k = C.FAMILY_SIZE[family]
        for sub in itertools.combinations(range(n_atoms), k):
            s = set(sub)
            internal = [e for e in edges if e[0] in s and e[1] in s]
            if len(internal) < k - 1:
                continue
            # connectivity by BFS over internal edges
            seen = {sub[0]}
            frontier = [sub[0]]
            while frontier:
                x = frontier.pop()
                for a, b in internal:
                    if a == x and b not in seen:
                        seen.add(b)
                        frontier.append(b)
                    elif b == x and a not in seen:
                        seen.add(a)
                        frontier.append(a)
            if len(seen) != k:
                continue
            if family.startswith("cycle"):
                if len(internal) != k:
                    continue
                deg = {v: 0 for v in sub}
                for a, b in internal:
                    deg[a] += 1
                    deg[b] += 1
                if any(d != 2 for d in deg.values()):
                    continue
            out.append((fam_idx, tuple(sub)))
    return out


def brute_force_key(
    nodes: Sequence[int],
    atom_feat: np.ndarray,
    bond_attr: dict[tuple[int, int], tuple],
    atom_columns: Sequence[int],
    bond_columns: Sequence[int],
) -> tuple:
    """Minimum over *all* node permutations (no WL cells); reference for tests."""
    k = len(nodes)
    local = {v: i for i, v in enumerate(nodes)}
    attrs = {i: tuple(int(atom_feat[v][c]) for c in atom_columns) for i, v in enumerate(nodes)}
    edges = {}
    for (a, b), battr in bond_attr.items():
        i, j = local[a], local[b]
        edges[(min(i, j), max(i, j))] = tuple(int(battr[c]) for c in bond_columns)
    best: tuple | None = None
    for perm in itertools.permutations(range(k)):
        pos = {v: p for p, v in enumerate(perm)}
        node_part = tuple(attrs[v] for v in perm)
        edge_part = tuple(
            sorted((min(pos[i], pos[j]), max(pos[i], pos[j]), battr) for (i, j), battr in edges.items())
        )
        cand = (node_part, edge_part)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# readable depictions (registries / report tables)
# ---------------------------------------------------------------------------

_ELEMENTS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
    20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
    29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb",
    38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd",
    47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
    56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta",
    74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb",
    83: "Bi", 84: "Po", 85: "At", 86: "Rn", 92: "U",
}
_BOND = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROM"}
_HYBRID = {0: "?", 1: "s", 2: "sp", 3: "sp2", 4: "sp3", 5: "sp3d", 6: "sp3d2"}


def atom_label(attrs: Sequence[int], granularity: str) -> str:
    if granularity == "COARSE":
        z, charge, aromatic = (int(a) for a in attrs)
        sym = _ELEMENTS.get(z, f"Z{z}")
        text = sym + ("ar" if aromatic else "")
        if charge:
            text += f"{charge:+d}"
        return text
    z, chir, deg, charge, nh, rad, hyb, arom, ring = (int(a) for a in attrs)
    sym = _ELEMENTS.get(z, f"Z{z}")
    parts = [sym]
    if arom:
        parts.append("ar")
    if ring:
        parts.append("R")
    if hyb:
        parts.append(_HYBRID.get(hyb, f"h{hyb}"))
    if nh:
        parts.append(f"H{nh}")
    if charge:
        parts.append(f"{charge:+d}")
    if chir:
        parts.append(f"chi{chir}")
    if rad:
        parts.append(f"rad{rad}")
    parts.append(f"deg{deg}")
    return "".join(parts)


def bond_label(attrs: Sequence[int], granularity: str) -> str:
    if granularity == "COARSE":
        return _BOND.get(int(attrs[0]), f"B{int(attrs[0])}")
    bt, stereo, conj = (int(a) for a in attrs)
    text = _BOND.get(bt, f"B{bt}")
    if stereo:
        text += f"/st{stereo}"
    if conj:
        text += "/cj"
    return text


def describe_key(key: tuple, granularity: str) -> dict:
    """Collapse the canonical key into a readable class-level structural formula."""
    node_part, edge_part = key
    labels = [atom_label(n, granularity) for n in node_part]
    class_of: dict[str, str] = {}
    for lab in labels:
        if lab not in class_of:
            class_of[lab] = chr(ord("a") + len(class_of))
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    node_sig = " ".join(f"{class_of[lab]}{lab}x{n}" for lab, n in sorted(counts.items()))
    bond_counts: dict[tuple[str, str, str], int] = {}
    for i, j, battr in edge_part:
        a, b = class_of[labels[i]], class_of[labels[j]]
        bl = bond_label(battr, granularity)
        key2 = (min(a, b), max(a, b), bl)
        bond_counts[key2] = bond_counts.get(key2, 0) + 1
    edge_sig = " ".join(
        f"{a}-{b}:{bl}x{n}" for (a, b, bl), n in sorted(bond_counts.items())
    )
    return {
        "n_nodes": len(labels),
        "n_bonds": len(edge_part),
        "node_signature": node_sig,
        "bond_signature": edge_sig,
        "canonical_key": key_to_string(key),
    }
