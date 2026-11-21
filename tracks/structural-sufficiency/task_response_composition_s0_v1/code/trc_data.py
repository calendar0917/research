"""Raw graph access, exact integer cuts, rooted 1-4 objects and exact keys.

Fixed by the brief, not searched:

* **object** = ``(G, root, atom_set)`` with ``1 <= |atom_set| <= 4``, root in the
  set and ``G[atom_set]`` connected.  All are enumerated; they may overlap.
* **cut** = a bridge bond ``b = (u, v)`` of the original graph.  Removing ``b``
  splits the molecule into the two sides; each side with ``1 <= |S| <= 8`` is a
  parse, the other side being the environment ``C``.
* **fallback** = a molecule with no valid bridge parse contributes one parse per
  atom ``p``: ``S = whole G``, ``C = empty``, ``p_S = p`` (independent null cut).

Exact keys are canonical SMILES of a colour-encoded auxiliary graph (RDKit);
the colouring uses disjoint isotope ranges for ``(ATOM, 9-tuple)``,
``(BOND, 3-tuple)`` and the ``PORT`` marker.  No sanitisation to chemistry.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import trc_common as C

# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------


@dataclass
class Molecule:
    row_id: int
    n_atoms: int
    atom_feat: np.ndarray      # [n, 9] uint8
    bond_u: np.ndarray         # [e] int32, u < v
    bond_v: np.ndarray         # [e] int32
    bond_feat: np.ndarray      # [e, 3] uint8

    def adjacency_lists(self) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(self.n_atoms)]
        for a, b in zip(self.bond_u.tolist(), self.bond_v.tolist()):
            adj[a].append(b)
            adj[b].append(a)
        return adj

    def bond_index(self) -> dict[tuple[int, int], int]:
        out: dict[tuple[int, int], int] = {}
        for k, (a, b) in enumerate(zip(self.bond_u.tolist(), self.bond_v.tolist())):
            out[(a, b)] = k
            out[(b, a)] = k
        return out

    def permute(self, perm: Sequence[int]) -> "Molecule":
        perm_arr = np.asarray(perm, dtype=np.int64)
        new_atom = np.zeros_like(self.atom_feat)
        new_atom[perm_arr] = self.atom_feat
        nu = perm_arr[self.bond_u.astype(np.int64)]
        nv = perm_arr[self.bond_v.astype(np.int64)]
        lo = np.minimum(nu, nv).astype(np.int32)
        hi = np.maximum(nu, nv).astype(np.int32)
        order = np.lexsort((hi, lo))
        return Molecule(
            row_id=self.row_id,
            n_atoms=self.n_atoms,
            atom_feat=new_atom,
            bond_u=lo[order],
            bond_v=hi[order],
            bond_feat=self.bond_feat[order],
        )


def load_raw_cache(cfg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    key = "raw_cache" if dataset == "lipo" else "raw_cache_esol"
    path = C.REPO_ROOT / str(cfg[key])
    if not path.is_file():
        raise FileNotFoundError(f"raw label-free cache missing: {path}")
    digest = C.sha256_file(path)
    expected = str(cfg["raw_cache_sha256"] if dataset == "lipo" else cfg["raw_cache_esol_sha256"])
    if digest != expected:
        raise ValueError(f"raw cache sha256 mismatch: {digest} != {expected}")
    prefix = "" if dataset == "lipo" else "esol_"
    with np.load(path, allow_pickle=False) as handle:
        payload = {key2: handle[key2] for key2 in handle.files}
    out: dict[str, Any] = {}
    for name in ("node_ptr", "bond_ptr", "atom_feat", "bond_u", "bond_v", "bond_feat",
                 "labels", "row_ids", "train_idx", "valid_idx", "test_idx",
                 "graph_n_components"):
        out[name] = np.asarray(payload[prefix + name])
    out["cache_sha256"] = digest
    out["cache_path"] = str(path)
    out["dataset"] = dataset
    return out


def raw_molecule(raw: Mapping[str, Any], row: int) -> Molecule:
    node_ptr = np.asarray(raw["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], dtype=np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    return Molecule(
        row_id=int(np.asarray(raw["row_ids"])[row]),
        n_atoms=int(a1 - a0),
        atom_feat=np.asarray(raw["atom_feat"])[a0:a1].astype(np.uint8, copy=True),
        bond_u=np.asarray(raw["bond_u"])[b0:b1].astype(np.int32, copy=True),
        bond_v=np.asarray(raw["bond_v"])[b0:b1].astype(np.int32, copy=True),
        bond_feat=np.asarray(raw["bond_feat"])[b0:b1].astype(np.uint8, copy=True),
    )


# ---------------------------------------------------------------------------
# bridges / components
# ---------------------------------------------------------------------------


def connected_components(adj: Sequence[Sequence[int]], n_atoms: int,
                         banned: tuple[int, int] | None = None) -> list[int]:
    comp = np.full(int(n_atoms), -1, np.int32)
    label = 0
    for start in range(int(n_atoms)):
        if comp[start] >= 0:
            continue
        stack = [start]
        comp[start] = label
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if banned is not None and ((u == banned[0] and w == banned[1])
                                           or (u == banned[1] and w == banned[0])):
                    continue
                if comp[w] < 0:
                    comp[w] = label
                    stack.append(w)
        label += 1
    return comp.tolist()


def bridge_bonds(mol: Molecule, adj: Sequence[Sequence[int]]) -> list[int]:
    """Bond indices whose removal strictly increases the component count."""
    base = len(set(connected_components(adj, mol.n_atoms)))
    out: list[int] = []
    for e, (a, b) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
        comp = connected_components(adj, mol.n_atoms, banned=(int(a), int(b)))
        if len(set(comp)) > base:
            out.append(e)
    return out


def distance_within(n_atoms: int, adj: Sequence[Sequence[int]], atoms: Sequence[int],
                    source: int) -> dict[int, int]:
    """BFS restricted to ``atoms`` (the fragment interior)."""
    allowed = set(int(a) for a in atoms)
    dist = {int(source): 0}
    frontier = [int(source)]
    while frontier:
        nxt: list[int] = []
        for u in frontier:
            for w in adj[u]:
                if w in allowed and w not in dist:
                    dist[w] = dist[u] + 1
                    nxt.append(w)
        frontier = nxt
    return dist


def distance_bucket(d: int | None) -> int:
    if d is None:
        return 5
    return int(d) if d <= 3 else 4


# ---------------------------------------------------------------------------
# cuts
# ---------------------------------------------------------------------------


@dataclass
class CutParse:
    """One directed parse of a molecule."""

    atoms_s: tuple[int, ...]
    atoms_c: tuple[int, ...]
    port_s: int
    port_c: int | None
    bond_index: int | None       # None for the null/fallback cut


def cut_parses(mol: Molecule, adj: Sequence[Sequence[int]], max_side: int = 8) -> tuple[list[CutParse], int]:
    """Every directed valid bridge parse; plus the number of bridge bonds seen."""
    parses: list[CutParse] = []
    bridges = bridge_bonds(mol, adj)
    for e in bridges:
        u = int(mol.bond_u[e])
        v = int(mol.bond_v[e])
        comp = connected_components(adj, mol.n_atoms, banned=(u, v))
        side_u = tuple(i for i in range(mol.n_atoms) if comp[i] == comp[u])
        side_v = tuple(i for i in range(mol.n_atoms) if comp[i] == comp[v])
        for atoms_s, port_s, atoms_c, port_c in (
            (side_u, u, side_v, v),
            (side_v, v, side_u, u),
        ):
            if not (1 <= len(atoms_s) <= max_side):
                continue
            if len(atoms_c) == 0:
                continue
            parses.append(CutParse(atoms_s, atoms_c, port_s, port_c, e))
    return parses, len(bridges)


def fallback_parses(mol: Molecule) -> list[CutParse]:
    whole = tuple(range(mol.n_atoms))
    return [CutParse(whole, tuple(), int(p), None, None) for p in range(mol.n_atoms)]


# ---------------------------------------------------------------------------
# rooted 1-4 objects
# ---------------------------------------------------------------------------


def rooted_subsets(adj: Sequence[Sequence[int]], n_atoms: int, max_k: int = 4,
                   allowed: Sequence[int] | None = None) -> list[tuple[int, tuple[int, ...]]]:
    """Every ``(root, atoms)`` with ``|atoms| <= max_k`` and connected induced graph."""
    verts = list(range(int(n_atoms))) if allowed is None else sorted(int(a) for a in allowed)
    vset = set(verts)
    out: list[tuple[int, tuple[int, ...]]] = []
    for root in verts:
        frontier = [frozenset((root,))]
        out.append((root, (root,)))
        for _size in range(2, max_k + 1):
            nxt: set[frozenset[int]] = set()
            for s in frontier:
                for u in s:
                    for w in adj[u]:
                        if w in vset and w not in s:
                            nxt.add(s | {w})
            frontier = list(nxt)
            for s in frontier:
                out.append((root, tuple(sorted(s))))
    return out


def _pairs(k: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(k), 2))


def _permuted_bits(masks: np.ndarray, perm: Sequence[int], pairs: Sequence[tuple[int, int]],
                   pair_index: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.zeros(masks.shape, dtype=np.uint32)
    for e_slot, (i, j) in enumerate(pairs):
        a, b = sorted((int(perm[i]), int(perm[j])))
        out |= ((masks >> pair_index[(a, b)]) & 1) << e_slot
    return out


def build_lut(k: int, max_k: int = C.MAX_OBJECT_ATOMS) -> dict[str, np.ndarray]:
    """Canonical adjacency bit + every winning root-fixed permutation for ``k``."""
    pairs = _pairs(k)
    n_pairs = len(pairs)
    n_masks = 1 << n_pairs
    masks = np.arange(n_masks, dtype=np.uint32)
    perms = [(0,) + p for p in itertools.permutations(range(1, k))]
    pair_index = {p: i for i, p in enumerate(pairs)}
    cand = np.empty((n_masks, len(perms)), dtype=np.uint16)
    for c, perm in enumerate(perms):
        cand[:, c] = _permuted_bits(masks, perm, pairs, pair_index).astype(np.uint16)
    canon = cand.min(axis=1).astype(np.uint16)
    hit = cand == canon[:, None]
    rows, cols = np.nonzero(hit)
    ptr = np.searchsorted(rows, np.arange(n_masks + 1, dtype=np.int64)).astype(np.int64)
    perm_table = np.asarray(perms, dtype=np.uint8)
    winner = perm_table[cols] if cols.size else np.zeros((0, k), np.uint8)
    return {
        "k": np.asarray(k, np.uint8),
        "canon": canon,
        "count": np.diff(ptr).astype(np.int32),
        "ptr": ptr,
        "winner_perm": winner,
    }


def build_luts(max_k: int = C.MAX_OBJECT_ATOMS) -> dict[int, dict[str, np.ndarray]]:
    return {k: build_lut(k, max_k) for k in range(1, max_k + 1)}


@dataclass
class FrameSet:
    """Frames of a list of rooted objects, in the 4-slot / 6-pair layout."""

    obj_frame_ptr: np.ndarray     # [n_obj+1]
    atom_rows: np.ndarray         # [n_frames, 4] int32 -- global attribute rows
    bond_rows: np.ndarray         # [n_frames, 6] int32 -- global attribute rows or -1
    frame_k: np.ndarray           # [n_frames] uint8


def frames_for_objects(
    objects: Sequence[tuple[int, tuple[int, ...]]],
    atom_row: np.ndarray,
    bond_row: np.ndarray,
    luts: Mapping[int, Mapping[str, np.ndarray]],
    max_k: int = C.MAX_OBJECT_ATOMS,
) -> FrameSet:
    """Rooted frames with the root pinned at slot 0, all winning permutations kept."""
    n_obj = len(objects)
    sizes = np.fromiter((len(s) for _, s in objects), np.int64, count=n_obj)
    frame_counts = np.zeros(n_obj, dtype=np.int64)
    atom_parts: list[np.ndarray] = []
    bond_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []
    pair_slot = {p: i for i, p in enumerate(C.FRAME_PAIRS)}
    start = 0
    while start < n_obj:
        k = int(sizes[start])
        stop = start
        while stop < n_obj and int(sizes[stop]) == k:
            stop += 1
        group = objects[start:stop]
        n = len(group)
        L = np.empty((n, k), dtype=np.int64)
        for i, (root, atoms) in enumerate(group):
            if int(root) not in atoms:
                raise ValueError("root not in object")
            others = [int(a) for a in atoms if int(a) != int(root)]
            if len(others) != k - 1:
                raise ValueError("object set does not match its size")
            L[i, 0] = int(root)
            L[i, 1:] = others
        pairs = _pairs(k)
        adj = bond_row >= 0
        bits = np.zeros(n, dtype=np.uint32)
        for e, (i, j) in enumerate(pairs):
            bits |= adj[L[:, i], L[:, j]].astype(np.uint32) << e
        lut = luts[k]
        canon = lut["canon"][bits]
        counts = lut["count"][bits]
        total = int(counts.sum())
        obj_pos = np.repeat(np.arange(n, dtype=np.int64), counts)
        starts = np.cumsum(counts, dtype=np.int64) - counts
        local = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
        offsets = np.repeat(lut["ptr"][bits], counts) + local
        perm = lut["winner_perm"][offsets]
        slots = L[obj_pos[:, None], perm.astype(np.int64)]
        n_frames = slots.shape[0]
        a_attr = np.zeros((n_frames, max_k), dtype=np.int32)
        a_attr[:, :k] = atom_row[slots]
        b_attr = np.full((n_frames, len(C.FRAME_PAIRS)), -1, dtype=np.int32)
        for e, (i, j) in enumerate(pairs):
            present = ((canon[obj_pos] >> e) & 1).astype(bool)
            vals = bond_row[slots[:, i], slots[:, j]]
            b_attr[:, pair_slot[(i, j)]] = np.where(present, vals, -1)
        atom_parts.append(a_attr)
        bond_parts.append(b_attr)
        k_parts.append(np.full(n_frames, k, dtype=np.uint8))
        frame_counts[start:stop] = counts
        if int(counts.sum()) != n_frames:
            raise AssertionError("frame expansion lost frames")
        start = stop
    if atom_parts:
        atom_rows = np.concatenate(atom_parts, axis=0)
        bond_rows = np.concatenate(bond_parts, axis=0)
        frame_k = np.concatenate(k_parts, axis=0)
    else:
        atom_rows = np.zeros((0, max_k), np.int32)
        bond_rows = np.zeros((0, len(C.FRAME_PAIRS)), np.int32)
        frame_k = np.zeros(0, np.uint8)
    return FrameSet(
        obj_frame_ptr=np.concatenate([[0], np.cumsum(frame_counts)]).astype(np.int64),
        atom_rows=atom_rows,
        bond_rows=bond_rows,
        frame_k=frame_k,
    )


# ---------------------------------------------------------------------------
# exact coloured-graph keys
# ---------------------------------------------------------------------------

PORT_ISOTOPE = 3001
ATOM_ISOTOPE_BASE = 1
ATOM_ISOTOPE_LIMIT = 999
BOND_ISOTOPE_BASE = 1001
BOND_ISOTOPE_LIMIT = 1999


class ColorSpace:
    """Global, label-free colour assignment over disjoint isotope ranges."""

    def __init__(self) -> None:
        self.atom_color: dict[bytes, int] = {}
        self.bond_color: dict[bytes, int] = {}

    def atom(self, row: np.ndarray | Sequence[int]) -> int:
        key = np.asarray(row, dtype=np.uint8).tobytes()
        got = self.atom_color.get(key)
        if got is None:
            got = ATOM_ISOTOPE_BASE + len(self.atom_color)
            if got > ATOM_ISOTOPE_LIMIT:
                raise ValueError("atom colour range exhausted")
            self.atom_color[key] = got
        return got

    def bond(self, row: np.ndarray | Sequence[int]) -> int:
        key = np.asarray(row, dtype=np.uint8).tobytes()
        got = self.bond_color.get(key)
        if got is None:
            got = BOND_ISOTOPE_BASE + len(self.bond_color)
            if got > BOND_ISOTOPE_LIMIT:
                raise ValueError("bond colour range exhausted")
            self.bond_color[key] = got
        return got

    def atom_table(self) -> np.ndarray:
        return (np.frombuffer(b"".join(self.atom_color), np.uint8)
                .reshape(-1, C.ATOM_COLUMNS).copy())

    def bond_table(self) -> np.ndarray:
        return (np.frombuffer(b"".join(self.bond_color), np.uint8)
                .reshape(-1, C.BOND_COLUMNS).copy())


def _dummy(isotope: int):
    from rdkit import Chem

    atom = Chem.Atom(0)
    atom.SetIsotope(int(isotope))
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    return atom


def canonical_key(
    atom_feats: Sequence[Sequence[int]],
    local_bonds: Sequence[tuple[int, int, Sequence[int]]],
    port: int,
    colors: ColorSpace,
) -> str:
    """Exact coloured-graph key of a fragment with its port.

    ``atom_feats`` are raw 9-tuples in local (sorted) order; ``local_bonds`` are
    ``(i, j, raw 3-tuple)`` with ``i < j`` local indices.  Only graph encoding:
    the auxiliary graph is never sanitised into chemistry.
    """
    from rdkit import Chem

    rw = Chem.RWMol()
    n = len(atom_feats)
    atom_idx = [rw.AddAtom(_dummy(colors.atom(row))) for row in atom_feats]
    for (i, j, br) in local_bonds:
        bidx = rw.AddAtom(_dummy(colors.bond(br)))
        rw.AddBond(atom_idx[int(i)], bidx, Chem.BondType.SINGLE)
        rw.AddBond(bidx, atom_idx[int(j)], Chem.BondType.SINGLE)
    pidx = rw.AddAtom(_dummy(PORT_ISOTOPE))
    rw.AddBond(atom_idx[int(port)], pidx, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
