"""Mid-range atom-pair preprocessing over the frozen LCRC cache.

For every molecule the module enumerates unordered atom pairs ``(u, v)`` with
``u < v`` whose *unweighted shortest-path graph distance* over the LCRC heavy
atom graph is exactly 3, 4 or 5.  Distances come only from the frozen cache's
``atom_ptr`` / ``bond_ptr`` / ``bond_index`` arrays (BFS on the bond list); no
label, no RDKit call and no extra chemistry descriptor is involved.

Two distance semantics are supported:

``true``
    the bin of a pair is its real shortest-path distance (3, 4 or 5);

``null``
    every molecule applies its *own* fixed permutation of the three class
    names, derived from ``sha256(molecule_id + "|" + salt)``.  The pair set,
    the atom features and the graph structure are unchanged: only the
    distance-class label is shuffled per molecule, which destroys consistency
    of the class semantics across molecules while preserving within-molecule
    grouping.

The pair arrays are stored per molecule with an explicit pointer
(``pair_ptr``) so that :func:`pair_collate` can offset them into a batch in
exactly the same way as :func:`v2_data.collate` offsets bonds and paths.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pilot_common as C

#: The six permutations of the three distance classes (as bin indices 0/1/2).
PERMUTATIONS = tuple(itertools.permutations((0, 1, 2)))


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def bfs_distances(n_atoms: int, bonds: np.ndarray) -> np.ndarray:
    """Unweighted all-pairs shortest-path matrix; ``-1`` for unreachable pairs."""
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for u, v in np.asarray(bonds, dtype=np.int64).reshape(-1, 2):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    dist = np.full((int(n_atoms), int(n_atoms)), -1, dtype=np.int64)
    for source in range(int(n_atoms)):
        dist[source, source] = 0
        queue = deque([source])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if dist[source, nxt] < 0:
                    dist[source, nxt] = dist[source, cur] + 1
                    queue.append(nxt)
    return dist


def molecule_pairs(n_atoms: int, bonds: np.ndarray, distances: Sequence[int] = C.PAIR_DISTANCES):
    """All ``u < v`` pairs whose graph distance is in ``distances``."""
    wanted = {int(d): i for i, d in enumerate(distances)}
    dist = bfs_distances(n_atoms, bonds)
    us: list[int] = []
    vs: list[int] = []
    bins: list[int] = []
    for u in range(int(n_atoms)):
        for v in range(u + 1, int(n_atoms)):
            d = int(dist[u, v])
            if d in wanted:
                us.append(u)
                vs.append(v)
                bins.append(wanted[d])
    return np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64), np.asarray(bins, dtype=np.int64)


# ---------------------------------------------------------------------------
# per-molecule null permutations
# ---------------------------------------------------------------------------


def _null_seed(molecule_id: str, salt: str = C.PAIR_NULL_SALT) -> int:
    digest = hashlib.sha256(f"{molecule_id}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def null_permutation(molecule_id: str, salt: str = C.PAIR_NULL_SALT) -> tuple[int, int, int]:
    """Deterministic molecule-specific permutation of the three class bins."""
    rng = random.Random(_null_seed(str(molecule_id), salt))
    return tuple(int(x) for x in rng.choice(PERMUTATIONS))


def null_permutation_table(molecule_ids: Sequence[str], salt: str = C.PAIR_NULL_SALT) -> np.ndarray:
    """``(n_molecules, 3)`` table mapping original bin -> replacement bin."""
    table = np.zeros((len(molecule_ids), 3), dtype=np.int64)
    for i, molecule_id in enumerate(molecule_ids):
        table[i] = np.asarray(null_permutation(str(molecule_id), salt), dtype=np.int64)
    return table


# ---------------------------------------------------------------------------
# pair cache
# ---------------------------------------------------------------------------


def build_pair_cache(cache: Mapping[str, Any], *, semantics: str) -> dict[str, Any]:
    """Precompute the distance-3/4/5 pair lists for every molecule in ``cache``."""
    if semantics not in ("true", "null"):
        raise ValueError("semantics must be 'true' or 'null'")
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    n_molecules = int(cache["n_molecules"])

    pointers = [0]
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    bins: list[np.ndarray] = []
    true_counts = np.zeros((n_molecules, len(C.PAIR_DISTANCES)), dtype=np.int64)
    for m in range(n_molecules):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        b0, b1 = int(bond_ptr[m]), int(bond_ptr[m + 1])
        u, v, b = molecule_pairs(a1 - a0, bond_index[b0:b1])
        us.append(u)
        vs.append(v)
        bins.append(b)
        true_counts[m] = np.bincount(b, minlength=len(C.PAIR_DISTANCES)) if b.size else 0
        pointers.append(pointers[-1] + int(u.size))

    pair_u = np.concatenate(us) if us else np.zeros(0, dtype=np.int64)
    pair_v = np.concatenate(vs) if vs else np.zeros(0, dtype=np.int64)
    pair_bin = np.concatenate(bins) if bins else np.zeros(0, dtype=np.int64)
    pair_graph = np.repeat(np.arange(n_molecules, dtype=np.int64), np.diff(np.asarray(pointers, dtype=np.int64)))

    perm_table = np.zeros((n_molecules, 3), dtype=np.int64)
    if semantics == "null":
        perm_table = null_permutation_table([str(x) for x in cache["molecule_ids"]])
        pair_bin = perm_table[pair_graph, pair_bin]

    out = {
        "semantics": semantics,
        "n_molecules": n_molecules,
        "pair_ptr": np.asarray(pointers, dtype=np.int64),
        "pair_u": pair_u,
        "pair_v": pair_v,
        "pair_bin": pair_bin,
        "pair_graph": pair_graph,
        "true_bin_counts": true_counts,
        "null_permutations": perm_table,
        "distances": list(C.PAIR_DISTANCES),
        "salt": C.PAIR_NULL_SALT if semantics == "null" else None,
        "n_pairs_total": int(pair_u.size),
    }
    validate_pair_cache(cache, out)
    return out


def validate_pair_cache(cache: Mapping[str, Any], pairs: Mapping[str, Any]) -> None:
    """Structural invariants of a pair cache (label blind)."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    n_molecules = int(cache["n_molecules"])
    ptr = np.asarray(pairs["pair_ptr"], dtype=np.int64)
    if ptr.shape != (n_molecules + 1,) or int(ptr[0]) != 0 or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("pair_ptr must be a monotone (n_molecules + 1) pointer")
    u = np.asarray(pairs["pair_u"], dtype=np.int64)
    v = np.asarray(pairs["pair_v"], dtype=np.int64)
    g = np.asarray(pairs["pair_graph"], dtype=np.int64)
    b = np.asarray(pairs["pair_bin"], dtype=np.int64)
    n = int(ptr[-1])
    if not (u.shape == v.shape == g.shape == b.shape == (n,)):
        raise ValueError("pair arrays must have equal length")
    if np.any(u >= v):
        raise ValueError("pairs must satisfy u < v")
    if np.any(b < 0) or np.any(b >= len(C.PAIR_DISTANCES)):
        raise ValueError("pair bin outside the distance vocabulary")
    if np.any(g < 0) or np.any(g >= n_molecules):
        raise ValueError("pair_graph outside the molecule range")
    expected_g = np.repeat(np.arange(n_molecules, dtype=np.int64), np.diff(ptr))
    if not np.array_equal(g, expected_g):
        raise ValueError("pair_graph is not consistent with pair_ptr")
    for m in range(n_molecules):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        sl = slice(int(ptr[m]), int(ptr[m + 1]))
        um, vm = u[sl], v[sl]
        if um.size and (um.min() < 0 or vm.max() >= a1 - a0):
            raise ValueError(f"molecule {m}: pair atom index outside the molecule")
        if um.size:
            keys = set(zip(um.tolist(), vm.tolist()))
            if len(keys) != int(um.size):
                raise ValueError(f"molecule {m}: duplicate atom pair")
    if pairs["semantics"] == "null":
        perm = np.asarray(pairs["null_permutations"], dtype=np.int64)
        if perm.shape != (n_molecules, 3):
            raise ValueError("null permutation table has the wrong shape")
        for m in range(n_molecules):
            if sorted(int(x) for x in perm[m]) != [0, 1, 2]:
                raise ValueError(f"molecule {m}: null permutation is not a permutation")


def same_pair_content(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two pair caches: identical (u, v) lists, permuted bins only."""
    same_ptr = np.array_equal(np.asarray(a["pair_ptr"]), np.asarray(b["pair_ptr"]))
    same_u = np.array_equal(np.asarray(a["pair_u"]), np.asarray(b["pair_u"]))
    same_v = np.array_equal(np.asarray(a["pair_v"]), np.asarray(b["pair_v"]))
    bin_true = np.asarray(a["pair_bin"], dtype=np.int64)
    bin_null = np.asarray(b["pair_bin"], dtype=np.int64)
    graph = np.asarray(a["pair_graph"], dtype=np.int64)
    perm = np.asarray(b.get("null_permutations"), dtype=np.int64)
    mapped = perm[graph, bin_true] if perm.size else bin_true
    ptr = np.asarray(a["pair_ptr"], dtype=np.int64)
    per_molecule_ok = True
    for m in range(int(ptr.size) - 1):
        sl = slice(int(ptr[m]), int(ptr[m + 1]))
        if sorted(np.bincount(bin_true[sl], minlength=3).tolist()) != sorted(
            np.bincount(bin_null[sl], minlength=3).tolist()
        ):
            per_molecule_ok = False
            break
    return {
        "pair_ptr_identical": bool(same_ptr),
        "pair_u_identical": bool(same_u),
        "pair_v_identical": bool(same_v),
        "bin_is_per_molecule_permutation": bool(np.array_equal(mapped, bin_null)),
        "per_molecule_bin_multiset_preserved": bool(per_molecule_ok),
        "n_pairs": int(np.asarray(a["pair_u"]).size),
        "n_pairs_identical": bool(np.asarray(a["pair_u"]).size == np.asarray(b["pair_u"]).size),
        "bin_counts_true": np.bincount(bin_true, minlength=3).tolist(),
        "bin_counts_null": np.bincount(bin_null, minlength=3).tolist(),
        "n_molecules_with_identity_permutation": int(
            sum(1 for row in perm.tolist() if row == [0, 1, 2])
        ),
        "n_distinct_permutations_used": int(len({tuple(r) for r in perm.tolist()})),
    }


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def pair_collate(
    cache: Mapping[str, Any], pairs: Mapping[str, Any], rows: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Offset the cached per-molecule pairs into global batch atom indices."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    counts = (atom_ptr[rows + 1] - atom_ptr[rows]).astype(np.int64)
    atom_off = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64) if rows.size else np.zeros(0, np.int64)

    ptr = np.asarray(pairs["pair_ptr"], dtype=np.int64)
    starts = ptr[rows]
    ends = ptr[rows + 1]
    pair_counts = (ends - starts).astype(np.int64)
    if int(pair_counts.sum()):
        idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)])
    else:
        idx = np.zeros(0, dtype=np.int64)
    graph = np.repeat(np.arange(rows.size, dtype=np.int64), pair_counts)
    graph_off = atom_off[graph] if graph.size else np.zeros(0, dtype=np.int64)

    def _t(arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.int64))

    return {
        "pair_u": _t(np.asarray(pairs["pair_u"])[idx] + graph_off),
        "pair_v": _t(np.asarray(pairs["pair_v"])[idx] + graph_off),
        "pair_graph": _t(graph),
        "pair_bin": _t(np.asarray(pairs["pair_bin"])[idx]),
    }


def make_batch(
    cache: Mapping[str, Any], rows: Sequence[int], pairs: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    batch = C.V2D.collate(cache, rows)
    if pairs is not None:
        batch.update(pair_collate(cache, pairs, rows))
    return batch


# ---------------------------------------------------------------------------
# coverage statistics / atom renumbering (gates)
# ---------------------------------------------------------------------------


def coverage(pairs: Mapping[str, Any], molecule_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Per-bin coverage and mean pair counts, over ``true`` distance classes."""
    counts = np.asarray(pairs["true_bin_counts"], dtype=np.int64)
    n_molecules = int(counts.shape[0])
    out: dict[str, Any] = {"n_molecules": n_molecules, "bins": []}
    for k, d in enumerate(C.PAIR_DISTANCES):
        column = counts[:, k]
        out["bins"].append(
            {
                "distance": int(d),
                "n_pairs": int(column.sum()),
                "n_molecules_with_pair": int((column > 0).sum()),
                "coverage_fraction": float((column > 0).mean()) if n_molecules else 0.0,
                "mean_pairs_per_molecule": float(column.mean()) if n_molecules else 0.0,
                "mean_pairs_over_covered_molecules": (
                    float(column[column > 0].mean()) if int((column > 0).sum()) else 0.0
                ),
                "max_pairs_per_molecule": int(column.max()) if n_molecules else 0,
            }
        )
    total = counts.sum(axis=1)
    out["n_pairs_total"] = int(total.sum())
    out["n_molecules_with_any_pair"] = int((total > 0).sum())
    out["coverage_fraction_any_distance"] = float((total > 0).mean()) if n_molecules else 0.0
    out["mean_pairs_per_molecule"] = float(total.mean()) if n_molecules else 0.0
    out["max_pairs_per_molecule"] = int(total.max()) if n_molecules else 0
    if molecule_ids is not None:
        out["molecule_id_sample"] = [str(x) for x in list(molecule_ids)[:5]]
    return out


def permute_cache_atoms(cache: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy of ``cache`` with a random atom relabelling per molecule.

    Only the *index* representation changes: atom rows are permuted, bond
    endpoints and S2 path endpoints are remapped, and every per-atom record
    block (S1 code, S2 path record) moves with its atom.  All structural
    content is preserved, so any permutation-invariant model must return the
    same graph-level prediction.
    """
    import copy

    rng = np.random.default_rng(int(seed))
    out = copy.deepcopy(dict(cache))
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    nbr_code = np.asarray(cache["nbr_code"])
    path_cat = np.asarray(cache["path_cat"])
    path_end63 = np.asarray(cache["path_end63"])
    path_end = np.asarray(cache["path_end"], dtype=np.int64)
    path_closure = np.asarray(cache["path_closure"])

    n_molecules = int(cache["n_molecules"])
    new_bond_index = bond_index.copy()
    new_path_end = np.zeros_like(path_end)
    atom_order: list[np.ndarray] = []
    nbr_order: list[np.ndarray] = []
    path_order: list[np.ndarray] = []
    nbr_counts_new: list[int] = []
    path_counts_new: list[int] = []
    for m in range(n_molecules):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        n = a1 - a0
        perm = rng.permutation(n)
        atom_order.append(perm)
        inverse = np.empty(n, dtype=np.int64)
        inverse[perm] = np.arange(n, dtype=np.int64)
        b0, b1 = int(bond_ptr[m]), int(bond_ptr[m + 1])
        for k in range(b0, b1):
            u, v = int(bond_index[k, 0]), int(bond_index[k, 1])
            nu, nv = int(inverse[u]), int(inverse[v])
            new_bond_index[k] = (min(nu, nv), max(nu, nv))
        for atom in range(a0, a1):
            p0, p1 = int(path_ptr[atom]), int(path_ptr[atom + 1])
            for k in range(p0, p1):
                new_path_end[k] = int(inverse[int(path_end[k])])
        for j in range(n):
            atom = a0 + int(perm[j])
            nbr_order.append(np.arange(int(nbr_ptr[atom]), int(nbr_ptr[atom + 1]), dtype=np.int64))
            path_order.append(np.arange(int(path_ptr[atom]), int(path_ptr[atom + 1]), dtype=np.int64))
            nbr_counts_new.append(int(nbr_ptr[atom + 1]) - int(nbr_ptr[atom]))
            path_counts_new.append(int(path_ptr[atom + 1]) - int(path_ptr[atom]))

    def _blocks(arr: np.ndarray, order: list[np.ndarray]) -> np.ndarray:
        if not order:
            return np.asarray(arr)[:0]
        return np.concatenate([np.asarray(arr)[idx] for idx in order], axis=0)

    for key in ("atom_feat", "atom_type", "atom_scalar"):
        arr = np.asarray(cache[key])
        out[key] = np.concatenate(
            [arr[int(atom_ptr[m]) : int(atom_ptr[m + 1])][atom_order[m]] for m in range(n_molecules)],
            axis=0,
        )
    out["nbr_code"] = _blocks(nbr_code, nbr_order)
    out["path_cat"] = _blocks(path_cat, path_order)
    out["path_end63"] = _blocks(path_end63, path_order)
    out["path_end"] = _blocks(new_path_end, path_order).astype(np.int64)
    out["path_closure"] = _blocks(path_closure, path_order)
    out["nbr_ptr"] = np.concatenate([[0], np.cumsum(np.asarray(nbr_counts_new, dtype=np.int64))]).astype(np.int64)
    out["path_ptr"] = np.concatenate([[0], np.cumsum(np.asarray(path_counts_new, dtype=np.int64))]).astype(np.int64)
    out["bond_index"] = new_bond_index.astype(np.int64)
    out["atom_permutation_seed"] = int(seed)
    return out


__all__ = [
    "PERMUTATIONS",
    "bfs_distances",
    "molecule_pairs",
    "_null_seed",
    "null_permutation",
    "null_permutation_table",
    "build_pair_cache",
    "validate_pair_cache",
    "same_pair_content",
    "pair_collate",
    "make_batch",
    "coverage",
    "permute_cache_atoms",
]
