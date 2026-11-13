"""The two fixed structural vectors of this round (no learned state).

``U3`` (arm LOCAL3) is the exact-distance-3 atom-type count of a root::

    U3[v, t] = #{ u : shortest_path_distance(v, u) == 3 and atom_type(u) == t }

It is a count of *distinct* atoms at graph distance exactly three.  Distance-1
and distance-2 atoms are excluded, triangles do not appear, and unreachable
atoms (``dist == -1``) are excluded rather than treated as a large finite
distance.

``U2`` (arm DUP2) is the per-root type histogram of the existing S2 path
occurrences::

    U2[v, t] = #{ path occurrences rooted at v with path_end63 // 5 == t }

``path_end63 = atom_type[end] * 5 + (bond2 - 1)``, so ``// 5`` is the end atom
type.  Repeated end atoms and triangle closures are kept exactly as the frozen
enumerator produced them; this is the deterministic read-out of the S2
end-contingency column that the C1 forward already consumes.

Both vectors are ``log1p``-transformed and standardised with Train3360
per-column ``mean`` / ``std(ddof=0)`` only.  A zero-variance column uses
``scale = 1``.  No target byte is read.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import common as C

UNREACHABLE = -1


# ---------------------------------------------------------------------------
# graph distances (explicit BFS, independent of any helper cache)
# ---------------------------------------------------------------------------


def molecule_adjacency(n_atoms: int, bond_index: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for u, v in np.asarray(bond_index, dtype=np.int64).tolist():
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def all_pairs_distances(n_atoms: int, bond_index: np.ndarray) -> np.ndarray:
    """Unweighted shortest-path distances; unreachable pairs are ``-1``."""
    n = int(n_atoms)
    adj = molecule_adjacency(n, bond_index)
    dist = np.full((n, n), UNREACHABLE, dtype=np.int64)
    for s in range(n):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    if dist[s, v] == UNREACHABLE:
                        dist[s, v] = d
                        nxt.append(int(v))
            frontier = nxt
    return dist


def u3_counts(atom_type: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """``(n_atoms, 16)`` exact distance-3 atom-type counts."""
    atom_type = np.asarray(atom_type, dtype=np.int64).reshape(-1)
    n = int(atom_type.size)
    if dist.shape != (n, n):
        raise ValueError("distance matrix shape mismatch")
    onehot = np.zeros((n, C.N_ATOM_TYPES), dtype=np.int64)
    onehot[np.arange(n), atom_type] = 1
    return ((dist == 3).astype(np.int64) @ onehot).astype(np.int64)


def u2_counts(n_atoms: int, path_ptr_local: np.ndarray, path_end63: np.ndarray) -> np.ndarray:
    """``(n_atoms, 16)`` S2 path-occurrence end-type counts."""
    n = int(n_atoms)
    ptr = np.asarray(path_ptr_local, dtype=np.int64)
    if ptr.shape != (n + 1,):
        raise ValueError("path pointer shape mismatch")
    end63 = np.asarray(path_end63, dtype=np.int64).reshape(-1)
    counts = np.zeros((n, C.N_ATOM_TYPES), dtype=np.int64)
    if end63.size:
        roots = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
        types = end63 // C.N_BOND_TYPES
        if np.any(types < 0) or np.any(types >= C.N_ATOM_TYPES):
            raise ValueError("path_end63 // 5 outside the atom-type vocabulary")
        np.add.at(counts, (roots, types), 1)
    return counts


def distance3_partners(n_atoms: int, bond_index: np.ndarray, root: int) -> list[tuple[int, int]]:
    """``(atom, distance)`` list of the root's distance-3 partners (test helper)."""
    dist = all_pairs_distances(n_atoms, bond_index)
    row = dist[int(root)]
    return [(int(u), int(row[u])) for u in np.flatnonzero(row == 3).tolist()]


# ---------------------------------------------------------------------------
# per-split raw vectors
# ---------------------------------------------------------------------------


def split_raw_vectors(cache: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compute ``u3`` / ``u2`` (and the distance matrix only transiently)."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    path_end63 = np.asarray(cache["path_end63"], dtype=np.int64)
    n_mol = int(cache["n_molecules"])
    n_atoms = int(atom_ptr[-1])
    u3 = np.zeros((n_atoms, C.N_ATOM_TYPES), dtype=np.int64)
    u2 = np.zeros((n_atoms, C.N_ATOM_TYPES), dtype=np.int64)
    for mol in range(n_mol):
        a0, a1 = int(atom_ptr[mol]), int(atom_ptr[mol + 1])
        b0, b1 = int(bond_ptr[mol]), int(bond_ptr[mol + 1])
        p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
        n = a1 - a0
        dist = all_pairs_distances(n, bond_index[b0:b1])
        u3[a0:a1] = u3_counts(atom_type[a0:a1], dist)
        u2[a0:a1] = u2_counts(n, path_ptr[a0 : a1 + 1] - p0, path_end63[p0:p1])
    return {"u3": u3, "u2": u2}


# ---------------------------------------------------------------------------
# Train-only normalisation
# ---------------------------------------------------------------------------


def fit_vector_norm(raw: np.ndarray, name: str) -> dict[str, Any]:
    x = np.log1p(np.asarray(raw, dtype=np.float64))
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    const = np.flatnonzero(std <= 0.0)
    scale = np.where(std > 0.0, std, 1.0)
    return {
        "name": str(name),
        "fit_split": "train",
        "fit_rows": "all atoms of the published Train3360",
        "n_atoms_fit": int(x.shape[0]),
        "log1p": True,
        "ddof": 0,
        "zero_variance_policy": "scale = 1",
        "zero_variance_indices": const.tolist(),
        "mean": mean.astype(np.float64).tolist(),
        "scale": scale.astype(np.float64).tolist(),
    }


def apply_vector_norm(raw: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    x = np.log1p(np.asarray(raw, dtype=np.float64))
    mean = np.asarray(norm["mean"], dtype=np.float64)
    scale = np.asarray(norm["scale"], dtype=np.float64)
    return ((x - mean) / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# persisted feature bundle
# ---------------------------------------------------------------------------


def build_features(caches: Mapping[str, Mapping[str, Any]], *, write: bool = True) -> dict[str, Any]:
    raw = {split: split_raw_vectors(caches[split]) for split in C.SPLITS}
    norm3 = fit_vector_norm(raw["train"]["u3"], "u3")
    norm2 = fit_vector_norm(raw["train"]["u2"], "u2")
    out: dict[str, Any] = {
        "schema": "lcrc-lipo-s3-locality-features-v1",
        "round_id": C.ROUND_ID,
        "train_only_norm": True,
        "target_read": False,
        "norm_s3": norm3,
        "norm_s2": norm2,
    }
    for split in C.SPLITS:
        cache = caches[split]
        out[f"{split}_rows"] = np.asarray(cache["rows"], dtype=np.int64)
        out[f"{split}_atom_ptr"] = np.asarray(cache["atom_ptr"], dtype=np.int64)
        out[f"{split}_u3"] = raw[split]["u3"].astype(np.int32)
        out[f"{split}_u2"] = raw[split]["u2"].astype(np.int32)
        out[f"{split}_s3"] = apply_vector_norm(raw[split]["u3"], norm3)
        out[f"{split}_s2"] = apply_vector_norm(raw[split]["u2"], norm2)
    if write:
        C.feature_path().parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(C.feature_path(), **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
        C.write_json(C.CACHE / "features_manifest.json", {k: v for k, v in out.items() if not isinstance(v, np.ndarray)})
    return out


def load_features() -> dict[str, Any]:
    path = C.feature_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run code/prepare.py --cache-only first")
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    manifest = C.read_json(C.CACHE / "features_manifest.json")
    if manifest.get("schema") != "lcrc-lipo-s3-locality-features-v1":
        raise ValueError("unexpected feature manifest schema")
    out: dict[str, Any] = dict(manifest)
    out.update(arrays)
    return out


def split_features(features: Mapping[str, Any], split: str, cache: Mapping[str, Any]) -> dict[str, Any]:
    rows = np.asarray(features[f"{split}_rows"], dtype=np.int64)
    if [int(x) for x in rows.tolist()] != [int(x) for x in np.asarray(cache["rows"]).tolist()]:
        raise ValueError(f"{split}: feature rows differ from the structure cache")
    atom_ptr = np.asarray(features[f"{split}_atom_ptr"], dtype=np.int64)
    if not np.array_equal(atom_ptr, np.asarray(cache["atom_ptr"], dtype=np.int64)):
        raise ValueError(f"{split}: feature atom_ptr differs from the structure cache")
    return {
        "rows": rows,
        "atom_ptr": atom_ptr,
        "u3": np.asarray(features[f"{split}_u3"], dtype=np.float32),
        "u2": np.asarray(features[f"{split}_u2"], dtype=np.float32),
        "s3": np.asarray(features[f"{split}_s3"], dtype=np.float32),
        "s2": np.asarray(features[f"{split}_s2"], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# NULL3 degeneracy accounting
# ---------------------------------------------------------------------------


def null_group_stats(cache: Mapping[str, Any], s3: np.ndarray) -> dict[str, Any]:
    """How often the same-molecule / same-atom-type mean field is degenerate.

    A root is *degenerate* when its group (the atoms of the same molecule with
    the same ``atom_type``, including itself) has size one, or when every
    ``s3`` vector in the group is identical.  In both cases the NULL3 operator
    equals LOCAL3 for that root by construction.
    """
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    n_mol = int(cache["n_molecules"])
    s3 = np.asarray(s3, dtype=np.float64)
    singleton = 0
    all_equal = 0
    total = 0
    degenerate_atoms = 0
    groups: list[dict[str, Any]] = []
    for mol in range(n_mol):
        a0, a1 = int(atom_ptr[mol]), int(atom_ptr[mol + 1])
        types = atom_type[a0:a1]
        vecs = s3[a0:a1]
        total += a1 - a0
        for t in np.unique(types).tolist():
            idx = np.flatnonzero(types == t)
            size = int(idx.size)
            if size == 1:
                singleton += 1
                degenerate_atoms += 1
                groups.append({"mol": int(mol), "atom_type": int(t), "size": 1, "reason": "singleton"})
                continue
            same = bool(np.all(vecs[idx] == vecs[idx][0]))
            if same:
                all_equal += 1
                degenerate_atoms += size
            groups.append(
                {"mol": int(mol), "atom_type": int(t), "size": size, "reason": "all_equal" if same else "nondegenerate"}
            )
    n_groups = len(groups)
    degenerate = singleton + all_equal
    return {
        "n_atoms": int(total),
        "n_groups": int(n_groups),
        "n_singleton_groups": int(singleton),
        "n_all_equal_groups": int(all_equal),
        "n_degenerate_groups": int(degenerate),
        "n_degenerate_atoms": int(degenerate_atoms),
        "group_degenerate_fraction": float(degenerate / max(n_groups, 1)),
        "root_degenerate_fraction": float(degenerate_atoms / max(total, 1)),
        "classification": "group level; root fraction counts every atom of a degenerate group",
        "group_size_histogram": {
            str(k): int(sum(1 for g in groups if g["size"] == k))
            for k in sorted({g["size"] for g in groups})
        },
    }


__all__ = [
    "UNREACHABLE",
    "molecule_adjacency",
    "all_pairs_distances",
    "u3_counts",
    "u2_counts",
    "distance3_partners",
    "split_raw_vectors",
    "fit_vector_norm",
    "apply_vector_norm",
    "build_features",
    "load_features",
    "split_features",
    "null_group_stats",
]
