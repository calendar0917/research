"""CMA static structural objects (task §2.1/§2.2) -- label-blind enumeration.

Definitions implemented literally:

* every **directed** 3-bond simple path ``(i,j,k,l)`` with four distinct nodes
  and both centre-bond orientations is materialised, so a path and its reverse
  both exist as records;
* ``type(u,w) in {0: no edge, 1/2/3: bond type + 1}``;
* the two-atom fragment ``z_ijk = 0.5 [F2(i,j,k) + F2(k,j,i)]`` only needs the
  *unordered* outer pair, so ``z`` is indexed by ``(centre, min(a,b), max(a,b))``
  and no index ever picks a direction for a path.

Nothing here reads labels, dataset statistics or the target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import xs_common as C

CACHE_DIR = C.CMA_DIR / "cache"


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------


def enumerate_molecule(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> dict:
    """Enumerate CMA objects for one molecule.

    ``bonds`` is ``[(u, v, bond_type), ...]`` with ``bond_type in {0,1,2,3}``
    (single/double/triple/aromatic), matching the frozen ``bond_type`` cache.
    """
    adj: list[list[tuple[int, int]]] = [[] for _ in range(int(n_atoms))]
    for e, (u, v, _t) in enumerate(bonds):
        u, v = int(u), int(v)
        adj[u].append((v, e))
        adj[v].append((u, e))

    bond_of: dict[tuple[int, int], int] = {}
    type_of_bond: dict[tuple[int, int], int] = {}
    for e, (u, v, t) in enumerate(bonds):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        bond_of[key] = e
        type_of_bond[key] = int(t) + 1

    def chord_type(u: int, v: int) -> int:
        return type_of_bond.get((min(u, v), max(u, v)), 0)

    def bond_id(u: int, v: int) -> int:
        return bond_of[(min(u, v), max(u, v))]

    # ---- z triples: (centre j, unordered outer pair {a,b}) -----------------
    z_index: dict[tuple[int, int, int], int] = {}
    z_atom: list[tuple[int, int, int]] = []
    z_bond: list[tuple[int, int]] = []
    z_chord: list[int] = []
    for j in range(int(n_atoms)):
        nbrs = sorted(v for v, _ in adj[j])
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                a, b = nbrs[ai], nbrs[bi]
                z_index[(j, a, b)] = len(z_atom)
                z_atom.append((j, a, b))
                z_bond.append((bond_id(j, a), bond_id(j, b)))
                z_chord.append(chord_type(a, b))

    # ---- directed paths, both centre-bond orientations ---------------------
    path_z: list[tuple[int, int]] = []
    path_bond: list[int] = []
    path_chord: list[tuple[int, int, int]] = []
    directed_per_bond = np.zeros(len(bonds), dtype=np.int64)
    for e, (u, v, _t) in enumerate(bonds):
        for centre, other in ((int(u), int(v)), (int(v), int(u))):
            for i, _e1 in adj[centre]:
                if i == other:
                    continue
                for l, _e2 in adj[other]:
                    if l == centre or l == i:
                        continue
                    # z_ijk with centre=centre, outer pair {i, other}
                    key_l = (centre, min(i, other), max(i, other))
                    key_r = (other, min(centre, l), max(centre, l))
                    assert key_l in z_index and key_r in z_index, (key_l, key_r)
                    path_z.append((z_index[key_l], z_index[key_r]))
                    path_bond.append(e)
                    path_chord.append((chord_type(i, other), chord_type(centre, l),
                                       chord_type(i, l)))
                    directed_per_bond[e] += 1

    # ``directed_per_bond`` is exactly ``2 * n_e`` by construction.
    n_paths = directed_per_bond // 2
    assert np.all(directed_per_bond % 2 == 0)

    return {
        "n_atoms": int(n_atoms),
        "n_bonds": int(len(bonds)),
        "z_atom": np.asarray(z_atom, dtype=np.int64).reshape(-1, 3),
        "z_bond": np.asarray(z_bond, dtype=np.int64).reshape(-1, 2),
        "z_chord": np.asarray(z_chord, dtype=np.int64),
        "path_z": np.asarray(path_z, dtype=np.int64).reshape(-1, 2),
        "path_bond": np.asarray(path_bond, dtype=np.int64),
        "path_chord": np.asarray(path_chord, dtype=np.int64).reshape(-1, 3),
        "bond_npaths": np.asarray(n_paths, dtype=np.int64),
    }


def brute_force_directed_paths(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> list[tuple]:
    """Independent reference: enumerate all node quadruples, no shared helpers."""
    edge: dict[tuple[int, int], int] = {}
    for u, v, t in bonds:
        edge[(int(u), int(v))] = int(t) + 1
        edge[(int(v), int(u))] = int(t) + 1

    def typ(u, v):
        return edge.get((u, v), 0)

    out = []
    nodes = range(int(n_atoms))
    for i in nodes:
        for j in nodes:
            if j == i:
                continue
            for k in nodes:
                if k in (i, j):
                    continue
                for l in nodes:
                    if l in (i, j, k):
                        continue
                    if (i, j) in edge and (j, k) in edge and (k, l) in edge:
                        out.append((i, j, k, l, typ(i, k), typ(j, l), typ(i, l)))
    return out


# ---------------------------------------------------------------------------
# whole split
# ---------------------------------------------------------------------------


def _molecules(cache: Mapping[str, Any]) -> list[tuple[int, list[tuple[int, int, int]]]]:
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    out = []
    for m in range(len(ap) - 1):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        out.append((a1 - a0, [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e]))
                              for e in range(b0, b1)]))
    return out


def build_split(cache: Mapping[str, Any], name: str) -> dict:
    mols = _molecules(cache)
    z_ptr, path_ptr = [0], [0]
    z_atom, z_bond, z_chord = [], [], []
    p_z, p_bond, p_chord = [], [], []
    bond_npaths = []
    for n_atoms, bonds in mols:
        obj = enumerate_molecule(n_atoms, bonds)
        z_atom.append(obj["z_atom"] + 0)
        z_bond.append(obj["z_bond"] + 0)
        z_chord.append(obj["z_chord"])
        p_z.append(obj["path_z"] + 0)  # molecule-local z rows
        p_bond.append(obj["path_bond"] + 0)
        p_chord.append(obj["path_chord"])
        bond_npaths.append(obj["bond_npaths"])
        z_ptr.append(len(obj["z_atom"]) + z_ptr[-1])
        path_ptr.append(len(obj["path_z"]) + path_ptr[-1])

    payload = {
        "version": C.STRUCTURE_VERSION,
        "family": "cma",
        "split": name,
        "n_molecules": len(mols),
        "z_ptr": np.asarray(z_ptr, dtype=np.int64),
        "path_ptr": np.asarray(path_ptr, dtype=np.int64),
        "z_atom": _cat(z_atom, 3, np.int64),
        "z_bond": _cat(z_bond, 2, np.int64),
        "z_chord": _cat(z_chord, 1, np.int64).reshape(-1),
        "path_z": _cat(p_z, 2, np.int64),
        "path_bond": _cat(p_bond, 1, np.int64).reshape(-1),
        "path_chord": _cat(p_chord, 3, np.int64),
        "bond_npaths": _cat(bond_npaths, 1, np.int64).reshape(-1),
    }
    return payload


def _cat(seq: Iterable[np.ndarray], last_dim: int, dtype) -> np.ndarray:
    items = [np.asarray(x).reshape(-1, last_dim) if last_dim > 1 else np.asarray(x).reshape(-1)
             for x in seq]
    if not items:
        return np.zeros((0, last_dim) if last_dim > 1 else (0,), dtype=dtype)
    return np.concatenate(items, 0).astype(dtype)


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"cma_{split}.pt"


def load_split(split: str, *, cache: Mapping[str, Any] | None = None, build: bool = True) -> dict:
    import torch

    path = cache_path(split)
    if path.exists():
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("version") == C.STRUCTURE_VERSION:
            return obj
    if not build:
        raise FileNotFoundError(f"missing CMA structure cache {path}")
    if cache is None:
        cache = C.load_cache(split)
    obj = build_split(cache, split)
    C.ensure_dir(CACHE_DIR)
    C.atomic_torch_save(obj, path)
    return obj


def coverage_stats(obj: Mapping[str, Any], cache: Mapping[str, Any]) -> dict:
    path_ptr = np.asarray(obj["path_ptr"])
    z_ptr = np.asarray(obj["z_ptr"])
    n_paths = np.diff(path_ptr) // 2               # undirected paths per molecule
    n_bonds = np.asarray(cache["n_bonds_graph"], dtype=np.int64)
    per_bond = np.asarray(obj["bond_npaths"], dtype=np.int64)
    with_path = per_bond > 0
    return {
        "n_molecules": int(len(n_paths)),
        "paths_per_molecule": _describe(n_paths),
        "directed_records_per_molecule": _describe(np.diff(path_ptr)),
        "z_triples_per_molecule": _describe(np.diff(z_ptr)),
        "npaths_per_bond": _describe(per_bond),
        "frac_bonds_with_path": float(with_path.mean()),
        "frac_molecules_with_path": float((n_paths > 0).mean()),
        "total_directed_records": int(np.diff(path_ptr).sum()),
        "total_z_triples": int(np.diff(z_ptr).sum()),
        "max_npaths_per_bond": int(per_bond.max()) if per_bond.size else 0,
        "n_bonds": int(n_bonds.sum()),
    }


def _describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


# ---------------------------------------------------------------------------
# batch collation
# ---------------------------------------------------------------------------


def collate(obj: Mapping[str, Any], base: Mapping[str, Any], rows: np.ndarray) -> dict:
    """Add the CMA tensors for ``rows`` to a base batch."""
    import torch
    from xs_data import segments

    rows = np.asarray(rows, dtype=np.int64)
    dev = base["atom_feat"].device
    z_idx, z_mol, z_off = segments(obj["z_ptr"], rows)
    p_idx, p_mol, p_off = segments(obj["path_ptr"], rows)

    z_atom = torch.as_tensor(np.asarray(obj["z_atom"])[z_idx], dtype=torch.int64, device=dev)
    z_bond = torch.as_tensor(np.asarray(obj["z_bond"])[z_idx], dtype=torch.int64, device=dev)
    z_chord = torch.as_tensor(np.asarray(obj["z_chord"])[z_idx], dtype=torch.int64, device=dev)

    atom_off = base["atom_off"][z_mol]
    bond_off = base["bond_off"][z_mol]
    z_atom = z_atom + torch.as_tensor(atom_off, dtype=torch.int64, device=dev).unsqueeze(1)
    z_bond = z_bond + torch.as_tensor(bond_off, dtype=torch.int64, device=dev).unsqueeze(1)

    # paths: z rows are batch-local, bonds are batch-local
    path_z_local = np.asarray(obj["path_z"])[p_idx]          # local z ids
    z_base = z_off[p_mol][:, None]
    path_z = torch.as_tensor(path_z_local + z_base, dtype=torch.int64, device=dev)
    path_bond = torch.as_tensor(
        np.asarray(obj["path_bond"])[p_idx] + base["bond_off"][p_mol],
        dtype=torch.int64, device=dev)
    path_chord = torch.as_tensor(np.asarray(obj["path_chord"])[p_idx],
                                 dtype=torch.int64, device=dev)

    # one event per physical bond of the batch (n_e == 0 included, m_e == 0)
    b_idx, b_mol, _ = segments(np.asarray(base["bond_ptr"], dtype=np.int64), rows)
    event_n = np.asarray(obj["bond_npaths"])[b_idx]
    out = {
        "cma_z_atom": z_atom,
        "cma_z_bond": z_bond,
        "cma_z_chord": z_chord,
        "cma_path_z": path_z,
        "cma_path_bond": path_bond,
        "cma_path_chord": path_chord,
        "cma_event_n": torch.as_tensor(event_n, dtype=torch.float32, device=dev),
    }
    return out
