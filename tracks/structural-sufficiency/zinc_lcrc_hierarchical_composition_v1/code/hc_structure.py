"""Static three-bond path objects for LCRC-HC3-v1.

The enumeration is label-blind and is the exact semantic content already
validated in ``zinc_compositional_research_tree_v1/cma/cma_structure.py``:

* **z triples**: for every atom ``j`` and every *unordered* pair of its
  neighbours ``{a,b}`` a record ``z_atom=(j,a,b)`` with the two incident bond
  rows and the induced chord type ``type(a,b) in {0,1,2,3}`` (0 = no edge,
  1/2/3 = single/double/triple/aromatic + 1).
* **directed three-bond simple paths** ``(i,j,k,l)``: for every physical centre
  bond ``e={j,k}`` *both* centre orientations, then all ``i in N(j)\\{k}`` and
  ``l in N(k)\\{j}`` with ``i != l``.  The reverse path is a separate record.
  Per record we store the four atoms, the three physical bond rows
  ``(i-j, j-k, k-l)``, the centre bond row and the three chord types
  ``(type(i,k), type(j,l), type(i,l))``.
* ``zL`` of a record is the triple ``(i,j,k)`` and ``zR`` the triple ``(j,k,l)``.

Everything is sorted into a canonical order ``(i,j,k,l)`` for the paths and
``(j,a,b)`` for the z triples, so path/bond input order cannot choose a
direction or an ordering.  ``n_e`` is the number of **undirected** paths whose
centre bond is ``e``; it equals ``directed_per_bond / 2`` exactly.

The module also provides the flat per-split cache and a collate function that
turns base LCRC batches plus these rows into batch-local tensors.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

import hc_common as C

N_CHORD = 4


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------


def enumerate_molecule(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> dict:
    """Enumerate the canonical z triples and directed three-bond paths.

    ``bonds`` is ``[(u, v, bond_type), ...]`` with ``bond_type in {0,1,2,3}``
    (the frozen LCRC cache convention; 0 is *not* used by ZINC's bond table,
    but the chord vocabulary is ``{0,1,2,3}``).
    """
    bonds = [(int(u), int(v), int(t)) for u, v, t in bonds]
    n_atoms = int(n_atoms)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for e, (u, v, _t) in enumerate(bonds):
        adj[u].append((v, e))
        adj[v].append((u, e))

    bond_of: dict[tuple[int, int], int] = {}
    type_of_bond: dict[tuple[int, int], int] = {}
    for e, (u, v, t) in enumerate(bonds):
        key = (min(u, v), max(u, v))
        bond_of[key] = e
        type_of_bond[key] = t + 1

    def chord_type(u: int, v: int) -> int:
        return type_of_bond.get((min(u, v), max(u, v)), 0)

    def bond_id(u: int, v: int) -> int:
        return bond_of[(min(u, v), max(u, v))]

    # ---- z triples, canonical order (centre, low outer, high outer) --------
    z_atom: list[tuple[int, int, int]] = []
    z_bond: list[tuple[int, int]] = []
    z_chord: list[int] = []
    z_index: dict[tuple[int, int, int], int] = {}
    for j in range(n_atoms):
        nbrs = sorted(v for v, _ in adj[j])
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                a, b = nbrs[ai], nbrs[bi]
                z_index[(j, a, b)] = len(z_atom)
                z_atom.append((j, a, b))
                z_bond.append((bond_id(j, a), bond_id(j, b)))
                z_chord.append(chord_type(a, b))

    # ---- directed paths, both centre orientations -------------------------
    records: list[tuple] = []
    directed_per_bond = np.zeros(len(bonds), dtype=np.int64)
    for e, (u, v, _t) in enumerate(bonds):
        for centre, other in ((u, v), (v, u)):
            for i, _e1 in adj[centre]:
                if i == other:
                    continue
                for l, _e2 in adj[other]:
                    if l == centre or l == i:
                        continue
                    i_centre = bond_id(i, centre)
                    other_l = bond_id(other, l)
                    records.append((
                        i, centre, other, l,
                        i_centre, e, other_l,
                        chord_type(i, other), chord_type(centre, l), chord_type(i, l),
                        z_index[(centre, min(i, other), max(i, other))],
                        z_index[(other, min(centre, l), max(centre, l))],
                    ))
                    directed_per_bond[e] += 1

    assert np.all(directed_per_bond % 2 == 0), directed_per_bond
    records.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    keys = [(r[0], r[1], r[2], r[3]) for r in records]
    assert len(keys) == len(set(keys)), "duplicate node quadruple; multi-edge graph"

    if records:
        arr = np.asarray(records, dtype=np.int64)
        path_atom = arr[:, 0:4]
        path_bonds = arr[:, 4:7]
        path_chord = arr[:, 7:10]
        path_z = arr[:, 10:12]
    else:
        path_atom = np.zeros((0, 4), dtype=np.int64)
        path_bonds = np.zeros((0, 3), dtype=np.int64)
        path_chord = np.zeros((0, 3), dtype=np.int64)
        path_z = np.zeros((0, 2), dtype=np.int64)

    return {
        "n_atoms": n_atoms,
        "n_bonds": int(len(bonds)),
        "z_atom": np.asarray(z_atom, dtype=np.int64).reshape(-1, 3),
        "z_bond": np.asarray(z_bond, dtype=np.int64).reshape(-1, 2),
        "z_chord": np.asarray(z_chord, dtype=np.int64),
        "path_atom": path_atom,
        "path_bonds": path_bonds,
        "path_bond": path_bonds[:, 1] if len(path_bonds) else np.zeros(0, dtype=np.int64),
        "path_chord": path_chord,
        "path_z": path_z,
        "bond_npaths": directed_per_bond // 2,
    }


def brute_force_directed_paths(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> list[tuple]:
    """Independent reference: enumerate every ordered quadruple."""
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


def _molecules(cache: Mapping[str, object]) -> list[tuple[int, list[tuple[int, int, int]]]]:
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    nb = np.diff(bp)
    out = []
    for m in range(len(ap) - 1):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        if b1 - b0 != int(nb[m]):
            raise RuntimeError("bond pointer / n_bonds mismatch")
        out.append((a1 - a0, [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e]))
                              for e in range(b0, b1)]))
    return out


def _cat(seq: Iterable[np.ndarray], last_dim: int, dtype) -> np.ndarray:
    items = [np.asarray(x).reshape(-1, last_dim) if last_dim > 1 else np.asarray(x).reshape(-1)
             for x in seq]
    if not items:
        return np.zeros((0, last_dim) if last_dim > 1 else (0,), dtype=dtype)
    return np.concatenate(items, 0).astype(dtype)


def build_split(cache: Mapping[str, object], name: str) -> dict:
    if name not in ("train", "valid"):
        raise PermissionError(f"split {name!r} is outside the LCRC-HC3 data boundary")
    mols = _molecules(cache)
    z_ptr, path_ptr = [0], [0]
    z_atom, z_bond, z_chord = [], [], []
    p_atom, p_bonds, p_chord, p_z, p_bond = [], [], [], [], []
    bond_npaths = []
    for n_atoms, bonds in mols:
        obj = enumerate_molecule(n_atoms, bonds)
        z_atom.append(obj["z_atom"])
        z_bond.append(obj["z_bond"])
        z_chord.append(obj["z_chord"])
        p_atom.append(obj["path_atom"])
        p_bonds.append(obj["path_bonds"])
        p_chord.append(obj["path_chord"])
        p_z.append(obj["path_z"])
        p_bond.append(obj["path_bond"])
        bond_npaths.append(obj["bond_npaths"])
        z_ptr.append(len(obj["z_atom"]) + z_ptr[-1])
        path_ptr.append(len(obj["path_atom"]) + path_ptr[-1])

    payload = {
        "version": C.STRUCTURE_VERSION,
        "family": "hc3",
        "split": name,
        "n_molecules": len(mols),
        "z_ptr": np.asarray(z_ptr, dtype=np.int64),
        "path_ptr": np.asarray(path_ptr, dtype=np.int64),
        "z_atom": _cat(z_atom, 3, np.int32),
        "z_bond": _cat(z_bond, 2, np.int32),
        "z_chord": _cat(z_chord, 1, np.int64).reshape(-1),
        "path_atom": _cat(p_atom, 4, np.int32),
        "path_bonds": _cat(p_bonds, 3, np.int32),
        "path_bond": _cat(p_bond, 1, np.int32).reshape(-1),
        "path_chord": _cat(p_chord, 3, np.int64),
        "path_z": _cat(p_z, 2, np.int32),
        "bond_npaths": _cat(bond_npaths, 1, np.int64).reshape(-1),
        "bond_ptr": np.asarray(cache["bond_ptr"], dtype=np.int64),
    }
    # cheap structural identities (no labels, no model)
    assert len(payload["bond_npaths"]) == int(np.sum(
        np.diff(np.asarray(cache["bond_ptr"], dtype=np.int64))))
    if len(payload["path_atom"]):
        assert len(payload["path_bond"]) == len(payload["path_atom"])
        assert len(payload["path_chord"]) == len(payload["path_atom"])
        assert len(payload["path_z"]) == len(payload["path_atom"])
    return payload


def struct_cache_path(split: str):
    if split not in ("train", "valid"):
        raise PermissionError(f"split {split!r} is outside the LCRC-HC3 data boundary")
    return C.CACHE / f"hc3_{split}.pt"


def load_split(split: str, *, cache: Mapping[str, object] | None = None,
               build: bool = True) -> dict:
    if split not in ("train", "valid"):
        raise PermissionError(f"split {split!r} is outside the LCRC-HC3 data boundary")
    path = struct_cache_path(split)
    if path.exists():
        obj = C.atomic_torch_load(path)
        if obj.get("version") == C.STRUCTURE_VERSION:
            return obj
    if not build:
        raise FileNotFoundError(f"missing HC3 structure cache {path}")
    if cache is None:
        cache = C.load_cache(split)
    obj = build_split(cache, split)
    C.ensure_dir(path.parent)
    C.atomic_torch_save(obj, path)
    return obj


# ---------------------------------------------------------------------------
# batch collation
# ---------------------------------------------------------------------------


def collate(struct: Mapping[str, object], base: Mapping[str, object],
            rows: Sequence[int]) -> dict:
    """Add HC3 tensors (all batch-local) to a frozen LCRC base batch."""
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    dev = base["atom_feat"].device
    z_idx, z_mol, z_start, _ = C.ranges(struct["z_ptr"], rows)
    p_idx, p_mol, p_start, _ = C.ranges(struct["path_ptr"], rows)
    b_idx, b_mol, _b_start, _ = C.ranges(np.asarray(struct["bond_ptr"], dtype=np.int64), rows)

    atom_off_np = np.asarray(base["atom_off"], dtype=np.int64)
    bond_counts = np.asarray(base["bond_counts"], dtype=np.int64)
    bond_off_np = np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64) \
        if len(bond_counts) else np.zeros(0, dtype=np.int64)
    atom_off = torch.as_tensor(atom_off_np, dtype=torch.int64, device=dev)
    bond_off = torch.as_tensor(bond_off_np, dtype=torch.int64, device=dev)
    z_start_t = torch.as_tensor(z_start, dtype=torch.int64, device=dev)
    atom_off_t = atom_off[torch.as_tensor(z_mol, dtype=torch.int64, device=dev)].unsqueeze(1)
    bond_off_t = bond_off[torch.as_tensor(z_mol, dtype=torch.int64, device=dev)].unsqueeze(1)
    z_atom = torch.as_tensor(np.asarray(struct["z_atom"])[z_idx], dtype=torch.int64,
                             device=dev) + atom_off_t
    z_bond = torch.as_tensor(np.asarray(struct["z_bond"])[z_idx], dtype=torch.int64,
                             device=dev) + bond_off_t

    p_atom_off = atom_off[torch.as_tensor(p_mol, dtype=torch.int64, device=dev)].unsqueeze(1)
    p_bond_off = bond_off[torch.as_tensor(p_mol, dtype=torch.int64, device=dev)].unsqueeze(1)
    path_atom = torch.as_tensor(np.asarray(struct["path_atom"])[p_idx], dtype=torch.int64,
                                device=dev) + p_atom_off
    path_bonds = torch.as_tensor(np.asarray(struct["path_bonds"])[p_idx], dtype=torch.int64,
                                 device=dev) + p_bond_off
    path_bond = torch.as_tensor(np.asarray(struct["path_bond"])[p_idx], dtype=torch.int64,
                                device=dev) + p_bond_off[:, 0]
    path_z = torch.as_tensor(np.asarray(struct["path_z"])[p_idx], dtype=torch.int64,
                             device=dev) + z_start_t[torch.as_tensor(
                                 p_mol, dtype=torch.int64, device=dev)].unsqueeze(1)

    event_n = np.asarray(struct["bond_npaths"], dtype=np.int64)[b_idx]
    return {
        "hc_z_atom": z_atom,
        "hc_z_bond": z_bond,
        "hc_z_chord": torch.as_tensor(np.asarray(struct["z_chord"])[z_idx],
                                      dtype=torch.int64, device=dev),
        "hc_path_atom": path_atom,
        "hc_path_bonds": path_bonds,
        "hc_path_bond": path_bond,
        "hc_path_chord": torch.as_tensor(np.asarray(struct["path_chord"])[p_idx],
                                         dtype=torch.int64, device=dev),
        "hc_path_z": path_z,
        "hc_event_n": torch.as_tensor(event_n, dtype=torch.float32, device=dev),
        "hc_path_batch": torch.as_tensor(p_mol, dtype=torch.int64, device=dev),
        "hc_z_batch": torch.as_tensor(z_mol, dtype=torch.int64, device=dev),
    }


# ---------------------------------------------------------------------------
# coverage diagnostics
# ---------------------------------------------------------------------------


def coverage_stats(obj: Mapping[str, object], cache: Mapping[str, object]) -> dict:
    path_ptr = np.asarray(obj["path_ptr"], dtype=np.int64)
    z_ptr = np.asarray(obj["z_ptr"], dtype=np.int64)
    n_paths = np.diff(path_ptr) // 2
    per_bond = np.asarray(obj["bond_npaths"], dtype=np.int64)
    return {
        "n_molecules": int(len(n_paths)),
        "total_directed_records": int(np.diff(path_ptr).sum()),
        "total_z_triples": int(np.diff(z_ptr).sum()),
        "n_bonds": int(len(per_bond)),
        "mean_paths_per_molecule": float(n_paths.mean()) if n_paths.size else 0.0,
        "max_paths_per_molecule": int(n_paths.max()) if n_paths.size else 0,
        "mean_npaths_per_bond": float(per_bond.mean()) if per_bond.size else 0.0,
        "max_npaths_per_bond": int(per_bond.max()) if per_bond.size else 0,
        "frac_bonds_with_path": float((per_bond > 0).mean()) if per_bond.size else 0.0,
        "frac_molecules_with_path": float((n_paths > 0).mean()) if n_paths.size else 0.0,
    }


__all__ = ["N_CHORD", "enumerate_molecule", "brute_force_directed_paths",
           "build_split", "struct_cache_path", "load_split", "collate", "coverage_stats"]
