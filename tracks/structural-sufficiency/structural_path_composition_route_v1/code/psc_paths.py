"""Simple-path objects for PSC: enumeration, canonical keys, cache, batching.

Frozen design (PROTOCOL.md section 5):

* every molecule contributes every simple (vertex-distinct) undirected path of
  length ``l = 0..L``; a length-``l`` path has ``l+1`` atoms and ``l`` bonds and
  is canonically oriented as ``min(tokens, reversed tokens)``;
* a path is stored as the raw 9-column atom rows and 3-column bond rows.  No
  recomputed degree / ring / ring-count field is added;
* repeated occurrences of the same canonical key inside one molecule are kept
  as ``multiplicity`` (they are real distinct paths, not duplicates);
* the cache key is a *cache key only*: it is not a learnable id and no
  frequency statistic is attached to it.

Cache schema (``psc-path-cache-v1``) members:

    meta_json          JSON: schema, L, raw hash, code hash, counts, timing
    rows               [R] int64  molecule row ids covered (Train + Valid only)
    mol_ptr            [R+1] int64  entry range of each molecule
    ent_path           [E] int32  global canonical path id
    ent_mult           [E] int32  occurrences of that key inside the molecule
    mol_len_counts     [R, L+1] int32  true geometric path counts per length
    path_len           [P] uint8
    path_atom          [P, L+1, 9] uint8   canonical ORDER orientation
    path_bond          [P, L, 3] uint8
    path_atom_bag      [P, L+1, 9] uint8   member-sorted atoms (BAG input)
    path_bond_bag      [P, L, 3] uint8
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import psc_common as C

SCHEMA = "psc-path-cache-v1"


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------


def adjacency(n_atoms: int, bond_u: np.ndarray, bond_v: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for u, v in zip(np.asarray(bond_u, np.int64).tolist(), np.asarray(bond_v, np.int64).tolist()):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def enumerate_paths(adj: Sequence[Sequence[int]], max_len: int
                    ) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """All simple undirected paths with ``len(vertices) - 1 <= max_len``.

    A path and its reverse are the same geometric path and are enumerated once
    (the vertex tuple is kept only when ``tuple <= tuple[::-1]``).  Every atom
    is one length-0 path.
    """
    paths: list[tuple[int, ...]] = []
    counts = np.zeros(int(max_len) + 1, np.int64)
    for root in range(len(adj)):
        counts[0] += 1
        paths.append((int(root),))
        stack: list[tuple[int, ...]] = [(int(root),)]
        while stack:
            seq = stack.pop()
            if len(seq) - 1 >= int(max_len):
                continue
            for nxt in adj[seq[-1]]:
                if nxt in seq:
                    continue
                cand = seq + (int(nxt),)
                if cand <= cand[::-1]:
                    counts[len(cand) - 1] += 1
                    paths.append(cand)
                stack.append(cand)
    return paths, counts


def tokens_for(seq: Sequence[int], atom_feat: np.ndarray, bond_feat: np.ndarray,
               bond_row: np.ndarray) -> tuple[np.ndarray, np.ndarray, bytes]:
    l = len(seq) - 1
    atoms = np.zeros((len(seq), C.ATOM_COLUMNS), np.uint8)
    bonds = np.zeros((max(l, 0), C.BOND_COLUMNS), np.uint8)
    for i, a in enumerate(seq):
        atoms[i] = atom_feat[int(a)]
    for i in range(l):
        e = int(bond_row[int(seq[i]), int(seq[i + 1])])
        if e < 0:
            raise ValueError("path edge missing from the bond lookup")
        bonds[i] = bond_feat[e]
    rev_atoms = atoms[::-1].copy()
    rev_bonds = bonds[::-1].copy()
    # canonical orientation: lexicographic minimum of (atoms, bonds) tokens
    if (rev_atoms.tobytes(), rev_bonds.tobytes()) < (atoms.tobytes(), bonds.tobytes()):
        atoms, bonds = rev_atoms, rev_bonds
    return atoms, bonds, atoms.tobytes() + bonds.tobytes()


def _sort_rows(rows: np.ndarray) -> np.ndarray:
    """Deterministic lexicographic sort of ``[n, k]`` uint8 rows."""
    if rows.size == 0:
        return rows
    return rows[np.lexsort(rows.T[::-1])]


# ---------------------------------------------------------------------------
# cache build
# ---------------------------------------------------------------------------


def build_cache(raw: Mapping[str, Any], rows: Sequence[int], max_len: int,
                cache_path: Path | str = C.PATH_CACHE, log=print) -> dict[str, Any]:
    rows = np.asarray(rows, np.int64)
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"], np.uint8)
    bond_feat = np.asarray(raw["bond_feat"], np.uint8)
    bond_u = np.asarray(raw["bond_u"], np.int64)
    bond_v = np.asarray(raw["bond_v"], np.int64)

    vocab: dict[bytes, int] = {}
    path_atom: list[np.ndarray] = []
    path_bond: list[np.ndarray] = []
    ent_path: list[np.ndarray] = []
    ent_mult: list[np.ndarray] = []
    mol_ptr = np.zeros(rows.size + 1, np.int64)
    len_counts = np.zeros((rows.size, int(max_len) + 1), np.int32)
    t0 = time.time()
    max_atoms = 0
    for pos, row in enumerate(rows.tolist()):
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        max_atoms = max(max_atoms, n)
        mol_atoms = atom_feat[a0:a1]
        mol_bonds = bond_feat[b0:b1]
        adj = adjacency(n, bond_u[b0:b1], bond_v[b0:b1])
        bond_row = np.full((n, n), -1, np.int64)
        for e, (u, v) in enumerate(zip(bond_u[b0:b1].tolist(), bond_v[b0:b1].tolist())):
            bond_row[int(u), int(v)] = e
            bond_row[int(v), int(u)] = e
        paths, counts = enumerate_paths(adj, int(max_len))
        len_counts[pos] = counts
        local: dict[int, int] = {}
        for seq in paths:
            atoms, bonds, key = tokens_for(seq, mol_atoms, mol_bonds, bond_row)
            pid = vocab.get(key)
            if pid is None:
                pid = len(path_atom)
                vocab[key] = pid
                path_atom.append(atoms)
                path_bond.append(bonds)
            local[pid] = local.get(pid, 0) + 1
        ids = np.fromiter(local.keys(), np.int32, len(local))
        mult = np.fromiter(local.values(), np.int32, len(local))
        order = np.argsort(ids, kind="stable")
        ids, mult = ids[order], mult[order]
        ent_path.append(ids)
        ent_mult.append(mult)
        mol_ptr[pos + 1] = mol_ptr[pos] + ids.size
    build_seconds = time.time() - t0

    n_paths = len(path_atom)
    L = int(max_len)
    pa = np.zeros((n_paths, L + 1, C.ATOM_COLUMNS), np.uint8)
    pb = np.zeros((n_paths, L, C.BOND_COLUMNS), np.uint8)
    plen = np.zeros(n_paths, np.uint8)
    for pid in range(n_paths):
        a = path_atom[pid]
        b = path_bond[pid]
        pa[pid, :a.shape[0]] = a
        pb[pid, :b.shape[0]] = b
        plen[pid] = a.shape[0] - 1
    ent_path_arr = (np.concatenate(ent_path) if ent_path else np.zeros(0, np.int32))
    ent_mult_arr = (np.concatenate(ent_mult) if ent_mult else np.zeros(0, np.int32))
    # BAG inputs are a pure function of the ORDER key (members only), so they
    # are precomputed once and stored next to it.
    pa_bag = np.zeros_like(pa)
    pb_bag = np.zeros_like(pb)
    t1 = time.time()
    for pid in range(n_paths):
        l = int(plen[pid])
        pa_bag[pid] = pa[pid]
        if l:
            pa_bag[pid, :l + 1] = _sort_rows(pa[pid, :l + 1])
            pb_bag[pid, :l] = _sort_rows(pb[pid, :l])
    bag_seconds = time.time() - t1

    meta = {
        "schema": SCHEMA,
        "max_len": L,
        "n_rows": int(rows.size),
        "n_entries": int(ent_path_arr.size),
        "n_paths": int(n_paths),
        "max_atom_slots_used": int(max_atoms),
        "atom_columns": C.ATOM_COLUMNS,
        "bond_columns": C.BOND_COLUMNS,
        "raw_cache_sha256": str(raw["cache_sha256"]),
        "code_sha256": C.code_sha(["psc_paths.py"]),
        "built_utc": C.now_utc(),
        "build_seconds": build_seconds,
        "bag_seconds": bag_seconds,
        "mean_paths_per_molecule_by_len": len_counts.mean(0).tolist(),
        "rows_sha256": C.array_sha256(rows),
        "definition": "all simple undirected paths length 0..L; canonical raw-token key; "
                      "multiplicity kept per molecule",
    }
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        rows=rows, mol_ptr=mol_ptr, ent_path=ent_path_arr, ent_mult=ent_mult_arr,
        mol_len_counts=len_counts, path_len=plen,
        path_atom=pa, path_bond=pb, path_atom_bag=pa_bag, path_bond_bag=pb_bag,
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    meta["cache_path"] = str(cache_path)
    meta["cache_sha256"] = C.sha256_file(cache_path)
    meta["cache_bytes"] = int(cache_path.stat().st_size)
    log(f"  path cache: {n_paths} unique keys, {ent_path_arr.size} entries, "
        f"{meta['cache_bytes'] / 1e6:.1f} MB, build {build_seconds:.1f}s "
        f"(bag {bag_seconds:.1f}s)")
    return meta


def ensure_cache(raw: Mapping[str, Any], rows: Sequence[int], max_len: int,
                 cache_path: Path | str = C.PATH_CACHE, log=print) -> dict[str, Any]:
    cache_path = Path(cache_path)
    rows = np.asarray(rows, np.int64)
    code_hash = C.code_sha(["psc_paths.py"])
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            stored_rows = np.asarray(z["rows"], np.int64)
        ok = (meta.get("schema") == SCHEMA
              and int(meta.get("max_len", -1)) == int(max_len)
              and meta.get("raw_cache_sha256") == str(raw["cache_sha256"])
              and meta.get("rows_sha256") == C.array_sha256(rows)
              and meta.get("code_sha256") == code_hash)
        if ok:
            meta["cache_path"] = str(cache_path)
            meta["cache_sha256"] = C.sha256_file(cache_path)
            meta["compat"] = "reused (schema/L/raw/code/rows match)"
            return meta
        log(f"  path cache incompatible (schema/L/raw/code/rows), rebuilding: "
            f"{[(k, meta.get(k)) for k in ('schema', 'max_len', 'code_sha256')]}")
    meta = build_cache(raw, rows, max_len, cache_path, log=log)
    meta["compat"] = "built"
    return meta


# ---------------------------------------------------------------------------
# batched tensors
# ---------------------------------------------------------------------------


class PathCache:
    """In-memory view of a built cache plus the batch index builder."""

    def __init__(self, meta: Mapping[str, Any], path: Path | str = C.PATH_CACHE,
                 device: Any = "cpu") -> None:
        import torch

        self.meta = dict(meta)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self.rows = np.asarray(z["rows"], np.int64)
            self.mol_ptr = np.asarray(z["mol_ptr"], np.int64)
            self.ent_path = np.asarray(z["ent_path"], np.int32)
            self.ent_mult = np.asarray(z["ent_mult"], np.int32)
            self.mol_len_counts = np.asarray(z["mol_len_counts"], np.int32)
            self.path_len = np.asarray(z["path_len"], np.uint8)
            self.path_atom = np.asarray(z["path_atom"], np.uint8)
            self.path_bond = np.asarray(z["path_bond"], np.uint8)
            self.path_atom_bag = np.asarray(z["path_atom_bag"], np.uint8)
            self.path_bond_bag = np.asarray(z["path_bond_bag"], np.uint8)
        self.row_pos = {int(r): i for i, r in enumerate(self.rows.tolist())}
        self.L = int(self.meta["max_len"])
        self.n_paths = int(self.path_len.size)
        self._t: dict[str, Any] = {}

    # -- tensor cache (CPU) --------------------------------------------------
    def tensor(self, name: str):
        import torch

        if name not in self._t:
            src = {"atom": self.path_atom, "bond": self.path_bond,
                   "atom_bag": self.path_atom_bag, "bond_bag": self.path_bond_bag,
                   "plen": self.path_len}[name]
            tensor = torch.as_tensor(np.ascontiguousarray(src))
            self._t[name] = tensor.long().to(self.device)
        return self._t[name]

    def positions(self, rows: Sequence[int]) -> np.ndarray:
        return np.asarray([self.row_pos[int(r)] for r in np.asarray(rows, np.int64)], np.int64)

    def gather(self, rows: Sequence[int], c_l: np.ndarray, bag: bool) -> dict[str, Any]:
        """Batch tensors for ``rows``.

        Entries are grouped per molecule, deduplicated by the *ORDER* canonical
        key inside the batch, and weighted by ``multiplicity * c_len``.
        """
        import torch

        pos = self.positions(rows)
        starts = self.mol_ptr[pos]
        ends = self.mol_ptr[pos + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        ent_idx = np.empty(total, np.int64)
        owner = np.empty(total, np.int64)
        offset = 0
        for i, (s, n) in enumerate(zip(starts.tolist(), lengths.tolist())):
            if n:
                ent_idx[offset:offset + n] = np.arange(s, s + n)
                owner[offset:offset + n] = i
                offset += n
        ids = self.ent_path[ent_idx].astype(np.int64)
        mult = self.ent_mult[ent_idx].astype(np.float32)
        plen_entry = self.path_len[ids].astype(np.int64)
        weight = mult * c_l[plen_entry].astype(np.float32)
        uniq, inv = np.unique(ids, return_inverse=True)
        plen_u = torch.as_tensor(self.path_len[uniq].astype(np.int64)).to(self.device)
        atom_src = self.tensor("atom_bag" if bag else "atom")
        bond_src = self.tensor("bond_bag" if bag else "bond")
        return {
            "n_molecules": int(pos.size),
            "n_entries": total,
            "ids": torch.as_tensor(uniq).to(self.device),
            "inv": torch.as_tensor(inv.astype(np.int64)).to(self.device),
            "owner": torch.as_tensor(owner).to(self.device),
            "weight": torch.as_tensor(weight).to(self.device),
            "plen": plen_u,
            "atom": atom_src[torch.as_tensor(uniq).to(self.device)],
            "bond": bond_src[torch.as_tensor(uniq).to(self.device)],
            "is_bag": bool(bag),
        }

    def path_counts(self, rows: Sequence[int]) -> np.ndarray:
        """True geometric path counts ``|P_l|`` for the given rows (cache order)."""
        return self.mol_len_counts[self.positions(rows)]


def c_l_from_cache(cache: PathCache, fit_rows: Sequence[int]) -> np.ndarray:
    counts = cache.path_counts(fit_rows).astype(np.float64)
    mean = counts.mean(0)
    return 1.0 / np.maximum(1.0, mean)