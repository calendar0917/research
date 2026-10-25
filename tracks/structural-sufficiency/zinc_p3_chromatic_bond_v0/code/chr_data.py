"""Label-free bond-pair relation cache and P3N collation for the chromatic round.

The relation is the lifted atom random-walk bond-pair relation from
``zinc_relation_composition_probe_v1`` (k = 0..20) plus one chordless-cycle
co-membership indicator (length 3..18).  Only the raw atom/bond list is used.
Official test is never referenced.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SS = _EXP.parent
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402

KSTEPS = 21
REL_DIM = KSTEPS + 1
RING_MIN = 3
RING_MAX = 18
CACHE = _EXP / "cache"
ANALYSIS = _EXP / "analysis"
CONFIG = _EXP / "configs" / "chr_v0.json"


# ---------------------------------------------------------------------------
# fixed relation
# ---------------------------------------------------------------------------
def random_walk_powers(edge_index: np.ndarray, n: int, ksteps: int = KSTEPS) -> np.ndarray:
    """``P^0..P^(ksteps-1)`` as ``[n,n,ksteps]`` float64.

    ``P = D^{-1} A`` on the raw unweighted undirected graph; zero-degree rows
    stay zero.  One row per physical bond; parallel edges would collapse to one
    adjacency entry, which is the P3 convention (the raw cache has none).
    """
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n):
        raise ValueError("edge index out of range")
    a = np.zeros((n, n), dtype=np.float64)
    if len(edge_index):
        a[edge_index[:, 0], edge_index[:, 1]] = 1.0
        a[edge_index[:, 1], edge_index[:, 0]] = 1.0
        a = (a > 0).astype(np.float64)
    deg = a.sum(axis=1)
    p = np.zeros_like(a)
    nz = deg > 0
    p[nz] = a[nz] / deg[nz, None]
    out = np.empty((n, n, ksteps), dtype=np.float64)
    q = np.eye(n, dtype=np.float64)
    for k in range(ksteps):
        out[:, :, k] = q
        q = q @ p
    return out


def bond_relation(edge_index: np.ndarray, n: int, ksteps: int = KSTEPS) -> np.ndarray:
    """``[E,E,ksteps]`` lifted bond-pair relation, float64."""
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    powers = random_walk_powers(edge_index, n, ksteps)
    e = len(edge_index)
    if e == 0:
        return np.zeros((0, 0, ksteps), dtype=np.float64)
    u, v = edge_index[:, 0], edge_index[:, 1]
    out = np.empty((e, e, ksteps), dtype=np.float64)
    for k in range(ksteps):
        q = powers[:, :, k]
        out[:, :, k] = 0.5 * (q[np.ix_(u, u)] + q[np.ix_(u, v)]
                              + q[np.ix_(v, u)] + q[np.ix_(v, v)])
    return out


def chordless_ring_indicator(edge_index: np.ndarray, n: int,
                             ring_min: int = RING_MIN, ring_max: int = RING_MAX
                             ) -> tuple[np.ndarray, list[int]]:
    """``[E,E]`` uint8 co-membership in at least one chordless cycle.

    A chordless (induced) cycle of length ``ring_min..ring_max`` is enumerated
    on the raw undirected graph.  All pairs of its edges, including a bond with
    itself, are marked.  Returns the matrix and the sorted list of cycle
    lengths found (one entry per cycle).
    """
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    e = len(edge_index)
    ring = np.zeros((e, e), dtype=np.uint8)
    if e == 0:
        return ring, []
    pairs = [tuple(sorted((int(u), int(v)))) for u, v in edge_index]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate physical bond in a molecule")
    edge_id = {p: i for i, p in enumerate(pairs)}
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(pairs)
    lengths: list[int] = []
    for cyc in nx.chordless_cycles(g, length_bound=int(ring_max)):
        if len(cyc) < int(ring_min) or len(cyc) > int(ring_max):
            continue
        idx = []
        for i in range(len(cyc)):
            a, b = int(cyc[i]), int(cyc[(i + 1) % len(cyc)])
            idx.append(edge_id[tuple(sorted((a, b)))])
        ring[np.ix_(idx, idx)] = 1
        lengths.append(int(len(cyc)))
    return ring, lengths


def build_molecule(edge_index: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """``([E,E,21] float64, [E,E] uint8, cycle lengths)`` for one molecule."""
    r = bond_relation(edge_index, n)
    ring, lengths = chordless_ring_indicator(edge_index, n)
    return r, ring, lengths


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def rel_path(split: str) -> Path:
    if split not in ("train", "valid"):
        raise ValueError(f"split {split!r} is not allowed; official test is never read")
    return CACHE / f"chr_rel_{split}.npy"


def meta_path(split: str) -> Path:
    return CACHE / f"chr_rel_{split}_meta.json"


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}.npy")
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)
    for extra in path.parent.glob(path.name + ".tmp*"):
        if extra.name != path.name and not extra.name.endswith(str(os.getpid()) + ".npy"):
            try:
                extra.unlink()
            except OSError:
                pass


def build_split(split: str, *, force: bool = False) -> dict[str, Any]:
    """Build the label-free relation cache for train/valid.

    The P3 cache is read only for ``atom_ptr``, ``bond_ptr`` and
    ``bond_index``.  Labels are not accessed by this function.
    """
    if split not in ("train", "valid"):
        raise ValueError("official test is never read")
    mpath = meta_path(split)
    rpath = rel_path(split)
    if mpath.exists() and rpath.exists() and not force:
        return json.loads(mpath.read_text(encoding="utf-8"))

    cache = D.load_cache(split)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    n_mol = int(len(atom_ptr) - 1)
    n_bonds = np.diff(bond_ptr).astype(np.int64)
    n_atoms = np.diff(atom_ptr).astype(np.int64)
    total_pairs = int((n_bonds.astype(np.int64) ** 2).sum())
    out = np.empty((total_pairs, REL_DIM), dtype=np.float32)
    offset = np.zeros(n_mol + 1, dtype=np.int64)
    cycle_hist: dict[str, int] = {}
    ring_pairs = 0
    per_mol_ring_frac: list[float] = []
    per_mol_bonds_in_ring: list[int] = []
    ring_sums = np.zeros(KSTEPS, dtype=np.float64)
    nonring_sums = np.zeros(KSTEPS, dtype=np.float64)
    ring_count = 0
    nonring_count = 0
    t0 = time.time()
    for row in range(n_mol):
        n = int(n_atoms[row])
        e0, e1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        e = e1 - e0
        off = int(offset[row])
        offset[row + 1] = off + e * e
        if e == 0:
            per_mol_ring_frac.append(0.0)
            per_mol_bonds_in_ring.append(0)
            continue
        edges = bond_index[e0:e1]
        r, ring, lengths = build_molecule(edges, n)
        block = np.empty((e, e, REL_DIM), dtype=np.float32)
        block[:, :, :KSTEPS] = r.astype(np.float32)
        block[:, :, KSTEPS] = ring.astype(np.float32)
        out[off:off + e * e] = block.reshape(e * e, REL_DIM)
        for length in lengths:
            cycle_hist[str(length)] = cycle_hist.get(str(length), 0) + 1
        rp = int(ring.sum())
        ring_pairs += rp
        per_mol_ring_frac.append(rp / float(e * e))
        per_mol_bonds_in_ring.append(int((ring.diagonal() > 0).sum()))
        mask = ring.astype(bool)
        if mask.any():
            ring_sums += r[mask].sum(axis=0)
            ring_count += int(mask.sum())
        if (~mask).any():
            nonring_sums += r[~mask].sum(axis=0)
            nonring_count += int((~mask).sum())
    if int(offset[-1]) != total_pairs:
        raise RuntimeError(f"offset end {offset[-1]} != {total_pairs}")

    ring_diff = None
    if ring_count and nonring_count:
        ring_diff = (ring_sums / ring_count - nonring_sums / nonring_count).tolist()
    # point-biserial correlation between the ring indicator and each RW column
    corr = []
    if ring_pairs and total_pairs > ring_pairs:
        ring_flat = out[:, KSTEPS].astype(np.float64)
        rs = float(ring_flat.std())
        for k in range(KSTEPS):
            x = out[:, k].astype(np.float64)
            xs = float(x.std())
            if rs == 0.0 or xs == 0.0:
                corr.append(0.0)
            else:
                corr.append(float(((x - x.mean()) * (ring_flat - ring_flat.mean())).mean() / (xs * rs)))
    else:
        corr = [0.0] * KSTEPS

    meta = {
        "split": split,
        "ksteps": KSTEPS,
        "rel_dim": REL_DIM,
        "ring_min": RING_MIN,
        "ring_max": RING_MAX,
        "n_molecules": n_mol,
        "total_pairs": total_pairs,
        "total_bonds": int(n_bonds.sum()),
        "n_bonds": n_bonds.tolist(),
        "n_atoms": n_atoms.tolist(),
        "offset": offset.tolist(),
        "bond_ptr": bond_ptr.tolist(),
        "atom_ptr": atom_ptr.tolist(),
        "stats": {
            "ring_pair_count": int(ring_pairs),
            "ring_pair_fraction": float(ring_pairs / max(total_pairs, 1)),
            "molecules_with_ring": int(sum(1 for x in per_mol_ring_frac if x > 0)),
            "per_molecule_ring_pair_fraction_mean": float(np.mean(per_mol_ring_frac)) if per_mol_ring_frac else 0.0,
            "per_molecule_ring_pair_fraction_median": float(np.median(per_mol_ring_frac)) if per_mol_ring_frac else 0.0,
            "per_molecule_ring_pair_fraction_p10": float(np.quantile(per_mol_ring_frac, 0.10)) if per_mol_ring_frac else 0.0,
            "per_molecule_ring_pair_fraction_p90": float(np.quantile(per_mol_ring_frac, 0.90)) if per_mol_ring_frac else 0.0,
            "bonds_in_ring_fraction": float(sum(per_mol_bonds_in_ring) / max(int(n_bonds.sum()), 1)),
            "cycle_length_histogram": dict(sorted(cycle_hist.items(), key=lambda x: int(x[0]))),
            "ring_minus_nonring_mean_rw": ring_diff,
            "point_biserial_ring_vs_rw": corr,
        },
        "provenance": {
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": C.sha256_file(Path(__file__).resolve()),
            "ad_cache": str(D.cache_path(split)),
            "ad_cache_sha256": C.sha256_file(D.cache_path(split)),
            "labels_read": False,
            "official_test_loaded": False,
            "built_utc": C.utcnow(),
            "build_seconds": float(time.time() - t0),
        },
    }
    _atomic_npy(rpath, out)
    meta["relation_cache"] = str(rpath)
    meta["relation_cache_sha256"] = C.sha256_file(rpath)
    mpath.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_meta(split: str) -> dict[str, Any]:
    if split not in ("train", "valid"):
        raise ValueError("official test is never read")
    return json.loads(meta_path(split).read_text(encoding="utf-8"))


def open_relation(split: str, *, mmap: bool = True) -> np.ndarray:
    if split not in ("train", "valid"):
        raise ValueError("official test is never read")
    return np.load(rel_path(split), mmap_mode="r" if mmap else None)


# ---------------------------------------------------------------------------
# batch collation
# ---------------------------------------------------------------------------
def line_degree_from_junctions(junc_bond: torch.Tensor, n_bonds: int) -> torch.Tensor:
    """Number of junction incidences per physical bond (line-graph degree)."""
    out = torch.zeros((int(n_bonds),), dtype=torch.float32, device=junc_bond.device)
    if junc_bond.numel():
        out.index_add_(0, junc_bond.reshape(-1).to(torch.long),
                       torch.ones(junc_bond.numel(), dtype=torch.float32,
                                  device=junc_bond.device))
    return out


def make_collate(split: str):
    """P3N collate plus one shared dense relation block per batch."""
    meta = load_meta(split)
    rel = open_relation(split, mmap=True)
    offset = np.asarray(meta["offset"], dtype=np.int64)
    n_bonds_all = np.asarray(meta["n_bonds"], dtype=np.int64)
    n_atoms_all = np.asarray(meta["n_atoms"], dtype=np.int64)

    def collate(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        rows = np.asarray(rows, dtype=np.int64)
        batch = T1.collate(cache, rows, "P3N", valid=valid, walk_cols=walk_cols)
        b = len(rows)
        bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
        ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
        counts = (bp[rows + 1] - bp[rows]).astype(np.int64)
        n_atoms = (ap[rows + 1] - ap[rows]).astype(np.int64)
        if not np.array_equal(counts, n_bonds_all[rows]) or not np.array_equal(n_atoms, n_atoms_all[rows]):
            raise RuntimeError(
                f"collate split mismatch: cache counts do not match this relation split "
                f"(rows {rows[:3]}...); use make_train_eval_collate()"
            )
        bb = int(counts.max()) if b else 0
        rel_block = np.zeros((b, bb, bb, REL_DIM), dtype=np.float32)
        rel_mask = np.zeros((b, bb), dtype=bool)
        for i, row in enumerate(rows):
            e = int(counts[i])
            if e == 0:
                continue
            s, t = int(offset[row]), int(offset[row + 1])
            if t - s != e * e:
                raise RuntimeError(f"row {row}: relation offset {t-s} != E^2 {e*e}")
            rel_block[i, :e, :e] = rel[s:t].reshape(e, e, REL_DIM)
            rel_mask[i, :e] = True
        local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts]) if counts.sum() else np.zeros(0, np.int64)
        batch.update({
            "rel": torch.from_numpy(rel_block),
            "rel_mask": torch.from_numpy(rel_mask),
            "bond_local": torch.from_numpy(local),
            "n_bonds": torch.from_numpy(counts.astype(np.int64)),
            "n_atoms": torch.from_numpy(n_atoms.astype(np.int64)),
            "n_bonds_pad": bb,
        })
        return batch

    collate.meta = meta
    return collate


def make_train_eval_collate():
    """Dispatch train/valid caches to their own relation splits.

    ``ad_train.train`` uses one collate callable for both training and
    validation; the ``valid`` flag identifies the split.
    """
    c_train = make_collate("train")
    c_valid = make_collate("valid")

    def _matches(fn, cache, rows: np.ndarray) -> bool:
        meta = fn.meta
        nb = np.asarray(meta["n_bonds"], dtype=np.int64)
        na = np.asarray(meta["n_atoms"], dtype=np.int64)
        bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
        ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= len(nb)):
            return False
        return (np.array_equal(bp[rows + 1] - bp[rows], nb[rows])
                and np.array_equal(ap[rows + 1] - ap[rows], na[rows]))

    def collate(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        rows = np.asarray(rows, dtype=np.int64)
        candidates = [fn for fn in (c_train, c_valid) if _matches(fn, cache, rows)]
        if len(candidates) == 1:
            fn = candidates[0]
        else:
            # Fallback: the two splits have distinct sizes in this round.
            fn = c_valid if len(cache["labels"]) == 1000 else c_train
        return fn(cache, rows, candidate, valid=valid, walk_cols=walk_cols)

    collate.meta = {"train": c_train.meta, "valid": c_valid.meta}
    return collate


# ---------------------------------------------------------------------------
# synthetic graphs for boundary checks
# ---------------------------------------------------------------------------
def junctions_from_edges(edge_index: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """All unordered pairs of incident bonds at each atom, centre in column 1."""
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    inc: list[list[int]] = [[] for _ in range(n)]
    for k, (u, v) in enumerate(edge_index):
        inc[int(u)].append(k)
        inc[int(v)].append(k)
    ji, jb = [], []
    for centre, bonds in enumerate(inc):
        bonds = sorted(bonds)
        for a in range(len(bonds)):
            for b in range(a + 1, len(bonds)):
                e1, e2 = bonds[a], bonds[b]
                u = int(edge_index[e1, 0] if edge_index[e1, 1] == centre else edge_index[e1, 1])
                v = int(edge_index[e2, 0] if edge_index[e2, 1] == centre else edge_index[e2, 1])
                ji.append((u, centre, v))
                jb.append((e1, e2))
    return (np.asarray(ji, dtype=np.int64).reshape(-1, 3),
            np.asarray(jb, dtype=np.int64).reshape(-1, 2))


def synthetic_batch(edge_index: np.ndarray, n: int) -> dict[str, torch.Tensor]:
    """Minimal P3N-compatible one-molecule batch with zero structural features.

    Used only for boundary/finiteness checks.  The junction list is real for the
    supplied bond graph so the P3 junction encoder sees a valid incidence.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    e = len(edge_index)
    ji, jb = junctions_from_edges(edge_index, n)
    r, ring, _ = build_molecule(edge_index, n)
    rel = np.zeros((1, e, e, REL_DIM), dtype=np.float32)
    if e:
        rel[0, :, :, :KSTEPS] = r.astype(np.float32)
        rel[0, :, :, KSTEPS] = ring.astype(np.float32)
    return {
        "n_graphs": 1,
        "atom_feat": torch.zeros((n, 108), dtype=torch.float32),
        "atom_batch": torch.zeros((n,), dtype=torch.long),
        "atom_type": torch.zeros((n,), dtype=torch.long),
        "bond_feat": torch.zeros((e, 7), dtype=torch.float32),
        "bond_type": torch.ones((e,), dtype=torch.long),
        "bond_batch": torch.zeros((e,), dtype=torch.long),
        "bond_index": torch.from_numpy(edge_index.copy()),
        "junc_feat": torch.zeros((len(jb), 32), dtype=torch.float32),
        "junc_batch": torch.zeros((len(jb),), dtype=torch.long),
        "junc_index": torch.from_numpy(ji.copy()),
        "junc_bond": torch.from_numpy(jb.copy()),
        "global": torch.zeros((1, 33), dtype=torch.float32),
        "rel": torch.from_numpy(rel),
        "rel_mask": torch.ones((1, e), dtype=torch.bool) if e else torch.zeros((1, 0), dtype=torch.bool),
        "bond_local": torch.arange(e, dtype=torch.long),
        "n_bonds": torch.tensor([e], dtype=torch.long),
        "n_atoms": torch.tensor([n], dtype=torch.long),
        "n_bonds_pad": e,
    }
