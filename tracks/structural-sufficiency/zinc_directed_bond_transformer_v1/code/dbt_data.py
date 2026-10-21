"""Static directed-bond objects + bounded exact simple-path relations (DBT).

Everything here is a deterministic, label-free function of the raw atom/bond
graph (through the frozen P3 object cache, which itself is a label-free
one-shot function of that graph).  No ``y``, no target component, no
valid/test statistic enters any feature.

Two levels of objects
---------------------
A physical (undirected) bond ``{u,v}`` produces two directed objects
``e = (u->v)`` and ``reverse(e) = (v->u)``.  The physical bond table is
canonical (``u < v``, from ``unique_undirected_bonds``), so object ``2k`` is
``(u_k -> v_k)`` and object ``2k+1`` is ``(v_k -> u_k)``.

For every directed object we store, as *indices only* (no learned content):

* ``dir_index``  the two atom indices ``(source, target)``,
* ``dir_bond``   the physical bond it belongs to,
* ``src_junc``   the real junctions centred on the **source** atom that pair
  the target with each of the source's other neighbours, i.e. the junction
  objects ``(k, u, v)`` with ``k in N(u) \\ {v}`` in P3 ``(i, j, k)`` notation
  (``j`` is the centre).  The number of such junctions is ``deg(u) - 1``.
* ``dst_junc``   the same at the **target**: junctions ``(u, v, k)`` with
  ``k in N(v) \\ {u}``; count ``deg(v) - 1``.

``J_src`` / ``J_dst`` are the means of the *learned* P3 junction states ``w``
over those index lists, computed at batch time; the means are never cached.

Atom-pair relation cache
------------------------
For every molecule we cache the label-free atom-pair tables

* ``path``   ``S_l(i,j)`` = number of **simple** paths (no repeated node) from
  ``i`` to ``j`` with exactly ``l`` edges, ``l = 1..L_MAX`` (default 12).
  Row ``l-1``; ``S_l(i,i) = 0`` for ``l >= 1``.  Symmetric by construction.
  Counts are stored as ``uint8``; the builder refuses to write if any count
  exceeds 255 (the observed maximum over train/valid/test is 12).
* ``dist``   shortest-path distance: ``0..12`` exact, ``13`` = connected but
  longer than 12, ``14`` = unreachable.  Long distance and unreachable are
  therefore never conflated.

The 12-length bound is a *bounded exact* count (exhaustive enumeration of all
simple paths with at most ``L_MAX`` edges), not a per-graph truncation, not a
sample, and not an all-length "complete structure" object.

At batch time the atom-pair tables are gathered into the four directed-pair
endpoint blocks ``(u,p) (u,q) (v,p) (v,q)`` for ``e=(u->v)``, ``f=(p->q)``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_REPO = _SSUF.parents[1]
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_REPO / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as AT  # noqa: E402

CACHE = _EXP / "cache"

PATH_VERSION = "dbt-path-v1-L12"
L_MAX = 12                 # exact simple-path lengths 1..L_MAX
PATH_COUNTS = L_MAX        # stored columns per atom pair
DIST_EXACT_MAX = 12
DIST_LONG = 13             # connected, shortest path longer than DIST_EXACT_MAX
DIST_UNREACH = 14          # no path at all
DIST_BUCKETS = 15

PATH_EMB_DIM = 8           # shared Linear(PATH_COUNTS -> 8) per endpoint block
DIST_EMB_DIM = 6           # shared Embedding(DIST_BUCKETS -> 6) per block
N_ENDPOINT_BLOCKS = 4      # (u,p) (u,q) (v,p) (v,q)
N_EQ_FLAGS = 4             # endpoint equality per block
N_STRUCT_FLAGS = 5         # same bond / reverse / self / same source / same target
REL_RAW_DIM = N_ENDPOINT_BLOCKS * (PATH_EMB_DIM + DIST_EMB_DIM) + N_EQ_FLAGS + N_STRUCT_FLAGS

GLOBAL_DIM = D.GLOBAL_DIM
ATOM_OUT = 96              # P3 atom/junction width
BOND_OUT = 96              # P3 bond width


# ---------------------------------------------------------------------------
# per-molecule static builders
# ---------------------------------------------------------------------------


def _adjacency(n_atoms: int, bond_index: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for u, v in bond_index:
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)
    for a in range(n_atoms):
        adj[a].sort()
    return adj


def simple_path_counts(n_atoms: int, adj: Sequence[Sequence[int]], l_max: int = L_MAX) -> np.ndarray:
    """Exact counts of simple paths with 1..l_max edges, shape [n, n, l_max].

    Exhaustive depth-first enumeration from every source with a visited flag,
    ``O(#self-avoiding walks of length <= l_max)``.  Increments the counter for
    the endpoint reached at each depth, so ``out[s, t, l-1] = S_l(s,t)``.
    """
    out = np.zeros((n_atoms, n_atoms, l_max), dtype=np.int64)
    visited = np.zeros(n_atoms, dtype=bool)

    def dfs(src: int, cur: int, depth: int) -> None:
        if depth >= 1:
            out[src, cur, depth - 1] += 1
        if depth == l_max:
            return
        for nxt in adj[cur]:
            if not visited[nxt]:
                visited[nxt] = True
                dfs(src, nxt, depth + 1)
                visited[nxt] = False

    for s in range(n_atoms):
        visited[s] = True
        dfs(s, s, 0)
        visited[s] = False
    return out


def exact_distances(n_atoms: int, adj: Sequence[Sequence[int]]) -> np.ndarray:
    """All-pairs BFS distance; unreachable stays at a large sentinel."""
    inf = 1 << 20
    dist = np.full((n_atoms, n_atoms), inf, dtype=np.int64)
    for s in range(n_atoms):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if dist[s, v] == inf:
                        dist[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def build_relation_mol(obj) -> dict[str, np.ndarray]:
    """One P3 ``MolObjects`` -> static relation / directed-object tables."""
    n = int(obj.n_atoms)
    bi = np.asarray(obj.bond_index, dtype=np.int64).reshape(-1, 2)
    n_bonds = int(bi.shape[0])
    adj = _adjacency(n, bi)

    pc = simple_path_counts(n, adj)
    if int(pc.max()) > 255:
        raise RuntimeError(f"simple-path count {int(pc.max())} exceeds uint8")
    path = pc.astype(np.uint8).reshape(-1)

    dist = exact_distances(n, adj)
    dist_b = np.where(dist <= DIST_EXACT_MAX, dist,
                      np.where(dist < (1 << 20), DIST_LONG, DIST_UNREACH)).astype(np.uint8)

    # junction lookup: (centre, neighbour) -> junction ids, from the real P3
    # incidence table (junc_index[:, 1] is the centre atom).
    junc = np.asarray(obj.junc_index, dtype=np.int64).reshape(-1, 3)
    at: dict[tuple[int, int], list[int]] = {}
    for jid in range(junc.shape[0]):
        i, j, k = (int(junc[jid, 0]), int(junc[jid, 1]), int(junc[jid, 2]))
        at.setdefault((j, i), []).append(jid)
        at.setdefault((j, k), []).append(jid)

    dir_index = np.zeros((2 * n_bonds, 2), dtype=np.int16)
    dir_bond = np.zeros(2 * n_bonds, dtype=np.int16)
    src_lists: list[list[int]] = []
    dst_lists: list[list[int]] = []
    for k in range(n_bonds):
        u, v = int(bi[k, 0]), int(bi[k, 1])
        for slot, (s, t) in enumerate(((u, v), (v, u))):
            oid = 2 * k + slot
            dir_index[oid] = (s, t)
            dir_bond[oid] = k
            src_lists.append(sorted(at.get((s, t), [])))
            dst_lists.append(sorted(at.get((t, s), [])))

    src_len = np.asarray([len(x) for x in src_lists], dtype=np.int64)
    dst_len = np.asarray([len(x) for x in dst_lists], dtype=np.int64)
    src_junc = np.asarray([j for x in src_lists for j in x], dtype=np.int32)
    dst_junc = np.asarray([j for x in dst_lists for j in x], dtype=np.int32)

    # structural invariants of this incidence mapping
    if src_junc.size != int(src_len.sum()) or dst_junc.size != int(dst_len.sum()):
        raise RuntimeError("junction incidence length mismatch")
    n_junc = int(junc.shape[0])
    if int(src_len.sum()) != 2 * n_junc or int(dst_len.sum()) != 2 * n_junc:
        raise RuntimeError("each junction must contribute to exactly 2 src and 2 dst sides")
    for oid in range(2 * n_bonds):
        s, t = int(dir_index[oid, 0]), int(dir_index[oid, 1])
        if int(src_len[oid]) != len(adj[s]) - 1 or int(dst_len[oid]) != len(adj[t]) - 1:
            raise RuntimeError("junction count must equal deg(endpoint)-1")
    if len(dir_index) >= 2:
        # reverse object exists and carries the swapped role lists
        if not np.array_equal(dir_index[0::2], dir_index[1::2][:, ::-1]):
            raise RuntimeError("directed object table is not paired")
        if not np.array_equal(dir_bond[0::2], dir_bond[1::2]):
            raise RuntimeError("directed object bond table is not paired")
        for k in range(n_bonds):
            if not np.array_equal(src_lists[2 * k], dst_lists[2 * k + 1]):
                raise RuntimeError("src/dst junction roles do not swap under reversal")

    return {
        "n_atoms": n,
        "n_bonds": n_bonds,
        "n_junc": n_junc,
        "n_dir": 2 * n_bonds,
        "path": path,
        "dist": dist_b.reshape(-1),
        "dir_index": dir_index,
        "dir_bond": dir_bond,
        "src_len": src_len,
        "dst_len": dst_len,
        "src_junc": src_junc,
        "dst_junc": dst_junc,
    }


# ---------------------------------------------------------------------------
# split cache
# ---------------------------------------------------------------------------


def cache_path(split: str) -> Path:
    return CACHE / f"{PATH_VERSION}_{split}.pt"


def build_relation_split(split: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    objs, obj_source = D.load_p3_objects(split)
    n_mol = len(objs)

    pair_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    dir_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    src_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    dst_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    n_atoms = np.zeros(n_mol, dtype=np.int64)
    n_bonds = np.zeros(n_mol, dtype=np.int64)
    n_junc = np.zeros(n_mol, dtype=np.int64)

    path_chunks, dist_chunks = [], []
    dir_index_chunks, dir_bond_chunks = [], []
    src_junc_chunks, dst_junc_chunks = [], []
    src_len_chunks, dst_len_chunks = [], []
    max_count = 0

    for i, obj in enumerate(objs):
        r = build_relation_mol(obj)
        n = r["n_atoms"]
        n_atoms[i] = n
        n_bonds[i] = r["n_bonds"]
        n_junc[i] = r["n_junc"]
        pair_ptr[i + 1] = pair_ptr[i] + n * n
        dir_ptr[i + 1] = dir_ptr[i] + r["n_dir"]
        src_ptr[i + 1] = src_ptr[i] + int(r["src_len"].sum())
        dst_ptr[i + 1] = dst_ptr[i] + int(r["dst_len"].sum())
        path_chunks.append(r["path"])
        dist_chunks.append(r["dist"])
        dir_index_chunks.append(r["dir_index"])
        dir_bond_chunks.append(r["dir_bond"])
        src_junc_chunks.append(r["src_junc"])
        dst_junc_chunks.append(r["dst_junc"])
        src_len_chunks.append(r["src_len"].astype(np.uint8))
        dst_len_chunks.append(r["dst_len"].astype(np.uint8))
        max_count = max(max_count, int(np.asarray(r["path"]).max(initial=0)))
        if (i + 1) % 2000 == 0:
            C.log(f"[dbt prep {split}] {i + 1}/{n_mol} "
                  f"{time.perf_counter() - t0:.1f}s max_count={max_count}")

    data = {
        "split": split,
        "n_molecules": int(n_mol),
        "pair_ptr": pair_ptr,
        "path": np.concatenate(path_chunks, axis=0).astype(np.uint8),
        "dist": np.concatenate(dist_chunks, axis=0).astype(np.uint8),
        "dir_ptr": dir_ptr,
        "dir_index": np.concatenate(dir_index_chunks, axis=0).astype(np.int16),
        "dir_bond": np.concatenate(dir_bond_chunks, axis=0).astype(np.int16),
        "src_ptr": src_ptr,
        "dst_ptr": dst_ptr,
        "src_junc": np.concatenate(src_junc_chunks, axis=0).astype(np.int32),
        "dst_junc": np.concatenate(dst_junc_chunks, axis=0).astype(np.int32),
        "src_len": np.concatenate(src_len_chunks, axis=0).astype(np.uint8),
        "dst_len": np.concatenate(dst_len_chunks, axis=0).astype(np.uint8),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_junc": n_junc,
        "meta": {
            "path_version": PATH_VERSION,
            "L_MAX": int(L_MAX),
            "path_semantics": "exact number of simple paths with exactly l edges (1<=l<=L_MAX)",
            "dist_semantics": f"0..{DIST_EXACT_MAX} exact; {DIST_LONG}=connected longer; "
                              f"{DIST_UNREACH}=unreachable",
            "max_path_count": int(max_count),
            "n_atoms_total": int(n_atoms.sum()),
            "n_bonds_total": int(n_bonds.sum()),
            "n_dir_total": int(dir_ptr[-1]),
            "n_junc_total": int(n_junc.sum()),
            "src_incidence": int(src_ptr[-1]),
            "dst_incidence": int(dst_ptr[-1]),
            "object_source": obj_source,
            "seconds": float(time.perf_counter() - t0),
            "created": C.utcnow(),
        },
    }
    return data


def save_relation_split(data: Mapping[str, Any]) -> Path:
    path = cache_path(str(data["split"]))
    C.atomic_torch_save(dict(data), path)
    return path


def load_relation_split(split: str) -> dict:
    path = cache_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run dbt_prep.py prep")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def _ranges(ptr: np.ndarray, rows: np.ndarray):
    rows = np.asarray(rows, dtype=np.int64)
    starts = ptr[rows].astype(np.int64)
    ends = ptr[rows + 1].astype(np.int64)
    counts = ends - starts
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)])
    batch = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
    local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])
    return idx, batch, local, counts


def collate(cache: Mapping[str, Any], rel: Mapping[str, Any], rows: Sequence[int],
            *, valid: bool = False) -> dict[str, Any]:
    """Assemble one batch: P3 object tables + directed objects + atom-pair grid.

    P3 atom/bond/junction tensors are produced by the existing ``T`` collate so
    that the encoders see exactly the same layout as in every earlier round.
    """
    base = AT.collate(cache, rows, "T", valid=valid)
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)

    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    junc_ptr = np.asarray(cache["junc_ptr"])
    atom_counts = np.diff(atom_ptr)[rows]
    bond_counts = np.diff(bond_ptr)[rows]
    junc_counts = np.diff(junc_ptr)[rows]
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64)
    bond_off = np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64)
    junc_off = np.concatenate([[0], np.cumsum(junc_counts)[:-1]]).astype(np.int64)

    dir_ptr = np.asarray(rel["dir_ptr"])
    dir_idx, dir_batch, dir_local, dir_counts = _ranges(dir_ptr, rows)
    di = np.asarray(rel["dir_index"])[dir_idx].astype(np.int64)
    di = di + atom_off[dir_batch].reshape(-1, 1)
    dbond = np.asarray(rel["dir_bond"])[dir_idx].astype(np.int64) + bond_off[dir_batch]

    src_ptr = np.asarray(rel["src_ptr"])
    dst_ptr = np.asarray(rel["dst_ptr"])
    s_idx, s_batch, _, s_counts = _ranges(src_ptr, rows)
    d_idx, d_batch, _, d_counts = _ranges(dst_ptr, rows)
    src_junc = np.asarray(rel["src_junc"])[s_idx].astype(np.int64) + junc_off[s_batch]
    dst_junc = np.asarray(rel["dst_junc"])[d_idx].astype(np.int64) + junc_off[d_batch]
    src_ptr_b = np.concatenate([[0], np.cumsum(s_counts)]).astype(np.int64)
    dst_ptr_b = np.concatenate([[0], np.cumsum(d_counts)]).astype(np.int64)
    dir_ptr_b = np.concatenate([[0], np.cumsum(dir_counts)]).astype(np.int64)
    src_len_sel = np.asarray(rel["src_len"])[dir_idx].astype(np.int64)
    dst_len_sel = np.asarray(rel["dst_len"])[dir_idx].astype(np.int64)
    src_obj_ptr = np.concatenate([[0], np.cumsum(src_len_sel)]).astype(np.int64)
    dst_obj_ptr = np.concatenate([[0], np.cumsum(dst_len_sel)]).astype(np.int64)

    # padded atom-pair relation tables [B, Nmax, Nmax, ...]
    Nmax = int(max(1, int(atom_counts.max())))
    pair_ptr = np.asarray(rel["pair_ptr"])
    path = np.asarray(rel["path"])
    dist = np.asarray(rel["dist"])
    path_pad = np.zeros((B, Nmax, Nmax, PATH_COUNTS), dtype=np.uint8)
    dist_pad = np.full((B, Nmax, Nmax), DIST_UNREACH, dtype=np.uint8)
    for i, row in enumerate(rows):
        n = int(atom_counts[i])
        s, e = int(pair_ptr[row]), int(pair_ptr[row + 1])
        npairs = n * n
        if e - s != npairs:
            raise RuntimeError(f"row {row}: pair pointer {e - s} != n^2 {npairs}")
        path_pad[i, :n, :n, :] = path[s * PATH_COUNTS:e * PATH_COUNTS].reshape(
            n, n, PATH_COUNTS)
        dist_pad[i, :n, :n] = dist[s:e].reshape(n, n)

    out = dict(base)
    out.update(
        {
            "n_atoms": torch.from_numpy(atom_counts.astype(np.float32)),
            "n_bonds": torch.from_numpy(bond_counts.astype(np.float32)),
            "n_junc": torch.from_numpy(junc_counts.astype(np.float32)),
            "dir_index": torch.from_numpy(di),
            "dir_bond": torch.from_numpy(dbond),
            "dir_graph": torch.from_numpy(dir_batch),
            "dir_ptr": torch.from_numpy(dir_ptr_b),
            "src_ptr": torch.from_numpy(src_ptr_b),
            "dst_ptr": torch.from_numpy(dst_ptr_b),
            "atom_ptr": torch.from_numpy(
                np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64)),
            "src_obj_ptr": torch.from_numpy(src_obj_ptr),
            "dst_obj_ptr": torch.from_numpy(dst_obj_ptr),
            "src_junc": torch.from_numpy(src_junc),
            "dst_junc": torch.from_numpy(dst_junc),
            "pair_path": torch.from_numpy(path_pad),
            "pair_dist": torch.from_numpy(dist_pad),
            "Nmax": int(Nmax),
            "Dmax": int(max(1, int(dir_counts.max(initial=0)))),
        }
    )
    return out


def fit_graph_scales(rel: Mapping[str, Any]) -> dict[str, float]:
    """Train-only fixed constants used to scale the two pooled moments.

    ``n_dir_ref`` is the mean directed-object count per train molecule and
    ``n_atom_ref`` the mean atom count.  They are frozen numbers, identical for
    every split; no per-molecule normalisation is performed.
    """
    n_dir = 2.0 * np.asarray(rel["n_bonds"], dtype=np.float64)
    n_atom = np.asarray(rel["n_atoms"], dtype=np.float64)
    return {
        "n_dir_ref": float(n_dir.mean()),
        "n_atom_ref": float(n_atom.mean()),
        "fit_split": "train",
        "valid_or_test_used": False,
    }


# ---------------------------------------------------------------------------
# compact P3 object cache (only the fields the encoders and the T collate use)
# ---------------------------------------------------------------------------

OBJECT_VERSION = "dbt-objects-v1"
OBJECT_FIELDS = (
    "atom_ptr", "atom_feat", "atom_type", "bond_ptr", "bond_feat", "bond_type",
    "bond_index", "junc_ptr", "junc_feat", "junc_index", "junc_bond",
    "global_feat", "labels", "n_atoms_graph", "n_bonds_graph",
)


def object_cache_path(split: str) -> Path:
    return CACHE / f"{OBJECT_VERSION}_{split}.pt"


def build_object_cache(split: str) -> dict[str, Any]:
    """Compact copy of the frozen P3 object tables for one split.

    The train/valid tables are byte-identical to the ``ad_*`` caches of the
    earlier rounds (they are built from the same frozen ``MolObjects``); the
    test tables have to be rebuilt from official PyG because no frozen test
    object cache exists on the compute host.  Only the fields used by the
    encoders / ``T`` collate / label vector are kept.
    """
    t0 = time.perf_counter()
    objs, src = D.load_p3_objects(split)
    labels = D.official_labels(split)
    if len(labels) != len(objs):
        raise RuntimeError(f"{split}: labels {len(labels)} vs objects {len(objs)}")
    n_mol = len(objs)
    atom_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    bond_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    junc_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    for i, o in enumerate(objs):
        atom_ptr[i + 1] = atom_ptr[i] + int(o.n_atoms)
        bond_ptr[i + 1] = bond_ptr[i] + int(o.bond_index.shape[0])
        junc_ptr[i + 1] = junc_ptr[i] + int(o.junc_index.shape[0])
    data = {
        "split": split,
        "n_molecules": int(n_mol),
        "labels": np.asarray(labels, dtype=np.float32),
        "atom_ptr": atom_ptr,
        "atom_feat": np.concatenate([np.asarray(o.atom_feat, np.float32) for o in objs]),
        "atom_type": np.concatenate(
            [np.asarray(o.atom_feat)[:, :21].argmax(1).astype(np.int64) for o in objs]),
        "bond_ptr": bond_ptr,
        "bond_feat": np.concatenate([np.asarray(o.bond_feat, np.float32) for o in objs]),
        "bond_type": np.concatenate([np.asarray(o.bond_type, np.int64) for o in objs]),
        "bond_index": np.concatenate([np.asarray(o.bond_index, np.int16) for o in objs]),
        "junc_ptr": junc_ptr,
        "junc_feat": np.concatenate([np.asarray(o.junc_feat, np.float32) for o in objs]),
        "junc_index": np.concatenate([np.asarray(o.junc_index, np.int16) for o in objs]),
        "junc_bond": np.concatenate([np.asarray(o.junc_bond, np.int16) for o in objs]),
        "global_feat": np.stack([np.asarray(o.global_feat, np.float32) for o in objs]),
        "n_atoms_graph": np.diff(atom_ptr).astype(np.int16),
        "n_bonds_graph": np.diff(bond_ptr).astype(np.int16),
        "meta": {
            "object_version": OBJECT_VERSION,
            "split": split,
            "n_molecules": int(n_mol),
            "object_source": src,
            "labels_sha256": C.sha256_array(np.asarray(labels, np.float64)),
            "seconds": float(time.perf_counter() - t0),
            "created": C.utcnow(),
        },
    }
    return data


def load_object_cache(split: str, *, rebuild: bool = False) -> dict:
    path = object_cache_path(split)
    if path.exists() and not rebuild:
        return torch.load(path, map_location="cpu", weights_only=False)
    data = build_object_cache(split)
    C.atomic_torch_save(data, path)
    return data


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "inspect"])
    p.add_argument("--splits", default="train,valid")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    splits = tuple(x for x in args.splits.split(",") if x)
    if args.command == "prep":
        out: dict[str, Any] = {}
        for split in splits:
            path = cache_path(split)
            if path.exists() and not args.force:
                C.log(f"[dbt prep] reuse {path}")
                d = load_relation_split(split)
            else:
                C.log(f"[dbt prep] build {split}")
                d = build_relation_split(split)
                save_relation_split(d)
            out[split] = {"path": str(path), "bytes": path.stat().st_size,
                          "meta": d["meta"], "sha256": C.sha256_file(path)}
        if "train" in splits:
            scales = fit_graph_scales(load_relation_split("train"))
            C.write_json(_EXP / "analysis" / "graph_scales.json", scales)
            out["graph_scales"] = scales
        C.write_json(_EXP / "analysis" / "prep.json", out)
        C.log({k: (v if k != "meta" else "…") for k, v in out.items()})
    else:
        for split in splits:
            d = load_relation_split(split)
            C.log({k: (v.shape if hasattr(v, "shape") else v)
                   for k, v in d.items() if k not in ("meta",)})
            C.log(d["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
