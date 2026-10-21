"""Pair-object / spectral data path for zinc-arch-exploration-v3.

Two new representations are built here, both pure functions of the raw atom /
bond graph (``x`` / ``edge_index`` / ``edge_attr``).  No label, no target
component, and no split statistic ever enters them.

``pair_*`` cache (used by arm **A**, and by **A-SSL**)
    For every *unordered by index but ordered (i,j) pair* of atoms of one
    molecule, a small set of static structural descriptors:

    * ``pair_dist``      shortest-path distance, clipped to 0..8, 9 = no path
    * ``pair_common``    number of common neighbours, clipped to 0..5
    * ``pair_ring``      the physical bond (i,j) lies on a cycle (topology only)
    * ``pair_sameblock`` i and j sit in the same biconnected block with >= 3
                         edges (same ring/fused block; topology only)
    * ``pair_bond``      index of the physical bond for adjacent pairs, -1 else

    The raw bond *type* of an adjacent pair is **not** stored: it is read from
    the (possibly masked) ``bond_type`` table at collate time, so a masked
    graph automatically produces masked typed pair features.

``kernel_*`` tensors (used by arm **B**) are not cached; they are built per
batch from the padded adjacency in ``zae_models.build_kernels`` so that the
operator library is a deterministic function of the (masked) raw topology.

Everything here is label-blind and deterministic.  ``build_pair_split`` only
reads the raw PyG ZINC molecule (plus the frozen P3 object cache for the bond
table, which itself is a label-free function of that molecule).
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
_SUF = _EXP.parent
_REPO = _SUF.parents[1]
_V1 = _SUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_REPO / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

PAIR_FEATURE_VERSION = "zae-v3-pair-struct-v2-kernels"
PAIR_CACHE = _EXP / "cache"

N_ATOM_TYPES = D.N_ATOM_TYPES          # 21 raw ZINC atom classes
ATOM_VOCAB = 28                        # P3 atom one-hot width (mask token = 27)
BOND_VOCAB = D.BOND_FEAT_DIM           # 7 = 4 (type one-hot incl. mask 0) + 3 flags
N_BOND_TYPES = D.N_BOND_TYPES          # 3
DIST_BUCKETS = 10                      # 0..8 distance, 9 = unreachable
COMMON_CAP = 6                         # common-neighbour count buckets 0..5
PAIR_EMB_DIM = 8 + 8 + 8 + 4 + 4       # dist + bond-type + common + ring + block
BOND_FEAT_DIM = D.BOND_FEAT_DIM
GLOBAL_DIM = D.GLOBAL_DIM

_REPO_ROOT = C.REPO


# ---------------------------------------------------------------------------
# static pair descriptors
# ---------------------------------------------------------------------------


def _ring_and_block(n_atoms: int, bond_index: np.ndarray, bond_type: np.ndarray):
    """Ring-bond / same-block pair flags, topology only (bond types ignored)."""
    import structural_relation_objects_v1 as sro

    bonds = [(int(u), int(v), int(bt)) for (u, v), bt in zip(bond_index, bond_type)]
    is_ring_bond, _ring_atom, _n_ring = sro._ring_bond_flags(n_atoms, bonds)
    blocks = sro._biconnected_blocks(n_atoms, bonds)
    block_id = np.full(n_atoms, -1, dtype=np.int64)
    next_id = 0
    for nodes, edges in blocks:
        if len(edges) >= 3:
            for a in nodes:
                block_id[a] = next_id
            next_id += 1
    return is_ring_bond, block_id


def build_pair_mol(n_atoms: int, bond_index: np.ndarray, bond_type: np.ndarray) -> dict:
    """One molecule -> flat n*n pair descriptors (deterministic, label-free)."""
    import structural_relation_objects_v1 as sro

    bonds = [(int(u), int(v), int(bt)) for (u, v), bt in zip(bond_index, bond_type)]
    adj = sro._adjacency(n_atoms, bonds)
    dist = sro._all_pairs_distances(n_atoms, adj)
    dist_bucket = np.clip(dist, 0, 8).astype(np.uint8)
    dist_bucket[dist >= (1 << 20)] = 9

    is_ring_bond, block_id = _ring_and_block(n_atoms, bond_index, bond_type)
    bond_key = {(min(int(u), int(v)), max(int(u), int(v))): k
                for k, (u, v) in enumerate(bond_index)}
    pair_bond = np.full(n_atoms * n_atoms, -1, dtype=np.int16)
    pair_ring = np.zeros(n_atoms * n_atoms, dtype=np.uint8)
    for (u, v), k in bond_key.items():
        pair_bond[u * n_atoms + v] = k
        pair_bond[v * n_atoms + u] = k
        pair_ring[u * n_atoms + v] = np.uint8(is_ring_bond[k])
        pair_ring[v * n_atoms + u] = np.uint8(is_ring_bond[k])

    nbrs = [[v for v, _ in adj[i]] for i in range(n_atoms)]
    pair_common = np.zeros(n_atoms * n_atoms, dtype=np.uint8)
    for i in range(n_atoms):
        si = set(nbrs[i])
        for j in range(n_atoms):
            if j == i:
                continue
            c = len(si.intersection(nbrs[j]))
            pair_common[i * n_atoms + j] = min(c, COMMON_CAP - 1)

    pair_block = np.zeros(n_atoms * n_atoms, dtype=np.uint8)
    for i in range(n_atoms):
        if block_id[i] < 0:
            continue
        for j in range(n_atoms):
            if block_id[j] == block_id[i]:
                pair_block[i * n_atoms + j] = 1
    return {
        "dist": dist_bucket.reshape(-1),
        "common": pair_common.reshape(-1),
        "ring": pair_ring.reshape(-1),
        "block": pair_block.reshape(-1),
        "bond": pair_bond.reshape(-1),
    }


HEAT_SCALES = (0.5, 1.0, 2.0)
TYPED_OPS = (1, 2, 3)


def build_kernel_mol(n_atoms: int, bond_index: np.ndarray, bond_type: np.ndarray) -> np.ndarray:
    """Static operator library for one molecule: [n*n, 3 + 3] float32.

    Columns 0..2 are ``exp(-t L)`` for the symmetric normalized Laplacian at the
    declared scales; columns 3..5 are the symmetric normalized typed adjacencies
    for raw bond types 1..3.  A full eigendecomposition is used, so the heat
    kernels are complete matrix functions (no eigenvector selection).
    """
    n = int(n_atoms)
    adj = np.zeros((n, n), dtype=np.float64)
    typed = np.zeros((n, n, len(TYPED_OPS)), dtype=np.float64)
    for (u, v), bt in zip(bond_index, bond_type):
        u, v, bt = int(u), int(v), int(bt)
        adj[u, v] = adj[v, u] = 1.0
        if bt in TYPED_OPS:
            typed[u, v, TYPED_OPS.index(bt)] = 1.0
            typed[v, u, TYPED_OPS.index(bt)] = 1.0
    deg = adj.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1.0)), 0.0)
    ahat = adj * dinv[:, None] * dinv[None, :]
    lap = np.eye(n) - ahat
    lam, q = np.linalg.eigh(lap)
    ops = []
    for t in HEAT_SCALES:
        ops.append((q * np.exp(-t * lam)[None, :]) @ q.T)
    for k in range(len(TYPED_OPS)):
        ops.append(typed[..., k] * dinv[:, None] * dinv[None, :])
    return np.stack([o.astype(np.float32).reshape(-1) for o in ops], axis=1)


def build_pair_split(split: str) -> dict:
    """Build the whole-split flat pair cache."""
    t0 = time.perf_counter()
    objs, obj_source = D.load_p3_objects(split)
    n_mol = len(objs)
    pair_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    keys = ("dist", "common", "ring", "block", "bond")
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    kern_chunks: list[np.ndarray] = []
    n_atoms = np.zeros(n_mol, dtype=np.int16)
    for i, obj in enumerate(objs):
        n = int(obj.n_atoms)
        n_atoms[i] = n
        mol = build_pair_mol(n, np.asarray(obj.bond_index, dtype=np.int64),
                             np.asarray(obj.bond_type, dtype=np.int64))
        for k in keys:
            chunks[k].append(mol[k])
        kern_chunks.append(build_kernel_mol(n, np.asarray(obj.bond_index, dtype=np.int64),
                                            np.asarray(obj.bond_type, dtype=np.int64)))
        pair_ptr[i + 1] = pair_ptr[i] + n * n
        if (i + 1) % 2000 == 0:
            C.log(f"[zae prep {split}] {i + 1}/{n_mol} {time.perf_counter() - t0:.1f}s")
    data = {
        "split": split,
        "n_molecules": int(n_mol),
        "n_atoms_graph": n_atoms,
        "pair_ptr": pair_ptr,
        "dist": np.concatenate(chunks["dist"]),
        "common": np.concatenate(chunks["common"]),
        "ring": np.concatenate(chunks["ring"]),
        "block": np.concatenate(chunks["block"]),
        "bond": np.concatenate(chunks["bond"]),
        "kern": np.concatenate(kern_chunks, axis=0).astype(np.float32),
        "meta": {
            "feature_version": PAIR_FEATURE_VERSION,
            "object_source": obj_source,
            "dist_buckets": int(DIST_BUCKETS),
            "common_cap": int(COMMON_CAP),
            "kernels": ["heat0.5", "heat1", "heat2", "type1", "type2", "type3"],
            "label_blind": True,
            "prep_seconds": float(time.perf_counter() - t0),
            "created": C.utcnow(),
        },
    }
    return data


def pair_cache_path(split: str) -> Path:
    return PAIR_CACHE / f"pair_{split}.pt"


def save_pair_split(data: Mapping[str, Any], split: str) -> Path:
    path = pair_cache_path(split)
    C.atomic_torch_save(dict(data), path)
    return path


def load_pair_split(split: str) -> dict:
    path = pair_cache_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing pair cache {path}; run `zae_runner.py prep` first")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# collate: cached molecule table -> one pair-object batch
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


def collate_pair(cache: Mapping[str, Any], pair_cache: Mapping[str, Any],
                 rows: Sequence[int], *, valid: bool = False) -> dict[str, Any]:
    """Batch layout for arms A / B (dense padded pairs + packed pair list).

    ``cache`` is the frozen v1 molecule table (atoms/bonds/junctions/global);
    ``pair_cache`` holds the static pair descriptors.  For A-SSL the caller
    passes a masked cache with the *same* layout, so no masking logic leaks in
    here.
    """
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    junc_ptr = np.asarray(cache["junc_ptr"])
    atom_idx, atom_batch, atom_local, atom_counts = _ranges(atom_ptr, rows)
    bond_idx, bond_batch, bond_local, bond_counts = _ranges(bond_ptr, rows)
    junc_idx, junc_batch, junc_local, junc_counts = _ranges(junc_ptr, rows)

    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64)
    bond_off = np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64)
    bi = np.asarray(cache["bond_index"])[bond_idx].astype(np.int64) + atom_off[bond_batch].reshape(-1, 1)
    ji = np.asarray(cache["junc_index"])[junc_idx].astype(np.int64) + atom_off[junc_batch].reshape(-1, 1)
    jb = np.asarray(cache["junc_bond"])[junc_idx].astype(np.int64) + bond_off[junc_batch].reshape(-1, 1)
    atom_type = torch.from_numpy(np.asarray(cache["atom_type"])[atom_idx].astype(np.int64))
    bond_type = torch.from_numpy(np.asarray(cache["bond_type"])[bond_idx].astype(np.int64))

    Nmax = int(atom_counts.max())
    n_pairs = int(np.sum(atom_counts ** 2))
    pair_ptr = np.asarray(pair_cache["pair_ptr"])
    p_starts = pair_ptr[rows].astype(np.int64)
    p_counts = atom_counts ** 2

    # packed pair rows -> (graph, local i, local j)
    pair_i = np.concatenate([np.repeat(np.arange(n, dtype=np.int64), n) for n in atom_counts])
    pair_j = np.concatenate([np.tile(np.arange(n, dtype=np.int64), n) for n in atom_counts])
    pair_graph = np.repeat(np.arange(B, dtype=np.int64), p_counts)
    dense_idx = pair_graph * (Nmax * Nmax) + pair_i * Nmax + pair_j
    nm = np.repeat(atom_counts.astype(np.int64), p_counts)

    dist = np.concatenate([np.asarray(pair_cache["dist"])[s:s + n * n] for s, n in zip(p_starts, atom_counts)])
    common = np.concatenate([np.asarray(pair_cache["common"])[s:s + n * n] for s, n in zip(p_starts, atom_counts)])
    ring = np.concatenate([np.asarray(pair_cache["ring"])[s:s + n * n] for s, n in zip(p_starts, atom_counts)])
    block = np.concatenate([np.asarray(pair_cache["block"])[s:s + n * n] for s, n in zip(p_starts, atom_counts)])
    pair_bond_local = np.concatenate(
        [np.asarray(pair_cache["bond"])[s:s + n * n] for s, n in zip(p_starts, atom_counts)]
    )

    adj_mask = (pair_bond_local >= 0).astype(np.float32)
    # non-adjacent pairs point at bond row 0 and are zeroed by ``adj_mask`` when
    # the bond content is read, so no fake bond row is introduced.
    pair_bond = np.where(pair_bond_local >= 0,
                         pair_bond_local.astype(np.int64) + bond_off[pair_graph], 0)
    pair_bond = np.clip(pair_bond, 0, max(int(bond_idx.size) - 1, 0))

    # dense typed adjacency for the spectral arm
    src = bi[:, 0] - atom_off[bond_batch]
    dst = bi[:, 1] - atom_off[bond_batch]
    adj_typed = np.zeros((B, Nmax, Nmax, N_BOND_TYPES), dtype=np.float32)
    bt = np.asarray(cache["bond_type"])[bond_idx].astype(np.int64)
    typed_ok = bt >= 1  # 0 = the mask token: a masked bond carries no typed edge
    if typed_ok.any():
        adj_typed[bond_batch[typed_ok], src[typed_ok], dst[typed_ok], bt[typed_ok] - 1] = 1.0
        adj_typed[bond_batch[typed_ok], dst[typed_ok], src[typed_ok], bt[typed_ok] - 1] = 1.0
    adj_dense = adj_typed.sum(-1)
    kern_dense = np.zeros((B, Nmax, Nmax, 6), dtype=np.float32)
    if "kern" in pair_cache and not pair_cache.get("masked_table", False):
        kern_src = np.asarray(pair_cache["kern"])
        for gi, (s_, n_) in enumerate(zip(p_starts, atom_counts)):
            kern_dense[gi, :n_, :n_, :] = kern_src[s_:s_ + n_ * n_].reshape(n_, n_, 6)

    node_mask = np.arange(Nmax).reshape(1, Nmax) < atom_counts.reshape(B, 1)

    batch = {
        "n_graphs": B,
        "Nmax": Nmax,
        "atom_feat": torch.from_numpy(np.asarray(cache["atom_feat"])[atom_idx].astype(np.float32)),
        "atom_batch": torch.from_numpy(atom_batch),
        "atom_local": torch.from_numpy(atom_local),
        "atom_type": atom_type,
        "bond_feat": torch.from_numpy(np.asarray(cache["bond_feat"])[bond_idx].astype(np.float32)),
        "bond_type": bond_type,
        "bond_batch": torch.from_numpy(bond_batch),
        "bond_index": torch.from_numpy(bi),
        "junc_feat": torch.from_numpy(np.asarray(cache["junc_feat"])[junc_idx].astype(np.float32)),
        "junc_batch": torch.from_numpy(junc_batch),
        "junc_index": torch.from_numpy(ji),
        "junc_bond": torch.from_numpy(jb),
        "global": torch.from_numpy(np.asarray(cache["global_feat"])[rows].astype(np.float32)),
        "n_atoms": torch.from_numpy(atom_counts.astype(np.float32)),
        "atom_n": torch.from_numpy(atom_counts.astype(np.float32)[atom_batch]),
        "n_bonds": torch.from_numpy(bond_counts.astype(np.float32)),
        "n_junc": torch.from_numpy(junc_counts.astype(np.float32)),
        # pair list
        "pair_graph": torch.from_numpy(pair_graph),
        "pair_i": torch.from_numpy(atom_off[pair_graph] + pair_i),
        "pair_j": torch.from_numpy(atom_off[pair_graph] + pair_j),
        "pair_dense": torch.from_numpy(dense_idx),
        "pair_nm": torch.from_numpy(nm.astype(np.float32)),
        "pair_dist": torch.from_numpy(dist.astype(np.int64)),
        "pair_common": torch.from_numpy(common.astype(np.int64)),
        "pair_ring": torch.from_numpy(ring.astype(np.int64)),
        "pair_block": torch.from_numpy(block.astype(np.int64)),
        "pair_bond": torch.from_numpy(pair_bond.astype(np.int64)),
        "pair_adj": torch.from_numpy(adj_mask),
        "pair_self": torch.from_numpy((pair_i == pair_j).astype(np.float32)),
        # dense graph tensors
        "node_mask": torch.from_numpy(node_mask),
        "atom_slot": torch.from_numpy(atom_batch.astype(np.int64) * Nmax + atom_local),
        "adj_dense": torch.from_numpy(adj_dense),
        "adj_typed": torch.from_numpy(adj_typed),
        "kern_dense": torch.from_numpy(kern_dense),
        "bond_dense_ij": torch.from_numpy(
            bond_batch.astype(np.int64) * (Nmax * Nmax) + src * Nmax + dst
        ),
        "bond_dense_ji": torch.from_numpy(
            bond_batch.astype(np.int64) * (Nmax * Nmax) + dst * Nmax + src
        ),
        "n_pairs": int(n_pairs),
    }
    if valid and "labels" in cache:
        batch["labels"] = torch.from_numpy(np.asarray(cache["labels"])[rows].astype(np.float32))
    if "atom_mask_flag" in cache:  # A-SSL pretext table
        batch["atom_mask_flag"] = torch.from_numpy(
            np.asarray(cache["atom_mask_flag"])[atom_idx].astype(np.float32))
        batch["atom_mask_target"] = torch.from_numpy(
            np.asarray(cache["atom_mask_target"])[atom_idx].astype(np.int64))
        batch["bond_mask_flag"] = torch.from_numpy(
            np.asarray(cache["bond_mask_flag"])[bond_idx].astype(np.float32))
        batch["bond_mask_target"] = torch.from_numpy(
            np.asarray(cache["bond_mask_target"])[bond_idx].astype(np.int64))
    return batch


# ---------------------------------------------------------------------------
# masked-object provider for A-SSL (fresh mask every visit, full rebuild)
# ---------------------------------------------------------------------------


def pack_objects(objs: Sequence[Any], atom_type: np.ndarray, bond_type: np.ndarray) -> dict:
    """Pack a list of MolObjects into the same flat table as the v1 cache.

    ``atom_type`` / ``bond_type`` are the *raw* codes that encode this object
    table (the masked codes for SSL, which is exactly what the input is built
    from).  No unmasked copy of any feature is included.
    """
    atom_ptr = np.zeros(len(objs) + 1, dtype=np.int64)
    bond_ptr = np.zeros(len(objs) + 1, dtype=np.int64)
    junc_ptr = np.zeros(len(objs) + 1, dtype=np.int64)
    atom_feats, bond_feats, bond_indices = [], [], []
    junc_feats, junc_indices, junc_bonds = [], [], []
    global_feats = []
    for i, obj in enumerate(objs):
        n = int(obj.n_atoms)
        m = int(obj.bond_index.shape[0])
        atom_ptr[i + 1] = atom_ptr[i] + n
        bond_ptr[i + 1] = bond_ptr[i] + m
        junc_ptr[i + 1] = junc_ptr[i] + int(obj.junc_index.shape[0])
        atom_feats.append(np.asarray(obj.atom_feat, dtype=np.float32))
        bond_feats.append(np.asarray(obj.bond_feat, dtype=np.float32))
        bond_indices.append(np.asarray(obj.bond_index, dtype=np.int16))
        junc_feats.append(np.asarray(obj.junc_feat, dtype=np.float32))
        junc_indices.append(np.asarray(obj.junc_index, dtype=np.int16))
        junc_bonds.append(np.asarray(obj.junc_bond, dtype=np.int16))
        global_feats.append(np.asarray(obj.global_feat, dtype=np.float32))
    return {
        "n_molecules": len(objs),
        "atom_ptr": atom_ptr,
        "bond_ptr": bond_ptr,
        "junc_ptr": junc_ptr,
        "atom_feat": np.concatenate(atom_feats, axis=0),
        "atom_type": np.asarray(atom_type, dtype=np.int64),
        "bond_feat": np.concatenate(bond_feats, axis=0),
        "bond_type": np.asarray(bond_type, dtype=np.int64),
        "bond_index": np.concatenate(bond_indices, axis=0),
        "junc_feat": np.concatenate(junc_feats, axis=0),
        "junc_index": np.concatenate(junc_indices, axis=0),
        "junc_bond": np.concatenate(junc_bonds, axis=0),
        "global_feat": np.stack(global_feats, axis=0),
        "n_atoms_graph": np.diff(atom_ptr).astype(np.int16),
        "n_bonds_graph": np.diff(bond_ptr).astype(np.int16),
    }
