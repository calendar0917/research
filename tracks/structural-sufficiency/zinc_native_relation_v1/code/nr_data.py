"""Compact, label-blind relation cache for zinc_native_relation_v1.

The frozen P3 object cache already contains the atom / bond / junction objects
and their all-pairs distance matrices.  This module adds **only** the relation
arrays between tokens that already exist in P3:

    token 0 .. n_junc-1          junction, anchor = centre atom
    token n_junc .. n_junc+nb-1  original bond, anchor = its two endpoints

Pair set (identical to the earlier HR round, whose arrays are replayed as an
audit): all unordered token pairs with clipped anchor distance <= 3 **or** a
shared biconnected ring block.  Relation id:

    rel = 1 + 8 * min(distance, 3) + 4 * same_ring_block + min(A^2, 3)

Self loops are not stored here; the collator adds one rel-0 self loop per token.

The null arm keeps the pair set, role pair (JJ / JB / BB) and the per-stratum
relation multiset, and permutes relation labels among non-self pairs **within
each molecule and each role stratum** with a fixed RNG.  It never reads labels,
predictions or errors and is cached once.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))

import ad_common as AC
import ad_data as AD
import ad_train as AT

import nr_common as C

REL_SELF = 0
REL_TYPES = 33
DIST_CAP = 3
WALK2_CAP = 3
#: fixed RNG namespace for the relation-label permutation; independent of data
NULL_SEED = 20261107
#: initialisation namespace for the new relation module (independent of R/N)
REL_INIT_BASE = 80261107

ROLE_LABELS = {(0, 0): "JJ", (0, 1): "JB", (1, 1): "BB"}


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------


def _ring_blocks(n_atoms: int, bonds: Sequence[tuple[int, int]]) -> list[set[int]]:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(int(n_atoms)))
    graph.add_edges_from((int(u), int(v)) for u, v in bonds)
    return [set(b) for b in nx.biconnected_components(graph) if len(b) >= 3]


def _atom_blocks(n_atoms: int, bonds: Sequence[tuple[int, int]]) -> list[set[int]]:
    blocks = _ring_blocks(n_atoms, bonds)
    out: list[set[int]] = [set() for _ in range(int(n_atoms))]
    for bi, block in enumerate(blocks):
        for a in block:
            out[a].add(bi)
    return out


def relation_labels(
    dist: np.ndarray,
    n_atoms: int,
    bond_index: np.ndarray,
    junc_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the HR relation arrays from the frozen P3 objects.

    ``dist`` is the P3 cached all-pairs distance matrix for this molecule.
    """
    bonds = [(int(u), int(v)) for u, v in np.asarray(bond_index)]
    n_bonds = len(bonds)
    nj = int(np.asarray(junc_index).shape[0])
    atom_blocks = _atom_blocks(int(n_atoms), bonds)
    adjacency = (np.asarray(dist) == 1).astype(np.int64)
    walk2 = adjacency @ adjacency

    anchors: list[tuple[int, ...]] = [(int(x),) for x in np.asarray(junc_index)[:, 1]]
    anchors += [(u, v) for u, v in bonds]

    n_tokens = nj + n_bonds
    r_u: list[int] = []
    r_v: list[int] = []
    r_rel: list[int] = []
    for a in range(n_tokens):
        for b in range(a + 1, n_tokens):
            d = min(int(dist[x, y]) for x in anchors[a] for y in anchors[b])
            same = bool(
                set().union(*(atom_blocks[x] for x in anchors[a]))
                & set().union(*(atom_blocks[x] for x in anchors[b]))
            )
            w2 = min(int(walk2[x, y]) for x in anchors[a] for y in anchors[b])
            if d <= DIST_CAP or same:
                d_b = min(max(d, 0), DIST_CAP)
                w_b = min(max(w2, 0), WALK2_CAP)
                r_u.append(a)
                r_v.append(b)
                r_rel.append(int(1 + 8 * d_b + 4 * int(same) + w_b))
    return (
        np.asarray(r_u, dtype=np.int32),
        np.asarray(r_v, dtype=np.int32),
        np.asarray(r_rel, dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# null permutation
# ---------------------------------------------------------------------------


def _entropy(labels: np.ndarray) -> tuple[float, float]:
    m = len(labels)
    if m == 0:
        return 0.0, 0.0
    counts = np.bincount(labels.astype(np.int64))
    p = counts[counts > 0].astype(np.float64) / float(m)
    h = float(-(p * np.log(p)).sum())
    h_norm = float(h / math.log(m)) if m > 1 else 0.0
    return h, h_norm


def null_relation_labels(
    r_u: np.ndarray,
    r_v: np.ndarray,
    r_rel: np.ndarray,
    n_junc: int,
    mol_index: int,
    *,
    null_seed: int = NULL_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Permute relation labels inside each (molecule, role-pair) stratum.

    A random permutation is drawn from a fixed per-stratum RNG.  If it happens
    to be the identity and the stratum has at least two pairs, a cyclic shift
    by one is used instead, so the null is a real change wherever the stratum
    is not label-homogeneous.  Bidirectional edges carry one label, so the
    permutation is applied once to the stored undirected pair.
    """
    r_u = np.asarray(r_u, dtype=np.int64)
    r_v = np.asarray(r_v, dtype=np.int64)
    r_rel = np.asarray(r_rel, dtype=np.int64)
    out = r_rel.copy()
    role_u = (r_u >= int(n_junc)).astype(np.int64)
    role_v = (r_v >= int(n_junc)).astype(np.int64)
    stats: dict[str, Any] = {
        "n_non_self_pairs": int(len(r_rel)),
        "n_changed": 0,
        "n_permutable_groups": 0,
        "n_singleton_groups": 0,
        "n_homogeneous_groups": 0,
        "n_forced_nonidentity": 0,
        "n_role_groups": 0,
        "entropy_sum": 0.0,
        "entropy_pairs_weighted_sum": 0.0,
        "homogeneous_pairs": 0,
        "role_groups": {},
    }
    for role_pair, name in ROLE_LABELS.items():
        mask = (role_u == role_pair[0]) & (role_v == role_pair[1])
        idx = np.flatnonzero(mask)
        m = int(len(idx))
        group: dict[str, Any] = {"n_pairs": m}
        if m == 0:
            stats["role_groups"][name] = group
            continue
        stats["n_role_groups"] += 1
        labels = r_rel[idx]
        h, h_norm = _entropy(labels)
        unique = np.unique(labels)
        group.update(
            {
                "n_unique_labels": int(len(unique)),
                "entropy": h,
                "entropy_normalized": h_norm,
                "homogeneous": bool(len(unique) == 1),
            }
        )
        stats["entropy_sum"] += h
        stats["entropy_pairs_weighted_sum"] += h * m
        if m < 2:
            stats["n_singleton_groups"] += 1
        else:
            stats["n_permutable_groups"] += 1
            rng = np.random.default_rng([int(null_seed), int(mol_index), int(role_pair[0] * 2 + role_pair[1])])
            perm = rng.permutation(m)
            if np.array_equal(perm, np.arange(m)):
                perm = np.roll(np.arange(m), 1)
                stats["n_forced_nonidentity"] += 1
            group["n_changed"] = int((labels[perm] != labels).sum())
            out[idx] = labels[perm]
        if len(unique) == 1:
            stats["n_homogeneous_groups"] += 1
            group["homogeneous"] = True
            stats["homogeneous_pairs"] += m
        stats["role_groups"][name] = group
    stats["n_changed"] = int((out != r_rel).sum())
    stats["changed_fraction"] = float(stats["n_changed"] / max(len(r_rel), 1))
    stats["entropy_mean_per_group"] = float(
        stats["entropy_sum"] / max(stats["n_role_groups"], 1)
    )
    stats["entropy_mean_pairs_weighted"] = float(
        stats["entropy_pairs_weighted_sum"] / max(len(r_rel), 1)
    )
    stats["homogeneous_pair_fraction"] = float(stats["homogeneous_pairs"] / max(len(r_rel), 1))
    return out.astype(np.int8), stats


# ---------------------------------------------------------------------------
# cache build
# ---------------------------------------------------------------------------


def build_entry(cache: Mapping[str, Any], mol_index: int) -> dict[str, Any]:
    a0, a1 = int(cache["atom_ptr"][mol_index]), int(cache["atom_ptr"][mol_index + 1])
    b0, b1 = int(cache["bond_ptr"][mol_index]), int(cache["bond_ptr"][mol_index + 1])
    j0, j1 = int(cache["junc_ptr"][mol_index]), int(cache["junc_ptr"][mol_index + 1])
    n_atoms = a1 - a0
    bond_index = np.asarray(cache["bond_index"][b0:b1], dtype=np.int64)
    junc_index = np.asarray(cache["junc_index"][j0:j1], dtype=np.int64)
    ptr = int(cache["dist_ptr"][mol_index]), int(cache["dist_ptr"][mol_index + 1])
    dist = np.asarray(cache["dist_flat"][ptr[0] : ptr[1]], dtype=np.int64).reshape(n_atoms, n_atoms)
    r_u, r_v, r_rel = relation_labels(dist, n_atoms, bond_index, junc_index)
    r_null, null_stats = null_relation_labels(r_u, r_v, r_rel, len(junc_index), mol_index)
    return {
        "mol_index": int(mol_index),
        "n_atoms": int(n_atoms),
        "n_bonds": int(len(bond_index)),
        "n_junc": int(len(junc_index)),
        "n_tokens": int(len(junc_index) + len(bond_index)),
        "r_u": r_u,
        "r_v": r_v,
        "r_rel": r_rel,
        "r_rel_null": r_null,
        "null_stats": null_stats,
    }


def build_split(split: str) -> list[dict[str, Any]]:
    cache = AD.load_cache(split)
    n = int(cache["n_molecules"])
    entries: list[dict[str, Any]] = []
    for i in range(n):
        entries.append(build_entry(cache, i))
    return entries


def relation_fingerprint(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    def _hash(key: str) -> str:
        chunks = []
        for e in entries:
            arr = np.ascontiguousarray(e[key])
            chunks.append(str(arr.shape).encode())
            chunks.append(arr.tobytes())
        return C.sha256_bytes(chunks)

    return {
        "pair_u_sha256": _hash("r_u"),
        "pair_v_sha256": _hash("r_v"),
        "real_rel_sha256": _hash("r_rel"),
        "null_rel_sha256": _hash("r_rel_null"),
    }


def aggregate_null_stats(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_pairs = sum(int(e["null_stats"]["n_non_self_pairs"]) for e in entries)
    n_changed = sum(int(e["null_stats"]["n_changed"]) for e in entries)
    n_perm = sum(int(e["null_stats"]["n_permutable_groups"]) for e in entries)
    n_single = sum(int(e["null_stats"]["n_singleton_groups"]) for e in entries)
    n_hom = sum(int(e["null_stats"]["n_homogeneous_groups"]) for e in entries)
    n_forced = sum(int(e["null_stats"]["n_forced_nonidentity"]) for e in entries)
    n_groups = sum(int(e["null_stats"]["n_role_groups"]) for e in entries)
    ent_sum = sum(float(e["null_stats"]["entropy_sum"]) for e in entries)
    ent_w = sum(float(e["null_stats"]["entropy_pairs_weighted_sum"]) for e in entries)
    hom_pairs = sum(int(e["null_stats"]["homogeneous_pairs"]) for e in entries)
    by_role: dict[str, Any] = {}
    for name in ("JJ", "JB", "BB"):
        gp = sum(int(e["null_stats"]["role_groups"].get(name, {}).get("n_pairs", 0)) for e in entries)
        gc = sum(int(e["null_stats"]["role_groups"].get(name, {}).get("n_changed", 0)) for e in entries)
        gperm = sum(1 for e in entries if int(e["null_stats"]["role_groups"].get(name, {}).get("n_pairs", 0)) >= 2)
        ghom = sum(1 for e in entries if e["null_stats"]["role_groups"].get(name, {}).get("homogeneous", False))
        by_role[name] = {
            "n_pairs": gp,
            "n_changed": gc,
            "changed_fraction": float(gc / max(gp, 1)),
            "n_molecule_groups_permutable": gperm,
            "n_molecule_groups_homogeneous": ghom,
        }
    return {
        "n_molecules": len(entries),
        "n_non_self_pairs": int(n_pairs),
        "n_changed_pairs": int(n_changed),
        "changed_fraction": float(n_changed / max(n_pairs, 1)),
        "n_role_groups": int(n_groups),
        "n_permutable_groups": int(n_perm),
        "n_singleton_groups": int(n_single),
        "n_homogeneous_groups": int(n_hom),
        "homogeneous_pair_fraction": float(hom_pairs / max(n_pairs, 1)),
        "n_forced_nonidentity": int(n_forced),
        "entropy_mean_per_group": float(ent_sum / max(n_groups, 1)),
        "entropy_mean_pairs_weighted": float(ent_w / max(n_pairs, 1)),
        "by_role_pair": by_role,
    }


def save_cache(entries: Sequence[Mapping[str, Any]], path: Path) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(entries), path)
    return C.sha256_file(path)


def load_cache(path: Path) -> list[dict[str, Any]]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------


def collate_relations(
    entries: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    rows: Sequence[int],
) -> dict[str, Any]:
    """Build the P3 batch-local junction/bond token ids plus relation arrays.

    Token ids are global over the type-flat layout, exactly as in P3:
    junction token ``i`` keeps its flat junction row; bond token ``b`` becomes
    ``n_junctions_batch + b``.  Both directions of every stored pair and one
    rel-0 self loop per token are appended.
    """
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    n_graphs = len(rows)
    nb = np.diff(np.asarray(cache["bond_ptr"]))[rows]
    nj = np.diff(np.asarray(cache["junc_ptr"]))[rows]
    boff = np.concatenate([[0], np.cumsum(nb)[:-1]]).astype(np.int64)
    joff = np.concatenate([[0], np.cumsum(nj)[:-1]]).astype(np.int64)
    n_junc_total = int(nj.sum())
    n_bonds_total = int(nb.sum())

    pair_counts = np.asarray([len(entries[int(r)]["r_u"]) for r in rows], dtype=np.int64)
    r_mol = np.repeat(np.arange(n_graphs, dtype=np.int64), pair_counts)
    if pair_counts.sum():
        r_u = np.concatenate([np.asarray(entries[int(r)]["r_u"], dtype=np.int64) for r in rows])
        r_v = np.concatenate([np.asarray(entries[int(r)]["r_v"], dtype=np.int64) for r in rows])
        r_rel = np.concatenate([np.asarray(entries[int(r)]["r_rel"], dtype=np.int64) for r in rows])
        r_reln = np.concatenate([np.asarray(entries[int(r)]["r_rel_null"], dtype=np.int64) for r in rows])
    else:
        r_u = r_v = r_rel = r_reln = np.zeros(0, dtype=np.int64)

    nj_m = nj[r_mol]
    boff_m = boff[r_mol]
    joff_m = joff[r_mol]
    gid_u = np.where(r_u < nj_m, joff_m + r_u, n_junc_total + boff_m + (r_u - nj_m))
    gid_v = np.where(r_v < nj_m, joff_m + r_v, n_junc_total + boff_m + (r_v - nj_m))
    src = np.concatenate([gid_u, gid_v])
    dst = np.concatenate([gid_v, gid_u])
    rel = np.concatenate([r_rel, r_rel])
    reln = np.concatenate([r_reln, r_reln])

    self_parts = [joff[i] + np.arange(nj[i], dtype=np.int64) for i in range(n_graphs)]
    self_parts += [n_junc_total + boff[i] + np.arange(nb[i], dtype=np.int64) for i in range(n_graphs)]
    self_ids = np.concatenate(self_parts) if self_parts else np.zeros(0, dtype=np.int64)
    src = np.concatenate([src, self_ids])
    dst = np.concatenate([dst, self_ids])
    rel = np.concatenate([rel, np.zeros(len(self_ids), dtype=np.int64)])
    reln = np.concatenate([reln, np.zeros(len(self_ids), dtype=np.int64)])

    return {
        "n_junc_total": int(n_junc_total),
        "n_bonds_total": int(n_bonds_total),
        "n_tokens_total": int(n_junc_total + n_bonds_total),
        "r_src": torch.as_tensor(np.ascontiguousarray(src), dtype=torch.long),
        "r_dst": torch.as_tensor(np.ascontiguousarray(dst), dtype=torch.long),
        "r_rel": torch.as_tensor(np.ascontiguousarray(rel), dtype=torch.long),
        "r_rel_null": torch.as_tensor(np.ascontiguousarray(reln), dtype=torch.long),
    }


def _relation_batch(entries, cache, rows, candidate, valid: bool, walk_cols):
    if candidate not in ("P3N", "P", "R", "N"):
        raise ValueError(f"relation collate expects a P3-family candidate, got {candidate!r}")
    batch = AT.collate(cache, rows, "P3N", valid=valid)
    batch.update(collate_relations(entries, cache, rows))
    return batch


def make_collate(entries: Sequence[Mapping[str, Any]], *, split: str | None = None):
    """Return a collate function with the exact P3 batch plus relation arrays.

    ``split`` is optional for one-split callers and asserted when given.
    Training must use :func:`make_split_collate` so the trainer's one collate
    callback can serve both the train and valid caches.
    """

    def collate_with_relations(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        if split is not None and str(cache["meta"].get("split")) != split:
            raise ValueError(f"collate bound to {split!r}, got {cache['meta'].get('split')!r}")
        return _relation_batch(entries, cache, rows, candidate, valid, walk_cols)

    return collate_with_relations


def make_split_collate(entries_train: Sequence[Mapping[str, Any]],
                       entries_valid: Sequence[Mapping[str, Any]]):
    """Collate that dispatches on cache meta split; used by the shared trainer."""

    def collate_with_relations(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        split = str(cache["meta"].get("split"))
        if split == "train":
            entries = entries_train
        elif split == "valid":
            entries = entries_valid
        else:
            raise ValueError(f"unexpected split {split!r} in relation collate")
        return _relation_batch(entries, cache, rows, candidate, valid, walk_cols)

    return collate_with_relations


# ---------------------------------------------------------------------------
# optional replay audit against the old HR cache (evidence only)
# ---------------------------------------------------------------------------


def replay_against_hr(hr_cache_path: Path, split: str, *, limit: int | None = None) -> dict[str, Any]:
    import torch

    cache = AD.load_cache(split)
    old = torch.load(hr_cache_path, map_location="cpu", weights_only=False)
    n = len(old) if limit is None else min(int(limit), len(old))
    mismatch = {"r_u": 0, "r_v": 0, "r_rel": 0, "token_count": 0}
    max_tokens = 0
    for i in range(n):
        e = build_entry(cache, i)
        o = old[i]
        if not np.array_equal(e["r_u"], np.asarray(o["r_u"])):
            mismatch["r_u"] += 1
        if not np.array_equal(e["r_v"], np.asarray(o["r_v"])):
            mismatch["r_v"] += 1
        if not np.array_equal(e["r_rel"], np.asarray(o["r_rel"])):
            mismatch["r_rel"] += 1
        if int(e["n_tokens"]) != int(o["n_tokens"]):
            mismatch["token_count"] += 1
        max_tokens = max(max_tokens, int(e["n_tokens"]))
    return {
        "source": str(hr_cache_path),
        "split": split,
        "rows_checked": int(n),
        "mismatch_rows": mismatch,
        "max_tokens": max_tokens,
        "all_match": bool(all(v == 0 for v in mismatch.values())),
    }
