#!/usr/bin/env python
"""TCCD-v2 GRAD-CONT shared local-chemistry / ordinal-geometry utilities.

This module is the single place that defines, for a canonical TCCD-v0 714-D
radius-2 patch:

  * the deterministic structure-only chemistry decode (root, degree, shell-1
    atom/bond counts, shell-2 atom/bond counts, connectivity to shell 2,
    canonical radius-1 and radius-2 keys);
  * the two edit counts used by the GRAD-CONT round:
        d1 = radius-1 edit count  (audit ``c1_dist``)
        d2 = peripheral / radius-2 edit count (audit ``d2_dist``)
  * a Pareto partial order on candidate environments that never picks a
    scalar chemical distance and never weights the two shells by hand:
        j ~< k   iff  d1(i,j) <= d1(i,k) and d2(i,j) <= d2(i,k)
                      and at least one coordinate is strictly smaller;
  * deterministic cross-molecule ordinal-triplet construction and the
    held-out pair/triplet sampling used by the continuity audit.

The decoding construction is the union of the two frozen definitions:
``tracks/ksvd/code/tccd_v2_chemcont.py`` (radius-1 view, used for training) and
``tracks/ksvd/audit/tccd_v2_continuity/run_tccd_v2_continuity_audit.py``
(radius-1 + radius-2 view, used for the audit). No target ``y`` is ever read.

Nothing here trains or writes formal result files.
"""

from __future__ import annotations

import collections
import itertools
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V

PROTOCOL_VERSION = "tccd_v2_local_chem"

# --- ordinal binning (used for coverage tables AND triplet eligibility) ----
# bin value b means: b if b < MAX else ">= MAX"
D1_MAX_BIN = 4  # bins 0,1,2,3,4(>=4)
D2_MAX_BIN = 6  # bins 0,1,2,3,4,5,6(>=6)

# --- triplet eligibility constants (frozen in the pre-registration) --------
ORDINAL_SEED = T.SPLIT_SEED
MAX_TRIPLETS_PER_ANCHOR = 4
N_RANDOM_CANDIDATES = 384
N_L1_CANDIDATES = 96
FAR_D1_CAP = 3  # farther environment bin may not exceed this
FAR_D2_CAP = 5  # farther environment bin may not exceed this
SIZE_TOL = 2  # |n_atoms difference| tolerance


# ===========================================================================
# chemistry decoding
# ===========================================================================
def decode_local_chem(
    records: Sequence[Mapping[str, Any]],
    graph_indices: Sequence[int],
    *,
    layout: T.PatchLayout | None = None,
    log=print,
) -> dict[str, Any]:
    """Decode deterministic structure-only local chemistry for every patch.

    Returns flat patch-level arrays (all ``int64`` unless noted). No target
    ``y`` is read. ``graph_of`` indexes into ``graph_indices`` (0-based), not
    into ``records``.
    """
    layout = V.frozen_layout() if layout is None else layout
    cap = layout.capacity
    na = layout.n_atom
    nb = layout.n_bond
    sl = layout.block_slices()
    pi, pj = layout.pair_i, layout.pair_j
    pid = np.full((cap, cap), -1, dtype=np.int64)
    pid[pi, pj] = np.arange(len(pi))

    gid_list: list[int] = []
    key_list: list[bytes] = []
    root_l: list[int] = []
    degree_l: list[int] = []
    n_l: list[int] = []
    n2_l: list[int] = []
    l1_l: list[int] = []
    s1_l: list[np.ndarray] = []
    rb_l: list[np.ndarray] = []
    s2_l: list[np.ndarray] = []
    at_l: list[np.ndarray] = []
    s2b_l: list[np.ndarray] = []
    l1map: dict[Any, int] = {}

    def l1_key(atom, shell, edges):
        sh1 = [s for s in range(len(shell)) if shell[s] == 1]
        rb, inner = {}, {}
        for a, b, bt in edges:
            if a == 0 and b in sh1:
                rb[b] = bt
            elif b == 0 and a in sh1:
                rb[a] = bt
            elif a in sh1 and b in sh1:
                inner[(min(a, b), max(a, b))] = bt
        best = None
        for perm in itertools.permutations(sh1):
            pos = {v: i for i, v in enumerate(perm)}
            ent = []
            for v in perm:
                sub = tuple(
                    sorted(
                        (inner[(min(v, w), max(v, w))], pos[w])
                        for w in sh1
                        if w != v and (min(v, w), max(v, w)) in inner
                    )
                )
                ent.append((int(atom[v]), int(rb.get(v, -1)), sub))
            cand = (int(atom[0]), tuple(ent))
            if best is None or cand < best:
                best = cand
        return best

    t0 = time.time()
    for gpos, gi in enumerate(graph_indices):
        rec = records[int(gi)]
        n = int(rec["n"])
        X = np.asarray(rec["X"], dtype=np.float32)
        keys = rec["keys"]
        for v in range(n):
            row = X[v]
            mask = row[sl["mask"]] > 0.5
            ni = int(mask.sum())
            atom = row[sl["atom"]].reshape(cap, na).argmax(1)[:ni].astype(np.int64)
            shell = row[sl["shell"]].reshape(cap, 3).argmax(1)[:ni].astype(np.int64)
            topo = row[sl["topology"]] > 0.5
            bond = row[sl["bond"]].reshape(nb, len(pi)).argmax(0)
            edges = []
            for s in range(ni):
                for t in range(s + 1, ni):
                    q = pid[s, t]
                    if topo[q]:
                        edges.append((s, t, int(bond[q])))
            s1 = np.zeros(na, dtype=np.int64)
            rb = np.zeros(nb, dtype=np.int64)
            s2 = np.zeros(na, dtype=np.int64)
            at = np.zeros(na * nb, dtype=np.int64)
            s2b = np.zeros(nb, dtype=np.int64)
            n2 = 0
            for s in range(ni):
                if shell[s] == 1:
                    s1[int(atom[s])] += 1
                elif shell[s] == 2:
                    n2 += 1
                    s2[int(atom[s])] += 1
            for a, b, bt in edges:
                if a == 0 and shell[b] == 1:
                    rb[bt] += 1
                elif b == 0 and shell[a] == 1:
                    rb[bt] += 1
                elif (shell[a] == 1 and shell[b] == 2) or (shell[a] == 2 and shell[b] == 1):
                    s1t = int(atom[a if shell[a] == 1 else b])
                    at[s1t * nb + bt] += 1
                elif shell[a] == 2 and shell[b] == 2:
                    s2b[bt] += 1
            k = l1_key(atom, shell, edges)
            if k not in l1map:
                l1map[k] = len(l1map)
            gid_list.append(gpos)
            key_list.append(keys[v])
            root_l.append(int(atom[0]))
            degree_l.append(int(s1.sum()))
            n_l.append(ni)
            n2_l.append(n2)
            s1_l.append(s1)
            rb_l.append(rb)
            s2_l.append(s2)
            at_l.append(at)
            s2b_l.append(s2b)
            l1_l.append(l1map[k])

    key_ids = np.empty(len(key_list), dtype=np.int64)
    kmap: dict[bytes, int] = {}
    for i, k in enumerate(key_list):
        if k not in kmap:
            kmap[k] = len(kmap)
        key_ids[i] = kmap[k]

    out = {
        "n_patches": len(root_l),
        "n_graphs": len(graph_indices),
        "graph_of": np.asarray(gid_list, dtype=np.int64),
        "keys": key_list,
        "key_id": key_ids,
        "root": np.asarray(root_l, dtype=np.int64),
        "degree": np.asarray(degree_l, dtype=np.int64),
        "n_atoms": np.asarray(n_l, dtype=np.int64),
        "n2": np.asarray(n2_l, dtype=np.int64),
        "l1id": np.asarray(l1_l, dtype=np.int64),
        "s1atom": np.asarray(s1_l, dtype=np.int64),
        "rbond": np.asarray(rb_l, dtype=np.int64),
        "s2atom": np.asarray(s2_l, dtype=np.int64),
        "attach": np.asarray(at_l, dtype=np.int64),
        "s2bond": np.asarray(s2b_l, dtype=np.int64),
        "n_unique_l1": len(l1map),
        "n_unique_keys": len(kmap),
    }
    log(
        f"[chem] decoded {out['n_patches']} patches from {len(graph_indices)} graphs "
        f"({out['n_unique_l1']} radius-1 keys, {out['n_unique_keys']} canonical keys) "
        f"in {time.time() - t0:.1f}s"
    )
    return out


# ===========================================================================
# edit counts and bins
# ===========================================================================
def d1_row(chem: Mapping[str, Any], i: int, cand: np.ndarray) -> np.ndarray:
    """Radius-1 edit count between anchor ``i`` and every index in ``cand``."""
    s1 = chem["s1atom"]
    rb = chem["rbond"]
    a = np.abs(s1[cand] - s1[int(i)]).sum(1)
    b = np.abs(rb[cand] - rb[int(i)]).sum(1)
    return (0.5 * a + 0.5 * b).astype(np.int64)


def d2_row(chem: Mapping[str, Any], i: int, cand: np.ndarray) -> np.ndarray:
    """Peripheral / radius-2 edit count between anchor ``i`` and ``cand``."""
    n2 = chem["n2"]
    s2 = chem["s2atom"]
    at = chem["attach"]
    s2b = chem["s2bond"]
    a = np.abs(n2[cand] - n2[int(i)])
    b = np.abs(s2[cand] - s2[int(i)]).sum(1)
    c = np.abs(at[cand] - at[int(i)]).sum(1)
    d = np.abs(s2b[cand] - s2b[int(i)]).sum(1)
    return (a + 0.5 * b + 0.5 * c + 0.5 * d).astype(np.int64)


def d1_pairs(chem: Mapping[str, Any], i: np.ndarray, j: np.ndarray) -> np.ndarray:
    s1 = chem["s1atom"]
    rb = chem["rbond"]
    a = np.abs(s1[i] - s1[j]).sum(1)
    b = np.abs(rb[i] - rb[j]).sum(1)
    return (0.5 * a + 0.5 * b).astype(np.int64)


def d2_pairs(chem: Mapping[str, Any], i: np.ndarray, j: np.ndarray) -> np.ndarray:
    n2 = chem["n2"]
    s2 = chem["s2atom"]
    at = chem["attach"]
    s2b = chem["s2bond"]
    a = np.abs(n2[i] - n2[j])
    b = np.abs(s2[i] - s2[j]).sum(1)
    c = np.abs(at[i] - at[j]).sum(1)
    d = np.abs(s2b[i] - s2b[j]).sum(1)
    return (a + 0.5 * b + 0.5 * c + 0.5 * d).astype(np.int64)


def bin_d1(v: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(v, dtype=np.int64), D1_MAX_BIN)


def bin_d2(v: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(v, dtype=np.int64), D2_MAX_BIN)


def is_exact(chem: Mapping[str, Any], i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Exact pair: canonical key equal OR zero total edit count."""
    same_key = chem["key_id"][i] == chem["key_id"][j]
    zero = (d1_pairs(chem, i, j) == 0) & (d2_pairs(chem, i, j) == 0)
    return same_key | zero


def pareto_compare(d1a, d2a, d1b, d2b):
    """Return True where (d1a,d2a) strictly Pareto-below (d1b,d2b)."""
    le = (d1a <= d1b) & (d2a <= d2b)
    strict = (d1a < d1b) | (d2a < d2b)
    return le & strict


def comparison_type(d1a, d2a, d1b, d2b):
    if d1a == d1b:
        return "same_d1"
    if d2a == d2b:
        return "same_d2"
    return "both"


# ===========================================================================
# ordinal triplet construction (deterministic, cross-molecule, train/dev)
# ===========================================================================
def _group_indices(keys: np.ndarray) -> dict[Any, np.ndarray]:
    groups: dict[Any, list[int]] = collections.defaultdict(list)
    for i in range(len(keys)):
        groups[int(keys[i])].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def _sample_pool(pool: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(pool) <= cap:
        return pool
    return pool[rng.choice(len(pool), size=cap, replace=False)]


def build_ordinal_triplets(
    chem: Mapping[str, Any],
    *,
    seed: int = ORDINAL_SEED,
    max_per_anchor: int = MAX_TRIPLETS_PER_ANCHOR,
    n_random_candidates: int = N_RANDOM_CANDIDATES,
    n_l1_candidates: int = N_L1_CANDIDATES,
    far_d1_cap: int = FAR_D1_CAP,
    far_d2_cap: int = FAR_D2_CAP,
    size_tol: int = SIZE_TOL,
    log=print,
) -> dict[str, Any]:
    """Build deterministic cross-molecule Pareto ordinal triplets.

    For anchor ``i`` the candidate pool is the set of decoded patches with the
    same root atom category and the same root degree, in a different molecule,
    with ``|n_atoms - n_atoms_i| <= size_tol``. Candidates are drawn as the
    union of a uniform sample of the ``(root, degree)`` group and a sample of
    the radius-1-identical group (so short radius-2 chains are not missed). For
    every candidate ``d1``/``d2`` are computed exactly. Bins are formed with
    the frozen caps, and every strictly Pareto-ordered bin pair ``A ~< B`` with
    the farther bin bounded by ``(far_d1_cap, far_d2_cap)`` is a legal ordinal
    comparison. The nearest valid bin pairs are consumed first.

    Returns flat arrays ``anchor``, ``closer``, ``farther`` (indices into the
    decoded patch arrays), ``anchor_pos`` (graph position, for per-batch
    gathering), ``ctype`` (0/1/2 = same_d1/same_d2/both), the realized
    ``d1_close``/``d2_close``/``d1_far``/``d2_far``, and a statistics dict.
    """
    P = int(chem["n_patches"])
    graph_of = chem["graph_of"]
    root = chem["root"]
    degree = chem["degree"]
    n_atoms = chem["n_atoms"]
    l1id = chem["l1id"]
    key_id = chem["key_id"]

    rd_groups = _group_indices(root * 1000 + degree)
    l1_groups = _group_indices(l1id)

    rng = np.random.default_rng(int(seed))
    A: list[int] = []
    Cl: list[int] = []
    Fa: list[int] = []
    Ap: list[int] = []
    Ct: list[int] = []
    D1c: list[int] = []
    D2c: list[int] = []
    D1f: list[int] = []
    D2f: list[int] = []
    bin_pair_hist: collections.Counter = collections.Counter()
    type_hist: collections.Counter = collections.Counter()
    cand_bin_hist: collections.Counter = collections.Counter()
    n_eligible = 0
    n_no_candidate = 0

    t0 = time.time()
    for i in range(P):
        pool = rd_groups.get(int(root[i]) * 1000 + int(degree[i]))
        if pool is None or len(pool) < 2:
            continue
        pool = pool[graph_of[pool] != graph_of[i]]
        if len(pool) < 2:
            continue
        rand = _sample_pool(pool, n_random_candidates, rng)
        l1pool = l1_groups.get(int(l1id[i]))
        if l1pool is not None and len(l1pool) > 0:
            l1pool = l1pool[graph_of[l1pool] != graph_of[i]]
            l1pool = l1pool[l1pool != i]
            l1pool = _sample_pool(l1pool, n_l1_candidates, rng)
            cand = np.unique(np.concatenate([rand, l1pool]))
        else:
            cand = np.unique(rand)

        d1 = d1_row(chem, i, cand)
        d2 = d2_row(chem, i, cand)
        # candidate legality: not exact, bounded size
        legal = ~(is_exact(chem, np.full_like(cand, i), cand))
        legal &= np.abs(n_atoms[cand] - n_atoms[i]) <= size_tol
        cand = cand[legal]
        d1 = d1[legal]
        d2 = d2[legal]
        if cand.size < 2:
            n_no_candidate += 1
            continue
        b1 = bin_d1(d1)
        b2 = bin_d2(d2)
        for a, b in zip(b1.tolist(), b2.tolist()):
            cand_bin_hist[(a, b)] += 1

        # bucket candidate local positions by bin
        buckets: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        for pos in range(cand.size):
            buckets[(int(b1[pos]), int(b2[pos]))].append(pos)

        # enumerate legal Pareto bin pairs, nearest-farther first
        pairs = []
        for (a1, a2), la in buckets.items():
            for (f1, f2), lf in buckets.items():
                if (a1, a2) == (f1, f2):
                    continue
                if f1 > far_d1_cap or f2 > far_d2_cap:
                    continue
                if pareto_compare(a1, a2, f1, f2):
                    pairs.append(((f1 + f2, a1 + a2), (a1, a2), (f1, f2), la, lf))
        if not pairs:
            n_no_candidate += 1
            continue
        pairs.sort(key=lambda t: t[0])

        used_j = set()
        used_k = set()
        made = 0
        for _, (a1, a2), (f1, f2), la, lf in pairs:
            if made >= max_per_anchor:
                break
            # shuffle candidate choices for this bin pair
            jl = la[:]
            kl = lf[:]
            rng.shuffle(jl)
            rng.shuffle(kl)
            for jp in jl:
                if made >= max_per_anchor:
                    break
                if jp in used_j:
                    continue
                for kp in kl:
                    if made >= max_per_anchor:
                        break
                    if kp == jp or kp in used_k:
                        continue
                    ji = int(cand[jp])
                    ki = int(cand[kp])
                    if graph_of[ji] == graph_of[ki]:
                        continue
                    a1v, a2v = int(d1[jp]), int(d2[jp])
                    f1v, f2v = int(d1[kp]), int(d2[kp])
                    if not pareto_compare(a1v, a2v, f1v, f2v):
                        continue
                    A.append(i)
                    Cl.append(ji)
                    Fa.append(ki)
                    Ap.append(int(graph_of[i]))
                    Ct.append({"same_d1": 0, "same_d2": 1, "both": 2}[comparison_type(a1v, a2v, f1v, f2v)])
                    D1c.append(a1v)
                    D2c.append(a2v)
                    D1f.append(f1v)
                    D2f.append(f2v)
                    bin_pair_hist[(f"{a1},{a2}", f"{f1},{f2}")] += 1
                    type_hist[comparison_type(a1v, a2v, f1v, f2v)] += 1
                    used_j.add(jp)
                    used_k.add(kp)
                    made += 1
                    break
        if made > 0:
            n_eligible += 1

    anchor = np.asarray(A, dtype=np.int64)
    closer = np.asarray(Cl, dtype=np.int64)
    farther = np.asarray(Fa, dtype=np.int64)
    anchor_pos = np.asarray(Ap, dtype=np.int64)
    ctype = np.asarray(Ct, dtype=np.int64)
    d1c = np.asarray(D1c, dtype=np.int64)
    d2c = np.asarray(D2c, dtype=np.int64)
    d1f = np.asarray(D1f, dtype=np.int64)
    d2f = np.asarray(D2f, dtype=np.int64)

    cross = float((graph_of[closer] != graph_of[anchor]).mean()) if anchor.size else 0.0
    cross2 = float((graph_of[farther] != graph_of[anchor]).mean()) if anchor.size else 0.0
    cross3 = float((graph_of[closer] != graph_of[farther]).mean()) if anchor.size else 0.0
    stats: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "seed": int(seed),
        "n_patches": P,
        "n_anchors_total": P,
        "n_anchors_eligible": int(n_eligible),
        "n_anchors_ineligible": int(P - n_eligible),
        "eligible_fraction": float(n_eligible / max(P, 1)),
        "n_triplets": int(anchor.size),
        "max_triplets_per_anchor": int(max_per_anchor),
        "n_random_candidates": int(n_random_candidates),
        "n_l1_candidates": int(n_l1_candidates),
        "far_d1_cap": int(far_d1_cap),
        "far_d2_cap": int(far_d2_cap),
        "size_tol": int(size_tol),
        "graph_coverage": float(np.unique(anchor_pos).size / max(int(chem["n_graphs"]), 1))
        if anchor.size
        else 0.0,
        "cross_molecule_anchor_closer": cross,
        "cross_molecule_anchor_farther": cross2,
        "cross_molecule_closer_farther": cross3,
        "comparison_type_hist": dict(type_hist),
        "top_bin_pairs": [
            {"closer_bin": k[0], "farther_bin": k[1], "n": int(v)}
            for k, v in sorted(bin_pair_hist.items(), key=lambda kv: -kv[1])[:20]
        ],
        "candidate_bin_hist": {
            f"{a},{b}": int(v) for (a, b), v in sorted(cand_bin_hist.items())
        },
        "closer_d1_hist": _hist(d1c),
        "closer_d2_hist": _hist(d2c),
        "farther_d1_hist": _hist(d1f),
        "farther_d2_hist": _hist(d2f),
        "official_test_loaded": False,
        "uses_target_label": False,
        "wall_s": time.time() - t0,
    }
    log(
        f"[ordinal] {stats['n_triplets']} triplets from {stats['n_anchors_eligible']}/"
        f"{P} anchors ({stats['eligible_fraction']:.3f}), types={dict(type_hist)}, "
        f"{time.time() - t0:.1f}s"
    )
    return {
        "anchor": anchor,
        "closer": closer,
        "farther": farther,
        "anchor_pos": anchor_pos,
        "ctype": ctype,
        "d1_close": d1c,
        "d2_close": d2c,
        "d1_far": d1f,
        "d2_far": d2f,
        "stats": stats,
    }


def _hist(v: np.ndarray) -> dict[str, int]:
    if v.size == 0:
        return {}
    u, c = np.unique(v, return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(u, c)}


# ===========================================================================
# bin-pair sampling for geometry curves (same pairs per checkpoint)
# ===========================================================================
def _sample_within_groups(groups, target: int, rng: np.random.Generator):
    """Sample ~``target`` unordered pairs, uniform-over-pairs across groups.

    Each group contributes a number of pairs proportional to its pair count, so
    the pooled sample is uniform over the union of all within-group pairs (the
    same weighting the frozen continuity audit used).
    """
    groups = [np.asarray(g, dtype=np.int64) for g in groups if len(g) >= 2]
    if not groups:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    sizes = np.array([len(g) for g in groups], dtype=np.int64)
    w = sizes * (sizes - 1) // 2
    W = int(w.sum())
    if W <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    I, J = [], []
    for g, wg in zip(groups, w):
        if wg < 1:
            continue
        n_g = int(round(target * float(wg) / W))
        n_g = max(1, min(n_g, int(wg)))
        m = len(g)
        cand = n_g + max(20, n_g // 4)
        a = rng.integers(0, m, cand)
        b = rng.integers(0, m, cand)
        ok = a != b
        a, b = a[ok], b[ok]
        if a.size == 0:
            continue
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        kk = lo.astype(np.int64) * m + hi
        _, u = np.unique(kk, return_index=True)
        lo, hi = lo[u][:n_g], hi[u][:n_g]
        I.append(g[lo])
        J.append(g[hi])
    if not I:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(I), np.concatenate(J)


def sample_bin_pairs(
    chem: Mapping[str, Any],
    *,
    seed: int = ORDINAL_SEED,
    target_pairs: int = 1_500_000,
    exact_target: int = 200_000,
    size_tol: int = SIZE_TOL,
    log=print,
) -> dict[str, Any]:
    """Sample cross-molecule anchor-partner dev pairs binned by ``(d1,d2)``.

    Sampling is **uniform over pairs**: within each ``(root, degree)`` group the
    number of sampled pairs is proportional to the group's pair count. This is
    the same weighting the frozen continuity audit used, so the low-``d2``
    numbers are directly comparable to the frozen BASE / CHEM-CONT values.
    Same-canonical-key (exact) pairs are sampled separately, uniform over
    pairs. Zero-edit partners with a different canonical key stay in the
    non-same-key stream and fall into bin ``(0,0)``. The same returned pair set
    is evaluated for every checkpoint, so the curve comparison is paired.
    """
    graph_of = chem["graph_of"]
    root = chem["root"]
    degree = chem["degree"]
    n_atoms = chem["n_atoms"]
    key_id = chem["key_id"]

    rd_groups = _group_indices(root * 1000 + degree)
    key_groups = _group_indices(key_id)

    rng = np.random.default_rng(int(seed))
    t0 = time.time()
    # non-same-key pairs (oversample to absorb graph/size filters)
    ni, nj = _sample_within_groups(list(rd_groups.values()), int(target_pairs * 1.4), rng)
    keep = (graph_of[ni] != graph_of[nj]) & (key_id[ni] != key_id[nj])
    keep &= np.abs(n_atoms[ni] - n_atoms[nj]) <= size_tol
    ni, nj = ni[keep], nj[keep]
    if ni.size > target_pairs:
        sel = rng.permutation(ni.size)[:target_pairs]
        ni, nj = ni[sel], nj[sel]
    # same-canonical-key (exact) pairs
    ei, ej = _sample_within_groups(list(key_groups.values()), int(exact_target * 1.4), rng)
    ekeep = (graph_of[ei] != graph_of[ej]) & (np.abs(n_atoms[ei] - n_atoms[ej]) <= size_tol)
    ei, ej = ei[ekeep], ej[ekeep]
    if ei.size > exact_target:
        sel = rng.permutation(ei.size)[:exact_target]
        ei, ej = ei[sel], ej[sel]

    anchor = np.concatenate([ni, ei])
    partner = np.concatenate([nj, ej])
    is_same_key = np.concatenate([np.zeros(ni.size, np.int64), np.ones(ei.size, np.int64)])
    cross = float((graph_of[anchor] != graph_of[partner]).mean()) if anchor.size else 0.0
    log(
        f"[bin-pairs] {anchor.size} cross-molecule pairs "
        f"(non_same_key={ni.size}, same_key={ei.size}, cross={cross:.4f}) in {time.time() - t0:.1f}s"
    )
    return {
        "anchor": anchor,
        "partner": partner,
        "same_key": is_same_key,
        "stats": {
            "n_pairs": int(anchor.size),
            "n_same_key": int(is_same_key.sum()),
            "n_non_same_key": int((is_same_key == 0).sum()),
            "cross_molecule_rate": cross,
            "target_pairs": int(target_pairs),
            "exact_target": int(exact_target),
            "weighting": "uniform_over_pairs",
            "official_test_loaded": False,
            "uses_target_label": False,
            "wall_s": time.time() - t0,
        },
    }


def save_triplets(triplets: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        anchor=triplets["anchor"],
        closer=triplets["closer"],
        farther=triplets["farther"],
        anchor_pos=triplets["anchor_pos"],
        ctype=triplets["ctype"],
    )
    return path


def load_triplets(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    obj = np.load(path)
    return {
        "anchor": obj["anchor"],
        "closer": obj["closer"],
        "farther": obj["farther"],
        "anchor_pos": obj["anchor_pos"],
        "ctype": obj["ctype"],
    }
