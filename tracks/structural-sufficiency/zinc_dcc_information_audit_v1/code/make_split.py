#!/usr/bin/env python
"""Freeze the label-free Fit8k / ProbeFit1k / AuditEval1k split.

Structure-only: WL content hash, exact isomorphism check for hash collisions,
and strata by n_atoms x n_ring_systems.  Exact duplicate structures are all
assigned to Fit so that no duplicate can leak into ProbeFit/AuditEval.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audit_common as AC  # noqa: E402

SIZE_BINS = ((0, 20), (21, 24), (25, 28), (29, 10**9))
RING_BINS = ((0, 0), (1, 3), (4, 10**9))
TARGETS = {"fit": 8000, "probe": 1000, "audit": 1000}


def _graph(cache, i):
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    at = np.asarray(cache["atom_type"])
    bi = np.asarray(cache["bond_index"])
    bt = np.asarray(cache["bond_type"])
    g = nx.Graph()
    atoms = at[ap[i]:ap[i + 1]]
    for j, a in enumerate(atoms):
        g.add_node(int(j), atom=int(a))
    for (u, v), t in zip(bi[bp[i]:bp[i + 1]], bt[bp[i]:bp[i + 1]]):
        g.add_edge(int(u), int(v), bond=int(t))
    return g, len(atoms), int(bp[i + 1] - bp[i])


def _size_bin(n: int) -> int:
    for k, (lo, hi) in enumerate(SIZE_BINS):
        if lo <= n <= hi:
            return k
    raise ValueError(n)


def _ring_bin(n: int) -> int:
    for k, (lo, hi) in enumerate(RING_BINS):
        if lo <= n <= hi:
            return k
    raise ValueError(n)


def _content_key(g, n_atoms: int, n_bonds: int) -> str:
    h = nx.weisfeiler_lehman_graph_hash(g, node_attr="atom", edge_attr="bond", iterations=3)
    return f"{h}|{n_atoms}|{n_bonds}"


def _largest_remainder(ns: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Allocate integer counts per stratum and role with exact global targets."""
    ns = np.asarray(ns, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    p = targets.astype(np.float64) / float(ns.sum())
    q = ns[:, None].astype(np.float64) * p[None, :]
    a = np.floor(q).astype(np.int64)
    cap = ns - a.sum(axis=1)
    rem = targets - a.sum(axis=0)
    frac = q - a
    while np.any(rem > 0):
        best = None
        for s in range(len(ns)):
            if cap[s] <= 0:
                continue
            for r in range(len(targets)):
                if rem[r] <= 0:
                    continue
                key = (frac[s, r], -s, -r)
                if best is None or key > best[0]:
                    best = (key, s, r)
        if best is None:
            raise RuntimeError(f"cannot allocate: rem={rem}, cap={cap}")
        _, s, r = best
        a[s, r] += 1
        cap[s] -= 1
        rem[r] -= 1
    assert np.array_equal(a.sum(axis=0), targets), (a.sum(axis=0), targets)
    assert np.array_equal(a.sum(axis=1), ns), (a.sum(axis=1), ns)
    return a


def main() -> int:
    AC.install_load_guard()
    cache = AC.load_train_cache()
    n = int(cache["n_molecules"])
    ap = np.asarray(cache["atom_ptr"])
    gfeat = np.asarray(cache["global_feat"])
    graph_info = []
    groups = defaultdict(list)
    for i in range(n):
        g, n_atoms, n_bonds = _graph(cache, i)
        key = _content_key(g, n_atoms, n_bonds)
        graph_info.append({"i": i, "key": key, "n_atoms": n_atoms, "n_bonds": n_bonds,
                           "size_bin": _size_bin(n_atoms),
                           "ring_bin": _ring_bin(int(round(float(gfeat[i, 6]))))})
        groups[key].append(i)

    # Exact isomorphism verification for every non-singleton WL group.
    duplicate_rows = []
    duplicate_groups = []
    noniso_collisions = []
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        base_g = _graph(cache, rows[0])[0]
        all_iso = True
        for r in rows[1:]:
            other = _graph(cache, r)[0]
            if not nx.is_isomorphic(base_g, other,
                                    node_match=lambda a, b: a["atom"] == b["atom"],
                                    edge_match=lambda a, b: a["bond"] == b["bond"]):
                all_iso = False
                noniso_collisions.append({"key": key, "rows": rows})
                break
        if all_iso:
            duplicate_groups.append({"key": key, "rows": rows})
            duplicate_rows.extend(rows)
    duplicate_rows = sorted(set(int(x) for x in duplicate_rows))

    # Pool excludes duplicate rows; duplicates go to Fit.
    dup_set = set(duplicate_rows)
    pool = [i for i in range(n) if i not in dup_set]
    strata = defaultdict(list)
    for i in pool:
        strata[(graph_info[i]["size_bin"], graph_info[i]["ring_bin"])].append(i)
    strata_keys = sorted(strata.keys())
    ns = np.asarray([len(strata[k]) for k in strata_keys], dtype=np.int64)
    targets = np.asarray([TARGETS["fit"] - len(duplicate_rows), TARGETS["probe"], TARGETS["audit"]], dtype=np.int64)
    alloc = _largest_remainder(ns, targets)
    roles = {"fit": [], "probe": [], "audit": []}
    for s, key in enumerate(strata_keys):
        rows = sorted(strata[key], key=lambda i: hashlib.sha256(graph_info[i]["key"].encode()).hexdigest())
        a0, a1, a2 = [int(x) for x in alloc[s]]
        if a0 + a1 + a2 != len(rows):
            raise RuntimeError("stratum allocation mismatch")
        roles["fit"].extend(rows[:a0])
        roles["probe"].extend(rows[a0:a0 + a1])
        roles["audit"].extend(rows[a0 + a1:])
    roles["fit"].extend(duplicate_rows)
    for k in roles:
        roles[k] = sorted(int(x) for x in roles[k])
    assert len(roles["fit"]) == 8000, len(roles["fit"])
    assert len(roles["probe"]) == 1000, len(roles["probe"])
    assert len(roles["audit"]) == 1000, len(roles["audit"])
    assert len(set(roles["fit"] + roles["probe"] + roles["audit"])) == n

    # No content key may span roles.
    row_role = {}
    for k, rows in roles.items():
        for r in rows:
            row_role[r] = k
    spans = []
    for key, rows in groups.items():
        rs = {row_role[r] for r in rows}
        if len(rs) > 1:
            spans.append({"key": key, "rows": rows, "roles": sorted(rs)})
    if spans:
        raise RuntimeError(f"content key spans roles: {spans[:3]}")

    def role_hash(rows):
        return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()

    def role_summary(rows):
        nat = np.asarray([graph_info[i]["n_atoms"] for i in rows])
        rings = np.asarray([int(round(float(gfeat[i, 6]))) for i in rows])
        return {
            "n": int(len(rows)),
            "row_sha256": role_hash(rows),
            "n_atoms": {"mean": float(nat.mean()), "median": float(np.median(nat)),
                        "min": int(nat.min()), "max": int(nat.max())},
            "n_ring_systems": {str(int(k)): int(v) for k, v in zip(*np.unique(rings, return_counts=True))},
            "size_bin_counts": {str(k): int(np.sum([graph_info[i]["size_bin"] == k for i in rows])) for k in range(len(SIZE_BINS))},
            "ring_bin_counts": {str(k): int(np.sum([graph_info[i]["ring_bin"] == k for i in rows])) for k in range(len(RING_BINS))},
        }

    out = {
        "protocol_id": "zinc-dcc-information-audit-v1",
        "frozen_at_utc": "2026-10-26T01:32:08Z",
        "repo_head": "58e6837b91507aae3feb9ad9370ef75d1959b7ec",
        "split_rule": {
            "content_key": "WL_hash(3, node atom type, edge bond type) + n_atoms + n_bonds",
            "duplicate_rule": "exact isomorphic duplicates are verified and all assigned to Fit; they are excluded from the stratified allocation pool",
            "strata": "n_atoms bins [0-20,21-24,25-28,29+] x n_ring_systems bins [0,1-3,4+]",
            "within_stratum_order": "sha256(content_key) ascending",
            "allocation": "largest-remainder rounding to exact 8000/1000/1000; duplicate rows are added to Fit and the pool targets are reduced accordingly",
            "label_used": False,
        },
        "rows": {k: roles[k] for k in ("fit", "probe", "audit")},
        "role_hashes": {k: role_hash(roles[k]) for k in roles},
        "role_summaries": {k: role_summary(roles[k]) for k in roles},
        "duplicates": {
            "n_duplicate_rows": len(duplicate_rows),
            "duplicate_rows": duplicate_rows,
            "duplicate_groups": duplicate_groups,
            "nonisomorphic_wl_collisions": noniso_collisions,
        },
        "content_key_groups": len(groups),
        "n_rows": n,
        "label_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    AC.write_json(AC.CONFIGS / "split_fit8k_probe_audit.json", AC.jsonable(out))
    np.save(AC.ANALYSIS / "row_role.npy", np.asarray([row_role[i] for i in range(n)], dtype="U5"))
    print(json.dumps({
        "roles": {k: len(v) for k, v in roles.items()},
        "duplicates": len(duplicate_rows),
        "noniso_collisions": len(noniso_collisions),
        "role_hashes": out["role_hashes"],
        "role_summaries": {k: {"n_atoms_mean": v["n_atoms"]["mean"], "rings": v["n_ring_systems"]}
                           for k, v in out["role_summaries"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
