"""Structural / mechanism audits for LCRC-RHIC.

Modes
-----
``structure``
    CPU-only, label-free audit.  Re-verifies the frozen CPC path cache identity
    (437,929 rows, 34,369,144 paths), derives the directed branch id of every
    cached path and checks it against the real bond table, audits the two-level
    grouping on hand-built graphs against an independent enumeration and an
    explicit float64 reference, computes the Train structure-coverage table and
    runs the branch-grouping witness.

``preflight``
    Pre-training gate.  Exact parameter budget, bitwise W2 seed-0 part, epoch-0
    logit parity with untrained W2 (CPU exact, CUDA <= 1e-4), one-shot call and
    dependency audit, atom-renumbering / bond-direction / bond-order /
    path-order / molecule-isolation invariance, the independent two-level
    statistics check on a real batch, the witness, gradients at steps
    1/3/10/100 and a 100-batch real cost profile.  No formal training step.

No mode reads an official Test target; ``test_rows`` are never indexed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import rhic_common as C  # noqa: E402  sets up the inherited code paths first
import cpc_audit as CA  # noqa: E402  (audited permutation helpers, small-graph cases)
import cpc_paths as P  # noqa: E402
import rhic_branches as B  # noqa: E402
import rhic_model as M  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

FULL_CACHE_BLOCK = 4_000_000


# ---------------------------------------------------------------------------
# small synthetic graphs (complete batches: every field the W2 trunk reads)
# ---------------------------------------------------------------------------


def synthetic_batch(
    spec: Any,
    n_atoms: int,
    edges: Sequence[Sequence[int]],
    *,
    seed: int = 0,
    atom_features: np.ndarray | None = None,
    bond_features: np.ndarray | None = None,
    bond_type: int = 1,
) -> dict[str, Any]:
    """One-molecule batch with the exact frozen collate schema.

    Atom/bond features are drawn from a fixed private generator unless supplied
    (the hand-built grouping witness supplies controlled atom features through
    ``h0`` instead and only needs a structurally valid batch here).
    """
    n = int(n_atoms)
    e = sorted((int(min(u, v)), int(max(u, v))) for u, v in edges)
    eu = np.asarray([x[0] for x in e], dtype=np.int64)
    ev = np.asarray([x[1] for x in e], dtype=np.int64)
    et = np.full(len(e), int(bond_type), dtype=np.int64)
    adj = P.molecule_adjacency(n, eu, ev, et)
    paths = P.enumerate_cpc_molecule(n, adj)
    counts = paths["counts"]
    rng = np.random.default_rng(int(seed))
    af = (
        np.asarray(atom_features, dtype=np.float32)
        if atom_features is not None
        else rng.standard_normal((n, spec.atom_feat_dim)).astype(np.float32)
    )
    bf = (
        np.asarray(bond_features, dtype=np.float32)
        if bond_features is not None
        else rng.standard_normal((len(e), spec.bond_feat_dim)).astype(np.float32)
    )
    if af.shape != (n, spec.atom_feat_dim) or bf.shape != (len(e), spec.bond_feat_dim):
        raise ValueError("synthetic feature shapes do not match the spec")
    nbr_atom = np.repeat(np.arange(n, dtype=np.int64), [len(a) for a in adj]).astype(np.int64)
    path_atom = np.repeat(np.arange(n, dtype=np.int64), counts).astype(np.int64)
    n_paths = int(counts.sum())
    zero = np.zeros(n_paths, dtype=np.int64)
    return {
        "n_graphs": 1,
        "atom_feat": torch.from_numpy(np.ascontiguousarray(af)),
        "atom_type": torch.zeros(n, dtype=torch.long),
        "atom_scalar": torch.zeros((n, spec.n_scalars), dtype=torch.float32),
        "atom_batch": torch.zeros(n, dtype=torch.long),
        "atom_off": torch.zeros(1, dtype=torch.long),
        "bond_feat": torch.from_numpy(np.ascontiguousarray(bf)),
        "bond_type": torch.from_numpy(et),
        "bond_index": torch.from_numpy(np.stack([eu, ev], axis=1)) if len(e) else torch.zeros((0, 2), dtype=torch.long),
        "bond_batch": torch.zeros(len(e), dtype=torch.long),
        "nbr_atom": torch.from_numpy(nbr_atom),
        "nbr_code": torch.ones(nbr_atom.size, dtype=torch.long),
        "path_atom": torch.from_numpy(path_atom),
        "path_cat": torch.ones(n_paths, dtype=torch.long),
        "path_end63": torch.ones(n_paths, dtype=torch.long),
        "path_end": torch.from_numpy(paths["end"].astype(np.int64)),
        "path_closure": torch.from_numpy(zero.copy()),
        "path_graph": torch.zeros(n_paths, dtype=torch.long),
        "global": torch.zeros((1, spec.global_dim), dtype=torch.float32),
        "atom_counts": torch.tensor([n], dtype=torch.long),
        "bond_counts": torch.tensor([len(e)], dtype=torch.long),
        "cpc_path_atom": torch.from_numpy(path_atom),
        "cpc_middle": torch.from_numpy(paths["middle"].astype(np.int64)),
        "cpc_end": torch.from_numpy(paths["end"].astype(np.int64)),
        "cpc_bond_ij": torch.from_numpy(paths["bond_ij"].astype(np.int64)),
        "cpc_bond_jk": torch.from_numpy(paths["bond_jk"].astype(np.int64)),
    }


def _graph(n: int, edges: Sequence[Sequence[int]]) -> dict[str, Any]:
    e = sorted((int(min(u, v)), int(max(u, v))) for u, v in edges)
    return {"n": int(n), "edges": e}


def branch_audit_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Independent enumeration of branches, path→branch membership and n_b."""
    n = int(graph["n"])
    e = [tuple(x) for x in graph["edges"]]
    eu = np.asarray([x[0] for x in e], dtype=np.int64)
    ev = np.asarray([x[1] for x in e], dtype=np.int64)
    et = np.ones(len(e), dtype=np.int64)
    adj = P.molecule_adjacency(n, eu, ev, et)
    deg = np.asarray([len(a) for a in adj], dtype=np.int64)
    paths = P.enumerate_cpc_molecule(n, adj)
    root = np.repeat(np.arange(n, dtype=np.int64), paths["counts"])
    mid = paths["middle"].astype(np.int64)
    bond_ij = paths["bond_ij"].astype(np.int64)
    bond_jk = paths["bond_jk"].astype(np.int64)
    end = paths["end"].astype(np.int64)
    # expected: branch key (root, middle) with count deg(middle) - 1
    expected_nb: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j, _bt, _e in adj[i]:
            nb = int(deg[int(j)] - 1)
            if nb > 0:
                expected_nb[(int(i), int(j))] = nb
    got_nb: dict[tuple[int, int], int] = {}
    for p in range(root.size):
        key = (int(root[p]), int(mid[p]))
        got_nb[key] = got_nb.get(key, 0) + 1
    # expected branch id from the physical bond table
    bond_of = {tuple(sorted((int(x[0]), int(x[1])))): k for k, x in enumerate(e)}
    expected_bid = []
    for p in range(root.size):
        u, v = e[bond_of[tuple(sorted((int(root[p]), int(mid[p]))))]]
        direction = 0 if u == int(root[p]) else 1
        expected_bid.append(2 * bond_of[tuple(sorted((u, v)))] + direction)
    # path-level bond endpoints
    ends_ok = True
    for p in range(root.size):
        u1, v1 = e[int(bond_ij[p])]
        u2, v2 = e[int(bond_jk[p])]
        ends_ok &= {int(u1), int(v1)} == {int(root[p]), int(mid[p])}
        ends_ok &= {int(u2), int(v2)} == {int(mid[p]), int(end[p])}
    return {
        "n_atoms": n,
        "n_bonds": len(e),
        "n_paths": int(root.size),
        "expected_branches": {f"{k[0]},{k[1]}": int(v) for k, v in sorted(expected_nb.items())},
        "path_branch_counts": {f"{k[0]},{k[1]}": int(v) for k, v in sorted(got_nb.items())},
        "branches_match": got_nb == expected_nb,
        "expected_branch_ids": [int(x) for x in expected_bid],
        "path_bond_endpoints_match": bool(ends_ok),
        "root_paths": [int(x) for x in paths["counts"]],
        "root_branches": [
            int(sum(1 for j, _bt, _e in adj[i] if deg[int(j)] - 1 > 0)) for i in range(n)
        ],
    }


def small_graph_audits(spec: Any, model: M.RHICModel) -> dict[str, Any]:
    cases = [
        ("single_atom", _graph(1, [])),
        ("single_bond_zero_path", _graph(2, [(0, 1)])),
        ("chain_3_single_path", _graph(3, [(0, 1), (1, 2)])),
        ("chain_4", _graph(4, [(0, 1), (1, 2), (2, 3)])),
        ("triangle", _graph(3, [(0, 1), (1, 2), (2, 0)])),
        ("square", _graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])),
        ("branched_tree", _graph(5, [(0, 1), (0, 2), (0, 3), (1, 4)])),
        ("single_branch_two_paths", _graph(4, [(0, 1), (1, 2), (1, 3)])),
        ("two_branches", _graph(6, [(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)])),
        ("multi_path_ring", _graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])),
        ("k4", _graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])),
    ]
    model.eval()
    out: dict[str, Any] = {}
    all_ok = True
    for case_index, (name, graph) in enumerate(cases):
        batch = synthetic_batch(spec, graph["n"], graph["edges"], seed=1000 + case_index)
        ref = branch_audit_graph(graph)
        with torch.inference_mode():
            _logits, aux = model(batch, return_aux=True)
        rhic = aux["rhic"]
        bid = rhic["branch_id"]
        expected_ids = torch.tensor(ref["expected_branch_ids"], dtype=torch.long)
        ids_ok = bool(torch.equal(torch.sort(bid).values, torch.sort(expected_ids).values)) if bid.numel() else expected_ids.numel() == 0
        # branch counts per (root, middle), mapped to the model's active ids
        counts_ok = True
        keys = {}
        for p in range(int(batch["cpc_path_atom"].numel())):
            key = (int(batch["cpc_path_atom"][p]), int(batch["cpc_middle"][p]))
            keys.setdefault(int(bid[p]), set()).add(key)
        counts_ok &= all(len(v) == 1 for v in keys.values())
        for b, key_set in keys.items():
            key = next(iter(key_set))
            want = ref["expected_branches"].get(f"{key[0]},{key[1]}")
            counts_ok &= int(rhic["branch_counts"][b]) == int(want)
        # root counts
        root_ok = True
        for i in range(graph["n"]):
            want = int(ref["root_branches"][i])
            root_ok &= int(rhic["root_branch_count"][i]) == want
            if want == 0:
                root_ok &= bool(torch.equal(aux["delta"][i], torch.zeros_like(aux["delta"][i])))
        # q is exactly zero for every branch with n_b < 2
        nb = rhic["branch_n"]
        if nb.numel():
            small = nb < 2
            q_small_zero = bool(torch.equal(rhic["branch_q"][small], torch.zeros_like(rhic["branch_q"][small])))
        else:
            q_small_zero = True
        # cross is exactly zero for roots with d < 2 (masked by d>0 anyway)
        d = rhic["root_branch_count"]
        cross_small_zero = bool(torch.equal(rhic["cross_branch"][d < 2], torch.zeros_like(rhic["cross_branch"][d < 2])))
        finite = bool(torch.isfinite(aux["h"]).all().item())
        ok = bool(ids_ok and counts_ok and root_ok and q_small_zero and cross_small_zero and finite and ref["path_bond_endpoints_match"] and ref["branches_match"])
        all_ok &= ok
        out[name] = {
            **ref,
            "model_n_paths": int(rhic["n_paths"]),
            "model_n_branches": int(rhic["n_branches"]),
            "branch_ids_match_independent_enumeration": ids_ok,
            "branch_membership_matches": bool(counts_ok),
            "root_branch_counts_match": bool(root_ok),
            "q_zero_for_n_lt_2": q_small_zero,
            "cross_zero_for_d_lt_2": cross_small_zero,
            "finite": finite,
            "checks": {
                "ids": ids_ok,
                "membership": bool(counts_ok),
                "root_counts": bool(root_ok),
                "q_small": q_small_zero,
                "cross_small": cross_small_zero,
                "finite": finite,
            },
            "all_ok": ok,
        }
    return {"cases": out, "all_ok": all_ok}


# ---------------------------------------------------------------------------
# full-cache identity + branch audit
# ---------------------------------------------------------------------------


def cache_identity(cpc: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(cpc["meta"])
    recorded = C.read_json(C.CPC_STRUCTURE_ARTIFACT)["cache_meta"]
    checks = {
        "n_rows": int(meta["n_rows"]) == int(C.CPC_CACHE["n_rows"]),
        "n_paths": int(meta["n_paths"]) == int(C.CPC_CACHE["n_paths"]),
        "rows_sha256": meta["rows_sha256"] == C.CPC_CACHE["rows_sha256"],
        "cache_sha256": meta["cache_sha256"] == C.CPC_CACHE["cache_sha256"],
        "schema": meta["enumeration_schema_version"] == C.CPC_CACHE["enumeration_schema_version"],
        "no_labels": bool(meta.get("labels_present")) is False,
        "no_test_labels": bool(meta.get("test_labels_present")) is False,
        "field_hashes_match_cpc_round": meta["field_sha256"] == recorded["field_sha256"],
        "bytes_match_cpc_round": int(meta["bytes"]) == int(recorded["bytes"]),
    }
    return {
        "path": C.track_rel(cpc["path"]),
        "n_rows": int(meta["n_rows"]),
        "n_atoms": int(meta["n_atoms"]),
        "n_paths": int(meta["n_paths"]),
        "bytes": int(meta["bytes"]),
        "meta_sha256": C.sha256_file(Path(cpc["path"]) / "meta.json"),
        "field_sha256": meta["field_sha256"],
        "checks": checks,
        "all_ok": bool(all(checks.values())),
    }


def full_cache_branch_audit(cpc: Mapping[str, Any], task_cache: Mapping[str, Any]) -> dict[str, Any]:
    """One vectorised pass over all 34,369,144 cached paths.

    Every path must (i) have its root and middle on its two real bonds, (ii) get
    exactly one direction, (iii) land in ``[0, 2 n_bonds_row)``, and (iv) share
    its branch id with exactly the paths of the same ``(root, middle)``.
    """
    cpc_ptr = np.asarray(cpc["cpc_ptr"], dtype=np.int64)
    middle = np.asarray(cpc["cpc_middle"], dtype=np.int64)
    bond_ij = np.asarray(cpc["cpc_bond_ij"], dtype=np.int64)
    bond_jk = np.asarray(cpc["cpc_bond_jk"], dtype=np.int64)
    end = np.asarray(cpc["cpc_end"], dtype=np.int64)
    row_atom_ptr = np.asarray(cpc["row_atom_ptr"], dtype=np.int64)
    row_bond_ptr = np.asarray(cpc["row_bond_ptr"], dtype=np.int64)
    row_n_bonds = np.asarray(cpc["row_n_bonds"], dtype=np.int64)
    atom_ptr = np.asarray(task_cache["atom_ptr"], dtype=np.int64)
    if int(atom_ptr[-1]) != int(cpc_ptr.size - 1):
        raise RuntimeError("atom bookkeeping mismatch between the frozen cache and the CPC cache")
    bond_index = np.asarray(task_cache["bond_index"], dtype=np.int64)
    n_rows = int(row_atom_ptr.size - 1)
    atom_row = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(row_atom_ptr).astype(np.int64))
    atom_off = row_atom_ptr[:-1]
    bond_off = row_bond_ptr[:-1]
    total = int(cpc_ptr[-1])
    n_atoms_total = int(cpc_ptr.size - 1)

    bad_direction = 0
    bad_middle = 0
    bad_second_bond = 0
    bad_range = 0
    paths = 0
    branch_ids = np.empty(total, dtype=np.int64)
    keys = np.empty(total, dtype=np.int64)
    for start in range(0, total, FULL_CACHE_BLOCK):
        stop = min(start + FULL_CACHE_BLOCK, total)
        pos = np.arange(start, stop, dtype=np.int64)
        root = np.searchsorted(cpc_ptr, pos, side="right") - 1
        row = atom_row[root]
        mid = atom_off[row] + middle[start:stop]
        gid = bond_off[row] + bond_ij[start:stop]
        gid2 = bond_off[row] + bond_jk[start:stop]
        # the frozen row bond table stores *row-local* atom ids; the CPC path
        # arrays store *row-local* bond ids.  Shift both to the cached frame.
        u = atom_off[row] + bond_index[gid, 0]
        v = atom_off[row] + bond_index[gid, 1]
        forward = u == root
        backward = v == root
        bad_direction += int((~(forward ^ backward)).sum())
        bad_middle += int((~np.where(forward, v == mid, u == mid)).sum())
        u2 = atom_off[row] + bond_index[gid2, 0]
        v2 = atom_off[row] + bond_index[gid2, 1]
        e = end[start:stop] + atom_off[row]
        ok2 = ((u2 == mid) & (v2 == e)) | ((u2 == e) & (v2 == mid))
        bad_second_bond += int((~ok2).sum())
        bid_local = 2 * bond_ij[start:stop] + (~forward).astype(np.int64)
        bad_range += int((bid_local < 0).sum() + (bid_local >= 2 * row_n_bonds[row]).sum())
        bid = 2 * gid + (~forward).astype(np.int64)
        branch_ids[start:stop] = bid
        keys[start:stop] = root * n_atoms_total + mid
        paths += int(stop - start)
    if paths != total:
        raise RuntimeError("full-cache branch audit did not visit every path")

    # (root, middle) <-> branch id must be a bijection on the realised paths.
    order = np.argsort(branch_ids, kind="stable")
    bid_sorted = branch_ids[order]
    key_sorted = keys[order]
    db = bid_sorted[1:] != bid_sorted[:-1]
    dk = key_sorted[1:] != key_sorted[:-1]
    key_split_within_branch = int(np.count_nonzero(dk & ~db))
    n_unique_branches = int(np.count_nonzero(~db)) + 1 if total else 0
    n_unique_keys = int(np.count_nonzero(~dk)) + 1 if total else 0
    uniq_bid = np.unique(branch_ids) if total else np.zeros(0, dtype=np.int64)
    uniq_key = np.unique(keys) if total else np.zeros(0, dtype=np.int64)
    del order, bid_sorted, key_sorted, db, dk, branch_ids, keys

    checks = {
        "paths_visited": paths == total,
        "direction_exactly_one": bad_direction == 0,
        "middle_is_other_bond_endpoint": bad_middle == 0,
        "second_bond_connects_middle_end": bad_second_bond == 0,
        "branch_id_in_range": bad_range == 0,
        "branch_ids_bijective_with_root_middle": bool(
            key_split_within_branch == 0 and n_unique_branches == n_unique_keys
        ),
    }
    return {
        "paths": total,
        "atoms": n_atoms_total,
        "rows": n_rows,
        "block": FULL_CACHE_BLOCK,
        "bad_direction": bad_direction,
        "bad_middle": bad_middle,
        "bad_second_bond": bad_second_bond,
        "bad_range": bad_range,
        "key_split_within_branch": key_split_within_branch,
        "unique_branches": n_unique_branches,
        "unique_root_middle_keys": n_unique_keys,
        "unique_branch_ids_sha256": C.sha256_bytes(np.ascontiguousarray(uniq_bid).tobytes()),
        "unique_keys_sha256": C.sha256_bytes(np.ascontiguousarray(uniq_key).tobytes()),
        "checks": checks,
        "all_ok": bool(all(checks.values())),
        "note": "distinct branches with >=1 path only; a directed bond with no legal path is never realised",
    }


def batch_local_branch_audit(task: Any, cpc: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Branch ids, offsets and molecule isolation on a real mixed batch."""
    batch = P.collate_with_cpc(task, cpc, rows)
    iso = P.check_batch_isolation(batch)
    bid = B.branch_ids(
        batch["cpc_path_atom"],
        batch["cpc_middle"],
        batch["cpc_bond_ij"],
        batch["bond_index"],
        verify=True,
    )
    n_branches = 2 * int(batch["bond_index"].shape[0])
    counts, active, inv, active_ids = B.branch_group_layout(bid, n_branches)
    root_b, mid_b, phys_b = B.branch_endpoints(active_ids, batch["bond_index"])
    # every branch id must map to exactly one (root, middle), and different
    # branch ids must map to different (root, middle) pairs
    key_of: dict[int, set[tuple[int, int]]] = {}
    for p in range(int(bid.numel())):
        key_of.setdefault(int(bid[p]), set()).add(
            (int(batch["cpc_path_atom"][p]), int(batch["cpc_middle"][p]))
        )
    pair_ok = all(len(v) == 1 for v in key_of.values())
    pair_unique = len({next(iter(v)) for v in key_of.values()}) == len(key_of) == int(active_ids.numel())
    # every active branch is rooted at the molecule of its paths
    same_molecule = True
    for p in range(int(bid.numel())):
        b = int(bid[p])
        same_molecule &= int(batch["bond_batch"][b // 2]) == int(batch["atom_batch"][batch["cpc_path_atom"][p]])
    # per-molecule active branch count matches the cache-derived structure
    per_atom = B.per_atom_branch_structure(cpc)
    expected = 0
    for row in rows:
        a0 = int(cpc["row_atom_ptr"][cpc["row_index"][row]])
        a1 = int(cpc["row_atom_ptr"][cpc["row_index"][row] + 1])
        expected += int(per_atom["branch_count"][a0:a1].sum())
    checks = {
        "isolated": bool(iso["isolated"] and iso["bonds_same_molecule"] and iso["bond_endpoints_match_path"]),
        "branch_ids_in_range": bool(bid.numel() == 0 or (bid.min() >= 0 and bid.max() < n_branches)),
        "branch_sizes_match_counts": bool(torch.equal(counts.index_select(0, active_ids), torch.bincount(inv))),
        "one_root_middle_per_branch": bool(pair_ok),
        "root_middle_pairs_unique": bool(pair_unique),
        "branch_bond_molecule_matches_root": bool(same_molecule),
        "active_branch_count_matches_cache_structure": int(active_ids.numel()) == expected,
    }
    return {
        "rows": [int(x) for x in rows],
        "n_paths": int(bid.numel()),
        "n_directed_branches": int(n_branches),
        "n_active_branches": int(active_ids.numel()),
        "n_active_branches_expected": int(expected),
        "isolation": iso,
        "checks": checks,
        "all_ok": bool(all(checks.values())),
    }


# ---------------------------------------------------------------------------
# independent torch-vs-numpy statistics check
# ---------------------------------------------------------------------------


def independent_stats_check(model: M.RHICModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        _logits, aux = model(batch, return_aux=True)
    r = aux["rhic"]
    v = r["v"].double().numpy()
    inv = r["branch_inv"].numpy()
    n_branches = int(r["n_branches"])
    ref_b = B.reference_two_level_stats(v, inv.tolist(), n_groups=n_branches)
    mu_path = r["branch_mu"].double().numpy()[inv]
    q_path = r["branch_q"].double().numpy()[inv]
    ref_mu_path = ref_b["mu"][ref_b["inv"]]
    ref_q_path = ref_b["q"][ref_b["inv"]]
    d_branch = np.abs(mu_path - ref_mu_path)
    d_q = np.abs(q_path - ref_q_path)
    branch_values = r["branch_value"].double().numpy()
    branch_root = r["branch_root"].numpy()
    n_atoms = int(r["root_branch_count"].numel())
    ref_r = B.reference_two_level_stats(branch_values, branch_root.tolist(), n_groups=n_atoms)
    mean_path = r["mean_branch"].double().numpy()[branch_root]
    cross_path = r["cross_branch"].double().numpy()[branch_root]
    d_mean = np.abs(mean_path - ref_r["mu"][ref_r["inv"]])
    d_cross = np.abs(cross_path - ref_r["q"][ref_r["inv"]])
    checks = {
        "mu_max_abs_diff_le_1e_5": float(d_branch.max()) <= 1e-5,
        "q_max_abs_diff_le_1e_5": float(d_q.max()) <= 1e-5,
        "mean_branch_max_abs_diff_le_1e_5": float(d_mean.max()) <= 1e-5,
        "cross_branch_max_abs_diff_le_1e_5": float(d_cross.max()) <= 1e-5,
    }
    return {
        "n_paths": int(v.shape[0]),
        "n_active_branches": n_branches,
        "n_atoms": n_atoms,
        "mu_max_abs_diff": float(d_branch.max()) if d_branch.size else 0.0,
        "q_max_abs_diff": float(d_q.max()) if d_q.size else 0.0,
        "mean_branch_max_abs_diff": float(d_mean.max()) if d_mean.size else 0.0,
        "cross_branch_max_abs_diff": float(d_cross.max()) if d_cross.size else 0.0,
        "naive_pair_term_matches_model_q": checks["q_max_abs_diff_le_1e_5"],
        "checks": checks,
        "all_ok": bool(all(checks.values())),
        "reference": "explicit float64 (S^2 - T)/(n(n-1)) with self-products excluded",
    }


# ---------------------------------------------------------------------------
# the branch-grouping witness
# ---------------------------------------------------------------------------


def _multiset_close(a: torch.Tensor, b: torch.Tensor, *, tol: float) -> bool:
    """Row-multiset equality within ``tol`` (exact matching; witnesses are tiny)."""
    if a.shape != b.shape or a.shape[0] != b.shape[0]:
        return False
    used = [False] * int(b.shape[0])
    for i in range(int(a.shape[0])):
        dist = (b - a[i]).abs().max(dim=1).values
        best = None
        for j in range(int(b.shape[0])):
            if used[j]:
                continue
            if best is None or float(dist[j]) < float(dist[best]):
                best = j
        if best is None or float(dist[best]) > tol:
            return False
        used[best] = True
    return True


def witness(task: Any) -> dict[str, Any]:
    """Two branch groupings with identical path-token multisets.

    H graph, root 0 with two indistinguishable first-hop branches ``j1`` and
    ``j2`` and two endpoints per branch.  The middle atoms and all bond feature
    rows that enter a path token are identical, so a path token depends only on
    its endpoint state ``x`` or ``y``.

    A: branch 1 = {x, x}, branch 2 = {y, y}
    B: branch 1 = {x, y}, branch 2 = {x, y}

    Every root/middle/endpoint marginal, the 4-token multiset, the branch count
    and the bond features are the same.  CPC's global root mean is therefore
    identical; RHIC's branch pair term is not.  The RHIC weights are explicitly
    set to a fixed non-degenerate pattern (this is a witness, not a trained
    model).
    """
    model, _audit = M.build_rhic_model(
        task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu", zero_output=False
    )
    model.eval()
    wg = torch.Generator().manual_seed(424242)
    with torch.no_grad():
        for sub in model.rhic.modules():
            if isinstance(sub, torch.nn.Linear):
                sub.weight.normal_(0.0, 0.1, generator=wg)
                if sub.bias is not None:
                    sub.bias.normal_(0.0, 0.01, generator=wg)
    g = torch.Generator().manual_seed(20261110)
    h0 = torch.randn((7, C.WIDTH), generator=g)
    h0[2] = h0[1]  # middle j1 == middle j2
    b = torch.randn((6, C.BOND_WIDTH), generator=g)
    b[1] = b[0]  # bond i-j1 == bond i-j2
    b[3] = b[2]
    b[4] = b[2]
    b[5] = b[2]  # all four second bonds identical
    x_state = torch.randn(C.WIDTH, generator=g)
    y_state = torch.randn(C.WIDTH, generator=g)
    h_a = h0.clone()
    h_a[3] = x_state
    h_a[4] = x_state
    h_a[5] = y_state
    h_a[6] = y_state
    h_b = h0.clone()
    h_b[3] = x_state
    h_b[4] = y_state
    h_b[5] = x_state
    h_b[6] = y_state
    path_atom = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    middle = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    end = torch.tensor([3, 4, 5, 6], dtype=torch.long)
    bij = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    bjk = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    bond_index = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]], dtype=torch.long
    )
    batch = {
        "cpc_path_atom": path_atom,
        "cpc_middle": middle,
        "cpc_end": end,
        "cpc_bond_ij": bij,
        "cpc_bond_jk": bjk,
        "bond_index": bond_index,
    }
    with torch.inference_mode():
        da, aux_a = model.rhic(h_a, b, batch, mode="real")
        db, aux_b = model.rhic(h_b, b, batch, mode="real")
        r_a, m_a, d_a, e_a = model.rhic.project(h_a, b)
        _z_a, _g_a, v_a, xa = model.rhic.path_tokens(
            r_a, m_a, d_a, e_a, path_atom, middle, end, bij, bjk, return_features=True
        )
        r_b, m_b, d_b, e_b = model.rhic.project(h_b, b)
        _z_b, _g_b, v_b, xb = model.rhic.path_tokens(
            r_b, m_b, d_b, e_b, path_atom, middle, end, bij, bjk, return_features=True
        )
    token_tol = 1e-5
    x_multiset_equal = _multiset_close(xa, xb, tol=token_tol)
    v_multiset_equal = _multiset_close(v_a, v_b, tol=token_tol)
    context_a = v_a.sum(0) / 4.0
    context_b = v_b.sum(0) / 4.0
    context_diff = float((context_a - context_b).abs().max().item())
    q_diff = float((aux_a["branch_q"] - aux_b["branch_q"]).abs().max().item())
    bi_diff = float((aux_a["branch_input"] - aux_b["branch_input"]).abs().max().item())
    delta_diff = float((da[0] - db[0]).abs().max().item())
    branch_value_diff = float((aux_a["branch_value"] - aux_b["branch_value"]).abs().max().item())
    n_equal = bool(torch.equal(aux_a["branch_n"], aux_b["branch_n"]))
    checks = {
        "path_token_multiset_identical": x_multiset_equal,
        "path_value_multiset_identical": v_multiset_equal,
        "same_path_count": int(v_a.shape[0]) == int(v_b.shape[0]) == 4,
        "same_branch_count": int(aux_a["n_branches"]) == int(aux_b["n_branches"]) == 2,
        "same_branch_sizes": n_equal,
        "same_root_state": bool(torch.equal(r_a[0], r_b[0])),
        "same_bond_features": bool(torch.equal(e_a, e_b)),
        "cpc_global_mean_identical": context_diff <= 1e-6,
        "branch_q_differs": q_diff > 1e-5,
        "branch_input_differs": bi_diff > 1e-5,
        "branch_value_differs": branch_value_diff > 1e-5,
        "root_delta_differs": delta_diff > 1e-5,
    }
    return {
        "graph": "root 0 with branches (0,1) and (0,2); endpoints 3,4 under 1 and 5,6 under 2",
        "A": {"branch1": "x,x", "branch2": "y,y"},
        "B": {"branch1": "x,y", "branch2": "x,y"},
        "n_paths": 4,
        "n_branches": 2,
        "weights": "all RHIC Linear layers reset from a private generator, std 0.1 (explicit non-degenerate witness weights)",
        "token_tolerance": token_tol,
        "cpc_global_mean_max_abs_diff": context_diff,
        "branch_q_max_abs_diff": q_diff,
        "branch_input_max_abs_diff": bi_diff,
        "branch_value_max_abs_diff": branch_value_diff,
        "root_delta_max_abs_diff": delta_diff,
        "checks": checks,
        "all_ok": bool(all(checks.values())),
        "note": (
            "The witness only proves that the RHIC interface can distinguish 'same path-token "
            "multiset, different branch grouping'.  It is not an expressiveness theorem for W2 "
            "and not a claim of chemical meaning."
        ),
    }


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def coverage(task: Any, cpc: Mapping[str, Any]) -> dict[str, Any]:
    structure = B.per_atom_branch_structure(cpc)
    row_atom_ptr = np.asarray(cpc["row_atom_ptr"], dtype=np.int64)
    row_index = np.asarray(cpc["row_index"], dtype=np.int64)
    pc = structure["path_count"]
    bc = structure["branch_count"]
    mp = structure["multi_path_branches"]
    row_starts = row_atom_ptr[:-1]
    molecule = {
        "rows": int(len(task.train_rows)),
        "any_root_ge2_paths": float(
            np.mean(np.add.reduceat((pc >= 2).astype(np.int64), row_starts)[row_index[task.train_rows]] > 0)
        ),
        "any_root_ge2_branches": float(
            np.mean(np.add.reduceat((bc >= 2).astype(np.int64), row_starts)[row_index[task.train_rows]] > 0)
        ),
        "any_branch_ge2_paths": float(
            np.mean(np.add.reduceat((mp >= 1).astype(np.int64), row_starts)[row_index[task.train_rows]] > 0)
        ),
        "any_root_ge2_multi_path_branches": float(
            np.mean(np.add.reduceat((mp >= 2).astype(np.int64), row_starts)[row_index[task.train_rows]] > 0)
        ),
    }
    # atoms of the frozen Train molecules (structures only, never labels)
    mask = np.zeros(int(row_atom_ptr[-1]), dtype=bool)
    for start in range(0, len(task.train_rows), 20_000):
        rows = np.asarray(task.train_rows[start : start + 20_000], dtype=np.int64)
        idx = row_index[rows]
        a0 = row_atom_ptr[idx]
        counts = (row_atom_ptr[idx + 1] - a0).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            continue
        off = np.repeat(a0, counts) + (
            np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        )
        mask[off] = True
    summary = B.coverage_summary(structure, mask)
    return {
        "scope": "frozen official Train molecules only (350,343 rows); structures, no labels",
        "atoms": int(mask.sum()),
        "rows": int(len(task.train_rows)),
        "atom_level": summary,
        "molecule_level": molecule,
    }


# ---------------------------------------------------------------------------
# preflight gates
# ---------------------------------------------------------------------------


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _logits(model, batch, **kwargs) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return model(batch, **kwargs).detach().float()


def gate_init(task: Any, device: torch.device) -> dict[str, Any]:
    model, audit = M.build_rhic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    ref, ref_audit = M.untrained_w2(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    rows = [int(x) for x in task.valid_rows[:16]]
    batch_cpu = P.collate_with_cpc(task, task.cpc, rows)
    model.eval()
    ref.eval()
    with torch.inference_mode():
        a = model(batch_cpu)
        b = ref(batch_cpu)
        diff_cpu = _maxdiff(a, b)
    out: dict[str, Any] = {
        "parameters": audit["parameters"],
        "w2_part_bitwise_equal_reference": audit["w2_part_bitwise_equal_reference"],
        "w2_trunk_init_fingerprint": audit["w2_trunk_init_fingerprint"],
        "w2_trunk_fingerprint_matches_recorded_w2": audit["w2_trunk_fingerprint_matches_recorded_w2"],
        "init": audit["init"],
        "rhic_state_sha256": audit["rhic_state_sha256"],
        "epoch0_logits_max_abs_diff_cpu": float(diff_cpu),
        "epoch0_logits_tolerance_cpu": 0.0,
        "epoch0_logits_cpu_exact": bool(diff_cpu == 0.0),
        "zeroed_final_root_linear": {
            "weight_zero": bool(float(model.rhic.root2.weight.abs().max()) == 0.0),
            "bias_zero": bool(float(model.rhic.root2.bias.abs().max()) == 0.0),
            "branch1_nonzero": bool(float(model.rhic.branch1.weight.abs().max()) > 0.0),
            "root1_nonzero": bool(float(model.rhic.root1.weight.abs().max()) > 0.0),
        },
    }
    if device.type == "cuda":
        model.to(device)
        batch = M.to_device(batch_cpu, device)
        with torch.inference_mode():
            a_gpu = model(batch)
            ref.to(device)
            b_gpu = ref(batch)
            diff_gpu = _maxdiff(a_gpu.cpu(), b_gpu.cpu())
        out["epoch0_logits_max_abs_diff_cuda"] = float(diff_gpu)
        out["epoch0_logits_cuda_within_1e-4"] = bool(diff_gpu <= 1e-4)
        with torch.inference_mode():
            cpu_logits = model.to("cpu")(batch_cpu).detach()
        model.to(device)
        with torch.inference_mode():
            gpu_logits = model(batch).detach().cpu()
        out["cpu_cuda_logits_max_abs_diff"] = _maxdiff(cpu_logits, gpu_logits)
        out["cpu_cuda_within_1e-4"] = bool(_maxdiff(cpu_logits, gpu_logits) <= 1e-4)
    out["all_ok"] = bool(
        audit["parameters"]["all_match"]
        and audit["w2_part_bitwise_equal_reference"]["all_equal"]
        and out["w2_trunk_fingerprint_matches_recorded_w2"]
        and out["epoch0_logits_cpu_exact"]
        and (device.type != "cuda" or out.get("epoch0_logits_cuda_within_1e-4", False))
    )
    return out


def permute_paths(batch: Mapping[str, Any], *, seed: int = 20261111) -> dict[str, Any]:
    n = int(batch["cpc_path_atom"].numel())
    gen = torch.Generator().manual_seed(int(seed))
    p = torch.randperm(n, generator=gen)
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for key in ("cpc_path_atom", "cpc_middle", "cpc_end", "cpc_bond_ij", "cpc_bond_jk"):
        out[key] = batch[key].index_select(0, p)
    return out


def permute_bonds(batch: Mapping[str, Any], *, seed: int = 20261112) -> dict[str, Any]:
    """Renumber physical bonds *within each molecule* (a branch-order shuffle)."""
    bi = batch["bond_index"]
    bond_batch = batch["bond_batch"]
    n_bonds = int(bi.shape[0])
    gen = torch.Generator().manual_seed(int(seed))
    m = torch.arange(n_bonds, dtype=torch.long)
    for g in sorted(set(int(x) for x in bond_batch.tolist())):
        ids = (bond_batch == g).nonzero(as_tuple=False).reshape(-1)
        m[ids] = ids[torch.randperm(ids.numel(), generator=gen)]
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for key in ("bond_index", "bond_feat", "bond_type", "bond_batch"):
        moved = torch.empty_like(batch[key])
        moved.index_copy_(0, m, batch[key])
        out[key] = moved
    for key in ("cpc_bond_ij", "cpc_bond_jk"):
        out[key] = m.index_select(0, batch[key])
    # nbr_code is keyed by the neighbour list, not by the bond id: unchanged.
    return out


def gate_architecture(task: Any, device: torch.device) -> dict[str, Any]:
    # non-degenerate initialisation: with the zeroed final linear every delta is
    # exactly zero and the invariance / propagation checks below would be vacuous.
    model, _audit = M.build_rhic_model(
        task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device, zero_output=False
    )
    rows = [int(x) for x in task.valid_rows[:8]]
    batch = M.to_device(P.collate_with_cpc(task, task.cpc, rows), device)
    model.eval()

    counters = {
        name: 0 for name in ("atom_encoder", "bond_encoder", "local", "drop", "rhic", "readout", "f", "rho")
    }
    handles = []
    for name in tuple(counters):
        module = getattr(model, name)

        def _hook(_m, _inp, _out, key=name):
            counters[key] += 1

        handles.append(module.register_forward_hook(_hook))
    rhic_inputs: list[Any] = []
    rhic_handle = model.rhic.register_forward_pre_hook(lambda _m, args: rhic_inputs.append(args))
    post_calls: list[Any] = []
    original_post = model.post_local

    def _wrapped_post(h, b, **kwargs):
        post_calls.append(h)
        return original_post(h, b, **kwargs)

    model.post_local = _wrapped_post
    try:
        with torch.inference_mode():
            logits, aux = model(batch, return_aux=True)
    finally:
        for h in handles:
            h.remove()
        rhic_handle.remove()
        model.post_local = original_post

    with torch.inference_mode():
        _a, _b, _s, h0_ref = model.encode_h0(batch)
    rhic_h0 = rhic_inputs[0][0] if rhic_inputs else None
    rhic_b = rhic_inputs[0][1] if rhic_inputs else None
    h0_diff = _maxdiff(rhic_h0, h0_ref) if rhic_h0 is not None else None
    b_diff = _maxdiff(rhic_b, aux["b"]) if rhic_b is not None else None
    post_matches = bool(len(post_calls) == 1 and torch.equal(post_calls[0], aux["h"]))
    h_is_sum = bool(torch.equal(aux["h"], aux["h0"] + aux["delta"]))

    perm, _p = CA._permute_atoms_with_paths(batch, seed=20261110)
    perm = M.to_device(perm, device)
    logits_perm = _logits(model, perm)
    perm_diff = _maxdiff(logits, logits_perm)

    flip = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    flip["bond_index"] = batch["bond_index"][:, [1, 0]].contiguous()
    logits_flip = _logits(model, flip)
    flip_diff = _maxdiff(logits, logits_flip)

    path_shuffle = permute_paths(batch)
    logits_paths = _logits(model, path_shuffle)
    path_diff = _maxdiff(logits, logits_paths)

    bond_shuffle = permute_bonds(batch)
    logits_bonds = _logits(model, bond_shuffle)
    bond_diff = _maxdiff(logits, logits_bonds)

    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    mask = (batch["atom_batch"] == 1).unsqueeze(1)
    gen = torch.Generator().manual_seed(7)
    noise = torch.randn(perturbed["atom_feat"].shape, generator=gen).to(device)
    perturbed["atom_feat"] = torch.where(mask, perturbed["atom_feat"] + noise, perturbed["atom_feat"])
    logits_pert = _logits(model, perturbed)
    iso_0 = _maxdiff(logits[0], logits_pert[0])
    iso_1 = _maxdiff(logits[1], logits_pert[1])

    # direct remote-endpoint dependence of the root delta (one-shot path read)
    with torch.inference_mode():
        delta_a, aux_a = model.rhic(aux["h0"], aux["b"], batch, mode="real")
    h0_b = aux["h0"].clone()
    # change one endpoint atom that is the endpoint of a path rooted elsewhere
    path_atom = batch["cpc_path_atom"]
    end = batch["cpc_end"]
    root_atoms = set(int(x) for x in path_atom.tolist())
    far = next(
        (
            (int(end[p]), int(path_atom[p]))
            for p in range(int(end.numel()))
            if int(end[p]) not in root_atoms
        ),
        None,
    )
    if far is None:
        # fall back to any path whose far endpoint differs from its own root
        far = next(
            (
                (int(end[p]), int(path_atom[p]))
                for p in range(int(end.numel()))
                if int(end[p]) != int(path_atom[p])
            ),
            None,
        )
    remote_diff = None
    if far is not None:
        endpoint_atom, root_atom = far
        h0_b[endpoint_atom] += 1.0
        with torch.inference_mode():
            delta_b, _aux_b = model.rhic(h0_b, aux["b"], batch, mode="real")
        remote_diff = float((delta_a[root_atom] - delta_b[root_atom]).abs().max().item())

    # branch-assignment shuffle: same tokens, same branch counts, different grouping
    with torch.inference_mode():
        delta_s, aux_s = model.rhic(aux["h0"], aux["b"], batch, mode="branch_assignment_shuffle", shuffle_seed=0)
    shuffle_diff = float((delta_a - delta_s).abs().max().item())
    token_same = bool(torch.equal(aux_a["v"], aux_s["v"]))
    branch_counts_same = bool(torch.equal(aux_a["branch_counts"], aux_s["branch_counts"]))
    branch_groups_differ = bool(not torch.equal(aux_a["branch_inv"], aux_s["branch_inv"]))

    checks = {
        "atom_encoder_once": counters["atom_encoder"] == 1,
        "bond_encoder_once": counters["bond_encoder"] == 1,
        "local_once": counters["local"] == 1,
        "drop_once": counters["drop"] == 1,
        "rhic_once": counters["rhic"] == 1,
        "f_once": counters["f"] == 1,
        "rho_once": counters["rho"] == 1,
        "readout_once": counters["readout"] == 1,
        "post_local_once": len(post_calls) == 1,
        "rhic_reads_same_h0": h0_diff is not None and h0_diff <= 1e-6,
        "rhic_reads_same_b": b_diff is not None and b_diff <= 1e-6,
        "post_local_reads_h0_plus_delta": post_matches and h_is_sum,
        "atom_renumbering_invariant": perm_diff <= 1e-5,
        "bond_storage_direction_invariant": flip_diff <= 1e-5,
        "bond_id_order_invariant": bond_diff <= 1e-5,
        "path_order_invariant": path_diff <= 1e-5,
        "molecule_isolation": iso_0 <= 1e-5 and iso_1 > 1e-4,
        "remote_endpoint_changes_root_delta": remote_diff is not None and remote_diff > 1e-8,
        "branch_shuffle_keeps_tokens_and_counts": token_same and branch_counts_same,
        "branch_shuffle_changes_groups": branch_groups_differ,
        "branch_shuffle_changes_delta": shuffle_diff > 1e-8,
        "finite": bool(torch.isfinite(logits).all().item()),
    }
    return {
        "module_calls": counters,
        "rhic_forward_calls": len(rhic_inputs),
        "post_local_calls": len(post_calls),
        "rhic_input_h0_equal_drop_local": h0_diff is not None and h0_diff <= 1e-6,
        "rhic_input_h0_max_abs_diff": h0_diff,
        "rhic_input_b_equal_bond_encoder": b_diff is not None and b_diff <= 1e-6,
        "rhic_input_b_max_abs_diff": b_diff,
        "post_local_input_is_h0_plus_delta": post_matches and h_is_sum,
        "atom_renumbering_max_abs_logit_diff": perm_diff,
        "bond_storage_direction_max_abs_logit_diff": flip_diff,
        "bond_id_order_max_abs_logit_diff": bond_diff,
        "path_order_max_abs_logit_diff": path_diff,
        "molecule_isolation_logit_diff_molecule0": iso_0,
        "molecule_isolation_logit_diff_molecule1": iso_1,
        "remote_endpoint_root_delta_max_abs_diff": remote_diff,
        "branch_shuffle_root_delta_max_abs_diff": shuffle_diff,
        "branch_shuffle_tokens_identical": token_same,
        "branch_shuffle_branch_counts_identical": branch_counts_same,
        "branch_shuffle_branch_groups_differ": branch_groups_differ,
        "logits_abs_max": float(logits.abs().max().item()),
        "delta_rms": M.rms(aux["delta"]),
        "h0_rms": M.rms(aux["h0"]),
        "checks": checks,
        "all_ok": bool(all(checks.values())),
    }


def gate_gradients_and_cost(task: Any, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, audit = M.build_rhic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    C.seed_everything(C.SEED)
    plan = C.SD.make_plan40(task, seed=C.SEED)
    batches = plan[0]["batches"]
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    audit_steps = set(int(x) for x in C.GRAD_AUDIT_STEPS)
    rows_out: dict[int, Any] = {}
    step_times: list[float] = []
    losses: list[float] = []
    warmup = int(C.BUDGET["profile_warmup_steps"])
    measured = int(C.BUDGET["profile_real_batches"])
    n_steps = warmup + measured
    t_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for step, rows in enumerate(batches[:n_steps], start=1):
        t0 = time.perf_counter()
        batch = stager.batch(P.collate_with_cpc(task, task.cpc, rows.tolist()))
        y = stager.labels(task.labels[rows])
        lr = C.set_lrs(optimizer, step=step - 1)
        logits = model(batch)
        loss = C.MET.masked_bce_loss(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at profile step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = {
            n: (None if p.grad is None else float(p.grad.detach().norm().item()))
            for n, p in model.named_parameters()
            if C.is_new_parameter(n)
        }
        if step in audit_steps:
            rows_out[step] = {
                "loss": float(loss.detach()),
                "lr": lr,
                "new_grads": grads,
                "nonzero_new_tensors": sorted(n for n, v in grads.items() if v is not None and v > 0.0),
                "zero_or_none_new_tensors": sorted(n for n, v in grads.items() if v is None or v == 0.0),
                "new_grad_total_norm": math.sqrt(sum(float(v) ** 2 for v in grads.values() if v is not None)),
                "w2_grad_total_norm": math.sqrt(
                    sum(
                        float(p.grad.detach().pow(2).sum().item())
                        for n, p in model.named_parameters()
                        if not C.is_new_parameter(n) and p.grad is not None
                    )
                ),
            }
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = float(time.perf_counter() - t0)
        if step > warmup:
            step_times.append(dt)
        losses.append(float(loss.detach()))
    wall = float(time.perf_counter() - t_start)
    stager.release()
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    arr = np.asarray(step_times, dtype=np.float64)
    val_rows = np.asarray(task.valid_rows[:2048], dtype=np.int64)
    model.eval()
    t_val = time.perf_counter()
    with torch.inference_mode():
        M.predict_logits(model, task, val_rows, device, batch_size=256)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    val_2048 = float(time.perf_counter() - t_val)
    valid_full_est = val_2048 * (len(task.valid_rows) / len(val_rows))
    step_p50 = float(np.percentile(arr, 50)) if arr.size else float("nan")
    train_epoch_est = step_p50 * C.STEPS_PER_EPOCH
    projected_hours = 40.0 * (train_epoch_est + valid_full_est) / 3600.0
    grad_checks = {}
    for s in C.GRAD_AUDIT_STEPS:
        row = rows_out.get(int(s), {})
        grad_checks[f"step_{s}_all_new_tensors_nonzero"] = bool(
            row and not row.get("zero_or_none_new_tensors")
        )
    grad_checks["step_3_10_100_all_new_tensors_nonzero"] = bool(
        all(grad_checks.get(f"step_{s}_all_new_tensors_nonzero", False) for s in (3, 10, 100))
    )
    return {
        "model": C.MODEL_NAME,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "warmup_steps": warmup,
        "measured_steps": measured,
        "wall_seconds": wall,
        "step_ms_p50": step_p50 * 1e3,
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
        "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
        "steps_per_second": float(len(step_times) / arr.sum()) if arr.size and arr.sum() > 0 else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "gpu_peak_allocated_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "validation_2048_rows_seconds": val_2048,
        "validation_full_projected_seconds": valid_full_est,
        "train_epoch_projected_seconds": train_epoch_est,
        "projected_40_epoch_gpu_hours": projected_hours,
        "budget_target_hours": C.BUDGET["gpu_target_hours"],
        "budget_hard_hours": C.BUDGET["gpu_hard_hours"],
        "within_target": bool(projected_hours <= C.BUDGET["gpu_target_hours"]),
        "within_hard_cap": bool(projected_hours <= C.BUDGET["gpu_hard_hours"]),
        "gradient_audit": {str(k): v for k, v in rows_out.items()},
        "gradient_checks": grad_checks,
        "parameter_audit_all_match": audit["parameters"]["all_match"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_structure(args: argparse.Namespace) -> int:
    task = C.SD.load_task()
    cache_dir = Path(args.cache)
    cpc = C.ensure_cpc_cache(task, cache_dir)
    task.cpc = cpc
    identity = cache_identity(cpc)
    full = full_cache_branch_audit(cpc, task.cache)
    small_model, _audit = M.build_rhic_model(
        task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu", zero_output=False
    )
    small = small_graph_audits(task.spec, small_model)
    rows = [int(task.train_rows[0]), int(task.train_rows[1]), int(task.valid_rows[0])]
    batch_local = batch_local_branch_audit(task, cpc, rows)
    batch = P.collate_with_cpc(task, cpc, rows)
    stats = independent_stats_check(small_model, batch)
    wit = witness(task)
    cov = coverage(task, cpc)
    payload = {
        "schema": C.SCHEMA_STRUCTURE,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "cache_identity": identity,
        "full_cache_branch_audit": full,
        "small_graphs": small,
        "batch_local": batch_local,
        "independent_stats": stats,
        "witness": wit,
        "coverage": cov,
        "all_ok": bool(
            identity["all_ok"]
            and full["all_ok"]
            and small["all_ok"]
            and batch_local["all_ok"]
            and stats["all_ok"]
            and wit["all_ok"]
        ),
    }
    C.write_json(C.ANALYSIS / "structure.json", payload)
    C.write_json(C.ANALYSIS / "coverage.json", cov)
    C.write_json(C.ANALYSIS / "witness.json", wit)
    C.log(
        f"[structure] all_ok={payload['all_ok']} paths={full['paths']} "
        f"branches={full['unique_branches']} train_branch_ge2={cov['atom_level']['root_with_at_least_two_branches']:.4f}"
    )
    return 0 if payload["all_ok"] else 1


def run_preflight(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    task = C.SD.load_task()
    cache_dir = Path(args.cache)
    task.cpc = C.ensure_cpc_cache(task, cache_dir)
    init = gate_init(task, device)
    arch = gate_architecture(task, device)
    wit = witness(task)
    cost = gate_gradients_and_cost(task, device)
    checks = {
        "init": init["all_ok"],
        "architecture": arch["all_ok"],
        "witness": wit["all_ok"],
        "parameter_audit": cost["parameter_audit_all_match"],
        "gradients_step_3_10_100": cost["gradient_checks"]["step_3_10_100_all_new_tensors_nonzero"],
        "within_hard_cap": cost["within_hard_cap"],
    }
    payload = {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "device": str(device),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "cache": P.cpc_cache_identity(cache_dir),
        "init": init,
        "architecture": arch,
        "witness": wit,
        "cost": cost,
        "checks": checks,
        "all_ok": bool(all(checks.values())),
    }
    C.write_json(C.ANALYSIS / "preflight.json", payload)
    C.write_json(C.ANALYSIS / "init_parity.json", init)
    C.write_json(C.ANALYSIS / "architecture_audit.json", arch)
    C.write_json(C.ANALYSIS / "gpu_profile.json", cost)
    C.log(
        f"[preflight] all_ok={payload['all_ok']} epoch0_cpu={init['epoch0_logits_max_abs_diff_cpu']} "
        f"arch={arch['all_ok']} witness={wit['all_ok']} "
        f"projected={cost['projected_40_epoch_gpu_hours']:.4f}h"
    )
    return 0 if payload["all_ok"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("structure", "preflight"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", default=str(P.CPC_CACHE_DIR))
    args = parser.parse_args(argv)
    try:
        if args.mode == "structure":
            return run_structure(args)
        return run_preflight(args)
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            C.ANALYSIS / f"audit_failed_{args.mode}.json",
            {
                "schema": "lcrc-pcba-rhic-audit-failure-v1",
                "round_id": C.ROUND_ID,
                "mode": args.mode,
                "created_utc": C.utcnow(),
                "git_commit": C.git_commit(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        C.log(f"[audit] FAILED {args.mode}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
