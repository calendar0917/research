"""Label-free real-edit catalogue for the Lipo task-transfer round.

The catalogue is built **before** labels or model predictions are read.  Every
legal non-ring single-bond bridge cut of every eligible original molecule is
enumerated; the common core is certified exactly by the parent round's
coloured-incidence-graph canonical serialization.  Hash/WL buckets are not used
as identity here: every candidate receives an exact certificate.

Definition (frozen in ``PROTOCOL.md`` / ``configs/task_transfer_s0.json``):

* molecule: one connected component, no explicit H atom, original OGB graph;
* cut: a bridge edge with OGB bond type SINGLE (0), i.e. a non-ring single bond;
* variable side S: 1..8 heavy atoms;
* core C: complement of S, at least 10 heavy atoms and at least 75% of the
  parent's heavy atoms;
* C identity: exact coloured graph (all nine OGB atom columns), port and the
  three cut-bond columns; no approximate or SMILES-only matching;
* one definition per unordered molecule pair: maximum C size, then C canonical
  key, then the two S keys, then the cut-edge indices;
* direction A -> B: lexicographically smaller S native key is A;
* ``delta = B - A``; ``transformation_id`` = ordered (S_A key, S_B key, cut
  bond columns).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tt_common as TC  # noqa: E402
import rd_fragment as RF  # noqa: E402

MAX_S = 8
MIN_C = 10
CORE_FRACTION = 0.75


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------


def _molecule_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    atom_all = np.asarray(payload["atom_feat"])
    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    n_atoms = np.asarray(payload["graph_n_atoms"], dtype=np.int64)
    n_components = np.asarray(payload["graph_n_components"], dtype=np.int64)
    multi = np.flatnonzero(n_components != 1).tolist()
    explicit_h = []
    for row in range(len(n_atoms)):
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        if np.any(np.asarray(atom_all[a0:a1])[:, 0] == 0):  # OGB index 0 == Z=1
            explicit_h.append(int(row))
    return {
        "n_rows": int(len(n_atoms)),
        "multi_component_rows": [int(x) for x in multi],
        "explicit_h_rows": [int(x) for x in explicit_h],
        "n_excluded_multi_component": int(len(multi)),
        "n_excluded_explicit_h": int(len(explicit_h)),
    }


def enumerate_candidates(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """All legal cuts, the audit counters, and the per-row official mapping."""
    official = TC.official_split(payload)
    canon = TC.canonical_isomeric_smiles(payload)
    stereo = [TC.stereo_tokens(s) for s in canon]
    n_atoms_all = np.asarray(payload["graph_n_atoms"], dtype=np.int64)
    n_components = np.asarray(payload["graph_n_components"], dtype=np.int64)
    audit = _molecule_audit(payload)
    mol_audit = {
        "n_rows": int(len(n_atoms_all)),
        "n_single_component_no_explicit_h": 0,
        "n_rows_with_legal_cut": 0,
        "n_rows_without_legal_cut": 0,
        "n_bridges_total": 0,
        "n_single_bridges_total": 0,
        "n_non_single_bridges_skipped": 0,
        "n_s_out_of_range": 0,
        "n_core_too_small": 0,
        "n_core_fraction_too_small": 0,
        "n_candidates": 0,
        "n_reassembly_verified": 0,
        "n_reassembly_failed": 0,
        "n_parent_filter_would_exclude_eligible": 0,
        "n_parent_filter_would_exclude_all_single_bridges": 0,
        "n_duplicate_canonical_rows": 0,
        "n_stereo_molecules": int(sum(stereo)),
    }
    seen_canon: Counter[str] = Counter(canon)
    mol_audit["n_duplicate_canonical_rows"] = int(sum(v - 1 for v in seen_canon.values() if v > 1))
    candidates: list[dict[str, Any]] = []
    legal_cuts_per_row: dict[int, int] = defaultdict(int)
    for row in range(len(n_atoms_all)):
        n = int(n_atoms_all[row])
        if int(n_components[row]) != 1:
            continue
        graph = TC.row_graph(payload, row)
        x = graph["atom_feat"]
        if np.any(x[:, 0] == 0):
            continue
        mol_audit["n_single_component_no_explicit_h"] += 1
        if n < MIN_C + 1:
            mol_audit["n_rows_without_legal_cut"] += 1
            continue
        u = graph["bond_u"]
        v = graph["bond_v"]
        bf = graph["bond_feat"]
        bridges = RF.find_bridges(n, u, v)
        mol_audit["n_bridges_total"] += len(bridges)
        for e in bridges:
            if int(bf[e, 0]) != 0:  # not a single bond
                mol_audit["n_non_single_bridges_skipped"] += 1
                continue
            mol_audit["n_single_bridges_total"] += 1
            # The parent module's literal filter reads column 2 as "is_in_ring";
            # OGB actually stores ``is_conjugated`` there.  The bridge test
            # already supplies "non-ring", so the chemically correct filter is
            # bond type == SINGLE.  We record the literal filter's effect only.
            if int(bf[e, 2]) != 0:
                mol_audit["n_parent_filter_would_exclude_all_single_bridges"] += 1
            labels = RF.component_labels(n, u, v, skip_edge=e)
            ea, eb = int(u[e]), int(v[e])
            side_a = np.flatnonzero(labels == labels[ea])
            side_b = np.flatnonzero(labels == labels[eb])
            for side, port_s, port_c in ((side_a, ea, eb), (side_b, eb, ea)):
                s_size = int(len(side))
                c_size = int(n - s_size)
                if not (1 <= s_size <= MAX_S):
                    mol_audit["n_s_out_of_range"] += 1
                    continue
                if c_size < MIN_C:
                    mol_audit["n_core_too_small"] += 1
                    continue
                if c_size < CORE_FRACTION * n:
                    mol_audit["n_core_fraction_too_small"] += 1
                    continue
                comp = np.asarray(
                    [i for i in range(n) if i not in set(side.tolist())], dtype=np.int64
                )
                sf = TC.exact_fragment(x, u, v, bf, side, port_s)
                cf = TC.exact_fragment(x, u, v, bf, comp, port_c)
                view = RF.View(
                    s=sf,
                    c=cf,
                    cut_bond_feat=TC.cut_feat(bf, e),
                    cut_bond_u_global=ea,
                    cut_bond_v_global=eb,
                    s_atoms_global=tuple(int(i) for i in sorted(side.tolist())),
                    c_atoms_global=tuple(int(i) for i in comp.tolist()),
                )
                try:
                    RF.verify_reassembly(x, u, v, bf, view)
                except AssertionError:
                    mol_audit["n_reassembly_failed"] += 1
                    raise
                mol_audit["n_reassembly_verified"] += 1
                if int(bf[e, 2]) != 0:
                    mol_audit["n_parent_filter_would_exclude_eligible"] += 1
                candidates.append(
                    {
                        "cut_id": len(candidates),
                        "row": row,
                        "split": str(official[row]),
                        "n_atoms": n,
                        "s_size": s_size,
                        "c_size": c_size,
                        "edge": int(e),
                        "cut": TC.cut_feat(bf, e),
                        "s_port_global": int(port_s),
                        "c_port_global": int(port_c),
                        "s_key": sf.key,
                        "c_key": cf.key,
                        # exact canonical serialization; the SHA-256 above is only
                        # a compact label, never the identity used for grouping.
                        "c_canonical": cf.canonical,
                        "reassembly_verified": 1,
                        "parent_filter_would_exclude": int(int(bf[e, 2]) != 0),
                    }
                )
                legal_cuts_per_row[row] += 1
    mol_audit["multi_component_rows"] = list(audit["multi_component_rows"])
    mol_audit["explicit_h_rows"] = list(audit["explicit_h_rows"])
    mol_audit["n_candidates"] = len(candidates)
    mol_audit["n_rows_with_legal_cut"] = int(sum(1 for v in legal_cuts_per_row.values() if v > 0))
    mol_audit["n_rows_without_legal_cut"] = int(
        mol_audit["n_single_component_no_explicit_h"] - mol_audit["n_rows_with_legal_cut"]
    )
    mol_audit["stereo_definition"] = "canonical isomeric SMILES contains @, / or \\\\"
    mol_audit["parent_filter_note"] = (
        "rd_fragment.enumerate_views also rejects bond_feat[:,2] != 0.  The OGB field "
        "is_conjugated (see ogb.utils.features.bond_to_feature_vector), not is_in_ring; "
        "the bridge test already gives non-ring.  This round therefore uses bond type "
        "SINGLE only and reports the literal filter's audit count."
    )
    rows = []
    for row in range(len(n_atoms_all)):
        eligible = int(n_components[row]) == 1 and not bool(np.any(
            np.asarray(payload["atom_feat"])[
                int(payload["node_ptr"][row]):int(payload["node_ptr"][row + 1])
            ][:, 0] == 0
        ))
        reason = ""
        if int(n_components[row]) != 1:
            reason = "MULTI_COMPONENT"
        elif any(int(x) == row for x in audit["explicit_h_rows"]):
            reason = "EXPLICIT_H"
        elif legal_cuts_per_row.get(row, 0) == 0:
            reason = "NO_LEGAL_CUT"
        rows.append(
            {
                "row_id": int(row),
                "split": str(official[row]),
                "canonical_isomeric_smiles": canon[row],
                "n_atoms": int(n_atoms_all[row]),
                "n_components": int(n_components[row]),
                "single_component": int(int(n_components[row]) == 1),
                "no_explicit_h": int(not any(int(x) == row for x in audit["explicit_h_rows"])),
                "stereo_tokens": int(stereo[row]),
                "eligible_molecule": int(eligible),
                "n_legal_cuts": int(legal_cuts_per_row.get(row, 0)),
                "exclusion_reason": reason,
            }
        )
    return candidates, mol_audit, rows


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------


def _pair_choice_key(item: tuple[int, int], candidates: Sequence[Mapping[str, Any]]) -> tuple:
    ia, ib = item
    ca, cb = candidates[ia], candidates[ib]
    return (
        -int(ca["c_size"]),
        str(ca["c_key"]),
        tuple(sorted((str(ca["s_key"]), str(cb["s_key"])))),
        tuple(sorted((int(ca["edge"]), int(cb["edge"])))),
    )


def build_pairs(
    candidates: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    official = TC.official_split(payload)
    canon = TC.canonical_isomeric_smiles(payload)
    stereo = [TC.stereo_tokens(s) for s in canon]

    groups: dict[tuple[str, tuple[int, int, int]], list[int]] = defaultdict(list)
    for i, cand in enumerate(candidates):
        # Group by the exact canonical serialization, not by its SHA-256 label.
        groups[(str(cand["c_canonical"]), tuple(cand["cut"]))].append(i)
    core_uid_by_key = {key: uid for uid, key in enumerate(sorted(groups))}
    for key, idxs in groups.items():
        uid = int(core_uid_by_key[key])
        for i in idxs:
            candidates[i]["core_uid"] = uid

    pair_candidates: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    group_rows: dict[tuple[str, tuple[int, int, int]], set[int]] = {}
    for core, idxs in groups.items():
        by_row: dict[int, dict[tuple[str, int, tuple[int, int, int]], int]] = defaultdict(dict)
        for i in idxs:
            cand = candidates[i]
            by_row[int(cand["row"])][(str(cand["s_key"]), int(cand["edge"]), tuple(cand["cut"]))] = i
        rows = sorted(by_row)
        group_rows[core] = set(rows)
        for ii in range(len(rows)):
            for jj in range(ii + 1, len(rows)):
                ra, rb = rows[ii], rows[jj]
                if canon[ra] == canon[rb]:
                    continue
                for ia in by_row[ra].values():
                    for ib in by_row[rb].values():
                        if candidates[ia]["s_key"] == candidates[ib]["s_key"]:
                            continue
                        pair_candidates[(ra, rb)].append((int(ia), int(ib)))

    pairs: list[dict[str, Any]] = []
    for (lo, hi), items in pair_candidates.items():
        ia, ib = min(items, key=lambda item: _pair_choice_key(item, candidates))
        ca, cb = candidates[ia], candidates[ib]
        if str(ca["s_key"]) < str(cb["s_key"]):
            a, b = ca, cb
        else:
            a, b = cb, ca
        cut = tuple(int(x) for x in a["cut"])
        core_id = f"{a['c_key']}|{cut[0]},{cut[1]},{cut[2]}"
        transform_id = f"{a['s_key']}>{b['s_key']}|{cut[0]},{cut[1]},{cut[2]}"
        pairs.append(
            {
                "pair_id": f"nt{int(lo):05d}_{int(hi):05d}",
                "row_a": int(a["row"]),
                "row_b": int(b["row"]),
                "row_lo": int(lo),
                "row_hi": int(hi),
                "split_a": str(official[int(a["row"])]),
                "split_b": str(official[int(b["row"])]),
                "cell": _cell_name(str(official[int(a["row"])]), str(official[int(b["row"])])),
                "component_id": None,
                "cut_id_a": int(a["cut_id"]),
                "cut_id_b": int(b["cut_id"]),
                "core_id": core_id,
                "core_uid": int(a["core_uid"]),
                "c_key": str(a["c_key"]),
                "c_canonical": str(a["c_canonical"]),
                "core_size": int(a["c_size"]),
                "cut_bond_type": cut[0],
                "cut_bond_stereo": cut[1],
                "cut_bond_conjugated": cut[2],
                "cut_edge_a": int(a["edge"]),
                "cut_edge_b": int(b["edge"]),
                "s_key_a": str(a["s_key"]),
                "s_key_b": str(b["s_key"]),
                "s_size_a": int(a["s_size"]),
                "s_size_b": int(b["s_size"]),
                "max_s_size": int(max(a["s_size"], b["s_size"])),
                "s_bin": "1-4" if max(a["s_size"], b["s_size"]) <= 4 else "5-8",
                "transform_id": transform_id,
                "stereo_tokens_a": int(stereo[int(a["row"])]),
                "stereo_tokens_b": int(stereo[int(b["row"])]),
                "old_stereo_ok_both": int(
                    not stereo[int(a["row"])] and not stereo[int(b["row"])]
                ),
                "n_candidate_definitions": int(len(items)),
            }
        )
    pairs.sort(key=lambda p: (p["row_lo"], p["row_hi"]))
    audit = {
        "n_candidate_cuts": int(len(candidates)),
        "n_exact_core_groups": int(len(groups)),
        "n_core_groups_with_multiple_rows": int(
            sum(1 for rows in group_rows.values() if len(rows) >= 2)
        ),
        "n_row_pairs_with_candidates": int(len(pair_candidates)),
        "n_pairs": int(len(pairs)),
        "n_pairs_with_multiple_candidate_definitions": int(
            sum(1 for v in pair_candidates.values() if len(v) > 1)
        ),
        "max_candidate_definitions_per_row_pair": int(
            max((len(v) for v in pair_candidates.values()), default=0)
        ),
        "n_duplicate_canonical_row_pairs_skipped": 0,
    }
    return pairs, audit


def _cell_name(a: str, b: str) -> str:
    order = {s: i for i, s in enumerate(TC.SPLITS)}
    parts = sorted((a, b), key=lambda s: order[s])
    return "-".join(s.capitalize() for s in parts)


# ---------------------------------------------------------------------------
# components and support
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_components(
    pairs: Sequence[dict[str, Any]], payload: Mapping[str, Any]
) -> tuple[dict[int, int], dict[int, dict[str, Any]], dict[int, int]]:
    n_rows = len(payload["smiles"])
    uf = UnionFind(n_rows)
    canon = TC.canonical_isomeric_smiles(payload)
    by_smiles: dict[str, list[int]] = defaultdict(list)
    for row, smi in enumerate(canon):
        by_smiles[smi].append(row)
    for rows in by_smiles.values():
        for row in rows[1:]:
            uf.union(rows[0], row)
    for pair in pairs:
        uf.union(int(pair["row_lo"]), int(pair["row_hi"]))
    core_rows: dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        core_rows[int(pair["core_uid"])].update((int(pair["row_lo"]), int(pair["row_hi"])))
    for rows in core_rows.values():
        rows_sorted = sorted(rows)
        for row in rows_sorted[1:]:
            uf.union(rows_sorted[0], row)

    groups: dict[int, list[int]] = defaultdict(list)
    for row in range(n_rows):
        groups[uf.find(row)].append(row)
    root_min = {root: min(rows) for root, rows in groups.items()}
    component_of_row = {int(row): int(root_min[uf.find(row)]) for row in range(n_rows)}

    pair_rows: Counter[int] = Counter()
    for pair in pairs:
        pair_rows[int(pair["row_lo"])] += 1
        pair_rows[int(pair["row_hi"])] += 1
    component_stats: dict[int, dict[str, Any]] = {}
    for pair in pairs:
        cid = component_of_row[int(pair["row_lo"])]
        if component_of_row[int(pair["row_hi"])] != cid:
            raise AssertionError("pair endpoints ended in different components")
        pair["component_id"] = int(cid)
        stats = component_stats.setdefault(
            cid,
            {
                "component_id": int(cid),
                "n_pairs": 0,
                "molecules": set(),
                "splits": set(),
                "cores": set(),
                "transforms": set(),
                "e_t_pairs": 0,
                "e_hh_t_pairs": 0,
                "e_strict_pairs": 0,
            },
        )
        stats["n_pairs"] += 1
        stats["molecules"].update((int(pair["row_lo"]), int(pair["row_hi"])))
        stats["splits"].update((str(pair["split_a"]), str(pair["split_b"])))
        stats["cores"].add(int(pair["core_uid"]))
        stats["transforms"].add(str(pair["transform_id"]))
    return component_of_row, component_stats, pair_rows


def add_support_flags(
    pairs: Sequence[dict[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    frag_payload: Mapping[str, Any],
) -> dict[str, Any]:
    official = TC.official_split(payload)
    selected = TC.selected_views(payload, frag_payload)
    inventory = TC.train_word_inventory(frag_payload)
    train_core_full: set[int] = set()
    train_core_c_only: set[str] = set()
    for cand in candidates:
        if str(official[int(cand["row"])]) != "train":
            continue
        train_core_full.add(int(cand["core_uid"]))
        train_core_c_only.add(str(cand["c_key"]))

    train_train = [p for p in pairs if p["split_a"] == "train" and p["split_b"] == "train"]
    transform_train_comps: dict[str, set[int]] = defaultdict(set)
    for pair in train_train:
        transform_train_comps[str(pair["transform_id"])].add(int(pair["component_id"]))

    for pair in pairs:
        key = (str(pair["c_key"]), (
            int(pair["cut_bond_type"]),
            int(pair["cut_bond_stereo"]),
            int(pair["cut_bond_conjugated"]),
        ))
        pair["core_seen_train_full"] = int(int(pair["core_uid"]) in train_core_full)
        pair["core_seen_train_c_only"] = int(str(pair["c_key"]) in train_core_c_only)
        other_comps = [
            c for c in transform_train_comps.get(str(pair["transform_id"]), set())
            if int(c) != int(pair["component_id"])
        ]
        pair["transform_seen_train_other_component"] = int(len(other_comps) > 0)
        pair["transform_train_other_components"] = int(len(other_comps))
        pair["s_in_inventory_a"] = int(str(pair["s_key_a"]) in inventory)
        pair["s_in_inventory_b"] = int(str(pair["s_key_b"]) in inventory)
        pair["shared_selected_core_view"] = int(
            key in selected.get(int(pair["row_a"]), set())
            and key in selected.get(int(pair["row_b"]), set())
        )
        pair["is_train_a"] = int(pair["split_a"] == "train")
        pair["is_train_b"] = int(pair["split_b"] == "train")
        pair["is_test_a"] = int(pair["split_a"] == "test")
        pair["is_test_b"] = int(pair["split_b"] == "test")
        pair["is_valid_a"] = int(pair["split_a"] == "valid")
        pair["is_valid_b"] = int(pair["split_b"] == "valid")
        pair["e_t"] = int(bool(pair["is_test_a"] or pair["is_test_b"]))
        pair["e_hh_t"] = int(
            not pair["is_train_a"] and not pair["is_train_b"] and bool(pair["e_t"])
        )
        pair["e_strict_full"] = int(
            bool(pair["e_hh_t"])
            and not pair["core_seen_train_full"]
            and bool(pair["transform_seen_train_other_component"])
        )
        pair["e_strict_c_only"] = int(
            bool(pair["e_hh_t"])
            and not pair["core_seen_train_c_only"]
            and bool(pair["transform_seen_train_other_component"])
        )
    audit = {
        "n_train_core_full": int(len(train_core_full)),
        "n_train_core_c_only": int(len(train_core_c_only)),
        "n_train_train_pairs": int(len(train_train)),
        "n_transforms_with_train_train": int(len(transform_train_comps)),
        "n_train_inventory_words": int(len(inventory)),
        "n_pairs_s_in_inventory_both": int(sum(
            1 for p in pairs if p["s_in_inventory_a"] and p["s_in_inventory_b"]
        )),
        "n_pairs_s_in_inventory_one": int(sum(
            1 for p in pairs if bool(p["s_in_inventory_a"]) != bool(p["s_in_inventory_b"])
        )),
        "n_pairs_s_in_inventory_none": int(sum(
            1 for p in pairs if not p["s_in_inventory_a"] and not p["s_in_inventory_b"]
        )),
        "n_pairs_shared_selected_core": int(sum(int(p["shared_selected_core_view"]) for p in pairs)),
    }
    return audit


def scope_stats(
    pairs: Sequence[Mapping[str, Any]], mask: np.ndarray
) -> dict[str, Any]:
    sub = [p for p, keep in zip(pairs, mask.tolist()) if keep]
    rows: set[int] = set()
    comps: set[int] = set()
    cores: set[int] = set()
    transforms: set[str] = set()
    for p in sub:
        rows.update((int(p["row_a"]), int(p["row_b"])))
        comps.add(int(p["component_id"]))
        cores.add(int(p["core_uid"]))
        transforms.add(str(p["transform_id"]))
    split_of = {}
    for p in sub:
        split_of[int(p["row_a"])] = str(p["split_a"])
        split_of[int(p["row_b"])] = str(p["split_b"])
    n_non_train = sum(1 for r in rows if split_of[r] != "train")
    n_test = sum(1 for r in rows if split_of[r] == "test")
    n_valid = sum(1 for r in rows if split_of[r] == "valid")
    return {
        "n_pairs": int(len(sub)),
        "n_molecules": int(len(rows)),
        "n_non_train_molecules": int(n_non_train),
        "n_test_molecules": int(n_test),
        "n_valid_molecules": int(n_valid),
        "n_train_molecules": int(len(rows) - n_non_train),
        "n_components": int(len(comps)),
        "n_cores": int(len(cores)),
        "n_transforms": int(len(transforms)),
        "n_pairs_with_shared_selected_core": int(
            sum(int(p["shared_selected_core_view"]) for p in sub)
        ),
        "n_pairs_s_both_in_inventory": int(
            sum(int(p["s_in_inventory_a"]) and int(p["s_in_inventory_b"]) for p in sub)
        ),
        "n_pairs_s_one_in_inventory": int(
            sum(bool(int(p["s_in_inventory_a"])) != bool(int(p["s_in_inventory_b"])) for p in sub)
        ),
        "n_pairs_s_none_in_inventory": int(
            sum(not int(p["s_in_inventory_a"]) and not int(p["s_in_inventory_b"]) for p in sub)
        ),
    }


# ---------------------------------------------------------------------------
# old 823-pair comparison (read-only, label-free)
# ---------------------------------------------------------------------------


def _old_side(
    payload: Mapping[str, Any], row: int, core_atoms: Sequence[int], var_atoms: Sequence[int], attach: int
) -> dict[str, Any]:
    n = int(payload["graph_n_atoms"][row])
    core = set(int(x) for x in core_atoms)
    var = set(int(x) for x in var_atoms)
    if core & var or core | var != set(range(n)):
        return {"ok": False, "reason": "PARTITION"}
    graph = TC.row_graph(payload, row)
    x = graph["atom_feat"]
    u = graph["bond_u"]
    v = graph["bond_v"]
    bf = graph["bond_feat"]
    if int(attach) not in core:
        return {"ok": False, "reason": "ATTACH_NOT_IN_CORE"}
    edges = [
        k for k in range(len(u))
        if (int(u[k]) == int(attach) and int(v[k]) in var)
        or (int(v[k]) == int(attach) and int(u[k]) in var)
    ]
    if len(edges) != 1:
        return {"ok": False, "reason": "CUT_EDGE"}
    e = int(edges[0])
    frag = TC.exact_fragment(x, u, v, bf, core, int(attach))
    return {
        "ok": True,
        "c_key": frag.key,
        "cut": TC.cut_feat(bf, e),
        "cut_edge": e,
        "core_size": len(core),
        "s_size": len(var),
    }


def old_comparison(
    pairs: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old, _ = TC.old_pair_catalogue()
    official = TC.official_split(payload)
    new_by_rows = {(int(p["row_lo"]), int(p["row_hi"])): p for p in pairs}
    out: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    attr_counts: Counter[str] = Counter()
    for op in old:
        ra, rb = int(op["row_a"]), int(op["row_b"])
        lo, hi = min(ra, rb), max(ra, rb)
        side_a = _old_side(payload, ra, op["core_atoms_a"], op["var_atoms_a"], int(op["attach_a"]))
        side_b = _old_side(payload, rb, op["core_atoms_b"], op["var_atoms_b"], int(op["attach_b"]))
        if not (side_a["ok"] and side_b["ok"]):
            raise ValueError(f"old pair {op['pair_id']} has an invalid stored mapping")
        old_exact = (
            side_a["c_key"] == side_b["c_key"] and side_a["cut"] == side_b["cut"]
        )
        new_pair = new_by_rows.get((lo, hi))
        old_attr = ""
        if side_a["c_key"] != side_b["c_key"] and side_a["cut"] != side_b["cut"]:
            old_attr = "C_AND_CUT_DIFF"
        elif side_a["c_key"] != side_b["c_key"]:
            old_attr = "C_DIFF"
        elif side_a["cut"] != side_b["cut"]:
            old_attr = "CUT_DIFF"
        if new_pair is None:
            status = "ABSENT"
            if old_exact:
                reason = "ABSENT_OLD_CORE_EXACT_UNENUMERATED"
            else:
                reason = "ABSENT_OLD_CORE_NOT_EXACT"
        elif old_exact and (
            str(new_pair["c_key"]) == side_a["c_key"]
            and (
                int(new_pair["cut_bond_type"]),
                int(new_pair["cut_bond_stereo"]),
                int(new_pair["cut_bond_conjugated"]),
            ) == tuple(side_a["cut"])
        ):
            status = "EXACT_SAME_DEFINITION"
            reason = "EXACT_SAME_DEFINITION"
        else:
            status = "PRESENT_DIFFERENT_DEFINITION"
            if not old_exact:
                reason = "PRESENT_OLD_CORE_NOT_EXACT"
            elif int(new_pair["core_size"]) > int(side_a["core_size"]):
                reason = "PRESENT_NEW_LARGER_CORE"
            else:
                reason = "PRESENT_NEW_SAME_SIZE_OTHER_CORE"
        status_counts[status] += 1
        reason_counts[reason] += 1
        if old_attr:
            attr_counts[old_attr] += 1
        out.append(
            {
                "old_pair_id": str(op["pair_id"]),
                "row_a": ra,
                "row_b": rb,
                "row_lo": lo,
                "row_hi": hi,
                "split_a_official": str(official[ra]),
                "split_b_official": str(official[rb]),
                "old_core_size": int(side_a["core_size"]),
                "old_s_size_a": int(side_a["s_size"]),
                "old_s_size_b": int(side_b["s_size"]),
                "old_c_key_a": str(side_a["c_key"]),
                "old_c_key_b": str(side_b["c_key"]),
                "old_cut_a": "%d,%d,%d" % tuple(side_a["cut"]),
                "old_cut_b": "%d,%d,%d" % tuple(side_b["cut"]),
                "old_exact_full_attribute": int(old_exact),
                "old_attribute_difference": old_attr,
                "new_pair_present": int(new_pair is not None),
                "new_pair_id": "" if new_pair is None else str(new_pair["pair_id"]),
                "new_core_size": "" if new_pair is None else int(new_pair["core_size"]),
                "new_s_size_a": "" if new_pair is None else int(new_pair["s_size_a"]),
                "new_s_size_b": "" if new_pair is None else int(new_pair["s_size_b"]),
                "new_max_s_size": "" if new_pair is None else int(new_pair["max_s_size"]),
                "new_stereo_ok_both": "" if new_pair is None else int(new_pair["old_stereo_ok_both"]),
                "status": status,
                "reason": reason,
            }
        )
    summary = {
        "n_old_pairs": int(len(old)),
        "n_old_unique_row_pairs": int(len({(int(p['row_a']), int(p['row_b'])) for p in old})),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "old_full_attribute_difference_counts": dict(sorted(attr_counts.items())),
        "n_old_row_pairs_present": int(sum(1 for p in out if p["new_pair_present"])),
        "n_old_row_pairs_absent": int(sum(1 for p in out if not p["new_pair_present"])),
        "n_exact_definition_intersection": int(status_counts.get("EXACT_SAME_DEFINITION", 0)),
        "n_redefined_old_row_pairs": int(status_counts.get("PRESENT_DIFFERENT_DEFINITION", 0)),
        "n_old_pairs_not_exact_under_full_attributes": int(sum(1 for p in out if not p["old_exact_full_attribute"])),
    }
    return out, summary


def new_source_strata(
    pairs: Sequence[Mapping[str, Any]], old: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_rowpairs = {(int(p["row_lo"]), int(p["row_hi"])) for p in old}
    new_only = [
        p for p in pairs if (int(p["row_lo"]), int(p["row_hi"])) not in old_rowpairs
    ]
    redefined = [
        p for p in pairs if (int(p["row_lo"]), int(p["row_hi"])) in old_rowpairs
        and str(p.get("old_pair_status", "")) != "EXACT_SAME_DEFINITION"
    ]
    bins = ("1-4", "5-8")
    stereo_states = (("stereo_ok_both", 1), ("stereo_ineligible", 0))
    out: list[dict[str, Any]] = []
    for sbin in bins:
        for name, flag in stereo_states:
            sub = [
                p for p in new_only
                if str(p["s_bin"]) == sbin and int(p["old_stereo_ok_both"]) == flag
            ]
            mask = np.zeros(len(pairs), dtype=bool)
            index = {id(p): i for i, p in enumerate(pairs)}
            for p in sub:
                mask[index[id(p)]] = True
            stats = scope_stats(pairs, mask)
            out.append(
                {
                    "stratum": "new_rowpair",
                    "s_bin": sbin,
                    "old_stereo_eligibility": name,
                    "n_pairs": stats["n_pairs"],
                    "n_molecules": stats["n_molecules"],
                    "n_non_train_molecules": stats["n_non_train_molecules"],
                    "n_test_molecules": stats["n_test_molecules"],
                    "n_components": stats["n_components"],
                    "n_e_t_pairs": int(sum(int(p["e_t"]) for p in sub)),
                    "n_e_hh_t_pairs": int(sum(int(p["e_hh_t"]) for p in sub)),
                    "n_e_strict_pairs": int(sum(int(p["e_strict_full"]) for p in sub)),
                }
            )
    # the complete partitions, including all new-only and redefined rows
    for label, sub in (
        ("new_rowpair_all", new_only),
        ("redefined_old_rowpair", redefined),
    ):
        mask = np.zeros(len(pairs), dtype=bool)
        index = {id(p): i for i, p in enumerate(pairs)}
        for p in sub:
            mask[index[id(p)]] = True
        stats = scope_stats(pairs, mask)
        for sbin in ("1-4", "5-8", "all"):
            sub2 = [p for p in sub if sbin == "all" or str(p["s_bin"]) == sbin]
            mask2 = np.zeros(len(pairs), dtype=bool)
            for p in sub2:
                mask2[index[id(p)]] = True
            st = scope_stats(pairs, mask2)
            for name, flag in (("stereo_ok_both", 1), ("stereo_ineligible", 0), ("both", None)):
                sub3 = [
                    p for p in sub2
                    if flag is None or int(p["old_stereo_ok_both"]) == flag
                ]
                mask3 = np.zeros(len(pairs), dtype=bool)
                for p in sub3:
                    mask3[index[id(p)]] = True
                st3 = scope_stats(pairs, mask3)
                out.append(
                    {
                        "stratum": label,
                        "s_bin": sbin,
                        "old_stereo_eligibility": name,
                        "n_pairs": st3["n_pairs"],
                        "n_molecules": st3["n_molecules"],
                        "n_non_train_molecules": st3["n_non_train_molecules"],
                        "n_test_molecules": st3["n_test_molecules"],
                        "n_components": st3["n_components"],
                        "n_e_t_pairs": int(sum(int(p["e_t"]) for p in sub3)),
                        "n_e_hh_t_pairs": int(sum(int(p["e_hh_t"]) for p in sub3)),
                        "n_e_strict_pairs": int(sum(int(p["e_strict_full"]) for p in sub3)),
                    }
                )
    summary = {
        "n_new_rowpairs": int(len(new_only)),
        "n_redefined_old_rowpairs": int(len(redefined)),
        "n_new_rowpairs_smax_1_4": int(sum(1 for p in new_only if p["s_bin"] == "1-4")),
        "n_new_rowpairs_smax_5_8": int(sum(1 for p in new_only if p["s_bin"] == "5-8")),
        "n_new_rowpairs_stereo_ok_both": int(sum(1 for p in new_only if p["old_stereo_ok_both"])),
        "n_new_rowpairs_stereo_ineligible": int(sum(1 for p in new_only if not p["old_stereo_ok_both"])),
    }
    return out, summary


# ---------------------------------------------------------------------------
# writers / entry point
# ---------------------------------------------------------------------------


def _write_csv_gz(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def build_structure() -> dict[str, Any]:
    t0 = time.perf_counter()
    payload = TC.load_payload()
    frag_payload = TC.load_fragment_payload()
    TC.STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

    candidates, mol_audit, official_rows = enumerate_candidates(payload)
    pairs, pair_audit = build_pairs(candidates, payload)
    component_of_row, component_stats, pair_rows = build_components(pairs, payload)
    support_audit = add_support_flags(pairs, candidates, payload, frag_payload)

    # old 823-pair comparison and new-source strata
    old_rows, old_summary = old_comparison(pairs, payload)
    old_status_by_rowpair = {}
    for row in old_rows:
        old_status_by_rowpair[(int(row["row_lo"]), int(row["row_hi"]))] = str(row["status"])
    for pair in pairs:
        pair["old_pair_status"] = old_status_by_rowpair.get(
            (int(pair["row_lo"]), int(pair["row_hi"])), "NOT_IN_OLD_823"
        )
    strata, strata_summary = new_source_strata(pairs, old_rows)

    # scope masks
    index = {id(p): i for i, p in enumerate(pairs)}
    def mask_for(fn):
        mask = np.zeros(len(pairs), dtype=bool)
        for p in pairs:
            if fn(p):
                mask[index[id(p)]] = True
        return mask
    scopes: list[tuple[str, np.ndarray]] = [("ALL", np.ones(len(pairs), dtype=bool))]
    for split_a in TC.SPLITS:
        for split_b in TC.SPLITS:
            if TC.SPLITS.index(split_b) < TC.SPLITS.index(split_a):
                continue
            cell = _cell_name(split_a, split_b)
            scopes.append((cell, mask_for(lambda p, a=split_a, b=split_b: {
                str(p["split_a"]), str(p["split_b"])
            } == {a, b} and _cell_name(str(p["split_a"]), str(p["split_b"])) == cell)))
    scopes.append(("E_T", mask_for(lambda p: bool(int(p["e_t"])))))
    scopes.append(("E_HH_T", mask_for(lambda p: bool(int(p["e_hh_t"])))))
    scopes.append(("E_STRICT", mask_for(lambda p: bool(int(p["e_strict_full"])))))
    scopes.append(("E_STRICT_C_ONLY", mask_for(lambda p: bool(int(p["e_strict_c_only"])))))

    support_rows = []
    for name, mask in scopes:
        stats = scope_stats(pairs, mask)
        support_rows.append({"scope": name, **stats})
    TC.write_csv(TC.STRUCTURE_DIR / "support_by_cell.csv", support_rows)

    # candidate cuts
    cut_columns = [
        "cut_id", "row_id", "split", "n_atoms", "s_size", "c_size", "cut_edge",
        "cut_bond_type", "cut_bond_stereo", "cut_bond_conjugated",
        "s_port_global", "c_port_global", "s_key", "c_key", "core_uid",
        "reassembly_verified", "parent_filter_would_exclude",
    ]
    _write_csv_gz(
        TC.STRUCTURE_DIR / "candidate_cuts.csv.gz",
        [
            {
                "cut_id": int(c["cut_id"]),
                "row_id": int(c["row"]),
                "split": str(c["split"]),
                "n_atoms": int(c["n_atoms"]),
                "s_size": int(c["s_size"]),
                "c_size": int(c["c_size"]),
                "cut_edge": int(c["edge"]),
                "cut_bond_type": int(c["cut"][0]),
                "cut_bond_stereo": int(c["cut"][1]),
                "cut_bond_conjugated": int(c["cut"][2]),
                "s_port_global": int(c["s_port_global"]),
                "c_port_global": int(c["c_port_global"]),
                "s_key": str(c["s_key"]),
                "c_key": str(c["c_key"]),
                "core_uid": int(c["core_uid"]),
                "reassembly_verified": int(c["reassembly_verified"]),
                "parent_filter_would_exclude": int(c["parent_filter_would_exclude"]),
            }
            for c in candidates
        ],
        cut_columns,
    )

    pair_columns = [
        "pair_id", "row_a", "row_b", "row_lo", "row_hi", "split_a", "split_b", "cell",
        "component_id", "cut_id_a", "cut_id_b", "core_id", "core_uid", "c_key", "core_size",
        "cut_bond_type", "cut_bond_stereo", "cut_bond_conjugated",
        "cut_edge_a", "cut_edge_b", "s_key_a", "s_key_b", "s_size_a", "s_size_b",
        "max_s_size", "s_bin", "transform_id", "stereo_tokens_a", "stereo_tokens_b",
        "old_stereo_ok_both", "n_candidate_definitions",
        "core_seen_train_full", "core_seen_train_c_only",
        "transform_seen_train_other_component", "transform_train_other_components",
        "s_in_inventory_a", "s_in_inventory_b", "shared_selected_core_view",
        "is_train_a", "is_train_b", "is_test_a", "is_test_b", "is_valid_a", "is_valid_b",
        "e_t", "e_hh_t", "e_strict_full", "e_strict_c_only", "old_pair_status",
    ]
    TC.write_csv(
        TC.STRUCTURE_DIR / "catalogue_pairs.csv",
        [{key: p.get(key) for key in pair_columns} for p in pairs],
        pair_columns,
    )

    # components / official rows
    official = TC.official_split(payload)
    canon = TC.canonical_isomeric_smiles(payload)
    component_rows = []
    for row in range(len(payload["smiles"])):
        cid = component_of_row.get(row, row)
        component_rows.append(
            {
                "row_id": int(row),
                "split": str(official[row]),
                "canonical_isomeric_smiles": canon[row],
                "component_id": int(cid),
                "in_pair": int(int(pair_rows.get(row, 0)) > 0),
                "n_pairs": int(pair_rows.get(row, 0)),
            }
        )
    TC.write_csv(TC.STRUCTURE_DIR / "components.csv", component_rows)
    component_columns = [
        "component_id", "n_pairs", "n_molecules", "splits", "n_cores", "n_transforms",
        "e_t_pairs", "e_hh_t_pairs", "e_strict_pairs",
    ]
    # recompute the set flags on component stats
    e_t_by_comp: Counter[int] = Counter()
    e_hh_by_comp: Counter[int] = Counter()
    strict_by_comp: Counter[int] = Counter()
    for pair in pairs:
        cid = int(pair["component_id"])
        e_t_by_comp[cid] += int(pair["e_t"])
        e_hh_by_comp[cid] += int(pair["e_hh_t"])
        strict_by_comp[cid] += int(pair["e_strict_full"])
    comp_rows = []
    for cid, stats in sorted(component_stats.items()):
        comp_rows.append(
            {
                "component_id": int(cid),
                "n_pairs": int(stats["n_pairs"]),
                "n_molecules": int(len(stats["molecules"])),
                "splits": "|".join(sorted(stats["splits"])),
                "n_cores": int(len(stats["cores"])),
                "n_transforms": int(len(stats["transforms"])),
                "e_t_pairs": int(e_t_by_comp[cid]),
                "e_hh_t_pairs": int(e_hh_by_comp[cid]),
                "e_strict_pairs": int(strict_by_comp[cid]),
            }
        )
    TC.write_csv(TC.STRUCTURE_DIR / "pair_components.csv", comp_rows, component_columns)

    core_rows: dict[int, dict[str, Any]] = {}
    for pair in pairs:
        cid = int(pair["core_uid"])
        rec = core_rows.setdefault(
            cid,
            {
                "core_uid": cid,
                "core_id": str(pair["core_id"]),
                "c_key": str(pair["c_key"]),
                "c_canonical": str(pair["c_canonical"]),
                "core_size": int(pair["core_size"]),
                "cut_bond_type": int(pair["cut_bond_type"]),
                "cut_bond_stereo": int(pair["cut_bond_stereo"]),
                "cut_bond_conjugated": int(pair["cut_bond_conjugated"]),
                "molecules": set(),
                "pairs": 0,
                "components": set(),
                "train_core_seen_full": int(pair["core_seen_train_full"]),
                "train_core_seen_c_only": int(pair["core_seen_train_c_only"]),
            },
        )
        rec["molecules"].update((int(pair["row_a"]), int(pair["row_b"])))
        rec["pairs"] += 1
        rec["components"].add(int(pair["component_id"]))
    core_out = []
    for rec in sorted(core_rows.values(), key=lambda r: int(r["core_uid"])):
        core_out.append(
            {
                "core_uid": rec["core_uid"],
                "core_id": rec["core_id"],
                "c_key": rec["c_key"],
                "c_canonical": rec["c_canonical"],
                "core_size": rec["core_size"],
                "cut_bond_type": rec["cut_bond_type"],
                "cut_bond_stereo": rec["cut_bond_stereo"],
                "cut_bond_conjugated": rec["cut_bond_conjugated"],
                "n_molecules": len(rec["molecules"]),
                "n_pairs": rec["pairs"],
                "n_components": len(rec["components"]),
                "train_core_seen_full": rec["train_core_seen_full"],
                "train_core_seen_c_only": rec["train_core_seen_c_only"],
                "component_ids": "|".join(str(x) for x in sorted(rec["components"])),
            }
        )
    TC.write_csv(TC.STRUCTURE_DIR / "core_table.csv", core_out)

    TC.write_csv(TC.STRUCTURE_DIR / "official_rows.csv", official_rows)
    TC.write_csv(TC.STRUCTURE_DIR / "old_pair_comparison.csv", old_rows)
    TC.write_csv(TC.STRUCTURE_DIR / "new_source_strata.csv", strata)

    old_split_bad = TC.old_split_mismatch_count(payload)
    support = {
        "definition": {
            "max_s_atoms": MAX_S,
            "min_c_atoms": MIN_C,
            "core_min_fraction": CORE_FRACTION,
            "cut": "bridge with OGB bond type SINGLE (0); non-ring by bridge",
            "identity": "exact coloured incidence graph of C (9 atom columns), port, 3 cut-bond columns",
            "direction": "S native key lexicographic smaller = A; delta = B - A",
            "one_definition_per_row_pair": [
                "max core_size", "c_key", "sorted(s_key_a,s_key_b)",
                "sorted(cut_edge_a,cut_edge_b)",
            ],
            "component_merge": [
                "identical full canonical isomeric SMILES rows",
                "pair endpoints share a molecule",
                "pairs share the exact matched core_id",
            ],
        },
        "counts": {
            **mol_audit,
            **pair_audit,
            **support_audit,
            "old_stored_split_mismatches_reproduced": int(old_split_bad),
        },
        "scopes": {row["scope"]: row for row in support_rows},
        "old_823_comparison": old_summary,
        "new_source_strata": strata_summary,
        "selected_view_coverage": {
            "n_pairs_total": int(len(pairs)),
            "n_pairs_shared_selected_core": int(sum(int(p["shared_selected_core_view"]) for p in pairs)),
            "fraction_shared_selected_core": float(np.mean([int(p["shared_selected_core_view"]) for p in pairs])),
            "note": ("offline C sharing does not imply the fixed <=4 selected views of both "
                     "endpoints contain that C; this field is coverage only and never changes "
                     "the evaluation subset"),
        },
        "inventory_note": (
            "S inventory is the existing dictionary's 702 Train S-word candidates from the "
            "fixed <=4 selected Train views; it is not the larger set of all legal Train cuts."
        ),
    }
    TC.write_json(TC.STRUCTURE_DIR / "support.json", support)
    TC.write_json(
        TC.STRUCTURE_DIR / "exclusions.json",
        {
            "multi_component_rows": mol_audit["multi_component_rows"],
            "explicit_h_rows": mol_audit["explicit_h_rows"],
            "n_duplicate_canonical_rows": mol_audit["n_duplicate_canonical_rows"],
            "n_reassembly_failed": mol_audit["n_reassembly_failed"],
            "n_rdkit_parse_failures": 0,
            "stereo_policy": ("stereo is kept; every candidate carries all OGB atom/bond columns; "
                              "no chirality or bond stereo is removed"),
            "exclusion_reason": ("only the single multi-component row (official row 1646, valid) "
                                 "is excluded from edit evaluation; no explicit-H molecule exists; "
                                 "molecules with no legal cut remain in the official mapping but "
                                 "contribute no pair"),
            "parent_literal_filter_note": mol_audit["parent_filter_note"],
        },
    )

    # freeze the structure catalogue before labels/predictions are connected
    files = {}
    for path in sorted(TC.STRUCTURE_DIR.rglob("*")):
        if path.is_file() and path.name != "structure_freeze.json":
            files[str(path.relative_to(TC.STRUCTURE_DIR))] = TC.sha256_file(path)
    freeze = {
        "round_id": TC.load_config()["round_id"],
        "definition_id": TC.load_config()["definition_id"],
        "config_sha256": TC.sha256_file(TC.CONFIG_PATH),
        "protocol_sha256": TC.sha256_file(TC.PROTOCOL_PATH),
        "files": files,
        "structure_hash": TC.sha256_json(files),
        "n_pairs": int(len(pairs)),
        "n_candidates": int(len(candidates)),
        "n_components": int(len(component_stats)),
        "n_E_T": int(sum(int(p["e_t"]) for p in pairs)),
        "n_E_HH_T": int(sum(int(p["e_hh_t"]) for p in pairs)),
        "n_E_STRICT": int(sum(int(p["e_strict_full"]) for p in pairs)),
        "seconds": time.perf_counter() - t0,
        "note": "frozen before labels or frozen model predictions were read",
    }
    TC.write_json(TC.STRUCTURE_DIR / "structure_freeze.json", freeze)
    return {
        "candidates": len(candidates),
        "pairs": len(pairs),
        "components": len(component_stats),
        "support": support,
        "old": old_summary,
        "strata": strata_summary,
        "freeze": freeze,
        "seconds": time.perf_counter() - t0,
    }


def main(argv=None) -> int:
    result = build_structure()
    print(json.dumps(
        {
            "candidates": result["candidates"],
            "pairs": result["pairs"],
            "components": result["components"],
            "E_T": result["freeze"]["n_E_T"],
            "E_HH_T": result["freeze"]["n_E_HH_T"],
            "E_STRICT": result["freeze"]["n_E_STRICT"],
            "structure_hash": result["freeze"]["structure_hash"],
            "seconds": result["seconds"],
            "old": result["old"],
            "strata": result["strata"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
