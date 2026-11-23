"""Real-edit catalogue for route C (label-free, frozen before labels are read).

Every eligible official-Train molecule is cut at every non-ring single-bond
bridge.  The variable side S has 1..8 heavy atoms; the core C has >= 10 heavy
atoms and >= 75 % of the parent.  Both fragments keep all 9 atom / 3 bond raw
columns, the port, and the three cut-bond columns; every cut is verified by the
parent reassembly check.  Identity uses the parent round's exact coloured
incidence-graph certificate (pynauty), never a hash bucket.

Pairs are formed inside an exact ``(C, cut)`` group, one definition per unordered
molecule pair (parent choice rule), skipping identical molecules.  Orientation is
fixed by the lexicographic order of the variable-fragment exact key, so
``Delta y = y_b - y_a`` can only be evaluated after this file is frozen.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sr_common as C
import sr_data as D

import tt_common as TC
import tt_catalogue as CAT
import rd_fragment as RF

MAX_S = 8
MIN_C = 10
CORE_FRACTION = 0.75
FREEZE_DIR = C.ANALYSIS_DIR / "catalogue"


def _row_graph(bundle: Any, row: int) -> dict[str, Any]:
    v = D.view(bundle, row)
    return {
        "row": int(row),
        "n_atoms": int(v["n_atoms"]),
        "n_bonds": int(v["n_bonds"]),
        "atom_feat": np.asarray(v["atom_feat"], np.int64),
        "bond_u": np.asarray(v["bond_u"], np.int64),
        "bond_v": np.asarray(v["bond_v"], np.int64),
        "bond_feat": np.asarray(v["bond_feat"], np.int64),
        "n_components": int(v["graph_n_components"]),
        "smiles": str(v["smiles"]),
    }


def _canonical_smiles(smiles: Sequence[str]) -> list[str]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for text in smiles:
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            out.append(f"__UNPARSED__{text}")
        else:
            out.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    return out


def enumerate_candidates(bundle: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """All legal cuts of the official Train rows, with the parent's counters."""
    rows = D.all_rows(bundle).tolist()
    audit: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    cuts_per_row: dict[int, int] = defaultdict(int)
    for row in rows:
        g = _row_graph(bundle, row)
        n = g["n_atoms"]
        if g["n_components"] != 1:
            audit["skipped_multi_component"] += 1
            continue
        if np.any(g["atom_feat"][:, 0] == 0):
            audit["skipped_explicit_h"] += 1
            continue
        audit["eligible_molecules"] += 1
        if n < MIN_C + 1:
            audit["skipped_too_small"] += 1
            continue
        u, v, bf = g["bond_u"], g["bond_v"], g["bond_feat"]
        bridges = RF.find_bridges(n, u, v)
        audit["bridges_total"] += len(bridges)
        for e in bridges:
            if int(bf[e, 0]) != 0:
                audit["skipped_not_single"] += 1
                continue
            audit["single_bridges"] += 1
            labels = RF.component_labels(n, u, v, skip_edge=e)
            ea, eb = int(u[e]), int(v[e])
            side_a = np.flatnonzero(labels == labels[ea])
            side_b = np.flatnonzero(labels == labels[eb])
            for side, port_s, port_c in ((side_a, ea, eb), (side_b, eb, ea)):
                s_size = int(len(side))
                c_size = int(n - s_size)
                if not (1 <= s_size <= MAX_S):
                    audit["rejected_s_size"] += 1
                    continue
                if c_size < MIN_C:
                    audit["rejected_core_small"] += 1
                    continue
                if c_size < CORE_FRACTION * n:
                    audit["rejected_core_fraction"] += 1
                    continue
                comp = np.asarray([i for i in range(n) if i not in set(side.tolist())],
                                  dtype=np.int64)
                sf = TC.exact_fragment(g["atom_feat"], u, v, bf, side, port_s)
                cf = TC.exact_fragment(g["atom_feat"], u, v, bf, comp, port_c)
                view = RF.View(
                    s=sf, c=cf, cut_bond_feat=tuple(int(x) for x in bf[e]),
                    cut_bond_u_global=ea, cut_bond_v_global=eb,
                    s_atoms_global=tuple(int(i) for i in sorted(side.tolist())),
                    c_atoms_global=tuple(int(i) for i in comp.tolist()))
                RF.verify_reassembly(g["atom_feat"], u, v, bf, view)
                audit["verified_views"] += 1
                candidates.append({
                    "cut_id": len(candidates), "row": int(row), "n_atoms": n,
                    "s_size": s_size, "c_size": c_size, "edge": int(e),
                    "cut": tuple(int(x) for x in bf[e]),
                    "s_port": int(port_s), "c_port": int(port_c),
                    "s_key": sf.key, "c_key": cf.key,
                    "c_canonical": cf.canonical,
                })
                cuts_per_row[int(row)] += 1
    audit["n_candidates"] = len(candidates)
    audit["rows_with_legal_cut"] = len(cuts_per_row)
    audit["rows_without_legal_cut"] = int(audit["eligible_molecules"]
                                          - len(cuts_per_row))
    audit["max_cuts_per_row"] = int(max(cuts_per_row.values(), default=0))
    audit["parent_coordinate_official_train_pairs"] = 2133
    audit["definition"] = {
        "cut": "bridge, OGB bond type SINGLE (0); the bridge test supplies non-ring",
        "s_size": [1, MAX_S], "c_min_atoms": MIN_C,
        "c_min_fraction": CORE_FRACTION,
        "identity": "parent exact coloured-incidence-graph certificate (pynauty)",
        "reassembly": "parent verify_reassembly on every view",
        "note": "OGB bond column 2 is is_conjugated, not is_in_ring; it is not used "
                "as a ring filter",
    }
    return candidates, dict(audit)


def _choice_key(item: tuple[int, int],
                candidates: Sequence[Mapping[str, Any]]) -> tuple:
    """The parent round's fixed rule for one definition per molecule pair."""
    return CAT._pair_choice_key(item, candidates)


def build_pairs(bundle: Any, candidates: Sequence[Mapping[str, Any]],
                canon: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, tuple[int, int, int]], list[int]] = defaultdict(list)
    for i, cand in enumerate(candidates):
        groups[(str(cand["c_canonical"]), tuple(cand["cut"]))].append(i)
    core_uid = {key: uid for uid, key in enumerate(sorted(groups))}
    for key, idxs in groups.items():
        uid = int(core_uid[key])
        for i in idxs:
            candidates[i]["core_uid"] = uid

    by_core_row: dict[tuple[str, tuple[int, int, int]],
                      dict[int, dict[tuple[str, int, tuple[int, int, int]], int]]] = {}
    for key, idxs in groups.items():
        table: dict[int, dict[tuple[str, int, tuple[int, int, int]], int]] = defaultdict(dict)
        for i in idxs:
            cand = candidates[i]
            table[int(cand["row"])][(str(cand["s_key"]), int(cand["edge"]),
                                     tuple(cand["cut"]))] = i
        by_core_row[key] = table

    pair_candidates: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for key, table in by_core_row.items():
        rows = sorted(table)
        for ra, rb in itertools.combinations(rows, 2):
            if canon[ra] == canon[rb]:
                continue
            for ia in table[ra].values():
                for ib in table[rb].values():
                    if candidates[ia]["s_key"] == candidates[ib]["s_key"]:
                        continue
                    pair_candidates[(ra, rb)].append((int(ia), int(ib)))

    pairs: list[dict[str, Any]] = []
    for (lo, hi), items in pair_candidates.items():
        ia, ib = min(items, key=lambda item: _choice_key(item, candidates))
        ca, cb = candidates[ia], candidates[ib]
        if str(ca["s_key"]) < str(cb["s_key"]):
            a, b = ca, cb
        else:
            a, b = cb, ca
        pairs.append({
            "pair_id": f"px{int(lo):05d}_{int(hi):05d}",
            "row_a": int(a["row"]), "row_b": int(b["row"]),
            "row_lo": int(lo), "row_hi": int(hi),
            "edge_a": int(a["edge"]), "edge_b": int(b["edge"]),
            "s_port_a": int(a["s_port"]), "s_port_b": int(b["s_port"]),
            "c_port_a": int(a["c_port"]), "c_port_b": int(b["c_port"]),
            "s_size_a": int(a["s_size"]), "s_size_b": int(b["s_size"]),
            "c_size": int(a["c_size"]),
            "cut_a": tuple(int(x) for x in a["cut"]),
            "cut_b": tuple(int(x) for x in b["cut"]),
            "core_uid": int(a["core_uid"]),
            "s_key_a": str(a["s_key"]), "s_key_b": str(b["s_key"]),
            "c_key": str(a["c_key"]),
            "c_canonical": str(a["c_canonical"]),
            "n_candidate_definitions": int(len(items)),
        })
    pairs.sort(key=lambda p: (p["row_lo"], p["row_hi"]))
    for uid, pair in enumerate(pairs):
        pair["pair_uid"] = uid
        pair["transform_id"] = (f"{pair['s_key_a']}>{pair['s_key_b']}|"
                                f"{pair['cut_a'][0]},{pair['cut_a'][1]},{pair['cut_a'][2]}")
    audit = {
        "n_candidate_cuts": int(len(candidates)),
        "n_exact_core_groups": int(len(groups)),
        "n_core_groups_with_multiple_rows": int(sum(1 for t in by_core_row.values()
                                                    if len(t) >= 2)),
        "n_row_pairs_with_candidates": int(len(pair_candidates)),
        "n_pairs": int(len(pairs)),
        "n_pairs_with_multiple_definitions": int(sum(
            1 for v in pair_candidates.values() if len(v) > 1)),
        "max_definitions_per_row_pair": int(max((len(v) for v in
                                                 pair_candidates.values()), default=0)),
        "n_transform_ids": int(len({p["transform_id"] for p in pairs})),
        "n_core_keys": int(len({p["c_key"] for p in pairs})),
    }
    return pairs, audit


def _bytes(values: Sequence[str]) -> np.ndarray:
    return np.asarray([str(v).encode("ascii") for v in values],
                      dtype="S64") if values else np.zeros(0, "S64")


def freeze(bundle: Any, out_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates, enum_audit = enumerate_candidates(bundle)
    official = np.sort(np.asarray(bundle.raw["train_idx"], np.int64)).tolist()
    canon_list = [""] * int(np.max(official) + 1)
    smiles = np.asarray(bundle.raw["smiles"])
    can = _canonical_smiles([str(smiles[r]) for r in official])
    for r, text in zip(official, can):
        canon_list[r] = text
    pairs, pair_audit = build_pairs(bundle, candidates, canon_list)

    def cat(pairs_, key, dtype=np.int64, width=None):
        if width is not None:
            arr = np.asarray([p[key] for p in pairs_], dtype=dtype)
            return arr
        return np.asarray([p[key] for p in pairs_], dtype=dtype)

    arrays = {
        "pair_uid": np.arange(len(pairs), dtype=np.int32),
        "row_a": cat(pairs, "row_a", np.int32),
        "row_b": cat(pairs, "row_b", np.int32),
        "edge_a": cat(pairs, "edge_a", np.int16),
        "edge_b": cat(pairs, "edge_b", np.int16),
        "s_port_a": cat(pairs, "s_port_a", np.int16),
        "s_port_b": cat(pairs, "s_port_b", np.int16),
        "c_port_a": cat(pairs, "c_port_a", np.int16),
        "c_port_b": cat(pairs, "c_port_b", np.int16),
        "s_size_a": cat(pairs, "s_size_a", np.int8),
        "s_size_b": cat(pairs, "s_size_b", np.int8),
        "c_size": cat(pairs, "c_size", np.int16),
        "core_uid": cat(pairs, "core_uid", np.int32),
        "cut_a": np.asarray([p["cut_a"] for p in pairs], np.int8).reshape(-1, 3),
        "cut_b": np.asarray([p["cut_b"] for p in pairs], np.int8).reshape(-1, 3),
        "s_key_a": _bytes([p["s_key_a"] for p in pairs]),
        "s_key_b": _bytes([p["s_key_b"] for p in pairs]),
        "c_key": _bytes([p["c_key"] for p in pairs]),
        "transform_id": _bytes([p["transform_id"] for p in pairs]),
        "n_candidate_definitions": cat(pairs, "n_candidate_definitions", np.int16),
        # candidate-level legal cuts (label-free); used for the strict
        # "core seen in any legal Fit cut" stratum
        "cand_row": np.asarray([c["row"] for c in candidates], np.int32),
        "cand_core_uid": np.asarray([c["core_uid"] for c in candidates], np.int32),
        "cand_c_key": _bytes([c["c_key"] for c in candidates]),
        "cand_s_key": _bytes([c["s_key"] for c in candidates]),
        "cand_cut": np.asarray([c["cut"] for c in candidates], np.int8).reshape(-1, 3),
        "cand_edge": np.asarray([c["edge"] for c in candidates], np.int16),
    }
    npz_path = out_dir / "pairs_frozen.npz"
    np.savez_compressed(npz_path, **arrays)
    payload = {
        "generated_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "labels_read": False,
        "definition": enum_audit["definition"],
        "enumeration_audit": {k: v for k, v in enum_audit.items()
                              if k != "definition"},
        "pair_audit": pair_audit,
        "pairs_file": str(npz_path.relative_to(C.REPO_ROOT)),
        "pairs_sha256": C.sha256_file(npz_path),
        "pair_uid_sha256": C.array_sha256(arrays["pair_uid"]),
        "rows_sha256": C.array_sha256(arrays["row_a"], arrays["row_b"]),
        "cut_sha256": C.array_sha256(arrays["cut_a"], arrays["cut_b"]),
        "keys_sha256": C.array_sha256(arrays["s_key_a"], arrays["s_key_b"],
                                      arrays["c_key"]),
        "n_pairs": int(len(pairs)),
        "seconds": float(time.time() - t0),
        "parent_coordinate_check": {
            "expected_pairs": 2133, "found_pairs": int(len(pairs)),
            "match": bool(len(pairs) == 2133)},
        "d0_fit_rows_sha256": C.array_sha256(np.asarray(bundle.fit_rows)),
        "d0_eval_rows_sha256": C.array_sha256(np.asarray(bundle.eval_rows)),
    }
    C.write_json(out_dir / "catalogue_freeze.json", payload)
    return payload


def load_frozen() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    meta = C.read_json(FREEZE_DIR / "catalogue_freeze.json")
    path = C.REPO_ROOT / meta["pairs_file"]
    if C.sha256_file(path) != meta["pairs_sha256"]:
        raise ValueError("frozen pair catalogue changed")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    return arrays, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze"])
    parser.add_argument("--out", default=str(FREEZE_DIR))
    args = parser.parse_args()
    C.assert_pins()
    bundle = D.load_bundle("D0")
    if args.action == "freeze":
        payload = freeze(bundle, Path(args.out))
        print(json.dumps(C.jsonable(payload), indent=1, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())