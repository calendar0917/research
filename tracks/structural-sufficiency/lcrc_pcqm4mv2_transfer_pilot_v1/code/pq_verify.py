"""Verify one merged feature cache: structure, S1/S2 identity, exact content.

Checks (all label-blind):

* pointer shapes, value ranges, dtypes, finiteness, index bounds
  (``lcrc_data.validate_cache`` + the round's structure audit),
* the all-atom S2 count identity ``#S2(root) == sum_{w in N(root)} (deg(w)-1)``
  and ``#S1 == 2 * #bonds``,
* exact re-derivation of a random sample: the stored arrays of molecule ``row``
  must be reproduced byte-for-byte from its stored SMILES through the frozen
  adapter chain (atom features, bond features, scalars, global statistics, S1
  codes, S2 codes / ends / closure flags),
* gid integrity: unique, inside the official split, disjoint between the Train
  and Validation caches, and equal to the requested row list,
* label alignment: the label artifact is finite exactly on the cache gids.

``--official`` additionally re-reads the official dataset and checks that the
stored SMILES of every checked row is the official SMILES of its gid.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_common as C  # noqa: E402
import pq_data as D  # noqa: E402

import feature_spec as FS  # noqa: E402
import lcrc_data as PREP_DATA  # noqa: E402
import mb2_rdkit as R  # noqa: E402
import molecular_graph as MG  # noqa: E402

FIELDS = (
    "atom_feat",
    "atom_scalar",
    "bond_feat",
    "bond_type",
    "bond_index",
    "nbr_code",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
    "global",
)


def _molecule_slice(cache, row: int) -> dict[str, Any]:
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
    s0, s1 = int(nbr_ptr[a0]), int(nbr_ptr[a1])
    return {
        "n_atoms": a1 - a0,
        "n_bonds": b1 - b0,
        "atom_feat": np.asarray(cache["atom_feat"][a0:a1]),
        "atom_scalar": np.asarray(cache["atom_scalar"][a0:a1]),
        "bond_feat": np.asarray(cache["bond_feat"][b0:b1]),
        "bond_type": np.asarray(cache["bond_type"][b0:b1]),
        "bond_index": np.asarray(cache["bond_index"][b0:b1]),
        "nbr_code": np.asarray(cache["nbr_code"][s0:s1]),
        "path_cat": np.asarray(cache["path_cat"][p0:p1]),
        "path_end63": np.asarray(cache["path_end63"][p0:p1]),
        "path_end": np.asarray(cache["path_end"][p0:p1]),
        "path_closure": np.asarray(cache["path_closure"][p0:p1]),
        "global": np.asarray(cache["global"][row]),
    }


def exact_sample_check(cache, spec, rows: Sequence[int]) -> dict[str, Any]:
    """Rebuild the stored molecule from its SMILES and compare every field."""
    mismatches: list[dict[str, Any]] = []
    modes = {"strict": 0, "unsanitized_fallback": 0}
    checked = 0
    for row in rows:
        row = int(row)
        smi = cache["smiles"][row]
        gid = int(np.asarray(cache["gids"])[row])
        try:
            raw, mode = R.smiles_to_raw_row_relaxed(smi, gid)
            graph = R.raw_row_to_graph(raw, gid, spec)
            mol = PREP_DATA.build_molecule(graph, spec, validate=True)
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            mismatches.append({"row": row, "error": f"{type(exc).__name__}: {exc}"})
            continue
        modes[mode] += 1
        stored = _molecule_slice(cache, row)
        rebuilt = {
            "atom_feat": np.asarray(mol["atom_feat"]),
            "atom_scalar": np.asarray(mol["atom_scalar"]),
            "bond_feat": np.asarray(mol["bond_feat"]),
            "bond_type": np.asarray(mol["bond_type"]),
            "bond_index": np.asarray(mol["bond_index"]),
            "nbr_code": np.asarray(mol["nbr_code"]),
            "path_cat": np.asarray(mol["path_cat"]),
            "path_end63": np.asarray(mol["path_end63"]),
            "path_end": np.asarray(mol["path_end"]),
            "path_closure": np.asarray(mol["path_closure"]),
            "global": np.asarray(mol["global"]),
        }
        if stored["n_atoms"] != int(mol["n_atoms"]) or stored["n_bonds"] != int(mol["n_bonds"]):
            mismatches.append({"row": row, "kind": "counts"})
            continue
        for key in FIELDS:
            if not np.array_equal(stored[key], rebuilt[key]):
                mismatches.append({"row": row, "kind": f"content:{key}"})
        # independent S1/S2 enumeration from the physical bond graph
        analysis = MG.analyse_graph(graph)
        s1 = [code for root in MG.enumerate_s1(graph.atom_type, analysis.adj, spec.n_bond_types) for code in root]
        s2 = [rec for root in MG.enumerate_s2(graph.atom_type, analysis.adj, spec.n_bond_types, analysis.dist) for rec in root]
        if not np.array_equal(stored["nbr_code"], np.asarray(s1, dtype=np.int32)):
            mismatches.append({"row": row, "kind": "s1_enumeration"})
        if not np.array_equal(stored["path_cat"], np.asarray([r["path_cat"] for r in s2], dtype=np.int32)):
            mismatches.append({"row": row, "kind": "s2_enumeration_cat"})
        if not np.array_equal(stored["path_closure"], np.asarray([r["closure"] for r in s2], dtype=np.int8)):
            mismatches.append({"row": row, "kind": "s2_enumeration_closure"})
        if int(stored["nbr_code"].size) != 2 * stored["n_bonds"]:
            mismatches.append({"row": row, "kind": "s1_count"})
        checked += 1
    return {
        "n_requested": len(rows),
        "n_checked": int(checked),
        "rows": [int(r) for r in rows],
        "parse_modes": modes,
        "mismatches": mismatches[:20],
        "all_fields_exact": bool(not mismatches and checked == len(rows)),
    }


def verify(tag: str, *, sample: int = 24, official: bool = False, seed: int = 20261110) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = FS.load_spec(C.ANALYSIS / "pcqm_spec.json")
    path = C.CACHE / tag
    cache = D.load_cache(path)
    structure = D.cache_structure_audit(cache, spec)
    n = int(cache["n_molecules"])
    rng = np.random.default_rng(int(seed))
    k = min(int(sample), n)
    rows = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64).tolist()
    rows = [0, n - 1] + [r for r in rows if r not in (0, n - 1)]
    exact = exact_sample_check(cache, spec, rows)

    gids = np.asarray(cache["gids"], dtype=np.int64)
    labels = np.load(C.CACHE / "pcqm_labels_trainvalid.npy").astype(np.float64)
    gid_checks = {
        "n_gids": int(gids.size),
        "n_unique_gids": int(np.unique(gids).size),
        "gids_unique": bool(np.unique(gids).size == gids.size),
        "gids_non_negative": bool((gids >= 0).all()),
        "labels_finite_on_gids": bool(np.isfinite(labels[gids]).all()),
        "labels_finite_fraction": float(np.isfinite(labels[gids]).mean()),
        "gid_max": int(gids.max()),
        "gid_min": int(gids.min()),
    }
    split_sizes = C.read_json(C.ANALYSIS / "pcqm_data_audit.json")["split_sizes"]
    n_total = int(C.read_json(C.ANALYSIS / "pcqm_data_audit.json")["dataset"]["n_total"])
    gid_checks["gids_within_dataset"] = bool(gids.max() < n_total)
    official_check: dict[str, Any] = {"requested": bool(official)}
    if official:
        info = D.load_official()
        splits = info["split"]
        membership = {}
        for name in C.SPLIT_KEYS:
            rows_set = np.asarray(splits[name], dtype=np.int64)
            membership[name] = int(np.isin(gids, rows_set).sum())
        official_check.update(
            {
                "split_membership_counts": membership,
                "all_gids_in_the_expected_split": {
                    "train": membership["train"] == gids.size,
                    "valid": membership["valid"] == gids.size,
                },
                "smiles_match_official": bool(
                    all(cache["smiles"][int(r)] == info["smiles"][int(gids[int(r)])] for r in rows)
                ),
                "checked_rows": [int(r) for r in rows],
                "split_sizes": split_sizes,
            }
        )

    # Train / Validation gid disjointness (only when both caches exist)
    disjoint = None
    train_dir, valid_dir = C.CACHE / "pcqm_train1m", C.CACHE / "pcqm_valid"
    if train_dir.is_dir() and valid_dir.is_dir():
        tg = np.load(C.CACHE / "pcqm_subset_rows.npy").astype(np.int64)
        vg = np.asarray(D.load_cache(valid_dir)["gids"], dtype=np.int64)
        disjoint = {
            "train_rows": int(tg.size),
            "valid_rows": int(vg.size),
            "intersection": int(np.intersect1d(tg, vg).size),
            "ok": bool(np.intersect1d(tg, vg).size == 0),
        }

    manifest = C.read_json(C.ANALYSIS / "pcqm_cache_manifest.json") if (
        C.ANALYSIS / "pcqm_cache_manifest.json"
    ).exists() else {}
    tree_sha = C.sha256_tree(path)
    manifest_sha = (manifest.get(tag) or {}).get("cache_sha256")
    gates = {
        "structure_ok": bool(structure["ok"]),
        "s2_all_atom_formula_identity": bool(structure["formula_s2"]["ok"]),
        "s1_equals_2_bonds": bool(structure["formula_s2"]["s1_total_equals_2_bonds"]),
        "sample_exact_rebuild": bool(exact["all_fields_exact"]),
        "gids_unique": bool(gid_checks["gids_unique"]),
        "labels_finite_on_gids": bool(gid_checks["labels_finite_on_gids"]),
        "gids_within_dataset": bool(gid_checks["gids_within_dataset"]),
    }
    if manifest_sha is not None:
        gates["cache_tree_hash_matches_manifest"] = bool(tree_sha == manifest_sha)
    if disjoint is not None:
        gates["train_valid_gid_disjoint"] = bool(disjoint["ok"])
    if official:
        expected = "train" if tag == "pcqm_train1m" else "valid"
        gates["gids_in_expected_official_split"] = bool(
            official_check["all_gids_in_the_expected_split"][expected]
        )
        gates["smiles_match_official"] = bool(official_check["smiles_match_official"])

    payload = {
        "schema": "lcrc-pcqm4mv2-cache-verify-v1",
        "round_id": C.ROUND_ID,
        "tag": tag,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "hostname": C.host_facts()["hostname"],
        "cache": C.track_rel(path),
        "cache_tree_sha256": tree_sha,
        "manifest_cache_sha256": manifest_sha,
        "seconds": float(time.perf_counter() - t0),
        "feature_spec_hash": spec.hash(),
        "structure": structure,
        "exact_rebuild_sample": exact,
        "gids": gid_checks,
        "train_valid_disjointness": disjoint,
        "official_crosscheck": official_check,
        "gates": gates,
    }
    payload["status"] = "OK" if all(gates.values()) else "FAILED"
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=["pcqm_train1m", "pcqm_valid"])
    parser.add_argument("--sample", type=int, default=24)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    payload = verify(args.tag, sample=int(args.sample), official=bool(args.official))
    out = Path(args.out) if args.out else C.ANALYSIS / f"{args.tag}_cache_verify.json"
    C.write_json(out, payload)
    C.log(f"[verify] {args.tag}: {payload['status']} -> {C.track_rel(out)}")
    return 0 if payload["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
