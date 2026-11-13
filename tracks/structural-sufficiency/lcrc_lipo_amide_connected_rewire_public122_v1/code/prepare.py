"""Structure-only preparation: donor catalog, populations, ranges, frozen manifests.

Everything in this step is label-blind and prediction-blind.  It produces the
frozen protocol inputs that the remote inference step consumes and nothing else:

* ``analysis/inputs_manifest.json`` - byte hashes, split rows, checkpoint, feature
  spec, code hashes, structural coverage, range definition.
* ``analysis/motif_catalog.csv`` - one row per restricted-amide instance of a
  single-component molecule that owns at least one donor.
* ``analysis/donor_catalog.csv`` - one row per ``(instance, donor)`` pair with the
  T/K deleted and added bond sets, the structural and chemical verdicts, the
  Train-range verdict and the population flags.
* ``analysis/view_manifest.csv`` - one row per unique rewired view (key = row id +
  removed bond set + added bond set), with the scopes to predict.
* ``analysis/view_out_of_range.csv`` - every out-of-range view dimension of the
  supplementary population (dimension, value, Train min/max, excess).
* ``analysis/prepare_checks.json`` - pre-registered counts, structural gates,
  RDKit chemistry verdicts, AF51 identity gate, zero-rewire re-encoding identity
  against the frozen caches, and the raw Train connected-component census.
* ``analysis/freeze.json`` - sha256 of every frozen input consumed by inference.

A mismatch with a pre-registered count is treated as an implementation/index bug
and stops the round; a reduction caused by an *extra* legality check is listed
per pair with its reason and never silently absorbed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import chemcheck as CHEM
import graphlib as G
import feats as F
import rewire as R

# ---------------------------------------------------------------------------
# pre-registered structural counts (new protocol, section 4 of the brief)
# ---------------------------------------------------------------------------

PRECOUNTS = {
    "train": {"molecules": 870, "instances": 1021, "pairs": 2506},
    "valid": {"molecules": 121, "instances": 150, "pairs": 332},
    "test": {"molecules": 105, "instances": 120, "pairs": 271},
}

#: Previous round's strict MATCHED instances that own a donor: supplementary only.
PRECOUNTS_MATCHED_SUBSET = {"valid": {"molecules": 35, "instances": 36}, "test": {"molecules": 18, "instances": 18}}

SCOPE_O = "FULL"
SCOPE_CUT = "FULL,G_FIXED"


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_value(row.get(k, "")) for k in columns})
    return p


def _csv_value(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, float) and not np.isfinite(v):
        return ""
    return v


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Train input range (raw global42), from the frozen Train structure cache
# ---------------------------------------------------------------------------


def train_global_range() -> dict[str, Any]:
    import torch

    path = C.DD.cache_path(C.DATA_REPEAT, "train")
    if not path.exists():
        raise FileNotFoundError(f"MISSING_BASELINE: frozen Train cache not found at {C.track_rel(path)}")
    train = torch.load(path, map_location="cpu", weights_only=False)
    g = np.asarray(train["global"], dtype=np.float64)
    spec = C.DD.frozen_spec()
    names = [f.name for f in spec.global_feature_fields]
    return {
        "path": C.track_rel(path),
        "sha256": C.sha256_file(path),
        "n_molecules": int(train["n_molecules"]),
        "content_fingerprint": train["meta"].get("content_fingerprint"),
        "min": g.min(axis=0),
        "max": g.max(axis=0),
        "names": names,
        "raw_global": g,
        "rows": np.asarray(train["rows"], dtype=np.int64),
        "n_components_column": np.asarray(train["n_components"], dtype=np.int64),
    }


def range_verdict(
    global_vec: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float
) -> tuple[bool, list[dict[str, Any]]]:
    v = np.asarray(global_vec, dtype=np.float64).reshape(-1)
    below = v < (lo - tol)
    above = v > (hi + tol)
    bad = []
    for i in np.flatnonzero(below | above):
        excess = float((lo[i] - v[i]) if below[i] else (v[i] - hi[i]))
        bad.append(
            {
                "dim": int(i),
                "value": float(v[i]),
                "train_min": float(lo[i]),
                "train_max": float(hi[i]),
                "excess": excess,
                "side": "below" if below[i] else "above",
            }
        )
    return (len(bad) == 0), bad


# ---------------------------------------------------------------------------
# catalogs
# ---------------------------------------------------------------------------


def build_catalogs(
    *,
    rows: Sequence[Mapping[str, Any]],
    splits: Mapping[int, str],
    prev: Mapping[str, Any],
    spec,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Enumerate instances/donors, build T/K rows, verify, and cache every view."""
    counts: dict[str, dict[str, int]] = {
        s: {"molecules": 0, "instances": 0, "pairs": 0} for s in C.SPLITS
    }
    matched_subset: dict[str, dict[str, int]] = {
        s: {"molecules": 0, "instances": 0} for s in C.SPLITS
    }
    instance_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    views: dict[str, dict[str, Any]] = {}
    chem_rejects: list[dict[str, Any]] = []
    struct_failures: list[dict[str, Any]] = []
    mapping_methods: dict[str, int] = {}
    out_of_range_rows: list[dict[str, Any]] = []

    for rid in range(C.N_MOLECULES):
        split = splits[rid]
        g = G.summarise_raw(rows[rid], rid)
        if g.n_components != 1:
            continue
        insts = G.find_instances(g)
        if not insts:
            continue
        base_row = F.row_from_artifact({**dict(rows[rid]), "mol_index": rid})
        base_row["n_components"] = g.n_components
        mol_mapping = None
        if split in ("valid", "test"):
            mol_mapping = CHEM.smiles_mapping(str(rows[rid]["smiles"]), rows[rid], rid)
            mapping_methods[mol_mapping["method"]] = mapping_methods.get(mol_mapping["method"], 0) + 1
        mol_struct_ok = False
        mol_matched_ok = False
        for inst in insts:
            donors = R.donor_bonds(g, inst)
            if not donors:
                continue
            inst_id = inst.instance_id
            inst_row = dict(inst.as_row())
            inst_row["split"] = split
            inst_row["n_donors"] = len(donors)
            inst_row["scaffold_key"] = prev["scaffold"].get(rid, "")
            n_pairs = 0
            n_struct_ok = 0
            n_chem_ok = 0
            n_range_ok = 0
            for donor in donors:
                pair = R.build_pair(g, inst, donor)
                pair_id = f"{inst_id}:d{pair.donor.pair_key}"
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "row_id": rid,
                    "split": split,
                    "instance_id": inst_id,
                    "c": inst.c,
                    "o": inst.o,
                    "n": inst.n,
                    "acyl_c": inst.acyl_c,
                    "matched": int(inst.matched),
                    "scaffold_key": prev["scaffold"].get(rid, ""),
                    "t_remove": R._bonds_code(pair.t_remove),
                    "t_add": R._bonds_code(pair.t_add),
                    "k_remove": R._bonds_code(pair.k_remove),
                    "k_add": R._bonds_code(pair.k_add),
                    "t_view_key": pair.t_key,
                    "k_view_key": pair.k_key,
                    "legal_T_crossings": pair.n_legal_T,
                    "legal_K_crossings": pair.n_legal_K,
                    "structural_ok": 0,
                    "structural_reasons": ";".join(pair.rejections),
                    "chem_ok": 0,
                    "chem_reason": "",
                    "range_supported": 0,
                    "out_of_range_views": "",
                    "out_of_range_dims": 0,
                }
                row.update(donor.as_row())
                n_pairs += 1
                if not pair.ok:
                    pair_rows.append(row)
                    continue
                t_row, _ = R.edit_row(base_row, pair.t_remove, pair.t_add)
                k_row, _ = R.edit_row(base_row, pair.k_remove, pair.k_add)
                vt = R.verify_view_structure(
                    g, t_row, remove=pair.t_remove, add=pair.t_add, kind="T", inst=inst
                )
                vk = R.verify_view_structure(
                    g, k_row, remove=pair.k_remove, add=pair.k_add, kind="K", inst=inst
                )
                row["T_instance_count_delta"] = int(vt["instance_count_after"] - vt["instance_count_before"])
                row["K_instance_count_delta"] = int(vk["instance_count_after"] - vk["instance_count_before"])
                row["T_af51_checked_later"] = 1
                if vt["pass"] and vk["pass"]:
                    row["structural_ok"] = 1
                    n_struct_ok += 1
                else:
                    struct_failures.append(
                        {
                            "pair_id": pair_id,
                            "row_id": rid,
                            "T_pass": bool(vt["pass"]),
                            "K_pass": bool(vk["pass"]),
                            "T_failed": [k for k, v in vt.items() if v is False],
                            "K_failed": [k for k, v in vk.items() if v is False],
                        }
                    )
                    pair_rows.append(row)
                    continue
                if split in ("valid", "test"):
                    chem_t = CHEM.check_rewired_rdkit(
                        str(rows[rid]["smiles"]), rid, rows[rid], t_row, mol_mapping["raw_to_rdkit"],
                        remove=pair.t_remove, add=pair.t_add,
                    )
                    chem_k = CHEM.check_rewired_rdkit(
                        str(rows[rid]["smiles"]), rid, rows[rid], k_row, mol_mapping["raw_to_rdkit"],
                        remove=pair.k_remove, add=pair.k_add,
                    )
                    if chem_t["pass"] and chem_k["pass"]:
                        row["chem_ok"] = 1
                        n_chem_ok += 1
                    else:
                        row["chem_reason"] = ";".join(
                            r for r in (chem_t.get("reason", ""), chem_k.get("reason", "")) if r
                        )
                        chem_rejects.append(
                            {"pair_id": pair_id, "row_id": rid, "T": chem_t, "K": chem_k}
                        )
                        pair_rows.append(row)
                        continue
                # keep the views (deduplicated by key); only Valid/Test are predicted
                if split not in ("valid", "test"):
                    row["range_supported"] = -1
                    pair_rows.append(row)
                    continue
                for kind, key, vrow, remove, add in (
                    ("T", pair.t_key, t_row, pair.t_remove, pair.t_add),
                    ("K", pair.k_key, k_row, pair.k_remove, pair.k_add),
                ):
                    views.setdefault(
                        key,
                        {
                            "view_key": key,
                            "row_id": rid,
                            "split": split,
                            "kind": kind,
                            "remove": R._bonds_code(remove),
                            "add": R._bonds_code(add),
                            "scopes": SCOPE_CUT,
                            "row": vrow,
                            "pairs": [],
                        },
                    )
                    views[key]["pairs"].append(pair_id)
                row["range_supported"] = -1  # decided below from the built cache
                pair_rows.append(row)
            inst_row["n_structural_ok_pairs"] = n_struct_ok
            inst_row["n_chem_ok_pairs"] = n_chem_ok
            inst_row["n_range_supported_pairs"] = n_range_ok
            instance_rows.append(inst_row)
            if n_struct_ok:
                counts[split]["instances"] += 1
                mol_struct_ok = True
                if inst.matched:
                    mol_matched_ok = True
                    matched_subset[split]["instances"] += 1
                    inst_row["in_matched_subset"] = 1
            counts[split]["pairs"] += n_struct_ok
        if mol_struct_ok:
            counts[split]["molecules"] += 1
        if mol_matched_ok:
            matched_subset[split]["molecules"] += 1
    return instance_rows, pair_rows, views, {
        "counts": counts,
        "matched_subset": matched_subset,
        "chem_rejects": chem_rejects,
        "struct_failures": struct_failures,
        "mapping_methods": mapping_methods,
        "out_of_range_rows": out_of_range_rows,
    }


# ---------------------------------------------------------------------------
# zero-rewire identity against the frozen caches
# ---------------------------------------------------------------------------


def _global_delta_stats(
    glob: np.ndarray,
    keys: Sequence[Mapping[str, Any]],
    cache_index: Mapping[str, int],
    o_index: Mapping[int, int],
    names: Sequence[str],
    train_min: np.ndarray,
    train_max: np.ndarray,
) -> dict[str, Any]:
    """Per-dimension global42 movement of the rewired views against their own O view."""
    deltas: dict[str, list[np.ndarray]] = {"T": [], "K": []}
    for v in keys:
        if v["kind"] == "O":
            continue
        i = cache_index[v["view_key"]]
        j = o_index[v["row_id"]]
        deltas.setdefault(v["kind"], []).append(glob[i] - glob[j])
    out: dict[str, Any] = {"unit": "raw global42", "dims": {}}
    for i, name in enumerate(names):
        block: dict[str, Any] = {
            "train_min": float(train_min[i]),
            "train_max": float(train_max[i]),
        }
        for kind in ("T", "K"):
            arr = deltas.get(kind) or []
            if not arr:
                continue
            d = np.asarray(arr)[:, i]
            block[f"{kind}_mean_abs_change"] = float(np.abs(d).mean())
            block[f"{kind}_max_abs_change"] = float(np.abs(d).max())
            block[f"{kind}_mean_signed_change"] = float(d.mean())
        out["dims"][name] = block
    return out


def check_zero_rewire_identity(
    rows: Sequence[Mapping[str, Any]], spec, *, splits: Sequence[str] = ("valid", "test")
) -> dict[str, Any]:
    """Rebuilding an unedited molecule must equal the frozen structure cache."""
    src = C.load_source("none")
    out: dict[str, Any] = {"splits": {}, "pass": True}
    for split in splits:
        cache = C.DD.load_or_build_split_cache(
            C.DATA_REPEAT, split, src["rows"][split], src["smiles"], spec, rebuild=False
        )
        bad: list[int] = []
        for i, rid in enumerate(np.asarray(cache["rows"]).tolist()):
            row = F.row_from_artifact({**dict(rows[int(rid)]), "mol_index": int(rid)})
            row["n_components"] = G.n_components(
                int(row["atom_class"].shape[0]), *G.physical_bonds(row)[:2]
            )
            graph = F.view_graph(row, spec)
            mol = C.DD.PREP_DATA.build_molecule(graph, spec)
            rebuilt = {
                "n_atoms": int(mol["n_atoms"]),
                "n_bonds": int(mol["n_bonds"]),
                "atom_feat": mol["atom_feat"],
                "atom_type": mol["atom_type"],
                "atom_scalar": mol["atom_scalar"],
                "bond_feat": mol["bond_feat"],
                "bond_type": mol["bond_type"],
                "bond_index": mol["bond_index"],
                "nbr_ptr": mol["nbr_ptr"],
                "nbr_code": mol["nbr_code"],
                "path_ptr": mol["path_ptr"],
                "path_cat": mol["path_cat"],
                "path_end63": mol["path_end63"],
                "path_end": mol["path_end"],
                "path_closure": mol["path_closure"],
                "global": mol["global"],
                "n_components": int(mol["n_components"]),
                "n_junctions": int(mol["n_junctions"]),
                "n_ring_systems": int(mol["n_ring_systems"]),
            }
            cmp = F.compare_cached_molecule(rebuilt, F.cache_slice(cache, i))
            if not cmp["all_equal"]:
                bad.append(int(rid))
        ok = not bad
        out["splits"][split] = {
            "n_checked": int(cache["n_molecules"]),
            "mismatch_rows": bad[:20],
            "n_mismatch": len(bad),
            "pass": ok,
            "cache_content_fingerprint": cache["meta"].get("content_fingerprint"),
        }
        out["pass"] = out["pass"] and ok
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    spec = C.DD.frozen_spec()
    src = C.load_source("all")
    payload = C.load_raw_rows()
    rows = payload["rows"]["Lipophilicity"]
    splits = C.split_of_row(src)
    prev = C.load_previous_inputs()

    ck = C.load_checkpoint()
    norm = C.frozen_norm(ck)
    tgt = C.target_norm(ck)

    C.log("raw Train connected-component census ...")
    trange = train_global_range()
    census = {
        "train": {int(k): int(v) for k, v in zip(*np.unique(trange["n_components_column"], return_counts=True))},
    }
    for split in ("valid", "test"):
        comps = []
        for rid in [int(x) for x in src["rows"][split]]:
            g = G.summarise_raw(rows[rid], rid)
            comps.append(g.n_components)
        uniq, cnt = np.unique(np.asarray(comps), return_counts=True)
        census[split] = {int(k): int(v) for k, v in zip(uniq, cnt)}
    census["note"] = (
        "n_components census of the raw physical graph: Train is *near*-constant "
        "(one multi-component molecule), not identically constant; Valid/Test are all 1"
    )

    C.log("building donor catalogs + structural verification + chemistry ...")
    instance_rows, pair_rows, views, extra = build_catalogs(
        rows=rows,
        splits=splits,
        prev=prev,
        spec=spec,
        lo=trange["min"],
        hi=trange["max"],
    )
    counts = extra["counts"]

    mismatches: dict[str, Any] = {}
    for split, want in PRECOUNTS.items():
        for field, value in want.items():
            got = int(counts[split][field])
            if got != value:
                mismatches[f"{split}/{field}"] = {"expected": value, "actual": got}
    for split, want in PRECOUNTS_MATCHED_SUBSET.items():
        for field, value in want.items():
            got = int(extra["matched_subset"][split][field])
            if got != value:
                mismatches[f"matched_subset/{split}/{field}"] = {"expected": value, "actual": got}
    if mismatches:
        raise SystemExit(f"STRUCTURE_COUNT_MISMATCH: {mismatches}")

    # ---- O views for every Valid/Test molecule (baseline + comparisons) ----
    for split in ("valid", "test"):
        for rid in [int(x) for x in src["rows"][split]]:
            base_row = F.row_from_artifact({**dict(rows[rid]), "mol_index": rid})
            base_row["n_components"] = G.summarise_raw(rows[rid], rid).n_components
            key = R.o_view_key(rid)
            views[key] = {
                "view_key": key,
                "row_id": rid,
                "split": split,
                "kind": "O",
                "remove": "",
                "add": "",
                "scopes": SCOPE_O,
                "row": base_row,
                "pairs": [],
            }

    # ---- build one cache per split and derive global42 verdicts -----------
    C.log(f"building structure caches for {len(views)} views ...")
    per_split_meta: dict[str, Any] = {}
    cache_index: dict[str, int] = {}
    for split in ("valid", "test"):
        keys = sorted(
            (v for v in views.values() if v["split"] == split),
            key=lambda v: (v["row_id"], 0 if v["kind"] == "O" else 1, v["kind"], v["view_key"]),
        )
        entries = [(v["view_key"], v["row"]) for v in keys]
        cache = F.build_cache(entries, spec, split=f"rewire_views_{split}")
        for i, v in enumerate(keys):
            cache_index[v["view_key"]] = i
        glob = np.asarray(cache["global"], dtype=np.float64)
        comp = np.asarray(cache["n_components"], dtype=np.int64)
        frac_unreachable = glob[:, 10]
        o_index = {}
        for i, v in enumerate(keys):
            if v["kind"] == "O":
                o_index[v["row_id"]] = i
        af51_fail: list[dict[str, Any]] = []
        n_af51_checked = 0
        max_feat_diff = 0.0
        max_scalar_diff = 0.0
        n_s1_changed = 0
        n_s2_changed = 0
        for v in keys:
            if v["kind"] == "O":
                continue
            i = cache_index[v["view_key"]]
            j = o_index[v["row_id"]]
            dif = F.af51_difference(cache, j, i)
            n_af51_checked += 1
            max_feat_diff = max(max_feat_diff, float(dif["max_abs_atom_feat_diff"]))
            max_scalar_diff = max(max_scalar_diff, float(dif["max_abs_atom_scalar_diff"]))
            n_s1_changed += int(not dif["s1_nbr_code_equal"])
            n_s2_changed += int(not dif["s2_path_code_equal"])
            if not dif["atom_feat_equal"] or not dif["atom_type_equal"]:
                af51_fail.append({"view_key": v["view_key"], **dif})
        # component / unreachable-pair equality across O, T, K
        comp_ok = bool(np.all(comp == 1))
        unreach_ok = bool(np.all(np.abs(frac_unreachable) <= 1e-12))
        # range verdicts per pair
        for pr in pair_rows:
            if pr["split"] != split or not pr["structural_ok"]:
                continue
            if pr["chem_ok"] != 1:
                continue
            keys_here = [R.o_view_key(int(pr["row_id"])), pr["t_view_key"], pr["k_view_key"]]
            bad_all: list[dict[str, Any]] = []
            for key, label in zip(keys_here, ("O", "T", "K")):
                i = cache_index[key]
                ok, bad = range_verdict(glob[i], trange["min"], trange["max"], C.RANGE_TOL)
                for b in bad:
                    b["view"] = label
                    b["view_key"] = key
                    b["pair_id"] = pr["pair_id"]
                    b["row_id"] = pr["row_id"]
                    b["split"] = split
                    bad_all.append(b)
            pr["range_supported"] = 0 if bad_all else 1
            pr["out_of_range_dims"] = len(bad_all)
            pr["out_of_range_views"] = ",".join(sorted({b["view"] for b in bad_all}))
            if bad_all:
                out_of_range_rows.extend(bad_all)
        per_split_meta[split] = {
            "n_views": len(keys),
            "n_O": len(o_index),
            "n_cut": len(keys) - len(o_index),
            "cache_content_fingerprint": cache["meta"].get("content_fingerprint"),
            "cache_sha256_like": None,
            "n_components_all_one": comp_ok,
            "frac_unreachable_all_zero": unreach_ok,
            "frac_unreachable_max": float(np.max(np.abs(frac_unreachable))),
            "af51_checked": n_af51_checked,
            "af51_failures": af51_fail[:10],
            "af51_n_failures": len(af51_fail),
            "af51_max_abs_atom_feat_diff": max_feat_diff,
            "atom_scalar_max_abs_diff": max_scalar_diff,
            "views_with_changed_s1": n_s1_changed,
            "views_with_changed_s2": n_s2_changed,
            "global_stats": _global_delta_stats(
                glob, keys, cache_index, o_index, trange["names"], trange["min"], trange["max"]
            ),
            "n_pairs_range_supported": sum(
                1 for pr in pair_rows if pr["split"] == split and pr["structural_ok"] and pr["chem_ok"] and pr["range_supported"] == 1
            ),
            "n_pairs_chem_ok": sum(1 for pr in pair_rows if pr["split"] == split and pr["chem_ok"] == 1),
        }
        if not comp_ok or not unreach_ok or af51_fail:
            raise SystemExit(f"REWIRE_INPUT_INVARIANT_FAILED: {split} {per_split_meta[split]}")

    # per-instance donor counts that depend on the range verdict
    inst_by_id: dict[str, dict[str, Any]] = {r["instance_id"]: r for r in instance_rows}
    for pr in pair_rows:
        if pr["range_supported"] == 1:
            inst_by_id[pr["instance_id"]]["n_range_supported_pairs"] = (
                int(inst_by_id[pr["instance_id"]].get("n_range_supported_pairs", 0)) + 1
            )

    # ---- populations -----------------------------------------------------
    for pr in pair_rows:
        pr["pop_struct_chem"] = int(pr["structural_ok"] == 1 and pr["chem_ok"] == 1)
        pr["pop_main"] = int(pr["pop_struct_chem"] == 1 and pr["range_supported"] == 1)
        pr["pop_matched_supp"] = int(pr["pop_struct_chem"] == 1 and int(pr["matched"]) == 1)

    # ---- population support counts --------------------------------------
    pop_support: dict[str, Any] = {}
    for split in ("valid", "test"):
        block: dict[str, Any] = {}
        for pop in ("main", "struct_chem_all", "matched_supp"):
            if pop == "main":
                sel = [p for p in pair_rows if p["split"] == split and p["pop_main"]]
            elif pop == "struct_chem_all":
                sel = [p for p in pair_rows if p["split"] == split and p["pop_struct_chem"]]
            else:
                sel = [p for p in pair_rows if p["split"] == split and p["pop_matched_supp"]]
            block[pop] = {
                "molecules": len({p["row_id"] for p in sel}),
                "instances": len({p["instance_id"] for p in sel}),
                "pairs": len(sel),
                "scaffolds": len({p["scaffold_key"] for p in sel}),
            }
        pop_support[split] = block

    # ---- selection analysis (why pairs drop out) -------------------------
    drop = {
        "pairs_total": len(pair_rows),
        "structural_ok": sum(1 for p in pair_rows if p["structural_ok"]),
        "chem_ok_given_struct": sum(1 for p in pair_rows if p["chem_ok"]),
        "range_ok_given_chem": sum(1 for p in pair_rows if p["pop_main"]),
        "struct_failures": len(extra["struct_failures"]),
        "chem_rejects": len(extra["chem_rejects"]),
        "pairs_with_out_of_range_dims": len({p["pair_id"] for p in pair_rows if p["out_of_range_dims"]}),
        "mapping_methods": extra["mapping_methods"],
    }

    # ---- zero-rewire identity -------------------------------------------
    C.log("zero-rewire re-encoding identity vs frozen caches ...")
    zero = check_zero_rewire_identity(rows, spec)
    if not zero["pass"]:
        raise SystemExit(f"ZERO_REWIRE_IDENTITY_FAILED: {zero}")

    # ---- write frozen inputs --------------------------------------------
    view_rows = []
    for v in sorted(views.values(), key=lambda x: (x["row_id"], x["kind"], x["view_key"])):
        view_rows.append(
            {
                "view_key": v["view_key"],
                "row_id": v["row_id"],
                "split": v["split"],
                "kind": v["kind"],
                "remove": v["remove"],
                "add": v["add"],
                "scopes": v["scopes"],
                "n_pairs": len(v["pairs"]),
                "pairs": ";".join(sorted(set(v["pairs"]))),
            }
        )
    view_columns = ["view_key", "row_id", "split", "kind", "remove", "add", "scopes", "n_pairs", "pairs"]
    pair_columns = [
        "pair_id", "row_id", "split", "instance_id", "scaffold_key", "matched",
        "c", "o", "n", "acyl_c",
        "donor_u", "donor_v", "donor_bond_id", "donor_degree_u", "donor_degree_v",
        "donor_atom_h_u", "donor_atom_h_v", "donor_ring_u", "donor_ring_v", "donor_distance_to_c",
        "t_remove", "t_add", "k_remove", "k_add", "t_view_key", "k_view_key",
        "legal_T_crossings", "legal_K_crossings",
        "structural_ok", "structural_reasons",
        "T_instance_count_delta", "K_instance_count_delta",
        "chem_ok", "chem_reason",
        "range_supported", "out_of_range_views", "out_of_range_dims",
        "pop_struct_chem", "pop_main", "pop_matched_supp",
    ]
    instance_columns = [
        "instance_id", "row_id", "split", "scaffold_key", "c", "o", "n", "acyl_c",
        "degree_n", "degree_acyl", "ring_n", "ring_acyl", "matched",
        "n_side", "acyl_side", "side_diff", "component_size",
        "bond_cn", "bond_c_acyl", "bond_c_o",
        "n_donors", "n_structural_ok_pairs", "n_chem_ok_pairs", "n_range_supported_pairs",
        "in_matched_subset",
    ]
    for r in instance_rows:
        r.setdefault("in_matched_subset", 0)
    written = {
        "motif_catalog.csv": write_csv(C.ANALYSIS / "motif_catalog.csv", instance_rows, instance_columns),
        "donor_catalog.csv": write_csv(C.ANALYSIS / "donor_catalog.csv", pair_rows, pair_columns),
        "view_manifest.csv": write_csv(C.ANALYSIS / "view_manifest.csv", view_rows, view_columns),
        "view_out_of_range.csv": write_csv(
            C.ANALYSIS / "view_out_of_range.csv",
            extra["out_of_range_rows"],
            ["pair_id", "row_id", "split", "view", "view_key", "dim", "value", "train_min", "train_max", "excess", "side"],
        ),
    }
    written_hashes = {name: C.sha256_file(path) for name, path in written.items()}

    manifest = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "generated_utc": C.utcnow(),
        "config": C.config_snapshot(),
        "source_files": {k: C.track_rel(v) for k, v in C.resolve_source_files().items()},
        "data": {
            "source_commit": C.SOURCE_COMMIT,
            "csv_sha256": C.LIPOP_CSV_SHA256,
            "split_file": C.SPLIT_FILE,
            "split_file_sha256": C.SPLIT_FILE_SHA256,
            "split_rows_sha256": dict(C.SPLIT_ROWS_SHA256),
            "raw_artifact_sha256": C.RAW_ARTIFACT_SHA256,
            "rows": {s: [int(x) for x in src["rows"][s]] for s in C.SPLITS},
            "row_sha256_recomputed": {s: C.row_sha256(src["rows"][s]) for s in C.SPLITS},
            "n_components_census": census,
        },
        "previous_round_inputs": {
            "files": dict(prev["files"]),
            "note": "original labels, frozen O predictions and scaffold keys reused verbatim from the executed "
            "previous round (revision a0850d89, run amide-dep-infer-v3)",
        },
        "model": {
            "checkpoint": C.CHECKPOINT_REL,
            "checkpoint_sha256": C.CHECKPOINT_SHA256,
            "state_hash": C.STATE_HASH,
            "variant": C.VARIANT,
            "parameter_count": C.EXPECTED_PARAMS,
            "target_norm": tgt,
            "norm_fingerprint": C.DD.PREP_DATA.norm_fingerprint(norm),
            "norm_source": "checkpoint buffers (frozen Train-only standardisation)",
            "reference_metrics": dict(C.REFERENCE),
        },
        "train_global_range": {
            "path": trange["path"],
            "sha256": trange["sha256"],
            "n_molecules": trange["n_molecules"],
            "content_fingerprint": trange["content_fingerprint"],
            "tol": C.RANGE_TOL,
            "unit": "raw (un-normalised) global42",
            "min": [float(x) for x in trange["min"]],
            "max": [float(x) for x in trange["max"]],
            "dim_names": list(trange["names"]),
        },
        "structural_counts": counts,
        "precounts": PRECOUNTS,
        "matched_subset_counts": extra["matched_subset"],
        "precounts_matched_subset": PRECOUNTS_MATCHED_SUBSET,
        "population_support": pop_support,
        "selection": drop,
        "views": per_split_meta,
        "code_hashes": C.code_hashes(),
        "artifacts": {
            "files": written_hashes,
            "note": "sha256 of the committed protocol inputs consumed by infer.py",
        },
    }
    C.write_json(C.ANALYSIS / "inputs_manifest.json", manifest)

    checks = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "precounts_match_protocol": True,
        "observed_counts": counts,
        "observed_matched_subset": extra["matched_subset"],
        "n_components_census": census,
        "selection": drop,
        "population_support": pop_support,
        "structural_failures": extra["struct_failures"][:50],
        "chem_rejects": [
            {"pair_id": r["pair_id"], "row_id": r["row_id"], "reason": r["T"].get("reason") or r["K"].get("reason")}
            for r in extra["chem_rejects"][:50]
        ],
        "views": {s: {k: v for k, v in blk.items() if k != "global_stats"} for s, blk in per_split_meta.items()},
        "zero_rewire_identity": zero,
        "checkpoint": {
            "sha256": C.CHECKPOINT_SHA256,
            "state_hash": C.STATE_HASH,
            "variant": C.VARIANT,
            "parameter_count": C.EXPECTED_PARAMS,
            "best_epoch": C.BEST_EPOCH,
            "pass": True,
        },
    }
    C.write_json(C.ANALYSIS / "prepare_checks.json", checks)

    freeze = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "files": {
            **written_hashes,
            "inputs_manifest.json": C.sha256_file(C.ANALYSIS / "inputs_manifest.json"),
            "prepare_checks.json": C.sha256_file(C.ANALYSIS / "prepare_checks.json"),
        },
        "note": "inference may only read these frozen bytes; no re-selection after prediction",
    }
    C.write_json(C.ANALYSIS / "freeze.json", freeze)

    C.log(f"instances={len(instance_rows)} pairs={len(pair_rows)} views={len(view_rows)}")
    for s in ("valid", "test"):
        C.log(f"{s}: population support {pop_support[s]}")
    C.log(f"selection: {drop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
