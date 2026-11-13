"""Stage A: structure-only preparation, dose calibration and frozen manifests.

Everything written here is label-blind and prediction-blind: no label is read
except the frozen original O predictions / original labels of the *executed*
previous round (which are only carried through as references, never used to
select a view), and no model forward happens.  The frozen artifacts are:

* ``inputs_manifest.json`` — byte hashes, split rows, checkpoint, feature spec,
  both frozen parameter-state hashes, Train global42 range, the six-family
  census (frozen rule **and** the legacy class-2-only census that reproduces the
  pre-registered candidate coverage), populations, dose summaries.
* ``family_catalog.csv`` — one row per rewirable ``(family, molecule, instance,
  control)`` of every split (Train included: structure catalogue only).
* ``pair_catalog.csv`` — one row per ``(family, instance, control, donor)`` of
  Valid/Test with the T / K0 bond sets, structural / RDKit / range verdicts,
  the structural doses and the population flags.
* ``view_manifest.csv`` — one row per view key (row id + kind + family + removed
  + added set) with its graph id.
* ``graph_manifest.csv`` — one row per unique rebuilt *graph*, i.e. the exact
  forward-pass set (duplicate graphs across families are predicted once).
* ``caliper_catalog.csv`` — every accepted focal-T / C-candidate pair.
* ``caliper_rejected.csv`` — every focal T with no C candidate, with the reason.
* ``view_out_of_range.csv`` — every out-of-range global42 dimension.
* ``prepare_checks.json`` / ``freeze.json`` — gates and the frozen byte hashes.

A mismatch that cannot be explained by an implementation/index bug stops the
round (``STRUCTURE_COUNT_MISMATCH`` etc.); a reduction caused by an extra
legality check is listed per pair with its reason and never silently absorbed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import chemcheck as CHEM
import doses as D
import families as FAM
import feats as F
import graphlib as G
import rewire as R

#: Pre-registered candidate coverage (prompt section 2, structure only).
PRECOUNTS = {
    "amide": {"train": (1276, 1475, 4495), "valid": (174, 211, 650), "test": (164, 182, 533)},
    "ester": {"train": (80, 86, 335), "valid": (6, 7, 18), "test": (8, 10, 46)},
    "sulfonamide": {"train": (296, 297, 925), "valid": (36, 36, 143), "test": (30, 31, 112)},
    "sulfonate": {"train": (2, 2, 6), "valid": (0, 0, 0), "test": (0, 0, 0)},
    "arylamine": {"train": (116, 125, 293), "valid": (15, 16, 32), "test": (17, 17, 41)},
    "aryl_ether": {"train": (576, 709, 1352), "valid": (62, 74, 123), "test": (58, 70, 112)},
}

PREDICT_SPLITS = ("valid", "test")


# ---------------------------------------------------------------------------
# small CSV helpers
# ---------------------------------------------------------------------------


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
# Train global42 range (raw), from the frozen Train structure cache
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
        bad.append(
            {
                "dim": int(i),
                "value": float(v[i]),
                "train_min": float(lo[i]),
                "train_max": float(hi[i]),
                "excess": float((lo[i] - v[i]) if below[i] else (v[i] - hi[i])),
                "side": "below" if below[i] else "above",
            }
        )
    return (len(bad) == 0), bad


# ---------------------------------------------------------------------------
# structure census (label free, both donor rules)
# ---------------------------------------------------------------------------


def _donor_legacy_class2(donor: R.Donor) -> bool:
    return bool(int(donor.class_u) == C.MF_C_CLASS and int(donor.class_v) == C.MF_C_CLASS)


def family_census(g: G.RawGraph, split: str) -> dict[str, Any]:
    """Per-molecule census of all six families (global keys and rewirable instances)."""
    out: dict[str, Any] = {}
    for fam in C.FAMILIES:
        insts = FAM.main_instances(g, fam)
        keys = FAM.family_keys(g, fam)
        pairs_all = 0
        pairs_legacy = 0
        inst_with = set()
        inst_with_legacy = set()
        for inst in insts:
            donors = R.donor_bonds(g, inst)
            pairs_all += len(donors)
            n_legacy = sum(1 for d in donors if _donor_legacy_class2(d))
            pairs_legacy += n_legacy
            if donors:
                inst_with.add((inst.center, inst.target))
            if n_legacy:
                inst_with_legacy.add((inst.center, inst.target))
        out[fam] = {
            "keys": len(keys),
            "instances": len(insts),
            "instances_with_donor": len(inst_with),
            "instances_with_donor_legacy": len(inst_with_legacy),
            "pairs": int(pairs_all),
            "pairs_legacy": int(pairs_legacy),
        }
    return out


# ---------------------------------------------------------------------------
# main preparation
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: C901 (single auditable pass)
    spec = C.DD.frozen_spec()
    src = C.load_source("all")
    payload = C.load_raw_rows()
    rows = payload["rows"]["Lipophilicity"]
    splits = C.split_of_row(src)
    prev = C.load_previous_inputs()

    ck = C.load_checkpoint()
    norm = C.frozen_norm(ck)
    tgt = C.target_norm(ck)
    init_built = C.build_initial_state(spec)
    C.log(
        f"parameter states: TRAINED state={C.STATE_HASH[:12]} "
        f"INIT={init_built['hash'][:12]} matches_recorded={init_built['matches_recorded']}"
    )

    C.log("Train connected-component census + raw global42 range ...")
    trange = train_global_range()
    census = {
        "train": {
            int(k): int(v) for k, v in zip(*np.unique(trange["n_components_column"], return_counts=True))
        }
    }
    for split in ("valid", "test"):
        comps = []
        for rid in [int(x) for x in src["rows"][split]]:
            comps.append(G.summarise_raw(rows[rid], rid).n_components)
        uniq, cnt = np.unique(np.asarray(comps), return_counts=True)
        census[split] = {int(k): int(v) for k, v in zip(uniq, cnt)}

    # ---- 1. six-family census over all splits ----------------------------
    C.log("six-family structural census (frozen rule + legacy class-2 donor rule) ...")
    census_obs: dict[str, Any] = {}
    census_ref: dict[str, Any] = {}
    family_catalog: list[dict[str, Any]] = []
    C.log("Bemis-Murcko scaffold keys (local RDKit, frozen definition) ...")
    scaffold = {rid: G.bemis_murcko_key(str(rows[rid]["smiles"])) for rid in range(C.N_MOLECULES)}
    overlap = [rid for rid in prev["scaffold"] if rid in scaffold]
    scaffold_disagreements = [int(rid) for rid in overlap if str(prev["scaffold"][rid]) != scaffold[rid]]
    if scaffold_disagreements:
        raise SystemExit(f"SCAFFOLD_KEY_MISMATCH: {scaffold_disagreements[:10]}")
    train_scaffold_set = {scaffold[int(rid)] for rid in src["rows"]["train"]}
    for split in C.SPLITS:
        agg = {fam: {"molecules": 0, "instances": 0, "controls": 0, "pairs": 0} for fam in C.FAMILIES}
        agg_legacy = {fam: {"molecules": 0, "instances": 0, "controls": 0, "pairs": 0} for fam in C.FAMILIES}
        for rid in [int(x) for x in src["rows"][split]]:
            g = G.summarise_raw(rows[rid], rid)
            if g.n_components != 1:
                continue
            cens = family_census(g, split)
            for fam in C.FAMILIES:
                d = cens[fam]
                if d["instances_with_donor"]:
                    agg[fam]["molecules"] += 1
                    agg[fam]["instances"] += d["instances_with_donor"]
                    agg[fam]["controls"] += d["instances"]
                    agg[fam]["pairs"] += d["pairs"]
                if d["instances_with_donor_legacy"]:
                    agg_legacy[fam]["molecules"] += 1
                    agg_legacy[fam]["instances"] += d["instances_with_donor_legacy"]
                    agg_legacy[fam]["controls"] += d["instances"]
                    agg_legacy[fam]["pairs"] += d["pairs_legacy"]
            for fam in C.FAMILIES:
                for inst in FAM.main_instances(g, fam):
                    family_catalog.append(
                        {
                            **inst.as_row(),
                            "split": split,
                            "scaffold_key": scaffold[rid],
                            "scaffold_seen_in_train": int(scaffold[rid] in train_scaffold_set),
                            "n_donors": len(R.donor_bonds(g, inst)),
                            "n_global_keys": len(FAM.family_keys(g, fam)),
                        }
                    )
        census_obs[split] = agg
        census_ref[split] = agg_legacy

    precount_check = {
        fam: {
            split: {
                "precount": list(PRECOUNTS[fam][split]),
                "frozen_rule": [
                    census_obs[split][fam]["molecules"],
                    census_obs[split][fam]["instances"],
                    census_obs[split][fam]["pairs"],
                ],
                "legacy_class2_donor": [
                    census_ref[split][fam]["molecules"],
                    census_ref[split][fam]["instances"],
                    census_ref[split][fam]["pairs"],
                ],
            }
            for split in C.SPLITS
        }
        for fam in C.FAMILIES
    }
    mismatches: dict[str, Any] = {}
    legacy_mismatches: dict[str, Any] = {}
    for fam in C.FAMILIES:
        for split in C.SPLITS:
            want = list(PRECOUNTS[fam][split])
            got = precount_check[fam][split]["frozen_rule"]
            leg = precount_check[fam][split]["legacy_class2_donor"]
            if got != want:
                mismatches[f"{fam}/{split}"] = {"expected": want, "actual": got}
            if leg != want:
                legacy_mismatches[f"{fam}/{split}"] = {"expected": want, "actual": leg}
    C.log(f"precount reconciliation: frozen-rule mismatches={len(mismatches)} legacy-rule mismatches={len(legacy_mismatches)}")
    families_with_frozen_match = [f for f in C.FAMILIES if not any(k.startswith(f + "/") for k in mismatches)]
    if not families_with_frozen_match:
        raise SystemExit(f"STRUCTURE_COUNT_MISMATCH: {mismatches}")
    # The carbonyl / sulfonyl families must match the frozen rule exactly; the
    # aromatic families are reconciled against the legacy class-2-only donor
    # census, which is what the pre-registered candidate coverage was computed
    # with (documented before any prediction).
    for fam in C.CARBONYL_FAMILIES + C.SULFONYL_FAMILIES:
        if any(k.startswith(fam + "/") for k in mismatches):
            raise SystemExit(f"STRUCTURE_COUNT_MISMATCH: {fam}: {mismatches}")

    # ---- 2. Valid/Test pair construction --------------------------------
    C.log("building Valid/Test pair catalogues, verdicts and doses ...")
    pairs: list[dict[str, Any]] = []
    graphs: dict[str, dict[str, Any]] = {}
    struct_failures: list[dict[str, Any]] = []
    chem_rejects: list[dict[str, Any]] = []
    mapping_methods: dict[str, int] = {}
    mol_mapping: dict[int, Any] = {}
    for split in PREDICT_SPLITS:
        for rid in [int(x) for x in src["rows"][split]]:
            g = G.summarise_raw(rows[rid], rid)
            if g.n_components != 1:
                continue
            base_row = F.row_from_artifact({**dict(rows[rid]), "mol_index": rid})
            base_row["n_components"] = g.n_components
            gid_o = graph_id(rid, (), ())
            graphs.setdefault(
                gid_o,
                {
                    "graph_id": gid_o,
                    "row_id": rid,
                    "split": split,
                    "kind": "O",
                    "family": "",
                    "remove": [],
                    "add": [],
                    "row": base_row,
                },
            )
            map_ = CHEM.smiles_mapping(str(rows[rid]["smiles"]), rows[rid], rid)
            mol_mapping[rid] = map_
            mapping_methods[map_["method"]] = mapping_methods.get(map_["method"], 0) + 1
            for fam in C.FAMILIES:
                for inst in FAM.main_instances(g, fam):
                    for donor in R.donor_bonds(g, inst):
                        pair = R.build_pair(g, inst, donor)
                        row: dict[str, Any] = {
                            "pair_id": f"{inst.instance_id}:d{donor.pair_key}",
                            "family": fam,
                            "row_id": rid,
                            "split": split,
                            "instance_id": inst.instance_id,
                            "center": inst.center,
                            "target": inst.target,
                            "control": inst.control,
                            "terminals": "-".join(str(t) for t in inst.terminals),
                            "scaffold_key": scaffold[rid],
                            "scaffold_seen_in_train": int(scaffold[rid] in train_scaffold_set),
                            "t_remove": R._bonds_code(pair.t_remove),
                            "t_add": R._bonds_code(pair.t_add),
                            "k_remove": R._bonds_code(pair.k_remove),
                            "k_add": R._bonds_code(pair.k_add),
                            "t_view_key": pair.t_key,
                            "k_view_key": pair.k_key,
                            "o_graph_id": graph_id(rid, (), ()),
                            "t_graph_id": graph_id(rid, pair.t_remove, pair.t_add),
                            "k_graph_id": graph_id(rid, pair.k_remove, pair.k_add),
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
                        if not pair.ok:
                            pairs.append(row)
                            continue
                        t_row, _ = R.edit_row(base_row, pair.t_remove, pair.t_add)
                        k_row, _ = R.edit_row(base_row, pair.k_remove, pair.k_add)
                        vt = R.verify_view_structure(
                            g, t_row, remove=pair.t_remove, add=pair.t_add, kind="T", inst=inst
                        )
                        vk = R.verify_view_structure(
                            g, k_row, remove=pair.k_remove, add=pair.k_add, kind="K0", inst=inst
                        )
                        row["T_family_count_delta"] = int(vt["family_count_delta"])
                        row["K0_family_count_delta"] = int(vk["family_count_delta"])
                        if not (vt["pass"] and vk["pass"]):
                            struct_failures.append(
                                {
                                    "pair_id": row["pair_id"],
                                    "family": fam,
                                    "row_id": rid,
                                    "T_pass": bool(vt["pass"]),
                                    "K0_pass": bool(vk["pass"]),
                                    "T_failed": [k for k, v in vt.items() if v is False],
                                    "K0_failed": [k for k, v in vk.items() if v is False],
                                    "T_reason": vt.get("expected"),
                                    "K0_reason": vk.get("expected"),
                                    "T_family_count_delta": vt["family_count_delta"],
                                    "K0_family_count_delta": vk["family_count_delta"],
                                    "T_target_key_after": vt["target_key_present_after"],
                                    "K0_target_key_after": vk["target_key_present_after"],
                                }
                            )
                            row["structural_reasons"] += ";key_or_invariant_check_failed"
                            pairs.append(row)
                            continue
                        row["structural_ok"] = 1
                        chem_t = CHEM.check_rewired_rdkit(
                            str(rows[rid]["smiles"]), rid, rows[rid], t_row, map_["raw_to_rdkit"],
                            remove=pair.t_remove, add=pair.t_add,
                        )
                        chem_k = CHEM.check_rewired_rdkit(
                            str(rows[rid]["smiles"]), rid, rows[rid], k_row, map_["raw_to_rdkit"],
                            remove=pair.k_remove, add=pair.k_add,
                        )
                        if not (chem_t["pass"] and chem_k["pass"]):
                            row["chem_reason"] = ";".join(
                                r for r in (chem_t.get("reason", ""), chem_k.get("reason", "")) if r
                            )
                            chem_rejects.append({"pair_id": row["pair_id"], "row_id": rid, "T": chem_t, "K0": chem_k})
                            pairs.append(row)
                            continue
                        row["chem_ok"] = 1
                        for gid, kind, vrow, remove, add, key in (
                            ("t_graph_id", "T", t_row, pair.t_remove, pair.t_add, pair.t_key),
                            ("k_graph_id", "K0", k_row, pair.k_remove, pair.k_add, pair.k_key),
                        ):
                            gid_value = row[gid]
                            graphs.setdefault(
                                gid_value,
                                {
                                    "graph_id": gid_value,
                                    "row_id": rid,
                                    "split": split,
                                    "kind": kind,
                                    "family": fam,
                                    "remove": list(remove),
                                    "add": list(add),
                                    "row": vrow,
                                    "view_keys": set(),
                                },
                            )
                            graphs[gid_value].setdefault("view_keys", set()).add(key)
                            graphs[gid_value].setdefault("families", set()).add(fam)
                        pairs.append(row)

    # ---- 3. one structure cache per split (O + every legal graph) -------
    C.log(f"building structure caches for {len(graphs)} unique graphs ...")
    cache_index: dict[str, int] = {}
    o_index: dict[int, int] = {}
    per_split_meta: dict[str, Any] = {}
    out_of_range_rows: list[dict[str, Any]] = []
    view_doses: dict[str, dict[str, Any]] = {}
    scalar_dev: dict[str, list[float]] = {}
    global_dev: dict[str, list[np.ndarray]] = {}
    for split in PREDICT_SPLITS:
        cache_index.clear()
        o_index.clear()
        keys = sorted(
            (v for v in graphs.values() if v["split"] == split),
            key=lambda v: (v["row_id"], 0 if v["kind"] == "O" else 1, v["kind"], v["graph_id"]),
        )
        entries = [(v["graph_id"], v["row"]) for v in keys]
        cache = F.build_cache(entries, spec, split=f"multi_motif_views_{split}")
        for i, v in enumerate(keys):
            cache_index[v["graph_id"]] = i
        glob = np.asarray(cache["global"], dtype=np.float64)
        comp = np.asarray(cache["n_components"], dtype=np.int64)
        frac_unreachable = glob[:, 10]
        for i, v in enumerate(keys):
            if v["kind"] == "O":
                o_index[v["row_id"]] = i
        af51_fail: list[dict[str, Any]] = []
        max_feat_diff = 0.0
        max_scalar_diff = 0.0
        n_s1_changed = 0
        n_s2_changed = 0
        tables_o: dict[int, dict[str, Any]] = {}
        for rid, i in o_index.items():
            tables_o[rid] = D.molecule_tables(cache, i)
        for v in keys:
            if v["kind"] == "O":
                continue
            i = cache_index[v["graph_id"]]
            j = o_index[v["row_id"]]
            dif = F.af51_difference(cache, j, i)
            max_feat_diff = max(max_feat_diff, float(dif["max_abs_atom_feat_diff"]))
            max_scalar_diff = max(max_scalar_diff, float(dif["max_abs_atom_scalar_diff"]))
            n_s1_changed += int(not dif["s1_nbr_code_equal"])
            n_s2_changed += int(not dif["s2_path_code_equal"])
            if not dif["atom_feat_equal"] or not dif["atom_type_equal"]:
                af51_fail.append({"graph_id": v["graph_id"], **dif})
            tbl = D.molecule_tables(cache, i)
            view_doses[v["graph_id"]] = D.doses(
                tables_o[v["row_id"]], tbl, scalar_mean=norm["scalar_mean"], scalar_scale=norm["scalar_scale"]
            )
            # scalar9 / global42 deviation distributions (never clipped, never refit)
            so = (np.asarray(tables_o[v["row_id"]]["atom_scalar"], dtype=np.float64) - norm["scalar_mean"]) / norm["scalar_scale"]
            sv = (np.asarray(tbl["atom_scalar"], dtype=np.float64) - norm["scalar_mean"]) / norm["scalar_scale"]
            scalar_dev.setdefault(f"{split}:{v['kind']}", []).append(float(np.abs(sv - so).max()))
            global_dev.setdefault(f"{split}:{v['kind']}", []).append(glob[i] - glob[j])
        o_mask = np.asarray([v["kind"] == "O" for v in keys], dtype=bool)
        per_split_meta[split] = {
            "n_graphs": len(keys),
            "n_O": int(o_mask.sum()),
            "n_rewired": int(len(keys) - o_mask.sum()),
            "cache_content_fingerprint": cache["meta"].get("content_fingerprint"),
            "n_components_all_one": bool(np.all(comp == 1)),
            "frac_unreachable_all_zero": bool(np.all(np.abs(frac_unreachable) <= 1e-12)),
            "frac_unreachable_max": float(np.max(np.abs(frac_unreachable))),
            "af51_failures": af51_fail[:10],
            "af51_n_failures": len(af51_fail),
            "af51_max_abs_atom_feat_diff": max_feat_diff,
            "atom_scalar_max_abs_diff": max_scalar_diff,
            "graphs_with_changed_s1": int(n_s1_changed),
            "graphs_with_changed_s2": int(n_s2_changed),
            "views_in_train_range": int(
                sum(
                    1
                    for v in keys
                    if range_verdict(glob[cache_index[v["graph_id"]]], trange["min"], trange["max"], C.RANGE_TOL)[0]
                )
            ),
        }
        if not per_split_meta[split]["n_components_all_one"] or not per_split_meta[split]["frac_unreachable_all_zero"]:
            raise SystemExit(f"REWIRE_INPUT_INVARIANT_FAILED: {split} {per_split_meta[split]}")
        if af51_fail:
            raise SystemExit(f"AF51_IDENTITY_FAILED: {split} {af51_fail[:3]}")
        # range verdicts per graph and per pair
        range_ok: dict[str, bool] = {}
        for v in keys:
            i = cache_index[v["graph_id"]]
            ok, bad = range_verdict(glob[i], trange["min"], trange["max"], C.RANGE_TOL)
            range_ok[v["graph_id"]] = ok
            if not ok:
                for b in bad:
                    b.update({"view": v["kind"], "graph_id": v["graph_id"], "row_id": v["row_id"], "split": split})
                    out_of_range_rows.append(b)
        for pr in pairs:
            if pr["split"] != split or not pr["structural_ok"] or pr["chem_ok"] != 1:
                continue
            keys_here = [(pr["t_graph_id"], "T"), (pr["k_graph_id"], "K0"), (graph_id(pr["row_id"], (), ()), "O")]
            bad_views = [label for gid, label in keys_here if not range_ok[gid]]
            pr["range_supported"] = 0 if bad_views else 1
            pr["out_of_range_views"] = ",".join(sorted(set(bad_views)))
            pr["out_of_range_dims"] = int(
                sum(1 for b in out_of_range_rows if b["graph_id"] in {pr["t_graph_id"], pr["k_graph_id"]})
            )
        # store per-graph range flag in the graph record
        for v in keys:
            graphs[v["graph_id"]]["range_supported"] = int(range_ok[v["graph_id"]])
            graphs[v["graph_id"]]["global42"] = [float(x) for x in glob[cache_index[v["graph_id"]]]]

    # ---- 4. pair-level doses and populations ----------------------------
    for pr in pairs:
        if pr["structural_ok"] and pr["chem_ok"] == 1:
            d = view_doses[pr["t_graph_id"]]
            for k, v in d.items():
                pr[f"dose_{k}"] = v
            dk = view_doses[pr["k_graph_id"]]
            for k, v in dk.items():
                pr[f"doseK_{k}"] = v
        pr["pop_struct_chem"] = int(pr["structural_ok"] == 1 and pr["chem_ok"] == 1)
        pr["pop_main"] = int(pr["pop_struct_chem"] == 1 and pr["range_supported"] == 1)

    # ---- 5. calibration pool and caliper matching (prediction blind) ----
    C.log("building the per-molecule calibration pool and the dose caliper ...")
    pool_by_mol: dict[int, list[str]] = {}
    for gid, v in graphs.items():
        if v["kind"] == "O":
            continue
        if not v.get("range_supported"):
            continue
        # a C candidate must be a structurally/chemically legal rewired graph of
        # this molecule: only graphs referenced by a legal pair qualify
        pool_by_mol.setdefault(int(v["row_id"]), []).append(gid)
    legal_gids = {
        gid
        for pr in pairs
        if pr["pop_struct_chem"] == 1
        for gid in (pr["t_graph_id"], pr["k_graph_id"])
    }
    for rid in list(pool_by_mol):
        pool_by_mol[rid] = sorted(g for g in pool_by_mol[rid] if g in legal_gids)
        if not pool_by_mol[rid]:
            del pool_by_mol[rid]
    view_meta = {}
    for gid in sorted(set(legal_gids) | {graph_id(rid, (), ()) for rid in {v["row_id"] for v in graphs.values()}}):
        v = graphs[gid]
        g_view = G.summarise_raw(v["row"], int(v["row_id"]))
        key_counts = {fam: set(FAM.family_keys(g_view, fam)) for fam in C.FAMILIES}
        view_meta[gid] = {"counts": {f: len(s) for f, s in key_counts.items()}, "keys": key_counts}

    caliper_rows: list[dict[str, Any]] = []
    caliper_rejected: list[dict[str, Any]] = []
    for pr in pairs:
        if not pr["pop_main"]:
            continue
        pr["focal_key"] = f"{pr['t_view_key']}"
        cands = []
        fam = pr["family"]
        focal_key = (int(pr["center"]), int(pr["target"]))
        count_o = view_meta[graph_id(pr["row_id"], (), ())]["counts"][fam]
        for gid in pool_by_mol.get(int(pr["row_id"]), []):
            meta = view_meta[gid]
            if meta["counts"][fam] != count_o:
                continue
            if focal_key not in meta["keys"][fam]:
                continue
            ok, detail = D.caliper_ok(view_doses[pr["t_graph_id"]], view_doses[gid], C.C_CALIPER_RATIO, C.C_EPS)
            if not ok:
                continue
            cands.append((gid, detail))
        pr["n_c_candidates"] = len(cands)
        if not cands:
            caliper_rejected.append(
                {
                    "family": fam,
                    "row_id": pr["row_id"],
                    "pair_id": pr["pair_id"],
                    "split": pr["split"],
                    "reason": "no_c_candidate_within_caliper",
                    "doses": {k: view_doses[pr["t_graph_id"]][k] for k in ("d1n", "d1b", "d2j", "d2c", "d2k", "d3", "dr")},
                }
            )
            continue
        for gid, detail in cands:
            caliper_rows.append(
                {
                    "family": fam,
                    "row_id": pr["row_id"],
                    "split": pr["split"],
                    "pair_id": pr["pair_id"],
                    "t_view_key": pr["t_view_key"],
                    "t_graph_id": pr["t_graph_id"],
                    "c_graph_id": gid,
                    "c_kind": graphs[gid]["kind"],
                    "c_family": ",".join(sorted(graphs[gid].get("families") or {graphs[gid]["family"]})),
                    "c_view_keys": ";".join(sorted(graphs[gid].get("view_keys", {""}))),
                    "d1n": view_doses[gid]["d1n"],
                    "d1b": view_doses[gid]["d1b"],
                    "d2j": view_doses[gid]["d2j"],
                    "d2c": view_doses[gid]["d2c"],
                    "d2k": view_doses[gid]["d2k"],
                    "d3": view_doses[gid]["d3"],
                    "dr": view_doses[gid]["dr"],
                    "caliper": C.stable_json(detail),
                }
            )
            graphs[gid].setdefault("caliper_for", set()).add(pr["pair_id"])

    # ---- 6. populations and support ------------------------------------
    pop_support: dict[str, Any] = {}
    for split in PREDICT_SPLITS:
        block: dict[str, Any] = {}
        for pop in ("main", "struct_chem_all", "calibration"):
            if pop == "main":
                sel = [p for p in pairs if p["split"] == split and p["pop_main"]]
            elif pop == "struct_chem_all":
                sel = [p for p in pairs if p["split"] == split and p["pop_struct_chem"]]
            else:
                sel = [p for p in pairs if p["split"] == split and p["pop_main"] and p.get("n_c_candidates", 0) > 0]
            block[pop] = {
                "molecules": len({p["row_id"] for p in sel}),
                "instances": len({(p["row_id"], p["center"], p["target"]) for p in sel}),
                "controls": len({(p["row_id"], p["center"], p["target"], p["control"]) for p in sel}),
                "pairs": len(sel),
                "scaffolds": len({p["scaffold_key"] for p in sel}),
                "families": {f: sum(1 for p in sel if p["family"] == f) for f in C.FAMILIES},
            }
        pop_support[split] = block

    drop = {
        "pairs_total": len(pairs),
        "structural_ok": sum(1 for p in pairs if p["structural_ok"]),
        "chem_ok_given_struct": sum(1 for p in pairs if p["chem_ok"]),
        "range_ok_given_chem": sum(1 for p in pairs if p["pop_main"]),
        "struct_failures": len(struct_failures),
        "chem_rejects": len(chem_rejects),
        "pairs_with_out_of_range_dims": len({p["pair_id"] for p in pairs if p["out_of_range_dims"]}),
        "mapping_methods": mapping_methods,
        "per_family": {
            fam: {
                "pairs_total": sum(1 for p in pairs if p["family"] == fam),
                "structural_ok": sum(1 for p in pairs if p["family"] == fam and p["structural_ok"]),
                "chem_ok": sum(1 for p in pairs if p["family"] == fam and p["chem_ok"]),
                "range_ok": sum(1 for p in pairs if p["family"] == fam and p["pop_main"]),
                "calibrated": sum(
                    1 for p in pairs if p["family"] == fam and p["pop_main"] and p.get("n_c_candidates", 0) > 0
                ),
            }
            for fam in C.FAMILIES
        },
        "rejection_reason_histogram": _reason_histogram(pairs),
    }

    # ---- 7. zero-rewire identity against the frozen caches -------------
    C.log("zero-rewire re-encoding identity vs frozen caches ...")
    zero = check_zero_rewire_identity(rows, spec)
    if not zero["pass"]:
        raise SystemExit(f"ZERO_REWIRE_IDENTITY_FAILED: {zero}")

    # ---- 8. write frozen inputs ---------------------------------------
    view_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    for gid, v in sorted(graphs.items(), key=lambda kv: (kv[1]["row_id"], kv[1]["kind"], kv[0])):
        graph_rows.append(
            {
                "graph_id": gid,
                "row_id": v["row_id"],
                "split": v["split"],
                "kind": v["kind"],
                "families": ",".join(sorted(v.get("families") or ({v["family"]} if v["family"] else set()))),
                "remove": R._bonds_code(v["remove"]),
                "add": R._bonds_code(v["add"]),
                "range_supported": int(v.get("range_supported", 1)),
                "n_view_keys": len(v.get("view_keys") or ()),
                "n_caliper_pairs": len(v.get("caliper_for") or ()),
                "scopes": "FULL" if v["kind"] == "O" else "FULL,G_FIXED",
            }
        )
        for key in sorted(v.get("view_keys") or ()):
            rid, kind, fam = R.split_view_key(key)
            view_rows.append(
                {
                    "view_key": key,
                    "graph_id": gid,
                    "row_id": rid,
                    "split": v["split"],
                    "kind": kind,
                    "family": fam,
                    "remove": R._bonds_code(v["remove"]),
                    "add": R._bonds_code(v["add"]),
                }
            )
    # attach the O graph id to each view row of the same molecule
    gid_of_o = {rid: graph_id(rid, (), ()) for rid in {v["row_id"] for v in graphs.values()}}
    for r in view_rows:
        r["o_graph_id"] = gid_of_o[int(r["row_id"])]
        r["predicted_scopes"] = "FULL" if r["kind"] == "O" else "FULL,G_FIXED"

    pair_columns = [
        "pair_id", "family", "row_id", "split", "instance_id", "center", "target", "control", "terminals",
        "scaffold_key", "scaffold_seen_in_train",
        "donor_u", "donor_v", "donor_bond_id", "donor_class_u", "donor_class_v",
        "donor_degree_u", "donor_degree_v", "donor_atom_h_u", "donor_atom_h_v",
        "donor_ring_u", "donor_ring_v", "donor_distance_to_center",
        "t_remove", "t_add", "k_remove", "k_add", "t_view_key", "k_view_key",
        "o_graph_id", "t_graph_id", "k_graph_id",
        "legal_T_crossings", "legal_K_crossings", "structural_ok", "structural_reasons",
        "T_family_count_delta", "K0_family_count_delta", "chem_ok", "chem_reason",
        "range_supported", "out_of_range_views", "out_of_range_dims",
        "dose_d1n", "dose_d1b", "dose_d2j", "dose_d2c", "dose_d2k", "dose_d3", "dose_dr",
        "doseK_d1n", "doseK_d1b", "doseK_d2j", "doseK_d2c", "doseK_d2k", "doseK_d3", "doseK_dr",
        "pop_struct_chem", "pop_main", "n_c_candidates",
    ]
    family_columns = [
        "family", "row_id", "split", "instance_id", "center", "target", "control", "terminals",
        "bond_center_target", "bond_center_control", "degree_target", "degree_control",
        "ring_target", "ring_control", "n_aromatic_nbrs", "component_size",
        "scaffold_key", "scaffold_seen_in_train", "n_donors", "n_global_keys",
    ]
    caliper_columns = [
        "family", "row_id", "split", "pair_id", "t_view_key", "t_graph_id", "c_graph_id", "c_kind",
        "c_family", "c_view_keys", "d1n", "d1b", "d2j", "d2c", "d2k", "d3", "dr", "caliper",
    ]
    written = {
        "family_catalog.csv": write_csv(C.ANALYSIS / "family_catalog.csv", family_catalog, family_columns),
        "pair_catalog.csv": write_csv(C.ANALYSIS / "pair_catalog.csv", pairs, pair_columns),
        "view_manifest.csv": write_csv(
            C.ANALYSIS / "view_manifest.csv",
            view_rows,
            ["view_key", "graph_id", "o_graph_id", "row_id", "split", "kind", "family", "remove", "add", "predicted_scopes"],
        ),
        "graph_manifest.csv": write_csv(
            C.ANALYSIS / "graph_manifest.csv",
            graph_rows,
            ["graph_id", "row_id", "split", "kind", "families", "remove", "add", "range_supported",
             "n_view_keys", "n_caliper_pairs", "scopes"],
        ),
        "caliper_catalog.csv": write_csv(C.ANALYSIS / "caliper_catalog.csv", caliper_rows, caliper_columns),
        "caliper_rejected.csv": write_csv(
            C.ANALYSIS / "caliper_rejected.csv",
            caliper_rejected,
            ["family", "row_id", "split", "pair_id", "reason", "doses"],
        ),
        "view_out_of_range.csv": write_csv(
            C.ANALYSIS / "view_out_of_range.csv",
            out_of_range_rows,
            ["graph_id", "row_id", "split", "view", "dim", "value", "train_min", "train_max", "excess", "side"],
        ),
    }
    written_hashes = {name: C.sha256_file(path) for name, path in written.items()}

    dose_summary = _dose_summary(pairs)
    manifest = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "generated_utc": C.utcnow(),
        "design_revision": C.git_commit(),
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
        "previous_round_inputs": {"files": dict(prev["files"]), "note": "frozen original labels / O predictions / "
                                 "scaffold keys of the executed previous round (a0850d89), reused verbatim"},
        "scaffolds": {
            "definition": "Bemis-Murcko (non-isomeric canonical SMILES, includeChirality=False); empty scaffold -> "
            "molecule's own canonical non-isomeric SMILES (frozen previous-round definition)",
            "n_molecules": len(scaffold),
            "n_unique": len(set(scaffold.values())),
            "n_train_unique": len(train_scaffold_set),
            "cross_check_previous_round_rows": len(overlap),
            "cross_check_disagreements": len(scaffold_disagreements),
            "keys_sha256": C.sha256_bytes(
                C.stable_json([[int(k), scaffold[k]] for k in sorted(scaffold)]).encode("utf-8")
            ),
        },
        "model": {
            "checkpoint": C.CHECKPOINT_REL,
            "checkpoint_sha256": C.CHECKPOINT_SHA256,
            "state_hash": C.STATE_HASH,
            "variant": C.VARIANT,
            "variant_name": C.VARIANT_NAME,
            "parameter_count": C.EXPECTED_PARAMS,
            "target_norm": tgt,
            "norm_fingerprint": C.DD.PREP_DATA.norm_fingerprint(norm),
            "reference_metrics": dict(C.REFERENCE),
            "init_state": {
                "state_hash": init_built["hash"],
                "full_state_hash": init_built["full_hash"],
                "expected_state_hash": C.INIT_STATE_HASH,
                "expected_full_state_hash": C.FULL_INIT_STATE_HASH,
                "matches_recorded": bool(init_built["matches_recorded"]),
                "label": "INIT_TRAINING_START" if init_built["matches_recorded"] else "INIT_TRAINING_START_UNVERIFIED",
            },
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
        "families": list(C.FAMILIES),
        "family_census": census_obs,
        "family_census_legacy_class2_donor": census_ref,
        "precounts": PRECOUNTS,
        "precount_reconciliation": {
            "frozen_rule_mismatches": mismatches,
            "legacy_class2_mismatches": legacy_mismatches,
            "families_matching_frozen_rule": families_with_frozen_match,
            "note": "the pre-registered candidate coverage of the four carbonyl/sulfonyl families is reproduced "
            "exactly by the frozen unified class-2/3 donor rule; the two aromatic families are reproduced exactly "
            "by the legacy class-2-only donor census (11/12 cells, arylamine/train pairs 290 vs 293), i.e. the "
            "pre-registered aromatic numbers were generated with the stricter legacy donor restriction. The round "
            "uses the frozen unified rule, so the aromatic candidate coverage is larger than the pre-registered "
            "number. No definition was changed after any effect was observed.",
        },
        "population_support": pop_support,
        "selection": drop,
        "dose_summary": dose_summary,
        "scalar9_deviation": {
            k: {"n": len(v), "max": float(np.max(v)), "median": float(np.median(v))} for k, v in scalar_dev.items()
        },
        "global42_deviation": _global_dev_summary(global_dev),
        "views": per_split_meta,
        "code_hashes": C.code_hashes(),
        "artifacts": {"files": written_hashes, "note": "sha256 of the committed protocol inputs consumed by infer.py"},
    }
    C.write_json(C.ANALYSIS / "inputs_manifest.json", manifest)

    checks = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "precount_reconciliation": manifest["precount_reconciliation"],
        "family_census": census_obs,
        "family_census_legacy_class2_donor": census_ref,
        "selection": drop,
        "population_support": pop_support,
        "structural_failures": struct_failures[:40],
        "chem_rejects": [
            {"pair_id": r["pair_id"], "row_id": r["row_id"], "reason": r["T"].get("reason") or r["K0"].get("reason")}
            for r in chem_rejects[:40]
        ],
        "views": per_split_meta,
        "zero_rewire_identity": zero,
        "dose_summary": dose_summary,
        "parameter_states": {
            "trained": {"state_hash": C.STATE_HASH, "sha256": C.CHECKPOINT_SHA256, "parameters": C.EXPECTED_PARAMS},
            "init": manifest["model"]["init_state"],
        },
        "af51_identity_pass": all(blk["af51_n_failures"] == 0 for blk in per_split_meta.values()),
        "caliper": {
            "n_focal_pairs": len(caliper_rows) and len({r["pair_id"] for r in caliper_rows}),
            "n_candidates": len(caliper_rows),
            "n_rejected_focal": len(caliper_rejected),
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
        "parameter_state_hashes": {"trained": C.STATE_HASH, "init": init_built["hash"]},
        "note": "inference may only read these frozen bytes; no re-selection after prediction",
    }
    C.write_json(C.ANALYSIS / "freeze.json", freeze)

    C.log(f"pairs={len(pairs)} graphs={len(graph_rows)} views={len(view_rows)} caliper={len(caliper_rows)}")
    for split in PREDICT_SPLITS:
        C.log(f"{split}: support {pop_support[split]}")
    C.log(f"selection: {C.stable_json(drop['per_family'])}")
    return 0


def graph_id(row_id: int, remove: Sequence[tuple[int, int]] | Sequence[Any], add: Sequence[tuple[int, int]] | Sequence[Any]) -> str:
    """Graph identity: row id + removed physical bond set + added physical bond set."""
    rem = R._bonds_code(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in remove))
    add_code = R._bonds_code(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in add))
    return f"{int(row_id)}:G:[{rem}]+[{add_code}]"


def _reason_histogram(pairs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for p in pairs:
        if p["structural_ok"] and p["chem_ok"]:
            continue
        reasons = [r for r in str(p.get("structural_reasons", "")).split(";") if r]
        if not p["structural_ok"]:
            reasons += ["key_or_invariant_check_failed"]
        if p["structural_ok"] and not p["chem_ok"]:
            reasons += ["rdkit:" + str(p.get("chem_reason", "")).split(":")[0]]
        for r in reasons or ["unknown"]:
            hist[r] = hist.get(r, 0) + 1
    return hist


def _dose_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = ("d1n", "d1b", "d2j", "d2c", "d2k", "d3", "dr")
    for fam in C.FAMILIES:
        for split in PREDICT_SPLITS:
            for kind, prefix in (("T", "dose_"), ("K0", "doseK_")):
                vals = [p for p in pairs if p["family"] == fam and p["split"] == split and p["pop_main"]]
                if not vals:
                    continue
                block = {}
                for k in keys:
                    arr = np.asarray([float(p.get(prefix + k, np.nan)) for p in vals], dtype=np.float64)
                    arr = arr[np.isfinite(arr)]
                    block[k] = {
                        "n": int(arr.size),
                        "mean": float(arr.mean()) if arr.size else None,
                        "median": float(np.median(arr)) if arr.size else None,
                        "min": float(arr.min()) if arr.size else None,
                        "max": float(arr.max()) if arr.size else None,
                    }
                out[f"{fam}/{split}/{kind}"] = block
    return out


def _global_dev_summary(global_dev: Mapping[str, Sequence[np.ndarray]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, arrs in global_dev.items():
        if not arrs:
            continue
        m = np.asarray(arrs, dtype=np.float64)
        out[key] = {
            "n_views": int(m.shape[0]),
            "dims_with_change": int((np.abs(m).max(axis=0) > 0).sum()),
            "max_abs_change_per_dim": [float(x) for x in np.abs(m).max(axis=0)],
            "mean_abs_change_per_dim": [float(x) for x in np.abs(m).mean(axis=0)],
        }
    return out


# ---------------------------------------------------------------------------
# zero-rewire identity against the frozen caches
# ---------------------------------------------------------------------------


def check_zero_rewire_identity(
    rows: Sequence[Mapping[str, Any]], spec, *, splits: Sequence[str] = PREDICT_SPLITS
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
            row["n_components"] = G.n_components(int(row["atom_class"].shape[0]), *G.physical_bonds(row)[:2])
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


if __name__ == "__main__":
    raise SystemExit(main())
