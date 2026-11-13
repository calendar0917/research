"""Part-B Stage A: build the two oracle label tables and the training/CF isolation lists.

Runs locally on CPU, before any Part-B training, and writes frozen bytes only:

* ``wc_logp_oracle.csv`` / ``molwt_oracle.csv`` — one row per graph (all 4200
  original molecules plus every frozen Valid/Test O/T/K0/C graph), keyed by
  ``graph_id`` + ``row_id`` + ``split`` + ``kind`` with a per-row sha256.
* ``oracle_labels.csv`` — the shared wide table (both targets + canonical
  isomeric SMILES + exact-Train-duplicate flag).
* ``cf_isolation.csv`` — one row per frozen Valid/Test main (range-supported)
  pair with the train-duplicate marks and the main-CF flag (O, T and K0 all
  absent from the Train original set).
* ``test_train_duplicates.csv`` — original Test rows exactly equal (isomeric
  SMILES) to a Train original, with the novel-structure stratum.
* ``oracle_manifest.json`` / ``freeze_oracle.json`` — checks and byte hashes.

Label-blind with respect to the model: no prediction and no original ``exp``
label is read here.  Only composition / structure and the two RDKit functions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import round_common as RC
import oracle as O

PRIMARY_TASKS = RC.TASKS


def _label_record_sha(graph_id: str, row_id: int, split: str, kind: str, task: str, value: float) -> str:
    payload = f"{graph_id}|{int(row_id)}|{split}|{kind}|{task}|{float(value):.17g}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scan_feature_leak(spec, train_cache, labels: Mapping[int, float], task: str) -> dict[str, Any]:
    """Direct-label-leak check: no global42 or atom-scalar column equals the target."""
    from rdkit.Chem import Descriptors

    global_names = [f.name for f in spec.global_feature_fields]
    scalar_names = [f.name for f in spec.atom_scalar_fields]
    tokens = ("logp", "log_p", "crippen", "molwt", "mol_wt", "molecular_weight", "wildman")
    name_hits = [
        f"global:{n}" for n in global_names if any(t in n.lower() for t in tokens)
    ] + [f"scalar:{n}" for n in scalar_names if any(t in n.lower() for t in tokens)]
    # numeric exact-equality check over the frozen Train original cache
    g = np.asarray(train_cache["global"], dtype=np.float64)
    rows = [int(x) for x in np.asarray(train_cache["rows"]).tolist()]
    y = np.asarray([float(labels[int(r)]) for r in rows], dtype=np.float64)
    max_abs_dim_diff = [float(np.abs(g[:, d] - y).max()) for d in range(g.shape[1])]
    min_dim_diff = float(np.min(max_abs_dim_diff)) if max_abs_dim_diff else float("nan")
    exact_global_match = int(np.sum([m <= 1e-9 for m in max_abs_dim_diff]))
    return {
        "target": task,
        "name_token_hits": name_hits,
        "global_fields": global_names,
        "atom_scalar_fields": scalar_names,
        "global_min_maxabs_diff_to_target": min_dim_diff,
        "global_dims_exactly_equal_to_target": exact_global_match,
        "direct_leak": bool(name_hits) or bool(exact_global_match > 0),
        "note": "name scan over the frozen lipo_v1 global42 / scalar9 vocabulary plus an exact-equality "
                "numeric check on the Train originals; a direct leak would stop Part B",
    }


def main() -> int:
    RC.NEW_ANALYSIS.mkdir(parents=True, exist_ok=True)
    spec = RC.frozen_spec()
    src = RC.SC.load_source("none")
    raw = RC.SC.load_raw_rows()["rows"]["Lipophilicity"]
    prev = RC.SC.load_previous_inputs()  # original labels / split / scaffold (for isolation only)
    split_of = {int(r): str(prev["split"][int(r)]) for r in prev["split"]}
    # The previous-round molecule_catalog only covers molecules with an amide instance;
    # recompute the frozen Bemis-Murcko key for all 4200 rows (same definition).
    scaffold = {rid: RC.G.bemis_murcko_key(str(raw[rid]["smiles"])) for rid in range(RC.SC.N_MOLECULES)}
    train_scaffold_set = {scaffold[int(r)] for r in RC.SC.load_source("none")["rows"]["train"]}

    C = RC.log
    C("oracle: building original RDKit molecules for all 4200 rows ...")
    originals: dict[int, dict[str, Any]] = {}
    mapping: dict[int, list[int]] = {}
    mapping_methods: dict[str, int] = {}
    for rid in range(RC.SC.N_MOLECULES):
        smiles = str(raw[rid]["smiles"])
        mol = O.parse_smiles(smiles)
        mp = RC.CHEM.smiles_mapping(smiles, raw[rid], rid)
        if mp["method"] == "none":
            raise SystemExit(f"ORACLE_MAPPING_FAILED: row {rid}")
        mapping_methods[mp["method"]] = mapping_methods.get(mp["method"], 0) + 1
        mapping[rid] = [int(x) for x in mp["raw_to_rdkit"]]
        if mp["method"] == "identity" and mapping[rid] != list(range(int(np.asarray(raw[rid]["atom_class"]).size))):
            raise SystemExit(f"ORACLE_MAPPING_NONIDENTITY_ORDER: row {rid}")
        vals = O.compute_both(mol)
        originals[rid] = {
            "mol": mol,
            "wc_logp": float(vals["wc_logp"]),
            "molwt": float(vals["molwt"]),
            "canonical_isomeric": O.canonical_isomeric(mol),
            "canonical_non_isomeric": O.canonical_non_isomeric(mol),
        }
    C(f"oracle: originals done; mapping_methods={mapping_methods}")

    for task in PRIMARY_TASKS:
        arr = np.asarray([originals[r][task] for r in range(RC.SC.N_MOLECULES)], dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise SystemExit(f"ORACLE_NONFINITE: {task}")
        train_rows = [int(x) for x in src["rows"]["train"]]
        tr = np.asarray([originals[r][task] for r in train_rows], dtype=np.float64)
        if float(tr.std(ddof=0)) <= 0.0:
            raise SystemExit(f"ORACLE_TRAIN_STD_ZERO: {task}")
        C(f"oracle: {task} originals finite, train mean={tr.mean():.6f} std={tr.std(ddof=0):.6f}")

    # ---- Train canonical set (isolation marker) -------------------------
    train_rows = [int(x) for x in src["rows"]["train"]]
    train_canonical = {originals[r]["canonical_isomeric"] for r in train_rows}

    # ---- graph catalogue (Valid/Test O/T/K0; C graphs are a subset) -----
    graph_rows = RC.read_csv(RC.SOURCE_ANALYSIS / "graph_manifest.csv")
    C(f"oracle: {len(graph_rows)} frozen graphs to label")

    labels: list[dict[str, Any]] = []
    independent_checks: list[dict[str, Any]] = []
    edit_checks: list[dict[str, Any]] = []
    n_nonO_seen = 0
    for g in graph_rows:
        gid = str(g["graph_id"])
        rid = int(g["row_id"])
        split = str(g["split"])
        kind = str(g["kind"])
        remove = RC.parse_bonds(g["remove"])
        add = RC.parse_bonds(g["add"])
        if kind == "O":
            mol = originals[rid]["mol"]
            vals = {"wc_logp": originals[rid]["wc_logp"], "molwt": originals[rid]["molwt"]}
            canon = originals[rid]["canonical_isomeric"]
            canon_non = originals[rid]["canonical_non_isomeric"]
            o_wc = originals[rid]["wc_logp"]
            o_mw = originals[rid]["molwt"]
            source_kind = "original"
        else:
            mol = O.edited_rdkit_mol(originals[rid]["mol"], mapping[rid], remove, add)
            # raw view row for the independent bond/atom identity check
            base = RC.FEAT.row_from_artifact({**dict(raw[rid]), "mol_index": rid})
            view_row, _info = RC.R.edit_row(base, remove, add)
            ident = O.verify_edit_identity(
                originals[rid]["mol"], mol, view_row, mapping[rid], tol=RC.ORACLE_INVARIANT_TOL
            )
            edit_checks.append({"graph_id": gid, "row_id": rid, "kind": kind, **ident})
            if not ident["pass"]:
                raise SystemExit(f"ORACLE_EDIT_IDENTITY_FAILED: {gid}: {ident}")
            vals = O.compute_both(mol)
            canon = O.canonical_isomeric(mol)
            canon_non = O.canonical_non_isomeric(mol)
            o_wc = originals[rid]["wc_logp"]
            o_mw = originals[rid]["molwt"]
            source_kind = "rewire"
            if n_nonO_seen < 20:
                indep = O.independent_reparse_values(mol)
                independent_checks.append(
                    {
                        "graph_id": gid,
                        "row_id": rid,
                        "kind": kind,
                        "smiles": indep["smiles"],
                        "wc_logp_direct": float(vals["wc_logp"]),
                        "wc_logp_reparse": float(indep["wc_logp"]),
                        "molwt_direct": float(vals["molwt"]),
                        "molwt_reparse": float(indep["molwt"]),
                        "abs_wc_diff": abs(float(vals["wc_logp"]) - float(indep["wc_logp"])),
                        "abs_mw_diff": abs(float(vals["molwt"]) - float(indep["molwt"])),
                        "smiles_equal": indep["smiles"] == canon,
                    }
                )
                n_nonO_seen += 1
        labels.append(
            {
                "graph_id": gid,
                "row_id": rid,
                "split": split,
                "kind": kind,
                "source_kind": source_kind,
                "families": str(g.get("families", "")),
                "n_view_keys": int(g.get("n_view_keys", 0) or 0),
                "canonical_isomeric": canon,
                "canonical_non_isomeric": canon_non,
                "wc_logp": float(vals["wc_logp"]),
                "molwt": float(vals["molwt"]),
                "wc_delta_vs_O": float(vals["wc_logp"]) - float(o_wc),
                "molwt_delta_vs_O": float(vals["molwt"]) - float(o_mw),
                "exact_train_duplicate": int(canon in train_canonical),
            }
        )

    # ---- original-row labels for all 4200 (Train included) --------------
    for rid in range(RC.SC.N_MOLECULES):
        labels.append(
            {
                "graph_id": f"{rid}:G:[]+[]",
                "row_id": rid,
                "split": split_of[rid],
                "kind": "O",
                "source_kind": "original",
                "families": "",
                "n_view_keys": 0,
                "canonical_isomeric": originals[rid]["canonical_isomeric"],
                "canonical_non_isomeric": originals[rid]["canonical_non_isomeric"],
                "wc_logp": originals[rid]["wc_logp"],
                "molwt": originals[rid]["molwt"],
                "wc_delta_vs_O": 0.0,
                "molwt_delta_vs_O": 0.0,
                "exact_train_duplicate": int(originals[rid]["canonical_isomeric"] in train_canonical),
            }
        )

    # de-duplicate the O rows that appear both in graph_manifest and the 4200 loop
    seen_keys: set[tuple[str, str]] = set()
    uniq: list[dict[str, Any]] = []
    for r in labels:
        key = (r["graph_id"], r["source_kind"]) if r["graph_id"].endswith(":G:[]+[]") and r["source_kind"] == "original" else (r["graph_id"], "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        uniq.append(r)
    labels = uniq

    # ---- checks ---------------------------------------------------------
    rewire = [r for r in labels if r["source_kind"] == "rewire"]
    if not rewire:
        raise SystemExit("ORACLE_NO_REWIRE_GRAPHS")
    mw_deltas = np.asarray([abs(r["molwt_delta_vs_O"]) for r in rewire], dtype=np.float64)
    wc_deltas = np.asarray([abs(r["wc_delta_vs_O"]) for r in rewire], dtype=np.float64)
    molwt_invariant_pass = bool(float(mw_deltas.max()) <= RC.ORACLE_INVARIANT_TOL)
    wc_changed_frac = float(np.mean(wc_deltas > RC.WC_LOG_CHANGE_TOL))
    independent_pass = bool(
        all(c["abs_wc_diff"] <= 1e-9 and c["abs_mw_diff"] <= 1e-9 for c in independent_checks)
    )
    edit_identity_pass = all(e["pass"] for e in edit_checks)
    if not molwt_invariant_pass:
        raise SystemExit(f"ORACLE_MOLWT_INVARIANT_FAILED: max |delta| {float(mw_deltas.max()):.3e}")
    if not independent_pass:
        raise SystemExit("ORACLE_INDEPENDENT_REPARSE_FAILED")
    if not edit_identity_pass:
        raise SystemExit("ORACLE_EDIT_IDENTITY_FAILED_EARLY")

    task_stats: dict[str, Any] = {}
    for task in PRIMARY_TASKS:
        train_vals = np.asarray([originals[r][task] for r in train_rows], dtype=np.float64)
        task_stats[task] = {
            "train_mean": float(train_vals.mean()),
            "train_std_ddof0": float(train_vals.std(ddof=0)),
            "train_min": float(train_vals.min()),
            "train_max": float(train_vals.max()),
            "n": int(train_vals.size),
            "units": RC.TASK_UNITS[task],
            "call": RC.WC_LOGP_CALL if task == "wc_logp" else RC.MOLWT_CALL,
        }

    # ---- Train/CF isolation pairs --------------------------------------
    pair_rows = RC.read_csv(RC.SOURCE_ANALYSIS / "pair_catalog.csv")
    caliper = RC.read_csv(RC.SOURCE_ANALYSIS / "caliper_catalog.csv")
    c_by_pair: dict[str, set[str]] = {}
    for r in caliper:
        c_by_pair.setdefault(str(r["pair_id"]), set()).add(str(r["c_graph_id"]))
    label_by_gid = {r["graph_id"]: r for r in labels}
    cf_rows: list[dict[str, Any]] = []
    for p in pair_rows:
        if int(float(p["pop_main"] or 0)) != 1:
            continue
        o_gid = str(p["o_graph_id"]); t_gid = str(p["t_graph_id"]); k_gid = str(p["k_graph_id"])
        o_l = label_by_gid[o_gid]; t_l = label_by_gid[t_gid]; k_l = label_by_gid[k_gid]
        main_cf = int(
            not o_l["exact_train_duplicate"] and not t_l["exact_train_duplicate"] and not k_l["exact_train_duplicate"]
        )
        cf_rows.append(
            {
                "pair_id": str(p["pair_id"]),
                "family": str(p["family"]),
                "row_id": int(p["row_id"]),
                "split": str(p["split"]),
                "instance_id": str(p["instance_id"]),
                "center": int(p["center"]),
                "target": int(p["target"]),
                "control": int(p["control"]),
                "o_graph_id": o_gid,
                "t_graph_id": t_gid,
                "k_graph_id": k_gid,
                "scaffold_key": str(p["scaffold_key"]),
                "scaffold_seen_in_train": int(p["scaffold_seen_in_train"]),
                "n_c_candidates": len(c_by_pair.get(str(p["pair_id"]), set())),
                "c_is_own_k0_only": 0,
                "o_exact_train_duplicate": int(o_l["exact_train_duplicate"]),
                "t_exact_train_duplicate": int(t_l["exact_train_duplicate"]),
                "k_exact_train_duplicate": int(k_l["exact_train_duplicate"]),
                "main_cf": main_cf,
                "cf_population": "main_cf" if main_cf else "supplement_full_queue",
            }
        )

    # ---- original Test duplicates of Train originals -------------------
    test_rows = [int(x) for x in src["rows"]["test"]]
    dup_rows = [
        {
            "row_id": rid,
            "canonical_isomeric": originals[rid]["canonical_isomeric"],
            "split": "test",
            "exact_train_duplicate": int(originals[rid]["canonical_isomeric"] in train_canonical),
            "structure_stratum": "seen_exact_structure_in_train"
            if originals[rid]["canonical_isomeric"] in train_canonical
            else ("seen_scaffold_in_train" if scaffold[rid] in train_scaffold_set else "novel_scaffold"),
        }
        for rid in test_rows
    ]
    valid_rows = [int(x) for x in src["rows"]["valid"]]
    dup_valid = [
        {
            "row_id": rid,
            "canonical_isomeric": originals[rid]["canonical_isomeric"],
            "split": "valid",
            "exact_train_duplicate": int(originals[rid]["canonical_isomeric"] in train_canonical),
        }
        for rid in valid_rows
    ]

    # ---- direct leakage check ------------------------------------------
    train_cache = RC.load_original_cache("train")
    leak_checks = {task: _scan_feature_leak(spec, train_cache, {r: originals[r][task] for r in train_rows}, task) for task in PRIMARY_TASKS}
    if any(v["direct_leak"] for v in leak_checks.values()):
        raise SystemExit(f"ORACLE_DIRECT_LABEL_LEAK: {leak_checks}")

    # ---- write label tables --------------------------------------------
    wide_cols = [
        "graph_id", "row_id", "split", "kind", "source_kind", "families", "n_view_keys",
        "canonical_isomeric", "canonical_non_isomeric", "wc_logp", "molwt",
        "wc_delta_vs_O", "molwt_delta_vs_O", "exact_train_duplicate",
    ]
    RC.write_csv(RC.NEW_ANALYSIS / "oracle_labels.csv", labels, wide_cols)
    files: dict[str, str] = {"oracle_labels.csv": RC.sha256_file(RC.NEW_ANALYSIS / "oracle_labels.csv")}
    for task in PRIMARY_TASKS:
        rows = [
            {
                "graph_id": r["graph_id"],
                "row_id": r["row_id"],
                "split": r["split"],
                "kind": r["kind"],
                "source_kind": r["source_kind"],
                "families": r["families"],
                "value": r[task],
                "task": task,
                "units": RC.TASK_UNITS[task],
                "sha256": _label_record_sha(r["graph_id"], r["row_id"], r["split"], r["kind"], task, r[task]),
            }
            for r in labels
        ]
        name = f"{task}_oracle.csv"
        RC.write_csv(RC.NEW_ANALYSIS / name, rows,
                     ["graph_id", "row_id", "split", "kind", "source_kind", "families", "value", "task", "units", "sha256"])
        files[name] = RC.sha256_file(RC.NEW_ANALYSIS / name)

    rc_cf = [
        "pair_id", "family", "row_id", "split", "instance_id", "center", "target", "control",
        "o_graph_id", "t_graph_id", "k_graph_id", "scaffold_key", "scaffold_seen_in_train",
        "n_c_candidates", "o_exact_train_duplicate", "t_exact_train_duplicate", "k_exact_train_duplicate",
        "main_cf", "cf_population",
    ]
    RC.write_csv(RC.NEW_ANALYSIS / "cf_isolation.csv", cf_rows, rc_cf)
    files["cf_isolation.csv"] = RC.sha256_file(RC.NEW_ANALYSIS / "cf_isolation.csv")
    RC.write_csv(RC.NEW_ANALYSIS / "test_train_duplicates.csv", dup_rows,
                 ["row_id", "canonical_isomeric", "split", "exact_train_duplicate", "structure_stratum"])
    RC.write_csv(RC.NEW_ANALYSIS / "valid_train_duplicates.csv", dup_valid,
                 ["row_id", "canonical_isomeric", "split", "exact_train_duplicate"])
    files["test_train_duplicates.csv"] = RC.sha256_file(RC.NEW_ANALYSIS / "test_train_duplicates.csv")
    files["valid_train_duplicates.csv"] = RC.sha256_file(RC.NEW_ANALYSIS / "valid_train_duplicates.csv")

    manifest = {
        "round_id": RC.ROUND_ID,
        "protocol_id": RC.PROTOCOL_ID,
        "generated_utc": RC.utcnow(),
        "git_commit": RC.git_commit(),
        "git_dirty": RC.git_dirty(),
        "source_round": {
            "track": RC.track_rel(RC.SOURCE_TRACK),
            "protocol_revision": RC.SOURCE_PROTOCOL_REVISION,
            "report_revision": RC.SOURCE_REPORT_REVISION,
            "pair_catalog_sha256": RC.sha256_file(RC.SOURCE_ANALYSIS / "pair_catalog.csv"),
            "caliper_catalog_sha256": RC.sha256_file(RC.SOURCE_ANALYSIS / "caliper_catalog.csv"),
            "graph_manifest_sha256": RC.sha256_file(RC.SOURCE_ANALYSIS / "graph_manifest.csv"),
        },
        "rdkit_version": RC.rdkit_version(),
        "oracle_calls": {"wc_logp": RC.WC_LOGP_CALL, "molwt": RC.MOLWT_CALL},
        "tasks": list(PRIMARY_TASKS),
        "target_norm_train_only": {task: task_stats[task] for task in PRIMARY_TASKS},
        "feature_spec": {
            "hash": RC.SC.DC.FEATURE_SPEC_HASH,
            "global_dim": RC.SC.DC.GLOBAL_DIM,
            "atom_scalar_dim": RC.SC.DC.ATOM_SCALAR_DIM,
            "note": "frozen lipo_v1 spec; the feature normalisation is the checkpoint's frozen Train-only buffers "
                    "and is never refitted",
        },
        "originals": {
            "n": int(RC.SC.N_MOLECULES),
            "mapping_methods": mapping_methods,
        },
        "checks": {
            "originals_finite": True,
            "train_std_positive": True,
            "molwt_invariant_max_abs_delta": float(mw_deltas.max()),
            "molwt_invariant_pass": bool(molwt_invariant_pass),
            "wc_logp_rewire_fraction_changed_gt_1e-8": float(wc_changed_frac),
            "wc_logp_rewire_abs_delta": {
                "n": int(wc_deltas.size),
                "mean": float(wc_deltas.mean()),
                "median": float(np.median(wc_deltas)),
                "p95": float(np.percentile(wc_deltas, 95)),
                "max": float(wc_deltas.max()),
                "n_changed": int((wc_deltas > RC.WC_LOG_CHANGE_TOL).sum()),
            },
            "independent_reparse": {
                "n_checked": len(independent_checks),
                "pass": bool(independent_pass),
                "max_abs_wc_diff": float(max((c["abs_wc_diff"] for c in independent_checks), default=0.0)),
                "max_abs_mw_diff": float(max((c["abs_mw_diff"] for c in independent_checks), default=0.0)),
                "all_smiles_roundtrip_equal": bool(all(c["smiles_equal"] for c in independent_checks)),
            },
            "edit_identity": {
                "n_checked": len(edit_checks),
                "pass": bool(edit_identity_pass),
                "n_failed": int(sum(1 for e in edit_checks if not e["pass"])),
            },
            "direct_leak": leak_checks,
            "train_canonical_set_size": len(train_canonical),
            "cf_pairs": {
                "n_main_queue": len(cf_rows),
                "n_main_cf": int(sum(r["main_cf"] for r in cf_rows)),
                "n_supplement": int(sum(1 - r["main_cf"] for r in cf_rows)),
            },
            "test_originals_exact_train_duplicates": int(sum(r["exact_train_duplicate"] for r in dup_rows)),
            "valid_originals_exact_train_duplicates": int(sum(r["exact_train_duplicate"] for r in dup_valid)),
        },
        "independent_reparse_detail": independent_checks,
        "artifacts": files,
        "code_hashes": RC.code_hashes(),
    }
    RC.write_json(RC.NEW_ANALYSIS / "oracle_manifest.json", manifest)
    freeze = {
        "round_id": RC.ROUND_ID,
        "generated_utc": RC.utcnow(),
        "files": {**files, "oracle_manifest.json": RC.sha256_file(RC.NEW_ANALYSIS / "oracle_manifest.json")},
        "note": "Part B may only read these frozen bytes; no label re-selection after any training result",
    }
    RC.write_json(RC.NEW_ANALYSIS / "freeze_oracle.json", freeze)
    RC.log(f"oracle done: {len(labels)} label rows, {len(cf_rows)} CF pairs "
           f"(main_cf={manifest['checks']['cf_pairs']['n_main_cf']}), wc_changed_frac={wc_changed_frac:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
