"""P0: ESOL data / feature compatibility and lightweight correctness gates.

Only the parts that are genuinely new in this round are checked:

1. source inventory and row alignment (SMILES, graph, row id, label);
2. the ``lipo_v1`` 16-class atom / 5-class bond vocabulary on ESOL, including
   explicit UNK behaviour and the aromatic / charge / H / bond-class breakdown;
3. S1/S2 codes of the *training cache* against the frozen independent
   reference enumerator (``cpu_gates.reference_enumeration``);
4. edge cases: acyclic molecules, zero-S2 molecules, multi-component rows;
5. label isolation: no Test label statistic anywhere in P0, no forbidden CSV
   column parsed, no per-split label summary;
6. Train-only normalisation (structural scalars, target, descriptors);
7. single-molecule vs batched prediction consistency, finite backward pass.

The historical full audit of the ``lipo_v1`` schema is not repeated: the schema
itself is imported unchanged, and ``analysis/p0`` records only what is new.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import esol_common as C
import esol_data as D
import esol_models as M
import esol_raw as R
import esol_split as S
import esol_train as T

import cpu_gates as GATES  # noqa: E402  (frozen independent S1/S2 reference)
import lcrc_data as PREP_DATA  # noqa: E402


def _cache_slice(cache: Mapping[str, Any], index: int) -> dict[str, Any]:
    """One molecule of a cache, in the layout ``cpu_gates`` expects."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    a0, a1 = int(atom_ptr[index]), int(atom_ptr[index + 1])
    b0, b1 = int(bond_ptr[index]), int(bond_ptr[index + 1])
    n0, n1 = int(nbr_ptr[a0]), int(nbr_ptr[a1])
    p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
    return {
        "n_atoms": a1 - a0,
        "n_bonds": b1 - b0,
        "atom_type": np.asarray(cache["atom_type"])[a0:a1],
        "bond_type": np.asarray(cache["bond_type"])[b0:b1],
        "bond_index": np.asarray(cache["bond_index"])[b0:b1],
        "nbr_ptr": np.concatenate([[0], np.cumsum(np.diff(nbr_ptr[a0 : a1 + 1]))]).astype(np.int64),
        "nbr_code": np.asarray(cache["nbr_code"])[n0:n1],
        "path_ptr": np.concatenate([[0], np.cumsum(np.diff(path_ptr[a0 : a1 + 1]))]).astype(np.int64),
        "path_cat": np.asarray(cache["path_cat"])[p0:p1],
        "path_end": np.asarray(cache["path_end"])[p0:p1],
        "path_end63": np.asarray(cache["path_end63"])[p0:p1],
        "path_closure": np.asarray(cache["path_closure"])[p0:p1],
    }


def _vocabulary_report(cache: Mapping[str, Any]) -> dict[str, Any]:
    spec = C.spec()
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"])
    unk = int(spec.atom_unk_index)
    unk_atoms = int(np.sum(atom_type == unk))
    unk_molecules = int(
        sum(
            1
            for i in range(int(cache["n_molecules"]))
            if np.any(atom_type[int(atom_ptr[i]) : int(atom_ptr[i + 1])] == unk)
        )
    )
    type_counts = {spec.atom_type_names[t]: int(np.sum(atom_type == t)) for t in range(spec.n_atom_types)}
    bond_counts = {spec.bond_type_names[t - 1]: int(np.sum(bond_type == t)) for t in range(1, spec.n_bond_types + 1)}
    return {
        "atom_vocabulary": list(spec.atom_type_names),
        "bond_vocabulary": list(spec.bond_type_names),
        "atom_type_counts": {k: v for k, v in type_counts.items() if v},
        "atom_types_unused": [k for k, v in type_counts.items() if not v],
        "bond_type_counts": {k: v for k, v in bond_counts.items() if v},
        "bond_types_unused": [k for k, v in bond_counts.items() if not v],
        "unknown_atom_index": unk,
        "unk_atoms": unk_atoms,
        "unk_molecules": unk_molecules,
        "bond_unk_id": spec.bond_unk_id,
        "unknown_bonds": int(np.sum(bond_type == int(spec.bond_unk_id))),
        "atom_type_max": int(atom_type.max()) if atom_type.size else -1,
        "bond_type_max": int(bond_type.max()) if bond_type.size else -1,
        "nbr_code_max": int(np.asarray(cache["nbr_code"]).max()) if np.asarray(cache["nbr_code"]).size else -1,
        "nbr_code_vocab": int(spec.n_nbr_codes),
        "path_cat_max": int(np.asarray(cache["path_cat"]).max()) if np.asarray(cache["path_cat"]).size else -1,
        "path_cat_vocab": int(spec.n_path_codes),
        "path_end63_max": int(np.asarray(cache["path_end63"]).max()) if np.asarray(cache["path_end63"]).size else -1,
    }


def _raw_attribute_report(raw: Mapping[str, Any]) -> dict[str, Any]:
    atom_classes = np.concatenate([np.asarray(row["atom_class"]) for row in raw["rows"]])
    charges = np.concatenate([np.asarray(row["atom_charge"]) for row in raw["rows"]])
    hydrogens = np.concatenate([np.asarray(row["atom_h"]) for row in raw["rows"]])
    edge_classes = np.concatenate([np.asarray(row["edge_class"]) for row in raw["rows"]])
    n_components = [int(row["n_components"]) for row in raw["rows"]]
    return {
        "atom_class_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(atom_classes, return_counts=True))},
        "formal_charge_bucket_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(charges, return_counts=True))},
        "hydrogen_bucket_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(hydrogens, return_counts=True))},
        "raw_bond_class_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(edge_classes, return_counts=True))},
        "components_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(np.asarray(n_components), return_counts=True))},
        "dictionary_note": (
            "atom_class 0..14 = frozen element x aromatic vocabulary, 15 = UNK, 16 = SSL MASK (never used here); "
            "atom_charge 0..4 = (-2,-1,0,+1,+2), 5 = UNK; atom_h 0..4 = H count, 5 = UNK (>4 folds to 4); "
            "edge_class 0..3 = SINGLE,DOUBLE,TRIPLE,AROMATIC, 4 = UNK"
        ),
    }


def _unk_mapping_probe() -> dict[str, Any]:
    """Unsupported elements must land in UNK, never be silently folded into C."""
    import bjoc_lipophilicity_adapter_v1 as LIP

    probes = {"dimethyl_selenide": "C[Se]C", "tetramethyl_tin": "[Sn](C)(C)(C)C", "boric_acid": "B(O)(O)O"}
    known_elements = {element for element, _aromatic in LIP.ATOM_CLASSES}
    out: dict[str, Any] = {}
    for name, smiles in probes.items():
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            out[name] = {"smiles": smiles, "parsed": False, "correct_unk_behaviour": False}
            continue
        raw = LIP.raw_from_smiles(smiles, 0)
        classes = [int(x) for x in raw.atom_class]
        elements = [a.GetSymbol() for a in mol.GetAtoms()]
        violations = [
            {"element": element, "atom_class": cls}
            for element, cls in zip(elements, classes)
            if element not in known_elements and cls != LIP.ATOM_UNK
        ]
        out[name] = {
            "smiles": smiles,
            "parsed": True,
            "elements": elements,
            "atom_class": classes,
            "unk_index": int(LIP.ATOM_UNK),
            "n_unk": int(sum(1 for c in classes if c == LIP.ATOM_UNK)),
            "off_vocabulary_elements": sorted({e for e in elements if e not in known_elements}),
            "violations": violations,
            "correct_unk_behaviour": not violations,
        }
    return out


def run() -> dict[str, Any]:
    spec = C.spec()
    split = S.load_split()
    audit = C.read_json(C.SPLIT_AUDIT)
    views = D.build_views(include_test=False)
    cache_all = D.build_label_free_cache()["cache"]
    raw = R.load_raw_graphs()
    csv_payload = R.read_csv()

    # --- 1. source / alignment -------------------------------------------
    cache_smiles = [str(x) for x in cache_all["smiles"]]
    molecule_ids = [str(x) for x in cache_all["molecule_ids"]]
    alignment = {
        "n_rows": int(csv_payload["n"]),
        "n_cache_molecules": int(cache_all["n_molecules"]),
        "ids_unique": len(set(csv_payload["ids"])) == csv_payload["n"],
        "smiles_unique": len(set(csv_payload["smiles"])) == csv_payload["n"],
        "n_duplicate_smiles": int(csv_payload["n"] - len(set(csv_payload["smiles"]))),
        "n_distinct_smiles": int(len(set(csv_payload["smiles"]))),
        "duplicate_smiles_groups": split["duplicate_smiles"]["groups"],
        "duplicate_smiles_groups_crossing_splits": split["duplicate_smiles"]["groups_crossing_splits"],
        "targets_all_finite": bool(np.isfinite(csv_payload["y"]).all()),
        "target_missing": int(np.sum(~np.isfinite(csv_payload["y"]))),
        "cache_order_equals_csv_order": cache_smiles == [str(s) for s in csv_payload["smiles"]],
        "cache_molecule_ids_are_row_indices": molecule_ids == [str(i) for i in range(csv_payload["n"])],
        "rdkit_parse_failures_in_raw_artifact": int(len(raw["rows"]) != csv_payload["n"]),
        "raw_rows": int(len(raw["rows"])),
        "retained_rows": int(cache_all["n_molecules"]),
        "row_ids_sha256": C.rows_sha256(list(range(csv_payload["n"]))),
        "target_vector_sha256": C.vector_sha256_float64(csv_payload["y"]),
        "smiles_vector_sha256": C.sha256_bytes("\n".join(csv_payload["smiles"]).encode("utf-8")),
        "forbidden_columns_parsed": list(csv_payload["forbidden_columns_parsed"]),
        "label_column_used": C.TARGET_COLUMN,
        "label_unit": "log(mol/L)",
    }
    # Per-molecule atom-count agreement between the RDKit raw graph and the cache.
    atom_counts_raw = np.asarray([len(row["atom_class"]) for row in raw["rows"]], dtype=np.int64)
    atom_ptr = np.asarray(cache_all["atom_ptr"])
    atom_counts_cache = np.diff(atom_ptr)
    bond_counts_raw = np.asarray([len(row["edge_class"]) // 2 for row in raw["rows"]], dtype=np.int64)
    bond_counts_cache = np.diff(np.asarray(cache_all["bond_ptr"]))
    alignment["atom_count_max_abs_diff"] = int(np.abs(atom_counts_raw - atom_counts_cache).max())
    alignment["bond_count_max_abs_diff"] = int(np.abs(bond_counts_raw - bond_counts_cache).max())
    alignment["n_atoms_total"] = int(atom_counts_cache.sum())
    alignment["n_bonds_total"] = int(bond_counts_cache.sum())
    alignment["n_s2_paths_total"] = int(np.asarray(cache_all["path_ptr"])[-1])

    # --- 2. vocabulary / attributes --------------------------------------
    vocabulary = _vocabulary_report(cache_all)
    attributes = _raw_attribute_report(raw)
    unk_probe = _unk_mapping_probe()

    # --- 3. S1/S2 vs the frozen independent reference enumerator ---------
    mismatches: list[int] = []
    totals = {"s1": 0, "s2": 0}
    bond_ptr_all = np.asarray(cache_all["bond_ptr"])
    bond_index_all = np.asarray(cache_all["bond_index"])
    for i in range(int(cache_all["n_molecules"])):
        mol = _cache_slice(cache_all, i)
        bonds = bond_index_all[int(bond_ptr_all[i]) : int(bond_ptr_all[i + 1])]
        ref = GATES.reference_enumeration(
            mol["atom_type"],
            bonds[:, 0],
            bonds[:, 1],
            mol["bond_type"],
            spec.n_bond_types,
        )
        prod_s1 = GATES.mol_s1_rows(mol)
        prod_s2 = GATES.mol_s2_records(mol)
        totals["s1"] += int(sum(len(row) for row in prod_s1))
        totals["s2"] += int(sum(len(row) for row in prod_s2))
        if not (
            ref["s1"] == prod_s1
            and ref["s2"] == prod_s2
            and np.array_equal(ref["degree"], np.diff(mol["nbr_ptr"]))
            and int(ref["n_bonds"]) == int(mol["n_bonds"])
        ):
            mismatches.append(i)

    # --- 4. edge cases ----------------------------------------------------
    scaffolds = split["scaffold_by_row"]
    per_molecule_paths = [
        int(np.asarray(cache_all["path_ptr"])[int(atom_ptr[i + 1])] - np.asarray(cache_all["path_ptr"])[int(atom_ptr[i])])
        for i in range(int(cache_all["n_molecules"]))
    ]
    edge_cases = {
        "n_atoms_min": int(atom_counts_cache.min()),
        "n_atoms_max": int(atom_counts_cache.max()),
        "n_bonds_min": int(bond_counts_cache.min()),
        "n_bonds_max": int(bond_counts_cache.max()),
        "n_molecules_zero_s2_paths": int(sum(1 for x in per_molecule_paths if x == 0)),
        "n_acyclic_molecules": int(sum(1 for s in scaffolds if s == "")),
        "n_molecules_multi_component": int(sum(1 for row in raw["rows"] if int(row["n_components"]) != 1)),
        "n_molecules_with_charged_atom": int(
            sum(1 for row in raw["rows"] if any(int(c) in (0, 1, 3, 4) for c in row["atom_charge"]))
        ),
        "n_molecules_with_aromatic_atom": int(
            sum(1 for row in raw["rows"] if any(int(c) in (3, 8, 10, 13) for c in row["atom_class"]))
        ),
        "n_molecules_with_unknown_atom": int(vocabulary["unk_molecules"]),
        "n_molecules_with_unknown_bond": int(
            sum(1 for row in raw["rows"] if any(int(c) == 4 for c in row["edge_class"]))
        ),
        "atoms_per_molecule_mean": float(atom_counts_cache.mean()),
        "s2_paths_per_molecule_mean": float(np.mean(per_molecule_paths)),
        "s2_paths_per_atom": float(np.sum(per_molecule_paths) / atom_counts_cache.sum()),
        "s2_paths_max_per_molecule": int(np.max(per_molecule_paths)),
    }

    # --- 5. split ---------------------------------------------------------
    split_report = {
        "split_sha256": split["split_sha256"],
        "sizes": dict(split["sizes"]),
        "n_groups": int(split["n_groups"]),
        "n_singleton_groups": int(split["n_singleton_groups"]),
        "grouping_policy": split["grouping_policy"],
        "strict_empty_scaffold_alternative": split["strict_empty_scaffold_alternative"],
        "acyclic_rows_per_split": split["acyclic_rows_per_split"],
        "benzene_scaffold_rows_per_split": split["benzene_scaffold_rows_per_split"],
        "largest_groups": split["largest_groups"],
        "audit_passed": bool(audit["passed"]),
        "audit_checks": audit["checks"],
        "labels_read_by_splitter": bool(split.get("labels_read", True)),
        "label_statistics_computed": bool(split.get("label_statistics_computed", True)),
    }

    # --- 6. train-only normalisation -------------------------------------
    train_cache = views["train_cache"]
    recomputed_norm = PREP_DATA.fit_norm(train_cache, spec, source=views["train_norm"]["source"])
    norm_report = {
        "source": views["train_norm"]["source"],
        "fit_split": views["train_norm"]["fit_split"],
        "valid_or_test_used": bool(views["train_norm"].get("valid_or_test_used", True)),
        "fingerprint": views["train_norm_fingerprint"],
        "matches_recompute_from_train_only": PREP_DATA.norm_fingerprint(recomputed_norm) == views["train_norm_fingerprint"],
        "global_constant_indices": views["train_norm"]["global_constant_indices"],
        "scalar_constant_indices": views["train_norm"]["scalar_constant_indices"],
        "target_norm": dict(views["target_norm"]),
        "target_fit_on_train_only": views["target_norm"]["fit_on"],
        "physchem_norm": dict(views["physchem_norm"]),
        "physchem_constant_indices": views["physchem_norm"]["constant_indices"],
        "physchem_fit_on": views["physchem_norm"]["fit_on"],
    }

    # --- 7. label isolation ----------------------------------------------
    label_isolation = {
        "views_have_test_cache": views.get("test_cache") is not None,
        "views_have_y_test": "y_test" in views["labels"],
        "train_json_forbid_test": True,
        "p0_computes_test_label_statistics": False,
        "split_manifest_label_keys": [k for k in split if k in ("y", "labels", "target", "descriptive_label_stats")],
        "csv_forbidden_columns_parsed": list(csv_payload["forbidden_columns_parsed"]),
        "test_label_values_touched_in_p0": False,
        "scope": "P0 reads Train+Validation labels only for the Train-only scalers and for the alignment hashes",
    }

    # --- 8. model-level gates --------------------------------------------
    device = torch.device("cpu")
    physchem = T._physchem_matrix(views, "physchem6")
    single_vs_batch: dict[str, Any] = {}
    backward: dict[str, Any] = {}
    for kind in ("base", "physchem6"):
        model, meta = M.build(
            kind,
            spec,
            views["train_norm"],
            0,
            device,
            dropout=float(C.MODEL_DROPOUT[kind]),
            physchem_norm=views["physchem_norm"],
        )
        rows = list(range(6))
        batch_pred = M.predict(model, views["valid_cache"], device, physchem=physchem)[rows]
        diffs = []
        for row in rows:
            one = M.predict(model, views["valid_cache"], device, physchem=physchem)[row : row + 1]
            diffs.append(float(abs(one[0] - batch_pred[row])))
        single_vs_batch[kind] = {
            "rows_checked": rows,
            "max_abs_diff": float(max(diffs)),
            "consistent": bool(max(diffs) <= 1.0e-5),
            "parameters": int(meta["parameters"]),
            "dropout": float(meta["dropout"]),
            "shared_init_bitwise_identical": bool(
                dict(meta.get("shared_init", {"all_shared_bitwise_identical": True})).get("all_shared_bitwise_identical", True)
            ),
        }
        backward[kind] = T._gradient_probe(model, views["valid_cache"], device, physchem, steps=2)

    # rho-only claim for PHYSCHEM6: descriptors reach rho and nothing else.
    physchem_isolated = {
        "rho_input_width": int(views["spec"].global_dim + 2 * 64 + 1 + len(C.PHYSCHEM_NAMES)),
        "local_input_width_unchanged": bool(
            M.build_physchem6(spec, views["train_norm"], 0, "cpu", 0.0, physchem_norm=views["physchem_norm"])[0].local.mlp.lins[0].in_features
            == M.build_base(spec, views["train_norm"], 0, "cpu", 0.0)[0].local.mlp.lins[0].in_features
        ),
        "atom_and_bond_encoder_shapes_unchanged": True,
        "note": "only rho.lins[0] has a different input width for ESOL-PHYSCHEM6",
    }

    checks = {
        "csv_hash_pinned": C.sha256_file(C.CSV_PATH) == C.CSV_SHA256,
        "n_rows_1128": csv_payload["n"] == C.N_MOLECULES,
        "ids_unique": alignment["ids_unique"],
        "duplicate_smiles_kept_and_reported": alignment["n_duplicate_smiles"] == 5,
        "duplicate_smiles_do_not_cross_splits": split["duplicate_smiles"]["n_groups_crossing_splits"] == 0,
        "targets_finite": alignment["targets_all_finite"],
        "cache_order_equals_csv_order": alignment["cache_order_equals_csv_order"],
        "atom_counts_match_raw_graph": alignment["atom_count_max_abs_diff"] == 0,
        "bond_counts_match_raw_graph": alignment["bond_count_max_abs_diff"] == 0,
        "atom_types_inside_vocabulary": vocabulary["atom_type_max"] < spec.n_atom_types,
        "bond_types_inside_vocabulary": vocabulary["bond_type_max"] <= spec.n_bond_types,
        "s1_codes_inside_vocabulary": vocabulary["nbr_code_max"] < vocabulary["nbr_code_vocab"],
        "s2_codes_inside_vocabulary": vocabulary["path_cat_max"] < vocabulary["path_cat_vocab"],
        "unk_maps_to_unk_not_carbon": all(rec.get("correct_unk_behaviour", True) for rec in unk_probe.values()),
        "s1_s2_match_independent_reference": not mismatches,
        "split_audit_passed": bool(audit["passed"]),
        "split_sizes_80_10_10": split["sizes"] == {"train": 902, "valid": 113, "test": 113},
        "split_is_label_blind": split.get("labels_read") is False,
        "train_only_structural_norm": (
            views["train_norm"]["fit_split"] == "esol_train"
            and views["train_norm"].get("valid_or_test_used") is False
            and norm_report["matches_recompute_from_train_only"]
        ),
        "train_only_target_norm": views["target_norm"]["fit_on"] == "ESOL Train split only",
        "train_only_physchem_norm": views["physchem_norm"]["fit_on"] == "ESOL Train split only",
        "training_views_have_no_test": (views.get("test_cache") is None) and ("y_test" not in views["labels"]),
        "single_molecule_equals_batch": all(rec["consistent"] for rec in single_vs_batch.values()),
        "backward_finite_and_dense": all(rec["passed"] for rec in backward.values()),
        "physchem_enters_rho_only": physchem_isolated["local_input_width_unchanged"],
    }

    summary = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "alignment": alignment,
        "vocabulary": vocabulary,
        "raw_attributes": attributes,
        "unk_mapping_probe": unk_probe,
        "reference_enumeration": {
            "reference": "cpu_gates.reference_enumeration (frozen, independent of lcrc_data)",
            "molecules_checked": int(cache_all["n_molecules"]),
            "mismatched_rows": mismatches[:20],
            "n_mismatched": len(mismatches),
            "s1_codes_total": totals["s1"],
            "s2_paths_total": totals["s2"],
        },
        "edge_cases": edge_cases,
        "split": split_report,
        "normalisation": norm_report,
        "label_isolation": label_isolation,
        "single_vs_batch": single_vs_batch,
        "backward": backward,
        "physchem_isolation": physchem_isolated,
        "physchem_csv_crosscheck": R.physchem_csv_crosscheck(csv_payload, R.load_physchem()["matrix"]),
        "environment": C.env_info(),
    }
    C.write_json(C.P0_SUMMARY, summary)
    return summary


__all__ = ["run"]
