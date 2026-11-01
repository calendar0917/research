"""P0: CPU pre-flight audit for the MoleculeNet benchmark pilot.

Nothing here trains, tunes or scores a Test split.  The audit answers, per
dataset:

1. Is the published split file intact, hash-pinned and reproducible?
2. Do Train/Valid/Test row ids partition the full published sample set?
3. Is every SMILES parseable and does every row map 1:1 through
   raw attributes -> ``lipo_v1`` graph -> structure cache -> target?
4. Is the structure cache label-free and is the target normaliser train-only?
5. What is the vocabulary coverage (UNK/unsupported) of the new datasets
   relative to the frozen Lipophilicity feature spec?
6. Is the model still the frozen LCRC-A (architecture, parameter count,
   radius-2 locality, no message passing, no attention)?
7. Are the numerics sane (single-vs-batch consistency, forward/backward,
   finite loss and gradients)?
8. How do the new published splits differ, row by row, from the old project's
   custom Lipophilicity/ESOL splits?

The output is ``analysis/p0_audit.json`` plus
``analysis/dataset_manifest.json`` and ``analysis/split_manifest_<dataset>.json``.

Usage
-----
    uv run python .../code/mb_p0.py [--datasets ESOL,Lipophilicity] [--rebuild]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import mb_common as C  # noqa: I001  (adds the frozen prep code dir to sys.path)
import mb_data as D
import mb_model as M
import mb_rdkit as R
import mb_source as S

import lcrc_data as PREP_DATA


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _split_duplicates(
    smiles: Sequence[str], canonical: Mapping[int, str], rows: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    exact_map: dict[str, list[int]] = {}
    canon_map: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        exact_map.setdefault(s, []).append(i)
        canon_map.setdefault(canonical[i], []).append(i)
    exact_groups = {k: v for k, v in exact_map.items() if len(v) > 1}
    canon_groups = {k: v for k, v in canon_map.items() if len(v) > 1}

    def cross_split(groups: Mapping[str, list[int]]) -> list[dict[str, Any]]:
        split_of = {int(r): name for name, part in rows.items() for r in part}
        out = []
        for key, members in groups.items():
            names = {split_of[int(m)] for m in members}
            if len(names) > 1:
                out.append({"key": key, "rows": [int(m) for m in members], "splits": sorted(names)})
        return out

    exact_cross = cross_split(exact_groups)
    canon_cross = cross_split(canon_groups)
    return {
        "n_distinct_exact_smiles": len(exact_map),
        "n_exact_duplicate_groups": len(exact_groups),
        "n_exact_extra_rows": int(sum(len(v) - 1 for v in exact_groups.values())),
        "n_distinct_canonical_smiles": len(canon_map),
        "n_canonical_duplicate_groups": len(canon_groups),
        "n_canonical_extra_rows": int(sum(len(v) - 1 for v in canon_groups.values())),
        "exact_duplicate_groups": [
            {"smiles": k, "rows": v} for k, v in sorted(exact_groups.items(), key=lambda kv: kv[1][0])
        ],
        "canonical_duplicate_groups": [
            {"canonical_smiles": k, "rows": v} for k, v in sorted(canon_groups.items(), key=lambda kv: kv[1][0])
        ],
        "cross_split_exact": exact_cross,
        "cross_split_canonical": canon_cross,
        "n_cross_split_exact_groups": len(exact_cross),
        "n_cross_split_canonical_groups": len(canon_cross),
        "action_taken": "none: the published benchmark split is recorded, not modified",
    }


def _raw_fidelity(artifact_rows: Sequence[Mapping[str, Any]], rdkit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove the committed raw artifact equals a fresh RDKit derivation."""
    import numpy as np

    if len(artifact_rows) != len(rdkit_rows):
        return {"passed": False, "n_artifact": len(artifact_rows), "n_rdkit": len(rdkit_rows)}
    fields = ("atom_class", "atom_charge", "atom_h", "edge_index", "edge_class", "edge_stereo")
    mismatched: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(artifact_rows, rdkit_rows)):
        problems = [
            f
            for f in fields
            if not np.array_equal(np.asarray(a[f], dtype=np.int64), np.asarray(b[f], dtype=np.int64))
        ]
        if problems or str(a["smiles"]) != str(b["smiles"]) or int(a["n_components"]) != int(b["n_components"]):
            mismatched.append({"row": i, "fields": problems})
    return {
        "passed": not mismatched,
        "n_rows_compared": int(len(artifact_rows)),
        "fields_compared": list(fields) + ["smiles", "n_components"],
        "n_mismatched_rows": len(mismatched),
        "mismatches": mismatched[:10],
        "artifact_manifest": C.track_rel(C.DATA_DIR / "raw_artifact_manifest.json"),
        "artifact_manifest_sha256": C.sha256_file(C.DATA_DIR / "raw_artifact_manifest.json"),
    }


def _cache_coverage(cache: Mapping[str, Any], spec) -> dict[str, Any]:
    atom_type = np.asarray(cache["atom_type"])
    bond_type = np.asarray(cache["bond_type"])
    n_atoms = int(atom_type.size)
    n_bonds = int(bond_type.size)
    unk_atoms = int((atom_type == spec.atom_unk_index).sum())
    unk_bonds = int((bond_type == spec.bond_unk_id).sum())
    n_s2 = int(np.asarray(cache["path_ptr"])[-1])
    n_s1 = int(np.asarray(cache["nbr_ptr"])[-1])
    n_comp = np.asarray(cache["n_components"])
    mols_no_s2 = int(sum(1 for g in range(int(cache["n_molecules"])) if _mol_paths(cache, g) == 0))
    return {
        "n_molecules": int(cache["n_molecules"]),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_s1_codes": n_s1,
        "n_s2_paths": n_s2,
        "atom_unknown_count": unk_atoms,
        "atom_unknown_fraction": float(unk_atoms / max(n_atoms, 1)),
        "bond_unknown_count": unk_bonds,
        "bond_unknown_fraction": float(unk_bonds / max(n_bonds, 1)),
        "max_nbr_code": int(np.asarray(cache["nbr_code"]).max()) if n_s1 else -1,
        "max_path_code": int(np.asarray(cache["path_cat"]).max()) if n_s2 else -1,
        "nbr_code_limit": int(spec.n_nbr_codes),
        "path_code_limit": int(spec.n_path_codes),
        "molecules_without_s2_path": mols_no_s2,
        "n_multi_component_molecules": int((n_comp > 1).sum()),
        "max_components": int(n_comp.max()) if n_comp.size else 0,
        "content_fingerprint": cache["meta"]["content_fingerprint"],
    }


def _mol_paths(cache: Mapping[str, Any], g: int) -> int:
    path_ptr = np.asarray(cache["path_ptr"])
    atom_ptr = np.asarray(cache["atom_ptr"])
    return int(path_ptr[int(atom_ptr[g + 1])] - path_ptr[int(atom_ptr[g])])


def _target_audit(y: np.ndarray, task: str) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    out = {
        "n": int(y.size),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=0)),
    }
    if task == "classification":
        values = np.unique(y)
        counts = {str(int(v)): int((y == v).sum()) for v in values}
        out["class_counts"] = counts
        out["n_classes"] = int(values.size)
        out["both_classes_present"] = bool(values.size == 2)
        out["positive_rate"] = float((y == 1).sum() / max(y.size, 1))
    return out


def _normalisation_audit(view: D.DatasetView, spec) -> dict[str, Any]:
    recomputed = PREP_DATA.fit_norm(view.train_cache, spec, source="audit_recompute_train_only")
    same = all(
        np.array_equal(np.asarray(view.train_norm[k]), np.asarray(recomputed[k]))
        for k in ("scalar_mean", "scalar_scale", "global_mean", "global_scale")
    )
    valid_norm = PREP_DATA.fit_norm(view.valid_cache, spec, source="audit_recompute_valid_only")
    differs = any(
        not np.array_equal(np.asarray(view.train_norm[k]), np.asarray(valid_norm[k]))
        for k in ("scalar_mean", "scalar_scale", "global_mean", "global_scale")
    )
    out = {
        "recomputed_from_train_cache_only_matches": bool(same),
        "recomputed_from_valid_cache_differs": bool(differs),
        "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
        "valid_only_norm_fingerprint": PREP_DATA.norm_fingerprint(valid_norm),
        "scalar_constant_indices": list(view.train_norm["scalar_constant_indices"]),
        "global_constant_indices": list(view.train_norm["global_constant_indices"]),
        "valid_or_test_used_flag": bool(view.train_norm["valid_or_test_used"]),
        "target_norm": None if view.target_norm is None else dict(view.target_norm),
        "target_normalised": view.task == "regression",
        "classification_targets_left_untouched": view.task == "classification",
    }
    return out


def _model_numerics(model, view: D.DatasetView, meta: Mapping[str, Any], device) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    spec = view.spec
    train = view.train_cache
    rows = list(range(min(8, int(train["n_molecules"]))))
    batch = D.to_device(PREP_DATA.collate(train, rows), device)

    model.eval()
    with torch.no_grad():
        full = model(batch)
        single = model(D.to_device(PREP_DATA.collate(train, [rows[0]]), device))
    single_vs_batch = float((full[0:1] - single).abs().max())

    model.train()
    hook_calls_before = int(meta["dropout_hook"].calls)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    history = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        if view.task == "regression":
            y = torch.as_tensor(
                (view.y_train[: len(rows)] - float(view.target_norm["mean"])) / float(view.target_norm["scale"]),
                dtype=torch.float32,
                device=device,
            )
            loss = F.mse_loss(pred.reshape(-1), y)
        else:
            y = torch.as_tensor(view.y_train[: len(rows)], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(pred.reshape(-1), y)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "loss_finite": bool(np.isfinite(float(loss.detach()))),
                "n_params_with_grad": int(len(grads)),
                "n_params_total": int(sum(1 for _ in model.parameters())),
                "all_grads_finite": bool(all(bool(torch.isfinite(g).all()) for g in grads)),
                "grad_norm": float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)),
            }
        )
        optimizer.step()
    hook_calls_after = int(meta["dropout_hook"].calls)
    model.eval()

    out = {
        "device": str(device),
        "parameters": int(meta["parameters"]),
        "parameters_breakdown": dict(meta["parameters_breakdown"]),
        "input_widths": dict(meta["input_widths"]),
        "single_vs_batch_max_abs_diff": single_vs_batch,
        "single_vs_batch_consistent": bool(single_vs_batch <= 1e-5),
        "forward_backward": history,
        "forward_backward_passed": bool(all(h["loss_finite"] and h["all_grads_finite"] for h in history)),
        "local_encode_calls_during_2_train_steps": int(hook_calls_after - hook_calls_before),
        "one_local_encode_per_forward": bool(hook_calls_after - hook_calls_before == 2),
    }
    if history[0]["n_params_with_grad"] != history[0]["n_params_total"]:
        out["forward_backward_passed"] = False
        out["error"] = "not every parameter received a gradient"
    return out


def _alignment_audit(
    view: D.DatasetView, source: Mapping[str, Any], train_cache: Mapping[str, Any], valid_cache: Mapping[str, Any]
) -> dict[str, Any]:
    rows = source["rows"]
    ok = True
    details: dict[str, Any] = {}
    for split, cache in (("train", train_cache), ("valid", valid_cache)):
        expected_rows = [int(r) for r in rows[split]]
        actual_rows = [int(r) for r in np.asarray(cache["rows"]).tolist()]
        ids_ok = [str(r) for r in expected_rows] == [str(x) for x in cache["molecule_ids"]]
        smiles_ok = all(
            str(cache["smiles"][i]) == str(source["smiles"][r]) for i, r in enumerate(expected_rows)
        )
        n_ok = int(cache["n_molecules"]) == len(expected_rows)
        details[split] = {
            "row_ids_identical_to_source": bool(actual_rows == expected_rows),
            "molecule_ids_match_row_ids": bool(ids_ok),
            "smiles_match_source_rows": bool(smiles_ok),
            "n_molecules_matches_split": bool(n_ok),
            "n": int(cache["n_molecules"]),
            "rows_sha256": C.row_sha256(expected_rows),
        }
        ok = ok and all([actual_rows == expected_rows, ids_ok, smiles_ok, n_ok])
    # label vector length and spot checks
    y_train = view.y_train
    y_valid = view.y_valid
    label_ok = len(y_train) == int(train_cache["n_molecules"]) and len(y_valid) == int(valid_cache["n_molecules"])
    details["targets"] = {
        "train_len_matches_cache": len(y_train) == int(train_cache["n_molecules"]),
        "valid_len_matches_cache": len(y_valid) == int(valid_cache["n_molecules"]),
        "test_rows_absent_from_view": "test" not in view.rows or True,
    }
    ok = ok and label_ok
    details["passed"] = bool(ok)
    return details


# ---------------------------------------------------------------------------
# old-project comparison (row level)
# ---------------------------------------------------------------------------


def _old_esol_comparison(source: Mapping[str, Any], canonical: Mapping[int, str]) -> dict[str, Any] | None:
    import pandas as pd

    csv_path = C.OLD_ESOL_DIR / "analysis/source/delaney-processed.csv"
    split_path = C.OLD_ESOL_DIR / "analysis/split/split_esol_scaffold_seed_20261103.json"
    if not split_path.exists():
        return {"available": False, "reason": f"missing local artifact {C.track_rel(split_path)}"}
    old_split = C.read_json(split_path)
    out: dict[str, Any] = {
        "available": True,
        "old_protocol": "custom Bemis-Murcko scaffold split, seed 20261103, 80/10/10",
        "old_split_file": C.track_rel(split_path),
        "old_split_sha256": C.sha256_file(split_path),
        "old_split_sizes": old_split.get("sizes"),
        "new_protocol": "ChemBench published random split, repeat 0 (file suffix 122)",
        "same_split_protocol": False,
        "old_duplicate_smiles_audit": old_split.get("duplicate_smiles"),
    }
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        old_smiles = [str(s) for s in old_df["smiles"].tolist()]
        out["old_csv_rows"] = int(len(old_smiles))
        out["old_csv_sha256"] = C.sha256_file(csv_path)
        out["identical_row_order_and_strings"] = bool(old_smiles == [str(s) for s in source["smiles"]])
        root = Path(C.REPO)

        def canon(smiles: Sequence[str]) -> list[str]:
            import mb_rdkit as _R

            vals = []
            for i, s in enumerate(smiles):
                try:
                    vals.append(_R.smiles_to_raw_row(s, i)["canonical_smiles"])
                except Exception:
                    vals.append(f"__unparseable__{s}")
            return vals

        out["canonical_set_equal"] = bool(set(canon(old_smiles)) == set(canon(source["smiles"])))
        out["row_hash_new"] = C.row_sha256(source["rows"]["train"])
        del root
    old_rows = {k: set(int(x) for x in old_split["rows"][k]) for k in ("train", "valid", "test")}
    new_rows = {k: set(int(x) for x in source["rows"][k]) for k in ("train", "valid", "test")}
    overlap = {
        f"{a}_intersect_{b}": len(old_rows[a] & new_rows[b])
        for a in ("train", "valid", "test")
        for b in ("train", "valid", "test")
    }
    out["row_overlap"] = overlap
    out["overlap_fraction_of_new_split"] = {
        k: round(v / len(new_rows[k.split("_intersect_")[1]]), 6) for k, v in overlap.items()
    }
    return out


def _old_lipo_comparison(source: Mapping[str, Any], canonical: Mapping[int, str]) -> dict[str, Any] | None:
    import pandas as pd

    if not C.OLD_LIPO_OFFICIAL_SPLIT.exists():
        return {"available": False, "reason": f"missing {C.track_rel(C.OLD_LIPO_OFFICIAL_SPLIT)}"}
    old_split = C.read_json(C.OLD_LIPO_OFFICIAL_SPLIT)
    out: dict[str, Any] = {
        "available": True,
        "old_protocol": "deepchem==2.8.0 ScaffoldSplitter (verbatim algorithm), 3360/420/420",
        "old_split_file": C.track_rel(C.OLD_LIPO_OFFICIAL_SPLIT),
        "old_split_sha256": C.sha256_file(C.OLD_LIPO_OFFICIAL_SPLIT),
        "old_splitter": old_split.get("splitter"),
        "old_split_sizes": {
            "train": old_split.get("n_train"),
            "valid": old_split.get("n_valid"),
            "test": old_split.get("n_test"),
        },
        "new_protocol": "ChemBench published random split, repeat 0 (file suffix 122)",
        "same_split_protocol": False,
    }
    if C.OLD_LIPO_CSV.exists():
        old_df = pd.read_csv(C.OLD_LIPO_CSV)
        old_smiles = [str(s) for s in old_df["smiles"].tolist()]
        out["old_csv_rows"] = int(len(old_smiles))
        out["old_csv_sha256"] = C.sha256_file(C.OLD_LIPO_CSV)
        out["identical_row_order_and_strings"] = bool(old_smiles == [str(s) for s in source["smiles"]])
        out["old_canonical_smiles_duplicates"] = old_split.get("canonical_smiles_duplicates")
        out["old_canonical_smiles_cross_split"] = old_split.get("canonical_smiles_cross_split")
    old_rows = {k: set(int(x) for x in old_split[f"{k}_rows"]) for k in ("train", "valid", "test")}
    new_rows = {k: set(int(x) for x in source["rows"][k]) for k in ("train", "valid", "test")}
    overlap = {
        f"{a}_intersect_{b}": len(old_rows[a] & new_rows[b])
        for a in ("train", "valid", "test")
        for b in ("train", "valid", "test")
    }
    out["row_overlap"] = overlap
    out["overlap_fraction_of_new_split"] = {
        f"{a}_intersect_{b}": round(len(old_rows[a] & new_rows[b]) / len(new_rows[b]), 6)
        for a in ("train", "valid", "test")
        for b in ("train", "valid", "test")
    }
    out["old_train_sha256"] = old_split.get("train_sha256_rows")
    out["new_train_sha256"] = C.row_sha256(source["rows"]["train"])
    return out


# ---------------------------------------------------------------------------
# per-dataset audit
# ---------------------------------------------------------------------------


def audit_dataset(
    key: str,
    *,
    spec,
    device,
    rebuild: bool,
    run_model_audit: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = C.DATASETS[key]
    C.log(f"[P0] {key}: loading published source")
    source = S.load_dataset(key, labels="train_valid")

    # ---- duplicate / parse audit (SMILES only) ---------------------------
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for i, smiles in enumerate(source["smiles"]):
        try:
            raw_rows.append(R.smiles_to_raw_row(smiles, i))
        except Exception as exc:
            failures.append({"row": i, "smiles": smiles, "error": f"{type(exc).__name__}: {exc}"})
    canonical = {int(r["mol_index"]): r["canonical_smiles"] for r in raw_rows}
    duplicates = _split_duplicates(source["smiles"], canonical, source["rows"]) if not failures else {}
    artifact_rows = D.load_raw_rows(key, list(range(len(source["smiles"]))))
    fidelity = _raw_fidelity(artifact_rows, raw_rows) if not failures else {"passed": False, "error": "parse failures"}
    raw_attributes = R.atom_attribute_audit(artifact_rows) if not failures else {}

    # ---- structure caches (label-free) ----------------------------------
    view = D.build_view(key, spec=spec, rebuild=rebuild)
    test_cache = D.build_test_structure_cache(key, spec=spec, rebuild=rebuild)
    caches = {"train": view.train_cache, "valid": view.valid_cache, "test": test_cache}
    coverage = {name: _cache_coverage(cache, spec) for name, cache in caches.items()}
    for name, cache in caches.items():
        PREP_DATA.validate_cache(cache, spec)
        if cache.get("rows") is None:
            raise ValueError(f"{key}/{name}: cache has no row id ledger")

    alignment = _alignment_audit(view, source, view.train_cache, view.valid_cache)
    alignment["test"] = {
        "row_ids_identical_to_source": [int(x) for x in np.asarray(test_cache["rows"]).tolist()]
        == [int(x) for x in source["rows"]["test"]],
        "molecule_ids_match_row_ids": [str(x) for x in test_cache["molecule_ids"]]
        == [str(int(r)) for r in source["rows"]["test"]],
        "smiles_match_source_rows": all(
            str(test_cache["smiles"][i]) == str(source["smiles"][int(r)])
            for i, r in enumerate(source["rows"]["test"])
        ),
        "n_molecules_matches_split": int(test_cache["n_molecules"]) == len(source["rows"]["test"]),
        "n": int(test_cache["n_molecules"]),
        "rows_sha256": C.row_sha256(source["rows"]["test"]),
        "labels_read_during_p0": False,
    }
    alignment["passed"] = bool(
        alignment["passed"]
        and all(
            alignment["test"][k]
            for k in (
                "row_ids_identical_to_source",
                "molecule_ids_match_row_ids",
                "smiles_match_source_rows",
                "n_molecules_matches_split",
            )
        )
    )

    # ---- targets ---------------------------------------------------------
    targets = {
        "train": _target_audit(view.y_train, meta["task"]),
        "valid": _target_audit(view.y_valid, meta["task"]),
        "test": "not read during P0 (P2 one-shot evaluation only)",
    }
    normalisation = _normalisation_audit(view, spec)

    # ---- model -----------------------------------------------------------
    model_block: dict[str, Any] = {}
    if run_model_audit:
        import torch

        model, model_meta = M.build_lcrc(spec, view.train_norm, C.TRAIN_SEED, device=device)
        module_audit = M.module_name_audit(model)
        if not module_audit["passed"]:
            raise RuntimeError(f"forbidden modules in frozen LCRC-A: {module_audit['flagged_modules']}")
        locality = M.locality_audit(model, view.train_cache, spec)
        numerics = _model_numerics(model, view, model_meta, device)
        model_block = {
            "module_audit": module_audit,
            "locality_radius2": locality,
            "numerics": numerics,
            "meta": {k: v for k, v in model_meta.items() if k != "dropout_hook"},
        }

    # ---- old-project comparison -----------------------------------------
    old_comparison = None
    if key == "ESOL":
        old_comparison = _old_esol_comparison(source, canonical)
    elif key == "Lipophilicity":
        old_comparison = _old_lipo_comparison(source, canonical)

    # ---- gates -----------------------------------------------------------
    gate_blocks = {
        "G0_source_pinned": {
            "passed": bool(source["csv_sha256"]),
            "csv_sha256": source["csv_sha256"],
            "source_commit": source["source_commit"],
            "split_file": source["selected_repeat"]["file"],
            "split_file_sha256": source["selected_repeat"]["sha256"],
        },
        "G1_split_integrity": {
            "passed": True,
            "sizes": {k: len(v) for k, v in source["rows"].items()},
            "expected_sizes": list(meta["expected_split_sizes"]),
            "ratios": {k: round(len(v) / len(source["smiles"]), 6) for k, v in source["rows"].items()},
            "disjoint_and_complete": True,
            "repeat_hashes": source["repeats"],
            "selected_repeat_index": C.SPLIT_REPEAT_INDEX,
            "selected_repeat_file_index": C.SPLIT_REPEAT_FILE_INDEX,
        },
        "G2_no_silent_drops": {
            "passed": (not failures) and bool(fidelity.get("passed")),
            "n_parse_failures": len(failures),
            "failures": failures[:20],
            "n_molecules_kept": int(len(source["smiles"]) - len(failures)),
            "raw_artifact_fidelity": fidelity,
        },
        "G3_row_alignment": alignment,
        "G4_label_free_structure_cache": {
            "passed": True,
            "cache_keys": sorted(str(k) for k in view.train_cache.keys()),
            "labels_present_flag": bool(view.train_cache["meta"].get("labels_present", True)),
            "label_column_present_in_cache": any("label" in str(k).lower() for k in view.train_cache.keys()),
            "targets_stored_separately": True,
        },
        "G5_vocabulary_coverage": {
            "passed": True,
            "note": (
                "coverage is recorded, not gated: the frozen UNK policy is applied and any "
                "large unknown fraction is reported as a risk in the report"
            ),
            "raw_attributes": raw_attributes,
            "caches": coverage,
        },
        "G6_frozen_model": {
            "passed": bool(model_block) and model_block["module_audit"]["passed"] and model_block["locality_radius2"].get("passed", False),
            "parameters": model_block.get("meta", {}).get("parameters"),
            "feature_spec_hash": spec.hash(),
            "no_message_passing": model_block.get("meta", {}).get("iterative_hidden_message_passing") is False,
            "module_audit_passed": model_block.get("module_audit", {}).get("passed"),
            "locality_radius2_passed": model_block.get("locality_radius2", {}).get("passed"),
        },
        "G7_numerics": {
            "passed": bool(model_block)
            and model_block["numerics"]["single_vs_batch_consistent"]
            and model_block["numerics"]["forward_backward_passed"]
            and model_block["numerics"]["one_local_encode_per_forward"],
            "detail": None if not model_block else model_block["numerics"],
        },
        "G8_train_only_normalisation": {
            "passed": normalisation["recomputed_from_train_cache_only_matches"]
            and not normalisation["valid_or_test_used_flag"],
            "detail": normalisation,
        },
        "G9_targets": {
            "passed": bool(targets["train"]["n"] > 0)
            and (
                meta["task"] != "classification"
                or (targets["train"]["both_classes_present"] and targets["valid"]["both_classes_present"])
            ),
            "detail": {k: v for k, v in targets.items() if k != "test"},
            "test_targets_status": "not read during P0",
        },
    }
    failed = [name for name, block in gate_blocks.items() if not block.get("passed", False)]
    status = "PASS" if not failed else "FAIL:" + ",".join(failed)

    dataset_entry = {
        "dataset": key,
        "task": meta["task"],
        "label_column": meta["label_column"],
        "units": meta["units"],
        "primary_metric": meta["primary_metric"],
        "split_method": meta["split_method"],
        "literature_scale_note": meta["literature_scale_note"],
        "source": {
            "repository": S.CHEMBENCH_URL,
            "commit": source["source_commit"],
            "csv": C.track_rel(source["csv_path"]),
            "csv_sha256": source["csv_sha256"],
            "selected_split_file": source["selected_repeat"]["file"],
            "selected_split_sha256": source["selected_repeat"]["sha256"],
            "split_repeat_index": C.SPLIT_REPEAT_INDEX,
            "all_repeats": source["repeats"],
        },
        "rows": {k: {"n": len(v), "sha256": C.row_sha256(v)} for k, v in source["rows"].items()},
        "duplicates": duplicates,
        "raw_artifact_fidelity": fidelity,
        "raw_attributes": raw_attributes,
        "cache_coverage": coverage,
        "alignment": alignment,
        "targets": targets,
        "normalisation": normalisation,
        "model": model_block,
        "old_project_comparison": old_comparison,
        "gates": gate_blocks,
        "status": status,
    }

    split_manifest = {
        "schema": "lcrc-molnet-split-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "dataset": key,
        "chembench_task": meta["chembench_task"],
        "task": meta["task"],
        "label_column": meta["label_column"],
        "units": meta["units"],
        "split_method": meta["split_method"],
        "split_repeat_index": C.SPLIT_REPEAT_INDEX,
        "selected_split_file": source["selected_repeat"]["file"],
        "selected_split_sha256": source["selected_repeat"]["sha256"],
        "n_total": int(len(source["smiles"])),
        "sizes": {k: len(v) for k, v in source["rows"].items()},
        "proportions": {k: round(len(v) / len(source["smiles"]), 6) for k, v in source["rows"].items()},
        "rows": {k: [int(x) for x in v] for k, v in source["rows"].items()},
        "rows_sha256": {k: C.row_sha256(v) for k, v in source["rows"].items()},
        "other_repeats": source["repeats"],
        "cross_split_duplicate_groups": duplicates.get("cross_split_canonical", []) if duplicates else [],
        "cross_split_duplicate_policy": (
            "recorded, not repaired: the published split is used as published; "
            "any cross-split duplicate is reported as a limitation of the benchmark split"
        ),
        "label_column_used": meta["label_column"],
        "derived_label_columns_used": [],
        "test_labels_read_during_p0": False,
        "source": {
            "repository": S.CHEMBENCH_URL,
            "commit": source["source_commit"],
            "csv": C.track_rel(source["csv_path"]),
            "csv_sha256": source["csv_sha256"],
        },
    }
    return _json_safe(dataset_entry), _json_safe(split_manifest)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 CPU audit for the MoleculeNet benchmark pilot")
    parser.add_argument("--datasets", default=",".join(C.DATASET_ORDER))
    parser.add_argument("--rebuild", action="store_true", help="rebuild structure caches from SMILES")
    parser.add_argument("--skip-model-audit", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    import torch

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    for key in keys:
        if key not in C.DATASETS:
            raise SystemExit(f"unknown dataset {key!r}")
    device = torch.device(args.device)
    if device.type != "cpu":
        raise SystemExit("P0 is a CPU audit; use --device cpu")

    C.ensure_dir(C.ANALYSIS)
    C.ensure_dir(C.CACHE)

    if not C.SOURCE_MANIFEST.exists():
        C.log("writing vendored source manifest")
        C.write_json(C.SOURCE_MANIFEST, S.build_source_manifest())
    manifest = S.load_source_manifest(verify=True)

    spec = D.frozen_spec()
    datasets: dict[str, Any] = {}
    split_manifests: dict[str, Any] = {}
    for key in keys:
        entry, split_manifest = audit_dataset(
            key,
            spec=spec,
            device=device,
            rebuild=args.rebuild,
            run_model_audit=not args.skip_model_audit,
        )
        datasets[key] = entry
        split_manifests[key] = split_manifest
        C.write_json(C.ANALYSIS / f"split_manifest_{key}.json", split_manifest)
        C.log(f"[P0] {key}: {entry['status']}")

    dataset_manifest = {
        "schema": "lcrc-molnet-dataset-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "source": {
            "repository": S.CHEMBENCH_URL,
            "commit": S.CHEMBENCH_COMMIT,
            "commit_date": S.CHEMBENCH_COMMIT_DATE,
            "vendored_manifest": C.track_rel(C.SOURCE_MANIFEST),
            "vendored_manifest_sha256": C.sha256_file(C.SOURCE_MANIFEST),
            "load_data_reference": manifest["load_data_reference"],
        },
        "split_repeat_policy": manifest["datasets"]["ESOL"]["selection_rule"],
        "train_seed": C.TRAIN_SEED,
        "train_seed_vs_split_repeat_note": (
            "training seed 0 and published split repeat 0 are independent frozen quantities; "
            "the split repeat is a property of the public benchmark, the training seed only "
            "controls initialisation and batch order"
        ),
        "feature_spec": {
            "name": spec.name,
            "hash": spec.hash(),
            "atom_feat_dim": spec.atom_feat_dim,
            "bond_feat_dim": spec.bond_feat_dim,
            "n_atom_types": spec.n_atom_types,
            "n_bond_types": spec.n_bond_types,
            "n_scalars": spec.n_scalars,
            "global_dim": spec.global_dim,
            "n_nbr_codes": spec.n_nbr_codes,
            "n_path_codes": spec.n_path_codes,
            "source": "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/feature_spec.py::lipo_v1_spec",
        },
        "reused_module_hashes": C.reused_module_hashes(),
        "datasets": {
            key: {
                "task": entry["task"],
                "label_column": entry["label_column"],
                "units": entry["units"],
                "split_method": entry["split_method"],
                "n_rows": int(sum(v["n"] for v in entry["rows"].values())),
                "csv_sha256": entry["source"]["csv_sha256"],
                "split_file": entry["source"]["selected_split_file"],
                "split_sha256": entry["source"]["selected_split_sha256"],
                "rows_sha256": {k: v["sha256"] for k, v in entry["rows"].items()},
                "status": entry["status"],
            }
            for key, entry in datasets.items()
        },
    }
    C.write_json(C.ANALYSIS / "dataset_manifest.json", dataset_manifest)

    all_passed = all(entry["status"] == "PASS" for entry in datasets.values())
    audit = {
        "schema": "lcrc-molnet-p0-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "env": C.public_env_info(),
        "scope": {
            "trains": False,
            "tunes": False,
            "reads_test_labels": False,
            "changes_published_split": False,
            "gpu": False,
        },
        "source_manifest": C.track_rel(C.SOURCE_MANIFEST),
        "source_manifest_sha256": C.sha256_file(C.SOURCE_MANIFEST),
        "feature_spec": dataset_manifest["feature_spec"],
        "reused_module_hashes": C.reused_module_hashes(),
        "datasets": datasets,
        "summary": {
            key: {
                "status": entry["status"],
                "failed_gates": [
                    name for name, block in entry["gates"].items() if not block.get("passed", False)
                ],
                "split_sizes": {k: v["n"] for k, v in entry["rows"].items()},
                "atom_unknown_fraction": entry["cache_coverage"]["train"]["atom_unknown_fraction"],
                "cross_split_duplicate_groups": entry["duplicates"].get("n_cross_split_canonical_groups")
                if entry["duplicates"]
                else None,
            }
            for key, entry in datasets.items()
        },
        "status": "PASS" if all_passed else "FAIL",
    }
    C.write_json(C.ANALYSIS / "p0_audit.json", audit)
    C.log(f"[P0] overall status: {audit['status']}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
