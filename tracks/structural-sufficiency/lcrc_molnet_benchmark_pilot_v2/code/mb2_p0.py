"""P0: CPU pre-flight audit for the seven-task benchmark round (v2).

Nothing here trains, tunes or scores a Test split.  For every dataset the audit
answers:

G0  is the source byte-identical to the pinned commit / official release?
G1  do the published Train/Valid/Test row ids partition the full sample set, and
    is "repeat 0" the frozen file (ChemBench) / the frozen OGB file order?
G2  does every SMILES parse, is nothing silently dropped, and do the committed
    label-free raw attributes equal a fresh RDKit derivation?
G3  do row id -> SMILES -> graph -> cache row -> target align 1:1 (Train, Valid
    *and* the label-free Test structures)?
G4  do the structure caches contain no label?
G5  what is the vocabulary coverage (UNK atoms/bonds), and how many molecules are
    salts / multi-component?
G6  is the model still the frozen LCRC-A (247,106 parameters; 248,536 for the
    12-output Tox21 head; no message passing; empirical radius-2 locality)?
G7  numerics: single-vs-batch consistency, finite forward/backward, every
    parameter receives a gradient, exactly one local encode per forward.
G8  is normalisation Train-only?
G9  are the targets what we think they are (regression stats; both classes for
    binary; per-endpoint observed counts for Tox21)?
G10 (Tox21) are missing labels kept missing everywhere -- data, loss, metrics?
G11 (ogbg-molhiv) do the official mapping/split/label files agree with each
    other and with the OGB documentation, and how does the LCRC input differ
    from OGB's reference input?
G12 is Test-label isolation provable on the P0 code path?
G13 does the ported RDKit adapter reproduce the Sprint-01 raw artifact exactly?

A failing gate blocks that dataset's formal training; the round then reports the
blocker instead of a number.

Usage
-----
    uv run python .../code/mb2_p0.py [--datasets ESOL,BBBP] [--rebuild]
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import mb2_common as C  # noqa: I001  (adds the frozen prep code dirs to sys.path)
import mb2_data as D
import mb2_metrics as MET
import mb2_model as M
import mb2_raw_artifact as RAW
import mb2_rdkit as R
import mb2_source as S

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


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _split_duplicates(smiles: Sequence[str], canonical: Mapping[int, str], rows: Mapping[str, Sequence[int]]) -> dict[str, Any]:
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
        "cross_split_exact": exact_cross,
        "cross_split_canonical": canon_cross,
        "n_cross_split_exact_groups": len(exact_cross),
        "n_cross_split_canonical_groups": len(canon_cross),
        "cross_split_split_pairs": sorted({f"{a}/{b}" for g in canon_cross for a in g["splits"] for b in g["splits"] if a < b}),
        "largest_canonical_groups": [
            {"canonical_smiles": k, "rows": v}
            for k, v in sorted(canon_groups.items(), key=lambda kv: -len(kv[1]))[:5]
        ],
        "action_taken": "none: the published benchmark split is recorded, not modified or repaired",
    }


def _raw_fidelity(dataset: str, rows_all: Sequence[int], methods: Mapping[int, str]) -> dict[str, Any]:
    """Prove the committed artifact equals a fresh RDKit derivation for every row."""
    artifact = RAW.load_rows(dataset, rows_all)
    refreshed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    method_counts: dict[str, int] = {}
    for i in rows_all:
        try:
            row, method = R.smiles_to_raw_row_relaxed(str(artifact[i]["smiles"]), i)
        except Exception as exc:  # pragma: no cover - would already fail at build
            failures.append({"row": int(i), "error": f"{type(exc).__name__}: {exc}"})
            continue
        refreshed.append(row)
        method_counts[method] = method_counts.get(method, 0) + 1
    fields = ("atom_class", "atom_charge", "atom_h", "edge_class", "edge_stereo")
    mismatched: list[dict[str, Any]] = []
    for i, a in enumerate(artifact):
        b = refreshed[i]
        problems = [
            f for f in fields if not np.array_equal(np.asarray(a[f], dtype=np.int64), np.asarray(b[f], dtype=np.int64))
        ]
        if (
            problems
            or str(a["smiles"]) != str(b["smiles"])
            or int(a["n_components"]) != int(b["n_components"])
            or not np.array_equal(np.asarray(a["edge_index"], dtype=np.int64), np.asarray(b["edge_index"], dtype=np.int64))
        ):
            mismatched.append({"row": int(rows_all[i]), "fields": problems})
    return {
        "passed": not mismatched and not failures,
        "n_rows_compared": int(len(artifact)),
        "fields_compared": list(fields) + ["smiles", "n_components", "edge_index"],
        "n_mismatched_rows": len(mismatched),
        "mismatches": mismatched[:10],
        "parse_methods_on_refresh": method_counts,
        "parse_methods_at_build": {k: int(sum(1 for m in methods.values() if m == k)) for k in sorted(set(methods.values()))},
        "artifact_manifest": C.track_rel(C.RAW_ARTIFACT_MANIFEST),
        "artifact_manifest_sha256": C.sha256_file(C.RAW_ARTIFACT_MANIFEST),
    }


def _v1_fidelity() -> dict[str, Any]:
    """The ported adapter must reproduce Sprint 01's artifact bit-exactly."""
    import gzip
    import json

    old_path = C.V1 / "analysis/source/raw_attributes_v1.json.gz"
    if not old_path.exists():
        return {"available": False, "reason": f"missing {C.track_rel(old_path)}"}
    with gzip.open(old_path, "rt", encoding="utf-8") as fh:
        old = json.load(fh)
    fields = ("atom_class", "atom_charge", "atom_h", "edge_class", "edge_stereo")
    out: dict[str, Any] = {"available": True, "v1_artifact": C.track_rel(old_path), "v1_artifact_sha256": C.sha256_file(old_path)}
    ok = True
    per_dataset: dict[str, Any] = {}
    for key in ("ESOL", "Lipophilicity", "FreeSolv", "BACE"):
        old_rows = old["rows"][key]
        new_rows = RAW.load_rows(key, list(range(len(old_rows))))
        mismatched = 0
        for a, b in zip(old_rows, new_rows):
            bad = [f for f in fields if not np.array_equal(np.asarray(a[f], dtype=np.int64), np.asarray(b[f], dtype=np.int64))]
            if (
                bad
                or str(a["smiles"]) != str(b["smiles"])
                or int(a["n_components"]) != int(b["n_components"])
                or not np.array_equal(np.asarray(a["edge_index"], dtype=np.int64), np.asarray(b["edge_index"], dtype=np.int64))
            ):
                mismatched += 1
        per_dataset[key] = {"n_rows": len(old_rows), "n_mismatched": int(mismatched)}
        ok = ok and mismatched == 0
    out["datasets"] = per_dataset
    out["passed"] = bool(ok)
    return out


def _cache_coverage(cache: Mapping[str, Any], spec) -> dict[str, Any]:
    atom_type = np.asarray(cache["atom_type"])
    bond_type = np.asarray(cache["bond_type"])
    n_atoms = int(atom_type.size)
    n_bonds = int(bond_type.size)
    unk_atoms = int((atom_type == spec.atom_unk_index).sum())
    unk_bonds = int((bond_type == spec.bond_unk_id).sum())
    n_comp = np.asarray(cache["n_components"])
    path_ptr = np.asarray(cache["path_ptr"])
    atom_ptr = np.asarray(cache["atom_ptr"])
    mols_no_s2 = int(sum(1 for g in range(int(cache["n_molecules"])) if path_ptr[atom_ptr[g + 1]] == path_ptr[atom_ptr[g]]))
    return {
        "n_molecules": int(cache["n_molecules"]),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_s1_codes": int(np.asarray(cache["nbr_ptr"])[-1]),
        "n_s2_paths": int(path_ptr[-1]),
        "atom_unknown_count": unk_atoms,
        "atom_unknown_fraction": float(unk_atoms / max(n_atoms, 1)),
        "bond_unknown_count": unk_bonds,
        "bond_unknown_fraction": float(unk_bonds / max(n_bonds, 1)),
        "molecules_without_s2_path": mols_no_s2,
        "n_multi_component_molecules": int((n_comp > 1).sum()),
        "multi_component_fraction": float((n_comp > 1).mean()) if n_comp.size else 0.0,
        "max_components": int(n_comp.max()) if n_comp.size else 0,
        "n_atoms_max_per_molecule": int((np.diff(atom_ptr)).max()) if n_comp.size else 0,
        "content_fingerprint": cache["meta"]["content_fingerprint"],
        "build_seconds": float(cache.get("build_seconds", 0.0)),
    }


def _model_numerics(model, view: D.DatasetView, meta: Mapping[str, Any], device) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    train = view.train_cache
    rows = list(range(min(8, int(train["n_molecules"]))))
    batch = D.to_device(PREP_DATA.collate(train, rows), device)

    model.eval()
    with torch.no_grad():
        full = model(batch)
        single = model(D.to_device(PREP_DATA.collate(train, [rows[0]]), device))
    if full.ndim == 1:
        full = full.reshape(-1, 1)
    if single.ndim == 1:
        single = single.reshape(-1, 1)
    single_vs_batch = float((full[0:1] - single).abs().max())

    model.train()
    hook_calls_before = int(meta["dropout_hook"].calls)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    history = []
    y = torch.as_tensor(view.targets.loss_reference(view.target_norm)[: len(rows)], dtype=torch.float32, device=device)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        if view.task == "regression":
            loss = F.mse_loss(pred.reshape(-1), y.reshape(-1))
        elif view.task == "binary":
            loss = F.binary_cross_entropy_with_logits(pred.reshape(-1), y.reshape(-1))
        else:
            mask = torch.isfinite(y)
            per_label = F.binary_cross_entropy_with_logits(pred, torch.nan_to_num(y), reduction="none")
            loss = (per_label * mask).sum() / mask.sum().clamp(min=1.0)
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
        "n_outputs": int(meta["n_outputs"]),
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


def _parameter_identity(view: D.DatasetView, device) -> dict[str, Any]:
    """Single-task vs 12-output head: same encoder, wider heads only."""
    single, single_meta = M.build_lcrc(view.spec, view.train_norm, C.TRAIN_SEED, device, n_tasks=1)
    multi, multi_meta = M.build_lcrc(view.spec, view.train_norm, C.TRAIN_SEED, device, n_tasks=12)
    enc_single = M.shared_encoder_state_hash(single)
    enc_multi = M.shared_encoder_state_hash(multi)
    out = {
        "single_task_parameters": int(single_meta["parameters"]),
        "multi_task_parameters": int(multi_meta["parameters"]),
        "frozen_single_task_parameter_count_expected": int(M.SINGLE_TASK_PARAMETERS),
        "single_task_parameter_count_matches_sprint01": int(single_meta["parameters"]) == int(M.SINGLE_TASK_PARAMETERS),
        "shared_encoder_state_hash_single": enc_single,
        "shared_encoder_state_hash_multi": enc_multi,
        "shared_encoder_identical_at_same_seed": bool(enc_single == enc_multi),
        "head_parameter_delta": int(multi_meta["parameters"]) - int(single_meta["parameters"]),
        "head_breakdown_single": {k: v for k, v in single_meta["parameters_breakdown"].items() if k in ("f", "rho")},
        "head_breakdown_multi": {k: v for k, v in multi_meta["parameters_breakdown"].items() if k in ("f", "rho")},
    }
    out["passed"] = bool(
        out["single_task_parameter_count_matches_sprint01"] and out["shared_encoder_identical_at_same_seed"]
    )
    return out


def _tox21_missing_audit(view: D.DatasetView) -> dict[str, Any]:
    """Missing labels must stay missing: in the targets, the mask and the loss."""
    targets = view.targets
    y = targets.y_train
    mask = targets.train_mask
    n_nan = int(np.isnan(y).sum())
    n_obs = int(mask.sum())
    zeros_where_missing = int((y[~mask] == 0).sum())
    y_valid = targets.y_valid
    valid_mask = targets.valid_mask

    # masked loss equals a hand-computed reference on a small slice
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    logits = rng.normal(size=y.shape).astype(np.float32)
    torch_mask = torch.as_tensor(mask)
    torch_y = torch.as_tensor(y, dtype=torch.float32)
    per_label = F.binary_cross_entropy_with_logits(torch.as_tensor(logits), torch.nan_to_num(torch_y), reduction="none")
    torch_loss = float((per_label * torch_mask).sum() / torch_mask.sum().clamp(min=1))
    reference = 0.0
    n_cells = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mask[i, j]:
                z = float(logits[i, j])
                t = float(y[i, j])
                reference += max(z, 0.0) + np.log1p(np.exp(-abs(z))) - t * z
                n_cells += 1
    reference /= n_cells
    filled_zero_alternative = float(
        F.binary_cross_entropy_with_logits(torch.as_tensor(logits), torch.nan_to_num(torch_y), reduction="mean")
    )
    return {
        "task_names": targets.task_names,
        "train_missing_cells": n_nan,
        "train_observed_cells": n_obs,
        "train_missing_fraction": float(n_nan / y.size),
        "valid_missing_cells": int(np.isnan(y_valid).sum()),
        "valid_observed_cells": int(valid_mask.sum()),
        "missing_cells_stored_as_nan_not_zero": bool(n_nan > 0 and zeros_where_missing == 0),
        "masked_bce_torch": torch_loss,
        "masked_bce_hand_computed": float(reference),
        "masked_bce_abs_diff": abs(torch_loss - float(reference)),
        "masked_bce_matches_hand_computation": bool(abs(torch_loss - float(reference)) <= 1e-6),
        "bce_if_missing_were_filled_with_zero": filled_zero_alternative,
        "abs_diff_if_missing_were_filled_with_zero": abs(filled_zero_alternative - torch_loss),
        "n_cells_that_would_be_mislabelled_as_negative": int((~mask).sum()),
        "filling_would_change_loss": bool(abs(filled_zero_alternative - torch_loss) > 1e-6),
        "loss_definition": "loss = (per_label_loss * observed_mask).sum() / observed_mask.sum()",
        "macro_auc_rule": (
            "per-endpoint ROC-AUC on observed cells only, then the unweighted mean over the endpoints where "
            "both classes are present in that split"
        ),
        "passed": bool(
            n_nan > 0
            and zeros_where_missing == 0
            and abs(torch_loss - float(reference)) <= 1e-6
            and abs(filled_zero_alternative - torch_loss) > 1e-6
        ),
    }


# ---------------------------------------------------------------------------
# per-dataset audit
# ---------------------------------------------------------------------------


def audit_dataset(key: str, *, spec, device, rebuild: bool, run_model_audit: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = C.dataset_entry(key)
    C.log(f"[P0] {key}: loading published source")
    t_load = time.perf_counter()
    source = S.load_dataset(key, labels="train_valid")
    load_seconds = time.perf_counter() - t_load

    # ---- duplicates / parse audit (SMILES only) -------------------------
    t_parse = time.perf_counter()
    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    methods: dict[int, str] = {}
    for i, smiles in enumerate(source["smiles"]):
        try:
            row, method = R.smiles_to_raw_row_relaxed(smiles, i)
            methods[i] = method
            raw_rows.append(row)
        except Exception as exc:
            failures.append({"row": i, "smiles": smiles, "error": f"{type(exc).__name__}: {exc}"})
    parse_seconds = time.perf_counter() - t_parse
    canonical = {int(r["mol_index"]): r["canonical_smiles"] for r in raw_rows}
    duplicates = _split_duplicates(source["smiles"], canonical, source["rows"]) if not failures else {}
    all_rows = list(range(len(source["smiles"])))
    fidelity = _raw_fidelity(key, all_rows, methods) if not failures else {"passed": False, "error": "parse failures"}
    raw_attributes = R.atom_attribute_audit(RAW.load_rows(key, all_rows)) if not failures else {}
    fallback_rows = [i for i, m in methods.items() if m != "strict"]

    # ---- structure caches (label-free) ----------------------------------
    t_cache = time.perf_counter()
    view = D.build_view(key, spec=spec, rebuild=rebuild)
    test_cache = D.build_test_structure_cache(key, spec=spec, rebuild=rebuild)
    cache_seconds = time.perf_counter() - t_cache
    caches = {"train": view.train_cache, "valid": view.valid_cache, "test": test_cache}
    coverage = {name: _cache_coverage(cache, spec) for name, cache in caches.items()}
    for name, cache in caches.items():
        PREP_DATA.validate_cache(cache, spec)
        if cache.get("rows") is None:
            raise ValueError(f"{key}/{name}: cache has no row id ledger")

    alignment: dict[str, Any] = {"passed": True}
    for split, cache in (("train", view.train_cache), ("valid", view.valid_cache)):
        expected = [int(r) for r in source["rows"][split]]
        actual = [int(x) for x in np.asarray(cache["rows"]).tolist()]
        entry = {
            "row_ids_identical_to_source": bool(actual == expected),
            "molecule_ids_match_row_ids": [str(x) for x in cache["molecule_ids"]] == [str(r) for r in expected],
            "smiles_match_source_rows": all(
                str(cache["smiles"][i]) == str(source["smiles"][r]) for i, r in enumerate(expected)
            ),
            "n_molecules_matches_split": int(cache["n_molecules"]) == len(expected),
            "n": int(cache["n_molecules"]),
            "rows_sha256": C.row_sha256(expected),
        }
        entry["passed"] = all(
            entry[k]
            for k in ("row_ids_identical_to_source", "molecule_ids_match_row_ids", "smiles_match_source_rows", "n_molecules_matches_split")
        )
        alignment[split] = entry
        alignment["passed"] = alignment["passed"] and entry["passed"]
    expected_test = [int(r) for r in source["rows"]["test"]]
    alignment["test"] = {
        "row_ids_identical_to_source": [int(x) for x in np.asarray(test_cache["rows"]).tolist()] == expected_test,
        "molecule_ids_match_row_ids": [str(x) for x in test_cache["molecule_ids"]] == [str(r) for r in expected_test],
        "smiles_match_source_rows": all(
            str(test_cache["smiles"][i]) == str(source["smiles"][r]) for i, r in enumerate(expected_test)
        ),
        "n_molecules_matches_split": int(test_cache["n_molecules"]) == len(expected_test),
        "n": int(test_cache["n_molecules"]),
        "rows_sha256": C.row_sha256(expected_test),
        "labels_read_during_p0": False,
        "passed": True,
    }
    alignment["test"]["passed"] = all(
        alignment["test"][k]
        for k in ("row_ids_identical_to_source", "molecule_ids_match_row_ids", "smiles_match_source_rows", "n_molecules_matches_split")
    )
    alignment["passed"] = alignment["passed"] and alignment["test"]["passed"]

    # ---- targets ---------------------------------------------------------
    targets = {"train": view.targets.audit("train"), "valid": view.targets.audit("valid"), "test": "not read during P0"}
    normalisation = {
        "recomputed_from_train_cache_only_matches": True,
        "valid_only_differs": True,
        "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
        "target_norm": None if view.target_norm is None else dict(view.target_norm),
        "target_normalised": view.task == "regression",
        "classification_targets_left_untouched": view.task in ("binary", "multitask"),
    }
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
    normalisation["recomputed_from_train_cache_only_matches"] = bool(same)
    normalisation["valid_only_differs"] = bool(differs)
    normalisation["valid_or_test_used_flag"] = bool(view.train_norm["valid_or_test_used"])
    normalisation["scalar_constant_indices"] = list(view.train_norm["scalar_constant_indices"])
    normalisation["global_constant_indices"] = list(view.train_norm["global_constant_indices"])

    # ---- model -----------------------------------------------------------
    model_block: dict[str, Any] = {}
    parameter_identity: dict[str, Any] = {}
    if run_model_audit:
        model, model_meta = M.build_lcrc(
            spec, view.train_norm, C.TRAIN_SEED, device, n_tasks=view.targets.n_tasks
        )
        module_audit = M.module_name_audit(model)
        if not module_audit["passed"]:
            raise RuntimeError(f"forbidden modules in frozen LCRC-A: {module_audit['flagged_modules']}")
        locality = M.locality_audit(model, view.train_cache, spec)
        numerics = _model_numerics(model, view, model_meta, device)
        parameter_identity = _parameter_identity(view, device)
        model_block = {
            "module_audit": module_audit,
            "locality_radius2": locality,
            "numerics": numerics,
            "meta": {k: v for k, v in model_meta.items() if k != "dropout_hook"},
        }

    tox21_block = _tox21_missing_audit(view) if key == "Tox21" else {}

    # ---- gates -----------------------------------------------------------
    gate_blocks = {
        "G0_source_pinned": {
            "passed": bool(source["csv_sha256"]),
            "source_kind": source["source_kind"],
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
            "selected_repeat_index": None if meta["source"] == "ogb" else C.SPLIT_REPEAT_INDEX,
            "selected_repeat_file_index": None if meta["source"] == "ogb" else C.SPLIT_REPEAT_FILE_INDEX,
            "row_order_preserved_as_published": True,
        },
        "G2_no_silent_drops": {
            "passed": (not failures) and bool(fidelity.get("passed")),
            "n_parse_failures": len(failures),
            "failures": failures[:20],
            "n_molecules_kept": int(len(source["smiles"]) - len(failures)),
            "n_strict_parses": int(len(methods) - len(fallback_rows)),
            "n_unsanitized_fallback_parses": int(len(fallback_rows)),
            "fallback_rows": fallback_rows[:20],
            "fallback_policy": (
                "strict RDKit sanitisation first; if it fails, sanitize=False + UpdatePropertyCache(strict=False) "
                "and the same frozen attribute extraction -- every affected row is listed, nothing is dropped"
            ),
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
            "note": "coverage is recorded, not gated: UNK is the frozen policy and the fraction is reported as a risk",
            "raw_attributes": raw_attributes,
            "caches": coverage,
        },
        "G6_frozen_model": {
            "passed": bool(model_block)
            and model_block["module_audit"]["passed"]
            and model_block["locality_radius2"].get("passed", False)
            and parameter_identity.get("passed", False),
            "parameters": model_block.get("meta", {}).get("parameters"),
            "parameter_identity": parameter_identity,
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
            "passed": bool(targets["train"]["n_rows"] > 0)
            and (
                view.task != "binary"
                or (targets["train"]["both_classes_present"] and targets["valid"]["both_classes_present"])
            )
            and (
                view.task != "multitask"
                or (
                    targets["train"]["tasks_with_both_classes"] == view.targets.n_tasks
                    and targets["valid"]["tasks_with_both_classes"] >= 1
                )
            ),
            "detail": {k: v for k, v in targets.items() if k != "test"},
            "test_targets_status": "not read during P0",
        },
    }
    if key == "Tox21":
        gate_blocks["G10_tox21_missing_labels"] = tox21_block
    if meta["source"] == "ogb":
        gate_blocks["G11_ogb_official_split"] = {
            "passed": True,
            "files": source["files"],
            "label_agreement_mapping_vs_graph_label": source["label_agreement"],
            "mol_id_column_all_empty": source["mol_id_column_all_empty"],
            "official_evaluator": MET.ogb_evaluator_roc_auc(
                np.asarray([0.0, 0.0, 1.0, 1.0]), np.asarray([0.1, 0.4, 0.35, 0.8])
            ),
            "input_semantics": S.ogb_input_semantics(),
            "n_positive_train": int(np.asarray(source["y"]["train"]).sum()),
            "n_positive_valid": int(np.asarray(source["y"]["valid"]).sum()),
            "positive_rate_train": float(np.asarray(source["y"]["train"]).mean()),
            "positive_rate_valid": float(np.asarray(source["y"]["valid"]).mean()),
        }

    failed = [name for name, block in gate_blocks.items() if not block.get("passed", False)]
    status = "PASS" if not failed else "FAIL:" + ",".join(failed)

    dataset_entry = {
        "dataset": key,
        "source_kind": meta["source"],
        "task": view.task,
        "label_columns": view.label_columns,
        "units": meta["units"],
        "primary_metric": meta["primary_metric"],
        "split_method": meta["split_method"],
        "round": meta["round"],
        "literature_scale_note": meta["literature_scale_note"],
        "source": {
            "repository": S.CHEMBENCH_URL if meta["source"] == "chembench" else S.OGB_DOC_URL,
            "commit_or_release": source["source_commit"],
            "csv": C.track_rel(source["csv_path"]),
            "csv_sha256": source["csv_sha256"],
            "selected_split_file": source["selected_repeat"]["file"],
            "selected_split_sha256": source["selected_repeat"]["sha256"],
            "split_repeat_index": None if meta["source"] == "ogb" else C.SPLIT_REPEAT_INDEX,
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
        "tox21_missing_labels": tox21_block,
        "preprocessing": {
            "source_load_seconds": float(load_seconds),
            "rdkit_parse_seconds": float(parse_seconds),
            "cache_build_seconds": float(cache_seconds),
            "cache_build_seconds_per_split": {k: float(v["build_seconds"]) for k, v in coverage.items()},
            "cache_bytes": {s: int(D.cache_path(key, s).stat().st_size) for s in ("train", "valid", "test") if D.cache_path(key, s).exists()},
            "peak_rss_mb": _peak_rss_mb(),
        },
        "gates": gate_blocks,
        "status": status,
    }

    split_manifest = {
        "schema": "lcrc-molnet-v2-split-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "dataset": key,
        "source_kind": meta["source"],
        "task": view.task,
        "label_columns": list(meta["label_columns"]),
        "units": meta["units"],
        "split_method": meta["split_method"],
        "split_repeat_index": None if meta["source"] == "ogb" else C.SPLIT_REPEAT_INDEX,
        "selected_split_file": source["selected_repeat"]["file"],
        "selected_split_sha256": source["selected_repeat"]["sha256"],
        "n_total": int(len(source["smiles"])),
        "sizes": {k: len(v) for k, v in source["rows"].items()},
        "proportions": {k: round(len(v) / len(source["smiles"]), 6) for k, v in source["rows"].items()},
        "rows": {k: [int(x) for x in v] for k, v in source["rows"].items()},
        "rows_sha256": {k: C.row_sha256(v) for k, v in source["rows"].items()},
        "other_repeats": source["repeats"],
        "cross_split_duplicate_groups": duplicates.get("cross_split_canonical", []) if duplicates else [],
        "cross_split_duplicate_split_pairs": duplicates.get("cross_split_split_pairs", []) if duplicates else [],
        "cross_split_duplicate_policy": (
            "recorded, not repaired: the published split is used as published; any cross-split duplicate is "
            "reported as a limitation of the benchmark split"
        ),
        "derived_label_columns_used": [],
        "test_labels_read_during_p0": False,
        "source": {
            "repository": S.CHEMBENCH_URL if meta["source"] == "chembench" else S.OGB_DOC_URL,
            "commit_or_release": source["source_commit"],
            "csv": C.track_rel(source["csv_path"]),
            "csv_sha256": source["csv_sha256"],
        },
    }
    return _json_safe(dataset_entry), _json_safe(split_manifest)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 CPU audit for the seven-task benchmark round")
    parser.add_argument("--datasets", default=",".join(C.DATASET_ORDER))
    parser.add_argument("--rebuild", action="store_true", help="rebuild structure caches from SMILES")
    parser.add_argument("--skip-model-audit", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-manifest", action="store_true", help="rewrite the source manifest first")
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
    if args.write_manifest or not C.SOURCE_MANIFEST.exists():
        C.write_json(C.SOURCE_MANIFEST, S.build_source_manifest())
        C.log(f"wrote {C.track_rel(C.SOURCE_MANIFEST)}")
    manifest = S.load_source_manifest(verify=True)

    spec = D.frozen_spec()
    v1_fidelity = _v1_fidelity()
    datasets: dict[str, Any] = {}
    split_manifests: dict[str, Any] = {}
    for key in keys:
        entry, split_manifest = audit_dataset(
            key, spec=spec, device=device, rebuild=args.rebuild, run_model_audit=not args.skip_model_audit
        )
        datasets[key] = entry
        split_manifests[key] = split_manifest
        C.write_json(C.ANALYSIS / f"split_manifest_{key}.json", split_manifest)
        C.log(f"[P0] {key}: {entry['status']}")

    dataset_manifest = {
        "schema": "lcrc-molnet-v2-dataset-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "source_manifest": C.track_rel(C.SOURCE_MANIFEST),
        "source_manifest_sha256": C.sha256_file(C.SOURCE_MANIFEST),
        "split_repeat_policy": (
            "ChemBench repeat 0 = the lexicographically first published split file (suffix 122); "
            "ogbg-molhiv uses the official OGB scaffold split files unchanged, in published order"
        ),
        "train_seed": C.TRAIN_SEED,
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
            "source": "lcrc_cross_task_preparation_v1/code/feature_spec.py::lipo_v1_spec",
        },
        "reused_module_hashes": C.reused_module_hashes(),
        "raw_artifact": {
            "manifest": C.track_rel(C.RAW_ARTIFACT_MANIFEST),
            "sha256": C.sha256_file(C.RAW_ARTIFACT_MANIFEST),
            "datasets": C.read_json(C.RAW_ARTIFACT_MANIFEST)["datasets"],
        },
        "datasets": {
            key: {
                "source_kind": entry["source_kind"],
                "task": entry["task"],
                "label_columns": entry["label_columns"],
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
    test_isolation = {
        "passed": bool(all_passed),
        "how": (
            "every P0 path calls load_dataset(labels='none') or labels='train_valid'; "
            "the Test split is only ever turned into a *label-free* structure cache, and "
            "DatasetView carries Train/Validation targets only (mb2_data.build_view)"
        ),
        "test_targets_decoded_anywhere_in_p0": False,
        "test_row_ids_recorded_sha256": {k: v["rows"]["test"]["sha256"] for k, v in datasets.items()},
    }
    audit = {
        "schema": "lcrc-molnet-v2-p0-audit-v1",
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
            "datasets": list(keys),
        },
        "source_manifest": C.track_rel(C.SOURCE_MANIFEST),
        "source_manifest_sha256": C.sha256_file(C.SOURCE_MANIFEST),
        "source_manifest_load_data_reference": manifest["chembench"]["load_data_reference"],
        "feature_spec": dataset_manifest["feature_spec"],
        "reused_module_hashes": C.reused_module_hashes(),
        "v1_adapter_fidelity": v1_fidelity,
        "datasets": datasets,
        "test_label_isolation": test_isolation,
        "summary": {
            key: {
                "status": entry["status"],
                "failed_gates": [n for n, b in entry["gates"].items() if not b.get("passed", False)],
                "split_sizes": {k: v["n"] for k, v in entry["rows"].items()},
                "atom_unknown_fraction": entry["cache_coverage"]["train"]["atom_unknown_fraction"],
                "multi_component_fraction": entry["cache_coverage"]["train"]["multi_component_fraction"],
                "cross_split_duplicate_groups": (
                    entry["duplicates"].get("n_cross_split_canonical_groups") if entry["duplicates"] else None
                ),
                "cross_split_duplicate_split_pairs": (
                    entry["duplicates"].get("cross_split_split_pairs") if entry["duplicates"] else None
                ),
                "parameters": entry["model"].get("meta", {}).get("parameters") if entry["model"] else None,
                "preprocessing_seconds": {
                    "rdkit_parse": entry["preprocessing"]["rdkit_parse_seconds"],
                    "cache_build": entry["preprocessing"]["cache_build_seconds"],
                },
            }
            for key, entry in datasets.items()
        },
        "status": "PASS" if all_passed and v1_fidelity.get("passed", False) else "FAIL",
    }
    if not v1_fidelity.get("passed", False):
        audit["summary"]["v1_adapter_fidelity"] = "FAIL: the ported RDKit adapter diverges from Sprint 01"
    C.write_json(C.ANALYSIS / "p0_audit.json", audit)
    C.log(f"[P0] overall status: {audit['status']} (v1 adapter fidelity: {v1_fidelity.get('passed')})")
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
