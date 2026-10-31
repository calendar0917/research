"""Validation-only analysis for BASE vs CONTEXT.

Everything here is descriptive.  Inputs are the two frozen run artifacts
(checkpoint, train/valid predictions), the frozen Validation420 labels and the
frozen split SMILES.  No Test label, prediction, metric or per-molecule Test
information is read.

The subgroup definitions copy the frozen ``lcrc_lipo_random42_gap_audit_v1``
rules exactly:

* scaffold-seen: Bemis-Murcko scaffold canonical SMILES appears in Train3360;
  empty scaffold falls back to the molecule's canonical SMILES;
* heavy atom / TPSA / MolLogP quartiles: Validation420 sample 25/50/75
  percentiles, right-closed inner boundaries (a value equal to a boundary goes
  to the lower group).

The five tail rows are the row IDs pre-identified by that audit
``{1630, 2924, 822, 1153, 3634}``.  They are used for post-hoc description only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import x1_common as C
import x1_data as XD

import r42_metrics as MET  # noqa: E402


QUARTILE_LABELS = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]


# ---------------------------------------------------------------------------
# run-artifact verification
# ---------------------------------------------------------------------------


def _run_dir(arm: str) -> Path:
    return C.RUNS / C.ARM_RUN_KEY[arm]


def load_run(arm: str, split: Mapping[str, Any], labels) -> dict[str, Any]:
    run_dir = _run_dir(arm)
    payload = C.read_json(run_dir / "train.json")
    if payload.get("protocol_id") != C.PROTOCOL_ID or payload.get("arm") != arm:
        raise PermissionError(f"{arm}: train.json protocol/arm mismatch")
    checkpoint = run_dir / "checkpoint.pt"
    valid_npz = run_dir / "valid_predictions.npz"
    train_npz = run_dir / "train_predictions.npz"
    for path in (checkpoint, valid_npz):
        if not path.exists():
            raise FileNotFoundError(path)
    if C.sha256_file(checkpoint) != payload["checkpoint_sha256"]:
        raise PermissionError(f"{arm}: checkpoint hash mismatch")
    if C.sha256_file(valid_npz) != payload["predictions"]["valid_sha256"]:
        raise PermissionError(f"{arm}: valid prediction hash mismatch")
    if C.sha256_file(train_npz) != payload["predictions"]["train_sha256"]:
        raise PermissionError(f"{arm}: train prediction hash mismatch")
    with np.load(valid_npz) as archive:
        rows = np.asarray(archive["rows"], dtype=np.int64)
        pred = np.asarray(archive["pred"], dtype=np.float64)
        y_true = np.asarray(archive["y_true"], dtype=np.float64)
    with np.load(train_npz) as archive:
        train_rows = np.asarray(archive["rows"], dtype=np.int64)
        train_pred = np.asarray(archive["pred"], dtype=np.float64)
    expected_rows = np.asarray(split["rows"]["valid"], dtype=np.int64)
    expected_train_rows = np.asarray(split["rows"]["train"], dtype=np.int64)
    if not np.array_equal(rows, expected_rows):
        raise PermissionError(f"{arm}: Validation420 rows are not in frozen order")
    if not np.array_equal(train_rows, expected_train_rows):
        raise PermissionError(f"{arm}: Train3360 rows are not in frozen order")
    if not np.array_equal(y_true, np.asarray(labels.y_valid, dtype=np.float64)):
        raise PermissionError(f"{arm}: Validation labels drifted")
    metrics = {
        "train": MET.all_metrics(train_pred, np.asarray(labels.y_train, dtype=np.float64)),
        "valid": MET.all_metrics(pred, y_true),
    }
    return {
        "arm": arm,
        "run_key": C.ARM_RUN_KEY[arm],
        "run_dir": run_dir,
        "payload": payload,
        "rows": rows,
        "train_rows": train_rows,
        "pred": pred,
        "train_pred": train_pred,
        "y_true": y_true,
        "metrics": metrics,
        "reported": payload["metrics"],
        "parameters": int(payload["model_audit"]["parameters"]),
        "parameters_new": int(payload["model_audit"]["parameters_new"]),
    }


# ---------------------------------------------------------------------------
# subgroup helpers (frozen gap-audit definitions)
# ---------------------------------------------------------------------------


def assign_quartiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, [0.0, 0.25, 0.50, 0.75, 1.0])
    inner = edges[1:-1]
    group = np.searchsorted(inner, values, side="left")
    return np.clip(group, 0, 3).astype(np.int64), edges


def scaffold_and_fallback(mol) -> tuple[str, bool]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:  # RDKit can fail on unusual valence; fallback is defined.
        scaffold = None
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), True
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True), False


def descriptor_table(split: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    smiles = [str(split["smiles"][int(r)]) for r in rows]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(mol is None for mol in mols):
        raise ValueError("RDKit failed to parse a Validation420 SMILES")
    return {
        "smiles": smiles,
        "canonical": [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in mols],
        "scaffold": [scaffold_and_fallback(mol)[0] for mol in mols],
        "scaffold_fallback": [scaffold_and_fallback(mol)[1] for mol in mols],
        "heavy_atoms": np.asarray([mol.GetNumHeavyAtoms() for mol in mols], dtype=np.int64),
        "tpsa": np.asarray([Descriptors.TPSA(mol) for mol in mols], dtype=np.float64),
        "mol_logp": np.asarray([Descriptors.MolLogP(mol) for mol in mols], dtype=np.float64),
    }


def subgroup_records(
    *,
    base_sq: np.ndarray,
    context_sq: np.ndarray,
    base_abs: np.ndarray,
    context_abs: np.ndarray,
    net_sse_improvement: float,
    families: Mapping[str, Mapping[str, np.ndarray]],
    stable_n: int = 30,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    d = context_sq - base_sq  # negative = CONTEXT better
    for family, groups in families.items():
        for group, mask in groups.items():
            mask = np.asarray(mask, dtype=bool)
            n = int(mask.sum())
            if n == 0:
                raise ValueError(f"empty subgroup {family}/{group}")
            rmse_base = float(np.sqrt(np.mean(base_sq[mask])))
            rmse_context = float(np.sqrt(np.mean(context_sq[mask])))
            sse_base = float(np.sum(base_sq[mask]))
            sse_context = float(np.sum(context_sq[mask]))
            gain = float(rmse_base - rmse_context)
            sse_gain = float(-np.sum(d[mask]))  # positive = CONTEXT better
            records.append(
                {
                    "family": family,
                    "group": group,
                    "n": n,
                    "rmse_base": rmse_base,
                    "rmse_context": rmse_context,
                    "rmse_gain": gain,
                    "mae_base": float(np.mean(base_abs[mask])),
                    "mae_context": float(np.mean(context_abs[mask])),
                    "mae_change": float(np.mean(context_abs[mask]) - np.mean(base_abs[mask])),
                    "sse_base": sse_base,
                    "sse_context": sse_context,
                    "sse_gain": sse_gain,
                    "share_of_net_sse_improvement": (sse_gain / net_sse_improvement) if net_sse_improvement != 0.0 else None,
                    "n_context_better_abs": int(np.sum(context_abs[mask] < base_abs[mask])),
                    "n_base_better_abs": int(np.sum(context_abs[mask] > base_abs[mask])),
                    "n_tie_abs": int(np.sum(context_abs[mask] == base_abs[mask])),
                    "stability": "stable" if n >= stable_n else f"small_n_unstable_n<{stable_n}",
                }
            )
    return records


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def analyze() -> dict[str, Any]:
    cfg = C.load_config()
    split, labels = XD.load_frozen_split_and_labels()
    views = XD.build_lcrc_views(split, labels)
    base = load_run("base", split, labels)
    context = load_run("context", split, labels)
    if not np.array_equal(base["rows"], context["rows"]):
        raise PermissionError("BASE/CONTEXT Validation rows differ")

    y = base["y_true"]
    pred_base = base["pred"]
    pred_context = context["pred"]
    if not np.array_equal(y, context["y_true"]):
        raise PermissionError("BASE/CONTEXT Validation labels differ")

    base_sq = (pred_base - y) ** 2
    context_sq = (pred_context - y) ** 2
    base_abs = np.abs(pred_base - y)
    context_abs = np.abs(pred_context - y)
    sse_base = float(np.sum(base_sq))
    sse_context = float(np.sum(context_sq))
    net_sse_improvement = sse_base - sse_context  # positive = CONTEXT better
    rmse_base = float(np.sqrt(np.mean(base_sq)))
    rmse_context = float(np.sqrt(np.mean(context_sq)))
    mae_base = float(np.mean(base_abs))
    mae_context = float(np.mean(context_abs))
    gain = rmse_base - rmse_context

    improved = int(np.sum(context_abs < base_abs))
    worsened = int(np.sum(context_abs > base_abs))
    ties = int(np.sum(context_abs == base_abs))

    # ---- five pre-identified tail rows (post-hoc description only) --------
    row_to_index = {int(r): i for i, r in enumerate(base["rows"])}
    missing = [r for r in C.TAIL_ROWS if r not in row_to_index]
    if missing:
        raise ValueError(f"tail rows missing from Validation420: {missing}")
    tail_indices = np.asarray([row_to_index[r] for r in C.TAIL_ROWS], dtype=np.int64)
    tail_mask = np.zeros(y.size, dtype=bool)
    tail_mask[tail_indices] = True
    tail_rows_payload = []
    for row in C.TAIL_ROWS:
        i = row_to_index[row]
        tail_rows_payload.append(
            {
                "row_id": int(row),
                "y_logD": float(y[i]),
                "pred_base": float(pred_base[i]),
                "pred_context": float(pred_context[i]),
                "abs_error_base": float(base_abs[i]),
                "abs_error_context": float(context_abs[i]),
                "sq_error_base": float(base_sq[i]),
                "sq_error_context": float(context_sq[i]),
                "sse_improvement_context": float(base_sq[i] - context_sq[i]),
                "abs_error_improvement_context": float(base_abs[i] - context_abs[i]),
                "context_better": bool(context_abs[i] < base_abs[i]),
            }
        )
    keep = ~tail_mask
    rmse_base_without = float(np.sqrt(np.mean(base_sq[keep])))
    rmse_context_without = float(np.sqrt(np.mean(context_sq[keep])))
    gain_without = rmse_base_without - rmse_context_without
    tail_sse_gain = float(np.sum(base_sq[tail_indices] - context_sq[tail_indices]))
    tail_share = (tail_sse_gain / net_sse_improvement) if net_sse_improvement != 0.0 else None
    tail_concentrated = bool(
        net_sse_improvement > 0.0
        and (
            (tail_share is not None and tail_share >= C.TAIL_SHARE_THRESHOLD)
            or gain_without <= 0.0
        )
    )
    tail_audit = {
        "status": "POST_HOC_DESCRIPTIVE_ONLY",
        "source_audit": C.track_rel(C.GAP_AUDIT),
        "tail_rows_fixed_before_this_round": list(C.TAIL_ROWS),
        "definition": "row IDs pre-identified by lcrc_lipo_random42_gap_audit_v1; no new tail search",
        "rows": tail_rows_payload,
        "n_tail_context_better": int(sum(1 for r in tail_rows_payload if r["context_better"])),
        "sse_base_total": sse_base,
        "sse_context_total": sse_context,
        "net_sse_improvement_context_minus_base": net_sse_improvement,
        "tail_sse_improvement": tail_sse_gain,
        "tail_share_of_net_sse_improvement": tail_share,
        "rmse_base_without_tail": rmse_base_without,
        "rmse_context_without_tail": rmse_context_without,
        "gain_without_tail": float(gain_without),
        "gain_all": float(gain),
        "tail_rule": cfg["decision"]["tail_rule"],
        "tail_share_threshold": C.TAIL_SHARE_THRESHOLD,
        "tail_concentrated": tail_concentrated,
        "tail_label": "KNOWN_TAIL_CONCENTRATED" if tail_concentrated else "NOT_TAIL_CONCENTRATED",
    }

    # ---- fixed decision ---------------------------------------------------
    promising = bool(gain >= C.GAIN_THRESHOLD and (mae_context - mae_base) <= C.MAE_WORSENING_TOLERANCE)
    exploratory_decision = "PROMISING_EXPLORATORY" if promising else "NO_CLEAR_GAIN"
    decision = {
        "protocol_id": C.PROTOCOL_ID,
        "status": "POST_TRAINING_FIXED_RULE",
        "primary_metric": cfg["decision"]["primary_metric"],
        "gain_definition": cfg["decision"]["gain_definition"],
        "gain": float(gain),
        "gain_threshold": C.GAIN_THRESHOLD,
        "mae_base": mae_base,
        "mae_context": mae_context,
        "mae_change": float(mae_context - mae_base),
        "mae_worsening_tolerance": C.MAE_WORSENING_TOLERANCE,
        "mae_tolerance_pass": bool((mae_context - mae_base) <= C.MAE_WORSENING_TOLERANCE),
        "exploratory_decision": exploratory_decision,
        "tail_status": tail_audit["tail_label"],
        "known_tail_concentrated": tail_concentrated,
        "outcome": f"{exploratory_decision} | {tail_audit['tail_label']}",
        "no_checkpoint_selection_by_validation": True,
        "scope": "single seed (0), random42 Train3360/Validation420, descriptive only",
    }

    # ---- subgroup tables --------------------------------------------------
    rows = [int(r) for r in base["rows"]]
    desc = descriptor_table(split, rows)
    train_smiles = [str(split["smiles"][int(r)]) for r in split["rows"]["train"]]
    from rdkit import Chem

    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    if any(mol is None for mol in train_mols):
        raise ValueError("RDKit failed to parse a Train3360 SMILES")
    train_scaffolds = {scaffold_and_fallback(mol)[0] for mol in train_mols}
    scaffold_seen = np.asarray([s in train_scaffolds for s in desc["scaffold"]], dtype=bool)
    size_group, size_edges = assign_quartiles(desc["heavy_atoms"])
    tpsa_group, tpsa_edges = assign_quartiles(desc["tpsa"])
    logp_group, logp_edges = assign_quartiles(desc["mol_logp"])
    families: dict[str, dict[str, np.ndarray]] = {
        "scaffold": {
            "seen": scaffold_seen,
            "unseen": ~scaffold_seen,
        },
        "heavy_atom_quartile": {label: size_group == i for i, label in enumerate(QUARTILE_LABELS)},
        "tpsa_quartile": {label: tpsa_group == i for i, label in enumerate(QUARTILE_LABELS)},
        "mol_logp_quartile": {label: logp_group == i for i, label in enumerate(QUARTILE_LABELS)},
    }
    records = subgroup_records(
        base_sq=base_sq,
        context_sq=context_sq,
        base_abs=base_abs,
        context_abs=context_abs,
        net_sse_improvement=net_sse_improvement,
        families=families,
    )
    all_record = subgroup_records(
        base_sq=base_sq,
        context_sq=context_sq,
        base_abs=base_abs,
        context_abs=context_abs,
        net_sse_improvement=net_sse_improvement,
        families={"all": {"all": np.ones(y.size, dtype=bool)}},
    )[0]
    subgroup_payload = {
        "status": "POST_HOC_DESCRIPTIVE_ONLY",
        "source_definitions": C.track_rel(C.GAP_AUDIT),
        "quartile_rule": (
            "Validation420 sample 25/50/75 percentiles; inner boundaries right-closed "
            "(values equal to a boundary go to the lower group)"
        ),
        "scaffold_rule": (
            "Bemis-Murcko scaffold canonical SMILES; empty scaffold falls back to the "
            "molecule's canonical SMILES; seen = scaffold string present in Train3360"
        ),
        "descriptor_rule": "RDKit from the frozen split SMILES: GetNumHeavyAtoms(), Descriptors.TPSA(), Descriptors.MolLogP()",
        "quartile_edges": {
            "heavy_atoms": [float(x) for x in size_edges],
            "tpsa": [float(x) for x in tpsa_edges],
            "mol_logp": [float(x) for x in logp_edges],
        },
        "counts": {
            "scaffold_seen": int(scaffold_seen.sum()),
            "scaffold_unseen": int((~scaffold_seen).sum()),
            "n_train_unique_scaffolds": len(train_scaffolds),
            "n_valid_unique_scaffolds": int(len(set(desc["scaffold"]))),
            "n_valid_empty_scaffold_fallback": int(sum(desc["scaffold_fallback"])),
        },
        "all": all_record,
        "records": records,
    }

    # ---- per-molecule table ----------------------------------------------
    per_molecule_path = C.PER_MOLECULE
    with open(per_molecule_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "row_id",
            "smiles",
            "y_logD",
            "pred_base",
            "pred_context",
            "abs_err_base",
            "abs_err_context",
            "sq_err_base",
            "sq_err_context",
            "sse_improvement_context",
            "abs_err_improvement_context",
            "context_better_abs",
            "is_fixed_tail_row",
            "scaffold_seen",
            "heavy_atoms",
            "tpsa",
            "mol_logp",
            "heavy_atom_quartile",
            "tpsa_quartile",
            "mol_logp_quartile",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(
                {
                    "row_id": int(row),
                    "smiles": desc["smiles"][i],
                    "y_logD": float(y[i]),
                    "pred_base": float(pred_base[i]),
                    "pred_context": float(pred_context[i]),
                    "abs_err_base": float(base_abs[i]),
                    "abs_err_context": float(context_abs[i]),
                    "sq_err_base": float(base_sq[i]),
                    "sq_err_context": float(context_sq[i]),
                    "sse_improvement_context": float(base_sq[i] - context_sq[i]),
                    "abs_err_improvement_context": float(base_abs[i] - context_abs[i]),
                    "context_better_abs": bool(context_abs[i] < base_abs[i]),
                    "is_fixed_tail_row": bool(tail_mask[i]),
                    "scaffold_seen": bool(scaffold_seen[i]),
                    "heavy_atoms": int(desc["heavy_atoms"][i]),
                    "tpsa": float(desc["tpsa"][i]),
                    "mol_logp": float(desc["mol_logp"][i]),
                    "heavy_atom_quartile": QUARTILE_LABELS[int(size_group[i])],
                    "tpsa_quartile": QUARTILE_LABELS[int(tpsa_group[i])],
                    "mol_logp_quartile": QUARTILE_LABELS[int(logp_group[i])],
                }
            )

    # ---- context usage ----------------------------------------------------
    base_diag = base["payload"].get("diagnostics") or {}
    context_diag = context["payload"].get("diagnostics") or {}
    usage = context_diag.get("context_usage") or {}
    if not usage:
        raise PermissionError("CONTEXT run has no context-usage diagnostics")
    gradient = usage.get("gradient_probe") or {}
    ablation = usage.get("w_out_zero_ablation_valid") or {}
    trained_delta = usage.get("trained_new_parameter_delta_from_init") or {}
    path_used = bool(
        usage.get("valid420", {}).get("change_l2_mean_over_hetero_atoms", 0.0) > 0.0
        and gradient.get("all_new_tensors_received_nonzero_gradient", False)
        and all(v > 0.0 for v in trained_delta.values())
        and ablation.get("mean_abs_prediction_change_vs_full_context", 0.0) > 0.0
    )
    context_usage = {
        "status": "POST_TRAINING_DESCRIPTIVE",
        "trained": {
            "all_new_parameter_tensors_moved_from_init": bool(all(v > 0.0 for v in trained_delta.values())),
            "max_abs_parameter_delta_from_init": trained_delta,
            "gradient_probe": gradient,
        },
        "used": {
            "valid420": usage.get("valid420"),
            "train3360": usage.get("train"),
            "w_out_zero_ablation_valid": ablation,
        },
        "context_path_trained_and_used": path_used,
        "interpretation_note": (
            "trained-and-used is a mechanical statement: the four new tensors moved from "
            "their initialisation, each received a non-zero gradient, the learned correction "
            "is non-zero, and zeroing W_out changes the Validation predictions.  It is not a "
            "claim that the correction improves generalisation."
        ),
    }

    # ---- metric table -----------------------------------------------------
    def metric_block(run: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "arm": run["arm"],
            "run_key": run["run_key"],
            "parameters": run["parameters"],
            "parameters_new": run["parameters_new"],
            "train": run["metrics"]["train"],
            "valid": run["metrics"]["valid"],
            "reported_train": run["reported"]["train"],
            "reported_valid": run["reported"]["valid"],
            "checkpoint_sha256": run["payload"]["checkpoint_sha256"],
            "valid_predictions_sha256": run["payload"]["predictions"]["valid_sha256"],
            "epochs_run": run["payload"]["recipe"]["epochs_run"],
            "optimizer_steps": run["payload"]["recipe"]["optimizer_steps"],
            "grad_clip_events": run["payload"]["anomalies"]["grad_clip_events"],
            "wall_seconds": run["payload"]["timing"]["train_wall_seconds"],
            "device": run["payload"]["device"],
            "cuda_device_name": run["payload"]["cuda_device_name"],
            "hostname": run["payload"]["hostname"],
            "torch": run["payload"]["torch"],
            "torch_cuda": run["payload"]["torch_cuda"],
            "git_commit": run["payload"]["git_commit"],
            "pool": run["payload"]["pool"],
        }

    comparison = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "POST_TRAINING_DESCRIPTIVE",
        "scope": "single seed 0, random42 Train3360 / Validation420, raw logD units",
        "models": {"base": metric_block(base), "context": metric_block(context)},
        "paired": {
            "n_valid": int(y.size),
            "sse_base": sse_base,
            "sse_context": sse_context,
            "sse_difference_base_minus_context": net_sse_improvement,
            "rmse_base": rmse_base,
            "rmse_context": rmse_context,
            "gain_rmse_base_minus_context": float(gain),
            "mae_base": mae_base,
            "mae_context": mae_context,
            "mae_change_context_minus_base": float(mae_context - mae_base),
            "r2_base": float(MET.r2(pred_base, y)),
            "r2_context": float(MET.r2(pred_context, y)),
            "n_context_better_abs": improved,
            "n_base_better_abs": worsened,
            "n_tie_abs": ties,
            "fraction_context_better_abs": improved / max(y.size, 1),
            "max_abs_prediction_difference": float(np.abs(pred_context - pred_base).max()),
            "mean_abs_prediction_difference": float(np.abs(pred_context - pred_base).mean()),
        },
        "decision": decision,
        "historical_reference": {
            "model": "LCRC-A-R3-scratch (random42 v1)",
            "valid_rmse": C.HISTORICAL_LCRC_VALID_RMSE,
            "note": (
                "Descriptive only. The historical random42 v1 LCRC training loop never applied "
                "the configured dropout, so its Validation RMSE 0.589007 is not a matched "
                "comparison for either arm of this round."
            ),
            "metrics_validation_file": C.track_rel(C.V1_METRICS_VALID),
            "metrics_validation_sha256": C.V1_METRICS_VALID_SHA256,
        },
        "estimated_metric": {
            "note": "hand-free recomputation of both arms from the frozen valid_predictions.npz",
            "reported_vs_recomputed_max_abs_diff": float(
                max(
                    abs(base["reported"]["valid"]["rmse"] - base["metrics"]["valid"]["rmse"]),
                    abs(base["reported"]["valid"]["mae"] - base["metrics"]["valid"]["mae"]),
                    abs(context["reported"]["valid"]["rmse"] - context["metrics"]["valid"]["rmse"]),
                    abs(context["reported"]["valid"]["mae"] - context["metrics"]["valid"]["mae"]),
                )
            ),
        },
        "role_coverage": {
            "train3360": XD.role_coverage(views["train_cache"], views["spec"]),
            "valid420": XD.role_coverage(views["valid_cache"], views["spec"]),
        },
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }

    C.write_json(C.COMPARISON, comparison)
    C.write_json(C.TAIL_AUDIT, tail_audit)
    C.write_json(C.SUBGROUPS, subgroup_payload)
    C.write_json(C.CONTEXT_USAGE, context_usage)
    C.write_json(C.DECISION, decision)
    C.write_json(
        C.METRICS_VALIDATION,
        {
            "protocol_id": C.PROTOCOL_ID,
            "created_utc": C.utcnow(),
            "scope": "Validation420 raw logD; recomputed from frozen valid_predictions.npz",
            "models": {arm: run["metrics"] for arm, run in (("base", base), ("context", context))},
        },
    )
    return {
        "comparison": comparison,
        "tail_audit": tail_audit,
        "subgroups": subgroup_payload,
        "context_usage": context_usage,
        "decision": decision,
        "per_molecule": C.track_rel(per_molecule_path),
        "per_molecule_sha256": C.sha256_file(per_molecule_path),
    }


__all__ = [
    "analyze",
    "load_run",
    "assign_quartiles",
    "scaffold_and_fallback",
    "subgroup_records",
]
