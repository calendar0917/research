"""Per-dataset audit worker: identity, corrected nulls and response localisation.

The worker never selects anything.  For every molecule of the pilot's frozen
64-per-split sample it:

1. rebuilds the molecule from the frozen cache and re-verifies the occurrence map;
2. re-identifies the pilot's stored gradient, discovery score, Top-1 bond set and
   search-log entry (a mismatch stops the dataset);
3. re-identifies the pilot's stored real masked effect of that Top-1 (a mismatch
   stops the dataset — the brief forbids silently reporting a different number);
4. re-accounts the S2 dose as internal / boundary / affected and proves on the
   frozen function that ``n_s2_affected`` is exactly the set of occurrences whose
   gate weight goes from 1 to 0;
5. draws the corrected affected-matched null, re-evaluates every control with the
   frozen forward, and records the corrected effect ratio / advantage;
6. records the before/after per-root-node local output ``c_v = f(h_v)`` and the
   inside/outside response split.

Everything is chunked per molecule and written per split, so peak memory does not
grow with the number of splits or controls.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import au_common as C
import au_controls as K
import au_intervene as I
import au_s2 as S2

import as_data as D
import as_gate as G
import as_search as AS

import cr_data as SD          # noqa: E402  (frozen read-only reuse)
import cr_train as T          # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _configure_threads() -> int:
    """Bound BLAS/torch threads: this round only ever batches one small molecule."""
    import torch

    threads = int(os.environ.get("AU_TORCH_THREADS", "4"))
    threads = max(1, min(threads, 8))
    torch.set_num_threads(threads)
    return threads


def _device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _peak_rss_mb() -> float:
    import resource

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # linux reports KiB, macOS bytes
    return peak / (1024.0 * 1024.0) if peak > 1e7 else peak / 1024.0


def _max_abs_err(a: float, b: float) -> float:
    return float(abs(float(a) - float(b)))


def _torch_threads() -> int:
    import torch

    return int(torch.get_num_threads())


def results_dir(dataset: str) -> Path:
    return C.RESULTS_DIR / dataset


def split_dir(dataset: str, split: str) -> Path:
    return results_dir(dataset) / split


def load_arm(dataset: str, bundle, device: str):
    path = C.upstream_paths(dataset)["checkpoint"]
    arm, payload = T.load_arm_from_checkpoint(bundle, path, device=device)
    arm.eval()
    return arm, payload


# ---------------------------------------------------------------------------
# data / training / sample identity
# ---------------------------------------------------------------------------


def data_identity(dataset: str) -> dict[str, Any]:
    verified = SD.verify_frozen_files(dataset)
    payload = SD.load_payload(dataset)
    splits = {s: np.asarray(payload[C.SC.SPLIT_FILE[s]], dtype=np.int64) for s in C.SPLITS}
    ids = {s: splits[s].tolist() for s in C.SPLITS}
    pairwise: dict[str, Any] = {}
    for i, a in enumerate(C.SPLITS):
        for b in C.SPLITS[i + 1:]:
            inter = set(ids[a]) & set(ids[b])
            pairwise[f"{a}|{b}"] = {"n_intersection": len(inter), "empty": not inter}
    sizes = {s: int(splits[s].size) for s in C.SPLITS}
    union = np.concatenate([splits[s] for s in C.SPLITS])
    scaler = C.source_scaler(dataset)
    return {
        "dataset": dataset,
        "frozen_files": verified,
        "split_sizes": sizes,
        "expected_split_sizes": dict(C.EXPECTED_SPLIT_SIZES[dataset]),
        "split_sizes_match": sizes == dict(C.EXPECTED_SPLIT_SIZES[dataset]),
        "split_disjointness": pairwise,
        "splits_cover_all_rows": bool(
            union.size == int(payload["labels"].size)
            and np.array_equal(np.sort(union), np.arange(payload["labels"].size))
        ),
        "split_index_sha256": {s: C.sha256_array(splits[s]) for s in C.SPLITS},
        "target_scaler_train_only": {k: scaler[k] for k in ("mean", "std", "n", "ddof", "fit_split")},
        "pass": bool(
            sizes == dict(C.EXPECTED_SPLIT_SIZES[dataset])
            and all(v["empty"] for v in pairwise.values())
            and scaler["fit_split"] == "train"
            and int(scaler["n"]) == int(sizes["train"])
        ),
    }


def sample_identity(dataset: str, bundle) -> dict[str, Any]:
    """The 64 official row ids per split must be the pilot's, bit for bit."""
    pilot = C.read_json(C.upstream_paths(dataset)["samples"])
    out: dict[str, Any] = {"dataset": dataset, "splits": {}, "pass": True}
    for split in C.SPLITS:
        reproduced = C.select_samples(dataset, split, bundle.global_row_ids[split])
        reproduced["cache_rows"] = [
            int(np.where(bundle.global_row_ids[split] == rid)[0][0]) for rid in reproduced["row_ids"]
        ]
        stored = pilot["splits"][split]
        same_ids = list(map(int, stored["row_ids"])) == list(map(int, reproduced["row_ids"]))
        same_rows = list(map(int, stored["cache_rows"])) == list(map(int, reproduced["cache_rows"]))
        same_hash = str(stored["row_ids_sha256"]) == str(reproduced["row_ids_sha256"])
        out["splits"][split] = {
            "n_selected": int(len(reproduced["row_ids"])),
            "row_ids": [int(r) for r in reproduced["row_ids"]],
            "cache_rows": [int(r) for r in reproduced["cache_rows"]],
            "row_ids_match_pilot": bool(same_ids),
            "cache_rows_match_pilot": bool(same_rows),
            "row_ids_sha256": reproduced["row_ids_sha256"],
            "pilot_row_ids_sha256": stored["row_ids_sha256"],
            "row_ids_sha256_match": bool(same_hash),
            "reselect_reproduces_pilot": bool(same_ids and same_rows and same_hash),
        }
        out["pass"] = bool(out["pass"] and out["splits"][split]["reselect_reproduces_pilot"])
    out["rule"] = "ascending sha256(seed|dataset|split|row_id), ties by ascending row_id; first 64"
    return out


def training_identity(dataset: str) -> dict[str, Any]:
    paths = C.upstream_paths(dataset)
    ckpt, traj_path = paths["checkpoint"], paths["trajectory"]
    out: dict[str, Any] = {
        "dataset": dataset,
        "retrained": False,
        "reused_from": C.relpath(C.SOURCE_ROUND_RESULTS / dataset / "A"),
        "paths": {
            "checkpoint_best_val.pt": C.relpath(ckpt),
            "trajectory.json": C.relpath(traj_path),
            "pilot_run_manifest.json": C.relpath(C.UPSTREAM_ANALYSIS / "run_manifest.json"),
        },
        "sha256": {
            "checkpoint_best_val.pt": C.sha256_file(ckpt),
            "trajectory.json": C.sha256_file(traj_path),
        },
        "expected_sha256": {
            "checkpoint_best_val.pt": C.EXPECTED_CHECKPOINT_SHA256[dataset],
            "trajectory.json": C.EXPECTED_TRAJECTORY_SHA256[dataset],
        },
    }
    traj = C.read_json(traj_path)
    curves = traj["curves"]
    best_epoch = None
    best_val = None
    for rec in curves:
        epoch = int(rec["epoch"])
        if epoch == 0:
            continue
        val = float(rec["valid_rmse_raw"])
        if best_val is None or val < best_val:
            best_val, best_epoch = val, epoch
    out["trajectory"] = {
        "epochs": int(traj["epochs"]),
        "n_curve_records": int(len(curves)),
        "test_eval_count": int(traj["test_eval_count"]),
        "selected_epoch": int(traj["selected"]["epoch"]),
        "selected_valid_rmse_raw": float(traj["selected"]["valid_rmse_raw"]),
        "selected_test_rmse_raw": float(traj["selected"]["test_rmse_raw"]),
        "per_epoch_split_coverage": {
            "all_three_splits": bool(all(all(f"{s}_rmse_raw" in rec for s in C.SPLITS) for rec in curves)),
            "n_with_train": int(sum("train_rmse_raw" in rec for rec in curves)),
            "n_with_valid": int(sum("valid_rmse_raw" in rec for rec in curves)),
            "n_with_test": int(sum("test_rmse_raw" in rec for rec in curves)),
        },
        "argmin_valid_epoch_recomputed": best_epoch,
        "argmin_valid_rmse_recomputed": best_val,
        "selection_rule_matches": bool(best_epoch == int(traj["selected"]["epoch"])),
    }
    pilot_source = C.read_json(C.UPSTREAM_ANALYSIS / "training_source.json")["datasets"][dataset]
    out["pilot_recorded"] = {
        "checkpoint_sha256": pilot_source["sha256"]["checkpoint_best_val.pt"],
        "trajectory_sha256": pilot_source["sha256"]["trajectory.json"],
        "selected_epoch": pilot_source["trajectory"]["selected_epoch"],
    }
    out["pass"] = bool(
        out["sha256"]["checkpoint_best_val.pt"] == out["expected_sha256"]["checkpoint_best_val.pt"]
        and out["sha256"]["trajectory.json"] == out["expected_sha256"]["trajectory.json"]
        and out["sha256"]["checkpoint_best_val.pt"] == out["pilot_recorded"]["checkpoint_sha256"]
        and out["sha256"]["trajectory.json"] == out["pilot_recorded"]["trajectory_sha256"]
        and int(traj["epochs"]) == int(C.UP.TRAIN_EPOCHS)
        and out["trajectory"]["per_epoch_split_coverage"]["all_three_splits"]
        and out["trajectory"]["selection_rule_matches"]
    )
    return out


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------


def analyse_molecule(
    arm,
    cache: Mapping[str, Any],
    occ: D.SplitOcc,
    row: int,
    *,
    dataset: str,
    split: str,
    spec,
    sigma: float,
    mu: float,
    train_std: float,
    y_raw: float,
    row_id: int,
    pilot: Mapping[str, Any],
    search_log: Mapping[str, np.ndarray] | None,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Audit one pilot molecule.  Returns (record, ledger, atom rows, hard failures)."""
    failures: list[str] = []
    mol = G.SingleMolecule(cache, int(row), spec, strict=True)
    if int(mol.verification["n_failed"]):
        failures.append(f"occurrence mapping failed: {mol.verification['failed']}")

    grad = AS.gradient_pass(arm, D.collate_gated(cache, [int(row)], occ), sigma=sigma, mu=mu)
    g_new = np.asarray(grad["grad_transformed"][0][: mol.n_bonds], dtype=np.float64)
    g_old = np.asarray(pilot["gradient"]["gradient_raw"], dtype=np.float64)
    grad_err = float("inf")
    if g_old.size != g_new.size:
        failures.append(f"stored gradient has {g_old.size} entries for {g_new.size} bonds")
    else:
        tol = C.replay_tolerance(np.maximum(np.abs(g_new), np.abs(g_old)))
        grad_err = float(np.max(np.abs(g_new - g_old) / tol)) if g_old.size else 0.0
        if grad_err > 1.0:
            failures.append(f"stored gradient does not replay (scaled err {grad_err:.3f})")

    top1 = mol_rec_top1(pilot)
    if not top1:
        raise RuntimeError(f"{dataset}/{split}/row{row_id}: pilot record has no Top-1")
    top1_bonds = [int(b) for b in top1["bonds"]]
    if not (C.UP.SEARCH_MIN_K <= len(top1_bonds) <= C.UP.SEARCH_MAX_K):
        failures.append(f"pilot Top-1 has {len(top1_bonds)} bonds, outside the pilot search space")
    if len(set(top1_bonds)) != len(top1_bonds) or min(top1_bonds) < 0 or max(top1_bonds) >= mol.n_bonds:
        failures.append("pilot Top-1 bond set is not a valid set of physical bond ids")
    if not mol.atoms_connected(top1_bonds):
        failures.append("pilot Top-1 bond set is not connected")

    g_sum = float(sum(g_new[e] for e in top1_bonds))
    score_new = abs(g_sum) / np.sqrt(len(top1_bonds))
    score_pilot = float(top1["score_A"])
    score_err = abs(score_new - score_pilot)
    # the stored discovery score must be the pilot's own formula on its own stored
    # gradient (exact), and separately the replayed gradient must reproduce it
    g_sum_stored = float(sum(float(g_old[e]) for e in top1_bonds)) if g_old.size == g_new.size else float("nan")
    score_stored_recomputed = abs(g_sum_stored) / np.sqrt(len(top1_bonds))
    score_stored_err = abs(score_stored_recomputed - score_pilot)
    score_replay_tol = float(C.replay_tolerance(score_pilot))
    if not np.isfinite(score_stored_err) or score_stored_err > 1e-9:
        failures.append(
            f"stored discovery score is inconsistent with the stored gradient "
            f"(|d|={score_stored_err:.3e})"
        )
    if score_err > score_replay_tol:
        failures.append(f"discovery score does not replay (|d|={score_err:.3e})")
    in_log, log_score = _search_log_entry(search_log, row, split, top1_bonds)
    if search_log is not None and not in_log:
        failures.append("Top-1 bond set is absent from the pilot search log")
    if in_log and abs(float(log_score) - score_pilot) > 1e-9:
        failures.append("pilot search-log score disagrees with the stored discovery score")

    base_pred = float(grad["pred_raw"][0])
    base_err = _max_abs_err(base_pred, float(pilot["pred_raw_base"]))
    if base_err > float(C.replay_tolerance(float(pilot["pred_raw_base"]))):
        failures.append(f"BASE prediction does not replay (|d|={base_err:.3e})")

    accounting = S2.dose_accounting(mol, top1_bonds)
    rng = C.control_rng(dataset, split, int(row_id))
    controls = K.matched_random_controls(mol, top1_bonds, rng)

    variants = [I.variant("BASE", "baseline", [])]
    variants.append(I.variant("TOP1", "discovery_mask", top1_bonds))
    for rec in controls["matched"]:
        variants.append(I.variant(f"MATCHED_RANDOM_{rec['index']:02d}", "matched_random", rec["bonds"],
                                  index=int(rec["index"])))
    evaluated = I.run_variants(arm, cache, int(row), occ, variants, sigma=sigma, mu=mu, device=device)
    by_name = {rec["name"]: rec for rec in evaluated}
    base = by_name["BASE"]
    after = by_name["TOP1"]

    base_eval_err = _max_abs_err(base["pred_raw"], float(pilot["pred_raw_base"]))
    if base_eval_err > float(C.replay_tolerance(float(pilot["pred_raw_base"]))):
        failures.append(f"gated BASE differs from the pilot value (|d|={base_eval_err:.3e})")

    top1_effect = I.effect_record(base, after, train_std=train_std)
    pilot_effect = pilot["topk"][0]["effect"]
    pilot_check = _pilot_effect_check(top1_effect, pilot_effect)
    if not pilot_check["passed"]:
        failures.append(
            "recomputed Top-1 masked effect disagrees with the pilot record: "
            f"{pilot_check['failed_fields']}"
        )

    inside_atoms = np.zeros(mol.n_atoms, dtype=bool)
    inside_atoms[np.asarray(mol.bond_atoms(top1_bonds), dtype=np.int64)] = True
    localization = I.localization_record(
        base, after, inside_atoms=inside_atoms, accounting=accounting, train_std=train_std
    )
    if not localization["closure_local_sum_passed"]:
        failures.append(
            f"localisation closure failed: sum_v delta_c_v != delta_L "
            f"({localization['closure_local_sum_error']:.3e} > "
            f"{localization['closure_local_sum_tolerance']:.3e})"
        )
    if not localization["closure_total_passed"]:
        failures.append(
            f"readout closure failed: delta_L + delta_rho != delta_total "
            f"({localization['closure_total_error']:.3e} > "
            f"{localization['closure_total_tolerance']:.3e})"
        )
    if not top1_effect["closure_total_passed"]:
        failures.append("Top-1 delta_total != delta_L + delta_rho")

    # corrected null effects
    control_records = []
    for rec in controls["matched"]:
        name = f"MATCHED_RANDOM_{rec['index']:02d}"
        eff = I.effect_record(base, by_name[name], train_std=train_std)
        control_records.append({**rec, "effect": eff})
        if not eff["closure_total_passed"]:
            failures.append(f"{name}: delta_total != delta_L + delta_rho")
    abs_controls = [r["effect"]["abs_delta_total"] for r in control_records]
    med_control = C.median(abs_controls) if abs_controls else float("nan")
    top1_abs = top1_effect["abs_delta_total"]
    matched_effect = None
    if control_records and controls["status"] == "OK":
        matched_effect = {
            "n_matched": len(control_records),
            "median_abs_delta": med_control,
            "effect_ratio": top1_abs / (med_control + 1e-12),
            "effect_advantage": top1_abs - med_control,
            "ratio_gt1": bool(top1_abs > med_control),
            "matched_success": True,
        }

    # the S2 gate must remove exactly the accounted occurrences (real function)
    gate_checks = {
        "top1": S2.gate_drop_equivalence(
            arm, cache, int(row), occ, mol, top1_bonds, accounting, sigma=sigma, mu=mu
        )
    }
    for rec in control_records[:2]:
        acc_c = S2.dose_accounting(mol, rec["bonds"])
        gate_checks[f"matched_{rec['index']:02d}"] = S2.gate_drop_equivalence(
            arm, cache, int(row), occ, mol, rec["bonds"], acc_c, sigma=sigma, mu=mu
        )
    for key, chk in gate_checks.items():
        if not chk["passed"]:
            failures.append(f"S2 gate/drop equivalence failed for {key}")

    # pilot's cheap local-output heuristic: reused verbatim, never re-searched
    heur = pilot.get("local_heuristic") or {}
    heur_effect = heur.get("effect") or {}
    heuristic = None
    if heur_effect:
        heuristic = {
            "reused_from_pilot": True,
            "bonds": [int(b) for b in heur.get("bonds", [])],
            "n_bonds": int(heur.get("n_bonds", 0)),
            "n_s2_internal": int(heur.get("n_s2_occurrences", 0)),
            "abs_delta": float(heur_effect["abs_delta"]),
            "signed_delta": float(heur_effect["signed_delta"]),
            "delta_L": float(heur_effect["delta_L"]),
            "delta_rho": float(heur_effect["delta_P"]),
            "advantage_vs_top1": top1_abs - float(heur_effect["abs_delta"]),
            "top1_ge_heuristic": bool(top1_abs >= float(heur_effect["abs_delta"])),
        }

    old_controls = K.old_control_dose(mol, top1_bonds, [
        e["bonds"] for e in ((pilot.get("matched_random") or {}).get("effects") or {}).get("effects", []) or []
    ])
    old_summary = _summarise_old_controls(old_controls)

    record = {
        "dataset": dataset,
        "split": split,
        "row_id": int(row_id),
        "cache_row": int(row),
        "smiles": mol.smiles,
        "n_atoms": int(mol.n_atoms),
        "n_bonds": int(mol.n_bonds),
        "n_s2_total": int(mol.occ.n_path),
        "n_s1_total": int(mol.occ.n_nbr),
        "y_raw": float(y_raw),
        "train_std": float(train_std),
        "pilot": {
            "pred_raw_base": float(pilot["pred_raw_base"]),
            "local_sum_raw_base": float(pilot["local_sum_raw_base"]),
            "rho_raw_base": float(pilot["rho_raw_base"]),
            "top1_bonds": [int(b) for b in pilot["topk"][0]["bonds"]],
            "top1_score_A": score_pilot,
            "top1_n_s2_internal_pilot_field": int(pilot["topk"][0]["n_s2_occurrences"]),
            "top1_n_s2_either_pilot_field": int(pilot["topk"][0].get("n_s2_occurrences_either_bond", -1)),
            "gradient_n": int(g_old.size),
        },
        "replay": {
            "base_pred_raw": base_pred,
            "base_pred_abs_err": base_err,
            "base_pred_passed": bool(base_err <= float(C.replay_tolerance(float(pilot["pred_raw_base"])))),
            "gradient_scaled_err": grad_err if g_old.size == g_new.size else None,
            "gradient_passed": bool(g_old.size == g_new.size and grad_err <= 1.0),
            "score_A_recomputed": float(score_new),
            "score_A_abs_err": float(score_err),
            "score_A_replay_tolerance": score_replay_tol,
            "score_A_from_stored_gradient": float(score_stored_recomputed),
            "score_A_stored_abs_err": float(score_stored_err),
            "score_A_passed": bool(
                np.isfinite(score_stored_err)
                and score_stored_err <= 1e-9
                and score_err <= score_replay_tol
            ),
            "top1_in_pilot_search_log": bool(in_log),
            "pilot_search_log_score": float(log_score) if in_log else None,
            "search_log_passed": bool(search_log is None or (in_log and abs(float(log_score) - score_pilot) <= 1e-9)),
            "gated_base_pred_raw": float(base["pred_raw"]),
            "gated_base_abs_err": float(base_eval_err),
            "gated_base_passed": bool(base_eval_err <= float(C.replay_tolerance(float(pilot["pred_raw_base"])))),
        },
        "top1": {
            "bonds": top1_bonds,
            "atoms": mol.bond_atoms(top1_bonds),
            "n_bonds": int(accounting.n_bonds),
            "n_atoms_involved": int(accounting.n_atoms_inside),
            "score_A": float(score_pilot),
            "predicted_delta_from_g": float(g_sum),
            "dose": accounting.record(),
            "effect": top1_effect,
            "pilot_effect": {
                "signed_delta": float(pilot_effect["signed_delta"]),
                "abs_delta": float(pilot_effect["abs_delta"]),
                "delta_L": float(pilot_effect["delta_L"]),
                "delta_rho": float(pilot_effect["delta_P"]),
                "pred_raw_masked": float(pilot_effect["pred_raw"]),
            },
            "pilot_effect_check": pilot_check,
        },
        "controls": {
            "status": controls["status"],
            "rule": controls["rule"],
            "attempts": int(controls["attempts"]),
            "growth_exhausted": bool(controls["growth_exhausted"]),
            "n_matched": int(controls["n_matched"]),
            "n_candidates": int(controls["n_candidates"]),
            "n_rejected_affected": int(controls["n_rejected_affected"]),
            "n_duplicate_draws": int(controls["n_duplicate_draws"]),
            "matched": control_records,
            "summary": matched_effect,
        },
        "old_controls_dose": old_summary,
        "localization": localization,
        "heuristic": heuristic,
        "checks": {
            "occurrence_mapping_n_failed": int(mol.verification["n_failed"]),
            "occurrence_mapping_checks": int(mol.verification["n_checks"]),
            "n_variants_evaluated": int(len(evaluated)),
            "n_s2_weight_audits": len(gate_checks),
            "s2_gate_drop_all_passed": bool(all(c["passed"] for c in gate_checks.values())),
            "s2_weight_count_all_passed": bool(all(c["weight_audit"]["passed"] for c in gate_checks.values())),
        },
    }

    ledger_entries = [
        {"mol_index": int(row_id), **e} for e in controls["ledger"]
    ]
    atom_rows = _atom_rows(dataset, split, row_id, inside_atoms, base, after)
    return record, ledger_entries, atom_rows, failures


def mol_rec_top1(pilot: Mapping[str, Any]) -> Mapping[str, Any] | None:
    topk = pilot.get("topk") or []
    return topk[0] if topk else None


def _pilot_effect_check(new: Mapping[str, Any], pilot: Mapping[str, Any]) -> dict[str, Any]:
    pairs = {
        "signed_delta": (new["delta_total"], pilot["signed_delta"]),
        "delta_L": (new["delta_L"], pilot["delta_L"]),
        "delta_rho": (new["delta_rho"], pilot["delta_P"]),
        "pred_raw_masked": (new["pred_raw"], pilot["pred_raw"]),
    }
    fields: dict[str, Any] = {}
    failed: list[str] = []
    for name, (a, b) in pairs.items():
        err = _max_abs_err(a, b)
        tol = float(max(C.replay_tolerance(float(a)), C.replay_tolerance(float(b))))
        ok = bool(err <= tol)
        fields[name] = {"recomputed": float(a), "pilot": float(b), "abs_err": err,
                        "tolerance": tol, "passed": ok}
        if not ok:
            failed.append(name)
    return {"fields": fields, "failed_fields": failed, "passed": not failed}


def _search_log_entry(
    payload: Mapping[str, np.ndarray] | None,
    row: int,
    split: str,
    bonds: Sequence[int],
) -> tuple[bool, float]:
    if payload is None:
        return False, float("nan")
    molecule = int(row) * 4 + C.SPLITS.index(split)
    target = tuple(sorted(int(b) for b in bonds))
    for rec in AS.decode_search_log(payload, molecule):
        if tuple(sorted(int(b) for b in rec["bonds"])) == target:
            return True, float(rec["score"])
    return False, float("nan")


def _summarise_old_controls(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    internal = np.asarray([r["internal_rel_diff"] for r in rows], dtype=np.float64)
    affected = np.asarray([r["affected_rel_diff"] for r in rows], dtype=np.float64)
    diff = np.asarray([r["affected_minus_internal"] for r in rows], dtype=np.float64)
    outside = np.asarray([not bool(r["would_pass_affected_tolerance"]) for r in rows], dtype=np.float64)
    return {
        "n": int(len(rows)),
        "median_internal_rel_diff": float(np.median(internal)),
        "max_internal_rel_diff": float(np.max(internal)),
        "median_affected_rel_diff": float(np.median(affected)),
        "q75_affected_rel_diff": float(np.quantile(affected, 0.75)),
        "max_affected_rel_diff": float(np.max(affected)),
        "frac_outside_affected_tolerance": float(np.mean(outside)),
        "median_affected_minus_internal": float(np.median(diff)),
        "controls": [dict(r) for r in rows],
    }


def _atom_rows(
    dataset: str,
    split: str,
    row_id: int,
    inside_atoms: np.ndarray,
    base: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    delta = np.asarray(base["contrib_atom_raw"], dtype=np.float64) - np.asarray(
        after["contrib_atom_raw"], dtype=np.float64
    )
    rows = []
    for v in range(delta.size):
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "row_id": int(row_id),
                "atom": int(v),
                "inside": bool(inside_atoms[v]),
                "c_original_raw": float(np.asarray(base["contrib_atom_raw"])[v]),
                "c_masked_raw": float(np.asarray(after["contrib_atom_raw"])[v]),
                "delta_c_raw": float(delta[v]),
                "abs_delta_c_raw": float(abs(delta[v])),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# dataset run
# ---------------------------------------------------------------------------


def preflight_dataset(
    dataset: str,
    *,
    bundle=None,
    arm=None,
    occs: Mapping[str, D.SplitOcc] | None = None,
    samples: Mapping[str, Any] | None = None,
    smoke_molecules: int = 8,
) -> dict[str, Any]:
    """Identity gates plus a smoke pass of the full audit path (never aggregated)."""
    import torch

    _configure_threads()
    bundle = bundle or SD.build_bundle(dataset)
    device = _device()
    out: dict[str, Any] = {
        "dataset": dataset,
        "round_id": C.ROUND_ID,
        "upstream_round_id": C.UPSTREAM_ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "condition_name": C.CONDITION_NAME,
        "generated_at": C.utcnow(),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads": int(torch.get_num_threads()),
        "freeze_hash": C.freeze_hash(),
        "data": data_identity(dataset),
        "samples": samples or sample_identity(dataset, bundle),
    }
    source = training_identity(dataset)
    out["training_source"] = source
    arm = arm or load_arm(dataset, bundle, device)[0]
    sigma = float(bundle.scaler["std"])
    mu = float(bundle.scaler["mean"])
    state_before = C.sha256_state(arm.state_dict())
    occs = occs or {s: D.build_split_occ(bundle.caches[s], bundle.spec, dataset, s) for s in C.SPLITS}
    pilot_preflight = C.read_json(C.upstream_paths(dataset)["preflight"])

    # all-ones equivalence on each split (the gated path must still be arm A)
    equivalence = {}
    for split in C.SPLITS:
        rows = list(range(min(8, int(bundle.caches[split]["n_molecules"]))))
        gb = D.collate_gated(bundle.caches[split], rows, occs[split])
        eq = G.equivalence_check(arm, gb, sigma=sigma, mu=mu)
        equivalence[split] = eq
    parameters = {
        "n": int(sum(p.numel() for p in arm.parameters())),
        "trainable": int(sum(p.numel() for p in arm.parameters() if p.requires_grad)),
        "readout": int(sum(p.numel() for p in arm.f.parameters()) + sum(p.numel() for p in arm.rho.parameters())),
        "readout_expected": 15298,
        "total_expected": 228674,
    }
    parameters["matches_expected"] = bool(
        parameters["n"] == parameters["total_expected"] and parameters["readout"] == parameters["readout_expected"]
    )

    # smoke pass over the first N selected Valid molecules through the real path
    smoke_path = C.upstream_paths(dataset)
    pilot_valid = C.upstream_molecules(dataset, "valid")
    log = np.load(smoke_path["candidates"]) if smoke_path["candidates"].is_file() else None
    smoke_rows = samples["splits"]["valid"]["cache_rows"][: int(smoke_molecules)]
    smoke: list[dict[str, Any]] = []
    failures: list[str] = []
    train_std = float(bundle.scaler["std"])
    for i, row in enumerate(smoke_rows):
        pilot = pilot_valid[i]
        rec, _ledger, _atoms, fails = analyse_molecule(
            arm, bundle.caches["valid"], occs["valid"], int(row),
            dataset=dataset, split="valid", spec=bundle.spec,
            sigma=sigma, mu=mu, train_std=train_std,
            y_raw=float(bundle.targets_raw["valid"][int(row)]),
            row_id=int(bundle.global_row_ids["valid"][int(row)]),
            pilot=pilot, search_log=log, device=device,
        )
        failures.extend(f"smoke valid row {rec['row_id']}: {f}" for f in fails)
        smoke.append(
            {
                "row_id": rec["row_id"],
                "cache_row": rec["cache_row"],
                "n_bonds": rec["n_bonds"],
                "top1_bonds": rec["top1"]["bonds"],
                "dose": rec["top1"]["dose"],
                "pilot_effect_check_passed": rec["top1"]["pilot_effect_check"]["passed"],
                "controls_status": rec["controls"]["status"],
                "n_matched": rec["controls"]["n_matched"],
                "outside_response_fraction": rec["localization"]["outside_response_fraction"],
                "gate_drop_passed": rec["checks"]["s2_gate_drop_all_passed"],
                "weight_count_passed": rec["checks"]["s2_weight_count_all_passed"],
                "occurrence_mapping_n_failed": int(rec["checks"]["occurrence_mapping_n_failed"]),
                "occurrence_mapping_checks": int(rec["checks"]["occurrence_mapping_checks"]),
                "localization_closure_passed": rec["localization"]["closure_local_sum_passed"],
                "readout_closure_passed": rec["localization"]["closure_total_passed"],
            }
        )
    state_after = C.sha256_state(arm.state_dict())
    gates = [
        {"gate": "DATA_IDENTITY", "passed": bool(out["data"]["pass"]),
         "detail": "frozen caches + official split sizes/disjointness"},
        {"gate": "PILOT_SAMPLE_IDENTITY", "passed": bool(out["samples"]["pass"]),
         "detail": "the pilot's 64 row ids per split are reproduced exactly"},
        {"gate": "TRAINING_IDENTITY", "passed": bool(source["pass"]),
         "detail": "checkpoint/trajectory SHA-256, 360-epoch 3-split coverage, selected-epoch argmin"},
        {"gate": "MODEL_EQUIVALENCE_ALL_ONES",
         "passed": bool(all(v["passed"] for v in equivalence.values())),
         "detail": "all-ones gates reproduce frozen arm A on every split"},
        {"gate": "OCCURRENCE_MAPPING",
         "passed": bool(pilot_preflight["engineering_gate"] == "PASS" and all(
             v["occurrence_mapping_n_failed"] == 0 for v in smoke)),
         "detail": "occurrence map rebuilt and verified again on the smoke molecules"},
        {"gate": "TOP1_SOURCE_IDENTITY",
         "passed": bool(all(v["pilot_effect_check_passed"] for v in smoke)
                        and all(v.get("controls_status") in ("OK", "INSUFFICIENT_MATCHED_NULL") for v in smoke)),
         "detail": "pilot Top-1 bond set, gradient, discovery score, search-log entry and masked effect replay"},
        {"gate": "S2_AND_OR_ACCOUNTING",
         "passed": bool(all(v["gate_drop_passed"] for v in smoke)),
         "detail": "masking S removes exactly the e1-in-S OR e2-in-S occurrences on the frozen forward"},
        {"gate": "S2_WEIGHT_COUNT",
         "passed": bool(all(v["weight_count_passed"] for v in smoke)),
         "detail": "the real s2_weight tensor has exactly n_s2_affected zeros and n_s2_unaffected ones"},
        {"gate": "LOCALIZATION_CLOSURE",
         "passed": bool(all(v["localization_closure_passed"] and v["readout_closure_passed"] for v in smoke)),
         "detail": "sum_v delta_c_v == delta_L and delta_L + delta_rho == delta_total"},
        {"gate": "PARAMETER_HASH_UNCHANGED",
         "passed": bool(state_before == state_after),
         "detail": "no parameter was modified by the audit path"},
        {"gate": "PARAMETER_BUDGET", "passed": bool(parameters["matches_expected"]),
         "detail": "228674 total / 15298 readout"},
    ]
    out.update(
        {
            "engineering_gate": "PASS" if all(g["passed"] for g in gates) else "FAIL",
            "gates": gates,
            "model_equivalence": equivalence,
            "parameters": parameters,
            "state_hash": {"before": state_before, "after": state_after, "unchanged": bool(state_before == state_after)},
            "pilot_engineering_gate": pilot_preflight["engineering_gate"],
            "smoke": {
                "note": "full audit path on the first 8 selected Valid molecules; never aggregated",
                "n_molecules": len(smoke),
                "records": smoke,
                "failures": failures,
                "passed": bool(not failures),
            },
        }
    )
    out["pass"] = bool(out["engineering_gate"] == "PASS" and not failures)
    return out


def run_dataset(dataset: str, *, stage: str = "audit", limit: int | None = None) -> dict[str, Any]:
    _configure_threads()
    t_start = time.perf_counter()
    out_dir = results_dir(dataset)
    C.ensure_dir(out_dir)
    bundle = SD.build_bundle(dataset)
    arm, _payload = load_arm(dataset, bundle, _device())
    sigma = float(bundle.scaler["std"])
    mu = float(bundle.scaler["mean"])
    train_std = float(bundle.scaler["std"])
    device = _device()
    occs = {s: D.build_split_occ(bundle.caches[s], bundle.spec, dataset, s) for s in C.SPLITS}
    samples = sample_identity(dataset, bundle)
    preflight = preflight_dataset(
        dataset, bundle=bundle, arm=arm, occs=occs, samples=samples
    )
    C.ensure_dir(out_dir)
    C.atomic_write_json(out_dir / "preflight.json", preflight)
    C.atomic_write_json(out_dir / "samples.json", {
        "dataset": dataset,
        "seed": C.SAMPLE_SEED,
        "n_per_split": C.N_SAMPLES_PER_SPLIT,
        "splits": {
            s: {
                "row_ids": [int(r) for r in samples["splits"][s]["row_ids"]],
                "cache_rows": [int(r) for r in samples["splits"][s]["cache_rows"]],
                "row_ids_sha256": samples["splits"][s]["row_ids_sha256"],
            }
            for s in C.SPLITS
        },
        "identity": samples,
    })
    if stage == "preflight":
        summary = {
            "dataset": dataset, "stage": stage,
            "engineering_gate": preflight["engineering_gate"],
            "pass": preflight["pass"],
            "n_smoke": preflight["smoke"]["n_molecules"],
            "failures": preflight["smoke"]["failures"],
        }
        C.atomic_write_json(out_dir / "run_summary.json", summary)
        return summary
    if not preflight["pass"]:
        raise RuntimeError(
            f"{dataset}: ENGINEERING_GATE FAILED -> "
            f"{[g['gate'] for g in preflight['gates'] if not g['passed']]} "
            f"failures={preflight['smoke']['failures']}"
        )

    search_log = np.load(C.upstream_paths(dataset)["candidates"])
    n_planned = 0
    n_completed = 0
    compute_seconds = 0.0
    hard_failures: list[str] = []
    per_split_counts: dict[str, int] = {}
    for split in C.SPLITS:
        pilot_records = C.upstream_molecules(dataset, split)
        rows = [int(r) for r in samples["splits"][split]["cache_rows"]]
        if len(rows) != len(pilot_records):
            raise RuntimeError(
                f"{dataset}/{split}: {len(rows)} selected rows for {len(pilot_records)} pilot records"
            )
        records: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        atoms: list[dict[str, Any]] = []
        plan = list(zip(rows, pilot_records))
        if limit is not None:
            plan = plan[: int(limit)]
        n_planned += len(plan)
        for row, pilot in plan:
            t0 = time.perf_counter()
            rec, led, atom_rows, fails = analyse_molecule(
                arm, bundle.caches[split], occs[split], int(row),
                dataset=dataset, split=split, spec=bundle.spec,
                sigma=sigma, mu=mu, train_std=train_std,
                y_raw=float(bundle.targets_raw[split][int(row)]),
                row_id=int(bundle.global_row_ids[split][int(row)]),
                pilot=pilot, search_log=search_log, device=device,
            )
            compute_seconds += time.perf_counter() - t0
            records.append(rec)
            ledger.extend(led)
            atoms.extend(atom_rows)
            n_completed += 1
            hard_failures.extend(f"{split}/row{rec['row_id']}: {f}" for f in fails)
            del rec, led, atom_rows
        C.ensure_dir(split_dir(dataset, split))
        C.atomic_write_json(split_dir(dataset, split) / "records.json", records)
        C.atomic_write_npz(split_dir(dataset, split) / "candidates.npz", K.ledger_arrays(ledger))
        atom_columns = ["dataset", "split", "row_id", "atom", "inside",
                        "c_original_raw", "c_masked_raw", "delta_c_raw", "abs_delta_c_raw"]
        C.write_csv(split_dir(dataset, split) / "atom_response.csv", atoms, columns=atom_columns)
        per_split_counts[split] = len(records)
        summary_rows = [
            {
                "dataset": dataset, "split": split, "row_id": r["row_id"],
                "n_bonds": r["n_bonds"], "top1_n_bonds": r["top1"]["n_bonds"],
                "controls_status": r["controls"]["status"], "n_matched": r["controls"]["n_matched"],
                "outside_response_fraction": r["localization"]["outside_response_fraction"],
            }
            for r in records
        ]
        C.write_csv(split_dir(dataset, split) / "summary.csv", summary_rows)
        del records, ledger, atoms

    budget = {
        "dataset": dataset,
        "stage": stage,
        "n_molecules_planned": int(n_planned),
        "n_molecules_completed": int(n_completed),
        "per_split": per_split_counts,
        "compute_seconds": compute_seconds,
        "wall_seconds": time.perf_counter() - t_start,
        "device": device,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "memory_guard": {
            "max_variants_per_forward": int(C.MAX_VARIANTS_PER_FORWARD),
            "max_forward_atoms": int(C.MAX_FORWARD_ATOMS),
            "max_forward_bonds": int(C.MAX_FORWARD_BONDS),
            "torch_threads": int(_torch_threads()),
        },
        "gpu_hours": 0.0 if device == "cpu" else compute_seconds / 3600.0,
    }
    C.atomic_write_json(out_dir / "budget.json", budget)
    summary = {
        "dataset": dataset,
        "stage": stage,
        "engineering_gate": preflight["engineering_gate"],
        "n_molecules": int(n_completed),
        "per_split": per_split_counts,
        "hard_failures": hard_failures,
        "budget": budget,
    }
    C.atomic_write_json(out_dir / "run_summary.json", summary)
    del arm, bundle, occs, search_log
    if hard_failures:
        raise RuntimeError(
            f"{dataset}: {len(hard_failures)} hard identity/closure failures; stopping instead of "
            f"reporting unreconciled numbers: {hard_failures[:5]}"
        )
    return summary


__all__ = [
    "data_identity",
    "sample_identity",
    "training_identity",
    "load_arm",
    "analyse_molecule",
    "preflight_dataset",
    "run_dataset",
    "results_dir",
    "split_dir",
]
