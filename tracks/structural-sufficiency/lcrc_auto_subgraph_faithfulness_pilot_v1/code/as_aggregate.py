"""Deterministic aggregate stage: tables, bootstrap, failure cases, verdicts.

CPU only.  Every number is read back from the per-molecule records that the GPU
workers wrote; nothing is re-derived from a figure and nothing is typed by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import as_common as C


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_dataset(dataset: str) -> dict[str, Any]:
    base = C.RESULTS_DIR / dataset
    out: dict[str, Any] = {
        "dataset": dataset,
        "samples": C.read_json(base / "samples.json"),
        "preflight": C.read_json(base / "preflight.json"),
        "training_source": C.read_json(base / "training_source.json"),
        "budget": C.read_json(base / "budget.json"),
        "run_summary": C.read_json(base / "run_summary.json"),
        "molecules": {},
    }
    for split in C.SPLITS:
        path = base / split / "molecules.json"
        out["molecules"][split] = C.read_json(path) if path.is_file() else []
    return out


def _dataset_bundle(dataset: str):
    import cr_data as SD

    return SD.build_bundle(dataset)


def train_std_of(dataset: str) -> float:
    return float(C.source_scaler(dataset)["std"])


# ---------------------------------------------------------------------------
# per-molecule derived quantities
# ---------------------------------------------------------------------------


def _top1(mol: Mapping[str, Any]) -> Mapping[str, Any] | None:
    topk = mol.get("topk") or []
    return topk[0] if topk else None


def molecule_row(mol: Mapping[str, Any], train_std: float) -> dict[str, Any]:
    """One flat table row per explained molecule."""
    top1 = _top1(mol)
    matched = mol.get("matched_random", {})
    eff = (matched or {}).get("effects")
    heur = mol.get("local_heuristic") or {}
    dec = mol.get("decomposition") or {}
    dose = mol.get("dose") or {}
    row: dict[str, Any] = {
        "dataset": mol["dataset"],
        "split": mol["split"],
        "row_id": mol["row_id"],
        "cache_row": mol["cache_row"],
        "n_atoms": mol["n_atoms"],
        "n_bonds": mol["n_bonds"],
        "y_raw": mol["y_raw"],
        "pred_raw": mol["pred_raw_base"],
        "abs_error_raw": abs(mol["pred_raw_base"] - mol["y_raw"]),
        "base_rho_share": mol["base_rho_share"],
        "train_target_std": train_std,
        "has_top1": int(top1 is not None),
        "top1_bonds": " ".join(str(b) for b in (top1 or {}).get("bonds", [])),
        "top1_n_bonds": (top1 or {}).get("n_bonds"),
        "top1_n_atoms": (top1 or {}).get("n_atoms_involved"),
        "top1_score_A": (top1 or {}).get("score_A"),
        "top1_n_s2_occurrences": (top1 or {}).get("n_s2_occurrences"),
        "top1_n_s1_codes": (top1 or {}).get("n_s1_codes"),
        "top1_s1_coverage": (
            (top1 or {}).get("n_s1_codes", 0) / mol["n_s1_total"] if mol.get("n_s1_total") else None
        ),
        "top1_s2_coverage": (
            (top1 or {}).get("n_s2_occurrences", 0) / mol["n_s2_total"] if mol.get("n_s2_total") else None
        ),
        "top1_predicted_delta_from_g": (top1 or {}).get("predicted_delta_from_g"),
        "top1_sign_agrees": int(bool((top1 or {}).get("sign_agrees"))),
    }
    if top1:
        e = top1["effect"]
        row.update(
            {
                "top1_signed_delta": e["signed_delta"],
                "top1_abs_delta": e["abs_delta"],
                "top1_abs_delta_over_train_std": e["abs_delta_over_train_std"],
                "top1_delta_L": e["delta_L"],
                "top1_delta_P": e["delta_P"],
                "top1_rho_share": e["rho_share"],
                "top1_cancellation": e["cancellation"],
                "top1_linearisation_abs_error": abs(e["signed_delta"] - top1["predicted_delta_from_g"]),
                "top1_linearisation_rel_error": (
                    abs(e["signed_delta"] - top1["predicted_delta_from_g"])
                    / max(abs(top1["predicted_delta_from_g"]), 1e-12)
                ),
                "top1_meets_effect_floor": int(e["abs_delta"] >= C.EFFECT_FLOOR_STD_FRACTION * train_std),
            }
        )
    for key, name in (("s1_only", "top1_s1_only"), ("s2_only", "top1_s2_only"), ("s1_s2", "top1_s1_s2")):
        if dec.get(key):
            row[f"{name}_signed_delta"] = dec[key]["signed_delta"]
            row[f"{name}_abs_delta"] = dec[key]["abs_delta"]
    if dec:
        row["top1_interaction_delta"] = dec.get("interaction_delta")
        row["top1_interaction_abs"] = abs(dec.get("interaction_delta", 0.0))
    if dose:
        row["dose_gate"] = dose.get("gate_value")
        row["dose_signed_delta"] = dose.get("signed_delta")
        row["dose_abs_delta"] = dose.get("abs_delta")
        row["dose_same_direction"] = int(bool(dose.get("same_direction_as_full_mask")))
        row["dose_linearisation_abs_error"] = dose.get("linearisation_abs_error")
    row["matched_status"] = (matched or {}).get("status")
    row["matched_attempts"] = (matched or {}).get("attempts")
    row["matched_n"] = (matched or {}).get("n_matched")
    if eff:
        row.update(
            {
                "matched_median_abs_delta": eff["median_abs_delta"],
                "effect_ratio": eff["effect_ratio"],
                "effect_advantage": eff["effect_advantage"],
                "ratio_gt1": int(eff["effect_ratio"] > 1.0),
            }
        )
    else:
        row["matched_median_abs_delta"] = None
        row["effect_ratio"] = None
        row["effect_advantage"] = None
        row["ratio_gt1"] = None
    if heur.get("effect"):
        row["heuristic_abs_delta"] = heur["effect"]["abs_delta"]
        row["heuristic_advantage"] = (
            top1["effect"]["abs_delta"] - heur["effect"]["abs_delta"] if top1 else None
        )
        row["heuristic_bonds"] = " ".join(str(b) for b in heur["bonds"])
        row["heuristic_n_bonds"] = heur["n_bonds"]
    # sign agreement pooled over Top-1..3
    topk = mol.get("topk") or []
    row["n_topk"] = len(topk)
    row["sign_agrees_topk_mean"] = (
        float(np.mean([bool(t["sign_agrees"]) for t in topk])) if topk else None
    )
    row["n_s2_total"] = mol.get("n_s2_total")
    return row


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def _boot(values: Sequence[float], *, statistic: str = "median") -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if statistic == "median":
        return C.bootstrap_median_ci(arr)
    if statistic == "mean":
        return C.bootstrap_mean_ci(arr)
    if statistic == "proportion":
        return C.bootstrap_mean_ci(arr)
    raise KeyError(statistic)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    # q90 is used by the diagnostics, so it must be requested explicitly
    return C.quantiles(values, qs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------


MAIN_COLUMNS = [
    "dataset", "split", "n_selected", "n_interpreted", "coverage",
    "top1_n_bonds_median", "top1_n_atoms_median", "top1_n_s2_occurrences_median",
    "abs_delta_top1_median", "abs_delta_top1_median_over_train_std",
    "frac_top1_ge_1pct_train_std",
    "sign_agreement_top1_frac", "sign_agreement_topk_frac",
    "linearisation_median_abs_err", "linearisation_median_rel_err",
    "matched_n_molecules", "matched_success_rate", "matched_median_effect_ratio",
    "matched_ratio_gt1_frac", "matched_median_effect_advantage",
    "matched_median_abs_delta_random",
    "local_heuristic_median_abs_delta", "local_heuristic_median_advantage",
    "frac_top1_gt_local_heuristic",
]


def main_table(rows: Sequence[Mapping[str, Any]], dataset: str, split: str, n_selected: int) -> dict[str, Any]:
    sub = [r for r in rows if r["dataset"] == dataset and r["split"] == split]
    interpreted = [r for r in sub if r["has_top1"]]
    train_std = float(sub[0]["train_target_std"]) if sub else float("nan")
    matched = [r for r in interpreted if r["effect_ratio"] is not None]
    heur = [r for r in interpreted if r.get("heuristic_advantage") is not None]
    return {
        "dataset": dataset,
        "split": split,
        "n_selected": int(n_selected),
        "n_interpreted": len(interpreted),
        "coverage": len(interpreted) / n_selected if n_selected else 0.0,
        "top1_n_bonds_median": C.median([r["top1_n_bonds"] for r in interpreted]),
        "top1_n_atoms_median": C.median([r["top1_n_atoms"] for r in interpreted]),
        "top1_n_s2_occurrences_median": C.median([r["top1_n_s2_occurrences"] for r in interpreted]),
        "abs_delta_top1_median": C.median([r["top1_abs_delta"] for r in interpreted]),
        "abs_delta_top1_median_over_train_std": C.median(
            [r["top1_abs_delta_over_train_std"] for r in interpreted]
        ),
        "frac_top1_ge_1pct_train_std": (
            float(np.mean([r["top1_meets_effect_floor"] for r in interpreted])) if interpreted else float("nan")
        ),
        "sign_agreement_top1_frac": (
            float(np.mean([r["top1_sign_agrees"] for r in interpreted])) if interpreted else float("nan")
        ),
        "sign_agreement_topk_frac": (
            float(np.mean([r["sign_agrees_topk_mean"] for r in interpreted])) if interpreted else float("nan")
        ),
        "linearisation_median_abs_err": C.median([r["top1_linearisation_abs_error"] for r in interpreted]),
        "linearisation_median_rel_err": C.median(
            [r["top1_linearisation_rel_error"] for r in interpreted]
        ),
        "matched_n_molecules": len(matched),
        "matched_success_rate": len(matched) / len(interpreted) if interpreted else float("nan"),
        "matched_median_effect_ratio": C.median([r["effect_ratio"] for r in matched]),
        "matched_ratio_gt1_frac": (
            float(np.mean([r["ratio_gt1"] for r in matched])) if matched else float("nan")
        ),
        "matched_median_effect_advantage": C.median([r["effect_advantage"] for r in matched]),
        "matched_median_abs_delta_random": C.median([r["matched_median_abs_delta"] for r in matched]),
        "local_heuristic_median_abs_delta": C.median([r["heuristic_abs_delta"] for r in heur]),
        "local_heuristic_median_advantage": C.median([r["heuristic_advantage"] for r in heur]),
        "frac_top1_gt_local_heuristic": (
            float(np.mean([r["heuristic_advantage"] > 0 for r in heur])) if heur else float("nan")
        ),
        "_train_std": train_std,
    }


DIAG_COLUMNS = [
    "dataset", "split", "n",
    "rho_share_median", "rho_share_q25", "rho_share_q75", "frac_delta_P_dominates",
    "cancellation_median", "cancellation_q75", "frac_material_cancellation",
    "abs_s1_only_median", "abs_s2_only_median", "abs_s1_s2_median",
    "frac_s2_gt_s1", "signed_interaction_median", "abs_interaction_median",
    "frac_positive_interaction", "dose_same_direction_frac", "dose_abs_ratio_median",
    "base_rho_share_median", "local_sum_abs_share_median",
    "top1_s1_coverage_median", "top1_s2_coverage_median",
    "median_abs_delta_over_train_std", "p90_abs_delta_over_train_std",
]


def diag_table(rows: Sequence[Mapping[str, Any]], dataset: str, split: str) -> dict[str, Any]:
    sub = [r for r in rows if r["dataset"] == dataset and r["split"] == split and r["has_top1"]]
    rho = [r["top1_rho_share"] for r in sub]
    canc = [r["top1_cancellation"] for r in sub]
    abs_s1 = [abs(r.get("top1_s1_only_signed_delta", 0.0)) for r in sub]
    abs_s2 = [abs(r.get("top1_s2_only_signed_delta", 0.0)) for r in sub]
    abs_both = [abs(r.get("top1_s1_s2_signed_delta", 0.0)) for r in sub]
    inter = [r.get("top1_interaction_delta", 0.0) for r in sub]
    dose_same = [r.get("dose_same_direction") for r in sub if r.get("dose_same_direction") is not None]
    dose_ratio = [
        r["dose_abs_delta"] / r["top1_abs_delta"]
        for r in sub
        if r.get("dose_abs_delta") is not None and r.get("top1_abs_delta")
    ]
    local_share = [
        abs(r["top1_delta_L"]) / (abs(r["top1_delta_L"]) + abs(r["top1_delta_P"]) + 1e-12) for r in sub
    ]
    s1_cov = [r.get("top1_s1_coverage") for r in sub]
    s2_cov = [r.get("top1_s2_coverage") for r in sub]
    over_std = [r["top1_abs_delta_over_train_std"] for r in sub]
    return {
        "dataset": dataset,
        "split": split,
        "n": len(sub),
        "rho_share_median": C.median(rho),
        "rho_share_q25": _quantiles(rho).get("q25"),
        "rho_share_q75": _quantiles(rho).get("q75"),
        "frac_delta_P_dominates": float(np.mean([x > 0.5 for x in rho])) if rho else float("nan"),
        "cancellation_median": C.median(canc),
        "cancellation_q75": _quantiles(canc).get("q75"),
        "frac_material_cancellation": float(np.mean([x > 0.5 for x in canc])) if canc else float("nan"),
        "abs_s1_only_median": C.median(abs_s1),
        "abs_s2_only_median": C.median(abs_s2),
        "abs_s1_s2_median": C.median(abs_both),
        "frac_s2_gt_s1": float(np.mean([b > a for a, b in zip(abs_s1, abs_s2)])) if sub else float("nan"),
        "signed_interaction_median": C.median(inter),
        "abs_interaction_median": C.median([abs(x) for x in inter]),
        "frac_positive_interaction": float(np.mean([x > 0 for x in inter])) if inter else float("nan"),
        "dose_same_direction_frac": float(np.mean(dose_same)) if dose_same else float("nan"),
        "dose_abs_ratio_median": C.median(dose_ratio),
        "base_rho_share_median": C.median([r["base_rho_share"] for r in sub]),
        "local_sum_abs_share_median": C.median(local_share),
        "top1_s1_coverage_median": C.median([x for x in s1_cov if x is not None]),
        "top1_s2_coverage_median": C.median([x for x in s2_cov if x is not None]),
        "median_abs_delta_over_train_std": C.median(over_std),
        "p90_abs_delta_over_train_std": _quantiles(over_std).get("q90"),
    }


# ---------------------------------------------------------------------------
# bootstrap + failure cases
# ---------------------------------------------------------------------------


def bootstrap_payload(datasets: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": (
            "percentile bootstrap over molecules; it describes sample uncertainty under this "
            "fixed checkpoint and this fixed dataset. It is NOT a cross-seed stability claim."
        ),
        "replicates": C.BOOTSTRAP_REPLICATES,
        "seed": C.BOOTSTRAP_SEED,
        "unit": "molecule",
        "datasets": {},
    }
    for dataset in datasets:
        dsub = [r for r in rows if r["dataset"] == dataset]
        ds: dict[str, Any] = {"splits": {}}
        for split in C.SPLITS:
            sub = [r for r in dsub if r["split"] == split and r["has_top1"]]
            matched = [r for r in sub if r["effect_ratio"] is not None]
            heur = [r for r in sub if r.get("heuristic_advantage") is not None]
            ds["splits"][split] = {
                "n_interpreted": len(sub),
                "n_matched": len(matched),
                "effect_ratio": _boot([r["effect_ratio"] for r in matched], statistic="median"),
                "effect_ratio_gt1_proportion": _boot([float(r["ratio_gt1"]) for r in matched],
                                                     statistic="proportion"),
                "effect_advantage": _boot([r["effect_advantage"] for r in matched], statistic="median"),
                "abs_delta_top1": _boot([r["top1_abs_delta"] for r in sub], statistic="median"),
                "abs_delta_top1_over_train_std": _boot(
                    [r["top1_abs_delta_over_train_std"] for r in sub], statistic="median"
                ),
                "effect_floor_proportion": _boot([float(r["top1_meets_effect_floor"]) for r in sub],
                                                 statistic="proportion"),
                "heuristic_advantage": _boot([r["heuristic_advantage"] for r in heur], statistic="median"),
                "rho_share": _boot([r["top1_rho_share"] for r in sub], statistic="median"),
                "cancellation": _boot([r["top1_cancellation"] for r in sub], statistic="median"),
            }
        ds["paired_valid_test"] = {}
        by_id = {(r["split"], r["row_id"]): r for r in dsub}
        out["datasets"][dataset] = ds
        del by_id
    return out


def failure_cases(rows: Sequence[Mapping[str, Any]], top_n: int = 8) -> dict[str, Any]:
    """Cases where the discovery score and the real masked response disagree."""
    out: dict[str, Any] = {"high_gradient_low_effect": [], "high_local_output_low_effect": []}
    for dataset in C.DATASETS:
        for split in C.SPLITS:
            sub = [
                r for r in rows
                if r["dataset"] == dataset and r["split"] == split and r["has_top1"]
            ]
            if not sub:
                continue
            graded = sorted(
                sub, key=lambda r: (-abs(r["top1_predicted_delta_from_g"]), r["row_id"])
            )[: max(1, len(sub) // 4)]
            low = sorted(graded, key=lambda r: (r["top1_abs_delta"], r["row_id"]))[:top_n]
            for r in low:
                out["high_gradient_low_effect"].append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "row_id": r["row_id"],
                        "predicted_delta_from_g": r["top1_predicted_delta_from_g"],
                        "actual_abs_delta": r["top1_abs_delta"],
                        "ratio_actual_to_predicted": r["top1_abs_delta"]
                        / max(abs(r["top1_predicted_delta_from_g"]), 1e-12),
                        "harmful_for_discovery": bool(r["top1_abs_delta"] < abs(r["top1_predicted_delta_from_g"])),
                    }
                )
            # local-output heuristic failure: large |c_v| sum but tiny real response
            heur = [r for r in sub if r.get("heuristic_abs_delta") is not None]
            if heur:
                top_local = sorted(heur, key=lambda r: (-r["heuristic_abs_delta"], r["row_id"]))[
                    : max(1, len(heur) // 4)
                ]
                for r in sorted(top_local, key=lambda r: (r["top1_abs_delta"], r["row_id"]))[:top_n]:
                    out["high_local_output_low_effect"].append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "row_id": r["row_id"],
                            "heuristic_abs_delta": r["heuristic_abs_delta"],
                            "discovery_abs_delta": r["top1_abs_delta"],
                        }
                    )
    return out


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def signal_condition(main: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    """The frozen, development-only DISCOVERY_SIGNAL rule, per split."""
    out: dict[str, Any] = {"dataset": dataset, "splits": {}, "pass": None}
    per_split_pass = []
    for split in ("valid", "test"):
        row = main[(dataset, split)]
        n_interpreted = int(row["n_interpreted"])
        n_matched = int(row["matched_n_molecules"])
        enough = bool(n_matched >= max(32, int(0.8 * n_interpreted)) and n_interpreted > 0)
        ratio_median = row["matched_median_effect_ratio"]
        ratio_frac = row["matched_ratio_gt1_frac"]
        floor_frac = row["frac_top1_ge_1pct_train_std"]
        checks = {
            "enough_matched_controls": {
                "value": n_matched,
                "rule": "matched molecules >= max(32, 80% of interpreted molecules)",
                "passed": enough,
            },
            "effect_ratio_median_ge_1p5": {
                "value": ratio_median,
                "rule": ">= 1.5",
                "passed": bool(np.isfinite(ratio_median) and ratio_median >= C.SIGNAL_EFFECT_RATIO_MIN),
            },
            "effect_ratio_gt1_fraction_ge_0p70": {
                "value": ratio_frac,
                "rule": ">= 0.70",
                "passed": bool(np.isfinite(ratio_frac) and ratio_frac >= C.SIGNAL_EFFECT_RATIO_FRACTION_MIN),
            },
            "effect_floor_fraction_ge_0p50": {
                "value": floor_frac,
                "rule": ">= 0.50 of molecules reach 1% of the Train target standard deviation",
                "passed": bool(np.isfinite(floor_frac) and floor_frac >= C.SIGNAL_EFFECT_FLOOR_FRACTION_MIN),
            },
        }
        passed = all(c["passed"] for c in checks.values())
        out["splits"][split] = {"checks": checks, "passed": passed}
        per_split_pass.append(passed)
    out["pass"] = bool(all(per_split_pass))
    out["failed_conditions"] = [
        f"{split}:{name}"
        for split, rec in out["splits"].items()
        for name, chk in rec["checks"].items()
        if not chk["passed"]
    ]
    return out


def architecture_diagnosis(diag: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Reported evidence only -- no automatic 'must change the readout' claim."""
    out: dict[str, Any] = {"datasets": {}, "note": (
        "descriptive only: rho_share / cancellation / S1-S2 interaction / local path coverage. "
        "This stage does not decide whether the architecture must change."
    )}
    for dataset in C.DATASETS:
        rec = {}
        for split in C.SPLITS:
            d = diag.get((dataset, split))
            if d:
                rec[split] = {
                    k: d[k]
                    for k in (
                        "n", "rho_share_median", "rho_share_q25", "rho_share_q75",
                        "frac_delta_P_dominates", "cancellation_median", "cancellation_q75",
                        "frac_material_cancellation", "abs_s1_only_median", "abs_s2_only_median",
                        "abs_s1_s2_median", "frac_s2_gt_s1", "signed_interaction_median",
                        "abs_interaction_median", "frac_positive_interaction",
                        "dose_same_direction_frac", "dose_abs_ratio_median", "base_rho_share_median",
                        "local_sum_abs_share_median", "top1_s1_coverage_median",
                        "top1_s2_coverage_median",
                    )
                }
        out["datasets"][dataset] = rec
    return out


__all__ = [
    "load_dataset",
    "molecule_row",
    "main_table",
    "diag_table",
    "bootstrap_payload",
    "failure_cases",
    "signal_condition",
    "architecture_diagnosis",
    "MAIN_COLUMNS",
    "DIAG_COLUMNS",
]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    C.ensure_dir(path.parent)
    C.write_csv(path, rows, columns)
    return C.relpath(path)


def build_rows(per_dataset: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in C.DATASETS:
        std = train_std_of(dataset)
        for split in C.SPLITS:
            for mol in per_dataset[dataset]["molecules"][split]:
                rows.append(molecule_row(mol, std))
    return rows


def merge_preflight(per_dataset: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for dataset in C.DATASETS:
        pf = per_dataset[dataset]["preflight"]
        gates[dataset] = {
            "engineering_gate": pf["engineering_gate"],
            "gates": pf["gates"],
            "occurrence_mapping": pf["occurrence_mapping"],
            "model_equivalence": pf["model_equivalence"],
            "model_equivalence_other_splits": pf["model_equivalence_other_splits"],
            "batched_vs_single": pf["batched_vs_single"],
            "atom_renumbering": {
                k: v for k, v in pf["atom_renumbering"].items() if k != "records"
            },
            "atom_renumbering_records": pf["atom_renumbering"]["records"],
            "finite_difference": {
                k: v for k, v in pf["finite_difference"].items() if k != "records"
            },
            "finite_difference_records": pf["finite_difference"]["records"],
            "masked_occurrence_single_removal": pf["masked_occurrence_single_removal"],
            "checkpoint_replay": pf["checkpoint_replay"],
            "state_hash": pf["state_hash"],
            "parameters": pf["parameters"],
            "data": {
                "frozen_files": pf["data"]["frozen_files"],
                "split_sizes": pf["data"]["split_sizes"],
                "split_disjointness": pf["data"]["split_disjointness"],
                "splits_cover_all_rows": pf["data"]["splits_cover_all_rows"],
                "target_scaler_train_only": pf["data"]["target_scaler_train_only"],
            },
        }
    payload = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "condition_name": C.CONDITION_NAME,
        "generated_at": C.utcnow(),
        "freeze_hash": C.freeze_hash(),
        "datasets": gates,
        "engineering_gate": (
            "PASS" if all(g["engineering_gate"] == "PASS" for g in gates.values()) else "FAIL"
        ),
        "critical_failures": [
            f"{d}:{gate['gate']}" for d, g in gates.items() for gate in g["gates"] if not gate["passed"]
        ],
    }
    C.atomic_write_json(C.ANALYSIS_DIR / "preflight.json", payload)
    return payload


def merge_training_source(per_dataset: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = {
        "round_id": C.ROUND_ID,
        "generated_at": C.utcnow(),
        "policy": (
            "reuse only: no arm-A retraining was performed in this round. Both datasets' "
            "checkpoint, trajectory and prediction identity were verified against the source "
            "round's recorded SHA-256 before any explanation was computed."
        ),
        "reused_from": "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/results/<dataset>/A",
        "datasets": {d: per_dataset[d]["training_source"] for d in C.DATASETS},
        "all_verified": all(per_dataset[d]["training_source"]["pass"] for d in C.DATASETS),
        "retrained_datasets": [],
    }
    C.atomic_write_json(C.ANALYSIS_DIR / "training_source.json", payload)
    return payload


def execution_provenance() -> dict[str, Any]:
    """Execution-regime provenance of the GPU workers, from the control-plane records.

    ``.rr/pulled/<run>/meta.json`` is written by ``rr`` itself (commit, pool, driver,
    node, torch backend, state).  It is local bookkeeping, not part of the repo, so
    this block is best-effort: a fresh clone without ``.rr/`` reports
    ``available: false`` while every number in the round is unaffected.
    """
    root = C.REPO_ROOT / ".rr" / "pulled"
    runs: list[dict[str, Any]] = []
    if root.is_dir():
        for meta in sorted(root.glob("*/meta.json")):
            try:
                payload = C.read_json(meta)
            except Exception:                       # noqa: BLE001 - provenance only
                continue
            paths = payload.get("result_paths") or []
            needle = C.ROUND_ID.replace("-", "_")      # repo dir uses underscores
            if not any(needle in str(p) for p in paths):
                continue
            runtime = payload.get("runtime") or {}
            command = str(payload.get("command") or "")
            stage = None
            for token in command.replace("'", " ").split():
                if token.startswith("stage="):
                    stage = token.split("=", 1)[1]
            runs.append(
                {
                    "stage": stage,
                    "experiment": payload.get("experiment"),
                    "run_id": payload.get("run_id"),
                    "state": payload.get("state"),
                    "pool": payload.get("pool"),
                    "backend": payload.get("backend"),
                    "git_commit": payload.get("git_commit"),
                    "git_branch": payload.get("git_branch"),
                    "git_dirty": payload.get("git_dirty"),
                    "torch_backend": payload.get("torch_backend"),
                    "node": runtime.get("node"),
                    "driver_version": runtime.get("driver_version"),
                    "gpu_names": (runtime.get("gpus") or {}).get("names"),
                    "python_version": runtime.get("python_version"),
                    "created_at": payload.get("created_at"),
                    "command": command or None,
                }
            )
    worker_commits = sorted(
        {r["git_commit"] for r in runs if r.get("stage") == "dataset_worker" and r.get("git_commit")}
    )
    return {
        "available": bool(runs),
        "results_commit": worker_commits[0] if len(worker_commits) == 1 else worker_commits,
        "source": ".rr/pulled/*/meta.json (rr run records; local, not committed)",
        "runs": runs,
        "analysis_commit": C.git_commit(),
        "analysis_branch": C.git_branch(),
        "analysis_worktree_dirty": C.worktree_dirty(),
        "note": (
            "the per-molecule records under results/ were produced by the GPU workers listed in "
            "`runs` (their commit is the deployed revision); this file was regenerated by the "
            "CPU-only aggregate stage whose commit is `analysis_commit`. "
            "The aggregate stage never changes a weight or a per-molecule record."
        ),
    }


def write_run_manifest(per_dataset: Mapping[str, Mapping[str, Any]],
                       extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import cr_data as SD

    cache_hashes = {}
    for dataset in C.DATASETS:
        cache_hashes[dataset] = {
            "graph_cache": C.sha256_file(C.SQL_DATA / SD.cache_filename(dataset)),
            "lipo_raw": C.sha256_file(C.SQL_DATA / SD.lipo_raw_filename(dataset)),
        }
    payload = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "generated_at": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_branch": C.git_branch(),
        "protocol_hash": C.freeze_hash(),
        "protocol_file_sha256": C.sha256_file(C.ROUND_DIR / "PROTOCOL.md")
        if (C.ROUND_DIR / "PROTOCOL.md").is_file() else None,
        "code_sha256": C.code_hashes(),
        "frozen_inputs": {
            "caches": cache_hashes,
            "checkpoints": {d: C.EXPECTED_CHECKPOINT_SHA256[d] for d in C.DATASETS},
            "trajectories": {d: C.EXPECTED_TRAJECTORY_SHA256[d] for d in C.DATASETS},
            "readonly_reuse": {
                "tracks/structural-query-learning/code": "read-only",
                "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code": "read-only",
                "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code": "read-only",
                "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1": "read-only",
            },
        },
        "environment": C.environment_snapshot(),
        "gpu": {
            d: {
                "device": per_dataset[d]["preflight"]["device"],
                "budget": per_dataset[d]["budget"],
            }
            for d in C.DATASETS
        },
        "sample_selection": {
            d: {
                split: {
                    "n_selected": per_dataset[d]["samples"]["splits"][split]["n_selected"],
                    "row_ids_sha256": per_dataset[d]["samples"]["splits"][split]["row_ids_sha256"],
                }
                for split in C.SPLITS
            }
            for d in C.DATASETS
        },
        "test_policy": "always",
        "test_role": "DEVELOPMENT_TEST (evaluated, never used for any selection in this round)",
        "execution_provenance": execution_provenance(),
    }
    if extra:
        payload.update(dict(extra))
    C.atomic_write_json(C.ANALYSIS_DIR / "run_manifest.json", payload)
    return payload


def write_protocol_freeze() -> dict[str, Any]:
    payload = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "generated_at": C.utcnow(),
        "frozen_constants": C.FROZEN_CONSTANTS,
        "freeze_hash": C.freeze_hash(),
        "code_sha256": C.code_hashes(),
        "source_round_identity": {
            "round": "lcrc-readout-context-pilot-small-v1",
            "arm": "A",
            "checkpoint_sha256": dict(C.EXPECTED_CHECKPOINT_SHA256),
            "trajectory_sha256": dict(C.EXPECTED_TRAJECTORY_SHA256),
        },
        "notes": {
            "intervention_name": C.CONDITION_NAME,
            "intervention_is_not": "a chemical bond deletion, a group deletion or a full counterfactual",
            "search_space_limitation": (
                "only connected subgraphs of 2..6 physical bonds were searched; nothing is claimed "
                "about structures larger than 6 bonds"
            ),
            "effect_floor": "1% of the Train target standard deviation is a pilot resolution scale, not a universal threshold",
        },
    }
    C.atomic_write_json(C.ANALYSIS_DIR / "protocol_freeze.json", payload)
    return payload


def decision_payload(main: Mapping[tuple[str, str], Mapping[str, Any]],
                     signals: Mapping[str, Mapping[str, Any]],
                     diag: Mapping[str, Any],
                     engineering_gate: str) -> dict[str, Any]:
    per_dataset = {}
    for dataset in C.DATASETS:
        per_dataset[dataset] = {
            "signal": signals[dataset],
            "verdict": "PILOT_SIGNAL" if signals[dataset]["pass"] else "WEAK_OR_INCONCLUSIVE",
        }
    overall = (
        "PILOT_SIGNAL"
        if engineering_gate == "PASS" and all(v["verdict"] == "PILOT_SIGNAL" for v in per_dataset.values())
        else "WEAK_OR_INCONCLUSIVE"
    )
    return {
        "round_id": C.ROUND_ID,
        "generated_at": C.utcnow(),
        "engineering_gate": engineering_gate,
        "discovery_signal_overall": overall,
        "discovery_signal_per_dataset": per_dataset,
        "architecture_diagnosis": diag,
        "rule_reminder": (
            "DISCOVERY_SIGNAL is a development investment rule fixed before the run. It is not a "
            "significance test and not a scientific conclusion."
        ),
    }


def run(datasets: Sequence[str] = C.DATASETS, ledger=None) -> dict[str, Any]:
    per_dataset = {d: load_dataset(d) for d in datasets}
    rows = build_rows(per_dataset)
    C.ensure_dir(C.ANALYSIS_DIR)

    # ---- tables ---------------------------------------------------------
    main: dict[tuple[str, str], dict[str, Any]] = {}
    main_rows = []
    for dataset in datasets:
        samples = per_dataset[dataset]["samples"]["splits"]
        for split in C.SPLITS:
            rec = main_table(rows, dataset, split, int(samples[split]["n_selected"]))
            main[(dataset, split)] = rec
            main_rows.append(rec)
    _write_csv(C.ANALYSIS_DIR / "main_table.csv",
               [{k: r[k] for k in MAIN_COLUMNS} for r in main_rows], MAIN_COLUMNS)

    diag: dict[tuple[str, str], dict[str, Any]] = {}
    diag_rows = []
    for dataset in datasets:
        for split in C.SPLITS:
            rec = diag_table(rows, dataset, split)
            diag[(dataset, split)] = rec
            diag_rows.append(rec)
    _write_csv(C.ANALYSIS_DIR / "architecture_diagnostics.csv",
               [{k: r[k] for k in DIAG_COLUMNS} for r in diag_rows], DIAG_COLUMNS)

    detail_columns = sorted({k for r in rows for k in r})
    _write_csv(C.ANALYSIS_DIR / "molecule_table.csv", rows, detail_columns)

    boot = bootstrap_payload(datasets, rows)
    C.atomic_write_json(C.ANALYSIS_DIR / "bootstrap.json", boot)

    failures = failure_cases(rows)
    C.atomic_write_json(C.ANALYSIS_DIR / "failure_cases.json", failures)

    preflight = merge_preflight(per_dataset)
    training = merge_training_source(per_dataset)
    freeze = write_protocol_freeze()
    del freeze

    signals = {d: signal_condition(main, d) for d in datasets}
    diag_payload = architecture_diagnosis(diag)
    decision = decision_payload(main, signals, diag_payload, preflight["engineering_gate"])
    C.atomic_write_json(C.ANALYSIS_DIR / "decision.json", decision)

    # ---- figures --------------------------------------------------------
    import as_figures as F

    figures: dict[str, Any] = {"molecules": [], "distributions": {}, "readout": {}}
    for dataset in datasets:
        recs = [m for split in C.SPLITS for m in per_dataset[dataset]["molecules"][split]]
        sel = F.figure_selection(recs)
        fig_dir = C.FIGURES_DIR / dataset
        for rec in sel["main"]:
            bond_index = list(zip(rec["gradient"]["atom_u"], rec["gradient"]["atom_v"]))
            name = f"{rec['split']}_row{rec['row_id']:05d}.png"
            title = _figure_title(rec, tag="fixed sample")
            info = F.render_molecule(rec, bond_index, fig_dir / name, title_lines=title)
            figures["molecules"].append({"dataset": dataset, "row_id": rec["row_id"],
                                         "split": rec["split"], "kind": "fixed_sample", **info})
        for label, cases in (("largest_response", sel["diagnostics"]["largest"]),
                             ("smallest_response", sel["diagnostics"]["smallest"])):
            for rec in cases:
                bond_index = list(zip(rec["gradient"]["atom_u"], rec["gradient"]["atom_v"]))
                name = f"diag_{label}_{rec['split']}_row{rec['row_id']:05d}.png"
                title = _figure_title(rec, tag=f"diagnostic ({label}, result-driven)")
                info = F.render_molecule(rec, bond_index, fig_dir / name, title_lines=title)
                figures["molecules"].append({"dataset": dataset, "row_id": rec["row_id"],
                                             "split": rec["split"], "kind": label, **info})
        figures["distributions"][dataset] = F.distribution_figure(
            rows, dataset, fig_dir / "distribution_top1.png"
        )
        figures["readout"][dataset] = F.readout_figure(rows, dataset, fig_dir / "readout_decomposition.png")
    C.atomic_write_json(C.ANALYSIS_DIR / "figures" / "index.json", figures)

    manifest = write_run_manifest(per_dataset, extra={"figures": {
        "n_molecule_figures": len(figures["molecules"]),
        "index": C.relpath(C.ANALYSIS_DIR / "figures" / "index.json"),
    }})

    return {
        "engineering_gate": preflight["engineering_gate"],
        "discovery_signal": {d: signals[d]["pass"] for d in datasets},
        "main_table": C.relpath(C.ANALYSIS_DIR / "main_table.csv"),
        "architecture_diagnostics": C.relpath(C.ANALYSIS_DIR / "architecture_diagnostics.csv"),
        "molecule_table": C.relpath(C.ANALYSIS_DIR / "molecule_table.csv"),
        "bootstrap": C.relpath(C.ANALYSIS_DIR / "bootstrap.json"),
        "decision": C.relpath(C.ANALYSIS_DIR / "decision.json"),
        "run_manifest": C.relpath(C.ANALYSIS_DIR / "run_manifest.json"),
        "figures_rendered": sum(1 for f in figures["molecules"] if f.get("rendered")),
        "figures_failed": sum(1 for f in figures["molecules"] if not f.get("rendered")),
        "training_source_verified": training["all_verified"],
        "main": main,
        "diag": diag,
        "signals": signals,
        "rows": rows,
        "manifest": manifest,
    }


def _figure_title(rec: Mapping[str, Any], *, tag: str) -> list[str]:
    import as_figures as F

    controls = (rec.get("matched_random") or {}).get("effects")
    return F._title_lines(rec, tag=tag, controls=controls)

