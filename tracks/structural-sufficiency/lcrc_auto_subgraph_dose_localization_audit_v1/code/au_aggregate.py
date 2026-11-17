"""Deterministic CPU aggregate stage: corrected tables, bootstrap, verdicts.

Every number is read back from the per-molecule records; nothing is re-derived
from a figure and nothing is typed by hand.  The pilot's numbers are recomputed
from the pilot's own per-molecule records with the pilot's own definitions, and
that recomputation is checked against the pilot's committed ``main_table.csv``
before any old-vs-corrected comparison is allowed to be reported.

The two protocols are reported side by side and are **never pooled**:

``pilot_internal_matched``  the pilot's null, matched on ``n_s2_internal``;
``corrected_affected_matched``  this round's null, matched on ``n_s2_affected``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import au_common as C
import au_controls as K


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_new(dataset: str) -> dict[str, Any]:
    base = C.RESULTS_DIR / dataset
    out: dict[str, Any] = {
        "dataset": dataset,
        "preflight": C.read_json(base / "preflight.json"),
        "samples": C.read_json(base / "samples.json"),
        "budget": C.read_json(base / "budget.json") if (base / "budget.json").is_file() else None,
        "run_summary": C.read_json(base / "run_summary.json"),
        "records": {},
    }
    for split in C.SPLITS:
        path = base / split / "records.json"
        out["records"][split] = C.read_json(path) if path.is_file() else []
    return out


def load_pilot(dataset: str) -> dict[str, Any]:
    paths = C.upstream_paths(dataset)
    return {
        "dataset": dataset,
        "samples": C.read_json(paths["samples"]),
        "preflight": C.read_json(paths["preflight"]),
        "main_rows": C.read_csv(C.UPSTREAM_ANALYSIS / "main_table.csv"),
        "molecules": {s: C.upstream_molecules(dataset, s) for s in C.SPLITS},
    }


# ---------------------------------------------------------------------------
# per-molecule flat rows (both protocols use the same field names)
# ---------------------------------------------------------------------------


def pilot_rows(dataset: str, split: str, records: Sequence[Mapping[str, Any]],
               train_std: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mol in records:
        topk = mol.get("topk") or []
        if not topk:
            continue
        top1 = topk[0]
        eff = top1["effect"]
        matched = (mol.get("matched_random") or {}).get("effects") or {}
        matched_effs = matched.get("effects") or []
        heur = (mol.get("local_heuristic") or {}).get("effect") or {}
        ratio = matched.get("effect_ratio")
        adv = matched.get("effect_advantage")
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "row_id": int(mol["row_id"]),
                "protocol": "pilot_internal_matched",
                "top1_n_bonds": int(top1["n_bonds"]),
                "top1_abs_delta": float(eff["abs_delta"]),
                "top1_signed_delta": float(eff["signed_delta"]),
                "top1_delta_L": float(eff["delta_L"]),
                "top1_delta_rho": float(eff["delta_P"]),
                "top1_abs_delta_over_train_std": float(eff["abs_delta_over_train_std"]),
                "top1_meets_effect_floor": bool(
                    float(eff["abs_delta"]) >= C.EFFECT_FLOOR_STD_FRACTION * float(train_std)
                ),
                "n_matched": int(matched.get("n_matched") or len(matched_effs)),
                "matched_abs": [float(e["abs_delta"]) for e in matched_effs],
                "matched_median_abs": (float(matched["median_abs_delta"]) if matched_effs else None),
                "effect_ratio": float(ratio) if ratio is not None else None,
                "effect_advantage": float(adv) if adv is not None else None,
                "ratio_gt1": bool(float(ratio) > 1.0) if ratio is not None else None,
                "heuristic_abs_delta": float(heur["abs_delta"]) if heur else None,
                "heuristic_bonds": list((mol.get("local_heuristic") or {}).get("bonds") or []),
                "matched_status": (mol.get("matched_random") or {}).get("status"),
                "n_s2_total": int(mol.get("n_s2_total", 0)),
            }
        )
    return rows


def corrected_rows(dataset: str, split: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mol in records:
        top1 = mol["top1"]
        eff = top1["effect"]
        controls = mol["controls"]
        summary = controls.get("summary")
        heur = mol.get("heuristic")
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "row_id": int(mol["row_id"]),
                "protocol": "corrected_affected_matched",
                "top1_n_bonds": int(top1["n_bonds"]),
                "top1_abs_delta": float(eff["abs_delta_total"]),
                "top1_signed_delta": float(eff["delta_total"]),
                "top1_delta_L": float(eff["delta_L"]),
                "top1_delta_rho": float(eff["delta_rho"]),
                "top1_abs_delta_over_train_std": float(eff["abs_delta_total_over_train_std"]),
                "top1_meets_effect_floor": bool(
                    float(eff["abs_delta_total"]) >= C.EFFECT_FLOOR_STD_FRACTION * float(mol["train_std"])
                ),
                "n_matched": int(controls.get("n_matched") or 0),
                "matched_abs": [float(r["effect"]["abs_delta_total"]) for r in controls.get("matched") or []],
                "matched_median_abs": float(summary["median_abs_delta"]) if summary else None,
                "effect_ratio": float(summary["effect_ratio"]) if summary else None,
                "effect_advantage": float(summary["effect_advantage"]) if summary else None,
                "ratio_gt1": bool(summary["ratio_gt1"]) if summary else None,
                "heuristic_abs_delta": float(heur["abs_delta"]) if heur else None,
                "heuristic_bonds": list(heur["bonds"]) if heur else [],
                "matched_status": controls.get("status"),
                "n_s2_total": int(mol["n_s2_total"]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _boot(values: Sequence[float], statistic: str = "median") -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if statistic == "median":
        return C.bootstrap_median_ci(arr)
    if statistic in ("mean", "proportion"):
        return C.bootstrap_mean_ci(arr)
    raise KeyError(statistic)


def protocol_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    interpreted = [r for r in rows if r["top1_abs_delta"] is not None]
    matched = [r for r in interpreted if r["effect_ratio"] is not None]
    heur = [r for r in interpreted if r["heuristic_abs_delta"] is not None]
    n_sel = len(rows)
    top1_abs = [r["top1_abs_delta"] for r in interpreted]
    top1_over_std = [r["top1_abs_delta_over_train_std"] for r in interpreted]
    ratios = [r["effect_ratio"] for r in matched]
    advs = [r["effect_advantage"] for r in matched]
    heur_adv = [r["top1_abs_delta"] - r["heuristic_abs_delta"] for r in heur]
    return {
        "n_molecules": int(n_sel),
        "n_interpreted": len(interpreted),
        "coverage": (len(interpreted) / n_sel) if n_sel else float("nan"),
        "top1_n_bonds_median": C.median([r["top1_n_bonds"] for r in interpreted]),
        "median_top1_abs_delta": C.median(top1_abs),
        "median_top1_abs_delta_over_train_std": C.median(top1_over_std),
        "frac_top1_ge_1pct_train_std": (
            float(np.mean([bool(r["top1_meets_effect_floor"]) for r in interpreted])) if interpreted else float("nan")
        ),
        "matched_n_molecules": len(matched),
        "matched_success_rate": (len(matched) / len(interpreted)) if interpreted else float("nan"),
        "matched_n_controls_total": int(sum(int(r["n_matched"]) for r in matched)),
        "insufficient_matched_null_n": int(sum(1 for r in interpreted if r["matched_status"] == "INSUFFICIENT_MATCHED_NULL")),
        "median_matched_abs_delta": C.median([r["matched_median_abs"] for r in matched]),
        "median_effect_ratio": C.median(ratios),
        "ratio_gt1_frac": (float(np.mean([bool(r["ratio_gt1"]) for r in matched])) if matched else float("nan")),
        "median_effect_advantage": C.median(advs),
        "median_local_heuristic_abs_delta": C.median([r["heuristic_abs_delta"] for r in heur]),
        "median_top1_minus_heuristic": C.median(heur_adv),
        "frac_top1_gt_local_heuristic": (float(np.mean([x > 0 for x in heur_adv])) if heur_adv else float("nan")),
        "_match": {
            "effect_ratio": _boot(ratios, "median"),
            "effect_advantage": _boot(advs, "median"),
            "ratio_gt1": _boot([float(bool(r["ratio_gt1"])) for r in matched], "proportion"),
            "top1_abs_delta_over_train_std": _boot(top1_over_std, "median"),
            "top1_minus_heuristic": _boot(heur_adv, "median"),
        },
    }


def dose_summary(rows: Sequence[Mapping[str, Any]], *, protocol: str) -> dict[str, Any]:
    """Distribution of the pilot's / corrected null's realised dose mismatch."""
    if protocol == "pilot_internal_matched":
        internal = [r["internal_rel_diff"] for r in rows]
        affected = [r["affected_rel_diff"] for r in rows]
        outside = [not bool(r["would_pass_affected_tolerance"]) for r in rows]
        top1_i = [r["top1_n_s2_internal"] for r in rows]
        top1_a = [r["top1_n_s2_affected"] for r in rows]
        diff = [r["affected_minus_internal"] for r in rows]
    else:
        internal = [r["internal_rel_diff"] for r in rows]
        affected = [r["affected_rel_diff"] for r in rows]
        outside = [not bool(r["accepted_affected"]) for r in rows]
        top1_i = [r["top1_n_s2_internal"] for r in rows]
        top1_a = [r["top1_n_s2_affected"] for r in rows]
        diff = [r["n_s2_affected"] - r["n_s2_internal"] for r in rows]
    return {
        "protocol": protocol,
        "n_controls": int(len(rows)),
        "median_internal_rel_diff": C.median(internal),
        "max_internal_rel_diff": float(np.max(internal)) if internal else float("nan"),
        "median_affected_rel_diff": C.median(affected),
        "q75_affected_rel_diff": C.quantiles(affected).get("q75"),
        "p95_affected_rel_diff": C.quantiles(affected).get("q95"),
        "max_affected_rel_diff": float(np.max(affected)) if affected else float("nan"),
        "frac_controls_outside_affected_tol": float(np.mean(outside)) if outside else float("nan"),
        "median_affected_minus_internal": C.median(diff),
        "median_top1_n_s2_internal": C.median(top1_i),
        "median_top1_n_s2_affected": C.median(top1_a),
    }


# ---------------------------------------------------------------------------
# localisation
# ---------------------------------------------------------------------------


def localization_rows(dataset: str, split: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for mol in records:
        loc = mol["localization"]
        dose = mol["top1"]["dose"]
        out.append(
            {
                "dataset": dataset,
                "split": split,
                "row_id": int(mol["row_id"]),
                "n_atoms": int(loc["n_atoms"]),
                "n_atoms_inside": int(loc["n_atoms_inside"]),
                "n_atoms_outside": int(loc["n_atoms_outside"]),
                "top1_n_bonds": int(mol["top1"]["n_bonds"]),
                "top1_n_atoms": int(mol["top1"]["n_atoms_involved"]),
                "top1_score_A": float(mol["top1"]["score_A"]),
                "top1_abs_delta_total": float(loc["abs_delta_total"]),
                "top1_signed_delta_total": float(loc["delta_total"]),
                "delta_L": float(loc["delta_L"]),
                "delta_rho": float(loc["delta_rho"]),
                "inside_abs_response": float(loc["inside_abs_response"]),
                "outside_abs_response": float(loc["outside_abs_response"]),
                "total_abs_response": float(loc["total_abs_response"]),
                "outside_response_fraction": float(loc["outside_response_fraction"]),
                "n_root_inside_moved": int(loc["n_root_inside_moved"]),
                "n_root_outside_moved": int(loc["n_root_outside_moved"]),
                "n_s2_total": int(dose["n_s2_total"]),
                "n_s2_internal": int(dose["n_s2_internal"]),
                "n_s2_boundary": int(dose["n_s2_boundary"]),
                "n_s2_affected": int(dose["n_s2_affected"]),
                "n_s2_unaffected": int(dose["n_s2_unaffected"]),
                "n_s2_boundary_first_masked": int(dose["n_s2_boundary_first_masked"]),
                "n_s2_boundary_second_masked": int(dose["n_s2_boundary_second_masked"]),
                "n_s2_affected_root_inside": int(dose["n_s2_affected_root_inside"]),
                "n_s2_affected_root_outside": int(dose["n_s2_affected_root_outside"]),
                "boundary_over_affected": float(dose["boundary_over_affected"]),
                "internal_over_total": float(dose["internal_over_total"]),
                "boundary_over_total": float(dose["boundary_over_total"]),
                "affected_over_total": float(dose["affected_over_total"]),
                "affected_over_internal": float(dose["affected_over_internal"]),
                "affected_root_outside_fraction": float(dose["affected_root_outside_fraction"]),
                "closure_local_sum_error": float(loc["closure_local_sum_error"]),
                "closure_local_sum_tolerance": float(loc["closure_local_sum_tolerance"]),
                "closure_total_error": float(loc["closure_total_error"]),
                "closure_total_tolerance": float(loc["closure_total_tolerance"]),
                "n_matched": int(mol["controls"]["n_matched"]),
                "controls_status": mol["controls"]["status"],
                "outside_fraction_gt_0p5": bool(float(loc["outside_response_fraction"]) > 0.5),
                "outside_fraction_le_0p25": bool(float(loc["outside_response_fraction"]) <= 0.25),
            }
        )
    return out


LOCALIZATION_CORRELATES = (
    ("top1_n_bonds", "top1_n_bonds"),
    ("abs_score_A", "top1_score_A"),
    ("abs_delta_total", "top1_abs_delta_total"),
    ("n_s2_affected", "n_s2_affected"),
    ("boundary_over_affected", "boundary_over_affected"),
    ("affected_root_outside_fraction", "affected_root_outside_fraction"),
    ("affected_over_internal", "affected_over_internal"),
)


def localization_summary(dataset: str, split: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frac = [r["outside_response_fraction"] for r in rows]
    inside = [r["inside_abs_response"] for r in rows]
    outside = [r["outside_abs_response"] for r in rows]
    q = C.quantiles(frac, qs=(0.05, 0.25, 0.50, 0.75, 0.95))
    median_frac = q.get("q50", float("nan"))
    if not np.isfinite(median_frac):
        verdict = "UNDEFINED"
    elif median_frac <= C.LOCALIZATION_CONCENTRATED_MAX:
        verdict = "LOCAL_CONCENTRATED"
    elif median_frac > C.LOCALIZATION_SPREAD_MIN:
        verdict = "SPREAD_TO_SURROUNDING_ROOTS"
    else:
        verdict = "MIXED"
    corr = {}
    for name, field in LOCALIZATION_CORRELATES:
        x = [abs(float(r[field])) if name == "abs_score_A" else float(r[field]) for r in rows]
        y = [float(v) for v in frac]
        corr[name] = C.spearman(x, y)
    return {
        "dataset": dataset,
        "split": split,
        "n": len(rows),
        "outside_fraction_median": median_frac,
        "outside_fraction_q05": q.get("q05"),
        "outside_fraction_q25": q.get("q25"),
        "outside_fraction_q75": q.get("q75"),
        "outside_fraction_q95": q.get("q95"),
        "outside_fraction_mean": float(np.mean(frac)) if frac else float("nan"),
        "frac_outside_gt_0p5": float(np.mean([x > 0.5 for x in frac])) if frac else float("nan"),
        "frac_outside_le_0p25": float(np.mean([x <= 0.25 for x in frac])) if frac else float("nan"),
        "median_inside_abs_response": C.median(inside),
        "median_outside_abs_response": C.median(outside),
        "median_total_abs_response": C.median([r["total_abs_response"] for r in rows]),
        "median_delta_L": C.median([r["delta_L"] for r in rows]),
        "median_delta_rho": C.median([r["delta_rho"] for r in rows]),
        "median_abs_delta_total": C.median([r["top1_abs_delta_total"] for r in rows]),
        "median_top1_n_bonds": C.median([r["top1_n_bonds"] for r in rows]),
        "median_n_s2_internal": C.median([r["n_s2_internal"] for r in rows]),
        "median_n_s2_boundary": C.median([r["n_s2_boundary"] for r in rows]),
        "median_n_s2_affected": C.median([r["n_s2_affected"] for r in rows]),
        "median_boundary_over_affected": C.median([r["boundary_over_affected"] for r in rows]),
        "median_internal_over_total": C.median([r["internal_over_total"] for r in rows]),
        "median_boundary_over_total": C.median([r["boundary_over_total"] for r in rows]),
        "median_affected_over_total": C.median([r["affected_over_total"] for r in rows]),
        "median_affected_root_outside_fraction": C.median([r["affected_root_outside_fraction"] for r in rows]),
        "frac_root_inside_moved": C.median(
            [r["n_root_inside_moved"] / r["n_atoms_inside"] for r in rows if r["n_atoms_inside"]]
        ),
        "frac_root_outside_moved": C.median(
            [r["n_root_outside_moved"] / r["n_atoms_outside"] for r in rows if r["n_atoms_outside"]]
        ),
        "verdict": verdict,
        "spearman_vs_outside_fraction": corr,
        "bootstrap_outside_fraction_median": C.bootstrap_median_ci(frac),
        "note": (
            "inside/outside is where the *local readout response* lands, not a unique or "
            "independent chemical contribution assignment"
        ),
    }


def localization_bins(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Outside fraction grouped by Top-1 bond count (the brief asks for the relation)."""
    out = []
    for k in sorted({int(r["top1_n_bonds"]) for r in rows}):
        sub = [r for r in rows if int(r["top1_n_bonds"]) == k]
        out.append(
            {
                "top1_n_bonds": int(k),
                "n": len(sub),
                "outside_fraction_median": C.median([r["outside_response_fraction"] for r in sub]),
                "median_abs_delta_total": C.median([r["top1_abs_delta_total"] for r in sub]),
                "median_n_s2_affected": C.median([r["n_s2_affected"] for r in sub]),
                "median_boundary_over_affected": C.median([r["boundary_over_affected"] for r in sub]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


OLD_VS_COLUMNS = [
    "dataset", "split", "protocol", "dose_quantity",
    "n_molecules", "n_interpreted", "matched_n_molecules", "matched_success_rate",
    "matched_n_controls_total", "insufficient_matched_null_n",
    "top1_n_bonds_median",
    "median_top1_abs_delta", "median_top1_abs_delta_over_train_std",
    "frac_top1_ge_1pct_train_std",
    "median_matched_abs_delta", "median_effect_ratio", "ratio_gt1_frac",
    "median_effect_advantage",
    "effect_ratio_ci_low", "effect_ratio_ci_high",
    "effect_advantage_ci_low", "effect_advantage_ci_high",
    "ratio_gt1_ci_low", "ratio_gt1_ci_high",
    "median_local_heuristic_abs_delta", "median_top1_minus_heuristic",
    "frac_top1_gt_local_heuristic",
    "median_control_internal_rel_diff", "median_control_affected_rel_diff",
    "frac_controls_outside_affected_tol",
    "median_top1_n_s2_internal", "median_top1_n_s2_affected",
    "pilot_metric_replay_passed", "note",
]

MATCHED_CONTROL_COLUMNS = [
    "dataset", "split", "row_id", "kind", "control_index", "bonds", "n_bonds",
    "n_s2_internal", "n_s2_boundary", "n_s2_affected", "n_s2_unaffected", "n_s2_total",
    "affected_rel_diff", "internal_rel_diff", "boundary_over_affected",
    "affected_root_outside_fraction",
    "abs_delta_total", "signed_delta_total", "delta_L", "delta_rho",
    "closure_total_error", "closure_total_tolerance",
    "controls_status", "accepted",
]

LOCALIZATION_COLUMNS = [
    "dataset", "split", "row_id", "n_atoms", "n_atoms_inside", "n_atoms_outside",
    "top1_n_bonds", "top1_n_atoms", "top1_score_A", "top1_abs_delta_total",
    "top1_signed_delta_total", "delta_L", "delta_rho",
    "inside_abs_response", "outside_abs_response", "total_abs_response",
    "outside_response_fraction", "n_root_inside_moved", "n_root_outside_moved",
    "n_s2_total", "n_s2_internal", "n_s2_boundary", "n_s2_affected", "n_s2_unaffected",
    "n_s2_boundary_first_masked", "n_s2_boundary_second_masked",
    "n_s2_affected_root_inside", "n_s2_affected_root_outside",
    "boundary_over_affected", "internal_over_total", "boundary_over_total",
    "affected_over_total", "affected_over_internal",
    "affected_root_outside_fraction",
    "closure_local_sum_error", "closure_local_sum_tolerance",
    "closure_total_error", "closure_total_tolerance",
    "n_matched", "controls_status", "outside_fraction_gt_0p5", "outside_fraction_le_0p25",
]

OLD_CONTROL_COLUMNS = [
    "dataset", "split", "row_id", "control_index", "bonds", "n_bonds",
    "n_s2_internal", "n_s2_boundary", "n_s2_affected",
    "internal_rel_diff", "affected_rel_diff", "affected_minus_internal",
    "would_pass_affected_tolerance", "top1_n_s2_internal", "top1_n_s2_affected",
]


def matched_control_rows(dataset: str, split: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mol in records:
        top1 = mol["top1"]
        dose = top1["dose"]
        effect = top1["effect"]
        rows.append(
            {
                "dataset": dataset, "split": split, "row_id": int(mol["row_id"]),
                "kind": "top1", "control_index": -1,
                "bonds": " ".join(str(b) for b in top1["bonds"]),
                "n_bonds": int(top1["n_bonds"]),
                "n_s2_internal": int(dose["n_s2_internal"]),
                "n_s2_boundary": int(dose["n_s2_boundary"]),
                "n_s2_affected": int(dose["n_s2_affected"]),
                "n_s2_unaffected": int(dose["n_s2_unaffected"]),
                "n_s2_total": int(dose["n_s2_total"]),
                "affected_rel_diff": 0.0,
                "internal_rel_diff": 0.0,
                "boundary_over_affected": float(dose["boundary_over_affected"]),
                "affected_root_outside_fraction": float(dose["affected_root_outside_fraction"]),
                "abs_delta_total": float(effect["abs_delta_total"]),
                "signed_delta_total": float(effect["delta_total"]),
                "delta_L": float(effect["delta_L"]),
                "delta_rho": float(effect["delta_rho"]),
                "closure_total_error": float(effect["closure_total_error"]),
                "closure_total_tolerance": float(effect["closure_total_tolerance"]),
                "controls_status": mol["controls"]["status"],
                "accepted": True,
            }
        )
        for rec in mol["controls"].get("matched") or []:
            eff = rec["effect"]
            rows.append(
                {
                    "dataset": dataset, "split": split, "row_id": int(mol["row_id"]),
                    "kind": "matched_random", "control_index": int(rec["index"]),
                    "bonds": " ".join(str(b) for b in rec["bonds"]),
                    "n_bonds": int(rec["n_bonds"]),
                    "n_s2_internal": int(rec["n_s2_internal"]),
                    "n_s2_boundary": int(rec["n_s2_boundary"]),
                    "n_s2_affected": int(rec["n_s2_affected"]),
                    "n_s2_unaffected": int(mol["n_s2_total"]) - int(rec["n_s2_affected"]),
                    "n_s2_total": int(mol["n_s2_total"]),
                    "affected_rel_diff": float(rec["affected_rel_diff"]),
                    "internal_rel_diff": float(rec["internal_rel_diff"]),
                    "boundary_over_affected": float(rec["boundary_over_affected"]),
                    "affected_root_outside_fraction": float(rec["affected_root_outside_fraction"]),
                    "abs_delta_total": float(eff["abs_delta_total"]),
                    "signed_delta_total": float(eff["delta_total"]),
                    "delta_L": float(eff["delta_L"]),
                    "delta_rho": float(eff["delta_rho"]),
                    "closure_total_error": float(eff["closure_total_error"]),
                    "closure_total_tolerance": float(eff["closure_total_tolerance"]),
                    "controls_status": mol["controls"]["status"],
                    "accepted": True,
                }
            )
    return rows


def old_control_rows(dataset: str, split: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mol in records:
        for rec in (mol["old_controls_dose"] or {}).get("controls") or []:
            rows.append(
                {
                    "dataset": dataset, "split": split, "row_id": int(mol["row_id"]),
                    "control_index": int(rec["index"]),
                    "bonds": " ".join(str(b) for b in rec["bonds"]),
                    "n_bonds": int(rec["n_bonds"]),
                    "n_s2_internal": int(rec["n_s2_internal"]),
                    "n_s2_boundary": int(rec["n_s2_boundary"]),
                    "n_s2_affected": int(rec["n_s2_affected"]),
                    "internal_rel_diff": float(rec["internal_rel_diff"]),
                    "affected_rel_diff": float(rec["affected_rel_diff"]),
                    "affected_minus_internal": int(rec["affected_minus_internal"]),
                    "would_pass_affected_tolerance": bool(rec["would_pass_affected_tolerance"]),
                    "top1_n_s2_internal": int(rec["top1_n_s2_internal"]),
                    "top1_n_s2_affected": int(rec["top1_n_s2_affected"]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# old-metric replay + old vs corrected
# ---------------------------------------------------------------------------


def replay_pilot_metrics(dataset: str, pilot: Mapping[str, Any], train_std: float) -> dict[str, Any]:
    """Recompute the pilot's own metrics from its records and check its table."""
    table = {(r["dataset"], r["split"]): r for r in pilot["main_rows"]}
    out: dict[str, Any] = {"dataset": dataset, "splits": {}, "pass": True}
    for split in C.SPLITS:
        rows = pilot_rows(dataset, split, pilot["molecules"][split], train_std)
        metrics = protocol_metrics(rows)
        stored = table.get((dataset, split))
        checks = {}
        if stored is not None:
            pairs = {
                "matched_n_molecules": metrics["matched_n_molecules"],
                "matched_success_rate": metrics["matched_success_rate"],
                "matched_median_effect_ratio": metrics["median_effect_ratio"],
                "matched_ratio_gt1_frac": metrics["ratio_gt1_frac"],
                "matched_median_effect_advantage": metrics["median_effect_advantage"],
                "matched_median_abs_delta_random": metrics["median_matched_abs_delta"],
                "abs_delta_top1_median": metrics["median_top1_abs_delta"],
                "local_heuristic_median_abs_delta": metrics["median_local_heuristic_abs_delta"],
                "local_heuristic_median_advantage": metrics["median_top1_minus_heuristic"],
                "frac_top1_gt_local_heuristic": metrics["frac_top1_gt_local_heuristic"],
                "frac_top1_ge_1pct_train_std": metrics["frac_top1_ge_1pct_train_std"],
            }
            for field, value in pairs.items():
                recorded = float(stored[field])
                err = abs(float(value) - recorded)
                checks[field] = {
                    "recomputed": float(value),
                    "pilot_table": recorded,
                    "abs_err": float(err),
                    "passed": bool(err <= 1e-9 * max(1.0, abs(recorded))),
                }
            split_ok = bool(all(c["passed"] for c in checks.values()))
        else:
            split_ok = False
            checks["missing_pilot_row"] = {"passed": False}
        out["splits"][split] = {"checks": checks, "passed": split_ok, "metrics": metrics}
        out["pass"] = bool(out["pass"] and split_ok)
    return out


def build_old_vs_corrected(dataset: str, pilot: Mapping[str, Any], new: Mapping[str, Any],
                           train_std: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay = replay_pilot_metrics(dataset, pilot, train_std)
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {"dataset": dataset, "replay": replay, "splits": {}}
    for split in C.SPLITS:
        p_rows = pilot_rows(dataset, split, pilot["molecules"][split], train_std)
        c_rows = corrected_rows(dataset, split, new["records"][split])
        p_metrics = protocol_metrics(p_rows)
        c_metrics = protocol_metrics(c_rows)
        p_dose = _flatten_old_dose_summary(pilot["molecules"][split])
        c_dose = _corrected_dose_summary(new["records"][split])
        for protocol, metrics, dose, note in (
            ("pilot_internal_matched", p_metrics, p_dose,
             "pilot protocol: null matched on n_s2_internal (e1 AND e2 in S); reused, not re-run"),
            ("corrected_affected_matched", c_metrics, c_dose,
             "corrected protocol: null matched on n_s2_affected (e1 OR e2 in S); different protocol, never pooled with the pilot row"),
        ):
            boot = metrics["_match"]
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "protocol": protocol,
                    "dose_quantity": "n_s2_internal" if protocol.startswith("pilot") else "n_s2_affected",
                    "n_molecules": metrics["n_molecules"],
                    "n_interpreted": metrics["n_interpreted"],
                    "matched_n_molecules": metrics["matched_n_molecules"],
                    "matched_success_rate": metrics["matched_success_rate"],
                    "matched_n_controls_total": metrics["matched_n_controls_total"],
                    "insufficient_matched_null_n": metrics["insufficient_matched_null_n"],
                    "top1_n_bonds_median": metrics["top1_n_bonds_median"],
                    "median_top1_abs_delta": metrics["median_top1_abs_delta"],
                    "median_top1_abs_delta_over_train_std": metrics["median_top1_abs_delta_over_train_std"],
                    "frac_top1_ge_1pct_train_std": metrics["frac_top1_ge_1pct_train_std"],
                    "median_matched_abs_delta": metrics["median_matched_abs_delta"],
                    "median_effect_ratio": metrics["median_effect_ratio"],
                    "ratio_gt1_frac": metrics["ratio_gt1_frac"],
                    "median_effect_advantage": metrics["median_effect_advantage"],
                    "effect_ratio_ci_low": boot["effect_ratio"]["ci_low"],
                    "effect_ratio_ci_high": boot["effect_ratio"]["ci_high"],
                    "effect_advantage_ci_low": boot["effect_advantage"]["ci_low"],
                    "effect_advantage_ci_high": boot["effect_advantage"]["ci_high"],
                    "ratio_gt1_ci_low": boot["ratio_gt1"]["ci_low"],
                    "ratio_gt1_ci_high": boot["ratio_gt1"]["ci_high"],
                    "median_local_heuristic_abs_delta": metrics["median_local_heuristic_abs_delta"],
                    "median_top1_minus_heuristic": metrics["median_top1_minus_heuristic"],
                    "frac_top1_gt_local_heuristic": metrics["frac_top1_gt_local_heuristic"],
                    "median_control_internal_rel_diff": dose["median_internal_rel_diff"],
                    "median_control_affected_rel_diff": dose["median_affected_rel_diff"],
                    "frac_controls_outside_affected_tol": dose["frac_controls_outside_affected_tol"],
                    "median_top1_n_s2_internal": dose["median_top1_n_s2_internal"],
                    "median_top1_n_s2_affected": dose["median_top1_n_s2_affected"],
                    "pilot_metric_replay_passed": bool(replay["pass"]),
                    "note": note,
                }
            )
        paired = _paired_change(p_rows, c_rows)
        detail["splits"][split] = {
            "pilot": p_metrics,
            "corrected": c_metrics,
            "pilot_dose": p_dose,
            "corrected_dose": c_dose,
            "paired_change": paired,
            "pilot_replay": replay["splits"][split],
        }
    detail["old_vs_corrected_rows"] = rows
    return rows, detail


def _flatten_old_dose(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mol in records:
        for rec in (mol["old_controls_dose"] or {}).get("controls") or []:
            out.append(
                {
                    "internal_rel_diff": float(rec["internal_rel_diff"]),
                    "affected_rel_diff": float(rec["affected_rel_diff"]),
                    "would_pass_affected_tolerance": bool(rec["would_pass_affected_tolerance"]),
                    "top1_n_s2_internal": int(rec["top1_n_s2_internal"]),
                    "top1_n_s2_affected": int(rec["top1_n_s2_affected"]),
                    "affected_minus_internal": int(rec["affected_minus_internal"]),
                }
            )
    return out


def _flatten_old_dose_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dose_summary(_flatten_old_dose(records), protocol="pilot_internal_matched")


def _corrected_dose_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for mol in records:
        top1 = mol["top1"]["dose"]
        for rec in mol["controls"].get("matched") or []:
            rows.append(
                {
                    "internal_rel_diff": float(rec["internal_rel_diff"]),
                    "affected_rel_diff": float(rec["affected_rel_diff"]),
                    "accepted_affected": True,
                    "top1_n_s2_internal": int(top1["n_s2_internal"]),
                    "top1_n_s2_affected": int(top1["n_s2_affected"]),
                    "n_s2_affected": int(rec["n_s2_affected"]),
                    "n_s2_internal": int(rec["n_s2_internal"]),
                }
            )
    return dose_summary(rows, protocol="corrected_affected_matched")


def _paired_change(p_rows: Sequence[Mapping[str, Any]],
                   c_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {int(r["row_id"]): r for r in c_rows}
    pairs = []
    for r in p_rows:
        c = by_id.get(int(r["row_id"]))
        if c is None or r["effect_ratio"] is None or c["effect_ratio"] is None:
            continue
        pairs.append(
            {
                "row_id": int(r["row_id"]),
                "ratio_pilot": float(r["effect_ratio"]),
                "ratio_corrected": float(c["effect_ratio"]),
                "ratio_delta": float(c["effect_ratio"]) - float(r["effect_ratio"]),
                "advantage_pilot": float(r["effect_advantage"]),
                "advantage_corrected": float(c["effect_advantage"]),
                "advantage_delta": float(c["effect_advantage"]) - float(r["effect_advantage"]),
            }
        )
    ratio_delta = [p["ratio_delta"] for p in pairs]
    adv_delta = [p["advantage_delta"] for p in pairs]
    return {
        "n_paired_molecules": len(pairs),
        "ratio_delta_median": C.median(ratio_delta),
        "ratio_delta_bootstrap": C.bootstrap_median_ci(ratio_delta),
        "advantage_delta_median": C.median(adv_delta),
        "advantage_delta_bootstrap": C.bootstrap_median_ci(adv_delta),
        "frac_ratio_decreased": float(np.mean([d < 0 for d in ratio_delta])) if ratio_delta else float("nan"),
        "n_ratio_decreased": int(sum(1 for d in ratio_delta if d < 0)),
        "n_ratio_increased": int(sum(1 for d in ratio_delta if d > 0)),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def dose_verdict(dataset: str, pilot: Mapping[str, Any]) -> dict[str, Any]:
    rows = _flatten_old_dose_summary([m for s in C.SPLITS for m in pilot["molecules"][s]])
    frac = float(rows["frac_controls_outside_affected_tol"])
    median_rel = float(rows["median_affected_rel_diff"])
    substantive = bool(
        frac >= C.DOSE_DEFECT_SUBSTANTIVE_CONTROL_FRACTION or median_rel > C.S2_AFFECTED_REL_TOL
    )
    return {
        "dataset": dataset,
        "grade": "SUBSTANTIVE" if substantive else ("MARGINAL" if frac > 0 else "NONE"),
        "frac_old_controls_outside_affected_tol": frac,
        "median_old_control_affected_rel_diff": median_rel,
        "median_old_control_internal_rel_diff": float(rows["median_internal_rel_diff"]),
        "median_old_control_affected_minus_internal": float(rows["median_affected_minus_internal"]),
        "median_top1_n_s2_internal": float(rows["median_top1_n_s2_internal"]),
        "median_top1_n_s2_affected": float(rows["median_top1_n_s2_affected"]),
        "rule": {
            "substantive_if_control_fraction_outside_tol_at_least": C.DOSE_DEFECT_SUBSTANTIVE_CONTROL_FRACTION,
            "or_median_affected_rel_diff_gt": C.S2_AFFECTED_REL_TOL,
        },
        "n_old_controls": int(rows["n_controls"]),
    }


def signal_verdict(metrics: Mapping[str, Any], split: str) -> dict[str, Any]:
    conditions = {
        "effect_ratio_median_ge_1p5": {
            "value": float(metrics["median_effect_ratio"]),
            "rule": f">= {C.SIGNAL_EFFECT_RATIO_MIN}",
            "passed": bool(
                np.isfinite(metrics["median_effect_ratio"])
                and float(metrics["median_effect_ratio"]) >= C.SIGNAL_EFFECT_RATIO_MIN
            ),
        },
        "effect_ratio_gt1_fraction_ge_0p70": {
            "value": float(metrics["ratio_gt1_frac"]),
            "rule": f">= {C.SIGNAL_EFFECT_RATIO_FRACTION_MIN}",
            "passed": bool(
                np.isfinite(metrics["ratio_gt1_frac"])
                and float(metrics["ratio_gt1_frac"]) >= C.SIGNAL_EFFECT_RATIO_FRACTION_MIN
            ),
        },
        "effect_floor_fraction_ge_0p50": {
            "value": float(metrics["frac_top1_ge_1pct_train_std"]),
            "rule": f">= {C.SIGNAL_EFFECT_FLOOR_FRACTION_MIN}",
            "passed": bool(
                np.isfinite(metrics["frac_top1_ge_1pct_train_std"])
                and float(metrics["frac_top1_ge_1pct_train_std"]) >= C.SIGNAL_EFFECT_FLOOR_FRACTION_MIN
            ),
        },
        "enough_matched_controls": {
            "value": float(metrics["matched_n_molecules"]),
            "rule": f">= max(32, {C.MIN_MATCHED_COVERAGE_FRACTION:.0%} of interpreted molecules)",
            "passed": bool(
                int(metrics["matched_n_molecules"])
                >= max(32, int(np.ceil(C.MIN_MATCHED_COVERAGE_FRACTION * max(1, metrics["n_interpreted"]))))
            ),
        },
    }
    ratio_ci_low = metrics["_match"]["effect_ratio"]["ci_low"]
    if not np.isfinite(float(metrics["median_effect_ratio"])):
        verdict = "UNDEFINED"
    elif all(c["passed"] for c in conditions.values()) and float(ratio_ci_low) > 1.0:
        verdict = "SURVIVES"
    elif float(metrics["median_effect_ratio"]) > 1.0 and float(ratio_ci_low) >= 1.0:
        verdict = "ATTENUATED"
    else:
        verdict = "NOT_SUPPORTED"
    return {
        "split": split,
        "conditions": conditions,
        "all_conditions_passed": bool(all(c["passed"] for c in conditions.values())),
        "effect_ratio_ci_low": float(ratio_ci_low),
        "median_effect_ratio": float(metrics["median_effect_ratio"]),
        "ratio_gt1_frac": float(metrics["ratio_gt1_frac"]),
        "matched_success_rate": float(metrics["matched_success_rate"]),
        "median_effect_advantage": float(metrics["median_effect_advantage"]),
        "effect_advantage_ci_low": float(metrics["_match"]["effect_advantage"]["ci_low"]),
        "effect_advantage_ci_high": float(metrics["_match"]["effect_advantage"]["ci_high"]),
        "verdict": verdict,
    }


def build_decision(datasets: Sequence[str], detail: Mapping[str, Any],
                   localization: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "upstream_round_id": C.UPSTREAM_ROUND_ID,
        "generated_at": C.utcnow(),
        "rules_frozen_before_the_run": {
            "dose_defect": C.FROZEN_CONSTANTS["dose_defect_rule"],
            "signal": C.FROZEN_CONSTANTS["signal_rule"],
            "localization": {
                "concentrated_max": C.LOCALIZATION_CONCENTRATED_MAX,
                "spread_min": C.LOCALIZATION_SPREAD_MIN,
            },
        },
        "datasets": {},
    }
    for dataset in datasets:
        d = detail[dataset]
        entry = {
            "dose_matching_defect": d["dose_verdict"],
            "pilot_replay_passed": bool(d["build"]["replay"]["pass"]),
            "splits": {},
            "pilot_vs_corrected": {},
        }
        for split in C.SPLITS:
            s = d["build"]["splits"][split]
            entry["splits"][split] = {
                "pilot": signal_verdict(s["pilot"], split),
                "corrected": signal_verdict(s["corrected"], split),
                "paired_change": {
                    "n_paired_molecules": s["paired_change"]["n_paired_molecules"],
                    "ratio_delta_median": s["paired_change"]["ratio_delta_median"],
                    "ratio_delta_ci": s["paired_change"]["ratio_delta_bootstrap"],
                    "advantage_delta_median": s["paired_change"]["advantage_delta_median"],
                    "frac_ratio_decreased": s["paired_change"]["frac_ratio_decreased"],
                },
            }
        entry["localization"] = {
            split: {
                "verdict": localization[dataset]["splits"][split]["verdict"],
                "outside_fraction_median": localization[dataset]["splits"][split]["outside_fraction_median"],
                "frac_outside_gt_0p5": localization[dataset]["splits"][split]["frac_outside_gt_0p5"],
                "median_boundary_over_affected": localization[dataset]["splits"][split]["median_boundary_over_affected"],
                "median_affected_root_outside_fraction": localization[dataset]["splits"][split]["median_affected_root_outside_fraction"],
            }
            for split in C.SPLITS
        }
        entry["top1_vs_heuristic"] = {
            split: {
                "pilot_frac_top1_gt_heuristic": d["build"]["splits"][split]["pilot"]["frac_top1_gt_local_heuristic"],
                "corrected_frac_top1_gt_heuristic": d["build"]["splits"][split]["corrected"]["frac_top1_gt_local_heuristic"],
                "corrected_median_top1_minus_heuristic": d["build"]["splits"][split]["corrected"]["median_top1_minus_heuristic"],
                "heuristic_source": "pilot local_heuristic.effect reused verbatim (no re-search)",
            }
            for split in C.SPLITS
        }
        out["datasets"][dataset] = entry
    out["summary"] = summarise_answers(out)
    return out


def summarise_answers(decision: Mapping[str, Any]) -> dict[str, Any]:
    """The five brief questions, answered only from the frozen rules."""
    answers: dict[str, Any] = {"q1_dose_matching_defect": {}, "q2_corrected_effect": {},
                               "q3_top1_vs_controls": {}, "q4_localization": {}, "q5_scope": {}}
    for dataset, entry in decision["datasets"].items():
        answers["q1_dose_matching_defect"][dataset] = {
            "grade": entry["dose_matching_defect"]["grade"],
            "frac_old_controls_outside_affected_tol": entry["dose_matching_defect"]["frac_old_controls_outside_affected_tol"],
            "median_old_control_affected_rel_diff": entry["dose_matching_defect"]["median_old_control_affected_rel_diff"],
            "median_top1_n_s2_internal": entry["dose_matching_defect"]["median_top1_n_s2_internal"],
            "median_top1_n_s2_affected": entry["dose_matching_defect"]["median_top1_n_s2_affected"],
        }
        answers["q2_corrected_effect"][dataset] = {
            split: {
                "effect_ratio_median": entry["splits"][split]["corrected"]["conditions"]["effect_ratio_median_ge_1p5"]["value"],
                "ratio_gt1_fraction": entry["splits"][split]["corrected"]["conditions"]["effect_ratio_gt1_fraction_ge_0p70"]["value"],
                "effect_ratio_ci_low": entry["splits"][split]["corrected"]["effect_ratio_ci_low"],
                "median_effect_advantage": entry["splits"][split]["corrected"].get("median_effect_advantage"),
                "verdict": entry["splits"][split]["corrected"]["verdict"],
            }
            for split in ("valid", "test")
        }
        answers["q3_top1_vs_controls"][dataset] = {
            "corrected_verdict_valid": entry["splits"]["valid"]["corrected"]["verdict"],
            "corrected_verdict_test": entry["splits"]["test"]["corrected"]["verdict"],
            "frac_top1_gt_heuristic_valid": entry["top1_vs_heuristic"]["valid"]["corrected_frac_top1_gt_heuristic"],
            "frac_top1_gt_heuristic_test": entry["top1_vs_heuristic"]["test"]["corrected_frac_top1_gt_heuristic"],
        }
        answers["q4_localization"][dataset] = {
            split: entry["localization"][split] for split in C.SPLITS
        }
    answers["q5_scope"] = {
        "supported": [
            "the pilot's null was matched on n_s2_internal, a different quantity from the S2 dose the gate removes",
            "the corrected affected-matched null reproduces on the frozen function (n_s2_affected is exactly the 1->0 weight set)",
            "the per-molecule effect ratio / advantage under the corrected null, with molecule-level bootstrap",
            "where the masked local readout moves: inside the highlighted atoms vs the remaining roots",
        ],
        "not_supported": [
            "any cross-seed or cross-checkpoint stability claim (single frozen checkpoint)",
            "any chemical counterfactual interpretation (atom_feat/atom_scalar/global/atom counts are not gated)",
            "a claim that the model does not use structure, or that the architecture must change",
            "a new Top-1, a new search space, a new architecture or a new dataset",
        ],
        "still_undetermined": [
            "whether the corrected verdict generalises beyond the 64 sampled molecules per split",
            "whether the outside-root response is necessary for prediction quality (not tested here)",
            "whether a different matched-null definition would change the ranking (only affected-dose matching was run)",
        ],
    }
    return answers


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def write_protocol_freeze() -> dict[str, Any]:
    payload = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "freeze_hash": C.freeze_hash(),
        "frozen_constants": C.FROZEN_CONSTANTS,
        "protocol_file_sha256": (
            C.sha256_file(C.ROUND_DIR / "PROTOCOL.md")
            if (C.ROUND_DIR / "PROTOCOL.md").is_file() else None
        ),
        "note": "constants were frozen before the audit ran; the audit re-selects nothing",
    }
    C.atomic_write_json(C.ANALYSIS_DIR / "protocol_freeze.json", payload)
    return payload


def merge_preflight(per_dataset: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for dataset, payload in per_dataset.items():
        pf = payload["preflight"]
        gates[dataset] = {
            "engineering_gate": pf["engineering_gate"],
            "pass": bool(pf["pass"]),
            "gates": pf["gates"],
            "data": pf["data"],
            "samples": {
                split: {
                    "n_selected": pf["samples"]["splits"][split]["n_selected"],
                    "row_ids_sha256": pf["samples"]["splits"][split]["row_ids_sha256"],
                    "reselect_reproduces_pilot": pf["samples"]["splits"][split]["reselect_reproduces_pilot"],
                }
                for split in C.SPLITS
            },
            "training_source": {
                "sha256": pf["training_source"]["sha256"],
                "expected_sha256": pf["training_source"]["expected_sha256"],
                "selected_epoch": pf["training_source"]["trajectory"]["selected_epoch"],
                "epochs": pf["training_source"]["trajectory"]["epochs"],
                "test_eval_count": pf["training_source"]["trajectory"]["test_eval_count"],
                "pass": pf["training_source"]["pass"],
            },
            "model_equivalence": pf["model_equivalence"],
            "parameters": pf["parameters"],
            "state_hash": pf["state_hash"],
            "pilot_engineering_gate": pf["pilot_engineering_gate"],
            "smoke": {
                "n_molecules": pf["smoke"]["n_molecules"],
                "passed": pf["smoke"]["passed"],
                "failures": pf["smoke"]["failures"],
                "records": pf["smoke"]["records"],
            },
        }
    payload = {
        "round_id": C.ROUND_ID,
        "upstream_round_id": C.UPSTREAM_ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "condition_name": C.CONDITION_NAME,
        "generated_at": C.utcnow(),
        "freeze_hash": C.freeze_hash(),
        "engineering_gate": "PASS" if all(g["engineering_gate"] == "PASS" for g in gates.values()) else "FAIL",
        "datasets": gates,
        "critical_failures": [
            f"{d}:{gate['gate']}" for d, g in gates.items() for gate in g["gates"] if not gate["passed"]
        ],
    }
    C.atomic_write_json(C.ANALYSIS_DIR / "preflight.json", payload)
    return payload


def write_run_manifest(per_dataset: Mapping[str, Mapping[str, Any]],
                       old_replay_passed: bool) -> dict[str, Any]:
    cache_hashes = {}
    pilot_hashes = {}
    for dataset in C.DATASETS:
        paths = C.upstream_paths(dataset)
        cache_hashes[dataset] = {
            "frozen_graph_cache": C.sha256_file(C.SQL_DATA / C.SD.cache_filename(dataset)),
            "lipo_raw": C.sha256_file(C.SQL_DATA / C.SD.lipo_raw_filename(dataset)),
        }
        pilot_hashes[dataset] = {
            "split_records": {
                split: C.sha256_file(paths[split]) for split in C.SPLITS
            },
            "search_log": C.sha256_file(paths["candidates"]),
            "samples": C.sha256_file(paths["samples"]),
        }
    pilot_manifest = C.read_json(C.UPSTREAM_ANALYSIS / "run_manifest.json")
    payload = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "upstream_round_id": C.UPSTREAM_ROUND_ID,
        "generated_at": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_branch": C.git_branch(),
        "git_worktree_dirty": C.worktree_dirty(),
        "protocol_hash": C.freeze_hash(),
        "protocol_file_sha256": (
            C.sha256_file(C.ROUND_DIR / "PROTOCOL.md")
            if (C.ROUND_DIR / "PROTOCOL.md").is_file() else None
        ),
        "code_sha256": C.code_hashes(),
        "upstream": {
            "round_id": C.UPSTREAM_ROUND_ID,
            "protocol_hash": pilot_manifest.get("protocol_hash"),
            "results_commit": (pilot_manifest.get("execution_provenance") or {}).get("results_commit"),
            "analysis_commit": (pilot_manifest.get("execution_provenance") or {}).get("analysis_commit"),
            "pilot_metric_replay_passed": bool(old_replay_passed),
            "readonly_paths": [
                C.relpath(C.UPSTREAM_ROUND_DIR),
                C.relpath(C.SQL_DIR),
                C.relpath(C.PREP_DIR),
                C.relpath(C.DISSECT_DIR),
                C.relpath(C.SOURCE_ROUND_DIR),
            ],
        },
        "frozen_inputs": {
            "caches": cache_hashes,
            "checkpoints": {d: C.EXPECTED_CHECKPOINT_SHA256[d] for d in C.DATASETS},
            "trajectories": {d: C.EXPECTED_TRAJECTORY_SHA256[d] for d in C.DATASETS},
            "pilot_records": pilot_hashes,
        },
        "environment": C.environment_snapshot(),
        "execution": {
            dataset: {
                "device": per_dataset[dataset]["preflight"]["device"],
                "torch_version": per_dataset[dataset]["preflight"]["torch_version"],
                "cuda_available": per_dataset[dataset]["preflight"]["cuda_available"],
                "budget": per_dataset[dataset]["budget"],
            }
            for dataset in C.DATASETS
        },
        "selection": {
            "top1": "reused from the pilot; never re-selected",
            "sample": "reused from the pilot; re-selected and hash-verified",
            "checkpoint": "reused; not retrained",
            "seed": {"training": 0, "sample": C.SAMPLE_SEED, "controls": C.CONTROL_SEED,
                     "bootstrap": C.BOOTSTRAP_SEED},
        },
        "metrics_protocols": {
            "pilot_internal_matched": "the pilot's own null and effects, recomputed from its records",
            "corrected_affected_matched": "this round's affected-matched null and re-evaluated effects",
            "note": "different null-matching protocols; the two rows are never pooled or substituted",
        },
    }
    C.atomic_write_json(C.ANALYSIS_DIR / "run_manifest.json", payload)
    return payload


def run(datasets: Sequence[str] = C.DATASETS) -> dict[str, Any]:
    freeze = write_protocol_freeze()
    per_dataset: dict[str, Any] = {}
    all_old_vs: list[dict[str, Any]] = []
    all_controls: list[dict[str, Any]] = []
    all_old_controls: list[dict[str, Any]] = []
    all_localization: list[dict[str, Any]] = []
    localization_detail: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    for dataset in datasets:
        new = load_new(dataset)
        if not all(new["records"][split] for split in C.SPLITS):
            raise RuntimeError(f"{dataset}: missing audit records; run the audit stage first")
        pilot = load_pilot(dataset)
        train_std = C.train_std(dataset)
        rows, build_detail = build_old_vs_corrected(dataset, pilot, new, train_std)
        all_old_vs.extend(rows)
        for split in C.SPLITS:
            all_controls.extend(matched_control_rows(dataset, split, new["records"][split]))
            all_old_controls.extend(old_control_rows(dataset, split, new["records"][split]))
            all_localization.extend(localization_rows(dataset, split, new["records"][split]))
        loc_splits = {
            split: localization_summary(dataset, split, localization_rows(dataset, split, new["records"][split]))
            for split in C.SPLITS
        }
        loc_bins = {
            split: localization_bins(localization_rows(dataset, split, new["records"][split]))
            for split in C.SPLITS
        }
        pooled = [r for split in C.SPLITS for r in localization_rows(dataset, split, new["records"][split])]
        localization_detail[dataset] = {
            "dataset": dataset,
            "splits": loc_splits,
            "bins_by_n_bonds": loc_bins,
            "pooled": localization_summary(dataset, "all", pooled),
            "note": (
                "outside_response_fraction describes where the local readout response lands; "
                "it is not an independent per-atom chemical contribution"
            ),
        }
        detail[dataset] = {
            "build": build_detail,
            "dose_verdict": dose_verdict(dataset, pilot),
            "new": {
                "run_summary": new["run_summary"],
                "budget": new["budget"],
            },
            "pilot": {
                "preflight_engineering_gate": pilot["preflight"]["engineering_gate"],
                "samples": pilot["samples"],
            },
        }
        per_dataset[dataset] = new

    preflight = merge_preflight(per_dataset)
    C.write_csv_file(C.ANALYSIS_DIR / "old_vs_corrected.csv", all_old_vs, columns=OLD_VS_COLUMNS)
    C.write_csv_file(C.ANALYSIS_DIR / "matched_controls.csv", all_controls, columns=MATCHED_CONTROL_COLUMNS)
    C.write_csv_file(C.ANALYSIS_DIR / "old_control_dose.csv", all_old_controls, columns=OLD_CONTROL_COLUMNS)
    C.write_csv_file(C.ANALYSIS_DIR / "localization.csv", all_localization, columns=LOCALIZATION_COLUMNS)
    C.atomic_write_json(C.ANALYSIS_DIR / "localization_summary.json", localization_detail)

    old_replay_passed = bool(all(detail[d]["build"]["replay"]["pass"] for d in datasets))
    decision = build_decision(list(datasets), detail, localization_detail)
    decision["engineering_gate"] = preflight["engineering_gate"]
    decision["pilot_metric_replay_passed"] = old_replay_passed
    C.atomic_write_json(C.ANALYSIS_DIR / "decision.json", decision)

    bootstrap = {
        "round_id": C.ROUND_ID,
        "replicates": C.BOOTSTRAP_REPLICATES,
        "seed": C.BOOTSTRAP_SEED,
        "unit": "molecule",
        "note": (
            "percentile bootstrap over molecules under one fixed checkpoint and one fixed dataset; "
            "not a cross-seed stability claim"
        ),
        "protocols": {},
    }
    for dataset in datasets:
        bootstrap["protocols"][dataset] = {
            split: {
                "pilot": detail[dataset]["build"]["splits"][split]["pilot"]["_match"],
                "corrected": detail[dataset]["build"]["splits"][split]["corrected"]["_match"],
            }
            for split in C.SPLITS
        }
    C.atomic_write_json(C.ANALYSIS_DIR / "bootstrap.json", bootstrap)

    manifest = write_run_manifest(per_dataset, old_replay_passed)
    summary = {
        "round_id": C.ROUND_ID,
        "engineering_gate": preflight["engineering_gate"],
        "pilot_metric_replay_passed": old_replay_passed,
        "freeze_hash": freeze["freeze_hash"],
        "datasets": {
            d: {
                "dose_verdict": detail[d]["dose_verdict"],
                "old_vs_corrected": {s: detail[d]["build"]["splits"][s] for s in C.SPLITS},
                "localization": localization_detail[d]["splits"],
            }
            for d in datasets
        },
        "manifest": manifest,
        "decision": decision["summary"],
    }
    C.atomic_write_json(C.ANALYSIS_DIR / "analysis_summary.json", summary)
    return {
        "preflight": preflight,
        "decision": decision,
        "old_vs_corrected_rows": all_old_vs,
        "localization_rows": all_localization,
        "detail": detail,
        "localization_detail": localization_detail,
        "manifest": manifest,
        "summary": summary,
    }


__all__ = [
    "load_new", "load_pilot", "pilot_rows", "corrected_rows", "protocol_metrics",
    "dose_summary", "localization_rows", "localization_summary", "localization_bins",
    "matched_control_rows", "old_control_rows", "replay_pilot_metrics",
    "build_old_vs_corrected", "dose_verdict", "signal_verdict", "build_decision",
    "merge_preflight", "write_run_manifest", "write_protocol_freeze", "run",
]
