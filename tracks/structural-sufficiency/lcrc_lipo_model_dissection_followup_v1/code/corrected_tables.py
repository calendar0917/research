"""Corrected cross-repeat tables for the completed dissection round.

Nothing is re-trained, re-selected or re-scored.  The script reads the already
reported per-run scalar table (and, for Test, only its already-reported scalar
columns), computes contrasts strictly inside one repeat with matched seeds, and
writes the correction next to -- never over -- the original tables.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence

import numpy as np

import followup_common as F

# Definitions copied from the frozen round (candidate - reference; negative is
# lower candidate error).
A_CONTRASTS = [
    ("A2", "A0", "NO_G - FULL (remove g, keep additive head)"),
    ("A4", "A3", "POOL_ONLY - NO_ADD (remove g, pooled readout)"),
    ("A3", "A0", "NO_ADD - FULL (remove additive head, keep g)"),
    ("A4", "A2", "POOL_ONLY - NO_G (remove additive head, no g)"),
]
B_CONTRASTS = [
    ("B1", "A0", "NO_JOINT - FULL"),
    ("B3", "B2", "NO_JOINT_CONTENT - NO_CONTENT"),
    ("B2", "A0", "NO_CONTENT - FULL"),
    ("B3", "B1", "NO_JOINT_CONTENT - NO_JOINT"),
]
OTHER_CONTRASTS = [
    ("B0", "A0", "TYPED_S2_OFF - FULL"),
    ("B4", "A0", "MARGINAL_ALL - FULL"),
    ("B5", "A0", "MARGINAL_BT2 - FULL"),
    ("B6", "A0", "NO_CLOSURE_CHANNELS - FULL"),
] + [(v, "A0", f"{v} - FULL") for v in ("C0", "C1", "C2", "C3", "C4")]
CONTRASTS = A_CONTRASTS + B_CONTRASTS + OTHER_CONTRASTS
# Anchor-only pairs used by the follow-up prompt/report.  They are kept
# separate so the legacy-bug reproduction stays identical to the original
# script's predefined list.
SUPPLEMENTAL_CONTRASTS = [
    ("A4", "A0", "supplemental anchor: POOL_ONLY - FULL"),
    ("A5", "A0", "supplemental anchor: ADD_ONLY - FULL"),
]
ALL_CONTRASTS = CONTRASTS + SUPPLEMENTAL_CONTRASTS
SUPPLEMENTAL_PAIRS = {(cand, ref) for cand, ref, _ in SUPPLEMENTAL_CONTRASTS}

INTERACTIONS = [
    (
        "A_INTERACTION",
        "I_A = (A2-A0) - (A4-A3) = A2-A0-A4+A3",
        ("A0", "A2", "A3", "A4"),
    ),
    (
        "B_INTERACTION",
        "I_B = (B1-A0) - (B3-B2) = B1-A0-B3+B2",
        ("A0", "B1", "B2", "B3"),
    ),
]

ANCHORS_RMSE = {
    "A2-A0": 0.0086991922,
    "A3-A0": -0.0087501425,
    "A4-A0": 0.0544958416,
    "A5-A0": -0.0124607112,
    "B0-A0": 0.0541728602,
    "B4-A0": 0.0070292609,
    "B5-A0": 0.0107162144,
    "C1-A0": -0.0012788261,
    "A4-A2": 0.04579665,
    "A_INTERACTION": -0.054546754,
    "B_INTERACTION": -0.004906714,
}
ANCHORS_MAE = {
    "A3-A0": 0.0266419937,
    "A5-A0": 0.0149033115,
    "B5-A0": 0.0167945243,
    "C1-A0": 0.0009023733,
}
ANCHORS_CROSS_REPEAT = {
    ("C1", "A0", 122): -0.00127883,
    ("C1", "A0", 123): -0.00190287,
    ("C1", "A0", 124): -0.00057491,
    ("B5", "A0", 122): 0.01071621,
    ("B5", "A0", 123): 0.01830643,
    ("B5", "A0", 124): 0.00896354,
}


# ---------------------------------------------------------------------------
# strict identity indexing
# ---------------------------------------------------------------------------


def identity_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, int], Mapping[str, Any]]:
    """Index by the full identity; silently overwriting is an error."""
    out: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["repeat"]), str(row["variant"]), int(row["seed"]))
        if key in out:
            raise ValueError(f"duplicate identity {key}; refusing silent overwrite")
        out[key] = row
    return out


def _metric_value(row: Mapping[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    return None if value is None else float(value)


def contrast_for_repeat(
    index: Mapping[tuple[int, str, int], Mapping[str, Any]],
    repeat: int,
    cand: str,
    ref: str,
    label: str,
    metric: str,
) -> dict[str, Any]:
    """One per-repeat contrast with same-repeat, same-seed matching only."""
    per_seed: dict[str, Any] = {}
    matched: list[int] = []
    missing: list[dict[str, Any]] = []
    for seed in F.SEEDS:
        a = index.get((int(repeat), cand, seed))
        b = index.get((int(repeat), ref, seed))
        av = _metric_value(a, metric)
        bv = _metric_value(b, metric)
        reasons = []
        if a is None:
            reasons.append(f"missing candidate cell {repeat}/{cand}/s{seed}")
        if b is None:
            reasons.append(f"missing reference cell {repeat}/{ref}/s{seed}")
        if a is not None and av is None:
            reasons.append(f"missing {metric} in candidate {repeat}/{cand}/s{seed}")
        if b is not None and bv is None:
            reasons.append(f"missing {metric} in reference {repeat}/{ref}/s{seed}")
        if reasons:
            missing.append({"seed": seed, "reasons": reasons})
            continue
        # mypy-free narrowing: av/bv are floats here
        diff = float(av) - float(bv)
        matched.append(seed)
        per_seed[str(seed)] = {
            "candidate_value": float(av),
            "reference_value": float(bv),
            "diff": diff,
        }
    diffs = [per_seed[str(s)]["diff"] for s in matched]
    if len(matched) == len(F.SEEDS):
        status = "complete"
    elif matched:
        status = "partial"
    else:
        status = "unavailable"
    return {
        "repeat": int(repeat),
        "contrast": f"{cand}-{ref}",
        "label": label,
        "candidate": cand,
        "reference": ref,
        "metric": metric,
        "formula": "candidate - reference, same repeat and same seed",
        "matched_seeds": matched,
        "n_matched": len(matched),
        "per_seed": per_seed,
        "mean_diff": float(np.mean(diffs)) if diffs else None,
        "sd_diff_ddof1": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else (0.0 if diffs else None),
        "min_diff": float(np.min(diffs)) if diffs else None,
        "max_diff": float(np.max(diffs)) if diffs else None,
        "status": status,
        "missing_cells": missing,
    }


def interaction_for_repeat(
    index: Mapping[tuple[int, str, int], Mapping[str, Any]],
    repeat: int,
    tag: str,
    formula: str,
    variants: Sequence[str],
    metric: str,
) -> dict[str, Any]:
    """All four cells must be present in the *same* repeat and *same* seed."""
    a0, a2, a3, a4 = variants
    per_seed: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    for seed in F.SEEDS:
        cells = {v: index.get((int(repeat), v, seed)) for v in variants}
        values = {v: _metric_value(cells[v], metric) for v in variants}
        absent = [v for v, value in values.items() if value is None]
        if absent:
            missing.append({"seed": seed, "missing_variants": absent})
            continue
        # I_A and I_B both equal cell[0]-cell[1] expression written explicitly.
        if tag == "A_INTERACTION":
            value = (float(values[a0]) - float(values[a0]))  # replaced below, kept explicit
            value = (float(values[a2]) - float(values[a0])) - (float(values[a4]) - float(values[a3]))
        elif tag == "B_INTERACTION":
            value = (float(values["B1"]) - float(values["A0"])) - (float(values["B3"]) - float(values["B2"]))
        else:
            raise KeyError(tag)
        per_seed[str(seed)] = {
            "cells": {v: float(values[v]) for v in variants},
            "value": float(value),
        }
    vals = [per_seed[str(s)]["value"] for s in F.SEEDS if str(s) in per_seed]
    if len(vals) == len(F.SEEDS):
        status = "complete"
    elif vals:
        status = "partial"
    else:
        status = "unavailable"
    return {
        "repeat": int(repeat),
        "contrast": tag,
        "formula": formula,
        "metric": metric,
        "variants": list(variants),
        "matched_seeds": [s for s in F.SEEDS if str(s) in per_seed],
        "n_matched": len(vals),
        "per_seed": per_seed,
        "mean_diff": float(np.mean(vals)) if vals else None,
        "sd_diff_ddof1": float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else None),
        "min_diff": float(np.min(vals)) if vals else None,
        "max_diff": float(np.max(vals)) if vals else None,
        "status": status,
        "missing_cells": missing,
    }


def build_tables(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    index = identity_index(rows)
    rmse_key = f"{split}_rmse"
    mae_key = f"{split}_mae"
    entries: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for repeat in F.REPEATS:
        for cand, ref, label in ALL_CONTRASTS:
            for metric in (rmse_key, mae_key):
                entries.append(contrast_for_repeat(index, repeat, cand, ref, label, metric))
        for tag, formula, variants in INTERACTIONS:
            for metric in (rmse_key, mae_key):
                interactions.append(interaction_for_repeat(index, repeat, tag, formula, variants, metric))

    for item in entries:
        item["supplemental"] = (item["candidate"], item["reference"]) in SUPPLEMENTAL_PAIRS

    # Equal-weight descriptive cross-repeat summaries.  Each repeat first
    # contributes its own same-seed paired mean; repeats sharing molecules are
    # not treated as independent datasets and no interval is constructed.
    cross: dict[str, Any] = {}
    for item in entries:
        key = item["contrast"] + "|" + item["metric"]
        slot = cross.setdefault(
            key,
            {
                "contrast": item["contrast"],
                "metric": item["metric"],
                "per_repeat_mean_diff": {},
                "per_repeat_status": {},
                "n_repeats_with_same_repeat_estimate": 0,
                "equal_weight_mean_across_repeats": None,
            },
        )
        if item["mean_diff"] is not None:
            slot["per_repeat_mean_diff"][str(item["repeat"])] = item["mean_diff"]
            slot["per_repeat_status"][str(item["repeat"])] = item["status"]
    for item in interactions:
        key = item["contrast"] + "|" + item["metric"]
        slot = cross.setdefault(
            key,
            {
                "contrast": item["contrast"],
                "metric": item["metric"],
                "per_repeat_mean_diff": {},
                "per_repeat_status": {},
                "n_repeats_with_same_repeat_estimate": 0,
                "equal_weight_mean_across_repeats": None,
            },
        )
        if item["mean_diff"] is not None:
            slot["per_repeat_mean_diff"][str(item["repeat"])] = item["mean_diff"]
            slot["per_repeat_status"][str(item["repeat"])] = item["status"]
    for slot in cross.values():
        values = list(slot["per_repeat_mean_diff"].values())
        slot["n_repeats_with_same_repeat_estimate"] = len(values)
        slot["equal_weight_mean_across_repeats"] = float(np.mean(values)) if values else None
        slot["note"] = (
            "Each repeat is reduced to its own same-seed paired mean first; the cross-repeat number is an "
            "equal-weight descriptive average. The three public repeats share molecules, so this is not an "
            "independent-dataset estimate and no confidence interval is constructed."
        )

    return {
        "schema": "lcrc-lipo-followup-corrected-tables-v1",
        "split": split,
        "status_note": (
            "EXPOSED_TEST_SCALAR_DIFF_ONLY: differences recomputed only from Test RMSE/MAE scalars already "
            "reported in main_runs.csv. No Test prediction file, target, cache or new inference was used."
            if split == "test"
            else "Valid scalars from the already completed runs; no new inference."
        ),
        "identity": "row identity is (repeat, variant, seed); duplicate identities raise",
        "contrast_rule": "candidate - reference, same repeat and same seed; missing cells are never filled from another repeat",
        "entries": entries,
        "interactions": interactions,
        "cross_repeat_summary": cross,
    }


# ---------------------------------------------------------------------------
# legacy-bug reproduction and checks
# ---------------------------------------------------------------------------


def legacy_by_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Reproduce the original ``_seed_diffs`` dictionary, including its overwrite."""
    by: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        by[(str(row["variant"]), int(row["seed"]))] = row
    return by


def reproduce_legacy_bug(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by = legacy_by_rows(rows)
    out: dict[str, Any] = {}
    for split in ("valid", "test"):
        rmse_key = f"{split}_rmse"
        entries = []
        for cand, ref, label in CONTRASTS:
            ds = []
            sources = []
            for seed in F.SEEDS:
                a, b = by.get((cand, seed)), by.get((ref, seed))
                if a is None or b is None or a.get(rmse_key) is None or b.get(rmse_key) is None:
                    continue
                ds.append(float(a[rmse_key]) - float(b[rmse_key]))
                sources.append(
                    {
                        "seed": seed,
                        "candidate_repeat": int(a["repeat"]),
                        "reference_repeat": int(b["repeat"]),
                    }
                )
            if ds:
                entries.append(
                    {
                        "contrast": f"{cand}-{ref}",
                        "label": label,
                        "legacy_mean_diff": float(np.mean(ds)),
                        "legacy_per_seed": ds,
                        "sources": sources,
                    }
                )
        # Original interaction loops use the same overwritten dictionary.
        for tag, formula, variants in INTERACTIONS:
            vals = []
            if tag == "A_INTERACTION":
                for seed in F.SEEDS:
                    cells = [by.get((v, seed)) for v in variants]
                    if any(c is None for c in cells):
                        continue
                    vals.append(
                        (float(cells[1][rmse_key]) - float(cells[0][rmse_key]))
                        - (float(cells[3][rmse_key]) - float(cells[2][rmse_key]))
                    )
            else:
                for seed in F.SEEDS:
                    cells = [by.get((v, seed)) for v in variants]
                    if any(c is None for c in cells):
                        continue
                    vals.append(
                        (float(cells[1][rmse_key]) - float(cells[0][rmse_key]))
                        - (float(cells[3][rmse_key]) - float(cells[2][rmse_key]))
                    )
            if vals:
                entries.append(
                    {
                        "contrast": tag,
                        "label": formula,
                        "legacy_mean_diff": float(np.mean(vals)),
                        "legacy_per_seed": vals,
                    }
                )
        out[split] = entries
    return out


def compare_legacy_to_corrected(
    legacy: Mapping[str, Any], corrected: Mapping[str, Any], split: str = "valid"
) -> list[dict[str, Any]]:
    metric = f"{split}_rmse"
    corrected_map: dict[tuple[str, str], float] = {}
    for item in corrected["entries"] + corrected["interactions"]:
        if item["repeat"] == F.MAIN_REPEAT and item.get("mean_diff") is not None:
            corrected_map[(item["contrast"], item["metric"])] = float(item["mean_diff"])
    out = []
    for item in legacy[split]:
        key = (item["contrast"], metric)
        if key not in corrected_map:
            continue
        diff = float(item["legacy_mean_diff"]) - corrected_map[key]
        sources = item.get("sources", [])
        wrong_repeat_source = any(
            src.get("candidate_repeat") != F.MAIN_REPEAT or src.get("reference_repeat") != F.MAIN_REPEAT
            for src in sources
        )
        if abs(diff) > 1e-12 or wrong_repeat_source:
            out.append(
                {
                    "split": split,
                    "contrast": item["contrast"],
                    "legacy_mean_diff": item["legacy_mean_diff"],
                    "corrected_repeat122_mean_diff": corrected_map[key],
                    "legacy_minus_corrected": diff,
                    "wrong_repeat_source": bool(wrong_repeat_source),
                    "sources": sources,
                }
            )
    return out


def anchor_checks(rows: Sequence[Mapping[str, Any]], tables: Mapping[str, Any]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    ok_all = True
    max_abs_residual = 0.0

    def add(contrast: str, metric: str, expected: float, got: float) -> None:
        nonlocal ok_all, max_abs_residual
        residual = float(got) - float(expected)
        passed = abs(residual) <= 1e-6
        ok_all = ok_all and passed
        max_abs_residual = max(max_abs_residual, abs(residual))
        details.append(
            {
                "contrast": contrast,
                "metric": metric,
                "expected": float(expected),
                "computed": float(got),
                "residual": residual,
                "tolerance": 1e-6,
                "passed": bool(passed),
            }
        )

    def lookup(contrast: str, metric: str) -> float:
        pool = tables["interactions"] if contrast.endswith("INTERACTION") else tables["entries"]
        hits = [
            x
            for x in pool
            if x["repeat"] == F.MAIN_REPEAT and x["contrast"] == contrast and x["metric"] == metric
        ]
        if hits:
            return float(hits[0]["mean_diff"])
        # Anchor-only pairs that are not part of the predefined contrast list
        # (A4-A0, A5-A0) are still computed directly from the same-repeat index.
        if "-" in contrast and not contrast.endswith("INTERACTION"):
            cand, ref = contrast.split("-")
            index = identity_index(rows)
            vals = []
            for seed in F.SEEDS:
                a = index.get((F.MAIN_REPEAT, cand, seed))
                b = index.get((F.MAIN_REPEAT, ref, seed))
                if a is not None and b is not None and a.get(metric) is not None and b.get(metric) is not None:
                    vals.append(float(a[metric]) - float(b[metric]))
            if vals:
                return float(np.mean(vals))
        raise AssertionError(f"anchor contrast not found: {contrast} / {metric}")

    for contrast, expected in ANCHORS_RMSE.items():
        add(contrast, "valid_rmse", expected, lookup(contrast, "valid_rmse"))
    for contrast, expected in ANCHORS_MAE.items():
        add(contrast, "valid_mae", expected, lookup(contrast, "valid_mae"))
    for (cand, ref, repeat), expected in ANCHORS_CROSS_REPEAT.items():
        hits = [
            x
            for x in tables["entries"]
            if x["repeat"] == repeat and x["contrast"] == f"{cand}-{ref}" and x["metric"] == "valid_rmse"
        ]
        if not hits:
            raise AssertionError(f"cross anchor not found: {cand}-{ref} repeat {repeat}")
        add(f"{cand}-{ref}@{repeat}", "valid_rmse", expected, float(hits[0]["mean_diff"]))

    index = identity_index(rows)
    a4a2 = lookup("A4-A2", "valid_rmse")
    full_means = {
        r: float(np.mean([float(index[(r, "A0", s)]["valid_rmse"]) for s in F.SEEDS])) for r in F.REPEATS
    }
    legacy = reproduce_legacy_bug(rows)
    bug_a2 = float([x for x in legacy["valid"] if x["contrast"] == "A2-A0"][0]["legacy_mean_diff"])
    correct_a2 = lookup("A2-A0", "valid_rmse")
    offset = bug_a2 - correct_a2
    expected_offset = full_means[122] - full_means[124]
    decomposition = {
        "legacy_A2_minus_A0": bug_a2,
        "correct_repeat122_A2_minus_A0": correct_a2,
        "offset_legacy_minus_correct": offset,
        "FULL122_mean_valid_rmse": full_means[122],
        "FULL124_mean_valid_rmse": full_means[124],
        "FULL122_minus_FULL124_mean": expected_offset,
        "offset_matches_FULL122_minus_FULL124": bool(abs(offset - expected_offset) <= 1e-6),
        "explanation": (
            "The overwritten dictionary uses A2 from repeat 122 and A0 from repeat 124. "
            "legacy = A2_122 - A0_124 = (A2_122 - A0_122) + (A0_122 - A0_124)."
        ),
    }
    return {
        "schema": "lcrc-lipo-followup-anchor-checks-v1",
        "all_passed": bool(ok_all and decomposition["offset_matches_FULL122_minus_FULL124"]),
        "max_abs_residual": max_abs_residual,
        "residual_tolerance": 1e-6,
        "details": details,
        "A4_A2": {
            "computed": a4a2,
            "expected": 0.04579665,
            "passed": abs(a4a2 - 0.04579665) <= 1e-6,
            "note": "A4-A2 is not A4-A0; both are reported separately.",
        },
        "legacy_A2_A0_decomposition": decomposition,
    }



def cross_repeat_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify the original per-repeat cross table against the per-run rows."""
    original = F.read_json(F.ORIG_ANALYSIS / "tables" / "cross_repeat.json")
    index = identity_index(rows)
    details = []
    max_abs = 0.0
    all_ok = True
    for entry in original:
        variant = str(entry["variant"])
        for repeat_key, item in entry.get("per_repeat", {}).items():
            repeat = int(repeat_key)
            sel = [index[(repeat, variant, seed)] for seed in F.SEEDS if (repeat, variant, seed) in index]
            if len(sel) != 3:
                details.append({"variant": variant, "repeat": repeat, "status": "incomplete_in_main_rows", "n": len(sel)})
                all_ok = False
                continue
            expected = {
                "n_seeds": 3,
                "valid_rmse_mean": float(np.mean([float(x["valid_rmse"]) for x in sel])),
                "valid_mae_mean": float(np.mean([float(x["valid_mae"]) for x in sel])),
                "test_rmse_mean": float(np.mean([float(x["test_rmse"]) for x in sel])),
                "test_mae_mean": float(np.mean([float(x["test_mae"]) for x in sel])),
            }
            for key, value in expected.items():
                got = item.get(key)
                if got is None:
                    continue
                residual = float(got) - float(value)
                max_abs = max(max_abs, abs(residual))
                passed = abs(residual) <= 1e-9
                all_ok = all_ok and passed
                details.append(
                    {
                        "variant": variant,
                        "repeat": repeat,
                        "field": key,
                        "original_cross_repeat": float(got),
                        "recomputed_from_main_rows": float(value),
                        "residual": residual,
                        "passed": bool(passed),
                    }
                )
    return {
        "schema": "lcrc-lipo-followup-cross-repeat-check-v1",
        "file": F.rel(F.ORIG_ANALYSIS / "tables" / "cross_repeat.json"),
        "sha256": F.sha256_file(F.ORIG_ANALYSIS / "tables" / "cross_repeat.json"),
        "all_fields_match": bool(all_ok),
        "max_abs_residual": max_abs,
        "tolerance": 1e-9,
        "details": details,
        "note": (
            "The original cross_repeat table groups rows by (repeat, variant) before averaging, so it is not "
            "affected by the _seed_diffs overwrite. Its per-repeat cells are verified here against main_runs.csv."
        ),
    }

def bootstrap_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    boot = F.read_json(F.ORIG_ANALYSIS / "bootstrap.json")
    tables = build_tables(rows, "valid")
    corrected_122 = {
        (x["contrast"], x["metric"]): x["mean_diff"]
        for x in tables["entries"] + tables["interactions"]
        if x["repeat"] == F.MAIN_REPEAT and x["mean_diff"] is not None
    }
    details = []
    max_abs = 0.0
    for contrast, item in boot.get("valid", {}).get("contrasts", {}).items():
        for metric, field in (("valid_rmse", "d_rmse_mean_point"), ("valid_mae", "d_mae_mean_point")):
            key = (contrast, metric)
            if key not in corrected_122 or item.get(field) is None:
                continue
            residual = float(item[field]) - float(corrected_122[key])
            max_abs = max(max_abs, abs(residual))
            details.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "bootstrap_point": float(item[field]),
                    "corrected_repeat122_point": float(corrected_122[key]),
                    "residual": residual,
                }
            )
    return {
        "interpretation": (
            "The existing bootstrap used only repeat 122. Its point estimates must agree with the corrected "
            "repeat-122 contrast. Its intervals remain evaluation-sample (scaffold) intervals for fixed "
            "training results; they were not re-run and are not training-randomness intervals."
        ),
        "max_abs_residual": max_abs,
        "tolerance": 1e-6,
        "all_points_agree": bool(max_abs <= 1e-6),
        "details": details,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bootstrap-check", action="store_true")
    args = ap.parse_args(argv)

    F.TABLES.mkdir(parents=True, exist_ok=True)
    rows = F.load_main_rows()
    identity = F.check_main_rows(rows)
    valid = build_tables(rows, "valid")
    test = build_tables(rows, "test")
    anchors = anchor_checks(rows, valid)
    if not anchors["all_passed"]:
        raise SystemExit(f"anchor checks failed: {anchors}")

    legacy = reproduce_legacy_bug(rows)
    affected = compare_legacy_to_corrected(legacy, valid, "valid")
    affected_test = compare_legacy_to_corrected(legacy, test, "test")

    meta = {
        "schema": "lcrc-lipo-followup-corrected-tables-v1",
        "created_utc": F.utcnow(),
        "git_commit": F.git_commit(),
        "source_round": F.rel(F.ORIG),
        "source_round_commit_expected": "4a393476aace617666642d87e0443ea3c3af87e0",
        "source_table": F.rel(F.MAIN_RUNS),
        "source_table_sha256": F.sha256_file(F.MAIN_RUNS),
        "new_training_runs": 0,
        "new_test_inference_or_rescoring": 0,
        "identity_check": identity,
        "legacy_bug_note": (
            "The original _seed_diffs used a dict keyed by (variant, seed) after appending repeats "
            "122, 123 and 124. Variants present only in 122 were compared with A0 from 124; overlapping "
            "A0/C1/B5 cells were overwritten by 124. The per-run table, the per-repeat cross_repeat table, "
            "the repeat-122 bootstrap point estimates, and all Valid predictions are unaffected."
        ),
    }
    valid["meta"] = meta
    test["meta"] = {**meta, "test_scalar_only": True}
    F.write_json(F.TABLES / "corrected_contrasts_valid.json", valid)
    F.write_json(F.TABLES / "corrected_contrasts_test_exposed.json", test)
    F.write_json(F.TABLES / "anchor_checks.json", anchors)
    F.write_json(
        F.TABLES / "legacy_affected_differences.json",
        {"affected_valid": affected, "affected_test_exposed": affected_test, "legacy": legacy},
    )
    cross_check = cross_repeat_check(rows)
    F.write_json(F.TABLES / "cross_repeat_check.json", cross_check)
    if not cross_check["all_fields_match"]:
        raise SystemExit(f"cross_repeat verification failed: {cross_check['max_abs_residual']}")
    F.write_json(F.TABLES / "input_identity_check.json", identity)

    flat = []
    for item in valid["entries"] + valid["interactions"]:
        flat.append(
            {
                "repeat": item["repeat"],
                "contrast": item["contrast"],
                "kind": "interaction" if item["contrast"].endswith("INTERACTION") else "contrast",
                "label": item.get("label", item.get("formula")),
                "metric": item["metric"],
                "candidate": item.get("candidate"),
                "reference": item.get("reference"),
                "formula": item.get("formula"),
                "matched_seeds": ",".join(str(s) for s in item["matched_seeds"]),
                "n_matched": item["n_matched"],
                "mean_diff": item["mean_diff"],
                "sd_diff_ddof1": item["sd_diff_ddof1"],
                "min_diff": item["min_diff"],
                "max_diff": item["max_diff"],
                "status": item["status"],
                "per_seed_json": json.dumps(item["per_seed"], sort_keys=True),
            }
        )
    F.write_csv(F.TABLES / "corrected_contrasts_valid.csv", flat)
    test_flat = []
    for item in test["entries"] + test["interactions"]:
        test_flat.append(
            {
                "repeat": item["repeat"],
                "contrast": item["contrast"],
                "kind": "interaction" if item["contrast"].endswith("INTERACTION") else "contrast",
                "label": item.get("label", item.get("formula")),
                "metric": item["metric"],
                "candidate": item.get("candidate"),
                "reference": item.get("reference"),
                "matched_seeds": ",".join(str(s) for s in item["matched_seeds"]),
                "n_matched": item["n_matched"],
                "mean_diff": item["mean_diff"],
                "sd_diff_ddof1": item["sd_diff_ddof1"],
                "status": item["status"],
            }
        )
    F.write_csv(F.TABLES / "corrected_contrasts_test_exposed.csv", test_flat)

    if not args.skip_bootstrap_check:
        boot_check = bootstrap_consistency(rows)
        F.write_json(F.TABLES / "bootstrap_point_check.json", boot_check)
        if not boot_check["all_points_agree"]:
            raise SystemExit(f"bootstrap point check failed: {boot_check['max_abs_residual']}")

    print(
        F.write_json
        and f"wrote corrected tables: {len(valid['entries'])} valid contrast cells, "
        f"{len(test['entries'])} test-scalar cells, {len(affected)} valid and {len(affected_test)} test legacy differing rows"
    )
    print(f"anchor max residual {anchors['max_abs_residual']:.3e}; all_passed={anchors['all_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
