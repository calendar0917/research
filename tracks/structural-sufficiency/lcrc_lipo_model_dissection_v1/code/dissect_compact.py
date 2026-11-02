"""Select the single simplification candidate (COMPACT) from public122 Validation.

The script reads *only* the frozen Validation metrics of the public122 stage
freezes and the trainable parameter counts.  It never opens a ``test_eval.json``
or a ``test_predictions.npz``; the input file list and hashes are recorded.

Pre-registered purchase rule (practical tolerances, not statistical equivalence)::

    a. trainable parameters(variant) < trainable parameters(FULL)
    b. mean_s dR_s <= 0.010
    c. max_s  dR_s <= 0.020
    d. mean_s dA_s <= 0.010

    dR_s = ValidRMSE(variant, s) - ValidRMSE(FULL, s)
    dA_s = ValidMAE(variant, s)  - ValidMAE(FULL, s)

Candidates are ordered by (fewest parameters, lowest mean Valid RMSE, variant
order).  If none qualifies, COMPACT = FULL and nothing is rescued.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

import dissect_common as C

STAGE_FREEZES = ("A_r122", "B_r122", "C_r122")
FORBIDDEN_INPUTS = ("test_eval.json", "test_predictions.npz", "score_")


def _load_entries() -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    by_variant: dict[str, dict[int, dict[str, Any]]] = {}
    inputs: list[dict[str, Any]] = []
    for tag in STAGE_FREEZES:
        path = C.FREEZE / f"{tag}.json"
        if not path.exists():
            raise SystemExit(f"missing stage freeze {C.track_rel(path)}")
        if any(token in path.name for token in FORBIDDEN_INPUTS):
            raise SystemExit("refusing to read a Test artifact")
        payload = C.read_json(path)
        if payload.get("test_scored"):
            raise SystemExit(f"{C.track_rel(path)} is marked as Test-scored; selection must stay Validation-only")
        inputs.append({"path": C.track_rel(path), "sha256": C.sha256_file(path), "tag": tag})
        for entry in payload["entries"]:
            by_variant.setdefault(entry["variant"], {})[int(entry["seed"])] = entry
    return by_variant, inputs


def select() -> dict[str, Any]:
    by_variant, inputs = _load_entries()
    for key, entries in by_variant.items():
        if key in ("A0", "FULL"):
            continue
        for entry in entries.values():
            leaked = [k for k in entry if str(k).startswith("test")]
            if leaked:
                raise SystemExit(f"Test artifact leaked into the selection inputs: {key} {leaked}")

    full = by_variant.get("A0")
    if not full or set(full) != set(C.SEEDS):
        raise SystemExit("FULL (A0) must be frozen for all three training seeds")
    full_rmse = {s: float(full[s]["valid_rmse"]) for s in C.SEEDS}
    full_mae = {s: float(full[s]["valid_mae"]) for s in C.SEEDS}
    full_params = int(full[C.SEEDS[0]]["parameters"])

    rows: list[dict[str, Any]] = []
    for variant in C.VARIANT_ORDER:
        if variant == "A0":
            continue
        entries = by_variant.get(variant)
        if not entries or set(entries) != set(C.SEEDS):
            rows.append(
                {
                    "variant": variant,
                    "variant_name": C.VARIANT_NAMES[variant],
                    "available": False,
                    "eligible": False,
                    "reason": "missing seeds",
                }
            )
            continue
        params = int(entries[C.SEEDS[0]]["parameters"])
        dr = {s: float(entries[s]["valid_rmse"]) - full_rmse[s] for s in C.SEEDS}
        da = {s: float(entries[s]["valid_mae"]) - full_mae[s] for s in C.SEEDS}
        mean_dr = float(np.mean([dr[s] for s in C.SEEDS]))
        max_dr = float(np.max([dr[s] for s in C.SEEDS]))
        mean_da = float(np.mean([da[s] for s in C.SEEDS]))
        mean_valid = float(np.mean([float(entries[s]["valid_rmse"]) for s in C.SEEDS]))
        checks = {
            "a_fewer_parameters": bool(params < full_params),
            "b_mean_dR": bool(mean_dr <= C.TOL_MEAN_DR),
            "c_max_dR": bool(max_dr <= C.TOL_MAX_DR),
            "d_mean_dA": bool(mean_da <= C.TOL_MEAN_DA),
        }
        eligible = all(checks.values())
        rows.append(
            {
                "variant": variant,
                "variant_name": C.VARIANT_NAMES[variant],
                "available": True,
                "parameters": params,
                "parameter_delta_vs_full": int(params - full_params),
                "seed_dR": {str(s): dr[s] for s in C.SEEDS},
                "seed_dA": {str(s): da[s] for s in C.SEEDS},
                "mean_dR": mean_dr,
                "max_dR": max_dr,
                "mean_dA": mean_da,
                "mean_valid_rmse": mean_valid,
                "checks": checks,
                "eligible": eligible,
            }
        )

    candidates = [r for r in rows if r["available"] and r["eligible"]]
    candidates.sort(
        key=lambda r: (
            r["parameters"],
            r["mean_valid_rmse"],
            C.VARIANT_ORDER.index(r["variant"]),
        )
    )
    if candidates:
        chosen = candidates[0]["variant"]
        status = "SIMPLER_CANDIDATE_SELECTED"
    else:
        chosen = "A0"
        status = "NO_SIMPLER_CANDIDATE_SELECTED"

    payload = {
        "schema": "lcrc-lipo-dissection-compact-selection-v1",
        "created_utc": C.utcnow(),
        "repeat": C.MAIN_REPEAT,
        "selection_inputs": inputs,
        "selection_inputs_exclude_test": True,
        "tolerances_logD": {
            "mean_dR": C.TOL_MEAN_DR,
            "max_dR": C.TOL_MAX_DR,
            "mean_dA": C.TOL_MEAN_DA,
            "status": "practical simplification-purchase tolerances, NOT statistical equivalence",
        },
        "full": {
            "variant": "A0",
            "parameters": full_params,
            "seed_valid_rmse": {str(s): full_rmse[s] for s in C.SEEDS},
            "seed_valid_mae": {str(s): full_mae[s] for s in C.SEEDS},
            "mean_valid_rmse": float(np.mean([full_rmse[s] for s in C.SEEDS])),
        },
        "rows": rows,
        "candidates_eligible": [r["variant"] for r in candidates],
        "selected": chosen,
        "selected_name": C.VARIANT_NAMES[chosen],
        "status": status,
        "order_rule": "parameters asc, mean Valid RMSE asc, frozen variant order",
    }
    C.write_json(C.ANALYSIS / "compact_selection.json", payload)
    C.log(f"COMPACT = {chosen} ({C.VARIANT_NAMES[chosen]}), status={status}")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Select COMPACT from public122 Validation only")
    parser.parse_args(argv)
    select()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
