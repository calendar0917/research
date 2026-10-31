"""Frozen run registry for the v3 pilot.

Exactly two unconditional GPU runs and one conditional third run.  No other
seed, fold, recipe, SSL arm, descriptor combination or full-Fit retrain is
registered.
"""

from __future__ import annotations

from typing import Any, Mapping

import v3_common as C


def _row(key: str, arm: str, *, conditional: bool, comparator: str, condition: str | None = None) -> dict[str, Any]:
    base_arm, variant = C.ARM_VARIANT[arm]
    return {
        "key": key,
        "kind": "train",
        "arm": arm,
        "base_arm": base_arm,
        "feature_variant": variant,
        "fold": int(C.FOLD),
        "recipe": C.RECIPE,
        "seed": int(C.SEED),
        "variant": "scratch",
        "conditional": bool(conditional),
        "comparator": comparator,
        "condition": condition,
        "host": "res",
        "pool": "res-gpu0",
    }


def planned_registry() -> dict[str, Any]:
    runs = [
        _row(
            "F1_A_Chem12_R3_s0",
            "A_Chem12",
            conditional=False,
            comparator=C.HISTORICAL["A"]["key"],
        ),
        _row(
            "F1_A_Chem13_R3_s0",
            "A_Chem13",
            conditional=False,
            comparator=C.HISTORICAL["A"]["key"],
        ),
        _row(
            "F1_C_Chem12_R3_s0",
            "C_Chem12",
            conditional=True,
            comparator=C.HISTORICAL["C"]["key"],
            condition=(
                "A_Chem12 vs historical A-R3: RMSE improvement >= 0.03, MAE improvement > 0, "
                "and the gain is not tail-dependent under the pre-registered top-2% rule"
            ),
        ),
    ]
    keys = [r["key"] for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate run key in registry")
    return {
        "protocol_id": C.PROTOCOL_ID,
        "round_id": C.ROUND_ID,
        "runs": runs,
        "n_runs": len(runs),
        "n_unconditional": sum(1 for r in runs if not r["conditional"]),
        "n_conditional": sum(1 for r in runs if r["conditional"]),
        "n_gpu_jobs": len(runs),
        "allowed_extra_runs": 0,
        "forbidden": [
            "new seeds",
            "second fold",
            "SSL",
            "additional descriptor combinations",
            "full-Fit retrain",
            "Monitor evaluation",
        ],
        "comparators": {
            "A": C.HISTORICAL["A"]["key"],
            "C": C.HISTORICAL["C"]["key"],
        },
        "conditional_rule": {
            "arm": C.CONDITIONAL_ARM,
            "run_key": "F1_C_Chem12_R3_s0",
            "rmse_gain_min": C.MATERIALITY_RMSE,
            "tail_fraction": C.TAIL_FRACTION,
            "tail_min_retained_fraction": C.TAIL_MIN_RETAINED_FRACTION,
            "must_not_be_tail_dependent": True,
            "evaluated_against": C.HISTORICAL["C"]["key"],
        },
    }


def run_by_key(key: str) -> dict[str, Any]:
    for row in planned_registry()["runs"]:
        if row["key"] == key:
            return row
    raise KeyError(f"unknown v3 run key {key!r}")


def unconditional_keys() -> list[str]:
    return [r["key"] for r in planned_registry()["runs"] if not r["conditional"]]


def conditional_key() -> str:
    rows = [r for r in planned_registry()["runs"] if r["conditional"]]
    if len(rows) != 1:
        raise ValueError("registry must contain exactly one conditional run")
    return rows[0]["key"]


__all__ = ["planned_registry", "run_by_key", "unconditional_keys", "conditional_key"]
