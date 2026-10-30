"""Planned v2 run registry and conditional resolution.

The registry is frozen before GPU work.  Conditional keys use ``SEL`` for the
recipe selected on Fold 0, so the exact Fold 1/2 and full-Fit recipe is resolved
only after the pre-registered Fold 0 selection rule has run.
"""

from __future__ import annotations

from typing import Any, Mapping

import v2_common as C


def _train(key, arm, fold, recipe, seed, variant="scratch", conditional=False, note=None):
    return {
        "key": key,
        "kind": "train",
        "arm": arm,
        "fold": fold,
        "recipe": recipe,
        "seed": int(seed),
        "variant": variant,
        "conditional": bool(conditional),
        "note": note,
    }


def _ssl(key, fold, seed, conditional=False):
    return {
        "key": key,
        "kind": "ssl",
        "arm": "A",
        "fold": fold,
        "recipe": "SSL",
        "seed": int(seed),
        "variant": "ssl_pretrain",
        "conditional": bool(conditional),
        "note": None,
    }


def fold0_runs() -> list[dict[str, Any]]:
    out = []
    for seed in C.SEEDS:
        out.append(_train(f"F0_A_R0_s{seed}", "A", 0, "R0", seed))
    for seed in (0, 1):
        for recipe in ("R1", "R2", "R3"):
            out.append(_train(f"F0_A_{recipe}_s{seed}", "A", 0, recipe, seed))
    for seed in C.SEEDS:
        out.append(_train(f"F0_C_R0_s{seed}", "C", 0, "R0", seed))
    for seed in (0, 1):
        for recipe in ("R1", "R2", "R3"):
            out.append(_train(f"F0_C_{recipe}_s{seed}", "C", 0, recipe, seed))
    for seed in C.SEEDS:
        out.append(_train(f"F0_E_R0_s{seed}", "E", 0, "R0", seed))
    return out


def fold0_ssl_runs() -> list[dict[str, Any]]:
    out = []
    for seed in C.SEEDS:
        out.append(_ssl(f"F0_SSLpre_s{seed}", 0, seed, conditional=True))
    for seed in C.SEEDS:
        out.append(
            _train(
                f"F0_A_SSL_sel_s{seed}",
                "A",
                0,
                "SEL",
                seed,
                variant="ssl",
                conditional=True,
                note="requires SSL gate",
            )
        )
    out.append(
        _train(
            "F0_A_sel_s2",
            "A",
            0,
            "SEL",
            2,
            conditional=True,
            note="scratch seed 2 for the three-seed SSL comparison when SEL != R0",
        )
    )
    return out


def fold_runs(fold: int) -> list[dict[str, Any]]:
    out = []
    for seed in (0, 1):
        out.append(_train(f"F{fold}_A_R0_s{seed}", "A", fold, "R0", seed))
        out.append(_train(f"F{fold}_E_R0_s{seed}", "E", fold, "R0", seed))
        out.append(_train(f"F{fold}_A_sel_s{seed}", "A", fold, "SEL", seed, conditional=True))
        out.append(_train(f"F{fold}_C_sel_s{seed}", "C", fold, "SEL", seed, conditional=True))
        out.append(_ssl(f"F{fold}_SSLpre_s{seed}", fold, seed, conditional=True))
        out.append(
            _train(
                f"F{fold}_A_SSL_sel_s{seed}",
                "A",
                fold,
                "SEL",
                seed,
                variant="ssl",
                conditional=True,
                note="requires SSL gate",
            )
        )
    return out


def full_runs() -> list[dict[str, Any]]:
    out = []
    for seed in C.SEEDS:
        out.append(_train(f"full_A_scratch_s{seed}", "A", "full", "SEL", seed))
        out.append(_train(f"full_E_R0_s{seed}", "E", "full", "R0", seed))
        out.append(_train(f"full_C_tuned_s{seed}", "C", "full", "SEL", seed))
        out.append(_ssl(f"full_SSLpre_s{seed}", "full", seed, conditional=True))
        out.append(
            _train(
                f"full_A_SSL_sel_s{seed}",
                "A",
                "full",
                "SEL",
                seed,
                variant="ssl",
                conditional=True,
                note="requires SSL gate",
            )
        )
    return out


def planned_registry() -> dict[str, Any]:
    runs = fold0_runs() + fold0_ssl_runs() + fold_runs(1) + fold_runs(2) + full_runs()
    keys = [r["key"] for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate run key in registry")
    return {
        "protocol_id": C.PROTOCOL_ID,
        "runs": runs,
        "n_runs": len(runs),
        "n_train": sum(1 for r in runs if r["kind"] == "train"),
        "n_ssl_pretrain": sum(1 for r in runs if r["kind"] == "ssl"),
        "n_gpu_jobs": len(runs),
        "conditional_rules": {
            "SEL": "Fold-0 recipe selected per arm by the frozen rule",
            "SSL": "only if the Fold-0 three-seed pretext gate passes",
            "reuse": "an identical existing run is reused and not re-executed",
        },
    }


def run_by_key(key: str) -> dict[str, Any]:
    for row in planned_registry()["runs"]:
        if row["key"] == key:
            return row
    raise KeyError(f"unknown v2 run key {key!r}")


def resolve_recipe(key: str, recipe_lock: Mapping[str, Any]) -> str:
    row = run_by_key(key)
    if row["recipe"] != "SEL":
        return row["recipe"]
    arm = row["arm"]
    selected = recipe_lock.get("selection", {}).get(arm)
    if selected not in C.RECIPES:
        raise PermissionError(f"recipe lock has no valid selection for arm {arm}")
    return str(selected)


def planned_predictions(recipe_lock: Mapping[str, Any], ssl_gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Final Fit2352 prediction registry, resolved after recipe/SSL locks."""
    out = []
    a_recipe = recipe_lock["selection"]["A"]
    c_recipe = recipe_lock["selection"]["C"]
    for seed in C.SEEDS:
        out.append({"key": f"full_A_scratch_s{seed}", "arm": "A", "recipe": a_recipe, "seed": seed, "variant": "scratch"})
        out.append({"key": f"full_E_R0_s{seed}", "arm": "E", "recipe": "R0", "seed": seed, "variant": "scratch"})
        out.append({"key": f"full_C_tuned_s{seed}", "arm": "C", "recipe": c_recipe, "seed": seed, "variant": "scratch"})
        if ssl_gate.get("gate_pass") is True:
            out.append({"key": f"full_A_SSL_sel_s{seed}", "arm": "A", "recipe": a_recipe, "seed": seed, "variant": "ssl"})
    cpu = C.read_json(C.ANALYSIS / "cpu_baselines_v2.json")["selection"]["chosen"]
    out.append({"key": f"CPU_{cpu}", "arm": "CPU", "recipe": "CPU", "seed": None, "variant": "cpu"})
    return out


__all__ = [
    "planned_registry",
    "run_by_key",
    "resolve_recipe",
    "fold0_runs",
    "fold0_ssl_runs",
    "fold_runs",
    "full_runs",
    "planned_predictions",
]
