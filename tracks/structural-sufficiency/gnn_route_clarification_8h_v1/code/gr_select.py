"""Frozen selection rules (Val only) shared by the trainer, the worker and the
analysis.  Every rule is copied from the task brief; nothing is re-tuned here.
"""

from __future__ import annotations

from typing import Any

import gr_common as C

_CACHE: dict[str, Any] = {}


def _trajectory(stage: str, dataset: str, model: str, recipe: str) -> dict[str, Any] | None:
    path = C.task_dir(stage, dataset, model, recipe) / "trajectory.json"
    if not path.is_file():
        return None
    return C.read_json(path)


def stage1_cell(dataset: str, model: str, recipe: str) -> dict[str, Any] | None:
    return _trajectory(C.STAGE1, dataset, model, recipe)


def selected_recipe(dataset: str, model: str, *, stage: str = C.STAGE1) -> str:
    """minimum Val RMSE over the two frozen recipes; ties keep R3."""
    best = None
    for recipe in C.RECIPE_ORDER:
        traj = _trajectory(stage, dataset, model, recipe)
        if traj is None:
            continue
        value = float(traj["selected"]["valid_rmse_raw"])
        if best is None or value < best[1]:
            best = (recipe, value)
    if best is None:
        raise FileNotFoundError(f"no completed trajectory for {stage}/{dataset}/{model}")
    return best[0]


def best_val(dataset: str, model: str) -> float:
    recipe = selected_recipe(dataset, model)
    traj = stage1_cell(dataset, model, recipe)
    return float(traj["selected"]["valid_rmse_raw"])


def choose_trunk() -> dict[str, Any]:
    if "trunk" in _CACHE:
        return _CACHE["trunk"]
    ratios = {}
    for dataset in C.DATASETS:
        ratios[dataset] = best_val(dataset, "Gx") / best_val(dataset, "G0")
    r_geo = float((ratios[C.DATASETS[0]] * ratios[C.DATASETS[1]]) ** 0.5)
    r_max = float(max(ratios.values()))
    chosen = "Gx" if (r_geo <= C.BRIDGE_R_GEO and r_max <= C.BRIDGE_R_MAX) else "G0"
    out = {
        "r_d": ratios,
        "r_geo": r_geo,
        "r_max": r_max,
        "rule": "T=Gx iff r_geo<=0.99 and max_d r_d<=1.10; otherwise T=G0",
        "chosen": chosen,
        "preference_note": "engineering preference for the simpler architecture; not a significance test",
    }
    _CACHE["trunk"] = out
    return out


def selected_trunk_model() -> str:
    return str(choose_trunk()["chosen"])


def lipo_recipe_for_model(model: str) -> str:
    """The real-Lipo Val-selected recipe of one model family."""
    if model in ("L", "T"):
        arch = "L" if model == "L" else selected_trunk_model()
        return selected_recipe("ogbg-mollipo", arch)
    if model == "A":
        return selected_recipe("ogbg-mollipo", "A", stage=C.STAGE2)
    raise KeyError(model)


def stage1_table() -> list[dict[str, Any]]:
    rows = []
    for dataset in C.DATASETS:
        for arch in C.ARCHS:
            for recipe in C.RECIPE_ORDER:
                traj = stage1_cell(dataset, arch, recipe)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": arch,
                        "recipe": recipe,
                        "status": "COMPLETE" if traj else "MISSING",
                        "selected_epoch": (traj or {}).get("selected", {}).get("epoch"),
                        "selected": (traj or {}).get("selected", {}),
                        "final": (traj or {}).get("final", {}),
                        "last20": (traj or {}).get("last20", {}),
                        "n_parameters": (traj or {}).get("n_parameters"),
                    }
                )
    return rows
