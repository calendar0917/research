"""Descriptive prediction-stratified localisation tables (no new selection)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import xr_common as C


METRICS = ("atom_jaccard_adv", "atom_recall_adv", "bond_recall_adv", "hit_boundary_adv")


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _boot(vals: np.ndarray, comps: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool) & np.isfinite(vals)
    if not mask.any():
        return {"point": None, "ci95": [None, None], "n": 0, "n_components": 0, "support": "NO_SUPPORT"}
    out = C.component_bootstrap({"v": vals}, comps, lambda d: {"v": float(np.mean(d["v"]))}, mask=mask, n_boot=C.BOOTSTRAP_REPLICATES, seed=C.BOOTSTRAP_SEED)
    return {
        "point": out["point"]["v"],
        "ci95": out["ci95"]["v"],
        "n": int(mask.sum()),
        "n_components": out["n_components"],
        "support": C.support_label(out["n_components"], int(mask.sum())),
    }


def main() -> int:
    pairs = _load_csv(C.ANALYSIS_DIR / "measured_pair_results.csv")
    oracle = _load_csv(C.ANALYSIS_DIR / "oracle_localization_results.csv")
    comps = np.asarray([int(r["component"]) for r in pairs], dtype=np.int64)
    rows: list[dict[str, Any]] = []

    groups = {
        "direction_correct": lambda r: bool(int(r["direction_evaluable"])) and str(r["direction_LCRC"]).lower() in ("true", "1"),
        "direction_wrong": lambda r: bool(int(r["direction_evaluable"])) and str(r["direction_LCRC"]).lower() not in ("true", "1"),
        "pair_accurate": lambda r: bool(int(r["pair_accurate_LCRC"])),
        "pair_inaccurate": lambda r: not bool(int(r["pair_accurate_LCRC"])),
    }
    # Use model-specific direction/accuracy when evaluating that model.
    for model in ("LCRC", "GINE"):
        for group in ("direction_correct", "direction_wrong", "pair_accurate", "pair_inaccurate"):
            if group.startswith("direction"):
                mask = np.asarray([bool(int(r["direction_evaluable"])) for r in pairs], dtype=bool)
                key = f"direction_{model}"
                ok = np.asarray([str(r[key]).lower() in ("true", "1") for r in pairs], dtype=bool)
                mask = mask & (ok if group == "direction_correct" else ~ok)
            else:
                key = f"pair_accurate_{model}"
                ok = np.asarray([bool(int(r[key])) for r in pairs], dtype=bool)
                mask = ok if group == "pair_accurate" else ~ok
            for metric in METRICS:
                vals = np.asarray([_f(r.get(f"{metric}_{model}")) if _f(r.get(f"{metric}_{model}")) is not None else np.nan for r in pairs], dtype=np.float64)
                boot = _boot(vals, comps, mask)
                rows.append({
                    "model": model, "group": group, "metric": metric, "unit": "pairs",
                    "n_units": boot["n"], "n_components": boot["n_components"], "support": boot["support"],
                    "point": boot["point"], "ci_low": boot["ci95"][0], "ci_high": boot["ci95"][1],
                })
            # oracle enrichment from endpoint rows in the same pair groups
            pair_ids = {str(r["pair_id"]) for r, keep in zip(pairs, mask) if keep}
            osub = [r for r in oracle if str(r["pair_id"]) in pair_ids and r["model"] == model]
            if osub:
                ocomps = np.asarray([int(r["component"]) for r in osub], dtype=np.int64)
                vals = np.asarray([_f(r.get("oracle_enrichment")) if _f(r.get("oracle_enrichment")) is not None else np.nan for r in osub], dtype=np.float64)
                boot = _boot(vals, ocomps, np.ones(len(osub), dtype=bool))
                rows.append({
                    "model": model, "group": group, "metric": "oracle_enrichment", "unit": "endpoints",
                    "n_units": boot["n"], "n_components": boot["n_components"], "support": boot["support"],
                    "point": boot["point"], "ci_low": boot["ci95"][0], "ci_high": boot["ci95"][1],
                })

    path = C.ANALYSIS_DIR / "stratified_localization.csv"
    cols = ["model", "group", "metric", "unit", "n_units", "n_components", "support", "point", "ci_low", "ci_high"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row[k] is None else (f"{row[k]:.10g}" if isinstance(row[k], float) else row[k])) for k in cols})
    print({"path": str(path.relative_to(C.ROUND_DIR)), "n": len(rows), "sha256": C.sha256_file(path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
