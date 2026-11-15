"""Assemble a compact JSON-only summary table across every present arm.

No tensors are loaded, so this stays well within local RAM.  Each arm reports
its Validation-selected grid point (protocol-primary) and its best-Val epoch
(auxiliary), plus a cost line from ``metrics.timing``.
"""

from __future__ import annotations

import json
from pathlib import Path

import se_common as C

RESULTS = C.ANALYSIS / "results"


def _load(p: Path):
    return json.loads(p.read_text())


def arm_summary(name: str) -> dict:
    d = RESULTS / name
    out: dict = {"arm": name}
    m = d / "metrics.json"
    if m.is_file():
        mj = _load(m)
        bv = mj["summary"]["best_valid"]
        out["best_valid"] = {"epoch": bv["epoch"],
                             "valid_rmse": bv["valid"]["rmse"], "test_rmse": bv["test"]["rmse"],
                             "valid_mae": bv["valid"]["mae"], "test_mae": bv["test"]["mae"]}
        out["family"] = mj.get("family"); out["recipe"] = mj.get("recipe")
        out["dataset"] = mj.get("dataset"); out["budget"] = mj.get("budget")
        out["gpu_hours"] = mj["timing"].get("gpu_hours_charged")
        out["mean_epoch_s"] = mj["timing"].get("mean_epoch_seconds")
    g = d / "grid.json"
    if g.is_file():
        sel = _load(g)["selection"]
        out["grid"] = {"j": sel["j"], "update": sel["update"],
                       "valid_rmse": sel["valid_rmse"], "test_rmse": sel["test_rmse"]}
    return out


def group_key(name: str) -> str:
    parts = name.split("_")
    # family_budget_dataset_recipe
    if len(parts) == 4:
        fam, budget, ds, recipe = parts
        return f"{ds}|{budget}|{recipe}"
    return name


def build() -> dict:
    arms = sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and (p / "metrics.json").is_file())
    summ = {a: arm_summary(a) for a in arms}
    groups: dict[str, list[str]] = {}
    for a in arms:
        groups.setdefault(group_key(a), []).append(a)
    return {"arms": summ, "groups": groups, "n_arms": len(arms)}


if __name__ == "__main__":
    payload = build()
    C.write_json(C.ANALYSIS / "tables_all_arms.json", payload)
    print(f"{'arm':20s} {'sel_j':>5s} {'val':>7s} {'test':>7s}   {'best_ep':>7s} {'bval':>7s} {'btest':>7s}")
    for a, s in payload["arms"].items():
        g = s.get("grid", {}); b = s.get("best_valid", {})
        print(f"{a:20s} {g.get('j',-1):5d} {g.get('valid_rmse',float('nan')):7.4f} {g.get('test_rmse',float('nan')):7.4f}   "
              f"{b.get('epoch',-1):7d} {b.get('valid_rmse',float('nan')):7.4f} {b.get('test_rmse',float('nan')):7.4f}")
