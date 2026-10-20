"""Build the main comparison table and the fixed-object P3 comparison.

Reads only training logs and already-frozen score files; it never loads test
labels or test graphs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import zpp_common as C
import zpp_zsrt as Z


def _load_zsrt_rows(variants: list[str]) -> list[dict]:
    rows = []
    for name in variants:
        variant = Z.load_variant(name)
        for seed_dir in sorted(C.RUNS.glob(f"zsrt_{name}_s*")):
            info_path = seed_dir / "train.json"
            if not info_path.exists():
                continue
            info = C.read_json(info_path)
            rows.append({
                "model": "ZSRT",
                "config": name,
                "seed": int(info["seed"]),
                "parameters": int(info["parameter_count"]),
                "pretraining": "none",
                "batch_size": int(info["recipe"]["batch_size"]),
                "max_epochs": int(info["recipe"]["epochs"]),
                "actual_epochs": int(len(info["curve"])),
                "updates": int(info["recipe"]["updates"]),
                "best_epoch": int(info["best_valid"]["epoch"]),
                "best_valid_mae": float(info["best_valid"]["valid_mae"]),
                "final_valid_mae": float(info["fixed_objects"]["final"]["valid_mae"]),
                "soup_valid_mae": float(info["fixed_objects"]["soup"]["valid_mae"]),
                "best_train_mae_eval_mode": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
                "final_train_mae_eval_mode": float(info["fixed_objects"]["final"]["train_mae_eval_mode"]),
                "gpu_hours": info.get("gpu_hours"),
                "wall_seconds": float(info["wall_seconds"]),
                "rrwp": bool(info["variant"]["rrwp"]),
                "ksteps": int(info["variant"]["ksteps"]),
            })
    return rows


def _load_p3_rows() -> list[dict]:
    rows = []
    for seed in (0, 1):
        path = C.RUNS / f"p3_s{seed}" / "train.json"
        if not path.exists():
            continue
        info = C.read_json(path)
        rows.append({
            "model": "P3",
            "config": "R3",
            "seed": int(seed),
            "parameters": int(info["parameter_count"]),
            "pretraining": "frozen train-only masked SSL",
            "batch_size": int(info["recipe"]["batch_size"]),
            "max_epochs": int(info["recipe"]["epochs"]),
            "actual_epochs": int(len(info["curve"])),
            "updates": int(info["recipe"]["updates"]),
            "best_epoch": int(info["best_valid"]["epoch"]),
            "best_valid_mae": float(info["best_valid"]["valid_mae"]),
            "final_valid_mae": float(info["fixed_objects"]["final"]["valid_mae"]),
            "soup_valid_mae": float(info["fixed_objects"]["soup"]["valid_mae"]),
            "best_train_mae_eval_mode": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
            "final_train_mae_eval_mode": float(info["fixed_objects"]["final"]["train_mae_eval_mode"]),
            "gpu_hours": info.get("gpu_hours"),
            "wall_seconds": float(info["wall_seconds"]),
            "rrwp": False,
            "ksteps": 0,
        })
    return rows


def build(variants: list[str]) -> dict:
    p3 = _load_p3_rows()
    zsrt = _load_zsrt_rows(variants)
    test_path = C.ANALYSIS / "test_scores.json"
    test = C.read_json(test_path) if test_path.exists() else None
    if test is not None:
        for row in p3:
            key = str(row["seed"])
            if key in test["rows"]["p3"]:
                row["test_mae"] = test["rows"]["p3"][key]["test_mae"]
                row["test_rmse"] = test["rows"]["p3"][key]["test_rmse"]
        for row in zsrt:
            key = str(row["seed"])
            if key in test["rows"]["zsrt"] and test["config_name"] == row["config"]:
                row["test_mae"] = test["rows"]["zsrt"][key]["test_mae"]
                row["test_rmse"] = test["rows"]["zsrt"][key]["test_rmse"]

    # Screening table (seed 0) and final comparison on shared seeds.
    screen = [r for r in zsrt if r["seed"] == 0]
    p3_best = {r["seed"]: r["best_valid_mae"] for r in p3}
    paired = {}
    for r in zsrt:
        if r["config"] == (test["config_name"] if test else None) and r["seed"] in p3_best:
            paired[str(r["seed"])] = r["best_valid_mae"] - p3_best[r["seed"]]

    out = {
        "protocol": C.PROTOCOL_ID,
        "rows": p3 + zsrt,
        "p3_reference": p3,
        "zsrt_screen_seed0": screen,
        "zsrt_all": zsrt,
        "paired_valid_delta_zsrt_minus_p3": paired,
        "paired_valid_mean_delta": float(np.mean(list(paired.values()))) if paired else None,
        "test": test,
        "note": (
            "best_valid_mae is the per-seed selector; final/soup valid are fixed-object "
            "diagnostics. P3 soup is not a new-protocol selector."
        ),
    }
    C.write_json(C.ANALYSIS / "main_table.json", out)
    C.log(f"main table: {len(p3)} P3 rows, {len(zsrt)} ZSRT rows, test={test is not None}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", default="A_main,B_lr_low,C_wd_high,D_no_rrwp")
    args = p.parse_args(argv)
    build([v.strip() for v in args.variants.split(",") if v.strip()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
