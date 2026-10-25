"""Analysis and frozen decision for zinc-p3-chromatic-bond-v0.

Run after both formal runs are pulled:

    uv run python tracks/structural-sufficiency/zinc_p3_chromatic_bond_v0/code/chr_analysis.py

Writes analysis/main_table.json, analysis/decision.json, analysis/timing.json,
analysis/curves/*.csv and analysis/per_molecule_valid.csv.
Official test is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SS = _EXP.parent
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

EXP = _EXP
ANALYSIS = EXP / "analysis"
CURVES = ANALYSIS / "curves"
CONFIG = EXP / "configs" / "chr_v0.json"
DCC_RUN = _SS / "zinc_dual_channel_composer_v1" / "runs" / "L_s0"
DCC_ANCHOR_SEED0 = 0.07853714964160463
DCC_ANCHOR_4SEED = 0.075990
ORDINARY_CUT = -3.0
BOOT_REPS = 5000
BOOT_SEED = 20261026


def _cfg() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(arm: str) -> Path:
    return EXP / "runs" / f"{arm}_s0"


def _curve_median(curve: list[dict[str, Any]], lo: int, hi: int) -> tuple[float, int]:
    vals = [float(e["valid_mae"]) for e in curve if lo <= int(e["epoch"]) <= hi]
    return (float(np.median(vals)) if vals else float("nan")), len(vals)


def _w20(curve: list[dict[str, Any]]) -> tuple[float, int, list[dict[str, Any]]]:
    by_epoch = {int(e["epoch"]): float(e["valid_mae"]) for e in curve}
    windows = []
    for start in range(1, 600 - 20 + 2):
        vals = [by_epoch.get(t) for t in range(start, start + 20)]
        if any(v is None or not math.isfinite(v) for v in vals):
            continue
        windows.append({"start": start, "end": start + 19, "median": float(np.median(vals))})
    if not windows:
        return float("nan"), 0, []
    best = min(windows, key=lambda w: w["median"])
    return float(best["median"]), int(best["start"]), windows


def _pred_map(path: Path, n: int = 1000) -> np.ndarray:
    z = np.load(path)
    out = np.full(n, np.nan, dtype=np.float64)
    for r, p in zip(z["row"].astype(np.int64), z["pred"].astype(np.float64)):
        if 0 <= int(r) < n:
            out[int(r)] = float(p)
    if not np.isfinite(out).all():
        raise ValueError(f"missing predictions in {path}")
    return out


def _load_arm(arm: str) -> dict[str, Any]:
    run = _run(arm)
    train = _load_json(run / "train.json")
    curve = train["curve"]
    w20, w20_start, windows = _w20(curve)
    tail100, tail_n = _curve_median(curve, 501, 600)
    pred = _pred_map(run / "predictions" / "valid_best.npz")
    diag = None
    if (run / "diagnostics.json").exists():
        diag = _load_json(run / "diagnostics.json")
    return {
        "arm": arm,
        "run_dir": str(run.relative_to(EXP)),
        "train": train,
        "curve": curve,
        "w20": w20,
        "w20_start": w20_start,
        "w20_windows": windows,
        "tail100": tail100,
        "tail100_n": tail_n,
        "pred": pred,
        "diagnostics": diag,
    }


def _stats(curve: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(e["valid_mae"]) for e in curve]
    diffs = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    after10 = [abs(vals[i] - vals[i - 1]) for i in range(10, len(vals))]
    return {
        "finite": bool(all(math.isfinite(v) for v in vals)),
        "max_abs_step": float(max(diffs)) if diffs else None,
        "max_abs_step_after_epoch10": float(max(after10)) if after10 else None,
        "min_valid": float(min(vals)) if vals else None,
        "max_valid": float(max(vals)) if vals else None,
    }


def _timing_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    per = {}
    for arm in ("C", "S"):
        p = ANALYSIS / f"timing_{arm}.json"
        if p.exists():
            per[arm] = _load_json(p)
    complete = set(per) == {"C", "S"}
    merged = {
        "round_id": cfg["round_id"],
        "per_arm": per,
        "complete": complete,
        "soft_budget_a100_hours": float(cfg["budget"]["soft_a100_hours"]),
        "hard_budget_a100_hours": float(cfg["budget"]["hard_a100_hours"]),
        "projection_rule": cfg["budget"]["projection_margin"],
        "official_test_loaded": False,
        "utc": C.utcnow(),
    }
    if complete:
        projected = float(per["C"]["projected_600ep_gpu_hours"]) + float(per["S"]["projected_600ep_gpu_hours"])
        merged["projected_total_a100_hours"] = projected
        merged["within_soft_budget"] = projected <= float(cfg["budget"]["soft_a100_hours"])
        merged["within_hard_budget"] = projected <= float(cfg["budget"]["hard_a100_hours"])
        merged["stop_before_gpu"] = projected > float(cfg["budget"]["hard_a100_hours"])
    C.write_json(ANALYSIS / "timing.json", merged)
    return merged


def _bootstrap(a: np.ndarray, b: np.ndarray, reps: int = BOOT_REPS,
               seed: int = BOOT_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(a)
    delta = a - b
    means = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        means[i] = delta[idx].mean()
    return {
        "n": int(n),
        "mean_a_minus_b": float(delta.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "prob_a_better_than_b": float((means < 0).mean()),
        "reps": int(reps),
        "seed": int(seed),
        "note": "paired nonparametric bootstrap over valid molecules; descriptive only",
    }


def _subgroup(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    out = {}
    for name, mask in (("ordinary", y > ORDINARY_CUT), ("tail", y <= ORDINARY_CUT)):
        out[name] = {
            "n": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mae_a": float(np.abs(pred_a[mask] - y[mask]).mean()) if mask.any() else None,
            "mae_b": float(np.abs(pred_b[mask] - y[mask]).mean()) if mask.any() else None,
        }
        if mask.any():
            out[name]["delta_a_minus_b"] = float(np.abs(pred_a[mask] - y[mask]).mean()
                                                  - np.abs(pred_b[mask] - y[mask]).mean())
    return out


def _write_curve(name: str, curve: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    CURVES.mkdir(parents=True, exist_ok=True)
    p = CURVES / f"{name}_valid.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "valid_mae", "online_train_mae", "lr", "seconds", "steps"])
        for e in curve:
            w.writerow([int(e["epoch"]), repr(float(e["valid_mae"])),
                        repr(float(e["online_train_mae"])), repr(float(e["lr"])),
                        repr(float(e["seconds"])), int(e["steps"])])
    C.write_json(CURVES / f"{name}_valid.json", {
        "name": name,
        "meta": meta,
        "n_epochs": len(curve),
        "official_test_loaded": False,
    })


def analyse() -> dict[str, Any]:
    cfg = _cfg()
    norm = D.load_norm()
    valid_cache = D.load_cache("valid")
    y = np.asarray(valid_cache["labels"], dtype=np.float64).reshape(-1)
    if y.shape[0] != 1000:
        raise ValueError(f"valid labels shape {y.shape}; expected 1000")

    arms = {a: _load_arm(a) for a in ("C", "S")}
    dcc_train = _load_json(DCC_RUN / "train.json")
    dcc_pred = _pred_map(DCC_RUN / "predictions" / "valid_best.npz")

    timing = _timing_summary(cfg)
    if not timing["complete"]:
        raise RuntimeError("timing_C.json / timing_S.json missing; run timing for both arms first")

    rows = np.arange(1000, dtype=np.int64)
    ae = {a: np.abs(arms[a]["pred"] - y) for a in ("C", "S")}
    ae["DCCL"] = np.abs(dcc_pred - y)
    stats = {a: _stats(arms[a]["curve"]) for a in ("C", "S")}

    C_best = float(arms["C"]["train"]["best_valid"]["valid_mae"])
    S_best = float(arms["S"]["train"]["best_valid"]["valid_mae"])
    C_W20 = float(arms["C"]["w20"])
    S_W20 = float(arms["S"]["w20"])
    gain = S_best - C_best
    gain_W20 = S_W20 - C_W20
    competitive_C = C_best <= DCC_ANCHOR_SEED0 and C_W20 <= 0.080000
    competitive_S = S_best <= DCC_ANCHOR_SEED0 and S_W20 <= 0.080000

    sub = {a: _subgroup(y, arms[a]["pred"], arms["S"]["pred"] if a == "C" else arms["C"]["pred"])
           for a in ("C", "S")}
    # Explicit clean subgroup table for the C-vs-S contrast.
    c_ord = float(np.abs(arms["C"]["pred"][y > ORDINARY_CUT] - y[y > ORDINARY_CUT]).mean())
    s_ord = float(np.abs(arms["S"]["pred"][y > ORDINARY_CUT] - y[y > ORDINARY_CUT]).mean())
    c_tail = float(np.abs(arms["C"]["pred"][y <= ORDINARY_CUT] - y[y <= ORDINARY_CUT]).mean())
    s_tail = float(np.abs(arms["S"]["pred"][y <= ORDINARY_CUT] - y[y <= ORDINARY_CUT]).mean())
    catastrophic = bool(c_ord > s_ord + 0.002 or c_tail > s_tail + 0.010)

    boots = {
        "C_minus_S": _bootstrap(ae["C"], ae["S"]),
        "C_minus_DCCL": _bootstrap(ae["C"], ae["DCCL"]),
        "S_minus_DCCL": _bootstrap(ae["S"], ae["DCCL"]),
    }

    worst10 = np.argsort(ae["DCCL"])[-10:].astype(np.int64)
    keep = np.ones(1000, dtype=bool)
    keep[worst10] = False
    dcc_worst10 = {
        "rows": [int(x) for x in worst10],
        "mae_C_keep": float(ae["C"][keep].mean()),
        "mae_S_keep": float(ae["S"][keep].mean()),
        "mae_DCCL_keep": float(ae["DCCL"][keep].mean()),
        "n_keep": int(keep.sum()),
        "note": "descriptive fixed-set removal; not a formal test",
    }

    diag = {}
    for a in ("C", "S"):
        d = arms[a]["diagnostics"]
        diag[a] = None if d is None else {
            "checkpoint_sha256": d.get("checkpoint_sha256"),
            "epoch": d.get("epoch"),
            "stored_valid_mae": d.get("stored_valid_mae"),
            "n_rows": d.get("n_rows"),
            "ring_off_mae_delta": d.get("ring_off_mae_delta"),
            "ring_off_mean_abs_pred_delta": d.get("ring_off_mean_abs_pred_delta"),
            "ring_off_max_abs_pred_delta": d.get("ring_off_max_abs_pred_delta"),
            "rel_channel_weight_rms": d.get("rel_channel_weight_rms"),
            "layer_stats": d.get("layer_stats"),
            "train_eval_max_abs_delta": d.get("train_eval_max_abs_delta"),
        }

    # Stability screen (frozen in PROTOCOL.md section 7).
    stability = {
        "curve_finite": all(stats[a]["finite"] for a in ("C", "S")),
        "max_abs_step_after_epoch10": {
            a: stats[a]["max_abs_step_after_epoch10"] for a in ("C", "S")},
        "jump_gt_0.05_after_epoch10": any(
            (stats[a]["max_abs_step_after_epoch10"] or 0.0) > 0.05 for a in ("C", "S")),
        "best_valid_gt_0.15": any(
            float(arms[a]["train"]["best_valid"]["valid_mae"]) > 0.15 for a in ("C", "S")),
        "train_eval_best_gt_0.03": any(
            float(arms[a]["train"]["fixed_objects"]["best"].get("train_mae_eval_mode", 0.0)) > 0.03
            for a in ("C", "S")),
        "train_eval_final_gt_0.03": any(
            float(arms[a]["train"]["fixed_objects"]["final"].get("train_mae_eval_mode", 0.0)) > 0.03
            for a in ("C", "S")),
        "train_eval_identity": {},
    }
    for a in ("C", "S"):
        d = arms[a]["diagnostics"]
        stability["train_eval_identity"][a] = (
            None if d is None else d.get("train_eval_max_abs_delta"))
    if any(v is not None and float(v) > 1e-5 for v in stability["train_eval_identity"].values()):
        stability["train_eval_mismatch"] = True
    else:
        stability["train_eval_mismatch"] = False
    stability["fired"] = bool(
        (not stability["curve_finite"])
        or stability["jump_gt_0.05_after_epoch10"]
        or stability["best_valid_gt_0.15"]
        or stability["train_eval_best_gt_0.03"]
        or stability["train_eval_final_gt_0.03"]
        or stability["train_eval_mismatch"]
    )

    # Frozen decision order.
    if stability["fired"]:
        category = "UNSTABLE_OR_UNDERFIT"
        rule = "stability screen"
    elif (gain >= 0.003 and C_W20 <= S_W20 and C_best <= 0.073 and C_W20 <= 0.075
          and not catastrophic):
        category = "COLOR_GAIN_AND_COMPETITIVE"
        rule = "rule 1"
    elif competitive_C or competitive_S:
        category = "COMPETITIVE_COLOR_UNPROVEN"
        rule = "rule 2"
    elif gain >= 0.003 and C_W20 <= S_W20:
        category = "COLOR_SIGNAL_ONLY"
        rule = "rule 3"
    else:
        category = "NO_GAIN"
        rule = "rule 4"

    decision = {
        "round_id": cfg["round_id"],
        "category": category,
        "rule_fired": rule,
        "stability": stability,
        "metrics": {
            "C_best": C_best,
            "S_best": S_best,
            "C_W20": C_W20,
            "S_W20": S_W20,
            "C_W20_start": arms["C"]["w20_start"],
            "S_W20_start": arms["S"]["w20_start"],
            "gain_best_S_minus_C": gain,
            "gain_W20_S_minus_C": gain_W20,
            "C_best_epoch": int(arms["C"]["train"]["best_valid"]["epoch"]),
            "S_best_epoch": int(arms["S"]["train"]["best_valid"]["epoch"]),
            "C_tail100_median": float(arms["C"]["tail100"]),
            "S_tail100_median": float(arms["S"]["tail100"]),
            "C_final": float(arms["C"]["curve"][-1]["valid_mae"]),
            "S_final": float(arms["S"]["curve"][-1]["valid_mae"]),
            "C_ordinary_MAE": c_ord,
            "S_ordinary_MAE": s_ord,
            "C_tail_MAE": c_tail,
            "S_tail_MAE": s_tail,
            "catastrophic_ordinary_or_tail": catastrophic,
            "competitive_C": bool(competitive_C),
            "competitive_S": bool(competitive_S),
            "dcc_l_seed0_anchor": DCC_ANCHOR_SEED0,
            "dcc_l_4seed_mean_anchor": DCC_ANCHOR_4SEED,
        },
        "conditions": {
            "gain_ge_0.003": bool(gain >= 0.003),
            "C_W20_le_S_W20": bool(C_W20 <= S_W20),
            "C_best_le_0.073": bool(C_best <= 0.073),
            "C_W20_le_0.075": bool(C_W20 <= 0.075),
            "no_catastrophic_degradation": bool(not catastrophic),
            "competitive_C": bool(competitive_C),
            "competitive_S": bool(competitive_S),
        },
        "subgroups_C_vs_S": sub["C"],
        "bootstrap": boots,
        "dcc_worst10_removal": dcc_worst10,
        "diagnostics": diag,
        "timing": {
            "projected_total_a100_hours": timing.get("projected_total_a100_hours"),
            "within_soft_budget": timing.get("within_soft_budget"),
            "within_hard_budget": timing.get("within_hard_budget"),
        },
        "official_test_loaded": False,
        "caveats": [
            "Single seed; not a family-level causal claim.",
            "C and S share tensors and initial state but do not have matched effective capacity.",
            "DCC-L anchors are historical, not paired random variables.",
            "The CSA layer is prior public work; this round is not a CSA reproduction.",
            "No official test read; valid is the exposed dev split.",
        ],
        "utc": C.utcnow(),
    }

    main_table = {
        "round_id": cfg["round_id"],
        "rows": {
            "C": {
                "params": int(arms["C"]["train"]["parameter_count"]),
                "best_valid": C_best,
                "best_epoch": int(arms["C"]["train"]["best_valid"]["epoch"]),
                "W20": C_W20,
                "W20_start": arms["C"]["w20_start"],
                "tail100_median": float(arms["C"]["tail100"]),
                "final_valid": float(arms["C"]["curve"][-1]["valid_mae"]),
                "train_eval_best": float(arms["C"]["train"]["fixed_objects"]["best"].get("train_mae_eval_mode")),
                "train_eval_final": float(arms["C"]["train"]["fixed_objects"]["final"].get("train_mae_eval_mode")),
                "gpu_hours": float(arms["C"]["train"]["gpu_hours"]),
                "wall_seconds": float(arms["C"]["train"]["wall_seconds"]),
            },
            "S": {
                "params": int(arms["S"]["train"]["parameter_count"]),
                "best_valid": S_best,
                "best_epoch": int(arms["S"]["train"]["best_valid"]["epoch"]),
                "W20": S_W20,
                "W20_start": arms["S"]["w20_start"],
                "tail100_median": float(arms["S"]["tail100"]),
                "final_valid": float(arms["S"]["curve"][-1]["valid_mae"]),
                "train_eval_best": float(arms["S"]["train"]["fixed_objects"]["best"].get("train_mae_eval_mode")),
                "train_eval_final": float(arms["S"]["train"]["fixed_objects"]["final"].get("train_mae_eval_mode")),
                "gpu_hours": float(arms["S"]["train"]["gpu_hours"]),
                "wall_seconds": float(arms["S"]["train"]["wall_seconds"]),
            },
            "DCC_L_seed0_historical": {
                "params": int(dcc_train["parameter_count"]),
                "best_valid": float(dcc_train["best_valid"]["valid_mae"]),
                "best_epoch": int(dcc_train["best_valid"]["epoch"]),
                "W20": None,
                "tail100_median": None,
                "final_valid": float(dcc_train["curve"][-1]["valid_mae"]),
                "train_eval_best": float(dcc_train["fixed_objects"]["best"].get("train_mae_eval_mode")),
                "train_eval_final": float(dcc_train["fixed_objects"]["final"].get("train_mae_eval_mode")),
                "gpu_hours": float(dcc_train["gpu_hours"]),
                "wall_seconds": float(dcc_train["wall_seconds"]),
                "note": "historical same-seed anchor; not paired with this round",
            },
        },
        "anchors": {
            "DCC_L_seed0_valid": DCC_ANCHOR_SEED0,
            "DCC_L_4seed_mean_valid": DCC_ANCHOR_4SEED,
            "note": "historical background only; no official test read",
        },
        "paired": {
            "gain_best_S_minus_C": gain,
            "gain_W20_S_minus_C": gain_W20,
            "bootstrap": boots,
            "subgroups": {
                "ordinary": {"n": int((y > ORDINARY_CUT).sum()),
                             "C_mae": c_ord, "S_mae": s_ord,
                             "C_minus_S": c_ord - s_ord},
                "tail": {"n": int((y <= ORDINARY_CUT).sum()),
                         "C_mae": c_tail, "S_mae": s_tail,
                         "C_minus_S": c_tail - s_tail},
            },
            "dcc_worst10_removal": dcc_worst10,
        },
        "stability": stability,
        "diagnostics": diag,
        "timing": timing,
        "decision": category,
        "official_test_loaded": False,
        "utc": C.utcnow(),
    }

    C.write_json(ANALYSIS / "decision.json", decision)
    C.write_json(ANALYSIS / "main_table.json", main_table)

    # curves
    _write_curve("C", arms["C"]["curve"], {"arm": "C", "run": arms["C"]["run_dir"]})
    _write_curve("S", arms["S"]["curve"], {"arm": "S", "run": arms["S"]["run_dir"]})
    _write_curve("DCC_L", dcc_train["curve"],
                 {"arm": "DCC_L", "run": "zinc_dual_channel_composer_v1/runs/L_s0",
                  "note": "historical reference"})

    # per-molecule valid predictions
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with (ANALYSIS / "per_molecule_valid.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "y", "pred_C", "pred_S", "pred_DCC_L",
                    "ae_C", "ae_S", "ae_DCC_L", "subgroup",
                    "C_minus_S_ae", "C_minus_DCCL_ae"])
        for i in rows:
            w.writerow([int(i), repr(float(y[i])), repr(float(arms["C"]["pred"][i])),
                        repr(float(arms["S"]["pred"][i])), repr(float(dcc_pred[i])),
                        repr(float(ae["C"][i])), repr(float(ae["S"][i])), repr(float(ae["DCCL"][i])),
                        "ordinary" if y[i] > ORDINARY_CUT else "tail",
                        repr(float(ae["C"][i] - ae["S"][i])),
                        repr(float(ae["C"][i] - ae["DCCL"][i]))])

    print(json.dumps({"decision": category, "rule": rule,
                      "C_best": C_best, "S_best": S_best,
                      "C_W20": C_W20, "S_W20": S_W20,
                      "gain": gain, "projected_a100_h": timing.get("projected_total_a100_hours")},
                     indent=2))
    return main_table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timing-only", action="store_true",
                   help="only merge timing_C/timing_S into timing.json")
    args = p.parse_args(argv)
    cfg = _cfg()
    if args.timing_only:
        print(json.dumps(_timing_summary(cfg), indent=2))
    else:
        analyse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
