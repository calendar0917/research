"""Valid-only analysis and seed/promotion logic for ECLC-v1.

Reads the completed run directories, replays every best checkpoint on the
official valid split, builds the frozen slices, computes paired bootstrap
contrasts (REAL - NULL; negative favours REAL), applies the frozen seed-0
promotion rule, and writes:

  analysis/main_table.{json,md}
  analysis/slices.{json,md}
  analysis/promotion.json
  analysis/analysis.md

The official test split is never constructed, loaded, predicted or scored.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import eclc_common as E
import eclc_models as EM
import eclc_train as T

D = E.D
L = E.L

# Reuse only the frozen anchor/mask/MAE/bootstrap definitions from LCRC-v1.
import lc_analysis as LA  # noqa: E402

HISTORICAL_LCRC_A_VALID = 0.075990
HISTORICAL_DRIFT_BAND = 0.0026
PROMOTION = {
    "full_max": 0.075,
    "full_delta_max": -0.003,
    "r2_unseen_ordinary_delta_max": -0.005,
    "non_worse_slack": 0.001,
    "base_ablation_min": 1e-4,
}


def _load_run(run_dir: Path) -> dict[str, Any]:
    info = E.read_json(run_dir / "train.json")
    arm = str(info["arm"])
    seed = int(info["seed"])
    valid = E.load_valid_cache()
    y = np.asarray(valid["labels"], dtype=np.float64)
    best_npz = np.load(run_dir / "predictions" / "valid_best.npz")
    pred_best = np.asarray(best_npz["pred"], dtype=np.float64)
    rows = np.asarray(best_npz["row"], dtype=np.int64)
    if not np.array_equal(rows, np.arange(len(y))):
        raise RuntimeError(f"{run_dir}: valid rows are not 0..n-1")
    mae = LA._mae(pred_best, y)
    declared = float(info["best"]["valid_mae"])
    if abs(mae - declared) > 1e-6:
        raise RuntimeError(f"{run_dir}: replay {mae} vs declared {declared}")
    final_npz = np.load(run_dir / "predictions" / "valid_final.npz")
    final_pred = np.asarray(final_npz["pred"], dtype=np.float64)
    final_mae = LA._mae(final_pred, y)
    if abs(final_mae - float(info["final_valid_mae"])) > 1e-6:
        raise RuntimeError(f"{run_dir}: final replay mismatch")
    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if int(ckpt["seed"]) != seed or str(ckpt["arm"]) != arm:
        raise RuntimeError(f"{run_dir}: checkpoint arm/seed mismatch")
    if abs(float(ckpt["valid_mae"]) - declared) > 1e-6:
        raise RuntimeError(f"{run_dir}: checkpoint valid_mae mismatch")
    return {
        "dir": str(run_dir),
        "run_id": run_dir.name,
        "arm": arm,
        "seed": seed,
        "info": info,
        "y": y,
        "pred_best": pred_best,
        "pred_final": final_pred,
        "best_valid": mae,
        "final_valid": final_mae,
        "checkpoint": ckpt_path,
        "checkpoint_sha256": E.sha256_file(ckpt_path),
        "state": ckpt["state_dict"],
    }


def _slices():
    anchor = LA._read_anchor()
    masks, worst_rows = LA._fixed_masks(anchor)
    masks = dict(masks)
    masks["seen_ordinary"] = (~anchor["unseen"]) & (anchor["y"] > -3.0)
    return anchor, masks, worst_rows


def _slice_table(run: Mapping[str, Any], masks: Mapping[str, np.ndarray]) -> dict[str, dict]:
    y = run["y"]
    out = {}
    for name, mask in masks.items():
        out[name] = {
            "n": int(mask.sum()),
            "mae": float(LA._mae(run["pred_best"], y, mask)),
        }
    return out


def _zero_base_ablation(run: Mapping[str, Any]) -> dict[str, Any]:
    norm = E.load_norm()
    valid = E.load_valid_cache()
    rows = np.arange(len(valid["labels"]), dtype=np.int64)
    model = EM.load_model_from_state(run["arm"], norm, run["state"], "cpu")
    pred, mae, aux = T.evaluate(model, valid, rows, device="cpu", batch_size=256,
                                return_aux=True)
    base = np.asarray(aux["base"], dtype=np.float64)
    base_std = float(np.std(base))
    base_abs_mean = float(np.mean(np.abs(base)))
    with torch.no_grad():
        for p in model.base.parameters():
            p.zero_()
    pred_zero, mae_zero = T.evaluate(model, valid, rows, device="cpu", batch_size=256)
    delta = float(abs(mae_zero - mae))
    max_delta_pred = float(np.max(np.abs(pred_zero.astype(np.float64) - pred.astype(np.float64))))
    return {
        "valid_mae_with_base": float(mae),
        "valid_mae_zero_base": float(mae_zero),
        "abs_mae_change": float(delta),
        "max_abs_prediction_change": float(max_delta_pred),
        "base_per_molecule_std": float(base_std),
        "base_per_molecule_abs_mean": float(base_abs_mean),
        "base_nonzero": bool(base_std > 0),
    }


def _contrast(real: Mapping[str, Any], null: Mapping[str, Any],
              masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    y = real["y"]
    ae_real = np.abs(real["pred_best"] - y)
    ae_null = np.abs(null["pred_best"] - y)
    out = {}
    for name, mask in masks.items():
        d = ae_real[mask] - ae_null[mask]
        boot = LA._bootstrap_delta(ae_real[mask], ae_null[mask], n_boot=10000,
                                   seed=20261108 + int(real["seed"]))
        out[name] = {
            "n": int(mask.sum()),
            "mae_real": float(ae_real[mask].mean()),
            "mae_null": float(ae_null[mask].mean()),
            "delta_real_minus_null": float(d.mean()),
            "ci95": boot["ci95"],
            "win_rate_real_better": float(boot["win_rate_first_better"]),
            "median_abs_delta": float(np.median(d)),
        }
    return out


def _promotion(real: Mapping[str, Any], null: Mapping[str, Any],
               contrasts: Mapping[str, Mapping[str, Any]],
               ablation: Mapping[str, Any]) -> dict[str, Any]:
    c = contrasts
    cond = {
        "1_full_valid_at_most_0.075": {
            "pass": bool(real["best_valid"] <= PROMOTION["full_max"]),
            "value": float(real["best_valid"]),
            "threshold": float(PROMOTION["full_max"]),
        },
        "2_full_delta_at_most_-0.003": {
            "pass": bool(c["all"]["delta_real_minus_null"] <= PROMOTION["full_delta_max"]),
            "value": float(c["all"]["delta_real_minus_null"]),
            "threshold": float(PROMOTION["full_delta_max"]),
        },
        "3_r2_unseen_ordinary_delta_at_most_-0.005": {
            "pass": bool(c["r2_unseen_ordinary"]["delta_real_minus_null"]
                         <= PROMOTION["r2_unseen_ordinary_delta_max"]),
            "value": float(c["r2_unseen_ordinary"]["delta_real_minus_null"]),
            "threshold": float(PROMOTION["r2_unseen_ordinary_delta_max"]),
        },
        "4_not_worse_ordinary_and_minus_row172": {
            "pass": bool(c["ordinary_y_gt_-3"]["delta_real_minus_null"]
                         <= PROMOTION["non_worse_slack"]
                         and c["all_minus_row172"]["delta_real_minus_null"]
                         <= PROMOTION["non_worse_slack"]),
            "ordinary_delta": float(c["ordinary_y_gt_-3"]["delta_real_minus_null"]),
            "minus_row172_delta": float(c["all_minus_row172"]["delta_real_minus_null"]),
            "threshold": float(PROMOTION["non_worse_slack"]),
        },
        "5_windows_and_base_path_active": {
            "pass": bool(real["info"]["W20"] is not None
                         and null["info"]["W20"] is not None
                         and real["info"]["W20"] < null["info"]["W20"]
                         and real["info"]["T100"] < null["info"]["T100"]
                         and ablation["abs_mae_change"] >= PROMOTION["base_ablation_min"]
                         and ablation["base_nonzero"]),
            "real_W20": real["info"]["W20"],
            "null_W20": null["info"]["W20"],
            "real_T100": real["info"]["T100"],
            "null_T100": null["info"]["T100"],
            "base_ablation_abs_mae_change": float(ablation["abs_mae_change"]),
            "base_ablation_threshold": float(PROMOTION["base_ablation_min"]),
            "base_per_molecule_std": float(ablation["base_per_molecule_std"]),
        },
    }
    return {
        "seed": int(real["seed"]),
        "conditions": cond,
        "all_pass": bool(all(v["pass"] for v in cond.values())),
    }


def analyse(seeds: Sequence[int], analysis_dir: Path = E.ANALYSIS) -> dict[str, Any]:
    anchor, masks, worst_rows = _slices()
    runs: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        runs[int(seed)] = {}
        for arm in E.ARMS:
            d = T.run_dir(arm, int(seed))
            if not (d / "train.json").exists():
                raise FileNotFoundError(f"missing run: {d}")
            runs[int(seed)][arm] = _load_run(d)
    main = {}
    slices = {}
    promotion = {}
    for seed, pair in runs.items():
        real, null = pair["REAL"], pair["NULL"]
        main[str(seed)] = {
            "REAL": {
                "run_id": real["run_id"], "dir": real["dir"],
                "best_valid": real["best_valid"],
                "best_epoch": int(real["info"]["best"]["epoch"]),
                "final_valid": real["final_valid"],
                "W20": real["info"]["W20"], "T100": real["info"]["T100"],
                "train_at_best": float(real["info"]["best_train_mae_eval"]),
                "parameters": int(real["info"]["param_audit"]["parameters"]),
                "updates": int(real["info"]["total_updates"]),
                "wall_seconds": float(real["info"]["wall_seconds"]),
                "gpu_hours": float(real["info"]["gpu_hours_estimate"]),
                "checkpoint_sha256": real["checkpoint_sha256"],
                "git_commit": real["info"].get("git_commit"),
            },
            "NULL": {
                "run_id": null["run_id"], "dir": null["dir"],
                "best_valid": null["best_valid"],
                "best_epoch": int(null["info"]["best"]["epoch"]),
                "final_valid": null["final_valid"],
                "W20": null["info"]["W20"], "T100": null["info"]["T100"],
                "train_at_best": float(null["info"]["best_train_mae_eval"]),
                "parameters": int(null["info"]["param_audit"]["parameters"]),
                "updates": int(null["info"]["total_updates"]),
                "wall_seconds": float(null["info"]["wall_seconds"]),
                "gpu_hours": float(null["info"]["gpu_hours_estimate"]),
                "checkpoint_sha256": null["checkpoint_sha256"],
                "git_commit": null["info"].get("git_commit"),
            },
        }
        slices[str(seed)] = {
            "REAL": _slice_table(real, masks),
            "NULL": _slice_table(null, masks),
        }
        c = _contrast(real, null, masks)
        ablation = _zero_base_ablation(real)
        p = _promotion(real, null, c, ablation)
        p["contrasts"] = c
        p["base_ablation"] = ablation
        p["seen_unseen_gap"] = {
            "REAL": float(slices[str(seed)]["REAL"]["r2_unseen_ordinary"]["mae"]
                          - slices[str(seed)]["REAL"]["seen_ordinary"]["mae"]),
            "NULL": float(slices[str(seed)]["NULL"]["r2_unseen_ordinary"]["mae"]
                          - slices[str(seed)]["NULL"]["seen_ordinary"]["mae"]),
        }
        promotion[str(seed)] = p

    seed0 = promotion.get("0")
    decision = "PENDING"
    if seed0 is not None:
        if not seed0["all_pass"]:
            decision = "STOP_NO_COMPOSITION_GAIN"
        elif len(promotion) == 1:
            decision = "SEED0_PROMOTED_PENDING_SEED1"
        else:
            def direction_ok(p):
                c = p["contrasts"]
                return (c["all"]["delta_real_minus_null"] < 0
                        and c["r2_unseen_ordinary"]["delta_real_minus_null"] < 0)
            same_direction = all(direction_ok(p) for p in promotion.values())
            seed1_pass = bool(promotion.get("1", {}).get("all_pass", False))
            if same_direction and seed1_pass:
                decision = "COMPOSITION_TRANSFER_CANDIDATE"
            elif same_direction:
                decision = "NO_TWO_SEED_CONFIRMATION"
            else:
                decision = "NO_TWO_SEED_CONFIRMATION"

    payload = {
        "round": E.ROUND_ID,
        "created": E.now(),
        "git_commit": E.git_commit(),
        "seeds": [int(s) for s in seeds],
        "historical_reference": {
            "lcrc_a_4seed_valid": HISTORICAL_LCRC_A_VALID,
            "execution_drift_band": HISTORICAL_DRIFT_BAND,
        },
        "promotion_rule": PROMOTION,
        "main": main,
        "slices": slices,
        "promotion": promotion,
        "decision": decision,
        "official_test_loaded": False,
    }
    E.ensure_dir(analysis_dir)
    E.write_json(analysis_dir / "main_table.json", payload)
    E.write_json(analysis_dir / "slices.json", {str(s): slices[str(s)] for s in seeds})
    E.write_json(analysis_dir / "promotion.json", {
        "round": E.ROUND_ID, "created": payload["created"],
        "git_commit": payload["git_commit"], "promotion_rule": PROMOTION,
        "promotion": promotion, "decision": decision,
        "historical_reference": payload["historical_reference"],
    })
    _write_markdown(payload, analysis_dir / "analysis.md")
    return payload


def _fmt(x, digits=6):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = ["# ECLC-v1 valid-only analysis", "",
             f"- Round: `{payload['round']}`",
             f"- Created: `{payload['created']}`",
             f"- Decision: **`{payload['decision']}`**",
             f"- Historical LCRC-A 4-seed valid: {HISTORICAL_LCRC_A_VALID:.6f}; "
             f"execution-drift band ≈ {HISTORICAL_DRIFT_BAND:.4f}", ""]
    for seed in payload["seeds"]:
        s = str(seed)
        m = payload["main"][s]
        lines += [f"## Seed {s}", "",
                  "| arm | run | best valid | ep | final | W20 | T100 | params | updates | GPU-h |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in ("REAL", "NULL"):
            r = m[arm]
            lines.append(
                f"| {arm} | `{r['run_id']}` | {_fmt(r['best_valid'])} | {r['best_epoch']} | "
                f"{_fmt(r['final_valid'])} | {_fmt(r['W20'])} | {_fmt(r['T100'])} | "
                f"{r['parameters']} | {r['updates']} | {r['gpu_hours']:.3f} |")
        lines += ["", "### Slices", "",
                  "| slice | n | REAL | NULL | REAL−NULL | 95% CI | REAL win |",
                  "|---|---:|---:|---:|---:|---|---:|"]
        c = payload["promotion"][s]["contrasts"]
        for name in ["all", "ordinary_y_gt_-3", "r2_unseen_ordinary", "seen_ordinary",
                     "all_minus_row172", "all_minus_worst10", "tail_y_le_-3"]:
            row = c[name]
            lines.append(
                f"| {name} | {row['n']} | {_fmt(row['mae_real'])} | {_fmt(row['mae_null'])} | "
                f"{_fmt(row['delta_real_minus_null'])} | "
                f"[{_fmt(row['ci95'][0])}, {_fmt(row['ci95'][1])}] | "
                f"{row['win_rate_real_better']:.3f} |")
        p = payload["promotion"][s]
        lines += ["", "### Promotion conditions", "",
                  "| condition | pass | value |", "|---|:---:|---|"]
        for name, v in p["conditions"].items():
            vals = "; ".join(f"{k}={_fmt(val)}" for k, val in v.items() if k != "pass")
            lines.append(f"| {name} | {'yes' if v['pass'] else 'no'} | {vals} |")
        a = p["base_ablation"]
        lines += ["", f"Base ablation: MAE {_fmt(a['valid_mae_with_base'])} → "
                  f"{_fmt(a['valid_mae_zero_base'])} (|Δ|={a['abs_mae_change']:.3e}); "
                  f"per-molecule base std={a['base_per_molecule_std']:.6f}.", ""]
    path.write_text("\n".join(lines).rstrip() + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0")
    args = ap.parse_args(argv)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out = analyse(seeds)
    print(out["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
