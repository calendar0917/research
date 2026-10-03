"""Fixed analysis for zinc-full-cycle-target-decomposition-v1 (local, CPU).

Reads the four paired arm artifacts and the frozen split/decomposition and
writes the pre-registered tables and gate.  No fitting, no probe, no new model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent
DECOMP = RESULTS_DIR / "target_decomposition.npz"
PREP = RESULTS_DIR.parent / "zinc_joint_dictionary_decision_v1" / "prep" / "fold_objects.npz"

SEEDS = (0, 1)
ARMS = ("Y", "O")
BOOT_SEED = 20261003
N_BOOT = 1000
SEVERE_MAX = -2
GATE_MEAN_GAIN = 0.003
GATE_G0_WORSEN_MAX = 0.001
GATE_G0_MEAN_GAIN = 0.002


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def load_arm(seed: int, arm: str) -> dict[str, Any]:
    meta = json.loads((RESULTS_DIR / f"{arm}_seed{seed}.json").read_text())
    preds = dict(np.load(RESULTS_DIR / f"{arm}_seed{seed}_predictions.npz", allow_pickle=False))
    return {"meta": meta, "preds": preds}


def group_name(k: int) -> str:
    return "k0" if k == 0 else ("k-1" if k == -1 else "kle-2")


def group_table(pred_cal: np.ndarray, pred_raw: np.ndarray, y: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    n_total = int(len(y))
    out: dict[str, Any] = {}
    for name, sel in (
        ("k0", k == 0),
        ("k-1", k == -1),
        ("kle-2", k <= SEVERE_MAX),
    ):
        sel = np.asarray(sel)
        if not sel.any():
            out[name] = {"n": 0, "mae": None, "raw_mae": None, "signed_residual": None, "contribution": 0.0}
            continue
        err = pred_cal[sel] - y[sel]
        out[name] = {
            "n": int(sel.sum()),
            "mae": float(np.abs(err).mean()),
            "raw_mae": float(np.abs(pred_raw[sel] - y[sel]).mean()),
            "signed_residual": float((y[sel] - pred_cal[sel]).mean()),
            "contribution": float(np.abs(err).sum() / n_total),
        }
    out["overall"] = {
        "n": n_total,
        "mae": float(np.abs(pred_cal - y).mean()),
        "raw_mae": float(np.abs(pred_raw - y).mean()),
        "signed_residual": float((y - pred_cal).mean()),
        "contribution": float(np.abs(pred_cal - y).sum() / n_total),
    }
    return out


def bootstrap_gains(
    y: np.ndarray,
    k: np.ndarray,
    gid_dev: np.ndarray,
    preds: Mapping[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    groups = np.unique(gid_dev)
    group_rows = {int(g): np.where(gid_dev == g)[0] for g in groups}
    rng = np.random.default_rng(int(seed))
    per_seed = {s: {"gain": [], "g0_gain": []} for s in SEEDS}
    mean_gain = []
    mean_g0 = []
    for _ in range(int(n_boot)):
        chosen = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([group_rows[int(groups[c])] for c in chosen])
        gains = []
        g0s = []
        for s in SEEDS:
            yy = y[idx]
            y_raw, y_cal = preds[(s, "Y")]
            o_raw, o_cal = preds[(s, "O")]
            gain = float(np.abs(y_cal[idx] - yy).mean() - np.abs(o_cal[idx] - yy).mean())
            sel0 = k[idx] == 0
            g0 = float(
                np.abs(y_cal[idx][sel0] - yy[sel0]).mean()
                - np.abs(o_cal[idx][sel0] - yy[sel0]).mean()
            )
            per_seed[s]["gain"].append(gain)
            per_seed[s]["g0_gain"].append(g0)
            gains.append(gain)
            g0s.append(g0)
        mean_gain.append(float(np.mean(gains)))
        mean_g0.append(float(np.mean(g0s)))

    def summarize(values: list[float]) -> dict[str, Any]:
        arr = np.asarray(values, np.float64)
        return {
            "mean": float(arr.mean()),
            "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)),
        }

    out: dict[str, Any] = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "unit": "canonical_group_id, joint across arms and seeds",
        "mean_gain": summarize(mean_gain),
        "mean_g0_gain": summarize(mean_g0),
        "per_seed": {s: {"gain": summarize(per_seed[s]["gain"]), "g0_gain": summarize(per_seed[s]["g0_gain"])} for s in SEEDS},
        "note": "conditional description of this diagnostic dev set and these two seeds only; not a cross-seed population CI.",
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    global RESULTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--indir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    RESULTS_DIR = Path(args.indir)

    with np.load(DECOMP, allow_pickle=False) as z:
        y_all = z["y"]
        c_all = z["c"]
        g_all = z["g"]
        k_all = z["k"]
        gid_all = z["gid"]
    with np.load(PREP, allow_pickle=False) as z:
        dev_idx = z["dev_idx"].astype(np.int64)
        fit_idx = z["fit_idx"].astype(np.int64)

    y = y_all[dev_idx]
    c = c_all[dev_idx]
    g = g_all[dev_idx]
    k = k_all[dev_idx]
    gid_dev = gid_all[dev_idx]
    n = len(y)

    arms = {(s, a): load_arm(s, a) for s in SEEDS for a in ARMS}

    # --- per-arm predictions (raw / calibrated), oracle offset for O -------
    preds: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for s in SEEDS:
        for a in ARMS:
            rec = arms[(s, a)]
            raw_model = rec["preds"]["dev_raw"].astype(np.float64)
            b = float(rec["meta"]["calibration"]["b"])
            if a == "Y":
                raw = raw_model
                cal = raw_model + b
            else:
                raw = raw_model + c
                cal = raw_model + b + c
            preds[(s, a)] = (raw, cal)

    # --- main table --------------------------------------------------------
    main_rows = []
    for s in SEEDS:
        for a in ARMS:
            rec = arms[(s, a)]
            raw, cal = preds[(s, a)]
            fit_raw = rec["preds"]["fit_raw"].astype(np.float64)
            fit_y = rec["preds"]["fit_y"].astype(np.float64)
            fit_g = rec["preds"]["fit_g"].astype(np.float64)
            fit_c = rec["preds"]["fit_c"].astype(np.float64)
            b = float(rec["meta"]["calibration"]["b"])
            fit_target = fit_y if a == "Y" else fit_g
            fit_raw_mae = float(np.abs(fit_raw - fit_target).mean())
            fit_cal_mae = float(np.abs(fit_raw + b - fit_target).mean())
            main_rows.append(
                {
                    "seed": s,
                    "arm": a,
                    "b": b,
                    "dev_raw_mae": float(np.abs(raw - y).mean()),
                    "dev_cal_mae": float(np.abs(cal - y).mean()),
                    "fit_raw_mae": fit_raw_mae,
                    "fit_cal_mae": fit_cal_mae,
                    "fit_dev_gap": float(np.abs(cal - y).mean() - fit_cal_mae),
                    "b_from_y_check": rec["meta"]["calibration"].get("b_from_y"),
                    "replay_max_abs_diff": rec["meta"].get("replay_max_abs_diff"),
                    "soup_state_sha256": rec["meta"].get("soup_state_sha256"),
                    "init_state_sha256": rec["meta"].get("init_state_sha256"),
                }
            )
    # gains
    gains = {}
    for s in SEEDS:
        y_cal = preds[(s, "Y")][1]
        o_cal = preds[(s, "O")][1]
        y_raw = preds[(s, "Y")][0]
        o_raw = preds[(s, "O")][0]
        gains[s] = {
            "cal_gain": float(np.abs(y_cal - y).mean() - np.abs(o_cal - y).mean()),
            "raw_gain": float(np.abs(y_raw - y).mean() - np.abs(o_raw - y).mean()),
        }
        for row in main_rows:
            if row["seed"] == s and row["arm"] == "Y":
                row["cal_gain_vs_O"] = gains[s]["cal_gain"]
                row["raw_gain_vs_O"] = gains[s]["raw_gain"]
            if row["seed"] == s and row["arm"] == "O":
                row["cal_gain_vs_O"] = gains[s]["cal_gain"]
                row["raw_gain_vs_O"] = gains[s]["raw_gain"]
    mean_cal_gain = float(np.mean([gains[s]["cal_gain"] for s in SEEDS]))
    mean_raw_gain = float(np.mean([gains[s]["raw_gain"] for s in SEEDS]))

    # --- group table -------------------------------------------------------
    group_rows = []
    for s in SEEDS:
        for a in ARMS:
            raw, cal = preds[(s, a)]
            table = group_table(cal, raw, y, k)
            for gname in ("k0", "k-1", "kle-2", "overall"):
                entry = table[gname]
                group_rows.append(
                    {
                        "seed": s,
                        "arm": a,
                        "group": gname,
                        "n": entry["n"],
                        "mae": entry["mae"],
                        "raw_mae": entry["raw_mae"],
                        "signed_residual": entry["signed_residual"],
                        "contribution": entry["contribution"],
                    }
                )
    # group gain (Y - O) per seed and mean
    group_gain = {}
    for s in SEEDS:
        yt = group_table(preds[(s, "Y")][1], preds[(s, "Y")][0], y, k)
        ot = group_table(preds[(s, "O")][1], preds[(s, "O")][0], y, k)
        group_gain[s] = {
            gname: {
                "mae_gain": float(yt[gname]["mae"] - ot[gname]["mae"]) if yt[gname]["mae"] is not None else None,
                "contribution_gain": float(yt[gname]["contribution"] - ot[gname]["contribution"]),
            }
            for gname in ("k0", "k-1", "kle-2", "overall")
        }
    mean_group_gain = {
        gname: float(np.mean([group_gain[s][gname]["contribution_gain"] for s in SEEDS]))
        for gname in ("k0", "k-1", "kle-2", "overall")
    }
    mean_group_mae_gain = {
        gname: float(np.mean([group_gain[s][gname]["mae_gain"] for s in SEEDS]))
        for gname in ("k0", "k-1", "kle-2")
    }
    # identity checks
    identity_checks = {}
    for s in SEEDS:
        table = group_table(preds[(s, "Y")][1], preds[(s, "Y")][0], y, k)
        contrib_sum = sum(table[gname]["contribution"] for gname in ("k0", "k-1", "kle-2"))
        gain_sum = sum(group_gain[s][gname]["contribution_gain"] for gname in ("k0", "k-1", "kle-2"))
        identity_checks[f"seed{s}"] = {
            "contribution_sum_minus_overall": float(contrib_sum - table["overall"]["contribution"]),
            "gain_sum_minus_total": float(gain_sum - gains[s]["cal_gain"]),
            "ok": bool(abs(contrib_sum - table["overall"]["contribution"]) < 1e-9
                       and abs(gain_sum - gains[s]["cal_gain"]) < 1e-9),
        }

    # --- per-row paired predictions ---------------------------------------
    rows = []
    for i in range(n):
        row = {
            "dev_pos": i,
            "global_index": int(dev_idx[i]),
            "canonical_group_id": int(gid_dev[i]),
            "k": int(k[i]),
            "y": float(y[i]),
            "c": float(c[i]),
            "g": float(g[i]),
        }
        for s in SEEDS:
            for a in ARMS:
                raw, cal = preds[(s, a)]
                row[f"{a}_s{s}_raw"] = float(raw[i])
                row[f"{a}_s{s}_cal"] = float(cal[i])
                row[f"{a}_s{s}_err_cal"] = float(cal[i] - y[i])
        rows.append(row)

    # --- severe rows -------------------------------------------------------
    sev_sel = k <= SEVERE_MAX
    severe_rows = []
    for i in np.where(sev_sel)[0]:
        row = {
            "dev_pos": int(i),
            "global_index": int(dev_idx[i]),
            "canonical_group_id": int(gid_dev[i]),
            "k": int(k[i]),
            "y": float(y[i]),
            "c": float(c[i]),
            "g": float(g[i]),
        }
        for s in SEEDS:
            for a in ARMS:
                raw, cal = preds[(s, a)]
                row[f"{a}_s{s}_raw"] = float(raw[i])
                row[f"{a}_s{s}_cal"] = float(cal[i])
                row[f"{a}_s{s}_err_cal"] = float(cal[i] - y[i])
        severe_rows.append(row)

    # --- sensitivity: drop control max-error row / drop severe ------------
    sensitivity = {}
    for s in SEEDS:
        y_cal = preds[(s, "Y")][1]
        o_cal = preds[(s, "O")][1]
        err = np.abs(y_cal - y)
        drop = int(np.argmax(err))
        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        sensitivity[f"seed{s}"] = {
            "control_max_error_row": {
                "dev_pos": int(drop),
                "global_index": int(dev_idx[drop]),
                "k": int(k[drop]),
                "y": float(y[drop]),
                "c": float(c[drop]),
                "y_cal": float(y_cal[drop]),
                "abs_err": float(err[drop]),
            },
            "gain_without_control_max_error_row": float(
                np.abs(y_cal[keep] - y[keep]).mean() - np.abs(o_cal[keep] - y[keep]).mean()
            ),
            "gain_without_severe": float(
                np.abs(y_cal[~sev_sel] - y[~sev_sel]).mean() - np.abs(o_cal[~sev_sel] - y[~sev_sel]).mean()
            ),
            "severe_contribution_gain": float(
                group_gain[s]["kle-2"]["contribution_gain"]
            ),
            "g0_contribution_gain": float(group_gain[s]["k0"]["contribution_gain"]),
        }

    # --- bootstrap ---------------------------------------------------------
    boot = bootstrap_gains(y, k, gid_dev, preds)

    # --- gate --------------------------------------------------------------
    g0_worsen = {s: float(-group_gain[s]["k0"]["mae_gain"]) for s in SEEDS}  # positive = worsened
    gate_conditions = {
        "both_seeds_cal_gain_positive": bool(all(gains[s]["cal_gain"] > 0 for s in SEEDS)),
        "mean_cal_gain_ge_0.003": bool(mean_cal_gain >= GATE_MEAN_GAIN),
        "each_seed_g0_worsen_le_0.001": bool(all(g0_worsen[s] <= GATE_G0_WORSEN_MAX for s in SEEDS)),
    }
    gate_conditions["route_signal_met"] = bool(all(gate_conditions.values()))
    branch = select_branch(gains, mean_cal_gain, mean_group_gain, mean_group_mae_gain, g0_worsen, group_gain)
    gate = {
        "thresholds": {
            "mean_cal_gain": GATE_MEAN_GAIN,
            "g0_worsen_max": GATE_G0_WORSEN_MAX,
            "g0_mean_gain_positive": GATE_G0_MEAN_GAIN,
        },
        "per_seed_cal_gain": {str(s): gains[s]["cal_gain"] for s in SEEDS},
        "per_seed_raw_gain": {str(s): gains[s]["raw_gain"] for s in SEEDS},
        "mean_cal_gain": mean_cal_gain,
        "mean_raw_gain": mean_raw_gain,
        "per_seed_g0_worsen": {str(s): g0_worsen[s] for s in SEEDS},
        "mean_group_contribution_gain": mean_group_gain,
        "mean_group_mae_gain": mean_group_mae_gain,
        "conditions": gate_conditions,
        "branch": branch,
        "identity_checks": identity_checks,
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }

    # --- write -------------------------------------------------------------
    import csv

    def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            path.write_text("", encoding="utf-8")
            return
        fields: list[str] = []
        for rec in records:
            for key in rec:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)

    write_csv(out_dir / "main_table.csv", main_rows)
    write_csv(out_dir / "group_table.csv", group_rows)
    write_csv(out_dir / "paired_dev_predictions.csv", rows)
    write_csv(out_dir / "severe_rows.csv", severe_rows)
    _write_json(out_dir / "gate.json", gate)
    _write_json(out_dir / "bootstrap.json", boot)
    _write_json(out_dir / "sensitivity.json", sensitivity)
    summary = {
        "main_rows": main_rows,
        "gains": {str(s): gains[s] for s in SEEDS},
        "mean_cal_gain": mean_cal_gain,
        "mean_raw_gain": mean_raw_gain,
        "mean_group_contribution_gain": mean_group_gain,
        "mean_group_mae_gain": mean_group_mae_gain,
        "group_gain_per_seed": {str(s): group_gain[s] for s in SEEDS},
        "sensitivity": sensitivity,
        "bootstrap": boot,
        "gate": gate,
        "official_test_loaded": False,
    }
    _write_json(out_dir / "analysis_summary.json", summary)
    print(json.dumps(_jsonable(gate), indent=2))
    return 0


def select_branch(gains, mean_cal_gain, mean_group_gain, mean_group_mae_gain, g0_worsen, group_gain) -> dict[str, Any]:
    severe_contrib = mean_group_gain["kle-2"]
    g0_contrib = mean_group_gain["k0"]
    g0_mae_gain = {s: group_gain[s]["k0"]["mae_gain"] for s in SEEDS}
    mean_g0_mae_gain = float(np.mean(list(g0_mae_gain.values())))
    signal = bool(
        all(gains[s]["cal_gain"] > 0 for s in SEEDS)
        and mean_cal_gain >= GATE_MEAN_GAIN
        and all(g0_worsen[s] <= GATE_G0_WORSEN_MAX for s in SEEDS)
    )
    severe_share = float(severe_contrib / mean_cal_gain) if abs(mean_cal_gain) > 1e-12 else None
    if signal and mean_g0_mae_gain >= GATE_G0_MEAN_GAIN and all(v > 0 for v in g0_mae_gain.values()):
        name = "B1_bulk_also_moves"
    elif signal and abs(g0_contrib) <= GATE_G0_WORSEN_MAX and severe_contrib > 0 and (severe_share or 0) >= 0.5:
        name = "B2_g0_preserved_severe_pool"
    elif (severe_contrib > 0 and any(g0_worsen[s] > GATE_G0_WORSEN_MAX for s in SEEDS)) or any(
        gains[s]["cal_gain"] < 0 for s in SEEDS
    ):
        name = "B3_tradeoff_or_conflict"
    elif mean_cal_gain < GATE_MEAN_GAIN and any(gains[s]["cal_gain"] <= 0 for s in SEEDS):
        name = "B4_small_or_negative_gain"
    else:
        name = "NO_CLEAN_BRANCH"
    return {
        "name": name,
        "signal_met": signal,
        "mean_g0_mae_gain": mean_g0_mae_gain,
        "per_seed_g0_mae_gain": {str(s): g0_mae_gain[s] for s in SEEDS},
        "severe_contribution_gain": severe_contrib,
        "g0_contribution_gain": g0_contrib,
        "severe_contribution_share_of_mean_gain": severe_share,
        "note": "branch chosen top-down from the frozen table; near-threshold conditions are recorded as weak/ambiguous.",
    }


if __name__ == "__main__":
    raise SystemExit(main())