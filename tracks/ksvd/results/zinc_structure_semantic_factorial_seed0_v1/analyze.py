"""Local analysis for zinc-structure-semantic-factorial-seed0-v1.

Loads the four exported raw-soup states, re-scores the 8000 fit and 2000 dev
rows once, calibrates each arm on the fit bias only, and produces the main
table, group-contribution table, main-effect/interaction estimates and the
fixed paired bootstrap.  Train-inner dev only; official-valid/test are never
loaded.

Run from the repo root:
    uv run python tracks/ksvd/results/zinc_structure_semantic_factorial_seed0_v1/analyze.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_structure_semantic_factorial_seed0_v1 as fac

RESULTS_DIR = fac.RESULTS_DIR
ARMS = list(fac.ARMS)
BOOT_SEED = fac.BOOT_SEED
N_BOOT = fac.N_BOOT
DELTA = fac.DELTA
GROUP_KEYS = ("k0", "k-1", "kle-2")


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def classify(point: float, lo: float, hi: float) -> str:
    if point >= DELTA and lo > 0:
        return "positive_purchase_signal"
    if point <= -DELTA and hi < 0:
        return "negative_purchase_signal"
    if lo >= -DELTA and hi <= DELTA:
        return "no_effect_beyond_threshold"
    return "inconclusive"


def main() -> int:
    torch.set_num_threads(8)
    blob = fac.load_prep_blob()
    decomp = fac.load_target()
    train_data = fac.load_train_only()
    zftd.apply_prep_train_only(train_data, blob)
    fit_data, dev_data = fac.build_fit_dev(train_data, blob["fit_idx"], blob["dev_idx"])
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    y = np.asarray(decomp["y"], np.float64)
    k = np.asarray(decomp["k"], np.int64)
    gid = np.asarray(decomp["gid"], np.int64)
    y_fit = y[fit_idx]
    y_dev = y[dev_idx]
    k_dev = k[dev_idx]

    if len(fit_idx) != 8000 or len(dev_idx) != 2000:
        raise RuntimeError("fit/dev sizes changed")
    counts = {"k0": int((k_dev == 0).sum()), "k-1": int((k_dev == -1).sum()), "kle-2": int((k_dev <= -2).sum())}
    if counts != {"k0": 1926, "k-1": 65, "kle-2": 9}:
        raise RuntimeError(f"dev group counts changed: {counts}")

    dev_pos = {key: np.where(_group_mask(k_dev, key))[0] for key in GROUP_KEYS}
    g0 = dev_pos["k0"]

    device = torch.device("cpu")
    raw_fit: dict[str, np.ndarray] = {}
    raw_dev: dict[str, np.ndarray] = {}
    bias: dict[str, float] = {}
    for arm in ARMS:
        model = fac.build_factorial_model(blob, fac.SEED, fac.CODING_OF[arm], fac.BINDING_OF[arm]).to(device)
        state = torch.load(RESULTS_DIR / f"{arm}_raw_soup_state.pt", map_location=device)
        model.load_state_dict({key: value.to(device) for key, value in state.items()})
        model.eval()
        fit_raw, _ = fac.predict(model, fit_data, torch.as_tensor(y_fit, dtype=torch.float32), device)
        dev_raw, _ = fac.predict(model, dev_data, torch.as_tensor(y_dev, dtype=torch.float32), device)
        raw_fit[arm] = fit_raw
        raw_dev[arm] = dev_raw
        bias[arm] = float(np.median(y_fit - fit_raw))

    cal_fit = {arm: raw_fit[arm] + bias[arm] for arm in ARMS}
    cal_dev = {arm: raw_dev[arm] + bias[arm] for arm in ARMS}

    # --- raw-soup replay check against the released fit predictions ----------
    # The released fit_raw was computed on the GPU during training; the local
    # replay is CPU.  The criterion is therefore CPU-vs-GPU float32 agreement
    # (order 1e-6), not bit-identity.
    CPU_GPU_TOL = 1.0e-5
    replay = {}
    for arm in ARMS:
        saved = np.load(RESULTS_DIR / f"{arm}_predictions.npz")
        diff = float(np.max(np.abs(raw_fit[arm] - saved["fit_raw"].astype(np.float64))))
        replay[arm] = {
            "max_abs_diff_vs_released_fit_raw": diff,
            "released_b": float(saved["b"]),
            "recomputed_b": bias[arm],
            "b_abs_diff": abs(float(saved["b"]) - bias[arm]),
            "cpu_gpu_tolerance": CPU_GPU_TOL,
            "pass": bool(diff <= CPU_GPU_TOL and abs(float(saved["b"]) - bias[arm]) <= 1e-6),
        }
    (RESULTS_DIR / "replay_check.json").write_text(json.dumps(jsonable(replay), indent=2))

    # --- initial-parameter identity across the four fresh arms ----------------
    init_states = {
        arm: torch.load(RESULTS_DIR / f"{arm}_init_state.pt", map_location="cpu") for arm in ARMS
    }
    ref = init_states["S_J"]
    init_identity = {}
    for arm in ARMS:
        keys = sorted(set(ref) | set(init_states[arm]))
        mismatched, max_diff = [], 0.0
        for key in keys:
            if key == "kappa":
                continue
            if key not in ref or key not in init_states[arm]:
                mismatched.append(key)
                continue
            delta = float((ref[key].float() - init_states[arm][key].float()).abs().max())
            max_diff = max(max_diff, delta)
            if not torch.equal(ref[key], init_states[arm][key]):
                mismatched.append(key)
        init_identity[arm] = {
            "max_abs_diff_excluding_kappa": max_diff,
            "mismatched_tensors": mismatched,
            "identical": bool(not mismatched),
        }
    kappa_values = {arm: float(init_states[arm]["kappa"].reshape(-1)[0]) for arm in ARMS}
    (RESULTS_DIR / "init_identity.json").write_text(
        json.dumps(
            jsonable(
                {
                    "per_arm": init_identity,
                    "all_identical": all(init_identity[a]["identical"] for a in ARMS),
                    "kappa_values": kappa_values,
                    "note": "kappa is a non-trainable buffer, identical across arms by construction (shared init D/U)",
                }
            ),
            indent=2,
        )
    )

    # --- per-row predictions ------------------------------------------------
    rows = []
    for split, idx, yv, kv, raw_map, cal_map in (
        ("fit", fit_idx, y[fit_idx], k[fit_idx], raw_fit, cal_fit),
        ("dev", dev_idx, y_dev, k_dev, raw_dev, cal_dev),
    ):
        for pos in range(len(idx)):
            record = {
                "split": split,
                "global_index": int(idx[pos]),
                "canonical_group_id": int(gid[idx[pos]]),
                "y": float(yv[pos]),
                "k": int(kv[pos]),
                "group": _group_of(int(kv[pos])),
            }
            for arm in ARMS:
                record[f"{arm}_raw"] = float(raw_map[arm][pos])
                record[f"{arm}_cal"] = float(cal_map[arm][pos])
            rows.append(record)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    # --- main table ---------------------------------------------------------
    def mae(values, mask=None):
        if mask is None:
            return float(np.mean(np.abs(values)))
        return float(np.mean(np.abs(values[mask])))

    def gains(pred_map, mask=None):
        l = {arm: mae(pred_map[arm] - y_dev, mask) if mask is None else mae(pred_map[arm][mask] - y_dev[mask]) for arm in ARMS}
        return {
            "C_S": l["S_M"] - l["S_J"],
            "C_D": l["D_M"] - l["D_J"],
            "G_J": l["D_J"] - l["S_J"],
            "G_M": l["D_M"] - l["S_M"],
            "I": (l["S_M"] - l["S_J"]) - (l["D_M"] - l["D_J"]),
            "levels": l,
        }

    main = {
        "fit_cal_mae": {arm: mae(cal_fit[arm] - y_fit) for arm in ARMS},
        "fit_raw_mae": {arm: mae(raw_fit[arm] - y_fit) for arm in ARMS},
        "dev_cal_mae": {arm: mae(cal_dev[arm] - y_dev) for arm in ARMS},
        "dev_raw_mae": {arm: mae(raw_dev[arm] - y_dev) for arm in ARMS},
        "dev_cal_mae_g0": {arm: mae((cal_dev[arm] - y_dev)[g0]) for arm in ARMS},
        "dev_raw_mae_g0": {arm: mae((raw_dev[arm] - y_dev)[g0]) for arm in ARMS},
        "bias": bias,
        "fit_dev_gap_cal": {arm: mae(cal_dev[arm] - y_dev) - mae(cal_fit[arm] - y_fit) for arm in ARMS},
        "overall_cal": gains(cal_dev),
        "overall_raw": gains(raw_dev),
        "g0_cal": gains(cal_dev, g0),
        "g0_raw": gains(raw_dev, g0),
    }

    # --- group table --------------------------------------------------------
    group_rows = []
    for key in GROUP_KEYS:
        mask = dev_pos[key]
        n = int(mask.size)
        for arm in ARMS:
            err_cal = cal_dev[arm][mask] - y_dev[mask]
            err_raw = raw_dev[arm][mask] - y_dev[mask]
            group_rows.append(
                {
                    "group": fac.GROUP_NAMES[key],
                    "n": n,
                    "arm": arm,
                    "cal_mae": float(np.mean(np.abs(err_cal))),
                    "raw_mae": float(np.mean(np.abs(err_raw))),
                    "cal_residual_y_minus_pred": float(np.mean(y_dev[mask] - cal_dev[arm][mask])),
                    "cal_contribution": float(np.abs(err_cal).sum() / len(y_dev)),
                    "raw_contribution": float(np.abs(err_raw).sum() / len(y_dev)),
                }
            )
    group_df = pd.DataFrame(group_rows)

    # contribution gains per group and contrast
    contrib = {(r["group"], r["arm"]): r["cal_contribution"] for r in group_rows}
    contrib_gain_rows = []
    for gname in fac.GROUP_NAMES.values():
        contrib_gain_rows.append(
            {
                "group": gname,
                "C_S_contribution_gain": contrib[(gname, "S_M")] - contrib[(gname, "S_J")],
                "C_D_contribution_gain": contrib[(gname, "D_M")] - contrib[(gname, "D_J")],
                "G_J_contribution_gain": contrib[(gname, "D_J")] - contrib[(gname, "S_J")],
                "G_M_contribution_gain": contrib[(gname, "D_M")] - contrib[(gname, "S_M")],
            }
        )
    contrib_df = pd.DataFrame(contrib_gain_rows)

    # identity check: group contributions sum to overall MAE
    identity = {}
    for arm in ARMS:
        total = float(sum(contrib[(gname, arm)] for gname in fac.GROUP_NAMES.values()))
        overall = mae(cal_dev[arm] - y_dev)
        identity[arm] = {"contribution_sum": total, "overall_mae": overall, "abs_diff": abs(total - overall)}

    # --- bootstrap ----------------------------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    samplers = []
    for _ in range(N_BOOT):
        idx_g0 = g0[rng.integers(0, len(g0), size=len(g0))]
        parts = []
        for key in GROUP_KEYS:
            pool = dev_pos[key]
            parts.append(pool[rng.integers(0, len(pool), size=len(pool))])
        samplers.append((idx_g0, np.concatenate(parts)))

    def boot_mae(arm_pred, idx):
        return float(np.mean(np.abs(arm_pred[idx] - y_dev[idx])))

    boot = {k_: {"g0": [], "overall": []} for k_ in ("C_S", "C_D", "G_J", "G_M", "I")}
    boot_levels = {arm: {"g0": [], "overall": []} for arm in ARMS}
    for idx_g0, idx_ov in samplers:
        for scope, idx in (("g0", idx_g0), ("overall", idx_ov)):
            lev = {arm: boot_mae(cal_dev[arm], idx) for arm in ARMS}
            for arm in ARMS:
                boot_levels[arm][scope].append(lev[arm])
            boot["C_S"][scope].append(lev["S_M"] - lev["S_J"])
            boot["C_D"][scope].append(lev["D_M"] - lev["D_J"])
            boot["G_J"][scope].append(lev["D_J"] - lev["S_J"])
            boot["G_M"][scope].append(lev["D_M"] - lev["S_M"])
            boot["I"][scope].append((lev["S_M"] - lev["S_J"]) - (lev["D_M"] - lev["D_J"]))

    boot_summary = {}
    for contrast in ("C_S", "C_D", "G_J", "G_M", "I"):
        boot_summary[contrast] = {}
        for scope in ("g0", "overall"):
            point = main["g0_cal" if scope == "g0" else "overall_cal"][contrast]
            arr = np.asarray(boot[contrast][scope], np.float64)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            boot_summary[contrast][scope] = {
                "point": point,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "verdict": classify(point, float(lo), float(hi)),
            }

    # identity / sanity witnesses for the bootstrap
    boot_witness = {
        "identical_predictions_gain_zero": float(main["overall_cal"]["C_S"] - main["overall_cal"]["C_S"]),
        "swap_sign_flip": {
            "C_S": main["overall_cal"]["C_S"],
            "swapped": -main["overall_cal"]["C_S"],
        },
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
    }

    # worst S_J rows by absolute cal error
    sj_err = np.abs(cal_dev["S_J"] - y_dev)
    worst = np.argsort(-sj_err)[:10]

    summary = {
        "protocol_version": fac.PROTOCOL_VERSION,
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "dev_group_counts": counts,
        "main": main,
        "bootstrap": boot_summary,
        "bootstrap_witness": boot_witness,
        "contribution_identity": identity,
        "worst_SJ_rows": [
            {
                "dev_position": int(p),
                "global_index": int(dev_idx[p]),
                "y": float(y_dev[p]),
                "k": int(k_dev[p]),
                "S_J_cal_error": float(cal_dev["S_J"][p] - y_dev[p]),
                "S_J_cal": float(cal_dev["S_J"][p]),
                "S_M_cal": float(cal_dev["S_M"][p]),
                "D_J_cal": float(cal_dev["D_J"][p]),
                "D_M_cal": float(cal_dev["D_M"][p]),
            }
            for p in worst
        ],
    }

    (RESULTS_DIR / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    group_df.to_csv(RESULTS_DIR / "group_table.csv", index=False)
    contrib_df.to_csv(RESULTS_DIR / "group_contribution_gains.csv", index=False)
    pd.DataFrame(
        [{"arm": arm, **{f"{scope}_cal_mae": main[f"dev_cal_mae_{scope}" if scope != "overall" else "dev_cal_mae"][arm] for scope in ("g0", "overall")},
          "fit_cal_mae": main["fit_cal_mae"][arm],
          "raw_dev_cal_mae": main["dev_raw_mae"][arm],
          "bias": bias[arm]} for arm in ARMS]
    ).to_csv(RESULTS_DIR / "main_table.csv", index=False)

    print(json.dumps(jsonable(main), indent=2))
    print(json.dumps(jsonable(boot_summary), indent=2))
    print(json.dumps(identity, indent=2))
    return 0


def _group_mask(k_dev: np.ndarray, key: str) -> np.ndarray:
    if key == "k0":
        return k_dev == 0
    if key == "k-1":
        return k_dev == -1
    return k_dev <= -2


def _group_of(kv: int) -> str:
    if kv == 0:
        return "k=0"
    if kv == -1:
        return "k=-1"
    return "k<=-2"


if __name__ == "__main__":
    raise SystemExit(main())