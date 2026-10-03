"""Fixed analysis for zinc-frozen-chemistry-learned-cycle-v1 (local, CPU).

Reads the frozen Y/O artifacts, the new frozen-O + learned-cycle-head outputs and
the frozen split, and writes the pre-registered tables and gate.  No fitting, no
probe, no new model.

Official ZINC test is never instantiated / loaded / evaluated; the official
validation split is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent
FROZEN_DIR = RESULTS_DIR.parent / "zinc_full_cycle_target_decomposition_v1"
PREP_BLOB = RESULTS_DIR.parent / "zinc_joint_dictionary_decision_v1" / "prep" / "fold_objects.npz"

SEEDS = (0, 1)
ARMS = ("Y", "P", "O", "K")
SEVERE_MAX = -2
BOOT_SEED = 20261003
N_BOOT = 1000
GATE_MEAN_GAIN = 0.003
GATE_G0_WORSEN_MAX = 0.001
GATE_G0_MEAN_GAIN = 0.002
REPLAY_TOL = 2.0e-6
FIT_MAE_DROP_MIN = 0.50
FIT_SEVERE_MAE_MAX = 1.0


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


def _sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def group_name(k: int) -> str:
    return "k0" if k == 0 else ("k-1" if k == -1 else "kle-2")


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def load_frozen(seed: int, arm: str) -> dict[str, Any]:
    meta = json.loads((FROZEN_DIR / f"{arm}_seed{seed}.json").read_text())
    preds = dict(np.load(FROZEN_DIR / f"{arm}_seed{seed}_predictions.npz", allow_pickle=False))
    return {"meta": meta, "preds": preds}


def build_arms() -> dict[str, dict[str, Any]]:
    """Return per-seed raw/cal fit/dev predictions of Y/P/O/K against y."""
    out: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        y_pred = load_frozen(seed, "Y")
        o_pred = load_frozen(seed, "O")
        b_Y = float(y_pred["meta"]["calibration"]["b"])
        b_O = float(o_pred["meta"]["calibration"]["b"])
        y_fit_raw = y_pred["preds"]["fit_raw"].astype(np.float64)
        y_dev_raw = y_pred["preds"]["dev_raw"].astype(np.float64)
        h_fit = o_pred["preds"]["fit_raw"].astype(np.float64)
        h_dev = o_pred["preds"]["dev_raw"].astype(np.float64)

        p = dict(np.load(RESULTS_DIR / f"P_seed{seed}_predictions.npz", allow_pickle=False))
        q_fit = p["fit_q"].astype(np.float64)
        q_dev = p["dev_q"].astype(np.float64)
        b_P = float(p["b_P"][0])
        b_K = float(p["b_K"][0])
        med_c = float(p["median_fit_c"][0])

        t = dict(np.load(RESULTS_DIR / "cycle_targets.npz", allow_pickle=False))
        fit_idx = t["fit_idx"].astype(np.int64)
        dev_idx = t["dev_idx"].astype(np.int64)
        c_fit, c_dev = t["fit_c"].astype(np.float64), t["dev_c"].astype(np.float64)
        y_fit, y_dev = t["fit_y"].astype(np.float64), t["dev_y"].astype(np.float64)
        g_fit, g_dev = t["fit_g"].astype(np.float64), t["dev_g"].astype(np.float64)
        k_dev = t["dev_k"].astype(np.int64)
        # all-train k for fit grouping
        with np.load(FROZEN_DIR / "target_decomposition.npz", allow_pickle=False) as z:
            k_all = z["k"].astype(np.int64)
            gid_all = z["gid"].astype(np.int64)
        k_fit = k_all[fit_idx]
        gid_dev = gid_all[dev_idx]

        arms = {
            "Y": {
                "fit_raw": y_fit_raw,
                "fit_cal": y_fit_raw + b_Y,
                "dev_raw": y_dev_raw,
                "dev_cal": y_dev_raw + b_Y,
                "b": b_Y,
            },
            "P": {
                "fit_raw": h_fit + q_fit,
                "fit_cal": h_fit + q_fit + b_P,
                "dev_raw": h_dev + q_dev,
                "dev_cal": h_dev + q_dev + b_P,
                "b": b_P,
            },
            "O": {
                "fit_raw": h_fit + c_fit,
                "fit_cal": h_fit + c_fit + b_O,
                "dev_raw": h_dev + c_dev,
                "dev_cal": h_dev + c_dev + b_O,
                "b": b_O,
            },
            "K": {
                "fit_raw": h_fit + med_c,
                "fit_cal": h_fit + med_c + b_K,
                "dev_raw": h_dev + med_c,
                "dev_cal": h_dev + med_c + b_K,
                "b": b_K,
            },
        }
        out[str(seed)] = {
            "seed": seed,
            "arms": arms,
            "q_fit": q_fit,
            "q_dev": q_dev,
            "h_fit": h_fit,
            "h_dev": h_dev,
            "c_fit": c_fit,
            "c_dev": c_dev,
            "y_fit": y_fit,
            "y_dev": y_dev,
            "g_fit": g_fit,
            "g_dev": g_dev,
            "k_fit": k_fit,
            "k_dev": k_dev,
            "gid_dev": gid_dev,
            "fit_idx_global": fit_idx,
            "dev_idx_global": dev_idx,
            "median_fit_c": med_c,
            "b_P": b_P,
            "b_K": b_K,
            "b_Y": b_Y,
            "b_O": b_O,
        }
    return out


def group_stats(pred: np.ndarray, y: np.ndarray, k: np.ndarray, n_total: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, sel in (
        ("k0", k == 0),
        ("k-1", k == -1),
        ("kle-2", k <= SEVERE_MAX),
    ):
        sel = np.asarray(sel)
        if not sel.any():
            out[name] = {
                "n": 0, "mae": None, "signed_residual": None, "contribution": 0.0,
                "positive_error_share": None,
            }
            continue
        err = pred[sel] - y[sel]
        out[name] = {
            "n": int(sel.sum()),
            "mae": float(np.mean(np.abs(err))),
            "signed_residual": float(np.mean(y[sel] - pred[sel])),
            "contribution": float(np.sum(np.abs(err)) / n_total),
            "positive_error_share": float(np.mean(err > 0)),
        }
    out["overall"] = {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(pred - y))),
        "signed_residual": float(np.mean(y - pred)),
        "contribution": float(np.sum(np.abs(pred - y)) / n_total),
        "positive_error_share": float(np.mean((pred - y) > 0)),
    }
    return out


def component_stats(q: np.ndarray, c: np.ndarray, med_c: float, k: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, sel in (
        ("k0", k == 0),
        ("k-1", k == -1),
        ("kle-2", k <= SEVERE_MAX),
    ):
        sel = np.asarray(sel)
        if not sel.any():
            out[name] = {"n": 0}
            continue
        qs, cs = q[sel], c[sel]
        out[name] = {
            "n": int(sel.sum()),
            "q_vs_c_mae": float(np.mean(np.abs(qs - cs))),
            "q_vs_c_signed_residual": float(np.mean(cs - qs)),
            "constant_vs_c_mae": float(np.mean(np.abs(med_c - cs))),
            "q_std": float(np.std(qs)),
            "q_min": float(np.min(qs)),
            "q_max": float(np.max(qs)),
            "c_std": float(np.std(cs)),
            "c_min": float(np.min(cs)),
            "c_max": float(np.max(cs)),
        }
    out["overall"] = {
        "n": int(len(c)),
        "q_vs_c_mae": float(np.mean(np.abs(q - c))),
        "q_vs_c_signed_residual": float(np.mean(c - q)),
        "constant_vs_c_mae": float(np.mean(np.abs(med_c - c))),
        "q_std": float(np.std(q)),
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
        "c_std": float(np.std(c)),
        "c_min": float(np.min(c)),
        "c_max": float(np.max(c)),
    }
    return out


def bootstrap(datasets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ref = datasets["0"]
    groups = np.unique(ref["gid_dev"])
    group_rows = {int(g): np.where(ref["gid_dev"] == g)[0] for g in groups}
    rng = np.random.default_rng(int(BOOT_SEED))

    def draw(array: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return array[idx]

    records: dict[str, list[float]] = {
        "mean_gain_Y_to_P": [],
        "mean_gain_K_to_P": [],
        "mean_g0_gain_Y_to_P": [],
        "mean_g0_marker_P_minus_O": [],
    }
    per_seed: dict[str, dict[str, list[float]]] = {
        s: {"gain_Y_to_P": [], "gain_K_to_P": [], "g0_gain_Y_to_P": [], "g0_marker_P_minus_O": []}
        for s in ("0", "1")
    }
    for _ in range(N_BOOT):
        chosen = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([group_rows[int(groups[c])] for c in chosen])
        gains_y, gains_k, g0s, markers = [], [], [], []
        for s in ("0", "1"):
            d = datasets[s]
            y = draw(d["y_dev"], idx)
            k = draw(d["k_dev"], idx)
            y_cal = draw(d["arms"]["Y"]["dev_cal"], idx)
            p_cal = draw(d["arms"]["P"]["dev_cal"], idx)
            o_cal = draw(d["arms"]["O"]["dev_cal"], idx)
            kk_cal = draw(d["arms"]["K"]["dev_cal"], idx)
            gain_y = mae(y_cal, y) - mae(p_cal, y)
            gain_k = mae(kk_cal, y) - mae(p_cal, y)
            sel0 = k == 0
            g0_gain = mae(y_cal[sel0], y[sel0]) - mae(p_cal[sel0], y[sel0])
            g0_marker = mae(p_cal[sel0], y[sel0]) - mae(o_cal[sel0], y[sel0])
            per_seed[s]["gain_Y_to_P"].append(gain_y)
            per_seed[s]["gain_K_to_P"].append(gain_k)
            per_seed[s]["g0_gain_Y_to_P"].append(g0_gain)
            per_seed[s]["g0_marker_P_minus_O"].append(g0_marker)
            gains_y.append(gain_y)
            gains_k.append(gain_k)
            g0s.append(g0_gain)
            markers.append(g0_marker)
        records["mean_gain_Y_to_P"].append(float(np.mean(gains_y)))
        records["mean_gain_K_to_P"].append(float(np.mean(gains_k)))
        records["mean_g0_gain_Y_to_P"].append(float(np.mean(g0s)))
        records["mean_g0_marker_P_minus_O"].append(float(np.mean(markers)))

    def summarize(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, np.float64)
        return {
            "mean": float(arr.mean()),
            "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)),
        }

    out = {
        "n_boot": N_BOOT,
        "seed": BOOT_SEED,
        "unit": "canonical_group_id, joint across arms and seeds",
        "mean_gain_Y_to_P": summarize(records["mean_gain_Y_to_P"]),
        "mean_gain_K_to_P": summarize(records["mean_gain_K_to_P"]),
        "mean_g0_gain_Y_to_P": summarize(records["mean_g0_gain_Y_to_P"]),
        "mean_g0_marker_P_minus_O": summarize(records["mean_g0_marker_P_minus_O"]),
        "per_seed": {
            s: {key: summarize(values) for key, values in per_seed[s].items()} for s in ("0", "1")
        },
        "note": "conditional description of this reused diagnostic dev set and these two paired seeds only; not a cross-seed population CI.",
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)

    datasets = build_arms()
    with np.load(FROZEN_DIR / "target_decomposition.npz", allow_pickle=False) as z:
        gid_all = z["gid"].astype(np.int64)

    # --- main table --------------------------------------------------------
    main_rows = []
    for s in ("0", "1"):
        d = datasets[s]
        for arm in ARMS:
            a = d["arms"][arm]
            main_rows.append({
                "seed": d["seed"],
                "arm": arm,
                "b": a["b"],
                "fit_raw_mae": mae(a["fit_raw"], d["y_fit"]),
                "fit_cal_mae": mae(a["fit_cal"], d["y_fit"]),
                "dev_raw_mae": mae(a["dev_raw"], d["y_dev"]),
                "dev_cal_mae": mae(a["dev_cal"], d["y_dev"]),
                "fit_dev_gap": mae(a["fit_cal"], d["y_fit"]) - mae(a["dev_cal"], d["y_dev"]),
                "fit_bias": float(np.mean(d["y_fit"] - a["fit_cal"])),
                "dev_bias": float(np.mean(d["y_dev"] - a["dev_cal"])),
            })

    # --- gains -------------------------------------------------------------
    gains = {}
    for s in ("0", "1"):
        d = datasets[s]
        y = d["y_dev"]
        my, mp, mo, mk = (mae(d["arms"][a]["dev_cal"], y) for a in ("Y", "P", "O", "K"))
        mr_y, mr_p, mr_o, mr_k = (mae(d["arms"][a]["dev_raw"], y) for a in ("Y", "P", "O", "K"))
        gains[s] = {
            "gain_Y_to_P_cal": my - mp,
            "gain_Y_to_P_raw": mr_y - mr_p,
            "gain_K_to_P_cal": mk - mp,
            "gain_K_to_P_raw": mr_k - mr_p,
            "oracle_loss_cal": mp - mo,
            "oracle_gain_Y_to_O_cal": my - mo,
            "oracle_retention_ratio": (my - mp) / (my - mo),
        }
    mean_gain_y_to_p_cal = float(np.mean([gains[s]["gain_Y_to_P_cal"] for s in ("0", "1")]))
    mean_gain_y_to_p_raw = float(np.mean([gains[s]["gain_Y_to_P_raw"] for s in ("0", "1")]))
    mean_gain_k_to_p_cal = float(np.mean([gains[s]["gain_K_to_P_cal"] for s in ("0", "1")]))
    mean_gain_k_to_p_raw = float(np.mean([gains[s]["gain_K_to_P_raw"] for s in ("0", "1")]))
    mean_retention = float(np.mean([gains[s]["oracle_retention_ratio"] for s in ("0", "1")]))
    raw_cal_same_direction = bool(
        all(
            np.sign(gains[s]["gain_Y_to_P_cal"]) == np.sign(gains[s]["gain_Y_to_P_raw"])
            and np.sign(gains[s]["gain_K_to_P_cal"]) == np.sign(gains[s]["gain_K_to_P_raw"])
            for s in ("0", "1")
        )
    )

    # --- group table -------------------------------------------------------
    group_rows = []
    group_gain = {s: {} for s in ("0", "1")}
    for s in ("0", "1"):
        d = datasets[s]
        y, k = d["y_dev"], d["k_dev"]
        n = len(y)
        stats = {arm: group_stats(d["arms"][arm]["dev_cal"], y, k, n) for arm in ARMS}
        for gname in ("k0", "k-1", "kle-2", "overall"):
            row = {"seed": d["seed"], "group": gname, "n": stats["Y"][gname]["n"]}
            for arm in ARMS:
                row[f"{arm}_mae"] = stats[arm][gname]["mae"]
                row[f"{arm}_signed_residual"] = stats[arm][gname]["signed_residual"]
                row[f"{arm}_contribution"] = stats[arm][gname]["contribution"]
            group_rows.append(row)
        for gname in ("k0", "k-1", "kle-2", "overall"):
            group_gain[s][gname] = {
                "gain_Y_to_P_contribution": stats["Y"][gname]["contribution"] - stats["P"][gname]["contribution"],
                "gain_Y_to_P_mae": stats["Y"][gname]["mae"] - stats["P"][gname]["mae"],
                "gain_K_to_P_contribution": stats["K"][gname]["contribution"] - stats["P"][gname]["contribution"],
                "oracle_loss_contribution": stats["P"][gname]["contribution"] - stats["O"][gname]["contribution"],
            }

    # --- component (cycle head) table -------------------------------------
    component = {}
    for s in ("0", "1"):
        d = datasets[s]
        component[s] = {
            "fit": component_stats(d["q_fit"], d["c_fit"], d["median_fit_c"], d["k_fit"]),
            "dev": component_stats(d["q_dev"], d["c_dev"], d["median_fit_c"], d["k_dev"]),
        }

    head_fit_diagnostic = {}
    for s in ("0", "1"):
        comp = component[s]
        drop = (comp["fit"]["overall"]["constant_vs_c_mae"] - comp["fit"]["overall"]["q_vs_c_mae"]) / comp["fit"]["overall"]["constant_vs_c_mae"]
        head_fit_diagnostic[s] = {
            "fit_overall_q_mae": comp["fit"]["overall"]["q_vs_c_mae"],
            "fit_overall_constant_mae": comp["fit"]["overall"]["constant_vs_c_mae"],
            "fit_overall_mae_drop_fraction": drop,
            "fit_overall_drop_ge_50pct": bool(drop >= FIT_MAE_DROP_MIN),
            "fit_severe_q_mae": comp["fit"]["kle-2"]["q_vs_c_mae"],
            "fit_severe_constant_mae": comp["fit"]["kle-2"]["constant_vs_c_mae"],
            "fit_severe_mae_le_1.0": bool(comp["fit"]["kle-2"]["q_vs_c_mae"] <= FIT_SEVERE_MAE_MAX),
            "fit_severe_n": comp["fit"]["kle-2"]["n"],
        }
    head_fit_satisfied = bool(
        all(v["fit_overall_drop_ge_50pct"] and v["fit_severe_mae_le_1.0"] for v in head_fit_diagnostic.values())
    )

    # --- per-row paired table ---------------------------------------------
    paired = []
    for s in ("0", "1"):
        d = datasets[s]
        n_dev = len(d["y_dev"])
        e_g = d["g_dev"] - d["h_dev"]
        e_c = d["c_dev"] - d["q_dev"]
        e_p = e_g + e_c - d["b_P"]
        identity_residual = np.abs(e_p - (d["y_dev"] - d["arms"]["P"]["dev_cal"]))
        for i in range(n_dev):
            paired.append({
                "seed": d["seed"],
                "dev_pos": i,
                "global_index": int(_dev_global_index(d, i)),
                "k": int(d["k_dev"][i]),
                "y": float(d["y_dev"][i]),
                "c": float(d["c_dev"][i]),
                "g": float(d["g_dev"][i]),
                "h_raw": float(d["h_dev"][i]),
                "q": float(d["q_dev"][i]),
                "e_g": float(e_g[i]),
                "e_c": float(e_c[i]),
                "e_P": float(e_p[i]),
                "Y_cal": float(d["arms"]["Y"]["dev_cal"][i]),
                "P_raw": float(d["arms"]["P"]["dev_raw"][i]),
                "P_cal": float(d["arms"]["P"]["dev_cal"][i]),
                "O_cal": float(d["arms"]["O"]["dev_cal"][i]),
                "K_cal": float(d["arms"]["K"]["dev_cal"][i]),
                "err_Y": float(d["arms"]["Y"]["dev_cal"][i] - d["y_dev"][i]),
                "err_P": float(d["arms"]["P"]["dev_cal"][i] - d["y_dev"][i]),
                "err_O": float(d["arms"]["O"]["dev_cal"][i] - d["y_dev"][i]),
                "err_K": float(d["arms"]["K"]["dev_cal"][i] - d["y_dev"][i]),
                "identity_abs_residual": float(identity_residual[i]),
            })
        if float(identity_residual.max()) > 1e-9:
            raise RuntimeError(f"e_P identity violated for seed {s}: {float(identity_residual.max())}")

    # --- severe 9 rows -----------------------------------------------------
    severe = []
    for s in ("0", "1"):
        d = datasets[s]
        sel = np.where(d["k_dev"] <= SEVERE_MAX)[0]
        for i in sel:
            severe.append({
                "seed": d["seed"],
                "dev_pos": int(i),
                "global_index": int(_dev_global_index(d, int(i))),
                "k": int(d["k_dev"][i]),
                "y": float(d["y_dev"][i]),
                "c": float(d["c_dev"][i]),
                "g": float(d["g_dev"][i]),
                "h_raw": float(d["h_dev"][i]),
                "q": float(d["q_dev"][i]),
                "c_minus_q": float(d["c_dev"][i] - d["q_dev"][i]),
                "P_cal": float(d["arms"]["P"]["dev_cal"][i]),
                "O_cal": float(d["arms"]["O"]["dev_cal"][i]),
                "Y_cal": float(d["arms"]["Y"]["dev_cal"][i]),
                "err_P": float(d["arms"]["P"]["dev_cal"][i] - d["y_dev"][i]),
                "err_O": float(d["arms"]["O"]["dev_cal"][i] - d["y_dev"][i]),
                "err_Y": float(d["arms"]["Y"]["dev_cal"][i] - d["y_dev"][i]),
            })

    # --- sensitivity -------------------------------------------------------
    sensitivity = {}
    for s in ("0", "1"):
        d = datasets[s]
        y = d["y_dev"]
        err_y = np.abs(d["arms"]["Y"]["dev_cal"] - y)
        worst = int(np.argmax(err_y))
        keep = np.ones(len(y), dtype=bool)
        keep[worst] = False
        gain_drop_worst = mae(d["arms"]["Y"]["dev_cal"][keep], y[keep]) - mae(d["arms"]["P"]["dev_cal"][keep], y[keep])
        nos = d["k_dev"] > SEVERE_MAX
        gain_no_severe = mae(d["arms"]["Y"]["dev_cal"][nos], y[nos]) - mae(d["arms"]["P"]["dev_cal"][nos], y[nos])
        sensitivity[s] = {
            "control_max_error": {
                "dev_pos": worst,
                "global_index": int(_dev_global_index(d, worst)),
                "k": int(d["k_dev"][worst]),
                "abs_err": float(err_y[worst]),
            },
            "gain_dropping_control_max_error_row": float(gain_drop_worst),
            "gain_excluding_severe": float(gain_no_severe),
        }

    # --- wrapper / replay checks ------------------------------------------
    wrapper = json.loads((RESULTS_DIR / "wrapper_checks.json").read_text())
    replay = replay_head_states()
    identity = json.loads((RESULTS_DIR / "identity_checks.json").read_text())
    smoke = json.loads((RESULTS_DIR / "smoke_checks.json").read_text())

    # --- gate --------------------------------------------------------------
    per_seed_g0_worsen = {
        s: (
            group_stats(datasets[s]["arms"]["P"]["dev_cal"], datasets[s]["y_dev"], datasets[s]["k_dev"], len(datasets[s]["y_dev"]))["k0"]["mae"]
            - group_stats(datasets[s]["arms"]["Y"]["dev_cal"], datasets[s]["y_dev"], datasets[s]["k_dev"], len(datasets[s]["y_dev"]))["k0"]["mae"]
        )
        for s in ("0", "1")
    }
    per_seed_g0_marker = {
        s: (
            group_stats(datasets[s]["arms"]["P"]["dev_cal"], datasets[s]["y_dev"], datasets[s]["k_dev"], len(datasets[s]["y_dev"]))["k0"]["mae"]
            - group_stats(datasets[s]["arms"]["O"]["dev_cal"], datasets[s]["y_dev"], datasets[s]["k_dev"], len(datasets[s]["y_dev"]))["k0"]["mae"]
        )
        for s in ("0", "1")
    }
    conditions = {
        "both_seeds_gain_Y_to_P_cal_positive": bool(all(gains[s]["gain_Y_to_P_cal"] > 0 for s in ("0", "1"))),
        "mean_gain_Y_to_P_cal_ge_0.003": bool(mean_gain_y_to_p_cal >= GATE_MEAN_GAIN),
        "both_seeds_gain_K_to_P_cal_positive": bool(all(gains[s]["gain_K_to_P_cal"] > 0 for s in ("0", "1"))),
        "mean_gain_K_to_P_cal_ge_0.003": bool(mean_gain_k_to_p_cal >= GATE_MEAN_GAIN),
        "each_seed_P_G0_worsen_vs_Y_le_0.001": bool(all(per_seed_g0_worsen[s] <= GATE_G0_WORSEN_MAX for s in ("0", "1"))),
        "identity_ok": bool(identity["all_ok"]),
        "no_true_c_inference": bool(
            all(wrapper[s]["label_permutation_invariant"] and (not wrapper[s]["forward_accepts_labels"]) for s in ("0", "1"))
        ),
        "single_calibration": True,
        "replay_ok": bool(all(replay[s]["max_abs"] <= REPLAY_TOL for s in ("0", "1"))),
    }
    gate_met = bool(
        conditions["both_seeds_gain_Y_to_P_cal_positive"]
        and conditions["mean_gain_Y_to_P_cal_ge_0.003"]
        and conditions["both_seeds_gain_K_to_P_cal_positive"]
        and conditions["mean_gain_K_to_P_cal_ge_0.003"]
        and conditions["each_seed_P_G0_worsen_vs_Y_le_0.001"]
        and conditions["identity_ok"]
        and conditions["no_true_c_inference"]
        and conditions["replay_ok"]
    )
    secondary_marker = bool(all(per_seed_g0_marker[s] <= GATE_G0_WORSEN_MAX for s in ("0", "1")))

    # choose branch
    if not (identity["all_ok"] and conditions["replay_ok"]):
        branch = "INVALID_INCOMPLETE"
    elif gate_met and secondary_marker and all(component[s]["dev"]["kle-2"]["q_vs_c_mae"] < component[s]["dev"]["kle-2"]["constant_vs_c_mae"] for s in ("0", "1")):
        branch = "gate_met_severe_better_than_constant"
    elif not head_fit_satisfied:
        branch = "fit_readout_insufficient"
    elif all(component[s]["dev"]["kle-2"]["q_vs_c_mae"] >= component[s]["dev"]["kle-2"]["constant_vs_c_mae"] for s in ("0", "1")):
        branch = "fit_ok_dev_severe_transfer_failure"
    elif not gate_met:
        branch = "component_learned_but_gate_failed"
    else:
        branch = "gate_met_secondary_marker_failed"

    analysis = {
        "protocol_version": "zinc-frozen-chemistry-learned-cycle-v1",
        "main_rows": main_rows,
        "gains": gains,
        "mean_gain_Y_to_P_cal": mean_gain_y_to_p_cal,
        "mean_gain_Y_to_P_raw": mean_gain_y_to_p_raw,
        "mean_gain_K_to_P_cal": mean_gain_k_to_p_cal,
        "mean_gain_K_to_P_raw": mean_gain_k_to_p_raw,
        "mean_oracle_retention_ratio": mean_retention,
        "raw_cal_same_direction": raw_cal_same_direction,
        "group_rows": group_rows,
        "group_gain": group_gain,
        "component": component,
        "head_fit_diagnostic": head_fit_diagnostic,
        "head_fit_satisfied": head_fit_satisfied,
        "sensitivity": sensitivity,
        "wrapper_checks": wrapper,
        "replay_checks": replay,
        "identity_ok": identity["all_ok"],
        "smoke_ok": smoke["ok"],
        "gate": {
            "thresholds": {
                "mean_gain": GATE_MEAN_GAIN,
                "g0_worsen_max": GATE_G0_WORSEN_MAX,
                "g0_mean_gain_positive": GATE_G0_MEAN_GAIN,
            },
            "per_seed_gain_Y_to_P_cal": {s: gains[s]["gain_Y_to_P_cal"] for s in ("0", "1")},
            "per_seed_gain_Y_to_P_raw": {s: gains[s]["gain_Y_to_P_raw"] for s in ("0", "1")},
            "per_seed_gain_K_to_P_cal": {s: gains[s]["gain_K_to_P_cal"] for s in ("0", "1")},
            "per_seed_gain_K_to_P_raw": {s: gains[s]["gain_K_to_P_raw"] for s in ("0", "1")},
            "per_seed_oracle_loss_cal": {s: gains[s]["oracle_loss_cal"] for s in ("0", "1")},
            "per_seed_G0_worsen": per_seed_g0_worsen,
            "per_seed_G0_marker_P_minus_O": per_seed_g0_marker,
            "conditions": conditions,
            "gate_met": gate_met,
            "secondary_marker_met": secondary_marker,
            "branch": branch,
        },
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    _write_json(RESULTS_DIR / "analysis_summary.json", analysis)
    _write_json(RESULTS_DIR / "gate.json", analysis["gate"])
    _write_json(RESULTS_DIR / "bootstrap.json", bootstrap(datasets))
    _write_csv(RESULTS_DIR / "main_table.csv", main_rows)
    _write_csv(RESULTS_DIR / "group_table.csv", group_rows)
    _write_csv(RESULTS_DIR / "component_table.csv", component_rows(component))
    _write_csv(RESULTS_DIR / "paired_dev_predictions.csv", paired)
    _write_csv(RESULTS_DIR / "severe_rows.csv", severe)
    _write_csv(RESULTS_DIR / "group_gain_table.csv", group_gain_rows(group_gain))
    print(json.dumps(_jsonable(analysis["gate"]), indent=2))
    return 0


# --- dev global index helper (gid order == positional order) ---------------
def _dev_global_index(d: Mapping[str, Any], pos: int) -> int:
    # dev_pos -> global train index.  fit_idx/dev_idx are stored in cycle_targets.
    return int(d["dev_idx_global"][pos])


def replay_head_states() -> dict[str, Any]:
    import torch
    from tracks.ksvd.experiments.luyin16 import zinc_frozen_chemistry_learned_cycle_v1 as runner

    data = dict(np.load(RESULTS_DIR / "T25_all.npz", allow_pickle=False))
    T = data["T"].astype(np.float32)
    fit_idx = data["fit_idx"].astype(np.int64)
    dev_idx = data["dev_idx"].astype(np.int64)
    out = {}
    for s in SEEDS:
        head = runner.build_head(s, 0.0)
        state = torch.load(RESULTS_DIR / f"P_seed{s}_head_soup_state.pt", map_location="cpu")
        head.load_state_dict(state)
        head.eval()
        with torch.no_grad():
            q_all = head(torch.as_tensor(T, dtype=torch.float32)).view(-1).double().numpy()
        saved = dict(np.load(RESULTS_DIR / f"P_seed{s}_predictions.npz", allow_pickle=False))
        q_saved_fit = saved["fit_q"].astype(np.float64)
        q_saved_dev = saved["dev_q"].astype(np.float64)
        diff_fit = float(np.max(np.abs(q_all[fit_idx] - q_saved_fit)))
        diff_dev = float(np.max(np.abs(q_all[dev_idx] - q_saved_dev)))
        out[str(s)] = {
            "max_abs": max(diff_fit, diff_dev),
            "fit_max_abs": diff_fit,
            "dev_max_abs": diff_dev,
            "replayed_all_T25": int(len(T)),
        }
    return out


def component_rows(component: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for s in ("0", "1"):
        for split in ("fit", "dev"):
            for gname, stats in component[s][split].items():
                if not stats or "q_vs_c_mae" not in stats:
                    continue
                rows.append({"seed": int(s), "split": split, "group": gname, **stats})
    return rows


def group_gain_rows(group_gain: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for s in ("0", "1"):
        for gname, vals in group_gain[s].items():
            rows.append({"seed": int(s), "group": gname, **vals})
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())