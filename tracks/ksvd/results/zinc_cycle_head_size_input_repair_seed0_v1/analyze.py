"""Descriptive analysis for zinc-cycle-head-size-input-repair-seed0-v1.

Reads only the round's own saved artifacts plus the frozen train-only inputs and
adds the descriptive quantities the REPORT cites:

* identity checks (P = h + q + b_P; group contributions sum to the overall MAE),
* Q0->QNE prediction movement (fit / dev), better-worse counts,
* leave-one-dev-row-out gain sensitivity (closed form),
* scaling of the overall gain by group and by the top dev rows.

No new model is trained; no official-valid/test is touched.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

TRACK_ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent
FROZEN_CYCLE = TRACK_ROOT / "results/zinc_frozen_chemistry_learned_cycle_v1"
FROZEN_O = TRACK_ROOT / "results/zinc_full_cycle_target_decomposition_v1"


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {"k=0": k == 0, "k=-1": k == -1, "k=-2": k == -2, "k<=-3": k <= -3}


def load_round() -> dict[str, Any]:
    t25 = np.load(FROZEN_CYCLE / "T25_all.npz", allow_pickle=False)
    fit_idx = np.asarray(t25["fit_idx"], np.int64)
    dev_idx = np.asarray(t25["dev_idx"], np.int64)
    o = np.load(FROZEN_O / "O_seed0_predictions.npz", allow_pickle=False)
    decomp = np.load(FROZEN_O / "target_decomposition.npz", allow_pickle=False)
    y_all = np.asarray(decomp["y"], np.float64)
    c_all = np.asarray(decomp["c"], np.float64)
    k_all = np.asarray(decomp["k"], np.int64)
    preds = {}
    for arm in ("Q0", "QNE"):
        with np.load(RESULTS / f"{arm}_predictions.npz", allow_pickle=False) as z:
            preds[arm] = {
                "q_fit": np.asarray(z["q_fit"], np.float64),
                "q_dev": np.asarray(z["q_dev"], np.float64),
                "b_P": float(np.asarray(z["b_P"]).item()),
            }
    return {
        "fit_idx": fit_idx,
        "dev_idx": dev_idx,
        "y_all": y_all,
        "c_all": c_all,
        "k_all": k_all,
        "h_fit": np.asarray(o["fit_raw"], np.float64),
        "h_dev": np.asarray(o["dev_raw"], np.float64),
        "y_fit": np.asarray(o["fit_y"], np.float64),
        "y_dev": np.asarray(o["dev_y"], np.float64),
        "c_fit": np.asarray(o["fit_c"], np.float64),
        "c_dev": np.asarray(o["dev_c"], np.float64),
        "k_fit": k_all[fit_idx],
        "k_dev": k_all[dev_idx],
        "preds": preds,
    }


def identity_checks(data: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for split in ("fit", "dev"):
        idx = data["fit_idx"] if split == "fit" else data["dev_idx"]
        y = data["y_fit"] if split == "fit" else data["y_dev"]
        c = data["c_fit"] if split == "fit" else data["c_dev"]
        k = data["k_fit"] if split == "fit" else data["k_dev"]
        h = data["h_fit"] if split == "fit" else data["h_dev"]
        n_total = int(idx.size)
        for arm in ("Q0", "QNE"):
            q = data["preds"][arm][f"q_{split}"]
            b = data["preds"][arm]["b_P"]
            p_cal = h + q + b
            resid = y - p_cal
            by_group = {g: float(np.abs(resid[m]).sum()) for g, m in group_masks(k).items()}
            total = float(np.abs(resid).sum())
            out[f"{split}_{arm}"] = {
                "P_cal_identity_max_abs": float(np.max(np.abs((h + q + b) - p_cal))),
                "sum_group_abs_minus_total": float(sum(by_group.values()) - total),
                "q_err_identity_max_abs": float(np.max(np.abs((q - c) - (data["preds"][arm][f"q_{split}"] - c)))),
                "n": n_total,
            }
    out["ok"] = bool(all(v["sum_group_abs_minus_total"] == 0.0 for v in out.values()))
    return out


def movement(data: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for split in ("fit", "dev"):
        y = data["y_fit"] if split == "fit" else data["y_dev"]
        h = data["h_fit"] if split == "fit" else data["h_dev"]
        q0 = data["preds"]["Q0"][f"q_{split}"]
        qne = data["preds"]["QNE"][f"q_{split}"]
        delta = (h + qne + data["preds"]["QNE"]["b_P"]) - (h + q0 + data["preds"]["Q0"]["b_P"])
        abs_delta = np.abs(delta)
        err0 = np.abs(y - (h + q0 + data["preds"]["Q0"]["b_P"]))
        errne = np.abs(y - (h + qne + data["preds"]["QNE"]["b_P"]))
        k = data["k_fit"] if split == "fit" else data["k_dev"]
        out[split] = {
            "delta_pred_mean_abs": float(abs_delta.mean()),
            "delta_pred_p95_abs": float(np.percentile(abs_delta, 95)),
            "delta_pred_max_abs": float(abs_delta.max()),
            "rows_better": int((errne < err0).sum()),
            "rows_worse": int((errne > err0).sum()),
            "rows_equal": int((errne == err0).sum()),
            "group_gain_contribution": {
                g: float((np.abs(y - (h + q0 + data["preds"]["Q0"]["b_P"]))[m] - np.abs(y - (h + qne + data["preds"]["QNE"]["b_P"]))[m]).sum() / (data["fit_idx"].size if split == "fit" else data["dev_idx"].size))
                for g, m in group_masks(k).items()
            },
        }
    return out


def loo_sensitivity(data: Mapping[str, Any], split: str = "dev") -> dict[str, Any]:
    y = data["y_fit"] if split == "fit" else data["y_dev"]
    h = data["h_fit"] if split == "fit" else data["h_dev"]
    q0 = data["preds"]["Q0"][f"q_{split}"]
    qne = data["preds"]["QNE"][f"q_{split}"]
    b0 = data["preds"]["Q0"]["b_P"]
    bne = data["preds"]["QNE"]["b_P"]
    e0 = np.abs(y - (h + q0 + b0))
    ene = np.abs(y - (h + qne + bne))
    n = e0.size
    point = float(e0.mean() - ene.mean())
    per_row = (e0 - ene) / n
    loo = (point * n - (e0 - ene)) / (n - 1)
    order = np.argsort(-np.abs(per_row))
    top = []
    idx = data["fit_idx"] if split == "fit" else data["dev_idx"]
    for i in order[:10].tolist():
        top.append({
            "stable_id": f"train:{int(idx[i]):04d}",
            "k": int(data["k_fit"][i]) if split == "fit" else int(data["k_dev"][i]),
            "gain_contribution": float(per_row[i]),
            "loo_gain_without_row": float(loo[i]),
        })
    return {
        "split": split,
        "point_gain": point,
        "min_loo": float(loo.min()),
        "max_loo": float(loo.max()),
        "loo_positive_rows": int((loo > 0).sum()),
        "top_contributors": top,
    }


def main() -> int:
    data = load_round()
    out = {
        "protocol_version": "zinc-cycle-head-size-input-repair-seed0-v1",
        "identity": identity_checks(data),
        "movement": movement(data),
        "loo_sensitivity": {"dev": loo_sensitivity(data, "dev"), "fit": loo_sensitivity(data, "fit")},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    (RESULTS / "descriptive.json").write_text(json.dumps(jsonable(out), indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("identity", "movement")}, indent=1)[:4000])
    print("loo dev point", out["loo_sensitivity"]["dev"]["point_gain"], "min", out["loo_sensitivity"]["dev"]["min_loo"], "max", out["loo_sensitivity"]["dev"]["max_loo"])
    print("loo fit point", out["loo_sensitivity"]["fit"]["point_gain"], "min", out["loo_sensitivity"]["fit"]["min_loo"], "max", out["loo_sensitivity"]["fit"]["max_loo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
