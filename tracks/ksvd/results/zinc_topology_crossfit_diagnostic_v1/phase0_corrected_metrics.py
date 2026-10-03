"""Phase 0 — correct the historical calibration scope of zinc-joint-dictionary-decision-v1.

`zinc_joint_dictionary_decision_v1.run_arm` folds the train-fit median residual
into the reader output bias **and then** stores `dev_predictions_raw` collected
*after* that fold.  It then reports `dev_predictions_calibrated = dev_raw + delta`
by adding the SAME delta a second time.  Therefore:

* the true unfolded raw prediction is  `saved_dev_raw - delta`;
* the correctly single-calibrated prediction is `saved_dev_raw`;
* the historical `dev_predictions_calibrated` is double-calibrated.

No soup checkpoints were saved (only per-arm `soup_state_sha256`), so this is a
**cached-prediction recomputation**, not a checkpoint replay.  This script
recomputes the historical metrics from the saved per-row arrays, rebuilds the
group contributions and paired bootstrap, and records the minimal algebra check
that folding the bias shifts every prediction by exactly `delta`.

Official test is never touched.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

RESULTS = zjd.RESULTS_DIR
OUT = Path(__file__).resolve().parent
ARMS = ("F", "B", "D")


def load(arm: str) -> dict:
    return json.loads((RESULTS / f"{arm}_seed0.json").read_text())


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def strata_table(pred: np.ndarray, y: np.ndarray, pen: np.ndarray) -> dict:
    out = {}
    for name, sel in (("penalty_0", pen == 0), ("penalty_-1", pen == -1), ("penalty_le_-2", pen <= -2)):
        sel = np.asarray(sel)
        if not sel.any():
            out[name] = {"n": 0, "mae": None, "contrib": 0.0}
            continue
        err = np.abs(pred[sel] - y[sel])
        out[name] = {"n": int(sel.sum()), "mae": float(err.mean()),
                     "contrib": float(err.sum() / len(y))}
    out["overall"] = {"n": int(len(y)), "mae": mae(pred, y)}
    return out


def bootstrap_paired(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray,
                     n_boot: int = 1000, seed: int = 20261003) -> dict:
    """Descriptive paired row-bootstrap CI for MAE(a)-MAE(b)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    ea = np.abs(pred_a - y)
    eb = np.abs(pred_b - y)
    obs = float(ea.mean() - eb.mean())
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(ea[idx].mean() - eb[idx].mean())
    return {"point": obs, "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)), "n_boot": n_boot, "seed": seed}


def main() -> int:
    train_pen, _ = zjd._load_penalties()
    split = json.loads((zjd.PREP_DIR / "split.json").read_text())
    # dev_idx is stored in the prep blob, not split.json
    with np.load(zjd.PREP_DIR / "fold_objects.npz", allow_pickle=False) as z:
        dev_idx = z["dev_idx"].astype(np.int64)
    pen_dev = train_pen[dev_idx]

    runs = {a: load(a) for a in ARMS}
    y = np.asarray(runs["F"]["dev_targets"], dtype=np.float64)

    corrected = {}
    for a in ARMS:
        r = runs[a]
        delta = float(r["calibration"]["delta"])
        saved_raw = np.asarray(r["dev_predictions_raw"], dtype=np.float64)
        saved_cal = np.asarray(r["dev_predictions_calibrated"], dtype=np.float64)
        true_raw = saved_raw - delta
        # relation checks
        assert np.allclose(saved_cal, saved_raw + delta, atol=1e-9)
        corrected[a] = {
            "delta": delta,
            "reader_bias_before": r["calibration"]["reader_bias_before"],
            "reader_bias_after": r["calibration"]["reader_bias_after"],
            "saved_dev_raw_mae": r["dev_raw_mae"],
            "saved_dev_cal_mae_double_calibrated": r["dev_cal_mae"],
            "true_unfolded_raw_mae": mae(true_raw, y),
            "correct_single_calibrated_mae": mae(saved_raw, y),
            "true_raw_sha256": sha_arr(true_raw),
            "correct_cal_sha256": sha_arr(saved_raw),
            "historical_double_cal_sha256": sha_arr(saved_cal),
            "historical_strata_double": strata_table(saved_cal, y, pen_dev),
            "corrected_strata_single": strata_table(saved_raw, y, pen_dev),
            "corrected_strata_true_raw": strata_table(true_raw, y, pen_dev),
        }

    # paired gains: D relative to F and B, corrected (single calibration)
    pF = np.asarray(runs["F"]["dev_predictions_raw"], dtype=np.float64)
    pB = np.asarray(runs["B"]["dev_predictions_raw"], dtype=np.float64)
    pD = np.asarray(runs["D"]["dev_predictions_raw"], dtype=np.float64)
    gains = {
        "D_minus_F_corrected": {
            "gain_mae(F)-mae(D)": float(mae(pF, y) - mae(pD, y)),
            "bootstrap": bootstrap_paired(pD, pF, y),
        },
        "D_minus_B_corrected": {
            "gain_mae(B)-mae(D)": float(mae(pB, y) - mae(pD, y)),
            "bootstrap": bootstrap_paired(pD, pB, y),
        },
    }
    # historical double-calibrated versions for the errata record
    pFc = np.asarray(runs["F"]["dev_predictions_calibrated"], dtype=np.float64)
    pBc = np.asarray(runs["B"]["dev_predictions_calibrated"], dtype=np.float64)
    pDc = np.asarray(runs["D"]["dev_predictions_calibrated"], dtype=np.float64)
    gains["D_minus_F_historical_double"] = {
        "gain": float(mae(pFc, y) - mae(pDc, y)), "bootstrap": bootstrap_paired(pDc, pFc, y)}
    gains["D_minus_B_historical_double"] = {
        "gain": float(mae(pBc, y) - mae(pDc, y)), "bootstrap": bootstrap_paired(pDc, pBc, y)}

    soup_saved = sorted(str(p) for p in RESULTS.glob("*.pt"))
    payload = {
        "source": "cached prediction recomputation (no checkpoint replay)",
        "source_files": {a: str(RESULTS / f"{a}_seed0.json") for a in ARMS},
        "soup_checkpoints_present": soup_saved,
        "soup_state_sha256": {a: runs[a]["soup_state_sha256"] for a in ARMS},
        "dev_idx_sha256": split["dev_idx_sha256"],
        "n_dev": int(len(dev_idx)),
        "corrected_arms": corrected,
        "paired_gains": gains,
        "official_test_loaded": False,
    }
    (OUT / "corrected_historical_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "mae": {a: corrected[a]["correct_single_calibrated_mae"] for a in ARMS},
        "true_raw": {a: corrected[a]["true_unfolded_raw_mae"] for a in ARMS},
        "D-F_corrected": gains["D_minus_F_corrected"]["gain_mae(F)-mae(D)"],
        "D-B_corrected": gains["D_minus_B_corrected"]["gain_mae(B)-mae(D)"],
        "D-F_hist": gains["D_minus_F_historical_double"]["gain"],
        "D-B_hist": gains["D_minus_B_historical_double"]["gain"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
