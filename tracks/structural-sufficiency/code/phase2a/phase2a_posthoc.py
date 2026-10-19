"""SSRC Phase 2a post-hoc tail diagnostics on the frozen Monitor predictions.

Everything here is *descriptive*: no training, no model selection, no change to
any pre-registered threshold.  It exists because the pre-registered RMSE guard
is an absolute threshold that a single molecule can flip; the report needs to
say which molecule and whether the squared-error difference is distinguishable
from zero.

Inputs: frozen prediction arrays, the split, the Fit/Monitor label vault, the
Phase 1e label audit and the scored analysis.  Output: ``posthoc_phase2a.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import phase2a_common as C  # noqa: E402

OUT = C.RESULTS_2A / "posthoc_phase2a.json"
N_BOOT = 5000
BOOT_SEED = 20261024


def _pred(arm: str) -> np.ndarray:
    path = C.RESULTS_2A / f"monitor_pred_{arm}_s0.npz"
    freeze = C.read_json(C.RESULTS_2A / f"monitor_freeze_{arm}_s0.json")
    if C.sha256_file(path) != freeze["prediction_sha256"]:
        raise SystemExit(f"prediction file changed after the freeze: {arm}")
    with np.load(path) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
    if C.sha256_array(pred) != freeze["prediction_array_sha256"]:
        raise SystemExit(f"prediction array changed after the freeze: {arm}")
    return pred


def _rmse_diff_ci(sq_a: np.ndarray, sq_b: np.ndarray, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Bootstrap CI of sqrt(mean sq_a) - sqrt(mean sq_b) over molecules."""
    delta = sq_a - sq_b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    a = np.sqrt(sq_a[idx].mean(1))
    b = np.sqrt(sq_b[idx].mean(1))
    diff = a - b
    return {
        "mean": float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())),
        "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "mse_mean_diff": float(delta.mean()),
        "mse_ci95": [
            float(np.percentile(delta[idx].mean(1), 2.5)),
            float(np.percentile(delta[idx].mean(1), 97.5)),
        ],
        "n": int(delta.size),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    if OUT.exists():
        raise SystemExit("posthoc record exists; refusing to rewrite")

    split = C.load_split()
    labels = C.load_vault()
    y = C.row_labels(split.monitor, labels)
    groups = C.label_groups(split.monitor)
    masks = groups["masks"]
    p3e = C.load_p3e_predictions().pred
    preds = {"P3E": p3e, "F": _pred("F"), "M": _pred("M")}
    ae = {k: np.abs(v - y) for k, v in preds.items()}
    sq = {k: (v - y) ** 2 for k, v in preds.items()}

    contrasts: dict[str, Any] = {}
    for name, a, b in (("M_vs_P3E", "M", "P3E"), ("M_vs_F", "M", "F"), ("F_vs_P3E", "F", "P3E")):
        d = sq[a] - sq[b]
        order = np.argsort(-np.abs(d), kind="stable")
        keep = np.ones(d.size, dtype=bool)
        keep[order[0]] = False
        contrasts[name] = {
            "definition": "mean(|e_a|) and rmse with a > 0 meaning a is worse",
            "mae_delta": float(ae[a].mean() - ae[b].mean()),
            "rmse_delta": _rmse_diff_ci(sq[a], sq[b]),
            "rmse_delta_drop_worst_sq_contributor": float(
                np.sqrt(sq[a][keep].mean()) - np.sqrt(sq[b][keep].mean())
            ),
            "worst_sq_contributor": {
                "monitor_index": int(order[0]),
                "official_train_row": int(split.monitor[order[0]]),
                "y": float(y[order[0]]),
                "pred_a": float(preds[a][order[0]]),
                "pred_b": float(preds[b][order[0]]),
                "abs_error_a": float(ae[a][order[0]]),
                "abs_error_b": float(ae[b][order[0]]),
                "mse_contribution_share_of_total_diff": float(d[order[0]] / d.sum()),
                "in_label_penalised": bool(masks["label_penalised"][order[0]]),
                "in_permutation_sensitive": bool(masks["permutation_sensitive"][order[0]]),
            },
            "n_molecules_a_better_abs": int((ae[a] < ae[b]).sum()),
            "n_molecules_a_worse_abs": int((ae[a] > ae[b]).sum()),
        }

    # RMSE with the permutation-unstable (label is not a graph function) rows removed
    unstable = masks["permutation_sensitive"]
    stable = ~unstable
    unstable_report = {
        "n": int(unstable.sum()),
        "rows": [int(r) for r in split.monitor[unstable]],
        "mae": {k: float(ae[k][unstable].mean()) for k in ae},
        "rmse_on_stable_complement": {k: float(np.sqrt(sq[k][stable].mean())) for k in sq},
        "rmse_delta_M_minus_P3E_on_stable": float(
            np.sqrt(sq["M"][stable].mean()) - np.sqrt(sq["P3E"][stable].mean())
        ),
        "note": (
            "the Phase 1e audit shows these rows change their cycle-penalty level under "
            "atom permutation, so part of their target is not a function of the unordered graph"
        ),
    }

    validity = {
        "corr_pred_F_M": float(np.corrcoef(preds["F"], preds["M"])[0, 1]),
        "corr_pred_M_P3E": float(np.corrcoef(preds["M"], preds["P3E"])[0, 1]),
        "p3e_mae_recomputed": float(ae["P3E"].mean()),
        "p3e_mae_frozen": C.P3E_MAE,
        "p3e_mae_abs_diff": abs(float(ae["P3E"].mean()) - C.P3E_MAE),
        "label_alignment_ok": abs(float(ae["P3E"].mean()) - C.P3E_MAE) < 1e-9,
    }

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "posthoc_tail_diagnostics",
        "scope": (
            "descriptive only; frozen predictions, no training, no selection, no threshold change"
        ),
        "monitor_labels_read": "re-read of the already-scored Monitor (second read, no new data)",
        "reserved_labels_read": 0,
        "official_valid_test_loaded": False,
        "boot": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "molecule"},
        "validity": validity,
        "contrasts": contrasts,
        "permutation_unstable": unstable_report,
        "decision_note": (
            "the pre-registered RMSE guard is unchanged; this file only explains why it "
            "missed its absolute tolerance"
        ),
    }
    C.write_json(OUT, payload)
    print(f"[posthoc] rmse delta M-P3E {contrasts['M_vs_P3E']['rmse_delta']['mean']:+.6f} "
          f"CI {contrasts['M_vs_P3E']['rmse_delta']['ci95']} "
          f"drop-worst {contrasts['M_vs_P3E']['rmse_delta_drop_worst_sq_contributor']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
