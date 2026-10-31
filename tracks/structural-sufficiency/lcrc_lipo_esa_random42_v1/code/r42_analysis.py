"""Post-lock secondary analysis of the *already frozen* Test predictions.

This module is deliberately separate from ``r42_score``:

* it refuses to run unless ``analysis/test_read.lock`` exists, i.e. unless the
  one-shot Test evaluation has already happened;
* it only re-reads Test labels to quantify uncertainty (bootstrap CIs) and to
  break errors down by molecule size;
* it never fits, selects or re-scores a model - every prediction vector is
  re-hashed against ``analysis/prediction_manifest.json`` before use.

The numbers it produces are descriptive. The single pre-registered metric is
the RMSE written by ``r42_score``; nothing here can change it.
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

import r42_common as C
import r42_metrics as MET

N_BOOT = 10_000


def _test_labels() -> np.ndarray:
    split = C.read_json(C.SPLIT_FILE)
    rows = [int(r) for r in split["rows"]["test"]]
    values: list[float] = []
    with open(C.CSV_PATH, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            values.append(float(row["exp"]))
    y = np.array([values[r] for r in rows], dtype=np.float64)
    return y


def _frozen_predictions() -> dict[str, np.ndarray]:
    manifest = C.read_json(C.ANALYSIS / "pre_test_manifest.json")
    out: dict[str, np.ndarray] = {}
    for row in manifest["predictions"]:
        path = C.REPO / row["path"]
        if C.sha256_file(path) != row["sha256"]:
            raise PermissionError(f"{row['model']}: frozen prediction hash drifted since the manifest")
        with np.load(path) as npz:
            out[row["model"]] = np.asarray(npz["pred"], dtype=np.float64)
    return out


def run() -> dict[str, Any]:
    lock = C.ANALYSIS / "test_read.lock"
    if not lock.exists():
        raise PermissionError("refusing to run before the one-shot Test evaluation (no lock file)")
    y = _test_labels()
    scored = C.read_json(C.ANALYSIS / "metrics_test.json")
    if C.sha256_bytes(np.ascontiguousarray(y, dtype="<f8").tobytes()) != scored["test_label_sha256"]:
        raise PermissionError("Test label vector does not match the scored one")
    preds = _frozen_predictions()
    n = y.size

    rng = np.random.default_rng(0)
    boot = rng.integers(0, n, size=(N_BOOT, n))

    def rmse(p: np.ndarray, sel: np.ndarray) -> float:
        return float(np.sqrt(np.mean((p[sel] - y[sel]) ** 2)))

    idx = np.arange(n)
    per_model = {}
    for model, pred in preds.items():
        values = np.array([rmse(pred, sel) for sel in boot])
        per_model[model] = {
            "rmse": rmse(pred, idx),
            "rmse_ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "mae": MET.mae(pred, y),
            "r2": MET.r2(pred, y),
        }

    pairs = [
        ("PNA-native", "LCRC-A-R3-scratch"),
        ("ESA-native", "LCRC-A-R3-scratch"),
        ("PNA-native", "ESA-native"),
        ("ESA-native", "XGB-P3"),
        ("LCRC-A-R3-scratch", "XGB-P3"),
        ("PNA-native", "XGB-P3"),
    ]
    deltas = []
    for a, b in pairs:
        d = np.array([rmse(preds[a], sel) - rmse(preds[b], sel) for sel in boot])
        deltas.append(
            {
                "a": a,
                "b": b,
                "delta_rmse": rmse(preds[a], idx) - rmse(preds[b], idx),
                "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "prob_a_worse": float((d > 0).mean()),
            }
        )

    baseline = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    relative = {m: 1.0 - per_model[m]["rmse"] / baseline for m in per_model}

    sq = {m: (preds[m] - y) ** 2 for m in preds}
    names = list(preds)
    correlations = {
        f"{a}|{b}": float(np.corrcoef(sq[a], sq[b])[0, 1]) for i, a in enumerate(names) for b in names[i + 1 :]
    }

    argmin = np.argmin(np.stack([np.abs(preds[m] - y) for m in names]), axis=0)
    closest = {m: float(np.mean(argmin == i)) for i, m in enumerate(names)}

    with np.load(C.STRUCTURES_FILE) as npz:
        atom_ptr = npz["atom_ptr"]
    sizes = np.diff(atom_ptr)
    split = C.read_json(C.SPLIT_FILE)
    test_sizes = sizes[[int(r) for r in split["rows"]["test"]]]
    cuts = np.quantile(test_sizes, [1 / 3, 2 / 3])
    by_size = {}
    for m in names:
        err = np.abs(preds[m] - y)
        by_size[m] = {
            "mae_small": float(err[test_sizes <= cuts[0]].mean()),
            "mae_medium": float(err[(test_sizes > cuts[0]) & (test_sizes <= cuts[1])].mean()),
            "mae_large": float(err[test_sizes > cuts[1]].mean()),
        }

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "POST_LOCK_DESCRIPTIVE",
        "test_labels_read": "again, after the one-shot scoring lock, for uncertainty quantification only",
        "affects_scored_metrics": False,
        "n_test": int(n),
        "constant_baseline_rmse": baseline,
        "n_bootstrap": N_BOOT,
        "per_model": per_model,
        "relative_rmse_reduction_vs_baseline": relative,
        "paired_deltas": deltas,
        "squared_error_correlation": correlations,
        "closest_model_fraction": closest,
        "error_by_molecule_size_tertile": {
            "cuts_atoms": [float(x) for x in cuts],
            "per_model": by_size,
        },
    }
    C.write_json(C.ANALYSIS / "secondary_analysis.json", payload)
    C.log("[secondary] post-lock descriptive analysis written")
    return payload


if __name__ == "__main__":  # pragma: no cover
    run()
