"""Computational interventions on the trained LOCAL3 weights.

Three forward modes on **one fixed weight vector**:

* ``as_trained``        - the LOCAL3 operator (``h + psi(h, s3_v)``)
* ``correction_off``    - branch disabled, ``h' = h``
* ``null3_operator``    - the exact NULL3 mean-field operator with LOCAL3's own
                          ``s3`` normalisation and all of LOCAL3's weights

These are model-computation interventions on a single trained network.  They are
not chemical edits and they are not causal effects.  The primary mechanism
comparison remains the independently trained matched control (NULL3).
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C
import data as D
import features as F
import metrics as M
import model as MD


MODES = {
    "as_trained": "local3",
    "correction_off": "off",
    "null3_operator": "null3",
}


def run(device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    dataset = D.load_dataset(include_test=True)
    spec = dataset["spec"]
    norm = dataset["train_norm"]
    target_norm = dataset["target_norm"]
    features = F.load_features()
    feats = {
        split: F.split_features(features, split, dataset[f"{split}_cache"]) for split in C.SPLITS
    }
    run_dir = C.run_dir("LOCAL3")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    src = C.source_with_labels("all")
    rows = {s: [int(r) for r in src["rows"][s]] for s in C.SPLITS}
    y = {s: np.asarray(src["y"][s], dtype=np.float64) for s in C.SPLITS}

    out: dict[str, Any] = {
        "schema": "lcrc-lipo-s3-locality-interventions-v1",
        "round_id": C.ROUND_ID,
        "checkpoint": {
            "path": C.track_rel(run_dir / "best.pt"),
            "sha256": C.sha256_file(run_dir / "best.pt"),
            "best_epoch": int(ckpt["best_epoch"]),
            "state_hash": ckpt.get("state_hash"),
        },
        "modes": {},
        "note": (
            "one fixed trained weight vector; these are model-computation interventions, "
            "not chemical edits or training-post causal effects"
        ),
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for label, mode in MODES.items():
        Model = MD.load_trained_model(spec, norm, ckpt, device, mode=mode)
        Model.eval()
        metrics = {}
        preds = {}
        corr_stats = {}
        for split in C.SPLITS:
            std = D.predict_scores(Model, dataset[f"{split}_cache"], device, feats=feats[split])
            # the network predicts standardised targets; raw logD is produced once
            preds[split] = std * float(target_norm["scale"]) + float(target_norm["mean"])
            metrics[split] = M.regression_metrics(preds[split], y[split])
            corr_stats[split] = dict(getattr(Model, "s3_last_stats", {}))
        predictions[label] = preds
        out["modes"][label] = {
            "mode": mode,
            "metrics": metrics,
            "correction_stats_last_batch": corr_stats,
        }
        C.log(
            f"[intervene] {label}: valid={metrics['valid']['rmse']:.6f} test={metrics['test']['rmse']:.6f} "
            f"corr_norm={corr_stats['test'].get('correction_norm_mean', 0.0):.4f}"
        )

    # prediction movement relative to the as-trained operator
    base = predictions["as_trained"]
    movement = {}
    for label in ("correction_off", "null3_operator"):
        movement[label] = {
            split: {
                "rmse": float(np.sqrt(np.mean((predictions[label][split] - base[split]) ** 2))),
                "mae": float(np.mean(np.abs(predictions[label][split] - base[split]))),
                "max_abs": float(np.max(np.abs(predictions[label][split] - base[split]))),
            }
            for split in C.SPLITS
        }
    out["prediction_movement_vs_as_trained"] = movement

    case_rows = []
    for ra, rb in C.ALIAS_PAIRS:
        sa = next(s for s in C.SPLITS if ra in set(rows[s]))
        sb = next(s for s in C.SPLITS if rb in set(rows[s]))
        ya = float(y[sa][rows[sa].index(ra)])
        yb = float(y[sb][rows[sb].index(rb)])
        for label in MODES:
            pa = float(predictions[label][sa][rows[sa].index(ra)])
            pb = float(predictions[label][sb][rows[sb].index(rb)])
            case_rows.append(
                {
                    "arm": "LOCAL3",
                    "variant": label,
                    "row_a": ra,
                    "row_b": rb,
                    "split_a": sa,
                    "split_b": sb,
                    "y_a": ya,
                    "y_b": yb,
                    "y_diff": yb - ya,
                    "pred_a": pa,
                    "pred_b": pb,
                    "pred_diff": pb - pa,
                    "diff_error": (pb - pa) - (yb - ya),
                    "abs_diff_error": abs((pb - pa) - (yb - ya)),
                }
            )
    out["case_rows"] = case_rows

    # NULL3 degeneracy on the Train split, from the static structural vectors
    degenerate = {
        split: F.null_group_stats(dataset[f"{split}_cache"], np.asarray(features[f"{split}_s3"], dtype=np.float32))
        for split in C.SPLITS
    }
    out["null3_degeneracy"] = degenerate
    C.write_json(C.ANALYSIS / "interventions.json", out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LOCAL3 computational interventions")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    run(args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
