"""Checkpoint replay verification for the two Full bases (local CPU).

Loads each saved soup state, folds the recorded calibration delta ONCE, and
compares the first 128 outer-dev predictions to the saved per-row array.  This
turns the JSON predictions into *checkpoint-replayed* facts.  Official test is
not touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_topology_crossfit_diagnostic_v1 as d
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run

OUT = Path(__file__).resolve().parent


def main() -> int:
    idx = d.load_subfold_index()
    dev_idx = idx["dev_idx"]
    report = {}
    for fold in ("A", "B"):
        res = json.loads((d.RESULTS_DIR / f"F_{fold}.json").read_text())
        blob = d.load_blob(d.PREP_DIR / f"fold_{fold}.npz")
        train_data = p1run.load_split("train")
        valid_data = p1run.load_split("valid")
        d.zjd.apply_prep_blob(train_data, valid_data, blob)
        dev_set = set(int(i) for i in dev_idx.tolist())
        dev_data = [x for i, x in enumerate(train_data) if i in dev_set]
        model = d.zjd.make_arm_model("F", blob, seed=d.zjd.SEED)
        sd = torch.load(d.RESULTS_DIR / f"F_{fold}_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(sd["soup_state"])
        with torch.no_grad():
            model.reader.net[4].bias.add_(float(res["calibration"]["delta"]))
        model.eval()
        pred, y = d.zjd.collect_predictions(model, dev_data[:128], torch.device("cpu"))
        saved = np.asarray(res["dev_predictions_cal"], np.float32)[:128]
        report[fold] = {"delta": res["calibration"]["delta"],
                        "max_abs_diff_vs_saved": float(np.abs(pred - saved).max()),
                        "replay_matches": bool(np.abs(pred - saved).max() < 1e-4),
                        "n_checked": int(len(pred))}
        print(fold, report[fold])
    (OUT / "replay_verification.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())