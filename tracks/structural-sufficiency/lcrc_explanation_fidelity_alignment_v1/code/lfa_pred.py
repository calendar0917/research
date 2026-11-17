"""Freeze full-split Lipophilicity predictions for the auxiliary diagnostics.

The main experiment never needs a Lipophilicity prediction: it is a
description of the frozen models on the 823 fixed pairs, so the numbers must be
computed once and stored, not re-derived differently in the report.  Every
split is replayed in evaluation mode with the frozen checkpoint; the Valid RMSE
is compared with the checkpoint-identity record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import lfa_common as C
import lfa_data as D
import lfa_models as M


def run(*, out_path: Path | str | None = None, device: str = "cpu") -> dict[str, Any]:
    out_path = Path(out_path) if out_path is not None else C.REPLAY_DIR / "lipo_predictions.npz"
    C.ensure_dir(out_path.parent)
    b = D.bundle()
    payload: dict[str, np.ndarray] = {}
    reports = []
    for family, name in (("lcrc", "L_lipophilicity"), ("gine", "G0_lipophilicity")):
        model, meta = M.load_model(name, b, device=device)
        before = M.parameter_hash(model)
        for split in ("train", "valid", "test"):
            out = M.predict_split(model, b, split, device=device)
            payload[f"{split}_{family}_row"] = np.asarray(out["row_id"], dtype=np.int64)
            payload[f"{split}_{family}_y"] = np.asarray(out["y_raw"], dtype=np.float64)
            payload[f"{split}_{family}_pred"] = np.asarray(out["pred_raw"], dtype=np.float64)
            reports.append({"family": family, "split": split, "n": out["n_rows"],
                            "rmse": out["rmse_raw"], "mae": out["mae_raw"]})
        after = M.parameter_hash(model)
        expected = float(C.EXPECTED_CHECKPOINT_META[name]["valid_rmse_raw"])
        got = next(r["rmse"] for r in reports if r["family"] == family and r["split"] == "valid")
        if before != after or abs(got - expected) > 1e-6:
            raise RuntimeError(f"{name}: replay mismatch rmse={got} expected={expected} params={before == after}")
        del model
    C.write_npz(out_path, payload)
    record = {
        "round_id": C.ROUND_ID, "created_utc": C.utc_now(), "environment": C.environment(),
        "path": C.relpath(out_path), "sha256": C.sha256_file(out_path),
        "reports": reports, "code_hashes": C.code_hashes(),
    }
    C.write_json(C.ANALYSIS_DIR / "lipo_replay.json", record)
    print(f"[pred] {json.dumps(reports)}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.REPLAY_DIR / "lipo_predictions.npz"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(out_path=args.out, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
