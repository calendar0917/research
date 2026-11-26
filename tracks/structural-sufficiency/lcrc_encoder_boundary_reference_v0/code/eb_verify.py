"""Independent reload-and-score verification of the pulled B checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import eb_common as E
import cs_common as C

C.setup_paths()

import torch  # noqa: E402

import cs_data as D  # noqa: E402
import eb_data as BD  # noqa: E402
import eb_model as BM  # noqa: E402
import eb_train as BT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-run", default=str(E.RUNS_DIR / "gine_B_seed0"))
    parser.add_argument("--out", default=str(E.ANALYSIS_DIR / "verification.json"))
    args = parser.parse_args()
    b_dir = Path(args.b_run)
    run = E.read_json(b_dir / "run.json")
    bundle = BD.load_bundle()
    sq = BD.load_sq_cache()
    splits = BD.build_splits(bundle, sq)
    categories = BD.category_counts([splits["train"], splits["valid"]])

    ck_path = b_dir / "checkpoint_best.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model, audit = BM.build_model(bundle.spec, bundle.norm, categories, seed=C.SEED, device="cpu")
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch {missing} {unexpected}")
    model.eval()

    rows = np.arange(int(splits["train"].row_ids.size), dtype=np.int64)
    vrows = np.arange(int(splits["valid"].row_ids.size), dtype=np.int64)
    rep_tr = BT.evaluate(splits["train"], rows, np.asarray(bundle.targets_raw["train"]), model, bundle.scaler, torch.device("cpu"))
    rep_va = BT.evaluate(splits["valid"], vrows, np.asarray(bundle.targets_raw["valid"]), model, bundle.scaler, torch.device("cpu"))

    pred = np.load(b_dir / "predictions_best.npz")
    checks = []
    for split, rep in (("train", rep_tr), ("valid", rep_va)):
        saved_rmse = float(run["best_metrics"][split]["rmse_raw"])
        saved_mae = float(run["best_metrics"][split]["mae_raw"])
        e1 = abs(rep["rmse_raw"] - saved_rmse)
        e2 = abs(rep["mae_raw"] - saved_mae)
        order = np.argsort(np.asarray(pred[f"{split}_row_ids"], dtype=np.int64))
        p_saved = np.asarray(pred[f"{split}_pred_raw"], dtype=np.float64)[order]
        rows_saved = np.asarray(pred[f"{split}_row_ids"], dtype=np.int64)[order]
        e3 = float(np.max(np.abs(rep["pred_raw"] - p_saved))) if np.array_equal(rep["row_ids"], rows_saved) else float("inf")
        checks.append({
            "split": split, "n": int(rep["row_ids"].size),
            "recomputed_rmse_raw": rep["rmse_raw"], "saved_rmse_raw": saved_rmse, "abs_delta_rmse": e1,
            "recomputed_mae_raw": rep["mae_raw"], "saved_mae_raw": saved_mae, "abs_delta_mae": e2,
            "max_abs_prediction_delta_vs_saved": e3,
            "tolerance": {"rmse": 1e-7, "mae": 1e-7, "prediction": 1e-5},
            "ok": bool(e1 < 1e-7 and e2 < 1e-7 and e3 < 1e-5),
        })
    payload = {
        "round_id": E.ROUND_ID,
        "checkpoint": E.track_rel(ck_path), "checkpoint_sha256": E.sha256_file(ck_path),
        "checkpoint_epoch": int(ck["epoch"]), "recorded_best_epoch": int(run["best_epoch"]),
        "parameters": int(audit["parameter_groups"]["total"]),
        "checks": checks,
        "all_ok": int(all(c["ok"] for c in checks)),
        "created_utc": C.utcnow(),
    }
    E.write_json(Path(args.out), payload)
    C.log(f"verification all_ok={payload['all_ok']}: {Path(args.out)}")
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
