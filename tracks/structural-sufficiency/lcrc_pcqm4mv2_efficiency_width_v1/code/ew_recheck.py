"""Independent local re-evaluation of a new-round checkpoint (CPU, no GPU).

Loads ``runs/<run>/<checkpoint>``, restores it into a freshly built model with
the round's audit rules, and re-runs inference on the **full official
Validation split** from the local feature cache.  Nothing is trained, nothing is
selected, no Test row is touched.  This is the second, independent check of
every reported Validation MAE (the first is the official evaluator inside the
training process).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Sequence

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ew_common as C  # noqa: E402
import ew_data as DATA  # noqa: E402
import ew_model as M  # noqa: E402
import ew_train as T  # noqa: E402

import torch  # noqa: E402


def read_curve(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                parsed[key] = value
        out.append(parsed)
    return out


def recheck(run: str, checkpoint: str = "checkpoint_best_valid.pt", device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    path = C.RUNS / run / checkpoint
    if not path.exists():
        raise FileNotFoundError(path)
    t0 = time.perf_counter()
    task = DATA.load_task(run, verify_hashes=False)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model, model_audit = M.build_model(
        task["spec"], task["norm"], scale=int(task["scale"]), seed=int(task["seed"]), device=device
    )
    optimizer, _opt_audit = M.make_optimizer(model)
    restore_audit = T.restore(payload, model, optimizer, run=run, task=task)
    report, _pred = C.PQ_TRAIN.evaluate_split(model, task, device, which="valid", batch_size=512)
    saved = dict(payload["valid_report_summary"] or {})
    curve = read_curve(C.RUNS / run / "train_curve.csv")
    epoch = int(payload["epoch"])
    curve_row = next((r for r in curve if int(r["epoch"]) == epoch), {})
    out = {
        "schema": "lcrc-pcqm4mv2-efficiency-width-recheck-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "run": run,
        "model": task["model"],
        "scale": int(task["scale"]),
        "parameters": int(model_audit["total_parameters"]),
        "checkpoint": C.track_rel(path),
        "checkpoint_bytes": int(path.stat().st_size),
        "checkpoint_sha256": C.sha256_file(path),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "epoch": epoch,
        "global_step": int(payload["global_step"]),
        "device": str(device),
        "restore_audit": {
            "optimizer_state_restored": restore_audit["optimizer_state_restored"],
            "replayed_epoch_order_hashes_verified": restore_audit["replayed_epoch_order_hashes_verified"],
            "resumed_from_epoch": restore_audit["resumed_from_epoch"],
        },
        "n_valid": int(report["n"]),
        "recorded_valid_mae_eV": float(saved.get("mae_eV", float("nan"))),
        "recomputed_valid_mae_eV": float(report["mae_eV"]),
        "abs_delta_valid_mae_eV": abs(float(report["mae_eV"]) - float(saved.get("mae_eV", float("nan")))),
        "recorded_pred_sha256": saved.get("pred_sha256"),
        "recomputed_pred_sha256": report["pred_sha256"],
        # CPU and GPU reductions are not bitwise identical; this is informational
        "pred_sha256_matches": bool(saved.get("pred_sha256") == report["pred_sha256"]),
        "curve_valid_mae_eV": float(curve_row["valid_mae_eV"]) if curve_row else None,
        "curve_matches_recorded": (
            bool(abs(float(curve_row["valid_mae_eV"]) - float(saved.get("mae_eV", float("nan")))) <= 1e-12)
            if curve_row
            else None
        ),
        "numpy_mae_eV": float(report["mae_eV_numpy"]),
        "crosscheck_delta_eV": float(report["crosscheck_delta_eV"]),
        "seconds": float(time.perf_counter() - t0),
        "test_evaluated": False,
    }
    out["status"] = (
        "OK"
        if (
            out["abs_delta_valid_mae_eV"] <= 1e-6
            and out["n_valid"] == 73_545
            and out["curve_matches_recorded"] is not False
        )
        else "FAILED"
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(C.RUN_ORDER))
    parser.add_argument("--checkpoint", default="checkpoint_best_valid.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    payload = recheck(args.run, args.checkpoint, args.device)
    out = Path(args.out) if args.out else C.ANALYSIS / f"{args.run}_recheck.json"
    C.write_json(out, payload)
    C.log(
        f"[recheck] {args.run}: {payload['status']} recorded {payload['recorded_valid_mae_eV']:.9f} "
        f"recomputed {payload['recomputed_valid_mae_eV']:.9f} eV ({payload['seconds']:.1f}s) "
        f"-> {C.track_rel(out)}"
    )
    return 0 if payload["status"] == "OK" else 5


if __name__ == "__main__":
    raise SystemExit(main())
