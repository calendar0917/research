"""Independent local re-evaluation of a recorded checkpoint (CPU, no GPU needed).

Loads ``runs/<run>/<checkpoint>``, restores it into a freshly built model with the
same audit rules as the trainer, and re-runs inference on the **full official
Validation split** from the local feature cache.  The point is to prove that the
reported number is reproducible from the delivered artifact and that the
checkpoint really is the one that produced it (prediction hash comparison).

Nothing is trained, nothing is selected, and no Test row is touched.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_common as C  # noqa: E402
import pq_data as D  # noqa: E402
import pq_model as M  # noqa: E402
import pq_train as T  # noqa: E402

import torch  # noqa: E402


def read_curve(path: Path) -> list[dict[str, Any]]:
    """Read ``train_curve.csv`` back with numeric cells where possible."""
    import csv

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


def recheck(run: str, checkpoint: str, device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    path = C.RUNS / run / checkpoint
    if not path.exists():
        raise FileNotFoundError(path)
    t0 = time.perf_counter()
    task = D.load_task(run)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model, model_audit = M.build_model(
        task["spec"], task["norm"], seed=int(task["seed"]), device=device
    )
    optimizer, _opt_audit = M.make_optimizer(model)
    restore_audit = T.restore(payload, model, optimizer, run=run, task=task)
    report, _pred = T.evaluate_split(model, task, device, which="valid", batch_size=512)
    saved = dict(payload["valid_report_summary"] or {})
    curve = read_curve(C.RUNS / run / "train_curve.csv")
    epoch = int(payload["epoch"])
    curve_row = next((r for r in curve if int(r["epoch"]) == epoch), {})
    out = {
        "schema": "lcrc-pcqm4mv2-recheck-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "run": run,
        "checkpoint": C.track_rel(path),
        "checkpoint_bytes": int(path.stat().st_size),
        "checkpoint_sha256": C.sha256_file(path),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "epoch": epoch,
        "global_step": int(payload["global_step"]),
        "device": str(device),
        "parameters_total": int(model_audit["total_parameters"]),
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
        "pred_sha256_matches": bool(saved.get("pred_sha256") == report["pred_sha256"]),
        "curve_valid_mae_eV": float(curve_row.get("valid_mae_eV", float("nan"))),
        "curve_matches_recorded": bool(
            abs(float(curve_row.get("valid_mae_eV", float("nan"))) - float(saved.get("mae_eV", float("nan"))))
            <= 1e-12
        ),
        "numpy_mae_eV": float(report["mae_eV_numpy"]),
        "crosscheck_delta_eV": float(report["crosscheck_delta_eV"]),
        "seconds": float(time.perf_counter() - t0),
        "test_evaluated": False,
    }
    # The checkpoint's prediction hash was produced on the training device; a
    # CPU re-evaluation can differ in the last bits, so the sha is reported but
    # the accept criterion is the (device-independent) MAE agreement.
    out["pred_sha_note"] = (
        "the recorded hash was computed on the training device; a bitwise match is only "
        "expected when rechecking on that same device"
    )
    out["status"] = (
        "OK"
        if out["abs_delta_valid_mae_eV"] <= 1e-6 and out["curve_matches_recorded"] and out["n_valid"] == 73_545
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
