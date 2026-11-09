"""Materialise the missing Validation logits from the frozen best-Validation
checkpoints (read-only forward, CPU, no gradient, no training).

Historical rounds saved Test logits and *recorded* Validation logits hashes, but
not the Validation per-molecule logits.  This module performs exactly one
Validation forward per frozen checkpoint and gates the replay against the frozen
Validation AP (tolerance ``1e-4``).

Run::

    uv run python code/lc_valid.py run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import lc_common as C
import lc_data as D

for _p in (str(C.BOC_ROUND / "code"), str(C.S3SEL_ROUND / "code"), str(C.CC_ROUND / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import boc_test as BT  # noqa: E402
import sp_test as SPT  # noqa: E402

MANIFEST_PATH = C.LOCAL_VALID / "valid_prediction_manifest.json"
ROWS_PATH = C.LOCAL_VALID / "valid_row_ids.npy"


def _entry(label: str) -> dict[str, Any]:
    spec = C.MODELS[label]
    record = D.manifest_record(label)
    if C.sha256_file(spec["checkpoint"]) != record["checkpoint_sha256"]:
        raise RuntimeError(f"{label} checkpoint sha changed on disk")
    return {
        "label": label,
        "model": spec["model"],
        "epoch": int(record["epoch"]),
        "path": C.rel(spec["checkpoint"]),
        "sha256": record["checkpoint_sha256"],
        "valid_ap": float(record["valid_ap_recorded"]),
    }


def _checkpoint_payload_sha(label: str) -> dict[str, Any]:
    spec = C.MODELS[label]
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    return {
        "checkpoint_payload_epoch": int(payload.get("epoch", -1)),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "checkpoint_payload_keys": sorted(payload.keys()),
    }


def run(*, models: tuple[str, ...] = C.MODEL_ORDER, out_dir: Path = C.LOCAL_VALID) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    task = D.load_task()
    valid_rows = np.asarray(task.valid_rows, dtype=np.int64)
    if valid_rows.size != C.N_VALID_ROWS:
        raise RuntimeError(f"valid rows {valid_rows.size}")

    # Official Valid labels come from the frozen artifact (Test rows are NaN).
    y_valid = np.asarray(task.labels[valid_rows], dtype=np.float32)
    valid_rows_sha = C.sha256_int64(valid_rows)

    records: dict[str, Any] = {}
    if ROWS_PATH.is_file():
        prev = np.load(ROWS_PATH)
        if not np.array_equal(prev, valid_rows):
            raise RuntimeError("existing valid_row_ids.npy is not aligned to the official split")
    tmp = ROWS_PATH.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, valid_rows)
    tmp.replace(ROWS_PATH)

    # S3 needs the frozen Train+Valid S3 path cache; the cached task is reused.
    s3_task = None
    s3_struct = None
    struct = None
    rc_dist = None

    for label in models:
        spec = C.MODELS[label]
        entry = _entry(label)
        t0 = time.perf_counter()
        if label == "S3":
            if s3_task is None:
                import sp_common as SPC

                s3_task = SPC.load_task_gated()
                s3_struct = s3_task.s3
            model, payload, audit, sha = SPT.restore_model(entry, s3_task, device)
            predict = lambda m, rows: SPT.predict_rows(m, s3_task, s3_struct, rows, device, batch_size=256)
        else:
            if label in ("BOC_FULL", "BOC_BOND") and struct is None:
                import boc_structures as BS

                struct = BS.load_cache()
            if label == "RC_REAL" and rc_dist is None:
                import rc_dist as RD

                rc_dist = RD.load_cache()
            model, payload, audit, sha = BT.restore_model(entry, task, struct, device)
            model_key = spec["model"]
            predict = lambda m, rows: BT.predict_rows(  # noqa: E731
                m, model_key, task, struct, rows, device, batch_size=256, rc_dist=rc_dist
            )
        t_restore = time.perf_counter() - t0

        t1 = time.perf_counter()
        logits = np.ascontiguousarray(predict(model, valid_rows), dtype=np.float32)
        t_predict = time.perf_counter() - t1
        if logits.shape != (C.N_VALID_ROWS, C.N_TASKS):
            raise RuntimeError(f"{label} valid logits shape {logits.shape}")
        if not np.isfinite(logits).all():
            raise RuntimeError(f"{label} valid logits non-finite")

        block = D.ap_block(y_valid, logits)
        recorded = float(entry["valid_ap"])
        delta = abs(block["official_ap"] - recorded)
        if abs(block["official_ap"] - block["independent_ap"]) > C.AP_AGREEMENT_TOLERANCE:
            raise RuntimeError(f"{label} official/independent AP disagree: {block['abs_delta']}")
        if delta > C.VALID_REPLAY_TOLERANCE:
            raise RuntimeError(f"{label} valid replay {block['official_ap']} != recorded {recorded} (d={delta})")

        # Per-task replay against the historically recorded per-task AP (if any).
        hist_per_task = D.manifest_record(label).get("valid_per_task_ap")
        per_task_deltas: list[float] = []
        main_drift: list[float] = []
        if hist_per_task is not None:
            for t, (mine, hist, npos) in enumerate(
                zip(block["per_task_ap"], hist_per_task, block["positive"])
            ):
                if mine is None or hist is None:
                    if (mine is None) != (hist is None):
                        raise RuntimeError(f"{label} task {t}: scored/None mismatch vs history")
                    continue
                d_task = abs(float(mine) - float(hist))
                per_task_deltas.append(d_task)
                if int(npos) >= C.HIGH_SUPPORT_POS:
                    main_drift.append(d_task)
        max_task_delta = max(per_task_deltas) if per_task_deltas else None
        n_task_drift = sum(1 for d in per_task_deltas if d > C.VALID_REPLAY_TOLERANCE)
        # Hard gate on tasks that carry the main diagnostic (>=20 positives);
        # CPU-vs-GPU float replay can reorder near-ties in tiny tasks.
        if main_drift and max(main_drift) > C.PER_TASK_MAIN_TOLERANCE:
            raise RuntimeError(f"{label} per-task valid AP drifted on a high-support task by {max(main_drift)}")

        path = out_dir / f"{label}.npy"
        tmp = path.with_suffix(".npy.tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, logits)
        tmp.replace(path)
        sha_file = C.sha256_file(path)

        records[label] = {
            "label": label,
            "model": spec["model"],
            "display": spec["display"],
            "role": spec["role"],
            "round": spec["round"],
            "epoch": int(entry["epoch"]),
            "checkpoint": entry["path"],
            "checkpoint_sha256": entry["sha256"],
            "path": C.rel(path),
            "sha256": sha_file,
            "array_sha256_float32": C.array_sha256(logits),
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "rows_sha256_int64": valid_rows_sha,
            "valid_ap_recorded": recorded,
            "valid_ap_replay": float(block["official_ap"]),
            "valid_ap_replay_independent": float(block["independent_ap"]),
            "valid_ap_abs_delta": float(delta),
            "valid_per_task_ap": block["per_task_ap"],
            "valid_observed": block["observed"],
            "valid_positive": block["positive"],
            "valid_negative": block["negative"],
            "n_tasks_scored_valid": int(block["n_tasks_scored"]),
            "max_abs_per_task_delta_vs_history": (None if max_task_delta is None else float(max_task_delta)),
            "n_tasks_per_task_delta_gt_1e4": int(n_task_drift),
            "max_abs_per_task_delta_high_support": (None if not main_drift else float(max(main_drift))),
            "prediction_protocol": "model.eval() + torch.inference_mode(); batch=256; CPU; same frozen state_dict",
            "restore_seconds": float(t_restore),
            "predict_seconds": float(t_predict),
            "parameters": int(audit.get("total_parameters", audit.get("parameters", -1))),
            "device": "cpu",
            **_checkpoint_payload_sha(label),
        }
        print(
            f"[valid] {label:9s} valid AP {block['official_ap']:.9f} (recorded {recorded:.9f}, "
            f"d={delta:.2e}) {t_predict:.1f}s sha {sha_file[:12]}"
        )
        del model
        if payload is not None:
            del payload

    manifest = {
        "schema": "lcrc-pcba-bottleneck-diagnosis-valid-predictions-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "source": "single read-only Validation forward of each frozen best-Validation checkpoint",
        "device": "cpu",
        "batch_size": 256,
        "gradients": "disabled (torch.inference_mode); no backward/optimizer.step",
        "rows_path": C.rel(ROWS_PATH),
        "rows_sha256_int64": valid_rows_sha,
        "rows_file_sha256": C.sha256_file(ROWS_PATH),
        "valid_label_block_sha256": C.array_sha256(y_valid),
        "valid_labels_source": "frozen Train/Valid label artifact (Test rows NaN)",
        "models": records,
        "created_utc": C.utcnow(),
    }
    C.write_json(MANIFEST_PATH, manifest)
    print(f"[valid] manifest -> {C.rel(MANIFEST_PATH)}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--models", default=",".join(C.MODEL_ORDER))
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(models=tuple(m.strip() for m in args.models.split(",") if m.strip()))
        return 0
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
