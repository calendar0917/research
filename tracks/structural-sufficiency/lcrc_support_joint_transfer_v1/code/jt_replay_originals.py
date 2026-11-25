"""Re-evaluate the two reused PCBA originals with the round evaluator.

The W2 run never evaluated Test and the FIXED1 record was produced by the same
DSCR pipeline but on a different process/device.  For like-for-like deltas the
analysis needs both originals through one evaluator on one device, with logits
saved for molecule-level bootstrap.  Checkpoint selection is not changed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROUND = Path(__file__).resolve().parents[1]
CODE = ROUND / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import jt_common as C  # noqa: E402
import jt_models as M  # noqa: E402
import jt_support as S  # noqa: E402

S._prepare()
M.prepare_imports()

import bs_model as BM  # noqa: E402
import ds_common as DC  # noqa: E402
import ds_model as DM  # noqa: E402
import lr_data as LD  # noqa: E402


def _meta(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ap": float(report["ap"]),
        "bce": float(report["bce"]),
        "n_tasks_scored": int(report["n_tasks_scored"]),
        "per_task_ap": [None if x is None else float(x) for x in report["per_task_ap"]],
        "observed": [int(x) for x in report["observed"]],
        "positive": [int(x) for x in report["positive"]],
        "negative": [int(x) for x in report["negative"]],
        "logits_sha256": report["logits_sha256"],
        "seconds": float(report["seconds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", default="W2,FIXED1")
    parser.add_argument("--out-dir", default=str(ROUND / "analysis/replay_originals"))
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = LD.load_task()
    test_labels = np.asarray(DC.load_test_labels(task, cache_dir=ROUND / "runs/_shared")["labels"], dtype=np.float32)
    valid_y = np.asarray(task.labels[np.asarray(task.valid_rows, dtype=np.int64)], dtype=np.float32)
    test_y = test_labels

    for arm in [a for a in args.arms.split(",") if a]:
        if arm == "W2":
            model, build_audit = BM.build_model(
                "W2", task.spec, task.norm, int(C.PCBA_N_TASKS), seed=C.SEED, device="cpu", verify_parameters=True
            )
            checkpoint = C.W2_CHECKPOINT
            expected_valid = C.W2_EXPECTED["best_valid_ap"]
        elif arm == "FIXED1":
            model, build_audit = DM.build_ds_model(
                task.spec, task.norm, int(C.PCBA_N_TASKS), arm="FIXED1", seed=C.SEED, device="cpu"
            )
            checkpoint = C.FIXED1_CHECKPOINT
            expected_valid = C.FIXED1_EXPECTED["best_valid_ap"]
        else:
            raise KeyError(arm)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device).eval()
        reports = {}
        with torch.inference_mode():
            for split, y in (("valid", valid_y), ("test", test_y)):
                reports[split] = DC.evaluate_split(
                    model,
                    task,
                    device,
                    split,
                    y=y if split == "test" else None,
                    per_task=True,
                    keep_logits=True,
                    reason=f"replay_{arm}_{split}",
                )
        np.savez_compressed(
            out_dir / f"{arm}.npz",
            valid_logits=reports["valid"]["logits"].astype(np.float32),
            test_logits=reports["test"]["logits"].astype(np.float32),
            valid_y=valid_y,
            test_y=test_y,
            valid_per_task=np.asarray([np.nan if x is None else x for x in reports["valid"]["per_task_ap"]], dtype=np.float64),
            test_per_task=np.asarray([np.nan if x is None else x for x in reports["test"]["per_task_ap"]], dtype=np.float64),
        )
        meta = {
            "schema": "lcrc-support-joint-transfer-original-replay-v1",
            "round_id": C.ROUND_ID,
            "arm": arm,
            "device": str(device),
            "checkpoint": C.track_rel(checkpoint),
            "checkpoint_sha256": C.sha256_file(checkpoint),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "valid_ap": float(reports["valid"]["ap"]),
            "test_ap": float(reports["test"]["ap"]),
            "expected_valid_ap": float(expected_valid),
            "valid_abs_diff": abs(float(reports["valid"]["ap"]) - float(expected_valid)),
            "valid_bce": float(reports["valid"]["bce"]),
            "test_bce": float(reports["test"]["bce"]),
            "valid_per_task_ap": meta_per_task(reports["valid"]),
            "test_per_task_ap": meta_per_task(reports["test"]),
            "valid_positive": [int(x) for x in reports["valid"]["positive"]],
            "test_positive": [int(x) for x in reports["test"]["positive"]],
            "valid_observed": [int(x) for x in reports["valid"]["observed"]],
            "test_observed": [int(x) for x in reports["test"]["observed"]],
            "logits_sha256": {"valid": reports["valid"]["logits_sha256"], "test": reports["test"]["logits_sha256"]},
            "test_role": C.PCBA_TEST_ROLE,
            "test_used_for_selection": False,
            "environment": C.environment(),
        }
        C.write_json(out_dir / f"{arm}.json", meta)
        print(arm, "valid", meta["valid_ap"], "test", meta["test_ap"], "diff", meta["valid_abs_diff"])


def meta_per_task(report: dict[str, Any]) -> list[float | None]:
    return [None if x is None else float(x) for x in report["per_task_ap"]]


if __name__ == "__main__":
    main()
