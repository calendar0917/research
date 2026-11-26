#!/usr/bin/env python
"""Freeze the SHA-256 of every D0 (and later C/P) checkpoint before any edit
evaluation code opens the strict pair labels.

Writing order is the guarantee: ``run_analysis`` refuses to score a checkpoint
whose SHA-256 differs from the frozen manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import sr_common as C

TARGETS = ("BASE", "ENDPOINT", "DELTA")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="B", choices=["B", "C", "P"])
    parser.add_argument("--out", default=str(C.CKPT_DIR / "manifest.json"))
    args = parser.parse_args()

    decision = C.read_json(C.ANALYSIS_DIR / "branch_decision.json")
    entries = []
    if args.phase == "B":
        rows = [decision["d0_rows"][k] for k in ("1", "2", "G")]
        base = C.RESULTS_DIR / "B"
        for row in rows:
            for target in TARGETS:
                for ckpt in ("best", "final"):
                    path = base / row / target / f"{ckpt}.pt"
                    if not path.is_file():
                        continue
                    summary_path = base / row / target / "summary.json"
                    summary = C.read_json(summary_path) if summary_path.is_file() else {}
                    entries.append({
                        "phase": "B", "row": row, "target": target, "checkpoint": ckpt,
                        "path": str(path.relative_to(C.REPO_ROOT)),
                        "sha256": C.sha256_file(path),
                        "best_epoch": summary.get("best_epoch"),
                        "dev_rmse": (summary.get("best_eval_rmse") if ckpt == "best"
                                     else summary.get("final_eval_rmse")),
                        "fit_rows_sha256": summary.get("fit_rows_sha256"),
                        "dev_rows_sha256": summary.get("dev_rows_sha256"),
                        "schedule_sha256": summary.get("schedule_sha256"),
                        "fit_scaler": summary.get("fit_scaler"),
                        "n_parameters": summary.get("n_parameters"),
                    })
    else:
        base = C.RESULTS_DIR / args.phase
        for row_dir in sorted(base.iterdir()) if base.is_dir() else []:
            if not row_dir.is_dir():
                continue
            for ckpt in ("best", "final"):
                path = row_dir / f"{ckpt}.pt"
                if not path.is_file():
                    continue
                summary_path = row_dir / "summary.json"
                summary = C.read_json(summary_path) if summary_path.is_file() else {}
                entries.append({
                    "phase": args.phase, "row": row_dir.name, "checkpoint": ckpt,
                    "path": str(path.relative_to(C.REPO_ROOT)),
                    "sha256": C.sha256_file(path),
                    "best_epoch": summary.get("best_epoch"),
                })

    payload = {
        "round_id": C.ROUND_ID, "phase": args.phase, "frozen_at": C.utc_stamp(),
        "n_entries": len(entries), "entries": entries,
        "branch": decision["branch"], "chosen_H": decision.get("chosen_H"),
        "d0_rows": decision["d0_rows"],
        "note": "written before any strict-edit evaluation; run_analysis verifies it",
    }
    C.write_json(args.out, payload)
    C.write_json(C.ANALYSIS_DIR / "checkpoint_hashes.json", entries)
    print(json.dumps({"phase": args.phase, "n_entries": len(entries)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
