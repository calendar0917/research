"""FSR v0 checkpoint manifest (paths + SHA256 + selection record)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import frs_common as FC  # noqa: E402


def main() -> int:
    FC.RESULTS.mkdir(parents=True, exist_ok=True)
    arms: dict[str, Any] = {}
    total_bytes = 0
    for arm in FC.ARMS:
        out_dir = FC.arm_out_dir(arm)
        run_path = out_dir / "run.json"
        if not run_path.is_file():
            arms[arm] = {"status": "MISSING", "run_dir": str(out_dir.relative_to(FC.ROUND))}
            continue
        run = FC.read_json(run_path)
        entry: dict[str, Any] = {
            "status": run.get("status"),
            "horizon": run.get("horizon"),
            "epochs_run": run.get("epochs_run"),
            "best_epoch": run.get("best_epoch"),
            "best_valid_ap": run.get("best_valid_ap"),
            "selection_metric": "full official Valid macro AP",
            "tie_break": "earlier epoch",
            "parameters": run.get("parameters"),
            "gpu_hours": run.get("gpu_hours"),
            "run_json": FC.C.track_rel(run_path),
            "run_json_sha256": FC.sha256_file(run_path),
            "checkpoints": {},
        }
        for kind in ("best", "last"):
            path = out_dir / f"checkpoint_{kind}.pt"
            if not path.is_file():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            entry["checkpoints"][kind] = {
                "path": str(path.relative_to(FC.ROUND)),
                "sha256": FC.sha256_file(path),
                "bytes": int(path.stat().st_size),
                "epoch": int(payload.get("epoch")),
                "global_step": int(payload.get("global_step")),
                "checkpoint_kind": payload.get("checkpoint_kind"),
                "valid_ap": payload.get("valid_ap"),
                "parameters": payload.get("parameters"),
                "git_commit": payload.get("git_commit"),
                "test_scored": payload.get("test_scored"),
            }
            total_bytes += int(path.stat().st_size)
        arms[arm] = entry
    payload = {
        "schema": FC.SCHEMA_MANIFEST,
        "round_id": FC.ROUND_ID,
        "created_utc": FC.utcnow(),
        "git_commit": FC.C.git_commit(),
        "environment": FC.C.environment(),
        "round_dir": str(FC.ROUND.relative_to(FC.REPO)),
        "arms": arms,
        "checkpoint_bytes_total": total_bytes,
        "retention": "best + last per arm kept in the round's runs/ directory (13 MB each)",
        "no_test_checkpoint": True,
    }
    FC.write_json(FC.RESULTS / "manifest.json", payload)
    FC.log(f"manifest: {len(arms)} arms, {total_bytes/1e6:.1f} MB of checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
