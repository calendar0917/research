"""FSF follow-up v1 checkpoint manifest (paths + SHA256 + selection record)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fsf_common as F


def _record(path: Path, *, kind: str, source: bool) -> dict[str, Any]:
    payload = F.load_checkpoint(path)
    return {
        "path": F.track_rel(path),
        "sha256": F.sha256_file(path),
        "bytes": int(path.stat().st_size),
        "kind": kind,
        "source_round_checkpoint": bool(source),
        "epoch": int(payload.get("epoch")),
        "global_step": int(payload.get("global_step")),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "horizon": payload.get("horizon"),
        "valid_ap": payload.get("valid_ap"),
        "parameters": payload.get("parameters"),
        "rank": (payload.get("model_audit") or {}).get("rank"),
        "git_commit": payload.get("git_commit"),
        "test_scored": payload.get("test_scored"),
        "test_used_for_selection": payload.get("test_used_for_selection"),
    }


def main() -> int:
    F.RESULTS.mkdir(parents=True, exist_ok=True)
    run_path = F.RUN_DIR / "run.json"
    run = F.read_json(run_path) if run_path.is_file() else {}
    source_manifest = F.load_source_manifest()
    checkpoints: dict[str, Any] = {}
    checkpoints["source_epoch12_last"] = _record(F.SOURCE_LAST, kind="source_resume_last", source=True)
    checkpoints["source_epoch12_best"] = _record(F.SOURCE_BEST, kind="source_epoch12_best", source=True)
    for kind in ("best", "last"):
        path = F.RUN_DIR / f"checkpoint_{kind}.pt"
        if path.is_file():
            checkpoints[f"followup_{kind}"] = _record(path, kind=f"followup_{kind}", source=False)
    payload = {
        "schema": F.SCHEMA_MANIFEST,
        "round_id": F.ROUND_ID,
        "created_utc": F.utcnow(),
        "git_commit": F.C.git_commit(),
        "environment": F.C.environment(),
        "round_dir": F.track_rel(F.ROUND),
        "source_round": "fixed-source-role-separation-pilot-v0",
        "source_manifest": F.track_rel(F.SOURCE_MANIFEST),
        "source_manifest_sha256": F.sha256_file(F.SOURCE_MANIFEST),
        "source_manifest_record": source_manifest["arms"]["RSSC"],
        "selection_metric": "full official Valid macro AP over the complete 1-25 prefix; tie -> earlier epoch",
        "tie_break": "earlier epoch",
        "run_status": run.get("status"),
        "run_best_epoch": run.get("best_epoch"),
        "run_best_valid_ap": run.get("best_valid_ap"),
        "epochs_run": run.get("epochs_run"),
        "gpu_hours": run.get("gpu_hours"),
        "checkpoints": checkpoints,
        "checkpoint_bytes_total": int(sum(v["bytes"] for v in checkpoints.values())),
        "retention": "source epoch-12 best/last are read-only; this round keeps its own best and last",
        "no_test_checkpoint": True,
        "test_scored": False,
    }
    F.write_json(F.RESULTS / "manifest.json", payload)
    F.log(f"manifest: {len(checkpoints)} checkpoints, {payload['checkpoint_bytes_total']/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
