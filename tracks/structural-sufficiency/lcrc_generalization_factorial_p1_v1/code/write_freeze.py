"""Write (or extend) the LCRC-P1 protocol freeze.

The freeze binds

* `PROTOCOL.md` (the pre-registered questions, units, recipe, metrics, statistics,
  budget and stop rules),
* the analysis/调度 code of this round (byte hashes), and
* the pre-run reuse decision (which units come from the frozen historical round
  and which must be trained).

It must be written **before** the first GPU job of the round.  Later code fixes
are appended as `revisions` entries; the file is never silently rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from p1_common import (
    ANALYSIS,
    ARMS,
    BOOTSTRAP_SEED,
    N_BOOT,
    PROTOCOL_FREEZE,
    Q2_ARMS,
    ROUND,
    SEEDS,
    SPLIT_ORDER,
    TOP_FRACTION,
    analysis_code_hashes,
    git_commit,
    read_json,
    sha256_file,
    track_rel,
    utcnow,
    write_json,
)


def _audit_summary() -> dict[str, Any]:
    manifest_path = ANALYSIS / "run_reuse_manifest.json"
    integrity_path = ANALYSIS / "integrity_checks.json"
    if not manifest_path.exists() or not integrity_path.exists():
        raise SystemExit("run `run_analysis.py --audit-only --no-freeze-check` first to produce the pre-run audit")
    manifest = read_json(manifest_path)
    integrity = read_json(integrity_path)
    return {
        "reused_units": [u["unit_key"] for u in manifest["units"] if u["origin"] == "reused" and u.get("available")],
        "fresh_units_required": [u["unit_key"] for u in manifest["missing_units"]],
        "n_units_available_at_freeze": int(sum(1 for u in manifest["units"] if u.get("available"))),
        "integrity_at_freeze": {
            "n_checks": integrity["n_checks"],
            "n_passed": integrity["n_passed"],
            "n_failed": integrity["n_failed"],
            "failed": [c["check"] for c in integrity["failed"]],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="write or extend the LCRC-P1 protocol freeze")
    parser.add_argument("--revision", default=None, help="reason for a post-freeze code revision")
    args = parser.parse_args(argv)

    protocol = ROUND / "PROTOCOL.md"
    if not protocol.exists():
        raise SystemExit("PROTOCOL.md is missing")

    payload: dict[str, Any] = {
        "round_id": "lcrc-generalization-factorial-p1-v1",
        "created_at": utcnow(),
        "git_commit": git_commit(),
        "protocol": {"path": track_rel(protocol), "sha256": sha256_file(protocol)},
        "frozen_constants": {
            "splits": list(SPLIT_ORDER),
            "seed_design": {k: list(v) for k, v in SEEDS.items()},
            "arms_random122": list(ARMS),
            "arms_q2": list(Q2_ARMS),
            "n_boot": N_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_percentiles": [2.5, 97.5],
            "tail_fraction": TOP_FRACTION,
            "metric_reproduction_tolerance": 1e-5,
        },
        "analysis_code_hashes": analysis_code_hashes(),
        "pre_run_audit": _audit_summary(),
        "gpu_budget_cap_a100_hours": 2.0,
        "test_policy": "DEVELOPMENT_TEST",
        "revisions": [],
    }

    if PROTOCOL_FREEZE.exists():
        existing = read_json(PROTOCOL_FREEZE)
        if not args.revision:
            print(json.dumps({"existing": track_rel(PROTOCOL_FREEZE)}, indent=1))
            raise SystemExit("a freeze already exists; pass --revision '<reason>' to append a revision")
        revisions = existing.get("revisions", [])
        revisions.append(
            {
                "at": utcnow(),
                "reason": args.revision,
                "previous_analysis_code_hashes": existing.get("analysis_code_hashes"),
                "previous_protocol_sha256": (existing.get("protocol") or {}).get("sha256"),
                "new_analysis_code_hashes": payload["analysis_code_hashes"],
                "new_protocol_sha256": payload["protocol"]["sha256"],
                "git_commit": payload["git_commit"],
            }
        )
        payload["revisions"] = revisions
        payload["created_at"] = existing.get("created_at", payload["created_at"])
        payload["pre_run_audit"] = existing.get("pre_run_audit", payload["pre_run_audit"])

    write_json(PROTOCOL_FREEZE, payload)
    print(json.dumps({"written": track_rel(PROTOCOL_FREEZE), "code_files": len(payload["analysis_code_hashes"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
