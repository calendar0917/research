"""P0 entry: build/verify caches, real pairs, components and the frozen schedule.

Modes
-----
``--full`` (default, local): verify public bytes, enumerate all 4200 molecules,
write ``analysis/{data_audit.json,pairs.csv,pair_mappings.json,core_classes.json}``,
build the three label-free structure caches, and write the frozen schedule.

``--cache-only`` (remote): build/verify the three structure caches and re-verify
the committed pair/schedule hashes.  It writes only under ``runs/prepare/`` and
``cache/`` (both git-ignored), so the remote tracked checkout stays clean.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import common as C
import data as D
import pairs as P
import schedule as S

PROCESS_T0 = time.perf_counter()


def _source_summary() -> dict:
    source = C.verify_source()
    src = C.source_with_labels("none")
    rows = {s: [int(r) for r in src["rows"][s]] for s in C.SPLITS}
    return {
        "source": source,
        "splits": {s: {"n": len(rows[s]), "rows_sha256": C.row_sha256(rows[s])} for s in C.SPLITS},
    }


def full() -> int:
    C.log("[prepare] full P0 build")
    summary = _source_summary()
    built = P.build_all(write=True)
    audit = built["audit"]
    caches = D.build_all_caches()
    schedule = S.build_schedule(built["pairs"], write=True)
    payload = {
        "schema": "lcrc-lipo-edit-supervision-prepare-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "mode": "full",
        "source": summary["source"],
        "splits": summary["splits"],
        "data_audit_status": audit["status"],
        "gates": audit["gates"],
        "sets": {k: v for k, v in audit["sets"].items()},
        "caches": caches["splits"],
        "schedule": {
            "n_steps": schedule["manifest"]["n_steps"],
            "n_steps_redrawn": schedule["manifest"]["n_steps_redrawn"],
            "real_pair_indices_sha256": schedule["manifest"]["real_pair_indices"]["sha256"],
            "batch_order_sha256": schedule["manifest"]["batch_order"]["sha256"],
        },
        "environment": C.env_info(),
        "provenance": {"git_commit": C.git_commit(), "git_dirty_tracked": C.git_dirty()},
        "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
    }
    C.write_json(C.RUNS / "prepare" / "prepare.json", payload)
    C.log(f"[prepare] status={audit['status']} sets={ {k: v['n_pairs'] for k, v in audit['sets'].items()} }")
    return 0 if audit["status"] == "OK" else 2


def cache_only() -> int:
    C.log("[prepare] cache-only verification")
    summary = _source_summary()
    audit = C.read_json(C.ANALYSIS / "data_audit.json")
    manifest = C.read_json(C.ANALYSIS / "schedule_manifest.json")
    if audit.get("status") != "OK":
        C.log(f"[prepare] committed data audit status is {audit.get('status')!r}; refusing")
        return 2
    if summary["source"]["split_rows_sha256"] != audit["source"]["split_rows_sha256"]:
        raise C.DataMismatch("DATA_OR_SPLIT_MISMATCH: split row hashes differ from committed audit")
    pairs = P.load_pairs()
    p_train = P.p_train(pairs)
    if len(p_train) != int(audit["sets"]["P_train"]["n_pairs"]):
        raise C.DataMismatch("P_train count differs from committed audit")
    if C.sha256_text(C.stable_json([str(p["pair_id"]) for p in sorted(p_train, key=lambda p: str(p["pair_id"]))])) != manifest["p_train"]["pair_ids_sha256"]:
        raise C.DataMismatch("P_train pair-id hash differs from committed schedule manifest")
    canonical = P.canonical_smiles_map(summary_smiles := C.source_with_labels("none")["smiles"])
    caches = D.build_all_caches()
    schedule_rebuilt = S.ensure_schedule(p_train, manifest)
    schedule_check = S.verify_against(manifest, p_train, canonical)
    schedule_check["rebuilt_ignored_cache"] = bool(schedule_rebuilt)
    payload = {
        "schema": "lcrc-lipo-edit-supervision-prepare-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "mode": "cache-only",
        "source": summary["source"],
        "splits": summary["splits"],
        "data_audit_status": audit["status"],
        "p_train": {
            "n_pairs": len(p_train),
            "n_components": int(audit["sets"]["P_train"]["n_pair_bearing_components"]),
            "pair_ids_sha256": manifest["p_train"]["pair_ids_sha256"],
        },
        "caches": caches["splits"],
        "schedule": schedule_check,
        "environment": C.env_info(),
        "provenance": {"git_commit": C.git_commit(), "git_dirty_tracked": C.git_dirty()},
        "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
    }
    C.write_json(C.RUNS / "prepare" / "prepare.json", payload)
    C.log(f"[prepare] cache/schedule check pass={schedule_check['pass']}")
    return 0 if schedule_check["pass"] else 3


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P0 prepare / cache verification")
    ap.add_argument("--cache-only", action="store_true", help="remote: build caches, verify committed pairs+schedule")
    ap.add_argument("--full", action="store_true", help="local: full P0 build (default)")
    args = ap.parse_args(argv)
    if args.cache_only and args.full:
        raise SystemExit("choose --cache-only or --full")
    return cache_only() if args.cache_only else full()


if __name__ == "__main__":
    raise SystemExit(main())
