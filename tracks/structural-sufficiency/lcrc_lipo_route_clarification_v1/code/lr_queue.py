"""Persistent run queue and budget ledger.

The matrix is written once (before any formal result) as
``configs/run_matrix.csv`` with *every* arm of the pre-registered upper bound and
an explicit ``status``.  Conditional arms start as ``NOT_PURCHASED`` and are only
switched to ``PENDING`` by the frozen decision rules of ``PROTOCOL.md`` §9/§10.

``budget_ledger.json`` is the accounting record: wall clock, GPU hours, per-run
state, failure reasons and the not-purchased reasons.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import lr_common as C
import lr_data as D

MATRIX_CSV = C.CONFIGS / "run_matrix.csv"
QUEUE_JSON = C.ANALYSIS / "queue.json"
LEDGER_JSON = C.ANALYSIS / "budget_ledger.json"
SMOKE_JSON = C.ANALYSIS / "smoke.json"

PRIORITY = {  # lower number = bought first (PROTOCOL.md §14)
    "s0_core": 1,
    "s0_extra": 2,
    "s0_ctl": 3,
    "split_core": 4,
    "split_rest": 5,
    "split_ctl": 5,
    "sg_core": 6,
    "ft_core": 7,
    "sg_rest": 8,
    "split_R": 9,
    "lc": 10,
    "rep": 11,
}


def _row(split: str, model: str, loss: str, seed: int, stage: str, priority: int,
         status: str = "PENDING", gate: str = "-", note: str = "") -> dict[str, Any]:
    rid = C.run_id(split, model, loss, seed, stage if stage not in ("main",) else None)
    return {"run_id": rid, "split": split, "model": model, "loss": loss, "seed": int(seed),
            "stage": stage, "priority": priority, "status": status, "gate": gate, "note": note}


def build_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # --- S0 main matrix (33) ------------------------------------------
    for model in ("B", "S", "O", "D"):
        for loss in ("A", "R"):
            for seed in C.SEEDS:
                rows.append(_row("s0", model, loss, seed, "main", PRIORITY["s0_core"]))
    for loss in ("A", "R"):
        for seed in C.SEEDS:
            rows.append(_row("s0", "F", loss, seed, "main", PRIORITY["s0_extra"]))
    for seed in C.SEEDS:
        rows.append(_row("s0", "G", "A", seed, "main", PRIORITY["s0_extra"]))
    # --- S0 weighted / null controls (conditional, 6) ------------------
    for loss in ("W", "N"):
        for seed in C.SEEDS:
            rows.append(_row("s0", "C_loss", loss, seed, "main", PRIORITY["s0_ctl"],
                             status="NOT_PURCHASED", gate="c_loss",
                             note="purchased only if the S0 Diagnose purchase gate fires"))
    # --- S1/S2 = 36 core + 30 R + 12 controls -------------------------
    for split in ("s1", "s2"):
        for model in ("B", "S", "O", "D"):
            for seed in C.SEEDS:
                rows.append(_row(split, model, "A", seed, "main", PRIORITY["split_core"]))
        for model in ("F", "G"):
            for seed in C.SEEDS:
                rows.append(_row(split, model, "A", seed, "main", PRIORITY["split_rest"]))
        for model in ("B", "S", "O", "D", "F"):
            for seed in C.SEEDS:
                rows.append(_row(split, model, "R", seed, "main", PRIORITY["split_R"],
                                 status="NOT_PURCHASED", gate="c_loss",
                                 note="purchased only if a loss-control candidate exists"))
        for loss in ("W", "N"):
            for seed in C.SEEDS:
                rows.append(_row(split, "C_loss", loss, seed, "main", PRIORITY["split_ctl"],
                                 status="NOT_PURCHASED", gate="c_loss",
                                 note="only the C_loss arm of the S1/S2 R matrix"))
    # --- SG pressure test (<=21) --------------------------------------
    for model in ("B", "S", "O", "D"):
        for seed in C.SEEDS:
            rows.append(_row("sg", model, "A", seed, "sg", PRIORITY["sg_core"]))
    for model in ("F", "G"):
        for seed in C.SEEDS:
            rows.append(_row("sg", model, "A", seed, "sg", PRIORITY["sg_rest"]))
    for seed in C.SEEDS:
        rows.append(_row("sg", "C_deploy", "R", seed, "sg", PRIORITY["sg_rest"],
                         status="NOT_PURCHASED", gate="c_deploy_R",
                         note="only if C_deploy is an R arm"))
    # --- full Train development stage (<=12 + F-A) --------------------
    for model in ("B", "C_deploy", "D", "G"):
        for seed in C.SEEDS:
            rows.append(_row("ft", model, "A", seed, "ft", PRIORITY["ft_core"]))
    for seed in C.SEEDS:
        rows.append(_row("ft", "F", "A", seed, "ft", PRIORITY["ft_core"] + 1,
                         status="NOT_PURCHASED", gate="budget", note="light-weight full-Train reference"))
    # --- fixed supplementary branches (<=36 + 4) ----------------------
    for split in ("s1", "s2"):
        for model in ("B", "C_static", "D"):
            for seed in C.SEEDS:
                rows.append(_row(split, model, "A", seed, "half", PRIORITY["lc"],
                                 status="NOT_PURCHASED", gate="lc_trigger", note="learning curve, 1008 labels"))
        for model in ("B", "C_static", "D"):
            for seed in C.SEEDS:
                rows.append(_row(split, model, "A", seed, "threequarter", PRIORITY["lc"] + 1,
                                 status="NOT_PURCHASED", gate="lc_trigger", note="learning curve, 1512 labels"))
    for split in ("s1",):
        for model in ("B", "C_deploy"):
            rows.append(_row(split, model, "A", 0, "rep", PRIORITY["rep"],
                             status="NOT_PURCHASED", gate="budget", note="literal repeat, same seed"))
            rows.append(_row(split, model, "A", 0, "rep2", PRIORITY["rep"],
                             status="NOT_PURCHASED", gate="budget", note="literal repeat, same seed"))
    return rows


def write_matrix(rows: Sequence[Mapping[str, Any]]) -> Path:
    MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MATRIX_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "split", "model", "loss", "seed", "stage",
                                                "priority", "status", "gate", "note"])
        writer.writeheader()
        writer.writerows(rows)
    return MATRIX_CSV


def read_matrix() -> list[dict[str, Any]]:
    with open(MATRIX_CSV, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_matrix_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    with open(MATRIX_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "split", "model", "loss", "seed", "stage",
                                                "priority", "status", "gate", "note"])
        writer.writeheader()
        writer.writerows(rows)


def sync() -> dict[str, Any]:
    """Mark finished runs from their own ``train.json`` (never guess)."""
    rows = read_matrix()
    for row in rows:
        d = C.run_dir(row["run_id"])
        t = d / "train.json"
        if t.exists():
            payload = C.read_json(t)
            row["status"] = "COMPLETE"
            row["note"] = (row.get("note") or "") + f"|updates={payload['recipe']['updates']}" \
                if "updates=" not in (row.get("note") or "") else row["note"]
    write_matrix_rows(rows)
    C.write_json(QUEUE_JSON, {"schema": "lcrc-lipo-route-queue-v1", "synced_utc": C.utcnow(),
                              "rows": rows})
    return {"n": len(rows), "complete": sum(1 for r in rows if r["status"] == "COMPLETE")}


def estimate_seconds(kind: str) -> float:
    """Measured end-to-end time of a run class (x1.25 margin applied by callers)."""
    if not SMOKE_JSON.exists():
        return {"main": 600.0, "sg": 600.0, "ft": 900.0, "half": 600.0,
                "threequarter": 600.0, "rep": 600.0}.get(kind, 600.0)
    payload = C.read_json(SMOKE_JSON)
    return float(payload.get("projection", {}).get(kind, 600.0))


def projection(margin: float = 1.25) -> dict[str, Any]:
    rows = read_matrix()
    per_kind = {k: estimate_seconds(k) * margin / 3600.0 for k in
                ("main", "sg", "ft", "half", "threequarter", "rep")}
    by_priority: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_priority.setdefault(int(row["priority"]), []).append(row)
    cum = 0.0
    table = []
    for priority in sorted(by_priority):
        group = [r for r in by_priority[priority] if r["status"] in ("PENDING", "RUNNING")]
        hours = sum(per_kind[r["stage"]] for r in group)
        cum += hours
        table.append({"priority": priority, "n_runs": len(group), "gpu_hours": round(hours, 2),
                      "cumulative_gpu_hours": round(cum, 2),
                      "gpu_hours_two_gpus": round(cum / 2.0, 2)})
    return {"margin": margin, "per_kind_hours": {k: round(v, 3) for k, v in per_kind.items()},
            "table": table, "total_gpu_hours": round(cum, 2),
            "wall_hours_two_gpus": round(cum / 2.0, 2)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="run queue / budget ledger")
    ap.add_argument("command", choices=["init", "sync", "projection", "set", "budget"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--margin", type=float, default=1.25)
    args = ap.parse_args(argv)
    if args.command == "init":
        rows = build_matrix()
        write_matrix(rows)
        C.write_json(QUEUE_JSON, {"schema": "lcrc-lipo-route-queue-v1", "created_utc": C.utcnow(),
                                  "rows": rows})
        print(f"[queue] wrote {len(rows)} rows to {C.track_rel(MATRIX_CSV)}")
        print(json.dumps(projection(args.margin), indent=2))
        return 0
    if args.command == "sync":
        print(json.dumps(sync(), indent=2))
        return 0
    if args.command == "projection":
        print(json.dumps(projection(args.margin), indent=2))
        return 0
    if args.command == "set":
        rows = read_matrix()
        hit = [r for r in rows if r["run_id"] == args.run_id]
        if not hit:
            raise SystemExit(f"unknown run id {args.run_id!r}")
        hit[0]["status"] = args.status or hit[0]["status"]
        if args.note:
            hit[0]["note"] = args.note
        write_matrix_rows(rows)
        print(json.dumps(hit[0], indent=2))
        return 0
    if args.command == "budget":
        print(json.dumps(report_budget(), indent=2))
        return 0
    raise SystemExit(1)


def report_budget() -> dict[str, Any]:
    rows = read_matrix()
    gpu_hours = 0.0
    per_run = []
    for row in rows:
        t = C.run_dir(row["run_id"]) / "train.json"
        if not t.exists():
            continue
        payload = C.read_json(t)
        hours = float(payload["gpu"]["gpu_hours_charged"])
        gpu_hours += hours
        per_run.append({"run_id": row["run_id"], "status": payload.get("status"),
                        "gpu_hours": round(hours, 4), "updates": payload["recipe"]["updates"],
                        "split": row["split"], "model": row["model"], "loss": row["loss"],
                        "seed": row["seed"], "stage": row["stage"]})
    payload = {
        "schema": "lcrc-lipo-route-budget-ledger-v1", "created_utc": C.utcnow(),
        "protocol_id": C.ROUND_ID,
        "budget": {"wall_target_hours": C.WALL_TARGET_HOURS, "wall_hard_hours": C.WALL_HARD_HOURS,
                   "gpu_soft_cap_hours": C.GPU_SOFT_CAP_HOURS, "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS},
        "gpu_hours_used": round(gpu_hours, 4),
        "n_runs_completed": len(per_run),
        "runs": per_run,
        "counts_by_status": {s: sum(1 for r in rows if r["status"] == s)
                             for s in sorted({r["status"] for r in rows})},
        "not_purchased": [{"run_id": r["run_id"], "gate": r["gate"], "note": r["note"]}
                          for r in rows if r["status"] == "NOT_PURCHASED"],
    }
    C.write_json(LEDGER_JSON, payload)
    return {"gpu_hours_used": payload["gpu_hours_used"], "n_runs_completed": len(per_run),
            "counts_by_status": payload["counts_by_status"]}


if __name__ == "__main__":
    raise SystemExit(main())
