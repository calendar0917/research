"""Budget-aware sequential dispatcher for one GPU pool.

Each formal unit is an independent child process with its own identity, seed,
checkpoint, log and ledger entry, so an interrupted session can be resumed by
re-running the dispatcher: units whose ``run.json`` already exists are skipped.

Before every unit the dispatcher re-reads ``analysis/compute_ledger.json`` and
refuses to start a unit that would either break the GPU hard cap or eat into the
reporting reserve of the wall-clock cap.  Refused units are written to
``analysis/run_registry.json`` with ``NOT_RUN_BUDGET``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import ptb_common as C


def _projected_unit_hours(protocol: Mapping[str, Any]) -> dict[str, float]:
    cost = protocol["cost_projection"]
    mol = cost["molecule"]["per_dataset_model"]
    tu_rows = None
    for cand in cost["candidates"]:
        if cand["name"] == protocol["tu_experiment"]["scheme_name"]:
            tu_rows = cand
            break
    out: dict[str, float] = {}
    for key, row in mol.items():
        dataset, model = key.split(":")
        for seed in C.MOLECULE_SEEDS:
            out[C.mol_run_name(dataset, model, seed)] = float(row["unit_gpu_hours"]) * float(
                cost["safety_factor"]
            )
    if tu_rows is not None:
        per = _tu_per_unit(protocol)
        for name, hours in per.items():
            out[name] = hours
    return out


def _tu_per_unit(protocol: Mapping[str, Any]) -> dict[str, float]:
    """Per-unit TU cost from the measured projection stored in the frozen record."""
    payload = C.read_json(C.ANALYSIS / "cost_measurement.json")
    scheme = protocol["tu_experiment"]["scheme"]
    total = 0.0
    per: dict[str, float] = {}
    by_dataset: dict[str, float] = {}
    for row in payload["rows"]:
        if row["family"] != "tu":
            continue
        epoch_s = float(row["mean_epoch_seconds"])
        overhead = max(float(row["process_wall_seconds"]) - float(row["train_wall_seconds"]), 0.0)
        hours = (epoch_s * int(C.TU_EPOCHS) + overhead) / 3600.0 * float(C.COST_SAFETY_FACTOR)
        by_dataset[row["dataset"]] = max(by_dataset.get(row["dataset"], 0.0), hours)
    for row in payload["rows"]:
        if row["family"] != "tu":
            continue
        hours = by_dataset[row["dataset"]]
        for fold in range(int(scheme["n_splits"])):
            for seed in scheme["seeds"]:
                per[C.tu_run_name(row["dataset"], row["model"], fold, int(seed))] = hours
        total += hours
    return per


def build_plan(name: str, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    mol_units = C.mol_units()
    scheme = protocol["tu_experiment"]["scheme"]
    tu_units = C.tu_units(scheme)
    costs = _projected_unit_hours(protocol)

    def mol_entry(d: str, m: str, s: int) -> dict[str, Any]:
        unit = C.mol_run_name(d, m, s)
        return {
            "kind": "molecule",
            "unit": unit,
            "cmd": [
                sys.executable,
                str(C.CODE_DIR / "ptb_train_mol.py"),
                "--dataset",
                d,
                "--model",
                m,
                "--seed",
                str(s),
                "--out-dir",
                str(C.mol_run_dir(d, m, s)),
                "--device",
                "cuda",
            ],
            "result": str(C.mol_run_dir(d, m, s) / "run.json"),
            "projected_gpu_hours": float(costs.get(unit, 0.05)),
        }

    def tu_entry(d: str, m: str, f: int, s: int) -> dict[str, Any]:
        unit = C.tu_run_name(d, m, f, s)
        return {
            "kind": "tu",
            "unit": unit,
            "cmd": [
                sys.executable,
                str(C.CODE_DIR / "ptb_train_tu.py"),
                "--dataset",
                d,
                "--model",
                m,
                "--fold",
                str(f),
                "--seed",
                str(s),
                "--n-splits",
                str(int(scheme["n_splits"])),
                "--fold-seed",
                str(int(scheme["fold_seed"])),
                "--out-dir",
                str(C.tu_run_dir(d, m, f, s)),
                "--device",
                "cuda",
            ],
            "result": str(C.tu_run_dir(d, m, f, s) / "run.json"),
            "projected_gpu_hours": float(costs.get(unit, 0.1)),
        }

    def edit_entry(model: str, seed: int) -> dict[str, Any]:
        unit = f"edit_{model}_s{seed}"
        run_dir = C.EDIT / "runs" / f"edit_test_s{seed}_{model}"
        return {
            "kind": "local_edit",
            "unit": unit,
            "cmd": [
                sys.executable,
                str(C.EDIT / "code/edit_train.py"),
                "--model",
                model,
                "--seed",
                str(seed),
                "--device",
                "cuda",
            ],
            "result": str(run_dir / "train.json"),
            "projected_gpu_hours": 0.06,
        }

    edit_matrix = C.read_json(C.EDIT / "configs/run_matrix.json")
    edits = [edit_entry(u.split(":")[0], int(u.split(":s")[1])) for u in edit_matrix["allowed_units"]]

    mol = [mol_entry(d, m, s) for (d, m, s) in mol_units]
    tu = [tu_entry(d, m, f, s) for (d, m, f, s) in tu_units]
    # Longest projected units first keeps the two pools balanced.
    mol.sort(key=lambda e: (-e["projected_gpu_hours"], e["unit"]))
    tu.sort(key=lambda e: (-e["projected_gpu_hours"], e["unit"]))
    if name == "molecule":
        return mol
    if name == "tu":
        return tu
    if name == "mixed":
        return mol + tu
    if name == "local_edit":
        return edits
    if name == "local_edit_then_tu":
        return edits + tu
    if name == "tu_then_molecule":
        return tu + mol
    raise KeyError(name)


def _registry_add(entry: Mapping[str, Any]) -> None:
    path = C.ANALYSIS / "run_registry.json"
    payload = C.read_json(path) if path.exists() else {"schema": "lcrc-ptb-run-registry-v1", "units": []}
    payload["units"].append(dict(entry))
    payload["updated_utc"] = C.utcnow()
    C.write_json(path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, choices=["molecule", "tu", "mixed", "local_edit", "local_edit_then_tu", "tu_then_molecule"])
    ap.add_argument("--protocol", default=str(C.CONFIGS / "protocol.json"))
    ap.add_argument("--pool-label", default="unknown")
    ap.add_argument("--only-dataset", default=None, help="restrict to one dataset (unused in formal runs)")
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0, help="this queue handles shard K of --shards")
    ap.add_argument("--shards", type=int, default=1, help="split the plan across N queues by unit hash")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    protocol = C.read_json(args.protocol)
    plan = build_plan(args.plan, protocol)
    if args.only_dataset:
        plan = [e for e in plan if f"_{args.only_dataset}_" in e["unit"]]
    if args.shards and int(args.shards) > 1:
        # deterministic disjoint split so two pools never train the same unit
        def _shard_of(unit: str) -> int:
            return int(C.sha256_text(unit)[:8], 16) % int(args.shards)

        before = len(plan)
        plan = [e for e in plan if _shard_of(e["unit"]) == int(args.shard)]
        C.log(f"shard {args.shard}/{args.shards}: {len(plan)} of {before} planned units")
    if args.max_units:
        plan = plan[: int(args.max_units)]

    ledger = C.ledger_read()
    C.log(
        f"plan={args.plan} pool={args.pool_label} units={len(plan)} "
        f"ledger_gpu_h={ledger.get('total_gpu_hours', 0.0):.3f} "
        f"elapsed_h={C.elapsed_hours_since_start():.2f}"
    )
    started = time.perf_counter()
    completed = 0
    skipped = 0
    refused = 0
    for entry in plan:
        if Path(entry["result"]).exists():
            skipped += 1
            C.log(f"skip (already recorded): {entry['unit']}")
            continue
        spent = C.ledger_total()
        projected = float(entry["projected_gpu_hours"])
        elapsed = C.elapsed_hours_since_start()
        wall_left = C.WALL_CLOCK_CAP_HOURS - C.REPORTING_RESERVE_HOURS - elapsed
        if spent + projected > C.GPU_HARD_CAP_HOURS - 0.25:
            C.log(f"REFUSE {entry['unit']}: gpu hard cap guard ({spent:.2f}+{projected:.2f})")
            _registry_add({"unit": entry["unit"], "status": "NOT_RUN_BUDGET", "reason": "gpu_hard_cap", **{k: entry[k] for k in ("kind",)}})
            refused += 1
            continue
        if wall_left - projected <= 0.0:
            C.log(f"REFUSE {entry['unit']}: wall-clock reserve guard (left {wall_left:.2f} h)")
            _registry_add({"unit": entry["unit"], "status": "NOT_RUN_BUDGET", "reason": "wall_clock", **{k: entry[k] for k in ("kind",)}})
            refused += 1
            continue
        if args.dry_run:
            C.log(f"would run {entry['unit']} (~{projected:.3f} GPU-h)")
            continue
        C.log(f"START {entry['unit']} (~{projected:.3f} GPU-h, ledger {spent:.2f})")
        t0 = time.perf_counter()
        proc = subprocess.run(entry["cmd"], cwd=str(C.REPO), capture_output=False)
        wall = time.perf_counter() - t0
        status = "OK" if proc.returncode == 0 else "FAILED"
        C.ledger_add(
            {
                "unit": entry["unit"],
                "kind": entry["kind"],
                "pool": args.pool_label,
                "status": status,
                "returncode": int(proc.returncode),
                "wall_seconds": float(wall),
                "gpu_hours": float(wall / 3600.0),
                "projected_gpu_hours": projected,
                "started_utc": C.utcnow(),
            }
        )
        _registry_add(
            {
                "unit": entry["unit"],
                "kind": entry["kind"],
                "status": status,
                "returncode": int(proc.returncode),
                "pool": args.pool_label,
                "wall_seconds": float(wall),
                "gpu_hours": float(wall / 3600.0),
            }
        )
        if status == "OK":
            completed += 1
        else:
            C.log(f"FAILED {entry['unit']} rc={proc.returncode}; continuing with the remaining queue")
    C.log(
        f"queue done: completed={completed} skipped={skipped} refused={refused} "
        f"wall={(time.perf_counter() - started)/3600:.2f}h ledger={C.ledger_total():.3f} GPU-h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
