"""Frozen formal-task order (P0/P1/P2)."""

from __future__ import annotations

import argparse
from typing import Any

import bp_common as C


def build_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    # P1: molecular tasks first (short, high information).
    for ds in C.MOL_DATASETS:
        for seed in C.MOL_SEEDS:
            plan.append(C.plan_entry("mol", ds, seed))
    # P1: the four TU datasets that already existed in the repository.
    for ds in ("IMDB-BINARY", "IMDB-MULTI", "NCI1", "PROTEINS"):
        for fold in range(C.TU_N_SPLITS):
            plan.append(C.plan_entry("tu", ds, fold))
    # P2: Peptides-func.
    for seed in C.PEPTIDE_SEEDS:
        plan.append(C.plan_entry("peptide", C.PEPTIDE_DATASET, seed))
    # P2: remaining TU datasets.  Blocked datasets are skipped by the worker
    # when their analysis/<name>_blocked.json exists.
    for ds in ("MUTAG", "DD", "ENZYMES", "REDDIT-BINARY", "COLLAB"):
        for fold in range(C.TU_N_SPLITS):
            plan.append(C.plan_entry("tu", ds, fold))
    for i, entry in enumerate(plan):
        entry["order"] = int(i)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS / "formal_plan.json"))
    args = parser.parse_args()
    plan = build_plan()
    C.write_json(
        args.out,
        {
            "round_id": C.ROUND_ID,
            "result_label": C.RESULT_LABEL,
            "created_utc": C.utcnow(),
            "n_tasks": len(plan),
            "order": plan,
        },
    )
    print(C.stable_json({"n_tasks": len(plan), "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
