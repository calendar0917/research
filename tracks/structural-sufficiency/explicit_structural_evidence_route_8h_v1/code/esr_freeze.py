"""Freeze the machine-readable experiment definition and the run schedule.

Writes ``configs/experiment.json``, ``configs/run_matrix.json`` and
``analysis/schedule.json`` and records their SHA-256 in
``analysis/protocol_freeze.json``.  Run **before** the first formal GPU job; the
content is a pure function of the frozen constants in ``esr_common`` plus the
Phase-B/C/D ordering of the round brief.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

import esr_common as C

VERSION = "esr-protocol-v1"


def experiment() -> dict[str, Any]:
    return {
        "round_id": C.ROUND_ID,
        "version": VERSION,
        "t0_utc": C.T0_UTC,
        "wall_hard_hours": C.WALL_HARD_HOURS,
        "wall_train_stop_hours": C.WALL_TRAIN_STOP_HOURS,
        "gpu_soft_hours": C.GPU_SOFT_HOURS,
        "gpu_hard_hours": C.GPU_HARD_HOURS,
        "seed": 0,
        "datasets": {
            name: {
                "task": meta["task"],
                "n_tasks": meta["n_tasks"],
                "primary_metric": meta["primary_metric"],
                "direction": meta["direction"],
                "split": meta["split"],
                "n_train": meta["n_train"],
                "n_valid": meta.get("n_valid", C.DATASETS[name].get("n_valid")),
                "scale": meta["scale"],
                "K": meta["K"],
                "epochs": meta["epochs"],
                "units": meta["units"],
            }
            for name, meta in C.DATASETS.items()
        },
        "arms": {
            arm: {
                "note": C.ARM_NOTES[arm],
                "explicit_evidence": arm in C.EXPLICIT_ARMS,
                "readout_deleted_from_base": arm != C.BASE,
            }
            for arm in C.ARMS + C.CONDITIONAL_ARMS
        },
        "aggregation": {
            "tau": C.TAU,
            "pi_init": list(C.PI_INIT),
            "count_scaling": "A(s) = sqrt(m) * R(s); ADD = sum(s)/sqrt(m)",
            "R": "pi_mean*Mean + pi_pos*LME - pi_neg*LME(-s)",
            "LME": "tau*log(mean(exp(s/tau)))",
            "empty_set": 0.0,
            "single_token": "R(s) = s",
            "numerics": "max/min-shifted segment log-sum-exp, float32",
        },
        "tokens": {
            "local": "u_i = LN(Linear(2K,K)(SiLU(Linear(d,2K)(h0_i))))",
            "relation": "v_ij = LN(Linear(2K,K)(SiLU(Linear(3K+19,2K)(x_ij))))",
            "relation_input": "x_ij = [u_i+u_j, u_i*u_j, |u_i-u_j|, r_ij]",
            "relation_features": "5 distance one-hots (SELF,1,2,3,4) + log1p(common neighbours) + 13 raw bond features",
            "relation_dim": C.REL_DIM,
            "pair_set": "all unordered root pairs with 1 <= shortest-path distance <= 2 (D=4 only for BEST_EXPLICIT_RANGE4)",
        },
        "global_term": {
            "form": "c_t(G) = w_t^T g(G) + b_t",
            "g": "frozen label-free global descriptors (standardised) + Train-standardised log1p(n_atoms), log1p(n_relation_pairs)",
            "forbidden_inputs": ["pooled hidden state", "attention output", "independent graph MLP"],
        },
        "recipes": {"small": C.RECIPE_SMALL, "pcba": {**C.RECIPE_PCBA, "adam_betas": list(C.RECIPE_PCBA["adam_betas"])}},
        "pcba_protocol": {
            "formal_horizon_epochs": 25,
            "full_horizon_epochs": 40,
            "schedule": "1-epoch linear warmup 3e-5 -> 3e-4 then per-step cosine to 1e-5 over the 40-epoch horizon (never compressed)",
            "order_plan": "ld.make_order_plan(official train rows, batch 128, seed 0, 40 epochs)",
            "selection": "best Validation AP within the common horizon; extension resumes from checkpoint_last",
        },
        "levels": C.LEVELS,
        "small_signal_thresholds": C.SMALL_SIGNAL,
        "param_matching": {
            "hidden_widths": ["K/2", "K", "2K"],
            "rule": "closest active total parameter count to BANK; ties keep the smaller H; <=5% total or mark APPROX_CAPACITY_CONTROL",
        },
        "history": {"pcba": C.PCBA_HISTORY, "small": C.SMALL_HISTORY},
        "test_policy": {
            "label": "EXPOSED_TEST_ENGINEERING_COMPARISON_SET",
            "rule": "one final scoring pass after every architecture/checkpoint/conditional decision is frozen; no architecture change afterwards; PCBA Test labels are NaN in the frozen artifact and cannot be scored",
        },
    }


def run_matrix() -> dict[str, Any]:
    small_units = []
    for dataset in (C.LIPO, C.TOX21):
        for arm in C.ARMS:
            small_units.append(
                {
                    "run_id": f"{dataset.lower()}-{arm.lower()}-s0",
                    "dataset": dataset,
                    "model": arm,
                    "seed": 0,
                    "stage": "B",
                    "requested_epochs": C.DATASETS[dataset]["epochs"],
                    "priority": 1,
                    "gpu_pool": "any",
                }
            )
    esol_units = [
        {"run_id": f"esol-{arm.lower()}-s0", "dataset": C.ESOL, "model": arm, "seed": 0, "stage": "B3",
         "requested_epochs": C.DATASETS[C.ESOL]["epochs"], "priority": 2, "gpu_pool": "any"}
        for arm in (C.BASE, C.HIDDEN)
    ]
    esol_units.append(
        {
            "run_id": "esol-best-explicit-lipo-s0",
            "dataset": C.ESOL,
            "model": "SELECT_BEST_EXPLICIT_LIPO",
            "seed": 0,
            "stage": "B3",
            "requested_epochs": C.DATASETS[C.ESOL]["epochs"],
            "priority": 2,
            "selection_rule": "best Validation primary metric among ADD/EVID/BANK/COMPOUND on LIPO",
        }
    )
    esol_units.append(
        {
            "run_id": "esol-best-explicit-tox21-s0",
            "dataset": C.ESOL,
            "model": "SELECT_BEST_EXPLICIT_TOX21",
            "seed": 0,
            "stage": "B3",
            "requested_epochs": C.DATASETS[C.ESOL]["epochs"],
            "priority": 2,
            "selection_rule": "best Validation primary metric among ADD/EVID/BANK/COMPOUND on TOX21",
            "note": "run only if it differs from the LIPO-selected arm",
        }
    )
    pcba_units = [
        {
            "run_id": f"pcba-{arm.lower()}-s0-e25",
            "dataset": C.PCBA,
            "model": arm,
            "seed": 0,
            "stage": "C1",
            "requested_epochs": 25,
            "priority": 1 if arm in (C.ADD, C.EVID) else 2,
            "gpu_pool": "res-gpu1" if arm in (C.ADD, C.EVID) else "res-gpu0",
        }
        for arm in (C.ADD, C.EVID, C.BANK, C.HIDDEN, C.COMPOUND)
    ]
    conditional = [
        {
            "run_id": "pcba-unary-matched-s0-e25",
            "dataset": C.PCBA,
            "model": C.UNARY,
            "stage": "D1",
            "trigger": "BEST_EXPLICIT improves >= 0.005 AP over the same-horizon W2 reference, or reaches Level 2 (gap <= 0.003 to FIXED1/DYNAMIC)",
            "requested_epochs": 25,
            "matches_best_explicit": True,
        },
        {
            "run_id": "pcba-best-explicit-s0-e40",
            "dataset": C.PCBA,
            "model": "BEST_EXPLICIT",
            "stage": "D2",
            "trigger": "BEST_EXPLICIT reaches Level 1 and the Validation curve still improves",
            "requested_epochs": 40,
            "resume": True,
        },
        {
            "run_id": "pcba-paired-control-s0-e40",
            "dataset": C.PCBA,
            "model": "EXTENSION_PAIR_OF_BEST_EXPLICIT",
            "stage": "D2",
            "trigger": "same as D2; pairs EVID->ADD, BANK->HIDDEN, COMPOUND->BANK, ADD->HIDDEN",
            "requested_epochs": 40,
            "resume": True,
        },
        {
            "run_id": "pcba-bank-flex-s0-e25",
            "dataset": C.PCBA,
            "model": C.FLEX,
            "stage": "D3",
            "trigger": "HIDDEN - BEST_EXPLICIT >= 0.005 AP, HIDDEN reaches Level 1, diagnostics show no implementation or input confound",
            "requested_epochs": 25,
        },
        {
            "run_id": "pcba-best-explicit-blind-s0-e25",
            "dataset": C.PCBA,
            "model": "BEST_EXPLICIT",
            "stage": "D4",
            "trigger": "BEST_EXPLICIT Level 1+; D1 shows a positive real-relation gain; full training budget remains",
            "requested_epochs": 25,
            "relation_mode": "blind",
        },
        {
            "run_id": "pcba-best-explicit-range4-s0-e25",
            "dataset": C.PCBA,
            "model": "BEST_EXPLICIT",
            "stage": "D5",
            "trigger": "BEST_EXPLICIT Level 2/3; D1 supports relations; relation stream not collapsed; D2 done; independent budget",
            "requested_epochs": 25,
            "max_dist": 4,
        },
    ]
    return {
        "round_id": C.ROUND_ID,
        "version": VERSION,
        "small": small_units,
        "esol": esol_units,
        "pcba": pcba_units,
        "conditional": conditional,
        "priority_order": ["B", "C1", "B3", "C1-compound", "D1", "D2", "D3", "D4", "D5", "mechanism-extension"],
        "degrade_order": [
            "cancel D5",
            "cancel D4",
            "drop non-core ESOL candidates",
            "skip untriggered D branches",
            "cancel PCBA COMPOUND",
            "drop every PCBA core arm to 20 epochs (only before the first formal PCBA step)",
        ],
    }


def schedule() -> dict[str, Any]:
    return {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "gpus": {
            "res-gpu0": [
                "LIPO 7 arms (B1)",
                "TOX21 7 arms (B2)",
                "ESOL BASE/HIDDEN/candidates (B3)",
                "PCBA BANK, HIDDEN, COMPOUND (C1)",
                "conditional D1/D3/D4/D5",
            ],
            "res-gpu1": [
                "PCBA SREN-ADD, SREN-EVID (C1, first)",
                "PCBA COMPOUND if gpu0 is busy",
                "D2 paired 40-epoch extensions",
                "mechanism/decomposition inference",
            ],
        },
        "launch_order": [
            "CPU: pair caches (all datasets) + Phase-A checks",
            "GPU preflight on both cards (timing + memory)",
            "res-gpu1: pcba SREN-ADD (25 epochs), then pcba SREN-EVID",
            "res-gpu0: LIPO BASE, UNARY, ADD, EVID, BANK, COMPOUND, HIDDEN; then TOX21 same order; ESOL",
            "res-gpu0: PCBA BANK, HIDDEN, COMPOUND",
            "conditional branches by the frozen triggers",
        ],
        "stop_rules": {
            "no_new_formal_training_after_hours": C.WALL_TRAIN_STOP_HOURS,
            "hard_deadline_hours": C.WALL_HARD_HOURS,
            "gpu_soft_hours": C.GPU_SOFT_HOURS,
            "gpu_hard_hours": C.GPU_HARD_HOURS,
            "min_remaining_for_launch": "1.15 * projected job time + 1.0 h analysis reserve",
        },
    }


def sha256_json(payload: Any) -> str:
    import json

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="freeze the ESR protocol")
    ap.add_argument("--out-dir", default=str(C.CONFIGS))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    exp = experiment()
    matrix = run_matrix()
    sched = schedule()
    C.write_json(out / "experiment.json", exp)
    C.write_json(out / "run_matrix.json", matrix)
    C.write_json(C.ANALYSIS / "schedule.json", sched)
    freeze = {
        "round_id": C.ROUND_ID,
        "version": VERSION,
        "frozen_utc": C.utcnow(),
        "files": {
            "configs/experiment.json": {"sha256": sha256_json(exp), "bytes": (out / "experiment.json").stat().st_size},
            "configs/run_matrix.json": {"sha256": sha256_json(matrix), "bytes": (out / "run_matrix.json").stat().st_size},
            "analysis/schedule.json": {"sha256": sha256_json(sched), "bytes": (C.ANALYSIS / "schedule.json").stat().st_size},
        },
        "code_sha256": {
            p.name: C.sha256_file(p)
            for p in sorted((C.CODE_DIR).glob("*.py"))
        },
        "git": C.git_commit(),
        "note": "frozen before the first formal GPU step; see PROTOCOL.md",
    }
    C.write_json(C.ANALYSIS / "protocol_freeze.json", freeze)
    C.log(f"frozen protocol {VERSION} at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
