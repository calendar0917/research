"""CLI for the LCRC Lipophilicity v3 chemistry-feature pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import v3_common as C
import v3_budget as BUD
import v3_cpu_audit as AUD
import v3_descriptive as DESC
import v3_evaluate as EV
import v3_features as F
import v3_ledger as L
import v3_models as M
import v3_registry as R


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


def require_gpu_authorization(requested: str) -> dict[str, Any]:
    if not C.GPU_AUTHORIZATION.exists():
        raise PermissionError(
            "GPU_AUTHORIZATION_MISSING: the CPU phase must first report a freeze SHA; "
            "then the researcher must issue analysis/gpu_authorization_v3.json with that exact SHA."
        )
    auth = C.read_json(C.GPU_AUTHORIZATION)
    if auth.get("authorized") is not True:
        raise PermissionError("GPU_AUTHORIZATION_NOT_TRUE")
    if auth.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("GPU authorization protocol id mismatch")
    if not C.PROTOCOL_FREEZE.exists():
        raise PermissionError("GPU authorization exists but the protocol freeze is missing")
    current_sha = C.freeze_sha256()
    if auth.get("protocol_freeze_sha256") != current_sha:
        raise PermissionError(
            f"GPU authorization is pinned to {auth.get('protocol_freeze_sha256')!r}, "
            f"but the current freeze SHA is {current_sha!r}"
        )
    max_hours = float(auth.get("max_gpu_hours", 0.0))
    if max_hours <= 0 or max_hours > C.GPU_HARD_CAP_HOURS + 1e-12:
        raise PermissionError(f"authorized GPU hours {max_hours} exceed the v3 cap {C.GPU_HARD_CAP_HOURS}")
    runs = list(auth.get("authorized_runs") or [])
    if not runs:
        raise PermissionError("GPU authorization lists no runs")
    allowed = {row["key"] for row in R.planned_registry()["runs"]}
    if not set(runs) <= allowed:
        raise PermissionError(f"GPU authorization contains unknown runs: {sorted(set(runs) - allowed)}")
    if requested not in runs:
        raise PermissionError(f"run {requested} is not covered by the GPU authorization")
    if not str(auth.get("researcher_note", "")).strip():
        raise PermissionError("GPU authorization has no researcher note")
    if str(auth.get("host")) != "res":
        raise PermissionError("v3 pilot is pinned to host res for historical-regime comparability")
    if str(auth.get("pool")) not in ("res-gpu0", "res-gpu1"):
        raise PermissionError("v3 pilot is pinned to a res A100 pool (res-gpu0/res-gpu1)")
    if requested == R.conditional_key():
        gate = EV.conditional_condition_from_runs()
        if not gate.get("met"):
            raise PermissionError(f"conditional C_Chem12 run is not allowed: {gate}")
    return auth


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_features(_args) -> int:
    payload = AUD.audit_features()
    C.log(f"[features] chem12={payload['chem12']['shape']} chem13={payload['chem13']['shape']} "
          f"order_ok={payload['feature_order']['chem12_order_exact'] and payload['feature_order']['chem13_order_exact']}")
    return 0


def cmd_fold_alignment(_args) -> int:
    payload = AUD.audit_fold_alignment()
    C.log(f"[fold-alignment] train={payload['checks']['n_train']} dev={payload['checks']['n_dev']} ok")
    return 0


def cmd_norm(_args) -> int:
    payload = AUD.audit_norm()
    C.log(f"[norm] variants={list(payload['variants'])} train_only=True")
    return 0


def cmd_equiv(_args) -> int:
    payload = AUD.audit_forward_equivalence()
    C.log(f"[equiv] arms={list(payload['arms'])} zero_equiv={payload['checks']['zero_standardised_extra_forward_equivalent']}")
    return 0


def cmd_denorm(_args) -> int:
    payload = AUD.audit_denorm_and_rmse()
    C.log(f"[denorm] A={payload['comparators']['A']['rmse_from_stored_predictions']:.5f} "
          f"C={payload['comparators']['C']['rmse_from_stored_predictions']:.5f}")
    return 0


def cmd_descriptive(_args) -> int:
    payload = DESC.run_descriptive()
    C.log(f"[descriptive] groups={list(payload['comparators']['A']['groups'])}")
    return 0


def cmd_freeze_extras(_args) -> int:
    payload = F.export_frozen_extras()
    C.log(f"[freeze-extras] npz_sha={payload['npz_sha256'][:12]} variants={list(payload['variants'])}")
    return 0


def cmd_temp_train(_args) -> int:
    payload = AUD.audit_temporary_training()
    C.log(f"[temp-train] checks={payload['checks']}")
    return 0


def cmd_frozen(_args) -> int:
    payload = AUD.audit_frozen_artifacts()
    C.log(f"[frozen] v1={payload['checks']['v1_unchanged']} v2={payload['checks']['v2_unchanged']}")
    return 0


def cmd_cpu_gates(_args) -> int:
    payload = AUD.run_all()
    C.log(f"[cpu-gates] final_status={payload['final_status']} failed={payload['failed']}")
    return 0 if payload["final_status"] == "READY_FOR_GPU_AUTHORIZATION" else 2


def cmd_registry(_args) -> int:
    payload = R.planned_registry()
    C.write_json(C.RUN_REGISTRY, payload)
    C.log(f"[registry] wrote {C.track_rel(C.RUN_REGISTRY)} runs={payload['n_runs']} "
          f"unconditional={payload['n_unconditional']} conditional={payload['n_conditional']}")
    return 0


def cmd_budget(_args) -> int:
    payload = BUD.build_budget()
    C.log(f"[budget] estimated={payload['estimated_total_gpu_hours_all_three']:.4f} h "
          f"cap={payload['hard_cap_gpu_hours']} fits={payload['fits_within_hard_cap']}")
    return 0


def cmd_freeze_protocol(_args) -> int:
    gates_path = C.ANALYSIS / "cpu_gates_v3.json"
    if not gates_path.exists():
        raise SystemExit("run cpu-gates before freeze-protocol")
    gates = C.read_json(gates_path)
    if gates.get("final_status") != "READY_FOR_GPU_AUTHORIZATION":
        raise SystemExit(f"CPU gates are not ready: {gates.get('final_status')}")
    for required in (
        C.RUN_REGISTRY,
        C.ANALYSIS / "budget_v3.json",
        C.ANALYSIS / "frozen_extras_v3.npz",
        C.ANALYSIS / "frozen_extras_v3.json",
        C.ANALYSIS / "descriptive_groups_v3.json",
    ):
        if not required.exists():
            raise SystemExit(f"missing freeze prerequisite {C.track_rel(required)}")
    cfg = C.load_config()
    freeze = C.build_protocol_freeze(cfg)
    freeze["run_registry"] = R.planned_registry()
    freeze["run_registry_file"] = C.file_record(C.RUN_REGISTRY)
    freeze["budget"] = C.file_record(C.ANALYSIS / "budget_v3.json")
    freeze["frozen_extras"] = {
        "json": C.file_record(C.ANALYSIS / "frozen_extras_v3.json"),
        "npz": C.file_record(C.ANALYSIS / "frozen_extras_v3.npz"),
    }
    freeze["cpu_gates"] = C.file_record(gates_path)
    freeze["tests"] = {
        "suite": "tracks/structural-sufficiency/lcrc_lipo_chem_feature_pilot_v3/tests",
        "local_run": "uv run --group lipo pytest -q tracks/structural-sufficiency/lcrc_lipo_chem_feature_pilot_v3/tests",
    }
    C.write_json(C.PROTOCOL_FREEZE, freeze)
    sha = C.freeze_sha256()
    (C.ANALYSIS / "freeze_sha256_v3.txt").write_text(sha + "\n", encoding="utf-8")
    C.log(f"[freeze-protocol] FREEZE_SHA256={sha}")
    return 0


def cmd_authorize(args) -> int:
    if C.GPU_AUTHORIZATION.exists():
        raise SystemExit("gpu_authorization_v3.json already exists; preserve it and use a new round for changes")
    C.verify_protocol_freeze(require_clean=False)
    gates = C.read_json(C.ANALYSIS / "cpu_gates_v3.json")
    if gates.get("final_status") != "READY_FOR_GPU_AUTHORIZATION":
        raise SystemExit("CPU gates are not ready")
    sha = C.freeze_sha256()
    if args.confirm_freeze_sha != sha:
        raise SystemExit(f"--confirm-freeze-sha does not match the current freeze SHA {sha}")
    max_hours = float(args.max_hours)
    if max_hours <= 0 or max_hours > C.GPU_HARD_CAP_HOURS:
        raise SystemExit(f"max hours must be in (0, {C.GPU_HARD_CAP_HOURS}]")
    if not str(args.note).strip():
        raise SystemExit("--note must record the researcher's explicit authorization")
    payload = {
        "authorized": True,
        "protocol_id": C.PROTOCOL_ID,
        "protocol_freeze_sha256": sha,
        "max_gpu_hours": max_hours,
        "authorized_runs": [row["key"] for row in R.planned_registry()["runs"]],
        "host": "res",
        "pool": args.pool,
        "created_utc": C.utcnow(),
        "researcher_note": str(args.note),
        "budget_file": C.file_record(C.ANALYSIS / "budget_v3.json"),
        "cpu_gates": C.file_record(C.ANALYSIS / "cpu_gates_v3.json"),
    }
    C.write_json(C.GPU_AUTHORIZATION, payload)
    C.log(f"[authorize] wrote {C.track_rel(C.GPU_AUTHORIZATION)} freeze={sha[:12]} max={max_hours} h")
    return 0


def _run_dir(key: str) -> Path:
    return C.RUNS / key


def cmd_train(args) -> int:
    if args.device != "cuda":
        raise SystemExit("formal training is CUDA-only")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    auth = require_gpu_authorization(args.key)
    C.verify_protocol_freeze(require_clean=True)
    row = R.run_by_key(args.key)
    if row["kind"] != "train":
        raise SystemExit(f"{args.key} is not a supervised run")
    out_path = _run_dir(args.key) / "train.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing run {C.track_rel(out_path)}")
    fx = F.load_frozen_extras(row["feature_variant"])
    job = M.ChemTrainJob(
        key=args.key,
        arm=row["arm"],
        fold=int(row["fold"]),
        recipe=row["recipe"],
        seed=int(row["seed"]),
        variant="scratch",
        device="cuda",
        out_root=C.RUNS,
        global_extra_train=fx["train"],
        global_extra_dev=fx["dev"],
        global_extra_audit={
            **fx["audit"],
            "frozen_extras_npz_sha256": C.sha256_file(C.ANALYSIS / "frozen_extras_v3.npz"),
            "labels_used": False,
        },
        spec_override=fx["spec"],
        norm_override=fx["norm"],
        v3_audit={
            "frozen_extras": fx["audit"],
            "authorization": {
                "protocol_freeze_sha256": auth.get("protocol_freeze_sha256"),
                "max_gpu_hours": auth.get("max_gpu_hours"),
                "host": auth.get("host"),
                "pool": auth.get("pool"),
            },
        },
    )
    payload = M.train_chem_job(job)
    L.rebuild_ledger()
    C.log(
        f"[train] {args.key} dev={payload['metrics']['dev_rmse_raw']:.5f} "
        f"mae={payload['metrics']['dev_mae_raw']:.5f} wall={payload['wall_seconds']:.1f}s"
    )
    return 0


def cmd_analyze(_args) -> int:
    payload = EV.analyse()
    C.log(f"[analyze] status={payload['status']} verdict={payload['decision']['verdict']}")
    for row in payload["table"]:
        C.log(
            f"  {row['model']}: dev_rmse={row['dev_rmse']:.5f} dev_mae={row['dev_mae']:.5f} "
            f"delta={row['delta_rmse_vs_same_arch_original']:+.5f}"
        )
    return 0


def cmd_ledger(_args) -> int:
    payload = L.rebuild_ledger()
    C.log(f"[ledger] {payload['status']} {payload['total_gpu_hours']:.4f} h used, "
          f"{payload['remaining_gpu_hours']:.4f} h remaining")
    return 0


def cmd_verify(_args) -> int:
    frozen = C.verify_protocol_freeze(require_clean=False)
    v1 = C.assert_v1_unchanged()
    v2 = C.assert_v2_unchanged()
    C.log(json.dumps({
        "freeze_sha256": C.freeze_sha256(),
        "freeze_protocol_id": frozen["protocol_id"],
        "v1_ok": v1["ok"],
        "v2_ok": v2["ok"],
        "v2_checks": v2["n_checks"],
    }, indent=2))
    return 0 if v1["ok"] and v2["ok"] else 2


def cmd_status(_args) -> int:
    gates = C.read_json(C.ANALYSIS / "cpu_gates_v3.json") if (C.ANALYSIS / "cpu_gates_v3.json").exists() else None
    auth = C.read_json(C.GPU_AUTHORIZATION) if C.GPU_AUTHORIZATION.exists() else None
    ledger = C.read_json(C.ANALYSIS / "gpu_ledger_v3.json") if (C.ANALYSIS / "gpu_ledger_v3.json").exists() else None
    runs = sorted(p.parent.name for p in C.RUNS.glob("*/train.json")) if C.RUNS.exists() else []
    out = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "freeze": C.file_record(C.PROTOCOL_FREEZE),
        "freeze_sha256": C.freeze_sha256() if C.PROTOCOL_FREEZE.exists() else None,
        "cpu_gates": gates["final_status"] if gates else None,
        "gpu_authorization": bool(auth),
        "gpu_authorization_matches_current_freeze": bool(
            auth and C.PROTOCOL_FREEZE.exists() and auth.get("protocol_freeze_sha256") == C.freeze_sha256()
        ),
        "runs": runs,
        "ledger": {
            "status": ledger["status"],
            "total_gpu_hours": ledger["total_gpu_hours"],
            "remaining_gpu_hours": ledger["remaining_gpu_hours"],
        } if ledger else None,
        "next": "authorize GPU with the printed freeze SHA" if gates and gates.get("final_status") == "READY_FOR_GPU_AUTHORIZATION" and not auth else (
            "run A_Chem12 and A_Chem13" if auth else "run CPU gates and freeze"
        ),
    }
    C.log(json.dumps(out, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (
        ("features", cmd_features),
        ("fold-alignment", cmd_fold_alignment),
        ("norm", cmd_norm),
        ("equiv", cmd_equiv),
        ("denorm", cmd_denorm),
        ("descriptive", cmd_descriptive),
        ("freeze-extras", cmd_freeze_extras),
        ("temp-train", cmd_temp_train),
        ("frozen", cmd_frozen),
        ("cpu-gates", cmd_cpu_gates),
        ("registry", cmd_registry),
        ("budget", cmd_budget),
        ("freeze-protocol", cmd_freeze_protocol),
        ("analyze", cmd_analyze),
        ("ledger", cmd_ledger),
        ("verify", cmd_verify),
        ("status", cmd_status),
    ):
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("authorize")
    sp.add_argument("--confirm-freeze-sha", required=True)
    sp.add_argument("--note", required=True)
    sp.add_argument("--max-hours", type=float, default=C.GPU_HARD_CAP_HOURS)
    sp.add_argument("--pool", default="res-gpu0")
    sp.set_defaults(func=cmd_authorize)

    sp = sub.add_parser("train")
    sp.add_argument("--key", required=True)
    sp.add_argument("--device", default="cuda")
    sp.set_defaults(func=cmd_train)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
