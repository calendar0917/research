"""CLI for the LCRC Lipophilicity v4 local-chemistry pilot."""

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

import v4_common as C
import v4_atomchem as A
import v4_audit as AUD
import v4_descriptive as DESC
import v4_evaluate as EV
import v4_models as M
import v4_registry as R


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


def require_gpu_authorization(requested: str) -> dict[str, Any]:
    if not C.GPU_AUTHORIZATION.exists():
        raise PermissionError(
            "GPU_AUTHORIZATION_MISSING: freeze the protocol first, then record the researcher's "
            "authorization in analysis/gpu_authorization_v4.json"
        )
    auth = C.read_json(C.GPU_AUTHORIZATION)
    if auth.get("authorized") is not True:
        raise PermissionError("GPU_AUTHORIZATION_NOT_TRUE")
    if auth.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("GPU authorization protocol id mismatch")
    current_sha = C.freeze_sha256()
    if auth.get("protocol_freeze_sha256") != current_sha:
        raise PermissionError(
            f"GPU authorization is pinned to {auth.get('protocol_freeze_sha256')!r}, "
            f"but the current freeze SHA is {current_sha!r}"
        )
    max_hours = float(auth.get("max_gpu_hours", 0.0))
    if max_hours <= 0 or max_hours > C.GPU_HARD_CAP_HOURS + 1e-12:
        raise PermissionError(f"authorized GPU hours {max_hours} exceed the v4 cap {C.GPU_HARD_CAP_HOURS}")
    runs = list(auth.get("authorized_runs") or [])
    if requested not in runs:
        raise PermissionError(f"run {requested} is not covered by the GPU authorization")
    if str(auth.get("host")) != "res":
        raise PermissionError("v4 is pinned to host res for historical-regime comparability")
    if str(auth.get("pool")) not in ("res-gpu0", "res-gpu1"):
        raise PermissionError("v4 is pinned to a res A100 pool")
    if requested == R.conditional_key():
        gate = _p1_gate_from_disk()
        if not gate.get("met"):
            raise PermissionError(f"conditional broadcast run is not allowed: {gate}")
    return auth


def _p1_gate_from_disk() -> dict[str, Any]:
    p1_key = R.unconditional_keys()[0]
    hist = EV.load_historical()
    run = EV.load_run(p1_key)
    cmp = EV.compare(hist, run, label="P1_vs_historical")
    return EV.p1_gate(cmp)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_atomchem(_args) -> int:
    payload = AUD.audit_atomchem()
    C.log(
        f"[atomchem] atoms={payload['n_atoms']} logp_max={payload['crippen']['max_abs_sum_minus_MolLogP']:.2e} "
        f"tpsa_max={payload['tpsa']['max_abs_sum_minus_TPSA']:.2e} "
        f"flags_ok={payload['hbd_hba']['single_atom_only']}"
    )
    return 0


def cmd_descriptive(_args) -> int:
    payload = DESC.run_descriptive()
    C.log(
        f"[descriptive] dev_n={payload['scope']['dev_n']} charged_dev="
        f"{payload['ar3_dev_error_groups']['groups']['formal_charge']['charged']['n']} "
        f"labels_at_4p5={payload['all_fit']['label_at_4p5']}"
    )
    return 0


def cmd_fold_alignment(_args) -> int:
    payload = AUD.audit_fold_alignment()
    C.log(f"[fold-alignment] train={payload['n_train_molecules']} dev={payload['n_dev_molecules']} ok")
    return 0


def cmd_norm(_args) -> int:
    payload = AUD.audit_atom_norm()
    C.log(f"[atom-norm] mean={payload['standardisation']['mean']}")
    return 0


def cmd_equiv(_args) -> int:
    payload = AUD.audit_forward_equivalence()
    C.log(f"[equiv] arms={list(payload['arms'])} checks={payload['checks']}")
    return 0


def cmd_broadcast(_args) -> int:
    payload = AUD.audit_broadcast_control()
    C.log(f"[broadcast] checks={payload['checks']}")
    return 0


def cmd_denorm(_args) -> int:
    payload = AUD.audit_denorm()
    C.log(
        f"[denorm] hist_rmse={payload['historical']['rmse_from_stored']:.10f} "
        f"replay_max={payload['historical']['checkpoint_replay_max_abs_diff']:.2e}"
    )
    return 0


def cmd_frozen(_args) -> int:
    payload = AUD.audit_frozen_artifacts()
    C.log(f"[frozen] {payload['checks']}")
    return 0


def cmd_cpu_gates(_args) -> int:
    payload = AUD.run_all()
    C.log(f"[cpu-gates] final_status={payload['final_status']} failed={payload['failed']}")
    return 0 if payload["final_status"] == "READY_FOR_GPU_AUTHORIZATION" else 2


def cmd_freeze_extras(_args) -> int:
    payload = A.export_frozen_extras()
    C.log(f"[freeze-extras] npz_sha={payload['npz_sha256'][:12]} n_atom_extra={payload['n_atom_extra']}")
    return 0


def cmd_registry(_args) -> int:
    payload = R.planned_registry()
    C.write_json(C.RUN_REGISTRY, payload)
    C.log(
        f"[registry] runs={payload['n_runs']} unconditional={payload['n_unconditional']} "
        f"conditional={payload['n_conditional']}"
    )
    return 0


def cmd_budget(_args) -> int:
    payload = R.build_budget()
    C.log(
        f"[budget] estimated={payload['estimated_total_gpu_hours_all_runs']:.4f} h "
        f"cap={payload['hard_cap_gpu_hours']} fits={payload['fits_within_hard_cap']}"
    )
    return 0


def cmd_freeze_protocol(_args) -> int:
    gates_path = C.ANALYSIS / "cpu_gates_v4.json"
    if not gates_path.exists():
        raise SystemExit("run cpu-gates before freeze-protocol")
    gates = C.read_json(gates_path)
    if gates.get("final_status") != "READY_FOR_GPU_AUTHORIZATION":
        raise SystemExit(f"CPU gates are not ready: {gates.get('final_status')}")
    for required in (
        C.RUN_REGISTRY,
        C.ANALYSIS / "budget_v4.json",
        C.ANALYSIS / "frozen_extras_v4.npz",
        C.ANALYSIS / "frozen_extras_v4.json",
        C.ANALYSIS / "descriptive_v4.json",
        C.V3_MANIFEST,
    ):
        if not required.exists():
            raise SystemExit(f"missing freeze prerequisite {C.track_rel(required)}")
    cfg = C.load_config()
    freeze = C.build_protocol_freeze(cfg)
    freeze["run_registry"] = R.planned_registry()
    freeze["run_registry_file"] = C.file_record(C.RUN_REGISTRY)
    freeze["budget"] = C.file_record(C.ANALYSIS / "budget_v4.json")
    freeze["frozen_extras"] = {
        "json": C.file_record(C.ANALYSIS / "frozen_extras_v4.json"),
        "npz": C.file_record(C.ANALYSIS / "frozen_extras_v4.npz"),
    }
    freeze["cpu_gates"] = C.file_record(gates_path)
    freeze["tests"] = {
        "suite": "tracks/structural-sufficiency/lcrc_lipo_local_chem_pilot_v4/tests",
        "local_run": "uv run --group lipo pytest -q tracks/structural-sufficiency/lcrc_lipo_local_chem_pilot_v4/tests",
    }
    C.write_json(C.PROTOCOL_FREEZE, freeze)
    sha = C.freeze_sha256()
    (C.ANALYSIS / "freeze_sha256_v4.txt").write_text(sha + "\n", encoding="utf-8")
    C.log(f"[freeze-protocol] FREEZE_SHA256={sha}")
    return 0


def cmd_authorize(args) -> int:
    if C.GPU_AUTHORIZATION.exists():
        raise SystemExit("gpu_authorization_v4.json already exists; preserve it and use a new round for changes")
    C.verify_protocol_freeze(require_clean=False)
    gates = C.read_json(C.ANALYSIS / "cpu_gates_v4.json")
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
    if str(args.mode) not in ("freeze_reviewed", "pre_authorized_by_researcher_in_session"):
        raise SystemExit("unknown authorization mode")
    payload = {
        "authorized": True,
        "protocol_id": C.PROTOCOL_ID,
        "protocol_freeze_sha256": sha,
        "authorization_mode": str(args.mode),
        "researcher_reviewed_freeze_before_authorization": str(args.mode) == "freeze_reviewed",
        "max_gpu_hours": max_hours,
        "authorized_runs": [row["key"] for row in R.planned_registry()["runs"]],
        "host": "res",
        "pool": args.pool,
        "created_utc": C.utcnow(),
        "researcher_note": str(args.note),
        "budget_file": C.file_record(C.ANALYSIS / "budget_v4.json"),
        "cpu_gates": C.file_record(C.ANALYSIS / "cpu_gates_v4.json"),
        "conditional_rule": R.planned_registry()["conditional_rule"],
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
    out_path = _run_dir(args.key) / "train.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing run {C.track_rel(out_path)}")
    frozen = A.load_frozen_extras()
    extras = A.atom_arrays_for_variant(C.ARM_VARIANT[row["arm"]], frozen)
    job = M.LocalChemTrainJob(
        key=args.key,
        arm=row["arm"],
        fold=int(row["fold"]),
        recipe=row["recipe"],
        seed=int(row["seed"]),
        variant="scratch",
        device="cuda",
        out_root=C.RUNS,
        atom_extras=extras,
        spec_override=frozen["spec"],
        norm_override=None,
        v4_audit={
            "atomchem": {
                "names": list(C.ATOMCHEM_NAMES),
                "feature_variant": C.ARM_VARIANT[row["arm"]],
                "n_atom_extra": C.N_ATOM_EXTRA,
                "train_shape": list(extras["train"].shape),
                "dev_shape": list(extras["dev"].shape),
                "train_positions_sha256": C.row_sha256_bytes(extras["train_positions"]),
                "dev_positions_sha256": C.row_sha256_bytes(extras["dev_positions"]),
                "train_rows_sha256": C.row_sha256_bytes(extras["train_rows"]),
                "dev_rows_sha256": C.row_sha256_bytes(extras["dev_rows"]),
                "frozen_extras_npz_sha256": frozen["audit"]["npz_sha256"],
                "standardisation": frozen["audit"]["audit"]["standardisation"],
                "labels_used": False,
            },
            "authorization": {
                "protocol_freeze_sha256": auth.get("protocol_freeze_sha256"),
                "authorization_mode": auth.get("authorization_mode"),
                "max_gpu_hours": auth.get("max_gpu_hours"),
                "host": auth.get("host"),
                "pool": auth.get("pool"),
            },
        },
    )
    payload = M.train_local_chem_job(job)
    R.rebuild_ledger()
    C.log(
        f"[train] {args.key} dev={payload['metrics']['dev_rmse_raw']:.6f} "
        f"mae={payload['metrics']['dev_mae_raw']:.6f} wall={payload['wall_seconds']:.1f}s"
    )
    return 0


def cmd_analyze(_args) -> int:
    payload = EV.analyse()
    C.log(f"[analyze] status={payload['status']} verdict={payload['decision']['verdict']}")
    for row in payload["table"]:
        C.log(
            f"  {row['model']}: dev_rmse={row['dev_rmse']:.6f} dev_mae={row['dev_mae']:.6f} "
            f"delta={row['delta_rmse']:+.6f}"
        )
    return 0


def cmd_gate(_args) -> int:
    gate = _p1_gate_from_disk()
    C.log(json.dumps(gate, indent=2, default=str))
    return 0 if gate["met"] else 3


def cmd_ledger(_args) -> int:
    payload = R.rebuild_ledger()
    C.log(
        f"[ledger] {payload['status']} {payload['total_gpu_hours']:.4f} h used, "
        f"{payload['remaining_gpu_hours']:.4f} h remaining"
    )
    return 0


def cmd_verify(_args) -> int:
    frozen = C.verify_protocol_freeze(require_clean=False)
    v1 = C.assert_v1_unchanged()
    v2 = C.assert_v2_unchanged()
    v3 = C.assert_v3_unchanged(use_manifest=True)
    C.log(
        json.dumps(
            {
                "freeze_sha256": C.freeze_sha256(),
                "freeze_protocol_id": frozen["protocol_id"],
                "v1_ok": v1["ok"],
                "v2_ok": v2["ok"],
                "v3_ok": v3["ok"],
                "v2_checks": v2["n_checks"],
                "v3_checks": v3["n_checks"],
            },
            indent=2,
        )
    )
    return 0 if v1["ok"] and v2["ok"] and v3["ok"] else 2


def cmd_status(_args) -> int:
    gates = C.read_json(C.ANALYSIS / "cpu_gates_v4.json") if (C.ANALYSIS / "cpu_gates_v4.json").exists() else None
    auth = C.read_json(C.GPU_AUTHORIZATION) if C.GPU_AUTHORIZATION.exists() else None
    ledger = C.read_json(C.ANALYSIS / "gpu_ledger_v4.json") if (C.ANALYSIS / "gpu_ledger_v4.json").exists() else None
    runs = sorted(p.parent.name for p in C.RUNS.glob("*/train.json")) if C.RUNS.exists() else []
    out = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "freeze": C.file_record(C.PROTOCOL_FREEZE) if C.PROTOCOL_FREEZE.exists() else None,
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
        }
        if ledger
        else None,
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
        ("atomchem", cmd_atomchem),
        ("descriptive", cmd_descriptive),
        ("fold-alignment", cmd_fold_alignment),
        ("norm", cmd_norm),
        ("equiv", cmd_equiv),
        ("broadcast", cmd_broadcast),
        ("denorm", cmd_denorm),
        ("frozen", cmd_frozen),
        ("cpu-gates", cmd_cpu_gates),
        ("freeze-extras", cmd_freeze_extras),
        ("registry", cmd_registry),
        ("budget", cmd_budget),
        ("freeze-protocol", cmd_freeze_protocol),
        ("analyze", cmd_analyze),
        ("gate", cmd_gate),
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
    sp.add_argument("--mode", default="freeze_reviewed")
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
