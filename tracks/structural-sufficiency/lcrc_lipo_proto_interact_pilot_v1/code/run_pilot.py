"""Unified CLI for the LCRC Lipophilicity prototype/interaction pilot round.

    run_pilot.py prep        -- fold-2 data audit, pair coverage, protocol freeze
    run_pilot.py gates       -- 12 CPU correctness gates (must pass before GPU)
    run_pilot.py smoke       -- 2-epoch smoke run of all five arms (CPU or CUDA)
    run_pilot.py train       -- one formal arm (BASE_G0/G1, PROTO, PAIR_TRUE/NULL)
    run_pilot.py summarize   -- five-model table, paired comparisons, decision
    run_pilot.py ledger      -- GPU budget ledger from analysis/rr_status/*.json
    run_pilot.py verify      -- re-verify checkpoints, predictions and metrics

The formal ``train`` command refuses to run unless the CPU gates passed, the
protocol freeze matches the current code/config, the GPU authorization file is
valid, and the requested arm matches its fixed pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

import pilot_common as C
import pilot_gates as G
import pilot_models as M
import pilot_pairs as P
import pilot_train as T


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def require_gates() -> dict[str, Any]:
    if not C.CPU_GATES.exists():
        raise PermissionError("CPU_GATES_MISSING: run `run_pilot.py gates` first")
    payload = C.read_json(C.CPU_GATES)
    if payload.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("cpu_gates.json has the wrong protocol id")
    if payload.get("ok") is not True:
        raise PermissionError("cpu_gates.json records a failing gate")
    if payload.get("config_hash") != C.config_hash(C.load_config()):
        raise PermissionError("CPU gates were produced with a different config")
    return payload


def require_freeze() -> dict[str, Any]:
    if not C.PROTOCOL_FREEZE.exists():
        raise PermissionError("PROTOCOL_FREEZE_MISSING: run `run_pilot.py prep` first")
    frozen = C.read_json(C.PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("protocol freeze has the wrong protocol id")
    if frozen.get("config_hash") != C.config_hash(C.load_config()):
        raise PermissionError("config changed after the protocol freeze")
    current = C.code_file_hashes()
    if current != frozen.get("code_hashes"):
        changed = sorted(k for k in set(current) | set(frozen.get("code_hashes", {}))
                         if current.get(k) != frozen.get("code_hashes", {}).get(k))
        raise PermissionError(f"code changed after the protocol freeze: {changed}")
    return frozen


def require_authorization(arm: str) -> dict[str, Any]:
    if not C.GPU_AUTHORIZATION.exists():
        raise PermissionError("GPU_AUTHORIZATION_MISSING")
    auth = C.read_json(C.GPU_AUTHORIZATION)
    if auth.get("authorized") is not True:
        raise PermissionError("GPU_AUTHORIZATION_NOT_TRUE")
    if auth.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("GPU authorization protocol id mismatch")
    frozen = require_freeze()
    if auth.get("protocol_freeze_sha256") != C.sha256_file(C.PROTOCOL_FREEZE):
        raise PermissionError("GPU authorization does not match the current protocol freeze")
    max_hours = float(auth.get("max_gpu_hours", 0.0))
    if max_hours <= 0 or max_hours > C.GPU_HARD_CAP_HOURS:
        raise PermissionError(f"authorized GPU hours {max_hours} outside (0, {C.GPU_HARD_CAP_HOURS}]")
    if arm not in list(auth.get("authorized_runs") or []):
        raise PermissionError(f"run {arm} is not covered by the GPU authorization")
    if not str(auth.get("researcher_note", "")).strip():
        raise PermissionError("GPU authorization has no researcher note")
    del frozen
    return auth


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> dict[str, Any]:
    cfg = C.load_config()
    view = C.load_fold_view()
    folds = C.folds_payload()
    fold_entry = [f for f in folds["folds"] if int(f["fold"]) == C.FOLD][0]
    fold_rows = [int(x) for x in fold_entry["rows_global"]]
    dev_rows = [int(x) for x in fold_entry["rows_global"]]
    audit = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "config_hash": C.config_hash(cfg),
        "data": {
            "source_round": "lcrc_lipo_mechanism_generalization_ssl_v2",
            "fold_file": C.track_rel(C.V2C.FOLD_FILE),
            "fold_file_sha256": C.sha256_file(C.V2C.FOLD_FILE),
            "fold": C.FOLD,
            "n_train": int(view.train_cache["n_molecules"]),
            "n_dev": int(view.dev_cache["n_molecules"]),
            "train_positions_sha256": view.audit()["train_positions_sha256"],
            "dev_positions_sha256": view.audit()["dev_positions_sha256"],
            "dev_rows_global": dev_rows,
            "dev_rows_global_sha256": C.V2C.row_sha256_bytes(dev_rows),
            "fold_rows_global_sha256": C.V2C.row_sha256_bytes(fold_rows),
            "target_norm": {
                "source": view.target_norm["source"],
                "mean": float(view.target_norm["mean"]),
                "scale": float(view.target_norm["scale"]),
                "n": int(view.target_norm["n"]),
                "label_vector_sha256": view.target_norm["label_vector_sha256"],
            },
            "feature_norm_source": view.train_norm["source"],
            "feature_norm_n": int(
                sum(np.diff(np.asarray(view.train_cache["atom_ptr"], dtype=np.int64)).tolist())
            ),
            "fit_cache_sha256": C.sha256_file(C.V2C.load_fit_cache()["path"]),
            "label_scope": {
                "fit2352_only": True,
                "monitor_labels_read": False,
                "reserved_labels_read": False,
                "official_valid_test_read": False,
                "random42_splits_read": False,
            },
            "protocol_spec": {
                "name": C.V2C.load_frozen_spec().name,
                "atom_feat_dim": C.V2C.load_frozen_spec().atom_feat_dim,
                "bond_feat_dim": C.V2C.load_frozen_spec().bond_feat_dim,
                "global_dim": C.V2C.load_frozen_spec().global_dim,
                "n_scalars": C.V2C.load_frozen_spec().n_scalars,
            },
        },
        "pairs": {},
        "models": {},
    }
    for name, cache in (("train", view.train_cache), ("dev", view.dev_cache)):
        true_pairs = P.build_pair_cache(cache, semantics="true")
        null_pairs = P.build_pair_cache(cache, semantics="null")
        audit["pairs"][name] = {
            "true": {
                "coverage": P.coverage(true_pairs),
                "n_pairs_total": int(true_pairs["n_pairs_total"]),
            },
            "null": {
                "n_pairs_total": int(null_pairs["n_pairs_total"]),
                "n_distinct_permutations_used": int(
                    len({tuple(r) for r in np.asarray(null_pairs["null_permutations"]).tolist()})
                ),
                "n_molecules_with_identity_permutation": int(
                    sum(1 for r in np.asarray(null_pairs["null_permutations"]).tolist() if r == [0, 1, 2])
                ),
            },
            "content_check": P.same_pair_content(true_pairs, null_pairs),
        }
    for arm in C.ARMS:
        model, _meta = M.build_arm(arm, C.V2C.load_frozen_spec(), view.train_norm, C.SEED, "cpu")
        audit["models"][arm] = {
            "parameters": M.parameter_count(model),
            "parameters_breakdown": M.module_breakdown(model),
            "init_state_hash": C.state_dict_fingerprint(model.state_dict()),
            "pool": C.ARM_POOL[arm],
        }
    C.write_json(C.DATA_AUDIT, audit)

    freeze = {
        "protocol_id": C.PROTOCOL_ID,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(cfg),
        "config_path": C.track_rel(C.CONFIG_PATH),
        "code_hashes": C.code_file_hashes(),
        "data_audit": C.file_record(C.DATA_AUDIT),
        "frozen_constants": {
            "seed": C.SEED,
            "fold": C.FOLD,
            "n_train": C.N_TRAIN,
            "n_dev": C.N_DEV,
            "recipe": {
                "optimizer": "AdamW",
                "peak_lr": C.PEAK_LR,
                "min_lr": C.MIN_LR,
                "scheduler": "per_step_cosine",
                "weight_decay": C.WEIGHT_DECAY,
                "batch_size": C.BATCH_SIZE,
                "epochs": C.EPOCHS,
                "grad_clip": C.GRAD_CLIP,
                "dropout_at_h": C.DROPOUT,
                "soup_last": C.SOUP_LAST,
                "shuffle_offset": C.SHUFFLE_OFFSET,
            },
            "proto": {
                "name": C.PROTO_NAME,
                "n_prototypes": C.N_PROTOTYPES,
                "topk": C.PROTO_TOPK,
                "tau": C.PROTO_TAU,
            },
            "pair": {
                "distances": list(C.PAIR_DISTANCES),
                "z_dim": C.PAIR_Z_DIM,
                "hidden": C.PAIR_HIDDEN,
                "null_salt": C.PAIR_NULL_SALT,
            },
            "decision": {
                "proto_strong_gain": C.PROTO_STRONG_GAIN,
                "pair_strong_gain": C.PAIR_STRONG_GAIN,
                "pair_vs_null_gain": C.PAIR_VS_NULL_GAIN,
            },
            "gpu_plan": dict(C.ARM_POOL),
            "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        },
        "gpu_authorized": False,
        "gpu_hours_used": 0.0,
    }
    C.write_json(C.PROTOCOL_FREEZE, freeze)
    C.log(f"[prep] wrote {C.track_rel(C.DATA_AUDIT)} and {C.track_rel(C.PROTOCOL_FREEZE)}")
    return {"data_audit": audit, "protocol_freeze": freeze}


def cmd_gates(args: argparse.Namespace) -> dict[str, Any]:
    del args
    return G.run_gates()


def cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out_root = C.ANALYSIS / "smoke"
    results = {}
    for arm in C.ARMS:
        job = T.PilotJob(
            arm=arm,
            device=args.device,
            pool="smoke",
            out_root=out_root,
            epochs=int(args.epochs),
            smoke=True,
            smoke_batches=int(args.batches),
        )
        results[arm] = T.train_job(job)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "device": args.device,
        "epochs": int(args.epochs),
        "eval_batches": int(args.batches),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "arms": {arm: {"dev_rmse_raw": res["metrics"]["dev_rmse_raw"],
                       "train_rmse_raw": res["metrics"]["train_rmse_raw"],
                       "wall_seconds": res["timing"]["process_wall_seconds"],
                       "parameters": res["model"]["parameters"],
                       "cuda_device_name": res["cuda_device_name"],
                       "cuda_peak_memory_bytes": res["cuda_peak_memory_bytes"]}
                 for arm, res in results.items()},
    }
    C.write_json(C.ANALYSIS / "smoke" / "smoke.json", payload)
    return payload


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    gates = require_gates()
    require_freeze()
    auth = require_authorization(args.arm)
    budget = auth.get("max_gpu_hours")
    extra = {
        "gates_sha256": C.sha256_file(C.CPU_GATES),
        "gates_n_passed": int(gates["n_passed"]),
        "authorization": {
            "path": C.track_rel(C.GPU_AUTHORIZATION),
            "max_gpu_hours": float(budget),
            "authorized_runs": list(auth["authorized_runs"]),
        },
    }
    job = T.PilotJob(arm=args.arm, device=args.device, pool=args.pool, extra=extra)
    return T.train_job(job)


def cmd_summarize(args: argparse.Namespace) -> dict[str, Any]:
    del args
    return pilot_analysis_summarize()


def pilot_analysis_summarize() -> dict[str, Any]:
    import pilot_analysis as A

    return A.summarize()


def cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    import pilot_analysis as A

    return A.ledger(args.dir, write=True)


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    del args
    import pilot_analysis as A

    payload = A.verify()
    if not payload["ok"]:
        failed = [c for c in payload["checks"] if not c["ok"]]
        raise SystemExit(f"verify failed: {json.dumps(failed, indent=2)[:4000]}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prep", help="fold-2 audit, pair coverage and protocol freeze")

    sub.add_parser("gates", help="run the 12 CPU correctness gates")

    p_smoke = sub.add_parser("smoke", help="short multi-arm smoke run")
    p_smoke.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0"])
    p_smoke.add_argument("--epochs", type=int, default=2)
    p_smoke.add_argument("--batches", type=int, default=2, help="evaluation batches per split")

    p_train = sub.add_parser("train", help="train one formal arm")
    p_train.add_argument("--arm", required=True, choices=list(C.ARMS))
    p_train.add_argument("--device", default="cuda", choices=["cpu", "cuda", "cuda:0"])
    p_train.add_argument("--pool", default=None, help="rr pool label (must match the frozen plan)")

    sub.add_parser("summarize", help="five-model table, paired comparisons and decision")

    p_ledger = sub.add_parser("ledger", help="GPU budget ledger from rr status JSON")
    p_ledger.add_argument("--dir", default=None, help="directory with `rr status --json` files")

    sub.add_parser("verify", help="re-verify checkpoints, predictions and metrics")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and args.pool is None:
        args.pool = C.ARM_POOL[args.arm]
    handlers = {
        "prep": cmd_prep,
        "gates": cmd_gates,
        "smoke": cmd_smoke,
        "train": cmd_train,
        "summarize": cmd_summarize,
        "ledger": cmd_ledger,
        "verify": cmd_verify,
    }
    payload = handlers[args.command](args)
    if args.command in ("gates", "verify"):
        return 0 if payload.get("ok") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
