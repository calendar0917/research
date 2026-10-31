"""Command-line driver for lcrc-lipo-esa-random42-v1.

Sub-commands
------------
freeze-protocol   verify the CSV/split, write the frozen protocol + hashes
prep              build the CPU-side LCRC cache on the compute host
audit             Stage-1 data/protocol audit -> analysis/data_audit.json
gates             Stage-2 CPU implementation gates -> analysis/cpu_gates.json
smoke             tiny-batch CPU/GPU compatibility smoke for one model
train             formal training for one model (lcrc|pna|esa|xgb)
ledger            rebuild analysis/gpu_ledger.json from run manifests
implementation    freeze the implementation manifest (+ unit-test evidence)
manifest          build the frozen pre-test manifest (+ prediction manifest)
score             one-shot Test evaluation (refuses if already scored)
status            print the current round status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import r42_common as C


def _split_and_labels():
    import r42_data as D
    import r42_labels as L

    return D.load_split(), L.load_train_valid_labels()


def cmd_freeze(args: argparse.Namespace) -> int:
    import r42_data as D
    import r42_labels as L

    cfg = C.load_config()
    split = D.freeze_split()
    frozen = L.freeze_train_valid_labels()
    labels = L.load_train_valid_labels()
    names = None
    try:
        import r42_xgb as X

        names = X.freeze_descriptor_names()
    except Exception as exc:  # rdkit unavailable on the compute host
        C.log(f"[freeze] descriptor names not frozen here ({type(exc).__name__}); the CPU host does it")
    protocol = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "config_path": C.track_rel(C.CONFIG_PATH),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "config_canonical_sha256": C.sha256_json(cfg),
        "csv": {
            "path": C.track_rel(C.CSV_PATH),
            "sha256": C.sha256_file(C.CSV_PATH),
            "n_rows": C.N_MOLECULES,
        },
        "split": {
            "path": C.track_rel(C.SPLIT_FILE),
            "sha256": C.sha256_file(C.SPLIT_FILE),
            "file_sha256": C.sha256_file(C.SPLIT_FILE),
            "index_sha256": {k: C.rows_sha256_int64(split["rows"][k]) for k in ("train", "valid", "test")},
            "sizes": dict(C.SPLIT_SIZES),
            "seed": C.SPLIT_SEED,
        },
        "labels": {
            "path": C.track_rel(C.LABELS_TRAIN_VALID),
            "sha256": C.sha256_file(C.LABELS_TRAIN_VALID),
            "scope": frozen["scope"],
            "test_labels_included": frozen["test_labels_included"],
            "vector_sha256": frozen["vector_sha256"],
            "row_sha256_int64": frozen["row_sha256_int64"],
        },
        "structures": D.structure_summary() | {"meta_sha256": C.sha256_file(C.STRUCTURES_META)},
        "descriptors": None if names is None else {"n_names": names["n_names"], "names_sha256": names["names_sha256"]},
        "models": {
            "lcrc": C.config_hash(cfg["models"]["lcrc"]),
            "pna": C.config_hash(cfg["models"]["pna"]),
            "esa": C.config_hash(cfg["models"]["esa"]),
            "xgb": C.config_hash(cfg["models"]["xgb"]),
        },
        "code_hashes": {
            str(p.relative_to(C.EXP_DIR)): C.sha256_file(p)
            for p in sorted(C.CODE_DIR.glob("r42_*.py"))
        },
        "upstream_commit": C.UPSTREAM_COMMIT,
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "env": C.env_info(),
    }
    C.write_json(C.ANALYSIS / "protocol_freeze.json", protocol)
    C.log(f"[freeze] protocol frozen: {C.track_rel(C.ANALYSIS / 'protocol_freeze.json')} sha256={C.sha256_file(C.ANALYSIS / 'protocol_freeze.json')}")
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    """Build the CPU-side caches on the compute host (no GPU time charged)."""
    import time

    import r42_data as D

    started = time.time()
    built = D.build_lcrc_cache()
    summary = D.structure_summary()
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "hostname": __import__("socket").gethostname(),
        "git_commit": C.git_commit(),
        "lcrc_cache": {
            "path": C.track_rel(built["path"]),
            "file_sha256": built["file_sha256"],
            "content_fingerprint": built["content_fingerprint"],
            "rebuilt": bool(built["rebuilt"]),
            "n_molecules": int(built["cache"]["n_molecules"]),
            "labels_present": built["cache"]["meta"].get("labels_present"),
        },
        "structures": summary,
        "structures_meta_sha256": C.sha256_file(C.STRUCTURES_META),
        "seconds": float(time.time() - started),
        "gpu_used": False,
    }
    C.write_json(C.ANALYSIS / "prep.json", payload)
    C.log(f"[prep] lcrc cache {payload['lcrc_cache']['file_sha256']} rebuilt={built['rebuilt']} in {payload['seconds']:.1f}s")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    import r42_audit

    payload = r42_audit.audit()
    return 0 if payload["status"] == "PASS" else 1


def cmd_gates(args: argparse.Namespace) -> int:
    import r42_gates

    payload = r42_gates.run_cpu_gates(C.load_config(), skip_esa=bool(args.skip_esa))
    return 0 if payload["status"] == "PASS" else 1


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cmd_smoke(args: argparse.Namespace) -> int:
    import time

    import torch

    import r42_lcrc_train as LT
    import r42_native_train as NT

    cfg = C.load_config()
    split, labels = _split_and_labels()
    device = _device(args.device)
    run_dir = C.RUNS / f"smoke_{args.model}"
    envelope = {
        "protocol_id": C.PROTOCOL_ID,
        "run_key": f"smoke_{args.model}",
        "model": f"{args.model}-smoke",
        "started_utc": C.utcnow(),
        "started_epoch_seconds": time.time(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "hostname": __import__("socket").gethostname(),
        "git_commit": C.git_commit(),
        "argv": sys.argv,
        "status": "RUNNING",
    }
    C.write_json(run_dir / "envelope.json", envelope)
    if args.model == "lcrc":
        payload = LT.train_lcrc(
            cfg=cfg["models"]["lcrc"], split=split, labels=labels, device=device, run_dir=run_dir,
            limit_train=args.limit_train, limit_valid=args.limit_valid, max_epochs=args.epochs, soup_last=sum(1 for _ in range(args.epochs)), label="LCRC-smoke",
        )
    elif args.model in ("pna", "esa"):
        payload = NT.train_native(
            kind=args.model, cfg=cfg["models"][args.model], split=split, labels=labels, device=device,
            run_dir=run_dir, limit_train=args.limit_train, limit_valid=args.limit_valid, max_epochs=args.epochs, label=f"{args.model}-smoke",
        )
    else:
        raise SystemExit(f"unknown smoke model {args.model!r}")
    envelope.update({"finished_utc": C.utcnow(), "finished_epoch_seconds": time.time(), "status": "COMPLETED"})
    envelope["elapsed_seconds"] = envelope["finished_epoch_seconds"] - envelope["started_epoch_seconds"]
    C.write_json(run_dir / "envelope.json", envelope)
    C.log(f"[smoke] {args.model} ok: {payload['metrics']['valid']} elapsed={envelope['elapsed_seconds']:.1f}s")
    return 0


RUN_KEYS = {"lcrc": "lcrc_a_r3_s0", "pna": "pna_native_s0", "esa": "esa_native_s0", "xgb": "xgb_p3_s0"}


def cmd_train(args: argparse.Namespace) -> int:
    import json
    import time

    import torch

    import r42_lcrc_train as LT
    import r42_native_train as NT
    import r42_xgb as X

    cfg = C.load_config()
    split, labels = _split_and_labels()
    device = _device(args.device)
    model = args.model
    run_dir = C.RUNS / RUN_KEYS[model]
    if model == "xgb":
        envelope = {
            "protocol_id": C.PROTOCOL_ID,
            "run_key": RUN_KEYS[model],
            "model": "XGB-P3",
            "started_utc": C.utcnow(),
            "started_epoch_seconds": time.time(),
            "device": "cpu",
            "cuda_device_name": None,
            "hostname": __import__("socket").gethostname(),
            "git_commit": C.git_commit(),
            "argv": sys.argv,
            "status": "RUNNING",
        }
        C.write_json(run_dir / "envelope.json", envelope)
        try:
            payload = X.train_xgb(cfg["models"]["xgb"], split, labels, run_dir)
        except BaseException as exc:
            envelope.update(
                {
                    "finished_utc": C.utcnow(),
                    "finished_epoch_seconds": time.time(),
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            envelope["elapsed_seconds"] = envelope["finished_epoch_seconds"] - envelope["started_epoch_seconds"]
            C.write_json(run_dir / "envelope.json", envelope)
            raise
        envelope.update({"finished_utc": C.utcnow(), "finished_epoch_seconds": time.time(), "status": "COMPLETED"})
        envelope["elapsed_seconds"] = envelope["finished_epoch_seconds"] - envelope["started_epoch_seconds"]
        C.write_json(run_dir / "envelope.json", envelope)
        C.log(f"[train] xgb done: valid={payload['metrics']['valid']} elapsed={envelope['elapsed_seconds']:.1f}s")
        return 0
    if args.gpu_budget_check:
        import r42_ledger

        ledger = r42_ledger.build_ledger()
        if ledger["gpu_hours_charged"] >= C.GPU_HARD_CAP_HOURS:
            raise SystemExit(f"GPU budget exhausted: {ledger['gpu_hours_charged']:.3f} h >= {C.GPU_HARD_CAP_HOURS} h")
    envelope = {
        "protocol_id": C.PROTOCOL_ID,
        "run_key": RUN_KEYS[model],
        "model": model,
        "started_utc": C.utcnow(),
        "started_epoch_seconds": time.time(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "hostname": __import__("socket").gethostname(),
        "git_commit": C.git_commit(),
        "argv": sys.argv,
        "status": "RUNNING",
    }
    C.write_json(run_dir / "envelope.json", envelope)
    try:
        if model == "lcrc":
            payload = LT.train_lcrc(
                cfg=cfg["models"]["lcrc"], split=split, labels=labels, device=device, run_dir=run_dir,
                label="LCRC-A-R3-scratch",
            )
        else:
            payload = NT.train_native(
                kind=model, cfg=cfg["models"][model], split=split, labels=labels, device=device,
                run_dir=run_dir, label=f"{model}-native",
            )
    except BaseException as exc:
        envelope.update(
            {
                "finished_utc": C.utcnow(),
                "finished_epoch_seconds": time.time(),
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        envelope["elapsed_seconds"] = envelope["finished_epoch_seconds"] - envelope["started_epoch_seconds"]
        C.write_json(run_dir / "envelope.json", envelope)
        raise
    envelope.update(
        {
            "finished_utc": C.utcnow(),
            "finished_epoch_seconds": time.time(),
            "status": "COMPLETED",
        }
    )
    envelope["elapsed_seconds"] = envelope["finished_epoch_seconds"] - envelope["started_epoch_seconds"]
    C.write_json(run_dir / "envelope.json", envelope)
    C.log(f"[train] {model} done: valid={payload['metrics']['valid']} elapsed={envelope['elapsed_seconds']:.1f}s")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    import r42_ledger

    payload = r42_ledger.build_ledger()
    C.log(f"[ledger] {payload['gpu_hours_charged']:.4f} GPU-h over {len(payload['entries'])} entries")
    return 0


def cmd_implementation(args: argparse.Namespace) -> int:
    import r42_manifest

    payload = r42_manifest.build_implementation_manifest()
    C.log(f"[implementation] status={payload['status']}")
    return 0 if payload["status"] == "PASS" else 1


def cmd_manifest(args: argparse.Namespace) -> int:
    import r42_manifest

    payload = r42_manifest.build_pre_test_manifest()
    C.log(f"[manifest] {C.track_rel(C.ANALYSIS / 'pre_test_manifest.json')} sha256={payload['manifest_sha256']}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    import r42_score

    payload = r42_score.score_test(expected_manifest_sha256=args.manifest_sha256)
    C.log(f"[score] {payload['status']}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    import json

    out: dict[str, Any] = {}
    for name in ("data_audit", "cpu_gates", "gpu_ledger", "pre_test_manifest", "metrics_test"):
        path = C.ANALYSIS / f"{name}.json"
        if path.exists():
            payload = C.read_json(path)
            out[name] = {k: payload.get(k) for k in ("status", "gpu_hours_charged", "manifest_sha256") if k in payload}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-protocol").set_defaults(func=cmd_freeze)
    sub.add_parser("prep").set_defaults(func=cmd_prep)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    gates = sub.add_parser("gates")
    gates.add_argument("--skip-esa", action="store_true")
    gates.set_defaults(func=cmd_gates)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("model", choices=["lcrc", "pna", "esa"])
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--limit-train", type=int, default=32)
    smoke.add_argument("--limit-valid", type=int, default=16)
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.set_defaults(func=cmd_smoke)
    train = sub.add_parser("train")
    train.add_argument("model", choices=["lcrc", "pna", "esa", "xgb"])
    train.add_argument("--device", default="auto")
    train.add_argument("--gpu-budget-check", action="store_true")
    train.set_defaults(func=cmd_train)
    sub.add_parser("ledger").set_defaults(func=cmd_ledger)
    sub.add_parser("implementation").set_defaults(func=cmd_implementation)
    sub.add_parser("manifest").set_defaults(func=cmd_manifest)
    score = sub.add_parser("score")
    score.add_argument("--manifest-sha256", required=True)
    score.set_defaults(func=cmd_score)
    sub.add_parser("status").set_defaults(func=cmd_status)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
