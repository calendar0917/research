"""Freeze the final configuration, seed set and checkpoint hashes.

This step must run before any official-test read. It writes
`analysis/protocol_freeze.json` and refuses to silently change an existing
freeze.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import zpp_common as C
import zpp_zsrt as Z


def _checkpoint_hash(path: Path) -> str:
    return C.sha256_file(path)


def freeze(config_name: str, seeds: list[int]) -> dict:
    variant = Z.load_variant(config_name)
    p3_rows = {}
    for seed in (0, 1):
        info_path = C.RUNS / f"p3_s{seed}" / "train.json"
        best = C.RUNS / f"p3_s{seed}" / "best.pt"
        if not info_path.exists() or not best.exists():
            raise RuntimeError(f"missing P3 calibration artifacts for seed {seed}")
        info = C.read_json(info_path)
        if bool(info["data"]["official_test_loaded"]):
            raise RuntimeError("P3 calibration reports a test read")
        p3_rows[str(seed)] = {
            "path": str(best.relative_to(C.REPO)),
            "sha256": _checkpoint_hash(best),
            "best_epoch": int(info["best_valid"]["epoch"]),
            "best_valid_mae": float(info["best_valid"]["valid_mae"]),
            "final_valid_mae": float(info["fixed_objects"]["final"]["valid_mae"]),
            "soup_valid_mae": float(info["fixed_objects"]["soup"]["valid_mae"]),
            "parameter_count": int(info["parameter_count"]),
            "updates": int(info["recipe"]["updates"]),
            "wall_seconds": float(info["wall_seconds"]),
        }

    zsrt_rows = {}
    for seed in seeds:
        out = Z.run_dir(variant, seed)
        info_path = out / "train.json"
        best = out / "best.pt"
        if not info_path.exists() or not best.exists():
            raise RuntimeError(f"missing ZSRT artifacts for seed {seed}")
        info = C.read_json(info_path)
        if bool(info["data"]["official_test_loaded"]):
            raise RuntimeError("ZSRT training reports a test read")
        if info["variant"]["name"] != config_name:
            raise RuntimeError("variant name mismatch in train.json")
        zsrt_rows[str(seed)] = {
            "path": str(best.relative_to(C.REPO)),
            "sha256": _checkpoint_hash(best),
            "best_epoch": int(info["best_valid"]["epoch"]),
            "best_valid_mae": float(info["best_valid"]["valid_mae"]),
            "final_valid_mae": float(info["fixed_objects"]["final"]["valid_mae"]),
            "soup_valid_mae": float(info["fixed_objects"]["soup"]["valid_mae"]),
            "parameter_count": int(info["parameter_count"]),
            "updates": int(info["recipe"]["updates"]),
            "wall_seconds": float(info["wall_seconds"]),
        }

    code_files = [
        C.EXP_DIR / "code/zpp_common.py",
        C.EXP_DIR / "code/zpp_data.py",
        C.EXP_DIR / "code/zpp_p3.py",
        C.EXP_DIR / "code/zpp_zsrt.py",
        C.EXP_DIR / "code/zpp_score.py",
        C.PROTOCOL_YAML,
        C.ZSRT_YAML,
    ]
    payload = {
        "protocol": C.PROTOCOL_ID,
        "status": "FROZEN_FOR_TERMINAL_TEST",
        "config_name": config_name,
        "variant": variant,
        "zsrt_seeds": [int(s) for s in seeds],
        "p3_seeds": [0, 1],
        "screen_seed_participates": 0 in [int(s) for s in seeds],
        "selection_rule": C.load_yaml(C.PROTOCOL_YAML)["zsrt"]["selection_rule"],
        "checkpoints": {"p3": p3_rows, "zsrt": zsrt_rows},
        "code_sha256": {str(p.relative_to(C.REPO)): C.sha256_file(p) for p in code_files if p.exists()},
        "data_audit_sha256": C.sha256_file(C.DATA_AUDIT) if C.DATA_AUDIT.exists() else None,
        "git_commit": C.git_commit(),
        "freeze_timestamp": C.utcnow(),
        "test_policy": "one_shot_terminal_after_this_freeze",
        "test_history": "official ZINC test was exposed in earlier rounds; not a blind test",
        "no_test_used_for_selection": True,
    }
    if C.PROTOCOL_FREEZE.exists():
        old = C.read_json(C.PROTOCOL_FREEZE)
        if old.get("checkpoints") != payload["checkpoints"] or old.get("config_name") != config_name:
            raise RuntimeError(
                "an existing freeze differs; refusing to rewrite it. "
                "Use a new namespace for a new decision."
            )
        C.log("freeze already exists and matches; leaving it unchanged")
        return old
    C.write_json(C.PROTOCOL_FREEZE, payload)
    C.log(f"froze {config_name} seeds={seeds} -> {C.PROTOCOL_FREEZE.relative_to(C.REPO)}")
    return payload


def verify_freeze() -> dict:
    if not C.PROTOCOL_FREEZE.exists():
        raise RuntimeError("missing protocol_freeze.json; run freeze first")
    frozen = C.read_json(C.PROTOCOL_FREEZE)
    for family in ("p3", "zsrt"):
        for seed, row in frozen["checkpoints"][family].items():
            path = C.REPO / row["path"]
            if not path.exists():
                raise RuntimeError(f"frozen checkpoint missing: {path}")
            if C.sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"frozen checkpoint hash mismatch: {path}")
    for rel, want in frozen["code_sha256"].items():
        path = C.REPO / rel
        if not path.exists() or C.sha256_file(path) != want:
            raise RuntimeError(f"frozen code hash mismatch: {rel}")
    if frozen.get("data_audit_sha256") and C.sha256_file(C.DATA_AUDIT) != frozen["data_audit_sha256"]:
        raise RuntimeError("data audit changed after freeze")
    return frozen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "verify"])
    p.add_argument("--config", default="A_main")
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args(argv)
    if args.command == "freeze":
        freeze(args.config, [int(s) for s in args.seeds.split(",") if s.strip()])
    else:
        frozen = verify_freeze()
        C.log(f"freeze verified: {frozen['config_name']} seeds={frozen['zsrt_seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
