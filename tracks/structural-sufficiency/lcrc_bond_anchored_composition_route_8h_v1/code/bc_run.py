"""One formal unit per process (rr-run entry point).

    python code/bc_run.py --institution FROZEN --task lipo --variant JOINT  --seed 0
    python code/bc_run.py --institution E2E    --task lipo --variant MARGINAL --seed 1
    python code/bc_run.py --stage prepare
    python code/bc_run.py --stage checks
    python code/bc_run.py --stage smoke --device cuda

Every scientific constant is frozen in the code/config pair and hashed into
``analysis/protocol_freeze.json``; the CLI exposes only engineering knobs
(device, epochs for a smoke run, path3 rebuild).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bc_common as C


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="BAJC formal unit runner")
    parser.add_argument("--stage", default="train", choices=["train", "prepare", "checks", "freeze", "smoke"])
    parser.add_argument("--institution", default="FROZEN", choices=list(C.INSTITUTIONS))
    parser.add_argument("--task", default="lipo", choices=list(C.TASKS))
    parser.add_argument("--variant", default="JOINT", choices=list(C.VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None, help="engineering only: smoke runs")
    parser.add_argument("--rebuild-paths3", action="store_true")
    parser.add_argument("--no-freeze-check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.stage == "prepare":
        import bc_prepare

        bc_prepare.main()
        return 0
    if args.stage == "freeze":
        import bc_freeze

        bc_freeze.main()
        return 0
    if args.stage == "checks":
        import bc_checks as K

        result = K.run_all()
        C.atomic_write_json(C.ANALYSIS_DIR / "preflight.json", result)
        print("failed:", result["failed"], "all_pass:", result["all_pass"])
        return 0 if result["all_pass"] else 1
    if args.stage == "smoke":
        import bc_train as T

        status = T.run_unit(
            institution=args.institution,
            task=args.task,
            variant=args.variant,
            seed=args.seed,
            device=args.device,
            rebuild_paths=args.rebuild_paths3,
            max_epochs=int(args.epochs or 2),
        )
        print(json.dumps({k: status[k] for k in ("unit", "selected_epoch", "selected_valid_rmse_raw",
                                                 "epoch0_valid_rmse_raw", "elapsed_seconds")}, indent=2))
        return 0

    import bc_freeze
    import bc_train as T

    if not args.no_freeze_check:
        state = bc_freeze.verify(strict=False)
        if state.get("present") and not state.get("clean"):
            print(
                "WARNING: protocol freeze is not clean: "
                + json.dumps({"changed": state["changed_files"], "added": state["added_files"],
                              "protocol_hash_match": state["protocol_hash_match"]}),
                file=sys.stderr,
            )
    t0 = time.time()
    status = T.run_unit(
        institution=args.institution,
        task=args.task,
        variant=args.variant,
        seed=args.seed,
        device=args.device,
        rebuild_paths=args.rebuild_paths3,
        max_epochs=args.epochs,
    )
    print(json.dumps({k: status[k] for k in (
        "unit", "selected_epoch", "selected_valid_rmse_raw", "epoch0_valid_rmse_raw",
        "development_test_rmse_raw", "elapsed_seconds", "gpu_hours")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
