"""Run one missing LCRC-P1 unit with the frozen R3 training code.

The round adds **no** model, loss, feature or recipe code.  This module is a thin
wrapper that

1. resolves one unit of this round (split x arm x seed) to the frozen
   ``lcrc_performance_longrun_v1`` training driver,
2. imports ``cl_train.train_run`` unchanged (the driver re-verifies the frozen
   protocol hash manifest before training),
3. writes the artefacts into **this** round's ``analysis/runs/<unit_key>/``
   instead of the historical round directory.

Nothing else is overridden: the arm construction, BASE-initialised shared
tensors, zero-initialised extensions, R3 recipe, Validation-RMSE checkpoint
rule, single post-selection Test read and the full provenance record all come
from the frozen driver.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from p1_common import (
    ANALYSIS,
    LONGRUN_CODE,
    PROTOCOL_ID,
    ROUND,
    RUNS,
    SPLITS,
    track_rel,
    unit_key,
    write_json,
)

if str(LONGRUN_CODE) not in sys.path:
    sys.path.insert(0, str(LONGRUN_CODE))

import cl_common as C  # noqa: E402,I001
import cl_train as CT  # noqa: E402

#: the frozen training driver's code hashes must not drift inside this round
EXPECTED_CODE_HASHES = {
    "prep/feature_spec.py": "37f860a90bd805669b570d44394a791947c9eabdd7cc68a089264ced7c7a4260",
    "prep/lcrc_data.py": "fab6ce8ef61e4997f67516b2b2e5d861c72ed870d57a2599bc2787b8cd3ac3d3",
    "prep/lcrc_model.py": "d5af76cb9b5998e6bfa068afd28c483c93dde579cfb85344ed928582b05c6d37",
    "v2/mb2_data.py": "c57e2eb2934060634a3e61a74efdcff3471e0c170bcc0bdaf595ab41000ff767",
    "v2/mb2_metrics.py": "0c419a874f7012d1a76cc791f46b095e78e19fdf7bfdfa8cb50dd93e00c6e2a6",
    "v2/mb2_model.py": "75366675b98ffe5a8a82c96c241eb400dbce51be688070b4f4f7a142f4e0f4b4",
    "tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_chem8.py": "339dc22c6e189b03f111cca1247e1bd6a6bce44b008d6d8ab73888593ed605f7",
    "tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_common.py": "c2fba06310f0806d6374fab1c75d84f48dbd367178d3c0ace2c4b60163aae3d3",
    "tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_data.py": "8e37f1ece64843add0f300ca8779c88bb126d2fc20837ebba50b4b24dd5bda82",
    "tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_model.py": "7374253d27622505077e27ccf36db62105f889d0bf31e36464f2b42d7ccf7ff9",
    "tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_train.py": "7242bafcf1aa9a883d230034a15b961e4fb53f5820ebaff0a0408a46f4db39dc",
}


def verify_frozen_driver() -> dict[str, Any]:
    actual = C.code_hashes()
    mismatched = {k: (EXPECTED_CODE_HASHES.get(k), v) for k, v in actual.items() if EXPECTED_CODE_HASHES.get(k) != v}
    if mismatched:
        raise RuntimeError(f"frozen training code drifted: {mismatched}")
    freeze = C.verify_protocol_freeze()
    return {"code_hashes": actual, "protocol_freeze": freeze}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LCRC-P1 single-unit training wrapper")
    parser.add_argument("--split", required=True, choices=sorted(SPLITS))
    parser.add_argument("--arm", required=True, choices=["base", "chem8", "multicontrib", "combo_chem_contrib"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite an existing unit directory")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    if args.smoke and args.epochs is None:
        args.epochs = 2
    if args.epochs is not None and not args.smoke:
        planned = int(C.recipe_for("default")["epochs"])
        if args.epochs != planned:
            raise SystemExit(f"formal runs must use {planned} epochs")

    key = unit_key(args.split, args.arm, args.seed)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.smoke:
        # smoke runs must never land in the unit registry directory
        out_dir = ANALYSIS / "smoke" / key
    else:
        out_dir = RUNS / key
    if (out_dir / "train.json").exists() and not args.force:
        raise SystemExit(f"{track_rel(out_dir)} already holds a completed run; pass --force to overwrite")

    frozen = verify_frozen_driver()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    info = CT.train_run(
        dataset="Lipophilicity",
        split=SPLITS[args.split]["longrun_split"],
        arm=args.arm,
        seed=int(args.seed),
        device=device,
        out_dir=out_dir,
        protocol="default",
        scale="base",
        test=True,
        smoke=bool(args.smoke),
        epochs_override=args.epochs,
        run_key=key,
    )

    # record the round-level provenance next to the frozen driver payload
    write_json(
        out_dir / "p1_unit_provenance.json",
        {
            "round_id": PROTOCOL_ID,
            "unit_key": key,
            "split": args.split,
            "arm": args.arm,
            "seed": int(args.seed),
            "smoke": bool(args.smoke),
            "frozen_driver": frozen,
            "test_policy": "DEVELOPMENT_TEST: read once after the best-Validation checkpoint was frozen",
        },
    )
    print(json.dumps({"unit_key": key, "best_epoch": info["best_epoch"], "wall": info["wall"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
