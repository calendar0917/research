"""Two-epoch GPU smoke and budget projection for ECLC-v1.

The smoke runs one arm (NULL by default) for two epochs with the frozen pair
loss, measures steady epoch time, audits the pair/base gradient path, and
projects the cost of the two seed-0 arms.  If the projection exceeds the
2.0 A100-hour seed-0 cap, the decision is STOP_BUDGET and no formal run is
launched.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import eclc_common as E
import eclc_train as T

SMOKE_CAP_GPU_HOURS = 2.0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--arm", type=str, default="NULL", choices=list(E.ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(argv)
    timing = T.timing([args.arm], args.seed, args.device, epochs=args.epochs)
    run = timing[args.arm]
    projected_two_arm = 2.0 * float(run["projected_gpu_hours"])
    smoke_hours = float(sum(run["epoch_seconds"])) / 3600.0
    total_projected = projected_two_arm + smoke_hours
    g = run["first_batch_gradient_audit"] or {}
    p = run["first_pair_audit"] or {}
    pair_path_ok = bool(
        p.get("loss_pair", 0.0) > 0.0
        and g.get("base_last_weight_grad_norm", 0.0) > 0.0
        and g.get("f_context_last_weight_grad_norm", 0.0) > 0.0
        and g.get("rho_last_weight_grad_norm", 0.0) > 0.0
        and g.get("all_grad_finite", False)
    )
    decision = "PASS_BUDGET" if (total_projected <= SMOKE_CAP_GPU_HOURS and pair_path_ok) else "STOP_BUDGET"
    payload = {
        "round": E.ROUND_ID,
        "created": E.now(),
        "git_commit": E.git_commit(),
        "environment": E.L.env_summary(),
        "arm": args.arm,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "timing": run,
        "pair_path_ok": bool(pair_path_ok),
        "projected_two_arm_gpu_hours": float(projected_two_arm),
        "smoke_gpu_hours": float(smoke_hours),
        "total_projected_gpu_hours": float(total_projected),
        "seed0_cap_gpu_hours": float(SMOKE_CAP_GPU_HOURS),
        "decision": decision,
    }
    E.write_json(E.ANALYSIS / "smoke.json", payload)
    E.log(f"[smoke] projected two-arm seed0 = {total_projected:.3f} A100-h; "
          f"pair_path_ok={pair_path_ok}; decision={decision}")
    return 0 if decision == "PASS_BUDGET" else 1


if __name__ == "__main__":
    raise SystemExit(main())
