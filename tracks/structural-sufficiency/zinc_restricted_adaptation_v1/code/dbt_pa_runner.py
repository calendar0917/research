"""CLI for zinc-restricted-adaptation-v1 (DBT-PA).

    python .../dbt_pa_runner.py prep    --device cuda:0
    python .../dbt_pa_runner.py checks
    python .../dbt_pa_runner.py timing  --device cuda:0 --epochs 4
    python .../dbt_pa_runner.py plan    --spent-seconds <s>
    python .../dbt_pa_runner.py train   --device cuda:0 --epochs E
    python .../dbt_pa_runner.py test    --device cuda:0
    python .../dbt_pa_runner.py analysis
    python .../dbt_pa_runner.py groups
    python .../dbt_pa_runner.py references
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import dbt_pa_analysis as PA  # noqa: E402
import dbt_pa_checks as PCK  # noqa: E402
import dbt_pa_groups as PG  # noqa: E402
import dbt_pa_prep as PP  # noqa: E402
import dbt_pa_references as PR  # noqa: E402
import dbt_pa_test as PT  # noqa: E402
import dbt_pa_train as BT  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("checks")
    s = sub.add_parser("timing")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--epochs", type=int, default=4)
    s = sub.add_parser("plan")
    s.add_argument("--spent-seconds", type=float, required=True)
    s.add_argument("--budget-seconds", type=float, default=3600.0)
    s.add_argument("--reserve-seconds", type=float, default=180.0)
    s = sub.add_parser("train")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--epochs", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--save-every", type=int, default=25)
    s = sub.add_parser("test")
    s.add_argument("--device", default="cuda:0")
    s = sub.add_parser("analysis")
    s = sub.add_parser("groups")
    s.add_argument("--split", choices=["valid", "test", "both"], default="both")
    s = sub.add_parser("references")
    args = p.parse_args(argv)

    if args.cmd == "prep":
        PP.prep(torch.device(args.device), force=args.force)
        return 0
    if args.cmd == "checks":
        PCK.run_checks()
        return 0
    if args.cmd == "timing":
        PP.timing(torch.device(args.device), args.epochs)
        return 0
    if args.cmd == "plan":
        PP.plan(spent_seconds=args.spent_seconds,
                budget_seconds=args.budget_seconds,
                reserve_seconds=args.reserve_seconds)
        return 0
    if args.cmd == "train":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        BT.train(args.seed, device, epochs=args.epochs, force=args.force,
                 resume=args.resume, save_every=args.save_every)
        return 0
    if args.cmd == "test":
        return PT.main(["--device", str(args.device)])
    if args.cmd == "analysis":
        return PA.main([])
    if args.cmd == "groups":
        return PG.main(["--split", args.split])
    if args.cmd == "references":
        return PR.main()
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
