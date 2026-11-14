"""Single entry point: ``prepare`` / ``smoke`` / ``train`` / ``analyze`` / ``report``.

    uv run python code/run.py prepare
    uv run python code/run.py smoke --device cpu
    uv run python code/run.py train --model C1_R1 --dataset Lipophilicity --fit 25
    uv run python code/run.py analyze
    uv run python code/run.py report

``train`` skips an arm whose ``train.json`` already exists (idempotent), so an
interrupted queue can be resumed by simply re-running it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import lb_common as C


def _prepare(_args) -> int:
    import lb_prepare
    lb_prepare.prepare()
    return 0


def _smoke(args) -> int:
    import torch
    import lb_train
    arm = {"model": "C1_R1", "dataset": "Lipophilicity", "fit": "25"}
    device = torch.device(args.device)
    out = C.RUNS / "smoke" / "C1_R1_Lipophilicity_F25"
    lb_train.train_arm(**arm, device=device, out_dir=out, smoke=True)
    if args.device == "cuda" and torch.cuda.is_available():
        lb_train.train_arm(model="DMPNN_D2F", dataset="BACE", fit="50", device=device,
                           out_dir=C.RUNS / "smoke" / "DMPNN_D2F_BACE_F50", smoke=True)
    return 0


def _train(args) -> int:
    import torch
    import lb_train
    if args.all:
        arms = C.all_arms()
    else:
        if not (args.model and args.dataset and args.fit):
            raise SystemExit("train needs --model/--dataset/--fit or --all")
        arms = [{"model": args.model, "dataset": args.dataset, "fit": args.fit}]
    device = torch.device(args.device)
    for a in arms:
        arm_id = lb_train._arm_name(a["model"], a["fit"], a["dataset"])
        out = C.RUNS / arm_id
        if (out / "train.json").exists() and not args.force:
            C.log(f"[skip] {arm_id} already completed")
            continue
        lb_train.train_arm(model=a["model"], dataset=a["dataset"], fit=a["fit"], device=device,
                           out_dir=out, max_updates_override=args.max_updates)
    return 0


def _probe(args) -> int:
    import torch
    import lb_probe
    device = torch.device(args.device)
    ckpt = Path(args.ckpt)
    out = Path(args.out_dir)
    lb_probe.extract_h(dataset=args.dataset, ckpt=ckpt, device=device, force=args.force)
    lb_probe.train_probe(rule=args.rule, dataset=args.dataset, ckpt=ckpt, device=device,
                         out_dir=out, updates=args.updates)
    return 0


def _analyze(_args) -> int:
    import lb_report
    lb_report.synthesize()
    return 0


def _report(_args) -> int:
    import lb_report
    lb_report.write_all()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="bottleneck-diagnosis")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("prepare"); p.set_defaults(fn=_prepare)
    p = sub.add_parser("smoke"); p.add_argument("--device", default="cpu"); p.set_defaults(fn=_smoke)
    p = sub.add_parser("train")
    p.add_argument("--model"); p.add_argument("--dataset"); p.add_argument("--fit")
    p.add_argument("--all", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda"); p.add_argument("--max-updates", type=int, default=None)
    p.set_defaults(fn=_train)
    p = sub.add_parser("probe")
    p.add_argument("--rule", required=True, choices=list(C.PROBE_ARMS))
    p.add_argument("--dataset", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--updates", type=int, default=None); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=_probe)
    p = sub.add_parser("analyze"); p.set_defaults(fn=_analyze)
    p = sub.add_parser("report"); p.set_defaults(fn=_report)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
