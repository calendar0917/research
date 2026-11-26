#!/usr/bin/env python
"""Phase B: the D0 three-model x three-target matrix (Fit2856 -> Dev504).

Every model starts from random initial weights *inside D0*; no official-Train
checkpoint is used as an initialisation.  The target scaler is fitted on Fit2856
only.  The pair schedule is the single frozen file
``analysis/pair_schedule.npz`` shared by all families and targets.

    uv run python .../code/run_d0.py --row RESP_PRODUCT --targets BASE,ENDPOINT,DELTA --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sr_common as C
import sr_context as SX
import sr_resp as SR
import sr_train as ST

ROWS = {
    "ROOT_STAR": ("FGCR", ("IND", "STAR"), 200),
    "ROOT_FULL": ("FGCR", ("FULL", "FULL"), 200),
    "RESP_PRODUCT": ("RESP", "PRODUCT", 360),
    "RESP_FUSION_MLP": ("RESP", "FUSION_MLP", 360),
    "GINE_G0": ("GINE", None, 360),
}
SCHEDULE_PATH = C.ANALYSIS_DIR / "pair_schedule.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", required=True, choices=sorted(ROWS))
    parser.add_argument("--targets", default="BASE,ENDPOINT,DELTA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", default=str(C.RESULTS_DIR / "B"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    family, operator, epochs = ROWS[args.row]
    if args.preflight:
        epochs = 2
        args.out_root = str(C.RESULTS_DIR / "preflight_B")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    C.seed_everything(C.TRAIN_SEED)

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / f"d0_{args.row}{'_preflight' if args.preflight else ''}.log"
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    env = C.environment_snapshot()
    t0 = time.perf_counter()
    ctx = SX.build_lipo(include_valid=False)
    fit_rows = np.asarray(ctx.d0["fit_rows"], dtype=np.int64)
    dev_rows = np.asarray(ctx.d0["dev_rows"], dtype=np.int64)
    ctx.bundle.set_scaler(fit_rows)
    scaler = dict(ctx.bundle.scaler)
    log(f"row={args.row} family={family} operator={operator} device={args.device} "
        f"epochs={epochs} fit_scaler={scaler} plan={ctx.plan_stats['n_graphs']}")
    log(f"pairs={json.dumps(C.jsonable(ctx.pairs.meta))}")
    log(f"env={json.dumps(C.jsonable(env))}")

    steps_per_epoch = int(np.ceil(fit_rows.size / int(
        C.FGCR_RECIPE["batch_molecules"] if family == "FGCR" else C.RESP_RECIPE["batch_molecules"])))
    schedule = SX.load_or_build_schedule(ctx, 9000, SCHEDULE_PATH)
    log(f"pair schedule sha256={C.sha256_array(schedule)} "
        f"steps_per_epoch={steps_per_epoch}")

    order = SX.batch_order(fit_rows.size, epochs)
    pair_a, pair_b = SX.pair_rows(ctx)
    dev_labels = ctx.bundle.labels[dev_rows].astype(np.float64)

    summaries = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        if target not in ST.TARGETS:
            raise SystemExit(f"unknown target {target}")
        out_dir = root / args.row / target
        if args.preflight:
            out_dir = root / f"{args.row}__{target}"
        if (out_dir / "summary.json").is_file():
            log(f"{args.row}/{target}: complete, skipping")
            summaries[target] = C.read_json(out_dir / "summary.json")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"{args.row}/{target}: start")

        arm = _build_arm(ctx, family, operator, args.device, fit_rows, scaler,
                         (pair_a, pair_b), dev_labels, args.row)
        result = ST.train_one(
            arm=arm, target=target, train_rows=fit_rows, order=order,
            schedule=schedule, labels_std=ctx.bundle.labels_std,
            scaler=scaler, eval_rows=dev_rows, epochs=epochs, out_dir=out_dir,
            log=log, checkpoint_extra={"row": args.row, "family": family,
                                       "target": target,
                                       "operator": (operator if isinstance(operator, str) else None),
                                       "h_mode": (operator[0] if family == "FGCR" else None),
                                       "eta_mode": (operator[1] if family == "FGCR" else None),
                                       "fit_scaler": scaler,
                                       "fit_rows_sha256": C.sha256_array(fit_rows),
                                       "dev_rows_sha256": C.sha256_array(dev_rows),
                                       "schedule_sha256": C.sha256_array(schedule)})
        summary = dict(result["summary"])
        summary.update({
            "row": args.row, "dataset": "ogbg-mollipo", "device": args.device,
            "gpu_name": (torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"),
            "peak_vram_gb": (float(torch.cuda.max_memory_allocated() / 1024 ** 3)
                             if args.device.startswith("cuda") else 0.0),
            "fit_scaler": scaler, "commit": C.git_commit(), "dirty": C.git_dirty(),
            "environment": env, "finished_at": C.utc_stamp(),
            "n_fit": int(fit_rows.size), "n_dev": int(dev_rows.size),
            "best_checkpoint_sha256": C.sha256_file(out_dir / "best.pt"),
            "final_checkpoint_sha256": C.sha256_file(out_dir / "final.pt"),
        })
        C.write_json(out_dir / "summary.json", summary)
        C.write_json(out_dir / "curves.json", {"curves": result["curves"]})
        np.savez_compressed(
            out_dir / "predictions.npz",
            dev_row_ids=dev_rows,
            dev_y_true=dev_labels,
            best_dev_y_pred=result["best_pred_raw"],
            final_dev_y_pred=result["final_pred_raw"],
            fit_row_ids=fit_rows,
            fit_y_true=ctx.bundle.labels[fit_rows].astype(np.float64),
            best_epoch=np.int64(summary["best_epoch"]),
            scaler_mean=np.float64(scaler["mean"]),
            scaler_std=np.float64(scaler["std"]),
        )
        summaries[target] = summary
        log(f"{args.row}/{target}: done best_epoch={summary['best_epoch']} "
            f"dev_rmse={summary['best_eval_rmse']:.6f} seconds={summary['seconds']:.0f}")
        C.ledger_append({
            "phase": "B", "unit": f"{args.row}/{target}", "device": args.device,
            "gpu_hours": summary["seconds"] / 3600.0, "seconds": summary["seconds"],
            "peak_vram_gb": summary["peak_vram_gb"], "status": "completed",
            "finished_at": C.utc_stamp(),
        })

    C.write_json(root / f"summary_{args.row}{'_preflight' if args.preflight else ''}.json", summaries)
    log("done")
    handle.close()
    return 0


def _build_arm(ctx, family, operator, device, fit_rows, scaler, pair_rows,
               dev_labels, row_name):
    meta = {"row": row_name, "dataset": "ogbg-mollipo"}
    if family == "FGCR":
        init = SX.fgcr_init_state(ctx)
        h_mode, eta_mode = operator
        return SX.make_fgcr_arm(ctx, h_mode=h_mode, eta_mode=eta_mode,
                                init_state=init, device=device,
                                pair_rows=pair_rows, eval_labels_raw=dev_labels,
                                meta=meta)
    if family == "RESP":
        # the fragment subset spans Fit + Dev (all official Train rows) so that the
        # Dev evaluation rows are addressable; training still only sees Fit rows and
        # the target scaler stays Fit-only.
        universe = np.union1d(fit_rows, ctx.d0["dev_rows"]).astype(np.int64)
        bundle = SR.RespBundle(ctx.dataset, universe, scaler, split_name="train")
        init = SR.canonical_init(bundle.dims)
        arm, _ = SX.make_resp_arm(ctx, rows=universe, split_name="train",
                                  scaler=scaler, operator=operator,
                                  init_state=init, device=device,
                                  pair_rows=pair_rows, eval_labels_raw=dev_labels,
                                  fit_cache=bundle, meta=meta)
        return arm
    if family == "GINE":
        init = SX_gine_init(ctx)
        return SX.make_gine_arm(ctx, init_state=init, device=device,
                                pair_rows=pair_rows, eval_labels_raw=dev_labels,
                                meta=meta)
    raise KeyError(family)


def SX_gine_init(ctx):
    import sr_gine as SG
    return SG.make_init_state(ctx.bundle.gine_categories, seed=C.TRAIN_SEED)


if __name__ == "__main__":
    raise SystemExit(main())
