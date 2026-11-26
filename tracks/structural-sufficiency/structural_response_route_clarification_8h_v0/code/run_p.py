#!/usr/bin/env python
"""Conditional branch P: carry the single qualifying target row back to official Lipo.

Trains BASE / ENDPOINT / DELTA for the one selected row on official Train3360 with
the official-Train scaler, using **all 2133 official-Train pairs** grouped by their
391 components for the auxiliary term.  Selection is on Valid420 molecule RMSE.

BASE may be reused from a strictly identical phase-A arm; the identity argument is
recorded rather than assumed.

    uv run python .../code/run_p.py --row RESP_PRODUCT --device cuda:0 [--reuse-base results/A/A_IS]
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
import sr_gine as SG
import sr_pairs as SP
import sr_resp as SR
import sr_train as ST

ROWS = {
    "ROOT_STAR": ("FGCR", ("IND", "STAR"), 200),
    "ROOT_FULL": ("FGCR", ("FULL", "FULL"), 200),
    "RESP_PRODUCT": ("RESP", "PRODUCT", 360),
    "RESP_FUSION_MLP": ("RESP", "FUSION_MLP", 360),
    "GINE_G0": ("GINE", None, 360),
}
A_EQUIVALENT = {"ROOT_STAR": "A_IS", "ROOT_FULL": "A_FJ", "RESP_PRODUCT": None,
                "RESP_FUSION_MLP": None, "GINE_G0": None}
SCHEDULE = C.ANALYSIS_DIR / "pair_schedule_p2133.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", required=True, choices=sorted(ROWS))
    parser.add_argument("--targets", default="BASE,ENDPOINT,DELTA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", default=str(C.RESULTS_DIR / "P"))
    parser.add_argument("--reuse-base", default=None,
                        help="phase-A results directory whose BASE is strictly identical")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    family, operator, epochs = ROWS[args.row]
    if args.preflight:
        epochs = 2
        args.out_root = str(C.RESULTS_DIR / "preflight_P")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    C.seed_everything(C.TRAIN_SEED)

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / f"p_{args.row}{'_preflight' if args.preflight else ''}.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    ctx = SX.build_lipo(include_valid=True)
    index = ctx.pairs
    train_rows = np.asarray(ctx.split_rows["train"], dtype=np.int64)
    valid_rows = np.asarray(ctx.split_rows["valid"], dtype=np.int64)
    scaler = dict(ctx.bundle.scaler)
    log(f"row={args.row} family={family} operator={operator} device={args.device} "
        f"epochs={epochs} scaler={scaler} reuse_base={args.reuse_base}")
    log(f"env={json.dumps(C.jsonable(C.environment_snapshot()))}")

    all_pairs = np.arange(index.n, dtype=np.int64)
    comps = np.asarray(index.component_ids, dtype=np.int64)
    if SCHEDULE.is_file():
        with np.load(SCHEDULE) as h:
            schedule = h["schedule"]
        if int(h["seed"]) != C.PAIR_SEED:
            raise RuntimeError("P schedule seed mismatch")
    else:
        schedule = SP.pair_schedule_generic(all_pairs, comps, 9000)
        np.savez_compressed(SCHEDULE, schedule=schedule, seed=np.int64(C.PAIR_SEED),
                            components=np.int64(len(np.unique(comps))),
                            sha256=np.asarray(C.sha256_array(schedule)))
    log(f"P schedule sha256={C.sha256_array(schedule)} "
        f"n_pairs={index.n} n_components={len(np.unique(comps))}")
    pair_a = np.asarray([r["row_a"] for r in index.rows], dtype=np.int64)
    pair_b = np.asarray([r["row_b"] for r in index.rows], dtype=np.int64)

    order = SX.batch_order(train_rows.size, epochs)
    valid_labels = ctx.bundle.labels[valid_rows].astype(np.float64)
    summaries: dict[str, Any] = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        out_dir = root / args.row / target
        if args.preflight:
            out_dir = root / f"{args.row}__{target}"
        if (out_dir / "summary.json").is_file():
            log(f"{args.row}/{target}: complete, skipping")
            summaries[target] = C.read_json(out_dir / "summary.json")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        reuse = (target == "BASE" and args.reuse_base and A_EQUIVALENT[args.row])
        if reuse:
            src = C.REPO_ROOT / args.reuse_base if not Path(args.reuse_base).is_absolute() \
                else Path(args.reuse_base)
            preds = np.load(src / "predictions.npz")
            if not np.array_equal(np.asarray(preds["valid_row_ids"]), valid_rows):
                raise RuntimeError("phase-A BASE predictions are not on the same Valid rows")
            summary = {
                "family": family, "target": "BASE", "epochs": int(epochs), "seconds": 0.0,
                "best_epoch": int(np.asarray(preds["best_epoch"])),
                "best_eval_rmse": C.rmse(np.asarray(preds["best_valid_y_pred"]), valid_labels),
                "best_eval_mae": C.mae(np.asarray(preds["best_valid_y_pred"]), valid_labels),
                "final_eval_rmse": C.rmse(np.asarray(preds["final_valid_y_pred"]), valid_labels),
                "final_eval_mae": C.mae(np.asarray(preds["final_valid_y_pred"]), valid_labels),
                "last20_eval_rmse": float(np.asarray(preds["last20_valid_rmse"])),
                "last20_eval_mae": float(np.asarray(preds["last20_valid_mae"])),
                "reused_from": str(src),
                "reuse_identity": {
                    "arm": A_EQUIVALENT[args.row],
                    "same_rows": "official Train3360 rows, identical row-id array",
                    "same_order": "np.random.default_rng(0) per-epoch permutations over Train",
                    "same_recipe": "FGCR-R0, 200 epochs, batch 64, AdamW 3e-4->1e-5, clip 5",
                    "same_loss": "MSE on the same official-Train standardized target",
                    "same_init": "FGCR seed-0 init state, no checkpoint fine-tuning",
                    "n_parameters": 100926,
                },
            }
            best_pred = np.asarray(preds["best_valid_y_pred"], dtype=np.float64)
            final_pred = np.asarray(preds["final_valid_y_pred"], dtype=np.float64)
            arm = None
            curves: list[dict] = []
        else:
            log(f"{args.row}/{target}: start")
            arm = _build_arm(ctx, family, operator, args.device, scaler,
                            (pair_a, pair_b), valid_labels, args.row)
            result = ST.train_one(arm=arm, target=target, train_rows=train_rows,
                                  order=order, schedule=schedule,
                                  labels_std=ctx.bundle.labels_std, scaler=scaler,
                                  eval_rows=valid_rows, epochs=epochs, out_dir=out_dir,
                                  log=log,
                                  checkpoint_extra={"row": args.row, "family": family,
                                                    "target": target,
                                                    "operator": (operator if isinstance(operator, str) else None),
                                                    "h_mode": (operator[0] if family == "FGCR" else None),
                                                    "eta_mode": (operator[1] if family == "FGCR" else None),
                                                    "phase": "P"})
            summary = dict(result["summary"])
            best_pred = result["best_pred_raw"]
            final_pred = result["final_pred_raw"]
            curves = result["curves"]
        summary.update({
            "row": args.row, "dataset": "ogbg-mollipo", "phase": "P", "device": args.device,
            "peak_vram_gb": (float(torch.cuda.max_memory_allocated() / 1024 ** 3)
                             if args.device.startswith("cuda") else 0.0),
            "scaler": scaler, "commit": C.git_commit(),
            "environment": C.environment_snapshot(), "finished_at": C.utc_stamp(),
            "schedule_sha256": C.sha256_array(schedule),
            "n_pairs": int(index.n), "n_pair_components": int(len(np.unique(comps))),
        })
        C.write_json(out_dir / "summary.json", summary)
        C.write_json(out_dir / "curves.json", {"curves": curves})
        np.savez_compressed(out_dir / "predictions.npz",
                            valid_row_ids=valid_rows, valid_y_true=valid_labels,
                            best_valid_y_pred=best_pred, final_valid_y_pred=final_pred)
        summaries[target] = summary
        log(f"{args.row}/{target}: done best_epoch={summary['best_epoch']} "
            f"valid_rmse={summary['best_eval_rmse']:.6f}")
        C.ledger_append({"phase": "P", "unit": f"{args.row}/{target}",
                         "device": args.device, "gpu_hours": summary["seconds"] / 3600.0,
                         "seconds": summary["seconds"], "status": "completed",
                         "finished_at": C.utc_stamp()})

    C.write_json(root / f"summary_{args.row}{'_preflight' if args.preflight else ''}.json", summaries)
    handle.close()
    return 0


def _build_arm(ctx, family, operator, device, scaler, pair_rows, valid_labels, row_name):
    meta = {"row": row_name, "dataset": "ogbg-mollipo", "phase": "P"}
    if family == "FGCR":
        return SX.make_fgcr_arm(ctx, h_mode=operator[0], eta_mode=operator[1],
                                init_state=SX.fgcr_init_state(ctx), device=device,
                                pair_rows=pair_rows, eval_labels_raw=valid_labels,
                                meta=meta)
    if family == "RESP":
        universe = np.concatenate([ctx.split_rows["train"], ctx.split_rows["valid"]])
        universe = np.asarray(universe, dtype=np.int64)
        bundle = SR.RespBundle(ctx.dataset, universe, scaler, split_name="train")
        arm, _ = SX.make_resp_arm(ctx, rows=universe, split_name="train", scaler=scaler,
                                  operator=operator,
                                  init_state=SR.canonical_init(bundle.dims),
                                  device=device, pair_rows=pair_rows,
                                  eval_labels_raw=valid_labels, fit_cache=bundle,
                                  meta=meta)
        return arm
    if family == "GINE":
        return SX.make_gine_arm(ctx,
                                init_state=SG.make_init_state(ctx.bundle.gine_categories,
                                                              seed=C.TRAIN_SEED),
                                device=device, pair_rows=pair_rows,
                                eval_labels_raw=valid_labels, meta=meta)
    raise KeyError(family)


if __name__ == "__main__":
    raise SystemExit(main())
