#!/usr/bin/env python
"""Phase C: the three frozen BASE rows on ESOL (Train902 -> Valid113).

Only BASE is trained; the three-target matrix is not repeated.  Each family keeps
its own frozen recipe and epoch count; only the legal ESOL categorical schema is
adapted (embedding tables).  No Lipo supervised weight is transferred.

    uv run python .../code/run_c.py --row ROOT_STAR --device cuda:0
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
import sr_fgcr as SF
import sr_gine as SG
import sr_resp as SR
import sr_train as ST

ROWS = {
    "ROOT_STAR": ("FGCR", ("IND", "STAR"), 200),
    "ROOT_FULL": ("FGCR", ("FULL", "FULL"), 200),
    "RESP_PRODUCT": ("RESP", "PRODUCT", 360),
    "RESP_FUSION_MLP": ("RESP", "FUSION_MLP", 360),
    "GINE_G0": ("GINE", None, 360),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", required=True, choices=sorted(ROWS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", default=str(C.RESULTS_DIR / "C"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    family, operator, epochs = ROWS[args.row]
    if args.preflight:
        epochs = 2
        args.out_root = str(C.RESULTS_DIR / "preflight_C")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    C.seed_everything(C.TRAIN_SEED)

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / f"c_{args.row}{'_preflight' if args.preflight else ''}.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    env = C.environment_snapshot()
    ctx = SX.build_esol()
    train_rows = np.asarray(ctx.split_rows["train"], dtype=np.int64)
    valid_rows = np.asarray(ctx.split_rows["valid"], dtype=np.int64)
    scaler = dict(ctx.bundle.scaler)
    log(f"row={args.row} family={family} operator={operator} device={args.device} "
        f"epochs={epochs} plan={json.dumps(C.jsonable(ctx.plan_stats))} scaler={scaler}")
    log(f"env={json.dumps(C.jsonable(env))}")
    order = SX.batch_order(train_rows.size, epochs)
    valid_labels = ctx.bundle.labels[valid_rows].astype(np.float64)

    out_dir = root / args.row
    if (out_dir / "summary.json").is_file():
        log("already complete")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    arm = _build_arm(ctx, family, operator, args.device, (train_rows, valid_labels))
    result = ST.train_one(arm=arm, target="BASE", train_rows=train_rows, order=order,
                          schedule=np.zeros((0, 64), dtype=np.int32),
                          labels_std=ctx.bundle.labels_std, scaler=scaler,
                          eval_rows=valid_rows, epochs=epochs, out_dir=out_dir,
                          log=log, checkpoint_extra={"row": args.row, "dataset": "ogbg-molesol",
                                                     "family": family,
                                                     "operator": (operator if isinstance(operator, str) else None),
                                                     "h_mode": (operator[0] if family == "FGCR" else None),
                                                     "eta_mode": (operator[1] if family == "FGCR" else None)})
    summary = dict(result["summary"])
    summary.update({
        "row": args.row, "dataset": "ogbg-molesol", "device": args.device,
        "gpu_name": (torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"),
        "peak_vram_gb": (float(torch.cuda.max_memory_allocated() / 1024 ** 3)
                         if args.device.startswith("cuda") else 0.0),
        "scaler": scaler, "commit": C.git_commit(), "dirty": C.git_dirty(),
        "environment": env, "finished_at": C.utc_stamp(),
        "plan_stats": ctx.plan_stats,
        "n_train": int(train_rows.size), "n_valid": int(valid_rows.size),
        "best_checkpoint_sha256": C.sha256_file(out_dir / "best.pt"),
    })
    C.write_json(out_dir / "summary.json", summary)
    C.write_json(out_dir / "curves.json", {"curves": result["curves"]})
    np.savez_compressed(out_dir / "predictions.npz",
                        valid_row_ids=valid_rows, valid_y_true=valid_labels,
                        best_valid_y_pred=result["best_pred_raw"],
                        final_valid_y_pred=result["final_pred_raw"],
                        train_row_ids=train_rows,
                        best_epoch=np.int64(summary["best_epoch"]),
                        scaler_mean=np.float64(scaler["mean"]),
                        scaler_std=np.float64(scaler["std"]))
    C.ledger_append({"phase": "C", "unit": args.row, "device": args.device,
                     "gpu_hours": summary["seconds"] / 3600.0, "seconds": summary["seconds"],
                     "peak_vram_gb": summary["peak_vram_gb"], "status": "completed",
                     "finished_at": C.utc_stamp()})
    log(f"done best_epoch={summary['best_epoch']} valid_rmse={summary['best_eval_rmse']:.6f}")
    handle.close()
    return 0


def _build_arm(ctx, family, operator, device, extra):
    train_rows, valid_labels = extra
    if family == "FGCR":
        model, info = SF.build_variant(ctx.bundle.atom_dims, ctx.bundle.bond_dims,
                                       h_mode=operator[0], eta_mode=operator[1],
                                       seed=C.TRAIN_SEED)
        init = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return SX.make_fgcr_arm(ctx, h_mode=operator[0], eta_mode=operator[1],
                                init_state=init, device=device,
                                eval_labels_raw=valid_labels,
                                meta={"row": "ESOL", "dataset": "ogbg-molesol"})
    if family == "RESP":
        bundle = SR.RespBundle(ctx.dataset, train_rows, ctx.scaler, split_name="train")
        init = SR.canonical_init(bundle.dims)
        arm, _ = SX.make_resp_arm(ctx, rows=train_rows, split_name="train",
                                  scaler=ctx.scaler, operator=operator,
                                  init_state=init, device=device,
                                  eval_labels_raw=valid_labels, fit_cache=bundle,
                                  meta={"row": "ESOL", "dataset": "ogbg-molesol"})
        return arm
    if family == "GINE":
        init = SG.make_init_state(ctx.bundle.gine_categories, seed=C.TRAIN_SEED)
        return SX.make_gine_arm(ctx, init_state=init, device=device,
                                eval_labels_raw=valid_labels,
                                meta={"row": "ESOL", "dataset": "ogbg-molesol"})
    raise KeyError(family)


if __name__ == "__main__":
    raise SystemExit(main())
