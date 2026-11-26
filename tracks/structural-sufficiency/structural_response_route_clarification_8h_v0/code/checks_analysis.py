#!/usr/bin/env python
"""Independent re-derivation of the phase-B edit numbers from the raw family APIs.

Guards the analysis against the class of bug fixed on 2026-11-27 (a predictor that
scored the wrong molecules): the deltas written into ``predictions/*.npz`` are
recomputed here from the checkpoints without using ``run_analysis``'s wrappers, and
the component-equal MAE in ``edit_response_table.csv`` is recomputed from the deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import sr_common as C
import sr_context as SX
import sr_fgcr as SF
import sr_gine as SG
import sr_resp as SR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "checks_analysis.json"))
    args = parser.parse_args()
    ctx = SX.build_lipo(include_valid=False)
    fit_rows = np.asarray(ctx.d0["fit_rows"], dtype=np.int64)
    dev_rows = np.asarray(ctx.d0["dev_rows"], dtype=np.int64)
    ctx.bundle.set_scaler(fit_rows)
    scaler = dict(ctx.bundle.scaler)
    labels = ctx.bundle.labels.astype(np.float64)
    index = ctx.pairs
    dev_dev = index.dev_dev_index
    component = index.component_ids
    y_a = labels[np.asarray([r["row_a"] for r in index.rows], dtype=np.int64)]
    y_b = labels[np.asarray([r["row_b"] for r in index.rows], dtype=np.int64)]

    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, **details) -> None:
        checks[name] = {"pass": bool(passed), **C.jsonable(details)}

    manifests = {e["row"] + "|" + e["target"]: e
                 for e in C.read_json(C.CKPT_DIR / "manifest.json")["entries"]
                 if e["checkpoint"] == "best"}
    table = list(csv.DictReader((C.ROUND_DIR / "edit_response_table.csv").open()))
    max_delta_diff = 0.0
    max_ce_diff = 0.0
    compared = 0
    for key, entry in sorted(manifests.items()):
        if entry["checkpoint"] != "best":
            continue
        row, target = key.split("|")
        path = C.RESULTS_DIR / "B" / row / target / "best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["extra"]
        # independent path: build the model, predict the union directly
        union = np.union1d(np.asarray([r["row_a"] for r in index.rows]),
                           np.asarray([r["row_b"] for r in index.rows])).astype(np.int64)
        if row.startswith("GINE"):
            model = SG.build_model(ctx.bundle.gine_categories, payload["state_dict"])
            model.eval()
            std = SG.predict_std(model, ctx.bundle, union, "cpu", batch_size=256)
        else:
            universe = np.union1d(fit_rows, dev_rows).astype(np.int64)
            bundle = SR.RespBundle(ctx.dataset, universe, scaler, split_name="train")
            model = SR.build_model(bundle.dims, SR.canonical_init(bundle.dims),
                                   meta["operator"])
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            std = SR.predict_std(model, bundle, bundle.fit, bundle.fit.local(union), "cpu",
                                 batch_size=256)
        raw = std * float(scaler["std"]) + float(scaler["mean"])
        pos_a = np.searchsorted(union, np.asarray([r["row_a"] for r in index.rows]))
        pos_b = np.searchsorted(union, np.asarray([r["row_b"] for r in index.rows]))
        delta_hat = raw[pos_b] - raw[pos_a]

        with np.load(C.PRED_DIR / f"edit_response__{row}__{target}__best.npz") as handle:
            stored = np.asarray(handle["delta_hat"], dtype=np.float64)
            stored_y = np.asarray(handle["delta_y"], dtype=np.float64)
        max_delta_diff = max(max_delta_diff, float(np.max(np.abs(stored - delta_hat))))
        record(f"delta_y_matches_labels__{row}__{target}",
               float(np.max(np.abs(stored_y - (y_b - y_a)))) < 1e-12,
               max_abs_diff=float(np.max(np.abs(stored_y - (y_b - y_a)))))
        ce, _ = C.component_equal(np.abs(delta_hat[dev_dev] - stored_y[dev_dev]),
                                  component[dev_dev])
        csv_row = [r for r in table if r["row"] == row and r["target"] == target
                   and r["checkpoint"] == "best" and r["subset"] == "dev_dev_all"]
        if csv_row:
            max_ce_diff = max(max_ce_diff, abs(ce - float(csv_row[0]["ce_edit_mae"])))
        compared += 1

    record("deltas_reproduce_from_checkpoints_independently",
           max_delta_diff < 1e-5, units_compared=compared,
           max_abs_delta_diff=max_delta_diff,
           max_abs_ce_diff=max_ce_diff,
           note="recomputed without the run_analysis predictor wrappers")
    record("component_equal_table_reproduces", max_ce_diff < 1e-9,
           max_abs_ce_diff=max_ce_diff)

    payload = {"round_id": C.ROUND_ID, "generated_at": C.utc_stamp(), "checks": checks,
               "all_required_pass": all(v["pass"] for v in checks.values())}
    C.write_json(args.out, payload)
    print(json.dumps({"n_checks": len(checks),
                      "failed": [k for k, v in checks.items() if not v["pass"]],
                      "max_abs_delta_diff": max_delta_diff,
                      "max_abs_ce_diff": max_ce_diff, "units": compared}, indent=1))
    return 0 if payload["all_required_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
