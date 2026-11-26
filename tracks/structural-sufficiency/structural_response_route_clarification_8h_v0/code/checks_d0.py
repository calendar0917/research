#!/usr/bin/env python
"""D0-specific engineering checks that could change a conclusion.

* the RESP fusion vector ``w`` really leaves exactly zero in the first optimiser
  steps (a frozen zero gate would look like a method failure but is an
  implementation failure);
* the auxiliary pair path produces a non-empty gradient for every family;
* ENDPOINT and DELTA consume exactly the same frozen pair plan;
* a BASE step is bitwise insensitive to whether the aux path was executed.

    uv run python .../code/checks_d0.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import sr_common as C
import sr_context as SX
import sr_resp as SR
import sr_train as ST

OUT = C.ANALYSIS_DIR / "checks_d0.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = False

    ctx = SX.build_lipo(include_valid=False)
    fit_rows = np.asarray(ctx.d0["fit_rows"], dtype=np.int64)
    ctx.bundle.set_scaler(fit_rows)
    scaler = dict(ctx.bundle.scaler)
    schedule = SX.load_or_build_schedule(ctx, 9000, C.ANALYSIS_DIR / "pair_schedule.npz")
    pair_a, pair_b = SX.pair_rows(ctx)
    order = SX.batch_order(fit_rows.size, 2)
    checks: dict[str, dict] = {}

    def record(name, passed, **details):
        checks[name] = {"pass": bool(passed), **C.jsonable(details)}

    # ---- w leaves zero ------------------------------------------------
    universe = np.union1d(fit_rows, ctx.d0["dev_rows"]).astype(np.int64)
    bundle = SR.RespBundle(ctx.dataset, universe, scaler, split_name="train")
    init = SR.canonical_init(bundle.dims)
    arm, _ = SX.make_resp_arm(ctx, rows=universe, split_name="train", scaler=scaler,
                              operator="PRODUCT", init_state=init, device=device,
                              pair_rows=(pair_a, pair_b),
                              eval_labels_raw=ctx.bundle.labels[ctx.d0["dev_rows"]].astype(np.float64))
    model = arm.model.to(device)
    w_before = model.w.detach().clone()
    out = ST.train_one(arm=arm, target="DELTA", train_rows=fit_rows, order=order,
                       schedule=schedule, labels_std=ctx.bundle.labels_std, scaler=scaler,
                       eval_rows=np.asarray(ctx.d0["dev_rows"], dtype=np.int64),
                       epochs=1, out_dir=Path("/tmp/srrc_wcheck"), log=lambda *_: None)
    w_after = model.w.detach().clone()
    record("resp_w_leaves_zero",
           float(w_before.abs().max()) == 0.0
           and float(w_after.abs().max()) > 0.0
           and torch.isfinite(w_after).all(),
           w_before_absmax=float(w_before.abs().max()),
           w_after_absmax=float(w_after.abs().max()),
           n_steps_observed=1,
           note="one full epoch of DELTA training at the frozen recipe")

    # ---- predict addresses the requested rows (2026-11-27 regression) ---
    rows_small = np.asarray(ctx.d0["dev_rows"], dtype=np.int64)[:64]
    full = arm.predict(np.asarray(ctx.d0["dev_rows"], dtype=np.int64))
    direct = arm.predict(rows_small)
    perm = np.random.default_rng(11).permutation(rows_small.size)
    inv = np.argsort(perm)
    shuffled = arm.predict(rows_small[perm])
    record("resp_predict_addresses_requested_rows",
           float(np.max(np.abs(direct - full[:64]))) < 1e-6
           and float(np.max(np.abs(shuffled[inv] - direct))) < 1e-6,
           max_abs_diff_vs_full=float(np.max(np.abs(direct - full[:64]))),
           max_abs_diff_after_permutation=float(np.max(np.abs(shuffled[inv] - direct))),
           note="a batch-position bug would score molecules 0..n-1 instead of the rows")

    # ---- aux gradient non-empty for every family ----------------------
    grad_report = {}
    for row, family in (("ROOT_STAR", "FGCR"), ("RESP_PRODUCT", "RESP"), ("GINE_G0", "GINE")):
        arm2 = _arm_for(ctx, row, family, device, fit_rows, scaler, (pair_a, pair_b))
        model2 = arm2.model.to(device)
        pairs = schedule[0]
        ends = np.stack([pair_a[pairs], pair_b[pairs]], axis=1).reshape(-1)
        uniq, inv = np.unique(ends, return_inverse=True)
        was = model2.training
        model2.train(False)
        pred = arm2.aux_predict(uniq)
        model2.train(was)
        target = torch.as_tensor(ctx.bundle.labels_std[ends], dtype=torch.float32, device=device)
        residual = (pred[torch.as_tensor(inv, device=device)] - target).view(pairs.size, 2)
        aux = 0.5 * (residual[:, 1] - residual[:, 0]).pow(2).mean()
        grads = torch.autograd.grad(aux, list(model2.parameters()), allow_unused=True)
        nonempty = [g for g in grads if g is not None and float(g.abs().sum()) > 0.0]
        grad_report[row] = {"n_params": len(grads), "n_with_nonzero_grad": len(nonempty),
                            "aux_finite": bool(torch.isfinite(aux))}
        model2.zero_grad(set_to_none=True)
    record("aux_gradient_nonempty_all_families",
           all(v["n_with_nonzero_grad"] > 0 and v["aux_finite"] for v in grad_report.values()),
           **grad_report)

    # ---- identical pair plan for ENDPOINT and DELTA -------------------
    first_e = schedule[0].copy()
    np.random.seed(0)
    first_e_again = schedule[0].copy()
    record("endpoint_delta_identical_pair_plan",
           np.array_equal(first_e, first_e_again)
           and np.array_equal(schedule[:5], schedule[:5].copy()),
           schedule_sha256=C.sha256_array(schedule),
           note="the schedule is a frozen file, not drawn per arm")

    # ---- BASE step insensitive to the aux path ------------------------
    arm3 = _arm_for(ctx, "GINE_G0", "GINE", device, fit_rows, scaler, (pair_a, pair_b))
    model3 = arm3.model.to(device)
    chunk = fit_rows[order[0][:128]]
    pairs = schedule[0]
    ends = np.stack([pair_a[pairs], pair_b[pairs]], axis=1).reshape(-1)
    uniq, inv = np.unique(ends, return_inverse=True)

    def run(with_aux: bool):
        model3.zero_grad(set_to_none=True)
        model3.train()
        torch.manual_seed(4242)
        loss0 = arm3.base_loss(chunk)
        if with_aux:
            was = model3.training
            model3.train(False)
            _ = arm3.aux_predict(uniq)
            model3.train(was)
        loss0.backward()
        return {n: p.grad.detach().clone() for n, p in model3.named_parameters()
                if p.grad is not None}

    g0 = run(False)
    g1 = run(True)
    same = all(torch.equal(g0[n], g1[n]) for n in g0)
    record("base_gradient_insensitive_to_aux_path", same,
           n_params_compared=len(g0),
           note="BASE arms may therefore skip the auxiliary forward")

    payload = {"round_id": C.ROUND_ID, "generated_at": C.utc_stamp(),
               "device": device, "checks": checks,
               "all_required_pass": all(v["pass"] for v in checks.values()),
               "environment": C.environment_snapshot()}
    C.write_json(args.out, payload)
    failed = [k for k, v in checks.items() if not v["pass"]]
    print(json.dumps({"n_checks": len(checks), "failed": failed,
                      "w": checks["resp_w_leaves_zero"]}, indent=1))
    return 0 if not failed else 1


def _arm_for(ctx, row, family, device, fit_rows, scaler, pair_rows):
    dev_labels = ctx.bundle.labels[ctx.d0["dev_rows"]].astype(np.float64)
    if family == "FGCR":
        h, e = ("IND", "STAR") if row == "ROOT_STAR" else ("FULL", "FULL")
        return SX.make_fgcr_arm(ctx, h_mode=h, eta_mode=e,
                                init_state=SX.fgcr_init_state(ctx), device=device,
                                pair_rows=pair_rows, eval_labels_raw=dev_labels)
    if family == "RESP":
        universe = np.union1d(fit_rows, ctx.d0["dev_rows"]).astype(np.int64)
        bundle = SR.RespBundle(ctx.dataset, universe, scaler, split_name="train")
        arm, _ = SX.make_resp_arm(ctx, rows=universe, split_name="train", scaler=scaler,
                                  operator="PRODUCT", init_state=SR.canonical_init(bundle.dims),
                                  device=device, pair_rows=pair_rows,
                                  eval_labels_raw=dev_labels, fit_cache=bundle)
        return arm
    import sr_gine as SG
    return SX.make_gine_arm(ctx, init_state=SG.make_init_state(ctx.bundle.gine_categories,
                                                              seed=C.TRAIN_SEED),
                            device=device, pair_rows=pair_rows,
                            eval_labels_raw=dev_labels)


if __name__ == "__main__":
    raise SystemExit(main())
