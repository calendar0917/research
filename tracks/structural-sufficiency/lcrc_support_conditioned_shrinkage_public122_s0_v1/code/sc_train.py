"""Update-driven trainer for one (dataset, fit, arm) trajectory of this round.

Reused from the frozen diagnosis round (``lb_train``/``lb_data``/``lb_selection``):
the epoch-internal ``randperm`` batch plan, the shared initial parameters, the
shared structure/target norms, the RNG-safe evaluation, the 50-point common grid
and the three pre-registered selection rules.  New here:

* the arm chooses either the frozen C1 operator (``BASE``) or the gated joint
  block with a **frozen per-root alpha** loaded from the committed support table;
* every completed *fit* epoch is evaluated on the full ``train_fit`` / Valid /
  Test, and the common 50-point grid is evaluated on the same three splits;
* ``alpha`` never touches the RNG stream and adds no trainable parameter.

Run one arm:

    uv run python code/sc_train.py --dataset Lipophilicity --fit 100 --arm SELECT \
        --device cuda --out-dir runs/LIP_F100_SELECT
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import sc_common as C
import sc_model as M
import sc_support as S

import lb_train as LB
import lb_selection as LS

import lcpr_train as PT

PROCESS_T0 = time.perf_counter()

split_metrics = PT.split_metrics
rng_snapshot = PT.rng_snapshot
rng_restore = PT.rng_restore


def arm_dir_name(dataset: str, fit: str, arm: str) -> str:
    tag = "LIP" if dataset == C.PRIMARY_DATASET else "BACE"
    return f"{tag}_F{fit}_{arm}"


# ---------------------------------------------------------------------------
# runner with the frozen alpha attached to every batch
# ---------------------------------------------------------------------------


class GateRunner(LB.C1Runner):
    """``C1Runner`` that attaches ``root_alpha`` (per atom row) to every batch."""

    def __init__(self, model, device, caches, alpha: Mapping[str, np.ndarray | None]):
        super().__init__(model, device, caches)
        self.alpha = dict(alpha)

    def batch(self, split, positions):
        batch = super().batch(split, positions)
        a = self.alpha.get(split)
        if a is not None:
            atom_ptr = np.asarray(self.caches[split]["atom_ptr"], dtype=np.int64)
            rows = [int(r) for r in positions]
            vals = np.concatenate([np.asarray(a)[atom_ptr[r]: atom_ptr[r + 1]] for r in rows]) \
                if rows else np.zeros(0, np.float32)
            batch["root_alpha"] = torch.from_numpy(np.ascontiguousarray(vals, dtype=np.float32))
        return batch


# ---------------------------------------------------------------------------
# recipe / budget
# ---------------------------------------------------------------------------


def recipe_for(dataset: str) -> dict[str, Any]:
    return dict(C.C1_R3 if dataset == C.PRIMARY_DATASET else C.C1_R1)


def train_arm(*, dataset: str, fit: str, arm: str, device: torch.device, out_dir: Path,
              key: str | None = None, max_updates_override: int | None = None,
              smoke: bool = False) -> dict[str, Any]:
    if arm not in C.ARMS:
        raise ValueError(arm)
    if not smoke and C.git_dirty():
        raise SystemExit("formal run requested on a dirty tracked worktree; commit first")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = C.dataset_meta(dataset)["task"]
    recipe = recipe_for(dataset)
    bs = int(recipe["batch_size"])

    spec = C.LD.frozen_spec()
    tgt = C.LD.targets(dataset, include_test=True)
    caches = {s: C.LD.load_cache(dataset, s, tgt["rows"][s]) for s in C.SPLITS}
    subset = C.LD.nested_subset_positions(dataset, fit)
    fit_positions = subset["positions"]
    n_fit = len(fit_positions)
    snorm = C.LD.structure_norm(dataset)
    tnorm_info = C.LD.shared_target_norm(dataset)
    tnorm = tnorm_info["target_norm"]
    y_fit_all = np.asarray(tgt["y"]["train"], np.float64).reshape(-1)
    y_fit = y_fit_all[np.asarray(fit_positions, np.int64)]
    ftrain_rows = [int(tgt["rows"]["train"][p]) for p in fit_positions]
    valid_rows_arr = np.asarray([int(x) for x in tgt["rows"]["valid"]], np.int64)
    test_rows_arr = np.asarray([int(x) for x in tgt["rows"]["test"]], np.int64)
    fit_rows_arr = np.asarray(ftrain_rows, np.int64)
    y_valid = np.asarray(tgt["y"]["valid"], np.float64).reshape(-1)
    y_test = np.asarray(tgt["y"]["test"], np.float64).reshape(-1)

    U0 = C.u0(dataset)
    Umax = U0 if not smoke else C.SMOKE_MAX_UPDATES
    if max_updates_override is not None:
        Umax = int(max_updates_override)

    # ---- frozen support prior -------------------------------------------
    support_info: dict[str, Any] = {}
    alpha: dict[str, np.ndarray | None] = {s: None for s in C.SPLITS}
    if arm != C.ARM_BASE:
        if key is None:
            raise ValueError("a gated arm needs the frozen key")
        support = S.load_support(dataset, fit, key)
        for s in C.SPLITS:
            alpha[s] = S.alpha_for_split(support, arm, s)
        active = {s: np.asarray(support[f"active_{s}"]) for s in C.SPLITS}
        dist = {}
        for s in C.SPLITS:
            a_s = np.asarray(alpha[s], dtype=np.float64)
            act = active[s]
            dist[s] = {
                "n_roots": int(a_s.size), "n_active": int(act.sum()),
                "active_mean": float(a_s[act].mean()) if act.any() else None,
                "active_min": float(a_s[act].min()) if act.any() else None,
                "active_max": float(a_s[act].max()) if act.any() else None,
                "all_mean": float(a_s.mean()),
                "frac_at_one": float((a_s >= 1.0 - 1e-12).mean()),
            }
        support_info = {"key": key, "path": support["sha256"], "distribution": dist,
                        "a_oof": float(support["a_oof"]),
                        "fit_mean_a_select": float(support["fit_mean_a_select"]),
                        "n_support_structures": int(support["n_support"])}

    # ---- model ----------------------------------------------------------
    model_obj = M.build_arm_model(spec, snorm, arm)
    init_report = M.load_shared_init(model_obj, spec, snorm)
    init_report.pop("base_state", None)
    model_obj.to(device)
    runner = GateRunner(model_obj, device, caches, alpha)
    n_params = int(sum(p.numel() for p in model_obj.parameters()))
    if n_params != C.C1_PARAMS:
        raise RuntimeError(f"{arm}: parameters {n_params} != frozen C1 {C.C1_PARAMS}")
    init_state_hash = C.PC.DM.state_fingerprint({k: v.detach().cpu()
                                               for k, v in model_obj.state_dict().items()})

    torch.manual_seed(int(C.TRAIN_SEED))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(C.TRAIN_SEED))

    optimizer = torch.optim.AdamW(
        [p for p in model_obj.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]), betas=tuple(recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]), weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(U0, 1), eta_min=float(recipe["min_lr"]))

    plan = LB.BatchPlan(n_fit, bs)
    grid = [int(round(j * U0 / C.GRID_POINTS)) for j in range(1, C.GRID_POINTS + 1)]
    grid_set = set(grid)

    selector = LS.ProbeSelector(task)
    curves: list[dict[str, Any]] = []
    grid_preds: dict[str, list[np.ndarray]] = {"grid_updates": []}

    def strip(ev):
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in ev.items()}

    def eval_all():
        was = model_obj.training
        snap = rng_snapshot(device)
        model_obj.eval()
        try:
            p_fit = runner.predict("train", fit_positions)
            p_valid = runner.predict("valid")
            p_test = runner.predict("test")
        finally:
            rng_restore(snap, device)
            model_obj.train(was)
        return {
            "train_fit": {**split_metrics(task, p_fit, y_fit, tnorm), "_pred": p_fit,
                          "_rows": fit_rows_arr, "_y": y_fit},
            "valid": {**split_metrics(task, p_valid, y_valid, tnorm), "_pred": p_valid,
                      "_rows": valid_rows_arr, "_y": y_valid},
            "test": {**split_metrics(task, p_test, y_test, tnorm), "_pred": p_test,
                     "_rows": test_rows_arr, "_y": y_test},
        }

    train_seconds = eval_seconds = 0.0
    sample_presentations = partial_batches = 0

    def do_eval(update: int, *, on_grid: bool, is_epoch_end: bool) -> None:
        nonlocal eval_seconds
        t = time.perf_counter()
        ev = eval_all()
        eval_seconds += time.perf_counter() - t
        s = strip(ev)
        rec = {"update": int(update), "is_grid": bool(on_grid),
               "is_epoch_end": bool(is_epoch_end)}
        for name in ("train_fit", "valid", "test"):
            for k, v in s[name].items():
                if isinstance(v, (int, float)):
                    rec[f"{name}_{k}"] = float(v)
        curves.append(rec)
        if on_grid:
            selector.consider(update, s["valid"], model_obj.state_dict())
            grid_preds["grid_updates"].append(int(update))
            for name in ("train_fit", "valid", "test"):
                grid_preds.setdefault(name, []).append(np.asarray(ev[name]["_pred"], np.float64))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    t0 = time.perf_counter()
    ev0 = eval_all()
    eval_seconds += time.perf_counter() - t0
    s0 = strip(ev0)
    step0 = {
        "update": 0, "valid_metrics": s0["valid"], "train_fit_metrics": s0["train_fit"],
        "test_metrics": s0["test"],
        "state": {k: v.detach().cpu().clone() for k, v in model_obj.state_dict().items()},
        "rng": {"torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "numpy": np.random.get_state()},
    }
    torch.save(step0, out_dir / "step0.pt")
    curves.append({"update": 0, "is_grid": False, "is_epoch_end": True,
                   **{f"{n}_{k}": float(v) for n in ("train_fit", "valid", "test")
                      for k, v in s0[n].items() if isinstance(v, (int, float))}})

    init_dose = None
    if arm != C.ARM_BASE and smoke:
        init_dose = None
    elif arm != C.ARM_BASE:
        init_dose = dose_diagnostics(model_obj, runner, fit_positions, device,
                                     dataset=dataset, fit=fit, key=key)

    update = 0
    while update < Umax:
        model_obj.train()
        last = plan.last_of_epoch()
        rows_local = plan.next_batch()
        update += 1
        sample_presentations += len(rows_local)
        if len(rows_local) < bs:
            partial_batches += 1
        t0 = time.perf_counter()
        pred = runner.forward(runner.batch("train", [fit_positions[i] for i in rows_local]))
        if task == "regression":
            z = torch.as_tensor((y_fit[rows_local] - float(tnorm["mean"])) / float(tnorm["scale"]),
                                dtype=torch.float32, device=device)
            loss = F.mse_loss(pred, z)
        else:
            y = torch.as_tensor(y_fit[rows_local], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_obj.parameters(), float(recipe["gradient_clip"]))
        optimizer.step()
        train_seconds += time.perf_counter() - t0
        scheduler.step()

        if update in grid_set:
            do_eval(update, on_grid=True, is_epoch_end=False)
        elif last:
            do_eval(update, on_grid=False, is_epoch_end=True)

    if update not in {r["update"] for r in curves}:
        do_eval(update, on_grid=False, is_epoch_end=True)
    wall = time.perf_counter() - started

    # ---- dose at the final weights (measured, never used for selection) ---
    dose: dict[str, Any] = {}
    if arm != C.ARM_BASE:
        dose["final"] = dose_diagnostics(model_obj, runner, fit_positions, device,
                                         dataset=dataset, fit=fit, key=key)

    rules = selector.freeze()
    retained: dict[str, Any] = {}
    for rule, rec in rules.items():
        if rec["has_state"]:
            path = out_dir / f"best_{rule}.pt"
            torch.save({"state_dict": rec["state"], "update": rec["update"],
                        "criterion": rec["criterion"], "rule": rule,
                        "arm": arm, "dataset": dataset, "fit": fit, "key": key}, path)
            retained[rule] = {"update": rec["update"], "criterion": rec["criterion"],
                              "path": C.track_rel(path)}
        else:
            retained[rule] = {"update": None, "criterion": None, "path": None}
    rules_json = {r: {k: v for k, v in rec.items() if k != "state"} for r, rec in rules.items()}
    torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model_obj.state_dict().items()},
                "update": int(update)}, out_dir / "last.pt")

    if arm != C.ARM_BASE:
        sel_state = rules[C.R_SEL]["state"]
        if sel_state is not None:
            final_state = {k: v.detach().clone() for k, v in model_obj.state_dict().items()}
            model_obj.load_state_dict({k: v.to(device) for k, v in sel_state.items()})
            dose["selected"] = dose_diagnostics(model_obj, runner, fit_positions, device,
                                                dataset=dataset, fit=fit, key=key)
            dose["selected"]["update"] = int(rules[C.R_SEL]["update"])
            model_obj.load_state_dict(final_state)
        dose["init"] = init_dose

    np.savez_compressed(
        out_dir / "grid_predictions.npz",
        grid_updates=np.asarray(grid_preds["grid_updates"], np.int64),
        fit_rows=fit_rows_arr, valid_rows=valid_rows_arr, test_rows=test_rows_arr,
        fit_y=y_fit, valid_y=y_valid, test_y=y_test,
        **{name: (np.stack(grid_preds[name], axis=0) if grid_preds.get(name) else
                  np.zeros((0, len(fit_rows_arr if name == "train_fit" else
                                   (valid_rows_arr if name == "valid" else test_rows_arr))),
                           np.float64))
           for name in ("train_fit", "valid", "test")},
    )
    all_fields: list[str] = []
    for row in curves:
        for k in row:
            if k not in all_fields:
                all_fields.append(k)
    with open(out_dir / "curve.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_fields)
        w.writeheader()
        w.writerows(curves)

    payload = {
        "schema": "lcrc-support-conditioned-shrinkage-train-v1",
        "round_id": C.ROUND_ID, "protocol_id": C.PROTOCOL_ID, "status": "completed",
        "smoke": bool(smoke), "arm_id": arm_dir_name(dataset, fit, arm),
        "arm": arm, "dataset": dataset, "task": task, "fit": fit, "key": key,
        "test_use": C.TEST_USE, "status_class": C.STATUS_DEV_TEST,
        "subset": {"n": n_fit, "positions_sha256": subset["positions_sha256"],
                   "global_rows_sha256": subset["global_rows_sha256"]},
        "data": {"n_train_official": C.official_train_n(dataset), "n_fit": n_fit,
                 "n_valid": int(valid_rows_arr.size), "n_test": int(test_rows_arr.size),
                 "valid_rows_sha256": C.row_sha256(valid_rows_arr),
                 "test_rows_sha256": C.row_sha256(test_rows_arr),
                 "structure_norm_source": "all_official_train_structures",
                 "target_norm": tnorm,
                 "shared_norms_fingerprint": C.LD.shared_norms_fingerprint(dataset)},
        "model_info": {"parameters": n_params, "init_state_hash": init_state_hash,
                       "init": init_report, "gated": bool(model_obj.gated)},
        "support": support_info,
        "recipe": {**recipe, "U0": int(U0), "Umax": int(Umax),
                   "steps_per_epoch": int(plan.per_epoch), "batch_size": bs,
                   "batch_plan_fingerprint": plan.fingerprint(),
                   "selection_rules": list(C.SELECTION_RULES),
                   "grid_points": C.GRID_POINTS, "window": C.WINDOW},
        "selection": {"rules": rules_json, "retained": retained},
        "dose": dose,
        "cost": {"train_seconds": float(train_seconds), "eval_seconds": float(eval_seconds),
                 "wall_seconds": float(wall),
                 "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
                 "updates_run": int(update), "sample_presentations": int(sample_presentations),
                 "partial_batches": int(partial_batches),
                 "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0)
                 if device.type == "cuda" else 0.0},
        "environment": {**C.env_info(), "device": str(device)},
        "provenance": {"git_commit": C.git_commit(), "code_hashes": C.code_hashes()},
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(f"[{payload['arm_id']}] Umax={Umax} n_fit={n_fit} params={n_params} "
          f"train={train_seconds:.1f}s eval={eval_seconds:.1f}s wall={wall:.1f}s "
          f"R_SEL={rules_json[C.R_SEL]['update']}")
    return payload


# ---------------------------------------------------------------------------
# dose / intervention diagnostics (measured, never used for selection)
# ---------------------------------------------------------------------------


@torch.no_grad()
def dose_diagnostics(model_obj, runner, fit_positions, device, *,
                     key: str | None, dataset: str, fit: str,
                     max_molecules: int = 64) -> dict[str, Any]:
    """Measured J doses on the *selected* weights, per molecule then aggregated.

    ``||J_full - J_marg||``, ``||(1-a)(J_full - J_marg)||`` and the ratio to
    ``||J_full||`` are recomputed from the frozen prior and the trained weights.
    This is a forward-only diagnostic; it produces no new molecule and no label.
    """
    was = model_obj.training
    model_obj.eval()
    out: dict[str, Any] = {"definition": "per-molecule norms on fixed weights, then distribution",
                           "max_molecules": int(max_molecules)}
    try:
        rows = list(fit_positions)[: int(max_molecules)]
        batch = runner.batch("train", rows)
        a_atoms, b_atoms, scal = model_obj._encode(batch)
        local = model_obj.local
        j_full, j_marg = local.joint_parts(batch, a_atoms)
        atom_ptr = np.asarray(runner.caches["train"]["atom_ptr"], dtype=np.int64)
        idx = np.asarray(rows, dtype=np.int64)
        per_row = atom_ptr[idx + 1] - atom_ptr[idx]
        root_of_atom = np.repeat(np.arange(len(rows)), per_row)
        alpha = batch.get("root_alpha")
        diff = (j_full - j_marg).detach().double().cpu().numpy()
        full = j_full.detach().double().cpu().numpy()
        a_np = None if alpha is None else alpha.detach().double().cpu().numpy()

        per_mol = {"n_molecules": len(rows),
                   "norm_diff": [], "norm_scaled": [], "ratio_scaled_full": [],
                   "norm_full": [], "mean_a_active": []}
        for m in range(len(rows)):
            sel = root_of_atom == m
            d = np.linalg.norm(diff[sel], axis=1)          # per root
            f = np.linalg.norm(full[sel], axis=1)
            per_mol["norm_diff"].append(float(np.linalg.norm(d)))
            per_mol["norm_full"].append(float(np.linalg.norm(f)))
            if a_np is not None:
                scaled = (1.0 - a_np[sel])[:, None] * diff[sel]
                per_mol["norm_scaled"].append(float(np.linalg.norm(scaled)))
                denom = float(np.linalg.norm(f)) if np.linalg.norm(f) > 0 else 1.0
                per_mol["ratio_scaled_full"].append(float(np.linalg.norm(scaled) / denom))
                per_mol["mean_a_active"].append(float(a_np[sel].mean()))

        def _stats(xs):
            arr = np.asarray(xs, dtype=np.float64)
            if arr.size == 0:
                return {"n": 0}
            return {"n": int(arr.size), "mean": float(arr.mean()), "p50": float(np.median(arr)),
                    "p05": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)),
                    "max": float(arr.max()), "min": float(arr.min()),
                    "frac_zero": float((arr == 0).mean())}

        out["molecules"] = {k: _stats(v) for k, v in per_mol.items()
                            if k != "n_molecules"}
        out["n_molecules"] = per_mol["n_molecules"]
        out["n_roots"] = int(diff.shape[0])
        out["n_roots_with_effect"] = int((np.linalg.norm(diff, axis=1) > 0).sum())
    finally:
        model_obj.train(was)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="support-conditioned shrinkage: one arm")
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--arm", required=True, choices=list(C.ARMS))
    ap.add_argument("--key", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    key = args.key
    if key is None and args.arm != C.ARM_BASE:
        key = C.read_json(C.ANALYSIS / "key_selection.json")["chosen_key"]
    out = Path(args.out_dir) if args.out_dir else C.RUNS / arm_dir_name(args.dataset, args.fit, args.arm)
    train_arm(dataset=args.dataset, fit=args.fit, arm=args.arm, device=device, out_dir=out,
              key=key, max_updates_override=args.max_updates, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
