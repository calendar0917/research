"""One formal training run for the semantic-table round.

Supports:

* families ``LS`` / ``FS`` / ``LP`` / ``FP`` (semantic-table arms) and ``B``
  (the frozen C1 base arm, used only for the wd=1e-2 recipe coordinate),
* datasets ``lipo`` (official scaffold, full Train or a nested Fit subset) and
  ``esol`` (official ``ogbg-molesol`` scaffold),
* recipes ``R`` (wd 1e-3) / ``W0`` (wd 0) / ``Rwd`` (wd 1e-2).

Discipline (unchanged from the pilot round):

* one shared per-epoch row plan (``torch.Generator(seed + 917000)``) per
  ``(n_fit, epochs)``; every paired model with the same n_fit/epochs consumes the
  same plan and the same non-table initial state;
* a fixed update budget ``U = 360 * ceil(N_full / 128)`` for every budget: the
  last (possibly incomplete) epoch stops at ``U``;
* every completed Fit epoch evaluates the **full** official Train / Valid /
  Test; plus the shared 50-point update grid ``round(j*U/50)`` is evaluated;
* selection is Validation-only (grid for budget runs, best-epoch as auxiliary);
* Test is reported and this round is authorised to read it for route judgement.

Artifacts: ``curve.csv``, ``metrics.json``, ``checkpoint_{best,last}.pt``,
``pred_{split}_{best,last}.npz``, ``grid.json``, ``resume.pt`` (removed on
success).  Generator arms additionally export an equivalent pure-lookup state.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import se_common as C
import se_tables as T
import se_model as M
import se_data as SED

import lp_common as LPC
import lp_data as LPD
import dissect_model as DM
import lcrc_data as PREP_DATA

PROCESS_T0 = time.perf_counter()


class BestTracker:
    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_rmse = math.inf
        self.state: dict[str, torch.Tensor] | None = None
        self.improvements: list[int] = []

    def consider(self, epoch: int, rmse: float) -> bool:
        if self.best_epoch is not None and rmse >= self.best_rmse:
            return False
        self.best_epoch = int(epoch)
        self.best_rmse = float(rmse)
        self.improvements.append(int(epoch))
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_model(cfg: Mapping[str, Any], view, permutations) -> torch.nn.Module:
    fam = cfg["family"]
    if fam == "B":
        import lp_model as LPM

        model = LPM.build_arm(view.spec, view.train_norm, "B")
        return model
    return M.build_semantic_model(view.spec, view.train_norm, fam, permutations=permutations)


def _view_for(cfg: Mapping[str, Any]):
    if cfg["dataset"] == "lipo":
        base = LPD.build_view()
        return SED.lipo_budget_view(cfg["budget"], base=base), base
    if cfg["dataset"] == "esol":
        view = SED.build_esol_view()
        return view, view
    raise ValueError(cfg["dataset"])


def train_run(cfg: Mapping[str, Any], *, device: torch.device, out_dir: Path,
              epochs_override: int | None = None, smoke: bool = False,
              resume: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LPC.setup_determinism()
    view, base_view = _view_for(cfg)
    perms = T.build_permutations()

    batch_size = int(LPC.C1_RECIPE["batch_size"])
    if isinstance(view, SED.BudgetView):
        n_fit = len(view.fit_rows)
        plan_epochs = 360 if epochs_override is None else int(epochs_override)
        U = 360 * math.ceil(int(base_view.caches["train"]["n_molecules"]) / batch_size)
        steps_per_epoch = math.ceil(n_fit / batch_size)
        planned_epochs = math.ceil(U / steps_per_epoch)
    else:
        n_fit = int(view.caches["train"]["n_molecules"])
        U = 360 * math.ceil(n_fit / batch_size)
        steps_per_epoch = math.ceil(n_fit / batch_size)
        planned_epochs = 360
    if epochs_override is not None and not isinstance(view, SED.BudgetView):
        planned_epochs = int(epochs_override)
        U = steps_per_epoch * planned_epochs
    if smoke:
        planned_epochs = min(planned_epochs, 2)
        U = min(U, steps_per_epoch * planned_epochs)
    total_updates = int(U)

    # ---- model ----------------------------------------------------------
    model = build_model(cfg, view, perms)
    params_total = int(sum(p.numel() for p in model.parameters()))
    non_model_params = 0
    init_state_hash = LPD.state_hash(model.state_dict())
    model.to(device)

    # ---- optimiser / schedule -------------------------------------------
    recipe = dict(LPC.C1_RECIPE)
    recipe["weight_decay"] = float(C.RECIPES[cfg["recipe"]]["weight_decay"])
    torch.manual_seed(int(LPC.TRAIN_SEED))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(LPC.TRAIN_SEED))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(recipe["peak_lr"]),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]), weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(recipe["min_lr"])
    )

    # ---- plan -----------------------------------------------------------
    plan = LPD.epoch_plan(n_fit, planned_epochs, int(LPC.TRAIN_SEED))

    if isinstance(view, SED.BudgetView):
        y_fit_scaled = view.y_fit_scaled().to(device)
        fit_rows = view.fit_positions
    else:
        y_fit_scaled = None
        fit_rows = np.arange(n_fit, dtype=np.int64)

    grid_values = sorted({int(round(j * total_updates / 50.0)) for j in range(1, 51)})

    def predict_cached(split: str) -> np.ndarray:
        return LPD.predict_cache(model, view.caches[split], device)

    # ---- resume ---------------------------------------------------------
    tracker = BestTracker()
    curve: list[dict[str, Any]] = []
    grid: list[dict[str, Any]] = []
    start_epoch = 1
    global_updates = 0
    grid_ptr = 0
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak = 0
    resume_path = out_dir / "resume.pt"
    if resume and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in payload["model"].items()}, strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        curve = payload["curve"]
        grid = payload["grid"]
        start_epoch = int(payload["epoch"]) + 1
        global_updates = int(payload["global_updates"])
        grid_ptr = int(payload["grid_ptr"])
        tracker.best_epoch = payload["best_epoch"]
        tracker.best_rmse = payload["best_rmse"]
        tracker.improvements = payload["improvements"]
        tracker.state = payload["best_state"]
        clip_events = int(payload.get("clip_events", 0))
        grad_norm_max = float(payload.get("grad_norm_max", 0.0))
        C.log(f"resume at epoch {start_epoch} updates={global_updates}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    def snapshot() -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def full_eval() -> dict[str, dict[str, float]]:
        snap = LPD.rng_snapshot(device)
        metrics = {}
        try:
            for split in LPC.SPLITS:
                metrics[split] = LPD.raw_metrics(predict_cached(split), view.y[split], view.target_norm)
        finally:
            LPD.rng_restore(snap, device)
        return metrics

    started = time.perf_counter()
    for epoch in range(start_epoch, planned_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = np.asarray(plan["plan"][epoch - 1], dtype=np.int64)
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        abs_sum = 0.0
        sq_sum = 0.0
        seen = 0
        epoch_clips = 0
        epoch_gnorm_max = 0.0
        for rows in batches:
            if global_updates >= total_updates:
                break
            batch = LPD.to_device(PREP_DATA.collate(view.caches["train"], rows), device) if not isinstance(view, SED.BudgetView) else LPD.to_device(PREP_DATA.collate(view.caches["train"], fit_rows[rows]), device)
            if isinstance(view, SED.BudgetView):
                y = y_fit_scaled[torch.as_tensor(rows, device=device)]
            else:
                y = view.y_scaled("train").to(device)[torch.as_tensor(rows, device=device)]
            pred = model(batch).reshape(-1)
            loss = F.mse_loss(pred, y)
            err = pred - y
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
                epoch_clips += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            epoch_gnorm_max = max(epoch_gnorm_max, gnorm)
            optimizer.step()
            scheduler.step()
            global_updates += 1
            seen += int(y.numel())
            # grid evaluation at fixed update points
            while grid_ptr < len(grid_values) and global_updates >= grid_values[grid_ptr]:
                gm = full_eval()
                grid.append({
                    "j": grid_ptr + 1,
                    "update": int(global_updates),
                    "target_update": int(grid_values[grid_ptr]),
                    "valid_rmse": float(gm["valid"]["rmse"]),
                    "test_rmse": float(gm["test"]["rmse"]),
                    "train_rmse": float(gm["train"]["rmse"]),
                    "state_hash": LPD.state_hash(model.state_dict()),
                })
                grid_ptr += 1
        train_seconds = time.perf_counter() - t0

        t_eval = time.perf_counter()
        metrics = full_eval()
        eval_seconds = time.perf_counter() - t_eval

        improved = tracker.consider(epoch, float(metrics["valid"]["rmse"]))
        if improved:
            tracker.capture(model)
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))
        train_mse = float(sq_sum / max(seen, 1))
        curve.append({
            "epoch": epoch,
            "global_updates": int(global_updates),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm_max": float(epoch_gnorm_max),
            "grad_clip_events": int(epoch_clips),
            "train_seconds": float(train_seconds),
            "eval_seconds": float(eval_seconds),
            "is_best": bool(improved),
            "train_mse_standardized_online": train_mse,
            "train_rmse_standardized_online": math.sqrt(train_mse),
            "train_mae_standardized_online": float(abs_sum / max(seen, 1)),
            "state_hash": LPD.state_hash(model.state_dict()),
            **{f"{split}_rmse": float(metrics[split]["rmse"]) for split in LPC.SPLITS},
            **{f"{split}_mae": float(metrics[split]["mae"]) for split in LPC.SPLITS},
            **{f"{split}_bias": float(metrics[split]["bias"]) for split in LPC.SPLITS},
            **{f"{split}_r2": float(metrics[split]["r2"]) for split in LPC.SPLITS},
        })
        if epoch % 20 == 0 or epoch == planned_epochs or improved or global_updates >= total_updates:
            torch.save({
                "model": snapshot(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "curve": curve,
                "grid": grid, "global_updates": global_updates, "grid_ptr": grid_ptr,
                "best_epoch": tracker.best_epoch, "best_rmse": tracker.best_rmse,
                "best_state": tracker.state, "improvements": tracker.improvements,
                "clip_events": clip_events, "grad_norm_max": grad_norm_max,
            }, resume_path)
        if global_updates >= total_updates:
            break

    wall = time.perf_counter() - started
    if tracker.best_epoch is None or tracker.state is None:
        raise RuntimeError("training ended without any validation improvement")

    sync_plan = LPD.rng_snapshot(device)

    def eval_state(state: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        snap = LPD.rng_snapshot(device)
        try:
            return {split: predict_cached(split) for split in LPC.SPLITS}
        finally:
            LPD.rng_restore(snap, device)

    last_state = snapshot()
    best_preds = eval_state(tracker.state)
    best_metrics = {split: LPD.raw_metrics(best_preds[split], view.y[split], view.target_norm) for split in LPC.SPLITS}
    if abs(best_metrics["valid"]["rmse"] - tracker.best_rmse) > LPC.REPRO_TOL:
        raise RuntimeError("reloaded best Valid RMSE differs")
    last_preds = eval_state(last_state)
    last_metrics = {split: LPD.raw_metrics(last_preds[split], view.y[split], view.target_norm) for split in LPC.SPLITS}

    # grid selection (Validation-only), ties -> earliest
    sel = None
    if grid:
        best_j = min(range(len(grid)), key=lambda i: (grid[i]["valid_rmse"], grid[i]["j"]))
        sel = {"grid_index": int(best_j), "j": int(grid[best_j]["j"]), "update": int(grid[best_j]["update"]),
               "valid_rmse": float(grid[best_j]["valid_rmse"]), "test_rmse": float(grid[best_j]["test_rmse"])}
    LPD.rng_restore(sync_plan, device)

    # ---- artifacts ------------------------------------------------------
    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    C_dir_grid = out_dir / "grid.json"
    import json as _json
    C_dir_grid.write_text(_json.dumps({"grid_values": grid_values, "points": grid, "selection": sel}, indent=2) + "\n")

    export: dict[str, Any] = {}
    for tag, state, preds, metrics in (("best", tracker.state, best_preds, best_metrics), ("last", last_state, last_preds, last_metrics)):
        torch.save({
            "state_dict": state, "family": cfg["family"], "recipe": cfg["recipe"],
            "budget": cfg["budget"], "dataset": cfg["dataset"], "seed": int(LPC.TRAIN_SEED),
            "protocol_id": C.PROTOCOL_ID, "checkpoint": tag,
            "best_epoch": int(tracker.best_epoch),
            "state_hash": LPD.state_hash(state), "init_state_hash": init_state_hash,
            "parameters": params_total, "target_norm": dict(view.target_norm),
            "epoch_plan_sha256": plan["sha256"],
            "metrics": {split: dict(metrics[split]) for split in LPC.SPLITS},
        }, out_dir / f"checkpoint_{tag}.pt")
        for split in LPC.SPLITS:
            np.savez(out_dir / f"pred_{split}_{tag}.npz",
                     rows=np.asarray(view.rows[split], dtype=np.int64),
                     rows_sha256=LPC.row_sha256(view.rows[split].tolist()),
                     standardised=preds[split], raw=view.to_raw(preds[split]),
                     y_true=view.y[split], checkpoint=tag)

    # export generator model as a pure lookup state
    if getattr(model, "mode", None) == "generator":
        tables = model.materialise_tables(permute=model.permute, device=device)
        lookup = {k: v.detach().cpu().clone() for k, v in tracker.state.items() if ".table.weight" not in k and ".table." not in k}
        # free lookups replaced by materialised tables
        for bank, mat in tables.items():
            lookup[f"local.{bank}.table.weight"] = mat.detach().cpu().clone()
        torch.save({"state_dict": lookup, "exported_from": cfg["family"],
                    "note": "pure lookup state equivalent to the generator model"},
                   out_dir / "checkpoint_best_lookup.pt")
        export = {"exported_tables": {b: list(t.shape) for b, t in tables.items()}}

    late = 20
    late_rows = curve[-late:]
    summary = {
        "best_valid": {"epoch": int(tracker.best_epoch), **{s: dict(best_metrics[s]) for s in LPC.SPLITS}},
        "last_epoch": {"epoch": int(curve[-1]["epoch"]), "updates": int(global_updates), **{s: dict(last_metrics[s]) for s in LPC.SPLITS}},
        "grid_best": sel,
        "late_window": {
            "window": late, "epochs": [int(r["epoch"]) for r in late_rows],
            **{f"{s}_{st}": float(np.mean([r[f"{s}_{st}"] for r in late_rows])) for s in LPC.SPLITS for st in ("rmse", "mae")},
            "note": "mean of per-epoch metrics; not an ensemble",
        },
    }
    if resume_path.is_file():
        resume_path.unlink()

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID, "dataset": cfg["dataset"], "status": "completed", "smoke": bool(smoke),
        "family": cfg["family"], "recipe": cfg["recipe"], "budget": cfg["budget"],
        "seed": int(LPC.TRAIN_SEED), "device": str(device), "hostname": platform.node(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": platform.python_version(),
        "git_commit": LPC.git_commit(), "determinism": LPC.env_info(),
        "code_hashes": C.addr2line(),
        "data": view.audit(),
        "model": {"parameters": params_total, "mode": getattr(model, "mode", "base"),
                  "permute": bool(getattr(model, "permute", False)), "export": export,
                  "expected_parameters": C.FS_PARAMS if getattr(model, "mode", "") == "generator" else None},
        "recipe_full": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in recipe.items()},
                        "epochs": int(planned_epochs), "batch_size": batch_size,
                        "steps_per_epoch": int(steps_per_epoch), "total_updates": int(total_updates),
                        "loss": "mse_on_train_standardised_target", "n_fit": int(n_fit)},
        "summary": summary,
        "grid": {"path": C.track_rel(C_dir_grid), "n": len(grid), "selection": sel},
        "curve": {"path": C.track_rel(curve_path), "sha256": C.sha256_file(curve_path), "n_epochs": len(curve)},
        "anomalies": {"grad_clip_events": int(clip_events), "grad_norm_max": float(grad_norm_max), "nonfinite_events": 0},
        "timing": {"train_wall_seconds": float(wall),
                   "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
                   "mean_epoch_seconds": float(np.mean([r["train_seconds"] + r["eval_seconds"] for r in curve])),
                   "gpu_peak_bytes": int(gpu_peak),
                   "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0) if device.type == "cuda" else 0.0},
        "test_labels_read": True,
        "test_access": "development-evaluation (authorised for this round)",
    }
    C.write_json(out_dir / "metrics.json", payload)
    C.log(f"[{cfg['family']}/{cfg['recipe']}/{cfg['budget']}/{cfg['dataset']}] best_ep={tracker.best_epoch} "
          f"valid={best_metrics['valid']['rmse']:.6f} test={best_metrics['test']['rmse']:.6f} params={params_total} wall={wall:.1f}s")
    return payload


def write_json(path, obj):
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="one semantic-table training run")
    ap.add_argument("--family", required=True, choices=list(C.SEM_FAMILIES) + ["B"])
    ap.add_argument("--recipe", default="R", choices=list(C.RECIPES))
    ap.add_argument("--dataset", default="lipo", choices=["lipo", "esol"])
    ap.add_argument("--budget", default="N100", choices=["N100", "N50", "N25"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    cfg = {"family": args.family, "recipe": args.recipe, "dataset": args.dataset, "budget": args.budget}
    name = C.arm_name(args.family, args.recipe, args.dataset, args.budget)
    out_dir = Path(args.out_dir) if args.out_dir else C.RUNS / ("smoke_" + name if args.smoke else name)
    payload = train_run(cfg, device=device, out_dir=out_dir, epochs_override=args.epochs, smoke=args.smoke, resume=args.resume)
    C.log(f"wrote {out_dir}")
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
