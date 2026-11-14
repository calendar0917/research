"""Update-driven trainer for the bottleneck-diagnosis round.

One training unit = one (model, dataset, fit-fraction) trajectory.  A Lipo arm is
a single continuous run to 2U0 with an explicit U0 snapshot -- never two
independent runs.  U0 and 2U0 are two *observation budgets of the same path*.

Discipline (inherited from the correction round and extended):

* update-driven: a deterministic epoch-internal ``randperm`` stream over the
  current fit positions, trailing partial batch kept;
* shared initial parameters across N (C1 = ``build_full_state(seed 0)`` + frozen
  remap; D-MPNN = ``torch.manual_seed(0)``);
* shared norms (``lb_data``): structure norm once on all official Train
  structures, regression target norm once on the smallest fit subset;
* full ``train_fit``/Valid/Test eval at every completed fit epoch and at the
  frozen update grid; each eval preserves/restores RNG + mode and uses its own
  iterator, so it never advances the training stream;
* main checkpoint candidates are the common grid only (step 0 kept separately);
* streaming best-checkpoint per rule (R_SEL / A_SEL / W_SEL).
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

import lb_common as C
import lb_data as D

import lcpr_common as PC
import lcpr_model as PM
import lcpr_train as PT
import lcpr_dmpnn as PDN
from lb_selection import ProbeSelector as StreamSelector  # noqa: E402

PROCESS_T0 = time.perf_counter()

regression_metrics = PT.regression_metrics
binary_metrics = PT.binary_metrics
split_metrics = PT.split_metrics
primary_value = PT.primary_value
is_better = PT.is_better
rng_snapshot = PT.rng_snapshot
rng_restore = PT.rng_restore


# ---------------------------------------------------------------------------
# selection criteria
# ---------------------------------------------------------------------------


def criterion_value(rule: str, task: str, m: Mapping[str, Any]) -> float:
    if rule == C.R_SEL:
        return float(m["rmse"] if task == "regression" else m["roc_auc"])
    if rule == C.A_SEL:
        return float(m["mae"] if task == "regression" else m["bce"])
    raise KeyError(rule)


def criterion_better(rule: str, task: str, value: float, best: float | None) -> bool:
    if not math.isfinite(value):
        return False
    if best is None:
        return True
    if rule == C.R_SEL and task == "binary":
        return value > best
    return value < best


# ---------------------------------------------------------------------------
# batch plan
# ---------------------------------------------------------------------------


class BatchPlan:
    """Deterministic epoch-internal ``randperm`` stream over ``n`` fit positions."""

    def __init__(self, n: int, batch_size: int):
        self.n = int(n)
        self.bs = int(batch_size)
        self.per_epoch = math.ceil(self.n / self.bs)
        self.epoch = 0
        self.batch_in_epoch = 0
        self.gen = torch.Generator().manual_seed(int(PC.SHUFFLE_SEED))
        self._perm = torch.randperm(self.n, generator=self.gen).tolist()

    def next_batch(self) -> list[int]:
        if self.batch_in_epoch >= self.per_epoch:
            self.epoch += 1
            self.batch_in_epoch = 0
            self._perm = torch.randperm(self.n, generator=self.gen).tolist()
        lo = self.batch_in_epoch * self.bs
        out = self._perm[lo: lo + self.bs]
        self.batch_in_epoch += 1
        return out

    def last_of_epoch(self) -> bool:
        return self.batch_in_epoch == self.per_epoch - 1

    def state(self) -> dict[str, Any]:
        return {"n": self.n, "bs": self.bs, "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch, "generator": self.gen.get_state()}

    def load_state(self, st: Mapping[str, Any]) -> None:
        if int(st["n"]) != self.n or int(st["bs"]) != self.bs:
            raise ValueError("batch plan state mismatch")
        self.epoch = int(st["epoch"])
        self.batch_in_epoch = int(st["batch_in_epoch"])
        self.gen.set_state(st["generator"])
        gen2 = torch.Generator().manual_seed(int(PC.SHUFFLE_SEED))
        perm = None
        for _ in range(self.epoch + 1):
            perm = torch.randperm(self.n, generator=gen2)
        self._perm = perm.tolist()

    def fingerprint(self) -> str:
        return C.sha256_text(C.stable_json({"n": self.n, "bs": self.bs, "seed": int(PC.SHUFFLE_SEED)}))


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


class C1Runner:
    def __init__(self, model, device, caches, *, chem8=None):
        self.model = model
        self.device = device
        self.caches = caches
        self.chem8 = chem8

    def _cache(self, split):
        c = self.caches[split]
        if self.chem8 is not None:
            c = {**c, "atom_feat": D.concat_chem8(c, self.chem8[split])}
        return c

    def batch(self, split, positions):
        return D.to_device(D.collate(self._cache(split), list(positions)), self.device)

    def forward(self, batch):
        return self.model(batch).reshape(-1)

    def predict(self, split, positions=None, batch_size=C.EVAL_BATCH):
        cache = self._cache(split)
        n = int(cache["n_molecules"])
        all_pos = list(range(n))
        if positions is None:
            positions = all_pos
        positions = [int(p) for p in positions]
        out = []
        for start in range(0, len(positions), batch_size):
            chunk = positions[start: start + batch_size]
            out.append(self.forward(self.batch(split, chunk)).detach().float().cpu().numpy().reshape(-1))
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0, np.float64)


class DMPNNRunner:
    def __init__(self, model, device, feat):
        self.model = model
        self.device = device
        self.feat = feat

    def batch_global(self, global_rows):
        return PDN._batch(self.feat, list(global_rows), self.device)

    def forward(self, bmg):
        return self.model(bmg).reshape(-1)

    def predict_global(self, global_rows, batch_size=1024):
        global_rows = [int(x) for x in global_rows]
        out = []
        for start in range(0, len(global_rows), batch_size):
            rows = global_rows[start: start + batch_size]
            out.append(self.forward(self.batch_global(rows)).detach().float().cpu().numpy().reshape(-1))
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0, np.float64)


# ---------------------------------------------------------------------------
# model builders
# ---------------------------------------------------------------------------


def _model_family(model: str) -> str:
    return "dmpnn" if model == "DMPNN_D2F" else "c1"


def _kind_for(model: str) -> str:
    return {"C1_R1": "c1", "C1_R3": "c1", "CHEM8_R1": "k_chem8", "K_R1": "k_combo"}[model]


def _recipe_for(model: str) -> dict[str, Any]:
    if model == "C1_R1":
        return dict(C.C1_R1)
    if model == "C1_R3":
        return dict(C.C1_R3)
    if model in ("CHEM8_R1", "K_R1"):
        return dict(C.C1_R1)
    if model == "DMPNN_D2F":
        return dict(C.DMPNN_D2F)
    raise KeyError(model)


def _arm_name(model: str, fit: str, dataset: str) -> str:
    return f"{'LIP' if dataset == C.PRIMARY_DATASET else 'BACE'}_{model}_F{fit}"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_arm(
    *,
    model: str,
    dataset: str,
    fit: str,
    device: torch.device,
    out_dir: Path,
    max_updates_override: int | None = None,
    smoke: bool = False,
    restart_from_u0: Path | None = None,
) -> dict[str, Any]:
    if model not in C.FAMILIES:
        raise ValueError(model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = D.dataset_meta(dataset)["task"]
    recipe = _recipe_for(model)
    bs = int(recipe["batch_size"])
    family = _model_family(model)
    if not smoke and C.git_dirty():
        raise SystemExit("formal run requested on a dirty tracked worktree; commit first")

    spec = D.frozen_spec()
    tgt = D.targets(dataset, include_test=True)
    caches = {s: D.load_cache(dataset, s, tgt["rows"][s]) for s in C.SPLITS}
    subset = D.nested_subset_positions(dataset, fit)
    fit_positions = subset["positions"]
    n_fit = len(fit_positions)
    snorm = D.structure_norm(dataset)
    tnorm_info = D.shared_target_norm(dataset)
    tnorm = tnorm_info["target_norm"]
    y_fit_all = np.asarray(tgt["y"]["train"], np.float64).reshape(-1)
    y_fit = y_fit_all[np.asarray(fit_positions, np.int64)]
    ftrain_rows = [int(tgt["rows"]["train"][p]) for p in fit_positions]
    valid_rows_arr = np.asarray([int(x) for x in tgt["rows"]["valid"]], np.int64)
    test_rows_arr = np.asarray([int(x) for x in tgt["rows"]["test"]], np.int64)
    fit_rows_arr = np.asarray(ftrain_rows, np.int64)
    y_valid = np.asarray(tgt["y"]["valid"], np.float64).reshape(-1)
    y_test = np.asarray(tgt["y"]["test"], np.float64).reshape(-1)

    U0 = C.u0(model, dataset)
    Umax = C.max_updates(model, dataset) if not smoke else 2
    if max_updates_override is not None:
        Umax = int(max_updates_override)

    chem8 = {s: D.chem8_atoms(dataset, s, caches[s]) for s in C.SPLITS} if model in ("CHEM8_R1", "K_R1") else None

    # ---- build model ----------------------------------------------------
    if family == "c1":
        kind = _kind_for(model)
        _full, base_state = PM.c1_reference_state_full(spec, snorm)
        model_obj = PM.build_model(spec, snorm, kind, variant="C1")
        init_report = PM.load_init_state(model_obj, base_state, kind)
        model_obj.to(device)
        runner = C1Runner(model_obj, device, caches, chem8=chem8)
        _lock_trainable = not any(p.requires_grad is False for p in model_obj.parameters())
    else:
        feat = PDN.load_feat(PDN.featurize_path(dataset))
        torch.manual_seed(int(C.TRAIN_SEED))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(C.TRAIN_SEED))
        model_obj = PDN.DMPNNReference(d_h=int(PC.DMPNN_FAMILY["hidden"])).to(device)
        init_report = {"kind": "dmpnn", "features": PDN.FEATURE_VERSION}
        runner = DMPNNRunner(model_obj, device, feat)

    n_params = int(sum(p.numel() for p in model_obj.parameters()))
    init_state_hash = PC.DM.state_fingerprint({k: v.detach().cpu() for k, v in model_obj.state_dict().items()})

    torch.manual_seed(int(C.TRAIN_SEED))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(C.TRAIN_SEED))

    # ---- optimizer / schedule ------------------------------------------
    restarting = restart_from_u0 is not None
    if family == "c1":
        optimizer = torch.optim.AdamW(
            [p for p in model_obj.parameters() if p.requires_grad],
            lr=float(recipe["peak_lr"]), betas=tuple(recipe["adam_betas"]),
            eps=float(recipe["adam_eps"]), weight_decay=float(recipe["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(U0, 1) if not restarting else max(U0, 1),
            eta_min=float(recipe["min_lr"]))
    else:
        optimizer = torch.optim.Adam([p for p in model_obj.parameters() if p.requires_grad],
                                     lr=float(recipe["optimizer_base_lr"]),
                                     weight_decay=float(recipe["weight_decay"]))
        wu = C.dmpnn_warmup_steps(dataset)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: PDN._noam_lambda(
                s, wu, max(U0 - wu, 1), float(recipe["init_lr"]), float(recipe["peak_lr"]),
                float(recipe["min_lr"])))

    plan = BatchPlan(n_fit, bs)
    grids = C.grid_updates(model, dataset)
    grid_e1 = set(grids["E1"])
    grid_e2 = set(grids["E2"])
    grid_all = sorted(grid_e1 | grid_e2)

    selector = StreamSelector(task)
    curves: list[dict[str, Any]] = []
    grid_preds: dict[str, list[np.ndarray]] = {"grid_updates": []}

    def strip(ev):
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in ev.items()}

    def eval_all():
        was = model_obj.training
        snap = rng_snapshot(device)
        model_obj.eval()
        try:
            if family == "c1":
                p_fit = runner.predict("train", fit_positions)
                p_valid = runner.predict("valid")
                p_test = runner.predict("test")
            else:
                p_fit = runner.predict_global(ftrain_rows)
                p_valid = runner.predict_global(valid_rows_arr)
                p_test = runner.predict_global(test_rows_arr)
        finally:
            rng_restore(snap, device)
            model_obj.train(was)
        return {
            "train_fit": {**split_metrics(task, p_fit, y_fit, tnorm), "_pred": p_fit, "_rows": fit_rows_arr, "_y": y_fit},
            "valid": {**split_metrics(task, p_valid, y_valid, tnorm), "_pred": p_valid, "_rows": valid_rows_arr, "_y": y_valid},
            "test": {**split_metrics(task, p_test, y_test, tnorm), "_pred": p_test, "_rows": test_rows_arr, "_y": y_test},
        }

    train_seconds = 0.0
    eval_seconds = 0.0
    sample_presentations = 0
    partial_batches = 0
    u0_snapshot: dict[str, Any] | None = None

    def do_eval(update: int, *, grid: bool, is_epoch_end: bool) -> None:
        nonlocal eval_seconds
        t = time.perf_counter()
        ev = eval_all()
        eval_seconds += time.perf_counter() - t
        s = strip(ev)
        rec = {"update": int(update), "is_grid": bool(grid), "is_epoch_end": bool(is_epoch_end),
               "phase": ("E1" if update <= U0 else "E2")}
        for name in ("train_fit", "valid", "test"):
            for k, v in s[name].items():
                if isinstance(v, (int, float)):
                    rec[f"{name}_{k}"] = float(v)
        curves.append(rec)
        if grid:
            selector.consider(update, s["valid"], model_obj.state_dict())
            grid_preds["grid_updates"].append(int(update))
            for name in ("train_fit", "valid", "test"):
                grid_preds.setdefault(name, []).append(np.asarray(ev[name]["_pred"], np.float64))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    # step-0 (zero-update) evaluation, kept separate from the main candidate pool
    t0 = time.perf_counter()
    ev0 = eval_all()
    eval_seconds += time.perf_counter() - t0
    s0 = strip(ev0)
    step0 = {
        "update": 0,
        "valid_metrics": s0["valid"], "train_fit_metrics": s0["train_fit"], "test_metrics": s0["test"],
        "state": {k: v.detach().cpu().clone() for k, v in model_obj.state_dict().items()},
        "rng": {"torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "numpy": np.random.get_state()},
    }
    curves.append({"update": 0, "is_grid": False, "is_epoch_end": True, "phase": "E1",
                   **{f"{n}_{k}": float(v) for n in ("train_fit", "valid", "test")
                      for k, v in s0[n].items() if isinstance(v, (int, float))}})

    # ---- optional restart from a U0 snapshot ---------------------------
    start_update = 0
    if restarting:
        snap = torch.load(restart_from_u0, map_location="cpu", weights_only=False)
        model_obj.load_state_dict({k: v.to(device) for k, v in snap["model"].items()})
        optimizer.load_state_dict(snap["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = float(recipe["peak_lr"])
        plan.load_state(snap["batch_plan"])
        # fresh cosine horizon of U0 steps from peak lr (Adam state NOT reset)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(U0, 1), eta_min=float(recipe["min_lr"]))
        start_update = int(snap["update"])
        Umax = start_update + U0
        # restart only extends past the prefix; grid points of E1 are reused from
        # the base arm, so only E2 needs recalling (handled by the analysis)
        grid_e2 = set(range(start_update + 1, Umax + 1)) & grid_all
        grid_e1 = set()

    update = start_update
    while update < Umax:
        model_obj.train()
        last = plan.last_of_epoch()
        rows_local = plan.next_batch()
        update += 1
        sample_presentations += len(rows_local)
        if len(rows_local) < bs:
            partial_batches += 1
        t0 = time.perf_counter()
        if family == "c1":
            pred = runner.forward(runner.batch("train", [fit_positions[i] for i in rows_local]))
        else:
            pred = runner.forward(runner.batch_global([ftrain_rows[i] for i in rows_local]))
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

        if update <= U0:
            scheduler.step()
        else:
            for g in optimizer.param_groups:
                g["lr"] = float(recipe["min_lr"])

        if update == U0:
            u0_snapshot = {
                "update": int(U0),
                "model": {k: v.detach().cpu().clone() for k, v in model_obj.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "batch_plan": plan.state(),
                "rng": {"torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "numpy": np.random.get_state()},
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            torch.save(u0_snapshot, out_dir / "u0_snapshot.pt")

        if update in grid_all:
            do_eval(update, grid=True, is_epoch_end=False)
        elif last:
            do_eval(update, grid=False, is_epoch_end=True)

    if update not in {r["update"] for r in curves}:
        do_eval(update, grid=False, is_epoch_end=True)
    wall = time.perf_counter() - started

    # ---- retention ------------------------------------------------------
    rules = selector.freeze()
    retained: dict[str, Any] = {}
    for rule, rec in rules.items():
        if rec["has_state"]:
            path = out_dir / f"best_{rule}.pt"
            torch.save({"state_dict": rec["state"], "update": rec["update"], "criterion": rec["criterion"],
                        "rule": rule, "arm_id": _arm_name(model, fit, dataset)}, path)
            retained[rule] = {"update": rec["update"], "criterion": rec["criterion"], "path": C.track_rel(path)}
        else:
            retained[rule] = {"update": None, "criterion": None, "path": None}
    rules_json = {r: {k: v for k, v in rec.items() if k != "state"} for r, rec in rules.items()}
    torch.save(step0, out_dir / "step0.pt")
    torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model_obj.state_dict().items()},
                "update": int(update)}, out_dir / "last.pt")
    np.savez_compressed(
        out_dir / "grid_predictions.npz",
        grid_updates=np.asarray(grid_preds["grid_updates"], np.int64),
        fit_rows=fit_rows_arr, valid_rows=valid_rows_arr, test_rows=test_rows_arr,
        fit_y=y_fit, valid_y=y_valid, test_y=y_test,
        **{name: (np.stack(grid_preds[name], axis=0) if grid_preds.get(name)
                  else np.zeros((0, len(fit_rows_arr if name == "train_fit" else (valid_rows_arr if name == "valid" else test_rows_arr))), np.float64))
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
        "schema": "lcrc-bottleneck-diagnosis-train-v1",
        "round_id": C.ROUND_ID, "protocol_id": C.PROTOCOL_ID, "status": "completed",
        "smoke": bool(smoke), "arm_id": _arm_name(model, fit, dataset), "restart": bool(restarting),
        "model": model, "dataset": dataset, "task": task, "fit": fit, "family": family,
        "subset": {"n": n_fit, "positions_sha256": subset["positions_sha256"],
                   "global_rows_sha256": subset["global_rows_sha256"],
                   "positions": subset["positions"], "global_rows": subset["global_rows"]},
        "data": {"n_train_official": C.official_train_n(dataset), "n_fit": n_fit,
                 "n_valid": int(valid_rows_arr.size), "n_test": int(test_rows_arr.size),
                 "valid_rows_sha256": C.row_sha256(valid_rows_arr), "test_rows_sha256": C.row_sha256(test_rows_arr),
                 "structure_norm_source": "all_official_train_structures", "target_norm": tnorm,
                 "shared_norms_fingerprint": D.shared_norms_fingerprint(dataset)},
        "model_info": {"parameters": n_params, "init_state_hash": init_state_hash, "init": init_report},
        "recipe": {**recipe, "U0": int(U0), "Umax": int(Umax), "steps_per_epoch": int(plan.per_epoch),
                   "batch_size": bs, "runs_phase2": bool(C._runs_phase2(dataset)),
                   "batch_plan_fingerprint": plan.fingerprint(),
                   "selection_rules": list(C.SELECTION_RULES),
                   "grid_points_per_phase": C.GRID_POINTS, "window": C.WINDOW},
        "selection": {"rules": rules_json, "retained": retained},
        "cost": {"train_seconds": float(train_seconds), "eval_seconds": float(eval_seconds),
                 "wall_seconds": float(wall), "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
                 "updates_run": int(update - start_update), "sample_presentations": int(sample_presentations),
                 "partial_batches": int(partial_batches),
                 "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0)
                 if device.type == "cuda" else 0.0},
        "environment": {**C.env_info(), "device": str(device)},
        "provenance": {"git_commit": C.git_commit(), "code_hashes": code_hashes()},
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(f"[{payload['arm_id']}] Umax={Umax} n_fit={n_fit} params={n_params} "
          f"train={train_seconds:.1f}s eval={eval_seconds:.1f}s wall={wall:.1f}s")
    return payload


CODE_PATHS = ("code", "configs", "PROTOCOL.md")


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(C.CODE_DIR.rglob("*.py")):
        out[str(p.relative_to(C.TRACK))] = C.sha256_file(p)
    for p in sorted(C.TESTS.glob("*.py")):
        out[f"tests/{p.name}"] = C.sha256_file(p)
    for p in sorted(C.CONFIGS.glob("*.yaml")):
        out[f"configs/{p.name}"] = C.sha256_file(p)
    if (C.TRACK / "PROTOCOL.md").exists():
        out["PROTOCOL.md"] = C.sha256_file(C.TRACK / "PROTOCOL.md")
    for p in sorted(PC.CODE_DIR.rglob("*.py")):
        out[str(p.relative_to(C.REPO))] = C.sha256_file(p)
    for p in sorted(C.DISSECT_CODE.glob("*.py")):
        out[str(p.relative_to(C.REPO))] = C.sha256_file(p)
    for p in sorted(C.MOLNET_CODE.glob("*.py")):
        out[str(p.relative_to(C.REPO))] = C.sha256_file(p)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="bottleneck diagnosis: one arm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart-from-u0", default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    out = Path(args.out_dir) if args.out_dir else C.RUNS / _arm_name(args.model, args.fit, args.dataset)
    train_arm(model=args.model, dataset=args.dataset, fit=args.fit, device=device, out_dir=out,
              max_updates_override=args.max_updates, smoke=args.smoke,
              restart_from_u0=(Path(args.restart_from_u0) if args.restart_from_u0 else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
