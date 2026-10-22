"""Runner for ``zinc_p3_sam_v1`` (two single-model arms, one seed).

    C  P3-Control : P3 graph + recipe-B AdamW (the same-round control)
    S  P3-SAM     : identical model / init / data / schedule / AdamW + vanilla
                    global-L2 SAM, constant rho=0.05

Per arm the job performs, in order: CPU-gate check -> 4 timing epochs (saved
separately, discarded) -> formal 600-epoch run from the same initial state and
row-order RNG -> best/final scoring on the *normal* weights -> freeze of the
best checkpoint hash -> one descriptive official-test read.  Nothing is selected
from test, no checkpoint averaging, no ensemble, no second predictor.

Usage
-----
    python .../sam_runner.py run   --arm C|S --device cuda:0
    python .../sam_runner.py test  --arm C|S --device cuda:0   # post-freeze only
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_REF = _SS / "zinc_restricted_adaptation_v1" / "analysis"
for _p in (str(_HERE), str(_V2), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import sam_core as S  # noqa: E402
import sam_checks as K  # noqa: E402

ARMS = {
    "C": {"name": "P3-Control", "run": "C_control_s0", "dir": "P3_Control", "arm": "C"},
    "S": {"name": "P3-SAM", "run": "S_sam_s0", "dir": "P3_SAM", "arm": "S"},
}
PER_ARM_HARD_STOP_SECONDS = 3600.0


def arm_paths(arm: str, tag: str = "") -> dict[str, Path]:
    spec = ARMS[arm]
    suffix = f"_{tag}" if tag else ""
    return {
        "run": S.RUNS / f"{spec['run']}{suffix}",
        "analysis": S.ANALYSIS / f"{spec['dir']}{suffix}",
    }


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def gate(cfg: dict, norm: dict, device) -> dict:
    checks_path = S.ANALYSIS / "cpu_checks.json"
    if not checks_path.exists():
        C.log("[gate] cpu_checks.json missing; running sam_checks.py")
        import sam_checks

        code = sam_checks.main()
        if code != 0:
            raise RuntimeError("sam_checks gate failed")
    doc = C.read_json(checks_path)
    if doc.get("config_sha256") != cfg["_config_sha256"]:
        raise RuntimeError("cpu_checks.json was produced for a different config file")
    if doc.get("n_fail", 1) != 0:
        raise RuntimeError(f"cpu checks failed: {doc.get('n_fail')} failing")
    failed = [k for k, v in doc["checks"].items() if not v.get("pass")]
    if failed:
        raise RuntimeError(f"failing checks: {failed}")

    # on-host invariants that do not depend on the local gate file's age
    model, info = S.build_model(norm, torch.device("cpu"))
    host_hash = C.state_hash(model)
    params = C.parameter_count(model)
    if params != int(cfg["model"]["parameters_expected"]):
        raise RuntimeError(f"parameter count {params}")
    if host_hash != cfg["model"]["initial_state_sha256"]:
        raise RuntimeError(f"init state hash {host_hash} != config")
    if info["ssl"]["sha256"] != cfg["model"]["ssl"]["sha256"]:
        raise RuntimeError("SSL sha256 mismatch")
    opt = S.make_optimizer(model, cfg)
    if C.optimizer_parameter_count(opt) != params:
        raise RuntimeError("optimizer does not cover all parameters")
    if any("BatchNorm" in type(m).__name__ for m in model.modules()):
        raise RuntimeError("unexpected BatchNorm in P3")
    if any(isinstance(m, torch.nn.Dropout) and float(m.p) != 0.0 for m in model.modules()):
        raise RuntimeError("unexpected non-zero dropout in P3")
    steps = math.ceil(cfg["data"]["train"] / cfg["schedule"]["batch_size"])
    total = steps * int(cfg["schedule"]["epochs"])
    if steps != 79 or total != 47400:
        raise RuntimeError(f"schedule drift: {steps} steps, {total} updates")
    return {
        "cpu_checks_n_pass": int(doc["n_pass"]),
        "cpu_checks_n_fail": int(doc["n_fail"]),
        "cpu_checks_config_sha256": doc.get("config_sha256"),
        "host_init_state_sha256": host_hash,
        "host_parameters": int(params),
        "optimizer_covers_all": True,
        "steps_per_epoch": int(steps),
        "outer_updates": int(total),
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# timing / plan
# ---------------------------------------------------------------------------


def run_timing(cfg: dict, arm: str, norm: dict, device, *, epochs: int,
               train_cache, y, rows, valid_cache, valid_rows) -> dict:
    model, info = S.build_model(norm, device)
    init_hash = C.state_hash(model)
    params = S.optimizable_params(model)
    opt = S.make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(int(cfg["schedule"]["seed"]) +
                                        int(cfg["schedule"]["shuffle_seed_offset"]))
    total = int(cfg["schedule"]["outer_updates"])
    curve = []
    step = 0
    _, epoch0_valid = S.evaluate(model, valid_cache, valid_rows, device,
                                 int(cfg["schedule"]["valid_batch_size"]))
    for ep in range(1, int(epochs) + 1):
        rec, step = S.train_epoch(model, opt, params, train_cache, rows, y, gen,
                                  arm=arm, cfg=cfg, epoch=ep, total_updates=total,
                                  global_step=step, collate_fn=S.collate_p3, device=device)
        _, valid_mae = S.evaluate(model, valid_cache, valid_rows, device,
                                  int(cfg["schedule"]["valid_batch_size"]))
        rec["valid_mae"] = float(valid_mae)
        curve.append(rec)
        C.log(f"[timing {arm}] ep={ep} valid={valid_mae:.6f} "
              f"online={rec['online_train_mae']:.6f} "
              f"pert={rec['perturbed_train_mae']} {rec['seconds']:.2f}s")
    steady = float(np.mean([r["seconds"] for r in curve[1:]])) if len(curve) > 1 else \
        float(curve[0]["seconds"])
    return {
        "arm": arm, "arms_name": ARMS[arm]["name"], "candidate": S.CANDIDATE,
        "seed": int(cfg["schedule"]["seed"]), "timing_epochs": int(epochs),
        "epoch0_valid_diagnostic": float(epoch0_valid if epoch0_valid is not None else "nan"),
        "curve": curve, "steady_epoch_seconds": steady,
        "parameters": int(info["parameters"]), "model_info": info,
        "init_state_hash": init_hash, "created": C.utcnow(),
    }


def plan(cfg: dict, arm: str, timing: dict, *, used_seconds: float) -> dict:
    t_epoch = float(timing["steady_epoch_seconds"])
    epochs = int(cfg["schedule"]["epochs"])
    projected_train = t_epoch * epochs
    reserve = 240.0  # scoring both checkpoints, freeze, test, writes
    projected = used_seconds + projected_train + reserve
    doc = {
        "protocol": S.PROTOCOL, "arm": arm, "candidate": S.CANDIDATE,
        "seed": int(cfg["schedule"]["seed"]),
        "steady_epoch_seconds": t_epoch,
        "used_seconds_before_plan": float(used_seconds),
        "epochs": epochs,
        "steps_per_epoch": int(cfg["schedule"]["steps_per_epoch"]),
        "total_updates": int(cfg["schedule"]["outer_updates"]),
        "projected_train_seconds": float(projected_train),
        "projected_total_seconds": float(projected),
        "hard_stop_seconds": PER_ARM_HARD_STOP_SECONDS,
        "feasible": bool(projected <= PER_ARM_HARD_STOP_SECONDS),
        "recipe": {
            "optimizer": "adamw",
            "peak_lr": float(cfg["optimizer"]["lr"]),
            "min_lr": float(cfg["optimizer"]["min_lr"]),
            "weight_decay": float(cfg["optimizer"]["weight_decay"]),
            "betas": cfg["optimizer"]["betas"],
            "eps": float(cfg["optimizer"]["eps"]),
            "warmup_updates": int(cfg["schedule"]["warmup_updates"]),
            "grad_clip": float(cfg["optimizer"]["grad_clip"]),
            "sam_rho": float(cfg["sam"]["rho"]) if arm == "S" else None,
        },
        "early_stop": None,
        "note": ("600-epoch cosine fixed before launch; no patience, no horizon change, "
                 "no test-based choice"),
        "created": C.utcnow(),
    }
    return doc


# ---------------------------------------------------------------------------
# formal run
# ---------------------------------------------------------------------------


def run_formal(cfg: dict, arm: str, norm: dict, device, *, epochs: int,
               train_cache, y, rows, valid_cache, valid_rows,
               timing: dict, run: Path, ana: Path, smoke: bool = False) -> dict:
    run.mkdir(parents=True, exist_ok=True)
    ana.mkdir(parents=True, exist_ok=True)
    model, info = S.build_model(norm, device)
    init_hash = C.state_hash(model)
    if init_hash != timing["init_state_hash"] or init_hash != cfg["model"]["initial_state_sha256"]:
        raise RuntimeError(f"formal init {init_hash} != timing/config")
    params = S.optimizable_params(model)
    opt = S.make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(int(cfg["schedule"]["seed"]) +
                                        int(cfg["schedule"]["shuffle_seed_offset"]))
    steps = int(cfg["schedule"]["steps_per_epoch"])
    total = steps * int(epochs)
    if not smoke and total != int(cfg["schedule"]["outer_updates"]):
        raise RuntimeError(f"total updates {total} != declared horizon")
    valid_batch = int(cfg["schedule"]["valid_batch_size"])

    # epoch-0 (SSL init) valid is a diagnostic only; it is identical for both
    # arms and is not a selection candidate.
    _, epoch0_valid = S.evaluate(model, valid_cache, valid_rows, device, valid_batch)
    C.log(f"[{arm}] epoch0-diagnostic valid={epoch0_valid:.9f} (not a candidate)")

    curve: list[dict] = []
    best: dict[str, Any] = {"valid_mae": float("inf"), "epoch": -1}
    step = 0
    started = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        rec, step = S.train_epoch(model, opt, params, train_cache, rows, y, gen,
                                  arm=arm, cfg=cfg, epoch=ep, total_updates=total,
                                  global_step=step, collate_fn=S.collate_p3, device=device)
        _, valid_mae = S.evaluate(model, valid_cache, valid_rows, device, valid_batch)
        rec["valid_mae"] = float(valid_mae)
        is_best = bool(rec["valid_mae"] < best["valid_mae"])
        if is_best:
            best = {"valid_mae": float(rec["valid_mae"]), "epoch": int(ep)}
            C.atomic_torch_save(
                {"state_dict": C.cpu_state(model), "epoch": int(ep),
                 "valid_mae": float(rec["valid_mae"]), "candidate": S.CANDIDATE,
                 "seed": int(cfg["schedule"]["seed"]), "arm": arm,
                 "global_step": int(step), "perturbed_weights": False},
                run / "best.pt")
        rec["is_best"] = is_best
        curve.append(rec)
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(model), "epoch": int(ep),
             "valid_mae": float(rec["valid_mae"]), "best": dict(best),
             "global_step": int(step), "candidate": S.CANDIDATE,
             "seed": int(cfg["schedule"]["seed"]), "arm": arm,
             "perturbed_weights": False},
            run / "last.pt")
        C.log(f"[{arm}] ep={ep:03d} valid={rec['valid_mae']:.6f} "
              f"online={rec['online_train_mae']:.6f} "
              f"pert={rec['perturbed_train_mae'] if rec['perturbed_train_mae'] is None else round(rec['perturbed_train_mae'], 6)} "
              f"lr={rec['lr']:.3e} best={best['valid_mae']:.6f}@{best['epoch']} "
              f"{rec['seconds']:.2f}s")
        if ep % 5 == 0 or ep == int(epochs):
            C.write_curve_csv(run / "curve.csv", curve)
    formal_seconds = time.perf_counter() - started
    if step != total:
        raise RuntimeError(f"actual outer updates {step} != {total}")
    C.atomic_torch_save(
        {"state_dict": C.cpu_state(model), "optimizer": opt.state_dict(),
         "gen_state": gen.get_state(), "epoch": int(epochs),
         "valid_mae": float(curve[-1]["valid_mae"]), "global_step": int(step),
         "best": dict(best), "curve": curve, "candidate": S.CANDIDATE,
         "seed": int(cfg["schedule"]["seed"]), "arm": arm},
        run / "final.pt")
    C.write_curve_csv(run / "curve.csv", curve)
    order_hashes = {
        "arm": arm, "generator_seed": int(cfg["schedule"]["seed"]) +
        int(cfg["schedule"]["shuffle_seed_offset"]),
        "n_molecules": int(len(rows)),
        "epochs": [r["order_sha256"] for r in curve],
    }
    C.write_json(run / "order_hashes.json", order_hashes)
    diag = {
        "arm": arm,
        "outer_updates": int(step),
        "gradient_calculations": int(step * (2 if arm == "S" else 1)),
        "rho": float(cfg["sam"]["rho"]) if arm == "S" else None,
        "g1_over_clip_updates": int(sum(r["g1_over_clip_updates"] for r in curve)),
        "g1_norm_mean_of_epoch_means": float(np.mean([r["g1_norm_mean"] for r in curve])),
        "g1_norm_max": float(np.max([r["g1_norm_max"] for r in curve])),
        "g2_norm_mean_of_epoch_means": (float(np.mean([r["g2_norm_mean"] for r in curve]))
                                        if arm == "S" else None),
        "g2_norm_max": (float(np.max([r["g2_norm_max"] for r in curve]))
                        if arm == "S" else None),
        "g2_clip_fraction_weighted": (float(np.mean([r["g2_clip_fraction"] for r in curve]))
                                      if arm == "S" else None),
        "eps_norm_mean_of_epoch_means": (float(np.mean([r["eps_norm_mean"] for r in curve]))
                                         if arm == "S" else None),
        "eps_norm_rel_dev_max": (float(np.max([r["eps_norm_rel_dev_max"] for r in curve]))
                                 if arm == "S" else None),
        "perturb_checks": ({"checked": int(sum(r["perturb_checks_checked"] for r in curve)),
                            "exact": int(sum(r["perturb_checks_exact"] for r in curve))}
                           if arm == "S" else None),
        "restore_checks": ({"checked": int(sum(r["restore_checks_checked"] for r in curve)),
                            "exact": int(sum(r["restore_checks_exact"] for r in curve))}
                           if arm == "S" else None),
        "clip_semantics": (
            "C: single update gradient, norm recorded before clip_grad_norm_(5) and "
            "clipped when above 5"
            if arm == "C" else
            "S: g1 is raw and never clipped (g1_over_clip_updates counts norms>5); only "
            "the second-pass gradient g2 is clipped to 5 (g2_clip_fraction)"),
        "note": ("perturbed_train_mae is the second-pass normalised L1 at theta+eps; it is "
                 "not the normal train MAE and not a sharpness measurement"),
    }
    C.write_json(ana / "sam_diagnostics.json", diag)
    return {"info": info, "curve": curve, "best": best, "epoch0_valid": float(epoch0_valid),
            "epochs_run": int(epochs), "global_step": int(step),
            "total_updates": int(total), "formal_seconds": float(formal_seconds),
            "init_state_hash": init_hash}


# ---------------------------------------------------------------------------
# scoring / freeze / test
# ---------------------------------------------------------------------------


def score_checkpoints(cfg: dict, arm: str, norm: dict, device, *, train_cache, y,
                      rows, valid_cache, valid_rows, run: Path, ana: Path) -> dict:
    ana.mkdir(parents=True, exist_ok=True)
    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    valid_batch = int(cfg["schedule"]["valid_batch_size"])
    eval_train_batch = int(cfg["schedule"]["eval_train_batch_size"])
    out: dict[str, Any] = {}
    for tag in ("best", "final"):
        ck = torch.load(run / f"{tag}.pt", map_location="cpu", weights_only=False)
        model, _ = S.build_model(norm, device)
        C.load_state(model, ck["state_dict"], device)
        pred_valid, mae_valid = S.evaluate(model, valid_cache, valid_rows, device, valid_batch)
        _, mae_train = S.evaluate(model, train_cache, rows, device, eval_train_batch)
        if tag == "best" and abs(mae_valid - float(ck["valid_mae"])) > 1e-6:
            raise RuntimeError(f"reloaded best valid {mae_valid} != stored {ck['valid_mae']}")
        np.savez(pred_dir / f"valid_{tag}.npz", row=valid_rows, pred=pred_valid,
                 y=np.asarray(y, dtype=np.float32))
        out[tag] = {
            "epoch": int(ck["epoch"]),
            "valid_mae": float(mae_valid),
            "valid_mae_stored": float(ck["valid_mae"]) if "valid_mae" in ck else None,
            "train_mae_eval_mode": float(mae_train),
            "sha256": C.sha256_file(run / f"{tag}.pt"),
        }
    out["last_sha256"] = C.sha256_file(run / "last.pt")
    write_preds_csv(ana / "preds_valid.csv", valid_rows, y,
                    np.load(pred_dir / "valid_best.npz")["pred"])
    return out


def write_preds_csv(path: Path, rows: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "y_raw", "pred"])
        for r, t, p in zip(rows, y, pred):
            w.writerow([int(r), f"{float(t):.9f}", f"{float(p):.9f}"])


def freeze(path: Path, payload: dict) -> dict:
    if path.exists():
        old = C.read_json(path)
        if old.get("best_sha256") != payload["best_sha256"]:
            raise RuntimeError(f"{path} already freezes a different checkpoint")
        C.log(f"[freeze] identical freeze already present at {path}")
        return old
    C.write_json(path, payload)
    return payload


def test_reference() -> dict:
    import csv

    path = _REF / "reference_preds_test.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1000:
        raise RuntimeError(f"{path}: expected 1000 rows, got {len(rows)}")
    row = np.asarray([int(r["row"]) for r in rows], dtype=np.int64)
    if not np.array_equal(row, np.arange(1000)):
        raise RuntimeError(f"{path}: row ids are not 0..999")
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    preds = {name: np.asarray([float(r[f"pred_{name}"]) for r in rows], dtype=np.float64)
             for name in ("P3B_s0", "L_s0", "DBT_s0")}
    expected = {"P3B_s0": 0.069864841, "L_s0": 0.066266555, "DBT_s0": 0.069401494}
    mae = {k: float(np.abs(v - y).mean()) for k, v in preds.items()}
    for k, v in mae.items():
        if abs(v - expected[k]) > 1e-6:
            raise RuntimeError(f"reference {k} MAE {v} != {expected[k]}")
    official = D.official_labels("test").astype(np.float64)
    if not np.allclose(official, y, atol=1e-6):
        raise RuntimeError("reference rows do not match the official test labels")
    return {"path": str(path.relative_to(C.REPO)), "row": row, "y": y,
            "preds": preds, "reference_mae": mae,
            "labels_match_official": True, "row_identity": "row == 0..999"}


def aux_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    err = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "p95_ae": float(np.quantile(err, 0.95)),
        "max_ae": float(err.max()),
        "max_ae_row": int(np.argmax(err)),
    }


def ensure_test_cache() -> dict:
    path = D.cache_path("test")
    if not path.exists():
        C.log("[test] building ad_test cache from official PyG (post-freeze)")
        D.save_cache(D.build_split_v2("test"), "test")
    return {"path": str(path.relative_to(C.REPO)), "sha256": C.sha256_file(path),
            "size_bytes": int(path.stat().st_size)}


def run_test(cfg: dict, arm: str, norm: dict, device, *, freeze_doc: dict,
             run: Path, ana: Path) -> dict:
    if freeze_doc["best_sha256"] != C.sha256_file(run / "best.pt"):
        raise RuntimeError("best.pt changed after the freeze")
    cache_info = ensure_test_cache()
    test_cache, y_test, rows_test = S.load_split("test")
    ref = test_reference()
    if not np.array_equal(rows_test, ref["row"]):
        raise RuntimeError("test cache rows are not 0..999")
    if not np.allclose(y_test, ref["y"], atol=1e-6):
        raise RuntimeError("test cache labels disagree with reference rows")
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model, _ = S.build_model(norm, device)
    C.load_state(model, ck["state_dict"], device)
    pred, mae = S.evaluate(model, test_cache, rows_test, device,
                           int(cfg["schedule"]["valid_batch_size"]))
    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "test_best.npz", row=rows_test, pred=pred)
    write_preds_csv(ana / "preds_test.csv", rows_test, y_test, pred)
    report = {
        "protocol": S.PROTOCOL, "arm": arm, "arms_name": ARMS[arm]["name"],
        "candidate": S.CANDIDATE,
        "freeze": freeze_doc,
        "best_sha256": C.sha256_file(run / "best.pt"),
        "best_epoch": int(ck["epoch"]),
        "best_valid_mae": float(ck["valid_mae"]),
        "test": {"n": int(len(rows_test)), "label_sha256": C.sha256_array(y_test),
                 **aux_metrics(pred, y_test)},
        "test_prediction_path": str((pred_dir / "test_best.npz").relative_to(C.REPO)),
        "test_prediction_sha256": C.sha256_array(pred),
        "reference_alignment": {
            "reference_preds": ref["path"],
            "reference_mae_recomputed": ref["reference_mae"],
            "labels_match_official": ref["labels_match_official"],
            "row_identity": ref["row_identity"],
        },
        "caches": cache_info,
        "historically_exposed_test": True,
        "test_used_for_selection": False,
        "test_read": "one descriptive pass after the freeze; nothing selected or changed",
    }
    C.write_json(ana / "test_report.json", report)
    return report


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def command_run(cfg: dict, arm: str, args) -> int:
    formal_epochs = int(cfg["schedule"]["epochs"])
    smoke = args.epochs is not None and int(args.epochs) != formal_epochs
    epochs = int(args.epochs) if args.epochs is not None else formal_epochs
    paths = arm_paths(arm, tag=f"smoke{epochs}" if smoke else "")
    ana, run = paths["analysis"], paths["run"]
    if smoke:
        C.log(f"[smoke] {epochs} epochs into {run} (never frozen, never tested)")
    ana.mkdir(parents=True, exist_ok=True)
    if (run / "train.json").exists() and not args.force:
        raise RuntimeError(f"{run}/train.json exists; use --force or the `test` command")
    job_start = time.perf_counter()
    norm = D.load_norm()
    device = torch.device(args.device)
    gate_info = gate(cfg, norm, device)
    C.log(f"[gate {arm}] {gate_info}")
    train_cache, y, rows = S.load_split("train")
    valid_cache, y_valid, valid_rows = S.load_split("valid")
    C.log(f"[{arm}] data ready ({time.perf_counter() - job_start:.1f}s) on {device}")

    timing = run_timing(cfg, arm, norm, device, epochs=int(args.timing_epochs),
                        train_cache=train_cache, y=y, rows=rows,
                        valid_cache=valid_cache, valid_rows=valid_rows)
    C.write_json(ana / "timing.json", timing)
    used = time.perf_counter() - job_start
    plan_doc = plan(cfg, arm, timing, used_seconds=used)
    plan_doc["gate"] = gate_info
    C.write_json(ana / "plan.json", plan_doc)
    C.log(f"[plan {arm}] t_epoch={timing['steady_epoch_seconds']:.3f}s "
          f"projected={plan_doc['projected_total_seconds']:.0f}s "
          f"feasible={plan_doc['feasible']}")
    if not plan_doc["feasible"]:
        C.write_json(ana / "status.json", {
            "status": "INCOMPLETE",
            "reason": "measured epoch time projects past the 60-minute per-arm stop",
            "plan": plan_doc, "timing": timing, "created": C.utcnow()})
        C.log("[stop] infeasible under the time rule; nothing trained formally")
        return 2

    formal = run_formal(cfg, arm, norm, device, epochs=epochs,
                        train_cache=train_cache, y=y, rows=rows,
                        valid_cache=valid_cache, valid_rows=valid_rows, timing=timing,
                        run=run, ana=ana, smoke=smoke)
    scored = score_checkpoints(cfg, arm, norm, device, train_cache=train_cache, y=y,
                               rows=rows, valid_cache=valid_cache,
                               valid_rows=valid_rows, run=run, ana=ana)
    total_seconds = time.perf_counter() - job_start
    curve = formal["curve"]
    valid_tail = [r["valid_mae"] for r in curve[-100:]]
    info = {
        "protocol": S.PROTOCOL, "arm": arm, "arms_name": ARMS[arm]["name"],
        "candidate": S.CANDIDATE, "seed": int(cfg["schedule"]["seed"]),
        "smoke": bool(smoke),
        "git_commit": C.git_commit(), "environment": C.environment(),
        "config_path": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y)), "n_valid": int(len(y_valid)),
            "train_label_sha256": C.sha256_array(y),
            "valid_label_sha256": C.sha256_array(y_valid),
            "official_test_loaded_before_freeze": False,
        },
        "gate": gate_info,
        "model_info": formal["info"],
        "parameter_count": int(formal["info"]["parameters"]),
        "recipe": {
            "optimizer": "adamw",
            "lr": float(cfg["optimizer"]["lr"]),
            "min_lr": float(cfg["optimizer"]["min_lr"]),
            "weight_decay": float(cfg["optimizer"]["weight_decay"]),
            "betas": cfg["optimizer"]["betas"],
            "eps": float(cfg["optimizer"]["eps"]),
            "grad_clip": float(cfg["optimizer"]["grad_clip"]),
            "warmup_updates": int(cfg["schedule"]["warmup_updates"]),
            "batch_size": int(cfg["schedule"]["batch_size"]),
            "drop_last": bool(cfg["schedule"]["drop_last"]),
            "epochs": int(formal["epochs_run"]),
            "steps_per_epoch": int(cfg["schedule"]["steps_per_epoch"]),
            "updates": int(formal["total_updates"]),
            "actual_global_steps": int(formal["global_step"]),
            "loss": "raw-y L1",
            "valid_every_epoch": True,
            "dropout": 0.0,
            "early_stop": None,
        },
        "sam": ({
            "type": cfg["sam"]["type"], "rho": float(cfg["sam"]["rho"]),
            "adaptive": False, "first_pass_clipped": False,
            "second_pass_clip": float(cfg["optimizer"]["grad_clip"]),
            "gradient_calculations": int(2 * formal["global_step"]),
            "optimizer_steps": int(formal["global_step"]),
        } if arm == "S" else {
            "active": False, "gradient_calculations": int(formal["global_step"]),
            "optimizer_steps": int(formal["global_step"]),
        }),
        "timing": timing, "plan": plan_doc,
        "epoch0_valid_diagnostic": float(formal["epoch0_valid"]),
        "best_valid": formal["best"],
        "scored": scored,
        "valid_tail_100": {
            "median": float(np.median(valid_tail)),
            "min": float(np.min(valid_tail)),
            "mean": float(np.mean(valid_tail)),
            "best_epoch_in_tail": bool(int(formal["best"]["epoch"]) > int(formal["epochs_run"]) - 100),
        },
        "curve": curve,
        "wall_seconds": float(total_seconds),
        "formal_train_seconds": float(formal["formal_seconds"]),
        "gpu_hours": None if device.type == "cpu" else float(total_seconds) / 3600.0,
        "notes": [
            "best-valid uses strict <, ties keep the earlier epoch",
            "online_train_mae is the within-epoch pre-update normal-weight mean",
            "perturbed_train_mae is the SAM second-pass theta+eps mean (not normal train MAE)",
            "train_mae_eval_mode is the full-10k eval-mode score of the frozen checkpoint",
            "no test data was read before the freeze",
        ],
    }
    C.write_json(run / "train.json", info)
    if (run / "curve.csv").exists():
        (ana / "curve.csv").write_text((run / "curve.csv").read_text(encoding="utf-8"),
                                       encoding="utf-8")
    C.write_json(ana / "status.json", {
        "status": "SMOKE_COMPLETE" if smoke else "TRAINED",
        "wall_seconds": float(total_seconds), "created": C.utcnow()})
    C.log(f"[{arm}] trained best={formal['best']['valid_mae']:.6f}@"
          f"{formal['best']['epoch']} wall={total_seconds:.1f}s")

    if smoke:
        C.log(f"[smoke] {arm} stopped after scoring; no freeze, no test")
        return 0
    freeze_doc = {
        "protocol": S.PROTOCOL, "arm": arm, "arms_name": ARMS[arm]["name"],
        "candidate": S.CANDIDATE, "seed": int(cfg["schedule"]["seed"]),
        "git_commit": C.git_commit(), "config_sha256": cfg["_config_sha256"],
        "best_sha256": scored["best"]["sha256"],
        "best_epoch": int(scored["best"]["epoch"]),
        "best_valid_mae": float(scored["best"]["valid_mae"]),
        "best_valid_mae_stored": float(scored["best"]["valid_mae_stored"]),
        "best_train_mae_eval_mode": float(scored["best"]["train_mae_eval_mode"]),
        "final_sha256": scored["final"]["sha256"],
        "final_epoch": int(scored["final"]["epoch"]),
        "final_valid_mae": float(scored["final"]["valid_mae"]),
        "final_train_mae_eval_mode": float(scored["final"]["train_mae_eval_mode"]),
        "last_sha256": scored["last_sha256"],
        "init_state_sha256": formal["init_state_hash"],
        "ssl_sha256": formal["info"]["ssl"]["sha256"],
        "params": int(formal["info"]["parameters"]),
        "epochs_run": int(formal["epochs_run"]),
        "outer_updates": int(formal["global_step"]),
        "gradient_calculations": int(formal["global_step"] * (2 if arm == "S" else 1)),
        "recipe": info["recipe"], "sam": info["sam"],
        "data": info["data"],
        "test_policy": "one descriptive read after this freeze; no test-driven change",
        "frozen_at": C.utcnow(),
    }
    freeze_doc = freeze(ana / "freeze.json", freeze_doc)
    C.log(f"[freeze {arm}] best sha256 {freeze_doc['best_sha256']}")
    report = run_test(cfg, arm, norm, device, freeze_doc=freeze_doc, run=run, ana=ana)
    C.log(f"[test {arm}] test MAE {report['test']['mae']:.6f}")
    C.write_json(ana / "status.json", {"status": "COMPLETE",
                                       "wall_seconds": float(time.perf_counter() - job_start),
                                       "created": C.utcnow()})
    return 0


def command_test(cfg: dict, arm: str, args) -> int:
    paths = arm_paths(arm)
    freeze_doc = C.read_json(paths["analysis"] / "freeze.json")
    report = run_test(cfg, arm, D.load_norm(), torch.device(args.device),
                      freeze_doc=freeze_doc, run=paths["run"], ana=paths["analysis"])
    C.log(f"[test {arm}] test MAE {report['test']['mae']:.6f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "test"])
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timing-epochs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=None,
                   help="smoke only: !=600 writes a separate runs/*_smokeN dir and never freezes/tests")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = S.load_config()
    if args.command == "test":
        return command_test(cfg, args.arm, args)
    return command_run(cfg, args.arm, args)


if __name__ == "__main__":
    raise SystemExit(main())
