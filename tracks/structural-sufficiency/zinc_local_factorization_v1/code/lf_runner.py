"""Runner for ``zinc_local_factorization_v1`` (two single-model arms, one seed).

    D  LF-Dense       ordinary content MLP (98 -> 80 -> 32)
    F  LF-Factorized  shared atom/key projections + role modulation + z1/z2/z3
                      (rank 32, up to third order) -> 114 -> 64 -> 32

Both arms share the P3 carrier, the train-only masked-SSL start, the identical
``bf_data`` S3/S4/F4 objects, the identical root-preserving permutation average,
the identical 198-dim per-root statistics and port, the identical recipe,
batch order and horizon.  Only ``phi`` differs.

Per arm the job performs, in order: CPU-gate check -> 4 timing epochs (saved
separately, discarded) -> formal run from the same seeds / optimiser reset ->
best/final scoring on the *normal* weights -> freeze of the best checkpoint hash
-> one descriptive official-test read.  Nothing is selected from test and there
is no checkpoint averaging, ensemble or second predictor.

Usage
-----
    python .../lf_runner.py run  --arm D|F --device cuda:0
    python .../lf_runner.py test --arm D|F --device cuda:0   # post-freeze only
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
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
_REF = _SS / "zinc_restricted_adaptation_v1" / "analysis"
for _p in (str(_HERE), str(_V1), str(_BF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import bf_data as BFD  # noqa: E402
import lf_core as LC  # noqa: E402
import lf_models as LM  # noqa: E402

ARMS = {
    "D": {"name": "LF-Dense", "run": "LF_D_s0", "dir": "LF_D", "candidate": "LF-D"},
    "F": {"name": "LF-Factorized", "run": "LF_F_s0", "dir": "LF_F", "candidate": "LF-F"},
}
PER_ARM_HARD_STOP_SECONDS = 3600.0
TIME_BUDGET_SECONDS = 3000.0
RESERVE_SECONDS = 120.0
MIN_EPOCHS = 300


def arm_paths(arm: str, tag: str = "") -> dict[str, Path]:
    spec = ARMS[arm]
    suffix = f"_{tag}" if tag else ""
    return {
        "run": LC.RUNS / f"{spec['run']}{suffix}",
        "analysis": LC.ANALYSIS / f"{spec['dir']}{suffix}",
    }


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def gate(cfg: dict, norm: dict, device) -> dict:
    checks_path = LC.ANALYSIS / "cpu_checks.json"
    if not checks_path.exists():
        raise RuntimeError(f"missing {checks_path}; run lf_checks.py first")
    doc = C.read_json(checks_path)
    if doc.get("config_sha256") != cfg["_config_sha256"]:
        raise RuntimeError("cpu_checks.json was produced for a different config file")
    if int(doc.get("n_fail", 1)) != 0:
        raise RuntimeError(f"cpu checks failed: {doc.get('n_fail')} failing")
    failed = [k for k, v in doc["checks"].items() if not v.get("pass")]
    if failed:
        raise RuntimeError(f"failing checks: {failed}")

    out = {"cpu_checks_n_pass": int(doc["n_pass"]), "cpu_checks_n_fail": int(doc["n_fail"]),
           "cpu_checks_config_sha256": doc.get("config_sha256"), "arms": {}}
    for arm in LM.ARMS:
        model, info = LC.build(arm, norm, torch.device("cpu"))
        hashes = LC.init_hashes(model)
        expected = cfg["model"]["arms"][arm]
        if int(info["parameters"]) != int(expected["parameters_expected"]):
            raise RuntimeError(f"{arm}: parameter count {info['parameters']}")
        if hashes["init_state_sha256"] != expected["initial_state_sha256"]:
            raise RuntimeError(f"{arm}: init state hash {hashes['init_state_sha256']} != config")
        if hashes["public_state_sha256"] != cfg["model"]["carrier_initial_state_sha256"]:
            raise RuntimeError(f"{arm}: public state hash != declared SSL-start P3 carrier")
        if hashes["shared_embedding_sha256"] != cfg["model"]["shared_embedding_initial_sha256"]:
            raise RuntimeError(f"{arm}: shared embedding hash != config")
        if not hashes["w_s_zero"]:
            raise RuntimeError(f"{arm}: W_s is not exactly zero")
        if info["ssl_encoders"]["sha256"] != cfg["model"]["ssl"]["sha256"]:
            raise RuntimeError(f"{arm}: SSL sha256 mismatch")
        opt = LC.make_optimizer(model, cfg, int(cfg["schedule"]["outer_updates"]))
        if C.optimizer_parameter_count(opt) != int(info["parameters"]):
            raise RuntimeError(f"{arm}: optimizer does not cover all parameters")
        if any(isinstance(m, torch.nn.Dropout) and float(m.p) != 0.0 for m in model.modules()):
            raise RuntimeError(f"{arm}: unexpected non-zero dropout")
        if any("BatchNorm" in type(m).__name__ for m in model.modules()):
            raise RuntimeError(f"{arm}: unexpected BatchNorm")
        out["arms"][arm] = {"parameters": int(info["parameters"]), **hashes,
                            "optimizer_covers_all": True, "ssl_sha256":
                            info["ssl_encoders"]["sha256"]}
    steps = math.ceil(cfg["data"]["train"] / cfg["schedule"]["batch_size"])
    total = steps * int(cfg["schedule"]["epochs"])
    if steps != 79 or total != int(cfg["schedule"]["outer_updates"]):
        raise RuntimeError(f"schedule drift: {steps} steps, {total} updates")
    out.update({"steps_per_epoch": int(steps), "outer_updates": int(total),
                "device": str(device), "cuda_visible_devices":
                str(torch.cuda.device_count()) + " visible"})
    return out


# ---------------------------------------------------------------------------
# timing / plan
# ---------------------------------------------------------------------------


def run_timing(cfg: dict, arm: str, norm: dict, device, *, epochs: int,
               train_cache, y, rows, valid_cache, valid_rows) -> dict:
    model, info = LC.build(arm, norm, device)
    hashes = LC.init_hashes(model)
    opt = LC.make_optimizer(model, cfg, int(LC.SEED))
    gen = torch.Generator().manual_seed(int(cfg["schedule"]["seed"]) +
                                        int(cfg["schedule"]["shuffle_seed_offset"]))
    curve = []
    step = 0
    _, epoch0_valid = LC.evaluate(model, valid_cache, valid_rows, device,
                                  int(cfg["schedule"]["valid_batch_size"]))
    total = 79 * int(epochs)
    for ep in range(1, int(epochs) + 1):
        rec, step = LC.train_epoch(model, opt, train_cache, rows, y, gen, cfg=cfg, epoch=ep,
                                   total_updates=total, global_step=step, device=device)
        _, valid_mae = LC.evaluate(model, valid_cache, valid_rows, device,
                                   int(cfg["schedule"]["valid_batch_size"]))
        rec["valid_mae"] = float(valid_mae)
        curve.append(rec)
        C.log(f"[timing {arm}] ep={ep} valid={valid_mae:.6f} "
              f"online={rec['online_train_mae']:.6f} {rec['seconds']:.2f}s "
              f"port_rms={rec['port_rms']:.3e} phi_delta={rec['phi_epoch_delta']:.3e}")
    steady = float(np.mean([r["seconds"] for r in curve[1:]])) if len(curve) > 1 else \
        float(curve[0]["seconds"])
    return {
        "arm": arm, "arm_name": ARMS[arm]["name"], "candidate": ARMS[arm]["candidate"],
        "seed": int(LC.SEED), "timing_epochs": int(epochs),
        "epoch0_valid_diagnostic": float(epoch0_valid),
        "curve": curve, "steady_epoch_seconds": steady,
        "parameters": int(info["parameters"]), "model_info": info, "init_hashes": hashes,
        "created": C.utcnow(),
    }


def plan(cfg: dict, arm: str, timing: dict, *, used_seconds: float) -> dict:
    t_epoch = float(timing["steady_epoch_seconds"])
    nominal = int(cfg["schedule"]["epochs"])
    if 1.15 * t_epoch * nominal + RESERVE_SECONDS <= PER_ARM_HARD_STOP_SECONDS:
        epochs, rule = nominal, "1.15*t_slow*600+120 <= 3600 -> keep 600 epochs"
    else:
        epochs = int(min(nominal, 10 * math.floor((TIME_BUDGET_SECONDS / (1.15 * t_epoch)) / 10)))
        rule = ("1.15*t_slow*600+120 > 3600 -> E = min(600, 10*floor((3000/(1.15*t_slow))/10)), "
                "shared by both arms")
    projected = 1.15 * t_epoch * epochs + RESERVE_SECONDS
    doc = {
        "protocol": LC.PROTOCOL, "arm": arm, "arm_name": ARMS[arm]["name"],
        "candidate": ARMS[arm]["candidate"], "seed": int(LC.SEED),
        "steady_epoch_seconds": t_epoch,
        "used_seconds_before_plan": float(used_seconds),
        "nominal_epochs": nominal,
        "epochs": int(epochs),
        "steps_per_epoch": 79,
        "total_updates": int(79 * epochs),
        "projected_train_seconds": float(1.15 * t_epoch * epochs),
        "projected_total_seconds": float(projected),
        "hard_stop_seconds": PER_ARM_HARD_STOP_SECONDS,
        "feasible": bool(projected <= PER_ARM_HARD_STOP_SECONDS and epochs >= MIN_EPOCHS),
        "reduction_rule": rule,
        "recipe": {
            "optimizer": "adamw", "peak_lr": float(cfg["optimizer"]["lr"]),
            "min_lr": float(cfg["optimizer"]["min_lr"]),
            "weight_decay": float(cfg["optimizer"]["weight_decay"]),
            "betas": cfg["optimizer"]["betas"], "eps": float(cfg["optimizer"]["eps"]),
            "warmup_updates": int(cfg["schedule"]["warmup_updates"]),
            "grad_clip": float(cfg["optimizer"]["grad_clip"]),
            "horizon": f"per-step cosine over {79 * epochs} outer updates",
        },
        "early_stop": None,
        "note": ("no architecture / rank / LR / horizon selection from timing or valid; "
                 "the only timing-based decision is the shared epoch count defined in the "
                 "config"),
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
    model, info = LC.build(arm, norm, device)
    hashes = LC.init_hashes(model)
    if hashes["init_state_sha256"] != timing["init_hashes"]["init_state_sha256"]:
        raise RuntimeError(f"formal init {hashes['init_state_sha256']} != timing init")
    if hashes["init_state_sha256"] != cfg["model"]["arms"][arm]["initial_state_sha256"]:
        raise RuntimeError("formal init != config init hash")
    opt = LC.make_optimizer(model, cfg, 79 * int(epochs))
    gen = torch.Generator().manual_seed(int(cfg["schedule"]["seed"]) +
                                        int(cfg["schedule"]["shuffle_seed_offset"]))
    steps = 79
    total = steps * int(epochs)
    valid_batch = int(cfg["schedule"]["valid_batch_size"])

    # epoch-0 (SSL init) valid is a diagnostic only: identical for both arms.
    _, epoch0_valid = LC.evaluate(model, valid_cache, valid_rows, device, valid_batch)
    C.log(f"[{arm}] epoch0-diagnostic valid={epoch0_valid:.9f} (not a candidate)")

    curve: list[dict] = []
    best: dict[str, Any] = {"valid_mae": float("inf"), "epoch": -1}
    step = 0
    started = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        rec, step = LC.train_epoch(model, opt, train_cache, rows, y, gen, cfg=cfg, epoch=ep,
                                   total_updates=total, global_step=step, device=device)
        _, valid_mae = LC.evaluate(model, valid_cache, valid_rows, device, valid_batch)
        rec["valid_mae"] = float(valid_mae)
        is_best = bool(rec["valid_mae"] < best["valid_mae"])
        if is_best:
            best = {"valid_mae": float(rec["valid_mae"]), "epoch": int(ep)}
            C.atomic_torch_save(
                {"state_dict": C.cpu_state(model), "epoch": int(ep),
                 "valid_mae": float(rec["valid_mae"]), "candidate": ARMS[arm]["candidate"],
                 "seed": int(LC.SEED), "arm": arm, "global_step": int(step)},
                run / "best.pt")
        rec["is_best"] = is_best
        curve.append(rec)
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(model), "epoch": int(ep),
             "valid_mae": float(rec["valid_mae"]), "best": dict(best),
             "global_step": int(step), "candidate": ARMS[arm]["candidate"],
             "seed": int(LC.SEED), "arm": arm},
            run / "last.pt")
        C.log(f"[{arm}] ep={ep:03d} valid={rec['valid_mae']:.6f} "
              f"online={rec['online_train_mae']:.6f} lr={rec['lr']:.3e} "
              f"best={best['valid_mae']:.6f}@{best['epoch']} {rec['seconds']:.2f}s "
              f"w_s={rec['w_s_norm']:.4f} port={rec['port_rms']:.4f} "
              f"dphi={rec['phi_epoch_delta']:.4f}")
        if ep % 5 == 0 or ep == int(epochs):
            C.write_curve_csv(run / "curve.csv", curve)
    formal_seconds = time.perf_counter() - started
    if step != total:
        raise RuntimeError(f"actual outer updates {step} != {total}")
    C.atomic_torch_save(
        {"state_dict": C.cpu_state(model), "optimizer": opt.state_dict(),
         "gen_state": gen.get_state(), "epoch": int(epochs),
         "valid_mae": float(curve[-1]["valid_mae"]), "global_step": int(step),
         "best": dict(best), "curve": curve, "candidate": ARMS[arm]["candidate"],
         "seed": int(LC.SEED), "arm": arm},
        run / "final.pt")
    C.write_curve_csv(run / "curve.csv", curve)
    C.write_json(run / "order_hashes.json", {
        "arm": arm,
        "generator_seed": int(cfg["schedule"]["seed"]) + int(cfg["schedule"]["shuffle_seed_offset"]),
        "n_molecules": int(len(rows)),
        "epochs": [r["order_sha256"] for r in curve],
    })
    return {"info": info, "init_hashes": hashes, "curve": curve, "best": best,
            "epoch0_valid": float(epoch0_valid), "epochs_run": int(epochs),
            "global_step": int(step), "total_updates": int(total),
            "formal_seconds": float(formal_seconds)}


# ---------------------------------------------------------------------------
# scoring / freeze / test
# ---------------------------------------------------------------------------


def write_preds_csv(path: Path, rows: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = __import__("csv").writer(fh)
        w.writerow(["row", "y_raw", "pred"])
        for r, t, p in zip(rows, y, pred):
            w.writerow([int(r), f"{float(t):.9f}", f"{float(p):.9f}"])


def score_checkpoints(cfg: dict, arm: str, norm: dict, device, *, train_cache, y, rows,
                      valid_cache, valid_rows, run: Path, ana: Path) -> dict:
    ana.mkdir(parents=True, exist_ok=True)
    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    out: dict[str, Any] = {}
    for tag in ("best", "final"):
        ck = torch.load(run / f"{tag}.pt", map_location="cpu", weights_only=False)
        model, _ = LC.build(arm, norm, device)
        C.load_state(model, ck["state_dict"], device)
        pred_valid, mae_valid = LC.evaluate(model, valid_cache, valid_rows, device,
                                            int(cfg["schedule"]["valid_batch_size"]))
        _, mae_train = LC.evaluate(model, train_cache, rows, device,
                                   int(cfg["schedule"]["eval_train_batch_size"]))
        if tag == "best" and abs(mae_valid - float(ck["valid_mae"])) > 1e-6:
            raise RuntimeError(f"reloaded best valid {mae_valid} != stored {ck['valid_mae']}")
        np.savez(pred_dir / f"valid_{tag}.npz", row=valid_rows, pred=pred_valid,
                 y=np.asarray(y, dtype=np.float32))
        write_preds_csv(ana / f"preds_valid_{tag}.csv", valid_rows,
                        np.asarray(valid_cache["labels"], dtype=np.float64), pred_valid)
        out[tag] = {"epoch": int(ck["epoch"]), "valid_mae": float(mae_valid),
                    "valid_mae_stored": float(ck["valid_mae"]),
                    "train_mae_eval_mode": float(mae_train),
                    "sha256": C.sha256_file(run / f"{tag}.pt")}
    out["last_sha256"] = C.sha256_file(run / "last.pt")
    return out


def freeze(path: Path, payload: dict) -> dict:
    if path.exists():
        old = C.read_json(path)
        if old.get("best_sha256") != payload["best_sha256"]:
            raise RuntimeError(f"{path} already freezes a different checkpoint")
        C.log(f"[freeze] identical freeze already present at {path}")
        return old
    C.write_json(path, payload)
    return payload


def read_reference(split: str) -> dict:
    import csv

    path = _REF / f"reference_preds_{split}.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    if len(raw) != 1000:
        raise RuntimeError(f"{path}: expected 1000 rows, got {len(raw)}")
    row = np.asarray([int(r["row"]) for r in raw], dtype=np.int64)
    if not np.array_equal(row, np.arange(1000)):
        raise RuntimeError(f"{path}: row ids are not 0..999")
    y = np.asarray([float(r["y_raw"]) for r in raw], dtype=np.float64)
    preds = {name: np.asarray([float(r[f"pred_{name}"]) for r in raw], dtype=np.float64)
             for name in ("P3B_s0", "L_s0", "DBT_s0")}
    unseen = np.asarray([int(r["signature_unseen"]) for r in raw], dtype=np.int64)
    ordinary = np.asarray([int(r["ordinary_y_gt_minus3"]) for r in raw], dtype=np.int64)
    if not np.array_equal(ordinary, (y > -3.0).astype(np.int64)):
        raise RuntimeError(f"{split}: ordinary flag disagrees with y")
    return {"rows": row, "y": y, "refs": preds, "unseen": unseen, "ordinary": ordinary,
            "csv": str(path.relative_to(C.REPO))}


def ensure_test_caches() -> dict:
    out = {}
    ad_path = D.cache_path("test")
    if not ad_path.exists():
        C.log("[test] building ad_test cache from official PyG (post-freeze)")
        D.save_cache(D.build_split_v2("test"), "test")
    out["ad_test"] = {"path": str(ad_path.relative_to(C.REPO)), "sha256": C.sha256_file(ad_path)}
    if not BFD.cache_path("test").exists():
        C.log("[test] building fragment test cache (structure only, post-freeze)")
        BFD.save_split("test")
    pc = BFD.load_cache("test")
    out["bf_test"] = {"path": str(BFD.cache_path("test").relative_to(C.REPO)),
                      "sha256": C.sha256_file(BFD.cache_path("test")),
                      "meta": {k: pc["meta"][k] for k in
                               ("n_molecules", "n_entries", "n_unique_patterns",
                                "occurrences_by_family", "labels_used", "test_used")}}
    return out


def run_test(cfg: dict, arm: str, norm: dict, device, *, freeze_doc: dict,
             run: Path, ana: Path) -> dict:
    if freeze_doc["best_sha256"] != C.sha256_file(run / "best.pt"):
        raise RuntimeError("best.pt changed after the freeze")
    caches = ensure_test_caches()
    if caches["bf_test"]["meta"]["labels_used"] is not False:
        raise RuntimeError("test fragment cache is not label-free")
    test_cache, y_test, rows_test = LC.load_split("test")
    ref = read_reference("test")
    if not np.array_equal(rows_test, ref["rows"]):
        raise RuntimeError("test cache rows are not 0..999")
    if not np.allclose(y_test, ref["y"], atol=1e-6):
        raise RuntimeError("test cache labels disagree with the reference rows")
    official = D.official_labels("test").astype(np.float64)
    if not np.allclose(official, ref["y"], atol=1e-6):
        raise RuntimeError("official test labels disagree with the reference rows")
    LC._TABLES.pop("test", None)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model, _ = LC.build(arm, norm, device)
    C.load_state(model, ck["state_dict"], device)
    pred, mae = LC.evaluate(model, test_cache, rows_test, device,
                            int(cfg["schedule"]["valid_batch_size"]))
    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "test_best.npz", row=rows_test, pred=pred)
    write_preds_csv(ana / "preds_test_best.csv", rows_test, y_test, pred)
    err = np.abs(pred.astype(np.float64) - y_test)
    report = {
        "protocol": LC.PROTOCOL, "arm": arm, "arm_name": ARMS[arm]["name"],
        "candidate": ARMS[arm]["candidate"], "freeze": freeze_doc,
        "best_sha256": C.sha256_file(run / "best.pt"),
        "best_epoch": int(ck["epoch"]),
        "best_valid_mae": float(ck["valid_mae"]),
        "test": {"n": int(len(rows_test)), "mae": float(mae),
                 "rmse": float(np.sqrt(np.mean(err ** 2))),
                 "p95_ae": float(np.quantile(err, 0.95)),
                 "max_ae": float(err.max()), "max_ae_row": int(np.argmax(err)),
                 "label_sha256": C.sha256_array(y_test)},
        "test_prediction_path": str((pred_dir / "test_best.npz").relative_to(C.REPO)),
        "test_prediction_sha256": C.sha256_array(pred),
        "caches": caches,
        "reference_alignment": {
            "reference_preds": ref["csv"],
            "recomputed_reference_mae": {k: float(np.abs(v - ref["y"]).mean())
                                         for k, v in ref["refs"].items()},
            "labels_match_official": True, "row_identity": "row == 0..999"},
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
    train_cache, y, rows = LC.load_split("train")
    valid_cache, y_valid, valid_rows = LC.load_split("valid")
    C.log(f"[{arm}] data ready ({time.perf_counter() - job_start:.1f}s) on {device}")

    timing = run_timing(cfg, arm, norm, device, epochs=int(args.timing_epochs),
                        train_cache=train_cache, y=y, rows=rows,
                        valid_cache=valid_cache, valid_rows=valid_rows)
    C.write_json(ana / "timing.json", timing)
    used = time.perf_counter() - job_start
    plan_doc = plan(cfg, arm, timing, used_seconds=used)
    plan_doc["gate"] = gate_info
    C.write_json(ana / "plan.json", plan_doc)
    C.log(f"[plan {arm}] t_slow={timing['steady_epoch_seconds']:.3f}s "
          f"E={plan_doc['epochs']} projected={plan_doc['projected_total_seconds']:.0f}s "
          f"feasible={plan_doc['feasible']}")
    if not plan_doc["feasible"]:
        C.write_json(ana / "status.json", {
            "status": "INCOMPLETE", "reason":
            "measured epoch time projects past the per-arm stop or forces E<300; "
            "implementation/engineering report only, no formal training",
            "plan": plan_doc, "timing": timing, "created": C.utcnow()})
        C.log("[stop] infeasible under the declared time rule; nothing trained formally")
        return 2

    if not smoke and epochs != plan_doc["epochs"]:
        raise RuntimeError("declared epochs disagree with the timing plan")

    formal = run_formal(cfg, arm, norm, device, epochs=epochs,
                        train_cache=train_cache, y=y, rows=rows,
                        valid_cache=valid_cache, valid_rows=valid_rows, timing=timing,
                        run=run, ana=ana, smoke=smoke)
    scored = score_checkpoints(cfg, arm, norm, device, train_cache=train_cache, y=y,
                               rows=rows, valid_cache=valid_cache,
                               valid_rows=valid_rows, run=run, ana=ana)
    total_seconds = time.perf_counter() - job_start
    curve = formal["curve"]
    tail = [r["valid_mae"] for r in curve[-100:]]
    info = {
        "protocol": LC.PROTOCOL, "arm": arm, "arm_name": ARMS[arm]["name"],
        "candidate": ARMS[arm]["candidate"], "seed": int(LC.SEED), "smoke": bool(smoke),
        "git_commit": C.git_commit(), "environment": C.environment(),
        "config_path": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
        "data": {"source": "official PyG ZINC subset=True train/valid",
                 "n_train": int(len(y)), "n_valid": int(len(y_valid)),
                 "train_label_sha256": C.sha256_array(y),
                 "valid_label_sha256": C.sha256_array(y_valid),
                 "official_test_loaded_before_freeze": False},
        "gate": gate_info,
        "model_info": formal["info"],
        "init_hashes": formal["init_hashes"],
        "parameter_count": int(formal["info"]["parameters"]),
        "recipe": {
            "optimizer": "adamw", "lr": float(cfg["optimizer"]["lr"]),
            "min_lr": float(cfg["optimizer"]["min_lr"]),
            "weight_decay": float(cfg["optimizer"]["weight_decay"]),
            "betas": cfg["optimizer"]["betas"], "eps": float(cfg["optimizer"]["eps"]),
            "grad_clip": float(cfg["optimizer"]["grad_clip"]),
            "warmup_updates": int(cfg["schedule"]["warmup_updates"]),
            "batch_size": int(cfg["schedule"]["batch_size"]),
            "drop_last": bool(cfg["schedule"]["drop_last"]),
            "epochs": int(formal["epochs_run"]),
            "steps_per_epoch": 79, "updates": int(formal["total_updates"]),
            "actual_global_steps": int(formal["global_step"]),
            "loss": "raw-y L1", "valid_every_epoch": True, "dropout": 0.0,
            "early_stop": None, "ema_soup_swa": False,
        },
        "timing": timing, "plan": plan_doc,
        "epoch0_valid_diagnostic": float(formal["epoch0_valid"]),
        "best_valid": formal["best"], "scored": scored,
        "valid_tail_100": {"n": len(tail), "median": float(np.median(tail)),
                           "min": float(np.min(tail)), "mean": float(np.mean(tail)),
                           "best_epoch_in_tail": bool(
                               int(formal["best"]["epoch"]) > int(formal["epochs_run"]) - 100)},
        "intervention": {
            "w_s_norm_first": float(curve[0]["w_s_norm"]),
            "w_s_norm_last": float(curve[-1]["w_s_norm"]),
            "phi_norm_last": float(curve[-1]["phi_norm"]),
            "phi_epoch_delta_first": float(curve[0]["phi_epoch_delta"]),
            "phi_epoch_delta_last": float(curve[-1]["phi_epoch_delta"]),
            "port_rms_first": float(curve[0]["port_rms"]),
            "port_rms_last": float(curve[-1]["port_rms"]),
            "s_rms_last": float(curve[-1]["s_rms"]),
        },
        "curve": curve,
        "wall_seconds": float(total_seconds),
        "formal_train_seconds": float(formal["formal_seconds"]),
        "gpu_hours": (None if device.type == "cpu" else float(total_seconds) / 3600.0),
        "notes": [
            "best-valid uses strict <, ties keep the earlier epoch",
            "online_train_mae is the within-epoch pre-update normal-weight mean",
            "train_mae_eval_mode is the full-10k eval-mode score of a frozen checkpoint",
            "mean_sq is taken on the permutation-averaged phi_bar",
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
        "protocol": LC.PROTOCOL, "arm": arm, "arm_name": ARMS[arm]["name"],
        "candidate": ARMS[arm]["candidate"], "seed": int(LC.SEED),
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
        "init_state_sha256": formal["init_hashes"]["init_state_sha256"],
        "public_state_sha256": formal["init_hashes"]["public_state_sha256"],
        "ssl_sha256": formal["info"]["ssl_encoders"]["sha256"],
        "params": int(formal["info"]["parameters"]),
        "epochs_run": int(formal["epochs_run"]),
        "outer_updates": int(formal["global_step"]),
        "gradient_calculations": int(formal["global_step"]),
        "recipe": info["recipe"], "data": info["data"],
        "test_policy": "one descriptive read after this freeze; no test-driven change",
        "frozen_at": C.utcnow(),
    }
    freeze_doc = freeze(ana / "freeze.json", freeze_doc)
    C.log(f"[freeze {arm}] best sha256 {freeze_doc['best_sha256']}")
    report = run_test(cfg, arm, norm, device, freeze_doc=freeze_doc, run=run, ana=ana)
    C.log(f"[test {arm}] test MAE {report['test']['mae']:.6f}")
    C.write_json(ana / "status.json", {
        "status": "COMPLETE", "wall_seconds": float(time.perf_counter() - job_start),
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
                   help="smoke only: !=600 writes runs/*_smokeN and never freezes/tests")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = LC.load_config()
    if args.command == "test":
        return command_test(cfg, args.arm, args)
    return command_run(cfg, args.arm, args)


if __name__ == "__main__":
    raise SystemExit(main())
