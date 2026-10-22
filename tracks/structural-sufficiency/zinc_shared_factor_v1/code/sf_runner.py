"""Runner for ``zinc_shared_factor_v1`` (arms D / T / N, seeds 0 and 1).

Per arm ``(arm, seed)`` the job performs, in order:

1. the pre-training gate (``analysis/cpu_checks.json`` must pass for *this*
   config file, the arm's declared parameter count and initial-state hash must
   match, and for **N** the paired T arm's recorded initial state hash and
   batch-order hashes must match);
2. 4 timing epochs (written separately, discarded);
3. a feasibility plan: the horizon is fixed at 600 epochs / 47,400 updates.  If
   ``1.15 * t_slow * 600 + 120`` exceeds the 3,600 s per-arm hard stop the arm
   **stops** and reports the cost; the epoch count is never shortened (a
   shortened candidate would not be comparable to the full D arm);
4. the formal 600-epoch run from the same seeds, a fresh optimiser and a fresh
   order generator, with full-valid evaluation every epoch and strict-``<``
   single-checkpoint selection;
5. normal-weight scoring of best/last (full valid + full eval-mode train),
   per-molecule predictions, and a freeze of the best checkpoint hash.

No official-test data is read by this round: there is no test command.

Usage::

    python .../sf_runner.py run --arm T --seed 0 --device cuda:0
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
_LF = _SS / "zinc_local_factorization_v1" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_BF), str(_LF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import sf_core as SC  # noqa: E402

ARMS = {"D": "SF-Dense", "T": "SF-SharedState", "N": "SF-IndependentCopies"}
# the frozen config declares the P3 carrier for seed 0 only (a scalar); later seeds were
# not part of that pre-registration and take the value from the supplement file.
SUPPLEMENT_HASHES = (SC.EXP.parent / "zinc_shared_factor_supplement_v1" / "configs"
                     / "supplement_hashes.json")
PER_ARM_HARD_STOP_SECONDS = 3600.0
RESERVE_SECONDS = 120.0


def paths(arm: str, seed: int, tag: str = "") -> dict[str, Path]:
    suffix = f"_{tag}" if tag else ""
    return {
        "run": SC.RUNS / f"SF_{arm}_s{seed}{suffix}",
        "analysis": SC.ANALYSIS / f"SF_{arm}_s{seed}{suffix}",
    }


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def declared_carrier_hash(cfg: dict, seed: int) -> tuple[str, str]:
    """Declared P3-carrier hash for one seed, plus which artifact declared it.

    ``configs/sf_atomic_v1.json`` pins the carrier as a single scalar, which is the
    seed-0 value. That file is frozen, so seeds registered later (the supplement
    study) declare their carrier hash in the supplement pre-registration file
    instead. Both sources are hashes of the *initial* carrier, recorded before
    training.
    """
    declared = cfg["model"]["carrier_initial_state_sha256"]
    if isinstance(declared, dict):
        return str(declared[str(seed)]), "config"
    if int(seed) == 0:
        return str(declared), "config"
    doc = C.read_json(SUPPLEMENT_HASHES)
    return str(doc["carrier_initial_state_sha256"][str(seed)]), str(
        SUPPLEMENT_HASHES.relative_to(C.REPO))


def gate(cfg: dict, norm: dict, arm: str, seed: int, device) -> dict:
    checks_path = SC.ANALYSIS / "cpu_checks.json"
    if not checks_path.exists():
        raise RuntimeError(f"missing {checks_path}; run sf_checks.py first")
    doc = C.read_json(checks_path)
    if doc.get("config_sha256") != cfg["_config_sha256"]:
        raise RuntimeError("cpu_checks.json was produced for a different config file")
    if int(doc.get("n_fail", 1)) != 0:
        raise RuntimeError(f"cpu checks failed: {doc.get('n_fail')} failing")
    failed = [k for k, v in doc["checks"].items() if not v.get("pass")]
    if failed:
        raise RuntimeError(f"failing checks: {failed}")

    model, info = SC.build(arm, norm, torch.device("cpu"), seed=seed)
    hashes = SC.init_hashes(model)
    expected = cfg["model"]["arms"][arm]
    if int(info["parameters"]) != int(expected["parameters_expected"]):
        raise RuntimeError(f"{arm}: parameter count {info['parameters']}")
    if hashes["init_state_sha256"] != str(expected["init_state_sha256"][str(seed)]):
        raise RuntimeError(f"{arm}: init state hash {hashes['init_state_sha256']} != config")
    if hashes["content_state_sha256"] != str(expected["content_state_sha256"][str(seed)]):
        raise RuntimeError(f"{arm}: content state hash != config")
    declared_carrier, carrier_source = declared_carrier_hash(cfg, seed)
    if hashes["public_state_sha256"] != declared_carrier:
        raise RuntimeError(f"{arm}: public state hash {hashes['public_state_sha256']} "
                           f"!= declared P3 carrier {declared_carrier} ({carrier_source})")
    if not hashes["w_s_zero"]:
        raise RuntimeError(f"{arm}: W_s is not exactly zero")
    if info["ssl_encoders"]["sha256"] != cfg["ssl"]["sha256"]:
        raise RuntimeError(f"{arm}: SSL sha256 mismatch")
    opt = SC.make_optimizer(model, cfg, int(cfg["schedule"]["outer_updates"]))
    if C.optimizer_parameter_count(opt) != int(info["parameters"]):
        raise RuntimeError(f"{arm}: optimizer does not cover all parameters")
    if any(isinstance(m, torch.nn.Dropout) and float(m.p) != 0.0 for m in model.modules()):
        raise RuntimeError(f"{arm}: unexpected non-zero dropout")
    if any("BatchNorm" in type(m).__name__ for m in model.modules()):
        raise RuntimeError(f"{arm}: unexpected BatchNorm")

    out = {
        "cpu_checks_n_pass": int(doc["n_pass"]), "cpu_checks_n_fail": int(doc["n_fail"]),
        "cpu_checks_config_sha256": doc.get("config_sha256"),
        "arm": arm, "seed": int(seed), "parameters": int(info["parameters"]),
        "optimizer_covers_all": True, "ssl_sha256": info["ssl_encoders"]["sha256"],
        "carrier_declared": declared_carrier, "carrier_declared_source": carrier_source,
        **hashes,
    }

    # N must use the *paired* T initial state and the same data order.
    if arm == "N":
        t_init = SC.ANALYSIS / f"SF_T_s{seed}" / "init.json"
        if not t_init.exists():
            raise RuntimeError(f"missing paired T init record {t_init}; run T seed {seed} first")
        tdoc = C.read_json(t_init)
        if tdoc["init_state_sha256"] != hashes["init_state_sha256"]:
            raise RuntimeError("N init state hash differs from the paired T arm")
        if tdoc["content_state_sha256"] != hashes["content_state_sha256"]:
            raise RuntimeError("N content parameters differ from the paired T arm")
        out["paired_T_init"] = {
            "path": str(t_init.relative_to(C.REPO)),
            "init_state_sha256": tdoc["init_state_sha256"],
            "content_state_sha256": tdoc["content_state_sha256"],
            "identical": True,
        }
    steps = math.ceil(cfg["data"]["train"] / cfg["schedule"]["batch_size"])
    total = steps * int(cfg["schedule"]["epochs"])
    if steps != 79 or total != int(cfg["schedule"]["outer_updates"]):
        raise RuntimeError(f"schedule drift: {steps} steps, {total} updates")
    out.update({"steps_per_epoch": int(steps), "outer_updates": int(total),
                "device": str(device),
                "cuda_visible_devices": str(torch.cuda.device_count()) + " visible"})
    return out


# ---------------------------------------------------------------------------
# timing / plan
# ---------------------------------------------------------------------------


def run_timing(cfg: dict, arm: str, seed: int, norm: dict, device, *, epochs: int,
               train_cache, y, rows, valid_cache, valid_rows) -> dict:
    model, info = SC.build(arm, norm, device, seed=seed)
    hashes = SC.init_hashes(model)
    opt = SC.make_optimizer(model, cfg, 79 * int(cfg["schedule"]["epochs"]))
    gen = torch.Generator().manual_seed(int(seed) + int(cfg["schedule"]["shuffle_seed_offset"]))
    curve = []
    step = 0
    _, epoch0_valid = SC.evaluate(model, valid_cache, valid_rows, device,
                                  int(cfg["schedule"]["valid_batch_size"]))
    total = 79 * int(epochs)
    for ep in range(1, int(epochs) + 1):
        rec, step = SC.train_epoch(model, opt, train_cache, rows, y, gen, cfg=cfg, epoch=ep,
                                   total_updates=total, global_step=step, device=device)
        _, valid_mae = SC.evaluate(model, valid_cache, valid_rows, device,
                                   int(cfg["schedule"]["valid_batch_size"]))
        rec["valid_mae"] = float(valid_mae)
        curve.append(rec)
        C.log(f"[timing {arm}{seed}] ep={ep} valid={valid_mae:.6f} "
              f"online={rec['online_train_mae']:.6f} {rec['seconds']:.2f}s "
              f"content_delta={rec['content_epoch_delta']:.3e}")
    steady = float(np.mean([r["seconds"] for r in curve[1:]])) if len(curve) > 1 else \
        float(curve[0]["seconds"])
    return {
        "arm": arm, "arm_name": ARMS[arm], "seed": int(seed),
        "timing_epochs": int(epochs),
        "epoch0_valid_diagnostic": float(epoch0_valid),
        "curve": curve, "steady_epoch_seconds": steady,
        "parameters": int(info["parameters"]), "model_info": info, "init_hashes": hashes,
        "created": C.utcnow(),
    }


def plan(cfg: dict, arm: str, seed: int, timing: dict, *, used_seconds: float) -> dict:
    t_epoch = float(timing["steady_epoch_seconds"])
    epochs = int(cfg["schedule"]["epochs"])
    projected = 1.15 * t_epoch * epochs + RESERVE_SECONDS
    doc = {
        "protocol": SC.PROTOCOL, "arm": arm, "arm_name": ARMS[arm], "seed": int(seed),
        "steady_epoch_seconds": t_epoch,
        "used_seconds_before_plan": float(used_seconds),
        "epochs": int(epochs),
        "steps_per_epoch": 79,
        "total_updates": int(79 * epochs),
        "projected_train_seconds": float(1.15 * t_epoch * epochs),
        "projected_total_seconds": float(projected),
        "hard_stop_seconds": PER_ARM_HARD_STOP_SECONDS,
        "feasible": bool(projected <= PER_ARM_HARD_STOP_SECONDS),
        "epoch_reduction": "none by design: an infeasible plan stops the round instead of "
                           "truncating the candidate",
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
        "note": "no architecture / LR / channel / state / horizon selection from timing or valid",
        "created": C.utcnow(),
    }
    return doc


# ---------------------------------------------------------------------------
# formal run
# ---------------------------------------------------------------------------


def run_formal(cfg: dict, arm: str, seed: int, norm: dict, device, *, epochs: int,
               train_cache, y, rows, valid_cache, valid_rows,
               timing: dict, run: Path, ana: Path, smoke: bool = False) -> dict:
    run.mkdir(parents=True, exist_ok=True)
    ana.mkdir(parents=True, exist_ok=True)
    model, info = SC.build(arm, norm, device, seed=seed)
    hashes = SC.init_hashes(model)
    if hashes["init_state_sha256"] != timing["init_hashes"]["init_state_sha256"]:
        raise RuntimeError(f"formal init {hashes['init_state_sha256']} != timing init")
    if not smoke:
        expected = cfg["model"]["arms"][arm]["init_state_sha256"][str(seed)]
        if hashes["init_state_sha256"] != str(expected):
            raise RuntimeError("formal init != config init hash")
    opt = SC.make_optimizer(model, cfg, 79 * int(epochs))
    gen = torch.Generator().manual_seed(int(seed) + int(cfg["schedule"]["shuffle_seed_offset"]))
    steps = 79
    total = steps * int(epochs)
    valid_batch = int(cfg["schedule"]["valid_batch_size"])

    _, epoch0_valid = SC.evaluate(model, valid_cache, valid_rows, device, valid_batch)
    C.log(f"[{arm}{seed}] epoch0-diagnostic valid={epoch0_valid:.9f} (not a candidate)")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    curve: list[dict] = []
    best: dict[str, Any] = {"valid_mae": float("inf"), "epoch": -1}
    step = 0
    started = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        rec, step = SC.train_epoch(model, opt, train_cache, rows, y, gen, cfg=cfg, epoch=ep,
                                   total_updates=total, global_step=step, device=device)
        _, valid_mae = SC.evaluate(model, valid_cache, valid_rows, device, valid_batch)
        rec["valid_mae"] = float(valid_mae)
        is_best = bool(rec["valid_mae"] < best["valid_mae"])
        if is_best:
            best = {"valid_mae": float(rec["valid_mae"]), "epoch": int(ep)}
            C.atomic_torch_save(
                {"state_dict": C.cpu_state(model), "epoch": int(ep),
                 "valid_mae": float(rec["valid_mae"]), "candidate": f"SF-{arm}",
                 "seed": int(seed), "arm": arm, "global_step": int(step)},
                run / "best.pt")
        rec["is_best"] = is_best
        curve.append(rec)
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(model), "epoch": int(ep),
             "valid_mae": float(rec["valid_mae"]), "best": dict(best),
             "global_step": int(step), "candidate": f"SF-{arm}", "seed": int(seed),
             "arm": arm},
            run / "last.pt")
        C.log(f"[{arm}{seed}] ep={ep:03d} valid={rec['valid_mae']:.6f} "
              f"online={rec['online_train_mae']:.6f} lr={rec['lr']:.3e} "
              f"best={best['valid_mae']:.6f}@{best['epoch']} {rec['seconds']:.2f}s "
              f"w_s={rec['w_s_norm']:.4f} port={rec['port_rms']:.4f} "
              f"dcontent={rec['content_epoch_delta']:.4f}")
        if ep % 5 == 0 or ep == int(epochs):
            C.write_curve_csv(run / "curve.csv", curve)
    formal_seconds = time.perf_counter() - started
    if step != total:
        raise RuntimeError(f"actual outer updates {step} != {total}")
    peak_alloc = (torch.cuda.max_memory_allocated(device) / 2 ** 20) if device.type == "cuda" else 0.0
    peak_reserved = (torch.cuda.max_memory_reserved(device) / 2 ** 20) \
        if device.type == "cuda" else 0.0
    C.atomic_torch_save(
        {"state_dict": C.cpu_state(model), "optimizer": opt.state_dict(),
         "gen_state": gen.get_state(), "epoch": int(epochs),
         "valid_mae": float(curve[-1]["valid_mae"]), "global_step": int(step),
         "best": dict(best), "curve": curve, "candidate": f"SF-{arm}", "seed": int(seed),
         "arm": arm},
        run / "final.pt")
    C.write_curve_csv(run / "curve.csv", curve)
    C.write_json(run / "order_hashes.json", {
        "arm": arm, "seed": int(seed),
        "generator_seed": int(seed) + int(cfg["schedule"]["shuffle_seed_offset"]),
        "n_molecules": int(len(rows)),
        "epochs": [r["order_sha256"] for r in curve],
    })
    return {"info": info, "init_hashes": hashes, "curve": curve, "best": best,
            "epoch0_valid": float(epoch0_valid), "epochs_run": int(epochs),
            "global_step": int(step), "total_updates": int(total),
            "formal_seconds": float(formal_seconds),
            "peak_gpu_allocated_mib": float(peak_alloc),
            "peak_gpu_reserved_mib": float(peak_reserved)}


# ---------------------------------------------------------------------------
# scoring / freeze
# ---------------------------------------------------------------------------


def write_preds_csv(path: Path, rows: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = __import__("csv").writer(fh)
        w.writerow(["row", "y_raw", "pred"])
        for r, t, p in zip(rows, y, pred):
            w.writerow([int(r), f"{float(t):.9f}", f"{float(p):.9f}"])


def score_checkpoints(cfg: dict, arm: str, seed: int, norm: dict, device, *, train_cache, y,
                      rows, valid_cache, valid_rows, run: Path, ana: Path) -> dict:
    ana.mkdir(parents=True, exist_ok=True)
    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    out: dict[str, Any] = {}
    for tag in ("best", "final"):
        ck = torch.load(run / f"{tag}.pt", map_location="cpu", weights_only=False)
        model, _ = SC.build(arm, norm, device, seed=seed)
        C.load_state(model, ck["state_dict"], device)
        pred_valid, mae_valid = SC.evaluate(model, valid_cache, valid_rows, device,
                                           int(cfg["schedule"]["valid_batch_size"]))
        _, mae_train = SC.evaluate(model, train_cache, rows, device,
                                   int(cfg["schedule"]["eval_train_batch_size"]))
        if tag == "best" and abs(mae_valid - float(ck["valid_mae"])) > 1e-6:
            raise RuntimeError(f"reloaded best valid {mae_valid} != stored {ck['valid_mae']}")
        np.savez(pred_dir / f"valid_{tag}.npz", row=valid_rows, pred=pred_valid,
                 y=np.asarray(valid_cache["labels"], dtype=np.float32))
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


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def command_run(cfg: dict, arm: str, seed: int, args) -> int:
    formal_epochs = int(cfg["schedule"]["epochs"])
    smoke = args.epochs is not None and int(args.epochs) != formal_epochs
    epochs = int(args.epochs) if args.epochs is not None else formal_epochs
    p = paths(arm, seed, tag=f"smoke{epochs}" if smoke else "")
    ana, run = p["analysis"], p["run"]
    if smoke:
        C.log(f"[smoke] {arm} seed {seed} {epochs} epochs into {run} (never frozen)")
    ana.mkdir(parents=True, exist_ok=True)
    if (run / "train.json").exists() and not args.force:
        raise RuntimeError(f"{run}/train.json exists; use --force")
    job_start = time.perf_counter()
    norm = D.load_norm()
    device = torch.device(args.device)
    gate_info = gate(cfg, norm, arm, seed, device)
    C.log(f"[gate {arm}{seed}] {gate_info}")
    train_cache, y, rows = SC.load_split("train")
    valid_cache, y_valid, valid_rows = SC.load_split("valid")
    C.log(f"[{arm}{seed}] data ready ({time.perf_counter() - job_start:.1f}s) on {device}")

    timing = run_timing(cfg, arm, seed, norm, device, epochs=int(args.timing_epochs),
                        train_cache=train_cache, y=y, rows=rows,
                        valid_cache=valid_cache, valid_rows=valid_rows)
    C.write_json(ana / "timing.json", timing)
    used = time.perf_counter() - job_start
    plan_doc = plan(cfg, arm, seed, timing, used_seconds=used)
    plan_doc["gate"] = gate_info
    C.write_json(ana / "plan.json", plan_doc)
    C.log(f"[plan {arm}{seed}] t_slow={timing['steady_epoch_seconds']:.3f}s "
          f"E={plan_doc['epochs']} projected={plan_doc['projected_total_seconds']:.0f}s "
          f"feasible={plan_doc['feasible']}")
    C.write_json(ana / "init.json", {
        "protocol": SC.PROTOCOL, "arm": arm, "seed": int(seed),
        "init_state_sha256": gate_info["init_state_sha256"],
        "content_state_sha256": gate_info["content_state_sha256"],
        "public_state_sha256": gate_info["public_state_sha256"],
        "parameters": gate_info["parameters"],
        "ssl_sha256": gate_info["ssl_sha256"],
        "declared_init_state_sha256":
            str(cfg["model"]["arms"][arm]["init_state_sha256"][str(seed)]),
        "declared_content_state_sha256":
            str(cfg["model"]["arms"][arm]["content_state_sha256"][str(seed)]),
        "created": C.utcnow()})
    if not plan_doc["feasible"]:
        C.write_json(ana / "status.json", {
            "status": "INCOMPLETE", "reason":
            "measured epoch time projects past the 3600 s per-arm hard stop; the horizon is "
            "fixed at 600 epochs by pre-registration, so nothing is trained formally and the "
            "cost is reported instead of shortening the candidate",
            "plan": plan_doc, "timing": timing, "created": C.utcnow()})
        C.log("[stop] infeasible under the declared per-arm stop; nothing trained formally")
        return 2

    formal = run_formal(cfg, arm, seed, norm, device, epochs=epochs,
                        train_cache=train_cache, y=y, rows=rows,
                        valid_cache=valid_cache, valid_rows=valid_rows, timing=timing,
                        run=run, ana=ana, smoke=smoke)
    scored = score_checkpoints(cfg, arm, seed, norm, device, train_cache=train_cache, y=y,
                               rows=rows, valid_cache=valid_cache,
                               valid_rows=valid_rows, run=run, ana=ana)
    total_seconds = time.perf_counter() - job_start
    curve = formal["curve"]
    tail = [r["valid_mae"] for r in curve[-100:]]
    info = {
        "protocol": SC.PROTOCOL, "arm": arm, "arm_name": ARMS[arm],
        "candidate": f"SF-{arm}", "seed": int(seed), "smoke": bool(smoke),
        "git_commit": C.git_commit(), "environment": C.environment(),
        "config_path": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
        "data": {"source": "official PyG ZINC subset=True train/valid",
                 "n_train": int(len(y)), "n_valid": int(len(y_valid)),
                 "train_label_sha256": C.sha256_array(y),
                 "valid_label_sha256": C.sha256_array(y_valid),
                 "official_test_loaded": False},
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
            "content_norm_last": float(curve[-1]["content_norm"]),
            "content_epoch_delta_first": float(curve[0]["content_epoch_delta"]),
            "content_epoch_delta_last": float(curve[-1]["content_epoch_delta"]),
            "port_rms_first": float(curve[0]["port_rms"]),
            "port_rms_last": float(curve[-1]["port_rms"]),
            "s_rms_last": float(curve[-1]["s_rms"]),
        },
        "curve": curve,
        "wall_seconds": float(total_seconds),
        "formal_train_seconds": float(formal["formal_seconds"]),
        "gpu_hours": (None if device.type == "cpu" else float(total_seconds) / 3600.0),
        "peak_gpu_allocated_mib": float(formal["peak_gpu_allocated_mib"]),
        "peak_gpu_reserved_mib": float(formal["peak_gpu_reserved_mib"]),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "notes": [
            "best-valid uses strict <, ties keep the earlier epoch",
            "online_train_mae is the within-epoch pre-update normal-weight mean",
            "train_mae_eval_mode is the full-10k eval-mode score of a frozen checkpoint",
            "no official-test data was loaded in this round",
        ],
    }
    C.write_json(run / "train.json", info)
    if (run / "curve.csv").exists():
        (ana / "curve.csv").write_text((run / "curve.csv").read_text(encoding="utf-8"),
                                       encoding="utf-8")
    C.write_json(ana / "status.json", {
        "status": "SMOKE_COMPLETE" if smoke else "TRAINED",
        "wall_seconds": float(total_seconds),
        "gpu_hours": float(total_seconds) / 3600.0, "created": C.utcnow()})
    C.log(f"[{arm}{seed}] trained best={formal['best']['valid_mae']:.6f}@"
          f"{formal['best']['epoch']} wall={total_seconds:.1f}s "
          f"gpu_h={total_seconds / 3600.0:.3f}")
    if smoke:
        C.log(f"[smoke] {arm} seed {seed} stopped after scoring; no freeze")
        return 0

    freeze_doc = {
        "protocol": SC.PROTOCOL, "arm": arm, "arm_name": ARMS[arm],
        "candidate": f"SF-{arm}", "seed": int(seed),
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
        "content_state_sha256": formal["init_hashes"]["content_state_sha256"],
        "public_state_sha256": formal["init_hashes"]["public_state_sha256"],
        "ssl_sha256": formal["info"]["ssl_encoders"]["sha256"],
        "params": int(formal["info"]["parameters"]),
        "epochs_run": int(formal["epochs_run"]),
        "outer_updates": int(formal["global_step"]),
        "gradient_calculations": int(formal["global_step"]),
        "recipe": info["recipe"], "data": info["data"],
        "test_policy": "official test not read in this round",
        "frozen_at": C.utcnow(),
    }
    freeze_doc = freeze(ana / "freeze.json", freeze_doc)
    C.log(f"[freeze {arm}{seed}] best sha256 {freeze_doc['best_sha256']}")
    C.write_json(ana / "status.json", {
        "status": "COMPLETE", "wall_seconds": float(time.perf_counter() - job_start),
        "created": C.utcnow()})
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--seed", type=int, default=0, choices=[0, 1])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timing-epochs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=None,
                   help="smoke only: !=600 writes runs/*_smokeN and never freezes")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = SC.load_config()
    return command_run(cfg, args.arm, int(args.seed), args)


if __name__ == "__main__":
    raise SystemExit(main())
