"""HIS-v0 trainer: frozen LCRC-A recipe + one optional third-order penalty.

Two arms, identical in every other respect:

  REPLAY : the frozen LCRC-A training loss, bit-for-bit (no penalty term is even
           constructed, so REPLAY reproduces the frozen trainer's trajectory).
  HIS    : ``loss = L1(pred, y) + lambda * mean(H_3 ** 2)`` with the frozen
           ``lambda`` from ``configs/his_v0.json``.

``H_3`` is the pure three-way ANOVA residual of the path-code read-out
(``his_anova``).  Inference, inputs, initialisation, parameter count (237,519)
and checkpoint layout are unchanged; the penalty adds no deployment parameters.

Only official train/valid are read.  The official test split is never
constructed.

CLI
---
    python his_train.py calibrate --seed 0                      # freeze lambda
    python his_train.py train --arm REPLAY --seed 0 [--epochs 2]
    python his_train.py train --arm HIS --seed 0 [--epochs 2]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

import his_anova as A
import his_common as HIS

L, D, M, T = HIS.L, HIS.D, HIS.M, HIS.T

RECIPE = dict(T.RECIPE)                      # frozen; never edited here
CONFIG_PATH = HIS.CONFIGS / "his_v0.json"
LAMBDA_PATH = HIS.ANALYSIS / "lambda_calibration.json"
ARMS = ("REPLAY", "HIS")


# ---------------------------------------------------------------------------
# frozen config / calibration
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"missing frozen config {HIS.track_rel(CONFIG_PATH)}")
    return HIS.read_json(CONFIG_PATH)


def frozen_lambda() -> float:
    cfg = load_config()
    lam = cfg["his"].get("lambda")
    if lam is None:
        raise RuntimeError(
            "configs/his_v0.json has no frozen lambda; run `his_train.py calibrate` "
            "on the training host first and commit the value")
    lam = float(lam)
    if not math.isfinite(lam) or lam <= 0:
        raise RuntimeError(f"frozen lambda is not a positive finite number: {lam}")
    lo, hi = float(cfg["his"]["lambda_guards"]["min"]), float(cfg["his"]["lambda_guards"]["max"])
    if not (lo <= lam <= hi):
        raise RuntimeError(f"frozen lambda {lam} outside pre-registered guards [{lo}, {hi}]")
    return lam


def calibrate(seed: int, device: str, *, write: bool = True, repeats: int = 1) -> dict:
    """Gradient-norm matched lambda on the frozen first batch and initial model.

    No optimizer step, no valid/test access, RNG isolated.  ``lambda =
    0.10 * ||g_L1|| / ||g_reg||`` over the affected parameters only
    (``path_emb.weight`` and the 32 ``W_path`` columns of the first layer).

    ``repeats > 1`` re-runs the identical computation in the same process and
    reports the spread: CUDA ``index_add_`` reductions are not bit-deterministic,
    so this measures how reproducible ``lambda`` is on the execution regime
    before the value is frozen.
    """
    norm = D.load_norm()
    train = D.load_cache("train")
    model, info = M.build_model("A", norm, seed, device, dict(M.DEFAULT_CONFIG))
    if info["initial_state_sha256"] != HIS.FROZEN["lcrc_a_s0_initial_state_sha256"]:
        raise RuntimeError("initial state differs from the frozen LCRC-A seed-0 state")

    gen = torch.Generator().manual_seed(int(seed) + 101)     # frozen data-order seed
    order = torch.randperm(int(train["n_molecules"]), generator=gen).numpy()
    rows = order[: int(RECIPE["batch_size"])]
    with L.C.isolated_rng():
        batch = T._to_device(D.collate(train, rows), device)
        target = torch.as_tensor(np.asarray(train["labels"])[rows], dtype=torch.float32,
                                 device=device)
        pred = model(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        g_l1 = A.affected_grad_vector(model).clone()
        model.zero_grad(set_to_none=True)
        pen, pen_stats = A.his_penalty(model)
        pen.backward()
        g_reg = A.affected_grad_vector(model).clone()
        outside = A.outside_slice_grad_max(model)
        all_zero = all(
            (p.grad is None or float(p.grad.abs().max()) == 0.0)
            for n, p in model.named_parameters()
            if n not in ("local.path_emb.weight", "local.mlp.lins.0.weight"))
        model.zero_grad(set_to_none=True)

    n_l1 = float(g_l1.norm())
    n_reg = float(g_reg.norm())
    lam = 0.10 * n_l1 / n_reg if n_reg > 0 else float("inf")
    spread = None
    if repeats > 1:
        vals = [float(lam)]
        for _ in range(repeats - 1):
            with L.C.isolated_rng():
                model.zero_grad(set_to_none=True)
                pred2 = model(batch)
                l2 = F.l1_loss(pred2.view(-1), target.view(-1))
                model.zero_grad(set_to_none=True)
                l2.backward()
                g1 = A.affected_grad_vector(model).clone()
                model.zero_grad(set_to_none=True)
                pen2, _ = A.his_penalty(model)
                pen2.backward()
                g2 = A.affected_grad_vector(model).clone()
                model.zero_grad(set_to_none=True)
                vals.append(0.10 * float(g1.norm()) / float(g2.norm()))
        rel = [abs(v - lam) / max(abs(lam), 1e-30) for v in vals]
        spread = {"lambda_values": vals, "repeats": int(repeats),
                  "max_rel_deviation_from_first": float(max(rel))}
        L.log(f"[calibrate] repeats={repeats} lambda={vals} "
              f"max_rel_dev={spread['max_rel_deviation_from_first']:.3e}")
    out = {
        "round": HIS.ROUND_ID,
        "created": HIS.now(),
        "git_commit": L.C.git_commit(),
        "device": device,
        "environment": L.env_summary(),
        "seed": int(seed),
        "first_batch_rows": rows.tolist(),
        "first_batch_rows_sha256": L.C.sha256_array(rows),
        "data_order_seed": int(seed) + 101,
        "initial_state_sha256": info["initial_state_sha256"],
        "parameters": info["parameters"],
        "affected_parameters": ["local.path_emb.weight", "local.mlp.lins.0.weight[:,320:352]"],
        "affected_elements": int(g_l1.numel()),
        "l1_loss_on_first_batch": float(loss),
        "g_l1_affected_norm": n_l1,
        "g_reg_affected_norm": n_reg,
        "penalty_at_init": float(pen),
        "h3_ratio_at_init": pen_stats["h3_ratio"],
        "penalty_grad_confined_to_affected": bool(all_zero and outside == 0.0),
        "outside_slice_penalty_grad_max": outside,
        "lambda": float(lam),
        "lambda_rule": "lambda = 0.10 * ||g_L1|| / ||g_reg|| over the affected parameters",
        "lambda_guards": load_config()["his"]["lambda_guards"] if CONFIG_PATH.exists() else None,
        "lambda_repeat_spread": spread,
        "valid_or_test_used": False,
        "official_test_loaded": False,
    }
    if write:
        HIS.write_json(LAMBDA_PATH, out)
        L.log(f"[calibrate] ||g_L1||={n_l1:.6e} ||g_reg||={n_reg:.6e} -> lambda={lam:.6e}")
    return out


def _check_lambda_matches_calibration(device: str, lam: float, seed: int) -> dict:
    """Recompute the calibration inside the training process and compare."""
    fresh = calibrate(seed, device, write=False)
    rel = abs(fresh["lambda"] - lam) / max(abs(lam), 1e-30)
    tol = 1e-4      # guard only: lambda must be reproducible to 4 significant digits
    on_disk = HIS.read_json_or_none(LAMBDA_PATH)
    disk_rel = None
    if on_disk is not None:
        disk_rel = abs(float(on_disk["lambda"]) - lam) / max(abs(lam), 1e-30)
    if rel > tol:
        raise RuntimeError(f"lambda mismatch: frozen {lam!r} vs recomputed {fresh['lambda']!r} "
                           f"(rel {rel:.3e} > tol {tol:g})")
    return {"recomputed_lambda": fresh["lambda"], "rel_diff_vs_frozen": rel,
            "tolerance": tol,
            "rel_diff_vs_calibration_file": disk_rel,
            "recomputed_g_l1": fresh["g_l1_affected_norm"],
            "recomputed_g_reg": fresh["g_reg_affected_norm"],
            "first_batch_rows_match": fresh["first_batch_rows_sha256"]
            == (on_disk or fresh)["first_batch_rows_sha256"]}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return HIS.RUNS / f"{prefix}{arm}_s{int(seed)}"


def train(arm: str, seed: int, device: str, *, epochs: int | None = None,
          tag: str | None = None, force: bool = False, out_dir: Path | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    L.require_cuda(device)
    n_epochs = int(RECIPE["epochs"] if epochs is None else epochs)
    if int(seed) != 0:
        raise RuntimeError("HIS-v0 is a seed-0 screen; seed 1 must not be run")
    if tag is None and n_epochs != int(RECIPE["epochs"]):
        tag = f"{n_epochs}ep"
    out = Path(out_dir) if out_dir is not None else run_dir(arm, seed, tag)
    L.ensure_dir(out)
    train_json = out / "train.json"
    if train_json.exists() and not force:
        L.log(f"[{arm} s{seed}] already complete: {train_json}")
        return HIS.read_json(train_json)

    cfg_dict = load_config()
    lam = 0.0 if arm == "REPLAY" else frozen_lambda()
    lam_check = None
    if arm == "HIS":
        lam_check = _check_lambda_matches_calibration(device, lam, seed)

    norm = D.load_norm()
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    if train_cache["meta"]["feature_version"] != HIS.FROZEN["feature_version"]:
        raise RuntimeError("train cache feature version mismatch")
    if valid_cache["meta"]["feature_version"] != HIS.FROZEN["feature_version"]:
        raise RuntimeError("valid cache feature version mismatch")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)

    cfg = dict(M.DEFAULT_CONFIG)
    model, model_info = M.build_model("A", norm, seed, device, cfg)
    if model_info["initial_state_sha256"] != HIS.FROZEN["lcrc_a_s0_initial_state_sha256"]:
        raise RuntimeError("initial state differs from the frozen LCRC-A seed-0 state")
    if model_info["parameters"] != HIS.FROZEN["params"]:
        raise RuntimeError("parameter count changed")

    recipe = dict(RECIPE)
    if recipe["optimizer"] != "adamw":
        raise RuntimeError("frozen recipe is AdamW")
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                  weight_decay=recipe["weight_decay"])
    audit = T._param_audit(model, optimizer)
    if not audit["parameters_in_optimizer"]:
        raise RuntimeError(f"optimizer does not cover all parameters: {audit}")

    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(rows_train) / batch_size)
    total_updates = steps_per_epoch * n_epochs
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    best_sha = None
    started = time.perf_counter()
    global_step = 0
    last_lr = float(recipe["lr"])
    first_batch_rows = None
    first_batch_grad = None
    pen_first = None
    pen_last = None
    pen_min, pen_max = float("inf"), 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = rows_train[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = rows.copy()
            batch = T._to_device(D.collate(train_cache, rows), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            if lam > 0.0:
                pen, _ = A.his_penalty(model)
                if not math.isfinite(float(pen)):
                    raise RuntimeError(f"non-finite HIS penalty at epoch {epoch}")
                pen_min = min(pen_min, float(pen))
                pen_max = max(pen_max, float(pen))
                if pen_first is None:
                    pen_first = float(pen)
                pen_last = float(pen)
                total = loss + lam * pen
            else:
                total = loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if first_batch_grad is None:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                first_batch_grad = {
                    "grad_tensors": len(grads),
                    "nonzero_grad_tensors": int(sum(1 for g in grads
                                                     if torch.count_nonzero(g).item() > 0)),
                    "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "total_loss_finite": bool(torch.isfinite(total).item()),
                    "penalty": None if lam == 0.0 else float(pen),
                    # L1 + lam*reg combined, measured after backward / before clip;
                    # the calibration guarantees lam*||g_reg|| = 0.1*||g_L1|| at step 0
                    "affected_grad_norm_after_backward": float(
                        A.affected_grad_vector(model).norm()),
                }
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            global_step += 1
            last_lr = L.C.cosine_with_warmup_lr(
                global_step, total_updates, warmup, float(recipe["lr"]), float(recipe["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        _, valid_mae = T.evaluate(model, valid_cache, rows_valid, device=device,
                                  batch_size=int(recipe["valid_batch_size"]))
        with torch.no_grad():
            _, hstats = A.his_penalty(model)
        sec = time.perf_counter() - t_ep
        entry = {
            "epoch": int(epoch),
            "online_train_mae": float(online),
            "valid_mae": float(valid_mae),
            "lr": float(last_lr),
            "seconds": float(sec),
            "steps": int(steps_per_epoch),
            "h3_ratio": float(hstats["h3_ratio"]),
            "h3_rms": float(hstats["h3_rms"]),
            "h_rms": float(hstats["h_rms"]),
            "penalty": float(hstats["penalty"]),
        }
        curve.append(entry)
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = L.C.cpu_state(model)
            best_sha = T._save_checkpoint(out / "best.pt", model, arm=arm, seed=seed,
                                          epoch=epoch, valid_mae=valid_mae,
                                          extra={"runner_arm": arm, "lambda": lam})
        L.log(f"[{arm} s{seed}] ep={epoch:03d} valid={valid_mae:.6f} online={online:.6f} "
              f"lr={last_lr:.3e} h3r={hstats['h3_ratio']:.4f} best={best['valid_mae']:.6f}"
              f"@{best['epoch']} {sec:.2f}s")

    wall = time.perf_counter() - started
    final_sha = T._save_checkpoint(out / "final.pt", model, arm=arm, seed=seed,
                                   epoch=n_epochs, valid_mae=curve[-1]["valid_mae"],
                                   extra={"runner_arm": arm, "lambda": lam})
    L.C.atomic_torch_save(
        {"state_dict": L.C.cpu_state(model), "optimizer": optimizer.state_dict(),
         "gen_state": gen.get_state(), "epoch": n_epochs, "global_step": global_step,
         "best": best, "curve": curve, "arm": arm, "lambda": lam, "seed": int(seed)},
        out / "final_state.pt")

    best_model = M.load_model_from_state("A", norm, best_state, device, cfg)
    best_pred_valid, best_valid = T.evaluate(best_model, valid_cache, rows_valid, device=device,
                                             batch_size=int(recipe["valid_batch_size"]))
    _, best_train = T.evaluate(best_model, train_cache, rows_train, device=device,
                               batch_size=int(recipe["valid_batch_size"]))
    final_pred_valid, final_valid = T.evaluate(model, valid_cache, rows_valid, device=device,
                                               batch_size=int(recipe["valid_batch_size"]))
    _, final_train = T.evaluate(model, train_cache, rows_train, device=device,
                                batch_size=int(recipe["valid_batch_size"]))
    if abs(best_valid - best["valid_mae"]) > 1e-6:
        raise RuntimeError(f"best checkpoint replay mismatch: {best_valid} vs {best['valid_mae']}")
    if abs(final_valid - curve[-1]["valid_mae"]) > 1e-6:
        raise RuntimeError(f"final checkpoint replay mismatch: {final_valid} vs {curve[-1]['valid_mae']}")

    pred_dir = out / "predictions"
    L.ensure_dir(pred_dir)
    np.savez(pred_dir / "valid_best.npz", row=rows_valid, pred=best_pred_valid)
    np.savez(pred_dir / "valid_final.npz", row=rows_valid, pred=final_pred_valid)
    L.C.write_curve_csv(out / "curve.csv", curve)
    diag = T._window_diagnostics(curve)
    with torch.no_grad():
        _, best_h = A.his_penalty(best_model)
        _, final_h = A.his_penalty(model)

    payload = {
        "round": HIS.ROUND_ID,
        "candidate": f"HIS-v0-{arm}",
        "runner_arm": arm,
        "arm": "A",                      # inference arm is always LCRC-A
        "seed": int(seed),
        "status": "completed",
        "git_commit": L.C.git_commit(),
        "environment": L.env_summary(),
        "lambda": float(lam),
        "lambda_check": lam_check,
        "penalty_first_step": pen_first,
        "penalty_last_step": pen_last,
        "penalty_min": None if pen_first is None else float(pen_min),
        "penalty_max": None if pen_first is None else float(pen_max),
        "his_best_checkpoint": best_h,
        "his_final_checkpoint": final_h,
        "his_curve_epoch1": curve[0] if curve else None,
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_cache_file_sha256": L.C.sha256_file(D.cache_path("train")),
            "valid_cache_file_sha256": L.C.sha256_file(D.cache_path("valid")),
            "train_cache_content_sha256": D.content_fingerprint(train_cache),
            "valid_cache_content_sha256": D.content_fingerprint(valid_cache),
            "feature_version": HIS.FROZEN["feature_version"],
        },
        "model_info": model_info,
        "recipe": recipe,
        "warmup_steps": int(warmup),
        "steps_per_epoch": int(steps_per_epoch),
        "total_updates": int(total_updates),
        "epochs": int(n_epochs),
        "param_audit": audit,
        "gradient_audit_first_batch": first_batch_grad,
        "first_batch_rows": None if first_batch_rows is None else first_batch_rows.tolist(),
        "data_order_seed": int(seed) + 101,
        "norm": {"source": norm["source"], "fingerprint": D.norm_fingerprint(norm)},
        "ssl_sha256": L.C.sha256_file(M.ssl_path(seed)),
        "config_sha256": HIS.sha256_file(CONFIG_PATH) if CONFIG_PATH.exists() else None,
        "best": best,
        "final_valid_mae": float(final_valid),
        "best_valid_mae_eval": float(best_valid),
        "best_train_mae_eval": float(best_train),
        "final_train_mae_eval": float(final_train),
        "W20": diag["W20"],
        "T100": diag["T100"],
        "wall_seconds": float(wall),
        "gpu_hours_estimate": float(wall / 3600.0),
        "checkpoint_sha256": {"best": best_sha, "final": final_sha},
        "curve": curve,
    }
    HIS.write_json(train_json, payload)
    L.log(f"[{arm} s{seed}] done best={best['valid_mae']:.6f}@{best['epoch']} "
          f"final={final_valid:.6f} wall={wall:.1f}s lambda={lam:.6e}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HIS-v0 calibrate/train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--seed", type=int, default=0)
    p_cal.add_argument("--device", default="cuda:0")
    p_cal.add_argument("--repeats", type=int, default=1)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--arm", required=True, choices=list(ARMS))
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--device", default="cuda:0")
    p_tr.add_argument("--epochs", type=int, default=None)
    p_tr.add_argument("--tag", default=None)
    p_tr.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "calibrate":
        res = calibrate(args.seed, args.device, repeats=int(args.repeats))
        L.log(f"lambda = {res['lambda']:.6e}")
        return 0
    res = train(args.arm, args.seed, args.device, epochs=args.epochs, tag=args.tag,
                force=bool(args.force))
    L.log(f"best valid {res['best']['valid_mae']:.6f} @ {res['best']['epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
