"""ZPR-v1 trainer: frozen LCRC-A recipe, one loss switch, one passive EMA.

Two arms, identical in every respect except the training loss:

  G0  ``loss = F.l1_loss(pred, y)``                      (frozen LCRC-A path)
  G1  ``loss = F.smooth_l1_loss(pred, y, beta=0.05)``

Both arms maintain a **passive** parameter EMA (decay 0.999, initialised at
``ceil(0.25 * total_updates)`` updates, no gradient, no RNG consumption).  The
online model, the optimizer state, the data order and the LR schedule are
untouched by the EMA branch; a runtime assertion checks the RNG states around
the EMA block.

Everything else is the frozen LCRC-v1 recipe, copied *verbatim* in order of
operations from ``lc_train.train`` (which is also imported read-only and used by
the smoke gate as the reference implementation):

    AdamW lr 1e-3 -> 1e-5, wd 1e-5, warmup 500, batch 128, 600 epochs
    (79 steps/epoch, 47,400 updates), grad clip 5, dropout 0.

Checkpoint selection: per branch (online / ema) strict argmin official-valid
MAE, ties keep the earlier epoch.  Official test is never constructed.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import zpr_common as Z

L, D, M, T = Z.L, Z.D, Z.M, Z.T

RECIPE = dict(T.RECIPE)  # frozen; never edited here


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def hand_smooth_l1(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    """Independent reference for ``torch.nn.functional.smooth_l1_loss(beta=...)``.

    ``z = |d|``; ``0.5 * z^2 / beta`` for ``z < beta`` else ``z - 0.5 * beta``.
    Written as an explicit piecewise expression (not a torch reduction helper)
    so that a copy-paste error in the trainer loss cannot hide behind the same
    helper being wrong in both places.
    """
    d = (pred - target).abs()
    small = 0.5 * d * d / beta
    large = d - 0.5 * beta
    return torch.where(d < beta, small, large).mean()


def _loss(name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if name == "l1":
        return F.l1_loss(pred.view(-1), target.view(-1))
    if name == "smooth_l1":
        return F.smooth_l1_loss(pred.view(-1), target.view(-1), beta=Z.SMOOTH_L1_BETA)
    raise ValueError(f"unknown loss {name!r}")


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def ema_start_step(total_updates: int, start_frac: float = Z.EMA_START_FRAC) -> int:
    step = int(math.ceil(float(start_frac) * int(total_updates)))
    return max(1, min(step, int(total_updates)))


def ema_update_(ema_state: Sequence[torch.Tensor], model: torch.nn.Module,
                decay: float) -> None:
    """``ema <- decay * ema + (1 - decay) * online`` in place, without autograd."""
    with torch.no_grad():
        for slot, param in zip(ema_state, model.parameters()):
            slot.mul_(decay).add_(param.detach(), alpha=1.0 - decay)


def ema_init_state(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def tensor_list_hash(tensors: Sequence[torch.Tensor]) -> str:
    """Content hash of an EMA parameter list (float32, order preserved)."""
    import hashlib
    h = hashlib.sha256()
    for t in tensors:
        h.update(np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def param_state_hash(module: torch.nn.Module) -> str:
    """Hash of the module *parameters only*, matching ``tensor_list_hash`` order."""
    return tensor_list_hash([p for p in module.parameters()])


def copy_ema_into(ema_model: torch.nn.Module, ema_state: Sequence[torch.Tensor],
                  online: torch.nn.Module) -> None:
    """Copy EMA parameters *into* the evaluation model (never the reverse).

    The historical zps-round bug was ``online_module.copy_(fresh_model)``-style
    direction inversion; every call site goes through this helper and the gate
    in ``zpr_checks.py`` asserts the observable consequence (EMA evaluation is
    not the initialisation and is not the online model).
    """
    with torch.no_grad():
        for module_param, ema_slot in zip(ema_model.parameters(), ema_state):
            module_param.copy_(ema_slot)
        online_buffers = dict(online.named_buffers())
        for buf_name, buf in ema_model.named_buffers():
            src = online_buffers.get(buf_name)
            if src is not None:
                buf.copy_(src)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return Z.RUNS / f"{arm}{prefix}_s{int(seed)}"


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _save_checkpoint(path: Path, model, *, arm, seed, epoch, valid_mae,
                     policy: str, extra=None) -> str:
    payload = {
        "state_dict": L.C.cpu_state(model),
        "arm": arm,
        "policy": policy,
        "seed": int(seed),
        "epoch": int(epoch),
        "valid_mae": float(valid_mae),
        "config": model.cfg,
    }
    if extra:
        payload.update(extra)
    L.atomic_save(payload, path)
    return Z.sha256_file(path)


def train(arm: str, seed: int, device: str, *, epochs: int | None = None,
          tag: str | None = None, force: bool = False, out_dir: Path | None = None) -> dict:
    if arm not in Z.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    loss_name = Z.ARMS[arm]["loss"]
    L.require_cuda(device)
    out = Path(out_dir) if out_dir is not None else run_dir(arm, seed, tag)
    Z.ensure_dir(out)
    train_json = out / "train.json"
    if train_json.exists() and not force:
        L.log(f"[{arm} s{seed}] already complete: {train_json}")
        return Z.read_json(train_json)

    # ---- frozen inputs, verified -------------------------------------------
    norm = D.load_norm()
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    if train_cache["meta"]["feature_version"] != Z.FROZEN["feature_version"]:
        raise RuntimeError("train cache feature version mismatch")
    if valid_cache["meta"]["feature_version"] != Z.FROZEN["feature_version"]:
        raise RuntimeError("valid cache feature version mismatch")
    input_audit = {
        "train_cache_file_sha256": Z.sha256_file(D.cache_path("train")),
        "train_cache_content_sha256": D.content_fingerprint(train_cache),
        "valid_cache_file_sha256": Z.sha256_file(D.cache_path("valid")),
        "valid_cache_content_sha256": D.content_fingerprint(valid_cache),
        "norm_fingerprint": D.norm_fingerprint(norm),
        "feature_version": Z.FROZEN["feature_version"],
    }
    expect = {
        "train_cache_content_sha256": Z.FROZEN["train_cache_content_sha256"],
        "valid_cache_content_sha256": Z.FROZEN["valid_cache_content_sha256"],
        "norm_fingerprint": Z.FROZEN["norm_fingerprint"],
        "feature_version": Z.FROZEN["feature_version"],
    }
    gated = {k: input_audit[k] for k in expect}
    if gated != expect:
        raise RuntimeError(f"frozen input mismatch: {gated} != {expect}")

    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)

    # ---- model / optimizer (frozen construction path) ----------------------
    cfg = dict(M.DEFAULT_CONFIG)
    model, model_info = M.build_model("A", norm, seed, device, cfg)
    if model_info["initial_state_sha256"] != Z.FROZEN["lcrc_a_s0_initial_state_sha256"]:
        raise RuntimeError("initial state differs from the frozen LCRC-A seed-0 state")
    if model_info["parameters"] != Z.FROZEN["params"]:
        raise RuntimeError("parameter count differs from the frozen 237,519")

    # The EMA evaluation model is built once, up front, inside an isolated RNG
    # context so that it cannot perturb the online trajectory.
    with L.C.isolated_rng():
        ema_model, _ = M.build_model("A", norm, seed, device, cfg)

    recipe = dict(RECIPE)
    n_epochs = int(recipe["epochs"] if epochs is None else epochs)
    if recipe["optimizer"] != "adamw":
        raise RuntimeError("frozen recipe is AdamW")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )
    audit = T._param_audit(model, optimizer)
    if not audit["parameters_in_optimizer"]:
        raise RuntimeError(f"optimizer does not cover all parameters: {audit}")

    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(rows_train) / batch_size)
    total_updates = steps_per_epoch * n_epochs
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    if epochs is None:
        if total_updates != Z.FROZEN["total_updates"]:
            raise RuntimeError(f"formal run must be {Z.FROZEN['total_updates']} updates")
    ema_start = ema_start_step(total_updates)

    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    best_sha = None
    ema_best = {"valid_mae": float("inf"), "epoch": -1}
    ema_best_state = None
    ema_best_sha = None
    ema_state: list[torch.Tensor] | None = None
    ema_audit: dict[str, Any] = {"start_step": int(ema_start), "decay": float(Z.EMA_DECAY)}

    started = time.perf_counter()
    global_step = 0
    last_lr = float(recipe["lr"])
    first_batch_rows = None
    first_batch_grad = None
    first_loss_ref = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = rows_train[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = rows.copy()
            batch = _to_device(D.collate(train_cache, rows), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = _loss(loss_name, pred, target)
            if first_loss_ref is None:
                # independent reference on the very same tensors (step 1 only)
                ref = F.l1_loss(pred.detach().view(-1), target.view(-1)).item() \
                    if loss_name == "l1" else \
                    hand_smooth_l1(pred.detach().view(-1), target.view(-1), Z.SMOOTH_L1_BETA).item()
                first_loss_ref = {
                    "loss_name": loss_name,
                    "trainer_loss": float(loss.detach()),
                    "reference_loss": float(ref),
                    "abs_diff": abs(float(loss.detach()) - float(ref)),
                    "beta": Z.SMOOTH_L1_BETA if loss_name == "smooth_l1" else None,
                }
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_batch_grad is None:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                first_batch_grad = {
                    "grad_tensors": len(grads),
                    "nonzero_grad_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
                    "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                }
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            global_step += 1
            last_lr = L.C.cosine_with_warmup_lr(
                global_step, total_updates, warmup, float(recipe["lr"]), float(recipe["min_lr"])
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()

            # ---- passive EMA (no gradient, no RNG, online trajectory untouched)
            if global_step == ema_start:
                ema_state = ema_init_state(model)
                ema_audit["init_step"] = int(global_step)
                ema_audit["init_epoch"] = int(epoch)
                ema_audit["online_param_hash_at_init"] = param_state_hash(model)
                ema_audit["ema_param_hash_at_init"] = tensor_list_hash(ema_state)
                ema_audit["init_equals_online_params"] = (
                    ema_audit["online_param_hash_at_init"] == ema_audit["ema_param_hash_at_init"]
                )
            elif global_step > ema_start and ema_state is not None:
                if "rng_invariance" not in ema_audit:
                    torch_rng_before = torch.get_rng_state().clone()
                    gen_before = gen.get_state().clone()
                    ema_update_(ema_state, model, float(Z.EMA_DECAY))
                    ema_audit["rng_invariance"] = {
                        "step": int(global_step),
                        "torch_rng_state_unchanged": bool(torch.equal(torch.get_rng_state(), torch_rng_before)),
                        "data_order_gen_state_unchanged": bool(torch.equal(gen.get_state(), gen_before)),
                        "ema_update_under_no_grad": True,
                    }
                else:
                    ema_update_(ema_state, model, float(Z.EMA_DECAY))
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)

        online_train = total_err / max(n_mol, 1)
        _, valid_mae = T.evaluate(model, valid_cache, rows_valid, device=device,
                                  batch_size=int(recipe["valid_batch_size"]))
        ema_valid = None
        if ema_state is not None:
            copy_ema_into(ema_model, ema_state, model)
            _, ema_valid = T.evaluate(ema_model, valid_cache, rows_valid, device=device,
                                      batch_size=int(recipe["valid_batch_size"]))
            if ema_valid < ema_best["valid_mae"]:
                ema_best = {"valid_mae": float(ema_valid), "epoch": int(epoch)}
                ema_best_state = L.C.cpu_state(ema_model)
                ema_best_sha = _save_checkpoint(out / "ema_best.pt", ema_model, arm=arm,
                                                seed=seed, epoch=epoch, valid_mae=ema_valid,
                                                policy="ema",
                                                extra={"ema_decay": float(Z.EMA_DECAY),
                                                       "ema_start_step": int(ema_start)})
        sec = time.perf_counter() - t_ep
        curve.append({
            "epoch": int(epoch),
            "online_train_mae": float(online_train),
            "valid_mae": float(valid_mae),
            "ema_valid_mae": None if ema_valid is None else float(ema_valid),
            "lr": float(last_lr),
            "seconds": float(sec),
            "steps": int(steps_per_epoch),
        })
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = L.C.cpu_state(model)
            best_sha = _save_checkpoint(out / "best.pt", model, arm=arm, seed=seed,
                                        epoch=epoch, valid_mae=valid_mae, policy="online")
        L.log(f"[{arm} s{seed}] ep={epoch:03d} valid={valid_mae:.6f} "
              f"ema={'--' if ema_valid is None else f'{ema_valid:.6f}'} "
              f"online={online_train:.6f} lr={last_lr:.3e} "
              f"best={best['valid_mae']:.6f}@{best['epoch']} {sec:.2f}s")

    wall = time.perf_counter() - started
    final_sha = _save_checkpoint(out / "final.pt", model, arm=arm, seed=seed,
                                 epoch=n_epochs, valid_mae=curve[-1]["valid_mae"],
                                 policy="online")
    L.C.atomic_torch_save(
        {
            "state_dict": L.C.cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "gen_state": gen.get_state(),
            "epoch": n_epochs,
            "global_step": global_step,
            "best": best,
            "ema_best": ema_best,
            "curve": curve,
            "arm": arm,
            "seed": int(seed),
        },
        out / "final_state.pt",
    )

    # ---- fixed-checkpoint eval-mode measurements ---------------------------
    def _eval_ckpt(state, policy: str, checkpoint_path: Path) -> dict:
        model_ck = M.load_model_from_state("A", norm, state, device, cfg)
        pred_valid, valid = T.evaluate(model_ck, valid_cache, rows_valid, device=device,
                                      batch_size=int(recipe["valid_batch_size"]))
        _, train_mae = T.evaluate(model_ck, train_cache, rows_train, device=device,
                                  batch_size=int(recipe["valid_batch_size"]))
        return {"policy": policy, "pred_valid": pred_valid, "valid": float(valid),
                "train": float(train_mae), "checkpoint": str(checkpoint_path)}

    best_eval = _eval_ckpt(best_state, "online", out / "best.pt")
    if abs(best_eval["valid"] - best["valid_mae"]) > 1e-6:
        raise RuntimeError(f"best checkpoint replay mismatch: {best_eval['valid']} "
                           f"vs {best['valid_mae']}")
    ema_eval = None
    if ema_best_state is not None:
        ema_eval = _eval_ckpt(ema_best_state, "ema", out / "ema_best.pt")
        if abs(ema_eval["valid"] - ema_best["valid_mae"]) > 1e-6:
            raise RuntimeError(f"ema checkpoint replay mismatch: {ema_eval['valid']} "
                               f"vs {ema_best['valid_mae']}")

    masked = Z.group_masks()
    yv = np.asarray(valid_cache["labels"], dtype=np.float64)
    online_groups = Z.group_table(best_eval["pred_valid"], yv, masked)
    ema_groups = None if ema_eval is None else Z.group_table(ema_eval["pred_valid"], yv, masked)
    online_win = Z.window_diagnostics([c["valid_mae"] for c in curve])
    ema_curve = [c["ema_valid_mae"] for c in curve if c["ema_valid_mae"] is not None]
    ema_win = Z.window_diagnostics(ema_curve)

    pred_dir = out / "predictions"
    Z.ensure_dir(pred_dir)
    np.savez(pred_dir / "valid_best.npz", row=rows_valid, pred=best_eval["pred_valid"])
    if ema_eval is not None:
        np.savez(pred_dir / "valid_ema_best.npz", row=rows_valid, pred=ema_eval["pred_valid"])
    L.C.write_curve_csv(out / "curve.csv", curve)

    payload = {
        "round": Z.ROUND_ID,
        "arm": arm,
        "arm_label": Z.ARMS[arm]["label"],
        "seed": int(seed),
        "status": "completed",
        "git_commit": L.C.git_commit(),
        "environment": Z.L.env_summary(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_label_sha256": L.C.sha256_array(y_train),
            "valid_label_sha256": L.C.sha256_array(valid_cache["labels"]),
            **input_audit,
        },
        "model_info": model_info,
        "recipe": {**recipe, "loss": loss_name, "smooth_l1_beta":
                   (Z.SMOOTH_L1_BETA if loss_name == "smooth_l1" else None)},
        "warmup_steps": int(warmup),
        "steps_per_epoch": int(steps_per_epoch),
        "total_updates": int(total_updates),
        "epochs": int(n_epochs),
        "param_audit": audit,
        "gradient_audit_first_batch": first_batch_grad,
        "loss_reference_first_batch": first_loss_ref,
        "ema": {
            "enabled": True,
            "decay": float(Z.EMA_DECAY),
            "start_frac": float(Z.EMA_START_FRAC),
            "start_step": int(ema_start),
            "passive": True,
            "audit": ema_audit,
            "best": ema_best,
        },
        "first_batch_rows": None if first_batch_rows is None else first_batch_rows.tolist(),
        "data_order_seed": int(seed) + 101,
        "norm": {"source": norm["source"], "fingerprint": D.norm_fingerprint(norm)},
        "best": best,
        "online_eval": {k: v for k, v in best_eval.items() if k != "pred_valid"},
        "online_groups": online_groups,
        "online_windows": online_win,
        "ema_eval": None if ema_eval is None else {k: v for k, v in ema_eval.items() if k != "pred_valid"},
        "ema_groups": ema_groups,
        "ema_windows": ema_win,
        "wall_seconds": float(wall),
        "gpu_hours_estimate": float(wall / 3600.0),
        "checkpoint_sha256": {"best": best_sha, "final": final_sha, "ema_best": ema_best_sha},
        "curve": curve,
    }
    Z.write_json(train_json, payload)
    L.log(f"[{arm} s{seed}] done best={best['valid_mae']:.6f}@{best['epoch']} "
          f"ema_best={ema_best['valid_mae']:.6f}@{ema_best['epoch']} wall={wall:.1f}s")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ZPR-v1 LCRC-A recipe pilot trainer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--arm", required=True, choices=sorted(Z.ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "train":
        res = train(args.arm, args.seed, args.device, epochs=args.epochs, tag=args.tag,
                    force=bool(args.force),
                    out_dir=None if args.out is None else Path(args.out))
        L.log(f"best valid {res['best']['valid_mae']:.6f} @ {res['best']['epoch']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
