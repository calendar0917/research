"""P3 reference calibration under the new best-valid protocol.

The model, initialisation, data order and R3 recipe are the frozen P3/R3 ones.
The only additions are per-epoch validation, checkpoint retention, last-5 soup
extraction and eval-mode train snapshots. Test is never constructed here.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import zpp_common as C
import zpp_data as D

C.ensure_p3_modules()
import structural_relation_objects_v1 as sro  # noqa: E402
import esr_models as M  # noqa: E402

EPOCHS = 360
BATCH_SIZE = 128
LR = 1.0e-3
LR_MIN = 1.0e-5
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
SHUFFLE_OFFSET = 101
SOUP_LAST = 5
STEPS_PER_EPOCH = 10000 // BATCH_SIZE
TOTAL_UPDATES = STEPS_PER_EPOCH * EPOCHS
SNAPSHOT_EVERY = 20
SNAPSHOT_ROWS = 2000

CONT_MAX_EXTRA = 360
CONT_LR = 1.0e-5
CONT_PATIENCE = 60


def run_dir(seed: int) -> Path:
    return C.RUNS / f"p3_s{int(seed)}"


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_batch(objs, rows, device) -> dict:
    return _to_device(sro.batch_objects([objs[int(i)] for i in rows]), device)


@torch.no_grad()
def predict_objects(model, objs, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with C.isolated_rng():
        for start in range(0, len(objs), batch_size):
            rows = list(range(start, min(start + batch_size, len(objs))))
            pred = model(make_batch(objs, rows, device))
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros((0,), dtype=np.float32)


def eval_objects(model, objs, y, device, batch_size: int = 256) -> tuple[np.ndarray, float]:
    pred = predict_objects(model, objs, device, batch_size)
    return pred, C.mae(pred, y)


def _cpu_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state(model, state, device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def _state_mean(states) -> dict:
    return {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}


def _soup(states):
    return _state_mean(list(states))


def save_fixed(model, path: Path, *, epoch: int, valid_mae: float) -> None:
    torch.save({"state_dict": _cpu_state(model), "epoch": int(epoch), "valid_mae": float(valid_mae)}, path)


def write_curve_csv(path: Path, curve) -> None:
    keys = []
    for entry in curve:
        for k in entry:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for entry in curve:
            w.writerow(entry)


def load_inputs():
    verify = D.verify_p3_inputs()
    prep = C.read_json(C.P3_PREP)
    objs_train = D.load_p3_objects()
    objs_valid = D.load_p3_valid_objects()
    y_train = D.load_p3_train_y()
    y_valid = D.split_labels(D.build_data("valid", rrwp=False))
    if len(objs_train) != 10000 or len(objs_valid) != 1000:
        raise RuntimeError("frozen object counts are wrong")
    # The valid split alignment was asserted in the data audit; repeat cheaply.
    for seed, path in C.P3_HISTORICAL_VALID_PRED.items():
        pred = np.asarray(np.load(path)["pred"]).reshape(-1)
        if abs(C.mae(pred, y_valid) - C.P3_FROZEN_VALID_MAE[seed]) > 1e-9:
            raise RuntimeError(f"valid alignment failed for seed {seed}")
    return prep, objs_train, objs_valid, y_train, y_valid, verify


def build_model(seed: int, prep, device):
    model = M.build_arm("B", int(seed), prep, None, 0.0, device)
    params = int(sum(p.numel() for p in model.parameters()))
    if params != 413505:
        raise RuntimeError(f"P3 parameter audit failed: {params}")
    encoder = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
    if int(seed) not in (0, 1):
        raise RuntimeError("only frozen SSL encoders for seeds 0/1 are available")
    M.load_encoder(model, int(seed), encoder, device)
    return model


def _curve_entry(epoch, online, lr, seconds, snapshot, valid_mae):
    return {
        "epoch": int(epoch),
        "online_train_mae": float(online),
        "valid_mae": float(valid_mae),
        "snapshot_train_mae": (None if snapshot is None else float(snapshot)),
        "lr": float(lr),
        "seconds": float(seconds),
    }


def train(seed: int, device, *, force: bool = False) -> dict:
    out = run_dir(seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        C.log(f"[p3 s{seed}] already complete; use --force to rerun")
        return C.read_json(out / "train.json")

    prep, objs_train, objs_valid, y_train, y_valid, verify = load_inputs()
    model = build_model(seed, prep, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_UPDATES, eta_min=LR_MIN)
    gen = torch.Generator().manual_seed(int(seed) + SHUFFLE_OFFSET)

    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    order_hasher = __import__("hashlib").sha256()
    snap_rows = list(range(min(SNAPSHOT_ROWS, len(objs_train))))
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(objs_train), generator=gen)
        order_hasher.update(np.ascontiguousarray(order.numpy(), dtype=np.int64).tobytes())
        total_err, n_mol, steps = 0.0, 0, 0
        lr_last = float(optimizer.param_groups[0]["lr"])
        t_ep = time.perf_counter()
        for start in range(0, len(objs_train), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE].tolist()
            batch = make_batch(objs_train, rows, device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            lr_last = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
            steps += 1
        online = total_err / max(n_mol, 1)
        state = _cpu_state(model)
        states.append((epoch, state))

        valid_pred, valid_mae = eval_objects(model, objs_valid, y_valid, device)
        snapshot = None
        if epoch % SNAPSHOT_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            _, snapshot = eval_objects(model, objs_train[:SNAPSHOT_ROWS], y_train[:SNAPSHOT_ROWS], device)
        curve.append(_curve_entry(epoch, online, lr_last, time.perf_counter() - t_ep, snapshot, valid_mae))

        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = state
            save_fixed(model, out / "best.pt", epoch=epoch, valid_mae=valid_mae)
        C.log(
            f"[p3 s{seed}] ep={epoch:03d} valid={valid_mae:.6f} online={online:.6f} "
            f"snap={'-' if snapshot is None else f'{snapshot:.6f}'} "
            f"lr={lr_last:.3e} best={best['valid_mae']:.6f}@{best['epoch']} "
            f"{curve[-1]['seconds']:.1f}s"
        )

    wall = time.perf_counter() - started
    final_state = curve[-1]
    torch.save(
        {
            "model": _cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "gen_state": gen.get_state(),
            "epoch": EPOCHS,
            "best": best,
            "curve": curve,
        },
        out / "final.pt",
    )
    (out / "last5").mkdir(exist_ok=True)
    for e, st in list(states):
        torch.save({"state_dict": st, "epoch": int(e)}, out / "last5" / f"epoch_{int(e):03d}.pt")
    members = [int(e) for e, _ in states]
    soup = _soup([st for _, st in states])
    torch.save({"state_dict": soup, "members": members}, out / "soup.pt")

    # Fixed-object evaluation: best / final / soup on valid and full train.
    fixed = {}
    best_model = build_model(seed, prep, device)
    _load_state(best_model, best_state, device)
    bp_valid, bv = eval_objects(best_model, objs_valid, y_valid, device)
    bp_train, bt = eval_objects(best_model, objs_train, y_train, device)
    fixed["best"] = {"valid_mae": bv, "train_mae_eval_mode": bt, "epoch": best["epoch"]}

    fp_valid, fv = eval_objects(model, objs_valid, y_valid, device)
    fp_train, ft = eval_objects(model, objs_train, y_train, device)
    fixed["final"] = {"valid_mae": fv, "train_mae_eval_mode": ft, "epoch": EPOCHS}

    _load_state(model, soup, device)
    sp_valid, sv = eval_objects(model, objs_valid, y_valid, device)
    sp_train, st = eval_objects(model, objs_train, y_train, device)
    fixed["soup"] = {"valid_mae": sv, "train_mae_eval_mode": st, "members": members}

    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for name, pred in (("best", bp_valid), ("final", fp_valid), ("soup", sp_valid)):
        np.savez(pred_dir / f"valid_{name}.npz", row=np.arange(len(pred)), pred=pred)

    info = {
        "protocol": C.PROTOCOL_ID,
        "model": "P3 (BJOC-Null + frozen train-only masked SSL + R3)",
        "seed": int(seed),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": 10000,
            "n_valid": 1000,
            "official_test_loaded": False,
            "frozen_inputs": verify,
            "valid_labels_alignment": "exact to the frozen historical arrays",
        },
        "recipe": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "updates": TOTAL_UPDATES,
            "optimizer": "Adam",
            "lr": LR,
            "lr_min": LR_MIN,
            "schedule": "per-step cosine",
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "shuffle_offset": SHUFFLE_OFFSET,
            "loss": "raw-y L1",
        },
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "best_valid": best,
        "fixed_objects": fixed,
        "curve": curve,
        "batch_order_sha256": order_hasher.hexdigest(),
        "wall_seconds": float(wall),
        "gpu_hours": None if device == "cpu" else float(wall) / 3600.0,
        "device": str(device),
        "notes": [
            "online_train_mae is a within-epoch pre-update running mean",
            "snapshot_train_mae and train_mae_eval_mode are eval-mode fixed-model measurements",
            "soup is a secondary diagnostic, not the primary selector",
        ],
    }
    C.write_json(out / "train.json", info)
    write_curve_csv(out / "curve.csv", curve)
    C.log(f"[p3 s{seed}] done best={best['valid_mae']:.6f}@{best['epoch']} "
          f"final={fv:.6f} soup={sv:.6f} wall={wall:.1f}s")
    return info


def continue_train(seed: int, device, *, force: bool = False) -> dict:
    """360 + low-LR continuation with restored Adam state; never a new cosine."""
    out = run_dir(seed)
    base = C.read_json(out / "train.json")
    if (out / "continuation.json").exists() and not force:
        return C.read_json(out / "continuation.json")
    if not (out / "final.pt").exists():
        raise RuntimeError("missing final.pt for continuation")

    prep, objs_train, objs_valid, y_train, y_valid, _ = load_inputs()
    model = build_model(seed, prep, device)
    checkpoint = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    _load_state(model, checkpoint["model"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONT_LR, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = CONT_LR
    gen = torch.Generator()
    gen.set_state(checkpoint["gen_state"])
    best = dict(checkpoint["best"])
    last_improve = 0
    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict] = []
    started = time.perf_counter()

    for offset in range(1, CONT_MAX_EXTRA + 1):
        epoch = EPOCHS + offset
        model.train()
        order = torch.randperm(len(objs_train), generator=gen)
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(objs_train), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE].tolist()
            batch = make_batch(objs_train, rows, device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        states.append((epoch, _cpu_state(model)))
        _, valid_mae = eval_objects(model, objs_valid, y_valid, device)
        snapshot = None
        if epoch % SNAPSHOT_EVERY == 0:
            _, snapshot = eval_objects(model, objs_train[:SNAPSHOT_ROWS], y_train[:SNAPSHOT_ROWS], device)
        curve.append(_curve_entry(epoch, online, CONT_LR, time.perf_counter() - t_ep, snapshot, valid_mae))
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            last_improve = 0
            save_fixed(model, out / "cont_best.pt", epoch=epoch, valid_mae=valid_mae)
        else:
            last_improve += 1
        C.log(f"[p3 s{seed} cont] ep={epoch:03d} valid={valid_mae:.6f} "
              f"online={online:.6f} best={best['valid_mae']:.6f}@{best['epoch']} "
              f"patience={last_improve}/{CONT_PATIENCE}")
        if last_improve >= CONT_PATIENCE:
            break

    wall = time.perf_counter() - started
    torch.save(
        {
            "model": _cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "gen_state": gen.get_state(),
            "epoch": EPOCHS + len(curve),
            "best": best,
            "curve": curve,
        },
        out / "cont_final.pt",
    )
    (out / "cont_last5").mkdir(exist_ok=True)
    for e, st in list(states):
        torch.save({"state_dict": st, "epoch": int(e)}, out / "cont_last5" / f"epoch_{int(e):03d}.pt")
    info = {
        "protocol": C.PROTOCOL_ID,
        "kind": "360_plus_low_lr_continuation",
        "seed": int(seed),
        "base_epoch": EPOCHS,
        "extra_epochs": len(curve),
        "fixed_lr": CONT_LR,
        "scheduler": "disabled_after_finished_cosine",
        "patience": CONT_PATIENCE,
        "best": best,
        "curve": curve,
        "wall_seconds": float(wall),
        "git_commit": C.git_commit(),
        "official_test_loaded": False,
    }
    C.write_json(out / "continuation.json", info)
    write_curve_csv(out / "cont_curve.csv", curve)
    return info


def guard() -> dict:
    """Apply the pre-declared continuation rule to the two traces."""
    out = {}
    for seed in (0, 1):
        info = C.read_json(run_dir(seed) / "train.json")
        curve = {int(e["epoch"]): e for e in info["curve"]}
        v320 = float(curve[320]["valid_mae"])
        v360 = float(curve[360]["valid_mae"])
        best_epoch = int(info["best_valid"]["epoch"])
        gain = v320 - v360
        trigger = bool(best_epoch >= 320 and gain >= 0.001)
        out[str(seed)] = {
            "best_epoch": best_epoch,
            "best_valid_mae": float(info["best_valid"]["valid_mae"]),
            "valid_mae_epoch_320": v320,
            "valid_mae_epoch_360": v360,
            "terminal_gain_320_to_360": gain,
            "require_best_epoch_ge_320": bool(best_epoch >= 320),
            "require_gain_ge_0.001": bool(gain >= 0.001),
            "trigger": trigger,
        }
    out["any_trigger"] = any(v["trigger"] for k, v in out.items() if k in ("0", "1"))
    out["rule"] = "best_epoch>=320 AND valid(360) <= valid(320)-0.001"
    C.write_json(C.ANALYSIS / "p3_continue_decision.json", out)
    C.log(f"continuation decision: {out}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "continue", "guard", "eval"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "train":
        train(args.seed, device, force=args.force)
    elif args.command == "continue":
        continue_train(args.seed, device, force=args.force)
    elif args.command == "guard":
        guard()
    else:
        raise SystemExit("eval is handled by zpp_score.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
