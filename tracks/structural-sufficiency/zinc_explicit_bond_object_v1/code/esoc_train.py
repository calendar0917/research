"""ESOC-v1 training / profiling / overfit harness.

Three modes, all on the same code path:

* ``profile``  a few complete epochs on the full train split -> timing.json
* ``overfit``  a fixed subset of train molecules -> small-sample sanity gate
* ``full``     the declared schedule from ``configs/esoc_v1.json`` (600 epochs)

Only official PyG ZINC train (10,000) and valid (1,000) are used.  The official
test split is never constructed.  The checkpoint is selected on valid with a
strict ``<`` only; ``final`` and every epoch of the curve are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_data as D  # noqa: E402
import esoc_models as M  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
RUNS = EXP / "runs"
CONFIG = EXP / "configs" / "esoc_v1.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config() -> dict:
    return read_json(CONFIG)


# ---------------------------------------------------------------------------
# declared recipe (one trajectory, never tuned inside a seed)
# ---------------------------------------------------------------------------
def schedule_fn(total_steps: int, warmup_steps: int, peak: float, min_lr: float):
    def lr_at(step: int) -> float:
        step = int(step)
        if step < warmup_steps:
            return peak * float(step + 1) / float(max(1, warmup_steps))
        if step >= total_steps:
            return float(min_lr)
        frac = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        import math
        return float(min_lr) + 0.5 * (peak - float(min_lr)) * (1.0 + math.cos(math.pi * frac))
    return lr_at


def batch_order(seed: int, epoch: int, n: int, batch_size: int = 32) -> np.ndarray:
    rng = np.random.default_rng([int(seed), int(epoch), 20261024])
    perm = rng.permutation(n)
    return perm[: (n // batch_size) * batch_size]


def make_optimizer(model: torch.nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                             betas=(0.9, 0.95), eps=1e-8)


def _mae_eval(model, data: "D.SplitTensors", batch_size: int = 250,
              rows: np.ndarray | None = None) -> float:
    """Eval-mode MAE over the whole split or over an explicit row subset."""
    model.eval()
    rows = np.arange(data.n) if rows is None else np.asarray(rows, dtype=np.int64)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            b = data.batch(chunk)
            p = model(b)
            tot += float((p - b["y"]).abs().sum())
            cnt += len(chunk)
    model.train()
    return tot / max(cnt, 1)


def _per_molecule(model, data: "D.SplitTensors", batch_size: int = 250):
    model.eval()
    preds = np.zeros(data.n, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, data.n, batch_size):
            rows = np.arange(start, min(start + batch_size, data.n))
            b = data.batch(rows)
            preds[rows] = model(b).double().cpu().numpy()
    model.train()
    return preds


def _state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(arm: str, seed: int, epochs: int, device: str, *, mode: str,
          max_hours: float = 20.0, subset: int | None = None,
          config: dict | None = None, out_dir: Path | None = None) -> dict:
    cfg = (config or load_config())["recipe"]
    scales = read_json(ANALYSIS / "scales.json")
    n_layers = int(cfg["n_layers"])
    batch_size = int(cfg["batch_size"])
    peak, min_lr, wd = float(cfg["lr"]), float(cfg["lr_min"]), float(cfg["weight_decay"])
    warmup_epochs = int(cfg["warmup_epochs"])
    clip = float(cfg["grad_clip"])

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = M.build_model(seed, arm, scales, n_layers=n_layers).to(device)
    audit = M.param_audit(model)
    if audit["param_total"] > 500_000:
        raise RuntimeError(f"parameter budget exceeded: {audit['param_total']}")
    # Performance only: identical maths, fused kernels.  The state dict, the
    # optimiser and every checkpoint use the uncompiled module.
    net = model
    if cfg.get("compile", False):
        net = torch.compile(model, mode=str(cfg.get("compile_mode", "default")),
                            dynamic=False)

    # hardperm trains on permuted *objects* with the REAL input path; the valid
    # split is permuted with the same deterministic rule (no labels involved).
    variant = "hardperm" if arm == "hardperm" else None
    train_data = D.SplitTensors(D.load_cache("train", variant), device)
    valid_data = D.SplitTensors(D.load_cache("valid", variant), device)
    if subset:
        rng = np.random.default_rng([int(seed), 777])
        subset_rows = np.sort(rng.permutation(train_data.n)[:subset])

    n_upd = (len(subset_rows) if subset else train_data.n) // batch_size
    steps_per_epoch = n_upd
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    lr_at = schedule_fn(total_steps, warmup_steps, peak, min_lr)
    opt = make_optimizer(model, peak, wd)
    opt_params = sum(p.numel() for g in opt.param_groups for p in g["params"])

    out_dir = out_dir or (RUNS / f"{arm}_s{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    curve = []
    best = {"valid_mae": float("inf"), "epoch": -1, "step": -1}
    t0 = time.perf_counter()
    epoch_times = []
    step = 0
    aborted = None

    for epoch in range(1, epochs + 1):
        te = time.perf_counter()
        rows_all = batch_order(seed, epoch, len(subset_rows) if subset else train_data.n,
                               batch_size)
        rows_all = subset_rows[rows_all] if subset else rows_all
        model.train()
        tot, cnt = 0.0, 0
        for start in range(0, len(rows_all), batch_size):
            rows = rows_all[start:start + batch_size]
            b = train_data.batch(rows)
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            p = net(b)
            loss = (p - b["y"]).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += float(loss.detach()) * len(rows)
            cnt += len(rows)
            step += 1
        train_mae = tot / max(cnt, 1)
        valid_mae = _mae_eval(net, valid_data)
        # overfit mode selects on the training subset (the gate is "can the model
        # fit at all"), full mode selects on valid with a strict <.
        select = valid_mae if mode != "overfit" else train_mae
        if select == select and select < best["valid_mae"]:
            best = {"valid_mae": float(select), "epoch": epoch, "step": step,
                    "valid_mae_at_best": float(valid_mae), "train_mae_at_best": float(train_mae)}
            C.atomic_torch_save({"state": C.cpu_state(model), "epoch": epoch,
                                 "valid_mae": float(select), "arm": arm,
                                 "seed": int(seed)},
                                out_dir / "best.pt")
        curve.append({"epoch": epoch, "step": step, "lr": lr if step else lr_at(0),
                      "train_mae_online": train_mae, "valid_mae": valid_mae})
        epoch_times.append(time.perf_counter() - te)
        # Live curve: rewritten every epoch so the run can be watched and analysed
        # while it is still in flight (``rr logs`` only shows what we print).
        C.write_curve_csv(out_dir / "curve.csv", curve)
        # One log line per epoch: silence between checkpoints made a healthy run
        # indistinguishable from a hung one.
        improved = "*" if best["epoch"] == epoch else " "
        C.log(f"[esoc {arm} s{seed}] ep {epoch}/{epochs} "
              f"train {train_mae:.5f} valid {valid_mae:.5f} "
              f"best {best['valid_mae']:.5f}@{best['epoch']}{improved} "
              f"lr {lr:.2e} {(time.perf_counter() - t0) / 3600:.3f}h")
        if epoch % 100 == 0 or epoch == epochs:
            C.atomic_torch_save({"state": C.cpu_state(model), "epoch": epoch,
                                 "optimizer": opt.state_dict(), "step": step,
                                 "arm": arm, "seed": int(seed)},
                                out_dir / "last.pt")
            C.write_json(out_dir / "live.json", {
                "arm": arm, "seed": int(seed), "epoch": epoch, "epochs": int(epochs),
                "step": step, "train_mae_online": float(train_mae),
                "valid_mae": float(valid_mae),
                "best_valid": float(best["valid_mae"]), "best_epoch": int(best["epoch"]),
                "lr": float(lr), "elapsed_hours": float((time.perf_counter() - t0) / 3600),
                "epochs_per_hour": float(epoch / max((time.perf_counter() - t0) / 3600, 1e-9)),
            })
        if (time.perf_counter() - t0) / 3600.0 > max_hours:
            aborted = f"max_hours {max_hours} exceeded at epoch {epoch}"
            C.log(f"[esoc {arm} s{seed}] ABORT: {aborted}")
            break

    wall = time.perf_counter() - t0
    C.atomic_torch_save({"state": C.cpu_state(model), "epoch": epoch,
                         "arm": arm, "seed": int(seed)}, out_dir / "final.pt")

    # eval-mode train / valid with the frozen checkpoints
    info = {}
    for tag in ("best", "final"):
        path = out_dir / f"{tag}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        info[tag] = {
            "epoch": int(ck["epoch"]),
            "valid_mae": float(_mae_eval(model, valid_data)),
            "train_mae_eval_mode": float(_mae_eval(model, train_data, batch_size=500)),
            "sha256": C.sha256_file(path),
        }
        if subset:
            info[tag]["train_subset_eval_mode"] = float(
                _mae_eval(model, train_data, batch_size=500, rows=subset_rows))
        preds = _per_molecule(model, valid_data)
        np.savez(out_dir / "predictions" / f"valid_{tag}.npz",
                 pred=preds, row=np.arange(valid_data.n, dtype=np.int64),
                 y=valid_data.labels.cpu().numpy().astype(np.float64))
        info[tag]["valid_mae_recomputed"] = float(
            np.abs(preds - valid_data.labels.cpu().numpy()).mean())

    curve_note = {
        "epochs_declared": int(epochs),
        "epochs_completed": int(len(curve)),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps_declared": int(total_steps),
        "aborted": aborted,
    }
    result = {
        "arm": arm,
        "seed": int(seed),
        "mode": mode,
        "device": device,
        "param_audit": audit,
        "optimizer_params": int(opt_params),
        "recipe": cfg,
        "schedule": {"peak": peak, "min_lr": min_lr, "warmup_steps": int(warmup_steps),
                     "total_steps": int(total_steps)},
        "best": {"valid_mae": best["valid_mae"], "epoch": best["epoch"], "step": best["step"]},
        "checkpoints": info,
        "curve_note": curve_note,
        "wall_seconds": float(wall),
        "epoch_seconds_median": float(np.median(epoch_times)) if epoch_times else None,
        "epoch_seconds_first3": [float(v) for v in epoch_times[:3]],
        "train_subset": int(subset) if subset else None,
        "state_hash_final": _state_hash(model),
        "environment": C.environment(),
        "scales": scales,
    }
    C.write_json(out_dir / "train.json", result)
    C.write_curve_csv(out_dir / "curve.csv", curve)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ESOC-v1 runner")
    p.add_argument("--mode", default="full", choices=["full", "profile", "overfit"])
    p.add_argument("--arm", default="real", choices=["real", "null", "hardperm"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-hours", type=float, default=20.0)
    p.add_argument("--compile", dest="compile", action="store_true", default=None)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--compile-mode", default=None)
    p.add_argument("--molecules", type=int, default=48)
    args = p.parse_args(argv)

    cfg = load_config()
    if args.compile is not None:
        cfg["recipe"]["compile"] = bool(args.compile)
    if args.compile_mode is not None:
        cfg["recipe"]["compile_mode"] = str(args.compile_mode)
    if args.mode == "profile":
        epochs = args.epochs or 3
        res = train(args.arm, args.seed, epochs, args.device, mode="profile",
                    max_hours=1.0, config=cfg,
                    out_dir=RUNS / f"profile_{args.arm}_s{args.seed}")
        epoch_s = res["epoch_seconds_median"]
        full = int(cfg["recipe"]["epochs"])
        (ANALYSIS / "raw").mkdir(parents=True, exist_ok=True)
        C.write_json(ANALYSIS / "raw" / f"timing_{'compiled' if cfg['recipe'].get('compile') else 'eager'}.json", {
            "compile": bool(cfg["recipe"].get("compile")),
            "compile_mode": cfg["recipe"].get("compile_mode"),
            "arm": args.arm,
            "seed": args.seed,
            "epochs_measured": epochs,
            "epoch_seconds": res["epoch_seconds_first3"],
            "epoch_seconds_median": epoch_s,
            "steps_per_epoch": res["curve_note"]["steps_per_epoch"],
            "projected_hours_full_run": float(epoch_s) * full / 3600.0,
            "declared_epochs": full,
            "wall_seconds": res["wall_seconds"],
            "param_total": res["param_audit"]["param_total"],
            "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20)
            if torch.cuda.is_available() else None,
        })
        print(json.dumps({"epoch_seconds_median": epoch_s,
                          "projected_hours": float(epoch_s) * full / 3600.0}, indent=2))
        return 0
    if args.mode == "overfit":
        res = train(args.arm, args.seed, args.epochs or 120, args.device, mode="overfit",
                    max_hours=1.0, subset=args.molecules, config=cfg,
                    out_dir=RUNS / f"overfit_{args.arm}_s{args.seed}")
        first = res["curve_note"]
        curve = np.genfromtxt(RUNS / f"overfit_{args.arm}_s{args.seed}" / "curve.csv",
                              delimiter=",", names=True)
        print(json.dumps({
            "arm": args.arm,
            "molecules": args.molecules,
            "epochs": int(first["epochs_completed"]),
            "train_mae_first_epoch": float(curve["train_mae_online"][0]),
            "train_mae_last_epoch": float(curve["train_mae_online"][-1]),
            "train_mae_eval_mode_best": res["checkpoints"]["best"]["train_mae_eval_mode"],
        }, indent=2))
        return 0
    res = train(args.arm, args.seed, args.epochs or int(cfg["recipe"]["epochs"]),
                args.device, mode="full", max_hours=args.max_hours, config=cfg)
    print(json.dumps({"arm": res["arm"], "seed": res["seed"],
                      "best_valid": res["best"], "checkpoints": res["checkpoints"],
                      "wall_seconds": res["wall_seconds"],
                      "param_total": res["param_audit"]["param_total"],
                      "epochs_completed": res["curve_note"]["epochs_completed"],
                      "aborted": res["curve_note"]["aborted"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
