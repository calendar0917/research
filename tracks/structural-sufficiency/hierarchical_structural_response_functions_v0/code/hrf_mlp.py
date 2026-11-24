"""MLP reference arms on exactly the F_BASE / F_NULL / F_REAL matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import hrf_common as C
import hrf_design as D


def _log(msg: str) -> None:
    print(f"[mlp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden[0]), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        arr = p.detach().cpu().numpy()
        h.update(name.encode()); h.update(str(arr.shape).encode())
        h.update(arr.dtype.str.encode()); h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, rows: np.ndarray, y: np.ndarray,
             device: torch.device, y_mean: float, y_scale: float,
             batch: int = 1024) -> tuple[np.ndarray, float, float]:
    model.eval()
    preds = np.empty(rows.size, np.float64)
    for start in range(0, rows.size, batch):
        idx = rows[start:start + batch]
        xb = X[torch.as_tensor(idx, dtype=torch.long, device=device)]
        z = model(xb).detach().cpu().numpy().astype(np.float64)
        preds[start:start + idx.size] = y_mean + y_scale * z
    return preds, C.rmse(y[rows], preds), C.mae(y[rows], preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["BASE", "NULL", "REAL"])
    ap.add_argument("--stage0", default=str(C.DATA_DIR / "hsrf_stage0.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="debug smoke override; formal runs use the frozen 180")
    args = ap.parse_args()
    cfg = C.load_config()
    mcfg = cfg["mlp"]
    stage = D.Stage0(args.stage0)
    design_name = {"BASE": "F_BASE", "NULL": "F_NULL", "REAL": "F_REAL"}[args.arm]
    design = stage.design(design_name)
    X = np.asarray(design["X"], np.float32)
    y = stage.y
    y_std = stage.y_std
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MLP(X.shape[1], tuple(mcfg["hidden"]), float(mcfg["dropout"])).to(device)
    init_hash = state_hash(model)
    n_params = int(sum(p.numel() for p in model.parameters()))
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_std.astype(np.float32), dtype=torch.float32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
    batch_order_hash = ""
    _log(f"arm={args.arm} design={design_name} D={X.shape[1]} params={n_params} device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(mcfg["lr"]),
                            weight_decay=float(mcfg["weight_decay"]))
    epochs = int(mcfg["epochs"] if args.epochs is None else args.epochs)
    batch = int(mcfg["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=float(mcfg["min_lr"]))
    loss_fn = nn.MSELoss()
    best = {"epoch": -1, "valid_rmse": float("inf"), "state": None, "preds": None}
    curves = []
    train_rows = stage.rows["train"]
    t_train = 0.0
    t_eval = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t_epoch = time.time()
        order = torch.randperm(train_rows.size, generator=gen).numpy()
        if epoch == 1:
            batch_order_hash = hashlib.sha256(np.ascontiguousarray(order).tobytes()).hexdigest()
        total = 0.0
        for start in range(0, order.size, batch):
            idx = train_rows[order[start:start + batch]]
            xb = Xt[torch.as_tensor(idx, dtype=torch.long, device=device)]
            yb = yt[torch.as_tensor(idx, dtype=torch.long, device=device)]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(mcfg["grad_clip"]))
            opt.step()
            total += float(loss.detach().cpu()) * idx.size
        sched.step()
        train_loss = total / train_rows.size
        t_train += time.time() - t_epoch
        t_eval_start = time.time()
        row = {"epoch": epoch, "lr": float(sched.get_last_lr()[0]),
               "train_loss_std": float(train_loss)}
        all_pred = np.empty(stage.n, np.float64)
        for split in ("train", "valid", "test"):
            rows = stage.rows[split]
            pred, r, m = evaluate(model, Xt, rows, y, device,
                                   float(stage.scaler["mean"]), float(stage.scaler["std"]))
            all_pred[rows] = pred
            row[f"{split}_rmse"] = r
            row[f"{split}_mae"] = m
        t_eval += time.time() - t_eval_start
        row["wall_seconds"] = float(time.time() - t0)
        is_best = False
        if row["valid_rmse"] < best["valid_rmse"] - 1e-12:
            best = {"epoch": epoch, "valid_rmse": row["valid_rmse"],
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "preds": all_pred.copy(), "row": row}
            is_best = True
        row["best"] = is_best
        curves.append(row)
        if epoch % 20 == 0 or epoch == 1:
            _log(f"epoch {epoch}/{epochs} train {row['train_rmse']:.4f} "
                 f"valid {row['valid_rmse']:.4f} test {row['test_rmse']:.4f} "
                 f"best@{best['epoch']}")

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_preds = np.empty(stage.n, np.float64)
    for split in ("train", "valid", "test"):
        rows = stage.rows[split]
        pred, _, _ = evaluate(model, Xt, rows, y, device,
                              float(stage.scaler["mean"]), float(stage.scaler["std"]))
        final_preds[rows] = pred

    torch.save({"state_dict": best["state"], "epoch": best["epoch"],
                "valid_rmse": best["valid_rmse"], "arm": args.arm,
                "design": design_name, "n_features": int(X.shape[1])},
               out / "state_best.pt")
    torch.save({"state_dict": final_state, "epoch": epochs,
                "valid_rmse": curves[-1]["valid_rmse"], "arm": args.arm,
                "design": design_name, "n_features": int(X.shape[1])},
               out / "state_final.pt")
    C.write_csv(out / "curves.csv", curves)

    for tag, preds in (("best", best["preds"]), ("final", final_preds)):
        for split in ("train", "valid", "test"):
            rows = stage.rows[split]
            p = preds[rows]
            np.savez_compressed(out / f"predictions_{tag}_{split}.npz",
                                row_id=rows, y_true=y[rows], pred=p,
                                arm=np.asarray([args.arm]), split=np.asarray([split]))
            C.write_csv(out / f"predictions_{tag}_{split}.csv", [
                {"row_id": int(r), "split": split, "y_true": float(yy), "pred": float(pp)}
                for r, yy, pp in zip(rows, y[rows], p)
            ])
    manifest = {
        "round": cfg["round_id"], "arm": args.arm, "design": design_name,
        "commit": C.git_commit(), "created_at": C.now_utc(),
        "git_dirty": bool(os.popen("git status --porcelain").read().strip()),
        "stage0": str(stage.path), "stage0_sha256": C.sha256_file(stage.path),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "n_features": int(X.shape[1]), "n_params": n_params,
        "seed": int(args.seed), "device": str(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs": epochs, "batch_size": batch,
        "best_epoch": int(best["epoch"]),
        "best_valid_rmse": float(best["valid_rmse"]),
        "best_train_rmse": float(best["row"]["train_rmse"]),
        "best_test_rmse": float(best["row"]["test_rmse"]),
        "best_train_mae": float(best["row"]["train_mae"]),
        "best_valid_mae": float(best["row"]["valid_mae"]),
        "best_test_mae": float(best["row"]["test_mae"]),
        "final_valid_rmse": float(curves[-1]["valid_rmse"]),
        "final_test_rmse": float(curves[-1]["test_rmse"]),
        "train_seconds": float(t_train), "eval_seconds": float(t_eval),
        "total_seconds": float(time.time() - t0),
        "init_sha256": init_hash,
        "final_state_sha256": state_hash(model),
        "input_sha256_float32": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        "batch_order_first_epoch_sha256": batch_order_hash,
        "batch_order_generator": "torch.Generator(device='cpu').manual_seed(seed), one stream",
        "loss": "MSE on Train-standardised label",
        "grad_clip": float(mcfg["grad_clip"]),
        "weight_decay": float(mcfg["weight_decay"]),
        "lr_schedule": "cosine", "peak_lr": float(mcfg["lr"]),
        "min_lr": float(mcfg["min_lr"]),
        "dropout": float(mcfg["dropout"]),
        "hidden": list(mcfg["hidden"]),
        "anomalies": [],
    }
    C.write_json(out / "manifest.json", manifest)
    C.write_json(out / "run_meta.json", manifest)
    _log(f"done best@{best['epoch']} valid={best['valid_rmse']:.6f} "
         f"test={best['row']['test_rmse']:.6f} train_s={t_train:.1f} eval_s={t_eval:.1f}")


if __name__ == "__main__":
    main()