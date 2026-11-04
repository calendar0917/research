"""P0: fixed MLP readout on the FULL-S2 count matrix (diagnostic only).

Architecture and recipe are frozen before the run:

* input = ``log1p(FULL-S2 counts)``, 2048 dims
* 2048 -> 64 (SiLU) -> 10 logits; no dropout, no normalisation
* AdamW, peak LR 1e-3, weight decay 1e-4, betas (0.9, 0.999), eps 1e-8
* batch 256, 200 epochs, per-step cosine to 1e-5, gradient clip 5
* unweighted BCEWithLogitsLoss, seed 0
* checkpoint = best Validation macro AP; no early stopping, no search

This readout is a diagnostic and is not used to modify the P1 models.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import pc_common as C

GUARD = C.install_test_access_guard()


class MLPReadout(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 64, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def macro_ap(y: np.ndarray, probs: np.ndarray) -> tuple[float, list[float]]:
    from sklearn.metrics import average_precision_score

    aps = [float(average_precision_score(y[:, j], probs[:, j])) for j in range(y.shape[1])]
    return float(np.mean(aps)), aps


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def write_curve(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(C.ANALYSIS / "p0_mlp"))
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(8)

    with np.load(C.ANALYSIS / "features_train.npz") as z:
        x_train = np.log1p(np.asarray(z["counts_full"], dtype=np.float32))
    with np.load(C.ANALYSIS / "features_valid.npz") as z:
        x_valid = np.log1p(np.asarray(z["counts_full"], dtype=np.float32))
    y_train = np.load(C.PREV_ANALYSIS / "peptides_func_y_train.npy").astype(np.float32)
    y_valid = np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy").astype(np.float32)
    if x_train.shape != (10873, C.N_BUCKETS) or x_valid.shape != (2331, C.N_BUCKETS):
        raise SystemExit("feature shape gate failed")

    model = MLPReadout().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )
    steps_per_epoch = int(math.ceil(x_train.shape[0] / args.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps_per_epoch * args.epochs, 1), eta_min=1e-5
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    curve: list[dict] = []
    best_metric = -math.inf
    best_epoch = None
    best_state = None
    best_probs_valid = None
    best_probs_train = None
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(x_train.shape[0], generator=generator).numpy()
        losses: list[float] = []
        for start in range(0, perm.size, args.batch_size):
            rows = perm[start : start + args.batch_size]
            xb = torch.as_tensor(x_train[rows], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise SystemExit(f"non-finite loss/gradient at epoch {epoch}")
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))
        probs_valid = predict(model, x_valid, device)
        metric, per_label = macro_ap(y_valid, probs_valid)
        improved = metric > best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs_valid = probs_valid
            best_probs_train = predict(model, x_train, device)
        curve.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(np.mean(losses)),
                "valid_macro_ap": metric,
                "valid_macro_ap_pct": 100.0 * metric,
                "is_best": bool(improved),
                "per_label_valid_ap": per_label,
                "elapsed_seconds": float(time.perf_counter() - t_start),
            }
        )
        if epoch % 10 == 0 or improved:
            C.log(f"epoch {epoch}/{args.epochs} loss {curve[-1]['train_loss']:.5f} valid {100*metric:.2f}%{ ' *best*' if improved else ''}")
        write_curve(out / "curve.csv", curve)

    if best_state is None or best_probs_valid is None or best_probs_train is None:
        raise SystemExit("no checkpoint selected")
    torch.save(
        {
            "state_dict": best_state,
            "input": "log1p(FULL-S2 counts)",
            "architecture": [2048, 64, 10],
            "activation": "SiLU",
            "best_epoch": int(best_epoch),
            "best_valid_macro_ap": float(best_metric),
            "recipe": {
                "optimizer": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "loss": "BCEWithLogitsLoss (unweighted)",
                "scheduler": "per-step cosine, eta_min=1e-5",
                "gradient_clip": 5.0,
                "seed": int(args.seed),
            },
            "test_evaluated": False,
        },
        out / "checkpoint.pt",
    )
    train_ap, train_per_label = macro_ap(y_train, best_probs_train)
    valid_ap, valid_per_label = macro_ap(y_valid, best_probs_valid)
    # Reload and re-evaluate the selected state.
    reloaded = MLPReadout().to(device)
    ckpt = torch.load(out / "checkpoint.pt", map_location=device, weights_only=False)
    reloaded.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()}, strict=True)
    reload_probs = predict(reloaded, x_valid, device)
    reload_ap, _ = macro_ap(y_valid, reload_probs)
    reload_delta = abs(reload_ap - best_metric)
    if reload_delta > 1e-6:
        raise SystemExit(f"checkpoint reload mismatch: {reload_delta}")
    np.savez_compressed(
        out / "predictions.npz",
        probs_train=best_probs_train,
        probs_valid=best_probs_valid,
        y_train=y_train,
        y_valid=y_valid,
        labels=np.asarray(C.LABELS),
    )
    result = {
        "status": "complete_no_test",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "entry_point": "code/pc_p0_mlp.py",
        "input": "log1p(FULL-S2 counts)",
        "architecture": [2048, 64, 10],
        "activation": "SiLU",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "recipe": ckpt["recipe"],
        "epochs_run": len(curve),
        "best_epoch": int(best_epoch),
        "best_valid_macro_ap": float(best_metric),
        "best_valid_macro_ap_pct": 100.0 * best_metric,
        "reload_valid_macro_ap": float(reload_ap),
        "reload_delta": float(reload_delta),
        "train_macro_ap": float(train_ap),
        "train_macro_ap_pct": 100.0 * train_ap,
        "valid_macro_ap": float(valid_ap),
        "valid_macro_ap_pct": 100.0 * valid_ap,
        "per_label_train_ap": {name: train_per_label[j] for j, name in enumerate(C.LABELS)},
        "per_label_valid_ap": {name: valid_per_label[j] for j, name in enumerate(C.LABELS)},
        "curve_sha256": C.sha256_file(out / "curve.csv"),
        "checkpoint_sha256": C.sha256_file(out / "checkpoint.pt"),
        "predictions_sha256": C.sha256_file(out / "predictions.npz"),
        "seconds": float(time.perf_counter() - t_start),
        "test_access": {"attempts": list(GUARD["attempts"]), "strict": GUARD["strict"]},
        "test_evaluated": False,
        "environment": C.environment(),
    }
    C.write_json(out / "run.json", result)
    C.write_json(C.ANALYSIS / "p0_mlp.json", result)
    C.log(
        f"MLP readout: best epoch {best_epoch}, Train macro AP {100*train_ap:.2f}%, "
        f"Validation macro AP {100*valid_ap:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
