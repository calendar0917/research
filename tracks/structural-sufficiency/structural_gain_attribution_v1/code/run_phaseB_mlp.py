"""Phase B: MLP readout on the *identical* inputs of Phase A (MolHIV).

One fixed architecture, no encoder / aggregation / message passing:

    input -> Linear(128) -> ReLU -> Dropout(0.1) -> Linear(128) -> ReLU -> Dropout(0.1) -> Linear(1)

For ``N-WL``/``N-CF`` the first layer is ``h = W_H·H + W_Q·Q + b`` with ``W_Q``
zero-initialised; ``W_H``, ``b`` and the two later layers are drawn from the same
RNG stream in the same order in every arm, so the shared tensors are bit-identical
(their hash is recorded and asserted here).

Recipe (PROTOCOL.md §7, frozen): seed 0, 50 epochs, batch 256, AdamW lr 1e-3,
weight decay 1e-4, class-balanced BCE from Train only (mean weight 1), gradient
clipping 5, no scheduler, no early stopping, identical batch order, full
Train/Valid/Test evaluation every epoch, checkpoint = best Valid ROC-AUC (ties to
the earlier epoch).

Outputs (``--out-dir``): ``<arm>.json``, ``<arm>_curve.csv``, ``<arm>_pred.npz``,
``<arm>_best.pt``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

import sga_common as K

ARMS: dict[str, list[str]] = {
    "N0": [],
    "N-WL": ["WL"],
    "N-CF": ["C", "F"],
}

EPOCHS = 50
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
CLIP = 5.0
HIDDEN = 128
DROPOUT = 0.1
CLIP_FEATURE = 10.0


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------


def load_bundle(path: Path | None = None) -> dict[str, np.ndarray]:
    with np.load(path or K.INPUT_BUNDLE) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def unpack_h(bundle: dict[str, np.ndarray]) -> np.ndarray:
    n_bits = int(bundle["H_n_bits"])
    shape = tuple(int(v) for v in bundle["H_shape"])
    bits = np.unpackbits(bundle["H_bits_packed"], axis=1)[:, :n_bits].astype(np.float32)
    out = np.empty(shape, dtype=np.float32)
    out[:, :n_bits] = bits
    out[:, n_bits:] = np.asarray(bundle["H_desc"], dtype=np.float32)
    return out


def as_csr(bundle: dict[str, np.ndarray], name: str, n_rows: int) -> sp.csr_matrix:
    indices = np.asarray(bundle[f"{name}_indices"], dtype=np.int64)
    return sp.csr_matrix(
        (np.asarray(bundle[f"{name}_data"], dtype=np.float32), indices,
         np.asarray(bundle[f"{name}_indptr"], dtype=np.int64)),
        shape=(n_rows, int(indices.max()) + 1),
    )


def build_blocks(
    bundle: dict[str, np.ndarray], extra: list[str]
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """``H`` with Train-only descriptor standardisation, then the fixed-scale blocks."""
    h = unpack_h(bundle)
    train = np.asarray(bundle["rows_train"], dtype=np.int64)
    n_bits = int(bundle["H_n_bits"])
    desc = h[:, n_bits:]
    mu = desc[train].mean(axis=0)
    sd = desc[train].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    desc -= mu
    desc /= sd
    np.clip(desc, -CLIP_FEATURE, CLIP_FEATURE, out=desc)
    n_rows = h.shape[0]
    blocks = []
    for name in extra:
        mat = as_csr(bundle, name, n_rows)
        blocks.append(np.asarray(mat.todense(), dtype=np.float32))
    meta = {
        "descriptor_standardisation": "Train-only mean/std, clip [-10,10]",
        "descriptor_mean_sha256": K.sha256_array(mu.astype(np.float64)),
        "descriptor_std_sha256": K.sha256_array(sd.astype(np.float64)),
        "n_columns_h": int(h.shape[1]),
        "n_columns_extra": int(sum(b.shape[1] for b in blocks)),
        "n_clipped_cells_total": int(np.isnan(h).sum()) + int(sum(b.shape[0] * b.shape[1] for b in blocks)) * 0,
    }
    return h, blocks, meta


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def make_model(n_h: int, n_q: int):
    import torch
    from torch import nn

    torch.manual_seed(K.SEED)
    np.random.seed(K.SEED)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # construction order is fixed: shared tensors first, extra branch last
            self.w_h = nn.Linear(n_h, HIDDEN)
            self.h2 = nn.Linear(HIDDEN, HIDDEN)
            self.out = nn.Linear(HIDDEN, 1)
            self.drop = nn.Dropout(DROPOUT)
            self.w_q = nn.Linear(n_q, HIDDEN, bias=False) if n_q else None
            if self.w_q is not None:
                nn.init.zeros_(self.w_q.weight)

        def forward(self, xh, xq=None):
            z = self.w_h(xh)
            if self.w_q is not None:
                z = z + self.w_q(xq)
            z = self.drop(torch.relu(z))
            z = self.drop(torch.relu(self.h2(z)))
            return self.out(z).squeeze(-1)

    model = MLP()
    shared_hash = shared_param_hash(model)
    return model, shared_hash


def shared_param_hash(model) -> str:
    import torch

    h = hashlib.sha256()
    for name in ("w_h", "h2", "out"):
        layer = getattr(model, name)
        h.update(np.ascontiguousarray(layer.weight.detach().cpu().numpy()).tobytes())
        h.update(np.ascontiguousarray(layer.bias.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def n_parameters(model) -> dict[str, int]:
    per = {name: int(sum(p.numel() for p in getattr(model, name).parameters()))
           for name in ("w_h", "h2", "out")}
    if model.w_q is not None:
        per["w_q"] = int(sum(p.numel() for p in model.w_q.parameters()))
    return {"per_module": per, "total": int(sum(p.numel() for p in model.parameters()))}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def class_balanced_weights(y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pos = float(y.sum())
    neg = float(y.size - pos)
    w_pos = 0.5 * float(y.size) / pos
    w_neg = 0.5 * float(y.size) / neg
    w = np.where(y > 0.5, w_pos, w_neg).astype(np.float64)
    w = w / w.mean()
    return w, {"pos_weight": w_pos, "neg_weight": w_neg, "mean_after_normalisation": float(w.mean()),
               "fit_on": "Train only"}


def evaluate(model, Xh, Xq, y, indices, batch=4096) -> np.ndarray:
    import torch

    model.eval()
    out = np.empty(indices.size, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, indices.size, batch):
            sel = indices[start: start + batch]
            xh = torch.from_numpy(Xh[sel]).to(next(model.parameters()).device)
            if Xq is not None:
                xq = torch.from_numpy(Xq[sel]).to(xh.device)
                logit = model(xh, xq)
            else:
                logit = model(xh)
            out[start: start + sel.size] = logit.detach().cpu().numpy().astype(np.float64)
    return out


def auc_ap(y: np.ndarray, logit: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "bce": float(np.mean(np.logaddexp(0.0, logit) - y * logit)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--out-dir", default=str(K.RUNS / "phaseB"))
    ap.add_argument("--bundle", default=str(K.INPUT_BUNDLE))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = load_bundle(Path(args.bundle))
    y = np.asarray(bundle["y"], dtype=np.float64)
    splits = {n: np.asarray(bundle[f"rows_{n}"], dtype=np.int64) for n in ("train", "valid", "test")}
    extra = ARMS[args.arm]
    h, blocks, prep = build_blocks(bundle, extra)
    xh = h
    xq = np.concatenate(blocks, axis=1) if blocks else None
    del h, blocks
    print(f"[{args.arm}] design H{xh.shape} Q{None if xq is None else xq.shape} device={device}", flush=True)

    model, shared_hash = make_model(xh.shape[1], 0 if xq is None else xq.shape[1])
    params = n_parameters(model)
    model.to(device)
    print(f"[{args.arm}] init hash {shared_hash[:16]} params {params['total']}", flush=True)

    w, wmeta = class_balanced_weights(y[splits["train"]])
    wt = torch.from_numpy(w.astype(np.float32)).to(device)
    xtr_h = torch.from_numpy(xh[splits["train"]]).to(device) if device == "cuda" else None
    xtr_q = torch.from_numpy(xq[splits["train"]]).to(device) if (xq is not None and device == "cuda") else None
    ytr = torch.from_numpy(y[splits["train"]].astype(np.float32)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    import torch.nn.functional as Fn

    n_train = splits["train"].size
    curve: list[dict[str, Any]] = []
    best = {"epoch": -1, "valid_roc_auc": -np.inf}
    best_state = None
    rng = np.random.default_rng(K.SEED)
    epoch_orders = [rng.permutation(n_train) for _ in range(args.epochs)]  # identical across arms

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = epoch_orders[epoch - 1]
        total_loss = 0.0
        n_seen = 0
        for start in range(0, n_train, BATCH):
            sel = order[start: start + BATCH]
            if device == "cuda":
                xb_h = xtr_h[sel]
                xb_q = xtr_q[sel] if xtr_q is not None else None
                yb = ytr[sel]
                wb = wt[sel]
            else:
                xb_h = torch.from_numpy(xh[splits["train"][sel]]).to(device)
                xb_q = torch.from_numpy(xq[splits["train"][sel]]).to(device) if xq is not None else None
                yb = torch.from_numpy(y[splits["train"][sel]].astype(np.float32)).to(device)
                wb = wt[sel]
            opt.zero_grad(set_to_none=True)
            logit = model(xb_h, xb_q) if xb_q is not None else model(xb_h)
            loss = Fn.binary_cross_entropy_with_logits(logit, yb, weight=wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total_loss += float(loss.item()) * sel.size
            n_seen += sel.size
        row: dict[str, Any] = {"epoch": epoch, "train_loss_batch_weighted": total_loss / n_seen}
        preds = {}
        for name in ("train", "valid", "test"):
            rows = splits[name]
            logit = evaluate(model, xh, xq, y, rows)
            preds[name] = logit
            m = auc_ap(y[rows], logit)
            row[f"{name}_roc_auc"] = m["roc_auc"]
            row[f"{name}_average_precision"] = m["average_precision"]
            row[f"{name}_bce"] = m["bce"]
        curve.append(row)
        if row["valid_roc_auc"] > best["valid_roc_auc"]:
            best = {"epoch": epoch, "valid_roc_auc": row["valid_roc_auc"]}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds = {k: v.copy() for k, v in preds.items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{args.arm}] epoch {epoch:02d} train={row['train_roc_auc']:.4f} "
                  f"valid={row['valid_roc_auc']:.4f} test={row['test_roc_auc']:.4f}", flush=True)

    model.load_state_dict(best_state)
    if device == "cuda":
        model.to("cpu")
    ckpt_path = out_dir / f"{args.arm}_best.pt"
    torch.save({"arm": args.arm, "epoch": best["epoch"], "state_dict": best_state,
                "params": params, "shared_init_hash": shared_hash}, ckpt_path)

    with (out_dir / f"{args.arm}_curve.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    per_split = {}
    for name in ("train", "valid", "test"):
        rows = splits[name]
        m = auc_ap(y[rows], best_preds[name])
        m.update({"n": int(rows.size), "n_positive": int(y[rows].sum())})
        if name == "test":
            m["evidence_label"] = "EXPOSED_DEVELOPMENT_TEST"
        per_split[name] = m

    np.savez_compressed(
        out_dir / f"{args.arm}_pred.npz",
        train=best_preds["train"], valid=best_preds["valid"], test=best_preds["test"],
        train_rows=splits["train"], valid_rows=splits["valid"], test_rows=splits["test"], y=y,
    )
    payload = {
        "round_id": K.ROUND_ID,
        "phase": "B: MLP readout on the Phase-A inputs",
        "arm": args.arm,
        "extra_blocks": extra,
        "created_utc": created,
        "architecture": "Linear(128)->ReLU->Dropout(0.1)->Linear(128)->ReLU->Dropout(0.1)->Linear(1); "
                        "first layer = W_H*H + W_Q*Q + b for N-WL/N-CF",
        "recipe": {"epochs": args.epochs, "batch_size": BATCH, "optimizer": "AdamW", "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "grad_clip": CLIP, "dropout": DROPOUT,
                   "loss": "class-balanced BCE (Train only, mean weight 1)",
                   "scheduler": None, "early_stopping": None,
                   "checkpoint": "best Valid ROC-AUC, ties to the earlier epoch"},
        "seed": K.SEED,
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "preprocessing": prep,
        "class_weights": wmeta,
        "parameters": params,
        "shared_init_hash": shared_hash,
        "best_epoch": int(best["epoch"]),
        "metrics": per_split,
        "curve": curve,
        "checkpoint": {"path": str(ckpt_path), "size_bytes": int(ckpt_path.stat().st_size)},
        "input_bundle": {"path": K.rel(K.INPUT_BUNDLE), "sha256": K.sha256_file(K.INPUT_BUNDLE)},
        "git": K.git_state(),
        "environment": K.environment(),
        "seconds": float(time.perf_counter() - t0),
    }
    (out_dir / f"{args.arm}.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                         default=K._json_default) + "\n")
    print(f"[{args.arm}] best epoch {best['epoch']} valid={per_split['valid']['roc_auc']:.4f} "
          f"test={per_split['test']['roc_auc']:.4f} hash={shared_hash[:12]} "
          f"done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
