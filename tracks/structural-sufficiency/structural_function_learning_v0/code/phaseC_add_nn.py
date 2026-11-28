"""Phase C — neural readouts on the *same* raw feature sets as the trees.

Four new models (no GNN, no message passing, no extra inputs):

* ``H-NN-JOINT``      plain BCE MLP on the 4408 MolHIV columns
* ``H-NN-JOINT-BAL``  identical architecture/init/batch order, class-balanced BCE
* ``H-NN-ADD``        additive neural model, plain BCE, one 1-D piecewise-linear
                      function per input feature, f_j(0)=0 exactly
* ``Z-NN-ADD``        additive neural model on the frozen Z+Z-CF design, L1 on the
                      Train-standardised target (evaluated after inversion)

The two HIV MLPs are trained in the same process with ``torch.manual_seed(0)`` reset
before each construction and the same ``numpy`` batch-order generator; their initial
parameter hashes must be identical.  The only difference is the per-sample loss weight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import sfl_common as K

EPOCHS = 50
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
CLIP = 5.0
HIDDEN = 128
DROPOUT = 0.1
KNOTS = 4
KNOT_QUANTILES = (0.25, 0.50, 0.75, 0.90)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(n_in, HIDDEN)
        self.h2 = nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, 1)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop(Fn.relu(self.w1(x)))
        z = self.drop(Fn.relu(self.h2(z)))
        return self.out(z).squeeze(-1)


class AdditiveNN(nn.Module):
    """``b + sum_j f_j(x_j)`` with ``f_j(x) = a_j x + sum_k v_jk (relu(x-t_jk)-relu(-t_jk))``."""

    def __init__(self, knots: np.ndarray, active: np.ndarray) -> None:
        super().__init__()
        p = knots.shape[0]
        rng = np.random.default_rng(K.SEED)
        a0 = (rng.normal(0.0, 0.01, p) * active).astype(np.float32)
        v0 = (rng.normal(0.0, 0.01, (p, KNOTS)) * active[:, None]).astype(np.float32)
        self.a = nn.Parameter(torch.from_numpy(a0))
        self.v = nn.Parameter(torch.from_numpy(v0))
        self.t = nn.Parameter(torch.from_numpy(knots.astype(np.float32)))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("active", torch.from_numpy(active.astype(np.float32)))

    def feature_values(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.relu(x.unsqueeze(-1) - self.t) - torch.relu(-self.t)
        f = self.a * x + (self.v * r).sum(-1)
        return f * self.active

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_values(x)
        if self.training:
            f = Fn.dropout(f, p=DROPOUT, training=True)
        return self.bias + f.sum(dim=1)


def init_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def n_parameters(model: nn.Module) -> dict[str, int]:
    per = {name: int(p.numel()) for name, p in model.named_parameters()}
    return {"per_parameter": per, "total": int(sum(per.values()))}


def knot_init(x_train: np.ndarray, active_floor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Train non-zero 25/50/75/90 quantiles per column; inactive columns stay zero."""
    p = x_train.shape[1]
    knots = np.zeros((p, KNOTS), dtype=np.float32)
    active = np.zeros(p, dtype=bool)
    for j in range(p):
        col = x_train[:, j]
        nz = col[col != 0]
        if nz.size < active_floor:
            continue
        active[j] = True
        knots[j] = np.quantile(nz, KNOT_QUANTILES).astype(np.float32)
    return knots, active


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------


def class_balanced_weights(y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pos = float(y.sum())
    neg = float(y.size - pos)
    w_pos = 0.5 * float(y.size) / pos
    w_neg = 0.5 * float(y.size) / neg
    w = np.where(y > 0.5, w_pos, w_neg).astype(np.float64)
    w = w / w.mean()
    return w, {"pos_weight": w_pos, "neg_weight": w_neg, "fit_on": "Train only"}


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, rows: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = np.empty(rows.size, dtype=np.float64)
    for start in range(0, rows.size, batch):
        sel = rows[start:start + batch]
        out[start:start + sel.size] = model(x[sel]).detach().cpu().numpy()
    return out


def train_arm(arm: str, kind: str, x: np.ndarray, y: np.ndarray,
              rows: dict[str, np.ndarray], dataset: str,
              knots: np.ndarray | None = None, active: np.ndarray | None = None,
              balanced: bool = False, epochs: int = EPOCHS,
              out_dir: Path | None = None) -> dict[str, Any]:
    torch.manual_seed(K.SEED)
    np.random.seed(K.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_all = torch.from_numpy(x).to(device)
    n_train = rows["train"].size

    if kind == "mlp":
        model = MLP(x.shape[1]).to(device)
    elif kind == "add":
        assert knots is not None and active is not None
        model = AdditiveNN(knots, active).to(device)
    else:
        raise ValueError(kind)
    h_init = init_hash(model)
    params = n_parameters(model)
    print(f"[C] {arm} device={device} params={params['total']} init={h_init[:16]}", flush=True)

    if dataset == "hiv":
        ytr = y[rows["train"]].astype(np.float32)
        if balanced:
            w, wmeta = class_balanced_weights(ytr)
        else:
            w, wmeta = np.ones_like(ytr), {"pos_weight": 1.0, "neg_weight": 1.0,
                                           "fit_on": "Train only (unweighted)"}
        w_t = torch.from_numpy(w.astype(np.float32)).to(device)
        y_t = torch.from_numpy(ytr).to(device)
        y_std_mu, y_std_sd = 0.0, 1.0
    else:
        y_std_mu = float(y[rows["train"]].mean())
        y_std_sd = float(y[rows["train"]].std())
        y_t = torch.from_numpy(((y[rows["train"]] - y_std_mu) / y_std_sd)
                               .astype(np.float32)).to(device)
        w_t = torch.ones(n_train, dtype=torch.float32, device=device)
        wmeta = {"loss": "L1 on Train-standardised target", "mean": y_std_mu, "std": y_std_sd}

    if kind == "add":
        decay = [model.a, model.v, model.bias]
        no_decay = [model.t]
        opt = torch.optim.AdamW([
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=LR)
        optimizer_desc = (f"AdamW lr={LR} wd={WEIGHT_DECAY} on (a,v,b); "
                          "knots t: wd=0")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer_desc = f"AdamW lr={LR} wd={WEIGHT_DECAY}"

    rng = np.random.default_rng(K.SEED)
    orders = [rng.permutation(n_train) for _ in range(epochs)]
    curve: list[dict[str, Any]] = []
    best = {"epoch": -1, "value": -np.inf if dataset == "hiv" else np.inf}
    best_state = None
    best_pred: dict[str, np.ndarray] = {}
    last_pred: dict[str, np.ndarray] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        order = orders[epoch - 1]
        tot, seen = 0.0, 0
        for start in range(0, n_train, BATCH):
            sel = order[start:start + BATCH]
            opt.zero_grad(set_to_none=True)
            out = model(x_all[rows["train"][sel]])
            if dataset == "hiv":
                loss = Fn.binary_cross_entropy_with_logits(out, y_t[sel], weight=w_t[sel])
            else:
                loss = Fn.l1_loss(out, y_t[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            tot += float(loss.item()) * sel.size
            seen += sel.size
        row: dict[str, Any] = {"epoch": epoch, "train_loss": tot / seen}
        preds: dict[str, np.ndarray] = {}
        for split in K.SPLITS:
            raw = predict(model, x_all, rows[split])
            preds[split] = raw
            if dataset == "hiv":
                m = K.auc_ap(y[rows[split]], raw)
                row[f"{split}_roc_auc"] = m["roc_auc"]
                row[f"{split}_average_precision"] = m["average_precision"]
            else:
                p = raw * y_std_sd + y_std_mu
                preds[split] = p
                m = K.reg_metrics(y[rows[split]], p)
                row[f"{split}_mae"] = m["mae"]
                row[f"{split}_rmse"] = m["rmse"]
        curve.append(row)
        value = row["valid_roc_auc"] if dataset == "hiv" else row["valid_mae"]
        better = value > best["value"] if dataset == "hiv" else value < best["value"]
        if better:
            best = {"epoch": epoch, "value": float(value)}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = {k: v.copy() for k, v in preds.items()}
        last_pred = preds
        if epoch % 5 == 0 or epoch == 1:
            if dataset == "hiv":
                print(f"[C] {arm} ep{epoch:02d} train={row['train_roc_auc']:.4f} "
                      f"valid={row['valid_roc_auc']:.4f} test={row['test_roc_auc']:.4f}",
                      flush=True)
            else:
                print(f"[C] {arm} ep{epoch:02d} train={row['train_mae']:.4f} "
                      f"valid={row['valid_mae']:.4f} test={row['test_mae']:.4f}", flush=True)

    model.load_state_dict(best_state)
    # post-training additive checks ----------------------------------------
    checks: dict[str, Any] = {}
    if kind == "add":
        checks = additive_checks(model, x, rows, dataset)
    model_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out_dir = out_dir or K.RUNS
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{arm}_best.pt"
    torch.save({"arm": arm, "kind": kind, "dataset": dataset,
                "epoch": best["epoch"], "state_dict": model_cpu,
                "params": params, "init_hash": h_init,
                "balanced": bool(balanced), "knots_init": knots, "active": active,
                "target_standardisation": {"mean": y_std_mu, "std": y_std_sd},
                "additive_checks": checks}, ckpt_path)

    # metrics from the stored best predictions ------------------------------
    per_split = {}
    for split in K.SPLITS:
        r = rows[split]
        if dataset == "hiv":
            m = K.auc_ap(y[r], best_pred[split])
        else:
            m = K.reg_metrics(y[r], best_pred[split])
        m["evidence_label"] = ("DEVELOPMENT_TEST" if split == "test"
                               else ("VALID" if split == "valid" else "TRAIN"))
        per_split[split] = m
    np.savez_compressed(K.PREDICTIONS / f"{arm}.npz",
                        **{f"pred_{k}": v for k, v in best_pred.items()},
                        **{f"last_{k}": v for k, v in last_pred.items()},
                        y=y, graph_id=np.arange(y.size),
                        train_rows=rows["train"], valid_rows=rows["valid"],
                        test_rows=rows["test"], arm=arm, dataset=dataset,
                        best_epoch=int(best["epoch"]))
    K.CURVES.mkdir(parents=True, exist_ok=True)
    with (K.CURVES / f"{arm}_curve.csv").open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        wtr.writeheader()
        wtr.writerows(curve)
    payload = {
        "round_id": K.ROUND_ID, "phase": "C: neural readout", "arm": arm,
        "dataset": dataset, "kind": kind, "architecture": (
            "Linear(4408,128)->ReLU->Dropout(0.1)->Linear(128,128)->ReLU->"
            "Dropout(0.1)->Linear(128,1)" if kind == "mlp" else
            "b + sum_j f_j(x_j); f_j(x)=a_j x + sum_k v_jk(relu(x-t_jk)-relu(-t_jk)); "
            "4 learnable knots per feature; dropout(0.1) on each single-feature output"),
        "recipe": {"epochs": epochs, "batch_size": BATCH, "optimizer": optimizer_desc,
                   "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": CLIP,
                   "scheduler": None, "early_stopping": None,
                   "loss": ("binary CE" + (" with class-balanced weights" if balanced else "")
                            if dataset == "hiv" else "L1 on Train-standardised target"),
                   "checkpoint": ("max Valid ROC-AUC" if dataset == "hiv"
                                  else "min Valid MAE") + ", ties -> earlier epoch"},
        "seed": K.SEED, "device": device, "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": params, "init_hash": h_init,
        "class_weights": wmeta,
        "target_standardisation": {"mean": y_std_mu, "std": y_std_sd, "fit_on": "Train only"},
        "knot_init": ({"quantiles": list(KNOT_QUANTILES), "fit_on": "Train non-zero values",
                       "n_active_features": int(active.sum()),
                       "n_inactive_features": int((~active).sum())} if kind == "add" else None),
        "additive_checks": checks,
        "best_epoch": int(best["epoch"]), "metrics": per_split, "curve": curve,
        "checkpoint_path": K.rel(ckpt_path),
        "checkpoint_sha256": K.sha256_file(ckpt_path),
        "input_identity": {"n_rows": int(y.size), "n_columns": int(x.shape[1]),
                           "train_rows_sha256": K.sha256_array(rows["train"])},
        "seconds": None, "git": K.git_state(), "environment": K.environment(),
    }
    K.write_json(K.ANALYSIS / f"phaseC_{arm}.json", payload)
    return payload


@torch.no_grad()
def additive_checks(model: AdditiveNN, x: np.ndarray, rows: dict[str, np.ndarray],
                    dataset: str) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(K.SEED)
    sample = np.sort(rng.choice(x.shape[0], size=min(256, x.shape[0]), replace=False))
    xs = torch.from_numpy(x[sample]).to(device)
    f = model.feature_values(xs)
    pred = model(xs)
    zeros = torch.zeros((1, x.shape[1]), device=device)
    f0 = model.feature_values(zeros)
    # independent float64 recomputation of sum_j f_j(x_j) + b
    a = model.a.detach().cpu().numpy().astype(np.float64)
    v = model.v.detach().cpu().numpy().astype(np.float64)
    t = model.t.detach().cpu().numpy().astype(np.float64)
    bias = float(model.bias.detach().cpu())
    xd = x[sample].astype(np.float64)
    r = np.maximum(xd[:, :, None] - t[None], 0.0) - np.maximum(-t[None], 0.0)
    contrib = a[None] * xd + (v[None] * r).sum(axis=2)
    numpy_pred = bias + contrib.sum(axis=1)
    return {
        "f_j_at_zero_max_abs": float(np.max(np.abs(f0.detach().cpu().numpy()))),
        "forward_minus_feature_sum_max_abs": float(np.max(np.abs(
            pred.detach().cpu().numpy() - (bias + f.sum(1).detach().cpu().numpy())))),
        "dense_forward_vs_numpy_max_abs": float(np.max(np.abs(
            pred.detach().cpu().numpy() - numpy_pred))),
        "n_rows_checked": int(sample.size),
        "f_j_0_zero_analytic": True,
        "sparse_implementation_used": False,
        "note": ("dense evaluation used; f_j(0)=0 holds by construction because "
                 "relu(0-t)-relu(-t)==0"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="comma list of H-NN-JOINT,H-NN-JOINT-BAL,H-NN-ADD,Z-NN-ADD")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out-dir", default=str(K.RUNS))
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K.ANALYSIS.mkdir(parents=True, exist_ok=True)
    K.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    K.CURVES.mkdir(parents=True, exist_ok=True)
    K.MODELS.mkdir(parents=True, exist_ok=True)
    allowed = {"H-NN-JOINT", "H-NN-JOINT-BAL", "H-NN-ADD", "Z-NN-ADD"}
    if not set(arms) <= allowed:
        raise SystemExit(f"unknown arms {set(arms) - allowed}")

    for dataset in ("hiv", "zinc"):
        todo = [a for a in arms if (a.startswith("H-") == (dataset == "hiv"))]
        if not todo:
            continue
        print(f"[C] dataset={dataset} arms={todo}", flush=True)
        t0 = time.perf_counter()
        if dataset == "hiv":
            data = K.HIVData()
            x, prep = data.nn_design_dense()
            y = data.y
        else:
            data = K.ZincData()
            x, prep = data.nn_design_dense()
            y = data.y
        print(f"[C] design {x.shape} {x.nbytes / 1e9:.2f} GB", flush=True)
        knots = active = None
        if any(a.endswith("ADD") for a in todo):
            print("[C] computing Train non-zero knot quantiles", flush=True)
            knots, active = knot_init(x[data.rows["train"]])
            print(f"[C] knots active={int(active.sum())}/{active.size}", flush=True)
        h_joint = None
        for arm in todo:
            kind = "add" if arm.endswith("ADD") else "mlp"
            t_arm = time.perf_counter()
            payload = train_arm(
                arm, kind, x, np.asarray(y), data.rows, dataset,
                knots=knots if kind == "add" else None,
                active=active if kind == "add" else None,
                balanced=arm.endswith("BAL"), epochs=args.epochs, out_dir=out_dir)
            payload["preprocessing"] = prep
            payload["seconds"] = float(time.perf_counter() - t_arm)
            K.write_json(K.ANALYSIS / f"phaseC_{arm}.json", payload)
            if arm == "H-NN-JOINT":
                h_joint = payload["init_hash"]
            if arm == "H-NN-JOINT-BAL":
                assert h_joint == payload["init_hash"], (
                    "H-NN-JOINT and H-NN-JOINT-BAL initialisation differ")
                print("[C] JOINT / JOINT-BAL initialisation hashes match", flush=True)
            print(f"[C] {arm} best epoch {payload['best_epoch']} "
                  f"valid={payload['metrics']['valid']} "
                  f"({payload['seconds']:.1f}s)", flush=True)
        print(f"[C] dataset={dataset} total {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
