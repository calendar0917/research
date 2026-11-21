"""External prediction coordinate: the official GINE backbone only.

This module trains the *standard dense GINE* reference (5 layers, hidden 128,
sum+mean readout, official categorical atom/bond features, no ports, no cuts, no
words) on exactly the splits, target scaling, checkpoint rule and evaluation
protocol used by the response-composition units.  It exists to give the main
table a raw-GINE coordinate in the original units; it is never a residual, a
distillation source or a component of any candidate model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import trc_common as C
import trc_data as D

GINE_FACTORY = Path(__file__).resolve().parents[2] / "word_sentence_port_binding_context_v2" / "code"
RECIPE = {
    "peak_lr": 1e-3, "min_lr": 1e-5, "weight_decay": 1e-5, "clip": 5.0,
    "warmup_epochs": 5, "batch": 128, "eval_batch": 512, "epochs": 160, "seed": 0,
}


def _load_gine_factory():
    sys.path.insert(0, str(GINE_FACTORY))
    import wp_train as WT

    return WT.build_gine_model


def build_data(raw: Mapping[str, Any], rows: np.ndarray, scaler: Mapping[str, float]):
    from torch_geometric.data import Data

    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom = np.asarray(raw["atom_feat"], np.int64)
    bond = np.asarray(raw["bond_feat"], np.int64)
    bu = np.asarray(raw["bond_u"], np.int64)
    bv = np.asarray(raw["bond_v"], np.int64)
    labels = np.asarray(raw["labels"], np.float64)
    out = []
    for row in rows.tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        src = np.concatenate([bu[b0:b1], bv[b0:b1]])
        dst = np.concatenate([bv[b0:b1], bu[b0:b1]])
        eattr = np.concatenate([bond[b0:b1], bond[b0:b1]])
        data = Data(
            x_cat=_ct(atom[a0:a1]),
            edge_index=_ct(np.stack([src, dst])),
            edge_cat=_ct(eattr),
            y=_ct([(labels[row] - scaler["mean"]) / scaler["std"]], dtype=None),
            n_atoms=_ct([n]),
            num_nodes=n,
        )
        out.append(data)
    return out


def _ct(arr, dtype="long"):
    import torch

    t = torch.as_tensor(np.ascontiguousarray(arr))
    return t.to(torch.float64 if dtype is None else torch.long)


def evaluate(model, data, device: str, batch: int, scaler: Mapping[str, float]):
    import torch
    from torch_geometric.data import Batch

    model.eval()
    preds = np.zeros(len(data), np.float64)
    with torch.no_grad():
        for start in range(0, len(data), batch):
            chunk = data[start:start + batch]
            b = Batch.from_data_list(chunk).to(device)
            preds[start:start + len(chunk)] = model(b).detach().double().cpu().numpy()
    target = np.asarray([float(d.y.reshape(-1)[0]) for d in data], np.float64)
    raw_pred = preds * scaler["std"] + scaler["mean"]
    raw_true = target * scaler["std"] + scaler["mean"]
    return preds, {"rmse": C.rmse(raw_pred, raw_true), "mae": C.mae(raw_pred, raw_true)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dataset", default="lipo", choices=("lipo", "esol"))
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from torch_geometric.data import Batch

    cfg = dict(C.load_config())
    epochs = args.epochs or RECIPE["epochs"]
    C.set_deterministic(args.seed)
    out = C.run_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    t_start = time.perf_counter()
    log(f"run={args.run_id} GINE_RAW dataset={args.dataset} device={args.device}"
        f" epochs={epochs}")

    raw = D.load_raw_cache(cfg, args.dataset)
    scaler = C.target_scaler(np.asarray(raw["labels"], np.float64),
                             np.asarray(raw["train_idx"], np.int64))
    splits = {}
    for split in C.SPLITS:
        rows = np.asarray(raw[f"{split}_idx"], np.int64)
        splits[split] = build_data(raw, rows, scaler)
        log(f"  {split}: {len(rows)} molecules")
    labels = {s: np.asarray(raw["labels"], np.float64)[np.asarray(raw[f"{s}_idx"], np.int64)]
              for s in C.SPLITS}
    atom_dims, bond_dims = C.official_feature_dims()
    factory = _load_gine_factory()
    model = factory({}, args.seed, args.device, raw_features=True)
    log(f"  params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_loader_order = np.random.default_rng(args.seed)
    steps_per_epoch = int(np.ceil(len(splits["train"]) / RECIPE["batch"]))
    total_steps = epochs * steps_per_epoch
    warmup = RECIPE["warmup_epochs"] * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["peak_lr"],
                            weight_decay=RECIPE["weight_decay"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return RECIPE["peak_lr"] * (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))
        return RECIPE["min_lr"] + (RECIPE["peak_lr"] - RECIPE["min_lr"]) * cos

    best = {"valid_rmse": float("inf"), "epoch": -1, "state": None, "preds": None}
    rows_csv: list[dict[str, Any]] = []
    step = 0
    peak = 0
    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        order = train_loader_order.permutation(len(splits["train"]))
        loss_sum = 0.0
        for start in range(0, len(order), RECIPE["batch"]):
            idx = order[start:start + RECIPE["batch"]]
            batch = Batch.from_data_list([splits["train"][i] for i in idx]).to(args.device)
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = torch.mean((pred - batch.y.view(-1)) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["clip"])
            opt.step()
            step += 1
            loss_sum += float(loss.detach().cpu()) * int(batch.num_graphs)
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                peak = max(peak, int(torch.cuda.max_memory_allocated()))
        metrics = {}
        preds = {}
        for split in C.SPLITS:
            p, m = evaluate(model, splits[split], args.device, RECIPE["eval_batch"], scaler)
            preds[split] = p * scaler["std"] + scaler["mean"]
            metrics[split] = m
        secs = time.perf_counter() - t_epoch
        row = {"run_id": args.run_id, "epoch": epoch, "lr": lr_at(step),
               "train_loss": loss_sum / max(1, len(order)), "epoch_seconds": secs,
               "wall_seconds": time.perf_counter() - t_start}
        for split in C.SPLITS:
            row[f"{split}_rmse"] = metrics[split]["rmse"]
            row[f"{split}_mae"] = metrics[split]["mae"]
        rows_csv.append(row)
        log(f"[{args.run_id}] epoch {epoch:3d}/{epochs} lr={lr_at(step):.2e} "
            f"train={metrics['train']['rmse']:.4f} valid={metrics['valid']['rmse']:.4f} "
            f"test={metrics['test']['rmse']:.4f} {secs:.1f}s")
        if metrics["valid"]["rmse"] < best["valid_rmse"]:
            best = {"valid_rmse": metrics["valid"]["rmse"], "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "preds": {s: preds[s].copy() for s in C.SPLITS},
                    "metrics": {s: dict(metrics[s]) for s in C.SPLITS}}
        C.write_csv(out / "curves.csv", rows_csv)

    torch.save({"state": best["state"], "epoch": best["epoch"]}, out / "state_best.pt")
    payload = {}
    for split in C.SPLITS:
        payload[f"{split}_pred"] = best["preds"][split]
        payload[f"{split}_label"] = labels[split]
    np.savez(out / "predictions_best.npz", **payload)
    final = rows_csv[-1]
    payload_final = {}
    for split in C.SPLITS:
        p, _ = evaluate(model, splits[split], args.device, RECIPE["eval_batch"], scaler)
        payload_final[f"{split}_pred"] = p * scaler["std"] + scaler["mean"]
        payload_final[f"{split}_label"] = labels[split]
    np.savez(out / "predictions_final.npz", **payload_final)
    manifest = {
        "run_id": args.run_id, "arm": "GINE_RAW", "supervision": "NONE",
        "dataset": args.dataset, "epochs": epochs, "seed": args.seed,
        "recipe": RECIPE, "scaler": scaler,
        "factory": str(GINE_FACTORY.relative_to(C.ROUND_DIR.parents[1])),
        "params": {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "best": {"epoch": best["epoch"], **best["metrics"]},
        "final": {"epoch": epochs, "train_rmse": final["train_rmse"],
                  "valid_rmse": final["valid_rmse"], "test_rmse": final["test_rmse"]},
        "wall_seconds": time.perf_counter() - t_start,
        "gpu_hours": (time.perf_counter() - t_start) / 3600.0,
        "max_vram_bytes": int(peak),
        "commit": C.git_commit(), "environment": C.environment(),
        "note": "external prediction coordinate only: official categorical features, "
                "no ports, no cuts, no shared response state",
    }
    C.write_json(out / "manifest.json", manifest)
    log(f"done best_epoch={best['epoch']} valid={best['valid_rmse']:.4f} "
        f"test={best['metrics']['test']['rmse']:.4f} wall={(time.perf_counter() - t_start) / 60:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
