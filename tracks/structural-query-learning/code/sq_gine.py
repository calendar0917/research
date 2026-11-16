"""Neural performance reference G: frozen 5-layer GINE from scratch.

The reference exists only to say how far the query head is from a strong
neural reader on the same official split.  It is deliberately the plain,
well-understood recipe frozen by the round brief and nothing else:

* node init = sum of 9 per-column embeddings(128) + degree embedding + min-ring
  embedding; edge init = sum of 3 per-column bond embeddings(128);
* 5 ``GINEConv`` blocks (MLP 128->256->128, SiLU, ``train_eps=True``) each
  followed by BatchNorm, SiLU, dropout 0.10 and a residual;
* readout ``concat(sum_pool, mean_pool, log1p(n_atoms))`` -> MLP 257->128->1;
* MSE on the standardised target, batch 128, AdamW wd 1e-5, grad clip 5;
* 360 epochs; 5-epoch linear warmup from peak/10 then per-optimizer-step cosine
  down to 1e-5 (the base LR is never multiplied twice);
* two frozen peak LRs (1e-3 and 3e-4) with *identical* initial weights and
  identical Train batch order.

Every epoch runs a full eval-mode pass over Train/Val/Test; Train metrics are
recomputed with dropout off (the online training loss is never reported).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import sq_common as C
from sq_ast import PackedGraph
from sq_common import mae, rmse

EPOCHS = 360
BATCH_SIZE = 128
WARMUP_EPOCHS = 5
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
DROPOUT = 0.10
HIDDEN = 128
N_LAYERS = 5
PEAK_LRS: tuple[float, ...] = (1e-3, 3e-4)
DATALOADER_SEED = 20261116


def _torch():
    import torch

    return torch


def build_data_list(
    packed: PackedGraph,
    n_categories: Mapping[str, int],
    scaler: Mapping[str, float],
):
    """One ``torch_geometric.data.Data`` per molecule (both bond directions).

    ``Data.y`` is the *standardised* target (Train mean/std); ``_evaluate``
    converts back exactly once.  Storing raw labels here and de-standardising
    again would scale every reported RMSE by the Train standard deviation.
    """
    torch = _torch()
    from torch_geometric.data import Data

    data_list = []
    node_ptr = np.concatenate([[0], np.cumsum(packed.graph_n_atoms)])
    bond_start = np.concatenate([[0], np.cumsum(packed.graph_n_bonds)])
    for graph in range(packed.n_graphs):
        a0, a1 = int(node_ptr[graph]), int(node_ptr[graph + 1])
        b0, b1 = int(bond_start[graph]), int(bond_start[graph + 1])
        x = torch.as_tensor(packed.atom_feat[a0:a1].astype(np.int64), dtype=torch.long)
        deg = torch.as_tensor(packed.degree[a0:a1].astype(np.int64), dtype=torch.long)
        ring = torch.as_tensor(packed.min_ring[a0:a1].astype(np.int64), dtype=torch.long)
        bu = packed.bond_u[b0:b1].astype(np.int64) - a0
        bv = packed.bond_v[b0:b1].astype(np.int64) - a0
        bfeat = packed.bond_feat[b0:b1].astype(np.int64)
        edge_index = torch.as_tensor(
            np.concatenate([np.stack([bu, bv]), np.stack([bv, bu])], axis=1), dtype=torch.long
        )
        edge_cat = torch.as_tensor(np.concatenate([bfeat, bfeat], axis=0), dtype=torch.long)
        data = Data(
            num_nodes=int(a1 - a0),  # PyG's inference is unreliable for custom attrs
            x_cat=x,
            deg=deg,
            ring=ring,
            edge_index=edge_index,
            edge_cat=edge_cat,
            y=torch.as_tensor(
                [(float(packed.labels[graph]) - float(scaler["mean"])) / float(scaler["std"])],
                dtype=torch.float32,
            ),
            n_atoms=torch.as_tensor([int(a1 - a0)], dtype=torch.long),
            row_id=torch.as_tensor([int(packed.row_ids[graph])], dtype=torch.long),
        )
        data_list.append(data)
    return data_list


def category_counts(packed_splits: Sequence[PackedGraph]) -> dict[str, int]:
    """Label-free category counts observed in the cache (no target involved)."""
    atom = np.concatenate([p.atom_feat for p in packed_splits], axis=0)
    bond = np.concatenate([p.bond_feat for p in packed_splits], axis=0)
    degree = np.concatenate([p.degree for p in packed_splits])
    ring = np.concatenate([p.min_ring for p in packed_splits])
    counts = {f"atom_{j}": int(atom[:, j].max()) + 1 for j in range(atom.shape[1])}
    counts.update({f"bond_{j}": int(bond[:, j].max()) + 1 for j in range(bond.shape[1])})
    counts["degree"] = int(degree.max()) + 1
    counts["ring"] = int(ring.max()) + 1
    return counts


class GINEModel:
    """Plain module container (kept out of the import path until torch is needed)."""

    @staticmethod
    def build(n_categories: Mapping[str, int]):
        torch = _torch()
        import torch.nn as nn
        from torch_geometric.nn import GINEConv

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                h = HIDDEN
                self.atom_embeddings = nn.ModuleList(
                    [nn.Embedding(int(n_categories[f"atom_{j}"]), h) for j in range(9)]
                )
                self.degree_embedding = nn.Embedding(int(n_categories["degree"]), h)
                self.ring_embedding = nn.Embedding(int(n_categories["ring"]), h)
                self.bond_embeddings = nn.ModuleList(
                    [nn.Embedding(int(n_categories[f"bond_{j}"]), h) for j in range(3)]
                )
                self.convs = nn.ModuleList(
                    [
                        GINEConv(
                            nn.Sequential(
                                nn.Linear(h, 2 * h),
                                nn.SiLU(),
                                nn.Linear(2 * h, h),
                            ),
                            train_eps=True,
                        )
                        for _ in range(N_LAYERS)
                    ]
                )
                self.norms = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(N_LAYERS)])
                self.drop = nn.Dropout(DROPOUT)
                self.head = nn.Sequential(
                    nn.Linear(2 * h + 1, h),
                    nn.SiLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(h, 1),
                )

            def encode_nodes(self, x_cat, deg, ring):
                out = self.atom_embeddings[0](x_cat[:, 0])
                for j in range(1, 9):
                    out = out + self.atom_embeddings[j](x_cat[:, j])
                out = out + self.degree_embedding(deg) + self.ring_embedding(ring)
                return out

            def encode_edges(self, edge_cat):
                out = self.bond_embeddings[0](edge_cat[:, 0])
                for j in range(1, 3):
                    out = out + self.bond_embeddings[j](edge_cat[:, j])
                return out

            def forward(self, batch):
                x = self.encode_nodes(batch.x_cat, batch.deg, batch.ring)
                edge_attr = self.encode_edges(batch.edge_cat)
                for conv, norm in zip(self.convs, self.norms):
                    residual = x
                    update = conv(x, batch.edge_index, edge_attr)
                    x = norm(update)
                    x = torch.nn.functional.silu(x)
                    x = self.drop(x)
                    x = x + residual
                from torch_geometric.nn import global_add_pool, global_mean_pool

                batch_index = batch.batch
                summed = global_add_pool(x, batch_index)
                meaned = global_mean_pool(x, batch_index)
                n_atoms = batch.n_atoms.view(-1, 1).to(x.dtype)
                pooled = torch.cat([summed, meaned, torch.log1p(n_atoms)], dim=1)
                return self.head(pooled).view(-1)

        return _Net()


def _state_hash(model) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _evaluate(model, loader, device, scaler: Mapping[str, float]) -> dict[str, np.ndarray]:
    torch = _torch()
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.detach().cpu().numpy().reshape(-1))
            targets.append(batch.y.detach().cpu().numpy().reshape(-1))
            rows.append(batch.row_id.detach().cpu().numpy().reshape(-1))
    pred_std = np.concatenate(preds)
    target_std = np.concatenate(targets)
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    return {
        "pred_std": pred_std,
        "pred_raw": pred_std * std + mean,
        "target_raw": target_std * std + mean,
        "row_ids": np.concatenate(rows).astype(np.int64),
    }


def train_one_lr(
    *,
    peak_lr: float,
    device: str,
    data: Mapping[str, Any],
    scaler: Mapping[str, float],
    n_categories: Mapping[str, int],
    seed: int = C.TRAIN_SEED,
    epochs: int = EPOCHS,
) -> dict[str, Any]:
    torch = _torch()
    from torch_geometric.loader import DataLoader

    # Seed *before* model construction so both LR presets share one init.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = GINEModel.build(n_categories).to(device)
    init_hash = _state_hash(model)

    loader = DataLoader(
        data["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(DATALOADER_SEED),
    )
    eval_loaders = {
        split: DataLoader(data[split], batch_size=512, shuffle=False)
        for split in ("train", "valid", "test")
    }
    first_batch_rows = next(iter(loader)).row_id.view(-1).tolist()

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(loader))
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            frac = step / max(1, warmup_steps)
            return (peak_lr / 10.0) + (peak_lr - peak_lr / 10.0) * frac
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return MIN_LR + 0.5 * (peak_lr - MIN_LR) * (1.0 + np.cos(np.pi * progress))

    curves: list[dict[str, Any]] = []
    best = {"valid_rmse_raw": float("inf"), "epoch": None}
    best_state = None
    best_predictions: dict[str, Any] = {}
    final_predictions: dict[str, Any] = {}
    global_step = 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.param_groups[0]["lr"] = lr_at(global_step)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = torch.nn.functional.mse_loss(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        epoch_metrics: dict[str, Any] = {"epoch": epoch, "peak_lr": peak_lr}
        predictions: dict[str, Any] = {}
        for split, eval_loader in eval_loaders.items():
            out = _evaluate(model, eval_loader, device, scaler)
            predictions[split] = out
            epoch_metrics[f"{split}_rmse_raw"] = rmse(out["pred_raw"], out["target_raw"])
            epoch_metrics[f"{split}_mae_raw"] = mae(out["pred_raw"], out["target_raw"])
        epoch_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        epoch_metrics["seconds"] = time.perf_counter() - t0
        curves.append(epoch_metrics)
        if epoch_metrics["valid_rmse_raw"] < best["valid_rmse_raw"]:
            best = {
                "valid_rmse_raw": epoch_metrics["valid_rmse_raw"],
                "epoch": epoch,
            }
            best_state = copy.deepcopy(model.state_dict())
            best_predictions = predictions
        final_predictions = predictions
    seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "peak_lr": peak_lr,
        "init_state_hash": init_hash,
        "dataloader_seed": DATALOADER_SEED,
        "first_batch_row_ids": first_batch_rows,
        "steps_per_epoch": steps_per_epoch,
        "curves": curves,
        "best": {
            "epoch": best["epoch"],
            "valid_rmse_raw": best["valid_rmse_raw"],
            "train_rmse_raw": curves[best["epoch"] - 1]["train_rmse_raw"],
            "test_rmse_raw": curves[best["epoch"] - 1]["test_rmse_raw"],
            "train_mae_raw": curves[best["epoch"] - 1]["train_mae_raw"],
            "valid_mae_raw": curves[best["epoch"] - 1]["valid_mae_raw"],
            "test_mae_raw": curves[best["epoch"] - 1]["test_mae_raw"],
        },
        "final": {
            "epoch": epochs,
            "train_rmse_raw": curves[-1]["train_rmse_raw"],
            "valid_rmse_raw": curves[-1]["valid_rmse_raw"],
            "test_rmse_raw": curves[-1]["test_rmse_raw"],
        },
        "last20": {
            f"mean_last20_{split}_rmse_raw": float(
                np.mean([row[f"{split}_rmse_raw"] for row in curves[-20:]])
            )
            for split in ("train", "valid", "test")
        },
        "predictions_best": best_predictions,
        "predictions_final": final_predictions,
        "seconds": seconds,
        "global_step": global_step,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def run_gine(
    *,
    packed: Mapping[str, PackedGraph],
    scaler: Mapping[str, float],
    device: str,
    epochs: int = EPOCHS,
    peak_lrs: Sequence[float] = PEAK_LRS,
) -> dict[str, Any]:
    n_categories = category_counts([packed["train"], packed["valid"], packed["test"]])
    data = {
        split: build_data_list(packed[split], n_categories, scaler)
        for split in ("train", "valid", "test")
    }
    results = {}
    for peak_lr in peak_lrs:
        results[str(peak_lr)] = train_one_lr(
            peak_lr=float(peak_lr),
            device=device,
            data=data,
            scaler=scaler,
            n_categories=n_categories,
            epochs=epochs,
        )
    chosen_key = min(
        results,
        key=lambda key: (results[key]["best"]["valid_rmse_raw"], -float(key)),
    )
    chosen = results[chosen_key]
    init_hashes = {key: results[key]["init_state_hash"] for key in results}
    batch_orders = {key: results[key]["first_batch_row_ids"] for key in results}
    return {
        "method": "G",
        "selected_peak_lr": float(chosen_key),
        "selected": chosen["best"],
        "selected_predictions": chosen["predictions_best"],
        "per_lr": {
            key: {k: v for k, v in payload.items() if not k.startswith("predictions_")}
            for key, payload in results.items()
        },
        "curves": [
            {**row, "peak_lr": float(key)} for key, payload in results.items() for row in payload["curves"]
        ],
        "init_hashes_identical": len(set(init_hashes.values())) == 1,
        "init_hashes": init_hashes,
        "batch_order_identical": len({tuple(v) for v in batch_orders.values()}) == 1,
        "n_categories": n_categories,
        "n_parameters": chosen["n_parameters"],
        "seconds_total": float(sum(payload["seconds"] for payload in results.values())),
        "last20_selected": chosen["last20"],
        "final_round_selected": chosen["final"],
    }
