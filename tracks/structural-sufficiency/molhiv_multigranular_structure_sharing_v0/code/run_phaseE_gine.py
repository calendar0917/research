"""Phase E: conditional plain-GINE verification with an explicit structure branch.

Conditional stage -- PROTOCOL.md §8.  Only plain GINE: 3 layers, hidden 64,
global mean pooling, scalar graph logit, OGB atom/bond features.  Structure and
Morgan enter *only* through a simple graph-level linear branch:

    z = z_GINE + z_M + z_structure

Both linear branches are zero-initialised and regularised with an explicit
``0.5 * lambda * ||w||^2`` penalty (``lambda = 1 / (0.3 * n_train)``), never with
AdamW decoupled weight decay.

Arms
----
``G-M``          GINE + Morgan branch
``G-STRUCT-U``   G-M + the Phase-E-selected structure block, uniform penalty
``G-STRUCT-W``   same, with the true support weights ``a_f`` on the FULL columns

Only arms that the protocol's trigger conditions select are run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import random
import socket
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

import msds_common as C
import msds_data as D

BUNDLE = C.ROUND / "bundle" / "phaseE_inputs.npz"


def seed_all(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def param_hash(state: dict, torch) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(np.ascontiguousarray(state[name].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def load_bundle() -> dict[str, Any]:
    with np.load(BUNDLE) as z:
        out = {
            "C": sp.csr_matrix((z["C_data"], z["C_indices"], z["C_indptr"]),
                               shape=(int(z["n_rows"]), int(z["n_coarse"]))),
            "F": sp.csr_matrix((z["F_data"], z["F_indices"], z["F_indptr"]),
                               shape=(int(z["n_rows"]), int(z["n_full"]))),
            "morgan": np.unpackbits(z["morgan_packed"], axis=1)[:, : int(z["morgan_n_bits"])].astype(np.float32),
            "m_extra": np.asarray(z["m_extra"], dtype=np.float32),
            "n_atoms": z["n_atoms"], "n_bonds": z["n_bonds"],
            "support_weight": z["support_weight"],
        }
    out["M"] = sp.hstack(
        [sp.csr_matrix(out["morgan"]), sp.csr_matrix(out["m_extra"])], format="csr"
    )
    return out


def build_graphs(molecules: D.Molecules, torch):
    """PyG ``Data`` objects with the OGB raw attributes and physical bonds (doubled for message passing)."""
    from torch_geometric.data import Data

    graphs = []
    x_all = np.asarray(molecules.atom_feat, dtype=np.int64)
    for g in range(molecules.n_molecules):
        lo, hi = int(molecules.atom_offsets[g]), int(molecules.atom_offsets[g + 1])
        bu, bv, bf = molecules.bonds(g)
        src = np.concatenate([bu, bv])
        dst = np.concatenate([bv, bu])
        attrs = np.concatenate([bf, bf]).astype(np.int64)
        edge_index = torch.as_tensor(np.stack([src, dst]), dtype=torch.long)
        edge_attr = torch.as_tensor(attrs, dtype=torch.long)
        graphs.append(Data(
            x=torch.as_tensor(x_all[lo:hi], dtype=torch.long),
            edge_index=edge_index, edge_attr=edge_attr,
            y=torch.tensor([float(molecules.label[g])], dtype=torch.float32),
            graph_index=torch.tensor([g], dtype=torch.long),
        ))
    return graphs


def make_model(hidden: int, layers: int, n_m: int, n_s: int, use_struct: bool, torch):
    import torch.nn as nn
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    from torch_geometric.nn import GINEConv

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atom_encoder = AtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(layers):
                mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.head = nn.Linear(hidden, 1)
            self.m_branch = nn.Linear(n_m, 1, bias=False)
            self.s_branch = nn.Linear(n_s, 1, bias=False) if use_struct else None
            nn.init.zeros_(self.m_branch.weight)
            if self.s_branch is not None:
                nn.init.zeros_(self.s_branch.weight)

        def backbone_forward(self, data):
            import torch.nn.functional as F
            from torch_geometric.nn import global_mean_pool

            x = self.atom_encoder(data.x)
            edge = self.bond_encoder(data.edge_attr)
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, data.edge_index, edge)))
            h = global_mean_pool(x, data.batch)
            return self.head(h).view(-1)

    return Net()


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase E conditional GINE arms")
    ap.add_argument("--arm", required=True, choices=("G-M", "G-STRUCT-U", "G-STRUCT-W"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0, help="debug: stop an epoch after N batches")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    t_start = time.perf_counter()
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    seed_all(args.seed, torch)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    y = molecules.label.astype(np.float64)
    bundle = load_bundle()
    n_train = int(splits["train"].size)

    use_struct = args.arm != "G-M"
    #: Phase E input selection (PROTOCOL.md §8.1): C+F
    if use_struct:
        s_np = sp.hstack([bundle["C"], bundle["F"]], format="csr")
        if args.arm == "G-STRUCT-W":
            a = np.asarray(bundle["support_weight"], dtype=np.float64)
            factor = np.concatenate([np.ones(bundle["C"].shape[1]), 1.0 / np.sqrt(a)])
            s_np = sp.csr_matrix(s_np.multiply(factor[None, :]))
        n_struct = s_np.shape[1]
    else:
        s_np = sp.csr_matrix((bundle["C"].shape[0], 0))
        n_struct = 0

    def to_sparse_tensor(mat: sp.csr_matrix, torch):
        return torch.sparse_csr_tensor(
            torch.as_tensor(mat.indptr.astype(np.int64)), torch.as_tensor(mat.indices.astype(np.int64)),
            torch.as_tensor(mat.data.astype(np.float32)),
            size=mat.shape, device=device,
        )

    M_sp = to_sparse_tensor(bundle["M"], torch)
    S_sp = to_sparse_tensor(s_np, torch) if use_struct else None

    graphs = build_graphs(molecules, torch)
    model = make_model(args.hidden, args.layers, bundle["M"].shape[1], n_struct, use_struct, torch).to(device)
    init_backbone_hash = param_hash(
        {k: v for k, v in model.state_dict().items()
         if not (k.startswith("m_branch") or k.startswith("s_branch"))},
        torch,
    )
    lam = 1.0 / (0.3 * n_train)
    branch_params = [model.m_branch.weight]
    if use_struct:
        branch_params.append(model.s_branch.weight)
    backbone_params = [p for n, p in model.named_parameters() if not (n.startswith("m_branch") or n.startswith("s_branch"))]
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "weight_decay": args.backbone_weight_decay},
            {"params": branch_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    n_pos = int(y[splits["train"]].sum())
    n_neg = n_train - n_pos
    w_pos = n_train / (2.0 * n_pos)
    w_neg = n_train / (2.0 * n_neg)
    sample_w = np.where(y > 0, w_pos, w_neg).astype(np.float32)
    sample_w /= sample_w.mean()
    weight_t = torch.as_tensor(sample_w, dtype=torch.float32, device=device)

    def loader(rows: np.ndarray, shuffle: bool):
        gen = torch.Generator().manual_seed(args.seed + 9173) if shuffle else None
        return DataLoader([graphs[int(i)] for i in rows], batch_size=args.batch_size,
                          shuffle=shuffle, generator=gen, num_workers=0)

    def m_scores() -> Any:
        return torch.sparse.mm(M_sp, model.m_branch.weight.view(-1, 1)).view(-1)

    def s_scores() -> Any:
        return torch.sparse.mm(S_sp, model.s_branch.weight.view(-1, 1)).view(-1)

    def forward(batch):
        z = model.backbone_forward(batch)
        idx = batch.graph_index.view(-1)
        z = z + m_scores()[idx]
        if use_struct:
            z = z + s_scores()[idx]
        return z

    @torch.no_grad()
    def evaluate(rows: np.ndarray) -> dict[str, Any]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        model.eval()
        logits, labels = [], []
        for batch in loader(rows, False):
            batch = batch.to(device)
            logits.append(forward(batch).detach().cpu().numpy())
            labels.append(batch.y.view(-1).cpu().numpy())
        logit = np.concatenate(logits).astype(np.float64)
        lab = np.concatenate(labels).astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
        return {
            "n": int(lab.size), "n_positive": int(lab.sum()),
            "roc_auc": float(roc_auc_score(lab, p)),
            "average_precision": float(average_precision_score(lab, p)),
            "bce": float(np.mean(np.logaddexp(0.0, logit) - lab * logit)),
            "logit": logit,
        }

    history = []
    best = {"valid_roc_auc": -1.0, "epoch": -1}
    best_payload: dict[str, Any] = {}
    loaders = {name: loader(rows, False) for name, rows in splits.items()}
    train_loader = loader(splits["train"], True)
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        total, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            if args.max_batches and step >= args.max_batches:
                break
            batch = batch.to(device)
            logit = forward(batch)
            lab = batch.y.view(-1).float()
            w = weight_t[batch.graph_index.view(-1)]
            loss = F.binary_cross_entropy_with_logits(logit, lab, weight=w, reduction="mean")
            l2 = 0.5 * lam * sum((p * p).sum() for p in branch_params)
            loss = loss + l2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params + branch_params, args.grad_clip)
            opt.step()
            total += float(loss.detach()) * int(lab.numel())
            seen += int(lab.numel())
        entry: dict[str, Any] = {"epoch": epoch, "train_loss": total / max(seen, 1)}
        epoch_eval: dict[str, Any] = {}
        for split_name in ("train", "valid", "test"):
            ev = evaluate(splits[split_name])
            epoch_eval[split_name] = ev
            entry[f"{split_name}_roc_auc"] = ev["roc_auc"]
            entry[f"{split_name}_average_precision"] = ev["average_precision"]
        if epoch_eval["valid"]["roc_auc"] > best["valid_roc_auc"]:
            best = {"valid_roc_auc": epoch_eval["valid"]["roc_auc"], "epoch": epoch}
            best_payload = {
                "epoch": epoch,
                "valid": {k: v for k, v in epoch_eval["valid"].items() if k != "logit"},
                "test": {k: v for k, v in epoch_eval["test"].items() if k != "logit"},
                "train": {k: v for k, v in epoch_eval["train"].items() if k != "logit"},
                "valid_logit": epoch_eval["valid"]["logit"],
                "test_logit": epoch_eval["test"]["logit"],
                "train_logit": epoch_eval["train"]["logit"],
            }
        entry["elapsed_sec"] = time.perf_counter() - t_epoch
        entry["cumulative_sec"] = time.perf_counter() - t_start
        history.append(entry)
        print(
            f"[{args.arm}] epoch {epoch:02d} loss={entry['train_loss']:.4f} "
            f"valid_auc={entry['valid_roc_auc']:.4f} test_auc={entry['test_roc_auc']:.4f} "
            f"train_auc={entry['train_roc_auc']:.4f} {entry['elapsed_sec']:.1f}s", flush=True,
        )
        if args.timing_only and epoch >= 2:
            break

    final = {k: v for k, v in history[-1].items()}
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "phase": "E: conditional plain-GINE verification",
        "created_utc": created,
        "arm": args.arm,
        "config": vars(args),
        "structure_input": "C+F (uniform penalty)" if args.arm == "G-STRUCT-U" else
                           ("C+F with support weights a_f on FULL columns" if args.arm == "G-STRUCT-W" else "none"),
        "n_features": {"morgan": int(bundle["M"].shape[1]), "structure": int(n_struct)},
        "recipe": {
            "model": "plain GINE, 3 layers, hidden 64, global mean pooling, scalar graph logit",
            "logit": "z = z_GINE + z_M + z_structure (linear branches zero-initialised, bias-free)",
            "optimiser": "AdamW lr=1e-3, backbone weight_decay=1e-4, branches wd=0 + explicit L2",
            "lambda": lam,
            "explicit_l2": "0.5 * lambda * ||w_branch||^2",
            "class_balance": {"pos_weight": w_pos, "neg_weight": w_neg, "normalised_mean_weight": 1.0},
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "grad_clip": args.grad_clip,
            "dropout": 0.0,
            "full_three_split_evaluation_every_epoch": True,
            "checkpoint_rule": "highest Valid ROC-AUC",
        },
        "n_train": n_train,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "initial_backbone_param_sha256": init_backbone_hash,
        "history": history,
        "best_valid": {
            "epoch": best["epoch"],
            "valid_roc_auc": best["valid_roc_auc"],
            "metrics": {k: v for k, v in best_payload.items() if k not in ("valid_logit", "test_logit", "train_logit")},
            "test_evidence_label": "EXPOSED_DEVELOPMENT_TEST",
        },
        "final": {k: v for k, v in final.items() if k != "logit"},
        "seconds_total": time.perf_counter() - t_start,
        "timing_only": bool(args.timing_only),
        "provenance": {
            "hostname": socket.gethostname(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "driver": os.environ.get("NVIDIA_DRIVER_VERSION", "unknown"),
            "git": C.git_state(),
            "bundle_sha256": C.sha256_file(BUNDLE),
            "data_audit_sha256": C.sha256_file(C.ANALYSIS / "data_audit.json") if (C.ANALYSIS / "data_audit.json").exists() else None,
        },
    }
    out = Path(args.out) if args.out else (C.RUNS / f"phaseE_{args.arm}_seed0{'_timing' if args.timing_only else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    if best_payload:
        np.savez_compressed(
            out.with_suffix(".predictions.npz"),
            valid=best_payload["valid_logit"], test=best_payload["test_logit"],
            train=best_payload["train_logit"], valid_rows=splits["valid"],
            test_rows=splits["test"], train_rows=splits["train"],
            best_epoch=np.int64(best["epoch"]),
        )
    print(f"[{args.arm}] wrote {out} total {payload['seconds_total']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
