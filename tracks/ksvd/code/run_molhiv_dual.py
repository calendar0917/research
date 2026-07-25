"""
Dual-channel skeleton: GINE (attributes) ‖ KSVD s_G (structure) → fusion.

Protocol parent: ogb-molhiv-v0 (P2).

Requires torch + torch_geometric + ogb.
If missing, exits with install hint. Structure branch reuses graph_level.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fusion", type=str, default="concat", choices=["concat", "gate", "gine_only"])
    ap.add_argument("--pool", type=str, default="attn")
    ap.add_argument("--max-graphs", type=int, default=None, help="smoke limit")
    ap.add_argument("--struct-cache", type=str, default=None, help="optional .npz of s_G")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv, global_mean_pool
    except ImportError as e:
        log(f"Missing deps for dual-channel: {e}")
        log("Install: pip install -r tracks/ksvd/configs/requirements-molhiv.txt")
        log("Plus PyG extensions if needed: https://pytorch-geometric.readthedocs.io/")
        sys.exit(1)

    from code.data_molhiv import check_env, load_molhiv
    from code.graph_level import GraphLevelConfig, encode_graph, learn_shared_D_graph_level

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    repo = Path(__file__).resolve().parents[3]
    root = repo / "data" / "ogb"
    root.mkdir(parents=True, exist_ok=True)

    log("Loading PyG ogbg-molhiv...")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name="ogbg-molhiv")

    # optional smoke: subset by filtering indices
    if args.max_graphs is not None:
        n = min(args.max_graphs, len(dataset))
        for k in split_idx:
            split_idx[k] = split_idx[k][split_idx[k] < n]
        dataset = dataset[:n]

    # --- structure channel s_G (numpy, precompute) ---
    cache_path = Path(args.struct_cache) if args.struct_cache else (
        _TRACK / "results" / "molhiv" / f"struct_sg_seed{args.seed}.npz"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and args.max_graphs is None:
        log(f"Loading s_G cache {cache_path}")
        blob = np.load(cache_path)
        S = blob["S"]
    else:
        log("Computing structure s_G via CoverageRW-KSVD...")
        bundle = load_molhiv(root=root, max_graphs=args.max_graphs)
        graphs = bundle.graphs
        tr = split_idx["train"].numpy() if hasattr(split_idx["train"], "numpy") else np.asarray(split_idx["train"])
        cfg = GraphLevelConfig(
            n_atoms=16,
            T=3,
            ksvd_iter=6,
            readout_mode="pool",
            pool=args.pool,
            seed=args.seed,
            max_train_patches=8000,
        )
        D, dinfo = learn_shared_D_graph_level(graphs, tr, cfg, mode="coverage")
        log(f"  D recon={dinfo.get('recon_rel'):.4f}")
        rows = []
        for i, g in enumerate(graphs):
            s, _ = encode_graph(g, D, cfg, mode="coverage", seed=args.seed + i * 17)
            rows.append(s)
            if (i + 1) % 500 == 0:
                log(f"  encoded {i+1}/{len(graphs)}")
        d = max(r.shape[0] for r in rows)
        S = np.zeros((len(rows), d), dtype=np.float32)
        for i, r in enumerate(rows):
            S[i, : r.shape[0]] = r.astype(np.float32)
        np.savez_compressed(cache_path, S=S, pool=args.pool)
        log(f"  saved {cache_path} shape={S.shape}")

    struct_dim = S.shape[1]

    class GINEStack(nn.Module):
        def __init__(self, hidden: int, layers: int):
            super().__init__()
            self.atom_encoder = AtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.hidden = hidden

        def forward(self, data):
            x = self.atom_encoder(data.x)
            edge_attr = self.bond_encoder(data.edge_attr)
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, data.edge_index, edge_attr)
                x = bn(x)
                x = F.relu(x)
            return global_mean_pool(x, data.batch)

    class DualModel(nn.Module):
        def __init__(self, hidden, layers, struct_dim, fusion):
            super().__init__()
            self.gine = GINEStack(hidden, layers)
            self.fusion = fusion
            self.struct_dim = struct_dim
            if fusion == "gine_only":
                self.head = nn.Linear(hidden, 1)
            elif fusion == "concat":
                self.struct_proj = nn.Linear(struct_dim, hidden)
                self.head = nn.Linear(hidden * 2, 1)
            elif fusion == "gate":
                self.struct_proj = nn.Linear(struct_dim, hidden)
                self.gate = nn.Linear(hidden * 2, hidden)
                self.head = nn.Linear(hidden, 1)
            else:
                raise ValueError(fusion)

        def forward(self, data, s_batch: torch.Tensor):
            h = self.gine(data)
            if self.fusion == "gine_only":
                return self.head(h)
            z = self.struct_proj(s_batch)
            if self.fusion == "concat":
                return self.head(torch.cat([h, z], dim=-1))
            g = torch.sigmoid(self.gate(torch.cat([h, z], dim=-1)))
            return self.head(g * h + (1 - g) * z)

    # attach struct vector into batch via custom collate: store graph index
    # PyG Data has .idx if we set it
    for i in range(len(dataset)):
        dataset[i].idx = i  # may fail if Dataset is not mutable list-like

    # Safer: map via data loader and range
    # Re-wrap as list of Data with idx
    data_list = []
    for i in range(len(dataset)):
        d = dataset[i]
        d.idx = torch.tensor([i], dtype=torch.long)
        data_list.append(d)

    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx = split_idx["test"]
    if not torch.is_tensor(train_idx):
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        valid_idx = torch.tensor(valid_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)

    train_set = [data_list[i] for i in train_idx.tolist()]
    valid_set = [data_list[i] for i in valid_idx.tolist()]
    test_set = [data_list[i] for i in test_idx.tolist()]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = DualModel(args.hidden, args.layers, struct_dim, args.fusion).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    S_t = torch.tensor(S, dtype=torch.float32, device=device)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        ys, preds = [], []
        for batch in loader:
            batch = batch.to(device)
            # batch.idx may be (B,) after batching
            idx = batch.idx.view(-1)
            s_batch = S_t[idx]
            out = model(batch, s_batch).view(-1)
            y = batch.y.view(-1).float()
            # molhiv y may be -1 for missing; mask
            mask = y == y  # all finite
            if mask.sum() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += float(loss.item()) * int(mask.sum())
            ys.append(y.detach().cpu())
            preds.append(out.detach().cpu())
        y_all = torch.cat(ys, dim=0).numpy().reshape(-1, 1)
        p_all = torch.cat(preds, dim=0).sigmoid().numpy().reshape(-1, 1)
        # filter nan labels
        m = ~np.isnan(y_all.reshape(-1))
        input_dict = {"y_true": y_all[m], "y_pred": p_all[m]}
        auc = evaluator.eval(input_dict)["rocauc"]
        return total_loss / max(1, len(y_all)), float(auc)

    best_val, best_test, best_ep = -1.0, -1.0, -1
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_auc = run_epoch(train_loader, True)
        va_loss, va_auc = run_epoch(valid_loader, False)
        te_loss, te_auc = run_epoch(test_loader, False)
        history.append(
            {"epoch": ep, "train_auc": tr_auc, "valid_auc": va_auc, "test_auc": te_auc}
        )
        if va_auc > best_val:
            best_val, best_test, best_ep = va_auc, te_auc, ep
        if ep % 5 == 0 or ep == 1:
            log(
                f"ep {ep:03d} train={tr_auc:.4f} val={va_auc:.4f} test={te_auc:.4f} "
                f"best@val → test={best_test:.4f}"
            )

    results = {
        "protocol_id": "ogb-molhiv-v0",
        "fusion": args.fusion,
        "pool": args.pool,
        "best_epoch": best_ep,
        "best_valid_auc": best_val,
        "test_auc_at_best_val": best_test,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "layers": args.layers,
        "seed": args.seed,
        "max_graphs": args.max_graphs,
        "struct_dim": struct_dim,
        "elapsed_sec": round(time.time() - t0, 2),
        "history": history,
        "env": check_env(),
    }
    out = _TRACK / "results" / "molhiv" / f"dual_{args.fusion}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Done. test@best_val={best_test:.4f} (val={best_val:.4f} @ep{best_ep})")
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
