"""
Dual-channel: GINE (attributes) ‖ KSVD s_G (structure) → fusion.

Protocol parent: ogb-molhiv-v0 (P2).

Requires torch + torch_geometric + ogb.
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
    ap.add_argument("--max-graphs", type=int, default=None, help="smoke subsample size")
    ap.add_argument("--struct-cache", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    try:
        import torch

        if not getattr(torch.load, "_ksvd_patched", False):
            _orig_load = torch.load

            def _load(*a, **k):  # type: ignore[no-untyped-def]
                k.setdefault("weights_only", False)
                return _orig_load(*a, **k)

            _load._ksvd_patched = True  # type: ignore[attr-defined]
            torch.load = _load  # type: ignore[assignment]

        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv, global_mean_pool
    except ImportError as e:
        log(f"Missing deps for dual-channel: {e}")
        log("Install: pip install -r tracks/ksvd/configs/requirements-molhiv.txt")
        sys.exit(1)

    from code.data_molhiv import check_env, load_molhiv
    from code.graph_level import GraphLevelConfig, encode_graph, learn_shared_D_graph_level

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    repo = Path(__file__).resolve().parents[3]
    root = repo / "data" / "ogb"
    root.mkdir(parents=True, exist_ok=True)

    # 1) structure bundle defines which graphs + remapped splits
    log(f"Loading structure graphs (max_graphs={args.max_graphs})...")
    bundle = load_molhiv(root=root, max_graphs=args.max_graphs, seed=args.seed)
    graphs, y_np = bundle.graphs, bundle.y
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)
    log(f"  n={len(graphs)} train/val/test={len(tr)}/{len(va)}/{len(te)} pos={y_np.mean():.4f}")
    if min(len(tr), len(va), len(te)) < 1:
        log("ERROR: empty split")
        sys.exit(1)

    # 2) s_G
    tag = f"n{len(graphs)}_pool{args.pool}_seed{args.seed}"
    cache_path = Path(args.struct_cache) if args.struct_cache else (
        _TRACK / "results" / "molhiv" / f"struct_sg_{tag}.npz"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        log(f"Loading s_G cache {cache_path}")
        S = np.load(cache_path)["S"]
    else:
        log("Computing s_G (CoverageRW-KSVD)...")
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
        log(f"  D recon={dinfo.get('recon_rel'):.4f} patches={dinfo.get('n_patches_used')}")
        rows = []
        for i, g in enumerate(graphs):
            s, _ = encode_graph(g, D, cfg, mode="coverage", seed=args.seed + i * 17)
            rows.append(s)
            if (i + 1) % 500 == 0:
                log(f"  encoded {i + 1}/{len(graphs)}")
        d = max(r.shape[0] for r in rows)
        S = np.zeros((len(rows), d), dtype=np.float32)
        for i, r in enumerate(rows):
            S[i, : r.shape[0]] = r.astype(np.float32)
        np.savez_compressed(cache_path, S=S, pool=np.array(args.pool))
        log(f"  saved {cache_path} shape={S.shape}")

    struct_dim = int(S.shape[1])
    S_t = torch.tensor(S, dtype=torch.float32, device=device)

    # 3) PyG attributes — full dataset, map via original indices stored in bundle meta
    # When subsampled, we re-load full PyG and rebuild by re-querying OGB in same order as load_molhiv
    log("Loading PyG molhiv for attributes...")
    pyg = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    evaluator = Evaluator(name="ogbg-molhiv")

    # Rebuild original index list the same way as load_molhiv
    sp = pyg.get_idx_split()
    tr0 = np.asarray(sp["train"], dtype=np.int64)
    va0 = np.asarray(sp["valid"], dtype=np.int64)
    te0 = np.asarray(sp["test"], dtype=np.int64)
    n_full = len(pyg)
    if args.max_graphs is not None and args.max_graphs < n_full:
        rng = np.random.default_rng(args.seed)
        frac = args.max_graphs / n_full
        n_tr = max(1, int(round(len(tr0) * frac)))
        n_va = max(1, int(round(len(va0) * frac)))
        n_te = max(1, int(round(len(te0) * frac)))
        total = n_tr + n_va + n_te
        while total > args.max_graphs and n_tr > 1:
            n_tr -= 1
            total -= 1
        while total < args.max_graphs and n_tr < len(tr0):
            n_tr += 1
            total += 1
        tr_o = rng.choice(tr0, size=min(n_tr, len(tr0)), replace=False)
        va_o = rng.choice(va0, size=min(n_va, len(va0)), replace=False)
        te_o = rng.choice(te0, size=min(n_te, len(te0)), replace=False)
        keep = np.unique(np.concatenate([tr_o, va_o, te_o]))
        keep.sort()
    else:
        keep = np.arange(n_full, dtype=np.int64)

    assert len(keep) == len(graphs), f"index mismatch {len(keep)} vs {len(graphs)}"

    data_list: list[Data] = []
    for new_i, old_i in enumerate(keep):
        d = pyg[int(old_i)].clone()
        d.idx = torch.tensor([new_i], dtype=torch.long)
        # ensure y is float for BCE
        d.y = d.y.view(-1).float()
        data_list.append(d)

    train_set = [data_list[i] for i in tr.tolist()]
    valid_set = [data_list[i] for i in va.tolist()]
    test_set = [data_list[i] for i in te.tolist()]
    log(f"  loaders: {len(train_set)}/{len(valid_set)}/{len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

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
            z = F.relu(self.struct_proj(s_batch))
            if self.fusion == "concat":
                return self.head(torch.cat([h, z], dim=-1))
            g = torch.sigmoid(self.gate(torch.cat([h, z], dim=-1)))
            return self.head(g * h + (1 - g) * z)

    model = DualModel(args.hidden, args.layers, struct_dim, args.fusion).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss, n_ex = 0.0, 0
        ys, preds = [], []
        for batch in loader:
            batch = batch.to(device)
            idx = batch.idx.view(-1)
            s_batch = S_t[idx]
            out = model(batch, s_batch).view(-1)
            y = batch.y.view(-1).float()
            # valid labels only (molhiv is single-task 0/1)
            mask = (y == 0) | (y == 1)
            if int(mask.sum()) == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            n = int(mask.sum())
            total_loss += float(loss.item()) * n
            n_ex += n
            ys.append(y[mask].detach().cpu())
            preds.append(out[mask].detach().cpu())
        if not ys:
            return 0.0, float("nan")
        y_all = torch.cat(ys, dim=0).numpy().reshape(-1, 1)
        p_all = torch.cat(preds, dim=0).sigmoid().numpy().reshape(-1, 1)
        auc = float(evaluator.eval({"y_true": y_all, "y_pred": p_all})["rocauc"])
        return total_loss / max(1, n_ex), auc

    best_val, best_test, best_ep = -1.0, -1.0, -1
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        _, tr_auc = run_epoch(train_loader, True)
        _, va_auc = run_epoch(valid_loader, False)
        _, te_auc = run_epoch(test_loader, False)
        history.append({"epoch": ep, "train_auc": tr_auc, "valid_auc": va_auc, "test_auc": te_auc})
        if va_auc == va_auc and va_auc > best_val:  # not nan
            best_val, best_test, best_ep = va_auc, te_auc, ep
        if ep % 5 == 0 or ep == 1:
            log(
                f"ep {ep:03d} train={tr_auc:.4f} val={va_auc:.4f} test={te_auc:.4f} "
                f"best@val→test={best_test:.4f}"
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
        "n_used": len(graphs),
        "n_train": int(len(tr)),
        "n_valid": int(len(va)),
        "n_test": int(len(te)),
        "struct_dim": struct_dim,
        "elapsed_sec": round(time.time() - t0, 2),
        "history": history,
        "env": check_env(),
        "note": "smoke if max_graphs set; not full ogb-molhiv-v0 claim",
    }
    out = (
        _TRACK
        / "results"
        / "molhiv"
        / f"dual_{args.fusion}_{tag}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Done. test@best_val={best_test:.4f} (val={best_val:.4f} @ep{best_ep})")
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
