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
    ap.add_argument("--residual-lr-scale", type=float, default=0.1)
    ap.add_argument("--residual-weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fusion", type=str, default="concat", choices=["concat", "gate", "residual", "gine_only"])
    ap.add_argument("--pool", type=str, default="attn")
    ap.add_argument("--max-graphs", type=int, default=None, help="smoke subsample size")
    ap.add_argument("--struct-cache", type=str, default=None)
    ap.add_argument("--dictionary-npz", type=str, default=None)
    ap.add_argument("--dictionary-name", type=str, default=None)
    ap.add_argument("--valid-only", action="store_true", help="do not encode/evaluate test")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output", type=str, default=None)
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
    from code.graph_level import (
        GraphLevelConfig,
        bundle_to_Y,
        encode_graph,
        encode_patch_matrix,
        learn_shared_D_graph_level,
        sample_patches_graph_level,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    repo = Path(__file__).resolve().parents[3]
    root = repo / "data" / "ogb"
    root.mkdir(parents=True, exist_ok=True)

    # 1) structure bundle defines which graphs + remapped splits
    log(f"Loading structure graphs (max_graphs={args.max_graphs})...")
    bundle = load_molhiv(
        root=root,
        max_graphs=args.max_graphs,
        seed=args.seed,
        with_features=bool(args.dictionary_npz),
    )
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
        if args.dictionary_npz:
            if not args.dictionary_name:
                raise ValueError("--dictionary-name is required with --dictionary-npz")
            archive = np.load(args.dictionary_npz)
            dictionary_names = [x.strip() for x in args.dictionary_name.split(",") if x.strip()]
            missing = [x for x in dictionary_names if x not in archive.files]
            if missing:
                raise KeyError(f"missing dictionaries {missing}; available={archive.files}")
            dictionaries = [np.asarray(archive[x], dtype=np.float64) for x in dictionary_names]
            if len({D.shape[0] for D in dictionaries}) != 1:
                raise ValueError("all dictionaries must share the same feature dimension")
            cfg = GraphLevelConfig(
                n_atoms=dictionaries[0].shape[1], T=2, T_min=1, ksvd_iter=0,
                readout_mode="pool", pool=args.pool, seed=args.seed,
                max_train_patches=4000, max_patches_per_graph=8,
                patch_feat="wl_chem_ring", normalize_patches=True,
            )
            encode_indices = np.concatenate([tr, va]) if args.valid_only else np.arange(len(graphs))
            struct_width = sum(D.shape[1] for D in dictionaries)
            S = np.zeros((len(graphs), struct_width), dtype=np.float32)
            for count, raw_i in enumerate(encode_indices, 1):
                i = int(raw_i)
                patches, _ = sample_patches_graph_level(
                    graphs[i], cfg, seed=args.seed + i * 17
                )
                Y, _ = bundle_to_Y(
                    graphs[i], patches, cfg.max_nodes, cfg.order_mode,
                    patch_feat=cfg.patch_feat, node_feat=bundle.node_feats[i],
                    edge_feat=bundle.edge_feats[i],
                )
                s = np.concatenate(
                    [encode_patch_matrix(Y, D, cfg)[0] for D in dictionaries]
                )
                S[i, : len(s)] = s.astype(np.float32)
                if count % 500 == 0:
                    log(f"  encoded {count}/{len(encode_indices)}")
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
    # Train-only scaling prevents the residual branch from seeing validation
    # distribution statistics and makes zero initialization well-conditioned.
    s_mean = S[tr].mean(axis=0, keepdims=True)
    s_std = S[tr].std(axis=0, keepdims=True)
    S = (S - s_mean) / np.maximum(s_std, 1e-6)
    S_t = torch.tensor(S, dtype=torch.float32, device=device)

    # 3) PyG attributes — full dataset, map via original indices stored in bundle meta
    # When subsampled, we re-load full PyG and rebuild by re-querying OGB in same order as load_molhiv
    log("Loading PyG molhiv for attributes...")
    pyg = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    evaluator = Evaluator(name="ogbg-molhiv")

    # Reuse the exact original indices selected by load_molhiv.  This avoids
    # silently training GINE on a different non-stratified smoke subset.
    keep = np.asarray(bundle.meta["original_indices"], dtype=np.int64)

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
    test_set = [data_list[i] for i in te.tolist()] if not args.valid_only else []
    log(f"  loaders: {len(train_set)}/{len(valid_set)}/{len(test_set)}")

    # Use an explicit generator so GINE-only and residual runs with the same
    # seed see exactly the same minibatch order.  Otherwise constructing the
    # extra residual layer consumes global RNG state and breaks paired runs.
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + 9173)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False) if test_set else None

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
            elif fusion == "residual":
                self.head = nn.Linear(hidden, 1)
                self.residual_head = nn.Linear(struct_dim, 1)
                nn.init.zeros_(self.residual_head.weight)
                nn.init.zeros_(self.residual_head.bias)
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
            if self.fusion == "residual":
                return self.head(h) + self.residual_head(s_batch)
            z = F.relu(self.struct_proj(s_batch))
            if self.fusion == "concat":
                return self.head(torch.cat([h, z], dim=-1))
            g = torch.sigmoid(self.gate(torch.cat([h, z], dim=-1)))
            return self.head(g * h + (1 - g) * z)

    model = DualModel(args.hidden, args.layers, struct_dim, args.fusion).to(device)
    if args.fusion == "residual":
        residual_params = list(model.residual_head.parameters())
        residual_ids = {id(p) for p in residual_params}
        base_params = [p for p in model.parameters() if id(p) not in residual_ids]
        opt = torch.optim.Adam(
            [
                {"params": base_params, "lr": args.lr},
                {
                    "params": residual_params,
                    "lr": args.lr * args.residual_lr_scale,
                    "weight_decay": args.residual_weight_decay,
                },
            ]
        )
    else:
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
        te_auc = float("nan") if test_loader is None else run_epoch(test_loader, False)[1]
        row = {"epoch": ep, "train_auc": tr_auc, "valid_auc": va_auc}
        if test_loader is not None:
            row["test_auc"] = te_auc
        history.append(row)
        if va_auc == va_auc and va_auc > best_val:  # not nan
            best_val, best_test, best_ep = va_auc, te_auc, ep
        if ep % 5 == 0 or ep == 1:
            suffix = "" if test_loader is None else f" test={te_auc:.4f} best@val→test={best_test:.4f}"
            log(f"ep {ep:03d} train={tr_auc:.4f} val={va_auc:.4f}{suffix}")

    results = {
        "protocol_id": "ogb-molhiv-v0",
        "fusion": args.fusion,
        "pool": args.pool,
        "best_epoch": best_ep,
        "best_valid_auc": best_val,
        "test_auc_at_best_val": None if args.valid_only else best_test,
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
        "test_policy": "test was not structurally encoded or evaluated" if args.valid_only else "test evaluated at each epoch",
        "dictionary_npz": args.dictionary_npz,
        "dictionary_name": args.dictionary_name,
        "residual_lr_scale": args.residual_lr_scale,
        "residual_weight_decay": args.residual_weight_decay,
        "elapsed_sec": round(time.time() - t0, 2),
        "history": history,
        "env": check_env(),
        "note": "smoke if max_graphs set; not full ogb-molhiv-v0 claim",
    }
    out = Path(args.output) if args.output else (
        _TRACK / "results" / "molhiv" / f"dual_{args.fusion}_{tag}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if args.valid_only:
        log(f"Done. best_val={best_val:.4f} @ep{best_ep}; test untouched")
    else:
        log(f"Done. test@best_val={best_test:.4f} (val={best_val:.4f} @ep{best_ep})")
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
