"""Strict inner-validation protocol for GINE vs fixed-KSVD residual on MolHIV.

Protocol per model seed:
1. create a fixed stratified split *inside* the official training subset;
2. choose the epoch using only this inner validation split;
3. reset all model/loader RNG state and retrain on all official-train graphs
   for exactly the selected number of epochs;
4. evaluate the official validation split exactly once.

Official test graphs are neither structurally encoded nor loaded into PyG.
Run GINE-only and residual in separate commands with identical arguments/seed;
the base GINE initialization and minibatch sequence are then exactly paired.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30, help="maximum inner-selection epochs")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--residual-lr-scale", type=float, default=0.1)
    ap.add_argument("--residual-weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0, help="neural initialization/training seed")
    ap.add_argument("--data-seed", type=int, default=0, help="fixed graph-subset seed")
    ap.add_argument("--struct-seed", type=int, default=0, help="fixed patch-sampling seed")
    ap.add_argument("--inner-split-seed", type=int, default=1729)
    ap.add_argument("--inner-valid-fraction", type=float, default=0.15)
    ap.add_argument("--fusion", choices=("gine_only", "residual"), required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--dictionary-npz", type=str, required=True)
    ap.add_argument("--dictionary-name", type=str, default="ksvd_seed0")
    ap.add_argument("--struct-cache", type=str, default=None)
    ap.add_argument("--struct-readout", choices=("max", "moments", "recon", "rich"), default="max")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    try:
        import torch

        if not getattr(torch.load, "_ksvd_patched", False):
            original_load = torch.load

            def patched_load(*a, **kw):  # type: ignore[no-untyped-def]
                kw.setdefault("weights_only", False)
                return original_load(*a, **kw)

            patched_load._ksvd_patched = True  # type: ignore[attr-defined]
            torch.load = patched_load  # type: ignore[assignment]

        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv, global_mean_pool
    except ImportError as exc:
        raise RuntimeError(f"missing GINE dependencies: {exc}") from exc

    from code.data_molhiv import check_env, load_molhiv
    from code.graph_level import (
        GraphLevelConfig,
        bundle_to_Y,
        sample_patches_graph_level,
        sparse_code_patch_matrix,
        sparse_code_readouts,
    )

    if args.max_graphs <= 0:
        max_graphs = None
    else:
        max_graphs = args.max_graphs
    if not 0.05 <= args.inner_valid_fraction <= 0.4:
        raise ValueError("--inner-valid-fraction must be in [0.05, 0.4]")

    device = torch.device(args.device)
    repo = Path(__file__).resolve().parents[3]
    root = repo / "data" / "ogb"
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Dataset composition is independent of the neural seed.
    bundle = load_molhiv(
        root=root,
        max_graphs=max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    graphs, y_np = bundle.graphs, bundle.y
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.inner_valid_fraction,
        random_state=args.inner_split_seed,
    )
    inner_train_pos, inner_valid_pos = next(splitter.split(tr, y_np[tr]))
    inner_tr = tr[inner_train_pos]
    inner_va = tr[inner_valid_pos]
    log(
        f"n={len(graphs)} official train/valid={len(tr)}/{len(va)}; "
        f"inner train/valid={len(inner_tr)}/{len(inner_va)}"
    )

    # Fixed KSVD graph channel.  Only official train + valid are encoded.
    dictionary_path = Path(args.dictionary_npz)
    archive = np.load(dictionary_path)
    if args.dictionary_name not in archive.files:
        raise KeyError(f"{args.dictionary_name!r} not in {archive.files}")
    D = np.asarray(archive[args.dictionary_name], dtype=np.float64)
    cache_path = Path(args.struct_cache) if args.struct_cache else (
        _TRACK
        / "results"
        / "molhiv"
        / f"struct_inner_n{len(graphs)}_data{args.data_seed}_struct{args.struct_seed}_{args.dictionary_name}_{args.struct_readout}.npz"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_width = {
        "max": D.shape[1],
        "moments": 3 * D.shape[1],
        "recon": 8,
        "rich": 10 * D.shape[1] + 8,
    }[args.struct_readout]
    if cache_path.exists():
        with np.load(cache_path) as saved:
            S = np.asarray(saved["S"], dtype=np.float32)
            if S.shape != (len(graphs), expected_width):
                raise ValueError(
                    f"cache shape mismatch: {S.shape} vs {(len(graphs), expected_width)}"
                )
            if "original_indices" in saved.files:
                cached_indices = np.asarray(saved["original_indices"], dtype=np.int64)
                expected_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
                if not np.array_equal(cached_indices, expected_indices):
                    raise ValueError("cache original_indices do not match the data subset")
            if "dictionary_name" in saved.files and str(saved["dictionary_name"].item()) != args.dictionary_name:
                raise ValueError("cache dictionary_name does not match --dictionary-name")
            if "dictionary_npz" in saved.files:
                cached_dictionary = Path(str(saved["dictionary_npz"].item())).resolve()
                if cached_dictionary != dictionary_path.resolve():
                    raise ValueError("cache dictionary_npz does not match --dictionary-npz")
            readout_key = "struct_readout" if "struct_readout" in saved.files else "pool"
            if readout_key in saved.files and str(saved[readout_key].item()) != args.struct_readout:
                raise ValueError("cache readout does not match --struct-readout")
            if "struct_seed" in saved.files and int(saved["struct_seed"].item()) != args.struct_seed:
                raise ValueError("cache struct_seed does not match --struct-seed")
        # The cache has one row per selected molecule, but official-test rows
        # must remain untouched.  This turns the stated no-test-encoding policy
        # into an executable invariant rather than relying on the filename.
        if te.size and np.any(S[te] != 0):
            raise ValueError("structure cache contains encoded official-test rows")
        log(f"loaded validated structure cache {cache_path} shape={S.shape}")
    else:
        cfg = GraphLevelConfig(
            n_atoms=D.shape[1],
            T=2,
            T_min=1,
            ksvd_iter=0,
            readout_mode="pool",
            pool="max",
            seed=args.struct_seed,
            max_train_patches=4000,
            max_patches_per_graph=8,
            patch_feat="wl_chem_ring",
            normalize_patches=True,
        )
        encode_indices = np.concatenate([tr, va])
        S = np.zeros((len(graphs), expected_width), dtype=np.float32)
        for count, raw_i in enumerate(encode_indices, 1):
            i = int(raw_i)
            patches, _ = sample_patches_graph_level(
                graphs[i], cfg, seed=args.struct_seed + i * 13
            )
            Y, _ = bundle_to_Y(
                graphs[i],
                patches,
                cfg.max_nodes,
                cfg.order_mode,
                patch_feat=cfg.patch_feat,
                node_feat=bundle.node_feats[i],
                edge_feat=bundle.edge_feats[i],
            )
            Yn, X = sparse_code_patch_matrix(Y, D, cfg)
            denom = np.maximum(np.linalg.norm(Yn, axis=0), 1e-12)
            patch_errors = np.linalg.norm(Yn - D @ X, axis=0) / denom
            S[i] = sparse_code_readouts(X, patch_errors)[args.struct_readout].astype(np.float32)
            if count % 500 == 0 or count == len(encode_indices):
                log(f"encoded structure {count}/{len(encode_indices)}")
        np.savez_compressed(
            cache_path,
            S=S,
            original_indices=np.asarray(bundle.meta["original_indices"], dtype=np.int64),
            dictionary_name=np.array(args.dictionary_name),
            dictionary_npz=np.array(str(dictionary_path)),
            struct_readout=np.array(args.struct_readout),
            struct_seed=np.array(args.struct_seed),
        )
        log(f"saved structure cache {cache_path}")
    struct_dim = int(S.shape[1])

    # Index only selected train/valid PyG graphs into model data objects.
    pyg = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    evaluator = Evaluator(name="ogbg-molhiv")
    keep = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    needed = set(np.concatenate([tr, va]).tolist())
    data_by_idx: dict[int, Data] = {}
    for new_i in sorted(needed):
        data = pyg[int(keep[new_i])].clone()
        data.idx = torch.tensor([new_i], dtype=torch.long)
        data.y = data.y.view(-1).float()
        data_by_idx[new_i] = data

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
                x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
            return global_mean_pool(x, data.batch)

    class DualModel(nn.Module):
        def __init__(self):
            super().__init__()
            # These two modules are constructed first and identically for both
            # fusions, so their initial tensors are exactly paired by seed.
            self.gine = GINEStack(args.hidden, args.layers)
            self.head = nn.Linear(args.hidden, 1)
            self.fusion = args.fusion
            if args.fusion == "residual":
                self.residual_head = nn.Linear(struct_dim, 1)
                nn.init.zeros_(self.residual_head.weight)
                nn.init.zeros_(self.residual_head.bias)

        def forward(self, data, s_batch):
            out = self.head(self.gine(data))
            if self.fusion == "residual":
                out = out + self.residual_head(s_batch)
            return out

    def make_model_optimizer(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = DualModel().to(device)
        if args.fusion == "residual":
            residual_params = list(model.residual_head.parameters())
            residual_ids = {id(p) for p in residual_params}
            base_params = [p for p in model.parameters() if id(p) not in residual_ids]
            optimizer = torch.optim.Adam(
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
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return model, optimizer

    def make_loader(indices: np.ndarray, shuffle: bool, phase_offset: int):
        generator = None
        if shuffle:
            generator = torch.Generator()
            # Identical across fusions, deterministic within each phase.
            generator.manual_seed(args.seed + 9173 + phase_offset)
        return DataLoader(
            [data_by_idx[int(i)] for i in indices],
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def normalize_structure(fit_indices: np.ndarray) -> torch.Tensor:
        mean = S[fit_indices].mean(axis=0, keepdims=True)
        std = S[fit_indices].std(axis=0, keepdims=True)
        normalized = (S - mean) / np.maximum(std, 1e-6)
        return torch.tensor(normalized, dtype=torch.float32, device=device)

    def train_epoch(model, optimizer, loader, S_tensor):
        model.train()
        total, n_seen = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, S_tensor[batch.idx.view(-1)]).view(-1)
            labels = batch.y.view(-1).float()
            mask = (labels == 0) | (labels == 1)
            if not bool(mask.any()):
                continue
            loss = F.binary_cross_entropy_with_logits(pred[mask], labels[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = int(mask.sum())
            total += float(loss.item()) * n
            n_seen += n
        return total / max(n_seen, 1)

    def evaluate(model, loader, S_tensor) -> float:
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch, S_tensor[batch.idx.view(-1)]).view(-1)
                labels = batch.y.view(-1).float()
                mask = (labels == 0) | (labels == 1)
                ys.append(labels[mask].cpu())
                preds.append(pred[mask].sigmoid().cpu())
        y_true = torch.cat(ys).numpy().reshape(-1, 1)
        y_pred = torch.cat(preds).numpy().reshape(-1, 1)
        return float(evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"])

    # Phase 1: epoch selection on official-train only.
    selection_S = normalize_structure(inner_tr)
    selection_train_loader = make_loader(inner_tr, True, phase_offset=0)
    selection_valid_loader = make_loader(inner_va, False, phase_offset=0)
    model, optimizer = make_model_optimizer(args.seed)
    history: list[dict[str, Any]] = []
    best_epoch, best_inner_auc = 1, -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer, selection_train_loader, selection_S)
        inner_auc = evaluate(model, selection_valid_loader, selection_S)
        history.append({"epoch": epoch, "train_loss": loss, "inner_valid_auc": inner_auc})
        if inner_auc > best_inner_auc:
            best_epoch, best_inner_auc = epoch, inner_auc
        if epoch == 1 or epoch % 5 == 0:
            log(
                f"select ep={epoch:02d} loss={loss:.4f} inner={inner_auc:.4f} "
                f"best={best_inner_auc:.4f}@{best_epoch}"
            )

    # Phase 2: exact-seed reset, full official-train retraining, one official-valid evaluation.
    final_S = normalize_structure(tr)
    final_train_loader = make_loader(tr, True, phase_offset=100_000)
    official_valid_loader = make_loader(va, False, phase_offset=100_000)
    model, optimizer = make_model_optimizer(args.seed)
    final_losses = []
    for epoch in range(1, best_epoch + 1):
        loss = train_epoch(model, optimizer, final_train_loader, final_S)
        final_losses.append(loss)
        if epoch == 1 or epoch == best_epoch or epoch % 5 == 0:
            log(f"retrain ep={epoch:02d}/{best_epoch} loss={loss:.4f}")
    official_valid_auc = evaluate(model, official_valid_loader, final_S)
    log(f"official valid (single evaluation)={official_valid_auc:.6f}")

    result = {
        "protocol_id": "molhiv-ksvd-inner-checkpoint-v1",
        "fusion": args.fusion,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "struct_seed": args.struct_seed,
        "inner_split_seed": args.inner_split_seed,
        "inner_valid_fraction": args.inner_valid_fraction,
        "max_selection_epochs": args.epochs,
        "selected_epoch": best_epoch,
        "best_inner_valid_auc": best_inner_auc,
        "official_valid_auc": official_valid_auc,
        "official_valid_evaluations": 1,
        "test_policy": "official test graphs were not structurally encoded, indexed into model data objects, or evaluated",
        "n_used": len(graphs),
        "n_official_train": int(len(tr)),
        "n_official_valid": int(len(va)),
        "n_inner_train": int(len(inner_tr)),
        "n_inner_valid": int(len(inner_va)),
        "n_inner_train_pos": int(y_np[inner_tr].sum()),
        "n_inner_valid_pos": int(y_np[inner_va].sum()),
        "hidden": args.hidden,
        "layers": args.layers,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "residual_lr_scale": args.residual_lr_scale,
        "residual_weight_decay": args.residual_weight_decay,
        "dictionary_npz": str(dictionary_path),
        "dictionary_name": args.dictionary_name,
        "struct_cache": str(cache_path),
        "struct_dim": struct_dim,
        "struct_readout": args.struct_readout,
        "selection_history": history,
        "final_train_losses": final_losses,
        "elapsed_sec": time.time() - t0,
        "env": check_env(),
        "paired_invariants": {
            "fixed_data_subset_across_model_seeds": True,
            "same_base_initialization_for_same_seed_across_fusions": True,
            "same_minibatch_order_for_same_seed_across_fusions": True,
            "fixed_structure_features_across_model_seeds": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
