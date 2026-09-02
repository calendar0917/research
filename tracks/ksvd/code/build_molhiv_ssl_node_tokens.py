"""Build fold-only self-supervised local-GNN latent dictionaries for MolHIV.

The local encoder is trained without HIV labels to reconstruct masked atom
categories from a two-hop molecular context.  At encoding time every node is
masked exactly once in one of K deterministic groups, so its latent excludes
its own atom features while retaining mostly unmasked neighbors.  KSVD/PCA/
random dictionaries are then fit only on fold-inner context latents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.extmath import randomized_svd

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import _patch_torch_load_weights_only, load_molhiv
from code.ksvd import _omp, ksvd
from code.molhiv_node_tokens import graph_node_offsets


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _state_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(np.asarray(tensor.detach().cpu()).tobytes())
    return h.hexdigest()


def _random_patch_dictionary(Y: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    if Y.shape[1] < n_atoms:
        raise ValueError("fewer latent patches than dictionary atoms")
    rng = np.random.default_rng(seed)
    selected = rng.choice(Y.shape[1], size=n_atoms, replace=False)
    D = Y[:, selected].copy()
    norms = np.linalg.norm(D, axis=0)
    nonzero = norms > 1e-12
    D[:, nonzero] /= norms[nonzero]
    if not np.all(nonzero):
        raise RuntimeError("zero latent selected for random-patch dictionary")
    return D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--ssl-seed", type=int, default=271828)
    ap.add_argument("--ssl-hidden", type=int, default=64)
    ap.add_argument("--ssl-layers", type=int, default=2)
    ap.add_argument("--ssl-epochs", type=int, default=8)
    ap.add_argument("--ssl-batch-size", type=int, default=128)
    ap.add_argument("--ssl-lr", type=float, default=1e-3)
    ap.add_argument("--ssl-weight-decay", type=float, default=1e-5)
    ap.add_argument("--ssl-mask-prob", type=float, default=0.30)
    ap.add_argument("--context-mask-groups", type=int, default=4)
    ap.add_argument("--max-train-latents", type=int, default=6000)
    ap.add_argument("--n-atoms", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--ksvd-iter", type=int, default=3)
    ap.add_argument("--families", default="ksvd,pca,random_patch")
    ap.add_argument("--fit-indices-cache", required=True)
    ap.add_argument("--fit-fold", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.ssl_hidden <= 0 or args.ssl_layers <= 0 or args.ssl_epochs <= 0:
        raise ValueError("SSL hidden/layers/epochs must be positive")
    if args.ssl_batch_size <= 0 or args.max_train_latents <= 0:
        raise ValueError("batch size and latent reservoir must be positive")
    if not 0.0 < args.ssl_mask_prob < 1.0:
        raise ValueError("--ssl-mask-prob must be in (0,1)")
    if args.context_mask_groups < 2:
        raise ValueError("--context-mask-groups must be at least 2")
    if args.n_atoms <= 0 or not 1 <= args.sparsity <= args.n_atoms:
        raise ValueError("invalid dictionary size/sparsity")

    families = _csv(args.families)
    unknown = set(families) - {"ksvd", "pca", "random_patch"}
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")

    _patch_torch_load_weights_only()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred import PygGraphPropPredDataset
        from ogb.utils.features import get_atom_feature_dims
        from ogb.graphproppred.mol_encoder import BondEncoder
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv
    except ImportError as exc:
        raise RuntimeError(f"missing SSL GNN dependencies: {exc}") from exc

    torch.manual_seed(args.ssl_seed)
    np.random.seed(args.ssl_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.ssl_seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    t0 = time.time()
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=False)
    graphs = bundle.graphs
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)
    with np.load(args.fit_indices_cache, allow_pickle=False) as folds:
        fit_key = f"fold_{args.fit_fold}_train_indices"
        valid_key = f"fold_{args.fit_fold}_valid_indices"
        if fit_key not in folds.files or valid_key not in folds.files:
            raise KeyError(f"fold {args.fit_fold} missing from split cache")
        fit_indices = np.asarray(folds[fit_key], dtype=np.int64)
        heldout_indices = np.asarray(folds[valid_key], dtype=np.int64)
        if "official_train_indices" in folds.files and not np.array_equal(
            np.asarray(folds["official_train_indices"], dtype=np.int64), tr
        ):
            raise ValueError("fold cache official train does not match dataset")
    tr_set = set(tr.tolist())
    if not set(fit_indices.tolist()).issubset(tr_set):
        raise ValueError("SSL fit includes non-official-train graph")
    if set(fit_indices.tolist()) & set(heldout_indices.tolist()):
        raise ValueError("SSL train and held-out scaffold folds overlap")

    root = bundle.meta["root"]
    pyg = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    keep = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    offsets = graph_node_offsets(graphs)
    total_nodes = int(offsets[-1])

    data_by_idx: dict[int, Any] = {}
    for count, local_i in enumerate(tr.tolist(), 1):
        original_i = int(keep[local_i])
        data = pyg[original_i].clone()
        # Make HIV labels physically unavailable to the SSL DataLoader/model.
        if "y" in data:
            del data.y
        expected_nodes = int(offsets[local_i + 1] - offsets[local_i])
        if int(data.num_nodes) != expected_nodes:
            raise ValueError(f"node count mismatch for graph {local_i}")
        data.graph_id = torch.tensor([local_i], dtype=torch.long)
        local_nodes = torch.arange(expected_nodes, dtype=torch.long)
        # A graph-specific offset avoids using one global node-index coloring.
        group_offset = int(
            hashlib.sha256(f"{args.ssl_seed}:{original_i}".encode()).digest()[0]
        ) % args.context_mask_groups
        data.context_group = (local_nodes + group_offset) % args.context_mask_groups
        data_by_idx[local_i] = data
        if count % 1000 == 0 or count == len(tr):
            print(f"loaded label-free official-train graphs {count}/{len(tr)}", flush=True)

    atom_dims = list(get_atom_feature_dims())

    class MaskableAtomEncoder(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.embeddings = nn.ModuleList([
                nn.Embedding(int(dim) + 1, hidden) for dim in atom_dims
            ])
            for emb in self.embeddings:
                nn.init.xavier_uniform_(emb.weight.data)

        def forward(self, x, mask):
            out = 0.0
            for field, (dim, emb) in enumerate(zip(atom_dims, self.embeddings)):
                values = x[:, field]
                if mask is not None:
                    values = values.clone()
                    values[mask] = int(dim)
                out = out + emb(values)
            return out

    class MaskedContextGINE(nn.Module):
        def __init__(self, hidden: int, layers: int):
            super().__init__()
            self.atom_encoder = MaskableAtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.heads = nn.ModuleList([
                nn.Linear(hidden, int(dim)) for dim in atom_dims
            ])

        def forward(self, data, mask, predict: bool = True):
            x = self.atom_encoder(data.x, mask)
            edge_attr = self.bond_encoder(data.edge_attr)
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
            logits = [head(x) for head in self.heads] if predict else None
            return x, logits

    device = torch.device(args.device)
    model = MaskedContextGINE(args.ssl_hidden, args.ssl_layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )
    train_data = [data_by_idx[int(i)] for i in fit_indices]
    loader_generator = torch.Generator().manual_seed(args.ssl_seed + 1)
    train_loader = DataLoader(
        train_data,
        batch_size=args.ssl_batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    mask_generator = torch.Generator(device=device).manual_seed(args.ssl_seed + 2)
    loss_curve: list[float] = []
    accuracy_curve: list[list[float]] = []
    for epoch in range(1, args.ssl_epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0
        correct = np.zeros(len(atom_dims), dtype=np.int64)
        total = np.zeros(len(atom_dims), dtype=np.int64)
        for data in train_loader:
            data = data.to(device)
            mask = torch.rand(
                data.x.shape[0], generator=mask_generator, device=device
            ) < args.ssl_mask_prob
            if not bool(mask.any()):
                mask[0] = True
            _, logits = model(data, mask, predict=True)
            assert logits is not None
            losses = [
                F.cross_entropy(logit[mask], data.x[mask, field])
                for field, logit in enumerate(logits)
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
            with torch.no_grad():
                for field, logit in enumerate(logits):
                    pred = logit[mask].argmax(dim=-1)
                    target = data.x[mask, field]
                    correct[field] += int((pred == target).sum())
                    total[field] += int(target.numel())
        mean_loss = loss_sum / max(batches, 1)
        accuracies = (correct / np.maximum(total, 1)).tolist()
        loss_curve.append(mean_loss)
        accuracy_curve.append([float(x) for x in accuracies])
        print(
            f"ssl epoch={epoch:02d} loss={mean_loss:.5f} "
            f"atom0_acc={accuracies[0]:.3f} mean_field_acc={np.mean(accuracies):.3f}",
            flush=True,
        )

    # Deterministic context-only encoding: in K passes each node is masked once,
    # and only its masked-pass hidden state is retained.
    model.eval()
    latent = np.zeros((total_nodes, args.ssl_hidden), dtype=np.float32)
    encoded_mask = np.zeros(total_nodes, dtype=bool)
    encode_data = [data_by_idx[int(i)] for i in tr]
    encode_loader = DataLoader(
        encode_data, batch_size=args.ssl_batch_size, shuffle=False
    )
    with torch.no_grad():
        for batch_i, data in enumerate(encode_loader, 1):
            data = data.to(device)
            batch_latent = torch.zeros(
                (data.x.shape[0], args.ssl_hidden), dtype=torch.float32, device=device
            )
            assigned = torch.zeros(data.x.shape[0], dtype=torch.bool, device=device)
            for group in range(args.context_mask_groups):
                mask = data.context_group == group
                h, _ = model(data, mask, predict=False)
                batch_latent[mask] = h[mask]
                assigned |= mask
            if not bool(assigned.all()):
                raise AssertionError("some nodes lack a masked-context latent")
            graph_ids = data.graph_id.detach().cpu().numpy().astype(np.int64)
            ptr = data.ptr.detach().cpu().numpy().astype(np.int64)
            batch_np = batch_latent.detach().cpu().numpy()
            for j, local_i in enumerate(graph_ids.tolist()):
                lo, hi = int(offsets[local_i]), int(offsets[local_i + 1])
                part = batch_np[int(ptr[j]):int(ptr[j + 1])]
                if part.shape[0] != hi - lo:
                    raise AssertionError("batched latent/node offset mismatch")
                latent[lo:hi] = part
                encoded_mask[lo:hi] = True
            if batch_i % 10 == 0 or batch_i == len(encode_loader):
                print(
                    f"context-encoded batches {batch_i}/{len(encode_loader)}",
                    flush=True,
                )

    # No official-valid/test graph was loaded into data_by_idx or encoded.
    for local_i in np.concatenate([va, te]):
        lo, hi = int(offsets[int(local_i)]), int(offsets[int(local_i) + 1])
        if bool(encoded_mask[lo:hi].any()) or np.any(latent[lo:hi] != 0):
            raise AssertionError("official valid/test latent was encoded")

    fit_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in fit_indices
    ])
    if fit_rows.size < args.max_train_latents:
        raise RuntimeError("fewer fold-inner node latents than requested reservoir")
    reservoir_rng = np.random.default_rng(args.dict_seed)
    selected_rows = reservoir_rng.choice(
        fit_rows, size=args.max_train_latents, replace=False
    )
    latent_mean = latent[selected_rows].astype(np.float64).mean(axis=0)

    def transform_rows(rows: np.ndarray) -> np.ndarray:
        z = rows.astype(np.float64) - latent_mean[None, :]
        norms = np.linalg.norm(z, axis=1)
        nonzero = norms > 1e-12
        z[nonzero] /= norms[nonzero, None]
        z[~nonzero] = 0.0
        return z

    reservoir = transform_rows(latent[selected_rows])
    Ydict = reservoir.T
    covariance = reservoir.T @ reservoir / max(reservoir.shape[0], 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    effective_rank = float(
        np.exp(-np.sum(
            (eigenvalues / max(eigenvalues.sum(), 1e-12))
            * np.log(np.maximum(eigenvalues / max(eigenvalues.sum(), 1e-12), 1e-12))
        ))
    )
    print(
        f"latent reservoir {Ydict.shape}; effective_rank={effective_rank:.2f}",
        flush=True,
    )

    dictionaries: dict[str, np.ndarray] = {}
    dictionary_info: dict[str, Any] = {}
    if "random_patch" in families:
        dictionaries["random_patch"] = _random_patch_dictionary(
            Ydict, args.n_atoms, args.dict_seed
        )
        dictionary_info["random_patch"] = {"column_draw_seed": args.dict_seed}
    if "pca" in families:
        U, singular_values, _ = randomized_svd(
            Ydict,
            n_components=args.n_atoms,
            n_iter=5,
            random_state=args.dict_seed,
        )
        dictionaries["pca"] = np.asarray(U, dtype=np.float64)
        dictionary_info["pca"] = {
            "randomized_svd_seed": args.dict_seed,
            "singular_values": singular_values.tolist(),
        }
    if "ksvd" in families:
        D, _, info = ksvd(
            Ydict,
            n_atoms=args.n_atoms,
            T=args.sparsity,
            T_min=1,
            n_iter=args.ksvd_iter,
            seed=args.dict_seed,
        )
        dictionaries["ksvd"] = D
        dictionary_info["ksvd"] = info
    print(
        "learned SSL-latent dictionaries "
        + ", ".join(f"{name}:{D.shape}" for name, D in dictionaries.items()),
        flush=True,
    )

    tokens = {
        family: np.zeros((total_nodes, D.shape[1]), dtype=np.float32)
        for family, D in dictionaries.items()
    }
    encoded_rows = np.flatnonzero(encoded_mask)
    transformed_encoded = transform_rows(latent[encoded_rows])
    for count, (row, y) in enumerate(zip(encoded_rows.tolist(), transformed_encoded), 1):
        for family, D in dictionaries.items():
            tokens[family][row] = _omp(D, y, args.sparsity).astype(np.float32)
        if count % 25000 == 0 or count == len(encoded_rows):
            print(f"sparse-coded SSL latents {count}/{len(encoded_rows)}", flush=True)

    for local_i in np.concatenate([va, te]):
        lo, hi = int(offsets[int(local_i)]), int(offsets[int(local_i) + 1])
        for family in families:
            if np.any(tokens[family][lo:hi] != 0):
                raise AssertionError(f"{family} encoded official-valid/test nodes")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "offsets": offsets,
        "original_indices": keep,
        "train_indices": tr,
        "valid_indices": va,
        "test_indices": te,
        "ssl_latent_mean": latent_mean.astype(np.float32),
        "ssl_fit_rows": selected_rows.astype(np.int64),
    }
    for family, D in dictionaries.items():
        archive[f"dictionary_{family}"] = D.astype(np.float32)
        archive[f"tokens_{family}"] = tokens[family]
    np.savez_compressed(output, **archive)

    checkpoint_path = output.with_suffix(".ssl.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "atom_feature_dims": atom_dims,
            "config": vars(args),
            "fit_indices": fit_indices,
        },
        checkpoint_path,
    )
    meta = {
        "protocol_id": "molhiv-fold-only-masked-context-gine-ksvd-v1",
        "labels_used": False,
        "explicit_ring_features": False,
        "test_policy": (
            "official-valid/test graphs were not loaded into the SSL model, "
            "context-encoded, or sparse-coded"
        ),
        "config": vars(args),
        "data_meta": bundle.meta,
        "dictionary_fit_protocol": "official-train-inner-scaffold-fold-only-unlabeled",
        "dictionary_fit_n_graphs": int(len(fit_indices)),
        "dictionary_fit_indices_sha256": hashlib.sha256(
            fit_indices.tobytes()
        ).hexdigest(),
        "heldout_scaffold_n_graphs": int(len(heldout_indices)),
        "heldout_overlap_with_fit": 0,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "ssl_training": {
            "objective": "masked atom categorical reconstruction from local graph context",
            "own_atom_features_removed_in_cached_latent": True,
            "context_mask_groups": int(args.context_mask_groups),
            "loss_curve": loss_curve,
            "field_accuracy_curve": accuracy_curve,
            "final_mean_field_accuracy": float(np.mean(accuracy_curve[-1])),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
            "state_sha256": _state_sha256(model),
            "checkpoint": str(checkpoint_path),
        },
        "latent_stats": {
            "dimension": int(args.ssl_hidden),
            "reservoir_count": int(reservoir.shape[0]),
            "effective_rank": effective_rank,
            "coordinate_std_mean": float(reservoir.std(axis=0).mean()),
            "zero_centered_latents": int(
                (np.linalg.norm(reservoir, axis=1) <= 1e-12).sum()
            ),
        },
        "dictionary_info": dictionary_info,
        "families": families,
        "total_nodes": total_nodes,
        "encoded_official_train_nodes": int(encoded_mask.sum()),
        "encoded_official_valid_nodes": 0,
        "encoded_official_test_nodes": 0,
        "elapsed_sec": time.time() - t0,
        "archive": str(output),
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {output}, {meta_path}, and {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
