"""Extract the fold-specific masked-context SSL latents from a saved checkpoint.

This utility does not retrain the encoder.  It reconstructs the exact model used
by ``build_molhiv_ssl_node_tokens.py`` and deterministically encodes every node
in the official-training split.  Each node is masked in exactly one of K passes,
so its exported latent excludes its own atom fields.  Official-valid/test rows
remain zero and must not be used by selection-phase experiments.
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

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import _patch_torch_load_weights_only, load_molhiv
from code.molhiv_node_tokens import graph_node_offsets


def _state_sha256(state_dict: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        h.update(name.encode("utf-8"))
        h.update(np.asarray(tensor.detach().cpu()).tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    _patch_torch_load_weights_only()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred import PygGraphPropPredDataset
        from ogb.graphproppred.mol_encoder import BondEncoder
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv
    except ImportError as exc:
        raise RuntimeError(f"missing SSL GNN dependencies: {exc}") from exc

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    state_dict = checkpoint["state_dict"]
    atom_dims = [int(x) for x in checkpoint["atom_feature_dims"]]
    hidden = int(config["ssl_hidden"])
    layers = int(config["ssl_layers"])
    groups = int(config["context_mask_groups"])
    max_graphs = int(config["max_graphs"])
    data_seed = int(config["data_seed"])
    ssl_seed = int(config["ssl_seed"])

    class MaskableAtomEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList([
                nn.Embedding(dim + 1, hidden) for dim in atom_dims
            ])

        def forward(self, x, mask):
            out = 0.0
            for field, (dim, emb) in enumerate(zip(atom_dims, self.embeddings)):
                values = x[:, field]
                if mask is not None:
                    values = values.clone()
                    values[mask] = dim
                out = out + emb(values)
            return out

    class MaskedContextGINE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atom_encoder = MaskableAtomEncoder()
            self.bond_encoder = BondEncoder(hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.heads = nn.ModuleList([nn.Linear(hidden, dim) for dim in atom_dims])

        def forward(self, data, mask):
            x = self.atom_encoder(data.x, mask)
            edge_attr = self.bond_encoder(data.edge_attr)
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
            return x

    t0 = time.time()
    torch.manual_seed(ssl_seed)
    np.random.seed(ssl_seed)
    model = MaskedContextGINE()
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(torch.device(args.device))

    bundle = load_molhiv(
        max_graphs=None if max_graphs <= 0 else max_graphs,
        seed=data_seed,
        with_features=False,
    )
    graphs = bundle.graphs
    offsets = graph_node_offsets(graphs)
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    train_indices = np.asarray(bundle.split["train"], dtype=np.int64)
    valid_indices = np.asarray(bundle.split["valid"], dtype=np.int64)
    test_indices = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.token_cache, allow_pickle=False) as source:
        required = {
            "offsets", "original_indices", "train_indices", "valid_indices",
            "test_indices", "ssl_latent_mean",
        }
        missing = required.difference(source.files)
        if missing:
            raise KeyError(f"token cache is missing {sorted(missing)}")
        source_offsets = np.asarray(source["offsets"], dtype=np.int64)
        source_original = np.asarray(source["original_indices"], dtype=np.int64)
        source_train = np.asarray(source["train_indices"], dtype=np.int64)
        source_valid = np.asarray(source["valid_indices"], dtype=np.int64)
        source_test = np.asarray(source["test_indices"], dtype=np.int64)
        latent_mean = np.asarray(source["ssl_latent_mean"], dtype=np.float32)

    checks = [
        (offsets, source_offsets, "offsets"),
        (expected_original, source_original, "original_indices"),
        (train_indices, source_train, "train_indices"),
        (valid_indices, source_valid, "valid_indices"),
        (test_indices, source_test, "test_indices"),
    ]
    for actual, expected, name in checks:
        if not np.array_equal(actual, expected):
            raise ValueError(f"checkpoint dataset and token-cache {name} do not match")
    if latent_mean.shape != (hidden,):
        raise ValueError(f"unexpected latent mean shape {latent_mean.shape}")

    root = bundle.meta["root"]
    pyg = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    data_list = []
    for count, local_i in enumerate(train_indices.tolist(), 1):
        original_i = int(expected_original[local_i])
        data = pyg[original_i].clone()
        if "y" in data:
            del data.y
        expected_nodes = int(offsets[local_i + 1] - offsets[local_i])
        if int(data.num_nodes) != expected_nodes:
            raise ValueError(f"node count mismatch for graph {local_i}")
        data.graph_id = torch.tensor([local_i], dtype=torch.long)
        local_nodes = torch.arange(expected_nodes, dtype=torch.long)
        group_offset = int(
            hashlib.sha256(f"{ssl_seed}:{original_i}".encode()).digest()[0]
        ) % groups
        data.context_group = (local_nodes + group_offset) % groups
        data_list.append(data)
        if count % 1000 == 0 or count == len(train_indices):
            print(f"loaded official-train graphs {count}/{len(train_indices)}", flush=True)

    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    raw_latents = np.zeros((int(offsets[-1]), hidden), dtype=np.float32)
    encoded = np.zeros(int(offsets[-1]), dtype=bool)
    with torch.no_grad():
        for batch_i, data in enumerate(loader, 1):
            data = data.to(device)
            batch_latent = torch.zeros(
                (data.x.shape[0], hidden), dtype=torch.float32, device=device
            )
            assigned = torch.zeros(data.x.shape[0], dtype=torch.bool, device=device)
            for group in range(groups):
                mask = data.context_group == group
                h = model(data, mask)
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
                if part.shape != (hi - lo, hidden):
                    raise AssertionError("batched latent/node offset mismatch")
                raw_latents[lo:hi] = part
                encoded[lo:hi] = True
            if batch_i % 10 == 0 or batch_i == len(loader):
                print(f"encoded batches {batch_i}/{len(loader)}", flush=True)

    latents = raw_latents.astype(np.float64) - latent_mean.astype(np.float64)[None, :]
    norms = np.linalg.norm(latents, axis=1)
    valid_rows = encoded & (norms > 1e-12)
    latents[valid_rows] /= norms[valid_rows, None]
    latents[~encoded] = 0.0
    latents[encoded & ~valid_rows] = 0.0
    latents = latents.astype(np.float32)

    for local_i in np.concatenate([valid_indices, test_indices]):
        lo, hi = int(offsets[int(local_i)]), int(offsets[int(local_i) + 1])
        if encoded[lo:hi].any() or np.any(latents[lo:hi] != 0):
            raise AssertionError("official-valid/test latent was encoded")
    train_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in train_indices
    ])
    if not encoded[train_rows].all():
        raise AssertionError("some official-train rows were not encoded")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=expected_original,
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
        ssl_latent_mean=latent_mean,
        latents=latents,
    )
    train_norms = np.linalg.norm(latents[train_rows].astype(np.float64), axis=1)
    meta = {
        "protocol_id": "molhiv-fold-only-masked-context-latent-export-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_state_sha256": _state_sha256(state_dict),
        "token_cache": args.token_cache,
        "output": str(output),
        "config": config,
        "labels_used": False,
        "official_valid_test_encoded": False,
        "encoded_graphs": int(len(train_indices)),
        "encoded_nodes": int(encoded.sum()),
        "latent_dimension": hidden,
        "train_norm_quantiles": {
            str(q): float(v) for q, v in zip(
                [0.0, 0.01, 0.5, 0.99, 1.0],
                np.quantile(train_norms, [0.0, 0.01, 0.5, 0.99, 1.0]),
            )
        },
        "elapsed_sec": time.time() - t0,
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
