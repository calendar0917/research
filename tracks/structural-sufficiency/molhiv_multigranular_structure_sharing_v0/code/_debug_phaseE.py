"""Temporary isolation script for the Phase E CUDA index-out-of-bounds assert."""
from __future__ import annotations

import sys

import numpy as np
import torch

import msds_common as C
import msds_data as D
import run_phaseE_gine as G


def main() -> None:
    torch.manual_seed(0)
    molecules = D.load_molecules()
    bundle = G.load_bundle()
    print("M", bundle["M"].shape, bundle["M"].nnz, "max col", bundle["M"].indices.max(), flush=True)
    M_sp = torch.sparse_csr_tensor(
        torch.as_tensor(bundle["M"].indptr.astype(np.int64)),
        torch.as_tensor(bundle["M"].indices.astype(np.int64)),
        torch.as_tensor(bundle["M"].data.astype(np.float32)),
        size=bundle["M"].shape, device="cuda",
    )
    w = torch.zeros(bundle["M"].shape[1], 1, device="cuda")
    out = torch.sparse.mm(M_sp, w)
    print("mm ok", out.shape, flush=True)
    graphs = G.build_graphs(molecules, torch)
    from torch_geometric.loader import DataLoader

    loader = DataLoader([graphs[int(i)] for i in molecules.split["train"][:256]], batch_size=128)
    for batch in loader:
        idx = batch.graph_index.view(-1)
        print("idx dtype", idx.dtype, "min", int(idx.min()), "max", int(idx.max()),
              "n rows", int(M_sp.shape[0]), "batch size", int(idx.numel()), flush=True)
        z = torch.sparse.mm(M_sp, w).view(-1)[idx]
        print("index_select ok", z.shape, flush=True)
        mb = G.make_model(64, 3, bundle["M"].shape[1], 0, False, torch).cuda()
        b = batch.to("cuda")
        x = mb.atom_encoder(b.x)
        print("atom encoder ok", x.shape, "x max per col", b.x.max(dim=0).values.tolist(), flush=True)
        e = mb.bond_encoder(b.edge_attr)
        print("bond encoder ok", e.shape, "edge max per col", b.edge_attr.max(dim=0).values.tolist(), flush=True)
        break


if __name__ == "__main__":
    main()
