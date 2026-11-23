"""Fixed-depth message-range intervention (route B).

One architecture, two (or three) ranges:

* ``FULL``: five message-passing layers on the original molecule.
* ``EGO2``: for every atom, the complete induced radius-2 ego is packed as an
  independent graph; the same five layers and the same weights are applied, and
  only that ego's root vector is returned to the molecule.
* ``EGO3``: identical with radius 3.

The trunk has no dropout, no BatchNorm/GraphNorm and no graph-mode LayerNorm;
the only normalisation is per-node ``torch.nn.LayerNorm``.  Both arms share the
eight whole-graph scalars, so the intervention is message range, not "all global
information".
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import rice_energy as RE

import sr_common as C

MODES = ("FULL", "EGO2", "EGO3")
MODEL_ARMS = ("G_FULL5", "G_EGO2_5", "G_EGO3_5")


# ---------------------------------------------------------------------------
# pure structure precomputation
# ---------------------------------------------------------------------------
def _adjacency(n_atoms: int, u: np.ndarray, v: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def _bfs(n_atoms: int, adj: Sequence[Sequence[int]], source: int, radius: int) -> list[int]:
    dist = [-1] * int(n_atoms)
    dist[int(source)] = 0
    frontier = [int(source)]
    for d in range(1, int(radius) + 1):
        nxt: list[int] = []
        for x in frontier:
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = d
                    nxt.append(y)
        frontier = nxt
    return dist


def ego_block(st: Any, radius: int) -> dict[str, np.ndarray]:
    n = int(st.n_atoms)
    adj = _adjacency(n, st.bond_u, st.bond_v)
    atom: list[int] = []
    src: list[int] = []
    dst: list[int] = []
    bond: list[int] = []
    ptr: list[int] = [0]
    for root in range(n):
        dist = _bfs(n, adj, root, radius)
        nodes = [root] + [x for x in range(n) if 0 < dist[x] <= radius]
        base = len(atom)
        pos = {int(a): base + i for i, a in enumerate(nodes)}
        atom.extend(nodes)
        for bid, (a, b) in enumerate(zip(st.bond_u.tolist(), st.bond_v.tolist())):
            if int(a) in pos and int(b) in pos:
                ia, ib = pos[int(a)], pos[int(b)]
                src.extend([ia, ib])
                dst.extend([ib, ia])
                bond.extend([bid, bid])
        ptr.append(len(atom))
    return {
        "atom": np.asarray(atom, np.int32),
        "src": np.asarray(src, np.int32),
        "dst": np.asarray(dst, np.int32),
        "bond": np.asarray(bond, np.int32),
        "ptr": np.asarray(ptr, np.int64),
    }


def prepare_ranges(st: Any) -> None:
    if not hasattr(st, "ego2"):
        st.ego2 = ego_block(st, 2)
    if not hasattr(st, "ego3"):
        st.ego3 = ego_block(st, 3)


# ---------------------------------------------------------------------------
# collation
# ---------------------------------------------------------------------------
def collate(rows: Sequence[int], structs: Mapping[int, Any], mode: str,
            device: Any = "cpu") -> dict[str, Any]:
    rows = [int(r) for r in rows]
    if not rows:
        raise ValueError("empty batch")
    atom_feat: list[np.ndarray] = []
    bond_feat: list[np.ndarray] = []
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    root_index: list[np.ndarray] = []
    root_mol: list[np.ndarray] = []
    atom_mol: list[np.ndarray] = []
    graph: list[np.ndarray] = []
    off = 0
    for m, row in enumerate(rows):
        st = structs[row]
        graph.append(np.asarray(st.graph_scalars, np.float32))
        if mode == "FULL":
            n = int(st.n_atoms)
            atom_feat.append(st.atom_feat)
            u = st.bond_u.astype(np.int64)
            v = st.bond_v.astype(np.int64)
            src.append(np.concatenate([u, v]) + off)
            dst.append(np.concatenate([v, u]) + off)
            bf = st.bond_feat.astype(np.int64)
            bond_feat.append(np.concatenate([bf, bf], axis=0))
            root_index.append(np.arange(n, dtype=np.int64) + off)
            root_mol.append(np.full(n, m, np.int64))
            atom_mol.append(np.full(n, m, np.int64))
            off += n
        else:
            prepare_ranges(st)
            block = st.ego2 if mode == "EGO2" else st.ego3
            n_ego = int(block["ptr"].size - 1)
            atom_feat.append(st.atom_feat[block["atom"]])
            src.append(block["src"].astype(np.int64) + off)
            dst.append(block["dst"].astype(np.int64) + off)
            bond_feat.append(st.bond_feat[block["bond"]].astype(np.int64))
            root_index.append(block["ptr"][:-1].astype(np.int64) + off)
            root_mol.append(np.full(n_ego, m, np.int64))
            atom_mol.append(np.full(int(block["atom"].size), m, np.int64))
            off += int(block["atom"].size)

    def cat(parts: Sequence[np.ndarray], width: int, dtype: Any = np.int64) -> np.ndarray:
        return np.concatenate(parts).astype(dtype, copy=False) if parts \
            else np.zeros((0, width), dtype)

    out = {
        "atom_feat": torch.as_tensor(cat(atom_feat, 9), dtype=torch.long),
        "bond_feat": torch.as_tensor(cat(bond_feat, 3), dtype=torch.long),
        "edge_src": torch.as_tensor(np.concatenate(src), dtype=torch.long),
        "edge_dst": torch.as_tensor(np.concatenate(dst), dtype=torch.long),
        "root_index": torch.as_tensor(np.concatenate(root_index), dtype=torch.long),
        "root_mol": torch.as_tensor(np.concatenate(root_mol), dtype=torch.long),
        "atom_mol": torch.as_tensor(np.concatenate(atom_mol), dtype=torch.long),
        "graph_scalars": torch.as_tensor(np.stack(graph), dtype=torch.float32),
        "n_molecules": int(len(rows)),
        "n_atoms": int(sum(x.shape[0] for x in atom_feat)),
        "n_roots": int(sum(x.size for x in root_index)),
        "mode": mode,
        "rows": np.asarray(rows, np.int64),
    }
    if device != "cpu":
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def _mlp(dims: Sequence[int], dropout: float = 0.0,
         final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
    if final_activation:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class GNNModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 hidden: int = 128, mlp_hidden: int = 256, layers: int = 5,
                 readout_dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.n_layers = int(layers)
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), int(hidden))
                                       for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), int(hidden))
                                       for d in bond_dims])
        self.mlps = nn.ModuleList([
            _mlp([int(hidden), int(mlp_hidden), int(hidden)], final_activation=False)
            for _ in range(int(layers))])
        self.eps = nn.ParameterList([nn.Parameter(torch.zeros(()))
                                     for _ in range(int(layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(int(hidden))
                                    for _ in range(int(layers))])
        self.mid = 64
        self.readout_in_dim = 3 * int(hidden) + 1 + 8
        self.readout = nn.Sequential(
            nn.Linear(self.readout_in_dim, int(hidden)), nn.SiLU(),
            nn.Dropout(float(readout_dropout)),
            nn.Linear(int(hidden), int(self.mid)), nn.SiLU(),
            nn.Linear(int(self.mid), 1))
        self.beta = nn.Parameter(torch.zeros(()))

    def init_parameters(self, seed: int) -> dict[str, str]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 29)
        with torch.no_grad():
            for emb in list(self.atom_emb) + list(self.bond_emb):
                emb.weight.normal_(0.0, 0.1, generator=gen)
            for module in list(self.mlps) + [self.readout]:
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        bound = 1.0 / math.sqrt(layer.in_features)
                        layer.weight.uniform_(-bound, bound, generator=gen)
                        if layer.bias is not None:
                            layer.bias.uniform_(-bound, bound, generator=gen)
            for norm in self.norms:
                norm.weight.fill_(1.0)
                norm.bias.zero_()
            last = self.readout[-1]
            last.weight.normal_(0.0, 1e-3, generator=gen)
            last.bias.zero_()
            self.beta.zero_()
        return self.init_hashes()

    def init_hashes(self) -> dict[str, str]:
        return C.module_hashes({
            "atom_emb": self.atom_emb, "bond_emb": self.bond_emb,
            "mlps": self.mlps, "eps": self.eps, "norms": self.norms,
            "readout": self.readout, "beta": self.beta})

    def ledger(self) -> dict[str, int]:
        def n(m: Any) -> int:
            if isinstance(m, nn.Parameter):
                return int(m.numel())
            return int(sum(p.numel() for p in m.parameters()))

        out = {"atom_emb": n(self.atom_emb), "bond_emb": n(self.bond_emb),
               "mlps": n(self.mlps), "eps": n(self.eps), "norms": n(self.norms),
               "readout": n(self.readout), "beta": n(self.beta)}
        out["total"] = int(sum(out.values()))
        return out

    def zero_unseen(self, atom_ok: Sequence[np.ndarray],
                    bond_ok: Sequence[np.ndarray]) -> dict[str, int]:
        zeroed: dict[str, int] = {}
        with torch.no_grad():
            for i, emb in enumerate(self.atom_emb):
                mask = torch.as_tensor(np.asarray(atom_ok[i], bool), device=emb.weight.device)
                zeroed[f"atom_field_{i}"] = int((~mask).sum().item())
                emb.weight[~mask] = 0.0
            for i, emb in enumerate(self.bond_emb):
                mask = torch.as_tensor(np.asarray(bond_ok[i], bool), device=emb.weight.device)
                zeroed[f"bond_field_{i}"] = int((~mask).sum().item())
                emb.weight[~mask] = 0.0
        return zeroed

    def encode(self, batch: Mapping[str, torch.Tensor], n_layers: int | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        n_layers = self.n_layers if n_layers is None else int(n_layers)
        atom = batch["atom_feat"]
        bond = batch["bond_feat"]
        parts = [emb(atom[:, i]) for i, emb in enumerate(self.atom_emb)]
        h = torch.stack(parts, dim=0).sum(dim=0) / math.sqrt(len(parts))
        eparts = [emb(bond[:, i]) for i, emb in enumerate(self.bond_emb)]
        e = torch.stack(eparts, dim=0).sum(dim=0) / math.sqrt(len(eparts))
        src, dst = batch["edge_src"], batch["edge_dst"]
        for layer in range(n_layers):
            msg = torch.nn.functional.relu(h[src] + e)
            m = torch.zeros_like(h)
            m.index_add_(0, dst, msg)
            z = self.mlps[layer]((1.0 + self.eps[layer]) * h + m)
            h = self.norms[layer](h + z)
        return h, e

    def forward(self, batch: Mapping[str, torch.Tensor], n_layers: int | None = None,
                return_h: bool = False) -> dict[str, Any]:
        h, _e = self.encode(batch, n_layers)
        roots = h[batch["root_index"]]
        h_sum, h_mean, h_std, count = RE.seg_stats(
            roots, batch["root_mol"], int(batch["n_molecules"]), self.hidden)
        g = (batch["graph_scalars"] - self.graph_mean) / self.graph_scale
        readout_in = torch.cat([h_sum, h_mean, h_std,
                                torch.log1p(count).unsqueeze(1), g], dim=1)
        if int(readout_in.shape[1]) != self.readout_in_dim:
            raise ValueError("readout width mismatch")
        yhat = self.readout(readout_in).view(-1) + self.beta
        out: dict[str, Any] = {"yhat_std": yhat, "roots": roots,
                               "readout_in": readout_in}
        if return_h:
            out["h"] = h
        return out

    graph_mean: torch.Tensor
    graph_scale: torch.Tensor


def build_model(arm: str, atom_dims: Sequence[int], bond_dims: Sequence[int],
                seed: int, norm: Mapping[str, np.ndarray],
                cfg: Mapping[str, Any]) -> tuple[GNNModel, dict[str, Any]]:
    if arm not in MODEL_ARMS:
        raise KeyError(arm)
    dims = dict(cfg["gnn"])
    model = GNNModel(atom_dims, bond_dims,
                     hidden=int(dims["hidden"]),
                     mlp_hidden=int(dims["mlp_hidden"]),
                     layers=int(dims["layers"]),
                     readout_dropout=float(dims["readout_dropout"]))
    model.register_buffer("graph_mean",
                          torch.as_tensor(norm["graph_mean"], dtype=torch.float32))
    model.register_buffer("graph_scale",
                          torch.as_tensor(norm["graph_scale"], dtype=torch.float32))
    hashes = model.init_parameters(int(seed))
    info = {"arm": arm, "family": "G", "seed": int(seed),
            "mode": arm.replace("G_", "").replace("_5", ""),
            "parameters": model.ledger(), "init_hashes": hashes,
            "dims": dims}
    return model, info


def clone_initial(model: GNNModel) -> GNNModel:
    other = GNNModel.__new__(GNNModel)
    return other


def mode_of(arm: str) -> str:
    return arm.replace("G_", "").replace("_5", "")