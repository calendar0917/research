"""BBPA-v0 models: BLOCK_BAG / LOCAL_PORT / TREE_PORT.

All three arms instantiate the same modules with the same shapes and the same
explicit initialisation.  They differ only in the forward computation:

* BLOCK_BAG  : block-global summaries + public whole-graph summaries.
* LOCAL_PORT : bridge/port messages use the neighbouring block's *local*
               port representation ``u_(k|i)``.
* TREE_PORT  : bridge/port messages use the recursively composed directed
               subtree state ``s_(k->i)`` computed by tree DP.

No attention, no softmax top-k, no learnable ids, no message passing on the
original atom graph.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("BLOCK_BAG", "LOCAL_PORT", "TREE_PORT")
N_DIST_BUCKETS = 6


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0,
         final_silu: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(int(in_dim), int(hidden)), nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(int(hidden), int(out_dim)))
    if final_silu:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _seg_mean_std(values: torch.Tensor, seg: torch.Tensor, n: int
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Population mean/std and count for potentially empty segments."""
    d = int(values.shape[1])
    count = torch.zeros(int(n), device=values.device, dtype=values.dtype)
    count.index_add_(0, seg, torch.ones(seg.shape[0], device=values.device,
                                        dtype=values.dtype))
    total = torch.zeros(int(n), d, device=values.device, dtype=values.dtype)
    total.index_add_(0, seg, values)
    sq = torch.zeros(int(n), d, device=values.device, dtype=values.dtype)
    sq.index_add_(0, seg, values * values)
    denom = count.clamp(min=1.0)
    mean = total / denom[:, None]
    var = (sq / denom[:, None] - mean * mean).clamp(min=0.0)
    std = torch.sqrt(var + 1e-8)
    std = torch.where((count > 1)[:, None], std, torch.zeros_like(std))
    return mean, std, count


class BBPAModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 atom_dim: int = 64, bond_dim: int = 32, block_dim: int = 96,
                 subtree_dim: int = 96, interface_dim: int = 96,
                 block_hidden: int = 128, port_hidden: int = 128,
                 message_hidden: int = 96, update_hidden: int = 96,
                 q_hidden: int = 96, readout_hidden: int = 128,
                 readout_mid: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.block_dim = int(block_dim)
        self.subtree_dim = int(subtree_dim)
        self.interface_dim = int(interface_dim)
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), self.atom_dim)
                                       for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), self.bond_dim)
                                       for d in bond_dims])

        # internal-bond atom-pair joint representation
        pair_in = 3 * self.atom_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, 96), nn.SiLU(),
            nn.Linear(96, 64), nn.SiLU(),
        )
        block_in = (3 * self.atom_dim + 2 * self.bond_dim + 2 * 64 + 5)
        self.block_mlp = _mlp(block_in, block_hidden, self.block_dim, dropout=dropout)

        # u_global + port atom + 6*(atom sum + bond sum + 2 counts) + 3 scalars
        port_in = (self.block_dim + self.atom_dim + N_DIST_BUCKETS * (self.atom_dim
                    + self.bond_dim + 2) + 3)
        self.port_mlp = _mlp(port_in, port_hidden, self.block_dim, dropout=dropout)

        msg_in = (self.subtree_dim + self.bond_dim + self.atom_dim + 3 + N_DIST_BUCKETS)
        self.message_mlp = _mlp(msg_in, message_hidden, self.subtree_dim, dropout=0.0)
        update_in = self.subtree_dim + 3 * self.subtree_dim + 1
        self.update_mlp = _mlp(update_in, update_hidden, self.subtree_dim, dropout=0.0)
        self.norm = nn.LayerNorm(self.subtree_dim)

        q_in = 3 * self.interface_dim + self.bond_dim
        self.q_mlp = _mlp(q_in, q_hidden, self.interface_dim, dropout=0.0)

        public_dim = 3 * self.atom_dim + 3 * self.bond_dim + 6
        readout_in = 4 * self.interface_dim + public_dim
        self.readout_in_dim = int(readout_in)
        self.readout_mlp = nn.Sequential(
            nn.Linear(readout_in, readout_hidden), nn.SiLU(), nn.Dropout(float(dropout)),
            nn.Linear(readout_hidden, readout_mid), nn.SiLU(),
            nn.Linear(readout_mid, 1, bias=False),
        )
        self.beta = nn.Parameter(torch.zeros(()))

    # -- init ---------------------------------------------------------------
    def init_parameters(self, seed: int, last_std: float = 1e-3) -> dict[str, Any]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 17)
        with torch.no_grad():
            for emb in list(self.atom_emb) + list(self.bond_emb):
                emb.weight.normal_(0.0, 0.1, generator=gen)
            ordered = [
                self.pair_mlp, self.block_mlp, self.port_mlp, self.message_mlp,
                self.update_mlp, self.q_mlp, self.readout_mlp,
            ]
            for module in ordered:
                for layer in module.modules():
                    if not isinstance(layer, nn.Linear):
                        continue
                    bound = 1.0 / math.sqrt(layer.in_features)
                    layer.weight.uniform_(-bound, bound, generator=gen)
                    if layer.bias is not None:
                        layer.bias.uniform_(-bound, bound, generator=gen)
            last = [m for m in self.readout_mlp.modules()
                    if isinstance(m, nn.Linear) and m.out_features == 1][-1]
            last.weight.normal_(0.0, float(last_std), generator=gen)
            self.beta.zero_()
        return self.shared_hashes()

    def shared_hashes(self) -> dict[str, str]:
        import bbpa_common as C

        return C.module_hashes({
            "atom_emb": self.atom_emb,
            "bond_emb": self.bond_emb,
            "pair_mlp": self.pair_mlp,
            "block_mlp": self.block_mlp,
            "port_mlp": self.port_mlp,
            "message_mlp": self.message_mlp,
            "update_mlp": self.update_mlp,
            "q_mlp": self.q_mlp,
            "norm": self.norm,
            "readout_mlp": self.readout_mlp,
            "beta": self.beta,
        })

    def zero_unseen(self, atom_ok: Sequence[np.ndarray], bond_ok: Sequence[np.ndarray]
                    ) -> dict[str, int]:
        zeroed: dict[str, int] = {}

        def zero(module_list: nn.ModuleList, support: Sequence[np.ndarray], tag: str) -> None:
            for i, emb in enumerate(module_list):
                mask = torch.as_tensor(np.asarray(support[i], bool), device=emb.weight.device)
                if mask.numel() != emb.weight.shape[0]:
                    raise ValueError(f"{tag} support width mismatch")
                zeroed[f"{tag}_field_{i}"] = int((~mask).sum().item())
                with torch.no_grad():
                    emb.weight[~mask] = 0.0

        zero(self.atom_emb, atom_ok, "atom")
        zero(self.bond_emb, bond_ok, "bond")
        return zeroed

    # -- encoders -----------------------------------------------------------
    def embed_atoms(self, atom_feat: torch.Tensor) -> torch.Tensor:
        out = sum(emb(atom_feat[:, i]) for i, emb in enumerate(self.atom_emb))
        return out / math.sqrt(len(self.atom_emb))

    def embed_bonds(self, bond_feat: torch.Tensor) -> torch.Tensor:
        out = sum(emb(bond_feat[:, i]) for i, emb in enumerate(self.bond_emb))
        return out / math.sqrt(len(self.bond_emb))

    def encode_blocks(self, batch: Mapping[str, Any], A: torch.Tensor, B: torch.Tensor
                      ) -> torch.Tensor:
        n_blocks = int(batch["n_blocks"])
        n_atoms = batch["block_n_atoms"].to(A.dtype).clamp(min=1.0)
        n_bonds = batch["block_n_bonds"].to(A.dtype).clamp(min=1.0)
        sum_a = torch.zeros(n_blocks, self.atom_dim, device=A.device, dtype=A.dtype)
        sum_a.index_add_(0, batch["block_atom_seg"], A[batch["block_atom_idx"]])
        sq_a = torch.zeros_like(sum_a)
        sq_a.index_add_(0, batch["block_atom_seg"], A[batch["block_atom_idx"]] ** 2)
        mean_a = sum_a / n_atoms[:, None]
        std_a = torch.sqrt((sq_a / n_atoms[:, None] - mean_a * mean_a).clamp(min=0.0) + 1e-8)

        sum_b = torch.zeros(n_blocks, self.bond_dim, device=A.device, dtype=A.dtype)
        sum_b.index_add_(0, batch["block_bond_seg"], B[batch["block_bond_idx"]])
        mean_b = sum_b / n_bonds[:, None]

        a_u = A[batch["block_bond_u"]]
        a_v = A[batch["block_bond_v"]]
        pair_in = torch.cat([a_u + a_v, (a_u - a_v).abs(), a_u * a_v], dim=-1)
        pair = self.pair_mlp(pair_in)
        pair_sum = torch.zeros(n_blocks, pair.shape[1], device=A.device, dtype=A.dtype)
        pair_sum.index_add_(0, batch["block_bond_seg"], pair)
        pair_mean = pair_sum / n_bonds[:, None]

        n_ports = batch["block_n_ports"].to(A.dtype)
        cycle = batch["block_cycle_rank"].to(A.dtype)
        scalars = torch.stack([
            torch.log1p(n_atoms), torch.log1p(n_bonds), cycle / 5.0,
            torch.log1p(n_ports), n_ports / 5.0,
        ], dim=-1)
        block_in = torch.cat([
            sum_a / torch.sqrt(n_atoms)[:, None], mean_a, std_a,
            sum_b / torch.sqrt(n_bonds)[:, None], mean_b,
            pair_sum / torch.sqrt(n_bonds)[:, None], pair_mean, scalars,
        ], dim=-1)
        return self.block_mlp(block_in)

    def encode_ports(self, batch: Mapping[str, Any], A: torch.Tensor, B: torch.Tensor,
                     u_global: torch.Tensor) -> torch.Tensor:
        P = int(batch["n_ports"])
        seg_a = batch["port_bucket_atom_seg"]
        seg_b = batch["port_bucket_bond_seg"]
        bucket_a = torch.zeros(P * N_DIST_BUCKETS, self.atom_dim,
                               device=A.device, dtype=A.dtype)
        bucket_a.index_add_(0, seg_a, A[batch["port_bucket_atom_idx"]])
        bucket_b = torch.zeros(P * N_DIST_BUCKETS, self.bond_dim,
                               device=A.device, dtype=A.dtype)
        bucket_b.index_add_(0, seg_b, B[batch["port_bucket_bond_idx"]])
        ca = batch["port_bucket_atom_count"].to(A.dtype).view(P, N_DIST_BUCKETS)
        cb = batch["port_bucket_bond_count"].to(A.dtype).view(P, N_DIST_BUCKETS)
        port_block = batch["port_block"]
        scalars = torch.stack([
            batch["port_atom_degree"].to(A.dtype) / 5.0,
            batch["port_atom_n_ports"].to(A.dtype) / 5.0,
            batch["block_n_ports"][port_block].to(A.dtype) / 5.0,
        ], dim=-1)
        port_in = torch.cat([
            u_global[port_block], A[batch["port_atom_idx"]],
            bucket_a.view(P, N_DIST_BUCKETS * self.atom_dim),
            bucket_b.view(P, N_DIST_BUCKETS * self.bond_dim),
            ca, cb, scalars,
        ], dim=-1)
        return self.port_mlp(port_in)

    # -- tree DP ------------------------------------------------------------
    def dp_states(self, batch: Mapping[str, Any], A: torch.Tensor, B: torch.Tensor,
                  u_port: torch.Tensor, mode: str) -> torch.Tensor:
        n_edges = int(batch["n_edges"])
        states = torch.zeros(n_edges, self.subtree_dim, device=A.device, dtype=A.dtype)
        max_step = int(batch["max_step"])
        bridge_bond = batch["bridge_bond"]
        port_atom_idx = batch["port_atom_idx"]
        port_atom_degree = batch["port_atom_degree"].to(A.dtype)
        port_atom_n_ports = batch["port_atom_n_ports"].to(A.dtype)
        port_block = batch["port_block"]
        block_n_ports = batch["block_n_ports"].to(A.dtype)
        onehot = F.one_hot(batch["dep_dist_bucket"], N_DIST_BUCKETS).to(A.dtype)
        step_edge_ptr = batch["step_edge_ptr"].tolist()
        step_dep_ptr = batch["step_dep_ptr"].tolist()
        edge_order = batch["edge_order"]
        dep_order = batch["dep_order"]

        for s in range(max_step + 1):
            e0, e1 = int(step_edge_ptr[s]), int(step_edge_ptr[s + 1])
            if e1 <= e0:
                continue
            edge_ids = edge_order[e0:e1]
            u_src = u_port[batch["edge_port"][edge_ids]]
            d0, d1 = int(step_dep_ptr[s]), int(step_dep_ptr[s + 1])
            if d1 <= d0:
                states[edge_ids] = u_src
                continue
            dep_ids = dep_order[d0:d1]
            if mode == "TREE_PORT":
                dep_vals = states[batch["dep_edge"][dep_ids]]
            elif mode == "LOCAL_PORT":
                dep_vals = u_port[batch["dep_port"][dep_ids]]
            else:
                raise ValueError(f"mode {mode} has no DP states")
            dep_port_feat = batch["dep_port_feat"][dep_ids]
            msg_in = torch.cat([
                dep_vals,
                B[bridge_bond[batch["dep_bridge"][dep_ids]]],
                A[port_atom_idx[dep_port_feat]],
                torch.stack([
                    port_atom_degree[dep_port_feat] / 5.0,
                    port_atom_n_ports[dep_port_feat] / 5.0,
                    block_n_ports[port_block[dep_port_feat]] / 5.0,
                ], dim=-1),
                onehot[dep_ids],
            ], dim=-1)
            m = self.message_mlp(msg_in)
            n_e = e1 - e0
            target = batch["dep_target_local"][dep_ids]
            total = torch.zeros(n_e, self.subtree_dim, device=A.device, dtype=A.dtype)
            total.index_add_(0, target, m)
            sq = torch.zeros_like(total)
            sq.index_add_(0, target, m * m)
            count = torch.zeros(n_e, device=A.device, dtype=A.dtype)
            count.index_add_(0, target, torch.ones_like(target, dtype=A.dtype))
            denom = count.clamp(min=1.0)
            mean = total / denom[:, None]
            var = (sq / denom[:, None] - mean * mean).clamp(min=0.0)
            std = torch.where((count > 1)[:, None], torch.sqrt(var + 1e-8), torch.zeros_like(var))
            agg = torch.cat([total, mean, std, torch.log1p(count)[:, None]], dim=-1)
            cand = self.norm(u_src + self.update_mlp(torch.cat([u_src, agg], dim=-1)))
            states[edge_ids] = torch.where((count > 0)[:, None], cand, u_src)
        return states

    # -- readout ------------------------------------------------------------
    def graph_readout(self, batch: Mapping[str, Any], A: torch.Tensor, B: torch.Tensor,
                      u_global: torch.Tensor, q: torch.Tensor | None) -> torch.Tensor:
        n_mol = int(batch["n_molecules"])
        u_mean, u_std, _ = _seg_mean_std(u_global, batch["block_seg"], n_mol)
        if q is None or int(batch["n_bridges"]) == 0:
            q_mean = torch.zeros(n_mol, self.interface_dim, device=A.device, dtype=A.dtype)
            q_std = torch.zeros_like(q_mean)
        else:
            q_mean, q_std, _ = _seg_mean_std(q, batch["bridge_seg"], n_mol)

        n_atoms = torch.zeros(n_mol, device=A.device, dtype=A.dtype)
        n_atoms.index_add_(0, batch["atom_mol"], torch.ones(batch["n_atoms"],
                                                             device=A.device, dtype=A.dtype))
        n_bonds = torch.zeros(n_mol, device=A.device, dtype=A.dtype)
        n_bonds.index_add_(0, batch["bond_mol"], torch.ones(batch["n_bonds"],
                                                             device=A.device, dtype=A.dtype))
        atom_sum = torch.zeros(n_mol, self.atom_dim, device=A.device, dtype=A.dtype)
        atom_sum.index_add_(0, batch["atom_mol"], A)
        atom_sq = torch.zeros_like(atom_sum)
        atom_sq.index_add_(0, batch["atom_mol"], A * A)
        atom_mean = atom_sum / n_atoms[:, None]
        atom_std = torch.sqrt((atom_sq / n_atoms[:, None] - atom_mean * atom_mean).clamp(min=0.0) + 1e-8)
        bond_sum = torch.zeros(n_mol, self.bond_dim, device=A.device, dtype=A.dtype)
        bond_sum.index_add_(0, batch["bond_mol"], B)
        bond_sq = torch.zeros_like(bond_sum)
        bond_sq.index_add_(0, batch["bond_mol"], B * B)
        bond_mean = bond_sum / n_bonds[:, None]
        bond_std = torch.sqrt((bond_sq / n_bonds[:, None] - bond_mean * bond_mean).clamp(min=0.0) + 1e-8)
        public = torch.cat([
            atom_sum / torch.sqrt(n_atoms)[:, None], atom_mean, atom_std,
            bond_sum / torch.sqrt(n_bonds)[:, None], bond_mean, bond_std,
            torch.stack([
                torch.log1p(n_atoms), torch.log1p(n_bonds),
                torch.log1p(batch["mol_block_ptr"][1:] - batch["mol_block_ptr"][:-1]
                            ).to(A.dtype),
                torch.log1p(batch["mol_bridge_ptr"][1:] - batch["mol_bridge_ptr"][:-1]
                            ).to(A.dtype),
                (batch["mol_block_ptr"][1:] - batch["mol_block_ptr"][:-1]).to(A.dtype) / 20.0,
                (batch["mol_bridge_ptr"][1:] - batch["mol_bridge_ptr"][:-1]).to(A.dtype) / 20.0,
            ], dim=-1),
        ], dim=-1)
        readout = torch.cat([q_mean, q_std, u_mean, u_std, public], dim=-1)
        return self.readout_mlp(readout).squeeze(-1) + self.beta

    # -- forward ------------------------------------------------------------
    def forward_batch(self, batch: Mapping[str, Any], mode: str,
                      return_states: bool = False) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        A = self.embed_atoms(batch["atom_feat"])
        B = self.embed_bonds(batch["bond_feat"])
        u_global = self.encode_blocks(batch, A, B)
        q = None
        u_port = None
        states = None
        if mode != "BLOCK_BAG":
            u_port = self.encode_ports(batch, A, B, u_global)
            states = self.dp_states(batch, A, B, u_port, mode)
            if int(batch["n_bridges"]) > 0:
                s0 = states[0::2]
                s1 = states[1::2]
                q_in = torch.cat([
                    s0 + s1, (s0 - s1).abs(), s0 * s1, B[batch["bridge_bond"]],
                ], dim=-1)
                q = self.q_mlp(q_in)
        yhat = self.graph_readout(batch, A, B, u_global, q)
        out: dict[str, Any] = {"yhat_std": yhat}
        if return_states:
            out.update({"u_global": u_global, "u_port": u_port, "states": states, "q": q})
        return out


def parameter_ledger(model: nn.Module) -> dict[str, int]:
    groups: dict[str, int] = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + int(param.numel())
    groups["total"] = sum(int(p.numel()) for p in model.parameters())
    return groups


def build_model(atom_dims: Sequence[int], bond_dims: Sequence[int], seed: int,
                cfg: Mapping[str, Any] | None = None) -> tuple[BBPAModel, dict[str, Any]]:
    dims = dict((cfg or {}).get("dims", {}))
    model = BBPAModel(
        atom_dims, bond_dims,
        atom_dim=int(dims.get("atom_state", 64)),
        bond_dim=int(dims.get("bond_state", 32)),
        block_dim=int(dims.get("block_state", 96)),
        subtree_dim=int(dims.get("subtree_state", 96)),
        interface_dim=int(dims.get("interface_state", 96)),
        block_hidden=int(dims.get("block_hidden", 128)),
        port_hidden=int(dims.get("port_hidden", 128)),
        message_hidden=int(dims.get("message_hidden", 96)),
        update_hidden=int(dims.get("update_hidden", 96)),
        q_hidden=int(dims.get("q_hidden", 96)),
        readout_hidden=int(dims.get("readout_hidden", 128)),
        readout_mid=int(dims.get("readout_mid", 64)),
        dropout=float(dims.get("dropout", 0.1)),
    )
    hashes = model.init_parameters(int(seed), float(dims.get("last_layer_std", 1e-3)))
    return model, hashes