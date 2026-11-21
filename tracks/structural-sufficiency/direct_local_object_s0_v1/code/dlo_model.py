"""The fixed direct-local-object network.

One architecture for both arms (~0.33 M parameters):

    atom 9x8 -> 72 -> 64 -> 32
    bond 3x8 -> 24 -> 32 -> 16
    full 6-slot frame: 6*32 atom + 15*16 bond + masks/adj + size one-hot = 474
    474 -> 256 -> 128 -> LayerNorm
    per object: weighted frame average (weights sum to 1)
    per molecule: two buckets, sum/mu and sum/max(1,n), log counts, global counts
    516 -> 256 -> 128 -> 1

No dropout inside the frame path (identical inputs must give identical outputs
when averaged), no message passing, no teacher, no residual join, no zero-init
output layer.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import dlo_common as C


def _mlp(dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class DLOFrameModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], atom_dims: Sequence[int], bond_dims: Sequence[int],
                 atom_code, bond_code) -> None:
        super().__init__()
        d = cfg["dims"]
        self.cfg = cfg
        self.n_atom_cols = len(atom_dims)
        self.n_bond_cols = len(bond_dims)
        self.atom_emb = nn.ModuleList([nn.Embedding(int(x), int(d["atom_cat_dim"])) for x in atom_dims])
        self.atom_mlp = _mlp([len(atom_dims) * int(d["atom_cat_dim"]), int(d["atom_hidden"]), int(d["atom_out"])])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(x), int(d["bond_cat_dim"])) for x in bond_dims])
        self.bond_mlp = _mlp([len(bond_dims) * int(d["bond_cat_dim"]), int(d["bond_hidden"]), int(d["bond_out"])])
        self.frame_encoder = nn.Sequential(
            nn.Linear(int(d["frame_input"]), int(d["frame_hidden"])),
            nn.SiLU(),
            nn.Linear(int(d["frame_hidden"]), int(d["frame_out"])),
            nn.LayerNorm(int(d["frame_out"])),
        )
        self.graph_head = nn.Sequential(
            nn.Linear(int(d["graph_input"]), int(d["graph_hidden1"])),
            nn.SiLU(),
            nn.Dropout(float(d["graph_dropout"])),
            nn.Linear(int(d["graph_hidden1"]), int(d["graph_hidden2"])),
            nn.SiLU(),
            nn.Dropout(float(d["graph_dropout"])),
            nn.Linear(int(d["graph_hidden2"]), 1),
        )
        self.register_buffer("atom_code", torch.as_tensor(atom_code, dtype=torch.uint8), persistent=True)
        self.register_buffer("bond_code", torch.as_tensor(bond_code, dtype=torch.uint8), persistent=True)
        self.register_buffer("mu_bucket", torch.ones(2, dtype=torch.float32), persistent=True)
        if int(d["frame_input"]) != int(d["slots"]) * int(d["atom_out"]) + int(d["pairs"]) * int(d["bond_out"]) \
                + 2 * int(d["slots"]) + 2 * int(d["pairs"]):
            raise ValueError("frame input dimension does not match the 474 definition")
        if int(d["graph_input"]) != 4 * int(d["frame_out"]) + 4:
            raise ValueError("graph input dimension does not match the 516 definition")

    # ------------------------------------------------------------------ frames
    def encode_frames(self, atom_attr: torch.Tensor, bond_attr: torch.Tensor,
                      node_mask: torch.Tensor, edge_mask: torch.Tensor,
                      adjacency: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        atom_raw = self.atom_code[atom_attr.long()]              # [F, 6, 9] uint8
        parts = [emb(atom_raw[..., j].long()) for j, emb in enumerate(self.atom_emb)]
        a = self.atom_mlp(torch.cat(parts, dim=-1))              # [F, 6, 32]
        a = a * node_mask.unsqueeze(-1)
        # 65535 is "no bond"; row 0 is a valid bond attribute, so mask after the MLP
        bond_idx = bond_attr.long().clamp(max=int(self.bond_code.shape[0]) - 1)
        bond_raw = self.bond_code[bond_idx]                       # [F, 15, 3]
        parts = [emb(bond_raw[..., j].long()) for j, emb in enumerate(self.bond_emb)]
        e = self.bond_mlp(torch.cat(parts, dim=-1))              # [F, 15, 16]
        e = e * edge_mask.unsqueeze(-1)
        size_onehot = F.one_hot(size.long() - 1, int(self.cfg["dims"]["slots"])).to(a.dtype)
        x = torch.cat([
            a.flatten(1),
            e.flatten(1),
            node_mask,
            edge_mask,
            adjacency,
            size_onehot,
        ], dim=1)
        if x.shape[1] != int(self.cfg["dims"]["frame_input"]):
            raise AssertionError("frame input width changed")
        return self.frame_encoder(x)

    # ------------------------------------------------------------------- graph
    def forward_packed(self, b: Mapping[str, Any], zero_objects: bool = False) -> torch.Tensor:
        B = int(b["n_molecules"])
        n_obj = int(b["n_objects"])
        F_out = int(self.cfg["dims"]["frame_out"])
        H = self.encode_frames(
            b["frame_atom_attr"], b["frame_bond_attr"], b["frame_node_mask"],
            b["frame_edge_mask"], b["frame_adj"], b["frame_k"],
        ) if int(b["n_frames"]) > 0 else b["frame_weight"].new_zeros((0, F_out))
        h_frames = H[b["frame_inverse"]] * b["frame_weight"].unsqueeze(-1)
        h_obj = h_frames.new_zeros((n_obj, F_out))
        h_obj.index_add_(0, b["frame_to_obj"], h_frames)
        obj_to_mol = b["obj_to_mol"]
        if zero_objects:
            sum0 = h_obj.new_zeros((B, F_out))
            sum1 = h_obj.new_zeros((B, F_out))
        else:
            sum_all = h_obj.new_zeros((B, F_out)).index_add_(0, obj_to_mol, h_obj)
            m1 = b["obj_bucket"].bool()
            sum1 = h_obj.new_zeros((B, F_out)).index_add_(0, obj_to_mol[m1], h_obj[m1])
            sum0 = sum_all - sum1
        n_bucket = b["n_bucket"].to(h_obj.dtype)
        n0 = n_bucket[:, 0]
        n1 = n_bucket[:, 1]
        mu = self.mu_bucket.to(h_obj.dtype)
        u0 = sum0 / mu[0]
        v0 = sum0 / torch.clamp(n0, min=1.0).unsqueeze(-1)
        u1 = sum1 / mu[1]
        v1 = sum1 / torch.clamp(n1, min=1.0).unsqueeze(-1)
        logn0 = torch.log1p(n0).unsqueeze(-1)
        logn1 = torch.log1p(n1).unsqueeze(-1)
        log_atoms = torch.log1p(b["n_atoms"].to(h_obj.dtype)).unsqueeze(-1)
        log_bonds = torch.log1p(b["n_bonds"].to(h_obj.dtype)).unsqueeze(-1)
        graph = torch.cat([u0, v0, logn0, u1, v1, logn1, log_atoms, log_bonds], dim=1)
        if graph.shape[1] != int(self.cfg["dims"]["graph_input"]):
            raise AssertionError("graph input width changed")
        return self.graph_head(graph).squeeze(-1)

    # ---------------------------------------------------------------- bookkeeping
    def parameter_ledger(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            head = name.split(".")[0]
            groups[head] = groups.get(head, 0) + int(p.numel())
        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "by_module": groups,
            "trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }


def build_model(cfg: Mapping[str, Any], payload: Mapping[str, Any], *, seed: int = 0) -> DLOFrameModel:
    atom_dims, bond_dims = C.official_feature_dims()
    torch.manual_seed(int(seed))
    model = DLOFrameModel(cfg, atom_dims, bond_dims, payload["atom_code"], payload["bond_code"])
    return model


def apply_train_support(model: DLOFrameModel, stats: Mapping[str, Any]) -> dict[str, Any]:
    atom_dims, bond_dims = C.official_feature_dims()
    report: dict[str, Any] = {"atom": [], "bond": []}
    with torch.no_grad():
        for j, emb in enumerate(model.atom_emb):
            seen = set(int(v) for v in stats["train_support"]["atom"][j])
            missing = [v for v in range(int(atom_dims[j])) if v not in seen]
            for v in missing:
                emb.weight[v].zero_()
            report["atom"].append({
                "column": j,
                "n_categories": int(atom_dims[j]),
                "n_seen_train": len(seen),
                "zeroed": missing,
            })
        for j, emb in enumerate(model.bond_emb):
            seen = set(int(v) for v in stats["train_support"]["bond"][j])
            missing = [v for v in range(int(bond_dims[j])) if v not in seen]
            for v in missing:
                emb.weight[v].zero_()
            report["bond"].append({
                "column": j,
                "n_categories": int(bond_dims[j]),
                "n_seen_train": len(seen),
                "zeroed": missing,
            })
    return report


def bind_constants(model: DLOFrameModel, stats: Mapping[str, Any], arm: str) -> None:
    mu = stats["arms"][arm]["mu_bucket"]
    with torch.no_grad():
        model.mu_bucket.copy_(torch.as_tensor(mu, dtype=torch.float32))
