"""J/Q models for the rooted induced-path pilot.

``J`` keeps the full typed induced adjacency on every rooted simple path.
``Q`` keeps exactly the same path objects and multiplicities but replaces every
non-consecutive pair type by ``0`` (no edge), i.e. only the typed path sequence.

Both models share the P3 atom/bond/junction encoders, the P3 Null composer and
the P3 reader.  A per-atom increment ``d_v`` is computed from the occurrence
weighted pattern statistics and added to the P3 atom encoding **before** the
bond encoder, junction encoder and Null composer are recomputed.  The final
projection of the path branch is zero-initialised, so the initial forward is
exactly the same SSL-start P3 forward; after the first updates the projection
opens and the pattern encoder receives gradients.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad2_models as A2  # noqa: E402


class RootedPathBranch(nn.Module):
    """Encode unique raw patterns once, then scatter to root/length statistics."""

    def __init__(
        self,
        *,
        out_dim: int = 96,
        d_atom: int = 12,
        d_pos: int = 4,
        d_pair: int = 6,
        d_pair_pos: int = 4,
        phi_hidden: int = 80,
        phi_out: int = 32,
        proj_hidden: int = 128,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.phi_out = int(phi_out)
        self.atom_emb = nn.Embedding(D.N_ATOM_TYPES, int(d_atom))
        self.pos_emb = nn.Parameter(torch.randn(6, int(d_pos)) * 0.02)
        self.pair_emb = nn.Embedding(4, int(d_pair))  # 0 = no edge, 1..3 = raw bond type
        self.pair_pos_emb = nn.Parameter(torch.randn(15, int(d_pair_pos)) * 0.02)
        phi_in = 6 * (int(d_atom) + int(d_pos)) + 15 * (int(d_pair) + int(d_pair_pos)) + 6 + 15 + 1
        self.phi = M1.MLP(phi_in, int(phi_hidden), int(phi_out), 1, 0.0)
        # [mean(phi), mean(phi^2), log1p(count), existence mask] x 4 lengths
        proj_in = 4 * (2 * int(phi_out) + 2)
        self.proj = M1.MLP(proj_in, int(proj_hidden), int(out_dim), 1, 0.0)
        # The zero increment at initialisation is part of the declared interface.
        nn.init.zeros_(self.proj.lins[-1].weight)
        nn.init.zeros_(self.proj.lins[-1].bias)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        pat_atom = batch["pat_atom"].long()
        pat_pair = batch["pat_pair"].long()
        pat_len = batch["pat_len"].long()
        node_mask = batch["pat_node_mask"].float()
        pair_mask = batch["pat_pair_mask"].float()
        n_atoms = int(batch["n_atoms_total"])
        P = int(pat_atom.shape[0])
        dev = pat_atom.device
        dtype = self.pos_emb.dtype

        if P:
            atom_e = self.atom_emb(pat_atom) * node_mask.unsqueeze(-1)
            pos_e = self.pos_emb.unsqueeze(0) * node_mask.unsqueeze(-1)
            pair_e = self.pair_emb(pat_pair) * pair_mask.unsqueeze(-1)
            pair_pos_e = self.pair_pos_emb.unsqueeze(0) * pair_mask.unsqueeze(-1)
            length_scalar = ((pat_len.to(dtype) - 2.0) / 3.0).unsqueeze(1)
            x = torch.cat(
                [
                    atom_e.reshape(P, -1),
                    pos_e.reshape(P, -1),
                    pair_e.reshape(P, -1),
                    pair_pos_e.reshape(P, -1),
                    node_mask,
                    pair_mask,
                    length_scalar,
                ],
                dim=1,
            )
            phi = self.phi(x)  # [P, phi_out], recomputed every forward
            sums = phi.new_zeros((n_atoms, 4, self.phi_out))
            sqs = phi.new_zeros((n_atoms, 4, self.phi_out))
            counts = phi.new_zeros((n_atoms, 4))
            flat_idx = batch["occ_root"].long() * 4 + batch["occ_len"].long()
            occ_phi = phi[batch["occ_pat"].long()]
            occ_weight = batch["occ_count"].to(dtype).unsqueeze(1)
            sums.reshape(n_atoms * 4, self.phi_out).index_add_(0, flat_idx, occ_phi * occ_weight)
            sqs.reshape(n_atoms * 4, self.phi_out).index_add_(0, flat_idx, (occ_phi * occ_phi) * occ_weight)
            counts.view(n_atoms * 4).index_add_(0, flat_idx, batch["occ_count"].to(dtype))
        else:
            sums = torch.zeros((n_atoms, 4, self.phi_out), device=dev, dtype=dtype)
            sqs = torch.zeros((n_atoms, 4, self.phi_out), device=dev, dtype=dtype)
            counts = torch.zeros((n_atoms, 4), device=dev, dtype=dtype)

        n = counts.clamp(min=1.0)
        mean = sums / n.unsqueeze(-1)
        mean_sq = sqs / n.unsqueeze(-1)
        exist = (counts > 0).to(dtype)
        stats = torch.cat(
            [mean, mean_sq, torch.log1p(counts).unsqueeze(-1), exist.unsqueeze(-1)], dim=-1
        )
        d = self.proj(stats.reshape(n_atoms, -1))
        has_any = (counts.sum(dim=1) > 0).to(dtype).unsqueeze(1)
        return d * has_any


class RootedPathModel(A2.P3NullModel):
    """P3 Null graph with the rooted induced-path increment added before bonds."""

    def __init__(self, norm: Mapping[str, object], *, variant: str, dropout: float = 0.0):
        super().__init__(norm, dropout=dropout)
        if variant not in ("J", "Q"):
            raise ValueError(f"unknown path variant {variant!r}")
        self.variant = variant
        self.path_branch: RootedPathBranch | None = None

    def attach_path_branch(self) -> None:
        if self.path_branch is not None:
            raise RuntimeError("path branch already attached")
        self.path_branch = RootedPathBranch()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.path_branch is None:
            raise RuntimeError("path branch has not been attached")
        a0 = self.encoders.atom_encoder(batch["atom_feat"])
        d = self.path_branch(batch)
        a = a0 + d

        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bond_feat = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        b = self.encoders.bond_encoder(bond_feat)

        ji = batch["junc_index"]
        jb = batch["junc_bond"]
        ai, aj, ak = a[ji[:, 0]], a[ji[:, 1]], a[ji[:, 2]]
        b1, b2 = b[jb[:, 0]], b[jb[:, 1]]
        junc_feat = torch.cat(
            [ai + ak, (ai - ak).abs(), ai * ak, aj, b1 + b2, (b1 - b2).abs(), batch["junc_feat"]],
            dim=1,
        )
        w = self.encoders.junction_encoder(junc_feat)

        W, V, n_b = self.group_to_bonds(w, batch)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = c.new_zeros((n_graphs, c.shape[1])).index_add_(0, batch["bond_batch"], c)
        a_sum = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"].to(a.dtype) - self.global_mean) / self.global_scale
        return self.reader(torch.cat([z, a_sum, a_sq, g], dim=1)).view(-1)


def parameter_breakdown(model: RootedPathModel) -> dict[str, int]:
    groups = {}
    for name in ("encoders.atom_encoder", "encoders.bond_encoder", "encoders.junction_encoder",
                 "group_composer", "reader", "path_branch"):
        groups[name] = int(sum(p.numel() for p in model.get_submodule(name).parameters()))
    groups["total"] = int(sum(p.numel() for p in model.parameters()))
    return groups


def build_model(candidate: str, norm: Mapping[str, object], seed: int, device):
    """Build the P3 modules first, load SSL, then create the new path branch."""
    if candidate not in ("J", "Q"):
        raise ValueError(f"unknown induced-pattern candidate {candidate!r}")
    torch.manual_seed(int(seed))
    model = RootedPathModel(norm, variant=candidate)
    model = model.to(device)
    path = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
    if path.exists():
        ssl_info = A2._load_encoders(model, path, device)
    else:
        ssl_info = {"source": None, "note": "frozen SSL encoder unavailable; random init"}
    # Deliberately after the encoder state has been loaded: old P3 parameters
    # keep the same seed-derived starting point as the reference builder.
    model.attach_path_branch()
    model.path_branch.to(device)
    n_params = C.parameter_count(model)
    if n_params > 500_000:
        raise RuntimeError(f"{candidate}: {n_params} parameters exceeds the 500k cap")
    return model, {
        "candidate": candidate,
        "variant": candidate,
        "ssl": ssl_info,
        "parameters": n_params,
        "parameters_breakdown": parameter_breakdown(model),
        "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
        "parameters_total": sum(1 for p in model.parameters()),
    }
