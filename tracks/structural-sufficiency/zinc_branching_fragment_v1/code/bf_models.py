"""P3-BF: P3 Null model with typed rooted branching-fragment statistics.

The branch encodes every unique fragment content with one shared ``phi`` and
aggregates, per root and family, the occurrence-weighted mean, mean square,
count and existence.  ``s_v`` (198 dimensions) enters the P3 atom encoder as a
second linear term on the first layer:

    h_v = SiLU(W_x raw108_v + b + W_s s_v)
    a_v = head(h_v)

``W_s`` is zero-initialised and every inherited P3 tensor is loaded from the
P3B seed-0 supervised best-valid checkpoint, so the initial function is exactly
the supervised P3B.  The rest of P3 (bond encoder, junction encoder, Null
composer, reader, global33) is the unchanged P3 computation graph.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
_V2 = _HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_models as M1  # noqa: E402
import ad2_models as A2  # noqa: E402

P3B_CKPT = _HERE.parents[1] / "zinc_recipe_trunk_v2" / "runs" / "xBssl_P3N_s0" / "best.pt"
P3B_VALID = 0.07622375113883754

ATOM_CLASSES = 28
N_FAMILIES = 3
MAX_NODES = 5
N_PAIRS = 10
D_ATOM = 8
D_PAIR = 4
PAIR_J = tuple(j for _, j in ((i, j) for i in range(MAX_NODES) for j in range(i + 1, MAX_NODES)))
PHI_HIDDEN = 64
PHI_OUT = 32
S_DIM = N_FAMILIES * (2 * PHI_OUT + 2)  # 198


class BranchingFragmentBranch(nn.Module):
    """Shared content encoder phi + per-root/family weighted statistics."""

    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(ATOM_CLASSES, D_ATOM)
        self.pair_emb = nn.Embedding(4, D_PAIR)  # 0 = no edge, 1..3 = raw bond
        phi_in = MAX_NODES * D_ATOM + N_PAIRS * D_PAIR + MAX_NODES + N_PAIRS + N_FAMILIES  # 98
        self.phi = M1.MLP(phi_in, PHI_HIDDEN, PHI_OUT, 1, 0.0)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        pat_atom = batch["pat_atom"].long()
        pat_pair = batch["pat_pair"].long()
        pat_len = batch["pat_len"].long()
        pat_family = batch["pat_family"].long()
        n_atoms = int(batch["n_atoms_total"])
        P = int(pat_atom.shape[0])
        device = pat_atom.device
        dtype = self.atom_emb.weight.dtype

        node_mask = (
            torch.arange(MAX_NODES, device=device).unsqueeze(0) < pat_len.unsqueeze(1)
        ).to(dtype)
        pair_j = torch.as_tensor(PAIR_J, device=device)
        pair_mask = (pair_j.unsqueeze(0) < pat_len.unsqueeze(1)).to(dtype)
        family_onehot = F.one_hot(pat_family, num_classes=N_FAMILIES).to(dtype)

        if P:
            atom_e = self.atom_emb(pat_atom) * node_mask.unsqueeze(-1)
            pair_e = self.pair_emb(pat_pair) * pair_mask.unsqueeze(-1)
            x = torch.cat(
                [
                    atom_e.reshape(P, -1),
                    pair_e.reshape(P, -1),
                    node_mask,
                    pair_mask,
                    family_onehot,
                ],
                dim=1,
            )
            phi = self.phi(x)  # recomputed every forward, never cached detached
            occ_phi = phi[batch["occ_pat"].long()]
            occ_weight = batch["occ_count"].to(dtype).unsqueeze(1)
            flat = batch["occ_root"].long() * N_FAMILIES + batch["occ_family"].long()
            sums = phi.new_zeros((n_atoms * N_FAMILIES, PHI_OUT))
            sqs = phi.new_zeros((n_atoms * N_FAMILIES, PHI_OUT))
            counts = phi.new_zeros((n_atoms * N_FAMILIES,))
            sums.index_add_(0, flat, occ_phi * occ_weight)
            sqs.index_add_(0, flat, (occ_phi * occ_phi) * occ_weight)
            counts.index_add_(0, flat, batch["occ_count"].to(dtype))
        else:
            sums = torch.zeros((n_atoms * N_FAMILIES, PHI_OUT), device=device, dtype=dtype)
            sqs = torch.zeros((n_atoms * N_FAMILIES, PHI_OUT), device=device, dtype=dtype)
            counts = torch.zeros((n_atoms * N_FAMILIES,), device=device, dtype=dtype)

        n = counts.clamp(min=1.0)
        mean = sums / n.unsqueeze(-1)
        mean_sq = sqs / n.unsqueeze(-1)
        stats = torch.cat(
            [mean, mean_sq, torch.log1p(counts).unsqueeze(-1), (counts > 0).to(dtype).unsqueeze(-1)],
            dim=-1,
        )  # [n_atoms * 3, 66]
        return stats.reshape(n_atoms, S_DIM)


class BranchingFragmentModel(A2.P3NullModel):
    """Exact P3 graph with the fragment statistics as a second first-layer term."""

    def __init__(self, norm: Mapping[str, object], *, dropout: float = 0.0):
        super().__init__(norm, dropout=dropout)
        self.w_s = nn.Linear(S_DIM, 128, bias=False)
        nn.init.zeros_(self.w_s.weight)
        self.branch = BranchingFragmentBranch()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        s = self.branch(batch)
        first = self.encoders.atom_encoder.lins[0]
        a = F.silu(first(batch["atom_feat"]) + self.w_s(s))
        a = self.encoders.atom_encoder.lins[1](a)

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


def parameter_breakdown(model: BranchingFragmentModel) -> dict[str, int]:
    groups = {}
    for name in ("encoders.atom_encoder", "encoders.bond_encoder", "encoders.junction_encoder",
                 "group_composer", "reader", "w_s", "branch"):
        groups[name] = int(sum(p.numel() for p in model.get_submodule(name).parameters()))
    groups["total"] = int(sum(p.numel() for p in model.parameters()))
    return groups


def param_groups(model: BranchingFragmentModel):
    """Split named parameters into the inherited P3 group and the new group."""
    inherited, new = [], []
    for name, p in model.named_parameters():
        if name == "w_s.weight" or name.startswith("branch."):
            new.append((name, p))
        else:
            inherited.append((name, p))
    return inherited, new


def load_p3b(model: BranchingFragmentModel, path: Path = P3B_CKPT, device="cpu") -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    missing, unexpected = model.load_state_dict(
        {k: v.to(device) for k, v in state.items()}, strict=False
    )
    if unexpected:
        raise RuntimeError(f"unexpected P3B keys: {unexpected}")
    bad_missing = [k for k in missing if not (k == "w_s.weight" or k.startswith("branch."))]
    if bad_missing:
        raise RuntimeError(f"P3B does not cover inherited parameters: {bad_missing}")
    if model.w_s.weight.abs().sum().item() != 0.0:
        raise RuntimeError("W_s is not exactly zero after loading P3B")
    return {
        "source": str(path.relative_to(C.REPO)),
        "sha256": C.sha256_file(path),
        "checkpoint_epoch": int(ck["epoch"]),
        "checkpoint_valid_mae": float(ck["valid_mae"]),
        "missing_new_keys": sorted(missing),
        "unexpected_keys": unexpected,
        "w_s_zero": True,
    }


def build_model(norm: Mapping[str, object], seed: int = 0, device="cpu") -> tuple[BranchingFragmentModel, dict]:
    torch.manual_seed(int(seed))
    model = BranchingFragmentModel(norm)
    info = load_p3b(model, device=device)
    model = model.to(device)
    total = C.parameter_count(model)
    inherited, new = param_groups(model)
    info.update({
        "seed": int(seed),
        "parameters": total,
        "parameters_breakdown": parameter_breakdown(model),
        "parameters_inherited": int(sum(p.numel() for _, p in inherited)),
        "parameters_new": int(sum(p.numel() for _, p in new)),
        "parameter_cap": 500_000,
        "all_require_grad": bool(all(p.requires_grad for p in model.parameters())),
    })
    if total > 500_000:
        raise RuntimeError(f"P3-BF has {total} parameters > 500k")
    return model, info
