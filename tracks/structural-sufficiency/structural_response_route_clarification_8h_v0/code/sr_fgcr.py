"""FGCR response variants: ``H_FULL | H_IND`` x ``ETA_FULL | ETA_STAR``.

The frozen historical model (``full_graph_cavity_response_pilot_v0``) is reused
byte-for-byte: encoder, potentials, ``eta_norm``, sentence MLP, readout and
scalar bias.  Only the two tensors fed into ``eta_norm`` / the readout are
replaced, exactly as the frozen increment audit defined them::

    log p_i(s)     = log_softmax(u_i)(s)
    H_FULL         = logZ - sum_i logsumexp_s u_i(s)
    H_IND          = sum_(i,j) logsumexp_(s,t)[log p_i(s)+log p_j(t)+psi_ij(s,t)]
    ETA_FULL_i     = posterior_logodds_i - (u_i(1)-u_i(0))
    m_(j->i)(s)    = logsumexp_t[log p_j(t)+psi_ij(s,t)]
    ETA_STAR_i     = sum_j [m_(j->i)(1) - m_(j->i)(0)]

Both H slots of the original readout (H and H/(1+n_bonds)) are replaced
together.  The exact belief pass runs at most once per forward and only when
``H_FULL`` or ``ETA_FULL`` is requested.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sr_common as C

if str(C.FGCR_CODE) not in sys.path:
    sys.path.insert(0, str(C.FGCR_CODE))

import fgcr_inference as I  # noqa: E402
import fgcr_model as M  # noqa: E402


# ---------------------------------------------------------------------------
# local reference responses (identical tensor algebra to the frozen audit)
# ---------------------------------------------------------------------------


def local_references(u: torch.Tensor, log_psi: torch.Tensor,
                     bond_u: torch.Tensor, bond_v: torch.Tensor,
                     bond_mol: torch.Tensor, n_molecules: int, channels: int
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    log_p = torch.log_softmax(u, dim=-1)
    log_p_u = log_p.index_select(0, bond_u)
    log_p_v = log_p.index_select(0, bond_v)
    n_bonds = int(log_p_u.shape[0])
    if n_bonds == 0:
        zero_h = u.new_zeros((int(n_molecules), channels))
        zero_e = u.new_zeros((int(u.shape[0]), channels))
        return zero_h, zero_e
    edge_logits = log_p_u.unsqueeze(-1) + log_p_v.unsqueeze(-2) + log_psi
    h_edge = torch.logsumexp(edge_logits.reshape(n_bonds, channels, 4), dim=-1)
    h_ind = u.new_zeros((int(n_molecules), channels)).index_add_(0, bond_mol, h_edge)
    m_v_to_u = torch.logsumexp(log_p_v.unsqueeze(-2) + log_psi, dim=-1)
    m_u_to_v = torch.logsumexp(log_p_u.unsqueeze(-1) + log_psi, dim=-2)
    eta_star = u.new_zeros((int(u.shape[0]), channels))
    eta_star.index_add_(0, bond_u, m_v_to_u[..., 1] - m_v_to_u[..., 0])
    eta_star.index_add_(0, bond_v, m_u_to_v[..., 1] - m_u_to_v[..., 0])
    return h_ind, eta_star


class FGCRVariant(M.FGCRModel):
    """FGCR-v0 with an explicit ``(H, eta)`` response-mode pair."""

    def __init__(self, *args: Any, h_mode: str = "FULL", eta_mode: str = "FULL",
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if h_mode not in ("FULL", "IND") or eta_mode not in ("FULL", "STAR"):
            raise ValueError((h_mode, eta_mode))
        self.h_mode = h_mode
        self.eta_mode = eta_mode
        self.belief_calls = 0

    def forward(self, batch: Mapping[str, Any], *, return_aux: bool = False
                ) -> dict[str, Any]:
        atom_in, u, j, log_psi = self.potentials(batch)
        channels = self.channels
        n_mol = int(batch["n_molecules"])

        unary_lse = torch.logsumexp(u, dim=-1)
        unary_sum = u.new_zeros((n_mol, channels))
        unary_sum.index_add_(0, batch["atom_mol"], unary_lse)
        unary_logodds = u[..., 1] - u[..., 0]

        need_local = (self.h_mode == "IND") or (self.eta_mode == "STAR")
        h_ind, eta_star = (None, None)
        if need_local:
            h_ind, eta_star = local_references(
                u, log_psi, batch["bond_u"], batch["bond_v"], batch["bond_mol"],
                n_mol, channels)

        need_belief = (self.h_mode == "FULL") or (self.eta_mode == "FULL")
        responses = None
        if need_belief:
            responses = I.exact_responses(u, log_psi, batch["plan"])
            self.belief_calls += 1

        if self.h_mode == "FULL":
            H = responses["logz"] - unary_sum
        else:
            H = h_ind
        if self.eta_mode == "FULL":
            eta = (responses["logw"][..., 1] - responses["logw"][..., 0]) - unary_logodds
        else:
            eta = eta_star

        sentence_in = torch.cat([atom_in, self.eta_norm(eta)], dim=1)
        h = self.sentence_mlp(sentence_in)

        n_atoms = batch["graph_n_atoms"].unsqueeze(1)
        n_bonds = batch["graph_n_bonds"].unsqueeze(1)
        h_sum = h.new_zeros((n_mol, h.shape[1]))
        h_sum.index_add_(0, batch["atom_mol"], h)
        h_sumsq = h.new_zeros((n_mol, h.shape[1]))
        h_sumsq.index_add_(0, batch["atom_mol"], h * h)
        h_mean = h_sum / n_atoms
        h_var = (h_sumsq / n_atoms - h_mean * h_mean).clamp_min(0.0)
        h_std = torch.sqrt(h_var)

        readout_in = torch.cat(
            [h_sum, h_mean, h_std, H, H / (1.0 + n_bonds),
             torch.log1p(n_atoms), torch.log1p(n_bonds)], dim=1)
        if readout_in.shape[1] != self.readout_in_dim:
            raise RuntimeError(f"readout width {readout_in.shape[1]} != {self.readout_in_dim}")
        yhat_std = self.readout_mlp(readout_in).view(-1) + self.beta

        out: dict[str, Any] = {"yhat_std": yhat_std}
        if return_aux:
            out.update({"u": u, "j": j, "log_psi": log_psi, "H": H, "eta": eta,
                        "h": h, "readout_in": readout_in,
                        "H_ind": h_ind, "eta_star": eta_star})
        return out


def model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("model", {}))
    return dict(
        channels=int(cfg.get("channels", 16)),
        states=int(cfg.get("states", 2)),
        atom_emb=int(cfg.get("atom_emb", 16)),
        bond_emb=int(cfg.get("bond_emb", 12)),
        atom_hidden=int(cfg.get("atom_potential_dims", [144, 64, 32])[1]),
        bond_hidden=int(cfg.get("bond_potential_dims", [36, 64, 48])[1]),
        potential_bound=float(cfg.get("potential_bound", 2.0)),
        sentence_hidden=int(cfg.get("sentence_dims", [160, 128, 96])[1]),
        h_dim=int(cfg.get("sentence_dims", [160, 128, 96])[2]),
        readout_hidden=int(cfg.get("readout_dims", [322, 128, 64, 1])[1]),
        readout_mid=int(cfg.get("readout_dims", [322, 128, 64, 1])[2]),
        dropout=float(cfg.get("sentence_dropout", 0.1)),
    )


def historical_config() -> dict[str, Any]:
    return C.read_json(C.FGCR_CONFIG)


def build_variant(atom_dims: Sequence[int], bond_dims: Sequence[int], *,
                  h_mode: str, eta_mode: str, seed: int = 0,
                  config: Mapping[str, Any] | None = None,
                  init_state: Mapping[str, torch.Tensor] | None = None,
                  device: Any = "cpu") -> tuple[FGCRVariant, dict[str, Any]]:
    cfg = dict(config or historical_config())
    kwargs = model_kwargs(cfg)
    model = FGCRVariant(list(atom_dims), list(bond_dims), h_mode=h_mode,
                        eta_mode=eta_mode, **kwargs)
    if init_state is None:
        init_hash = model.init_parameters(int(seed))
    else:
        model.load_state_dict({k: v.clone() for k, v in init_state.items()}, strict=True)
        init_hash = model.parameter_hash()
    model = model.to(device)
    info = {
        "parameters": model.parameter_ledger(),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "init_hash": init_hash,
        "readout_in_dim": int(model.readout_in_dim),
        "h_mode": h_mode, "eta_mode": eta_mode,
    }
    return model, info


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def collate_rows(bundle, rows: Sequence[int], plan, row_to_plan: Mapping[int, int],
                 device: Any = "cpu") -> dict[str, Any]:
    rows_list = [int(r) for r in rows]
    atom_counts = [bundle.n_atoms(r) for r in rows_list]
    bond_counts = [bundle.n_bonds(r) for r in rows_list]
    plan_rows = [int(row_to_plan[r]) for r in rows_list]
    bp = plan.select(plan_rows, atom_counts, bond_counts).to(device)

    atom_parts, bond_parts, bu_parts, bv_parts = [], [], [], []
    node_off = 0
    for row in rows_list:
        a0, a1 = int(bundle.node_ptr[row]), int(bundle.node_ptr[row + 1])
        b0, b1 = int(bundle.bond_ptr[row]), int(bundle.bond_ptr[row + 1])
        atom_parts.append(bundle.atom_feat[a0:a1])
        bond_parts.append(bundle.bond_feat[b0:b1])
        bu_parts.append(bundle.bond_u[b0:b1] + node_off)
        bv_parts.append(bundle.bond_v[b0:b1] + node_off)
        node_off += a1 - a0

    atom_feat = np.concatenate(atom_parts, axis=0)
    bond_feat = np.concatenate(bond_parts, axis=0) if bond_parts else np.zeros((0, C.BOND_COLUMNS), np.int64)
    bond_u = np.concatenate(bu_parts) if bu_parts else np.zeros(0, np.int64)
    bond_v = np.concatenate(bv_parts) if bv_parts else np.zeros(0, np.int64)
    atom_mol = np.repeat(np.arange(len(rows_list), dtype=np.int64), np.asarray(atom_counts))
    bond_mol = np.repeat(np.arange(len(rows_list), dtype=np.int64), np.asarray(bond_counts))

    if bp.n_atoms != int(atom_feat.shape[0]) or bp.n_bonds != int(bond_feat.shape[0]):
        raise RuntimeError("batch plan atom/bond count mismatch")

    batch = {
        "atom_feat": torch.as_tensor(atom_feat, dtype=torch.long, device=device),
        "bond_feat": torch.as_tensor(bond_feat, dtype=torch.long, device=device),
        "atom_mol": torch.as_tensor(atom_mol, dtype=torch.long, device=device),
        "bond_mol": torch.as_tensor(bond_mol, dtype=torch.long, device=device),
        "bond_u": torch.as_tensor(bond_u, dtype=torch.long, device=device),
        "bond_v": torch.as_tensor(bond_v, dtype=torch.long, device=device),
        "n_atoms": int(atom_feat.shape[0]),
        "n_bonds": int(bond_feat.shape[0]),
        "n_molecules": len(rows_list),
        "graph_n_atoms": torch.as_tensor(atom_counts, dtype=torch.float32, device=device),
        "graph_n_bonds": torch.as_tensor(bond_counts, dtype=torch.float32, device=device),
        "plan": bp,
        "row_ids": torch.as_tensor(np.asarray(rows_list, dtype=np.int64),
                                   dtype=torch.long, device=device),
    }
    return batch


@torch.no_grad()
def predict_std(model: FGCRVariant, bundle, plan, row_to_plan: Mapping[int, int], rows: np.ndarray,
                batch_size: int = 256, device: Any = "cpu") -> np.ndarray:
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros(rows.size, dtype=np.float64)
    offset = 0
    for start in range(0, rows.size, int(batch_size)):
        part = rows[start:start + int(batch_size)]
        batch = collate_rows(bundle, part.tolist(), plan, row_to_plan, device)
        out[offset:offset + part.size] = model(batch)["yhat_std"].double().cpu().numpy()
        offset += int(part.size)
    return out
