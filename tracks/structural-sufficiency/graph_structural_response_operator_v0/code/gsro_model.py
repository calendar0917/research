"""GSRO v0 models: one shared parameter generator + two response operators.

FULL   : per graph and channel solve ``K x = s`` with ``K = R + L_w``.
LOCAL2 : closed-form 0/1/2 truncation of the same normalized inverse operator.
Readout: ``y_hat_std = b + sum_k (alpha_k/n_bar + beta_k/n) * Q_k``.

Both arms use *exactly* the same generator, initialisation, data and optimiser;
only the response computation differs.  Everything is fp32 in training; the
reference/checks/explanation paths run the same functions in fp64.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import gsro_common as C


# ---------------------------------------------------------------------------
# response operators (functional, no learned module here)
# ---------------------------------------------------------------------------


def masked_diag(
    r: torch.Tensor,
    w: torch.Tensor,
    bu: torch.Tensor,
    bv: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``D_v = r_v + sum_incident(w_e)``, plus the masked edge weights.

    Padding: ``r`` and ``w`` are zero on padding, and the diagonal of padding
    nodes is set to 1 so the padded block is identity and inert.
    """
    wm = w * edge_mask.unsqueeze(-1).to(w.dtype)
    d = r * node_mask.unsqueeze(-1).to(r.dtype)
    channels = wm.shape[-1]
    index_u = bu.unsqueeze(-1).expand(-1, -1, channels)
    index_v = bv.unsqueeze(-1).expand(-1, -1, channels)
    d = d.scatter_add(1, index_u, wm).scatter_add(1, index_v, wm)
    d = d + (~node_mask).unsqueeze(-1).to(d.dtype)
    return d, wm, r * node_mask.unsqueeze(-1).to(r.dtype)


def assemble_system(
    d: torch.Tensor,
    wm: torch.Tensor,
    bu: torch.Tensor,
    bv: torch.Tensor,
) -> torch.Tensor:
    """Dense batched ``K = diag(d) - A_w`` of shape [B, K, N, N]."""
    batch, n_nodes, channels = d.shape
    k_mat = torch.diag_embed(d.transpose(1, 2))
    edge_weights = wm.transpose(1, 2)
    b_idx = torch.arange(batch, device=d.device).view(batch, 1, 1)
    k_idx = torch.arange(channels, device=d.device).view(1, channels, 1)
    u_idx = bu.unsqueeze(1)
    v_idx = bv.unsqueeze(1)
    offsets = torch.zeros_like(k_mat)
    offsets[b_idx, k_idx, u_idx, v_idx] = -edge_weights
    offsets[b_idx, k_idx, v_idx, u_idx] = -edge_weights
    return k_mat + offsets


def full_response(
    s: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    bu: torch.Tensor,
    bv: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    return_x: bool = False,
    return_info: bool = False,
) -> Any:
    """Exact direct solution of ``K x = s`` per graph and channel."""
    d, wm, _ = masked_diag(r, w, bu, bv, node_mask, edge_mask)
    k_mat = assemble_system(d, wm, bu, bv)
    s_masked = s * node_mask.unsqueeze(-1).to(s.dtype)
    rhs = s_masked.transpose(1, 2).unsqueeze(-1)
    chol, info = torch.linalg.cholesky_ex(k_mat, check_errors=False)
    bad = int((info != 0).sum().item())
    if bad:  # pragma: no cover - defensive fallback, never expected
        jitter = torch.eye(k_mat.shape[-1], device=k_mat.device, dtype=k_mat.dtype) * 1e-6
        x = torch.linalg.solve_ex(k_mat + jitter, rhs, check_errors=False)[0]
    else:
        x = torch.cholesky_solve(rhs, chol)
    x = x.squeeze(-1).transpose(1, 2)
    q = (s_masked * x).sum(dim=1)
    if return_x and return_info:
        return q, x, {"n_failed_cholesky": bad}
    if return_x:
        return q, x
    if return_info:
        return q, {"n_failed_cholesky": bad}
    return q


def full_residual(
    s: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    bu: torch.Tensor,
    bv: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the dense system, solution and residual diagnostics."""
    d, wm, _ = masked_diag(r, w, bu, bv, node_mask, edge_mask)
    k_mat = assemble_system(d, wm, bu, bv)
    s_masked = s * node_mask.unsqueeze(-1).to(s.dtype)
    rhs = s_masked.transpose(1, 2).unsqueeze(-1)
    chol, info = torch.linalg.cholesky_ex(k_mat, check_errors=False)
    x = torch.cholesky_solve(rhs, chol) if not int((info != 0).sum()) else torch.linalg.solve(k_mat, rhs)
    residual = k_mat @ x - rhs
    return k_mat, x, residual, info


def local2_response(
    s: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    bu: torch.Tensor,
    bv: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    return_parts: bool = False,
) -> Any:
    """Explicit 0/1/2 truncation; no matrix solve.

    ``Q0 = sum s^2/D``, ``Q1 = 2 sum_e w t_u t_v``, ``Q2 = sum p^2/D`` with
    ``t = s/D`` and ``p_v = sum_u w_uv t_u`` (one sparse algebraic aggregation).
    """
    d, wm, _ = masked_diag(r, w, bu, bv, node_mask, edge_mask)
    channels = s.shape[-1]
    s = s * node_mask.unsqueeze(-1).to(s.dtype)
    t = s / d
    index_u = bu.unsqueeze(-1).expand(-1, -1, channels)
    index_v = bv.unsqueeze(-1).expand(-1, -1, channels)
    t_u = t.gather(1, index_u)
    t_v = t.gather(1, index_v)
    p = torch.zeros_like(s).scatter_add(1, index_u, wm * t_v).scatter_add(
        1, index_v, wm * t_u
    )
    q0 = (s * t).sum(dim=1)
    q1 = 2.0 * (wm * t_u * t_v).sum(dim=1)
    q2 = (p * p / d).sum(dim=1)
    q = q0 + q1 + q2
    if return_parts:
        return q, {"Q0": q0, "Q1": q1, "Q2": q2}
    return q


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------


class Readout(nn.Module):
    """Two shared linear coefficients per channel, size-normalised."""

    def __init__(self, channels: int, init_std: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(channels) * init_std)
        self.beta = nn.Parameter(torch.randn(channels) * init_std)
        self.bias = nn.Parameter(torch.zeros(1))

    def coefficients(self, n: torch.Tensor, n_bar: float) -> torch.Tensor:
        return self.alpha.unsqueeze(0) / n_bar + self.beta.unsqueeze(0) / n.unsqueeze(-1)

    def forward(
        self,
        q: torch.Tensor,
        n: torch.Tensor,
        n_bar: float,
        mu: float,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coeff = self.coefficients(n, n_bar)
        pred_std = self.bias + (coeff * q).sum(dim=-1)
        pred = mu + sigma * pred_std
        return pred, pred_std


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


class GSROModel(nn.Module):
    def __init__(self, prep, config: dict[str, Any], arm: str) -> None:
        super().__init__()
        if arm not in ("full", "local2"):
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.config = dict(config)
        self.n_bar = float(prep.n_bar)
        self.mu_y = float(prep.mu_y)
        self.sigma_y = float(prep.sigma_y)
        self.channels = int(config["channels"])
        atom_dim = int(config["atom_embed_dim"])
        bond_dim = int(config["bond_embed_dim"])
        node_h0, node_h1 = (int(config["node_hidden"][0]), int(config["node_hidden"][1]))
        edge_h0, edge_h1 = (int(config["edge_hidden"][0]), int(config["edge_hidden"][1]))
        if (node_h0, node_h1) != (128, 64) or (edge_h0, edge_h1) != (128, 64):
            raise ValueError("this round fixes hidden widths to node 128/64 and edge 128/64")
        if node_h1 != edge_h1:
            raise ValueError("node and edge representations must both be 64-d")

        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(size, atom_dim) for size in prep.atom_vocab_sizes]
        )
        self.bond_embeddings = nn.ModuleList(
            [nn.Embedding(size, bond_dim) for size in prep.bond_vocab_sizes]
        )
        node_input_dim = len(prep.atom_vocab_sizes) * atom_dim + prep.n_joint_features
        edge_input_dim = 3 * node_h1 + len(prep.bond_vocab_sizes) * bond_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, node_h0),
            nn.SiLU(),
            nn.Linear(node_h0, node_h1),
            nn.SiLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_h0),
            nn.SiLU(),
            nn.Linear(edge_h0, edge_h1),
            nn.SiLU(),
        )
        self.source_head = nn.Linear(node_h1, self.channels)
        self.self_head = nn.Linear(node_h1, self.channels)
        self.edge_head = nn.Linear(edge_h1, self.channels)
        self.readout = Readout(self.channels, float(config["readout_init_std"]))
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self._init_heads()

    # -- initialisation -----------------------------------------------------

    def _init_heads(self) -> None:
        cfg = self.config
        std = float(cfg["head_init_std"])
        with torch.no_grad():
            self.source_head.weight.normal_(0.0, float(cfg["source_head_init_std"]))
            bound = float(cfg["source_head_init_bias"])
            self.source_head.bias.uniform_(-bound, bound)
            self.self_head.weight.normal_(0.0, std)
            self.self_head.bias.fill_(float(cfg["self_head_init_logit"]))
            self.edge_head.weight.normal_(0.0, std)
            self.edge_head.bias.fill_(float(cfg["edge_head_init_logit"]))

    def initial_state_sha(self) -> str:
        digest = []
        for name, tensor in sorted(self.state_dict().items()):
            digest.append(name)
            digest.append(C.sha256_array(tensor.detach().cpu().numpy()))
        return C.sha256_bytes("|".join(digest).encode("utf-8"))

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    # -- forward pieces -----------------------------------------------------

    def encode_nodes(self, atom_cat: torch.Tensor, nb_counts: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat(
            [emb(atom_cat[..., j]) for j, emb in enumerate(self.atom_embeddings)], dim=-1
        )
        node_input = torch.cat([embedded, nb_counts.to(embedded.dtype)], dim=-1)
        return self.node_mlp(node_input)

    def encode_edges(
        self,
        h: torch.Tensor,
        bond_cat: torch.Tensor,
        bu: torch.Tensor,
        bv: torch.Tensor,
    ) -> torch.Tensor:
        hidden = h.shape[-1]
        batch, n_edges = bu.shape
        index_u = bu.unsqueeze(-1).expand(batch, n_edges, hidden)
        index_v = bv.unsqueeze(-1).expand(batch, n_edges, hidden)
        h_u = h.gather(1, index_u)
        h_v = h.gather(1, index_v)
        bond_emb = torch.cat(
            [emb(bond_cat[..., j]) for j, emb in enumerate(self.bond_embeddings)], dim=-1
        )
        edge_input = torch.cat(
            [h_u + h_v, (h_u - h_v).abs(), h_u * h_v, bond_emb], dim=-1
        )
        return self.edge_mlp(edge_input)

    def generate(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode_nodes(batch["atom_cat"], batch["nb_counts"])
        h_e = self.encode_edges(h, batch["bond_cat"], batch["bu"], batch["bv"])
        s = 2.0 * torch.tanh(self.source_head(h))
        r = 0.5 + 1.5 * torch.sigmoid(self.self_head(h))
        w = 2.0 * torch.sigmoid(self.edge_head(h_e))
        edge_mask = batch["edge_mask"].unsqueeze(-1).to(w.dtype)
        w = w * edge_mask
        return s, r, w

    # -- response / prediction ---------------------------------------------

    def response(
        self,
        s: torch.Tensor,
        r: torch.Tensor,
        w: torch.Tensor,
        batch: dict[str, Any],
        *,
        arm: str | None = None,
        return_x: bool = False,
        return_info: bool = False,
    ):
        arm = self.arm if arm is None else arm
        if arm == "full":
            return full_response(
                s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"],
                return_x=return_x, return_info=return_info,
            )
        if arm == "local2":
            return local2_response(
                s, r, w, batch["bu"], batch["bv"], batch["node_mask"], batch["edge_mask"]
            )
        raise ValueError(f"unknown arm {arm!r}")

    def predict_from_q(self, q: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.readout(q, n, self.n_bar, self.mu_y, self.sigma_y)

    def forward(
        self,
        batch: dict[str, Any],
        *,
        arm: str | None = None,
        return_x: bool = False,
        return_info: bool = False,
    ) -> dict[str, Any]:
        s, r, w = self.generate(batch)
        out = self.response(s, r, w, batch, arm=arm, return_x=return_x, return_info=return_info)
        q = out[0] if isinstance(out, tuple) else out
        x = out[1] if (return_x and isinstance(out, tuple)) else None
        info = out[-1] if (return_info and isinstance(out, tuple)) else None
        pred, pred_std = self.predict_from_q(q, batch["n"])
        result = {
            "pred": pred,
            "pred_std": pred_std,
            "q": q,
            "s": s,
            "r": r,
            "w": w,
            "row_ids": batch["row_ids"],
            "n": batch["n"],
        }
        if x is not None:
            result["x"] = x
        if info is not None:
            result["response_info"] = info
        return result
