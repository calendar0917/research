"""The word response operator: exact multi-port Gaussian composition, no message passing.

One independent scalar latent per atom per channel; every atom/bond/pair MLP is a
plain per-object two-layer MLP (no neighbour aggregation).  The word constructs a
local energy matrix ``Q_i = diag(d) + sum k l l^T + sum lambda l l^T`` whose
diagonal is dominated by ``d >= 0.5``, so ``Q_i`` and every assembled ``K`` are
strictly positive definite by construction.

* ``FULL_WORD_OP``  : solve in the whole atom space of the molecule.
* ``LOCAL_WORD_OP`` : true boundary assembly; crossing bonds keep their ``+k``
  diagonal and lose their ``-k`` off-diagonal, so each word is solved alone.
* ``PORT_DIAG_WORD_OP``: boundary assembly with ``S_i -> diag(diag(S_i))``, one
  global boundary solve (only unlocked by a GREEN decision).
* ``ATOMIC_OP_CAP`` : same global solve and same constant term, no word coupling
  (no PairMLP, no theta/psi).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

import wo_common as C
import wo_struct as S

ARMS = ("FULL_WORD_OP", "LOCAL_WORD_OP", "ATOMIC_OP_CAP", "PORT_DIAG_WORD_OP",
        "FULL_BOUNDARY_CHECK")
TRAINABLE_ARMS = ("FULL_WORD_OP", "LOCAL_WORD_OP", "ATOMIC_OP_CAP", "PORT_DIAG_WORD_OP")
MAXW = S.MAX_ATOMS


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


class OperatorNet(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], dims: Mapping[str, Sequence[int]],
                 n_templates: int, n_site_orbits: int, n_pair_orbits: int, z_dim: int,
                 arm: str, hidden: int | None = None, seed: int = 0) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm {arm} is not in the frozen allowlist {ARMS}")
        op = cfg["operator"]
        self.arm = arm
        self.use_word = arm != "ATOMIC_OP_CAP"
        self.channels = int(op["channels"])
        self.hidden = int(hidden if hidden is not None else op["hidden"])
        self.atom_dims = [int(d) for d in dims["atom"]]
        self.bond_dims = [int(d) for d in dims["bond"]]
        atom_embed = int(op["atom_embed_dim"])
        bond_embed = int(op["bond_embed_dim"])
        self.atom_in = atom_embed * len(self.atom_dims)
        self.bond_in = self.atom_in * 2 + bond_embed * len(self.bond_dims)
        self.pair_in = 36 + 36 + 6 + 3

        torch.manual_seed(int(seed))
        self.atom_embeddings = nn.ModuleList([nn.Embedding(d, atom_embed) for d in self.atom_dims])
        self.bond_embeddings = nn.ModuleList([nn.Embedding(d, bond_embed) for d in self.bond_dims])
        self.node = mlp(self.atom_in, self.hidden, 2 * self.channels)
        self.bond = mlp(self.bond_in, self.hidden, self.channels)
        if self.use_word:
            self.pair = mlp(self.pair_in, self.hidden, self.channels)
            last = [m for m in self.pair if isinstance(m, nn.Linear)][-1]
            nn.init.constant_(last.bias, float(op["pair_bias_init"]))
            self.theta = nn.Parameter(
                torch.randn(n_pair_orbits, self.channels) * float(op["theta_psi_init_std"]))
            self.psi = nn.Parameter(
                torch.randn(n_site_orbits, self.channels) * float(op["theta_psi_init_std"]))
        self.beta = nn.Parameter(torch.zeros(z_dim))
        self.beta0 = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.as_tensor(op["alpha_init"], dtype=torch.float32).clone())
        self.cholesky_jitter = 0

    # ------------------------------------------------------------------
    def _chol(self, matrix: torch.Tensor) -> torch.Tensor:
        L, info = torch.linalg.cholesky_ex(matrix)
        if int(info.max().item()) != 0:
            eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
            matrix = matrix + 1e-6 * eye
            L, info = torch.linalg.cholesky_ex(matrix)
            self.cholesky_jitter += int((info != 0).sum().item())
            if int(info.max().item()) != 0:
                raise RuntimeError("Cholesky failed even after the frozen 1e-6 jitter")
        return L

    # ------------------------------------------------------------------
    def atom_bond_pair(self, b) -> dict[str, Any]:
        x = torch.cat([emb(b["atom_cat"][..., j])
                       for j, emb in enumerate(self.atom_embeddings)], dim=-1)
        h = self.node(x)
        d = 0.5 + 1.5 * torch.sigmoid(h[..., :self.channels])
        bbase = torch.tanh(h[..., self.channels:])
        maskf = b["atom_mask"].to(d.dtype)[..., None]
        d = d * maskf
        bbase = bbase * maskf
        e = torch.cat([emb(b["bond_cat"][:, j])
                       for j, emb in enumerate(self.bond_embeddings)], dim=-1)
        xu = x[b["bond_mol"], b["bond_u_row"]]
        xv = x[b["bond_mol"], b["bond_v_row"]]
        k = 0.05 + 1.95 * torch.sigmoid(self.bond(torch.cat([xu + xv, xu * xv, e], dim=-1)))
        out = {"x": x, "d": d, "bbase": bbase, "k": k, "lambda": None}
        if self.use_word:
            xu = x[b["pair_mol"], b["pair_u_row"]]
            xv = x[b["pair_mol"], b["pair_v_row"]]
            dist = torch.nn.functional.one_hot(b["pair_dist"], 6).to(x.dtype)
            scal = torch.stack([torch.log1p(b["pair_n_atoms"].to(x.dtype)),
                                torch.log1p(b["pair_m"].to(x.dtype)),
                                b["pair_rank"].to(x.dtype)], dim=-1)
            raw = self.pair(torch.cat([xu + xv, xu * xv, dist, scal], dim=-1))
            logits = raw + b["word_s"][b["pair_word"]][:, None] * self.theta[b["pair_orbit"]]
            scale = 0.5 / (b["pair_n_atoms"].to(x.dtype) - 1.0)[:, None]
            out["lambda"] = scale * torch.sigmoid(logits)
        return out

    # ------------------------------------------------------------------
    def word_source(self, b, bbase: torch.Tensor) -> torch.Tensor:
        if not self.use_word:
            return bbase
        maskf = b["atom_mask"].to(bbase.dtype)
        psi_atom = torch.tanh(self.psi[b["atom_orbit_pad"]]) * maskf[..., None]
        psi_words = psi_atom[b["word_mol"][:, None], b["word_atoms_row"]] * \
            b["word_atom_mask"].to(bbase.dtype)[..., None]
        wmean = psi_words.sum(dim=1) / b["word_n_atoms"].to(bbase.dtype)[:, None]
        atom_word = b["atom_word_pad"].clamp(min=0)
        mean_atom = wmean[atom_word] * maskf[..., None]
        s_atom = b["word_s"][atom_word][..., None] * maskf[..., None]
        return bbase + 0.25 * s_atom * (psi_atom - mean_atom)

    # ------------------------------------------------------------------
    def forward(self, b) -> dict[str, Any]:
        device = b["atom_cat"].device
        B = int(b["atom_cat"].shape[0])
        n_pad = int(b["n_pad"])
        base = self.atom_bond_pair(b)
        bval = self.word_source(b, base["bbase"])
        d, k, lam = base["d"], base["k"], base["lambda"]
        maskf = b["atom_mask"].to(d.dtype)

        mol = torch.arange(B, device=device)[:, None]
        rows = torch.arange(n_pad, device=device)[None, :]
        atom_dense = mol * n_pad + rows          # padded row index of every atom
        atom_diag = atom_dense * n_pad + rows     # flat (row, col) index
        diag_idx = atom_diag[b["atom_mask"]]
        pad_diag_idx = atom_diag[~b["atom_mask"]]
        pad_ones = torch.ones(len(pad_diag_idx), device=device, dtype=d.dtype)
        n_flat = B * n_pad * n_pad

        def entries(u_row, v_row, mol_local):
            u = mol_local * n_pad + u_row
            v = mol_local * n_pad + v_row
            return u * n_pad + v_row, v * n_pad + u_row, u * n_pad + u_row, v * n_pad + v_row

        outputs: list[torch.Tensor] = []
        consts: list[torch.Tensor] = []
        zeros_B = torch.zeros(B, device=device, dtype=d.dtype)
        if self.arm in ("FULL_WORD_OP", "ATOMIC_OP_CAP"):
            bond_uv, bond_vu, bond_uu, bond_vv = entries(
                b["bond_u_row"], b["bond_v_row"], b["bond_mol"])
            if lam is not None:
                pair_uv, pair_vu, pair_uu, pair_vv = entries(
                    b["pair_u_row"], b["pair_v_row"], b["pair_mol"])
            for c in range(self.channels):
                uu, vv, uv, vu, vals = [bond_uu], [bond_vv], [bond_uv], [bond_vu], [k[:, c]]
                sign = [-k[:, c]]
                if lam is not None:
                    uu.append(pair_uu)
                    vv.append(pair_vv)
                    uv.append(pair_uv)
                    vu.append(pair_vu)
                    sign.append(-lam[:, c])
                sign = torch.cat(sign)
                idx = torch.cat([diag_idx, torch.cat(uu), torch.cat(vv),
                                 torch.cat(uv), torch.cat(vu)])
                values = torch.cat([d[..., c][b["atom_mask"]], sign.abs(), sign.abs(),
                                    sign, sign])
                q = torch.zeros(n_flat, device=device, dtype=d.dtype).index_add(0, idx, values)
                q = q.index_add(0, pad_diag_idx, pad_ones)
                Q = q.view(B, n_pad, n_pad)
                L = self._chol(Q)
                logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
                hvec = bval[..., c] * maskf
                sol = torch.cholesky_solve(hvec.unsqueeze(-1), L).squeeze(-1)
                outputs.append(0.5 * logdet - 0.5 * (hvec * sol).sum(-1))
                consts.append(zeros_B)
            F = torch.stack(outputs, dim=-1)
            const = torch.stack(consts, dim=-1).sum(-1)
        else:
            n_words = len(b["word_n_atoms"])
            word_n = b["word_n_atoms"].to(d.dtype)
            int_mask = b["bond_word_u"] == b["bond_word_v"]
            ii_sites, ii_mask = b["word_ii_sites"], b["word_ii_mask"]
            bnd_sites, bnd_mask = b["word_bnd_sites"], b["word_bnd_mask"]
            grid = torch.arange(MAXW, device=device)[None, :]
            ii_pos = grid < ii_mask.sum(dim=1, keepdim=True)      # positions inside Q_II
            bnd_pos = grid < bnd_mask.sum(dim=1, keepdim=True)    # positions inside S
            wmask = bnd_pos.to(d.dtype)
            w_pairs = bnd_pos[:, :, None] & bnd_pos[:, None, :]
            ii_pairs = ii_pos[:, :, None] & ii_pos[:, None, :]
            # static word-entry indices (shared across channels)
            bw_u = b["bond_word_u"][int_mask]
            bw_v = b["bond_word_v"][int_mask]
            bs_u = b["bond_site_u"][int_mask]
            bs_v = b["bond_site_v"][int_mask]
            base_uu = bw_u * (MAXW * MAXW) + bs_u * MAXW + bs_u
            base_vv = bw_v * (MAXW * MAXW) + bs_v * MAXW + bs_v
            base_uv = bw_u * (MAXW * MAXW) + bs_u * MAXW + bs_v
            base_vu = bw_v * (MAXW * MAXW) + bs_v * MAXW + bs_u
            atom_wd_base = (b["atom_word_pad"] * (MAXW * MAXW)
                            + b["atom_site_pad"] * MAXW + b["atom_site_pad"])
            atom_wd = atom_wd_base[b["atom_mask"]]
            if lam is not None:
                pw = b["pair_word"]
                ps_u, ps_v = b["pair_site_u"], b["pair_site_v"]
                pair_uv = pw * (MAXW * MAXW) + ps_u * MAXW + ps_v
                pair_vu = pw * (MAXW * MAXW) + ps_v * MAXW + ps_u
                pair_uu = pw * (MAXW * MAXW) + ps_u * MAXW + ps_u
                pair_vv = pw * (MAXW * MAXW) + ps_v * MAXW + ps_v
                base_uv_full = torch.cat([base_uv, pair_uv])
                base_vu_full = torch.cat([base_vu, pair_vu])
            else:
                base_uv_full, base_vu_full = base_uv, base_vu
                pair_uu = pair_vv = None
            slots = b["word_bnd_start"][:, None] + torch.arange(MAXW, device=device)[None, :]
            slot_row = b["word_mol"][:, None] * b["b_pad"] + slots
            kidx = slot_row[:, :, None] * b["b_pad"] + slots[:, None, :]
            pad_b = torch.arange(b["b_pad"], device=device)[None, :]
            real_b = pad_b < b["n_boundary"][:, None]
            pad_row = torch.arange(B, device=device)[:, None] * b["b_pad"] + pad_b
            pad_idx = (pad_row * b["b_pad"] + pad_b)[~real_b]
            ones_b = torch.ones(len(pad_idx), device=device, dtype=d.dtype)
            for c in range(self.channels):
                vals = [k[:, c][int_mask]]
                diag_vals = [d[..., c][b["atom_mask"]], k[:, c][int_mask].abs(),
                             k[:, c][int_mask].abs()]
                diag_idx_parts = [atom_wd, base_uu, base_vv]
                if lam is not None:
                    vals.append(lam[:, c])
                    diag_vals.extend([lam[:, c].abs(), lam[:, c].abs()])
                    diag_idx_parts.extend([pair_uu, pair_vv])
                sign = -torch.cat(vals)
                q = torch.zeros(n_words * MAXW * MAXW, device=device, dtype=d.dtype)
                q = q.index_add(0, torch.cat(diag_idx_parts), torch.cat(diag_vals))
                q = q.index_add(0, torch.cat([base_uv_full, base_vu_full]),
                                torch.cat([sign, sign]))
                Qw = q.view(n_words, MAXW, MAXW)
                bw = bval[..., c][b["word_mol"][:, None], b["word_atoms_row"]] * \
                    b["word_atom_mask"].to(d.dtype)
                QII = gather_matrix(Qw, ii_sites)
                QII = QII * ii_pairs.to(d.dtype)
                QII = QII + torch.diag_embed((~ii_pos).to(d.dtype))
                bI = torch.gather(bw, 1, ii_sites) * ii_pos.to(d.dtype)
                QIB = gather_matrix(Qw, ii_sites, bnd_sites)
                QIB = QIB * (ii_pos[:, :, None] & bnd_pos[:, None, :]).to(d.dtype)
                QBB = gather_matrix(Qw, bnd_sites)
                QBB = QBB * w_pairs.to(d.dtype)
                bB = torch.gather(bw, 1, bnd_sites) * wmask
                LI = self._chol(QII)
                logdetI = 2.0 * torch.log(torch.diagonal(LI, dim1=-2, dim2=-1)).sum(-1)
                solQIB = torch.cholesky_solve(QIB, LI)
                solbI = torch.cholesky_solve(bI.unsqueeze(-1), LI).squeeze(-1)
                Sred = QBB - torch.bmm(QIB.transpose(1, 2), solQIB)
                Sred = Sred * w_pairs.to(d.dtype) + torch.diag_embed((~bnd_pos).to(d.dtype))
                hred = (bB - torch.bmm(QIB.transpose(1, 2), solbI.unsqueeze(-1)).squeeze(-1))
                hred = hred * wmask
                c_word = 0.5 * logdetI - 0.5 * (bI * solbI).sum(-1)
                csum = zeros_B.index_add(0, b["word_mol"], c_word)
                cross_sel = b["bond_cross"]
                kcross = k[cross_sel, c]
                if self.arm == "LOCAL_WORD_OP":
                    diag_flat = torch.zeros(n_words * MAXW, device=device, dtype=d.dtype)
                    diag_flat = diag_flat.index_add(
                        0, b["cross_word_u"] * MAXW + b["cross_local_u"], kcross)
                    diag_flat = diag_flat.index_add(
                        0, b["cross_word_v"] * MAXW + b["cross_local_v"], kcross)
                    diag_add = diag_flat.view(n_words, MAXW)
                    Ssolve = Sred + torch.diag_embed(diag_add)
                    Ls = self._chol(Ssolve)
                    logdetS = 2.0 * torch.log(torch.diagonal(Ls, dim1=-2, dim2=-1)).sum(-1)
                    solh = torch.cholesky_solve(hred.unsqueeze(-1), Ls).squeeze(-1)
                    Fw = c_word + 0.5 * logdetS - 0.5 * (hred * solh).sum(-1)
                    outputs.append(zeros_B.index_add(0, b["word_mol"], Fw))
                else:  # PORT_DIAG_WORD_OP or the boundary-assembly equivalence check
                    if self.arm == "PORT_DIAG_WORD_OP":
                        sdiag = torch.diagonal(Sred, dim1=-2, dim2=-1)
                        Sdiag = torch.diag_embed(sdiag)
                    else:
                        Sdiag = Sred
                    Sdiag = Sdiag * w_pairs.to(d.dtype) + \
                        torch.diag_embed((~bnd_pos).to(d.dtype))
                    q = torch.zeros(B * b["b_pad"] * b["b_pad"], device=device, dtype=d.dtype)
                    q = q.index_add(0, kidx[w_pairs], Sdiag[w_pairs])
                    cu = b["cross_mol"] * b["b_pad"] + b["cross_slot_u"]
                    cv = b["cross_mol"] * b["b_pad"] + b["cross_slot_v"]
                    su, sv = b["cross_slot_u"], b["cross_slot_v"]
                    kn = -kcross
                    q = q.index_add(0, torch.cat([cu * b["b_pad"] + su,
                                                  cv * b["b_pad"] + sv]),
                                    torch.cat([-kn, -kn]))
                    q = q.index_add(0, torch.cat([cu * b["b_pad"] + sv,
                                                  cv * b["b_pad"] + su]),
                                    torch.cat([kn, kn]))
                    q = q.index_add(0, pad_idx, ones_b)
                    K = q.view(B, b["b_pad"], b["b_pad"])
                    LK = self._chol(K)
                    logdetK = 2.0 * torch.log(torch.diagonal(LK, dim1=-2, dim2=-1)).sum(-1)
                    hflat = torch.zeros(B * b["b_pad"], device=device, dtype=d.dtype).index_put(
                        (slot_row[bnd_pos],), hred[bnd_pos], accumulate=True)
                    hmat = hflat.view(B, b["b_pad"])
                    sol = torch.cholesky_solve(hmat.unsqueeze(-1), LK).squeeze(-1)
                    outputs.append(csum + 0.5 * logdetK - 0.5 * (hmat * sol).sum(-1))
                consts.append(csum)
            F = torch.stack(outputs, dim=-1)
            const = torch.stack(consts, dim=-1).sum(-1)
        A = self.beta0 + b["z_std"] @ self.beta
        yhat = A + (self.alpha[None, :] * F / float(b["n0"])).sum(-1)
        return {"yhat": yhat, "F": F, "A": A, "const": const}


def gather_matrix(mat: torch.Tensor, idx_rows: torch.Tensor,
                  idx_cols: torch.Tensor | None = None) -> torch.Tensor:
    if idx_cols is None:
        idx_cols = idx_rows
    rows = torch.gather(mat, 1, idx_rows.unsqueeze(-1).expand(-1, -1, mat.shape[-1]))
    return torch.gather(rows, 2, idx_cols.unsqueeze(1).expand(-1, mat.shape[1], -1))


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def select_atomic_width(cfg: Mapping[str, Any], dims, n_templates, n_site_orbits,
                        n_pair_orbits, z_dim, widths: Sequence[int] = (32, 48, 64, 96, 128, 192, 256)):
    """Width whose ATOMIC parameter count is closest to FULL (tie -> smaller)."""
    full = OperatorNet(cfg, dims, n_templates, n_site_orbits, n_pair_orbits, z_dim,
                       "FULL_WORD_OP")
    target = count_parameters(full)
    table = []
    for width in widths:
        model = OperatorNet(cfg, dims, n_templates, n_site_orbits, n_pair_orbits, z_dim,
                            "ATOMIC_OP_CAP", hidden=int(width))
        table.append({"width": int(width), "n_parameters": count_parameters(model)})
    best = min(table, key=lambda row: (abs(row["n_parameters"] - target), row["width"]))
    return {"full_parameters": target, "selected_width": int(best["width"]),
            "selected_parameters": int(best["n_parameters"]),
            "relative_gap": abs(best["n_parameters"] - target) / target,
            "within_20pct": abs(best["n_parameters"] - target) <= 0.2 * target,
            "table": table}
