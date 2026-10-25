"""Production models for zinc-statistical-program-search-v1.

Two independent bodies, one shared legal front-end:

Q  rooted empirical measures over real atom / physical-bond records, read with
   learned shared queries (log-mean-exp, tilted mean, tilted second moment).
F  exact content-conditioned rooted-forest marginals of the undirected
   physical-bond graph (dense Cholesky solves, fixed scales).

Neither body performs recursive neighbourhood hidden-state passing.  The only
trainable structure-dependent computation is (Q) a query read of static
per-root records, or (F) a learned positive edge weight entering an explicit
matrix statistic.

Parameter budget: every body is well under 500,000 and the exact count is
recorded in train.json.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

ATOM_FEAT_DIM = 108
BOND_FEAT_DIM = 7
GLOBAL_DIM = 33
WIDTH = 96
N_GROUPS = 6
N_QUERIES = 4
KEY_DIM = 16
VALUE_DIM = 8
DIST_EMB = 8
DIST_BUCKETS = 11  # 0..8 exact, 9 = >=9, 10 = unreachable
SCALES = (0.25, 1.0, 4.0, 16.0)


class MLP(nn.Module):
    """Same ``lins`` naming as the frozen P3 encoders, SiLU activations."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * n_hidden + [out_dim]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class CommonEncoders(nn.Module):
    """P3 atom/bond encoders only (no junction module, no composer, no reader).

    89,152 parameters: atom 108->128->96, bond (3*96+7)->160->96.  The bond
    input is endpoint-symmetric (sum, |diff|, product) so a physical bond is
    represented identically under endpoint swap.  Loaded from the train-only
    masked-SSL checkpoint and fine-tuned; the unused junction encoder is not
    instantiated and therefore is not counted as capacity.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.atom_encoder = MLP(ATOM_FEAT_DIM, 128, WIDTH, 1, dropout)
        self.bond_encoder = MLP(3 * WIDTH + BOND_FEAT_DIM, 160, WIDTH, 1, dropout)

    def forward(self, atom_feat: torch.Tensor, bond_feat: torch.Tensor,
                bond_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.atom_encoder(atom_feat)
        au, av = a[bond_index[:, 0]], a[bond_index[:, 1]]
        b = self.bond_encoder(
            torch.cat([au + av, (au - av).abs(), au * av, bond_feat], dim=1)
        )
        return a, b


def load_ssl(encoders: CommonEncoders, path, device) -> dict:
    """Load only the matching atom/bond encoders; ignore the junction module."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    encoders.atom_encoder.load_state_dict(state["atom_encoder"])
    encoders.bond_encoder.load_state_dict(state["bond_encoder"])
    return {
        "path": str(path),
        "keys_loaded": ["atom_encoder", "bond_encoder"],
        "junction_encoder_ignored": True,
        "sha256_note": "see RUN_LEDGER.md; sha checked by the caller",
    }


# ---------------------------------------------------------------------------
# Route Q: rooted measure network
# ---------------------------------------------------------------------------


class RootedMeasureModel(nn.Module):
    """Q-fixed / Q-adaptive: learned queries over six rooted record groups.

    Records are (root r, object o) pairs of the real molecule:
      atom record  x = (a_u, dist(r,u))                     group by dist in {0-2},{3-4},{>=5/oo}
      bond record  x = (b_e bound to g(r,u), g(r,v))        group by min endpoint dist
    Every real record has multiplicity one; no two records are merged unless the
    raw input fields are identical (they are not, because of the root binding).
    """

    def __init__(self, adaptive: bool = False, dropout: float = 0.0):
        super().__init__()
        self.adaptive = bool(adaptive)
        self.encoders = CommonEncoders(dropout=dropout)
        self.distance = nn.Embedding(DIST_BUCKETS, DIST_EMB)
        self.role_atom = MLP(WIDTH + DIST_EMB, 64, 32, 1, dropout)
        self.atom_memory = nn.Linear(32, KEY_DIM + VALUE_DIM)
        self.bond_memory = MLP(WIDTH + 3 * 32, 64, KEY_DIM + VALUE_DIM, 1, dropout)
        self.query = nn.Parameter(torch.randn(N_QUERIES, KEY_DIM) * 0.15)
        stat_dim = N_GROUPS * N_QUERIES * (1 + 2 * VALUE_DIM) + 2 * N_GROUPS + WIDTH
        self.root_encoder = MLP(stat_dim, 128, WIDTH, 1, dropout)
        self.query_update = nn.Linear(WIDTH, N_QUERIES * KEY_DIM) if self.adaptive else None
        self.pool_proj = nn.Linear(WIDTH, 32)  # shared by a and b
        self.reader = nn.Sequential(
            nn.Linear(2 * WIDTH + 4 * 32 + GLOBAL_DIM + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("global_mean", torch.zeros(GLOBAL_DIM))
        self.register_buffer("global_scale", torch.ones(GLOBAL_DIM))
        self.register_buffer("fit_mean_atoms", torch.tensor(23.0))
        self.register_buffer("fit_mean_bonds", torch.tensor(25.0))
        self.register_buffer("fit_group_count", torch.ones(N_GROUPS))

    # -- read of the static records -----------------------------------------
    def _read(self, k: torch.Tensor, v: torch.Tensor, root: torch.Tensor,
              grp: torch.Tensor, q: torch.Tensor, n_roots: int,
              group_count: torch.Tensor) -> torch.Tensor:
        """Exact empirical-measure query. Returns [n_roots, 6, 4, 1+2V]."""
        # The same four learned queries read all six groups; the group only
        # decides which empirical measure the queries are applied to.
        scores = (k.unsqueeze(1) * q.unsqueeze(0)).sum(-1) / math.sqrt(KEY_DIM)
        seg = root * N_GROUPS + grp                        # [R]
        nseg = n_roots * N_GROUPS
        neg = torch.finfo(scores.dtype).min
        mx = torch.full((nseg, N_QUERIES), neg, device=scores.device, dtype=scores.dtype)
        mx.scatter_reduce_(0, seg.unsqueeze(1).expand(-1, N_QUERIES), scores,
                           reduce="amax", include_self=True)
        e = torch.exp(scores - mx[seg])                    # [R,4]
        den = torch.zeros((nseg, N_QUERIES), device=scores.device, dtype=scores.dtype)
        den.index_add_(0, seg, e)
        mv = torch.zeros((nseg, N_QUERIES, VALUE_DIM), device=scores.device, dtype=scores.dtype)
        mv.index_add_(0, seg, e.unsqueeze(-1) * v.unsqueeze(1))
        ms = torch.zeros_like(mv)
        ms.index_add_(0, seg, e.unsqueeze(-1) * (v * v).unsqueeze(1))
        cnt = group_count.reshape(-1)                       # [nseg]
        safe = den.clamp(min=1e-20)
        ell = torch.log(safe) + mx - torch.log(cnt.clamp(min=1.0)).unsqueeze(1)
        mv = mv / safe.unsqueeze(-1)
        ms = ms / safe.unsqueeze(-1)
        live = (cnt > 0).unsqueeze(1)
        ell = ell * live
        mv = mv * live.unsqueeze(-1)
        ms = ms * live.unsqueeze(-1)
        out = torch.cat([ell.unsqueeze(-1), mv, ms], dim=-1)  # [nseg,4,17]
        return out.reshape(n_roots, N_GROUPS, N_QUERIES * (1 + 2 * VALUE_DIM))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b = self.encoders(batch["atom_feat"], batch["bond_feat"], batch["bond_index"])
        B = int(batch["n_graphs"])
        A = a.shape[0]

        g = self.role_atom(
            torch.cat([a[batch["atom_rec_atom"]], self.distance(batch["atom_rec_dist"])], dim=1)
        )
        am = self.atom_memory(g)
        gu, gv = g[batch["bond_rec_iu"]], g[batch["bond_rec_iv"]]
        bm = self.bond_memory(
            torch.cat([b[batch["bond_rec_bond"]], gu + gv, (gu - gv).abs(), gu * gv], dim=1)
        )
        mem = torch.cat([am, bm], dim=0)
        root = torch.cat([batch["atom_rec_root"], batch["bond_rec_root"]], dim=0)
        grp = torch.cat([batch["atom_rec_group"], batch["bond_rec_group"] + 3], dim=0)
        k = torch.tanh(mem[:, :KEY_DIM])
        v = mem[:, KEY_DIM:]

        q = 2.0 * torch.tanh(self.query)
        stats = self._read(k, v, root, grp, q, A, batch["root_cnt"])
        cnt = batch["root_cnt"]
        inp = torch.cat(
            [a, stats.reshape(A, -1), cnt / self.fit_group_count.unsqueeze(0), torch.log1p(cnt)],
            dim=1,
        )
        h = self.root_encoder(inp)
        if self.adaptive:
            q1 = q.unsqueeze(0) + 0.25 * torch.tanh(
                self.query_update(h).reshape(A, N_QUERIES, KEY_DIM)
            )
            # Identity of q per record requires per-root queries: read each group
            # with the root's own q1.  This is a static re-read of the same
            # records, not root-to-root communication.
            stats = self._read_adaptive(k, v, root, grp, q1, A, batch["root_cnt"])
            inp = torch.cat(
                [a, stats.reshape(A, -1), cnt / self.fit_group_count.unsqueeze(0), torch.log1p(cnt)],
                dim=1,
            )
            h = self.root_encoder(inp)

        z = h.new_zeros((B, WIDTH)).index_add_(0, batch["atom_batch"], h)
        z2 = h.new_zeros((B, WIDTH)).index_add_(0, batch["atom_batch"], h * h)
        ap = self.pool_proj(a)
        bp = self.pool_proj(b)
        aps = ap.new_zeros((B, 32)).index_add_(0, batch["atom_batch"], ap)
        aps2 = ap.new_zeros((B, 32)).index_add_(0, batch["atom_batch"], ap * ap)
        bps = bp.new_zeros((B, 32)).index_add_(0, batch["bond_batch"], bp)
        bps2 = bp.new_zeros((B, 32)).index_add_(0, batch["bond_batch"], bp * bp)
        gstats = (batch["global"] - self.global_mean) / self.global_scale
        mn = self.fit_mean_atoms
        mb = self.fit_mean_bonds
        pool = torch.cat(
            [z / mn, z2 / mn, aps / mn, aps2 / mn, bps / mb, bps2 / mb, gstats,
             torch.log1p(batch["atom_counts"]).unsqueeze(1),
             torch.log1p(batch["bond_counts"]).unsqueeze(1)],
            dim=1,
        )
        return self.reader(pool).view(-1)

    def _read_adaptive(self, k, v, root, grp, q_root, n_roots, group_count):
        """Same static records, root-specific queries (Q-adaptive only)."""
        q_rec = q_root[root]                               # [R,4,16]
        scores = (k.unsqueeze(1) * q_rec).sum(-1) / math.sqrt(KEY_DIM)
        seg = root * N_GROUPS + grp
        nseg = n_roots * N_GROUPS
        neg = torch.finfo(scores.dtype).min
        mx = torch.full((nseg, N_QUERIES), neg, device=scores.device, dtype=scores.dtype)
        mx.scatter_reduce_(0, seg.unsqueeze(1).expand(-1, N_QUERIES), scores,
                           reduce="amax", include_self=True)
        e = torch.exp(scores - mx[seg])
        den = torch.zeros((nseg, N_QUERIES), device=scores.device, dtype=scores.dtype)
        den.index_add_(0, seg, e)
        mv = torch.zeros((nseg, N_QUERIES, VALUE_DIM), device=scores.device, dtype=scores.dtype)
        mv.index_add_(0, seg, e.unsqueeze(-1) * v.unsqueeze(1))
        ms = torch.zeros_like(mv)
        ms.index_add_(0, seg, e.unsqueeze(-1) * (v * v).unsqueeze(1))
        cnt = group_count.reshape(-1)
        safe = den.clamp(min=1e-20)
        ell = torch.log(safe) + mx - torch.log(cnt.clamp(min=1.0)).unsqueeze(1)
        mv = mv / safe.unsqueeze(-1)
        ms = ms / safe.unsqueeze(-1)
        live = (cnt > 0).unsqueeze(1)
        out = torch.cat([ell.unsqueeze(-1), mv * live.unsqueeze(-1), ms * live.unsqueeze(-1)], -1)
        return out.reshape(n_roots, N_GROUPS, N_QUERIES * (1 + 2 * VALUE_DIM))


# ---------------------------------------------------------------------------
# Route F: content-conditioned rooted-forest statistics
# ---------------------------------------------------------------------------


class ForestStatisticModel(nn.Module):
    """F-content / F-fixed: learned positive edge weights, exact forest marginals.

    w_e = 0.1 + 1.9 * sigmoid(MLP(b_e)) in (0.1, 2.0), four fixed scales.
    r_i,tau   = Q_tau[i,i]                                        (root probability)
    p_e,tau   = tau*w_e*(Q_uu+Q_vv-Q_uv-Q_vu)                      (inclusion prob.)
    logZ_tau  = logdet(I + L_tau)
    with Q_tau = (I + J diag(tau*w) J^T)^{-1} from a dense Cholesky solve.
    The signs of the incidence columns are arbitrary; all reported quantities
    are invariant to them.  The pool uses per-object values only (plus the
    scalar logZ/n and global33), never a plain sum of roots and edges.
    """

    def __init__(self, content_weights: bool = True, dropout: float = 0.0):
        super().__init__()
        self.content_weights = bool(content_weights)
        self.encoders = CommonEncoders(dropout=dropout)
        if self.content_weights:
            self.conductance = MLP(WIDTH, 32, 1, 1, dropout)
        self.atom_lift = MLP(WIDTH + len(SCALES), 128, WIDTH, 1, dropout)
        self.bond_lift = MLP(WIDTH + len(SCALES), 128, WIDTH, 1, dropout)
        self.reader = nn.Sequential(
            nn.Linear(4 * WIDTH + len(SCALES) + GLOBAL_DIM + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("global_mean", torch.zeros(GLOBAL_DIM))
        self.register_buffer("global_scale", torch.ones(GLOBAL_DIM))
        self.register_buffer("fit_mean_atoms", torch.tensor(23.0))
        self.register_buffer("fit_mean_bonds", torch.tensor(25.0))
        self.register_buffer("scales", torch.tensor(SCALES))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b = self.encoders(batch["atom_feat"], batch["bond_feat"], batch["bond_index"])
        B = int(batch["n_graphs"])
        N = int(batch["Nmax"])
        E = int(batch["Emax"])
        C = len(SCALES)
        scale = self.scales.to(dtype=a.dtype)

        if self.content_weights:
            w = 0.1 + 1.9 * torch.sigmoid(self.conductance(b).squeeze(-1))      # [E]
        else:
            w = b.new_ones(b.shape[0])
        wflat = torch.zeros(B * E, device=a.device, dtype=a.dtype).scatter(
            0, batch["edge_batch"] * E + batch["edge_local"], w
        )
        w_pad = wflat.view(B, E)

        valid_e = batch["edge_mask"]                                            # [B,E]
        slot = torch.arange(E, device=a.device).expand(B, -1)
        bidx = torch.arange(B, device=a.device).unsqueeze(1).expand(-1, E)
        S = a.new_zeros((B, E, N))
        S[bidx[valid_e], slot[valid_e], batch["edge_u"][valid_e]] = 1.0
        S[bidx[valid_e], slot[valid_e], batch["edge_v"][valid_e]] = -1.0

        ws = w_pad.unsqueeze(1) * scale.view(1, C, 1)                           # [B,C,E]
        Lc = torch.einsum("bce,bei,bej->bcij", ws, S, S)                        # [B,C,N,N]
        eye = torch.eye(N, device=a.device, dtype=a.dtype).view(1, 1, N, N)
        chol = torch.linalg.cholesky(eye + Lc)
        flat = B * C
        Q = torch.cholesky_solve(
            eye.expand(B, C, N, N).reshape(flat, N, N), chol.reshape(flat, N, N)
        ).reshape(B, C, N, N)

        r_atom = torch.diagonal(Q, dim1=-2, dim2=-1).permute(0, 2, 1)          # [B,N,C]
        bu = batch["edge_u"].unsqueeze(1).expand(-1, C, -1)                    # [B,C,E]
        bv = batch["edge_v"].unsqueeze(1).expand(-1, C, -1)
        qflat = Q.reshape(flat, N * N)

        def at(i, j):
            # row (b,c) of qflat is exactly Q[b,c]; the offset is the row itself
            idx = (i * N + j).reshape(flat, E)
            return qflat.gather(1, idx).reshape(B, C, E)

        quu, qvv = at(bu, bu), at(bv, bv)
        quv, qvu = at(bu, bv), at(bv, bu)
        p_edge = (ws * (quu + qvv - quv - qvu)).permute(0, 2, 1)               # [B,E,C]
        logz = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)  # [B,C]

        n_atoms = batch["atom_counts"]
        ra = r_atom[batch["atom_batch"], batch["atom_local"]]                  # [A,C]
        pe = p_edge[batch["edge_batch"], batch["edge_local"]]                  # [E,C]
        ha = self.atom_lift(torch.cat([a, ra], dim=1))
        hb = self.bond_lift(torch.cat([b, pe], dim=1))

        has = ha.new_zeros((B, WIDTH)).index_add_(0, batch["atom_batch"], ha)
        has2 = ha.new_zeros((B, WIDTH)).index_add_(0, batch["atom_batch"], ha * ha)
        hbs = hb.new_zeros((B, WIDTH)).index_add_(0, batch["bond_batch"], hb)
        hbs2 = hb.new_zeros((B, WIDTH)).index_add_(0, batch["bond_batch"], hb * hb)
        gstats = (batch["global"] - self.global_mean) / self.global_scale
        pool = torch.cat(
            [has / self.fit_mean_atoms, has2 / self.fit_mean_atoms,
             hbs / self.fit_mean_bonds, hbs2 / self.fit_mean_bonds,
             logz / n_atoms.unsqueeze(1).clamp(min=1.0), gstats,
             torch.log1p(n_atoms).unsqueeze(1),
             torch.log1p(batch["bond_counts"]).unsqueeze(1)],
            dim=1,
        )
        return self.reader(pool).view(-1)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

CANDIDATES = ("QF", "QA", "FC", "FX")


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def build_model(candidate: str, fit: Mapping[str, Any], seed: int, device,
                *, ssl_path=None, dropout: float = 0.0):
    """Build one body; the common front-end is identical across Q and F."""
    torch.manual_seed(int(seed))
    if candidate == "QF":
        model = RootedMeasureModel(adaptive=False, dropout=dropout)
    elif candidate == "QA":
        model = RootedMeasureModel(adaptive=True, dropout=dropout)
    elif candidate == "FC":
        model = ForestStatisticModel(content_weights=True, dropout=dropout)
    elif candidate == "FX":
        model = ForestStatisticModel(content_weights=False, dropout=dropout)
    else:
        raise ValueError(f"unknown candidate {candidate!r}")

    # New (non-front-end) parameters get an independent, recorded RNG stream.
    torch.manual_seed(int(seed) + 90731)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("encoders."):
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                p.zero_()
        # queries are a declared parameter, not a bias: re-randomise them small
        for name, p in model.named_parameters():
            if name == "query":
                p.copy_(torch.randn_like(p) * 0.15)

    info: dict[str, Any] = {"candidate": candidate, "ssl": None}
    model.global_mean.copy_(torch.as_tensor(fit["global_mean"], dtype=torch.float32))
    model.global_scale.copy_(torch.as_tensor(fit["global_scale"], dtype=torch.float32))
    model.fit_mean_atoms.fill_(float(fit["fit_mean_atoms"]))
    model.fit_mean_bonds.fill_(float(fit["fit_mean_bonds"]))
    if candidate in ("QF", "QA"):
        model.fit_group_count.copy_(torch.as_tensor(fit["fit_group_count"], dtype=torch.float32))

    model = model.to(device)
    if ssl_path is not None:
        info["ssl"] = load_ssl(model.encoders, ssl_path, device)
    n_params = parameter_count(model)
    if n_params > 500_000:
        raise RuntimeError(f"{candidate}: {n_params} parameters exceeds the 500k cap")
    info.update(
        {
            "parameters": n_params,
            "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
            "parameters_total": sum(1 for p in model.parameters()),
            "parameters_breakdown": {
                "encoders": parameter_count(model.encoders),
                "head": n_params - parameter_count(model.encoders),
            },
            "rng": {
                "encoder_seed": int(seed),
                "head_seed": int(seed) + 90731,
                "ssl_initialised": ssl_path is not None,
            },
        }
    )
    return model, info
