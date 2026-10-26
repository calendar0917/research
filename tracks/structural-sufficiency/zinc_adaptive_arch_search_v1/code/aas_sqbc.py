"""SQBC — Static Query-Conditioned Bond Composer (direction B of AAS v1).

An independent, non-recursive, bond-centred body.  Nothing here is stacked on a
trained DCC-L checkpoint and nothing is iterative:

    a_v      = atom_encoder(atom_feat_v)                           108 -> 128 -> 96
    b_e      = bond_encoder([au+av, |au-av|, au*av, bond_feat_e])  295 -> 160 -> 96
    u_e      = Wq b_e,  w_e = Wk b_e                              96 -> 32 (no bias)
    for every ordered pair (e, f) of distinct bonds sharing a centre v:
        S1 (ordered, target-aware)   x = [u_e, w_f, u_e*w_f, |u_e-w_f|]
        S0 (symmetric, target-blind) x = [u_e+w_f, |u_e-w_f|,
                                          (u_e+w_f)*|u_e-w_f|, u_e*w_f]
        t_ef = [x, a_v, onehot(relation(e,f)), log1p(#partners)]
        w_ef = Phi(t_ef)                                          229 -> 96 -> 32
        P[v,e] = sum_{f != e} w_ef                                  explicit, one pass
    c_e      = Compose([b_e, P[u,e]+P[v,e], |P[u,e]-P[v,e]|,
                        P[u,e]*P[v,e], bond_feat_e, log1p(n_junc_e)])  200 -> 192 -> 96
    z_G      = [sum_e c_e, sum_e c_e^2, sum_v a_v, sum_v a_v^2, global33]
    yhat     = Reader(z_G)                                         417 -> 288 -> 288 -> 1

``S0`` is the matched control: it receives the *same* two bond tokens with the
same parameter shapes, but only through the elementary symmetric invariants of
the unordered pair — a complete invariant of {u_e, w_f}.  The contrast therefore
isolates exactly one question: does the target bond need an asymmetric role in
the pair encoding?

``S2`` is the **pooling-location** control, and it is parameter-identical to S1
(and shares its initial state): the *same* ordered pair content ``w_ef`` is
pooled at the graph level, P3-Null style,

    W_g = mean_{pairs in G} w_ef,   V_g = mean_{pairs in G} w_ef^2

and every bond receives ``[b_e, W_g, V_g, W_g*V_g, bond_feat, log1p(n_junc)]``
— the same three 32-d slots S1 fills with ``[P_u+P_v, |P_u-P_v|, P_u*P_v]``.
S1 vs S2 therefore isolates *where the identical pair content is pooled*
(local ownership per bond vs a graph-level mean), i.e. the DCC-L-L vs
P3-Null question at matched content, matched capacity and matched initial state.

The pair content is used once; ``P`` is never fed back to neighbouring bonds.
There is no junction encoder, no fixed moment bank, no graph-global relation
channel and no attention.  ``_swap_roles`` exists only for the CPU gate that
verifies the asymmetry of S1 (and its absence in S0).
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

import aas_common as _AC  # noqa: F401  (sys.path setup)
import bosi_models as BM

N_REL = BM.N_REL                 # 4 (T / Q / R / N of the frozen BOSI cache)
RANK = 32                        # pair token width
WIDTH = 96                       # atom / bond / bond-content width
PAIR_HIDDEN = 96
COMPOSE_HIDDEN = 192
READER_HIDDEN = 288
PAIR_IN = 4 * RANK + WIDTH + N_REL + 1        # 229
COMPOSE_IN = WIDTH + 3 * RANK + BM.BOND_FEAT + 1   # 200
READER_IN = 4 * WIDTH + BM.GLOBAL_DIM        # 417
ARMS = ("S1", "S0", "S2")


class SQBC(nn.Module):
    def __init__(self, norm: Mapping[str, object], *, mode: str = "S1",
                 dropout: float = 0.0):
        super().__init__()
        if mode not in ARMS:
            raise ValueError(f"unknown SQBC mode {mode!r}")
        self.mode = mode
        self.pooling = "global" if mode == "S2" else "local"
        self.atom_encoder = BM.MLP(BM.N_ATOM_FEAT, 128, WIDTH, 1, dropout)
        self.bond_encoder = BM.MLP(3 * WIDTH + BM.BOND_FEAT, 160, WIDTH, 1, dropout)
        self.pair_q = nn.Linear(WIDTH, RANK, bias=False)
        self.pair_k = nn.Linear(WIDTH, RANK, bias=False)
        self.pair_phi = BM.MLP(PAIR_IN, PAIR_HIDDEN, RANK, 1, dropout)
        self.compose = BM.MLP(COMPOSE_IN, COMPOSE_HIDDEN, WIDTH, 1, dropout)
        self.reader = BM.MLP(READER_IN, READER_HIDDEN, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean",
                             torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale",
                             torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self._swap_roles = False        # diagnostic only, never set in training

    # -- one-shot tokens -----------------------------------------------------
    def encode_tokens(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        b = self.bond_encoder(torch.cat([au + av, (au - av).abs(), au * av,
                                         batch["bond_feat"]], dim=1))
        return a, b

    # -- ordered / symmetric pair content -----------------------------------
    def pair_content(self, batch: Mapping[str, torch.Tensor], a, u, w):
        tgt = batch["partner_tgt"]          # half-edge rows of the target
        part = batch["partner_half"]        # half-edge rows of the partner
        e_t = batch["half_bond"][tgt]       # target bond row
        e_p = batch["half_bond"][part]      # partner bond row
        ut, wp = u[e_t], w[e_p]
        if self._swap_roles:
            ut, wp = wp, ut
        cross = ut * wp
        diff = (ut - wp).abs()
        if self.mode in ("S1", "S2"):
            block = torch.cat([ut, wp, cross, diff], dim=1)
        else:
            s = ut + wp
            block = torch.cat([s, diff, s * diff, cross], dim=1)
        n_par = torch.zeros(int(batch["half_atom"].numel()), 1,
                            dtype=a.dtype, device=a.device)
        n_par.index_add_(0, tgt, torch.ones((tgt.numel(), 1), dtype=a.dtype,
                                            device=a.device))
        rel = torch.nn.functional.one_hot(batch["partner_rel"], N_REL).to(a.dtype)
        centre = a[batch["half_atom"][tgt]]
        feat = torch.cat([block, centre, rel, torch.log1p(n_par[tgt])], dim=1)
        return self.pair_phi(feat), tgt

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_stats: bool = False):
        a, b = self.encode_tokens(batch)
        u = self.pair_q(b)
        w = self.pair_k(b)
        w_ef, tgt = self.pair_content(batch, a, u, w)
        H = int(batch["half_atom"].numel())
        P = a.new_zeros((H, RANK))
        P.index_add_(0, tgt, w_ef)

        hb = batch["half_of_bond"]
        half_deg = batch["half_deg"]
        n_junc = (half_deg[hb[:, 0]] - 1.0 + half_deg[hb[:, 1]] - 1.0).clamp(min=0.0)
        if self.pooling == "local":
            pu, pv = P[hb[:, 0]], P[hb[:, 1]]
            # ONLY symmetric combinations of the two endpoint pair-sums: the
            # storage direction of a bond is arbitrary, so putting ``pu`` and
            # ``pv`` in separate slots would make the prediction depend on it.
            s1_, s2_, s3_ = pu + pv, (pu - pv).abs(), pu * pv
        else:
            # graph-level pooling of the identical pair content (P3-Null form)
            pg = batch["atom_batch"][batch["half_atom"][tgt]]   # graph of each pair
            Bn0 = int(batch["n_graphs"])
            cnt = a.new_zeros((Bn0, 1))
            cnt.index_add_(0, pg, torch.ones((pg.numel(), 1), dtype=a.dtype,
                                             device=a.device))
            w1 = a.new_zeros((Bn0, RANK)).index_add_(0, pg, w_ef)
            w2 = a.new_zeros((Bn0, RANK)).index_add_(0, pg, w_ef * w_ef)
            w1 = w1 / cnt.clamp(min=1.0)
            w2 = w2 / cnt.clamp(min=1.0)
            bb0 = batch["bond_batch"]
            s1_, s2_, s3_ = w1[bb0], w2[bb0], w1[bb0] * w2[bb0]
        c = self.compose(torch.cat(
            [b, s1_, s2_, s3_, batch["bond_feat"],
             torch.log1p(n_junc).unsqueeze(1)], dim=1))
        Bn = int(batch["n_graphs"])
        bb, ab = batch["bond_batch"], batch["atom_batch"]
        zc = c.new_zeros((Bn, WIDTH)).index_add_(0, bb, c)
        zc2 = c.new_zeros((Bn, WIDTH)).index_add_(0, bb, c * c)
        za = c.new_zeros((Bn, WIDTH)).index_add_(0, ab, a)
        za2 = c.new_zeros((Bn, WIDTH)).index_add_(0, ab, a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        z = torch.cat([zc, zc2, za, za2, g], dim=1)
        y = self.reader(z).view(-1)
        if return_stats:
            return y, {"P": P, "c": c, "z": z, "a": a, "b": b, "u": u, "w": w,
                       "n_junc": n_junc}
        return y


def parameter_breakdown(model: nn.Module) -> dict:
    def n(m):
        return int(sum(p.numel() for p in m.parameters()))

    out = {k: n(getattr(model, k)) for k in
           ("atom_encoder", "bond_encoder", "pair_phi", "compose", "reader")}
    out["pair_projections"] = n(model.pair_q) + n(model.pair_k)
    out["total"] = n(model)
    return out


def build_model(norm: Mapping[str, object], arm: str, seed: int, device,
                *, dropout: float = 0.0):
    import aas_common as AC
    import bosi_models as BM2

    if arm not in ARMS:
        raise ValueError(f"unknown SQBC arm {arm!r}")
    torch.manual_seed(int(seed))
    model = SQBC(norm, mode=arm, dropout=dropout)
    info = {"ssl": BM2.load_ssl(model, AC.BC.ssl_path(seed), device)}
    model.to(device)
    info["parameters"] = int(sum(p.numel() for p in model.parameters()))
    info["parameters_breakdown"] = parameter_breakdown(model)
    info["arm"] = arm
    info["seed"] = int(seed)
    if info["parameters"] > AC.PARAM_CAP:
        raise RuntimeError(f"parameter cap exceeded: {info['parameters']}")
    return model, info
