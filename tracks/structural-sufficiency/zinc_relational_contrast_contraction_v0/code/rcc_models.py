"""RCC-v0: one shared bond composer, no attention, and a direct graph-level
second-order contraction T = Z^T M Z with M either the real relational contrast
``B = K_R - K_D`` or the row-sum-matched degree null ``K_D``.

Forward graph (no recursive message passing anywhere):

    a, b, w        = P3Encoders(batch)                      # encoders from SSL
    Wg, Vg, n_b    = null grouping of w                     # molecule-level moments
    Wl, Vl         = real junction ownership of w           # per-bond content
    h  = composer([b, Wl, Vl, Wg, Vg, log1p n_b, bond_feat])  # 96D, one per bond
    z  = sum_e h ;  z2 = sum_e h^2
    Z  = Linear96->16(h)
    T  = triu( Z^T M Z )                                    # 136 = upper triangle
    pred = reader([z, z2, sum a, sum a^2, global33, struct5, T/s_T])

The two arms differ *only* in ``M``.  Both arms share every parameter shape,
initial weight, input tensor, batch order and training row.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
import torch.nn as nn

import rcc_common as C
from rcc_common import A2, M1, RANK, T_DIM, WIDTH

BOND_FEAT_DIM = 7
GLOBAL_DIM = 33
N_STRUCT = 5

# composer input layout: [b, Wl, Vl, Wg, Vg, log1p n_b, bond_feat]
COMPOSER_SLICES = {
    "b": (0, 96),
    "Wl": (96, 192),
    "Vl": (192, 288),
    "Wg": (288, 384),
    "Vg": (384, 480),
    "n_b": (480, 481),
    "bond_feat": (481, 488),
}
COMPOSER_IN = 488
# reader input layout
READER_SLICES = {
    "z": (0, 96),
    "z2": (96, 192),
    "a_sum": (192, 288),
    "a_sq": (288, 384),
    "global33": (384, 417),
    "struct": (417, 422),
    "T": (422, 558),
}
READER_IN = 558
NEW_READER_BLOCKS = ("z2", "struct", "T")

_TRIU = torch.triu_indices(RANK, RANK)


class RCC(nn.Module):
    def __init__(
        self,
        norm: Mapping[str, Any],
        *,
        composer_hidden: int = 128,
        reader_hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoders = M1.P3Encoders(dropout=dropout)
        self.composer = M1.MLP(COMPOSER_IN, composer_hidden, WIDTH, 1, dropout)
        self.z_proj = nn.Linear(WIDTH, RANK)
        self.reader = M1.MLP(READER_IN, reader_hidden, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("t_scale", torch.ones(1, dtype=torch.float32))
        self.kernel_mode = "real"

    # -- groupings -------------------------------------------------------------
    @staticmethod
    def group_moments(w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        """Molecule-level junction moments replicated onto every bonded bond."""
        n_graphs = int(batch["n_graphs"])
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        w_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w)
        w2_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w * w)
        cnt = torch.bincount(batch["junc_batch"], minlength=n_graphs).clamp(min=1).to(w.dtype)
        mean_w = w_sum / cnt.unsqueeze(1)
        mean_w2 = w2_sum / cnt.unsqueeze(1)
        return n_b.unsqueeze(1) * mean_w[batch["bond_batch"]], n_b.unsqueeze(1) * mean_w2[batch["bond_batch"]], n_b

    @staticmethod
    def group_ownership(w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        """Real junction ownership: one static incidence sum per bond, no recursion."""
        jb = batch["junc_bond"]
        n = int(batch["bond_feat"].shape[0])
        W = torch.zeros((n, w.shape[1]), device=w.device, dtype=w.dtype)
        V = torch.zeros_like(W)
        W.index_add_(0, jb[:, 0], w)
        W.index_add_(0, jb[:, 1], w)
        w2 = w * w
        V.index_add_(0, jb[:, 0], w2)
        V.index_add_(0, jb[:, 1], w2)
        return W, V

    # -- contraction -----------------------------------------------------------
    @staticmethod
    def contrast(k_pad: torch.Tensor, mask: torch.Tensor):
        """(B_matrix, K_D) from the padded real kernel; zero mass -> zeros."""
        K = k_pad * mask.unsqueeze(1) * mask.unsqueeze(2)
        d = K.sum(-1)
        tot = d.sum(-1)
        denom = tot.clamp(min=1e-12).unsqueeze(1).unsqueeze(2)
        Kd = d.unsqueeze(2) * d.unsqueeze(1) / denom
        Kd = Kd * (tot > 0).view(-1, 1, 1).to(Kd.dtype)
        return K - Kd, Kd

    def forward(self, batch: Mapping[str, torch.Tensor]):
        return self.compute(batch)["pred"]

    def compute(self, batch: Mapping[str, torch.Tensor], *, mode: str | None = None) -> dict:
        """Full forward with intermediates (used by the gates and the analysis)."""
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_moments(w, batch)
        Wl, Vl = self.group_ownership(w, batch)
        h = self.composer(
            torch.cat([b, Wl, Vl, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        B = int(batch["n_graphs"])
        bond_batch = batch["bond_batch"]
        z = h.new_zeros((B, WIDTH)).index_add_(0, bond_batch, h)
        z2 = h.new_zeros((B, WIDTH)).index_add_(0, bond_batch, h * h)

        bmat, kd = self.contrast(batch["k_pad"].to(h.dtype), batch["bond_mask"].to(h.dtype))
        which = self.kernel_mode if mode is None else mode
        M = bmat if which == "real" else kd

        zr = self.z_proj(h)
        bmax = int(batch["k_pad"].shape[1])
        zp = zr.new_zeros((B, bmax, RANK))
        zp[bond_batch, batch["bond_local"]] = zr
        MZ = torch.bmm(M, zp)
        T = torch.einsum("bni,bnj->bij", zp, MZ)[:, _TRIU[0], _TRIU[1]]
        T = T / self.t_scale.to(T.dtype)

        a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        x = torch.cat([z, z2, a_sum, a_sq, g, batch["struct"].to(h.dtype), T], dim=1)
        return {
            "pred": self.reader(x).view(-1),
            "x": x,
            "h": h,
            "z": z,
            "z2": z2,
            "zr": zr,
            "T": T,
            "Tmat": T,
            "B": bmat,
            "Kd": kd,
            "k_pad": batch["k_pad"],
            "d": batch["k_pad"].to(h.dtype).mul(batch["bond_mask"].to(h.dtype).unsqueeze(1)).sum(-1),
        }


def param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def breakdown(module: RCC) -> dict[str, int]:
    return {
        "encoders": param_count(module.encoders),
        "composer": param_count(module.composer),
        "z_proj": param_count(module.z_proj),
        "reader": param_count(module.reader),
        "total": param_count(module),
    }


def build_initial_state(norm: Mapping[str, Any], seed: int) -> tuple[RCC, dict[str, Any]]:
    """One shared state; both arms are deep copies of it.

    The encoders are the historical P3 SSL encoders (loaded exactly as
    ``ad2_models.P3NullModel`` does); composer / z_proj / reader come from a
    dedicated fork of the RNG so that adding this module cannot perturb the
    historical P3 stream.  The three new reader blocks are zero-initialised so
    both arms start at exactly the same function.
    """
    torch.manual_seed(int(seed))
    p3 = A2.P3NullModel(norm)
    cfg = C.config()
    ssl_path = C.REPO / cfg["ssl_encoder"]["path"]
    ssl_info = A2._load_encoders(p3, ssl_path, torch.device("cpu"))
    p3_state = C.C.cpu_state(p3)

    torch.manual_seed(int(seed) + 61121)
    model = RCC(norm)
    enc = {k[len("encoders."):]: v for k, v in p3_state.items() if k.startswith("encoders.")}
    model.encoders.load_state_dict(enc, strict=True)
    with torch.no_grad():
        for name in NEW_READER_BLOCKS:
            lo, hi = READER_SLICES[name]
            model.reader.lins[0].weight[:, lo:hi].zero_()
    # The T scale is a frozen unit choice computed label-free from the fit rows.
    prep_path = C.ANALYSIS / "prep.json"
    t_scale = None
    if prep_path.exists():
        t_scale = float(C.read_json(prep_path)["t_scale"])
        with torch.no_grad():
            model.t_scale.fill_(t_scale)
    info = {
        "seed": int(seed),
        "ssl": ssl_info,
        "ssl_sha256": C.sha256_file(ssl_path),
        "p3_reference_state_sha256": C.state_hash(p3),
        "initial_state_sha256": C.state_hash(model),
        "parameters": param_count(model),
        "parameters_breakdown": breakdown(model),
        "zero_init_reader_blocks": list(NEW_READER_BLOCKS),
        "t_scale": float(model.t_scale.item()),
    }
    return model, info


def build_model(arm: str, norm: Mapping[str, Any], seed: int, device, **_: Any):
    if arm not in ("real", "null"):
        raise ValueError(f"unknown arm {arm!r}")
    shared, info = build_initial_state(norm, seed)
    model = copy.deepcopy(shared)
    model.kernel_mode = arm
    model.to(device)
    out = dict(info)
    out.update({"arm": arm, "kernel_mode": arm, "parameters": param_count(model)})
    return model, out
