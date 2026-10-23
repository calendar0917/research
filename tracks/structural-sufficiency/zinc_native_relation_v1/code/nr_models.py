"""Native P3 graph + one real structure-relation interaction.

Arms
----
P  the frozen ``P3NullModel`` computation graph, instantiated verbatim
   (encoders -> Null composer -> reader), 413,505 parameters.
R  the *same* graph, but after the full P3 encoder pass we form one token per
   existing junction and per original bond,
       h0 = concat(w0, b0)                      # 96 + 96
       u  = LayerNorm(h0 + role_embedding)
       q,k,v = Wq u, Wk u, Wv u                # rank 16
       score_ij = q_i.(k_j + e_rel_ij)/sqrt(16) + bias_rel_ij
       alpha = segment-softmax over the fixed candidate pairs of one molecule
       h1 = h0 + Wo (sum_j alpha_ij v_j)       # no LayerNorm after h1
   split h1 back into w1 (junction) and b1 (bond) and reuse P3's original
   group_to_bonds/composer/reader unchanged.  The main path h0 is never
   normalised; only the relation branch input is.
N  identical to R in parameters, initialisation, pair set, optimiser and data
   order; only the relation-label-to-pair correspondence is replaced by the
   cached within-molecule, within-role-stratum null permutation.

There is no patch encoder, no recursive hidden-state update, no zero-init
learnable gate, no attention stack: exactly one shared rank-16 interaction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
_SS = _HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_SS)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import ad_common as AC  # noqa: E402
import ad_data as AD  # noqa: E402
import ad_models as M1  # noqa: E402
import ad2_models as A2  # noqa: E402

import nr_common as C  # noqa: E402
import nr_data as D  # noqa: E402

D_MODEL = 96
RANK = 16
REL_TYPES = D.REL_TYPES


class NativeRelBlock(nn.Module):
    """One shared rank-16 relational update over junction/bond tokens."""

    def __init__(self, *, d: int = D_MODEL, rank: int = RANK, rel_types: int = REL_TYPES,
                 init_seed: int = 0):
        super().__init__()
        self.d = int(d)
        self.rank = int(rank)
        self.rel_types = int(rel_types)
        # construction under a forked RNG, then explicit generator init, so the
        # common P3 modules are untouched even if construction order changes
        with torch.random.fork_rng(devices=[]):
            self.role = nn.Embedding(2, self.d)
            self.norm = nn.LayerNorm(self.d)
            self.q = nn.Linear(self.d, self.rank, bias=False)
            self.k = nn.Linear(self.d, self.rank, bias=False)
            self.v = nn.Linear(self.d, self.rank, bias=False)
            self.rel_emb = nn.Embedding(self.rel_types, self.rank)
            self.rel_bias = nn.Parameter(torch.zeros(self.rel_types))
            self.out = nn.Linear(self.rank, self.d)
        self.reinit(init_seed)

    def reinit(self, seed: int) -> None:
        g = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.role.weight.normal_(0.0, 0.02, generator=g)
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()
            for lin in (self.q, self.k, self.v):
                fan_in = int(lin.weight.shape[1])
                bound = 1.0 / math.sqrt(fan_in)
                lin.weight.uniform_(-bound, bound, generator=g)
            self.rel_emb.weight.normal_(0.0, 0.02, generator=g)
            self.rel_bias.zero_()
            self.out.weight.normal_(0.0, 1.0e-3, generator=g)
            self.out.bias.zero_()

    def forward(
        self,
        h0: torch.Tensor,
        r_src: torch.Tensor,
        r_dst: torch.Tensor,
        r_rel: torch.Tensor,
        role: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n = int(h0.shape[0])
        zero = h0.new_tensor(0.0)
        stats = {
            "h0_norm_mean": zero,
            "u_norm_mean": zero,
            "message_norm_mean": zero,
            "branch_norm_mean": zero,
            "branch_to_main_ratio": zero,
            "alpha_max_mean": zero,
            "alpha_entropy_mean": zero,
        }
        if n == 0:
            return h0, stats
        if r_src.numel() == 0:
            raise RuntimeError("relation stream has no self loops")

        u = self.norm(h0 + self.role(role))
        q = self.q(u)
        k = self.k(u)
        v = self.v(u)
        rel_e = self.rel_emb(r_rel)
        score = (q[r_src] * (k[r_dst] + rel_e)).sum(dim=-1) / math.sqrt(self.rank)
        score = score + self.rel_bias[r_rel]

        with torch.no_grad():
            seg_max = torch.full((n,), -1.0e9, device=h0.device, dtype=score.dtype)
            seg_max.scatter_reduce_(0, r_src, score.detach(), reduce="amax", include_self=True)
        exp_score = torch.exp(score - seg_max[r_src])
        denom = torch.zeros(n, device=h0.device, dtype=score.dtype)
        denom.index_add_(0, r_src, exp_score)
        alpha = exp_score / denom[r_src].clamp_min(1.0e-12)
        message = v.new_zeros((n, self.rank))
        message.index_add_(0, r_src, alpha.unsqueeze(1) * v[r_dst])
        branch = self.out(message)
        h1 = h0 + branch

        with torch.no_grad():
            amax = torch.zeros(n, device=h0.device, dtype=score.dtype)
            amax.scatter_reduce_(0, r_src, alpha, reduce="amax", include_self=True)
            ent = torch.zeros(n, device=h0.device, dtype=score.dtype)
            ent.index_add_(0, r_src, -alpha * torch.log(alpha.clamp_min(1.0e-12)))
            main = h0.norm(dim=1)
            bnorm = branch.norm(dim=1)
            stats = {
                "h0_norm_mean": main.mean(),
                "u_norm_mean": u.norm(dim=1).mean(),
                "message_norm_mean": message.norm(dim=1).mean(),
                "branch_norm_mean": bnorm.mean(),
                "branch_to_main_ratio": (bnorm / main.clamp_min(1.0e-12)).mean(),
                "alpha_max_mean": amax.mean(),
                "alpha_entropy_mean": ent.mean(),
            }
        return h1, stats


class NativeRelationModel(A2.P3NullModel):
    """P3NullModel with one relation update inserted before the Null composer."""

    def __init__(self, norm: Mapping[str, Any], *, arm: str, seed: int,
                 rel_init_seed: int | None = None, dropout: float = 0.0):
        if arm not in ("R", "N"):
            raise ValueError("NativeRelationModel is only built for R/N")
        # super() constructs the exact P3 common modules in their original order
        # under the supervised seed; the fork keeps this out of the global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            super().__init__(norm, dropout=dropout)
        self.d = int(D_MODEL)
        self.arm = arm
        self.relation_mode = "real" if arm == "R" else "null"
        rel_seed = int(D.REL_INIT_BASE + int(seed)) if rel_init_seed is None else int(rel_init_seed)
        self.relation = NativeRelBlock(init_seed=rel_seed)
        self._last_relation_stats: dict[str, torch.Tensor] = {}

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        relation_mode: str | None = None,
        *,
        bypass: bool = False,
        return_stats: bool = False,
    ):
        mode = self.relation_mode if relation_mode is None else str(relation_mode)
        if mode not in ("real", "null", "bypass"):
            raise ValueError(f"unknown relation mode {mode!r}")
        a, b0, w0 = self.encoders(batch)
        if bypass or mode == "bypass":
            w1, b1 = w0, b0
            rel_stats = {
                "h0_norm_mean": w0.new_tensor(0.0),
                "u_norm_mean": w0.new_tensor(0.0),
                "message_norm_mean": w0.new_tensor(0.0),
                "branch_norm_mean": w0.new_tensor(0.0),
                "branch_to_main_ratio": w0.new_tensor(0.0),
                "alpha_max_mean": w0.new_tensor(0.0),
                "alpha_entropy_mean": w0.new_tensor(0.0),
            }
        else:
            h0 = torch.cat([w0, b0], dim=0)
            rel = batch["r_rel"] if mode == "real" else batch["r_rel_null"]
            nj = int(w0.shape[0])
            role = torch.cat(
                [
                    torch.zeros(nj, dtype=torch.long, device=h0.device),
                    torch.ones(int(b0.shape[0]), dtype=torch.long, device=h0.device),
                ],
                dim=0,
            )
            h1, rel_stats = self.relation(h0, batch["r_src"], batch["r_dst"], rel, role)
            w1, b1 = h1[:nj], h1[nj:]
        self._last_relation_stats = {k: v.detach() for k, v in rel_stats.items()}

        W, V, n_b = self.group_to_bonds(w1, batch)
        compose_in = torch.cat(
            [b1, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = c.new_zeros((n_graphs, c.shape[1])).index_add_(0, batch["bond_batch"], c)
        a_sum = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        pred = self.reader(torch.cat([z, a_sum, a_sq, g], dim=1)).view(-1)
        if return_stats:
            return pred, {"relation": rel_stats, "a": a, "b0": b0, "w0": w0, "b1": b1, "w1": w1}
        return pred


# ---------------------------------------------------------------------------
# builders / hashing
# ---------------------------------------------------------------------------


def load_ssl_encoders(model: nn.Module, ssl_path: Path) -> dict[str, Any]:
    state = torch.load(ssl_path, map_location="cpu", weights_only=False)
    enc = model.encoders if hasattr(model, "encoders") else model
    loaded = {}
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        if name not in state:
            raise SystemExit(f"SSL checkpoint {ssl_path} has no {name}")
        getattr(enc, name).load_state_dict(state[name], strict=True)
        loaded[name] = C.sha256_state_dict(state[name])
    return {
        "source": str(ssl_path.relative_to(C.REPO)),
        "sha256": C.sha256_file(ssl_path),
        "encoder_tensor_hashes": loaded,
        "note": "train-only masked atom/bond pretext; same checkpoint for supervised seed 0 and 1",
    }


def _ssl_path_from_config() -> Path:
    cfg = C.load_config()
    path = C.REPO / cfg["ssl"]["path"]
    if not path.exists():
        raise SystemExit(f"missing SSL checkpoint: {path}")
    got = C.sha256_file(path)
    if got != cfg["ssl"]["sha256"]:
        raise SystemExit(f"SSL checkpoint hash mismatch: {got} != {cfg['ssl']['sha256']}")
    return path


def common_state_hash(model: nn.Module) -> str:
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("relation.")}
    return C.sha256_state_dict(state)


def relation_state_hash(model: nn.Module) -> str:
    if not hasattr(model, "relation"):
        return ""
    return C.sha256_state_dict(model.relation.state_dict())


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    out = {name: int(sum(p.numel() for p in mod.parameters())) for name, mod in model.named_children()}
    out["total"] = int(sum(p.numel() for p in model.parameters()))
    return out


def build_arm(arm: str, norm: Mapping[str, Any], seed: int, device, *, ssl_path: Path | None = None):
    if arm not in C.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    ssl_path = _ssl_path_from_config() if ssl_path is None else Path(ssl_path)
    if arm == "P":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            model = A2.P3NullModel(norm)
    else:
        model = NativeRelationModel(norm, arm=arm, seed=int(seed))
    ssl_info = load_ssl_encoders(model, ssl_path)
    model = model.to(device)
    params = int(sum(p.numel() for p in model.parameters()))
    info = {
        "arm": arm,
        "seed": int(seed),
        "ssl": ssl_info,
        "parameters": params,
        "parameters_total": params,
        "requires_grad_tensors": sum(1 for p in model.parameters() if p.requires_grad),
        "parameter_breakdown": parameter_breakdown(model),
        "common_state_sha256": common_state_hash(model),
        "relation_state_sha256": relation_state_hash(model),
    }
    if arm == "P":
        info["common_state_sha256"] = C.sha256_state_dict(model.state_dict())
    return model, info


def build_p3_mature(norm: Mapping[str, Any], seed: int, device):
    """The historical recipe builder for P (used only for a seed-0 equality gate)."""
    return A2.build_model("P3N", norm, int(seed), device)


def load_checkpoint(model: nn.Module, path: Path, device) -> dict[str, Any]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state_dict"] if "state_dict" in ck else ck.get("state", ck)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    return ck


def state_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["state_dict"] if "state_dict" in ck else ck.get("state", ck)
