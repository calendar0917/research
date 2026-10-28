"""CJP-SSL models: frozen LCRC-A body + a throw-away 63-way pretext head.

Nothing here changes the deployed inference function of LCRC-A.  The pretext
head (``Linear(192,128) -> SiLU -> Linear(128,63)``) exists only in the
pretraining code and is never part of a supervised checkpoint.

Two arms differ **only** in how the 63 joint logits are read out:

* ``JOINT``: ordinary CE on ``flatten(L)``.
* ``MARG`` : CE on ``flatten(L_marg)`` with
  ``L_marg[a,b] = mean_b L[a,b] + mean_a L[a,b] - mean_ab L[a,b]``.

``softmax(L_marg)`` factorises exactly into the product of the two conditional
marginals, so the MARG arm can only ever express factorised joint
distributions.  It is the matched control for "does the true joint conditional
actually matter".
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cjp_common as C

M = C.M
L = C.L


class PretextHead(nn.Module):
    """Discarded-after-pretraining head: 192 -> 128 -> 63 (SiLU)."""

    def __init__(self, q_dim: int = 192, hidden: int = 128, out: int = C.N_TARGET_CLASSES):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(q_dim, hidden), nn.Linear(hidden, out)])

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.lins[1](F.silu(self.lins[0](q)))


# ---------------------------------------------------------------------------
# MARG projection and joint interaction
# ---------------------------------------------------------------------------


def marg_projection(logits: torch.Tensor) -> torch.Tensor:
    """Additive (marginally-factorised) projection of a 63-way joint table."""
    shape = logits.shape
    table = logits.reshape(-1, C.TARGET_ATOM_CLASSES, C.TARGET_BOND_CLASSES)
    row = table.mean(dim=2, keepdim=True)
    col = table.mean(dim=1, keepdim=True)
    glob = table.mean(dim=(1, 2), keepdim=True)
    return (row + col - glob).reshape(*shape)


def interaction_residual(logits: torch.Tensor) -> torch.Tensor:
    return logits - marg_projection(logits)


def interaction_rms(logits: torch.Tensor) -> float:
    return float(interaction_residual(logits).detach().pow(2).mean().sqrt())


def marg_softmax_factors(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row/column marginal logits whose softmax product equals softmax(L_marg)."""
    table = logits.reshape(-1, C.TARGET_ATOM_CLASSES, C.TARGET_BOND_CLASSES)
    return table.mean(dim=2), table.mean(dim=1)


# ---------------------------------------------------------------------------
# frozen-body construction
# ---------------------------------------------------------------------------


def _head_init(seed: int, device) -> PretextHead:
    torch.manual_seed(int(seed) + int(C.PRETRAIN["head_init_seed_offset"]))
    head = PretextHead()
    return head.to(device)


def joint_state_hash(model: nn.Module, head: nn.Module) -> str:
    h = C.state_hash(model) + "|" + C.state_hash(head)
    return C.sha256_bytes(h.encode())


def build_ssl_model(
    norm: Mapping[str, Any],
    seed: int,
    device,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, PretextHead, dict[str, Any]]:
    """Shared construction for both pretraining arms (bit-identical init)."""
    torch.manual_seed(int(seed))
    model = M.LCRC("A", norm, cfg)
    ssl = M.load_ssl_encoders(model, M.ssl_path(seed))
    for p in model.atom_encoder.parameters():
        p.requires_grad_(False)
    for p in model.bond_encoder.parameters():
        p.requires_grad_(False)
    head = _head_init(seed, device)
    model.to(device)
    encoder_sha = {
        "atom_encoder": C.state_hash(model.atom_encoder),
        "bond_encoder": C.state_hash(model.bond_encoder),
    }
    local_sha = C.state_hash(model.local)
    info = {
        "seed": int(seed),
        "ssl": ssl,
        "parameters_total": int(sum(p.numel() for p in model.parameters())),
        "parameters_local": int(sum(p.numel() for p in model.local.parameters())),
        "parameters_head": int(sum(p.numel() for p in head.parameters())),
        "parameters_trainable": int(
            sum(p.numel() for p in model.local.parameters())
            + sum(p.numel() for p in head.parameters())
        ),
        "encoder_sha256": encoder_sha,
        "local_sha256": local_sha,
        "head_sha256": C.state_hash(head),
        "joint_init_sha256": joint_state_hash(model, head),
        "frozen_modules": ["atom_encoder", "bond_encoder"],
        "excluded_modules": ["f", "rho"],
        "config": dict(model.cfg),
    }
    return model, head, info


def build_supervised_model(
    norm: Mapping[str, Any],
    seed: int,
    device,
    cfg: Mapping[str, Any] | None = None,
    *,
    load_ssl: bool = True,
    local_state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build LCRC-A in exactly the frozen order and optionally swap LocalContent.

    * ``load_ssl=True`` (A-REPLAY / MARG-SSL / JOINT-SSL) loads the historical
      atom/bond masked-SSL checkpoint through ``lc_models.ssl_path(seed)``.
    * ``load_ssl=False`` (SCRATCH) leaves the freshly seeded atom/bond encoders.
      Construction consumes exactly the same RNG stream, so ``local``/``f``/
      ``rho`` are bit-identical to the ``load_ssl=True`` build.
    * ``local_state`` replaces only ``LocalContent`` after construction.
    """
    torch.manual_seed(int(seed))
    model = M.LCRC("A", norm, cfg)
    ssl_info: dict[str, Any]
    if load_ssl:
        ssl_info = M.load_ssl_encoders(model, M.ssl_path(seed))
        ssl_info["loaded"] = True
    else:
        ssl_info = {"loaded": False, "source": None, "sha256": None,
                    "loaded_modules": [], "reason": "SCRATCH control"}
    local_info: dict[str, Any] = {"replaced": False, "source": None, "sha256": None,
                                 "tensors": 0}
    if local_state is not None:
        state = {k: torch.as_tensor(v).detach().clone() for k, v in local_state.items()}
        model.local.load_state_dict(state)
        local_info = {
            "replaced": True,
            "source": "cjp_ssl_pretrain_local_state",
            "sha256": C.sha256_bytes(
                b"".join(
                    np.ascontiguousarray(state[k].cpu().numpy(), dtype=np.float32).tobytes()
                    for k in sorted(state)
                )
            ),
            "tensors": len(state),
            "keys": sorted(state),
        }
    hashes = C.module_hashes(model)
    params = int(sum(p.numel() for p in model.parameters()))
    if params != int(C.DEPLOY_PARAMETERS):
        raise RuntimeError(f"deployed parameter count {params} != {C.DEPLOY_PARAMETERS}")
    info = {
        "arm": "A",
        "seed": int(seed),
        "config": dict(model.cfg),
        "ssl": ssl_info,
        "local_state": local_info,
        "parameters": params,
        "parameters_breakdown": {
            name: int(sum(p.numel() for p in mod.parameters()))
            for name, mod in model.named_children()
        },
        "module_sha256": hashes,
        "initial_state_sha256": C.state_hash(model),
        "has_junction_module": False,
        "state_keys_with_junction": [],
    }
    model.to(device)
    return model, info


# ---------------------------------------------------------------------------
# pretext forward
# ---------------------------------------------------------------------------


def query_representation(h: torch.Tensor, query: Mapping[str, torch.Tensor]) -> torch.Tensor:
    hv = h[query["v"]]
    hw = h[query["w"]]
    return torch.cat([hv, hw, hv * hw], dim=1)


def pretext_logits(
    model: nn.Module,
    head: PretextHead,
    batch: Mapping[str, torch.Tensor],
    query: Mapping[str, torch.Tensor],
    *,
    arm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(used_logits, raw_joint_logits, h)`` on the masked graph."""
    a, b, h = model._encode(batch)
    q = query_representation(h, query)
    raw = head(q)
    used = marg_projection(raw) if arm == "MARG" else raw
    return used, raw, h


def per_query_ce(used_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(used_logits, target, reduction="none")


def molecule_mean_loss(
    ce: torch.Tensor,
    query: Mapping[str, torch.Tensor],
    n_molecules: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average CE within a molecule, then over molecules that have paths."""
    device = ce.device
    sums = torch.zeros(int(n_molecules), device=device, dtype=ce.dtype)
    counts = torch.zeros(int(n_molecules), device=device, dtype=ce.dtype)
    sums = sums.index_add(0, query["graph"], ce)
    counts = counts.index_add(0, query["graph"], torch.ones_like(ce))
    valid = counts > 0
    if not bool(valid.any()):
        return ce.new_zeros(()), sums, counts
    return (sums[valid] / counts[valid]).mean(), sums, counts


def effective_rank(h: torch.Tensor) -> dict[str, float]:
    """Effective rank (exp of singular-value entropy) and activation RMS."""
    x = h.detach().double().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    p = s * s
    total = p.sum()
    if total <= 0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0, "activation_rms": 0.0}
    p = p / total
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    return {
        "effective_rank": float(np.exp(ent)),
        "participation_ratio": float(1.0 / (p * p).sum()),
        "activation_rms": float(np.sqrt((x * x).mean())),
        "n_vectors": int(x.shape[0]),
        "dim": int(x.shape[1]),
    }


__all__ = [
    "PretextHead", "marg_projection", "interaction_residual", "interaction_rms",
    "marg_softmax_factors", "build_ssl_model", "build_supervised_model",
    "query_representation", "pretext_logits", "per_query_ce",
    "molecule_mean_loss", "effective_rank", "joint_state_hash",
]
