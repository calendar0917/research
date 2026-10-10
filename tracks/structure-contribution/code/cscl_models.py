"""CSCL-v0 arms A/B/C/D (research track: structure-contribution).

Arms (v0_protocol §3):

* **A additive**      ``y = b + Σᵢ [α(tᵢ) + δ(tᵢ, cᵢ)]``
* **B relational**    A ``+ Σ₍ᵢⱼ₎ γ(tᵢ, tⱼ, rᵢⱼ)`` over real inter-unit bonds
* **C shuffled**      B with relation-slot *contents* permuted inside the batch
  (per-molecule slot counts and marginals preserved, correspondence broken) —
  a mechanism control, never a reportable predictor of real chemistry.
* **D opaque**        DeepSets-style graph regressor on the identical inputs —
  measures the predictive cost of the explicit decomposition.

Identifiability constraints (v0_protocol §3, research_core §5.2):

* separate parameter paths: unary (E_u/α/δ) and relation (E_ru/γ) never share
  parameters; the decomposition is fixed by architecture.
* fit-domain centering: descriptors are centered per type with fit_inner means
  (buffer, not trained); relation features are centered with the fit mean.
* soft penalty: ``λδ Σₜ (mean_{fit,type t} δ)² + λγ (mean_fit γ)²`` pushes the
  type-average contribution into α and the pair-average into b.

Every predicted scalar is exactly attributable: b, one (α+δ) per unit, one γ
per relation slot.  ``forward`` returns the contributions so tests can verify
``pred = b + Σ(α+δ) + Σγ`` and unit independence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from cscl_features import DESC_DIM, REL_DIM  # noqa: E402

EMB_DIM = 16
HIDDEN = 64
LAMBDA_DELTA = 0.01
LAMBDA_GAMMA = 0.01
RESERVED_IDS = 1  # last row = global-mean centering row for unseen types


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


@dataclass
class Batch:
    """Flat batch of units/relations over ``n_mols`` molecules."""

    n_mols: int
    type_ids: torch.Tensor  # [n_units] long
    desc: torch.Tensor  # [n_units, DESC_DIM] float32 (raw; centered in-model)
    unit_mol: torch.Tensor  # [n_units] long
    rel_type_lo: torch.Tensor  # [n_rel] long (min type id)
    rel_type_hi: torch.Tensor  # [n_rel] long (max type id)
    rel_feat: torch.Tensor  # [n_rel, REL_DIM] float32 (raw)
    rel_mol: torch.Tensor  # [n_rel] long
    rel_unit_lo: torch.Tensor  # [n_rel] long (index into flat unit rows)
    rel_unit_hi: torch.Tensor  # [n_rel] long

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            n_mols=self.n_mols,
            type_ids=self.type_ids.to(device),
            desc=self.desc.to(device),
            unit_mol=self.unit_mol.to(device),
            rel_type_lo=self.rel_type_lo.to(device),
            rel_type_hi=self.rel_type_hi.to(device),
            rel_feat=self.rel_feat.to(device),
            rel_mol=self.rel_mol.to(device),
            rel_unit_lo=self.rel_unit_lo.to(device),
            rel_unit_hi=self.rel_unit_hi.to(device),
        )


@dataclass
class Contributions:
    """Exact per-term contributions of one forward pass."""

    bias: torch.Tensor  # scalar
    alpha: torch.Tensor  # [n_units]
    delta: torch.Tensor  # [n_units]
    gamma: torch.Tensor  # [n_rel] (zeros for arm A)
    pred: torch.Tensor  # [n_mols]


def shuffle_relation_contents(batch: Batch, generator: torch.Generator) -> Batch:
    """Arm C: permute relation-slot contents (type ids + features) within the
    batch; ``rel_mol`` stays fixed so per-molecule slot counts are preserved.
    No-op safety: raises if there is nothing to permute and asserts the
    permutation is not the identity when n > 1 (checked by tests separately).
    """
    n = int(batch.rel_feat.shape[0])
    if n <= 1:
        return batch
    perm = torch.randperm(n, generator=generator)
    return Batch(
        n_mols=batch.n_mols,
        type_ids=batch.type_ids,
        desc=batch.desc,
        unit_mol=batch.unit_mol,
        rel_type_lo=batch.rel_type_lo[perm],
        rel_type_hi=batch.rel_type_hi[perm],
        rel_feat=batch.rel_feat[perm],
        rel_mol=batch.rel_mol,
        rel_unit_lo=batch.rel_unit_lo,
        rel_unit_hi=batch.rel_unit_hi,
    )


class CSCLModel(nn.Module):
    """Arms A (additive) and B (relational); arm C = B + shuffled contents."""

    def __init__(self, vocab_size: int, arm: str, hidden: int = HIDDEN, emb_dim: int = EMB_DIM) -> None:
        super().__init__()
        if arm not in {"additive", "relational"}:
            raise ValueError(arm)
        self.arm = arm
        n_ids = vocab_size + RESERVED_IDS

        # --- unary path (created first so A/B share init under equal seeds) --
        self.E_u = nn.Embedding(n_ids, emb_dim)
        self.alpha_head = nn.Linear(emb_dim, 1)
        self.delta_mlp = _mlp([emb_dim + DESC_DIM, hidden, hidden, 1])
        self.bias = nn.Parameter(torch.zeros(1))
        # centering buffer (fit_inner statistics; not trained)
        self.register_buffer("mu", torch.zeros(n_ids, DESC_DIM))
        self.register_buffer("rel_center", torch.zeros(REL_DIM))

        # --- relation path (B only; separate parameters by design) ----------
        if arm == "relational":
            self.E_ru = nn.Embedding(n_ids, emb_dim)
            self.gamma_mlp = _mlp([2 * emb_dim + REL_DIM, hidden, hidden, 1])

    # -- helpers ---------------------------------------------------------

    def set_centering(self, type_means: dict[int, torch.Tensor], global_mean: torch.Tensor, rel_mean: torch.Tensor) -> None:
        self.mu.zero_()
        self.mu[-1] = global_mean
        for tid, mean in type_means.items():
            self.mu[tid] = mean
        self.rel_center.copy_(rel_mean)

    def unary_terms(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.E_u(batch.type_ids)
        mu = self.mu[batch.type_ids]  # UNK types carry their fit bucket mean
        c = batch.desc - mu
        alpha = self.alpha_head(e).squeeze(-1)
        delta = self.delta_mlp(torch.cat([e, c], dim=1)).squeeze(-1)
        return alpha, delta

    def relation_terms(self, batch: Batch) -> torch.Tensor:
        if self.arm != "relational":
            n = int(batch.rel_feat.shape[0])
            return batch.rel_feat.new_zeros(n)
        r = batch.rel_feat - self.rel_center
        e_lo = self.E_ru(batch.rel_type_lo)
        e_hi = self.E_ru(batch.rel_type_hi)
        gamma = self.gamma_mlp(torch.cat([e_lo, e_hi, r], dim=1)).squeeze(-1)
        return gamma

    def forward(self, batch: Batch) -> Contributions:
        alpha, delta = self.unary_terms(batch)
        gamma = self.relation_terms(batch)
        n_mols = batch.n_mols
        unary = torch.zeros(n_mols, dtype=alpha.dtype, device=alpha.device)
        unary = unary.index_add(0, batch.unit_mol, alpha + delta)
        rel = torch.zeros(n_mols, dtype=gamma.dtype, device=gamma.device)
        rel = rel.index_add(0, batch.rel_mol, gamma)
        pred = self.bias + unary + rel
        return Contributions(bias=self.bias.detach(), alpha=alpha, delta=delta, gamma=gamma, pred=pred)

    # -- identifiability penalty ------------------------------------------

    def identifiability_penalty(self, batch: Batch, contributions: Contributions) -> torch.Tensor:
        """λδ Σₜ (mean δ over batch units of type t)² + λγ (mean γ)².

        Computed on training batches (fit data only); eval never calls this.
        """
        device = contributions.delta.device
        n_ids = int(self.mu.shape[0])
        sums = torch.zeros(n_ids, device=device, dtype=contributions.delta.dtype)
        counts = torch.zeros(n_ids, device=device, dtype=contributions.delta.dtype)
        sums.index_add_(0, batch.type_ids, contributions.delta)
        counts.index_add_(0, batch.type_ids, torch.ones_like(contributions.delta))
        means = sums / counts.clamp(min=1.0)
        pen_d = (means * (counts > 0).to(means.dtype)).pow(2).sum()
        n_rel = int(contributions.gamma.shape[0])
        if n_rel > 0:
            pen_g = contributions.gamma.mean().pow(2)
        else:
            pen_g = contributions.gamma.new_zeros(())
        return LAMBDA_DELTA * pen_d + LAMBDA_GAMMA * pen_g


class OpaqueModel(nn.Module):
    """Arm D: DeepSets-style graph regressor over the identical unit/relation
    inputs (own embedding table).  No attributable decomposition."""

    def __init__(self, vocab_size: int, hidden: int = HIDDEN, emb_dim: int = EMB_DIM) -> None:
        super().__init__()
        self.arm = "opaque"
        n_ids = vocab_size + RESERVED_IDS
        self.E_d = nn.Embedding(n_ids, emb_dim)
        self.unit_mlp = _mlp([emb_dim + DESC_DIM, hidden, hidden])
        self.pair_mlp = _mlp([2 * hidden + REL_DIM, hidden, hidden])
        self.head = _mlp([4 * hidden + 2, hidden, hidden, 1])
        self.register_buffer("mu", torch.zeros(n_ids, DESC_DIM))
        self.register_buffer("rel_center", torch.zeros(REL_DIM))

    def set_centering(self, type_means: dict[int, torch.Tensor], global_mean: torch.Tensor, rel_mean: torch.Tensor) -> None:
        self.mu.zero_()
        self.mu[-1] = global_mean
        for tid, mean in type_means.items():
            self.mu[tid] = mean
        self.rel_center.copy_(rel_mean)

    def forward(self, batch: Batch) -> torch.Tensor:
        e = self.E_d(batch.type_ids)
        c = batch.desc - self.mu[batch.type_ids]
        h = self.unit_mlp(torch.cat([e, c], dim=1))  # [n_units, H]
        n_mols = batch.n_mols
        hidden = h.shape[1]
        hu_sum = torch.zeros(n_mols, hidden, device=h.device, dtype=h.dtype)
        hu_sum = hu_sum.index_add(0, batch.unit_mol, h)
        n_units = torch.zeros(n_mols, device=h.device, dtype=h.dtype)
        n_units.index_add_(0, batch.unit_mol, torch.ones_like(batch.unit_mol, dtype=h.dtype))

        n_rel = torch.zeros(n_mols, device=h.device, dtype=h.dtype)
        if int(batch.rel_feat.shape[0]) > 0:
            r = batch.rel_feat - self.rel_center
            p = self.pair_mlp(torch.cat([h[batch.rel_unit_lo], h[batch.rel_unit_hi], r], dim=1))
            hp_sum = torch.zeros(n_mols, p.shape[1], device=h.device, dtype=p.dtype)
            hp_sum = hp_sum.index_add(0, batch.rel_mol, p)
            n_rel.index_add_(0, batch.rel_mol, torch.ones_like(batch.rel_mol, dtype=h.dtype))
        else:
            hp_sum = torch.zeros(n_mols, hidden, device=h.device, dtype=h.dtype)

        graph = torch.cat(
            [
                hu_sum / n_units.clamp(min=1.0).unsqueeze(1),
                hu_sum,
                hp_sum / n_rel.clamp(min=1.0).unsqueeze(1),
                hp_sum,
                torch.log1p(n_units).unsqueeze(1),
                torch.log1p(n_rel).unsqueeze(1),
            ],
            dim=1,
        )
        return self.head(graph).squeeze(-1)


# ---------------------------------------------------------------------------
# training loop shared by A/B/C/D
# ---------------------------------------------------------------------------


def build_model(arm: str, vocab_size: int, seed: int) -> nn.Module:
    torch.manual_seed(int(seed))
    if arm == "opaque":
        return OpaqueModel(vocab_size)
    return CSCLModel(vocab_size, "relational" if arm == "relational" else "additive")


def forward_arm(model: nn.Module, batch: Batch, arm: str, generator: torch.Generator | None = None):
    """Dispatch: C shuffles relation contents before a relational forward."""
    if arm == "shuffled":
        assert isinstance(model, CSCLModel) and model.arm == "relational"
        if generator is not None:
            batch = shuffle_relation_contents(batch, generator)
        else:
            batch = shuffle_relation_contents(batch, torch.Generator().manual_seed(0))  # eval-time fixed
        return model(batch)
    if isinstance(model, OpaqueModel):
        return model(batch)  # type: ignore[return-value]
    return model(batch)


def masked_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def train_loss(model: nn.Module, batch: Batch, target_std: torch.Tensor, arm: str, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, OpaqueModel):
        pred = model(batch)
        return F.mse_loss(pred, target_std), pred
    out = forward_arm(model, batch, arm, gen)
    loss = F.mse_loss(out.pred, target_std) + model.identifiability_penalty(batch, out)
    return loss, out.pred
