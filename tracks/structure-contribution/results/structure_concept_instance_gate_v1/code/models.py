"""Minimal four-arm models for ``structure_concept_instance_gate_v1``.

The four arms share one local patch encoder (edge-anchored radius-1 patch
feature, dim 35) and one decoder architecture.  They differ only in the
prediction path:

* ``A`` Dense-Bag      : continuous patch embeddings, no instance relations
* ``B`` Dense-Relation : continuous patch embeddings + true instance relations
* ``C`` HardConcept-Bag: VQ concept bottleneck (hard forward) without relations
* ``D`` HardConcept-Rel: VQ concept bottleneck (hard forward) + true relations

Hard-forward guarantee
----------------------
For arms C/D the tensors entering the reader are built from the *hard*
concept identity only:

* the per-patch concept embedding is ``codebook[idx]`` (index = argmax over
  prototypes), passed through a straight-through estimator whose **forward
  value equals the codebook row exactly**;
* the bag histogram is the normalised hard count of concept ids;
* the relation mean/max terms are functions of hard concept rows plus the
  generator-declared structural relation matrix.

``logits_from_concepts(ids, rel, mask)`` recomputes the prediction from concept
ids alone and must agree bit-exactly with the full forward (asserted in the
tests and in the head audit).  It never receives encoder hidden states, soft
concept probabilities, graph ids or undeclared global features.

The decoder (auxiliary reconstruction loss, when enabled) also reads only the
bottleneck representation; there is no bypass decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_FEAT_DIM = 35
REL_FEAT_DIM = 11
ARMS = ("A", "B", "C", "D")
ARM_NAMES = {
    "A": "Dense-Bag",
    "B": "Dense-Relation",
    "C": "HardConcept-Bag",
    "D": "HardConcept-Relation",
}


@dataclass
class ModelConfig:
    arm: str
    feat_dim: int = PATCH_FEAT_DIM
    rel_dim: int = REL_FEAT_DIM
    d: int = 32
    hidden: int = 64
    n_concepts: int = 16
    tau: float = 0.1
    rec_weight: float = 0.5
    commit_weight: float = 0.1
    codebook_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(self.arm)


class PatchEncoder(nn.Module):
    def __init__(self, feat_dim: int, d: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDecoder(nn.Module):
    def __init__(self, feat_dim: int, d: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


class RelationMLP(nn.Module):
    def __init__(self, d: int, rel_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d + rel_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, ei: torch.Tensor, ej: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ei, ej, rel], dim=-1))


class VectorQuantizer(nn.Module):
    """Cosine prototype quantiser with a hard forward and ST gradient."""

    def __init__(self, n_concepts: int, d: int, tau: float = 0.1) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.tau = tau
        proto = torch.randn(n_concepts, d)
        proto = F.normalize(proto, dim=-1)
        self.prototypes = nn.Parameter(proto)

    def similarity(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zn = F.normalize(z, dim=-1)
        pn = F.normalize(self.prototypes, dim=-1)
        logits = zn @ pn.t() / self.tau
        soft = F.softmax(logits, dim=-1)
        idx = logits.argmax(dim=-1)
        return logits, soft, idx

    def hard_rows(self, idx: torch.Tensor) -> torch.Tensor:
        return self.prototypes[idx]

    def straight_through(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return _StraightThrough.apply(z, self.prototypes[idx])


def _masked_mean_max(terms: torch.Tensor, ok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """terms [P,P,d], ok [P,P] bool (diagonal / ablated pairs False)."""
    any_ok = ok.any(dim=-1, keepdim=True)  # [P,1]
    cnt = ok.sum(dim=-1, keepdim=True).clamp_min(1)  # [P,1]
    mean = (terms * ok.unsqueeze(-1)).sum(dim=1) / cnt  # [P,d]
    mean = torch.where(any_ok, mean, torch.zeros_like(mean))
    neg = torch.where(ok.unsqueeze(-1), terms, torch.full_like(terms, float("-inf")))
    mx = neg.max(dim=1).values  # [P,d]
    mx = torch.where(any_ok, mx, torch.zeros_like(mx))
    return mean, mx


class _StraightThrough(torch.autograd.Function):
    """Forward returns ``hard`` exactly; backward passes the gradient to ``z``."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        return hard

    @staticmethod
    def backward(ctx, grad):  # noqa: ANN001
        return grad, None


class GateModel(nn.Module):
    """One of the four frozen arms."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(cfg.feat_dim, cfg.d, cfg.hidden)
        self.decoder = PatchDecoder(cfg.feat_dim, cfg.d, cfg.hidden)
        if cfg.arm in ("B", "D"):
            self.rel_mlp = RelationMLP(cfg.d, cfg.rel_dim, cfg.hidden // 2)
        if cfg.arm in ("C", "D"):
            self.vq = VectorQuantizer(cfg.n_concepts, cfg.d, cfg.tau)
        # reader inputs are fixed by arm
        if cfg.arm == "A":
            in_dim = 2 * cfg.d
        elif cfg.arm == "B":
            in_dim = 4 * cfg.d
        elif cfg.arm == "C":
            in_dim = cfg.n_concepts + cfg.d
        else:
            in_dim = cfg.n_concepts + 3 * cfg.d
        self.reader = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, 1),
        )
        self._last_reader_inputs: dict[str, torch.Tensor] = {}
        self._last_reader_provenance: dict[str, str] = {}

    # -- concept helper ----------------------------------------------------
    def concept_rows(self, idx: torch.Tensor) -> torch.Tensor:
        return self.vq.hard_rows(idx) if hasattr(self, "vq") else torch.empty(0)

    def _hist(self, idx: torch.Tensor, n_patches: int) -> torch.Tensor:
        counts = torch.bincount(idx, minlength=self.cfg.n_concepts).float()
        return counts / max(int(n_patches), 1)

    # -- reader assembly ---------------------------------------------------
    def _reader_inputs(
        self,
        e: torch.Tensor,
        idx: torch.Tensor | None,
        rel: torch.Tensor,
        pair_ok: torch.Tensor,
        patch_ok: torch.Tensor,
        *,
        soft: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, str]]:
        cfg = self.cfg
        p = e.shape[0]
        if cfg.arm == "A":
            keep = patch_ok.unsqueeze(-1).float()
            n = patch_ok.sum().clamp_min(1)
            mean = (e * keep).sum(0) / n
            mx = torch.where(patch_ok.unsqueeze(-1), e, torch.full_like(e, float("-inf"))).max(0).values
            mx = torch.where(patch_ok.any(), mx, torch.zeros_like(mx))
            x = torch.cat([mean, mx])
            inputs = {"bag_mean": mean, "bag_max": mx}
            prov = {"bag_mean": "encoder hidden state (continuous)", "bag_max": "encoder hidden state (continuous)"}
        elif cfg.arm == "B":
            m = self.rel_mlp(e.unsqueeze(1).expand(p, p, -1), e.unsqueeze(0).expand(p, p, -1), rel)
            ok = pair_ok & patch_ok.unsqueeze(1) & patch_ok.unsqueeze(0)
            m_mean, m_max = _masked_mean_max(m, ok)
            keep = patch_ok.unsqueeze(-1).float()
            n = patch_ok.sum().clamp_min(1)
            h_mean = (e * keep).sum(0) / n
            h_max = torch.where(patch_ok.unsqueeze(-1), e, torch.full_like(e, float("-inf"))).max(0).values
            h_max = torch.where(patch_ok.any(), h_max, torch.zeros_like(h_max))
            pi = patch_ok.float() / n
            x = torch.cat([h_mean, h_max, (m_mean * pi.unsqueeze(-1)).sum(0), (m_max * pi.unsqueeze(-1)).sum(0)])
            inputs = {"bag_mean": h_mean, "bag_max": h_max, "pair_mean": m_mean, "pair_max": m_max}
            prov = {
                "bag_mean": "encoder hidden state (continuous)",
                "bag_max": "encoder hidden state (continuous)",
                "pair_mean": "relation MLP over continuous patch embeddings + declared relation matrix",
                "pair_max": "relation MLP over continuous patch embeddings + declared relation matrix",
            }
        elif cfg.arm == "C":
            assert idx is not None
            if soft is None:
                hist = self._hist(idx, p)
                mean = e[patch_ok].mean(0) if patch_ok.any() else torch.zeros(cfg.d)
            else:
                hist = soft[patch_ok].mean(0)
                mean = (e * patch_ok.unsqueeze(-1).float()).sum(0) / patch_ok.sum().clamp_min(1)
            x = torch.cat([hist, mean])
            inputs = {"concept_histogram": hist, "concept_mean": mean}
            prov = {
                "concept_histogram": "normalised hard concept-id counts",
                "concept_mean": "mean of hard concept codebook rows",
            }
        else:
            assert idx is not None
            m = self.rel_mlp(e.unsqueeze(1).expand(p, p, -1), e.unsqueeze(0).expand(p, p, -1), rel)
            ok = pair_ok & patch_ok.unsqueeze(1) & patch_ok.unsqueeze(0)
            m_mean, m_max = _masked_mean_max(m, ok)
            if soft is None:
                hist = self._hist(idx, p)
                e_mean = e[patch_ok].mean(0) if patch_ok.any() else torch.zeros(cfg.d)
            else:
                hist = soft[patch_ok].mean(0)
                e_mean = (e * patch_ok.unsqueeze(-1).float()).sum(0) / patch_ok.sum().clamp_min(1)
            pi = patch_ok.float() / patch_ok.sum().clamp_min(1)
            x = torch.cat([hist, e_mean, (m_mean * pi.unsqueeze(-1)).sum(0), (m_max * pi.unsqueeze(-1)).sum(0)])
            inputs = {
                "concept_histogram": hist,
                "concept_mean": e_mean,
                "pair_mean": m_mean,
                "pair_max": m_max,
            }
            prov = {
                "concept_histogram": "normalised hard concept-id counts",
                "concept_mean": "mean of hard concept codebook rows",
                "pair_mean": "relation MLP over hard concept rows + declared relation matrix",
                "pair_max": "relation MLP over hard concept rows + declared relation matrix",
            }
        self._last_reader_inputs = inputs
        self._last_reader_provenance = prov
        return x, inputs, prov

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        rel: torch.Tensor,
        pair_ok: torch.Tensor,
        patch_ok: torch.Tensor | None = None,
        *,
        mode: str = "hard",
    ) -> tuple[torch.Tensor, dict]:
        """Return (logit[1], info).

        ``mode='soft'`` is a diagnostic that relaxes the hard forward by using
        the soft concept posterior.  It is never used for training or for any
        reported decision except the explicit soft-leakage check.
        """
        cfg = self.cfg
        p = x.shape[0]
        if patch_ok is None:
            patch_ok = torch.ones(p, dtype=torch.bool)
        info: dict = {}
        if cfg.arm in ("A", "B"):
            e = self.encoder(x)
            idx = None
            soft = None
        else:
            z = self.encoder(x)
            logits, soft, idx = self.vq.similarity(z)
            e_hard = self.vq.prototypes[idx]
            if mode == "hard":
                e = self.vq.straight_through(z, idx)
            elif mode == "soft":
                e = soft @ self.vq.prototypes
            else:
                raise ValueError(mode)
            info["concept_logits"] = logits
            info["concept_soft"] = soft
            info["concept_ids"] = idx
            info["z"] = z
            info["hard"] = e_hard
        xin, inputs, prov = self._reader_inputs(e, idx, rel, pair_ok, patch_ok, soft=soft if mode == "soft" else None)
        logit = self.reader(xin)
        if cfg.arm in ("C", "D"):
            info["recon"] = self.decoder(e)
        else:
            info["recon"] = self.decoder(e)
        info["arm"] = cfg.arm
        info["reader_inputs"] = inputs
        info["reader_provenance"] = prov
        return logit.reshape(()), info

    # -- concept-only reader (audit path) -----------------------------------
    def logits_from_concepts(
        self,
        idx: torch.Tensor,
        rel: torch.Tensor,
        pair_ok: torch.Tensor,
        patch_ok: torch.Tensor | None = None,
        *,
        mode: str = "hard",
        extra_patch_mask: torch.Tensor | None = None,
        pair_allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recompute the reader output from hard concept ids and relations only.

        This is the audit path: it never receives encoder hidden states.  It is
        used to (a) prove the full forward is a function of (idx, rel) only and
        (b) run concept interventions.
        """
        if self.cfg.arm not in ("C", "D"):
            raise ValueError("concept-only path exists for arms C/D")
        p = idx.shape[0]
        if patch_ok is None:
            patch_ok = torch.ones(p, dtype=torch.bool)
        if extra_patch_mask is not None:
            patch_ok = patch_ok & extra_patch_mask
        if pair_allowed is not None:
            pair_ok = pair_ok & pair_allowed
        with torch.no_grad():
            e = self.vq.hard_rows(idx)
            soft = None
        xin, _, _ = self._reader_inputs(e, idx, rel, pair_ok, patch_ok)
        return self.reader(xin).reshape(())

    def forward_from_z(
        self,
        z: torch.Tensor,
        rel: torch.Tensor,
        pair_ok: torch.Tensor,
        patch_ok: torch.Tensor | None = None,
        *,
        mode: str = "hard",
    ) -> tuple[torch.Tensor, dict]:
        """Same as forward but skips the patch encoder (audit: reader cannot
        depend on the continuous encoder output beyond the argmax id)."""
        if self.cfg.arm not in ("C", "D"):
            raise ValueError("forward_from_z exists for arms C/D")
        p = z.shape[0]
        if patch_ok is None:
            patch_ok = torch.ones(p, dtype=torch.bool)
        logits, soft, idx = self.vq.similarity(z)
        e_hard = self.vq.prototypes[idx]
        if mode == "hard":
            e = self.vq.straight_through(z, idx)
        else:
            e = soft @ self.vq.prototypes
        xin, inputs, prov = self._reader_inputs(e, idx, rel, pair_ok, patch_ok, soft=soft if mode == "soft" else None)
        logit = self.reader(xin).reshape(())
        return logit, {"concept_ids": idx, "z": z, "hard": e_hard, "reader_inputs": inputs, "reader_provenance": prov}

    # -- parameter accounting ----------------------------------------------
    def n_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters() if trainable_only else self.state_dict().values()
        return sum(int(p.numel()) for p in ps)

    def parameter_breakdown(self) -> dict[str, int]:
        out = {}
        for name, module in (
            ("encoder", self.encoder),
            ("reader", self.reader),
            ("decoder", self.decoder),
            ("rel_mlp", getattr(self, "rel_mlp", None)),
            ("vq", getattr(self, "vq", None)),
        ):
            if module is None:
                continue
            out[name] = sum(int(p.numel()) for p in module.parameters())
        out["total"] = self.n_parameters()
        return out

    def reader_input_table(self) -> list[dict]:
        return [
            {"name": k, "shape": list(v.shape), "provenance": self._last_reader_provenance[k]}
            for k, v in self._last_reader_inputs.items()
        ]


def build_model(arm: str, seed: int, **kwargs) -> GateModel:
    """Deterministic construction; the encoder gets the same init for all arms."""
    cfg = ModelConfig(arm=arm, **kwargs)
    torch.manual_seed(seed)
    return GateModel(cfg)


def shared_encoder_init(seed: int, d: int = 32, hidden: int = 64, feat_dim: int = PATCH_FEAT_DIM) -> dict:
    """Encoder state dict for cross-arm init matching."""
    torch.manual_seed(seed)
    enc = PatchEncoder(feat_dim, d, hidden)
    return {k: v.clone() for k, v in enc.state_dict().items()}


def apply_shared_encoder_init(model: GateModel, state: Mapping[str, torch.Tensor]) -> None:
    model.encoder.load_state_dict({k: v.clone() for k, v in state.items()})


def concept_assignment(model: GateModel, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z = model.encoder(x)
        _, _, idx = model.vq.similarity(z)
    return idx
