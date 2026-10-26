"""Pretext model for the BOSI statistics-only SSL initialisation.

Two heads, both discarded after pretraining, both fed **only** by the frozen
BOSI bond state ``c[e]``:

* ``H_atom`` — for every masked atom ``v``, the input is the sum of ``c[e]``
  over the physical bonds incident to ``v``; the target is ``v``'s raw atom
  type.  ``v`` is a *localised* object (the head knows exactly which atom it is
  predicting), so this is not the forbidden "predict an arbitrary, unlocalised
  partner from the aggregated partner moments" task;
* ``H_bond`` — for every masked physical bond ``e``, the input is ``c[e]``
  itself; the target is ``e``'s raw bond type.

Nothing else is reachable: no ``global33``, no reader, no graph-level pooling,
no message passing beyond the frozen BOSI forward.  ``atom_encoder`` and
``bond_encoder`` are frozen at the historical per-seed SSL checkpoint and
``reader`` is never executed by the pretext path, so the only trainable
parameters are ``half_encoder``, ``side``, ``compose`` and the two heads.
"""

from __future__ import annotations

import contextlib
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import s_common as SC
import s_mask as SM

BM = SC.BM


class PretextHeads(nn.Module):
    def __init__(self, width: int = BM.WIDTH, hidden: int = SC.HEAD_HIDDEN,
                 n_atom: int = 28, n_bond: int = 4):
        super().__init__()
        self.atom_head = BM.MLP(width, hidden, n_atom)
        self.bond_head = BM.MLP(width, hidden, n_bond)

    def forward(self, atom_ctx: torch.Tensor, bond_ctx: torch.Tensor):
        return self.atom_head(atom_ctx), self.bond_head(bond_ctx)


def build_backbone(norm: Mapping[str, object], seed: int, device):
    """The historical BOSI-B initial state of this seed (arm B, no RNG change)."""
    model, info = BM.build_model(norm, "B", seed, device)
    model.arm = "B"
    return model, info


def freeze_encoders(model: BM.BOSI) -> dict:
    """Freeze atom/bond encoders (the historical SSL checkpoints) and the reader."""
    for name in ("atom_encoder", "bond_encoder", "reader"):
        mod = getattr(model, name)
        for p in mod.parameters():
            p.requires_grad_(False)
    trainable = [f"{n}" for n in ("half_encoder", "side", "compose")]
    frozen = [f"{n}" for n in ("atom_encoder", "bond_encoder", "reader")]
    return {"trainable_modules": trainable, "frozen_modules": frozen}


def pretrainable_parameters(model: BM.BOSI) -> list:
    out = []
    for name in ("half_encoder", "side", "compose"):
        out.extend([p for p in getattr(model, name).parameters() if p.requires_grad])
    return out


# ---------------------------------------------------------------------------
# forward: exactly the frozen model forward up to ``c`` (no reader)
# ---------------------------------------------------------------------------
def forward_to_c(model: BM.BOSI, batch: Mapping[str, torch.Tensor]) -> dict:
    a, b = model.encode_tokens(batch)
    x = model.half_edges(batch, a, b)
    M, n = model.relation_stats(batch, x)
    half_deg = batch["half_deg"].view(-1, 1)
    s = model.side(torch.cat(
        [a[batch["half_atom"]], b[batch["half_bond"]], x, M, n, batch["closure"],
         torch.log1p(half_deg)], dim=1))
    hb = batch["half_of_bond"]
    su, sv = s[hb[:, 0]], s[hb[:, 1]]
    c = model.compose(torch.cat(
        [b, su + sv, (su - sv).abs(), su * sv, batch["bond_feat"]], dim=1))
    return {"a": a, "b": b, "x": x, "M": M, "n": n, "s": s, "c": c}


def atom_context(c: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """``sum_{e in Inc(v)} c[e]`` for every atom ``v`` of the batch."""
    agg = c.new_zeros((int(batch["atom_feat"].shape[0]), c.shape[1]))
    bi = batch["bond_index"]
    agg.index_add_(0, bi[:, 0], c)
    agg.index_add_(0, bi[:, 1], c)
    return agg


def pretext_logits(model: BM.BOSI, heads: PretextHeads,
                   batch: Mapping[str, torch.Tensor]) -> dict:
    st = forward_to_c(model, batch)
    agg = atom_context(st["c"], batch)
    atom_logits = heads.atom_head(agg[batch["masked_atom_ids"]])
    bond_logits = heads.bond_head(st["c"][batch["masked_bond_ids"]])
    return {"atom_logits": atom_logits, "bond_logits": bond_logits, "stats": st}


def pretext_loss(model: BM.BOSI, heads: PretextHeads, batch: Mapping[str, torch.Tensor],
                 atom_targets: torch.Tensor, bond_targets: torch.Tensor) -> dict:
    out = pretext_logits(model, heads, batch)
    ce_atom = (F.cross_entropy(out["atom_logits"], atom_targets)
               if atom_targets.numel() else None)
    ce_bond = (F.cross_entropy(out["bond_logits"], bond_targets)
               if bond_targets.numel() else None)
    if ce_atom is None and ce_bond is None:
        raise ValueError("empty pretext batch")
    loss = torch.zeros((), device=out["atom_logits"].device)
    if ce_atom is not None:
        loss = loss + SC.LOSS_WEIGHT_ATOM * ce_atom
    if ce_bond is not None:
        loss = loss + SC.LOSS_WEIGHT_BOND * ce_bond
    return {"loss": loss, "ce_atom": ce_atom, "ce_bond": ce_bond, **out}


# ---------------------------------------------------------------------------
# interventions used by the pilot mechanism read (inference only)
# ---------------------------------------------------------------------------
M_START = 2 * BM.WIDTH + BM.RANK
M_STOP = M_START + SC.BD.N_REL * 3 * BM.RANK


@contextlib.contextmanager
def zero_moment_block(model: BM.BOSI):
    """Force the 384-d partner-moment block of the side input to zero."""
    orig = model.side.forward

    def fwd(x):
        x = x.clone()
        x[:, M_START:M_STOP] = 0.0
        return orig(x)

    model.side.forward = fwd
    try:
        yield
    finally:
        model.side.forward = orig


def shuffle_partner_stats(batch: Mapping[str, torch.Tensor], seed: int) -> dict:
    """Same batch, but every target half-edge gets random partner half-edges.

    ``partner_tgt`` / ``partner_rel`` are untouched, so every target keeps its
    exact count and relation structure; only *which* half-edge states are
    aggregated is randomised.  This is the "replace/scramble the partner
    statistics, keep every other input fixed" probe of PROTOCOL.md §3.
    """
    g = torch.Generator().manual_seed(int(seed))
    ph = batch["partner_half"]
    if ph.numel() == 0:
        return dict(batch)
    perm = torch.randperm(int(ph.numel()), generator=g)
    out = dict(batch)
    out["partner_half"] = ph[perm]
    return out


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def module_grad_norms(model: BM.BOSI, heads: PretextHeads | None = None) -> dict:
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        key = name.split(".")[0]
        out[key] = out.get(key, 0.0) + float(p.grad.detach().pow(2).sum())
    if heads is not None:
        for name, p in heads.named_parameters():
            if p.grad is None:
                continue
            key = f"heads.{name.split('.')[0]}"
            out[key] = out.get(key, 0.0) + float(p.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(out.items())}


def class_report(logits: torch.Tensor, targets: torch.Tensor, n_classes: int) -> dict:
    """CE, accuracy, per-class accuracy and confusion counts (descriptive)."""
    if targets.numel() == 0:
        return {"n": 0}
    ce = float(F.cross_entropy(logits, targets))
    pred = logits.argmax(dim=1)
    correct = pred.eq(targets)
    per_class = {}
    for c in range(n_classes):
        sel = targets == c
        n_c = int(sel.sum())
        if n_c == 0:
            continue
        per_class[str(c)] = {"n": n_c,
                             "acc": float(correct[sel].float().mean()),
                             "pred_top1_share": float((pred[sel] == c).float().mean())}
    counts = torch.bincount(targets, minlength=n_classes).to(torch.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    marginal_ce = float(-(nz * nz.log()).sum())
    return {
        "n": int(targets.numel()),
        "ce": ce,
        "accuracy": float(correct.float().mean()),
        "majority_class": int(counts.argmax()),
        "majority_accuracy": float(counts.max() / counts.sum()),
        "marginal_ce": marginal_ce,
        "per_class": per_class,
    }
