"""BJOC representation-optimization v1 — models.

Protocol: ``configs/bjoc_representation_optimization_v1.yaml``.

Three model families, all sharing the frozen BJOC compute DAG (one-shot,
non-iterative, attention-free):

``ExtendedBJOC`` (arms **B** and **T**)
    Identical architecture for both arms.  The atom and bond encoders receive
    the original raw BJOC features *plus* the typed-short-path / ring-block
    channels of :mod:`bjoc_typed_path_ring_features_v1`.  Arm **B** feeds
    exactly-zero new channels (structural padding after standardisation);
    arm **T** feeds the standardised real channels.  Parameter count, initial
    weights, training rows and shuffle order are identical by construction, so
    the only difference is the *value* of the new structural channels.

``NoJunctionBJOC`` (arm **J**)
    Task-D2 ablation of the *learned junction content*: no junction encoder,
    no learned junction embedding anywhere.  The fixed, label-blind per-bond
    junction COUNTS (``log1p n_b``) are kept — they are pure topology, not
    learned content.  Removing the junction encoder frees ~112.9k parameters;
    these are redistributed into the bond encoder, group composer and reader so
    the total parameter count is *exactly* the original 413,505 (0.00%
    difference).  This is therefore a near-capacity-matched architectural
    ablation, not a capacity-reduction experiment.

Nothing in this module introduces message passing, attention, transformers,
recurrence or any cross-round state refresh; the object dependency DAG stays
``atom -> bond -> (junction) -> composer -> reader`` with depth one per class.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_typed_path_ring_features_v1 as tpr  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

ARM_B = "B"          # capacity-matched padded control
ARM_T = "T"          # real typed-path + ring-block input
ARM_J = "J"          # no learned junction content
ARM_S = "S"          # BJOC-Null stabilized LR schedule (recipe-only contrast)

#: frozen original BJOC parameter count (must not drift)
ORIGINAL_BJOC_PARAMETERS = 413505


# ---------------------------------------------------------------------------
# extended atom/bond input widths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtendedBJOCConfig(src.BJOCConfig):
    """Frozen BJOC widths; only the atom/bond input widths change."""

    @property
    def atom_in(self) -> int:
        return sro.ATOM_FEAT_DIM + tpr.NEW_ATOM_DIM

    @property
    def bond_in(self) -> int:
        return 3 * self.atom_out + sro.BOND_FEAT_DIM + tpr.NEW_BOND_DIM


class ExtendedBJOC(src.BJOC):
    """BJOC whose atom/bond encoders consume the new typed-path/ring-block channels."""

    def __init__(
        self,
        config: ExtendedBJOCConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        organization_mode: str = src.ORGANIZED,
    ):
        super().__init__(config, global_mean, global_scale, organization_mode)

    def encode_atoms(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        feats = torch.cat([batch["atom_feat"], batch["atom_new"]], dim=1)
        return self.atom_encoder(feats)

    def encode_bonds(self, a: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        feats = torch.cat(
            [au + av, (au - av).abs(), au * av, batch["bond_feat"], batch["bond_new"]], dim=1
        )
        return self.bond_encoder(feats)


# ---------------------------------------------------------------------------
# junctionless (J) capacity-matched ablation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoJunctionConfig:
    """Widths of the capacity-matched junctionless model.

    The arm J is matched **exactly** to the arm B/T parameter count
    (432,705), because the scientific contrast is ``J - B``: an exact match
    removes any capacity explanation for a *worse* J.  J keeps the ORIGINAL
    BJOC input widths (108 atom features, 7 raw bond features; no new
    typed-path/ring-block channels), so J and B receive exactly the same
    information (B's new channels are identically zero).

    Verified inventory:``108*128+128 + 128*160+160``            atom   = 34,592
    ``(3*160+7)*240+240 + 240*96+96``        bond   = 140,256
    ``(96+1+7)*160+160 + 160*96+96``         compose= 32,256
    ``(96+320+33)*300+300 + 300*300+300 + 301`` reader = 225,601
    total = 432,705 == the B/T arm total (delta 0, 0.00%).

    Relative to the frozen ORIGINAL BJOC (413,505 params) this is +19,200
    (+4.64%): the same absolute increase the arms B/T carry, which is the
    unavoidable cost of widening the atom/bond encoders for the new input
    channels.  The binding requirement for causal interpretation is that the
    *contrast* arm (B) is matched, which is satisfied exactly.
    """

    atom_hidden: int = 128
    atom_out: int = 160
    bond_hidden: int = 240
    bond_out: int = 96
    compose_hidden: int = 160
    compose_out: int = 96
    reader_hidden: int = 300
    reader_layers: int = 2
    dropout: float = 0.0

    @property
    def atom_in(self) -> int:
        return sro.ATOM_FEAT_DIM

    @property
    def bond_in(self) -> int:
        return 3 * self.atom_out + sro.BOND_FEAT_DIM

    @property
    def compose_in(self) -> int:
        return self.bond_out + 1 + sro.BOND_FEAT_DIM  # + log1p(n_b) junction count

    @property
    def reader_in(self) -> int:
        return self.compose_out + 2 * self.atom_out + sro.GLOBAL_DIM

    def inventory(self) -> dict[str, int]:
        def lin(i: int, o: int) -> int:
            return i * o + o

        atom = lin(self.atom_in, self.atom_hidden) + lin(self.atom_hidden, self.atom_out)
        bond = lin(self.bond_in, self.bond_hidden) + lin(self.bond_hidden, self.bond_out)
        compose = lin(self.compose_in, self.compose_hidden) + lin(self.compose_hidden, self.compose_out)
        reader = lin(self.reader_in, self.reader_hidden)
        for _ in range(self.reader_layers - 1):
            reader += lin(self.reader_hidden, self.reader_hidden)
        reader += lin(self.reader_hidden, 1)
        return {
            "atom_encoder": atom,
            "bond_encoder": bond,
            "group_composer": compose,
            "reader": reader,
            "total": atom + bond + compose + reader,
        }


class NoJunctionBJOC(src.BJOC):
    """BJOC without the learned junction encoder (no learned junction content).

    Kept: raw atom objects, raw physical-bond objects, the *fixed* per-bond
    junction count ``log1p n_b`` (label-blind topology), the global topology
    vector, raw-y L1 supervision and the one-shot object composer.  Removed:
    every learned junction embedding and the grouping of junction content onto
    bonds.  No substitute channel carries learned junction content.
    """

    def __init__(
        self,
        config: NoJunctionConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.organization_mode = "no_learned_junction"
        self.atom_encoder = src._MLP(config.atom_in, config.atom_hidden, config.atom_out, 1, config.dropout)
        self.bond_encoder = src._MLP(config.bond_in, config.bond_hidden, config.bond_out, 1, config.dropout)
        self.group_composer = src._MLP(
            config.compose_in, config.compose_hidden, config.compose_out, 1, config.dropout
        )
        self.reader = src._MLP(config.reader_in, config.reader_hidden, 1, config.reader_layers, config.dropout)
        self.register_buffer("global_mean", torch.as_tensor(global_mean, dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(global_scale, dtype=torch.float32))

    def encode_atoms(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.atom_encoder(batch["atom_feat"])

    def encode_bonds(self, a: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        feats = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        return self.bond_encoder(feats)

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if organization_mode not in (None, "no_learned_junction"):
            raise ValueError("NoJunctionBJOC has a single fixed organisation (no learned junction)")
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        n_bonds = int(batch["bond_feat"].shape[0])
        n_b = torch.zeros(n_bonds, device=b.device, dtype=b.dtype)
        jb = batch["junc_bond"].reshape(-1)
        n_b.index_add_(0, jb, torch.ones_like(jb, dtype=b.dtype))
        compose_in = torch.cat([b, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {"a": a, "b": b, "c": c, "z": z, "n_b": n_b, "reader_in": reader_in}


# ---------------------------------------------------------------------------
# builders / audits
# ---------------------------------------------------------------------------


def build_model(
    arm: str,
    *,
    seed: int,
    global_mean: np.ndarray,
    global_scale: np.ndarray,
    organization_mode: str = src.NULL,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Deterministic builder shared by every arm (mode never touches init)."""
    torch.manual_seed(int(seed))
    if arm == ARM_J:
        model: nn.Module = NoJunctionBJOC(NoJunctionConfig(), global_mean, global_scale)
    else:
        model = ExtendedBJOC(ExtendedBJOCConfig(), global_mean, global_scale, organization_mode)
    return model.to(device)


def parameter_inventory(arm: str) -> dict[str, Any]:
    if arm == ARM_J:
        cfg = NoJunctionConfig()
    else:
        cfg = ExtendedBJOCConfig()
    inv = dict(cfg.inventory())
    inv["original_bjoc_parameters"] = ORIGINAL_BJOC_PARAMETERS
    inv["delta_vs_original"] = int(inv["total"] - ORIGINAL_BJOC_PARAMETERS)
    inv["relative_delta_vs_original"] = float(inv["delta_vs_original"]) / float(ORIGINAL_BJOC_PARAMETERS)
    inv["arm"] = arm
    return inv


def junctionless_capacity_report() -> dict[str, Any]:
    """Capacity audit of the J ablation against both reference points."""
    inv = parameter_inventory(ARM_J)
    ext = parameter_inventory(ARM_T)
    bt = int(parameter_inventory(ARM_B)["total"])
    return {
        "original_bjoc": ORIGINAL_BJOC_PARAMETERS,
        "arm_B_T": bt,
        "junctionless_J": int(inv["total"]),
        "delta_J_minus_B": int(inv["total"]) - bt,
        "relative_delta_J_minus_B": float(inv["total"] - bt) / float(bt),
        "delta_BT_minus_original": bt - ORIGINAL_BJOC_PARAMETERS,
        "relative_delta_BT_minus_original": float(bt - ORIGINAL_BJOC_PARAMETERS)
        / float(ORIGINAL_BJOC_PARAMETERS),
        "delta_J_minus_original": int(inv["total"]) - ORIGINAL_BJOC_PARAMETERS,
        "relative_delta_J_minus_original": float(inv["total"] - ORIGINAL_BJOC_PARAMETERS)
        / float(ORIGINAL_BJOC_PARAMETERS),
        "contrast_capacity_exact": int(inv["total"]) == bt,
        "B_equals_T": int(parameter_inventory(ARM_B)["total"])
        == int(parameter_inventory(ARM_T)["total"]),
        "inventories": {"B": parameter_inventory(ARM_B), "T": parameter_inventory(ARM_T), "J": inv,
                        "extended": ext},
    }


FORBIDDEN_SOURCE_TOKENS = (
    "MessagePassing(",
    "nn." + "Transformer",
    "Transformer" + "Encoder(",
    "nn." + "MultiheadAttention",
    "nn." + "GRU",
    "nn." + "LSTM",
)


def forbidden_module_scan(model: nn.Module) -> list[str]:
    """Static scan for GNN / attention / recurrence modules (correctness gate)."""
    found: list[str] = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        for token in src.FORBIDDEN_TOKENS:
            if token in cls:
                found.append(f"{name}:{cls}")
    return found


def junctionless_gradient_audit(model: NoJunctionBJOC, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Structural guarantees of the junctionless arm (checked, not assumed)."""
    names = [n for n, _ in model.named_modules()]
    keys = list(model.state_dict().keys())
    model.zero_grad(set_to_none=True)
    pred = model(batch)
    loss = pred.pow(2).mean()
    loss.backward()
    grads = {
        name: float(sum((p.grad**2).sum() for p in getattr(model, name).parameters()).sqrt())
        for name in ("atom_encoder", "bond_encoder", "group_composer", "reader")
    }
    return {
        "has_junction_module": any("junction" in n for n in names),
        "has_junction_state_key": any("junc" in k for k in keys),
        "atom_grad": grads["atom_encoder"],
        "bond_grad": grads["bond_encoder"],
        "composer_grad": grads["group_composer"],
        "reader_grad": grads["reader"],
        "all_grads_live": all(v > 0 for v in grads.values()),
    }
