"""Energy / free-response arms on one shared head.

Family ``E`` is the frozen RICE-v0 model (``RiceModel``) used unchanged; family
``F`` keeps exactly the same embeddings, order-normalisation, sentence MLP,
readout and beta, deletes the two energy MLPs and inserts the free small-subgraph
response (``sr_fresponse``).  The common head of E and F therefore starts from
byte-identical tensors; only the response module differs.

For both families the pooled sentence sees the response of every rooted support
of size 1..4, either undecomposed (RAW) or after root-anchored Mobius inversion
(MOB).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import rice_energy as RE
import rice_model as RM

import sr_common as C
import sr_fresponse as FR

ARMS = ("E_RAW", "E_MOB", "E_MOB_BIG", "F_RAW", "F_MOB", "F_MOB_BIG")
E_ARMS = {"E_RAW": "RICE_RAW", "E_MOB": "RICE_CUMULANT",
          "E_MOB_BIG": "RICE_CUMULANT"}
F_ARMS = {"F_RAW": "RAW", "F_MOB": "MOB", "F_MOB_BIG": "MOB"}
SIZES = (1, 2, 3, 4)


class EFModel(nn.Module):
    def __init__(self, family: str, arm: str, atom_dims: Sequence[int],
                 bond_dims: Sequence[int], norm: Mapping[str, np.ndarray],
                 dims: Mapping[str, Any], seed: int) -> None:
        super().__init__()
        if family not in ("E", "F"):
            raise ValueError(family)
        self.family = family
        self.arm = arm
        self.rice = RM.RiceModel(
            atom_dims, bond_dims,
            channels=int(dims.get("channels", 16)),
            states=int(dims.get("states", 2)),
            atom_emb=int(dims.get("atom_emb", 16)),
            bond_emb=int(dims.get("bond_emb", 12)),
            atom_hidden=int(dims.get("atom_hidden", 64)),
            bond_hidden=int(dims.get("bond_hidden", 64)),
            potential_bound=float(dims.get("potential_bound", 2.0)),
            h_dim=int(dims.get("h_dim", 96)),
            sentence_hidden=int(dims.get("sentence_hidden", 128)),
            readout_hidden=int(dims.get("readout_hidden", 128)),
            readout_mid=int(dims.get("readout_mid", 64)),
            dropout=float(dims.get("dropout", 0.1)),
            energy_chunk=int(dims.get("energy_chunk", 4096)))
        self.rice.register_buffer(
            "graph_mean", torch.as_tensor(norm["graph_mean"], dtype=torch.float32))
        self.rice.register_buffer(
            "graph_scale", torch.as_tensor(norm["graph_scale"], dtype=torch.float32))
        self.rice.init_parameters(int(seed), last_std=1e-3)
        self.free: FR.FreeResponse | None = None
        if family == "F":
            del self.rice.atom_mlp
            del self.rice.bond_mlp
            self.free = FR.FreeResponse(
                atom_emb=int(dims.get("atom_emb", 16)),
                bond_emb=int(dims.get("bond_emb", 12)),
                support_hidden=int(dims.get("support_hidden", 96)),
                out_dim=int(dims.get("support_out", 32)))
            self.free.init_parameters(int(seed), salt=97)
        self.registry: dict[str, Any] | None = None

    # -- registry -----------------------------------------------------------
    def set_registry(self, tensors: Mapping[str, Any]) -> None:
        self.registry = dict(tensors)

    # -- parameter bookkeeping ---------------------------------------------
    def ledger(self) -> dict[str, int]:
        def n(module: nn.Module) -> int:
            return int(sum(p.numel() for p in module.parameters()))

        keys = ["atom_emb", "bond_emb", "order_norm", "sentence_mlp",
                "readout_mlp"]
        out = {k: n(getattr(self.rice, k)) for k in keys}
        out["beta"] = int(self.rice.beta.numel())
        if self.family == "E":
            out["atom_mlp"] = n(self.rice.atom_mlp)
            out["bond_mlp"] = n(self.rice.bond_mlp)
        else:
            free_ledger = self.free.ledger()
            out.update({k: v for k, v in free_ledger.items() if k != "total"})
        out["total"] = int(sum(v for k, v in out.items() if k != "total"))
        return out

    def init_hashes(self) -> dict[str, str]:
        mods = ["atom_emb", "bond_emb", "order_norm", "sentence_mlp",
                "readout_mlp", "beta"]
        out = C.module_hashes({k: getattr(self.rice, k) for k in mods})
        if self.family == "E":
            out.update(C.module_hashes({"atom_mlp": self.rice.atom_mlp,
                                        "bond_mlp": self.rice.bond_mlp}))
        else:
            out.update(self.free.hashes())
        return out

    # -- fit-only unseen-category zeroing ----------------------------------
    def zero_unseen(self, atom_ok: Sequence[np.ndarray],
                    bond_ok: Sequence[np.ndarray]) -> dict[str, int]:
        return self.rice.zero_unseen(atom_ok, bond_ok)

    # -- forward ------------------------------------------------------------
    def _free_values(self, batch: Mapping[str, torch.Tensor]
                     ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        assert self.free is not None and self.registry is not None
        atom_emb = list(self.rice.atom_emb)
        bond_emb = list(self.rice.bond_emb)
        psi = {1: self.free.psi(batch["type_1"], self.registry, atom_emb, bond_emb),
               2: self.free.psi(batch["type_2"], self.registry, atom_emb, bond_emb),
               3: self.free.psi(batch["type_3"], self.registry, atom_emb, bond_emb),
               4: self.free.psi(batch["type_4"], self.registry, atom_emb, bond_emb)}
        return psi[1], {s: psi[s] for s in (2, 3, 4)}

    def _head(self, batch: Mapping[str, torch.Tensor], values: Mapping[int, torch.Tensor],
              extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        r = self.rice
        u_flat = values[1]
        n_atoms = int(batch["n_atoms"])
        blocks: list[torch.Tensor] = []
        aux: dict[int, Any] = {}
        for order in SIZES:
            if order == 1:
                vals = u_flat
                seg = torch.arange(n_atoms, device=u_flat.device)
            else:
                vals = values[order]
                seg = batch[f"atoms_{order}"][:, 0]
            total, mean, std, count = RE.seg_stats(vals, seg, n_atoms, r.kappa_dim)
            pooled = torch.cat([total, mean, std], dim=1)
            normed = r.order_norm[order - 1](pooled)
            blocks.append(torch.cat([normed, torch.log1p(count).unsqueeze(1)], dim=1))
            aux[order] = vals
        sentence = torch.cat(blocks, dim=1)
        h = r.sentence_mlp(sentence)
        h_sum, h_mean, h_std, atom_count = RE.seg_stats(
            h, batch["atom_mol"], int(batch["n_molecules"]), r.h_dim)
        g = (batch["graph_scalars"] - r.graph_mean) / r.graph_scale
        readout_in = torch.cat([h_sum, h_mean, h_std,
                                torch.log1p(atom_count).unsqueeze(1), g], dim=1)
        if int(readout_in.shape[1]) != r.readout_in_dim:
            raise ValueError(f"readout width {int(readout_in.shape[1])} != {r.readout_in_dim}")
        yhat = r.readout_mlp(readout_in).view(-1) + r.beta
        out: dict[str, Any] = {"yhat_std": yhat}
        if extra is not None:
            out.update(extra)
            out["h"] = h
        return out

    def forward(self, batch: Mapping[str, torch.Tensor], *,
                return_aux: bool = False) -> dict[str, Any]:
        if self.family == "E":
            arm = E_ARMS[self.arm]
            out = self.rice(batch, arm=arm, return_aux=return_aux)
            if return_aux:
                out["response"] = out["kappa"] if arm == "RICE_CUMULANT" else out["phi"]
            return out
        psi1, psi_hi = self._free_values(batch)
        atoms = {s: batch[f"atoms_{s}"] for s in (2, 3, 4)}
        kappa = RE.mobius(psi1, psi_hi, atoms,
                          {s: batch[f"sub2_{s}"] for s in (3, 4)},
                          batch["sub3_4"])
        values = kappa if F_ARMS[self.arm] == "MOB" else psi_hi
        if F_ARMS[self.arm] == "MOB":
            values = {1: kappa[1], **values}
        else:
            values = {1: psi1, **values}
        extra = {"psi1": psi1, "psi": psi_hi, "kappa": kappa} if return_aux else None
        return self._head(batch, values, extra)


def dims_of(cfg: Mapping[str, Any], arm: str) -> dict[str, Any]:
    base = dict(cfg["dims"])
    if arm.endswith("_BIG"):
        base.update(cfg["big_dims"])
    return base


def build_model(arm: str, atom_dims: Sequence[int], bond_dims: Sequence[int],
                seed: int, norm: Mapping[str, np.ndarray],
                cfg: Mapping[str, Any]) -> tuple[EFModel, dict[str, Any]]:
    if arm not in ARMS:
        raise KeyError(arm)
    family = arm[0]
    dims = dims_of(cfg, arm)
    model = EFModel(family, arm, atom_dims, bond_dims, norm, dims, seed)
    info = {"arm": arm, "family": family, "seed": int(seed),
            "dims": dims, "parameters": model.ledger(),
            "init_hashes": model.init_hashes()}
    return model, info


# ---------------------------------------------------------------------------
# BIG / SMALL construction and cross-family copying
# ---------------------------------------------------------------------------

RESPONSE_PREFIXES = ("rice.atom_emb.", "rice.bond_emb.")
HEAD_PREFIXES = ("rice.order_norm.", "rice.sentence_mlp.", "rice.readout_mlp.")
HEAD_KEYS = ("rice.beta", "rice.graph_mean", "rice.graph_scale")


def _copy_matching(dst: nn.Module, src: nn.Module, prefixes: Sequence[str],
                   also: Sequence[str] = ()) -> list[str]:
    """Copy src tensors whose key starts with a prefix and has the same shape."""
    dst_state = dst.state_dict()
    src_state = src.state_dict()
    copied = []
    for key, value in src_state.items():
        if not (any(key.startswith(p) for p in prefixes) or key in also):
            continue
        if key not in dst_state or dst_state[key].shape != value.shape:
            continue
        dst_state[key].copy_(value)
        copied.append(key)
    return copied


def copy_response(dst: EFModel, src: EFModel) -> list[str]:
    copied = _copy_matching(dst, src, RESPONSE_PREFIXES)
    if src.family == "E":
        copied += _copy_matching(dst, src, ("rice.atom_mlp.", "rice.bond_mlp."))
    else:
        copied += _copy_matching(dst, src, ("free.",))
    return copied


def copy_head(dst: EFModel, src: EFModel) -> list[str]:
    return _copy_matching(dst, src, HEAD_PREFIXES, also=HEAD_KEYS)


def build_family_models(cfg: Mapping[str, Any], atom_dims: Sequence[int],
                        bond_dims: Sequence[int], seed: int,
                        norm: Mapping[str, np.ndarray], arm: str
                        ) -> tuple[EFModel, dict[str, Any]]:
    """Build one arm.  BIG arms derive their response from the same-seed SMALL."""
    model, info = build_model(arm, atom_dims, bond_dims, seed, norm, cfg)
    if arm.endswith("_BIG"):
        family = arm[0]
        small_arm = f"{family}_MOB"
        small, _ = build_model(small_arm, atom_dims, bond_dims, seed, norm, cfg)
        copied = copy_response(model, small)
        info["response_copied_from"] = small_arm
        info["response_copied_keys"] = len(copied)
        if family == "F":
            big_e, _ = build_model("E_MOB_BIG", atom_dims, bond_dims, seed, norm, cfg)
            copied_head = copy_head(model, big_e)
            info["head_copied_from"] = "E_MOB_BIG"
            info["head_copied_keys"] = len(copied_head)
        info["init_hashes"] = model.init_hashes()
        info["parameters"] = model.ledger()
    return model, info