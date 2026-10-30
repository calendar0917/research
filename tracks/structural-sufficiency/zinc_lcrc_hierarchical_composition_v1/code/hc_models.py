"""LCRC-HC3-v1 bodies: B (frozen LCRC-A), H (hierarchical), F (flat).

The shared trunk is literally the frozen
``zinc_local_relation_composition_v1/code/lc_models.py``: atom/bond encoders,
``LocalContent`` statistics, ``f`` and ``rho`` keep their classes, construction
order and initial tensors.  The new composition path is attached as:

    z triples / directed 3-bond paths (label-blind, static)
        -> q_path  (H: F2 484->96->48 then F3 252->96->64;
                    F: Fflat 684->109->64 on forward and reverse, averaged)
        -> m_e = sum_oriented q_path / (2 sqrt(1+n_e))       (per physical bond)
        -> r_v = sum_{e incident v} m_e / sqrt(1+degree_v)   (per atom)
        -> W_comp(r_v), Linear(64,64) with zero weight **and** zero bias
        -> h_v = LN(MLP(X_v) + W_comp(r_v))

Step 0 therefore reproduces the frozen LCRC-A forward bit-for-bit.  The
projection receives gradient at step 1; the path encoder receives a non-zero
gradient only from step 2 onward (its Jacobian is multiplied by the still-zero
``W_comp`` at the first backward pass).  The CPU gates check both facts.

Only B is constructed through the frozen builder.  H/F construct the frozen
``LCRC`` first (identical RNG consumption and SSL load), then replace the
``LocalContent`` instance by an otherwise identical subclass that accepts the
injection vector, and finally draw the new modules under a dedicated seed.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import hc_common as C

LC = C.LC
LM = C.LM

KINDS = ("B", "H", "F")

N_CHORD = 4
Z_DIM = 48
F3_OUT = 64
F2_IN = 3 * 96 + 2 * 96 + N_CHORD                    # 484
F3_IN = 3 * Z_DIM + 96 + 3 * N_CHORD                 # 252
FLAT_IN = 4 * 96 + 3 * 96 + 3 * N_CHORD              # 684


class CompLocalContent(LM.LocalContent):
    """Frozen ``LocalContent`` with one extra pre-LayerNorm injection."""

    def forward(self, batch, a, b, scal, comp=None):
        n_atoms = a.shape[0]
        width = a.shape[1]
        bond_index = batch["bond_index"]
        u, v = bond_index[:, 0], bond_index[:, 1]
        degree = a.new_zeros(n_atoms)
        degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
        degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
        b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        b_mean = b_sum / degree.clamp(min=1.0).unsqueeze(1)

        nbr = a.new_zeros((n_atoms, self.rank)).index_add_(
            0, batch["nbr_atom"], self.nbr_emb(batch["nbr_code"])
        )
        path_joint = self.joint_proj(
            self.path_u(batch["path_cat"]) * self.path_v(batch["path_end63"])
        )
        path_emb = torch.cat(
            [
                self.path_emb(batch["path_cat"]),
                self.end_emb(batch["path_end63"]),
                path_joint,
                self.close_emb(batch["path_closure"]),
            ],
            dim=1,
        )
        path_agg = a.new_zeros((n_atoms, path_emb.shape[1])).index_add_(
            0, batch["path_atom"], path_emb
        )
        gate = self.path_gate(batch["path_cat"])
        path_content = a[batch["path_end"]] * gate
        content_agg = a.new_zeros((n_atoms, width)).index_add_(
            0, batch["path_atom"], path_content
        )
        x = torch.cat([a, b_sum, b_mean, nbr, path_agg, content_agg, scal], dim=1)
        pre = self.mlp(x)
        if comp is not None:
            pre = pre + comp
        return self.norm(pre)


class HPathEncoder(nn.Module):
    """CMA-verified fragment semantics re-deployed on the LCRC interfaces."""

    def __init__(self):
        super().__init__()
        self.F2 = LM.MLP([F2_IN, 96, Z_DIM])
        self.F3 = LM.MLP([F3_IN, 96, F3_OUT])

    def forward(self, batch: Mapping[str, torch.Tensor],
                a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        za = batch["hc_z_atom"]
        zb = batch["hc_z_bond"]
        ch = F.one_hot(batch["hc_z_chord"], N_CHORD).to(a.dtype)
        # z_ijk = 0.5 [F2(i,j,k) + F2(k,j,i)], exactly as frozen CMA
        fwd = torch.cat([a[za[:, 1]], a[za[:, 0]], a[za[:, 2]],
                         b[zb[:, 0]], b[zb[:, 1]], ch], dim=1)
        rev = torch.cat([a[za[:, 2]], a[za[:, 0]], a[za[:, 1]],
                         b[zb[:, 1]], b[zb[:, 0]], ch], dim=1)
        out = self.F2(torch.cat([fwd, rev], dim=0))
        T = za.shape[0]
        out = out.view(2, T, Z_DIM)
        z = 0.5 * (out[0] + out[1])

        pz = batch["hc_path_z"]
        zL, zR = z[pz[:, 0]], z[pz[:, 1]]
        cd = batch["hc_path_chord"]
        oh_ik = F.one_hot(cd[:, 0], N_CHORD).to(a.dtype)
        oh_jl = F.one_hot(cd[:, 1], N_CHORD).to(a.dtype)
        oh_il = F.one_hot(cd[:, 2], N_CHORD).to(a.dtype)
        rel = torch.cat([oh_ik + oh_jl, (oh_ik - oh_jl).abs(), oh_il], dim=1)
        b_centre = b[batch["hc_path_bond"]]
        u = torch.cat([zL + zR, (zL - zR).abs(), zL * zR, b_centre, rel], dim=1)
        return self.F3(u)


class FPathEncoder(nn.Module):
    """Direct encoding of the full four-atom / three-bond path."""

    def __init__(self):
        super().__init__()
        self.Fflat = LM.MLP([FLAT_IN, 109, F3_OUT])

    @staticmethod
    def _encode(atoms, bonds, chords, a, b):
        oh_ik = F.one_hot(chords[:, 0], N_CHORD).to(a.dtype)
        oh_jl = F.one_hot(chords[:, 1], N_CHORD).to(a.dtype)
        oh_il = F.one_hot(chords[:, 2], N_CHORD).to(a.dtype)
        return torch.cat([
            a[atoms[:, 0]], a[atoms[:, 1]], a[atoms[:, 2]], a[atoms[:, 3]],
            b[bonds[:, 0]], b[bonds[:, 1]], b[bonds[:, 2]],
            oh_ik, oh_jl, oh_il,
        ], dim=1)

    def forward(self, batch: Mapping[str, torch.Tensor],
                a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        atoms = batch["hc_path_atom"]
        bonds = batch["hc_path_bonds"]
        cd = batch["hc_path_chord"]
        fwd = self._encode(atoms, bonds, cd, a, b)
        # reverse path (l,k,j,i): bonds reversed, and chord_ik <-> chord_jl
        rev_chords = torch.stack([cd[:, 1], cd[:, 0], cd[:, 2]], dim=1)
        rev = self._encode(atoms.flip(1), bonds.flip(1), rev_chords, a, b)
        return 0.5 * (self.Fflat(fwd) + self.Fflat(rev))


class HCLCRC(LM.LCRC):
    """Frozen LCRC body plus the shared hierarchical/flat composition path."""

    kind = "H"

    def bond_m(self, batch: Mapping[str, torch.Tensor], q: torch.Tensor) -> torch.Tensor:
        """``m_e = sum_oriented q_path / (2 sqrt(1+n_e))`` on every physical bond."""
        bond_index = batch["bond_index"]
        n_bonds = int(bond_index.shape[0])
        n_e = batch["hc_event_n"].to(q.dtype)
        num = q.new_zeros((n_bonds, q.shape[1])).index_add_(0, batch["hc_path_bond"], q)
        m = num / (2.0 * torch.sqrt(1.0 + n_e)).unsqueeze(1)
        return torch.where((n_e > 0).unsqueeze(1), m, torch.zeros_like(m))

    def atom_r(self, batch: Mapping[str, torch.Tensor], m: torch.Tensor) -> torch.Tensor:
        """``r_v = sum_{e incident v} m_e / sqrt(1+degree_v)`` on every atom."""
        bond_index = batch["bond_index"]
        u, v = bond_index[:, 0], bond_index[:, 1]
        n_atoms = int(batch["atom_batch"].shape[0])
        degree = m.new_zeros(n_atoms).index_add_(0, u, torch.ones_like(u, dtype=m.dtype))
        degree = degree.index_add_(0, v, torch.ones_like(v, dtype=m.dtype))
        r = m.new_zeros((n_atoms, m.shape[1])).index_add_(0, u, m).index_add_(0, v, m)
        return r / torch.sqrt(1.0 + degree).unsqueeze(1)

    def aggregate(self, batch: Mapping[str, torch.Tensor],
                  q: torch.Tensor) -> torch.Tensor:
        """The shared key->atom aggregation interface."""
        return self.atom_r(batch, self.bond_m(batch, q))

    def _encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bond_feat = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        b = self.bond_encoder(bond_feat)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        q = self.path_encoder(batch, a, b)
        r = self.aggregate(batch, q)
        comp = self.comp_proj(r)
        h = self.local(batch, a, b, scal, comp)
        return a, b, h


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _breakdown(model: nn.Module) -> dict[str, int]:
    out = {
        "atom_encoder": _param_count(model.atom_encoder),
        "bond_encoder": _param_count(model.bond_encoder),
        "local": _param_count(model.local),
        "f": _param_count(model.f),
        "rho": _param_count(model.rho),
        "base_total": _param_count(model.atom_encoder) + _param_count(model.bond_encoder)
        + _param_count(model.local) + _param_count(model.f) + _param_count(model.rho),
    }
    if hasattr(model, "comp_proj"):
        out["comp_proj"] = _param_count(model.comp_proj)
        out["path_encoder"] = _param_count(model.path_encoder)
        if model.kind == "H":
            out["F2"] = _param_count(model.path_encoder.F2)
            out["F3"] = _param_count(model.path_encoder.F3)
        else:
            out["Fflat"] = _param_count(model.path_encoder.Fflat)
    out["total"] = _param_count(model)
    return out


def _check_params(kind: str, total: int) -> None:
    expected = {"B": 237_519, "H": 323_391, "F": 323_384}[kind]
    if total != expected:
        raise RuntimeError(f"{kind}: parameter count {total} != frozen {expected}")
    if total > C.PARAM_CAP:
        raise RuntimeError(f"{kind}: {total} parameters exceeds cap {C.PARAM_CAP}")


def build_model(kind: str, norm: Mapping[str, Any], seed: int, device,
                cfg: Mapping[str, Any] | None = None):
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    cfg = dict(LM.DEFAULT_CONFIG if cfg is None else {**LM.DEFAULT_CONFIG, **dict(cfg)})

    if kind == "B":
        model, info = LM.build_model("A", norm, seed, device, cfg)
        info = dict(info)
        info["kind"] = "B"
        info["parameters_breakdown"] = _breakdown(model)
        _check_params("B", info["parameters"])
        return model, info

    # --- frozen base construction (same seed and RNG consumption) ---------
    torch.manual_seed(int(seed))
    model = HCLCRC("A", norm, cfg)
    ssl = LM.load_ssl_encoders(model, LM.ssl_path(seed))
    old_local = model.local
    new_local = CompLocalContent(model.cfg)
    new_local.load_state_dict(old_local.state_dict())
    model.local = new_local

    # --- new modules under a dedicated seed --------------------------------
    torch.manual_seed(int(seed) + C.HC_SEED_OFFSET)
    width = int(model.cfg["width"])
    model.comp_proj = nn.Linear(width, width)
    nn.init.zeros_(model.comp_proj.weight)
    nn.init.zeros_(model.comp_proj.bias)
    model.kind = kind
    model.path_encoder = HPathEncoder() if kind == "H" else FPathEncoder()
    model.to(device)

    base_keys = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                 if not k.startswith("comp_proj.") and not k.startswith("path_encoder.")}
    base_hash = C.sha256_state(base_keys)
    info = {
        "kind": kind,
        "arm": kind,
        "seed": int(seed),
        "config": model.cfg,
        "ssl": ssl,
        "parameters": _param_count(model),
        "parameters_breakdown": _breakdown(model),
        "base_state_sha256": base_hash,
        "initial_state_sha256": C.state_hash(model),
        "comp_proj_zero_initialised": True,
        "path_encoder_seed": int(seed) + C.HC_SEED_OFFSET,
        "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
        "parameter_tensors": sum(1 for p in model.parameters()),
        "has_junction_module": any("junction" in n for n, _ in model.named_modules()),
    }
    _check_params(kind, info["parameters"])
    return model, info


def load_model_from_state(kind: str, norm, state: Mapping[str, torch.Tensor], device,
                          cfg: Mapping[str, Any] | None = None):
    model, _info = build_model(kind, norm, seed=0, device="cpu", cfg=cfg)
    model.load_state_dict({k: v.to("cpu") for k, v in state.items()})
    model.to(device)
    return model


__all__ = ["KINDS", "N_CHORD", "Z_DIM", "F2_IN", "F3_IN", "FLAT_IN",
           "CompLocalContent", "HPathEncoder", "FPathEncoder", "HCLCRC",
           "build_model", "load_model_from_state"]
