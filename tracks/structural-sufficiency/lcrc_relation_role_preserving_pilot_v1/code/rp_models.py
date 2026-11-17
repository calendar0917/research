"""RR_REAL / RR_CONTROL: the frozen C1 function with a two-group S2 role scale.

The C1 function is reused **unchanged**:

    a_v   = AtomEncoder(atom_feat_v)
    b_e   = BondEncoder([a_u + a_v, |a_u - a_v|, a_u * a_v, bond_feat_e])
    x_v   = [ a_v,
              sum_{e incident v} b_e,
              sum_{S1 occurrence of v} nbr_emb(code),
              sum_{S2 occurrence of v} [type, end, joint, closure],
              sum_{S2 occurrence of v} a_end * path_gate(code),
              atom_scalars_v ]                                   (433 columns)
    h_v   = LayerNorm(LocalMLP(x_v))        one shared dropout(0.10)
    y     = sum_v f(h_v) + rho([g, mean_v h, mean_v h^2, log1p n])

The only change: the 200 S2 columns of ``x_v`` are split into the two role
groups produced by the frozen grouping rule,

    Q_internal_v = sum of the S2 records of v whose occurrence is in group 0
    Q_cross_v    = sum of the S2 records of v whose occurrence is in group 1
    Q_v          = gamma_internal * Q_internal_v + gamma_cross * Q_cross_v

with ``gamma_internal, gamma_cross in R^200`` trainable per-channel vectors
initialised to **exactly one**.  Because ``Q_internal + Q_cross == Q`` (the
frozen C1 S2 aggregate, verified by a gate), ``gamma == 1`` makes the model
bit-for-bit the C1 function at the same canonical initial state.

Grouping rules (frozen, label free):

    RR_REAL    : owner(e1) == owner(e2) -> group 0, else group 1
                 (the frozen bridges/cycles/chains partition of the previous round)
    RR_CONTROL : blake2b(salt | path_cat | path_end63 | path_closure) low bit
                 (the frozen S2 *type* hash; no atom index, no row id, no label)

Both models have exactly the same number of trainable parameters
(``C1 + 2 * 200``) and the same initial state (gammas are all ones, everything
else is the canonical C1 state), so a REAL/CONTROL difference cannot be a
capacity or initialisation difference.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

import rp_common as C

M = C.modules()
DM = M["dissect_model"]
CM = M["cr_models"]

WIDTH = C.WIDTH
TRUNK_PREFIXES = ("atom_encoder.", "bond_encoder.", "local.")
TRUNK_BUFFERS = ("global_mean", "global_scale", "scalar_mean", "scalar_scale")
ARM_PREFIX = "trunk."
S2_TYPED_DIM = C.S2_TYPED_DIM
S2_CONTENT_DIM = C.S2_CONTENT_DIM
S2_ROLE_DIM = C.S2_ROLE_DIM


# ---------------------------------------------------------------------------
# trunk: bitwise the C1 local encoder, with the S2 block split in two
# ---------------------------------------------------------------------------


class SplitTrunk(CM.LocalTrunk):
    """``cr_models.LocalTrunk`` (same modules, same state keys) + split S2 path."""

    def s2_evidence(self, batch: Mapping[str, torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        """The 200-d S2 record of every occurrence: [type, end, joint, closure, content]."""
        local = self.local
        rank = int(local.rank)
        n_occ = int(batch["path_cat"].numel())
        if n_occ:
            typed = torch.cat(
                [
                    local.path_emb(batch["path_cat"]),
                    local.end_emb(batch["path_end63"]),
                    local.joint_proj(local.path_u(batch["path_cat"]) * local.path_v(batch["path_end63"])),
                    local.close_emb(batch["path_closure"]),
                ],
                dim=1,
            )
            content = a[batch["path_end"]] * local.path_gate(batch["path_cat"])
        else:
            typed = a.new_zeros((0, 3 * rank + 8))
            content = a.new_zeros((0, a.shape[1]))
        evidence = torch.cat([typed, content], dim=1)
        if evidence.shape[1] != S2_ROLE_DIM:
            raise RuntimeError(f"S2 evidence width {evidence.shape[1]} != frozen {S2_ROLE_DIM}")
        return evidence

    def local_inputs(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
        evidence: torch.Tensor,
        role: torch.Tensor,
        gamma_internal: torch.Tensor,
        gamma_cross: torch.Tensor,
        keep_occ: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frozen 433-column C1 input with the S2 columns replaced by ``Q_v``."""
        local = self.local
        n_atoms = int(a.shape[0])
        rank = int(local.rank)
        width = int(a.shape[1])

        idx = batch["bond_index"]
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        else:
            b_sum = a.new_zeros((n_atoms, width))

        if batch["nbr_code"].numel():
            nbr = a.new_zeros((n_atoms, rank)).index_add_(
                0, batch["nbr_atom"], local.nbr_emb(batch["nbr_code"])
            )
        else:
            nbr = a.new_zeros((n_atoms, rank))

        if keep_occ is not None:
            evidence = evidence * keep_occ.to(evidence.dtype).unsqueeze(1)
        root = batch["path_atom"]
        role = role.to(torch.int64)
        if role.numel() != evidence.shape[0]:
            raise RuntimeError(
                f"role vector has {role.numel()} entries but the batch has {evidence.shape[0]} S2 occurrences"
            )
        mask0 = role == 0
        q0 = a.new_zeros((n_atoms, S2_ROLE_DIM))
        q1 = a.new_zeros((n_atoms, S2_ROLE_DIM))
        if bool(mask0.any()):
            q0 = q0.index_add_(0, root[mask0], evidence[mask0])
        mask1 = ~mask0
        if bool(mask1.any()):
            q1 = q1.index_add_(0, root[mask1], evidence[mask1])
        s2 = gamma_internal * q0 + gamma_cross * q1

        scal_cols = list(local.scalar_cols)
        x = torch.cat([a, b_sum, nbr, s2, scal[:, scal_cols]], dim=1)
        if x.shape[1] != C.LOCAL_INPUT_DIM:
            raise RuntimeError(f"local input width {x.shape[1]} != frozen C1 width {C.LOCAL_INPUT_DIM}")
        return x, q0, q1


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class RolePreservingC1(nn.Module):
    def __init__(self, spec, norm: Mapping[str, Any], variant: str) -> None:
        super().__init__()
        if variant not in C.MODELS:
            raise KeyError(variant)
        self.variant = variant
        self.spec = spec
        self.trunk = SplitTrunk(spec, norm)
        self.f = DM.MLP([WIDTH, WIDTH, 1])
        self.rho_dim = int(spec.global_dim) + 2 * WIDTH + 1
        self.rho = DM.MLP([self.rho_dim, C.RHO_HIDDEN, 1])
        self.gamma_internal = nn.Parameter(torch.ones(S2_ROLE_DIM, dtype=torch.float32))
        self.gamma_cross = nn.Parameter(torch.ones(S2_ROLE_DIM, dtype=torch.float32))

    # -- construction ------------------------------------------------------
    def load_canonical(self, state: Mapping[str, Any]) -> None:
        self.trunk.load_canonical(state)
        for prefix, module in (("f.", self.f), ("rho.", self.rho)):
            subset = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            module.load_state_dict({k: v.clone() for k, v in subset.items()}, strict=True)

    @property
    def role_key(self) -> str:
        return C.MODEL_ROLE_KEY[self.variant]

    def parameter_report(self) -> dict[str, Any]:
        named = list(self.named_parameters())
        trunk = sum(p.numel() for k, p in named if k.startswith(ARM_PREFIX))
        role = sum(p.numel() for k, p in named if k.startswith("gamma_"))
        readout = sum(p.numel() for k, p in named if not k.startswith(ARM_PREFIX) and not k.startswith("gamma_"))
        breakdown: dict[str, int] = {}
        for key, value in named:
            top = ".".join(key.split(".")[:2]) if key.startswith(ARM_PREFIX) else key.split(".")[0]
            breakdown[top] = breakdown.get(top, 0) + int(value.numel())
        return {
            "variant": self.variant,
            "total": int(sum(p.numel() for _, p in named)),
            "trunk": int(trunk),
            "readout": int(readout),
            "role_gamma": int(role),
            "breakdown": dict(sorted(breakdown.items())),
            "role_parameter_names": sorted(k for k, _ in named if k.startswith("gamma_")),
        }

    def role_state(self) -> dict[str, torch.Tensor]:
        return {
            "gamma_internal": self.gamma_internal.detach().clone(),
            "gamma_cross": self.gamma_cross.detach().clone(),
        }

    def canonical_parameters(self) -> dict[str, torch.Tensor]:
        """Every *trainable* parameter except the two role vectors (the C1 part).

        Buffers are excluded on purpose: the frozen input-normalisation constants
        (42+42+9+9 elements) live in the C1 trunk as buffers and are not part of
        the 228674 trainable C1 parameters.
        """
        return {
            k: v.detach().clone()
            for k, v in self.named_parameters()
            if not k.startswith("gamma_")
        }

    # -- forward -----------------------------------------------------------
    def prepare(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        a, b, scal = self.trunk.encode(batch)
        evidence = self.trunk.s2_evidence(batch, a)
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        count = a.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=a.dtype)
        )
        return {
            "a": a,
            "b": b,
            "scal": scal,
            "evidence": evidence,
            "g": self.trunk.z_global(batch),
            "count": count,
            "atom_batch": atom_batch,
            "n_graphs": n_graphs,
        }

    def predict(
        self,
        prep: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
        *,
        role_key: str | None = None,
        keep_occ: torch.Tensor | None = None,
        gammas: Sequence[torch.Tensor] | None = None,
        return_aux: bool = False,
    ):
        a, b, scal = prep["a"], prep["b"], prep["scal"]
        evidence = prep["evidence"]
        role = batch[role_key or self.role_key]
        gi, gc = (gammas if gammas is not None else (self.gamma_internal, self.gamma_cross))
        x, q0, q1 = self.trunk.local_inputs(
            batch, a, b, scal, evidence, role, gi, gc, keep_occ=keep_occ
        )
        h = self.trunk.local.norm(self.trunk.local.mlp(x))
        h = self.trunk.drop(h)          # the one shared dropout of the C1 local path

        atom_batch = prep["atom_batch"]
        n_graphs = int(prep["n_graphs"])
        count = prep["count"]
        contrib_atom = self.f(h).view(-1)
        local_sum = contrib_atom.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        denom = count.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((n_graphs, WIDTH)).index_add_(0, atom_batch, h) / denom
        mean_h2 = h.new_zeros((n_graphs, WIDTH)).index_add_(0, atom_batch, h * h) / denom
        rho_in = torch.cat([prep["g"], mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        if rho_in.shape[1] != self.rho_dim:
            raise RuntimeError(f"rho input width {rho_in.shape[1]} != frozen {self.rho_dim}")
        rho = self.rho(rho_in).view(-1)
        pred = local_sum + rho
        if not return_aux:
            return pred
        return pred, {
            "pred_std": pred,
            "rho": rho,
            "rho_input": rho_in,
            "local_sum": local_sum,
            "contrib_atom": contrib_atom,
            "h": h,
            "local_input": x,
            "q_internal": q0,
            "q_cross": q1,
            "q_total": q0 + q1,
            "count": count,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "gamma_internal": gi,
            "gamma_cross": gc,
            "term_names": ["local_sum", "rho"],
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        role_key: str | None = None,
        keep_occ: torch.Tensor | None = None,
        gammas: Sequence[torch.Tensor] | None = None,
        return_aux: bool = False,
    ):
        prep = self.prepare(batch)
        return self.predict(
            prep,
            batch,
            role_key=role_key,
            keep_occ=keep_occ,
            gammas=gammas,
            return_aux=return_aux,
        )


# ---------------------------------------------------------------------------
# construction / parameter accounting
# ---------------------------------------------------------------------------


def canonical_state(spec, norm) -> dict[str, torch.Tensor]:
    """The single canonical C1 initial state of the read-only source round."""
    return CM.canonical_c1_state(spec, norm)


def build_model(
    variant: str,
    spec,
    norm: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> RolePreservingC1:
    C.set_global_seeds(C.TRAIN_SEED)
    model = RolePreservingC1(spec, norm, variant)
    model.load_canonical(canonical)
    return model.to(device)


def expected_parameter_counts(n_global: int) -> dict[str, int]:
    """Frozen analytic counts (checked against the built modules)."""
    trunk = int(CM.TRUNK_PARAMS)
    f = C.WIDTH * C.WIDTH + C.WIDTH + C.WIDTH + 1
    rho = (int(n_global) + 2 * C.WIDTH + 1) * C.RHO_HIDDEN + C.RHO_HIDDEN + C.RHO_HIDDEN + 1
    role = 2 * C.S2_ROLE_DIM
    total = trunk + f + rho + role
    return {
        "trunk": trunk,
        "f": int(f),
        "rho": int(rho),
        "role_gamma": int(role),
        "c1_total": int(CM.FULL_C1_PARAMS),
        "rr_total": int(total),
        "delta_over_c1": int(total - int(CM.FULL_C1_PARAMS)),
        "ratio_over_c1": float(total) / float(CM.FULL_C1_PARAMS),
    }


def block_keep(owner1: torch.Tensor, owner2: torch.Tensor, block_index: int) -> torch.Tensor:
    """Native block mask: an S2 occurrence survives iff neither owner is the block.

    This is exactly the routing semantics of the previous round (an internal S2
    of block ``i`` and every interface incident to ``i`` disappear when ``i`` is
    switched off; occurrences owned by two other blocks survive).
    """
    return (owner1 != int(block_index)) & (owner2 != int(block_index))


__all__ = [
    "WIDTH",
    "S2_ROLE_DIM",
    "SplitTrunk",
    "RolePreservingC1",
    "canonical_state",
    "build_model",
    "expected_parameter_counts",
    "block_keep",
]
