"""The gated model: C1 with the joint (J) block contracted by a frozen prior.

The only change relative to the frozen C1 graph is

    J_new(v) = J_marg(v) + a_v * (J_full(v) - J_marg(v))

with ``J_full`` / ``J_marg`` exactly the upstream operators (``agg="full"`` and
``agg="marginal_bt2"``, the latter being ``marginal_joint_agg`` on the
``(root, bt2)`` grouping).  ``a_v`` is a **frozen, weight-independent** number
per root atom supplied through ``batch["root_alpha"]``; it has no trainable
parameter.  ``a_v == 1`` gives FULL, ``a_v == 0`` gives the marginal (B5)
operator.

Two pitfalls this module handles explicitly:

* the historical ``ShrinkLocalContent`` guard ``if agg == "full": return ...``
  must **not** disable the gate; here the gate is only reached when
  ``self.s2_agg == "marginal_bt2"`` (asserted at construction and at use);
* ``content`` (``C_v``) must stay FULL.  It is forced to ``"full"`` here even
  though the joint block is routed through the marginal grouping.

``type`` / ``end`` / ``closure`` blocks, the S1 block, the scalar/global inputs,
``f``, ``rho`` and the dropout rule are untouched.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

import sc_common as C

import dissect_model as DM
import lcpr_model as PM


class SupportLocalContent(DM.LocalContent):
    """``LocalContent`` whose joint block is handled by a frozen per-root prior."""

    def __init__(self, spec, variant: str, cfg: Mapping[str, Any], *, gated: bool):
        super().__init__(spec, variant, cfg)
        self.gated = bool(gated)
        if self.gated:
            if not self.s2_joint:
                raise ValueError("the gated model requires the S2 joint block")
            # route the *joint* block through the marginal grouping so that
            # ``forward`` calls ``_joint_agg`` with agg != "full"
            self.s2_agg = "marginal_bt2"
            if self.s2_agg == "full":       # pragma: no cover - defensive
                raise AssertionError("the gate would be short-circuited by agg == 'full'")

    # -- content stays FULL --------------------------------------------------
    def _content_agg(self, batch, a, agg):
        if not self.gated:
            return super()._content_agg(batch, a, agg)
        return super()._content_agg(batch, a, "full")

    # -- the gated joint block ----------------------------------------------
    def _joint_agg(self, batch, a, agg):
        if not self.gated:
            return super()._joint_agg(batch, a, agg)      # exact frozen operator
        if agg == "full":                                  # pragma: no cover
            raise AssertionError("gated joint block reached with agg == 'full'")
        j_full = super()._joint_agg(batch, a, "full")
        j_marg = super()._joint_agg(batch, a, "marginal_bt2")
        alpha = batch.get("root_alpha")
        if alpha is None:
            raise ValueError("gated model needs batch['root_alpha'] (per atom row)")
        coeff = alpha.reshape(-1, 1).to(dtype=a.dtype, device=a.device)
        if coeff.shape[0] != j_full.shape[0]:
            raise ValueError(f"root_alpha has {coeff.shape[0]} rows, atoms {j_full.shape[0]}")
        return j_marg + coeff * (j_full - j_marg)

    # -- a=1 / a=0 endpoint fixtures (tests + dose diagnostics) --------------
    def joint_parts(self, batch, a):
        """``(J_full, J_marg)`` for one batch (both in the model's dtype)."""
        j_full = super()._joint_agg(batch, a, "full")
        j_marg = super()._joint_agg(batch, a, "marginal_bt2")
        return j_full, j_marg


class SupportLCRC(PM.RouteLCRC):
    """C1 with the joint block optionally gated by the frozen support prior.

    ``gated=False`` reproduces the frozen C1 computation (the original operator,
    not a limit of it).  ``gated=True`` consumes ``batch["root_alpha"]``.
    Parameter count is identical in both cases (228,674): ``a_v`` is not learned.
    """

    def __init__(self, spec, norm: Mapping[str, Any], *, gated: bool):
        super().__init__(spec, norm, "c1")
        with torch.random.fork_rng(devices=[]):
            self.local = SupportLocalContent(spec, self.variant, self.cfg, gated=gated)
        self.gated = bool(gated)
        if self.parameter_count() != C.C1_PARAMS:      # pragma: no cover - defensive
            raise RuntimeError(f"parameter count {self.parameter_count()} != {C.C1_PARAMS}")


def build_arm_model(spec, norm: Mapping[str, Any], arm: str) -> SupportLCRC:
    if arm not in C.ARMS:
        raise KeyError(arm)
    return SupportLCRC(spec, norm, gated=(arm != C.ARM_BASE))


def load_shared_init(model: SupportLCRC, spec, norm) -> dict[str, Any]:
    """Copy the frozen FULL(seed 0) -> C1 state and verify it bitwise."""
    _full, base_state = PM.c1_reference_state_full(spec, norm)
    report = PM.load_init_state(model, base_state, "c1")
    check = PM.shared_state_equal(model, base_state, "c1")
    if not check["all_equal"]:                          # pragma: no cover - defensive
        raise RuntimeError("shared C1 state is not bitwise equal to the frozen remap")
    return {"copied": report["copied"], "new_extension_tensors": report["new_extension_tensors"],
            "n_parameters": report["n_parameters"], "shared_state_equal": check["n_checked"],
            "base_state": base_state}


__all__ = ["SupportLocalContent", "SupportLCRC", "build_arm_model", "load_shared_init"]
