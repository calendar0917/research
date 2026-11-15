"""The semantic-table LCRC model: a C1 model with its six free tables replaced.

The backbone is unchanged: ``DissectLCRC(spec, norm, "C1")`` with the frozen
atom/bond encoders, S1/S2 sums, ``joint_proj(path_u * path_v)``, closure, gated
end attribute, LocalMLP, ``f + rho`` and the single h-dropout.

Two modifications happen *only* on the six free tables of ``local``:

* every table is wrapped in a :class:`se_tables.CodeTable` (free or generator
  mode with an optional composite-code permutation), and
* when a generator is present it is registered once on the model (so ``.to()``,
  ``state_dict`` and ``parameters()`` see it exactly once) and the CodeTables
  hold it by reference without re-registering it.

No input, vocabulary, path enumeration, root ownership, pooling or supervision
changes.  ``joint_proj`` and ``close_emb`` are left untouched.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import se_common as C
import se_tables as T

import lp_common as LPC
import lp_model as LPM
import dissect_model as DM


#: local.* table name -> bank key in se_common.TABLES
LOCAL_TABLE_BANKS = {
    "nbr_emb": "nbr_emb",
    "path_emb": "path_emb",
    "end_emb": "end_emb",
    "path_u": "path_u",
    "path_v": "path_v",
    "path_gate": "path_gate",
}


class SemanticLCRC(nn.Module):
    """C1 model + optional generator that produces the six embedding tables."""

    def __init__(
        self,
        base: DM.DissectLCRC,
        *,
        mode: str,
        generator: T.SemanticGenerator | None = None,
        buffers: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        permutations: Mapping[str, np.ndarray] | None = None,
        init_tables: Mapping[str, torch.Tensor] | None = None,
        permute: bool = False,
    ):
        super().__init__()
        if mode not in ("free", "generator"):
            raise ValueError(mode)
        self.mode = mode
        self.permute = bool(permute)
        self.base = base
        self.generator = generator  # may be None; registered once here
        self._sem_buffers = buffers
        self._permutations = permutations
        self._install(init_tables or {})

    # -- installation -----------------------------------------------------
    def _install(self, init_tables: Mapping[str, torch.Tensor]) -> None:
        local = self.base.local
        for attr, bank in LOCAL_TABLE_BANKS.items():
            old = getattr(local, attr)
            perm = None
            if self.permute and self._permutations is not None:
                perm = self._permutations[T.perm_for_bank(bank)]
            table = T.CodeTable(
                bank,
                self.mode,
                generator=self.generator,
                buffers=self._sem_buffers,
                perm=perm,
                init_table=init_tables.get(bank),
            )
            # preserve the original embedding dtype/device placement later via .to()
            if table.mode == "free" and init_tables.get(bank) is None:
                with torch.no_grad():
                    table.table.weight.copy_(old.weight.detach())
            setattr(local, attr, table)
        # drop the free nn.Embedding parameter registrations that no longer exist
        self._generator_registered_once = self.generator is not None

    # -- convenience ------------------------------------------------------
    def generator_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.generator.parameters()) if self.generator is not None else []

    def forward(self, batch, **kwargs):
        return self.base(batch, **kwargs)

    def materialise_tables(self, *, permute: bool | None = None, device=None) -> dict[str, torch.Tensor]:
        if self.mode != "generator":
            raise ValueError("not a generator model")
        use_perm = self.permute if permute is None else permute
        return T.materialise_all(
            self.generator, self._sem_buffers, self._permutations, permute=use_perm, device=device
        )


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _c1_base(spec, norm) -> DM.DissectLCRC:
    """Frozen C1 base with the shared untrained initial state (seed 0)."""
    full_state = DM.build_full_state(spec, norm, int(LPC.TRAIN_SEED))
    c1_state = DM.remap_state("C1", full_state)
    model = DM.DissectLCRC(spec, norm, "C1")
    model.load_state_dict(c1_state, strict=True)
    return model


def _e0_tables(spec, norm) -> dict[str, np.ndarray]:
    """The original C1 initial table values (theta_0 of the free lookups)."""
    full_state = DM.build_full_state(spec, norm, int(LPC.TRAIN_SEED))
    c1_state = DM.remap_state("C1", full_state)
    return {
        bank: c1_state[f"local.{bank}.weight"].detach().cpu().numpy().astype(np.float64)
        for bank in C.TABLES
    }


def build_semantic_model(
    spec,
    norm,
    family: str,
    *,
    permutations: Mapping[str, np.ndarray] | None = None,
) -> SemanticLCRC:
    """Build one of LS / FS / LP / FP.

    * ``LS``: free tables, initial values = FS's theta_0 materialisation.
    * ``FS``: generator, real codec.
    * ``LP``: free tables, initial values = FP's theta_0 materialisation.
    * ``FP``: generator, composite-code permutation applied before decoding.

    Cross-arm initial-prediction equality holds inside each pair (LS-FS and
    LP-FP) because the free table is initialised to that generator's own theta_0
    materialised values and every non-table weight is the shared C1 state.
    """
    if family not in C.SEM_FAMILIES:
        raise ValueError(family)
    base = _c1_base(spec, norm)
    e0 = _e0_tables(spec, norm)
    permute = family in ("LP", "FP")
    if permutations is None:
        permutations = T.build_permutations()

    gen = T.SemanticGenerator()
    generator = torch.Generator(device="cpu").manual_seed(int(C.GENERATOR_SEED))
    gen.init_module(generator)
    buffers = T.build_buffers(gen, e0)

    if family in ("FS", "FP"):
        model = SemanticLCRC(
            base, mode="generator", generator=gen, buffers=buffers,
            permutations=permutations, permute=permute,
        )
        return model

    # free tables initialised to the paired generator's theta_0 materialisation
    init_tables = T.materialise_all(gen, buffers, permutations, permute=permute, device=None)
    model = SemanticLCRC(
        base, mode="free", generator=None, buffers=buffers,
        permutations=permutations, permute=False, init_tables=init_tables,
    )
    return model


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def geometry_report(model: SemanticLCRC) -> dict[str, Any]:
    return {
        "mode": model.mode,
        "permute": model.permute,
        "parameters": parameter_count(model),
        "generator_parameters": int(sum(p.numel() for p in model.generator_parameters())),
        "has_generator": model.generator is not None,
    }


__all__ = ["SemanticLCRC", "build_semantic_model", "parameter_count", "geometry_report"]
