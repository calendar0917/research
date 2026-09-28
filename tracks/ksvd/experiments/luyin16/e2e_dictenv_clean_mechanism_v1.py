"""E2E-DictEnv-Clean-Mechanism-v1 — core mechanism-convergence module.

Round ``e2e_dictenv_clean_mechanism_v1`` (Workstream Z).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_clean_mechanism_v1_preregistration.md``.

This module never modifies the frozen P1 / P2-ABS / clarity-audit
implementations.  It adds, on top of ``AuditModel``:

* the canonical ``C6`` mask of the clarity audit (as a set, not a duplicated
  tuple) plus the nested relation candidates;
* ``CleanMechModel`` — H1 with selectable node / edge binding operators:

  - ``paired`` (default, bit-identical to H1):
      ``U[is] = 1/sqrt(D) * sum_v a_v * c_v``
  - ``indep`` (analytic assignment-independence null, the expectation of the
    paired statistic under a uniform random pairing of the same multisets):
      ``U[is] = 1/(n_is sqrt(D)) * (sum_v a_v) * (sum_v c_v)``

  and a selectable coding operator (``sparse`` tied IHT vs ``dense_tied``
  ``phi @ Dbar``, the P1 DenseTied control);
* per-slot paired / independence / residual norm diagnostics with
  per-shell and per-shellpair distributions;
* the pre-registered gates for Q1 / Q2 / Q3 / Q4;
* a thin ``train_spec`` wrapper over ``audit.train_cpu``'s optional
  ``model_factory`` hook (the default H1 path is unchanged).

CPU only.  ``official_test_loaded`` is ``False`` everywhere.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit

PROTOCOL_VERSION = "e2e_dictenv_clean_mechanism_v1"

H1_CONFIG = audit.H1_CONFIG
H1_LAMBDA = audit.H1_LAMBDA

NODE_BINDINGS = ("paired", "indep")
EDGE_BINDINGS = ("paired", "indep")
CODINGS = ("sparse", "dense_tied")

D_A = int(p2.D_A)
D_E = 48

# ---------------------------------------------------------------------------
# masks (canonical definitions; see preregistration section 1.2)
# ---------------------------------------------------------------------------

#: the clarity-audit C6 candidate as a set of channels.
C6_MASK = audit.AuditMask(
    global_zero_groups=("atom_histogram", "bond_histogram"),
    unary_zero_blocks=("count",),
    pair_zero_blocks=("count",),
    relation_zero_groups=("path_count",),
)

#: the clarity-audit C1 candidate (graph chemistry marginal only).
C1_MASK = audit.AuditMask(global_zero_groups=("atom_histogram", "bond_histogram"))

#: nested relation candidates on top of C6 (path count is already removed).
RELATION_CANDIDATES = {
    "REL-FULL-CLEAN": C6_MASK,
    "REL-DIST-BOUNDARY": dataclasses.replace(
        C6_MASK, relation_zero_groups=("overlap", "path_count")
    ),
    "REL-DIST": dataclasses.replace(
        C6_MASK, relation_zero_groups=("boundary", "overlap", "path_count")
    ),
}


def c6_equivalence_check() -> bool:
    """The canonical C6 mask and the recorded clarity-audit C6 mask agree.

    The recorded mask lists ``bond_histogram`` twice (an idempotent duplicate
    created by ``GLOBAL_CHEMISTRY_GROUPS + ("bond_histogram",)``); the set of
    zeroed channels is identical.
    """
    recorded = audit.candidate_mask("C6")
    return (
        set(recorded.global_zero_groups) == set(C6_MASK.global_zero_groups)
        and set(recorded.unary_zero_blocks) == set(C6_MASK.unary_zero_blocks)
        and set(recorded.pair_zero_blocks) == set(C6_MASK.pair_zero_blocks)
        and set(recorded.relation_zero_groups) == set(C6_MASK.relation_zero_groups)
    )


# ---------------------------------------------------------------------------
# arm registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanMechSpec:
    """Declarative arm spec: mask + binding operators + coding operator."""

    tag: str
    mask_kind: str = "BASE"  # BASE | C1 | C6 | REL-FULL-CLEAN | REL-DIST-BOUNDARY | REL-DIST
    node_binding: str = "paired"
    edge_binding: str = "paired"
    coding: str = "sparse"

    def __post_init__(self) -> None:
        if self.node_binding not in NODE_BINDINGS:
            raise ValueError(f"unknown node_binding {self.node_binding!r}")
        if self.edge_binding not in EDGE_BINDINGS:
            raise ValueError(f"unknown edge_binding {self.edge_binding!r}")
        if self.coding not in CODINGS:
            raise ValueError(f"unknown coding {self.coding!r}")

    def mask(self) -> audit.AuditMask | None:
        return arm_mask(self)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def arm_mask(spec: CleanMechSpec) -> audit.AuditMask | None:
    """Resolve the forward mask of an arm (``None`` = untouched H1 forward)."""
    kind = spec.mask_kind
    if kind == "BASE":
        return None
    if kind == "C1":
        return C1_MASK
    if kind == "C6":
        return C6_MASK
    if kind in RELATION_CANDIDATES:
        return RELATION_CANDIDATES[kind]
    raise KeyError(f"unknown mask kind {kind!r}")


#: arm registry referenced by the runner and the report.
CLEAN_MECH_ARMS: dict[str, CleanMechSpec] = {
    "BASE": CleanMechSpec("BASE", "BASE"),
    "C1": CleanMechSpec("C1", "C1"),
    "C6": CleanMechSpec("C6", "C6"),
    "C6-NODE-INDEP": CleanMechSpec("C6-NODE-INDEP", "C6", node_binding="indep"),
    "C6-EDGE-INDEP": CleanMechSpec("C6-EDGE-INDEP", "C6", edge_binding="indep"),
    "C6-BOTH-INDEP": CleanMechSpec(
        "C6-BOTH-INDEP", "C6", node_binding="indep", edge_binding="indep"
    ),
    "C1-NODE-INDEP": CleanMechSpec("C1-NODE-INDEP", "C1", node_binding="indep"),
    "C1-EDGE-INDEP": CleanMechSpec("C1-EDGE-INDEP", "C1", edge_binding="indep"),
    "BASE-NODE-INDEP": CleanMechSpec("BASE-NODE-INDEP", "BASE", node_binding="indep"),
    "BASE-EDGE-INDEP": CleanMechSpec("BASE-EDGE-INDEP", "BASE", edge_binding="indep"),
    "REL-DIST-BOUNDARY": CleanMechSpec("REL-DIST-BOUNDARY", "REL-DIST-BOUNDARY"),
    "REL-DIST": CleanMechSpec("REL-DIST", "REL-DIST"),
    "C6-DENSE-TIED": CleanMechSpec("C6-DENSE-TIED", "C6", coding="dense_tied"),
}


def make_spec(
    tag: str,
    *,
    mask_kind: str,
    node_binding: str = "paired",
    edge_binding: str = "paired",
    coding: str = "sparse",
) -> CleanMechSpec:
    """Build an arm spec (used by the runner for gate-composed final arms)."""
    return CleanMechSpec(tag, mask_kind, node_binding, edge_binding, coding)


def merge_masks(
    base: audit.AuditMask | None, probe: audit.AuditMask | None
) -> audit.AuditMask | None:
    """Union of two masks (arm training mask + probe mask).

    Used by the Stage-B revalidation so that a probe is applied on top of the
    arm's own training mask rather than replacing it.
    """
    if base is None:
        return probe
    if probe is None:
        return base
    return audit.AuditMask(
        anchor_zero_groups=tuple(sorted(set(base.anchor_zero_groups) | set(probe.anchor_zero_groups))),
        global_zero_groups=tuple(sorted(set(base.global_zero_groups) | set(probe.global_zero_groups))),
        topology_zero=bool(base.topology_zero or probe.topology_zero),
        node_binding_zero=bool(base.node_binding_zero or probe.node_binding_zero),
        edge_binding_zero=bool(base.edge_binding_zero or probe.edge_binding_zero),
        use_node_shuffle=bool(base.use_node_shuffle or probe.use_node_shuffle),
        use_edge_shuffle=bool(base.use_edge_shuffle or probe.use_edge_shuffle),
        coord_zero=bool(base.coord_zero or probe.coord_zero),
        relation_zero_groups=tuple(
            sorted(set(base.relation_zero_groups) | set(probe.relation_zero_groups))
        ),
        unary_zero_blocks=tuple(sorted(set(base.unary_zero_blocks) | set(probe.unary_zero_blocks))),
        pair_zero_blocks=tuple(sorted(set(base.pair_zero_blocks) | set(probe.pair_zero_blocks))),
        graph_hidden_zero=bool(base.graph_hidden_zero or probe.graph_hidden_zero),
        gate_off=bool(base.gate_off or probe.gate_off),
        pair_projection_zero=bool(base.pair_projection_zero or probe.pair_projection_zero),
    )


def final_clean_mask(
    clean_base: str,
    *,
    relation_kind: str | None = None,
    node_binding: str = "paired",
    edge_binding: str = "paired",
) -> tuple[audit.AuditMask | None, CleanMechSpec]:
    """Compose FINAL-CLEAN from the adopted Stage-C/D simplifications.

    ``clean_base`` is ``C6`` / ``C1`` / ``BASE``; ``relation_kind`` (when
    adopted) is one of ``RELATION_CANDIDATES``; the binding modes are adopted
    independently.  The returned spec is the corresponding sparse arm.
    """
    base = {"BASE": None, "C1": C1_MASK, "C6": C6_MASK}[clean_base]
    if relation_kind is not None:
        if relation_kind not in RELATION_CANDIDATES:
            raise KeyError(f"unknown relation kind {relation_kind!r}")
        base = RELATION_CANDIDATES[relation_kind]
    kind = clean_base if relation_kind is None else relation_kind
    return base, CleanMechSpec(
        f"FINAL-CLEAN-{kind}",
        kind,
        node_binding=node_binding,
        edge_binding=edge_binding,
        coding="sparse",
    )


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class CleanMechModel(audit.AuditModel):
    """H1 / audit model with selectable binding and coding operators.

    With the default arguments (``paired`` / ``paired`` / ``sparse``) the model
    is identical to ``AuditModel`` including its initialisation RNG stream;
    this is asserted by a focused test.
    """

    def __init__(
        self,
        config: p2.P2Config,
        dictionary: np.ndarray,
        *,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
    ) -> None:
        super().__init__(config, dictionary)
        if node_binding not in NODE_BINDINGS:
            raise ValueError(f"unknown node_binding {node_binding!r}")
        if edge_binding not in EDGE_BINDINGS:
            raise ValueError(f"unknown edge_binding {edge_binding!r}")
        if coding not in CODINGS:
            raise ValueError(f"unknown coding {coding!r}")
        self.node_binding = str(node_binding)
        self.edge_binding = str(edge_binding)
        self.coding = str(coding)

    # -- coding ---------------------------------------------------------------

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        if self.coding == "sparse":
            return super().code(phi)
        Dbar = v0.normalized_dictionary(self.D)
        return phi @ Dbar

    # -- slot operators (shared by forward and diagnostics) -------------------

    def _node_factor_tensors(
        self, coord: torch.Tensor, data: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.one_hot(data.dict_atom, num_classes=p2.ATOM_CATEGORIES).to(coord.dtype)
        occ = data.env_occ_node.to(coord.device)
        a = coord[occ] @ self.W_A_S
        c = q[occ] @ self.W_A_C
        index = data.env_occ_root.to(coord.device) * p2.N_SHELLS + data.env_occ_shell.to(coord.device)
        return a, c, index

    def _edge_factor_tensors(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bond_u is None:
            bond_u = data.env_bond_u
        if bond_v is None:
            bond_v = data.env_bond_v
        bond_u = bond_u.to(coord.device)
        bond_v = bond_v.to(coord.device)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        s = g @ self.W_E_S
        b = F.one_hot(data.env_bond_type, num_classes=p2.BOND_CATEGORIES).to(coord.dtype)
        c = b @ self.W_E_C
        index = data.env_bond_root.to(coord.device) * p2.SHELLPAIR_CLASSES + data.env_bond_shellpair.to(
            coord.device
        )
        return s, c, index

    def node_slots(self, coord: torch.Tensor, data: Any, binding: str | None = None) -> torch.Tensor:
        """Per-(root, shell) node slots ``U`` as ``[n, N_SHELLS, D_A]``."""
        binding = self.node_binding if binding is None else str(binding)
        n = int(coord.shape[0])
        a, c, index = self._node_factor_tensors(coord, data)
        d = float(D_A)
        if binding == "paired":
            u = (a * c) / math.sqrt(d)
            flat = torch.zeros((n * p2.N_SHELLS, int(a.shape[1])), device=u.device, dtype=u.dtype)
            flat.index_add_(0, index, u)
            return flat.view(n, p2.N_SHELLS, int(a.shape[1]))
        sum_a = torch.zeros((n * p2.N_SHELLS, int(a.shape[1])), device=a.device, dtype=a.dtype)
        sum_c = torch.zeros_like(sum_a)
        count = torch.zeros((n * p2.N_SHELLS, 1), device=a.device, dtype=a.dtype)
        sum_a.index_add_(0, index, a)
        sum_c.index_add_(0, index, c)
        count.index_add_(0, index, torch.ones_like(index, dtype=a.dtype).unsqueeze(1))
        denominator = torch.clamp(count, min=1.0) * math.sqrt(d)
        out = (sum_a * sum_c) / denominator
        return out.masked_fill(count <= 0, 0.0).view(n, p2.N_SHELLS, int(a.shape[1]))

    def edge_slots(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        binding: str | None = None,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-(root, shellpair) edge role slots as ``[n, SHELLPAIR_CLASSES, D_E]``."""
        binding = self.edge_binding if binding is None else str(binding)
        n = int(coord.shape[0])
        s, c, index = self._edge_factor_tensors(coord, data, bond_u=bond_u, bond_v=bond_v)
        d = float(s.shape[1])
        if binding == "paired":
            ue = (s * c) / math.sqrt(d)
            flat = torch.zeros(
                (n * p2.SHELLPAIR_CLASSES, int(s.shape[1])), device=ue.device, dtype=ue.dtype
            )
            flat.index_add_(0, index, ue)
            return flat.view(n, p2.SHELLPAIR_CLASSES, int(s.shape[1]))
        sum_s = torch.zeros(
            (n * p2.SHELLPAIR_CLASSES, int(s.shape[1])), device=s.device, dtype=s.dtype
        )
        sum_c = torch.zeros_like(sum_s)
        count = torch.zeros((n * p2.SHELLPAIR_CLASSES, 1), device=s.device, dtype=s.dtype)
        sum_s.index_add_(0, index, s)
        sum_c.index_add_(0, index, c)
        count.index_add_(0, index, torch.ones_like(index, dtype=s.dtype).unsqueeze(1))
        denominator = torch.clamp(count, min=1.0) * math.sqrt(d)
        out = (sum_s * sum_c) / denominator
        return out.masked_fill(count <= 0, 0.0).view(n, p2.SHELLPAIR_CLASSES, int(s.shape[1]))

    # -- masked environments (bit-identical to AuditModel when paired/paired) --

    def environments_masked(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: audit.AuditMask,
        fill: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.node_binding == "paired" and self.edge_binding == "paired":
            return super().environments_masked(coord, data, mask, fill)
        if mask.use_node_shuffle or mask.use_edge_shuffle:
            raise RuntimeError(
                "assignment shuffles are not defined on an independence-null model "
                "(the null already removes the correspondence)"
            )
        n = int(coord.shape[0])
        anchor = data.anchor.to(coord.dtype)
        if int(anchor.shape[1]) != audit.ANCHOR_DIM_EXPECTED:
            raise RuntimeError(
                f"anchor width {int(anchor.shape[1])} != {audit.ANCHOR_DIM_EXPECTED}"
            )
        if mask.anchor_zero_groups:
            anchor = audit._replace_grouped_columns(
                anchor, audit.ANCHOR_GROUPS, mask.anchor_zero_groups, fill, "anchor:"
            )
        node_slots = self.node_slots(coord, data)
        if mask.node_binding_zero:
            node_slots = torch.zeros_like(node_slots)
        edge_slots = self.edge_slots(coord, data)
        if mask.edge_binding_zero:
            edge_slots = torch.zeros_like(edge_slots)
        if self.env_mlp is not None:
            z = torch.cat([anchor, node_slots.reshape(n, -1), edge_slots.reshape(n, -1)], dim=1)
            return self.env_mlp(z)
        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        anchor_out = self.anchor_encoder(anchor)
        fused = torch.cat([anchor_out, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
        return self.fusion(fused)


def build_clean_mech_model(
    dictionary: np.ndarray,
    seed: int = 0,
    spec: CleanMechSpec | None = None,
) -> CleanMechModel:
    """Build an arm model with the same RNG stream as ``audit.build_audit_model``."""
    spec = CLEAN_MECH_ARMS["BASE"] if spec is None else spec
    torch.manual_seed(int(seed))
    return CleanMechModel(
        H1_CONFIG,
        dictionary,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
    )


# ---------------------------------------------------------------------------
# diagnostics: assignment-independence norms
# ---------------------------------------------------------------------------


@dataclass
class _NormAccumulator:
    pair: list[torch.Tensor] = dataclasses.field(default_factory=list)
    indep: list[torch.Tensor] = dataclasses.field(default_factory=list)
    residual: list[torch.Tensor] = dataclasses.field(default_factory=list)
    ratio: list[torch.Tensor] = dataclasses.field(default_factory=list)
    count: list[torch.Tensor] = dataclasses.field(default_factory=list)

    def add(
        self, pair: torch.Tensor, indep: torch.Tensor, count: torch.Tensor
    ) -> None:
        residual = (pair - indep).detach()
        pair_norm = pair.detach().norm(dim=1)
        indep_norm = indep.detach().norm(dim=1)
        self.pair.append(pair_norm.cpu())
        self.indep.append(indep_norm.cpu())
        self.residual.append(residual.norm(dim=1).cpu())
        self.ratio.append((residual.norm(dim=1) / torch.clamp(pair_norm, min=1e-12)).cpu())
        self.count.append(count.detach().reshape(-1).cpu())

    def summary(self) -> dict[str, Any]:
        pair = torch.cat(self.pair).double()
        indep = torch.cat(self.indep).double()
        residual = torch.cat(self.residual).double()
        ratio = torch.cat(self.ratio).double()
        count = torch.cat(self.count)
        n_slots = int(count.numel())
        quantiles = [0.5, 0.9, 0.95]

        def _q(values: torch.Tensor) -> dict[str, float]:
            if int(values.numel()) == 0:
                return {f"p{int(q * 100)}": float("nan") for q in quantiles}
            return {f"p{int(q * 100)}": float(torch.quantile(values, q)) for q in quantiles}

        return {
            "n_slots": n_slots,
            "n_zero": int((count == 0).sum()),
            "n_one": int((count == 1).sum()),
            "n_multi": int((count >= 2).sum()),
            "count_mean": float(count.double().mean()) if n_slots else float("nan"),
            "pair_norm_mean": float(pair.mean()) if n_slots else float("nan"),
            "indep_norm_mean": float(indep.mean()) if n_slots else float("nan"),
            "residual_norm_mean": float(residual.mean()) if n_slots else float("nan"),
            "ratio_mean": float(ratio.mean()) if n_slots else float("nan"),
            "ratio_zero_fraction": float((ratio <= 1e-12).double().mean()) if n_slots else float("nan"),
            "ratio_median": float(ratio.median()) if n_slots else float("nan"),
            **_q(ratio),
        }


def independence_norm_stats(model: CleanMechModel, loader: Any, device: torch.device) -> dict[str, Any]:
    """Paired / independence / residual norm distributions per shell and shellpair.

    Computed from the model weights and the data only; independent of the
    model's own binding mode (both operators are always evaluated).
    """
    if device.type != "cpu":
        raise RuntimeError(f"independence diagnostics are CPU-only, got device={device}")
    node_shells = [_NormAccumulator() for _ in range(p2.N_SHELLS)]
    edge_pairs = [_NormAccumulator() for _ in range(p2.SHELLPAIR_CLASSES)]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            coord = model.code(batch.dict_phi)
            n = int(coord.shape[0])
            pair = model.node_slots(coord, batch, binding="paired").reshape(
                n * p2.N_SHELLS, D_A
            )
            indep = model.node_slots(coord, batch, binding="indep").reshape(n * p2.N_SHELLS, D_A)
            index = batch.env_occ_root.to(coord.device) * p2.N_SHELLS + batch.env_occ_shell.to(
                coord.device
            )
            count = torch.zeros((n * p2.N_SHELLS, 1), device=coord.device, dtype=coord.dtype)
            count.index_add_(
                0, index, torch.ones((index.shape[0], 1), device=coord.device, dtype=coord.dtype)
            )
            for shell in range(p2.N_SHELLS):
                mask = torch.arange(n * p2.N_SHELLS) % p2.N_SHELLS == int(shell)
                node_shells[shell].add(pair[mask], indep[mask], count[mask])

            pair_e = model.edge_slots(coord, batch, binding="paired").reshape(
                n * p2.SHELLPAIR_CLASSES, D_E
            )
            indep_e = model.edge_slots(coord, batch, binding="indep").reshape(
                n * p2.SHELLPAIR_CLASSES, D_E
            )
            index_e = batch.env_bond_root.to(coord.device) * p2.SHELLPAIR_CLASSES + (
                batch.env_bond_shellpair.to(coord.device)
            )
            count_e = torch.zeros(
                (n * p2.SHELLPAIR_CLASSES, 1), device=coord.device, dtype=coord.dtype
            )
            count_e.index_add_(
                0,
                index_e,
                torch.ones((index_e.shape[0], 1), device=coord.device, dtype=coord.dtype),
            )
            for shellpair in range(p2.SHELLPAIR_CLASSES):
                mask = torch.arange(n * p2.SHELLPAIR_CLASSES) % p2.SHELLPAIR_CLASSES == int(
                    shellpair
                )
                edge_pairs[shellpair].add(pair_e[mask], indep_e[mask], count_e[mask])
    return {
        "node": {f"shell{index}": accumulator.summary() for index, accumulator in enumerate(node_shells)},
        "edge": {
            f"shellpair{index}": accumulator.summary() for index, accumulator in enumerate(edge_pairs)
        },
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# frozen binding-replacement diagnostics
# ---------------------------------------------------------------------------


def evaluate_binding_variant(
    dictionary: np.ndarray,
    state: Mapping[str, torch.Tensor],
    loader: Any,
    device: torch.device,
    mask: audit.AuditMask | None,
    *,
    node_binding: str,
    edge_binding: str,
    coding: str = "sparse",
    baseline_predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the same trained state with a different binding operator."""
    model = build_clean_mech_model(
        dictionary,
        0,
        make_spec(
            "variant",
            mask_kind="BASE",
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        ),
    )
    model.load_state_dict({key: value.float() for key, value in state.items()})
    result = audit.evaluate_mask(model, loader, device, mask)
    payload = {
        "node_binding": node_binding,
        "edge_binding": edge_binding,
        "coding": coding,
        "valid_mae": float(result["mae"]),
        "n_molecules": int(result["n_molecules"]),
    }
    if baseline_predictions is not None:
        predictions = result["predictions"]
        payload["mean_abs_prediction_delta"] = float(
            np.mean(np.abs(predictions - baseline_predictions))
        )
        payload["max_abs_prediction_delta"] = float(
            np.max(np.abs(predictions - baseline_predictions))
        )
        if float(np.std(predictions)) > 0 and float(np.std(baseline_predictions)) > 0:
            payload["prediction_correlation"] = float(np.corrcoef(predictions, baseline_predictions)[0, 1])
        else:
            payload["prediction_correlation"] = float("nan")
    payload["official_test_loaded"] = False
    return payload


# ---------------------------------------------------------------------------
# gates (preregistration sections 3.2, 5.4, 6.3, 8)
# ---------------------------------------------------------------------------


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else float("nan"),
        "std": float(array.std(ddof=0)) if array.size else float("nan"),
        "median": float(np.median(array)) if array.size else float("nan"),
        "min": float(array.min()) if array.size else float("nan"),
        "max": float(array.max()) if array.size else float("nan"),
        "range": float(array.max() - array.min()) if array.size else float("nan"),
    }


def c6_gate(deltas: Mapping[int, float]) -> dict[str, Any]:
    """Pre-registered Q1 gate: C6 - BASE per-seed deltas (lower is better)."""
    values = [float(value) for value in deltas.values()]
    stats = _distribution(values)
    n = len(values)
    if n == 0:
        raise ValueError("c6_gate needs at least one seed")
    fraction_better = sum(value < 0 for value in values) / n
    fraction_clean = sum(value <= 0.003 for value in values) / n
    fraction_worse = sum(value > 0.003 for value in values) / n
    if stats["range"] > 0.015:
        verdict = "UNSTABLE"
    elif stats["mean"] < 0 and fraction_better >= 2 / 3 and stats["max"] <= 0.010:
        verdict = "STRONG_SUPPORT"
    elif stats["mean"] <= 0.003 and fraction_clean >= 2 / 3 and stats["max"] <= 0.010:
        verdict = "CLEANNESS_SUPPORT"
    elif stats["mean"] > 0.003 and fraction_worse >= 2 / 3:
        verdict = "REJECT"
    else:
        verdict = "MIXED_NOT_ADOPTED"
    adopted = verdict in ("STRONG_SUPPORT", "CLEANNESS_SUPPORT")
    return {
        "verdict": verdict,
        "adopted": bool(adopted),
        "per_seed": {int(seed): float(value) for seed, value in deltas.items()},
        "stats": stats,
        "thresholds": {
            "strong_max_seed_delta": 0.010,
            "clean_mean": 0.003,
            "unstable_range": 0.015,
        },
    }


def independence_gate(role: str, deltas: Mapping[int, float]) -> dict[str, Any]:
    """Pre-registered Q2 extension gate for the node / edge independence nulls.

    ``role`` is ``node`` or ``edge``.  ``deltas`` are
    ``MAE(indep) - MAE(paired (CLEAN_BASE))`` per seed.
    """
    if role not in ("node", "edge"):
        raise ValueError(f"unknown independence role {role!r}")
    values = [float(value) for value in deltas.values()]
    if not values:
        raise ValueError("independence_gate needs at least one seed")
    stats = _distribution(values)
    seed0 = float(deltas[0]) if 0 in deltas else None
    if role == "node":
        if seed0 is not None and seed0 <= 0.005:
            verdict = "EXTEND_SEEDS"
        elif seed0 is not None and seed0 > 0.015:
            verdict = "STOP_MATERIALLY_NEEDED"
        else:
            verdict = "DECIDE_BY_CURVE_AND_MECHANISM"
        gate = {"seed0_extend_max": 0.005, "seed0_stop_min": 0.015}
    else:
        if seed0 is not None and seed0 > 0.015:
            verdict = "STOP_EDGE_ASSIGNMENT_NEEDED"
        elif seed0 is not None and seed0 <= 0.010:
            verdict = "EXTEND_SEEDS"
        else:
            verdict = "INCONCLUSIVE_KEEP_PAIRED"
        gate = {"seed0_stop_min": 0.015, "seed0_extend_max": 0.010}
    return {
        "role": str(role),
        "verdict": verdict,
        "per_seed": {int(seed): float(value) for seed, value in deltas.items()},
        "stats": stats,
        "thresholds": gate,
    }


def relation_gate(delta: float) -> dict[str, Any]:
    """Pre-registered Q3 gate for a 320-epoch relation candidate."""
    value = float(delta)
    if value <= 0.005:
        verdict = "EXTEND_SEEDS"
    elif value > 0.015:
        verdict = "STOP_TOO_BIG"
    else:
        verdict = "INCONCLUSIVE_KEEP_FULL"
    return {
        "delta": value,
        "verdict": verdict,
        "thresholds": {"extend_max": 0.005, "stop_min": 0.015},
    }


def specificity_gate(delta_dense_minus_sparse: float) -> dict[str, Any]:
    """Pre-registered Q4 gate: positive delta means the sparse arm is better."""
    value = float(delta_dense_minus_sparse)
    verdict = "SPARSE_SPECIFIC_CANDIDATE" if value >= 0.003 else "SPECIFICITY_NOT_ESTABLISHED"
    return {
        "delta_dense_minus_sparse": value,
        "verdict": verdict,
        "threshold": 0.003,
    }


# ---------------------------------------------------------------------------
# dictionary diagnostics and the training wrapper
# ---------------------------------------------------------------------------


def dictionary_diagnostics(
    model: CleanMechModel,
    loader: Any,
    device: torch.device,
    dictionary_init: np.ndarray,
    *,
    gradient_batch: Any | None = None,
) -> dict[str, Any]:
    """Usage / health / movement metrics of the (tied) dictionary."""
    if device.type != "cpu":
        raise RuntimeError(f"dictionary diagnostics are CPU-only, got device={device}")
    model.eval()
    usage = torch.zeros(int(model.D.shape[1]), dtype=torch.float64)
    n_rows = 0
    numerator = 0.0
    denominator = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            phi = batch.dict_phi
            coord = model.code(phi).detach().double()
            usage += (coord != 0).double().sum(dim=0)
            n_rows += int(coord.shape[0])
            reconstruction = model.reconstruct(phi, coord).detach()
            numerator += float(((phi.double() - reconstruction.double()) ** 2).sum())
            denominator += float((phi.double() ** 2).sum())
    share = usage / max(n_rows, 1)
    active = int((usage > 0).sum())
    probabilities = share / max(float(share.sum()), 1e-12)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * nonzero.log()).sum()) if int(nonzero.numel()) else 0.0
    first = torch.as_tensor(np.asarray(dictionary_init, dtype=np.float32))
    movement = float((model.D.detach().cpu() - first).norm())
    payload = {
        "active_atoms": active,
        "effective_atoms": float(math.exp(entropy)),
        "coord_sparsity_mean": float((usage > 0).double().sum() / max(n_rows, 1)),
        "usage_top1_share": float(share.max()) if int(share.numel()) else float("nan"),
        "valid_reconstruction_relative": float(numerator / max(denominator, 1e-12)),
        "dictionary_movement_frobenius": movement,
        "n_rows": int(n_rows),
    }
    if gradient_batch is not None:
        model.train()
        batch = gradient_batch.to(device)
        prediction, aux = model(batch, mask=None, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(H1_LAMBDA) * rec
        model.zero_grad(set_to_none=True)
        loss.backward()
        payload["task_gradient_norm_D"] = float(model.D.grad.detach().norm()) if model.D.grad is not None else 0.0
        model.zero_grad(set_to_none=True)
        model.eval()
    payload["official_test_loaded"] = False
    return payload


def train_spec(
    spec: CleanMechSpec,
    *,
    epochs: int,
    threads: int,
    out_dir: Any,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    seed: int = 0,
    init_state: Mapping[str, torch.Tensor] | None = None,
    tag: str | None = None,
    log: bool = True,
    save_states: bool = True,
) -> dict[str, Any]:
    """One CPU training run of an arm through the frozen ``audit.train_cpu`` loop."""
    tag = spec.tag if tag is None else str(tag)

    def factory(dictionary: np.ndarray, run_seed: int) -> CleanMechModel:
        return build_clean_mech_model(dictionary, run_seed, spec)

    return audit.train_cpu(
        tag=tag,
        mask=arm_mask(spec),
        epochs=int(epochs),
        threads=int(threads),
        out_dir=out_dir,
        init_state=init_state,
        train_data=train_data,
        valid_data=valid_data,
        seed=int(seed),
        save_states=bool(save_states),
        log=bool(log),
        model_factory=factory,
        arm_spec=spec.as_dict(),
    )


__all__ = [
    "PROTOCOL_VERSION",
    "H1_CONFIG",
    "H1_LAMBDA",
    "NODE_BINDINGS",
    "EDGE_BINDINGS",
    "CODINGS",
    "D_A",
    "D_E",
    "C6_MASK",
    "C1_MASK",
    "RELATION_CANDIDATES",
    "c6_equivalence_check",
    "CleanMechSpec",
    "CLEAN_MECH_ARMS",
    "arm_mask",
    "make_spec",
    "merge_masks",
    "final_clean_mask",
    "CleanMechModel",
    "build_clean_mech_model",
    "independence_norm_stats",
    "evaluate_binding_variant",
    "c6_gate",
    "independence_gate",
    "relation_gate",
    "specificity_gate",
    "dictionary_diagnostics",
    "train_spec",
]
