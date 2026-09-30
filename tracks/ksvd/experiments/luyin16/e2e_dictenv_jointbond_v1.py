"""E2E-DictEnv-JointBond-v1 — Key-level Joint Structure-Semantics Fusion (core).

Round ``e2e_dictenv_jointbond_v1`` (Workstream Z, ZINC).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_jointbond_v1_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_jointbond_v1_prior_artifact_audit.md``.

Scientific question (luyin19).  The parent binding forms a per-role product of a
shared structural coordinate and the *coarse* root semantics and then aggregates
before any nonlinearity; RNDB showed that a rolewise nonlinearity is load-bearing
but the task band did not move.  This round tests **one** hypothesis:

    Binding the structure coordinate of *each* bond endpoint to *that* endpoint's
    atom type, and then fusing both endpoints together with the bond type before
    the shellpair aggregation, improves prediction.

Exactly one additive residual branch on the real bonds of the frozen
``SEM108Model`` parent (which is itself ``CSSD-q1 + C6``)::

    alpha_v = coord[v, common_dim:]                       # 32-D residual code only
    q_v     = one_hot(dict_atom[v])                       # 28 atom categories
    b_uv    = one_hot(env_bond_type[uv])                  # 4 bond categories

    h_v          = (alpha_v @ A) * (q_v @ C)              # 16-D
    t_uv         = [h_u + h_v ; |h_u - h_v| ; h_u * h_v]  # 48-D
    j_uv         = F(t_uv) * (b_uv @ B)                   # 48-D
    ue_uv        = parent_edge_response_uv + j_uv         # added BEFORE index_add_

with ``A: 32->16``, ``C: 28->16``, ``B: 4->48`` bias-free linear maps and
``F = Linear(48,32,bias=False) -> SiLU -> Linear(32,48,bias=False)``.

The parent ``SEM108Model`` is *extended*, never copied: the only change is an
overridable edge-response hook (:meth:`SEM108Model._edge_response_delta`) whose
default returns ``None`` so the parent path stays bit-identical.  Everything else
(node path, original edge path, shell/shellpair routing, environment fusion,
relation / distance / global / topology backend, moment readout, reconstruction
objective, C6 mask, IHT-10, ``K=32``, ``s=8``, ``lambda=33.95873017865987``,
Adam / batch / clip, Top-5 soup) is the frozen parent's code.

CPU only.  ``official_test_loaded = False`` in every payload.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem

PROTOCOL_VERSION = "e2e_dictenv_jointbond_v1"

#: frozen widths (pre-registration section 2).  Never tuned this round.
JOINT_TAG = "JOINTBOND"
JOINT_ALPHA_DIM = int(cssd.K_ATOMS)  # 32 residual dictionary codes (common excluded)
JOINT_H = 16
JOINT_T_DIM = 3 * JOINT_H  # 48
JOINT_F_HIDDEN = 32
JOINT_ATOM_DIM = int(p2.ATOM_CATEGORIES)  # 28
JOINT_BOND_DIM = int(p2.BOND_CATEGORIES)  # 4
JOINT_EDGE_DIM = int(cm.D_E)  # 48, the frozen parent edge slot width
JOINT_INIT_STD = 0.01

#: exact expected parameter increment / total (pre-registration section 2).
JOINT_NEW_PARAMETERS = (
    JOINT_ALPHA_DIM * JOINT_H
    + JOINT_ATOM_DIM * JOINT_H
    + JOINT_BOND_DIM * JOINT_T_DIM
    + JOINT_T_DIM * JOINT_F_HIDDEN
    + JOINT_F_HIDDEN * JOINT_T_DIM
)  # 512 + 448 + 192 + 1536 + 1536 = 4224
SEM108_CANDIDATE_PARAMETERS = 97709
JOINT_EXPECTED_TOTAL = SEM108_CANDIDATE_PARAMETERS + JOINT_NEW_PARAMETERS  # 101933

#: parent interface the candidate extends (bit-identical shared parameters).
PARENT_MODEL = "e2e_dictenv_sem108_v1::SEM108Model"
PARENT_COMMIT = "da2af280d4229aa96fac248eecf38109ee56716b"
PARENT_SOUP_VALID_MAE = 0.123704927947314


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


def cpu_only_guard(device: Any) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise RuntimeError(f"JointBond round is CPU-only, got device={resolved}")


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointBondMask(sem.SEMMask):
    """Frozen-parent ``SEMMask`` plus the JointBond branch interventions.

    * ``joint_branch_off`` — skip ``j_uv`` entirely (bit-identical to the parent).
    * ``joint_alpha_zero`` — zero the residual dictionary code *inside the new
      branch only* (the common coordinate is never read by the branch).
    * ``use_joint_alpha_shuffle`` / ``joint_alpha_shuffle_seed`` — permute the
      residual code rows across the roots of each molecule for the branch only.
    * ``use_joint_atom_shuffle`` / ``joint_atom_shuffle_seed`` — permute the atom
      one-hot rows across the roots of each molecule for the branch only.
    """

    joint_branch_off: bool = False
    joint_alpha_zero: bool = False
    use_joint_alpha_shuffle: bool = False
    joint_alpha_shuffle_seed: int = 0
    use_joint_atom_shuffle: bool = False
    joint_atom_shuffle_seed: int = 0

    def is_identity(self) -> bool:
        if any(
            (
                self.joint_branch_off,
                self.joint_alpha_zero,
                self.use_joint_alpha_shuffle,
                self.use_joint_atom_shuffle,
            )
        ):
            return False
        return super().is_identity()

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload["is_identity"] = self.is_identity()
        return payload


def _as_joint_mask(mask: audit.AuditMask | None) -> JointBondMask:
    if mask is None:
        return JointBondMask()
    if isinstance(mask, JointBondMask):
        return mask
    fields = {field.name for field in dataclasses.fields(audit.AuditMask)}
    payload = {name: getattr(mask, name) for name in fields}
    for name in (
        "joint_branch_off",
        "joint_alpha_zero",
        "use_joint_alpha_shuffle",
        "joint_alpha_shuffle_seed",
        "use_joint_atom_shuffle",
        "joint_atom_shuffle_seed",
    ):
        payload[name] = getattr(mask, name, JointBondMask.__dataclass_fields__[name].default)
    return JointBondMask(**payload)


def merge_joint_mask(base: audit.AuditMask | None, probe: JointBondMask | None) -> JointBondMask:
    """Union of the frozen training mask (C6) and any Sem108 / JointBond probe."""
    merged = sem.merge_sem_mask(base, probe)
    probe = _as_joint_mask(probe)
    fields = {
        field.name: getattr(merged, field.name) for field in dataclasses.fields(sem.SEMMask)
    }
    fields.update(
        joint_branch_off=bool(probe.joint_branch_off),
        joint_alpha_zero=bool(probe.joint_alpha_zero),
        use_joint_alpha_shuffle=bool(probe.use_joint_alpha_shuffle),
        joint_alpha_shuffle_seed=int(
            probe.joint_alpha_shuffle_seed if probe.use_joint_alpha_shuffle else 0
        ),
        use_joint_atom_shuffle=bool(probe.use_joint_atom_shuffle),
        joint_atom_shuffle_seed=int(
            probe.joint_atom_shuffle_seed if probe.use_joint_atom_shuffle else 0
        ),
    )
    return JointBondMask(**fields)


# ---------------------------------------------------------------------------
# the new branch as a pure function of the two endpoint pairs
# ---------------------------------------------------------------------------


def joint_branch_from_endpoints(
    model: "JointBondModel",
    alpha_u: torch.Tensor,
    q_u: torch.Tensor,
    alpha_v: torch.Tensor,
    q_v: torch.Tensor,
    bond_type: torch.Tensor,
) -> torch.Tensor:
    """``j_uv`` from the two *endpoint-bound* (structure, semantics) pairs.

    ``alpha_*`` are the 32-D residual dictionary codes, ``q_*`` the 28-D atom
    one-hots and ``bond_type`` the 4 categories.  The structural role of an
    endpoint multiplies **that endpoint's** atom semantics, so swapping both
    endpoints (structure and semantics together) leaves the response identical,
    while swapping only the semantics or only the structure generally changes it.
    """
    h_u = model.joint_A(alpha_u) * model.joint_C(q_u)
    h_v = model.joint_A(alpha_v) * model.joint_C(q_v)
    t = torch.cat([h_u + h_v, torch.abs(h_u - h_v), h_u * h_v], dim=1)
    b = F.one_hot(bond_type.to(torch.long), num_classes=JOINT_BOND_DIM).to(h_u.dtype)
    return model.joint_F(t) * model.joint_B(b)


def _shuffle_rows(values: torch.Tensor, batch: torch.Tensor | None, seed: int) -> torch.Tensor:
    """Permute rows *within each molecule* (fixed seed, no cross-graph leakage)."""
    generator = torch.Generator().manual_seed(int(seed))
    n_rows = int(values.shape[0])
    if batch is None:
        return values[torch.randperm(n_rows, generator=generator)]
    n_graphs = int(batch.max().item()) + 1 if int(batch.numel()) else 0
    index = torch.empty(n_rows, dtype=torch.long)
    for graph in range(n_graphs):
        rows = (batch == graph).nonzero(as_tuple=False).view(-1)
        index[rows] = rows[torch.randperm(int(rows.numel()), generator=generator)]
    return values[index]


# ---------------------------------------------------------------------------
# the model (extends the parent through one hook; never a re-typed copy)
# ---------------------------------------------------------------------------


class JointBondModel(sem.SEM108Model):
    """``SEM108Model`` + the key-level joint structure-semantics residual branch.

    The branch is summed into the per-bond edge response *before* the shellpair
    ``index_add_``.  With ``joint_branch_off`` the forward is bit-identical to
    ``SEM108Model`` with the same weights.  The parent parameters keep the exact
    ``build_sem108_model`` RNG stream; the branch is initialised afterwards and
    never consumes parent RNG draws.
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: cssd.CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
        )
        if self.node_binding != "paired" or self.edge_binding != "paired":
            raise ValueError("JointBond requires the paired node and edge bindings")
        if int(self.config.d_e) != JOINT_EDGE_DIM:
            raise ValueError(
                f"JointBond is frozen on the parent edge width {JOINT_EDGE_DIM}, "
                f"got d_e={int(self.config.d_e)}"
            )
        self.joint_A = nn.Linear(JOINT_ALPHA_DIM, JOINT_H, bias=False)
        self.joint_C = nn.Linear(JOINT_ATOM_DIM, JOINT_H, bias=False)
        self.joint_B = nn.Linear(JOINT_BOND_DIM, JOINT_T_DIM, bias=False)
        self.joint_F = nn.Sequential(
            nn.Linear(JOINT_T_DIM, JOINT_F_HIDDEN, bias=False),
            nn.SiLU(),
            nn.Linear(JOINT_F_HIDDEN, JOINT_T_DIM, bias=False),
        )
        # frozen one-shot initialisation (pre-registration section 2):
        # A / C / B / F-W1 Kaiming-uniform, F-W2 Normal(0, 0.01) once.
        for layer in (self.joint_A, self.joint_C, self.joint_B, self.joint_F[0]):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5.0))
        with torch.no_grad():
            self.joint_F[2].weight.normal_(0.0, JOINT_INIT_STD)

    # -- branch inputs ---------------------------------------------------------

    def joint_endpoint_inputs(
        self,
        coord: torch.Tensor,
        data: Any,
        bond_u: torch.Tensor,
        bond_v: torch.Tensor,
        mask: audit.AuditMask | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(alpha_u, q_u, alpha_v, q_v, bond_type)`` for every real bond occurrence.

        Structure and semantics are gathered at the *same* endpoint index, so the
        endpoint correspondence can never be silently broken: the existing
        structure-assignment shuffle moves ``bond_u`` / ``bond_v`` and both are
        read at the shuffled (still real) endpoint.
        """
        alpha = coord[:, self.common_dim :]
        if int(alpha.shape[1]) != JOINT_ALPHA_DIM:
            raise RuntimeError(
                f"residual code width {int(alpha.shape[1])} != {JOINT_ALPHA_DIM}"
            )
        q = F.one_hot(data.dict_atom, num_classes=JOINT_ATOM_DIM).to(coord.dtype)
        probe = None if mask is None else _as_joint_mask(mask)
        if probe is not None and probe.joint_alpha_zero:
            alpha = torch.zeros_like(alpha)
        if probe is not None and probe.use_joint_alpha_shuffle:
            alpha = _shuffle_rows(alpha, getattr(data, "batch", None), probe.joint_alpha_shuffle_seed)
        if probe is not None and probe.use_joint_atom_shuffle:
            q = _shuffle_rows(q, getattr(data, "batch", None), probe.joint_atom_shuffle_seed)
        bu = bond_u.to(coord.device)
        bv = bond_v.to(coord.device)
        return (
            alpha[bu],
            q[bu],
            alpha[bv],
            q[bv],
            data.env_bond_type.to(coord.device),
        )

    def joint_edge_response(
        self,
        coord: torch.Tensor,
        data: Any,
        bond_u: torch.Tensor,
        bond_v: torch.Tensor,
        mask: audit.AuditMask | None = None,
    ) -> torch.Tensor:
        """``j_uv`` for every bond occurrence, ``[n_bond, 48]``."""
        alpha_u, q_u, alpha_v, q_v, bond_type = self.joint_endpoint_inputs(
            coord, data, bond_u, bond_v, mask
        )
        return joint_branch_from_endpoints(self, alpha_u, q_u, alpha_v, q_v, bond_type)

    def _resolve_bond_endpoints(
        self,
        data: Any,
        mask: audit.AuditMask | None,
        bond_u: torch.Tensor | None,
        bond_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The endpoints the environment path would actually read (shuffle-aware)."""
        if (
            bond_u is None
            and bond_v is None
            and mask is not None
            and bool(getattr(mask, "use_edge_shuffle", False))
        ):
            shuffled_u = getattr(data, "env_bond_u_shuffled", None)
            shuffled_v = getattr(data, "env_bond_v_shuffled", None)
            if shuffled_u is None or shuffled_v is None:
                raise RuntimeError(
                    "edge-shuffle intervention requires shuffled endpoint fields"
                )
            return shuffled_u, shuffled_v
        return (
            data.env_bond_u if bond_u is None else bond_u,
            data.env_bond_v if bond_v is None else bond_v,
        )

    def joint_edge_delta(
        self,
        coord: torch.Tensor,
        data: Any,
        *,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        mask: audit.AuditMask | None = None,
        edge_binding_zero: bool = False,
    ) -> torch.Tensor | None:
        """The branch's contribution to the environment edge response (``None`` if off)."""
        bu, bv = self._resolve_bond_endpoints(data, mask, bond_u, bond_v)
        return self._edge_response_delta(
            coord, data, bu, bv, mask, edge_binding_zero=edge_binding_zero
        )

    # -- the single parent hook -------------------------------------------------

    def _edge_response_delta(
        self,
        coord: torch.Tensor,
        data: Any,
        bond_u: torch.Tensor,
        bond_v: torch.Tensor,
        mask: audit.AuditMask | None = None,
        *,
        edge_binding_zero: bool = False,
    ) -> torch.Tensor | None:
        probe = None if mask is None else _as_joint_mask(mask)
        if probe is not None and probe.joint_branch_off:
            return None
        return self.joint_edge_response(coord, data, bond_u, bond_v, mask)

    # -- diagnostics -----------------------------------------------------------

    def joint_edge_slots(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: audit.AuditMask | None = None,
        *,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        edge_binding_zero: bool = False,
    ) -> torch.Tensor:
        """Environment edge slots including the branch (uses the parent routing)."""
        bu, bv = self._resolve_bond_endpoints(data, mask, bond_u, bond_v)
        return self._edge_env_slots(
            coord,
            data,
            bu,
            bv,
            edge_binding_zero=bool(edge_binding_zero),
            mask=mask,
        )

    def joint_parameters(self) -> dict[str, nn.Parameter]:
        return {
            "joint_A.weight": self.joint_A.weight,
            "joint_C.weight": self.joint_C.weight,
            "joint_B.weight": self.joint_B.weight,
            "joint_F.0.weight": self.joint_F[0].weight,
            "joint_F.2.weight": self.joint_F[2].weight,
        }


def build_jointbond_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    spec: cm.CleanMechSpec | None = None,
) -> JointBondModel:
    """Build JointBond with the exact parent RNG stream for all shared params."""
    spec = cssd.CSSD_SPEC if spec is None else spec
    torch.manual_seed(int(seed))
    return JointBondModel(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
    )


# ---------------------------------------------------------------------------
# parameter audit
# ---------------------------------------------------------------------------


def parameter_audit(model: JointBondModel, parent_params: int) -> dict[str, Any]:
    """Exact parameter accounting of the frozen branch (no width change)."""
    breakdown = {
        "joint_A": int(sum(p.numel() for p in model.joint_A.parameters())),
        "joint_C": int(sum(p.numel() for p in model.joint_C.parameters())),
        "joint_B": int(sum(p.numel() for p in model.joint_B.parameters())),
        "joint_F": int(sum(p.numel() for p in model.joint_F.parameters())),
    }
    new_params = int(sum(breakdown.values()))
    total = int(sum(p.numel() for p in model.parameters()))
    parent = int(parent_params)
    return {
        "parent_model": PARENT_MODEL,
        "parent_commit": PARENT_COMMIT,
        "parent_params": parent,
        "candidate_params": total,
        "new_params": new_params,
        "breakdown": breakdown,
        "expected_new_params": int(JOINT_NEW_PARAMETERS),
        "new_params_exact": bool(new_params == JOINT_NEW_PARAMETERS),
        "expected_total": int(parent + JOINT_NEW_PARAMETERS),
        "total_exact": bool(total == parent + JOINT_NEW_PARAMETERS),
        "delta": int(total - parent),
        "relative_delta": float((total - parent) / max(parent, 1)),
        "widths": {
            "A": [JOINT_ALPHA_DIM, JOINT_H],
            "C": [JOINT_ATOM_DIM, JOINT_H],
            "B": [JOINT_BOND_DIM, JOINT_T_DIM],
            "F": [JOINT_T_DIM, JOINT_F_HIDDEN, JOINT_T_DIM],
        },
        "parameter_matched": False,
        "parameter_note": (
            "the round brief authorises this un-matched +4224 parameter residual branch; "
            "no width was changed to match a budget"
        ),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# synthetic batches (data-free correctness checks)
# ---------------------------------------------------------------------------


def synthetic_batch(
    n_nodes: int = 40,
    n_occ: int = 120,
    n_bond: int = 90,
    seed: int = 0,
    *,
    n_graphs: int = 1,
) -> Any:
    """A deterministic synthetic batch with the real field names / dtypes."""
    generator = torch.Generator().manual_seed(int(seed))
    data = _namespace()
    data.dict_phi = torch.randn(n_nodes, cssd.PHI_DIM, generator=generator)
    data.dict_atom = torch.randint(0, JOINT_ATOM_DIM, (n_nodes,), generator=generator)
    data.anchor = torch.randn(n_nodes, audit.ANCHOR_DIM_EXPECTED, generator=generator)
    data.patch_cont = torch.randn(n_nodes, sem.PATCH_CONT_DIM, generator=generator)
    data.env_occ_node = torch.randint(0, n_nodes, (n_occ,), generator=generator)
    data.env_occ_root = torch.randint(0, n_nodes, (n_occ,), generator=generator)
    data.env_occ_shell = torch.randint(0, int(p2.N_SHELLS), (n_occ,), generator=generator)
    data.env_bond_u = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_v = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_type = torch.randint(0, JOINT_BOND_DIM, (n_bond,), generator=generator)
    data.env_bond_root = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_shellpair = torch.randint(
        0, int(p2.SHELLPAIR_CLASSES), (n_bond,), generator=generator
    )
    if n_graphs > 1:
        data.batch = torch.sort(
            torch.randint(0, int(n_graphs), (n_nodes,), generator=generator)
        ).values
    else:
        data.batch = torch.zeros(n_nodes, dtype=torch.long)
    data.num_graphs = int(data.batch.max().item()) + 1
    data.y = torch.randn(int(data.num_graphs), generator=generator)
    data.global_context = torch.randn(int(data.num_graphs), p2.GLOBAL_WIDTH, generator=generator)
    data.topology_features = torch.randn(int(data.num_graphs), p2.TOPOLOGY_IN, generator=generator)
    data.pair_relation = torch.randn(5, 21, generator=generator)
    data.pair_bucket = torch.randint(0, int(p2.DISTANCE_BUCKETS), (5,), generator=generator)
    data.pair_index = torch.stack(
        [torch.randint(0, n_nodes, (5,), generator=generator) for _ in range(2)]
    )
    return data


def _namespace() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace()


def reference_edge_slots(
    model: JointBondModel,
    coord: torch.Tensor,
    data: Any,
    *,
    bond_u: torch.Tensor | None = None,
    bond_v: torch.Tensor | None = None,
    mask: audit.AuditMask | None = None,
    edge_binding_zero: bool = False,
) -> np.ndarray:
    """Independent float64 reference of the environment edge slots.

    Recomputes the parent response and the branch from the endpoint pairs and
    aggregates with ``numpy.add.at`` on ``root * SHELLPAIR_CLASSES + shellpair``,
    i.e. without the model's ``index_add_``.  Used to validate routing / batch
    offsets.
    """
    bu = (data.env_bond_u if bond_u is None else bond_u).to(coord.device)
    bv = (data.env_bond_v if bond_v is None else bond_v).to(coord.device)
    with torch.no_grad():
        x = coord.double()
        cu, cv = x[bu], x[bv]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        if edge_binding_zero:
            g = torch.zeros_like(g)
        b = F.one_hot(data.env_bond_type, num_classes=JOINT_BOND_DIM).to(x.dtype)
        ue = (g @ model.W_E_S.double()) * (b @ model.W_E_C.double()) / math.sqrt(float(JOINT_EDGE_DIM))
        delta = model._edge_response_delta(
            coord, data, bu, bv, mask, edge_binding_zero=edge_binding_zero
        )
        if delta is not None:
            ue = ue + delta.double()
    n = int(coord.shape[0])
    out = np.zeros((n * int(p2.SHELLPAIR_CLASSES), JOINT_EDGE_DIM), dtype=np.float64)
    index = (
        data.env_bond_root.to(coord.device).numpy() * int(p2.SHELLPAIR_CLASSES)
        + data.env_bond_shellpair.to(coord.device).numpy()
    )
    np.add.at(out, index, ue.numpy())
    return out.reshape(n, int(p2.SHELLPAIR_CLASSES), JOINT_EDGE_DIM)


def parameter_norms(model: JointBondModel) -> dict[str, Any]:
    """Weight-scale / near-zero health of the frozen branch parameters."""
    payload: dict[str, Any] = {"official_test_loaded": False}
    for name, parameter in model.joint_parameters().items():
        value = parameter.detach().double()
        payload[name] = {
            "numel": int(value.numel()),
            "norm": float(value.norm()),
            "absmax": float(value.abs().max()),
            "absmean": float(value.abs().mean()),
            "frac_abs_lt_1e12": float((value.abs() < 1e-12).double().mean()),
        }
    payload["joint_branch_dead"] = bool(
        max(float(payload[name]["absmax"]) for name in model.joint_parameters()) < 1.0e-30
    )
    return payload


def response_scale_stats(
    model: JointBondModel,
    coord: torch.Tensor,
    data: Any,
    mask: audit.AuditMask | None = None,
    *,
    edge_binding_zero: bool = False,
) -> dict[str, Any]:
    """Per-bond magnitudes of the branch and of the parent edge response."""
    bu = data.env_bond_u
    bv = data.env_bond_v
    with torch.no_grad():
        delta = model.joint_edge_delta(coord, data, mask=mask, edge_binding_zero=edge_binding_zero)
        parent_ue, _delta = model._edge_env_parts(
            coord, data, bu, bv, edge_binding_zero=bool(edge_binding_zero), mask=None
        )

    def _stats(values: torch.Tensor) -> dict[str, float]:
        norm = values.detach().double().norm(dim=1)
        if int(norm.numel()) == 0:
            return {key: float("nan") for key in ("mean", "median", "p90", "max")}
        return {
            "mean": float(norm.mean()),
            "median": float(norm.median()),
            "p90": float(torch.quantile(norm, 0.90)),
            "max": float(norm.max()),
        }

    zeros = (
        delta
        if delta is not None
        else torch.zeros(int(bu.numel()), JOINT_EDGE_DIM, dtype=parent_ue.dtype)
    )
    parent_norm = parent_ue.detach().double().norm(dim=1)
    branch_norm = zeros.detach().double().norm(dim=1)
    ratio = branch_norm / torch.clamp(parent_norm, min=1e-12)
    return {
        "parent_edge_response": _stats(parent_ue),
        "joint_branch_response": _stats(zeros),
        "joint_branch_zero_fraction": float((branch_norm == 0).double().mean()),
        "branch_to_parent_norm_ratio": {
            "mean": float(ratio.mean()),
            "median": float(ratio.median()),
            "p90": float(torch.quantile(ratio, 0.90)),
        },
        "official_test_loaded": False,
    }


__all__ = [
    "PROTOCOL_VERSION",
    "JOINT_TAG",
    "JOINT_ALPHA_DIM",
    "JOINT_H",
    "JOINT_T_DIM",
    "JOINT_F_HIDDEN",
    "JOINT_ATOM_DIM",
    "JOINT_BOND_DIM",
    "JOINT_EDGE_DIM",
    "JOINT_INIT_STD",
    "JOINT_NEW_PARAMETERS",
    "SEM108_CANDIDATE_PARAMETERS",
    "JOINT_EXPECTED_TOTAL",
    "PARENT_MODEL",
    "PARENT_COMMIT",
    "PARENT_SOUP_VALID_MAE",
    "official_test_blocker",
    "cpu_only_guard",
    "JointBondMask",
    "merge_joint_mask",
    "joint_branch_from_endpoints",
    "JointBondModel",
    "build_jointbond_model",
    "parameter_audit",
    "synthetic_batch",
    "reference_edge_slots",
    "parameter_norms",
    "response_scale_stats",
]
