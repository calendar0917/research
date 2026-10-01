"""E2E-DictEnv-Latent-Bridge-v1 — SEM108 + local task dictionary bridge (core).

Round ``e2e_dictenv_latent_bridge_v1`` (Workstream Z, ZINC).
Pre-registration: ``tracks/ksvd/notes/zinc_e2e_dictenv_latent_bridge_v1_preregistration.md``.
Delivered reference package (vendored):
``tracks/ksvd/experiments/luyin16/latent_bridge_reference/v1_20261001/``.

The candidate keeps the complete frozen Sem108 computation and inserts **one**
shared low-dimensional task dictionary between the local fusion output
``h_i in R^48`` and every downstream consumer of the local environment::

    static local object -> Sem108 fusion -> h_i[48]
        -> shared task dictionary (soft-threshold coding) alpha_i[96]
        -> dictionary value table E_i[48]
        -> unchanged unary pooling + unchanged static pair computation
        -> unchanged single graph readout

* Same ``D_L`` / ``V_L`` for every molecule and every local object.
* No message passing, no graph index, no node/graph hidden state and no
  neighbour write-back enters the bridge.  The unrolled 16-step sparse solve is
  per-object only.
* All local and pair paths read ``E``; there is **no** ``h + correction``
  residual bypass and no new raw-feature bypass.

The math is the delivered NumPy reference translated to PyTorch:

    rho_i = sqrt(mean(h_i^2) + 1e-12),  x_i = h_i / rho_i
    Dbar_L = column-normalised D_L                         (48 x 96)
    L = 1.05 * eigmax(Dbar_L Dbar_L^T) + lambda2            (detached solver step)
    alpha^0 = 0
    alpha^{t+1} = soft_threshold(alpha^t + eta*((x - alpha^t Dbar^T) Dbar
                                                - lambda2 alpha^t), eta*lambda1)
    eta = 1 / L,  lambda1 = 0.05,  lambda2 = 0.01,  t = 0..15
    E_i = rho_i * (alpha_i V_L)

Fixed 16 steps: the module is an unrolled ISTA encoder, **not** an exact
elastic-net optimum claim.  ``L`` is a detached solver setting; the coding
computation keeps the task gradient (D_L and V_L both accept task gradients).

Initialisation (private NumPy RNG, never the parent torch stream)::

    Q = deterministic QR of a 48x48 Gaussian matrix (seed 0)
    D_L = [I48, Q]     (two orthonormal bases, unit columns)
    V_L = D_L.T        (same 96 entries, independent task-trained value table)

CPU only.  ``official_test_loaded = False`` always.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem

PROTOCOL_VERSION = "e2e_dictenv_latent_bridge_v1"

#: environment width of the frozen parent (the bridge input and output width).
BRIDGE_DIM = int(p2.ENV_DIM)  # 48
#: number of shared dictionary entries (two 48-wide orthogonal bases).
BRIDGE_ATOMS = 96
#: fixed unrolled ISTA steps (not tuned, not selected on validation).
BRIDGE_STEPS = 16
BRIDGE_LAMBDA1 = 0.05
BRIDGE_LAMBDA2 = 0.01
BRIDGE_STEP_SAFETY = 1.05
BRIDGE_EPS = 1.0e-12
#: private initialisation seed, deliberately independent of the model seed.
BRIDGE_SEED = 0

#: parameter contract: D_L + V_L only.
BRIDGE_PARAMETERS = int(BRIDGE_DIM * BRIDGE_ATOMS * 2)
SEM108_PARAMETERS = 97709
CANDIDATE_PARAMETERS = int(SEM108_PARAMETERS + BRIDGE_PARAMETERS)


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


# ---------------------------------------------------------------------------
# reference frame initialisation (NumPy, private RNG)
# ---------------------------------------------------------------------------


def frame_initialization(dim: int = BRIDGE_DIM, seed: int = BRIDGE_SEED):
    """Unit-column tight frame ``[I, Q]`` and its value table ``D.T``.

    ``Q`` is the sign-corrected ``Q`` factor of a deterministic Gaussian QR
    factorisation drawn from a private ``np.random.default_rng(seed)`` stream.
    Using NumPy here guarantees that building the candidate never consumes or
    reorders the parent torch RNG stream.
    """
    rng = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(rng.normal(size=(int(dim), int(dim))))
    q = q * np.where(np.diag(r) < 0.0, -1.0, 1.0)[None, :]
    dictionary = np.concatenate([np.eye(int(dim)), q], axis=1).astype(np.float32)
    values = dictionary.T.copy()
    return dictionary, values


def soft_threshold(value: torch.Tensor, threshold: float) -> torch.Tensor:
    """``sign(v) * max(|v| - t, 0)`` (torch translation of the reference)."""
    return torch.sign(value) * torch.clamp(value.abs() - float(threshold), min=0.0)


# ---------------------------------------------------------------------------
# the bridge module
# ---------------------------------------------------------------------------


class LatentDictionaryBridge(nn.Module):
    """Per-object soft-threshold dictionary bridge ``h[48] -> E[48]``.

    The module is deliberately row-local: it never receives ``edge_index``,
    ``graph_id`` or any other node/graph hidden state, and it performs no
    state write-back.  The 16 ISTA iterations run inside each object only.
    """

    def __init__(
        self,
        dim: int = BRIDGE_DIM,
        n_atoms: int = BRIDGE_ATOMS,
        *,
        seed: int = BRIDGE_SEED,
        lambda1: float = BRIDGE_LAMBDA1,
        lambda2: float = BRIDGE_LAMBDA2,
        steps: int = BRIDGE_STEPS,
    ) -> None:
        super().__init__()
        dim = int(dim)
        n_atoms = int(n_atoms)
        if n_atoms != 2 * dim:
            raise ValueError(
                f"bridge frame requires n_atoms == 2 * dim, got {n_atoms} vs {dim}"
            )
        if int(steps) <= 0 or float(lambda1) < 0.0 or float(lambda2) < 0.0:
            raise ValueError("invalid bridge solver setting")
        dictionary, values = frame_initialization(dim, int(seed))
        self.dim = dim
        self.n_atoms = n_atoms
        self.seed = int(seed)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.steps = int(steps)
        self.D_L = nn.Parameter(torch.as_tensor(dictionary, dtype=torch.float32).clone())
        self.V_L = nn.Parameter(torch.as_tensor(values, dtype=torch.float32).clone())
        #: non-persistent initial state for the reset-to-init inference probe.
        self.register_buffer(
            "D_init", torch.as_tensor(dictionary, dtype=torch.float32).clone(), persistent=False
        )
        self.register_buffer(
            "V_init", torch.as_tensor(values, dtype=torch.float32).clone(), persistent=False
        )
        #: inference-only intervention flag (zero the whole bridge code).
        self._zero_code = False

    # -- frozen operators ------------------------------------------------------

    def normalized_dictionary(self) -> torch.Tensor:
        """``Dbar_L``: unit-column dictionary (solver input, keeps gradients)."""
        norms = self.D_L.norm(dim=0, keepdim=True).clamp_min(BRIDGE_EPS)
        return self.D_L / norms

    def solver_step(self, dictionary: torch.Tensor) -> float:
        """Detached conservative step ``1 / (1.05 * eigmax(Dbar Dbar^T) + lambda2)``."""
        with torch.no_grad():
            gram = dictionary.detach() @ dictionary.detach().t()
            spectral = float(torch.linalg.eigvalsh(gram.double()).max().item())
        return float(1.0 / (BRIDGE_STEP_SAFETY * spectral + self.lambda2))

    def codes(self, x: torch.Tensor, dictionary: torch.Tensor, step: float) -> torch.Tensor:
        """Unrolled 16-step ISTA on the normalised coordinates (gradients kept)."""
        alpha = torch.zeros(
            (int(x.shape[0]), self.n_atoms), dtype=x.dtype, device=x.device
        )
        threshold = float(step) * self.lambda1
        for _ in range(self.steps):
            gradient = (alpha @ dictionary.t() - x) @ dictionary + self.lambda2 * alpha
            alpha = soft_threshold(alpha - float(step) * gradient, threshold)
        return alpha

    # -- forward ---------------------------------------------------------------

    def forward(
        self, h: torch.Tensor, *, return_aux: bool = False
    ):
        if int(h.shape[1]) != self.dim:
            raise RuntimeError(f"bridge input width {int(h.shape[1])} != {self.dim}")
        scale = torch.sqrt(h.pow(2).mean(dim=1, keepdim=True) + BRIDGE_EPS)
        x = h / scale
        dictionary = self.normalized_dictionary()
        if self._zero_code:
            alpha = torch.zeros(
                (int(h.shape[0]), self.n_atoms), dtype=h.dtype, device=h.device
            )
        else:
            step = self.solver_step(dictionary)
            alpha = self.codes(x, dictionary, step)
        E = scale * (alpha @ self.V_L)
        if return_aux:
            return E, {"alpha": alpha, "normalized": x, "scale": scale}
        return E

    # -- inference-only interventions -----------------------------------------

    def reset_to_initialization(self) -> None:
        """Reset D_L / V_L to the frozen initial frame (inference diagnostic)."""
        with torch.no_grad():
            self.D_L.copy_(self.D_init)
            self.V_L.copy_(self.V_init)

    def set_zero_code(self, enabled: bool) -> None:
        self._zero_code = bool(enabled)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


def _permute_rows_within_graph(h: torch.Tensor, batch: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic within-molecule row permutation (inference-only probe)."""
    batch = batch.to(h.device)
    generator = torch.Generator().manual_seed(int(seed))
    n_rows = int(h.shape[0])
    if int(batch.numel()) != n_rows:
        raise RuntimeError("batch vector must align with the bridge rows")
    index = torch.empty(n_rows, dtype=torch.long)
    n_graphs = int(batch.max().item()) + 1 if n_rows else 0
    for graph in range(n_graphs):
        rows = (batch == graph).nonzero(as_tuple=False).view(-1)
        index[rows.to(h.device)] = rows[torch.randperm(int(rows.numel()), generator=generator)]
    return h[index.to(h.device)]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class LatentBridgeSEM108(sem.SEM108Model):
    """Sem108 with the shared local task dictionary inserted after the fusion.

    The single insertion point is ``_environment_from_parts``; both the plain
    ``environments`` path and the training ``environments_masked`` path call it,
    so every downstream consumer of ``E`` (unary pooling and pair projection)
    reads the bridged environment.  No residual bypass is added.
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
        fusion_hidden: int = sem.SEM_FUSION_HIDDEN,
        keep_anchor_encoder: bool = False,
        parent_interface: bool = False,
        bridge_dim: int = BRIDGE_DIM,
        bridge_atoms: int = BRIDGE_ATOMS,
        bridge_seed: int = BRIDGE_SEED,
        bridge_lambda1: float = BRIDGE_LAMBDA1,
        bridge_lambda2: float = BRIDGE_LAMBDA2,
        bridge_steps: int = BRIDGE_STEPS,
    ) -> None:
        super().__init__(
            config,
            dictionary,
            subspace=subspace,
            node_binding=node_binding,
            edge_binding=edge_binding,
            coding=coding,
            fusion_hidden=fusion_hidden,
            keep_anchor_encoder=keep_anchor_encoder,
            parent_interface=parent_interface,
        )
        if int(bridge_dim) != int(p2.ENV_DIM):
            raise ValueError(
                f"the bridge must keep the parent environment width {p2.ENV_DIM}, got {bridge_dim}"
            )
        # Private NumPy RNG: the parent torch stream is untouched, so every
        # shared parameter name keeps the frozen Sem108 initialisation.
        self.local_dictionary_bridge = LatentDictionaryBridge(
            dim=int(bridge_dim),
            n_atoms=int(bridge_atoms),
            seed=int(bridge_seed),
            lambda1=float(bridge_lambda1),
            lambda2=float(bridge_lambda2),
            steps=int(bridge_steps),
        )
        #: inference-only within-molecule row permutation seed (``None`` = off).
        self._bridge_permute_seed: int | None = None

    # -- single insertion point -------------------------------------------------

    def _environment_from_parts(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        h = super()._environment_from_parts(*args, **kwargs)
        if self._bridge_permute_seed is not None:
            data = kwargs.get("data")
            if data is None and len(args) >= 2:
                data = args[1]
            batch = getattr(data, "batch", None)
            if batch is None:
                raise RuntimeError("within-molecule code permutation requires data.batch")
            h = _permute_rows_within_graph(h, batch, int(self._bridge_permute_seed))
        return self.local_dictionary_bridge(h)

    # -- inference-only interventions -----------------------------------------

    def set_bridge_intervention(
        self, *, zero_code: bool | None = None, permute_seed: int | None = None
    ) -> None:
        if zero_code is not None:
            self.local_dictionary_bridge.set_zero_code(bool(zero_code))
        if permute_seed is not None:
            self._bridge_permute_seed = int(permute_seed)

    def clear_bridge_intervention(self) -> None:
        self.local_dictionary_bridge.set_zero_code(False)
        self._bridge_permute_seed = None

    def reset_bridge_to_initialization(self) -> None:
        self.local_dictionary_bridge.reset_to_initialization()

    # -- diagnostics ------------------------------------------------------------

    def environment_parts_with_aux(self, coord: torch.Tensor, data: Any, **kwargs: Any):
        """``(E, bridge_aux)`` on the plain path (diagnostics / audits only)."""
        h = sem.SEM108Model._environment_from_parts(self, coord, data, **kwargs)
        return self.local_dictionary_bridge(h, return_aux=True)


def build_latent_bridge_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: cssd.CommonSubspace,
    spec: cm.CleanMechSpec | None = None,
    **bridge_kwargs: Any,
) -> LatentBridgeSEM108:
    """Build the candidate with the exact Sem108 RNG stream for shared params.

    Extra keyword arguments are forwarded to ``LatentDictionaryBridge`` (used by
    the acceptance audit to build the ``lambda1 = lambda2 = 0`` identity mode).
    """
    spec = cssd.CSSD_SPEC if spec is None else spec
    torch.manual_seed(int(seed))
    return LatentBridgeSEM108(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
        **bridge_kwargs,
    )


# ---------------------------------------------------------------------------
# parameter audit
# ---------------------------------------------------------------------------


def bridge_parameter_audit(model: LatentBridgeSEM108, *, expected_sem_params: int = SEM108_PARAMETERS) -> dict[str, Any]:
    """Exact accounting of the two new matrices and the unchanged Sem108 body."""
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    bridge = int(model.local_dictionary_bridge.parameter_count())
    body = int(total - bridge)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "dictionary_shape": [int(model.local_dictionary_bridge.dim), int(model.local_dictionary_bridge.n_atoms)],
        "values_shape": [int(model.local_dictionary_bridge.n_atoms), int(model.local_dictionary_bridge.dim)],
        "bridge_params": bridge,
        "sem108_body_params": body,
        "expected_bridge_params": BRIDGE_PARAMETERS,
        "expected_sem108_body_params": int(expected_sem_params),
        "sem108_body_unchanged": bool(body == int(expected_sem_params)),
        "bridge_exact": bool(bridge == BRIDGE_PARAMETERS),
        "candidate_params": total,
        "expected_candidate_params": CANDIDATE_PARAMETERS,
        "candidate_exact": bool(total == CANDIDATE_PARAMETERS),
        "dictionary_parameter_names": [
            "local_dictionary_bridge.D_L",
            "local_dictionary_bridge.V_L",
        ],
        "parent_D_untouched": True,
    }


def bridge_code_stats(alpha: torch.Tensor) -> dict[str, Any]:
    """Variable-density summary of the bridge code (never called 'fixed s')."""
    alpha = alpha.detach().double()
    support = alpha != 0.0
    l0 = support.sum(dim=1).to(dtype=torch.float64)
    if int(alpha.numel()) == 0:
        return {
            "n_objects": 0,
            "mean_nonzero": 0.0,
            "mean_nonzero_fraction": 0.0,
            "l0_p50": 0.0,
            "l0_p95": 0.0,
            "l0_max": 0.0,
        }
    return {
        "n_objects": int(alpha.shape[0]),
        "mean_nonzero": float(l0.mean()),
        "mean_nonzero_fraction": float(l0.mean() / float(alpha.shape[1])),
        "l0_p50": float(torch.quantile(l0, 0.50)),
        "l0_p95": float(torch.quantile(l0, 0.95)),
        "l0_max": float(l0.max()),
        "note": "variable density; no fixed l0 budget and no top-k truncation",
    }


def bridge_reconstruction_diagnostic(
    model: LatentBridgeSEM108, aux: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """``||x - alpha Dbar^T||^2`` diagnostic (never added to the outer loss)."""
    x = aux["normalized"].detach().double()
    alpha = aux["alpha"].detach().double()
    dictionary = model.local_dictionary_bridge.normalized_dictionary().detach().double()
    residual = x - alpha @ dictionary.t()
    per_row = (residual * residual).sum(dim=1)
    denominator = (x * x).sum(dim=1).clamp_min(BRIDGE_EPS)
    per_row_relative = per_row / denominator
    return {
        "mean_squared": float(per_row.mean()),
        "p95_squared": float(torch.quantile(per_row, 0.95)),
        "mean_relative": float(per_row_relative.mean()),
        "p95_relative": float(torch.quantile(per_row_relative, 0.95)),
        "note": "diagnostic only; the low-dimensional reconstruction term is never optimised",
    }


__all__ = [
    "PROTOCOL_VERSION",
    "BRIDGE_DIM",
    "BRIDGE_ATOMS",
    "BRIDGE_STEPS",
    "BRIDGE_LAMBDA1",
    "BRIDGE_LAMBDA2",
    "BRIDGE_STEP_SAFETY",
    "BRIDGE_EPS",
    "BRIDGE_SEED",
    "BRIDGE_PARAMETERS",
    "SEM108_PARAMETERS",
    "CANDIDATE_PARAMETERS",
    "frame_initialization",
    "soft_threshold",
    "LatentDictionaryBridge",
    "LatentBridgeSEM108",
    "build_latent_bridge_model",
    "bridge_parameter_audit",
    "bridge_code_stats",
    "bridge_reconstruction_diagnostic",
    "official_test_blocker",
]
