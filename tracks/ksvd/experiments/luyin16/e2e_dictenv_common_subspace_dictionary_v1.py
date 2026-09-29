"""E2E-DictEnv-Common-Subspace-Dictionary-v1 — CSSD core module.

Round ``e2e_dictenv_common_subspace_dictionary_v1`` (Workstream Z, ZINC).
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_common_subspace_dictionary_v1_preregistration.md``.

Question.  The closed ``e2e_dictenv_dictionary_coder_audit_v1`` round showed
that the reused FINAL-CLEAN sparse dictionaries spend three of their eight
activations on a near-duplicate DC/scale triplet (atoms 6/24/27, activation
rate ~1.0, aligned with the train phi mean) plus a PC1-like atom (23).  This
round tests whether **explicitly separating the dataset-common structural
subspace** from the patch-specific residual makes the remaining sparse
dictionary a less redundant, more structurally specialised and more
cross-graph reusable vocabulary.

The module reuses, instead of duplicating:

* the raw phi65 cache / split pipeline (``zinc_e2e_dictenv_p1``) and the C6
  mask and binding operators (``e2e_dictenv_clean_mechanism_v1``);
* the frozen tied-IHT coder / column-normalised dictionary
  (``e2e_dictenv_v0``);
* the audit metrics (usage, specialisation, reconstruction, recoverability,
  stop gradients) from ``e2e_dictenv_dictionary_coder_audit_v1``;
* the frozen training loop shape from ``e2e_dictenv_h1_clarity_audit`` (a
  faithful copy with an optional epoch callback, proven bit-equivalent by a
  focused test).

CPU only.  ``official_test_loaded = False`` in every payload.  The module
imports nothing that reads the official test split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_dictionary_coder_audit_v1 as dca
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

PROTOCOL_VERSION = "e2e_dictenv_common_subspace_dictionary_v1"

PHI_DIM = int(dca.PHI_DIM)          # 65
K_ATOMS = int(dca.K_ATOMS)          # 32
SPARSITY = int(dca.SPARSITY)        # 8
IHT_STEPS = int(v0.IHT_STEPS)       # 10
EPS = float(v0.EPS)

#: the round's candidate tag / arm (only one candidate is ever trained).
CSSD_TAG = "CSSD"
CSSD_SPEC = cm.CleanMechSpec(CSSD_TAG, "C6", coding="sparse")
CSSD_MASK = cm.C6_MASK

#: frozen reference dictionary indices.
DC_TRIPLET: tuple[int, ...] = (6, 24, 27)
PC1_LIKE_ATOM = 23

# ---------------------------------------------------------------------------
# frozen reference values (pre-registration section 0.1)
# ---------------------------------------------------------------------------

REF_NEFF_VALID = 14.520461423605052
REF_TOP5_VALID = 4.795910410258632
REF_WEIGHTED_SPEC = 0.20673863977279258
REF_DC_COUNT_GT095 = 3
REF_MAX_ACTIVATION_RATE = 1.0
REF_SPARSE_SEED0_SOUP_MAE = 0.12849851670576026
REF_DENSE_SEED0_SOUP_MAE = 0.1255625270641758
REFERENCE_TOLERANCE = 1.0e-9

# ---------------------------------------------------------------------------
# frozen selection thresholds (pre-registration sections 4-5)
# ---------------------------------------------------------------------------

SELECT_SEEDS_REQUIRED = 2
COND_A_MAX_RATE = 0.95
COND_A_RAW_MIN_RATE = 0.99
COND_B_TOP5_DROP = 0.4
COND_C_CENTERED_KEEP = 0.5
COND_D2_ATOM23_RATIO = 0.6

#: mean-centring is a diagnostic variant only, never a candidate.
VARIANTS: tuple[str, ...] = ("RAW", "MC", "Q1", "Q2")
CANDIDATE_VARIANTS: tuple[str, ...] = ("Q1", "Q2")

# ---------------------------------------------------------------------------
# frozen epoch-40 gate constants (pre-registration section 9)
# ---------------------------------------------------------------------------

GATE_DC_MAX = 1
GATE_NEFF_GAIN = 1.0
GATE_SPEC_GAIN = 0.2
GATE_TOP5_DROP = 0.4
GATE_GRAD_MIN = 1.0e-8
GATE_COLUMN_MIN = 1.0e-8
CATASTROPHIC_MAE = 2.0

#: projection / dead-column fallback tolerance.
PROJECTION_EPS = 1.0e-8

# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


def cpu_only_guard(device: Any) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise RuntimeError(f"common-subspace dictionary round is CPU-only, got device={resolved}")


# ---------------------------------------------------------------------------
# Stage B — train-only common subspace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommonSubspace:
    """Train-only common structural subspace ``x = U c + r``.

    ``components`` is orthonormal ``[PHI_DIM, q]``; ``rms`` holds the train RMS
    of the raw common coordinates (the inverse scaling used downstream).
    """

    components: np.ndarray
    rms: np.ndarray
    kind: str = "q1"

    @property
    def q(self) -> int:
        return int(self.components.shape[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "q": self.q,
            "components": self.components.tolist(),
            "rms": self.rms.tolist(),
            "official_test_loaded": False,
        }


def build_common_subspace(train_phi: np.ndarray, q: int = 1, kind: str | None = None) -> CommonSubspace:
    """Fit the frozen common subspace on the train rows only.

    ``q=1``: ``u1 = mu / ||mu||`` (uncentred train mean direction).
    ``q=2``: extend q1 by the PC1 of the residual ``r1``, orthogonalised
    against ``u1``.  No other q is supported.
    """
    X = np.asarray(train_phi, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != PHI_DIM:
        raise ValueError(f"train phi must be [N, {PHI_DIM}], got {X.shape}")
    if int(q) not in (1, 2):
        raise ValueError(f"cssd supports q in {{1, 2}}, got {q}")
    mu = X.mean(axis=0)
    norm = float(np.linalg.norm(mu))
    if norm <= 0:
        raise ValueError("train mean direction is degenerate")
    u1 = mu / norm
    columns = [u1]
    if int(q) == 2:
        r1 = X - np.outer(X @ u1, u1)
        centred = r1 - r1.mean(axis=0)
        _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
        v2 = np.asarray(vt[0], dtype=np.float64)
        v2 = v2 - float(v2 @ u1) * u1
        v2Norm = float(np.linalg.norm(v2))
        if v2Norm <= 0:
            raise ValueError("second common direction is degenerate")
        columns.append(v2 / v2Norm)
    U = np.stack(columns, axis=1)
    c = X @ U
    rms = np.sqrt(np.mean(c**2, axis=0))
    rms = np.maximum(rms, 1.0e-12)
    kind = f"q{int(q)}" if kind is None else str(kind)
    return CommonSubspace(components=U, rms=rms, kind=kind)


def decompose(phi: np.ndarray, subspace: CommonSubspace) -> tuple[np.ndarray, np.ndarray]:
    """``(c, r)`` with ``x = U c + r`` and ``U^T r = 0`` (float64)."""
    X = np.asarray(phi, dtype=np.float64)
    c = X @ subspace.components
    r = X - c @ subspace.components.T
    return c, r


def scaled_common(c: np.ndarray, subspace: CommonSubspace) -> np.ndarray:
    """``c~ = c / max(s, eps)`` with the frozen train RMS."""
    return np.asarray(c, dtype=np.float64) / subspace.rms.reshape(1, -1)


def subspace_energy(
    phi: np.ndarray, subspace: CommonSubspace, mean_vector: np.ndarray | None = None
) -> dict[str, Any]:
    """Common / residual energy accounting (pre-registration section 2)."""
    X = np.asarray(phi, dtype=np.float64)
    c, r = decompose(X, subspace)
    total = float((X**2).sum())
    common = float(((c @ subspace.components.T) ** 2).sum())
    residual = float((r**2).sum())
    mu = X.mean(axis=0) if mean_vector is None else np.asarray(mean_vector, dtype=np.float64)
    centred = X - mu
    centred_total = float((centred**2).sum())
    centred_common = float((((centred @ subspace.components) @ subspace.components.T) ** 2).sum())
    centred_residual = float(((r - r.mean(axis=0)) ** 2).sum())
    return {
        "n_rows": int(X.shape[0]),
        "total_energy": total,
        "common_energy_fraction": common / (total + EPS),
        "residual_energy_fraction": residual / (total + EPS),
        "centered_common_fraction": centred_common / (centred_total + EPS),
        "centered_residual_fraction": centred_residual / (centred_total + EPS),
        "common_mean_norm": float(np.linalg.norm(c.mean(axis=0))),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# descriptor variants for the zero-training probe (Stage C)
# ---------------------------------------------------------------------------


def descriptor_variants(
    phi: np.ndarray,
    mean_vector: np.ndarray,
    subspace_q1: CommonSubspace,
    subspace_q2: CommonSubspace | None = None,
) -> dict[str, np.ndarray]:
    """The four frozen probe target spaces for one split (float64)."""
    X = np.asarray(phi, dtype=np.float64)
    mu = np.asarray(mean_vector, dtype=np.float64).reshape(-1)
    _c1, r1 = decompose(X, subspace_q1)
    variants: dict[str, np.ndarray] = {"RAW": X, "MC": X - mu, "Q1": r1}
    if subspace_q2 is None:
        variants["Q2"] = r1
    else:
        _c2, r2 = decompose(X, subspace_q2)
        variants["Q2"] = r2
    return variants


def _variant_energy(
    raw: np.ndarray, variant: np.ndarray, subspace: CommonSubspace, variant_name: str
) -> tuple[float, float]:
    total = float((raw**2).sum()) + EPS
    if variant_name == "RAW":
        return 0.0, 1.0
    if variant_name == "MC":
        return 0.0, float((variant**2).sum()) / total
    c = raw @ subspace.components
    common = float(((c @ subspace.components.T) ** 2).sum())
    return common / total, float((variant**2).sum()) / total


def probe_variant_metrics(
    *,
    seed: int,
    variant: str,
    split: str,
    raw: np.ndarray,
    target: np.ndarray,
    codes: np.ndarray,
    reference_codes: np.ndarray,
    Dbar: np.ndarray,
    subspace: CommonSubspace,
) -> dict[str, Any]:
    """Frozen zero-training metrics for one (seed, variant, split)."""
    frequencies = dca.activation_frequency(codes)
    concentration = dca.usage_concentration(frequencies)
    comparison = dca.support_agreement(codes, reference_codes) if variant != "RAW" else None
    coefficient = dca.coefficient_agreement(codes, reference_codes) if variant != "RAW" else None
    common_fraction, residual_fraction = _variant_energy(raw, target, subspace, variant)
    return {
        "seed": int(seed),
        "variant": str(variant),
        "split": str(split),
        "common_dim": int(subspace.q) if variant in ("Q1", "Q2") else 0,
        "common_energy_fraction": float(common_fraction),
        "residual_energy_fraction": float(residual_fraction),
        "reconstruction_error": float(
            dca.reconstruction_metrics(target, codes, Dbar)["recon_frobenius"]
        ),
        "recon_mean_row_squared": float(
            dca.reconstruction_metrics(target, codes, Dbar)["recon_mean_row_squared"]
        ),
        "active_atoms": int(concentration["active_atoms"]),
        "effective_atoms": float(concentration["effective_atoms"]),
        "top1_usage": float(concentration["top1_share"]),
        "top3_usage": float(concentration["top3_share"]),
        "top5_usage": float(concentration["top5_share"]),
        "top8_usage": float(concentration["top8_share"]),
        "max_activation_rate": float(concentration["max_activation_rate"]),
        "n_atoms_gt_090": int((frequencies > 0.90).sum()),
        "n_atoms_gt_095": int((frequencies > 0.95).sum()),
        "n_atoms_gt_099": int((frequencies > 0.99).sum()),
        "atom6_rate": float(frequencies[6]),
        "atom23_rate": float(frequencies[PC1_LIKE_ATOM]),
        "atom24_rate": float(frequencies[24]),
        "atom27_rate": float(frequencies[27]),
        "usage_entropy": float(concentration["usage_entropy"]),
        "gini": float(concentration["gini"]),
        "support_jaccard_vs_raw": (
            float(comparison["jaccard"]["mean"]) if comparison is not None else 1.0
        ),
        "code_cosine_vs_raw": (
            float(coefficient["cosine"]["mean"]) if coefficient is not None else 1.0
        ),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# Stage C conditions and selection
# ---------------------------------------------------------------------------


def condition_a(row: Mapping[str, Any], raw_row: Mapping[str, Any]) -> dict[str, Any]:
    """At least two of the DC triplet drop below 0.95 (RAW sanity >= 0.99)."""
    raw_rates = [float(raw_row[f"atom{atom}_rate"]) for atom in DC_TRIPLET]
    rates = [float(row[f"atom{atom}_rate"]) for atom in DC_TRIPLET]
    raw_ok = all(rate >= COND_A_RAW_MIN_RATE for rate in raw_rates)
    dropped = [int(rate < COND_A_MAX_RATE) for rate in rates]
    return {
        "raw_precondition": bool(raw_ok),
        "raw_rates": raw_rates,
        "rates": rates,
        "n_below": int(sum(dropped)),
        "satisfied": bool(raw_ok and sum(dropped) >= 2),
    }


def condition_b(row: Mapping[str, Any], raw_row: Mapping[str, Any]) -> dict[str, Any]:
    drop = float(raw_row["top5_usage"]) - float(row["top5_usage"])
    return {"top5_drop": float(drop), "satisfied": bool(drop >= COND_B_TOP5_DROP)}


def condition_c(energy: Mapping[str, Any]) -> dict[str, Any]:
    keep = float(energy["centered_residual_fraction"])
    return {"centered_residual_fraction": keep, "satisfied": bool(keep >= COND_C_CENTERED_KEEP)}


def condition_d2(row: Mapping[str, Any], raw_row: Mapping[str, Any]) -> dict[str, Any]:
    raw_rate = float(raw_row["atom23_rate"])
    rate = float(row["atom23_rate"])
    ratio = rate / raw_rate if raw_rate > 0 else float("inf")
    return {"atom23_ratio": float(ratio), "satisfied": bool(ratio <= COND_D2_ATOM23_RATIO)}


def selection_rule(
    *,
    probe: Mapping[int, Mapping[str, Mapping[str, Any]]],
    energy: Mapping[int, Mapping[str, Mapping[str, Any]]],
    split: str = "valid",
) -> dict[str, Any]:
    """Frozen q1/q2 selection rule (pre-registration section 5).

    ``probe[seed][variant][split]`` and ``energy[seed][variant][split]``.
    """
    per_seed: dict[str, dict[str, Any]] = {}
    q1_pass: list[int] = []
    q2_pass: list[int] = []
    for seed, variants in sorted(probe.items()):
        raw = variants["RAW"][split]
        entry: dict[str, Any] = {"seed": int(seed)}
        for variant in CANDIDATE_VARIANTS:
            row = variants[variant][split]
            a = condition_a(row, raw)
            b = condition_b(row, raw)
            c = condition_c(energy[seed][variant][split]) if variant == "Q1" else None
            d2 = condition_d2(row, raw) if variant == "Q2" else None
            if variant == "Q1":
                passed = bool(a["satisfied"] and b["satisfied"] and c["satisfied"])
            else:
                passed = bool(a["satisfied"] and b["satisfied"] and d2["satisfied"])
            entry[variant] = {
                "condition_a": a,
                "condition_b": b,
                "condition_c": c,
                "condition_d2": d2,
                "passed": passed,
            }
            if passed:
                (q1_pass if variant == "Q1" else q2_pass).append(int(seed))
        per_seed[str(int(seed))] = entry
    if len(q1_pass) >= SELECT_SEEDS_REQUIRED:
        selected = "q1"
    elif len(q2_pass) >= SELECT_SEEDS_REQUIRED:
        selected = "q2"
    else:
        selected = None
    return {
        "split": str(split),
        "seeds_required": SELECT_SEEDS_REQUIRED,
        "q1_passed_seeds": q1_pass,
        "q2_passed_seeds": q2_pass,
        "selected": selected,
        "no_candidate": selected is None,
        "per_seed": per_seed,
        "thresholds": {
            "cond_a_max_rate": COND_A_MAX_RATE,
            "cond_a_raw_min_rate": COND_A_RAW_MIN_RATE,
            "cond_b_top5_drop": COND_B_TOP5_DROP,
            "cond_c_centered_keep": COND_C_CENTERED_KEEP,
            "cond_d2_atom23_ratio": COND_D2_ATOM23_RATIO,
        },
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# Stage D — the CSSD model
# ---------------------------------------------------------------------------


class CSSDModel(cm.CleanMechModel):
    """FINAL-CLEAN with an explicit common coordinate and residual dictionary.

    * every forward projects the raw dictionary parameter onto the residual
      subspace and column-normalises it (``U^T Dbar_perp = 0``);
    * IHT-10 codes the residual ``r = (I - U U^T) phi``;
    * the structural coordinate is ``z = [c~; alpha_res]`` of width ``32+q``;
    * with ``common_dim == 0`` the model is *not* a RAW fallback (the RAW path
      is the untouched ``CleanMechModel``).
    """

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        subspace: CommonSubspace,
        node_binding: str = "paired",
        edge_binding: str = "paired",
        coding: str = "sparse",
    ) -> None:
        super().__init__(
            config, dictionary, node_binding=node_binding, edge_binding=edge_binding, coding=coding
        )
        self.common_dim = int(subspace.q)
        if self.common_dim not in (1, 2):
            raise ValueError(f"common_dim must be 1 or 2, got {self.common_dim}")
        self.register_buffer(
            "U", torch.as_tensor(np.asarray(subspace.components, dtype=np.float32))
        )
        self.register_buffer(
            "common_rms", torch.as_tensor(np.asarray(subspace.rms, dtype=np.float32))
        )
        self.dead_column_fallbacks = self._widen_binding_(self.common_dim)
        self.dead_column_fallbacks += self._ensure_live_projection_()

    # -- widening (frozen deterministic initialization) -----------------------

    def _widen_binding_(self, q: int) -> int:
        """Insert the common-coordinate rows (zero init) into both bindings."""
        with torch.no_grad():
            old_a = self.W_A_S.detach().clone()          # [32, 48]
            new_a = torch.zeros(q + K_ATOMS, old_a.shape[1], dtype=old_a.dtype)
            new_a[q:] = old_a
            self.W_A_S = torch.nn.Parameter(new_a)

            old_e = self.W_E_S.detach().clone()          # [3*32, 48]
            d_e = old_e.shape[1]
            new_e = torch.zeros(3 * (q + K_ATOMS), d_e, dtype=old_e.dtype)
            for block in range(3):
                new_e[block * (q + K_ATOMS) + q : block * (q + K_ATOMS) + q + K_ATOMS] = (
                    old_e[block * K_ATOMS : (block + 1) * K_ATOMS]
                )
            self.W_E_S = torch.nn.Parameter(new_e)
        return 0

    def _ensure_live_projection_(self) -> int:
        """Deterministic dead-column fallback at initialization (pre-registered)."""
        D, count = dead_column_fallback(self.D.detach(), self.U.detach())
        if count:
            with torch.no_grad():
                self.D.copy_(D)
        return int(count)

    # -- frozen operators ------------------------------------------------------

    def residual_dictionary(self) -> torch.Tensor:
        """``Dbar_perp = colnorm((I - U U^T) D_raw)`` (hard constraint)."""
        U = self.U.to(dtype=self.D.dtype)
        D_perp = self.D - U @ (U.t() @ self.D)
        return v0.normalized_dictionary(D_perp)

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        """``z = [c~; IHT10(Dbar_perp, r)]`` — width ``32 + q``."""
        U = self.U.to(dtype=phi.dtype)
        c = phi @ U
        r = phi - c @ U.t()
        alpha = v0.tied_iht_codes(
            self.residual_dictionary(), r, s=SPARSITY, steps=IHT_STEPS
        )
        scale = self.common_rms.to(dtype=phi.dtype)
        return torch.cat([c / scale, alpha], dim=1)

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """Residual reconstruction ``r_hat = alpha_res Dbar_perp^T``."""
        alpha = coord[:, self.common_dim :]
        return alpha @ self.residual_dictionary().t()

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """Frozen form with the residual numerator and the raw phi denominator."""
        U = self.U.to(dtype=phi.dtype)
        r = phi - (phi @ U) @ U.t()
        phi_hat = self.reconstruct(phi, coord)
        numerator = ((r - phi_hat) ** 2).sum(dim=1)
        denominator = (phi**2).sum(dim=1) + EPS
        return (numerator / denominator).mean()


def dead_column_fallback(D: torch.Tensor, U: torch.Tensor, eps: float = PROJECTION_EPS):
    """Deterministic initialization-time replacement of dead projected columns.

    Returns ``(D_out, count)``.  A column ``j`` whose projected norm is
    ``<= eps * ||d_j||`` is replaced by ``(I - U U^T) d_{j*}`` with
    ``j* = argmax_k ||(I - U U^T) d_k||``; if even that vector is degenerate,
    the first canonical basis direction with a non-zero residual component is
    used.  No runtime path ever mutates the dictionary this way.
    """
    D = torch.as_tensor(D, dtype=torch.float64).clone()
    U = torch.as_tensor(U, dtype=torch.float64)
    Dp = D - U @ (U.t() @ D)
    norms = torch.linalg.norm(Dp, dim=0)
    scale = torch.linalg.norm(D, dim=0).clamp_min(eps)
    dead = norms <= eps * scale
    count = int(dead.sum())
    if not count:
        return D.to(torch.float32), 0
    jstar = int(torch.argmax(norms))
    replacement = Dp[:, jstar].clone()
    if float(norms[jstar]) <= eps:
        replacement = None
        for k in range(D.shape[0]):
            candidate = torch.zeros(D.shape[0], dtype=D.dtype)
            candidate[k] = 1.0
            candidate = candidate - U @ (U.t() @ candidate)
            if float(candidate.norm()) > eps:
                replacement = candidate
                break
        if replacement is None:  # pragma: no cover - orthonormal U cannot fill R^65
            raise RuntimeError("no residual direction available for the dead-column fallback")
    D[:, dead] = replacement.unsqueeze(1)
    return D.to(torch.float32), count


def build_cssd_model(
    dictionary: np.ndarray,
    seed: int,
    subspace: CommonSubspace,
    spec: cm.CleanMechSpec | None = None,
) -> CSSDModel:
    """Build the CSSD model with the exact FINAL-CLEAN RNG stream."""
    spec = CSSD_SPEC if spec is None else spec
    torch.manual_seed(int(seed))
    return CSSDModel(
        cm.H1_CONFIG,
        dictionary,
        subspace=subspace,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
    )


# ---------------------------------------------------------------------------
# residual codes and Stage G / epoch-40 metrics
# ---------------------------------------------------------------------------


def residual_codes(
    model: CSSDModel, phi: np.ndarray, chunk: int = 65536, device: torch.device | None = None
) -> np.ndarray:
    """Chunked ``alpha_res = IHT10(Dbar_perp, r)`` on a raw phi65 matrix."""
    import torch as _torch

    target = _torch.device("cpu") if device is None else device
    cpu_only_guard(target)
    model.eval()
    X = np.asarray(phi, dtype=np.float32)
    outputs: list[np.ndarray] = []
    with _torch.no_grad():
        for start in range(0, X.shape[0], int(chunk)):
            current = _torch.as_tensor(X[start : start + int(chunk)], dtype=_torch.float32)
            U = model.U.to(dtype=current.dtype)
            c = current @ U
            r = current - c @ U.t()
            alpha = v0.tied_iht_codes(
                model.residual_dictionary(), r, s=SPARSITY, steps=IHT_STEPS
            )
            outputs.append(alpha.cpu().numpy().astype(np.float64))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, K_ATOMS))


def raw_codes(phi: np.ndarray, D: np.ndarray) -> np.ndarray:
    """The frozen RAW IHT-10 codes on the untouched dictionary."""
    Dbar = dca.effective_dictionary(D)
    Dbar_torch = dca.effective_dictionary_torch(D)
    return dca.codes_for("iht10", np.asarray(phi), Dbar, Dbar_torch=Dbar_torch)


def usage_payload(codes: np.ndarray) -> dict[str, Any]:
    """Concentration metrics with the audit definitions (frequencies included)."""
    frequencies = dca.activation_frequency(codes)
    payload = dca.usage_concentration(frequencies)
    payload["frequencies"] = frequencies.tolist()
    return payload


def atom_coherence(Dbar: np.ndarray) -> dict[str, float]:
    """Off-diagonal ``|Dbar^T Dbar|`` distribution (diagnostic only)."""
    matrix = np.asarray(Dbar, dtype=np.float64)
    gram = np.abs(matrix.T @ matrix)
    n = gram.shape[0]
    off = gram[~np.eye(n, dtype=bool)]
    return {
        "mean": float(off.mean()),
        "median": float(np.median(off)),
        "p90": float(np.quantile(off, 0.90)),
        "max": float(off.max()),
    }


def graph_coverage(
    codes: np.ndarray, graph_ids: np.ndarray, n_graphs: int, top_fraction: float = 0.10
) -> dict[str, list[float]]:
    """Per-atom cross-graph reuse statistics (pre-registration section 10)."""
    codes = np.asarray(codes, dtype=np.float64)
    graph_ids = np.asarray(graph_ids, dtype=np.int64)
    if codes.shape[0] != graph_ids.shape[0]:
        raise ValueError("codes and graph ids must share the row dimension")
    support = np.abs(codes) > 0.0
    n_atoms = int(codes.shape[1])
    coverage: list[float] = []
    acts_per_active: list[float] = []
    mean_active: list[float] = []
    median_active: list[float] = []
    top10_share: list[float] = []
    n_active_graphs: list[int] = []
    for atom in range(n_atoms):
        rows = graph_ids[support[:, atom]]
        counts = np.bincount(rows, minlength=int(n_graphs))
        active = counts > 0
        n_active = int(active.sum())
        coverage.append(float(n_active / max(int(n_graphs), 1)))
        n_active_graphs.append(n_active)
        if n_active:
            values = counts[active].astype(np.float64)
            mean_active.append(float(values.mean()))
            median_active.append(float(np.median(values)))
            acts_per_active.append(float(values.mean()))
            k = max(1, int(math.ceil(float(top_fraction) * n_active)))
            order = np.sort(values)[::-1]
            top10_share.append(float(order[:k].sum() / values.sum()))
        else:
            mean_active.append(0.0)
            median_active.append(0.0)
            acts_per_active.append(0.0)
            top10_share.append(0.0)
    return {
        "coverage": coverage,
        "n_active_graphs": [float(value) for value in n_active_graphs],
        "mean_activations_per_active_graph": mean_active,
        "median_activations_per_active_graph": median_active,
        "top10_graph_share": top10_share,
    }


# ---------------------------------------------------------------------------
# epoch-40 gate
# ---------------------------------------------------------------------------


def epoch40_gate(
    *,
    dc_count_gt095: int,
    top5_share: float,
    effective_atoms: float,
    usage_weighted_spec: float,
    gradient_norm_D: float,
    column_norm_min: float,
) -> dict[str, Any]:
    """Frozen continuation gate (pre-registration section 9)."""
    condition_a1 = int(dc_count_gt095) <= GATE_DC_MAX
    top5_drop = float(REF_TOP5_VALID) - float(top5_share)
    condition_a2 = bool(top5_drop >= GATE_TOP5_DROP)
    condition_a = bool(condition_a1 or condition_a2)
    neff_gain = float(effective_atoms) - float(REF_NEFF_VALID)
    condition_b1 = bool(neff_gain >= GATE_NEFF_GAIN)
    spec_ratio = float(usage_weighted_spec) / float(REF_WEIGHTED_SPEC)
    condition_b2 = bool(spec_ratio >= 1.0 + GATE_SPEC_GAIN)
    condition_b = bool(condition_b1 or condition_b2)
    condition_c = bool(
        np.isfinite(gradient_norm_D)
        and float(gradient_norm_D) > GATE_GRAD_MIN
        and np.isfinite(column_norm_min)
        and float(column_norm_min) > GATE_COLUMN_MIN
    )
    passed = bool(condition_a and condition_b and condition_c)
    return {
        "condition_a": {
            "dc_count_gt095": int(dc_count_gt095),
            "dc_count_max": GATE_DC_MAX,
            "dc_satisfied": bool(condition_a1),
            "top5_drop_vs_raw": float(top5_drop),
            "top5_drop_required": GATE_TOP5_DROP,
            "top5_satisfied": bool(condition_a2),
            "satisfied": condition_a,
        },
        "condition_b": {
            "effective_atoms": float(effective_atoms),
            "reference_effective_atoms": float(REF_NEFF_VALID),
            "neff_gain": float(neff_gain),
            "neff_gain_required": GATE_NEFF_GAIN,
            "neff_satisfied": bool(condition_b1),
            "usage_weighted_spec": float(usage_weighted_spec),
            "reference_usage_weighted_spec": float(REF_WEIGHTED_SPEC),
            "spec_ratio": float(spec_ratio),
            "spec_ratio_required": 1.0 + GATE_SPEC_GAIN,
            "spec_satisfied": bool(condition_b2),
            "satisfied": condition_b,
        },
        "condition_c": {
            "gradient_norm_D": float(gradient_norm_D),
            "gradient_min": GATE_GRAD_MIN,
            "column_norm_min": float(column_norm_min),
            "column_min": GATE_COLUMN_MIN,
            "satisfied": condition_c,
        },
        "passed": passed,
        "verdict": "CSSD_CONTINUE" if passed else "CSSD_REPRESENTATION_GATE_FAIL",
        "official_test_loaded": False,
    }


def catastrophic_check(train_mae: float, valid_mae: float, rec_loss: float) -> dict[str, Any]:
    finite = bool(np.isfinite(train_mae) and np.isfinite(valid_mae) and np.isfinite(rec_loss))
    triggered = bool(
        (not finite) or float(train_mae) > CATASTROPHIC_MAE or float(valid_mae) > CATASTROPHIC_MAE
    )
    return {
        "finite": finite,
        "train_mae": float(train_mae),
        "valid_mae": float(valid_mae),
        "rec_loss": float(rec_loss),
        "threshold": CATASTROPHIC_MAE,
        "triggered": triggered,
    }


# ---------------------------------------------------------------------------
# training loop (faithful copy of ``audit.train_cpu`` + epoch callback)
# ---------------------------------------------------------------------------


def train_cssd(
    *,
    tag: str,
    epochs: int,
    threads: int,
    out_dir: Any,
    subspace: CommonSubspace,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    seed: int = 0,
    callback: Callable[[int, CSSDModel, Any, list[dict[str, Any]]], tuple[bool, Mapping[str, Any] | None]]
    | None = None,
    model_factory: Callable[[np.ndarray, int, CommonSubspace], CSSDModel] | None = None,
    save_states: bool = True,
    log: bool = True,
) -> dict[str, Any]:
    """One CPU CSSD run; identical to ``audit.train_cpu`` when ``callback=None``.

    The only additions over the frozen loop are the optional epoch callback
    (evaluated after the epoch's valid pass, before the next epoch) and the
    early stop it can request, plus an optional ``model_factory`` hook so a
    subclass with an extra capacity module can reuse the identical training
    path (``None`` keeps the frozen ``build_cssd_model``).  Data order, RNG use,
    loss, optimizer, clipping and the Top-5 soup are unchanged.
    """
    import time
    from pathlib import Path

    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    device = audit.attach_cpu(int(threads))
    if device.type != "cpu":
        raise RuntimeError("cssd training is CPU-only")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary, dict_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    p2run._seed_everything(int(seed))
    factory = build_cssd_model if model_factory is None else model_factory
    model = factory(dictionary, int(seed), subspace)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY)
    )
    loader = p1.make_env_loader(
        train_data, int(p2run.BATCH_SIZE), True, int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    eval_loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, int(seed) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    mask = cm.arm_mask(CSSD_SPEC)
    lam = float(cm.H1_LAMBDA)
    soup_k = int(p2run.SOUP_K)
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    callback_payloads: dict[int, Mapping[str, Any]] = {}
    curve: list[dict[str, Any]] = []
    stopped_at: int | None = None
    rss_start = audit._rss_mb()
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        task_sum = rec_sum = rec_term_sum = 0.0
        n_mol = n_nodes = n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            rec_term_sum += float(rec.detach())
            n_batches += 1
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi**2).sum(dim=1) + v0.EPS)).sum())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        train_rec_term = float(rec_term_sum / max(n_batches, 1))
        valid = audit._evaluate_model(model, eval_loader, device, mask)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "train_rec_term": train_rec_term,
                "valid_mae": float(valid["mae"]),
                "d_norm": float(model.D.detach().norm()),
            }
        )
        epoch_states[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:soup_k]}
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 10 == 0 or epoch == int(epochs)):
            print(
                f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} rec={train_rec_term:.3e} "
                f"rec_full={train_rec:.3e} valid={float(valid['mae']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
        if callback is not None:
            keep_going, payload = callback(int(epoch), model, optimizer, curve)
            if payload is not None:
                callback_payloads[int(epoch)] = payload
            if not keep_going:
                stopped_at = int(epoch)
                print(f"[{tag}] stopped at epoch {epoch} by callback", flush=True)
                break
    wall = float(time.perf_counter() - started)
    assert best_state is not None
    members = sorted(
        int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:soup_k]
    )
    soup_state = {
        k: torch.stack([epoch_states[e][k].float() for e in members]).mean(0)
        for k in epoch_states[members[0]]
    }
    soup_model = factory(dictionary, int(seed), subspace)
    soup_model.load_state_dict(soup_state)
    soup_valid = audit._evaluate_model(soup_model, eval_loader, device, mask)
    projected = np.asarray(model.D.detach().cpu(), dtype=np.float64)
    U = subspace.components
    projected = projected - U @ (U.T @ projected)
    projected_norms = np.linalg.norm(projected, axis=0)
    Dbar_perp = dca.effective_dictionary(projected)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "device": "cpu",
        "threads": int(threads),
        "seed": int(seed),
        "mask": mask.as_dict() if mask is not None else None,
        "config": cm.H1_CONFIG.as_dict(),
        "common_dim": int(subspace.q),
        "subspace_kind": subspace.kind,
        "common_rms": subspace.rms.tolist(),
        "dictionary_sha256": dict_sha,
        "projected_column_norm_min": float(projected_norms.min()),
        "projected_column_norm_max": float(projected_norms.max()),
        "U_T_Dbar_perp_max_abs": float(np.abs(U.T @ Dbar_perp).max()),
        "dead_column_fallbacks": int(model.dead_column_fallbacks),
        "lambda_rec": lam,
        "epoch_budget": int(epochs),
        "epochs_run": int(len(curve)),
        "stopped_at_epoch": stopped_at,
        "completed": bool(stopped_at is None and len(curve) == int(epochs)),
        "parameters": p2.total_parameter_count(cm.H1_CONFIG),
        "actual_params": int(sum(p.numel() for p in model.parameters())),
        "curve": curve,
        "callback_payloads": {str(k): dict(v) for k, v in callback_payloads.items()},
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(curve[-1]["valid_mae"]),
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "soup": {
            "members": members,
            "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members],
            "soup_valid_mae": float(soup_valid["mae"]),
        },
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        "rss_start_mb": float(rss_start),
        "rss_end_mb": float(audit._rss_mb()),
        "peak_rss_mb": audit._peak_rss_mb(),
        "official_test_loaded": False,
    }
    if save_states:
        torch.save(best_state, out_dir / f"{tag}_raw_state.pt")
        torch.save(soup_state, out_dir / f"{tag}_soup_state.pt")
        torch.save(
            {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
            out_dir / f"{tag}_final_state.pt",
        )
    payload["best_state_sha256"] = audit.state_sha256(best_state)
    payload["soup_state_sha256"] = audit.state_sha256(soup_state)
    return payload


def _current_dictionary(model: CSSDModel) -> np.ndarray:
    return np.asarray(model.D.detach().cpu(), dtype=np.float32)


__all__ = [
    "PROTOCOL_VERSION",
    "PHI_DIM",
    "K_ATOMS",
    "SPARSITY",
    "IHT_STEPS",
    "EPS",
    "CSSD_TAG",
    "CSSD_SPEC",
    "CSSD_MASK",
    "DC_TRIPLET",
    "PC1_LIKE_ATOM",
    "REF_NEFF_VALID",
    "REF_TOP5_VALID",
    "REF_WEIGHTED_SPEC",
    "REF_DC_COUNT_GT095",
    "REF_MAX_ACTIVATION_RATE",
    "REF_SPARSE_SEED0_SOUP_MAE",
    "REF_DENSE_SEED0_SOUP_MAE",
    "REFERENCE_TOLERANCE",
    "SELECT_SEEDS_REQUIRED",
    "COND_A_MAX_RATE",
    "COND_A_RAW_MIN_RATE",
    "COND_B_TOP5_DROP",
    "COND_C_CENTERED_KEEP",
    "COND_D2_ATOM23_RATIO",
    "VARIANTS",
    "CANDIDATE_VARIANTS",
    "GATE_DC_MAX",
    "GATE_NEFF_GAIN",
    "GATE_SPEC_GAIN",
    "GATE_TOP5_DROP",
    "GATE_GRAD_MIN",
    "GATE_COLUMN_MIN",
    "CATASTROPHIC_MAE",
    "PROJECTION_EPS",
    "official_test_blocker",
    "cpu_only_guard",
    "CommonSubspace",
    "build_common_subspace",
    "decompose",
    "scaled_common",
    "subspace_energy",
    "descriptor_variants",
    "probe_variant_metrics",
    "condition_a",
    "condition_b",
    "condition_c",
    "condition_d2",
    "selection_rule",
    "CSSDModel",
    "dead_column_fallback",
    "build_cssd_model",
    "residual_codes",
    "raw_codes",
    "usage_payload",
    "atom_coherence",
    "graph_coverage",
    "epoch40_gate",
    "catastrophic_check",
    "train_cssd",
]
