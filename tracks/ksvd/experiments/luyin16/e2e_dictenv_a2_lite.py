"""E2E-DictEnv-A2-Lite — minimum paired frozen-OMP screen (compute amendment).

This module is *not* part of the frozen ``E2E-DictEnv-A2`` preregistration
(commit ``1813f53``); it implements the user-requested **compute-budget
amendment** recorded in ``tracks/ksvd/notes/e2e_dictenv_a2_compute_amendment.md``.

The amendment truncates the original A2 execution plan and answers exactly one
question with the minimum matched comparison:

    Do the frozen exact-OMP codes carry a *task* advantage for the real
    structure<->attribute pairing (``REAL``) over the assignment-independent
    control (``INDEP``) at the frozen H1 protocol?

Everything reused is reused by *call*, never re-derived: the frozen A1/A2 code,
the 433-D coordinates, the scaler, the K32/s8 dictionaries, the exact-OMP
codes, the trainer, and ``sdb_v0.fit_pca_rank`` for the optional dense control.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2 as a2

ROUND = "E2E-DictEnv-A2-Lite"
PROTOCOL_VERSION = "e2e_dictenv_a2_lite"
SUBTITLE = (
    "Minimum paired frozen-OMP screen (REAL vs INDEP) under the user "
    "compute-budget amendment"
)
STUDY = a2.STUDY
PARENT_ROUND = a2.ROUND
PARENT_PROTOCOL_VERSION = a2.PROTOCOL_VERSION

#: the exact parent revision this lite continuation may attach to (the A2
#: artifacts referenced by ``verify`` were produced by this commit).
PARENT_A2_COMMIT = "24d528635fab49c082115d90592cb7d5938eeb37"
PARENT_A2_PREREG_COMMIT = "1813f53"
AMENDMENT_NOTE = "tracks/ksvd/notes/e2e_dictenv_a2_compute_amendment.md"
#: names the A2 identity record must contain (and pass) before screening
REQUIRED_A2_IDENTITY_ENTRIES: tuple[str, ...] = (
    "cache_train_phi",
    "cache_valid_phi",
    "scaler_real",
    "scaler_indep",
    "dictionary_REAL",
    "dictionary_INDEP",
    "omp_REAL_train",
    "omp_REAL_valid",
    "omp_INDEP_train",
    "omp_INDEP_valid",
    "matched_init_REAL",
    "matched_init_INDEP",
)

#: the minimum question needs exactly these two matched arms (no TOPO).
SCREEN_ARMS: tuple[str, ...] = ("REAL", "INDEP")
SCREEN_LABEL: dict[str, str] = {"REAL": "R0", "INDEP": "I0"}
FORBIDDEN_ARMS: tuple[str, ...] = ("TOPO",)

#: frozen by the amendment, *before* the screen ran (no sweep, no tuning).
SCREEN_HORIZON = 160
STRONG_POSITIVE = 0.006
PCA_MATERIAL = float(a2.MATERIAL)  # 0.003, unchanged from the frozen round
PCA_RANK = int(a2.PCA_RANK)  # 32, unchanged from the frozen round
LATE_WINDOW_START = 121
LATE_WINDOW_END = SCREEN_HORIZON
LATE_POSITIVE_FRACTION = 0.75

VERDICT_WORTH_FULL_CONFIRMATION = "PAIRING_SIGNAL_WORTH_FULL_CONFIRMATION"
VERDICT_COMPRESSION_BOTTLENECK = "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK"
VERDICT_NO_SIGNAL = "NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D"
VERDICT_UNRESOLVED = "LITE_SCREEN_UNRESOLVED_PENDING_REPAIR"

VERDICTS: tuple[str, ...] = (
    VERDICT_WORTH_FULL_CONFIRMATION,
    VERDICT_COMPRESSION_BOTTLENECK,
    VERDICT_NO_SIGNAL,
    VERDICT_UNRESOLVED,
)

#: intermediate state: the sparse screen did not clear the strong bar, so the
#: amendment authorises the cheap dense rank-32 diagnostic (never a verdict).
STATE_PCA_AUTHORISED = "PCA_DIAGNOSTIC_AUTHORISED"

#: stages the amendment defers to a later user decision (never auto-started).
DEFERRED_STAGES: tuple[str, ...] = (
    "TOPO-OMP predictor baseline",
    "IHT-10/30/100/200 coder qualification",
    "task-coupled E2E sparse dictionary (E0/E1/E2)",
    "REAL->INDEP code-pairing-removal mechanism",
    "node-only / edge-only pairing interventions",
    "DenseTied specificity control",
    "seed 1 replication",
    "official test split",
    "K64 dictionary",
    "s12 dictionary",
    "any architecture / hyper-parameter sweep",
)

#: sha256 of the A2 producer modules that generated the referenced artifacts and
#: of the frozen A2 preregistration.  The lite round refuses to run if the live
#: files differ: the references must stay attached to the exact producing code.
A2_PRODUCER_SHA256: dict[str, str] = {
    "tracks/ksvd/experiments/luyin16/e2e_dictenv_a2.py":
        "f9862714161d936c637e74675addd781d32b277a5b4df09dda3d5ad872072fad",
    "tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_a2.py":
        "a35af037f02b16d11fb4ea3a6de0179d0e8537cf74756f7176487b816485ec4d",
    "tracks/ksvd/notes/e2e_dictenv_a2_preregistration.md":
        "ec875eb2bec925814a326b8e2efdc868939bf2c205a9a0680d532873a0157498",
}

CURVE_FIELDS: tuple[str, ...] = ("epoch", "valid_mae")


class CurveError(RuntimeError):
    """A training curve is missing, empty, malformed or non-finite."""


def read_curve(path: Path) -> dict[int, float]:
    """Read ``epoch -> valid_mae`` from a frozen ``train_arm`` curve CSV.

    Fail-closed: a missing, empty, malformed or non-finite curve raises, so no
    caller can silently treat unusable evidence as "no effect".
    """
    path = Path(path)
    if not path.exists():
        raise CurveError(f"curve missing: {path}")
    rows: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = [field.strip() for field in handle.readline().strip().split(",")]
        missing = [field for field in CURVE_FIELDS if field not in header]
        if missing:
            raise CurveError(f"curve {path} lacks fields {missing}")
        epoch_at = header.index("epoch")
        value_at = header.index("valid_mae")
        for line in handle:
            if not line.strip():
                continue
            fields = line.strip().split(",")
            if len(fields) <= max(epoch_at, value_at):
                raise CurveError(f"curve {path} has a truncated row: {line!r}")
            epoch = int(fields[epoch_at])
            value = float(fields[value_at])
            if not np.isfinite(value):
                raise CurveError(f"curve {path} has a non-finite valid_mae at epoch {epoch}")
            rows[epoch] = value
    if not rows:
        raise CurveError(f"curve {path} is empty")
    return rows


def curve_or_none(path: Path) -> dict[int, float] | None:
    """``read_curve`` without raising — ``None`` means "evidence unusable"."""
    try:
        return read_curve(path)
    except (CurveError, OSError, ValueError):
        return None


def late_direction(
    curve_indep: Mapping[int, float],
    curve_real: Mapping[int, float],
    *,
    start: int = LATE_WINDOW_START,
    end: int = LATE_WINDOW_END,
    min_fraction: float = LATE_POSITIVE_FRACTION,
) -> dict[str, Any]:
    """Paired late-window direction of ``INDEP - REAL`` (positive = REAL better).

    The screen is only "stable" when the late checkpoints agree in direction:
    a majority of the late epochs, their mean, and the best late epoch must all
    favour REAL.  Epochs are paired; a missing epoch in either curve is an error.
    """
    epochs = list(range(int(start), int(end) + 1))
    missing = [
        epoch
        for epoch in epochs
        if epoch not in curve_indep or epoch not in curve_real
    ]
    if missing:
        raise CurveError(f"late window epochs missing from a curve: {missing[:8]}...")
    deltas = [float(curve_indep[epoch]) - float(curve_real[epoch]) for epoch in epochs]
    if not all(np.isfinite(delta) for delta in deltas):
        raise CurveError("late window contains a non-finite paired delta")
    positive_fraction = float(np.mean([delta > 0.0 for delta in deltas]))
    best_delta = float(min(curve_indep[epoch] for epoch in epochs)) - float(
        min(curve_real[epoch] for epoch in epochs)
    )
    stable = bool(
        positive_fraction >= float(min_fraction)
        and float(np.mean(deltas)) > 0.0
        and best_delta > 0.0
    )
    return {
        "window": [int(start), int(end)],
        "n_epochs": len(epochs),
        "positive_fraction": positive_fraction,
        "min_positive_fraction": float(min_fraction),
        "mean_delta": float(np.mean(deltas)),
        "best_delta": best_delta,
        "delta_first": deltas[0],
        "delta_last": deltas[-1],
        "stable": stable,
    }


def paired_epochs(
    curve_indep: Mapping[int, float], curve_real: Mapping[int, float]
) -> int:
    """Number of epochs present in both curves (screen pairing check)."""
    return len(set(curve_indep) & set(curve_real))


def screen_decision(
    values: Mapping[str, float],
    *,
    direction: Mapping[str, Any] | None,
    curves_finite: bool,
    horizon: int = SCREEN_HORIZON,
    threshold: float = STRONG_POSITIVE,
) -> dict[str, Any]:
    """Frozen lite-screen decision (pure function).

    ``strong_positive`` requires the paired soup gap to clear ``threshold``
    *and* the late checkpoints to be directionally stable.  Anything else
    (including unusable curves) authorises only the cheap dense diagnostic —
    except unusable curves, which yield ``VERDICT_UNRESOLVED`` and stop.
    """
    g_pair = float(values["INDEP"]) - float(values["REAL"])
    stable = bool(direction is not None and direction.get("stable"))
    strong = bool(curves_finite and g_pair >= float(threshold) and stable)
    if not curves_finite:
        return {
            "G_pair_screen": g_pair,
            "strong_positive": False,
            "direction_stable": stable,
            "pca_authorised": False,
            "verdict": VERDICT_UNRESOLVED,
            "reason": "training curves missing or non-finite; screen unusable",
            "threshold": float(threshold),
            "horizon": int(horizon),
        }
    if strong:
        return {
            "G_pair_screen": g_pair,
            "strong_positive": True,
            "direction_stable": True,
            "pca_authorised": False,
            "verdict": VERDICT_WORTH_FULL_CONFIRMATION,
            "reason": (
                f"G_pair_screen {g_pair:.6f} >= {float(threshold):.3f} with a stable "
                "late-window direction; full confirmation is a later user decision"
            ),
            "threshold": float(threshold),
            "horizon": int(horizon),
        }
    return {
        "G_pair_screen": g_pair,
        "strong_positive": False,
        "direction_stable": stable,
        "pca_authorised": True,
        "verdict": STATE_PCA_AUTHORISED,
        "reason": (
            f"G_pair_screen {g_pair:.6f} < {float(threshold):.3f} "
            f"(direction_stable={stable}); cheap dense rank-{PCA_RANK} diagnostic authorised"
        ),
        "threshold": float(threshold),
        "horizon": int(horizon),
    }


def pca_label(
    g_pair_pca: float,
    *,
    curves_finite: bool,
    threshold: float = PCA_MATERIAL,
) -> dict[str, Any]:
    """Frozen dense-control decision (pure function)."""
    if not curves_finite:
        return {
            "G_pair_PCA": float(g_pair_pca),
            "primary_pass": False,
            "verdict": VERDICT_UNRESOLVED,
            "reason": "dense control curves missing or non-finite; diagnosis unusable",
            "threshold": float(threshold),
        }
    if float(g_pair_pca) >= float(threshold):
        return {
            "G_pair_PCA": float(g_pair_pca),
            "primary_pass": True,
            "verdict": VERDICT_COMPRESSION_BOTTLENECK,
            "reason": (
                f"sparse screen weak but G_pair_PCA {float(g_pair_pca):.6f} >= "
                f"{float(threshold):.3f}: the pairing is expressible at rank {PCA_RANK}"
            ),
            "threshold": float(threshold),
        }
    return {
        "G_pair_PCA": float(g_pair_pca),
        "primary_pass": False,
        "verdict": VERDICT_NO_SIGNAL,
        "reason": (
            f"G_pair_PCA {float(g_pair_pca):.6f} < {float(threshold):.3f}: no attributed "
            f"code-formation signal at rank {PCA_RANK}"
        ),
        "threshold": float(threshold),
    }


def deferred_stage_record() -> dict[str, str]:
    """The amendment's deferral record (no stage is cancelled permanently)."""
    return {name: "DEFERRED_PENDING_USER_AUTHORIZATION" for name in DEFERRED_STAGES}


def screen_arm_list() -> Sequence[str]:
    """Guard: the lite round may only ever run the two minimum arms."""
    for arm in SCREEN_ARMS:
        if arm in FORBIDDEN_ARMS:
            raise RuntimeError(f"lite screen must not include the {arm} arm")
    return SCREEN_ARMS
