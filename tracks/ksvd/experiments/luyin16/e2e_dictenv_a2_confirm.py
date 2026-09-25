"""E2E-DictEnv-A2-Confirm core — frozen 320-epoch exact-OMP pairing confirmation.

Round ``E2E-DictEnv-A2-Confirm`` / protocol ``e2e_dictenv_a2_confirm``.
Preregistration: ``tracks/ksvd/notes/e2e_dictenv_a2_confirm_preregistration.md``
(written and committed *before* any confirmation training).

This module holds only frozen constants and pure decision functions; the runner
(``zinc_e2e_dictenv_a2_confirm``) drives the unchanged frozen A1 trainer.  The
round answers one question with the parent protocol's own horizon (320 epochs):

    under the frozen exact-OMP code, is ``ATTR-REAL`` better than ``ATTR-INDEP``,
    and is ``REAL`` competitive with the completed ``TOPO`` baseline?

Nothing is re-derived: the 433-D objects, train-only scaler, K-SVD ``K=32``/
``s=8`` dictionaries, exact-OMP codes, the matched H1 model/trainer and the
matched initialisation are the identity-verified parent A2 artifacts.  The
``TOPO-OMP`` 320-epoch arm is a completed parent result and is never retrained.
Official ZINC **test is never loaded**.  All CUDA work is physical **GPU1 only**.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2 as a2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2_lite as a2lite

ROUND = "E2E-DictEnv-A2-Confirm"
PROTOCOL_VERSION = "e2e_dictenv_a2_confirm"
SUBTITLE = (
    "Frozen exact-OMP 320-epoch pairing confirmation (REAL vs INDEP) with a "
    "practical TOPO comparison"
)
STUDY = a2.STUDY
PARENT_ROUND = a2.ROUND
PARENT_PROTOCOL_VERSION = a2.PROTOCOL_VERSION
PREDECESSOR_ROUND = a2lite.ROUND
PREDECESSOR_PROTOCOL_VERSION = a2lite.PROTOCOL_VERSION

#: notes/files frozen by digest before the round ran
PREREGISTRATION_NOTE = "tracks/ksvd/notes/e2e_dictenv_a2_confirm_preregistration.md"

#: the exact parent revision the reused artifacts were produced by
PARENT_A2_COMMIT = a2lite.PARENT_A2_COMMIT
PARENT_A2_PREREG_COMMIT = a2lite.PARENT_A2_PREREG_COMMIT
#: producer modules and the frozen parent preregistration that must stay
#: byte-identical while the referenced artifacts are reused
A2_PRODUCER_SHA256: dict[str, str] = dict(a2lite.A2_PRODUCER_SHA256)
#: A2 identity entries this round depends on (they must all have passed)
REQUIRED_A2_IDENTITY_ENTRIES: tuple[str, ...] = tuple(a2lite.REQUIRED_A2_IDENTITY_ENTRIES)

#: exactly these two matched arms; TOPO is a completed parent result, not a run
CONFIRM_ARMS: tuple[str, ...] = ("REAL", "INDEP")
CONFIRM_LABEL: dict[str, str] = {"REAL": "R0", "INDEP": "I0"}
FORBIDDEN_ARMS: tuple[str, ...] = ("TOPO",)

#: frozen by the parent protocol; the confirmation may not change the horizon
CONFIRM_HORIZON = 320
HORIZON_LABEL = "parent_A2_horizon"

#: frozen parent material bar, unchanged (0.003)
MATERIAL = float(a2.MATERIAL)
#: practical tolerance for the TOPO comparison (user-frozen for this round)
TOPO_TOLERANCE = 0.003

#: completed parent TOPO-OMP 320-epoch result (never retrained); the decision
#: stage re-reads the file and refuses to run if the live value differs
TOPO_320_SOUP = 0.12681294702464949
TOPO_320_BEST = 0.13136472144449363
TOPO_320_SOURCE = "tracks/ksvd/results/e2e_dictenv_a2/omp_screen_topo.json"
TOPO_320_ARM = "TOPO"
TOPO_320_HORIZON = 320

#: completed lite 160-epoch screen, reused as fixed context for the 160 -> 320
#: diagnostic only (the confirmation verdict never depends on these numbers
#: beyond being reported next to them)
LITE_160_RESULTS_DIR = "tracks/ksvd/results/e2e_dictenv_a2_lite"
LITE_160_SOUP: dict[str, float] = {
    "REAL": 0.15437116196932038,
    "INDEP": 0.15993232336913935,
}
LITE_160_TAG: dict[str, str] = {"REAL": "lite_omp_R0", "INDEP": "lite_omp_I0"}
LITE_160_HORIZON = a2lite.SCREEN_HORIZON

#: paired late window of the confirmation run (diagnostic only, no gate)
LATE_WINDOW_START = 241
LATE_WINDOW_END = CONFIRM_HORIZON

#: 160 -> 320 gap classification tolerance (descriptive, not a gate)
GAP_CHANGE_TOLERANCE = 0.001

#: continuation regimes; exactly one is used for *both* arms (no mixing)
CONTINUATION_FRESH = "fresh_matched_320"
CONTINUATION_RESUME = "exact_resume_161_320"
CONTINUATION_MODES: tuple[str, ...] = (CONTINUATION_FRESH, CONTINUATION_RESUME)
#: what an exact continuation would have to carry (checked, then recorded)
REQUIRED_RESUME_STATE: tuple[str, ...] = (
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "current_epoch",
    "rng_states",
    "loader_order_state",
    "soup_member_states_for_full_trajectory",
)

VERDICT_COMPETITIVE = "ATTRIBUTED_PAIRING_SUPPORTED_AND_COMPETITIVE"
VERDICT_NOT_COMPETITIVE = "ATTRIBUTED_PAIRING_SUPPORTED_BUT_NOT_COMPETITIVE"
VERDICT_NO_SIGNAL = "NO_CONFIRMED_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D"
VERDICT_IDENTITY_FAILURE = "ARTIFACT_IDENTITY_FAILURE"
#: internal guard: unusable evidence is a stopped run, not a scientific label
VERDICT_UNUSABLE = "CONFIRMATION_UNUSABLE_EVIDENCE"

VERDICTS: tuple[str, ...] = (
    VERDICT_COMPETITIVE,
    VERDICT_NOT_COMPETITIVE,
    VERDICT_NO_SIGNAL,
    VERDICT_IDENTITY_FAILURE,
)

#: never auto-started by this round, whatever the outcome
DEFERRED_STAGES: tuple[str, ...] = (
    "seed 1 replication",
    "PCA32 dense control",
    "continuity-v2 diagnostic",
    "IHT-10/30/100/200 coder qualification",
    "task-coupled E2E sparse dictionary (E0/E1/E2)",
    "REAL->INDEP code-pairing-removal mechanism",
    "node-only / edge-only pairing interventions",
    "DenseTied specificity control",
    "TOPO-OMP retraining",
    "K64 dictionary",
    "s12 dictionary",
    "official test split",
    "any architecture / hyper-parameter sweep",
)


def confirm_arm_list() -> Sequence[str]:
    """Guard: the confirmation may only ever run REAL and INDEP."""
    for arm in CONFIRM_ARMS:
        if arm in FORBIDDEN_ARMS:
            raise RuntimeError(f"confirmation must not run the {arm} arm")
    if tuple(CONFIRM_ARMS) != ("REAL", "INDEP"):
        raise RuntimeError("confirmation arms are frozen to REAL and INDEP")
    return CONFIRM_ARMS


def topo_delta(real_soup: float, topo_soup: float) -> float:
    """``MAE_REAL - MAE_TOPO``; positive = REAL worse, negative = REAL better."""
    return float(real_soup) - float(topo_soup)


def late_window_stats(
    curve_indep: Mapping[int, float],
    curve_real: Mapping[int, float],
    *,
    start: int = LATE_WINDOW_START,
    end: int = LATE_WINDOW_END,
) -> dict[str, Any]:
    """Paired late-window statistics of ``INDEP - REAL`` (diagnostic only).

    Epochs are paired; a missing or non-finite epoch raises (fail-closed), so a
    truncated run can never be silently summarised as "no difference".
    """
    epochs = list(range(int(start), int(end) + 1))
    missing = [e for e in epochs if e not in curve_indep or e not in curve_real]
    if missing:
        raise a2lite.CurveError(f"late window epochs missing from a curve: {missing[:8]}...")
    deltas = [float(curve_indep[e]) - float(curve_real[e]) for e in epochs]
    if not all(np.isfinite(d) for d in deltas):
        raise a2lite.CurveError("late window contains a non-finite paired delta")
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "window": [int(start), int(end)],
        "n_epochs": len(epochs),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "delta_first": float(array[0]),
        "delta_last": float(array[-1]),
        "min_delta": float(array.min()),
        "max_delta": float(array.max()),
        "note": "diagnostic only; the frozen verdict never uses this window",
    }


def gap_change(
    g_pair_160: float, g_pair_320: float, *, tolerance: float = GAP_CHANGE_TOLERANCE
) -> dict[str, Any]:
    """Describe how the paired gap moved from the 160 screen to the 320 run."""
    delta = float(g_pair_320) - float(g_pair_160)
    if abs(delta) <= float(tolerance):
        classification = "KEPT"
    elif delta > 0.0:
        classification = "WIDENED"
    else:
        classification = "SHRUNK"
    return {
        "G_pair_160": float(g_pair_160),
        "G_pair_320": float(g_pair_320),
        "change": delta,
        "tolerance": float(tolerance),
        "classification": classification,
        "sign_reversed": bool((float(g_pair_160) > 0.0) != (float(g_pair_320) > 0.0)),
        "note": "diagnostic only; the 320 confirmation outranks the 160 screen",
    }


def confirm_decision(
    g_pair_320: float,
    delta_vs_topo: float,
    *,
    identity_ok: bool,
    curves_finite: bool,
    material: float = MATERIAL,
    tolerance: float = TOPO_TOLERANCE,
) -> dict[str, Any]:
    """Frozen confirmation decision matrix (pure function).

    * identity failure  -> ``ARTIFACT_IDENTITY_FAILURE`` (stop, nothing rebuilt)
    * unusable curves   -> internal guard label; the runner refuses to write a
      scientific verdict (a broken run is not a negative result)
    * ``G_pair_320 < material`` -> Case A
    * else competitive iff ``Delta_vs_TOPO <= tolerance`` -> Case B, otherwise C
    """
    g_pair = float(g_pair_320)
    delta = float(delta_vs_topo)
    if not identity_ok:
        return {
            "case": "IDENTITY",
            "verdict": VERDICT_IDENTITY_FAILURE,
            "primary_pass": False,
            "competitive": None,
            "G_pair_320": g_pair,
            "Delta_vs_TOPO": delta,
            "material": float(material),
            "tolerance": float(tolerance),
            "reason": "frozen artifact provenance mismatch; the confirmation is not authorised",
        }
    if not curves_finite:
        return {
            "case": "UNUSABLE",
            "verdict": VERDICT_UNUSABLE,
            "primary_pass": False,
            "competitive": None,
            "G_pair_320": g_pair,
            "Delta_vs_TOPO": delta,
            "material": float(material),
            "tolerance": float(tolerance),
            "reason": "320-epoch curves missing or non-finite; no scientific verdict is written",
        }
    primary = bool(g_pair >= float(material))
    competitive = bool(primary and delta <= float(tolerance))
    if not primary:
        return {
            "case": "A",
            "verdict": VERDICT_NO_SIGNAL,
            "primary_pass": False,
            "competitive": None,
            "G_pair_320": g_pair,
            "Delta_vs_TOPO": delta,
            "material": float(material),
            "tolerance": float(tolerance),
            "reason": (
                f"G_pair_320 {g_pair:.6f} < {float(material):.3f}: the 320-epoch frozen-OMP "
                "pairing gap is not material; the 32-D attributed-code-formation line closes"
            ),
        }
    if competitive:
        return {
            "case": "B",
            "verdict": VERDICT_COMPETITIVE,
            "primary_pass": True,
            "competitive": True,
            "G_pair_320": g_pair,
            "Delta_vs_TOPO": delta,
            "material": float(material),
            "tolerance": float(tolerance),
            "reason": (
                f"G_pair_320 {g_pair:.6f} >= {float(material):.3f} and Delta_vs_TOPO "
                f"{delta:.6f} <= {float(tolerance):.3f}: the real pairing adds task value "
                "without a practical penalty against the topology-only baseline"
            ),
        }
    return {
        "case": "C",
        "verdict": VERDICT_NOT_COMPETITIVE,
        "primary_pass": True,
        "competitive": False,
        "G_pair_320": g_pair,
        "Delta_vs_TOPO": delta,
        "material": float(material),
        "tolerance": float(tolerance),
        "reason": (
            f"G_pair_320 {g_pair:.6f} >= {float(material):.3f} but Delta_vs_TOPO "
            f"{delta:.6f} > {float(tolerance):.3f}: the pairing carries task information "
            "yet the attributed dictionary/code formation is not competitive with the "
            "topology-only dictionary; no attributed E2E/IHT follows"
        ),
    }


def continuation_decision(per_arm: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Choose one continuation regime for *both* arms (never mixed).

    ``per_arm[arm][item]`` must be a mapping with ``present_and_provable: bool``
    for every item in :data:`REQUIRED_RESUME_STATE`.  Exact continuation is used
    only when every item is present and provable for **every** arm; otherwise
    the round restarts both arms as fresh matched 320-epoch runs.  Mixing is
    prohibited by construction: a single mode is returned.
    """
    missing: dict[str, list[str]] = {}
    for arm in CONFIRM_ARMS:
        evidence = per_arm.get(arm)
        if not evidence:
            missing[arm] = list(REQUIRED_RESUME_STATE)
            continue
        missing[arm] = [
            item
            for item in REQUIRED_RESUME_STATE
            if not bool(evidence.get(item, {}).get("present_and_provable"))
        ]
    resumable = bool(all(not items for items in missing.values()))
    if resumable:
        return {
            "mode": CONTINUATION_RESUME,
            "resume_authorised": True,
            "missing": missing,
            "mixing_prohibited": True,
            "reason": (
                "every arm carries a complete, provable continuation state; both arms "
                "resume at epoch 161 with the same regime"
            ),
        }
    return {
        "mode": CONTINUATION_FRESH,
        "resume_authorised": False,
        "missing": missing,
        "mixing_prohibited": True,
        "reason": (
            "exact continuation cannot be proven for every arm (missing: "
            + "; ".join(f"{arm}:{','.join(items)}" for arm, items in missing.items() if items)
            + "); both arms restart as fresh matched 320-epoch runs, no pseudo-resume"
        ),
    }


def deferred_stage_record() -> dict[str, str]:
    """Deferral record (nothing is cancelled permanently)."""
    return {name: "DEFERRED_PENDING_USER_AUTHORIZATION" for name in DEFERRED_STAGES}
