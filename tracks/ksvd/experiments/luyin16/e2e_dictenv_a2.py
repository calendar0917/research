"""E2E-DictEnv-A2 — Attributed Code Value & Compression Control (core helpers).

Round ``e2e_dictenv_a2`` (study ``zinc-context-gap``).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_a2_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_a2_prior_artifact_audit.md``.

A2 re-asks the question A1 never reached — does the *real* structure↔attribute
pairing inside dictionary-code formation carry task value? — and adds the
control that separates "pairing is worthless" from "pairing is worth something
but the frozen ``K=32, s=8`` sparse code destroys it":

```
Stage 1  frozen exact-OMP code screen        REAL vs INDEP (primary), REAL vs TOPO
Stage 2  train-only dense rank-32 PCA32      REAL vs INDEP   (only if Stage 1 fails)
Stage 3  shared-step IHT coder qualification (only on the sparse route)
Stage 4  matched E2E sparse dictionary       REAL vs INDEP / TOPO
         + inference-only code-pairing removal mechanism
Stage 5  sparse vs dense-tied specificity    (only if Stage 4 + mechanism pass)
```

This module holds only *new* A2 logic: the continuity-v2 pair population and
statistics (a diagnostic, never a gate), the rank-32 dense compression reference
(delegating the PCA itself to the already-audited ``sdb_v0.fit_pca_rank``), the
frozen per-arm liveness criterion, and the frozen decision table.  Every
scientific object (``phi65``, ``chi_REAL433``, ``chi_INDEP433``, scaler,
dictionaries, OMP codes, H1 model, coder) is imported from
``e2e_dictenv_a1`` and is not re-derived here.

Official ZINC **test is never loaded** anywhere in this round.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb

PROTOCOL_VERSION = "e2e_dictenv_a2"
ROUND = "E2E-DictEnv-A2"
SUBTITLE = "Attributed Code Value & Compression Control"
STUDY = "zinc-context-gap"

# ---------------------------------------------------------------------------
# frozen stage labels
# ---------------------------------------------------------------------------

#: Stage-1 / Stage-2 frozen-code screen arms (the A2 brief's T0 / I0 / R0)
SCREEN_LABEL: dict[str, str] = {"TOPO": "T0", "INDEP": "I0", "REAL": "R0"}
#: Stage-4 / Stage-5 trained end-to-end arms (the A2 brief's E0 / E1 / E2)
E2E_LABEL: dict[str, str] = {"TOPO": "E0", "INDEP": "E1", "REAL": "E2"}
SCREEN_ARMS: tuple[str, ...] = ("TOPO", "INDEP", "REAL")
PCA_ARMS: tuple[str, ...] = ("INDEP", "REAL")

#: frozen material thresholds (identical to A1/P1/P2-ABS: no new bar)
MATERIAL = float(a1.MATERIAL)                       # 0.003
MECHANISM_THRESHOLD = float(a1.MECHANISM_THRESHOLD)  # 0.010
#: the dense compression control must have exactly the sparse code width
PCA_RANK = int(a1.DICT_K)                            # 32

# ---------------------------------------------------------------------------
# continuity-v2 (diagnostic only — no gate, no threshold, no selection)
# ---------------------------------------------------------------------------

CONTINUITY_V2_POOL = 2000
CONTINUITY_V2_SEED = 20260930
CONTINUITY_V2_ROUNDS = int(a1.CONTINUITY_ROUNDS)          # 3
CONTINUITY_V2_PAIRS_PER_STRATUM = 4000
#: A1-pool reconciliation (descriptive only; reproduces the audit §3 populations)
CONTINUITY_A1_POOL = int(a1.CONTINUITY_POOL)              # 1500
CONTINUITY_A1_SEED = int(sdb.DICT_SEED) + int(a1.CONTINUITY_SEED_OFFSET)  # 20260933
CONTINUITY_A1_SAMPLED_PAIRS = 200000
CONTINUITY_A1_POSTHOC_SEED_OFFSET = 1

#: fixed bins on the official-train radius-2 patch-size distribution (audit §7):
#: small = n_patch <= 5 (31.83 %), medium = 6..8 (61.95 %), large = >= 9 (6.22 %)
STRATA: tuple[tuple[str, int, int], ...] = (
    ("small", 0, 5),
    ("medium", 6, 8),
    ("large", 9, 1 << 30),
)

try:  # pragma: no cover - the runner is imported by tests either way
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as _a1run

    IHT_CANDIDATE_STEPS = tuple(int(step) for step in _a1run.a1.IHT_CANDIDATE_STEPS)
    IHT_QUALIFY_MAX_REC = float(_a1run.a1.IHT_QUALIFY_MAX_REC)
except Exception:  # pragma: no cover
    IHT_CANDIDATE_STEPS = (10, 30, 100, 200)
    IHT_QUALIFY_MAX_REC = 0.002

# ---------------------------------------------------------------------------
# Stage-3 coder qualification rule (frozen): one shared step count
# ---------------------------------------------------------------------------


def select_common_iht_steps(
    per_arm_normalized_err: Mapping[str, Mapping[str, float]],
    candidates: Sequence[int] = IHT_CANDIDATE_STEPS,
    bar: float = IHT_QUALIFY_MAX_REC,
) -> dict[str, Any]:
    """Pick the smallest IHT step count that qualifies for *every* arm.

    A step count is qualified only if the IHT reconstruction of every arm is at
    most ``bar`` in normalized error (same bar for all arms, no per-arm ``λ``),
    and a single shared count is used for all three arms downstream.
    """
    steps = [int(step) for step in candidates]
    arms = list(per_arm_normalized_err)
    for arm in arms:
        missing = [str(step) for step in steps if str(step) not in per_arm_normalized_err[arm]]
        if missing:
            raise RuntimeError(
                f"incomplete coder evidence for {arm}: missing candidate steps {missing}"
            )
    qualified = [
        step
        for step in steps
        if all(float(per_arm_normalized_err[arm][str(step)]) <= bar for arm in arms)
    ]
    selected = min(qualified) if qualified else None
    return {
        "candidates": steps,
        "bar": float(bar),
        "arms": arms,
        "qualified_steps": qualified,
        "selected_steps": selected,
        "coder_qualified": bool(qualified),
    }


# ---------------------------------------------------------------------------
# per-arm liveness criterion (frozen; guards against FSAB-style dead channels)
# ---------------------------------------------------------------------------

LIVENESS_MIN_ACTIVE_ATOMS = 24
LIVENESS_MIN_EFFECTIVE_ATOMS = 8.0
LIVENESS_MIN_MOVEMENT_RELATIVE = 0.01
LIVENESS_MIN_USAGE_SPEARMAN = 0.5
LIVENESS_SLOTS = ("node_encoder", "edge_encoder", "anchor_encoder")

# ---------------------------------------------------------------------------
# the frozen verdict label set (pre-registration §14)
# ---------------------------------------------------------------------------

VERDICT_ARTIFACT_IDENTITY_FAILURE = "ARTIFACT_IDENTITY_FAILURE"
VERDICT_GATE0_NOT_QUALIFIED = "GATE0_NOT_QUALIFIED"
VERDICT_CODER_NOT_QUALIFIED = "CODER_NOT_QUALIFIED"
VERDICT_CODE_FORMATION_NOT_LIVE = "CODE_FORMATION_NOT_LIVE"
VERDICT_PAIRING_SUPPORTED_AT_FROZEN_OMP = "PAIRING_SUPPORTED_AT_FROZEN_OMP"
VERDICT_PAIRING_NOT_SPECIFIC = "ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC"
VERDICT_SPARSE_SUPPORTED = "ATTRIBUTED_SPARSE_DICTIONARY_SUPPORTED"
VERDICT_COMPRESSION_BOTTLENECK = "ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK"
VERDICT_NO_SIGNAL = "NO_ATTRIBUTED_CODE_FORMATION_SIGNAL"

VERDICTS: tuple[str, ...] = (
    VERDICT_ARTIFACT_IDENTITY_FAILURE,
    VERDICT_GATE0_NOT_QUALIFIED,
    VERDICT_CODER_NOT_QUALIFIED,
    VERDICT_CODE_FORMATION_NOT_LIVE,
    VERDICT_PAIRING_SUPPORTED_AT_FROZEN_OMP,
    VERDICT_PAIRING_NOT_SPECIFIC,
    VERDICT_SPARSE_SUPPORTED,
    VERDICT_COMPRESSION_BOTTLENECK,
    VERDICT_NO_SIGNAL,
)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation of ranks (the frozen A1/TCCD implementation).

    ``nan`` for fewer than three pairs, exactly as in the A1 post-hoc paths.
    """
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size:
        raise ValueError("spearman inputs must have equal length")
    if left.size < 3:
        return float("nan")
    return float(
        np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1]
    )


def angular_distances(table: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """``arccos(clip(cos, -1, 1))`` between the rows of ``table`` given by ``pairs``.

    Scale invariant, so it is the pre-registered primary distance for
    continuity-v2.  (Rank-equivalent to the A1 post-hoc ``1 - cos`` form.)
    """
    table = np.asarray(table, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    left = table[pairs[:, 0]]
    right = table[pairs[:, 1]]
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-30
    )
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    return np.arccos(cosine)


def euclidean_distances(table: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(table[pairs[:, 0]] - table[pairs[:, 1]], axis=1)


def graded_geometry(
    table: np.ndarray, pairs: np.ndarray, similarities: np.ndarray
) -> dict[str, Any]:
    """Graded relation between attributed-WL similarity and representation distance.

    The pre-registered primary quantity is
    ``Spearman(WL similarity, -angular distance)``; the Euclidean version is a
    secondary column, and the mean distances are descriptive.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    similarities = np.asarray(similarities, dtype=np.float64).reshape(-1)
    angular = angular_distances(table, pairs)
    euclidean = euclidean_distances(table, pairs)
    return {
        "n_pairs": int(pairs.shape[0]),
        "rho_angular": spearman(similarities, -angular),
        "rho_euclidean": spearman(similarities, -euclidean),
        "mean_angular_distance": float(angular.mean()) if angular.size else None,
        "mean_euclidean_distance": float(euclidean.mean()) if euclidean.size else None,
        "mean_wl_similarity": float(similarities.mean()) if similarities.size else None,
    }


def stratum_of(size: int) -> str:
    """Stratum name of one radius-2 patch size (frozen fixed bins)."""
    value = int(size)
    for name, low, high in STRATA:
        if low <= value <= high:
            return name
    raise ValueError(f"patch size {value} outside every stratum")


def stratum_table() -> dict[str, Any]:
    return {
        "bins": [{"name": name, "low": int(low), "high": None if high > 1 << 29 else int(high)} for name, low, high in STRATA],
        "source": "official-train radius-2 n_patch quantiles (audit §7): q25=5, q90=8",
        "shares": {"small": 0.3183015056288418, "medium": 0.6194963395262104, "large": 0.06220215484494786},
    }


def equal_size_molecule_disjoint_pairs(
    sizes: Sequence[int],
    molecules: Sequence[int],
    key_ids: Sequence[int],
    *,
    cap_per_stratum: int,
    seed: int,
) -> dict[str, Any]:
    """The frozen continuity-v2 pair population.

    Every pool pair ``i < j`` is kept iff

    * ``sizes[i] == sizes[j]``            (equal patch size), and
    * ``molecules[i] != molecules[j]``    (molecule-disjoint), and
    * ``key_ids[i] != key_ids[j]``        (not isomorphic: distinct rooted
      typed-WL canonical slot key).

    Valid pairs are then deterministically permuted (``seed``) and capped at
    ``cap_per_stratum`` **per size stratum**; the pooled row uses the same
    permutation over all valid pairs (cap ``4 x cap_per_stratum`` so that it is
    never the binding constraint).  Fully deterministic given the pool.
    """
    sizes_arr = np.asarray(sizes, dtype=np.int64).reshape(-1)
    molecules_arr = np.asarray(molecules, dtype=np.int64).reshape(-1)
    key_arr = np.asarray(key_ids, dtype=np.int64).reshape(-1)
    n_pool = sizes_arr.shape[0]
    if not (molecules_arr.shape[0] == key_arr.shape[0] == n_pool):
        raise ValueError("sizes / molecules / key_ids must have equal length")
    left, right = np.triu_indices(n_pool, k=1)
    keep = (
        (sizes_arr[left] == sizes_arr[right])
        & (molecules_arr[left] != molecules_arr[right])
        & (key_arr[left] != key_arr[right])
    )
    valid = np.column_stack([left[keep], right[keep]]).astype(np.int64)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(valid.shape[0]) if valid.shape[0] else np.zeros(0, dtype=np.int64)
    shuffled = valid[order] if valid.shape[0] else valid
    out: dict[str, Any] = {
        "n_pool": int(n_pool),
        "n_pool_pairs": int(left.shape[0]),
        "n_valid_pairs": int(valid.shape[0]),
        "n_eligible_pairs": int(valid.shape[0]),
        "cap_per_stratum": int(cap_per_stratum),
        "seed": int(seed),
        "strata": {},
        "stratum_counts": {},
        "pooled": shuffled[: 4 * int(cap_per_stratum)].copy(),
        "pooled_sampling": (
            "unbiased pooled sample: the first 4*cap of the same shuffled valid-pair "
            "permutation as the strata (the strata are equal-n capped subsamples of that "
            "same permutation, so a pair may appear in both the pooled row and one stratum; "
            "the three strata are pairwise disjoint and partition the size axis)"
        ),
    }
    consumed = np.zeros(shuffled.shape[0], dtype=bool)
    for name, low, high in STRATA:
        mask = (sizes_arr[shuffled[:, 0]] >= low) & (sizes_arr[shuffled[:, 0]] <= high)
        subset = shuffled[mask]
        out["strata"][name] = subset[: int(cap_per_stratum)].copy()
        out["stratum_counts"][name] = {
            "n_valid": int(subset.shape[0]),
            "n_used": int(min(subset.shape[0], int(cap_per_stratum))),
            "size_low": int(low),
            "size_high": None if high > 1 << 29 else int(high),
        }
        consumed |= mask
    out["n_used_pooled"] = int(out["pooled"].shape[0])
    out["n_valid_pairs_outside_strata"] = int((~consumed).sum())
    return out


def reconstruct_a1_posthoc_pairs(
    sizes: Sequence[int],
    molecules: Sequence[int],
    key_ids: Sequence[int],
    *,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Reproduce the A1 post-hoc sampled-pair population (audit §3, Path R/D).

    Same RNG stream and same acceptance rule as
    ``zinc_e2e_dictenv_a1.posthoc_continuity_stage`` (``i < j`` and distinct
    rooted key), so that A2 can report the *labeled* A1 numbers without reusing
    them as a threshold or baseline.
    """
    sizes_arr = np.asarray(sizes, dtype=np.int64).reshape(-1)
    molecules_arr = np.asarray(molecules, dtype=np.int64).reshape(-1)
    key_arr = np.asarray(key_ids, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    draw = np.column_stack(
        [rng.integers(0, sizes_arr.shape[0], size=int(samples)), rng.integers(0, sizes_arr.shape[0], size=int(samples))]
    )
    keep = (draw[:, 0] < draw[:, 1]) & (key_arr[draw[:, 0]] != key_arr[draw[:, 1]])
    pairs = draw[keep].astype(np.int64)
    equal_size = sizes_arr[pairs[:, 0]] == sizes_arr[pairs[:, 1]]
    molecule_disjoint = molecules_arr[pairs[:, 0]] != molecules_arr[pairs[:, 1]]
    return {
        "pairs": pairs,
        "equal_size": equal_size,
        "molecule_disjoint": molecule_disjoint,
        "equal_size_molecule_disjoint": equal_size & molecule_disjoint,
    }


# ---------------------------------------------------------------------------
# dense rank-32 compression reference (delegates the PCA to sdb_v0)
# ---------------------------------------------------------------------------


def fit_dense_rank(
    X_train: np.ndarray, *, rank: int = PCA_RANK
) -> tuple[sdb.PCARank, dict[str, Any]]:
    """Train-only affine rank-``rank`` PCA of one 433-D object.

    The PCA itself is ``sdb_v0.fit_pca_rank`` — the shared, already-audited
    implementation used by SDB-v0 Stage 1/2 and FEC-D1.  A2 adds no second PCA.
    """
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != a1.A1_DIM:
        raise RuntimeError(f"dense rank-{rank} input must be [N, {a1.A1_DIM}], got {X.shape}")
    pca = sdb.fit_pca_rank(X, rank=int(rank))
    mean = np.asarray(pca.mean, dtype=np.float64)
    components = np.asarray(pca.components, dtype=np.float64)
    centered = X - mean[None, :]
    codes = centered @ components.T
    reconstruction = mean[None, :] + codes @ components
    residual = X - reconstruction
    total = float(((X - mean[None, :]) ** 2).sum())
    projection = np.asarray(components.T, dtype=np.float32)
    tied = codes @ components
    column_norms = np.linalg.norm(np.asarray(projection, dtype=np.float64), axis=0)
    normalized = np.asarray(
        np.asarray(projection, dtype=np.float64)
        / np.maximum(column_norms[None, :], 1e-12),
        dtype=np.float32,
    )
    meta = {
        "rank": int(components.shape[0]),
        "input_dim": int(X.shape[1]),
        "n_fit_rows": int(X.shape[0]),
        "mean_subtracted": True,
        "fit_split": "official train",
        "components_sha256_f32": a1_hash(projection),
        "train_normalized_rec": float(
            (residual ** 2).sum() / max((X ** 2).sum(), 1e-30)
        ),
        "train_explained_variance_fraction_centered": float(
            1.0 - ((residual ** 2).sum() / max(total, 1e-30))
        ),
        "component_norm_min": float(np.linalg.norm(components, axis=1).min()),
        "component_norm_max": float(np.linalg.norm(components, axis=1).max()),
        # The frozen A1 tied decoder reconstructs as ``coord @ Dbar.T`` with
        # ``Dbar = F.normalize(D, dim=0)`` and adds no mean, so the object the
        # model sees is exactly ``projection`` (its columns are already unit
        # norm: they are the rows of ``components``) and the attached centred
        # codes are the frozen coordinates (pre-registration §5).  Both the
        # affine-PCA residual and the tied (mean-free) residual are reported.
        "projection_column_norm_min": float(column_norms.min()),
        "projection_column_norm_max": float(column_norms.max()),
        "projection_vs_normalized_dictionary_max_abs": float(
            np.abs(normalized - projection).max()
        ),
        "code_convention": "(X - mean_train) @ V^T attached as frozen precomputed coordinates",
        "reconstruction_convention": "coord @ D with D = V^T (no mean added; frozen A1 tied decoder)",
        "train_tied_normalized_rec": float(
            ((X - tied) ** 2).sum() / max((X ** 2).sum(), 1e-30)
        ),
        "implementation": "sdb_v0.fit_pca_rank (shared; no A2 re-implementation)",
    }
    if meta["rank"] != PCA_RANK:
        raise RuntimeError(f"dense compression rank {meta['rank']} != frozen {PCA_RANK}")
    if meta["projection_vs_normalized_dictionary_max_abs"] > 1e-5:
        raise RuntimeError(
            "the frozen PCA projection is not what the model's normalized dictionary evaluates to"
        )
    return pca, meta


def dense_codes_f32(pca: sdb.PCARank, X: np.ndarray) -> np.ndarray:
    """Frozen centred rank-32 codes, float32 (no refit, valid never enters)."""
    codes = np.asarray(pca.dense_codes(np.asarray(X, dtype=np.float64)), dtype=np.float32)
    if codes.shape[1] != PCA_RANK:
        raise RuntimeError(f"dense code width {codes.shape[1]} != frozen {PCA_RANK}")
    return codes


def pca_projection_matrix(pca: sdb.PCARank) -> np.ndarray:
    """``[input_dim, 32]`` projection whose frozen rec is the PCA reconstruction."""
    return np.asarray(np.asarray(pca.components, dtype=np.float32).T, dtype=np.float32)


def a1_hash(array: np.ndarray) -> str:
    """sha256 of contiguous float32 bytes (same convention as the A1 runner)."""
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(np.asarray(array, dtype=np.float32)).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# liveness gate
# ---------------------------------------------------------------------------


def liveness_gate(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen per-arm liveness criterion (pre-registration §9).

    Returns ``{"passed": bool, "reasons": [...], "checks": {...}}``.  A dead
    dictionary/branch cannot be reported as a mechanism success, whatever the
    MAE looks like.
    """
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    def record(name: str, ok: bool) -> None:
        checks[name] = bool(ok)
        if not ok:
            reasons.append(name)

    grad = entry.get("dictionary_grad_norm")
    record("dictionary_grad_norm_positive", grad is not None and float(grad) > 0.0)

    movement = entry.get("soup_D_vs_ksvd_init", {}).get("relative")
    record(
        "dictionary_moved_from_init",
        movement is not None and float(movement) > LIVENESS_MIN_MOVEMENT_RELATIVE,
    )
    record(
        "active_atoms",
        int(entry.get("active_atoms_train", 0)) >= LIVENESS_MIN_ACTIVE_ATOMS,
    )
    record(
        "effective_atoms",
        float(entry.get("effective_atoms_train", 0.0)) >= LIVENESS_MIN_EFFECTIVE_ATOMS,
    )
    record(
        "code_variance",
        float(entry.get("code_variance_mean_train", 0.0)) > 0.0,
    )
    record(
        "train_valid_usage_spearman",
        float(entry.get("train_valid_usage_spearman", 0.0)) >= LIVENESS_MIN_USAGE_SPEARMAN,
    )
    slots = entry.get("slot_gradients") or {}
    for slot in LIVENESS_SLOTS:
        slot_entry = slots.get(slot) or {}
        # Operationalization of the frozen "slot gradient norms all > 0": the
        # primary instrument is the *parameter* gradient norm of the slot module
        # (a branch is alive iff its parameters receive task gradient).  A1's
        # input-tensor probe is reported next to it; it is used only when the
        # parameter norm was not recorded (the anchor encoder's input tensor is
        # grad-free by construction, so the input probe alone would score a live
        # branch as dead instrumentation).
        value = slot_entry.get("param_grad_norm")
        if value is None:
            value = slot_entry.get("slot_grad_norm")
        record(f"slot_grad_{slot}", value is not None and float(value) > 0.0)
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "reasons": reasons,
        "thresholds": {
            "min_active_atoms": LIVENESS_MIN_ACTIVE_ATOMS,
            "min_effective_atoms": LIVENESS_MIN_EFFECTIVE_ATOMS,
            "min_movement_relative": LIVENESS_MIN_MOVEMENT_RELATIVE,
            "min_usage_spearman": LIVENESS_MIN_USAGE_SPEARMAN,
        },
    }


def liveness_all_pass(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply :func:`liveness_gate` to every formal arm."""
    per_arm = {arm: liveness_gate(entry) for arm, entry in arms.items()}
    return {
        "arms": per_arm,
        "all_pass": bool(all(entry["passed"] for entry in per_arm.values())),
        "failing_arms": [arm for arm, entry in per_arm.items() if not entry["passed"]],
    }


# ---------------------------------------------------------------------------
# the frozen decision table
# ---------------------------------------------------------------------------


def _require(payload: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    if payload is None:
        raise ValueError(
            f"{name} is required at this point of the frozen decision table; "
            "an interrupted run must be resumed, not decided"
        )
    return payload


def verdict_from_evidence(
    *,
    identity_passed: bool,
    gate0_passed: bool,
    stage1: Mapping[str, Any] | None = None,
    stage2: Mapping[str, Any] | None = None,
    stage3: Mapping[str, Any] | None = None,
    stage4: Mapping[str, Any] | None = None,
    liveness: Mapping[str, Any] | None = None,
    mechanism: Mapping[str, Any] | None = None,
    specificity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen A2 decision table (pre-registration §14), as a pure function.

    Each stage payload carries at least ``{"run": bool, ...}`` plus its gate
    value.  Raises ``ValueError`` on an evidence combination that the frozen
    order can never produce (an interrupted run), rather than inventing a
    verdict.
    """
    if not identity_passed:
        return {
            "verdict": VERDICT_ARTIFACT_IDENTITY_FAILURE,
            "reason": "reused-artifact identity check failed",
            "branch": "identity",
        }
    if not gate0_passed:
        return {
            "verdict": VERDICT_GATE0_NOT_QUALIFIED,
            "reason": "correctness / assignment / health / official-test blocker failed",
            "branch": "gate0",
        }
    stage1 = _require(stage1, "stage1 omp decision")
    if not stage1.get("run", True):
        raise ValueError("stage1 not run")
    if stage1.get("primary_pass"):
        stage3 = _require(stage3, "stage3 coder decision")
        if not stage3.get("coder_qualified"):
            return {
                "verdict": VERDICT_CODER_NOT_QUALIFIED,
                "reason": "no frozen IHT step count satisfied the shared 0.002 reconstruction bar",
                "branch": "sparse",
            }
        stage4 = _require(stage4, "stage4 formal decision")
        liveness = _require(liveness, "liveness audit")
        if not liveness.get("all_pass"):
            return {
                "verdict": VERDICT_CODE_FORMATION_NOT_LIVE,
                "reason": "a formal arm failed the frozen liveness criterion",
                "branch": "sparse",
                "failing_arms": list(liveness.get("failing_arms", [])),
            }
        if not stage4.get("primary_pass"):
            return {
                "verdict": VERDICT_PAIRING_SUPPORTED_AT_FROZEN_OMP,
                "reason": "Stage-1 G_pair_OMP passed but Stage-4 G_pair_E2E did not",
                "branch": "sparse",
            }
        mechanism = _require(mechanism, "mechanism")
        if not mechanism.get("pass"):
            return {
                "verdict": VERDICT_PAIRING_NOT_SPECIFIC,
                "reason": "code_pairing_below_threshold",
                "branch": "sparse",
            }
        if specificity is None:
            raise ValueError("stage5 specificity is required once Stage 4 and the mechanism both pass")
        if specificity.get("pass"):
            return {
                "verdict": VERDICT_SPARSE_SUPPORTED,
                "reason": "sparse REAL beats matched dense-tied REAL",
                "branch": "sparse",
            }
        return {
            "verdict": VERDICT_PAIRING_NOT_SPECIFIC,
            "reason": "dense_tied_not_worse",
            "branch": "sparse",
        }
    stage2 = _require(stage2, "stage2 compression decision")
    if stage2.get("primary_pass"):
        return {
            "verdict": VERDICT_COMPRESSION_BOTTLENECK,
            "reason": "dense rank-32 compression recovers the pairing contrast that K32/s8 loses",
            "branch": "compression",
        }
    return {
        "verdict": VERDICT_NO_SIGNAL,
        "reason": "neither frozen sparse OMP nor dense rank-32 compression separates REAL from INDEP",
        "branch": "compression",
    }


__all__ = [
    "PROTOCOL_VERSION",
    "ROUND",
    "SUBTITLE",
    "STUDY",
    "SCREEN_LABEL",
    "E2E_LABEL",
    "SCREEN_ARMS",
    "PCA_ARMS",
    "MATERIAL",
    "MECHANISM_THRESHOLD",
    "PCA_RANK",
    "CONTINUITY_V2_POOL",
    "CONTINUITY_V2_SEED",
    "CONTINUITY_V2_ROUNDS",
    "CONTINUITY_V2_PAIRS_PER_STRATUM",
    "CONTINUITY_A1_POOL",
    "CONTINUITY_A1_SEED",
    "CONTINUITY_A1_SAMPLED_PAIRS",
    "CONTINUITY_A1_POSTHOC_SEED_OFFSET",
    "STRATA",
    "IHT_CANDIDATE_STEPS",
    "IHT_QUALIFY_MAX_REC",
    "select_common_iht_steps",
    "LIVENESS_MIN_ACTIVE_ATOMS",
    "LIVENESS_MIN_EFFECTIVE_ATOMS",
    "LIVENESS_MIN_MOVEMENT_RELATIVE",
    "LIVENESS_MIN_USAGE_SPEARMAN",
    "VERDICTS",
    "spearman",
    "angular_distances",
    "euclidean_distances",
    "graded_geometry",
    "stratum_of",
    "stratum_table",
    "equal_size_molecule_disjoint_pairs",
    "reconstruct_a1_posthoc_pairs",
    "fit_dense_rank",
    "dense_codes_f32",
    "pca_projection_matrix",
    "a1_hash",
    "liveness_gate",
    "liveness_all_pass",
    "verdict_from_evidence",
]
