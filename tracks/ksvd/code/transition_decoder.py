"""Low-capacity transition-aware decoders for overlap patch chains."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .from_scratch_unplanted_representation import upper_vector_to_adjacency
from .overlap_cover import PatchCover


EPS = 1e-12


@dataclass(frozen=True)
class RidgeDecoder:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray


def fit_ridge_decoder(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 1e-2,
) -> RidgeDecoder:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("ridge features/targets must be aligned matrices")
    if alpha < 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0, ddof=0)
    scale[scale <= EPS] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack([standardized, np.ones(features.shape[0])])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    return RidgeDecoder(mean, scale, coefficients)


def predict_ridge_decoder(model: RidgeDecoder, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != model.feature_mean.size:
        raise ValueError("ridge prediction features have the wrong shape")
    standardized = (features - model.feature_mean) / model.feature_scale
    design = np.column_stack([standardized, np.ones(features.shape[0])])
    return design @ model.coefficients


def make_transition_features(
    codes: np.ndarray,
    base_patch_predictions: np.ndarray,
    cover: PatchCover,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current-only features and t>0 transition-aware features."""
    codes = np.asarray(codes, dtype=np.float64)
    predictions = np.asarray(base_patch_predictions, dtype=np.float64)
    patch_count = len(cover.patches)
    patch_size = len(cover.patches[0].node_ids)
    if codes.ndim != 2 or codes.shape[1] != patch_count:
        raise ValueError("codes must have one column per patch")
    if predictions.shape != (patch_count, patch_size * (patch_size - 1) // 2):
        raise ValueError("base predictions have the wrong patch shape")
    current = codes.T.copy()
    transition_rows = []
    for index in range(1, patch_count):
        previous_adjacency = upper_vector_to_adjacency(
            predictions[index - 1], patch_size
        )
        previous_degrees = np.sum(previous_adjacency, axis=1)
        aligned_degrees = np.zeros(patch_size, dtype=np.float64)
        source_slots = np.full(patch_size, -1.0, dtype=np.float64)
        shared_mask = np.zeros(patch_size, dtype=np.float64)
        for left_slot, right_slot in cover.transitions[index - 1].left_to_right_slots:
            aligned_degrees[right_slot] = previous_degrees[left_slot]
            source_slots[right_slot] = left_slot / max(patch_size - 1, 1)
            shared_mask[right_slot] = 1.0
        transition_rows.append(
            np.concatenate(
                [
                    codes[:, index],
                    codes[:, index - 1],
                    aligned_degrees,
                    source_slots,
                    shared_mask,
                ]
            )
        )
    return current, np.stack(transition_rows, axis=0)


def shuffle_transition_context(features: np.ndarray, code_dimension: int) -> np.ndarray:
    """Cyclically misalign context while preserving each current code row."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not 0 < code_dimension < features.shape[1]:
        raise ValueError("invalid transition feature matrix")
    shuffled = features.copy()
    if features.shape[0] > 1:
        shuffled[:, code_dimension:] = np.roll(
            features[:, code_dimension:], shift=1, axis=0
        )
    return shuffled
