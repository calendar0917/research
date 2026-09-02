"""Shared split and audit helpers for MolHIV development and frozen official evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    fit_indices: np.ndarray
    heldout_indices: np.ndarray
    scaffold_groups: np.ndarray
    official_valid_evaluations: int
    official_test_evaluations: int = 0


def resolve_evaluation_protocol(
    *,
    evaluation_split: str,
    fold_cache: str,
    fold: int,
    expected_original: np.ndarray,
    official_train: np.ndarray,
    official_valid: np.ndarray,
    official_test: np.ndarray,
) -> EvaluationProtocol:
    """Resolve an official-train inner fold or a frozen train -> official split run."""
    with np.load(fold_cache, allow_pickle=False) as folds:
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64), expected_original
        ):
            raise ValueError("fold cache original_indices mismatch")
        if "official_train_indices" in folds.files and not np.array_equal(
            np.asarray(folds["official_train_indices"], dtype=np.int64), official_train
        ):
            raise ValueError("fold cache official train mismatch")
        scaffold_groups = np.asarray(folds["train_scaffold_groups"]).astype(str)
        if len(scaffold_groups) != len(official_train):
            raise ValueError("fold cache scaffold groups are not aligned to official train")
        if evaluation_split == "scaffold_fold":
            fit_indices = np.asarray(folds[f"fold_{fold}_train_indices"], dtype=np.int64)
            heldout_indices = np.asarray(folds[f"fold_{fold}_valid_indices"], dtype=np.int64)
        elif evaluation_split in ("official_valid", "official_test"):
            fit_indices = np.asarray(official_train, dtype=np.int64).copy()
            heldout_source = official_valid if evaluation_split == "official_valid" else official_test
            heldout_indices = np.asarray(heldout_source, dtype=np.int64).copy()
        else:
            raise ValueError(f"unknown evaluation split: {evaluation_split}")

    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("fit and heldout indices overlap")
    if evaluation_split == "scaffold_fold":
        if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != set(official_train.tolist()):
            raise AssertionError("outer scaffold fold does not partition official train")
        valid_evaluations = 0
        test_evaluations = 0
    else:
        if not np.array_equal(fit_indices, official_train):
            raise AssertionError(f"{evaluation_split} fit split is not exact official train")
        expected_heldout = official_valid if evaluation_split == "official_valid" else official_test
        if not np.array_equal(heldout_indices, expected_heldout):
            raise AssertionError(f"{evaluation_split} heldout split mismatch")
        valid_evaluations = int(evaluation_split == "official_valid")
        test_evaluations = int(evaluation_split == "official_test")
    return EvaluationProtocol(
        name=evaluation_split,
        fit_indices=fit_indices,
        heldout_indices=heldout_indices,
        scaffold_groups=scaffold_groups,
        official_valid_evaluations=valid_evaluations,
        official_test_evaluations=test_evaluations,
    )


def forbidden_encoded_indices(
    evaluation_split: str,
    official_valid: np.ndarray,
    official_test: np.ndarray,
) -> np.ndarray:
    if evaluation_split == "official_valid":
        return np.asarray(official_test, dtype=np.int64)
    if evaluation_split == "official_test":
        return np.asarray(official_valid, dtype=np.int64)
    return np.concatenate([official_valid, official_test]).astype(np.int64, copy=False)


def validate_latent_sidecar(path: str, evaluation_split: str, expected_fit_sha: str) -> dict[str, Any]:
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"missing latent-cache audit sidecar: {sidecar}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    valid_encoded = bool(meta.get("official_valid_encoded", False))
    test_encoded = bool(meta.get("official_test_encoded", True))
    if evaluation_split == "official_valid":
        if not valid_encoded or test_encoded:
            raise ValueError("official-valid mode requires valid encoded and test unencoded")
    elif evaluation_split == "official_test":
        if valid_encoded or not test_encoded:
            raise ValueError("official-test mode requires valid unencoded and test encoded")
    elif valid_encoded or test_encoded:
        raise ValueError("scaffold-fold mode requires official valid/test unencoded")
    if meta.get("fit_indices_sha256") != expected_fit_sha:
        raise ValueError("latent metric was not fit on the requested fit split")
    return meta


def validate_upstream_result(
    doc: dict[str, Any],
    *,
    name: str,
    evaluation_split: str,
    fold: int,
    fit_sha: str,
) -> None:
    doc_split = doc.get("evaluation_split", "scaffold_fold")
    if doc_split != evaluation_split or doc.get("fit_indices_sha256") != fit_sha:
        raise ValueError(f"{name} evaluation split mismatch")
    if evaluation_split == "scaffold_fold" and int(doc.get("fold", -1)) != fold:
        raise ValueError(f"{name} fold mismatch")
    policy = doc.get("selection_policy", {})
    expected_valid = int(evaluation_split == "official_valid")
    expected_test = int(evaluation_split == "official_test")
    if int(policy.get("official_valid_evaluations", -1)) != expected_valid:
        raise ValueError(f"{name} official-valid evaluation count mismatch")
    if int(policy.get("official_test_evaluations", -1)) != expected_test:
        raise ValueError(f"{name} official-test evaluation count mismatch")


def evaluation_description(evaluation_split: str, max_graphs: int) -> str:
    if evaluation_split == "official_valid":
        return "fit on complete official train; one frozen evaluation on official valid; official test untouched"
    if evaluation_split == "official_test":
        return "fit on complete official train; official valid excluded; one frozen terminal evaluation on official test"
    if max_graphs <= 0:
        return "official-train only; one of three full outer scaffold folds"
    return f"official-train only; one outer scaffold fold with max_graphs={max_graphs}"
