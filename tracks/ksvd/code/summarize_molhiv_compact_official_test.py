"""Summarize the single frozen real-prototype compact official-test run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocabulary-result", required=True)
    ap.add_argument("--gate-result", required=True)
    ap.add_argument("--valid-vocabulary-result", required=True)
    ap.add_argument("--valid-gate-result", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    vocabulary = json.loads(Path(args.vocabulary_result).read_text(encoding="utf-8"))
    gate = json.loads(Path(args.gate_result).read_text(encoding="utf-8"))
    valid_vocabulary = json.loads(Path(args.valid_vocabulary_result).read_text(encoding="utf-8"))
    valid_gate = json.loads(Path(args.valid_gate_result).read_text(encoding="utf-8"))
    for name, doc in (("vocabulary", vocabulary), ("gate", gate)):
        if doc.get("evaluation_split") != "official_test":
            raise ValueError(f"{name} is not an official-test result")
        policy = doc.get("selection_policy", {})
        if int(policy.get("official_valid_evaluations", -1)) != 0:
            raise ValueError(f"{name} touched official valid during terminal run")
        if int(policy.get("official_test_evaluations", -1)) != 1:
            raise ValueError(f"{name} official-test count mismatch")

    families = ("farthest", "scaffold_facility")
    heldout = [vocabulary["results"][family]["heldout"] for family in families]
    graph_indices = np.asarray(heldout[0]["graph_indices"], dtype=np.int64)
    labels = np.asarray(heldout[0]["labels"], dtype=np.float64)
    for family, row in zip(families[1:], heldout[1:]):
        if not np.array_equal(graph_indices, np.asarray(row["graph_indices"], dtype=np.int64)):
            raise ValueError(f"{family} graph order mismatch")
        if not np.array_equal(labels, np.asarray(row["labels"], dtype=np.float64)):
            raise ValueError(f"{family} labels mismatch")
    occurrence_probability = np.mean(
        [sigmoid(np.asarray(row["scores"], dtype=np.float64)) for row in heldout], axis=0
    )
    occurrence_auc = float(roc_auc_score(labels, occurrence_probability))
    stored_occurrence_auc = float(gate["base_occurrence"]["heldout_auc"])
    if abs(occurrence_auc - stored_occurrence_auc) > 1e-12:
        raise AssertionError("reconstructed occurrence ensemble does not match gate base")

    real = gate["results"]["uniform__real"]["heldout"]
    shuffled = gate["results"]["uniform__assignment_shuffled"]["heldout"]
    for name, row in (("real", real), ("assignment_shuffled", shuffled)):
        if not np.array_equal(graph_indices, np.asarray(row["graph_indices"], dtype=np.int64)):
            raise ValueError(f"{name} graph order mismatch")
        if not np.array_equal(labels, np.asarray(row["labels"], dtype=np.float64)):
            raise ValueError(f"{name} labels mismatch")

    real_auc = float(roc_auc_score(labels, np.asarray(real["scores"], dtype=np.float64)))
    shuffled_auc = float(roc_auc_score(labels, np.asarray(shuffled["scores"], dtype=np.float64)))
    valid_real_auc = float(valid_gate["results"]["uniform__real"]["heldout"]["auc"])
    valid_occurrence_auc = float(valid_gate["base_occurrence"]["heldout_auc"])
    valid_shuffled_auc = float(
        valid_gate["results"]["uniform__assignment_shuffled"]["heldout"]["auc"]
    )

    fit_invariance = {
        "fit_indices_sha256_equal": vocabulary["fit_indices_sha256"] == valid_vocabulary["fit_indices_sha256"],
        "vocabulary": {},
        "compact": {},
    }
    for family in families:
        test_family = vocabulary["results"][family]
        valid_family = valid_vocabulary["results"][family]
        fit_invariance["vocabulary"][family] = {
            "prototype_sha256_equal": test_family["selection"]["prototype_sha256"] == valid_family["selection"]["prototype_sha256"],
            "fit_score_sha256_equal": test_family["fit"]["score_sha256"] == valid_family["fit"]["score_sha256"],
        }
    if gate["base_occurrence"]["fit_score_sha256"] != valid_gate["base_occurrence"]["fit_score_sha256"]:
        raise AssertionError("terminal run changed fitted occurrence predictions")
    for mode in ("uniform__real", "uniform__assignment_shuffled"):
        test_fit = gate["results"][mode]["fit"]
        valid_fit = valid_gate["results"][mode]["fit"]
        fit_invariance["compact"][mode] = {
            "fit_score_sha256_equal": test_fit["score_sha256"] == valid_fit["score_sha256"],
            "fit_residual_sha256_equal": test_fit["residual_sha256"] == valid_fit["residual_sha256"],
        }
    invariant_flags = [fit_invariance["fit_indices_sha256_equal"]]
    for section in ("vocabulary", "compact"):
        for checks in fit_invariance[section].values():
            invariant_flags.extend(checks.values())
    if not all(invariant_flags):
        raise AssertionError("terminal run is not an exact refit of the frozen official-valid configuration")

    summary = {
        "protocol_id": "molhiv_realprototype_compact_official_test_summary_v1",
        "date": "2026-07-28",
        "status": "single_frozen_terminal_test_complete",
        "fit_policy": "exact official train only; official valid excluded from fitting",
        "model_policy": "seed-0 32+32 real prototypes, uniform gate, exact distance-1/2 compact relation; no post-test tuning",
        "n_test": int(len(labels)),
        "n_test_positive": int(labels.sum()),
        "test_indices_sha256": sha256(graph_indices),
        "frozen_fit_invariance_audit": fit_invariance,
        "official_valid_reference": {
            "occurrence_probability_ensemble_auc": valid_occurrence_auc,
            "uniform_compact_real_auc": valid_real_auc,
            "uniform_compact_assignment_shuffled_auc": valid_shuffled_auc,
            "compact_real_gain_over_occurrence": valid_real_auc - valid_occurrence_auc,
            "compact_real_minus_assignment_shuffled": valid_real_auc - valid_shuffled_auc,
        },
        "official_test": {
            "farthest_occurrence_auc": float(heldout[0]["auc"]),
            "scaffold_facility_occurrence_auc": float(heldout[1]["auc"]),
            "occurrence_probability_ensemble_auc": occurrence_auc,
            "uniform_compact_real_auc": real_auc,
            "uniform_compact_assignment_shuffled_auc": shuffled_auc,
            "compact_real_gain_over_occurrence": real_auc - occurrence_auc,
            "compact_real_minus_assignment_shuffled": real_auc - shuffled_auc,
        },
        "valid_to_test_change": {
            "occurrence_probability_ensemble": occurrence_auc - valid_occurrence_auc,
            "uniform_compact_real": real_auc - valid_real_auc,
            "uniform_compact_assignment_shuffled": shuffled_auc - valid_shuffled_auc,
        },
        "inputs": {
            "vocabulary_result": args.vocabulary_result,
            "gate_result": args.gate_result,
            "valid_vocabulary_result": args.valid_vocabulary_result,
            "valid_gate_result": args.valid_gate_result,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
