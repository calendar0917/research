"""Tests for the identity incremental-information audit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from tracks.ksvd.experiments.luyin16 import identity_incremental_information_audit as audit

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = REPO_ROOT / "tracks/ksvd/results/identity_incremental_information"


def test_collision_stats_synthetic() -> None:
    signatures = np.asarray([0, 0, 0, 1, 1, 2])
    identities = np.asarray([10, 10, 20, 30, 31, 40])
    stats = audit.collision_stats(signatures, identities)
    assert stats["unique_descriptor_signatures"] == 3
    assert stats["collision_mass"] == 5  # signature 0 (3) + signature 1 (2)
    assert stats["max_identity_multiplicity_per_signature"] == 2
    assert stats["H_identity_given_descriptor_nats"] > 0.0


def test_collision_stats_no_collision() -> None:
    signatures = np.asarray([0, 1, 2])
    identities = np.asarray([10, 20, 30])
    stats = audit.collision_stats(signatures, identities)
    assert stats["collision_mass"] == 0
    assert stats["H_identity_given_descriptor_nats"] == 0.0


def test_descriptor_signature_ids_exact_bytes() -> None:
    matrix = np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    ids, unique = audit._descriptor_signature_ids(matrix)
    assert unique == 2
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_fingerprint_bytes64_is_signed() -> None:
    value = audit._fingerprint_bytes64(b"anything")
    assert 0 <= value < (1 << 63)


def test_partial_spearman_removes_common_driver() -> None:
    rng = np.random.default_rng(0)
    driver = rng.normal(size=500)
    x = driver + 0.01 * rng.normal(size=500)
    y = driver + 0.01 * rng.normal(size=500)
    raw = float(np.corrcoef(x, y)[0, 1])
    partial = audit._partial_spearman(x, y, driver)
    assert raw > 0.9
    assert abs(partial) < 0.35 * raw


def test_frequency_statistics() -> None:
    stats = audit.frequency_statistics(Counter({1: 5, 2: 3, 3: 1}))
    assert stats["unique_types"] == 3
    assert stats["singleton_types"] == 1
    assert stats["le_5_types"] == 3
    assert stats["max_frequency"] == 5


def test_historical_to_corrected_multiplicity() -> None:
    historical = np.asarray([1, 1, 1, 2])
    corrected = np.asarray([10, 11, 12, 20])
    stats = audit.historical_to_corrected_multiplicity(historical, corrected)
    assert stats["historical_tokens"] == 2
    assert stats["max_corrected_classes_per_historical_token"] == 3
    assert stats["fraction_historical_tokens_split"] == 0.5


def test_historical_reconciliation_covers_required_experiments() -> None:
    names = " ".join(row["experiment"] for row in audit.HISTORICAL_RECONCILIATION)
    for required in (
        "tokenizer correctness",
        "fragmentation",
        "compositional patch sharing",
        "compact-v6",
        "SBCI",
        "recurrent Cell A",
        "MolHIV parameter attribution",
    ):
        assert required in names


def test_dataflow_artifact_matches_code_if_present() -> None:
    path = RESULT_DIR / "dataflow.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["zinc"]["total_trainable_params_at_historical_vocab"] == 99613
    assert payload["molhiv"]["typed_embedding_params"] == (26232 + 1) * 32
    assert payload["molhiv"]["total_params_frozen_checkpoint"] == 1076589


def test_molecule_signature_is_relabel_invariant() -> None:
    # A tiny GraphRecord-like object with two interchangeable patches.
    class _Patch:
        def __init__(self, descriptor, certificate):
            self.shell_descriptor = descriptor
            self.typed_certificate = certificate

    class _Record:
        def __init__(self, patches, pair_index, pair_relation, pair_bucket):
            self.patches = patches
            self.pair_index = pair_index
            self.pair_relation = pair_relation
            self.pair_bucket = pair_bucket
            self.global_context = np.zeros(3, dtype=np.float32)
            self.topology_features = None

    descriptor_a = np.asarray([1.0, 0.0], dtype=np.float32)
    descriptor_b = np.asarray([0.0, 1.0], dtype=np.float32)
    patch_a = _Patch(descriptor_a, b"a")
    patch_b = _Patch(descriptor_b, b"b")
    relation = np.asarray([[1.0]], dtype=np.float32)
    record_one = _Record(
        (patch_a, patch_b),
        np.asarray([[0], [1]], dtype=np.int64),
        relation,
        np.asarray([0]),
    )
    record_two = _Record(
        (patch_b, patch_a),
        np.asarray([[0], [1]], dtype=np.int64),
        relation,
        np.asarray([0]),
    )
    assert audit._molecule_signature(record_one, True) == audit._molecule_signature(record_two, True)
    assert audit._molecule_signature(record_one, False) == audit._molecule_signature(record_two, False)
