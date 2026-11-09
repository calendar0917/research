"""Artifact-level audit tests: row alignment, replay gates, task marking and the
read-only guarantee for historical rounds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import lc_common as C  # noqa: E402
import lc_data as D  # noqa: E402

INVENTORY = C.ANALYSIS / "prediction_inventory.json"
ALIGNMENT = C.ANALYSIS / "alignment_report.json"
TASK_SETS = C.ANALYSIS / "task_sets.json"

HISTORICAL_DIRS = (
    "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1",
    "tracks/structural-sufficiency/lcrc_pcba_s3_v1",
    "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1",
    "tracks/structural-sufficiency/lcrc_pcba_rc_v1",
    "tracks/structural-sufficiency/lcrc_pcba_boc_v1",
    "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1",
    "tracks/structural-sufficiency/lcrc_posthoc_diagnosis_v1",
)

pytestmark = pytest.mark.skipif(
    not (INVENTORY.is_file() and ALIGNMENT.is_file() and TASK_SETS.is_file()),
    reason="run code/lc_inventory.py first",
)


@pytest.fixture(scope="module")
def inventory():
    return C.read_json(INVENTORY)


@pytest.fixture(scope="module")
def task_sets():
    return C.read_json(TASK_SETS)


def test_all_models_share_one_test_and_one_valid_row_order(inventory):
    official_test = C.sha256_int64(D.split_rows("test"))
    official_valid = C.sha256_int64(D.split_rows("valid"))
    for label in C.MODEL_ORDER:
        assert inventory["models"][label]["test"]["shape"] == [C.N_TEST_ROWS, C.N_TASKS]
        assert inventory["models"][label]["valid"]["shape"] == [C.N_VALID_ROWS, C.N_TASKS]
    assert inventory["row_alignment"]["test_predictions"]["all_models_share_rows"] is True
    assert inventory["row_alignment"]["valid_predictions"]["rows_sha256_int64"] == official_valid
    assert inventory["row_alignment"]["splits"]["test"]["task_rows_sha256_int64"] == official_test
    assert inventory["row_alignment"]["splits"]["test"]["elementwise_equal"] is True
    assert inventory["row_alignment"]["splits"]["valid"]["elementwise_equal"] is True


def test_historical_test_row_ids_equal_the_official_split():
    arr_boc = np.load(C.BOC_ROUND / "runs/freeze/test_predictions/test_row_ids.npy")
    arr_s3 = np.load(C.S3SEL_ROUND / "analysis/test_predictions/test_row_ids.npy")
    official = D.split_rows("test")
    assert np.array_equal(arr_boc.astype(np.int64), official)
    assert np.array_equal(arr_s3.astype(np.int64), official)


def test_test_labels_come_from_the_gate_and_match_inventory_stats(inventory):
    gate = D.official_test_labels()
    y = gate["labels"]
    assert y.shape == (C.N_TEST_ROWS, C.N_TASKS)
    finite = np.isfinite(y)
    assert int(finite.sum()) == inventory["label_stats"]["test"]["observed_entries"]
    assert int((y[finite] == 1.0).sum()) == inventory["label_stats"]["test"]["positive_entries"]
    assert gate["train_valid_alignment_verified"] is True
    assert gate["read_after_freeze"] is True


def test_replay_gates_passed(inventory):
    for label, m in inventory["models"].items():
        assert m["identity_status"] == "VERIFIED_ARTIFACT"
        assert m["test"]["ap_replay_abs_delta"] <= C.VALID_REPLAY_TOLERANCE
        assert m["valid"]["ap_replay_abs_delta"] <= C.VALID_REPLAY_TOLERANCE
        assert m["test"]["ap_abs_delta_official_vs_independent"] <= C.AP_AGREEMENT_TOLERANCE
        assert m["valid"]["ap_abs_delta_official_vs_independent"] <= C.AP_AGREEMENT_TOLERANCE


def test_task_sets_are_label_defined_and_consistent(task_sets):
    all_valid = set(task_sets["ALL_valid"])
    all_test = set(task_sets["ALL_test"])
    high = set(task_sets["HIGH_SUPPORT"])
    assert len(all_valid) == 127 and len(all_test) == 127
    assert high <= all_valid and high <= all_test
    for t in high:
        row = task_sets["per_task"][str(t)]
        assert row["valid_positive"] >= 20 and row["test_positive"] >= 20
        assert row["valid_negative"] > 0 and row["test_negative"] > 0
    low = set(task_sets["LOW_SUPPORT_test"])
    mid = set(task_sets["TEST_HIGH_BUT_VALID_LOW_SUPPORT"])
    assert high | low | mid == all_test
    assert not (high & low) and not (high & mid) and not (low & mid)
    # every task is marked, including unscorable ones
    assert set(task_sets["per_task"].keys()) == {str(t) for t in range(C.N_TASKS)}
    for row in task_sets["per_task"].values():
        assert row["test_stratum"] in C.SUPPORT_STRATA
        assert row["valid_stratum"] in C.SUPPORT_STRATA


def test_ap_is_none_for_a_task_without_both_classes():
    y = np.array([0.0, 1.0, np.nan, 0.0, 1.0])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert D.task_ap(y, scores) is not None
    assert D.task_ap(np.array([0.0, 0.0, np.nan]), np.array([0.1, 0.2, 0.3])) is None


def test_structure_support_rows_are_aligned_to_the_official_split():
    for split in ("valid", "test"):
        path = C.ANALYSIS / f"structure_support_{split}.csv"
        if not path.is_file():
            pytest.skip("structure support not built yet")
        df = pd.read_csv(path)
        rows = D.split_rows(split)
        assert np.array_equal(df["ogb_row"].to_numpy().astype(np.int64), rows)
        assert np.array_equal(df["split_index"].to_numpy().astype(np.int64), np.arange(rows.size))
        assert set(df["novelty_group"].unique()) <= set(C.NOVELTY_GROUPS)
        # no activity label may appear in the structural feature table
        assert not any("positive" in c or "label" in c for c in df.columns)


def test_novelty_thresholds_are_identical_for_both_splits():
    frames = {}
    for split in ("valid", "test"):
        path = C.ANALYSIS / f"structure_support_{split}.csv"
        if not path.is_file():
            pytest.skip("structure support not built yet")
        frames[split] = pd.read_csv(path)
    for split, df in frames.items():
        for value, group in zip(df["fraction_unseen"].to_numpy(), df["novelty_group"].to_numpy()):
            assert C.novelty_group(float(value)) == group


def test_historical_rounds_are_unmodified():
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *HISTORICAL_DIRS],
        cwd=C.REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = [line for line in out.stdout.splitlines() if line.strip()]
    assert dirty == [], f"historical rounds modified: {dirty}"


def test_round_does_not_modify_test_predictions():
    """The round's own analysis must not contain re-calibrated Test logits."""
    bad = [
        p.name
        for p in C.ANALYSIS.rglob("*.npy")
        if "test" in p.name.lower() or "logits" in p.name.lower()
    ]
    assert bad == [], f"unexpected Test/logit artifacts inside this round's analysis/: {bad}"


def test_checkpoint_and_test_logit_hashes_still_match_the_manifests(inventory):
    """Re-verify the SHA chains directly from disk (catches silent overwrites)."""
    for label in C.MODEL_ORDER:
        spec = C.MODELS[label]
        record = D.manifest_record(label)
        assert C.sha256_file(spec["checkpoint"]) == record["checkpoint_sha256"]
        assert C.sha256_file(D.test_logits_path(label)) == record["test_logits_sha256"]
        assert inventory["models"][label]["test"]["logits_sha256"] == record["test_logits_sha256"]


def test_topk_table_is_per_task_and_within_bounds():
    path = C.ANALYSIS / "topk_by_task.csv"
    if not path.is_file():
        pytest.skip("run code/lc_rank.py first")
    df = pd.read_csv(path)
    # one row per (pair, split, task) -- never a pooled cross-task candidate set
    assert not df.duplicated(subset=["pair", "split", "task"]).any()
    assert (df["k1"] >= 1).all() and (df["k1"] <= df["n_observed"]).all()
    assert (df["k5"] >= df["k1"]).all() and (df["k5"] <= df["n_observed"]).all()
    for col in ("pos_in_topk1_new", "pos_in_topk1_reference"):
        assert (df[col] <= df["k1"]).all()
    for col in ("pos_in_topk5_new", "pos_in_topk5_reference"):
        assert (df[col] <= df["k5"]).all()
    # mutual-exclusion bookkeeping closes per cell
    for k in ("k1", "k5"):
        total = (
            df[f"both_pos_{k}"]
            + df[f"only_new_pos_{k}"]
            + df[f"only_reference_pos_{k}"]
            + df[f"neither_pos_{k}"]
        )
        assert np.array_equal(total.to_numpy(), df["n_positive"].to_numpy())


def test_topk_never_crosses_molecule_blocks():
    """Top-K positions stay inside the per-assay vector they were computed on."""
    import lc_metrics as M

    rng = np.random.default_rng(3)
    block_a = rng.normal(size=40)
    block_b = rng.normal(size=60) + 100.0  # would dominate a naive pooled sort
    idx_a = {int(i) for i in M.top_k_indices(block_a, np.arange(40), 4)}
    idx_b = {int(i) for i in M.top_k_indices(block_b, np.arange(40, 100), 4)}
    assert idx_a <= set(range(40))
    # positions index block_b's own vector; the high offset never leaks in
    assert idx_b <= set(range(60))
    assert max(block_b[list(idx_b)]) > 100.0
