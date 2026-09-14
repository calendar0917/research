"""Tests for the ZINC capacity-decomposition winner official-test closure.

Static / unit tests only.  The official test split is **never** loaded here:
the tests exercise the frozen cell-A accounting, the pre-registered freeze
record and the exact pre-test refusal logic with a redirected result directory.
"""

from __future__ import annotations

import json

import pytest

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_test_closure as tc,
)


@pytest.fixture()
def _frozen_assets(monkeypatch, tmp_path):
    """Redirect the result dir and stub the (remote-only) state assets."""
    results = tmp_path / "results"
    states = tmp_path / "states"
    soup = tmp_path / "soup_states"
    states.mkdir(parents=True)
    soup.mkdir(parents=True)
    monkeypatch.setattr(tc, "RESULTS_DIR", results)
    monkeypatch.setattr(tc, "_state_dir", lambda: states)
    monkeypatch.setattr(tc, "_soup_dir", lambda: soup)
    for seed in (0, 1):
        tc._selection_path(seed).write_bytes(b"stub")
        tc._soup_path(seed).write_bytes(b"stub")
    return results


def test_01_selected_cell_is_the_frozen_cell_A():
    assert tc.SELECTED_CELL == "A"
    assert tc.SELECTED_H == 64
    assert tc.SELECTED_Q == 16
    assert tc.SELECTED_PATCH_HIDDEN == 64
    assert tc.SELECTED_GLOBAL_HIDDEN == 32
    assert tc.SELECTED_ROUNDS == 2


def test_02_cell_A_parameter_count():
    model = cd.build_cell("A", 0)
    assert cap._n_params(model) == tc.cd.EXPECTED_PARAMS["A"] == 85763


def test_03_selection_value_matches_the_capacity_decision():
    decision = json.loads((tc.SOURCE_DIR / "decision.json").read_text(encoding="utf-8"))
    assert decision["best_cell"] == "A"
    assert abs(float(decision["soup_2seed_mean"]["A"]) - tc.SELECTION_VALUE) <= 1.0e-12


def test_04_sanity_passes_and_reproduces_the_selection_value(_frozen_assets):
    payload = tc.sanity()
    assert payload["passed"] is True
    assert abs(payload["soup_2seed_mean"] - tc.SELECTION_VALUE) <= 1.0e-9


def test_05_frozen_asset_paths_point_into_the_source_directory():
    assert tc._selection_path(0).parent == tc.SOURCE_DIR / "states"
    assert tc._soup_path(1).parent == tc.SOURCE_DIR / "soup_states"
    assert tc._selection_path(0).name == "cell_A_seed0_selection_state.pt"
    assert tc._soup_path(1).name == "cell_A_seed1_top5_soup.pt"


def test_06_freeze_record_asserts_a_pre_test_freeze(_frozen_assets):
    record = tc.freeze()
    assert record["test_loaded_at_freeze_time"] is False
    assert record["test_status"] == "not yet loaded"
    assert record["selected_architecture"] == "cell_A"
    assert record["params"] == 85763
    assert record["selection_metric"] == "2-seed fixed Top-5 soup valid MAE"
    assert record["seeds"] == [0, 1]
    assert record["execution_regime"] == "deterministic A100"
    assert record["soup_rule"]["frozen_before_test"] is True
    assert len(record["selection_state_sha256"]) == 2
    assert len(record["soup_state_sha256"]) == 2
    assert (_frozen_assets / "architecture_freeze.json").exists()


def test_07_freeze_refuses_to_overwrite(_frozen_assets):
    tc.freeze()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        tc.freeze()


def test_08_freeze_refuses_after_unlock(_frozen_assets, monkeypatch):
    tc.freeze()
    (tc.RESULTS_DIR / "official_test_unlock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already unlocked"):
        tc.freeze()


def test_09_report_assembles_written_artifacts(_frozen_assets):
    tc.params()
    tc.sanity()
    tc.freeze()
    payload = tc.report()
    assert payload["architecture_freeze"]["selected_architecture"] == "cell_A"
    assert payload["official_test_results"] is None
    assert payload["parameter_accounting"]["matches_expected"] is True
