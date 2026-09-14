"""Static / unit tests for the seed-variance-source decomposition module.

No training, no model forward, official test never loaded.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_seed_variance_source as svs,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sh

RESULTS = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "compact_v4_recurrent_seed_variance_source"
)


def test_01_factorial_cells_are_a_balanced_2x2():
    assert set(svs.CELLS) == {"A", "B", "C", "D"}
    assert (svs.CELLS["A"]["init_seed"], svs.CELLS["A"]["shuffle_seed"]) == (0, 0)
    assert (svs.CELLS["B"]["init_seed"], svs.CELLS["B"]["shuffle_seed"]) == (0, 1)
    assert (svs.CELLS["C"]["init_seed"], svs.CELLS["C"]["shuffle_seed"]) == (1, 0)
    assert (svs.CELLS["D"]["init_seed"], svs.CELLS["D"]["shuffle_seed"]) == (1, 1)
    assert svs.CANONICAL_INIT_SHUFFLE == {
        "A": (0, 0),
        "B": (0, 1),
        "C": (1, 0),
        "D": (1, 1),
    }
    assert svs.DETERMINISTIC is True


def test_02_factorial_effect_arithmetic_matches_closed_form():
    mae = {"A": 0.146289, "B": 0.142657, "C": 0.142101, "D": 0.139254}
    out = svs._factorial_effects(mae)
    assert abs(out["init_effect_shuffle0"] - abs(mae["A"] - mae["C"])) < 1e-12
    assert abs(out["init_effect_shuffle1"] - abs(mae["B"] - mae["D"])) < 1e-12
    assert abs(out["shuffle_effect_init0"] - abs(mae["A"] - mae["B"])) < 1e-12
    assert abs(out["shuffle_effect_init1"] - abs(mae["C"] - mae["D"])) < 1e-12
    assert abs(
        out["mean_init_effect"]
        - 0.5 * (abs(mae["A"] - mae["C"]) + abs(mae["B"] - mae["D"]))
    ) < 1e-12
    assert abs(
        out["mean_shuffle_effect"]
        - 0.5 * (abs(mae["A"] - mae["B"]) + abs(mae["C"] - mae["D"]))
    ) < 1e-12
    inter = (mae["D"] - mae["C"]) - (mae["B"] - mae["A"])
    assert abs(out["two_way"]["interaction_signed"] - inter) < 1e-12
    assert abs(out["two_way"]["interaction_abs"] - abs(inter)) < 1e-12
    # A-D gap is the sum of the two (signed) main effects
    assert abs(
        (mae["A"] - mae["D"])
        - (out["two_way"]["init_main_effect"] + out["two_way"]["shuffle_main_effect"])
    ) < 1e-12


def test_03_case_gate_thresholds_are_two_thousandths():
    assert svs.EFFECT_GATE == 0.002
    assert svs.INTERACTION_GATE == 0.002
    assert svs._case_from(0.003, 0.003, 0.0005)[0] == "both"
    assert svs._case_from(0.003, 0.003, 0.003)[0] == "interaction"
    assert svs._case_from(0.005, 0.001, 0.0005)[0] == "init"
    assert svs._case_from(0.001, 0.005, 0.0005)[0] == "shuffle"
    assert svs._case_from(0.001, 0.001, 0.0005)[0] == "neither"
    # a large interaction is reported when neither factor dominates
    assert svs._case_from(0.003, 0.0025, 0.02)[0] == "interaction"
    assert svs._case_from(0.01, 0.0005, 0.02)[0] == "init"


def test_04_data_seed_is_default_inert_everywhere():
    for func in (sh.train_model, rec.train):
        assert inspect.signature(func).parameters["data_seed"].default is None
    assert inspect.signature(vd.run_training).parameters["data_seed"].default is None
    assert inspect.signature(vd.run_training).parameters["seed"].default is (
        inspect.Parameter.empty
    )


def test_05_determinism_sanity_records_non_reproducibility():
    path = RESULTS / "determinism_sanity.json"
    payload = json.loads(path.read_text())
    assert payload["separate_processes"] is True
    assert payload["bit_identical"] is False
    assert payload["official_test_loaded"] is False
    assert payload["max_valid_mae_diff"] > 0.0

    det = json.loads((RESULTS / "determinism_sanity_deterministic.json").read_text())
    assert det["bit_identical"] is True
    assert det["runs"]["detA"]["best_valid_mae"] == det["runs"]["detB"]["best_valid_mae"]
    assert det["concurrency_safe"] is True


def test_06_environment_fingerprint_records_deterministic_flag():
    fp = sh._environment_fingerprint()
    assert "deterministic_algorithms" in fp
    assert isinstance(fp["deterministic_algorithms"], bool)
    assert fp["torch_threads"] >= 1


def test_07_decomposition_and_decision_artifacts_are_consistent():
    dec = json.loads((RESULTS / "decomposition.json").read_text())
    assert dec["deterministic_algorithms"] is True
    assert dec["official_test_loaded"] is False
    det = svs._factorial_effects(
        {c: dec["cells"][c]["valid_mae"] for c in ("A", "B", "C", "D")}
    )
    assert abs(det["mean_init_effect"] - dec["mean_init_effect"]) < 1e-12
    assert abs(det["mean_shuffle_effect"] - dec["mean_shuffle_effect"]) < 1e-12
    assert (
        abs(det["two_way"]["interaction_abs"] - dec["two_way"]["interaction_abs"])
        < 1e-12
    )
    canon = svs._factorial_effects(
        {c: dec["canonical_factorial"]["cells"][c]["valid_mae"] for c in "ABCD"}
    )
    assert (
        abs(canon["mean_init_effect"] - dec["canonical_factorial"]["mean_init_effect"])
        < 1e-12
    )
    decision = json.loads((RESULTS / "decision.json").read_text())
    assert decision["case"] == "interaction"
    assert decision["deterministic_case"] == "both"
    assert decision["canonical_case"] == "interaction"
    assert decision["canonical_interaction_abs"] >= svs.INTERACTION_GATE
    assert decision["official_test_loaded"] is False


def test_08_soup_is_a_reproducible_stabilizer():
    summary = json.loads((RESULTS / "soup_summary.json").read_text())
    gains = np.array(list(summary["gains"].values()), dtype=float)
    assert np.all(gains > 0.0)
    assert summary["all_positive"] is True
    assert summary["reproducible_stabilizer"] is True
    assert abs(summary["mean_gain"] - gains.mean()) < 1e-12
    assert abs(summary["std_gain"] - gains.std()) < 1e-12
    assert abs(summary["min_gain"] - gains.min()) < 1e-12
    assert summary["official_test_loaded"] is False


def test_09_no_forbidden_knobs_in_module():
    names = {name for name, _ in inspect.getmembers(svs, inspect.isfunction)}
    assert not any("distill" in name.lower() for name in names)
    assert not any("teacher" in name.lower() for name in names)
    source = inspect.getsource(svs)
    # no search sweeps / alternative architectures
    for forbidden in ("build_q24", "build_pair_to_pair", "learning_rate"):
        assert forbidden not in source
    assert "3e-4" not in source
