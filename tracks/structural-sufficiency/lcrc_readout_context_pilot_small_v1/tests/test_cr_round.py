"""Unit tests for round lcrc-readout-context-pilot-small-v1.

These tests are fast (no trajectory is trained) and assert the invariants that
make the formal runs interpretable: identity, frozen-file gates, the
budget-matched readout forms, the single-variable factor family, the ledger
accounting identity, and the declared protocol rules (including
``test_policy: always``).
"""

from __future__ import annotations

import numpy as np
import pytest

import cr_common as C


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_identity_constants() -> None:
    assert C.ROUND_ID == "lcrc-readout-context-pilot-small-v1"
    assert C.PROTOCOL_ID == C.ROUND_ID
    assert C.TRAIN_SEED == 0
    assert C.ARMS == ("A", "B", "C")
    assert C.DATASETS == ("ogbg-molesol", "ogbg-mollipo")
    assert C.CONTEXT_FIELDS == ("log1p_n_atoms", "ring_atom_fraction")
    assert C.CONTEXT_QUANTILES == (25.0, 75.0)
    assert len(set((C.INIT_STREAM_GLOBAL, C.INIT_STREAM_CONTEXT, C.INIT_STREAM_LOCAL))) == 3


def test_paths_point_into_this_round() -> None:
    assert C.CODE_DIR == C.ROUND_DIR / "code"
    assert C.RESULTS_DIR == C.ROUND_DIR / "results"
    assert C.ANALYSIS_DIR == C.ROUND_DIR / "analysis"
    for path in (C.SQL_CODE, C.PREP_CODE, C.DISSECT_CODE):
        assert path.is_dir(), path
    # read-only reuse must never point inside this round
    for path in (C.SQL_DIR, C.PREP_DIR, C.DISSECT_DIR):
        assert C.ROUND_DIR not in path.parents


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------


def test_trace_tolerance_scale() -> None:
    tol = C.trace_tolerance(np.asarray([0.0, 1.0, -1.0, 1e6]))
    # floor of 1 inside max(1, |x|): the absolute floor is 1.1e-5
    assert tol[0] == pytest.approx(1.1e-5)
    assert tol[1] == pytest.approx(1.1e-5)
    assert tol[2] == pytest.approx(1.1e-5)
    assert tol[3] == pytest.approx(1e-5 + 1e-6 * 1e6)
    assert C.trace_tolerance(0.0) == pytest.approx(1.1e-5)


def test_metrics_and_scaler() -> None:
    assert C.rmse(np.asarray([0.0, 0.0]), np.asarray([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))
    assert C.mae(np.asarray([0.0, 0.0]), np.asarray([3.0, 4.0])) == pytest.approx(3.5)
    scaler = C.target_scaler(np.asarray([1.0, 3.0, 5.0]))
    assert scaler["mean"] == pytest.approx(3.0)
    assert scaler["std"] == pytest.approx(np.std([1.0, 3.0, 5.0]))  # ddof=0
    assert scaler["n"] == 3


def test_sha_helpers_are_stable() -> None:
    assert C.sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    a = np.arange(6, dtype=np.int64)
    assert C.sha256_array(a) == C.sha256_array(a.copy())
    assert C.sha256_array(a) != C.sha256_array(a + 1)
    assert C.sha256_json({"b": 1, "a": 2}) == C.sha256_json({"a": 2, "b": 1})


# ---------------------------------------------------------------------------
# frozen files
# ---------------------------------------------------------------------------


def test_frozen_cache_gate() -> None:
    import cr_data as D

    for dataset in C.DATASETS:
        verified = D.verify_frozen_files(dataset)
        assert verified["all_match"], verified
        for name in (D.cache_filename(dataset), D.lipo_raw_filename(dataset)):
            entry = verified["files"][name]
            assert entry["match"], entry
            assert entry["actual_sha256"] == C.EXPECTED_CACHE_SHA256[name]
            assert entry["bytes"] > 0


# ---------------------------------------------------------------------------
# models / budget
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle_molesol():
    import cr_data as D

    return D.build_bundle("ogbg-molesol")


def test_arm_a_is_the_frozen_c1_head(bundle_molesol) -> None:
    import cr_checks

    result = cr_checks.check_armA_equivalence(bundle_molesol, __import__("cr_models").canonical_c1_state(
        bundle_molesol.spec, bundle_molesol.norm
    ))
    assert result["pass"], result


def test_budget_match(bundle_molesol) -> None:
    import cr_models as M

    specs = M.arm_specs(bundle_molesol)
    budget = specs["A"].readout_params_nominal
    assert budget == int(M.READOUT_BUDGET) == 15298
    for arm in ("B", "C"):
        delta = abs(specs[arm].readout_params_nominal - budget)
        # one hidden-width step of the arm's own readout formula
        assert delta <= 66, (arm, specs[arm].readout_params_nominal)


def test_three_arms_share_one_trunk_init(bundle_molesol) -> None:
    import cr_checks

    result = cr_checks.check_shared_init(
        bundle_molesol, __import__("cr_models").canonical_c1_state(bundle_molesol.spec, bundle_molesol.norm)
    )
    assert result["pass"], result


def test_single_variable_family_closed_form() -> None:
    import cr_checks

    result = cr_checks.check_single_variable_functions()
    assert result["pass"], result


def test_ledger_closure_on_a_fixed_state(bundle_molesol) -> None:
    import cr_checks

    result = cr_checks.check_ledger_closure(
        bundle_molesol, __import__("cr_models").canonical_c1_state(bundle_molesol.spec, bundle_molesol.norm)
    )
    assert result["pass"], result


# ---------------------------------------------------------------------------
# declared protocol rules
# ---------------------------------------------------------------------------


def _yaml(path):
    yaml = pytest.importorskip("yaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_protocol_declares_test_policy_always() -> None:
    protocol = _yaml(C.ROUND_DIR / "protocols" / f"{C.ROUND_ID}.yaml")
    assert protocol["id"] == C.ROUND_ID
    assert protocol["test_policy"] == "always"
    assert protocol["seed_semantics"]["seeds"] == [0]
    assert protocol["split"]["sizes"]["ogbg-molesol"] == {"train": 902, "valid": 113, "test": 113}
    assert protocol["split"]["sizes"]["ogbg-mollipo"] == {"train": 3360, "valid": 420, "test": 420}
    assert protocol["metric"]["ranking_metric"] == "valid_rmse"
    assert "Test RMSE" in protocol["validation_selection_policy"]["forbidden"]


def test_study_and_config_parse() -> None:
    study = _yaml(C.ROUND_DIR / "studies" / f"{C.ROUND_ID}.yaml")
    config = _yaml(C.ROUND_DIR / "configs" / f"{C.ROUND_ID}.yaml")
    assert study["id"] == C.ROUND_ID and study["protocol"] == C.ROUND_ID
    assert config["protocol_id"] == C.ROUND_ID
    assert config["seed"] == 0
    assert config["stage"] in ("prepare", "smoke", "dataset_worker", "aggregate")
    assert config["device"].startswith("cuda")


def test_recipe_is_frozen() -> None:
    import cr_train as T

    recipe = T.recipe_record()
    assert recipe["epochs"] == 360
    assert recipe["batch_size"] == 128
    assert recipe["eval_batch_size"] == 256
    assert recipe["peak_lr"] == pytest.approx(3e-4)
    assert recipe["min_lr"] == pytest.approx(1e-5)
    assert recipe["weight_decay"] == pytest.approx(1e-3)
    assert recipe["gradient_clip"] == pytest.approx(5.0)
    assert recipe["test_policy"] == "always"
    for flag in ("warmup_epochs", "early_stopping"):
        assert not recipe[flag]
    for flag in ("amp", "ema", "swa", "soup"):
        assert recipe[flag] is False
    assert recipe["selection_split"] == "valid"
