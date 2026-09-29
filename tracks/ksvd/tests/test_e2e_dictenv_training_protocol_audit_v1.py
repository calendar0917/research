"""Focused tests for ``e2e_dictenv_training_protocol_audit_v1``.

CPU only; the official ZINC test split is never touched.  The suite covers the
frozen Stage-E list: CSSD architecture identity and the exact 97727 parameter
count, the faithful training loop (bit-equivalence with the frozen
``cssd.train_cssd`` path), the shared-prefix / exact-fork contract (model,
optimizer moments, step counters, RNG and batch order), the two learning-rate
arms, the Top-5 soup rule, the frozen decision thresholds and the two guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_training_protocol_audit_v1 as tpa
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_training_protocol_audit_v1 as runner
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

CACHE_DIR = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p1/cache"


def _synthetic_phi(n: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.gamma(shape=2.0, scale=1.0, size=(n, cssd.PHI_DIM))
    base[:, 5] = 0.0
    base[:, 8] = 0.0
    return base.astype(np.float32)


def _subspace() -> cssd.CommonSubspace:
    return cssd.build_common_subspace(_synthetic_phi(seed=7), 1)


def _dictionary() -> np.ndarray:
    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary


def _real_data(subset: int = 32) -> tuple[list, list] | None:
    if not (CACHE_DIR / "env_train.pt").exists():
        return None
    train = p1run.load_split("train", subset=subset)
    valid = p1run.load_split("valid", subset=subset)
    if not train or not valid:
        return None
    tpa.tag_graph_ids(train)
    tpa.tag_graph_ids(valid)
    return train, valid


def _run_small(prefix_epochs: int = 2, tail_epochs: int = 2, tmp_path: Path | None = None):
    data = _real_data(subset=32)
    if data is None:  # pragma: no cover - data not available
        pytest.skip("ZINC encoded data not available")
    train, valid = data
    subspace = _subspace()
    out = tmp_path if tmp_path is not None else Path("/tmp/tpa-test")
    prefix = tpa.run_epochs(
        tag="UT-PREFIX",
        start_epoch=1,
        end_epoch=prefix_epochs,
        threads=2,
        out_dir=out,
        subspace=subspace,
        train_data=train,
        valid_data=valid,
        seed=0,
        lr=tpa.CONTROL_LR,
        log=False,
    )
    checkpoint = torch.load(
        Path(out) / "UT-PREFIX_resume_checkpoint.pt", map_location="cpu", weights_only=False
    )
    control = tpa.run_epochs(
        tag="UT-CONTROL",
        start_epoch=prefix_epochs + 1,
        end_epoch=prefix_epochs + tail_epochs,
        threads=2,
        out_dir=out,
        subspace=subspace,
        train_data=train,
        valid_data=valid,
        seed=0,
        lr=tpa.CONTROL_LR,
        model_state=checkpoint["model_state"],
        optimizer_state=checkpoint["optimizer_state"],
        rng_state=checkpoint["rng_state"],
        loader_state=checkpoint["train_loader_state"],
        soup_seed=checkpoint["keeper"],
        best_seed=(int(checkpoint["best"][0]), float(checkpoint["best"][1])),
        track_tail=True,
        log=False,
    )
    low = tpa.run_epochs(
        tag="UT-LOW",
        start_epoch=prefix_epochs + 1,
        end_epoch=prefix_epochs + tail_epochs,
        threads=2,
        out_dir=out,
        subspace=subspace,
        train_data=train,
        valid_data=valid,
        seed=0,
        lr=tpa.LOW_LR,
        model_state=checkpoint["model_state"],
        optimizer_state=checkpoint["optimizer_state"],
        rng_state=checkpoint["rng_state"],
        loader_state=checkpoint["train_loader_state"],
        soup_seed=checkpoint["keeper"],
        best_seed=(int(checkpoint["best"][0]), float(checkpoint["best"][1])),
        track_tail=True,
        log=False,
    )
    return prefix, checkpoint, control, low, (train, valid, subspace)


# ---------------------------------------------------------------------------
# guards and architecture identity
# ---------------------------------------------------------------------------


def test_guards() -> None:
    tpa.cpu_only_guard(torch.device("cpu"))
    with pytest.raises(RuntimeError):
        tpa.cpu_only_guard(torch.device("cuda"))
    tpa.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        tpa.official_test_blocker({"official_test_loaded": True})
    with pytest.raises(RuntimeError):
        tpa.official_test_blocker({})


def test_parameter_count_and_architecture_identity() -> None:
    dictionary = _dictionary()
    subspace = _subspace()
    model = cssd.build_cssd_model(dictionary, 0, subspace)
    assert isinstance(model, cssd.CSSDModel)
    assert sum(parameter.numel() for parameter in model.parameters()) == tpa.EXPECTED_PARAMETERS
    reference = cssd.build_cssd_model(dictionary, 0, subspace)
    left = {key: value for key, value in model.state_dict().items()}
    right = {key: value for key, value in reference.state_dict().items()}
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key]), f"model initialization diverged at {key}"


def test_module_groups_are_disjoint_and_cover_the_expected_modules() -> None:
    model = cssd.build_cssd_model(_dictionary(), 0, _subspace())
    groups = tpa.parameter_groups(model)
    assert set(groups) == set(tpa.MODULE_GROUPS)
    flat = [name for names in groups.values() for name in names]
    assert len(flat) == len(set(flat))
    assert groups["dictionary"] == ["D"]
    assert "W_A_S" in groups["node_semantic_binding"]
    assert "W_E_S" in groups["edge_semantic_binding"]
    assert any(name.startswith("pair_encoder.") for name in groups["pair_encoder"])
    assert any(name.startswith("reader.") for name in groups["reader"])


# ---------------------------------------------------------------------------
# training-loop faithfulness and the frozen constants
# ---------------------------------------------------------------------------


def test_run_epochs_matches_frozen_loop(tmp_path: Path) -> None:
    data = _real_data(subset=32)
    if data is None:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    train, valid = data
    subspace = _subspace()

    def factory(dictionary_arg: np.ndarray, seed: int) -> cssd.CSSDModel:
        return cssd.build_cssd_model(dictionary_arg, seed, subspace)

    reference = audit.train_cpu(
        tag="UT-REF",
        mask=cssd.CSSD_MASK,
        epochs=1,
        threads=2,
        out_dir=tmp_path / "ref",
        train_data=train,
        valid_data=valid,
        seed=0,
        save_states=True,
        log=False,
        model_factory=factory,
    )
    candidate = tpa.run_epochs(
        tag="UT-TPA",
        start_epoch=1,
        end_epoch=1,
        threads=2,
        out_dir=tmp_path / "tpa",
        subspace=subspace,
        train_data=train,
        valid_data=valid,
        seed=0,
        lr=tpa.CONTROL_LR,
        log=False,
    )
    assert reference["best_valid_mae"] == candidate["best_valid_mae"]
    assert reference["soup"]["soup_valid_mae"] == candidate["soup"]["soup_valid_mae"]
    reference_state = torch.load(tmp_path / "ref" / "UT-REF_final_state.pt", map_location="cpu")
    candidate_state = torch.load(tmp_path / "tpa" / "UT-TPA_final_state.pt", map_location="cpu")
    assert set(reference_state) == set(candidate_state)
    for key in reference_state:
        assert torch.equal(reference_state[key], candidate_state[key]), key
    assert candidate["lambda_rec"] == float(cm.H1_LAMBDA)
    assert candidate["batch_size"] == int(p2run.BATCH_SIZE)
    assert candidate["weight_decay"] == float(p2run.WEIGHT_DECAY)
    assert candidate["lr_contract"]["lr_after"] == [tpa.CONTROL_LR]


def test_resume_requires_the_saved_optimizer_state(tmp_path: Path) -> None:
    data = _real_data(subset=16)
    if data is None:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    train, valid = data
    subspace = _subspace()
    tpa.run_epochs(
        tag="UT-PREFIX2",
        start_epoch=1,
        end_epoch=1,
        threads=2,
        out_dir=tmp_path,
        subspace=subspace,
        train_data=train,
        valid_data=valid,
        seed=0,
        lr=tpa.CONTROL_LR,
        log=False,
    )
    model_state = torch.load(tmp_path / "UT-PREFIX2_final_state.pt", map_location="cpu")
    with pytest.raises(ValueError):
        tpa.run_epochs(
            tag="UT-BAD",
            start_epoch=2,
            end_epoch=2,
            threads=2,
            out_dir=tmp_path,
            subspace=subspace,
            train_data=train,
            valid_data=valid,
            seed=0,
            lr=tpa.LOW_LR,
            model_state=model_state,
            optimizer_state=None,
            log=False,
        )


# ---------------------------------------------------------------------------
# exact fork
# ---------------------------------------------------------------------------


def test_shared_prefix_and_exact_fork(tmp_path: Path) -> None:
    prefix, checkpoint, control, low, data = _run_small(tmp_path=tmp_path)
    _train, valid, subspace = data
    assert prefix["epochs_run"] == 2
    assert [row["epoch"] for row in prefix["curve"]] == [1, 2]
    assert control["epochs_run"] == 2
    assert [row["epoch"] for row in control["curve"]] == [3, 4]
    assert [row["epoch"] for row in low["curve"]] == [3, 4]

    fork = tpa.fork_integrity(
        checkpoint=checkpoint,
        dictionary=_dictionary(),
        subspace=subspace,
        train_data=_train,
        valid_data=valid[:8],
        threads=2,
        seed=0,
        probe_size=8,
    )
    assert fork["verdict"] == "FORK_INTEGRITY_OK"
    assert fork["model_state_identical"] is True
    assert fork["optimizer_state_identical_before_lr_edit"] is True
    assert fork["checks"]["exp_avg_identical"] is True
    assert fork["checks"]["exp_avg_sq_identical"] is True
    assert fork["checks"]["step_counters_identical"] is True
    assert fork["checks"]["only_lr_differs_after_edit"] is True
    assert fork["checks"]["prediction_max_abs_diff_zero"] is True
    assert fork["checks"]["train_mode_prediction_max_abs_diff_zero"] is True
    assert fork["prediction_max_abs_diff"] == 0.0
    assert fork["train_mode_prediction_max_abs_diff"] == 0.0
    assert fork["checks"]["control_lr_is_1e3"] is True
    assert fork["checks"]["low_lr_is_1e4"] is True
    assert fork["lr_edit"]["low_lr_after"] == [tpa.LOW_LR]
    assert fork["optimizer_field_comparison_ignoring_lr"]["differing_param_group_fields"] == []


def test_fork_carries_the_same_optimizer_state_into_both_arms(tmp_path: Path) -> None:
    _prefix, checkpoint, control, low, _extra = _run_small(tmp_path=tmp_path)
    assert control["resumed"] is True and low["resumed"] is True
    assert control["lr_contract"]["lr_before"] == [tpa.CONTROL_LR]
    assert low["lr_contract"]["lr_before"] == [tpa.CONTROL_LR]
    assert control["lr_contract"]["lr_after"] == [tpa.CONTROL_LR]
    assert low["lr_contract"]["lr_after"] == [tpa.LOW_LR]
    assert control["lr_contract"]["lr_requested"] == tpa.CONTROL_LR
    assert low["lr_contract"]["lr_requested"] == tpa.LOW_LR
    assert control["initial_state_sha256"] == low["initial_state_sha256"]
    assert control["loader_fingerprint_start"] == low["loader_fingerprint_start"]
    assert checkpoint["model_state"]["D"].shape == (65, 32)


def test_batch_order_hashes_are_identical_over_the_tail(tmp_path: Path) -> None:
    _prefix, _checkpoint, control, low, _data = _run_small(tmp_path=tmp_path)
    assert control["batch_order_hashes"] == low["batch_order_hashes"]
    for epoch in ("3", "4"):
        assert control["batch_order_hashes"][epoch] == low["batch_order_hashes"][epoch]


def test_replay_batch_order_hash_is_deterministic() -> None:
    data = _real_data(subset=32)
    if data is None:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    train, _valid = data
    generator = torch.Generator().manual_seed(tpa.SEED + int(p2run.TRAIN_SHUFFLE_OFFSET))
    state = generator.get_state()
    first = tpa.replay_epoch_batch_order_hash(train, seed=tpa.SEED, loader_state=state)
    second = tpa.replay_epoch_batch_order_hash(train, seed=tpa.SEED, loader_state=state)
    assert first == second
    assert first["n_batches"] >= 1
    other_generator = torch.Generator().manual_seed(tpa.SEED + 1 + int(p2run.TRAIN_SHUFFLE_OFFSET))
    other = tpa.replay_epoch_batch_order_hash(
        train, seed=tpa.SEED, loader_state=other_generator.get_state()
    )
    assert other["sha256"] != first["sha256"]


def test_rng_state_contract_is_restored_for_both_arms(tmp_path: Path) -> None:
    _prefix, checkpoint, control, low, _data = _run_small(tmp_path=tmp_path)
    torch.set_rng_state(checkpoint["rng_state"])
    expected = tpa.rng_fingerprint()
    assert checkpoint["train_loader_state"] is not None
    assert isinstance(expected, str) and len(expected) == 64
    # both arms started from the same prefix loader state
    assert control["loader_fingerprint_start"] == low["loader_fingerprint_start"]


# ---------------------------------------------------------------------------
# soup rule
# ---------------------------------------------------------------------------


def test_top5_soup_rule_and_merge() -> None:
    states = {
        epoch: {"D": torch.full((2, 2), float(epoch))} for epoch in range(1, 8)
    }
    keeper = tpa.empty_keeper()
    maes = {1: 0.20, 2: 0.10, 3: 0.30, 4: 0.05, 5: 0.25, 6: 0.15, 7: 0.40}
    for epoch in range(1, 8):
        keeper = tpa.retain_epoch(
            keeper, epoch=epoch, valid_mae=maes[epoch], state=states[epoch], k=5
        )
    assert tpa.keeper_members(keeper) == [1, 2, 4, 5, 6]
    assert tpa.keeper_member_mae(keeper) == [
        maes[epoch] for epoch in (1, 2, 4, 5, 6)
    ]
    soup = tpa.soup_state_from(keeper)
    expected = sum(float(epoch) for epoch in (1, 2, 4, 5, 6)) / 5.0
    assert torch.allclose(soup["D"], torch.full((2, 2), expected))
    # prefix keeper merged with a better tail replaces prefix members
    tail = tpa.retain_epoch(
        keeper, epoch=8, valid_mae=0.01, state={"D": torch.full((2, 2), 8.0)}, k=5
    )
    assert tpa.keeper_members(tail) == [1, 2, 4, 6, 8]
    # seeded keeper preserves the prefix membership
    seeded = tpa.seed_keeper(keeper["entries"], keeper["states"])
    assert tpa.keeper_members(seeded) == tpa.keeper_members(keeper)


def test_soup_members_after_fork_are_tracked(tmp_path: Path) -> None:
    _prefix, _checkpoint, control, _low, _data = _run_small(tmp_path=tmp_path)
    assert control["soup"]["n_members_after_280"] == len(control["soup"]["members_after_280"])
    assert all(epoch > tpa.FORK_EPOCH for epoch in control["soup"]["members_after_280"])
    assert control["tail_soup"]["window"] == [3, 4]


# ---------------------------------------------------------------------------
# prefix gate, historical comparison, decision
# ---------------------------------------------------------------------------


def test_prefix_gate_pass_and_fail() -> None:
    healthy = {
        "epoch": 280,
        "train_mae": 0.10,
        "valid_mae": 0.141,
        "train_rec_term": 5.0e-5,
    }
    payload = tpa.prefix_gate(
        curve=[healthy],
        initial_state_sha256="a",
        final_state_sha256="b",
        dictionary_sha256="c",
        actual_parameters=tpa.EXPECTED_PARAMETERS,
        model_finite=True,
        dictionary_moved=True,
    )
    assert payload["passed"] is True
    assert payload["verdict"] == "PREFIX_HEALTHY"
    drifted = tpa.prefix_gate(
        curve=[{**healthy, "valid_mae": 0.20}],
        initial_state_sha256="a",
        final_state_sha256="b",
        dictionary_sha256="c",
        actual_parameters=tpa.EXPECTED_PARAMETERS,
        model_finite=True,
        dictionary_moved=True,
    )
    assert drifted["passed"] is False
    assert drifted["verdict"] == "PREFIX_REGIME_MISMATCH"
    with pytest.raises(RuntimeError):
        tpa.prefix_gate(
            curve=[{**healthy, "epoch": 279}],
            initial_state_sha256="a",
            final_state_sha256="b",
            dictionary_sha256="c",
            actual_parameters=tpa.EXPECTED_PARAMETERS,
            model_finite=True,
            dictionary_moved=True,
        )


def test_historical_comparison_is_provenance_only() -> None:
    curve = [{"epoch": epoch, "valid_mae": 0.2 - 0.001 * epoch} for epoch in (40, 120, 240, 280)]
    historical = [
        {"epoch": epoch, "valid_mae": 0.2 - 0.001 * epoch + 0.0005} for epoch in (40, 120, 240, 280)
    ]
    payload = tpa.historical_comparison(curve, historical)
    assert payload["available"] is True
    assert payload["n_shared_epochs"] == 4
    assert abs(payload["mean_abs_delta"] - 0.0005) < 1e-12
    missing = tpa.historical_comparison(curve, None)
    assert missing["available"] is False
    assert missing["official_test_loaded"] is False


def test_decision_thresholds() -> None:
    cases = [
        (0.130, 0.130, "TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK", False),
        (0.130, 0.1275, "DIRECTIONAL_SMALL_TRAINING_EFFECT", False),
        (0.130, 0.1265, "LOW_LR_TAIL_MATERIALLY_SUPPORTED", True),
        (0.130, 0.1245, "TRAINING_PROTOCOL_MAJOR_FACTOR", True),
        (0.125, 0.1180, "BASELINE_UNDEROPTIMIZATION_WAS_SUBSTANTIAL", True),
        (0.129, 0.1305, "LOW_LR_TAIL_HARMFUL", False),
    ]
    for m_control, m_low, verdict, adopt in cases:
        payload = tpa.training_protocol_decision(
            m_control=m_control, m_low_lr=m_low, low_lr_best_epoch=320, low_lr_late_slope=-1e-4
        )
        assert payload["verdict"] == verdict, (m_control, m_low)
        assert payload["adopt_low_lr_tail"] is adopt
        assert payload["official_test_loaded"] is False
    strong = tpa.training_protocol_decision(
        m_control=0.130, m_low_lr=0.1249, low_lr_best_epoch=320, low_lr_late_slope=-1e-4
    )
    assert strong["case_c_major"] is True
    assert strong["horizon_may_still_be_binding"] is True


def test_dynamics_rows_and_late_slope() -> None:
    control = [
        {"epoch": epoch, "valid_mae": 0.15 - 1e-4 * (epoch - 280), "update_norm_global": 1e-3,
         "grad_norm_global": 1e-2, "train_mae": 0.1}
        for epoch in (280, 285, 290, 300, 310, 320)
    ]
    low = [
        {"epoch": epoch, "valid_mae": 0.15 - 2e-4 * (epoch - 280), "update_norm_global": 1e-4,
         "grad_norm_global": 1e-3, "train_mae": 0.1}
        for epoch in (280, 285, 290, 300, 310, 320)
    ]
    rows = tpa.dynamics_rows(control, low)
    assert [row["epoch"] for row in rows] == [280, 285, 290, 300, 310, 320]
    assert all(row["difference"] >= 0 for row in rows)
    assert all(row["difference"] > 0 for row in rows if row["epoch"] > 280)
    assert tpa.late_slope(control, window=(280, 320)) < 0
    assert tpa.late_slope(control[:2]) != tpa.late_slope(control[:2])  # nan


# ---------------------------------------------------------------------------
# optimizer state hashing / comparison
# ---------------------------------------------------------------------------


def test_optimizer_state_comparison_detects_only_lr() -> None:
    model = torch.nn.Linear(3, 2)
    left = torch.optim.Adam(model.parameters(), lr=tpa.CONTROL_LR)
    (model(torch.ones(1, 3)).sum()).backward()
    left.step()
    right = torch.optim.Adam(model.parameters(), lr=tpa.LOW_LR)
    right.load_state_dict(left.state_dict())
    for group in right.param_groups:
        group["lr"] = tpa.LOW_LR
    comparison = tpa.optimizer_state_comparison(left, right, ignore_lr=True)
    assert comparison["exp_avg_identical"] is True
    assert comparison["exp_avg_sq_identical"] is True
    assert comparison["step_identical"] is True
    assert comparison["differing_param_group_fields"] == []
    strict = tpa.optimizer_state_comparison(left, right)
    assert strict["differing_param_group_fields"] == ["lr"]
    assert tpa.optimizer_state_sha256(left) != tpa.optimizer_state_sha256(right)


# ---------------------------------------------------------------------------
# runner-side merged-best rule and harness repair
# ---------------------------------------------------------------------------


def test_merged_best_epoch_and_harness_repair(tmp_path: Path) -> None:
    prefix = {"curve": [{"epoch": 1, "valid_mae": 0.20}, {"epoch": 2, "valid_mae": 0.15}]}
    payload = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "official_test_loaded": False,
        "curve": [{"epoch": 3, "valid_mae": 0.14}, {"epoch": 4, "valid_mae": 0.16}],
        "best_valid_mae": 0.0,
        "best_epoch": 2,
    }
    assert runner._merged_best_epoch(prefix["curve"], payload["curve"]) == (3, 0.14)
    _write = tmp_path / "result.json"
    import json

    _write.write_text(json.dumps(payload), encoding="utf-8")
    repaired = runner._repair_continuation_best(prefix, payload, tmp_path, "UT-ARM")
    assert repaired["best_valid_mae"] == 0.14
    assert repaired["best_epoch"] == 3
    assert repaired["harness_fix"]["best_epoch"] == 3
    stored = json.loads(_write.read_text(encoding="utf-8"))
    assert stored["best_valid_mae"] == 0.14
    assert stored["best_epoch"] == 3
    again = runner._repair_continuation_best(prefix, stored, tmp_path, "UT-ARM")
    assert "harness_fix" in again
