"""Fast correctness tests for cscl-bjoc-recipe-tuning-v1.

No ZINC data, no checkpoints, no GPU.  Covers the frozen candidate registry,
the cosine/constant schedule contract, the batch-order prefix identity, the
soup window, the parameter budget, the pre-registered screening /
confirmation gates and the no-test / no-frozen-module-mutation discipline.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import run_bjoc_recipe_tuning_v1 as rt  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402

ZERO_MEAN = np.zeros(24, dtype=np.float32)
ONE_SCALE = np.ones(24, dtype=np.float32)


def _cfg():
    return yaml.safe_load(rt.CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# frozen candidate registry
# ---------------------------------------------------------------------------


def test_candidate_registry_is_exactly_the_five_frozen_recipes():
    assert set(rt.CANDIDATES) == {"R0", "R1", "R2", "R3", "R4"}
    assert set(rt.SCREENING_CANDIDATES) == {"R1", "R2", "R3", "R4"}
    assert [rt.CANDIDATES[k]["epochs"] for k in ("R0", "R1", "R4")] == [240, 240, 240]
    assert [rt.CANDIDATES[k]["epochs"] for k in ("R2", "R3")] == [360, 360]
    assert rt.CANDIDATES["R1"]["schedule"] == "cosine"
    assert rt.CANDIDATES["R3"]["schedule"] == "cosine"
    assert rt.CANDIDATES["R0"]["schedule"] == rt.CANDIDATES["R2"]["schedule"] == rt.CANDIDATES["R4"]["schedule"] == "constant"
    assert rt.CANDIDATES["R0"]["weight_decay"] == rt.CANDIDATES["R1"]["weight_decay"] == rt.CANDIDATES["R2"]["weight_decay"] == rt.CANDIDATES["R3"]["weight_decay"] == 1.0e-5
    assert rt.CANDIDATES["R4"]["weight_decay"] == 1.0e-4
    assert rt.CANDIDATES["R0"]["reuse"] == "historical_P_checkpoint"
    assert all(rt.CANDIDATES[k]["reuse"] is None for k in ("R1", "R2", "R3", "R4"))
    assert rt.STEPS_PER_EPOCH == 79
    assert rt.REF_UPDATES == 18960
    assert rt.CANDIDATES["R2"]["epochs"] * rt.STEPS_PER_EPOCH == 28440


def test_config_and_code_agree():
    cfg = _cfg()
    rt._check_code_matches_config(cfg)
    assert cfg["data"]["test"] == "forbidden"
    assert cfg["official_test"] == "never_read"
    assert cfg["evaluation"]["no_test_access"] is True
    assert cfg["frozen"]["expected_parameters"] == 413505
    assert cfg["frozen"]["steps_per_epoch"] == rt.STEPS_PER_EPOCH
    assert cfg["frozen"]["ref_epochs"] == rt.REF_EPOCHS


def test_recipe_of_rejects_unknown_candidate():
    with pytest.raises(SystemExit):
        rt.recipe_of("R9")


# ---------------------------------------------------------------------------
# schedule contract
# ---------------------------------------------------------------------------


def test_cosine_endpoints_and_monotonicity():
    lr0, lr_min, total = 1.0e-3, 1.0e-5, 28440
    values = [rt.cosine_lr(k, total, lr0, lr_min) for k in range(total + 1)]
    assert abs(values[0] - lr0) < 1e-15
    assert abs(values[-1] - lr_min) < 1e-15
    diffs = np.diff(values)
    assert (diffs <= 1e-18).all()


def test_cosine_matches_torch_cosine_annealing_lr():
    for total in (240 * 79, 360 * 79):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([param], lr=1.0e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1.0e-5)
        max_dev = 0.0
        for step in range(total):
            actual = opt.param_groups[0]["lr"]
            max_dev = max(max_dev, abs(actual - rt.cosine_lr(step, total, 1.0e-3, 1.0e-5)))
            opt.step()
            sched.step()
        assert max_dev < 1e-15
        assert abs(opt.param_groups[0]["lr"] - 1.0e-5) < 1e-15


def test_constant_schedule_stays_at_lr():
    recipe = rt.CANDIDATES["R4"]
    for step in (0, 1, 18959, 99999):
        assert rt.schedule_lr(recipe, step, 18960) == 1.0e-3


def test_schedule_lr_rejects_unknown_schedule():
    with pytest.raises(SystemExit):
        rt.schedule_lr({"schedule": "linear", "lr": 1e-3, "lr_min": None}, 0, 10)


# ---------------------------------------------------------------------------
# batch order / soup
# ---------------------------------------------------------------------------


def _accumulate_order_hash(seed: int, n: int, batch_size: int, epochs: int) -> str:
    gen = torch.Generator().manual_seed(int(seed) + rt.TRAIN_SHUFFLE_OFFSET)
    h = hashlib.sha256()
    for _ in range(epochs):
        for indices in f10k._shuffle_batches(n, batch_size, gen):
            h.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
    return h.hexdigest()


def test_batch_order_matches_f10k_and_prefix_is_stable():
    seed, n = 3, 1000
    h240 = _accumulate_order_hash(seed, n, 128, 240)
    assert h240 == f10k.batch_order_hash(seed, n, 128, 240)
    # the first 240 epochs of a 360-epoch run use the same generator draws
    gen = torch.Generator().manual_seed(seed + rt.TRAIN_SHUFFLE_OFFSET)
    h_prefix = hashlib.sha256()
    for epoch in range(1, 361):
        for indices in f10k._shuffle_batches(n, 128, gen):
            if epoch <= 240:
                h_prefix.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
    assert h_prefix.hexdigest() == h240
    assert h240 != f10k.batch_order_hash(seed, n, 128, 360)


def test_last_soup_uses_the_final_five_epochs_of_the_declared_length():
    states = [{"w": torch.tensor([float(i), 2.0 * float(i)])} for i in range(5)]
    members, soup = f10k._last_soup(states, 360)
    assert members == [356, 357, 358, 359, 360]
    assert torch.allclose(soup["w"], torch.tensor([2.0, 4.0]))
    members240, _ = f10k._last_soup(states, 240)
    assert members240 == [236, 237, 238, 239, 240]


# ---------------------------------------------------------------------------
# model budget / invariance
# ---------------------------------------------------------------------------


def test_model_budget_and_matched_init_are_candidate_independent():
    norm = {"global_mean": ZERO_MEAN, "global_scale": ONE_SCALE}
    model_a = f10k._make_downstream_model(0, norm, "cpu")
    model_b = f10k._make_downstream_model(0, norm, "cpu")
    audit = src.parameter_audit(src.BJOCConfig(), model_a)
    assert audit["parameter_exact"]
    assert audit["actual_parameters"] == 413505
    assert model_a.organization_mode == src.NULL
    assert src.forbidden_module_scan(model_a) == []
    # the candidate recipe never touches the model: same seed -> identical init
    assert src.state_dict_hash(model_a) == src.state_dict_hash(model_b)
    other = f10k._make_downstream_model(1, norm, "cpu")
    assert src.state_dict_hash(model_a) != src.state_dict_hash(other)


# ---------------------------------------------------------------------------
# pre-registered gates
# ---------------------------------------------------------------------------


def _contrast_stub(delta=-0.004, ordinary=-0.001, without=-0.003, ci_hi=-0.001, per_seed=None):
    return {
        "delta_seed_mean": delta,
        "per_seed_delta": per_seed if per_seed is not None else {"seed0": delta, "seed1": delta},
        "ordinary_y_gt_-3": {"delta": ordinary},
        "tail_y_le_-3": {"delta": 0.01},
        "delta_top10_removed": -0.003,
        "extreme_row": {"delta_without_row": without},
        "bootstrap": {"ci_hi": ci_hi, "ci_lo": -0.006},
    }


def test_screening_gate_thresholds():
    assert all(rt.screen_gates(_contrast_stub(), True).values())
    assert not rt.screen_gates(_contrast_stub(delta=-0.0025), True)["seed0_mae_delta_le_-0.003"]
    assert not rt.screen_gates(_contrast_stub(ordinary=0.0001), True)["ordinary_not_worse"]
    assert not rt.screen_gates(_contrast_stub(without=-0.001), True)["extreme_row_excluded_delta_le_-0.002"]
    assert not rt.screen_gates(_contrast_stub(), False)["numerics_and_audits_ok"]


def test_confirmation_gate_and_classification():
    good = _contrast_stub(per_seed={"seed0": -0.005, "seed1": -0.004})
    conditions = rt.confirmation_conditions(good, True)
    assert all(conditions.values())
    assert rt.classify_confirmation(conditions, good) == "RECIPE_GAIN_CANDIDATE"
    ci_cross = _contrast_stub(per_seed={"seed0": -0.005, "seed1": -0.004}, ci_hi=0.001)
    assert rt.classify_confirmation(rt.confirmation_conditions(ci_cross, True), ci_cross) == "WEAK_OR_INCONCLUSIVE"
    one_seed_off = _contrast_stub(per_seed={"seed0": -0.005, "seed1": 0.001}, delta=-0.002)
    assert rt.classify_confirmation(rt.confirmation_conditions(one_seed_off, True), one_seed_off) == "WEAK_OR_INCONCLUSIVE"
    negative = _contrast_stub(delta=0.0005, per_seed={"seed0": 0.0004, "seed1": 0.0006})
    assert rt.classify_confirmation(rt.confirmation_conditions(negative, True), negative) == "NO_ROBUST_RECIPE_GAIN"


def test_confirmation_requires_numeric_audit():
    good = _contrast_stub(per_seed={"seed0": -0.005, "seed1": -0.004})
    assert not all(rt.confirmation_conditions(good, False).values())


# ---------------------------------------------------------------------------
# pair statistics and paths
# ---------------------------------------------------------------------------


def test_contrast_statistics_and_bootstrap_determinism():
    rng = np.random.default_rng(0)
    y = rng.normal(-1.0, 1.5, rt.N_VALID)
    ref = y + rng.normal(0.0, 0.1, rt.N_VALID)
    new = y + rng.normal(-0.01, 0.1, rt.N_VALID)
    out = rt._contrast([new], [ref], y, "X-R0", seeds=[0])
    delta = np.abs(new - y) - np.abs(ref - y)
    assert abs(out["delta_seed_mean"] - delta.mean()) < 1e-12
    assert out["per_seed_delta"]["seed0"] == pytest.approx(delta.mean())
    assert out["ordinary_y_gt_-3"]["n"] + out["tail_y_le_-3"]["n"] == rt.N_VALID
    assert out["extreme_row"]["delta_without_row"] == pytest.approx(np.delete(delta, rt.EXTREME_VALID_ROW).mean())
    b1 = f10k.paired_bootstrap(delta)
    b2 = f10k.paired_bootstrap(delta)
    assert b1 == b2
    assert b1["n_resamples"] == 5000


def test_prediction_path_conventions():
    assert rt.prediction_path("R00") == rt.F10K / "predictions" / "valid_P0.npz"
    assert rt.prediction_path("P1") == rt.F10K / "predictions" / "valid_P1.npz"
    assert rt.prediction_path("R10") == rt.PRED_DIR / "valid_R10.npz"
    assert rt.prediction_path("R31") == rt.PRED_DIR / "valid_R31.npz"
    assert rt._checkpoint_path("R0", 1) == rt.F10K / "soup_P_seed1.pt"
    assert rt._checkpoint_path("R2", 0) == rt.OUT / "soup_R2_seed0.pt"


# ---------------------------------------------------------------------------
# discipline scans
# ---------------------------------------------------------------------------


def test_runner_has_no_official_test_path():
    text = (_CODE_DIR / "run_bjoc_recipe_tuning_v1.py").read_text(encoding="utf-8")
    for token in ('split="test"', "split='test'", "encoded_test", "valid_test", "test_loader", "test_dataset"):
        assert token not in text
    assert 'official_test_' in text  # the disclosure flags exist
    for line in text.splitlines():
        if "ZINC(" in line:
            assert "test" not in line


def test_runner_never_mutates_frozen_module_constants():
    text = (_CODE_DIR / "run_bjoc_recipe_tuning_v1.py").read_text(encoding="utf-8")
    frozen_names = ("LR", "EPOCHS", "WEIGHT_DECAY", "BATCH_SIZE", "GRAD_CLIP", "SOUP_LAST", "STEPS_PER_EPOCH", "UPDATES")
    for line in text.splitlines():
        stripped = line.strip()
        for name in frozen_names:
            assert f"f10k.{name} =" not in stripped
            assert f"f10k.{name}=" not in stripped
        assert "setattr(f10k" not in stripped


def test_train_recipe_refuses_r0(monkeypatch):
    def fake_verify():
        return {}

    monkeypatch.setattr(rt, "verify_freeze", fake_verify)
    with pytest.raises(SystemExit):
        rt.train_recipe("R0", 0, torch.device("cpu"))


@pytest.mark.slow
def test_r0_replay_and_input_audit_when_artifacts_present():
    required = [
        rt.F10K / "objects_valid.pt",
        rt.F10K / "soup_P_seed0.pt",
        rt.F10K / "predictions" / "valid_P0.npz",
    ]
    if not all(p.exists() for p in required):
        pytest.skip("frozen P artifacts not present locally")
    replay = rt.replay_r0()
    assert replay["all_ok"]
