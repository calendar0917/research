"""Fresh-clone-safe pure-math tests for ZPR-v1 (no data, no checkpoints, no GPU).

Run:
    uv run pytest -q tracks/structural-sufficiency/zinc_perf_recipe_pilot_v1/tests
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import zpr_common as Z  # noqa: E402
import zpr_train as T  # noqa: E402


def _ref(d: float, beta: float) -> float:
    z = abs(d)
    return 0.5 * z * z / beta if z < beta else z - 0.5 * beta


@pytest.mark.parametrize("beta", [0.02, 0.05])
@pytest.mark.parametrize("d", [-2.0, -0.2, -0.05, -0.02, -1e-3, 0.0, 1e-3, 0.02, 0.05, 0.2, 5.0])
def test_smooth_l1_value_and_gradient_match_hand_formula(beta: float, d: float) -> None:
    pred = torch.tensor([d], dtype=torch.float64, requires_grad=True)
    loss = F.smooth_l1_loss(pred, torch.zeros(1, dtype=torch.float64), beta=beta)
    loss.backward()
    assert abs(loss.item() - _ref(d, beta)) < 1e-12
    expected_grad = d / beta if abs(d) < beta else math.copysign(1.0, d)
    assert abs(pred.grad.item() - expected_grad) < 1e-12


def test_smooth_l1_is_continuous_at_beta() -> None:
    beta = 0.05
    assert abs(_ref(beta, beta) - _ref(beta - 1e-12, beta)) < 1e-9


def test_loss_dispatch_matches_frozen_semantics() -> None:
    torch.manual_seed(0)
    pred = torch.randn(37, dtype=torch.float64)
    target = torch.randn(37, dtype=torch.float64)
    l1 = T._loss("l1", pred, target)
    assert torch.equal(l1, F.l1_loss(pred, target))
    s = T._loss("smooth_l1", pred, target)
    assert torch.allclose(s, T.hand_smooth_l1(pred, target, Z.SMOOTH_L1_BETA), atol=0, rtol=0)
    assert torch.allclose(s, F.smooth_l1_loss(pred, target, beta=Z.SMOOTH_L1_BETA), atol=0, rtol=0)


def test_ema_start_step() -> None:
    assert T.ema_start_step(47400) == 11850
    assert T.ema_start_step(158) == 40
    assert T.ema_start_step(3) == 1


def test_ema_recurrence_and_initialisation() -> None:
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 1))
    start = 5
    decay = 0.9
    ema = None
    online_history = []
    ref_history = []
    ema_history = []
    for step in range(1, 12):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        online_history.append([p.detach().clone() for p in model.parameters()])
        if step == start:
            ema = T.ema_init_state(model)
            assert all(torch.equal(a, b) for a, b in zip(ema, model.parameters()))
        elif step > start:
            T.ema_update_(ema, model, decay)
        ema_history.append(None if ema is None else [t.detach().clone() for t in ema])
        # independent reference
        if step < start:
            ref_history.append(None)
        elif step == start:
            ref_history.append([p.detach().clone() for p in model.parameters()])
        else:
            prev = ref_history[-1]
            ref_history.append([decay * r + (1 - decay) * p.detach()
                                for r, p in zip(prev, model.parameters())])
    for step in range(start, 12):
        assert ema_history[step - 1] is not None
        assert all(torch.allclose(a, b, atol=1e-6, rtol=0)
                   for a, b in zip(ema_history[step - 1], ref_history[step - 1]))
    # before the start step the EMA does not exist at all
    assert all(h is None for h in ema_history[: start - 1])


def test_copy_ema_into_direction() -> None:
    """The historical zps bug was the copy direction; this pins the correct one."""
    torch.manual_seed(11)
    ema_model = torch.nn.Linear(3, 2)
    online = torch.nn.Linear(3, 2)
    ema_state = [torch.full_like(p, 7.0) for p in online.parameters()]
    T.copy_ema_into(ema_model, ema_state, online)
    assert all(torch.equal(p, q) for p, q in zip(ema_model.parameters(), ema_state))
    assert not any(torch.equal(p, q) for p, q in zip(ema_model.parameters(), online.parameters()))


def test_group_masks_are_the_frozen_lcrc_groups() -> None:
    masks = Z.group_masks()
    assert int(masks["full"].sum()) == 1000
    assert int(masks["ordinary"].sum()) == 935
    assert int(masks["r2_unseen"].sum()) == 251
    assert int(masks["r2_unseen_ordinary"].sum()) == 225
    assert int(masks["seen_ordinary"].sum()) == 710
    assert int(masks["minus_row172"].sum()) == 999
    assert int(masks["minus_fixed_worst10"].sum()) == 990
    assert int(masks["fixed_worst10"].sum()) == 10
    assert not masks["minus_row172"][Z.ROW172]


def test_frozen_constants_are_frozen() -> None:
    assert Z.FROZEN["params"] == 237519
    assert Z.FROZEN["total_updates"] == 47400
    assert Z.FROZEN["steps_per_epoch"] == 79
    assert Z.FROZEN["ema_start_step"] == 11850
    assert Z.SMOOTH_L1_BETA == 0.05
    assert Z.EMA_DECAY == 0.999
    assert set(Z.ARMS) == {"G0", "G1"}
    assert Z.ARMS["G0"]["loss"] == "l1"
    assert Z.ARMS["G1"]["loss"] == "smooth_l1"


def test_window_diagnostics_matches_frozen_definition() -> None:
    vals = [float(v) for v in np.linspace(1.0, 0.1, 40)]
    out = Z.window_diagnostics(vals)
    expected_w20 = min(float(np.median(vals[i:i + 20])) for i in range(21))
    assert abs(out["W20"] - expected_w20) < 1e-12
    assert abs(out["T100"] - float(np.median(vals[-40:]))) < 1e-12


def test_frozen_constants_match_lcrc_preregistration() -> None:
    """Catch hand-transcription drift against the frozen LCRC-v1 config."""
    cfg = Z.read_json(Z.LCRC_CONFIG)
    assert Z.FROZEN["train_cache_content_sha256"] == cfg["data"]["train_cache"]["content_sha256"]
    assert Z.FROZEN["valid_cache_content_sha256"] == cfg["data"]["valid_cache"]["content_sha256"]
    assert Z.FROZEN["ssl_s0_sha256"] == cfg["data"]["ssl_s0"]["sha256"]
    assert Z.FROZEN["norm_fingerprint"] == cfg["data"]["normalisation"]["fingerprint"]
    assert Z.FROZEN["params"] == cfg["model"]["params"]["A"]
    assert Z.FROZEN["total_updates"] == cfg["recipe"]["expected_total_updates"]
    assert Z.FROZEN["steps_per_epoch"] == cfg["recipe"]["expected_steps_per_epoch"]


# --------------------------------------------------------------------------- #
# §2.4 decision rule (the verdict must follow the registered rule, not a
# narratively convenient variant of it)
# --------------------------------------------------------------------------- #
def _rule():
    import zpr_compare

    return zpr_compare.evaluate_rule


def test_decision_rule_requires_all_three_conditions() -> None:
    rule = _rule()
    assert rule(-0.0031, -0.0001, -0.0001)["verdict"] == "SEED0_CANDIDATE_SMOOTHL1"


def test_decision_rule_rejects_gain_below_threshold_even_if_others_hold() -> None:
    rule = _rule()
    out = rule(-0.0029, -0.01, -0.01)
    assert out["delta_full_le_minus_0.003"] is False
    assert out["verdict"] == "NO_COMPELLING_RECIPE_GAIN"


def test_decision_rule_rejects_full_gain_with_worse_bulk_or_worst_row() -> None:
    rule = _rule()
    assert rule(-0.01, +0.0001, -0.01)["verdict"] == "NO_COMPELLING_RECIPE_GAIN"
    assert rule(-0.01, -0.01, +0.0001)["verdict"] == "NO_COMPELLING_RECIPE_GAIN"


def test_decision_rule_is_exactly_at_the_registered_boundary() -> None:
    """Δfull == -0.003 is a pass (<=), Δord == 0 is not (< 0)."""
    rule = _rule()
    assert rule(-0.003, -1e-9, -1e-9)["delta_full_le_minus_0.003"] is True
    assert rule(-0.003, 0.0, -1e-9)["delta_ordinary_lt_0"] is False


def test_decision_rule_rejects_a_better_candidate_that_only_wins_on_tail() -> None:
    """A tail-only gain with worse bulk is still NO_COMPELLING_RECIPE_GAIN."""
    rule = _rule()
    assert rule(+0.00298, +0.00407, +0.00537)["verdict"] == "NO_COMPELLING_RECIPE_GAIN"
