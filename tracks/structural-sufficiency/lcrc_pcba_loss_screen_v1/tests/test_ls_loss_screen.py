"""Targeted CPU tests for the LCRC PCBA loss screen.

The fast tests need no cache and no GPU.  The loss suite is intentionally the
same function used to write ``analysis/loss_definition_audit.json``, so the
pytest result and the recorded audit cannot drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ls_common as C  # noqa: E402
import ls_losses as L  # noqa: E402


def test_cpu_loss_suite_all_passed():
    suite = L.cpu_validation_suite()
    assert suite["all_passed"] is True, suite
    assert suite["checks"]["task_weighting_really_changes"] is True
    assert suite["checks"]["batch_weighted_aggregation"] is True
    assert suite["checks"]["equal_coverage_equivalence"] is True
    assert suite["checks"]["fully_unlabelled_task_zero_gradient"] is True


def test_task_macro_formula_matches_manual_float64():
    y = np.array(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, np.nan],
            [1.0, np.nan, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    logits = np.array(
        [[0.2, -0.5, 0.1], [-0.3, 0.7, -0.2], [0.6, 0.1, -0.8], [-0.1, -0.4, 0.5]],
        dtype=np.float32,
    )
    n_task = np.array([9, 13, 17], dtype=np.int64)
    import torch

    loss, raw = L.task_macro_bce_loss(
        torch.tensor(logits), torch.tensor(y), torch.tensor(n_task)
    )
    ref = L.numpy_task_macro_reference(logits, y, n_task)
    scale = max(1.0, abs(ref))
    assert abs(float(loss) - ref) <= 1e-6 * scale
    raw_ref = L.numpy_micro_reference(logits, y)
    assert abs(float(raw) - raw_ref) <= 1e-6 * max(1.0, abs(raw_ref))


def test_nan_labels_cannot_reach_loss():
    import torch

    y = torch.tensor(
        [[1.0, float("nan"), float("nan")], [0.0, float("nan"), float("nan")], [1.0, 1.0, float("nan")]]
    )
    logits = torch.zeros((3, 3), requires_grad=True)
    loss, raw = L.task_macro_bce_loss(logits, y, np.array([10, 20, 30]))
    assert torch.isfinite(loss)
    assert torch.isfinite(raw)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # The fully unlabelled task contributes no gradient.
    assert float(logits.grad[:, 2].abs().max()) == 0.0


def test_fixed_recipe_and_groups_are_frozen():
    assert C.RECIPE["learning_rate"] == 5e-5
    assert C.RECIPE["weight_decay"] == 1e-3
    assert tuple(C.RECIPE["adam_betas"]) == (0.9, 0.999)
    assert C.RECIPE["adam_eps"] == 1e-8
    assert C.RECIPE["dropout"] == 0.10
    assert C.RECIPE["batch_size"] == 128
    assert C.RECIPE["gradient_clip"] == 5.0
    assert C.RECIPE["scheduler"] == "none"
    assert C.RECIPE["epochs"] == 6
    assert list(C.RECIPE["eval_epochs"]) == [0, 2, 4, 6]
    assert C.GROUPS["M"]["loss"] == "masked_bce"
    assert C.GROUPS["T"]["loss"] == "task_macro_bce"
    assert C.GROUPS["M"]["assigned_pool"] == "res-gpu0"
    assert C.GROUPS["T"]["assigned_pool"] == "res-gpu1"
    assert C.SOURCE["best_epoch"] == 9
    assert C.SOURCE["best_valid_ap"] == 0.2374902968831333
    assert C.SOURCE["optimizer_step_per_param"] == 24642
    C.assert_inherited_constants()


def test_order_plan_repeatable_and_isolation():
    rng = np.random.default_rng(3)
    train = rng.choice(50_000, size=4000, replace=False).astype(np.int64)
    valid = np.array([50_001, 50_002, 50_003], dtype=np.int64)
    test = np.array([50_004, 50_005], dtype=np.int64)
    a = C.make_order_plan(train, epochs=2)
    b = C.make_order_plan(train, epochs=2)
    assert [x["row_order_sha256"] for x in a] == [x["row_order_sha256"] for x in b]
    assert [x["batch_plan_sha256"] for x in a] == [x["batch_plan_sha256"] for x in b]
    iso = C.assert_order_isolation(a, train, valid, test)
    assert iso["passed"] is True
    for record in a:
        assert np.unique(record["rows"]).size == train.size
        assert np.intersect1d(record["rows"], valid).size == 0
        assert np.intersect1d(record["rows"], test).size == 0


def test_decision_labels_are_verbatim():
    def label(m, t, n_better=64, n_scored=127, sm="COMPLETED", st="COMPLETED", config=True):
        return C.decide_screen_label(
            ap_m6=m,
            ap_t6=t,
            status_m=sm,
            status_t=st,
            n_t_better=n_better,
            n_scored_both=n_scored,
            config_valid=config,
        )["label"]

    ap0 = C.DECISION["ap0"]
    # Just above the +0.003 floor, T at AP0, 64/127 tasks better.
    assert label(ap0 - 0.003, ap0 + 1e-9, 64, 127) == "SCREEN_PROMISING"
    # Floor met but only 63 tasks better.
    assert label(ap0 - 0.003, ap0 + 1e-9, 63, 127) == "SCREEN_MIXED"
    # Floor met, majority met, but T below AP0.
    assert label(ap0 - 0.0040, ap0 - 0.0010, 100, 127) == "SCREEN_MIXED"
    # Below the floor.
    assert label(0.2300, 0.2329, 100, 127) == "SCREEN_NO_SIGNAL"
    # Incomplete / invalid.
    assert label(0.2300, 0.2400, 100, 127, sm="INCOMPLETE") == "INCOMPLETE_OR_INVALID"
    assert label(0.2300, 0.2400, 100, 127, config=False) == "INCOMPLETE_OR_INVALID"
    # Wrong scorable-task count cannot be PROMISING.
    assert label(0.2300, 0.2400, 100, 126) == "SCREEN_MIXED"


def test_diagnostic_official_and_independent_ap_agree():
    from ls_diagnostic import _mean, _official_ap, _per_task_metrics, _stratum

    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=(40, 128)).astype(np.float32)
    score = rng.normal(size=(40, 128)).astype(np.float32)
    # Introduce missing labels and tasks with only one class.
    y[::7, ::11] = np.nan
    y[:, 0] = 1.0
    y[:, 1] = 0.0
    y[0, 1] = np.nan
    official = _official_ap(y, score)["value"]
    per = _per_task_metrics(y, score)
    assert per[0]["ap"] is None
    assert per[1]["ap"] is None
    independent = _mean([r["ap"] for r in per])
    assert abs(official - independent) <= 1e-8
    assert _stratum(100) == "sparse_le_100"
    assert _stratum(101) == "medium_101_1000"
    assert _stratum(1000) == "medium_101_1000"
    assert _stratum(1001) == "ample_gt_1000"


def test_optimizer_state_fingerprint_roundtrip():
    import copy
    import torch

    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    opt.step()
    state = opt.state_dict()
    cpu = {
        "state": {
            pid: {k: v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in vals.items()}
            for pid, vals in state["state"].items()
        },
        "param_groups": copy.deepcopy(state["param_groups"]),
    }
    from ls_checkpoint import optimizer_state_equal, optimizer_state_fingerprint

    fp = optimizer_state_fingerprint(cpu)
    assert len(fp) == 64
    assert optimizer_state_equal(cpu, cpu)["equal"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
