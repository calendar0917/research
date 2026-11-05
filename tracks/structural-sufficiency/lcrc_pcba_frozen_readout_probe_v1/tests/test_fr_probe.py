"""CPU-only correctness tests for the frozen-readout probe.

These tests never touch the PCBA cache, the checkpoint, or a GPU.  They pin the
mechanisms the formal round depends on:

* Train-only standardisation with safe zero-variance columns,
* masked BCE over observed entries only,
* the exact-zero initialisation of both residual heads,
* the ``rho`` forward-hook capture used for extraction,
* the paired row-order plan (identical for both arms, never a Validation row),
* the per-task delta aggregation and the pre-registered decision rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import fr_common as C  # noqa: E402
import fr_metrics as FM  # noqa: E402
import fr_model as MODEL  # noqa: E402


# ---------------------------------------------------------------------------
# standardisation
# ---------------------------------------------------------------------------


def test_standardizer_uses_train_rows_only_and_protects_zero_variance():
    train = torch.zeros((32, 4), dtype=torch.float32)
    train[:, 0] = torch.arange(32, dtype=torch.float32)
    train[:, 1] = 3.0  # constant column -> must stay constant
    train[:, 2] = torch.linspace(-2.0, 2.0, 32)
    train[:, 3] = 1.0
    stats = FM.fit_standardizer(train)
    assert stats["n_rows_fit"] == 32
    assert stats["zero_variance_columns"] == [1, 3]
    assert torch.all(stats["scale"][[1, 3]] == 1.0)
    assert torch.allclose(stats["mean"], train.mean(dim=0), atol=1e-6)

    # The constant Train columns map to exactly 0 and never divide by zero.
    train_std = train.clone()
    FM.standardize_(train_std, stats["mean"], stats["scale"])
    assert torch.all(train_std[:, 1] == 0.0)
    assert torch.all(train_std[:, 3] == 0.0)
    assert torch.isfinite(train_std).all()

    # Fit is a pure function of Train: a different Validation block cannot move it.
    stats_again = FM.fit_standardizer(train)
    assert torch.equal(stats_again["mean"], stats["mean"])
    assert torch.equal(stats_again["scale"], stats["scale"])
    valid_b = torch.full((8, 4), 100.0, dtype=torch.float32)
    out_b = valid_b.clone()
    FM.standardize_(out_b, stats["mean"], stats["scale"])
    assert torch.isfinite(out_b).all()
    # Constant column keeps its offset (scale 1), so Validation is shifted, not destroyed.
    assert torch.allclose(out_b[:, 1], torch.full((8,), 97.0))
    assert torch.allclose(out_b[:, 3], torch.full((8,), 99.0))


def test_standardize_matches_reference_formula():
    torch.manual_seed(0)
    x = torch.randn(64, 5)
    stats = FM.fit_standardizer(x)
    out = x.clone()
    FM.standardize_(out, stats["mean"], stats["scale"])
    reference = (x - x.mean(dim=0)) / x.std(dim=0, unbiased=False)
    assert torch.allclose(out, reference, atol=1e-5)
    again = FM.column_stats(out)
    assert again["column_mean_abs_max"] < 1e-5
    assert abs(again["column_std_min"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# masked BCE
# ---------------------------------------------------------------------------


def test_masked_bce_weights_only_observed_entries():
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    y = torch.tensor(
        [
            [1.0, float("nan"), 0.0],
            [0.0, 1.0, float("nan")],
            [1.0, 0.0, 1.0],
            [float("nan"), 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, float("nan"), 1.0],
        ]
    )
    mask = torch.isfinite(y)
    observed = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
    value = FM.masked_bce_torch(logits, y)
    assert torch.allclose(value, observed)
    # Changing a masked-out entry cannot change the loss.
    logits2 = logits.clone()
    logits2[~mask] = 1234.0
    assert torch.allclose(FM.masked_bce_torch(logits2, y), observed)
    # The numpy path agrees with the torch path.
    assert abs(FM.masked_bce_numpy(logits.numpy(), y.numpy()) - float(observed)) < 1e-6


# ---------------------------------------------------------------------------
# residual heads
# ---------------------------------------------------------------------------


def test_linear_head_is_exactly_zero_at_initialisation():
    head = FM.build_head("L", 7)
    x = torch.randn(11, 7)
    frozen = torch.randn(11, C.N_TASKS)
    audit = FM.zero_init_audit(head, x, frozen)
    assert audit["residual_is_exactly_zero"]
    assert audit["logits_bitwise_equal_frozen"]
    assert torch.equal(head(x), torch.zeros(11, C.N_TASKS))
    assert torch.equal(head.weight, torch.zeros_like(head.weight))
    assert torch.equal(head.bias, torch.zeros_like(head.bias))


def test_nonlinear_head_is_zero_at_init_and_deterministically_initialised():
    x = torch.randn(9, 7)
    frozen = torch.randn(9, C.N_TASKS)
    head = FM.build_head("N", 7)
    audit = FM.zero_init_audit(head, x, frozen)
    assert audit["residual_is_exactly_zero"] and audit["logits_bitwise_equal_frozen"]
    assert torch.equal(head[-1].weight, torch.zeros_like(head[-1].weight))
    assert not torch.equal(head[0].weight, torch.zeros_like(head[0].weight))
    fp_a = FM.head_parameter_fingerprint(head)["combined_sha256"]
    head_b = FM.build_head("N", 7)
    fp_b = FM.head_parameter_fingerprint(head_b)["combined_sha256"]
    assert fp_a == fp_b
    other = FM.build_head("N", 7, seed=C.SEED + 1)
    assert FM.head_parameter_fingerprint(other)["combined_sha256"] != fp_a
    assert head[0].weight.shape == (C.N_TASKS, 7)
    assert head[-1].weight.shape == (C.N_TASKS, C.N_TASKS)


def test_heads_have_the_frozen_trainable_parameter_counts():
    assert FM.head_parameter_fingerprint(FM.build_head("L", C.RHO_IN_DIM_EXPECTED))["n_parameters"] == int(
        C.ARMS["L"]["trainable_parameters"]
    )
    assert FM.head_parameter_fingerprint(FM.build_head("N", C.RHO_IN_DIM_EXPECTED))["n_parameters"] == int(
        C.ARMS["N"]["trainable_parameters"]
    )


# ---------------------------------------------------------------------------
# rho capture
# ---------------------------------------------------------------------------


class _MiniRhoModel(torch.nn.Module):
    """Minimal stand-in exposing the same ``rho`` call pattern as LCRC-C1."""

    def __init__(self, d: int, n_tasks: int):
        super().__init__()
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(2 * d, 8), torch.nn.SiLU(), torch.nn.Linear(8, n_tasks)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho_in = torch.cat([x, torch.log1p(x.abs())], dim=1)
        return self.rho(rho_in), rho_in


def test_rho_input_hook_records_the_exact_tensor_passed_to_rho():
    model = _MiniRhoModel(4, 3).eval()
    capture = MODEL.RhoInputCapture(model.rho)
    x = torch.randn(5, 4)
    with torch.inference_mode():
        logits, expected = model(x)
        assert capture.calls == 1
        captured = capture.take()
        assert torch.equal(captured, expected)
        assert tuple(captured.shape) == (5, 8)
    # A module whose rho is never called must fail loudly instead of caching zeros.
    with pytest.raises(RuntimeError):
        capture.take()
    capture.close()
    with pytest.raises(RuntimeError):
        capture.take()


# ---------------------------------------------------------------------------
# paired row order
# ---------------------------------------------------------------------------


def test_row_plan_is_deterministic_shared_and_isolated():
    n = 5_000
    plan_a = FM.row_plan(n, batch_size=1024, epochs=C.EPOCHS, seed=C.SEED)
    plan_b = FM.row_plan(n, batch_size=1024, epochs=C.EPOCHS, seed=C.SEED)
    assert plan_a["global_plan_sha256"] == plan_b["global_plan_sha256"]
    assert len(plan_a["perms"]) == C.EPOCHS
    assert plan_a["steps_per_epoch"] == 5
    for perm in plan_a["perms"]:
        assert sorted(perm.tolist()) == list(range(n))
    forbidden = list(range(4_000, n))
    assert FM.plan_isolated(plan_a, forbidden, n)["passed"] is False
    assert FM.plan_isolated(plan_a, [], n)["passed"] is True
    other_seed = FM.row_plan(n, batch_size=1024, epochs=C.EPOCHS, seed=C.SEED + 1)
    assert other_seed["global_plan_sha256"] != plan_a["global_plan_sha256"]


# ---------------------------------------------------------------------------
# deltas and decision rule
# ---------------------------------------------------------------------------


def test_delta_summary_counts_improvements_and_concentration():
    base = [0.10, 0.20, 0.30, None, 0.50]
    new = [0.11, 0.19, 0.40, 0.99, 0.50]
    out = FM.delta_summary(base, new)
    assert out["n_paired_tasks"] == 4
    assert out["n_improved"] == 2  # tasks 0 and 2
    assert out["n_worse"] == 1  # task 1
    assert out["n_tied"] == 1  # task 4
    assert out["largest_delta_task"] == 2
    assert out["smallest_delta_task"] == 1
    assert out["sum_delta"] == pytest.approx(0.01 - 0.01 + 0.10 + 0.0)
    assert out["top1_delta_share_of_sum"] == pytest.approx(1.0)


def test_decision_rule_follows_the_preregistered_thresholds():
    base = dict(
        status_l="COMPLETED",
        status_n="COMPLETED",
        gates_ok=True,
        improved_l=70,
        improved_n=20,
        n_scorable=127,
    )
    signal = C.decide_label(d_l=0.006, d_n=0.001, d_nl=-0.005, **base)
    assert signal["label"] == "READOUT_SIGNAL"
    assert signal["nonlinear_extra_signal"] is False
    assert signal["arm_passes_readout_floor"] == {"L": True, "N": False}

    extra = C.decide_label(d_l=0.002, d_n=0.007, d_nl=0.005, improved_l=20, improved_n=80, n_scorable=127,
                           status_l="COMPLETED", status_n="COMPLETED", gates_ok=True)
    assert extra["label"] == "READOUT_SIGNAL"
    assert extra["nonlinear_extra_signal"] is True

    narrow = C.decide_label(d_l=0.006, d_n=0.0, d_nl=-0.006, **{**base, "improved_l": 63})
    assert narrow["label"] == "WEAK_OR_MIXED"

    weak = C.decide_label(d_l=0.001, d_n=-0.002, d_nl=-0.003, **base)
    assert weak["label"] == "WEAK_OR_MIXED"

    none = C.decide_label(d_l=-0.001, d_n=-0.002, d_nl=-0.001, **base)
    assert none["label"] == "NO_FAST_READOUT_SIGNAL"

    incomplete = C.decide_label(d_l=-0.001, d_n=-0.002, d_nl=-0.001, **{**base, "status_n": "INCOMPLETE"})
    assert incomplete["label"] == "INCOMPLETE"

    failed_gate = C.decide_label(d_l=0.02, d_n=0.03, d_nl=0.01, **{**base, "gates_ok": False})
    assert failed_gate["label"] == "INCOMPLETE"

    missing = C.decide_label(d_l=None, d_n=None, d_nl=None, **base)
    assert missing["label"] == "INCOMPLETE"


def test_frozen_constants_match_the_round_instruction():
    assert C.AP0 == 0.2374902968831333
    assert abs(C.TRAIN_AP0 - 0.388336) < 1e-5
    assert C.N_SCORABLE_TASKS_EXPECTED == 127
    assert C.RHO_IN_DIM_EXPECTED == 171
    assert C.DECISION["improvement_floor"] == 0.005
    assert C.DECISION["task_majority"] == 64
    assert C.DECISION["nonlinear_extra_floor"] == 0.003
    assert C.EVAL_EPOCHS == (0, 2, 4, 8)
    assert C.MAIN_EPOCH == 8
    assert C.SOURCE["checkpoint_sha256"] == "b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5"
    assert C.BUDGET["gpu_hours_total"] == 0.4
    assert C.HEAD_RECIPE["learning_rate"] == 3e-4
    assert C.HEAD_RECIPE["weight_decay"] == 1e-3
    assert C.BATCH_SIZE == 1024 and C.EPOCHS == 8
    assert C.STEPS_PER_EPOCH == 343


def test_hash_helpers_are_content_sensitive():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    h = C.sha256_array(arr)
    assert h == C.sha256_array(arr.copy())
    assert h != C.sha256_array(arr + 1)
    assert h != C.sha256_array(arr.astype(np.float64))
