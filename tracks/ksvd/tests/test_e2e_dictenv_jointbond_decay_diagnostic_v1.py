"""Focused data-free tests for the JointBond-Decay-Diagnostic-v1 runner.

Covers the pieces that the round's interpretation depends on:

* ``torch.optim.Adam(weight_decay=...)`` is coupled L2 (not AdamW);
* the telemetry never conflates a float32 norm underflow with a true zero;
* the pre-registered verdict classifier's four cases;
* the two arms are built from one initialisation and are bit-identical;
* the batch fingerprint is deterministic and tensor-sensitive.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_jointbond_decay_diagnostic_v1 as diag,
)


def test_coupled_l2_is_not_adamw() -> None:
    probe = diag._adam_coupling_probe()
    assert probe["coupled_l2_confirmed"] is True
    # coupled L2: a zero gradient plus wd=1e-5 still produces an ~lr-sized step
    assert abs(probe["adam_wd1e-5"]["delta"] + diag.LEARNING_RATE) < 1e-3
    # no decay with wd=0
    assert probe["adam_wd0"]["delta"] == 0.0
    # AdamW's decoupled decay is wd-sized, not lr-sized
    assert abs(probe["adamw_wd1e-5"]["delta"]) < 1e-5


def test_norm_underflow_is_not_a_true_zero() -> None:
    tiny = float(torch.finfo(torch.float32).tiny)
    subnormal = torch.full((8,), tiny / 4.0, dtype=torch.float32)
    stats = diag.tensor_stats("synthetic", subnormal)
    assert stats["weight_counts"]["subnormal"] == 8
    assert stats["weight_norm_original_float32"] == 0.0
    assert stats["weight_norm_float64"] > 0.0
    assert stats["weight_float32_norm_zero_but_float64_norm_nonzero"] is True

    zero = torch.zeros(8, dtype=torch.float32)
    zero_stats = diag.tensor_stats("synthetic_zero", zero)
    assert zero_stats["weight_counts"]["zero"] == 8
    assert zero_stats["weight_float32_norm_zero_but_float64_norm_nonzero"] is False


def test_tensor_stats_reports_gradient_and_update() -> None:
    weight = torch.tensor([[1e-6, -2e-6], [0.0, 3e-6]], dtype=torch.float32)
    grad = torch.full_like(weight, 1e-6)
    update = torch.full_like(weight, -1e-9)
    stats = diag.tensor_stats("w", weight, grad=grad, update=update, decay_applied=True)
    assert stats["grad_absmax_float32"] == pytest.approx(1e-6, rel=1e-6)
    assert stats["grad_nonzero_fraction"] == 1.0
    assert stats["grad_norm_float64"] > 0.0
    assert stats["decay_over_task_ratio_defined"] is True
    assert stats["decay_over_task_ratio"] == pytest.approx(
        diag.WEIGHT_DECAY_REFERENCE * float(weight.double().norm()) / float(grad.double().norm()),
        rel=1e-12,
    )
    assert stats["update_norm_float64"] > 0.0
    assert stats["weight_changed"] is True
    assert stats["reference_decay_applied"] is True


def test_verdict_classifier_cases() -> None:
    # signature: _classify(d_fit_wd, d_fit_nowd, weight_norm_wd, weight_norm_nowd)
    assert diag._classify(-0.0001, -0.01, 1.0, 1.0) == "DECAY_SUPPRESSION_CONFIRMED"
    assert diag._classify(-0.01, -0.01, 1.0, 1.0) == "BOTH_LEARN_DECAY_NOT_THE_CAUSE"
    assert diag._classify(-0.003, -0.01, 1.0, 1.0) == "WD_SUPPRESSES_BUT_BOTH_LEARN"
    assert diag._classify(-0.0001, -0.0001, 0.5, 1.0) == "SURVIVAL_ONLY"
    assert diag._classify(-0.0001, -0.0001, 1.0, 0.5) == "NEITHER_LEARNS_NOT_DECAY"
    assert diag._classify(-0.01, -0.0001, 1.0, 0.5) == "WD_ARM_LEARNS_NOWD_DOES_NOT"


def test_two_arms_are_bit_identical_and_branch_is_alive() -> None:
    torch.set_num_threads(2)
    arm_wd, arm_nowd, info = diag.build_arms()
    assert info["full_state_sha256_wd"] == info["full_state_sha256_nowd"]
    assert info["missing_keys"] == sorted(diag.BRANCH_KEYS)
    assert info["unexpected_keys"] == []
    assert info["fresh_joint_state_sha256"] != info["dead_joint_state_sha256"]
    assert all(value > 1e-6 for value in info["fresh_joint_absmax"].values())
    assert info["parent_state_sha256"] == diag.SEM108_SOUP_SHA256
    assert info["branch_params"] == 4224
    assert torch.equal(arm_wd.joint_A.weight.detach(), arm_nowd.joint_A.weight.detach())
    assert all(parameter.requires_grad for parameter in arm_wd.parameters())


def test_batch_fingerprint_is_deterministic_and_sensitive() -> None:
    def make(seed: int) -> SimpleNamespace:
        generator = torch.Generator().manual_seed(seed)
        return SimpleNamespace(
            dict_phi=torch.randn(6, 4, generator=generator),
            dict_atom=torch.randint(0, 5, (6,), generator=generator),
            y=torch.randn(2, generator=generator),
        )

    first = diag._batch_fingerprint(make(0))
    again = diag._batch_fingerprint(make(0))
    other = diag._batch_fingerprint(make(1))
    assert first["sha256"] == again["sha256"]
    assert first["sha256"] != other["sha256"]
    assert first["tensor_keys"] == ["dict_phi", "dict_atom", "y"]
