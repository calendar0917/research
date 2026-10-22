"""Fast CPU unit tests for the P3-SAM implementation (no data, no GPU).

These are narrower copies of the ``sam_checks`` gate so that a regression in the
two-pass logic / schedule / restore path fails a normal test run.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
_V2 = _HERE.parents[2] / "zinc_recipe_trunk_v2" / "code"
_V1 = _HERE.parents[2] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_CODE), str(_V2), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sam_checks as K  # noqa: E402
import sam_core as S  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return S.load_config()


def test_sam_epsilon_is_detached_and_has_rho_norm(cfg):
    x, y = K._tiny_data()
    torch.manual_seed(7)
    model = K.Tiny()
    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    F.l1_loss(model(x), y).backward()
    norm = S.grad_global_norm(params)
    eps = S.sam_epsilon(params, norm, 0.05)
    assert all(e is not None and not e.requires_grad for e in eps)
    got = float(S.epsilon_norm(eps))
    assert got == pytest.approx(0.05, rel=1e-6)


def test_grad_global_norm_uses_raw_gradients(cfg):
    x, y = K._tiny_data()
    torch.manual_seed(7)
    model = K.Tiny()
    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    F.l1_loss(model(x), y).backward()
    got = float(S.grad_global_norm(params))
    ref = float(math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in params)))
    assert got == pytest.approx(ref, rel=1e-12)


def test_rho0_equals_adamw_five_updates(cfg):
    d = K.check_rho0_equivalence(cfg)
    assert d["pass"], d
    assert d["optimizer_steps"] == 5


def test_two_pass_matches_hand_computation(cfg):
    d = K.check_two_pass_handcomputed(cfg)
    assert d["pass"], d
    assert d["adamw_step_counts"] == [1]


def test_first_pass_gradient_not_clipped(cfg):
    d = K.check_epsilon_and_no_first_clip(cfg)
    assert d["pass"], d
    assert d["raw_g1_exceeds_clip"]


def test_first_pass_no_leak(cfg):
    d = K.check_first_pass_no_leak(cfg)
    assert d["pass"], d


def test_exception_path_restores_parameters(cfg):
    d = K.check_exception_restores(cfg)
    assert d["pass"], d
    assert d["params_bit_identical_after_exception"]


def test_perturb_apply_and_restore_bitwise(cfg):
    x, y = K._tiny_data()
    torch.manual_seed(7)
    model = K.Tiny()
    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    F.l1_loss(model(x), y).backward()
    norm = S.grad_global_norm(params)
    eps = S.sam_epsilon(params, norm, 0.05)
    backup = S.save_params(params)
    S.apply_perturbation(params, eps)
    assert S.params_match_perturbed(params, backup, eps)
    assert any(not torch.equal(p.detach(), b) for p, b in zip(params, backup))
    S.restore_params(params, backup)
    assert S.params_match_backup(params, backup)


def test_schedule_counts_and_endpoints(cfg):
    d = K.check_schedule(cfg)
    assert d["pass"], d
    assert d["steps_per_epoch"] == 79 and d["total_updates"] == 47400


def test_best_selection_ties_keep_earlier(cfg):
    assert K.check_best_selection(cfg)["pass"]


def test_order_rng_not_consumed_by_sam_forward(cfg):
    n, updates = 200, 3
    ref = torch.Generator().manual_seed(0 + 101)
    expected = [S.order_sha256(S.epoch_order(ref, n)) for _ in range(updates + 1)]
    gen = torch.Generator().manual_seed(0 + 101)
    x, y = K._tiny_data()
    torch.manual_seed(7)
    model = K.Tiny()
    params = list(model.parameters())
    opt = S.make_optimizer(model, cfg)
    got = [S.order_sha256(S.epoch_order(gen, n))]
    for i in range(updates):
        S.update_sam(model, opt, params, x, y, rho=0.05, clip=5.0, epoch=1)
        S.set_lr(opt, i + 1, 47400, cfg)
        opt.step()
        got.append(S.order_sha256(S.epoch_order(gen, n)))
    assert got == expected


def test_no_batchnorm_or_dropout_in_p3(cfg):
    import ad_data as D

    norm = D.load_norm()
    model, info = S.build_model(norm, torch.device("cpu"))
    assert info["parameters"] == 413505
    names = [n for n, m in model.named_modules() if "BatchNorm" in type(m).__name__]
    assert names == []
    rates = {float(m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)}
    assert all(r == 0.0 for r in rates)
