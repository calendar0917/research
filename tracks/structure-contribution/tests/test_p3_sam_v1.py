"""Fast CPU correctness tests for cscl-p3-sam-v1 (P3 vs P3-SAM).

Synthetic molecules only: no ZINC dataset, no checkpoints, no GPU, no split
labels.  Covers the two-step SAM primitive, exact parameter restoration, the
perturbation norm, optimizer/scheduler step counts, gradient-clip ordering,
the last-5 soup rule, the rho=0 equivalence with the frozen P3 trainer, the
Fit-size update budget, and the frozen Monitor gate arithmetic.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import p3_sam_v1 as ps  # noqa: E402
import run_p3_sam_v1 as drv  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/p3_sam_v1.yaml"


def _norm():
    return {
        "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    }


def _objects():
    return ps.synthetic_objects()


def _y(n=8):
    return np.linspace(-0.8, 0.9, n, dtype=np.float64)


def test_config_matches_frozen_recipe() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    recipe = bp.RECIPES[cfg["recipe"]["key"]]
    assert cfg["protocol"] == ps.PROTOCOL_ID
    assert cfg["salt"] == ps.SPLIT_SALT
    assert tuple(cfg["seeds"]) == ps.SEEDS and tuple(cfg["arms"]) == ps.ARMS
    assert float(cfg["sam"]["rho"]) == ps.SAM_RHO
    assert int(cfg["recipe"]["epochs"]) == int(recipe["epochs"])
    assert int(cfg["recipe"]["batch_size"]) == int(bp.BATCH_SIZE)
    assert int(cfg["recipe"]["soup_last"]) == int(bp.SOUP_LAST)
    assert int(cfg["ssl"]["updates"]) == int(bp.PRETRAIN_UPDATES)


def test_parameter_perturbation_restore_exact() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 1)
    x = torch.as_tensor([[1.0, 2.0, -1.0], [0.5, -0.2, 0.3]], dtype=torch.float32)
    loss = F.l1_loss(lin(x).view(-1), torch.as_tensor([1.0, -1.0]))
    loss.backward()
    sam = ps.TwoStepSAM(lin.parameters(), rho=ps.SAM_RHO)
    before = {k: v.detach().clone() for k, v in lin.state_dict().items()}
    sam.perturb()
    moved = max(float((lin.state_dict()[k] - before[k]).abs().max()) for k in before)
    assert moved > 0.0
    r = sam.restore()
    after = max(float((lin.state_dict()[k] - before[k]).abs().max()) for k in before)
    assert after == 0.0
    assert r["max_abs_error_after_restore"] == 0.0
    assert r["max_abs_delta_before_restore"] > 0.0


def test_perturbation_norm_equals_rho() -> None:
    torch.manual_seed(1)
    lin = torch.nn.Linear(4, 1)
    x = torch.randn(5, 4)
    F.mse_loss(lin(x), torch.randn(5, 1)).backward()
    sam = ps.TwoStepSAM(lin.parameters(), rho=ps.SAM_RHO)
    p = sam.perturb()
    sam.restore()
    assert abs(p["perturbation_norm"] - ps.SAM_RHO) <= 1e-6


def test_optimizer_and_scheduler_step_counts() -> None:
    model = ps.build_p3(7, _norm(), "cpu")
    info = ps.train_supervised_sam(
        arm="P3SAM",
        seed=7,
        model=model,
        train_objs=_objects(),
        y_fit=_y(),
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
        log_fn=lambda _m: None,
    )
    expected = 2 * math.ceil(8 / 4)
    a = info["anomalies"]
    assert info["updates"] == expected
    assert a["optimizer_step_count"] == expected
    assert a["scheduler_step_count"] == expected
    assert a["scheduler_last_epoch"] == expected
    assert info["cost"]["forward_calls"] == 2 * expected
    assert info["cost"]["backward_calls"] == 2 * expected


def test_gradient_clip_only_after_second_backward() -> None:
    model = ps.build_p3(8, _norm(), "cpu")
    calls: list[float] = []

    def recording_clip(parameters, max_norm):
        calls.append(ps._grad_norm(parameters))
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    info = ps.train_supervised_sam(
        arm="P3SAM",
        seed=8,
        model=model,
        train_objs=_objects(),
        y_fit=_y(),
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
        grad_clip=1e-8,
        clip_fn=recording_clip,
        _allow_test_grad_clip=True,
        log_fn=lambda _m: None,
    )
    expected = 2 * math.ceil(8 / 4)
    a = info["anomalies"]
    assert len(calls) == expected
    assert a["grad_clip_calls"] == expected
    assert a["grad_clip_events"] == expected
    assert all(v > 1e-8 for v in calls)
    assert abs(max(calls) - a["second_grad_norm_max"]) <= 1e-6
    assert a["clip_return_minus_computed_max"] <= 1e-5


def test_last_five_soup_rule() -> None:
    model = ps.build_p3(9, _norm(), "cpu")
    info = ps.train_supervised_sam(
        arm="P3SAM",
        seed=9,
        model=model,
        train_objs=_objects(),
        y_fit=_y(),
        device=torch.device("cpu"),
        epochs=6,
        batch_size=4,
        log_fn=lambda _m: None,
        collect_epoch_states=True,
    )
    states = info.pop("epoch_states")
    members = [2, 3, 4, 5, 6]
    assert info["soup_members"] == members
    expected = {k: torch.stack([states[i - 1][k] for i in members]).mean(0) for k in states[0]}
    diff = max(float((info["soup_state"][k] - expected[k]).abs().max()) for k in expected)
    assert diff <= 1e-7


def test_rho_zero_equals_frozen_p3_trainer() -> None:
    objs = _objects()
    y = _y()
    info_a = bp.train_supervised(
        arm="P3",
        seed=10,
        recipe_key=ps.RECIPE_KEY,
        init="scratch",
        train_objs=objs,
        rows=[o.mol_index for o in objs],
        y_fit=y,
        norm=_norm(),
        device=torch.device("cpu"),
        smoke_epochs=2,
        log_fn=lambda _m: None,
    )
    model_b = ps.build_p3(10, _norm(), "cpu")
    info_b = ps.train_supervised_sam(
        arm="P3SAM",
        seed=10,
        model=model_b,
        train_objs=objs,
        y_fit=y,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=bp.BATCH_SIZE,
        rho=0.0,
        _allow_test_rho_zero=True,
        log_fn=lambda _m: None,
    )
    diff = max(float((info_a["soup_state"][k] - info_b["soup_state"][k]).abs().max()) for k in info_a["soup_state"])
    assert diff <= 1e-7
    assert info_a["batch_order"]["full_hash"] == info_b["batch_order"]["full_hash"]
    assert info_a["updates"] == info_b["updates"]


def test_fit_size_update_budget_and_order() -> None:
    n_fit = 7996
    epochs = int(bp.RECIPES[ps.RECIPE_KEY]["epochs"])
    updates = epochs * math.ceil(n_fit / bp.BATCH_SIZE)
    assert updates == 22_680
    h0 = bp.batch_order_hash(0, n_fit, bp.BATCH_SIZE, epochs)
    h1 = bp.batch_order_hash(1, n_fit, bp.BATCH_SIZE, epochs)
    assert h0 != h1


def test_monitor_gate_arithmetic() -> None:
    rng = np.random.default_rng(11)
    n = 128
    y = rng.normal(0.3, 0.9, n)
    y[:8] = -3.5
    ref = {s: y + rng.normal(0.0, 0.20, n) for s in ps.SEEDS}
    new = {s: y + rng.normal(0.0, 0.05, n) for s in ps.SEEDS}
    c = drv._contrast(y, new, ref)
    g = drv._gate(c)
    assert set(c) >= {"delta_seed_mean", "per_seed_delta", "bootstrap", "ordinary_delta", "top10_removed_delta"}
    assert c["bootstrap"]["ci_lo"] <= c["bootstrap"]["ci_hi"]
    assert g["passed"] is True
    assert drv._gate({**c, "delta_seed_mean": 0.001})["passed"] is False
    assert drv._gate({**c, "per_seed_delta": {"0": -0.01, "1": 0.01}})["passed"] is False
    assert drv._gate({**c, "ordinary_delta": 0.001})["passed"] is False
    assert drv._gate({**c, "top10_removed_delta": 0.001})["passed"] is False
    assert drv._gate({**c, "tail_delta": 0.02})["passed"] is False
    assert drv._gate({**c, "rmse_new": c["rmse_ref"] + 0.02})["passed"] is False
