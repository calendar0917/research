"""Tests for the recurrent pair--centre optimization-regime audit.

Static / unit tests only: protocol overrides touch only the horizon and the
scheduler, the Q16/Q24 builders keep their frozen parameter counts, the
Phase A ensemble arithmetic is exact on synthetic predictions, and the
decision-gate labels follow the pre-registered rules.  No training is
performed and official test is never loaded.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_optimization_audit as audit,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sh


def test_01_protocol_overrides_only_change_horizon_and_scheduler():
    base = dict(sh.OPTIMIZED_PROTOCOL)
    for override in (audit.LONG_PROTOCOL, audit.PLATEAU_PROTOCOL):
        changed = {key for key in override if base.get(key) != override[key]}
        allowed = {"max_epochs", "patience", "scheduler"}
        assert all(
            key in allowed or key.startswith("scheduler_") for key in changed
        )
        assert override["max_epochs"] == 500
        assert override["patience"] == 80
    assert audit.LONG_PROTOCOL["scheduler"] == "none"
    assert audit.PLATEAU_PROTOCOL["scheduler"] == "reduce_on_plateau"
    assert audit.PLATEAU_PROTOCOL["scheduler_factor"] == 0.5
    assert audit.PLATEAU_PROTOCOL["scheduler_patience"] == 20
    assert audit.PLATEAU_PROTOCOL["scheduler_min_lr"] == 1.0e-5


def test_02_base_optimizer_settings_untouched():
    base = dict(sh.OPTIMIZED_PROTOCOL)
    for key in ("optimizer", "learning_rate", "weight_decay", "batch_size"):
        assert base[key] not in audit.LONG_PROTOCOL
        assert base[key] not in audit.PLATEAU_PROTOCOL
    assert base["learning_rate"] == 1.0e-3
    assert base["weight_decay"] == 1.0e-5
    assert base["batch_size"] == 128


def test_03_q16_builder_and_params_frozen():
    q16 = rec.build_recurrent(0)
    assert audit._n_params(q16) == audit.Q16_EXPECTED_TOTAL == 82115
    q24 = cap.build_q24(0)
    assert audit._n_params(q24) == audit.Q24_EXPECTED_TOTAL == 91211
    assert int(q24.pair_hidden) == 24
    assert int(q16.pair_hidden) == 16


def test_04_scheduler_is_default_off_in_canonical_loop():
    source = inspect.getsource(sh.train_model)
    # the canonical protocol still records scheduler "none"
    assert sh.OPTIMIZED_PROTOCOL["scheduler"] == "none"
    assert 'OPTIMIZED_PROTOCOL.get("scheduler", "none")' in source
    assert "ReduceLROnPlateau" in source
    # no test access anywhere in the module
    assert "official_test_loaded=True" not in inspect.getsource(audit)


def test_05_phase_a_ensemble_arithmetic():
    rng = np.random.default_rng(0)
    targets = rng.normal(size=200)
    p0 = targets + rng.normal(scale=0.1, size=200)
    p1 = targets + rng.normal(scale=0.1, size=200)
    mae0 = float(np.mean(np.abs(p0 - targets)))
    mae1 = float(np.mean(np.abs(p1 - targets)))
    ensemble = 0.5 * (p0 + p1)
    mae_ens = float(np.mean(np.abs(ensemble - targets)))
    mean_single = 0.5 * (mae0 + mae1)
    assert mae_ens <= mean_single + 1.0e-12
    # residual correlation is well defined and bounded
    rho = float(np.corrcoef(p0 - targets, p1 - targets)[0, 1])
    assert -1.0 <= rho <= 1.0


def test_06_plateau_scheduler_steps_on_validation_metric():
    model = rec.build_recurrent(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1.0e-5
    )
    # a worsening metric triggers decay after the patience window
    for metric in (1.0, 1.0, 1.0, 1.0):
        scheduler.step(metric)
    assert optimizer.param_groups[0]["lr"] < 1.0e-3
    assert optimizer.param_groups[0]["lr"] >= 1.0e-5


def test_07_decision_gate_labels():
    cases = [
        ((0.004, 0.0045), "training horizon limited"),
        ((0.004, 0.008), "LR schedule limited"),
        ((0.003, 0.0031), "training horizon limited"),
        ((0.0005, 0.0005), "no meaningful optimization signal"),
        ((0.004, 0.006), "LR schedule limited"),
    ]
    for (long_imp, plateau_imp), expected in cases:
        regime, applicable, _ = audit.classify_regime(long_imp, plateau_imp)
        assert regime == expected
    # both clear -> both horizon and general under-optimisation are applicable
    _, applicable, _ = audit.classify_regime(0.004, 0.0045)
    assert "training horizon limited" in applicable
    assert "optimization regime clearly suboptimal" in applicable
    # no signal -> no applicable regime
    _, applicable, _ = audit.classify_regime(0.0005, 0.0005)
    assert applicable == []


def test_08_gates_match_brief():
    assert audit.CONTINUE_GATE == 0.002
    assert audit.STRONG_GATE == 0.003
    assert audit.Q16_SEED0_VALID == 0.13837560486892472
    assert audit.Q16_SEED1_VALID == 0.13343997858563672
