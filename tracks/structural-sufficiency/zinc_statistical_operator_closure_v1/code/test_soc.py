"""pytest entry points for SOC-v1 (CPU, real train cache, no valid/test labels)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SUFF = _HERE.parents[1]
for _p in (str(_HERE), str(_SUFF / "zinc_arch_discovery_v1" / "code"),
           str(_SUFF / "zinc_dual_channel_composer_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_models as DM  # noqa: E402
import soc_models as SM  # noqa: E402


def _fixtures():
    cache = D.load_cache("train")
    norm = D.load_norm()
    rows = np.arange(8, dtype=np.int64)
    batch = T1.collate(cache, rows, "P3N", valid=True)
    return cache, norm, batch


def test_initial_state_is_dcc_l():
    _, norm, batch = _fixtures()
    dm, _ = DM.build_initial_state(norm, 0)
    soc, info = SM.build_initial_state(norm, 0)
    dm.eval()
    soc.eval()
    with torch.no_grad():
        soc.closure.gate.fill_(-30.0)
        a = dm(batch)
        b = soc(batch)
    assert info["parameters"] == 512888
    assert float((a - b).abs().max()) < 1e-8


def test_mean_readout_is_also_identity_at_zero_gate():
    _, norm, batch = _fixtures()
    dm, _ = DM.build_initial_state(norm, 0)
    soc, _ = SM.build_initial_state(norm, 0, readout="mean")
    dm.eval()
    soc.eval()
    with torch.no_grad():
        soc.closure.gate.fill_(-30.0)
        assert float((dm(batch) - soc(batch)).abs().max()) < 1e-8


def test_control_matches_dcc_l_and_parameter_count():
    _, norm, batch = _fixtures()
    dm, _ = DM.build_initial_state(norm, 0)
    ctrl, info = SM.build_control_state(norm, 0, "cpu", reader_hidden=274)
    dm.eval()
    ctrl.eval()
    with torch.no_grad():
        assert float((dm(batch) - ctrl(batch)).abs().max()) < 1e-5
    assert info["parameters"] == 513021


def test_closure_changes_states_and_is_batch_invariant():
    cache, norm, batch = _fixtures()
    model, _ = SM.build_initial_state(norm, 0)
    model.eval()
    with torch.no_grad():
        model.closure.gate.fill_(-2.0)
        (a, b, w), traj = model.closure(*model.encoders(batch), batch, return_trajectory=True)
    assert len(traj) == 4
    assert float((traj[-1][1] - traj[0][1]).abs().max()) > 1e-5
    # batch invariance on 8 molecules alone vs 8+8
    with torch.no_grad():
        p8 = model(batch)
        batch16 = T1.collate(cache, np.arange(16), "P3N", valid=True)
        p16 = model(batch16)
    assert float((p8 - p16[:8]).abs().max()) < 1e-5


def test_gradients_reach_all_closure_parameters():
    _, norm, batch = _fixtures()
    model, _ = SM.build_initial_state(norm, 0)
    model.train()
    with torch.no_grad():
        model.closure.gate.fill_(-2.0)
    model(batch).abs().mean().backward()
    for name, p in model.closure.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0, name
