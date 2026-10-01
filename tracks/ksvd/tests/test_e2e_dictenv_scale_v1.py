"""Focused tests for ``e2e_dictenv_scale_v1`` (unified Small/Full scaling).

CPU only, no ZINC data and no checkpoints: the model tests use a random
structural dictionary and a random one-dimensional common subspace.  Covers the
pre-registered acceptance list at unit level: the closed-form parameter
inventory for m = 1/2/3, Small identity with the frozen ``LatentBridgeSEM108``,
the Full width / fixed-module contract, the eval-mode containment witness, the
identity mode, task-path gradients, single-call wiring with no ``h`` bypass and
the official-test blocker.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_scale_v1 as run


def _artifacts():
    rng = np.random.default_rng(0)
    dictionary = rng.normal(size=(cssd.PHI_DIM, cssd.K_ATOMS)).astype(np.float32)
    u, _ = np.linalg.qr(rng.normal(size=(cssd.PHI_DIM, 1)))
    subspace = cssd.CommonSubspace(
        components=np.asarray(u, dtype=np.float64),
        rms=np.asarray([1.0], dtype=np.float64),
        kind="q1",
    )
    return dictionary, subspace


# ---------------------------------------------------------------------------
# spec / accounting
# ---------------------------------------------------------------------------


def test_scale_spec_inventory_and_parameter_counts() -> None:
    assert sc.SMALL.parameters() == 106925
    assert sc.FULL.parameters() == 408651
    assert sc.AUDIT_ONLY_M2.parameters() == 233203
    assert sc.SMALL.d == 48 and sc.SMALL.k == 96 and sc.SMALL.p == 16
    assert sc.SMALL.reader_input == 302
    assert sc.FULL.d == 144 and sc.FULL.k == 288 and sc.FULL.p == 48
    assert sc.FULL.reader_input == 814
    assert sc.FULL.inventory()["task_dictionary_D_V"] == 82944
    assert sc.FULL.inventory()["fusion"] == 202266
    assert sc.FULL.inventory()["reader"] == 33385
    assert sc.FULL.parameters() - sc.FULL.inventory()["task_dictionary_D_V"] == 325707
    with pytest.raises(ValueError):
        sc.ScaleSpec("bad", 4)
    assert sc.scale_spec("full") is sc.FULL
    assert sc.scale_spec(2).parameters() == 233203


def test_readout_layout_is_derived_from_spec() -> None:
    small = sc.readout_layout(sc.SMALL)
    full = sc.readout_layout(sc.FULL)
    assert small["unary"] == {"first": (0, 48), "second": (48, 96), "count": (96, 97)}
    assert small["pair_per_bucket"] == {"first": (0, 16), "second": (16, 32), "count": (32, 33)}
    assert small["reader_input"] == 302
    assert full["unary"] == {"first": (0, 144), "second": (144, 288), "count": (288, 289)}
    assert full["pair_per_bucket"] == {"first": (0, 48), "second": (48, 96), "count": (96, 97)}
    assert full["pair_block_dim"] == 97
    assert full["reader_input"] == 814
    assert sc.reader_input_map(1).tolist() == list(range(302))
    wide = sc.reader_input_map(3)
    assert wide.shape == (814,)
    assert sc.pair_input_map(3).shape == (192,)


# ---------------------------------------------------------------------------
# parameter audit / identity
# ---------------------------------------------------------------------------


def test_small_build_matches_latent_bridge() -> None:
    dictionary, subspace = _artifacts()
    reference = lb.build_latent_bridge_model(dictionary, 0, subspace)
    small = sc.build_small_scale_model(dictionary, 0, subspace)
    state_reference = reference.state_dict()
    state_small = small.state_dict()
    assert sorted(state_reference) == sorted(state_small)
    assert all(
        torch.equal(state_reference[key], state_small[key]) for key in state_reference
    )
    audit = sc.scale_parameter_audit(small)
    assert audit["inventory_exact"] and audit["parameter_exact"]
    assert audit["actual_parameters"] == 106925
    assert audit["all_trainable"]


def test_full_parameter_width_and_fixed_module_contract() -> None:
    dictionary, subspace = _artifacts()
    small = sc.build_small_scale_model(dictionary, 0, subspace)
    full = sc.build_scale_model(dictionary, 0, subspace, sc.FULL)
    audit = sc.scale_parameter_audit(full)
    assert audit["inventory_exact"] and audit["parameter_exact"]
    assert audit["actual_parameters"] == 408651
    assert audit["within_budget"] and audit["all_trainable"]
    fixed = sc.compare_fixed_modules(small, full)
    assert fixed["identical"] and fixed["n_fixed_tensors"] > 0

    batch = run._synthetic_batch()
    widths = sc.capture_task_path_widths(full, batch, mask=cm.C6_MASK)
    assert widths == {
        "h": 144,
        "E": 144,
        "alpha": 288,
        "u": 48,
        "pair": 48,
        "reader_input": 814,
    }
    small_widths = sc.capture_task_path_widths(small, batch, mask=cm.C6_MASK)
    assert small_widths == {
        "h": 48,
        "E": 48,
        "alpha": 96,
        "u": 16,
        "pair": 16,
        "reader_input": 302,
    }


# ---------------------------------------------------------------------------
# containment / bridge math
# ---------------------------------------------------------------------------


def test_containment_witness_eval_mode() -> None:
    dictionary, subspace = _artifacts()
    small = sc.build_small_scale_model(dictionary, 0, subspace)
    for spec, tolerance in ((sc.AUDIT_ONLY_M2, 1e-4), (sc.FULL, 1e-4)):
        wide = sc.build_scale_model(dictionary, 0, subspace, spec)
        delta = sc.containment_witness_delta(small, wide, run._synthetic_batch(), mask=cm.C6_MASK)
        assert delta["plain_prediction_max_delta"] <= tolerance
        assert delta["masked_prediction_max_delta"] <= tolerance


def test_identity_and_formal_initialisation() -> None:
    dictionary, subspace = _artifacts()
    full = sc.build_scale_model(dictionary, 0, subspace, sc.FULL)
    full.eval()
    batch = run._synthetic_batch()
    h = sc._capture_bridge_input(full, batch, cm.C6_MASK)
    identity = lb.LatentDictionaryBridge(dim=sc.FULL.d, n_atoms=sc.FULL.k, lambda1=0.0, lambda2=0.0)
    with torch.no_grad():
        e_identity = identity(h)
        e_identity_repeat = identity(h)
        e_formal = full.local_dictionary_bridge(h)
    assert float(torch.norm(e_identity - h) / torch.norm(h)) <= 1e-6
    assert torch.equal(e_identity, e_identity_repeat)
    assert bool(torch.isfinite(e_formal).all())
    relative_sq = float(((e_formal - h) ** 2).sum() / (h * h).sum())
    assert relative_sq < 0.05


def test_task_path_gradients() -> None:
    dictionary, subspace = _artifacts()
    full = sc.build_scale_model(dictionary, 0, subspace, sc.FULL)
    full.eval()
    batch = run._synthetic_batch()
    gradients = run._task_gradients(full, batch, cm.C6_MASK)
    for key in (
        "task_grad_bridge_D_L",
        "task_grad_bridge_V_L",
        "task_grad_fusion_W1",
        "task_grad_fusion_W2",
        "task_grad_pair_projection",
        "task_grad_pair_encoder_W1",
        "task_grad_pair_encoder_W2",
        "task_grad_reader_W1",
    ):
        assert gradients[key] > 0.0, key
    assert np.isfinite(gradients["task_loss"])


def test_wiring_single_call_no_bypass_and_zero_code() -> None:
    dictionary, subspace = _artifacts()
    full = sc.build_scale_model(dictionary, 0, subspace, sc.FULL)
    full.eval()
    batch = run._synthetic_batch()
    calls: list[int] = []
    handle = full.local_dictionary_bridge.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        full(batch, mask=cm.C6_MASK)
    assert len(calls) == 1
    with torch.no_grad():
        full(batch)
    assert len(calls) == 2
    handle.remove()

    pair_inputs: dict[str, torch.Tensor] = {}
    pair_handle = full.pair_projection.register_forward_pre_hook(
        lambda _module, inputs: pair_inputs.__setitem__("value", inputs[0].detach())
    )
    with torch.no_grad():
        _prediction, aux = full(batch, mask=cm.C6_MASK, return_aux=True)
    pair_handle.remove()
    assert torch.equal(pair_inputs["value"], aux["E"].detach())

    h = sc._capture_bridge_input(full, batch, cm.C6_MASK)
    with torch.no_grad():
        e_formal = full.local_dictionary_bridge(h)
    assert float((e_formal - h).abs().max()) > 0.0  # no identity bypass

    full.set_bridge_intervention(zero_code=True)
    with torch.no_grad():
        e_zero = full.local_dictionary_bridge(h)
        env_zero = full.environments(full.code(batch.dict_phi), batch)
    full.clear_bridge_intervention()
    assert torch.equal(e_zero, torch.zeros_like(e_zero))
    assert torch.equal(env_zero, torch.zeros_like(env_zero))


def test_official_test_blocker_raises() -> None:
    with pytest.raises(RuntimeError):
        sc.official_test_blocker({"official_test_loaded": True})
    sc.official_test_blocker({"official_test_loaded": False})
    assert sem.SEM108Model is not None  # the Small path keeps the frozen parent
