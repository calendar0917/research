"""Focused tests for ``e2e_dictenv_latent_bridge_v1`` (SEM108 + latent bridge).

CPU only, no ZINC data and no checkpoints: the model tests use a random
structural dictionary and a random one-dimensional common subspace.  Covers the
pre-registered acceptance list at unit level: frame initialisation, the
identity mode, row locality / invariance, the exact parameter contract, shared
initial state with Sem108, the single insertion point on both paths, bridge
gradients, reset-to-init and the official-test blocker.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_latent_bridge_v1 as run


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
# frame / bridge math
# ---------------------------------------------------------------------------


def test_frame_initialization_is_a_unit_column_tight_frame() -> None:
    dictionary, values = lb.frame_initialization()
    assert dictionary.shape == (lb.BRIDGE_DIM, lb.BRIDGE_ATOMS)
    assert values.shape == (lb.BRIDGE_ATOMS, lb.BRIDGE_DIM)
    assert np.allclose(np.linalg.norm(dictionary, axis=0), 1.0, atol=1e-6)
    assert np.allclose(dictionary @ dictionary.T, 2.0 * np.eye(lb.BRIDGE_DIM), atol=1e-6)
    assert np.array_equal(values, dictionary.T)
    # two distinct orthonormal bases, not a duplicate block
    assert not np.allclose(dictionary[:, : lb.BRIDGE_DIM], dictionary[:, lb.BRIDGE_DIM :])


def test_bridge_parameter_count_exact() -> None:
    bridge = lb.LatentDictionaryBridge()
    assert bridge.parameter_count() == lb.BRIDGE_PARAMETERS == 9216
    names = sorted(name for name, _ in bridge.named_parameters())
    assert names == ["D_L", "V_L"]


def test_identity_mode_and_zero_object() -> None:
    bridge = lb.LatentDictionaryBridge(lambda1=0.0, lambda2=0.0)
    generator = torch.Generator().manual_seed(0)
    h = torch.randn(64, lb.BRIDGE_DIM, generator=generator)
    h[0] = 0.0
    with torch.no_grad():
        e, aux = bridge(h, return_aux=True)
    relative = float(torch.norm(e - h) / torch.norm(h))
    assert relative <= 1e-6
    assert torch.isfinite(e).all()
    assert torch.equal(e[0], torch.zeros(lb.BRIDGE_DIM))
    assert torch.isfinite(aux["alpha"]).all()


def test_production_threshold_is_variable_density_and_finite() -> None:
    bridge = lb.LatentDictionaryBridge()
    generator = torch.Generator().manual_seed(1)
    h = torch.randn(256, lb.BRIDGE_DIM, generator=generator)
    with torch.no_grad():
        e, aux = bridge(h, return_aux=True)
    assert torch.isfinite(e).all() and torch.isfinite(aux["alpha"]).all()
    stats = lb.bridge_code_stats(aux["alpha"])
    assert 0.0 < stats["mean_nonzero_fraction"] < 1.0
    assert stats["l0_max"] <= lb.BRIDGE_ATOMS  # variable density, no fixed l0 claim


def test_bridge_is_row_local() -> None:
    bridge = lb.LatentDictionaryBridge()
    generator = torch.Generator().manual_seed(2)
    h = torch.randn(48, lb.BRIDGE_DIM, generator=generator)
    permutation = torch.randperm(len(h), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        e = bridge(h)
        e_permuted = bridge(h[permutation])
        e_first = bridge(h[:17])
    assert torch.allclose(e_permuted, e[permutation], atol=1e-6)
    assert torch.allclose(e_first, e[:17], atol=1e-6)


def test_zero_code_intervention_and_reset() -> None:
    bridge = lb.LatentDictionaryBridge()
    h = torch.randn(16, lb.BRIDGE_DIM)
    bridge.set_zero_code(True)
    with torch.no_grad():
        assert torch.equal(bridge(h), torch.zeros_like(h))
    bridge.set_zero_code(False)
    with torch.no_grad():
        bridge.D_L.add_(1.0)
    bridge.reset_to_initialization()
    assert torch.equal(bridge.D_L.detach(), bridge.D_init)
    assert torch.equal(bridge.V_L.detach(), bridge.V_init)


def test_bridge_accepts_task_gradient_on_both_matrices() -> None:
    bridge = lb.LatentDictionaryBridge()
    h = torch.randn(32, lb.BRIDGE_DIM)
    loss = (bridge(h) ** 2).mean()
    bridge.zero_grad()
    loss.backward()
    assert bridge.D_L.grad is not None and float(bridge.D_L.grad.norm()) > 0.0
    assert bridge.V_L.grad is not None and float(bridge.V_L.grad.norm()) > 0.0


# ---------------------------------------------------------------------------
# model wiring
# ---------------------------------------------------------------------------


def test_model_shared_init_and_single_insertion_point() -> None:
    dictionary, subspace = _artifacts()
    sem_model = sem.build_sem108_model(dictionary, 0, subspace)
    candidate = lb.build_latent_bridge_model(dictionary, 0, subspace)
    state_sem = sem_model.state_dict()
    state_candidate = candidate.state_dict()
    mismatch = [
        key
        for key, value in state_sem.items()
        if key in state_candidate
        and not key.startswith("local_dictionary_bridge.")
        and not torch.equal(value, state_candidate[key])
    ]
    assert not mismatch
    assert not hasattr(candidate, "B")
    audit = lb.bridge_parameter_audit(candidate)
    assert audit["bridge_exact"] and audit["candidate_exact"]

    data = run._synthetic_batch()
    sem_model.eval()
    candidate.eval()
    calls: list[int] = []
    handle = candidate.local_dictionary_bridge.register_forward_hook(
        lambda *args: calls.append(1)
    )
    with torch.no_grad():
        candidate(data, mask=cm.C6_MASK)
    assert len(calls) == 1
    with torch.no_grad():
        candidate(data)
    assert len(calls) == 2
    handle.remove()

    # E enters unary and the static pair projection on the synthetic batch
    pair_inputs: dict[str, torch.Tensor] = {}
    pair_handle = candidate.pair_projection.register_forward_pre_hook(
        lambda _module, inputs: pair_inputs.__setitem__("value", inputs[0].detach())
    )
    with torch.no_grad():
        _prediction, aux = candidate(data, mask=cm.C6_MASK, return_aux=True)
    pair_handle.remove()
    assert "value" in pair_inputs
    assert torch.equal(pair_inputs["value"], aux["E"].detach())
    assert float(aux["E"].abs().sum()) > 0.0


def test_official_test_blocker_raises() -> None:
    with pytest.raises(RuntimeError):
        lb.official_test_blocker({"official_test_loaded": True})
    lb.official_test_blocker({"official_test_loaded": False})


def test_cached_interface_geometry_unchanged() -> None:
    # the bridge keeps the parent environment width; geometry stays Sem108.
    assert lb.BRIDGE_DIM == sem.SEM_FUSION_OUT == 48
    assert lb.BRIDGE_ATOMS == 2 * lb.BRIDGE_DIM
    assert (lb.BRIDGE_LAMBDA1, lb.BRIDGE_LAMBDA2, lb.BRIDGE_STEPS) == (0.05, 0.01, 16)
