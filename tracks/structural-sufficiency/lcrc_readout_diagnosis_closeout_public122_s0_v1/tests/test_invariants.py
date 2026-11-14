"""Fixture + invariants for the closeout readout probe.

The synthetic-h fixtures are *computation-path* checks, not scientific results:

* the three arms agree at step 0 (POST/PRE have a zero delta last layer);
* POST/PRE share identical initial psi/delta (private seed 1101);
* ``mean_v psi(h_v)`` and ``psi(mean_v h_v)`` differ for non-degenerate h, with
  correct per-molecule shapes (no flatten/broadcast error);
* with a zero delta last layer the first backward only touches the last layer,
  and after one effective update the gradient reaches the delta first layer and
  ``psi`` (this is why "zero last layer" alone cannot explain a weak response
  after 1800 steps).

No data / checkpoint / GPU is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import rc_common as R  # noqa: E402
import rc_probe as P  # noqa: E402
import lb_selection as SEL  # noqa: E402


def _rand_state(prefix: str, dims: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    out = {}
    for i, (o, inp) in enumerate(dims):
        out[f"{prefix}lins.{i}.weight"] = torch.randn(o, inp, generator=g)
        out[f"{prefix}lins.{i}.bias"] = torch.randn(o, generator=g)
    return out


def _tensors(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    counts = [3, 1, 4, 2, 5]
    n_atoms = sum(counts)
    h = torch.randn(n_atoms, 64, generator=g)
    ag = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    zp = torch.randn(len(counts), 171, generator=g)
    return h, ag, zp, len(counts)


def _heads():
    f_state = _rand_state("f.", [(64, 64), (1, 64)])
    rho_state = _rand_state("rho.", [(64, 171), (1, 64)])
    return {r: P.ProbeHead(r, f_state, rho_state) for r in R.PROBE_ARMS}, f_state, rho_state


def test_new_param_count_is_exact():
    heads, _, _ = _heads()
    assert heads["P_WARM"].n_new_params() == 0
    assert heads["P_POST"].n_new_params() == R.PROBE_NEW_PARAMS == 14705
    assert heads["P_PRE"].n_new_params() == R.PROBE_NEW_PARAMS == 14705


def test_three_arms_identical_at_step0():
    heads, _, _ = _heads()
    h, ag, zp, ng = _tensors()
    with torch.no_grad():
        y_warm = heads["P_WARM"].forward_frozen(h, ag, zp, ng)
        for rule in ("P_POST", "P_PRE"):
            y = heads[rule].forward_frozen(h, ag, zp, ng)
            assert torch.allclose(y, y_warm, atol=1e-7), rule
            assert torch.allclose(y, heads[rule].warm_only(h, ag, zp, ng), atol=1e-7), rule


def test_post_pre_share_initial_psi_delta():
    heads, _, _ = _heads()
    a, b = heads["P_POST"].state_dict(), heads["P_PRE"].state_dict()
    for k in a:
        if k.startswith("psi.") or k.startswith("delta."):
            assert torch.equal(a[k], b[k]), k


def test_mean_psi_and_psi_mean_differ_no_broadcast_error():
    heads, _, _ = _heads()
    head = heads["P_POST"]
    h, ag, zp, ng = _tensors(seed=3)
    with torch.no_grad():
        extra = head.extra_feature(h, ag, ng)
        assert extra.shape == (ng, 16)
        n = head._counts(h, ag, ng)
        den = n.clamp(min=1.0).unsqueeze(1)
        mean_h = h.new_zeros((ng, h.shape[1])).index_add_(0, ag, h) / den
        psi_mean = head.psi(mean_h)
        psi_h = head.psi(h)
        mean_psi = psi_h.new_zeros((ng, psi_h.shape[1])).index_add_(0, ag, psi_h) / den
        assert psi_mean.shape == (ng, 16) and mean_psi.shape == (ng, 16)
        # for non-degenerate h these two pooled orders must differ
        assert float((mean_psi - psi_mean).abs().max()) > 1e-4
        # POST uses psi(mean h); PRE uses mean psi(h)
        assert torch.allclose(extra, psi_mean, atol=1e-6)
        head_pre = heads["P_PRE"]
        extra_pre = head_pre.extra_feature(h, ag, ng)
        assert torch.allclose(extra_pre, mean_psi, atol=1e-6)


def test_zero_last_layer_first_step_and_then_gradient_flow():
    heads, _, _ = _heads()
    head = heads["P_POST"]
    h, ag, zp, ng = _tensors(seed=11)
    target = torch.randn(ng, generator=torch.Generator().manual_seed(12))

    def loss_and_backward():
        pred = head.forward_frozen(h, ag, zp, ng)
        loss = F.mse_loss(pred, target)
        head.zero_grad(set_to_none=True)
        loss.backward()
        return loss

    loss_and_backward()
    psi_norm = P._grad_norm(list(head.psi.parameters()))
    first_norm = P._grad_norm([head.delta.lins[0].weight, head.delta.lins[0].bias])
    last_norm = P._grad_norm([head.delta.lins[1].weight, head.delta.lins[1].bias])
    assert last_norm > 0
    assert psi_norm == 0.0, psi_norm
    assert first_norm == 0.0, first_norm

    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    opt.step()
    loss_and_backward()
    assert P._grad_norm([head.delta.lins[0].weight, head.delta.lins[0].bias]) > 0.0
    assert P._grad_norm(list(head.psi.parameters())) > 0.0


def test_rsel_includes_step0():
    sel = SEL.ProbeSelector("regression")
    sel.consider(0, {"rmse": 0.70, "mae": 0.50}, {"x": torch.zeros(1)})
    sel.consider(36, {"rmse": 0.75, "mae": 0.55}, {"x": torch.ones(1)})
    rules = sel.freeze()
    assert rules["R_SEL"]["update"] == 0
    assert rules["A_SEL"]["update"] == 0


def test_hcache_identity_is_not_dataset_only():
    rows = {"train": "a", "valid": "b", "test": "c"}
    norms = {f"struct_norm_{R.DATASET}.npz": "s", f"target_norm_{R.DATASET}.npz": "t"}
    i1 = R.hcache_identity(ckpt_state_hash="state-a", spec_hash="spec", norms=norms, rows_hashes=rows)
    i2 = R.hcache_identity(ckpt_state_hash="state-b", spec_hash="spec", norms=norms, rows_hashes=rows)
    assert i1["identity_sha256"] != i2["identity_sha256"]
    p1 = R.hcache_path(i1).name
    assert i1["identity_sha256"][:16] in p1
    assert "hcache_Lipophilicity" not in p1  # old dataset-only naming must not be reused
