"""Fast tests for the composition-transfer round (no data cache required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(CODE),
           str(CODE.parents[1] / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zct_split as SP  # noqa: E402
import zct_support as SU  # noqa: E402
import zct_train as ZT  # noqa: E402


def test_ramp_shape():
    assert ZT.ramp(50) == 0.0
    assert ZT.ramp(51) == pytest.approx(1.0 / 50.0)
    assert ZT.ramp(100) == 1.0
    assert ZT.ramp(600) == 1.0
    assert all(ZT.ramp(e) <= ZT.ramp(e + 1) for e in range(1, 120))


def test_schedule_index_matches_recipe():
    total = 79 * 600
    assert ZT.lr_at(1, total) == pytest.approx(1e-3 / 500.0)
    assert ZT.lr_at(500, total) == pytest.approx(1e-3)
    assert ZT.lr_at(total, total) == pytest.approx(1e-5)
    assert ZT.lr_at(1000, total) > ZT.lr_at(40000, total) > ZT.lr_at(total, total)


def test_random_partition_balanced():
    rng = np.random.default_rng([0, 1, 3])
    q = SP.random_partition(128, rng)
    assert q.dtype == bool
    assert q.sum() == 64


def test_mask_lookup_roundtrip():
    def pack(mask):
        return np.packbits(np.concatenate([np.asarray(mask, bool),
                                           np.zeros(128 - len(mask), bool)]))

    part = {
        "batch_rows": np.asarray([16, 128], dtype=np.int16),
        "r_packed": np.stack([pack([1] * 8 + [0] * 8), pack([1] * 64 + [0] * 64)]),
        "s_packed": np.stack([pack([1] * 4 + [0] * 12), pack([1] * 32 + [0] * 96)]),
        "summary": {"steps_per_epoch": 1},
    }
    m = SP.mask_for(part, 1, 0, "R", 16, steps_per_epoch=1)
    assert m.sum() == 8 and m[0]
    m = SP.mask_for(part, 2, 0, "S", 128, steps_per_epoch=1)
    assert m.sum() == 32


def test_smd_zero_for_identical_groups():
    values = np.array([1.0, 2.0, 2.0, 1.0])
    q = np.array([True, False, True, False])
    assert SP.smd(values, q, 1.0) == pytest.approx(0.0)


def test_fit_batches_cover_rows_once():
    base = np.arange(1000, dtype=np.int64)
    donor = np.arange(1000, 1200, dtype=np.int64)
    batches = SU.fit_batches(base, donor, epoch=1, seed=0, batch_size=128)
    seen_b = np.sort(np.concatenate([b for b, _ in batches]))
    seen_d = np.sort(np.concatenate([d for _, d in batches]))
    assert np.array_equal(seen_b, base)
    assert np.array_equal(seen_d, donor)
    assert all(len(b) + len(d) <= 128 for b, d in batches)
    steps = len(batches)
    assert steps == int(np.ceil((len(base) + len(donor)) / 128))
    again = SU.fit_batches(base, donor, epoch=1, seed=0, batch_size=128)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
               for a, b in zip(batches, again))


def test_alpha_zero_virtual_step_is_identity():
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 1)
    params = dict(lin.named_parameters())
    values = list(params.values())
    x = torch.randn(8, 3)
    g = torch.autograd.grad(torch.nn.functional.mse_loss(lin(x), torch.randn(8)), values,
                            create_graph=True)
    gn = torch.sqrt(sum((t * t).sum() for t in g))
    c = torch.clamp(5.0 / (gn + 1e-8), max=1.0)
    shifted = {k: p - 0.0 * c * gi for k, p, gi in zip(params, values, g)}
    with torch.no_grad():
        # alpha = 0 leaves the parameters bitwise unchanged
        assert torch.equal(shifted["weight"], lin.weight)
        assert torch.equal(shifted["bias"], lin.bias)


def test_test_report_markdown_is_well_formed():
    """The one-pass test read's report must stay a table of frozen checkpoints."""
    import zct_analyze as AN

    run = {"run": "R_s0_k04", "which": "best", "arm": "R", "seed": 0, "kappa": 0.4,
           "epoch": 486, "test_mae": 0.0704, "valid_mae": 0.0785,
           "test_minus_valid": -0.0081, "median_ae": 0.0364,
           "trimmed_top1pct_mae": 0.0605,
           "groups": {"seen_ordinary": 0.0469, "unseen_ordinary": 0.1174,
                      "seen_tail": 0.1177, "unseen_tail": 0.2195}}
    summary = {"test_read": "one pass, post-selection", "n": 1000, "source": "x.csv",
               "source_sha256": "0" * 64,
               "group_sizes": {"seen_ordinary": 705, "unseen_ordinary": 218,
                               "seen_tail": 51, "unseen_tail": 26},
               "runs": {"R_s0_k04_best": run},
               "comparisons": {"R_s0_k04_best_minus_E_s0_best": {
                   "delta_mean": -0.008, "ci95": [-0.012, -0.004], "delta_median": -0.005,
                   "frac_molecules_a_better": 0.565, "sign_test_p": 0.0002}}}
    md = AN._test_md(summary)
    assert "R_s0_k04_best" in md and "0.070400" in md and "-0.008100" in md
    assert "0.11740" in md and "705" in md
    assert "post-selection" in md
    # no accidental float formatting of None values
    assert "None" not in md
