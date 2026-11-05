"""CPU-only correctness tests for the PCBA batch-construction paths.

These tests need the frozen 4.5 GB memory-mapped feature cache, so they are
marked ``slow`` and are skipped when the cache is absent (fresh clone).

The gates here are the CPU half of the correctness threshold in the round:

* ``reference``  = ``lcrc_data.collate`` (frozen, what training runs today)
* ``instrumented`` = timing mirror of ``reference``
* ``fast`` = the optimised construction (only present after the optimisation)

Every field must match in shape, dtype, order and value -- not "close", equal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

pytestmark = pytest.mark.slow

import pf_common as PF  # noqa: E402


def _need_cache() -> None:
    if not Path(PF.CACHE_PATH).is_dir():
        pytest.skip(f"frozen feature cache absent: {PF.CACHE_PATH}")


@pytest.fixture(scope="module")
def measured_batches():
    _need_cache()
    inputs = PF.load_inputs()
    batches = PF.fixed_batches(inputs.train_rows, PF.TIMED_STEPS)
    return inputs, PF.diverse_batches(inputs.cache, batches)


def _assert_same_batch(got, want, label: str) -> None:
    assert set(got.keys()) == set(want.keys()), f"{label}: key set differs"
    assert list(got.keys()) == list(want.keys()), f"{label}: key order differs"
    for key in want:
        a, b = got[key], want[key]
        if not hasattr(b, "shape"):
            assert a == b, f"{label}.{key}: {a!r} != {b!r}"
            continue
        assert hasattr(a, "shape"), f"{label}.{key}: not a tensor ({type(a)!r})"
        assert tuple(a.shape) == tuple(b.shape), f"{label}.{key}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        assert a.dtype == b.dtype, f"{label}.{key}: dtype {a.dtype} != {b.dtype}"
        assert a.device.type == b.device.type, f"{label}.{key}: device {a.device} != {b.device}"
        assert np.array_equal(a.numpy(), b.numpy()), f"{label}.{key}: values differ"


def test_frozen_reference_batch_fields_and_dtypes(measured_batches):
    """The reference output has exactly the documented fields and dtypes."""
    import torch

    inputs, chosen = measured_batches
    import pf_collate as PC_C

    for name, rows in chosen.items():
        out = PC_C.collate_reference(inputs.cache, rows.tolist())
        assert out["n_graphs"] == len(rows)
        assert set(out) == {"n_graphs", *PC_C.FIELD_ORDER}
        counts = {"atom_feat": (int(out["atom_counts"].sum()), 51),
                  "atom_scalar": (int(out["atom_counts"].sum()), 9),
                  "bond_feat": (int(out["bond_counts"].sum()), 13),
                  "global": (len(rows), 42)}
        for key, shape in counts.items():
            assert tuple(out[key].shape) == shape, (name, key)
        for key in PC_C.FIELD_ORDER:
            assert isinstance(out[key], torch.Tensor)
        for key in ("atom_feat", "atom_scalar", "bond_feat", "global"):
            assert out[key].dtype == torch.float32, (name, key)
        for key in PC_C.FIELD_ORDER:
            if key not in ("atom_feat", "atom_scalar", "bond_feat", "global"):
                assert out[key].dtype == torch.int64, (name, key)


def test_instrumented_collate_matches_frozen(measured_batches):
    """The timing mirror must be bit-identical to the frozen collate."""
    inputs, chosen = measured_batches
    import pf_collate as PC_C

    for name, rows in chosen.items():
        want = PC_C.collate_reference(inputs.cache, rows.tolist())
        got, marks = PC_C.collate_instrumented(inputs.cache, rows.tolist())
        _assert_same_batch(got, want, name)
        assert len(marks.rows) >= 8


def test_index_ranges_property_different_sizes():
    """``_ranges`` semantics: contiguous runs, per-molecule local offsets."""
    _need_cache()
    import pf_collate as PC_C

    ptr = np.asarray([0, 3, 3, 4, 9], dtype=np.int64)  # one empty molecule in the middle
    rows = [
        np.array([0, 1, 2, 3]),
        np.array([0, 3]),
        np.array([2, 2]),
        np.array([3, 0]),
    ]
    for r in rows:
        idx, batch, local, counts = PC_C.ranges_reference(ptr, r)
        want_idx, want_batch, want_local = [], [], []
        for i, row in enumerate(r):
            c = int(ptr[row + 1] - ptr[row])
            want_idx.extend(range(int(ptr[row]), int(ptr[row + 1])))
            want_batch.extend([i] * c)
            want_local.extend(range(c))
        assert idx.tolist() == want_idx
        assert batch.tolist() == want_batch
        assert local.tolist() == want_local
        assert counts.tolist() == [int(ptr[x + 1] - ptr[x]) for x in r]


def test_collate_rejects_bad_rows(measured_batches):
    inputs, _ = measured_batches
    import pf_collate as PC_C

    with pytest.raises(ValueError):
        PC_C.collate_reference(inputs.cache, [0, 0])
    with pytest.raises(ValueError):
        PC_C.collate_reference(inputs.cache, [-1])
    with pytest.raises(ValueError):
        PC_C.collate_reference(inputs.cache, [int(inputs.cache["n_molecules"])])


# ---------------------------------------------------------------------------
# fast path (added by the optimisation; skipped when not present)
# ---------------------------------------------------------------------------


def test_fast_collate_matches_frozen(measured_batches):
    """The optimised collate must reproduce the frozen output exactly."""
    inputs, chosen = measured_batches
    import pf_collate as PC_C

    if not hasattr(PC_C, "collate_fast"):
        pytest.skip("no fast path in this revision")
    for name, rows in chosen.items():
        want = PC_C.collate_reference(inputs.cache, rows.tolist())
        got = PC_C.collate_fast(inputs.cache, rows.tolist())
        _assert_same_batch(got, want, name)


def test_fast_collate_matches_frozen_on_many_measured_batches():
    """All 300 measured batches, not just the diverse six."""
    _need_cache()
    import pf_collate as PC_C

    if not hasattr(PC_C, "collate_fast"):
        pytest.skip("no fast path in this revision")
    inputs = PF.load_inputs()
    batches = PF.fixed_batches(inputs.train_rows, PF.TIMED_STEPS)
    for i, rows in enumerate(batches):
        want = PC_C.collate_reference(inputs.cache, rows.tolist())
        got = PC_C.collate_fast(inputs.cache, rows.tolist())
        if i == 0:
            _assert_same_batch(got, want, f"batch{i}")
        else:  # cheaper comparison for the rest
            for key in want:
                a, b = got[key], want[key]
                if hasattr(b, "shape"):
                    assert a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape), (i, key)
                    assert np.array_equal(a.numpy(), b.numpy()), (i, key)
                else:
                    assert a == b, (i, key)
