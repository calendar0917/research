from __future__ import annotations

import tempfile
from pathlib import Path

import v2_common as C
import v2_training as T


def _v1_snapshot():
    return C.assert_v1_unchanged()


def test_v1_frozen_manifest_and_references_are_unchanged():
    snap = _v1_snapshot()
    assert snap["ok"] is True, snap["failed"]
    assert snap["n_checks"] > 0
    # The v1 freeze records its own manifest and every prediction-sidecar hash.
    paths = [row["path"] for row in snap["checks"]]
    assert any("MANIFEST" in path for path in paths)
    assert any("monitor_metrics" in path for path in paths)
    assert all(row["ok"] for row in snap["checks"])


def test_v2_smoke_does_not_touch_v1():
    before = _v1_snapshot()
    with tempfile.TemporaryDirectory() as td:
        T.train_job(
            T.TrainJob(
                key="v1_guard_smoke",
                arm="A",
                fold=0,
                recipe="R0",
                seed=0,
                device="cpu",
                out_root=Path(td),
                smoke_train_limit=32,
                smoke_epochs=1,
            )
        )
    after = _v1_snapshot()
    assert before["ok"] and after["ok"]
    assert before["checks"] == after["checks"]
