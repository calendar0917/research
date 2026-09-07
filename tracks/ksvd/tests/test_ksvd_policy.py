"""Test-access policy lives in the runtime control plane, not in the runner.

The runner never re-derives terminal/test_policy semantics; policy is
resolved once by ``runtime.policy.resolve_test_access`` and handed to the
runner inside ``RunContext.test_access``.
"""

from __future__ import annotations

import pytest

from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.policy import BLOCKED, GRANTED, UNGUARDED, resolve_test_access


# ---------------------------------------------------------------------------
# resolve_test_access
# ---------------------------------------------------------------------------


def test_terminal_policy_blocks_scratch():
    assert resolve_test_access({"test_policy": "terminal"}, "scratch") == BLOCKED


def test_terminal_policy_blocks_screen():
    assert resolve_test_access({"test_policy": "terminal"}, "screen") == BLOCKED


def test_terminal_policy_blocks_confirm():
    assert resolve_test_access({"test_policy": "terminal"}, "confirm") == BLOCKED


def test_terminal_policy_grants_terminal():
    assert resolve_test_access({"test_policy": "terminal"}, "terminal") == GRANTED


def test_no_protocol_is_unguarded():
    assert resolve_test_access(None, "scratch") == UNGUARDED
    assert resolve_test_access(None, "terminal") == UNGUARDED


def test_no_declared_policy_is_unguarded():
    assert resolve_test_access({}, "scratch") == UNGUARDED


def test_legacy_always_policy_is_granted():
    assert resolve_test_access({"test_policy": "always"}, "scratch") == GRANTED


def test_unknown_policy_fails_loud():
    with pytest.raises(ValueError):
        resolve_test_access({"test_policy": "maybe"}, "scratch")


# ---------------------------------------------------------------------------
# the ZINC runner only obeys context.test_access
# ---------------------------------------------------------------------------


def _fake_legacy_result() -> dict:
    return {
        "status": "completed",
        "runtime": {"seconds": 1.0},
        "data": {"sizes": {"train": 10000, "valid": 1000, "test": 1000}},
        "evaluation": {
            "parameters": 100,
            "valid": {
                "mae": 0.5,
                "best_mae": 0.5,
                "selected_epoch": 2,
                "epochs_run": 2,
            },
            "test_after_train_valid_refit": {"mae": 0.6, "epochs_run": 2},
        },
    }


def _fake_legacy(monkeypatch, tmp_path):
    fake = type("FakeLegacy", (), {})()
    fake.run = lambda input_path: _fake_legacy_result()
    import ksvd_research.runners.zinc_patch_path_pooling as runner_module

    monkeypatch.setattr(runner_module, "_legacy_module", lambda: fake)
    return runner_module


def test_runner_uses_context_test_access(monkeypatch, tmp_path):
    runner = _fake_legacy(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifacts").mkdir()
    config = {"test_policy": "always", "seed": 0, "model": {"epochs": 2}}
    context = RunContext(
        run_id="r-1",
        run_dir=run_dir,
        artifact_dir=run_dir / "artifacts",
        mode="scratch",
        protocol={"test_policy": "terminal"},
        study=None,
        test_access="blocked",
    )
    result: RunResult = runner.run(config, context)
    assert result.status == "completed"
    assert result.metrics["test_access"] == "blocked"
    from ksvd_research.runtime.serialization import load_yaml

    injected = load_yaml(run_dir / "legacy_input.yaml")
    assert injected["test_policy"] == "no_test"


def test_runner_obeys_granted_access(monkeypatch, tmp_path):
    runner = _fake_legacy(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifacts").mkdir()
    config = {"test_policy": "no_test", "seed": 0}
    context = RunContext(
        run_id="r-2",
        run_dir=run_dir,
        artifact_dir=run_dir / "artifacts",
        mode="terminal",
        protocol={"test_policy": "terminal"},
        study=None,
        test_access="granted",
    )
    result = runner.run(config, context)
    assert result.metrics["test_access"] == "granted"
    assert result.metrics["test_after_train_valid_refit_mae"] == 0.6
