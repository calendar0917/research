"""Comparison correctness: protocol metric schema, direction safety, no fallback.

The whole point of this file: the control plane must never produce a
confidently wrong ranking.  Requested metrics are read exactly, missing
metrics are never substituted, directions are never mixed, and an unknown
direction is a hard error.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from ksvd_research.cli import (
    _cmd_compare,
    manifest_metric,
    metric_for_protocol,
)


def _manifest(run_id: str, protocol_id: str, metrics: dict, **extra) -> dict:
    return {
        "run_id": run_id,
        "status": "completed",
        "protocol_id": protocol_id,
        "mode": "terminal",
        "test_access": "granted",
        "metrics": dict(metrics),
        **extra,
    }


def test_protocol_metric_reads_nested_metric():
    assert metric_for_protocol("molhiv-cross-scaffold-interaction") == (
        "valid_roc_auc",
        "higher_is_better",
    )
    assert metric_for_protocol("zinc-context-gap") == ("valid_mae", "lower_is_better")


def test_invalid_metric_direction_raises(monkeypatch):
    import ksvd_research.cli as cli

    monkeypatch.setattr(
        cli,
        "load_protocol",
        lambda protocol_id: {"metric": {"ranking_metric": "x", "metric_direction": "sideways"}},
    )
    with pytest.raises(ValueError):
        metric_for_protocol("anything")


def test_legacy_top_level_metric_direction_still_supported(monkeypatch):
    import ksvd_research.cli as cli

    monkeypatch.setattr(
        cli,
        "load_protocol",
        lambda protocol_id: {"ranking_metric": "fancy", "metric_direction": "higher_is_better"},
    )
    assert metric_for_protocol("legacy") == ("fancy", "higher_is_better")


def test_manifest_metric_never_falls_back_to_valid_mae():
    manifest = {"metrics": {"valid_mae": 0.18}}
    assert manifest_metric(manifest, "test_after_train_valid_refit_mae") is None
    assert manifest_metric(manifest, "valid_mae") == 0.18


def test_compare_higher_is_better(monkeypatch, capsys):
    import ksvd_research.cli as cli

    manifests = [
        _manifest("run-a", "molhiv-cross-scaffold-interaction", {"valid_roc_auc": 0.80}),
        _manifest("run-b", "molhiv-cross-scaffold-interaction", {"valid_roc_auc": 0.84}),
    ]
    monkeypatch.setattr(cli, "list_runs", lambda study_id=None, status=None, limit=None, **kw: manifests)
    code = _cmd_compare(Namespace(study="molhiv", metric=None, override=False, run_ids=[]))
    assert code == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("#")]
    assert lines[0].split()[:2] == ["#1", "run-b"]
    assert lines[1].split()[:2] == ["#2", "run-a"]


def test_compare_lower_is_better(monkeypatch, capsys):
    import ksvd_research.cli as cli

    manifests = [
        _manifest("run-a", "zinc-context-gap", {"valid_mae": 0.20}),
        _manifest("run-b", "zinc-context-gap", {"valid_mae": 0.18}),
    ]
    monkeypatch.setattr(cli, "list_runs", lambda study_id=None, status=None, limit=None, **kw: manifests)
    code = _cmd_compare(Namespace(study="zinc", metric=None, override=False, run_ids=[]))
    assert code == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("#")]
    assert lines[0].split()[:2] == ["#1", "run-b"]
    assert lines[1].split()[:2] == ["#2", "run-a"]


def test_compare_missing_metric_is_not_ranked(monkeypatch, capsys):
    import ksvd_research.cli as cli

    manifests = [
        _manifest("run-a", "zinc-context-gap", {"valid_mae": 0.18}),
        _manifest(
            "run-b",
            "zinc-context-gap",
            {"valid_mae": 0.19, "test_after_train_valid_refit_mae": 0.16},
        ),
    ]
    monkeypatch.setattr(cli, "list_runs", lambda study_id=None, status=None, limit=None, **kw: manifests)
    code = _cmd_compare(
        Namespace(
            study="zinc",
            metric="test_after_train_valid_refit_mae",
            override=False,
            run_ids=[],
        )
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "MISSING METRIC:" in out
    assert "run-a" in out.split("MISSING METRIC:")[1].splitlines()[1]
    lines = [line for line in out.splitlines() if line.startswith("#")]
    assert len(lines) == 1
    assert lines[0].split()[:2] == ["#1", "run-b"]


def test_compare_all_missing_returns_error(monkeypatch, capsys):
    import ksvd_research.cli as cli

    manifests = [
        _manifest("run-a", "zinc-context-gap", {"valid_mae": 0.18}),
        _manifest("run-b", "zinc-context-gap", {"valid_mae": 0.19}),
    ]
    monkeypatch.setattr(cli, "list_runs", lambda study_id=None, status=None, limit=None, **kw: manifests)
    code = _cmd_compare(
        Namespace(study="zinc", metric="test_after_train_valid_refit_mae", override=False, run_ids=[])
    )
    assert code != 0
    assert "no run has metric" in capsys.readouterr().out


def test_compare_direction_conflict_refuses(monkeypatch, capsys):
    import ksvd_research.cli as cli

    manifests = [
        _manifest("run-a", "zinc-context-gap", {"test_refit_mae": 0.18}),
        _manifest("run-b", "molhiv-cross-scaffold-interaction", {"test_refit_mae": 0.84}),
    ]
    monkeypatch.setattr(cli, "list_runs", lambda study_id=None, status=None, limit=None, **kw: manifests)
    monkeypatch.setattr(
        cli,
        "metric_for_protocol",
        lambda protocol_id: {
            "zinc-context-gap": ("test_refit_mae", "lower_is_better"),
            "molhiv-cross-scaffold-interaction": ("test_refit_mae", "higher_is_better"),
        }[protocol_id],
    )
    code = _cmd_compare(
        Namespace(study="z", metric="test_refit_mae", override=True, run_ids=[])
    )
    assert code != 0
    out = capsys.readouterr().out
    assert "METRIC DIRECTION CONFLICT" in out
    assert "lower_is_better: run-a" in out
    assert "higher_is_better: run-b" in out
    assert "#1" not in out


def test_compare_uses_protocol_metric_when_not_given(monkeypatch, capsys):
    import ksvd_research.cli as cli

    manifests = [
        _manifest("run-a", "molhiv-cross-scaffold-interaction", {"valid_roc_auc": 0.80}),
        _manifest("run-b", "molhiv-cross-scaffold-interaction", {"valid_roc_auc": 0.84}),
    ]
    monkeypatch.setattr(cli, "list_runs", lambda study_id=None, status=None, limit=None, **kw: manifests)
    code = _cmd_compare(Namespace(study="molhiv", metric=None, override=False, run_ids=[]))
    assert code == 0
    assert "valid_roc_auc=" in capsys.readouterr().out
