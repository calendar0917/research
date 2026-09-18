"""Fast, CPU-only tests for the research-integrity validator.

These never import torch, never touch data and never run an experiment; they
are safe for CI and for the local fast subset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ksvd_research.runtime import integrity


def _run_repo_checks():
    return integrity.integrity_checks()


def test_repo_integrity_passes():
    checks = _run_repo_checks()
    errors = [c for c in checks if not c.ok and c.severity == "error"]
    assert not errors, [c.detail for c in errors]
    assert integrity.check_exit_code(checks) == 0


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_tree(tmp_path: Path, *, records):
    """Create a minimal STATE/records tree; returns the records root."""
    records_root = tmp_path / "records"
    _write(
        records_root / "claims" / "claim-keep-20260101.yaml",
        "claim_id: claim-keep-20260101\ncreated: '2026-01-01T00:00:00'\nstatus: supported\n",
    )
    for name, body in records.items():
        kind = "claims" if name.startswith("claim-") else "decisions"
        _write(records_root / kind / f"{name}.yaml", body)
    _write(
        tmp_path / "STATE.yaml",
        "\n".join(
            [
                "phase: test",
                "active_studies: [zinc-context-gap]",
                "focus_study: zinc-context-gap",
                "open_questions: ['still open?']",
                "guardrails: ['keep']",
            ]
        ),
    )
    return records_root


@pytest.fixture
def patched(tmp_path, monkeypatch):
    records_root = _fake_tree(tmp_path, records={})
    monkeypatch.setattr(integrity, "records_root", lambda: records_root)
    monkeypatch.setattr(integrity, "state_file", lambda: tmp_path / "STATE.yaml")
    monkeypatch.setattr(integrity, "track_root", lambda: tmp_path)
    return tmp_path


def test_undocumented_future_date_is_error(patched, monkeypatch):
    from datetime import date

    _write(
        patched / "records" / "claims" / "claim-future-20991231.yaml",
        "claim_id: claim-future-20991231\ncreated: '2099-12-31T00:00:00'\nstatus: supported\n",
    )
    checks = integrity.integrity_checks(today=date(2026, 9, 18))
    future = [c for c in checks if c.name == "records.no_undocumented_future_dates"][0]
    assert not future.ok
    assert "claim-future-20991231" in future.detail
    assert integrity.check_exit_code(checks) == 1


def test_documented_future_date_is_not_an_error(patched, monkeypatch):
    from datetime import date

    _write(
        patched / "records" / "claims" / "claim-future-20260920.yaml",
        "claim_id: claim-future-20260920\ncreated: '2026-09-20T00:00:00'\nstatus: supported\n",
    )
    _write(
        patched / "records" / "PROVENANCE_DATES.md",
        "```yaml\ndocumented_future_dated_records:\n  - record: claim-future-20260920\n```\n",
    )
    checks = integrity.integrity_checks(today=date(2026, 9, 18))
    future = [c for c in checks if c.name == "records.no_undocumented_future_dates"][0]
    assert future.ok


def test_dangling_reference_is_error(patched, monkeypatch):
    _write(
        patched / "STATE.yaml",
        "\n".join(
            [
                "phase: test",
                "active_studies: [zinc-context-gap]",
                "focus_study: zinc-context-gap",
                "open_questions: ['see decision-does-not-exist-20260101']",
                "guardrails: ['keep']",
            ]
        ),
    )
    checks = integrity.integrity_checks()
    refs = [c for c in checks if c.name == "records.references_resolve"][0]
    assert not refs.ok
    assert "decision-does-not-exist-20260101" in refs.detail


def test_id_filename_mismatch_is_error(patched, monkeypatch):
    _write(
        patched / "records" / "claims" / "claim-abc-20260101.yaml",
        "claim_id: claim-wrong-20260101\ncreated: '2026-01-01T00:00:00'\nstatus: supported\n",
    )
    checks = integrity.integrity_checks()
    ids = [c for c in checks if c.name == "records.id_matches_filename"][0]
    assert not ids.ok


def test_cli_verify_json(monkeypatch, capsys):
    import argparse

    from ksvd_research import cli

    rc = cli._cmd_verify(argparse.Namespace(json=True))
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    assert rc == 0
