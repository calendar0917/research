"""Promoted records must act as a durable run source (fresh-clone semantics).

A fresh clone has an empty/absent ``runs/`` (git-ignored) but the git-tracked
``records/runs/`` sitting in the repo.  All tests here simulate exactly that:
``runs_root`` points into an empty temp directory while ``records_root`` is
either the real repo (real promoted record) or a temp dir with a synthetic
record.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

import ksvd_research.runtime.run_store as store
from ksvd_research.runtime.run_store import (
    find_duplicate_run,
    list_runs,
    load_run,
    run_view_from_record,
)
from ksvd_research.runtime.serialization import write_json_atomic

PROMOTED_RUN_ID = "20260907-123049-e96f14e4"


def _empty_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "runs_root", lambda: tmp_path / "runs")


def test_run_view_from_record_flattens_provenance():
    record = {
        "run_id": "r-id",
        "study_id": "s",
        "protocol_id": "p",
        "metrics": {"valid_mae": 0.1},
        "provenance": {
            "runner": "rn",
            "git": {"commit": "c" * 7, "diff_hash": ""},
            "dataset_fingerprint": {"dataset_fingerprint": "d"},
            "split_fingerprint": "sp",
            "test_access": "granted",
        },
    }
    view = run_view_from_record(record)
    assert view["runner"] == "rn"
    assert view["dataset_fingerprint"] == {"dataset_fingerprint": "d"}
    assert view["split_fingerprint"] == "sp"
    assert view["test_access"] == "granted"
    assert view["metrics"] == {"valid_mae": 0.1}
    assert view["source"] == "promoted" and view["promoted"] is True
    assert view["protocol_hash"] is None


def test_list_runs_includes_promoted_records(tmp_path, monkeypatch):
    _empty_runs(tmp_path, monkeypatch)
    runs = list_runs(source="all")
    run_ids = [run["run_id"] for run in runs]
    assert PROMOTED_RUN_ID in run_ids
    view = next(run for run in runs if run["run_id"] == PROMOTED_RUN_ID)
    assert view["source"] == "promoted"
    assert view["promoted"] is True
    assert view["metrics"].get("valid_mae") is not None
    # promoted-only listing
    promoted_ids = [run["run_id"] for run in list_runs(source="promoted")]
    assert PROMOTED_RUN_ID in promoted_ids
    # local listing is empty in a fresh clone
    assert list_runs(source="local") == []


def test_list_runs_deduplicates_local_and_promoted(tmp_path, monkeypatch):
    _empty_runs(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "2026" / "09" / "07" / PROMOTED_RUN_ID
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / "manifest.json",
        {
            "run_id": PROMOTED_RUN_ID,
            "status": "completed",
            "study_id": "zinc-context-gap",
            "protocol_id": "zinc-context-gap",
            "mode": "terminal",
            "started_at": "2026-09-07T12:30:49",
            "metrics": {"valid_mae": 0.5},
            "source": None,
        },
    )
    runs = list_runs(source="all")
    run_ids = [run["run_id"] for run in runs]
    assert run_ids.count(PROMOTED_RUN_ID) == 1
    row = runs[0]
    assert row["source"] == "local"
    assert row["promoted"] is True
    assert row["metrics"]["valid_mae"] == 0.5
    local = list_runs(source="local")
    assert local[0]["source"] == "local" and local[0]["promoted"] is True


def test_load_run_falls_back_to_promoted_record(tmp_path, monkeypatch):
    _empty_runs(tmp_path, monkeypatch)
    view = load_run(PROMOTED_RUN_ID)
    assert view["run_id"] == PROMOTED_RUN_ID
    assert view["source"] == "promoted"
    assert view["runner"] == "zinc_patch_path_pooling"
    with pytest.raises(FileNotFoundError):
        load_run(PROMOTED_RUN_ID, source="local")
    assert load_run(PROMOTED_RUN_ID, source="promoted")["source"] == "promoted"


def test_show_promoted_record_without_local_run(tmp_path, monkeypatch, capsys):
    from ksvd_research.cli import _cmd_show

    _empty_runs(tmp_path, monkeypatch)
    code = _cmd_show(Namespace(run_id=PROMOTED_RUN_ID, json=False))
    assert code == 0
    out = capsys.readouterr().out
    assert "source:        promoted" in out
    assert "local artifacts: unavailable" in out
    assert "metrics:" in out and "valid_mae" in out


def test_compare_promoted_record_without_local_run(tmp_path, monkeypatch, capsys):
    from ksvd_research.cli import _cmd_compare

    _empty_runs(tmp_path, monkeypatch)
    code = _cmd_compare(
        Namespace(study="zinc-context-gap", metric=None, override=False, run_ids=[], json=False, source="all")
    )
    assert code == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("#")]
    assert lines and lines[0].startswith("#1 ")
    assert PROMOTED_RUN_ID in lines[0]


def test_compare_promoted_source_arg(tmp_path, monkeypatch, capsys):
    from ksvd_research.cli import _cmd_compare

    _empty_runs(tmp_path, monkeypatch)
    code = _cmd_compare(
        Namespace(study="zinc-context-gap", metric=None, override=True, run_ids=[], json=False, source="promoted")
    )
    assert code == 0
    out = capsys.readouterr().out
    assert PROMOTED_RUN_ID in out


def test_duplicate_detection_can_use_promoted_fingerprint(tmp_path, monkeypatch):
    fake_records = tmp_path / "records"
    write_json_atomic(
        fake_records / "runs" / "run-promoted-1.json",
        {
            "kind": "run_record",
            "record_id": "record-run-promoted-1",
            "run_id": "run-promoted-1",
            "study_id": "zinc-context-gap",
            "protocol_id": "zinc-context-gap",
            "status": "completed",
            "fingerprint": "fp-promoted-abc",
            "metrics": {"valid_mae": 0.1},
        },
    )
    monkeypatch.setattr(store, "records_root", lambda: fake_records)
    _empty_runs(tmp_path, monkeypatch)
    runs = list_runs(source="all", status="completed")
    dup = find_duplicate_run(runs, "fp-promoted-abc")
    assert dup is not None
    assert dup["run_id"] == "run-promoted-1"


def test_duplicate_ignores_record_without_fingerprint(tmp_path, monkeypatch):
    fake_records = tmp_path / "records"
    write_json_atomic(
        fake_records / "runs" / "run-old.json",
        {
            "kind": "run_record",
            "record_id": "record-run-old",
            "run_id": "run-old",
            "status": "completed",
        },
    )
    monkeypatch.setattr(store, "records_root", lambda: fake_records)
    _empty_runs(tmp_path, monkeypatch)
    runs = list_runs(source="all", status="completed")
    assert find_duplicate_run(runs, "fp-promoted-abc") is None


def test_unknown_source_raises(tmp_path, monkeypatch):
    import pytest

    _empty_runs(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        list_runs(source="bogus")
    with pytest.raises(ValueError):
        load_run(PROMOTED_RUN_ID, source="bogus")
