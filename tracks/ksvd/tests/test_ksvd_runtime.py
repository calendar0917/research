"""Unit tests for the runtime control-plane primitives (Phase 1)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from ksvd_research.runtime import (
    apply_override,
    capture_git_state,
    config_hash,
    create_run_directory,
    find_duplicate_run,
    jsonable,
    load_json,
    new_run_id,
    run_fingerprint,
    write_json_atomic,
    write_yaml_atomic,
)
from ksvd_research.runtime.config import coerce_scalar, parse_override_spec, scientific_config
from ksvd_research.runtime.manifest import (
    MANIFEST_VERSION,
    RunSpec,
    build_manifest,
    write_manifest,
)
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, run_directory
from ksvd_research.runtime.run_store import (
    promote_run,
    record_path_for_run,
)


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_jsonable_handles_paths_bytes_numpy():
    import numpy as np

    payload = {
        "path": Path("a/b"),
        "raw": b"\x00\xff",
        "vector": np.asarray([1.0, 2.5]),
        "scalar": np.float32(0.5),
        "mapping": {"nested": [1, "two"]},
    }
    clean = jsonable(payload)
    assert clean["path"] == "a/b"
    assert clean["raw"]["cty"] == "bytes_hex"
    assert clean["vector"] == [1.0, 2.5]
    assert clean["scalar"] == 0.5
    json.dumps(clean)  # must be round-trippable


def test_atomic_json_roundtrip(tmp_path):
    target = tmp_path / "nested" / "file.json"
    write_json_atomic(target, {"a": 1, "b": [True, None]})
    assert load_json(target) == {"a": 1, "b": [True, None]}
    assert not list(target.parent.glob("*.tmp"))


def test_atomic_yaml_roundtrip(tmp_path):
    target = tmp_path / "file.yaml"
    write_yaml_atomic(target, {"study": "zinc-context-gap", "n": 2})
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["n"] == 2


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_repo_and_track_root():
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert TRACK_ROOT.name == "ksvd"
    assert (TRACK_ROOT / "STATE.yaml").is_file()


def test_run_directory_layout():
    created = datetime(2026, 9, 7, 10, 30, 0)
    target = run_directory("run-1", created)
    assert target.name == "run-1"
    assert target.parent.name == "07"
    assert target.parents[1].name == "09"
    assert target.parents[2].name == "2026"


def test_create_run_directory(tmp_path, monkeypatch):
    import ksvd_research.runtime.paths as paths

    monkeypatch.setattr(paths, "runs_root", lambda: tmp_path / "runs")
    run_id = "unit-test-run-1"
    root = create_run_directory(run_id, datetime(2026, 1, 2, 3, 4, 5))
    assert (root / "artifacts").is_dir()
    assert root.parents[2].name == "2026"
    with pytest.raises(OSError):
        create_run_directory(run_id, datetime(2026, 1, 2, 3, 4, 5))


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _spec(run_dir: Path) -> RunSpec:
    return RunSpec(
        run_id="unit-run-1",
        study_id="zinc-context-gap",
        candidate_id="cand-a",
        protocol_id="zinc-context-gap",
        mode="scratch",
        purpose="unit test",
        runner="unit",
        config_hash="hash-a",
        dataset_fingerprint={"dataset_fingerprint": "dset-1", "split": {}},
        split_fingerprint="split-1",
        seeds=[0],
        test_access="unguarded",
        fingerprint="fp-1",
        run_dir=run_dir,
        git={"commit": "c0ffee", "dirty": False, "diff_hash": "", "patch": "", "untracked": []},
        environment={"python": "3.12"},
    )


def test_manifest_roundtrip(tmp_path, monkeypatch):
    import ksvd_research.runtime.paths as paths

    monkeypatch.setattr(paths, "runs_root", lambda: tmp_path / "runs")
    run_dir = create_run_directory("unit-run-1", datetime(2026, 1, 2, 3, 4, 5))
    started = datetime(2026, 1, 2, 3, 4, 5)
    manifest = build_manifest(
        spec=_spec(run_dir),
        started_at=started,
        status="running",
        metrics={"valid_mae": 0.5},
        config={"protocol_id": "x"},
    )
    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["run_id"] == "unit-run-1"
    write_manifest(run_dir, manifest)
    loaded = load_json(run_dir / "manifest.json")
    assert loaded["metrics"] == {"valid_mae": 0.5}
    assert loaded["git"]["commit"] == "c0ffee"
    assert loaded["fingerprint"] == "fp-1"


# ---------------------------------------------------------------------------
# config override + scientific hash
# ---------------------------------------------------------------------------


def test_coerce_and_parse():
    assert coerce_scalar("3") == 3
    assert coerce_scalar("3.5") == 3.5
    assert coerce_scalar("true") is True
    assert coerce_scalar("none") is None
    assert coerce_scalar("cpu") == "cpu"
    key, value = parse_override_spec("model.epochs=2")
    assert (key, value) == ("model.epochs", 2)


def test_apply_override_nested():
    config = {"model": {"epochs": 50, "dropout": 0.05}, "seed": 0}
    out = apply_override(apply_override(config, "model.epochs", 2), "seed", 3)
    assert out["model"]["epochs"] == 2 and out["seed"] == 3
    assert config["model"]["epochs"] == 50  # original untouched


def test_scientific_config_excludes_plumbing():
    config = {"protocol_id": "p", "model": {"x": 1}, "output": {"json": "x.json"}, "runtime": {}}
    sci = scientific_config(config)
    assert "output" not in sci and "runtime" not in sci
    assert config_hash(config) == config_hash(sci)


def test_scientific_config_hash_changes_with_science():
    config = {"model": {"epochs": 50}}
    assert config_hash(config) != config_hash({"model": {"epochs": 51}})


# ---------------------------------------------------------------------------
# run store: ids, fingerprints, duplicates, promotion
# ---------------------------------------------------------------------------


def test_run_id_unique():
    assert new_run_id() != new_run_id()


def test_run_fingerprint_sensitive_to_code_state():
    base = dict(runner="r", study_id="s", protocol_id="p", candidate_id="c", config={"a": 1}, seeds=[0])
    f1 = run_fingerprint(**base, git={"commit": "x", "diff_hash": ""}, dataset_fingerprint={"dataset_fingerprint": "d"}, split_fingerprint="sp")
    f2 = run_fingerprint(**base, git={"commit": "x", "diff_hash": "dirty-patch"}, dataset_fingerprint={"dataset_fingerprint": "d"}, split_fingerprint="sp")
    assert f1 != f2
    f3 = run_fingerprint(**base, git={"commit": "x", "diff_hash": ""}, dataset_fingerprint={"dataset_fingerprint": "d"}, split_fingerprint="sp")
    assert f1 == f3


def test_find_duplicate_run():
    runs = [
        {"run_id": "r1", "status": "completed", "fingerprint": "fp-1", "metrics": {"valid_mae": 0.1}},
        {"run_id": "r2", "status": "running", "fingerprint": "fp-1"},
    ]
    assert find_duplicate_run(runs, "fp-none") is None
    duplicate = find_duplicate_run(runs, "fp-1")
    assert duplicate["run_id"] == "r1"


def test_promote_creates_track_record_and_is_immutable(tmp_path, monkeypatch):
    import ksvd_research.runtime.run_store as store

    monkeypatch.setattr(store, "records_root", lambda: tmp_path / "records")
    manifest = build_manifest(
        spec=_spec(tmp_path / "run"),
        started_at=datetime(2026, 1, 2, 3, 4, 5),
        status="completed",
        metrics={"valid_mae": 0.12},
        config={"model": {"epochs": 2}},
    )
    config = {"model": {"epochs": 2}, "output": {"json": "ignored"}}
    record = promote_run(manifest, config=config, run_dir=tmp_path / "run")
    path = record_path_for_run("unit-run-1")
    assert path.is_file()
    assert record["run_id"] == "unit-run-1"
    assert "output" not in record["scientific_config"]
    promote_run(manifest, config=config, run_dir=tmp_path / "run")  # idempotent
    different = {**manifest, "config_hash": "changed"}
    with pytest.raises(RuntimeError):
        promote_run(different, config=config, run_dir=tmp_path / "run")


# ---------------------------------------------------------------------------
# git state
# ---------------------------------------------------------------------------


def test_git_state_captures_commit():
    state = capture_git_state()
    assert state.commit
    assert state.diff_hash == "" if not state.dirty else state.diff_hash


def test_git_dirty_detection():
    probe = Path(REPO_ROOT) / f"tmp-ksvd-probe-{new_run_id()}.py"
    try:
        state = capture_git_state()
        dirty_before = state.dirty
        probe.write_text("probe = 1\n", encoding="utf-8")
        state_after = capture_git_state()
        assert state_after.dirty
        assert any(entry.path == probe.name for entry in state_after.untracked)
        assert state_after.code_state_hash
    finally:
        probe.unlink(missing_ok=True)
        assert not dirty_before or True
