"""Tests for the shipped remote runner assets (rr_runner.sh / rr_meta.py)."""

from __future__ import annotations

import json
import subprocess
import sys
import importlib.util
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent.parent / "runner"
RR_RUNNER = RUNNER_DIR / "rr_runner.sh"
RR_META = RUNNER_DIR / "rr_meta.py"


def test_runner_assets_exist():
    assert RR_RUNNER.is_file()
    assert RR_META.is_file()


def test_rr_runner_is_valid_bash():
    cp = subprocess.run(["bash", "-n", str(RR_RUNNER)], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr


def _load_meta_module():
    spec = importlib.util.spec_from_file_location("rr_meta_test", RR_META)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_driver_compare():
    mod = _load_meta_module()
    assert mod._driver_at_least("525.85.12", "525.60.13")
    assert not mod._driver_at_least("510.108.03", "525.60.13")
    assert mod._driver_at_least("550.1", None)


def test_cmd_start_records_provenance_and_fails_on_old_driver(tmp_path, monkeypatch):
    mod = _load_meta_module()
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"state": "submitted", "runtime": {}}))

    monkeypatch.setattr(mod, "probe_driver", lambda: "510.108.03")
    monkeypatch.setattr(mod, "probe_gpus", lambda: {"count": 0, "names": []})
    rc = mod.cmd_start(str(meta_path), probe_torch_flag=False, min_driver="525.60.13")
    assert rc == 78
    data = json.loads(meta_path.read_text())
    assert data["state"] == "failed"
    assert data["runtime"]["driver_version"] == "510.108.03"
    assert data["runtime"]["failure_reason"] == "driver_too_old"


def test_cmd_start_records_provenance_on_compatible_driver(tmp_path, monkeypatch):
    mod = _load_meta_module()
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"state": "submitted", "runtime": {}}))

    monkeypatch.setattr(mod, "probe_driver", lambda: "525.85.12")
    monkeypatch.setattr(mod, "probe_gpus", lambda: {"count": 4, "names": ["A100"]})
    rc = mod.cmd_start(str(meta_path), probe_torch_flag=False, min_driver="525.60.13")
    assert rc == 0
    data = json.loads(meta_path.read_text())
    assert data["state"] == "running"
    assert data["runtime"]["gpu_count"] == 4
    assert data["runtime"]["driver_version"] == "525.85.12"


def test_cmd_finish(tmp_path):
    mod = _load_meta_module()
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"state": "running", "runtime": {}}))
    mod.cmd_finish(str(meta_path), 1)
    data = json.loads(meta_path.read_text())
    assert data["state"] == "failed"
    assert data["runtime"]["exit_code"] == 1


def test_probe_torch_shape():
    mod = _load_meta_module()
    info = mod.probe_torch()
    assert "torch_version" in info
    assert "cuda_available" in info