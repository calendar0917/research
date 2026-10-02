"""Tests for `rr doctor` node probing policy and `logs --follow` termination."""

from __future__ import annotations

import json
import time

from rr import services as svc
from rr.cli import follow_should_stop
from rr.config import load_config


class _CP:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ProbeRemote:
    def __init__(self, submit="", status="", collect=""):
        self.submit = submit
        self.status = status
        self.collect = collect
        self.runs: list[tuple[str, tuple]] = []

    def expand(self, value):
        return value

    def run(self, script, args=(), **kwargs):
        self.runs.append((script, tuple(args)))
        if "RRPROBEJOB" in script:
            return _CP(0, self.submit, "")
        if "RRPROBEALIVE" in script:
            return _CP(0, self.status, "")
        if "RRPROBEDRV" in script:
            return _CP(0, self.collect, "")
        return _CP(0, "", "")

    def scancel_calls(self):
        return [c for c in self.runs if 'scancel "$j"' in c[0] and "RRPROBE" not in c[0]]


def _cfg_host():
    cfg = load_config()
    return cfg, cfg.get_host("res-2")


# ---------------------------------------------------------------------------
# default doctor must be non-invasive
# ---------------------------------------------------------------------------

def test_probe_default_never_constructs_remote(monkeypatch, tmp_path):
    cfg, host = _cfg_host()

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("default doctor must not touch the cluster")

    monkeypatch.setattr(svc, "Remote", Boom)
    monkeypatch.setattr(svc, "node_probe_cache_file", lambda h: tmp_path / "missing.json")
    assert svc.probe_node_drivers(cfg, host, ["c05"], refresh=False) == {}


def test_probe_default_uses_fresh_cache(monkeypatch, tmp_path):
    cfg, host = _cfg_host()
    cache = tmp_path / "nodes-res-2.json"
    cache.write_text(json.dumps({"at": time.time(), "drivers": {"c05": "525.85.12"}}))

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("must not touch the cluster")

    monkeypatch.setattr(svc, "Remote", Boom)
    monkeypatch.setattr(svc, "node_probe_cache_file", lambda h: cache)
    assert svc.probe_node_drivers(cfg, host, ["c05"], refresh=False) == {"c05": "525.85.12"}


def test_probe_default_ignores_stale_cache(monkeypatch, tmp_path):
    cfg, host = _cfg_host()
    cache = tmp_path / "nodes-res-2.json"
    cache.write_text(json.dumps({"at": time.time() - 10 * 3600, "drivers": {"c05": "525.85.12"}}))
    monkeypatch.setattr(svc, "node_probe_cache_file", lambda h: cache)
    assert svc.probe_node_drivers(cfg, host, ["c05"], refresh=False) == {}


# ---------------------------------------------------------------------------
# --refresh submits, tracks and cleans up only its own jobs
# ---------------------------------------------------------------------------

def test_probe_refresh_records_job_ids_and_caches(monkeypatch, tmp_path):
    cfg, host = _cfg_host()
    remote = ProbeRemote(
        submit="RRPROBEJOB|111\nRRPROBEJOB|112\n",
        status="",  # not alive anymore
        collect="RRPROBEDRV|c05|Kernel Module  525.85.12\nRRPROBEDRV|c06|Kernel Module  525.85.12\n",
    )
    monkeypatch.setattr(svc, "Remote", lambda alias, opts: remote)
    cache = tmp_path / "nodes-res-2.json"
    monkeypatch.setattr(svc, "node_probe_cache_file", lambda h: cache)

    drivers = svc.probe_node_drivers(cfg, host, ["c05", "c06"], refresh=True, poll_interval=0)
    assert drivers == {"c05": "525.85.12", "c06": "525.85.12"}
    data = json.loads(cache.read_text())
    assert data["job_ids"] == ["111", "112"]

    submit = [c for c in remote.runs if "RRPROBEJOB" in c[0]][0]
    assert submit[1][2] and submit[1][0].endswith(submit[1][2])
    # every submitted id is scancelled afterwards and the scratch dir is removed
    assert [c[1] for c in remote.scancel_calls()] == [("111", "112")]
    assert any("rm -rf" in s for s, _ in remote.runs)


def test_probe_refresh_tracks_only_its_own_jobs(monkeypatch, tmp_path):
    cfg, host = _cfg_host()
    remote = ProbeRemote(
        submit="RRPROBEJOB|501\n",
        status="RRPROBEALIVE|501|PENDING\n",  # stays alive -> loop reaches deadline
        collect="RRPROBEDRV|c05|Kernel Module  525.85.12\n",
    )
    monkeypatch.setattr(svc, "Remote", lambda alias, opts: remote)
    monkeypatch.setattr(svc, "node_probe_cache_file", lambda h: tmp_path / "n.json")

    svc.probe_node_drivers(cfg, host, ["c05"], refresh=True, max_wait=0, poll_interval=0)
    # status polling only ever targets the id we submitted
    status_calls = [c for c in remote.runs if "RRPROBEALIVE" in c[0]]
    if status_calls:
        assert all(c[1] == ("501",) for c in status_calls)
    assert [c[1] for c in remote.scancel_calls()] == [("501",)]


def test_probe_refresh_interrupt_still_scancels(monkeypatch, tmp_path):
    cfg, host = _cfg_host()

    class InterruptRemote(ProbeRemote):
        def run(self, script, args=(), **kwargs):
            self.runs.append((script, tuple(args)))
            if "RRPROBEJOB" in script:
                return _CP(0, "RRPROBEJOB|777\n", "")
            if "RRPROBEALIVE" in script:
                raise KeyboardInterrupt()
            return _CP(0, "", "")

    remote = InterruptRemote()
    monkeypatch.setattr(svc, "Remote", lambda alias, opts: remote)
    monkeypatch.setattr(svc, "node_probe_cache_file", lambda h: tmp_path / "n.json")

    import pytest

    with pytest.raises(KeyboardInterrupt):
        svc.probe_node_drivers(cfg, host, ["c05"], refresh=True, poll_interval=0)
    assert [c[1] for c in remote.scancel_calls()] == [("777",)]


def test_probe_submit_script_is_concurrency_safe():
    _, host = _cfg_host()
    script = svc.node_probe_submit_script(host)
    assert "rrnode-$token" in script
    assert "RRPROBEJOB" in script
    # guarded trap: cancel only if the submit phase did not finish
    assert "trap cleanup EXIT HUP INT TERM" in script
    assert 'scancel "$j"' in script
    # must NOT select jobs by name (that would collide with concurrent doctors)
    assert "grep -c" not in script
    assert svc.node_probe_token() != svc.node_probe_token()
    # short time limit bounds any orphan even after a hard kill
    assert "-t 3" in script


def test_probe_status_script_targets_given_ids():
    _, host = _cfg_host()
    script = svc.node_probe_status_script(host)
    assert 'squeue -h -j "$j"' in script
    assert "RRPROBEALIVE" in script


def test_parse_node_probe_output():
    text = "RRPROBEJOB|9\nRRPROBEDRV|c07|Kernel Module  510.108.03\nRRPROBEDRV|c08|\n"
    drivers, jobs = svc.parse_node_probe_output(text)
    assert jobs == ["9"]
    assert drivers == {"c07": "510.108.03", "c08": ""}


# ---------------------------------------------------------------------------
# follow termination
# ---------------------------------------------------------------------------

def test_follow_stops_on_every_terminal_state():
    for state in ["completed", "failed", "cancelled", "timeout", "oom", "lost"]:
        assert follow_should_stop({"state": state, "live": False}, "running", 0), state


def test_follow_continues_while_live():
    assert not follow_should_stop({"state": "running", "live": True}, "running", 100)


def test_follow_continues_while_pending():
    assert not follow_should_stop({"state": "pending", "live": False}, "submitted", 100)


def test_follow_is_bounded_when_not_live():
    assert not follow_should_stop({"state": "submitted", "live": False}, "submitted", 3, max_idle=15)
    assert follow_should_stop({"state": "submitted", "live": False}, "submitted", 15, max_idle=15)
