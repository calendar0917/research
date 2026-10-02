"""Unit tests for backend dispatch and command/script generation."""

from __future__ import annotations

import json

from rr.backends import ProcessBackend, SlurmBackend, get_backend
from rr.config import load_config


class _CP:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class DummyRemote:
    alias = "dummy"
    ssh_opts: list[str] = []

    def __init__(self):
        self.calls = []

    def home(self):
        return "/home/dummy"

    def expand(self, value):
        if isinstance(value, str) and "$HOME" in value:
            return value.replace("$HOME", self.home())
        return value

    def out(self, script, args=(), timeout=None):  # noqa: ANN001
        self.calls.append((script, tuple(args)))
        return ""

    def ok(self, script, args=(), timeout=None):  # noqa: ANN001
        return True

    def get_text(self, path):
        return ""

    def put_text(self, path, text):
        self.calls.append(("put_text", path))

    def run(self, script, args=(), **kwargs):
        self.calls.append(("run", script, tuple(args)))
        return _CP()


class StateRemote(DummyRemote):
    """Remote that scripts the `_query` output and captures meta writes."""

    def __init__(self, query="", meta_text=""):
        super().__init__()
        self.query = query
        self.meta_text = meta_text
        self.puts: dict[str, str] = {}

    def out(self, script, args=(), timeout=None):
        return self.query

    def get_text(self, path):
        return self.meta_text

    def put_text(self, path, text):
        self.puts[path] = text


def _cfg():
    return load_config()


def test_get_backend_dispatch():
    cfg = _cfg()
    assert isinstance(get_backend(cfg.get_host("res"), []), ProcessBackend)
    assert isinstance(get_backend(cfg.get_host("res-2"), []), SlurmBackend)


def test_unknown_backend_raises(tmp_path):
    cfg = _cfg()
    host = cfg.get_host("res")
    host.backend = "quantum"
    import pytest

    from rr.errors import ConfigError

    with pytest.raises(ConfigError):
        get_backend(host, [])


def test_slurm_sbatch_script_structure():
    cfg = _cfg()
    host = cfg.get_host("res-2")
    pool = cfg.get_pool("res2-cu124")
    backend = SlurmBackend(host, DummyRemote())
    meta = {"experiment": "zinc-seed0", "requested": {"gpus": 2, "cpus": 8, "mem": "48G", "time": "12:00:00"}}
    script = backend._sbatch_script(meta, "/share/runs/run-1", "export FOO=bar", pool)
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=48G" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --nodelist=c05,c06" in script
    assert "--output=/share/runs/run-1/slurm-%j.out" in script
    assert "export FOO=bar" in script
    assert 'exec bash "$RR_RUNNER"' in script
    assert "pam_slurm" not in script


def test_slurm_sbatch_script_uses_pool_nodes_not_hardcoded():
    cfg = _cfg()
    host = cfg.get_host("res-2")
    backend = SlurmBackend(host, DummyRemote())
    meta = {"experiment": "x", "requested": {"eligible_nodes": ["c01", "c02", "c03"]}}
    script = backend._sbatch_script(meta, "/r", "")
    assert "#SBATCH --nodelist=c01,c02,c03" in script


def test_slurm_sbatch_script_no_nodelist_when_pool_unconstrained():
    cfg = _cfg()
    host = cfg.get_host("res-2")
    backend = SlurmBackend(host, DummyRemote())
    meta = {"experiment": "x", "requested": {}}
    script = backend._sbatch_script(meta, "/r", "")
    assert "--nodelist" not in script


def test_slurm_omits_gres_when_zero_gpus():
    cfg = _cfg()
    host = cfg.get_host("res-2")
    backend = SlurmBackend(host, DummyRemote())
    meta = {"experiment": "x", "requested": {"gpus": 0}}
    script = backend._sbatch_script(meta, "/r", "")
    assert "--gres" not in script
    assert "#SBATCH --cpus-per-task" in script


def test_process_remote_env_export_offline():
    cfg = _cfg()
    host = cfg.get_host("res-2")
    pool = cfg.get_pool("res2-cu124")
    backend = ProcessBackend(host, DummyRemote())
    block = backend.remote_env_export(pool)
    assert "UV_OFFLINE=1" in block
    assert "UV_CACHE_DIR=" in block
    assert "UV_PYTHON_INSTALL_DIR=" in block
    assert "PATH=" in block


def test_process_remote_env_export_online_res():
    cfg = _cfg()
    host = cfg.get_host("res")
    pool = cfg.get_pool("res-local")
    backend = ProcessBackend(host, DummyRemote())
    block = backend.remote_env_export(pool)
    assert "UV_OFFLINE" not in block
    assert ".local/bin" in block


def test_res_gpu_pools_pin_cuda_visible_devices():
    """P0: process-backend GPU isolation via explicit pools."""
    cfg = _cfg()
    host = cfg.get_host("res")
    b0 = ProcessBackend(host, DummyRemote())
    b1 = ProcessBackend(host, DummyRemote())
    env0 = b0.remote_env_export(cfg.get_pool("res-gpu0"))
    env1 = b1.remote_env_export(cfg.get_pool("res-gpu1"))
    assert 'export CUDA_VISIBLE_DEVICES=0' in env0
    assert 'export CUDA_VISIBLE_DEVICES=1' in env1
    # res-local is the unconstrained host-level pool: it must NOT pin a device
    assert "CUDA_VISIBLE_DEVICES" not in b0.remote_env_export(cfg.get_pool("res-local"))
    # each pool requests exactly one GPU
    assert cfg.get_pool("res-gpu0").gpus == 1
    assert cfg.get_pool("res-gpu1").gpus == 1


def test_venv_path_selection():
    cfg = _cfg()
    host = cfg.get_host("res-2")
    assert cfg.get_pool("res2-cu124").venv_path(host).endswith("research/.venv")
    assert cfg.get_pool("res2-cu118-all").venv_path(host).endswith("research/.venv-cu118")


def test_run_prefix_default():
    cfg = _cfg()
    backend = ProcessBackend(cfg.get_host("res"), DummyRemote())
    assert backend.run_prefix() == "uv run --no-sync"


# ---------------------------------------------------------------------------
# Slurm state normalization + reconciliation
# ---------------------------------------------------------------------------

def _slurm(query, meta):
    cfg = _cfg()
    host = cfg.get_host("res-2")
    remote = StateRemote(query=query, meta_text=json.dumps(meta))
    return SlurmBackend(host, remote), remote


def test_slurm_live_state_normalizes_cancelled_plus():
    meta = {"run_id": "r1", "state": "running", "slurm_job_id": "123", "runtime": {}}
    backend, _ = _slurm("ACCT|CANCELLED+|0:0|00:01:00|c05", meta)
    live = backend.live_state(meta)
    assert live["scheduler_state"] == "CANCELLED"
    assert live["state"] == "cancelled"
    assert live["live"] is False


def test_slurm_live_state_normalizes_cancelled_by_uid():
    meta = {"run_id": "r1", "state": "running", "slurm_job_id": "123", "runtime": {}}
    backend, _ = _slurm("ACCT|CANCELLED by 12345|0:0|00:01:00|c05", meta)
    live = backend.live_state(meta)
    assert live["scheduler_state"] == "CANCELLED"


def test_slurm_reconcile_cancelled_does_not_fake_exit_zero():
    meta = {"run_id": "r1", "state": "running", "slurm_job_id": "1", "runtime": {"node": "c05"}}
    backend, remote = _slurm("ACCT|CANCELLED by 12345|0:0|00:02:00|c05", meta)
    live = backend.live_state(meta)
    updated = backend.reconcile(meta, live)
    assert updated["state"] == "cancelled"
    assert updated["runtime"]["exit_code"] is None
    assert updated["runtime"]["failure_reason"] == "cancelled"
    assert updated["runtime"]["scheduler_state"] == "CANCELLED"
    # atomically written (temp file then rename)
    written = [p for p in remote.puts if p.endswith("meta.json.tmp")]
    assert written
    payload = json.loads(remote.puts[written[0]])
    assert payload["state"] == "cancelled"


def test_slurm_reconcile_terminal_states():
    cases = {
        "OUT_OF_MEMORY": ("oom", "137:0"),
        "TIMEOUT": ("timeout", "0:15"),
        "NODE_FAIL": ("failed", "0:0"),
        "PREEMPTED": ("failed", "0:0"),
        "FAILED": ("failed", "1:0"),
        "COMPLETED": ("completed", "0:0"),
    }
    for scheduler, (expected_state, code) in cases.items():
        meta = {"run_id": "r1", "state": "running", "slurm_job_id": "1", "runtime": {}}
        backend, _ = _slurm(f"ACCT|{scheduler}|{code}|00:02:00|", meta)
        live = backend.live_state(meta)
        updated = backend.reconcile(meta, live)
        assert updated is not None, scheduler
        assert updated["state"] == expected_state, scheduler


def test_slurm_reconcile_ignores_non_terminal():
    meta = {"run_id": "r1", "state": "running", "slurm_job_id": "1", "runtime": {}}
    backend, remote = _slurm("QUEUE|RUNNING|c05|00:01:00|c05", meta)
    live = backend.live_state(meta)
    assert backend.reconcile(meta, live) is None
    assert not remote.puts


def test_slurm_reconcile_ignores_already_terminal():
    meta = {"run_id": "r1", "state": "completed", "slurm_job_id": "1", "runtime": {"exit_code": 0}}
    backend, remote = _slurm("ACCT|CANCELLED+|0:0|00:01:00|c05", meta)
    assert backend.reconcile(meta, {"scheduler_state": "CANCELLED", "live": False}) is None
    assert not remote.puts


def test_slurm_cancel_does_not_fake_exit_zero():
    meta = {"run_id": "r1", "state": "running", "slurm_job_id": "1", "runtime": {}}

    class _CancelRemote(StateRemote):
        def run(self, script, args=(), **kwargs):
            return _CP(0, "cancelled 1\n", "")

    cfg = _cfg()
    backend = SlurmBackend(cfg.get_host("res-2"), _CancelRemote(meta_text=json.dumps(meta)))
    result = backend.cancel(meta)
    assert result["state"] == "cancelled"
    written = [p for p in backend.remote.puts if p.endswith("meta.json")][0]
    payload = json.loads(backend.remote.puts[written])
    assert payload["runtime"].get("exit_code") is None
    assert payload["runtime"]["failure_reason"] == "cancelled"


# ---------------------------------------------------------------------------
# process reconciliation
# ---------------------------------------------------------------------------

def test_process_reconcile_marks_lost_when_not_live():
    cfg = _cfg()
    meta = {"run_id": "r1", "state": "running", "runtime": {}}
    remote = StateRemote(meta_text=json.dumps(meta))
    backend = ProcessBackend(cfg.get_host("res"), remote)
    updated = backend.reconcile(meta, {"state": "running", "live": False, "pid": "999"})
    assert updated["state"] == "lost"
    assert updated["runtime"]["failure_reason"] == "process_died_without_completion"
    assert any(p.endswith("meta.json.tmp") for p in remote.puts)


def test_process_reconcile_leaves_running_alone():
    cfg = _cfg()
    meta = {"run_id": "r1", "state": "running", "runtime": {}}
    remote = StateRemote(meta_text=json.dumps(meta))
    backend = ProcessBackend(cfg.get_host("res"), remote)
    assert backend.reconcile(meta, {"state": "running", "live": True, "pid": "1"}) is None
    assert not remote.puts
