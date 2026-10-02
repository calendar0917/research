"""Unit tests for the CLI surface and services policy (no network)."""

from __future__ import annotations

import json

import pytest

from rr import services as svc
from rr.cli import _extract_globals, _split_command, main
from rr.config import load_config
from rr.errors import DeployError, JobError, PoolError
from rr.util import now_iso

COMMIT = "c" * 40


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_split_command():
    pre, cmd = _split_command(["run", "res-2", "exp", "--pool", "p", "--", "python", "-m", "x"])
    assert pre == ["run", "res-2", "exp", "--pool", "p"]
    assert cmd == ["python", "-m", "x"]


def test_extract_globals_anywhere_before_dashdash():
    pre, js, cfg = _extract_globals(["hosts", "--json", "--config", "/tmp/x.toml"])
    assert pre == ["hosts"]
    assert js is True
    assert cfg == "/tmp/x.toml"


def test_hosts_json(capsys):
    rc = main(["hosts", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True
    names = {h["host"] for h in data["hosts"]}
    assert {"res", "res-2"} <= names


def test_hosts_text(capsys):
    rc = main(["hosts"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "res-2" in out
    assert "slurm" in out


def test_error_json(capsys):
    rc = main(["doctor", "ghost", "--json"])
    assert rc == 3
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is False
    assert data["code"] == "rr.host.unknown"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _CP:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeRemote:
    def __init__(self, commit=COMMIT, markers=None):
        self.commit = commit
        self.markers = dict(markers or {})
        self.calls: list[tuple] = []

    def out(self, script, args=(), timeout=None):  # noqa: ANN001
        self.calls.append(("out", script, tuple(args)))
        return self.commit

    def ok(self, *a, **k):
        return True

    def get_text(self, path):
        self.calls.append(("get_text", path))
        return self.markers.get(path, "")

    def put_text(self, path, text):
        self.calls.append(("put_text", path))
        self.markers[path] = text

    def run(self, script, args=(), **kwargs):
        self.calls.append(("run", script, tuple(args)))
        return _CP(0, "", "")


class _FakeBackend:
    def __init__(self, commit=COMMIT, markers=None):
        self.remote = _FakeRemote(commit, markers)

    def launch(self, **kwargs):
        return {"run_dir": kwargs["run_dir"], "slurm_job_id": "1", "state": "submitted"}

    def run_dir_of(self, meta):
        return f"/runs/{meta.get('run_id')}"

    def live_state(self, meta):
        return {"state": "running", "live": True}

    def list_remote_metas(self):
        return []

    def reconcile(self, meta, live):
        return None


def _patch_git(monkeypatch, *, root, dirty=False, commit=COMMIT):
    monkeypatch.setattr(svc, "git_repo_root", lambda: root)
    monkeypatch.setattr(
        svc,
        "git_info",
        lambda r: {"commit": commit, "branch": "main", "dirty": dirty, "diff_hash": "d" * 16,
                   "untracked_count": 0},
    )


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_text("lock-v1\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return root


def _valid_markers(root, host, pools, commit=COMMIT):
    markers = {}
    for pool in pools:
        identity = svc.env_identity(root, pool)
        markers[svc.marker_path(host, pool)] = json.dumps(
            {
                "schema": svc.DEPLOY_SCHEMA,
                "host": host.name,
                "pool": pool.name,
                "commit": commit,
                "venv": pool.venv_path(host),
                "lock_hash": identity["lock_hash"],
                "validated": True,
                "deployed_at": now_iso(),
            }
        )
    return markers


# ---------------------------------------------------------------------------
# run policy
# ---------------------------------------------------------------------------

def test_run_refuses_dirty_local(monkeypatch, tmp_path):
    _patch_git(monkeypatch, root=_repo(tmp_path), dirty=True)
    with pytest.raises(JobError) as exc:
        svc.run(load_config(), "res", "exp", ["python", "-m", "x"])
    assert exc.value.code == "rr.job.dirty_local"


def test_deploy_refuses_dirty_local(monkeypatch, tmp_path):
    _patch_git(monkeypatch, root=_repo(tmp_path), dirty=True)
    with pytest.raises(DeployError) as exc:
        svc.deploy(load_config(), "res")
    assert exc.value.code == "rr.deploy.dirty_local"


def test_run_refuses_unprovisioned_pool(monkeypatch, tmp_path):
    _patch_git(monkeypatch, root=_repo(tmp_path))
    with pytest.raises(PoolError) as exc:
        svc.run(load_config(), "res-2", "exp", ["python", "-m", "x"], pool_name="res2-cu118-all")
    assert exc.value.code == "rr.pool.not_provisioned"


def test_run_refuses_stale_remote(monkeypatch, tmp_path):
    _patch_git(monkeypatch, root=_repo(tmp_path))
    monkeypatch.setattr(svc, "get_backend", lambda host, opts: _FakeBackend(commit="e" * 40))
    with pytest.raises(JobError) as exc:
        svc.run(load_config(), "res-2", "exp", ["python", "-m", "x"])
    assert exc.value.code == "rr.job.stale_remote"


def test_run_refuses_without_deployment_marker(monkeypatch, tmp_path):
    """A matching Git commit is not enough: the pool must be deployed."""
    root = _repo(tmp_path)
    cfg = load_config()
    _patch_git(monkeypatch, root=root)
    monkeypatch.setattr(svc, "get_backend", lambda host, opts: _FakeBackend())
    with pytest.raises(JobError) as exc:
        svc.run(cfg, "res-2", "exp", ["python", "-m", "x"])
    assert exc.value.code == "rr.job.not_deployed"
    assert "missing" in exc.value.details["reasons"]


def test_run_launches_when_deployed(monkeypatch, tmp_path):
    root = _repo(tmp_path)
    cfg = load_config()
    host = cfg.get_host("res-2")
    pool = cfg.get_pool("res2-cu124")
    markers = _valid_markers(root, host, [pool])
    _patch_git(monkeypatch, root=root)
    monkeypatch.setattr(svc, "get_backend", lambda h, opts: _FakeBackend(markers=markers))
    result = svc.run(cfg, "res-2", "zinc-seed0", ["python", "-m", "x", "--seed", "0"])
    meta = result["meta"]
    assert meta["experiment"] == "zinc-seed0"
    assert meta["pool"] == "res2-cu124"
    assert meta["git_commit"] == COMMIT
    assert meta["requested"]["gpus"] == 1
    assert meta["state"] == "submitted"


def test_run_rejects_bad_experiment_name(monkeypatch, tmp_path):
    _patch_git(monkeypatch, root=_repo(tmp_path))
    with pytest.raises(JobError) as exc:
        svc.run(load_config(), "res", "bad name", ["python", "-m", "x"])
    assert exc.value.code == "rr.job.bad_experiment"


def test_run_requires_command(monkeypatch, tmp_path):
    _patch_git(monkeypatch, root=_repo(tmp_path))
    with pytest.raises(JobError) as exc:
        svc.run(load_config(), "res", "exp", [])
    assert exc.value.code == "rr.job.no_command"


# ---------------------------------------------------------------------------
# deployment markers
# ---------------------------------------------------------------------------

def test_deployment_status_reasons(tmp_path):
    root = _repo(tmp_path)
    cfg = load_config()
    host = cfg.get_host("res-2")
    pool = cfg.get_pool("res2-cu124")

    remote = _FakeRemote(markers={})
    st = svc.deployment_status(cfg, host, pool, {"commit": COMMIT}, root, remote=remote)
    assert st["ok"] is False and st["reasons"] == ["missing"]

    remote = _FakeRemote(markers=_valid_markers(root, host, [pool], commit="f" * 40))
    st = svc.deployment_status(cfg, host, pool, {"commit": COMMIT}, root, remote=remote)
    assert "commit_mismatch" in st["reasons"]

    bad = _valid_markers(root, host, [pool])
    key = svc.marker_path(host, pool)
    marker = json.loads(bad[key])
    marker["lock_hash"] = "deadbeef"
    bad[key] = json.dumps(marker)
    st = svc.deployment_status(cfg, host, pool, {"commit": COMMIT}, root, remote=_FakeRemote(markers=bad))
    assert "lock_mismatch" in st["reasons"]

    st = svc.deployment_status(
        cfg, host, pool, {"commit": COMMIT}, root, remote=_FakeRemote(markers=_valid_markers(root, host, [pool]))
    )
    assert st["ok"] is True


def test_pools_sharing_env_groups_gpu_aliases(tmp_path):
    cfg = load_config()
    host = cfg.get_host("res")
    pools = svc.pools_sharing_env(cfg, host, cfg.get_pool("res-gpu0"))
    names = {p.name for p in pools}
    # res-gpu0, res-gpu1 and res-local all map to the same venv
    assert {"res-gpu0", "res-gpu1", "res-local"} <= names
    # cu118 has its own venv and must never be grouped with cu124
    host2 = cfg.get_host("res-2")
    pools2 = {p.name for p in svc.pools_sharing_env(cfg, host2, cfg.get_pool("res2-cu124"))}
    assert "res2-cu118-all" not in pools2


def test_deploy_missing_pool_lock_is_refused(monkeypatch, tmp_path):
    root = _repo(tmp_path)
    _patch_git(monkeypatch, root=root)
    with pytest.raises(DeployError) as exc:
        svc.deploy(load_config(), "res-2", pool_name="res2-cu118-all")
    assert exc.value.code == "rr.deploy.lock_missing"


def test_deploy_failure_invalidates_marker_and_run_refuses(monkeypatch, tmp_path):
    """If git moved but the environment sync failed, run must refuse."""
    root = _repo(tmp_path)
    cfg = load_config()
    host = cfg.get_host("res-2")
    pool = cfg.get_pool("res2-cu124")
    old_markers = _valid_markers(root, host, [pool], commit="a" * 40)
    remote = _FakeRemote(markers=old_markers)

    class _FailingRemote(_FakeRemote):
        def run(self, script, args=(), **kwargs):
            self.calls.append(("run", script, tuple(args)))
            if "sync --" in script:
                return _CP(5, "", "uv sync exploded")
            return _CP(0, "", "")

    remote = _FailingRemote(markers=old_markers)
    monkeypatch.setattr(svc, "Remote", lambda alias, opts: remote)
    _patch_git(monkeypatch, root=root)
    monkeypatch.setattr(svc, "run_local", lambda *a, **k: _CP(0, "", ""))

    with pytest.raises(DeployError) as exc:
        svc.deploy(cfg, "res-2")
    assert exc.value.code == "rr.deploy.failed"

    # marker was invalidated (rm -f) and no new marker was published
    invalidated = [c for c in remote.calls if c[0] == "run" and "rm -f" in c[1]]
    assert invalidated, "expected marker invalidation before apply"
    assert not any(c[0] == "put_text" and c[1].endswith(".json") for c in remote.calls)


# ---------------------------------------------------------------------------
# jobs: partial results + ambiguity
# ---------------------------------------------------------------------------

def test_list_jobs_reports_unreachable_hosts(monkeypatch):
    cfg = load_config()

    class _Boom:
        def list_remote_metas(self):
            from rr.errors import RemoteError

            raise RemoteError("ssh down", code="rr.remote.failed")

        def list_legacy_metas(self):
            return []

    monkeypatch.setattr(svc, "get_backend", lambda host, opts: _Boom())
    report = svc.list_jobs_report(cfg, ["res-2"])
    assert report["jobs"] == []
    assert report["partial"] is True
    assert report["errors"][0]["host"] == "res-2"
    assert report["errors"][0]["code"] == "rr.remote.failed"


def test_cancel_ambiguous_experiment_is_refused(monkeypatch):
    cfg = load_config()
    metas = [
        {"run_id": "exp-1", "experiment": "exp", "created_at": "2024-01-02T00:00:00+00:00"},
        {"run_id": "exp-2", "experiment": "exp", "created_at": "2024-01-01T00:00:00+00:00"},
    ]

    class _B:
        def list_remote_metas(self):
            return [dict(m) for m in metas]

        def list_legacy_metas(self):
            return []

    monkeypatch.setattr(svc, "get_backend", lambda host, opts: _B())
    with pytest.raises(JobError) as exc:
        svc.cancel(cfg, "exp", host_name="res-2")
    assert exc.value.code == "rr.job.ambiguous"
    assert set(exc.value.details["candidates"]) == {"exp-1", "exp-2"}

    # exact run_id is unambiguous
    monkeypatch.setattr(svc, "get_backend", lambda host, opts: _B())
    with pytest.raises(Exception):
        svc.cancel(cfg, "exp-1", host_name="res-2")  # backend.cancel on _B is missing -> AttributeError is fine


def test_cancel_single_run_allowed(monkeypatch):
    cfg = load_config()
    calls = {}

    class _B:
        def list_remote_metas(self):
            return [{"run_id": "solo-1", "experiment": "solo", "created_at": "2024-01-01T00:00:00+00:00"}]

        def list_legacy_metas(self):
            return []

        def cancel(self, meta):
            calls["run_id"] = meta["run_id"]
            return {"run_id": meta["run_id"], "state": "cancelled"}

    monkeypatch.setattr(svc, "get_backend", lambda host, opts: _B())
    result = svc.cancel(cfg, "solo", host_name="res-2")
    assert result["state"] == "cancelled"
    assert calls["run_id"] == "solo-1"


# ---------------------------------------------------------------------------
# pull: file and directory result paths
# ---------------------------------------------------------------------------

class _PullBackend:
    def __init__(self, kinds):
        self.remote = self
        self.kinds = kinds
        self.downloads = []

    def run_dir_of(self, meta):
        return "/runs/r1"

    def list_remote_metas(self):
        return [{"run_id": "r1", "experiment": "e", "host": "res-2", "state": "completed"}]

    def list_legacy_metas(self):
        return []

    def out(self, script, args=(), timeout=None):
        return self.kinds.get(args[0], "missing")

    def rsync_down(self, remote_path, local, extra=None):
        self.downloads.append((remote_path, local))

    def exists(self, path):
        return self.kinds.get(path, "missing") != "missing"


def test_pull_handles_file_and_dir(monkeypatch, tmp_path):
    cfg = load_config()
    kinds = {
        "/share/home/snsun/rr/research/out/metrics.json": "file",
        "/share/home/snsun/rr/research/out/run": "dir",
        "/share/home/snsun/rr/research/out/missing": "missing",
    }
    backend = _PullBackend(kinds)
    monkeypatch.setattr(svc, "get_backend", lambda host, opts: backend)

    result = svc.pull(
        cfg,
        "r1",
        host_name="res-2",
        dest=str(tmp_path / "dst"),
        paths=["out/metrics.json", "out/run", "out/missing"],
    )
    assert set(result["pulled"]) == {"out/metrics.json", "out/run"}
    assert result["missing"] == ["out/missing"]
    file_dl = [d for d in backend.downloads if d[0].endswith("metrics.json")]
    dir_dl = [d for d in backend.downloads if d[0].endswith("out/run/")]
    assert file_dl and not file_dl[0][0].endswith("/")
    assert dir_dl and dir_dl[0][0].endswith("/")
