"""Unit tests for rr config loading / host / pool / regime resolution."""

from __future__ import annotations

import textwrap

import pytest

from rr.config import Host, Pool, _deep_merge, load_config
from rr.errors import ConfigError, HostError, PoolError


def test_default_config_has_hosts_and_pools():
    cfg = load_config()
    assert "res" in cfg.hosts
    assert "res-2" in cfg.hosts
    assert cfg.hosts["res"].backend == "process"
    assert cfg.hosts["res-2"].backend == "slurm"
    assert "res2-cu124" in cfg.pools
    assert "res2-cu118-all" in cfg.pools


def test_cu124_pool_is_bootstrap_constrained():
    cfg = load_config()
    pool = cfg.get_pool("res2-cu124")
    assert pool.eligible_nodes == ["c05", "c06"]
    assert pool.min_driver == "525.60.13"
    assert pool.torch_backend == "cu124"
    assert pool.provisioned is True


def test_cpu_pool_exists_and_requests_no_gpu():
    cfg = load_config()
    pool = cfg.get_pool("res2-cpu")
    assert pool.gpus == 0
    assert pool.eligible_nodes == []


def test_cu118_pool_is_architecturally_ready_but_not_provisioned():
    cfg = load_config()
    pool = cfg.get_pool("res2-cu118-all")
    assert pool.eligible_nodes == [f"c0{i}" for i in range(1, 9)]
    assert pool.provisioned is False
    # Per-pool environment strategy: a separate committed lockfile, so cu124
    # can never be silently reused as cu118.
    assert pool.lock_file == "uv.cu118.lock"
    assert pool.sync_mode == "frozen"
    # future driver upgrade must be a config-only change
    assert pool.venv_path(cfg.get_host("res-2")).endswith(".venv-cu118")


def test_res_gpu_pools_pin_single_devices():
    cfg = load_config()
    g0 = cfg.get_pool("res-gpu0")
    g1 = cfg.get_pool("res-gpu1")
    assert g0.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert g1.env["CUDA_VISIBLE_DEVICES"] == "1"
    assert g0.gpus == 1 and g1.gpus == 1
    # res-local stays the unconstrained default
    assert cfg.hosts["res"].default_pool == "res-local"
    assert "CUDA_VISIBLE_DEVICES" not in cfg.get_pool("res-local").env


def test_default_pools_have_no_lock_override():
    cfg = load_config()
    assert cfg.get_pool("res2-cu124").lock_file is None
    assert cfg.get_pool("res2-cu124").sync_mode == "frozen"


def test_host_marker_dir_defaults_under_tool_dir():
    cfg = load_config()
    host = cfg.get_host("res-2")
    assert host.marker_dir().endswith("/.rr/deployments")


def test_pool_host_mismatch_raises():
    cfg = load_config()
    host = cfg.get_host("res")
    with pytest.raises(PoolError):
        cfg.resolve_pool(host, "res2-cu124")


def test_resolve_pool_default():
    cfg = load_config()
    assert cfg.resolve_pool(cfg.get_host("res"), None).name == "res-local"
    assert cfg.resolve_pool(cfg.get_host("res-2"), None).name == "res2-cu124"


def test_unknown_host_and_pool():
    cfg = load_config()
    with pytest.raises(HostError):
        cfg.get_host("nope")
    with pytest.raises(PoolError):
        cfg.get_pool("nope")


def test_user_override_deep_merge(tmp_path):
    override = tmp_path / "rr.toml"
    override.write_text(
        textwrap.dedent(
            """
            [hosts.res-2]
            runs = "/share/home/snsun/other-runs"

            [pools.res2-cu124]
            eligible_nodes = ["c05", "c06"]
            provisioned = false
            """
        )
    )
    cfg = load_config(str(override))
    assert cfg.hosts["res-2"].runs == "/share/home/snsun/other-runs"
    # unspecified fields survive the merge
    assert cfg.hosts["res-2"].code == "/share/home/snsun/rr/research"
    assert cfg.pools["res2-cu124"].provisioned is False
    assert cfg.pools["res2-cu124"].min_driver == "525.60.13"


def test_adding_a_new_host_is_config_only(tmp_path):
    override = tmp_path / "rr.toml"
    override.write_text(
        textwrap.dedent(
            """
            [hosts.res-3]
            ssh_alias = "res-3"
            backend = "slurm"
            code = "/share/home/snsun/rr3/research"
            runs = "/share/home/snsun/rr3/runs"

            [pools.res3-cu124]
            host = "res-3"
            backend = "slurm"
            eligible_nodes = ["g01", "g02"]
            min_driver = "525.60.13"
            """
        )
    )
    cfg = load_config(str(override))
    assert cfg.get_host("res-3").backend == "slurm"
    assert cfg.get_pool("res3-cu124").host == "res-3"


def test_bad_host_missing_ssh_alias(tmp_path):
    override = tmp_path / "rr.toml"
    override.write_text("[hosts.bad]\ncode = '/x'\n")
    with pytest.raises(ConfigError):
        load_config(str(override))


def test_pool_referencing_unknown_host(tmp_path):
    override = tmp_path / "rr.toml"
    override.write_text("[pools.orphan]\nhost = 'ghost'\n")
    with pytest.raises(ConfigError):
        load_config(str(override))


def test_deep_merge():
    assert _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}