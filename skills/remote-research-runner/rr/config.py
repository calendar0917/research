"""Configuration loading: bundled defaults + user override (TOML, stdlib).

Only the read path is needed; rr never rewrites config.  Adding a host, a pool,
or a new execution regime is therefore a config-only change, not a code change
and never a Skill change.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError, HostError, PoolError

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = SKILL_DIR / "rr" / "default_config.toml"
USER_CONFIG = Path.home() / ".config" / "rr" / "rr.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}", path=str(path)) from exc


@dataclass
class Host:
    name: str
    ssh_alias: str
    backend: str
    code: str
    runs: str
    default_pool: str | None = None
    deploy_method: str = "bundle"
    github: str | None = None
    tool_dir: str = "$HOME/.rr"
    legacy_runs: str | None = None
    deployments_dir: str | None = None
    env: dict = field(default_factory=dict)
    slurm: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def is_slurm(self) -> bool:
        return self.backend == "slurm"

    def marker_dir(self) -> str:
        """Directory holding per-pool deployment markers (host-local)."""
        if self.deployments_dir:
            return self.deployments_dir
        return self.tool_dir.rstrip("/") + "/deployments"


@dataclass
class Pool:
    name: str
    host: str
    backend: str
    gpus: int = 1
    eligible_nodes: list[str] = field(default_factory=list)
    min_driver: str | None = None
    torch_backend: str = "cu124"
    provisioned: bool = True
    venv: str | None = None
    # Per-pool environment strategy: which lockfile defines this environment
    # and how `uv sync` must treat it.  ``lock_file`` is relative to the repo
    # root (default: the project's uv.lock).  A pool that needs a *different*
    # dependency set (e.g. cu118) must declare its own committed lockfile so
    # the environment identity is explicit and reproducible.
    lock_file: str | None = None
    sync_mode: str = "frozen"
    env: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def venv_path(self, host: Host) -> str:
        if self.venv:
            return self.venv
        if self.torch_backend and self.torch_backend != "cu124":
            return f"{host.code.rstrip('/')}/.venv-{self.torch_backend}"
        return f"{host.code.rstrip('/')}/.venv"


@dataclass
class Config:
    ssh_opts: list[str]
    hosts: dict[str, Host]
    pools: dict[str, Pool]
    source: str = str(DEFAULT_CONFIG)

    # -- lookup -----------------------------------------------------------
    def get_host(self, name: str) -> Host:
        try:
            return self.hosts[name]
        except KeyError:
            raise HostError(
                f"unknown host '{name}'",
                code="rr.host.unknown",
                known=sorted(self.hosts),
            ) from None

    def get_pool(self, name: str) -> Pool:
        try:
            return self.pools[name]
        except KeyError:
            raise PoolError(
                f"unknown pool '{name}'",
                code="rr.pool.unknown",
                known=sorted(self.pools),
            ) from None

    def pools_for_host(self, host: str) -> list[Pool]:
        return [p for p in self.pools.values() if p.host == host]

    def resolve_pool(self, host: Host, pool_name: str | None) -> Pool | None:
        """Pick the pool for a run.

        Explicit ``--pool`` wins; otherwise the host's ``default_pool``; a host
        with no pool at all (plain ssh/process) returns ``None``.
        """
        if pool_name:
            pool = self.get_pool(pool_name)
            if pool.host != host.name:
                raise PoolError(
                    f"pool '{pool.name}' belongs to host '{pool.host}', not '{host.name}'",
                    code="rr.pool.host_mismatch",
                    pool=pool.name,
                    pool_host=pool.host,
                    host=host.name,
                )
            return pool
        if host.default_pool:
            return self.get_pool(host.default_pool)
        candidates = self.pools_for_host(host.name)
        if len(candidates) == 1:
            return candidates[0]
        return None


def load_config(path: str | Path | None = None) -> Config:
    """Load bundled defaults merged with the user override.

    ``RR_CONFIG`` (or an explicit path) replaces the user override entirely.
    """
    base = _read_toml(DEFAULT_CONFIG)
    override_path = Path(path) if path else Path(os.environ.get("RR_CONFIG", USER_CONFIG))
    override = _read_toml(override_path) if override_path.exists() else {}
    merged = _deep_merge(base, override)

    ssh_opts = list(merged.get("ssh", {}).get("opts", []))
    if not ssh_opts:
        ssh_opts = [
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
        ]

    hosts: dict[str, Host] = {}
    for name, raw in merged.get("hosts", {}).items():
        if "ssh_alias" not in raw:
            raise ConfigError(f"host '{name}' is missing ssh_alias", host=name)
        if "code" not in raw:
            raise ConfigError(f"host '{name}' is missing code (remote checkout path)", host=name)
        hosts[name] = Host(
            name=name,
            ssh_alias=raw["ssh_alias"],
            backend=raw.get("backend", "process"),
            code=raw["code"],
            runs=raw.get("runs", f"{raw['code'].rstrip('/')}/../rr-runs"),
            default_pool=raw.get("default_pool"),
            deploy_method=raw.get("deploy_method", "bundle"),
            github=raw.get("github"),
            tool_dir=raw.get("tool_dir", "$HOME/.rr"),
            legacy_runs=raw.get("legacy_runs"),
            deployments_dir=raw.get("deployments_dir"),
            env=raw.get("env", {}),
            slurm=raw.get("slurm", {}),
            raw=raw,
        )

    pools: dict[str, Pool] = {}
    for name, raw in merged.get("pools", {}).items():
        host_name = raw.get("host")
        if host_name not in hosts:
            raise ConfigError(
                f"pool '{name}' references unknown host '{host_name}'",
                pool=name,
                host=host_name,
            )
        backend = raw.get("backend", hosts[host_name].backend)
        pools[name] = Pool(
            name=name,
            host=host_name,
            backend=backend,
            gpus=int(raw.get("gpus", 1)),
            eligible_nodes=list(raw.get("eligible_nodes", [])),
            min_driver=raw.get("min_driver"),
            torch_backend=raw.get("torch_backend", "cu124"),
            provisioned=bool(raw.get("provisioned", True)),
            venv=raw.get("venv"),
            lock_file=raw.get("lock_file"),
            sync_mode=raw.get("sync_mode", "frozen"),
            env=raw.get("env", {}),
            resources=raw.get("resources", {}),
            raw=raw,
        )

    return Config(ssh_opts=ssh_opts, hosts=hosts, pools=pools, source=str(override_path))