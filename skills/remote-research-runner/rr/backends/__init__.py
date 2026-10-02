"""Backend registry."""

from __future__ import annotations

from ..config import Host
from ..errors import ConfigError
from ..remote import Remote
from .base import Backend
from .process import ProcessBackend
from .slurm import SlurmBackend

_BACKENDS = {
    "process": ProcessBackend,
    "ssh": ProcessBackend,
    "slurm": SlurmBackend,
}


def get_backend(host: Host, ssh_opts: list[str]) -> Backend:
    try:
        cls = _BACKENDS[host.backend]
    except KeyError:
        raise ConfigError(
            f"host '{host.name}' has unknown backend '{host.backend}'",
            host=host.name,
            backend=host.backend,
            known=sorted(_BACKENDS),
        ) from None
    return cls(host, Remote(host.ssh_alias, ssh_opts))


__all__ = ["Backend", "ProcessBackend", "SlurmBackend", "get_backend"]