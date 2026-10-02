"""Typed errors so the CLI can print actionable, machine-parseable failures."""

from __future__ import annotations


class RRException(Exception):
    """Base error. ``code`` is a stable identifier for agents/MCP callers."""

    code = "rr.error"

    def __init__(self, message: str, *, code: str | None = None, **details):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.details = details

    def as_dict(self) -> dict:
        out = {"ok": False, "code": self.code, "message": self.message}
        if self.details:
            out["details"] = self.details
        return out


class ConfigError(RRException):
    code = "rr.config"


class HostError(RRException):
    code = "rr.host"


class PoolError(RRException):
    code = "rr.pool"


class DeployError(RRException):
    code = "rr.deploy"


class RemoteError(RRException):
    code = "rr.remote"


class JobError(RRException):
    code = "rr.job"
