"""SSH / rsync transport helpers.

SSH is transport, not business logic: rr only ever refers to a host by its ssh
alias.  IPs, ports and ProxyJump stay in ``~/.ssh/config``.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass

from .errors import RemoteError


@dataclass
class Remote:
    alias: str
    ssh_opts: list[str]

    _home: str | None = None

    # -- path expansion ---------------------------------------------------
    def home(self) -> str:
        """Resolve the remote ``$HOME`` once (needed for rsync, which does not
        run a shell to expand variables)."""
        if self._home is None:
            self._home = self.run('printf %s "$HOME"').stdout.strip()
        return self._home

    def expand(self, value: str) -> str:
        if isinstance(value, str) and "$HOME" in value:
            return value.replace("$HOME", self.home())
        return value

    # -- ssh --------------------------------------------------------------
    def _ssh_argv(self, extra: list[str] | None = None) -> list[str]:
        return ["ssh", *self.ssh_opts, self.alias, *(extra or [])]

    def run(
        self,
        script: str,
        args: tuple | list = (),
        *,
        timeout: float | None = None,
        check: bool = True,
        capture: bool = True,
        stdin_text: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a bash script on the remote host (script passed on stdin).

        Positional args are forwarded to the script as ``$1``, ``$2`` ... so we
        never have to quote a command into an ssh argument string.  A leading
        ``$HOME`` in an argument is expanded locally to the remote home.
        """
        args = [self.expand(str(a)) for a in args]
        # Build a single quoted command so empty arguments survive (ssh joins
        # remote words with spaces and would otherwise drop them).
        remote_cmd = "bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
        argv = self._ssh_argv([remote_cmd])
        payload = stdin_text if stdin_text is not None else script
        try:
            cp = subprocess.run(
                argv,
                input=payload,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(
                f"ssh to '{self.alias}' timed out after {timeout}s",
                code="rr.remote.timeout",
                host=self.alias,
            ) from exc
        except FileNotFoundError as exc:
            raise RemoteError("ssh not found on PATH", code="rr.remote.no_ssh") from exc
        if check and cp.returncode != 0:
            raise RemoteError(
                f"remote command failed on '{self.alias}' (exit {cp.returncode})",
                code="rr.remote.failed",
                host=self.alias,
                exit_code=cp.returncode,
                stderr=(cp.stderr or "").strip()[-2000:],
                stdout=(cp.stdout or "").strip()[-2000:],
            )
        return cp

    def out(self, script: str, args: tuple | list = (), *, timeout: float | None = None) -> str:
        return self.run(script, args, timeout=timeout, check=True).stdout

    def ok(self, script: str, args: tuple | list = (), *, timeout: float | None = None) -> bool:
        try:
            cp = self.run(script, args, timeout=timeout, check=False)
        except RemoteError:
            return False
        return cp.returncode == 0

    # -- files ------------------------------------------------------------
    def put_text(self, path: str, text: str) -> None:
        # Create the parent dir and stream the payload to the file.
        path = self.expand(path)
        script = 'd="$(dirname "$1")"; mkdir -p "$d"; cat > "$1"'
        argv = self._ssh_argv(["bash -c", shlex.quote(script), "_", shlex.quote(path)])
        cp = subprocess.run(argv, input=text, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RemoteError(
                f"failed to write '{path}' on '{self.alias}'",
                code="rr.remote.put_failed",
                host=self.alias,
                stderr=cp.stderr.strip()[-1000:],
            )

    def get_text(self, path: str) -> str:
        return self.out('cat "$1" 2>/dev/null || true', [path])

    def exists(self, path: str) -> bool:
        return self.ok('test -e "$1"', [path])

    # -- rsync ------------------------------------------------------------
    def _rsync_ssh(self) -> str:
        return "ssh " + " ".join(shlex.quote(o) for o in self.ssh_opts)

    def rsync_up(self, local: str, remote_path: str, *, extra: list[str] | None = None) -> None:
        target = f"{self.alias}:{self.expand(remote_path)}"
        argv = ["rsync", "-a", "-e", self._rsync_ssh(), *(extra or []), local, target]
        cp = subprocess.run(argv, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RemoteError(
                f"rsync upload to '{remote_path}' failed",
                code="rr.remote.rsync_failed",
                host=self.alias,
                stderr=cp.stderr.strip()[-2000:],
            )

    def rsync_down(self, remote_path: str, local: str, *, extra: list[str] | None = None) -> None:
        source = f"{self.alias}:{self.expand(remote_path)}"
        argv = ["rsync", "-a", "-e", self._rsync_ssh(), *(extra or []), source, local]
        cp = subprocess.run(argv, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RemoteError(
                f"rsync download from '{remote_path}' failed",
                code="rr.remote.rsync_failed",
                host=self.alias,
                stderr=cp.stderr.strip()[-2000:],
            )