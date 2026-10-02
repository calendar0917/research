"""Small shared helpers (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def short_hex(n: int = 8) -> str:
    return secrets.token_hex(n // 2 + n % 2)[:n]


def make_run_id(experiment: str) -> str:
    return f"{experiment}-{timestamp_tag()}-{short_hex(8)}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def valid_run_id(value: str) -> bool:
    return bool(RUN_ID_RE.match(value))


def b64(text: str) -> str:
    import base64

    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def run_local(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
    input_text: str | None = None,
    capture: bool = True,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a local command. Never uses a shell unless the caller does."""
    return subprocess.run(
        argv,
        cwd=cwd,
        input=input_text,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=check,
    )


def git_repo_root(cwd: str | Path | None = None) -> Path | None:
    try:
        out = run_local(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return Path(out.stdout.strip())


def git_info(root: str | Path) -> dict:
    """Collect commit / branch / dirty state / diff hash for provenance."""
    root = Path(root)

    def _git(*args: str) -> str:
        r = run_local(["git", *args], cwd=root)
        return r.stdout.strip() if r.returncode == 0 else ""

    commit = _git("rev-parse", "HEAD")
    branch = _git("symbolic-ref", "--short", "HEAD") or "DETACHED"
    # Dirty = tracked modifications only.  Untracked files (e.g. result dirs)
    # are normal in the research repo and do not change committed code, but
    # they are still recorded for provenance.
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    diff = _git("diff", "HEAD")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    dirty = bool(tracked.strip())
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "diff_hash": sha256_text(diff)[:16],
        "untracked_count": len([x for x in untracked.splitlines() if x.strip()]),
    }


def write_json_stdout(obj: dict) -> None:
    print(json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False))


def sha256_file(path: str | Path) -> str | None:
    """Hash a local file, returning None when it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Canonical Slurm states as reported by `squeue -o %T` / `sacct -o State`.
# sacct decorates cancelled jobs as e.g. "CANCELLED+", "CANCELLED by 12345".
SLURM_TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def normalize_slurm_state(raw: str | None) -> str:
    """Normalize a raw Slurm state into a canonical uppercase token.

    Handles the decorations sacct adds, e.g. ``CANCELLED+``,
    ``CANCELLED by 12345``, ``CANCELLED by somebody``, ``OUT_OF_MEMORY``,
    ``TIMEOUT``, ``NODE_FAIL``.  Returns ``""`` for empty/unknown input.
    """
    if not raw:
        return ""
    text = raw.strip().strip('"')
    if not text:
        return ""
    # Cut any trailing decoration: "+" suffix or " by <user>" annotation.
    text = text.split("+", 1)[0]
    text = re.split(r"\s+by\s+", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = text.strip().split()[0] if text.strip() else ""
    token = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").upper()
    aliases = {
        "OUTOFMEMORY": "OUT_OF_MEMORY",
        "NODEFAIL": "NODE_FAIL",
        "BOOTFAIL": "BOOT_FAIL",
        "NODE_FAILED": "NODE_FAIL",
    }
    return aliases.get(token.replace("_", ""), aliases.get(token, token))


def expected_cuda_for_backend(torch_backend: str | None) -> str | None:
    """Map a torch backend tag to the CUDA version it should report.

    ``cu118 -> 11.8``, ``cu121 -> 12.1``, ``cu124 -> 12.4``, ``cu128 -> 12.8``.
    Unknown tags return ``None`` so callers skip the check instead of failing
    spuriously.
    """
    if not torch_backend:
        return None
    tag = torch_backend.strip().lower()
    m = re.fullmatch(r"cu(\d+(?:\.\d+)?)", tag)
    if not m:
        return None
    body = m.group(1)
    if "." in body:
        return body
    if len(body) < 3:
        return f"{body}.0"
    # cu118 -> 11.8 ; cu124 -> 12.4 ; cu1210 -> 12.10
    return f"{body[:2]}.{body[2:]}"


def parse_driver(version: str) -> tuple[int, ...]:
    """Parse an NVIDIA driver version into a comparable tuple.

    ``"525.85.12" -> (525, 85, 12)``.  Missing/odd values degrade to a
    best-effort tuple so comparisons never raise.
    """
    parts: list[int] = []
    for chunk in re.split(r"[.\-+ ]", version.strip()):
        m = re.match(r"^(\d+)", chunk)
        parts.append(int(m.group(1)) if m else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def driver_at_least(actual: str | None, minimum: str | None) -> bool:
    if not minimum:
        return True
    if not actual:
        return False
    return parse_driver(actual) >= parse_driver(minimum)


def shell_export_lines(env: dict[str, str]) -> str:
    """Render env vars as safe ``export K=V`` lines."""
    lines = []
    for key, value in env.items():
        if value is None or value == "":
            continue
        lines.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join(lines)


def expand_remote(path: str, *, home: str = "$HOME") -> str:
    """Expand a leading ``~`` to ``$HOME`` for use inside a remote bash script."""
    if path == "~":
        return home
    if path.startswith("~/"):
        return home + path[1:]
    return path


def sleep_retry(fn, attempts: int, delay: float, *, on_error=None):
    """Run ``fn`` up to ``attempts`` times, returning the first truthy result."""
    last = None
    for i in range(attempts):
        try:
            result = fn()
            if result:
                return result
        except Exception as exc:  # noqa: BLE001 - retry wrapper
            last = exc
        if i < attempts - 1:
            time.sleep(delay)
    if last is not None:
        raise last
    return None


def human_age(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600:02d}h"
