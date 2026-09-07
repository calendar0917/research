"""Runtime control plane for KSVD research runs.

This package is deliberately dependency-light (standard library plus
``pyyaml``) so it can be used from tests and tooling without pulling in
torch/networkx.  Research components import *from* here; runtime never
imports runner or experiment modules.
"""

from .config import (
    apply_override,
    config_hash,
    deep_merge,
    parse_override_spec,
    scientific_config,
)
from .environment import capture_environment
from .git_state import GitState, capture_git_state
from .paths import (
    REPO_ROOT,
    TRACK_ROOT,
    records_root,
    repo_root,
    resolve_path,
    runs_root,
    track_root,
)
from .serialization import (
    jsonable,
    load_json,
    load_yaml,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_yaml_atomic,
)
from .run_store import (
    create_run_directory,
    find_duplicate_run,
    load_run,
    list_runs,
    new_run_id,
    promote_run,
    run_fingerprint,
)

__all__ = [
    "REPO_ROOT",
    "TRACK_ROOT",
    "GitState",
    "apply_override",
    "capture_environment",
    "capture_git_state",
    "config_hash",
    "create_run_directory",
    "deep_merge",
    "find_duplicate_run",
    "jsonable",
    "list_runs",
    "load_json",
    "load_run",
    "load_yaml",
    "new_run_id",
    "parse_override_spec",
    "paths",
    "promote_run",
    "records_root",
    "repo_root",
    "resolve_path",
    "run_fingerprint",
    "runs_root",
    "scientific_config",
    "sha256_file",
    "sha256_text",
    "track_root",
    "write_json_atomic",
    "write_yaml_atomic",
]
