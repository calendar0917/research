#!/usr/bin/env bash
set -euo pipefail

SSH_ALIAS="${RESEARCH_SSH_ALIAS:-res}"
REMOTE_REPO="${RESEARCH_REMOTE_REPO:-/home/hxy/cy/research}"

# `uv` is typically installed in ~/.local/bin, which is not on PATH in
# non-interactive SSH sessions (plain `ssh host 'cmd'`). Prefix remote commands
# that need `uv` with this export so `uv` resolves without a login shell.
REMOTE_ENV='export PATH="$HOME/.local/bin:$PATH"'

local_repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || {
    echo "error: run from inside the local research Git repository" >&2
    exit 2
  }
}

quote_args() {
  local out="" q
  for arg in "$@"; do
    printf -v q '%q' "$arg"
    out+="${q} "
  done
  printf '%s' "$out"
}
