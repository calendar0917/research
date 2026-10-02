#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu:0|1> <uv-run command...>" >&2
  echo "example: $0 0 python -m tracks.ksvd.experiments.foo --seed 0 --device cuda" >&2
  exit 2
fi

gpu="$1"
shift
if [[ "$gpu" != "0" && "$gpu" != "1" ]]; then
  echo "error: gpu must be 0 or 1" >&2
  exit 2
fi

# Two portability traps when driving commands over ssh:
#   1. the remote login shell may be /bin/sh (dash), which lacks `pipefail`, so we
#      run the payload under bash explicitly;
#   2. `ssh host bash -s -- a b "multi word"` re-splits the multi-word argument,
#      so the shell-quoted command is base64-encoded into a single space-free token.
cmd_q="$(quote_args "$@")"
cmd_b64="$(printf '%s' "$cmd_q" | base64 | tr -d '\n')"

ssh "$SSH_ALIAS" bash -s -- "$REMOTE_REPO" "$gpu" "$cmd_b64" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
repo="$1"
gpu="$2"
cmd="$(printf '%s' "$3" | base64 -d)"
cd "$repo"
printf 'remote_commit=%s\n' "$(git rev-parse HEAD)"
printf 'gpu_physical=%s\n' "$gpu"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
eval "uv run $cmd"
REMOTE
