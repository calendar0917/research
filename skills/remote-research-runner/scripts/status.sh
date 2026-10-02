#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

ROOT="$(local_repo_root)"
cd "$ROOT"
local_branch="$(git symbolic-ref --short HEAD 2>/dev/null || echo DETACHED)"
local_commit="$(git rev-parse HEAD)"
local_dirty="clean"
[[ -n "$(git status --porcelain)" ]] && local_dirty="dirty"

printf 'local  branch=%s commit=%s state=%s\n' "$local_branch" "$local_commit" "$local_dirty"
run_dir="${RESEARCH_REMOTE_RUN_DIR:-__RRC_DEFAULT__}"
ssh "$SSH_ALIAS" bash -s -- "$REMOTE_REPO" "$run_dir" <<'REMOTE'
set -uo pipefail
repo="$1"
rundir="$2"
[[ "$rundir" == "__RRC_DEFAULT__" || -z "$rundir" ]] && rundir="$HOME/.research_runs"
cd "$repo"
branch="$(git symbolic-ref --short HEAD 2>/dev/null || echo DETACHED)"
commit="$(git rev-parse HEAD)"
state="clean"
[[ -n "$(git status --porcelain)" ]] && state="dirty"
printf 'remote branch=%s commit=%s state=%s\n' "$branch" "$commit" "$state"

printf '\nremote runs (%s):\n' "$rundir"
if [[ -d "$rundir" ]]; then
  shopt -s nullglob
  metas=("$rundir"/*.meta)
  if (( ${#metas[@]} == 0 )); then
    echo '  (none)'
  fi
  for meta in "${metas[@]}"; do
    tag="$(basename "$meta" .meta)"
    exitf="$rundir/$tag.exit"
    pidf="$rundir/$tag.pid"
    if [[ -f "$exitf" ]]; then
      status="done exit=$(cat "$exitf")"
    elif [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      status="running pid=$(cat "$pidf")"
    else
      status="unknown"
    fi
    gpu="$(sed -n 's/^gpu=//p' "$meta" | head -1)"
    printf '  %-24s gpu=%s %s\n' "$tag" "$gpu" "$status"
  done
else
  echo '  (no run dir yet)'
fi
REMOTE
