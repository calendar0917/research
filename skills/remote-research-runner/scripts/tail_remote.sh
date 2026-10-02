#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

# Inspect one launched run: metadata, liveness, and the tail of its log.
# Safe to call repeatedly and across SSH drops.

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run-tag> [tail-lines]" >&2
  exit 2
fi
tag="$1"
lines="${2:-20}"
if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: run-tag must match [A-Za-z0-9._-]+" >&2
  exit 2
fi

run_dir="${RESEARCH_REMOTE_RUN_DIR:-__RRC_DEFAULT__}"
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  "$SSH_ALIAS" bash -s -- "$tag" "$run_dir" "$lines" <<'REMOTE'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
tag="$1"
rundir="$2"
[[ "$rundir" == "__RRC_DEFAULT__" || -z "$rundir" ]] && rundir="$HOME/.research_runs"
lines="$3"
log="$rundir/$tag.log"
exitf="$rundir/$tag.exit"
pidf="$rundir/$tag.pid"
metaf="$rundir/$tag.meta"

if [[ -f "$metaf" ]]; then cat "$metaf"; else echo "no metadata for tag '$tag' ($metaf)"; fi
if [[ -f "$exitf" ]]; then
  printf 'status=done exit=%s\n' "$(cat "$exitf")"
elif [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
  printf 'status=running pid=%s\n' "$(cat "$pidf")"
else
  printf 'status=unknown (no exit file and pid not alive)\n'
fi
printf -- '--- tail %s %s ---\n' "$lines" "$log"
tail -n "$lines" "$log" 2>/dev/null || true
REMOTE
