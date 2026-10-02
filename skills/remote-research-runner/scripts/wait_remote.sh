#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

# Wait for a run started with launch_remote.sh. The remote job is already
# detached, so this waiter only needs short, restartable SSH polls: if the
# connection drops, the job keeps running and the waiter retries instead of
# aborting. It returns the remote job's exit code once the .exit file appears.

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run-tag> [timeout-seconds] [poll-seconds]" >&2
  echo "  timeout-seconds: 0 (default) waits forever" >&2
  exit 2
fi

tag="$1"
timeout_s="${2:-0}"
poll_s="${3:-15}"
if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: run-tag must match [A-Za-z0-9._-]+" >&2
  exit 2
fi

run_dir="${RESEARCH_REMOTE_RUN_DIR:-__RRC_DEFAULT__}"
start="$(date +%s)"
consecutive_failures=0

ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)

poll_remote() {
  ssh "${ssh_opts[@]}" "$SSH_ALIAS" bash -s -- "$tag" "$run_dir" <<'REMOTE'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
tag="$1"
rundir="$2"
[[ "$rundir" == "__RRC_DEFAULT__" || -z "$rundir" ]] && rundir="$HOME/.research_runs"
log="$rundir/$tag.log"
exitf="$rundir/$tag.exit"
metaf="$rundir/$tag.meta"
if [[ -f "$metaf" ]]; then grep -E '^(gpu|commit|cmd)=' "$metaf" || true; fi
if [[ -f "$exitf" ]]; then
  printf 'RRC_STATUS=done exit=%s\n' "$(cat "$exitf")"
else
  printf 'RRC_STATUS=running\n'
fi
printf -- '--- tail %s ---\n' "$log"
tail -n 5 "$log" 2>/dev/null || true
REMOTE
}

while :; do
  if out="$(poll_remote 2>&1)"; then
    consecutive_failures=0
    if grep -q '^RRC_STATUS=done' <<<"$out"; then
      exit_code="$(sed -n 's/^RRC_STATUS=done exit=\(.*\)$/\1/p' <<<"$out")"
      printf '%s\n' "$out"
      printf 'run %s finished with exit=%s\n' "$tag" "$exit_code"
      exit "${exit_code:-0}"
    fi
  else
    consecutive_failures=$((consecutive_failures + 1))
    echo "warn: ssh poll failed ($consecutive_failures in a row); remote job keeps running" >&2
    if (( consecutive_failures >= 10 )); then
      echo "error: giving up after $consecutive_failures consecutive SSH failures; job may still be running on the server" >&2
      exit 5
    fi
  fi

  if [[ "$timeout_s" != "0" ]]; then
    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      echo "error: timed out after ${timeout_s}s waiting for '$tag' (still running)" >&2
      exit 124
    fi
  fi
  sleep "$poll_s"
done
