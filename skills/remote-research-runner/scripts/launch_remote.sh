#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

# Launch a long remote run in its own session so an SSH drop cannot kill it.
# The job is fully detached (setsid + nohup + stdin from /dev/null), writes a
# log/exit/pid/meta triple under the remote run dir (outside the repo, so it can
# never dirty the remote Git worktree), and keeps running after the launching
# SSH connection closes. Use wait_remote.sh to follow it to completion.

force=0
if [[ "${1:-}" == "--force" ]]; then
  force=1
  shift
fi

if [[ $# -lt 3 ]]; then
  echo "usage: $0 [--force] <gpu:0|1> <run-tag> <uv-run command...>" >&2
  echo "example: $0 0 zinc-A 'python -m tracks.ksvd.experiments.foo train_queue --device cuda'" >&2
  exit 2
fi

gpu="$1"
shift
tag="$1"
shift
if [[ "$gpu" != "0" && "$gpu" != "1" ]]; then
  echo "error: gpu must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: run-tag must match [A-Za-z0-9._-]+ (got '$tag')" >&2
  exit 2
fi

cmd_q="$(quote_args "$@")"
cmd_b64="$(printf '%s' "$cmd_q" | base64 | tr -d '\n')"
# Never pass an empty argument: ssh joins remote args with spaces, so an empty
# arg would be dropped and shift every following positional. Use a sentinel.
run_dir="${RESEARCH_REMOTE_RUN_DIR:-__RRC_DEFAULT__}"

ssh "$SSH_ALIAS" bash -s -- "$REMOTE_REPO" "$gpu" "$tag" "$cmd_b64" "$run_dir" "$force" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
repo="$1"
gpu="$2"
tag="$3"
cmd_b64="$4"
rundir="$5"
force="$6"
[[ "$rundir" == "__RRC_DEFAULT__" || -z "$rundir" ]] && rundir="$HOME/.research_runs"

cd "$repo"
commit="$(git rev-parse HEAD)"

mkdir -p "$rundir"
log="$rundir/$tag.log"
exitf="$rundir/$tag.exit"
pidf="$rundir/$tag.pid"
metaf="$rundir/$tag.meta"

# Idempotency / double-launch guard: refuse to start a tag whose process is
# still alive and has not written an exit code.
if [[ "$force" != "1" && -f "$pidf" && ! -f "$exitf" ]]; then
  oldpid="$(cat "$pidf" 2>/dev/null || true)"
  if [[ -n "$oldpid" ]] && kill -0 "$oldpid" 2>/dev/null; then
    echo "error: run tag '$tag' is already running (pid $oldpid, log $log); use --force to override" >&2
    exit 3
  fi
fi

cmd="$(printf '%s' "$cmd_b64" | base64 -d)"
rm -f "$exitf" "$pidf"

printf 'tag=%s\ngpu=%s\ncommit=%s\nstarted=%s\ncmd=%s\n' \
  "$tag" "$gpu" "$commit" "$(date -Is)" "$cmd" > "$metaf"

# The child is a new session leader (setsid), immune to SIGHUP and to the
# controlling terminal going away, with stdin detached. It records its own pid
# and writes the real exit code to $exitf, which is the authoritative completion
# signal (pid liveness is only a convenience / guard).
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export RRC_CMD="$cmd"
export RRC_LOG="$log"
export RRC_EXIT="$exitf"
export RRC_PID="$pidf"

setsid bash -c 'echo $$ > "$RRC_PID"; eval "uv run $RRC_CMD" >"$RRC_LOG" 2>&1; echo $? > "$RRC_EXIT"' \
  </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true

# Wait briefly for the session leader to publish its pid.
for _ in $(seq 1 50); do
  [[ -s "$pidf" ]] && break
  sleep 0.1
done

if [[ ! -s "$pidf" ]]; then
  echo "error: run '$tag' did not start; see $log" >&2
  tail -n 20 "$log" 2>/dev/null >&2 || true
  exit 4
fi

pid="$(cat "$pidf")"
if ! kill -0 "$pid" 2>/dev/null; then
  # The process is gone; give the shell a moment to flush the exit file.
  for _ in $(seq 1 20); do
    [[ -f "$exitf" ]] && break
    sleep 0.1
  done
  echo "error: run '$tag' exited immediately (pid $pid); see $log" >&2
  tail -n 20 "$log" 2>/dev/null >&2 || true
  if [[ -f "$exitf" ]]; then
    real="$(cat "$exitf")"
    echo "exit=$real" >&2
    exit "$real"
  fi
  exit 4
fi

printf 'launched tag=%s pid=%s gpu=%s commit=%s\nlog=%s\n' "$tag" "$pid" "$gpu" "$commit" "$log"
REMOTE
