#!/usr/bin/env bash
# rr remote job runner (backend-agnostic).
#
# Runs one experiment command and records execution-regime provenance into the
# run's meta.json.  Used by both the process backend (res) and the Slurm
# backend (res-2).  All state is written to the filesystem, so the run is
# recoverable after the launching SSH session (or rr itself) goes away.
#
# Environment (set by rr):
#   RR_RUN_DIR          absolute run directory
#   RR_META_B64         base64 of the initial meta.json
#   RR_CMD_B64          base64 of the command string (without the run prefix)
#   RR_META_PY          path to rr_meta.py on the remote
#   RR_PYTHON           venv python for provenance probing (optional)
#   RR_RUN_PREFIX       e.g. "uv run --no-sync"; prepended to the command
#   RR_ENV_B64          base64 of `export K=V` lines (optional)
#   RR_LOGIN_SETUP_B64  base64 of a shell snippet to source (optional)
set -uo pipefail

decode() { printf '%s' "$1" | base64 -d 2>/dev/null; }

mkdir -p "$RR_RUN_DIR"
meta="$RR_RUN_DIR/meta.json"
stdout_log="$RR_RUN_DIR/stdout.log"
stderr_log="$RR_RUN_DIR/stderr.log"

# Write the initial metadata only if the backend did not already write a richer
# copy (the Slurm backend records the job id before the job starts).
if [ ! -f "$meta" ]; then
  decode "$RR_META_B64" > "$meta"
fi

if [[ -n "${RR_LOGIN_SETUP_B64:-}" ]]; then
  # shellcheck disable=SC1090
  eval "$(decode "$RR_LOGIN_SETUP_B64")" >/dev/null 2>&1 || true
fi

if [[ -n "${RR_ENV_B64:-}" ]]; then
  eval "$(decode "$RR_ENV_B64")"
fi
export PYTHONUNBUFFERED=1

if [[ -n "${RR_WORKDIR:-}" ]]; then
  cd "$RR_WORKDIR" || { echo "[rr] cannot cd to $RR_WORKDIR" >>"$stderr_log"; exit 66; }
fi

# ---- provenance: start -----------------------------------------------------
probe_python="$RR_PYTHON"
if [[ ! -x "$probe_python" ]]; then
  probe_python="$(command -v python3 || true)"
fi
min_driver_arg=()
if [[ -n "${RR_MIN_DRIVER:-}" ]]; then
  min_driver_arg=("--min-driver=$RR_MIN_DRIVER")
fi
if [[ -n "$probe_python" && -f "$RR_META_PY" ]]; then
  "$probe_python" "$RR_META_PY" start "$meta" "${min_driver_arg[@]}" >/dev/null 2>&1
  probe_rc=$?
  if [[ "$probe_rc" == "78" ]]; then
    printf '[rr] fatal: allocated node driver is older than the pool requires (%s)\n' "$RR_MIN_DRIVER" >>"$stderr_log"
    exit 78
  fi
fi

cmd="$(decode "$RR_CMD_B64")"

# ---- run the experiment ----------------------------------------------------
set +e
eval "$RR_RUN_PREFIX $cmd" >"$stdout_log" 2>"$stderr_log"
rc=$?
set -e

# ---- provenance: finish ----------------------------------------------------
if [[ -n "$probe_python" && -f "$RR_META_PY" ]]; then
  "$probe_python" "$RR_META_PY" finish "$meta" "$rc" >/dev/null 2>&1 || true
fi

printf '\n[rr] command exited with code %s at %s\n' "$rc" "$(date -Is)" >>"$stdout_log"
exit "$rc"
