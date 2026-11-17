#!/usr/bin/env bash
# Official entry point for gnn-route-clarification-8h-v1.
#
# Local (worktree) examples
#   bash .../run_remote.sh prepare
#   bash .../run_remote.sh plan   res-gpu0
#   bash .../run_remote.sh smoke
#   bash .../run_remote.sh worker res-gpu0 2026-11-17T00:20:00Z
#   bash .../run_remote.sh aggregate
#
# Remote (through rr, one persistent worker per physical card)
#   rr run res gnn-route-gpu0 --pool res-gpu0 \
#       --result tracks/structural-sufficiency/gnn_route_clarification_8h_v1 \
#       -- bash tracks/structural-sufficiency/gnn_route_clarification_8h_v1/run_remote.sh \
#              worker res-gpu0 2026-11-17T00:20:00Z
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
if [[ ! -f "$ROOT/pyproject.toml" || ! -f "$HERE/code/gr_common.py" ]]; then
  echo "run_remote.sh: repo root resolution failed (ROOT=$ROOT)" >&2
  exit 66
fi

# rr prepends `uv run --no-sync`; locally resolve the shared venv explicitly.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  elif [[ -x "$(dirname "$ROOT")/research/.venv/bin/python" ]]; then
    PYTHON_BIN="$(dirname "$ROOT")/research/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
CODE="$HERE/code"

cmd="${1:-help}"
shift || true
case "$cmd" in
  prepare)   exec "$PYTHON_BIN" "$CODE/gr_prepare.py" "$@" ;;
  plan)      exec "$PYTHON_BIN" "$CODE/gr_worker.py" --plan --pool "${1:-res-gpu0}" ;;
  smoke)     exec "$PYTHON_BIN" "$CODE/gr_worker.py" --smoke ;;
  worker)    exec "$PYTHON_BIN" "$CODE/gr_worker.py" --pool "${1:-res-gpu0}" --deadline "${2:-2026-11-17T00:20:00Z}" ;;
  pairs)     exec "$PYTHON_BIN" "$CODE/gr_pairs.py" "$@" ;;
  ledgers)   exec "$PYTHON_BIN" "$CODE/gr_ledgers.py" "$@" ;;
  aggregate) exec "$PYTHON_BIN" "$CODE/gr_analysis.py" "$@" ;;
  report)    exec "$PYTHON_BIN" "$CODE/gr_report.py" "$@" ;;
  *)
    echo "usage: run_remote.sh {prepare|plan|smoke|worker|pairs|ledgers|aggregate|report} ..." >&2
    exit 64
    ;;
esac
