#!/usr/bin/env bash
# Launch one formal worker.  Always invoke through `rr run` so the remote
# checkout, commit and execution regime are recorded.
#
#   ./run_remote.sh measured tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/results/measured
#   ./run_remote.sh oracle   tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/results/oracle
set -euo pipefail
ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROUND_DIR/../../.." && pwd)"
JOB="${1:?usage: run_remote.sh measured|oracle [out-dir] [device]}"
OUT="${2:-tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/results/${JOB}}"
DEVICE="${3:-cuda}"
cd "$REPO_ROOT"
exec python "$ROUND_DIR/code/xr_worker.py" "$JOB" --out-dir "$OUT" --device "$DEVICE"
