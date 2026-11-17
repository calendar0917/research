#!/usr/bin/env bash
# Local (CPU) analysis of pulled checkpoints, after both workers finished.
#
#   GR_RESULTS_DIR=/path/to/merged/results bash run_analysis.sh
#
# This never trains and never touches the remote; it runs the frozen ledger,
# oracle-calibration and report pipeline against the pulled best checkpoints.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  # worktrees usually do not carry their own venv; reuse the main checkout's.
  PY="/home/calendar/code/research/.venv/bin/python"
fi
CODE="$HERE/code"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$CODE/gr_ledgers.py" --job all --dataset ogbg-mollipo
"$PY" "$CODE/gr_ledgers.py" --job oracle --dataset ogbg-mollipo
"$PY" "$CODE/gr_analysis.py" all
"$PY" "$CODE/gr_analysis.py" oracle
"$PY" "$CODE/gr_report.py"
