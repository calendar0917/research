#!/usr/bin/env bash
# Local CPU profile.  The machine has a prepared Python 3.10 CPU/GPS environment;
# keep GPU-heavy methods on Kaggle and write local results to a separate registry.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${HOD_GNN_LOCAL_PY:-$HERE/.venv-gps/bin/python}"
VENDOR="${HOD_GNN_LOCAL_VENDOR:-$HERE/code/vendor}"
RESULTS="${HOD_GNN_LOCAL_RESULTS:-$HERE/results/local}"

if [[ ! -x "$PY" ]]; then
  echo "[local] missing Python environment: $PY" >&2
  exit 1
fi
if [[ ! -d "$VENDOR/GraphGPS" ]]; then
  echo "[local] missing GraphGPS checkout: $VENDOR/GraphGPS" >&2
  exit 1
fi

cd "$HERE"
export KAGGLE_WORKING_DIR="$HERE"
export HOD_GNN_VENDOR_DIR="$VENDOR"
export HOD_GNN_VENV_PY="$PY"
export HOD_GNN_RESULTS_DIR="$RESULTS"

exec "$PY" code/kaggle_run.py \
  --method gps --dataset molhiv --seeds 2 3 \
  --protocol paper-seeds-provisional --resume --no-setup --keep-work "$@"
