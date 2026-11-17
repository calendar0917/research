#!/usr/bin/env bash
# Stage entry point executed on the remote checkout through `rr run`.
#
#   ./run_remote.sh fields-lcrc            # res-gpu0, LCRC fields (wc + lipo)
#   ./run_remote.sh fields-gine            # res-gpu1, GINE fields (wc + lipo)
#   ./run_remote.sh pool                   # res-local CPU, shared candidate pool
#   ./run_remote.sh select-lcrc-main       # res-gpu0
#   ./run_remote.sh select-lcrc-aux        # res-gpu0
#   ./run_remote.sh select-gine-main       # res-gpu1
#   ./run_remote.sh select-gine-aux        # res-gpu1
set -euo pipefail
ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROUND_DIR/../../.." && pwd)"
CODE="$ROUND_DIR/code"
RESULTS="$ROUND_DIR/results"
STAGE="${1:?usage: run_remote.sh <stage>}"
DEVICE="${2:-cuda}"
cd "$REPO_ROOT"

case "$STAGE" in
  fields-lcrc)
    exec python "$CODE/lfa_fields.py" --family lcrc --out-dir "$RESULTS/fields" --device "$DEVICE" ;;
  fields-gine)
    exec python "$CODE/lfa_fields.py" --family gine --out-dir "$RESULTS/fields" --device "$DEVICE" ;;
  pool)
    exec python "$CODE/lfa_pool.py" --fields-dir "$RESULTS/fields" --out-dir "$RESULTS/pool" ;;
  select-lcrc-main)
    exec python "$CODE/lfa_select.py" --family lcrc --scope main --device "$DEVICE" \
      --pool "$RESULTS/pool/pool_main.jsonl" --pool-manifest "$RESULTS/pool/pool_manifest.json" \
      --out-dir "$RESULTS/select" ;;
  select-lcrc-aux)
    exec python "$CODE/lfa_select.py" --family lcrc --scope aux --device "$DEVICE" --no-fidelity \
      --pool "$RESULTS/pool/pool_aux.jsonl" --pool-manifest "$RESULTS/pool/pool_manifest.json" \
      --out-dir "$RESULTS/select" ;;
  select-gine-main)
    exec python "$CODE/lfa_select.py" --family gine --scope main --device "$DEVICE" \
      --pool "$RESULTS/pool/pool_main.jsonl" --pool-manifest "$RESULTS/pool/pool_manifest.json" \
      --out-dir "$RESULTS/select" ;;
  select-gine-aux)
    exec python "$CODE/lfa_select.py" --family gine --scope aux --device "$DEVICE" --no-fidelity \
      --pool "$RESULTS/pool/pool_aux.jsonl" --pool-manifest "$RESULTS/pool/pool_manifest.json" \
      --out-dir "$RESULTS/select" ;;
  *)
    echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
