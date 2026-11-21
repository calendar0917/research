#!/usr/bin/env bash
# Launch one word-port training run.  Usage: wp_launch.sh <arm> <seed> [extra args...]
# Records nothing itself: the run writes results/<run_id>/run.json with the
# launch commit, spec, scaler, curves and checkpoints.
set -euo pipefail

ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROUND_DIR/../.." && pwd)"

ARM="${1:?arm required}"
SEED="${2:?seed required}"
shift 2

DATASET="${WP_DATASET:-lipo}"
DEVICE="${WP_DEVICE:-cpu}"
THREADS="${WP_THREADS:-7}"
EPOCHS="${WP_EPOCHS:-240}"
DEADLINE="${WP_DEADLINE:-}"
case "$DATASET" in
  lipo|ogbg-mollipo) SHORT=mollipo ;;
  esol|ogbg-molesol) SHORT=molesol ;;
  *) SHORT="$DATASET" ;;
esac
LOG_DIR="$ROUND_DIR/results/$(printf "%s" "${WP_RUNID:-${SHORT}_${ARM}_s${SEED}}")"
mkdir -p "$LOG_DIR"

export WP_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

EXTRA=()
if [[ -n "$DEADLINE" ]]; then EXTRA+=(--deadline "$DEADLINE"); fi
if [[ -n "${WP_RUNID:-}" ]]; then EXTRA+=(--run-id "$WP_RUNID"); fi
if [[ -n "${WP_RECIPE:-}" ]]; then EXTRA+=(--recipe "$WP_RECIPE"); fi

cd "$ROUND_DIR/code"
exec uv run --project "$REPO_ROOT" python wp_run.py train \
  --dataset "$DATASET" --arm "$ARM" --seed "$SEED" --device "$DEVICE" \
  --epochs "$EPOCHS" "${EXTRA[@]}" "$@" >"$LOG_DIR/train.log" 2>&1
