#!/usr/bin/env bash
# After the current GPS/ZINC seed-2 continuation reaches a terminal Kaggle
# state, submit seed 3, then queue one GraphViT unit.  The token is read only
# from the inherited environment.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

while true; do
  status="$(./code/kaggle_ctl.sh status 2>&1 || true)"
  if [[ "$status" == *"COMPLETE"* ]]; then
    break
  fi
  if [[ "$status" == *"ERROR"* || "$status" == *"CANCELED"* || "$status" == *"CANCELLED"* ]]; then
    echo "[queue] current Kaggle run ended abnormally; not submitting seed 3" >&2
    exit 2
  fi
  sleep 60
done

./code/kaggle_ctl.sh set \
  --method gps --dataset zinc --seeds 3 \
  --protocol paper-seeds-provisional --resume \
  --unit-timeout-seconds 21600

while true; do
  status="$(./code/kaggle_ctl.sh status 2>&1 || true)"
  if [[ "$status" == *"COMPLETE"* ]]; then
    break
  fi
  if [[ "$status" == *"ERROR"* || "$status" == *"CANCELED"* || "$status" == *"CANCELLED"* ]]; then
    echo "[queue] GPS/ZINC seed 3 ended abnormally; not submitting GraphViT" >&2
    exit 2
  fi
  sleep 60
done

exec ./code/kaggle_ctl.sh set \
  --method graphvit --dataset molhiv --seeds 2 \
  --protocol paper-seeds-provisional --resume \
  --unit-timeout-seconds 21600
