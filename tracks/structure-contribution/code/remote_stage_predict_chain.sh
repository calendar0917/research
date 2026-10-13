#!/usr/bin/env bash
# Freeze all Stage-B/C predictions (Monitor + Locked) before the single Monitor
# label read of protocol cscl-bjoc-performance-mechanism-long-v1.
#
# A1 is trained for seed 0 only, so its two entries are appended to the manifest
# by the helper (stage-b-predict otherwise aborts on the missing seed-1 soup).
#
# Usage: bash remote_stage_predict_chain.sh
set -euo pipefail

DRIVER=tracks/structure-contribution/code/run_bjoc_performance_mechanism_long_v1.py
HELPER=tracks/structure-contribution/code/add_predictions_to_manifest.py

python "${DRIVER}" stage-b-predict --tag B --arms B0 B3 P0 P3 --seeds 0 1 --device cuda
python "${HELPER}" --tag B --arm A1 --seeds 0 --drop
python "${DRIVER}" stage-c-predict --tag B --seeds 0 --device cuda
echo "[chain] prediction freeze complete"
