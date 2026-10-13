#!/usr/bin/env bash
# Stage-B supervised arms for protocol cscl-bjoc-performance-mechanism-long-v1.
#
# Sequential, resumable chain: each arm refuses to overwrite an existing
# checkpoint, so a re-run after a partial failure skips completed arms.
# Usage: bash remote_stage_b_arms_chain.sh <seed>
set -euo pipefail

SEED="${1:?usage: remote_stage_b_arms_chain.sh <seed>}"
DRIVER=tracks/structure-contribution/code/run_bjoc_performance_mechanism_long_v1.py
OUT=tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1

for arm in B0 B3 P0 P3; do
  if [ -f "${OUT}/soup_${arm}_B_s${SEED}.pt" ]; then
    echo "[chain] skip ${arm} s${SEED}: checkpoint exists"
    continue
  fi
  echo "[chain] train ${arm} s${SEED}"
  python "${DRIVER}" stage-b-arm --tag B --arm "${arm}" --seed "${SEED}" --device cuda
done
echo "[chain] stage-b arms s${SEED} complete"
