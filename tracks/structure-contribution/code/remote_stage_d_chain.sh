#!/usr/bin/env bash
# Stage-D second-split replication training: fresh Fit-<tag> masked SSL per seed
# plus the P0/P3 supervised arms.  Run only after the Stage-B Monitor recipe gate
# passed (recorded purchase rule).  Sequential and resumable.
#
# Usage: bash remote_stage_d_chain.sh [tag]   (default tag C)
set -euo pipefail

TAG="${1:-C}"
DRIVER=tracks/structure-contribution/code/run_bjoc_performance_mechanism_long_v1.py
OUT=tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1

for seed in 0 1; do
  if [ ! -f "${OUT}/pretrain_${TAG}_s${seed}.pt" ]; then
    echo "[chain] Fit-${TAG} masked SSL seed ${seed}"
    python "${DRIVER}" stage-b-ssl --tag "${TAG}" --seed "${seed}" --mode atom_bond --device cuda
  fi
done

ARMS="P0 P3" bash "$(dirname "$0")/remote_stage_b_arms_chain.sh" 0 "${TAG}"
ARMS="P0 P3" bash "$(dirname "$0")/remote_stage_b_arms_chain.sh" 1 "${TAG}"

echo "[chain] stage-d tag=${TAG} complete"
