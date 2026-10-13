#!/usr/bin/env bash
# Stage-E (atom-only SSL + A1) and Stage-C (C1) GPU training for protocol
# cscl-bjoc-performance-mechanism-long-v1.  All of these are seed 0, split B.
#
# A1 is trained unconditionally before the first Monitor label read (documented
# deviation D1 in notes/bjoc_performance_mechanism_long_v1.md): the Stage-E
# purchase rule needs the Monitor gate, but every prediction must be frozen
# before the label read.
#
# Sequential, resumable (existing checkpoints are skipped).
set -euo pipefail

DRIVER=tracks/structure-contribution/code/run_bjoc_performance_mechanism_long_v1.py
OUT=tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1

if [ ! -f "${OUT}/pretrain_B_s0_atomonly.pt" ]; then
  echo "[chain] atom-only masked SSL seed 0"
  python "${DRIVER}" stage-b-ssl --tag B --seed 0 --mode atom_only --device cuda
fi

if [ ! -f "${OUT}/soup_A1_B_s0.pt" ]; then
  echo "[chain] A1 (atom-only SSL init, R3, seed 0)"
  python "${DRIVER}" stage-b-arm --tag B --arm A1 --seed 0 --device cuda
fi

if [ ! -f "${OUT}/soup_C1_B_s0.pt" ]; then
  echo "[chain] C1 (second-moment reader, P3 init/recipe, seed 0)"
  python "${DRIVER}" stage-c-train --tag B --seed 0 --device cuda
fi

echo "[chain] stage-e/c training complete"
