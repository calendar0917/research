#!/usr/bin/env bash
# Unattended launcher for the repaired capacity-localization round
# (CPU only, no GPU, no SSH, no remote compute).  The official ZINC test split
# is never loaded.  Phase A calibrates M0 at Adam(lr = 1e-4) for 20 epochs and
# stops the round if the frozen stability gate fails; Phase B is the four
# differential-LR 40-epoch arms (wave 1 = M0/F/R, wave 2 = G); Phase C is the
# single from-scratch 320-epoch winner run, bought only when a candidate passes
# the frozen S1-S4 gate.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=""
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_capacity_localization_v2 chain --threads 4 "$@"
