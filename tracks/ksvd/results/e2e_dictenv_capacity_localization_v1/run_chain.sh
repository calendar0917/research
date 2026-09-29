#!/usr/bin/env bash
# Unattended launcher for the capacity-localization round (CPU only, no GPU,
# no SSH, no remote compute).  The official ZINC test split is never loaded.
# Wave 1 = M0/F/R, wave 2 = G; the single 320-epoch from-scratch winner run is
# bought only when the frozen screening gate fires.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=""
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_capacity_localization_v1 chain --threads 4 "$@"
