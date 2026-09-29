#!/usr/bin/env bash
# Unattended launcher for the common-subspace-separated dictionary round
# (CSSD, CPU-only).  The official ZINC test is never loaded.  Training runs
# only when the frozen zero-training selection fires and continues only if
# the epoch-40 gate passes on the live model.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES=""
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_common_subspace_dictionary_v1 chain --threads 4 "$@"
uv run python -m tracks.ksvd.experiments.luyin16.e2e_dictenv_common_subspace_dictionary_v1_extra
