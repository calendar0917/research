#!/usr/bin/env bash
# Unattended launcher for the dictionary-coder audit round (CPU-only).
# The audit never loads the official test; training only runs if gate.json fires.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CUDA_VISIBLE_DEVICES=""
exec uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_dictionary_coder_audit_v1 chain --threads 4 --concurrency 3 "$@"
