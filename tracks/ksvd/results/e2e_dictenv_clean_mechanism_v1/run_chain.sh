#!/usr/bin/env bash
# Unattended launcher for the round chain (CPU-only).
cd /home/calendar/code/research
export CUDA_VISIBLE_DEVICES=""
exec uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_clean_mechanism_v1 chain --threads 4 --concurrency 3 "$@"
