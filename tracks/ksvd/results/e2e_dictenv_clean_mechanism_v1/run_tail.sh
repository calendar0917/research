#!/usr/bin/env bash
# Gate F refresh + Stage B probes + report on the fixed revision.
cd /home/calendar/code/research
export CUDA_VISIBLE_DEVICES=""
set -x
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_clean_mechanism_v1 gate-f
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_clean_mechanism_v1 stage-b --threads 4
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_clean_mechanism_v1 report
