#!/usr/bin/env bash
# Entry point for round css-pcba-s0-v1 (conditional structure statistics, PCBA,
# seed 0) through the shared control plane.
#
#   bash tracks/structural-sufficiency/conditional_structure_statistics_pcba_s0_v1/run_remote.sh \
#       --purpose "freeze identities" --set stage=prepare
#   bash .../run_remote.sh --set stage=preflight --set device=cuda:0
#   bash .../run_remote.sh --set stage=train --set arm=E --set device=cuda:0
#   bash .../run_remote.sh --set stage=train --set arm=L --set device=cuda:0
#   bash .../run_remote.sh --set stage=train --set arm=S --set device=cuda:0
#   bash .../run_remote.sh --set stage=analyze
#
# RESEARCH_TRACK_ROOT / RESEARCH_EXTRA_RUNNER_PATHS / RESEARCH_EXTRA_RUNNERS are
# the documented opt-in mechanism (ksvd_research.runners._extra_runners): they put
# this round's code and its read-only predecessor modules on sys.path, register
# css_runner and write run records under this track.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RESEARCH_TRACK_ROOT=tracks/structural-sufficiency/conditional_structure_statistics_pcba_s0_v1
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/structural-sufficiency/conditional_structure_statistics_pcba_s0_v1/code:tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code:tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1/code:tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1/code:tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/code:tracks/structural-sufficiency/lcrc_pcba_cpc_v1/code:tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1/code:tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code:tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/code:tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code
export RESEARCH_EXTRA_RUNNERS=css_runner
exec uv run research run css-pcba-s0-v1 \
    --study css-pcba-s0-v1 --mode screen "$@"
