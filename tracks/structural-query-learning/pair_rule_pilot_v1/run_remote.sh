#!/usr/bin/env bash
# Entry point for sq-lipo-pair-rule-v1 through the shared control plane.
#
#   bash tracks/structural-query-learning/pair_rule_pilot_v1/run_remote.sh \
#       --set stage=prepare
#
# The environment variables are the documented opt-in mechanism
# (ksvd_research.runners._extra_runners): they put this round's code on
# sys.path, register sq_pair_runner and write run records under this track.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RESEARCH_TRACK_ROOT=tracks/structural-query-learning
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/structural-query-learning/pair_rule_pilot_v1/code
export RESEARCH_EXTRA_RUNNERS=sq_pair_runner
# The three additive arms are CPU-only; cap the math libraries at four threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
exec uv run research run sq-lipo-pair-rule-v1 --study sq-lipo-pair-rule-v1 --mode screen "$@"
