#!/usr/bin/env bash
# Entry point for the sq-pilot-small-v1 round through the shared control plane.
#
#   bash tracks/structural-query-learning/run_remote.sh --purpose "..." \
#       --set stage=dataset_worker --set dataset=ogbg-molesol
#
# The environment variables are the documented opt-in mechanism
# (ksvd_research.runners._extra_runners): they put this track's code on
# sys.path, register sq_runner and write run records under this track.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RESEARCH_TRACK_ROOT=tracks/structural-query-learning
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/structural-query-learning/code
export RESEARCH_EXTRA_RUNNERS=sq_runner
exec uv run research run sq-pilot-small-v1 --study sq-pilot-small-v1 --mode screen "$@"
