#!/usr/bin/env bash
# Entry point for the rd-pilot-small-v1 round through the shared control plane.
#
#   bash tracks/reference-delta-learning/run_remote.sh --purpose "..." \
#       --set stage=dataset_worker --set dataset=ogbg-molesol
#
# The environment variables are the documented opt-in mechanism
# (ksvd_research.runners._extra_runners): they put this track's code on
# sys.path, register rd_runner and write run records under this track.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RESEARCH_TRACK_ROOT=tracks/reference-delta-learning
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/reference-delta-learning/code
export RESEARCH_EXTRA_RUNNERS=rd_runner
exec uv run research run rd-pilot-small-v1 --study rd-pilot-small-v1 --mode screen "$@"
