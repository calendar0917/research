#!/usr/bin/env bash
# Entry point for lcrc-auto-subgraph-faithfulness-pilot-v1 through the shared
# control plane.
#
#   bash tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/run_remote.sh \
#       --purpose "..." --set stage=dataset_worker --set dataset=ogbg-molesol
#
# The environment variables are the documented opt-in mechanism
# (ksvd_research.runners._extra_runners): they put this round's code on sys.path,
# register as_runner and write run records under this track.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RESEARCH_TRACK_ROOT=tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/code:tracks/structural-query-learning/code:tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code:tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code:tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/code
export RESEARCH_EXTRA_RUNNERS=as_runner
exec uv run research run lcrc-auto-subgraph-faithfulness-pilot-v1 --study lcrc-auto-subgraph-faithfulness-pilot-v1 --mode screen "$@"
