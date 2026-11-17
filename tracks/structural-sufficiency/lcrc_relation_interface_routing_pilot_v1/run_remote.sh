#!/usr/bin/env bash
# Entry point for the lcrc-relation-interface-routing-pilot-v1 round through the
# shared control plane.
#
#   bash tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1/run_remote.sh \
#       --purpose "..." --set stage=prepare
#   bash .../run_remote.sh --set stage=checks
#   bash .../run_remote.sh --set stage=train --set task=lipo --set model=R1 --set device=cuda:0
#   bash .../run_remote.sh --set stage=aggregate
#
# The environment variables are the documented opt-in mechanism
# (ksvd_research.runners._extra_runners): they put this round's code on
# sys.path, register rr_runner and write run records under this track.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RESEARCH_TRACK_ROOT=tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1/code:tracks/structural-query-learning/code:tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/code:tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code:tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code
export RESEARCH_EXTRA_RUNNERS=rr_runner
exec uv run research run lcrc-relation-interface-routing-pilot-v1 \
    --study lcrc-relation-interface-routing-pilot-v1 --mode screen "$@"
