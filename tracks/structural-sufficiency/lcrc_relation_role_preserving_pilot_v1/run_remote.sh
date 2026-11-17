#!/usr/bin/env bash
# Entry point for the lcrc-relation-role-preserving-pilot-v1 round through the
# shared control plane.
#
#   bash tracks/structural-sufficiency/lcrc_relation_role_preserving_pilot_v1/run_remote.sh \
#       --purpose "..." --set stage=prepare
#   bash .../run_remote.sh --set stage=checks
#   bash .../run_remote.sh --set stage=train --set task=lipo    --set model=RR_REAL
#   bash .../run_remote.sh --set stage=train --set task=wc_logp --set model=RR_REAL
#   bash .../run_remote.sh --set stage=train --set task=lipo    --set model=RR_CONTROL
#   bash .../run_remote.sh --set stage=train --set task=wc_logp --set model=RR_CONTROL
#   bash .../run_remote.sh --set stage=aggregate
#
# The environment variables are the documented opt-in mechanism
# (ksvd_research.runners._extra_runners): they put this round's code and its
# read-only predecessor modules on sys.path, register rp_runner and write run
# records under this track.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RESEARCH_TRACK_ROOT=tracks/structural-sufficiency/lcrc_relation_role_preserving_pilot_v1
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/structural-sufficiency/lcrc_relation_role_preserving_pilot_v1/code:tracks/structural-query-learning/code:tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/code:tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code:tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code:tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1/code
export RESEARCH_EXTRA_RUNNERS=rp_runner
exec uv run research run lcrc-relation-role-preserving-pilot-v1 \
    --study lcrc-relation-role-preserving-pilot-v1 --mode screen "$@"
