#!/usr/bin/env bash
# Entry point for lcrc-auto-subgraph-dose-localization-audit-v1 through the
# shared control plane.  CPU-first: no GPU is required for this audit.
#
#   bash tracks/structural-sufficiency/lcrc_auto_subgraph_dose_localization_audit_v1/run_local.sh \
#       --purpose "preflight: identity + full-path smoke" --set stage=preflight
#   bash .../run_local.sh --purpose "audit molesol" --set stage=audit --set dataset=ogbg-molesol
#   bash .../run_local.sh --purpose "audit mollipo" --set stage=audit --set dataset=ogbg-mollipo
#   bash .../run_local.sh --purpose "aggregate" --set stage=aggregate
#
# The environment variables are the documented opt-in mechanism
# (ksvd_research.runners._extra_runners): they put this round's code (and its
# read-only upstream code) on sys.path, register au_runner, and write run records
# under the structural-sufficiency track.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RESEARCH_TRACK_ROOT=tracks/structural-sufficiency/lcrc_auto_subgraph_dose_localization_audit_v1
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/structural-sufficiency/lcrc_auto_subgraph_dose_localization_audit_v1/code:tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/code:tracks/structural-query-learning/code:tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code:tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code:tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/code
export RESEARCH_EXTRA_RUNNERS=au_runner
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export AU_TORCH_THREADS="${AU_TORCH_THREADS:-4}"
exec uv run research run lcrc-auto-subgraph-dose-localization-audit-v1 \
    --study lcrc-auto-subgraph-dose-localization-audit-v1 --mode screen "$@"
