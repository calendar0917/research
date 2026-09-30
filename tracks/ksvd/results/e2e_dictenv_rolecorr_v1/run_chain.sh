#!/usr/bin/env bash
# Reproducible command chain for round e2e_dictenv_rolecorr_v1 (RoleCorr).
#
# Prerequisites: the frozen CSSD-q1 common subspace at
# tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json
# and the frozen SDB dictionary; official ZINC train/val under data/ZINC.
# The official test split is never touched (mode=screen -> test_access blocked).
#
# Environment: CPU-only, torch_threads=8 (config runtime.torch_threads).
# Wall clock on the authoring machine (16 vCPU): ~62 min total
#   cache 52 s, scaler/audit ~10 s, dictionary fitting 10 min,
#   smoke 12 s, formal training 33 min + 37 min, interventions ~2 min.
#
#   uv run research run zinc_e2e_dictenv_rolecorr_v1 \
#     --study zinc-context-gap --mode screen --force \
#     --purpose "RoleCorr-v1 primary screen (stage=primary)"
#
# The runner executes, in order:
#   cache -> scaler -> audit -> dictionaries -> correctness -> smoke
#         -> train (TOPO, CORR; seed 0; 320 epochs each; Top-5 soup)
#         -> interventions (M_A0 / M_C0 / M_S0 / M_Cshuf x5 / per-molecule)
#         -> analysis (summary.json, REPORT.md, DECISION.md)
#
# The controls are gated behind the frozen 2% screening gate:
#   uv run research run zinc_e2e_dictenv_rolecorr_v1 \
#     --study zinc-context-gap --mode screen --force \
#     --set model.stage=controls
# (only authorised if the primary verdict is ROLE_CORR_SCREEN_PROCEED;
#  it was NOT fired, so this command was never run for this commit)
#
# Focused CPU unit tests (13 checks: geometry, invariance, scaler, K-SVD
# sparsity, PCA equivalence, model layout/freezing, label-free builder):
#   uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_rolecorr_v1.py
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run research run zinc_e2e_dictenv_rolecorr_v1 \
  --study zinc-context-gap --mode screen --force \
  --purpose "RoleCorr-v1 primary screen (stage=primary)"
