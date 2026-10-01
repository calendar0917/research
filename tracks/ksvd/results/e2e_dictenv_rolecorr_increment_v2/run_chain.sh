#!/usr/bin/env bash
# Reproducible command chain for round e2e_dictenv_rolecorr_increment_v2
# (route 1: width-49 appended-block increment screen).
#
# Prerequisites (frozen, SHA-verified by the runner):
#   - frozen SDB K32/s8 dictionary + frozen CSSD-q1 common subspace;
#   - RoleCorr-v1 correspondence cache / scaler / D_S / D_C (read-only reuse);
#   - official ZINC train/valid under data/ZINC.  The official test split is
#     never instantiated (mode=screen -> test_access blocked).
#
# Environment: LOCAL CPU regime, torch_threads=8.  This is pre-registration
# amendment 1 (2026-10-01): the remote A100 host could not initialise CUDA
# (GPU0 wedged, cuInit(0)=999), so the user explicitly authorised the local
# CPU regime before the first formal run; arms, gates, probes, seeds and stop
# rules are unchanged.  Wall clock on the authoring machine (16 vCPU):
#   cache/verify/pca16/correctness ~60 s, smoke ~90 s,
#   train A 1832 s + B 2165 s + C 2143 s, interventions + analysis ~150 s,
#   control-plane total 6217 s (~104 min).
#
# Executed command (revision 8244602, run id 20261001-110113-4ed73ccc):
#   uv run research run zinc_e2e_dictenv_rolecorr_increment_v2 \
#     --study zinc-context-gap --mode screen \
#     --purpose "Increment-v2 route1 primary screen (user-authorized local CPU regime, prereg amendment 1)" \
#     --set runtime.device=cpu
# (add --force to re-execute; a rerun must not be used to replace the frozen
#  round, and the pre-registration forbids any rescue of its result)
#
# The runner executes, in order:
#   cache -> verify -> pca16 -> correctness -> smoke
#         -> train (EXTRA-STRUCT / CORR-ADD / CORR-PCA-ADD; seed 0; 320 epochs
#            each; Top-5 soup; resumable checkpoints)
#         -> interventions (block zero / block shuffle x5 / object shuffle x5 /
#            base zero / paired per-molecule / code usage)
#         -> analysis (summary.json, REPORT.md, DECISION.md)
#
# Route 2 (joint 709-D environment dictionary) is conditional and its training
# stages are deliberately NOT implemented: the route-1 gate did not fire, and
# entering route 2 requires a fresh explicit authorisation and budget.  Only
# its preparation stages exist:
#   uv run research run zinc_e2e_dictenv_rolecorr_increment_v2 \
#     --study zinc-context-gap --mode scratch --set model.stage=route2
# (never run for this commit)
#
# Focused CPU unit tests (13 checks: geometry/slices, base and appended-block
# bit-identity with the frozen CSSD / RoleCorr paths, binding extension layout,
# shared init, purity, block shuffle, joint scaler/PCA, device resolver, test
# blocker):
#   uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_rolecorr_increment_v2.py
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run research run zinc_e2e_dictenv_rolecorr_increment_v2 \
  --study zinc-context-gap --mode screen \
  --purpose "Increment-v2 route1 primary screen (user-authorized local CPU regime, prereg amendment 1)" \
  --set runtime.device=cpu
