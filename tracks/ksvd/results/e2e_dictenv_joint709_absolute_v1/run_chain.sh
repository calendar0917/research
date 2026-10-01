#!/usr/bin/env bash
# Reproducible command chain for round e2e_dictenv_joint709_absolute_v1
# (single-candidate absolute-performance screen of the frozen K48/s12 joint
#  709-D environment dictionary; local CPU regime).
#
# Prerequisites (frozen, SHA/geometry-verified by the runner):
#   - frozen CSSD-q1 common subspace + frozen SDB K32/s8 dictionary;
#   - RoleCorr raw 536-D caches (train/valid) from `e2e_dictenv_rolecorr_v1`
#     (reused read-only; node/edge SHA pinned in the runner);
#   - frozen P1 env caches for official train 10000 / official valid 1000;
#     the official test split is NEVER instantiated (screen mode ->
#     test_access blocked).
#
# This round has exactly ONE candidate and NO control arm: no PCA48, no
# random / shuffled / no-dictionary training, no seed 1, no K/s/LR/horizon
# sweep, no dictionary fine-tuning.
#
# Stage 1 (scratch, train-only): verify frozen objects, fit the joint scaler,
# build the 709-D train/valid caches (+ row-order checks), fit the frozen
# K48/s12 K-SVD dictionary, and compute the real tied-IHT reconstruction
# diagnostic.  K-SVD on all 231,664 train rows, 10 epochs, torch_threads=8:
# 86.6 min wall on the authoring machine (16 vCPU; measured 5194.1 s).
#
# Stage 2 (screen, formal): correctness gates -> smoke (3 epochs x 2048/512)
# -> 320-epoch seed-0 training (batch 128, Adam 1e-3 / wd 1e-5 / clip 5,
# Top-5 soup) -> frozen soup probes (joint-code zero + 2 within-molecule row
# shuffles, official-train soup evaluation) -> analysis (summary.json,
# REPORT.md, DECISION.md; absolute band from the pre-registered thresholds
# 0.115 / 0.120 / 0.1233).
#
# Executed commands (from revision d287b1d6bfbfddadd7733b908f085cee3a3f73c5):
#   prepare run 20261001-131619-843f4adc (scratch), 5248.1 s, K-SVD 5194.1 s;
#   screen  run 20261001-144617-89099a3f (screen),  2680.8 s, train 2631.6 s
#   (8.224 s/epoch), correctness+smoke+probes+analysis included.
#   Result: soup valid MAE 0.132442119 -> band `stop` (JOINT709_STOP);
#   endpoint caveat: coordinate binding denormal-zero at the soup state
#   (zero/shuffle probes exactly 0.0) -> the number measures an inert
#   dictionary coordinate, see notes/e2e_dictenv_joint709_absolute_v1_analysis.md
#   # prepare (scratch)
#   uv run research run zinc_e2e_dictenv_joint709_absolute_v1 \
#     --study zinc-context-gap --mode scratch \
#     --purpose "Joint709 prepare (train-only scaler/cache/frozen K48/s12 dictionary, no test access) CPU" \
#     --set runtime.device=cpu --set model.stage=prepare
#   # formal screen
#   uv run research run zinc_e2e_dictenv_joint709_absolute_v1 \
#     --study zinc-context-gap --mode screen \
#     --purpose "Joint709 dictionary absolute-performance CPU screen" \
#     --set runtime.device=cpu
# (add --force to re-execute; a rerun must not replace the frozen round and the
#  pre-registration forbids any rescue of its result.)
#
# Focused CPU unit tests (data-free; geometry, absolute bands, tied-IHT /
# reconstruction diagnostic, route-2 coordinate path, attach offsets,
# refactored signatures, runner contract):
#   uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_joint709_absolute_v1.py
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run research run zinc_e2e_dictenv_joint709_absolute_v1 \
  --study zinc-context-gap --mode screen \
  --purpose "Joint709 dictionary absolute-performance CPU screen" \
  --set runtime.device=cpu
