#!/usr/bin/env bash
# Reproduce round e2e_dictenv_rndb_v1 (Rolewise Nonlinear Dictionary Binding).
#
# CPU only; the official ZINC test split is never loaded.  Exactly one
# from-scratch RNDB seed-0 trajectory.  Results JSON/CSV/checkpoints are
# git-ignored local evidence; REPORT.md / DECISION.md / analysis_tables.md are
# tracked.
set -euo pipefail

cd "$(dirname "$0")/../../../.."   # repo root

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

# Ordered stages; `chain` runs them all.  `--force` re-runs cached stages.
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_rndb_v1 references
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_rndb_v1 preflight
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_rndb_v1 correctness
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_rndb_v1 smoke
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_rndb_v1 train      # ~1.6 h CPU
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_rndb_v1 interventions
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_rndb_v1 analysis

# Focused tests
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_rndb_v1.py
