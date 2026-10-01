#!/usr/bin/env bash
# E2E-DictEnv-Hier-Relation-v1 — exact reproduction chain (local CPU, 8 threads).
# Revision: cc4cf83dbce3c40977c36ff5125a47c73afd525c (tracked tree unchanged).
# Expected: prepare ~31 s; screen ~1328 s (320 epochs at ~4.13 s/epoch).
# Verdict: HIERREL_STOP, soup official-valid MAE 0.182343094.
set -euo pipefail

uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_hier_relation_v1.py

uv run research run zinc_e2e_dictenv_hier_relation_v1 \
  --study zinc-context-gap --mode scratch \
  --purpose "Hierarchical static relation dictionary train-only preparation, CPU" \
  --set runtime.device=cpu --set model.stage=prepare

uv run research run zinc_e2e_dictenv_hier_relation_v1 \
  --study zinc-context-gap --mode screen \
  --purpose "Single-model hierarchical relation dictionary absolute-performance screen, CPU" \
  --set runtime.device=cpu --set model.stage=screen

# Expected printed verdict: HIERREL_STOP (band: stop, > 0.1233).
# Official valid 1000 molecules is selection-only; the official test split is
# never instantiated by this runner (test_access blocked).
