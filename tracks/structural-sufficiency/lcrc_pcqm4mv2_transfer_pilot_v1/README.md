# LCRC-W2 → PCQM4Mv2 transfer pilot v1

Round id: `lcrc-pcqm4mv2-transfer-pilot-v1`

Question: can the frozen **no-iterative-message-passing** LCRC-W2 learn the
PCQM4Mv2 2D HOMO–LUMO gap regression, and does 300K → 1M training molecules
give a meaningful Validation-MAE improvement?

Two formal runs only (`PCQM_W2_300K`, `PCQM_W2_1M`), one seed, 40 epochs each,
fresh initialisation, official Train/Validation split, MAE in eV.

* `PROTOCOL.md` — the frozen protocol (read this first).
* `REPORT.md` — results, audit and conclusions.
* `code/` — `pq_common` (frozen constants/recipe), `pq_data` (subsets, sharded
  cache, norms, batch order), `pq_model` (the migrated W2 with one output),
  `pq_train` (formal trainer/evaluator), `pq_prepare` (CPU prep pipeline),
  `pq_verify` (cache verification), `pq_audit` (parsing / OOV audit),
  `pq_smoke` (the ≤10k-molecule end-to-end preflight), `pq_analysis` (report
  artifacts).
* `tests/` — CPU-only unit tests (merge equivalence with the pinned PCBA
  implementation, parameter accounting, schedule, subsets, metric).
* `analysis/` — small committed JSON artifacts (audits, norms, smoke, curves
  summary).
* `cache/`, `runs/` — git-ignored: the memory-mapped feature caches, run
  directories and checkpoints. Their real paths and SHA256 are recorded in
  `analysis/pcqm_cache_manifest.json` and each `runs/*/run.json`.

## Reproduce

```bash
# 0) once, CPU only: official dataset audit, subsets, cache, norms
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step data
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step estimate --workers 12
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step build --workers 12
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step norms
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_prepare.py --step manifest

# 0b) cache verification + parsing/OOV audit
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_verify.py --tag pcqm_train1m --official
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_verify.py --tag pcqm_valid --official
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_audit.py --tag pcqm_train1m

# 1) GPU preflight (<= 10k molecules), through rr
rr run res pqcm-smoke --pool res-gpu1 -- python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_smoke.py

# 2) the two formal runs
rr run res pcqm-w2-300k --pool res-gpu1 -- \
  python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_train.py \
  --run PCQM_W2_300K --out-dir tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/runs/PCQM_W2_300K \
  --soft-seconds 12600 --hard-seconds 14400
# (same for PCQM_W2_1M)

# 3) analysis + report
uv run python tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/code/pq_analysis.py
```

Nothing here reads a hidden Test label, and no run is started from another
run's checkpoint.
