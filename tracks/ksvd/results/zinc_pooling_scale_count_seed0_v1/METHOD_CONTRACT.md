# METHOD_CONTRACT — `zinc_pooling_scale_count_seed0_v1`

What is actually computed, where, and what is held fixed. Frozen before the two formal runs.

## 1. Data and frames

* `fresh.load_fresh_objects` → the frozen source fresh-fold objects copied into this round's
  results directory (`fresh_fold.npz`, `fresh_targets.npz`, `fresh_tuple_payload.npz`,
  `fresh_prep.npz`, `fresh_kappa.json`, `fresh_manifest.json`), SHA256-verified against the
  source manifest. `--prepare` performs the copy and writes `frozen_input_copy.json`.
* `fresh.build_prepared_data(..., verify_prep=True)`: fit 8000 / dev 2000, fit-only body
  standardizers. `g = y − c`, source fit-only constants. No dev fit or selection.
* Official-valid/test never loaded (`official_valid_loaded=false`, `official_test_loaded=false`).

## 2. Pooling (the only intervention)

* Round mask = source C6 mask minus the unary/pair count zeroing
  (`AuditMask(global_zero_groups=("atom_histogram","bond_histogram"),
  relation_zero_groups=("path_count",))`). Global chemistry histograms and relation
  `path_count` stay masked.
* `pool_moments_scaled(E, batch, n_graphs, mode)`: `[Σz, Σz², log1p(n)]` for `sum`;
  `[Σz/max(n,1), Σz²/max(n,1), log1p(n)]` for `mean`, `n = bincount(batch)`.
* `pool_pair_moments_scaled(pair_value, pair_batch, pair_bucket, n_graphs, mode)`: same per each
  of the five distance buckets, `n` = actual pair rows in the bucket per graph; empty bucket →
  two zero moments, count 0, denominator 1.
* The `sum` functions are asserted bit-identical to `audit.pool_moments_masked(...,
  zero_blocks=())` / `audit.pool_pair_moments_masked(..., zero_blocks=())`.
* Six count input coordinates: unary `log1p(n)` at runtime index `2*E_dim` (= 288 for the
  144-wide bridge) and each pair bucket's `log1p(n)` at `bucket*block + 2*P_dim` (= 96 within
  the 97-wide block). Runtime shapes are asserted (`E` 144, `pair_value` 48, reader input 814).

## 3. Models

* `ScaledLocalTupleFull(prev.LocalTupleFull)`: construction identical to the source `M`
  constructor (`LocalTupleEncoderM`, `A_raw = D_loc_init.T`, zero `W_loc`, kappa_M from the
  frozen payload); the only added state is the plain `pool_mode` attribute ("sum" for C, "mean"
  for N). The forward body mirrors `audit.AuditModel.forward` line by line; only the two pooling
  calls are routed to the round functions. `forward_equivalence` asserts C's forward is
  bit-identical to the inherited audit forward under the round mask.
* `init_identity_check` asserts all state keys/shapes/values byte-identical to the frozen source
  `M_init_state.pt`; expected 297,499 trainable parameters for both arms.
* No warm start; no trained state is ever loaded into a training model.

## 4. Training

Exactly the frozen recipe (PROTOCOL §4). `seed_everything(0)` before construction; the build
RNG and train RNG streams are asserted equal to the frozen source hashes
(`a2e8a8ab…` / `1ccf1725…`); the schedule hash is asserted equal to `7b11a529…`; the actual
global graph-ID stream is hashed and expected `69187f13…` for both arms. Saved per arm:
`{arm}_init_state.pt`, `{arm}_epoch40_state.pt`, `{arm}_last_state.pt`,
`{arm}_raw_soup_state.pt`, `{arm}_raw_predictions.npz` (init/last/raw_soup × fit/dev),
`{arm}_meta.json`, `{arm}_curve.json`, `{arm}_probe.json`, `{arm}_diagnostics.json`,
`{arm}_gpu_runtime.json`. A CPU/GPU soup replay check must be ≤ 1e-5 on 128 fit rows.

## 5. Evaluation and classification

As PROTOCOL §5–6. `analyze` recomputes all metrics from stored predictions only (no training,
no data refit); the source `M` (and `B`, context only) predictions are read from the source
round and the source `M` soup is independently replayed with the original C6 path (≤1e-5) and
its main metrics reproduced against the source `analysis.json`. Group contributions and gains
are checked against the overall values. Bootstrap seed `20261006`;
witnesses (same → zero, swap → mirrored) are recorded. Sensitivity: drop the single worst
mean-error row per comparison.

## 6. Diagnostics (no decision use)

* Pre-reader block RMS on the fixed first-128 fit batch at epoch 0/40/240 and for the soup:
  unary first/second/count, five pair buckets' first/second/count, graph hidden, topology.
* Reader-first-layer per-input-block gradient norms at probe epochs 1/40/120/240, extracted
  from the normal training backward.
* `count_snapshot.npz`: the representative fit batch input snapshot at init (batch arrays,
  six count vectors, pooled blocks, `E`, `pair_value`), plus pre-reader block RMS; used for the
  pair-count/moment-scale table. Diagnostics consume no training RNG.

## 7. Outputs

`PROTOCOL`, `METHOD_CONTRACT`, `EVIDENCE_SCOPE`, `REPORT`, `DECISION`, `EXECUTION`,
`RESEARCH_STATE.md`, `evidence_ledger.csv`; frozen input copies + `frozen_input_copy.json`;
`pre_checks.json`, `count_snapshot.npz`, `count_witness.json`, `smoke/smoke_checks_{C,N}.json`;
per-arm states/curve/meta/probe/diagnostics/gpu_runtime/predictions; `analysis.json`,
`gains.json`, `gate.json`, `bootstrap.json`, `classification.json`, `mechanism_evidence.json`,
`replay_checks.json`, `main_table.csv`, `group_table.csv`, `group_gain_table.csv`,
`scale_table.csv`, `budget.json`, `manifest.json`, `input_manifest.json`.

No new loss/head/probe/capacity search, no kappa or coefficient scan, no post-pool
standardisation/LayerNorm/count embedding, no seed/fold/threshold search.
